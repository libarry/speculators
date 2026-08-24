import copy
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
)
from typing_extensions import Unpack

from speculators.models.eagle3.rotary_partial import apply_neox_rotary

if TYPE_CHECKING:
    from collections.abc import Callable


def _rope_params_attr(config: Any) -> tuple[str | None, dict]:
    for attr in ("rope_parameters", "rope_scaling"):
        value = getattr(config, attr, None)
        if value is None:
            continue
        if isinstance(value, dict):
            return attr, value
        if hasattr(value, "to_dict"):
            return attr, dict(value.to_dict())
        try:
            return attr, dict(value)
        except (TypeError, ValueError):
            continue
    return None, {}


def resolve_partial_rotary_factor(config: Any) -> float:
    """Read ``partial_rotary_factor`` from rope params, defaulting to full-head."""
    _, params = _rope_params_attr(config)
    if "partial_rotary_factor" in params:
        return float(params["partial_rotary_factor"])
    return float(getattr(config, "partial_rotary_factor", 1.0) or 1.0)


def resolve_head_dim(config: Any) -> int:
    return int(
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )


def build_gqa_rotary_embedding(config: Any) -> Qwen3RotaryEmbedding:
    """Build Qwen3 RoPE tables whose last dim matches vLLM ``rotary_dim``.

    HF ``Qwen3RotaryEmbedding`` ignores ``partial_rotary_factor`` on the
    default rope path and always emits full ``head_dim`` cos/sin. When the
    verifier uses partial rotary (e.g. Qwen3.5 ``0.25``), shrink the rotary
    module's ``head_dim`` so training rotates the same leading channels as
    vLLM inference. YaRN already scales by ``partial_rotary_factor``; the
    copy sets that factor to 1.0 so it is not applied twice.
    """
    head_dim = resolve_head_dim(config)
    partial = resolve_partial_rotary_factor(config)
    if not 0.0 < partial < 1.0:
        return Qwen3RotaryEmbedding(config)

    rotary_dim = int(head_dim * partial)
    if rotary_dim <= 0 or rotary_dim > head_dim:
        return Qwen3RotaryEmbedding(config)

    rotary_config = copy.deepcopy(config)
    rotary_config.head_dim = rotary_dim
    attr, params = _rope_params_attr(rotary_config)
    if attr is not None:
        params = dict(params)
        params["partial_rotary_factor"] = 1.0
        setattr(rotary_config, attr, params)
    return Qwen3RotaryEmbedding(rotary_config)


def apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids=None,  # noqa: ARG001
    unsqueeze_dim=1,
):
    """Apply rotary embeddings, including vLLM-style partial NeoX RoPE.

    Query tokens sit at the tail of the concatenated context+mask sequence, so
    Q is rotated with the last ``q_len`` position tables. K uses the full
    tables. When ``cos`` is shorter than ``head_dim``, only the leading
    ``rotary_dim`` channels are rotated.
    """
    q_len = q.size(-2)
    q_embed = apply_neox_rotary(
        q, cos[..., -q_len:, :], sin[..., -q_len:, :], unsqueeze_dim=unsqueeze_dim
    )
    k_embed = apply_neox_rotary(k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Implements the custom attention which injects the target models
    # hidden states into the kv cache.
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,  # type: ignore[operator]
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads  # type: ignore[operator]
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            config.num_attention_heads * self.head_dim,  # type: ignore[operator]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.k_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            config.num_key_value_heads * self.head_dim,  # type: ignore[operator]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.v_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            config.num_key_value_heads * self.head_dim,  # type: ignore[operator]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,  # type: ignore[operator]
            config.hidden_size,  # type: ignore[arg-type]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # type: ignore[arg-type]
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # type: ignore[arg-type]
        self.sliding_window = (
            config.sliding_window
            if hasattr(config, "layer_types")
            and config.layer_types is not None
            and config.layer_types[layer_idx] == "sliding_attention"  # type: ignore[index]
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Instead of computing the k and v matricies from the hidden states,
        # the target_hidden is injected into the kv cache, (shape is context
        # length + block size)
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        # This is the main difference from the usual attention mechanism.
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        # note the length becomes context length + block size
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if (
            self.config._attn_implementation is not None  # noqa: SLF001
            and self.config._attn_implementation != "eager"  # noqa: SLF001
        ):
            attn_fn = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation  # noqa: SLF001
            ]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # type: ignore[arg-type]
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,  # type: ignore[arg-type]
        )

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        # necessary, but kept here for BC
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        # The main difference between this method and the qwen 3 layer it is
        # built from is that it
        # passes the extra hidden states to the self attention from the verifier model.
        # Note that target_hidden is not modified here.
        assert hidden_states is not None  # noqa: S101
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states  # type: ignore[operator]
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states  # type: ignore[operator,return-value]
