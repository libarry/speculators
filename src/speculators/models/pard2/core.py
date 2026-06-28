"""PARD-2 draft model with target-aligned feature injection."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import SpeculatorModel
from speculators.models.base_components import model_classes
from speculators.models.pard2.config import Pard2SpeculatorConfig
from speculators.models.pard2.data import build_pard2_prev_prob_batch, compute_teacher_gold_prob
from speculators.models.pard2.loss_ops import (
    DEFAULT_SEQ_CHUNK_SIZE,
    DEFAULT_VOCAB_CHUNK_SIZE,
    apply_rms_norm,
    gather_label_log_prob_chunked,
    gather_log_prob_chunked,
    token_kd_sum_chunked,
)
from speculators.models.utils import resolve_target_layer_ids
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.noise_transforms import AddUniformNoise
from speculators.utils.loading import load_model_layers

__all__ = [
    "Pard2DraftModel",
    "load_verifier_lm_head",
    "weighted_ce_kd_loss_chunked",
    "weighted_ce_kd_loss_from_hidden_chunked",
    "weighted_ce_loss_chunked",
    "weighted_kd_loss_chunked",
]

# Sequence-chunk size for the hidden-state fused loss. Larger than the
# logits-based path (``DEFAULT_SEQ_CHUNK_SIZE``) because each chunk is wrapped in
# activation checkpointing, so the per-chunk logits are transient and recomputed
# in the backward pass instead of being held resident.
DEFAULT_HIDDEN_SEQ_CHUNK_SIZE = 256


def _shift_parallel_step_ids(
    labels: torch.Tensor,
    para_num: int,
    warp_single_length: torch.Tensor,
    warp_indices: torch.Tensor | None,
    warp_indices_len: torch.Tensor | None,
) -> torch.Tensor | None:
    """Map each label position to its PARD-2 parallel block index in ``[0, para_num)``."""
    if para_num <= 0 or labels.shape[1] <= 1:
        return None

    device = labels.device
    batch_size, seq_len = labels.shape
    step_ids = torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)

    for batch_idx in range(batch_size):
        sample_single_len = int(warp_single_length[batch_idx].item())
        if sample_single_len <= 0:
            continue

        total_warp_len = sample_single_len * para_num
        if total_warp_len <= 0:
            continue

        base_steps = (
            torch.arange(total_warp_len, device=device, dtype=torch.long)
            // sample_single_len
        )
        if warp_indices is not None and warp_indices_len is not None:
            sample_idx_len = int(warp_indices_len[batch_idx].item())
            sample_indices = warp_indices[batch_idx, :sample_idx_len].to(
                device=device,
                dtype=torch.long,
            )
            sample_indices = sample_indices.clamp(min=0, max=total_warp_len - 1)
            sample_steps = torch.roll(
                torch.roll(base_steps.unsqueeze(0), shifts=-1, dims=1)[:, sample_indices],
                shifts=1,
                dims=1,
            ).squeeze(0)
        else:
            sample_steps = base_steps

        copy_len = min(int(sample_steps.numel()), seq_len)
        if copy_len > 0:
            step_ids[batch_idx, :copy_len] = sample_steps[:copy_len]

    return step_ids[:, 1:]


def _raw_warp_index(single_length: int, anchor: int, depth: int) -> int:
    """Pre-COD warped draft position for ``(anchor, depth)`` in parallel warp."""
    return depth * single_length + (anchor + depth)


def _build_raw_to_filtered_map(
    single_length: int,
    para_num: int,
    warp_indices: torch.Tensor | None,
    warp_indices_len: int,
) -> dict[int, int]:
    """Map pre-COD warped indices to filtered sequence positions."""
    total_warp_len = single_length * para_num
    if warp_indices is None or warp_indices_len <= 0:
        return {raw: raw for raw in range(total_warp_len)}

    raw_to_filtered: dict[int, int] = {}
    for filt_pos in range(warp_indices_len):
        raw = int(warp_indices[filt_pos].item())
        if 0 <= raw < total_warp_len:
            raw_to_filtered[raw] = filt_pos
    return raw_to_filtered


def _greedy_draft_target_match(
    student_logits: torch.Tensor,
    teacher_hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> torch.Tensor:
    """Greedy verify match: ``draft_argmax == target_argmax`` (PARD infer parity)."""
    shift_logits = student_logits[..., :-1, :]
    shift_teacher = teacher_hidden[..., :-1, :]
    draft_tokens = shift_logits.argmax(dim=-1)
    teacher_hidden_normed = (
        apply_rms_norm(shift_teacher, norm_weight, eps=norm_eps)
        if norm_weight is not None
        else shift_teacher
    )
    target_tokens = F.linear(
        teacher_hidden_normed,
        lm_head_weight,
        lm_head_bias,
    ).argmax(dim=-1)
    return draft_tokens == target_tokens


def compute_draft_acceptance_prob(
    student_logits: torch.Tensor,
    teacher_hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    *,
    vocab_chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE,
) -> torch.Tensor:
    """Per-position draft acceptance probability under target verification.

    For draft token ``x`` at each position, returns
    ``min(1, P_target(x) / P_draft(x))``, matching standard speculative-decoding
    rejection-sampling acceptance.
    """
    shift_logits = student_logits[..., :-1, :]
    shift_teacher = teacher_hidden[..., :-1, :]
    draft_tokens = shift_logits.argmax(dim=-1)

    log_p_draft = gather_log_prob_chunked(
        shift_logits,
        draft_tokens,
        vocab_chunk_size=vocab_chunk_size,
    )
    teacher_hidden_normed = (
        apply_rms_norm(shift_teacher, norm_weight, eps=norm_eps)
        if norm_weight is not None
        else shift_teacher
    )
    log_p_teacher = gather_label_log_prob_chunked(
        teacher_hidden_normed,
        draft_tokens,
        lm_head_weight,
        lm_head_bias,
        vocab_chunk_size=vocab_chunk_size,
    )
    ratio = (log_p_teacher - log_p_draft).exp()
    return torch.clamp(ratio, max=1.0)


def compute_step_acceptance_metrics(
    student_logits: torch.Tensor,
    teacher_hidden: torch.Tensor,
    labels: torch.Tensor,
    para_num: int,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    warp_indices: torch.Tensor | None = None,
    warp_indices_len: torch.Tensor | None = None,
    warp_single_length: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Conditional greedy acceptance ``acc_{d}`` aligned with PARD ``accept_ratio``.

    For each original-sequence anchor, walk parallel depths ``0..para_num-1`` and
    count a match at depth ``d`` only if depths ``0..d-1`` already matched. This is
    the training-time analogue of PARD inference's per-depth conditional verify rate
    (``draft_argmax == target_argmax`` at each speculative depth).
    """
    if para_num <= 0 or labels.shape[1] <= 1 or warp_single_length is None:
        return {}

    shift_labels = labels[..., 1:]
    valid_mask = shift_labels != -100
    max_shift = shift_labels.shape[1]

    device = student_logits.device
    greedy_match = _greedy_draft_target_match(
        student_logits,
        teacher_hidden,
        lm_head_weight.to(device),
        lm_head_bias.to(device) if lm_head_bias is not None else None,
        norm_weight.to(device) if norm_weight is not None else None,
        norm_eps,
    )

    acceptance_sum = torch.zeros(para_num, device=device, dtype=torch.float32)
    acceptance_total = torch.zeros(para_num, device=device, dtype=torch.float32)
    batch_size = labels.shape[0]

    for batch_idx in range(batch_size):
        single_length = int(warp_single_length[batch_idx].item())
        if single_length <= 1:
            continue

        sample_warp_indices = None
        sample_warp_len = 0
        if warp_indices is not None and warp_indices_len is not None:
            sample_warp_len = int(warp_indices_len[batch_idx].item())
            if sample_warp_len > 0:
                sample_warp_indices = warp_indices[batch_idx, :sample_warp_len]

        raw_to_filtered = _build_raw_to_filtered_map(
            single_length,
            para_num,
            sample_warp_indices,
            sample_warp_len,
        )

        for anchor in range(single_length - 1):
            chain_alive = True
            for depth in range(para_num):
                if not chain_alive:
                    break

                raw_idx = _raw_warp_index(single_length, anchor, depth)
                filt_pos = raw_to_filtered.get(raw_idx)
                if filt_pos is None or filt_pos >= max_shift:
                    break
                if not bool(valid_mask[batch_idx, filt_pos].item()):
                    break

                acceptance_total[depth] = acceptance_total[depth] + 1.0
                if bool(greedy_match[batch_idx, filt_pos].item()):
                    acceptance_sum[depth] = acceptance_sum[depth] + 1.0
                else:
                    chain_alive = False

    metrics: dict[str, torch.Tensor] = {}
    for step_idx in range(para_num):
        metrics[f"acc_{step_idx}_sum"] = acceptance_sum[step_idx]
        metrics[f"acc_{step_idx}_total"] = acceptance_total[step_idx]
    return metrics


def compute_teacher_supervision_diagnostics(
    gold_input_ids: torch.Tensor,
    gold_teacher_hidden: torch.Tensor,
    gold_seq_len: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    *,
    device: torch.device,
    gold_teacher_gold_prob: torch.Tensor | None = None,
    uniform_prob: float = 1.0 / 151936,
) -> dict[str, torch.Tensor]:
    """Diagnostics for teacher gold-prob supervision (hidden health + prob scale)."""
    del lm_head_bias
    metrics: dict[str, torch.Tensor] = {
        "diag_gold_hidden_abs_mean_sum": torch.tensor(0.0, device=device),
        "diag_gold_hidden_abs_mean_total": torch.tensor(0.0, device=device),
        "diag_gold_prob_mean_sum": torch.tensor(0.0, device=device),
        "diag_gold_prob_mean_total": torch.tensor(0.0, device=device),
        "diag_gold_prob_max_sum": torch.tensor(0.0, device=device),
        "diag_gold_prob_max_total": torch.tensor(0.0, device=device),
        "diag_gold_prob_uniform_frac_sum": torch.tensor(0.0, device=device),
        "diag_gold_prob_uniform_frac_total": torch.tensor(0.0, device=device),
    }

    batch_size = gold_input_ids.shape[0]
    uniform_threshold = uniform_prob * 2.0
    for batch_idx in range(batch_size):
        seq_len = int(gold_seq_len[batch_idx].item())
        if seq_len <= 1:
            continue

        ids = gold_input_ids[batch_idx : batch_idx + 1, :seq_len].to(device)
        hidden = gold_teacher_hidden[batch_idx : batch_idx + 1, :seq_len].to(device)
        metrics["diag_gold_hidden_abs_mean_sum"] = (
            metrics["diag_gold_hidden_abs_mean_sum"] + hidden.abs().mean()
        )
        metrics["diag_gold_hidden_abs_mean_total"] = (
            metrics["diag_gold_hidden_abs_mean_total"] + 1.0
        )

        gold_prob = (
            gold_teacher_gold_prob[batch_idx : batch_idx + 1, :seq_len]
            .to(device=device, dtype=torch.float32)
            if gold_teacher_gold_prob is not None
            else compute_teacher_gold_prob(
                ids,
                hidden,
                lm_head_weight,
                None,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
            )
        )
        pos_probs = gold_prob[0, 1:seq_len]
        metrics["diag_gold_prob_mean_sum"] = (
            metrics["diag_gold_prob_mean_sum"] + pos_probs.mean()
        )
        metrics["diag_gold_prob_mean_total"] = (
            metrics["diag_gold_prob_mean_total"] + 1.0
        )
        metrics["diag_gold_prob_max_sum"] = (
            metrics["diag_gold_prob_max_sum"] + pos_probs.max()
        )
        metrics["diag_gold_prob_max_total"] = (
            metrics["diag_gold_prob_max_total"] + 1.0
        )
        near_uniform = (pos_probs < uniform_threshold).float().mean()
        metrics["diag_gold_prob_uniform_frac_sum"] = (
            metrics["diag_gold_prob_uniform_frac_sum"] + near_uniform
        )
        metrics["diag_gold_prob_uniform_frac_total"] = (
            metrics["diag_gold_prob_uniform_frac_total"] + 1.0
        )

    return metrics


def weighted_kd_loss_chunked(
    shift_student_logits: torch.Tensor,
    shift_teacher_hidden: torch.Tensor,
    lm_head: nn.Linear,
    temperature: float,
    token_weight: torch.Tensor | None = None,
    *,
    chunk_size: int = DEFAULT_SEQ_CHUNK_SIZE,
    vocab_chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE,
) -> torch.Tensor:
    """Memory-efficient KD loss over long sequences.

    Processes the sequence in small chunks and the vocabulary in sub-chunks so
    full ``log_softmax`` tensors are never materialized at once.
    """
    _, seq_len, _ = shift_student_logits.shape
    weighted_sum = shift_student_logits.new_zeros(())
    weight_sum = shift_student_logits.new_zeros(())

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        student_chunk = shift_student_logits[:, start:end]
        teacher_chunk = shift_teacher_hidden[:, start:end].to(dtype=lm_head.weight.dtype)
        teacher_logits = lm_head(teacher_chunk)

        token_kd = token_kd_sum_chunked(
            student_chunk,
            teacher_logits,
            temperature,
            vocab_chunk_size=vocab_chunk_size,
        )

        if token_weight is not None:
            chunk_weight = token_weight[:, start:end].to(token_kd.dtype)
            weighted_sum = weighted_sum + (token_kd * chunk_weight).sum()
            weight_sum = weight_sum + chunk_weight.sum()
        else:
            weighted_sum = weighted_sum + token_kd.sum()
            weight_sum = weight_sum + token_kd.numel()

        del teacher_logits, token_kd

    return weighted_sum / weight_sum.clamp_min(1e-6)


def weighted_ce_loss_chunked(
    shift_student_logits: torch.Tensor,
    shift_labels: torch.Tensor,
    token_weight: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_SEQ_CHUNK_SIZE,
) -> torch.Tensor:
    """Memory-efficient weighted CE over long PARD-2 warped sequences."""
    _, seq_len, vocab_size = shift_student_logits.shape
    weighted_sum = shift_student_logits.new_zeros(())
    weight_sum = shift_student_logits.new_zeros(())

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        student_chunk = shift_student_logits[:, start:end]
        labels_chunk = shift_labels[:, start:end]
        weight_chunk = token_weight[:, start:end].to(dtype=torch.float32)
        ce_chunk = F.cross_entropy(
            student_chunk.float().reshape(-1, vocab_size),
            labels_chunk.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(labels_chunk)
        weighted_sum = weighted_sum + (ce_chunk * weight_chunk).sum()
        weight_sum = weight_sum + weight_chunk.sum()

    return weighted_sum / weight_sum.clamp_min(1e-6)


def weighted_ce_kd_loss_chunked(
    shift_student_logits: torch.Tensor,
    shift_labels: torch.Tensor,
    shift_teacher_hidden: torch.Tensor,
    lm_head: nn.Linear,
    token_weight: torch.Tensor,
    temperature: float,
    ce_alpha: float,
    kd_alpha: float,
    *,
    chunk_size: int = DEFAULT_SEQ_CHUNK_SIZE,
    vocab_chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused CE + KD over sequence chunks to reduce peak activation memory."""
    _, seq_len, vocab_size = shift_student_logits.shape
    ce_weighted_sum = shift_student_logits.new_zeros(())
    kd_weighted_sum = shift_student_logits.new_zeros(())
    weight_sum = shift_student_logits.new_zeros(())

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        student_chunk = shift_student_logits[:, start:end]
        labels_chunk = shift_labels[:, start:end]
        weight_chunk = token_weight[:, start:end].to(dtype=torch.float32)

        ce_chunk = F.cross_entropy(
            student_chunk.float().reshape(-1, vocab_size),
            labels_chunk.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(labels_chunk)

        teacher_chunk = shift_teacher_hidden[:, start:end].to(dtype=lm_head.weight.dtype)
        teacher_logits = lm_head(teacher_chunk)
        token_kd = token_kd_sum_chunked(
            student_chunk,
            teacher_logits,
            temperature,
            vocab_chunk_size=vocab_chunk_size,
        )

        ce_weighted_sum = ce_weighted_sum + (ce_chunk * weight_chunk).sum()
        kd_weighted_sum = kd_weighted_sum + (token_kd * weight_chunk).sum()
        weight_sum = weight_sum + weight_chunk.sum()

        del teacher_logits, token_kd, ce_chunk

    denom = weight_sum.clamp_min(1e-6)
    ce_loss = ce_weighted_sum / denom
    kd_loss = kd_weighted_sum / denom
    final_loss = ce_loss * ce_alpha + kd_loss * kd_alpha
    return final_loss, ce_loss, kd_loss


def weighted_ce_kd_loss_from_hidden_chunked(
    shift_draft_hidden: torch.Tensor,
    draft_lm_weight: torch.Tensor,
    shift_labels: torch.Tensor,
    shift_teacher_hidden: torch.Tensor,
    verifier_lm_weight: torch.Tensor,
    token_weight: torch.Tensor,
    temperature: float,
    ce_alpha: float,
    kd_alpha: float,
    *,
    chunk_size: int = DEFAULT_HIDDEN_SEQ_CHUNK_SIZE,
    vocab_chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE,
    use_checkpoint: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused CE + KD computed directly from the draft hidden states.

    This is numerically equivalent to ``weighted_ce_kd_loss_chunked`` called on
    ``draft_lm_head(shift_draft_hidden)`` but never materializes the full
    ``[batch, seq, vocab]`` student-logits tensor (~2.5 GB at seq 2k / vocab
    152k in bf16). Each sequence chunk projects the hidden state to logits inside
    an activation-checkpointed closure, so those logits are transient in the
    forward pass and recomputed on demand during backward. Peak activation
    memory drops from ``O(seq * vocab)`` to ``O(chunk * vocab)``.
    """
    _, seq_len, _ = shift_draft_hidden.shape
    ce_weighted_sum = shift_draft_hidden.new_zeros((), dtype=torch.float32)
    kd_weighted_sum = shift_draft_hidden.new_zeros((), dtype=torch.float32)
    weight_sum = shift_draft_hidden.new_zeros((), dtype=torch.float32)

    def _chunk_loss(
        draft_hidden_chunk: torch.Tensor,
        teacher_hidden_chunk: torch.Tensor,
        labels_chunk: torch.Tensor,
        weight_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        student_logits = F.linear(draft_hidden_chunk, draft_lm_weight)
        vocab_size = student_logits.shape[-1]
        ce_chunk = F.cross_entropy(
            student_logits.float().reshape(-1, vocab_size),
            labels_chunk.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(labels_chunk)

        teacher_logits = F.linear(teacher_hidden_chunk, verifier_lm_weight)
        token_kd = token_kd_sum_chunked(
            student_logits,
            teacher_logits,
            temperature,
            vocab_chunk_size=vocab_chunk_size,
        )
        wf = weight_chunk.to(torch.float32)
        return (ce_chunk * wf).sum(), (token_kd * wf).sum(), wf.sum()

    checkpointing = (
        use_checkpoint
        and torch.is_grad_enabled()
        and shift_draft_hidden.requires_grad
    )

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        draft_chunk = shift_draft_hidden[:, start:end]
        teacher_chunk = shift_teacher_hidden[:, start:end].to(
            dtype=verifier_lm_weight.dtype
        )
        labels_chunk = shift_labels[:, start:end]
        weight_chunk = token_weight[:, start:end]

        if checkpointing:
            ce_s, kd_s, w_s = checkpoint(
                _chunk_loss,
                draft_chunk,
                teacher_chunk,
                labels_chunk,
                weight_chunk,
                use_reentrant=False,
            )
        else:
            ce_s, kd_s, w_s = _chunk_loss(
                draft_chunk, teacher_chunk, labels_chunk, weight_chunk
            )

        ce_weighted_sum = ce_weighted_sum + ce_s
        kd_weighted_sum = kd_weighted_sum + kd_s
        weight_sum = weight_sum + w_s

    denom = weight_sum.clamp_min(1e-6)
    ce_loss = ce_weighted_sum / denom
    kd_loss = kd_weighted_sum / denom
    final_loss = ce_loss * ce_alpha + kd_loss * kd_alpha
    return final_loss, ce_loss, kd_loss


def load_verifier_lm_head(path: str) -> nn.Linear:
    weights = load_model_layers(["lm_head.weight"], path)
    weight = weights["lm_head.weight"]
    lm_head = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    lm_head.load_state_dict({"weight": weight.detach().clone()}, strict=False)
    return lm_head


def load_verifier_final_norm(path: str) -> tuple[nn.Module, float]:
    """Load verifier ``model.norm`` for logits-aligned gold-prob / KD."""
    verifier_config = AutoConfig.from_pretrained(path)
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config
    model_type = getattr(verifier_config, "model_type", None)
    if model_type not in model_classes:
        raise ValueError(
            f"Unsupported verifier model_type={model_type!r} for PARD-2 final norm"
        )
    norm_class = model_classes[model_type].norm_class
    norm_eps = float(getattr(verifier_config, "rms_norm_eps", 1e-6))
    norm = norm_class(verifier_config.hidden_size, eps=norm_eps)
    weights = load_model_layers(["model.norm.weight"], path)
    norm.load_state_dict({"weight": weights["model.norm.weight"].detach().clone()})
    for param in norm.parameters():
        param.requires_grad_(False)
    norm._num_hidden_layers = int(  # noqa: SLF001
        getattr(verifier_config, "num_hidden_layers", 0)
    )
    return norm, norm_eps


@SpeculatorModel.register("pard2")
class Pard2DraftModel(SpeculatorModel):
    """PARD-2 target-aligned parallel draft model."""

    config_class: ClassVar[type[Pard2SpeculatorConfig]] = Pard2SpeculatorConfig  # type: ignore[misc]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc]
        "verifier_lm_head.weight",
        "verifier_lm_head.bias",
        "verifier_norm.weight",
    ]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "verifier_lm_head.weight",
        "verifier_lm_head.bias",
        "verifier_norm.weight",
    ]

    def __init__(self, config: Pard2SpeculatorConfig):
        super().__init__(config=config)
        draft_config = AutoConfig.from_pretrained(config.draft_name_or_path)
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            config.draft_name_or_path,
            torch_dtype=torch.bfloat16,
        )
        hidden_size = draft_config.hidden_size
        self.target_proj = nn.Linear(
            config.target_feat_dim,
            hidden_size,
            bias=config.proj_bias,
        )
        verifier_path = config.speculators_config.verifier.name_or_path  # type: ignore[union-attr]
        self.verifier_lm_head = load_verifier_lm_head(verifier_path)
        self.verifier_norm, self.verifier_norm_eps = load_verifier_final_norm(
            verifier_path
        )
        for param in self.verifier_lm_head.parameters():
            param.requires_grad_(False)

        # The verifier head/norm are populated with real weights via
        # ``load_state_dict`` above, which does NOT mark them as HF-initialized.
        # Without this flag, ``self.post_init()`` re-runs HF weight init over
        # them and clobbers the loaded weights (lm_head -> normal(0, 0.02),
        # norm -> ones). That silently turns the KD target, prev_prob weights,
        # and acceptance metric into garbage (loss explodes, acc ~ 0).
        self.verifier_lm_head._is_hf_initialized = True  # noqa: SLF001
        self.verifier_norm._is_hf_initialized = True  # noqa: SLF001

        self.ce_alpha = config.ce_alpha
        self.kd_alpha = config.kd_alpha
        self.kd_temperature = config.kd_temperature
        self.target_feat_mask = config.target_feat_mask
        self.prev_prob_loss = config.prev_prob_loss
        self.feat_scale = config.feat_scale

        # The PARD-2 draft was trained on the verifier's POST-final-norm last
        # hidden state for its ``-1`` target-feature block (HF
        # ``outputs.hidden_states[-1]`` is returned AFTER ``model.norm``). vLLM's
        # aux-hidden-state extraction instead exports the PRE-norm residual last
        # hidden (mid layers are pre-norm in both, so only the last block
        # differs, by ~8x in scale). Locate that block so ``forward`` can apply
        # the verifier final norm and match the official feature distribution.
        # ``verifier_last_hidden_states`` (teacher hidden) is intentionally left
        # PRE-norm because the loss/metric paths apply the norm themselves.
        target_layer_ids = list(config.target_layer_ids)
        num_target_layers = max(len(target_layer_ids), 1)
        self._verifier_hidden_size = config.target_feat_dim // num_target_layers
        verifier_num_layers = int(
            getattr(self.verifier_norm, "_num_hidden_layers", 0)
        )
        self._target_feat_last_block: int | None = None
        for idx, layer_id in enumerate(target_layer_ids):
            if layer_id == -1 or (
                verifier_num_layers and layer_id == verifier_num_layers
            ):
                self._target_feat_last_block = idx
                break

        # Whether to compute the (expensive) acceptance/diagnostic metrics during
        # ``forward``. These require an extra full-vocab verifier ``lm_head`` pass
        # plus a full-vocab pass over the student logits, purely for logging. The
        # trainer disables them on steps that are not going to be logged (see
        # ``set_metrics_enabled``) so the bulk of training only pays for the loss.
        self.compute_acceptance_metrics = True

        self.post_init()

    def set_metrics_enabled(self, enabled: bool) -> None:
        """Toggle computation of the logging-only acceptance/diagnostic metrics."""
        self.compute_acceptance_metrics = bool(enabled)

    @property
    def layers(self) -> nn.ModuleList:
        return self.draft_model.model.layers  # type: ignore[attr-defined,union-attr]

    @property
    def target_layer_ids(self) -> list[int]:
        return list(self.config.target_layer_ids)

    def gradient_checkpointing_enable(self, *args: Any, **kwargs: Any):
        if hasattr(self.draft_model, "gradient_checkpointing_enable"):
            return self.draft_model.gradient_checkpointing_enable(*args, **kwargs)
        return None

    def gradient_checkpointing_disable(self, *args: Any, **kwargs: Any):
        if hasattr(self.draft_model, "gradient_checkpointing_disable"):
            return self.draft_model.gradient_checkpointing_disable(*args, **kwargs)
        return None

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        target_feat: torch.Tensor | None = None,
        teacher_hidden: torch.Tensor | None = None,
        prev_prob: torch.Tensor | None = None,
        gold_input_ids: torch.Tensor | None = None,
        gold_teacher_hidden: torch.Tensor | None = None,
        gold_teacher_gold_prob: torch.Tensor | None = None,
        gold_labels: torch.Tensor | None = None,
        gold_seq_len: torch.Tensor | None = None,
        warp_single_length: torch.Tensor | None = None,
        warp_indices: torch.Tensor | None = None,
        warp_indices_len: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[None, torch.Tensor, dict[str, torch.Tensor]]:
        del kwargs

        if input_ids is None:
            raise ValueError("input_ids are required")

        inputs_embeds = self.draft_model.get_input_embeddings()(input_ids)
        device = inputs_embeds.device

        if (
            self.prev_prob_loss
            and teacher_hidden is not None
            and gold_input_ids is not None
            and gold_teacher_hidden is not None
            and gold_labels is not None
            and gold_seq_len is not None
            and warp_indices is not None
            and warp_indices_len is not None
        ):
            lm_head = self.verifier_lm_head
            norm_weight = self.verifier_norm.weight.to(device)
            prev_prob = build_pard2_prev_prob_batch(
                gold_input_ids,
                gold_teacher_hidden,
                gold_labels,
                gold_seq_len,
                warp_indices,
                warp_indices_len,
                lm_head.weight.to(device),
                lm_head.bias.to(device) if lm_head.bias is not None else None,
                self.config.para_num,
                input_ids.shape[1],
                device=device,
                norm_weight=norm_weight,
                norm_eps=self.verifier_norm_eps,
                gold_teacher_gold_prob=gold_teacher_gold_prob,
            )
            if self.compute_acceptance_metrics:
                diag_metrics = compute_teacher_supervision_diagnostics(
                    gold_input_ids,
                    gold_teacher_hidden,
                    gold_seq_len,
                    lm_head.weight.to(device),
                    lm_head.bias.to(device) if lm_head.bias is not None else None,
                    norm_weight,
                    self.verifier_norm_eps,
                    device=device,
                    gold_teacher_gold_prob=gold_teacher_gold_prob,
                )
            else:
                diag_metrics = {}
        else:
            diag_metrics = {}

        if target_feat is not None:
            tf = target_feat.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            if self._target_feat_last_block is not None:
                # Match the official feature distribution: the ``-1`` block is
                # exported PRE-final-norm by vLLM but the draft expects the
                # POST-final-norm last hidden. Normalize just that block.
                h = self._verifier_hidden_size
                start = self._target_feat_last_block * h
                block = tf[..., start : start + h]
                tf = tf.clone()
                tf[..., start : start + h] = apply_rms_norm(
                    block,
                    self.verifier_norm.weight.to(block.device),
                    eps=self.verifier_norm_eps,
                )
            tf_proj = self.target_proj(tf) * self.feat_scale
            if tf_proj.shape[:2] != inputs_embeds.shape[:2]:
                raise ValueError(
                    f"target_feat seq mismatch: proj={tuple(tf_proj.shape)} "
                    f"vs embeds={tuple(inputs_embeds.shape)}"
                )
            bsz = tf_proj.shape[0]
            keep_mask = (
                torch.rand(bsz, 1, 1, device=tf_proj.device) > self.target_feat_mask
            ).to(tf_proj.dtype)
            inputs_embeds = inputs_embeds + tf_proj * keep_mask

        # Run only the draft transformer (not the built-in ``lm_head``) so the
        # full ``[batch, seq, vocab]`` logits tensor is never materialized for the
        # loss. Logits are projected per-chunk inside the fused loss and, only
        # when needed, on demand for the logging metrics below.
        draft_hidden = self.draft_model.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state
        draft_lm_weight = self.draft_model.get_output_embeddings().weight
        device = draft_hidden.device

        ce_loss: torch.Tensor | None = None
        kd_loss: torch.Tensor | None = None
        token_weight: torch.Tensor | None = None
        student_logits: torch.Tensor | None = None

        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            valid_mask = (shift_labels != -100).to(draft_hidden.dtype)

            if prev_prob is not None and self.prev_prob_loss:
                shift_prev_prob = prev_prob[..., 1:].to(
                    device=device,
                    dtype=draft_hidden.dtype,
                )
                token_weight = valid_mask * shift_prev_prob
            else:
                token_weight = valid_mask

        has_teacher = teacher_hidden is not None
        if labels is not None and has_teacher and token_weight is not None:
            teacher_hidden_t = teacher_hidden.to(
                device=device,
                dtype=draft_hidden.dtype,
            )
            final_loss, ce_loss, kd_loss = weighted_ce_kd_loss_from_hidden_chunked(
                draft_hidden[..., :-1, :],
                draft_lm_weight,
                shift_labels,
                teacher_hidden_t[..., :-1, :],
                self.verifier_lm_head.weight.to(device),
                token_weight,
                float(self.kd_temperature),
                float(self.ce_alpha),
                float(self.kd_alpha),
            )
        else:
            # Edge paths (CE-only / KD-only / no labels) are rare in PARD-2
            # training; materialize the logits lazily for them.
            student_logits = F.linear(draft_hidden, draft_lm_weight)
            if labels is not None and token_weight is not None:
                ce_loss = weighted_ce_loss_chunked(
                    student_logits[..., :-1, :],
                    shift_labels,
                    token_weight,
                )

            if has_teacher:
                teacher_hidden_t = teacher_hidden.to(
                    device=student_logits.device,
                    dtype=student_logits.dtype,
                )
                shift_student_logits = student_logits[..., :-1, :]
                shift_teacher_hidden = teacher_hidden_t[..., :-1, :]

                if labels is not None and token_weight is not None:
                    kd_weight = token_weight
                elif prev_prob is not None and self.prev_prob_loss:
                    kd_weight = prev_prob[..., 1:].to(
                        device=student_logits.device,
                        dtype=student_logits.dtype,
                    )
                else:
                    kd_weight = None

                kd_loss = weighted_kd_loss_chunked(
                    shift_student_logits,
                    shift_teacher_hidden,
                    self.verifier_lm_head,
                    float(self.kd_temperature),
                    kd_weight,
                )

            if ce_loss is not None and kd_loss is not None:
                final_loss = ce_loss * self.ce_alpha + kd_loss * self.kd_alpha
            elif ce_loss is not None:
                final_loss = ce_loss
            elif kd_loss is not None:
                final_loss = kd_loss
            else:
                raise ValueError("No loss was computed")

        metrics: dict[str, torch.Tensor] = {
            "loss_sum": final_loss.detach().clone(),
            "loss_total": torch.tensor(1.0, device=final_loss.device),
        }
        if ce_loss is not None:
            metrics["ce_loss_sum"] = ce_loss.detach().clone()
            metrics["ce_loss_total"] = torch.tensor(1.0, device=final_loss.device)
        if kd_loss is not None:
            metrics["kd_loss_sum"] = kd_loss.detach().clone()
            metrics["kd_loss_total"] = torch.tensor(1.0, device=final_loss.device)
        if (
            self.compute_acceptance_metrics
            and labels is not None
            and teacher_hidden is not None
            and warp_single_length is not None
        ):
            lm_head = self.verifier_lm_head
            norm_weight = self.verifier_norm.weight.to(device)
            # The acceptance metric only needs detached logits; project them here
            # (under ``no_grad`` so no extra activation graph is built) when the
            # fused-from-hidden loss path did not already materialize them.
            if student_logits is None:
                with torch.no_grad():
                    metric_logits = F.linear(draft_hidden.detach(), draft_lm_weight)
            else:
                metric_logits = student_logits
            step_acceptance_metrics = compute_step_acceptance_metrics(
                student_logits=metric_logits,
                teacher_hidden=teacher_hidden,
                labels=labels,
                para_num=int(self.config.para_num),
                lm_head_weight=lm_head.weight.to(device),
                lm_head_bias=(
                    lm_head.bias.to(device)
                    if lm_head.bias is not None
                    else None
                ),
                norm_weight=norm_weight,
                norm_eps=self.verifier_norm_eps,
                warp_indices=warp_indices,
                warp_indices_len=warp_indices_len,
                warp_single_length=warp_single_length,
            )
            metrics.update(step_acceptance_metrics)
            del metric_logits
        if diag_metrics:
            metrics.update(diag_metrics)

        return None, final_loss, metrics

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,  # noqa: ARG003
        d2t: torch.Tensor | None = None,  # noqa: ARG003
        **kwargs: Any,
    ) -> "Pard2DraftModel":
        if t2d is not None or d2t is not None:
            raise ValueError("PARD-2 does not use draft vocabulary mappings")

        draft_name_or_path = kwargs.get("draft_name_or_path")
        if not draft_name_or_path:
            raise ValueError("--draft-name-or-path is required for pard2 training")

        verifier_name_or_path = kwargs["verifier_name_or_path"]
        target_layer_ids = resolve_target_layer_ids(
            kwargs.get("target_layer_ids"),
            verifier_name_or_path,
        )
        target_feat_dim = verifier_config.hidden_size * len(target_layer_ids)

        para_num = int(kwargs.get("para_num", 16))
        mask_token_ids = kwargs.get("mask_token_ids")
        if mask_token_ids is None:
            mask_token_id = kwargs.get("mask_token_id")
            if mask_token_id is None:
                raise ValueError(
                    "PARD-2 training requires --mask-token-id or --mask-token-ids"
                )
            mask_token_ids = [int(mask_token_id)] * max(para_num - 1, 1)
        if len(mask_token_ids) < para_num - 1:
            raise ValueError(
                f"Need at least {para_num - 1} mask token ids, got {len(mask_token_ids)}"
            )

        config = Pard2SpeculatorConfig(
            draft_name_or_path=draft_name_or_path,
            para_num=para_num,
            down_sample_ratio=float(kwargs.get("down_sample_ratio", 0.7)),
            down_sample_ratio_min=float(kwargs.get("down_sample_ratio_min", 0.1)),
            mask_token_ids=list(mask_token_ids),
            feat_scale=float(kwargs.get("feat_scale", 0.02)),
            proj_bias=bool(kwargs.get("proj_bias", False)),
            target_layer_ids=[int(x) for x in target_layer_ids],
            target_feat_dim=target_feat_dim,
            target_feat_mask=float(kwargs.get("target_feat_mask", 0.2)),
            ce_alpha=float(kwargs.get("ce_alpha", 0.1)),
            kd_alpha=float(kwargs.get("kd_alpha", 1.0)),
            kd_temperature=float(kwargs.get("kd_temperature", 1.0)),
            prev_prob_loss=bool(kwargs.get("prev_prob_loss", True)),
            end_token_id=kwargs.get("end_token_id"),
            pard_token=int(mask_token_ids[0]) if mask_token_ids else -1,
            speculators_config=SpeculatorsConfig(
                algorithm="pard2",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        speculative_tokens=para_num,
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(
                    verifier_config,
                    name_or_path=verifier_name_or_path,
                ),
            ),
        )
        return cls(config=config)

    @staticmethod
    def get_dataset_transform(noise_std: float):
        if noise_std <= 0:
            return None
        return AddUniformNoise(std=noise_std, tensors=("multi_layer_hidden_states",))

    @staticmethod
    def get_trainer_kwargs(**kwargs: Any) -> tuple[dict, dict]:
        del kwargs
        return {}, {}
