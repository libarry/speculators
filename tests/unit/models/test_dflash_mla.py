"""Unit tests for DFlash/DSpark MLA attention."""

import math

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import (
    DFlashDraftModel,
    resolve_mla_config_kwargs,
)
from speculators.models.dflash.mla_rope import (
    MLAYarnRotaryEmbedding,
    yarn_get_mscale,
)
from speculators.models.dflash.model_definitions import (
    Qwen3DFlashAttention,
    Qwen3DFlashMLAAttention,
)
from speculators.models.dspark import DSparkDraftModel, DSparkSpeculatorConfig
from speculators.train.config.schema import TrainConfig


def _tiny_tl_config(**overrides) -> Qwen3Config:
    kwargs = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=2048,
        rope_theta=50000.0,
        _attn_implementation="eager",
    )
    kwargs.update(overrides)
    return Qwen3Config(**kwargs)  # type: ignore[arg-type]


def _mla_speculator_config(**overrides) -> DSparkSpeculatorConfig:
    tl_config = _tiny_tl_config()
    kwargs = dict(
        transformer_layer_config=tl_config,
        draft_vocab_size=128,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        attention_type="mla",
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        markov_rank=8,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
    )
    kwargs.update(overrides)
    return DSparkSpeculatorConfig(**kwargs)


def _init_vocab_weights(model: DFlashDraftModel) -> None:
    torch.nn.init.normal_(model.embed_tokens.weight)
    torch.nn.init.normal_(model.lm_head.weight)
    torch.nn.init.normal_(model.verifier_lm_head.weight)
    torch.nn.init.ones_(model.verifier_norm.weight)


class TestMLAWeightLayout:
    def test_mla_layout_matches_torchspec(self):
        model = DSparkDraftModel(_mla_speculator_config())
        sd = model.state_dict()
        for name in (
            "layers.0.self_attn.q_a_proj.weight",
            "layers.0.self_attn.q_a_layernorm.weight",
            "layers.0.self_attn.q_b_proj.weight",
            "layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "layers.0.self_attn.kv_a_layernorm.weight",
            "layers.0.self_attn.kv_b_proj.weight",
            "layers.0.self_attn.o_proj.weight",
            "layers.1.self_attn.o_proj.weight",
            "markov_head.markov_w1.weight",
            "confidence_head.proj.weight",
        ):
            assert name in sd
        assert "layers.0.self_attn.q_norm.weight" not in sd
        assert "layers.0.self_attn.k_norm.weight" not in sd
        # H=4, qk_head_dim=24, v_head_dim=16, hidden=64, q_lora=32, kv_lora=16
        assert sd["layers.0.self_attn.q_b_proj.weight"].shape == (4 * 24, 32)
        assert sd["layers.0.self_attn.kv_a_proj_with_mqa.weight"].shape == (
            16 + 8,
            64,
        )
        assert sd["layers.0.self_attn.kv_b_proj.weight"].shape == (
            4 * (16 + 16),
            16,
        )
        assert sd["layers.0.self_attn.o_proj.weight"].shape == (64, 4 * 16)
        assert isinstance(model.layers[0].self_attn, Qwen3DFlashMLAAttention)

    def test_gqa_default_uses_qwen_attention(self):
        tl_config = _tiny_tl_config()
        config = DFlashSpeculatorConfig(
            transformer_layer_config=tl_config,
            draft_vocab_size=128,
            block_size=4,
            aux_hidden_state_layer_ids=[0],
            mask_token_id=0,
        )
        model = DFlashDraftModel(config)
        assert isinstance(model.layers[0].self_attn, Qwen3DFlashAttention)
        assert config.attention_type == "gqa"

    def test_output_gate_unsupported(self):
        with pytest.raises(NotImplementedError, match="mla_use_output_gate"):
            DSparkDraftModel(_mla_speculator_config(mla_use_output_gate=True))

    def test_missing_mla_dims_raises(self):
        with pytest.raises(ValueError, match="requires MLA dims"):
            DFlashDraftModel(
                DFlashSpeculatorConfig(
                    transformer_layer_config=_tiny_tl_config(),
                    draft_vocab_size=128,
                    block_size=4,
                    aux_hidden_state_layer_ids=[0],
                    mask_token_id=0,
                    attention_type="mla",
                )
            )


class TestMLAForward:
    def test_mla_backbone_forward_finite(self):
        torch.manual_seed(0)
        model = DSparkDraftModel(_mla_speculator_config())
        _init_vocab_weights(model)
        model.eval()

        seq_len = 16
        hidden_states = torch.randn(1, seq_len, 2 * 64)
        verifier_last = torch.randn(1, seq_len, 64)
        input_ids = torch.randint(0, 128, (1, seq_len))
        loss_mask = torch.ones(1, seq_len)
        document_ids = torch.zeros(1, seq_len, dtype=torch.long)

        with torch.no_grad():
            hidden, logits, targets, aligned_loss_mask, _ = model._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last,
                document_ids,
                max_anchors=4,
            )
        assert torch.isfinite(hidden).all()
        assert torch.isfinite(logits).all()
        assert torch.isfinite(targets).all()
        assert aligned_loss_mask.shape[-1] == 16  # 4 anchors * block_size 4


class TestMLAConfigResolution:
    def test_resolve_uses_defaults_when_unset(self):
        tl = _tiny_tl_config()
        resolved = resolve_mla_config_kwargs(
            attention_type="mla",
            verifier_config=tl,
        )
        assert resolved["attention_type"] == "mla"
        assert resolved["q_lora_rank"] == 1536
        assert resolved["kv_lora_rank"] == 512
        assert resolved["qk_nope_head_dim"] == 128
        assert resolved["qk_rope_head_dim"] == 64
        assert resolved["v_head_dim"] == 128

    def test_resolve_inherits_from_verifier_attrs(self):
        tl = _tiny_tl_config()
        tl.q_lora_rank = 64
        tl.kv_lora_rank = 32
        tl.qk_nope_head_dim = 20
        tl.qk_rope_head_dim = 12
        tl.v_head_dim = 20
        resolved = resolve_mla_config_kwargs(
            attention_type="mla",
            verifier_config=tl,
        )
        assert resolved["q_lora_rank"] == 64
        assert resolved["kv_lora_rank"] == 32
        assert resolved["qk_nope_head_dim"] == 20
        assert resolved["qk_rope_head_dim"] == 12
        assert resolved["v_head_dim"] == 20

    def test_gqa_clears_mla_dims(self):
        resolved = resolve_mla_config_kwargs(
            attention_type="gqa",
            verifier_config=_tiny_tl_config(),
            q_lora_rank=64,
            kv_lora_rank=32,
        )
        assert resolved == {
            "attention_type": "gqa",
            "q_lora_rank": None,
            "kv_lora_rank": None,
            "qk_nope_head_dim": None,
            "qk_rope_head_dim": None,
            "v_head_dim": None,
            "mla_use_output_gate": False,
        }

    def test_build_base_config_kwargs_wires_mla(self, monkeypatch):
        from speculators.config import VerifierConfig

        monkeypatch.setattr(
            VerifierConfig,
            "from_pretrained",
            classmethod(
                lambda cls, name_or_path, **kwargs: cls(
                    name_or_path=name_or_path, architectures=["Qwen3ForCausalLM"]
                )
            ),
        )
        kwargs = DFlashDraftModel._build_base_config_kwargs(
            algorithm="dspark",
            verifier_config=_tiny_tl_config(),
            verifier_name_or_path="unused",
            draft_vocab_size=128,
            block_size=4,
            target_layer_ids=[0],
            attention_type="mla",
            q_lora_rank=32,
            kv_lora_rank=16,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
        )
        assert kwargs["attention_type"] == "mla"
        assert kwargs["kv_lora_rank"] == 16
        config = DSparkSpeculatorConfig(**kwargs)
        assert config.attention_type == "mla"
        assert config.qk_rope_head_dim == 8

    def test_train_config_attention_type_cli_field(self):
        flat = TrainConfig.from_flat(
            {
                "verifier_name_or_path": "m",
                "speculator_type": "dspark",
                "attention_type": "mla",
                "kv_lora_rank": 256,
            }
        ).flatten()
        assert flat["attention_type"] == "mla"
        assert flat["kv_lora_rank"] == 256


class TestMLAYarnRope:
    def test_yarn_softmax_scale(self):
        tl = _tiny_tl_config()
        tl.rope_scaling = {
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 64,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        }
        config = _mla_speculator_config(transformer_layer_config=tl)
        model = DSparkDraftModel(config)
        attn = model.layers[0].self_attn
        assert isinstance(attn.rotary_emb, MLAYarnRotaryEmbedding)
        assert attn.rotary_emb.dim == 8
        mscale = yarn_get_mscale(32.0, 1.0)
        expected = (mscale * mscale) / math.sqrt(24)
        assert attn.scaling == pytest.approx(expected)
