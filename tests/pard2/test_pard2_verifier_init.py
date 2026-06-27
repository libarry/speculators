"""Regression tests for two silent PARD-2 bugs that drive acceptance to ~0.

1. ``post_init`` must not clobber the loaded verifier ``lm_head`` / ``norm``
   weights (they back the KD target, prev_prob weights, and acc metric).
2. ``forward`` must apply the verifier final norm to the ``-1`` target-feature
   block, because vLLM exports that layer PRE-final-norm while the draft was
   trained on the POST-final-norm last hidden.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from transformers import LlamaConfig  # noqa: E402

from speculators.models.pard2.core import Pard2DraftModel  # noqa: E402


def _tiny_cfg() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=64,
    )


class _FakeInner(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(cfg.hidden_size, cfg.hidden_size)]
        )

    def forward(self, inputs_embeds=None, **kwargs):
        del kwargs
        return type("Out", (), {"last_hidden_state": inputs_embeds})()


class _FakeBase(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.embed = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.model = _FakeInner(cfg)
        self.lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, inputs_embeds=None, **kwargs):
        del kwargs
        return type("Out", (), {"logits": self.lm_head(inputs_embeds)})()


_LM_SENTINEL = 0.1234
_NORM_SENTINEL = 0.5678


def _build_model(monkeypatch) -> Pard2DraftModel:
    cfg = _tiny_cfg()

    def _make_lm_head(_path):
        head = torch.nn.Linear(32, 64, bias=False)
        torch.nn.init.constant_(head.weight, _LM_SENTINEL)
        return head

    def _make_norm(_path):
        norm = torch.nn.RMSNorm(32, eps=1e-6)
        torch.nn.init.constant_(norm.weight, _NORM_SENTINEL)
        return norm, 1e-6

    monkeypatch.setattr(
        "speculators.models.pard2.core.AutoConfig.from_pretrained",
        lambda path, **kwargs: cfg,
    )
    monkeypatch.setattr(
        "speculators.models.pard2.core.AutoModelForCausalLM.from_pretrained",
        lambda path, **kwargs: _FakeBase(cfg),
    )
    monkeypatch.setattr(
        "speculators.models.pard2.core.load_verifier_lm_head", _make_lm_head
    )
    monkeypatch.setattr(
        "speculators.models.pard2.core.load_verifier_final_norm", _make_norm
    )

    return Pard2DraftModel.from_training_args(
        verifier_config=cfg,
        verifier_name_or_path="fake-verifier",
        draft_name_or_path="fake-draft",
        para_num=2,
        mask_token_ids=[63, 63],
        target_layer_ids=[-1, -2, -3, -4],
        feat_scale=0.02,
        target_feat_mask=0.0,
    )


def test_verifier_weights_survive_post_init(monkeypatch):
    """post_init must not re-initialize the loaded verifier head/norm."""
    model = _build_model(monkeypatch)

    assert torch.allclose(
        model.verifier_lm_head.weight,
        torch.full_like(model.verifier_lm_head.weight, _LM_SENTINEL),
    ), "verifier_lm_head was clobbered by post_init (KD target would be garbage)"
    assert torch.allclose(
        model.verifier_norm.weight,
        torch.full_like(model.verifier_norm.weight, _NORM_SENTINEL),
    ), "verifier_norm was clobbered by post_init (gold_prob would be garbage)"


def test_target_feat_last_block_is_norm_invariant(monkeypatch):
    """The -1 feature block is RMS-normed, so scaling it must not change loss."""
    model = _build_model(monkeypatch).eval()
    assert model._target_feat_last_block == 0
    assert model._verifier_hidden_size == 32

    torch.manual_seed(0)
    seq = 6
    input_ids = torch.randint(0, 64, (1, seq))
    target_feat = torch.randn(1, seq, 32 * 4)

    scaled = target_feat.clone()
    scaled[..., 0:32] *= 7.0  # scale only the -1 (last-layer) block

    with torch.no_grad():
        _, loss_a, _ = model(
            input_ids=input_ids, labels=input_ids, target_feat=target_feat
        )
        _, loss_b, _ = model(
            input_ids=input_ids, labels=input_ids, target_feat=scaled
        )

    assert torch.allclose(loss_a, loss_b, atol=1e-4), (
        "scaling the -1 block changed the loss -> verifier final norm not applied"
    )

    # A non-normed block (e.g. -2) should NOT be scale-invariant: sanity guard
    scaled_other = target_feat.clone()
    scaled_other[..., 32:64] *= 7.0
    with torch.no_grad():
        _, loss_c, _ = model(
            input_ids=input_ids, labels=input_ids, target_feat=scaled_other
        )
    assert not torch.allclose(loss_a, loss_c, atol=1e-4)
