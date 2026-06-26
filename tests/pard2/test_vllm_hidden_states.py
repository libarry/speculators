"""Tests for vLLM hidden-state layer reordering."""

from speculators.models.pard2.vllm_hidden_states import (
    pard_layers_to_vllm_ids,
    reorder_vllm_hidden_states,
    vllm_hidden_state_permutation,
)


def test_pard_layers_to_vllm_ids_qwen3_8b():
    assert pard_layers_to_vllm_ids([-1, -8, -16, -24], 36) == [36, 29, 21, 13]


def test_vllm_hidden_state_permutation_qwen3_8b():
  # vLLM export order is ascending capture order [13, 21, 29, 36]
    assert vllm_hidden_state_permutation([-1, -8, -16, -24], 36) == [3, 2, 1, 0]


def test_reorder_vllm_hidden_states():
    import torch

    hs = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    reordered = reorder_vllm_hidden_states(hs, [-1, -2, -3, -4], 4)
    assert torch.equal(reordered[:, 0], hs[:, 3])
    assert torch.equal(reordered[:, 1], hs[:, 2])
    assert torch.equal(reordered[:, 2], hs[:, 1])
    assert torch.equal(reordered[:, 3], hs[:, 0])
