"""Map vLLM extract_hidden_states tensors to official PARD layer order."""

from __future__ import annotations


def pard_layers_to_vllm_ids(
    target_layer_ids: list[int],
    num_hidden_layers: int,
) -> list[int]:
    """Convert PARD negative layer ids to vLLM 1-based eagle aux layer ids."""
    return [
        num_hidden_layers + 1 + layer_id if layer_id < 0 else layer_id
        for layer_id in target_layer_ids
    ]


def vllm_hidden_state_permutation(
    target_layer_ids: list[int],
    num_hidden_layers: int,
) -> list[int]:
    """Index permutation from vLLM export order to PARD ``target_layer_ids`` order.

    vLLM appends ``aux_hidden_states`` in ascending layer id during the forward
    pass (e.g. 13, 21, 29, 36), while PARD concatenates in ``target_layer_ids``
    order (e.g. -1, -8, -16, -24 -> 36, 29, 21, 13).
    """
    vllm_layer_ids = pard_layers_to_vllm_ids(target_layer_ids, num_hidden_layers)
    ascending_indices = {layer_id: idx for idx, layer_id in enumerate(sorted(vllm_layer_ids))}
    return [ascending_indices[layer_id] for layer_id in vllm_layer_ids]


def reorder_vllm_hidden_states(
    hidden_states,
    target_layer_ids: list[int],
    num_hidden_layers: int,
):
    """Reorder ``[seq, num_layers, hidden]`` from vLLM export to PARD order."""
    perm = vllm_hidden_state_permutation(target_layer_ids, num_hidden_layers)
    return hidden_states[:, perm, :]
