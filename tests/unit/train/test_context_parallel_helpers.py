import torch

from speculators.models.eagle3.core import (
    _build_document_ids,
    _extract_lengths_from_document_ids,
)


def test_build_document_ids_with_padding():
    lengths = torch.tensor([2, 1], dtype=torch.long)
    doc_ids = _build_document_ids(lengths, total_seq_len=5)
    expected = torch.tensor([[0, 0, 1, -1, -1]], dtype=torch.long)
    assert torch.equal(doc_ids, expected)


def test_extract_lengths_from_document_ids():
    doc_ids = torch.tensor([[0, 0, 1, -1, 1]], dtype=torch.long)
    lengths = _extract_lengths_from_document_ids(doc_ids)
    expected = torch.tensor([2, 2], dtype=torch.long)
    assert torch.equal(lengths, expected)
