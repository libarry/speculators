"""Logging / metric reduction under Ulysses sequence parallel."""

import pytest
import torch

from speculators.train.utils import (
    is_replica_weighted_total_key,
    normalize_counted_metrics,
    scale_replica_totals_for_sp,
)


def test_replica_total_keys_vs_token_counts():
    assert is_replica_weighted_total_key("loss_total")
    assert is_replica_weighted_total_key("loss_0_total")
    assert is_replica_weighted_total_key("ce_loss_0_total")
    assert is_replica_weighted_total_key("kl_div_loss_total")
    assert is_replica_weighted_total_key("confidence_loss_total")
    assert is_replica_weighted_total_key("eal_total")
    assert not is_replica_weighted_total_key("full_acc_0_total")
    assert not is_replica_weighted_total_key("cond_acc_2_total")
    assert not is_replica_weighted_total_key("accept_rate_total")
    assert not is_replica_weighted_total_key("confidence_abs_error_total")


def test_scale_replica_totals_noop_without_sp():
    metrics = {"loss_total": torch.tensor(1.0), "full_acc_0_total": torch.tensor(8.0)}
    scale_replica_totals_for_sp(metrics, sp_size=1)
    assert metrics["loss_total"].item() == 1.0
    assert metrics["full_acc_0_total"].item() == 8.0


def test_scale_replica_totals_divides_only_replica_weights():
    metrics = {
        "loss_total": torch.tensor(1.0),
        "loss_0_total": torch.tensor(1.0),
        "full_acc_0_total": torch.tensor(8.0),
    }
    scale_replica_totals_for_sp(metrics, sp_size=4)
    assert metrics["loss_total"].item() == pytest.approx(0.25)
    assert metrics["loss_0_total"].item() == pytest.approx(0.25)
    assert metrics["full_acc_0_total"].item() == 8.0


def test_logged_loss_after_sp_world_sum_matches_global_mean():
    """SP ranks hold scaled shards ``S_local / D_global``; totals are 1/sp_size.

    WORLD SUM then ``sum/total`` must recover the global mean, not mean/sp_size.
    """
    sp_size = 4
    # Rank 0 prompt-only (S=0); ranks 1-3 have S_local=4, D_global=6 → mean=2.
    shard_loss_sums = [0.0, 4.0 / 6.0, 4.0 / 6.0, 4.0 / 6.0]
    reduced = {"loss_sum": 0.0, "loss_total": 0.0, "full_acc_0_sum": 0.0, "full_acc_0_total": 0.0}
    for s_loss, acc_sum, acc_tot in zip(
        shard_loss_sums,
        [0.0, 1.0, 1.0, 2.0],
        [0.0, 2.0, 2.0, 2.0],
        strict=True,
    ):
        local = {
            "loss_sum": s_loss,
            "loss_total": 1.0,
            "full_acc_0_sum": acc_sum,
            "full_acc_0_total": acc_tot,
        }
        scale_replica_totals_for_sp(local, sp_size)
        for k in reduced:
            reduced[k] += local[k]

    out = normalize_counted_metrics(reduced, world_size=1)  # dp_size=1
    assert out["loss"] == pytest.approx(2.0)
    assert out["full_acc_0"] == pytest.approx(4.0 / 6.0)


def test_dp_only_replica_average_unchanged():
    # Two DP ranks, no SP: means 1.0 and 3.0 → logged 2.0.
    reduced = {"loss_sum": 1.0 + 3.0, "loss_total": 1.0 + 1.0}
    out = normalize_counted_metrics(reduced, world_size=2)
    assert out["loss"] == pytest.approx(2.0)
