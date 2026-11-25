from __future__ import annotations

import pytest

from beaver.experiment_trace import (
    A2DPSSHandoverMetrics,
    validate_a2dpss_handover_metrics,
)


def test_successor_timeline_uses_relative_monotonic_boundaries() -> None:
    metrics = A2DPSSHandoverMetrics(
        protocol="dag-tob",
        node=0,
        qid="qid-a2dpss",
        is_holder=True,
        is_successor=True,
    )
    metrics.start(100.0)
    metrics.mark_contribution_sent(101.0)
    metrics.mark_pool_ready(102.0)
    metrics.mark_ordered(105.0)
    metrics.mark_public_finalized(106.0)
    metrics.mark_successor_done(107.5)

    snapshot = metrics.snapshot()
    assert snapshot["timestamps_sec"] == {
        "t0": 0.0,
        "t_contribution_sent": 1.0,
        "t_pool_ready": 2.0,
        "t_ordered": 5.0,
        "t_public_finalized": 6.0,
        "t_successor_done": 7.5,
        "frozen_at": 7.5,
    }
    assert snapshot["durations_sec"] == {
        "contribution_generation": 1.0,
        "pool_ready_latency": 2.0,
        "ordering_latency": 3.0,
        "public_finalize_latency": 1.0,
        "successor_latency": 1.5,
        "handover_latency": 7.5,
        "local_completion_latency": 7.5,
    }
    assert validate_a2dpss_handover_metrics(
        snapshot,
        protocol="dag-tob",
        node=0,
        qid="qid-a2dpss",
        is_successor=True,
    ) == []
    with pytest.raises(RuntimeError, match="frozen"):
        metrics.mark_ordered(108.0)


def test_non_successor_freezes_at_public_finalization() -> None:
    metrics = A2DPSSHandoverMetrics(
        protocol="acs-tob",
        node=3,
        qid="qid-a2dpss",
        is_holder=False,
        is_successor=False,
    )
    metrics.start(20.0)
    metrics.mark_pool_ready(21.0)
    metrics.mark_ordered(22.0)
    metrics.mark_public_finalized(23.0)

    snapshot = metrics.snapshot()
    assert snapshot["timestamps_sec"]["t_successor_done"] is None
    assert snapshot["timestamps_sec"]["frozen_at"] == 3.0
    assert snapshot["durations_sec"]["handover_latency"] is None
    assert snapshot["durations_sec"]["local_completion_latency"] == 3.0
    assert validate_a2dpss_handover_metrics(
        snapshot,
        protocol="acs-tob",
        node=3,
        qid="qid-a2dpss",
        is_successor=False,
    ) == []


def test_invalid_order_is_rejected() -> None:
    metrics = A2DPSSHandoverMetrics(
        protocol="acs-tob",
        node=0,
        qid="qid-a2dpss",
        is_holder=True,
        is_successor=True,
    )
    metrics.start(10.0)
    with pytest.raises(RuntimeError, match="pool ready"):
        metrics.mark_ordered(11.0)
    metrics.mark_contribution_sent(12.0)
    with pytest.raises(ValueError, match="not monotonic"):
        metrics.mark_pool_ready(11.0)
