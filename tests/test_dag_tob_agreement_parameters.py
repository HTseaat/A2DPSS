from __future__ import annotations

from copy import deepcopy

import pytest

from beaver.dag_tob import DagTOB, Vertex
from scripts.analyze_a2dpss_network_logs import (
    _dag_agreement_parameter_view,
    _validate_dag_selectblock_metrics,
)
from scripts.analyze_a2dpss_tob_pair import _agreement_view


def _summary(**overrides):
    summary = {
        "dag_agreement_parameters_schema": 1,
        "M": 4,
        "committee_m": 4,
        "quorum_h": 3,
        "agreement_t": 1,
        "coverage_threshold": 2,
    }
    summary.update(overrides)
    return summary


def test_analyzer_accepts_phase1_dag_agreement_parameters() -> None:
    assert _dag_agreement_parameter_view(_summary(), node=0) == {
        "committee_m": 4,
        "quorum_h": 3,
        "agreement_t": 1,
        "coverage_threshold": 2,
    }


def test_analyzer_keeps_phase0_log_compatibility() -> None:
    assert _dag_agreement_parameter_view({"M": 4}, node=0) is None


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"agreement_t": 0},
            "agreement_t does not equal committee_m - quorum_h",
        ),
        (
            {
                "M": 5,
                "committee_m": 5,
                "quorum_h": 3,
                "agreement_t": 2,
                "coverage_threshold": 3,
            },
            "violates the DAG Byzantine quorum bound",
        ),
        (
            {"coverage_threshold": 3},
            "invalid qid coverage threshold",
        ),
    ],
)
def test_analyzer_rejects_invalid_phase1_dag_agreement_parameters(
    overrides, message
) -> None:
    with pytest.raises(SystemExit, match=message):
        _dag_agreement_parameter_view(_summary(**overrides), node=2)


def _selectblock_summary():
    dag = DagTOB(my_id=0, M=4, quorum_h=3, agreement_t=1)
    dag.deliver_message("qid-a", "mid-a", b"A")
    dag.deliver_message("qid-b", "mid-b", b"B")
    parents = frozenset(dag.DAG[0])
    for source in (0, 1):
        dag.on_r_deliver(
            Vertex(
                round=1,
                source=source,
                block={"qid-a": ["mid-a"]},
                strong_edges=parents,
                weak_edges=frozenset(),
            )
        )
    dag.create_new_vertex(2, source=2)
    return {
        **_summary(),
        "agreement_parameters_schema": 1,
        "justifier_metrics": (
            dag.wait_predicate.registry.metrics_snapshot()
        ),
        "dag_qid_causal_coverage": (
            dag.snapshot_qid_causal_coverage()
        ),
        "dag_selectblock_metrics": (
            dag.dag_selectblock_metrics_snapshot()
        ),
        "protocol_delivery_events": [],
    }


def test_selectblock_analyzer_rebuilds_suppression_and_predicate_counts() -> None:
    summary = _selectblock_summary()

    metrics = _validate_dag_selectblock_metrics(summary, node=0)

    assert metrics["candidate_qids"] == 2
    assert metrics["evaluated_qids"] == 1
    assert metrics["skipped_causally_covered_qids"] == 1


def test_selectblock_analyzer_rejects_nonprior_coverage_witness() -> None:
    summary = deepcopy(_selectblock_summary())
    summary["dag_selectblock_metrics"]["coverage_records"][0][
        "cover_round"
    ] = 2
    summary["dag_qid_causal_coverage"]["coverage_records"][0][
        "cover_round"
    ] = 2

    with pytest.raises(SystemExit, match="prior-round witness"):
        _validate_dag_selectblock_metrics(summary, node=0)


def test_selectblock_analyzer_rejects_predicate_record_mismatch() -> None:
    summary = deepcopy(_selectblock_summary())
    summary["justifier_metrics"]["select"]["records"] = []

    with pytest.raises(SystemExit, match="proposer predicate records"):
        _validate_dag_selectblock_metrics(summary, node=0)


def test_paired_agreement_view_is_protocol_neutral() -> None:
    summary = {**_summary(), "agreement_parameters_schema": 1}

    assert _agreement_view(summary) == {
        "M": 4,
        "committee_m": 4,
        "quorum_h": 3,
        "agreement_t": 1,
        "coverage_threshold": 2,
    }
