"""Phase-4 contracts for protocol-neutral A²DPSS CPU accounting."""

from beaver.dag_tob import JustifierRegistry, Qid
from beaver.experiment_trace import (
    aggregate_a2dpss_computation_metrics,
    build_a2dpss_computation_metrics,
    validate_a2dpss_computation_metrics,
)


def _snapshot(protocol: str, node: int, scale: float = 1.0) -> dict:
    proposer_record = {
        "qid": "qid",
        "cpu_seconds": 0.25 * scale,
        "accepted": True,
    }
    receiver_record = {
        "qid": "qid",
        "cpu_seconds": 0.5 * scale,
        "accepted": True,
    }
    return build_a2dpss_computation_metrics(
        protocol=protocol,
        node=node,
        proposer_predicate={
            "operation": "select" if protocol == "dag-tob" else "none",
            "invocations": 1,
            "cpu_seconds": proposer_record["cpu_seconds"],
            "records": [proposer_record],
        },
        receiver_predicate={
            "operation": "verify",
            "invocations": 1,
            "cpu_seconds": receiver_record["cpu_seconds"],
            "records": [receiver_record],
        },
        candidate_formation={
            "invocations": 2,
            "cpu_seconds": 0.1 * scale,
        },
        pvss_generation={"invocations": 1, "cpu_seconds": 1.0 * scale},
        pvss_verification={"invocations": 2, "cpu_seconds": 2.0 * scale},
        public_finalization={"invocations": 1, "cpu_seconds": 3.0 * scale},
        successor_reconstruction={
            "invocations": 1,
            "cpu_seconds": 4.0 * scale,
        },
    )


def test_computation_metrics_validate_and_exclude_common_coin() -> None:
    snapshot = _snapshot("dag-tob", 0)

    assert validate_a2dpss_computation_metrics(
        snapshot,
        protocol="dag-tob",
        node=0,
    ) == []
    assert snapshot["predicate"]["total"] == {
        "invocations": 2,
        "cpu_seconds": 0.75,
    }
    assert snapshot["auxiliary"]["candidate_formation"][
        "formal_predicate"
    ] is False
    assert snapshot["excluded"] == ["threshold_bls_common_coin"]
    assert "common_coin" not in snapshot["resharing"]


def test_computation_metrics_network_total_is_node_sum() -> None:
    snapshots = {
        0: _snapshot("acs-tob", 0, 1.0),
        1: _snapshot("acs-tob", 1, 2.0),
    }

    aggregate = aggregate_a2dpss_computation_metrics(snapshots)

    assert aggregate["nodes"] == [0, 1]
    assert aggregate["predicate"]["total"]["invocations"] == 4
    assert aggregate["predicate"]["total"]["cpu_seconds"] == 2.25
    assert aggregate["resharing"]["pvss_generation"] == {
        "invocations": 2,
        "cpu_seconds": 3.0,
    }
    assert aggregate["resharing"]["successor_reconstruction"] == {
        "invocations": 2,
        "cpu_seconds": 12.0,
    }


def test_registry_records_wall_and_cpu_for_each_formal_call() -> None:
    registry = JustifierRegistry()
    qid = Qid.build("demo", sid="phase4").encode()
    entries = [("node_0", b"payload")]

    assert registry.select_justified_subset(qid, entries) == entries
    assert registry.verify_qid_set(qid, entries)

    metrics = registry.metrics_snapshot()
    for phase in ("select", "verify"):
        metric = metrics[phase]
        assert metric["invocations"] == 1
        assert metric["elapsed_sec"] >= 0.0
        assert metric["cpu_seconds"] >= 0.0
        assert len(metric["records"]) == 1
        assert metric["records"][0]["accepted"] is True
        assert metric["records"][0]["cpu_seconds"] >= 0.0

