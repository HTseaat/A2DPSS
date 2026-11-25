"""R0 contract tests for the current-paper ATCaaS DAG-TOB.

These tests deliberately separate behavior already supported by the legacy
implementation from target behavior that belongs to later revision stages.
Known protocol gaps are marked ``xfail(strict=True)``.  When a revision stage
implements one of those behaviors, its marker must be removed in the same
change so an unexpected pass cannot be silently ignored.

No cryptographic common-coin protocol is used.  R4 tests replay a pre-generated
VRF committee-role trace through the production ideal-coin interface.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Iterable, Optional, Sequence

import pytest

from beaver.async_yoso_acs import (
    CoreBlock,
    LocalAsyncYosoACS,
    NetworkAsyncYosoACS,
    ProposedBlock,
)
from beaver.async_yoso_coin import SharedIdealCoinOracle
from beaver.dag_tob import (
    DagTOB,
    DeterministicLeaderOracle,
    IdealCoinLeaderOracle,
    Justifier,
    JustifierRegistry,
    PoolEntry,
    Qid,
    RefBlock,
    TypedWaitPredicate,
    Vertex,
    VertexId,
    simulate_one_wave,
)
from beaver.protocol_trace import RecordedCommitteeRoleTrace, RecordedIdealCoin


class KindValidityJustifier(Justifier):
    """Accept exactly one configured qid kind; supports old and target APIs."""

    def __init__(self, accepted: bool) -> None:
        self.accepted = bool(accepted)

    def select_justified_subset(
        self,
        qid: Qid,
        entries: Sequence[PoolEntry],
    ) -> Optional[list[PoolEntry]]:
        return list(entries) if self.accepted and entries else None

    def verify_qid_set(
        self,
        qid: Qid,
        entries: Sequence[PoolEntry],
    ) -> bool:
        return self.accepted and bool(entries)


def _make_vertex(
    *,
    round_id: int,
    source: int,
    block: Optional[RefBlock] = None,
    strong_edges: Iterable[VertexId] = (),
    weak_edges: Iterable[VertexId] = (),
) -> Vertex:
    return Vertex(
        round=int(round_id),
        source=int(source),
        block=block,
        strong_edges=frozenset(strong_edges),
        weak_edges=frozenset(weak_edges),
        proof=None,
    )


def _install_vertex(dag: DagTOB, vertex: Vertex) -> None:
    dag.DAG.setdefault(vertex.round, {})[vertex.vid] = vertex


def _install_ordered_vertex(dag: DagTOB, vertex: Vertex) -> None:
    _install_vertex(dag, vertex)
    dag.ordered_vertices.append(vertex.vid)
    dag._ordered_vertices_set.add(vertex.vid)


def _parent_edges(dag: DagTOB, round_id: int) -> frozenset[VertexId]:
    return frozenset(dag.DAG.get(round_id, {}))


def _typed_node(*, deliver_cb=None, M: int = 3, quorum_h: int = 2) -> DagTOB:
    registry = JustifierRegistry()
    registry.register("accept", KindValidityJustifier(True))
    registry.register("reject", KindValidityJustifier(False))
    return DagTOB(
        my_id=0,
        M=M,
        quorum_h=quorum_h,
        agreement_t=0,
        wait_predicate=TypedWaitPredicate(registry),
        deliver_cb=deliver_cb,
    )


def _drive_ordered_delivery(dag: DagTOB) -> None:
    dag._deliver_queries_from_ordered_vertices()


def test_coverage_requires_admitted_vertices_before_suppression() -> None:
    dag = DagTOB(my_id=0, M=4, quorum_h=3, agreement_t=1)

    assert dag.agreement_t == 1
    assert dag.coverage_threshold == 2

    dag.deliver_message("qid-demo", "mid-demo", b"payload")
    first = dag.create_new_vertex(1, source=0)
    second = dag.create_new_vertex(2, source=0)

    # Locally created but unadmitted vertices are not causal witnesses.
    assert first.block == {"qid-demo": ["mid-demo"]}
    assert second.block == {"qid-demo": ["mid-demo"]}
    assert (
        dag.wait_predicate.registry.metrics_snapshot()["select"]["invocations"]
        == 2
    )


def _coverage_node() -> DagTOB:
    return DagTOB(my_id=0, M=4, quorum_h=3, agreement_t=1)


def _admit(
    dag: DagTOB,
    *,
    round_id: int,
    source: int,
    block: Optional[RefBlock],
    strong_edges: Optional[Iterable[VertexId]] = None,
) -> Vertex:
    vertex = _make_vertex(
        round_id=round_id,
        source=source,
        block=block,
        strong_edges=(
            _parent_edges(dag, round_id - 1)
            if strong_edges is None
            else strong_edges
        ),
    )
    dag.on_r_deliver(vertex)
    return vertex


def test_admitted_vertices_trigger_per_round_causal_coverage() -> None:
    dag = _coverage_node()
    dag.deliver_message("qid-a", "mid-a", b"A")

    _admit(dag, round_id=1, source=0, block={"qid-a": ["mid-a"]})
    assert dag.qid_covered_at_round == {}
    _admit(dag, round_id=1, source=1, block={"qid-a": ["mid-a"]})

    snapshot = dag.snapshot_qid_causal_coverage()
    assert snapshot["reference_counts_by_round"] == {
        "1": {"qid-a": 2}
    }
    assert snapshot["covered_at_round"] == {"qid-a": 1}
    assert snapshot["coverage_records"] == [
        {"qid": "qid-a", "cover_round": 1, "witness_count": 2}
    ]


def test_buffered_vertex_counts_only_after_availability_promotion() -> None:
    dag = _coverage_node()
    vertex = _admit(
        dag,
        round_id=1,
        source=0,
        block={"qid-late": ["mid-late"]},
    )

    assert vertex.vid in dag.buffer
    assert dag.snapshot_qid_causal_coverage()[
        "reference_counts_by_round"
    ] == {}

    dag.deliver_message("qid-late", "mid-late", b"late")

    assert vertex.vid in dag.DAG[1]
    assert dag.snapshot_qid_causal_coverage()[
        "reference_counts_by_round"
    ] == {"1": {"qid-late": 1}}


def test_coverage_does_not_add_witnesses_across_rounds() -> None:
    dag = _coverage_node()
    dag.deliver_message("qid-a", "mid-a", b"A")
    _admit(dag, round_id=1, source=0, block={"qid-a": ["mid-a"]})
    _admit(dag, round_id=1, source=1, block=None)
    _admit(dag, round_id=1, source=2, block=None)
    _admit(dag, round_id=2, source=0, block={"qid-a": ["mid-a"]})

    snapshot = dag.snapshot_qid_causal_coverage()
    assert snapshot["reference_counts_by_round"] == {
        "1": {"qid-a": 1},
        "2": {"qid-a": 1},
    }
    assert snapshot["covered_at_round"] == {}


def test_delayed_lower_round_updates_first_covered_round() -> None:
    dag = _coverage_node()
    dag.deliver_message("qid-a", "mid-a", b"A")
    for source in (0, 1, 2):
        _admit(dag, round_id=1, source=source, block=None)
    for source in (0, 1):
        _admit(
            dag,
            round_id=2,
            source=source,
            block={"qid-a": ["mid-a"]},
        )
    assert dag.qid_covered_at_round == {"qid-a": 2}

    for source in (3, 4):
        _admit(
            dag,
            round_id=1,
            source=source,
            block={"qid-a": ["mid-a"]},
        )

    assert dag.qid_covered_at_round == {"qid-a": 1}
    assert dag.snapshot_qid_causal_coverage()["coverage_records"] == [
        {"qid": "qid-a", "cover_round": 1, "witness_count": 2}
    ]


def test_duplicate_vertex_and_multiple_references_count_once() -> None:
    dag = _coverage_node()
    dag.deliver_message("qid-a", "mid-a", b"A")
    dag.deliver_message("qid-a", "mid-b", b"B")
    vertex = _admit(
        dag,
        round_id=1,
        source=0,
        block={"qid-a": ["mid-a", "mid-b"]},
    )
    dag.on_r_deliver(vertex)

    assert dag.snapshot_qid_causal_coverage()[
        "reference_counts_by_round"
    ] == {"1": {"qid-a": 1}}


def test_multi_qid_coverage_is_independent_and_snapshot_is_stable() -> None:
    dag = _coverage_node()
    for qid, mid in (("qid-z", "mid-z"), ("qid-a", "mid-a")):
        dag.deliver_message(qid, mid, qid.encode())
    block = {"qid-z": ["mid-z"], "qid-a": ["mid-a"]}
    _admit(dag, round_id=1, source=0, block=block)
    _admit(dag, round_id=1, source=1, block=block)

    snapshot = dag.snapshot_qid_causal_coverage()
    assert snapshot == dag.snapshot_qid_causal_coverage()
    assert list(snapshot["reference_counts_by_round"]["1"]) == [
        "qid-a",
        "qid-z",
    ]
    assert snapshot["covered_at_round"] == {"qid-a": 1, "qid-z": 1}


def test_causal_coverage_suppresses_only_later_round_selection() -> None:
    dag = _coverage_node()
    dag.deliver_message("qid-a", "mid-a", b"A")
    _admit(dag, round_id=1, source=0, block={"qid-a": ["mid-a"]})
    _admit(dag, round_id=1, source=1, block={"qid-a": ["mid-a"]})

    same_round = dag.create_new_vertex(1, source=2)
    later_round = dag.create_new_vertex(2, source=2)

    assert same_round.block == {"qid-a": ["mid-a"]}
    assert later_round.block is None
    registry_metrics = dag.wait_predicate.registry.metrics_snapshot()
    assert registry_metrics["select"]["invocations"] == 1
    metrics = dag.dag_selectblock_metrics_snapshot()
    assert metrics["evaluated_qids"] == 1
    assert metrics["skipped_causally_covered_qids"] == 1
    assert metrics["per_vertex_records"][1][
        "causal_coverage_exclusions"
    ] == ["qid-a"]


def test_uncovered_qid_is_still_evaluated_next_round() -> None:
    dag = _coverage_node()
    dag.deliver_message("qid-a", "mid-a", b"A")
    _admit(dag, round_id=1, source=0, block={"qid-a": ["mid-a"]})

    vertex = dag.create_new_vertex(2, source=1)

    assert vertex.block == {"qid-a": ["mid-a"]}
    assert dag.wait_predicate.registry.metrics_snapshot()["select"][
        "invocations"
    ] == 1


def test_partial_coverage_evaluates_only_uncovered_qids() -> None:
    dag = _coverage_node()
    dag.deliver_message("qid-a", "mid-a", b"A")
    dag.deliver_message("qid-b", "mid-b", b"B")
    _admit(dag, round_id=1, source=0, block={"qid-a": ["mid-a"]})
    _admit(dag, round_id=1, source=1, block={"qid-a": ["mid-a"]})

    vertex = dag.create_new_vertex(2, source=2)

    assert vertex.block == {"qid-b": ["mid-b"]}
    records = dag.wait_predicate.registry.metrics_snapshot()["select"][
        "records"
    ]
    assert [record["qid"] for record in records] == ["qid-b"]


def test_covered_qid_remains_resolvable_and_deliverable() -> None:
    delivered = []
    dag = DagTOB(
        my_id=0,
        M=4,
        quorum_h=3,
        agreement_t=1,
        deliver_cb=lambda qid, entries: delivered.append((qid, entries)),
    )
    dag.deliver_message("qid-a", "mid-a", b"A")
    first = _admit(
        dag,
        round_id=1,
        source=0,
        block={"qid-a": ["mid-a"]},
    )
    _admit(dag, round_id=1, source=1, block={"qid-a": ["mid-a"]})
    assert "qid-a" in dag.Pool

    dag.ordered_vertices.append(first.vid)
    dag._ordered_vertices_set.add(first.vid)
    _drive_ordered_delivery(dag)
    _drive_ordered_delivery(dag)

    assert len(delivered) == 1
    assert dag.delivery_count == 1
    assert "qid-a" in dag.DeliveredPool
    assert dag.wait_predicate.registry.metrics_snapshot()["verify"][
        "invocations"
    ] == 1


def test_remote_repeated_qid_vertex_is_not_rejected_after_coverage() -> None:
    dag = _coverage_node()
    dag.deliver_message("qid-a", "mid-a", b"A")
    _admit(dag, round_id=1, source=0, block={"qid-a": ["mid-a"]})
    _admit(dag, round_id=1, source=1, block={"qid-a": ["mid-a"]})
    _admit(dag, round_id=1, source=2, block=None)

    repeated = _admit(
        dag,
        round_id=2,
        source=3,
        block={"qid-a": ["mid-a"]},
    )

    assert repeated.vid in dag.DAG[2]
    assert dag.snapshot_qid_causal_coverage()[
        "reference_counts_by_round"
    ]["2"] == {"qid-a": 1}


@pytest.mark.parametrize(
    "kwargs, error_type, message",
    [
        (
            {"M": 4, "quorum_h": 3, "agreement_t": -1},
            ValueError,
            "non-negative",
        ),
        (
            {"M": 5, "quorum_h": 3, "agreement_t": 2},
            ValueError,
            "2 * agreement_t < quorum_h",
        ),
        (
            {"M": 3, "quorum_h": 3, "agreement_t": 1},
            ValueError,
            "must not exceed",
        ),
        (
            {"M": 4, "quorum_h": 3, "agreement_t": True},
            TypeError,
            "must be an integer",
        ),
    ],
)
def test_agreement_t_invalid_combinations_fail_closed(
    kwargs, error_type, message
) -> None:
    with pytest.raises(error_type, match=message.replace("*", r"\*")):
        DagTOB(my_id=0, **kwargs)


def test_recorded_vrf_committee_and_ideal_coin_fixture_is_view_independent() -> None:
    trace = RecordedCommitteeRoleTrace({1: (7, 8, 9)})
    oracle = IdealCoinLeaderOracle(RecordedIdealCoin({1: 4}), trace)
    left = DagTOB(
        my_id=0, M=10, quorum_h=1, agreement_t=0, leader_oracle=oracle
    )
    right = DagTOB(
        my_id=1, M=10, quorum_h=1, agreement_t=0, leader_oracle=oracle
    )

    selected = _make_vertex(round_id=1, source=8, strong_edges={(0, 0)})
    unrelated_left = _make_vertex(round_id=1, source=2, strong_edges={(0, 0)})
    unrelated_right = _make_vertex(round_id=1, source=5, strong_edges={(0, 0)})
    for dag, unrelated in ((left, unrelated_left), (right, unrelated_right)):
        _install_vertex(dag, selected)
        _install_vertex(dag, unrelated)

    assert oracle.get_wave_leader(left, 1) == (1, 8)
    assert oracle.get_wave_leader(right, 1) == (1, 8)

    right.DAG[1].pop((1, 8))
    assert oracle.get_wave_leader(right, 1) is None


def test_vertex_waits_for_missing_strong_parent_then_promotes() -> None:
    dag = DagTOB(my_id=0, M=3, quorum_h=2, agreement_t=0)
    vertex = _make_vertex(
        round_id=1,
        source=0,
        strong_edges={(0, 0), (0, 2)},
    )

    dag.on_r_deliver(vertex)
    assert vertex.vid in dag.buffer
    assert vertex.vid not in dag.DAG.get(1, {})

    missing_genesis = _make_vertex(round_id=0, source=2)
    _install_vertex(dag, missing_genesis)
    dag._try_promote_buffer()

    assert vertex.vid not in dag.buffer
    assert vertex.vid in dag.DAG[1]


def test_vertex_waits_for_message_then_message_delivery_promotes_it() -> None:
    dag = DagTOB(my_id=0, M=1, quorum_h=1, agreement_t=0)
    qid = Qid.build("accept", instance="availability").encode()
    vertex = _make_vertex(
        round_id=1,
        source=0,
        block={qid: ["mid-1"]},
        strong_edges=_parent_edges(dag, 0),
    )

    dag.on_r_deliver(vertex)
    assert vertex.vid in dag.buffer
    assert vertex.vid not in dag.DAG.get(1, {})

    dag.deliver_message(qid, "mid-1", b"payload-1")

    assert vertex.vid not in dag.buffer
    assert vertex.vid in dag.DAG[1]


def test_message_arrival_order_changes_only_admission_time() -> None:
    qid = Qid.build("accept", instance="arrival-order").encode()
    vertex = _make_vertex(
        round_id=1,
        source=0,
        block={qid: ["mid-1"]},
        strong_edges={(0, 0)},
    )
    message_first = DagTOB(my_id=0, M=1, quorum_h=1, agreement_t=0)
    vertex_first = DagTOB(my_id=0, M=1, quorum_h=1, agreement_t=0)

    message_first.deliver_message(qid, "mid-1", b"payload")
    message_first.on_r_deliver(vertex)

    vertex_first.on_r_deliver(vertex)
    assert vertex.vid in vertex_first.buffer
    vertex_first.deliver_message(qid, "mid-1", b"payload")

    assert vertex.vid in message_first.DAG[1]
    assert vertex.vid in vertex_first.DAG[1]
    assert message_first.DAG == vertex_first.DAG
    assert message_first.buffer == vertex_first.buffer == {}


def test_vertex_waits_for_missing_weak_parent_then_promotes() -> None:
    dag = DagTOB(my_id=0, M=2, quorum_h=1, agreement_t=0)
    strong_parent = _make_vertex(
        round_id=2,
        source=0,
        strong_edges={(1, 0)},
    )
    _install_vertex(dag, strong_parent)
    vertex = _make_vertex(
        round_id=3,
        source=0,
        strong_edges={strong_parent.vid},
        weak_edges={(1, 1)},
    )

    dag.on_r_deliver(vertex)
    assert vertex.vid in dag.buffer

    weak_parent = _make_vertex(
        round_id=1,
        source=1,
        strong_edges={(0, 0)},
    )
    _install_vertex(dag, weak_parent)
    dag._try_promote_buffer()

    assert vertex.vid not in dag.buffer
    assert vertex.vid in dag.DAG[3]


def test_vertex_wire_roundtrip_preserves_strong_and_weak_edges() -> None:
    vertex = _make_vertex(
        round_id=4,
        source=2,
        block={"qid": ["mid"]},
        strong_edges={(3, 0), (3, 1)},
        weak_edges={(1, 4), (2, 3)},
    )

    encoded = vertex.to_bytes()
    restored = Vertex.from_bytes(encoded)

    assert b'"strong_edges"' in encoded
    assert b'"weak_edges"' in encoded
    assert b'"height"' not in encoded
    assert restored == vertex

    legacy = vertex.to_dict()
    legacy["height"] = 7
    assert Vertex.from_dict(legacy) == vertex


def test_set_weak_edges_connects_only_unreachable_old_vertices() -> None:
    dag = DagTOB(my_id=0, M=2, quorum_h=1, agreement_t=0)
    reachable_old = _make_vertex(
        round_id=1,
        source=0,
        strong_edges={(0, 0)},
    )
    unreachable_old = _make_vertex(
        round_id=1,
        source=1,
        strong_edges={(0, 0)},
    )
    strong_parent = _make_vertex(
        round_id=2,
        source=0,
        strong_edges={reachable_old.vid},
    )
    for vertex in (reachable_old, unreachable_old, strong_parent):
        _install_vertex(dag, vertex)

    new_vertex = dag.create_new_vertex(3, source=0)

    assert new_vertex.strong_edges == frozenset({strong_parent.vid})
    assert new_vertex.weak_edges == frozenset({unreachable_old.vid})

    _install_vertex(dag, new_vertex)
    assert dag.path(new_vertex.vid, unreachable_old.vid)
    assert not dag.strong_path(new_vertex.vid, unreachable_old.vid)
    assert dag.strong_path(new_vertex.vid, reachable_old.vid)


def test_wave_support_does_not_count_weak_only_path_to_leader() -> None:
    trace = RecordedCommitteeRoleTrace({1: (0,)})
    oracle = IdealCoinLeaderOracle(RecordedIdealCoin({1: 0}), trace)
    dag = DagTOB(
        my_id=0, M=2, quorum_h=1, agreement_t=0, leader_oracle=oracle
    )

    leader = _make_vertex(round_id=1, source=0, strong_edges={(0, 0)})
    other_round_one = _make_vertex(round_id=1, source=1, strong_edges={(0, 0)})
    round_two = _make_vertex(
        round_id=2,
        source=1,
        strong_edges={other_round_one.vid},
        weak_edges={leader.vid},
    )
    round_three = _make_vertex(
        round_id=3,
        source=1,
        strong_edges={round_two.vid},
    )
    supporter = _make_vertex(
        round_id=4,
        source=1,
        strong_edges={round_three.vid},
    )
    for vertex in (leader, other_round_one, round_two, round_three, supporter):
        _install_vertex(dag, vertex)

    assert dag.path(supporter.vid, leader.vid)
    assert not dag.strong_path(supporter.vid, leader.vid)
    assert not dag.wave_ready(1)


def test_delivered_qid_does_not_make_an_unseen_mid_available() -> None:
    dag = DagTOB(my_id=0, M=1, quorum_h=1, agreement_t=0)
    qid = Qid.build("accept", instance="delivered-pool").encode()

    dag.DeliveredPool[qid] = {("old-mid", b"old-payload")}

    vertex = _make_vertex(
        round_id=1,
        source=0,
        block={qid: ["late-mid"]},
        strong_edges=_parent_edges(dag, 0),
    )
    dag.on_r_deliver(vertex)

    assert vertex.vid in dag.buffer
    assert vertex.vid not in dag.DAG.get(1, {})

    dag.deliver_message(qid, "late-mid", b"late-payload")
    assert vertex.vid not in dag.buffer
    assert vertex.vid in dag.DAG[1]
    assert ("late-mid", b"late-payload") in dag.DeliveredPool[qid]


def test_query_delivery_moves_all_active_messages_to_delivered_pool() -> None:
    events = []
    dag = _typed_node(deliver_cb=lambda *args: events.append(args), M=1, quorum_h=1)
    qid = Qid.build("accept", instance="move-not-drop").encode()
    dag.deliver_message(qid, "selected-mid", b"selected")
    dag.deliver_message(qid, "extra-mid", b"extra")
    vertex = _make_vertex(
        round_id=1,
        source=0,
        block={qid: ["selected-mid"]},
        strong_edges={(0, 0)},
    )
    _install_ordered_vertex(dag, vertex)

    _drive_ordered_delivery(dag)

    assert len(events) == 1
    assert qid not in dag.Pool
    assert dag.DeliveredPool[qid] == {
        ("selected-mid", b"selected"),
        ("extra-mid", b"extra"),
    }
    assert dag.ledger == [{qid: [("selected-mid", b"selected")]}]
    assert events == [(qid, [("selected-mid", b"selected")])]

    dag.deliver_message(qid, "late-mid", b"late")
    assert qid not in dag.Pool
    assert ("late-mid", b"late") in dag.DeliveredPool[qid]


def test_resolve_is_all_or_nothing_across_both_pools() -> None:
    dag = DagTOB(my_id=0, M=1, quorum_h=1, agreement_t=0)
    active_qid = Qid.build("accept", instance="active").encode()
    delivered_qid = Qid.build("accept", instance="delivered").encode()
    dag.deliver_message(active_qid, "active-mid", b"active")
    dag.DeliveredPool[delivered_qid] = {("old-mid", b"old")}
    block = {
        active_qid: ["active-mid"],
        delivered_qid: ["old-mid", "missing-mid"],
    }

    assert not dag._available(block)
    assert dag._resolve(block) is None

    dag.deliver_message(delivered_qid, "missing-mid", b"late")

    assert dag._available(block)
    assert dag._resolve(block) == {
        active_qid: [("active-mid", b"active")],
        delivered_qid: [("missing-mid", b"late"), ("old-mid", b"old")],
    }


def test_valid_qid_delivers_even_if_same_vertex_has_invalid_qid() -> None:
    events = []
    dag = _typed_node(deliver_cb=lambda *args: events.append(args), M=1, quorum_h=1)
    accept_qid = Qid.build("accept", instance="A").encode()
    reject_qid = Qid.build("reject", instance="B").encode()
    dag.deliver_message(accept_qid, "mid-a", b"payload-a")
    dag.deliver_message(reject_qid, "mid-b", b"payload-b")

    vertex = _make_vertex(
        round_id=1,
        source=0,
        block={accept_qid: ["mid-a"], reject_qid: ["mid-b"]},
        strong_edges=_parent_edges(dag, 0),
    )
    _install_ordered_vertex(dag, vertex)
    _drive_ordered_delivery(dag)

    assert len(events) == 1
    delivered_qid, delivered_entries = events[0]
    assert delivered_qid == accept_qid
    assert delivered_entries == [("mid-a", b"payload-a")]


def test_same_qid_is_delivered_at_most_once() -> None:
    events = []
    dag = _typed_node(deliver_cb=lambda *args: events.append(args), M=1, quorum_h=1)
    qid = Qid.build("accept", instance="once").encode()
    dag.deliver_message(qid, "mid-first", b"first")
    dag.deliver_message(qid, "mid-second", b"second")

    first = _make_vertex(
        round_id=1,
        source=0,
        block={qid: ["mid-first"]},
        strong_edges={(0, 0)},
    )
    second = _make_vertex(
        round_id=2,
        source=0,
        block={qid: ["mid-second"]},
        strong_edges={(1, 0)},
    )
    _install_ordered_vertex(dag, first)
    _install_ordered_vertex(dag, second)
    _drive_ordered_delivery(dag)

    assert len(events) == 1


def test_multi_qid_vertex_emits_independent_deterministic_events() -> None:
    events = []
    dag = _typed_node(deliver_cb=lambda *args: events.append(args), M=1, quorum_h=1)
    qid_b = Qid.build("accept", instance="B").encode()
    qid_a = Qid.build("accept", instance="A").encode()
    dag.deliver_message(qid_b, "mid-b", b"B")
    dag.deliver_message(qid_a, "mid-a", b"A")
    vertex = _make_vertex(
        round_id=1,
        source=0,
        block={qid_b: ["mid-b"], qid_a: ["mid-a"]},
        strong_edges={(0, 0)},
    )
    _install_ordered_vertex(dag, vertex)

    _drive_ordered_delivery(dag)
    _drive_ordered_delivery(dag)

    assert [qid for qid, _entries in events] == sorted((qid_a, qid_b))
    assert dag.delivery_sequence_number == 2
    assert len(dag.ledger) == 2
    assert all(len(event) == 1 for event in dag.ledger)
    assert not hasattr(dag, "next_height")


def test_delivery_sequences_are_prefix_comparable_for_ordered_prefixes() -> None:
    short_events = []
    long_events = []
    short = _typed_node(
        deliver_cb=lambda qid, entries: short_events.append((qid, entries)),
        M=1,
        quorum_h=1,
    )
    long = _typed_node(
        deliver_cb=lambda qid, entries: long_events.append((qid, entries)),
        M=1,
        quorum_h=1,
    )
    qid_a = Qid.build("accept", instance="prefix-A").encode()
    qid_b = Qid.build("accept", instance="prefix-B").encode()
    for dag in (short, long):
        dag.deliver_message(qid_a, "mid-a", b"A")
        dag.deliver_message(qid_b, "mid-b", b"B")

    first = _make_vertex(
        round_id=1,
        source=0,
        block={qid_a: ["mid-a"]},
        strong_edges={(0, 0)},
    )
    second = _make_vertex(
        round_id=2,
        source=0,
        block={qid_b: ["mid-b"]},
        strong_edges={(1, 0)},
    )
    _install_ordered_vertex(short, first)
    _install_ordered_vertex(long, first)
    _install_ordered_vertex(long, second)

    _drive_ordered_delivery(short)
    _drive_ordered_delivery(long)

    assert short_events == long_events[: len(short_events)]
    assert [qid for qid, _entries in long_events] == [qid_a, qid_b]


def test_dynamic_query_is_not_retroactively_added_to_older_vertex() -> None:
    dag = DagTOB(my_id=0, M=1, quorum_h=1, agreement_t=0)
    qid_a = Qid.build("accept", instance="A").encode()
    qid_b = Qid.build("accept", instance="B").encode()
    dag.deliver_message(qid_a, "mid-a", b"A")
    early = dag.create_new_vertex(1, source=0)

    dag.deliver_message(qid_b, "mid-b", b"B")
    later = dag.create_new_vertex(5, source=0)

    assert early.block is not None
    assert qid_a in early.block
    assert qid_b not in early.block
    assert later.block is not None
    assert qid_b in later.block


def test_one_wave_suppresses_qid_after_round1_causal_coverage() -> None:

    nodes = simulate_one_wave(
        M=4,
        quorum_h=3,
        agreement_t=1,
        epoch="qid-causal-coverage-phase0-baseline",
        sources=(0, 1, 2),
    )

    for node in nodes:
        round1 = list(node.DAG.get(1, {}).values())
        assert len(round1) == 3
        assert all(
            vertex.block is not None and "qid-demo" in vertex.block
            for vertex in round1
        )
        for round_id in range(2, 5):
            vertices = list(node.DAG.get(round_id, {}).values())
            assert len(vertices) == 3
            assert all(vertex.block is None for vertex in vertices)

    select_invocations = []
    verify_invocations = []
    for node in nodes:
        metrics = node.wait_predicate.registry.metrics_snapshot()
        select_invocations.append(metrics["select"]["invocations"])
        verify_invocations.append(metrics["verify"]["invocations"])

    assert select_invocations == [1, 1, 1, 0]
    assert sum(select_invocations) == 3
    assert verify_invocations == [1, 1, 1, 1]
    assert all(node.delivery_count == 1 for node in nodes)
    assert [
        node.dag_selectblock_metrics_snapshot()[
            "skipped_causally_covered_qids"
        ]
        for node in nodes
    ] == [3, 3, 3, 0]


def test_backward_leaders_are_processed_in_increasing_wave_order() -> None:
    trace = RecordedCommitteeRoleTrace({1: (0,), 2: (0,)})
    oracle = IdealCoinLeaderOracle(RecordedIdealCoin({1: 0, 2: 0}), trace)
    dag = DagTOB(
        my_id=0, M=1, quorum_h=1, agreement_t=0, leader_oracle=oracle
    )

    parent = (0, 0)
    for round_id in range(1, 9):
        vertex = _make_vertex(
            round_id=round_id,
            source=0,
            strong_edges={parent},
        )
        _install_vertex(dag, vertex)
        parent = vertex.vid

    processed = []

    def capture_stack_order() -> None:
        while dag.leaders_stack:
            processed.append(dag.leaders_stack.pop())

    dag.advance_ledger = capture_stack_order
    assert dag.wave_ready(2)
    assert processed == [(1, 0), (5, 0)]


def test_production_leader_choice_never_substitutes_local_candidate() -> None:
    """A shared coin may yield absence, but not two different local leaders."""

    oracle = DeterministicLeaderOracle("r0-local-view-gap")
    left = DagTOB(
        my_id=0, M=2, quorum_h=1, agreement_t=0, leader_oracle=oracle
    )
    right = DagTOB(
        my_id=1, M=2, quorum_h=1, agreement_t=0, leader_oracle=oracle
    )
    _install_vertex(left, _make_vertex(round_id=1, source=0, strong_edges={(0, 0)}))
    _install_vertex(right, _make_vertex(round_id=1, source=1, strong_edges={(0, 0)}))

    left_choice = oracle.get_wave_leader(left, 1)
    right_choice = oracle.get_wave_leader(right, 1)
    assert not (
        left_choice is not None
        and right_choice is not None
        and left_choice != right_choice
    )


def test_acs_leader_is_chosen_from_local_candidates_not_role_trace() -> None:
    coin = RecordedIdealCoin({1: 4})
    committees = RecordedCommitteeRoleTrace({1: (7, 8, 9)})
    dag_oracle = IdealCoinLeaderOracle(coin, committees)
    acs_coin = SharedIdealCoinOracle(coin)
    acs = LocalAsyncYosoACS(
        M=10,
        t=3,
        epoch="shared-r4-trace",
        coin_oracle=acs_coin,
        committee_trace=committees,
    )

    coin_bytes = asyncio.run(acs_coin.next_coin("ledger-1:round:1", 1))

    assert dag_oracle.selected_vid(1) == (1, 8)
    block_7 = CoreBlock.build(())
    block_9 = CoreBlock.build((ProposedBlock.build(9, ()),))
    assert acs._leader_from_coin({7: block_7}, coin_bytes, 1) == 7
    assert acs._leader_from_coin({9: block_9}, coin_bytes, 1) == 9
    assert acs._leader_from_coin(
        {7: block_7, 9: block_9}, coin_bytes, 1
    ) in {7, 9}


def test_vertex_requires_recorded_vrf_role_and_valid_proof() -> None:
    committees = RecordedCommitteeRoleTrace({1: (1,)})

    def verify_vertex(vertex: Vertex) -> bool:
        return (
            vertex.source in committees.roles(1)
            and vertex.proof == {"vrf_valid": True}
        )

    dag = DagTOB(
        my_id=0,
        M=3,
        quorum_h=1,
        agreement_t=0,
        vertex_verifier=verify_vertex,
    )
    invalid_role = _make_vertex(
        round_id=1,
        source=0,
        strong_edges={(0, 0)},
    )
    invalid_proof = Vertex(
        round=1,
        source=1,
        block=None,
        strong_edges=frozenset({(0, 0)}),
        weak_edges=frozenset(),
        proof={"vrf_valid": False},
    )
    valid = Vertex(
        round=1,
        source=1,
        block=None,
        strong_edges=frozenset({(0, 0)}),
        weak_edges=frozenset(),
        proof={"vrf_valid": True},
    )

    dag.on_r_deliver(invalid_role)
    dag.on_r_deliver(invalid_proof)
    assert invalid_role.vid not in dag.buffer
    assert invalid_proof.vid not in dag.buffer

    dag.on_r_deliver(valid)
    assert valid.vid in dag.DAG[1]


def test_recorded_protocol_trace_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        RecordedCommitteeRoleTrace({1: ()})
    with pytest.raises(ValueError, match="duplicate"):
        RecordedCommitteeRoleTrace({1: (2, 2)})
    with pytest.raises(ValueError, match="non-negative"):
        RecordedCommitteeRoleTrace({1: (-1,)})

    coin = RecordedIdealCoin({1: 7})
    committees = RecordedCommitteeRoleTrace({1: (0, 1)})
    with pytest.raises(KeyError, match="no index 2"):
        coin.get(2)
    with pytest.raises(KeyError, match="no index 2"):
        committees.roles(2)


def test_network_vrf_mode_requires_explicit_committee_trace() -> None:
    with pytest.raises(ValueError, match="requires an explicit"):
        NetworkAsyncYosoACS(
            my_id=0,
            M=4,
            t=1,
            epoch="r4-vrf-fail-closed",
            send=None,
            shared_subscribe=None,
            crypto=None,
            vrf_sk=None,
            vrf_pks=None,
            activation_mode="vrf",
        )
