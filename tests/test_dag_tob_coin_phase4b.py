from __future__ import annotations

import asyncio

import pytest

from beaver.async_yoso_coin import CoinOracle, ObservedCoinOracle
from beaver.dag_tob import (
    DagTOB,
    ThresholdCoinLeaderOracle,
    Vertex,
)
from beaver.protocol_trace import RecordedCommitteeRoleTrace
from scripts.run_dag_tob_phase3 import _dag_coin_mode, _node_id_set


class _GatedCoin(CoinOracle):
    def __init__(self, output: bytes = b"\x00") -> None:
        self.output = output
        self.release = asyncio.Event()
        self.calls = []

    async def next_coin(self, session_id: str, attempt: int) -> bytes:
        self.calls.append((session_id, attempt))
        await self.release.wait()
        return self.output


class _ImmediateCoin(CoinOracle):
    def __init__(self, outputs) -> None:
        self.outputs = dict(outputs)
        self.calls = []

    async def next_coin(self, session_id: str, attempt: int) -> bytes:
        self.calls.append((session_id, attempt))
        return self.outputs[attempt]


def _vertex(round_id: int, parent, *, source: int = 0) -> Vertex:
    return Vertex(
        round=round_id,
        source=source,
        block=None,
        strong_edges=frozenset({parent}),
        weak_edges=frozenset(),
        proof=None,
    )


async def _spin_until(predicate, steps: int = 20) -> None:
    for _ in range(steps):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("async condition did not become true")


def test_threshold_leader_oracle_deduplicates_and_maps_public_roles() -> None:
    async def scenario():
        coin = _GatedCoin(b"\x05")
        observed = ObservedCoinOracle(coin, timeout=1.0)
        oracle = ThresholdCoinLeaderOracle(
            observed,
            RecordedCommitteeRoleTrace({1: (7, 8, 9)}),
        )
        left = asyncio.create_task(oracle.prepare_wave(1))
        right = asyncio.create_task(oracle.prepare_wave(1))
        await _spin_until(lambda: bool(coin.calls))
        assert coin.calls == [(ThresholdCoinLeaderOracle.SESSION_ID, 1)]

        coin.release.set()
        assert await left == (1, 9)
        assert await right == (1, 9)
        assert await oracle.prepare_wave(1) == (1, 9)
        assert len(coin.calls) == 1
        snapshot = oracle.snapshot()
        assert snapshot["ready_waves"] == [1]
        assert snapshot["waves"][0]["roles"] == [7, 8, 9]
        assert snapshot["waves"][0]["leader"] == {"round": 1, "source": 9}
        assert len(snapshot["waves"][0]["coin_digest"]) == 64
        assert (
            snapshot["waves"][0]["coin_digest"]
            == observed.snapshot()["observations"][0]["coin_digest"]
        )
        await oracle.close()

    asyncio.run(scenario())


def test_wave_waits_for_coin_while_later_round_is_admitted_then_resumes() -> None:
    async def scenario():
        coin = _GatedCoin()
        oracle = ThresholdCoinLeaderOracle(
            coin,
            RecordedCommitteeRoleTrace({1: (0,)}),
        )
        coordinator_tasks = {}
        requested = []
        dag = None

        async def prepare_and_resume(wave: int) -> None:
            await oracle.prepare_wave(wave)
            dag.notify_wave_coin_ready(wave)

        def request_coin(wave: int) -> None:
            requested.append(wave)
            coordinator_tasks[wave] = asyncio.create_task(
                prepare_and_resume(wave)
            )

        dag = DagTOB(
            my_id=0,
            M=1,
            quorum_h=1,
            agreement_t=0,
            leader_oracle=oracle,
            wave_coin_needed_cb=request_coin,
        )
        parent = (0, 0)
        for round_id in range(1, 5):
            current = _vertex(round_id, parent)
            dag.on_r_deliver(current)
            parent = current.vid

        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert requested == [1]
        assert coin.calls == [(ThresholdCoinLeaderOracle.SESSION_ID, 1)]
        assert dag.decided_wave == 0
        assert dag.snapshot_wave_coin_state() == {
            "requested_waves": [1],
            "fired_waves": [],
        }

        # Coin waiting gates only the wave decision.  DAG reception and later
        # round construction continue on the normal event loop.
        round_five = _vertex(5, parent)
        dag.on_r_deliver(round_five)
        assert round_five.vid in dag.DAG[5]
        assert dag.decided_wave == 0
        dag._fire_wave_if_needed(4)
        assert requested == [1]

        coin.release.set()
        await coordinator_tasks[1]
        assert dag.decided_wave == 1
        assert dag.snapshot_wave_coin_state() == {
            "requested_waves": [],
            "fired_waves": [1],
        }
        assert len(coin.calls) == 1
        await oracle.close()

    asyncio.run(scenario())


def test_coin_ready_before_dag_quorum_is_cached_until_round_four() -> None:
    async def scenario():
        coin = _ImmediateCoin({1: b"\x00"})
        oracle = ThresholdCoinLeaderOracle(
            coin,
            RecordedCommitteeRoleTrace({1: (0,)}),
        )
        await oracle.prepare_wave(1)
        dag = DagTOB(
            my_id=0,
            M=1,
            quorum_h=1,
            agreement_t=0,
            leader_oracle=oracle,
        )

        assert not dag.notify_wave_coin_ready(1)
        parent = (0, 0)
        for round_id in range(1, 5):
            current = _vertex(round_id, parent)
            dag.on_r_deliver(current)
            parent = current.vid

        assert dag.decided_wave == 1
        assert coin.calls == [(ThresholdCoinLeaderOracle.SESSION_ID, 1)]
        await oracle.close()

    asyncio.run(scenario())


def test_backward_leader_lookup_uses_prepared_wave_cache_only() -> None:
    async def scenario():
        coin = _ImmediateCoin({1: b"\x00", 2: b"\x00"})
        oracle = ThresholdCoinLeaderOracle(
            coin,
            RecordedCommitteeRoleTrace({1: (0,), 2: (0,)}),
        )
        await oracle.prepare_wave(1)
        await oracle.prepare_wave(2)
        dag = DagTOB(
            my_id=0,
            M=1,
            quorum_h=1,
            agreement_t=0,
            leader_oracle=oracle,
        )
        parent = (0, 0)
        for round_id in range(1, 9):
            current = _vertex(round_id, parent)
            dag.on_r_deliver(current)
            parent = current.vid

        assert dag.decided_wave == 2
        assert coin.calls == [
            (ThresholdCoinLeaderOracle.SESSION_ID, 1),
            (ThresholdCoinLeaderOracle.SESSION_ID, 2),
        ]
        assert oracle.get_wave_leader(dag, 1) == (1, 0)
        assert len(coin.calls) == 2
        await oracle.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("requested", "replay", "expected"),
    [
        ("ideal", False, ("ideal", False)),
        ("threshold-bls", False, ("threshold-bls", False)),
        ("threshold-bls", True, ("threshold-bls", False)),
    ],
)
def test_dag_coin_mode_resolution(requested, replay, expected) -> None:
    assert _dag_coin_mode(requested, replay_enabled=replay) == expected


def test_dag_coin_configuration_rejects_unknown_mode_and_bad_node() -> None:
    with pytest.raises(ValueError, match="DAG_TOB_COIN_MODE"):
        _dag_coin_mode("local-simulation", replay_enabled=False)
    assert _node_id_set("3,1,3", M=4, name="SILENT") == frozenset({1, 3})
    with pytest.raises(ValueError, match="out-of-range"):
        _node_id_set("4", M=4, name="SILENT")
