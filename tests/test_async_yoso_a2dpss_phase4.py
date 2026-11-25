"""Phase 4 contracts for A²DPSS dissemination into Async-YOSO Pending."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path

import beaver.async_yoso_tob as async_tob
from beaver.a2dpss_acs_adapter import A2DPSSACSAdapter
from beaver.async_yoso_tob import (
    AsyncYosoMessageRBCManager,
    AsyncYosoTOB,
    BroadcastEnvelope,
)
from scripts.run_async_yoso_tob import (
    _a2dpss_pending_snapshot,
    _build_a2dpss_dissemination_state,
    _wait_for_a2dpss_candidate,
)


class _FakeRBC:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.output_queue = asyncio.Queue()
        self._closed = asyncio.Event()
        self.__class__.instances.append(self)

    async def reactor(self):
        await self._closed.wait()

    async def sender(self, _payload):
        return None


def _config():
    path = (
        Path(__file__).resolve().parents[1]
        / "conf/admpc_100_4_4/local.0.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_srs(config):
    decoded = json.loads(
        base64.b64decode(config["extra"]["SRS"]).decode("utf-8")
    )
    return {
        "Pk": json.dumps(decoded["Pk"]).encode("utf-8"),
        "Vk": json.dumps(decoded["Vk"]).encode("utf-8"),
    }


def test_message_rbc_uses_injected_dag_equivalent_parameters(
    monkeypatch,
) -> None:
    async def scenario():
        _FakeRBC.instances = []
        monkeypatch.setattr(async_tob, "YosoRBC", _FakeRBC)
        tob = AsyncYosoTOB(my_id=0, M=4, epoch="phase4-params")
        manager = AsyncYosoMessageRBCManager(
            tob=tob,
            epoch="phase4-params",
            my_id=0,
            M=4,
            t=1,
            send=None,
            shared_subscribe=None,
            crypto=(None, None, None, None, None),
            vrf_sk=None,
            vrf_pks=None,
            threshold_fraction=0.75,
            expected_committee_size=3,
            committee_m=3,
            quorum_h=3,
        )

        manager.ensure_instance(0, 0)
        rbc = _FakeRBC.instances[0]
        assert manager.parameters_snapshot() == {
            "threshold_fraction": 0.75,
            "expected_committee_size": 3,
            "committee_m": 3,
                "quorum_h": 3,
                "ready_amplify_threshold": 3,
                "require_vrf_proofs": False,
        }
        assert rbc.args[4] == 3
        assert rbc.kwargs["threshold_fraction"] == 0.75
        assert rbc.kwargs["expected_committee_size"] == 3
        assert rbc.kwargs["committee_m"] == 3
        assert rbc.kwargs["quorum_h"] == 3
        assert rbc.kwargs["ready_amplify_threshold"] == 3
        assert rbc.kwargs["require_vrf_proofs"] is False
        assert await manager.stop(timeout=1.0)

    asyncio.run(scenario())


def test_runner_projects_same_a2dpss_handover_state_as_dag() -> None:
    config = _config()
    common = {
        "extras": config["extra"],
        "srs": _runtime_srs(config),
        "M": 4,
        "service_t": 1,
    }

    holder = _build_a2dpss_dissemination_state(my_id=0, **common)
    non_holder = _build_a2dpss_dissemination_state(my_id=3, **common)

    assert holder["source_holder_ids"] == (0, 1, 2)
    assert holder["holder_state"] is not None
    assert non_holder["holder_state"] is None
    assert holder["context"] == non_holder["context"]
    assert holder["context"].service_t == 1
    assert holder["adapter"].qid_for_payload


def test_original_contribution_bytes_reach_structural_candidate_without_predicate(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    tob = AsyncYosoTOB(my_id=0, M=13, epoch="phase4-pending")
    adapter = A2DPSSACSAdapter(setup["justifier"])
    for holder_id in (10, 11):
        mid, payload = setup["entries"][holder_id]
        assert tob.schedule_message(
            BroadcastEnvelope(
                sender=holder_id,
                counter=0,
                mid=mid,
                payload=payload,
            )
        )
        assert tob.pending[setup["refs"][holder_id]].payload == payload

    before = setup["justifier"].validation_metrics_snapshot()
    candidate_refs = asyncio.run(
        _wait_for_a2dpss_candidate(
            tob=tob,
            adapter=adapter,
            start_run=time.time(),
            max_runtime=0.1,
        )
    )

    assert candidate_refs == [setup["refs"][10], setup["refs"][11]]
    assert tob.ledger == []
    after = setup["justifier"].validation_metrics_snapshot()
    for stage in ("schema_identity", "pvss", "anchor"):
        assert after[stage]["invocations"] == before[stage]["invocations"]


def test_pending_snapshot_binds_ref_mid_holder_qid_and_digest(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    tob = AsyncYosoTOB(my_id=0, M=13, epoch="phase4-snapshot")
    adapter = A2DPSSACSAdapter(setup["justifier"])
    holder_id = 10
    mid, payload = setup["entries"][holder_id]
    tob.schedule_message(
        BroadcastEnvelope(
            sender=holder_id,
            counter=0,
            mid=mid,
            payload=payload,
        )
    )

    before = setup["justifier"].validation_metrics_snapshot()
    records = _a2dpss_pending_snapshot(
        tob,
        adapter,
        setup["context"],
    )

    assert len(records) == 1
    assert records[0]["ref"] == setup["refs"][holder_id].encode()
    assert records[0]["mid"] == mid
    assert records[0]["holder_id"] == holder_id
    assert records[0]["qid"] == setup["context"].encode_qid()
    assert records[0]["structurally_valid"] is True
    assert records[0]["predicate_checked"] is False
    after = setup["justifier"].validation_metrics_snapshot()
    for stage in ("schema_identity", "pvss", "anchor"):
        assert after[stage]["invocations"] == before[stage]["invocations"]
