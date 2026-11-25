"""S6 tests for the formal Algorithm-4 A²DPSS YOSO-RBC transport."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import pytest

import beaver.a2dpss_rbc_manager as a2_rbc
from beaver.a2dpss import (
    A2DPSSContext,
    make_a2dpss_contribution,
)
from beaver.chunk_elgamal import derive_public_key
from beaver.g1_backend import G1Point, g1_from_hex
from beaver.ocdpss import OCDPSSContext, make_ocdpss_bundle
from beaver.ocdpss_rbc_manager import OCDPSSBundleRBCManager
from beaver.operation_share import ThresholdExponentiationOperation
from beaver.sigma import hash_to_scalar
from beaver.handover_fixtures import make_genesis_shares


def _raw_g1(point: dict) -> str:
    return (
        b"\x04"
        + int(point["X"]).to_bytes(48, byteorder="big")
        + int(point["Y"]).to_bytes(48, byteorder="big")
    ).hex()


@pytest.fixture(scope="module")
def setup():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "conf/admpc_100_4_4/local.0.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    srs = json.loads(
        base64.b64decode(config["extra"]["SRS"]).decode("utf-8")
    )
    g_hex = _raw_g1(srs["Pk"]["G1_g"][0])
    h_hex = _raw_g1(srs["Pk"]["G1_h"][0])
    genesis = make_genesis_shares(
        g_hex=g_hex,
        h_hex=h_hex,
        secret=12345,
        secret_hat=67890,
        holder_ids=[0, 1],
        service_t=1,
    )
    successor_rpks = [
        derive_public_key(rsk, g_hex)
        for rsk in (17, 29)
    ]
    common = {
        "epoch": 1,
        "sid": "s6-network",
        "source_committee": "A",
        "successor_committee": "B",
        "transfer_seq": 0,
        "g_hex": g_hex,
        "h_hex": h_hex,
        "successor_rpk_list": successor_rpks,
        "service_t": 1,
        "old_commitment_vector": genesis.old_commitment_vector,
        "holder_eval_points": genesis.holder_eval_points,
        "successor_role_indices": {0: 0, 1: 1},
    }
    a2_context = A2DPSSContext(**common)
    share_s, share_s_hat = genesis.share_for(0)
    contribution = make_a2dpss_contribution(
        context=a2_context,
        holder_id=0,
        share_s=share_s,
        share_s_hat=share_s_hat,
    )

    g = g1_from_hex(g_hex)

    def operation_base(value: bytes) -> G1Point:
        return g * hash_to_scalar(
            "s6-cross-payload-operation",
            value.hex(),
            nonzero=True,
        )

    adapter = ThresholdExponentiationOperation(
        operation_id="threshold-sign-v1",
        g_hex=g_hex,
        h_hex=h_hex,
        base_deriver=operation_base,
    )
    operation_input = b"s6-cross-payload"
    ocd_context = OCDPSSContext(
        **common,
        operation_id=adapter.operation_id,
        operation_input=operation_input,
        operation_public_parameters=adapter.public_context(
            operation_input
        ),
    )
    bundle = make_ocdpss_bundle(
        context=ocd_context,
        holder_id=0,
        share_s=share_s,
        share_s_hat=share_s_hat,
        operation_adapter=adapter,
    )
    return {
        "a2_context": a2_context,
        "contribution": contribution,
        "ocd_context": ocd_context,
        "bundle": bundle,
        "genesis": genesis,
    }


class _FakeDag:
    def __init__(self):
        self.delivered = []

    def deliver_message(self, qid, mid, payload):
        self.delivered.append((qid, mid, payload))


class _FakeRBC:
    instances = []

    def __init__(self, *_args, **_kwargs):
        self.output_queue = asyncio.Queue()
        self.sent = []
        self._closed = asyncio.Event()
        self.__class__.instances.append(self)

    async def reactor(self):
        await self._closed.wait()

    async def sender(self, payload):
        self.sent.append(payload)
        await self.output_queue.put((payload, None))


def _manager(monkeypatch, dag):
    _FakeRBC.instances = []
    monkeypatch.setattr(a2_rbc, "YosoRBC", _FakeRBC)
    return a2_rbc.A2DPSSContributionRBCManager(
        dag=dag,
        epoch="s6-test",
        my_id=0,
        M=4,
        t=1,
        threshold_fraction=0.75,
        expected_committee_size=3,
        committee_m=3,
        quorum_h=3,
        send=None,
        shared_subscribe=None,
        crypto=(None, None, None, None, None),
        vrf_sk=None,
        vrf_pks=None,
    )


def test_contribution_envelope_roundtrip_and_sizes(setup) -> None:
    context = setup["a2_context"]
    raw = setup["contribution"].to_bytes()
    payload = a2_rbc.A2DPSSContributionRBCManager._encode_payload(
        context,
        0,
        raw,
    )
    mid, restored, decoded = (
        a2_rbc.A2DPSSContributionRBCManager._decode_payload(
            payload,
            context=context,
            holder_id=0,
        )
    )

    assert mid == "node_0"
    assert restored == raw
    assert decoded == setup["contribution"]
    sizes = a2_rbc.classify_a2dpss_contribution(raw)
    assert sizes.contribution_bytes == len(raw)
    assert 0 < sizes.pvss_bytes < sizes.contribution_bytes


def test_a2_and_ocdpss_rbc_envelopes_fail_closed_when_crossed(
    setup,
) -> None:
    a2_context = setup["a2_context"]
    ocd_context = setup["ocd_context"]
    a2_payload = (
        a2_rbc.A2DPSSContributionRBCManager._encode_payload(
            a2_context,
            0,
            setup["contribution"].to_bytes(),
        )
    )
    ocd_payload = OCDPSSBundleRBCManager._encode_payload(
        ocd_context,
        0,
        setup["bundle"].to_bytes(),
    )

    with pytest.raises(ValueError, match="fields|kind|payload"):
        a2_rbc.A2DPSSContributionRBCManager._decode_payload(
            ocd_payload,
            context=a2_context,
            holder_id=0,
        )
    with pytest.raises(ValueError, match="fields|kind|payload"):
        OCDPSSBundleRBCManager._decode_payload(
            a2_payload,
            context=ocd_context,
            holder_id=0,
        )


def test_contribution_rbc_delivers_once_rejects_conflict_and_stops(
    setup,
    monkeypatch,
) -> None:
    context = setup["a2_context"]
    contribution = setup["contribution"]

    async def scenario():
        dag = _FakeDag()
        manager = _manager(monkeypatch, dag)
        manager.ensure_context(context, [0, 1])
        assert manager.active_count() == 2

        await manager.broadcast_contribution(
            context,
            holder_id=0,
            contribution_bytes=contribution.to_bytes(),
        )
        await manager.broadcast_contribution(
            context,
            holder_id=0,
            contribution_bytes=contribution.to_bytes(),
        )
        share_s, share_s_hat = setup["genesis"].share_for(0)
        conflicting = make_a2dpss_contribution(
            context=context,
            holder_id=0,
            share_s=share_s,
            share_s_hat=share_s_hat,
        )
        with pytest.raises(ValueError, match="conflicting"):
            await manager.broadcast_contribution(
                context,
                holder_id=0,
                contribution_bytes=conflicting.to_bytes(),
            )

        await asyncio.wait_for(
            manager.instances[
                (context.context_digest, 0)
            ].output_task,
            timeout=1.0,
        )
        assert dag.delivered == [
            (
                context.encode_qid(),
                "node_0",
                contribution.to_bytes(),
            )
        ]
        assert manager.metrics.outbound_contributions == 1
        assert manager.metrics.delivered_contributions == 1
        assert manager.metrics.outbound_pvss_bytes > 0
        assert await manager.stop(timeout=1.0)
        assert manager.active_count() == 0

    asyncio.run(scenario())


def test_invalid_a2_network_payload_never_enters_dag_pool(
    setup,
    monkeypatch,
) -> None:
    context = setup["a2_context"]

    async def scenario():
        dag = _FakeDag()
        manager = _manager(monkeypatch, dag)
        manager.ensure_instance(context, 0)
        payload = json.loads(
            manager._encode_payload(
                context,
                0,
                setup["contribution"].to_bytes(),
            ).decode("utf-8")
        )
        payload["version"] = 999
        instance = manager.instances[
            (context.context_digest, 0)
        ]
        await instance.rbc.output_queue.put(
            (json.dumps(payload).encode("utf-8"), None)
        )
        await asyncio.wait_for(
            instance.output_task,
            timeout=1.0,
        )

        assert dag.delivered == []
        assert manager.metrics.rejected_payloads == 1
        assert await manager.stop(timeout=1.0)

    asyncio.run(scenario())
