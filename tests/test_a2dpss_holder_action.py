"""S5 tests for the operation-independent one-shot holder action."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
from pathlib import Path

import pytest

from beaver.a2dpss import (
    A2DPSSContext,
    A2DPSSHolderState,
    make_a2dpss_contribution,
    run_a2dpss_handover_role,
    validate_a2_contribution,
)
from beaver.chunk_elgamal import derive_public_key
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
    holder_ids = [10, 11, 12]
    successor_ids = [20, 21, 22]
    successor_rpks = [
        derive_public_key(secret, g_hex)
        for secret in (17, 29, 43)
    ]
    genesis = make_genesis_shares(
        g_hex=g_hex,
        h_hex=h_hex,
        secret=123456789,
        secret_hat=987654321,
        holder_ids=holder_ids,
        service_t=1,
    )
    context = A2DPSSContext(
        epoch=7,
        sid="service-key-3",
        source_committee="committee-A",
        successor_committee="committee-B",
        transfer_seq=2,
        g_hex=g_hex,
        h_hex=h_hex,
        successor_rpk_list=successor_rpks,
        service_t=1,
        old_commitment_vector=genesis.old_commitment_vector,
        holder_eval_points=genesis.holder_eval_points,
        successor_role_indices={
            node_id: role_index
            for role_index, node_id in enumerate(successor_ids)
        },
    )
    return {"context": context, "genesis": genesis}


def test_builder_constructs_a_valid_operation_independent_contribution(
    setup,
) -> None:
    context = setup["context"]
    share_s, share_s_hat = setup["genesis"].share_for(10)
    contribution = make_a2dpss_contribution(
        context=context,
        holder_id=10,
        share_s=share_s,
        share_s_hat=share_s_hat,
    )

    assert contribution.context_digest == context.context_digest
    assert contribution.holder_eval_point == (
        context.holder_eval_points[10]
    )
    assert validate_a2_contribution(context, contribution)
    parameters = inspect.signature(
        run_a2dpss_handover_role
    ).parameters
    assert "operation_adapter" not in parameters
    source = inspect.getsource(run_a2dpss_handover_role)
    assert "operation_share" not in source
    assert "partial_eval" not in source


def test_holder_action_broadcasts_identical_bytes_once_and_consumes_state(
    setup,
) -> None:
    context = setup["context"]
    share_s, share_s_hat = setup["genesis"].share_for(10)
    holder_state = A2DPSSHolderState(share_s, share_s_hat)
    calls = []
    observed_payloads = []

    async def broadcast(qid, mid, payload):
        calls.append((qid, mid))
        observed_payloads.extend(bytes(payload) for _ in range(4))

    result = asyncio.run(
        run_a2dpss_handover_role(
            context=context,
            holder_id=10,
            share_s=share_s,
            share_s_hat=share_s_hat,
            broadcast_contribution=broadcast,
            holder_state=holder_state,
        )
    )

    assert calls == [(context.encode_qid(), "node_10")]
    assert observed_payloads == [result.contribution_bytes] * 4
    assert result.contribution_bytes == result.contribution.to_bytes()
    assert result.metrics.contribution_bytes == len(
        result.contribution_bytes
    )
    assert result.metrics.pvss_seconds >= 0
    assert result.metrics.serialization_seconds >= 0
    assert result.metrics.broadcast_seconds >= 0
    assert holder_state.consumed
    assert holder_state.share_s is None
    assert holder_state.share_s_hat is None

    with pytest.raises(RuntimeError, match="consumed|one-shot"):
        asyncio.run(
            run_a2dpss_handover_role(
                context=context,
                holder_id=10,
                share_s=share_s,
                share_s_hat=share_s_hat,
                broadcast_contribution=broadcast,
                holder_state=holder_state,
            )
        )
    assert len(calls) == 1


def test_concurrent_calls_cannot_broadcast_same_a2_role_twice(
    setup,
) -> None:
    context = setup["context"]
    share_s, share_s_hat = setup["genesis"].share_for(11)
    holder_state = A2DPSSHolderState(share_s, share_s_hat)
    payloads = []

    async def broadcast(_qid, _mid, payload):
        payloads.append(bytes(payload))
        await asyncio.sleep(0)

    async def scenario():
        calls = [
            run_a2dpss_handover_role(
                context=context,
                holder_id=11,
                share_s=share_s,
                share_s_hat=share_s_hat,
                broadcast_contribution=broadcast,
                holder_state=holder_state,
            )
            for _ in range(2)
        ]
        return await asyncio.gather(*calls, return_exceptions=True)

    results = asyncio.run(scenario())
    assert len(payloads) == 1
    assert sum(isinstance(item, RuntimeError) for item in results) == 1
    assert sum(not isinstance(item, Exception) for item in results) == 1


def test_broadcast_failure_consumes_a2_state_and_blocks_retry(
    setup,
) -> None:
    context = setup["context"]
    share_s, share_s_hat = setup["genesis"].share_for(12)
    holder_state = A2DPSSHolderState(share_s, share_s_hat)
    calls = 0

    async def failing_broadcast(_qid, _mid, _payload):
        nonlocal calls
        calls += 1
        raise RuntimeError("transport failed")

    arguments = dict(
        context=context,
        holder_id=12,
        share_s=share_s,
        share_s_hat=share_s_hat,
        broadcast_contribution=failing_broadcast,
        holder_state=holder_state,
    )
    with pytest.raises(RuntimeError, match="transport failed"):
        asyncio.run(run_a2dpss_handover_role(**arguments))
    assert holder_state.consumed

    with pytest.raises(RuntimeError, match="consumed|one-shot"):
        asyncio.run(run_a2dpss_handover_role(**arguments))
    assert calls == 1


def test_unknown_holder_is_rejected_before_consuming_a2_state(
    setup,
) -> None:
    share_s, share_s_hat = setup["genesis"].share_for(10)
    holder_state = A2DPSSHolderState(share_s, share_s_hat)

    with pytest.raises(ValueError, match="source committee"):
        asyncio.run(
            run_a2dpss_handover_role(
                context=setup["context"],
                holder_id=99,
                share_s=share_s,
                share_s_hat=share_s_hat,
                broadcast_contribution=lambda *_args: None,
                holder_state=holder_state,
            )
        )
    assert not holder_state.consumed
