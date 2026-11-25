"""S7 tests for consecutive operation-independent A²DPSS handovers."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from beaver.a2dpss import (
    A2DPSSContext,
    A2DPSSJustifierManager,
    A2DPSSPublicFinalizer,
    A2DPSSSuccessorReconstructionManager,
    make_a2dpss_contribution,
)
from beaver.a2dpss_lifecycle import (
    A2DPSSHandoverLifecycle,
    LOGICAL_ERASURE_LIMITATION,
    wipe_mutable_buffer,
)
from beaver.chunk_elgamal import derive_public_key
from beaver.g1_backend import G1Point
from beaver.pvss import pvss_interpolate, pvss_verify_anchor
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
    holder_ids = (10, 11, 12)
    successor_ids = (20, 21)
    successor_rsks = (17, 29)
    successor_rpks = tuple(
        derive_public_key(secret_key, g_hex)
        for secret_key in successor_rsks
    )
    secret = 135792468
    secret_hat = 864297531
    genesis = make_genesis_shares(
        g_hex=g_hex,
        h_hex=h_hex,
        secret=secret,
        secret_hat=secret_hat,
        holder_ids=holder_ids,
        service_t=1,
    )
    context = A2DPSSContext(
        epoch=12,
        sid="service-key-s7",
        source_committee="committee-A",
        successor_committee="committee-B",
        transfer_seq=0,
        g_hex=g_hex,
        h_hex=h_hex,
        successor_rpk_list=successor_rpks,
        service_t=1,
        old_commitment_vector=genesis.old_commitment_vector,
        holder_eval_points=genesis.holder_eval_points,
        successor_role_indices={20: 0, 21: 1},
    )
    entries = []
    for holder_id in holder_ids:
        share_s, share_s_hat = genesis.share_for(holder_id)
        contribution = make_a2dpss_contribution(
            context=context,
            holder_id=holder_id,
            share_s=share_s,
            share_s_hat=share_s_hat,
        )
        entries.append((f"node_{holder_id}", contribution.to_bytes()))
    justifier = A2DPSSJustifierManager([context])
    selected = justifier.select_justified_subset(
        context.build_qid(), list(reversed(entries))
    )
    assert selected is not None
    public_result = A2DPSSPublicFinalizer(justifier).on_deliver(
        context.encode_qid(), selected
    )
    assert public_result is not None
    return {
        "g_hex": g_hex,
        "h_hex": h_hex,
        "secret": secret,
        "secret_hat": secret_hat,
        "context": context,
        "selected": selected,
        "public_result": public_result,
        "successor_ids": successor_ids,
        "successor_rsks": successor_rsks,
    }


def _successor_lifecycle(setup, node_id, rsk):
    justifier = A2DPSSJustifierManager()
    reconstruction = A2DPSSSuccessorReconstructionManager()
    lifecycle = A2DPSSHandoverLifecycle(
        my_node_id=node_id,
        justifier=justifier,
        reconstruction=reconstruction,
    )
    lifecycle.register_initial(
        setup["context"],
        my_rsk=rsk,
    )
    result = reconstruction.reconstruct(
        setup["public_result"], setup["selected"]
    )
    assert result is not None
    return lifecycle, reconstruction, result


def test_two_hop_a2dpss_consumes_protocol_outputs_as_next_inputs(
    setup,
) -> None:
    next_successor_ids = (30, 31)
    next_successor_rsks = (41, 43)
    next_successor_rpks = tuple(
        derive_public_key(secret_key, setup["g_hex"])
        for secret_key in next_successor_rsks
    )
    advances = {}
    first_private = {}
    managers = {}
    for node_id, rsk in zip(
        setup["successor_ids"], setup["successor_rsks"]
    ):
        lifecycle, manager, result = _successor_lifecycle(
            setup, node_id, rsk
        )
        first_private[node_id] = result
        managers[node_id] = manager
        advances[node_id] = lifecycle.advance(
            setup["public_result"],
            next_successor_committee="committee-C",
            next_successor_rpk_list=next_successor_rpks,
            next_successor_role_indices={30: 0, 31: 1},
        )

    first_advance = advances[20]
    second_context = first_advance.next_context
    assert advances[21].next_context == second_context
    assert second_context.transfer_seq == 1
    assert second_context.source_committee == "committee-B"
    assert (
        second_context.old_commitment_vector
        == setup["public_result"].successor_commitment_vector
    )
    assert dict(second_context.holder_eval_points) == {20: 1, 21: 2}

    second_entries = []
    for holder_id in setup["successor_ids"]:
        holder_state = advances[holder_id].holder_state
        assert holder_state is not None
        share_s, share_s_hat = holder_state.consume()
        assert (share_s, share_s_hat) == (
            first_private[holder_id].share_s,
            first_private[holder_id].share_s_hat,
        )
        contribution = make_a2dpss_contribution(
            context=second_context,
            holder_id=holder_id,
            share_s=share_s,
            share_s_hat=share_s_hat,
        )
        assert pvss_verify_anchor(
            contribution.transcript,
            setup["public_result"].successor_commitment_vector,
        )
        assert not pvss_verify_anchor(
            contribution.transcript,
            setup["context"].old_commitment_vector,
        )
        second_entries.append(
            (f"node_{holder_id}", contribution.to_bytes())
        )
        with pytest.raises(RuntimeError, match="consumed|one-shot"):
            holder_state.consume()
        with pytest.raises(RuntimeError, match="already consumed"):
            managers[holder_id].consume_result_to_holder_state(
                setup["context"]
            )

    second_justifier = A2DPSSJustifierManager([second_context])
    second_selected = second_justifier.select_justified_subset(
        second_context.build_qid(), list(reversed(second_entries))
    )
    assert second_selected is not None
    second_public = A2DPSSPublicFinalizer(
        second_justifier
    ).on_deliver(second_context.encode_qid(), second_selected)
    assert second_public is not None
    assert G1Point.from_hex(
        second_public.successor_commitment_vector[0]
    ) == G1Point.from_hex(
        setup["public_result"].successor_commitment_vector[0]
    )

    final_results = []
    for node_id, rsk in zip(
        next_successor_ids, next_successor_rsks
    ):
        manager = A2DPSSSuccessorReconstructionManager()
        manager.register_context(
            second_context, my_node_id=node_id, my_rsk=rsk
        )
        result = manager.reconstruct(second_public, second_selected)
        assert result is not None
        final_results.append(result)
    recovered = pvss_interpolate(
        [1, 2],
        [
            (result.share_s, result.share_s_hat)
            for result in final_results
        ],
        second_context.service_t,
    )
    assert recovered == (setup["secret"], setup["secret_hat"])


def test_public_advance_is_independent_of_private_successor_membership(
    setup,
) -> None:
    observer_id = 99
    observer_rsk = 47
    observer_rpk = derive_public_key(observer_rsk, setup["g_hex"])
    other_rpk = derive_public_key(53, setup["g_hex"])
    justifier = A2DPSSJustifierManager()
    reconstruction = A2DPSSSuccessorReconstructionManager()
    lifecycle = A2DPSSHandoverLifecycle(
        my_node_id=observer_id,
        justifier=justifier,
        reconstruction=reconstruction,
    )
    lifecycle.register_initial(setup["context"])
    assert reconstruction.reconstruct(
        setup["public_result"], setup["selected"]
    ) is None

    advanced = lifecycle.advance(
        setup["public_result"],
        next_successor_committee="committee-observer",
        next_successor_rpk_list=(observer_rpk, other_rpk),
        next_successor_role_indices={observer_id: 0, 100: 1},
        my_next_rsk=observer_rsk,
    )
    assert advanced.holder_state is None
    assert lifecycle.active_context == advanced.next_context
    next_registration = reconstruction.registrations[
        advanced.next_context.context_digest
    ]
    assert next_registration["my_role_idx"] == 0
    assert next_registration["my_rsk"] == observer_rsk

    lifecycle.retire_active()
    assert lifecycle.snapshot()["active_qid"] is None
    assert justifier.contexts == {}
    assert reconstruction.registrations == {}


def test_best_effort_wipe_is_explicitly_logical_only() -> None:
    secret_buffer = bytearray(b"a2dpss-private-material")
    wipe_mutable_buffer(secret_buffer)
    assert secret_buffer == bytearray(len(secret_buffer))
    assert "cannot guarantee physical zeroization" in (
        LOGICAL_ERASURE_LIMITATION
    )
