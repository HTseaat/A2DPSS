"""S4 tests for fixed-public-T A²DPSS successor reconstruction."""

from __future__ import annotations

import base64
import dataclasses
import json
from pathlib import Path

import pytest

from beaver.a2dpss import (
    A2DPSSContext,
    A2DPSSContribution,
    A2DPSSJustifierManager,
    A2DPSSPublicFinalizer,
    A2DPSSPublicResult,
    A2DPSSSuccessorReconstructionManager,
    A2DPSSSuccessorResult,
    make_a2dpss_contribution,
)
from beaver.chunk_elgamal import derive_public_key
from beaver.g1_backend import G1Point
from beaver.pedersen_shamir import pedersen_commit
from beaver.pvss import pvss_interpolate
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
    srs = json.loads(base64.b64decode(config["extra"]["SRS"]).decode("utf-8"))
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
        sid="service-key-s4",
        source_committee="committee-A",
        successor_committee="committee-B",
        transfer_seq=6,
        g_hex=g_hex,
        h_hex=h_hex,
        successor_rpk_list=successor_rpks,
        service_t=1,
        old_commitment_vector=genesis.old_commitment_vector,
        holder_eval_points=genesis.holder_eval_points,
        successor_role_indices={20: 0, 21: 1},
    )
    contributions = {}
    entries = {}
    for holder_id in holder_ids:
        share_s, share_s_hat = genesis.share_for(holder_id)
        contribution = make_a2dpss_contribution(
            context=context,
            holder_id=holder_id,
            share_s=share_s,
            share_s_hat=share_s_hat,
        )
        contributions[holder_id] = contribution
        entries[holder_id] = (
            f"node_{holder_id}",
            contribution.to_bytes(),
        )
    justifier = A2DPSSJustifierManager([context])
    selected = justifier.select_justified_subset(
        context.build_qid(),
        [entries[12], entries[10], entries[11]],
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
        "entries": entries,
        "selected": selected,
        "public_result": public_result,
        "successor_ids": successor_ids,
        "successor_rsks": successor_rsks,
    }


@pytest.fixture(scope="module")
def successor_results(setup):
    results = {}
    for node_id, rsk in zip(
        setup["successor_ids"], setup["successor_rsks"]
    ):
        manager = A2DPSSSuccessorReconstructionManager()
        manager.register_context(
            setup["context"], my_node_id=node_id, my_rsk=rsk
        )
        result = manager.reconstruct(
            setup["public_result"], list(reversed(setup["selected"]))
        )
        assert result is not None
        results[node_id] = result
    return results


def test_successors_reconstruct_only_public_T_and_open_vc_entries(
    setup,
    successor_results,
) -> None:
    context = setup["context"]
    for node_id, result in successor_results.items():
        role_idx = context.successor_role_indices[node_id]
        assert result.my_role_idx == role_idx
        assert (
            result.accepted_eval_points
            == setup["public_result"].accepted_eval_points
        )
        assert (
            result.accepted_contribution_digests
            == setup["public_result"].accepted_contribution_digests
        )
        commitment = pedersen_commit(
            context.g_hex,
            context.h_hex,
            result.share_s,
            result.share_s_hat,
        )
        assert G1Point.from_hex(commitment) == G1Point.from_hex(
            setup["public_result"].successor_commitment_vector[
                role_idx + 1
            ]
        )
        assert A2DPSSSuccessorResult.from_bytes(
            result.to_bytes()
        ) == result

    recovered = pvss_interpolate(
        [1, 2],
        [
            (
                successor_results[20].share_s,
                successor_results[20].share_s_hat,
            ),
            (
                successor_results[21].share_s,
                successor_results[21].share_s_hat,
            ),
        ],
        context.service_t,
    )
    assert recovered == (setup["secret"], setup["secret_hat"])


def test_reconstruction_is_idempotent_and_non_successor_has_no_private_result(
    setup,
) -> None:
    callback_values = []
    manager = A2DPSSSuccessorReconstructionManager()

    def callback(share, blind):
        callback_values.append((share, blind))

    manager.register_context(
        setup["context"],
        my_node_id=20,
        my_rsk=setup["successor_rsks"][0],
        on_new_share=callback,
    )
    first = manager.reconstruct(
        setup["public_result"], setup["selected"]
    )
    second = manager.reconstruct(
        setup["public_result"], list(reversed(setup["selected"]))
    )

    assert first is second
    assert callback_values == [(first.share_s, first.share_s_hat)]
    metrics = manager.metrics_snapshot()
    assert metrics["commitment_recompute"]["invocations"] == 1
    assert metrics["pvss_decrypt"]["invocations"] == 1
    assert metrics["share_interpolation"]["invocations"] == 1
    assert metrics["commitment_opening_check"]["invocations"] == 1

    observer = A2DPSSSuccessorReconstructionManager()
    observer.register_context(setup["context"], my_node_id=99)
    assert observer.reconstruct(
        setup["public_result"], setup["selected"]
    ) is None


def test_reconstruction_rejects_alternative_T_reordered_digests_and_vc(
    setup,
) -> None:
    context = setup["context"]
    manager = A2DPSSSuccessorReconstructionManager()
    manager.register_context(
        context,
        my_node_id=20,
        my_rsk=setup["successor_rsks"][0],
    )

    with pytest.raises(ValueError, match="public result T|do not match"):
        manager.reconstruct(
            setup["public_result"],
            [setup["entries"][10], setup["entries"][12]],
        )

    reordered_digests = dataclasses.replace(
        setup["public_result"],
        accepted_contribution_digests=tuple(
            reversed(
                setup["public_result"].accepted_contribution_digests
            )
        ),
    )
    with pytest.raises(
        ValueError, match="public T|public result T|do not match"
    ):
        manager.reconstruct(reordered_digests, setup["selected"])

    tampered_vc = dataclasses.replace(
        setup["public_result"],
        successor_commitment_vector=context.old_commitment_vector,
    )
    with pytest.raises(ValueError, match="commitment vector"):
        manager.reconstruct(tampered_vc, setup["selected"])


def test_registration_rejects_wrong_successor_secret_key(setup) -> None:
    manager = A2DPSSSuccessorReconstructionManager()
    with pytest.raises(ValueError, match="does not match"):
        manager.register_context(
            setup["context"],
            my_node_id=20,
            my_rsk=setup["successor_rsks"][0] + 1,
        )
    with pytest.raises(ValueError, match="must not register"):
        manager.register_context(
            setup["context"],
            my_node_id=99,
            my_rsk=17,
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.update(version=99), "version"),
        (lambda value: value.pop("share_s"), "missing"),
        (lambda value: value.update(extra="unexpected"), "unknown"),
        (lambda value: value.update(context_digest="00" * 32), "qid|context"),
        (lambda value: value.update(share_s="01"), "canonical"),
        (
            lambda value: value.update(
                accepted_contribution_digests=["ab" * 32, "ab" * 32]
            ),
            "distinct",
        ),
    ],
)
def test_successor_result_decoder_is_strict(
    successor_results,
    mutation,
    match,
) -> None:
    value = successor_results[20].to_obj()
    mutation(value)
    with pytest.raises((TypeError, ValueError), match=match):
        A2DPSSSuccessorResult.from_bytes(
            json.dumps(value).encode("utf-8")
        )


def test_successor_result_rejects_duplicate_keys(successor_results) -> None:
    encoded = successor_results[20].to_bytes()
    duplicate = encoded.replace(
        b'"version":1', b'"version":1,"version":1', 1
    )
    with pytest.raises(ValueError, match="duplicate"):
        A2DPSSSuccessorResult.from_bytes(duplicate)
