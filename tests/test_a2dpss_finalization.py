"""S3 tests for fixed-T A²DPSS public finalization."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from beaver.a2dpss import (
    A2DPSSContext,
    A2DPSSContribution,
    A2DPSSJustifierManager,
    A2DPSSPublicFinalizer,
    A2DPSSPublicResult,
    a2dpss_contribution_digest,
    make_a2dpss_contribution,
)
from beaver.chunk_elgamal import derive_public_key
from beaver.dag_tob import Qid
from beaver.g1_backend import G1Point
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
    successor_rsks = (17, 29)
    successor_rpks = tuple(
        derive_public_key(secret_key, g_hex)
        for secret_key in successor_rsks
    )
    genesis = make_genesis_shares(
        g_hex=g_hex,
        h_hex=h_hex,
        secret=246813579,
        secret_hat=975318642,
        holder_ids=holder_ids,
        service_t=1,
    )
    context = A2DPSSContext(
        epoch=11,
        sid="service-key-s3",
        source_committee="committee-A",
        successor_committee="committee-B",
        transfer_seq=5,
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
        "context": context,
        "genesis": genesis,
        "contributions": contributions,
        "entries": entries,
        "justifier": justifier,
        "selected": selected,
        "public_result": public_result,
    }


def test_all_observers_derive_same_T_and_successor_commitments(setup) -> None:
    context = setup["context"]
    selected = setup["selected"]
    observers = [
        A2DPSSPublicFinalizer(setup["justifier"]) for _ in range(3)
    ]
    results = [
        observer.on_deliver(
            context.encode_qid(),
            selected if index % 2 == 0 else list(reversed(selected)),
        )
        for index, observer in enumerate(observers)
    ]

    assert all(result == results[0] for result in results)
    result = results[0]
    assert result is not None
    assert result.accepted_holder_ids == (10, 11)
    assert result.accepted_eval_points == (1, 2)
    assert len(result.accepted_contribution_digests) == 2
    assert len(result.successor_commitment_vector) == 3
    assert G1Point.from_hex(
        result.successor_commitment_vector[0]
    ) == G1Point.from_hex(context.old_commitment_vector[0])
    assert A2DPSSPublicResult.from_bytes(result.to_bytes()) == result


def test_public_finalizer_is_idempotent_but_rejects_conflicting_T(setup) -> None:
    finalizer = A2DPSSPublicFinalizer(setup["justifier"])
    first = finalizer.on_deliver(
        setup["context"].encode_qid(), setup["selected"]
    )
    second = finalizer.on_deliver(
        setup["context"].encode_qid(),
        list(reversed(setup["selected"])),
    )

    assert first is second
    with pytest.raises(ValueError, match="conflicting T"):
        finalizer.on_deliver(
            setup["context"].encode_qid(),
            [setup["entries"][10], setup["entries"][12]],
        )


def test_verified_delivery_skips_predicate_and_matches_safe_entrypoint(
    setup,
) -> None:
    context = setup["context"]
    selected = setup["selected"]
    justifier = A2DPSSJustifierManager([context])

    # Model the immediately preceding committed-order check in DagTOB.
    assert justifier.verify_qid_set(context.build_qid(), selected)
    before = justifier.validation_metrics_snapshot()

    trusted_result = A2DPSSPublicFinalizer(
        justifier
    ).on_verified_deliver(context.encode_qid(), selected)
    after = justifier.validation_metrics_snapshot()

    assert trusted_result is not None
    assert after == before
    safe_result = A2DPSSPublicFinalizer(
        A2DPSSJustifierManager([context])
    ).on_deliver(context.encode_qid(), selected)
    assert trusted_result == safe_result


def test_verified_delivery_still_rejects_structural_identity_mismatch(
    setup,
) -> None:
    finalizer = A2DPSSPublicFinalizer(setup["justifier"])
    _mid, payload = setup["selected"][0]
    malformed = [("node_999", payload), setup["selected"][1]]

    assert finalizer.on_verified_deliver(
        setup["context"].encode_qid(), malformed
    ) is None


def test_public_finalizer_rejects_non_threshold_or_unknown_delivery(setup) -> None:
    finalizer = A2DPSSPublicFinalizer(setup["justifier"])
    context = setup["context"]

    assert finalizer.on_deliver(
        context.encode_qid(),
        [
            setup["entries"][10],
            setup["entries"][11],
            setup["entries"][12],
        ],
    ) is None
    assert finalizer.on_deliver(
        context.encode_qid(),
        [setup["entries"][10], setup["entries"][10]],
    ) is None
    unknown = Qid.build(
        "a2dpss",
        context_digest="00" * 32,
    )
    assert finalizer.on_deliver(
        unknown, [setup["entries"][10], setup["entries"][11]]
    ) is None


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.update(version=99), "version"),
        (lambda value: value.pop("accepted_holder_ids"), "missing"),
        (lambda value: value.update(extra="unexpected"), "unknown"),
        (lambda value: value.update(context_digest="00" * 32), "qid|context"),
        (
            lambda value: value.update(
                accepted_contribution_digests=["ab" * 32, "ab" * 32]
            ),
            "distinct",
        ),
    ],
)
def test_public_result_decoder_is_strict(setup, mutation, match) -> None:
    value = setup["public_result"].to_obj()
    mutation(value)
    with pytest.raises((TypeError, ValueError), match=match):
        A2DPSSPublicResult.from_bytes(json.dumps(value).encode("utf-8"))


def test_public_result_rejects_duplicate_keys(setup) -> None:
    encoded = setup["public_result"].to_bytes()
    duplicate = encoded.replace(
        b'"version":1', b'"version":1,"version":1', 1
    )
    with pytest.raises(ValueError, match="duplicate"):
        A2DPSSPublicResult.from_bytes(duplicate)


def test_contribution_digest_is_canonical_and_domain_separated(setup) -> None:
    payload = setup["contributions"][10].to_bytes()
    digest = a2dpss_contribution_digest(payload)

    assert len(digest) == 64
    assert digest == a2dpss_contribution_digest(bytearray(payload))
    assert digest != setup["context"].context_digest
    with pytest.raises(TypeError, match="bytes"):
        a2dpss_contribution_digest("not-bytes")
