"""S2 contracts for A²DPSS validation and OCDPSS base-layer reuse."""

from __future__ import annotations

import base64
import itertools
import json
from pathlib import Path

import pytest

from beaver.a2dpss import (
    A2DPSSContext,
    A2DPSSContribution,
    A2DPSSJustifierManager,
    select_canonical_threshold,
    validate_a2_contribution,
)
from beaver.chunk_elgamal import derive_public_key
from beaver.dag_tob import Qid
from beaver.g1_backend import G1Point, g1_from_hex
from beaver.ocdpss import (
    OCDPSSBundle,
    OCDPSSContext,
    OCDPSSJustifierManager,
    make_ocdpss_bundle,
)
from beaver.operation_share import ThresholdExponentiationOperation
from beaver.pvss import PVSSTranscript, pvss_verify
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
    srs = json.loads(base64.b64decode(config["extra"]["SRS"]).decode("utf-8"))
    g_hex = _raw_g1(srs["Pk"]["G1_g"][0])
    h_hex = _raw_g1(srs["Pk"]["G1_h"][0])
    g = g1_from_hex(g_hex)
    operation_input = b"sign:epoch-10:s2"

    def derive_base(value: bytes) -> G1Point:
        return g * hash_to_scalar(
            "a2dpss-s2-operation-base", value.hex(), nonzero=True
        )

    adapter = ThresholdExponentiationOperation(
        operation_id="threshold-sign-v1",
        g_hex=g_hex,
        h_hex=h_hex,
        base_deriver=derive_base,
    )
    holder_ids = (10, 11, 12)
    successor_rsks = (17, 29, 43)
    successor_ids = (20, 21, 22)
    successor_rpks = tuple(
        derive_public_key(secret_key, g_hex)
        for secret_key in successor_rsks
    )
    genesis = make_genesis_shares(
        g_hex=g_hex,
        h_hex=h_hex,
        secret=1122334455,
        secret_hat=9988776655,
        holder_ids=holder_ids,
        service_t=1,
    )
    common_context = {
        "epoch": 10,
        "sid": "service-key-s2",
        "source_committee": "committee-A",
        "successor_committee": "committee-B",
        "transfer_seq": 4,
        "g_hex": g_hex,
        "h_hex": h_hex,
        "successor_rpk_list": successor_rpks,
        "service_t": 1,
        "old_commitment_vector": genesis.old_commitment_vector,
        "holder_eval_points": genesis.holder_eval_points,
        "successor_role_indices": {
            node_id: role_index
            for role_index, node_id in enumerate(successor_ids)
        },
    }
    a2_context = A2DPSSContext(**common_context)
    ocdpss_context = OCDPSSContext(
        **common_context,
        operation_id=adapter.operation_id,
        operation_input=operation_input,
        operation_public_parameters=adapter.public_context(operation_input),
    )

    a2_contributions = {}
    a2_entries = {}
    ocdpss_bundles = {}
    ocdpss_entries = {}
    for holder_id in holder_ids:
        share_s, share_s_hat = genesis.share_for(holder_id)
        bundle = make_ocdpss_bundle(
            context=ocdpss_context,
            holder_id=holder_id,
            share_s=share_s,
            share_s_hat=share_s_hat,
            operation_adapter=adapter,
        )
        contribution = A2DPSSContribution(
            holder_id=holder_id,
            holder_eval_point=genesis.holder_eval_points[holder_id],
            context_digest=a2_context.context_digest,
            transcript=bundle.transcript,
        )
        a2_contributions[holder_id] = contribution
        a2_entries[holder_id] = (
            f"node_{holder_id}",
            contribution.to_bytes(),
        )
        ocdpss_bundles[holder_id] = bundle
        ocdpss_entries[holder_id] = (
            f"node_{holder_id}",
            bundle.to_bytes(),
        )

    return {
        "adapter": adapter,
        "genesis": genesis,
        "a2_context": a2_context,
        "ocdpss_context": ocdpss_context,
        "a2_contributions": a2_contributions,
        "a2_entries": a2_entries,
        "ocdpss_bundles": ocdpss_bundles,
        "ocdpss_entries": ocdpss_entries,
        "a2_manager": A2DPSSJustifierManager([a2_context]),
        "ocdpss_manager": OCDPSSJustifierManager(
            [(ocdpss_context, adapter)]
        ),
    }


def test_a2dpss_selection_is_canonical_for_every_arrival_order(setup) -> None:
    context = setup["a2_context"]
    entries = [
        setup["a2_entries"][12],
        setup["a2_entries"][10],
        setup["a2_entries"][11],
    ]
    expected = [setup["a2_entries"][10], setup["a2_entries"][11]]

    for permutation in itertools.permutations(entries):
        assert setup["a2_manager"].select_justified_subset(
            context.build_qid(), list(permutation)
        ) == expected

    assert setup["a2_manager"].verify_qid_set(
        context.build_qid(), expected
    )
    assert setup["a2_manager"].verify_qid_set(
        context.build_qid(), list(reversed(expected))
    )
    assert not setup["a2_manager"].verify_qid_set(
        context.build_qid(), entries
    )
    assert not setup["a2_manager"].verify_qid_set(
        context.build_qid(), expected[:1]
    )


def test_common_selector_and_both_protocols_choose_the_same_base_T(setup) -> None:
    a2_context = setup["a2_context"]
    validated_items = [
        (
            setup["a2_entries"][holder_id],
            setup["a2_contributions"][holder_id],
        )
        for holder_id in (12, 10, 11)
    ]
    directly_selected = select_canonical_threshold(
        validated_items, a2_context.service_t + 1
    )
    a2_selected = setup["a2_manager"].select_justified_subset(
        a2_context.build_qid(),
        [setup["a2_entries"][12], setup["a2_entries"][10], setup["a2_entries"][11]],
    )
    ocdpss_selected = setup["ocdpss_manager"].select_justified_subset(
        setup["ocdpss_context"].build_qid(),
        [
            setup["ocdpss_entries"][12],
            setup["ocdpss_entries"][10],
            setup["ocdpss_entries"][11],
        ],
    )

    assert directly_selected == a2_selected
    assert [mid for mid, _payload in a2_selected] == ["node_10", "node_11"]
    assert [mid for mid, _payload in ocdpss_selected] == [
        "node_10",
        "node_11",
    ]


def test_honest_contribution_has_the_same_base_validity_in_both_protocols(
    setup,
) -> None:
    for holder_id in (10, 11, 12):
        contribution = setup["a2_contributions"][holder_id]
        a2_entry = setup["a2_entries"][holder_id]
        bundle = setup["ocdpss_bundles"][holder_id]

        assert validate_a2_contribution(
            setup["a2_context"], contribution
        )
        assert setup["a2_manager"].validate_contribution(
            setup["a2_context"], *a2_entry
        ) == contribution
        assert setup["ocdpss_manager"].validate_bundle(
            setup["ocdpss_context"],
            setup["adapter"],
            f"node_{holder_id}",
            bundle.to_bytes(),
        ) == bundle


def test_wrong_anchor_or_tampered_pvss_fails_both_protocols(setup) -> None:
    a2_context = setup["a2_context"]
    ocdpss_context = setup["ocdpss_context"]
    bundle_10 = setup["ocdpss_bundles"][10]
    transcript_11 = setup["ocdpss_bundles"][11].transcript
    wrong_anchor_transcript = PVSSTranscript(
        dealer_id=10,
        dealer_eval_point=a2_context.holder_eval_points[10],
        commitment_vector=list(transcript_11.commitment_vector),
        enc_results=list(transcript_11.enc_results),
        pi_ve=dict(transcript_11.pi_ve),
    )
    assert pvss_verify(
        a2_context.g_hex,
        a2_context.h_hex,
        a2_context.successor_rpk_list,
        wrong_anchor_transcript,
        a2_context.service_t,
        key_base_hex=a2_context.key_base,
    )
    wrong_anchor_a2 = A2DPSSContribution(
        holder_id=10,
        holder_eval_point=a2_context.holder_eval_points[10],
        context_digest=a2_context.context_digest,
        transcript=wrong_anchor_transcript,
    )
    wrong_anchor_ocdpss = OCDPSSBundle(
        holder_id=10,
        holder_eval_point=ocdpss_context.holder_eval_points[10],
        context_digest=ocdpss_context.context_digest,
        transcript=wrong_anchor_transcript,
        operation_share=bundle_10.operation_share,
        operation_proof=bundle_10.operation_proof,
    )

    assert not validate_a2_contribution(a2_context, wrong_anchor_a2)
    assert setup["a2_manager"].validate_contribution(
        a2_context, "node_10", wrong_anchor_a2.to_bytes()
    ) is None
    assert setup["ocdpss_manager"].validate_bundle(
        ocdpss_context,
        setup["adapter"],
        "node_10",
        wrong_anchor_ocdpss.to_bytes(),
    ) is None

    tampered_transcript = PVSSTranscript(
        dealer_id=10,
        dealer_eval_point=a2_context.holder_eval_points[10],
        commitment_vector=list(bundle_10.transcript.commitment_vector),
        enc_results=list(bundle_10.transcript.enc_results),
        pi_ve={},
    )
    tampered_a2 = A2DPSSContribution(
        holder_id=10,
        holder_eval_point=a2_context.holder_eval_points[10],
        context_digest=a2_context.context_digest,
        transcript=tampered_transcript,
    )
    assert not validate_a2_contribution(a2_context, tampered_a2)


def test_bad_operation_proof_fails_only_the_ocdpss_extension(setup) -> None:
    contribution = setup["a2_contributions"][10]
    bundle_obj = setup["ocdpss_bundles"][10].to_obj()
    bundle_obj["operation_proof"]["z1"] = str(
        int(bundle_obj["operation_proof"]["z1"]) + 1
    )
    bad_bundle = OCDPSSBundle.from_obj(bundle_obj)

    assert validate_a2_contribution(setup["a2_context"], contribution)
    assert setup["a2_manager"].validate_contribution(
        setup["a2_context"],
        "node_10",
        contribution.to_bytes(),
    ) == contribution
    assert setup["ocdpss_manager"].validate_bundle(
        setup["ocdpss_context"],
        setup["adapter"],
        "node_10",
        bad_bundle.to_bytes(),
    ) is None


def test_duplicate_holder_or_unknown_qid_never_reaches_threshold(setup) -> None:
    context = setup["a2_context"]
    duplicate = [setup["a2_entries"][10], setup["a2_entries"][10]]

    assert setup["a2_manager"].select_justified_subset(
        context.build_qid(), duplicate
    ) is None
    assert not setup["a2_manager"].verify_qid_set(
        context.build_qid(), duplicate
    )

    unknown_qid = Qid.build(
        "a2dpss",
        context_digest="00" * 32,
    )
    valid = [setup["a2_entries"][10], setup["a2_entries"][11]]
    assert setup["a2_manager"].select_justified_subset(
        unknown_qid, valid
    ) is None
    assert not setup["a2_manager"].verify_qid_set(unknown_qid, valid)

    transcript_10 = setup["a2_contributions"][10].transcript
    duplicate_eval_contribution = A2DPSSContribution(
        holder_id=99,
        holder_eval_point=1,
        context_digest=context.context_digest,
        transcript=PVSSTranscript(
            dealer_id=99,
            dealer_eval_point=1,
            commitment_vector=list(transcript_10.commitment_vector),
            enc_results=list(transcript_10.enc_results),
            pi_ve=dict(transcript_10.pi_ve),
        ),
    )
    assert select_canonical_threshold(
        [
            (
                setup["a2_entries"][10],
                setup["a2_contributions"][10],
            ),
            (
                ("node_99", duplicate_eval_contribution.to_bytes()),
                duplicate_eval_contribution,
            ),
        ],
        threshold=2,
    ) is None


def test_a2dpss_validation_cache_preserves_results(setup) -> None:
    manager = A2DPSSJustifierManager([setup["a2_context"]])
    entry = setup["a2_entries"][10]

    first = manager.validate_contribution(
        setup["a2_context"], *entry
    )
    second = manager.validate_contribution(
        setup["a2_context"], *entry
    )
    metrics = manager.validation_metrics_snapshot()

    assert first == second == setup["a2_contributions"][10]
    assert metrics["cache_hits"] == 1
    assert metrics["schema_identity"]["invocations"] == 1
    assert metrics["pvss"]["invocations"] == 1
    assert metrics["anchor"]["invocations"] == 1
