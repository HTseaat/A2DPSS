"""S1 contracts for the canonical Algorithm-4 A²DPSS data model."""

from __future__ import annotations

import dataclasses
import json

import pytest

from beaver.a2dpss import (
    A2DPSS_CONTEXT_VERSION,
    A2DPSS_CONTRIBUTION_VERSION,
    A2DPSS_QID_KIND,
    A2DPSSContext,
    A2DPSSContribution,
    a2dpss_qid_matches_context,
    build_a2dpss_qid,
    canonical_a2dpss_context_digest,
    is_a2dpss_qid,
)
from beaver.pvss import PVSSTranscript


def _context(**overrides) -> A2DPSSContext:
    values = {
        "epoch": 7,
        "sid": "service-key-3",
        "source_committee": "committee-A",
        "successor_committee": "committee-B",
        "transfer_seq": 2,
        "g_hex": "generator-g",
        "h_hex": "generator-h",
        "key_base": "encryption-base",
        "successor_rpk_list": ["rpk-0", "rpk-1", "rpk-2"],
        "service_t": 1,
        "old_commitment_vector": ["vc-0", "vc-1", "vc-2", "vc-3"],
        "holder_eval_points": {10: 1, 11: 2, 12: 3},
        "successor_role_indices": {20: 0, 21: 1, 22: 2},
    }
    values.update(overrides)
    return A2DPSSContext(**values)


def _transcript(holder_id: int = 10, eval_point: int = 1) -> PVSSTranscript:
    return PVSSTranscript(
        dealer_id=holder_id,
        dealer_eval_point=eval_point,
        commitment_vector=["commitment-at-zero", "commitment-at-one"],
        enc_results=[{"ciphertext": "encrypted-share"}],
        pi_ve={"proof": "pvss-proof"},
    )


def _contribution(**overrides) -> A2DPSSContribution:
    values = {
        "version": A2DPSS_CONTRIBUTION_VERSION,
        "holder_id": 10,
        "holder_eval_point": 1,
        "context_digest": "ab" * 32,
        "transcript": _transcript(),
    }
    values.update(overrides)
    return A2DPSSContribution(**values)


def test_context_roundtrip_digest_and_qid_are_canonical() -> None:
    context = _context()
    restored = A2DPSSContext.from_bytes(context.to_bytes())

    assert A2DPSS_CONTEXT_VERSION == 1
    assert A2DPSS_QID_KIND == "a2dpss"
    assert restored == context
    assert restored.to_bytes() == context.to_bytes()
    assert restored.context_digest == canonical_a2dpss_context_digest(restored)
    assert len(restored.context_digest) == 64

    qid = build_a2dpss_qid(context)
    assert qid == context.build_qid()
    assert qid.encode() == context.encode_qid()
    assert is_a2dpss_qid(qid)
    assert is_a2dpss_qid(qid.encode())
    assert a2dpss_qid_matches_context(qid, context)
    fields = dict(qid.fields)
    assert fields["context_digest"] == context.context_digest
    assert fields["epoch"] == str(context.epoch)
    assert "operation_id" not in fields


def test_context_has_no_operation_coupling_fields() -> None:
    field_names = {field.name for field in dataclasses.fields(A2DPSSContext)}
    assert not {
        "operation_id",
        "operation_input",
        "operation_public_parameters",
        "operation_adapter",
    } & field_names


def test_context_digest_binds_every_transfer_security_boundary() -> None:
    base = _context()
    variants = [
        _context(old_commitment_vector=["vc-0", "vc-X", "vc-2", "vc-3"]),
        _context(successor_rpk_list=["rpk-0", "rpk-1", "rpk-X"]),
        _context(service_t=2),
        _context(source_committee="committee-X"),
        _context(successor_committee="committee-X"),
        _context(transfer_seq=3),
    ]

    assert len({base.context_digest, *(item.context_digest for item in variants)}) == 7
    assert all(item.encode_qid() != base.encode_qid() for item in variants)
    assert all(
        not a2dpss_qid_matches_context(base.build_qid(), item)
        for item in variants
    )


def test_context_digest_is_mapping_order_independent_and_type_aware() -> None:
    first = {
        "kind": "atcaas-a2dpss-context",
        "version": 1,
        "epoch": 7,
        "metadata": {"b": 2, "a": [1, "1"]},
    }
    reordered = {
        "metadata": {"a": [1, "1"], "b": 2},
        "epoch": 7,
        "version": 1,
        "kind": "atcaas-a2dpss-context",
    }
    as_text = dict(first, epoch="7")

    assert canonical_a2dpss_context_digest(
        first
    ) == canonical_a2dpss_context_digest(reordered)
    assert canonical_a2dpss_context_digest(
        first
    ) != canonical_a2dpss_context_digest(as_text)


def test_context_copies_mutable_inputs_before_digesting() -> None:
    rpks = ["rpk-0", "rpk-1", "rpk-2"]
    old_commitments = ["vc-0", "vc-1", "vc-2", "vc-3"]
    holder_points = {10: 1, 11: 2, 12: 3}
    successor_indices = {20: 0, 21: 1, 22: 2}
    context = _context(
        successor_rpk_list=rpks,
        old_commitment_vector=old_commitments,
        holder_eval_points=holder_points,
        successor_role_indices=successor_indices,
    )
    original_bytes = context.to_bytes()

    rpks[0] = "mutated-rpk"
    old_commitments[1] = "mutated-vc"
    holder_points[10] = 3
    successor_indices[20] = 2

    assert context.to_bytes() == original_bytes
    assert context.successor_rpk_list[0] == "rpk-0"
    assert context.old_commitment_vector[1] == "vc-1"
    assert context.holder_eval_points[10] == 1
    assert context.successor_role_indices[20] == 0


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.update(version=99), "version"),
        (lambda value: value.pop("old_commitment_vector"), "missing"),
        (lambda value: value.update(context_digest="00" * 32), "does not match"),
        (lambda value: value.update(extra="unexpected"), "unknown"),
        (lambda value: value.update(service_t=True), "integer"),
        (
            lambda value: value.update(holder_eval_points={"010": 1, "11": 2}),
            "canonical integer",
        ),
    ],
)
def test_context_decoder_rejects_invalid_schema(mutation, match) -> None:
    value = _context().to_obj()
    mutation(value)
    with pytest.raises((TypeError, ValueError), match=match):
        A2DPSSContext.from_bytes(json.dumps(value).encode("utf-8"))


def test_contribution_roundtrip_is_canonical_and_copies_transcript() -> None:
    source_transcript = _transcript()
    contribution = _contribution(transcript=source_transcript)
    restored = A2DPSSContribution.from_bytes(contribution.to_bytes())

    assert restored == contribution
    assert restored.to_bytes() == contribution.to_bytes()
    assert restored.transcript.dealer_id == restored.holder_id
    assert (
        restored.transcript.dealer_eval_point
        == restored.holder_eval_point
    )

    source_transcript.commitment_vector.append("mutated")
    source_transcript.enc_results.append({"mutated": True})
    source_transcript.pi_ve["mutated"] = True
    assert contribution == restored


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.update(version=99), "version"),
        (lambda value: value.pop("transcript"), "missing"),
        (lambda value: value.update(extra="unexpected"), "unknown"),
        (lambda value: value.update(context_digest="not-a-digest"), "context_digest"),
        (lambda value: value.update(holder_id=11), "holder_id"),
        (lambda value: value.update(holder_eval_point=2), "holder_eval_point"),
        (lambda value: value.update(holder_id=True), "integer"),
    ],
)
def test_contribution_decoder_rejects_invalid_schema(mutation, match) -> None:
    value = _contribution().to_obj()
    mutation(value)
    with pytest.raises((TypeError, ValueError), match=match):
        A2DPSSContribution.from_bytes(json.dumps(value).encode("utf-8"))


def test_contribution_decoder_rejects_raw_or_non_strict_transcript() -> None:
    with pytest.raises(ValueError, match="contribution"):
        A2DPSSContribution.from_bytes(_transcript().to_bytes())

    value = _contribution().to_obj()
    value["transcript"]["unexpected"] = True
    with pytest.raises(ValueError, match="transcript.*unknown"):
        A2DPSSContribution.from_bytes(json.dumps(value).encode("utf-8"))


def test_decoders_reject_duplicate_json_keys_and_non_finite_numbers() -> None:
    duplicate_contribution = (
        b'{"kind":"atcaas-a2dpss-contribution","kind":"other"}'
    )
    with pytest.raises(ValueError, match="duplicate"):
        A2DPSSContribution.from_bytes(duplicate_contribution)

    context_json = _context().to_bytes()
    duplicate_context = context_json.replace(
        b'"version":1', b'"version":1,"version":1', 1
    )
    with pytest.raises(ValueError, match="duplicate"):
        A2DPSSContext.from_bytes(duplicate_context)

    non_finite = context_json.replace(b'"epoch":7', b'"epoch":NaN', 1)
    with pytest.raises(ValueError, match="non-finite"):
        A2DPSSContext.from_bytes(non_finite)


def test_qid_matching_fails_closed() -> None:
    context = _context()
    other_context = _context(transfer_seq=context.transfer_seq + 1)

    assert not is_a2dpss_qid("not-json")
    assert not a2dpss_qid_matches_context("not-json", context)
    assert not a2dpss_qid_matches_context(
        other_context.build_qid(), context
    )
    assert not a2dpss_qid_matches_context(
        context.build_qid(), object()
    )
