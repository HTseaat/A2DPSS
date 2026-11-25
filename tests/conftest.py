"""Shared fixtures for repository-level protocol contract tests."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from beaver.a2dpss import (
    A2DPSSContext,
    A2DPSSContribution,
    A2DPSSJustifierManager,
    make_a2dpss_contribution,
    validate_a2_contribution,
)
from beaver.async_yoso_tob import MessageRef, PendingEntry
from beaver.chunk_elgamal import derive_public_key
from beaver.handover_fixtures import make_genesis_shares
from beaver.pvss import PVSSTranscript, pvss_verify


def _raw_g1(point: dict) -> str:
    return (
        b"\x04"
        + int(point["X"]).to_bytes(48, byteorder="big")
        + int(point["Y"]).to_bytes(48, byteorder="big")
    ).hex()


@pytest.fixture(scope="session")
def a2dpss_acs_contract_setup():
    """Build one real A²DPSS context with three valid holder transcripts."""

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
    successor_rsks = (17, 29, 43)
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
    context = A2DPSSContext(
        epoch=20,
        sid="service-key-acs-contract",
        source_committee="committee-A",
        successor_committee="committee-B",
        transfer_seq=7,
        g_hex=g_hex,
        h_hex=h_hex,
        successor_rpk_list=successor_rpks,
        service_t=1,
        old_commitment_vector=genesis.old_commitment_vector,
        holder_eval_points=genesis.holder_eval_points,
        successor_role_indices={20: 0, 21: 1, 22: 2},
    )

    contributions = {}
    entries = {}
    pending = {}
    refs = {}
    for holder_id in holder_ids:
        share_s, share_s_hat = genesis.share_for(holder_id)
        contribution = make_a2dpss_contribution(
            context=context,
            holder_id=holder_id,
            share_s=share_s,
            share_s_hat=share_s_hat,
        )
        mid = f"node_{holder_id}"
        payload = contribution.to_bytes()
        ref = MessageRef(sender=holder_id, counter=0)
        contributions[holder_id] = contribution
        entries[holder_id] = (mid, payload)
        refs[holder_id] = ref
        pending[ref] = PendingEntry(
            ref=ref,
            mid=mid,
            payload=payload,
            received_at=0.0,
        )

    transcript_11 = contributions[11].transcript
    wrong_anchor_transcript = PVSSTranscript(
        dealer_id=12,
        dealer_eval_point=context.holder_eval_points[12],
        commitment_vector=list(transcript_11.commitment_vector),
        enc_results=list(transcript_11.enc_results),
        pi_ve=dict(transcript_11.pi_ve),
    )
    assert pvss_verify(
        context.g_hex,
        context.h_hex,
        context.successor_rpk_list,
        wrong_anchor_transcript,
        context.service_t,
        key_base_hex=context.key_base,
    )
    wrong_anchor = A2DPSSContribution(
        holder_id=12,
        holder_eval_point=context.holder_eval_points[12],
        context_digest=context.context_digest,
        transcript=wrong_anchor_transcript,
    )
    assert not validate_a2_contribution(context, wrong_anchor)

    transcript_12 = contributions[12].transcript
    tampered_pvss_transcript = PVSSTranscript(
        dealer_id=12,
        dealer_eval_point=context.holder_eval_points[12],
        commitment_vector=list(transcript_12.commitment_vector),
        enc_results=list(transcript_12.enc_results),
        pi_ve={},
    )
    tampered_pvss = A2DPSSContribution(
        holder_id=12,
        holder_eval_point=context.holder_eval_points[12],
        context_digest=context.context_digest,
        transcript=tampered_pvss_transcript,
    )
    assert not validate_a2_contribution(context, tampered_pvss)

    return {
        "context": context,
        "contributions": contributions,
        "entries": entries,
        "refs": refs,
        "pending": pending,
        "wrong_anchor_entry": ("node_12", wrong_anchor.to_bytes()),
        "tampered_pvss_entry": ("node_12", tampered_pvss.to_bytes()),
        "justifier": A2DPSSJustifierManager([context]),
    }
