"""S0 behavior baseline for the explicit legacy transcript-only DPSS path.

These tests intentionally freeze historical protocol behavior, not formal
Algorithm-4 naming: valid anchored PVSS transcripts are selected
deterministically and the same selected set reconstructs successor shares.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from beaver.chunk_elgamal import derive_public_key
from beaver.pvss import (
    PVSSTranscript,
    pvss_interpolate,
    pvss_verify,
    pvss_verify_anchor,
)
from beaver.legacy_yoso_dpss import (
    DPSSContext,
    DPSSJustifierManager,
    DPSSReconstructionManager,
    make_genesis_shares,
    make_resharing_transcript,
)


def _raw_g1(point: dict) -> str:
    return (
        b"\x04"
        + int(point["X"]).to_bytes(48, byteorder="big")
        + int(point["Y"]).to_bytes(48, byteorder="big")
    ).hex()


@pytest.fixture(scope="module")
def transcript_only_baseline():
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
    secret = 314159265
    secret_hat = 271828182
    genesis = make_genesis_shares(
        g_hex=g_hex,
        h_hex=h_hex,
        secret=secret,
        secret_hat=secret_hat,
        holder_ids=holder_ids,
        service_t=1,
    )
    context = DPSSContext(
        epoch=9,
        sid="service-key-a2-baseline",
        source_committee="committee-A",
        successor_committee="committee-B",
        transfer_seq=0,
        g_hex=g_hex,
        h_hex=h_hex,
        successor_rpk_list=successor_rpks,
        service_t=1,
        old_commitment_vector=genesis.old_commitment_vector,
        holder_eval_points=genesis.holder_eval_points,
        successor_role_indices=dict(zip(successor_ids, (0, 1))),
    )

    transcripts = {}
    entries = {}
    for holder_id in holder_ids:
        share_s, share_s_hat = genesis.share_for(holder_id)
        transcript = make_resharing_transcript(
            g_hex=g_hex,
            h_hex=h_hex,
            my_share_s=share_s,
            my_share_s_hat=share_s_hat,
            successor_rpk_list=successor_rpks,
            dealer_eval_point=genesis.holder_eval_points[holder_id],
            service_t=context.service_t,
            my_id=holder_id,
        )
        transcripts[holder_id] = transcript
        entries[holder_id] = (f"node_{holder_id}", transcript.to_bytes())

    return {
        "g_hex": g_hex,
        "h_hex": h_hex,
        "secret": secret,
        "secret_hat": secret_hat,
        "genesis": genesis,
        "context": context,
        "transcripts": transcripts,
        "entries": entries,
        "holder_ids": holder_ids,
        "successor_ids": successor_ids,
        "successor_rsks": successor_rsks,
    }


def test_transcript_only_justifier_selects_a_canonical_threshold_set(
    transcript_only_baseline,
) -> None:
    baseline = transcript_only_baseline
    context = baseline["context"]
    entries = baseline["entries"]
    manager = DPSSJustifierManager([context])

    selected = manager.select_justified_subset(
        context.build_qid(),
        [entries[12], entries[11], entries[10]],
    )

    assert selected is not None
    assert [mid for mid, _payload in selected] == ["node_10", "node_11"]
    selected_transcripts = [
        PVSSTranscript.from_bytes(payload) for _mid, payload in selected
    ]
    assert [item.dealer_eval_point for item in selected_transcripts] == [1, 2]
    assert manager.verify_qid_set(context.build_qid(), selected)


def test_transcript_only_justifier_skips_valid_pvss_with_wrong_anchor(
    transcript_only_baseline,
) -> None:
    baseline = transcript_only_baseline
    context = baseline["context"]
    entries = baseline["entries"]
    share_s, share_s_hat = baseline["genesis"].share_for(10)
    wrong_anchor = make_resharing_transcript(
        g_hex=baseline["g_hex"],
        h_hex=baseline["h_hex"],
        my_share_s=share_s + 1,
        my_share_s_hat=share_s_hat,
        successor_rpk_list=context.successor_rpk_list,
        dealer_eval_point=context.holder_eval_points[10],
        service_t=context.service_t,
        my_id=10,
    )

    assert pvss_verify(
        context.g_hex,
        context.h_hex,
        context.successor_rpk_list,
        wrong_anchor,
        context.service_t,
        key_base_hex=context.key_base,
    )
    assert not pvss_verify_anchor(
        wrong_anchor, context.old_commitment_vector
    )

    manager = DPSSJustifierManager([context])
    selected = manager.select_justified_subset(
        context.build_qid(),
        [
            ("node_09-wrong-anchor", wrong_anchor.to_bytes()),
            entries[12],
            entries[11],
            entries[10],
        ],
    )

    assert selected is not None
    assert [mid for mid, _payload in selected] == ["node_10", "node_11"]


def test_transcript_only_reconstruction_uses_the_selected_eval_points(
    transcript_only_baseline,
) -> None:
    baseline = transcript_only_baseline
    context = baseline["context"]
    entries = baseline["entries"]
    justifier = DPSSJustifierManager([context])
    selected = justifier.select_justified_subset(
        context.build_qid(),
        [entries[12], entries[10], entries[11]],
    )
    assert selected is not None

    successor_results = []
    for node_id, rsk in zip(
        baseline["successor_ids"], baseline["successor_rsks"]
    ):
        callbacks = []
        manager = DPSSReconstructionManager()
        manager.register_context(
            context,
            my_node_id=node_id,
            my_rsk=rsk,
            on_new_share=lambda share_s, share_s_hat, out=callbacks: out.append(
                (share_s, share_s_hat)
            ),
        )

        delivered = manager.on_deliver(
            context.encode_qid(), list(reversed(selected))
        )

        assert len(delivered) == 1
        result = delivered[0]
        assert result.dealer_eval_points == (1, 2)
        assert callbacks == [(result.share_s, result.share_s_hat)]
        assert manager.on_deliver(context.encode_qid(), selected) == []
        successor_results.append((result.share_s, result.share_s_hat))

    assert pvss_interpolate(
        [1, 2], successor_results, context.service_t
    ) == (baseline["secret"], baseline["secret_hat"])
