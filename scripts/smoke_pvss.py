from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beaver.chunk_elgamal import derive_public_key
from beaver.g1_backend import G1Point
from beaver.pedersen_shamir import pedersen_commit
from beaver.pvss import (
    PVSSTranscript,
    pvss_deal,
    pvss_deliver,
    pvss_interpolate,
    pvss_verify,
    pvss_verify_anchor,
)


def _g1_raw(point: dict) -> str:
    return (
        b"\x04"
        + int(point["X"]).to_bytes(48, byteorder="big")
        + int(point["Y"]).to_bytes(48, byteorder="big")
    ).hex()


def _load_srs_generators(config_path: Path) -> tuple[str, str]:
    with config_path.open() as fp:
        cfg = json.load(fp)
    srs = json.loads(base64.b64decode(cfg["extra"]["SRS"]).decode("utf-8"))
    return _g1_raw(srs["Pk"]["G1_g"][0]), _g1_raw(srs["Pk"]["G1_h"][0])


def _assert_fails(label: str, fn) -> None:
    if fn():
        raise AssertionError(f"{label} unexpectedly passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test PVSS primitives.")
    parser.add_argument(
        "--config",
        default="conf/admpc_100_4_4/local.0.json",
        type=Path,
        help="ATCaaS config containing extra.SRS",
    )
    args = parser.parse_args()

    g_hex, h_hex = _load_srs_generators(args.config)
    t = 1
    secret = 123456789
    secret_hat = 987654321
    dealer_eval_point = 2
    rsks = [17, 29, 43]
    rpks = [derive_public_key(sk, g_hex) for sk in rsks]

    transcript = pvss_deal(
        g_hex,
        h_hex,
        rpks,
        secret,
        secret_hat,
        t,
        dealer_id=7,
        dealer_eval_point=dealer_eval_point,
    )
    if not pvss_verify(g_hex, h_hex, rpks, transcript, t):
        raise AssertionError("valid PVSS transcript failed verification")

    roundtrip = PVSSTranscript.from_bytes(transcript.to_bytes())
    if roundtrip != transcript:
        raise AssertionError("PVSS transcript serialization round-trip failed")

    old_commitments = [G1Point.identity().to_hex()] * (dealer_eval_point + 2)
    old_commitments[dealer_eval_point] = transcript.commitment_vector[0]
    if not pvss_verify_anchor(transcript, old_commitments):
        raise AssertionError("valid PVSS anchor failed")

    share_pairs = [
        pvss_deliver(g_hex, h_hex, transcript, sk, role_idx)
        for role_idx, sk in enumerate(rsks)
    ]
    for idx, pair in enumerate(share_pairs):
        commitment = pedersen_commit(g_hex, h_hex, pair[0], pair[1])
        if G1Point.from_hex(commitment) != G1Point.from_hex(transcript.commitment_vector[idx + 1]):
            raise AssertionError(f"delivered share commitment mismatch at role {idx}")

    recovered = pvss_interpolate([1, 2], share_pairs[:2], t)
    if recovered != (secret, secret_hat):
        raise AssertionError(f"PVSS interpolation mismatch: {recovered}")

    bad = PVSSTranscript.from_bytes(transcript.to_bytes())
    bad.commitment_vector[1] = G1Point.identity().to_hex()
    _assert_fails(
        "tampered commitment",
        lambda: pvss_verify(g_hex, h_hex, rpks, bad, t),
    )

    bad = PVSSTranscript.from_bytes(transcript.to_bytes())
    bad.enc_results[0]["enc_s"]["chunks"][0]["C2"] = G1Point.identity().to_hex()
    _assert_fails(
        "tampered enc_s",
        lambda: pvss_verify(g_hex, h_hex, rpks, bad, t),
    )

    bad = PVSSTranscript.from_bytes(transcript.to_bytes())
    bad.pi_ve["pi_share"]["responses"]["S"] += 1
    _assert_fails(
        "tampered pi_ve",
        lambda: pvss_verify(g_hex, h_hex, rpks, bad, t),
    )

    bad_anchor = list(old_commitments)
    bad_anchor[dealer_eval_point] = G1Point.identity().to_hex()
    _assert_fails(
        "tampered anchor",
        lambda: pvss_verify_anchor(transcript, bad_anchor),
    )

    print("PVSS smoke ok")


if __name__ == "__main__":
    main()
