import argparse
import base64
import json
import os
import secrets
import hashlib
from ctypes import CDLL, c_char_p, c_void_p, string_at
from pathlib import Path
from typing import Tuple

from Crypto.Util.number import long_to_bytes

from scripts.run_key_gen_dyn import (
    acss_key_gen_with_srs,
    generate_serialized_keys,
    lib as kzg_lib,
)
from beaver.handover_fixtures import make_genesis_shares
from beaver.chunk_elgamal import generate_lookup_table_artifacts
from beaver.g1_backend import FIELD_ORDER
from beaver.atcaas_params import ATCaaSProtocolParameters
from beaver.yoso_re import generate_re_stage_config
from scripts.generate_coin_keys import generate_and_install_coin_keys


bulletproof_lib = CDLL("./libbulletproofs_amcl.so")
bulletproof_lib.pyComputeCommitmentGH.argtypes = [c_char_p]
bulletproof_lib.pyComputeCommitmentGH.restype = c_void_p
bulletproof_lib.pyFreeString.argtypes = [c_void_p]
bulletproof_lib.pyFreeString.restype = None


def _g1_uncompressed(point: dict) -> str:
    return (
        b"\x04"
        + long_to_bytes(int(point["X"]), 48)
        + long_to_bytes(int(point["Y"]), 48)
    ).hex()


def _srs_generators(srs_bytes: bytes) -> Tuple[str, str]:
    srs = json.loads(srs_bytes.decode("utf-8"))
    g = _g1_uncompressed(srs["Pk"]["G1_g"][0])
    h = _g1_uncompressed(srs["Pk"]["G1_h"][0])
    return g, h


def _point_hex_to_dict(point_hex: str) -> dict:
    x_hex, y_hex = point_hex.strip("()").split(",")
    return {"X": str(int(x_hex, 16)), "Y": str(int(y_hex, 16))}


def _vrf_public_key(sk_hex: str, g_hex: str, h_hex: str) -> dict:
    payload = {
        "g": g_hex,
        "h": h_hex,
        "m": [sk_hex],
        "m_prime": ["0"],
    }
    ptr = bulletproof_lib.pyComputeCommitmentGH(
        json.dumps(payload).encode("utf-8")
    )
    try:
        points = json.loads(string_at(ptr).decode("utf-8"))
    finally:
        bulletproof_lib.pyFreeString(ptr)
    if len(points) != 1:
        raise ValueError("VRF public-key generation returned an unexpected point list")
    return _point_hex_to_dict(points[0])


def _random_secret_scalar() -> int:
    """Return a non-zero canonical scalar accepted by A²DPSS/VRF code."""

    return secrets.randbelow(FIELD_ORDER - 1) + 1


def _canonical_role_public_key(point: dict) -> str:
    """Encode one BLS12-381 G1 point in the parser's fixed-width form."""

    return f"({int(point['X']):096x},{int(point['Y']):096x})"


def generate_vrf_keys(M: int, srs_bytes: bytes):
    g_hex, h_hex = _srs_generators(srs_bytes)
    vrf_sks = []
    vrf_pks = []
    for _ in range(M):
        sk = _random_secret_scalar()
        sk_hex = format(sk, "x")
        vrf_sks.append(str(sk))
        vrf_pks.append(_vrf_public_key(sk_hex, g_hex, h_hex))
    return vrf_sks, vrf_pks


def generate_dpss_role_keys(M: int, srs_bytes: bytes):
    g_hex, h_hex = _srs_generators(srs_bytes)
    role_keys = []
    for _ in range(M):
        sk = _random_secret_scalar()
        sk_hex = format(sk, "x")
        point = _vrf_public_key(sk_hex, g_hex, h_hex)
        role_keys.append(
            {
                "rsk": str(sk),
                "rpk": _canonical_role_public_key(point),
            }
        )
    return role_keys


def generate_dpss_node_configs(
    *,
    M: int,
    n_r: int,
    t_r: int,
    srs_bytes: bytes,
    role_keys,
    source_committee=None,
    successor_committee=None,
):
    g_hex, h_hex = _srs_generators(srs_bytes)
    source_committee = list(
        range(n_r) if source_committee is None else source_committee
    )
    successor_committee = list(
        range(n_r) if successor_committee is None else successor_committee
    )
    for name, committee in (
        ("source_committee", source_committee),
        ("successor_committee", successor_committee),
    ):
        if len(committee) != n_r or len(set(committee)) != n_r:
            raise ValueError(f"{name} must contain n_r distinct nodes")
        if any(node_id < 0 or node_id >= M for node_id in committee):
            raise ValueError(f"{name} contains a node outside [0, M)")
    holder_eval_points = {
        str(node_id): idx + 1
        for idx, node_id in enumerate(source_committee)
    }
    secret = 42
    secret_hat = 7
    phi_coeffs = [secret] + [1000 + idx for idx in range(t_r)]
    phi_hat_coeffs = [secret_hat] + [2000 + idx for idx in range(t_r)]
    genesis = make_genesis_shares(
        g_hex=g_hex,
        h_hex=h_hex,
        secret=secret,
        secret_hat=secret_hat,
        holder_ids=source_committee,
        service_t=t_r,
        holder_eval_points={
            int(node_id): int(point)
            for node_id, point in holder_eval_points.items()
        },
        phi_coeffs=phi_coeffs,
        phi_hat_coeffs=phi_hat_coeffs,
    )
    successor_rpk_list = [
        role_keys[node_id]["rpk"] for node_id in successor_committee
    ]
    public = {
        "version": 2,
        "private_material_scope": "node-local",
        "epoch": 0,
        "sid": "a2dpss-phase6",
        "source_committee_id": "source",
        "successor_committee_id": "successor",
        "transfer_seq": 0,
        "n_r": n_r,
        "t_r": t_r,
        "service_t": t_r,
        "source_committee": source_committee,
        "successor_committee": successor_committee,
        "holder_eval_points": holder_eval_points,
        "successor_role_indices": {
            str(node_id): idx
            for idx, node_id in enumerate(successor_committee)
        },
        "successor_rpk_list": successor_rpk_list,
        "successor_role_key_digest": hashlib.sha256(
            json.dumps(
                {
                    "epoch": 0,
                    "transfer_seq": 0,
                    "successor_role_indices": {
                        str(node_id): idx
                        for idx, node_id in enumerate(successor_committee)
                    },
                    "successor_rpk_list": successor_rpk_list,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "old_commitment_vector": list(genesis.old_commitment_vector),
    }
    by_node = {}
    for node_id in range(M):
        local = dict(public)
        if node_id in genesis.holder_shares:
            share = genesis.holder_shares[node_id]
            local["my_holder_share"] = {
                "node_id": node_id,
                "s": str(share[0]),
                "s_hat": str(share[1]),
                "eval_point": holder_eval_points[str(node_id)],
            }
        if node_id in successor_committee:
            local["my_successor_role"] = {
                "node_id": node_id,
                "role_idx": successor_committee.index(node_id),
                "rsk": role_keys[node_id]["rsk"],
                "epoch": 0,
                "transfer_seq": 0,
            }
        by_node[node_id] = local
    return by_node


def _resolve_parameters(args) -> ATCaaSProtocolParameters:
    legacy_n = args.n
    legacy_t = args.t
    n_r = args.n_r if args.n_r is not None else legacy_n
    t_r = args.t_r if args.t_r is not None else legacy_t
    n_a = (
        args.n_a
        if args.n_a is not None
        else args.expected_committee_size
        if args.expected_committee_size is not None
        else legacy_n
    )
    t_a = args.t_a if args.t_a is not None else legacy_t
    missing = [
        name
        for name, value in (
            ("n_r/--n", n_r),
            ("t_r/--t", t_r),
            ("n_a/--n", n_a),
            ("t_a/--t", t_a),
        )
        if value is None
    ]
    if missing:
        raise ValueError("missing Phase-6 parameters: " + ", ".join(missing))
    m_a = (
        args.m_a
        if args.m_a is not None
        else args.committee_m
        if args.committee_m is not None
        else n_a
    )
    quorum_h = args.quorum_h if args.quorum_h is not None else n_a - t_a
    coin_c = args.coin_c if args.coin_c is not None else args.M
    coin_k = args.coin_k if args.coin_k is not None else coin_c - t_a
    if args.coin_t is None:
        raise ValueError("missing Phase-6 parameter: coin_t/--coin-t")
    return ATCaaSProtocolParameters(
        M=args.M,
        n_r=n_r,
        t_r=t_r,
        n_a=n_a,
        t_a=t_a,
        m_a=m_a,
        quorum_h=quorum_h,
        coin_c=coin_c,
        coin_t=args.coin_t,
        coin_k=coin_k,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate ATCaaS long-lived node configs."
    )
    parser.add_argument("--M", required=True, type=int, help="long-lived node count")
    parser.add_argument("--n", type=int, help="legacy alias for n_r=n_a")
    parser.add_argument("--t", type=int, help="legacy alias for t_r=t_a")
    parser.add_argument("--n-r", type=int, help="DPSS committee size")
    parser.add_argument("--t-r", type=int, help="DPSS corruption threshold")
    parser.add_argument("--n-a", type=int, help="expected agreement committee size")
    parser.add_argument("--t-a", type=int, help="agreement corrupted-role bound")
    parser.add_argument("--m-a", type=int, help="agreement committee size upper bound")
    parser.add_argument(
        "--expected-committee-size",
        type=int,
        default=None,
        help="deprecated alias for --n-a",
    )
    parser.add_argument(
        "--committee-m",
        type=int,
        default=None,
        help="deprecated alias for --m-a",
    )
    parser.add_argument(
        "--quorum-h",
        type=int,
        default=None,
        help="paper agreement quorum; must equal n_a-t_a",
    )
    parser.add_argument("--coin-c", type=int, help="common-coin committee size")
    parser.add_argument(
        "--coin-t",
        type=int,
        help="maximum Byzantine members in the common-coin committee",
    )
    parser.add_argument("--coin-k", type=int, help="common-coin BLS threshold")
    parser.add_argument(
        "--coin-committee",
        nargs="+",
        type=int,
        help="explicit coin committee node ids; defaults to range(coin_c)",
    )
    parser.add_argument("--source-committee", nargs="+", type=int)
    parser.add_argument("--successor-committee", nargs="+", type=int)
    parser.add_argument("--port", default=10000, type=int, help="base local port")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output config directory; defaults to conf/atcaas_<M>_<n>",
    )
    args = parser.parse_args()

    params = _resolve_parameters(args)
    coin_committee = (
        list(range(params.coin_c))
        if args.coin_committee is None
        else list(args.coin_committee)
    )
    if len(coin_committee) != params.coin_c:
        raise ValueError("coin committee must contain exactly coin_c nodes")

    crypto_t = max(params.t_r, params.agreement_t_impl, 1)
    srs_bytes = kzg_lib.pyNewSRS(crypto_t)
    srs_encoded = base64.b64encode(srs_bytes).decode("utf-8")

    tbls_public_key, tbls_private_keys = generate_serialized_keys(
        args.M, params.agreement_t_impl
    )
    acss_public_keys, acss_private_keys = acss_key_gen_with_srs(srs_bytes, args.M)
    vrf_private_keys, vrf_public_keys = generate_vrf_keys(args.M, srs_bytes)
    dpss_role_keys = generate_dpss_role_keys(args.M, srs_bytes)
    dpss_by_node = generate_dpss_node_configs(
        M=args.M,
        n_r=params.n_r,
        t_r=params.t_r,
        srs_bytes=srs_bytes,
        role_keys=dpss_role_keys,
        source_committee=args.source_committee,
        successor_committee=args.successor_committee,
    )
    re_stage = generate_re_stage_config(M=args.M, service_t=params.t_r)

    out_dir = args.out_dir or f"conf/atcaas_{args.M}_{params.n_a}"
    os.makedirs(out_dir, exist_ok=True)
    g_hex, h_hex = _srs_generators(srs_bytes)
    chunk_lookup_tables = generate_lookup_table_artifacts(
        Path(out_dir) / "lookup_tables",
        g_hex=g_hex,
        h_hex=h_hex,
    )

    peers = [f"localhost:{args.port + i}" for i in range(args.M)]
    for node_id in range(args.M):
        data = {
            "N": args.M,
            "M": args.M,
            "n": params.n_a,
            "expected_committee_size": params.n_a,
            "committee_m": params.m_a,
            "quorum_h": params.quorum_h,
            "t": params.agreement_t_impl,
            "n_r": params.n_r,
            "t_r": params.t_r,
            "n_a": params.n_a,
            "t_a": params.t_a,
            "m_a": params.m_a,
            "coin_c": params.coin_c,
            "coin_t": params.coin_t,
            "coin_k": params.coin_k,
            "atcaas_parameters": params.to_config_obj(),
            "my_id": node_id,
            "peers": peers,
            "extra": {
                "k": 4,
                "run_id": "atcaas-local",
                "public_key": tbls_public_key,
                "private_key": tbls_private_keys[node_id],
                "SRS": srs_encoded,
                "pks_acss": acss_public_keys[node_id],
                "sk_acss": acss_private_keys[node_id],
                "vrf_sk": vrf_private_keys[node_id],
                "vrf_pks": vrf_public_keys,
                "vrf_pk": vrf_public_keys[node_id],
                "dpss": dpss_by_node[node_id],
                "re_stage": re_stage,
                "chunk_lookup_tables": chunk_lookup_tables,
            },
        }
        with open(os.path.join(out_dir, f"local.{node_id}.json"), "w") as fp:
            json.dump(data, fp, indent=4)

    generate_and_install_coin_keys(
        config_dir=os.path.abspath(out_dir),
        network_size=params.M,
        committee=coin_committee,
        threshold=params.coin_k,
    )

    print(
        f"Wrote {args.M} Phase-6 ATCaaS configs to {out_dir}: "
        + json.dumps(params.to_config_obj(), sort_keys=True)
    )


if __name__ == "__main__":
    main()
