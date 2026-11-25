"""Isolated local multi-process correctness fixtures and public result checks."""
import base64
import itertools
import json
import os
from pathlib import Path
import secrets
import socket
from dataclasses import asdict
from beaver.a2dpss import A2DPSSContext, A2DPSSSuccessorResult
from beaver.g1_backend import G1Point, field_random
from beaver.handover_fixtures import make_genesis_shares
from beaver.pedersen_shamir import interpolate_at_zero
from .crypto import generate_identity
from .model import Parameters, canonical, decode, digest


def private_write(path, value):
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical(value))


def consume_bootstrap_seed(path, value):
    """Atomically drop the root signing seed from this one-use local shard."""
    path = Path(path)
    temporary = path.with_name(path.name + ".consumed-" + secrets.token_hex(8))
    private_write(temporary, value)
    os.replace(str(temporary), str(path))


def create_fixture(directory, config, application="a2dpss", population=5, m=5, h=4,
                   mode="bernoulli", probability=1.0, service_t=1, max_views=16):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "private").mkdir(mode=0o700)
    (directory / "public").mkdir()
    parameters = Parameters(population, m, h, probability, mode, "correctness-fixture", max_views)
    if application not in ("agreement", "a2dpss"):
        raise ValueError("unknown application")
    if not 0 <= service_t < population:
        raise ValueError("invalid service threshold")
    config = decode(Path(config).read_bytes())
    srs = json.loads(base64.b64decode(config["extra"]["SRS"]))["Pk"]
    def raw(point):
        return (b"\x04" + int(point["X"]).to_bytes(48, "big") + int(point["Y"]).to_bytes(48, "big")).hex()
    g, generator_h = raw(srs["G1_g"][0]), raw(srs["G1_h"][0])
    identities = [generate_identity(g) for _ in range(population)]
    sockets, ports = [], []
    try:
        for _ in range(population):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
            ports.append(sock.getsockname()[1])
    finally:
        for sock in sockets:
            sock.close()
    session = "acd25-" + secrets.token_hex(16)
    manifest = {"version": 1, "protocol": "ACD25-NWA", "session": session,
                "application": application, "parameters": asdict(parameters),
                "g": g, "h": generator_h, "public_keys": [p for _, p in identities],
                "peers": [["127.0.0.1", port] for port in ports],
                "scope": "single-decision correctness gate; not a performance result",
                "crypto": {"vrf": "BLS12-381 real backend", "pvss": "Pedersen field PVSS v2",
                           "coin": "none; native AVSS/Gather/PE",
                           "signatures": "ratcheted Ed25519; logical Python erasure",
                           "recipient_encryption": "ECDH/HKDF/AES-GCM PoC; not KM-NCE"},
                "committee_fixture": "all-member" if probability == 1 and mode == "bernoulli" else mode,
                "online_avss_setup": True, "a2_predicate_cache": False}
    manifest["transport"] = "existing ZMQ IPC restricted to trusted loopback; signed protocol envelopes"
    successors = [field_random() for _ in range(population)]
    if application == "a2dpss":
        secret, secret_hat = field_random(), field_random()
        genesis = make_genesis_shares(g_hex=g, h_hex=generator_h, secret=secret, secret_hat=secret_hat,
                                      holder_ids=list(range(population)), service_t=service_t)
        context = A2DPSSContext(epoch=1, sid=session, source_committee="old", successor_committee="new",
            transfer_seq=1, g_hex=g, h_hex=generator_h, service_t=service_t,
            successor_rpk_list=tuple((G1Point.from_hex(g) * key).to_hex() for key in successors),
            old_commitment_vector=genesis.old_commitment_vector, holder_eval_points=genesis.holder_eval_points,
            successor_role_indices={i: i for i in range(population)})
        manifest["context"] = context.to_obj()
        manifest["old_secret_commitment"] = (G1Point.from_hex(g) * secret + G1Point.from_hex(generator_h) * secret_hat).to_hex()
    for node, (private, _) in enumerate(identities):
        shard = {"node": node, "identity": private}
        if application == "a2dpss":
            shard.update(old_share=list(genesis.share_for(node)), successor_key=successors[node])
        private_write(directory / "private" / ("node-%d.json" % node), shard)
    (directory / "public" / "manifest.json").write_bytes(canonical(manifest))
    return manifest


def analyze(directory):
    directory = Path(directory)
    manifest = decode((directory / "public" / "manifest.json").read_bytes())
    count = manifest["parameters"]["population"]
    results = [decode((directory / "public" / ("result-%d.json" % i)).read_bytes()) for i in range(count)]
    if any(result["node"] != i or result["session"] != manifest["session"] or result["status"] != "ok"
           for i, result in enumerate(results)):
        raise ValueError("missing, failed, duplicate, or wrong-session participant")
    decisions = {digest(result["decision"]) for result in results}
    if len(decisions) != 1:
        raise ValueError("agreement failure")
    if any(result["metrics"].get("avss_rec_started", 0) < 1 for result in results):
        raise ValueError("missing native election reconstruction")
    summary = {"status": "ok", "session": manifest["session"], "application": manifest["application"],
               "participants": count, "decision_digest": next(iter(decisions)),
               "decision_views": [r["metrics"]["decision_view"] for r in results],
               "scope": manifest["scope"], "committee_fixture": manifest["committee_fixture"],
               "crypto": manifest["crypto"], "wire_bytes": sum(r["wire_bytes"] for r in results),
               "elapsed_seconds": [r["elapsed_seconds"] for r in results]}
    if manifest["application"] == "a2dpss":
        context = A2DPSSContext.from_obj(manifest["context"])
        if len({digest(r["public_result"]) for r in results}) != 1:
            raise ValueError("different finalized T or commitments")
        if any(len(r["public_result"]["accepted_eval_points"]) != context.service_t + 1 for r in results):
            raise ValueError("T is not the exact reconstruction threshold")
        # Local correctness supervisor privately checks every threshold subset.
        # Neither scalars nor reconstructed secrets are copied into public output.
        private = [A2DPSSSuccessorResult.from_obj(decode(
            (directory / "private" / ("successor-%d.json" % i)).read_bytes())) for i in range(count)]
        checks = 0
        for group in itertools.combinations(private, context.service_t + 1):
            points = [r.my_role_idx + 1 for r in group]
            secret = interpolate_at_zero(points, [r.share_s for r in group])
            secret_hat = interpolate_at_zero(points, [r.share_s_hat for r in group])
            recovered = G1Point.from_hex(context.g_hex) * secret + G1Point.from_hex(context.h_hex) * secret_hat
            if recovered != G1Point.from_hex(manifest["old_secret_commitment"]):
                raise ValueError("handover did not preserve the old sharing")
            checks += 1
        summary.update(all_successors_reconstructed=True, threshold_subsets_checked=checks,
                       t_size=context.service_t + 1)
    return summary
