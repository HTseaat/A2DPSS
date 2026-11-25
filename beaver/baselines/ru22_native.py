"""RU22 Algorithm-1 leader/verifier refresh using the existing chunk PVSS.

No DAG, RBC delivery, common order, or selected global leader is used here.
Verifier votes carry the entire selected-PVR package, so an honest verifier
also provides reconstruction availability without a subsequent request phase.

Each leader has its own n-member verifier committee. A verifier role has a
dedicated signing key, signs once for its assigned leader, and then closes.
Physical machines may host several distinct roles.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace

from beaver.baselines.ru22_certificate import (
    RU22QuorumCertificate, canonical, digest, qvc_statement,
    sign_statement, verify_statement, verify_qvc,
)
from beaver.baselines.ru22_dpss import (
    RU22Context, RU22VPS, RU22InitialCertificate, RU22PVR,
    RU22ContributionBatch, RU22LocalVPSShare, RU22HolderState,
    validate_ru22_contribution_batch_shape, verify_ru22_pvr_crypto,
    _decode_object,
)
from beaver.chunk_elgamal import derive_public_key, decrypt_elgamal_ciphertext
from beaver.pvss import interpolate_c_v_vector, pvss_deliver, pvss_interpolate
from beaver.pedersen_shamir import pedersen_commit
from beaver.g1_backend import G1Point


CRYPTO_STAGES = ("source_pvss_generation", "source_dleq_generation",
                 "leader_pvss_verification", "leader_dleq_verification",
                 "verifier_pvss_verification", "verifier_dleq_verification")


class NativeMetrics:
    def __init__(self):
        self.stages = defaultdict(lambda: dict(invocations=0, wall_seconds=0., cpu_seconds=0.))
        # Nested measurements: never add these to the non-overlapping CPU total.
        self.crypto = defaultdict(lambda: dict(invocations=0, wall_seconds=0., cpu_seconds=0.))

    def crypto_recorder(self, role):
        def record(stage, wall_started, cpu_started):
            m = self.crypto[role + "_" + stage + "_verification"]
            m["invocations"] += 1
            m["wall_seconds"] += time.perf_counter() - wall_started
            m["cpu_seconds"] += time.process_time() - cpu_started
        return record

    @contextmanager
    def measure(self, stage):
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            m = self.stages[stage]
            m["invocations"] += 1
            m["wall_seconds"] += time.perf_counter() - wall
            m["cpu_seconds"] += time.process_time() - cpu

    def snapshot(self):
        return {s: dict(v) for s, v in self.stages.items()}


class NativeParentRegistry:
    """Pin genesis inputs; authenticate subsequent VPS against trusted committees."""
    def __init__(self, context, *, genesis=(), authorities=()):
        self.context = context
        self.parents = {}
        self.genesis = {v.vps_id: v.to_bytes() for v in genesis}
        self.authorities = {a.committee_digest: a for a in authorities}
        for v in genesis:
            if not isinstance(v.certificate, RU22InitialCertificate):
                raise ValueError("genesis pin must be an initial certificate")
            self.add(v)

    def add(self, vps):
        vps = RU22VPS.from_bytes(vps.to_bytes())
        c = self.context
        if (vps.epoch != c.source_epoch or vps.committee != c.source_committee
                or len(vps.encrypted_commitments) != c.resharing_n):
            raise ValueError("parent does not match source committee/epoch")
        cert = vps.certificate
        if isinstance(cert, RU22InitialCertificate):
            if self.genesis.get(vps.vps_id) != vps.to_bytes():
                raise ValueError("untrusted initial VPS")
        elif isinstance(cert, RU22QuorumCertificate):
            authority = self.authorities.get(cert.committee_digest)
            if authority is None or not authority.verify(vps):
                raise ValueError("invalid or untrusted parent QVC")
        else:
            raise ValueError("native RU22 does not accept a DAG certificate")
        self.parents[vps.vps_id] = vps
        return vps


@dataclass(frozen=True)
class NativeAuthority:
    """Committee identity is bound to the target context and pinned key set."""
    context_bytes: bytes
    leader_keys: tuple[tuple[int, str], ...]
    verifier_committees: tuple[tuple[int, tuple[tuple[int, str], ...]], ...]

    def __post_init__(self):
        c = RU22Context.from_bytes(self.context_bytes)
        def committee(entries):
            keys = tuple(sorted(entries))
            if len(keys) != c.resharing_n or len({i for i, _ in keys}) != len(keys):
                raise ValueError("native intermediate committee must have n distinct members")
            if len({k for _, k in keys}) != len(keys):
                raise ValueError("committee members must have distinct signing keys")
            from beaver.baselines.ru22_certificate import strict_hex, node_id
            for i, key in keys:
                node_id(i); strict_hex(key, 32)
            return keys
        leaders = committee(self.leader_keys)
        verifiers = tuple(sorted((i, committee(keys)) for i, keys in self.verifier_committees))
        if tuple(i for i, _ in verifiers) != tuple(i for i, _ in leaders):
            raise ValueError("each leader must have exactly one verifier committee")
        all_keys = [k for _, k in leaders] + [k for _, keys in verifiers for _, k in keys]
        if len(set(all_keys)) != len(all_keys):
            raise ValueError("distinct native roles must have distinct signing keys")
        object.__setattr__(self, "leader_keys", leaders)
        object.__setattr__(self, "verifier_committees", verifiers)
        object.__setattr__(self, "context_bytes", c.to_bytes())

    @property
    def context(self):
        return RU22Context.from_bytes(self.context_bytes)

    @property
    def committee_digest(self):
        return digest(dict(kind="ru22-native-authority/v2",
                           context=self.context.context_digest,
                           leaders=self.leader_keys, verifiers=self.verifier_committees))

    def verifiers_for(self, leader):
        return dict(dict(self.verifier_committees).get(leader, ()))

    def verify(self, vps):
        c = self.context
        return (isinstance(vps.certificate, RU22QuorumCertificate)
                and vps.epoch == c.target_epoch and vps.committee == c.successor_committee
                and vps.committee_context_digest == c.context_digest
                and len(vps.encrypted_commitments) == c.resharing_n
                and verify_qvc(vps.certificate, committee_digest=self.committee_digest,
                    signing_keys=self.verifiers_for(vps.certificate.leader_id), threshold=c.resharing_threshold + 1,
                    leader_ids=dict(self.leader_keys), vps_id=vps.vps_id))


def _single_batch(context, pvr):
    return RU22ContributionBatch(qid=context.encode_qid(),
        context_digest=context.context_digest, holder_role_id=pvr.dealer_role_id,
        holder_eval_point=pvr.dealer_eval_point, pvr_entries=[pvr])


def _check_pvr(context, parent, pvr, *, record_stage=lambda *_: None):
    validate_ru22_contribution_batch_shape(context, _single_batch(context, pvr))
    if pvr.parent_vps_id != parent.vps_id:
        raise ValueError("wrong parent VPS")
    if not verify_ru22_pvr_crypto(context, parent, pvr, record_stage=record_stage):
        raise ValueError("invalid PVSS or old-to-new binding proof")


@dataclass(frozen=True)
class NativeCombine:
    """Immutable selected-PVR package. No references to mutable local pool state."""
    payload: bytes

    def __post_init__(self):
        obj = _decode_object(self.payload, name="native combine")
        if set(obj) != {"schema", "context", "leader", "parent", "pvrs", "combined"}:
            raise ValueError("invalid combine fields")
        if obj["schema"] != "ru22-native-combine/v1":
            raise ValueError("unsupported combine version")
        from beaver.baselines.ru22_certificate import node_id, strict_hex
        node_id(obj["leader"]); strict_hex(obj["context"], 32)
        if not isinstance(obj["pvrs"], list) or not obj["pvrs"]:
            raise ValueError("empty PVR selection")
        RU22VPS.from_obj(obj["parent"])
        for p in obj["pvrs"]:
            RU22PVR.from_obj(p)
        object.__setattr__(self, "payload", canonical(obj))

    def to_obj(self):
        return json.loads(self.payload)

    def reconstruction_pvrs(self):
        return self.to_obj()["pvrs"]

    @property
    def package_digest(self):
        return digest(self.to_obj())

    def unsigned_vps(self, context):
        obj = self.to_obj()
        if obj["context"] != context.context_digest:
            raise ValueError("combine belongs to another context")
        # Placeholder solely for computing ciphertext identity, never a trusted root.
        return RU22VPS(epoch=context.target_epoch, committee=context.successor_committee,
            committee_context_digest=context.context_digest,
            encrypted_commitments=obj["combined"],
            certificate=RU22InitialCertificate("00" * 32))

    def statement(self, authority):
        return qvc_statement(authority.committee_digest,
            self.unsigned_vps(authority.context).vps_id, self.package_digest,
            self.to_obj()["leader"])


def make_combine(context, leader, parent, pvrs):
    """Combine an already validated same-parent threshold set."""
    pvrs = tuple(sorted(pvrs, key=lambda p: p.dealer_eval_point))
    if (len(pvrs) != context.resharing_threshold + 1
            or len({p.dealer_role_id for p in pvrs}) != len(pvrs)
            or len({p.dealer_eval_point for p in pvrs}) != len(pvrs)
            or any(p.parent_vps_id != parent.vps_id for p in pvrs)):
        raise ValueError("invalid same-parent threshold set")
    combined = interpolate_c_v_vector([p.dealer_eval_point for p in pvrs],
        [p.pvss_transcript.derived_c_v_vector for p in pvrs], context.resharing_threshold)
    return NativeCombine(canonical(dict(schema="ru22-native-combine/v1",
        context=context.context_digest, leader=leader, parent=parent.to_obj(),
        pvrs=[p.to_obj() for p in pvrs], combined=combined)))


def verify_combine(authority, registry, package, *, record_stage=lambda *_: None):
    c, obj = authority.context, package.to_obj()
    if obj["leader"] not in dict(authority.leader_keys) or obj["context"] != c.context_digest:
        raise ValueError("unknown leader or wrong combine context")
    parent = registry.add(RU22VPS.from_obj(obj["parent"]))
    pvrs = tuple(RU22PVR.from_obj(p) for p in obj["pvrs"])
    if len(pvrs) != c.resharing_threshold + 1:
        raise ValueError("incorrect PVR threshold")
    for pvr in pvrs:
        _check_pvr(c, parent, pvr, record_stage=record_stage)
    expected = make_combine(c, obj["leader"], parent, pvrs)
    if expected.payload != package.payload:
        raise ValueError("incorrect homomorphic combination or noncanonical selection")


class NativeLeader:
    def __init__(self, authority, registry, leader_id):
        if leader_id not in dict(authority.leader_keys):
            raise ValueError("not a leader")
        self.authority, self.registry, self.leader_id = authority, registry, leader_id
        self.buckets = defaultdict(dict)
        self.output = None
        self.metrics = NativeMetrics()
        self.rejected_pvrs = 0

    def receive(self, sender, batch, parents):
        if self.output is not None:
            return None
        c = self.authority.context
        batch = RU22ContributionBatch.from_bytes(batch.to_bytes())
        validate_ru22_contribution_batch_shape(c, batch)
        if sender != batch.holder_role_id:
            raise ValueError("source authentication mismatch")
        supplied = {v.vps_id: v for v in parents}
        # A bad sibling PVR must not suppress valid contributions in this batch.
        for pvr in batch.pvr_entries:
            if sender in self.buckets[pvr.parent_vps_id]:
                continue
            try:
                with self.metrics.measure("leader_pvr_verification"):
                    parent = self.registry.add(supplied[pvr.parent_vps_id])
                    _check_pvr(c, parent, pvr, record_stage=self.metrics.crypto_recorder("leader"))
            except (KeyError, ValueError, TypeError):
                self.rejected_pvrs += 1
                continue
            bucket = self.buckets[parent.vps_id]
            bucket[sender] = pvr
            if len(bucket) >= c.resharing_threshold + 1:
                with self.metrics.measure("leader_combination"):
                    self.output = make_combine(c, self.leader_id, parent, tuple(bucket.values()))
                return self.output
        return None


class NativeVerifier:
    def __init__(self, authority, registry, leader_id, verifier_id, signing_seed, *, verify_package=verify_combine):
        from Crypto.PublicKey import ECC
        key = ECC.construct(curve="Ed25519", seed=bytes(signing_seed))
        if authority.verifiers_for(leader_id).get(verifier_id) != key.public_key().export_key(format="raw").hex():
            raise ValueError("verifier signing key mismatch")
        self.authority, self.registry, self.verifier_id = authority, registry, verifier_id
        self._seed = bytearray(signing_seed)
        self.leader_id, self.sent, self.closed = leader_id, False, False
        self.verify_package = verify_package
        self.metrics = NativeMetrics()

    def receive(self, sender, package):
        if sender != self.leader_id or sender != package.to_obj()["leader"]:
            raise ValueError("leader authentication mismatch")
        if self.closed:
            return None
        self.verify(package)
        # Freeze before signing/sending. A failed send cannot reuse the role.
        self.sent = True
        try:
            with self.metrics.measure("verifier_signature_generation"):
                signature = sign_statement(self._seed, package.statement(self.authority))
        finally:
            self.close()
        return dict(schema="ru22-native-vote/v1", verifier=self.verifier_id,
                    package=package.to_obj(), signature=signature)

    def verify(self, package):
        with self.metrics.measure("verifier_combine_verification"):
            if self.verify_package is verify_combine:
                verify_combine(self.authority, self.registry, package,
                               record_stage=self.metrics.crypto_recorder("verifier"))
            else:
                self.verify_package(self.authority, self.registry, package)

    def close(self):
        self._seed[:] = b"\x00" * len(self._seed)
        self.closed = True


class NativeSuccessor:
    def __init__(self, authority, node, rsk, *, package_type=NativeCombine):
        c = authority.context
        if node not in c.successor_role_indices:
            raise ValueError("not a successor")
        index = c.successor_role_indices[node]
        if derive_public_key(rsk, c.successor_key_base_hex) != c.successor_rpk_list[index]:
            raise ValueError("successor decryption key mismatch")
        self.authority, self.node, self.index, self._rsk = authority, node, index, rsk
        self.votes = defaultdict(dict)
        self.results = {}
        self.leader_outputs = {}
        self.metrics = NativeMetrics()
        self.closed = False
        self.package_type = package_type

    def receive(self, sender, vote):
        if self.closed:
            raise RuntimeError("successor epoch has been consumed")
        if (set(vote) != {"schema", "verifier", "package", "signature"}
                or vote["schema"] != "ru22-native-vote/v1"
                or type(sender) is not int or sender != vote["verifier"]):
            raise ValueError("invalid verifier envelope")
        c = self.authority.context
        package = self.package_type(canonical(vote["package"]))
        leader = package.to_obj()["leader"]
        if leader not in dict(self.authority.leader_keys):
            raise ValueError("unknown LB-refresh leader")
        signing_keys = self.authority.verifiers_for(leader)
        if sender not in signing_keys:
            raise ValueError("verifier outside this leader's committee")
        statement = package.statement(self.authority)
        with self.metrics.measure("successor_signature_verification"):
            if not verify_statement(signing_keys[sender], statement, vote["signature"]):
                raise ValueError("invalid verifier signature")
        if leader in self.leader_outputs:
            return None
        bucket = self.votes[package.package_digest]
        bucket[sender] = vote["signature"]
        if len(bucket) < c.resharing_threshold + 1:
            return None
        cert = RU22QuorumCertificate(**{k: v for k, v in statement.items() if k != "kind"},
                                     signatures=tuple(sorted(bucket.items())))
        vps = replace(package.unsigned_vps(c), certificate=cert)
        with self.metrics.measure("successor_qvc_verification"):
            if not self.authority.verify(vps):
                raise ValueError("invalid quorum certificate")
        # QVC attests the full immutable package; no additional PVSS verification.
        # The selected chunks are present even if the leader has disappeared.
        with self.metrics.measure("successor_reconstruction"):
            pvrs = [RU22PVR.from_obj(p) for p in package.reconstruction_pvrs()]
            shares = [pvss_deliver(c.g_hex, c.h_hex, p.pvss_transcript, self._rsk, self.index) for p in pvrs]
            share, blind = pvss_interpolate([p.dealer_eval_point for p in pvrs], shares, c.resharing_threshold)
            plain = decrypt_elgamal_ciphertext(self._rsk, vps.encrypted_commitments[self.index].to_obj())
            if G1Point.from_hex(plain) != G1Point.from_hex(pedersen_commit(c.g_hex, c.h_hex, share, blind)):
                raise ValueError("new share does not open certified ciphertext")
        result = RU22LocalVPSShare(vps=vps, share_s=share, share_s_hat=blind)
        self.results.setdefault(vps.vps_id, result)
        self.leader_outputs[leader] = vps.vps_id
        return result

    def consume_to_holder(self):
        if self.closed or not self.results:
            raise RuntimeError("successor has no consumable VPS list")
        # Epoch switching is an external protocol event, never first-QVC arrival.
        state = RU22HolderState(holder_role_id=self.node, holder_eval_point=self.index + 1,
            recipient_index=self.index, recipient_secret_key=self._rsk,
            resharing_n=self.authority.context.resharing_n,
            local_vps_shares=list(self.results.values()))
        self.closed = True
        self._rsk = None
        self.results.clear()
        return state
