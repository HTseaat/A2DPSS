"""Native RU22 quorum certificates. Parsing is not authentication.

Certificates bind the ciphertext vector AND the chunk reconstruction package.
The latter must accompany verifier multicast: a commitment ciphertext alone
cannot recover a field-element share in this implementation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Mapping

from Crypto.Signature import eddsa
from Crypto.PublicKey import ECC


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def strict_hex(value, length):
    if not isinstance(value, str) or len(value) != 2 * length:
        raise ValueError("incorrect hex length")
    if bytes.fromhex(value).hex() != value:
        raise ValueError("noncanonical hex")
    return value


def node_id(value):
    if type(value) is not int or value < 0:
        raise ValueError("invalid role identity")
    return value


@dataclass(frozen=True)
class RU22QuorumCertificate:
    committee_digest: str
    vps_id: str
    reconstruction_digest: str
    leader_id: int
    signatures: tuple[tuple[int, str], ...]
    kind: str = field(default="ru22-qvc-v1", init=False)

    def __post_init__(self):
        for name in ("committee_digest", "vps_id", "reconstruction_digest"):
            strict_hex(getattr(self, name), 32)
        node_id(self.leader_id)
        signatures = tuple((node_id(i), strict_hex(s, 64)) for i, s in self.signatures)
        if not signatures or signatures != tuple(sorted(signatures)):
            raise ValueError("QVC signatures must be nonempty and sorted")
        if len({i for i, _ in signatures}) != len(signatures):
            raise ValueError("duplicate QVC signer")
        object.__setattr__(self, "signatures", signatures)

    def statement(self):
        return dict(kind=self.kind, committee_digest=self.committee_digest,
                    vps_id=self.vps_id, reconstruction_digest=self.reconstruction_digest,
                    leader_id=self.leader_id)

    def to_obj(self):
        return dict(self.statement(), signatures=[[i, s] for i, s in self.signatures])

    @classmethod
    def from_obj(cls, obj):
        fields = {"kind", "committee_digest", "vps_id", "reconstruction_digest",
                  "leader_id", "signatures"}
        if not isinstance(obj, Mapping) or set(obj) != fields or obj["kind"] != "ru22-qvc-v1":
            raise ValueError("invalid native QVC schema")
        return cls(**{k: v for k, v in obj.items() if k != "kind"})


def qvc_statement(committee_digest, vps_id, reconstruction_digest, leader_id):
    return dict(kind="ru22-qvc-v1", committee_digest=strict_hex(committee_digest, 32),
                vps_id=strict_hex(vps_id, 32),
                reconstruction_digest=strict_hex(reconstruction_digest, 32),
                leader_id=node_id(leader_id))


def sign_statement(seed: bytes, statement) -> str:
    key = ECC.construct(curve="Ed25519", seed=bytes(seed))
    return eddsa.new(key, "rfc8032").sign(canonical(statement)).hex()


def verify_statement(public_key: str, statement, signature: str) -> bool:
    try:
        key = eddsa.import_public_key(bytes.fromhex(strict_hex(public_key, 32)))
        eddsa.new(key, "rfc8032").verify(canonical(statement),
                                       bytes.fromhex(strict_hex(signature, 64)))
        return True
    except (TypeError, ValueError):
        return False


def verify_qvc(certificate, *, committee_digest, signing_keys,
               threshold, leader_ids, vps_id) -> bool:
    """Check against an externally trusted epoch committee, never QVC-supplied keys."""
    if not isinstance(certificate, RU22QuorumCertificate):
        return False
    if (type(threshold) is not int or not 1 <= threshold <= len(signing_keys)
            or certificate.committee_digest != committee_digest
            or certificate.vps_id != vps_id
            or certificate.leader_id not in leader_ids
            or len(certificate.signatures) < threshold):
        return False
    return all(i in signing_keys and verify_statement(signing_keys[i],
               certificate.statement(), sig) for i, sig in certificate.signatures)
