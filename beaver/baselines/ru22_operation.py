"""RU22 operation-coupled extension (implementation Steps 2--4).

The pure RU22 PVR wire remains unchanged.  This module wraps each base PVR
with two independent operation proofs:

* ``pi_link`` proves that a public ``vk`` contains the same share encrypted in
  the parent VPS coordinate; and
* ``pi_op`` proves that the operation share contains the same exponent as
  ``vk``.

The module owns local one-shot generation, strict application wire objects,
an uncached formal justifier, public finalization, and local successor
reconstruction.  DAG/network transport is intentionally deferred.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from beaver.baselines.ru22_dpss import (
    RU22ContributionBatch,
    RU22Context,
    RU22DAGDeliveryCertificate,
    RU22DPSSJustifierManager,
    RU22ElGamalCiphertext,
    RU22HolderMetrics,
    RU22HolderState,
    RU22JustifiedSelection,
    RU22LocalVPSShare,
    RU22PVR,
    RU22ValidatedBatch,
    RU22VPS,
    make_ru22_contribution_batch,
)
from beaver.chunk_elgamal import decrypt_elgamal_ciphertext, derive_public_key
from beaver.dag_tob import Justifier, PoolEntry, Qid
from beaver.g1_backend import (
    FIELD_ORDER,
    G1Point,
    field_add,
    field_mul,
    field_random,
)
from beaver.operation_vk import (
    VKOperationShareProof,
    VKOperationShareStatement,
    combine_vk_operation_shares,
    evaluate_and_prove_vk_operation_share,
    verify_vk_operation_share,
)
from beaver.pedersen_shamir import pedersen_commit
from beaver.pvss import pvss_deliver, pvss_interpolate
from beaver.sigma import canonical_json, hash_to_scalar
from beaver.threshold_operation_contract import (
    OPERATION_VK_BASE_HEX,
    RU22_OPERATION_PVR_SCHEMA,
    RU22_OPERATION_QID_KIND,
    ThresholdOperationContext,
    operation_output_digest,
)


LINK_STATEMENT_SCHEMA = "ru22-encrypted-commitment-vk-statement/v1"
LINK_PROOF_KIND = "ru22-encrypted-commitment-to-vk/v1"
LINK_PROOF_VERSION = 1
OPERATION_BATCH_SCHEMA = "ru22-operation-contribution-batch/v1"
OPERATION_PUBLIC_RESULT_SCHEMA = "ru22-operation-public-result/v1"

_LINK_STATEMENT_DOMAIN = b"ATCaaS/RU22/operation-link/statement/v1\x00"
_LINK_FS_DOMAIN = "atcaas-ru22-encrypted-commitment-to-vk-v1"
_OPERATION_PVR_DIGEST_DOMAIN = b"ATCaaS/RU22/operation-pvr/v1\x00"
_OPERATION_BATCH_DIGEST_DOMAIN = b"ATCaaS/RU22/operation-batch/v1\x00"
_PUBLIC_RESULT_DIGEST_DOMAIN = b"ATCaaS/RU22/operation-result/v1\x00"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_SCALAR_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")

_OPERATION_QID_FIELDS = frozenset(
    {
        "base_context_digest",
        "operation_context_digest",
        "source_epoch",
        "target_epoch",
        "sid",
        "source_committee",
        "successor_committee",
        "transfer_seq",
        "trial_id",
    }
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _decode_object(
    value: bytes | bytearray | memoryview | str, *, name: str
) -> dict[str, Any]:
    try:
        raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed {name} JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must encode a JSON object")
    return decoded


def _exact_fields(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    missing = expected - set(value)
    extra = set(value) - expected
    if missing:
        raise ValueError(f"{name} is missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"{name} has unknown fields: {sorted(extra)}")


def _strict_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _scalar(value: Any, *, name: str, nonzero: bool = False) -> int:
    scalar = _strict_int(value, name=name)
    if scalar >= FIELD_ORDER or (nonzero and scalar == 0):
        qualifier = "non-zero " if nonzero else ""
        raise ValueError(f"{name} must be a canonical {qualifier}scalar")
    return scalar


def _scalar_text(value: Any, *, name: str) -> int:
    if not isinstance(value, str) or _SCALAR_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical decimal scalar string")
    return _scalar(int(value), name=name)


def _nonempty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a canonical non-empty string")
    return value


def _digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase sha256 hex")
    return value


def _point(value: Any, *, name: str, nonidentity: bool = True) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a canonical G1 string")
    try:
        point = G1Point.from_hex(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a valid G1 point") from exc
    if nonidentity and point.is_identity():
        raise ValueError(f"{name} must not be the identity point")
    if point.to_hex() != value:
        raise ValueError(f"{name} must use canonical AMCL G1 encoding")
    return value


def _domain_digest(domain: bytes, value: Mapping[str, Any]) -> str:
    return hashlib.sha256(domain + canonical_json(dict(value))).hexdigest()


def build_ru22_operation_qid(
    context: RU22Context,
    operation_context: ThresholdOperationContext,
    *,
    trial_id: str,
) -> Qid:
    if not isinstance(context, RU22Context):
        raise TypeError("context must be an RU22Context")
    if not isinstance(operation_context, ThresholdOperationContext):
        raise TypeError("operation_context must be a ThresholdOperationContext")
    trial = _nonempty_text(trial_id, name="trial_id")
    return Qid.build(
        RU22_OPERATION_QID_KIND,
        base_context_digest=context.context_digest,
        operation_context_digest=operation_context.context_digest,
        source_epoch=context.source_epoch,
        target_epoch=context.target_epoch,
        sid=context.sid,
        source_committee=context.source_committee,
        successor_committee=context.successor_committee,
        transfer_seq=context.transfer_seq,
        trial_id=trial,
    )


def _canonical_operation_qid(
    value: Qid | str,
    *,
    base_context_digest: str,
    operation_context_digest: str,
    trial_id: str,
) -> str:
    encoded = value.encode() if isinstance(value, Qid) else value
    if not isinstance(encoded, str):
        raise TypeError("qid must be a Qid or canonical string")
    qid = Qid.decode(encoded)
    if qid.encode() != encoded or qid.kind != RU22_OPERATION_QID_KIND:
        raise ValueError("qid must be a canonical RU22 operation qid")
    fields = dict(qid.fields)
    if set(fields) != _OPERATION_QID_FIELDS:
        raise ValueError("RU22 operation qid fields are not canonical")
    if fields["base_context_digest"] != base_context_digest:
        raise ValueError("qid base_context_digest mismatch")
    if fields["operation_context_digest"] != operation_context_digest:
        raise ValueError("qid operation_context_digest mismatch")
    if fields["trial_id"] != trial_id:
        raise ValueError("qid trial_id mismatch")
    return encoded


@dataclass(frozen=True)
class RU22EncryptedCommitmentVKStatement:
    """Public statement for the three-relation RU22 link proof."""

    qid: str
    arm_context_digest: str
    operation_context_digest: str
    parent_vps_id: str
    coordinate_index: int
    dealer_role_id: int
    dealer_eval_point: int
    key_base_hex: str
    recipient_public_key_hex: str
    ciphertext: RU22ElGamalCiphertext | Mapping[str, str]
    g_hex: str
    h_hex: str
    operation_vk_base_hex: str
    operation_vk_hex: str
    statement_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.qid, str):
            raise TypeError("qid must be a canonical string")
        qid = Qid.decode(self.qid)
        if qid.encode() != self.qid or qid.kind != RU22_OPERATION_QID_KIND:
            raise ValueError("link statement qid must be a RU22 operation qid")
        ciphertext = (
            self.ciphertext
            if isinstance(self.ciphertext, RU22ElGamalCiphertext)
            else RU22ElGamalCiphertext.from_obj(self.ciphertext)
        )
        values = (
            ("qid", self.qid),
            (
                "arm_context_digest",
                _digest(self.arm_context_digest, name="arm_context_digest"),
            ),
            (
                "operation_context_digest",
                _digest(
                    self.operation_context_digest,
                    name="operation_context_digest",
                ),
            ),
            (
                "parent_vps_id",
                _digest(self.parent_vps_id, name="parent_vps_id"),
            ),
            (
                "coordinate_index",
                _strict_int(self.coordinate_index, name="coordinate_index"),
            ),
            (
                "dealer_role_id",
                _strict_int(self.dealer_role_id, name="dealer_role_id"),
            ),
            (
                "dealer_eval_point",
                _strict_int(
                    self.dealer_eval_point,
                    name="dealer_eval_point",
                    minimum=1,
                ),
            ),
            ("key_base_hex", _point(self.key_base_hex, name="key_base_hex")),
            (
                "recipient_public_key_hex",
                _point(
                    self.recipient_public_key_hex,
                    name="recipient_public_key_hex",
                ),
            ),
            ("ciphertext", RU22ElGamalCiphertext.from_obj(ciphertext.to_obj())),
            ("g_hex", _point(self.g_hex, name="g_hex")),
            ("h_hex", _point(self.h_hex, name="h_hex")),
            (
                "operation_vk_base_hex",
                _point(
                    self.operation_vk_base_hex,
                    name="operation_vk_base_hex",
                ),
            ),
            (
                "operation_vk_hex",
                _point(self.operation_vk_hex, name="operation_vk_hex"),
            ),
        )
        for name, item in values:
            object.__setattr__(self, name, item)
        if self.g_hex == self.h_hex:
            raise ValueError("g_hex and h_hex must be distinct")
        if self.operation_vk_base_hex != OPERATION_VK_BASE_HEX:
            raise ValueError("operation_vk_base_hex must equal the frozen G_vk")
        object.__setattr__(self, "statement_digest", self.digest())

    def to_obj(self) -> dict[str, Any]:
        return {
            "schema": LINK_STATEMENT_SCHEMA,
            "qid": self.qid,
            "arm_context_digest": self.arm_context_digest,
            "operation_context_digest": self.operation_context_digest,
            "parent_vps_id": self.parent_vps_id,
            "coordinate_index": self.coordinate_index,
            "dealer_role_id": self.dealer_role_id,
            "dealer_eval_point": self.dealer_eval_point,
            "key_base_hex": self.key_base_hex,
            "recipient_public_key_hex": self.recipient_public_key_hex,
            "ciphertext": self.ciphertext.to_obj(),
            "g_hex": self.g_hex,
            "h_hex": self.h_hex,
            "operation_vk_base_hex": self.operation_vk_base_hex,
            "operation_vk_hex": self.operation_vk_hex,
        }

    def digest(self) -> str:
        return _domain_digest(_LINK_STATEMENT_DOMAIN, self.to_obj())


@dataclass(frozen=True)
class RU22EncryptedCommitmentVKProof:
    statement_digest: str
    t_public_key_hex: str
    t_ciphertext_hex: str
    t_vk_hex: str
    z_key: int
    z_share: int
    z_blind: int

    def __post_init__(self) -> None:
        for name, item in (
            (
                "statement_digest",
                _digest(self.statement_digest, name="statement_digest"),
            ),
            (
                "t_public_key_hex",
                _point(self.t_public_key_hex, name="t_public_key_hex"),
            ),
            (
                "t_ciphertext_hex",
                _point(self.t_ciphertext_hex, name="t_ciphertext_hex"),
            ),
            ("t_vk_hex", _point(self.t_vk_hex, name="t_vk_hex")),
            ("z_key", _scalar(self.z_key, name="z_key")),
            ("z_share", _scalar(self.z_share, name="z_share")),
            ("z_blind", _scalar(self.z_blind, name="z_blind")),
        ):
            object.__setattr__(self, name, item)

    def to_obj(self) -> dict[str, Any]:
        return {
            "kind": LINK_PROOF_KIND,
            "version": LINK_PROOF_VERSION,
            "statement_digest": self.statement_digest,
            "t_public_key_hex": self.t_public_key_hex,
            "t_ciphertext_hex": self.t_ciphertext_hex,
            "t_vk_hex": self.t_vk_hex,
            "z_key": str(self.z_key),
            "z_share": str(self.z_share),
            "z_blind": str(self.z_blind),
        }

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "RU22EncryptedCommitmentVKProof":
        expected = {
            "kind",
            "version",
            "statement_digest",
            "t_public_key_hex",
            "t_ciphertext_hex",
            "t_vk_hex",
            "z_key",
            "z_share",
            "z_blind",
        }
        _exact_fields(value, expected, name="RU22 encrypted-commitment-vk proof")
        if value["kind"] != LINK_PROOF_KIND:
            raise ValueError("unexpected RU22 link proof kind")
        if type(value["version"]) is not int or value["version"] != LINK_PROOF_VERSION:
            raise ValueError("unsupported RU22 link proof version")
        return cls(
            statement_digest=value["statement_digest"],
            t_public_key_hex=value["t_public_key_hex"],
            t_ciphertext_hex=value["t_ciphertext_hex"],
            t_vk_hex=value["t_vk_hex"],
            z_key=_scalar_text(value["z_key"], name="z_key"),
            z_share=_scalar_text(value["z_share"], name="z_share"),
            z_blind=_scalar_text(value["z_blind"], name="z_blind"),
        )

    @classmethod
    def from_bytes(
        cls, value: bytes | bytearray | memoryview | str
    ) -> "RU22EncryptedCommitmentVKProof":
        return cls.from_obj(_decode_object(value, name="RU22 link proof"))


def _link_challenge(
    statement: RU22EncryptedCommitmentVKStatement,
    *,
    t_public_key_hex: str,
    t_ciphertext_hex: str,
    t_vk_hex: str,
) -> int:
    return hash_to_scalar(
        _LINK_FS_DOMAIN,
        {
            "proof_kind": LINK_PROOF_KIND,
            "proof_version": LINK_PROOF_VERSION,
            "statement": statement.to_obj(),
            "statement_digest": statement.statement_digest,
            "t_public_key_hex": t_public_key_hex,
            "t_ciphertext_hex": t_ciphertext_hex,
            "t_vk_hex": t_vk_hex,
        },
        nonzero=True,
    )


def prove_ru22_encrypted_commitment_vk(
    statement: RU22EncryptedCommitmentVKStatement,
    *,
    recipient_secret_key: int,
    share: int,
    blind_share: int,
) -> RU22EncryptedCommitmentVKProof:
    """Prove the parent ciphertext-to-vk relation without a PVSS argument."""

    if not isinstance(statement, RU22EncryptedCommitmentVKStatement):
        raise TypeError("statement must be an RU22 link statement")
    key = _scalar(recipient_secret_key, name="recipient_secret_key", nonzero=True)
    secret_share = _scalar(share, name="share", nonzero=True)
    blind = _scalar(blind_share, name="blind_share")
    key_base = G1Point.from_hex(statement.key_base_hex)
    public_key = G1Point.from_hex(statement.recipient_public_key_hex)
    ciphertext_a = G1Point.from_hex(statement.ciphertext.c1)
    ciphertext_b = G1Point.from_hex(statement.ciphertext.c2)
    g = G1Point.from_hex(statement.g_hex)
    h = G1Point.from_hex(statement.h_hex)
    vk_base = G1Point.from_hex(statement.operation_vk_base_hex)
    vk = G1Point.from_hex(statement.operation_vk_hex)
    if key_base * key != public_key:
        raise ValueError("recipient_secret_key does not open the public key")
    if ciphertext_a * key + g * secret_share + h * blind != ciphertext_b:
        raise ValueError("witness does not open the parent ciphertext")
    if vk_base * secret_share != vk:
        raise ValueError("share does not open operation_vk_hex")

    r_key = field_random()
    r_share = field_random()
    r_blind = field_random()
    t_public_key = key_base * r_key
    t_ciphertext = ciphertext_a * r_key + g * r_share + h * r_blind
    t_vk = vk_base * r_share
    challenge = _link_challenge(
        statement,
        t_public_key_hex=t_public_key.to_hex(),
        t_ciphertext_hex=t_ciphertext.to_hex(),
        t_vk_hex=t_vk.to_hex(),
    )
    return RU22EncryptedCommitmentVKProof(
        statement_digest=statement.statement_digest,
        t_public_key_hex=t_public_key.to_hex(),
        t_ciphertext_hex=t_ciphertext.to_hex(),
        t_vk_hex=t_vk.to_hex(),
        z_key=field_add(r_key, field_mul(challenge, key)),
        z_share=field_add(r_share, field_mul(challenge, secret_share)),
        z_blind=field_add(r_blind, field_mul(challenge, blind)),
    )


def verify_ru22_encrypted_commitment_vk(
    statement: RU22EncryptedCommitmentVKStatement,
    proof: RU22EncryptedCommitmentVKProof | Mapping[str, Any],
) -> bool:
    """Verify the three link relations; malformed input fails closed."""

    try:
        if not isinstance(statement, RU22EncryptedCommitmentVKStatement):
            return False
        parsed = (
            proof
            if isinstance(proof, RU22EncryptedCommitmentVKProof)
            else RU22EncryptedCommitmentVKProof.from_obj(proof)
        )
        if not hmac.compare_digest(
            statement.statement_digest, parsed.statement_digest
        ):
            return False
        challenge = _link_challenge(
            statement,
            t_public_key_hex=parsed.t_public_key_hex,
            t_ciphertext_hex=parsed.t_ciphertext_hex,
            t_vk_hex=parsed.t_vk_hex,
        )
        key_base = G1Point.from_hex(statement.key_base_hex)
        public_key = G1Point.from_hex(statement.recipient_public_key_hex)
        ciphertext_a = G1Point.from_hex(statement.ciphertext.c1)
        ciphertext_b = G1Point.from_hex(statement.ciphertext.c2)
        g = G1Point.from_hex(statement.g_hex)
        h = G1Point.from_hex(statement.h_hex)
        vk_base = G1Point.from_hex(statement.operation_vk_base_hex)
        vk = G1Point.from_hex(statement.operation_vk_hex)
        public_key_relation = (
            key_base * parsed.z_key
            == G1Point.from_hex(parsed.t_public_key_hex) + public_key * challenge
        )
        ciphertext_relation = (
            ciphertext_a * parsed.z_key
            + g * parsed.z_share
            + h * parsed.z_blind
            == G1Point.from_hex(parsed.t_ciphertext_hex)
            + ciphertext_b * challenge
        )
        vk_relation = (
            vk_base * parsed.z_share
            == G1Point.from_hex(parsed.t_vk_hex) + vk * challenge
        )
        return bool(public_key_relation and ciphertext_relation and vk_relation)
    except Exception:
        return False


def _copy_base_pvr(value: RU22PVR | Mapping[str, Any]) -> RU22PVR:
    if isinstance(value, RU22PVR):
        return RU22PVR.from_bytes(value.to_bytes())
    return RU22PVR.from_obj(value)


def _copy_link_proof(
    value: RU22EncryptedCommitmentVKProof | Mapping[str, Any],
) -> RU22EncryptedCommitmentVKProof:
    if isinstance(value, RU22EncryptedCommitmentVKProof):
        return RU22EncryptedCommitmentVKProof.from_bytes(value.to_bytes())
    return RU22EncryptedCommitmentVKProof.from_obj(value)


def _copy_operation_proof(
    value: VKOperationShareProof | Mapping[str, Any],
) -> VKOperationShareProof:
    if isinstance(value, VKOperationShareProof):
        return VKOperationShareProof.from_bytes(value.to_bytes())
    return VKOperationShareProof.from_obj(value)


@dataclass(frozen=True)
class RU22OperationPVR:
    """Application wrapper around an unchanged pure RU22 PVR."""

    base_pvr: RU22PVR | Mapping[str, Any]
    operation_vk: str
    encrypted_commitment_vk_proof: (
        RU22EncryptedCommitmentVKProof | Mapping[str, Any]
    )
    operation_share: str
    vk_operation_share_proof: VKOperationShareProof | Mapping[str, Any]
    operation_pvr_digest: str = field(init=False)

    def __post_init__(self) -> None:
        base_pvr = _copy_base_pvr(self.base_pvr)
        operation_vk = _point(self.operation_vk, name="operation_vk")
        link_proof = _copy_link_proof(self.encrypted_commitment_vk_proof)
        operation_share = _point(self.operation_share, name="operation_share")
        operation_proof = _copy_operation_proof(self.vk_operation_share_proof)
        for name, value in (
            ("base_pvr", base_pvr),
            ("operation_vk", operation_vk),
            ("encrypted_commitment_vk_proof", link_proof),
            ("operation_share", operation_share),
            ("vk_operation_share_proof", operation_proof),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "operation_pvr_digest",
            _domain_digest(
                _OPERATION_PVR_DIGEST_DOMAIN,
                self.digest_material(),
            ),
        )

    @property
    def parent_vps_id(self) -> str:
        return self.base_pvr.parent_vps_id

    def digest_material(self) -> dict[str, Any]:
        return {
            "schema": RU22_OPERATION_PVR_SCHEMA,
            "base_pvr": self.base_pvr.to_obj(),
            "operation_vk": self.operation_vk,
            "encrypted_commitment_vk_proof": (
                self.encrypted_commitment_vk_proof.to_obj()
            ),
            "operation_share": self.operation_share,
            "vk_operation_share_proof": (
                self.vk_operation_share_proof.to_obj()
            ),
        }

    def to_obj(self) -> dict[str, Any]:
        value = self.digest_material()
        value["operation_pvr_digest"] = self.operation_pvr_digest
        return value

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "RU22OperationPVR":
        expected = {
            "schema",
            "base_pvr",
            "operation_vk",
            "encrypted_commitment_vk_proof",
            "operation_share",
            "vk_operation_share_proof",
            "operation_pvr_digest",
        }
        _exact_fields(value, expected, name="RU22 operation PVR")
        if value["schema"] != RU22_OPERATION_PVR_SCHEMA:
            raise ValueError("unsupported RU22 operation PVR schema")
        supplied_digest = _digest(
            value["operation_pvr_digest"], name="operation_pvr_digest"
        )
        pvr = cls(
            base_pvr=value["base_pvr"],
            operation_vk=value["operation_vk"],
            encrypted_commitment_vk_proof=value[
                "encrypted_commitment_vk_proof"
            ],
            operation_share=value["operation_share"],
            vk_operation_share_proof=value["vk_operation_share_proof"],
        )
        if not hmac.compare_digest(pvr.operation_pvr_digest, supplied_digest):
            raise ValueError("RU22 operation_pvr_digest mismatch")
        return pvr

    @classmethod
    def from_bytes(
        cls, value: bytes | bytearray | memoryview | str
    ) -> "RU22OperationPVR":
        return cls.from_obj(_decode_object(value, name="RU22 operation PVR"))


@dataclass(frozen=True)
class RU22OperationContributionBatch:
    qid: str | Qid
    base_context_digest: str
    operation_context_digest: str
    trial_id: str
    holder_role_id: int
    holder_eval_point: int
    pvr_entries: Sequence[RU22OperationPVR | Mapping[str, Any]]
    batch_digest: str = field(init=False)

    def __post_init__(self) -> None:
        base_digest = _digest(
            self.base_context_digest, name="base_context_digest"
        )
        operation_digest = _digest(
            self.operation_context_digest,
            name="operation_context_digest",
        )
        trial_id = _nonempty_text(self.trial_id, name="trial_id")
        qid = _canonical_operation_qid(
            self.qid,
            base_context_digest=base_digest,
            operation_context_digest=operation_digest,
            trial_id=trial_id,
        )
        role_id = _strict_int(self.holder_role_id, name="holder_role_id")
        eval_point = _strict_int(
            self.holder_eval_point, name="holder_eval_point", minimum=1
        )
        if not isinstance(self.pvr_entries, (list, tuple)):
            raise TypeError("pvr_entries must be a sequence")
        entries = tuple(
            RU22OperationPVR.from_bytes(item.to_bytes())
            if isinstance(item, RU22OperationPVR)
            else RU22OperationPVR.from_obj(item)
            for item in self.pvr_entries
        )
        if not entries:
            raise ValueError("pvr_entries must be non-empty")
        parent_ids = set()
        for entry in entries:
            base = entry.base_pvr
            if (
                base.context_digest != base_digest
                or base.dealer_role_id != role_id
                or base.dealer_eval_point != eval_point
            ):
                raise ValueError(
                    "operation tuple/base PVR identity does not match batch"
                )
            if base.parent_vps_id in parent_ids:
                raise ValueError("batch contains duplicate parent_vps_id")
            parent_ids.add(base.parent_vps_id)
        for name, value in (
            ("qid", qid),
            ("base_context_digest", base_digest),
            ("operation_context_digest", operation_digest),
            ("trial_id", trial_id),
            ("holder_role_id", role_id),
            ("holder_eval_point", eval_point),
            ("pvr_entries", entries),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "batch_digest",
            _domain_digest(
                _OPERATION_BATCH_DIGEST_DOMAIN,
                self.digest_material(),
            ),
        )

    def digest_material(self) -> dict[str, Any]:
        return {
            "schema": OPERATION_BATCH_SCHEMA,
            "qid": self.qid,
            "base_context_digest": self.base_context_digest,
            "operation_context_digest": self.operation_context_digest,
            "trial_id": self.trial_id,
            "holder_role_id": self.holder_role_id,
            "holder_eval_point": self.holder_eval_point,
            "pvr_entries": [entry.to_obj() for entry in self.pvr_entries],
        }

    def to_obj(self) -> dict[str, Any]:
        value = self.digest_material()
        value["batch_digest"] = self.batch_digest
        return value

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_obj())

    def to_base_batch(self) -> RU22ContributionBatch:
        return RU22ContributionBatch(
            qid=self.pvr_entries[0].base_pvr.qid,
            context_digest=self.base_context_digest,
            holder_role_id=self.holder_role_id,
            holder_eval_point=self.holder_eval_point,
            pvr_entries=[entry.base_pvr for entry in self.pvr_entries],
        )

    @classmethod
    def from_obj(
        cls, value: Mapping[str, Any]
    ) -> "RU22OperationContributionBatch":
        expected = {
            "schema",
            "qid",
            "base_context_digest",
            "operation_context_digest",
            "trial_id",
            "holder_role_id",
            "holder_eval_point",
            "pvr_entries",
            "batch_digest",
        }
        _exact_fields(value, expected, name="RU22 operation contribution batch")
        if value["schema"] != OPERATION_BATCH_SCHEMA:
            raise ValueError("unsupported RU22 operation batch schema")
        supplied_digest = _digest(value["batch_digest"], name="batch_digest")
        batch = cls(
            qid=value["qid"],
            base_context_digest=value["base_context_digest"],
            operation_context_digest=value["operation_context_digest"],
            trial_id=value["trial_id"],
            holder_role_id=value["holder_role_id"],
            holder_eval_point=value["holder_eval_point"],
            pvr_entries=value["pvr_entries"],
        )
        if not hmac.compare_digest(batch.batch_digest, supplied_digest):
            raise ValueError("RU22 operation batch_digest mismatch")
        return batch

    @classmethod
    def from_bytes(
        cls, value: bytes | bytearray | memoryview | str
    ) -> "RU22OperationContributionBatch":
        return cls.from_obj(_decode_object(value, name="RU22 operation batch"))


@dataclass
class RU22OperationHolderMetrics:
    base: RU22HolderMetrics = field(default_factory=RU22HolderMetrics)
    link_proof_seconds: float = 0.0
    link_proof_cpu_seconds: float = 0.0
    operation_proof_seconds: float = 0.0
    operation_proof_cpu_seconds: float = 0.0
    serialization_seconds: float = 0.0
    serialization_cpu_seconds: float = 0.0
    link_proof_bytes: int = 0
    operation_proof_bytes: int = 0
    operation_vk_bytes: int = 0
    operation_share_bytes: int = 0
    operation_pvr_bytes: int = 0
    batch_bytes: int = 0
    pvr_count: int = 0


@dataclass(frozen=True)
class RU22OperationHolderActionResult:
    qid: str
    mid: str
    batch: RU22OperationContributionBatch
    batch_bytes: bytes
    metrics: RU22OperationHolderMetrics


def _link_statement(
    *,
    context: RU22Context,
    operation_context: ThresholdOperationContext,
    qid: str,
    parent: RU22VPS,
    base_pvr: RU22PVR,
    recipient_index: int,
    operation_vk: str,
) -> RU22EncryptedCommitmentVKStatement:
    return RU22EncryptedCommitmentVKStatement(
        qid=qid,
        arm_context_digest=context.context_digest,
        operation_context_digest=operation_context.context_digest,
        parent_vps_id=parent.vps_id,
        coordinate_index=recipient_index,
        dealer_role_id=base_pvr.dealer_role_id,
        dealer_eval_point=base_pvr.dealer_eval_point,
        key_base_hex=context.source_key_base_hex,
        recipient_public_key_hex=context.source_rpk_list[recipient_index],
        ciphertext=parent.encrypted_commitments[recipient_index],
        g_hex=context.g_hex,
        h_hex=context.h_hex,
        operation_vk_base_hex=operation_context.operation_vk_base_hex,
        operation_vk_hex=operation_vk,
    )


def _operation_statement(
    *,
    context: RU22Context,
    operation_context: ThresholdOperationContext,
    trial_id: str,
    qid: str,
    operation_pvr: RU22OperationPVR,
) -> VKOperationShareStatement:
    base = operation_pvr.base_pvr
    return VKOperationShareStatement(
        protocol="ru22",
        trial_id=trial_id,
        qid=qid,
        parent_state_digest=base.parent_vps_id,
        arm_context_digest=context.context_digest,
        operation_context_digest=operation_context.context_digest,
        operation_id=operation_context.operation_id,
        operation_input_digest=operation_context.operation_input_digest,
        holder_role_id=base.dealer_role_id,
        evaluation_point=base.dealer_eval_point,
        operation_base_hex=operation_context.operation_base_hex,
        operation_vk_base_hex=operation_context.operation_vk_base_hex,
        operation_vk_hex=operation_pvr.operation_vk,
        operation_share_hex=operation_pvr.operation_share,
    )


def make_ru22_operation_contribution_batch(
    *,
    context: RU22Context,
    operation_context: ThresholdOperationContext,
    trial_id: str,
    holder_role_id: int,
    holder_eval_point: int,
    recipient_index: int,
    recipient_secret_key: int,
    local_vps_shares: Sequence[RU22LocalVPSShare],
    metrics: RU22OperationHolderMetrics | None = None,
) -> RU22OperationContributionBatch:
    """Generate base and operation material from one in-memory private take."""

    if not isinstance(operation_context, ThresholdOperationContext):
        raise TypeError("operation_context must be a ThresholdOperationContext")
    timings = metrics if metrics is not None else RU22OperationHolderMetrics()
    if not isinstance(timings, RU22OperationHolderMetrics):
        raise TypeError("metrics must be RU22OperationHolderMetrics")
    base_batch = make_ru22_contribution_batch(
        context=context,
        holder_role_id=holder_role_id,
        holder_eval_point=holder_eval_point,
        recipient_index=recipient_index,
        recipient_secret_key=recipient_secret_key,
        local_vps_shares=local_vps_shares,
        metrics=timings.base,
    )
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    base_batch_bytes = base_batch.to_bytes()
    timings.base.serialization_seconds += time.perf_counter() - wall_started
    timings.base.serialization_cpu_seconds += time.process_time() - cpu_started
    timings.base.batch_bytes = len(base_batch_bytes)
    qid = build_ru22_operation_qid(
        context, operation_context, trial_id=trial_id
    ).encode()
    local_by_parent = {entry.vps.vps_id: entry for entry in local_vps_shares}
    operation_entries = []
    for base_pvr in base_batch.pvr_entries:
        local = local_by_parent[base_pvr.parent_vps_id]

        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        operation_statement, operation_proof = (
            evaluate_and_prove_vk_operation_share(
                protocol="ru22",
                trial_id=trial_id,
                qid=qid,
                parent_state_digest=base_pvr.parent_vps_id,
                arm_context_digest=context.context_digest,
                operation_context=operation_context,
                holder_role_id=holder_role_id,
                evaluation_point=holder_eval_point,
                share=local.share_s,
            )
        )
        timings.operation_proof_seconds += time.perf_counter() - wall_started
        timings.operation_proof_cpu_seconds += time.process_time() - cpu_started

        link_statement = _link_statement(
            context=context,
            operation_context=operation_context,
            qid=qid,
            parent=local.vps,
            base_pvr=base_pvr,
            recipient_index=recipient_index,
            operation_vk=operation_statement.operation_vk_hex,
        )
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        link_proof = prove_ru22_encrypted_commitment_vk(
            link_statement,
            recipient_secret_key=recipient_secret_key,
            share=local.share_s,
            blind_share=local.share_s_hat,
        )
        timings.link_proof_seconds += time.perf_counter() - wall_started
        timings.link_proof_cpu_seconds += time.process_time() - cpu_started
        operation_entries.append(
            RU22OperationPVR(
                base_pvr=base_pvr,
                operation_vk=operation_statement.operation_vk_hex,
                encrypted_commitment_vk_proof=link_proof,
                operation_share=operation_statement.operation_share_hex,
                vk_operation_share_proof=operation_proof,
            )
        )

    batch = RU22OperationContributionBatch(
        qid=qid,
        base_context_digest=context.context_digest,
        operation_context_digest=operation_context.context_digest,
        trial_id=trial_id,
        holder_role_id=holder_role_id,
        holder_eval_point=holder_eval_point,
        pvr_entries=operation_entries,
    )
    timings.pvr_count = len(operation_entries)
    timings.link_proof_bytes = sum(
        len(item.encrypted_commitment_vk_proof.to_bytes())
        for item in operation_entries
    )
    timings.operation_proof_bytes = sum(
        len(item.vk_operation_share_proof.to_bytes())
        for item in operation_entries
    )
    timings.operation_vk_bytes = sum(
        len(item.operation_vk.encode("ascii")) for item in operation_entries
    )
    timings.operation_share_bytes = sum(
        len(item.operation_share.encode("ascii")) for item in operation_entries
    )
    timings.operation_pvr_bytes = sum(
        len(item.to_bytes()) for item in operation_entries
    )
    return batch


def consume_ru22_operation_handover_role(
    *,
    context: RU22Context,
    operation_context: ThresholdOperationContext,
    trial_id: str,
    holder_state: RU22HolderState,
) -> RU22OperationHolderActionResult:
    """Atomically consume one holder and serialize its application batch."""

    if not isinstance(holder_state, RU22HolderState):
        raise TypeError("holder_state must be an RU22HolderState")
    holder_state.validate_public(context)
    (
        holder_role_id,
        holder_eval_point,
        recipient_index,
        recipient_secret_key,
        local_vps_shares,
    ) = holder_state.consume()
    metrics = RU22OperationHolderMetrics()
    batch = make_ru22_operation_contribution_batch(
        context=context,
        operation_context=operation_context,
        trial_id=trial_id,
        holder_role_id=holder_role_id,
        holder_eval_point=holder_eval_point,
        recipient_index=recipient_index,
        recipient_secret_key=recipient_secret_key,
        local_vps_shares=local_vps_shares,
        metrics=metrics,
    )
    del recipient_secret_key, local_vps_shares
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    payload = batch.to_bytes()
    metrics.serialization_seconds = time.perf_counter() - wall_started
    metrics.serialization_cpu_seconds = time.process_time() - cpu_started
    metrics.batch_bytes = len(payload)
    return RU22OperationHolderActionResult(
        qid=batch.qid,
        mid=f"node_{holder_role_id}",
        batch=batch,
        batch_bytes=payload,
        metrics=metrics,
    )


@dataclass(frozen=True)
class RU22ValidatedOperationBatch:
    entry: PoolEntry
    batch: RU22OperationContributionBatch
    base_batch: RU22ContributionBatch
    valid_pvrs: tuple[RU22OperationPVR, ...]


@dataclass(frozen=True)
class RU22OperationJustifiedSelection:
    base_selection: RU22JustifiedSelection
    operation_pvrs: tuple[RU22OperationPVR, ...]

    @property
    def parent_vps_id(self) -> str:
        return self.base_selection.parent_vps_id

    @property
    def entries(self) -> tuple[PoolEntry, ...]:
        return self.base_selection.entries

    @property
    def selected_role_ids(self) -> tuple[int, ...]:
        return self.base_selection.selected_role_ids

    @property
    def selected_eval_points(self) -> tuple[int, ...]:
        return self.base_selection.selected_eval_points

    @property
    def lagrange_coefficients(self) -> tuple[int, ...]:
        return self.base_selection.lagrange_coefficients


@dataclass(frozen=True)
class _RU22OperationRegistration:
    context: RU22Context
    operation_context: ThresholdOperationContext
    trial_id: str
    qid: str


def _normalize_entry(value: Any) -> PoolEntry | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    mid, payload = value
    if not isinstance(mid, str) or not mid:
        return None
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        return None
    return mid, bytes(payload)


def _entry_sort_key(value: PoolEntry) -> tuple[str, str]:
    return value[0], value[1].hex()


class RU22OperationJustifierManager(Justifier):
    """Uncached formal verifier for complete RU22 operation tuples."""

    kind = RU22_OPERATION_QID_KIND

    def __init__(self) -> None:
        self._registrations: dict[str, _RU22OperationRegistration] = {}
        self._base = RU22DPSSJustifierManager()
        self._verified: dict[
            tuple[str, tuple[PoolEntry, ...]],
            RU22OperationJustifiedSelection,
        ] = {}
        self._metrics = {
            name: {
                "invocations": 0,
                "elapsed_sec": 0.0,
                "cpu_seconds": 0.0,
            }
            for name in ("schema_base", "link_proof", "operation_proof")
        }
        self._formal_invocations = 0

    def register_context(
        self,
        context: RU22Context,
        operation_context: ThresholdOperationContext,
        *,
        trial_id: str,
        parent_vps_catalog: Sequence[RU22VPS],
    ) -> str:
        trial = _nonempty_text(trial_id, name="trial_id")
        qid = build_ru22_operation_qid(
            context, operation_context, trial_id=trial
        ).encode()
        registration = _RU22OperationRegistration(
            context=context,
            operation_context=operation_context,
            trial_id=trial,
            qid=qid,
        )
        existing = self._registrations.get(qid)
        if existing is not None and existing != registration:
            raise ValueError("conflicting RU22 operation registration")
        self._registrations[qid] = registration
        self._base.register_context(context, parent_vps_catalog)
        for key in tuple(self._verified):
            if key[0] == qid:
                self._verified.pop(key, None)
        return qid

    def _registration(self, qid: Qid | str) -> _RU22OperationRegistration:
        encoded = qid.encode() if isinstance(qid, Qid) else qid
        if not isinstance(encoded, str):
            raise TypeError("qid must be a Qid or string")
        parsed = Qid.decode(encoded)
        if parsed.encode() != encoded or parsed.kind != RU22_OPERATION_QID_KIND:
            raise ValueError("expected a canonical RU22 operation qid")
        try:
            registration = self._registrations[encoded]
        except KeyError as exc:
            raise KeyError("unknown RU22 operation qid") from exc
        expected = build_ru22_operation_qid(
            registration.context,
            registration.operation_context,
            trial_id=registration.trial_id,
        ).encode()
        if not hmac.compare_digest(encoded, expected):
            raise ValueError("RU22 operation qid does not match registration")
        return registration

    def _record(self, stage: str, wall_started: float, cpu_started: float) -> None:
        metric = self._metrics[stage]
        metric["invocations"] += 1
        metric["elapsed_sec"] += time.perf_counter() - wall_started
        metric["cpu_seconds"] += time.process_time() - cpu_started

    def metrics_snapshot(self) -> dict[str, Any]:
        return {
            "formal_validation_mode": "uncached-per-candidate-invocation",
            "formal_invocations": self._formal_invocations,
            "formal_cache_hits": 0,
            **{
                stage: {
                    "invocations": int(metric["invocations"]),
                    "elapsed_sec": float(metric["elapsed_sec"]),
                    "cpu_seconds": float(metric["cpu_seconds"]),
                }
                for stage, metric in self._metrics.items()
            },
        }

    def _validate_batch_uncached(
        self,
        registration: _RU22OperationRegistration,
        mid: str,
        payload: bytes,
    ) -> RU22ValidatedOperationBatch | None:
        context = registration.context
        operation_context = registration.operation_context
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        try:
            batch = RU22OperationContributionBatch.from_bytes(payload)
            if (
                batch.qid != registration.qid
                or batch.base_context_digest != context.context_digest
                or batch.operation_context_digest
                != operation_context.context_digest
                or batch.trial_id != registration.trial_id
                or mid != f"node_{batch.holder_role_id}"
                or context.holder_eval_points.get(batch.holder_role_id)
                != batch.holder_eval_point
            ):
                raise ValueError("operation batch identity mismatch")
            base_batch = batch.to_base_batch()
            base_validated = self._base._validate_batch_uncached(
                context,
                mid,
                base_batch.to_bytes(),
            )
            if base_validated is None:
                raise ValueError("base RU22 batch is invalid")
        except Exception:
            self._record("schema_base", wall_started, cpu_started)
            return None
        self._record("schema_base", wall_started, cpu_started)

        base_valid_digests = {
            pvr.pvr_digest for pvr in base_validated.valid_pvrs
        }
        parent_catalog = self._base.parent_catalogs.get(
            context.context_digest, {}
        )
        valid_operation_pvrs = []
        for operation_pvr in batch.pvr_entries:
            base_pvr = operation_pvr.base_pvr
            if base_pvr.pvr_digest not in base_valid_digests:
                continue
            try:
                parent = parent_catalog[base_pvr.parent_vps_id]
                recipient_index = context.source_role_indices[
                    base_pvr.dealer_role_id
                ]
                link_statement = _link_statement(
                    context=context,
                    operation_context=operation_context,
                    qid=registration.qid,
                    parent=parent,
                    base_pvr=base_pvr,
                    recipient_index=recipient_index,
                    operation_vk=operation_pvr.operation_vk,
                )
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            wall_started = time.perf_counter()
            cpu_started = time.process_time()
            link_valid = verify_ru22_encrypted_commitment_vk(
                link_statement,
                operation_pvr.encrypted_commitment_vk_proof,
            )
            self._record("link_proof", wall_started, cpu_started)
            if not link_valid:
                continue
            try:
                operation_statement = _operation_statement(
                    context=context,
                    operation_context=operation_context,
                    trial_id=registration.trial_id,
                    qid=registration.qid,
                    operation_pvr=operation_pvr,
                )
            except (TypeError, ValueError):
                continue
            wall_started = time.perf_counter()
            cpu_started = time.process_time()
            operation_valid = verify_vk_operation_share(
                operation_statement,
                operation_pvr.vk_operation_share_proof,
            )
            self._record("operation_proof", wall_started, cpu_started)
            if operation_valid:
                valid_operation_pvrs.append(operation_pvr)

        return RU22ValidatedOperationBatch(
            entry=(mid, bytes(payload)),
            batch=batch,
            base_batch=base_batch,
            valid_pvrs=tuple(valid_operation_pvrs),
        )

    @staticmethod
    def _derive_from_validated(
        context: RU22Context,
        validated: Sequence[RU22ValidatedOperationBatch],
    ) -> RU22OperationJustifiedSelection | None:
        base_validated = [
            RU22ValidatedBatch(
                entry=item.entry,
                batch=item.base_batch,
                valid_pvrs=tuple(pvr.base_pvr for pvr in item.valid_pvrs),
            )
            for item in validated
        ]
        base_selection = RU22DPSSJustifierManager._derive_from_validated(
            context, base_validated
        )
        if base_selection is None:
            return None
        by_identity = {
            (item.entry, pvr.base_pvr.pvr_digest): pvr
            for item in validated
            for pvr in item.valid_pvrs
        }
        try:
            operation_pvrs = tuple(
                by_identity[(entry, base_pvr.pvr_digest)]
                for entry, base_pvr in zip(
                    base_selection.entries, base_selection.pvrs
                )
            )
        except KeyError:
            return None
        return RU22OperationJustifiedSelection(
            base_selection=base_selection,
            operation_pvrs=operation_pvrs,
        )

    def _validated_candidates(
        self,
        registration: _RU22OperationRegistration,
        entries: Sequence[PoolEntry],
        *,
        reject_malformed: bool,
    ) -> list[RU22ValidatedOperationBatch] | None:
        validated = []
        normalized = []
        for entry in entries:
            item = _normalize_entry(entry)
            if item is None:
                if reject_malformed:
                    return None
                continue
            normalized.append(item)
        for mid, payload in sorted(normalized, key=_entry_sort_key):
            item = self._validate_batch_uncached(registration, mid, payload)
            if item is None:
                if reject_malformed:
                    return None
                continue
            validated.append(item)
        return validated

    @staticmethod
    def _exact_entries(
        entries: Sequence[PoolEntry],
    ) -> tuple[PoolEntry, ...] | None:
        normalized = []
        try:
            for entry in entries:
                item = _normalize_entry(entry)
                if item is None:
                    return None
                normalized.append(item)
        except TypeError:
            return None
        return tuple(sorted(normalized, key=_entry_sort_key))

    def derive_selection(
        self,
        qid: Qid | str,
        entries: Sequence[PoolEntry],
    ) -> RU22OperationJustifiedSelection | None:
        try:
            registration = self._registration(qid)
        except (KeyError, TypeError, ValueError):
            return None
        threshold = registration.context.resharing_threshold + 1
        if len(entries) != threshold:
            return None
        validated = self._validated_candidates(
            registration, entries, reject_malformed=True
        )
        if validated is None or len(validated) != threshold:
            return None
        selection = self._derive_from_validated(registration.context, validated)
        supplied = self._exact_entries(entries)
        if selection is None or supplied is None:
            return None
        if tuple(sorted(selection.entries, key=_entry_sort_key)) != supplied:
            return None
        return selection

    def select_justified_subset(
        self,
        qid: Qid,
        entries: Sequence[PoolEntry],
    ) -> list[PoolEntry] | None:
        try:
            registration = self._registration(qid)
        except (KeyError, TypeError, ValueError):
            return None
        self._formal_invocations += 1
        validated = self._validated_candidates(
            registration, entries, reject_malformed=False
        )
        if validated is None:
            return None
        selection = self._derive_from_validated(registration.context, validated)
        return list(selection.entries) if selection is not None else None

    def verify_qid_set(
        self,
        qid: Qid,
        entries: Sequence[PoolEntry],
    ) -> bool:
        self._formal_invocations += 1
        selection = self.derive_selection(qid, entries)
        encoded = qid.encode() if isinstance(qid, Qid) else str(qid)
        normalized = self._exact_entries(entries)
        if normalized is not None:
            key = (encoded, normalized)
            if selection is None:
                self._verified.pop(key, None)
            else:
                self._verified[key] = selection
        return selection is not None

    def verified_selection(
        self,
        qid: Qid | str,
        entries: Sequence[PoolEntry],
    ) -> RU22OperationJustifiedSelection | None:
        try:
            registration = self._registration(qid)
        except (KeyError, TypeError, ValueError):
            return None
        normalized = self._exact_entries(entries)
        if normalized is None:
            return None
        return self._verified.get((registration.qid, normalized))


def _text_tuple(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty sequence")
    result = tuple(_nonempty_text(item, name=f"{name} item") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be distinct")
    return result


def _digest_tuple(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty sequence")
    result = tuple(_digest(item, name=f"{name} item") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be distinct")
    return result


def _int_tuple(
    value: Sequence[int], *, name: str, minimum: int
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty sequence")
    result = tuple(
        _strict_int(item, name=f"{name} item", minimum=minimum)
        for item in value
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be distinct")
    return result


def _point_tuple(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty sequence")
    return tuple(_point(item, name=f"{name} item") for item in value)


@dataclass(frozen=True)
class RU22OperationPublicResult:
    qid: str | Qid
    base_context_digest: str
    operation_context_digest: str
    trial_id: str
    parent_vps_id: str
    selected_mids: Sequence[str]
    selected_role_ids: Sequence[int]
    selected_eval_points: Sequence[int]
    lagrange_coefficients: Sequence[int]
    selected_base_pvr_digests: Sequence[str]
    selected_operation_pvr_digests: Sequence[str]
    selected_operation_vks: Sequence[str]
    selected_operation_shares: Sequence[str]
    combined_operation_vk: str
    operation_output: str
    operation_output_digest: str
    new_vps: RU22VPS | Mapping[str, Any]
    result_digest: str = field(init=False)

    def __post_init__(self) -> None:
        base_digest = _digest(
            self.base_context_digest, name="base_context_digest"
        )
        operation_digest = _digest(
            self.operation_context_digest,
            name="operation_context_digest",
        )
        trial = _nonempty_text(self.trial_id, name="trial_id")
        qid = _canonical_operation_qid(
            self.qid,
            base_context_digest=base_digest,
            operation_context_digest=operation_digest,
            trial_id=trial,
        )
        parent_id = _digest(self.parent_vps_id, name="parent_vps_id")
        mids = _text_tuple(self.selected_mids, name="selected_mids")
        role_ids = _int_tuple(
            self.selected_role_ids, name="selected_role_ids", minimum=0
        )
        eval_points = _int_tuple(
            self.selected_eval_points,
            name="selected_eval_points",
            minimum=1,
        )
        if not isinstance(self.lagrange_coefficients, (list, tuple)):
            raise TypeError("lagrange_coefficients must be a sequence")
        coefficients = tuple(
            _scalar(item, name="lagrange coefficient")
            for item in self.lagrange_coefficients
        )
        base_pvr_digests = _digest_tuple(
            self.selected_base_pvr_digests,
            name="selected_base_pvr_digests",
        )
        operation_pvr_digests = _digest_tuple(
            self.selected_operation_pvr_digests,
            name="selected_operation_pvr_digests",
        )
        vks = _point_tuple(
            self.selected_operation_vks,
            name="selected_operation_vks",
        )
        operation_shares = _point_tuple(
            self.selected_operation_shares,
            name="selected_operation_shares",
        )
        width = len(mids)
        if not all(
            len(value) == width
            for value in (
                role_ids,
                eval_points,
                coefficients,
                base_pvr_digests,
                operation_pvr_digests,
                vks,
                operation_shares,
            )
        ):
            raise ValueError("RU22 operation result selection widths differ")
        combined_vk = _point(
            self.combined_operation_vk, name="combined_operation_vk"
        )
        output = _point(self.operation_output, name="operation_output")
        output_digest = _digest(
            self.operation_output_digest,
            name="operation_output_digest",
        )
        new_vps = (
            RU22VPS.from_bytes(self.new_vps.to_bytes())
            if isinstance(self.new_vps, RU22VPS)
            else RU22VPS.from_obj(self.new_vps)
        )
        if not isinstance(new_vps.certificate, RU22DAGDeliveryCertificate):
            raise ValueError("operation result new VPS requires a DAG certificate")
        if new_vps.certificate.qid != qid:
            raise ValueError("new VPS certificate qid does not match operation qid")
        if set(new_vps.certificate.selected_mids) != set(mids):
            raise ValueError("new VPS certificate mids do not match selection")
        for name, value in (
            ("qid", qid),
            ("base_context_digest", base_digest),
            ("operation_context_digest", operation_digest),
            ("trial_id", trial),
            ("parent_vps_id", parent_id),
            ("selected_mids", mids),
            ("selected_role_ids", role_ids),
            ("selected_eval_points", eval_points),
            ("lagrange_coefficients", coefficients),
            ("selected_base_pvr_digests", base_pvr_digests),
            ("selected_operation_pvr_digests", operation_pvr_digests),
            ("selected_operation_vks", vks),
            ("selected_operation_shares", operation_shares),
            ("combined_operation_vk", combined_vk),
            ("operation_output", output),
            ("operation_output_digest", output_digest),
            ("new_vps", new_vps),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "result_digest",
            _domain_digest(
                _PUBLIC_RESULT_DIGEST_DOMAIN,
                self.digest_material(),
            ),
        )

    def digest_material(self) -> dict[str, Any]:
        return {
            "schema": OPERATION_PUBLIC_RESULT_SCHEMA,
            "qid": self.qid,
            "base_context_digest": self.base_context_digest,
            "operation_context_digest": self.operation_context_digest,
            "trial_id": self.trial_id,
            "parent_vps_id": self.parent_vps_id,
            "selected_mids": list(self.selected_mids),
            "selected_role_ids": list(self.selected_role_ids),
            "selected_eval_points": list(self.selected_eval_points),
            "lagrange_coefficients": [
                str(value) for value in self.lagrange_coefficients
            ],
            "selected_base_pvr_digests": list(
                self.selected_base_pvr_digests
            ),
            "selected_operation_pvr_digests": list(
                self.selected_operation_pvr_digests
            ),
            "selected_operation_vks": list(self.selected_operation_vks),
            "selected_operation_shares": list(
                self.selected_operation_shares
            ),
            "combined_operation_vk": self.combined_operation_vk,
            "operation_output": self.operation_output,
            "operation_output_digest": self.operation_output_digest,
            "new_vps": self.new_vps.to_obj(),
        }

    def to_obj(self) -> dict[str, Any]:
        value = self.digest_material()
        value["result_digest"] = self.result_digest
        return value

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "RU22OperationPublicResult":
        expected = {
            "schema",
            "qid",
            "base_context_digest",
            "operation_context_digest",
            "trial_id",
            "parent_vps_id",
            "selected_mids",
            "selected_role_ids",
            "selected_eval_points",
            "lagrange_coefficients",
            "selected_base_pvr_digests",
            "selected_operation_pvr_digests",
            "selected_operation_vks",
            "selected_operation_shares",
            "combined_operation_vk",
            "operation_output",
            "operation_output_digest",
            "new_vps",
            "result_digest",
        }
        _exact_fields(value, expected, name="RU22 operation public result")
        if value["schema"] != OPERATION_PUBLIC_RESULT_SCHEMA:
            raise ValueError("unsupported RU22 operation public-result schema")
        if not isinstance(value["lagrange_coefficients"], (list, tuple)):
            raise TypeError("lagrange_coefficients must be a sequence")
        supplied_digest = _digest(value["result_digest"], name="result_digest")
        result = cls(
            qid=value["qid"],
            base_context_digest=value["base_context_digest"],
            operation_context_digest=value["operation_context_digest"],
            trial_id=value["trial_id"],
            parent_vps_id=value["parent_vps_id"],
            selected_mids=value["selected_mids"],
            selected_role_ids=value["selected_role_ids"],
            selected_eval_points=value["selected_eval_points"],
            lagrange_coefficients=[
                _scalar_text(item, name="lagrange coefficient")
                for item in value["lagrange_coefficients"]
            ],
            selected_base_pvr_digests=value["selected_base_pvr_digests"],
            selected_operation_pvr_digests=value[
                "selected_operation_pvr_digests"
            ],
            selected_operation_vks=value["selected_operation_vks"],
            selected_operation_shares=value["selected_operation_shares"],
            combined_operation_vk=value["combined_operation_vk"],
            operation_output=value["operation_output"],
            operation_output_digest=value["operation_output_digest"],
            new_vps=value["new_vps"],
        )
        if not hmac.compare_digest(result.result_digest, supplied_digest):
            raise ValueError("RU22 operation result_digest mismatch")
        return result

    @classmethod
    def from_bytes(
        cls, value: bytes | bytearray | memoryview | str
    ) -> "RU22OperationPublicResult":
        return cls.from_obj(_decode_object(value, name="RU22 operation result"))


class RU22OperationPublicFinalizer:
    def __init__(self, justifier: RU22OperationJustifierManager) -> None:
        if not isinstance(justifier, RU22OperationJustifierManager):
            raise TypeError("justifier must be an RU22OperationJustifierManager")
        self.justifier = justifier
        self.results: dict[str, RU22OperationPublicResult] = {}

    def finalize_selection(
        self,
        qid: Qid | str,
        selection: RU22OperationJustifiedSelection,
        *,
        ordered_vertex: tuple[int, int] = (0, 0),
    ) -> RU22OperationPublicResult:
        registration = self.justifier._registration(qid)
        if not isinstance(selection, RU22OperationJustifiedSelection):
            raise TypeError("selection must be an RU22 operation selection")
        if (
            not isinstance(ordered_vertex, (list, tuple))
            or len(ordered_vertex) != 2
        ):
            raise ValueError("ordered_vertex must be a pair")
        vertex = (
            _strict_int(ordered_vertex[0], name="ordered vertex round"),
            _strict_int(ordered_vertex[1], name="ordered vertex source"),
        )
        context = registration.context
        operation_context = registration.operation_context
        base = selection.base_selection
        mids = tuple(entry[0] for entry in base.entries)
        certificate = RU22DAGDeliveryCertificate(
            qid=registration.qid,
            ordered_vertex=vertex,
            selected_mids=tuple(sorted(mids)),
        )
        new_vps = RU22VPS(
            epoch=context.target_epoch,
            committee=context.successor_committee,
            committee_context_digest=context.context_digest,
            encrypted_commitments=base.combined_c_v_vector,
            certificate=certificate,
        )
        operation_vks = tuple(pvr.operation_vk for pvr in selection.operation_pvrs)
        operation_shares = tuple(
            pvr.operation_share for pvr in selection.operation_pvrs
        )
        combined_vk = combine_vk_operation_shares(
            base.selected_eval_points, operation_vks
        )
        combined_output = combine_vk_operation_shares(
            base.selected_eval_points, operation_shares
        )
        if combined_vk != G1Point.from_hex(
            operation_context.maintained_global_public_key
        ):
            raise ValueError("selected operation vks do not match the maintained key")
        output_digest = operation_output_digest(
            operation_id=operation_context.operation_id,
            operation_input=operation_context.operation_input,
            operation_output=combined_output,
        )
        if not hmac.compare_digest(
            output_digest,
            operation_context.expected_operation_output_digest,
        ):
            raise ValueError("operation output does not match expected digest")
        result = RU22OperationPublicResult(
            qid=registration.qid,
            base_context_digest=context.context_digest,
            operation_context_digest=operation_context.context_digest,
            trial_id=registration.trial_id,
            parent_vps_id=base.parent_vps_id,
            selected_mids=mids,
            selected_role_ids=base.selected_role_ids,
            selected_eval_points=base.selected_eval_points,
            lagrange_coefficients=base.lagrange_coefficients,
            selected_base_pvr_digests=tuple(
                pvr.pvr_digest for pvr in base.pvrs
            ),
            selected_operation_pvr_digests=tuple(
                pvr.operation_pvr_digest for pvr in selection.operation_pvrs
            ),
            selected_operation_vks=operation_vks,
            selected_operation_shares=operation_shares,
            combined_operation_vk=combined_vk.to_hex(),
            operation_output=combined_output.to_hex(),
            operation_output_digest=output_digest,
            new_vps=new_vps,
        )
        existing = self.results.get(registration.qid)
        if existing is not None:
            if existing != result:
                raise ValueError("conflicting RU22 operation result")
            return existing
        self.results[registration.qid] = result
        return result

    def on_deliver(
        self,
        qid: Qid | str,
        entries: Sequence[PoolEntry],
        *,
        ordered_vertex: tuple[int, int] = (0, 0),
    ) -> RU22OperationPublicResult | None:
        selection = self.justifier.derive_selection(qid, entries)
        if selection is None:
            return None
        return self.finalize_selection(
            qid, selection, ordered_vertex=ordered_vertex
        )

    def on_verified_deliver(
        self,
        qid: Qid | str,
        entries: Sequence[PoolEntry],
        *,
        ordered_vertex: tuple[int, int] = (0, 0),
    ) -> RU22OperationPublicResult | None:
        """Consume the exact selection cached by DAG committed verification."""

        selection = self.justifier.verified_selection(qid, entries)
        if selection is None:
            return None
        return self.finalize_selection(
            qid, selection, ordered_vertex=ordered_vertex
        )


def reconstruct_ru22_operation_successor(
    *,
    context: RU22Context,
    result: RU22OperationPublicResult,
    selection: RU22OperationJustifiedSelection,
    my_node_id: int,
    my_role_idx: int,
    my_rsk: int,
) -> RU22LocalVPSShare:
    """Reconstruct only from the exact selected base PVRs in the result."""

    if not isinstance(context, RU22Context):
        raise TypeError("context must be an RU22Context")
    if not isinstance(result, RU22OperationPublicResult):
        raise TypeError("result must be an RU22OperationPublicResult")
    if not isinstance(selection, RU22OperationJustifiedSelection):
        raise TypeError("selection must be an RU22OperationJustifiedSelection")
    node_id = _strict_int(my_node_id, name="my_node_id")
    role_idx = _strict_int(my_role_idx, name="my_role_idx")
    rsk = _scalar(my_rsk, name="my_rsk", nonzero=True)
    if context.successor_role_indices.get(node_id) != role_idx:
        raise ValueError("successor node/role does not match context")
    if not 0 <= role_idx < context.resharing_n:
        raise ValueError("successor role index is out of range")
    if derive_public_key(
        rsk, context.successor_key_base_hex
    ) != context.successor_rpk_list[role_idx]:
        raise ValueError("my_rsk does not match successor public key")
    base = selection.base_selection
    if (
        result.base_context_digest != context.context_digest
        or result.parent_vps_id != base.parent_vps_id
        or result.selected_role_ids != base.selected_role_ids
        or result.selected_eval_points != base.selected_eval_points
        or result.lagrange_coefficients != base.lagrange_coefficients
        or result.selected_base_pvr_digests
        != tuple(pvr.pvr_digest for pvr in base.pvrs)
        or result.selected_operation_pvr_digests
        != tuple(
            pvr.operation_pvr_digest for pvr in selection.operation_pvrs
        )
    ):
        raise ValueError("result does not match exact RU22 operation selection")
    share_pairs = [
        pvss_deliver(
            context.g_hex,
            context.h_hex,
            pvr.pvss_transcript,
            rsk,
            role_idx,
        )
        for pvr in base.pvrs
    ]
    share_s, share_s_hat = pvss_interpolate(
        base.selected_eval_points,
        share_pairs,
        context.resharing_threshold,
    )
    commitment = pedersen_commit(
        context.g_hex, context.h_hex, share_s, share_s_hat
    )
    decrypted = decrypt_elgamal_ciphertext(
        rsk, result.new_vps.encrypted_commitments[role_idx].to_obj()
    )
    if G1Point.from_hex(commitment) != G1Point.from_hex(decrypted):
        raise ValueError("successor share does not open the new VPS coordinate")
    return RU22LocalVPSShare(
        vps=result.new_vps,
        share_s=share_s,
        share_s_hat=share_s_hat,
    )


__all__ = [
    "LINK_PROOF_KIND",
    "LINK_PROOF_VERSION",
    "LINK_STATEMENT_SCHEMA",
    "OPERATION_BATCH_SCHEMA",
    "OPERATION_PUBLIC_RESULT_SCHEMA",
    "RU22EncryptedCommitmentVKProof",
    "RU22EncryptedCommitmentVKStatement",
    "RU22OperationContributionBatch",
    "RU22OperationHolderActionResult",
    "RU22OperationHolderMetrics",
    "RU22OperationJustifiedSelection",
    "RU22OperationJustifierManager",
    "RU22OperationPVR",
    "RU22OperationPublicFinalizer",
    "RU22OperationPublicResult",
    "build_ru22_operation_qid",
    "consume_ru22_operation_handover_role",
    "make_ru22_operation_contribution_batch",
    "prove_ru22_encrypted_commitment_vk",
    "reconstruct_ru22_operation_successor",
    "verify_ru22_encrypted_commitment_vk",
]
