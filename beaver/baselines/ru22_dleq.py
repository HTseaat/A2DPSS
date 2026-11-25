"""RU22 old-VPS to new-PVR commitment binding proof.

For an ElGamal ciphertext ``c_v = (C1, C2)`` under
``PK = secret_key * key_base`` and a claimed PVR constant-term Pedersen
commitment ``V0``, this module proves the Chaum--Pedersen relation

    log_key_base(PK) = log_C1(C2 - V0).

The Fiat--Shamir statement binds the proof to the complete RU22 handover
identity.  The module is deliberately independent of PVR batching, DAG
selection, and successor reconstruction; those belong to later phases.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping

from beaver.chunk_elgamal import normalize_elgamal_ciphertext
from beaver.dag_tob import Qid
from beaver.g1_backend import (
    FIELD_ORDER,
    G1Point,
    field_add,
    field_mul,
    field_normalize,
    field_random,
)


PROOF_VERSION = 1
PROOF_KIND = "atcaas-ru22-old-to-new-dleq"
QID_KIND = "ru22-dpss"
FS_DOMAIN = b"ATCaaS/RU22/old-to-new-dleq/v1\x00"

_QID_FIELDS = frozenset(
    {
        "context_digest",
        "epoch",
        "sid",
        "source_committee",
        "successor_committee",
        "transfer_seq",
    }
)
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_json_object(
    value: bytes | bytearray | memoryview | str,
    *,
    name: str,
) -> dict:
    raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
    try:
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


def _strict_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _canonical_decimal(value: Any, *, name: str) -> int:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a canonical decimal string")
    if value != "0" and (
        not value.isdigit() or value.startswith("0")
    ):
        raise ValueError(f"{name} must be a canonical decimal string")
    if value == "0" or value.isdigit():
        scalar = int(value)
    else:
        raise ValueError(f"{name} must be a canonical decimal string")
    if not 0 <= scalar < FIELD_ORDER:
        raise ValueError(f"{name} must be a canonical field scalar")
    return scalar


def _digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return value


def _point(
    value: str | G1Point,
    *,
    name: str,
    nonidentity: bool,
) -> G1Point:
    try:
        point = value if isinstance(value, G1Point) else G1Point.from_hex(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid G1 point") from exc
    if nonidentity and point.is_identity():
        raise ValueError(f"{name} must not be the identity point")
    return point


def _canonical_qid(
    value: Qid | str,
    *,
    context_digest: str,
    source_epoch: int,
) -> str:
    if isinstance(value, Qid):
        qid = value
        encoded = qid.encode()
    elif isinstance(value, str):
        qid = Qid.decode(value)
        encoded = value
        if qid.encode() != value:
            raise ValueError("qid must use canonical Qid encoding")
    else:
        raise TypeError("qid must be a Qid or canonical encoded string")
    if qid.kind != QID_KIND:
        raise ValueError(f"qid kind must be {QID_KIND!r}")
    fields = dict(qid.fields)
    if set(fields) != _QID_FIELDS:
        raise ValueError("qid fields do not match the RU22 Phase-0 contract")
    if fields["context_digest"] != context_digest:
        raise ValueError("qid context_digest does not match statement")
    if fields["epoch"] != str(source_epoch):
        raise ValueError("qid epoch does not match source_epoch")
    if not fields["sid"] or not fields["source_committee"] or not fields[
        "successor_committee"
    ]:
        raise ValueError("qid service and committee fields must be non-empty")
    transfer_seq = fields["transfer_seq"]
    if transfer_seq != "0" and (
        not transfer_seq.isdigit() or transfer_seq.startswith("0")
    ):
        raise ValueError("qid transfer_seq must be a canonical non-negative integer")
    return encoded


def _normalized_statement(
    *,
    qid: Qid | str,
    context_digest: str,
    source_epoch: int,
    target_epoch: int,
    parent_vps_id: str,
    dealer_role_id: int,
    dealer_eval_point: int,
    recipient_index: int,
    key_base_hex: str | G1Point,
    recipient_public_key_hex: str | G1Point,
    c_v: Mapping[str, str],
    constant_term_commitment_hex: str | G1Point,
) -> tuple[dict, G1Point, G1Point, G1Point, G1Point]:
    context = _digest(context_digest, name="context_digest")
    source = _strict_int(source_epoch, name="source_epoch", minimum=1)
    target = _strict_int(target_epoch, name="target_epoch", minimum=2)
    if target != source + 1:
        raise ValueError("target_epoch must equal source_epoch + 1")
    parent = _digest(parent_vps_id, name="parent_vps_id")
    role_id = _strict_int(dealer_role_id, name="dealer_role_id")
    eval_point = _strict_int(
        dealer_eval_point,
        name="dealer_eval_point",
        minimum=1,
    )
    recipient = _strict_int(recipient_index, name="recipient_index")
    qid_text = _canonical_qid(
        qid,
        context_digest=context,
        source_epoch=source,
    )

    key_base = _point(
        key_base_hex,
        name="key_base",
        nonidentity=True,
    )
    public_key = _point(
        recipient_public_key_hex,
        name="recipient_public_key",
        nonidentity=True,
    )
    normalized_c_v = normalize_elgamal_ciphertext(dict(c_v))
    c1 = _point(
        normalized_c_v["C1"],
        name="c_v.C1",
        nonidentity=True,
    )
    c2 = _point(
        normalized_c_v["C2"],
        name="c_v.C2",
        nonidentity=False,
    )
    constant_commitment = _point(
        constant_term_commitment_hex,
        name="constant_term_commitment",
        nonidentity=False,
    )
    masked_commitment = c2 - constant_commitment
    if masked_commitment.is_identity():
        raise ValueError("c_v masked commitment must not be the identity point")

    statement = {
        "kind": PROOF_KIND,
        "version": PROOF_VERSION,
        "qid": qid_text,
        "context_digest": context,
        "source_epoch": source,
        "target_epoch": target,
        "parent_vps_id": parent,
        "dealer_role_id": role_id,
        "dealer_eval_point": eval_point,
        "recipient_index": recipient,
        "key_base": key_base.to_hex(),
        "recipient_public_key": public_key.to_hex(),
        "c_v": normalized_c_v,
        "constant_term_commitment": constant_commitment.to_hex(),
        "masked_commitment": masked_commitment.to_hex(),
    }
    return statement, key_base, public_key, c1, masked_commitment


def old_to_new_dleq_statement(**kwargs) -> dict:
    """Return the normalized public Fiat--Shamir statement."""

    statement, _key_base, _public_key, _c1, _masked = (
        _normalized_statement(**kwargs)
    )
    return statement


def _challenge(
    statement: Mapping[str, Any],
    announcement_key: G1Point,
    announcement_ciphertext: G1Point,
) -> int:
    material = {
        "statement": dict(statement),
        "announcement_key": announcement_key.to_hex(),
        "announcement_ciphertext": announcement_ciphertext.to_hex(),
    }
    challenge = int.from_bytes(
        hashlib.sha256(FS_DOMAIN + _canonical_json(material)).digest(),
        byteorder="big",
    ) % FIELD_ORDER
    return challenge or 1


@dataclass(frozen=True)
class RU22OldToNewDLEQProof:
    """Canonical Chaum--Pedersen proof wire object."""

    announcement_key: str
    announcement_ciphertext: str
    response: int

    def __post_init__(self) -> None:
        announcement_key = _point(
            self.announcement_key,
            name="announcement_key",
            nonidentity=True,
        )
        announcement_ciphertext = _point(
            self.announcement_ciphertext,
            name="announcement_ciphertext",
            nonidentity=True,
        )
        response = _strict_int(self.response, name="response")
        if response >= FIELD_ORDER:
            raise ValueError("response must be a canonical field scalar")
        object.__setattr__(
            self,
            "announcement_key",
            announcement_key.to_hex(),
        )
        object.__setattr__(
            self,
            "announcement_ciphertext",
            announcement_ciphertext.to_hex(),
        )
        object.__setattr__(self, "response", response)

    def to_obj(self) -> dict:
        return {
            "kind": PROOF_KIND,
            "version": PROOF_VERSION,
            "announcement_key": self.announcement_key,
            "announcement_ciphertext": self.announcement_ciphertext,
            "response": str(self.response),
        }

    def to_bytes(self) -> bytes:
        return _canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "RU22OldToNewDLEQProof":
        if not isinstance(value, Mapping):
            raise ValueError("RU22 DLEQ proof must be an object")
        expected = {
            "kind",
            "version",
            "announcement_key",
            "announcement_ciphertext",
            "response",
        }
        missing = expected - set(value)
        extra = set(value) - expected
        if missing:
            raise ValueError(
                f"RU22 DLEQ proof is missing fields: {sorted(missing)}"
            )
        if extra:
            raise ValueError(
                f"RU22 DLEQ proof has unknown fields: {sorted(extra)}"
            )
        if value["kind"] != PROOF_KIND:
            raise ValueError("unexpected RU22 DLEQ proof kind")
        if (
            isinstance(value["version"], bool)
            or value["version"] != PROOF_VERSION
        ):
            raise ValueError("unsupported RU22 DLEQ proof version")
        return cls(
            announcement_key=value["announcement_key"],
            announcement_ciphertext=value["announcement_ciphertext"],
            response=_canonical_decimal(
                value["response"],
                name="response",
            ),
        )

    @classmethod
    def from_bytes(
        cls,
        value: bytes | bytearray | memoryview | str,
    ) -> "RU22OldToNewDLEQProof":
        return cls.from_obj(
            _load_json_object(value, name="RU22 DLEQ proof")
        )


def prove_old_to_new_dleq(
    *,
    recipient_secret_key: int,
    **statement_kwargs,
) -> RU22OldToNewDLEQProof:
    """Prove that ``c_v`` decrypts to the new constant commitment."""

    if (
        isinstance(recipient_secret_key, bool)
        or not isinstance(recipient_secret_key, int)
    ):
        raise TypeError("recipient_secret_key must be an integer")
    if not 0 < recipient_secret_key < FIELD_ORDER:
        raise ValueError(
            "recipient_secret_key must be a canonical non-zero field scalar"
        )
    secret_key = recipient_secret_key
    statement, key_base, public_key, c1, masked_commitment = (
        _normalized_statement(**statement_kwargs)
    )

    # Do not add local witness-consistency exponentiations here.  The
    # Chaum--Pedersen prover itself costs exactly the two scalar
    # multiplications below, matching the RU22/ATCaaS accounting.  A stale
    # secret key or mismatched plaintext deterministically produces a proof
    # that the public verifier rejects.
    blinding = field_random()
    announcement_key = key_base * blinding
    announcement_ciphertext = c1 * blinding
    challenge = _challenge(
        statement,
        announcement_key,
        announcement_ciphertext,
    )
    return RU22OldToNewDLEQProof(
        announcement_key=announcement_key.to_hex(),
        announcement_ciphertext=announcement_ciphertext.to_hex(),
        response=field_add(
            blinding,
            field_mul(challenge, secret_key),
        ),
    )


def verify_old_to_new_dleq(
    *,
    proof: RU22OldToNewDLEQProof | Mapping[str, Any],
    **statement_kwargs,
) -> bool:
    """Verify the RU22 binding proof; malformed adversarial input is false."""

    try:
        parsed = (
            proof
            if isinstance(proof, RU22OldToNewDLEQProof)
            else RU22OldToNewDLEQProof.from_obj(proof)
        )
        statement, key_base, public_key, c1, masked_commitment = (
            _normalized_statement(**statement_kwargs)
        )
        announcement_key = _point(
            parsed.announcement_key,
            name="announcement_key",
            nonidentity=True,
        )
        announcement_ciphertext = _point(
            parsed.announcement_ciphertext,
            name="announcement_ciphertext",
            nonidentity=True,
        )
        response = _strict_int(parsed.response, name="response")
        if response >= FIELD_ORDER:
            return False
        challenge = _challenge(
            statement,
            announcement_key,
            announcement_ciphertext,
        )
        key_relation = (
            key_base * response
            == announcement_key + public_key * challenge
        )
        ciphertext_relation = (
            c1 * response
            == announcement_ciphertext + masked_commitment * challenge
        )
        return bool(key_relation and ciphertext_relation)
    except (KeyError, TypeError, ValueError):
        return False


def proof_size_bytes(
    proof: RU22OldToNewDLEQProof | Mapping[str, Any],
) -> int:
    parsed = (
        proof
        if isinstance(proof, RU22OldToNewDLEQProof)
        else RU22OldToNewDLEQProof.from_obj(proof)
    )
    return len(parsed.to_bytes())


def timed_prove_old_to_new_dleq(
    **kwargs,
) -> tuple[RU22OldToNewDLEQProof, dict]:
    """Generate one proof and return experiment-facing timing/size metrics."""

    started = time.perf_counter()
    proof = prove_old_to_new_dleq(**kwargs)
    elapsed = time.perf_counter() - started
    return proof, {
        "proof_gen_ms": elapsed * 1000.0,
        "proof_bytes": proof_size_bytes(proof),
    }


def timed_verify_old_to_new_dleq(
    **kwargs,
) -> tuple[bool, dict]:
    """Verify one proof and return experiment-facing timing metrics."""

    started = time.perf_counter()
    accepted = verify_old_to_new_dleq(**kwargs)
    elapsed = time.perf_counter() - started
    return accepted, {
        "proof_verify_ms": elapsed * 1000.0,
    }
