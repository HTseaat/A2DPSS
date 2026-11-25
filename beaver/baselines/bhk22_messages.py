"""Strict canonical Step-2 public wire for BHK+22 + DAG-TOB.

This module defines representation and structural validation only.  Protocol
proof/decryption checks deliberately remain in the committee processors, not
in the DAG ordering predicate.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence, TypeVar

from beaver.baselines.bhk22_context import (
    canonical_json,
    canonical_point,
    decode_object,
    digest_hex,
    exact_fields,
    nonempty_text,
    strict_int,
)
from beaver.baselines.bhk22_contract import (
    ALGORITHM_SUITE,
    ROLE_DEALER,
    ROLE_RESPONDER,
    ROLE_SHAREHOLDER,
    ROLE_VERIFIER,
    validate_public_object,
)
from beaver.baselines.bhk22_crypto import (
    BHK22AEADCiphertext,
    BHK22CoordinateOpening,
    BHK22ElGamalCiphertext,
    BHK22ProofContext,
    BHK22ProofEnvelope,
    BHK22RecipientCiphertext,
    BHK22VectorCommitment,
)


WIRE_VERSION = 1
MESSAGE_DIGEST_DOMAIN = b"ATCaaS/BHK22/public-wire/v1\x00"

DEALER_BATCH_KIND = "atcaas-bhk22-dealer-batch"
VERIFIER_BATCH_KIND = "atcaas-bhk22-verifier-batch"
RESPONDER_BATCH_KIND = "atcaas-bhk22-responder-batch"
FINALIZE_OUTPUT_KIND = "atcaas-bhk22-public-finalize-output"


def _message_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(MESSAGE_DIGEST_DOMAIN + canonical_json(dict(payload))).hexdigest()


def _typed_tuple(values: Sequence[Any], expected_type: type, *, name: str) -> tuple:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(values)
    if not all(isinstance(value, expected_type) for value in result):
        raise TypeError(f"{name} contains an object of the wrong type")
    return result


def _canonical_id_set(
    values: Sequence[int], *, name: str, n_r: int, allow_empty: bool = True
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an integer sequence")
    result = tuple(strict_int(value, name=f"{name} entry") for value in values)
    if (not allow_empty and not result) or result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be sorted, unique, and canonical")
    if any(value >= n_r for value in result):
        raise ValueError(f"{name} contains an out-of-range role id")
    return result


def _validate_proof_binding(
    proof: BHK22ProofEnvelope,
    *,
    root_context_digest: str,
    phase_qid: str,
    role_type: str,
    role_id: int,
    proof_kind: str,
    algorithm: str,
) -> None:
    expected_context = BHK22ProofContext(
        root_context_digest=root_context_digest,
        phase_qid=phase_qid,
        role_type=role_type,
        role_id=role_id,
    )
    if (
        proof.context != expected_context
        or proof.proof_kind != proof_kind
        or proof.algorithm != algorithm
    ):
        raise ValueError(f"{proof_kind} proof envelope binding mismatch")


@dataclass(frozen=True)
class BHK22DealerVerifierColumn:
    verifier_role_id: int
    column_commitment: str
    encrypted_row: BHK22AEADCiphertext
    encrypted_row_key: BHK22RecipientCiphertext

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "verifier_role_id",
            strict_int(self.verifier_role_id, name="verifier_role_id"),
        )
        object.__setattr__(
            self,
            "column_commitment",
            canonical_point(self.column_commitment, name="dealer column commitment"),
        )
        if not isinstance(self.encrypted_row, BHK22AEADCiphertext):
            raise TypeError("encrypted_row must be BHK22AEADCiphertext")
        if not isinstance(self.encrypted_row_key, BHK22RecipientCiphertext):
            raise TypeError("encrypted_row_key must be BHK22RecipientCiphertext")
        if self.encrypted_row_key.purpose != "verifier-key":
            raise ValueError("dealer verifier key has the wrong encryption purpose")

    def to_obj(self) -> dict[str, Any]:
        return {
            "verifier_role_id": self.verifier_role_id,
            "column_commitment": self.column_commitment,
            "encrypted_row": self.encrypted_row.to_obj(),
            "encrypted_row_key": self.encrypted_row_key.to_obj(),
        }

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22DealerVerifierColumn":
        exact_fields(
            value,
            {
                "verifier_role_id",
                "column_commitment",
                "encrypted_row",
                "encrypted_row_key",
            },
            name="dealer verifier column",
        )
        return cls(
            verifier_role_id=value["verifier_role_id"],
            column_commitment=value["column_commitment"],
            encrypted_row=BHK22AEADCiphertext.from_obj(value["encrypted_row"]),
            encrypted_row_key=BHK22RecipientCiphertext.from_obj(
                value["encrypted_row_key"]
            ),
        )


@dataclass(frozen=True)
class BHK22DealerResponderPackage:
    responder_role_id: int
    key_share_commitment: BHK22VectorCommitment
    encrypted_future_state: BHK22RecipientCiphertext

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "responder_role_id",
            strict_int(self.responder_role_id, name="responder_role_id"),
        )
        if not isinstance(self.key_share_commitment, BHK22VectorCommitment):
            raise TypeError("key_share_commitment must be BHK22VectorCommitment")
        if not isinstance(self.encrypted_future_state, BHK22RecipientCiphertext):
            raise TypeError("encrypted_future_state must be BHK22RecipientCiphertext")
        if self.encrypted_future_state.purpose != "responder-future-state":
            raise ValueError("responder state has the wrong encryption purpose")

    def to_obj(self) -> dict[str, Any]:
        return {
            "responder_role_id": self.responder_role_id,
            "key_share_commitment": self.key_share_commitment.to_obj(),
            "encrypted_future_state": self.encrypted_future_state.to_obj(),
        }

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22DealerResponderPackage":
        exact_fields(
            value,
            {
                "responder_role_id",
                "key_share_commitment",
                "encrypted_future_state",
            },
            name="dealer responder package",
        )
        return cls(
            responder_role_id=value["responder_role_id"],
            key_share_commitment=BHK22VectorCommitment.from_obj(
                value["key_share_commitment"]
            ),
            encrypted_future_state=BHK22RecipientCiphertext.from_obj(
                value["encrypted_future_state"]
            ),
        )


@dataclass(frozen=True)
class BHK22VerifierShareholderPackage:
    shareholder_role_id: int
    vector_commitment: BHK22VectorCommitment
    encrypted_payload: BHK22RecipientCiphertext

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shareholder_role_id",
            strict_int(self.shareholder_role_id, name="shareholder_role_id"),
        )
        if not isinstance(self.vector_commitment, BHK22VectorCommitment):
            raise TypeError("vector_commitment must be BHK22VectorCommitment")
        if not isinstance(self.encrypted_payload, BHK22RecipientCiphertext):
            raise TypeError("encrypted_payload must be BHK22RecipientCiphertext")
        if self.encrypted_payload.purpose != "shareholder-private-payload":
            raise ValueError("shareholder payload has the wrong encryption purpose")

    def to_obj(self) -> dict[str, Any]:
        return {
            "shareholder_role_id": self.shareholder_role_id,
            "vector_commitment": self.vector_commitment.to_obj(),
            "encrypted_payload": self.encrypted_payload.to_obj(),
        }

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22VerifierShareholderPackage":
        exact_fields(
            value,
            {"shareholder_role_id", "vector_commitment", "encrypted_payload"},
            name="verifier shareholder package",
        )
        return cls(
            shareholder_role_id=value["shareholder_role_id"],
            vector_commitment=BHK22VectorCommitment.from_obj(
                value["vector_commitment"]
            ),
            encrypted_payload=BHK22RecipientCiphertext.from_obj(
                value["encrypted_payload"]
            ),
        )


@dataclass(frozen=True)
class BHK22ResponderOpening:
    dealer_role_id: int
    verifier_role_id: int
    key_share_opening: BHK22CoordinateOpening

    def __post_init__(self) -> None:
        dealer = strict_int(self.dealer_role_id, name="response dealer_role_id")
        verifier = strict_int(self.verifier_role_id, name="response verifier_role_id")
        if not isinstance(self.key_share_opening, BHK22CoordinateOpening):
            raise TypeError("key_share_opening must be BHK22CoordinateOpening")
        if self.key_share_opening.index != verifier:
            raise ValueError("response opening index must equal verifier_role_id")
        object.__setattr__(self, "dealer_role_id", dealer)
        object.__setattr__(self, "verifier_role_id", verifier)

    def to_obj(self) -> dict[str, Any]:
        return {
            "dealer_role_id": self.dealer_role_id,
            "verifier_role_id": self.verifier_role_id,
            "key_share_opening": self.key_share_opening.to_obj(),
        }

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22ResponderOpening":
        exact_fields(
            value,
            {"dealer_role_id", "verifier_role_id", "key_share_opening"},
            name="responder opening",
        )
        return cls(
            dealer_role_id=value["dealer_role_id"],
            verifier_role_id=value["verifier_role_id"],
            key_share_opening=BHK22CoordinateOpening.from_obj(
                value["key_share_opening"]
            ),
        )


_MessageT = TypeVar("_MessageT", bound="_CanonicalMessage")


class _CanonicalMessage:
    KIND: ClassVar[str]
    ROLE_TYPE: ClassVar[str]

    def _payload_obj(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_obj(self) -> dict[str, Any]:
        value = self._payload_obj()
        value["message_digest"] = self.message_digest
        return value

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_obj())

    @classmethod
    def from_bytes(cls: type[_MessageT], data: bytes | str) -> _MessageT:
        raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        message = cls.from_obj(decode_object(raw, name=cls.KIND))
        if not hmac.compare_digest(raw, message.to_bytes()):
            raise ValueError(f"{cls.KIND} is not canonical JSON")
        return message

    @property
    def wire_size_bytes(self) -> int:
        return len(self.to_bytes())


def _normalize_common(message: Any, *, role_type: str) -> None:
    root = digest_hex(message.root_context_digest, name="root_context_digest")
    qid = nonempty_text(message.phase_qid, name="phase_qid")
    n_r = strict_int(message.n_r, name="n_r", minimum=1)
    role_id = strict_int(message.role_id, name="role_id")
    sender = strict_int(message.network_sender_id, name="network_sender_id")
    if role_id >= n_r:
        raise ValueError("wire role_id is outside n_r")
    object.__setattr__(message, "root_context_digest", root)
    object.__setattr__(message, "phase_qid", qid)
    object.__setattr__(message, "n_r", n_r)
    object.__setattr__(message, "role_id", role_id)
    object.__setattr__(message, "network_sender_id", sender)
    if message.ROLE_TYPE != role_type:
        raise ValueError("internal BHK22 wire role mismatch")


def _finish_message(message: _CanonicalMessage) -> None:
    digest = _message_digest(message._payload_obj())
    object.__setattr__(message, "message_digest", digest)
    validate_public_object(message.to_obj())


def _check_header(
    value: Mapping[str, Any],
    *,
    kind: str,
    role_type: str,
    supplied_digest: str,
    reconstructed: _CanonicalMessage,
) -> None:
    if value["kind"] != kind or value["version"] != WIRE_VERSION:
        raise ValueError(f"unsupported {kind} wire version")
    if value["role_type"] != role_type:
        raise ValueError(f"{kind} has the wrong role_type")
    supplied = digest_hex(supplied_digest, name="message_digest")
    if not hmac.compare_digest(supplied, reconstructed.message_digest):
        raise ValueError(f"{kind} message digest mismatch")


@dataclass(frozen=True)
class BHK22DealerBatch(_CanonicalMessage):
    KIND: ClassVar[str] = DEALER_BATCH_KIND
    ROLE_TYPE: ClassVar[str] = ROLE_DEALER

    root_context_digest: str
    phase_qid: str
    n_r: int
    role_id: int
    network_sender_id: int
    verifier_columns: Sequence[BHK22DealerVerifierColumn]
    responder_packages: Sequence[BHK22DealerResponderPackage]
    figure8_ciphertexts: Sequence[BHK22ElGamalCiphertext]
    c_i0: str
    c_prime_i0: str
    delta_i: str
    figure7_proof: BHK22ProofEnvelope
    figure8_proof: BHK22ProofEnvelope
    message_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _normalize_common(self, role_type=ROLE_DEALER)
        columns = _typed_tuple(
            self.verifier_columns, BHK22DealerVerifierColumn, name="verifier_columns"
        )
        responders = _typed_tuple(
            self.responder_packages,
            BHK22DealerResponderPackage,
            name="responder_packages",
        )
        ciphertexts = _typed_tuple(
            self.figure8_ciphertexts,
            BHK22ElGamalCiphertext,
            name="figure8_ciphertexts",
        )
        if tuple(item.verifier_role_id for item in columns) != tuple(range(self.n_r)):
            raise ValueError("dealer verifier columns must cover role ids 0..n_r-1")
        if tuple(item.responder_role_id for item in responders) != tuple(range(self.n_r)):
            raise ValueError("dealer responder packages must cover role ids 0..n_r-1")
        if any(item.key_share_commitment.length != self.n_r for item in responders):
            raise ValueError("gamma commitment length must equal n_r")
        if len(ciphertexts) != self.n_r + 1:
            raise ValueError("Figure 8 ciphertext vector must cover indices 0..n_r")
        object.__setattr__(self, "verifier_columns", columns)
        object.__setattr__(self, "responder_packages", responders)
        object.__setattr__(self, "figure8_ciphertexts", ciphertexts)
        object.__setattr__(self, "c_i0", canonical_point(self.c_i0, name="C_i,0"))
        object.__setattr__(
            self, "c_prime_i0", canonical_point(self.c_prime_i0, name="C'_i,0")
        )
        object.__setattr__(self, "delta_i", canonical_point(self.delta_i, name="Delta_i"))
        if not isinstance(self.figure7_proof, BHK22ProofEnvelope) or not isinstance(
            self.figure8_proof, BHK22ProofEnvelope
        ):
            raise TypeError("dealer proofs must be BHK22ProofEnvelope")
        _validate_proof_binding(
            self.figure7_proof,
            root_context_digest=self.root_context_digest,
            phase_qid=self.phase_qid,
            role_type=ROLE_DEALER,
            role_id=self.role_id,
            proof_kind="figure7",
            algorithm=ALGORITHM_SUITE.figure7_proof,
        )
        _validate_proof_binding(
            self.figure8_proof,
            root_context_digest=self.root_context_digest,
            phase_qid=self.phase_qid,
            role_type=ROLE_DEALER,
            role_id=self.role_id,
            proof_kind="figure8-randomness",
            algorithm=ALGORITHM_SUITE.figure8_proof,
        )
        _finish_message(self)

    def _payload_obj(self) -> dict[str, Any]:
        return {
            "kind": self.KIND,
            "version": WIRE_VERSION,
            "root_context_digest": self.root_context_digest,
            "phase_qid": self.phase_qid,
            "role_type": self.ROLE_TYPE,
            "role_id": self.role_id,
            "network_sender_id": self.network_sender_id,
            "n_r": self.n_r,
            "verifier_columns": [item.to_obj() for item in self.verifier_columns],
            "responder_packages": [item.to_obj() for item in self.responder_packages],
            "figure8_ciphertexts": [item.to_obj() for item in self.figure8_ciphertexts],
            "c_i0": self.c_i0,
            "c_prime_i0": self.c_prime_i0,
            "delta_i": self.delta_i,
            "figure7_proof": self.figure7_proof.to_obj(),
            "figure8_proof": self.figure8_proof.to_obj(),
        }

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22DealerBatch":
        expected = {
            "kind", "version", "root_context_digest", "phase_qid", "role_type",
            "role_id", "network_sender_id", "n_r", "verifier_columns",
            "responder_packages", "figure8_ciphertexts", "c_i0", "c_prime_i0",
            "delta_i", "figure7_proof", "figure8_proof", "message_digest",
        }
        exact_fields(value, expected, name=cls.KIND)
        message = cls(
            root_context_digest=value["root_context_digest"],
            phase_qid=value["phase_qid"],
            n_r=value["n_r"],
            role_id=value["role_id"],
            network_sender_id=value["network_sender_id"],
            verifier_columns=tuple(
                BHK22DealerVerifierColumn.from_obj(item)
                for item in value["verifier_columns"]
            ),
            responder_packages=tuple(
                BHK22DealerResponderPackage.from_obj(item)
                for item in value["responder_packages"]
            ),
            figure8_ciphertexts=tuple(
                BHK22ElGamalCiphertext.from_obj(item)
                for item in value["figure8_ciphertexts"]
            ),
            c_i0=value["c_i0"],
            c_prime_i0=value["c_prime_i0"],
            delta_i=value["delta_i"],
            figure7_proof=BHK22ProofEnvelope.from_obj(value["figure7_proof"]),
            figure8_proof=BHK22ProofEnvelope.from_obj(value["figure8_proof"]),
        )
        _check_header(
            value, kind=cls.KIND, role_type=cls.ROLE_TYPE,
            supplied_digest=value["message_digest"], reconstructed=message,
        )
        return message


@dataclass(frozen=True)
class BHK22VerifierBatch(_CanonicalMessage):
    KIND: ClassVar[str] = VERIFIER_BATCH_KIND
    ROLE_TYPE: ClassVar[str] = ROLE_VERIFIER

    root_context_digest: str
    phase_qid: str
    n_r: int
    role_id: int
    network_sender_id: int
    accused_dealer_ids: Sequence[int]
    no_accusation_dealer_ids: Sequence[int]
    shareholder_packages: Sequence[BHK22VerifierShareholderPackage]
    forwarding_points: Sequence[str]
    forwarding_proof: BHK22ProofEnvelope
    message_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _normalize_common(self, role_type=ROLE_VERIFIER)
        accused = _canonical_id_set(
            self.accused_dealer_ids, name="accused_dealer_ids", n_r=self.n_r
        )
        clean = _canonical_id_set(
            self.no_accusation_dealer_ids,
            name="no_accusation_dealer_ids",
            n_r=self.n_r,
        )
        if set(accused) & set(clean) or set(accused) | set(clean) != set(range(self.n_r)):
            raise ValueError("accusation and NA sets must partition all dealer ids")
        packages = _typed_tuple(
            self.shareholder_packages,
            BHK22VerifierShareholderPackage,
            name="shareholder_packages",
        )
        if tuple(item.shareholder_role_id for item in packages) != tuple(range(self.n_r)):
            raise ValueError("shareholder packages must cover role ids 0..n_r-1")
        if not clean:
            raise ValueError("verifier batch requires a non-empty NA set")
        if any(item.vector_commitment.length != len(clean) for item in packages):
            raise ValueError("shareholder h commitment length must equal |NA_j|")
        forwarding = tuple(
            canonical_point(item, name="verifier forwarding point")
            for item in self.forwarding_points
        )
        if len(forwarding) != self.n_r + 1:
            raise ValueError("forwarding point vector must cover indices 0..n_r")
        if not isinstance(self.forwarding_proof, BHK22ProofEnvelope):
            raise TypeError("forwarding_proof must be BHK22ProofEnvelope")
        _validate_proof_binding(
            self.forwarding_proof,
            root_context_digest=self.root_context_digest,
            phase_qid=self.phase_qid,
            role_type=ROLE_VERIFIER,
            role_id=self.role_id,
            proof_kind="forwarding",
            algorithm=ALGORITHM_SUITE.verifier_forwarding_proof,
        )
        object.__setattr__(self, "accused_dealer_ids", accused)
        object.__setattr__(self, "no_accusation_dealer_ids", clean)
        object.__setattr__(self, "shareholder_packages", packages)
        object.__setattr__(self, "forwarding_points", forwarding)
        _finish_message(self)

    def _payload_obj(self) -> dict[str, Any]:
        return {
            "kind": self.KIND,
            "version": WIRE_VERSION,
            "root_context_digest": self.root_context_digest,
            "phase_qid": self.phase_qid,
            "role_type": self.ROLE_TYPE,
            "role_id": self.role_id,
            "network_sender_id": self.network_sender_id,
            "n_r": self.n_r,
            "accused_dealer_ids": list(self.accused_dealer_ids),
            "no_accusation_dealer_ids": list(self.no_accusation_dealer_ids),
            "shareholder_packages": [item.to_obj() for item in self.shareholder_packages],
            "forwarding_points": list(self.forwarding_points),
            "forwarding_proof": self.forwarding_proof.to_obj(),
        }

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22VerifierBatch":
        exact_fields(
            value,
            {
                "kind", "version", "root_context_digest", "phase_qid", "role_type",
                "role_id", "network_sender_id", "n_r", "accused_dealer_ids",
                "no_accusation_dealer_ids", "shareholder_packages",
                "forwarding_points", "forwarding_proof", "message_digest",
            },
            name=cls.KIND,
        )
        message = cls(
            root_context_digest=value["root_context_digest"],
            phase_qid=value["phase_qid"],
            n_r=value["n_r"],
            role_id=value["role_id"],
            network_sender_id=value["network_sender_id"],
            accused_dealer_ids=value["accused_dealer_ids"],
            no_accusation_dealer_ids=value["no_accusation_dealer_ids"],
            shareholder_packages=tuple(
                BHK22VerifierShareholderPackage.from_obj(item)
                for item in value["shareholder_packages"]
            ),
            forwarding_points=value["forwarding_points"],
            forwarding_proof=BHK22ProofEnvelope.from_obj(value["forwarding_proof"]),
        )
        _check_header(
            value, kind=cls.KIND, role_type=cls.ROLE_TYPE,
            supplied_digest=value["message_digest"], reconstructed=message,
        )
        return message


@dataclass(frozen=True)
class BHK22ResponderBatch(_CanonicalMessage):
    KIND: ClassVar[str] = RESPONDER_BATCH_KIND
    ROLE_TYPE: ClassVar[str] = ROLE_RESPONDER

    root_context_digest: str
    phase_qid: str
    n_r: int
    role_id: int
    network_sender_id: int
    openings: Sequence[BHK22ResponderOpening]
    message_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _normalize_common(self, role_type=ROLE_RESPONDER)
        openings = _typed_tuple(
            self.openings, BHK22ResponderOpening, name="responder openings"
        )
        pairs = tuple((item.dealer_role_id, item.verifier_role_id) for item in openings)
        if pairs != tuple(sorted(set(pairs))):
            raise ValueError("response openings must be sorted by unique dealer/verifier pair")
        if any(
            dealer >= self.n_r
            or verifier >= self.n_r
            or opening.key_share_opening.length != self.n_r
            for (dealer, verifier), opening in zip(pairs, openings)
        ):
            raise ValueError("response opening id/length is outside n_r")
        object.__setattr__(self, "openings", openings)
        _finish_message(self)

    def _payload_obj(self) -> dict[str, Any]:
        return {
            "kind": self.KIND,
            "version": WIRE_VERSION,
            "root_context_digest": self.root_context_digest,
            "phase_qid": self.phase_qid,
            "role_type": self.ROLE_TYPE,
            "role_id": self.role_id,
            "network_sender_id": self.network_sender_id,
            "n_r": self.n_r,
            "openings": [item.to_obj() for item in self.openings],
        }

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22ResponderBatch":
        exact_fields(
            value,
            {
                "kind", "version", "root_context_digest", "phase_qid", "role_type",
                "role_id", "network_sender_id", "n_r", "openings", "message_digest",
            },
            name=cls.KIND,
        )
        message = cls(
            root_context_digest=value["root_context_digest"],
            phase_qid=value["phase_qid"],
            n_r=value["n_r"],
            role_id=value["role_id"],
            network_sender_id=value["network_sender_id"],
            openings=tuple(BHK22ResponderOpening.from_obj(item) for item in value["openings"]),
        )
        _check_header(
            value, kind=cls.KIND, role_type=cls.ROLE_TYPE,
            supplied_digest=value["message_digest"], reconstructed=message,
        )
        return message


@dataclass(frozen=True)
class BHK22PublicFinalizeOutput(_CanonicalMessage):
    KIND: ClassVar[str] = FINALIZE_OUTPUT_KIND
    ROLE_TYPE: ClassVar[str] = ROLE_SHAREHOLDER

    root_context_digest: str
    phase_qid: str
    n_r: int
    role_id: int
    network_sender_id: int
    qualified_dealer_ids: Sequence[int]
    selected_dealer_ids: Sequence[int]
    aggregate_ciphertext: BHK22ElGamalCiphertext
    public_value: str
    correct_decryption_proof: BHK22ProofEnvelope
    aggregate_ciphertext_digest: str = field(init=False)
    message_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _normalize_common(self, role_type=ROLE_SHAREHOLDER)
        qualified = _canonical_id_set(
            self.qualified_dealer_ids,
            name="qualified_dealer_ids",
            n_r=self.n_r,
            allow_empty=False,
        )
        selected = _canonical_id_set(
            self.selected_dealer_ids,
            name="selected_dealer_ids",
            n_r=self.n_r,
            allow_empty=False,
        )
        if not set(selected) <= set(qualified):
            raise ValueError("selected dealers must be a subset of qualified dealers")
        if not isinstance(self.aggregate_ciphertext, BHK22ElGamalCiphertext):
            raise TypeError("aggregate_ciphertext must be BHK22ElGamalCiphertext")
        public_value = canonical_point(self.public_value, name="successor public value")
        if not isinstance(self.correct_decryption_proof, BHK22ProofEnvelope):
            raise TypeError("correct_decryption_proof must be BHK22ProofEnvelope")
        _validate_proof_binding(
            self.correct_decryption_proof,
            root_context_digest=self.root_context_digest,
            phase_qid=self.phase_qid,
            role_type=ROLE_SHAREHOLDER,
            role_id=self.role_id,
            proof_kind="correct-decryption",
            algorithm=ALGORITHM_SUITE.correct_decryption_proof,
        )
        aggregate_digest = hashlib.sha256(
            canonical_json(self.aggregate_ciphertext.to_obj())
        ).hexdigest()
        object.__setattr__(self, "qualified_dealer_ids", qualified)
        object.__setattr__(self, "selected_dealer_ids", selected)
        object.__setattr__(self, "public_value", public_value)
        object.__setattr__(self, "aggregate_ciphertext_digest", aggregate_digest)
        _finish_message(self)

    def _payload_obj(self) -> dict[str, Any]:
        return {
            "kind": self.KIND,
            "version": WIRE_VERSION,
            "root_context_digest": self.root_context_digest,
            "phase_qid": self.phase_qid,
            "role_type": self.ROLE_TYPE,
            "role_id": self.role_id,
            "network_sender_id": self.network_sender_id,
            "n_r": self.n_r,
            "qualified_dealer_ids": list(self.qualified_dealer_ids),
            "selected_dealer_ids": list(self.selected_dealer_ids),
            "aggregate_ciphertext": self.aggregate_ciphertext.to_obj(),
            "aggregate_ciphertext_digest": self.aggregate_ciphertext_digest,
            "public_value": self.public_value,
            "correct_decryption_proof": self.correct_decryption_proof.to_obj(),
        }

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22PublicFinalizeOutput":
        exact_fields(
            value,
            {
                "kind", "version", "root_context_digest", "phase_qid", "role_type",
                "role_id", "network_sender_id", "n_r", "qualified_dealer_ids",
                "selected_dealer_ids", "aggregate_ciphertext",
                "aggregate_ciphertext_digest", "public_value",
                "correct_decryption_proof", "message_digest",
            },
            name=cls.KIND,
        )
        message = cls(
            root_context_digest=value["root_context_digest"],
            phase_qid=value["phase_qid"],
            n_r=value["n_r"],
            role_id=value["role_id"],
            network_sender_id=value["network_sender_id"],
            qualified_dealer_ids=value["qualified_dealer_ids"],
            selected_dealer_ids=value["selected_dealer_ids"],
            aggregate_ciphertext=BHK22ElGamalCiphertext.from_obj(
                value["aggregate_ciphertext"]
            ),
            public_value=value["public_value"],
            correct_decryption_proof=BHK22ProofEnvelope.from_obj(
                value["correct_decryption_proof"]
            ),
        )
        supplied_aggregate = digest_hex(
            value["aggregate_ciphertext_digest"],
            name="aggregate_ciphertext_digest",
        )
        if not hmac.compare_digest(
            supplied_aggregate, message.aggregate_ciphertext_digest
        ):
            raise ValueError("aggregate ciphertext digest mismatch")
        _check_header(
            value, kind=cls.KIND, role_type=cls.ROLE_TYPE,
            supplied_digest=value["message_digest"], reconstructed=message,
        )
        return message


PUBLIC_WIRE_TYPES = (
    BHK22DealerBatch,
    BHK22VerifierBatch,
    BHK22ResponderBatch,
    BHK22PublicFinalizeOutput,
)


__all__ = [
    "BHK22DealerBatch",
    "BHK22DealerResponderPackage",
    "BHK22DealerVerifierColumn",
    "BHK22PublicFinalizeOutput",
    "BHK22ResponderBatch",
    "BHK22ResponderOpening",
    "BHK22VerifierBatch",
    "BHK22VerifierShareholderPackage",
    "DEALER_BATCH_KIND",
    "FINALIZE_OUTPUT_KIND",
    "PUBLIC_WIRE_TYPES",
    "RESPONDER_BATCH_KIND",
    "VERIFIER_BATCH_KIND",
    "WIRE_VERSION",
]
