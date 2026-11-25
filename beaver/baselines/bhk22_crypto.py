"""Step-2 cryptographic and proof adapters for BHK+22 + DAG-TOB."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import re
from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from Crypto.Cipher import AES

from beaver.baselines import benhamouda_sigma as sigma_reference
from beaver.baselines.bhk22_context import (
    canonical_json,
    canonical_point,
    canonical_scalar,
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
    ROLE_TYPES,
    ROLE_VERIFIER,
)
from beaver.g1_backend import G1Point, field_random, g1_linear_combination


VECTOR_COMMITMENT_VERSION = 1
AEAD_CIPHERTEXT_VERSION = 1
RECIPIENT_CIPHERTEXT_VERSION = 1
ELGAMAL_CIPHERTEXT_VERSION = 1
PROOF_ENVELOPE_VERSION = 1

VECTOR_LEAF_DOMAIN = b"ATCaaS/BHK22/vector-commitment/leaf/v1\x00"
VECTOR_NODE_DOMAIN = b"ATCaaS/BHK22/vector-commitment/node/v1\x00"
VECTOR_EMPTY_DOMAIN = b"ATCaaS/BHK22/vector-commitment/empty/v1\x00"
HKDF_INFO_DOMAIN = b"ATCaaS/BHK22/hkdf-info/v1\x00"
PROOF_STATEMENT_DOMAIN = b"ATCaaS/BHK22/proof-statement/v1\x00"
_SIGMA_RESPONSE_NAME_RE = re.compile(r"^(?:anchor|sk|[frx]:[0-9]+)$")

RECIPIENT_PURPOSES = (
    "verifier-key",
    "responder-future-state",
    "shareholder-private-payload",
)
RECIPIENT_PURPOSE_ROLES = {
    "verifier-key": (ROLE_DEALER, ROLE_VERIFIER),
    "responder-future-state": (ROLE_DEALER, ROLE_RESPONDER),
    "shareholder-private-payload": (ROLE_VERIFIER, ROLE_SHAREHOLDER),
}
PROOF_KIND_CONTRACT = {
    "figure7": (ALGORITHM_SUITE.figure7_proof, "resharing"),
    "figure8-randomness": (ALGORITHM_SUITE.figure8_proof, "randomness"),
    "forwarding": (ALGORITHM_SUITE.verifier_forwarding_proof, "dl"),
    "correct-decryption": (ALGORITHM_SUITE.correct_decryption_proof, "decryption"),
}


def _canonical_hex(
    value: Any,
    *,
    name: str,
    exact_bytes: int | None = None,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be lowercase hexadecimal")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be lowercase hexadecimal") from exc
    if raw.hex() != value or (not allow_empty and not raw):
        raise ValueError(f"{name} must be canonical lowercase hexadecimal")
    if exact_bytes is not None and len(raw) != exact_bytes:
        raise ValueError(f"{name} must contain exactly {exact_bytes} bytes")
    return value


def _sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_deep_thaw(item) for item in value]
    return value


def _scalar_bytes(value: int) -> bytes:
    return canonical_scalar(value, name="vector value").to_bytes(32, "big")


def _vector_leaf(length: int, index: int, value: int, salt: bytes) -> bytes:
    return hashlib.sha256(
        VECTOR_LEAF_DOMAIN
        + length.to_bytes(4, "big")
        + index.to_bytes(4, "big")
        + _scalar_bytes(value)
        + salt
    ).digest()


def _vector_parent(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(VECTOR_NODE_DOMAIN + left + right).digest()


def _merkle_levels(leaves: Sequence[bytes]) -> list[list[bytes]]:
    if not leaves:
        return [[hashlib.sha256(VECTOR_EMPTY_DOMAIN).digest()]]
    levels = [list(leaves)]
    while len(levels[-1]) > 1:
        current = levels[-1]
        next_level = []
        for index in range(0, len(current), 2):
            left = current[index]
            right = current[index + 1] if index + 1 < len(current) else left
            next_level.append(_vector_parent(left, right))
        levels.append(next_level)
    return levels


@dataclass(frozen=True)
class BHK22VectorCommitment:
    length: int
    root: str

    def __post_init__(self) -> None:
        length = strict_int(self.length, name="vector commitment length", minimum=1)
        root = _canonical_hex(
            self.root, name="vector commitment root", exact_bytes=32
        )
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "root", root)

    def to_obj(self) -> dict[str, Any]:
        return {
            "suite": ALGORITHM_SUITE.vector_commitment,
            "version": VECTOR_COMMITMENT_VERSION,
            "length": self.length,
            "root": self.root,
        }

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22VectorCommitment":
        exact_fields(
            value, {"suite", "version", "length", "root"}, name="vector commitment"
        )
        if (
            value["suite"] != ALGORITHM_SUITE.vector_commitment
            or value["version"] != VECTOR_COMMITMENT_VERSION
        ):
            raise ValueError("unsupported BHK22 vector commitment")
        return cls(length=value["length"], root=value["root"])


@dataclass(frozen=True)
class BHK22MerklePathEntry:
    side: str
    digest: str

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("Merkle path side must be left or right")
        object.__setattr__(
            self,
            "digest",
            _canonical_hex(self.digest, name="Merkle sibling", exact_bytes=32),
        )

    def to_obj(self) -> dict[str, str]:
        return {"side": self.side, "digest": self.digest}

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22MerklePathEntry":
        exact_fields(value, {"side", "digest"}, name="Merkle path entry")
        return cls(side=value["side"], digest=value["digest"])


@dataclass(frozen=True)
class BHK22CoordinateOpening:
    length: int
    index: int
    value: int
    salt: str
    path: Sequence[BHK22MerklePathEntry]

    def __post_init__(self) -> None:
        length = strict_int(self.length, name="opening length", minimum=1)
        index = strict_int(self.index, name="opening index")
        if index >= length:
            raise ValueError("opening index is outside commitment length")
        scalar = canonical_scalar(self.value, name="opening value")
        salt = _canonical_hex(self.salt, name="opening salt", exact_bytes=32)
        path = tuple(
            item
            if isinstance(item, BHK22MerklePathEntry)
            else BHK22MerklePathEntry.from_obj(item)
            for item in self.path
        )
        expected_depth = 0
        width = length
        while width > 1:
            expected_depth += 1
            width = (width + 1) // 2
        if len(path) != expected_depth:
            raise ValueError("coordinate opening has the wrong Merkle depth")
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "value", scalar)
        object.__setattr__(self, "salt", salt)
        object.__setattr__(self, "path", path)

    def to_obj(self) -> dict[str, Any]:
        return {
            "suite": ALGORITHM_SUITE.vector_commitment,
            "version": VECTOR_COMMITMENT_VERSION,
            "length": self.length,
            "index": self.index,
            "value": str(self.value),
            "salt": self.salt,
            "path": [entry.to_obj() for entry in self.path],
        }

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22CoordinateOpening":
        exact_fields(
            value,
            {"suite", "version", "length", "index", "value", "salt", "path"},
            name="coordinate opening",
        )
        if (
            value["suite"] != ALGORITHM_SUITE.vector_commitment
            or value["version"] != VECTOR_COMMITMENT_VERSION
        ):
            raise ValueError("unsupported BHK22 coordinate opening")
        return cls(
            length=value["length"],
            index=value["index"],
            value=canonical_scalar(value["value"], name="opening value"),
            salt=value["salt"],
            path=tuple(BHK22MerklePathEntry.from_obj(item) for item in value["path"]),
        )


@dataclass(frozen=True)
class BHK22FullOpening:
    values: Sequence[int]
    salts: Sequence[str]

    def __post_init__(self) -> None:
        values = tuple(
            canonical_scalar(value, name="full-opening value") for value in self.values
        )
        salts = tuple(
            _canonical_hex(salt, name="full-opening salt", exact_bytes=32)
            for salt in self.salts
        )
        if not values or len(values) != len(salts):
            raise ValueError("full opening requires equal non-empty values/salts")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "salts", salts)

    def to_obj(self) -> dict[str, Any]:
        return {
            "suite": ALGORITHM_SUITE.vector_commitment,
            "version": VECTOR_COMMITMENT_VERSION,
            "values": [str(value) for value in self.values],
            "salts": list(self.salts),
        }

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22FullOpening":
        exact_fields(value, {"suite", "version", "values", "salts"}, name="full opening")
        if (
            value["suite"] != ALGORITHM_SUITE.vector_commitment
            or value["version"] != VECTOR_COMMITMENT_VERSION
        ):
            raise ValueError("unsupported BHK22 full opening")
        return cls(
            values=tuple(
                canonical_scalar(item, name="full-opening value")
                for item in value["values"]
            ),
            salts=value["salts"],
        )


def commit_vector(
    values: Sequence[int], *, salts: Sequence[bytes] | None = None
) -> tuple[BHK22VectorCommitment, BHK22FullOpening]:
    normalized = tuple(
        canonical_scalar(value, name="vector commitment value") for value in values
    )
    if not normalized:
        raise ValueError("cannot commit an empty vector")
    if salts is None:
        salt_bytes = tuple(secrets.token_bytes(32) for _ in normalized)
    else:
        salt_bytes = tuple(bytes(salt) for salt in salts)
        if len(salt_bytes) != len(normalized) or any(
            len(salt) != 32 for salt in salt_bytes
        ):
            raise ValueError("vector commitment salts must be 32 bytes each")
    leaves = [
        _vector_leaf(len(normalized), index, value, salt_bytes[index])
        for index, value in enumerate(normalized)
    ]
    root = _merkle_levels(leaves)[-1][0].hex()
    return (
        BHK22VectorCommitment(length=len(normalized), root=root),
        BHK22FullOpening(
            values=normalized, salts=tuple(salt.hex() for salt in salt_bytes)
        ),
    )


def open_vector_coordinate(
    full_opening: BHK22FullOpening, index: int
) -> BHK22CoordinateOpening:
    if not isinstance(full_opening, BHK22FullOpening):
        raise TypeError("full_opening must be BHK22FullOpening")
    index = strict_int(index, name="coordinate index")
    if index >= len(full_opening.values):
        raise ValueError("coordinate index is outside vector")
    salts = tuple(bytes.fromhex(salt) for salt in full_opening.salts)
    leaves = [
        _vector_leaf(len(full_opening.values), offset, value, salts[offset])
        for offset, value in enumerate(full_opening.values)
    ]
    levels = _merkle_levels(leaves)
    cursor = index
    path = []
    for level in levels[:-1]:
        if cursor % 2:
            sibling_index = cursor - 1
            side = "left"
        else:
            sibling_index = cursor + 1 if cursor + 1 < len(level) else cursor
            side = "right"
        path.append(BHK22MerklePathEntry(side=side, digest=level[sibling_index].hex()))
        cursor //= 2
    return BHK22CoordinateOpening(
        length=len(full_opening.values),
        index=index,
        value=full_opening.values[index],
        salt=full_opening.salts[index],
        path=path,
    )


def verify_coordinate_opening(
    commitment: BHK22VectorCommitment,
    opening: BHK22CoordinateOpening,
) -> bool:
    try:
        if commitment.length != opening.length:
            return False
        current = _vector_leaf(
            opening.length,
            opening.index,
            opening.value,
            bytes.fromhex(opening.salt),
        )
        for entry in opening.path:
            sibling = bytes.fromhex(entry.digest)
            current = (
                _vector_parent(sibling, current)
                if entry.side == "left"
                else _vector_parent(current, sibling)
            )
        return hmac.compare_digest(current.hex(), commitment.root)
    except (TypeError, ValueError):
        return False


def verify_full_opening(
    commitment: BHK22VectorCommitment,
    opening: BHK22FullOpening,
) -> bool:
    try:
        if commitment.length != len(opening.values):
            return False
        rebuilt, _ = commit_vector(
            opening.values,
            salts=tuple(bytes.fromhex(salt) for salt in opening.salts),
        )
        return hmac.compare_digest(rebuilt.root, commitment.root)
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class BHK22AAD:
    root_context_digest: str
    phase_qid: str
    sender_role: str
    sender_role_id: int
    recipient_role: str
    recipient_role_id: int
    payload_kind: str
    payload_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "root_context_digest",
            digest_hex(self.root_context_digest, name="root_context_digest"),
        )
        object.__setattr__(self, "phase_qid", nonempty_text(self.phase_qid, name="phase_qid"))
        if self.sender_role not in ROLE_TYPES or self.recipient_role not in ROLE_TYPES:
            raise ValueError("AAD contains unsupported role type")
        object.__setattr__(
            self, "sender_role_id", strict_int(self.sender_role_id, name="sender_role_id")
        )
        object.__setattr__(
            self,
            "recipient_role_id",
            strict_int(self.recipient_role_id, name="recipient_role_id"),
        )
        object.__setattr__(
            self, "payload_kind", nonempty_text(self.payload_kind, name="payload_kind")
        )
        object.__setattr__(
            self,
            "payload_version",
            strict_int(self.payload_version, name="payload_version", minimum=1),
        )

    def to_obj(self) -> dict[str, Any]:
        return {
            "root_context_digest": self.root_context_digest,
            "phase_qid": self.phase_qid,
            "sender_role": self.sender_role,
            "sender_role_id": self.sender_role_id,
            "recipient_role": self.recipient_role,
            "recipient_role_id": self.recipient_role_id,
            "payload_kind": self.payload_kind,
            "payload_version": self.payload_version,
        }

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_obj())


def hkdf_sha256(
    input_key_material: bytes,
    *,
    salt: bytes,
    info: bytes,
    length: int = 32,
) -> bytes:
    if not isinstance(input_key_material, bytes) or not input_key_material:
        raise ValueError("HKDF input key material must be non-empty bytes")
    if not isinstance(salt, bytes) or not salt:
        raise ValueError("HKDF salt must be non-empty bytes")
    if not isinstance(info, bytes) or not info:
        raise ValueError("HKDF info must be non-empty bytes")
    length = strict_int(length, name="HKDF length", minimum=1)
    if length > 255 * hashlib.sha256().digest_size:
        raise ValueError("HKDF output length is too large")
    prk = hmac.new(salt, input_key_material, hashlib.sha256).digest()
    output = bytearray()
    previous = b""
    counter = 1
    while len(output) < length:
        previous = hmac.new(
            prk, previous + info + bytes([counter]), hashlib.sha256
        ).digest()
        output.extend(previous)
        counter += 1
    return bytes(output[:length])


def derive_aead_key(
    input_key_material: bytes,
    *,
    root_context_digest: str,
    purpose: str,
    binding: bytes,
) -> bytes:
    root = bytes.fromhex(digest_hex(root_context_digest, name="root_context_digest"))
    purpose = nonempty_text(purpose, name="KDF purpose")
    if not isinstance(binding, bytes) or not binding:
        raise ValueError("KDF binding must be non-empty bytes")
    info = HKDF_INFO_DOMAIN + purpose.encode("utf-8") + b"\x00" + binding
    return hkdf_sha256(input_key_material, salt=root, info=info, length=32)


@dataclass(frozen=True)
class BHK22AEADCiphertext:
    nonce: str
    body: str
    tag: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "nonce", _canonical_hex(self.nonce, name="AEAD nonce", exact_bytes=12)
        )
        object.__setattr__(
            self,
            "body",
            _canonical_hex(self.body, name="AEAD body", allow_empty=True),
        )
        object.__setattr__(
            self, "tag", _canonical_hex(self.tag, name="AEAD tag", exact_bytes=16)
        )

    def to_obj(self) -> dict[str, Any]:
        return {
            "suite": ALGORITHM_SUITE.aead,
            "version": AEAD_CIPHERTEXT_VERSION,
            "nonce": self.nonce,
            "body": self.body,
            "tag": self.tag,
        }

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22AEADCiphertext":
        exact_fields(value, {"suite", "version", "nonce", "body", "tag"}, name="AEAD ciphertext")
        if value["suite"] != ALGORITHM_SUITE.aead or value["version"] != 1:
            raise ValueError("unsupported BHK22 AEAD ciphertext")
        return cls(nonce=value["nonce"], body=value["body"], tag=value["tag"])


def aead_encrypt(
    key: bytes,
    plaintext: bytes,
    *,
    aad: BHK22AAD,
    nonce: bytes | None = None,
) -> BHK22AEADCiphertext:
    if not isinstance(key, bytes) or len(key) != 32:
        raise ValueError("AES-256-GCM key must be exactly 32 bytes")
    if not isinstance(plaintext, bytes):
        raise TypeError("AEAD plaintext must be bytes")
    if not isinstance(aad, BHK22AAD):
        raise TypeError("aad must be BHK22AAD")
    nonce_bytes = secrets.token_bytes(12) if nonce is None else bytes(nonce)
    if len(nonce_bytes) != 12:
        raise ValueError("AES-GCM nonce must be exactly 12 bytes")
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce_bytes, mac_len=16)
    cipher.update(aad.to_bytes())
    body, tag = cipher.encrypt_and_digest(plaintext)
    return BHK22AEADCiphertext(
        nonce=nonce_bytes.hex(), body=body.hex(), tag=tag.hex()
    )


def aead_decrypt(
    key: bytes,
    ciphertext: BHK22AEADCiphertext,
    *,
    aad: BHK22AAD,
) -> bytes:
    if not isinstance(key, bytes) or len(key) != 32:
        raise ValueError("AES-256-GCM key must be exactly 32 bytes")
    if not isinstance(ciphertext, BHK22AEADCiphertext):
        raise TypeError("ciphertext must be BHK22AEADCiphertext")
    if not isinstance(aad, BHK22AAD):
        raise TypeError("aad must be BHK22AAD")
    cipher = AES.new(
        key,
        AES.MODE_GCM,
        nonce=bytes.fromhex(ciphertext.nonce),
        mac_len=16,
    )
    cipher.update(aad.to_bytes())
    try:
        return cipher.decrypt_and_verify(
            bytes.fromhex(ciphertext.body), bytes.fromhex(ciphertext.tag)
        )
    except ValueError as exc:
        raise ValueError("BHK22 AEAD authentication failed") from exc


@dataclass(frozen=True)
class BHK22RecipientCiphertext:
    purpose: str
    ephemeral_public_key: str
    ciphertext: BHK22AEADCiphertext

    def __post_init__(self) -> None:
        if self.purpose not in RECIPIENT_PURPOSES:
            raise ValueError("unsupported BHK22 recipient-encryption purpose")
        object.__setattr__(
            self,
            "ephemeral_public_key",
            canonical_point(
                self.ephemeral_public_key, name="ephemeral public key"
            ),
        )
        if not isinstance(self.ciphertext, BHK22AEADCiphertext):
            raise TypeError("recipient ciphertext requires BHK22AEADCiphertext")

    def to_obj(self) -> dict[str, Any]:
        return {
            "suite": ALGORITHM_SUITE.recipient_pke,
            "version": RECIPIENT_CIPHERTEXT_VERSION,
            "purpose": self.purpose,
            "ephemeral_public_key": self.ephemeral_public_key,
            "ciphertext": self.ciphertext.to_obj(),
        }

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22RecipientCiphertext":
        exact_fields(
            value,
            {"suite", "version", "purpose", "ephemeral_public_key", "ciphertext"},
            name="recipient ciphertext",
        )
        if value["suite"] != ALGORITHM_SUITE.recipient_pke or value["version"] != 1:
            raise ValueError("unsupported BHK22 recipient ciphertext")
        return cls(
            purpose=value["purpose"],
            ephemeral_public_key=value["ephemeral_public_key"],
            ciphertext=BHK22AEADCiphertext.from_obj(value["ciphertext"]),
        )


def recipient_encrypt(
    *,
    key_base_hex: str,
    recipient_public_key_hex: str,
    plaintext: bytes,
    aad: BHK22AAD,
    purpose: str,
    ephemeral_secret: int | None = None,
    nonce: bytes | None = None,
) -> BHK22RecipientCiphertext:
    if purpose not in RECIPIENT_PURPOSES:
        raise ValueError("unsupported BHK22 recipient-encryption purpose")
    expected_sender, expected_recipient = RECIPIENT_PURPOSE_ROLES[purpose]
    if (aad.sender_role, aad.recipient_role) != (
        expected_sender,
        expected_recipient,
    ):
        raise ValueError("recipient-encryption purpose/role mismatch")
    normalized_base = canonical_point(key_base_hex, name="recipient key base")
    normalized_recipient = canonical_point(
        recipient_public_key_hex, name="recipient public key"
    )
    base = G1Point.from_hex(normalized_base)
    recipient = G1Point.from_hex(normalized_recipient)
    scalar = field_random() if ephemeral_secret is None else canonical_scalar(
        ephemeral_secret, name="ephemeral secret", nonzero=True
    )
    ephemeral_public = base * scalar
    shared = recipient * scalar
    binding = canonical_json(
        {
            "suite": ALGORITHM_SUITE.recipient_pke,
            "purpose": purpose,
            "key_base": normalized_base,
            "recipient_public_key": normalized_recipient,
            "ephemeral_public_key": ephemeral_public.to_hex(),
            "aad": aad.to_obj(),
        }
    )
    key = derive_aead_key(
        shared.to_hex().encode("ascii"),
        root_context_digest=aad.root_context_digest,
        purpose=purpose,
        binding=binding,
    )
    return BHK22RecipientCiphertext(
        purpose=purpose,
        ephemeral_public_key=ephemeral_public.to_hex(),
        ciphertext=aead_encrypt(key, plaintext, aad=aad, nonce=nonce),
    )


def recipient_decrypt(
    *,
    key_base_hex: str,
    recipient_secret_key: int,
    ciphertext: BHK22RecipientCiphertext,
    aad: BHK22AAD,
    purpose: str,
) -> bytes:
    if purpose != ciphertext.purpose or purpose not in RECIPIENT_PURPOSES:
        raise ValueError("recipient-encryption purpose mismatch")
    expected_sender, expected_recipient = RECIPIENT_PURPOSE_ROLES[purpose]
    if (aad.sender_role, aad.recipient_role) != (
        expected_sender,
        expected_recipient,
    ):
        raise ValueError("recipient-encryption purpose/role mismatch")
    normalized_base = canonical_point(key_base_hex, name="recipient key base")
    base = G1Point.from_hex(normalized_base)
    secret = canonical_scalar(
        recipient_secret_key, name="recipient secret key", nonzero=True
    )
    ephemeral_public = G1Point.from_hex(ciphertext.ephemeral_public_key)
    shared = ephemeral_public * secret
    recipient_public = base * secret
    binding = canonical_json(
        {
            "suite": ALGORITHM_SUITE.recipient_pke,
            "purpose": purpose,
            "key_base": normalized_base,
            "recipient_public_key": recipient_public.to_hex(),
            "ephemeral_public_key": ciphertext.ephemeral_public_key,
            "aad": aad.to_obj(),
        }
    )
    key = derive_aead_key(
        shared.to_hex().encode("ascii"),
        root_context_digest=aad.root_context_digest,
        purpose=purpose,
        binding=binding,
    )
    return aead_decrypt(key, ciphertext.ciphertext, aad=aad)


@dataclass(frozen=True)
class BHK22ElGamalCiphertext:
    X: str
    Y: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "X", canonical_point(self.X, name="ElGamal X"))
        object.__setattr__(self, "Y", canonical_point(self.Y, name="ElGamal Y"))

    def to_obj(self) -> dict[str, Any]:
        return {
            "suite": ALGORITHM_SUITE.public_value_encryption,
            "version": ELGAMAL_CIPHERTEXT_VERSION,
            "X": self.X,
            "Y": self.Y,
        }

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22ElGamalCiphertext":
        exact_fields(value, {"suite", "version", "X", "Y"}, name="Figure 8 ciphertext")
        if (
            value["suite"] != ALGORITHM_SUITE.public_value_encryption
            or value["version"] != 1
        ):
            raise ValueError("unsupported BHK22 Figure 8 ciphertext")
        return cls(X=value["X"], Y=value["Y"])


def encrypt_public_value(
    *,
    generator_hex: str,
    public_key_hex: str,
    plaintext_hex: str,
    randomness: int | None = None,
) -> tuple[BHK22ElGamalCiphertext, int]:
    generator = G1Point.from_hex(canonical_point(generator_hex, name="ElGamal generator"))
    public_key = G1Point.from_hex(canonical_point(public_key_hex, name="ElGamal public key"))
    plaintext = G1Point.from_hex(canonical_point(plaintext_hex, name="ElGamal plaintext"))
    scalar = field_random() if randomness is None else canonical_scalar(
        randomness, name="ElGamal randomness", nonzero=True
    )
    return (
        BHK22ElGamalCiphertext(
            X=(generator * scalar).to_hex(),
            Y=((public_key * scalar) + plaintext).to_hex(),
        ),
        scalar,
    )


def decrypt_public_value(
    ciphertext: BHK22ElGamalCiphertext, *, secret_key: int
) -> str:
    if not isinstance(ciphertext, BHK22ElGamalCiphertext):
        raise TypeError("ciphertext must be BHK22ElGamalCiphertext")
    secret = canonical_scalar(secret_key, name="ElGamal secret key", nonzero=True)
    return (
        G1Point.from_hex(ciphertext.Y) - G1Point.from_hex(ciphertext.X) * secret
    ).to_hex()


def combine_public_value_ciphertexts(
    ciphertexts: Sequence[BHK22ElGamalCiphertext],
    coefficients: Sequence[int],
) -> BHK22ElGamalCiphertext:
    if not ciphertexts or len(ciphertexts) != len(coefficients):
        raise ValueError("ciphertexts/coefficients need equal non-empty lengths")
    if not all(isinstance(item, BHK22ElGamalCiphertext) for item in ciphertexts):
        raise TypeError("all ciphertexts must be BHK22ElGamalCiphertext")
    scalars = [canonical_scalar(value, name="ElGamal coefficient") for value in coefficients]
    return BHK22ElGamalCiphertext(
        X=g1_linear_combination([item.X for item in ciphertexts], scalars).to_hex(),
        Y=g1_linear_combination([item.Y for item in ciphertexts], scalars).to_hex(),
    )


@dataclass(frozen=True)
class BHK22ProofContext:
    root_context_digest: str
    phase_qid: str
    role_type: str
    role_id: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "root_context_digest",
            digest_hex(self.root_context_digest, name="proof root context digest"),
        )
        object.__setattr__(self, "phase_qid", nonempty_text(self.phase_qid, name="proof phase qid"))
        if self.role_type not in ROLE_TYPES:
            raise ValueError("unsupported proof role type")
        object.__setattr__(self, "role_id", strict_int(self.role_id, name="proof role id"))

    def to_obj(self) -> dict[str, Any]:
        return {
            "root_context_digest": self.root_context_digest,
            "phase_qid": self.phase_qid,
            "role_type": self.role_type,
            "role_id": self.role_id,
        }


@dataclass(frozen=True)
class BHK22ProofEnvelope:
    proof_kind: str
    algorithm: str
    context: BHK22ProofContext
    statement_digest: str
    sigma: Mapping[str, Any]

    def __post_init__(self) -> None:
        proof_kind = nonempty_text(self.proof_kind, name="proof_kind")
        algorithm = nonempty_text(self.algorithm, name="proof algorithm")
        if proof_kind not in PROOF_KIND_CONTRACT:
            raise ValueError("unsupported BHK22 proof kind")
        expected_algorithm, expected_sigma_kind = PROOF_KIND_CONTRACT[proof_kind]
        if algorithm != expected_algorithm:
            raise ValueError("BHK22 proof kind/algorithm mismatch")
        object.__setattr__(self, "proof_kind", proof_kind)
        object.__setattr__(self, "algorithm", algorithm)
        if not isinstance(self.context, BHK22ProofContext):
            raise TypeError("proof context has wrong type")
        object.__setattr__(
            self,
            "statement_digest",
            digest_hex(self.statement_digest, name="proof statement digest"),
        )
        if not isinstance(self.sigma, Mapping):
            raise TypeError("sigma proof must be a mapping")
        sigma = dict(self.sigma)
        exact_fields(sigma, {"version", "kind", "sigma"}, name="wrapped Sigma proof")
        if sigma["version"] != 1 or sigma["kind"] != expected_sigma_kind:
            raise ValueError("wrapped Sigma proof kind/version mismatch")
        inner = sigma["sigma"]
        exact_fields(
            inner,
            {"version", "announcements", "responses"},
            name="linear-relation Sigma proof",
        )
        if inner["version"] != 1:
            raise ValueError("unsupported linear-relation Sigma proof")
        if isinstance(inner["announcements"], (str, bytes, bytearray)):
            raise TypeError("Sigma announcements must be a sequence")
        announcements = [
            canonical_point(item, name="Sigma announcement")
            for item in inner["announcements"]
        ]
        if not announcements:
            raise ValueError("Sigma proof must contain announcements")
        if not isinstance(inner["responses"], Mapping) or not inner["responses"]:
            raise TypeError("Sigma responses must be a non-empty mapping")
        responses = {}
        for name, value in inner["responses"].items():
            if not isinstance(name, str) or _SIGMA_RESPONSE_NAME_RE.fullmatch(name) is None:
                raise ValueError("Sigma response name is not canonical")
            responses[name] = canonical_scalar(value, name=f"Sigma response {name}")
        normalized_sigma = {
            "version": 1,
            "kind": expected_sigma_kind,
            "sigma": {
                "version": 1,
                "announcements": announcements,
                "responses": responses,
            },
        }
        object.__setattr__(self, "sigma", _deep_freeze(normalized_sigma))

    def to_obj(self) -> dict[str, Any]:
        return {
            "version": PROOF_ENVELOPE_VERSION,
            "proof_kind": self.proof_kind,
            "algorithm": self.algorithm,
            "context": self.context.to_obj(),
            "statement_digest": self.statement_digest,
            "sigma": _deep_thaw(self.sigma),
        }

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22ProofEnvelope":
        exact_fields(
            value,
            {"version", "proof_kind", "algorithm", "context", "statement_digest", "sigma"},
            name="BHK22 proof envelope",
        )
        if value["version"] != PROOF_ENVELOPE_VERSION:
            raise ValueError("unsupported BHK22 proof envelope")
        exact_fields(
            value["context"],
            {"root_context_digest", "phase_qid", "role_type", "role_id"},
            name="BHK22 proof context",
        )
        return cls(
            proof_kind=value["proof_kind"],
            algorithm=value["algorithm"],
            context=BHK22ProofContext(**value["context"]),
            statement_digest=value["statement_digest"],
            sigma=value["sigma"],
        )


def _proof_label(kind: str, context: BHK22ProofContext) -> str:
    binding = _sha256_hex(canonical_json(context.to_obj()))
    return f"ATCaaS/BHK22/{kind}/v1/{binding}"


def _statement_digest(kind: str, context: BHK22ProofContext, statement: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        PROOF_STATEMENT_DOMAIN
        + kind.encode("ascii")
        + b"\x00"
        + canonical_json(context.to_obj())
        + b"\x00"
        + canonical_json(dict(statement))
    ).hexdigest()


def _proof_matches(
    envelope: BHK22ProofEnvelope,
    *,
    kind: str,
    algorithm: str,
    context: BHK22ProofContext,
    statement: Mapping[str, Any],
) -> bool:
    return (
        isinstance(envelope, BHK22ProofEnvelope)
        and envelope.proof_kind == kind
        and envelope.algorithm == algorithm
        and envelope.context == context
        and hmac.compare_digest(
            envelope.statement_digest,
            _statement_digest(kind, context, statement),
        )
    )


def prove_figure7(
    *,
    context: BHK22ProofContext,
    anchor_base_hex: str,
    share_base_hex: str,
    eval_bases: Sequence[str],
    share_point_hex: str,
    aggregate_commitment_hex: str,
    polynomial_coeffs: Sequence[int],
    anchor_scalar: int,
    eval_points: Sequence[int],
) -> BHK22ProofEnvelope:
    statement = {
        "anchor_base": canonical_point(anchor_base_hex, name="Figure 7 anchor base"),
        "share_base": canonical_point(share_base_hex, name="Figure 7 share base"),
        "eval_bases": [canonical_point(item, name="Figure 7 eval base") for item in eval_bases],
        "share_point": canonical_point(share_point_hex, name="Figure 7 share point"),
        "aggregate_commitment": canonical_point(
            aggregate_commitment_hex, name="Figure 7 aggregate commitment"
        ),
        "degree": len(polynomial_coeffs) - 1,
        "eval_points": [strict_int(item, name="Figure 7 eval point", minimum=1) for item in eval_points],
    }
    proof = sigma_reference.prove_resharing_consistency(
        anchor_base_hex=statement["anchor_base"],
        share_base_hex=statement["share_base"],
        eval_bases=statement["eval_bases"],
        share_point_hex=statement["share_point"],
        aggregate_commitment_hex=statement["aggregate_commitment"],
        polynomial_coeffs=polynomial_coeffs,
        anchor_scalar=anchor_scalar,
        eval_points=statement["eval_points"],
        label=_proof_label("figure7", context),
    )
    return BHK22ProofEnvelope(
        proof_kind="figure7",
        algorithm=ALGORITHM_SUITE.figure7_proof,
        context=context,
        statement_digest=_statement_digest("figure7", context, statement),
        sigma=proof,
    )


def verify_figure7(
    *,
    context: BHK22ProofContext,
    anchor_base_hex: str,
    share_base_hex: str,
    eval_bases: Sequence[str],
    share_point_hex: str,
    aggregate_commitment_hex: str,
    degree: int,
    eval_points: Sequence[int],
    proof: BHK22ProofEnvelope,
) -> bool:
    try:
        statement = {
            "anchor_base": canonical_point(anchor_base_hex, name="Figure 7 anchor base"),
            "share_base": canonical_point(share_base_hex, name="Figure 7 share base"),
            "eval_bases": [canonical_point(item, name="Figure 7 eval base") for item in eval_bases],
            "share_point": canonical_point(share_point_hex, name="Figure 7 share point"),
            "aggregate_commitment": canonical_point(
                aggregate_commitment_hex, name="Figure 7 aggregate commitment"
            ),
            "degree": strict_int(degree, name="Figure 7 degree"),
            "eval_points": [strict_int(item, name="Figure 7 eval point", minimum=1) for item in eval_points],
        }
        return _proof_matches(
            proof,
            kind="figure7",
            algorithm=ALGORITHM_SUITE.figure7_proof,
            context=context,
            statement=statement,
        ) and sigma_reference.verify_resharing_consistency(
            anchor_base_hex=statement["anchor_base"],
            share_base_hex=statement["share_base"],
            eval_bases=statement["eval_bases"],
            share_point_hex=statement["share_point"],
            aggregate_commitment_hex=statement["aggregate_commitment"],
            degree=statement["degree"],
            proof=proof.sigma,
            eval_points=statement["eval_points"],
            label=_proof_label("figure7", context),
        )
    except (TypeError, ValueError):
        return False


def prove_figure8_randomness(
    *,
    context: BHK22ProofContext,
    delta_hex: str,
    ciphertexts: Sequence[BHK22ElGamalCiphertext],
    h_bases: Sequence[str],
    g_bases: Sequence[str],
    randomness: Sequence[int],
) -> BHK22ProofEnvelope:
    statement = {
        "delta": canonical_point(delta_hex, name="Figure 8 delta"),
        "x_points": [item.X for item in ciphertexts],
        "h_bases": [canonical_point(item, name="Figure 8 H base") for item in h_bases],
        "g_bases": [canonical_point(item, name="Figure 8 G base") for item in g_bases],
    }
    proof = sigma_reference.prove_randomness_consistency(
        delta_hex=statement["delta"],
        x_points=statement["x_points"],
        h_bases=statement["h_bases"],
        g_bases=statement["g_bases"],
        randomness=randomness,
        label=_proof_label("figure8-randomness", context),
    )
    return BHK22ProofEnvelope(
        proof_kind="figure8-randomness",
        algorithm=ALGORITHM_SUITE.figure8_proof,
        context=context,
        statement_digest=_statement_digest(
            "figure8-randomness", context, statement
        ),
        sigma=proof,
    )


def verify_figure8_randomness(
    *,
    context: BHK22ProofContext,
    delta_hex: str,
    ciphertexts: Sequence[BHK22ElGamalCiphertext],
    h_bases: Sequence[str],
    g_bases: Sequence[str],
    proof: BHK22ProofEnvelope,
) -> bool:
    try:
        statement = {
            "delta": canonical_point(delta_hex, name="Figure 8 delta"),
            "x_points": [item.X for item in ciphertexts],
            "h_bases": [canonical_point(item, name="Figure 8 H base") for item in h_bases],
            "g_bases": [canonical_point(item, name="Figure 8 G base") for item in g_bases],
        }
        return _proof_matches(
            proof,
            kind="figure8-randomness",
            algorithm=ALGORITHM_SUITE.figure8_proof,
            context=context,
            statement=statement,
        ) and sigma_reference.verify_randomness_consistency(
            delta_hex=statement["delta"],
            x_points=statement["x_points"],
            h_bases=statement["h_bases"],
            g_bases=statement["g_bases"],
            proof=proof.sigma,
            label=_proof_label("figure8-randomness", context),
        )
    except (AttributeError, TypeError, ValueError):
        return False


def prove_forwarding_logs(
    *,
    context: BHK22ProofContext,
    bases: Sequence[str],
    points: Sequence[str],
    logs: Sequence[int],
) -> BHK22ProofEnvelope:
    statement = {
        "bases": [canonical_point(item, name="forwarding base") for item in bases],
        "points": [canonical_point(item, name="forwarding point") for item in points],
    }
    proof = sigma_reference.prove_discrete_logs(
        statement["bases"],
        statement["points"],
        logs,
        label=_proof_label("forwarding", context),
    )
    return BHK22ProofEnvelope(
        proof_kind="forwarding",
        algorithm=ALGORITHM_SUITE.verifier_forwarding_proof,
        context=context,
        statement_digest=_statement_digest("forwarding", context, statement),
        sigma=proof,
    )


def verify_forwarding_logs(
    *,
    context: BHK22ProofContext,
    bases: Sequence[str],
    points: Sequence[str],
    proof: BHK22ProofEnvelope,
) -> bool:
    try:
        statement = {
            "bases": [canonical_point(item, name="forwarding base") for item in bases],
            "points": [canonical_point(item, name="forwarding point") for item in points],
        }
        return _proof_matches(
            proof,
            kind="forwarding",
            algorithm=ALGORITHM_SUITE.verifier_forwarding_proof,
            context=context,
            statement=statement,
        ) and sigma_reference.verify_discrete_logs(
            statement["bases"],
            statement["points"],
            proof.sigma,
            label=_proof_label("forwarding", context),
        )
    except (TypeError, ValueError):
        return False


def prove_correct_decryption(
    *,
    context: BHK22ProofContext,
    generator_hex: str,
    ciphertext: BHK22ElGamalCiphertext,
    public_key_hex: str,
    plaintext_hex: str,
    secret_key: int,
) -> BHK22ProofEnvelope:
    statement = {
        "generator": canonical_point(generator_hex, name="decryption generator"),
        "ciphertext": ciphertext.to_obj(),
        "public_key": canonical_point(public_key_hex, name="decryption public key"),
        "plaintext": canonical_point(plaintext_hex, name="decryption plaintext"),
    }
    proof = sigma_reference.prove_decryption_correctness(
        key_base_hex=statement["generator"],
        ciphertext_c1_hex=ciphertext.X,
        ciphertext_c2_hex=ciphertext.Y,
        public_key_hex=statement["public_key"],
        plaintext_hex=statement["plaintext"],
        secret_key=secret_key,
        label=_proof_label("correct-decryption", context),
    )
    return BHK22ProofEnvelope(
        proof_kind="correct-decryption",
        algorithm=ALGORITHM_SUITE.correct_decryption_proof,
        context=context,
        statement_digest=_statement_digest(
            "correct-decryption", context, statement
        ),
        sigma=proof,
    )


def verify_correct_decryption(
    *,
    context: BHK22ProofContext,
    generator_hex: str,
    ciphertext: BHK22ElGamalCiphertext,
    public_key_hex: str,
    plaintext_hex: str,
    proof: BHK22ProofEnvelope,
) -> bool:
    try:
        statement = {
            "generator": canonical_point(generator_hex, name="decryption generator"),
            "ciphertext": ciphertext.to_obj(),
            "public_key": canonical_point(public_key_hex, name="decryption public key"),
            "plaintext": canonical_point(plaintext_hex, name="decryption plaintext"),
        }
        return _proof_matches(
            proof,
            kind="correct-decryption",
            algorithm=ALGORITHM_SUITE.correct_decryption_proof,
            context=context,
            statement=statement,
        ) and sigma_reference.verify_decryption_correctness(
            key_base_hex=statement["generator"],
            ciphertext_c1_hex=ciphertext.X,
            ciphertext_c2_hex=ciphertext.Y,
            public_key_hex=statement["public_key"],
            plaintext_hex=statement["plaintext"],
            proof=proof.sigma,
            label=_proof_label("correct-decryption", context),
        )
    except (TypeError, ValueError):
        return False


def proof_size_bytes(proof: BHK22ProofEnvelope) -> int:
    if not isinstance(proof, BHK22ProofEnvelope):
        raise TypeError("proof must be BHK22ProofEnvelope")
    return len(canonical_json(proof.to_obj()))


def primitive_size_bytes(value: Any) -> int:
    """Return canonical public bytes for Step-2 primitive accounting."""

    if not hasattr(value, "to_obj"):
        raise TypeError("public primitive must provide to_obj()")
    return len(canonical_json(value.to_obj()))


def timed_crypto_call(callable_, /, *args, **kwargs) -> tuple[Any, float]:
    """Execute one primitive and return ``(result, cpu_ms)`` independently."""

    if not callable(callable_):
        raise TypeError("timed crypto target must be callable")
    started = perf_counter()
    result = callable_(*args, **kwargs)
    return result, (perf_counter() - started) * 1000.0


__all__ = [
    "BHK22AAD",
    "BHK22AEADCiphertext",
    "BHK22CoordinateOpening",
    "BHK22ElGamalCiphertext",
    "BHK22FullOpening",
    "BHK22MerklePathEntry",
    "BHK22ProofContext",
    "BHK22ProofEnvelope",
    "BHK22RecipientCiphertext",
    "BHK22VectorCommitment",
    "RECIPIENT_PURPOSES",
    "RECIPIENT_PURPOSE_ROLES",
    "PROOF_KIND_CONTRACT",
    "aead_decrypt",
    "aead_encrypt",
    "combine_public_value_ciphertexts",
    "commit_vector",
    "decrypt_public_value",
    "derive_aead_key",
    "encrypt_public_value",
    "hkdf_sha256",
    "open_vector_coordinate",
    "proof_size_bytes",
    "primitive_size_bytes",
    "prove_correct_decryption",
    "prove_figure7",
    "prove_figure8_randomness",
    "prove_forwarding_logs",
    "recipient_decrypt",
    "recipient_encrypt",
    "timed_crypto_call",
    "verify_coordinate_opening",
    "verify_correct_decryption",
    "verify_figure7",
    "verify_figure8_randomness",
    "verify_forwarding_logs",
    "verify_full_opening",
]
