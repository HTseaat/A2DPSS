"""Canonical encrypted BHK source coordinates shared by fixtures and operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from beaver.baselines.bhk22_context import (
    canonical_point, digest_hex, domain_digest, exact_fields, strict_int,
)
from beaver.baselines.bhk22_crypto import BHK22ElGamalCiphertext
from beaver.g1_backend import G1Point

ENCRYPTED_SOURCE_VECTOR_SCHEMA = "bhk22-encrypted-source-vector/v1"
ENCRYPTED_SOURCE_COORDINATE_SCHEMA = "bhk22-encrypted-source-coordinate/v1"
_SOURCE_VECTOR_DOMAIN = b"ATCaaS/BHK22/encrypted-source-vector/v1\x00"


def _point(value: Any, *, name: str) -> str:
    point = canonical_point(value, name=name)
    if G1Point.from_hex(point).is_identity():
        raise ValueError(f"{name} must not be the identity")
    return point


@dataclass(frozen=True)
class BHK22EncryptedSourceCoordinate:
    coordinate_index: int
    evaluation_point: int
    generator_hex: str
    public_key_hex: str
    ciphertext: BHK22ElGamalCiphertext

    def __post_init__(self) -> None:
        index = strict_int(self.coordinate_index, name="coordinate_index")
        point = strict_int(self.evaluation_point, name="evaluation_point", minimum=1)
        if point != index + 1:
            raise ValueError("encrypted source evaluation point mismatch")
        if not isinstance(self.ciphertext, BHK22ElGamalCiphertext):
            raise TypeError("encrypted source ciphertext has the wrong type")
        object.__setattr__(self, "generator_hex", _point(self.generator_hex, name="source generator"))
        object.__setattr__(self, "public_key_hex", _point(self.public_key_hex, name="source public key"))

    def to_obj(self) -> dict[str, Any]:
        return {
            "schema": ENCRYPTED_SOURCE_COORDINATE_SCHEMA,
            "coordinate_index": self.coordinate_index,
            "evaluation_point": self.evaluation_point,
            "generator_hex": self.generator_hex,
            "public_key_hex": self.public_key_hex,
            "ciphertext": self.ciphertext.to_obj(),
        }

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22EncryptedSourceCoordinate":
        exact_fields(value, {"schema", "coordinate_index", "evaluation_point", "generator_hex", "public_key_hex", "ciphertext"}, name="encrypted source coordinate")
        if value["schema"] != ENCRYPTED_SOURCE_COORDINATE_SCHEMA:
            raise ValueError("unsupported encrypted source coordinate")
        return cls(
            coordinate_index=value["coordinate_index"],
            evaluation_point=value["evaluation_point"],
            generator_hex=value["generator_hex"],
            public_key_hex=value["public_key_hex"],
            ciphertext=BHK22ElGamalCiphertext.from_obj(value["ciphertext"]),
        )


@dataclass(frozen=True)
class BHK22EncryptedSourceVector:
    coordinates: Sequence[BHK22EncryptedSourceCoordinate]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        coordinates = tuple(self.coordinates)
        if not coordinates or not all(isinstance(item, BHK22EncryptedSourceCoordinate) for item in coordinates):
            raise ValueError("encrypted source vector must be non-empty and typed")
        if tuple(item.coordinate_index for item in coordinates) != tuple(range(len(coordinates))):
            raise ValueError("encrypted source coordinates must be ordered 0..n_r-1")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "digest", domain_digest(_SOURCE_VECTOR_DOMAIN, self.digest_material()))

    def digest_material(self) -> dict[str, Any]:
        return {"schema": ENCRYPTED_SOURCE_VECTOR_SCHEMA, "coordinates": [item.to_obj() for item in self.coordinates]}

    def to_obj(self) -> dict[str, Any]:
        value = self.digest_material()
        value["encrypted_source_digest"] = self.digest
        return value

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22EncryptedSourceVector":
        exact_fields(value, {"schema", "coordinates", "encrypted_source_digest"}, name="encrypted source vector")
        if value["schema"] != ENCRYPTED_SOURCE_VECTOR_SCHEMA:
            raise ValueError("unsupported encrypted source vector")
        result = cls(tuple(BHK22EncryptedSourceCoordinate.from_obj(item) for item in value["coordinates"]))
        if result.digest != digest_hex(value["encrypted_source_digest"], name="encrypted_source_digest"):
            raise ValueError("encrypted source vector digest mismatch")
        return result


