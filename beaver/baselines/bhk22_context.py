"""Canonical public context and CRS for BHK+22 + DAG-TOB.

Step 1 owns only public, immutable protocol metadata.  Private fixture shards
and one-shot role state live in :mod:`beaver.baselines.bhk22_experiment`.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from beaver.atcaas_params import ATCaaSProtocolParameters
from beaver.baselines.bhk22_contract import (
    ALGORITHM_SUITE,
    BHK22IndexContract,
    BHK22_PROTOCOL_KIND,
    BHK22_PROTOCOL_VERSION,
    ROLE_TYPES,
    make_crs_contract_metadata,
    validate_bhk22_parameters,
    validate_crs_construction_metadata,
    validate_public_object,
)
from beaver.g1_backend import FIELD_ORDER, G1Point


CONTEXT_KIND = "atcaas-bhk22-public-context"
ENCRYPTED_CONTEXT_KIND = "atcaas-bhk22-encrypted-source-context"
CONTEXT_VERSION = 1
CRS_KIND = "atcaas-bhk22-crs"
CRS_VERSION = 1
COMMITTEE_MAP_KIND = "atcaas-bhk22-committee-map"
COMMITTEE_MAP_VERSION = 2
KEY_CATALOG_KIND = "atcaas-bhk22-role-public-key-catalog"
KEY_CATALOG_VERSION = 1

COMMITTEE_SELECTION_EXTERNAL = "externally-established"
COMMITTEE_SELECTION_VRF_EXACT = "vrf-bernoulli-exact-size"
COMMITTEE_SELECTION_MODES = frozenset(
    {COMMITTEE_SELECTION_EXTERNAL, COMMITTEE_SELECTION_VRF_EXACT}
)
COMMITTEE_SORTITION_DOMAIN = b"ATCaaS/BHK22/committee-sortition/v1\x00"

CONTEXT_DIGEST_DOMAIN = b"ATCaaS/BHK22/context/v1\x00"
CRS_DIGEST_DOMAIN = b"ATCaaS/BHK22/crs/v1\x00"
HASH_TO_G1_DOMAIN = b"ATCaaS/BHK22/hash-to-g1/v1\x00"

_BLS12_381_BASE_FIELD = int(
    "1a0111ea397fe69a4b1ba7b6434bacd7"
    "64774b84f38512bf6730d2a0f6b0f624"
    "1eabfffeb153ffffb9feffffffffaaab",
    16,
)
_BLS12_381_G1_COFACTOR = int("396c8c005555e1568c00aaab0000aaab", 16)
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _typed_canonical(value: Any) -> Any:
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ["bytes", bytes(value).hex()]
    if isinstance(value, Mapping):
        pairs = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical mapping keys must be strings")
            pairs.append([key, _typed_canonical(item)])
        return ["map", sorted(pairs, key=lambda pair: pair[0])]
    if isinstance(value, (list, tuple)):
        return ["list", [_typed_canonical(item) for item in value]]
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def domain_digest(domain: bytes, material: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        bytes(domain) + canonical_json(_typed_canonical(material))
    ).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def decode_object(data: bytes | bytearray | memoryview | str, *, name: str) -> dict:
    raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed {name} JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must encode a JSON object")
    return value


def exact_fields(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    missing = expected - set(value)
    extra = set(value) - expected
    if missing:
        raise ValueError(f"{name} is missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"{name} has unknown fields: {sorted(extra)}")


def strict_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def nonempty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def digest_hex(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


def canonical_scalar(value: Any, *, name: str, nonzero: bool = False) -> int:
    if isinstance(value, str):
        if value != "0" and (not value.isdigit() or value.startswith("0")):
            raise ValueError(f"{name} must be a canonical decimal scalar")
        value = int(value)
    scalar = strict_int(value, name=name)
    if scalar >= FIELD_ORDER or (nonzero and scalar == 0):
        raise ValueError(f"{name} is outside the scalar field")
    return scalar


def canonical_point(value: str | G1Point, *, name: str) -> str:
    try:
        point = value if isinstance(value, G1Point) else G1Point.from_hex(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid G1 point") from exc
    if point.is_identity():
        raise ValueError(f"{name} must not be the identity")
    normalized = point.to_hex()
    if not _is_in_g1_prime_subgroup(normalized):
        raise ValueError(f"{name} must be in the prime-order G1 subgroup")
    return normalized


def _jacobian_double(point: tuple[int, int, int]) -> tuple[int, int, int]:
    x, y, z = point
    if z == 0 or y == 0:
        return (0, 1, 0)
    field = _BLS12_381_BASE_FIELD
    a = x * x % field
    b = y * y % field
    c = b * b % field
    d = 2 * ((x + b) * (x + b) - a - c) % field
    e = 3 * a % field
    f = e * e % field
    x3 = (f - 2 * d) % field
    y3 = (e * (d - x3) - 8 * c) % field
    z3 = 2 * y * z % field
    return (x3, y3, z3)


def _jacobian_add_affine(
    point: tuple[int, int, int], affine: tuple[int, int]
) -> tuple[int, int, int]:
    x1, y1, z1 = point
    x2, y2 = affine
    if z1 == 0:
        return (x2, y2, 1)
    field = _BLS12_381_BASE_FIELD
    z1z1 = z1 * z1 % field
    u2 = x2 * z1z1 % field
    s2 = y2 * z1 % field * z1z1 % field
    if u2 == x1:
        if s2 == y1:
            return _jacobian_double(point)
        return (0, 1, 0)
    h = (u2 - x1) % field
    hh = h * h % field
    i = 4 * hh % field
    j = h * i % field
    r = 2 * (s2 - y1) % field
    v = x1 * i % field
    x3 = (r * r - j - 2 * v) % field
    y3 = (r * (v - x3) - 2 * y1 * j) % field
    z3 = ((z1 + h) * (z1 + h) - z1z1 - hh) % field
    return (x3, y3, z3)


def _is_in_g1_prime_subgroup(normalized: str) -> bool:
    if not normalized.startswith("(") or not normalized.endswith(")"):
        return False
    try:
        x_hex, y_hex = normalized[1:-1].split(",")
        x = int(x_hex, 16)
        y = int(y_hex, 16)
    except (TypeError, ValueError):
        return False
    field = _BLS12_381_BASE_FIELD
    if not (0 <= x < field and 0 <= y < field):
        return False
    if y * y % field != (pow(x, 3, field) + 4) % field:
        return False
    result = (0, 1, 0)
    affine = (x, y)
    for bit in bin(FIELD_ORDER)[2:]:
        result = _jacobian_double(result)
        if bit == "1":
            result = _jacobian_add_affine(result, affine)
    return result[2] == 0


def _hash_frame(domain: bytes, message: bytes, counter: int) -> bytes:
    return (
        HASH_TO_G1_DOMAIN
        + len(domain).to_bytes(2, "big")
        + domain
        + len(message).to_bytes(4, "big")
        + message
        + counter.to_bytes(4, "big")
    )


def hash_to_g1(domain: bytes, message: bytes) -> str:
    """Deterministically map bytes to a non-identity BLS12-381 G1 point.

    The mapping hashes candidate x-coordinates, selects the even square root,
    clears the G1 cofactor, and lets the AMCL backend normalize and validate the
    result.  It never derives a base as a known scalar multiple of another
    published base.
    """

    if not isinstance(domain, bytes) or not domain:
        raise ValueError("hash-to-G1 domain must be non-empty bytes")
    if not isinstance(message, bytes):
        raise TypeError("hash-to-G1 message must be bytes")
    field = _BLS12_381_BASE_FIELD
    for counter in range(1 << 32):
        x = int.from_bytes(
            hashlib.sha256(_hash_frame(domain, message, counter)).digest(),
            "big",
        ) % field
        rhs = (pow(x, 3, field) + 4) % field
        y = pow(rhs, (field + 1) // 4, field)
        if pow(y, 2, field) != rhs:
            continue
        if y & 1:
            y = field - y
        raw = b"\x04" + x.to_bytes(48, "big") + y.to_bytes(48, "big")
        point = G1Point.from_hex(raw.hex()) * _BLS12_381_G1_COFACTOR
        if not point.is_identity():
            return point.to_hex()
    raise RuntimeError("hash-to-G1 counter space exhausted")


def derive_nonzero_scalar(seed: str, label: str) -> int:
    seed = nonempty_text(seed, name="scalar seed")
    label = nonempty_text(label, name="scalar label")
    counter = 0
    while True:
        raw = hashlib.sha256(
            b"ATCaaS/BHK22/fixture-scalar/v1\x00"
            + seed.encode("utf-8")
            + b"\x00"
            + label.encode("utf-8")
            + b"\x00"
            + counter.to_bytes(4, "big")
        ).digest()
        scalar = int.from_bytes(raw, "big") % FIELD_ORDER
        if scalar:
            return scalar
        counter += 1


@dataclass(frozen=True)
class BHK22CommitteeMap:
    role_type: str
    role_to_node: Mapping[int, int]
    evaluation_points: Mapping[int, int]
    n_r: int
    M: int
    selection_mode: str = COMMITTEE_SELECTION_EXTERNAL
    selection_attempt: int = 0

    def __post_init__(self) -> None:
        role_type = str(self.role_type)
        if role_type not in ROLE_TYPES:
            raise ValueError("unsupported BHK22 role type")
        n_r = strict_int(self.n_r, name="n_r", minimum=1)
        M = strict_int(self.M, name="M", minimum=1)
        selection_mode = str(self.selection_mode)
        if selection_mode not in COMMITTEE_SELECTION_MODES:
            raise ValueError("unsupported BHK22 committee selection mode")
        selection_attempt = strict_int(
            self.selection_attempt, name="committee selection attempt"
        )
        if (
            selection_mode == COMMITTEE_SELECTION_EXTERNAL
            and selection_attempt != 0
        ):
            raise ValueError("externally established committee cannot have a retry attempt")
        if n_r > M:
            raise ValueError("committee size must not exceed M")
        role_to_node = {
            strict_int(role, name="role id"): strict_int(
                node, name="network node id"
            )
            for role, node in self.role_to_node.items()
        }
        evaluation_points = {
            strict_int(role, name="evaluation role id"): strict_int(
                point, name="evaluation point", minimum=1
            )
            for role, point in self.evaluation_points.items()
        }
        expected = set(range(n_r))
        if set(role_to_node) != expected or set(evaluation_points) != expected:
            raise ValueError("committee maps must cover exactly role ids 0..n_r-1")
        if len(set(role_to_node.values())) != n_r:
            raise ValueError("one role type must map to distinct network nodes")
        if any(node >= M for node in role_to_node.values()):
            raise ValueError("committee node id is outside M")
        if evaluation_points != {role: role + 1 for role in range(n_r)}:
            raise ValueError("evaluation point must equal role_id + 1")
        object.__setattr__(self, "role_type", role_type)
        object.__setattr__(self, "role_to_node", MappingProxyType(role_to_node))
        object.__setattr__(
            self, "evaluation_points", MappingProxyType(evaluation_points)
        )
        object.__setattr__(self, "n_r", n_r)
        object.__setattr__(self, "M", M)
        object.__setattr__(self, "selection_mode", selection_mode)
        object.__setattr__(self, "selection_attempt", selection_attempt)

    @property
    def requires_vrf_proof(self) -> bool:
        return self.selection_mode == COMMITTEE_SELECTION_VRF_EXACT

    def to_obj(self) -> dict[str, Any]:
        return {
            "kind": COMMITTEE_MAP_KIND,
            "version": COMMITTEE_MAP_VERSION,
            "role_type": self.role_type,
            "role_to_node": {
                str(role): node for role, node in self.role_to_node.items()
            },
            "evaluation_points": {
                str(role): point for role, point in self.evaluation_points.items()
            },
            "n_r": self.n_r,
            "M": self.M,
            "selection_mode": self.selection_mode,
            "selection_attempt": self.selection_attempt,
        }

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22CommitteeMap":
        version = value.get("version") if isinstance(value, Mapping) else None
        fields = {
            "kind",
            "version",
            "role_type",
            "role_to_node",
            "evaluation_points",
            "n_r",
            "M",
            "selection_mode",
            "selection_attempt",
        }
        exact_fields(value, fields, name="BHK22 committee map")
        if (
            value["kind"] != COMMITTEE_MAP_KIND
            or version != COMMITTEE_MAP_VERSION
        ):
            raise ValueError("unsupported BHK22 committee map")
        return cls(
            role_type=value["role_type"],
            role_to_node={int(k): v for k, v in value["role_to_node"].items()},
            evaluation_points={
                int(k): v for k, v in value["evaluation_points"].items()
            },
            n_r=value["n_r"],
            M=value["M"],
            selection_mode=value["selection_mode"],
            selection_attempt=value["selection_attempt"],
        )


def bhk22_committee_sortition_label(
    *,
    trial_id: str,
    trial_epoch: int,
    source_epoch: int,
    target_epoch: int,
    sid: str,
    transfer_seq: int,
    role_trace_digest: str,
    role_type: str,
    selection_attempt: int,
) -> bytes:
    """Return the context-bound VRF label for one ephemeral BHK committee."""

    if role_type not in ROLE_TYPES:
        raise ValueError("unsupported BHK22 sortition role type")
    material = {
        "trial_id": nonempty_text(trial_id, name="trial_id"),
        "trial_epoch": strict_int(trial_epoch, name="trial_epoch"),
        "source_epoch": strict_int(source_epoch, name="source_epoch", minimum=1),
        "target_epoch": strict_int(target_epoch, name="target_epoch", minimum=2),
        "sid": nonempty_text(sid, name="sid"),
        "transfer_seq": strict_int(transfer_seq, name="transfer_seq"),
        "role_trace_digest": digest_hex(
            role_trace_digest, name="role_trace_digest"
        ),
        "role_type": role_type,
        "selection_attempt": strict_int(
            selection_attempt, name="committee selection attempt"
        ),
    }
    return COMMITTEE_SORTITION_DOMAIN + canonical_json(material)


def bhk22_context_committee_sortition_label(
    context: "BHK22PublicContext", role_type: str
) -> bytes:
    if not isinstance(context, BHK22PublicContext):
        raise TypeError("committee sortition requires BHK22PublicContext")
    committee = context.committees[role_type]
    if not committee.requires_vrf_proof:
        raise ValueError("committee does not use VRF selection")
    return bhk22_committee_sortition_label(
        trial_id=context.trial_id,
        trial_epoch=context.trial_epoch,
        source_epoch=context.source_epoch,
        target_epoch=context.target_epoch,
        sid=context.sid,
        transfer_seq=context.transfer_seq,
        role_trace_digest=context.role_trace_digest,
        role_type=role_type,
        selection_attempt=committee.selection_attempt,
    )


@dataclass(frozen=True)
class BHK22RolePublicKeyCatalog:
    role_type: str
    key_base_hex: str
    public_keys: Mapping[int, str]
    n_r: int

    def __post_init__(self) -> None:
        role_type = str(self.role_type)
        if role_type not in ROLE_TYPES:
            raise ValueError("unsupported BHK22 key-catalog role type")
        n_r = strict_int(self.n_r, name="n_r", minimum=1)
        base = canonical_point(self.key_base_hex, name="recipient key base")
        keys = {
            strict_int(role, name="key role id"): canonical_point(
                key, name="recipient public key"
            )
            for role, key in self.public_keys.items()
        }
        if set(keys) != set(range(n_r)):
            raise ValueError("public key catalog must cover role ids 0..n_r-1")
        if len(set(keys.values())) != n_r:
            raise ValueError("recipient public keys must be distinct")
        object.__setattr__(self, "role_type", role_type)
        object.__setattr__(self, "key_base_hex", base)
        object.__setattr__(self, "public_keys", MappingProxyType(keys))
        object.__setattr__(self, "n_r", n_r)

    def to_obj(self) -> dict[str, Any]:
        return {
            "kind": KEY_CATALOG_KIND,
            "version": KEY_CATALOG_VERSION,
            "role_type": self.role_type,
            "key_base_hex": self.key_base_hex,
            "public_keys": {
                str(role): key for role, key in self.public_keys.items()
            },
            "n_r": self.n_r,
        }

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22RolePublicKeyCatalog":
        exact_fields(
            value,
            {"kind", "version", "role_type", "key_base_hex", "public_keys", "n_r"},
            name="BHK22 role public-key catalog",
        )
        if value["kind"] != KEY_CATALOG_KIND or value["version"] != 1:
            raise ValueError("unsupported BHK22 role public-key catalog")
        return cls(
            role_type=value["role_type"],
            key_base_hex=value["key_base_hex"],
            public_keys={int(k): v for k, v in value["public_keys"].items()},
            n_r=value["n_r"],
        )


def _derive_crs_public_points(n_r: int, derivation_seed: str) -> tuple[tuple[str, ...], str]:
    n_r = BHK22IndexContract(n_r).n_r
    seed = nonempty_text(derivation_seed, name="CRS derivation seed").encode("utf-8")
    generators = tuple(
        hash_to_g1(b"crs/G", seed + b"\x00" + index.to_bytes(4, "big"))
        for index in range(n_r + 1)
    )
    h0 = hash_to_g1(b"crs/H0-no-witness", seed)
    return generators, h0


@dataclass(frozen=True)
class BHK22CRS:
    n_r: int
    derivation_seed: str
    generators: Sequence[str]
    public_value_keys: Sequence[str]
    construction_metadata: Mapping[str, Any]
    crs_digest: str = field(init=False)

    def __post_init__(self) -> None:
        n_r = BHK22IndexContract(self.n_r).n_r
        seed = nonempty_text(self.derivation_seed, name="CRS derivation seed")
        generators = tuple(
            canonical_point(point, name="G generator") for point in self.generators
        )
        keys = tuple(
            canonical_point(point, name="H public-value key")
            for point in self.public_value_keys
        )
        if len(generators) != n_r + 1 or len(keys) != n_r + 1:
            raise ValueError("CRS must contain G_0..G_n and H_0..H_n")
        if len(set(generators)) != n_r + 1:
            raise ValueError("G generators must be distinct")
        expected_generators, expected_h0 = _derive_crs_public_points(n_r, seed)
        if generators != expected_generators:
            raise ValueError("G generators do not match hash-to-G1 derivation")
        if keys[0] != expected_h0:
            raise ValueError("H_0 does not match no-witness hash derivation")
        metadata = dict(self.construction_metadata)
        validate_crs_construction_metadata(metadata, n_r=n_r)
        material = {
            "kind": CRS_KIND,
            "version": CRS_VERSION,
            "n_r": n_r,
            "derivation_seed": seed,
            "generators": list(generators),
            "public_value_keys": list(keys),
            "construction_metadata": metadata,
        }
        object.__setattr__(self, "n_r", n_r)
        object.__setattr__(self, "derivation_seed", seed)
        object.__setattr__(self, "generators", generators)
        object.__setattr__(self, "public_value_keys", keys)
        object.__setattr__(self, "construction_metadata", MappingProxyType(metadata))
        object.__setattr__(self, "crs_digest", domain_digest(CRS_DIGEST_DOMAIN, material))

    def to_obj(self) -> dict[str, Any]:
        return {
            "kind": CRS_KIND,
            "version": CRS_VERSION,
            "n_r": self.n_r,
            "derivation_seed": self.derivation_seed,
            "generators": list(self.generators),
            "public_value_keys": list(self.public_value_keys),
            "construction_metadata": dict(self.construction_metadata),
            "crs_digest": self.crs_digest,
        }

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22CRS":
        exact_fields(
            value,
            {
                "kind",
                "version",
                "n_r",
                "derivation_seed",
                "generators",
                "public_value_keys",
                "construction_metadata",
                "crs_digest",
            },
            name="BHK22 CRS",
        )
        if value["kind"] != CRS_KIND or value["version"] != 1:
            raise ValueError("unsupported BHK22 CRS")
        supplied = digest_hex(value["crs_digest"], name="crs_digest")
        crs = cls(
            n_r=value["n_r"],
            derivation_seed=value["derivation_seed"],
            generators=value["generators"],
            public_value_keys=value["public_value_keys"],
            construction_metadata=value["construction_metadata"],
        )
        if crs.crs_digest != supplied:
            raise ValueError("BHK22 CRS digest mismatch")
        return crs


def build_bhk22_crs(
    *,
    n_r: int,
    derivation_seed: str,
    shareholder_secret_keys: Sequence[int],
) -> BHK22CRS:
    index = BHK22IndexContract(n_r)
    secrets = tuple(
        canonical_scalar(value, name="shareholder public-value secret", nonzero=True)
        for value in shareholder_secret_keys
    )
    if len(secrets) != index.n_r:
        raise ValueError("need exactly n_r shareholder public-value secrets")
    generators, h0 = _derive_crs_public_points(index.n_r, derivation_seed)
    public_value_keys = (h0,) + tuple(
        (G1Point.from_hex(generators[role + 1]) * secrets[role]).to_hex()
        for role in index.role_ids
    )
    return BHK22CRS(
        n_r=index.n_r,
        derivation_seed=derivation_seed,
        generators=generators,
        public_value_keys=public_value_keys,
        construction_metadata=make_crs_contract_metadata(index.n_r),
    )


def derive_role_key_base(role_type: str, derivation_seed: str) -> str:
    if role_type not in ROLE_TYPES:
        raise ValueError("unsupported BHK22 role type")
    seed = nonempty_text(derivation_seed, name="key-base derivation seed")
    return hash_to_g1(b"recipient-key-base/" + role_type.encode("ascii"), seed.encode("utf-8"))


@dataclass(frozen=True)
class BHK22PublicContext:
    trial_id: str
    trial_epoch: int
    source_epoch: int
    target_epoch: int
    sid: str
    source_committee: str
    successor_committee: str
    transfer_seq: int
    parameters: ATCaaSProtocolParameters
    crs: BHK22CRS
    committees: Mapping[str, BHK22CommitteeMap]
    recipient_key_catalogs: Mapping[str, BHK22RolePublicKeyCatalog]
    source_public_values: Sequence[str]
    dag_config_digest: str
    coin_config_digest: str
    role_trace_digest: str
    encrypted_source: Any = None
    root_context_digest: str = field(init=False)

    def __post_init__(self) -> None:
        trial_id = nonempty_text(self.trial_id, name="trial_id")
        trial_epoch = strict_int(self.trial_epoch, name="trial_epoch")
        source_epoch = strict_int(self.source_epoch, name="source_epoch", minimum=1)
        target_epoch = strict_int(self.target_epoch, name="target_epoch", minimum=2)
        if target_epoch != source_epoch + 1:
            raise ValueError("target_epoch must equal source_epoch + 1")
        sid = nonempty_text(self.sid, name="sid")
        source = nonempty_text(self.source_committee, name="source_committee")
        successor = nonempty_text(
            self.successor_committee, name="successor_committee"
        )
        transfer_seq = strict_int(self.transfer_seq, name="transfer_seq")
        parameters = validate_bhk22_parameters(self.parameters)
        if not isinstance(self.crs, BHK22CRS) or self.crs.n_r != parameters.n_r:
            raise ValueError("CRS does not match BHK22 parameters")
        committees = dict(self.committees)
        catalogs = dict(self.recipient_key_catalogs)
        if set(committees) != set(ROLE_TYPES) or set(catalogs) != set(ROLE_TYPES):
            raise ValueError("context requires all four role maps/key catalogs")
        for role_type in ROLE_TYPES:
            committee = committees[role_type]
            catalog = catalogs[role_type]
            if not isinstance(committee, BHK22CommitteeMap):
                raise TypeError("committee entry has wrong type")
            if not isinstance(catalog, BHK22RolePublicKeyCatalog):
                raise TypeError("public-key catalog entry has wrong type")
            if (
                committee.role_type != role_type
                or catalog.role_type != role_type
                or committee.n_r != parameters.n_r
                or committee.M != parameters.M
                or catalog.n_r != parameters.n_r
            ):
                raise ValueError("role map/key catalog does not match context")
        source_values = tuple(
            canonical_point(value, name="source public value")
            for value in self.source_public_values
        )
        if self.encrypted_source is None:
            if len(source_values) != parameters.n_r:
                raise ValueError("source public values must contain n_r points")
        else:
            # Import at validation time to keep the context/crypto wire types
            # independent of the operation implementation.
            from beaver.baselines.bhk22_encrypted_source import BHK22EncryptedSourceVector

            if source_values:
                raise ValueError("encrypted source context forbids plaintext source values")
            if not isinstance(self.encrypted_source, BHK22EncryptedSourceVector):
                raise TypeError("encrypted source must be a typed source vector")
            if len(self.encrypted_source.coordinates) != parameters.n_r:
                raise ValueError("encrypted source must contain n_r coordinates")
            for role, coordinate in enumerate(self.encrypted_source.coordinates):
                if (
                    coordinate.generator_hex != self.crs.generators[role + 1]
                    or coordinate.public_key_hex != self.crs.public_value_keys[role + 1]
                ):
                    raise ValueError("encrypted source coordinate does not match CRS")
        for name, value in (
            ("dag_config_digest", self.dag_config_digest),
            ("coin_config_digest", self.coin_config_digest),
            ("role_trace_digest", self.role_trace_digest),
        ):
            digest_hex(value, name=name)
        object.__setattr__(self, "trial_id", trial_id)
        object.__setattr__(self, "trial_epoch", trial_epoch)
        object.__setattr__(self, "source_epoch", source_epoch)
        object.__setattr__(self, "target_epoch", target_epoch)
        object.__setattr__(self, "sid", sid)
        object.__setattr__(self, "source_committee", source)
        object.__setattr__(self, "successor_committee", successor)
        object.__setattr__(self, "transfer_seq", transfer_seq)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "committees", MappingProxyType(committees))
        object.__setattr__(self, "recipient_key_catalogs", MappingProxyType(catalogs))
        object.__setattr__(self, "source_public_values", source_values)
        digest = domain_digest(CONTEXT_DIGEST_DOMAIN, self.digest_material())
        object.__setattr__(self, "root_context_digest", digest)
        validate_public_object(self.to_obj())

    def digest_material(self) -> dict[str, Any]:
        material = {
            "kind": CONTEXT_KIND if self.encrypted_source is None else ENCRYPTED_CONTEXT_KIND,
            "version": CONTEXT_VERSION,
            "protocol": {
                "kind": BHK22_PROTOCOL_KIND,
                "version": BHK22_PROTOCOL_VERSION,
                "figures": [6, 7, 8],
                "strict_chaining_supported": True,
            },
            "trial_id": self.trial_id,
            "trial_epoch": self.trial_epoch,
            "source_epoch": self.source_epoch,
            "target_epoch": self.target_epoch,
            "sid": self.sid,
            "source_committee": self.source_committee,
            "successor_committee": self.successor_committee,
            "transfer_seq": self.transfer_seq,
            "parameters": self.parameters.to_config_obj(),
            "crs": self.crs.to_obj(),
            "committees": {
                role: self.committees[role].to_obj() for role in ROLE_TYPES
            },
            "recipient_key_catalogs": {
                role: self.recipient_key_catalogs[role].to_obj()
                for role in ROLE_TYPES
            },
            "source_public_values": list(self.source_public_values),
            "algorithm_suite": ALGORITHM_SUITE.to_obj(),
            "dag_config_digest": self.dag_config_digest,
            "coin_config_digest": self.coin_config_digest,
            "role_trace_digest": self.role_trace_digest,
        }
        if self.encrypted_source is not None:
            del material["source_public_values"]
            material["encrypted_source"] = self.encrypted_source.to_obj()
        return material

    def to_obj(self) -> dict[str, Any]:
        value = self.digest_material()
        value["root_context_digest"] = self.root_context_digest
        return value

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22PublicContext":
        encrypted = isinstance(value, Mapping) and value.get("kind") == ENCRYPTED_CONTEXT_KIND
        exact_fields(
            value,
            {
                "kind",
                "version",
                "protocol",
                "trial_id",
                "trial_epoch",
                "source_epoch",
                "target_epoch",
                "sid",
                "source_committee",
                "successor_committee",
                "transfer_seq",
                "parameters",
                "crs",
                "committees",
                "recipient_key_catalogs",
                "encrypted_source" if encrypted else "source_public_values",
                "algorithm_suite",
                "dag_config_digest",
                "coin_config_digest",
                "role_trace_digest",
                "root_context_digest",
            },
            name="BHK22 public context",
        )
        if value["kind"] not in {CONTEXT_KIND, ENCRYPTED_CONTEXT_KIND} or value["version"] != 1:
            raise ValueError("unsupported BHK22 public context")
        if value["protocol"] != {
            "kind": BHK22_PROTOCOL_KIND,
            "version": BHK22_PROTOCOL_VERSION,
            "figures": [6, 7, 8],
            "strict_chaining_supported": True,
        }:
            raise ValueError("BHK22 protocol selection mismatch")
        if value["algorithm_suite"] != ALGORITHM_SUITE.to_obj():
            raise ValueError("BHK22 algorithm suite mismatch")
        params_obj = value["parameters"]
        parameters = ATCaaSProtocolParameters.from_mapping(
            {"M": params_obj.get("M"), "atcaas_parameters": params_obj}
        )
        encrypted_source = None
        if encrypted:
            from beaver.baselines.bhk22_encrypted_source import BHK22EncryptedSourceVector

            encrypted_source = BHK22EncryptedSourceVector.from_obj(value["encrypted_source"])
        context = cls(
            trial_id=value["trial_id"],
            trial_epoch=value["trial_epoch"],
            source_epoch=value["source_epoch"],
            target_epoch=value["target_epoch"],
            sid=value["sid"],
            source_committee=value["source_committee"],
            successor_committee=value["successor_committee"],
            transfer_seq=value["transfer_seq"],
            parameters=parameters,
            crs=BHK22CRS.from_obj(value["crs"]),
            committees={
                role: BHK22CommitteeMap.from_obj(item)
                for role, item in value["committees"].items()
            },
            recipient_key_catalogs={
                role: BHK22RolePublicKeyCatalog.from_obj(item)
                for role, item in value["recipient_key_catalogs"].items()
            },
            source_public_values=() if encrypted else value["source_public_values"],
            dag_config_digest=value["dag_config_digest"],
            coin_config_digest=value["coin_config_digest"],
            role_trace_digest=value["role_trace_digest"],
            encrypted_source=encrypted_source,
        )
        supplied = digest_hex(
            value["root_context_digest"], name="root_context_digest"
        )
        if context.root_context_digest != supplied:
            raise ValueError("BHK22 root context digest mismatch")
        if context.to_bytes() != canonical_json(dict(value)):
            raise ValueError("BHK22 public context encoding is not canonical")
        return context

    @classmethod
    def from_bytes(
        cls, data: bytes | bytearray | memoryview | str
    ) -> "BHK22PublicContext":
        return cls.from_obj(decode_object(data, name="BHK22 public context"))


__all__ = [
    "BHK22CRS",
    "BHK22CommitteeMap",
    "BHK22PublicContext",
    "BHK22RolePublicKeyCatalog",
    "COMMITTEE_SELECTION_EXTERNAL",
    "COMMITTEE_SELECTION_VRF_EXACT",
    "CONTEXT_KIND",
    "CONTEXT_VERSION",
    "bhk22_committee_sortition_label",
    "bhk22_context_committee_sortition_label",
    "build_bhk22_crs",
    "canonical_json",
    "canonical_point",
    "canonical_scalar",
    "decode_object",
    "derive_nonzero_scalar",
    "derive_role_key_base",
    "digest_hex",
    "domain_digest",
    "exact_fields",
    "hash_to_g1",
    "nonempty_text",
    "strict_int",
]
