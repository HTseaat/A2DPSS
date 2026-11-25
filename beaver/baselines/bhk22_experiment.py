"""Deterministic Step-1 fixtures and node-local state for BHK+22.

The builder may hold every fixture secret transiently, but its public manifest
does not.  Runtime nodes load exactly one :class:`BHK22NodePrivateShard`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from beaver.atcaas_params import ATCaaSProtocolParameters
from beaver.baselines.bhk22_context import (
    BHK22CommitteeMap,
    BHK22PublicContext,
    BHK22RolePublicKeyCatalog,
    build_bhk22_crs,
    canonical_json,
    canonical_point,
    canonical_scalar,
    decode_object,
    derive_nonzero_scalar,
    derive_role_key_base,
    digest_hex,
    domain_digest,
    exact_fields,
    nonempty_text,
    strict_int,
)
from beaver.baselines.bhk22_contract import (
    COMPLETION_MODES,
    DEFAULT_COMPLETION_MODE,
    FIRST_EXPERIMENT_PARAMETERS,
    PATH_ONE_ACCUSATION,
    PATH_OPTIMISTIC,
    PATH_MODES,
    ROLE_DEALER,
    ROLE_RESPONDER,
    ROLE_SHAREHOLDER,
    ROLE_TYPES,
    ROLE_VERIFIER,
    validate_bhk22_parameters,
)
from beaver.g1_backend import G1Point
from beaver.pedersen_shamir import eval_poly
from beaver.baselines.bhk22_crypto import BHK22ElGamalCiphertext, decrypt_public_value


MANIFEST_KIND = "atcaas-bhk22-dag-experiment-manifest"
MANIFEST_VERSION = 2
LOCAL_SHARD_KIND = "atcaas-bhk22-node-private-shard"
LOCAL_SHARD_VERSION = 1
ROLE_PRIVATE_MATERIAL_KIND = "atcaas-bhk22-role-private-material"
ROLE_PRIVATE_MATERIAL_VERSION = 1

MANIFEST_DIGEST_DOMAIN = b"ATCaaS/BHK22/manifest/v2\x00"
SHARD_DIGEST_DOMAIN = b"ATCaaS/BHK22/node-shard/v1\x00"
COMMON_WORKLOAD_DIGEST_DOMAIN = b"ATCaaS/common-yoso-handover-workload/v1\x00"
BHK22_DAG_CONFIG_DIGEST_DOMAIN = b"ATCaaS/BHK22/dag-config/v1\x00"
BHK22_COIN_CONFIG_DIGEST_DOMAIN = b"ATCaaS/BHK22/coin-config/v1\x00"


@dataclass(frozen=True)
class BHK22FaultProfile:
    """Public, deterministic fault selected by the fixture path mode."""

    kind: str
    dealer_role_id: int
    verifier_role_id: int

    @property
    def expected_accusations(self) -> tuple[tuple[int, int], ...]:
        return ((self.dealer_role_id, self.verifier_role_id),)

    @property
    def expected_disqualified_dealer_ids(self) -> tuple[int, ...]:
        return (self.dealer_role_id,)

    def to_obj(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "dealer_role_id": self.dealer_role_id,
            "verifier_role_id": self.verifier_role_id,
            "expected_accusations": [
                list(pair) for pair in self.expected_accusations
            ],
            "expected_disqualified_dealer_ids": list(
                self.expected_disqualified_dealer_ids
            ),
        }


ONE_ACCUSATION_FAULT = BHK22FaultProfile(
    kind="dealer-verifier-row-aead-tag",
    dealer_role_id=2,
    verifier_role_id=0,
)


def bhk22_fault_profile(path_mode: str) -> BHK22FaultProfile | None:
    if path_mode == PATH_ONE_ACCUSATION:
        return ONE_ACCUSATION_FAULT
    if path_mode == PATH_OPTIMISTIC:
        return None
    raise ValueError("unsupported BHK22 fixture path mode")


def bhk22_dag_config_digest(
    parameters: ATCaaSProtocolParameters,
) -> str:
    """Bind the public DAG parameter tuple used by every process."""

    if not isinstance(parameters, ATCaaSProtocolParameters):
        raise TypeError("parameters must be ATCaaSProtocolParameters")
    return hashlib.sha256(
        BHK22_DAG_CONFIG_DIGEST_DOMAIN
        + canonical_json(parameters.to_config_obj())
    ).hexdigest()


def bhk22_coin_config_digest(extras: Mapping[str, Any]) -> str:
    """Bind the shared threshold-coin configuration, without private shares."""

    if not isinstance(extras, Mapping):
        raise TypeError("coin config extras must be a mapping")
    material = {
        field: extras.get(field)
        for field in (
            "coin_config_version",
            "coin_committee",
            "coin_threshold",
            "coin_public_key_digest",
            "coin_membership_digest",
        )
    }
    if any(value is None for value in material.values()):
        raise ValueError("BHK22 fixture requires complete public coin config")
    return hashlib.sha256(
        BHK22_COIN_CONFIG_DIGEST_DOMAIN + canonical_json(material)
    ).hexdigest()


def _role_mapping_obj(value: Mapping[str, Sequence[int]]) -> dict[str, list[int]]:
    return {role: list(value[role]) for role in ROLE_TYPES}


@dataclass(frozen=True)
class BHK22DAGExperimentManifest:
    fixture_seed: str
    path_mode: str
    context: BHK22PublicContext
    expected_active_roles: Mapping[str, Sequence[int]]
    common_workload_digest: str
    completion_mode: str = DEFAULT_COMPLETION_MODE
    fixture_only_deterministic_keys: bool = True
    not_for_production: bool = True

    def __post_init__(self) -> None:
        seed = nonempty_text(self.fixture_seed, name="fixture_seed")
        path_mode = str(self.path_mode)
        if path_mode not in PATH_MODES:
            raise ValueError("unsupported BHK22 fixture path mode")
        completion_mode = str(self.completion_mode)
        if completion_mode not in COMPLETION_MODES:
            raise ValueError("unsupported BHK22 completion mode")
        if not isinstance(self.context, BHK22PublicContext):
            raise TypeError("context must be BHK22PublicContext")
        if self.fixture_only_deterministic_keys is not True:
            raise ValueError("fixture-only deterministic keys flag is required")
        if self.not_for_production is not True:
            raise ValueError("not_for_production flag is required")
        active = {}
        for role_type in ROLE_TYPES:
            raw = self.expected_active_roles.get(role_type)
            if not isinstance(raw, (list, tuple)):
                raise TypeError("expected active roles must be sequences")
            roles = tuple(
                strict_int(role, name="expected active role") for role in raw
            )
            if roles != tuple(range(self.context.parameters.n_r)):
                raise ValueError(
                    "responsive fixture requires every role id to be active"
                )
            active[role_type] = roles
        if set(self.expected_active_roles) != set(ROLE_TYPES):
            raise ValueError("expected active roles require all role types")
        common_digest = digest_hex(
            self.common_workload_digest, name="common_workload_digest"
        )
        object.__setattr__(self, "fixture_seed", seed)
        object.__setattr__(self, "path_mode", path_mode)
        object.__setattr__(self, "completion_mode", completion_mode)
        object.__setattr__(self, "expected_active_roles", MappingProxyType(active))
        object.__setattr__(self, "common_workload_digest", common_digest)

    def digest_material(self) -> dict[str, Any]:
        return {
            "kind": MANIFEST_KIND,
            "version": MANIFEST_VERSION,
            "fixture_seed": self.fixture_seed,
            "path_mode": self.path_mode,
            "completion_mode": self.completion_mode,
            "fixture_only_deterministic_keys": True,
            "not_for_production": True,
            "context": self.context.to_obj(),
            "expected_active_roles": _role_mapping_obj(
                self.expected_active_roles
            ),
            "common_workload_digest": self.common_workload_digest,
        }

    @property
    def fault_profile(self) -> BHK22FaultProfile | None:
        return bhk22_fault_profile(self.path_mode)

    def digest(self) -> str:
        return domain_digest(MANIFEST_DIGEST_DOMAIN, self.digest_material())

    def to_obj(self) -> dict[str, Any]:
        value = self.digest_material()
        value["manifest_digest"] = self.digest()
        return value

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_obj())

    @classmethod
    def from_obj(
        cls, value: Mapping[str, Any]
    ) -> "BHK22DAGExperimentManifest":
        exact_fields(
            value,
            {
                "kind",
                "version",
                "fixture_seed",
                "path_mode",
                "completion_mode",
                "fixture_only_deterministic_keys",
                "not_for_production",
                "context",
                "expected_active_roles",
                "common_workload_digest",
                "manifest_digest",
            },
            name="BHK22 experiment manifest",
        )
        if value["kind"] != MANIFEST_KIND or value["version"] != MANIFEST_VERSION:
            raise ValueError("unsupported BHK22 experiment manifest")
        supplied = digest_hex(value["manifest_digest"], name="manifest_digest")
        manifest = cls(
            fixture_seed=value["fixture_seed"],
            path_mode=value["path_mode"],
            completion_mode=value["completion_mode"],
            context=BHK22PublicContext.from_obj(value["context"]),
            expected_active_roles=value["expected_active_roles"],
            common_workload_digest=value["common_workload_digest"],
            fixture_only_deterministic_keys=value[
                "fixture_only_deterministic_keys"
            ],
            not_for_production=value["not_for_production"],
        )
        if manifest.digest() != supplied:
            raise ValueError("BHK22 manifest digest mismatch")
        if manifest.to_bytes() != canonical_json(dict(value)):
            raise ValueError("BHK22 manifest encoding is not canonical")
        return manifest

    @classmethod
    def from_bytes(
        cls, data: bytes | bytearray | memoryview | str
    ) -> "BHK22DAGExperimentManifest":
        return cls.from_obj(decode_object(data, name="BHK22 manifest"))

    @classmethod
    def from_file(cls, path: str | Path) -> "BHK22DAGExperimentManifest":
        return cls.from_bytes(Path(path).read_bytes())


@dataclass(frozen=True)
class BHK22RolePrivateMaterial:
    role_type: str
    role_id: int
    evaluation_point: int
    recipient_secret_key: int
    source_share: int | None = None
    public_value_secret_key: int | None = None

    def __post_init__(self) -> None:
        role_type = str(self.role_type)
        if role_type not in ROLE_TYPES:
            raise ValueError("unsupported private role type")
        role_id = strict_int(self.role_id, name="role_id")
        evaluation_point = strict_int(
            self.evaluation_point, name="evaluation_point", minimum=1
        )
        if evaluation_point != role_id + 1:
            raise ValueError("private role evaluation point mismatch")
        recipient_secret = canonical_scalar(
            self.recipient_secret_key,
            name="recipient_secret_key",
            nonzero=True,
        )
        source_share = self.source_share
        if source_share is not None:
            source_share = canonical_scalar(source_share, name="source_share")
        public_value_secret = self.public_value_secret_key
        if public_value_secret is not None:
            public_value_secret = canonical_scalar(
                public_value_secret,
                name="public_value_secret_key",
                nonzero=True,
            )
        if role_type == ROLE_DEALER:
            if source_share is None or public_value_secret is not None:
                raise ValueError("dealer material requires only source_share")
        elif role_type == ROLE_SHAREHOLDER:
            if source_share is not None or public_value_secret is None:
                raise ValueError(
                    "shareholder material requires only public-value secret"
                )
        elif source_share is not None or public_value_secret is not None:
            raise ValueError("verifier/responder material has extra secrets")
        object.__setattr__(self, "role_type", role_type)
        object.__setattr__(self, "role_id", role_id)
        object.__setattr__(self, "evaluation_point", evaluation_point)
        object.__setattr__(self, "recipient_secret_key", recipient_secret)
        object.__setattr__(self, "source_share", source_share)
        object.__setattr__(
            self, "public_value_secret_key", public_value_secret
        )

    def to_obj(self) -> dict[str, Any]:
        return {
            "kind": ROLE_PRIVATE_MATERIAL_KIND,
            "version": ROLE_PRIVATE_MATERIAL_VERSION,
            "role_type": self.role_type,
            "role_id": self.role_id,
            "evaluation_point": self.evaluation_point,
            "recipient_secret_key": str(self.recipient_secret_key),
            "source_share": (
                str(self.source_share) if self.source_share is not None else None
            ),
            "public_value_secret_key": (
                str(self.public_value_secret_key)
                if self.public_value_secret_key is not None
                else None
            ),
        }

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22RolePrivateMaterial":
        exact_fields(
            value,
            {
                "kind",
                "version",
                "role_type",
                "role_id",
                "evaluation_point",
                "recipient_secret_key",
                "source_share",
                "public_value_secret_key",
            },
            name="BHK22 role private material",
        )
        if value["kind"] != ROLE_PRIVATE_MATERIAL_KIND or value["version"] != 1:
            raise ValueError("unsupported BHK22 role private material")
        return cls(
            role_type=value["role_type"],
            role_id=value["role_id"],
            evaluation_point=value["evaluation_point"],
            recipient_secret_key=canonical_scalar(
                value["recipient_secret_key"],
                name="recipient_secret_key",
                nonzero=True,
            ),
            source_share=(
                canonical_scalar(value["source_share"], name="source_share")
                if value["source_share"] is not None
                else None
            ),
            public_value_secret_key=(
                canonical_scalar(
                    value["public_value_secret_key"],
                    name="public_value_secret_key",
                    nonzero=True,
                )
                if value["public_value_secret_key"] is not None
                else None
            ),
        )

    def new_state(self):
        state_types = {
            ROLE_DEALER: BHK22DealerState,
            ROLE_VERIFIER: BHK22VerifierState,
            ROLE_RESPONDER: BHK22ResponderState,
            ROLE_SHAREHOLDER: BHK22ShareholderState,
        }
        return state_types[self.role_type](self)


@dataclass(frozen=True)
class BHK22NodePrivateShard:
    node_id: int
    manifest_digest: str
    role_materials: Sequence[BHK22RolePrivateMaterial]

    def __post_init__(self) -> None:
        node_id = strict_int(self.node_id, name="node_id")
        manifest_digest = digest_hex(
            self.manifest_digest, name="manifest_digest"
        )
        materials = tuple(
            material
            if isinstance(material, BHK22RolePrivateMaterial)
            else BHK22RolePrivateMaterial.from_obj(material)
            for material in self.role_materials
        )
        identities = [(item.role_type, item.role_id) for item in materials]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate local BHK22 role material")
        role_order = {role: index for index, role in enumerate(ROLE_TYPES)}
        materials = tuple(
            sorted(
                materials,
                key=lambda item: (role_order[item.role_type], item.role_id),
            )
        )
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "manifest_digest", manifest_digest)
        object.__setattr__(self, "role_materials", materials)

    def digest_material(self) -> dict[str, Any]:
        return {
            "kind": LOCAL_SHARD_KIND,
            "version": LOCAL_SHARD_VERSION,
            "node_id": self.node_id,
            "manifest_digest": self.manifest_digest,
            "role_materials": [item.to_obj() for item in self.role_materials],
        }

    def digest(self) -> str:
        return domain_digest(SHARD_DIGEST_DOMAIN, self.digest_material())

    def to_obj(self) -> dict[str, Any]:
        value = self.digest_material()
        value["shard_digest"] = self.digest()
        return value

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_obj())

    @classmethod
    def from_obj(
        cls,
        value: Mapping[str, Any],
        manifest: BHK22DAGExperimentManifest,
    ) -> "BHK22NodePrivateShard":
        exact_fields(
            value,
            {
                "kind",
                "version",
                "node_id",
                "manifest_digest",
                "role_materials",
                "shard_digest",
            },
            name="BHK22 node-private shard",
        )
        if value["kind"] != LOCAL_SHARD_KIND or value["version"] != 1:
            raise ValueError("unsupported BHK22 node-private shard")
        supplied = digest_hex(value["shard_digest"], name="shard_digest")
        shard = cls(
            node_id=value["node_id"],
            manifest_digest=value["manifest_digest"],
            role_materials=tuple(
                BHK22RolePrivateMaterial.from_obj(item)
                for item in value["role_materials"]
            ),
        )
        if shard.digest() != supplied:
            raise ValueError("BHK22 local shard digest mismatch")
        shard.validate(manifest)
        if shard.to_bytes() != canonical_json(dict(value)):
            raise ValueError("BHK22 local shard encoding is not canonical")
        return shard

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview | str,
        manifest: BHK22DAGExperimentManifest,
    ) -> "BHK22NodePrivateShard":
        return cls.from_obj(
            decode_object(data, name="BHK22 node-private shard"), manifest
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        manifest: BHK22DAGExperimentManifest,
    ) -> "BHK22NodePrivateShard":
        return cls.from_bytes(Path(path).read_bytes(), manifest)

    def validate(self, manifest: BHK22DAGExperimentManifest) -> None:
        if self.manifest_digest != manifest.digest():
            raise ValueError("local shard manifest digest mismatch")
        context = manifest.context
        if self.node_id >= context.parameters.M:
            raise ValueError("local shard node id is outside M")
        expected = {
            (role_type, role_id)
            for role_type in ROLE_TYPES
            for role_id, node in context.committees[role_type].role_to_node.items()
            if node == self.node_id
        }
        actual = {(item.role_type, item.role_id) for item in self.role_materials}
        if actual != expected:
            raise ValueError("local shard roles do not match public committee maps")
        for material in self.role_materials:
            role_type = material.role_type
            role_id = material.role_id
            committee = context.committees[role_type]
            if committee.evaluation_points[role_id] != material.evaluation_point:
                raise ValueError("local role evaluation point mismatch")
            catalog = context.recipient_key_catalogs[role_type]
            derived_public_key = (
                G1Point.from_hex(catalog.key_base_hex)
                * material.recipient_secret_key
            ).to_hex()
            if derived_public_key != catalog.public_keys[role_id]:
                raise ValueError("local recipient secret key mismatch")
            if role_type == ROLE_DEALER:
                source_value = (
                    G1Point.from_hex(context.crs.generators[role_id + 1])
                    * material.source_share
                ).to_hex()
                if context.encrypted_source is None:
                    if source_value != context.source_public_values[role_id]:
                        raise ValueError("dealer source share/public value mismatch")
                else:
                    # Application fixtures keep the source decryption key in
                    # the same node's shareholder role; no plaintext vector is
                    # reconstructed or published to validate the local share.
                    keys = [
                        item.public_value_secret_key for item in self.role_materials
                        if item.role_type == ROLE_SHAREHOLDER and item.role_id == role_id
                    ]
                    if len(keys) != 1:
                        raise ValueError("encrypted source dealer requires its local decryption key")
                    coordinate = context.encrypted_source.coordinates[role_id]
                    if decrypt_public_value(coordinate.ciphertext, secret_key=keys[0]) != source_value:
                        raise ValueError("dealer source share/encrypted value mismatch")
            if role_type == ROLE_SHAREHOLDER:
                public_value_key = (
                    G1Point.from_hex(context.crs.generators[role_id + 1])
                    * material.public_value_secret_key
                ).to_hex()
                if public_value_key != context.crs.public_value_keys[role_id + 1]:
                    raise ValueError("shareholder public-value secret mismatch")

    def new_role_states(
        self, manifest: BHK22DAGExperimentManifest
    ) -> Mapping[str, tuple["BHK22OneShotRoleState", ...]]:
        self.validate(manifest)
        by_type = {role_type: [] for role_type in ROLE_TYPES}
        for material in self.role_materials:
            by_type[material.role_type].append(material.new_state())
        return MappingProxyType(
            {role: tuple(by_type[role]) for role in ROLE_TYPES}
        )


def best_effort_wipe(value: bytearray | memoryview) -> None:
    """Overwrite a caller-owned mutable byte buffer in place."""

    if isinstance(value, memoryview):
        if value.readonly:
            raise TypeError("cannot wipe a read-only memoryview")
        value[:] = b"\x00" * len(value)
        return
    if not isinstance(value, bytearray):
        raise TypeError("best_effort_wipe requires bytearray or memoryview")
    value[:] = b"\x00" * len(value)


class _SecretScalarBox:
    def __init__(self, value: int, *, name: str) -> None:
        scalar = canonical_scalar(value, name=name)
        self._buffer = bytearray(scalar.to_bytes(32, "big"))
        self._taken = False

    def take(self) -> int:
        if self._taken:
            raise RuntimeError("secret scalar was already consumed")
        value = int.from_bytes(self._buffer, "big")
        best_effort_wipe(self._buffer)
        self._taken = True
        return value

    def peek_for_protocol(self) -> int:
        """Read inside the node-local protocol boundary without consuming.

        Callers must complete the public output first and then invoke the
        owning role state's ``consume_once()``.  This is intentionally not
        exposed by the public shard/material serialization boundary.
        """

        if self._taken:
            raise RuntimeError("secret scalar was already consumed")
        return int.from_bytes(self._buffer, "big")

    def wipe(self) -> None:
        best_effort_wipe(self._buffer)
        self._taken = True

    @property
    def consumed(self) -> bool:
        return self._taken


@dataclass(frozen=True)
class BHK22ConsumedRoleSecrets:
    role_type: str
    role_id: int
    evaluation_point: int
    recipient_secret_key: int
    source_share: int | None
    public_value_secret_key: int | None


class BHK22OneShotRoleState:
    expected_role_type = ""

    def __init__(self, material: BHK22RolePrivateMaterial) -> None:
        if material.role_type != self.expected_role_type:
            raise ValueError("private material has the wrong role type")
        self.role_type = material.role_type
        self.role_id = material.role_id
        self.evaluation_point = material.evaluation_point
        self._recipient_secret = _SecretScalarBox(
            material.recipient_secret_key, name="recipient_secret_key"
        )
        self._source_share = (
            _SecretScalarBox(material.source_share, name="source_share")
            if material.source_share is not None
            else None
        )
        self._public_value_secret = (
            _SecretScalarBox(
                material.public_value_secret_key,
                name="public_value_secret_key",
            )
            if material.public_value_secret_key is not None
            else None
        )
        self._consumed = False

    @property
    def consumed(self) -> bool:
        return self._consumed

    def _recipient_secret_for_protocol(self) -> int:
        if self._consumed:
            raise RuntimeError(f"{self.role_type} role state was already consumed")
        return self._recipient_secret.peek_for_protocol()

    def consume_once(self) -> BHK22ConsumedRoleSecrets:
        if self._consumed:
            raise RuntimeError(f"{self.role_type} role state was already consumed")
        consumed = BHK22ConsumedRoleSecrets(
            role_type=self.role_type,
            role_id=self.role_id,
            evaluation_point=self.evaluation_point,
            recipient_secret_key=self._recipient_secret.take(),
            source_share=(
                self._source_share.take() if self._source_share else None
            ),
            public_value_secret_key=(
                self._public_value_secret.take()
                if self._public_value_secret
                else None
            ),
        )
        self._consumed = True
        return consumed

    def wipe(self) -> None:
        self._recipient_secret.wipe()
        if self._source_share:
            self._source_share.wipe()
        if self._public_value_secret:
            self._public_value_secret.wipe()
        self._consumed = True


class BHK22DealerState(BHK22OneShotRoleState):
    expected_role_type = ROLE_DEALER

    def _source_share_for_protocol(self) -> int:
        if self._consumed or self._source_share is None:
            raise RuntimeError("dealer role state was already consumed")
        return self._source_share.peek_for_protocol()


class BHK22VerifierState(BHK22OneShotRoleState):
    expected_role_type = ROLE_VERIFIER


class BHK22ResponderState(BHK22OneShotRoleState):
    expected_role_type = ROLE_RESPONDER


class BHK22ShareholderState(BHK22OneShotRoleState):
    expected_role_type = ROLE_SHAREHOLDER

    def _public_value_secret_for_protocol(self) -> int:
        if self._consumed or self._public_value_secret is None:
            raise RuntimeError("shareholder role state was already consumed")
        return self._public_value_secret.peek_for_protocol()


class BHK22SuccessorResult:
    """Private successor share with one-time ownership transfer.

    Deliberately has no ``to_obj``/``to_bytes`` public serialization method.
    """

    def __init__(
        self,
        *,
        shareholder_role_id: int,
        evaluation_point: int,
        share: int,
        public_value: str,
        root_context_digest: str,
    ) -> None:
        self.shareholder_role_id = strict_int(
            shareholder_role_id, name="shareholder_role_id"
        )
        self.evaluation_point = strict_int(
            evaluation_point, name="evaluation_point", minimum=1
        )
        if self.evaluation_point != self.shareholder_role_id + 1:
            raise ValueError("successor result evaluation point mismatch")
        self.public_value = canonical_point(
            public_value, name="successor public value"
        )
        self.root_context_digest = digest_hex(
            root_context_digest, name="root_context_digest"
        )
        self._share = _SecretScalarBox(share, name="successor share")

    @property
    def consumed(self) -> bool:
        return self._share.consumed

    def take_share_once(self) -> int:
        return self._share.take()

    def wipe(self) -> None:
        self._share.wipe()


@dataclass(frozen=True)
class BHK22CarryForwardSecrets:
    """Private state transferred into the next source/dealer role."""

    shareholder_role_id: int
    evaluation_point: int
    share: int
    public_value_secret_key: int
    aggregate_ciphertext: BHK22ElGamalCiphertext
    terminal_phase_qid: str
    root_context_digest: str


class BHK22CarryForwardSuccessorState:
    """Silent successor state for strict chained YOSO DPSS.

    This object intentionally has no public serializer.  The successor has not
    spoken: it retains both the refreshed scalar share and the ElGamal key
    needed to open the encrypted public commitment in its next dealer speech.
    """

    def __init__(
        self,
        *,
        shareholder_role_id: int,
        evaluation_point: int,
        share: int,
        public_value_secret_key: int,
        aggregate_ciphertext: BHK22ElGamalCiphertext,
        terminal_phase_qid: str,
        root_context_digest: str,
    ) -> None:
        self.shareholder_role_id = strict_int(
            shareholder_role_id, name="shareholder_role_id"
        )
        self.evaluation_point = strict_int(
            evaluation_point, name="evaluation_point", minimum=1
        )
        if self.evaluation_point != self.shareholder_role_id + 1:
            raise ValueError("carry-forward evaluation point mismatch")
        if not isinstance(aggregate_ciphertext, BHK22ElGamalCiphertext):
            raise TypeError("carry-forward ciphertext has the wrong type")
        self.aggregate_ciphertext = aggregate_ciphertext
        self.terminal_phase_qid = nonempty_text(
            terminal_phase_qid, name="terminal_phase_qid"
        )
        self.root_context_digest = digest_hex(
            root_context_digest, name="root_context_digest"
        )
        self._share = _SecretScalarBox(share, name="carry-forward share")
        self._public_value_secret = _SecretScalarBox(
            public_value_secret_key,
            name="carry-forward public-value secret key",
        )
        self._consumed = False

    @property
    def consumed(self) -> bool:
        return self._consumed

    def _share_for_protocol(self) -> int:
        if self._consumed:
            raise RuntimeError("carry-forward state was already consumed")
        return self._share.peek_for_protocol()

    def _public_value_secret_for_protocol(self) -> int:
        if self._consumed:
            raise RuntimeError("carry-forward state was already consumed")
        return self._public_value_secret.peek_for_protocol()

    def take_for_next_dealer_once(self) -> BHK22CarryForwardSecrets:
        if self._consumed:
            raise RuntimeError("carry-forward state was already consumed")
        result = BHK22CarryForwardSecrets(
            shareholder_role_id=self.shareholder_role_id,
            evaluation_point=self.evaluation_point,
            share=self._share.take(),
            public_value_secret_key=self._public_value_secret.take(),
            aggregate_ciphertext=self.aggregate_ciphertext,
            terminal_phase_qid=self.terminal_phase_qid,
            root_context_digest=self.root_context_digest,
        )
        self._consumed = True
        return result

    def wipe(self) -> None:
        self._share.wipe()
        self._public_value_secret.wipe()
        self._consumed = True


@dataclass(frozen=True)
class BHK22FixtureBundle:
    manifest: BHK22DAGExperimentManifest
    shards: Mapping[int, BHK22NodePrivateShard]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, BHK22DAGExperimentManifest):
            raise TypeError("manifest has wrong type")
        shards = dict(self.shards)
        if set(shards) != set(range(self.manifest.context.parameters.M)):
            raise ValueError("fixture must contain one shard for every network node")
        for node_id, shard in shards.items():
            if not isinstance(shard, BHK22NodePrivateShard) or shard.node_id != node_id:
                raise ValueError("fixture shard/node mismatch")
            shard.validate(self.manifest)
        object.__setattr__(self, "shards", MappingProxyType(shards))


def _default_digest(seed: str, label: str) -> str:
    return hashlib.sha256(
        b"ATCaaS/BHK22/fixture-config-digest/v1\x00"
        + seed.encode("utf-8")
        + b"\x00"
        + label.encode("utf-8")
    ).hexdigest()


def _normalize_role_nodes(
    parameters: ATCaaSProtocolParameters,
    role_to_node: Mapping[str, Mapping[int, int]] | None,
) -> dict[str, dict[int, int]]:
    if role_to_node is None:
        return {
            role: {role_id: role_id for role_id in range(parameters.n_r)}
            for role in ROLE_TYPES
        }
    if set(role_to_node) != set(ROLE_TYPES):
        raise ValueError("role_to_node must define all four role types")
    return {
        role: {int(role_id): int(node) for role_id, node in role_to_node[role].items()}
        for role in ROLE_TYPES
    }


def build_bhk22_fixture(
    *,
    seed: str,
    trial_id: str = "bhk22-step1",
    trial_epoch: int = 1,
    source_epoch: int = 1,
    sid: str = "bhk22-service-key",
    transfer_seq: int = 0,
    source_committee: str = "source-committee",
    successor_committee: str = "successor-committee",
    parameters: ATCaaSProtocolParameters = FIRST_EXPERIMENT_PARAMETERS,
    path_mode: str = "optimistic",
    completion_mode: str = DEFAULT_COMPLETION_MODE,
    role_to_node: Mapping[str, Mapping[int, int]] | None = None,
    committee_selection: Mapping[str, Mapping[str, Any]] | None = None,
    source_secret: int | None = None,
    source_polynomial_coefficients: Sequence[int] | None = None,
    dag_config_digest: str | None = None,
    coin_config_digest: str | None = None,
    role_trace_digest: str | None = None,
    common_workload_digest: str | None = None,
) -> BHK22FixtureBundle:
    seed = nonempty_text(seed, name="fixture seed")
    parameters = validate_bhk22_parameters(parameters)
    if path_mode not in PATH_MODES:
        raise ValueError("unsupported BHK22 fixture path mode")
    if completion_mode not in COMPLETION_MODES:
        raise ValueError("unsupported BHK22 completion mode")
    role_nodes = _normalize_role_nodes(parameters, role_to_node)
    selection = {
        role: dict((committee_selection or {}).get(role, {}))
        for role in ROLE_TYPES
    }
    evaluation_points = {role: role + 1 for role in range(parameters.n_r)}
    committees = {
        role: BHK22CommitteeMap(
            role_type=role,
            role_to_node=role_nodes[role],
            evaluation_points=evaluation_points,
            n_r=parameters.n_r,
            M=parameters.M,
            selection_mode=selection[role].get(
                "selection_mode", "externally-established"
            ),
            selection_attempt=selection[role].get("selection_attempt", 0),
        )
        for role in ROLE_TYPES
    }

    if source_secret is None:
        source_secret = derive_nonzero_scalar(seed, "source-secret")
    source_secret = canonical_scalar(source_secret, name="source secret")
    if source_polynomial_coefficients is None:
        coefficients = (source_secret,) + tuple(
            derive_nonzero_scalar(seed, f"source-coefficient:{degree}")
            for degree in range(1, parameters.t_r + 1)
        )
    else:
        coefficients = tuple(
            canonical_scalar(value, name="source polynomial coefficient")
            for value in source_polynomial_coefficients
        )
        if len(coefficients) != parameters.t_r + 1:
            raise ValueError("source polynomial must have degree t_r")
        if coefficients[0] != source_secret:
            raise ValueError("source polynomial constant must equal source secret")
    if len(coefficients) != parameters.t_r + 1:
        raise ValueError("source polynomial must contain t_r+1 coefficients")

    recipient_secrets = {
        role: {
            role_id: derive_nonzero_scalar(
                seed, f"recipient-secret:{role}:{role_id}"
            )
            for role_id in range(parameters.n_r)
        }
        for role in ROLE_TYPES
    }
    key_seed = f"bhk22-key-bases-v1:n={parameters.n_r}"
    catalogs = {}
    for role in ROLE_TYPES:
        key_base = derive_role_key_base(role, key_seed)
        catalogs[role] = BHK22RolePublicKeyCatalog(
            role_type=role,
            key_base_hex=key_base,
            public_keys={
                role_id: (
                    G1Point.from_hex(key_base)
                    * recipient_secrets[role][role_id]
                ).to_hex()
                for role_id in range(parameters.n_r)
            },
            n_r=parameters.n_r,
        )

    shareholder_public_value_secrets = tuple(
        derive_nonzero_scalar(seed, f"public-value-secret:{role_id}")
        for role_id in range(parameters.n_r)
    )
    crs = build_bhk22_crs(
        n_r=parameters.n_r,
        derivation_seed=f"bhk22-crs-v1:n={parameters.n_r}",
        shareholder_secret_keys=shareholder_public_value_secrets,
    )
    source_shares = tuple(
        eval_poly(coefficients, point)
        for point in range(1, parameters.n_r + 1)
    )
    source_public_values = tuple(
        (
            G1Point.from_hex(crs.generators[role_id + 1])
            * source_shares[role_id]
        ).to_hex()
        for role_id in range(parameters.n_r)
    )
    digest_values = {
        "dag_config_digest": dag_config_digest
        or _default_digest(seed, "dag-config"),
        "coin_config_digest": coin_config_digest
        or _default_digest(seed, "coin-config"),
        "role_trace_digest": role_trace_digest
        or _default_digest(seed, "role-trace"),
    }
    context = BHK22PublicContext(
        trial_id=trial_id,
        trial_epoch=trial_epoch,
        source_epoch=source_epoch,
        target_epoch=source_epoch + 1,
        sid=sid,
        source_committee=source_committee,
        successor_committee=successor_committee,
        transfer_seq=transfer_seq,
        parameters=parameters,
        crs=crs,
        committees=committees,
        recipient_key_catalogs=catalogs,
        source_public_values=source_public_values,
        **digest_values,
    )
    workload_material = {
        "schema": "atcaas-common-yoso-handover-workload/v1",
        "fixture_seed": seed,
        "trial_id": trial_id,
        "trial_epoch": trial_epoch,
        "source_epoch": source_epoch,
        "target_epoch": source_epoch + 1,
        "sid": sid,
        "transfer_seq": transfer_seq,
        "parameters": parameters.to_config_obj(),
        "role_to_node": {
            role: {str(k): v for k, v in role_nodes[role].items()}
            for role in ROLE_TYPES
        },
        "source_secret": str(source_secret),
        "source_polynomial_coefficients": [str(value) for value in coefficients],
    }
    derived_workload_digest = domain_digest(
        COMMON_WORKLOAD_DIGEST_DOMAIN, workload_material
    )
    if common_workload_digest is None:
        common_workload_digest = derived_workload_digest
    elif (
        not isinstance(common_workload_digest, str)
        or len(common_workload_digest) != 64
    ):
        raise ValueError("common_workload_digest must be sha256 hex")
    else:
        bytes.fromhex(common_workload_digest)
    manifest = BHK22DAGExperimentManifest(
        fixture_seed=seed,
        path_mode=path_mode,
        completion_mode=completion_mode,
        context=context,
        expected_active_roles={
            role: tuple(range(parameters.n_r)) for role in ROLE_TYPES
        },
        common_workload_digest=common_workload_digest,
    )

    materials_by_node = {node_id: [] for node_id in range(parameters.M)}
    for role_type in ROLE_TYPES:
        for role_id in range(parameters.n_r):
            material = BHK22RolePrivateMaterial(
                role_type=role_type,
                role_id=role_id,
                evaluation_point=role_id + 1,
                recipient_secret_key=recipient_secrets[role_type][role_id],
                source_share=(
                    source_shares[role_id]
                    if role_type == ROLE_DEALER
                    else None
                ),
                public_value_secret_key=(
                    shareholder_public_value_secrets[role_id]
                    if role_type == ROLE_SHAREHOLDER
                    else None
                ),
            )
            materials_by_node[role_nodes[role_type][role_id]].append(material)
    shards = {
        node_id: BHK22NodePrivateShard(
            node_id=node_id,
            manifest_digest=manifest.digest(),
            role_materials=tuple(materials_by_node[node_id]),
        )
        for node_id in range(parameters.M)
    }
    return BHK22FixtureBundle(manifest=manifest, shards=shards)


def write_bhk22_fixture(
    bundle: BHK22FixtureBundle,
    output_directory: str | Path,
) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_bytes(bundle.manifest.to_bytes() + b"\n")
    for node_id, shard in bundle.shards.items():
        (output / f"node.{node_id}.private.json").write_bytes(
            shard.to_bytes() + b"\n"
        )


__all__ = [
    "BHK22CarryForwardSecrets",
    "BHK22CarryForwardSuccessorState",
    "BHK22FaultProfile",
    "ONE_ACCUSATION_FAULT",
    "bhk22_fault_profile",
    "bhk22_coin_config_digest",
    "bhk22_dag_config_digest",
    "BHK22ConsumedRoleSecrets",
    "BHK22DAGExperimentManifest",
    "BHK22DealerState",
    "BHK22FixtureBundle",
    "BHK22NodePrivateShard",
    "BHK22OneShotRoleState",
    "BHK22ResponderState",
    "BHK22RolePrivateMaterial",
    "BHK22ShareholderState",
    "BHK22SuccessorResult",
    "BHK22VerifierState",
    "best_effort_wipe",
    "build_bhk22_fixture",
    "write_bhk22_fixture",
]
