"""Operation-coupled encrypted carry-forward extension for BHK+22.

The pure BHK+22 implementation deliberately remains unchanged.  This module
adds the application variant required by the threshold-service experiment:

* the source public commitment vector contains ElGamal ciphertexts only;
* a dealer publishes ``vk_i`` plus two independent proofs;
* the application Figure-7 statement uses ``vk_i`` as its source anchor; and
* the terminal qualified set selects both resharing contributions and
  operation shares, while successor commitments remain encrypted.

DAG-TOB is only an ordering service.  All proof verification in this module is
performed by the BHK verifier/application processing path.
"""

from __future__ import annotations

import hmac
import json
from dataclasses import dataclass, field, replace
from time import perf_counter, process_time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from beaver.atcaas_params import ATCaaSProtocolParameters
from beaver.baselines.bhk22_context import (
    BHK22CRS,
    BHK22CommitteeMap,
    BHK22PublicContext,
    BHK22RolePublicKeyCatalog,
    canonical_json,
    canonical_point,
    canonical_scalar,
    derive_nonzero_scalar,
    digest_hex,
    domain_digest,
    exact_fields,
    nonempty_text,
    strict_int,
)
from beaver.baselines.bhk22_contract import (
    PATH_ONE_ACCUSATION,
    PATH_OPTIMISTIC,
    ROLE_DEALER,
    ROLE_TYPES,
)
from beaver.baselines.bhk22_crypto import (
    BHK22ElGamalCiphertext,
    BHK22ProofContext,
    combine_public_value_ciphertexts,
    decrypt_public_value,
    encrypt_public_value,
    verify_figure7,
    verify_figure8_randomness,
)
from beaver.baselines.bhk22_encrypted_source import (
    BHK22EncryptedSourceCoordinate,
    BHK22EncryptedSourceVector,
    ENCRYPTED_SOURCE_COORDINATE_SCHEMA,
    ENCRYPTED_SOURCE_VECTOR_SCHEMA,
)
from beaver.baselines.bhk22_dealer import (
    BHK22DealerRandomSource,
    BHK22DealerRole,
)
from beaver.baselines.bhk22_experiment import (
    BHK22CarryForwardSuccessorState,
    BHK22FixtureBundle,
    BHK22RolePrivateMaterial,
)
from beaver.baselines.bhk22_messages import BHK22DealerBatch
from beaver.dag_tob import Qid
from beaver.g1_backend import (
    FIELD_ORDER,
    G1Point,
    field_add,
    field_mul,
    field_random,
    g1_linear_combination,
)
from beaver.operation_vk import (
    VKOperationShareProof,
    VKOperationShareStatement,
    combine_vk_operation_shares,
    evaluate_and_prove_vk_operation_share,
    verify_vk_operation_share,
)
from beaver.pedersen_shamir import eval_poly, lagrange_coeff_at_zero, pedersen_deg_check
from beaver.sigma import hash_to_scalar
from beaver.threshold_operation_contract import (
    BHK22_ENCRYPTED_CARRY_FORWARD_BLOCK_COUNTS,
    BHK22_OPERATION_CONTEXT_SCHEMA,
    BHK22_OPERATION_DEALER_BATCH_SCHEMA,
    BHK22_OPERATION_QID_KINDS,
    OPERATION_VK_BASE_HEX,
    ThresholdOperationContext,
    accepts_operation_candidate,
    operation_output_digest,
)


LINK_STATEMENT_SCHEMA = "bhk22-encrypted-public-value-vk-statement/v1"
LINK_PROOF_KIND = "bhk22-encrypted-public-value-to-vk/v1"
LINK_PROOF_VERSION = 1
OPERATION_RESULT_SCHEMA = "bhk22-operation-public-result/v1"

_APP_CONTEXT_DOMAIN = b"ATCaaS/BHK22/operation-context/v1\x00"
_LINK_STATEMENT_DOMAIN = b"ATCaaS/BHK22/operation-link/statement/v1\x00"
_LINK_FS_DOMAIN = "atcaas-bhk22-encrypted-public-value-to-vk-v1"
_BATCH_DOMAIN = b"ATCaaS/BHK22/operation-dealer-batch/v1\x00"
_RESULT_DOMAIN = b"ATCaaS/BHK22/operation-result/v1\x00"


def _decode_object(value: bytes | bytearray | memoryview | str, *, name: str) -> dict[str, Any]:
    try:
        raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        decoded = json.loads(raw.decode("utf-8"))
    except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed {name} JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must encode an object")
    return decoded


def _point(value: Any, *, name: str) -> str:
    point = canonical_point(value, name=name)
    if G1Point.from_hex(point).is_identity():
        raise ValueError(f"{name} must not be the identity")
    return point


def _qid(value: str, *, phase: str) -> str:
    text = nonempty_text(value, name=f"{phase} qid")
    decoded = Qid.decode(text)
    if decoded.encode() != text or decoded.kind != BHK22_OPERATION_QID_KINDS[phase]:
        raise ValueError(f"qid must be a canonical BHK22 operation {phase} qid")
    return text


def build_bhk22_operation_qid(
    context: "BHK22OperationPublicContext", *, phase: str
) -> str:
    if phase not in BHK22_OPERATION_QID_KINDS:
        raise ValueError("unsupported BHK22 operation phase")
    return Qid.build(
        BHK22_OPERATION_QID_KINDS[phase],
        arm_context_digest=context.context_digest,
        operation_context_digest=context.operation_context.context_digest,
        encrypted_source_digest=context.encrypted_source.digest,
        trial_id=context.trial_id,
        source_epoch=context.source_epoch,
        target_epoch=context.target_epoch,
        transfer_seq=context.transfer_seq,
    ).encode()


@dataclass(frozen=True)
class BHK22OperationPublicContext:
    trial_id: str
    source_epoch: int
    target_epoch: int
    transfer_seq: int
    base_context_digest: str
    parameters: Any
    crs: BHK22CRS
    committees: Mapping[str, BHK22CommitteeMap]
    recipient_key_catalogs: Mapping[str, BHK22RolePublicKeyCatalog]
    operation_context: ThresholdOperationContext
    encrypted_source: BHK22EncryptedSourceVector
    context_digest: str = field(init=False)

    def __post_init__(self) -> None:
        trial = nonempty_text(self.trial_id, name="trial_id")
        source_epoch = strict_int(self.source_epoch, name="source_epoch", minimum=1)
        target_epoch = strict_int(self.target_epoch, name="target_epoch", minimum=2)
        if target_epoch != source_epoch + 1:
            raise ValueError("target_epoch must equal source_epoch + 1")
        if not isinstance(self.parameters, ATCaaSProtocolParameters):
            raise TypeError("parameters has the wrong type")
        if not isinstance(self.crs, BHK22CRS) or not isinstance(self.operation_context, ThresholdOperationContext):
            raise TypeError("invalid BHK22 operation public context")
        if not isinstance(self.encrypted_source, BHK22EncryptedSourceVector):
            raise TypeError("encrypted_source has the wrong type")
        if len(self.encrypted_source.coordinates) != self.parameters.n_r:
            raise ValueError("encrypted source length does not match n_r")
        committees = dict(self.committees)
        catalogs = dict(self.recipient_key_catalogs)
        if set(committees) != set(ROLE_TYPES) or set(catalogs) != set(ROLE_TYPES):
            raise ValueError("application context requires every BHK role map/catalog")
        object.__setattr__(self, "trial_id", trial)
        object.__setattr__(self, "source_epoch", source_epoch)
        object.__setattr__(self, "target_epoch", target_epoch)
        object.__setattr__(
            self,
            "transfer_seq",
            strict_int(self.transfer_seq, name="transfer_seq"),
        )
        object.__setattr__(self, "base_context_digest", digest_hex(self.base_context_digest, name="base_context_digest"))
        object.__setattr__(self, "committees", MappingProxyType(committees))
        object.__setattr__(self, "recipient_key_catalogs", MappingProxyType(catalogs))
        object.__setattr__(self, "context_digest", domain_digest(_APP_CONTEXT_DOMAIN, self.digest_material()))

    def digest_material(self) -> dict[str, Any]:
        return {
            "schema": BHK22_OPERATION_CONTEXT_SCHEMA,
            "trial_id": self.trial_id,
            "source_epoch": self.source_epoch,
            "target_epoch": self.target_epoch,
            "transfer_seq": self.transfer_seq,
            "base_context_digest": self.base_context_digest,
            "parameters": self.parameters.to_config_obj(),
            "crs": self.crs.to_obj(),
            "committees": {role: self.committees[role].to_obj() for role in ROLE_TYPES},
            "recipient_key_catalogs": {role: self.recipient_key_catalogs[role].to_obj() for role in ROLE_TYPES},
            "operation_context": self.operation_context.to_obj(),
            "encrypted_source": self.encrypted_source.to_obj(),
        }

    def to_obj(self) -> dict[str, Any]:
        value = self.digest_material()
        value["context_digest"] = self.context_digest
        return value

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22OperationPublicContext":
        expected = {
            "schema", "trial_id", "source_epoch", "target_epoch",
            "transfer_seq", "base_context_digest", "parameters", "crs",
            "committees", "recipient_key_catalogs", "operation_context",
            "encrypted_source", "context_digest",
        }
        exact_fields(value, expected, name="BHK22 operation public context")
        if value["schema"] != BHK22_OPERATION_CONTEXT_SCHEMA:
            raise ValueError("unsupported BHK22 operation public context")
        parameters = ATCaaSProtocolParameters.from_mapping(
            {"M": value["parameters"].get("M"), "atcaas_parameters": value["parameters"]}
        )
        result = cls(
            trial_id=value["trial_id"],
            source_epoch=value["source_epoch"],
            target_epoch=value["target_epoch"],
            transfer_seq=value["transfer_seq"],
            base_context_digest=value["base_context_digest"],
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
            operation_context=ThresholdOperationContext.from_obj(value["operation_context"]),
            encrypted_source=BHK22EncryptedSourceVector.from_obj(value["encrypted_source"]),
        )
        if result.context_digest != digest_hex(value["context_digest"], name="context_digest"):
            raise ValueError("BHK22 operation public context digest mismatch")
        return result

    @classmethod
    def from_bytes(
        cls, value: bytes | bytearray | memoryview | str
    ) -> "BHK22OperationPublicContext":
        return cls.from_obj(_decode_object(value, name="BHK22 operation public context"))


@dataclass(frozen=True)
class BHK22EncryptedPublicValueVKStatement:
    qid: str
    arm_context_digest: str
    operation_context_digest: str
    encrypted_source_digest: str
    terminal_state_identity: str
    coordinate: BHK22EncryptedSourceCoordinate
    dealer_role_id: int
    operation_vk_base_hex: str
    operation_vk_hex: str
    statement_digest: str = field(init=False)

    def __post_init__(self) -> None:
        qid = _qid(self.qid, phase="deal")
        if not isinstance(self.coordinate, BHK22EncryptedSourceCoordinate):
            raise TypeError("link statement coordinate has wrong type")
        dealer = strict_int(self.dealer_role_id, name="dealer_role_id")
        if dealer != self.coordinate.coordinate_index:
            raise ValueError("dealer/coordinate mismatch")
        object.__setattr__(self, "qid", qid)
        object.__setattr__(self, "arm_context_digest", digest_hex(self.arm_context_digest, name="arm_context_digest"))
        object.__setattr__(self, "operation_context_digest", digest_hex(self.operation_context_digest, name="operation_context_digest"))
        object.__setattr__(self, "encrypted_source_digest", digest_hex(self.encrypted_source_digest, name="encrypted_source_digest"))
        object.__setattr__(self, "terminal_state_identity", nonempty_text(self.terminal_state_identity, name="terminal_state_identity"))
        object.__setattr__(self, "operation_vk_base_hex", _point(self.operation_vk_base_hex, name="operation_vk_base_hex"))
        object.__setattr__(self, "operation_vk_hex", _point(self.operation_vk_hex, name="operation_vk_hex"))
        if self.operation_vk_base_hex != OPERATION_VK_BASE_HEX:
            raise ValueError("operation_vk_base_hex must equal frozen G_vk")
        object.__setattr__(self, "statement_digest", domain_digest(_LINK_STATEMENT_DOMAIN, self.to_obj()))

    def to_obj(self) -> dict[str, Any]:
        return {
            "schema": LINK_STATEMENT_SCHEMA,
            "qid": self.qid,
            "arm_context_digest": self.arm_context_digest,
            "operation_context_digest": self.operation_context_digest,
            "encrypted_source_digest": self.encrypted_source_digest,
            "terminal_state_identity": self.terminal_state_identity,
            "coordinate": self.coordinate.to_obj(),
            "dealer_role_id": self.dealer_role_id,
            "operation_vk_base_hex": self.operation_vk_base_hex,
            "operation_vk_hex": self.operation_vk_hex,
        }


@dataclass(frozen=True)
class BHK22EncryptedPublicValueVKProof:
    statement_digest: str
    t_public_key_hex: str
    t_ciphertext_hex: str
    t_vk_hex: str
    z_key: int
    z_share: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "statement_digest", digest_hex(self.statement_digest, name="statement_digest"))
        for name in ("t_public_key_hex", "t_ciphertext_hex", "t_vk_hex"):
            object.__setattr__(self, name, _point(getattr(self, name), name=name))
        object.__setattr__(self, "z_key", canonical_scalar(self.z_key, name="z_key"))
        object.__setattr__(self, "z_share", canonical_scalar(self.z_share, name="z_share"))

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
        }

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22EncryptedPublicValueVKProof":
        exact_fields(value, {"kind", "version", "statement_digest", "t_public_key_hex", "t_ciphertext_hex", "t_vk_hex", "z_key", "z_share"}, name="BHK22 encrypted-public-value-vk proof")
        if value["kind"] != LINK_PROOF_KIND or value["version"] != LINK_PROOF_VERSION:
            raise ValueError("unsupported BHK22 link proof")
        for name in ("z_key", "z_share"):
            if not isinstance(value[name], str) or str(int(value[name])) != value[name]:
                raise ValueError(f"{name} must be a canonical scalar string")
        return cls(
            statement_digest=value["statement_digest"],
            t_public_key_hex=value["t_public_key_hex"],
            t_ciphertext_hex=value["t_ciphertext_hex"],
            t_vk_hex=value["t_vk_hex"],
            z_key=int(value["z_key"]),
            z_share=int(value["z_share"]),
        )

    @classmethod
    def from_bytes(cls, value: bytes | bytearray | memoryview | str) -> "BHK22EncryptedPublicValueVKProof":
        return cls.from_obj(_decode_object(value, name="BHK22 link proof"))


def _link_challenge(statement: BHK22EncryptedPublicValueVKStatement, proof_points: Sequence[str]) -> int:
    return hash_to_scalar(
        _LINK_FS_DOMAIN,
        {"statement": statement.to_obj(), "commitments": list(proof_points)},
    )


def prove_bhk22_encrypted_public_value_vk(
    statement: BHK22EncryptedPublicValueVKStatement,
    *,
    public_value_secret_key: int,
    share: int,
) -> BHK22EncryptedPublicValueVKProof:
    key = canonical_scalar(public_value_secret_key, name="public value secret key", nonzero=True)
    witness = canonical_scalar(share, name="source share", nonzero=True)
    coordinate = statement.coordinate
    generator = G1Point.from_hex(coordinate.generator_hex)
    public_key = G1Point.from_hex(coordinate.public_key_hex)
    ciphertext_x = G1Point.from_hex(coordinate.ciphertext.X)
    ciphertext_y = G1Point.from_hex(coordinate.ciphertext.Y)
    vk_base = G1Point.from_hex(statement.operation_vk_base_hex)
    vk = G1Point.from_hex(statement.operation_vk_hex)
    if public_key != generator * key:
        raise ValueError("public-value key witness does not match source coordinate")
    if ciphertext_y != ciphertext_x * key + generator * witness:
        raise ValueError("share/key witnesses do not open encrypted source coordinate")
    if vk != vk_base * witness:
        raise ValueError("share witness does not open operation vk")
    w_key = field_random()
    w_share = field_random()
    commitments = (
        (generator * w_key).to_hex(),
        (ciphertext_x * w_key + generator * w_share).to_hex(),
        (vk_base * w_share).to_hex(),
    )
    challenge = _link_challenge(statement, commitments)
    return BHK22EncryptedPublicValueVKProof(
        statement_digest=statement.statement_digest,
        t_public_key_hex=commitments[0],
        t_ciphertext_hex=commitments[1],
        t_vk_hex=commitments[2],
        z_key=field_add(w_key, field_mul(challenge, key)),
        z_share=field_add(w_share, field_mul(challenge, witness)),
    )


def verify_bhk22_encrypted_public_value_vk(
    statement: BHK22EncryptedPublicValueVKStatement,
    proof: BHK22EncryptedPublicValueVKProof,
) -> bool:
    try:
        if not hmac.compare_digest(statement.statement_digest, proof.statement_digest):
            return False
        commitments = (proof.t_public_key_hex, proof.t_ciphertext_hex, proof.t_vk_hex)
        challenge = _link_challenge(statement, commitments)
        coordinate = statement.coordinate
        generator = G1Point.from_hex(coordinate.generator_hex)
        public_key = G1Point.from_hex(coordinate.public_key_hex)
        x_point = G1Point.from_hex(coordinate.ciphertext.X)
        y_point = G1Point.from_hex(coordinate.ciphertext.Y)
        vk_base = G1Point.from_hex(statement.operation_vk_base_hex)
        vk = G1Point.from_hex(statement.operation_vk_hex)
        return bool(
            generator * proof.z_key
            == G1Point.from_hex(proof.t_public_key_hex) + public_key * challenge
            and x_point * proof.z_key + generator * proof.z_share
            == G1Point.from_hex(proof.t_ciphertext_hex) + y_point * challenge
            and vk_base * proof.z_share
            == G1Point.from_hex(proof.t_vk_hex) + vk * challenge
        )
    except Exception:
        return False


def _link_statement_from_obj(
    value: Mapping[str, Any],
) -> BHK22EncryptedPublicValueVKStatement:
    exact_fields(
        value,
        {
            "schema", "qid", "arm_context_digest", "operation_context_digest",
            "encrypted_source_digest", "terminal_state_identity", "coordinate",
            "dealer_role_id", "operation_vk_base_hex", "operation_vk_hex",
        },
        name="BHK22 encrypted-public-value-vk statement",
    )
    if value["schema"] != LINK_STATEMENT_SCHEMA:
        raise ValueError("unsupported BHK22 link statement")
    return BHK22EncryptedPublicValueVKStatement(
        qid=value["qid"],
        arm_context_digest=value["arm_context_digest"],
        operation_context_digest=value["operation_context_digest"],
        encrypted_source_digest=value["encrypted_source_digest"],
        terminal_state_identity=value["terminal_state_identity"],
        coordinate=BHK22EncryptedSourceCoordinate.from_obj(value["coordinate"]),
        dealer_role_id=value["dealer_role_id"],
        operation_vk_base_hex=value["operation_vk_base_hex"],
        operation_vk_hex=value["operation_vk_hex"],
    )


class BHK22OperationSourceState:
    """One dealer's private encrypted-source state.

    This type intentionally has no public serializer.  The local binding check
    returns only success/failure and never returns the decrypted group point.
    """

    def __init__(
        self,
        *,
        role_id: int,
        evaluation_point: int,
        recipient_secret_key: int,
        share: int,
        public_value_secret_key: int,
        coordinate: BHK22EncryptedSourceCoordinate,
        encrypted_source_digest: str,
        terminal_state_identity: str,
    ) -> None:
        self.role_id = strict_int(role_id, name="role_id")
        self.evaluation_point = strict_int(
            evaluation_point, name="evaluation_point", minimum=1
        )
        if self.evaluation_point != self.role_id + 1:
            raise ValueError("source-state evaluation point mismatch")
        if coordinate.coordinate_index != self.role_id:
            raise ValueError("source-state coordinate mismatch")
        self.coordinate = coordinate
        self.encrypted_source_digest = digest_hex(
            encrypted_source_digest, name="encrypted_source_digest"
        )
        self.terminal_state_identity = nonempty_text(
            terminal_state_identity, name="terminal_state_identity"
        )
        self._recipient_secret = bytearray(
            canonical_scalar(
                recipient_secret_key, name="recipient_secret_key", nonzero=True
            ).to_bytes(32, "big")
        )
        self._share = bytearray(
            canonical_scalar(share, name="source share", nonzero=True).to_bytes(
                32, "big"
            )
        )
        self._public_value_secret = bytearray(
            canonical_scalar(
                public_value_secret_key,
                name="public_value_secret_key",
                nonzero=True,
            ).to_bytes(32, "big")
        )
        self._consumed = False
        self.validate_local_binding()

    @property
    def consumed(self) -> bool:
        return self._consumed

    def _peek(self, value: bytearray, *, name: str) -> int:
        if self._consumed:
            raise RuntimeError("BHK22 operation source state was already consumed")
        result = int.from_bytes(value, "big")
        return canonical_scalar(result, name=name, nonzero=True)

    def _share_for_protocol(self) -> int:
        return self._peek(self._share, name="source share")

    def _key_for_protocol(self) -> int:
        return self._peek(
            self._public_value_secret, name="public_value_secret_key"
        )

    def _recipient_key_for_protocol(self) -> int:
        return self._peek(self._recipient_secret, name="recipient_secret_key")

    def validate_local_binding(self) -> None:
        share = self._share_for_protocol()
        key = self._key_for_protocol()
        coordinate = self.coordinate
        if (
            G1Point.from_hex(coordinate.generator_hex) * key
        ).to_hex() != coordinate.public_key_hex:
            raise ValueError("private public-value key does not match ciphertext coordinate")
        decrypted = decrypt_public_value(coordinate.ciphertext, secret_key=key)
        expected = (
            G1Point.from_hex(coordinate.generator_hex) * share
        ).to_hex()
        if not hmac.compare_digest(decrypted, expected):
            raise ValueError("private source share/key do not match ciphertext")

    def consume_once(self) -> None:
        if self._consumed:
            raise RuntimeError("BHK22 operation source state was already consumed")
        for secret in (
            self._recipient_secret,
            self._share,
            self._public_value_secret,
        ):
            secret[:] = b"\x00" * len(secret)
        self._consumed = True

    def wipe(self) -> None:
        if not self._consumed:
            self.consume_once()


@dataclass(frozen=True)
class BHK22OperationFixture:
    public_context: BHK22OperationPublicContext
    protocol_context: BHK22PublicContext
    source_states: Mapping[int, BHK22OperationSourceState]
    successor_public_value_secret_keys: Mapping[int, int]

    def __post_init__(self) -> None:
        states = dict(self.source_states)
        secrets = {
            strict_int(role, name="successor role"): canonical_scalar(
                value, name="successor public-value key", nonzero=True
            )
            for role, value in self.successor_public_value_secret_keys.items()
        }
        expected = set(range(self.public_context.parameters.n_r))
        if set(states) != expected or set(secrets) != expected:
            raise ValueError("operation fixture must cover every coordinate")
        object.__setattr__(self, "source_states", MappingProxyType(states))
        object.__setattr__(
            self,
            "successor_public_value_secret_keys",
            MappingProxyType(secrets),
        )


def _fixture_materials(
    bundle: BHK22FixtureBundle,
) -> dict[tuple[str, int], BHK22RolePrivateMaterial]:
    return {
        (item.role_type, item.role_id): item
        for shard in bundle.shards.values()
        for item in shard.role_materials
    }


def build_bhk22_operation_fixture(
    *,
    base_fixture: BHK22FixtureBundle,
    operation_context: ThresholdOperationContext,
    encryption_seed: str,
) -> BHK22OperationFixture:
    """Convert a common BHK source sharing into encrypted application state."""

    if not isinstance(base_fixture, BHK22FixtureBundle):
        raise TypeError("base_fixture must be a BHK22FixtureBundle")
    if not isinstance(operation_context, ThresholdOperationContext):
        raise TypeError("operation_context has the wrong type")
    seed = nonempty_text(encryption_seed, name="encryption_seed")
    base = base_fixture.manifest.context
    materials = _fixture_materials(base_fixture)
    coordinates = []
    local_values = []
    public_value_secrets: dict[int, int] = {}
    for role in range(base.parameters.n_r):
        dealer = materials[(ROLE_DEALER, role)]
        shareholder = materials[("shareholder", role)]
        if dealer.source_share is None or shareholder.public_value_secret_key is None:
            raise ValueError("fixture is missing BHK source/private public-value key")
        share = canonical_scalar(dealer.source_share, name="source share", nonzero=True)
        key = canonical_scalar(
            shareholder.public_value_secret_key,
            name="public-value secret key",
            nonzero=True,
        )
        generator = base.crs.generators[role + 1]
        ciphertext, _ = encrypt_public_value(
            generator_hex=generator,
            public_key_hex=base.crs.public_value_keys[role + 1],
            plaintext_hex=(G1Point.from_hex(generator) * share).to_hex(),
            randomness=derive_nonzero_scalar(seed, f"source-ciphertext:{role}"),
        )
        coordinate = BHK22EncryptedSourceCoordinate(
            coordinate_index=role,
            evaluation_point=role + 1,
            generator_hex=generator,
            public_key_hex=base.crs.public_value_keys[role + 1],
            ciphertext=ciphertext,
        )
        coordinates.append(coordinate)
        local_values.append((dealer, share, key, coordinate))
        public_value_secrets[role] = key
    encrypted_source = BHK22EncryptedSourceVector(tuple(coordinates))
    base = replace(base, source_public_values=(), encrypted_source=encrypted_source)
    public_context = BHK22OperationPublicContext(
        trial_id=base.trial_id,
        source_epoch=base.source_epoch,
        target_epoch=base.target_epoch,
        transfer_seq=base.transfer_seq,
        base_context_digest=base.root_context_digest,
        parameters=base.parameters,
        crs=base.crs,
        committees=base.committees,
        recipient_key_catalogs=base.recipient_key_catalogs,
        operation_context=operation_context,
        encrypted_source=encrypted_source,
    )
    terminal_identity = f"genesis:{encrypted_source.digest}"
    states = {
        role: BHK22OperationSourceState(
            role_id=role,
            evaluation_point=role + 1,
            recipient_secret_key=dealer.recipient_secret_key,
            share=share,
            public_value_secret_key=key,
            coordinate=coordinate,
            encrypted_source_digest=encrypted_source.digest,
            terminal_state_identity=terminal_identity,
        )
        for role, (dealer, share, key, coordinate) in enumerate(local_values)
    }
    return BHK22OperationFixture(
        public_context=public_context,
        protocol_context=base,
        source_states=states,
        successor_public_value_secret_keys=public_value_secrets,
    )


@dataclass(frozen=True)
class BHK22OperationDealerBatch:
    arm_context_digest: str
    operation_context_digest: str
    encrypted_source_digest: str
    terminal_state_identity: str
    base_batch: BHK22DealerBatch
    link_statement: BHK22EncryptedPublicValueVKStatement
    link_proof: BHK22EncryptedPublicValueVKProof
    operation_statement: VKOperationShareStatement
    operation_proof: VKOperationShareProof
    message_digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "arm_context_digest",
            "operation_context_digest",
            "encrypted_source_digest",
        ):
            object.__setattr__(self, name, digest_hex(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "terminal_state_identity",
            nonempty_text(self.terminal_state_identity, name="terminal_state_identity"),
        )
        if not isinstance(self.base_batch, BHK22DealerBatch):
            raise TypeError("base_batch must be BHK22DealerBatch")
        if not isinstance(self.link_statement, BHK22EncryptedPublicValueVKStatement):
            raise TypeError("link_statement has wrong type")
        if not isinstance(self.link_proof, BHK22EncryptedPublicValueVKProof):
            raise TypeError("link_proof has wrong type")
        if not isinstance(self.operation_statement, VKOperationShareStatement):
            raise TypeError("operation_statement has wrong type")
        if not isinstance(self.operation_proof, VKOperationShareProof):
            raise TypeError("operation_proof has wrong type")
        dealer = self.base_batch.role_id
        if self.link_statement.dealer_role_id != dealer or self.operation_statement.holder_role_id != dealer:
            raise ValueError("operation dealer bundle role mismatch")
        if self.link_statement.operation_vk_hex != self.operation_statement.operation_vk_hex:
            raise ValueError("link and operation proofs must use the same vk")
        if self.link_statement.qid != self.base_batch.phase_qid or self.operation_statement.qid != self.base_batch.phase_qid:
            raise ValueError("operation dealer bundle qid mismatch")
        if (
            self.link_statement.arm_context_digest != self.arm_context_digest
            or self.operation_statement.arm_context_digest != self.arm_context_digest
            or self.link_statement.operation_context_digest != self.operation_context_digest
            or self.operation_statement.operation_context_digest != self.operation_context_digest
            or self.link_statement.encrypted_source_digest != self.encrypted_source_digest
            or self.operation_statement.parent_state_digest != self.encrypted_source_digest
            or self.link_statement.terminal_state_identity != self.terminal_state_identity
        ):
            raise ValueError("operation dealer bundle context binding mismatch")
        object.__setattr__(self, "message_digest", domain_digest(_BATCH_DOMAIN, self.payload_obj()))

    @property
    def dealer_role_id(self) -> int:
        return self.base_batch.role_id

    # The phase/RBC layer is intentionally structural and consumes the same
    # public header interface for base and application dealer messages.
    @property
    def role_id(self) -> int:
        return self.base_batch.role_id

    @property
    def root_context_digest(self) -> str:
        return self.base_batch.root_context_digest

    @property
    def phase_qid(self) -> str:
        return self.base_batch.phase_qid

    @property
    def n_r(self) -> int:
        return self.base_batch.n_r

    @property
    def network_sender_id(self) -> int:
        return self.base_batch.network_sender_id

    def payload_obj(self) -> dict[str, Any]:
        operation_statement_obj = self.operation_statement.to_obj()
        operation_statement_obj["statement_digest"] = (
            self.operation_statement.statement_digest
        )
        return {
            "schema": BHK22_OPERATION_DEALER_BATCH_SCHEMA,
            "arm_context_digest": self.arm_context_digest,
            "operation_context_digest": self.operation_context_digest,
            "encrypted_source_digest": self.encrypted_source_digest,
            "terminal_state_identity": self.terminal_state_identity,
            "base_batch": self.base_batch.to_obj(),
            "link_statement": self.link_statement.to_obj(),
            "link_proof": self.link_proof.to_obj(),
            "operation_statement": operation_statement_obj,
            "operation_proof": self.operation_proof.to_obj(),
        }

    def to_obj(self) -> dict[str, Any]:
        value = self.payload_obj()
        value["message_digest"] = self.message_digest
        return value

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22OperationDealerBatch":
        expected = {
            "schema", "arm_context_digest", "operation_context_digest",
            "encrypted_source_digest", "terminal_state_identity", "base_batch",
            "link_statement", "link_proof", "operation_statement",
            "operation_proof", "message_digest",
        }
        exact_fields(value, expected, name="BHK22 operation dealer batch")
        if value["schema"] != BHK22_OPERATION_DEALER_BATCH_SCHEMA:
            raise ValueError("unsupported BHK22 operation dealer batch")
        result = cls(
            arm_context_digest=value["arm_context_digest"],
            operation_context_digest=value["operation_context_digest"],
            encrypted_source_digest=value["encrypted_source_digest"],
            terminal_state_identity=value["terminal_state_identity"],
            base_batch=BHK22DealerBatch.from_obj(value["base_batch"]),
            link_statement=_link_statement_from_obj(value["link_statement"]),
            link_proof=BHK22EncryptedPublicValueVKProof.from_obj(value["link_proof"]),
            operation_statement=VKOperationShareStatement.from_obj(value["operation_statement"]),
            operation_proof=VKOperationShareProof.from_obj(value["operation_proof"]),
        )
        if result.message_digest != digest_hex(value["message_digest"], name="message_digest"):
            raise ValueError("BHK22 operation dealer batch digest mismatch")
        return result

    @classmethod
    def from_bytes(cls, value: bytes | bytearray | memoryview | str) -> "BHK22OperationDealerBatch":
        return cls.from_obj(_decode_object(value, name="BHK22 operation dealer batch"))


class _RecordingDealerRandomSource(BHK22DealerRandomSource):
    def __init__(self, delegate: BHK22DealerRandomSource) -> None:
        self.delegate = delegate
        self.scalars: dict[str, int] = {}

    def scalar(self, label: str) -> int:
        value = self.delegate.scalar(label)
        self.scalars[label] = value
        return value

    def bytes(self, label: str, length: int) -> bytes:
        return self.delegate.bytes(label, length)


@dataclass(frozen=True)
class BHK22OperationDealerGenerationResult:
    batch: BHK22OperationDealerBatch
    first_level_shares: Sequence[int]
    generation_ms: float
    generation_cpu_seconds: float = 0.0

    def __post_init__(self) -> None:
        shares = tuple(canonical_scalar(item, name="first-level share") for item in self.first_level_shares)
        if len(shares) != self.batch.base_batch.n_r:
            raise ValueError("first-level private handoff length mismatch")
        object.__setattr__(self, "first_level_shares", shares)
        if self.generation_ms < 0:
            raise ValueError("generation_ms cannot be negative")
        if self.generation_cpu_seconds < 0:
            raise ValueError("generation_cpu_seconds cannot be negative")


class BHK22OperationDealerRole:
    """Generate resharing, vk, pi_link, y and pi_op in one speech."""

    def __init__(
        self,
        *,
        context: BHK22OperationPublicContext,
        protocol_context: BHK22PublicContext,
        state: BHK22OperationSourceState,
        phase_qid: str | None = None,
        random_source: BHK22DealerRandomSource | None = None,
    ) -> None:
        if not isinstance(context, BHK22OperationPublicContext) or not isinstance(protocol_context, BHK22PublicContext):
            raise TypeError("BHK22 operation dealer context has wrong type")
        if not isinstance(state, BHK22OperationSourceState):
            raise TypeError("BHK22 operation dealer state has wrong type")
        if protocol_context.root_context_digest != context.base_context_digest:
            raise ValueError("base/application context digest mismatch")
        if protocol_context.crs != context.crs or protocol_context.parameters != context.parameters:
            raise ValueError("base/application public parameters mismatch")
        if state.encrypted_source_digest != context.encrypted_source.digest:
            raise ValueError("dealer state belongs to another encrypted source")
        self.context = context
        self.protocol_context = protocol_context
        self.state = state
        self.phase_qid = _qid(
            phase_qid or build_bhk22_operation_qid(context, phase="deal"),
            phase="deal",
        )
        self.random_source = random_source or BHK22DealerRandomSource()

    def generate_batch(self) -> BHK22OperationDealerGenerationResult:
        started = perf_counter()
        cpu_started = process_time()
        context = self.context
        state = self.state
        state.validate_local_binding()
        share = state._share_for_protocol()
        key = state._key_for_protocol()
        recipient_key = state._recipient_key_for_protocol()
        dealer = state.role_id
        coordinate = context.encrypted_source.coordinates[dealer]
        operation_statement, operation_proof = evaluate_and_prove_vk_operation_share(
            protocol="bhk22",
            trial_id=context.trial_id,
            qid=self.phase_qid,
            parent_state_digest=context.encrypted_source.digest,
            arm_context_digest=context.context_digest,
            operation_context=context.operation_context,
            holder_role_id=dealer,
            evaluation_point=state.evaluation_point,
            share=share,
        )
        link_statement = BHK22EncryptedPublicValueVKStatement(
            qid=self.phase_qid,
            arm_context_digest=context.context_digest,
            operation_context_digest=context.operation_context.context_digest,
            encrypted_source_digest=context.encrypted_source.digest,
            terminal_state_identity=state.terminal_state_identity,
            coordinate=coordinate,
            dealer_role_id=dealer,
            operation_vk_base_hex=context.operation_context.operation_vk_base_hex,
            operation_vk_hex=operation_statement.operation_vk_hex,
        )
        link_proof = prove_bhk22_encrypted_public_value_vk(
            link_statement,
            public_value_secret_key=key,
            share=share,
        )
        ephemeral_material = BHK22RolePrivateMaterial(
            role_type=ROLE_DEALER,
            role_id=dealer,
            evaluation_point=state.evaluation_point,
            recipient_secret_key=recipient_key,
            source_share=share,
            public_value_secret_key=None,
        )
        recording = _RecordingDealerRandomSource(self.random_source)
        base_result = BHK22DealerRole(
            context=self.protocol_context,
            state=ephemeral_material.new_state(),
            phase_qid=self.phase_qid,
            random_source=recording,
            application_figure7_share_base_hex=context.operation_context.operation_vk_base_hex,
            application_figure7_share_point_hex=operation_statement.operation_vk_hex,
        ).generate_batch()
        degree = context.parameters.t_r
        first_poly = (share,) + tuple(
            recording.scalars[f"dealer:{dealer}:first-level:coefficient:{power}"]
            for power in range(1, degree + 1)
        )
        first_level = tuple(
            eval_poly(first_poly, point)
            for point in range(1, context.parameters.n_r + 1)
        )
        batch = BHK22OperationDealerBatch(
            arm_context_digest=context.context_digest,
            operation_context_digest=context.operation_context.context_digest,
            encrypted_source_digest=context.encrypted_source.digest,
            terminal_state_identity=state.terminal_state_identity,
            base_batch=base_result.batch,
            link_statement=link_statement,
            link_proof=link_proof,
            operation_statement=operation_statement,
            operation_proof=operation_proof,
        )
        # Atomic application consumption happens only after every public object
        # and every private successor contribution has been formed.
        state.consume_once()
        return BHK22OperationDealerGenerationResult(
            batch=batch,
            first_level_shares=first_level,
            generation_ms=(perf_counter() - started) * 1000.0,
            generation_cpu_seconds=process_time() - cpu_started,
        )


@dataclass(frozen=True)
class BHK22OperationDealerVerification:
    accepted: bool
    checks: Mapping[str, bool]
    timings_ms: Mapping[str, float]
    cpu_seconds: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        checks = {str(name): bool(value) for name, value in self.checks.items()}
        timings = {str(name): float(value) for name, value in self.timings_ms.items()}
        cpu = {str(name): float(value) for name, value in self.cpu_seconds.items()}
        if self.accepted != all(checks.values()):
            raise ValueError("dealer verification accepted/check mismatch")
        if any(value < 0 for value in timings.values()):
            raise ValueError("dealer verification timings cannot be negative")
        if any(value < 0 for value in cpu.values()):
            raise ValueError("dealer verification CPU timings cannot be negative")
        object.__setattr__(self, "checks", MappingProxyType(checks))
        object.__setattr__(self, "timings_ms", MappingProxyType(timings))
        object.__setattr__(self, "cpu_seconds", MappingProxyType(cpu))


def verify_bhk22_operation_dealer_batch(
    context: BHK22OperationPublicContext,
    batch: BHK22OperationDealerBatch,
    *,
    expected_phase_qid: str | None = None,
) -> BHK22OperationDealerVerification:
    """Run base Figure-6/8, application Figure-7, link and op checks."""

    check_names = (
        "wire_context",
        "column_degree",
        "column_zero",
        "shifted_column",
        "delta",
        "base_figure7",
        "figure8_proof",
        "encrypted_public_value_to_vk",
        "vk_to_operation_share",
    )
    checks = {name: False for name in check_names}
    timings = {
        "base_public": 0.0,
        "base_figure7": 0.0,
        "figure8_proof": 0.0,
        "pi_link": 0.0,
        "pi_op": 0.0,
    }
    cpu_timings = {name: 0.0 for name in timings}
    try:
        if not isinstance(context, BHK22OperationPublicContext) or not isinstance(
            batch, BHK22OperationDealerBatch
        ):
            raise TypeError("wrong BHK22 operation dealer verifier input")
        base = batch.base_batch
        dealer = base.role_id
        base_started = perf_counter()
        base_cpu_started = process_time()
        checks["wire_context"] = bool(
            base.root_context_digest == context.base_context_digest
            and base.n_r == context.parameters.n_r
            and batch.arm_context_digest == context.context_digest
            and batch.operation_context_digest
            == context.operation_context.context_digest
            and batch.encrypted_source_digest == context.encrypted_source.digest
            and base.network_sender_id
            == context.committees[ROLE_DEALER].role_to_node[dealer]
            and (
                expected_phase_qid is None
                or base.phase_qid == _qid(expected_phase_qid, phase="deal")
            )
        )
        commitments = tuple(
            item.column_commitment for item in base.verifier_columns
        )
        evaluation_points = tuple(range(1, base.n_r + 1))
        checks["column_degree"] = pedersen_deg_check(
            commitments,
            context.parameters.t_r,
            eval_points=evaluation_points,
        )
        basis_points = evaluation_points[: context.parameters.t_r + 1]
        coefficients = tuple(
            lagrange_coeff_at_zero(basis_points, point)
            for point in basis_points
        )
        checks["column_zero"] = (
            g1_linear_combination(
                commitments[: context.parameters.t_r + 1], coefficients
            ).to_hex()
            == base.c_i0
        )
        checks["shifted_column"] = (
            g1_linear_combination(
                [item.Y for item in base.figure8_ciphertexts],
                [1] * (base.n_r + 1),
            ).to_hex()
            == base.c_prime_i0
        )
        checks["delta"] = (
            G1Point.from_hex(base.c_prime_i0) - G1Point.from_hex(base.c_i0)
        ).to_hex() == base.delta_i
        timings["base_public"] = (perf_counter() - base_started) * 1000.0
        cpu_timings["base_public"] = process_time() - base_cpu_started

        proof_context = BHK22ProofContext(
            root_context_digest=base.root_context_digest,
            phase_qid=base.phase_qid,
            role_type=ROLE_DEALER,
            role_id=dealer,
        )
        figure7_started = perf_counter()
        figure7_cpu_started = process_time()
        checks["base_figure7"] = verify_figure7(
            context=proof_context,
            anchor_base_hex=context.crs.generators[0],
            share_base_hex=context.operation_context.operation_vk_base_hex,
            eval_bases=context.crs.generators[1:],
            share_point_hex=batch.operation_statement.operation_vk_hex,
            aggregate_commitment_hex=base.c_i0,
            degree=context.parameters.t_r,
            eval_points=evaluation_points,
            proof=base.figure7_proof,
        )
        timings["base_figure7"] = (perf_counter() - figure7_started) * 1000.0
        cpu_timings["base_figure7"] = process_time() - figure7_cpu_started

        figure8_started = perf_counter()
        figure8_cpu_started = process_time()
        checks["figure8_proof"] = verify_figure8_randomness(
            context=proof_context,
            delta_hex=base.delta_i,
            ciphertexts=base.figure8_ciphertexts,
            h_bases=context.crs.public_value_keys,
            g_bases=context.crs.generators,
            proof=base.figure8_proof,
        )
        timings["figure8_proof"] = (perf_counter() - figure8_started) * 1000.0
        cpu_timings["figure8_proof"] = process_time() - figure8_cpu_started

        expected_coordinate = context.encrypted_source.coordinates[dealer]
        link_started = perf_counter()
        link_cpu_started = process_time()
        checks["encrypted_public_value_to_vk"] = bool(
            batch.link_statement.coordinate == expected_coordinate
            and batch.link_statement.arm_context_digest == context.context_digest
            and batch.link_statement.operation_context_digest
            == context.operation_context.context_digest
            and batch.link_statement.encrypted_source_digest
            == context.encrypted_source.digest
            and verify_bhk22_encrypted_public_value_vk(
                batch.link_statement, batch.link_proof
            )
        )
        timings["pi_link"] = (perf_counter() - link_started) * 1000.0
        cpu_timings["pi_link"] = process_time() - link_cpu_started

        op_started = perf_counter()
        op_cpu_started = process_time()
        checks["vk_to_operation_share"] = bool(
            batch.operation_statement.protocol == "bhk22"
            and batch.operation_statement.trial_id == context.trial_id
            and batch.operation_statement.arm_context_digest
            == context.context_digest
            and batch.operation_statement.operation_context_digest
            == context.operation_context.context_digest
            and batch.operation_statement.parent_state_digest
            == context.encrypted_source.digest
            and batch.operation_statement.operation_vk_hex
            == batch.link_statement.operation_vk_hex
            and verify_vk_operation_share(
                batch.operation_statement, batch.operation_proof
            )
        )
        timings["pi_op"] = (perf_counter() - op_started) * 1000.0
        cpu_timings["pi_op"] = process_time() - op_cpu_started
    except Exception:
        pass
    return BHK22OperationDealerVerification(
        accepted=all(checks.values()), checks=checks, timings_ms=timings,
        cpu_seconds=cpu_timings,
    )


@dataclass(frozen=True)
class BHK22OperationPublicResult:
    arm_context_digest: str
    operation_context_digest: str
    path_mode: str
    block_count: int
    terminal_phase_qid: str
    qualified_dealer_ids: Sequence[int]
    selected_dealer_ids: Sequence[int]
    dealer_lagrange_coefficients: Sequence[int]
    operation_output_hex: str
    operation_output_digest: str
    encrypted_successor: BHK22EncryptedSourceVector
    speech_counts: Mapping[int, int]
    result_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.path_mode not in BHK22_ENCRYPTED_CARRY_FORWARD_BLOCK_COUNTS:
            raise ValueError("unsupported BHK22 operation path mode")
        if self.block_count != BHK22_ENCRYPTED_CARRY_FORWARD_BLOCK_COUNTS[self.path_mode]:
            raise ValueError("BHK22 operation block count mismatch")
        qualified = tuple(strict_int(item, name="qualified dealer") for item in self.qualified_dealer_ids)
        selected = tuple(strict_int(item, name="selected dealer") for item in self.selected_dealer_ids)
        coefficients = tuple(canonical_scalar(item, name="dealer coefficient") for item in self.dealer_lagrange_coefficients)
        if tuple(sorted(set(qualified))) != qualified or not selected or not set(selected) <= set(qualified):
            raise ValueError("invalid BHK22 operation dealer selection")
        if len(selected) != len(coefficients):
            raise ValueError("selected dealers/coefficients length mismatch")
        speech = {strict_int(role, name="speech role"): strict_int(count, name="speech count") for role, count in self.speech_counts.items()}
        if any(count > 1 for count in speech.values()):
            raise ValueError("a BHK physical dealer role spoke more than once")
        object.__setattr__(self, "arm_context_digest", digest_hex(self.arm_context_digest, name="arm_context_digest"))
        object.__setattr__(self, "operation_context_digest", digest_hex(self.operation_context_digest, name="operation_context_digest"))
        object.__setattr__(self, "terminal_phase_qid", nonempty_text(self.terminal_phase_qid, name="terminal_phase_qid"))
        object.__setattr__(self, "qualified_dealer_ids", qualified)
        object.__setattr__(self, "selected_dealer_ids", selected)
        object.__setattr__(self, "dealer_lagrange_coefficients", coefficients)
        object.__setattr__(self, "operation_output_hex", _point(self.operation_output_hex, name="operation output"))
        object.__setattr__(self, "operation_output_digest", digest_hex(self.operation_output_digest, name="operation_output_digest"))
        if not isinstance(self.encrypted_successor, BHK22EncryptedSourceVector):
            raise TypeError("encrypted_successor has wrong type")
        object.__setattr__(self, "speech_counts", MappingProxyType(speech))
        object.__setattr__(self, "result_digest", domain_digest(_RESULT_DOMAIN, self.digest_material()))

    def digest_material(self) -> dict[str, Any]:
        return {
            "schema": OPERATION_RESULT_SCHEMA,
            "arm_context_digest": self.arm_context_digest,
            "operation_context_digest": self.operation_context_digest,
            "path_mode": self.path_mode,
            "block_count": self.block_count,
            "terminal_phase_qid": self.terminal_phase_qid,
            "qualified_dealer_ids": list(self.qualified_dealer_ids),
            "selected_dealer_ids": list(self.selected_dealer_ids),
            "dealer_lagrange_coefficients": [str(item) for item in self.dealer_lagrange_coefficients],
            "operation_output_hex": self.operation_output_hex,
            "operation_output_digest": self.operation_output_digest,
            "encrypted_successor": self.encrypted_successor.to_obj(),
            "speech_counts": {str(role): count for role, count in self.speech_counts.items()},
            "public_finalize_speech": False,
            "shareholder_public_speech": False,
        }

    def to_obj(self) -> dict[str, Any]:
        value = self.digest_material()
        value["result_digest"] = self.result_digest
        return value

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22OperationPublicResult":
        expected = {
            "schema", "arm_context_digest", "operation_context_digest",
            "path_mode", "block_count", "terminal_phase_qid",
            "qualified_dealer_ids", "selected_dealer_ids",
            "dealer_lagrange_coefficients", "operation_output_hex",
            "operation_output_digest", "encrypted_successor", "speech_counts",
            "public_finalize_speech", "shareholder_public_speech", "result_digest",
        }
        exact_fields(value, expected, name="BHK22 operation public result")
        if value["schema"] != OPERATION_RESULT_SCHEMA:
            raise ValueError("unsupported BHK22 operation public result")
        if value["public_finalize_speech"] is not False or value["shareholder_public_speech"] is not False:
            raise ValueError("encrypted carry-forward must not contain finalize speech")
        coefficients = value["dealer_lagrange_coefficients"]
        if not all(isinstance(item, str) and str(int(item)) == item for item in coefficients):
            raise ValueError("dealer coefficients must be canonical scalar strings")
        result = cls(
            arm_context_digest=value["arm_context_digest"],
            operation_context_digest=value["operation_context_digest"],
            path_mode=value["path_mode"],
            block_count=value["block_count"],
            terminal_phase_qid=value["terminal_phase_qid"],
            qualified_dealer_ids=value["qualified_dealer_ids"],
            selected_dealer_ids=value["selected_dealer_ids"],
            dealer_lagrange_coefficients=tuple(int(item) for item in coefficients),
            operation_output_hex=value["operation_output_hex"],
            operation_output_digest=value["operation_output_digest"],
            encrypted_successor=BHK22EncryptedSourceVector.from_obj(value["encrypted_successor"]),
            speech_counts={int(role): count for role, count in value["speech_counts"].items()},
        )
        if result.result_digest != digest_hex(value["result_digest"], name="result_digest"):
            raise ValueError("BHK22 operation public result digest mismatch")
        return result

    @classmethod
    def from_bytes(
        cls, value: bytes | bytearray | memoryview | str
    ) -> "BHK22OperationPublicResult":
        return cls.from_obj(_decode_object(value, name="BHK22 operation public result"))


@dataclass(frozen=True)
class BHK22OperationExecutionResult:
    public_result: BHK22OperationPublicResult
    successor_states: Mapping[int, BHK22CarryForwardSuccessorState]
    dealer_verifications: Mapping[int, BHK22OperationDealerVerification]
    application_checks: Mapping[str, bool]

    def __post_init__(self) -> None:
        states = dict(self.successor_states)
        verifications = dict(self.dealer_verifications)
        checks = {str(name): bool(value) for name, value in self.application_checks.items()}
        if not accepts_operation_candidate("bhk22", checks):
            raise ValueError("BHK22 operation application acceptance checks failed")
        object.__setattr__(self, "successor_states", MappingProxyType(states))
        object.__setattr__(self, "dealer_verifications", MappingProxyType(verifications))
        object.__setattr__(self, "application_checks", MappingProxyType(checks))


def execute_bhk22_operation_handover(
    *,
    context: BHK22OperationPublicContext,
    dealer_results: Sequence[BHK22OperationDealerGenerationResult],
    successor_public_value_secret_keys: Mapping[int, int],
    path_mode: str = PATH_OPTIMISTIC,
    pre_disqualified_dealer_ids: Sequence[int] = (),
) -> BHK22OperationExecutionResult:
    """No-network terminal qualification/combine for the BHK application arm."""

    if path_mode not in (PATH_OPTIMISTIC, PATH_ONE_ACCUSATION):
        raise ValueError("unsupported BHK operation path")
    results = {item.batch.dealer_role_id: item for item in dealer_results}
    if set(results) != set(range(context.parameters.n_r)):
        raise ValueError("dealer_results must cover every BHK dealer role")
    deal_qid = build_bhk22_operation_qid(context, phase="deal")
    verifications = {
        dealer: verify_bhk22_operation_dealer_batch(
            context, item.batch, expected_phase_qid=deal_qid
        )
        for dealer, item in results.items()
    }
    disqualified = {
        strict_int(item, name="pre-disqualified dealer")
        for item in pre_disqualified_dealer_ids
    }
    disqualified.update(
        dealer for dealer, verification in verifications.items() if not verification.accepted
    )
    qualified = tuple(
        dealer for dealer in range(context.parameters.n_r) if dealer not in disqualified
    )
    if len(qualified) < context.parameters.t_r + 1:
        raise ValueError("insufficient qualified BHK operation dealers")
    selected = qualified[: context.parameters.t_r + 1]
    evaluation_points = tuple(dealer + 1 for dealer in selected)
    coefficients = tuple(
        lagrange_coeff_at_zero(evaluation_points, point)
        for point in evaluation_points
    )
    operation_output = combine_vk_operation_shares(
        evaluation_points,
        [results[dealer].batch.operation_statement.operation_share_hex for dealer in selected],
    ).to_hex()
    output_digest = operation_output_digest(
        operation_id=context.operation_context.operation_id,
        operation_input=context.operation_context.operation_input,
        operation_output=operation_output,
    )
    if output_digest != context.operation_context.expected_operation_output_digest:
        raise ValueError("BHK operation output does not match expected digest")

    keys = {
        strict_int(role, name="successor role"): canonical_scalar(
            key, name="successor public-value key", nonzero=True
        )
        for role, key in successor_public_value_secret_keys.items()
    }
    if set(keys) != set(range(context.parameters.n_r)):
        raise ValueError("successor public-value keys must cover every coordinate")
    terminal_phase = "response" if path_mode == PATH_ONE_ACCUSATION else "verify"
    terminal_qid = build_bhk22_operation_qid(context, phase=terminal_phase)
    successor_coordinates = []
    successor_states: dict[int, BHK22CarryForwardSuccessorState] = {}
    terminal_matches = True
    for shareholder in range(context.parameters.n_r):
        share = sum(
            coefficient * results[dealer].first_level_shares[shareholder]
            for dealer, coefficient in zip(selected, coefficients)
        ) % FIELD_ORDER
        ciphertext = combine_public_value_ciphertexts(
            [
                results[dealer].batch.base_batch.figure8_ciphertexts[shareholder + 1]
                for dealer in selected
            ],
            coefficients,
        )
        coordinate = BHK22EncryptedSourceCoordinate(
            coordinate_index=shareholder,
            evaluation_point=shareholder + 1,
            generator_hex=context.crs.generators[shareholder + 1],
            public_key_hex=context.crs.public_value_keys[shareholder + 1],
            ciphertext=ciphertext,
        )
        successor_coordinates.append(coordinate)
        expected = (
            G1Point.from_hex(coordinate.generator_hex) * share
        ).to_hex()
        terminal_matches = terminal_matches and hmac.compare_digest(
            decrypt_public_value(ciphertext, secret_key=keys[shareholder]),
            expected,
        )
        successor_states[shareholder] = BHK22CarryForwardSuccessorState(
            shareholder_role_id=shareholder,
            evaluation_point=shareholder + 1,
            share=share,
            public_value_secret_key=keys[shareholder],
            aggregate_ciphertext=ciphertext,
            terminal_phase_qid=terminal_qid,
            root_context_digest=context.context_digest,
        )
    if not terminal_matches:
        raise ValueError("encrypted successor commitment does not match private state")
    encrypted_successor = BHK22EncryptedSourceVector(tuple(successor_coordinates))
    public_result = BHK22OperationPublicResult(
        arm_context_digest=context.context_digest,
        operation_context_digest=context.operation_context.context_digest,
        path_mode=path_mode,
        block_count=BHK22_ENCRYPTED_CARRY_FORWARD_BLOCK_COUNTS[path_mode],
        terminal_phase_qid=terminal_qid,
        qualified_dealer_ids=qualified,
        selected_dealer_ids=selected,
        dealer_lagrange_coefficients=coefficients,
        operation_output_hex=operation_output,
        operation_output_digest=output_digest,
        encrypted_successor=encrypted_successor,
        speech_counts={dealer: 1 for dealer in range(context.parameters.n_r)},
    )
    app_checks = {
        "base_resharing_valid": all(
            all(
                verification.checks[name]
                for name in (
                    "wire_context", "column_degree", "column_zero",
                    "shifted_column", "delta", "figure8_proof",
                )
            )
            for dealer, verification in verifications.items()
            if dealer in qualified
        ),
        "encrypted_public_value_to_vk_proof_valid": all(
            verifications[dealer].checks["encrypted_public_value_to_vk"]
            for dealer in qualified
        ),
        "vk_to_operation_share_proof_valid": all(
            verifications[dealer].checks["vk_to_operation_share"]
            for dealer in qualified
        ),
        "figure7_resharing_anchored_to_same_vk": all(
            verifications[dealer].checks["base_figure7"] for dealer in qualified
        ),
        "terminal_encrypted_state_matches": terminal_matches,
        "same_selected_set": tuple(public_result.selected_dealer_ids) == selected,
        "operation_output_matches_expected": output_digest
        == context.operation_context.expected_operation_output_digest,
    }
    return BHK22OperationExecutionResult(
        public_result=public_result,
        successor_states=successor_states,
        dealer_verifications=verifications,
        application_checks=app_checks,
    )


def advance_bhk22_operation_public_context(
    previous: BHK22OperationPublicContext,
    *,
    operation_context: ThresholdOperationContext,
    encrypted_source: BHK22EncryptedSourceVector,
) -> BHK22OperationPublicContext:
    """Build the next-hop public context without a plaintext commitment vector."""

    return BHK22OperationPublicContext(
        trial_id=previous.trial_id,
        source_epoch=previous.target_epoch,
        target_epoch=previous.target_epoch + 1,
        transfer_seq=previous.transfer_seq + 1,
        base_context_digest=previous.base_context_digest,
        parameters=previous.parameters,
        crs=previous.crs,
        committees=previous.committees,
        recipient_key_catalogs=previous.recipient_key_catalogs,
        operation_context=operation_context,
        encrypted_source=encrypted_source,
    )


def load_bhk22_carry_forward_dealer_states(
    *,
    context: BHK22OperationPublicContext,
    carry_forward_states: Mapping[int, BHK22CarryForwardSuccessorState],
    dealer_recipient_secret_keys: Mapping[int, int],
) -> Mapping[int, BHK22OperationSourceState]:
    """Consume hop-N carry state and load exact hop-(N+1) dealer state."""

    carries = dict(carry_forward_states)
    recipient_keys = dict(dealer_recipient_secret_keys)
    expected = set(range(context.parameters.n_r))
    if set(carries) != expected or set(recipient_keys) != expected:
        raise ValueError("carry-forward loader must cover every coordinate")
    loaded: dict[int, BHK22OperationSourceState] = {}
    for role in range(context.parameters.n_r):
        carry = carries[role]
        coordinate = context.encrypted_source.coordinates[role]
        if carry.aggregate_ciphertext != coordinate.ciphertext:
            raise ValueError("carry-forward ciphertext does not match public source")
        secrets = carry.take_for_next_dealer_once()
        loaded[role] = BHK22OperationSourceState(
            role_id=role,
            evaluation_point=role + 1,
            recipient_secret_key=recipient_keys[role],
            share=secrets.share,
            public_value_secret_key=secrets.public_value_secret_key,
            coordinate=coordinate,
            encrypted_source_digest=context.encrypted_source.digest,
            terminal_state_identity=secrets.terminal_phase_qid,
        )
    return MappingProxyType(loaded)


__all__ = [
    "BHK22EncryptedPublicValueVKProof",
    "BHK22EncryptedPublicValueVKStatement",
    "BHK22EncryptedSourceCoordinate",
    "BHK22EncryptedSourceVector",
    "BHK22OperationDealerBatch",
    "BHK22OperationDealerGenerationResult",
    "BHK22OperationDealerRole",
    "BHK22OperationDealerVerification",
    "BHK22OperationExecutionResult",
    "BHK22OperationFixture",
    "BHK22OperationPublicContext",
    "BHK22OperationPublicResult",
    "BHK22OperationSourceState",
    "advance_bhk22_operation_public_context",
    "build_bhk22_operation_fixture",
    "build_bhk22_operation_qid",
    "execute_bhk22_operation_handover",
    "load_bhk22_carry_forward_dealer_states",
    "prove_bhk22_encrypted_public_value_vk",
    "verify_bhk22_encrypted_public_value_vk",
    "verify_bhk22_operation_dealer_batch",
]
