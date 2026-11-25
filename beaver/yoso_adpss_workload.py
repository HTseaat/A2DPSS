"""Common, privacy-preserving fixture for selected YOSO ADPSS arms.

The public manifest binds the experiment input using digests.  Secret source
shares, recipient keys, and threshold-coin shares only occur in the shard for
the node that owns them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from beaver.a2dpss import A2DPSSContext
from beaver.a2dpss_experiment import g_h_from_config_extra
from beaver.atcaas_params import ATCaaSProtocolParameters
from beaver.baselines.bhk22_contract import (
    DEFAULT_COMPLETION_MODE,
    ROLE_DEALER,
    ROLE_RESPONDER,
    ROLE_SHAREHOLDER,
    ROLE_TYPES,
    ROLE_VERIFIER,
)
from beaver.baselines.bhk22_context import (
    COMMITTEE_SELECTION_EXTERNAL,
    COMMITTEE_SELECTION_VRF_EXACT,
    bhk22_committee_sortition_label,
)
from beaver.baselines.bhk22_experiment import (
    BHK22DAGExperimentManifest,
    BHK22NodePrivateShard,
    bhk22_coin_config_digest,
    bhk22_dag_config_digest,
    build_bhk22_fixture,
)
from beaver.baselines.ru22_experiment import (
    RU22DAGExperimentManifest,
    RU22DAGLocalShard,
    build_worst_case_multi_vps_fixture,
)
from beaver.g1_backend import FIELD_ORDER
from beaver.handover_fixtures import build_handover_local_config
from beaver.pedersen_shamir import eval_poly
from beaver.vrf import (
    vrf_backend_available,
    vrf_prove_sortition,
    vrf_verify_sortition,
)


WORKLOAD_SCHEMA = "yoso-adpss-dag-workload/v1"
ARM_SCHEMA = "yoso-adpss-dag-arm/v1"
PRIVATE_SHARD_SCHEMA = "yoso-adpss-dag-node-private/v1"
ARM_ORDER = ("a2dpss", "ru22", "bhk22")
DUAL_ARM_ORDER = ("a2dpss", "ru22")


def normalize_protocols(protocols: Sequence[str] = ARM_ORDER) -> tuple[str, ...]:
    """Select the thesis pair or the unchanged historical triplet."""
    values = tuple(protocols)
    if len(set(values)) != len(values) or set(values) not in (
        set(DUAL_ARM_ORDER), set(ARM_ORDER)
    ):
        raise ValueError("protocols must be a2dpss ru22, optionally followed by bhk22")
    return tuple(name for name in ARM_ORDER if name in values)

_WORKLOAD_DOMAIN = b"ATCaaS/YOSO-ADPSS/workload/v1\x00"
_ARM_DOMAIN = b"ATCaaS/YOSO-ADPSS/arm/v1\x00"
_SHARD_DOMAIN = b"ATCaaS/YOSO-ADPSS/node-private/v1\x00"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def domain_digest(domain: bytes, value: Any) -> str:
    return hashlib.sha256(domain + canonical_json(value)).hexdigest()


def _digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise ValueError(f"{name} must be lowercase sha256 hex")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be lowercase sha256 hex") from exc
    return value


def _exact(value: Mapping[str, Any], fields: set[str], *, name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} fields are not canonical")


def _parameter_tag(parameters: ATCaaSProtocolParameters) -> str:
    return (
        f"M{parameters.M}-nr{parameters.n_r}-tr{parameters.t_r}-"
        f"na{parameters.n_a}-ta{parameters.t_a}-ma{parameters.m_a}-"
        f"h{parameters.quorum_h}-c{parameters.coin_c}-"
        f"tc{parameters.coin_t}-k{parameters.coin_k}"
    )


def _poly_mul(left: Sequence[int], right: Sequence[int]) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] = (result[i + j] + int(a) * int(b)) % FIELD_ORDER
    return result


def interpolate_polynomial(points: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    """Return canonical coefficients of the unique polynomial through points."""

    normalized = tuple((int(x) % FIELD_ORDER, int(y) % FIELD_ORDER) for x, y in points)
    if not normalized or len({x for x, _ in normalized}) != len(normalized):
        raise ValueError("interpolation points must have distinct x coordinates")
    result = [0] * len(normalized)
    for i, (x_i, y_i) in enumerate(normalized):
        basis = [1]
        denominator = 1
        for j, (x_j, _y_j) in enumerate(normalized):
            if i == j:
                continue
            basis = _poly_mul(basis, (-x_j % FIELD_ORDER, 1))
            denominator = denominator * (x_i - x_j) % FIELD_ORDER
        scale = y_i * pow(denominator, FIELD_ORDER - 2, FIELD_ORDER) % FIELD_ORDER
        for degree, coefficient in enumerate(basis):
            result[degree] = (result[degree] + scale * coefficient) % FIELD_ORDER
    return tuple(result)


@dataclass(frozen=True)
class YOSOADPSSWorkloadManifest:
    trial_id: str
    trial_epoch: str
    seed_digest: str
    parameters: ATCaaSProtocolParameters
    source_role_to_node: Mapping[int, int]
    successor_role_to_node: Mapping[int, int]
    source_evaluation_points: Mapping[int, int]
    successor_evaluation_points: Mapping[int, int]
    initial_secret_digest: str
    source_sharing_digest: str
    source_public_state_digest: str
    dag_config_digest: str
    coin_config_digest: str
    role_trace_digest: str
    network_profile_digest: str
    arm_order: tuple[str, ...] = ARM_ORDER

    def __post_init__(self) -> None:
        if not self.trial_id or not self.trial_epoch:
            raise ValueError("trial identity must be non-empty")
        for name in (
            "seed_digest",
            "initial_secret_digest",
            "source_sharing_digest",
            "source_public_state_digest",
            "dag_config_digest",
            "coin_config_digest",
            "role_trace_digest",
            "network_profile_digest",
        ):
            _digest(getattr(self, name), name=name)
        if tuple(self.arm_order) != normalize_protocols(self.arm_order):
            raise ValueError("arm order must use the canonical prepared protocol order")
        expected = set(range(self.parameters.n_r))
        for name in (
            "source_role_to_node",
            "successor_role_to_node",
            "source_evaluation_points",
            "successor_evaluation_points",
        ):
            mapping = {int(k): int(v) for k, v in getattr(self, name).items()}
            if set(mapping) != expected:
                raise ValueError(f"{name} must define every resharing role")
            object.__setattr__(self, name, mapping)

    @property
    def parameter_tag(self) -> str:
        return _parameter_tag(self.parameters)

    def to_obj(self) -> dict[str, Any]:
        return {
            "schema": WORKLOAD_SCHEMA,
            "trial_id": self.trial_id,
            "trial_epoch": self.trial_epoch,
            "seed_digest": self.seed_digest,
            "parameters": self.parameters.to_config_obj(),
            "parameter_tag": self.parameter_tag,
            "source_role_to_node": {str(k): v for k, v in sorted(self.source_role_to_node.items())},
            "successor_role_to_node": {str(k): v for k, v in sorted(self.successor_role_to_node.items())},
            "source_evaluation_points": {str(k): v for k, v in sorted(self.source_evaluation_points.items())},
            "successor_evaluation_points": {str(k): v for k, v in sorted(self.successor_evaluation_points.items())},
            "initial_secret_digest": self.initial_secret_digest,
            "source_sharing_digest": self.source_sharing_digest,
            "source_public_state_digest": self.source_public_state_digest,
            "dag_config_digest": self.dag_config_digest,
            "coin_config_digest": self.coin_config_digest,
            "role_trace_digest": self.role_trace_digest,
            "network_profile_digest": self.network_profile_digest,
            "arms": list(self.arm_order),
        }

    def digest(self) -> str:
        return domain_digest(_WORKLOAD_DOMAIN, self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "YOSOADPSSWorkloadManifest":
        fields = {
            "schema", "trial_id", "trial_epoch", "seed_digest", "parameters",
            "parameter_tag", "source_role_to_node", "successor_role_to_node",
            "source_evaluation_points", "successor_evaluation_points",
            "initial_secret_digest", "source_sharing_digest",
            "source_public_state_digest", "dag_config_digest",
            "coin_config_digest", "role_trace_digest", "network_profile_digest",
            "arms",
        }
        _exact(value, fields, name="YOSO ADPSS workload manifest")
        if value["schema"] != WORKLOAD_SCHEMA:
            raise ValueError("unsupported workload schema")
        parameters = ATCaaSProtocolParameters.from_mapping(
            {"M": value["parameters"]["M"], "atcaas_parameters": value["parameters"]}
        )
        result = cls(
            trial_id=str(value["trial_id"]), trial_epoch=str(value["trial_epoch"]),
            seed_digest=value["seed_digest"], parameters=parameters,
            source_role_to_node=value["source_role_to_node"],
            successor_role_to_node=value["successor_role_to_node"],
            source_evaluation_points=value["source_evaluation_points"],
            successor_evaluation_points=value["successor_evaluation_points"],
            initial_secret_digest=value["initial_secret_digest"],
            source_sharing_digest=value["source_sharing_digest"],
            source_public_state_digest=value["source_public_state_digest"],
            dag_config_digest=value["dag_config_digest"],
            coin_config_digest=value["coin_config_digest"],
            role_trace_digest=value["role_trace_digest"],
            network_profile_digest=value["network_profile_digest"],
            arm_order=tuple(value["arms"]),
        )
        if value["parameter_tag"] != result.parameter_tag:
            raise ValueError("workload parameter tag mismatch")
        return result


@dataclass(frozen=True)
class YOSOADPSSArmManifest:
    protocol: str
    common_workload_digest: str
    initial_secret_digest: str
    source_sharing_digest: str
    parameter_tag: str
    role_trace_digest: str
    protocol_manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.protocol not in ARM_ORDER:
            raise ValueError("unsupported YOSO ADPSS arm")
        for name in (
            "common_workload_digest", "initial_secret_digest",
            "source_sharing_digest", "role_trace_digest",
        ):
            _digest(getattr(self, name), name=name)
        if not isinstance(self.protocol_manifest, Mapping):
            raise TypeError("protocol_manifest must be public mapping")

    def to_obj(self) -> dict[str, Any]:
        return {
            "schema": ARM_SCHEMA,
            "protocol": self.protocol,
            "common_workload_digest": self.common_workload_digest,
            "initial_secret_digest": self.initial_secret_digest,
            "source_sharing_digest": self.source_sharing_digest,
            "parameter_tag": self.parameter_tag,
            "role_trace_digest": self.role_trace_digest,
            "protocol_manifest": dict(self.protocol_manifest),
        }

    def digest(self) -> str:
        return domain_digest(_ARM_DOMAIN, self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "YOSOADPSSArmManifest":
        _exact(value, {
            "schema", "protocol", "common_workload_digest",
            "initial_secret_digest", "source_sharing_digest", "parameter_tag",
            "role_trace_digest", "protocol_manifest",
        }, name="YOSO ADPSS arm manifest")
        if value["schema"] != ARM_SCHEMA:
            raise ValueError("unsupported arm schema")
        return cls(**{key: value[key] for key in value if key != "schema"})


@dataclass(frozen=True)
class YOSOADPSSNodePrivateShard:
    node_id: int
    common_workload_digest: str
    arm_manifest_digests: Mapping[str, str]
    a2dpss: Mapping[str, Any]
    ru22: Mapping[str, Any]
    bhk22: Mapping[str, Any] | None
    threshold_bls: Mapping[str, Any] | None

    def to_obj(self) -> dict[str, Any]:
        material = {
            "schema": PRIVATE_SHARD_SCHEMA,
            "node_id": self.node_id,
            "common_workload_digest": self.common_workload_digest,
            "arm_manifest_digests": dict(sorted(self.arm_manifest_digests.items())),
            "a2dpss": dict(self.a2dpss), "ru22": dict(self.ru22),
            "threshold_bls": dict(self.threshold_bls) if self.threshold_bls else None,
        }
        if "bhk22" in self.arm_manifest_digests:
            if self.bhk22 is None:
                raise ValueError("prepared BHK arm has no private shard")
            material["bhk22"] = dict(self.bhk22)
        elif self.bhk22 is not None:
            raise ValueError("unprepared BHK material in private shard")
        return {**material, "shard_digest": domain_digest(_SHARD_DOMAIN, material)}

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "YOSOADPSSNodePrivateShard":
        protocols = normalize_protocols(value.get("arm_manifest_digests", {}))
        fields = {
            "schema", "node_id", "common_workload_digest", "arm_manifest_digests",
            "a2dpss", "ru22", "threshold_bls", "shard_digest",
        }
        if "bhk22" in protocols:
            fields.add("bhk22")
        _exact(value, fields, name="YOSO ADPSS private shard")
        if value["schema"] != PRIVATE_SHARD_SCHEMA:
            raise ValueError("unsupported private shard schema")
        material = {key: value[key] for key in value if key != "shard_digest"}
        if domain_digest(_SHARD_DOMAIN, material) != value["shard_digest"]:
            raise ValueError("private shard digest mismatch")
        constructor = {key: value[key] for key in material if key != "schema"}
        constructor.setdefault("bhk22", None)
        return cls(**constructor)


@dataclass(frozen=True)
class YOSOADPSSFixtureBundle:
    manifest: YOSOADPSSWorkloadManifest
    arms: Mapping[str, YOSOADPSSArmManifest]
    shards: Mapping[int, YOSOADPSSNodePrivateShard]

    def validate(self) -> None:
        common = self.manifest.digest()
        if set(self.arms) != set(self.manifest.arm_order):
            raise ValueError("fixture arms do not match prepared protocols")
        bindings = {
            (
                arm.common_workload_digest, arm.initial_secret_digest,
                arm.source_sharing_digest, arm.parameter_tag,
                arm.role_trace_digest,
            )
            for arm in self.arms.values()
        }
        expected = {(
            common, self.manifest.initial_secret_digest,
            self.manifest.source_sharing_digest, self.manifest.parameter_tag,
            self.manifest.role_trace_digest,
        )}
        if bindings != expected:
            raise ValueError("arm manifests do not bind the common workload")
        expected_arm_digests = {name: arm.digest() for name, arm in self.arms.items()}
        if set(self.shards) != set(range(self.manifest.parameters.M)):
            raise ValueError("fixture must contain one shard per DAG node")
        for node, shard in self.shards.items():
            if shard.node_id != node or shard.common_workload_digest != common:
                raise ValueError("private shard/common workload mismatch")
            if dict(shard.arm_manifest_digests) != expected_arm_digests:
                raise ValueError("private shard/arm manifest mismatch")
            if (shard.bhk22 is not None) != ("bhk22" in self.arms):
                raise ValueError("private BHK material does not match prepared protocols")


def _scrub_ru22_fixture(bundle, public_seed_digest: str):
    manifest = replace(bundle.manifest, fixture_seed=public_seed_digest)
    shards = tuple(replace(shard, manifest_digest=manifest.digest()) for shard in bundle.shards)
    return manifest, shards


def _scrub_bhk22_fixture(bundle, public_seed_digest: str):
    manifest = replace(bundle.manifest, fixture_seed=public_seed_digest)
    shards = {
        node: replace(shard, manifest_digest=manifest.digest())
        for node, shard in bundle.shards.items()
    }
    return manifest, shards


def _select_exact_bhk22_vrf_committee(
    *,
    configs: Sequence[Mapping[str, Any]],
    g_hex: str,
    n_r: int,
    trial_id: str,
    trial_epoch: int,
    source_epoch: int,
    sid: str,
    transfer_seq: int,
    role_trace_digest: str,
    role_type: str,
    max_attempts: int = 4096,
) -> tuple[dict[int, int], int]:
    """Generate a real VRF committee, retrying the bound label for exact n_r."""

    if not vrf_backend_available():
        raise RuntimeError("formal BHK22 committee selection requires the VRF backend")
    M = len(configs)
    if not 1 <= n_r <= M:
        raise ValueError("BHK22 committee size must be within the population")
    extras = [dict(config.get("extra") or {}) for config in configs]
    vrf_pks = extras[0].get("vrf_pks")
    if not isinstance(vrf_pks, list) or len(vrf_pks) != M:
        raise ValueError("BHK22 sortition requires M shared VRF public keys")
    if any(extra.get("vrf_pks") != vrf_pks for extra in extras[1:]):
        raise ValueError("node configs disagree on BHK22 VRF public keys")
    vrf_sks = [extra.get("vrf_sk") for extra in extras]
    if any(secret is None for secret in vrf_sks):
        raise ValueError("every BHK22 candidate requires a local VRF secret key")
    threshold_fraction = n_r / M
    for attempt in range(max_attempts):
        label = bhk22_committee_sortition_label(
            trial_id=trial_id,
            trial_epoch=trial_epoch,
            source_epoch=source_epoch,
            target_epoch=source_epoch + 1,
            sid=sid,
            transfer_seq=transfer_seq,
            role_trace_digest=role_trace_digest,
            role_type=role_type,
            selection_attempt=attempt,
        )
        selected = []
        for node_id, vrf_sk in enumerate(vrf_sks):
            is_selected, proof = vrf_prove_sortition(
                label=label,
                vrf_sk=vrf_sk,
                g_hex=g_hex,
                threshold_fraction=threshold_fraction,
            )
            if is_selected:
                if not vrf_verify_sortition(
                    label=label,
                    proof=proof or {},
                    sender_id=node_id,
                    vrf_pks=vrf_pks,
                    g_hex=g_hex,
                    threshold_fraction=threshold_fraction,
                ):
                    raise ValueError("generated BHK22 committee VRF proof is invalid")
                selected.append(node_id)
        if len(selected) == n_r:
            return {role_id: node for role_id, node in enumerate(selected)}, attempt
    raise RuntimeError("could not sample an exact-size BHK22 VRF committee")


def build_yoso_adpss_fixture(
    *,
    config_directory: str | Path,
    trial_id: str,
    trial_epoch: str,
    seed: str,
    role_trace_digest: str,
    network_profile: Mapping[str, Any] | None = None,
    bhk22_path_mode: str = "optimistic",
    protocols: Sequence[str] = ARM_ORDER,
) -> YOSOADPSSFixtureBundle:
    """Derive selected public manifests and node-private arm shards."""

    protocols = normalize_protocols(protocols)
    config_directory = Path(config_directory)
    configs = []
    for node in range(1_000_000):
        path = config_directory / f"local.{node}.json"
        if not path.is_file():
            break
        configs.append(json.loads(path.read_text(encoding="utf-8")))
    if not configs:
        raise ValueError("configuration directory contains no local configs")
    parameters = ATCaaSProtocolParameters.from_mapping(configs[0])
    if len(configs) != parameters.M:
        raise ValueError("configuration directory must contain exactly M nodes")
    if any(ATCaaSProtocolParameters.from_mapping(item) != parameters for item in configs):
        raise ValueError("node configs disagree on ATCaaS parameters")
    _digest(role_trace_digest, name="role_trace_digest")
    seed_digest = domain_digest(b"ATCaaS/YOSO-ADPSS/seed/v1\x00", seed)
    g_hex, h_hex = g_h_from_config_extra(configs[0]["extra"])
    locals_ = [
        build_handover_local_config(
            config["extra"], g_hex=g_hex, h_hex=h_hex, my_id=node,
            M=parameters.M, service_t=parameters.t_r, epoch=0,
        )
        for node, config in enumerate(configs)
    ]
    legacy = locals_[0].ctx
    for local in locals_[1:]:
        if local.ctx != legacy:
            raise ValueError("node configs disagree on the public source context")
    shares = {
        node: tuple(local.my_holder_share)
        for node, local in enumerate(locals_)
        if local.my_holder_share is not None
    }
    source_roles = tuple(sorted(legacy.holder_eval_points))
    if set(shares) != set(source_roles):
        raise ValueError("every source role must have exactly one node-local share")
    interpolation_roles = source_roles[: parameters.t_r + 1]
    secret_poly = interpolate_polynomial(tuple(
        (legacy.holder_eval_points[node], shares[node][0]) for node in interpolation_roles
    ))
    blind_poly = interpolate_polynomial(tuple(
        (legacy.holder_eval_points[node], shares[node][1]) for node in interpolation_roles
    ))
    if any(
        eval_poly(secret_poly, legacy.holder_eval_points[node]) != shares[node][0]
        or eval_poly(blind_poly, legacy.holder_eval_points[node]) != shares[node][1]
        for node in source_roles
    ):
        raise ValueError("source shares are not degree-t_r Shamir sharings")
    secret, blind = secret_poly[0], blind_poly[0]
    source_role_to_node = {role: role for role in source_roles}
    successor_role_to_node = {
        role: node for node, role in legacy.successor_role_indices.items()
    }
    source_evaluation_points = dict(legacy.holder_eval_points)
    successor_evaluation_points = {role: role + 1 for role in successor_role_to_node}
    initial_secret_digest = domain_digest(
        b"ATCaaS/YOSO-ADPSS/initial-secret/v1\x00",
        {"secret": str(secret), "private_binding_salt": seed},
    )
    source_sharing_digest = domain_digest(
        b"ATCaaS/YOSO-ADPSS/source-sharing/v1\x00",
        {
            "shares": {str(node): [str(v) for v in shares[node]] for node in source_roles},
            "evaluation_points": {str(k): v for k, v in sorted(source_evaluation_points.items())},
            "old_commitment_vector": list(legacy.old_commitment_vector),
            "private_binding_salt": seed,
        },
    )
    source_public_state_digest = domain_digest(
        b"ATCaaS/YOSO-ADPSS/source-public/v1\x00",
        {"old_commitment_vector": list(legacy.old_commitment_vector)},
    )
    dag_config_digest = domain_digest(
        b"ATCaaS/YOSO-ADPSS/dag-config/v1\x00", parameters.to_config_obj()
    )
    coin_public = {
        key: configs[0]["extra"][key]
        for key in (
            "coin_config_version", "coin_committee", "coin_threshold",
            "coin_public_key_digest", "coin_membership_digest",
        )
    }
    for config in configs[1:]:
        if any(config["extra"].get(key) != value for key, value in coin_public.items()):
            raise ValueError("node configs disagree on threshold-BLS public config")
    coin_config_digest = domain_digest(
        b"ATCaaS/YOSO-ADPSS/threshold-bls-config/v1\x00", coin_public
    )
    network_profile_digest = domain_digest(
        b"ATCaaS/YOSO-ADPSS/network-profile/v1\x00", dict(network_profile or {})
    )
    manifest = YOSOADPSSWorkloadManifest(
        trial_id=trial_id, trial_epoch=str(trial_epoch), seed_digest=seed_digest,
        parameters=parameters, source_role_to_node=source_role_to_node,
        successor_role_to_node=successor_role_to_node,
        source_evaluation_points=source_evaluation_points,
        successor_evaluation_points=successor_evaluation_points,
        initial_secret_digest=initial_secret_digest,
        source_sharing_digest=source_sharing_digest,
        source_public_state_digest=source_public_state_digest,
        dag_config_digest=dag_config_digest,
        coin_config_digest=coin_config_digest,
        role_trace_digest=role_trace_digest,
        network_profile_digest=network_profile_digest,
        arm_order=protocols,
    )
    common_digest = manifest.digest()
    a2_context = A2DPSSContext(
        epoch=legacy.epoch,
        sid=str(legacy.sid).replace("dpss", "a2dpss"),
        source_committee=legacy.source_committee,
        successor_committee=legacy.successor_committee,
        transfer_seq=legacy.transfer_seq, g_hex=legacy.g_hex, h_hex=legacy.h_hex,
        key_base=legacy.key_base, successor_rpk_list=legacy.successor_rpk_list,
        service_t=legacy.service_t,
        old_commitment_vector=legacy.old_commitment_vector,
        holder_eval_points=legacy.holder_eval_points,
        successor_role_indices=legacy.successor_role_indices,
    )
    ru_raw = build_worst_case_multi_vps_fixture(
        config_path=config_directory / "local.0.json", trial_id=trial_id,
        trial_epoch=str(trial_epoch), seed=f"{seed}:ru22", n_r=parameters.n_r,
        t_r=parameters.t_r, dag_m=parameters.M, dag_t=parameters.t_a,
        maintained_secret=secret, maintained_blind=blind,
        source_polynomial_coefficients=secret_poly,
        blind_polynomial_coefficients=blind_poly,
    )
    ru_manifest, ru_shards = _scrub_ru22_fixture(ru_raw, seed_digest)
    bhk_manifest, bhk_shards = None, {}
    if "bhk22" in protocols:
        bhk_source_epoch = max(1, int(legacy.epoch))
        bhk_sid = f"yoso-adpss-{trial_id}"
        verifier_nodes, verifier_attempt = _select_exact_bhk22_vrf_committee(
            configs=configs,
            g_hex=g_hex,
            n_r=parameters.n_r,
            trial_id=trial_id,
            trial_epoch=int(trial_epoch),
            source_epoch=bhk_source_epoch,
            sid=bhk_sid,
            transfer_seq=0,
            role_trace_digest=role_trace_digest,
            role_type=ROLE_VERIFIER,
        )
        responder_nodes, responder_attempt = _select_exact_bhk22_vrf_committee(
            configs=configs,
            g_hex=g_hex,
            n_r=parameters.n_r,
            trial_id=trial_id,
            trial_epoch=int(trial_epoch),
            source_epoch=bhk_source_epoch,
            sid=bhk_sid,
            transfer_seq=0,
            role_trace_digest=role_trace_digest,
            role_type=ROLE_RESPONDER,
        )
        role_to_node = {
            ROLE_DEALER: dict(source_role_to_node),
            ROLE_SHAREHOLDER: dict(successor_role_to_node),
            ROLE_VERIFIER: verifier_nodes,
            ROLE_RESPONDER: responder_nodes,
        }
        committee_selection = {
            ROLE_DEALER: {
                "selection_mode": COMMITTEE_SELECTION_EXTERNAL,
                "selection_attempt": 0,
            },
            ROLE_SHAREHOLDER: {
                "selection_mode": COMMITTEE_SELECTION_EXTERNAL,
                "selection_attempt": 0,
            },
            ROLE_VERIFIER: {
                "selection_mode": COMMITTEE_SELECTION_VRF_EXACT,
                "selection_attempt": verifier_attempt,
            },
            ROLE_RESPONDER: {
                "selection_mode": COMMITTEE_SELECTION_VRF_EXACT,
                "selection_attempt": responder_attempt,
            },
        }
        bhk_raw = build_bhk22_fixture(
            seed=f"{seed}:bhk22", trial_id=trial_id, trial_epoch=int(trial_epoch),
            source_epoch=bhk_source_epoch, sid=bhk_sid,
            source_committee=legacy.source_committee,
            successor_committee=legacy.successor_committee,
            parameters=parameters, path_mode=bhk22_path_mode,
            completion_mode=DEFAULT_COMPLETION_MODE, role_to_node=role_to_node,
            committee_selection=committee_selection,
            source_secret=secret, source_polynomial_coefficients=secret_poly,
            dag_config_digest=bhk22_dag_config_digest(parameters),
            coin_config_digest=bhk22_coin_config_digest(configs[0]["extra"]),
            role_trace_digest=role_trace_digest,
            common_workload_digest=common_digest,
        )
        bhk_manifest, bhk_shards = _scrub_bhk22_fixture(bhk_raw, seed_digest)
    public_protocol_manifests = {
        "a2dpss": {"context": a2_context.to_obj()},
        "ru22": ru_manifest.to_obj(),
    }
    if bhk_manifest is not None:
        public_protocol_manifests["bhk22"] = bhk_manifest.to_obj()
    arms = {
        protocol: YOSOADPSSArmManifest(
            protocol=protocol, common_workload_digest=common_digest,
            initial_secret_digest=initial_secret_digest,
            source_sharing_digest=source_sharing_digest,
            parameter_tag=manifest.parameter_tag,
            role_trace_digest=role_trace_digest,
            protocol_manifest=public_protocol_manifests[protocol],
        )
        for protocol in protocols
    }
    arm_digests = {name: arm.digest() for name, arm in arms.items()}
    shards = {}
    for node in range(parameters.M):
        holder = shares.get(node)
        a2_private = {
            "context_digest": a2_context.context_digest,
            "holder": (
                {"holder_id": node, "share_s": str(holder[0]), "share_s_hat": str(holder[1])}
                if holder is not None else None
            ),
            "successor": (
                {"recipient_secret_key": str(locals_[node].my_successor_rsk)}
                if locals_[node].my_successor_rsk is not None else None
            ),
        }
        extra = configs[node]["extra"]
        threshold_bls = None
        if "coin_private_key" in extra:
            threshold_bls = {
                "private_key": extra["coin_private_key"],
                "private_share_index": extra["coin_private_share_index"],
            }
        shards[node] = YOSOADPSSNodePrivateShard(
            node_id=node, common_workload_digest=common_digest,
            arm_manifest_digests=arm_digests, a2dpss=a2_private,
            ru22=ru_shards[node].to_obj(),
            bhk22=bhk_shards[node].to_obj() if "bhk22" in protocols else None,
            threshold_bls=threshold_bls,
        )
    bundle = YOSOADPSSFixtureBundle(manifest=manifest, arms=arms, shards=shards)
    bundle.validate()
    return bundle


def write_yoso_adpss_fixture(bundle: YOSOADPSSFixtureBundle, output: str | Path) -> None:
    bundle.validate()
    output = Path(output)
    (output / "arms").mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_bytes(canonical_json(bundle.manifest.to_obj()) + b"\n")
    for name, arm in bundle.arms.items():
        (output / "arms" / f"{name}.json").write_bytes(canonical_json(arm.to_obj()) + b"\n")
    for node, shard in bundle.shards.items():
        (output / f"node.{node}.private.json").write_bytes(canonical_json(shard.to_obj()) + b"\n")


def load_yoso_adpss_public_fixture(output: str | Path):
    output = Path(output)
    manifest = YOSOADPSSWorkloadManifest.from_obj(json.loads((output / "manifest.json").read_text()))
    arms = {
        name: YOSOADPSSArmManifest.from_obj(json.loads((output / "arms" / f"{name}.json").read_text()))
        for name in manifest.arm_order
    }
    return manifest, arms


def verify_same_initial_workload(
    manifest: YOSOADPSSWorkloadManifest,
    arms: Mapping[str, YOSOADPSSArmManifest],
) -> bool:
    """Public analyzer check; it deliberately has no private-shard argument."""

    common = manifest.digest()
    expected = (
        common, manifest.initial_secret_digest, manifest.source_sharing_digest,
        manifest.parameter_tag, manifest.role_trace_digest,
    )
    return set(arms) == set(manifest.arm_order) and all(
        (
            arm.common_workload_digest, arm.initial_secret_digest,
            arm.source_sharing_digest, arm.parameter_tag, arm.role_trace_digest,
        ) == expected
        for arm in arms.values()
    )


__all__ = [
    "ARM_ORDER", "DUAL_ARM_ORDER", "ARM_SCHEMA", "PRIVATE_SHARD_SCHEMA", "WORKLOAD_SCHEMA",
    "YOSOADPSSArmManifest", "YOSOADPSSFixtureBundle",
    "YOSOADPSSNodePrivateShard", "YOSOADPSSWorkloadManifest",
    "build_yoso_adpss_fixture", "interpolate_polynomial",
    "load_yoso_adpss_public_fixture", "normalize_protocols", "verify_same_initial_workload",
    "write_yoso_adpss_fixture",
]
