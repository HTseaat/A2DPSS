"""Node-isolated native RU22 fixtures, separate from the legacy DAG arm schema."""
import json
from pathlib import Path

from beaver.baselines.ru22_certificate import canonical, digest
from beaver.baselines.ru22_experiment import RU22DAGExperimentManifest, RU22DAGLocalShard
from beaver.baselines.ru22_native_roles import build_roles


PUBLIC_FILE = "ru22-native.public.json"
BROADCAST_DELIVERY = "full-population-broadcast"


def private_file(node):
    return f"ru22-native.node.{node}.private.json"


def write_json(path, obj, *, private=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Private bytes must never be briefly exposed with default file permissions.
    import os
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
    with os.fdopen(fd, "wb") as handle:
        handle.write(canonical(obj) + b"\n")


def write_native_fixture(directory, base_manifest, base_shards, *, common_workload_digest,
                         arm_manifest_digest=None, operation_context=None, operation_trial_id=None):
    directory = Path(directory)
    c = base_manifest.context
    roles, identities = build_roles(c, base_manifest.dag_m)
    public = dict(schema="ru22-native-experiment/v2", base=base_manifest.to_obj(), roles=roles,
                  protocol_delivery=BROADCAST_DELIVERY,
                  common_workload_digest=common_workload_digest,
                  arm_manifest_digest=arm_manifest_digest,
                  verifier_send_scope="once-per-dedicated-verifier-role",
                  verifier_committee_count=c.resharing_n,
                  verifier_role_count=c.resharing_n ** 2,
                  strict_refresh_wide_yoso=True,
                  yoso_scope="logical-role; physical machines may host multiple roles",
                  ciphertext_mode="chunk-pvss-forward-selected-pvrs")
    if operation_context is not None:
        if not isinstance(operation_trial_id, str) or not operation_trial_id:
            raise ValueError("native operation fixture requires a trial identity")
        public["operation_context"] = operation_context.to_obj()
        public["operation_trial_id"] = operation_trial_id
    public["digest"] = digest(public)
    write_json(directory / PUBLIC_FILE, public)
    for shard in base_shards:
        write_json(directory / private_file(shard.node_id), dict(
            schema="ru22-native-private/v1", native_manifest_digest=public["digest"],
            node_id=shard.node_id, base=shard.to_obj(), identity=identities[shard.node_id]), private=True)
    return public


def write_common_native_fixture(bundle, directory):
    base = RU22DAGExperimentManifest.from_obj(bundle.arms["ru22"].protocol_manifest)
    shards = tuple(RU22DAGLocalShard.from_obj(bundle.shards[i].ru22, base)
                   for i in range(bundle.manifest.parameters.M))
    return write_native_fixture(directory, base, shards,
        common_workload_digest=bundle.manifest.digest(), arm_manifest_digest=bundle.arms["ru22"].digest())


def load_native_public(directory):
    directory = Path(directory)
    public = json.loads((directory / PUBLIC_FILE).read_text())
    material = {k: v for k, v in public.items() if k != "digest"}
    if public.get("schema") != "ru22-native-experiment/v2" or digest(material) != public.get("digest"):
        raise ValueError("native public fixture digest mismatch")
    base = RU22DAGExperimentManifest.from_obj(public["base"])
    if public.get("protocol_delivery", BROADCAST_DELIVERY) != BROADCAST_DELIVERY:
        raise ValueError("invalid native protocol delivery mode")
    if (public.get("verifier_send_scope") != "once-per-dedicated-verifier-role"
            or public.get("strict_refresh_wide_yoso") is not True
            or public.get("verifier_committee_count") != base.context.resharing_n
            or public.get("verifier_role_count") != base.context.resharing_n ** 2):
        raise ValueError("invalid native dedicated-verifier semantics")
    from beaver.baselines.ru22_native_roles import public_authority
    public_authority(base.context, public["roles"])
    if "operation_context" in public:
        from beaver.threshold_operation_contract import ThresholdOperationContext
        ThresholdOperationContext.from_obj(public["operation_context"])
        if not isinstance(public.get("operation_trial_id"), str) or not public["operation_trial_id"]:
            raise ValueError("native operation trial missing")
    return public, base


def write_service_native_fixture(bundle, directory):
    base = RU22DAGExperimentManifest.from_obj(bundle.arms["ru22"].protocol_manifest["base_manifest"])
    shards = tuple(RU22DAGLocalShard.from_obj(bundle.shards[i].ru22["base_shard"], base)
                   for i in range(base.dag_m))
    return write_native_fixture(directory, base, shards,
        common_workload_digest=bundle.manifest.workload_digest,
        arm_manifest_digest=bundle.arms["ru22"].arm_digest,
        operation_context=bundle.manifest.operation_context,
        operation_trial_id=bundle.manifest.base_workload.trial_id)


def load_native_fixture(directory, node):
    public, base = load_native_public(directory)
    private = json.loads((Path(directory) / private_file(node)).read_text())
    if (private.get("schema") != "ru22-native-private/v1"
            or private.get("native_manifest_digest") != public["digest"]
            or type(private.get("node_id")) is not int or private["node_id"] != node):
        raise ValueError("native fixture identity/digest mismatch")
    shard = RU22DAGLocalShard.from_obj(private["base"], base)
    shard.validate(base)
    if shard.node_id != node:
        raise ValueError("wrong node private shard")
    return public, private, base, shard
