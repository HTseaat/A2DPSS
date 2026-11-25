"""Reuse the thesis RU22 preparation without generating cryptographic material.

``rebind_ru22_bundle(bundle: Path, peers: list[str]) -> dict`` mutates an already
copied, pristine *full master* produced by ``run_yoso_adpss_dist prepare`` with
``--protocols a2dpss ru22 --ru22-mode native`` and returns its updated settings.
Required directories are configs/, fixture/, nodes/0..M-1/, code/, and public/.
Call before pruning other nodes. Copy the master first; never pass the sealed
template itself. JSON replacement breaks hard links rather than altering their
other aliases. Lookup tables and code are never rewritten.

Only transport-bound outer digests change. Epoch, role trace, native authority
context, committee membership, VRF/signing keys, VPS, and private shares retain
their exact values. No generation, role selection, or protocol execution occurs.
The caller must write its authoritative cluster.json (SSH placement cannot be
inferred from endpoint strings alone). Historical BHK triplets, threshold-service
fixtures, partially pruned host bundles, and previously run bundles are rejected.
Validation completes before the first write; an I/O failure during replacement
requires discarding the working copy and cloning the template again.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def _read(path):
    return json.loads(Path(path).read_text())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _file_digest(path, cache):
    stat = path.stat()
    key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if key not in cache:
        value = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(chunk)
        cache[key] = value.hexdigest()
    return cache[key]


def _file(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("bundle file path must remain inside its directory")
    path = root / relative
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
        raise ValueError("bundle requires regular files without symlinks: " + str(path))
    return path


def _write(path, value):
    """Replace private JSON atomically, including read-only/hardlinked clones."""
    descriptor, temporary = tempfile.mkstemp(prefix=".rebind-", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_base_config_directory(directory, parameters):
    """Read-only preflight for an existing complete Phase-6 setup.

    Reject parameter aliases that disagree with the canonical schema, wrong node
    identities, inconsistent public/coin context, and invalid lookup artifacts.
    Returns the parsed node configs, in node order; generates nothing.
    """
    from beaver.atcaas_params import ATCaaSProtocolParameters
    from beaver.chunk_elgamal import (
        LOOKUP_TABLE_SET_SCHEMA, _normalize_point_hex, load_lookup_table_artifact)
    from beaver.handover_fixtures import build_handover_local_config
    from beaver.yoso_adpss_workload import g_h_from_config_extra

    directory = Path(directory).resolve()
    expected = {f"local.{node}.json" for node in range(parameters.M)}
    if {p.name for p in directory.glob("local.*.json")} != expected:
        raise ValueError("reused config directory must contain exactly M node configs")
    if not (directory / "lookup_tables").is_dir() or any(
            p.is_symlink() for p in (directory / "lookup_tables", *(directory / "lookup_tables").rglob("*"))):
        raise ValueError("reused setup requires regular lookup files without symlinks")
    if (directory / "coin_manifest.json").exists():
        _file(directory, "coin_manifest.json")
    configs = []
    public_context = None
    public_extra = None
    lookup = None
    aliases = dict(N=parameters.M, M=parameters.M, n=parameters.n_a,
                   expected_committee_size=parameters.n_a, committee_m=parameters.m_a,
                   t=parameters.agreement_t_impl, **{
                       k: v for k, v in parameters.to_config_obj().items()
                       if k not in ("schema", "M", "agreement_t_impl")})
    for node in range(parameters.M):
        config = _read(_file(directory, f"local.{node}.json"))
        if ATCaaSProtocolParameters.from_mapping(config) != parameters:
            raise ValueError(f"reused node {node} ATCaaS parameters differ")
        if type(config.get("my_id")) is not int or config["my_id"] != node:
            raise ValueError(f"reused node {node} identity differs")
        if any(type(config.get(k)) is not int or config[k] != v for k, v in aliases.items()):
            raise ValueError(f"reused node {node} parameter aliases differ")
        extra = config["extra"]
        dpss = extra.get("dpss", {})
        if (dpss.get("version", 0) < 2 or dpss.get("private_material_scope") != "node-local"
                or dpss.get("n_r") != parameters.n_r or dpss.get("t_r") != parameters.t_r
                or dpss.get("service_t") != parameters.t_r):
            raise ValueError("reused setup requires matching formal node-local DPSS parameters")
        if (len(extra.get("coin_committee", [])) != parameters.coin_c
                or extra.get("coin_threshold") != parameters.coin_k):
            raise ValueError("reused setup coin parameters differ")
        current_public = {key: extra[key] for key in (
            "SRS", "vrf_pks", "coin_config_version", "coin_committee", "coin_threshold",
            "coin_public_key_digest", "coin_membership_digest")}
        if node and current_public != public_extra:
            raise ValueError("reused node configs disagree on public keys/coin configuration")
        public_extra = current_public
        g, h = g_h_from_config_extra(extra)
        local = build_handover_local_config(extra, g_hex=g, h_hex=h, my_id=node,
                                            M=parameters.M, service_t=parameters.t_r, epoch=0)
        if node and local.ctx != public_context:
            raise ValueError("reused node configs disagree on the public source context")
        if (len(local.ctx.holder_eval_points) != parameters.n_r
                or len(local.ctx.successor_role_indices) != parameters.n_r):
            raise ValueError("reused setup resharing committees differ")
        public_context = local.ctx
        current_lookup = extra.get("chunk_lookup_tables")
        if (not isinstance(current_lookup, dict)
                or set(current_lookup) != {"schema", "offline_generation_seconds", "tables"}
                or current_lookup["schema"] != LOOKUP_TABLE_SET_SCHEMA
                or set(current_lookup.get("tables", {})) != {"g", "h"}):
            raise ValueError("reused setup requires complete g/h lookup artifacts")
        if node and current_lookup != lookup:
            raise ValueError("reused node configs disagree on lookup artifacts")
        lookup = current_lookup
        configs.append(config)
    for label, base in (("g", g), ("h", h)):
        artifact = lookup["tables"][label]
        if artifact["artifact_path"] != f"lookup_tables/{label}.lookup":
            raise ValueError("reused setup requires standard lookup_tables/g.lookup and h.lookup paths")
        _file(directory, artifact["artifact_path"])
        normalized, table = load_lookup_table_artifact(artifact, base_directory=directory)
        try:
            if normalized != _normalize_point_hex(base):
                raise ValueError("reused lookup base differs from configuration SRS")
        finally:
            table.close()
    return configs


def _load_bundle(bundle):
    from beaver.atcaas_params import ATCaaSProtocolParameters
    from beaver.baselines.ru22_native_experiment import load_native_public, private_file, PUBLIC_FILE
    from beaver.baselines.ru22_native_roles import NativeAuthentication
    from beaver.protocol_trace import CommitteeRoleTraceArtifact
    from beaver.yoso_adpss_workload import (
        YOSOADPSSFixtureBundle, YOSOADPSSNodePrivateShard,
        domain_digest, load_yoso_adpss_public_fixture)

    bundle = Path(bundle).resolve()
    for name in ("results", "output", "staged", "supervisor.json", ".stage-complete"):
        if (bundle / name).exists():
            raise ValueError("RU22 rebind requires a pristine full master: " + name)
    for root in (bundle / "nodes", bundle / "configs", bundle / "fixture"):
        if any(root.rglob("*.used")):
            raise ValueError("RU22 rebind refuses consumed node materials")
    settings = _read(_file(bundle, "public/experiment.json"))
    if (settings.get("schema") != "yoso-adpss-distributed/v1"
            or settings.get("digest") != _digest({k: v for k, v in settings.items() if k != "digest"})):
        raise ValueError("invalid prepared RU22 settings digest")
    if settings.get("ru22_mode") != "native" or settings.get("protocols") != ["a2dpss", "ru22"]:
        raise ValueError("rebind supports only native thesis dual-arm RU22 masters")
    parameters = ATCaaSProtocolParameters.from_mapping(
        {"M": settings["parameters"]["M"], "atcaas_parameters": settings["parameters"]})
    M = parameters.M
    if {p.name for p in (bundle / "nodes").iterdir()} != {str(n) for n in range(M)}:
        raise ValueError("rebind requires all M node packages before pruning")
    if set(settings["node_hashes"]) != {str(n) for n in range(M)}:
        raise ValueError("prepared settings must hash every node package")
    cache = {}
    for name, expected in settings["source_hashes"].items():
        if _file_digest(_file(bundle / "code", name), cache) != expected:
            raise ValueError("RU22 source checksum mismatch: " + name)
    for node in range(M):
        root = bundle / "nodes" / str(node)
        listed = settings["node_hashes"][str(node)]
        actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
        if actual != set(listed):
            raise ValueError("node package contents differ from manifest")
        for name, expected in listed.items():
            if _file_digest(_file(root, name), cache) != expected:
                raise ValueError(f"RU22 node {node} checksum mismatch: {name}")
    fixture = bundle / "fixture"
    manifest, arms = load_yoso_adpss_public_fixture(fixture)
    shards = {n: YOSOADPSSNodePrivateShard.from_obj(_read(
        _file(fixture, f"node.{n}.private.json"))) for n in range(M)}
    complete = YOSOADPSSFixtureBundle(manifest, arms, shards)
    complete.validate()
    if (manifest.parameters != parameters or manifest.digest() != settings["common_workload_digest"]
            or manifest.trial_epoch != settings["epoch"] or manifest.trial_id != settings["trial_id"]
            or manifest.arm_order != ("a2dpss", "ru22")
            or manifest.role_trace_digest != settings["role_trace_digest"]):
        raise ValueError("RU22 settings/common fixture bindings differ")
    if CommitteeRoleTraceArtifact.from_file(bundle / "public/role-trace.json").digest() != settings["role_trace_digest"]:
        raise ValueError("RU22 role trace digest mismatch")
    profile = {"kind": "distributed", "peers": settings["peers"]}
    if domain_digest(b"ATCaaS/YOSO-ADPSS/network-profile/v1\x00", profile) != manifest.network_profile_digest:
        raise ValueError("unsupported or inconsistent RU22 network profile")
    public, base = load_native_public(fixture)
    if (public["common_workload_digest"] != manifest.digest()
            or public["arm_manifest_digest"] != arms["ru22"].digest()
            or public["base"] != arms["ru22"].protocol_manifest
            or base.dag_m != M or base.context.resharing_n != parameters.n_r
            or public["roles"]["population"] != M):
        raise ValueError("native RU22 public fixture bindings differ")
    native_private, configs = {}, []
    for node in range(M):
        config = _read(_file(bundle / "configs", f"local.{node}.json"))
        if (ATCaaSProtocolParameters.from_mapping(config) != parameters
                or type(config.get("my_id")) is not int or config["my_id"] != node
                or config["peers"] != settings["peers"]):
            raise ValueError("RU22 config identity/parameters/peers differ")
        configs.append(config)
        private = _read(_file(fixture, private_file(node)))
        if (private.get("schema") != "ru22-native-private/v1"
                or private.get("native_manifest_digest") != public["digest"]
                or type(private.get("node_id")) is not int or private["node_id"] != node
                or private["base"] != shards[node].ru22):
            raise ValueError("RU22 native private wrapper bindings differ")
        inner = private["base"]
        if (inner.get("kind") != "atcaas-ru22-dag-local-shard" or inner.get("version") != 1
                or inner.get("node_id") != node or inner.get("manifest_digest") != base.digest()
                or inner.get("shard_digest") != domain_digest(b"ATCaaS/RU22/dag-local-shard/v1\x00",
                    {k: v for k, v in inner.items() if k != "shard_digest"})):
            raise ValueError("RU22 native base shard digest/binding differs")
        native_private[node] = private
        # Validate every physical node's complete set of independent logical
        # verifier/leader/source role seeds without generating a VRF proof.
        authentication = NativeAuthentication(base.context, public["roles"], node, private["identity"])
        authentication.close()
        node_root = bundle / "nodes" / str(node)
        if _read(node_root / "config.json") != config:
            raise ValueError("RU22 master/node configs differ")
        for name in ("manifest.json", PUBLIC_FILE, private_file(node), f"node.{node}.private.json",
                     "arms/a2dpss.json", "arms/ru22.json"):
            if _file_digest(_file(fixture, name), cache) != _file_digest(_file(node_root / "fixture", name), cache):
                raise ValueError("RU22 master/node fixture copies differ: " + name)
    preparation = _read(_file(bundle, "preparation.json"))
    if preparation != {"status": "ready", "digest": settings["digest"]}:
        raise ValueError("RU22 preparation marker differs")
    return settings, complete, public, native_private, configs


def validate_ru22_bundle(bundle: Path) -> dict:
    """Validate a complete pristine native dual-arm master; return its settings."""
    return _load_bundle(bundle)[0]


def rebind_ru22_bundle(bundle: Path, peers: list[str]) -> dict:
    """Rebind transport metadata in a cloned full RU22 master (see module docs)."""
    from beaver.baselines.ru22_native_experiment import PUBLIC_FILE, private_file
    from beaver.yoso_adpss_workload import domain_digest

    bundle = Path(bundle).resolve()
    settings, old, native, native_private, configs = _load_bundle(bundle)
    M = settings["parameters"]["M"]
    if (not isinstance(peers, list) or len(peers) != M
            or any(not isinstance(p, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+:[0-9]+", p)
                   or not 1024 <= int(p.rsplit(":", 1)[1]) <= 65535 for p in peers)
            or len(set(peers)) != M):
        raise ValueError("peers must contain M distinct hostname/IPv4:port endpoints")
    manifest = replace(old.manifest, network_profile_digest=domain_digest(
        b"ATCaaS/YOSO-ADPSS/network-profile/v1\x00", {"kind": "distributed", "peers": peers}))
    common = manifest.digest()
    arms = {name: replace(arm, common_workload_digest=common) for name, arm in old.arms.items()}
    arm_digests = {name: arm.digest() for name, arm in arms.items()}
    native = {**native, "common_workload_digest": common, "arm_manifest_digest": arm_digests["ru22"]}
    native["digest"] = _digest({k: v for k, v in native.items() if k != "digest"})
    documents = {"manifest.json": manifest.to_obj(), PUBLIC_FILE: native}
    documents.update({f"arms/{name}.json": arm.to_obj() for name, arm in arms.items()})
    for node in range(M):
        documents[f"node.{node}.private.json"] = replace(
            old.shards[node], common_workload_digest=common, arm_manifest_digests=arm_digests).to_obj()
        documents[private_file(node)] = {**native_private[node], "native_manifest_digest": native["digest"]}
    # All semantic checks precede mutations. The caller owns writable clone dirs.
    for name, value in documents.items():
        _write(bundle / "fixture" / name, value)
    cache = {}
    for node, config in enumerate(configs):
        config = {**config, "peers": peers}
        _write(bundle / "configs" / f"local.{node}.json", config)
        root = bundle / "nodes" / str(node)
        _write(root / "config.json", config)
        for name in ("manifest.json", PUBLIC_FILE, private_file(node), f"node.{node}.private.json",
                     "arms/a2dpss.json", "arms/ru22.json"):
            _write(root / "fixture" / name, documents[name])
        settings["node_hashes"][str(node)] = {
            str(p.relative_to(root)): _file_digest(p, cache) for p in root.rglob("*") if p.is_file()}
    settings["peers"] = peers
    settings["common_workload_digest"] = common
    settings["digest"] = _digest({k: v for k, v in settings.items() if k != "digest"})
    _write(bundle / "public/experiment.json", settings)
    _write(bundle / "preparation.json", {"status": "ready", "digest": settings["digest"]})
    return settings
