"""Portable experiment bundles shared by three single-handover backends."""
from __future__ import annotations

import copy
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

from beaver.a2dpss import A2DPSSContext
from beaver.a2dpss_experiment import g_h_from_config_extra, build_paired_fixture, A2DPSSPairedFixture
from beaver.atcaas_params import ATCaaSProtocolParameters
from beaver.handover_fixtures import build_handover_local_config
from beaver.async_algorand.crypto import generate_identity
from beaver.async_algorand.model import Parameters, canonical, decode, digest
from beaver.a2dpss_benchmark_metrics import aggregate_cpu, normalize_cpu


PROTOCOLS = ("dag", "acs", "algorand")
SCHEMA = "a2dpss-three-backend-benchmark/v1"


def read(path):
    return decode(Path(path).read_bytes())


def write(path, value, private=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600 if private else 0o644)
            stream.write(canonical(value) + b"\n")
            stream.flush()
        # Publish a complete file atomically, preserving exclusive-create
        # semantics even when two supervisors accidentally target the same node.
        os.link(temporary, str(path))
    finally:
        os.unlink(temporary)


def parse_peers(value, count):
    if not isinstance(value, list) or len(value) != count:
        raise ValueError("peers must be a JSON list with exactly M host:port strings")
    result = []
    for peer in value:
        if not isinstance(peer, str) or peer.count(":") != 1:
            raise ValueError("use IPv4 addresses or DNS names followed by :port")
        host, port = peer.split(":")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", host) or not port.isdigit() or not 1 <= int(port) <= 65535:
            raise ValueError("invalid peer host or port")
        result.append(f"{host}:{int(port)}")
    if len(set(result)) != count:
        raise ValueError("each node needs a unique host:port endpoint")
    return result


def local_context(config):
    params = ATCaaSProtocolParameters.from_mapping(config)
    extra = config["extra"]
    if (extra.get("dpss") or {}).get("private_material_scope") != "node-local":
        raise ValueError("prepare requires Phase-6 node-local DPSS configs; run run_key_gen_atcaas first")
    g, h = g_h_from_config_extra(extra)
    local = build_handover_local_config(extra, g_hex=g, h_hex=h, my_id=config["my_id"],
                                        M=params.M, service_t=params.t_r, epoch=0)
    # The legacy config context and Algorithm 4 have distinct wire schemas.
    ctx = local.ctx
    context = A2DPSSContext(epoch=ctx.epoch, sid=ctx.sid.replace("dpss", "a2dpss"),
        source_committee=ctx.source_committee, successor_committee=ctx.successor_committee,
        transfer_seq=ctx.transfer_seq, g_hex=g, h_hex=h, key_base=ctx.key_base,
        successor_rpk_list=ctx.successor_rpk_list, service_t=ctx.service_t,
        old_commitment_vector=ctx.old_commitment_vector, holder_eval_points=ctx.holder_eval_points,
        successor_role_indices=ctx.successor_role_indices)
    return context, local


def load_bundle(directory):
    directory = Path(directory)
    manifest = read(directory / "public/manifest.json")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported benchmark bundle")
    expected = manifest.get("bundle_digest")
    if digest({k: v for k, v in manifest.items() if k != "bundle_digest"}) != expected:
        raise ValueError("benchmark manifest digest mismatch")
    for relative, checksum in manifest["artifacts"].items():
        if relative not in ("public/role-trace.json", "public/fixture.json"):
            raise ValueError("unknown public artifact")
        if hashlib.sha256((directory / relative).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"artifact digest mismatch: {relative}")
    return manifest


def prepare(args):
    from scripts.generate_coin_keys import validate_coin_key_directory
    from scripts.generate_dag_role_trace import selected_roles_for_round
    from beaver.protocol_trace import CommitteeRoleTraceArtifact

    source = Path(args.config_dir).resolve()
    first = read(source / "local.0.json")
    params = ATCaaSProtocolParameters.from_mapping(first)
    # Reject incompatible Algorand thresholds before creating any output.
    # Fixed-size trials use exactly n_a roles, not the Bernoulli upper bound m_a.
    effective_m = params.n_a if args.selection_mode == "fixed-size" else params.m_a
    alg_params = Parameters(params.M, effective_m, params.quorum_h, params.n_a / params.M,
                            args.selection_mode, args.trial_id, args.max_views)
    for name in ("timeout", "startup_timeout", "coin_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if args.rounds < 4 or args.max_views < 1:
        raise ValueError("rounds must be >= 4 and max-views positive")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", args.trial_id):
        raise ValueError("trial-id must use 1–96 letters, digits, underscores or hyphens")
    peers = parse_peers(read(args.peers) if args.peers else first["peers"], params.M)
    validate_coin_key_directory(config_dir=source, network_size=params.M)
    configs, contexts = [], []
    for node in range(params.M):
        config = read(source / f"local.{node}.json")
        if config["my_id"] != node or ATCaaSProtocolParameters.from_mapping(config) != params:
            raise ValueError("node identity or protocol parameters disagree")
        if any(
            config["extra"].get(field) != first["extra"].get(field)
            for field in ("SRS", "vrf_pks", "chunk_lookup_tables")
        ):
            raise ValueError("SRS, VRF public keys or chunk lookup tables disagree")
        config["peers"] = peers
        config["extra"]["run_id"] = args.trial_id
        config["extra"]["dpss"]["sid"] = "a2dpss-" + args.trial_id
        context, _ = local_context(config)
        contexts.append(context)
        configs.append(config)
    context = contexts[0]
    if any(c.context_digest != context.context_digest for c in contexts):
        raise ValueError("initial A2DPSS sharing or successor roles disagree")
    g, h = context.g_hex, context.h_hex
    identities = [generate_identity(g) for _ in range(params.M)]
    # Reuse the same long-lived VRF setup; election labels remain protocol-native.
    for node, (private, public) in enumerate(identities):
        private["vrf"] = configs[node]["extra"]["vrf_sk"]
        public["vrf"] = copy.deepcopy(configs[node]["extra"]["vrf_pks"][node])
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    (directory / "nodes").mkdir(mode=0o700)
    artifacts = {}
    # DAG uses the role trace for wave leaders.  ACS derives each activation
    # committee from its own label plus the shared fixed-size seed, and applies
    # Fig. 18's candidate-set hash leader rule instead of the DAG role trace.
    vrf_sks = [c["extra"]["vrf_sk"] for c in configs]
    roles = {index: selected_roles_for_round(
        epoch=args.trial_id, parent_round=4 * (index - 1), M=params.M,
        threshold_fraction=params.n_a / params.M, vrf_sks=vrf_sks,
        vrf_pks=first["extra"]["vrf_pks"], g_hex=g, selection_mode=args.selection_mode,
        committee_size=params.n_a, fixed_committee_seed=args.trial_id)
        for index in range(1, max(args.rounds, args.max_views) + 1)}
    write(directory / "public/role-trace.json", CommitteeRoleTraceArtifact(roles).to_dict())
    artifacts["public/role-trace.json"] = hashlib.sha256((directory / "public/role-trace.json").read_bytes()).hexdigest()
    for node, config in enumerate(configs):
        write(directory / f"nodes/{node}/config.json", config, private=True)
        write(directory / f"nodes/{node}/algorand-key.json", identities[node][0], private=True)
        lookup_set = config["extra"].get("chunk_lookup_tables")
        if lookup_set is not None:
            for table in lookup_set.get("tables", {}).values():
                relative = Path(table["artifact_path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("chunk lookup artifact path is unsafe")
                source_artifact = source / relative
                if not source_artifact.is_file():
                    raise ValueError(f"missing chunk lookup artifact: {relative}")
                destination = directory / f"nodes/{node}" / relative
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                try:
                    os.link(source_artifact, destination)
                except OSError as exc:
                    if exc.errno != errno.EXDEV:
                        raise
                    shutil.copy2(source_artifact, destination)
    fixture_digest = None
    if args.view == "paired":
        # The existing builder reads sibling local.N.json files. Use a private
        # temporary setup directory; exported node bundles never need these siblings.
        with tempfile.TemporaryDirectory(prefix="a2dpss-fixture-") as temporary:
            for node, config in enumerate(configs):
                write(Path(temporary) / f"local.{node}.json", config, private=True)
            fixture = build_paired_fixture(config_path=str(Path(temporary) / "local.0.json"),
                trial_id=args.trial_id, trial_epoch=args.trial_id)
        context = fixture.context
        write(directory / "public/fixture.json", fixture.to_fixture_obj())
        artifacts["public/fixture.json"] = hashlib.sha256((directory / "public/fixture.json").read_bytes()).hexdigest()
        fixture_digest = fixture.digest()
    manifest = dict(schema=SCHEMA, trial_id=args.trial_id, view=args.view,
        parameters=params.to_config_obj(), peers=peers, context=context.to_obj(),
        input_digest=digest(context.to_obj()), fixture_digest=fixture_digest,
        source_nodes=sorted(context.holder_eval_points), successor_nodes=sorted(context.successor_role_indices),
        coin_public_key_digest=first["extra"]["coin_public_key_digest"],
        selection_mode=args.selection_mode, committee_seed=args.trial_id,
        algorand_parameters=alg_params.__dict__, algorand_public_keys=[p for _, p in identities],
        timeout=args.timeout, startup_timeout=args.startup_timeout, coin_timeout=args.coin_timeout,
        rounds=args.rounds, max_views=args.max_views, artifacts=artifacts,
        assumptions=["KM-NCE", "role assignment setup", "trusted experiment network"],
        vrf_backend_requirement="native libbulletproofs_amcl.so; worker preflight",
        online_costs={"dag": "threshold-BLS coin", "acs": "threshold-BLS coin",
                      "algorand": "native AVSS Setup/Share/Rec, Gather, PE, NWA"},
        latency_definition="successor-max of local monotonic time from ready barrier to reconstruction",
        contribution_generation="online" if args.view == "live" else "offline; paired TOB view",
        private_config_digests={str(i): digest(c) for i, c in enumerate(configs)})
    manifest["bundle_digest"] = digest(manifest)
    write(directory / "public/manifest.json", manifest)
    return manifest


def export_nodes(directory, output, nodes):
    manifest = load_bundle(directory)
    validate_nodes(nodes, manifest)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    shutil.copytree(Path(directory) / "public", output / "public")
    for node in nodes:
        source = Path(directory) / f"nodes/{node}"
        if any(source.glob("*.used")):
            raise ValueError("cannot export a node after a protocol has already started")
        (output / "nodes").mkdir(exist_ok=True, mode=0o700)
        shutil.copytree(source, output / f"nodes/{node}")
    return output


def validate_nodes(nodes, manifest):
    if not nodes or len(set(nodes)) != len(nodes) or any(type(i) is not int or not 0 <= i < manifest["parameters"]["M"] for i in nodes):
        raise ValueError("nodes must be distinct node ids in [0, M)")


def normalize_result(manifest, protocol, node, raw):
    context = A2DPSSContext.from_obj(manifest["context"])
    timeline = raw.get("a2dpss_handover_metrics")
    from beaver.experiment_trace import validate_a2dpss_handover_metrics
    errors = validate_a2dpss_handover_metrics(timeline, protocol={"dag": "dag-tob", "acs": "acs-tob", "algorand": "async-algorand"}[protocol],
        node=node, qid=context.encode_qid(), is_successor=node in manifest["successor_nodes"])
    events = [e for e in raw.get("a2dpss_events", []) if e.get("event") == "public_finalized"]
    if len(events) != 1:
        errors.append("expected exactly one public finalization")
    public = events[0] if len(events) == 1 else {}
    if public.get("context_digest") != context.context_digest or public.get("qid") != context.encode_qid():
        errors.append("public result belongs to another context")
    accepted = public.get("accepted_contribution_digests", [])
    if len(accepted) != context.service_t + 1 or len({str(x) for x in accepted}) != len(accepted):
        errors.append("public result must bind exactly t_r+1 distinct contributions")
    reconstructed = any(e.get("event") == "reconstructed" for e in raw.get("a2dpss_events", []))
    if reconstructed != (node in manifest["successor_nodes"]):
        errors.append("successor reconstruction role mismatch")
    if raw.get("cleanup_completed") is not True:
        errors.append("protocol cleanup did not complete")
    if protocol != "algorand" and (raw.get("coin_mode") != "threshold-bls" or raw.get("coin_public_key_digest") != manifest["coin_public_key_digest"]):
        errors.append("threshold coin configuration mismatch")
    backend = raw.get("vrf_backend") or {}
    implementations = backend.get("implementations") or {}
    if (
        backend.get("required") is not True
        or implementations.get("beaver.vrf") is not True
        or implementations.get("beaver.yosorbc") is not True
    ):
        errors.append("native VRF backend preflight is missing or failed")
    for value in (timeline or {}).get("timestamps_sec", {}).values():
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            errors.append("nonfinite or negative timestamp")
    if errors:
        raise ValueError("; ".join(errors))
    return dict(schema=SCHEMA, status="ok", protocol=protocol, node=node,
        trial_id=manifest["trial_id"], bundle_digest=manifest["bundle_digest"],
        input_digest=manifest["input_digest"], view=manifest["view"],
        public_result={k: public[k] for k in ("qid", "context_digest", "accepted_contribution_digests", "successor_commitment_vector")},
        reconstructed=reconstructed, timeline=timeline,
        protocol_wire_bytes=raw["benchmark_protocol_wire_bytes"],
        adpss_cpu=normalize_cpu(raw, protocol, node),
        raw=raw)


def collect(directory, results, protocol):
    manifest = load_bundle(directory)
    records, failures = [], []
    for node in range(manifest["parameters"]["M"]):
        path = Path(results) / protocol / f"node-{node}.json"
        if not path.is_file():
            failures.append(f"node {node}: missing result")
            continue
        try:
            record = read(path)
        except (ValueError, OSError) as error:
            failures.append(f"node {node}: unreadable or incomplete result: {error}")
            continue
        if any(record.get(k) != v for k, v in dict(node=node, protocol=protocol, bundle_digest=manifest["bundle_digest"]).items()):
            failures.append(f"node {node}: result identity mismatch")
        elif record.get("status") != "ok":
            failures.append(f"node {node}: {record.get('status')}: {record.get('reason', '')}")
        else:
            try:
                checked = normalize_result(manifest, protocol, node, record["raw"])
                stored = {k: v for k, v in record.items() if k != "machine"}
                # v1 node files predate the derived CPU field. Validate every
                # original field, then enrich from measured raw counters only.
                comparable = checked if "adpss_cpu" in stored else {
                    k: v for k, v in checked.items() if k != "adpss_cpu"}
                if comparable != stored:
                    raise ValueError("derived metrics differ from raw result")
                if "machine" in record:
                    checked["machine"] = record["machine"]
                records.append(checked)
            except (ValueError, KeyError, TypeError) as error:
                failures.append(f"node {node}: {error}")
    if records and len({digest(r["public_result"]) for r in records}) != 1:
        failures.append("observers finalized different T or successor commitments")
    successor_times = [r["timeline"]["durations_sec"]["handover_latency"] for r in records if r["reconstructed"]]
    if len(successor_times) != len(manifest["successor_nodes"]):
        failures.append("successor results are incomplete")
    if not failures and manifest["view"] == "paired":
        from beaver.a2dpss import A2DPSSJustifierManager, A2DPSSPublicFinalizer
        fixture = A2DPSSPairedFixture.from_file(str(Path(directory) / "public/fixture.json"))
        justifier = A2DPSSJustifierManager([fixture.context])
        selected = justifier.select_justified_subset(fixture.qid(), fixture.public_entries())
        expected = A2DPSSPublicFinalizer(justifier).on_deliver(fixture.qid(), selected).to_obj()
        if any(any(r["public_result"][k] != expected[k] for k in r["public_result"]) for r in records):
            failures.append("paired result does not match the fixture's canonical T")
    cpu = aggregate_cpu(records, manifest["parameters"]["M"])
    cpu["valid_protocol_result"] = not failures
    if failures:
        cpu["complete"] = False
        cpu["total_cpu_seconds"] = None
    return dict(schema=SCHEMA, status="failed" if failures else "ok", failures=failures,
        protocol=protocol, trial_id=manifest["trial_id"], bundle_digest=manifest["bundle_digest"],
        input_digest=manifest["input_digest"], fixture_digest=manifest["fixture_digest"],
        view=manifest["view"], parameters=manifest["parameters"], participants=len(records),
        successor_max_latency_seconds=max(successor_times) if successor_times and not failures else None,
        protocol_wire_bytes=sum(r["protocol_wire_bytes"] for r in records) if not failures else None,
        adpss_cpu=cpu,
        per_node=[{k: v for k, v in r.items() if k != "raw"} for r in records])


def compare(directory, dag_results, acs_results, algorand_results):
    """Revalidate three node-result directories against one trial bundle.

    Live executions generate independent fresh transcripts, so equal initial
    sharing/roles are required but equal newly decided T or vc' are not.
    Paired collect additionally checks each backend against the same fixture.
    """
    manifest = load_bundle(directory)
    summaries = {protocol: collect(directory, results, protocol)
        for protocol, results in zip(PROTOCOLS, (dag_results, acs_results, algorand_results))}
    failures = [f"{protocol}: {error}" for protocol, result in summaries.items()
                for error in result["failures"]]
    public_digests = {protocol: digest(result["per_node"][0]["public_result"])
        if result["status"] == "ok" and result["per_node"] else None
        for protocol, result in summaries.items()}
    equal_public = all(public_digests.values()) and len(set(public_digests.values())) == 1
    if manifest["view"] == "paired" and not equal_public:
        failures.append("paired backends did not all finalize the fixture's public result")
    cpu_comparable = not failures and all(s["adpss_cpu"]["complete"] for s in summaries.values())
    metrics = {protocol: {
        "successor_max_latency_seconds": result["successor_max_latency_seconds"],
        "protocol_wire_bytes": result["protocol_wire_bytes"],
        "adpss_cpu_seconds": result["adpss_cpu"]["total_cpu_seconds"],
    } for protocol, result in summaries.items()}
    ratios = {}
    for protocol in ("acs", "algorand"):
        ratios[protocol + "_over_dag"] = {}
        for metric, denominator in metrics["dag"].items():
            numerator = metrics[protocol][metric]
            eligible = not failures and (metric != "adpss_cpu_seconds" or cpu_comparable)
            ratios[protocol + "_over_dag"][metric] = (
                numerator / denominator if eligible and denominator and numerator is not None else None)
    return dict(schema="a2dpss-three-backend-comparison/v1",
        status="failed" if failures else "ok", failures=failures,
        trial_id=manifest["trial_id"], bundle_digest=manifest["bundle_digest"],
        input_digest=manifest["input_digest"], fixture_digest=manifest["fixture_digest"],
        view=manifest["view"], parameters=manifest["parameters"],
        comparison_scope="same initial sharing, roles and parameters" if manifest["view"] == "live"
            else "same frozen contribution fixture and canonical public result",
        public_results_equal=bool(equal_public), public_result_digests=public_digests,
        cpu_comparable=cpu_comparable, metrics_complete=cpu_comparable,
        metrics=metrics, ratios=ratios, protocols=summaries)
