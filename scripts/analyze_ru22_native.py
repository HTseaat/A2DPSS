"""Validate dedicated-verifier RU22 outputs and aggregate measured costs."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from beaver.baselines.ru22_dpss import RU22VPS
from beaver.baselines.ru22_native_experiment import load_native_public, BROADCAST_DELIVERY
from beaver.baselines.ru22_native import CRYPTO_STAGES
from beaver.baselines.ru22_native_roles import public_authority, CONTROL


def nonnegative(value, name, *, integer=False):
    if (type(value) not in ((int,) if integer else (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError("invalid " + name)
    return value


def analyze_native(fixture, nodes, *, experiment_digest=None):
    public, base = load_native_public(fixture)
    c = base.context
    authority = public_authority(c, public["roles"])
    operation_context = None
    if "operation_context" in public:
        from beaver.threshold_operation_contract import ThresholdOperationContext, operation_output_digest
        from beaver.g1_backend import G1Point
        operation_context = ThresholdOperationContext.from_obj(public["operation_context"])
    M = public["roles"]["population"]
    if len(nodes) != M or {n.get("node_id") for n in nodes} != set(range(M)):
        raise ValueError("missing or duplicate native node summaries")
    leaders = list(dict(authority.leader_keys))
    committees = {str(i): list(authority.verifiers_for(i)) for i in leaders}
    successors = sorted(c.successor_role_indices)
    arrivals, cpu_stages, control_cpu_stages = {}, {}, {}
    network, control_bytes, payload = 0, 0, 0
    by_phase, actions = {}, 0
    schemas = {row.get("schema") for row in nodes}
    if len(schemas) != 1:
        raise ValueError("cannot mix native measurement versions")
    detailed = schemas == {"ru22-native-node-metrics/v3"}
    if public.get("protocol_delivery") == BROADCAST_DELIVERY and not detailed:
        raise ValueError("broadcast fixture requires broadcast measurements")
    crypto_metrics = {stage: dict(invocations=0, wall_seconds=0., cpu_seconds=0.) for stage in CRYPTO_STAGES}
    for row in nodes:
        if (row.get("schema") not in ("ru22-native-node-metrics/v2", "ru22-native-node-metrics/v3")
                or row.get("protocol") != "ru22" or row.get("protocol_mode") != "native"
                or row.get("status") != "success" or type(row.get("node_id")) is not int
                or not row.get("cleanup", {}).get("completed")):
            raise ValueError("native node failed or cleanup incomplete")
        if detailed and row.get("protocol_delivery") != BROADCAST_DELIVERY:
            raise ValueError("native broadcast delivery mode missing")
        if not detailed and "protocol_delivery" in row:
            raise ValueError("legacy measurements cannot claim broadcast delivery")
        for name, expected in (("native_manifest_digest", public["digest"]),
                               ("common_workload_digest", public["common_workload_digest"]),
                               ("arm_manifest_digest", public["arm_manifest_digest"]),
                               ("authority_digest", authority.committee_digest)):
            if row.get(name) != expected:
                raise ValueError("native " + name + " mismatch")
        if experiment_digest is not None and row.get("experiment_digest") != experiment_digest:
            raise ValueError("native experiment digest mismatch")
        if (row.get("leader_nodes") != leaders or row.get("verifier_committees") != committees
                or row.get("successor_nodes") != successors
                or row.get("verifier_committee_count") != c.resharing_n
                or row.get("verifier_role_count") != c.resharing_n ** 2
                or row.get("verifier_send_scope") != public["verifier_send_scope"]
                or row.get("strict_refresh_wide_yoso") is not True):
            raise ValueError("native role assignment mismatch")
        if row.get("rejected_messages") != 0 or row.get("transport_errors") != []:
            raise ValueError("native benchmark reported message errors")
        node = row["node_id"]
        roles = row["verifier_roles"]
        if set(roles) != {str(i) for i in leaders if node in authority.verifiers_for(i)}:
            raise ValueError("wrong local verifier roles")
        for role in roles.values():
            if (type(role.get("sent")) is not bool or type(role.get("closed")) is not bool
                    or role["sent"] and not role["closed"]):
                raise ValueError("verifier role remained live after sending")
        local_actions = sum(v["sent"] for v in roles.values())
        if row.get("verifier_actions") != local_actions:
            raise ValueError("verifier action count mismatch")
        actions += local_actions
        timeline = row["timeline"]
        frozen = nonnegative(timeline["frozen_at"], "freeze time")
        certs = [RU22VPS.from_obj(v) for v in row["certified_vps"]]
        ids = {v.vps_id for v in certs}
        local = row["vps_arrivals"]
        if (len(ids) != len(certs) or len(ids) > c.resharing_n or set(local) != ids
                or any(not authority.verify(v) for v in certs)
                or row["local_opening_checks"] != len(ids)):
            raise ValueError("invalid native certified VPS output")
        if node in successors:
            if not ids:
                raise ValueError("successor has no reconstructed VPS")
            for elapsed in local.values():
                if nonnegative(elapsed, "arrival time") > frozen:
                    raise ValueError("arrival after snapshot")
            if timeline["t_first_local_vps"] != min(local.values()):
                raise ValueError("first VPS time mismatch")
            arrivals[node] = local
        elif ids or timeline["t_first_local_vps"] is not None:
            raise ValueError("non-successor reported shares")
        if operation_context:
            if (row.get("operation_context_digest") != operation_context.context_digest
                    or row.get("operation_trial_id") != public["operation_trial_id"]
                    or set(row.get("operation_results", {})) != ids):
                raise ValueError("native operation identity/output set mismatch")
            for vps in certs:
                value = row["operation_results"][vps.vps_id]
                output_digest = operation_output_digest(operation_id=operation_context.operation_id,
                    operation_input=operation_context.operation_input,
                    operation_output=G1Point.from_hex(value["operation_output"]))
                if (value["operation_context_digest"] != operation_context.context_digest
                        or value["vps_id"] != vps.vps_id
                        or value["reconstruction_digest"] != vps.certificate.reconstruction_digest
                        or value["operation_output_digest"] != output_digest
                        or output_digest != operation_context.expected_operation_output_digest):
                    raise ValueError("invalid native operation result or QVC binding")
        for stage, values in row["cpu"].items():
            nonnegative(values["invocations"], "CPU invocations", integer=True)
            nonnegative(values["wall_seconds"], "wall duration")
            seconds = nonnegative(values["cpu_seconds"], "CPU duration")
            target = control_cpu_stages if stage.startswith("control_") else cpu_stages
            target[stage] = target.get(stage, 0.) + seconds
        combines = nonnegative(row["leader_combines"], "leader combine count", integer=True)
        if combines > int(node in leaders):
            raise ValueError("leader sent more than once or has no role")
        required_counts = {"source_generation": int(node in c.source_role_indices),
                           "leader_combination": combines,
                           "verifier_signature_generation": local_actions,
                           "verifier_combine_verification": local_actions}
        for stage, count in required_counts.items():
            if row["cpu"].get(stage, {}).get("invocations", 0) != count:
                raise ValueError("missing or inconsistent native CPU stage: " + stage)
        if detailed:
            details = row.get("crypto_metrics", {})
            if set(details) != set(CRYPTO_STAGES):
                raise ValueError("missing native PVSS/DLEQ measurements")
            for role, parent in (("source", "source_generation"),
                                 ("leader", "leader_pvr_verification"),
                                 ("verifier", "verifier_combine_verification")):
                suffix = "generation" if role == "source" else "verification"
                names = [role + "_" + kind + "_" + suffix for kind in ("pvss", "dleq")]
                expected = (len(base.source_vps_views[node]) if node in c.source_role_indices else 0) if role == "source" else (
                    row["cpu"].get(parent, {}).get("invocations", 0) * (c.resharing_threshold + 1 if role == "verifier" else 1))
                for name in names:
                    values = details[name]
                    if set(values) != {"invocations", "wall_seconds", "cpu_seconds"}:
                        raise ValueError("invalid native crypto measurement fields")
                    for key, value in values.items():
                        nonnegative(value, "crypto " + key, integer=key == "invocations")
                        crypto_metrics[name][key] += value
                    if values["invocations"] != expected:
                        raise ValueError("native crypto invocation count mismatch")
                for key in ("wall_seconds", "cpu_seconds"):
                    if sum(details[name][key] for name in names) > row["cpu"].get(parent, {}).get(key, 0.) + 1e-6:
                        raise ValueError("native crypto submeasurements exceed parent stage")
            # Different leaders may certify the same VPS. Count actual QVC
            # checks/reconstructions, not deduplicated output identifiers.
            if row["cpu"].get("successor_qvc_verification", {}).get("invocations", 0) != row["cpu"].get("successor_reconstruction", {}).get("invocations", 0):
                raise ValueError("missing native successor QVC measurements")
        if ids and (row["cpu"].get("successor_reconstruction", {}).get("invocations", 0) < len(ids)
                    or row["cpu"].get("successor_signature_verification", {}).get("invocations", 0) < len(ids)):
            raise ValueError("missing native successor CPU measurements")
        comm = row["communication"]
        phases = comm["by_phase"]
        if set(phases) - (CONTROL | {"contribution", "combine", "vote"}):
            raise ValueError("unknown communication phase")
        for phase, size in phases.items():
            nonnegative(size, "network bytes", integer=True)
            by_phase[phase] = by_phase.get(phase, 0) + size
        net = sum(size for phase, size in phases.items() if phase not in CONTROL)
        ctl = sum(size for phase, size in phases.items() if phase in CONTROL)
        if (comm["total_network_bytes"] != net or comm["control_bytes"] != ctl
                or any(comm[k] != 0 for k in ("dag_protocol_bytes", "common_coin_bytes", "yoso_rbc_envelope_bytes"))):
            raise ValueError("native communication accounting mismatch")
        expected_counts = {
            "contribution": len(leaders) - int(node in leaders) if node in c.source_role_indices else 0,
            "combine": (len(committees[str(node)]) - int(node in committees[str(node)])) if row["leader_combines"] and node in leaders else 0,
            "vote": local_actions * (len(successors) - int(node in successors)),
        }
        if detailed:
            expected_counts = {"contribution": (M - 1) * int(node in c.source_role_indices),
                               "combine": (M - 1) * combines,
                               "vote": (M - 1) * local_actions}
        for phase, count in expected_counts.items():
            if comm["messages_by_phase"].get(phase, 0) != count:
                raise ValueError("native multicast count mismatch")
        network += net
        control_bytes += ctl
        payload += sum(nonnegative(v, "payload bytes", integer=True) for v in comm["application_payload_bytes"].values())
    common = set.intersection(*(set(v) for v in arrivals.values()))
    if not common:
        raise ValueError("successor VPS lists have no common reconstructible value")
    latency = min(max(v[vid] for v in arrivals.values()) for vid in common)
    comparison_stages = {name: value for name, value in cpu_stages.items()
                         if not name.startswith(("envelope_", "vrf_"))}
    return dict(schema="ru22-native-summary/v3" if detailed else "ru22-native-summary/v2", status="success", protocol="ru22", protocol_mode="native",
        protocol_delivery=BROADCAST_DELIVERY if detailed else "legacy-targeted-committee-multicast",
        crypto_metrics=crypto_metrics if detailed else None,
        crypto_accounting="nested subsets of cpu stages; not additional CPU; old-to-new DLEQ only" if detailed else "unavailable in legacy measurements",
        node_count=M, successor_nodes=successors, leader_nodes=leaders, verifier_committees=committees,
        verifier_committee_count=c.resharing_n, verifier_role_count=c.resharing_n ** 2,
        verifier_actions=actions, verifier_send_scope=public["verifier_send_scope"],
        yoso_scope=public["yoso_scope"], strict_refresh_wide_yoso=True,
        common_workload_digest=public["common_workload_digest"], arm_manifest_digest=public["arm_manifest_digest"],
        native_manifest_digest=public["digest"], experiment_digest=experiment_digest,
        common_vps_ids=sorted(common), successor_max_latency_sec=latency,
        latency_definition="earliest VPS reconstructed by every successor; maximum local elapsed time after startup barrier",
        cpu_seconds=sum(cpu_stages.values()), cpu_stage_seconds=cpu_stages,
        comparison_cpu_seconds=sum(comparison_stages.values()), comparison_cpu_stage_seconds=comparison_stages,
        comparison_cpu_accounting="measured ADPSS computation incl. native QVC; transport-envelope authentication, sortition and experiment control excluded",
        control_cpu_stage_seconds=control_cpu_stages,
        cpu_accounting="non-overlapping measured protocol stages, including VRF/authentication; experiment control separate",
        communication=dict(total_network_bytes=network, control_bytes=control_bytes, by_phase=by_phase,
            application_payload_bytes=payload, dag_protocol_bytes=0, common_coin_bytes=0, yoso_rbc_envelope_bytes=0),
        analyzer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()
    rows = [json.loads(p.read_text()) for p in sorted(args.results.glob("node.*.json"))]
    print(json.dumps(analyze_native(args.fixture, rows), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
