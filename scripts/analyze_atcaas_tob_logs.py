import argparse
import glob
import json
import re
import statistics
from pathlib import Path

from scripts.analyze_a2dpss_network_logs import (
    _dag_agreement_parameter_view,
    _validate_dag_selectblock_metrics,
)


SUMMARY_RE = re.compile(r"\[SUMMARY\]\s+({.*})\s*$")


def _parse_summary_line(line: str):
    match = SUMMARY_RE.search(line)
    if match is not None:
        return json.loads(match.group(1))
    line = line.strip()
    if line.startswith("{") and line.endswith("}"):
        data = json.loads(line)
        if "node" in data and "ledger" in data:
            return data
    return None


def _read_last_summary(path: str):
    last = None
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                parsed = _parse_summary_line(line)
            except json.JSONDecodeError:
                continue
            if parsed is not None:
                last = parsed
    return last


def _summary_stats(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": min(values),
        "avg": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": max(values),
    }


def _percentile(ordered, quantile):
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _is_prefix(left, right):
    return len(left) <= len(right) and left == right[: len(left)]


def _all_prefix_comparable(sequences):
    sequences = list(sequences)
    return all(
        _is_prefix(left, right) or _is_prefix(right, left)
        for index, left in enumerate(sequences)
        for right in sequences[index + 1 :]
    )


def _compact_ledger(ledger):
    return [
        {
            qid: len(entries)
            for qid, entries in sorted(block.items())
        }
        for block in ledger
    ]


def _compact_delivery_events(events):
    return [
        {
            "sequence": event.get("sequence"),
            "qid": event.get("qid"),
            "entry_count": len(event.get("entries", [])),
            "ordered_vertex": event.get("ordered_vertex"),
            "committed_wave": event.get("committed_wave"),
        }
        for event in events
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze ATCaaS DAG-TOB smoke logs."
    )
    parser.add_argument("--glob", required=True, help="Log glob, e.g. log/atcaas-*.log")
    parser.add_argument("--expected-nodes", type=int)
    parser.add_argument("--target-deliveries", type=int, default=1)
    parser.add_argument(
        "--expected-coin-mode",
        choices=("ideal", "threshold-bls"),
        default="ideal",
    )
    parser.add_argument(
        "--compact-ledger",
        action="store_true",
        help="Print qid -> entry-count instead of full payload hex.",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(args.glob))
    summaries = {}
    missing_summary = []
    for path in paths:
        summary = _read_last_summary(path)
        if summary is None:
            missing_summary.append(path)
            continue
        summaries[int(summary["node"])] = summary

    agreement_parameters = {}
    selectblock_metrics = {}
    for node, summary in summaries.items():
        parameter_view = _dag_agreement_parameter_view(summary, node)
        if parameter_view is not None:
            agreement_parameters[node] = parameter_view
        metrics_view = _validate_dag_selectblock_metrics(summary, node)
        if metrics_view is not None:
            selectblock_metrics[node] = metrics_view
    if agreement_parameters and len(agreement_parameters) != len(summaries):
        raise SystemExit("DAG agreement parameters are missing from some nodes")
    if selectblock_metrics and len(selectblock_metrics) != len(summaries):
        raise SystemExit("DAG SelectBlock metrics are missing from some nodes")

    ledgers = {node: summary.get("ledger", []) for node, summary in summaries.items()}
    first_ledger = next(iter(ledgers.values()), [])
    all_ledgers_equal = all(ledger == first_ledger for ledger in ledgers.values())
    delivery_events = {
        node: summary.get("protocol_delivery_events", [])
        for node, summary in summaries.items()
    }
    first_delivery_events = next(iter(delivery_events.values()), [])
    all_delivery_events_equal = all(
        events == first_delivery_events for events in delivery_events.values()
    )
    all_delivery_prefixes_comparable = _all_prefix_comparable(
        delivery_events.values()
    )
    ordered_vertices = {
        node: summary.get("ordered_vertices", [])
        for node, summary in summaries.items()
    }
    all_ordered_vertex_prefixes_comparable = _all_prefix_comparable(
        ordered_vertices.values()
    )
    deliver_counts = {
        node: int(summary.get("deliver_count", 0))
        for node, summary in summaries.items()
    }
    delivery_event_counts = {
        node: int(summary.get("delivery_event_count", 0))
        for node, summary in summaries.items()
    }
    delivery_event_counts_match = all(
        delivery_event_counts.get(node) == deliver_counts.get(node)
        == len(delivery_events.get(node, []))
        for node in summaries
    )
    enough_deliveries = all(
        count >= args.target_deliveries for count in deliver_counts.values()
    )
    cleanup = {
        node: {
            "cleanup_completed": summary.get("cleanup_completed"),
            "active_after": summary.get("active_rbc_instances_after_cleanup"),
        }
        for node, summary in summaries.items()
    }
    first_delivery_latencies = {
        node: summary.get("first_delivery_latency_sec")
        for node, summary in summaries.items()
    }
    bytes_sent = {
        node: int(summary.get("bytes_sent", 0))
        for node, summary in summaries.items()
    }
    communication_bits = {
        node: int(summary.get("communication_bits", bytes_sent[node] * 8))
        for node, summary in summaries.items()
    }
    workload_traces = {
        node: summary.get("workload_trace") for node, summary in summaries.items()
    }
    first_workload_trace = next(iter(workload_traces.values()), None)
    all_workload_traces_equal = all(
        trace == first_workload_trace for trace in workload_traces.values()
    )
    query_latencies = []
    makespans = []
    pipelining_overlaps = []
    for summary in summaries.values():
        lifecycle = summary.get("query_lifecycle") or {}
        makespans.append(lifecycle.get("makespan_sec"))
        pipelining_overlaps.append(lifecycle.get("pipelining_overlap_sec"))
        query_latencies.extend(
            item.get("ready_to_delivery_sec")
            for item in lifecycle.get("queries", [])
        )
    first_events = first_delivery_events
    batch_groups = {}
    for event in first_events:
        vertex = event.get("ordered_vertex") or {}
        key = (
            event.get("committed_wave"),
            vertex.get("round"),
            vertex.get("source"),
        )
        batch_groups[key] = batch_groups.get(key, 0) + 1

    coin_modes = {
        node: summary.get("coin_mode", "ideal")
        for node, summary in summaries.items()
    }
    all_coin_modes_expected = all(
        mode == args.expected_coin_mode for mode in coin_modes.values()
    )
    wave_coin_records = {
        node: {
            int(record["wave"]): record
            for record in (summary.get("wave_coin") or {}).get("waves", [])
        }
        for node, summary in summaries.items()
    }
    observed_waves = sorted(
        {
            wave
            for records in wave_coin_records.values()
            for wave in records
        }
    )
    wave_coin_agreement = {}
    for wave in observed_waves:
        views = [
            (
                records[wave].get("coin_id_digest"),
                records[wave].get("coin_digest"),
                records[wave].get("roles"),
                records[wave].get("leader"),
            )
            for records in wave_coin_records.values()
            if wave in records
        ]
        wave_coin_agreement[wave] = (
            len(views) == len(summaries)
            and bool(views)
            and all(view == views[0] for view in views)
        )
    all_wave_coins_agree = all(wave_coin_agreement.values())
    coin_calls_ready = all(
        set((summary.get("coin_observations") or {}).get("status_counts", {}))
        <= {"ready"}
        for summary in summaries.values()
    )
    coin_signed_once_per_wave = all(
        int((summary.get("coin_metrics") or {}).get("shares_signed", 0))
        <= len((summary.get("wave_coin") or {}).get("ready_waves", []))
        for summary in summaries.values()
    )
    coin_configs = {
        node: (
            tuple(summary.get("coin_committee", [])),
            summary.get("coin_t"),
            summary.get("coin_threshold"),
            summary.get("coin_membership_digest"),
            summary.get("coin_public_key_digest"),
        )
        for node, summary in summaries.items()
    }
    first_coin_config = next(iter(coin_configs.values()), None)
    all_coin_configs_equal = all(
        config == first_coin_config for config in coin_configs.values()
    )
    coin_fault_bound_valid = False
    if first_coin_config is not None:
        committee, coin_t, coin_k, _membership_digest, _public_key_digest = (
            first_coin_config
        )
        coin_c = len(committee)
        coin_fault_bound_valid = (
            not isinstance(coin_t, bool)
            and isinstance(coin_t, int)
            and not isinstance(coin_k, bool)
            and isinstance(coin_k, int)
            and 0 <= coin_t < coin_k <= coin_c - coin_t
        )
    coin_signer_permissions_valid = all(
        int((summary.get("coin_metrics") or {}).get("shares_signed", 0))
        == (
            0
            if node not in summary.get("coin_committee", [])
            or node in summary.get("coin_silent_nodes", [])
            else len((summary.get("wave_coin") or {}).get("ready_waves", []))
        )
        for node, summary in summaries.items()
    )
    coin_cleanup_complete = all(
        (summary.get("coin_metrics") or {}).get("closed") is True
        and int((summary.get("coin_metrics") or {}).get("pending_coins", -1)) == 0
        for summary in summaries.values()
    )
    coin_mode_not_forced = all(
        summary.get("coin_mode_forced_by_replay") is not True
        for summary in summaries.values()
    )
    role_trace_views = {
        node: (
            summary.get(
                "role_trace_source",
                summary.get("protocol_trace_source"),
            ),
            summary.get("role_trace_format"),
            summary.get(
                "role_trace_digest",
                summary.get("protocol_trace_digest"),
            ),
        )
        for node, summary in summaries.items()
    }
    first_role_trace = next(iter(role_trace_views.values()), None)
    all_role_traces_equal = (
        first_role_trace is not None
        and bool(first_role_trace[2])
        and all(
            view == first_role_trace for view in role_trace_views.values()
        )
    )

    result = {
        "log_glob": args.glob,
        "log_files": len(paths),
        "summary_nodes": sorted(summaries),
        "summary_count": len(summaries),
        "expected_nodes": args.expected_nodes,
        "missing_summary_files": [str(Path(path)) for path in missing_summary],
        "all_delivery_events_equal": all_delivery_events_equal,
        "all_delivery_prefixes_comparable": all_delivery_prefixes_comparable,
        "all_ordered_vertex_prefixes_comparable": (
            all_ordered_vertex_prefixes_comparable
        ),
        "ordered_vertices": ordered_vertices,
        "all_ledgers_equal": all_ledgers_equal,
        "target_deliveries": args.target_deliveries,
        "enough_deliveries": enough_deliveries,
        "deliver_counts": deliver_counts,
        "delivery_event_counts": delivery_event_counts,
        "delivery_event_counts_match": delivery_event_counts_match,
        "delivery_events": _compact_delivery_events(first_delivery_events),
        "committed_waves": {
            node: int(summary.get("committed_wave", 0))
            for node, summary in summaries.items()
        },
        "first_delivery_latencies_sec": first_delivery_latencies,
        "first_delivery_latency_summary_sec": _summary_stats(
            first_delivery_latencies.values()
        ),
        "ledger": _compact_ledger(first_ledger) if args.compact_ledger else first_ledger,
        "cleanup": cleanup,
        "bytes_sent": bytes_sent,
        "bytes_sent_summary": _summary_stats(bytes_sent.values()),
        "communication_bits": communication_bits,
        "communication_bits_summary": _summary_stats(communication_bits.values()),
        "workload_trace": first_workload_trace,
        "all_workload_traces_equal": all_workload_traces_equal,
        "query_latency_summary_sec": _summary_stats(query_latencies),
        "makespan_summary_sec": _summary_stats(makespans),
        "pipelining_overlap_summary_sec": _summary_stats(pipelining_overlaps),
        "batching": {
            "ordered_units": len(batch_groups),
            "queries_per_ordered_unit": list(batch_groups.values()),
            "queries_per_ordered_unit_summary": _summary_stats(
                batch_groups.values()
            ),
        },
        "justifier_metrics": {
            node: summary.get("justifier_metrics")
            for node, summary in summaries.items()
        },
        "dag_agreement_parameters": agreement_parameters,
        "dag_selectblock_metrics": selectblock_metrics,
        "dag_round_sizes": {
            node: summary.get("dag_round_sizes", {})
            for node, summary in summaries.items()
        },
        "accepted_rbc_vertices": {
            node: summary.get("accepted_rbc_vertices", 0)
            for node, summary in summaries.items()
        },
        "ideal_coin": all(
            summary.get("ideal_coin") is True
            for summary in summaries.values()
        ),
        "coin_costs": {
            node: summary.get("coin_cost")
            for node, summary in summaries.items()
        },
        "protocol_trace_sources": {
            node: summary.get("protocol_trace_source")
            for node, summary in summaries.items()
        },
        "coin_modes": coin_modes,
        "expected_coin_mode": args.expected_coin_mode,
        "all_coin_modes_expected": all_coin_modes_expected,
        "wave_coin_agreement": wave_coin_agreement,
        "all_wave_coins_agree": all_wave_coins_agree,
        "coin_calls_ready": coin_calls_ready,
        "coin_signed_once_per_wave": coin_signed_once_per_wave,
        "coin_configs": coin_configs,
        "all_coin_configs_equal": all_coin_configs_equal,
        "coin_fault_bound_valid": coin_fault_bound_valid,
        "coin_signer_permissions_valid": coin_signer_permissions_valid,
        "coin_cleanup_complete": coin_cleanup_complete,
        "coin_mode_not_forced_by_replay": coin_mode_not_forced,
        "role_traces": role_trace_views,
        "all_role_traces_equal": all_role_traces_equal,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.expected_nodes is not None and len(summaries) != args.expected_nodes:
        raise SystemExit("summary node count mismatch")
    if not all_ledgers_equal:
        raise SystemExit("ledger mismatch")
    if not all_delivery_events_equal:
        raise SystemExit("per-query delivery event mismatch")
    if not all_delivery_prefixes_comparable:
        raise SystemExit("per-query delivery histories are not prefix comparable")
    if not all_ordered_vertex_prefixes_comparable:
        raise SystemExit("ordered vertex histories are not prefix comparable")
    if not all_workload_traces_equal:
        raise SystemExit("workload trace mismatch")
    if not delivery_event_counts_match:
        raise SystemExit("delivery event count mismatch")
    if not enough_deliveries:
        raise SystemExit("not enough deliveries")
    if not all_coin_modes_expected:
        raise SystemExit("coin mode mismatch")
    if not coin_mode_not_forced:
        raise SystemExit("common-coin mode was changed by replay loading")
    if not all_role_traces_equal:
        raise SystemExit("committee role trace mismatch")
    if args.expected_coin_mode == "threshold-bls":
        if not observed_waves or not all_wave_coins_agree:
            raise SystemExit("wave common-coin/leader disagreement")
        if not coin_calls_ready:
            raise SystemExit("wave common coin did not complete cleanly")
        if not coin_signed_once_per_wave:
            raise SystemExit("coin committee member signed a wave more than once")
        if not all_coin_configs_equal:
            raise SystemExit("threshold coin committee configuration mismatch")
        if not coin_fault_bound_valid:
            raise SystemExit(
                "threshold coin must satisfy 0 <= coin_t < coin_k "
                "<= coin_c - coin_t"
            )
        if not coin_signer_permissions_valid:
            raise SystemExit("threshold coin signer permissions mismatch")
        if not coin_cleanup_complete:
            raise SystemExit("threshold coin cleanup incomplete")


if __name__ == "__main__":
    main()
