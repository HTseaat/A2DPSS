import argparse
import glob
import json
import re
import statistics
from pathlib import Path


SUMMARY_RE = re.compile(r"\[SUMMARY\]\s+({.*})\s*$")


def _parse_summary_line(line: str):
    match = SUMMARY_RE.search(line)
    if match is not None:
        return json.loads(match.group(1))
    line = line.strip()
    if line.startswith("{") and line.endswith("}"):
        data = json.loads(line)
        if data.get("protocol") == "async-yoso-tob":
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


def _qid_instance_from_payload_hex(payload_hex: str) -> str:
    try:
        payload = bytes.fromhex(payload_hex)
        data = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return ""
    qid_raw = data.get("qid")
    if qid_raw is None:
        return ""
    try:
        qid_data = json.loads(str(qid_raw))
    except json.JSONDecodeError:
        return str(qid_raw)
    fields = qid_data.get("fields", {})
    if isinstance(fields, dict) and "instance" in fields:
        return str(fields["instance"])
    return str(qid_raw)


def _block_qids_by_height(ledger):
    out = []
    for block in ledger:
        qids = {
            _qid_instance_from_payload_hex(payload_hex)
            for payload_hex in block
        }
        qids.discard("")
        out.append(sorted(qids))
    return out


def _parse_expected_block_qids(raw: str):
    if not raw:
        return None
    return [
        [item for item in part.split(",") if item]
        for part in raw.split("|")
    ]


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


def _compact_query_events(events):
    return [
        {
            "sequence": event.get("sequence"),
            "qid": event.get("qid"),
            "entry_count": event.get("entry_count"),
            "ledger_index": event.get("ledger_index"),
        }
        for event in events
    ]


def _coin_observation_view(summary):
    records = (summary.get("coin_observations") or {}).get(
        "observations", []
    )
    return sorted(
        (
            str(record.get("session_id")),
            int(record.get("attempt", 0)),
            record.get("coin_id_digest"),
            record.get("coin_digest"),
            record.get("status"),
        )
        for record in records
    )


def _acs_leader_view(summary):
    leaders = []
    for output in summary.get("acs_outputs", []):
        for event in output.get("trace", []):
            if event.get("phase") != "Weak.FirstRoundCandidate":
                continue
            leaders.append(
                (
                    int(event["coin_index"]),
                    int(event["round"]),
                    int(event["leader"]),
                )
            )
    return sorted(leaders)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Async YOSO TOB baseline smoke logs."
    )
    parser.add_argument("--glob", required=True, help="Log glob, e.g. log/async-*.log")
    parser.add_argument("--expected-nodes", type=int)
    parser.add_argument("--messages-per-node", type=int, default=1)
    parser.add_argument("--target-deliveries", type=int, default=0)
    parser.add_argument("--qid-count", type=int, default=1)
    parser.add_argument(
        "--expect-block-qids",
        default="",
        help="Expected qid instances by ledger height, e.g. A|B,C.",
    )
    parser.add_argument(
        "--expect-vrf",
        action="store_true",
        help="Require Phase-7B VRF/quorum/coin-index contracts.",
    )
    parser.add_argument(
        "--expect-coin-mode",
        choices=("ideal", "threshold-bls"),
        help="Require and validate the selected common-coin implementation.",
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

    expected_pending = None
    if args.expected_nodes is not None:
        expected_pending = next(
            (
                int(summary["expected_pending"])
                for summary in summaries.values()
                if summary.get("expected_pending") is not None
            ),
            args.expected_nodes * args.messages_per_node * args.qid_count,
        )

    pending_counts = {
        node: int(summary.get("pending_count", 0))
        for node, summary in summaries.items()
    }
    rbc_deliveries = {
        node: int(summary.get("message_rbc_deliveries", 0))
        for node, summary in summaries.items()
    }
    pending_refs = {
        node: [entry.get("ref") for entry in summary.get("pending", [])]
        for node, summary in summaries.items()
    }
    first_refs = next(iter(pending_refs.values()), [])
    all_pending_refs_equal = all(refs == first_refs for refs in pending_refs.values())
    enough_pending = True
    if expected_pending is not None:
        enough_pending = all(
            max(pending_counts.get(node, 0), rbc_deliveries.get(node, 0))
            >= expected_pending
            for node in summaries
        )

    cleanup = {
        node: {
            "cleanup_completed": summary.get("cleanup_completed"),
            "active_after": summary.get("active_rbc_instances_after_cleanup"),
        }
        for node, summary in summaries.items()
    }
    ledgers = {
        node: summary.get("ledger", [])
        for node, summary in summaries.items()
    }
    ledger_refs = {
        node: summary.get("ledger_refs", [])
        for node, summary in summaries.items()
    }
    first_ledger = next(iter(ledgers.values()), [])
    first_ledger_refs = next(iter(ledger_refs.values()), [])
    block_qids_by_height = _block_qids_by_height(first_ledger)
    expected_block_qids = _parse_expected_block_qids(args.expect_block_qids)
    all_ledgers_equal = all(ledger == first_ledger for ledger in ledgers.values())
    all_ledger_refs_equal = all(
        refs == first_ledger_refs for refs in ledger_refs.values()
    )
    deliver_counts = {
        node: int(summary.get("deliver_count", 0))
        for node, summary in summaries.items()
    }
    enough_deliveries = all(
        count >= args.target_deliveries for count in deliver_counts.values()
    )
    acs_rbc_instance_counts = {
        node: int(summary.get("acs_rbc_instance_count", 0))
        for node, summary in summaries.items()
    }
    acs_activation_modes = {
        node: summary.get("acs_activation_mode")
        for node, summary in summaries.items()
    }
    acs_collect_thresholds = {
        node: summary.get("acs_collect_threshold")
        for node, summary in summaries.items()
    }
    acs_protocol_quorums = {
        node: summary.get("acs_protocol_quorum")
        for node, summary in summaries.items()
    }
    acs_threshold_fractions = {
        node: summary.get("acs_threshold_fraction")
        for node, summary in summaries.items()
    }
    acs_activation_send_counts = {
        node: int(summary.get("acs_activation_send_count", 0))
        for node, summary in summaries.items()
    }
    acs_stage_collect_counts = {
        node: summary.get("acs_stage_collect_counts", [])
        for node, summary in summaries.items()
    }
    acs_coin_indices = {
        node: summary.get("acs_coin_indices") or {}
        for node, summary in summaries.items()
    }
    coin_modes = {
        node: summary.get("coin_mode", "ideal")
        for node, summary in summaries.items()
    }
    coin_configs = {
        node: {
            "committee": summary.get("coin_committee", []),
            "committee_size": summary.get("coin_committee_size"),
            "c": summary.get("coin_c"),
            "t": summary.get("coin_t"),
            "threshold": summary.get("coin_threshold"),
            "k": summary.get("coin_k"),
            "membership_digest": summary.get("coin_membership_digest"),
            "public_key_digest": summary.get("coin_public_key_digest"),
            "silent_nodes": summary.get("coin_silent_nodes", []),
        }
        for node, summary in summaries.items()
    }
    coin_observation_views = {
        node: _coin_observation_view(summary)
        for node, summary in summaries.items()
    }
    acs_leader_views = {
        node: _acs_leader_view(summary)
        for node, summary in summaries.items()
    }
    protocol_trace_digests = {
        node: summary.get(
            "role_trace_digest",
            summary.get("protocol_trace_digest"),
        )
        for node, summary in summaries.items()
    }
    first_delivery_latencies = {
        node: summary.get("first_delivery_latency_sec")
        for node, summary in summaries.items()
    }
    pending_wait_latencies = {
        node: summary.get("pending_wait_latency_sec")
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
    query_delivery_events = {
        node: _compact_query_events(summary.get("query_delivery_events", []))
        for node, summary in summaries.items()
    }
    first_query_events = next(iter(query_delivery_events.values()), [])
    all_query_delivery_events_equal = all(
        events == first_query_events for events in query_delivery_events.values()
    )
    all_query_delivery_prefixes_comparable = _all_prefix_comparable(
        query_delivery_events.values()
    )
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
    batch_sizes = [
        sum(event["entry_count"] is not None for event in first_query_events
            if event["ledger_index"] == ledger_index)
        for ledger_index in sorted(
            {event["ledger_index"] for event in first_query_events}
        )
    ]

    result = {
        "log_glob": args.glob,
        "log_files": len(paths),
        "summary_nodes": sorted(summaries),
        "summary_count": len(summaries),
        "expected_nodes": args.expected_nodes,
        "messages_per_node": args.messages_per_node,
        "qid_count": args.qid_count,
        "expected_pending": expected_pending,
        "missing_summary_files": [str(Path(path)) for path in missing_summary],
        "pending_counts": pending_counts,
        "rbc_deliveries": rbc_deliveries,
        "all_pending_refs_equal": all_pending_refs_equal,
        "enough_pending": enough_pending,
        "enough_rbc_deliveries_or_pending": enough_pending,
        "pending_refs": first_refs,
        "target_deliveries": args.target_deliveries,
        "deliver_counts": deliver_counts,
        "enough_deliveries": enough_deliveries,
        "all_ledgers_equal": all_ledgers_equal,
        "all_ledger_refs_equal": all_ledger_refs_equal,
        "all_query_delivery_events_equal": all_query_delivery_events_equal,
        "all_query_delivery_prefixes_comparable": (
            all_query_delivery_prefixes_comparable
        ),
        "query_delivery_events": first_query_events,
        "ledger": first_ledger,
        "ledger_refs": first_ledger_refs,
        "block_qids_by_height": block_qids_by_height,
        "expected_block_qids": expected_block_qids,
        "acs_rbc_instance_counts": acs_rbc_instance_counts,
        "acs_activation_modes": acs_activation_modes,
        "acs_collect_thresholds": acs_collect_thresholds,
        "acs_protocol_quorums": acs_protocol_quorums,
        "acs_threshold_fractions": acs_threshold_fractions,
        "acs_activation_send_counts": acs_activation_send_counts,
        "acs_stage_collect_counts": acs_stage_collect_counts,
        "acs_coin_indices": acs_coin_indices,
        "coin_modes": coin_modes,
        "coin_configs": coin_configs,
        "coin_observations": coin_observation_views,
        "acs_coin_leaders": acs_leader_views,
        "ideal_coin": all(
            summary.get("ideal_coin") is True
            for summary in summaries.values()
        ),
        "coin_costs": {
            node: summary.get("coin_cost")
            for node, summary in summaries.items()
        },
        "protocol_trace_sources": {
            node: summary.get(
                "role_trace_source",
                summary.get("protocol_trace_source"),
            )
            for node, summary in summaries.items()
        },
        "protocol_trace_digests": protocol_trace_digests,
        "first_delivery_latencies_sec": first_delivery_latencies,
        "first_delivery_latency_summary_sec": _summary_stats(
            first_delivery_latencies.values()
        ),
        "pending_wait_latencies_sec": pending_wait_latencies,
        "pending_wait_latency_summary_sec": _summary_stats(
            pending_wait_latencies.values()
        ),
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
            "ordered_units": len(batch_sizes),
            "queries_per_ordered_unit": batch_sizes,
            "queries_per_ordered_unit_summary": _summary_stats(batch_sizes),
        },
        "justifier_metrics": {
            node: summary.get("justifier_metrics")
            for node, summary in summaries.items()
        },
        "cleanup": cleanup,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.expected_nodes is not None and len(summaries) != args.expected_nodes:
        raise SystemExit("summary node count mismatch")
    if not enough_pending:
        raise SystemExit("not enough pending messages")
    if not all_pending_refs_equal:
        raise SystemExit("pending refs mismatch")
    if args.target_deliveries > 0 and not enough_deliveries:
        raise SystemExit("not enough ledger deliveries")
    if args.target_deliveries > 0 and not all_ledgers_equal:
        raise SystemExit("ledger mismatch")
    if args.target_deliveries > 0 and not all_ledger_refs_equal:
        raise SystemExit("ledger refs mismatch")
    if args.target_deliveries > 0 and not all_query_delivery_events_equal:
        raise SystemExit("per-query delivery event mismatch")
    if not all_query_delivery_prefixes_comparable:
        raise SystemExit("per-query delivery histories are not prefix comparable")
    if not all_workload_traces_equal:
        raise SystemExit("workload trace mismatch")
    if expected_block_qids is not None and block_qids_by_height != expected_block_qids:
        raise SystemExit(
            f"unexpected block qids by height: {block_qids_by_height}"
        )
    if args.expect_coin_mode is not None:
        if any(
            summary.get("coin_mode_forced_by_replay") is True
            for summary in summaries.values()
        ):
            raise SystemExit("common-coin mode was changed by replay loading")
        if any(
            mode != args.expect_coin_mode for mode in coin_modes.values()
        ):
            raise SystemExit(
                f"common-coin mode mismatch: expected {args.expect_coin_mode}"
            )
        if args.expect_coin_mode == "ideal":
            if any(
                summary.get("ideal_coin") is not True
                or summary.get("coin_cost") != "excluded"
                for summary in summaries.values()
            ):
                raise SystemExit("ideal common-coin accounting mismatch")
        else:
            first_config = next(iter(coin_configs.values()), None)
            if first_config is None or any(
                config != first_config for config in coin_configs.values()
            ):
                raise SystemExit("threshold coin committee configuration mismatch")
            if (
                not first_config.get("committee")
                or int(first_config.get("committee_size") or 0)
                != len(first_config["committee"])
                or int(first_config.get("c") or 0)
                != len(first_config["committee"])
                or not 0
                <= int(first_config.get("t", -1))
                < int(first_config.get("k") or 0)
                <= len(first_config["committee"])
                - int(first_config.get("t", -1))
                or not 1
                <= int(first_config.get("threshold") or 0)
                <= len(first_config["committee"])
                or first_config.get("k") != first_config.get("threshold")
                or not first_config.get("membership_digest")
                or not first_config.get("public_key_digest")
            ):
                raise SystemExit("threshold coin committee metadata is incomplete")
            first_coins = next(iter(coin_observation_views.values()), [])
            if not first_coins or any(
                view != first_coins
                for view in coin_observation_views.values()
            ):
                raise SystemExit("threshold coin observations mismatch")
            if any(record[-1] != "ready" for record in first_coins):
                raise SystemExit("threshold coin call did not complete")
            if any(
                not record[2] or not record[3] for record in first_coins
            ):
                raise SystemExit("threshold coin digest is missing")
            first_leaders = next(iter(acs_leader_views.values()), [])
            if not first_leaders or any(
                view != first_leaders for view in acs_leader_views.values()
            ):
                raise SystemExit("ACS common-coin leader mapping mismatch")
            committee = set(first_config["committee"])
            silent_nodes = set(first_config["silent_nodes"])
            for node, summary in summaries.items():
                if summary.get("ideal_coin") is not False:
                    raise SystemExit(f"node {node}: threshold coin marked ideal")
                if summary.get("coin_cost") != "included":
                    raise SystemExit(f"node {node}: threshold coin cost excluded")
                metrics = summary.get("coin_metrics") or {}
                if metrics.get("closed") is not True:
                    raise SystemExit(f"node {node}: threshold coin not closed")
                if int(metrics.get("pending_coins", -1)) != 0:
                    raise SystemExit(f"node {node}: threshold coin left pending")
                signed = int(metrics.get("shares_signed", 0))
                should_sign = node in committee and node not in silent_nodes
                if should_sign and signed != len(first_coins):
                    raise SystemExit(
                        f"node {node}: expected one share per common coin"
                    )
                if not should_sign and signed != 0:
                    raise SystemExit(
                        f"node {node}: ineligible common-coin signer"
                    )
    if args.expect_vrf:
        if any(mode != "vrf" for mode in acs_activation_modes.values()):
            raise SystemExit("VRF activation mode missing")
        if any(
            summary.get(
                "role_trace_source",
                summary.get("protocol_trace_source"),
            ) != "recorded"
            or summary.get("role_trace_format") != "role-only"
            or not summary.get(
                "role_trace_digest",
                summary.get("protocol_trace_digest"),
            )
            for summary in summaries.values()
        ):
            raise SystemExit("recorded protocol trace missing")
        if len(set(protocol_trace_digests.values())) != 1:
            raise SystemExit("protocol trace digest mismatch")
        coin_views = []
        for node, summary in summaries.items():
            quorum = int(summary.get("quorum_h", 0))
            committee_m = int(summary.get("committee_m", 0))
            if int(summary.get("agreement_t", -1)) != committee_m - quorum:
                raise SystemExit(f"node {node}: agreement_t mismatch")
            if int(summary.get("acs_collect_threshold", 0)) != quorum:
                raise SystemExit(f"node {node}: ACS threshold is not quorum_h")
            if int(summary.get("acs_protocol_quorum", 0)) != quorum:
                raise SystemExit(f"node {node}: protocol quorum mismatch")
            if summary.get("acs_stage_timeout") is None:
                raise SystemExit(f"node {node}: stage timeout missing")
            message_params = summary.get("message_rbc_parameters") or {}
            if message_params.get("require_vrf_proofs") is not True:
                raise SystemExit(
                    f"node {node}: message RBC VRF proofs not required"
                )
            uses = (summary.get("acs_coin_indices") or {}).get("uses", [])
            indices = [int(record["coin_index"]) for record in uses]
            if not indices or len(indices) != len(set(indices)):
                raise SystemExit(
                    f"node {node}: coin indices missing or reused"
                )
            coin_views.append(uses)
            completed_protocol_stages = 0
            for record in summary.get("acs_stage_collect_counts", []):
                if int(record.get("need", 0)) <= 0:
                    continue
                if record.get("status") != "complete":
                    raise SystemExit(
                        f"node {node}: incomplete ACS stage "
                        f"{record.get('stage')}"
                    )
                if int(record.get("proof_rejections", 0)) != 0:
                    raise SystemExit(
                        f"node {node}: activation proof rejection"
                    )
                if int(record.get("collected", 0)) < int(
                    record.get("need", 0)
                ):
                    raise SystemExit(
                        f"node {node}: stage quorum not collected"
                    )
                if int(record.get("verified_selected_senders", 0)) < int(
                    record.get("collected", 0)
                ):
                    raise SystemExit(
                        f"node {node}: unverified selected sender counted"
                    )
                if not str(record.get("stage", "")).startswith("SB_ECHO:"):
                    completed_protocol_stages += 1
                    if int(record.get("need", 0)) != quorum:
                        raise SystemExit(
                            f"node {node}: stage need is not quorum_h"
                        )
            if completed_protocol_stages < 9:
                raise SystemExit(f"node {node}: ACS stage trace incomplete")
        if any(view != coin_views[0] for view in coin_views[1:]):
            raise SystemExit("coin index mapping mismatch")


if __name__ == "__main__":
    main()
