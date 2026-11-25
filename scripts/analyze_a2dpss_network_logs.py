"""Validate the formal Algorithm-4 A²DPSS network smoke gate."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from beaver.experiment_trace import (
    aggregate_a2dpss_computation_metrics,
    validate_a2dpss_computation_metrics,
    validate_a2dpss_handover_metrics,
)
from scripts.analyze_yoso_dpss_network_logs import (
    _parse_nodes,
    _qid_kind,
    _read_last_summary,
)
from scripts.run_dag_tob_phase3 import _transport_family_totals


def _events(summary: dict, kind: str) -> list[dict]:
    return [
        event
        for event in summary.get("a2dpss_events", [])
        if event.get("event") == kind
    ]


def _dag_agreement_parameter_view(summary: dict, node: int):
    """Validate and normalize the Phase-1 DAG agreement parameters.

    A missing schema is accepted so the analyzer can still read the frozen
    Phase-0 baseline logs.  Once the schema is present, every relationship is
    checked fail closed before the values are included in analyzer output.
    """

    agreement_schema = summary.get("dag_agreement_parameters_schema")
    if agreement_schema is None:
        return None
    if agreement_schema != 1:
        raise SystemExit(
            f"node {node} has unsupported DAG agreement parameter schema"
        )

    M = int(summary.get("M", 0))
    committee_m = int(summary.get("committee_m", 0))
    quorum_h = int(summary.get("quorum_h", 0))
    agreement_t = int(summary.get("agreement_t", -1))
    coverage_threshold = int(summary.get("coverage_threshold", -1))
    if not 0 < quorum_h <= committee_m <= M:
        raise SystemExit(
            f"node {node} has invalid DAG committee/quorum parameters"
        )
    if agreement_t != committee_m - quorum_h:
        raise SystemExit(
            f"node {node} agreement_t does not equal "
            "committee_m - quorum_h"
        )
    if 2 * agreement_t >= quorum_h:
        raise SystemExit(
            f"node {node} violates the DAG Byzantine quorum bound"
        )
    if coverage_threshold != agreement_t + 1:
        raise SystemExit(
            f"node {node} has an invalid qid coverage threshold"
        )
    return {
        "committee_m": committee_m,
        "quorum_h": quorum_h,
        "agreement_t": agreement_t,
        "coverage_threshold": coverage_threshold,
    }


def _validate_dag_selectblock_metrics(summary: dict, node: int):
    """Fail closed on causal-coverage SelectBlock observability."""

    metrics = summary.get("dag_selectblock_metrics")
    if metrics is None:
        return None
    if (
        metrics.get("schema") != "dag-selectblock-metrics/v1"
        or metrics.get("applicable") is not True
    ):
        raise SystemExit(f"node {node} has invalid DAG SelectBlock metrics schema")

    agreement_t = int(summary.get("agreement_t", -1))
    coverage_threshold = int(summary.get("coverage_threshold", -1))
    if int(metrics.get("agreement_t", -1)) != agreement_t:
        raise SystemExit(f"node {node} SelectBlock agreement_t mismatch")
    if int(metrics.get("coverage_threshold", -1)) != coverage_threshold:
        raise SystemExit(f"node {node} SelectBlock coverage threshold mismatch")

    coverage_records = metrics.get("coverage_records") or []
    coverage_by_qid = {}
    for record in coverage_records:
        qid = str(record.get("qid", ""))
        cover_round = int(record.get("cover_round", 0))
        witness_count = int(record.get("witness_count", 0))
        if not qid or qid in coverage_by_qid:
            raise SystemExit(f"node {node} has duplicate/empty coverage record")
        if cover_round <= 0 or witness_count < coverage_threshold:
            raise SystemExit(f"node {node} has invalid coverage witness record")
        coverage_by_qid[qid] = record

    coverage_snapshot = summary.get("dag_qid_causal_coverage") or {}
    if coverage_snapshot:
        if coverage_snapshot.get("schema") != "dag-qid-causal-coverage/v1":
            raise SystemExit(f"node {node} has invalid coverage snapshot schema")
        if int(coverage_snapshot.get("coverage_threshold", -1)) != (
            coverage_threshold
        ):
            raise SystemExit(f"node {node} coverage snapshot threshold mismatch")
        if coverage_snapshot.get("coverage_records", []) != coverage_records:
            raise SystemExit(f"node {node} coverage records disagree")

    per_vertex_records = metrics.get("per_vertex_records") or []
    totals = {
        "candidate_qids": 0,
        "evaluated_qids": 0,
        "selected_qids": 0,
        "skipped_delivered_qids": 0,
        "skipped_causally_covered_qids": 0,
    }
    evaluated_sequence = []
    for record in per_vertex_records:
        round_id = int(record.get("round", 0))
        if round_id <= 0:
            raise SystemExit(f"node {node} has invalid SelectBlock round")
        lists = {}
        for field in (
            "candidate_qids",
            "delivered_exclusions",
            "causal_coverage_exclusions",
            "evaluated_qids",
            "selected_qids",
        ):
            values = list(record.get(field) or [])
            if values != sorted(set(values)):
                raise SystemExit(
                    f"node {node} SelectBlock {field} is not canonical"
                )
            lists[field] = values

        candidate = set(lists["candidate_qids"])
        delivered = set(lists["delivered_exclusions"])
        causal = set(lists["causal_coverage_exclusions"])
        evaluated = set(lists["evaluated_qids"])
        selected = set(lists["selected_qids"])
        if (
            delivered & causal
            or delivered & evaluated
            or causal & evaluated
            or candidate != delivered | causal | evaluated
            or not selected.issubset(evaluated)
        ):
            raise SystemExit(f"node {node} has inconsistent SelectBlock partition")
        for qid in causal:
            coverage = coverage_by_qid.get(qid)
            if coverage is None or int(coverage["cover_round"]) >= round_id:
                raise SystemExit(
                    f"node {node} causal skip lacks a prior-round witness"
                )

        totals["candidate_qids"] += len(candidate)
        totals["evaluated_qids"] += len(evaluated)
        totals["selected_qids"] += len(selected)
        totals["skipped_delivered_qids"] += len(delivered)
        totals["skipped_causally_covered_qids"] += len(causal)
        evaluated_sequence.extend(lists["evaluated_qids"])

    if int(metrics.get("created_vertices", -1)) != len(per_vertex_records):
        raise SystemExit(f"node {node} SelectBlock vertex total mismatch")
    for field, expected in totals.items():
        if int(metrics.get(field, -1)) != expected:
            raise SystemExit(f"node {node} SelectBlock {field} total mismatch")

    justifier_metrics = summary.get("justifier_metrics") or {}
    select_metrics = justifier_metrics.get("select") or {}
    predicate_records = select_metrics.get("records") or []
    predicate_qids = [str(record.get("qid", "")) for record in predicate_records]
    if (
        int(select_metrics.get("invocations", -1)) != len(predicate_records)
        or predicate_qids != evaluated_sequence
    ):
        raise SystemExit(
            f"node {node} evaluated qids do not match proposer predicate records"
        )

    accepted_verify_qids = {
        str(record.get("qid", ""))
        for record in (justifier_metrics.get("verify") or {}).get("records", [])
        if record.get("accepted") is True
    }
    delivered_qids = {
        str(event.get("qid", ""))
        for event in summary.get("protocol_delivery_events", [])
    }
    if not delivered_qids.issubset(accepted_verify_qids):
        raise SystemExit(f"node {node} delivery lacks committed-order verification")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Algorithm-4 A2DPSS network smoke logs."
    )
    parser.add_argument("--glob", required=True)
    parser.add_argument("--expected-nodes", type=int, required=True)
    parser.add_argument("--min-entry-count", type=int, default=2)
    parser.add_argument("--expected-hops", type=int, default=1)
    parser.add_argument("--successor-nodes", default="0,1,2")
    parser.add_argument("--non-successor-nodes", default="3")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summaries = {}
    for path in sorted(glob.glob(args.glob)):
        summary = _read_last_summary(path)
        if summary is not None:
            summaries[int(summary["node"])] = summary
    if len(summaries) != args.expected_nodes:
        raise SystemExit(
            "summary node count mismatch: "
            f"{len(summaries)} != {args.expected_nodes}"
        )

    public_events = {}
    reconstructed = set()
    new_share = set()
    entry_counts = {}
    metrics = {}
    successors = _parse_nodes(args.successor_nodes) or set()
    non_successors = _parse_nodes(args.non_successor_nodes) or set()
    handover_timelines = {}
    successor_handover_latencies = {}
    computation_timelines = {}
    agreement_parameters = {}
    selectblock_metrics = {}
    transport_metrics = {}
    for node, summary in summaries.items():
        if (
            summary.get("handover_mode") != "a2dpss"
            or summary.get("handover_algorithm") != "Algorithm 4"
            or summary.get("a2dpss_smoke") is not True
        ):
            raise SystemExit(
                f"node {node} did not identify Algorithm-4 A2DPSS"
            )
        agreement_parameter_view = _dag_agreement_parameter_view(
            summary, node
        )
        if agreement_parameter_view is not None:
            agreement_parameters[node] = agreement_parameter_view
        selectblock_view = _validate_dag_selectblock_metrics(summary, node)
        if selectblock_view is not None:
            selectblock_metrics[node] = selectblock_view
        if not summary.get("cleanup_completed"):
            raise SystemExit(
                f"node {node} did not complete cleanup"
            )
        if int(
            summary.get(
                "active_a2dpss_rbc_instances_after_cleanup",
                -1,
            )
        ) != 0:
            raise SystemExit(
                f"node {node} retained A2DPSS RBC instances"
            )
        if int(
            summary.get("a2dpss_requested_hops", -1)
        ) != args.expected_hops:
            raise SystemExit(
                f"node {node} requested-hop count mismatch"
            )
        if int(
            summary.get("a2dpss_scheduled_hops", -1)
        ) != args.expected_hops:
            raise SystemExit(
                f"node {node} scheduled-hop count mismatch"
            )
        if int(
            summary.get(
                "registered_a2dpss_contexts_after_cleanup", -1
            )
        ) != 0:
            raise SystemExit(
                f"node {node} retained justifier contexts"
            )
        if int(
            summary.get(
                "registered_a2dpss_reconstruction_contexts_after_cleanup",
                -1,
            )
        ) != 0:
            raise SystemExit(
                f"node {node} retained reconstruction contexts"
            )

        deliveries = [
            event
            for event in summary.get(
                "protocol_delivery_events",
                [],
            )
            if _qid_kind(str(event.get("qid", "")))
            == "a2dpss"
        ]
        if len(deliveries) != args.expected_hops:
            raise SystemExit(
                f"node {node} expected {args.expected_hops} "
                f"A2DPSS deliveries"
            )
        node_entry_counts = [
            len(event.get("entries", [])) for event in deliveries
        ]
        entry_counts[node] = node_entry_counts
        if any(
            count < args.min_entry_count
            for count in node_entry_counts
        ):
            raise SystemExit(
                f"node {node} has undersized A2DPSS T: "
                f"{node_entry_counts}"
            )

        finalized = sorted(
            _events(summary, "public_finalized"),
            key=lambda event: int(event["transfer_seq"]),
        )
        if len(finalized) != args.expected_hops:
            raise SystemExit(
                f"node {node} finalized {len(finalized)} of "
                f"{args.expected_hops} A2DPSS hops"
            )
        if (
            finalized[0].get("algorithm") != "Algorithm 4"
            or finalized[0].get("protocol") != "a2dpss"
        ):
            raise SystemExit(
                f"node {node} emitted ambiguous A2DPSS event labels"
            )
        public_events[node] = finalized
        handover = summary.get("a2dpss_handover_metrics")
        handover_timelines[node] = handover
        for failure in validate_a2dpss_handover_metrics(
            handover,
            protocol="dag-tob",
            node=node,
            qid=finalized[0]["qid"],
            is_successor=node in successors,
        ):
            raise SystemExit(
                f"node {node} handover metrics {failure}"
            )
        if node in successors:
            latency = (
                (handover.get("durations_sec") or {}).get(
                    "handover_latency"
                )
                if isinstance(handover, dict)
                else None
            )
            if latency is not None:
                successor_handover_latencies[node] = float(latency)
        computation = summary.get("a2dpss_computation_metrics")
        computation_timelines[node] = computation
        failures = validate_a2dpss_computation_metrics(
            computation,
            protocol="dag-tob",
            node=node,
        )
        if failures:
            raise SystemExit(
                f"node {node} computation metrics: "
                + "; ".join(failures)
            )
        if len(
            _events(summary, "reconstructed")
        ) == args.expected_hops:
            reconstructed.add(node)
        if len(_events(summary, "new_share")) == args.expected_hops:
            new_share.add(node)

        delivered_after_cleanup = summary.get(
            "delivered_a2dpss_pool_entry_counts_after_context_cleanup",
            {},
        )
        if len(delivered_after_cleanup) != args.expected_hops or any(
            int(count) < args.min_entry_count
            for count in delivered_after_cleanup.values()
        ):
            raise SystemExit(
                f"node {node} context cleanup damaged DeliveredPool evidence"
            )
        lifecycle = summary.get("a2dpss_lifecycle") or {}
        if lifecycle.get("active_qid") is not None:
            raise SystemExit(
                f"node {node} retained an active lifecycle context"
            )
        if len(
            lifecycle.get("advanced_context_digests", [])
        ) != max(0, args.expected_hops - 1):
            raise SystemExit(
                f"node {node} lifecycle advance count mismatch"
            )

        node_metrics = summary.get("a2dpss_rbc_metrics") or {}
        metrics[node] = node_metrics
        transport_metrics[node] = _transport_family_totals(
            summary.get("transport_bytes_by_tag") or {},
            total_bytes=int(summary.get("bytes_sent", 0)),
        )
        for field in (
            "delivered_contributions",
            "delivered_contribution_bytes",
            "delivered_pvss_bytes",
        ):
            if int(node_metrics.get(field, 0)) <= 0:
                raise SystemExit(
                    f"node {node} missing A2DPSS metric {field}"
                )

    if agreement_parameters:
        if len(agreement_parameters) != len(summaries):
            raise SystemExit(
                "DAG agreement parameters are missing from some nodes"
            )
        parameter_views = {
            json.dumps(parameters, sort_keys=True)
            for parameters in agreement_parameters.values()
        }
        if len(parameter_views) != 1:
            raise SystemExit(
                "DAG agreement parameters differ across nodes"
            )
    if selectblock_metrics and len(selectblock_metrics) != len(summaries):
        raise SystemExit("DAG SelectBlock metrics are missing from some nodes")

    first_events = next(iter(public_events.values()))
    for hop_index in range(args.expected_hops):
        expected_event = first_events[hop_index]
        if any(
            events[hop_index] != expected_event
            for events in public_events.values()
        ):
            raise SystemExit(
                "observers derived different A2DPSS T/vc' at "
                f"hop {hop_index}"
            )
        if len(
            expected_event.get(
                "accepted_contribution_digests", []
            )
        ) < args.min_entry_count:
            raise SystemExit(
                f"A2DPSS hop {hop_index} does not record threshold T"
            )

    transfer_sequences = [
        int(event["transfer_seq"]) for event in first_events
    ]
    if transfer_sequences != list(
        range(
            transfer_sequences[0],
            transfer_sequences[0] + args.expected_hops,
        )
    ):
        raise SystemExit(
            "A2DPSS transfer sequence is not monotonic and consecutive"
        )
    for hop_index in range(1, args.expected_hops):
        previous = first_events[hop_index - 1]
        current = first_events[hop_index]
        if current["context_old_commitment_vector"] != (
            previous["successor_commitment_vector"]
        ):
            raise SystemExit(
                f"hop {hop_index} did not anchor on prior protocol vc'"
            )
        if current["successor_commitment_vector"][0] != (
            previous["successor_commitment_vector"][0]
        ):
            raise SystemExit(
                f"hop {hop_index} did not preserve secret commitment"
            )

    if args.expected_hops > 1:
        successors_for_holders = (
            _parse_nodes(args.successor_nodes) or set()
        )
        for node, summary in summaries.items():
            holder_events = sorted(
                _events(summary, "holder_broadcast"),
                key=lambda event: int(event["transfer_seq"]),
            )
            if node in successors_for_holders:
                if len(holder_events) != args.expected_hops:
                    raise SystemExit(
                        f"node {node} holder broadcast count mismatch"
                    )
                for event in holder_events[1:]:
                    if not event.get(
                        "holder_state_source", ""
                    ).startswith("successor-result:"):
                        raise SystemExit(
                            f"node {node} reused config share after "
                            "first handover"
                        )

    if successors - reconstructed:
        raise SystemExit(
            "successors did not reconstruct: "
            f"{sorted(successors - reconstructed)}"
        )
    if successors - new_share:
        raise SystemExit(
            "successor callbacks missing: "
            f"{sorted(successors - new_share)}"
        )
    unexpected = non_successors & (
        reconstructed | new_share
    )
    if unexpected:
        raise SystemExit(
            "non-successors produced private results: "
            f"{sorted(unexpected)}"
        )

    transport_totals = {
        family: sum(node_metrics[family] for node_metrics in transport_metrics.values())
        for family in ("yoso_rbc", "dag_protocol", "common_coin", "total")
    }
    result = {
                "summary_nodes": sorted(summaries),
                "algorithm": "Algorithm 4",
                "protocol": "a2dpss",
                "all_public_results_equal": True,
                "expected_hops": args.expected_hops,
                "transfer_sequences": transfer_sequences,
                "public_results": first_events,
                "protocol_output_anchor_chain": True,
                "entry_counts": entry_counts,
                "reconstructed_nodes": sorted(reconstructed),
                "non_successor_nodes": sorted(non_successors),
                "cleanup_active_a2dpss_rbc": 0,
                "a2dpss_rbc_metrics": metrics,
                "network_communication_metrics": {
                    "application_payload_bytes": sum(
                        int(node_metrics.get("outbound_contribution_bytes", 0))
                        for node_metrics in metrics.values()
                    ),
                    "cryptographic_proof_bytes": sum(
                        int(node_metrics.get("outbound_pvss_bytes", 0))
                        for node_metrics in metrics.values()
                    ),
                    "transport_bytes_by_family": transport_totals,
                    "transport_bytes_by_node": transport_metrics,
                },
                "handover_latency_sec": max(
                    successor_handover_latencies.values()
                ),
                "successor_handover_latencies_sec": (
                    successor_handover_latencies
                ),
                "handover_metrics": handover_timelines,
                "computation_metrics": computation_timelines,
                "network_computation_metrics": (
                    aggregate_a2dpss_computation_metrics(
                        computation_timelines
                    )
                ),
                "dag_agreement_parameters": agreement_parameters,
                "dag_selectblock_metrics": selectblock_metrics,
            }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
