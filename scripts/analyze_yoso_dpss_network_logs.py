from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path


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


def _parse_nodes(raw: str | None) -> set[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return {
        int(item)
        for item in raw.replace(" ", "").split(",")
        if item != ""
    }


def _qid_kind(qid_raw: str) -> str:
    try:
        data = json.loads(qid_raw)
    except json.JSONDecodeError:
        return ""
    return str(data.get("kind", ""))


def _delivery_qids(summary: dict) -> list[str]:
    return [
        str(event.get("qid", ""))
        for event in summary.get("protocol_delivery_events", [])
        if event.get("qid")
    ]


def _event_kinds(summary: dict) -> list[str]:
    return [str(event.get("event", "")) for event in summary.get("dpss_events", [])]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze YOSO DPSS network smoke logs."
    )
    parser.add_argument("--glob", required=True, help="Log glob, e.g. log/dpss-*.log")
    parser.add_argument("--expected-nodes", type=int, required=True)
    parser.add_argument("--target-deliveries", type=int, default=1)
    parser.add_argument(
        "--min-entry-count",
        "--min-block-size",
        dest="min_entry_count",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--successor-nodes",
        default=None,
        help="Comma-separated nodes expected to reconstruct, e.g. 0,1,2.",
    )
    parser.add_argument(
        "--non-successor-nodes",
        default=None,
        help="Comma-separated nodes expected not to reconstruct.",
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

    if len(summaries) != args.expected_nodes:
        raise SystemExit(
            f"summary node count mismatch: {len(summaries)} != {args.expected_nodes}"
        )
    if missing_summary:
        raise SystemExit(
            "missing summary files: "
            + ",".join(str(Path(path)) for path in missing_summary)
        )

    delivery_qids = {
        node: _delivery_qids(summary)
        for node, summary in summaries.items()
    }
    delivery_events = {
        node: summary.get("protocol_delivery_events", [])
        for node, summary in summaries.items()
    }
    first_delivery_events = next(iter(delivery_events.values()))
    if any(
        events != first_delivery_events
        for events in delivery_events.values()
    ):
        raise SystemExit("DPSS per-query delivery event mismatch")
    first_qids = next(iter(delivery_qids.values()))
    if any(qids != first_qids for qids in delivery_qids.values()):
        raise SystemExit("delivery qid sequence mismatch")
    if len(first_qids) < args.target_deliveries:
        raise SystemExit("not enough DPSS ledger deliveries")
    if not first_qids:
        raise SystemExit("empty DPSS delivery sequence")
    if _qid_kind(first_qids[0]) != "dpss":
        raise SystemExit(
            f"first delivery is not a legacy transcript-only DPSS qid: {first_qids[0]}"
        )

    entry_counts = {}
    for node, summary in summaries.items():
        if summary.get("legacy_dpss_smoke") is not True:
            raise SystemExit(
                f"node {node} did not identify the run as legacy compatibility DPSS"
            )
        if (
            summary.get("handover_mode") != "legacy-dpss"
            or summary.get("handover_algorithm")
            != "legacy transcript-only compatibility"
        ):
            raise SystemExit(
                f"node {node} used ambiguous legacy protocol labels"
            )
        if not summary.get("dpss_smoke"):
            raise SystemExit(f"node {node} did not run with dpss_smoke=true")
        deliver_count = int(summary.get("deliver_count", 0))
        if deliver_count < args.target_deliveries:
            raise SystemExit(f"node {node} insufficient deliveries: {deliver_count}")
        events = summary.get("protocol_delivery_events", [])
        if not events:
            raise SystemExit(f"node {node} has no protocol delivery events")
        if (
            int(summary.get("delivery_event_count", 0)) != deliver_count
            or len(events) != deliver_count
        ):
            raise SystemExit(f"node {node} delivery event count mismatch")
        entry_count = len(events[0].get("entries", []))
        entry_counts[node] = entry_count
        if entry_count < args.min_entry_count:
            raise SystemExit(
                f"node {node} DPSS entry set too small: "
                f"{entry_count} < {args.min_entry_count}"
            )

    successor_nodes = _parse_nodes(args.successor_nodes)
    non_successor_nodes = _parse_nodes(args.non_successor_nodes)
    reconstructed_nodes = {
        node
        for node, summary in summaries.items()
        if "reconstructed" in _event_kinds(summary)
    }
    new_share_nodes = {
        node
        for node, summary in summaries.items()
        if "new_share" in _event_kinds(summary)
    }

    if successor_nodes is not None:
        missing = successor_nodes - reconstructed_nodes
        if missing:
            raise SystemExit(f"successor nodes did not reconstruct: {sorted(missing)}")
        missing_callback = successor_nodes - new_share_nodes
        if missing_callback:
            raise SystemExit(
                f"successor nodes did not call on_new_share: {sorted(missing_callback)}"
            )
    elif not reconstructed_nodes:
        raise SystemExit("no successor node reconstructed")

    if non_successor_nodes is not None:
        unexpected = non_successor_nodes & (reconstructed_nodes | new_share_nodes)
        if unexpected:
            raise SystemExit(
                f"non-successor nodes produced DPSS shares: {sorted(unexpected)}"
            )

    result = {
        "log_glob": args.glob,
        "summary_nodes": sorted(summaries),
        "target_deliveries": args.target_deliveries,
        "all_delivery_events_equal": True,
        "dpss_delivery_qids": first_qids,
        "entry_counts": entry_counts,
        "successor_nodes_expected": (
            sorted(successor_nodes) if successor_nodes is not None else None
        ),
        "non_successor_nodes_expected": (
            sorted(non_successor_nodes) if non_successor_nodes is not None else None
        ),
        "reconstructed_nodes": sorted(reconstructed_nodes),
        "new_share_nodes": sorted(new_share_nodes),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
