import argparse
import json
import logging

from beaver.dag_tob import simulate_one_wave


def main():
    parser = argparse.ArgumentParser(
        description="Run a local Phase-1 smoke test for ATCaaS DAG-TOB."
    )
    parser.add_argument("--M", type=int, default=4)
    parser.add_argument("--quorum-h", type=int, default=3)
    parser.add_argument("--agreement-t", type=int, default=1)
    parser.add_argument("--epoch", default="phase1-smoke")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    nodes = simulate_one_wave(
        M=args.M,
        quorum_h=args.quorum_h,
        agreement_t=args.agreement_t,
        epoch=args.epoch,
    )

    ledgers = [node.snapshot_ledger() for node in nodes]
    ordered = [node.snapshot_ordered_vertices() for node in nodes]

    first_ledger = ledgers[0]
    first_ordered = ordered[0]
    all_ledgers_equal = all(item == first_ledger for item in ledgers)
    all_ordered_equal = all(item == first_ordered for item in ordered)
    delivered_counts = [len(node.ledger) for node in nodes]
    ordered_counts = [len(node.ordered_vertices) for node in nodes]

    result = {
        "M": args.M,
        "quorum_h": args.quorum_h,
        "agreement_t": args.agreement_t,
        "coverage_threshold": args.agreement_t + 1,
        "all_ledgers_equal": all_ledgers_equal,
        "all_ordered_vertices_equal": all_ordered_equal,
        "delivered_counts": delivered_counts,
        "ordered_vertex_counts": ordered_counts,
        "ledger": first_ledger,
        "ordered_vertices": first_ordered,
    }
    print(json.dumps(result, indent=2))

    if not all_ledgers_equal:
        raise SystemExit("ledger mismatch")
    if not all_ordered_equal:
        raise SystemExit("ordered vertex mismatch")
    if not first_ledger:
        raise SystemExit("no block delivered")


if __name__ == "__main__":
    main()
