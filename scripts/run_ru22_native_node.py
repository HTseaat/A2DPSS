"""Run one dedicated-verifier RU22 node, without DAG/RBC/coin processes."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from beaver.baselines.ru22_native_runner import run_native_node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--node", type=int, required=True)
    parser.add_argument("--peers", required=True, help="JSON file containing host:port strings")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", help="node config with authenticated chunk lookup tables")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--startup-timeout", type=float, default=60)
    args = parser.parse_args()
    peers = json.loads(Path(args.peers).read_text())
    preload = None
    if args.config:
        from beaver.config import HbmpcConfig
        config = json.loads(Path(args.config).read_text())
        if config["my_id"] != args.node or config["peers"] != peers:
            raise ValueError("native node config/peer mismatch")
        sys.argv = [sys.argv[0], "-d", "-f", args.config, "-time", "0"]
        HbmpcConfig.load_config()
        preload = HbmpcConfig.lookup_table_preload_seconds
    result = asyncio.run(run_native_node(fixture=args.fixture, node=args.node, peers=peers,
        timeout=args.timeout, startup_timeout=args.startup_timeout))
    result["lookup_table_preload_seconds"] = preload
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"yoso_adpss_metrics": result}, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
