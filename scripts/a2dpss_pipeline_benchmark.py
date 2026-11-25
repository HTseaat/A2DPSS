#!/usr/bin/env python3
"""Prepare, run, validate and compare the A2DPSS DAG/ACS pipeline experiment."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare", help="freeze one workload, role trace and real A2DPSS fixture"
    )
    prepare.add_argument("--config-dir", required=True)
    prepare.add_argument(
        "--peers", help="JSON list of M host:port endpoints; defaults to config peers"
    )
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--trial-id", required=True)
    prepare.add_argument(
        "--committee-seed",
        help="fixed role-trace seed; defaults to trial-id",
    )
    prepare.add_argument("--initial-queries", type=int, default=1)
    prepare.add_argument("--late-queries", type=int, default=1)
    prepare.add_argument(
        "--selection-mode",
        choices=("fixed-size", "bernoulli"),
        default="fixed-size",
    )
    prepare.add_argument("--timeout", type=float, default=900)
    prepare.add_argument("--startup-timeout", type=float, default=300)
    prepare.add_argument("--coin-timeout", type=float, default=120)
    prepare.add_argument("--rounds", type=int, default=32)
    prepare.add_argument("--lookahead-rounds", type=int, default=4)
    prepare.add_argument("--acs-max-rounds", type=int, default=16)
    prepare.add_argument("--acs-stage-timeout", type=float, default=120)

    export = commands.add_parser(
        "export", help="copy public setup plus only this host's node configs"
    )
    export.add_argument("--bundle", required=True)
    export.add_argument("--nodes", type=int, nargs="+", required=True)
    export.add_argument("--output", required=True)

    run = commands.add_parser(
        "run", help="launch one ordering protocol for this host's nodes"
    )
    run.add_argument("--bundle", required=True)
    run.add_argument("--protocol", choices=("dag", "acs"), required=True)
    run.add_argument("--nodes", type=int, nargs="+", required=True)
    run.add_argument("--output", required=True)

    collect = commands.add_parser(
        "collect", help="fail-closed validation and aggregation for one protocol"
    )
    collect.add_argument("--bundle", required=True)
    collect.add_argument("--protocol", choices=("dag", "acs"), required=True)
    collect.add_argument("--results", required=True)
    collect.add_argument("--output", required=True)

    compare = commands.add_parser(
        "compare", help="compare validated DAG and ACS pipeline results"
    )
    compare.add_argument("--bundle", required=True)
    compare.add_argument("--dag-results", required=True)
    compare.add_argument("--acs-results", required=True)
    compare.add_argument("--output", required=True)

    worker = commands.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--bundle", required=True)
    worker.add_argument("--protocol", choices=("dag", "acs"), required=True)
    worker.add_argument("--node", type=int, required=True)
    worker.add_argument("--output", required=True)
    return root


async def launch(args) -> None:
    from beaver.a2dpss_pipeline_benchmark import load_bundle, validate_nodes
    from beaver.a2dpss_benchmark import write

    manifest = load_bundle(args.bundle)
    validate_nodes(args.nodes, manifest)
    loop = asyncio.get_running_loop()
    current = asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM, current.cancel)
    directory = Path(args.output) / args.protocol
    directory.mkdir(parents=True, exist_ok=True)
    for node in args.nodes:
        node_dir = Path(args.bundle) / f"nodes/{node}"
        if not (node_dir / "config.json").is_file():
            raise ValueError(
                f"missing config for node {node}; export this host's nodes first"
            )
        if (node_dir / f"{args.protocol}.used").exists():
            raise ValueError(f"node {node}/{args.protocol} already used")
        if (directory / f"node-{node}.json").exists():
            raise ValueError(f"result for node {node}/{args.protocol} already exists")

    processes, streams = {}, []
    failure = None
    try:
        for node in args.nodes:
            stream = (directory / f"node-{node}.log").open("xb")
            streams.append(stream)
            processes[node] = await asyncio.create_subprocess_exec(
                sys.executable,
                "-u",
                "-m",
                "scripts.a2dpss_pipeline_benchmark",
                "worker",
                "--bundle",
                args.bundle,
                "--protocol",
                args.protocol,
                "--node",
                str(node),
                "--output",
                args.output,
                cwd=str(ROOT),
                stdout=stream,
                stderr=stream,
            )
        print(
            f"Started {args.protocol}: nodes {args.nodes}; logs: {directory}",
            flush=True,
        )

        async def monitor():
            async def check(node, process):
                code = await process.wait()
                if code:
                    raise RuntimeError(
                        f"node {node} exited {code}; see "
                        f"{directory / f'node-{node}.log'}"
                    )

            await asyncio.gather(
                *(check(node, process) for node, process in processes.items())
            )

        deadline = manifest["startup_timeout"] + 3 * manifest["timeout"] + 30
        await asyncio.wait_for(monitor(), timeout=deadline)
    except BaseException as error:
        failure = error
        raise
    finally:
        for process in processes.values():
            if process.returncode is None:
                process.kill()
        await asyncio.gather(
            *(process.wait() for process in processes.values()),
            return_exceptions=True,
        )
        for stream in streams:
            stream.close()
        if failure is not None:
            for node in args.nodes:
                path = directory / f"node-{node}.json"
                if not path.exists():
                    write(
                        path,
                        {
                            "schema": manifest["schema"],
                            "status": (
                                "timeout"
                                if isinstance(failure, asyncio.TimeoutError)
                                else "failed"
                            ),
                            "protocol": args.protocol,
                            "node": node,
                            "bundle_digest": manifest["bundle_digest"],
                            "trial_id": manifest["trial_id"],
                            "reason": (
                                f"supervisor: {type(failure).__name__}: {failure}"
                            ),
                        },
                    )
    print(f"Completed {args.protocol}: local nodes {args.nodes}", flush=True)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    for field in (
        "config_dir",
        "peers",
        "bundle",
        "output",
        "results",
        "dag_results",
        "acs_results",
    ):
        value = getattr(args, field, None)
        if value:
            setattr(args, field, str(Path(value).resolve()))
    os.chdir(ROOT)
    (ROOT / "log").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    from beaver.a2dpss_benchmark import write
    from beaver.a2dpss_pipeline_benchmark import (
        collect,
        compare,
        export_nodes,
        prepare,
    )

    if args.command == "prepare":
        args.committee_seed = args.committee_seed or args.trial_id
        result = prepare(args)
        print(
            json.dumps(
                {
                    "bundle": args.output,
                    "trial_id": result["trial_id"],
                    "bundle_digest": result["bundle_digest"],
                    "parameters": result["parameters"],
                    "initial_queries": result["initial_query_count"],
                    "late_queries": result["late_query_count"],
                },
                indent=2,
            )
        )
    elif args.command == "export":
        print(export_nodes(args.bundle, args.output, args.nodes))
    elif args.command == "run":
        asyncio.run(launch(args))
    elif args.command == "worker":
        from beaver.a2dpss_pipeline_benchmark_runtime import worker

        asyncio.run(worker(args.bundle, args.output, args.protocol, args.node))
    elif args.command == "collect":
        result = collect(args.bundle, args.results, args.protocol)
        write(args.output, result)
        print(json.dumps({k: v for k, v in result.items() if k != "per_node"}, indent=2))
        return 0 if result["status"] == "ok" else 1
    else:
        result = compare(args.bundle, args.dag_results, args.acs_results)
        write(args.output, result)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "ok" else 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (
        asyncio.TimeoutError,
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
