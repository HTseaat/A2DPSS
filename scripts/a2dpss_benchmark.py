#!/usr/bin/env python3
"""Prepare, run, export, collect and compare A²DPSS consensus backends."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
from pathlib import Path
import sys

# Support both python -m scripts.a2dpss_benchmark and an absolute script path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="create one fresh bundle for all three backends")
    prepare.add_argument("--config-dir", required=True, help="Phase-6 local.N.json and threshold coin manifest")
    prepare.add_argument("--peers", help="JSON list of M host:port endpoints; defaults to config peers")
    prepare.add_argument("--output", required=True, help="new bundle directory")
    prepare.add_argument("--trial-id", required=True)
    prepare.add_argument("--view", choices=("live", "paired"), default="live")
    prepare.add_argument("--selection-mode", choices=("fixed-size", "bernoulli"), default="fixed-size")
    prepare.add_argument("--timeout", type=float, default=900, help="online protocol timeout, seconds")
    prepare.add_argument("--startup-timeout", type=float, default=300, help="wait for every node to be ready")
    prepare.add_argument("--coin-timeout", type=float, default=120)
    prepare.add_argument("--rounds", type=int, default=32)
    prepare.add_argument("--max-views", type=int, default=16)
    export = commands.add_parser("export", help="copy public setup and only this host's private node configs")
    export.add_argument("--bundle", required=True)
    export.add_argument("--nodes", required=True, type=int, nargs="+")
    export.add_argument("--output", required=True)
    run = commands.add_parser("run", help="launch only the selected protocol on this host")
    run.add_argument("--bundle", required=True)
    run.add_argument("--protocol", required=True, choices=("dag", "acs", "algorand"))
    run.add_argument("--nodes", required=True, type=int, nargs="+", help="node ids hosted on this machine")
    run.add_argument("--output", required=True, help="results directory; each protocol has its own subdirectory")
    collect = commands.add_parser("collect", help="validate collected public results for one protocol")
    collect.add_argument("--bundle", required=True)
    collect.add_argument("--protocol", required=True, choices=("dag", "acs", "algorand"))
    collect.add_argument("--results", required=True)
    collect.add_argument("--output", required=True, help="new aggregate JSON file")
    compare = commands.add_parser("compare", help="revalidate all three backends for the same trial bundle")
    compare.add_argument("--bundle", required=True)
    for protocol in ("dag", "acs", "algorand"):
        compare.add_argument("--" + protocol + "-results", required=True,
            help="results root containing the " + protocol + "/node-N.json files")
    compare.add_argument("--output", required=True, help="new comparison JSON file")
    worker = commands.add_parser("worker", help="internal node process started by run")
    worker.add_argument("--bundle", required=True)
    worker.add_argument("--output", required=True)
    worker.add_argument("--protocol", required=True, choices=("dag", "acs", "algorand"))
    worker.add_argument("--node", type=int, required=True)
    return root


async def launch(args):
    from beaver.a2dpss_benchmark import load_bundle, validate_nodes, write
    manifest = load_bundle(args.bundle)
    loop = asyncio.get_running_loop()
    current = asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM, current.cancel)
    validate_nodes(args.nodes, manifest)
    output = Path(args.output)
    directory = output / args.protocol
    directory.mkdir(parents=True, exist_ok=True)
    for node in args.nodes:
        source = Path(args.bundle) / f"nodes/{node}"
        if not (source / "config.json").is_file():
            raise ValueError(f"missing local config for node {node}; export this host's nodes first")
        if (source / f"{args.protocol}.used").exists() or (directory / f"node-{node}.json").exists():
            raise ValueError(f"node {node}/{args.protocol} already used; prepare a fresh trial")
    processes, streams = {}, []
    failure = None
    try:
        for node in args.nodes:
            stream = (directory / f"node-{node}.log").open("xb")
            streams.append(stream)
            processes[node] = await asyncio.create_subprocess_exec(
                sys.executable, "-u", "-m", "scripts.a2dpss_benchmark", "worker",
                "--bundle", args.bundle, "--protocol", args.protocol, "--node", str(node),
                "--output", args.output, cwd=str(ROOT), stdout=stream, stderr=stream)
        print(f"Started {args.protocol}: nodes {args.nodes}; logs: {directory}", flush=True)
        # Bound startup, online execution, global completion and teardown. A missing
        # remote host cannot keep the local supervisor or sockets alive forever.
        deadline = manifest["startup_timeout"] + 2 * manifest["timeout"] + 30
        async def monitor():
            async def check(node, process):
                code = await process.wait()
                if code:
                    raise RuntimeError(f"node {node} exited {code}; see {directory / f'node-{node}.log'}")
            await asyncio.gather(*(check(n, p) for n, p in processes.items()))
        await asyncio.wait_for(monitor(), deadline)
    except BaseException as error:
        failure = error
        raise
    finally:
        for process in processes.values():
            if process.returncode is None:
                process.kill()
        await asyncio.gather(*(p.wait() for p in processes.values()), return_exceptions=True)
        for stream in streams:
            stream.close()
        if failure is not None:
            for node in args.nodes:
                path = directory / f"node-{node}.json"
                if not path.exists():
                    write(path, dict(schema=manifest["schema"], status="timeout" if isinstance(failure, asyncio.TimeoutError) else "failed",
                        protocol=args.protocol, node=node, bundle_digest=manifest["bundle_digest"],
                        trial_id=manifest["trial_id"], reason=f"supervisor: {type(failure).__name__}: {failure}"))
    print(f"Completed {args.protocol}: local nodes {args.nodes}", flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    # Resolve user paths before moving to the repository for native .so loading.
    for field in ("config_dir", "peers", "bundle", "output", "results",
                  "dag_results", "acs_results", "algorand_results"):
        value = getattr(args, field, None)
        if value:
            setattr(args, field, str(Path(value).resolve()))
    os.chdir(ROOT)
    (ROOT / "log").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    from beaver.a2dpss_benchmark import prepare, export_nodes, collect, compare, write
    if args.command == "prepare":
        manifest = prepare(args)
        print(json.dumps({"bundle": args.output, "trial_id": manifest["trial_id"],
            "view": manifest["view"], "bundle_digest": manifest["bundle_digest"],
            "parameters": manifest["parameters"]}, indent=2))
    elif args.command == "export":
        print(export_nodes(args.bundle, args.output, args.nodes))
    elif args.command == "run":
        asyncio.run(launch(args))
    elif args.command == "worker":
        from beaver.a2dpss_benchmark_runtime import worker
        asyncio.run(worker(args.bundle, args.output, args.protocol, args.node))
    else:
        result = (compare(args.bundle, args.dag_results, args.acs_results, args.algorand_results)
                  if args.command == "compare" else collect(args.bundle, args.results, args.protocol))
        write(args.output, result)
        print(json.dumps({k: v for k, v in result.items() if k not in ("per_node", "protocols")}, indent=2))
        return 0 if result["status"] == "ok" else 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError, asyncio.TimeoutError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
