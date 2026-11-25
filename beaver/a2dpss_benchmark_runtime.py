"""One selected backend per process, with common distributed experiment barriers."""
from __future__ import annotations

import asyncio
import os
import logging
from pathlib import Path
import platform
import sys
import time

from beaver.a2dpss import A2DPSSContext
from beaver.a2dpss_experiment import A2DPSSPairedFixture
from beaver.a2dpss_benchmark import read, write, local_context, load_bundle, normalize_result
from beaver.async_algorand.model import Parameters, digest
from beaver.config import HbmpcConfig, NodeDetails
from beaver.ipc import ProcessProgramRunner


def require_native_vrf_backends() -> dict:
    """Fail a benchmark worker before measurement if either VRF FFI is absent."""

    from beaver.vrf import vrf_backend_available as utility_backend_available
    from beaver.yosorbc import vrf_backend_available as rbc_backend_available

    status = {
        "beaver.vrf": utility_backend_available(),
        "beaver.yosorbc": rbc_backend_available(),
    }
    missing = [name for name, loaded in status.items() if not loaded]
    if missing:
        raise RuntimeError(
            "native VRF backend is required for benchmark runs; missing: "
            + ", ".join(missing)
        )
    return {"required": True, "implementations": status}


class ExperimentBarrier:
    def __init__(self, manifest, protocol):
        self.manifest, self.protocol = manifest, protocol
        self.runner = None
        self.elapsed = {}

    async def __call__(self, runner, phase):
        self.runner = runner
        logging.warning("[BENCHMARK] node=%s barrier=%s enter", runner.my_id, phase)
        started = time.perf_counter()
        count = self.manifest["parameters"]["M"]
        tag = "benchmark/" + self.manifest["bundle_digest"] + "/" + self.protocol + "/" + phase
        send, recv = runner.get_send_recv(tag)

        async def gather():
            # A second exchange acknowledges that every peer reached the barrier.
            # No wall-clock synchronization or a fixed startup sleep is required.
            for step in ("arrived", "acknowledged"):
                for peer in range(count):
                    send(peer, (step, True))
                seen = set(early) if step == "acknowledged" else set()
                while len(seen) != count:
                    sender, value = await recv()
                    if sender not in range(count) or value != (step, True):
                        # A peer can send its acknowledgment while we still drain
                        # arrival messages. Keep it for the next phase.
                        if sender in range(count) and value == ("acknowledged", True):
                            early.add(sender)
                            continue
                        raise ValueError("invalid experiment barrier message")
                    seen.add(sender)
                    if step == "acknowledged":
                        seen.update(early)

        early = set()
        timeout = self.manifest["startup_timeout"] if phase == "ready" else self.manifest["timeout"]
        await asyncio.wait_for(gather(), timeout)
        logging.warning("[BENCHMARK] node=%s barrier=%s acknowledged", runner.my_id, phase)
        # send() enqueues locally. Flush those queues before a fast peer closes
        # its communicator, otherwise the last barrier ACK can be discarded.
        async def flush():
            await asyncio.sleep(0)
            queues = runner.node_communicator._sender_queues
            while any(not queue.empty() for i, queue in enumerate(queues) if i != runner.my_id):
                await asyncio.sleep(0.001)
        await asyncio.wait_for(flush(), timeout)
        self.elapsed[phase] = time.perf_counter() - started
        logging.warning("[BENCHMARK] node=%s barrier=%s flushed", runner.my_id, phase)

    def wire_bytes(self):
        # Count actual serialized network sends, including service until global
        # completion, while excluding the experiment's own control messages.
        return sum(value for tag, value in self.runner.node_communicator.bytes_by_tag_path.items()
                   if not tag.startswith("benchmark/"))


async def run_algorand(directory, manifest, node, config, fixture, barrier, private_output):
    from beaver.async_algorand.transport import Runtime
    from beaver.async_algorand.a2dpss import A2DPSSAdapter
    from beaver.async_algorand.experiment import consume_bootstrap_seed

    parameters = Parameters(**manifest["algorand_parameters"])
    context, local = local_context(config)
    context = fixture.context if fixture is not None else context
    private_path = directory / f"nodes/{node}/algorand-key.json"
    identity = read(private_path)
    peers = {i: NodeDetails(*[p.rsplit(":", 1)[0], int(p.rsplit(":", 1)[1])])
             for i, p in enumerate(manifest["peers"])}
    async with ProcessProgramRunner(peers, parameters.population, parameters.m - parameters.h,
                                    node, linger_timeout=1, drain_on_exit=True) as runner:
        outer_send, outer_recv = runner.get_send_recv("async_algorand")
        def send(peer, wire):
            outer_send(peer, ("wire", wire))
        async def recv():
            sender, (_, wire) = await outer_recv()
            return sender, wire
        runtime = Runtime(manifest["trial_id"], node, parameters, identity,
            manifest["algorand_public_keys"], context.g_hex, context.h_hex, send, recv).start()
        consume_bootstrap_seed(private_path, identity)
        adapter = A2DPSSAdapter(runtime, context, local.my_successor_rsk, fixture=fixture)
        try:
            await barrier(runner, "ready")
            decision, public, successor = await asyncio.wait_for(
                adapter.run(None if fixture is not None else local.my_holder_share), manifest["timeout"])
            events = [{"event": "public_finalized", **public.to_obj()}]
            if successor is not None:
                # Each successor verifies its own share against vc'. Collect only
                # public attestations, never the successor scalars across hosts.
                private_output.parent.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                write(private_output, successor.to_obj(), private=True)
                events.append({"event": "reconstructed", "my_node_id": node,
                               "my_role_idx": successor.my_role_idx})
            raw = dict(protocol="async-algorand", node=node, epoch=manifest["trial_id"],
                decision=decision, a2dpss_events=events, a2dpss_context_digest=context.context_digest,
                a2dpss_handover_metrics=adapter.timeline.snapshot(),
                metrics=dict(runtime.metrics),
                a2dpss_benchmark_cpu_stages=adapter.cpu_snapshot(),
                a2dpss_validation_metrics=adapter.justifier.validation_metrics_snapshot(),
                a2dpss_finalization_metrics=adapter.finalizer.metrics_snapshot(),
                a2dpss_reconstruction_metrics=adapter.reconstruction.metrics_snapshot(),
                coin_mode="native-avss-gather-pe", online_avss_setup=True)
            await barrier(runner, "done")
        finally:
            await runtime.close()
        raw["cleanup_completed"] = runtime.closed and not runtime.tasks
        return raw


async def run_existing(directory, manifest, node, config_path, fixture, barrier, protocol):
    # Run the established implementations through their existing Python API.
    # The only new runner hook is the optional ready/done experiment barrier.
    argv = sys.argv
    try:
        sys.argv = [argv[0], "-d", "-f", str(config_path), "-time", "0"]
        HbmpcConfig.load_config()
    finally:
        sys.argv = argv
    params = manifest["parameters"]
    kwargs = dict(peers=HbmpcConfig.peers, n=HbmpcConfig.N, t=HbmpcConfig.t, my_id=node,
        start_time=0, M=params["M"], epoch=manifest["trial_id"],
        expected_committee_size=params["n_a"], committee_m=params["m_a"], quorum_h=params["quorum_h"],
        max_runtime=manifest["timeout"], stop_grace_seconds=3, a2dpss_smoke=True,
        coin_mode_requested="threshold-bls", coin_timeout=manifest["coin_timeout"],
        a2dpss_fixture=fixture, experiment_barrier=barrier,
        **{
            key: params[key]
            for key in (
                "n_r",
                "t_r",
                "n_a",
                "t_a",
                "m_a",
                "coin_c",
                "coin_t",
                "coin_k",
            )
        })
    if protocol == "dag":
        from scripts.run_dag_tob_phase3 import _run
        kwargs.update(rounds=manifest["rounds"], target_deliveries=1)
    else:
        from scripts.run_async_yoso_tob import _run
        kwargs.update(a2dpss_e2e_smoke=True, acs_activation_mode="vrf", acs_collect_all=True,
            acs_collect_threshold=params["quorum_h"], acs_max_rounds=manifest["rounds"],
            acs_stage_timeout=manifest["timeout"], acs_output_service_grace=0.0)
    return await _run(**kwargs)


async def worker(directory, output, protocol, node):
    from beaver.chunk_elgamal import install_lookup_table_artifacts

    directory, output = Path(directory).resolve(), Path(output).resolve()
    manifest = load_bundle(directory)
    config_path = directory / f"nodes/{node}/config.json"
    result_path = output / protocol / f"node-{node}.json"
    config = read(config_path)
    if config["my_id"] != node or digest(config) != manifest["private_config_digests"][str(node)]:
        raise ValueError("private node configuration does not match the prepared bundle")
    context, _ = local_context(config)
    fixture = (A2DPSSPairedFixture.from_file(str(directory / "public/fixture.json"))
               if manifest["view"] == "paired" else None)
    if fixture is not None:
        context = fixture.context
    if digest(context.to_obj()) != manifest["input_digest"]:
        raise ValueError("node A2DPSS context does not match the experiment")
    lookup_preload_seconds = 0.0
    lookup_tables = (config.get("extra") or {}).get("chunk_lookup_tables")
    if lookup_tables is not None:
        lookup_started = time.perf_counter()
        install_lookup_table_artifacts(
            lookup_tables,
            base_directory=config_path.parent,
        )
        lookup_preload_seconds = time.perf_counter() - lookup_started
    if result_path.exists():
        raise FileExistsError("result already exists; use a fresh trial bundle")
    vrf_backend = require_native_vrf_backends()
    # A protocol/session is one-use even if the caller changes output directories.
    write(directory / f"nodes/{node}/{protocol}.used", {"trial_id": manifest["trial_id"]}, private=True)
    # Isolate this entry from legacy smoke environment variables and accidental
    # cached/ideal protocol replay. Every benchmark setting comes from the bundle.
    for key in list(os.environ):
        if key.startswith(("ATCAAS_", "DAG_TOB_", "ASYNC_YOSO_TOB_")):
            del os.environ[key]
    os.environ.update(ATCAAS_TRIAL_EPOCH=manifest["trial_id"],
        ATCAAS_ROLE_TRACE=str(directory / "public/role-trace.json"),
        ATCAAS_COMMITTEE_SELECTION_MODE=manifest["selection_mode"],
        ATCAAS_FIXED_COMMITTEE_SEED=manifest["committee_seed"],
        DAG_TOB_DPSS_POOL_WAIT_SECONDS=str(manifest["timeout"]))
    barrier = ExperimentBarrier(manifest, protocol)
    started = time.perf_counter()
    try:
        if protocol == "algorand":
            raw = await run_algorand(directory, manifest, node, config, fixture, barrier,
                                    output / "private" / protocol / f"successor-{node}.json")
        else:
            raw = await run_existing(directory, manifest, node, config_path, fixture, barrier, protocol)
        raw["lookup_table_preload_seconds"] = lookup_preload_seconds
        raw["vrf_backend"] = vrf_backend
        raw["benchmark_protocol_wire_bytes"] = barrier.wire_bytes()
        logging.warning("[BENCHMARK] node=%s backend returned", node)
        raw["benchmark_barrier_seconds"] = barrier.elapsed
        result = normalize_result(manifest, protocol, node, raw)
    except Exception as error:
        result = dict(schema=manifest["schema"], status="timeout" if isinstance(error, asyncio.TimeoutError) else "failed",
            protocol=protocol, node=node, bundle_digest=manifest["bundle_digest"],
            trial_id=manifest["trial_id"], reason=f"{type(error).__name__}: {error}",
            process_elapsed_seconds=time.perf_counter() - started)
        write(result_path, result)
        raise
    result["machine"] = {"hostname": platform.node(), "platform": platform.platform(),
        "python": sys.version, "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
    write(result_path, result)
    return result
