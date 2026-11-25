"""Experiment control only; shared by the three existing ADPSS DAG runners."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math


CONTROL_TAG = "YOSO_ADPSS_CONTROL/"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def valid_timeline(timeline):
    """Send and pool-ready are independent events, not a total chronological list."""
    start, end = timeline.get("t0"), timeline.get("frozen_at")
    if start is None or end is None:
        return False
    values = [v for v in timeline.values() if v is not None]
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in values):
        return False
    if start != 0 or any(v < start or v > end for v in values):
        return False
    public, successor = timeline.get("t_public_finalized"), timeline.get("t_successor_done")
    return public is not None and (successor is None or public <= successor)


class DistributedBarrier:
    # The DAG loop continues generating/serving vertices until this barrier
    # finishes, rather than stopping after a machine-dependent fixed sleep.
    keep_serving_until_done = True

    def __init__(self, settings, protocol):
        self.settings, self.protocol = settings, protocol
        self.elapsed = {}
        self.transport = None

    async def __call__(self, runner, phase):
        import time
        started = time.perf_counter()
        count = self.settings["parameters"]["M"]
        tag = CONTROL_TAG + self.settings["digest"] + "/" + self.protocol + "/" + phase
        send, recv = runner.get_send_recv(tag)
        early = set()

        async def gather():
            for step in ("arrived", "acknowledged"):
                for peer in range(count):
                    send(peer, step)
                seen = set(early) if step == "acknowledged" else set()
                while len(seen) < count:
                    sender, value = await recv()
                    if not isinstance(sender, int) or sender not in range(count):
                        raise ValueError("invalid control sender")
                    if value == "acknowledged" and step == "arrived":
                        early.add(sender)
                    elif value == step:
                        seen.add(sender)
                    else:
                        raise ValueError("invalid control barrier message")
            await asyncio.sleep(0)
            queues = runner.node_communicator._sender_queues
            while any(not q.empty() for i, q in enumerate(queues) if i != runner.my_id):
                await asyncio.sleep(0.001)

        limit = self.settings["startup_timeout"] if phase == "ready" else self.settings["timeout"]
        await asyncio.wait_for(gather(), limit)
        self.elapsed[phase] = time.perf_counter() - started
        if phase == "done":
            # Include actual protocol sends while serving slower peers, excluding
            # experiment-control messages. No subtraction from protocol latency.
            from scripts.run_dag_tob_phase3 import _transport_family_totals
            paths = {k: v for k, v in runner.node_communicator.bytes_by_tag_path.items()
                     if not k.startswith(CONTROL_TAG)}
            self.transport = _transport_family_totals(paths, total_bytes=sum(paths.values()))
