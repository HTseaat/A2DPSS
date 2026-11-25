"""Distributed entry contracts: isolation, stale data rejection and barriers."""
import asyncio
import copy
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.a2dpss_benchmark import parser
from beaver.a2dpss_benchmark import (
    prepare, export_nodes, load_bundle, local_context, normalize_result,
    collect, read, write, parse_peers)
from beaver.a2dpss_benchmark_runtime import (
    ExperimentBarrier,
    require_native_vrf_backends,
    run_existing,
)
from beaver.experiment_trace import A2DPSSHandoverMetrics


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    directory = tmp_path_factory.mktemp("three-backend")
    configs = directory / "configs"
    subprocess.run([sys.executable, "-m", "scripts.run_key_gen_atcaas", "--M", "5",
        "--n-r", "3", "--t-r", "1", "--n-a", "5", "--t-a", "1",
        "--coin-c", "5", "--coin-t", "1", "--coin-k", "2",
        "--out-dir", str(configs)],
        check=True, capture_output=True)
    peers = directory / "peers.json"
    write(peers, [f"10.0.0.{i+1}:24000" for i in range(5)])
    bundle = directory / "bundle"
    args = parser().parse_args(["prepare", "--config-dir", str(configs), "--peers", str(peers),
        "--output", str(bundle), "--trial-id", "test-distributed"])
    manifest = prepare(args)
    return bundle, manifest, args


def test_prepare_preserves_common_application_and_node_private_partition(setup, tmp_path):
    bundle, manifest, _ = setup
    contexts = []
    for node in range(5):
        config = read(bundle / f"nodes/{node}/config.json")
        assert config["peers"] == manifest["peers"]
        context, local = local_context(config)
        contexts.append(context.context_digest)
        assert local.am_holder == (node < 3)
        assert local.am_successor == (node < 3)
        assert (bundle / f"nodes/{node}/config.json").stat().st_mode & 0o777 == 0o600
    assert len(set(contexts)) == 1
    exported = export_nodes(bundle, tmp_path / "host", [1, 4])
    assert sorted(p.name for p in (exported / "nodes").iterdir()) == ["1", "4"]
    assert load_bundle(exported) == manifest
    # Public setup contains no private signing seed, encryption scalar or old share.
    public = (bundle / "public/manifest.json").read_text()
    for node in range(5):
        identity = read(bundle / f"nodes/{node}/algorand-key.json")
        assert identity["signing"] not in public
        assert str(identity["encryption"]) not in public


def test_reject_tampered_bundle_and_private_overwrite(setup, tmp_path):
    bundle, _, args = setup
    with pytest.raises(FileExistsError):
        prepare(args)
    exported = export_nodes(bundle, tmp_path / "tamper", [0])
    (exported / "public/role-trace.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact digest"):
        load_bundle(exported)


@pytest.mark.parametrize("peers", [["host:1", "host:1"], ["host:0", "host:2"], ["host:22"], ["*:2", "host:3"]])
def test_bad_topology_rejected(peers):
    with pytest.raises(ValueError):
        parse_peers(peers, 2)


def test_protocol_selection_is_explicit():
    with pytest.raises(SystemExit):
        parser().parse_args(["run", "--bundle", "x", "--nodes", "0", "--output", "y"])
    with pytest.raises(SystemExit):
        parser().parse_args(["run", "--bundle", "x", "--nodes", "0", "--output", "y", "--protocol", "all"])


def test_benchmark_vrf_preflight_fails_if_either_backend_is_missing(
    monkeypatch,
):
    import beaver.vrf as vrf
    import beaver.yosorbc as yosorbc

    status = require_native_vrf_backends()
    assert status == {
        "required": True,
        "implementations": {
            "beaver.vrf": True,
            "beaver.yosorbc": True,
        },
    }
    monkeypatch.setattr(yosorbc, "lib_bulletproof", None)
    with pytest.raises(RuntimeError, match="beaver.yosorbc"):
        require_native_vrf_backends()
    monkeypatch.setattr(yosorbc, "lib_bulletproof", object())
    monkeypatch.setattr(vrf, "lib_bulletproof", None)
    with pytest.raises(RuntimeError, match="beaver.vrf"):
        require_native_vrf_backends()


def test_barrier_waits_for_late_host_and_flushes_outbound_ack():
    async def check():
        queues = [defaultdict(asyncio.Queue) for _ in range(3)]
        released = []
        class Runner:
            def __init__(self, node):
                self.my_id = node
                self.node_communicator = SimpleNamespace(_sender_queues=[asyncio.Queue() for _ in range(3)])
            def get_send_recv(self, tag):
                def send(peer, message):
                    if peer == self.my_id:
                        queues[peer][tag].put_nowait((self.my_id, message))
                    else:
                        outbound = self.node_communicator._sender_queues[peer]
                        outbound.put_nowait(message)
                        def deliver():
                            queues[peer][tag].put_nowait((self.my_id, outbound.get_nowait()))
                        asyncio.get_running_loop().call_later(0.001, deliver)
                return send, queues[self.my_id][tag].get
        manifest = dict(parameters={"M": 3}, bundle_digest="test", startup_timeout=1, timeout=1)
        async def node(i):
            runner = Runner(i)
            await ExperimentBarrier(manifest, "dag")(runner, "ready")
            assert all(q.empty() for q in runner.node_communicator._sender_queues)
            released.append(i)
        early = [asyncio.create_task(node(i)) for i in (0, 1)]
        await asyncio.sleep(0.02)
        assert released == []
        await asyncio.gather(*early, node(2))
        assert set(released) == {0, 1, 2}
    asyncio.run(check())


def test_barrier_missing_host_times_out():
    async def check():
        queue = asyncio.Queue()
        runner = SimpleNamespace(my_id=0, get_send_recv=lambda _: (lambda peer, msg: queue.put_nowait((0, msg)) if peer == 0 else None, queue.get))
        manifest = dict(parameters={"M": 2}, bundle_digest="test", startup_timeout=0.02, timeout=1)
        with pytest.raises(asyncio.TimeoutError):
            await ExperimentBarrier(manifest, "algorand")(runner, "ready")
    asyncio.run(check())


def test_collect_rejects_missing_failed_or_truncated_nodes(setup, tmp_path):
    bundle, manifest, _ = setup
    write(tmp_path / "algorand/node-0.json", dict(node=0, protocol="algorand",
        bundle_digest=manifest["bundle_digest"], status="timeout", reason="missing peer"))
    (tmp_path / "algorand/node-1.json").write_text('{"status":')
    result = collect(bundle, tmp_path, "algorand")
    assert result["status"] == "failed"
    assert result["successor_max_latency_seconds"] is None
    assert any("timeout" in e for e in result["failures"])
    assert any("incomplete" in e for e in result["failures"])


@pytest.mark.parametrize("protocol", ["dag", "acs"])
def test_existing_dispatch_runs_only_selected_backend(setup, monkeypatch, protocol):
    bundle, manifest, _ = setup
    called = []
    async def selected(**kwargs):
        called.append(kwargs)
        return {"selected": protocol}
    async def forbidden(**kwargs):
        raise AssertionError("another backend was started")
    for key, module in [("dag", "scripts.run_dag_tob_phase3"), ("acs", "scripts.run_async_yoso_tob")]:
        monkeypatch.setitem(sys.modules, module, SimpleNamespace(_run=selected if key == protocol else forbidden))
    result = asyncio.run(run_existing(bundle, manifest, 0, bundle / "nodes/0/config.json", None, "barrier", protocol))
    assert result == {"selected": protocol} and len(called) == 1
    assert called[0]["coin_mode_requested"] == "threshold-bls"
    assert called[0]["experiment_barrier"] == "barrier"
    assert called[0]["n_r"] == 3 and called[0]["n_a"] == 5


def test_algorand_non_successor_timeline_has_no_private_latency():
    metrics = A2DPSSHandoverMetrics(protocol="async-algorand", node=4, qid="qid", is_holder=False, is_successor=False)
    metrics.start(0)
    metrics.mark_pool_ready(1)
    metrics.mark_ordered(2)
    metrics.mark_public_finalized(3)
    assert metrics.snapshot()["frozen"]
    assert metrics.snapshot()["durations_sec"]["handover_latency"] is None


@pytest.mark.skipif(os.environ.get("A2DPSS_BENCHMARK_NETWORK") != "1", reason="requires local TCP sockets")
def test_cancelled_socket_cannot_leave_infinite_context_linger():
    # The old communicator hangs in destroy() because s.close() ran first with
    # ZMQ's default infinite linger. Passing linger=0 to destroy() is too late.
    code = '''
import socket
from beaver.ipc import NodeCommunicator
from zmq import DEALER
reserved = socket.socket()
reserved.bind(("127.0.0.1", 0))
c = NodeCommunicator({0: None}, 0, linger_timeout=0)
s = c.zmq_context.socket(DEALER)
s.connect("tcp://127.0.0.1:%d" % reserved.getsockname()[1])
s.send(b"pending")
s.close()
del s
c.zmq_context.destroy(linger=0)
reserved.close()
'''
    subprocess.run([sys.executable, "-c", code], check=True, timeout=8, capture_output=True)
