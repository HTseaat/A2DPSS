import asyncio
import json
import tarfile
from types import SimpleNamespace

import pytest

from beaver.yoso_adpss_distributed import DistributedBarrier, valid_timeline
from scripts.run_yoso_adpss_dist import load_cluster, safe_extract, ssh_command, stage_host
from scripts import analyze_yoso_adpss_dag_arm as analyzer


def test_cluster_placement_separates_physical_hosts_from_logical_nodes(tmp_path):
    path = tmp_path / "cluster.json"
    path.write_text(json.dumps({"hosts": [
        {"name": "one", "transport": "local"},
        {"name": "two", "transport": "local"},
    ]}))
    hosts, peers = load_cluster(path, 5, 16000)
    assert hosts[0]["nodes"] == [0, 2, 4]
    assert hosts[1]["nodes"] == [1, 3]
    assert len(set(peers)) == 5
    path.write_text(json.dumps({"hosts": [
        {"transport": "local", "nodes": [0, 1]},
        {"transport": "local", "nodes": [1, 2]},
    ]}))
    with pytest.raises(ValueError, match="duplicate"):
        load_cluster(path, 3, 16000)


def test_stage_does_not_send_other_nodes_private_material(tmp_path):
    trial, run = tmp_path / "trial", tmp_path / "run"
    for folder in ("code", "public", "nodes/0", "nodes/1", "configs"):
        (trial / folder).mkdir(parents=True, exist_ok=True)
    run.mkdir()
    (trial / "code/test.py").write_text("# code")
    (trial / "public/role-trace.json").write_text("{}")
    (run / "experiment.json").write_text("{}")
    (trial / "nodes/0/config.json").write_text('"node zero secret"')
    (trial / "nodes/1/config.json").write_text('"node one secret"')
    (trial / "configs/master.json").write_text('"controller secrets"')
    import os
    (trial / "nodes/0/lookup.bin").write_bytes(b"shared lookup contents")
    os.link(trial / "nodes/0/lookup.bin", trial / "nodes/0/lookup-copy.bin")
    host = {"name": "one", "transport": "local", "nodes": [0]}
    destination = stage_host(trial, host, {}, "a2dpss", run)
    assert (destination / "nodes/0/config.json").exists()
    assert not (destination / "nodes/1").exists()
    assert not (destination / "configs").exists()
    assert (destination / "nodes/0/lookup-copy.bin").read_bytes() == b"shared lookup contents"


def test_extract_rejects_escape_and_ssh_preserves_literal_arguments(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.addfile(tarfile.TarInfo("../escape"))
    with pytest.raises(ValueError, match="unsafe"):
        safe_extract(archive, tmp_path / "out")
    host = {"ssh": "user@server", "ssh_port": 22}
    import shlex
    arguments = ["python3", "-c", "print('ok')", "/tmp/a path/$(no)"]
    assert shlex.split(ssh_command(host, arguments)[-1]) == arguments


def node_result(node_id, successor):
    from beaver.yoso_adpss_experiment import NODE_METRICS_SCHEMA
    return dict(schema=NODE_METRICS_SCHEMA, protocol="a2dpss", node_id=node_id,
                status="success", failure=None, common_workload_digest="common",
                arm_manifest_digest="arm", experiment_digest="experiment",
                timeline=dict(t0=0, t_first_role_message_sent=None, t_first_pool_ready=.1,
                              t_public_finalized=.2, t_successor_done=.3 if successor else None,
                              frozen_at=.4), cleanup={"completed": True}, cpu={},
                protocol_metrics={"formal_cache_hits": 0, "public_result_digest": "public"},
                communication=dict(application_payload_bytes=0, total_network_bytes=6,
                                   yoso_rbc_envelope_bytes=3, dag_protocol_bytes=2, common_coin_bytes=1,
                                   committed_application_block_count=1))


def test_single_arm_analyzer_requires_only_designated_successors(monkeypatch, tmp_path):
    manifest = SimpleNamespace(parameters=SimpleNamespace(M=5, to_config_obj=lambda: {"M": 5}),
                               successor_role_to_node={0: 1, 1: 3, 2: 4}, digest=lambda: "common")
    monkeypatch.setattr(analyzer, "load_yoso_adpss_public_fixture",
                        lambda _: (manifest, {"a2dpss": SimpleNamespace(digest=lambda: "arm")}))
    monkeypatch.setattr(analyzer, "verify_same_initial_workload", lambda *_: True)
    nodes = [node_result(i, i in (1, 3, 4)) for i in range(5)]
    result = analyzer.analyze_arm(tmp_path, "a2dpss", nodes, experiment_digest="experiment")
    assert result["status"] == "success" and result["successor_nodes"] == [1, 3, 4]
    nodes[3]["timeline"]["t_successor_done"] = None
    with pytest.raises(ValueError, match="successor"):
        analyzer.analyze_arm(tmp_path, "a2dpss", nodes)


def test_timeline_independent_events_and_nonfinite_rejection():
    timeline = dict(t0=0, t_first_role_message_sent=.2, t_first_pool_ready=.1,
                    t_public_finalized=.3, t_successor_done=.4, frozen_at=.5)
    assert valid_timeline(timeline)
    assert not valid_timeline({**timeline, "t_successor_done": .1})
    assert not valid_timeline({**timeline, "t_successor_done": float("nan")})


def test_cpu_total_excludes_nested_diagnostics_and_repeated_totals():
    record = lambda n: {"cpu_seconds": n, "invocations": 1}
    metrics = {"predicate": {"total": record(7), "roles": {"proposer": record(3), "receiver": record(4)}},
               "resharing": {"pvss_generation": record(2), "pvss_verification": record(6),
                             "public_finalization": record(1), "successor_reconstruction": record(5)}}
    assert sum(analyzer.cpu_breakdown("a2dpss", metrics).values()) == 15
    metrics = {"holder_generation": record(2), "formal": {"roles": {"proposer": record(7)}},
               "validation": {"pvss": record(6)}, "public_finalization": {"derivation": record(1)},
               "successor_reconstruction": {"total": record(5), "decrypt": record(4)}}
    assert sum(analyzer.cpu_breakdown("ru22", metrics).values()) == 15


def test_bhk_normalization_counts_actual_application_bytes():
    from beaver.yoso_adpss_experiment import normalize_dag_node_summary
    raw = dict(cleanup_completed=True, delivery_event_count=2, bhk22_summary={
        "status": "success", "timeline": {"t0": 0, "t_public_finalized": 1, "frozen_at": 2},
        "rbc_metrics": {"deal": {"outbound_application_bytes": 101, "outbound_proof_bytes": 11,
                                 "outbound_ciphertext_bytes": 50},
                        "verify": {"outbound_application_bytes": 42, "outbound_proof_bytes": 7,
                                   "outbound_ciphertext_bytes": 13}}})
    value = normalize_dag_node_summary(protocol="bhk22", node_id=4,
                                      common_workload_digest="c", arm_manifest_digest="a", legacy_summary=raw)
    assert value["communication"]["application_payload_bytes"] == 143
    assert value["communication"]["cryptographic_proof_bytes"] == 18
    assert value["communication"]["recipient_ciphertext_bytes"] == 63


def test_ssh_staging_and_collection_via_local_command_transport(tmp_path, monkeypatch):
    # Exercise the actual quoted SSH command and tar stream without contacting
    # a remote account. This is a transport test, not a multi-host measurement.
    import os
    from scripts.run_yoso_adpss_dist import collect_host
    binary = tmp_path / "bin"
    binary.mkdir()
    ssh = binary / "ssh"
    ssh.write_text("#!/usr/bin/env python3\nimport subprocess,sys\nsys.exit(subprocess.call(['sh','-c',sys.argv[-1]]))\n")
    ssh.chmod(0o700)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    trial, run = tmp_path / "trial", tmp_path / "run"
    for folder in ("code", "public", "nodes/0"):
        (trial / folder).mkdir(parents=True, exist_ok=True)
    run.mkdir()
    (trial / "public/role-trace.json").write_text("{}")
    (trial / "nodes/0/config.json").write_text("{}")
    (run / "experiment.json").write_text("{}")
    host = dict(name="remote", transport="ssh", nodes=[0], ssh="fake-host", ssh_port=22,
                python="python3", run_root=str(tmp_path / "remote with spaces"))
    target = stage_host(trial, host, {"trial_id": "t1", "digest": "a" * 64}, "ru22", run)
    assert (target / "nodes/0/config.json").exists()
    (target / "output").mkdir()
    (target / "output/node.0.json").write_text('{"status":"success"}')
    collect_host(host, target, run / "nodes")
    assert json.loads((run / "nodes/node.0.json").read_text())["status"] == "success"


def test_ready_barrier_waits_for_late_peer_and_done_collects_protocol_bytes():
    async def scenario():
        queues = {}
        class Runner:
            def __init__(self, node):
                self.my_id = node
                self.node_communicator = SimpleNamespace(
                    _sender_queues=[asyncio.Queue() for _ in range(3)],
                    bytes_by_tag_path={"DAGTOB:x": 10, "YOSO_ADPSS_CONTROL/x": 20})
            def get_send_recv(self, tag):
                inbox = queues.setdefault((self.my_id, tag), asyncio.Queue())
                def send(peer, value):
                    queues.setdefault((peer, tag), asyncio.Queue()).put_nowait((self.my_id, value))
                return send, inbox.get
        settings = {"parameters": {"M": 3}, "digest": "test", "startup_timeout": 2, "timeout": 2}
        barriers = [DistributedBarrier(settings, "a2dpss") for _ in range(3)]
        runners = [Runner(i) for i in range(3)]
        tasks = [asyncio.create_task(barriers[i](runners[i], "ready")) for i in (0, 1)]
        await asyncio.sleep(.02)
        assert all(not t.done() for t in tasks)
        tasks.append(asyncio.create_task(barriers[2](runners[2], "ready")))
        await asyncio.gather(*tasks)
        await asyncio.gather(*(barriers[i](runners[i], "done") for i in range(3)))
        assert all(b.transport["total"] == 10 for b in barriers)
    asyncio.run(scenario())
