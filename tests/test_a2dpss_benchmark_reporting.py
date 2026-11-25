"""Reporting-only tests: no key generation, prepare, sockets or real protocols."""
import asyncio
import copy
from collections import defaultdict
from types import SimpleNamespace

import pytest

from beaver import a2dpss_benchmark as benchmark
from beaver.a2dpss_benchmark_metrics import CPU_STAGES, aggregate_cpu, normalize_cpu
from beaver.experiment_trace import A2DPSSHandoverMetrics, build_a2dpss_computation_metrics
from scripts.a2dpss_benchmark import parser


def metric(cpu, calls=1):
    return {"cpu_seconds": cpu, "invocations": calls}


def legacy_cpu(protocol, node=0):
    return build_a2dpss_computation_metrics(
        protocol={"dag": "dag-tob", "acs": "acs-tob"}[protocol], node=node,
        proposer_predicate=metric(2), receiver_predicate=metric(3),
        candidate_formation=metric(5), pvss_generation=metric(7),
        pvss_verification=metric(100), public_finalization=metric(11),
        successor_reconstruction=metric(13))


@pytest.mark.parametrize("protocol", ["dag", "acs"])
def test_cpu_total_does_not_double_count_pvss_detail(protocol):
    result = normalize_cpu({"a2dpss_computation_metrics": legacy_cpu(protocol)}, protocol, 0)
    assert result["available"] and result["complete"]
    assert result["total_cpu_seconds"] == 41
    assert result["stages"]["formal_predicate"]["cpu_seconds"] == 5


def test_missing_legacy_cpu_is_unknown_and_partial_algorand_stays_partial():
    for protocol in ("dag", "acs", "algorand"):
        result = normalize_cpu({}, protocol, 0)
        assert not result["available"] and not result["complete"]
        assert result["total_cpu_seconds"] is None
    result = normalize_cpu({"metrics": {
        "a2_contribution_generation_seconds": 9,
        "a2_predicate_seconds": 10,
        "a2_receive_validation_cpu_seconds": 2,
        "a2_receive_validation_calls": 3,
    }, "a2dpss_finalization_metrics": {"total": metric(4)}}, "algorand", 0)
    assert result["available"] and not result["complete"]
    assert result["stages"]["formal_predicate"] is None
    assert result["stages"]["pvss_generation"] is None
    assert result["total_cpu_seconds"] is None


@pytest.mark.parametrize("bad_cpu", [float("nan"), float("inf"), -1, True])
def test_malformed_cpu_cannot_become_a_successful_total(bad_cpu):
    source = {name: metric(1) for name in CPU_STAGES}
    source["formal_predicate"] = metric(bad_cpu)
    with pytest.raises(ValueError, match="finite nonnegative"):
        normalize_cpu({"a2dpss_benchmark_cpu_stages": source}, "algorand", 0)


def test_network_cpu_is_node_sum_and_missing_node_has_no_total():
    raw = {"a2dpss_benchmark_cpu_stages": {name: metric(1) for name in CPU_STAGES}}
    records = [{"node": node, "adpss_cpu": normalize_cpu(raw, "algorand", node)} for node in range(2)]
    assert aggregate_cpu(records, 2)["total_cpu_seconds"] == 12
    result = aggregate_cpu(records[:1], 2)
    assert result["available"] and not result["complete"]
    assert result["total_cpu_seconds"] is None


@pytest.fixture
def public_bundle(monkeypatch):
    context = SimpleNamespace(context_digest="context", service_t=0, encode_qid=lambda: "qid")
    monkeypatch.setattr(benchmark, "A2DPSSContext", SimpleNamespace(from_obj=lambda _: context))
    manifest = dict(parameters={"M": 2}, context={}, successor_nodes=[0],
        trial_id="reporting-unit", bundle_digest="bundle", input_digest="input",
        fixture_digest=None, view="live", coin_public_key_digest="coin")
    monkeypatch.setattr(benchmark, "load_bundle", lambda _: copy.deepcopy(manifest))
    return manifest


def write_records(tmp_path, manifest, protocol, *, old=False, missing_cpu=False):
    for node in range(2):
        timing = A2DPSSHandoverMetrics(
            protocol={"dag": "dag-tob", "acs": "acs-tob", "algorand": "async-algorand"}[protocol],
            node=node, qid="qid", is_holder=False, is_successor=node == 0)
        timing.start(0)
        timing.mark_pool_ready(1)
        timing.mark_ordered(2)
        timing.mark_public_finalized(3)
        if node == 0:
            timing.mark_successor_done(4)
        raw = dict(a2dpss_handover_metrics=timing.snapshot(),
            a2dpss_events=[dict(event="public_finalized", qid="qid", context_digest="context",
                accepted_contribution_digests=["fresh-" + protocol],
                successor_commitment_vector=["commitment-" + protocol])],
            cleanup_completed=True, coin_mode="threshold-bls", coin_public_key_digest="coin",
            vrf_backend={"required": True, "implementations": {"beaver.vrf": True, "beaver.yosorbc": True}},
            benchmark_protocol_wire_bytes=10)
        if node == 0:
            raw["a2dpss_events"].append({"event": "reconstructed"})
        if not missing_cpu:
            if protocol == "algorand":
                raw["a2dpss_benchmark_cpu_stages"] = {name: metric(1) for name in CPU_STAGES}
            else:
                raw["a2dpss_computation_metrics"] = legacy_cpu(protocol, node)
        record = benchmark.normalize_result(manifest, protocol, node, raw)
        if old:
            record.pop("adpss_cpu")
        benchmark.write(tmp_path / protocol / f"node-{node}.json", record)


def test_collect_accepts_old_node_shape_without_inventing_cpu(tmp_path, public_bundle):
    write_records(tmp_path, public_bundle, "algorand", old=True, missing_cpu=True)
    result = benchmark.collect("unused", tmp_path, "algorand")
    assert result["status"] == "ok"
    assert result["successor_max_latency_seconds"] == 4
    assert result["protocol_wire_bytes"] == 20
    assert not result["adpss_cpu"]["complete"]
    assert result["adpss_cpu"]["total_cpu_seconds"] is None


def test_compare_live_allows_fresh_distinct_T_and_revalidates_nodes(tmp_path, public_bundle):
    for protocol in benchmark.PROTOCOLS:
        write_records(tmp_path, public_bundle, protocol, old=True)
    result = benchmark.compare("unused", tmp_path, tmp_path, tmp_path)
    assert result["status"] == "ok" and result["cpu_comparable"]
    assert result["public_results_equal"] is False
    assert result["metrics"]["dag"]["adpss_cpu_seconds"] == 82
    assert result["ratios"]["acs_over_dag"]["adpss_cpu_seconds"] == 1
    path = tmp_path / "acs/node-1.json"
    record = benchmark.read(path)
    record["bundle_digest"] = "wrong-trial"
    path.unlink()
    benchmark.write(path, record)
    result = benchmark.compare("unused", tmp_path, tmp_path, tmp_path)
    assert result["status"] == "failed" and not result["cpu_comparable"]
    assert all(v is None for v in result["ratios"]["acs_over_dag"].values())


def test_compare_legacy_algorand_keeps_latency_but_disables_cpu_ratio(tmp_path, public_bundle):
    for protocol in benchmark.PROTOCOLS:
        write_records(tmp_path, public_bundle, protocol, missing_cpu=protocol == "algorand")
    result = benchmark.compare("unused", tmp_path, tmp_path, tmp_path)
    assert result["status"] == "ok" and not result["cpu_comparable"]
    assert not result["metrics_complete"]
    assert result["ratios"]["algorand_over_dag"]["successor_max_latency_seconds"] == 1
    assert result["ratios"]["algorand_over_dag"]["adpss_cpu_seconds"] is None


def test_compare_paired_rejects_different_public_results(public_bundle, monkeypatch):
    # collect's cryptographic fixture validation is tested by the existing
    # fixture suite; this reporting test only checks the cross-arm boundary.
    public_bundle["view"] = "paired"
    def collected(directory, results, protocol):
        return dict(status="ok", failures=[], adpss_cpu={"complete": True, "total_cpu_seconds": 1},
            successor_max_latency_seconds=1, protocol_wire_bytes=1,
            per_node=[{"public_result": {"T": protocol}}])
    monkeypatch.setattr(benchmark, "collect", collected)
    result = benchmark.compare("unused", "d", "a", "g")
    assert result["status"] == "failed"


def test_compare_cli_requires_all_three_result_roots():
    args = parser().parse_args(["compare", "--bundle", "b", "--dag-results", "d",
        "--acs-results", "a", "--algorand-results", "g", "--output", "o"])
    assert (args.dag_results, args.acs_results, args.algorand_results) == ("d", "a", "g")
    with pytest.raises(SystemExit):
        parser().parse_args(["compare", "--bundle", "b", "--dag-results", "d", "--output", "o"])


def test_algorand_formal_J_records_process_time(monkeypatch):
    from beaver.async_algorand import a2dpss as module
    adapter = module.A2DPSSAdapter.__new__(module.A2DPSSAdapter)
    adapter.cpu_stages = {name: metric(0, 0) for name in CPU_STAGES}
    adapter.qid = "qid"
    adapter.resolve = lambda _: [("node_0", b"payload")]
    adapter.justifier = SimpleNamespace(verify_candidate_set=lambda *args: True)
    adapter.r = SimpleNamespace(metrics=defaultdict(int))
    ticks = iter((10.0, 12.5))
    monkeypatch.setattr(module.time, "process_time", lambda: next(ticks))
    assert adapter.valid({}) is True
    assert adapter.cpu_stages["formal_predicate"] == metric(2.5)
    assert adapter.r.metrics["a2_predicate_cpu_seconds"] == 2.5


def test_algorand_live_generation_persists_holder_cpu_without_real_protocol(monkeypatch):
    from beaver.async_algorand import a2dpss as module
    adapter = module.A2DPSSAdapter.__new__(module.A2DPSSAdapter)
    adapter.cpu_stages = {name: metric(0, 0) for name in CPU_STAGES}
    adapter.delivery_preparation_cpu = 0.0
    adapter.fixture, adapter.qid = None, "qid"
    adapter.context = SimpleNamespace(service_t=0, successor_role_indices={})
    adapter.holders = {0: "ref"}
    adapter.candidate_refs = ("ref",)
    adapter.timeline = SimpleNamespace(**{name: lambda *a: None for name in (
        "start", "mark_contribution_sent", "mark_pool_ready", "mark_ordered", "mark_public_finalized")})
    adapter.resolve = lambda _: [("node_0", b"payload")]
    adapter.justifier = SimpleNamespace(select_preverified_subset=lambda q, entries: entries)
    adapter.broadcast = SimpleNamespace(broadcast=lambda _: None)
    adapter.finalizer = SimpleNamespace(on_verified_deliver=lambda *args: "public")
    adapter.reconstruction = SimpleNamespace(reconstruct=lambda *args: None)
    async def immediate(predicate):
        assert predicate()
    adapter.r = SimpleNamespace(node=0, metrics=defaultdict(int), wait=immediate)
    seen = []
    def generate(**kwargs):
        seen.append(kwargs["metrics"])
        kwargs["metrics"].pvss_cpu_seconds = 0.25
        return SimpleNamespace(to_bytes=lambda: b"payload")
    class FakeAgreement:
        def __init__(self, *args):
            pass
        async def run_prevalidated(self, proposal):
            return proposal
    monkeypatch.setattr(module, "make_a2dpss_contribution", generate)
    monkeypatch.setattr(module, "NWA", FakeAgreement)
    asyncio.run(adapter.run((1, 2)))
    assert len(seen) == 1
    assert adapter.cpu_stages["pvss_generation"] == metric(0.25)
    assert adapter.r.metrics["a2_contribution_generation_cpu_seconds"] >= 0
    assert adapter.r.metrics["a2_contribution_generation_calls"] == 1
