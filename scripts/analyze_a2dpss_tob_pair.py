"""Fail-closed validation for one paired DAG/ACS A²DPSS trial."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from beaver.a2dpss import A2DPSSJustifierManager, A2DPSSPublicFinalizer
from beaver.a2dpss_experiment import A2DPSSPairedFixture
from beaver.experiment_trace import aggregate_a2dpss_computation_metrics
from scripts.analyze_atcaas_tob_logs import _read_last_summary as read_dag
from scripts.analyze_async_yoso_tob_logs import _read_last_summary as read_acs


def _read_group(pattern, reader):
    summaries = {}
    for path in sorted(glob.glob(pattern)):
        summary = reader(path)
        if summary is not None:
            summaries[int(summary["node"])] = summary
    return summaries


def _public_view(result):
    if not result:
        return None
    return {
        "qid": result.get("qid"),
        "context_digest": result.get("context_digest"),
        "accepted_contribution_digests": result.get(
            "accepted_contribution_digests"
        ),
        "successor_commitment_vector": result.get(
            "successor_commitment_vector"
        ),
    }


def _dag_public_results(summary):
    return [
        _public_view(event)
        for event in summary.get("a2dpss_events", [])
        if event.get("event") == "public_finalized"
    ]


def _latency(summary):
    handover = summary.get("a2dpss_handover_metrics") or {}
    return (handover.get("durations_sec") or {}).get("handover_latency")


def _coin_view(summary):
    return {
        "mode": summary.get("coin_mode"),
        "committee": summary.get("coin_committee"),
        "threshold": summary.get("coin_threshold"),
        "membership_digest": summary.get("coin_membership_digest"),
        "public_key_digest": summary.get("coin_public_key_digest"),
    }


def _phase6_parameter_view(summary):
    if summary.get("agreement_parameters_schema") != 1:
        raise ValueError("agreement parameter schema is missing")
    view = {
        "M": int(summary.get("M", 0)),
        "n_r": int(summary.get("n_r", 0)),
        "t_r": int(summary.get("t_r", -1)),
        "n_a": int(summary.get("n_a", 0)),
        "t_a": int(summary.get("t_a", -1)),
        "m_a": int(summary.get("m_a", 0)),
        "quorum_h": int(summary.get("quorum_h", 0)),
        "agreement_t_impl": int(summary.get("agreement_t_impl", -1)),
        "coin_c": int(summary.get("coin_c", 0)),
        "coin_t": int(summary.get("coin_t", -1)),
        "coin_k": int(summary.get("coin_k", 0)),
        "coverage_threshold": int(summary.get("coverage_threshold", -1)),
    }
    if not 0 <= view["t_r"] or not 2 * view["t_r"] < view["n_r"] <= view["M"]:
        raise ValueError("invalid resharing parameters")
    if not 0 < view["n_a"] <= view["m_a"] <= view["M"]:
        raise ValueError("invalid agreement committee parameters")
    if 3 * view["t_a"] >= view["n_a"]:
        raise ValueError("paper agreement bound t_a < n_a/3 is violated")
    if view["quorum_h"] != view["n_a"] - view["t_a"]:
        raise ValueError("quorum_h must equal n_a-t_a")
    if view["agreement_t_impl"] != view["m_a"] - view["quorum_h"]:
        raise ValueError("agreement_t_impl must equal m_a-quorum_h")
    if 2 * view["quorum_h"] - view["m_a"] <= view["t_a"]:
        raise ValueError("agreement quorum intersection bound is violated")
    if int(summary.get("agreement_t", -1)) != view["t_a"]:
        raise ValueError("agreement_t must expose the paper t_a")
    if view["coverage_threshold"] != view["t_a"] + 1:
        raise ValueError("coverage threshold mismatch")
    if not (
        0
        <= view["coin_t"]
        < view["coin_k"]
        <= view["coin_c"] - view["coin_t"]
        and view["coin_c"] <= view["M"]
    ):
        raise ValueError("invalid common-coin parameters")
    if int(summary.get("expected_committee_size", 0)) != view["n_a"]:
        raise ValueError("expected_committee_size alias does not match n_a")
    if int(summary.get("committee_m", 0)) != view["m_a"]:
        raise ValueError("committee_m alias does not match m_a")
    return view


def _parameter_tag(parameters, coin_config):
    if not parameters:
        return None
    digest = (coin_config or {}).get("public_key_digest") or "missing"
    return (
        f"M{parameters['M']}-nr{parameters['n_r']}-tr{parameters['t_r']}"
        f"-na{parameters['n_a']}-ta{parameters['t_a']}"
        f"-ma{parameters['m_a']}-h{parameters['quorum_h']}"
        f"-cc{parameters['coin_c']}-ct{parameters['coin_t']}"
        f"-ck{parameters['coin_k']}"
        f"-coin{digest[:12]}"
    )


def analyze_pair(
    *,
    dag,
    acs,
    fixture: A2DPSSPairedFixture,
    expected_nodes: int,
    dag_exit_code: int = 0,
    acs_exit_code: int = 0,
):
    failures = []
    if dag_exit_code:
        failures.append(f"DAG runner exited with status {dag_exit_code}")
    if acs_exit_code:
        failures.append(f"ACS runner exited with status {acs_exit_code}")
    expected_node_ids = set(range(expected_nodes))
    if set(dag) != expected_node_ids:
        failures.append(f"DAG summary nodes {sorted(dag)} != expected")
    if set(acs) != expected_node_ids:
        failures.append(f"ACS summary nodes {sorted(acs)} != expected")

    justifier = A2DPSSJustifierManager([fixture.context])
    selected = justifier.select_justified_subset(
        fixture.context.build_qid(), fixture.public_entries()
    )
    if selected is None:
        failures.append("fixture has no canonical threshold T")
        expected_public = None
    else:
        public = A2DPSSPublicFinalizer(justifier).on_deliver(
            fixture.qid(), selected
        )
        expected_public = _public_view(public.to_obj() if public else None)

    expected_fixture_digests = sorted(fixture.contribution_digests())
    expected_successors = set(fixture.successor_node_ids())
    all_summaries = list(dag.values()) + list(acs.values())
    parameter_views = []
    for summary in all_summaries:
        protocol = summary.get("protocol")
        node = summary.get("node")
        label = f"{protocol}/node-{node}"
        checks = {
            "paired mode": summary.get("a2dpss_paired") is True,
            "paired latency view": (
                summary.get("a2dpss_latency_view") == "paired-tob"
            ),
            "fixture digest": (
                summary.get("a2dpss_fixture_digest") == fixture.digest()
            ),
            "trial id": summary.get("a2dpss_trial_id") == fixture.trial_id,
            "trial epoch": (
                summary.get("a2dpss_trial_epoch") == fixture.trial_epoch
                and summary.get("epoch") == fixture.trial_epoch
            ),
            "qid": summary.get("a2dpss_qid") == fixture.qid(),
            "context": (
                summary.get("a2dpss_context_digest")
                == fixture.context.context_digest
            ),
            "source holders": (
                summary.get("a2dpss_source_holder_ids")
                == list(fixture.source_holder_ids())
            ),
            "successor roles": (
                summary.get("a2dpss_successor_node_ids")
                == list(fixture.successor_node_ids())
            ),
            "fixture contributions": (
                sorted(summary.get("a2dpss_fixture_contribution_digests", []))
                == expected_fixture_digests
            ),
            "fixture holder metrics": (
                summary.get("a2dpss_fixture_holder_metrics")
                == fixture.holder_metrics_summary()
            ),
            "observed contributions": (
                sorted(summary.get("a2dpss_observed_contribution_digests", []))
                == expected_fixture_digests
            ),
            "threshold coin": summary.get("coin_mode") == "threshold-bls",
            "role-only trace": (
                summary.get("role_trace_source") == "recorded"
                and summary.get("role_trace_format") == "role-only"
                and summary.get("role_trace_digest") is not None
            ),
            "no protocol replay": (
                summary.get("coin_mode_forced_by_replay") is False
                and summary.get("ideal_coin_trace_source") != "recorded"
            ),
            "one frozen handover": (
                (summary.get("a2dpss_handover_metrics") or {}).get("frozen")
                is True
            ),
        }
        try:
            parameter_views.append(_phase6_parameter_view(summary))
        except (TypeError, ValueError) as exc:
            failures.append(f"{label}: invalid agreement parameters: {exc}")
        if protocol == "atcaas-dag-tob":
            checks["one public finalization"] = (
                len(_dag_public_results(summary)) == 1
            )
            checks["DAG causal suppression metrics"] = (
                (summary.get("dag_selectblock_metrics") or {}).get(
                    "applicable"
                )
                is True
            )
        elif protocol == "async-yoso-tob":
            checks["one public finalization"] = (
                len(
                    [
                        event
                        for event in summary.get("a2dpss_events", [])
                        if event.get("event") == "public_finalized"
                    ]
                )
                == 1
                and summary.get("a2dpss_public_result") is not None
            )
            checks["ACS quorum collection"] = (
                summary.get("acs_collect_threshold")
                == summary.get("quorum_h")
            )
            checks["DAG suppression not applicable to ACS"] = (
                (summary.get("dag_selectblock_metrics") or {}).get(
                    "applicable"
                )
                is False
            )
        for name, passed in checks.items():
            if not passed:
                failures.append(f"{label}: {name} mismatch")

    trace_digests = {
        summary.get("role_trace_digest") for summary in all_summaries
    }
    if len(trace_digests) != 1 or None in trace_digests:
        failures.append("DAG and ACS did not use the same role trace")
    committee_selection_modes = {
        summary.get("committee_selection_mode") for summary in all_summaries
    }
    if committee_selection_modes != {"fixed-size"}:
        failures.append(
            "paired latency runs must use fixed-size committee fixtures"
        )
    fixed_committee_seeds = {
        summary.get("fixed_committee_seed") for summary in all_summaries
    }
    if (
        len(fixed_committee_seeds) != 1
        or None in fixed_committee_seeds
        or fixture.trial_epoch not in fixed_committee_seeds
    ):
        failures.append(
            "DAG and ACS fixed-size committee fixture seeds differ"
        )
    coin_views = {
        json.dumps(_coin_view(summary), sort_keys=True)
        for summary in all_summaries
    }
    if len(coin_views) != 1:
        failures.append("DAG and ACS threshold-coin configurations differ")
    canonical_parameter_views = {
        json.dumps(view, sort_keys=True) for view in parameter_views
    }
    if (
        len(parameter_views) != len(all_summaries)
        or len(canonical_parameter_views) != 1
    ):
        failures.append("DAG and ACS Phase-6 protocol parameters differ")

    dag_public = {
        json.dumps(view, sort_keys=True)
        for summary in dag.values()
        for view in _dag_public_results(summary)
    }
    if len(dag_public) != 1 or (
        expected_public is not None
        and json.dumps(expected_public, sort_keys=True) not in dag_public
    ):
        failures.append("DAG did not finalize exactly the canonical public result")
    acs_public = {
        json.dumps(_public_view(summary.get("a2dpss_public_result")), sort_keys=True)
        for summary in acs.values()
        if summary.get("a2dpss_public_result") is not None
    }
    if len(acs_public) != 1 or (
        expected_public is not None
        and json.dumps(expected_public, sort_keys=True) not in acs_public
    ):
        failures.append("ACS did not finalize exactly the canonical public result")
    if dag_public != acs_public:
        failures.append("DAG and ACS public finalization results differ")

    dag_reconstructed = {
        node
        for node, summary in dag.items()
        if len(
            [
                event
                for event in summary.get("a2dpss_events", [])
                if event.get("event") == "reconstructed"
            ]
        ) == 1
    }
    acs_reconstructed = {
        node
        for node, summary in acs.items()
        if (summary.get("a2dpss_private_reconstruction") or {}).get(
            "completed"
        )
        is True
    }
    if dag_reconstructed != expected_successors:
        failures.append("DAG successor reconstruction partition mismatch")
    if acs_reconstructed != expected_successors:
        failures.append("ACS successor reconstruction partition mismatch")

    dag_latencies = {
        node: float(value)
        for node, summary in dag.items()
        if node in expected_successors and (value := _latency(summary)) is not None
    }
    acs_latencies = {
        node: float(value)
        for node, summary in acs.items()
        if node in expected_successors and (value := _latency(summary)) is not None
    }
    if set(dag_latencies) != expected_successors:
        failures.append("DAG successor latency set is incomplete")
    if set(acs_latencies) != expected_successors:
        failures.append("ACS successor latency set is incomplete")

    def _aggregate(protocol, summaries):
        if not summaries:
            return None
        try:
            return aggregate_a2dpss_computation_metrics(
                {
                    node: summary["a2dpss_computation_metrics"]
                    for node, summary in summaries.items()
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(
                f"{protocol} computation metrics cannot be aggregated: {exc}"
            )
            return None

    dag_computation = _aggregate("DAG", dag)
    acs_computation = _aggregate("ACS", acs)
    coin_config = _coin_view(all_summaries[0]) if all_summaries else None
    protocol_parameters = parameter_views[0] if parameter_views else None

    return {
        "schema": "atcaas-a2dpss-paired-trial-summary/v1",
        "status": "passed" if not failures else "failed",
        "trial_id": fixture.trial_id,
        "trial_epoch": fixture.trial_epoch,
        "fixture_digest": fixture.digest(),
        "qid": fixture.qid(),
        "context_digest": fixture.context.context_digest,
        "canonical_T_contribution_digests": (
            expected_public or {}
        ).get("accepted_contribution_digests", []),
        "fixture_holder_generation_metrics": (
            fixture.holder_metrics_summary()
        ),
        "public_result": expected_public,
        "successor_nodes": sorted(expected_successors),
        "role_trace_digest": next(iter(trace_digests), None),
        "committee_selection_mode": next(
            iter(committee_selection_modes), None
        ),
        "fixed_committee_seed": next(iter(fixed_committee_seeds), None),
        "parameter_tag": _parameter_tag(protocol_parameters, coin_config),
        "coin_config": coin_config,
        "protocol_parameters": protocol_parameters,
        "dag": {
            "summary_nodes": sorted(dag),
            "successor_latencies_sec": dag_latencies,
            "successor_max_latency_sec": (
                max(dag_latencies.values()) if dag_latencies else None
            ),
            "network_computation_metrics": dag_computation,
        },
        "acs": {
            "summary_nodes": sorted(acs),
            "successor_latencies_sec": acs_latencies,
            "successor_max_latency_sec": (
                max(acs_latencies.values()) if acs_latencies else None
            ),
            "network_computation_metrics": acs_computation,
        },
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dag-glob", required=True)
    parser.add_argument("--acs-glob", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--expected-nodes", type=int, required=True)
    parser.add_argument("--dag-exit-code", type=int, default=0)
    parser.add_argument("--acs-exit-code", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = analyze_pair(
        dag=_read_group(args.dag_glob, read_dag),
        acs=_read_group(args.acs_glob, read_acs),
        fixture=A2DPSSPairedFixture.from_file(args.fixture),
        expected_nodes=args.expected_nodes,
        dag_exit_code=args.dag_exit_code,
        acs_exit_code=args.acs_exit_code,
    )
    Path(args.output).write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
