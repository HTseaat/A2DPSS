"""Public, immutable contributions for independent A²DPSS handovers.

Each query has a fresh secret and sharing.  Both ordering backends replay the
same bytes; no holder shares or successor secret keys enter this artifact.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from dataclasses import dataclass
from typing import Mapping

from beaver.a2dpss import A2DPSSContext, A2DPSSHolderMetrics, make_a2dpss_contribution
from beaver.a2dpss_experiment import (
    A2DPSSFixtureEntry,
    A2DPSSPairedFixture,
    g_h_from_config_extra,
)
from beaver.experiment_trace import ExperimentWorkloadTrace
from beaver.handover_fixtures import build_handover_local_config, make_genesis_shares
from beaver.pedersen_shamir import FIELD_ORDER


FIXTURE_KIND = "thesis-ch4-a2dpss-pipeline-fixture"
FIXTURE_VERSION = 1
FIXTURE_PROTOCOL = "a2dpss"
FIXTURE_ALGORITHM = 4
INDEPENDENT_SERVICE_STATES = "independent-secrets"
_FIXTURE_DIGEST_DOMAIN = b"ATCaaS/A2DPSS/pipeline-fixture/v1\x00"


def _canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class A2DPSSFixtureQuery:
    instance: str
    context: A2DPSSContext
    entries: tuple[A2DPSSFixtureEntry, ...]

    def to_obj(self) -> dict:
        return {"instance": self.instance, "context": self.context.to_obj(),
                "entries": [entry.to_obj() for entry in self.entries]}


class A2DPSSPairedWorkload:
    """ExperimentWorkloadTrace-compatible pure resharing fixture."""

    def __init__(self, trace: ExperimentWorkloadTrace,
                 queries: tuple[A2DPSSFixtureQuery, ...], *,
                 service_state_mode: str = INDEPENDENT_SERVICE_STATES) -> None:
        if service_state_mode != INDEPENDENT_SERVICE_STATES:
            raise ValueError("A2DPSS pipeline requires independent secrets")
        if trace.messages_per_node != 1:
            raise ValueError("A2DPSS pipeline requires one contribution per holder/qid")
        expected = [instance for batch in trace.batches for instance in batch.instances]
        if [query.instance for query in queries] != expected:
            raise ValueError("fixture query order does not match workload trace")
        self.trace, self.trace_id = trace, trace.trace_id
        self.messages_per_node, self.batches = 1, trace.batches
        self.queries, self.service_state_mode = tuple(queries), service_state_mode
        self._queries_by_instance = {query.instance: query for query in queries}
        self._queries_by_qid = {query.context.encode_qid(): query for query in queries}
        if len(self._queries_by_qid) != len(queries):
            raise ValueError("fixture contains duplicate A2DPSS qids")
        commitments = [query.context.old_commitment_vector[0] for query in queries]
        if len(set(commitments)) != len(commitments):
            raise ValueError("independent handovers must use distinct initial commitments")
        self._payload_qids = {}
        for query in queries:
            # Reuse the single-handover fixture's canonical contribution and
            # metrics checks.  These digests use public values only.
            A2DPSSPairedFixture(
                trial_id=trace.trace_id, trial_epoch=trace.trace_id,
                context=query.context,
                initial_sharing_digest=hashlib.sha256(_canonical_json(
                    list(query.context.old_commitment_vector))).hexdigest(),
                successor_role_key_digest=hashlib.sha256(_canonical_json(
                    list(query.context.successor_rpk_list))).hexdigest(),
                entries=query.entries,
            )
            for entry in query.entries:
                key = hashlib.sha256(entry.payload).hexdigest()
                if key in self._payload_qids:
                    raise ValueError("fixture contains duplicate contribution bytes")
                self._payload_qids[key] = (entry.payload, query.context.encode_qid())

    @classmethod
    def from_obj(cls, value: Mapping) -> "A2DPSSPairedWorkload":
        expected = {"kind", "version", "protocol", "algorithm", "sharing_state_mode",
                    "workload_trace", "queries"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("A2DPSS pipeline fixture fields are not canonical")
        if (value["kind"] != FIXTURE_KIND or value["version"] != FIXTURE_VERSION
                or value["protocol"] != FIXTURE_PROTOCOL
                or value["algorithm"] != FIXTURE_ALGORITHM):
            raise ValueError("unsupported A2DPSS pipeline fixture metadata")
        queries = []
        for raw in value["queries"]:
            if set(raw) != {"instance", "context", "entries"}:
                raise ValueError("A2DPSS fixture query fields are not canonical")
            entries = []
            for entry in raw["entries"]:
                if set(entry) != {"sender", "mid", "payload_hex", "holder_metrics"}:
                    raise ValueError("A2DPSS fixture entry fields are not canonical")
                payload = bytes.fromhex(entry["payload_hex"])
                if payload.hex() != entry["payload_hex"]:
                    raise ValueError("fixture payload must use canonical lowercase hex")
                entries.append(A2DPSSFixtureEntry(
                    sender=entry["sender"], mid=entry["mid"], payload=payload,
                    holder_metrics=dict(entry["holder_metrics"])))
            queries.append(A2DPSSFixtureQuery(
                instance=raw["instance"], context=A2DPSSContext.from_obj(raw["context"]),
                entries=tuple(entries)))
        return cls(ExperimentWorkloadTrace.from_dict(value["workload_trace"]),
                   tuple(queries), service_state_mode=value["sharing_state_mode"])

    @classmethod
    def from_file(cls, path: str) -> "A2DPSSPairedWorkload":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_obj(json.load(handle))

    def to_fixture_obj(self) -> dict:
        return {"kind": FIXTURE_KIND, "version": FIXTURE_VERSION,
                "protocol": FIXTURE_PROTOCOL, "algorithm": FIXTURE_ALGORITHM,
                "sharing_state_mode": self.service_state_mode,
                "workload_trace": self.trace.to_dict(),
                "queries": [query.to_obj() for query in self.queries]}

    def to_dict(self) -> dict:
        return self.trace.to_dict()

    def digest(self) -> str:
        return hashlib.sha256(_FIXTURE_DIGEST_DOMAIN + _canonical_json(
            self.to_fixture_obj())).hexdigest()

    def batches_for_trigger(self, trigger: str):
        return self.trace.batches_for_trigger(trigger)

    def batch_for_instance(self, instance: str):
        return self.trace.batch_for_instance(instance)

    def qid(self, instance: str) -> str:
        return self._queries_by_instance[instance].context.encode_qid()

    def qids(self) -> tuple[str, ...]:
        return tuple(query.context.encode_qid() for query in self.queries)

    def context_for_qid(self, qid: str) -> A2DPSSContext:
        return self._queries_by_qid[qid].context

    def sender_ids(self, instance: str) -> tuple[int, ...]:
        return tuple(entry.sender for entry in self._queries_by_instance[instance].entries)

    def should_broadcast(self, instance: str, sender: int) -> bool:
        return int(sender) in self.sender_ids(instance)

    def _entry(self, instance: str, sender: int, index: int) -> A2DPSSFixtureEntry:
        if int(index) != 0:
            raise IndexError("A2DPSS pipeline has one contribution per holder/qid")
        for entry in self._queries_by_instance[instance].entries:
            if entry.sender == int(sender):
                return entry
        raise KeyError(f"node {sender} is not a holder for {instance}")

    def mid(self, instance: str, sender: int, index: int) -> str:
        return self._entry(instance, sender, index).mid

    def payload(self, instance: str, sender: int, index: int) -> bytes:
        return self._entry(instance, sender, index).payload

    def qid_for_payload(self, payload: bytes) -> str:
        expected, qid = self._payload_qids.get(
            hashlib.sha256(bytes(payload)).hexdigest(), (None, ""))
        return qid if expected == bytes(payload) else ""

    def expected_entry_count(self, instance: str) -> int:
        return len(self._queries_by_instance[instance].entries)

    def public_entries(self, qid: str) -> list[tuple[str, bytes]]:
        return [(entry.mid, entry.payload) for entry in self._queries_by_qid[qid].entries]

    def registrations(self) -> tuple[A2DPSSContext, ...]:
        return tuple(query.context for query in self.queries)

    def holder_metrics_summary(self) -> dict:
        entries = [{"instance": query.instance, "sender": entry.sender,
                    **dict(entry.holder_metrics)}
                   for query in self.queries for entry in query.entries]
        fields = ("pvss_seconds", "pvss_cpu_seconds", "serialization_seconds",
                  "contribution_bytes", "pvss_bytes")
        return {"entry_count": len(entries), "entries": entries,
                "totals": {field: sum(entry[field] for entry in entries) for field in fields}}


def build_paired_fixture(*, config_path: str, workload_trace_path: str,
                         service_state_mode: str = INDEPENDENT_SERVICE_STATES
                         ) -> A2DPSSPairedWorkload:
    if service_state_mode != INDEPENDENT_SERVICE_STATES:
        raise ValueError("A2DPSS pipeline requires independent secrets")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    trace = ExperimentWorkloadTrace.from_file(workload_trace_path)
    if trace.messages_per_node != 1:
        raise ValueError("A2DPSS pipeline requires messages_per_node=1")
    g_hex, h_hex = g_h_from_config_extra(config["extra"])
    local = build_handover_local_config(
        config["extra"], g_hex=g_hex, h_hex=h_hex, my_id=0,
        M=int(config.get("M", config["N"])),
        service_t=int(config.get("t_r", config["t"])), epoch=0)
    queries = []
    for batch in trace.batches:
        for instance in batch.instances:
            genesis = make_genesis_shares(
                g_hex=g_hex, h_hex=h_hex,
                secret=secrets.randbelow(FIELD_ORDER - 1) + 1,
                secret_hat=secrets.randbelow(FIELD_ORDER - 1) + 1,
                holder_ids=tuple(sorted(local.ctx.holder_eval_points)),
                service_t=local.ctx.service_t,
                holder_eval_points=local.ctx.holder_eval_points)
            context = A2DPSSContext(
                epoch=local.ctx.epoch, sid=f"a2dpss-pipeline-{trace.trace_id}-{instance}",
                source_committee=local.ctx.source_committee,
                successor_committee=local.ctx.successor_committee, transfer_seq=0,
                g_hex=g_hex, h_hex=h_hex, key_base=local.ctx.key_base,
                successor_rpk_list=local.ctx.successor_rpk_list,
                service_t=local.ctx.service_t,
                old_commitment_vector=genesis.old_commitment_vector,
                holder_eval_points=local.ctx.holder_eval_points,
                successor_role_indices=local.ctx.successor_role_indices)
            entries = []
            for holder in sorted(context.holder_eval_points):
                share_s, share_s_hat = genesis.holder_shares[holder]
                metrics = A2DPSSHolderMetrics()
                contribution = make_a2dpss_contribution(
                    context=context, holder_id=holder, share_s=share_s,
                    share_s_hat=share_s_hat, metrics=metrics)
                started = time.perf_counter()
                payload = contribution.to_bytes()
                entries.append(A2DPSSFixtureEntry(
                    sender=holder, mid=f"node_{holder}", payload=payload,
                    holder_metrics={
                        "pvss_seconds": metrics.pvss_seconds,
                        "pvss_cpu_seconds": metrics.pvss_cpu_seconds,
                        "serialization_seconds": time.perf_counter() - started,
                        "contribution_bytes": len(payload),
                        "pvss_bytes": len(contribution.transcript.to_bytes())}))
            queries.append(A2DPSSFixtureQuery(instance, context, tuple(entries)))
    return A2DPSSPairedWorkload(trace, tuple(queries))
