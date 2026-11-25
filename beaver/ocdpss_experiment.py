"""Shared real-OCDPSS workload fixtures for paired DAG/ACS experiments.

OCDPSS bundle construction is randomized.  A paired comparison must therefore
construct each public bundle once and replay exactly the same bytes through
both ordering protocols.  The fixture intentionally contains only public
contexts, bundles, and generation timings; holder shares and successor secret
keys are never serialized.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from beaver.experiment_trace import ExperimentWorkloadTrace
from beaver.g1_backend import G1Point
from beaver.ocdpss import (
    OCDPSSBundle,
    OCDPSSContext,
    OCDPSSHolderMetrics,
    make_ocdpss_bundle,
)
from beaver.operation_share import ThresholdExponentiationOperation
from beaver.sigma import hash_to_scalar
from beaver.handover_fixtures import (
    build_handover_local_config,
    make_genesis_shares,
)
from beaver.pedersen_shamir import FIELD_ORDER


FIXTURE_KIND = "atcaas-ocdpss-paired-fixture"
FIXTURE_VERSION = 3
FIXTURE_PROTOCOL = "ocdpss"
FIXTURE_ALGORITHM = 5
FIXTURE_EXTENDS_PROTOCOL = "a2dpss"
_FIXTURE_DIGEST_DOMAIN = b"ATCaaS/OCDPSS/paired-fixture/v3\x00"
INDEPENDENT_SERVICE_STATES = "independent-keys"
LEGACY_SHARED_SERVICE_STATE = "shared-genesis"


def _canonical_json(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _raw_g1(point: Mapping) -> str:
    return (
        b"\x04"
        + int(point["X"]).to_bytes(48, byteorder="big")
        + int(point["Y"]).to_bytes(48, byteorder="big")
    ).hex()


def g_h_from_config_extra(extra: Mapping) -> tuple[str, str]:
    srs = json.loads(base64.b64decode(extra["SRS"]).decode("utf-8"))
    return _raw_g1(srs["Pk"]["G1_g"][0]), _raw_g1(srs["Pk"]["G1_h"][0])


def make_paired_operation_adapter(
    g_hex: str,
    h_hex: str,
) -> ThresholdExponentiationOperation:
    generator = G1Point.from_hex(g_hex)

    def operation_base(value: bytes) -> G1Point:
        return generator * hash_to_scalar(
            "atcaas-ocdpss-paired-operation-base-v1",
            bytes(value).hex(),
            nonzero=True,
        )

    return ThresholdExponentiationOperation(
        operation_id="atcaas-threshold-sign-v1",
        g_hex=g_hex,
        h_hex=h_hex,
        base_deriver=operation_base,
    )


@dataclass(frozen=True)
class OCDPSSFixtureEntry:
    sender: int
    mid: str
    payload: bytes
    holder_metrics: Mapping[str, float | int]

    def to_obj(self) -> dict:
        return {
            "sender": self.sender,
            "mid": self.mid,
            "payload_hex": self.payload.hex(),
            "holder_metrics": dict(self.holder_metrics),
        }


@dataclass(frozen=True)
class OCDPSSFixtureQuery:
    instance: str
    context: OCDPSSContext
    entries: tuple[OCDPSSFixtureEntry, ...]

    def to_obj(self) -> dict:
        return {
            "instance": self.instance,
            "context": self.context.to_obj(),
            "entries": [entry.to_obj() for entry in self.entries],
        }


class OCDPSSPairedWorkload:
    """ExperimentWorkloadTrace-compatible view over immutable OCDPSS bytes."""

    def __init__(
        self,
        trace: ExperimentWorkloadTrace,
        queries: tuple[OCDPSSFixtureQuery, ...],
        *,
        service_state_mode: str = INDEPENDENT_SERVICE_STATES,
    ) -> None:
        if service_state_mode not in {
            INDEPENDENT_SERVICE_STATES,
            LEGACY_SHARED_SERVICE_STATE,
        }:
            raise ValueError("unknown OCDPSS paired service-state mode")
        self.trace = trace
        self.trace_id = trace.trace_id
        self.messages_per_node = 1
        self.batches = trace.batches
        expected_instances = [
            instance for batch in trace.batches for instance in batch.instances
        ]
        if [query.instance for query in queries] != expected_instances:
            raise ValueError("fixture query order does not match workload trace")
        self.queries = queries
        self.service_state_mode = service_state_mode
        self._queries_by_instance = {query.instance: query for query in queries}
        self._queries_by_qid = {
            query.context.encode_qid(): query for query in queries
        }
        if len(self._queries_by_qid) != len(queries):
            raise ValueError("fixture contains duplicate OCDPSS qids")
        if service_state_mode == INDEPENDENT_SERVICE_STATES:
            maintained_keys = [
                query.context.old_commitment_vector[0]
                for query in queries
            ]
            if len(set(maintained_keys)) != len(maintained_keys):
                raise ValueError(
                    "independent OCDPSS services must use distinct maintained keys"
                )
        self._payload_qids = {}
        for query in queries:
            expected_senders = set(query.context.holder_eval_points)
            actual_senders = {entry.sender for entry in query.entries}
            if actual_senders != expected_senders:
                raise ValueError(
                    f"fixture sender set mismatch for {query.instance}: "
                    f"{sorted(actual_senders)} != {sorted(expected_senders)}"
                )
            if len(actual_senders) != len(query.entries):
                raise ValueError(f"fixture has duplicate senders for {query.instance}")
            for entry in query.entries:
                bundle = OCDPSSBundle.from_bytes(entry.payload)
                if bundle.context_digest != query.context.context_digest:
                    raise ValueError("fixture bundle/context digest mismatch")
                if bundle.holder_id != entry.sender:
                    raise ValueError("fixture bundle/sender mismatch")
                if entry.mid != f"node_{entry.sender}":
                    raise ValueError("fixture entry mid must bind holder identity")
                payload_digest = hashlib.sha256(entry.payload).hexdigest()
                if payload_digest in self._payload_qids:
                    raise ValueError("fixture contains duplicate bundle bytes")
                self._payload_qids[payload_digest] = (
                    entry.payload,
                    query.context.encode_qid(),
                )

    @classmethod
    def from_obj(cls, value: Mapping) -> "OCDPSSPairedWorkload":
        if not isinstance(value, Mapping):
            raise ValueError("OCDPSS paired fixture must be an object")
        expected_v2 = {
            "kind",
            "version",
            "protocol",
            "algorithm",
            "extends_protocol",
            "workload_trace",
            "queries",
        }
        expected_v3 = expected_v2 | {"service_state_mode"}
        version = value.get("version")
        expected = expected_v2 if version == 2 else expected_v3
        if set(value) != expected:
            raise ValueError("OCDPSS paired fixture fields are not canonical")
        if value["kind"] != FIXTURE_KIND or version not in (2, FIXTURE_VERSION):
            raise ValueError("unsupported OCDPSS paired fixture kind/version")
        if (
            value["protocol"] != FIXTURE_PROTOCOL
            or value["algorithm"] != FIXTURE_ALGORITHM
            or value["extends_protocol"] != FIXTURE_EXTENDS_PROTOCOL
        ):
            raise ValueError(
                "OCDPSS paired fixture protocol metadata mismatch"
            )
        trace = ExperimentWorkloadTrace.from_dict(value["workload_trace"])
        queries = []
        for raw_query in value["queries"]:
            if set(raw_query) != {"instance", "context", "entries"}:
                raise ValueError("OCDPSS fixture query fields are not canonical")
            context = OCDPSSContext.from_obj(raw_query["context"])
            entries = []
            for raw_entry in raw_query["entries"]:
                if set(raw_entry) != {
                    "sender",
                    "mid",
                    "payload_hex",
                    "holder_metrics",
                }:
                    raise ValueError("OCDPSS fixture entry fields are not canonical")
                payload = bytes.fromhex(raw_entry["payload_hex"])
                if payload.hex() != raw_entry["payload_hex"]:
                    raise ValueError("fixture payload must use canonical lowercase hex")
                entries.append(
                    OCDPSSFixtureEntry(
                        sender=int(raw_entry["sender"]),
                        mid=str(raw_entry["mid"]),
                        payload=payload,
                        holder_metrics=dict(raw_entry["holder_metrics"]),
                    )
                )
            queries.append(
                OCDPSSFixtureQuery(
                    instance=str(raw_query["instance"]),
                    context=context,
                    entries=tuple(entries),
                )
            )
        return cls(
            trace,
            tuple(queries),
            service_state_mode=(
                LEGACY_SHARED_SERVICE_STATE
                if version == 2
                else value["service_state_mode"]
            ),
        )

    @classmethod
    def from_file(cls, path: str) -> "OCDPSSPairedWorkload":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_obj(json.load(handle))

    def to_fixture_obj(self) -> dict:
        return {
            "kind": FIXTURE_KIND,
            "version": FIXTURE_VERSION,
            "protocol": FIXTURE_PROTOCOL,
            "algorithm": FIXTURE_ALGORITHM,
            "extends_protocol": FIXTURE_EXTENDS_PROTOCOL,
            "service_state_mode": self.service_state_mode,
            "workload_trace": self.trace.to_dict(),
            "queries": [query.to_obj() for query in self.queries],
        }

    def to_dict(self) -> dict:
        return self.trace.to_dict()

    def digest(self) -> str:
        return hashlib.sha256(
            _FIXTURE_DIGEST_DOMAIN + _canonical_json(self.to_fixture_obj())
        ).hexdigest()

    def batches_for_trigger(self, trigger: str):
        return self.trace.batches_for_trigger(trigger)

    def batch_for_instance(self, instance: str):
        return self.trace.batch_for_instance(instance)

    def qid(self, instance: str) -> str:
        return self._queries_by_instance[instance].context.encode_qid()

    def qids(self) -> tuple[str, ...]:
        return tuple(query.context.encode_qid() for query in self.queries)

    def context_for_qid(self, qid: str) -> OCDPSSContext:
        return self._queries_by_qid[qid].context

    def sender_ids(self, instance: str) -> tuple[int, ...]:
        return tuple(entry.sender for entry in self._queries_by_instance[instance].entries)

    def should_broadcast(self, instance: str, sender: int) -> bool:
        return int(sender) in set(self.sender_ids(instance))

    def _entry(self, instance: str, sender: int, index: int) -> OCDPSSFixtureEntry:
        if int(index) != 0:
            raise IndexError("OCDPSS paired workload has one bundle per holder/qid")
        for entry in self._queries_by_instance[instance].entries:
            if entry.sender == int(sender):
                return entry
        raise KeyError(f"node {sender} is not a holder for fixture query {instance}")

    def mid(self, instance: str, sender: int, index: int) -> str:
        return self._entry(instance, sender, index).mid

    def payload(self, instance: str, sender: int, index: int) -> bytes:
        return self._entry(instance, sender, index).payload

    def qid_for_payload(self, payload: bytes) -> str:
        digest = hashlib.sha256(bytes(payload)).hexdigest()
        expected, qid = self._payload_qids.get(digest, (None, ""))
        return qid if expected == bytes(payload) else ""

    def expected_entry_count(self, instance: str) -> int:
        return len(self._queries_by_instance[instance].entries)

    def public_entries(self, qid: str) -> list[tuple[str, bytes]]:
        return [
            (entry.mid, entry.payload)
            for entry in self._queries_by_qid[qid].entries
        ]

    def registrations(self):
        adapters = {}
        out = []
        for query in self.queries:
            key = (query.context.g_hex, query.context.h_hex)
            adapter = adapters.setdefault(
                key,
                make_paired_operation_adapter(*key),
            )
            out.append((query.context, adapter))
        return tuple(out)

    def holder_metrics_summary(self) -> dict:
        fields = (
            "operation_seconds",
            "pvss_seconds",
            "serialization_seconds",
            "bundle_bytes",
            "pvss_bytes",
            "operation_share_bytes",
            "operation_proof_bytes",
        )
        totals = {field: 0 for field in fields}
        entries = []
        for query in self.queries:
            for entry in query.entries:
                metrics = dict(entry.holder_metrics)
                entries.append(
                    {
                        "instance": query.instance,
                        "sender": entry.sender,
                        **metrics,
                    }
                )
                for field in fields:
                    totals[field] += metrics[field]
        return {"entry_count": len(entries), "totals": totals, "entries": entries}


def build_paired_fixture(
    *,
    config_path: str,
    workload_trace_path: str,
    service_state_mode: str = INDEPENDENT_SERVICE_STATES,
) -> OCDPSSPairedWorkload:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    trace = ExperimentWorkloadTrace.from_file(workload_trace_path)
    if trace.messages_per_node != 1:
        raise ValueError("OCDPSS paired fixture requires messages_per_node=1")
    extra = config["extra"]
    g_hex, h_hex = g_h_from_config_extra(extra)
    M = int(config.get("M", config["N"]))
    service_t = int(config.get("t_r", config["t"]))
    local = build_handover_local_config(
        extra,
        g_hex=g_hex,
        h_hex=h_hex,
        my_id=0,
        M=M,
        service_t=service_t,
        epoch=0,
    )
    adapter = make_paired_operation_adapter(g_hex, h_hex)
    if service_state_mode not in {
        INDEPENDENT_SERVICE_STATES,
        LEGACY_SHARED_SERVICE_STATE,
    }:
        raise ValueError("unknown OCDPSS paired service-state mode")
    shared_holder_shares = dict(local.holder_shares)
    if service_state_mode == LEGACY_SHARED_SERVICE_STATE:
        missing_holders = set(local.ctx.holder_eval_points).difference(
            shared_holder_shares
        )
        config_dir = Path(config_path).resolve().parent
        for holder_id in sorted(missing_holders):
            holder_path = config_dir / f"local.{holder_id}.json"
            if not holder_path.is_file():
                raise ValueError(f"missing node-local holder config {holder_path}")
            holder_config = json.loads(holder_path.read_text(encoding="utf-8"))
            holder_local = build_handover_local_config(
                holder_config["extra"],
                g_hex=g_hex,
                h_hex=h_hex,
                my_id=holder_id,
                M=M,
                service_t=service_t,
                epoch=0,
            )
            if holder_local.my_holder_share is None:
                raise ValueError(f"node-local holder {holder_id} has no share")
            shared_holder_shares[holder_id] = holder_local.my_holder_share
    queries = []
    for batch in trace.batches:
        for instance in batch.instances:
            operation_input = (
                f"atcaas-ocdpss-paired:{trace.trace_id}:{batch.batch_id}:{instance}"
            ).encode("utf-8")
            if service_state_mode == INDEPENDENT_SERVICE_STATES:
                genesis = make_genesis_shares(
                    g_hex=g_hex,
                    h_hex=h_hex,
                    secret=secrets.randbelow(FIELD_ORDER - 1) + 1,
                    secret_hat=secrets.randbelow(FIELD_ORDER - 1) + 1,
                    holder_ids=tuple(sorted(local.ctx.holder_eval_points)),
                    service_t=service_t,
                    holder_eval_points=local.ctx.holder_eval_points,
                )
                holder_shares = genesis.holder_shares
                old_commitment_vector = genesis.old_commitment_vector
            else:
                holder_shares = shared_holder_shares
                old_commitment_vector = local.ctx.old_commitment_vector
            context = OCDPSSContext(
                epoch=local.ctx.epoch,
                sid=f"ocdpss-paired-{trace.trace_id}-{instance}",
                source_committee=local.ctx.source_committee,
                successor_committee=local.ctx.successor_committee,
                transfer_seq=0,
                g_hex=g_hex,
                h_hex=h_hex,
                key_base=local.ctx.key_base,
                successor_rpk_list=local.ctx.successor_rpk_list,
                service_t=local.ctx.service_t,
                old_commitment_vector=old_commitment_vector,
                holder_eval_points=local.ctx.holder_eval_points,
                successor_role_indices=local.ctx.successor_role_indices,
                operation_id=adapter.operation_id,
                operation_input=operation_input,
                operation_public_parameters=adapter.public_context(operation_input),
            )
            entries = []
            for holder_id in sorted(context.holder_eval_points):
                share_s, share_s_hat = holder_shares[holder_id]
                metrics = OCDPSSHolderMetrics()
                bundle = make_ocdpss_bundle(
                    context=context,
                    holder_id=holder_id,
                    share_s=share_s,
                    share_s_hat=share_s_hat,
                    operation_adapter=adapter,
                    metrics=metrics,
                )
                started = time.perf_counter()
                payload = bundle.to_bytes()
                metrics.serialization_seconds = time.perf_counter() - started
                metrics.bundle_bytes = len(payload)
                entries.append(
                    OCDPSSFixtureEntry(
                        sender=holder_id,
                        mid=f"node_{holder_id}",
                        payload=payload,
                        holder_metrics={
                            "operation_seconds": metrics.operation_seconds,
                            "pvss_seconds": metrics.pvss_seconds,
                            "serialization_seconds": metrics.serialization_seconds,
                            "bundle_bytes": metrics.bundle_bytes,
                            "pvss_bytes": len(bundle.transcript.to_bytes()),
                            "operation_share_bytes": len(
                                _canonical_json(dict(bundle.operation_share))
                            ),
                            "operation_proof_bytes": len(
                                _canonical_json(dict(bundle.operation_proof))
                            ),
                        },
                    )
                )
            queries.append(
                OCDPSSFixtureQuery(
                    instance=instance,
                    context=context,
                    entries=tuple(entries),
                )
            )
    return OCDPSSPairedWorkload(
        trace,
        tuple(queries),
        service_state_mode=service_state_mode,
    )


__all__ = [
    "FIXTURE_KIND",
    "FIXTURE_VERSION",
    "FIXTURE_PROTOCOL",
    "FIXTURE_ALGORITHM",
    "FIXTURE_EXTENDS_PROTOCOL",
    "INDEPENDENT_SERVICE_STATES",
    "LEGACY_SHARED_SERVICE_STATE",
    "OCDPSSFixtureEntry",
    "OCDPSSFixtureQuery",
    "OCDPSSPairedWorkload",
    "build_paired_fixture",
    "g_h_from_config_extra",
    "make_paired_operation_adapter",
]
