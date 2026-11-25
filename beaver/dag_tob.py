"""
ATCaaS DAG-based Total-Order Broadcast state machine.

This module implements the local state from ATCaaS Algorithm 2 and exposes
small seams for networking, VRF role traces, and asynchronous common coin:

    Pool -> DAG/buffer -> leadersStack/orderedVertices -> per-query delivery

The thresholds use ``quorum_h`` and the separately supplied ``agreement_t``.
They are the engineering counterparts of the agreement committee's quorum
and fault bound under Bernoulli VRF sortition; neither is derived from the
long-lived population size ``M`` or from a service's resharing threshold.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from beaver.protocol_trace import (
    AllMembersCommitteeTrace,
    CommitteeRoleTrace,
    DeterministicIdealCoin,
    IdealCoin,
    RecordedCommitteeRoleTrace,
)


VertexId = Tuple[int, int]  # (round, source)
PoolEntry = Tuple[str, bytes]  # (mid, payload)
Reference = str  # a reference is just a mid
# RefBlock travels in the vertex / DAG / wire: it carries references only.
RefBlock = Dict[str, List[Reference]]  # qid -> [mid, ...]
# Block is a resolved qid mapping.  Vertices may contain several qids, while
# each R3 delivery record is a singleton Block kept for compatibility readers.
Block = Dict[str, List[PoolEntry]]  # qid -> [(mid, payload), ...]


@dataclass(frozen=True)
class DeliveryEvent:
    """Authoritative per-query output emitted from one ordered vertex."""

    sequence: int
    qid: str
    entries: Tuple[PoolEntry, ...]
    ordered_vertex: VertexId
    committed_wave: int

    def to_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "qid": self.qid,
            "entries": [
                {"mid": mid, "payload": payload.hex()}
                for mid, payload in self.entries
            ],
            "ordered_vertex": {
                "round": self.ordered_vertex[0],
                "source": self.ordered_vertex[1],
            },
            "committed_wave": self.committed_wave,
        }


def _stable_entry(entry: PoolEntry) -> Tuple[str, str]:
    mid, payload = entry
    return mid, payload.hex()


def _restore_entry(entry: Sequence[str]) -> PoolEntry:
    mid, payload_hex = entry
    return str(mid), bytes.fromhex(payload_hex)


def canonical_block(block: Optional[Block]) -> Optional[Block]:
    """Return a deterministically ordered copy of a *resolved* block."""
    if not block:
        return None
    out: Block = {}
    for qid in sorted(block):
        entries = sorted(block[qid], key=_stable_entry)
        if entries:
            out[qid] = list(entries)
    return out or None


def canonical_ref_block(block: Optional[RefBlock]) -> Optional[RefBlock]:
    """Return a deterministically ordered copy of a reference block."""
    if not block:
        return None
    out: RefBlock = {}
    for qid in sorted(block):
        mids = sorted({str(mid) for mid in block[qid]})
        if mids:
            out[qid] = mids
    return out or None


@dataclass(frozen=True)
class Qid:
    """Structured TOB service identifier with stable string encoding."""

    kind: str
    fields: Tuple[Tuple[str, str], ...] = ()

    @classmethod
    def build(cls, kind: str, **fields) -> "Qid":
        return cls(
            kind=str(kind),
            fields=tuple(
                sorted((str(key), str(value)) for key, value in fields.items())
            ),
        )

    @classmethod
    def decode(cls, raw: str) -> "Qid":
        try:
            data = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return cls("demo", (("legacy", str(raw)),))
        if not isinstance(data, dict) or "kind" not in data:
            return cls("demo", (("legacy", str(raw)),))
        fields = data.get("fields", {})
        if not isinstance(fields, dict):
            fields = {}
        return cls.build(str(data["kind"]), **fields)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "fields": {key: value for key, value in self.fields},
        }

    def encode(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class TobMessage:
    """Structured payload wrapper for messages stored under a qid."""

    kind: str
    sender: int
    payload: Mapping
    proof: Optional[dict] = None

    def to_dict(self) -> dict:
        data = {
            "kind": self.kind,
            "sender": self.sender,
            "payload": dict(self.payload),
        }
        if self.proof is not None:
            data["proof"] = self.proof
        return data

    def to_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_bytes(cls, raw: bytes) -> "TobMessage":
        data = json.loads(raw.decode("utf-8"))
        return cls(
            kind=str(data["kind"]),
            sender=int(data["sender"]),
            payload=data.get("payload", {}),
            proof=data.get("proof"),
        )


@dataclass(frozen=True)
class Vertex:
    round: int
    source: int
    block: Optional[RefBlock]
    strong_edges: frozenset[VertexId]
    weak_edges: frozenset[VertexId]
    proof: Optional[dict] = None

    @property
    def vid(self) -> VertexId:
        return self.round, self.source

    def to_dict(self) -> dict:
        block_obj = None
        if self.block is not None:
            block_obj = {
                qid: list(mids)
                for qid, mids in sorted(self.block.items())
            }
        return {
            "round": self.round,
            "source": self.source,
            "block": block_obj,
            "strong_edges": [
                {"round": r, "source": s}
                for r, s in sorted(self.strong_edges)
            ],
            "weak_edges": [
                {"round": r, "source": s}
                for r, s in sorted(self.weak_edges)
            ],
            "proof": self.proof,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vertex":
        block_data = data.get("block")
        block: Optional[RefBlock] = None
        if block_data is not None:
            block = {
                str(qid): [str(mid) for mid in mids]
                for qid, mids in block_data.items()
            }
            block = canonical_ref_block(block)
        # ``edges`` is accepted as a read-only migration fallback for vertices
        # serialized by the pre-R1 implementation.  New vertices always emit
        # the split strong/weak representation.
        strong_edges_data = data.get("strong_edges", data.get("edges", []))
        strong_edges = frozenset(
            (int(edge["round"]), int(edge["source"]))
            for edge in strong_edges_data
        )
        weak_edges = frozenset(
            (int(edge["round"]), int(edge["source"]))
            for edge in data.get("weak_edges", [])
        )
        return cls(
            round=int(data["round"]),
            source=int(data["source"]),
            block=block,
            strong_edges=strong_edges,
            weak_edges=weak_edges,
            proof=data.get("proof"),
        )

    def to_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_bytes(cls, payload: bytes) -> "Vertex":
        return cls.from_dict(json.loads(payload.decode("utf-8")))


class WaitPredicate:
    """Interface for SelectBlock and the per-query predicate J."""

    def select_block(
        self,
        pool: Dict[str, Set[PoolEntry]],
        excluded_qids: Set[str],
    ) -> Optional[RefBlock]:
        raise NotImplementedError

    def verify_qid_set(
        self,
        qid: str,
        entries: Sequence[PoolEntry],
    ) -> bool:
        raise NotImplementedError


class Justifier:
    """Per-service evidence checker for one qid and a candidate entry set."""

    def select_justified_subset(
        self,
        qid: Qid,
        entries: Sequence[PoolEntry],
    ) -> Optional[List[PoolEntry]]:
        ordered = sorted(entries, key=_stable_entry)
        if self.verify_qid_set(qid, ordered):
            return list(ordered)
        return None

    def verify_qid_set(
        self,
        qid: Qid,
        entries: Sequence[PoolEntry],
    ) -> bool:
        raise NotImplementedError


class LegacyJustifierInterfaceError(TypeError):
    """A ledger-prefix predicate cannot be attached to current DAG-TOB."""


def _validate_current_justifier(kind: str, justifier: Justifier) -> None:
    """Fail fast unless ``justifier`` implements the per-query R5 contract.

    The current-paper interface is exactly ``(qid, entries)`` for both
    selection and verification.  In particular, a legacy predicate must not
    receive the observer's mutable delivery transcript as a hidden third
    input: two honest observers may have different local prefixes when they
    validate the same qid.
    """

    name = type(justifier).__name__
    if getattr(justifier, "legacy_ledger_prefix_dependent", False):
        raise LegacyJustifierInterfaceError(
            f"justifier {name} for kind {kind!r} is isolated because it "
            "depends on a legacy ledger prefix; migrate it to "
            "(qid, entries) or run it in an explicit legacy harness"
        )

    for method_name in ("select_justified_subset", "verify_qid_set"):
        method = getattr(justifier, method_name, None)
        if not callable(method):
            raise TypeError(f"justifier {name} has no callable {method_name}")
        signature = inspect.signature(method)
        parameters = tuple(signature.parameters.values())
        is_current = (
            tuple(parameter.name for parameter in parameters) == ("qid", "entries")
            and all(
                parameter.kind
                in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                }
                for parameter in parameters
            )
        )
        if not is_current:
            raise LegacyJustifierInterfaceError(
                f"justifier {name}.{method_name}{signature} does not implement "
                "the current (qid, entries) interface; ledger-prefix "
                "justifiers are isolated from current DAG-TOB"
            )


class DemoJustifier(Justifier):
    """Phase-4A demo service: any non-empty set is justified."""

    def verify_qid_set(
        self,
        qid: Qid,
        entries: Sequence[PoolEntry],
    ) -> bool:
        return bool(entries)


class JustifierRegistry:
    """Dispatch qid kinds to service-specific justifiers."""

    def __init__(self, default: Optional[Justifier] = None):
        self.default = default or DemoJustifier()
        _validate_current_justifier("<default>", self.default)
        self._by_kind: Dict[str, Justifier] = {}
        self._metrics = {
            "select": {
                "invocations": 0,
                "elapsed_sec": 0.0,
                "cpu_seconds": 0.0,
                "by_qid": {},
                "records": [],
            },
            "verify": {
                "invocations": 0,
                "elapsed_sec": 0.0,
                "cpu_seconds": 0.0,
                "by_qid": {},
                "records": [],
            },
        }

    def register(self, kind: str, justifier: Justifier) -> None:
        normalized_kind = str(kind)
        _validate_current_justifier(normalized_kind, justifier)
        self._by_kind[normalized_kind] = justifier

    def get(self, qid: Qid) -> Justifier:
        return self._by_kind.get(qid.kind, self.default)

    def select_justified_subset(
        self,
        qid_raw: str,
        entries: Sequence[PoolEntry],
    ) -> Optional[List[PoolEntry]]:
        qid = Qid.decode(qid_raw)
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        selected = None
        try:
            selected = self.get(qid).select_justified_subset(qid, entries)
            return selected
        finally:
            self._record_metric(
                "select",
                qid_raw,
                elapsed=time.perf_counter() - wall_started,
                cpu_seconds=time.process_time() - cpu_started,
                entry_count=len(entries),
                selected_entry_count=(
                    len(selected) if selected is not None else 0
                ),
                accepted=bool(selected),
            )

    def verify_qid_set(
        self,
        qid_raw: str,
        entries: Sequence[PoolEntry],
    ) -> bool:
        qid = Qid.decode(qid_raw)
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        accepted = False
        try:
            accepted = bool(self.get(qid).verify_qid_set(qid, entries))
            return accepted
        finally:
            self._record_metric(
                "verify",
                qid_raw,
                elapsed=time.perf_counter() - wall_started,
                cpu_seconds=time.process_time() - cpu_started,
                entry_count=len(entries),
                selected_entry_count=len(entries) if accepted else 0,
                accepted=accepted,
            )

    def _record_metric(
        self,
        phase: str,
        qid: str,
        *,
        elapsed: float,
        cpu_seconds: float,
        entry_count: int,
        selected_entry_count: int,
        accepted: bool,
    ) -> None:
        metric = self._metrics[phase]
        metric["invocations"] += 1
        metric["elapsed_sec"] += elapsed
        metric["cpu_seconds"] += cpu_seconds
        by_qid = metric["by_qid"].setdefault(
            qid,
            {
                "invocations": 0,
                "elapsed_sec": 0.0,
                "cpu_seconds": 0.0,
            },
        )
        by_qid["invocations"] += 1
        by_qid["elapsed_sec"] += elapsed
        by_qid["cpu_seconds"] += cpu_seconds
        metric["records"].append(
            {
                "invocation": metric["invocations"],
                "qid": qid,
                "entry_count": int(entry_count),
                "selected_entry_count": int(selected_entry_count),
                "accepted": bool(accepted),
                "elapsed_sec": float(elapsed),
                "cpu_seconds": float(cpu_seconds),
            }
        )

    def metrics_snapshot(self) -> dict:
        """Return proposer-side selection and receiver-side J costs."""

        return {
            phase: {
                "invocations": metric["invocations"],
                "elapsed_sec": metric["elapsed_sec"],
                "cpu_seconds": metric["cpu_seconds"],
                "by_qid": {
                    qid: dict(values)
                    for qid, values in sorted(metric["by_qid"].items())
                },
                "records": [dict(record) for record in metric["records"]],
            }
            for phase, metric in self._metrics.items()
        }


class TypedWaitPredicate(WaitPredicate):
    """SelectBlock/J implementation backed by a JustifierRegistry."""

    def __init__(self, registry: Optional[JustifierRegistry] = None):
        self.registry = registry or JustifierRegistry()

    def select_block(
        self,
        pool: Dict[str, Set[PoolEntry]],
        excluded_qids: Set[str],
    ) -> Optional[RefBlock]:
        block: RefBlock = {}
        for qid in sorted(pool):
            if qid in excluded_qids:
                continue
            subset = self.registry.select_justified_subset(
                qid,
                sorted(pool[qid], key=_stable_entry),
            )
            if subset:
                # store references only (Algorithm 3, line 14)
                block[qid] = [mid for mid, _ in subset]
        return canonical_ref_block(block)

    def verify_qid_set(
        self,
        qid: str,
        entries: Sequence[PoolEntry],
    ) -> bool:
        return self.registry.verify_qid_set(qid, entries)


class TrivialWaitPredicate(WaitPredicate):
    """Phase-1 predicate: any non-empty qid set is valid."""

    def select_block(
        self,
        pool: Dict[str, Set[PoolEntry]],
        excluded_qids: Set[str],
    ) -> Optional[RefBlock]:
        block: RefBlock = {}
        for qid in sorted(pool):
            if qid in excluded_qids:
                continue
            entries = sorted(pool[qid], key=_stable_entry)
            if entries and self.verify_qid_set(qid, entries):
                # store references only (Algorithm 3, line 14)
                block[qid] = [mid for mid, _ in entries]
        return canonical_ref_block(block)

    def verify_qid_set(
        self,
        qid: str,
        entries: Sequence[PoolEntry],
    ) -> bool:
        return bool(entries)


class LeaderOracle:
    """Interface for wave leader selection."""

    def is_wave_ready(self, wave: int) -> bool:
        """Return whether leader selection can be read without blocking."""

        return True

    def get_wave_leader(self, dag_tob: "DagTOB", wave: int) -> Optional[VertexId]:
        raise NotImplementedError


class IdealCoinLeaderOracle(LeaderOracle):
    """Map ideal ``Coin.Get(wave)`` into a public VRF role namespace."""

    def __init__(
        self,
        coin: IdealCoin,
        committee_trace: CommitteeRoleTrace,
    ):
        self.coin = coin
        self.committee_trace = committee_trace

    def selected_vid(self, wave: int) -> VertexId:
        wave = int(wave)
        if wave <= 0:
            raise ValueError("wave must be positive")
        roles = self.committee_trace.roles(wave)
        leader_index = self.coin.get(wave) % len(roles)
        leader_round = 4 * (wave - 1) + 1
        return leader_round, roles[leader_index]

    def get_wave_leader(self, dag_tob: "DagTOB", wave: int) -> Optional[VertexId]:
        vid = self.selected_vid(wave)
        if not 0 <= vid[1] < dag_tob.M:
            raise ValueError(f"leader role {vid[1]} outside population M={dag_tob.M}")
        return vid if vid in dag_tob.DAG.get(vid[0], {}) else None


class ThresholdCoinLeaderOracle(LeaderOracle):
    """Async common-coin adapter with a synchronous DAG lookup cache.

    ``DagTOB`` remains a synchronous state machine.  The runner calls
    :meth:`prepare_wave` in a background task and later wakes the state
    machine with ``notify_wave_coin_ready``.  Historical leader lookups never
    start a second coin invocation; they only read this adapter's cache.
    """

    SESSION_ID = "ATCAAS-DAG-TOB-LEADER"
    # Match ObservedCoinOracle so the runner exposes one canonical digest for
    # the same secret coin output in both the call trace and wave trace.
    _OUTPUT_DIGEST_DOMAIN = b"ATCaaS/common-coin/observation/v1\x00"

    def __init__(self, coin_oracle, committee_trace: CommitteeRoleTrace):
        if not callable(getattr(coin_oracle, "next_coin", None)):
            raise TypeError("coin_oracle must provide async next_coin()")
        self.coin_oracle = coin_oracle
        self.committee_trace = committee_trace
        self._leaders: Dict[int, VertexId] = {}
        self._records: Dict[int, dict] = {}
        self._tasks: Dict[int, asyncio.Task] = {}

    def is_wave_ready(self, wave: int) -> bool:
        return int(wave) in self._leaders

    def _coin_id_digest(self, wave: int) -> Optional[str]:
        candidate = self.coin_oracle
        for _ in range(2):
            factory = getattr(candidate, "coin_id", None)
            if callable(factory):
                return factory(self.SESSION_ID, wave).digest.hex()
            candidate = getattr(candidate, "delegate", None)
            if candidate is None:
                break
        return None

    async def _prepare_wave_once(self, wave: int) -> VertexId:
        requested_at = time.time()
        started = time.perf_counter()
        coin = await self.coin_oracle.next_coin(self.SESSION_ID, wave)
        if not isinstance(coin, bytes):
            raise TypeError("common coin output must be bytes")
        roles = tuple(self.committee_trace.roles(wave))
        leader_index = int.from_bytes(coin, "big") % len(roles)
        leader = (4 * (wave - 1) + 1, roles[leader_index])
        self._leaders[wave] = leader
        self._records[wave] = {
            "wave": wave,
            "session_id": self.SESSION_ID,
            "attempt": wave,
            "coin_id_digest": self._coin_id_digest(wave),
            "coin_digest": hashlib.sha256(
                self._OUTPUT_DIGEST_DOMAIN + coin
            ).hexdigest(),
            "roles": list(roles),
            "leader_index": leader_index,
            "leader": {"round": leader[0], "source": leader[1]},
            "requested_at": requested_at,
            "ready_at": time.time(),
            "wait_ms": (time.perf_counter() - started) * 1000.0,
        }
        return leader

    async def prepare_wave(self, wave: int) -> VertexId:
        wave = int(wave)
        if wave <= 0:
            raise ValueError("wave must be positive")
        cached = self._leaders.get(wave)
        if cached is not None:
            return cached
        task = self._tasks.get(wave)
        if task is None:
            task = asyncio.create_task(self._prepare_wave_once(wave))
            self._tasks[wave] = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self._tasks.pop(wave, None)

    def selected_vid(self, wave: int) -> VertexId:
        wave = int(wave)
        if wave <= 0:
            raise ValueError("wave must be positive")
        try:
            return self._leaders[wave]
        except KeyError as exc:
            raise RuntimeError(f"common coin for wave {wave} is not ready") from exc

    def get_wave_leader(self, dag_tob: "DagTOB", wave: int) -> Optional[VertexId]:
        if not self.is_wave_ready(wave):
            return None
        vid = self.selected_vid(wave)
        if not 0 <= vid[1] < dag_tob.M:
            raise ValueError(f"leader role {vid[1]} outside population M={dag_tob.M}")
        return vid if vid in dag_tob.DAG.get(vid[0], {}) else None

    async def close(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def snapshot(self) -> dict:
        return {
            "session_id": self.SESSION_ID,
            "ready_waves": sorted(self._leaders),
            "pending_waves": sorted(self._tasks),
            "waves": [dict(self._records[wave]) for wave in sorted(self._records)],
        }


class DeterministicLeaderOracle(IdealCoinLeaderOracle):
    """Seeded ideal coin over an explicit, view-independent role namespace."""

    def __init__(
        self,
        epoch: str = "epoch0",
        committee_trace: Optional[CommitteeRoleTrace] = None,
        *,
        M: Optional[int] = None,
    ):
        if committee_trace is None:
            if M is None:
                # Compatibility default for callers that construct the oracle
                # before DagTOB. ``get_wave_leader`` replaces it with dag.M.
                self._deferred_all_members = True
                committee_trace = AllMembersCommitteeTrace(1)
            else:
                self._deferred_all_members = False
                committee_trace = AllMembersCommitteeTrace(M)
        else:
            self._deferred_all_members = False
        super().__init__(DeterministicIdealCoin(epoch), committee_trace)

    def selected_vid(self, wave: int) -> VertexId:
        if self._deferred_all_members:
            raise ValueError(
                "selected_vid requires M or an explicit CommitteeRoleTrace"
            )
        return super().selected_vid(wave)

    def get_wave_leader(self, dag_tob: "DagTOB", wave: int) -> Optional[VertexId]:
        if self._deferred_all_members:
            # Local compatibility harnesses model the committee with the
            # quorum_h genesis roles. Production runners pass an explicit
            # VRF-generated CommitteeRoleTrace instead.
            roles = AllMembersCommitteeTrace(dag_tob.quorum_h).roles(wave)
            leader_index = self.coin.get(wave) % len(roles)
            vid = (4 * (int(wave) - 1) + 1, roles[leader_index])
            return vid if vid in dag_tob.DAG.get(vid[0], {}) else None
        return super().get_wave_leader(dag_tob, wave)


class DagTOB:
    """Local ATCaaS DAG-TOB state machine."""

    def __init__(
        self,
        my_id: int,
        M: int,
        quorum_h: int,
        *,
        agreement_t: int,
        epoch: str = "epoch0",
        wait_predicate: Optional[WaitPredicate] = None,
        leader_oracle: Optional[LeaderOracle] = None,
        deliver_cb=None,
        vertex_admitted_cb=None,
        vertex_verifier=None,
        broadcast_hook=None,
        wave_coin_needed_cb=None,
        logger: Optional[logging.Logger] = None,
    ):
        if M <= 0:
            raise ValueError("M must be positive")
        if not 0 < quorum_h <= M:
            raise ValueError("quorum_h must satisfy 0 < quorum_h <= M")
        if isinstance(agreement_t, bool) or not isinstance(agreement_t, int):
            raise TypeError("agreement_t must be an integer")
        if agreement_t < 0:
            raise ValueError("agreement_t must be non-negative")
        if quorum_h + agreement_t > M:
            raise ValueError(
                "quorum_h + agreement_t must not exceed long-lived "
                "population M"
            )
        if 2 * agreement_t >= quorum_h:
            raise ValueError(
                "agreement threshold must satisfy 2 * agreement_t < quorum_h"
            )

        self.my_id = my_id
        self.M = M
        self.quorum_h = quorum_h
        self.agreement_t = agreement_t
        self.coverage_threshold = agreement_t + 1
        self.epoch = epoch
        # Optional dissemination seam (P-ref.2): when set, t_broadcast hands the
        # originating message to this hook to RBC it under mid.  Left None for
        # in-process use, where the harness replicates messages to all Pools.
        self.broadcast_hook = broadcast_hook

        self.DAG: Dict[int, Dict[VertexId, Vertex]] = {}
        self.buffer: Dict[VertexId, Vertex] = {}
        self.decided_wave = 0
        self.leaders_stack: List[VertexId] = []
        self.ordered_vertices: List[VertexId] = []
        self._ordered_vertices_set: Set[VertexId] = set()
        self._next_ordered_delivery_index = 0
        self.Pool: Dict[str, Set[PoolEntry]] = {}
        self.DeliveredPool: Dict[str, Set[PoolEntry]] = {}
        # Authoritative current-paper application output.  Each event is one
        # t_deliver(qid, S), annotated only with local observability metadata.
        self.delivery_events: List[DeliveryEvent] = []
        # Compatibility delivery transcript.  Each item is a singleton
        # ``{qid: entries}`` event; it has no height/protocol role and is never
        # passed to SelectBlock or J.  New callers must use delivery_events or
        # the delivery callback instead.
        self.ledger: List[Block] = []
        self.delivery_sequence_number = 0

        self.wait_predicate = wait_predicate or TypedWaitPredicate()
        self.leader_oracle = leader_oracle or DeterministicLeaderOracle(epoch)
        self.deliver_cb = deliver_cb
        self.vertex_admitted_cb = vertex_admitted_cb
        self.vertex_verifier = vertex_verifier
        self.wave_coin_needed_cb = wave_coin_needed_cb
        self.log = logger or logging.getLogger(f"dag_tob.{my_id}")
        self._wave_fired: Set[int] = set()
        self._wave_coin_requested: Set[int] = set()
        self._ready_rounds: Set[int] = set()

        # Causal coverage is derived only from vertices that passed every
        # admission gate and were installed in the local DAG.  Counts are
        # intentionally per round: witnesses from different rounds never add
        # up to the threshold.
        self.qid_reference_counts_by_round: Dict[int, Dict[str, int]] = {}
        self.qid_covered_at_round: Dict[str, int] = {}
        self._coverage_counted_vertices: Set[VertexId] = set()
        self._coverage_records: List[dict] = []
        self._selectblock_records: List[dict] = []

        self._init_genesis()

    def _init_genesis(self) -> None:
        self.DAG[0] = {}
        for source in range(self.quorum_h):
            v = Vertex(
                round=0,
                source=source,
                block=None,
                strong_edges=frozenset(),
                weak_edges=frozenset(),
            )
            self.DAG[0][v.vid] = v
        self._ready_rounds.add(0)

    # ------------------------------------------------------------------
    # Algorithm 2: message pool
    # ------------------------------------------------------------------
    def _pool_add(self, qid: str, mid: str, m: bytes) -> bool:
        """Store an RBC-delivered message in the active or delivered pool."""

        entry = (str(mid), bytes(m))
        for existing_mid, existing_payload in self._messages_for_qid(qid):
            if existing_mid != entry[0]:
                continue
            if existing_payload != entry[1]:
                self.log.warning(
                    "[POOL] reject conflicting payload qid=%s mid=%s",
                    qid,
                    mid,
                )
            return False

        delivered = bool(self.DeliveredPool.get(qid))
        target = self.DeliveredPool if delivered else self.Pool
        bucket = target.setdefault(qid, set())
        bucket.add(entry)
        self.log.debug(
            "[%s] added qid=%s mid=%s",
            "DELIVERED-POOL" if delivered else "POOL",
            qid,
            mid,
        )
        return True

    def _messages_for_qid(self, qid: str) -> Set[PoolEntry]:
        """Return all retained messages for qid across both local pools."""

        return set(self.Pool.get(qid, set())) | set(
            self.DeliveredPool.get(qid, set())
        )

    def t_broadcast(self, qid: str, mid: str, m: bytes) -> None:
        """Submit a message: add to local Pool and disseminate under mid.

        The actual RBC dissemination is delegated to ``broadcast_hook`` (set by
        the network runner); without a hook this only seeds the local Pool.
        """
        if not self._pool_add(qid, mid, m):
            return
        self._try_promote_buffer()
        if self.broadcast_hook is not None:
            self.broadcast_hook(qid, mid, m)

    def deliver_message(self, qid: str, mid: str, m: bytes) -> None:
        """Accept a message received via message-RBC into the proper local pool.

        Unlike ``t_broadcast`` this never re-disseminates (it is the receive
        side); a newly available message may unblock buffered vertices whose
        Available gate was previously unmet, so re-run promotion.
        """
        if self._pool_add(qid, mid, m):
            self._try_promote_buffer()

    # ------------------------------------------------------------------
    # Algorithm 3: vertex creation
    # ------------------------------------------------------------------
    def create_new_vertex(
        self,
        round_id: int,
        source: Optional[int] = None,
        proof: Optional[dict] = None,
    ) -> Vertex:
        """Create a new vertex for ``round_id`` from current local state."""
        if round_id <= 0:
            raise ValueError("round_id must be positive")
        source_id = self.my_id if source is None else source
        parent_round = round_id - 1
        strong_edges = frozenset(self.DAG.get(parent_round, {}).keys())
        weak_edges = self.set_weak_edges(round_id, strong_edges)

        candidate_qids = sorted(self.Pool)
        delivered_qids = set(self.DeliveredPool)
        covered_qids = {
            qid
            for qid, cover_round in self.qid_covered_at_round.items()
            if cover_round < round_id
        }
        delivered_exclusions = sorted(
            set(candidate_qids) & delivered_qids
        )
        causal_coverage_exclusions = sorted(
            (set(candidate_qids) & covered_qids) - delivered_qids
        )
        excluded_qids = delivered_qids | covered_qids
        evaluated_qids = sorted(set(candidate_qids) - excluded_qids)
        block = self.wait_predicate.select_block(self.Pool, excluded_qids)
        canonical_block_refs = canonical_ref_block(block)
        selected_qids = sorted(canonical_block_refs or {})
        self._selectblock_records.append(
            {
                "round": int(round_id),
                "source": int(source_id),
                "candidate_qids": candidate_qids,
                "delivered_exclusions": delivered_exclusions,
                "causal_coverage_exclusions": causal_coverage_exclusions,
                "evaluated_qids": evaluated_qids,
                "selected_qids": selected_qids,
            }
        )
        return Vertex(
            round=round_id,
            source=source_id,
            block=canonical_block_refs,
            strong_edges=strong_edges,
            weak_edges=weak_edges,
            proof=proof,
        )

    def _record_admitted_vertex_coverage(self, vertex: Vertex) -> None:
        """Count one witness per qid for an admitted vertex exactly once."""

        if vertex.vid in self._coverage_counted_vertices:
            return
        self._coverage_counted_vertices.add(vertex.vid)
        if not vertex.block:
            return

        round_counts = self.qid_reference_counts_by_round.setdefault(
            vertex.round, {}
        )
        for qid in sorted(vertex.block):
            witness_count = round_counts.get(qid, 0) + 1
            round_counts[qid] = witness_count
            previous_cover_round = self.qid_covered_at_round.get(qid)
            if witness_count >= self.coverage_threshold and (
                previous_cover_round is None
                or vertex.round < previous_cover_round
            ):
                self.qid_covered_at_round[qid] = vertex.round
                coverage_record = {
                    "qid": qid,
                    "cover_round": int(vertex.round),
                    "witness_count": int(witness_count),
                }
                self._coverage_records = [
                    record
                    for record in self._coverage_records
                    if record["qid"] != qid
                ]
                self._coverage_records.append(coverage_record)

    def snapshot_qid_causal_coverage(self) -> dict:
        """Return a deterministic, JSON-serializable coverage snapshot."""

        return {
            "schema": "dag-qid-causal-coverage/v1",
            "agreement_t": self.agreement_t,
            "coverage_threshold": self.coverage_threshold,
            "reference_counts_by_round": {
                str(round_id): {
                    qid: count
                    for qid, count in sorted(counts.items())
                }
                for round_id, counts in sorted(
                    self.qid_reference_counts_by_round.items()
                )
            },
            "covered_at_round": {
                qid: round_id
                for qid, round_id in sorted(
                    self.qid_covered_at_round.items()
                )
            },
            "coverage_records": [
                dict(record)
                for record in sorted(
                    self._coverage_records,
                    key=lambda item: (item["cover_round"], item["qid"]),
                )
            ],
        }

    def dag_selectblock_metrics_snapshot(self) -> dict:
        """Return structural SelectBlock decisions, excluding predicate CPU."""

        records = [
            {
                key: list(value) if isinstance(value, list) else value
                for key, value in record.items()
            }
            for record in self._selectblock_records
        ]
        return {
            "schema": "dag-selectblock-metrics/v1",
            "applicable": True,
            "agreement_t": self.agreement_t,
            "coverage_threshold": self.coverage_threshold,
            "created_vertices": len(records),
            "candidate_qids": sum(
                len(record["candidate_qids"]) for record in records
            ),
            "evaluated_qids": sum(
                len(record["evaluated_qids"]) for record in records
            ),
            "selected_qids": sum(
                len(record["selected_qids"]) for record in records
            ),
            "skipped_delivered_qids": sum(
                len(record["delivered_exclusions"]) for record in records
            ),
            "skipped_causally_covered_qids": sum(
                len(record["causal_coverage_exclusions"])
                for record in records
            ),
            "coverage_records": self.snapshot_qid_causal_coverage()[
                "coverage_records"
            ],
            "per_vertex_records": records,
        }

    def set_weak_edges(
        self,
        round_id: int,
        strong_edges: Iterable[VertexId],
    ) -> frozenset[VertexId]:
        """Return Algorithm 3 ``SetWeakEdges`` for a new round vertex.

        Starting from the new vertex's strong parents, scan older rounds from
        newest to oldest.  Add a direct weak edge to every vertex that is not
        already reachable through the strong parents or a weak edge selected
        earlier in the scan.  Genesis vertices are excluded, matching
        Algorithm 3's loop over rounds ``round_id - 2`` down to ``1``.
        """

        roots: Set[VertexId] = set(strong_edges)
        weak_edges: Set[VertexId] = set()
        for older_round in range(round_id - 2, 0, -1):
            for candidate in sorted(self.DAG.get(older_round, {})):
                if self._path_from_roots(roots, candidate):
                    continue
                weak_edges.add(candidate)
                roots.add(candidate)
        return frozenset(weak_edges)

    # ------------------------------------------------------------------
    # Algorithm 2: r_deliver and DAG admission
    # ------------------------------------------------------------------
    def on_r_deliver(
        self,
        v: Vertex,
        round_id: Optional[int] = None,
        source: Optional[int] = None,
    ) -> None:
        """Admit a delivered DAG vertex into buffer/DAG if well formed."""
        r = v.round if round_id is None else round_id
        src = v.source if source is None else source
        v = Vertex(
            round=r,
            source=src,
            block=canonical_ref_block(v.block),
            strong_edges=frozenset(v.strong_edges),
            weak_edges=frozenset(v.weak_edges),
            proof=v.proof,
        )

        if r <= 0:
            return
        if self.vertex_verifier is not None and not self.vertex_verifier(v):
            self.log.warning("[DELIVER] drop v(%d,%d): invalid vertex proof", r, src)
            return
        if len(v.strong_edges) < self.quorum_h:
            self.log.warning(
                "[DELIVER] drop v(%d,%d): only %d strong edges, need %d",
                r,
                src,
                len(v.strong_edges),
                self.quorum_h,
            )
            return
        if v.vid in self.DAG.get(r, {}) or v.vid in self.buffer:
            self.log.debug("[DELIVER] duplicate v(%d,%d) ignored", r, src)
            return

        self.buffer[v.vid] = v
        self.log.debug("[BUFFER] v(%d,%d) buffered", r, src)
        self._try_promote_buffer()

    def _try_promote_buffer(self) -> None:
        changed = True
        while changed:
            changed = False
            to_promote: List[Vertex] = []
            for v in list(self.buffer.values()):
                parent_dag = self.DAG.get(v.round - 1, {})
                older_vertices = {
                    vid
                    for older_round, vertices in self.DAG.items()
                    if older_round < v.round - 1
                    for vid in vertices
                }
                # Algorithm 2, line 17: strong parents must be in the previous
                # round, weak parents in earlier rounds, and every referenced
                # message must be locally available.
                if (
                    v.strong_edges.issubset(parent_dag.keys())
                    and v.weak_edges.issubset(older_vertices)
                    and self._available(v.block)
                ):
                    to_promote.append(v)

            for v in sorted(to_promote, key=lambda item: item.vid):
                self.buffer.pop(v.vid, None)
                self.DAG.setdefault(v.round, {})[v.vid] = v
                self._record_admitted_vertex_coverage(v)
                changed = True
                self.log.debug(
                    "[DAG] admitted v(%d,%d), |DAG[%d]|=%d",
                    v.round,
                    v.source,
                    v.round,
                    len(self.DAG[v.round]),
                )
                if self.vertex_admitted_cb is not None:
                    self.vertex_admitted_cb(v)
                self._mark_round_ready_if_needed(v.round)
                self._fire_wave_if_needed(v.round)

    def _mark_round_ready_if_needed(self, round_id: int) -> None:
        if len(self.DAG.get(round_id, {})) >= self.quorum_h:
            self._ready_rounds.add(round_id)

    def _fire_wave_if_needed(self, round_id: int) -> None:
        if round_id <= 0 or round_id % 4 != 0:
            return
        if len(self.DAG.get(round_id, {})) < self.quorum_h:
            return
        wave = round_id // 4
        if wave in self._wave_fired:
            return
        ready_check = getattr(self.leader_oracle, "is_wave_ready", None)
        coin_ready = True if ready_check is None else bool(ready_check(wave))
        if not coin_ready:
            if wave not in self._wave_coin_requested:
                self._wave_coin_requested.add(wave)
                self.log.info("[WAVE-COIN] requested wave=%d", wave)
                if self.wave_coin_needed_cb is None:
                    self.log.warning(
                        "[WAVE-COIN] wave=%d waits without a coordinator", wave
                    )
                    return
                try:
                    result = self.wave_coin_needed_cb(wave)
                except BaseException:
                    self._wave_coin_requested.discard(wave)
                    raise
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    self._wave_coin_requested.discard(wave)
                    raise TypeError(
                        "wave_coin_needed_cb must schedule async work and return None"
                    )
            return
        self._wave_fired.add(wave)
        self._wave_coin_requested.discard(wave)
        self.wave_ready(wave)

    def notify_wave_coin_ready(self, wave: int) -> bool:
        """Resume a quorum-ready wave after its asynchronous coin completes."""

        wave = int(wave)
        if wave <= 0:
            raise ValueError("wave must be positive")
        ready_check = getattr(self.leader_oracle, "is_wave_ready", None)
        coin_ready = True if ready_check is None else bool(ready_check(wave))
        if not coin_ready:
            raise RuntimeError(f"common coin for wave {wave} is not ready")
        self._fire_wave_if_needed(4 * wave)
        return wave in self._wave_fired

    def snapshot_wave_coin_state(self) -> dict:
        return {
            "requested_waves": sorted(self._wave_coin_requested),
            "fired_waves": sorted(self._wave_fired),
        }

    # ------------------------------------------------------------------
    # Algorithm 2: wave commit and ledger advancement
    # ------------------------------------------------------------------
    def wave_ready(self, wave: int) -> bool:
        leader_vid = self.leader_oracle.get_wave_leader(self, wave)
        if leader_vid is None:
            self.log.info("[WAVE] wave %d no leader", wave)
            return False

        round_4w = 4 * wave
        supporters = [
            u_vid
            for u_vid in self.DAG.get(round_4w, {})
            if self.strong_path(u_vid, leader_vid)
        ]
        if len(supporters) < self.quorum_h:
            self.log.info(
                "[WAVE] wave %d leader %s lacks support %d/%d",
                wave,
                leader_vid,
                len(supporters),
                self.quorum_h,
            )
            return False

        leaders_to_push = [leader_vid]
        cursor = leader_vid
        for prev_wave in range(wave - 1, self.decided_wave, -1):
            prev_leader = self.leader_oracle.get_wave_leader(self, prev_wave)
            if prev_leader is not None and self.strong_path(cursor, prev_leader):
                leaders_to_push.append(prev_leader)
                cursor = prev_leader

        # The current leader is pushed first, followed by earlier leaders in
        # decreasing wave order.  Popping the stack therefore processes the
        # oldest leader first, as required by Algorithm 2.
        for vid in leaders_to_push:
            self.leaders_stack.append(vid)

        self.decided_wave = wave
        self.advance_ledger()
        return True

    def advance_ledger(self) -> None:
        while self.leaders_stack:
            leader_vid = self.leaders_stack.pop()
            reachable: Set[VertexId] = set()
            self._collect_reachable(
                leader_vid,
                reachable,
                exclude=self._ordered_vertices_set,
            )
            for vid in sorted(reachable):
                if vid not in self._ordered_vertices_set:
                    self.ordered_vertices.append(vid)
                    self._ordered_vertices_set.add(vid)

            self._deliver_queries_from_ordered_vertices()

    def _deliver_queries_from_ordered_vertices(self) -> None:
        """Scan newly ordered vertices once and deliver each justified qid."""

        while self._next_ordered_delivery_index < len(self.ordered_vertices):
            vid = self.ordered_vertices[self._next_ordered_delivery_index]
            self._next_ordered_delivery_index += 1
            v = self._lookup_vertex(vid)
            if v is None or v.block is None:
                continue

            for qid in sorted(v.block):
                if qid in self.DeliveredPool:
                    continue
                resolved = self._resolve({qid: v.block[qid]})
                if resolved is None:
                    continue
                entries = resolved[qid]
                if not self.wait_predicate.verify_qid_set(qid, entries):
                    continue

                active_entries = self.Pool.pop(qid, set())
                delivered_entries = self.DeliveredPool.setdefault(qid, set())
                delivered_entries.update(active_entries)
                ordered_entries = tuple(sorted(entries, key=_stable_entry))
                delivery_event = DeliveryEvent(
                    sequence=len(self.delivery_events) + 1,
                    qid=qid,
                    entries=ordered_entries,
                    ordered_vertex=vid,
                    committed_wave=self.decided_wave,
                )
                self.delivery_events.append(delivery_event)
                self.delivery_sequence_number = delivery_event.sequence
                self.ledger.append({qid: list(ordered_entries)})
                self.log.info(
                    "[T_DELIVER] seq=%d qid=%s entries=%d from v(%d,%d) "
                    "committed_wave=%d",
                    self.delivery_sequence_number,
                    qid,
                    len(ordered_entries),
                    v.round,
                    v.source,
                    self.decided_wave,
                )
                if self.deliver_cb is not None:
                    self.deliver_cb(qid, list(ordered_entries))

    # ------------------------------------------------------------------
    # Algorithm 2: Resolve / Available (block-by-reference helpers)
    # ------------------------------------------------------------------
    def _resolve(self, block: Optional[RefBlock]) -> Optional[Block]:
        """Resolve every reference against Pool union DeliveredPool.

        Resolution is all-or-nothing.  A partially resolved block must not be
        passed to a justifier as if the absent references had never existed.
        """

        if not block:
            return None
        resolved: Block = {}
        for qid, mids in block.items():
            index = {mid: payload for mid, payload in self._messages_for_qid(qid)}
            if any(mid not in index for mid in mids):
                return None
            resolved[qid] = [(mid, index[mid]) for mid in mids]
        return canonical_block(resolved)

    def _available(self, block: Optional[RefBlock]) -> bool:
        """Algorithm 2 Available(B): every reference resolvable locally."""
        if not block:
            return True
        for qid, mids in block.items():
            have = {mid for mid, _ in self._messages_for_qid(qid)}
            for mid in mids:
                if mid not in have:
                    return False
        return True

    # ------------------------------------------------------------------
    # DAG traversal helpers
    # ------------------------------------------------------------------
    def strong_path(self, from_vid: VertexId, to_vid: VertexId) -> bool:
        """Return whether ``from_vid`` reaches ``to_vid`` using strong edges."""

        return self._path_with_edge_kinds(
            from_vid,
            to_vid,
            include_weak=False,
        )

    def path(self, from_vid: VertexId, to_vid: VertexId) -> bool:
        """Return whether ``from_vid`` reaches ``to_vid`` through either edge kind."""

        return self._path_with_edge_kinds(
            from_vid,
            to_vid,
            include_weak=True,
        )

    def _path_with_edge_kinds(
        self,
        from_vid: VertexId,
        to_vid: VertexId,
        *,
        include_weak: bool,
    ) -> bool:
        if from_vid == to_vid:
            return True
        visited: Set[VertexId] = set()
        stack = [from_vid]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            cur_v = self._lookup_vertex(cur)
            if cur_v is None:
                continue
            parents = set(cur_v.strong_edges)
            if include_weak:
                parents.update(cur_v.weak_edges)
            for parent in parents:
                if parent == to_vid:
                    return True
                stack.append(parent)
        return False

    def _path_from_roots(
        self,
        roots: Iterable[VertexId],
        to_vid: VertexId,
    ) -> bool:
        """Reachability from prospective vertex edges during SetWeakEdges."""

        return any(root == to_vid or self.path(root, to_vid) for root in roots)

    def _collect_reachable(
        self,
        root_vid: VertexId,
        out: Set[VertexId],
        exclude: Set[VertexId],
    ) -> None:
        stack = [root_vid]
        visited: Set[VertexId] = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            cur_v = self._lookup_vertex(cur)
            if cur_v is None:
                continue
            if cur[0] > 0 and cur not in exclude:
                out.add(cur)
            stack.extend(cur_v.strong_edges)
            stack.extend(cur_v.weak_edges)

    def _lookup_vertex(self, vid: VertexId) -> Optional[Vertex]:
        round_id, _ = vid
        return self.DAG.get(round_id, {}).get(vid)

    # ------------------------------------------------------------------
    # Test/support helpers
    # ------------------------------------------------------------------
    @property
    def delivery_count(self) -> int:
        return len(self.delivery_events)

    def snapshot_delivery_events(self) -> List[dict]:
        return [event.to_dict() for event in self.delivery_events]

    def is_round_ready(self, round_id: int) -> bool:
        return len(self.DAG.get(round_id, {})) >= self.quorum_h

    def snapshot_ledger(self) -> List[dict]:
        return [self._block_to_jsonable(block) for block in self.ledger]

    def snapshot_ordered_vertices(self) -> List[VertexId]:
        return list(self.ordered_vertices)

    @staticmethod
    def _block_to_jsonable(block: Block) -> dict:
        return {
            qid: [
                {"mid": mid, "payload": payload.hex()}
                for mid, payload in sorted(entries, key=_stable_entry)
            ]
            for qid, entries in sorted(block.items())
        }


def simulate_one_wave(
    *,
    M: int = 4,
    quorum_h: int = 3,
    agreement_t: int,
    epoch: str = "phase1-smoke",
    sources: Optional[Sequence[int]] = None,
) -> List[DagTOB]:
    """
    Phase-1 deterministic local simulation.

    It creates ``M`` local DagTOB instances, injects identical Pool messages,
    creates ``quorum_h`` vertices for rounds 1..4, and directly delivers each
    vertex to every simulated node.
    """
    selected_sources = list(sources) if sources is not None else list(range(quorum_h))
    if len(selected_sources) < quorum_h:
        raise ValueError("sources must contain at least quorum_h entries")

    nodes = [
        DagTOB(
            my_id=i,
            M=M,
            quorum_h=quorum_h,
            agreement_t=agreement_t,
            epoch=epoch,
            leader_oracle=DeterministicLeaderOracle(
                epoch,
                RecordedCommitteeRoleTrace({1: selected_sources[:quorum_h]}),
            ),
        )
        for i in range(M)
    ]

    # Block carries references only, so every referenced message must be
    # available in each node's Pool before the vertex can be promoted
    # (Algorithm 2, Available).  With no network yet, the harness disseminates
    # every producer's message to every node's Pool -- this stands in for the
    # per-mid message-RBC that P-ref.2 will add.
    for producer in range(M):
        for node in nodes:
            node.t_broadcast(
                "qid-demo",
                f"mid-{producer}",
                f"payload-{producer}".encode(),
            )

    for round_id in range(1, 5):
        vertices = [
            nodes[source].create_new_vertex(round_id, source=source)
            for source in selected_sources[:quorum_h]
        ]
        for node in nodes:
            for vertex in vertices:
                node.on_r_deliver(vertex)

    return nodes
