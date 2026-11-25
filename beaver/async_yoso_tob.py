import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from beaver.yosorbc import YosoRBC


def _stable_json(data) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True, order=True)
class MessageRef:
    sender: int
    counter: int

    def encode(self) -> str:
        return f"{self.sender}:{self.counter}"

    @classmethod
    def decode(cls, raw: str) -> "MessageRef":
        sender, counter = raw.split(":", 1)
        return cls(int(sender), int(counter))


@dataclass
class PendingEntry:
    ref: MessageRef
    mid: str
    payload: bytes
    received_at: float

    def to_summary(self) -> dict:
        return {
            "ref": self.ref.encode(),
            "mid": self.mid,
            "payload_hex": self.payload.hex(),
        }


@dataclass
class BroadcastEnvelope:
    sender: int
    counter: int
    mid: str
    payload: bytes

    @property
    def ref(self) -> MessageRef:
        return MessageRef(self.sender, self.counter)

    def to_bytes(self) -> bytes:
        return _stable_json(
            {
                "sender": self.sender,
                "counter": self.counter,
                "mid": self.mid,
                "payload": self.payload.hex(),
            }
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> "BroadcastEnvelope":
        data = json.loads(raw.decode("utf-8"))
        return cls(
            sender=int(data["sender"]),
            counter=int(data["counter"]),
            mid=str(data["mid"]),
            payload=bytes.fromhex(data["payload"]),
        )


def message_rbc_prefix(epoch: str, sender: int, counter: int) -> str:
    return f"AYTOB:{epoch}:MSG:{sender}:{counter}"


def message_rbc_echo_label(epoch: str, sender: int, counter: int) -> bytes:
    return f"AYTOB:RBC:{epoch}:MSG:{sender}:{counter}:ECHO".encode("utf-8")


def message_rbc_ready_label(epoch: str, sender: int, counter: int) -> bytes:
    return f"AYTOB:RBC:{epoch}:MSG:{sender}:{counter}:READY".encode("utf-8")


class AsyncYosoTOB:
    """Phase B0/B1 shell for the Async YOSO TOB baseline.

    This class currently implements the Fig.14 Broadcast/Schedule message
    layer.  Ledger delivery and ACS/SelectBlock are intentionally absent until
    Phase B2/B3, so ``ledger`` remains empty in B1 smoke tests.
    """

    def __init__(
        self,
        *,
        my_id: int,
        M: int,
        epoch: str,
        logger: Optional[logging.Logger] = None,
    ):
        self.my_id = my_id
        self.M = M
        self.epoch = epoch
        self.log = logger or logging.getLogger(f"async_yoso_tob.{my_id}")

        self.ledger: List[List[bytes]] = []
        self.ledger_refs: List[List[MessageRef]] = []
        self.pending: Dict[MessageRef, PendingEntry] = {}
        self.delivered_refs = set()
        self.broadcast_counter = 0
        self.message_manager: Optional["AsyncYosoMessageRBCManager"] = None

    def set_message_manager(self, manager: "AsyncYosoMessageRBCManager") -> None:
        self.message_manager = manager

    async def t_broadcast(self, mid: str, message: bytes) -> MessageRef:
        if self.message_manager is None:
            raise RuntimeError("message RBC manager is not attached")
        counter = self.broadcast_counter
        self.broadcast_counter += 1
        ref = MessageRef(self.my_id, counter)
        await self.message_manager.broadcast_message(counter, mid, message)
        return ref

    def schedule_message(self, envelope: BroadcastEnvelope) -> bool:
        ref = envelope.ref
        if ref in self.pending:
            return False
        self.pending[ref] = PendingEntry(
            ref=ref,
            mid=envelope.mid,
            payload=envelope.payload,
            received_at=time.time(),
        )
        self.log.info(
            "[SCHEDULE] ref=%s mid=%s pending=%d",
            ref.encode(),
            envelope.mid,
            len(self.pending),
        )
        return True

    def snapshot_pending(self) -> List[dict]:
        return [
            self.pending[ref].to_summary()
            for ref in sorted(self.pending)
        ]

    def pending_refs(self) -> List[MessageRef]:
        return [
            ref for ref in sorted(self.pending)
            if ref not in self.delivered_refs
        ]

    def missing_refs(self, refs: List[MessageRef]) -> List[MessageRef]:
        return [ref for ref in refs if ref not in self.pending]

    def append_acs_block(self, refs: List[MessageRef]) -> Tuple[int, List[bytes]]:
        missing = self.missing_refs(refs)
        if missing:
            raise RuntimeError(
                "cannot append ACS block with missing refs: "
                + ",".join(ref.encode() for ref in missing)
            )

        block_refs: List[MessageRef] = []
        block_payloads: List[bytes] = []
        for ref in refs:
            if ref in self.delivered_refs:
                continue
            entry = self.pending[ref]
            block_refs.append(ref)
            block_payloads.append(entry.payload)

        if not block_payloads:
            raise RuntimeError("ACS block contains no new messages")

        self.ledger_refs.append(block_refs)
        self.ledger.append(block_payloads)
        self.delivered_refs.update(block_refs)
        for ref in block_refs:
            self.pending.pop(ref, None)
        ledger_index = len(self.ledger)
        self.log.info(
            "[T_DELIVER] ledger_index=%d refs=%s block_size=%d",
            ledger_index,
            [ref.encode() for ref in block_refs],
            len(block_payloads),
        )
        return ledger_index, block_payloads

    def snapshot_ledger(self) -> List[List[str]]:
        return [
            [payload.hex() for payload in block]
            for block in self.ledger
        ]

    def snapshot_ledger_refs(self) -> List[List[str]]:
        return [
            [ref.encode() for ref in block_refs]
            for block_refs in self.ledger_refs
        ]


@dataclass
class _MessageRBCInstance:
    sender: int
    counter: int
    rbc: YosoRBC
    reactor_task: asyncio.Task
    output_task: asyncio.Task
    sender_task: Optional[asyncio.Task] = None
    delivered: bool = False
    stopped: bool = False


class AsyncYosoMessageRBCManager:
    """Manage per-message YosoRBC sessions for Async YOSO TOB B1."""

    def __init__(
        self,
        *,
        tob: AsyncYosoTOB,
        epoch: str,
        my_id: int,
        M: int,
        t: int,
        send,
        shared_subscribe,
        crypto,
        vrf_sk,
        vrf_pks,
        threshold_fraction: Optional[float] = None,
        expected_committee_size: Optional[int] = None,
        committee_m: Optional[int] = None,
        quorum_h: Optional[int] = None,
        ready_amplify_threshold: Optional[int] = None,
        require_vrf_proofs: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.tob = tob
        self.epoch = epoch
        self.my_id = my_id
        self.M = M
        self.t = t
        self.send = send
        self.shared_subscribe = shared_subscribe
        self.crypto = crypto
        self.vrf_sk = vrf_sk
        self.vrf_pks = vrf_pks
        self.log = logger or logging.getLogger(f"async_yoso_msg_rbc.{my_id}")

        self.expected_committee_size = int(
            M
            if expected_committee_size is None
            else expected_committee_size
        )
        self.committee_m = int(
            self.expected_committee_size
            if committee_m is None
            else committee_m
        )
        self.quorum_h = int(M - t if quorum_h is None else quorum_h)
        self.threshold_fraction = float(
            self.expected_committee_size / M
            if threshold_fraction is None
            else threshold_fraction
        )
        self.ready_amplify_threshold = int(
            (self.quorum_h + 1) // 2 + 1
            if ready_amplify_threshold is None
            else ready_amplify_threshold
        )
        if not 0.0 < self.threshold_fraction <= 1.0:
            raise ValueError("threshold_fraction must satisfy 0 < value <= 1")
        if not 1 <= self.expected_committee_size <= M:
            raise ValueError(
                "expected_committee_size must satisfy 1 <= value <= M"
            )
        if not 1 <= self.committee_m <= M:
            raise ValueError("committee_m must satisfy 1 <= value <= M")
        if not 1 <= self.quorum_h <= self.committee_m:
            raise ValueError(
                "quorum_h must satisfy 1 <= value <= committee_m"
            )
        if self.ready_amplify_threshold < 1:
            raise ValueError(
                "ready_amplify_threshold must be positive"
            )
        self.require_vrf_proofs = bool(require_vrf_proofs)

        self.instances: Dict[Tuple[int, int], _MessageRBCInstance] = {}
        self.delivered_messages = set()
        self._stopping = False

    def parameters_snapshot(self) -> dict:
        return {
            "threshold_fraction": self.threshold_fraction,
            "expected_committee_size": self.expected_committee_size,
            "committee_m": self.committee_m,
            "quorum_h": self.quorum_h,
            "ready_amplify_threshold": self.ready_amplify_threshold,
            "require_vrf_proofs": self.require_vrf_proofs,
        }

    def ensure_instance(self, sender: int, counter: int) -> None:
        key = (sender, counter)
        if key in self.instances:
            return

        pbk, pvk, pks, sk, srs = self.crypto
        rbc = YosoRBC(
            pks,
            sk,
            pbk,
            pvk,
            self.expected_committee_size,
            self.t,
            srs,
            self.my_id,
            self.send,
            None,
            prefix=message_rbc_prefix(self.epoch, sender, counter),
            shared_subscribe=self.shared_subscribe,
            threshold_fraction=self.threshold_fraction,
            network_size=self.M,
            vrf_sk=self.vrf_sk,
            vrf_pks=self.vrf_pks,
            expected_committee_size=self.expected_committee_size,
            committee_m=self.committee_m,
            quorum_h=self.quorum_h,
            ready_amplify_threshold=self.ready_amplify_threshold,
            vrf_echo_label=message_rbc_echo_label(self.epoch, sender, counter),
            vrf_ready_label=message_rbc_ready_label(self.epoch, sender, counter),
            require_vrf_proofs=self.require_vrf_proofs,
        )
        reactor_task = asyncio.create_task(rbc.reactor())
        output_task = asyncio.create_task(self._watch_output(sender, counter, rbc))
        self.instances[key] = _MessageRBCInstance(
            sender=sender,
            counter=counter,
            rbc=rbc,
            reactor_task=reactor_task,
            output_task=output_task,
        )

    def ensure_counter_window(self, max_counter_exclusive: int) -> None:
        created_before = len(self.instances)
        for sender in range(self.M):
            for counter in range(max_counter_exclusive):
                self.ensure_instance(sender, counter)
        if len(self.instances) != created_before:
            self.log.info(
                "[MSG-RBC] ensured counters=0..%d active=%d",
                max_counter_exclusive - 1,
                self.active_count(),
            )

    async def broadcast_message(self, counter: int, mid: str, payload: bytes) -> None:
        self.ensure_instance(self.my_id, counter)
        inst = self.instances[(self.my_id, counter)]
        if inst.sender_task is not None:
            return
        envelope = BroadcastEnvelope(
            sender=self.my_id,
            counter=counter,
            mid=mid,
            payload=payload,
        )

        async def _send():
            await asyncio.sleep(0.1)
            await inst.rbc.sender(envelope.to_bytes())

        inst.sender_task = asyncio.create_task(_send())
        self.log.info(
            "[MSG-RBC] sender started ref=%s mid=%s bytes=%d",
            envelope.ref.encode(),
            mid,
            len(payload),
        )

    async def _watch_output(self, sender: int, counter: int, rbc: YosoRBC) -> None:
        try:
            payload, _ = await rbc.output_queue.get()
            envelope = BroadcastEnvelope.from_bytes(payload)
            if envelope.sender != sender or envelope.counter != counter:
                self.log.warning(
                    "[MSG-RBC] prefix %d:%d delivered %s, ignored",
                    sender,
                    counter,
                    envelope.ref.encode(),
                )
                return
            self.delivered_messages.add((sender, counter))
            self.tob.schedule_message(envelope)
            self.log.info(
                "[MSG-DELIVER] ref=%s mid=%s delivered=%d pending=%d",
                envelope.ref.encode(),
                envelope.mid,
                len(self.delivered_messages),
                len(self.tob.pending),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.info(
                "[MSG-RBC] output watcher %d:%d ended: %s",
                sender,
                counter,
                exc,
            )
        finally:
            inst = self.instances.get((sender, counter))
            if inst is not None:
                inst.delivered = (sender, counter) in self.delivered_messages

    async def _stop_instance(self, key: Tuple[int, int]) -> None:
        inst = self.instances.pop(key, None)
        if inst is None or inst.stopped:
            return
        inst.stopped = True
        for task in (inst.sender_task, inst.output_task, inst.reactor_task):
            if task is not None and not task.done():
                task.cancel()
        tasks = [
            task for task in (inst.sender_task, inst.output_task, inst.reactor_task)
            if task is not None
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self, timeout: Optional[float] = None) -> bool:
        if self._stopping:
            return True
        self._stopping = True
        keys = list(self.instances)
        stop_all = asyncio.gather(
            *(self._stop_instance(key) for key in keys),
            return_exceptions=True,
        )
        if timeout is None:
            await stop_all
            return True
        try:
            await asyncio.wait_for(stop_all, timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def active_count(self) -> int:
        return len(self.instances)
