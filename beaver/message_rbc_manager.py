"""Per-message YOSO-RBC dissemination for DAG-TOB block-by-reference.

In the reference-based TOB (ATCaaS §4.2), a block carries references
``(qid, mid)`` only; the message ``m`` itself is disseminated separately via
YOSO-RBC keyed by ``mid`` (paper: "each submitted message is disseminated
through YOSO-RBC under the identifier mid").  This manager runs one RBC
instance per ``(qid, sender)``: the producer node runs the sender, every node
runs a receiver that writes the delivered message into its local active or
delivered pool via ``DagTOB.deliver_message``, which re-triggers buffer
promotion (Available).

It mirrors :class:`beaver.dag_rbc_manager.DagRBCManager`; the RBC protocol
(:class:`beaver.yosorbc.YosoRBC`) is reused unchanged.
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from beaver.dag_tob import DagTOB
from beaver.yosorbc import YosoRBC


def _qid_hash(qid: str) -> str:
    return hashlib.sha1(qid.encode("utf-8")).hexdigest()[:12]


def message_rbc_prefix(epoch: str, qid: str, sender: int) -> str:
    return f"MSGRBC:{epoch}:{_qid_hash(qid)}:{sender}"


def message_rbc_echo_label(epoch: str, qid: str, sender: int) -> bytes:
    return f"ATCAAS:RBC:{epoch}:MSG:{_qid_hash(qid)}:{sender}:ECHO".encode("utf-8")


def message_rbc_ready_label(epoch: str, qid: str, sender: int) -> bytes:
    return f"ATCAAS:RBC:{epoch}:MSG:{_qid_hash(qid)}:{sender}:READY".encode("utf-8")


def encode_message(mid: str, m: bytes) -> bytes:
    return json.dumps(
        {"mid": str(mid), "payload": m.hex()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def decode_message(raw: bytes) -> Tuple[str, bytes]:
    data = json.loads(raw.decode("utf-8"))
    return str(data["mid"]), bytes.fromhex(data["payload"])


@dataclass
class _MsgRBCInstance:
    qid: str
    sender: int
    rbc: YosoRBC
    reactor_task: asyncio.Task
    output_task: asyncio.Task
    sender_task: Optional[asyncio.Task] = None
    delivered: bool = False
    stopped: bool = False


class MessageRBCManager:
    """Background manager for per-(qid, sender) message YosoRBC instances."""

    def __init__(
        self,
        *,
        dag: DagTOB,
        epoch: str,
        my_id: int,
        M: int,
        t: int,
        threshold_fraction: float,
        expected_committee_size: int,
        committee_m: int,
        quorum_h: int,
        send,
        shared_subscribe,
        crypto,
        vrf_sk,
        vrf_pks,
        logger: Optional[logging.Logger] = None,
    ):
        self.dag = dag
        self.epoch = epoch
        self.my_id = my_id
        self.M = M
        self.t = t
        self.threshold_fraction = threshold_fraction
        self.expected_committee_size = expected_committee_size
        self.committee_m = committee_m
        self.quorum_h = quorum_h
        self.send = send
        self.shared_subscribe = shared_subscribe
        self.crypto = crypto
        self.vrf_sk = vrf_sk
        self.vrf_pks = vrf_pks
        self.log = logger or logging.getLogger(f"message_rbc_manager.{my_id}")

        self.instances: Dict[Tuple[str, int], _MsgRBCInstance] = {}
        self.delivered_messages = set()
        self._stopping = False

    def ensure_instance(self, qid: str, sender: int) -> None:
        key = (qid, sender)
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
            prefix=message_rbc_prefix(self.epoch, qid, sender),
            shared_subscribe=self.shared_subscribe,
            threshold_fraction=self.threshold_fraction,
            network_size=self.M,
            vrf_sk=self.vrf_sk,
            vrf_pks=self.vrf_pks,
            expected_committee_size=self.expected_committee_size,
            committee_m=self.committee_m,
            quorum_h=self.quorum_h,
            ready_amplify_threshold=(self.quorum_h + 1) // 2 + 1,
            vrf_echo_label=message_rbc_echo_label(self.epoch, qid, sender),
            vrf_ready_label=message_rbc_ready_label(self.epoch, qid, sender),
        )
        reactor_task = asyncio.create_task(rbc.reactor())
        output_task = asyncio.create_task(self._watch_output(qid, sender, rbc))
        self.instances[key] = _MsgRBCInstance(
            qid=qid,
            sender=sender,
            rbc=rbc,
            reactor_task=reactor_task,
            output_task=output_task,
        )

    def ensure_window(self, qid: str, senders: Iterable[int]) -> None:
        created_before = len(self.instances)
        for sender in senders:
            self.ensure_instance(qid, int(sender))
        if len(self.instances) != created_before:
            self.log.info(
                "[MSG-RBC] ensure_window qid=%s active=%d",
                qid,
                self.active_count(),
            )

    async def broadcast_message(self, qid: str, mid: str, m: bytes) -> None:
        """Disseminate a locally originated message; sender = this node."""
        self.ensure_instance(qid, self.my_id)
        inst = self.instances[(qid, self.my_id)]
        if inst.sender_task is not None:
            return

        async def _send():
            await asyncio.sleep(0.1)
            await inst.rbc.sender(encode_message(mid, m))

        inst.sender_task = asyncio.create_task(_send())
        self.log.info("[MSG-RBC] sender started qid=%s mid=%s", qid, mid)

    async def _watch_output(self, qid: str, sender: int, rbc: YosoRBC) -> None:
        try:
            payload, _ = await rbc.output_queue.get()
            mid, m = decode_message(payload)
            self.delivered_messages.add((qid, sender))
            self.dag.deliver_message(qid, mid, m)
            self.log.info(
                "[MSG-DELIVER] qid=%s sender=%d mid=%s active=%d delivered=%d buffer=%d",
                qid,
                sender,
                mid,
                len(self.dag.Pool.get(qid, set())),
                len(self.dag.DeliveredPool.get(qid, set())),
                len(self.dag.buffer),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.info(
                "[MSG-RBC] output watcher qid=%s sender=%d ended: %s",
                qid,
                sender,
                exc,
            )
        finally:
            inst = self.instances.get((qid, sender))
            if inst is not None:
                inst.delivered = (qid, sender) in self.delivered_messages

    async def gc_delivered(self, delivered_qids) -> None:
        """Retain receivers for delivered qids so late messages can arrive.

        R2 deliberately performs no per-qid message GC: stopping an
        undelivered sender instance here would make a late message unavailable
        to vertices that already reference it.  ``stop`` remains the explicit
        whole-manager shutdown path.
        """

        retained = sum(1 for qid, _ in self.instances if qid in delivered_qids)
        if retained:
            self.log.debug(
                "[MSG-RBC] retain closed-qid receivers=%d for DeliveredPool",
                retained,
            )

    async def _stop_instance(self, key: Tuple[str, int]) -> None:
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
            self.log.warning(
                "[MSG-RBC] stop timed out after %.3fs; active=%d",
                timeout,
                self.active_count(),
            )
            return False

    def active_count(self) -> int:
        return len(self.instances)
