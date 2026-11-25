import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from beaver.dag_tob import DagTOB, Vertex
from beaver.yosorbc import YosoRBC


def vertex_rbc_prefix(epoch: str, round_id: int, source: int) -> str:
    return f"DAGTOB:{epoch}:VERTEX_RBC:{round_id}:{source}"


def vertex_rbc_echo_label(epoch: str, round_id: int, source: int) -> bytes:
    return f"ATCAAS:RBC:{epoch}:DAGTOB_VERTEX:{round_id}:{source}:ECHO".encode(
        "utf-8"
    )


def vertex_rbc_ready_label(epoch: str, round_id: int, source: int) -> bytes:
    return f"ATCAAS:RBC:{epoch}:DAGTOB_VERTEX:{round_id}:{source}:READY".encode(
        "utf-8"
    )


@dataclass
class _RBCInstance:
    round_id: int
    source: int
    rbc: YosoRBC
    reactor_task: asyncio.Task
    output_task: asyncio.Task
    sender_task: Optional[asyncio.Task] = None
    delivered: bool = False
    stopped: bool = False


class DagRBCManager:
    """Background manager for per-vertex YosoRBC instances.

    The manager keeps receive reactors alive across round transitions.  The
    runner only triggers RoleAction sends; RBC output is delivered to DagTOB
    asynchronously.
    """

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
        self.log = logger or logging.getLogger(f"dag_rbc_manager.{my_id}")

        self.instances: Dict[Tuple[int, int], _RBCInstance] = {}
        self.delivered_vertices = set()
        self.accepted_vertices = set()
        self._stopping = False

    def ensure_instance(self, round_id: int, source: int) -> None:
        if round_id <= 0:
            return
        key = (round_id, source)
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
            prefix=vertex_rbc_prefix(self.epoch, round_id, source),
            shared_subscribe=self.shared_subscribe,
            threshold_fraction=self.threshold_fraction,
            network_size=self.M,
            vrf_sk=self.vrf_sk,
            vrf_pks=self.vrf_pks,
            expected_committee_size=self.expected_committee_size,
            committee_m=self.committee_m,
            quorum_h=self.quorum_h,
            ready_amplify_threshold=(self.quorum_h + 1) // 2 + 1,
            vrf_echo_label=vertex_rbc_echo_label(self.epoch, round_id, source),
            vrf_ready_label=vertex_rbc_ready_label(self.epoch, round_id, source),
        )
        reactor_task = asyncio.create_task(rbc.reactor())
        output_task = asyncio.create_task(self._watch_output(round_id, source, rbc))
        self.instances[key] = _RBCInstance(
            round_id=round_id,
            source=source,
            rbc=rbc,
            reactor_task=reactor_task,
            output_task=output_task,
        )

    def ensure_window(self, base_round: int, lookahead_rounds: int) -> None:
        start = max(1, base_round + 1)
        end = max(start, base_round + lookahead_rounds)
        created_before = len(self.instances)
        for round_id in range(start, end + 1):
            for source in range(self.M):
                self.ensure_instance(round_id, source)
        if len(self.instances) != created_before:
            self.log.info(
                "[RBC-MANAGER] ensure_window base=%d rounds=%d..%d active=%d",
                base_round,
                start,
                end,
                self.active_count(),
            )

    async def broadcast_vertex(self, vertex: Vertex) -> None:
        key = vertex.vid
        self.ensure_instance(vertex.round, vertex.source)
        inst = self.instances[key]
        if inst.sender_task is not None:
            return

        async def _send():
            await asyncio.sleep(0.1)
            await inst.rbc.sender(vertex.to_bytes())

        inst.sender_task = asyncio.create_task(_send())
        self.log.info(
            "[RBC-MANAGER] sender started v(%d,%d) "
            "strong_edges=%d weak_edges=%d block=%s",
            vertex.round,
            vertex.source,
            len(vertex.strong_edges),
            len(vertex.weak_edges),
            "yes" if vertex.block is not None else "bottom",
        )

    async def _watch_output(self, round_id: int, source: int, rbc: YosoRBC) -> None:
        try:
            payload, _ = await rbc.output_queue.get()
            vertex = Vertex.from_bytes(payload)
            if vertex.round != round_id or vertex.source != source:
                self.log.warning(
                    "[RBC-MANAGER] prefix v(%d,%d) delivered v(%d,%d), ignored",
                    round_id,
                    source,
                    vertex.round,
                    vertex.source,
                )
                return
            self.delivered_vertices.add((round_id, source))
            self.dag.on_r_deliver(vertex, round_id=round_id, source=source)
            if vertex.vid in self.dag.DAG.get(round_id, {}):
                self.accepted_vertices.add(vertex.vid)
                self.log.info(
                    "[R_DELIVER] accepted v(%d,%d); |DAG[%d]|=%d buffer=%d deliveries=%d",
                    round_id,
                    source,
                    round_id,
                    len(self.dag.DAG.get(round_id, {})),
                    len(self.dag.buffer),
                    self.dag.delivery_count,
                )
            else:
                self.log.info(
                    "[R_DELIVER] buffered/dropped v(%d,%d); |DAG[%d]|=%d buffer=%d",
                    round_id,
                    source,
                    round_id,
                    len(self.dag.DAG.get(round_id, {})),
                    len(self.dag.buffer),
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.info(
                "[RBC-MANAGER] output watcher v(%d,%d) ended: %s",
                round_id,
                source,
                exc,
            )
        finally:
            inst = self.instances.get((round_id, source))
            if inst is not None:
                inst.delivered = (round_id, source) in self.delivered_vertices

    async def gc(self, current_round: int, retention_rounds: int) -> None:
        cutoff = current_round - retention_rounds
        if cutoff <= 0:
            return
        to_stop = [
            key for key, inst in self.instances.items()
            if inst.round_id < cutoff
        ]
        if not to_stop:
            return
        await asyncio.gather(
            *(self._stop_instance(key) for key in to_stop),
            return_exceptions=True,
        )
        self.log.info(
            "[RBC-MANAGER] gc cutoff=%d stopped=%d active=%d",
            cutoff,
            len(to_stop),
            self.active_count(),
        )

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
            self.log.warning(
                "[RBC-MANAGER] stop timed out after %.3fs; active=%d",
                timeout,
                self.active_count(),
            )
            return False

    def active_count(self) -> int:
        return len(self.instances)
