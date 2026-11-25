import asyncio
import hashlib
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from beaver.async_yoso_coin import (
    CoinOracle,
    DeterministicCoinOracle,
    SessionCoinIndexAllocator,
)
from beaver.async_yoso_tob import MessageRef
from beaver.protocol_trace import AllMembersCommitteeTrace, CommitteeRoleTrace
from beaver.vrf import (
    BERNOULLI_COMMITTEE_MODE,
    FIXED_SIZE_COMMITTEE_MODE,
    fixed_size_committee,
    generator_hex_from_srs,
    normalize_committee_selection_mode,
    vrf_backend_available,
    vrf_prove_sortition,
    vrf_verify_sortition,
)
from beaver.yosorbc import YosoRBC


def _canonical_json(data) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


_LEADER_TICKET_DOMAIN = b"ASYNC-YOSO:WEAK-GRADED-SELECT-BLOCK:LEADER:v1\x00"


def _candidate_leader(
    U_i: Dict[int, "CoreBlock"],
    coin: bytes,
    *,
    population_size: int,
) -> int:
    """Choose Fig. 18's minimum ``H(P_j || coin)`` candidate in ``U_i``."""

    if not isinstance(coin, bytes) or not coin:
        raise ValueError("common coin must be non-empty bytes")
    candidates = sorted(U_i)
    if not candidates:
        raise ValueError("leader selection requires a non-empty candidate set")
    if any(
        isinstance(candidate, bool)
        or not isinstance(candidate, int)
        or not 0 <= candidate < population_size
        for candidate in candidates
    ):
        raise ValueError("leader candidate outside the protocol population")

    def ticket(candidate: int) -> Tuple[bytes, int]:
        encoded_party = candidate.to_bytes(8, byteorder="big", signed=False)
        return (
            hashlib.sha256(
                _LEADER_TICKET_DOMAIN + encoded_party + coin
            ).digest(),
            candidate,
        )

    return min(candidates, key=ticket)


@dataclass(frozen=True, order=True)
class ProposedBlock:
    proposer: int
    refs: Tuple[MessageRef, ...]

    @classmethod
    def build(cls, proposer: int, refs: Iterable[MessageRef]) -> "ProposedBlock":
        return cls(proposer=proposer, refs=tuple(sorted(refs)))

    def to_obj(self) -> dict:
        return {
            "proposer": self.proposer,
            "refs": [ref.encode() for ref in self.refs],
        }

    @classmethod
    def from_obj(cls, data: dict) -> "ProposedBlock":
        return cls.build(
            int(data["proposer"]),
            [MessageRef.decode(raw) for raw in data.get("refs", [])],
        )


@dataclass(frozen=True, order=True)
class CoreBlock:
    """A SelectBlock block value for the ACS baseline.

    In ΠACS, parties first collect proposer blocks into a core set U_i.  Then
    ΠSelectBlock treats that U_i as the block value.  The TOB layer later
    flattens the selected core into message references.
    """

    proposals: Tuple[ProposedBlock, ...]

    @classmethod
    def build(cls, proposals: Iterable[ProposedBlock]) -> "CoreBlock":
        dedup: Dict[int, ProposedBlock] = {}
        for proposal in proposals:
            dedup[proposal.proposer] = proposal
        return cls(proposals=tuple(sorted(dedup.values())))

    def to_obj(self) -> dict:
        return {"proposals": [proposal.to_obj() for proposal in self.proposals]}

    @classmethod
    def from_obj(cls, data: dict) -> "CoreBlock":
        return cls.build(
            ProposedBlock.from_obj(item)
            for item in data.get("proposals", [])
        )

    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_obj())).hexdigest()

    def flatten_refs(self) -> List[MessageRef]:
        refs: List[MessageRef] = []
        seen = set()
        for proposal in self.proposals:
            for ref in proposal.refs:
                if ref in seen:
                    continue
                seen.add(ref)
                refs.append(ref)
        return refs


@dataclass
class ACSOutput:
    node: int
    selected_core: CoreBlock
    block_refs: List[MessageRef]
    output_round: int
    selection_attempts: int
    coin_calls: int
    trace: List[dict] = field(default_factory=list)

    def to_summary(self) -> dict:
        return {
            "node": self.node,
            "selected_core_digest": self.selected_core.digest(),
            "block_refs": [ref.encode() for ref in self.block_refs],
            "output_round": self.output_round,
            "selection_attempts": self.selection_attempts,
            "coin_calls": self.coin_calls,
            "trace": self.trace,
        }


@dataclass(frozen=True)
class ProposalAdmission:
    """External-validity result for one resolved ACS proposal."""

    accepted: bool
    qid_count: int = 0
    transcript_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a bool")
        for name, value in (
            ("qid_count", self.qid_count),
            ("transcript_count", self.transcript_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


class EchoRoundDependencyUnavailable(LookupError):
    """A propagated grade-2 value awaits this node's same-round output."""


class CommitteeUndersamplingError(TimeoutError):
    """A VRF stage could not collect its configured committee quorum."""

    def __init__(
        self,
        *,
        stage: str,
        need: int,
        verified_selected: int,
        collected: int,
    ) -> None:
        self.stage = stage
        self.need = int(need)
        self.verified_selected = int(verified_selected)
        self.collected = int(collected)
        super().__init__(
            f"committee undersampling at {stage}: "
            f"verified_selected={verified_selected} "
            f"collected={collected} need={need}"
        )


@dataclass
class _NodeState:
    node: int
    input_block: CoreBlock
    current_block: CoreBlock
    grade: int = 0
    gave_output: bool = False
    final_block: Optional[CoreBlock] = None
    final_round: Optional[int] = None
    selection_attempts: int = 0
    coin_calls: int = 0
    trace: List[dict] = field(default_factory=list)


class LocalAsyncYosoACS:
    """Single-process all-parties simulation of Appendix C ΠACS/SelectBlock.

    This is Phase B2A.  It keeps the protocol stages explicit and deterministic
    while replacing causal-cast/RBC delivery with local all-parties delivery.
    B2B should preserve these stage boundaries and swap local delivery for
    ``YosoRBC``.
    """

    def __init__(
        self,
        *,
        M: int,
        t: int,
        epoch: str,
        coin_oracle: Optional[CoinOracle] = None,
        committee_trace: Optional[CommitteeRoleTrace] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if M <= 0:
            raise ValueError("M must be positive")
        if t < 0 or t >= M:
            raise ValueError("t must satisfy 0 <= t < M")
        self.M = M
        self.t = t
        self.quorum = M - t
        self.epoch = epoch
        self.coin_oracle = coin_oracle or DeterministicCoinOracle(epoch)
        self.coin_indices = SessionCoinIndexAllocator()
        self.committee_trace = committee_trace or AllMembersCommitteeTrace(M)
        self.log = logger or logging.getLogger("local_async_yoso_acs")

    async def run(
        self,
        candidate_refs_by_node: Dict[int, Iterable[MessageRef]],
        *,
        session_id: str = "acs-b2a",
        max_rounds: int = 10,
        proposal_validator=None,
        proposal_fixed_cb=None,
    ) -> List[ACSOutput]:
        self.coin_indices.reserve(session_id, max_rounds)
        proposed = await self._admit_proposals(
            candidate_refs_by_node,
            proposal_validator,
        )
        states = self._init_acs_inputs(proposed)
        if proposal_fixed_cb is not None:
            result = proposal_fixed_cb(
                {
                    "session_id": session_id,
                    "collected": len(proposed),
                    "core_digest": states[0].current_block.digest(),
                }
            )
            if inspect.isawaitable(result):
                await result
        for round_id in range(1, max_rounds + 1):
            unfinished = [state for state in states if not state.gave_output]
            if not unfinished:
                break

            await self._run_strongly_stable_select(states, session_id, round_id)
            self._apply_selectblock_echo_rules(states, round_id)

        if any(not state.gave_output for state in states):
            raise RuntimeError("LocalAsyncYosoACS reached max_rounds without output")

        return [
            ACSOutput(
                node=state.node,
                selected_core=state.final_block,
                block_refs=state.final_block.flatten_refs(),
                output_round=state.final_round,
                selection_attempts=state.selection_attempts,
                coin_calls=state.coin_calls,
                trace=state.trace,
            )
            for state in states
        ]

    async def _admit_proposals(
        self,
        candidate_refs_by_node: Dict[int, Iterable[MessageRef]],
        proposal_validator,
    ) -> Dict[int, ProposedBlock]:
        proposed: Dict[int, ProposedBlock] = {}
        for node in range(self.M):
            proposal = ProposedBlock.build(
                node, candidate_refs_by_node.get(node, [])
            )
            if proposal_validator is not None:
                accepted = proposal_validator(node, proposal.refs)
                if inspect.isawaitable(accepted):
                    accepted = await accepted
                if not bool(accepted):
                    self.log.info(
                        "[B2A][ACSPropose] rejected proposer=%d refs=%d",
                        node,
                        len(proposal.refs),
                    )
                    continue
            proposed[node] = proposal
        return proposed

    def _init_acs_inputs(
        self,
        proposed: Dict[int, ProposedBlock],
    ) -> List[_NodeState]:
        if len(proposed) < self.quorum:
            raise ValueError("not enough proposed blocks for ACS quorum")

        # ΠACS step 3: all-parties CACSPropose means everyone collects the same
        # justified core U_i.  This core is then the input block to ΠSelectBlock.
        collected_core = CoreBlock.build(proposed.values())
        states = [
            _NodeState(
                node=node,
                input_block=collected_core,
                current_block=collected_core,
                trace=[
                    {
                        "phase": "ACSPropose",
                        "collected": len(proposed),
                        "accepted_proposers": sorted(proposed),
                        "core_digest": collected_core.digest(),
                    },
                    {
                        "phase": "SelectBlock.Input",
                        "stage": "SB_INPUT",
                        "grade": 0,
                        "collected": self.quorum,
                        "selected_source": node,
                        "block": collected_core.digest(),
                    },
                ],
            )
            for node in range(self.M)
        ]
        self.log.info(
            "[B2A][ACSPropose] nodes=%d collected=%d quorum=%d core=%s",
            self.M,
            len(proposed),
            self.quorum,
            collected_core.digest(),
        )
        return states

    async def _run_strongly_stable_select(
        self,
        states: List[_NodeState],
        session_id: str,
        round_id: int,
    ) -> None:
        weak_blocks, weak_grades = await self._run_weak_graded_select(
            states,
            session_id,
            round_id,
        )

        # Fig.19 Upgrade1: CUpgrade1 causal-casts the weak block.
        upgrade1_msgs = {
            state.node: weak_blocks[state.node]
            for state in states
            if not state.gave_output
        }

        # Fig.19 Upgrade2: if one block has quorum support, send (B, 1).
        upgrade2_msgs: Dict[int, Tuple[Optional[CoreBlock], int]] = {}
        for state in states:
            if state.gave_output:
                continue
            block, h_i = self._upgrade2_value(upgrade1_msgs)
            upgrade2_msgs[state.node] = (block, h_i)
            state.trace.append(
                {
                    "phase": "StronglyStable.Upgrade2",
                    "round": round_id,
                    "h": h_i,
                    "block": block.digest() if block is not None else None,
                }
            )

        # Fig.19 final output from Upgrade2 messages.
        for state in states:
            if state.gave_output:
                continue
            block, grade = self._strongly_stable_output(
                state.current_block,
                upgrade2_msgs,
            )
            state.current_block = block
            state.grade = grade
            state.selection_attempts += 1
            state.trace.append(
                {
                    "phase": "StronglyStable.Output",
                    "round": round_id,
                    "grade": grade,
                    "block": block.digest(),
                    "weak_grade": weak_grades[state.node],
                }
            )

    async def _run_weak_graded_select(
        self,
        states: List[_NodeState],
        session_id: str,
        round_id: int,
    ) -> Tuple[Dict[int, CoreBlock], Dict[int, int]]:
        U_by_node, T_by_node = self._run_graded_gather(states, round_id)
        coin_index = self.coin_indices.index(session_id, round_id)
        coin = await self.coin_oracle.next_coin(
            f"{session_id}:round:{round_id}",
            coin_index,
        )
        for state in states:
            if not state.gave_output:
                state.coin_calls += 1

        # Fig.18 FirstRoundCandidate: choose the minimum H(P_j || coin) over U_i.
        first_round: Dict[int, int] = {}
        for state in states:
            if state.gave_output:
                continue
            U_i = U_by_node[state.node]
            leader = self._leader_from_coin(U_i, coin, coin_index)
            first_round[state.node] = leader
            state.trace.append(
                {
                    "phase": "Weak.FirstRoundCandidate",
                    "round": round_id,
                    "coin_index": coin_index,
                    "leader": leader,
                    "u_size": len(U_i),
                }
            )

        # Fig.18 SecondRoundCandidate: all identical -> b=True.
        identical = len(set(first_round.values())) == 1
        agreed_leader = next(iter(first_round.values())) if identical and first_round else None
        second_round = {
            state.node: (agreed_leader, identical)
            for state in states
            if not state.gave_output
        }
        true_counts: Dict[int, int] = {}
        for leader, flag in second_round.values():
            if flag and leader is not None:
                true_counts[leader] = true_counts.get(leader, 0) + 1

        weak_blocks: Dict[int, CoreBlock] = {}
        weak_grades: Dict[int, int] = {}
        for state in states:
            if state.gave_output:
                continue
            U_i = U_by_node[state.node]
            T_i = T_by_node[state.node]
            block, grade, output_leader, true_count = self._weak_output_from_counts(
                state.current_block,
                U_i,
                T_i,
                true_counts,
                self.quorum,
            )
            weak_blocks[state.node] = block
            weak_grades[state.node] = grade
            state.trace.append(
                {
                    "phase": "Weak.Output",
                    "round": round_id,
                    "leader": output_leader,
                    "first_round_identical": identical,
                    "true_count": true_count,
                    "true_counts": true_counts,
                    "grade": grade,
                    "block": block.digest(),
                }
            )
        return weak_blocks, weak_grades

    def _weak_output_from_counts(
        self,
        fallback: CoreBlock,
        U_i: Dict[int, CoreBlock],
        T_i: Dict[int, CoreBlock],
        true_counts: Dict[int, int],
        quorum: int,
    ) -> Tuple[CoreBlock, int, Optional[int], int]:
        for leader, count in sorted(true_counts.items()):
            if count >= quorum and leader in T_i:
                return T_i[leader], 2, leader, count
        for leader, count in sorted(true_counts.items()):
            if count > 0 and leader in U_i and leader not in T_i:
                return U_i[leader], 1, leader, count
        return fallback, 0, None, 0

    def _run_graded_gather(
        self,
        states: List[_NodeState],
        round_id: int,
    ) -> Tuple[Dict[int, Dict[int, CoreBlock]], Dict[int, Dict[int, CoreBlock]]]:
        # Fig.16 ΠGather: U_i^0={(P_i,B_i)}, then three union rounds.
        gather_U = {
            state.node: {state.node: state.current_block}
            for state in states
            if not state.gave_output
        }
        for gather_round in range(1, 4):
            union = self._union_core_maps(gather_U.values())
            gather_U = {
                state.node: dict(union)
                for state in states
                if not state.gave_output
            }
            for state in states:
                if state.gave_output:
                    continue
                state.trace.append(
                    {
                        "phase": "Gather",
                        "select_round": round_id,
                        "gather_round": gather_round,
                        "u_size": len(gather_U[state.node]),
                    }
                )

        # Fig.17 ΠGradedGather: collect U'_j, output union and intersection.
        union_U = self._union_core_maps(gather_U.values())
        inter_T = self._intersection_core_maps(gather_U.values())
        U_by_node = {}
        T_by_node = {}
        for state in states:
            if state.gave_output:
                continue
            U_by_node[state.node] = dict(union_U)
            T_by_node[state.node] = dict(inter_T)
            state.trace.append(
                {
                    "phase": "GradedGather",
                    "round": round_id,
                    "u_size": len(union_U),
                    "t_size": len(inter_T),
                }
            )
        return U_by_node, T_by_node

    def _apply_selectblock_echo_rules(
        self,
        states: List[_NodeState],
        round_id: int,
    ) -> None:
        # Fig.20 rule 4: grade-2 outputs are echoed; in local all-parties mode
        # every node observes all echoes immediately.
        grade2_blocks = {
            state.current_block
            for state in states
            if not state.gave_output and state.grade == 2
        }
        if not grade2_blocks:
            return
        selected = sorted(grade2_blocks, key=lambda block: block.digest())[0]
        for state in states:
            if state.gave_output:
                continue
            state.gave_output = True
            state.final_block = selected
            state.final_round = round_id
            state.trace.append(
                {
                    "phase": "SelectBlock.Output",
                    "round": round_id,
                    "block": selected.digest(),
                    "grade": state.grade,
                }
            )

    def _upgrade2_value(
        self,
        upgrade1_msgs: Dict[int, CoreBlock],
    ) -> Tuple[Optional[CoreBlock], int]:
        counts: Dict[CoreBlock, int] = {}
        for block in upgrade1_msgs.values():
            counts[block] = counts.get(block, 0) + 1
        for block, count in sorted(counts.items(), key=lambda item: item[0].digest()):
            if count >= self.quorum:
                return block, 1
        return None, 0

    def _strongly_stable_output(
        self,
        fallback: CoreBlock,
        upgrade2_msgs: Dict[int, Tuple[Optional[CoreBlock], int]],
    ) -> Tuple[CoreBlock, int]:
        non_bottom = [
            block for block, h in upgrade2_msgs.values()
            if h == 1 and block is not None
        ]
        if len(non_bottom) == len(upgrade2_msgs) and len(set(non_bottom)) == 1:
            return non_bottom[0], 2
        if non_bottom:
            return sorted(set(non_bottom), key=lambda block: block.digest())[0], 1
        return fallback, 0

    def _leader_from_coin(
        self,
        U_i: Dict[int, CoreBlock],
        coin: bytes,
        selection_index: int = 1,
    ) -> int:
        # ``selection_index`` remains in the compatibility signature because
        # it identifies the coin call.  Fig. 18 does not use the role trace for
        # leader selection: it hashes only candidates actually present in U_i.
        del selection_index
        return _candidate_leader(U_i, coin, population_size=self.M)

    def _union_core_maps(
        self,
        maps: Iterable[Dict[int, CoreBlock]],
    ) -> Dict[int, CoreBlock]:
        out: Dict[int, CoreBlock] = {}
        for mapping in maps:
            out.update(mapping)
        return out

    def _intersection_core_maps(
        self,
        maps: Iterable[Dict[int, CoreBlock]],
    ) -> Dict[int, CoreBlock]:
        maps = list(maps)
        if not maps:
            return {}
        common = set(maps[0])
        for mapping in maps[1:]:
            common &= set(mapping)
        return {party: maps[0][party] for party in sorted(common)}


def _core_map_to_obj(mapping: Dict[int, CoreBlock]) -> dict:
    return {
        str(party): block.to_obj()
        for party, block in sorted(mapping.items())
    }


def _core_map_from_obj(data: dict) -> Dict[int, CoreBlock]:
    return {
        int(party): CoreBlock.from_obj(block)
        for party, block in data.items()
    }


def acs_rbc_prefix(epoch: str, session_id: str, stage: str, source: int) -> str:
    return f"AYTOB:{epoch}:ACS:{session_id}:{stage}:{source}"


def acs_rbc_echo_label(epoch: str, session_id: str, stage: str, source: int) -> bytes:
    return f"AYTOB:RBC:{epoch}:ACS:{session_id}:{stage}:{source}:ECHO".encode(
        "utf-8"
    )


def acs_rbc_ready_label(epoch: str, session_id: str, stage: str, source: int) -> bytes:
    return f"AYTOB:RBC:{epoch}:ACS:{session_id}:{stage}:{source}:READY".encode(
        "utf-8"
    )


def acs_activation_label(epoch: str, session_id: str, stage: str) -> bytes:
    return f"AYTOB:ACS-ACT:{epoch}:{session_id}:{stage}".encode("utf-8")


class NetworkAsyncYosoACS:
    """Network all-parties implementation of the B2A stage boundaries.

    This is Phase B2B.  Every causal-cast/broadcast activation rule is mapped
    to one ``YosoRBC`` instance per potential sender.  The first smoke version
    defaults to waiting for all M outputs in each all-honest stage; set
    ``collect_all=False`` to use the paper threshold ``M-t``.
    """

    def __init__(
        self,
        *,
        my_id: int,
        M: int,
        t: int,
        epoch: str,
        send,
        shared_subscribe,
        crypto,
        vrf_sk,
        vrf_pks,
        coin_oracle: Optional[CoinOracle] = None,
        committee_trace: Optional[CommitteeRoleTrace] = None,
        collect_all: bool = True,
        activation_mode: str = "all",
        expected_committee_size: Optional[int] = None,
        committee_m: Optional[int] = None,
        quorum_h: Optional[int] = None,
        collect_threshold: Optional[int] = None,
        stage_timeout: Optional[float] = None,
        echo_adoption_wait: float = 0.25,
        output_service_grace: float = 0.25,
        committee_selection_mode: str = BERNOULLI_COMMITTEE_MODE,
        fixed_committee_seed: str = "",
        logger: Optional[logging.Logger] = None,
    ):
        if M <= 0:
            raise ValueError("M must be positive")
        if t < 0 or t >= M:
            raise ValueError("t must satisfy 0 <= t < M")
        if not 0 <= my_id < M:
            raise ValueError("my_id must satisfy 0 <= my_id < M")
        self.my_id = my_id
        self.M = M
        self.t = t
        self.quorum = M - t
        self.epoch = epoch
        self.send = send
        self.shared_subscribe = shared_subscribe
        self.crypto = crypto
        self.vrf_sk = vrf_sk
        self.vrf_pks = vrf_pks
        self.coin_oracle = coin_oracle or DeterministicCoinOracle(epoch)
        self.coin_indices = SessionCoinIndexAllocator()
        self.log = logger or logging.getLogger(f"network_async_yoso_acs.{my_id}")

        self.echo_adoption_wait = float(echo_adoption_wait)
        self.output_service_grace = float(output_service_grace)
        self.stage_timeout = (
            None if stage_timeout is None else float(stage_timeout)
        )
        if self.echo_adoption_wait < 0:
            raise ValueError("echo_adoption_wait must be non-negative")
        if self.output_service_grace < 0:
            raise ValueError("output_service_grace must be non-negative")
        if self.stage_timeout is not None and self.stage_timeout <= 0:
            raise ValueError("stage_timeout must be positive")

        if activation_mode not in {"all", "vrf"}:
            raise ValueError("activation_mode must be 'all' or 'vrf'")
        if activation_mode == "vrf" and committee_trace is None:
            raise ValueError(
                "VRF activation requires an explicit pre-generated "
                "committee role trace"
            )
        if activation_mode == "vrf" and not vrf_backend_available():
            raise RuntimeError(
                "cryptographic VRF backend is required in VRF mode"
            )
        if activation_mode == "vrf" and (
            vrf_sk is None or vrf_pks is None or len(vrf_pks) < M
        ):
            raise ValueError(
                "VRF mode requires a local secret key and M public keys"
            )
        self.committee_trace = committee_trace or AllMembersCommitteeTrace(M)
        self.activation_mode = activation_mode
        self.committee_selection_mode = normalize_committee_selection_mode(
            committee_selection_mode
        )
        self.fixed_committee_seed = str(fixed_committee_seed)
        if self.activation_mode == "vrf" and self.stage_timeout is None:
            self.stage_timeout = 30.0
        self.expected_committee_size = int(
            M if expected_committee_size is None else expected_committee_size
        )
        self.committee_m = int(
            self.expected_committee_size
            if committee_m is None
            else committee_m
        )
        self.quorum_h = int(self.quorum if quorum_h is None else quorum_h)
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
        if self.expected_committee_size < self.quorum_h:
            raise ValueError(
                "expected_committee_size must be at least quorum_h"
            )
        self.agreement_t = self.committee_m - self.quorum_h
        self.threshold_fraction = (
            1.0
            if self.activation_mode == "all"
            else self.expected_committee_size / self.M
        )
        if (
            self.activation_mode == "vrf"
            and collect_threshold is not None
            and int(collect_threshold) != self.quorum_h
        ):
            raise ValueError(
                "VRF collect_threshold must equal quorum_h"
            )
        if collect_threshold is not None:
            self.collect_threshold = collect_threshold
        elif self.activation_mode == "all":
            self.collect_threshold = M if collect_all else self.quorum
        else:
            self.collect_threshold = self.quorum_h
        self.protocol_quorum = (
            self.quorum if self.activation_mode == "all" else self.quorum_h
        )
        self.ready_amplify_threshold = (self.quorum_h + 1) // 2 + 1
        self.rbc_instance_count = 0
        self.activation_send_count = 0
        self.stage_collect_counts: List[dict] = []
        self.proposal_received_count = 0
        self.proposal_checked_count = 0
        self.proposal_buffered_count = 0
        self.proposal_accepted_count = 0
        self.proposal_rejected_count = 0
        self.proposal_validation_invocations = 0
        self.proposal_validation_seconds = 0.0
        self.proposal_buffered_wait_seconds = 0.0
        self.proposal_admission_records: List[dict] = []
        self._proposal_buffered_since: Dict[Tuple[str, int], float] = {}
        self._proposal_terminal_keys = set()
        self._proposal_received_keys = set()
        self.select_block_input_validation_invocations = 0
        self.select_block_input_validation_seconds = 0.0
        self._select_block_input_validation_cache = set()
        self._selectblock_round_outputs: Dict[
            Tuple[str, int], Tuple[CoreBlock, int]
        ] = {}
        self.echo_received_count = 0
        self.echo_accepted_count = 0
        self.echo_rejected_count = 0
        self.echo_adopted_count = 0
        self.echo_direct_output_count = 0
        self.output_service_windows = 0
        self.output_service_seconds = 0.0
        self._echo_received_keys = set()
        self._echo_terminal_keys = set()
        _, _, _, _, srs = self.crypto
        self.g_hex = generator_hex_from_srs(srs)

    def proposal_metrics_snapshot(self) -> dict:
        now = time.perf_counter()
        in_flight_wait = sum(
            now - started
            for started in self._proposal_buffered_since.values()
        )
        return {
            "received": self.proposal_received_count,
            "checked": self.proposal_checked_count,
            "buffered": self.proposal_buffered_count,
            "buffered_current": len(self._proposal_buffered_since),
            "buffered_wait_seconds": (
                self.proposal_buffered_wait_seconds + in_flight_wait
            ),
            "accepted": self.proposal_accepted_count,
            "rejected": self.proposal_rejected_count,
            "validation_invocations": (
                self.proposal_validation_invocations
            ),
            "validation_seconds": self.proposal_validation_seconds,
            "validator_attempts": self.proposal_validation_invocations,
            "validator_attempt_seconds": self.proposal_validation_seconds,
            "accounting_scope": "admission-callback-attempts",
            "records": list(self.proposal_admission_records),
        }

    def echo_metrics_snapshot(self) -> dict:
        return {
            "received": self.echo_received_count,
            "accepted": self.echo_accepted_count,
            "rejected": self.echo_rejected_count,
            "adopted": self.echo_adopted_count,
            "direct_outputs": self.echo_direct_output_count,
            "service_windows": self.output_service_windows,
            "service_seconds": self.output_service_seconds,
        }

    def select_block_input_metrics_snapshot(self) -> dict:
        return {
            "validation_invocations": (
                self.select_block_input_validation_invocations
            ),
            "validation_seconds": self.select_block_input_validation_seconds,
            "stage": "SB_INPUT",
            "sequential_yoso_rbc_stages_per_acs": 1,
        }

    async def run(
        self,
        candidate_refs: Iterable[MessageRef],
        *,
        session_id: str,
        max_rounds: int = 10,
        proposal_validator=None,
        proposal_fixed_cb=None,
        output_fixed_cb=None,
    ) -> ACSOutput:
        self.coin_indices.reserve(session_id, max_rounds)
        current_block = await self.collect_proposals(
            candidate_refs,
            session_id=session_id,
            proposal_validator=proposal_validator,
        )
        if proposal_fixed_cb is not None:
            result = proposal_fixed_cb(
                {
                    "session_id": session_id,
                    "collected": len(current_block.proposals),
                    "core_digest": current_block.digest(),
                    "refs": [
                        ref.encode() for ref in current_block.flatten_refs()
                    ],
                }
            )
            if inspect.isawaitable(result):
                await result
        trace = [
            {
                "phase": "ACSPropose",
                "collected": len(current_block.proposals),
                "core_digest": current_block.digest(),
                "admission": self.proposal_metrics_snapshot(),
            }
        ]
        current_block, input_trace = await self._select_block_input(
            session_id,
            current_block,
            proposal_validator=proposal_validator,
        )
        trace.append(input_trace)
        grade = 0
        coin_calls = 0
        selection_attempts = 0
        echo_tasks: Dict[int, asyncio.Task] = {}
        try:
            for round_id in range(1, max_rounds + 1):
                echo_tasks[round_id] = asyncio.create_task(
                    self._collect_selectblock_echo(
                        session_id,
                        round_id,
                    )
                )
                current_block, grade, round_trace, coin_used = (
                    await self._select_round(
                        session_id,
                        round_id,
                        current_block,
                    )
                )
                self._selectblock_round_outputs[
                    (session_id, round_id)
                ] = (current_block, grade)
                trace.extend(round_trace)
                coin_calls += coin_used
                selection_attempts += 1
                if grade == 2:
                    echo_trace = await self._send_selectblock_echo(
                        session_id,
                        round_id,
                        current_block,
                    )
                    self.echo_direct_output_count += 1
                    trace.append(echo_trace)
                    trace.append(
                        {
                            "phase": "SelectBlock.Output",
                            "round": round_id,
                            "block": current_block.digest(),
                            "grade": grade,
                            "agreement_path": "direct_grade2",
                            "echo_sent": echo_trace.get("sent", False),
                        }
                    )
                    output = ACSOutput(
                        node=self.my_id,
                        selected_core=current_block,
                        block_refs=current_block.flatten_refs(),
                        output_round=round_id,
                        selection_attempts=selection_attempts,
                        coin_calls=coin_calls,
                        trace=trace,
                    )
                    if output_fixed_cb is not None:
                        result = output_fixed_cb(output)
                        if inspect.isawaitable(result):
                            await result
                    await self._serve_after_selectblock_output()
                    return output

                propagated = await self._wait_for_selectblock_echo(
                    echo_tasks,
                    through_round=round_id,
                    timeout=self.echo_adoption_wait,
                )
                if propagated is not None:
                    echo_round, source, echo_block = propagated
                    self.echo_adopted_count += 1
                    trace.append(
                        {
                            "phase": "SelectBlock.AgreementEcho",
                            "round": round_id,
                            "echo_round": echo_round,
                            "echo_source": source,
                            "local_grade": grade,
                            "agreement_path": "propagated_grade2",
                            "block": echo_block.digest(),
                        }
                    )
                    trace.append(
                        {
                            "phase": "SelectBlock.Output",
                            "round": echo_round,
                            "observed_in_round": round_id,
                            "block": echo_block.digest(),
                            "grade": 2,
                            "agreement_path": "propagated_grade2",
                            "echo_source": source,
                        }
                    )
                    output = ACSOutput(
                        node=self.my_id,
                        selected_core=echo_block,
                        block_refs=echo_block.flatten_refs(),
                        output_round=echo_round,
                        selection_attempts=selection_attempts,
                        coin_calls=coin_calls,
                        trace=trace,
                    )
                    if output_fixed_cb is not None:
                        result = output_fixed_cb(output)
                        if inspect.isawaitable(result):
                            await result
                    await self._serve_after_selectblock_output()
                    return output

                trace.append(
                    {
                        "phase": "SelectBlock.AgreementEcho",
                        "round": round_id,
                        "local_grade": grade,
                        "agreement_path": "no_grade2_echo_yet",
                        "active_echo_rounds": sorted(echo_tasks),
                    }
                )

            raise RuntimeError(
                "NetworkAsyncYosoACS reached max_rounds without output"
            )
        finally:
            for task in echo_tasks.values():
                if not task.done():
                    task.cancel()
            if echo_tasks:
                await asyncio.gather(
                    *echo_tasks.values(),
                    return_exceptions=True,
                )
            for key in tuple(self._selectblock_round_outputs):
                if key[0] == session_id:
                    self._selectblock_round_outputs.pop(key, None)
            self._select_block_input_validation_cache = {
                key
                for key in self._select_block_input_validation_cache
                if key[0] != session_id
            }

    async def collect_proposals(
        self,
        candidate_refs: Iterable[MessageRef],
        *,
        session_id: str,
        proposal_validator=None,
    ) -> CoreBlock:
        """Collect only availability-resolved, externally valid proposals."""

        proposed = ProposedBlock.build(self.my_id, candidate_refs)

        async def _admit(source: int, msg: dict) -> bool:
            key = (session_id, source)
            if key not in self._proposal_received_keys:
                self._proposal_received_keys.add(key)
                self.proposal_received_count += 1
            try:
                if msg.get("kind") != "ACSPropose":
                    raise ValueError("expected ACSPropose message")
                proposal = ProposedBlock.from_obj(msg["proposal"])
                if proposal.proposer != source:
                    raise ValueError(
                        "proposal identity does not match RBC source"
                    )
            except (KeyError, TypeError, ValueError) as exc:
                self._finish_proposal_admission(
                    key,
                    source,
                    ProposalAdmission(False),
                    reason=f"malformed:{exc}",
                )
                return False

            started = time.perf_counter()
            self.proposal_validation_invocations += 1
            try:
                if proposal_validator is None:
                    result = ProposalAdmission(
                        True,
                        transcript_count=len(proposal.refs),
                    )
                else:
                    result = proposal_validator(source, proposal.refs)
                    if inspect.isawaitable(result):
                        result = await result
                    result = self._normalize_proposal_admission(
                        result,
                        transcript_count=len(proposal.refs),
                    )
            except LookupError as exc:
                # KeyError/IndexError describe malformed or unknown proposal
                # content, not a dependency that can arrive later.  Custom
                # LookupError subclasses (for example
                # MissingTranscriptReferences) are the retry signal.
                if isinstance(exc, (KeyError, IndexError)):
                    self._finish_proposal_admission(
                        key,
                        source,
                        ProposalAdmission(False),
                        reason=f"validator-error:{exc}",
                    )
                    return False
                if key not in self._proposal_buffered_since:
                    self._proposal_buffered_since[key] = (
                        time.perf_counter()
                    )
                    self.proposal_buffered_count += 1
                raise
            except Exception as exc:
                self._finish_proposal_admission(
                    key,
                    source,
                    ProposalAdmission(False),
                    reason=f"validator-error:{exc}",
                )
                return False
            finally:
                self.proposal_validation_seconds += (
                    time.perf_counter() - started
                )

            self._finish_proposal_admission(
                key,
                source,
                result,
                reason="accepted" if result.accepted else "predicate-false",
            )
            return result.accepted

        try:
            acs_msgs = await self._collect_json_stage(
                session_id,
                "ACSPROP",
                {"kind": "ACSPropose", "proposal": proposed.to_obj()},
                need=self.collect_threshold,
                message_validator=_admit,
            )
        finally:
            self._close_buffered_proposal_session(session_id)
        proposals = [
            ProposedBlock.from_obj(acs_msgs[source]["proposal"])
            for source in sorted(acs_msgs)
        ]
        if len(proposals) < self.collect_threshold:
            raise RuntimeError(
                "ACSPROP ended without enough justified proposals"
            )
        return CoreBlock.build(proposals)

    async def _select_block_input(
        self,
        session_id: str,
        current_block: CoreBlock,
        *,
        proposal_validator=None,
    ) -> Tuple[CoreBlock, dict]:
        """Implement Fig. 20's one-time ``C_SelectBlockInput`` handoff.

        Committee members causal-cast their locally justified ACS core with
        grade zero.  A receiver accepts a core only after independently
        validating its constituent proposer blocks, then uses one justified
        output as the first StronglyStableGradedSelectBlock input.
        """

        async def _validate(_source: int, msg: dict) -> bool:
            try:
                if msg.get("kind") != "SelectBlockInput":
                    raise ValueError("expected SelectBlockInput message")
                if int(msg.get("grade", -1)) != 0:
                    raise ValueError("SelectBlockInput grade must be zero")
                block_obj = msg["block"]
                block = CoreBlock.from_obj(block_obj)
                if block.to_obj() != block_obj:
                    raise ValueError("SelectBlockInput block is not canonical")
                if len(block.proposals) < self.protocol_quorum:
                    raise ValueError(
                        "SelectBlockInput core is smaller than protocol quorum"
                    )
                if any(
                    not 0 <= proposal.proposer < self.M
                    for proposal in block.proposals
                ):
                    raise ValueError(
                        "SelectBlockInput proposer is outside the population"
                    )
            except (KeyError, TypeError, ValueError):
                return False

            for proposal in block.proposals:
                cache_key = (session_id, proposal)
                if cache_key in self._select_block_input_validation_cache:
                    continue
                started = time.perf_counter()
                self.select_block_input_validation_invocations += 1
                try:
                    if proposal_validator is None:
                        result = ProposalAdmission(
                            True,
                            transcript_count=len(proposal.refs),
                        )
                    else:
                        result = proposal_validator(
                            proposal.proposer,
                            proposal.refs,
                        )
                        if inspect.isawaitable(result):
                            result = await result
                        result = self._normalize_proposal_admission(
                            result,
                            transcript_count=len(proposal.refs),
                        )
                    if not result.accepted:
                        return False
                    self._select_block_input_validation_cache.add(cache_key)
                finally:
                    self.select_block_input_validation_seconds += (
                        time.perf_counter() - started
                    )
            return True

        messages = await self._collect_json_stage(
            session_id,
            "SB_INPUT",
            {
                "kind": "SelectBlockInput",
                "grade": 0,
                "block": current_block.to_obj(),
            },
            need=self.collect_threshold,
            message_validator=_validate,
        )
        if len(messages) < self.collect_threshold:
            raise RuntimeError(
                "SelectBlockInput ended without enough justified inputs"
            )
        candidates = [
            (CoreBlock.from_obj(message["block"]), source)
            for source, message in messages.items()
        ]
        selected_block, selected_source = min(
            candidates,
            key=lambda item: (item[0].digest(), item[1]),
        )
        return selected_block, {
            "phase": "SelectBlock.Input",
            "stage": "SB_INPUT",
            "grade": 0,
            "collected": len(messages),
            "selected_source": selected_source,
            "block": selected_block.digest(),
            "validation_invocations": (
                self.select_block_input_validation_invocations
            ),
            "validation_seconds": self.select_block_input_validation_seconds,
        }

    def _close_buffered_proposal_session(
        self,
        session_id: str,
    ) -> None:
        now = time.perf_counter()
        for key, started in tuple(self._proposal_buffered_since.items()):
            if key[0] != session_id:
                continue
            waited = now - started
            self.proposal_buffered_wait_seconds += waited
            self._proposal_buffered_since.pop(key, None)
            self.proposal_admission_records.append(
                {
                    "session_id": key[0],
                    "proposer": key[1],
                    "accepted": False,
                    "reason": "unresolved-at-stage-close",
                    "qid_count": 0,
                    "transcript_count": 0,
                    "buffered_wait_seconds": waited,
                }
            )

    @staticmethod
    def _normalize_proposal_admission(
        value,
        *,
        transcript_count: int,
    ) -> ProposalAdmission:
        if isinstance(value, ProposalAdmission):
            return value
        if isinstance(value, dict):
            accepted = value.get("accepted", value.get("valid"))
            if not isinstance(accepted, bool):
                raise TypeError(
                    "proposal admission mapping needs bool accepted"
                )
            return ProposalAdmission(
                accepted=accepted,
                qid_count=int(value.get("qid_count", 0)),
                transcript_count=int(
                    value.get("transcript_count", transcript_count)
                ),
            )
        if isinstance(value, bool):
            return ProposalAdmission(
                value,
                transcript_count=transcript_count,
            )
        raise TypeError(
            "proposal validator must return bool, mapping, or "
            "ProposalAdmission"
        )

    def _finish_proposal_admission(
        self,
        key: Tuple[str, int],
        source: int,
        result: ProposalAdmission,
        *,
        reason: str,
    ) -> None:
        if key in self._proposal_terminal_keys:
            return
        self._proposal_terminal_keys.add(key)
        self.proposal_checked_count += 1
        if result.accepted:
            self.proposal_accepted_count += 1
        else:
            self.proposal_rejected_count += 1
        buffered_since = self._proposal_buffered_since.pop(key, None)
        buffered_wait = (
            time.perf_counter() - buffered_since
            if buffered_since is not None
            else 0.0
        )
        self.proposal_buffered_wait_seconds += buffered_wait
        self.proposal_admission_records.append(
            {
                "session_id": key[0],
                "proposer": source,
                "accepted": result.accepted,
                "reason": reason,
                "qid_count": result.qid_count,
                "transcript_count": result.transcript_count,
                "buffered_wait_seconds": buffered_wait,
            }
        )

    async def _select_round(
        self,
        session_id: str,
        round_id: int,
        current_block: CoreBlock,
    ) -> Tuple[CoreBlock, int, List[dict], int]:
        trace: List[dict] = []
        U_i, T_i, gather_trace = await self._graded_gather(
            session_id,
            round_id,
            current_block,
        )
        trace.extend(gather_trace)

        coin_index = self.coin_indices.index(session_id, round_id)
        coin = await self.coin_oracle.next_coin(
            f"{session_id}:round:{round_id}",
            coin_index,
        )
        leader = self._leader_from_coin(U_i, coin, coin_index)
        trace.append(
            {
                "phase": "Weak.FirstRoundCandidate",
                "round": round_id,
                "coin_index": coin_index,
                "leader": leader,
                "u_size": len(U_i),
            }
        )
        fr_msgs = await self._collect_json_stage(
            session_id,
            f"WGSB_FR:{round_id}",
            {"kind": "FirstRoundCandidate", "leader": leader},
            need=self.collect_threshold,
        )
        leaders = [int(msg["leader"]) for msg in fr_msgs.values()]
        identical = len(set(leaders)) == 1
        agreed_leader = leaders[0] if identical and leaders else None

        sr_msgs = await self._collect_json_stage(
            session_id,
            f"WGSB_SR:{round_id}",
            {
                "kind": "SecondRoundCandidate",
                "leader": agreed_leader,
                "b": identical,
            },
            need=self.collect_threshold,
        )
        true_counts: Dict[int, int] = {}
        for msg in sr_msgs.values():
            if not msg.get("b") or msg.get("leader") is None:
                continue
            leader = int(msg["leader"])
            true_counts[leader] = true_counts.get(leader, 0) + 1
        weak_block, weak_grade, output_leader, true_count = self._weak_output_from_counts(
            current_block,
            U_i,
            T_i,
            true_counts,
            self.protocol_quorum,
        )
        trace.append(
            {
                "phase": "Weak.Output",
                "round": round_id,
                "leader": output_leader,
                "first_round_identical": identical,
                "true_count": true_count,
                "true_counts": true_counts,
                "protocol_quorum": self.protocol_quorum,
                "grade": weak_grade,
                "block": weak_block.digest(),
            }
        )

        up1_msgs = await self._collect_json_stage(
            session_id,
            f"SSGSB_UP1:{round_id}",
            {"kind": "Upgrade1", "block": weak_block.to_obj()},
            need=self.collect_threshold,
        )
        up1_blocks = [
            CoreBlock.from_obj(msg["block"])
            for msg in up1_msgs.values()
            if msg.get("kind") == "Upgrade1"
        ]
        upgrade2_block, h_i = self._upgrade2_value(up1_blocks)
        trace.append(
            {
                "phase": "StronglyStable.Upgrade2",
                "round": round_id,
                "h": h_i,
                "block": upgrade2_block.digest() if upgrade2_block is not None else None,
            }
        )

        up2_msgs = await self._collect_json_stage(
            session_id,
            f"SSGSB_UP2:{round_id}",
            {
                "kind": "Upgrade2",
                "block": upgrade2_block.to_obj() if upgrade2_block is not None else None,
                "h": h_i,
            },
            need=self.collect_threshold,
        )
        up2_values = []
        for msg in up2_msgs.values():
            block = CoreBlock.from_obj(msg["block"]) if msg.get("block") else None
            up2_values.append((block, int(msg.get("h", 0))))
        strong_block, strong_grade = self._strongly_stable_output(
            current_block,
            up2_values,
        )
        trace.append(
            {
                "phase": "StronglyStable.Output",
                "round": round_id,
                "grade": strong_grade,
                "block": strong_block.digest(),
                "weak_grade": weak_grade,
            }
        )
        return strong_block, strong_grade, trace, 1

    async def _graded_gather(
        self,
        session_id: str,
        round_id: int,
        current_block: CoreBlock,
    ) -> Tuple[Dict[int, CoreBlock], Dict[int, CoreBlock], List[dict]]:
        trace: List[dict] = []
        U_i = {self.my_id: current_block}
        for gather_round in range(1, 4):
            msgs = await self._collect_json_stage(
                session_id,
                f"GATHER:{round_id}:{gather_round}",
                {
                    "kind": "Gather",
                    "round": round_id,
                    "gather_round": gather_round,
                    "U": _core_map_to_obj(U_i),
                },
                need=self.collect_threshold,
            )
            U_i = self._union_core_maps(
                _core_map_from_obj(msg["U"])
                for msg in msgs.values()
                if msg.get("kind") == "Gather"
            )
            trace.append(
                {
                    "phase": "Gather",
                    "select_round": round_id,
                    "gather_round": gather_round,
                    "u_size": len(U_i),
                }
            )

        gg_msgs = await self._collect_json_stage(
            session_id,
            f"GGATHER:{round_id}",
            {
                "kind": "GradedGather",
                "round": round_id,
                "U": _core_map_to_obj(U_i),
            },
            need=self.collect_threshold,
        )
        received = [
            _core_map_from_obj(msg["U"])
            for msg in gg_msgs.values()
            if msg.get("kind") == "GradedGather"
        ]
        U_out = self._union_core_maps(received)
        T_out = self._intersection_core_maps(received)
        trace.append(
            {
                "phase": "GradedGather",
                "round": round_id,
                "u_size": len(U_out),
                "t_size": len(T_out),
            }
        )
        return U_out, T_out, trace

    async def _collect_json_stage(
        self,
        session_id: str,
        stage: str,
        my_payload: Optional[dict],
        *,
        need: int,
        message_validator=None,
        dependency_retry_interval: float = 0.05,
    ) -> Dict[int, dict]:
        rbc_sessions = {}
        reactor_tasks = []
        output_tasks = {}
        pbk, pvk, pks, sk, srs = self.crypto
        for source in range(self.M):
            rbc = YosoRBC(
                pks,
                sk,
                pbk,
                pvk,
                self.expected_committee_size,
                self.agreement_t,
                srs,
                self.my_id,
                self.send,
                None,
                prefix=acs_rbc_prefix(self.epoch, session_id, stage, source),
                shared_subscribe=self.shared_subscribe,
                threshold_fraction=self.threshold_fraction,
                network_size=self.M,
                vrf_sk=self.vrf_sk,
                vrf_pks=self.vrf_pks,
                expected_committee_size=self.expected_committee_size,
                committee_m=self.committee_m,
                quorum_h=self.quorum_h,
                ready_amplify_threshold=self.ready_amplify_threshold,
                vrf_echo_label=acs_rbc_echo_label(self.epoch, session_id, stage, source),
                vrf_ready_label=acs_rbc_ready_label(self.epoch, session_id, stage, source),
                require_vrf_proofs=(self.activation_mode == "vrf"),
                committee_selection_mode=self.committee_selection_mode,
                fixed_committee_seed=self.fixed_committee_seed,
            )
            self.rbc_instance_count += 1
            rbc_sessions[source] = rbc
            reactor_tasks.append(asyncio.create_task(rbc.reactor()))
            output_tasks[asyncio.create_task(rbc.output_queue.get())] = source

        sender_task = None
        local_selected = False
        if my_payload is not None:
            selected, activation_proof = self._activation_proof(session_id, stage)
            local_selected = bool(selected)
            if selected:
                payload = dict(my_payload)
                payload["from"] = self.my_id
                if activation_proof is not None:
                    payload["activation_proof"] = activation_proof
                self.activation_send_count += 1
                sender_task = asyncio.create_task(
                    self._delayed_send(
                        rbc_sessions[self.my_id],
                        _canonical_json(payload),
                    )
                )
            else:
                self.log.info(
                    "[B5][%s] local node not selected for activation",
                    stage,
                )

        out: Dict[int, dict] = {}
        admission_tasks = {}
        received = set()
        proof_rejections = set()
        stage_started = time.perf_counter()
        stage_status = "incomplete"

        async def _evaluate_message(
            source: int,
            msg: dict,
            *,
            delay: float = 0.0,
        ):
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                accepted = message_validator(source, msg)
                if inspect.isawaitable(accepted):
                    accepted = await accepted
                return "accepted" if bool(accepted) else "rejected", None
            except LookupError as exc:
                return "pending", exc
            except Exception as exc:
                return "rejected", exc

        try:
            while len(out) < need and (
                output_tasks or admission_tasks
            ):
                wait_timeout = None
                if self.stage_timeout is not None:
                    wait_timeout = self.stage_timeout - (
                        time.perf_counter() - stage_started
                    )
                    if wait_timeout <= 0:
                        stage_status = "undersampled"
                        raise CommitteeUndersamplingError(
                            stage=stage,
                            need=need,
                            verified_selected=len(received),
                            collected=len(out),
                        )
                done, _ = await asyncio.wait(
                    set(output_tasks) | set(admission_tasks),
                    timeout=wait_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    stage_status = "undersampled"
                    raise CommitteeUndersamplingError(
                        stage=stage,
                        need=need,
                        verified_selected=len(received),
                        collected=len(out),
                    )
                for task in done:
                    if task in output_tasks:
                        source = output_tasks.pop(task)
                        try:
                            raw, _ = task.result()
                            msg = json.loads(raw.decode("utf-8"))
                            if not isinstance(msg, dict):
                                raise ValueError(
                                    "stage payload must be a JSON object"
                                )
                        except asyncio.CancelledError:
                            continue
                        except Exception as exc:
                            self.log.warning(
                                "[B2B] stage=%s source=%d malformed payload: %s",
                                stage,
                                source,
                                exc,
                            )
                            continue
                        if int(msg.get("from", source)) != source:
                            self.log.warning(
                                "[B2B] stage=%s prefix-source=%d "
                                "claimed=%s ignored",
                                stage,
                                source,
                                msg.get("from"),
                            )
                            continue
                        if not self._verify_activation_proof(
                            session_id,
                            stage,
                            source,
                            msg.get("activation_proof"),
                        ):
                            proof_rejections.add(source)
                            self.log.warning(
                                "[B5] stage=%s source=%d "
                                "activation proof rejected",
                                stage,
                                source,
                            )
                            continue
                        received.add(source)
                        if message_validator is None:
                            out[source] = msg
                        else:
                            admission_task = asyncio.create_task(
                                _evaluate_message(source, msg)
                            )
                            admission_tasks[admission_task] = (
                                source,
                                msg,
                            )
                        continue

                    source, msg = admission_tasks.pop(task)
                    try:
                        status, error = task.result()
                    except asyncio.CancelledError:
                        continue
                    if status == "accepted":
                        out[source] = msg
                    elif status == "pending":
                        admission_task = asyncio.create_task(
                            _evaluate_message(
                                source,
                                msg,
                                delay=dependency_retry_interval,
                            )
                        )
                        admission_tasks[admission_task] = (
                            source,
                            msg,
                        )
                    elif error is not None:
                        self.log.warning(
                            "[B2B] stage=%s source=%d external "
                            "validity rejected: %s",
                            stage,
                            source,
                            error,
                        )
            stage_status = (
                "complete" if len(out) >= need else "predicate-rejected"
            )
        finally:
            if sender_task is not None and not sender_task.done():
                sender_task.cancel()
            for task in output_tasks:
                task.cancel()
            for task in admission_tasks:
                task.cancel()
            for task in reactor_tasks:
                if not task.done():
                    task.cancel()
            tasks = (
                list(output_tasks)
                + list(admission_tasks)
                + reactor_tasks
            )
            if sender_task is not None:
                tasks.append(sender_task)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self.stage_collect_counts.append(
                {
                    "stage": stage,
                    "status": stage_status,
                    "activation_mode": self.activation_mode,
                    "local_selected": local_selected,
                    "selected_sender_ids": sorted(received),
                    "verified_selected_senders": len(received),
                    "proof_rejection_ids": sorted(proof_rejections),
                    "proof_rejections": len(proof_rejections),
                    "collected": len(out),
                    "need": need,
                    "protocol_quorum": self.protocol_quorum,
                    "received": len(received),
                    "pending_admissions": len(admission_tasks),
                    "elapsed_seconds": time.perf_counter()
                    - stage_started,
                }
            )

        self.log.info(
            "[B2B][%s] collected=%d need=%d",
            stage,
            len(out),
            need,
        )
        return out

    async def _collect_selectblock_echo(
        self,
        session_id: str,
        round_id: int,
    ) -> Tuple[int, int, CoreBlock]:
        """Receive one locally justified propagated grade-2 output."""

        async def _validate(source: int, msg: dict) -> bool:
            key = (session_id, round_id, source)
            if key not in self._echo_received_keys:
                self._echo_received_keys.add(key)
                self.echo_received_count += 1
            try:
                if msg.get("kind") != "SelectBlockEcho":
                    raise ValueError("expected SelectBlockEcho")
                if int(msg["round"]) != round_id:
                    raise ValueError("echo round does not match RBC stage")
                if int(msg["grade"]) != 2:
                    raise ValueError("echo must propagate grade 2")
                if int(msg["from"]) != source:
                    raise ValueError(
                        "echo identity does not match RBC source"
                    )
                block = CoreBlock.from_obj(msg["block"])
            except (KeyError, TypeError, ValueError) as exc:
                self._finish_echo_admission(key, accepted=False)
                self.log.warning(
                    "[B7A] malformed SelectBlock echo round=%d "
                    "source=%d: %s",
                    round_id,
                    source,
                    exc,
                )
                return False

            local = self._selectblock_round_outputs.get(
                (session_id, round_id)
            )
            if local is None:
                raise EchoRoundDependencyUnavailable(
                    f"local SelectBlock round {round_id} is incomplete"
                )
            local_block, local_grade = local
            if local_grade < 1 or local_block != block:
                self._finish_echo_admission(key, accepted=False)
                self.log.warning(
                    "[B7A] unjustified SelectBlock echo round=%d "
                    "source=%d local_grade=%d local=%s remote=%s",
                    round_id,
                    source,
                    local_grade,
                    local_block.digest(),
                    block.digest(),
                )
                return False
            self._finish_echo_admission(key, accepted=True)
            return True

        messages = await self._collect_json_stage(
            session_id,
            f"SB_ECHO:{round_id}",
            None,
            need=1,
            message_validator=_validate,
        )
        if not messages:
            raise RuntimeError(
                f"SelectBlock echo round {round_id} ended without output"
            )
        source = min(messages)
        block = CoreBlock.from_obj(messages[source]["block"])
        return round_id, source, block

    def _finish_echo_admission(
        self,
        key: Tuple[str, int, int],
        *,
        accepted: bool,
    ) -> None:
        if key in self._echo_terminal_keys:
            return
        self._echo_terminal_keys.add(key)
        if accepted:
            self.echo_accepted_count += 1
        else:
            self.echo_rejected_count += 1

    async def _wait_for_selectblock_echo(
        self,
        echo_tasks: Dict[int, asyncio.Task],
        *,
        through_round: int,
        timeout: float,
    ) -> Optional[Tuple[int, int, CoreBlock]]:
        relevant = {
            task
            for round_id, task in echo_tasks.items()
            if round_id <= through_round
        }
        if not relevant:
            return None
        done = {task for task in relevant if task.done()}
        if not done and timeout > 0:
            done, _ = await asyncio.wait(
                relevant,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
        candidates = []
        for task in done:
            try:
                candidates.append(task.result())
            except asyncio.CancelledError:
                continue
            except Exception as exc:
                self.log.warning(
                    "[B7A] SelectBlock echo listener failed: %s",
                    exc,
                )
        if not candidates:
            return None
        digests = {block.digest() for _round, _source, block in candidates}
        if len(digests) != 1:
            raise RuntimeError(
                "conflicting locally justified grade-2 echoes"
            )
        return min(
            candidates,
            key=lambda value: (
                value[0],
                value[1],
                value[2].digest(),
            ),
        )

    async def _serve_after_selectblock_output(self) -> None:
        """Keep shared RBC/dependency services alive for a bounded window."""

        self.output_service_windows += 1
        started = time.perf_counter()
        if self.output_service_grace > 0:
            await asyncio.sleep(self.output_service_grace)
        self.output_service_seconds += time.perf_counter() - started

    async def _send_selectblock_echo(
        self,
        session_id: str,
        round_id: int,
        block: CoreBlock,
    ) -> dict:
        stage = f"SB_ECHO:{round_id}"
        trace = {
            "phase": "SelectBlock.AgreementEcho",
            "round": round_id,
            "local_grade": 2,
            "agreement_path": "direct_grade2_echo_send",
            "sent": False,
        }
        selected, activation_proof = self._activation_proof(session_id, stage)
        trace["selected"] = selected
        if not selected:
            self.stage_collect_counts.append(
                {
                    "stage": stage,
                    "status": "send-skipped",
                    "activation_mode": self.activation_mode,
                    "local_selected": False,
                    "collected": 0,
                    "need": 0,
                    "send_only": True,
                    "sent": False,
                }
            )
            return trace

        payload = {
            "kind": "SelectBlockEcho",
            "round": round_id,
            "grade": 2,
            "block": block.to_obj(),
            "from": self.my_id,
        }
        if activation_proof is not None:
            payload["activation_proof"] = activation_proof
        pbk, pvk, pks, sk, srs = self.crypto
        rbc = YosoRBC(
            pks,
            sk,
            pbk,
            pvk,
            self.expected_committee_size,
            self.agreement_t,
            srs,
            self.my_id,
            self.send,
            None,
            prefix=acs_rbc_prefix(self.epoch, session_id, stage, self.my_id),
            shared_subscribe=self.shared_subscribe,
            threshold_fraction=self.threshold_fraction,
            network_size=self.M,
            vrf_sk=self.vrf_sk,
            vrf_pks=self.vrf_pks,
            expected_committee_size=self.expected_committee_size,
            committee_m=self.committee_m,
            quorum_h=self.quorum_h,
            ready_amplify_threshold=self.ready_amplify_threshold,
            vrf_echo_label=acs_rbc_echo_label(self.epoch, session_id, stage, self.my_id),
            vrf_ready_label=acs_rbc_ready_label(self.epoch, session_id, stage, self.my_id),
            require_vrf_proofs=(self.activation_mode == "vrf"),
            committee_selection_mode=self.committee_selection_mode,
            fixed_committee_seed=self.fixed_committee_seed,
        )
        self.rbc_instance_count += 1
        self.activation_send_count += 1
        await rbc.sender(_canonical_json(payload))
        trace["sent"] = True
        self.stage_collect_counts.append(
            {
                "stage": stage,
                "status": "send-complete",
                "activation_mode": self.activation_mode,
                "local_selected": True,
                "collected": 0,
                "need": 0,
                "send_only": True,
                "sent": True,
            }
        )
        return trace

    async def _delayed_send(self, rbc: YosoRBC, payload: bytes) -> None:
        await asyncio.sleep(0.1)
        await rbc.sender(payload)

    def _activation_proof(self, session_id: str, stage: str) -> Tuple[bool, Optional[dict]]:
        if self.activation_mode == "all":
            return True, None
        label = acs_activation_label(self.epoch, session_id, stage)
        if self.committee_selection_mode == FIXED_SIZE_COMMITTEE_MODE:
            roles = fixed_size_committee(
                label=label,
                population_size=self.M,
                committee_size=self.expected_committee_size,
                seed=self.fixed_committee_seed,
            )
            if self.my_id not in roles:
                return False, None
            return vrf_prove_sortition(
                label=label,
                vrf_sk=self.vrf_sk,
                g_hex=self.g_hex,
                threshold_fraction=1.0,
            )
        return vrf_prove_sortition(
            label=label,
            vrf_sk=self.vrf_sk,
            g_hex=self.g_hex,
            threshold_fraction=self.threshold_fraction,
        )

    def _verify_activation_proof(
        self,
        session_id: str,
        stage: str,
        source: int,
        proof: Optional[dict],
    ) -> bool:
        if self.activation_mode == "all":
            return True
        label = acs_activation_label(self.epoch, session_id, stage)
        if self.committee_selection_mode == FIXED_SIZE_COMMITTEE_MODE:
            roles = fixed_size_committee(
                label=label,
                population_size=self.M,
                committee_size=self.expected_committee_size,
                seed=self.fixed_committee_seed,
            )
            if source not in roles:
                return False
        return vrf_verify_sortition(
            label=label,
            proof=proof,
            sender_id=source,
            vrf_pks=self.vrf_pks,
            g_hex=self.g_hex,
            threshold_fraction=(
                1.0
                if self.committee_selection_mode == FIXED_SIZE_COMMITTEE_MODE
                else self.threshold_fraction
            ),
        )

    def _leader_from_coin(
        self,
        U_i: Dict[int, CoreBlock],
        coin: bytes,
        selection_index: int = 1,
    ) -> int:
        del selection_index
        return _candidate_leader(U_i, coin, population_size=self.M)

    def _weak_output_from_counts(
        self,
        fallback: CoreBlock,
        U_i: Dict[int, CoreBlock],
        T_i: Dict[int, CoreBlock],
        true_counts: Dict[int, int],
        quorum: int,
    ) -> Tuple[CoreBlock, int, Optional[int], int]:
        for leader, count in sorted(true_counts.items()):
            if count >= quorum and leader in T_i:
                return T_i[leader], 2, leader, count
        for leader, count in sorted(true_counts.items()):
            if count > 0 and leader in U_i and leader not in T_i:
                return U_i[leader], 1, leader, count
        return fallback, 0, None, 0

    def _upgrade2_value(
        self,
        blocks: List[CoreBlock],
    ) -> Tuple[Optional[CoreBlock], int]:
        counts: Dict[CoreBlock, int] = {}
        for block in blocks:
            counts[block] = counts.get(block, 0) + 1
        for block, count in sorted(counts.items(), key=lambda item: item[0].digest()):
            if count >= self.protocol_quorum:
                return block, 1
        return None, 0

    def _strongly_stable_output(
        self,
        fallback: CoreBlock,
        values: List[Tuple[Optional[CoreBlock], int]],
    ) -> Tuple[CoreBlock, int]:
        non_bottom = [block for block, h in values if h == 1 and block is not None]
        if len(non_bottom) == len(values) and len(set(non_bottom)) == 1:
            return non_bottom[0], 2
        if non_bottom:
            return sorted(set(non_bottom), key=lambda block: block.digest())[0], 1
        return fallback, 0

    def _union_core_maps(
        self,
        maps: Iterable[Dict[int, CoreBlock]],
    ) -> Dict[int, CoreBlock]:
        out: Dict[int, CoreBlock] = {}
        for mapping in maps:
            out.update(mapping)
        return out

    def _intersection_core_maps(
        self,
        maps: Iterable[Dict[int, CoreBlock]],
    ) -> Dict[int, CoreBlock]:
        maps = list(maps)
        if not maps:
            return {}
        common = set(maps[0])
        for mapping in maps[1:]:
            common &= set(mapping)
        return {party: maps[0][party] for party in sorted(common)}
