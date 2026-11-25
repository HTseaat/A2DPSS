"""Phase 6 contracts for common-core A²DPSS delivery and reconstruction."""

from __future__ import annotations

import pytest

from beaver.a2dpss import (
    A2DPSSPublicFinalizer,
    A2DPSSSuccessorReconstructionManager,
)
from beaver.a2dpss_acs_adapter import (
    A2DPSSACSAdapter,
    MissingTranscriptReferences,
)
from beaver.async_yoso_acs import CoreBlock, ProposedBlock
from beaver.async_yoso_tob import AsyncYosoTOB, MessageRef, PendingEntry


def _overlapping_core(refs):
    return CoreBlock.build(
        (
            ProposedBlock.build(0, (refs[10], refs[11])),
            ProposedBlock.build(1, (refs[11], refs[12])),
        )
    )


def test_common_core_union_finalizes_exact_T_and_reconstructs_successor(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    adapter = A2DPSSACSAdapter(setup["justifier"])
    plan = adapter.prepare_single_qid_delivery(
        _overlapping_core(setup["refs"]),
        setup["pending"],
    )

    assert list(plan.refs) == [
        setup["refs"][10],
        setup["refs"][11],
        setup["refs"][12],
    ]
    assert list(plan.s_out) == [
        setup["entries"][10],
        setup["entries"][11],
        setup["entries"][12],
    ]
    assert list(plan.selected_T) == [
        setup["entries"][10],
        setup["entries"][11],
    ]

    public_result = A2DPSSPublicFinalizer(
        setup["justifier"]
    ).on_deliver(plan.qid, plan.selected_T)
    assert public_result is not None
    assert len(public_result.accepted_contribution_digests) == 2

    reconstruction = A2DPSSSuccessorReconstructionManager()
    reconstruction.register_context(
        setup["context"],
        my_node_id=20,
        my_rsk=17,
    )
    private_result = reconstruction.reconstruct(
        public_result,
        plan.selected_T,
    )
    assert private_result is not None
    assert (
        private_result.accepted_contribution_digests
        == public_result.accepted_contribution_digests
    )

    tob = AsyncYosoTOB(my_id=0, M=13, epoch="phase6-delivery")
    tob.pending.update(setup["pending"])
    ledger_index, block = tob.append_acs_block(list(plan.refs))
    assert ledger_index == 1
    assert block == [entry[1] for entry in plan.s_out]
    assert tob.snapshot_ledger_refs() == [
        [ref.encode() for ref in plan.refs]
    ]
    with pytest.raises(RuntimeError, match="missing refs|no new messages"):
        tob.append_acs_block(list(plan.refs))


def test_verified_core_delivery_reuses_admission_without_crypto_recheck(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    adapter = A2DPSSACSAdapter(setup["justifier"])
    core = _overlapping_core(setup["refs"])
    session_id = "phase6-verified-core"
    before_admission = setup["justifier"].validation_metrics_snapshot()

    with pytest.raises(ValueError, match="without matching local validation"):
        adapter.prepare_verified_single_qid_delivery(
            core,
            setup["pending"],
            session_id,
        )

    # These are two distinct proposer blocks, so both formal predicate calls
    # execute even though they overlap.  Their exact successful admissions are
    # then reusable at the delivery boundary.
    for proposal in core.proposals:
        assert adapter.validate_candidate_refs(
            proposal.refs,
            setup["pending"],
        )
        adapter.record_verified_candidate(
            session_id,
            proposal.proposer,
            proposal.refs,
            setup["pending"],
        )
    after_admission = setup["justifier"].validation_metrics_snapshot()

    plan = adapter.prepare_verified_single_qid_delivery(
        core,
        setup["pending"],
        session_id,
    )
    after_delivery = setup["justifier"].validation_metrics_snapshot()

    assert list(plan.selected_T) == [
        setup["entries"][10],
        setup["entries"][11],
    ]
    for stage in ("schema_identity", "pvss", "anchor"):
        assert (
            after_admission[stage]["invocations"]
            - before_admission[stage]["invocations"]
        ) == 4
        assert after_delivery[stage] == after_admission[stage]


def test_verified_core_evidence_is_bound_to_pending_payload_bytes(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    adapter = A2DPSSACSAdapter(setup["justifier"])
    core = _overlapping_core(setup["refs"])
    session_id = "phase6-payload-binding"
    for proposal in core.proposals:
        assert adapter.validate_candidate_refs(
            proposal.refs,
            setup["pending"],
        )
        adapter.record_verified_candidate(
            session_id,
            proposal.proposer,
            proposal.refs,
            setup["pending"],
        )

    pending = dict(setup["pending"])
    ref = setup["refs"][12]
    pending[ref] = PendingEntry(
        ref=ref,
        mid="node_12",
        payload=setup["wrong_anchor_entry"][1],
        received_at=0.0,
    )
    with pytest.raises(ValueError, match="without matching local validation"):
        adapter.prepare_verified_single_qid_delivery(
            core,
            pending,
            session_id,
        )


def test_phase6_delivery_waits_for_every_core_reference(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    adapter = A2DPSSACSAdapter(setup["justifier"])
    pending = dict(setup["pending"])
    missing = setup["refs"][12]
    pending.pop(missing)

    with pytest.raises(MissingTranscriptReferences) as exc_info:
        adapter.prepare_single_qid_delivery(
            _overlapping_core(setup["refs"]),
            pending,
        )
    assert exc_info.value.refs == (missing,)


def test_phase6_rejects_invalid_transcript_in_core_union(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    adapter = A2DPSSACSAdapter(setup["justifier"])
    pending = dict(setup["pending"])
    ref_12 = setup["refs"][12]
    pending[ref_12] = PendingEntry(
        ref=ref_12,
        mid="node_12",
        payload=setup["wrong_anchor_entry"][1],
        received_at=0.0,
    )

    with pytest.raises(ValueError, match="does not satisfy"):
        adapter.prepare_single_qid_delivery(
            _overlapping_core(setup["refs"]),
            pending,
        )


def test_phase6_requires_exactly_one_qid() -> None:
    adapter = object.__new__(A2DPSSACSAdapter)
    ref = MessageRef(0, 0)
    core = CoreBlock.build((ProposedBlock.build(0, (ref,)),))
    adapter.union_core_refs = lambda _core: [ref]
    adapter.group_entries_by_qid = lambda _refs, _pending: {}

    with pytest.raises(ValueError, match="exactly one qid"):
        adapter.prepare_single_qid_delivery(core, {})
