"""Phase 0 contracts for ACS core union and A²DPSS delivery boundaries."""

from __future__ import annotations

import asyncio
import itertools

import pytest

from beaver.a2dpss import (
    A2DPSSContribution,
    A2DPSSJustifierManager,
    A2DPSSPublicFinalizer,
)
from beaver.a2dpss_acs_adapter import (
    A2DPSSACSAdapter,
    MissingTranscriptReferences,
)
from beaver.async_yoso_acs import (
    CoreBlock,
    LocalAsyncYosoACS,
    ProposedBlock,
)
from beaver.async_yoso_tob import MessageRef, PendingEntry
from beaver.dag_tob import Qid


def _new_adapter(setup):
    return A2DPSSACSAdapter(setup["justifier"])


def _assert_same_qid(actual, expected: Qid) -> None:
    if isinstance(actual, str):
        actual = Qid.decode(actual)
    assert actual == expected


def test_core_reference_union_is_deduplicated_and_deterministic() -> None:
    ref_10 = MessageRef(10, 0)
    ref_11 = MessageRef(11, 0)
    ref_12 = MessageRef(12, 0)
    expected = [ref_10, ref_11, ref_12]

    for proposal_order in itertools.permutations(
        (
            ProposedBlock.build(0, (ref_11, ref_10)),
            ProposedBlock.build(1, (ref_12, ref_11)),
        )
    ):
        core = CoreBlock.build(proposal_order)
        assert core.flatten_refs() == expected


def test_payload_qid_is_resolved_from_registered_context_without_rewrapping(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    adapter = _new_adapter(setup)
    original_payload = setup["entries"][10][1]

    resolved = adapter.qid_for_payload(original_payload)

    _assert_same_qid(resolved, setup["context"].build_qid())
    assert setup["entries"][10][1] == original_payload


def test_adapter_registration_is_the_payload_qid_authority(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    adapter = A2DPSSACSAdapter(A2DPSSJustifierManager())
    payload = setup["entries"][10][1]

    with pytest.raises(KeyError, match="unknown A2DPSS context"):
        adapter.qid_for_payload(payload)

    adapter.register_context(setup["context"])
    assert adapter.qid_for_payload(payload) == setup["context"].build_qid()


def test_missing_reference_is_not_silently_ignored(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    adapter = _new_adapter(setup)
    present = setup["refs"][10]
    missing = MessageRef(99, 0)

    with pytest.raises(
        MissingTranscriptReferences,
        match="missing|unavailable",
    ) as exc_info:
        adapter.resolve_refs(
            [present, missing],
            {present: setup["pending"][present]},
        )
    assert exc_info.value.refs == (missing,)


def test_refs_are_grouped_by_qid_without_changing_payload_bytes(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    adapter = _new_adapter(setup)
    qid = setup["context"].build_qid()
    refs = [setup["refs"][11], setup["refs"][10]]

    groups = adapter.group_entries_by_qid(refs, setup["pending"])

    assert groups == {
        qid: [setup["entries"][10], setup["entries"][11]]
    }


def test_candidate_ref_validation_is_thresholded_and_fail_closed(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    adapter = _new_adapter(setup)
    refs = setup["refs"]

    assert adapter.validate_candidate_refs(
        [refs[10], refs[11]], setup["pending"]
    )
    assert not adapter.validate_candidate_refs(
        [refs[10]], setup["pending"]
    )
    assert not adapter.validate_candidate_refs(
        [refs[10], refs[10], refs[11]], setup["pending"]
    )

    wrong_mid_pending = dict(setup["pending"])
    wrong_mid_pending[refs[12]] = PendingEntry(
        ref=refs[12],
        mid="node_99",
        payload=setup["entries"][12][1],
        received_at=0.0,
    )
    assert not adapter.validate_candidate_refs(
        [refs[10], refs[11], refs[12]],
        wrong_mid_pending,
    )


def test_candidate_refs_are_built_when_predicate_becomes_true(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    adapter = _new_adapter(setup)
    refs = setup["refs"]
    pending_one = {refs[10]: setup["pending"][refs[10]]}
    pending_two = {
        refs[10]: setup["pending"][refs[10]],
        refs[11]: setup["pending"][refs[11]],
    }

    assert adapter.build_candidate_refs(pending_one) == []
    assert adapter.build_candidate_refs(pending_two) == [
        refs[10],
        refs[11],
    ]
    assert adapter.build_candidate_refs(
        pending_two,
        delivered_qids=[setup["context"].encode_qid()],
    ) == []


def test_structural_candidate_readiness_does_not_run_crypto_predicate(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    adapter = _new_adapter(setup)
    refs = setup["refs"]
    pending = {
        refs[10]: setup["pending"][refs[10]],
        refs[12]: PendingEntry(
            ref=refs[12],
            mid="node_12",
            payload=setup["wrong_anchor_entry"][1],
            received_at=0.0,
        ),
    }
    before = adapter.justifier.validation_metrics_snapshot()

    # An invalid anchor is intentionally invisible to the scheduling check;
    # receiver-side ACSPROP admission is the sole formal predicate boundary.
    assert adapter.build_structurally_ready_candidate_refs(pending) == [
        refs[10],
        refs[12],
    ]
    after = adapter.justifier.validation_metrics_snapshot()
    for stage in ("schema_identity", "pvss", "anchor"):
        assert after[stage]["invocations"] == before[stage]["invocations"]


def test_overlapping_candidate_blocks_produce_common_union_and_exact_T(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    adapter = _new_adapter(setup)
    refs = setup["refs"]
    qid = setup["context"].build_qid()
    expected_s_out = [
        setup["entries"][10],
        setup["entries"][11],
        setup["entries"][12],
    ]
    expected_T = expected_s_out[:2]

    for proposal_order in itertools.permutations(
        (
            ProposedBlock.build(0, (refs[11], refs[10])),
            ProposedBlock.build(1, (refs[12], refs[11])),
        )
    ):
        core = CoreBlock.build(proposal_order)
        union_refs = adapter.union_core_refs(core)
        s_out = adapter.resolve_refs(union_refs, setup["pending"])
        selected = adapter.select_delivery_T(qid, s_out)

        assert s_out == expected_s_out
        assert selected == expected_T
        assert len(selected) == setup["context"].service_t + 1

    invalid_s_out = expected_s_out + [setup["wrong_anchor_entry"]]
    assert adapter.select_delivery_T(qid, invalid_s_out) is None


def test_public_finalizer_accepts_only_selected_T_not_acs_superset(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    qid = setup["context"].build_qid()
    s_out = [
        setup["entries"][10],
        setup["entries"][11],
        setup["entries"][12],
    ]
    selected = setup["justifier"].select_justified_subset(qid, s_out)
    finalizer = A2DPSSPublicFinalizer(setup["justifier"])

    assert selected is not None
    assert finalizer.on_deliver(qid, s_out) is None
    assert finalizer.on_deliver(qid, selected) is not None


def test_local_acs_observers_derive_same_union_T_and_public_result(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    adapter = _new_adapter(setup)
    refs = setup["refs"]
    candidates = {
        0: [refs[11], refs[10]],
        1: [refs[12], refs[11]],
        2: [refs[12], refs[10]],
        3: [refs[12], refs[11], refs[10]],
    }
    acs = LocalAsyncYosoACS(M=4, t=1, epoch="a2dpss-phase3")

    outputs = asyncio.run(
        acs.run(
            candidates,
            proposal_validator=lambda _proposer, proposal_refs: (
                adapter.validate_candidate_refs(
                    proposal_refs, setup["pending"]
                )
            ),
        )
    )

    assert len(outputs) == 4
    assert len({output.selected_core.digest() for output in outputs}) == 1
    expected_refs = [refs[10], refs[11], refs[12]]
    qid = setup["context"].build_qid()
    deliveries = []
    for output in outputs:
        union_refs = adapter.union_core_refs(output.selected_core)
        assert union_refs == expected_refs
        assert output.block_refs == expected_refs
        s_out = adapter.resolve_refs(union_refs, setup["pending"])
        selected = adapter.select_delivery_T(qid, s_out)
        assert selected == [setup["entries"][10], setup["entries"][11]]
        deliveries.append(
            A2DPSSPublicFinalizer(setup["justifier"]).on_deliver(
                qid, selected
            )
        )

    assert deliveries[0] is not None
    assert all(result == deliveries[0] for result in deliveries)


def test_local_acs_rejects_proposal_with_valid_and_invalid_qid_segments(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    adapter = _new_adapter(setup)
    refs = setup["refs"]
    unknown_ref = MessageRef(99, 0)
    source = setup["contributions"][12]
    unknown_context_contribution = A2DPSSContribution(
        holder_id=source.holder_id,
        holder_eval_point=source.holder_eval_point,
        context_digest="00" * 32,
        transcript=source.transcript,
    )
    pending = dict(setup["pending"])
    pending[unknown_ref] = PendingEntry(
        ref=unknown_ref,
        mid="node_12",
        payload=unknown_context_contribution.to_bytes(),
        received_at=0.0,
    )
    candidates = {
        0: [refs[10], refs[11]],
        1: [refs[11], refs[12]],
        2: [refs[10], refs[12]],
        3: [refs[10], refs[11], unknown_ref],
    }
    checked = []

    def validate(proposer, proposal_refs):
        accepted = adapter.validate_candidate_refs(
            proposal_refs, pending
        )
        checked.append((proposer, accepted))
        return accepted

    outputs = asyncio.run(
        LocalAsyncYosoACS(
            M=4, t=1, epoch="a2dpss-phase3-reject"
        ).run(candidates, proposal_validator=validate)
    )

    assert checked == [(0, True), (1, True), (2, True), (3, False)]
    for output in outputs:
        assert [
            proposal.proposer
            for proposal in output.selected_core.proposals
        ] == [0, 1, 2]
        assert output.trace[0]["accepted_proposers"] == [0, 1, 2]
        assert unknown_ref not in output.block_refs


def test_local_acs_fails_before_consensus_without_valid_proposal_quorum(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    adapter = _new_adapter(setup)
    refs = setup["refs"]
    candidates = {
        0: [refs[10], refs[11]],
        1: [refs[11], refs[12]],
        2: [refs[10]],
        3: [],
    }

    with pytest.raises(
        ValueError, match="not enough proposed blocks for ACS quorum"
    ):
        asyncio.run(
            LocalAsyncYosoACS(
                M=4, t=1, epoch="a2dpss-phase3-no-quorum"
            ).run(
                candidates,
                proposal_validator=lambda _proposer, proposal_refs: (
                    adapter.validate_candidate_refs(
                        proposal_refs, setup["pending"]
                    )
                ),
            )
        )
