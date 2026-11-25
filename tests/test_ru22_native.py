"""Native RU22 adversarial contracts and two real-crypto refreshes."""
import itertools
from dataclasses import replace
from pathlib import Path

import pytest
from Crypto.PublicKey import ECC

from beaver.baselines.ru22_certificate import canonical, verify_qvc
from beaver.baselines.ru22_dpss import (
    RU22ContributionBatch, RU22InitialCertificate, RU22VPS,
    make_ru22_contribution_batch,
)
from beaver.baselines.ru22_experiment import build_single_vps_fixture
from beaver.baselines.ru22_native import (
    NativeAuthority, NativeParentRegistry, NativeLeader, NativeVerifier,
    NativeSuccessor, NativeCombine, verify_combine,
)
from beaver.chunk_elgamal import derive_public_key
from beaver.pvss import pvss_interpolate


ROOT = Path(__file__).resolve().parents[1]


def authority(c):
    keys = {"leader": {i: ECC.generate(curve="Ed25519") for i in range(c.resharing_n)},
            "verifier": {leader: {i: ECC.generate(curve="Ed25519") for i in range(c.resharing_n)}
                         for leader in range(c.resharing_n)}}
    public = lambda entries: tuple((i, k.public_key().export_key(format="raw").hex())
                                   for i, k in entries.items())
    return NativeAuthority(c.to_bytes(), public(keys["leader"]),
        tuple((leader, public(entries)) for leader, entries in keys["verifier"].items())), keys


@pytest.fixture(scope="module")
def fixture():
    f = build_single_vps_fixture(config_path=ROOT / "conf/admpc_100_4_4/local.0.json",
        trial_id="native-test", trial_epoch="native-test", seed="native-test",
        n_r=3, t_r=1, dag_m=4, dag_t=1, maintained_secret=9123, maintained_blind=4567)
    c = f.manifest.context
    a, keys = authority(c)
    batches = {}
    for i in range(3):
        s = f.shards[i].source
        batches[i] = make_ru22_contribution_batch(context=c,
            holder_role_id=s.holder_role_id, holder_eval_point=s.holder_eval_point,
            recipient_index=s.recipient_index, recipient_secret_key=s.recipient_secret_key,
            local_vps_shares=s.local_vps_shares)
    return f, a, keys, batches


def registry(f):
    return NativeParentRegistry(f.manifest.context, genesis=f.manifest.parent_vps_catalog)


def proposal(f, a, batches, leader, order):
    state = NativeLeader(a, registry(f), leader)
    out = None
    for i in order:
        value = state.receive(i, batches[i], f.manifest.parent_vps_catalog)
        out = value or out
    assert out is not None
    return out


def votes(f, a, keys, packages):
    result = []
    for i in range(3):
        row = []
        for p in packages:
            leader = p.to_obj()["leader"]
            v = NativeVerifier(a, registry(f), leader, i, keys["verifier"][leader][i].seed)
            row.append(v.receive(leader, p))
            assert v.sent and v.closed and not any(v._seed)
            assert v.receive(leader, p) is None
        result.append(row)
    return result


def successor(f, a, node):
    return NativeSuccessor(a, node, f.shards[node].successor_recipient_secret_key)


def test_independent_leaders_and_no_wait_for_silent_source(fixture):
    f, a, _, batches = fixture
    p = proposal(f, a, batches, 0, [0, 1])
    q = proposal(f, a, batches, 1, [1, 2])
    assert p.unsigned_vps(a.context).vps_id != q.unsigned_vps(a.context).vps_id
    verify_combine(a, registry(f), p)
    verify_combine(a, registry(f), q)


def test_crypto_subtimings_and_duplicate_vps_qvc_counts(fixture):
    f, a, keys, batches = fixture
    packages = []
    for leader_id in (0, 1):
        leader = NativeLeader(a, registry(f), leader_id)
        assert leader.receive(0, batches[0], f.manifest.parent_vps_catalog) is None
        packages.append(leader.receive(1, batches[1], f.manifest.parent_vps_catalog))
        assert set(leader.metrics.crypto) == {"leader_pvss_verification", "leader_dleq_verification"}
        assert not set(leader.metrics.crypto) & set(leader.metrics.snapshot())
        assert all(m["invocations"] == 2 for m in leader.metrics.crypto.values())
        assert sum(m["cpu_seconds"] for m in leader.metrics.crypto.values()) <= leader.metrics.snapshot()["leader_pvr_verification"]["cpu_seconds"]
    s = successor(f, a, 0)
    for leader_id, package in enumerate(packages):
        for signer in (0, 1):
            verifier = NativeVerifier(a, registry(f), leader_id, signer, keys["verifier"][leader_id][signer].seed)
            vote = verifier.receive(leader_id, package)
            assert all(m["invocations"] == 2 for m in verifier.metrics.crypto.values())
            assert set(verifier.metrics.crypto) == {"verifier_pvss_verification", "verifier_dleq_verification"}
            assert not set(verifier.metrics.crypto) & set(verifier.metrics.snapshot())
            s.receive(signer, vote)
    assert len(s.results) == 1  # identical selection, two independently certified leaders
    assert s.metrics.snapshot()["successor_qvc_verification"]["invocations"] == 2
    assert s.metrics.snapshot()["successor_reconstruction"]["invocations"] == 2
    assert not s.metrics.crypto  # successors trust QVC, without re-verifying PVSS/DLEQ


def test_combination_and_proof_tampering_rejected(fixture):
    f, a, _, batches = fixture
    p = proposal(f, a, batches, 0, [0, 1])
    obj = p.to_obj()
    obj["combined"][0] = obj["combined"][1]
    with pytest.raises(ValueError):
        verify_combine(a, registry(f), NativeCombine(canonical(obj)))
    obj = p.to_obj()
    obj["pvrs"] = [obj["pvrs"][0], obj["pvrs"][0]]
    with pytest.raises(ValueError):
        verify_combine(a, registry(f), NativeCombine(canonical(obj)))
    wrong = replace(a.context, sid="other-request")
    a2 = replace(a, context_bytes=wrong.to_bytes())
    with pytest.raises(ValueError):
        verify_combine(a2, NativeParentRegistry(wrong), p)


def test_qvc_requires_distinct_authenticated_signers_and_exact_package(fixture):
    f, a, keys, batches = fixture
    p = proposal(f, a, batches, 0, [0, 1])
    v = votes(f, a, keys, [p])
    s = successor(f, a, 0)
    assert s.receive(0, v[0][0]) is None
    assert s.receive(0, v[0][0]) is None
    with pytest.raises(ValueError):
        s.receive(1, v[0][0])
    bad = dict(v[1][0], signature="00" * 64)
    with pytest.raises(ValueError):
        s.receive(1, bad)
    changed = dict(v[1][0], package=proposal(f, a, batches, 0, [1, 2]).to_obj())
    with pytest.raises(ValueError):
        s.receive(1, changed)
    result = s.receive(1, v[1][0])
    assert result and a.verify(result.vps)
    assert RU22VPS.from_bytes(result.vps.to_bytes()) == result.vps
    other_cert = replace(result.vps.certificate, committee_digest="00" * 32)
    assert not a.verify(replace(result.vps, certificate=other_cert))


def test_multi_vps_output_then_two_hops_without_genesis_reinjection(fixture):
    f, a, keys, batches = fixture
    packages = [proposal(f, a, batches, 0, [0, 1]), proposal(f, a, batches, 1, [1, 2])]
    v = votes(f, a, keys, packages)
    successors = [successor(f, a, i) for i in range(3)]
    # Different first outputs are legal; they must continue processing.
    for i, s in enumerate(successors):
        order = [i % 2, 1 - i % 2]
        for k in order:
            for signer in (0, 1):  # third verifier is silent
                s.receive(signer, v[signer][k])
        assert len(s.results) == 2
    ids = set(successors[0].results)
    for vid in ids:
        for indices in itertools.combinations(range(3), 2):
            pairs = [(successors[i].results[vid].share_s,
                      successors[i].results[vid].share_s_hat) for i in indices]
            assert pvss_interpolate([i + 1 for i in indices], pairs, 1) == (9123, 4567)
    old = a.context
    new_rsks = [101, 103, 107]
    new = replace(old, source_epoch=old.target_epoch, target_epoch=old.target_epoch + 1,
        transfer_seq=old.transfer_seq + 1, source_committee=old.successor_committee,
        successor_committee="native-second-successors", source_key_base_hex=old.successor_key_base_hex,
        source_rpk_list=old.successor_rpk_list,
        successor_rpk_list=tuple(derive_public_key(k, old.successor_key_base_hex) for k in new_rsks))
    a2, keys2 = authority(new)
    next_batches, parent_map = {}, {}
    for i, s in enumerate(successors):
        holder = s.consume_to_holder()
        with pytest.raises(RuntimeError):
            s.consume_to_holder()
        holder.validate_public(new)
        role, point, index, rsk, local = holder.consume()
        parent_map.update({entry.vps.vps_id: entry.vps for entry in local})
        next_batches[i] = make_ru22_contribution_batch(context=new,
            holder_role_id=role, holder_eval_point=point, recipient_index=index,
            recipient_secret_key=rsk, local_vps_shares=local)
    reg2 = lambda: NativeParentRegistry(new, authorities=[a])
    leader = NativeLeader(a2, reg2(), 2)
    assert leader.receive(0, next_batches[0], list(parent_map.values())) is None
    p2 = leader.receive(2, next_batches[2], list(parent_map.values()))
    assert p2
    second_votes = [NativeVerifier(a2, reg2(), 2, i, keys2["verifier"][2][i].seed).receive(2, p2) for i in (0, 2)]
    out = []
    for i in range(3):
        s = NativeSuccessor(a2, i, new_rsks[i])
        assert s.receive(0, second_votes[0]) is None
        out.append(s.receive(2, second_votes[1]))
    for indices in itertools.combinations(range(3), 2):
        assert pvss_interpolate([i + 1 for i in indices],
            [(out[i].share_s, out[i].share_s_hat) for i in indices], 1) == (9123, 4567)
    forged = replace(out[0].vps, certificate=RU22InitialCertificate("00" * 32))
    future = replace(new, source_epoch=new.target_epoch, target_epoch=new.target_epoch + 1,
                     source_committee=new.successor_committee)
    with pytest.raises(ValueError):
        NativeParentRegistry(future).add(forged)


def test_dedicated_committees_finish_independently_with_one_send_roles(fixture):
    f, a, keys, batches = fixture
    pa = proposal(f, a, batches, 0, [0, 1])
    pb = proposal(f, a, batches, 1, [1, 2])
    # Leader 2 and verifier 2 in both active committees remain silent.
    va = NativeVerifier(a, registry(f), 0, 0, keys["verifier"][0][0].seed)
    vb = NativeVerifier(a, registry(f), 1, 1, keys["verifier"][1][1].seed)
    first_a, first_b = va.receive(0, pa), vb.receive(1, pb)
    successors = [successor(f, a, i) for i in range(3)]
    for s in successors:
        assert s.receive(0, first_a) is None
        assert s.receive(1, first_b) is None
        assert not s.results
    # The late signatures come from DIFFERENT one-send roles, with their own
    # keys, even though the physical machines are reused across committees.
    late_a = NativeVerifier(a, registry(f), 0, 1, keys["verifier"][0][1].seed).receive(0, pa)
    late_b = NativeVerifier(a, registry(f), 1, 0, keys["verifier"][1][0].seed).receive(1, pb)
    for s in successors:
        assert s.receive(1, late_a) is not None
        assert s.receive(0, late_b) is not None
        assert len(s.results) == 2
    with pytest.raises(ValueError, match="leader authentication"):
        va.receive(1, pb)
    with pytest.raises(ValueError, match="signing key"):
        NativeVerifier(a, registry(f), 1, 0, keys["verifier"][0][0].seed)
    # A valid signature from the same machine's OTHER committee is insufficient.
    from beaver.baselines.ru22_certificate import sign_statement
    rogue = dict(first_b, verifier=0,
        signature=sign_statement(keys["verifier"][0][0].seed, pb.statement(a)))
    with pytest.raises(ValueError, match="signature"):
        successor(f, a, 0).receive(0, rogue)


def test_authority_requires_n_distinct_committees_and_role_keys(fixture):
    _, a, _, _ = fixture
    with pytest.raises(ValueError, match="each leader"):
        replace(a, verifier_committees=a.verifier_committees[:-1])
    with pytest.raises(ValueError, match="distinct native roles"):
        replace(a, verifier_committees=tuple((i, a.verifier_committees[0][1])
                                           for i, _ in a.verifier_committees))


def test_native_threshold_operation_uses_same_certified_pvrs(fixture):
    from beaver.baselines.ru22_operation import make_ru22_operation_contribution_batch
    from beaver.baselines.ru22_native_operation import (
        NativeOperationLeader, NativeOperationVerifier, NativeOperationSuccessor,
        NativeOperationPackage, verify_operation_package, verify_operation_pvr,
    )
    from beaver.g1_backend import G1Point
    from beaver.threshold_operation_contract import (
        ThresholdOperationContext, OPERATION_ID, OPERATION_VK_BASE_HEX,
        derive_operation_base, operation_output_digest,
    )
    f, a, keys, _ = fixture
    message = b"native-operation-test"
    expected = G1Point.from_hex(derive_operation_base(message)) * 9123
    op = ThresholdOperationContext(operation_input=message,
        maintained_global_public_key=(G1Point.from_hex(OPERATION_VK_BASE_HEX) * 9123).to_hex(),
        expected_operation_output_digest=operation_output_digest(operation_id=OPERATION_ID,
            operation_input=message, operation_output=expected), coin_public_key_digest="ab" * 32)
    batches = {}
    for i in range(3):
        s = f.shards[i].source
        batches[i] = make_ru22_operation_contribution_batch(context=a.context,
            operation_context=op, trial_id="native-op", holder_role_id=s.holder_role_id,
            holder_eval_point=s.holder_eval_point, recipient_index=s.recipient_index,
            recipient_secret_key=s.recipient_secret_key, local_vps_shares=s.local_vps_shares)
    packages = []
    for leader, order in ((0, (0, 1)), (1, (1, 2))):
        state = NativeOperationLeader(a, registry(f), leader, operation_context=op, trial_id="native-op")
        assert state.receive(order[0], batches[order[0]], f.manifest.parent_vps_catalog) is None
        package = state.receive(order[1], batches[order[1]], f.manifest.parent_vps_catalog)
        assert isinstance(package, NativeOperationPackage)
        verify_operation_package(a, registry(f), package, operation_context=op, trial_id="native-op")
        packages.append(package)
    successors = [NativeOperationSuccessor(a, i, f.shards[i].successor_recipient_secret_key,
        operation_context=op, trial_id="native-op") for i in range(3)]
    for leader, package in enumerate(packages):
        for signer in (0, 1):
            verifier = NativeOperationVerifier(a, registry(f), leader, signer,
                keys["verifier"][leader][signer].seed, operation_context=op, trial_id="native-op")
            vote = verifier.receive(leader, package)
            assert verifier.closed
            for successor in successors:
                successor.receive(signer, vote)
    for successor in successors:
        assert len(successor.results) == len(successor.operation_results) == 2
        assert {r["operation_output_digest"] for r in successor.operation_results.values()} == {op.expected_operation_output_digest}
    for vid in successors[0].results:
        assert pvss_interpolate([1, 2], [(s.results[vid].share_s, s.results[vid].share_s_hat)
                                       for s in successors[:2]], 1) == (9123, 4567)
    with pytest.raises(ValueError, match="context or trial"):
        verify_operation_package(a, registry(f), packages[0], operation_context=op, trial_id="replay")
    obj = packages[0].to_obj()
    obj["output"] = op.maintained_global_public_key
    with pytest.raises(ValueError, match="combination"):
        verify_operation_package(a, registry(f), NativeOperationPackage(canonical(obj)),
            operation_context=op, trial_id="native-op")
    original = packages[0].operation_pvrs[0]
    bad = replace(original, operation_share=op.maintained_global_public_key)
    with pytest.raises(ValueError, match="share proof"):
        verify_operation_pvr(a.context, op, "native-op", f.manifest.parent_vps_catalog[0], bad)
