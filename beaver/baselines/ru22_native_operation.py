"""Threshold operation coupled to dedicated-verifier RU22 refresh.

The operation and its two binding proofs travel with the selected original
PVRs. A QVC signs the entire package, including the operation result. No DAG
certificate, unique global leader, or additional sender phase is introduced.
"""
from contextlib import nullcontext

from beaver.baselines.ru22_certificate import canonical, strict_hex
from beaver.baselines.ru22_dpss import RU22ContributionBatch, RU22VPS, _decode_object
from beaver.baselines.ru22_native import (
    NativeCombine, NativeLeader, NativeVerifier, NativeSuccessor, verify_combine,
)
from beaver.baselines.ru22_operation import (
    RU22OperationContributionBatch, RU22OperationPVR, build_ru22_operation_qid,
    _link_statement, _operation_statement, verify_ru22_encrypted_commitment_vk,
)
from beaver.g1_backend import G1Point
from beaver.operation_vk import combine_vk_operation_shares, verify_vk_operation_share
from beaver.threshold_operation_contract import operation_output_digest


class NativeOperationPackage(NativeCombine):
    def __post_init__(self):
        obj = _decode_object(self.payload, name="native operation combine")
        if set(obj) != {"schema", "context", "leader", "base", "operation_context", "trial_id", "operation_entries", "output"}:
            raise ValueError("invalid native operation package fields")
        if obj["schema"] != "ru22-native-operation-combine/v1":
            raise ValueError("invalid native operation schema")
        base = NativeCombine(canonical(obj["base"]))
        if obj["context"] != base.to_obj()["context"] or obj["leader"] != base.to_obj()["leader"]:
            raise ValueError("operation/base package identity mismatch")
        strict_hex(obj["operation_context"], 32)
        if not isinstance(obj["trial_id"], str) or not obj["trial_id"]:
            raise ValueError("missing operation trial identity")
        G1Point.from_hex(obj["output"])
        if not isinstance(obj["operation_entries"], list) or len(obj["operation_entries"]) != len(base.reconstruction_pvrs()):
            raise ValueError("operation/base PVR count mismatch")
        for entry, pvr in zip(obj["operation_entries"], base.reconstruction_pvrs()):
            if not isinstance(entry, dict) or "base_pvr" in entry:
                raise ValueError("operation package repeats a base PVR")
            RU22OperationPVR.from_obj(dict(entry, base_pvr=pvr))
        object.__setattr__(self, "payload", canonical(obj))

    @property
    def base(self):
        return NativeCombine(canonical(self.to_obj()["base"]))

    def reconstruction_pvrs(self):
        return self.base.reconstruction_pvrs()

    @property
    def operation_pvrs(self):
        return tuple(RU22OperationPVR.from_obj(dict(entry, base_pvr=pvr))
            for entry, pvr in zip(self.to_obj()["operation_entries"], self.reconstruction_pvrs()))

    def unsigned_vps(self, context):
        return self.base.unsigned_vps(context)


def verify_operation_pvr(context, operation_context, trial_id, parent, pvr, *, metrics=None, prefix=""):
    qid = build_ru22_operation_qid(context, operation_context, trial_id=trial_id).encode()
    base = pvr.base_pvr
    if base.parent_vps_id != parent.vps_id:
        raise ValueError("operation PVR has wrong parent")
    statement = _link_statement(context=context, operation_context=operation_context,
        qid=qid, parent=parent, base_pvr=base,
        recipient_index=context.source_role_indices[base.dealer_role_id],
        operation_vk=pvr.operation_vk)
    with metrics.measure(prefix + "link_proof_verification") if metrics else nullcontext():
        if not verify_ru22_encrypted_commitment_vk(statement, pvr.encrypted_commitment_vk_proof):
            raise ValueError("invalid operation encrypted-commitment link proof")
    statement = _operation_statement(context=context, operation_context=operation_context,
        trial_id=trial_id, qid=qid, operation_pvr=pvr)
    with metrics.measure(prefix + "operation_proof_verification") if metrics else nullcontext():
        if not verify_vk_operation_share(statement, pvr.vk_operation_share_proof):
            raise ValueError("invalid operation share proof")


def combine_operation(operation_context, pvrs):
    points = [p.base_pvr.dealer_eval_point for p in pvrs]
    vk = combine_vk_operation_shares(points, [p.operation_vk for p in pvrs])
    output = combine_vk_operation_shares(points, [p.operation_share for p in pvrs])
    if vk != G1Point.from_hex(operation_context.maintained_global_public_key):
        raise ValueError("operation shares do not match the maintained application key")
    result_digest = operation_output_digest(operation_id=operation_context.operation_id,
        operation_input=operation_context.operation_input, operation_output=output)
    if result_digest != operation_context.expected_operation_output_digest:
        raise ValueError("unexpected native threshold operation output")
    return output.to_hex(), result_digest


def make_operation_package(base, operation_context, trial_id, pvrs):
    by_digest = {p.base_pvr.pvr_digest: p for p in pvrs}
    selected = tuple(by_digest[p["pvr_digest"]] for p in base.reconstruction_pvrs())
    output, _ = combine_operation(operation_context, selected)
    entries = []
    for p in selected:
        entry = p.to_obj()
        del entry["base_pvr"]
        entries.append(entry)
    return NativeOperationPackage(canonical(dict(schema="ru22-native-operation-combine/v1",
        context=base.to_obj()["context"], leader=base.to_obj()["leader"], base=base.to_obj(),
        operation_context=operation_context.context_digest, trial_id=trial_id,
        operation_entries=entries, output=output)))


def verify_operation_package(authority, registry, package, *, operation_context, trial_id, metrics=None):
    if (not isinstance(package, NativeOperationPackage)
            or package.to_obj()["operation_context"] != operation_context.context_digest
            or package.to_obj()["trial_id"] != trial_id):
        raise ValueError("wrong native operation context or trial")
    with metrics.measure("verifier_combine_verification") if metrics else nullcontext():
        verify_combine(authority, registry, package.base,
                       record_stage=metrics.crypto_recorder("verifier") if metrics else lambda *_: None)
    parent = registry.add(RU22VPS.from_obj(package.base.to_obj()["parent"]))
    for p in package.operation_pvrs:
        verify_operation_pvr(authority.context, operation_context, trial_id, parent, p,
                             metrics=metrics, prefix="verifier_")
    with metrics.measure("verifier_operation_combination") if metrics else nullcontext():
        expected = make_operation_package(package.base, operation_context, trial_id, package.operation_pvrs)
    if expected.payload != package.payload:
        raise ValueError("incorrect native operation combination")


class NativeOperationLeader(NativeLeader):
    def __init__(self, authority, registry, leader_id, *, operation_context, trial_id):
        super().__init__(authority, registry, leader_id)
        self.operation_context, self.trial_id = operation_context, trial_id
        self.operation_pvrs = {}

    def receive(self, sender, batch, parents):
        if self.output is not None:
            return None
        c = self.authority.context
        batch = RU22OperationContributionBatch.from_bytes(batch.to_bytes())
        expected_qid = build_ru22_operation_qid(c, self.operation_context, trial_id=self.trial_id).encode()
        if batch.qid != expected_qid or batch.holder_role_id != sender:
            raise ValueError("operation source/context mismatch")
        catalog = {p.vps_id: p for p in parents}
        valid = []
        for p in batch.pvr_entries:
            try:
                parent = self.registry.add(catalog[p.base_pvr.parent_vps_id])
                verify_operation_pvr(c, self.operation_context, self.trial_id, parent, p,
                                     metrics=self.metrics, prefix="leader_")
            except (KeyError, TypeError, ValueError):
                self.rejected_pvrs += 1
                continue
            valid.append(p.base_pvr)
            self.operation_pvrs[p.base_pvr.pvr_digest] = p
        if not valid:
            return None
        base_batch = RU22ContributionBatch(qid=c.encode_qid(), context_digest=c.context_digest,
            holder_role_id=sender, holder_eval_point=batch.holder_eval_point, pvr_entries=valid)
        base = super().receive(sender, base_batch, parents)
        if base is not None:
            with self.metrics.measure("leader_operation_combination"):
                self.output = make_operation_package(base, self.operation_context, self.trial_id,
                    tuple(self.operation_pvrs.values()))
        return self.output


class NativeOperationVerifier(NativeVerifier):
    def __init__(self, *args, operation_context, trial_id):
        super().__init__(*args)
        self.operation_context, self.trial_id = operation_context, trial_id

    def verify(self, package):
        verify_operation_package(self.authority, self.registry, package,
            operation_context=self.operation_context, trial_id=self.trial_id, metrics=self.metrics)


class NativeOperationSuccessor(NativeSuccessor):
    def __init__(self, *args, operation_context, trial_id):
        super().__init__(*args, package_type=NativeOperationPackage)
        self.operation_context, self.trial_id = operation_context, trial_id
        self.operation_results = {}

    def receive(self, sender, vote):
        package = NativeOperationPackage(canonical(vote["package"]))
        obj = package.to_obj()
        if obj["operation_context"] != self.operation_context.context_digest or obj["trial_id"] != self.trial_id:
            raise ValueError("wrong operation successor context")
        result = super().receive(sender, vote)
        if result:
            # QVC already attests the exact operation entries. Check the public
            # output against the fixed request without generating more speech.
            with self.metrics.measure("successor_operation_combination"):
                output, out_digest = combine_operation(self.operation_context, package.operation_pvrs)
                if output != obj["output"]:
                    raise ValueError("certified operation output mismatch")
            self.operation_results.setdefault(result.vps.vps_id, dict(operation_output=output,
                operation_output_digest=out_digest, operation_context_digest=self.operation_context.context_digest,
                vps_id=result.vps.vps_id, reconstruction_digest=package.package_digest))
        return result
