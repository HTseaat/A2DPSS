"""RU22 DAG-compiled DPSS baseline through Phase-5 local handoff.

This module owns the canonical public context, VPS/PVR/batch wire objects,
the old role's multi-VPS private state, its one-shot local ``speak`` action,
and the pure per-query DAG Justifier.  It deliberately does not issue a DAG
delivery certificate, decrypt combined PVRs, or persist successor shares;
those are later phases.

For every locally held VPS an old role stores the field pair opening its own
encrypted commitment.  The action first checks that opening, then creates a
fresh chunk-ElGamal PVSS transcript for the successor committee and the RU22
old-to-new Chaum--Pedersen proof.  All PVRs are sent in one batch under
``mid = node_<holder_role_id>``.
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import re
import threading
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Union

from beaver.baselines.ru22_certificate import RU22QuorumCertificate

from beaver.baselines.ru22_dleq import (
    RU22OldToNewDLEQProof,
    prove_old_to_new_dleq,
    verify_old_to_new_dleq,
)
from beaver.chunk_elgamal import (
    decrypt_elgamal_ciphertext,
    derive_public_key,
    normalize_elgamal_ciphertext,
)
from beaver.dag_tob import DeliveryEvent, Justifier, PoolEntry, Qid
from beaver.g1_backend import FIELD_ORDER, G1Point
from beaver.pedersen_shamir import (
    lagrange_coeff_at_zero,
    pedersen_commit,
)
from beaver.pvss import (
    TRANSCRIPT_VERSION,
    PVSSTranscript,
    interpolate_c_v_vector,
    pvss_deal,
    pvss_deliver,
    pvss_interpolate,
    pvss_verify,
)


CONTEXT_VERSION = 1
VPS_VERSION = 1
PVR_VERSION = 1
BATCH_VERSION = 1
DELIVERY_RESULT_VERSION = 1
SUCCESSOR_RESULT_VERSION = 1
QID_KIND = "ru22-dpss"

CONTEXT_KIND = "atcaas-ru22-context"
VPS_KIND = "atcaas-ru22-vps"
PVR_KIND = "atcaas-ru22-pvr"
BATCH_KIND = "atcaas-ru22-contribution-batch"
DELIVERY_RESULT_KIND = "atcaas-ru22-delivery-result"
SUCCESSOR_RESULT_KIND = "atcaas-ru22-successor-result"

CONTEXT_DIGEST_DOMAIN = b"ATCaaS/RU22/context/v1\x00"
VPS_ID_DOMAIN = b"ATCaaS/RU22/vps/v1\x00"
PVR_DIGEST_DOMAIN = b"ATCaaS/RU22/pvr/v1\x00"
BATCH_DIGEST_DOMAIN = b"ATCaaS/RU22/batch/v1\x00"
DELIVERY_RESULT_DIGEST_DOMAIN = b"ATCaaS/RU22/delivery/v1\x00"

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_QID_FIELDS = frozenset(
    {
        "context_digest",
        "epoch",
        "sid",
        "source_committee",
        "successor_committee",
        "transfer_seq",
    }
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _typed_canonical(value: Any) -> Any:
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ["bytes", bytes(value).hex()]
    if isinstance(value, Mapping):
        entries = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical mapping keys must be strings")
            entries.append([key, _typed_canonical(item)])
        entries.sort(key=lambda pair: pair[0])
        return ["map", entries]
    if isinstance(value, (list, tuple)):
        return ["list", [_typed_canonical(item) for item in value]]
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def _domain_digest(domain: bytes, material: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        domain + _canonical_json(_typed_canonical(material))
    ).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _decode_object(
    data: bytes | bytearray | memoryview | str,
    *,
    name: str,
) -> dict:
    raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed {name} JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must encode a JSON object")
    return decoded


def _exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    missing = expected - set(value)
    extra = set(value) - expected
    if missing:
        raise ValueError(f"{name} is missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"{name} has unknown fields: {sorted(extra)}")


def _strict_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _scalar(value: Any, *, name: str, nonzero: bool = False) -> int:
    scalar = _strict_int(value, name=name)
    if scalar >= FIELD_ORDER or (nonzero and scalar == 0):
        qualifier = "non-zero " if nonzero else ""
        raise ValueError(f"{name} must be a canonical {qualifier}field scalar")
    return scalar


def _canonical_decimal(value: Any, *, name: str) -> int:
    if not isinstance(value, str) or (
        value != "0"
        and (not value.isdigit() or value.startswith("0"))
    ):
        raise ValueError(f"{name} must be a canonical decimal string")
    return _scalar(int(value), name=name)


def _nonempty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return value


def _point(
    value: str | G1Point,
    *,
    name: str,
    nonidentity: bool = True,
) -> str:
    try:
        point = value if isinstance(value, G1Point) else G1Point.from_hex(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid G1 point") from exc
    if nonidentity and point.is_identity():
        raise ValueError(f"{name} must not be the identity point")
    return point.to_hex()


def _int_mapping(
    value: Mapping[Any, Any],
    *,
    name: str,
    value_minimum: int,
) -> Mapping[int, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized: dict[int, int] = {}
    for raw_key, raw_value in value.items():
        if isinstance(raw_key, bool):
            raise TypeError(f"{name} keys must be integer role ids")
        if isinstance(raw_key, int):
            key = raw_key
        elif isinstance(raw_key, str) and (
            raw_key == "0"
            or (raw_key.isdigit() and not raw_key.startswith("0"))
        ):
            key = int(raw_key)
        else:
            raise TypeError(f"{name} keys must be canonical integer role ids")
        if key < 0 or key in normalized:
            raise ValueError(f"{name} contains an invalid/duplicate role id")
        normalized[key] = _strict_int(
            raw_value,
            name=f"{name}[{key}]",
            minimum=value_minimum,
        )
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return MappingProxyType(dict(sorted(normalized.items())))


def _mapping_obj(value: Mapping[int, int]) -> dict[str, int]:
    return {str(key): item for key, item in value.items()}


def _canonical_qid(
    value: Qid | str,
    *,
    context_digest: str,
    source_epoch: int,
) -> str:
    if isinstance(value, Qid):
        qid = value
        encoded = qid.encode()
    elif isinstance(value, str):
        qid = Qid.decode(value)
        encoded = value
        if qid.encode() != encoded:
            raise ValueError("qid must use canonical Qid encoding")
    else:
        raise TypeError("qid must be a Qid or canonical encoded string")
    if qid.kind != QID_KIND:
        raise ValueError(f"qid kind must be {QID_KIND!r}")
    fields = dict(qid.fields)
    if set(fields) != _QID_FIELDS:
        raise ValueError("qid fields do not match the RU22 contract")
    if fields["context_digest"] != context_digest:
        raise ValueError("qid context_digest mismatch")
    if fields["epoch"] != str(source_epoch):
        raise ValueError("qid epoch mismatch")
    for field_name in ("sid", "source_committee", "successor_committee"):
        _nonempty_text(fields[field_name], name=f"qid {field_name}")
    transfer_seq = fields["transfer_seq"]
    if transfer_seq != "0" and (
        not transfer_seq.isdigit() or transfer_seq.startswith("0")
    ):
        raise ValueError("qid transfer_seq is not canonical")
    return encoded


@dataclass(frozen=True)
class RU22Context:
    source_epoch: int
    target_epoch: int
    sid: str
    source_committee: str
    successor_committee: str
    transfer_seq: int
    g_hex: str
    h_hex: str
    source_key_base_hex: str
    successor_key_base_hex: str
    source_rpk_list: Sequence[str]
    successor_rpk_list: Sequence[str]
    resharing_n: int
    resharing_threshold: int
    agreement_n: int
    agreement_threshold: int
    holder_eval_points: Mapping[int, int]
    source_role_indices: Mapping[int, int]
    successor_role_indices: Mapping[int, int]
    version: int = CONTEXT_VERSION
    context_digest: str = field(init=False)

    def __post_init__(self) -> None:
        version = _strict_int(self.version, name="version")
        if version != CONTEXT_VERSION:
            raise ValueError("unsupported RU22 context version")
        source_epoch = _strict_int(
            self.source_epoch, name="source_epoch", minimum=1
        )
        target_epoch = _strict_int(
            self.target_epoch, name="target_epoch", minimum=2
        )
        if target_epoch != source_epoch + 1:
            raise ValueError("target_epoch must equal source_epoch + 1")
        transfer_seq = _strict_int(self.transfer_seq, name="transfer_seq")
        resharing_n = _strict_int(
            self.resharing_n, name="resharing_n", minimum=1
        )
        resharing_threshold = _strict_int(
            self.resharing_threshold, name="resharing_threshold"
        )
        if resharing_n != 2 * resharing_threshold + 1:
            raise ValueError(
                "resharing_n must equal 2 * resharing_threshold + 1"
            )
        agreement_n = _strict_int(
            self.agreement_n, name="agreement_n", minimum=1
        )
        agreement_threshold = _strict_int(
            self.agreement_threshold, name="agreement_threshold"
        )
        if agreement_n < 3 * agreement_threshold + 1:
            raise ValueError(
                "agreement_n must be at least 3 * agreement_threshold + 1"
            )
        sid = _nonempty_text(self.sid, name="sid")
        source = _nonempty_text(
            self.source_committee, name="source_committee"
        )
        successor = _nonempty_text(
            self.successor_committee, name="successor_committee"
        )
        g_hex = _point(self.g_hex, name="g_hex")
        h_hex = _point(self.h_hex, name="h_hex")
        if g_hex == h_hex:
            raise ValueError("g_hex and h_hex must be distinct")
        source_key_base = _point(
            self.source_key_base_hex, name="source_key_base_hex"
        )
        successor_key_base = _point(
            self.successor_key_base_hex, name="successor_key_base_hex"
        )
        source_keys = tuple(
            _point(item, name="source_rpk_list item")
            for item in self.source_rpk_list
        )
        successor_keys = tuple(
            _point(item, name="successor_rpk_list item")
            for item in self.successor_rpk_list
        )
        for name, keys in (
            ("source_rpk_list", source_keys),
            ("successor_rpk_list", successor_keys),
        ):
            if len(keys) != resharing_n:
                raise ValueError(f"{name} must contain resharing_n keys")
            if len(set(keys)) != len(keys):
                raise ValueError(f"{name} entries must be distinct")

        holder_points = _int_mapping(
            self.holder_eval_points,
            name="holder_eval_points",
            value_minimum=1,
        )
        source_indices = _int_mapping(
            self.source_role_indices,
            name="source_role_indices",
            value_minimum=0,
        )
        successor_indices = _int_mapping(
            self.successor_role_indices,
            name="successor_role_indices",
            value_minimum=0,
        )
        role_ids = set(holder_points)
        if len(holder_points) != resharing_n:
            raise ValueError(
                "holder_eval_points must contain resharing_n roles"
            )
        if len(set(holder_points.values())) != resharing_n:
            raise ValueError("holder evaluation points must be distinct")
        if set(source_indices) != role_ids:
            raise ValueError(
                "source_role_indices must cover exactly the holder roles"
            )
        expected_indices = set(range(resharing_n))
        if set(source_indices.values()) != expected_indices:
            raise ValueError("source_role_indices must be a bijection")
        if len(successor_indices) != resharing_n or set(
            successor_indices.values()
        ) != expected_indices:
            raise ValueError("successor_role_indices must be a bijection")

        for name, value in (
            ("version", version),
            ("source_epoch", source_epoch),
            ("target_epoch", target_epoch),
            ("sid", sid),
            ("source_committee", source),
            ("successor_committee", successor),
            ("transfer_seq", transfer_seq),
            ("g_hex", g_hex),
            ("h_hex", h_hex),
            ("source_key_base_hex", source_key_base),
            ("successor_key_base_hex", successor_key_base),
            ("source_rpk_list", source_keys),
            ("successor_rpk_list", successor_keys),
            ("resharing_n", resharing_n),
            ("resharing_threshold", resharing_threshold),
            ("agreement_n", agreement_n),
            ("agreement_threshold", agreement_threshold),
            ("holder_eval_points", holder_points),
            ("source_role_indices", source_indices),
            ("successor_role_indices", successor_indices),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "context_digest",
            _domain_digest(CONTEXT_DIGEST_DOMAIN, self.digest_material()),
        )

    def digest_material(self) -> dict[str, Any]:
        return {
            "kind": CONTEXT_KIND,
            "version": self.version,
            "source_epoch": self.source_epoch,
            "target_epoch": self.target_epoch,
            "sid": self.sid,
            "source_committee": self.source_committee,
            "successor_committee": self.successor_committee,
            "transfer_seq": self.transfer_seq,
            "g_hex": self.g_hex,
            "h_hex": self.h_hex,
            "source_key_base_hex": self.source_key_base_hex,
            "successor_key_base_hex": self.successor_key_base_hex,
            "source_rpk_list": list(self.source_rpk_list),
            "successor_rpk_list": list(self.successor_rpk_list),
            "resharing_n": self.resharing_n,
            "resharing_threshold": self.resharing_threshold,
            "agreement_n": self.agreement_n,
            "agreement_threshold": self.agreement_threshold,
            "holder_eval_points": _mapping_obj(self.holder_eval_points),
            "source_role_indices": _mapping_obj(self.source_role_indices),
            "successor_role_indices": _mapping_obj(
                self.successor_role_indices
            ),
        }

    def to_obj(self) -> dict[str, Any]:
        obj = self.digest_material()
        obj["context_digest"] = self.context_digest
        return obj

    def to_bytes(self) -> bytes:
        return _canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "RU22Context":
        expected = {
            "kind",
            "version",
            "source_epoch",
            "target_epoch",
            "sid",
            "source_committee",
            "successor_committee",
            "transfer_seq",
            "g_hex",
            "h_hex",
            "source_key_base_hex",
            "successor_key_base_hex",
            "source_rpk_list",
            "successor_rpk_list",
            "resharing_n",
            "resharing_threshold",
            "agreement_n",
            "agreement_threshold",
            "holder_eval_points",
            "source_role_indices",
            "successor_role_indices",
            "context_digest",
        }
        _exact_fields(value, expected, name="RU22 context")
        if value["kind"] != CONTEXT_KIND:
            raise ValueError("unexpected RU22 context kind")
        supplied_digest = _digest(
            value["context_digest"], name="context_digest"
        )
        context = cls(
            version=value["version"],
            source_epoch=value["source_epoch"],
            target_epoch=value["target_epoch"],
            sid=value["sid"],
            source_committee=value["source_committee"],
            successor_committee=value["successor_committee"],
            transfer_seq=value["transfer_seq"],
            g_hex=value["g_hex"],
            h_hex=value["h_hex"],
            source_key_base_hex=value["source_key_base_hex"],
            successor_key_base_hex=value["successor_key_base_hex"],
            source_rpk_list=value["source_rpk_list"],
            successor_rpk_list=value["successor_rpk_list"],
            resharing_n=value["resharing_n"],
            resharing_threshold=value["resharing_threshold"],
            agreement_n=value["agreement_n"],
            agreement_threshold=value["agreement_threshold"],
            holder_eval_points=value["holder_eval_points"],
            source_role_indices=value["source_role_indices"],
            successor_role_indices=value["successor_role_indices"],
        )
        if not hmac.compare_digest(
            context.context_digest, supplied_digest
        ):
            raise ValueError("RU22 context_digest mismatch")
        return context

    @classmethod
    def from_bytes(
        cls, data: bytes | bytearray | memoryview | str
    ) -> "RU22Context":
        return cls.from_obj(_decode_object(data, name="RU22 context"))

    def build_qid(self) -> Qid:
        return Qid.build(
            QID_KIND,
            context_digest=self.context_digest,
            epoch=self.source_epoch,
            sid=self.sid,
            source_committee=self.source_committee,
            successor_committee=self.successor_committee,
            transfer_seq=self.transfer_seq,
        )

    def encode_qid(self) -> str:
        return self.build_qid().encode()


@dataclass(frozen=True)
class RU22InitialCertificate:
    certificate_digest: str
    kind: str = field(default="initial", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "certificate_digest",
            _digest(self.certificate_digest, name="certificate_digest"),
        )

    def to_obj(self) -> dict:
        return {
            "kind": self.kind,
            "certificate_digest": self.certificate_digest,
        }


@dataclass(frozen=True)
class RU22DAGDeliveryCertificate:
    qid: str | Qid
    ordered_vertex: tuple[int, int]
    selected_mids: Sequence[str]
    kind: str = field(default="dag-delivery", init=False)

    def __post_init__(self) -> None:
        if isinstance(self.qid, Qid):
            qid = self.qid.encode()
        elif isinstance(self.qid, str):
            decoded = Qid.decode(self.qid)
            qid = self.qid
            if decoded.encode() != qid:
                raise ValueError("certificate qid must be canonical")
        else:
            raise TypeError("certificate qid must be a Qid or string")
        if (
            not isinstance(self.ordered_vertex, (list, tuple))
            or len(self.ordered_vertex) != 2
        ):
            raise ValueError("ordered_vertex must be a (round, source) pair")
        vertex = (
            _strict_int(self.ordered_vertex[0], name="ordered round"),
            _strict_int(self.ordered_vertex[1], name="ordered source"),
        )
        if not isinstance(self.selected_mids, (list, tuple)):
            raise TypeError("selected_mids must be a sequence")
        mids = tuple(
            _nonempty_text(mid, name="selected mid")
            for mid in self.selected_mids
        )
        if not mids or len(set(mids)) != len(mids):
            raise ValueError("selected_mids must be non-empty and distinct")
        if tuple(sorted(mids)) != mids:
            raise ValueError("selected_mids must be sorted canonically")
        object.__setattr__(self, "qid", qid)
        object.__setattr__(self, "ordered_vertex", vertex)
        object.__setattr__(self, "selected_mids", mids)

    def to_obj(self) -> dict:
        return {
            "kind": self.kind,
            "qid": self.qid,
            "ordered_vertex": {
                "round": self.ordered_vertex[0],
                "source": self.ordered_vertex[1],
            },
            "selected_mids": list(self.selected_mids),
        }


RU22VPSCertificate = Union[
    RU22InitialCertificate,
    RU22DAGDeliveryCertificate,
    RU22QuorumCertificate,
]


def _certificate_from_obj(value: Mapping[str, Any]) -> RU22VPSCertificate:
    if not isinstance(value, Mapping):
        raise ValueError("VPS certificate must be an object")
    kind = value.get("kind")
    if kind == "ru22-qvc-v1":
        return RU22QuorumCertificate.from_obj(value)
    if kind == "initial":
        _exact_fields(
            value,
            {"kind", "certificate_digest"},
            name="initial certificate",
        )
        return RU22InitialCertificate(value["certificate_digest"])
    if kind == "dag-delivery":
        _exact_fields(
            value,
            {
                "kind",
                "qid",
                "ordered_vertex",
                "selected_mids",
            },
            name="DAG delivery certificate",
        )
        vertex = value["ordered_vertex"]
        _exact_fields(
            vertex, {"round", "source"}, name="ordered_vertex"
        )
        return RU22DAGDeliveryCertificate(
            qid=value["qid"],
            ordered_vertex=(vertex["round"], vertex["source"]),
            selected_mids=value["selected_mids"],
        )
    raise ValueError("unsupported VPS certificate kind")


@dataclass(frozen=True)
class RU22ElGamalCiphertext:
    c1: str
    c2: str

    def __post_init__(self) -> None:
        normalized = normalize_elgamal_ciphertext(
            {"C1": self.c1, "C2": self.c2}
        )
        object.__setattr__(self, "c1", normalized["C1"])
        object.__setattr__(self, "c2", normalized["C2"])

    def to_obj(self) -> dict[str, str]:
        return {"C1": self.c1, "C2": self.c2}

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "RU22ElGamalCiphertext":
        _exact_fields(value, {"C1", "C2"}, name="ElGamal ciphertext")
        return cls(c1=value["C1"], c2=value["C2"])


@dataclass(frozen=True)
class RU22VPS:
    epoch: int
    committee: str
    committee_context_digest: str
    encrypted_commitments: Sequence[
        RU22ElGamalCiphertext | Mapping[str, str]
    ]
    certificate: RU22VPSCertificate | Mapping[str, Any]
    version: int = VPS_VERSION
    vps_id: str = field(init=False)

    def __post_init__(self) -> None:
        version = _strict_int(self.version, name="version")
        if version != VPS_VERSION:
            raise ValueError("unsupported RU22 VPS version")
        epoch = _strict_int(self.epoch, name="epoch", minimum=1)
        committee = _nonempty_text(self.committee, name="committee")
        context_digest = _digest(
            self.committee_context_digest,
            name="committee_context_digest",
        )
        if not isinstance(self.encrypted_commitments, (list, tuple)):
            raise TypeError("encrypted_commitments must be a sequence")
        ciphertexts = tuple(
            item
            if isinstance(item, RU22ElGamalCiphertext)
            else RU22ElGamalCiphertext.from_obj(item)
            for item in self.encrypted_commitments
        )
        if not ciphertexts:
            raise ValueError("encrypted_commitments must be non-empty")
        certificate = (
            self.certificate
            if isinstance(
                self.certificate,
                (RU22InitialCertificate, RU22DAGDeliveryCertificate, RU22QuorumCertificate),
            )
            else _certificate_from_obj(self.certificate)
        )
        for name, value in (
            ("version", version),
            ("epoch", epoch),
            ("committee", committee),
            ("committee_context_digest", context_digest),
            ("encrypted_commitments", ciphertexts),
            ("certificate", certificate),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "vps_id",
            _domain_digest(VPS_ID_DOMAIN, self.id_material()),
        )

    def id_material(self) -> dict[str, Any]:
        return {
            "kind": VPS_KIND,
            "version": self.version,
            "epoch": self.epoch,
            "committee": self.committee,
            "committee_context_digest": self.committee_context_digest,
            "encrypted_commitments": [
                ciphertext.to_obj()
                for ciphertext in self.encrypted_commitments
            ],
        }

    def to_obj(self) -> dict[str, Any]:
        obj = self.id_material()
        obj["certificate"] = self.certificate.to_obj()
        obj["vps_id"] = self.vps_id
        return obj

    def to_bytes(self) -> bytes:
        return _canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "RU22VPS":
        _exact_fields(
            value,
            {
                "kind",
                "version",
                "epoch",
                "committee",
                "committee_context_digest",
                "encrypted_commitments",
                "certificate",
                "vps_id",
            },
            name="RU22 VPS",
        )
        if value["kind"] != VPS_KIND:
            raise ValueError("unexpected RU22 VPS kind")
        supplied_id = _digest(value["vps_id"], name="vps_id")
        vps = cls(
            version=value["version"],
            epoch=value["epoch"],
            committee=value["committee"],
            committee_context_digest=value["committee_context_digest"],
            encrypted_commitments=value["encrypted_commitments"],
            certificate=value["certificate"],
        )
        if not hmac.compare_digest(vps.vps_id, supplied_id):
            raise ValueError("RU22 vps_id mismatch")
        return vps

    @classmethod
    def from_bytes(
        cls, data: bytes | bytearray | memoryview | str
    ) -> "RU22VPS":
        return cls.from_obj(_decode_object(data, name="RU22 VPS"))


def _canonical_embedded_qid(
    value: Qid | str,
    *,
    context_digest: str,
) -> str:
    encoded = value.encode() if isinstance(value, Qid) else value
    if not isinstance(encoded, str):
        raise TypeError("qid must be a Qid or canonical encoded string")
    qid = Qid.decode(encoded)
    if qid.encode() != encoded or qid.kind != QID_KIND:
        raise ValueError("qid must be a canonical RU22 qid")
    fields = dict(qid.fields)
    if set(fields) != _QID_FIELDS:
        raise ValueError("qid fields do not match the RU22 contract")
    epoch_text = fields["epoch"]
    if (
        not epoch_text.isdigit()
        or epoch_text.startswith("0")
        or int(epoch_text) < 1
    ):
        raise ValueError("qid epoch must be a canonical positive integer")
    return _canonical_qid(
        encoded,
        context_digest=context_digest,
        source_epoch=int(epoch_text),
    )


def _copy_transcript(
    value: PVSSTranscript | Mapping[str, Any],
) -> PVSSTranscript:
    if isinstance(value, PVSSTranscript):
        transcript = PVSSTranscript.from_bytes(value.to_bytes())
    elif isinstance(value, Mapping):
        _strict_int(value.get("dealer_id"), name="PVSS dealer_id")
        _strict_int(
            value.get("dealer_eval_point"),
            name="PVSS dealer_eval_point",
            minimum=1,
        )
        transcript = PVSSTranscript.from_bytes(_canonical_json(value))
    else:
        raise TypeError("pvss_transcript must be a PVSSTranscript or object")
    if transcript.version != TRANSCRIPT_VERSION:
        raise ValueError("RU22 PVR requires a PVSS transcript v2")
    if transcript.derived_c_v_vector is None:
        raise ValueError("RU22 PVR requires an explicit c_v vector")
    return transcript


def _copy_dleq(
    value: RU22OldToNewDLEQProof | Mapping[str, Any],
) -> RU22OldToNewDLEQProof:
    if isinstance(value, RU22OldToNewDLEQProof):
        return RU22OldToNewDLEQProof.from_bytes(value.to_bytes())
    return RU22OldToNewDLEQProof.from_obj(value)


@dataclass(frozen=True)
class RU22PVR:
    qid: str | Qid
    context_digest: str
    dealer_role_id: int
    dealer_eval_point: int
    parent_vps_id: str
    pvss_transcript: PVSSTranscript | Mapping[str, Any]
    old_to_new_dleq_proof: RU22OldToNewDLEQProof | Mapping[str, Any]
    version: int = PVR_VERSION
    pvr_digest: str = field(init=False)

    def __post_init__(self) -> None:
        version = _strict_int(self.version, name="version")
        if version != PVR_VERSION:
            raise ValueError("unsupported RU22 PVR version")
        context_digest = _digest(
            self.context_digest, name="context_digest"
        )
        qid = _canonical_embedded_qid(
            self.qid, context_digest=context_digest
        )
        dealer_role_id = _strict_int(
            self.dealer_role_id, name="dealer_role_id"
        )
        dealer_eval_point = _strict_int(
            self.dealer_eval_point,
            name="dealer_eval_point",
            minimum=1,
        )
        parent_vps_id = _digest(
            self.parent_vps_id, name="parent_vps_id"
        )
        transcript = _copy_transcript(self.pvss_transcript)
        if transcript.dealer_id != dealer_role_id:
            raise ValueError(
                "PVSS dealer_id does not match PVR dealer_role_id"
            )
        if transcript.dealer_eval_point != dealer_eval_point:
            raise ValueError(
                "PVSS dealer_eval_point does not match PVR"
            )
        proof = _copy_dleq(self.old_to_new_dleq_proof)
        for name, item in (
            ("version", version),
            ("qid", qid),
            ("context_digest", context_digest),
            ("dealer_role_id", dealer_role_id),
            ("dealer_eval_point", dealer_eval_point),
            ("parent_vps_id", parent_vps_id),
            ("pvss_transcript", transcript),
            ("old_to_new_dleq_proof", proof),
        ):
            object.__setattr__(self, name, item)
        object.__setattr__(
            self,
            "pvr_digest",
            _domain_digest(PVR_DIGEST_DOMAIN, self.digest_material()),
        )

    def digest_material(self) -> dict[str, Any]:
        return {
            "kind": PVR_KIND,
            "version": self.version,
            "qid": self.qid,
            "context_digest": self.context_digest,
            "dealer_role_id": self.dealer_role_id,
            "dealer_eval_point": self.dealer_eval_point,
            "parent_vps_id": self.parent_vps_id,
            "pvss_transcript": self.pvss_transcript.to_obj(),
            "old_to_new_dleq_proof": (
                self.old_to_new_dleq_proof.to_obj()
            ),
        }

    def to_obj(self) -> dict[str, Any]:
        obj = self.digest_material()
        obj["pvr_digest"] = self.pvr_digest
        return obj

    def to_bytes(self) -> bytes:
        return _canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "RU22PVR":
        _exact_fields(
            value,
            {
                "kind",
                "version",
                "qid",
                "context_digest",
                "dealer_role_id",
                "dealer_eval_point",
                "parent_vps_id",
                "pvss_transcript",
                "old_to_new_dleq_proof",
                "pvr_digest",
            },
            name="RU22 PVR",
        )
        if value["kind"] != PVR_KIND:
            raise ValueError("unexpected RU22 PVR kind")
        supplied_digest = _digest(
            value["pvr_digest"], name="pvr_digest"
        )
        pvr = cls(
            version=value["version"],
            qid=value["qid"],
            context_digest=value["context_digest"],
            dealer_role_id=value["dealer_role_id"],
            dealer_eval_point=value["dealer_eval_point"],
            parent_vps_id=value["parent_vps_id"],
            pvss_transcript=value["pvss_transcript"],
            old_to_new_dleq_proof=value["old_to_new_dleq_proof"],
        )
        if not hmac.compare_digest(pvr.pvr_digest, supplied_digest):
            raise ValueError("RU22 pvr_digest mismatch")
        return pvr

    @classmethod
    def from_bytes(
        cls, data: bytes | bytearray | memoryview | str
    ) -> "RU22PVR":
        return cls.from_obj(_decode_object(data, name="RU22 PVR"))


@dataclass(frozen=True)
class RU22ContributionBatch:
    qid: str | Qid
    context_digest: str
    holder_role_id: int
    holder_eval_point: int
    pvr_entries: Sequence[RU22PVR | Mapping[str, Any]]
    version: int = BATCH_VERSION
    batch_digest: str = field(init=False)

    def __post_init__(self) -> None:
        version = _strict_int(self.version, name="version")
        if version != BATCH_VERSION:
            raise ValueError("unsupported RU22 contribution batch version")
        context_digest = _digest(
            self.context_digest, name="context_digest"
        )
        qid = _canonical_embedded_qid(
            self.qid, context_digest=context_digest
        )
        holder_role_id = _strict_int(
            self.holder_role_id, name="holder_role_id"
        )
        holder_eval_point = _strict_int(
            self.holder_eval_point,
            name="holder_eval_point",
            minimum=1,
        )
        if not isinstance(self.pvr_entries, (list, tuple)):
            raise TypeError("pvr_entries must be a sequence")
        entries = tuple(
            RU22PVR.from_bytes(item.to_bytes())
            if isinstance(item, RU22PVR)
            else RU22PVR.from_obj(item)
            for item in self.pvr_entries
        )
        if not entries:
            raise ValueError("pvr_entries must be non-empty")
        parent_ids: set[str] = set()
        for pvr in entries:
            if (
                pvr.qid != qid
                or pvr.context_digest != context_digest
                or pvr.dealer_role_id != holder_role_id
                or pvr.dealer_eval_point != holder_eval_point
            ):
                raise ValueError(
                    "batch PVR identity does not match the holder batch"
                )
            if pvr.parent_vps_id in parent_ids:
                raise ValueError(
                    "batch contains duplicate parent_vps_id"
                )
            parent_ids.add(pvr.parent_vps_id)
        for name, item in (
            ("version", version),
            ("qid", qid),
            ("context_digest", context_digest),
            ("holder_role_id", holder_role_id),
            ("holder_eval_point", holder_eval_point),
            ("pvr_entries", entries),
        ):
            object.__setattr__(self, name, item)
        object.__setattr__(
            self,
            "batch_digest",
            _domain_digest(BATCH_DIGEST_DOMAIN, self.digest_material()),
        )

    def digest_material(self) -> dict[str, Any]:
        return {
            "kind": BATCH_KIND,
            "version": self.version,
            "qid": self.qid,
            "context_digest": self.context_digest,
            "holder_role_id": self.holder_role_id,
            "holder_eval_point": self.holder_eval_point,
            "pvr_entries": [pvr.to_obj() for pvr in self.pvr_entries],
        }

    def to_obj(self) -> dict[str, Any]:
        obj = self.digest_material()
        obj["batch_digest"] = self.batch_digest
        return obj

    def to_bytes(self) -> bytes:
        return _canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "RU22ContributionBatch":
        _exact_fields(
            value,
            {
                "kind",
                "version",
                "qid",
                "context_digest",
                "holder_role_id",
                "holder_eval_point",
                "pvr_entries",
                "batch_digest",
            },
            name="RU22 contribution batch",
        )
        if value["kind"] != BATCH_KIND:
            raise ValueError("unexpected RU22 contribution batch kind")
        supplied_digest = _digest(
            value["batch_digest"], name="batch_digest"
        )
        batch = cls(
            version=value["version"],
            qid=value["qid"],
            context_digest=value["context_digest"],
            holder_role_id=value["holder_role_id"],
            holder_eval_point=value["holder_eval_point"],
            pvr_entries=value["pvr_entries"],
        )
        if not hmac.compare_digest(batch.batch_digest, supplied_digest):
            raise ValueError("RU22 batch_digest mismatch")
        return batch

    @classmethod
    def from_bytes(
        cls, data: bytes | bytearray | memoryview | str
    ) -> "RU22ContributionBatch":
        return cls.from_obj(
            _decode_object(data, name="RU22 contribution batch")
        )


def validate_ru22_contribution_batch_shape(
    context: RU22Context,
    batch: RU22ContributionBatch,
) -> None:
    """Validate Phase-3 public identity, width, and recipient-list bindings.

    This is intentionally not the Phase-4 cryptographic acceptance predicate:
    it does not verify PVSS or DLEQ proofs and does not resolve parent VPS
    certificates.  It freezes the context-dependent outer shape needed before
    a batch may enter that later verifier.
    """

    if not isinstance(context, RU22Context):
        raise TypeError("context must be an RU22Context")
    if not isinstance(batch, RU22ContributionBatch):
        raise TypeError("batch must be an RU22ContributionBatch")
    if (
        batch.qid != context.encode_qid()
        or batch.context_digest != context.context_digest
    ):
        raise ValueError("batch does not match the RU22 context/qid")
    if len(batch.pvr_entries) > context.resharing_n:
        raise ValueError("batch contains more than resharing_n PVRs")
    if (
        context.holder_eval_points.get(batch.holder_role_id)
        != batch.holder_eval_point
    ):
        raise ValueError("batch holder identity does not match context")

    for pvr in batch.pvr_entries:
        transcript = pvr.pvss_transcript
        if len(transcript.commitment_vector) != context.resharing_n + 1:
            raise ValueError("PVR commitment vector has the wrong width")
        if len(transcript.enc_results) != context.resharing_n:
            raise ValueError("PVR encrypted recipient list has the wrong width")
        if (
            transcript.derived_c_v_vector is None
            or len(transcript.derived_c_v_vector) != context.resharing_n
        ):
            raise ValueError("PVR explicit c_v vector has the wrong width")
        for recipient_index, (enc_result, expected_key) in enumerate(
            zip(transcript.enc_results, context.successor_rpk_list)
        ):
            if not isinstance(enc_result, Mapping):
                raise ValueError("PVR encrypted recipient must be an object")
            try:
                actual_index = _strict_int(
                    enc_result["recipient"],
                    name="PVSS recipient index",
                )
                actual_key = _point(
                    enc_result["rpk"],
                    name="PVSS recipient public key",
                )
            except KeyError as exc:
                raise ValueError(
                    "PVR encrypted recipient is missing identity fields"
                ) from exc
            if actual_index != recipient_index or actual_key != expected_key:
                raise ValueError(
                    "PVR encrypted recipient list does not match context"
                )


@dataclass(frozen=True)
class RU22LocalVPSShare:
    """One locally held VPS plus the field pair opening this role's c_v."""

    vps: RU22VPS
    share_s: int
    share_s_hat: int

    def __post_init__(self) -> None:
        if not isinstance(self.vps, RU22VPS):
            raise TypeError("vps must be an RU22VPS")
        object.__setattr__(
            self, "vps", RU22VPS.from_bytes(self.vps.to_bytes())
        )
        object.__setattr__(
            self, "share_s", _scalar(self.share_s, name="share_s")
        )
        object.__setattr__(
            self,
            "share_s_hat",
            _scalar(self.share_s_hat, name="share_s_hat"),
        )


@dataclass
class RU22HolderState:
    """One old role's one-shot private state for up to ``n`` local VPSs."""

    holder_role_id: int
    holder_eval_point: int
    recipient_index: int
    recipient_secret_key: int
    resharing_n: int
    local_vps_shares: Sequence[RU22LocalVPSShare]
    consumed: bool = field(default=False, init=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self.holder_role_id = _strict_int(
            self.holder_role_id, name="holder_role_id"
        )
        self.holder_eval_point = _strict_int(
            self.holder_eval_point,
            name="holder_eval_point",
            minimum=1,
        )
        self.recipient_index = _strict_int(
            self.recipient_index, name="recipient_index"
        )
        self.recipient_secret_key = _scalar(
            self.recipient_secret_key,
            name="recipient_secret_key",
            nonzero=True,
        )
        self.resharing_n = _strict_int(
            self.resharing_n, name="resharing_n", minimum=1
        )
        if not isinstance(self.local_vps_shares, (list, tuple)):
            raise TypeError("local_vps_shares must be a sequence")
        entries = tuple(self.local_vps_shares)
        if not entries or len(entries) > self.resharing_n:
            raise ValueError(
                "local_vps_shares must contain between 1 and resharing_n VPSs"
            )
        if not all(isinstance(item, RU22LocalVPSShare) for item in entries):
            raise TypeError(
                "local_vps_shares items must be RU22LocalVPSShare objects"
            )
        ids = [entry.vps.vps_id for entry in entries]
        if len(set(ids)) != len(ids):
            raise ValueError("local VPS list contains a duplicate VPS")
        self.local_vps_shares = entries

    def validate_public(self, context: RU22Context) -> None:
        """Reject public/configuration errors without reading private values."""

        if not isinstance(context, RU22Context):
            raise TypeError("context must be an RU22Context")
        if self.consumed:
            raise RuntimeError("RU22 holder state already consumed (one-shot)")
        if self.resharing_n != context.resharing_n:
            raise ValueError("holder state resharing_n does not match context")
        try:
            expected_eval = context.holder_eval_points[
                self.holder_role_id
            ]
            expected_index = context.source_role_indices[
                self.holder_role_id
            ]
        except KeyError as exc:
            raise ValueError(
                "holder role is not in the source committee"
            ) from exc
        if self.holder_eval_point != expected_eval:
            raise ValueError(
                "holder_eval_point does not match RU22 context"
            )
        if self.recipient_index != expected_index:
            raise ValueError("recipient_index does not match RU22 context")
        if len(self.local_vps_shares) > context.resharing_n:
            raise ValueError("local VPS list exceeds resharing_n")
        for entry in self.local_vps_shares:
            vps = entry.vps
            if (
                vps.epoch != context.source_epoch
                or vps.committee != context.source_committee
            ):
                raise ValueError("local VPS is for the wrong source state")
            if len(vps.encrypted_commitments) != context.resharing_n:
                raise ValueError(
                    "local VPS c_v width does not match resharing_n"
                )

    def consume(
        self,
    ) -> tuple[int, int, int, int, tuple[RU22LocalVPSShare, ...]]:
        """Atomically take and logically erase all role-owned private state."""

        with self._lock:
            if self.consumed:
                raise RuntimeError(
                    "RU22 holder state already consumed (one-shot)"
                )
            values = (
                self.holder_role_id,
                self.holder_eval_point,
                self.recipient_index,
                self.recipient_secret_key,
                tuple(self.local_vps_shares),
            )
            self.recipient_secret_key = None
            self.local_vps_shares = ()
            self.consumed = True
            return values


@dataclass
class RU22HolderMetrics:
    old_share_binding_seconds: float = 0.0
    old_share_binding_cpu_seconds: float = 0.0
    pvss_seconds: float = 0.0
    pvss_cpu_seconds: float = 0.0
    dleq_seconds: float = 0.0
    dleq_cpu_seconds: float = 0.0
    serialization_seconds: float = 0.0
    serialization_cpu_seconds: float = 0.0
    broadcast_seconds: float = 0.0
    broadcast_cpu_seconds: float = 0.0
    pvr_count: int = 0
    pvr_bytes: int = 0
    batch_bytes: int = 0


@dataclass(frozen=True)
class RU22HolderActionResult:
    qid: str
    mid: str
    batch: RU22ContributionBatch
    batch_bytes: bytes
    metrics: RU22HolderMetrics


def _validate_private_opening(
    *,
    context: RU22Context,
    recipient_index: int,
    recipient_secret_key: int,
    entry: RU22LocalVPSShare,
) -> None:
    ciphertext = entry.vps.encrypted_commitments[
        recipient_index
    ].to_obj()
    decrypted_commitment = decrypt_elgamal_ciphertext(
        recipient_secret_key, ciphertext
    )
    claimed_commitment = pedersen_commit(
        context.g_hex,
        context.h_hex,
        entry.share_s,
        entry.share_s_hat,
    )
    if G1Point.from_hex(decrypted_commitment) != G1Point.from_hex(
        claimed_commitment
    ):
        raise ValueError(
            "local share pair does not open the parent VPS c_v"
        )


def make_ru22_contribution_batch(
    *,
    context: RU22Context,
    holder_role_id: int,
    holder_eval_point: int,
    recipient_index: int,
    recipient_secret_key: int,
    local_vps_shares: Sequence[RU22LocalVPSShare],
    metrics: RU22HolderMetrics | None = None,
) -> RU22ContributionBatch:
    """Construct every PVR for one old role and package one speak-once batch."""

    if not isinstance(context, RU22Context):
        raise TypeError("context must be an RU22Context")
    holder_role_id = _strict_int(
        holder_role_id, name="holder_role_id"
    )
    holder_eval_point = _strict_int(
        holder_eval_point, name="holder_eval_point", minimum=1
    )
    recipient_index = _strict_int(
        recipient_index, name="recipient_index"
    )
    recipient_secret_key = _scalar(
        recipient_secret_key,
        name="recipient_secret_key",
        nonzero=True,
    )
    entries = tuple(local_vps_shares)
    if not entries or len(entries) > context.resharing_n:
        raise ValueError(
            "local_vps_shares must contain between 1 and resharing_n entries"
        )
    if not all(isinstance(item, RU22LocalVPSShare) for item in entries):
        raise TypeError(
            "local_vps_shares items must be RU22LocalVPSShare objects"
        )
    if context.holder_eval_points.get(holder_role_id) != holder_eval_point:
        raise ValueError("holder evaluation point does not match context")
    if context.source_role_indices.get(holder_role_id) != recipient_index:
        raise ValueError("source recipient index does not match context")
    if len({entry.vps.vps_id for entry in entries}) != len(entries):
        raise ValueError("local VPS list contains a duplicate VPS")
    timings = metrics if metrics is not None else RU22HolderMetrics()
    if not isinstance(timings, RU22HolderMetrics):
        raise TypeError("metrics must be RU22HolderMetrics")

    qid = context.encode_qid()
    expected_public_key = derive_public_key(
        recipient_secret_key, context.source_key_base_hex
    )
    if expected_public_key != context.source_rpk_list[recipient_index]:
        raise ValueError(
            "recipient_secret_key does not match the source public key"
        )
    pvr_entries: list[RU22PVR] = []
    for entry in entries:
        vps = entry.vps
        if (
            vps.epoch != context.source_epoch
            or vps.committee != context.source_committee
            or len(vps.encrypted_commitments) != context.resharing_n
        ):
            raise ValueError("local VPS does not match the source context")

        started = time.perf_counter()
        cpu_started = time.process_time()
        _validate_private_opening(
            context=context,
            recipient_index=recipient_index,
            recipient_secret_key=recipient_secret_key,
            entry=entry,
        )
        timings.old_share_binding_seconds += time.perf_counter() - started
        timings.old_share_binding_cpu_seconds += (
            time.process_time() - cpu_started
        )

        started = time.perf_counter()
        cpu_started = time.process_time()
        transcript = pvss_deal(
            context.g_hex,
            context.h_hex,
            context.successor_rpk_list,
            entry.share_s,
            entry.share_s_hat,
            context.resharing_threshold,
            dealer_id=holder_role_id,
            dealer_eval_point=holder_eval_point,
            key_base_hex=context.successor_key_base_hex,
        )
        timings.pvss_seconds += time.perf_counter() - started
        timings.pvss_cpu_seconds += time.process_time() - cpu_started

        started = time.perf_counter()
        cpu_started = time.process_time()
        proof = prove_old_to_new_dleq(
            recipient_secret_key=recipient_secret_key,
            qid=qid,
            context_digest=context.context_digest,
            source_epoch=context.source_epoch,
            target_epoch=context.target_epoch,
            parent_vps_id=vps.vps_id,
            dealer_role_id=holder_role_id,
            dealer_eval_point=holder_eval_point,
            recipient_index=recipient_index,
            key_base_hex=context.source_key_base_hex,
            recipient_public_key_hex=(
                context.source_rpk_list[recipient_index]
            ),
            c_v=vps.encrypted_commitments[recipient_index].to_obj(),
            constant_term_commitment_hex=(
                transcript.commitment_vector[0]
            ),
        )
        timings.dleq_seconds += time.perf_counter() - started
        timings.dleq_cpu_seconds += time.process_time() - cpu_started
        pvr_entries.append(
            RU22PVR(
                qid=qid,
                context_digest=context.context_digest,
                dealer_role_id=holder_role_id,
                dealer_eval_point=holder_eval_point,
                parent_vps_id=vps.vps_id,
                pvss_transcript=transcript,
                old_to_new_dleq_proof=proof,
            )
        )

    batch = RU22ContributionBatch(
        qid=qid,
        context_digest=context.context_digest,
        holder_role_id=holder_role_id,
        holder_eval_point=holder_eval_point,
        pvr_entries=pvr_entries,
    )
    validate_ru22_contribution_batch_shape(context, batch)
    timings.pvr_count = len(batch.pvr_entries)
    timings.pvr_bytes = sum(
        len(pvr.to_bytes()) for pvr in batch.pvr_entries
    )
    return batch


async def run_ru22_handover_role(
    *,
    context: RU22Context,
    holder_state: RU22HolderState,
    broadcast_batch,
) -> RU22HolderActionResult:
    """Consume one old RU22 role and broadcast exactly one PVR batch."""

    if not callable(broadcast_batch):
        raise TypeError("broadcast_batch must be callable")
    if not isinstance(holder_state, RU22HolderState):
        raise TypeError("holder_state must be an RU22HolderState")
    if not isinstance(context, RU22Context):
        raise TypeError("context must be an RU22Context")

    # All checks that only inspect public identity/state happen before consume.
    # Any later failure is fail-closed: retrying would produce fresh PVSS and
    # DLEQ randomness for a role that is allowed to speak only once.
    holder_state.validate_public(context)
    (
        holder_role_id,
        holder_eval_point,
        recipient_index,
        recipient_secret_key,
        local_vps_shares,
    ) = holder_state.consume()

    metrics = RU22HolderMetrics()
    batch = make_ru22_contribution_batch(
        context=context,
        holder_role_id=holder_role_id,
        holder_eval_point=holder_eval_point,
        recipient_index=recipient_index,
        recipient_secret_key=recipient_secret_key,
        local_vps_shares=local_vps_shares,
        metrics=metrics,
    )
    # Logical erasure only: immutable Python integers cannot be zeroized.
    del recipient_secret_key, local_vps_shares

    started = time.perf_counter()
    cpu_started = time.process_time()
    batch_bytes = batch.to_bytes()
    metrics.serialization_seconds = time.perf_counter() - started
    metrics.serialization_cpu_seconds = time.process_time() - cpu_started
    metrics.batch_bytes = len(batch_bytes)
    qid = context.encode_qid()
    mid = f"node_{holder_role_id}"

    started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        pending = broadcast_batch(qid, mid, batch_bytes)
        if inspect.isawaitable(pending):
            await pending
    finally:
        metrics.broadcast_seconds = time.perf_counter() - started
        metrics.broadcast_cpu_seconds = time.process_time() - cpu_started
    return RU22HolderActionResult(
        qid=qid,
        mid=mid,
        batch=batch,
        batch_bytes=batch_bytes,
        metrics=metrics,
    )


def _normalize_pool_entry(value: Any) -> PoolEntry | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    mid, payload = value
    if not isinstance(mid, str) or not mid:
        return None
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        return None
    return mid, bytes(payload)


def _stable_pool_entry(value: PoolEntry) -> tuple[str, str]:
    return value[0], value[1].hex()


@dataclass(frozen=True)
class RU22ValidatedBatch:
    """A structurally valid batch and the subset of cryptographically valid PVRs."""

    entry: PoolEntry
    batch: RU22ContributionBatch
    valid_pvrs: tuple[RU22PVR, ...]


@dataclass(frozen=True)
class RU22JustifiedSelection:
    """Deterministic public derivation from one justified threshold set."""

    parent_vps_id: str
    entries: tuple[PoolEntry, ...]
    batches: tuple[RU22ContributionBatch, ...]
    pvrs: tuple[RU22PVR, ...]
    selected_role_ids: tuple[int, ...]
    selected_eval_points: tuple[int, ...]
    lagrange_coefficients: tuple[int, ...]
    combined_c_v_vector: tuple[RU22ElGamalCiphertext, ...]


def verify_ru22_pvr_crypto(
    context: RU22Context, parent: RU22VPS, pvr: RU22PVR,
    record_stage=lambda *_: None,
) -> bool:
    """PVSS/chunk and encrypted-parent binding checks shared by both runtimes.

    The caller must authenticate the parent and validate the PVR context/shape.
    This function does not accept a DAG delivery or perform any ordering.
    """
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    accepted = pvss_verify(
        context.g_hex,
        context.h_hex,
        context.successor_rpk_list,
        pvr.pvss_transcript,
        context.resharing_threshold,
        key_base_hex=context.successor_key_base_hex,
    )
    record_stage("pvss", wall_started, cpu_started)
    if not accepted:
        return False

    try:
        recipient_index = context.source_role_indices[
            pvr.dealer_role_id
        ]
        old_c_v = parent.encrypted_commitments[
            recipient_index
        ].to_obj()
    except (KeyError, IndexError):
        return False
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    accepted = verify_old_to_new_dleq(
        proof=pvr.old_to_new_dleq_proof,
        qid=context.encode_qid(),
        context_digest=context.context_digest,
        source_epoch=context.source_epoch,
        target_epoch=context.target_epoch,
        parent_vps_id=parent.vps_id,
        dealer_role_id=pvr.dealer_role_id,
        dealer_eval_point=pvr.dealer_eval_point,
        recipient_index=recipient_index,
        key_base_hex=context.source_key_base_hex,
        recipient_public_key_hex=(
            context.source_rpk_list[recipient_index]
        ),
        c_v=old_c_v,
        constant_term_commitment_hex=(
            pvr.pvss_transcript.commitment_vector[0]
        ),
    )
    record_stage("dleq", wall_started, cpu_started)
    return bool(accepted)



class RU22DPSSJustifierManager(Justifier):
    """Pure ``(qid, entries)`` RU22 verifier and deterministic selector.

    A parent VPS is usable only after it is explicitly registered in the
    certified catalog for a context.  Registration is the Phase-4 trust
    boundary for either a genesis certificate or a prior DAG delivery record;
    parsing delivery results and advancing the lifecycle belong to Phase 5.
    """

    def __init__(
        self,
        contexts: Sequence[RU22Context] | None = None,
        validation_cache_limit: int = 4096,
    ) -> None:
        self.validation_cache_limit = _strict_int(
            validation_cache_limit,
            name="validation_cache_limit",
            minimum=1,
        )
        self.contexts: dict[str, RU22Context] = {}
        self.parent_catalogs: dict[str, dict[str, RU22VPS]] = {}
        self._validation_cache: dict[
            tuple[str, str, str],
            tuple[str, ...] | None,
        ] = {}
        self._validation_cache_hits = 0
        self._validation_metrics = {
            stage: {
                "invocations": 0,
                "elapsed_sec": 0.0,
                "cpu_seconds": 0.0,
            }
            for stage in ("schema_identity", "pvss", "dleq")
        }
        self._formal_metrics = {
            role: {
                "invocations": 0,
                "elapsed_sec": 0.0,
                "cpu_seconds": 0.0,
                "records": [],
            }
            for role in ("proposer", "committed_order")
        }
        self._verified_selections: dict[
            tuple[str, tuple[PoolEntry, ...]],
            RU22JustifiedSelection,
        ] = {}
        for context in contexts or ():
            self.register_context(context)

    def _record_stage(
        self,
        stage: str,
        wall_started: float,
        cpu_started: float,
    ) -> None:
        metric = self._validation_metrics[stage]
        metric["invocations"] += 1
        metric["elapsed_sec"] += time.perf_counter() - wall_started
        metric["cpu_seconds"] += time.process_time() - cpu_started

    def validation_metrics_snapshot(self) -> dict[str, Any]:
        return {
            "cache_hits": self._validation_cache_hits,
            "auxiliary_cache_hits": self._validation_cache_hits,
            "formal_cache_hits": 0,
            "formal_validation_mode": "uncached-per-invocation",
            **{
                stage: {
                    "invocations": int(metric["invocations"]),
                    "elapsed_sec": float(metric["elapsed_sec"]),
                    "cpu_seconds": float(metric["cpu_seconds"]),
                }
                for stage, metric in self._validation_metrics.items()
            },
        }

    def formal_metrics_snapshot(self) -> dict[str, Any]:
        return {
            "formal_validation_mode": "uncached-per-invocation",
            "formal_cache_hits": 0,
            "auxiliary_cache_hits": self._validation_cache_hits,
            "roles": {
                role: {
                    "invocations": int(metric["invocations"]),
                    "elapsed_sec": float(metric["elapsed_sec"]),
                    "cpu_seconds": float(metric["cpu_seconds"]),
                    "records": [
                        dict(record) for record in metric["records"]
                    ],
                }
                for role, metric in self._formal_metrics.items()
            },
        }

    def _stage_invocations(self) -> dict[str, int]:
        return {
            stage: int(metric["invocations"])
            for stage, metric in self._validation_metrics.items()
        }

    def _record_formal_invocation(
        self,
        role: str,
        *,
        qid: Qid,
        entry_count: int,
        selected_entry_count: int,
        accepted: bool,
        wall_started: float,
        cpu_started: float,
        stages_before: Mapping[str, int],
    ) -> None:
        metric = self._formal_metrics[role]
        elapsed = time.perf_counter() - wall_started
        cpu_seconds = time.process_time() - cpu_started
        metric["invocations"] += 1
        metric["elapsed_sec"] += elapsed
        metric["cpu_seconds"] += cpu_seconds
        stage_deltas = {
            stage: int(values["invocations"])
            - int(stages_before[stage])
            for stage, values in self._validation_metrics.items()
        }
        metric["records"].append(
            {
                "invocation": int(metric["invocations"]),
                "qid": qid.encode(),
                "entry_count": int(entry_count),
                "selected_entry_count": int(selected_entry_count),
                "accepted": bool(accepted),
                "elapsed_sec": float(elapsed),
                "cpu_seconds": float(cpu_seconds),
                "stage_invocations": stage_deltas,
            }
        )

    def _drop_cached_context(self, context_digest: str) -> None:
        for key in tuple(self._validation_cache):
            if key[0] == context_digest:
                self._validation_cache.pop(key, None)
        for key in tuple(self._verified_selections):
            if key[0] == context_digest:
                self._verified_selections.pop(key, None)

    def _cache_validation(
        self,
        key: tuple[str, str, str],
        valid_parent_ids: tuple[str, ...] | None,
    ) -> None:
        if key not in self._validation_cache and (
            len(self._validation_cache) >= self.validation_cache_limit
        ):
            self._validation_cache.pop(next(iter(self._validation_cache)))
        self._validation_cache[key] = valid_parent_ids

    def register_context(
        self,
        context: RU22Context,
        parent_vps_catalog: Sequence[RU22VPS] = (),
    ) -> None:
        if not isinstance(context, RU22Context):
            raise TypeError("context must be an RU22Context")
        existing = self.contexts.get(context.context_digest)
        if existing is not None and existing != context:
            raise ValueError("conflicting RU22 context for context_digest")
        self.contexts[context.context_digest] = context
        self.parent_catalogs.setdefault(context.context_digest, {})
        self._drop_cached_context(context.context_digest)
        for vps in parent_vps_catalog:
            self.register_parent_vps(context, vps)

    def register_parent_vps(
        self,
        context_or_digest: RU22Context | str,
        vps: RU22VPS,
    ) -> None:
        digest = (
            context_or_digest.context_digest
            if isinstance(context_or_digest, RU22Context)
            else _digest(context_or_digest, name="context_digest")
        )
        try:
            context = self.contexts[digest]
        except KeyError as exc:
            raise KeyError(f"unknown RU22 context: {digest}") from exc
        if not isinstance(vps, RU22VPS):
            raise TypeError("vps must be an RU22VPS")
        normalized = RU22VPS.from_bytes(vps.to_bytes())
        if (
            normalized.epoch != context.source_epoch
            or normalized.committee != context.source_committee
            or len(normalized.encrypted_commitments)
            != context.resharing_n
        ):
            raise ValueError("parent VPS does not match RU22 source context")
        catalog = self.parent_catalogs[digest]
        existing = catalog.get(normalized.vps_id)
        if (
            existing is not None
            and existing.id_material() != normalized.id_material()
        ):
            raise ValueError("conflicting parent VPS for vps_id")
        catalog[normalized.vps_id] = normalized
        self._drop_cached_context(digest)

    def unregister_context(
        self,
        context_or_digest: RU22Context | str,
    ) -> None:
        digest = (
            context_or_digest.context_digest
            if isinstance(context_or_digest, RU22Context)
            else _digest(context_or_digest, name="context_digest")
        )
        self.contexts.pop(digest, None)
        self.parent_catalogs.pop(digest, None)
        self._drop_cached_context(digest)

    def _registration(self, qid: Qid | str) -> RU22Context:
        if isinstance(qid, str):
            qid = Qid.decode(qid)
        if not isinstance(qid, Qid) or qid.kind != QID_KIND:
            raise ValueError("expected an RU22 qid")
        fields = dict(qid.fields)
        digest = _digest(
            fields.get("context_digest"),
            name="qid context_digest",
        )
        try:
            context = self.contexts[digest]
        except KeyError as exc:
            raise KeyError(f"unknown RU22 context: {digest}") from exc
        if qid != context.build_qid():
            raise ValueError("RU22 qid fields do not match registered context")
        return context

    def _valid_pvr(self, context, parent, pvr) -> bool:
        return verify_ru22_pvr_crypto(context, parent, pvr, self._record_stage)

    def validate_batch(
        self,
        context: RU22Context,
        mid: str,
        payload: bytes,
    ) -> RU22ValidatedBatch | None:
        """Cached validation for non-candidate auxiliary paths only."""

        if not isinstance(context, RU22Context):
            return None
        cache_key = (
            context.context_digest,
            str(mid),
            hashlib.sha256(bytes(payload)).hexdigest()
            if isinstance(payload, (bytes, bytearray, memoryview))
            else "malformed",
        )
        if cache_key in self._validation_cache:
            self._validation_cache_hits += 1
            valid_parent_ids = self._validation_cache[cache_key]
            if valid_parent_ids is None:
                return None
            try:
                batch = RU22ContributionBatch.from_bytes(payload)
                by_parent = {
                    pvr.parent_vps_id: pvr
                    for pvr in batch.pvr_entries
                }
                return RU22ValidatedBatch(
                    entry=(str(mid), bytes(payload)),
                    batch=batch,
                    valid_pvrs=tuple(
                        by_parent[parent_id]
                        for parent_id in valid_parent_ids
                    ),
                )
            except Exception:
                return None

        validated = self._validate_batch_uncached(context, mid, payload)
        valid_parent_ids = (
            tuple(
                pvr.parent_vps_id for pvr in validated.valid_pvrs
            )
            if validated is not None
            else None
        )
        self._cache_validation(cache_key, valid_parent_ids)
        return validated

    def _validate_batch_uncached(
        self,
        context: RU22Context,
        mid: str,
        payload: bytes,
    ) -> RU22ValidatedBatch | None:
        """Decode bytes and execute every RU22 batch/PVR check anew."""

        if not isinstance(context, RU22Context):
            return None
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        try:
            batch = RU22ContributionBatch.from_bytes(payload)
            if mid != f"node_{batch.holder_role_id}":
                raise ValueError("batch mid does not authenticate holder role")
            validate_ru22_contribution_batch_shape(context, batch)
        except Exception:
            self._record_stage(
                "schema_identity", wall_started, cpu_started
            )
            return None
        self._record_stage("schema_identity", wall_started, cpu_started)

        catalog = self.parent_catalogs.get(context.context_digest, {})
        valid_pvrs = []
        for pvr in batch.pvr_entries:
            parent = catalog.get(pvr.parent_vps_id)
            if parent is not None and self._valid_pvr(
                context, parent, pvr
            ):
                valid_pvrs.append(pvr)
        valid_pvrs.sort(key=lambda pvr: pvr.parent_vps_id)
        return RU22ValidatedBatch(
            entry=(str(mid), bytes(payload)),
            batch=batch,
            valid_pvrs=tuple(valid_pvrs),
        )

    @staticmethod
    def _derive_from_validated(
        context: RU22Context,
        validated_batches: Sequence[RU22ValidatedBatch],
    ) -> RU22JustifiedSelection | None:
        threshold = context.resharing_threshold + 1
        groups: dict[
            str,
            list[tuple[RU22ValidatedBatch, RU22PVR]],
        ] = {}
        for validated in validated_batches:
            for pvr in validated.valid_pvrs:
                groups.setdefault(pvr.parent_vps_id, []).append(
                    (validated, pvr)
                )

        for parent_vps_id in sorted(groups):
            candidates = sorted(
                groups[parent_vps_id],
                key=lambda item: (
                    item[1].dealer_eval_point,
                    item[1].dealer_role_id,
                    item[0].entry[0],
                    item[0].entry[1].hex(),
                ),
            )
            selected = []
            seen_roles: set[int] = set()
            seen_eval_points: set[int] = set()
            for validated, pvr in candidates:
                if (
                    pvr.dealer_role_id in seen_roles
                    or pvr.dealer_eval_point in seen_eval_points
                ):
                    continue
                selected.append((validated, pvr))
                seen_roles.add(pvr.dealer_role_id)
                seen_eval_points.add(pvr.dealer_eval_point)
                if len(selected) == threshold:
                    break
            if len(selected) != threshold:
                continue

            eval_points = tuple(
                pvr.dealer_eval_point for _validated, pvr in selected
            )
            coefficients = tuple(
                lagrange_coeff_at_zero(eval_points, point)
                for point in eval_points
            )
            combined = interpolate_c_v_vector(
                eval_points,
                [
                    pvr.pvss_transcript.derived_c_v_vector
                    for _validated, pvr in selected
                ],
                context.resharing_threshold,
            )
            return RU22JustifiedSelection(
                parent_vps_id=parent_vps_id,
                entries=tuple(
                    validated.entry for validated, _pvr in selected
                ),
                batches=tuple(
                    validated.batch for validated, _pvr in selected
                ),
                pvrs=tuple(pvr for _validated, pvr in selected),
                selected_role_ids=tuple(
                    pvr.dealer_role_id for _validated, pvr in selected
                ),
                selected_eval_points=eval_points,
                lagrange_coefficients=coefficients,
                combined_c_v_vector=tuple(
                    RU22ElGamalCiphertext.from_obj(item)
                    for item in combined
                ),
            )
        return None

    def _validated_candidates(
        self,
        context: RU22Context,
        entries: Sequence[PoolEntry],
        *,
        reject_malformed: bool,
    ) -> list[RU22ValidatedBatch] | None:
        normalized_entries = []
        for entry in entries:
            normalized = _normalize_pool_entry(entry)
            if normalized is None:
                if reject_malformed:
                    return None
                continue
            normalized_entries.append(normalized)

        validated_batches = []
        for mid, payload in sorted(
            normalized_entries, key=_stable_pool_entry
        ):
            validated = self._validate_batch_uncached(
                context, mid, payload
            )
            if validated is None:
                if reject_malformed:
                    return None
                continue
            validated_batches.append(validated)
        return validated_batches

    @staticmethod
    def _normalized_exact_entries(
        entries: Sequence[PoolEntry],
    ) -> tuple[PoolEntry, ...] | None:
        normalized_entries = []
        try:
            for entry in entries:
                normalized = _normalize_pool_entry(entry)
                if normalized is None:
                    return None
                normalized_entries.append(normalized)
        except TypeError:
            return None
        return tuple(
            sorted(normalized_entries, key=_stable_pool_entry)
        )

    @classmethod
    def _verified_selection_key(
        cls,
        context: RU22Context,
        entries: Sequence[PoolEntry],
    ) -> tuple[str, tuple[PoolEntry, ...]] | None:
        normalized = cls._normalized_exact_entries(entries)
        if normalized is None:
            return None
        return context.context_digest, normalized

    def _derive_exact_selection_uncached(
        self,
        context: RU22Context,
        entries: Sequence[PoolEntry],
    ) -> RU22JustifiedSelection | None:
        if len(entries) != context.resharing_threshold + 1:
            return None
        validated = self._validated_candidates(
            context, entries, reject_malformed=True
        )
        if validated is None or len(validated) != len(entries):
            return None
        selection = self._derive_from_validated(context, validated)
        if selection is None:
            return None
        supplied = self._normalized_exact_entries(entries)
        if supplied is None or tuple(
            sorted(selection.entries, key=_stable_pool_entry)
        ) != supplied:
            return None
        return selection

    def derive_selection(
        self,
        qid: Qid,
        entries: Sequence[PoolEntry],
    ) -> RU22JustifiedSelection | None:
        """Recompute the unique public derivation for a delivered threshold set."""

        try:
            context = self._registration(qid)
        except (KeyError, TypeError, ValueError):
            return None
        return self._derive_exact_selection_uncached(context, entries)

    def select_justified_subset(
        self,
        qid: Qid,
        entries: Sequence[PoolEntry],
    ) -> list[PoolEntry] | None:
        try:
            context = self._registration(qid)
        except (KeyError, TypeError, ValueError):
            return None
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        stages_before = self._stage_invocations()
        selected: list[PoolEntry] | None = None
        try:
            validated = self._validated_candidates(
                context, entries, reject_malformed=False
            )
            if validated is None:
                return None
            selection = self._derive_from_validated(context, validated)
            if selection is not None:
                selected = list(selection.entries)
            return selected
        finally:
            self._record_formal_invocation(
                "proposer",
                qid=context.build_qid(),
                entry_count=len(entries),
                selected_entry_count=(
                    len(selected) if selected is not None else 0
                ),
                accepted=selected is not None,
                wall_started=wall_started,
                cpu_started=cpu_started,
                stages_before=stages_before,
            )

    def verify_qid_set(
        self,
        qid: Qid,
        entries: Sequence[PoolEntry],
    ) -> bool:
        try:
            context = self._registration(qid)
        except (KeyError, TypeError, ValueError):
            return False
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        stages_before = self._stage_invocations()
        selection = None
        try:
            selection = self._derive_exact_selection_uncached(
                context, entries
            )
            key = self._verified_selection_key(context, entries)
            if selection is not None and key is not None:
                self._verified_selections[key] = selection
            elif key is not None:
                self._verified_selections.pop(key, None)
            return selection is not None
        finally:
            self._record_formal_invocation(
                "committed_order",
                qid=context.build_qid(),
                entry_count=len(entries),
                selected_entry_count=(
                    len(selection.entries)
                    if selection is not None
                    else 0
                ),
                accepted=selection is not None,
                wall_started=wall_started,
                cpu_started=cpu_started,
                stages_before=stages_before,
            )

    def verified_selection(
        self,
        qid: Qid | str,
        entries: Sequence[PoolEntry],
    ) -> RU22JustifiedSelection | None:
        """Return the exact result of a prior committed-order verification.

        This store is a trust-boundary handoff, not a validation cache: it is
        populated only after an uncached ``verify_qid_set`` invocation and is
        keyed by the complete immutable entry bytes.
        """

        try:
            context = self._registration(qid)
        except (KeyError, TypeError, ValueError):
            return None
        key = self._verified_selection_key(context, entries)
        if key is None:
            return None
        selection = self._verified_selections.get(key)
        if selection is None:
            return None

        normalized = key[1]
        decoded_by_entry = {}
        try:
            for mid, payload in normalized:
                batch = RU22ContributionBatch.from_bytes(payload)
                if mid != f"node_{batch.holder_role_id}":
                    return None
                validate_ru22_contribution_batch_shape(context, batch)
                decoded_by_entry[(mid, payload)] = batch
        except (TypeError, ValueError):
            return None
        if set(decoded_by_entry) != set(selection.entries):
            return None
        for entry, batch, pvr in zip(
            selection.entries, selection.batches, selection.pvrs
        ):
            decoded = decoded_by_entry.get(entry)
            if decoded is None or decoded != batch:
                return None
            if pvr.pvr_digest not in {
                candidate.pvr_digest for candidate in decoded.pvr_entries
            }:
                return None
        return selection


def _nonempty_unique_text_tuple(
    value: Sequence[str],
    *,
    name: str,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty sequence")
    normalized = tuple(
        _nonempty_text(item, name=f"{name} item") for item in value
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} entries must be distinct")
    return normalized


def _unique_int_tuple(
    value: Sequence[int],
    *,
    name: str,
    minimum: int,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty sequence")
    normalized = tuple(
        _strict_int(item, name=f"{name} item", minimum=minimum)
        for item in value
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} entries must be distinct")
    return normalized


def _digest_tuple(
    value: Sequence[str],
    *,
    name: str,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty sequence")
    normalized = tuple(
        _digest(item, name=f"{name} item") for item in value
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} entries must be distinct")
    return normalized


@dataclass(frozen=True)
class RU22DeliveryResult:
    """Canonical public result derived from one RU22 DAG delivery event."""

    qid: str | Qid
    context_digest: str
    parent_vps_id: str
    selected_mids: Sequence[str]
    selected_role_ids: Sequence[int]
    selected_eval_points: Sequence[int]
    lagrange_coefficients: Sequence[int]
    selected_pvr_digests: Sequence[str]
    combined_c_v_vector: Sequence[
        RU22ElGamalCiphertext | Mapping[str, str]
    ]
    new_vps_id: str
    dag_delivery: RU22DAGDeliveryCertificate | Mapping[str, Any]
    version: int = DELIVERY_RESULT_VERSION
    delivery_result_digest: str = field(init=False)

    def __post_init__(self) -> None:
        version = _strict_int(self.version, name="version")
        if version != DELIVERY_RESULT_VERSION:
            raise ValueError("unsupported RU22 delivery-result version")
        context_digest = _digest(
            self.context_digest, name="context_digest"
        )
        qid = _canonical_embedded_qid(
            self.qid, context_digest=context_digest
        )
        parent_vps_id = _digest(
            self.parent_vps_id, name="parent_vps_id"
        )
        mids = _nonempty_unique_text_tuple(
            self.selected_mids, name="selected_mids"
        )
        role_ids = _unique_int_tuple(
            self.selected_role_ids,
            name="selected_role_ids",
            minimum=0,
        )
        eval_points = _unique_int_tuple(
            self.selected_eval_points,
            name="selected_eval_points",
            minimum=1,
        )
        if not isinstance(self.lagrange_coefficients, (list, tuple)):
            raise TypeError("lagrange_coefficients must be a sequence")
        coefficients = tuple(
            _scalar(item, name="lagrange coefficient")
            for item in self.lagrange_coefficients
        )
        pvr_digests = _digest_tuple(
            self.selected_pvr_digests,
            name="selected_pvr_digests",
        )
        if not (
            len(mids)
            == len(role_ids)
            == len(eval_points)
            == len(coefficients)
            == len(pvr_digests)
        ):
            raise ValueError(
                "RU22 delivery threshold-set fields must have equal length"
            )
        if not isinstance(self.combined_c_v_vector, (list, tuple)):
            raise TypeError("combined_c_v_vector must be a sequence")
        combined = tuple(
            item
            if isinstance(item, RU22ElGamalCiphertext)
            else RU22ElGamalCiphertext.from_obj(item)
            for item in self.combined_c_v_vector
        )
        if not combined:
            raise ValueError("combined_c_v_vector must be non-empty")
        new_vps_id = _digest(self.new_vps_id, name="new_vps_id")
        certificate = (
            self.dag_delivery
            if isinstance(self.dag_delivery, RU22DAGDeliveryCertificate)
            else _certificate_from_obj(self.dag_delivery)
        )
        if not isinstance(certificate, RU22DAGDeliveryCertificate):
            raise ValueError(
                "delivery result requires a DAG delivery certificate"
            )
        if certificate.qid != qid:
            raise ValueError("DAG certificate qid does not match result")
        if set(certificate.selected_mids) != set(mids):
            raise ValueError(
                "DAG certificate mids do not match selected_mids"
            )
        for name, item in (
            ("version", version),
            ("qid", qid),
            ("context_digest", context_digest),
            ("parent_vps_id", parent_vps_id),
            ("selected_mids", mids),
            ("selected_role_ids", role_ids),
            ("selected_eval_points", eval_points),
            ("lagrange_coefficients", coefficients),
            ("selected_pvr_digests", pvr_digests),
            ("combined_c_v_vector", combined),
            ("new_vps_id", new_vps_id),
            ("dag_delivery", certificate),
        ):
            object.__setattr__(self, name, item)
        object.__setattr__(
            self,
            "delivery_result_digest",
            _domain_digest(
                DELIVERY_RESULT_DIGEST_DOMAIN,
                self.digest_material(),
            ),
        )

    def digest_material(self) -> dict[str, Any]:
        return {
            "kind": DELIVERY_RESULT_KIND,
            "version": self.version,
            "qid": self.qid,
            "context_digest": self.context_digest,
            "parent_vps_id": self.parent_vps_id,
            "selected_mids": list(self.selected_mids),
            "selected_role_ids": list(self.selected_role_ids),
            "selected_eval_points": list(self.selected_eval_points),
            "lagrange_coefficients": [
                str(item) for item in self.lagrange_coefficients
            ],
            "selected_pvr_digests": list(
                self.selected_pvr_digests
            ),
            "combined_c_v_vector": [
                item.to_obj() for item in self.combined_c_v_vector
            ],
            "new_vps_id": self.new_vps_id,
            "dag_delivery": self.dag_delivery.to_obj(),
        }

    def to_obj(self) -> dict[str, Any]:
        obj = self.digest_material()
        obj["delivery_result_digest"] = self.delivery_result_digest
        return obj

    def to_bytes(self) -> bytes:
        return _canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "RU22DeliveryResult":
        _exact_fields(
            value,
            {
                "kind",
                "version",
                "qid",
                "context_digest",
                "parent_vps_id",
                "selected_mids",
                "selected_role_ids",
                "selected_eval_points",
                "lagrange_coefficients",
                "selected_pvr_digests",
                "combined_c_v_vector",
                "new_vps_id",
                "dag_delivery",
                "delivery_result_digest",
            },
            name="RU22 delivery result",
        )
        if value["kind"] != DELIVERY_RESULT_KIND:
            raise ValueError("unexpected RU22 delivery-result kind")
        if not isinstance(value["lagrange_coefficients"], (list, tuple)):
            raise TypeError("lagrange_coefficients must be a sequence")
        supplied_digest = _digest(
            value["delivery_result_digest"],
            name="delivery_result_digest",
        )
        result = cls(
            version=value["version"],
            qid=value["qid"],
            context_digest=value["context_digest"],
            parent_vps_id=value["parent_vps_id"],
            selected_mids=value["selected_mids"],
            selected_role_ids=value["selected_role_ids"],
            selected_eval_points=value["selected_eval_points"],
            lagrange_coefficients=[
                _canonical_decimal(
                    item, name="lagrange coefficient"
                )
                for item in value["lagrange_coefficients"]
            ],
            selected_pvr_digests=value["selected_pvr_digests"],
            combined_c_v_vector=value["combined_c_v_vector"],
            new_vps_id=value["new_vps_id"],
            dag_delivery=value["dag_delivery"],
        )
        if not hmac.compare_digest(
            result.delivery_result_digest, supplied_digest
        ):
            raise ValueError("RU22 delivery_result_digest mismatch")
        return result

    @classmethod
    def from_bytes(
        cls, data: bytes | bytearray | memoryview | str
    ) -> "RU22DeliveryResult":
        return cls.from_obj(
            _decode_object(data, name="RU22 delivery result")
        )

    def build_new_vps(self, context: RU22Context) -> RU22VPS:
        if not isinstance(context, RU22Context):
            raise TypeError("context must be an RU22Context")
        if (
            self.qid != context.encode_qid()
            or self.context_digest != context.context_digest
        ):
            raise ValueError("delivery result does not match RU22 context")
        if len(self.combined_c_v_vector) != context.resharing_n:
            raise ValueError("combined_c_v_vector has the wrong width")
        vps = RU22VPS(
            epoch=context.target_epoch,
            committee=context.successor_committee,
            committee_context_digest=context.context_digest,
            encrypted_commitments=self.combined_c_v_vector,
            certificate=self.dag_delivery,
        )
        if not hmac.compare_digest(vps.vps_id, self.new_vps_id):
            raise ValueError("delivery result new_vps_id mismatch")
        return vps


class RU22PublicFinalizer:
    """Finalize exact DAG-delivered RU22 entries into one public VPS record."""

    def __init__(self, justifier: RU22DPSSJustifierManager) -> None:
        if not isinstance(justifier, RU22DPSSJustifierManager):
            raise TypeError(
                "justifier must be an RU22DPSSJustifierManager"
            )
        self.justifier = justifier
        self.results: dict[str, RU22DeliveryResult] = {}
        self._new_vps: dict[str, RU22VPS] = {}
        self._metrics = {
            "public_derivation": {
                "invocations": 0,
                "elapsed_sec": 0.0,
                "cpu_seconds": 0.0,
            }
        }

    def metrics_snapshot(self) -> dict[str, Any]:
        return {
            name: {
                "invocations": int(metric["invocations"]),
                "elapsed_sec": float(metric["elapsed_sec"]),
                "cpu_seconds": float(metric["cpu_seconds"]),
            }
            for name, metric in self._metrics.items()
        }

    def on_deliver(
        self,
        qid: Qid | str,
        entries: Sequence[PoolEntry],
        *,
        ordered_vertex: tuple[int, int],
        committed_wave: int,
    ) -> RU22DeliveryResult | None:
        """Defensively validate untrusted entries, then finalize them."""

        try:
            context = self.justifier._registration(qid)
            qid_obj = context.build_qid()
            vertex = self._normalize_delivery_position(
                ordered_vertex, committed_wave
            )
        except (KeyError, TypeError, ValueError):
            return None

        selection = self.justifier.derive_selection(qid_obj, entries)
        if selection is None:
            return None
        return self._finalize_selection(context, selection, vertex)

    def on_verified_deliver(
        self,
        qid: Qid | str,
        entries: Sequence[PoolEntry],
        *,
        ordered_vertex: tuple[int, int],
        committed_wave: int,
    ) -> RU22DeliveryResult | None:
        """Finalize the exact entries just accepted by local DAG delivery."""

        try:
            context = self.justifier._registration(qid)
            vertex = self._normalize_delivery_position(
                ordered_vertex, committed_wave
            )
        except (KeyError, TypeError, ValueError):
            return None
        selection = self.justifier.verified_selection(qid, entries)
        if selection is None:
            return None
        return self._finalize_selection(context, selection, vertex)

    @staticmethod
    def _normalize_delivery_position(
        ordered_vertex: tuple[int, int],
        committed_wave: int,
    ) -> tuple[int, int]:
        if (
            not isinstance(ordered_vertex, (list, tuple))
            or len(ordered_vertex) != 2
        ):
            raise ValueError(
                "ordered_vertex must be a (round, source) pair"
            )
        # Validate the local diagnostic even though it is deliberately not
        # included in the public VPS certificate or result identity.
        _strict_int(committed_wave, name="committed_wave")
        return (
            _strict_int(
                ordered_vertex[0], name="ordered vertex round"
            ),
            _strict_int(
                ordered_vertex[1], name="ordered vertex source"
            ),
        )

    def _finalize_selection(
        self,
        context: RU22Context,
        selection: RU22JustifiedSelection,
        vertex: tuple[int, int],
    ) -> RU22DeliveryResult:
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        mids = tuple(entry[0] for entry in selection.entries)
        certificate = RU22DAGDeliveryCertificate(
            qid=context.encode_qid(),
            ordered_vertex=vertex,
            selected_mids=tuple(sorted(mids)),
        )
        new_vps = RU22VPS(
            epoch=context.target_epoch,
            committee=context.successor_committee,
            committee_context_digest=context.context_digest,
            encrypted_commitments=selection.combined_c_v_vector,
            certificate=certificate,
        )
        result = RU22DeliveryResult(
            qid=context.encode_qid(),
            context_digest=context.context_digest,
            parent_vps_id=selection.parent_vps_id,
            selected_mids=mids,
            selected_role_ids=selection.selected_role_ids,
            selected_eval_points=selection.selected_eval_points,
            lagrange_coefficients=selection.lagrange_coefficients,
            selected_pvr_digests=tuple(
                pvr.pvr_digest for pvr in selection.pvrs
            ),
            combined_c_v_vector=selection.combined_c_v_vector,
            new_vps_id=new_vps.vps_id,
            dag_delivery=certificate,
        )
        metric = self._metrics["public_derivation"]
        metric["invocations"] += 1
        metric["elapsed_sec"] += time.perf_counter() - wall_started
        metric["cpu_seconds"] += time.process_time() - cpu_started

        existing = self.results.get(context.encode_qid())
        if existing is not None:
            if existing != result:
                raise ValueError(
                    "conflicting DAG delivery for completed RU22 qid"
                )
            return existing
        self.results[context.encode_qid()] = result
        self._new_vps[context.encode_qid()] = new_vps
        return result

    def on_delivery_event(
        self,
        event: DeliveryEvent,
    ) -> RU22DeliveryResult | None:
        if not isinstance(event, DeliveryEvent):
            raise TypeError("event must be a DeliveryEvent")
        return self.on_verified_deliver(
            event.qid,
            event.entries,
            ordered_vertex=event.ordered_vertex,
            committed_wave=event.committed_wave,
        )

    def new_vps_for(self, qid: Qid | str) -> RU22VPS | None:
        encoded = qid.encode() if isinstance(qid, Qid) else str(qid)
        return self._new_vps.get(encoded)


def _delivery_matches_selection(
    context: RU22Context,
    delivery_result: RU22DeliveryResult,
    selection: RU22JustifiedSelection,
) -> bool:
    if (
        delivery_result.qid != context.encode_qid()
        or delivery_result.context_digest != context.context_digest
        or delivery_result.parent_vps_id != selection.parent_vps_id
        or delivery_result.selected_mids
        != tuple(entry[0] for entry in selection.entries)
        or delivery_result.selected_role_ids
        != selection.selected_role_ids
        or delivery_result.selected_eval_points
        != selection.selected_eval_points
        or delivery_result.lagrange_coefficients
        != selection.lagrange_coefficients
        or delivery_result.selected_pvr_digests
        != tuple(pvr.pvr_digest for pvr in selection.pvrs)
    ):
        return False
    return tuple(
        item.to_obj() for item in delivery_result.combined_c_v_vector
    ) == tuple(
        item.to_obj() for item in selection.combined_c_v_vector
    )


def _resolve_delivery_selection(
    context: RU22Context,
    delivery_result: RU22DeliveryResult,
    entries: Sequence[PoolEntry],
) -> RU22JustifiedSelection:
    """Bind exact public PVR digests without repeating proof verification."""

    normalized = RU22DPSSJustifierManager._normalized_exact_entries(entries)
    if normalized is None or len(normalized) != (
        context.resharing_threshold + 1
    ):
        raise ValueError(
            "entries do not form the delivered RU22 threshold set"
        )
    if {mid for mid, _payload in normalized} != set(
        delivery_result.selected_mids
    ):
        raise ValueError("entries do not match delivered RU22 mids")

    by_pvr_digest = {}
    for entry in normalized:
        mid, payload = entry
        try:
            batch = RU22ContributionBatch.from_bytes(payload)
            if mid != f"node_{batch.holder_role_id}":
                raise ValueError("batch mid does not bind holder")
            validate_ru22_contribution_batch_shape(context, batch)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "successor entries contain a malformed RU22 batch"
            ) from exc
        for pvr in batch.pvr_entries:
            if pvr.pvr_digest in by_pvr_digest:
                raise ValueError(
                    "successor entries contain a duplicate PVR digest"
                )
            by_pvr_digest[pvr.pvr_digest] = (entry, batch, pvr)

    selected_entries = []
    selected_batches = []
    selected_pvrs = []
    for index, digest in enumerate(
        delivery_result.selected_pvr_digests
    ):
        try:
            entry, batch, pvr = by_pvr_digest[digest]
        except KeyError as exc:
            raise ValueError(
                "successor entries do not contain public result PVR"
            ) from exc
        if (
            entry[0] != delivery_result.selected_mids[index]
            or pvr.parent_vps_id != delivery_result.parent_vps_id
            or pvr.dealer_role_id
            != delivery_result.selected_role_ids[index]
            or pvr.dealer_eval_point
            != delivery_result.selected_eval_points[index]
        ):
            raise ValueError(
                "delivery result is not justified by successor PVR identity"
            )
        selected_entries.append(entry)
        selected_batches.append(batch)
        selected_pvrs.append(pvr)

    coefficients = tuple(
        lagrange_coeff_at_zero(
            delivery_result.selected_eval_points, point
        )
        for point in delivery_result.selected_eval_points
    )
    combined = tuple(
        RU22ElGamalCiphertext.from_obj(item)
        for item in interpolate_c_v_vector(
            delivery_result.selected_eval_points,
            [
                pvr.pvss_transcript.derived_c_v_vector
                for pvr in selected_pvrs
            ],
            context.resharing_threshold,
        )
    )
    selection = RU22JustifiedSelection(
        parent_vps_id=delivery_result.parent_vps_id,
        entries=tuple(selected_entries),
        batches=tuple(selected_batches),
        pvrs=tuple(selected_pvrs),
        selected_role_ids=tuple(delivery_result.selected_role_ids),
        selected_eval_points=tuple(
            delivery_result.selected_eval_points
        ),
        lagrange_coefficients=coefficients,
        combined_c_v_vector=combined,
    )
    if not _delivery_matches_selection(
        context, delivery_result, selection
    ):
        raise ValueError(
            "delivery result is not justified by the RU22 threshold set"
        )
    return selection


@dataclass(frozen=True)
class RU22SuccessorResult:
    """Private successor share plus its unique DAG-certified VPS."""

    qid: str | Qid
    context_digest: str
    my_node_id: int
    my_role_idx: int
    share_s: int
    share_s_hat: int
    parent_vps_id: str
    selected_pvr_digests: Sequence[str]
    new_vps: RU22VPS
    version: int = SUCCESSOR_RESULT_VERSION

    def __post_init__(self) -> None:
        version = _strict_int(self.version, name="version")
        if version != SUCCESSOR_RESULT_VERSION:
            raise ValueError("unsupported RU22 successor-result version")
        context_digest = _digest(
            self.context_digest, name="context_digest"
        )
        qid = _canonical_embedded_qid(
            self.qid, context_digest=context_digest
        )
        if not isinstance(self.new_vps, RU22VPS):
            raise TypeError("new_vps must be an RU22VPS")
        for name, item in (
            ("version", version),
            ("qid", qid),
            ("context_digest", context_digest),
            (
                "my_node_id",
                _strict_int(self.my_node_id, name="my_node_id"),
            ),
            (
                "my_role_idx",
                _strict_int(self.my_role_idx, name="my_role_idx"),
            ),
            ("share_s", _scalar(self.share_s, name="share_s")),
            (
                "share_s_hat",
                _scalar(self.share_s_hat, name="share_s_hat"),
            ),
            (
                "parent_vps_id",
                _digest(self.parent_vps_id, name="parent_vps_id"),
            ),
            (
                "selected_pvr_digests",
                _digest_tuple(
                    self.selected_pvr_digests,
                    name="selected_pvr_digests",
                ),
            ),
            (
                "new_vps",
                RU22VPS.from_bytes(self.new_vps.to_bytes()),
            ),
        ):
            object.__setattr__(self, name, item)

    def to_local_vps_share(self) -> RU22LocalVPSShare:
        return RU22LocalVPSShare(
            vps=self.new_vps,
            share_s=self.share_s,
            share_s_hat=self.share_s_hat,
        )

    def to_holder_state(
        self,
        recipient_secret_key: int,
    ) -> RU22HolderState:
        return RU22HolderState(
            holder_role_id=self.my_node_id,
            holder_eval_point=self.my_role_idx + 1,
            recipient_index=self.my_role_idx,
            recipient_secret_key=recipient_secret_key,
            resharing_n=len(self.new_vps.encrypted_commitments),
            local_vps_shares=[self.to_local_vps_share()],
        )


def reconstruct_ru22_successor(
    *,
    context: RU22Context,
    delivery_result: RU22DeliveryResult,
    selection: RU22JustifiedSelection,
    my_node_id: int,
    my_role_idx: int,
    my_rsk: int,
    stage_observer=None,
) -> RU22SuccessorResult:
    """Decrypt and interpolate one successor's share from the fixed public T."""

    if not isinstance(context, RU22Context):
        raise TypeError("context must be an RU22Context")
    if not isinstance(delivery_result, RU22DeliveryResult):
        raise TypeError("delivery_result must be an RU22DeliveryResult")
    if not isinstance(selection, RU22JustifiedSelection):
        raise TypeError("selection must be an RU22JustifiedSelection")
    my_node_id = _strict_int(my_node_id, name="my_node_id")
    my_role_idx = _strict_int(my_role_idx, name="my_role_idx")
    my_rsk = _scalar(my_rsk, name="my_rsk", nonzero=True)
    if context.successor_role_indices.get(my_node_id) != my_role_idx:
        raise ValueError("successor node/role does not match context")
    if not 0 <= my_role_idx < context.resharing_n:
        raise ValueError("successor role index is out of range")
    if derive_public_key(
        my_rsk, context.successor_key_base_hex
    ) != context.successor_rpk_list[my_role_idx]:
        raise ValueError("my_rsk does not match successor public key")

    started = time.perf_counter()
    if not _delivery_matches_selection(
        context, delivery_result, selection
    ):
        raise ValueError(
            "delivery result does not match the justified threshold set"
        )
    new_vps = delivery_result.build_new_vps(context)
    if stage_observer is not None:
        stage_observer(
            "public_recompute", time.perf_counter() - started
        )

    started = time.perf_counter()
    share_pairs = [
        pvss_deliver(
            context.g_hex,
            context.h_hex,
            pvr.pvss_transcript,
            my_rsk,
            my_role_idx,
        )
        for pvr in selection.pvrs
    ]
    if stage_observer is not None:
        stage_observer("pvss_decrypt", time.perf_counter() - started)

    started = time.perf_counter()
    share_s, share_s_hat = pvss_interpolate(
        selection.selected_eval_points,
        share_pairs,
        context.resharing_threshold,
    )
    if stage_observer is not None:
        stage_observer(
            "share_interpolation", time.perf_counter() - started
        )

    started = time.perf_counter()
    commitment = pedersen_commit(
        context.g_hex,
        context.h_hex,
        share_s,
        share_s_hat,
    )
    decrypted_combined = decrypt_elgamal_ciphertext(
        my_rsk,
        delivery_result.combined_c_v_vector[
            my_role_idx
        ].to_obj(),
    )
    if G1Point.from_hex(commitment) != G1Point.from_hex(
        decrypted_combined
    ):
        raise ValueError(
            "reconstructed successor share does not open combined c_v"
        )
    if stage_observer is not None:
        stage_observer(
            "combined_c_v_check", time.perf_counter() - started
        )
    return RU22SuccessorResult(
        qid=context.encode_qid(),
        context_digest=context.context_digest,
        my_node_id=my_node_id,
        my_role_idx=my_role_idx,
        share_s=share_s,
        share_s_hat=share_s_hat,
        parent_vps_id=selection.parent_vps_id,
        selected_pvr_digests=tuple(
            pvr.pvr_digest for pvr in selection.pvrs
        ),
        new_vps=new_vps,
    )


class RU22SuccessorReconstructionManager:
    """Idempotent local successor reconstruction for finalized RU22 output."""

    def __init__(self, justifier: RU22DPSSJustifierManager) -> None:
        if not isinstance(justifier, RU22DPSSJustifierManager):
            raise TypeError(
                "justifier must be an RU22DPSSJustifierManager"
            )
        self.justifier = justifier
        self.registrations: dict[str, dict[str, Any]] = {}
        self.results: dict[str, RU22SuccessorResult] = {}
        self._delivery_digests: dict[str, str] = {}
        self._consumed_results: set[str] = set()
        self._metrics = {
            stage: {"invocations": 0, "elapsed_sec": 0.0}
            for stage in (
                "public_recompute",
                "pvss_decrypt",
                "share_interpolation",
                "combined_c_v_check",
            )
        }
        self._total_metrics = {
            "invocations": 0,
            "elapsed_sec": 0.0,
            "cpu_seconds": 0.0,
        }

    def _record_stage(self, stage: str, elapsed: float) -> None:
        metric = self._metrics[stage]
        metric["invocations"] += 1
        metric["elapsed_sec"] += float(elapsed)

    def metrics_snapshot(self) -> dict[str, Any]:
        stages = {
            stage: {
                "invocations": int(metric["invocations"]),
                "elapsed_sec": float(metric["elapsed_sec"]),
            }
            for stage, metric in self._metrics.items()
        }
        stages["total"] = {
            "invocations": int(self._total_metrics["invocations"]),
            "elapsed_sec": float(self._total_metrics["elapsed_sec"]),
            "cpu_seconds": float(self._total_metrics["cpu_seconds"]),
        }
        return stages

    def register_context(
        self,
        context: RU22Context,
        *,
        my_node_id: int,
        my_rsk: int | None = None,
        on_new_share=None,
    ) -> None:
        if not isinstance(context, RU22Context):
            raise TypeError("context must be an RU22Context")
        registered = self.justifier._registration(context.build_qid())
        if registered != context:
            raise ValueError("context does not match RU22 Justifier")
        my_node_id = _strict_int(my_node_id, name="my_node_id")
        role_idx = context.successor_role_indices.get(my_node_id)
        if role_idx is None:
            if my_rsk is not None:
                raise ValueError(
                    "non-successor observer must not register an rsk"
                )
            normalized_rsk = None
        else:
            if my_rsk is None:
                raise ValueError("successor role requires my_rsk")
            normalized_rsk = _scalar(
                my_rsk, name="my_rsk", nonzero=True
            )
            if derive_public_key(
                normalized_rsk, context.successor_key_base_hex
            ) != context.successor_rpk_list[role_idx]:
                raise ValueError(
                    "my_rsk does not match successor role public key"
                )
        if on_new_share is not None and not callable(on_new_share):
            raise TypeError("on_new_share must be callable")
        registration = {
            "context": context,
            "my_node_id": my_node_id,
            "my_role_idx": role_idx,
            "my_rsk": normalized_rsk,
            "on_new_share": on_new_share,
        }
        existing = self.registrations.get(context.context_digest)
        if existing is not None:
            if existing != registration:
                raise ValueError(
                    "conflicting RU22 successor registration"
                )
            return
        self.registrations[context.context_digest] = registration

    def reconstruct(
        self,
        delivery_result: RU22DeliveryResult,
        entries: Sequence[PoolEntry],
    ) -> RU22SuccessorResult | None:
        if not isinstance(delivery_result, RU22DeliveryResult):
            raise TypeError(
                "delivery_result must be an RU22DeliveryResult"
            )
        local = self.registrations.get(
            delivery_result.context_digest
        )
        if local is None:
            return None
        context = local["context"]
        if delivery_result.qid != context.encode_qid():
            raise ValueError(
                "delivery result qid does not match registered context"
            )
        if local["my_role_idx"] is None:
            return None
        selection = _resolve_delivery_selection(
            context, delivery_result, entries
        )
        existing = self.results.get(context.context_digest)
        if existing is not None:
            if self._delivery_digests.get(
                context.context_digest
            ) != delivery_result.delivery_result_digest:
                raise ValueError(
                    "conflicting threshold set for completed reconstruction"
                )
            return existing
        started = time.perf_counter()
        cpu_started = time.process_time()
        try:
            result = reconstruct_ru22_successor(
                context=context,
                delivery_result=delivery_result,
                selection=selection,
                my_node_id=local["my_node_id"],
                my_role_idx=local["my_role_idx"],
                my_rsk=local["my_rsk"],
                stage_observer=self._record_stage,
            )
        finally:
            self._total_metrics["invocations"] += 1
            self._total_metrics["elapsed_sec"] += (
                time.perf_counter() - started
            )
            self._total_metrics["cpu_seconds"] += (
                time.process_time() - cpu_started
            )
        self.results[context.context_digest] = result
        self._delivery_digests[context.context_digest] = (
            delivery_result.delivery_result_digest
        )
        callback = local["on_new_share"]
        if callback is not None:
            callback(result.share_s, result.share_s_hat, result.new_vps)
        return result

    def consume_result_to_holder_state(
        self,
        context_or_digest: RU22Context | str,
    ) -> RU22HolderState:
        digest = (
            context_or_digest.context_digest
            if isinstance(context_or_digest, RU22Context)
            else _digest(context_or_digest, name="context_digest")
        )
        if digest in self._consumed_results:
            raise RuntimeError(
                "RU22 successor result already consumed"
            )
        try:
            result = self.results.pop(digest)
            registration = self.registrations[digest]
        except KeyError as exc:
            raise RuntimeError(
                "no reconstructed RU22 successor result to consume"
            ) from exc
        rsk = registration["my_rsk"]
        if rsk is None:
            raise RuntimeError("successor secret key is unavailable")
        registration["my_rsk"] = None
        self._delivery_digests.pop(digest, None)
        self._consumed_results.add(digest)
        return result.to_holder_state(rsk)


__all__ = [
    "BATCH_KIND",
    "BATCH_VERSION",
    "CONTEXT_KIND",
    "CONTEXT_VERSION",
    "DELIVERY_RESULT_KIND",
    "DELIVERY_RESULT_VERSION",
    "PVR_KIND",
    "PVR_VERSION",
    "QID_KIND",
    "VPS_KIND",
    "VPS_VERSION",
    "RU22Context",
    "RU22ContributionBatch",
    "RU22DAGDeliveryCertificate",
    "RU22DeliveryResult",
    "RU22ElGamalCiphertext",
    "RU22HolderActionResult",
    "RU22HolderMetrics",
    "RU22HolderState",
    "RU22InitialCertificate",
    "RU22JustifiedSelection",
    "RU22LocalVPSShare",
    "RU22PVR",
    "RU22DPSSJustifierManager",
    "RU22PublicFinalizer",
    "RU22SuccessorReconstructionManager",
    "RU22SuccessorResult",
    "RU22ValidatedBatch",
    "RU22VPS",
    "make_ru22_contribution_batch",
    "reconstruct_ru22_successor",
    "run_ru22_handover_role",
    "validate_ru22_contribution_batch_shape",
]
