from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import base64
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import logging
import pickle
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from beaver.protocol_trace import DeterministicIdealCoin, IdealCoin


COIN_PROTOCOL_DOMAIN = "ATCAAS-COMMON-COIN-V1"
COIN_SHARE_NETWORK_TAG = "ATCAAS_COIN_SHARE_V1"
COIN_SHARE_MESSAGE_VERSION = 1
COIN_BLS_HASH_DOMAIN = "ATCAAS-COMMON-COIN-BLS-V1:"
COIN_CONFIG_VERSION = 1

_COIN_ID_ENCODING_DOMAIN = b"ATCaaS/common-coin/id/v1\x00"
_COIN_ID_DIGEST_DOMAIN = b"ATCaaS/common-coin/id-digest/v1\x00"
_COIN_MEMBERSHIP_DOMAIN = b"ATCaaS/common-coin/membership/v1\x00"
_COIN_PUBLIC_KEY_DOMAIN = b"ATCaaS/common-coin/public-key/v1\x00"
_COIN_OUTPUT_DOMAIN = b"ATCaaS/common-coin/output/v1\x00"
_COIN_SHARE_WIRE_MAGIC = b"ATCAAS-COIN-SHARE-V1\x00"
_DIGEST_BYTES = hashlib.sha256().digest_size
_MAX_U64 = (1 << 64) - 1
MAX_COIN_SHARE_BYTES = 4096


def _require_plain_int(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _require_bytes(value, *, name: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be bytes-like")
    return bytes(value)


def _frame(value: bytes) -> bytes:
    """Length-prefix one value for an unambiguous canonical encoding."""

    if len(value) > _MAX_U64:
        raise ValueError("canonical coin field is too large")
    return len(value).to_bytes(8, "big") + value


def _encode_u64(value: int, *, name: str) -> bytes:
    value = _require_plain_int(value, name=name)
    if not 0 <= value <= _MAX_U64:
        raise ValueError(f"{name} must be between 0 and 2^64-1")
    return value.to_bytes(8, "big")


def coin_public_key_digest(public_key_bytes: bytes) -> bytes:
    """Bind membership metadata to the exact serialized coin public key."""

    public_key_bytes = _require_bytes(
        public_key_bytes,
        name="coin public key",
    )
    if not public_key_bytes:
        raise ValueError("coin public key cannot be empty")
    return hashlib.sha256(_COIN_PUBLIC_KEY_DOMAIN + public_key_bytes).digest()


def _canonical_committee(
    committee: Iterable[int],
    *,
    network_size: int,
) -> Tuple[int, ...]:
    if isinstance(committee, (str, bytes, bytearray, memoryview)):
        raise TypeError("coin committee must be an iterable of node ids")
    try:
        members = tuple(committee)
    except TypeError as exc:
        raise TypeError("coin committee must be an iterable of node ids") from exc
    if not members:
        raise ValueError("coin committee cannot be empty")
    for member in members:
        _require_plain_int(member, name="coin committee member")
        if not 0 <= member < network_size:
            raise ValueError(
                f"coin committee member {member} is outside network range "
                f"0..{network_size - 1}"
            )
    if len(set(members)) != len(members):
        raise ValueError("coin committee contains duplicate members")
    if members != tuple(sorted(members)):
        raise ValueError("coin committee must be in strictly increasing order")
    return members


def coin_membership_digest(
    *,
    network_size: int,
    committee: Iterable[int],
    threshold: int,
    public_key_digest: bytes,
) -> bytes:
    """Hash the public configuration that defines one coin membership."""

    network_size = _require_plain_int(network_size, name="network_size")
    if not 1 <= network_size <= _MAX_U64:
        raise ValueError("network_size must be between 1 and 2^64-1")
    members = _canonical_committee(committee, network_size=network_size)
    threshold = _require_plain_int(threshold, name="coin threshold")
    if not 1 <= threshold <= len(members):
        raise ValueError(
            "coin threshold must be between 1 and the committee size"
        )
    public_key_digest = _require_bytes(
        public_key_digest,
        name="coin public-key digest",
    )
    if len(public_key_digest) != _DIGEST_BYTES:
        raise ValueError(
            f"coin public-key digest must be {_DIGEST_BYTES} bytes"
        )

    encoded_members = b"".join(
        _encode_u64(member, name="coin committee member")
        for member in members
    )
    encoded = b"".join(
        (
            _COIN_MEMBERSHIP_DOMAIN,
            _frame(_encode_u64(network_size, name="network_size")),
            _frame(encoded_members),
            _frame(_encode_u64(threshold, name="coin threshold")),
            _frame(public_key_digest),
        )
    )
    return hashlib.sha256(encoded).digest()


@dataclass(frozen=True)
class CoinCommitteeConfig:
    """Validated public committee metadata for one local network node.

    ``private_share_index`` records which threshold-key share this node is
    allowed to load.  The private key itself is deliberately outside this
    Phase-0 data model and will be loaded by the threshold oracle later.
    """

    network_size: int
    my_id: int
    committee: Tuple[int, ...]
    threshold: int
    public_key_digest: bytes
    private_share_index: Optional[int] = None

    def __post_init__(self) -> None:
        network_size = _require_plain_int(
            self.network_size,
            name="network_size",
        )
        if not 1 <= network_size <= _MAX_U64:
            raise ValueError("network_size must be between 1 and 2^64-1")
        my_id = _require_plain_int(self.my_id, name="my_id")
        if not 0 <= my_id < network_size:
            raise ValueError(
                f"my_id must be inside network range 0..{network_size - 1}"
            )
        members = _canonical_committee(
            self.committee,
            network_size=network_size,
        )
        threshold = _require_plain_int(
            self.threshold,
            name="coin threshold",
        )
        if not 1 <= threshold <= len(members):
            raise ValueError(
                "coin threshold must be between 1 and the committee size"
            )
        public_key_digest = _require_bytes(
            self.public_key_digest,
            name="coin public-key digest",
        )
        if len(public_key_digest) != _DIGEST_BYTES:
            raise ValueError(
                f"coin public-key digest must be {_DIGEST_BYTES} bytes"
            )

        expected_share_index = (
            members.index(my_id) if my_id in members else None
        )
        share_index = self.private_share_index
        if expected_share_index is None:
            if share_index is not None:
                raise ValueError(
                    "non-committee node must not have a private share index"
                )
        else:
            if share_index is None:
                raise ValueError(
                    "committee node requires a private share index"
                )
            share_index = _require_plain_int(
                share_index,
                name="private share index",
            )
            if share_index != expected_share_index:
                raise ValueError(
                    "private share index does not match the node's position "
                    "in the coin committee"
                )

        object.__setattr__(self, "committee", members)
        object.__setattr__(self, "public_key_digest", public_key_digest)

    @property
    def is_member(self) -> bool:
        return self.my_id in self.committee

    def share_index_for(self, node_id: int) -> int:
        node_id = _require_plain_int(node_id, name="node_id")
        try:
            return self.committee.index(node_id)
        except ValueError as exc:
            raise KeyError(
                f"node {node_id} is not in the coin committee"
            ) from exc

    @property
    def membership_digest(self) -> bytes:
        return coin_membership_digest(
            network_size=self.network_size,
            committee=self.committee,
            threshold=self.threshold,
            public_key_digest=self.public_key_digest,
        )


@dataclass(frozen=True)
class CoinId:
    """Canonical identifier signed by the threshold coin committee."""

    epoch: str
    membership_digest: bytes
    session_id: str
    attempt: int

    def __post_init__(self) -> None:
        if not isinstance(self.epoch, str):
            raise TypeError("coin epoch must be a string")
        if not self.epoch:
            raise ValueError("coin epoch cannot be empty")
        if not isinstance(self.session_id, str):
            raise TypeError("coin session_id must be a string")
        if not self.session_id:
            raise ValueError("coin session_id cannot be empty")
        membership_digest = _require_bytes(
            self.membership_digest,
            name="coin membership digest",
        )
        if len(membership_digest) != _DIGEST_BYTES:
            raise ValueError(
                f"coin membership digest must be {_DIGEST_BYTES} bytes"
            )
        _encode_u64(self.attempt, name="coin attempt")
        object.__setattr__(self, "membership_digest", membership_digest)

    def encode(self) -> bytes:
        return b"".join(
            (
                _COIN_ID_ENCODING_DOMAIN,
                _frame(COIN_PROTOCOL_DOMAIN.encode("ascii")),
                _frame(self.epoch.encode("utf-8")),
                _frame(self.membership_digest),
                _frame(self.session_id.encode("utf-8")),
                _frame(_encode_u64(self.attempt, name="coin attempt")),
            )
        )

    @property
    def digest(self) -> bytes:
        return hashlib.sha256(_COIN_ID_DIGEST_DOMAIN + self.encode()).digest()


def encode_coin_id(
    *,
    epoch: str,
    membership_digest: bytes,
    session_id: str,
    attempt: int,
) -> bytes:
    return CoinId(
        epoch=epoch,
        membership_digest=membership_digest,
        session_id=session_id,
        attempt=attempt,
    ).encode()


def coin_bls_message(coin_id: CoinId) -> str:
    """Encode a CoinId for Charm's string-oriented hash-to-group backend.

    The MNT224 backend treats embedded NUL bytes as a terminator.  Hex is an
    injective encoding of the canonical binary CoinId and contains no NUL,
    while the prefix keeps this BLS use separate from other signatures.
    """

    if not isinstance(coin_id, CoinId):
        raise TypeError("coin_id must be a CoinId")
    return COIN_BLS_HASH_DOMAIN + coin_id.encode().hex()


@dataclass(frozen=True)
class CoinShareMessage:
    """Strict Phase-0 wire value; signer identity comes from transport."""

    coin_id_digest: bytes
    share: bytes

    def __post_init__(self) -> None:
        coin_id_digest = _require_bytes(
            self.coin_id_digest,
            name="coin id digest",
        )
        if len(coin_id_digest) != _DIGEST_BYTES:
            raise ValueError(f"coin id digest must be {_DIGEST_BYTES} bytes")
        share = _require_bytes(self.share, name="coin signature share")
        if not share:
            raise ValueError("coin signature share cannot be empty")
        if len(share) > MAX_COIN_SHARE_BYTES:
            raise ValueError(
                f"coin signature share exceeds {MAX_COIN_SHARE_BYTES} bytes"
            )
        object.__setattr__(self, "coin_id_digest", coin_id_digest)
        object.__setattr__(self, "share", share)

    def to_bytes(self) -> bytes:
        return b"".join(
            (
                _COIN_SHARE_WIRE_MAGIC,
                self.coin_id_digest,
                len(self.share).to_bytes(4, "big"),
                self.share,
            )
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> "CoinShareMessage":
        raw = _require_bytes(raw, name="coin share message")
        header_size = len(_COIN_SHARE_WIRE_MAGIC) + _DIGEST_BYTES + 4
        if len(raw) < header_size:
            raise ValueError("coin share message is truncated")
        if not raw.startswith(_COIN_SHARE_WIRE_MAGIC):
            raise ValueError("coin share message has an unsupported version")
        offset = len(_COIN_SHARE_WIRE_MAGIC)
        coin_id_digest = raw[offset : offset + _DIGEST_BYTES]
        offset += _DIGEST_BYTES
        share_size = int.from_bytes(raw[offset : offset + 4], "big")
        offset += 4
        if share_size == 0:
            raise ValueError("coin signature share cannot be empty")
        if share_size > MAX_COIN_SHARE_BYTES:
            raise ValueError(
                f"coin signature share exceeds {MAX_COIN_SHARE_BYTES} bytes"
            )
        if len(raw) != offset + share_size:
            raise ValueError("coin share message length does not match payload")
        return cls(coin_id_digest=coin_id_digest, share=raw[offset:])


def encode_coin_share_message(
    coin_id_digest: bytes,
    share: bytes,
) -> bytes:
    return CoinShareMessage(coin_id_digest, share).to_bytes()


def decode_coin_share_message(raw: bytes) -> CoinShareMessage:
    return CoinShareMessage.from_bytes(raw)


class CoinOracle(ABC):
    """Interface for the coin functionality used by Async YOSO SelectBlock.

    Phase B0/B1 only needs the interface.  Later phases will route all coin
    uses through this object so the deterministic stub can be replaced by a
    threshold/FCF-backed implementation without touching protocol logic.
    """

    @abstractmethod
    async def next_coin(self, session_id: str, attempt: int) -> bytes:
        raise NotImplementedError


class CoinWaitTimeout(TimeoutError):
    """A protocol call timed out specifically while waiting for common coin."""


class ObservedCoinOracle(CoinOracle):
    """Add bounded waiting and serializable observations to any coin oracle.

    This adapter deliberately leaves the ACS call site unchanged.  It is used
    by the Phase-4A runner to distinguish a common-coin timeout from an ACS
    network-stage timeout and to expose cross-node coin/leader agreement in
    experiment summaries without retaining the coin bytes themselves.
    """

    _OUTPUT_DIGEST_DOMAIN = b"ATCaaS/common-coin/observation/v1\x00"

    def __init__(
        self,
        delegate: CoinOracle,
        *,
        timeout: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not isinstance(delegate, CoinOracle):
            raise TypeError("delegate must implement CoinOracle")
        if timeout is not None:
            timeout = float(timeout)
            if timeout <= 0:
                raise ValueError("coin timeout must be positive")
        self.delegate = delegate
        self.timeout = timeout
        self.log = logger or logging.getLogger("observed_common_coin")
        self._observations: List[dict] = []

    def _coin_id_digest(self, session_id: str, attempt: int) -> Optional[str]:
        factory = getattr(self.delegate, "coin_id", None)
        if not callable(factory):
            return None
        return factory(session_id, attempt).digest.hex()

    async def next_coin(self, session_id: str, attempt: int) -> bytes:
        started = time.perf_counter()
        record = {
            "session_id": str(session_id),
            "attempt": int(attempt),
            "coin_id_digest": self._coin_id_digest(session_id, attempt),
            "status": "waiting",
            "wait_ms": None,
            "coin_digest": None,
        }
        self.log.info(
            "[COMMON-COIN] waiting session=%s attempt=%s timeout=%s",
            session_id,
            attempt,
            self.timeout,
        )
        try:
            request = self.delegate.next_coin(session_id, attempt)
            coin = (
                await request
                if self.timeout is None
                else await asyncio.wait_for(request, timeout=self.timeout)
            )
            coin = _require_bytes(coin, name="coin output")
            record["status"] = "ready"
            record["coin_digest"] = hashlib.sha256(
                self._OUTPUT_DIGEST_DOMAIN + coin
            ).hexdigest()
            return coin
        except asyncio.TimeoutError as exc:
            record["status"] = "timeout"
            raise CoinWaitTimeout(
                f"common coin timed out for session={session_id!r} "
                f"attempt={attempt} after {self.timeout}s"
            ) from exc
        except BaseException as exc:
            record["status"] = "cancelled" if isinstance(
                exc, asyncio.CancelledError
            ) else "error"
            record["error"] = type(exc).__name__
            raise
        finally:
            record["wait_ms"] = (time.perf_counter() - started) * 1000.0
            self._observations.append(record)
            self.log.log(
                logging.INFO if record["status"] == "ready" else logging.ERROR,
                "[COMMON-COIN] %s session=%s attempt=%s wait_ms=%.3f digest=%s",
                record["status"],
                session_id,
                attempt,
                record["wait_ms"],
                record["coin_digest"],
            )

    def snapshot(self) -> dict:
        status_counts: Dict[str, int] = defaultdict(int)
        for record in self._observations:
            status_counts[record["status"]] += 1
        return {
            "timeout_seconds": self.timeout,
            "calls": len(self._observations),
            "status_counts": dict(sorted(status_counts.items())),
            "observations": [dict(record) for record in self._observations],
        }


@dataclass
class ThresholdCoinMetrics:
    coin_calls: int = 0
    shares_signed: int = 0
    shares_sent: int = 0
    shares_received: int = 0
    shares_valid: int = 0
    shares_invalid: int = 0
    shares_duplicate: int = 0
    combine_count: int = 0
    cache_hits: int = 0

    def snapshot(self) -> dict:
        return {
            "coin_calls": self.coin_calls,
            "shares_signed": self.shares_signed,
            "shares_sent": self.shares_sent,
            "shares_received": self.shares_received,
            "shares_valid": self.shares_valid,
            "shares_invalid": self.shares_invalid,
            "shares_duplicate": self.shares_duplicate,
            "combine_count": self.combine_count,
            "cache_hits": self.cache_hits,
        }


class ThresholdBLSCoinOracle(CoinOracle):
    """Long-lived threshold-BLS implementation of the common-coin API.

    The transport is deliberately small: ``send(dest, (tag, payload))`` is
    the authenticated point-to-point sender already used by the top-level
    runners, and ``shared_subscribe`` is the tuple returned by
    :func:`beaver.utils.misc.subscribe_recv`.  The signer identity is always
    the sender returned by that dispatcher; it is not carried in the wire
    payload.

    The Boldyreva backend is imported lazily so users of the ideal coin do
    not load Charm or the native pairing dependencies.
    """

    def __init__(
        self,
        *,
        epoch: str,
        config: CoinCommitteeConfig,
        public_key_bytes: bytes,
        public_key: Any,
        private_key: Optional[Any],
        send: Callable[[int, object], None],
        shared_subscribe: tuple,
        signing_enabled: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not isinstance(epoch, str):
            raise TypeError("coin epoch must be a string")
        if not epoch:
            raise ValueError("coin epoch cannot be empty")
        if not isinstance(config, CoinCommitteeConfig):
            raise TypeError("config must be a CoinCommitteeConfig")
        if not callable(send):
            raise TypeError("coin send must be callable")
        if not isinstance(shared_subscribe, tuple) or len(shared_subscribe) != 2:
            raise TypeError("shared_subscribe must be a (task, subscribe) tuple")
        if not callable(shared_subscribe[1]):
            raise TypeError("shared_subscribe subscribe entry must be callable")
        if not isinstance(signing_enabled, bool):
            raise TypeError("signing_enabled must be a boolean")

        public_key_bytes = _require_bytes(
            public_key_bytes,
            name="coin public key",
        )
        if coin_public_key_digest(public_key_bytes) != config.public_key_digest:
            raise ValueError(
                "coin public key does not match the configured public-key digest"
            )
        self._validate_public_key(public_key, config)
        self._validate_private_key(private_key, public_key, config)

        self.epoch = epoch
        self.config = config
        self.public_key = public_key
        self.private_key = private_key
        self.send = send
        self.shared_subscribe = shared_subscribe
        self.signing_enabled = signing_enabled
        self.log = logger or logging.getLogger(
            f"threshold_bls_coin.{config.my_id}"
        )

        self.metrics = ThresholdCoinMetrics()
        self._receive = None
        self._receive_task: Optional[asyncio.Task] = None
        self._started = False
        self._closed = False
        self._coin_ids: Dict[bytes, CoinId] = {}
        self._result_futures: Dict[bytes, asyncio.Future] = {}
        self._completed: Dict[bytes, bytes] = {}
        self._shares: Dict[bytes, Dict[int, Any]] = defaultdict(dict)
        self._early_shares: Dict[bytes, Dict[int, bytes]] = defaultdict(dict)
        self._processed_share_wires = set()
        self._locally_signed = set()
        self._started_at: Dict[bytes, float] = {}
        self._latencies_ms: Dict[bytes, float] = {}

    @staticmethod
    def _validate_public_key(public_key: Any, config: CoinCommitteeConfig) -> None:
        if public_key is None:
            raise ValueError("coin public key is required")
        if getattr(public_key, "l", None) != len(config.committee):
            raise ValueError(
                "coin public key participant count does not match committee size"
            )
        if getattr(public_key, "k", None) != config.threshold:
            raise ValueError(
                "coin public key threshold does not match configured threshold"
            )
        verification_keys = getattr(public_key, "VKs", None)
        if verification_keys is None or len(verification_keys) != len(
            config.committee
        ):
            raise ValueError(
                "coin public key verification-share list does not match committee"
            )
        for method_name in (
            "hash_message",
            "verify_share",
            "combine_shares",
            "verify_signature",
        ):
            if not callable(getattr(public_key, method_name, None)):
                raise TypeError(
                    f"coin public key is missing {method_name}()"
                )

    @staticmethod
    def _validate_private_key(
        private_key: Optional[Any],
        public_key: Any,
        config: CoinCommitteeConfig,
    ) -> None:
        if not config.is_member:
            if private_key is not None:
                raise ValueError(
                    "non-committee node must not load a coin private-key share"
                )
            return
        if private_key is None:
            raise ValueError(
                "coin committee node requires a private-key share"
            )
        if not callable(getattr(private_key, "sign", None)):
            raise TypeError("coin private-key share is missing sign()")
        if getattr(private_key, "i", None) != config.private_share_index:
            raise ValueError(
                "coin private-key share index does not match local committee index"
            )
        if (
            getattr(private_key, "l", None) != getattr(public_key, "l", None)
            or getattr(private_key, "k", None) != getattr(public_key, "k", None)
            or getattr(private_key, "VK", None) != getattr(public_key, "VK", None)
            or getattr(private_key, "VKs", None) != getattr(public_key, "VKs", None)
        ):
            raise ValueError(
                "coin private-key share does not belong to the configured public key"
            )

    @staticmethod
    def _serialize_signature(signature: Any) -> bytes:
        from beaver.broadcast.crypto.boldyreva import serialize

        return serialize(signature)

    @staticmethod
    def _deserialize_share(share: bytes) -> Any:
        from beaver.broadcast.crypto.boldyreva import deserialize1

        return deserialize1(share)

    @property
    def receive_task(self) -> Optional[asyncio.Task]:
        return self._receive_task

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("threshold coin oracle is closed")
        if self._started:
            return
        self._receive = self.shared_subscribe[1](COIN_SHARE_NETWORK_TAG)
        if not callable(self._receive):
            raise TypeError("coin subscription must return an async receiver")
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._started = True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        task = self._receive_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        for future in self._result_futures.values():
            if not future.done():
                future.cancel()

    async def _receive_loop(self) -> None:
        while True:
            try:
                sender, raw = await self._receive()
                self._handle_wire_share(sender, raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Network input must not terminate the long-lived service.
                self.metrics.shares_invalid += 1
                self.log.warning("discarding malformed coin share: %s", exc)

    def _handle_wire_share(self, sender: int, raw: bytes) -> None:
        self.metrics.shares_received += 1
        try:
            sender = _require_plain_int(sender, name="coin share sender")
            self.config.share_index_for(sender)
            message = decode_coin_share_message(raw)
        except (KeyError, TypeError, ValueError) as exc:
            self.metrics.shares_invalid += 1
            self.log.warning("discarding invalid coin share envelope: %s", exc)
            return

        coin_digest = message.coin_id_digest
        coin_id = self._coin_ids.get(coin_digest)
        if coin_id is None:
            early = self._early_shares[coin_digest]
            if sender in early:
                self.metrics.shares_duplicate += 1
                return
            early[sender] = message.share
            return
        self._verify_and_store_share(coin_id, sender, message.share)

    def _verify_and_store_share(
        self,
        coin_id: CoinId,
        sender: int,
        share_bytes: bytes,
    ) -> None:
        coin_digest = coin_id.digest
        if coin_digest in self._completed:
            self.metrics.shares_duplicate += 1
            return
        shares = self._shares[coin_digest]
        if sender in shares:
            self.metrics.shares_duplicate += 1
            return
        wire_key = (coin_digest, sender, hashlib.sha256(share_bytes).digest())
        if wire_key in self._processed_share_wires:
            self.metrics.shares_duplicate += 1
            return
        self._processed_share_wires.add(wire_key)
        try:
            share_index = self.config.share_index_for(sender)
            share = self._deserialize_share(share_bytes)
            message_hash = self.public_key.hash_message(
                coin_bls_message(coin_id)
            )
            self.public_key.verify_share(share, share_index, message_hash)
        except Exception as exc:
            self.metrics.shares_invalid += 1
            self.log.warning(
                "discarding invalid coin signature share from node %s: %s",
                sender,
                exc,
            )
            return

        shares[sender] = share
        self.metrics.shares_valid += 1
        if len(shares) >= self.config.threshold:
            self._combine_and_finish(coin_id)

    def _combine_and_finish(self, coin_id: CoinId) -> None:
        coin_digest = coin_id.digest
        if coin_digest in self._completed:
            return
        shares_by_sender = self._shares[coin_digest]
        selected_senders = sorted(shares_by_sender)[: self.config.threshold]
        indexed_shares = {
            self.config.share_index_for(sender): shares_by_sender[sender]
            for sender in selected_senders
        }
        try:
            signature = self.public_key.combine_shares(indexed_shares)
            message_hash = self.public_key.hash_message(
                coin_bls_message(coin_id)
            )
            self.public_key.verify_signature(signature, message_hash)
            signature_bytes = self._serialize_signature(signature)
        except Exception as exc:
            self.log.error("failed to combine verified coin shares: %s", exc)
            return

        result = hashlib.sha256(
            _COIN_OUTPUT_DOMAIN + signature_bytes
        ).digest()
        self._completed[coin_digest] = result
        self.metrics.combine_count += 1
        started_at = self._started_at.get(coin_digest)
        if started_at is not None:
            self._latencies_ms[coin_digest] = (
                time.monotonic() - started_at
            ) * 1000.0
        future = self._result_futures.get(coin_digest)
        if future is not None and not future.done():
            future.set_result(result)

    def _register_coin(self, coin_id: CoinId) -> asyncio.Future:
        coin_digest = coin_id.digest
        existing = self._coin_ids.get(coin_digest)
        if existing is not None and existing.encode() != coin_id.encode():
            raise RuntimeError("common-coin id digest collision")
        self._coin_ids[coin_digest] = coin_id
        future = self._result_futures.get(coin_digest)
        if future is None:
            future = asyncio.get_running_loop().create_future()
            self._result_futures[coin_digest] = future
            self._started_at[coin_digest] = time.monotonic()

        early = self._early_shares.pop(coin_digest, {})
        for sender, share_bytes in sorted(early.items()):
            self._verify_and_store_share(coin_id, sender, share_bytes)
        return future

    def _sign_and_broadcast_once(self, coin_id: CoinId) -> None:
        if not self.config.is_member or not self.signing_enabled:
            return
        coin_digest = coin_id.digest
        if coin_digest in self._locally_signed:
            return
        message_hash = self.public_key.hash_message(coin_bls_message(coin_id))
        signature_share = self.private_key.sign(message_hash)
        share_bytes = self._serialize_signature(signature_share)
        wire = encode_coin_share_message(coin_digest, share_bytes)
        self._locally_signed.add(coin_digest)
        self.metrics.shares_signed += 1
        for destination in range(self.config.network_size):
            self.send(destination, (COIN_SHARE_NETWORK_TAG, wire))
            self.metrics.shares_sent += 1

    def coin_id(self, session_id: str, attempt: int) -> CoinId:
        """Return the canonical identifier used for one threshold coin call."""

        return CoinId(
            epoch=self.epoch,
            membership_digest=self.config.membership_digest,
            session_id=session_id,
            attempt=attempt,
        )

    async def next_coin(self, session_id: str, attempt: int) -> bytes:
        if not self._started:
            raise RuntimeError("threshold coin oracle must be started first")
        if self._closed:
            raise RuntimeError("threshold coin oracle is closed")
        self.metrics.coin_calls += 1
        coin_id = self.coin_id(session_id, attempt)
        coin_digest = coin_id.digest
        cached = self._completed.get(coin_digest)
        if cached is not None:
            self.metrics.cache_hits += 1
            return cached

        future = self._register_coin(coin_id)
        self._sign_and_broadcast_once(coin_id)
        return await asyncio.shield(future)

    def snapshot(self) -> dict:
        result = self.metrics.snapshot()
        result.update(
            {
                "started": self._started,
                "closed": self._closed,
                "signing_enabled": self.signing_enabled,
                "pending_coins": sum(
                    1 for future in self._result_futures.values()
                    if not future.done()
                ),
                "completed_coins": len(self._completed),
                "coin_latency_ms": {
                    digest.hex(): latency
                    for digest, latency in sorted(self._latencies_ms.items())
                },
            }
        )
        return result


@dataclass(frozen=True)
class ThresholdBLSCoinKeyMaterial:
    """Validated key objects loaded from one trusted local configuration."""

    config: CoinCommitteeConfig
    public_key_bytes: bytes
    public_key: Any
    private_key: Optional[Any]


def _decode_coin_config_digest(value: Any, *, name: str) -> bytes:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a hexadecimal string")
    try:
        result = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not valid hexadecimal") from exc
    if len(result) != _DIGEST_BYTES:
        raise ValueError(f"{name} must encode {_DIGEST_BYTES} bytes")
    return result


def _decode_coin_config_key(value: Any, *, name: str) -> bytes:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a base64 string")
    try:
        result = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"{name} is not valid base64") from exc
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def load_threshold_coin_key_material(
    extras: Mapping[str, Any],
    *,
    network_size: int,
    my_id: int,
) -> ThresholdBLSCoinKeyMaterial:
    """Load and fail-fast validate the Phase-2 coin fields.

    Config files are trusted local input: the BLS objects use Python pickle,
    matching the repository's existing threshold-key configuration format.
    Never call this loader on an untrusted network payload.
    """

    if not isinstance(extras, Mapping):
        raise TypeError("coin configuration extras must be a mapping")
    try:
        version = _require_plain_int(
            extras["coin_config_version"],
            name="coin_config_version",
        )
        committee = tuple(extras["coin_committee"])
        threshold = extras["coin_threshold"]
        public_key_encoded = extras["coin_public_key"]
        public_key_digest_encoded = extras["coin_public_key_digest"]
        membership_digest_encoded = extras["coin_membership_digest"]
    except KeyError as exc:
        raise ValueError(f"missing coin configuration field {exc.args[0]!r}") from exc
    except TypeError as exc:
        raise TypeError("coin_committee must be an iterable of node ids") from exc
    if version != COIN_CONFIG_VERSION:
        raise ValueError(
            f"unsupported coin_config_version {version}; "
            f"expected {COIN_CONFIG_VERSION}"
        )

    public_key_bytes = _decode_coin_config_key(
        public_key_encoded,
        name="coin_public_key",
    )
    configured_public_digest = _decode_coin_config_digest(
        public_key_digest_encoded,
        name="coin_public_key_digest",
    )
    actual_public_digest = coin_public_key_digest(public_key_bytes)
    if configured_public_digest != actual_public_digest:
        raise ValueError(
            "coin_public_key_digest does not match coin_public_key"
        )

    is_member = my_id in committee
    has_private_key = "coin_private_key" in extras
    has_private_index = "coin_private_share_index" in extras
    if is_member and not (has_private_key and has_private_index):
        raise ValueError(
            "coin committee node requires coin_private_key and "
            "coin_private_share_index"
        )
    if not is_member and (has_private_key or has_private_index):
        raise ValueError(
            "non-committee node must not contain coin private-key fields"
        )

    config = CoinCommitteeConfig(
        network_size=network_size,
        my_id=my_id,
        committee=committee,
        threshold=threshold,
        public_key_digest=configured_public_digest,
        private_share_index=(
            extras["coin_private_share_index"] if is_member else None
        ),
    )
    configured_membership_digest = _decode_coin_config_digest(
        membership_digest_encoded,
        name="coin_membership_digest",
    )
    if configured_membership_digest != config.membership_digest:
        raise ValueError(
            "coin_membership_digest does not match the coin membership fields"
        )

    try:
        public_key = pickle.loads(public_key_bytes)
    except Exception as exc:
        raise ValueError("coin_public_key cannot be deserialized") from exc
    private_key = None
    if is_member:
        private_key_bytes = _decode_coin_config_key(
            extras["coin_private_key"],
            name="coin_private_key",
        )
        try:
            private_key = pickle.loads(private_key_bytes)
        except Exception as exc:
            raise ValueError("coin_private_key cannot be deserialized") from exc

    ThresholdBLSCoinOracle._validate_public_key(public_key, config)
    ThresholdBLSCoinOracle._validate_private_key(
        private_key,
        public_key,
        config,
    )
    return ThresholdBLSCoinKeyMaterial(
        config=config,
        public_key_bytes=public_key_bytes,
        public_key=public_key,
        private_key=private_key,
    )


class DeterministicCoinOracle(CoinOracle):
    """Seeded adapter for the ideal coin, shared by DAG and ACS index."""

    def __init__(self, epoch: str):
        self.epoch = epoch
        self.calls = 0
        self.coin = DeterministicIdealCoin(epoch)

    async def next_coin(self, session_id: str, attempt: int) -> bytes:
        self.calls += 1
        return self.coin.get_bytes(attempt)


class SharedIdealCoinOracle(CoinOracle):
    """Async ACS adapter over the same ideal-coin trace used by DAG-TOB."""

    def __init__(self, coin: IdealCoin):
        self.coin = coin
        self.calls = 0

    async def next_coin(self, session_id: str, attempt: int) -> bytes:
        self.calls += 1
        return self.coin.get_bytes(attempt)


class SessionCoinIndexAllocator:
    """Reserve disjoint logical coin-index windows for ACS sessions.

    A session reserves ``max_rounds`` consecutive indices.  Therefore a
    later batch cannot reuse an earlier batch's round-1 coin even when some
    nodes stop the earlier session before consuming every reserved index.
    """

    def __init__(self) -> None:
        self._next_index = 1
        self._windows: Dict[str, dict] = {}
        self._uses: List[dict] = []

    def reserve(self, session_id: str, max_rounds: int) -> dict:
        session_id = str(session_id)
        max_rounds = int(max_rounds)
        if not session_id:
            raise ValueError("coin session_id cannot be empty")
        if max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if session_id in self._windows:
            raise ValueError(
                f"coin session {session_id!r} was already reserved"
            )
        window = {
            "session_id": session_id,
            "start_index": self._next_index,
            "max_rounds": max_rounds,
        }
        self._windows[session_id] = window
        self._next_index += max_rounds
        return dict(window)

    def index(self, session_id: str, round_id: int) -> int:
        try:
            window = self._windows[str(session_id)]
        except KeyError as exc:
            raise KeyError(
                f"coin session {str(session_id)!r} is not reserved"
            ) from exc
        round_id = int(round_id)
        if not 1 <= round_id <= window["max_rounds"]:
            raise ValueError(
                f"coin round {round_id} outside reserved window "
                f"1..{window['max_rounds']}"
            )
        coin_index = window["start_index"] + round_id - 1
        record = {
            "session_id": str(session_id),
            "round": round_id,
            "coin_index": coin_index,
        }
        if record not in self._uses:
            self._uses.append(record)
        return coin_index

    def snapshot(self) -> dict:
        return {
            "windows": [
                dict(window)
                for _session, window in sorted(
                    self._windows.items(),
                    key=lambda item: item[1]["start_index"],
                )
            ],
            "uses": list(self._uses),
            "unique_indices": len(
                {record["coin_index"] for record in self._uses}
            ),
        }
