"""
YOSO-style Reliable Broadcast (RBC) — plain Bracha variant (no RS codes).

Uses standard Bracha RBC semantics: SEND, ECHO, and READY all carry
the full message.  The SHA-256 hash is kept only as a compact local key
for counting matching messages.

No Reed-Solomon encoding is used because ATCaaS DAG-TOB vertex payloads
(VRF proof + edge hashes + client-request digest) are short.

Public API:
    YosoRBC.sender(msg)   -- run by the designated sender P_s
    YosoRBC.reactor()     -- run by every node
"""
import sys
import base64
import pickle
import asyncio
import hashlib
import json
import logging
import math
import os
from typing import Dict, Tuple
from Crypto.Util.number import long_to_bytes
from ctypes import *
from collections import defaultdict
from typing import DefaultDict
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Messaging helpers – identical to those used in batch_multiplication.py
# ---------------------------------------------------------------------------
from beaver.utils.misc import wrap_send, subscribe_recv
from beaver.vrf import (
    BERNOULLI_COMMITTEE_MODE,
    FIXED_SIZE_COMMITTEE_MODE,
    _beta_to_canonical_bytes,
    _passes,
    _r_from_beta,
    fixed_size_committee,
    normalize_committee_selection_mode,
)

# ---------------------------------------------------------------------------
#  VRF FFI (optional).  If libvrf.so is absent the code falls back to stubs.
# ---------------------------------------------------------------------------
try:

    lib_bulletproof = CDLL("./libbulletproofs_amcl.so")
    lib_bulletproof.pyVrfProve.argtypes = [c_char_p]
    lib_bulletproof.pyVrfProve.restype  = c_void_p

    lib_bulletproof.pyVrfVerify.argtypes = [c_char_p]
    lib_bulletproof.pyVrfVerify.restype = c_void_p

    lib_bulletproof.pyFreeString.argtypes = [c_void_p]  # ✅ 必须匹配！
    lib_bulletproof.pyFreeString.restype = None
except OSError:
    lib_bulletproof = None


def vrf_backend_available() -> bool:
    """Return whether this module's native VRF verifier is loaded."""

    return lib_bulletproof is not None


# ---------------------------------------------------------------------------
#  Constants – legacy defaults for older 4-node smoke tests.
# ---------------------------------------------------------------------------
M  = 4               # network size
t  = 1               # Byzantine threshold
n  = 3 * t + 1       # legacy committee size (= 4)




def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
#  Main class
# ---------------------------------------------------------------------------
class YosoRBC:
    def __init__(self,
                 public_keys, private_key,      # PKI
                 pkbls, skbls,                  # BLS
                 n, t, srs, my_id,              # Committee Size/Threshold
                 send, recv,
                 prefix: str = "RBC", shared_subscribe=None,
                 threshold_fraction: float = 1.0,   # n/M for ATCaaS; 1.0 = always-in (legacy demo)
                 network_size: int = None,           # M long-lived nodes; defaults to n for legacy callers
                 vrf_sk=None,
                 vrf_pks=None,
                 expected_committee_size: int = None,
                 committee_m: int = None,
                 quorum_h: int = None,
                 ready_amplify_threshold: int = None,
                 vrf_echo_label: bytes = None,
                 vrf_ready_label: bytes = None,
                 require_vrf_proofs: bool = False,
                 committee_selection_mode: str = None,
                 fixed_committee_seed: str = None,
                 designated_sender_id: int = None,
                 payload_validator=None):

        global logger
        logfile = f'./log/logs-{my_id}.log'

        logging.basicConfig(
            level=logging.INFO,
            format = '%(asctime)s:[%(filename)s:%(lineno)s]:[%(levelname)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=logfile,  
            filemode='w'
        )
        # logging.basicConfig(
        #     level=logging.INFO,
        #     format='%(asctime)s [%(levelname)s] %(message)s',
        #     handlers=[logging.StreamHandler(sys.stdout),
        #             logging.FileHandler(logfile, mode='w')]
        # )

        logger = logging.getLogger(__name__)

        self.public_keys, self.private_key, self.pkbls, self.skbls = (public_keys, private_key, pkbls, skbls)
        # ------------------------------------------------------------------
        #  Per‑instance tag prefix  (allows many concurrent RBC sessions)
        # ------------------------------------------------------------------
        self.prefix = prefix
        def _tag(kind: str) -> str:
            return f"{self.prefix}_{kind}"
        self._tag = _tag
        self.vrf_echo_label = (
            vrf_echo_label
            if vrf_echo_label is not None
            else f"{self.prefix}:ECHO".encode()
        )
        self.vrf_ready_label = (
            vrf_ready_label
            if vrf_ready_label is not None
            else f"{self.prefix}:READY".encode()
        )
        # Node identifier (needed throughout the class).  In ATCaaS these
        # parameters are distinct: M is the number of long-lived online nodes
        # and n is the expected VRF-selected committee size.  Legacy callers
        # pass only n, so M defaults to n.
        self.M = network_size if network_size is not None else n
        self.n, self.t, self.srs, self.my_id = (n, t, srs, my_id)
        self.send, self.recv = (send, recv)
        self.threshold_fraction = threshold_fraction   # committee sortition probability
        self.expected_committee_size = (
            expected_committee_size
            if expected_committee_size is not None
            else max(1, int(round(self.threshold_fraction * self.M)))
        )
        self.committee_m = committee_m if committee_m is not None else n
        self.quorum_h = quorum_h if quorum_h is not None else 2 * t + 1
        self.ready_amplify_threshold = (
            ready_amplify_threshold
            if ready_amplify_threshold is not None
            else math.ceil(self.quorum_h / 2) + 1
        )
        if not 0 < self.threshold_fraction <= 1:
            raise ValueError("threshold_fraction must be in (0, 1]")
        if not 0 < self.quorum_h <= self.committee_m:
            raise ValueError("quorum_h must satisfy 0 < quorum_h <= committee_m")
        if self.ready_amplify_threshold > self.quorum_h:
            raise ValueError("ready_amplify_threshold must be <= quorum_h")
        self.vrf_sk = vrf_sk
        self.vrf_pk_list = vrf_pks if vrf_pks is not None else []
        self.require_vrf_proofs = bool(require_vrf_proofs)
        self.committee_selection_mode = normalize_committee_selection_mode(
            committee_selection_mode
            if committee_selection_mode is not None
            else os.environ.get(
                "ATCAAS_COMMITTEE_SELECTION_MODE",
                BERNOULLI_COMMITTEE_MODE,
            )
        )
        self.fixed_committee_seed = str(
            fixed_committee_seed
            if fixed_committee_seed is not None
            else os.environ.get("ATCAAS_FIXED_COMMITTEE_SEED", "")
        )
        self.designated_sender_id = (
            None if designated_sender_id is None else int(designated_sender_id)
        )
        if self.designated_sender_id is not None and not (
            0 <= self.designated_sender_id < self.M
        ):
            raise ValueError("designated_sender_id must be within the network")
        if payload_validator is not None and not callable(payload_validator):
            raise TypeError("payload_validator must be callable or null")
        self.payload_validator = payload_validator
        self._validated_payload_hashes = set()
        self._rejected_payload_hashes = set()
        if self.require_vrf_proofs and lib_bulletproof is None:
            raise RuntimeError(
                "cryptographic VRF backend is required in VRF mode"
            )
        if self.require_vrf_proofs and (
            self.vrf_sk is None or len(self.vrf_pk_list) < self.M
        ):
            raise ValueError(
                "VRF mode requires a local secret key and M public keys"
            )
        logger.info(
            "[PARAM] node %d M=%d expected_committee_size=%d expected_fraction=%.6f committee_m=%d quorum_h=%d ready_amplify=%d selection_mode=%s",
            self.my_id,
            self.M,
            self.expected_committee_size,
            self.threshold_fraction,
            self.committee_m,
            self.quorum_h,
            self.ready_amplify_threshold,
            self.committee_selection_mode,
        )
        # Create a mechanism to split the `recv` channels based on `tag`
        # self.subscribe_recv_task, self.subscribe_recv = subscribe_recv(recv)
        # # Helper to fetch a tag‑scoped receiver just like we do for send.
        # def _recv(tag):
        #     return self.subscribe_recv(tag)
        # self._recv = _recv

        # ------------------------------------------------------------------
        # Glue code: ① 若调用方传进共享 dispatcher，就重用；
        #            ② 否则自己再开一个。
        # ------------------------------------------------------------------
        if shared_subscribe is None:
            self._subscribe_task, self._subscribe_recv = subscribe_recv(recv)
        else:
            self._subscribe_task, self._subscribe_recv = shared_subscribe

        def _recv(tag):
            return self._subscribe_recv(tag)
        self._recv = _recv

        # Wrap the point-to-point send into a broadcast function
        def _mk_broadcast(tag: str):
            p2p_send = wrap_send(tag, send)  # expects (dest, data)

            async def broadcast(msg: bytes):
                for dest in range(self.M):
                    p2p_send(dest, msg)
            return broadcast
        self.get_send = _mk_broadcast

        self._echo_recv_hash: DefaultDict[str, set] = defaultdict(set)  # hash -> set of sender ids

        self.output_queue = asyncio.Queue()
        # ------------------------------------------------------------------
        #  Pending‑pool & block‑production helpers
        # ------------------------------------------------------------------
        self.pending_pool: list[bytes] = []   # locally buffered RBC payloads
        self.block_height: int = 0            # monotonically increasing block ID

        self.benchmark_logger = logging.LoggerAdapter(
            logging.getLogger("benchmark_logger"), {"node_id": self.my_id}
        )

        #  -------------------  RBC state variables  --------------------
        self._send_seen   = False            # have we processed the first SEND yet?
        self._msg         = None             # cached payload bytes
        self._hash        = None             # SHA‑256(m)
        self._outputted   = False
        self._ready_sent = False
        self._ready_recv_hash: DefaultDict[str, set] = defaultdict(set)  # hash -> set of sender ids

    def _get_generator_hex(self):
        if hasattr(self, "_cached_g_hex"):
            return self._cached_g_hex
        pk_dict = json.loads(self.srs["Pk"].decode("utf-8"))
        # logger.info("self.srs: %s", self.srs)
        g0 = pk_dict["G1_g"][0]
        gx = long_to_bytes(int(g0["X"]), 48)
        gy = long_to_bytes(int(g0["Y"]), 48)
        self._cached_g_hex = (b'\x04' + gx + gy).hex()
        return self._cached_g_hex
    
    # ......................................................................
    #  VRF stubs / helpers
    # ......................................................................
    def _fixed_committee(self, label: bytes) -> Tuple[int, ...]:
        return fixed_size_committee(
            label=label,
            population_size=self.M,
            committee_size=self.expected_committee_size,
            seed=self.fixed_committee_seed,
        )

    def _is_committee_member(self, label: bytes) -> Tuple[bool, str]:
        """
        Dummy committee selection:
            – For this prototype every node is in every committee.
            – Returns (True, proof_json) so that downstream code
              can pretend it verified a VRF proof.
        """
        if (
            self.committee_selection_mode == FIXED_SIZE_COMMITTEE_MODE
            and self.my_id not in self._fixed_committee(label)
        ):
            return False, None
        if lib_bulletproof is None:
            return True, "{}"               # no VRF library – always succeed

        # msg = "43066057178372115162090031665738480497785504495963485110743715314730498099898"
        # msg_hex = hex(int(msg))[2:]  # 去掉 0x 前缀
        # Encode the VRF label itself (bytes) as a hex string to use as the message
        msg_hex = label.hex()
        logger.info("msg_hex: %s", msg_hex)

        # # logger.info("self.private_key: %s", self.private_key)
        # raw = self.private_key.decode("utf-8")
        # inner = json.loads(raw)
        # key_dict = json.loads(inner)
        # # logger.info("key_dict: %s", key_dict)

        # # 1. 取出 Base64
        # sk_b64 = key_dict[str(self.my_id)]

        # # 2. 解 Base64 → 去掉内层引号
        # sk_str = base64.b64decode(sk_b64).decode('utf-8').strip('"')

        # # 3. 转十进制 → ZR
        # sk_int  = int(sk_str)
        # sk_hex = hex(sk_int)[2:]  # 去掉 0x 前缀

        if self.vrf_sk is None:
            raise ValueError("VRF proofs require a generated node secret key")
        sk_hex = format(int(self.vrf_sk), "x")

        # logger.info("self.public_keys: %s", self.public_keys)
        
        payload = {
                "sk": sk_hex,
                "msg": msg_hex, 
                "g":   self._get_generator_hex()
            }
        json_input = json.dumps(payload).encode("utf-8")
        ptr_proof = lib_bulletproof.pyVrfProve(json_input)
        proof = json.loads(string_at(ptr_proof).decode("utf-8"))
        # logger.info("proof  : %s", proof)
        lib_bulletproof.pyFreeString(ptr_proof)

        # In fixed-size experiment mode, public fixture membership is the
        # selection decision.  The VRF proof authenticates the selected
        # role's label/key binding but is deliberately not Bernoulli-gated.
        if self.committee_selection_mode == FIXED_SIZE_COMMITTEE_MODE:
            logger.info(
                "[VRF] node %d label=%s selected=True mode=fixed-size",
                self.my_id,
                label.decode(errors="replace"),
            )
            return True, proof

        # Threshold check: r = SHA256(β), b = (r ≤ threshold_fraction * MAX)
        r = _r_from_beta(proof["beta"])
        selected = _passes(r, self.threshold_fraction)
        logger.info(
            "[VRF] node %d label=%s selected=%s threshold_fraction=%.6f",
            self.my_id,
            label.decode(errors="replace"),
            selected,
            self.threshold_fraction,
        )
        if not selected:
            return False, None
        return True, proof

    # ......................................................................
    #  VRF verification helper
    # ......................................................................
    def _verify_vrf_proof(self, label: bytes, proof: dict, sender_id: int) -> bool:
        """
        Verify a VRF proof coming from *sender_id* for the given *label*.
        Falls back to “accept‑all” if the FFI library is not available.
        """
        if (
            self.committee_selection_mode == FIXED_SIZE_COMMITTEE_MODE
            and sender_id not in self._fixed_committee(label)
        ):
            return False
        if lib_bulletproof is None:
            return True                       # no VRF library – trust by default

        msg_hex = label.hex()

        # -- look up sender’s configured VRF public key --------------------
        pk_entry = self.vrf_pk_list[sender_id]   # {"X": "...", "Y": "..."} or 04||X||Y hex
        # logger.info("pk_entry: %s", pk_entry)
        # logger.info("sender_id: %s", sender_id)
        if isinstance(pk_entry, dict):
            gx_dec = int(pk_entry["X"])
            gy_dec = int(pk_entry["Y"])
            gx_bytes = long_to_bytes(gx_dec, 48)
            gy_bytes = long_to_bytes(gy_dec, 48)
            # 0x04 || X || Y  — uncompressed point encoding
            uncompressed_pk = b'\x04' + gx_bytes + gy_bytes
            pk_hex = uncompressed_pk.hex()
        else:
            pk_hex = pk_entry
        # logger.info("pk_hex: %s", pk_hex)
        # logger.info("proof beta: %s", proof["beta"])
        # logger.info("proof c: %s", proof["c"])
        # logger.info("proof s: %s", proof["s"])

        # Convert beta from prove's "(X_hex,Y_hex)" to verify's 04||X||Y encoding
        beta_hex = _beta_to_canonical_bytes(proof["beta"]).hex()

        # -- call FFI -------------------------------------------------------
        payload = {
            "pk":   pk_hex,
            "msg":  msg_hex,
            "beta": beta_hex,
            "c":    proof["c"],
            "s":    proof["s"],
            "g":    self._get_generator_hex()
        }
        json_input = json.dumps(payload).encode("utf-8")
        ptr = lib_bulletproof.pyVrfVerify(json_input)
        raw = string_at(ptr).decode("utf-8")
        logger.info("VRF verify result: %s", raw)
        lib_bulletproof.pyFreeString(ptr)

        # library returns either plain "true"/"false" or {"ok": true} JSON
        try:
            res = json.loads(raw)
            valid = bool(res.get("valid", res.get("ok", False)))
        except Exception:
            valid = raw.strip().lower() in ("true", "1")

        if not valid:
            return False

        if self.committee_selection_mode == FIXED_SIZE_COMMITTEE_MODE:
            logger.info(
                "[VRF] verified sender=%d label=%s selected=True mode=fixed-size",
                sender_id,
                label.decode(errors="replace"),
            )
            return True

        # Threshold check: same logic as _is_committee_member
        r = _r_from_beta(proof["beta"])
        selected = _passes(r, self.threshold_fraction)
        logger.info(
            "[VRF] verified sender=%d label=%s selected=%s",
            sender_id,
            label.decode(errors="replace"),
            selected,
        )
        return selected

    # ......................................................................
    #  Pending pool / block‑packaging logic
    # ......................................................................
    def _predicate_W(self) -> bool:
        """
        Placeholder predicate for when a block can be formed.
        For now it always returns True; plug in real logic here later.
        """
        return True

    def _maybe_package_block(self):
        """
        If predicate W is satisfied, turn the current pending‑pool into a block.
        Right now we just log the block and clear the pool; hooking up an ACS
        instance can be done later where indicated.
        """
        if not self.pending_pool:
            return
        if self._predicate_W():
            block = {
                "height": self.block_height,
                "messages": self.pending_pool.copy(),
            }
            logger.info("[BLOCK] packaged block %d with %d msgs",
                        self.block_height, len(self.pending_pool))
            # TODO: replace the logging statement with a call to ACS.
            self.pending_pool.clear()
            self.block_height += 1

    def _add_to_pending(self, payload: bytes):
        """
        Add a freshly delivered RBC payload to the local pending pool and
        trigger block‑packaging if the predicate is met.
        """
        self.pending_pool.append(payload)
        self._maybe_package_block()

    # ......................................................................
    #  API: sender coroutine
    # ......................................................................
    async def sender(self, m: bytes):
        """Executed by P_s to initiate RBC with payload `m`."""
        # logger.info("[SEND] node %d sending initial message %s", self.my_id, m)
        tag = self._tag("SEND")
        # logger.info("[SEND] tag: %s", tag)
        send = self.get_send(tag)
        await send(json.dumps({
            "sender": self.my_id,
            "payload": m.hex(),
        }).encode())            # broadcast to all

        logger.info("[SEND] broadcasted initial message (%d bytes)", len(m))

    # ......................................................................
    #  API: reactor coroutine – every node runs exactly ONE instance
    # ......................................................................
    async def reactor(self):
        send_echo  = self.get_send(self._tag("ECHO"))
        send_ready = self.get_send(self._tag("READY"))

        recv_send   = self._recv(self._tag("SEND"))
        logger.info("[REACTOR] node %d waiting for SEND, %s", self.my_id, self._tag("SEND"))
        recv_echo   = self._recv(self._tag("ECHO"))
        recv_ready  = self._recv(self._tag("READY"))

        tasks = {
            asyncio.create_task(recv_send()):  "SEND",
            asyncio.create_task(recv_echo()):  "ECHO",
            asyncio.create_task(recv_ready()): "READY",
        }
        out_task_start = time.time()
        try:
            while tasks:
                done, _ = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
                task_start = time.time()
                for t in done:
                    tag_kind = tasks.pop(t)
                    try:
                        sender, raw = t.result()  # raw: bytes
                    except asyncio.CancelledError:
                        continue

                    msg = json.loads(raw.decode())

                    if tag_kind == "SEND":
                        logger.info("[SEND] node %d received SEND from %d", self.my_id, sender)
                        await self._handle_send(sender, msg, send_echo)
                        tasks[asyncio.create_task(recv_send())] = "SEND"
                    elif tag_kind == "ECHO":
                        logger.info("[ECHO] node %d received ECHO from %d", self.my_id, sender)
                        await self._handle_echo(sender, msg, send_ready)
                        tasks[asyncio.create_task(recv_echo())] = "ECHO"
                    elif tag_kind == "READY":
                        logger.info("[READY] node %d received READY from %d", self.my_id, sender)
                        await self._handle_ready(sender, msg)
                        tasks[asyncio.create_task(recv_ready())] = "READY"
                task_end = time.time()
                logger.info("[BENCHMARK] node %d reactor loop iteration time: %f seconds", self.my_id, task_end - task_start)
                if self._outputted:
                    out_task_end = time.time()
                    logger.info("[BENCHMARK] node %d total RBC time: %f seconds", self.my_id, out_task_end - out_task_start)
                    return
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks.keys(), return_exceptions=True)

    # ......................................................................
    #  Handlers
    # ......................................................................
    def _payload_is_admissible(self, payload: bytes) -> bool:
        digest = _hash(payload)
        if digest in self._validated_payload_hashes:
            return True
        if digest in self._rejected_payload_hashes:
            return False
        try:
            valid = self.payload_validator is None or bool(
                self.payload_validator(bytes(payload))
            )
        except Exception:
            valid = False
        (self._validated_payload_hashes if valid else self._rejected_payload_hashes).add(
            digest
        )
        return valid

    async def _handle_send(self, transport_sender, msg, send_echo):
        """Handle SEND: cache the full message locally, broadcast ECHO(m)."""
        handle_send_start = time.time()
        claimed_sender = msg.get("sender")
        if (
            isinstance(claimed_sender, bool)
            or not isinstance(claimed_sender, int)
            or claimed_sender != transport_sender
            or (
                self.designated_sender_id is not None
                and transport_sender != self.designated_sender_id
            )
        ):
            logger.warning(
                "[SEND] sender mismatch transport=%s claimed=%s designated=%s - discarded",
                transport_sender,
                claimed_sender,
                self.designated_sender_id,
            )
            return
        if self._send_seen:
            return
        try:
            m_hex = msg["payload"]
            candidate = bytes.fromhex(m_hex)
        except (KeyError, TypeError, ValueError):
            return
        if not self._payload_is_admissible(candidate):
            logger.warning("[SEND] inadmissible payload from %d - discarded", transport_sender)
            return
        self._send_seen = True
        self._msg = candidate
        self._hash = _hash(self._msg)

        # Committee filter (VRF): skip if not in YOSO-echo committee
        echo_label = self.vrf_echo_label
        logger.info("[SEND] node %d checking ECHO committee membership", self.my_id)
        in_committee, proof = self._is_committee_member(echo_label)
        if not in_committee:
            return

        # Classic Bracha RBC: ECHO carries the original message m.
        await send_echo(json.dumps({
            "hash":    self._hash,
            "from":    self.my_id,
            "proof":   proof,
            "payload": m_hex,
        }).encode())

        logger.info("[ECHO] node %d broadcast ECHO", self.my_id)
        handle_send_end = time.time()
        logger.info("[BENCHMARK] node %d handle_send time: %f seconds", self.my_id, handle_send_end - handle_send_start)

    async def _handle_echo(self, transport_sender, msg, send_ready):
        handle_echo_start = time.time()
        sender = int(msg.get("from", transport_sender))
        if sender != transport_sender:
            logger.warning("[ECHO] sender mismatch transport=%d claimed=%d - discarded",
                           transport_sender, sender)
            return

        echo_label = self.vrf_echo_label
        if not self._verify_vrf_proof(echo_label, msg["proof"], sender):
            logger.warning("[ECHO] invalid VRF proof from %d - discarded", sender)
            return

        if "payload" not in msg:
            logger.warning("[ECHO] missing payload from %d - discarded", sender)
            return

        m_candidate = bytes.fromhex(msg["payload"])
        h = _hash(m_candidate)
        if msg.get("hash") is not None and msg["hash"] != h:
            logger.warning("[ECHO] hash/payload mismatch from %d - discarded", sender)
            return
        if not self._payload_is_admissible(m_candidate):
            logger.warning("[ECHO] inadmissible payload from %d - discarded", sender)
            return
        logger.info("[ECHO] node %d received ECHO from %d with hash %s", self.my_id, sender, h)

        # If we missed SEND, recover m from the first valid ECHO/READY carrying it.
        if self._msg is None:
            self._msg = m_candidate
            self._hash = h
            logger.info("[ECHO] node %d recovered m from ECHO (%d bytes)", self.my_id, len(self._msg))
        elif _hash(self._msg) != h:
            logger.warning("[ECHO] conflicting payload from %d - discarded", sender)
            return

        if sender in self._echo_recv_hash[h]:
            return
        self._echo_recv_hash[h].add(sender)

        echo_cnt = len(self._echo_recv_hash[h])
        logger.info(
            "[ECHO] node %d verified ECHO count for %s: %d/%d",
            self.my_id,
            h,
            echo_cnt,
            self.quorum_h,
        )

        if echo_cnt >= self.quorum_h:
            if not self._ready_sent:
                self._ready_sent = True
                ready_label = self.vrf_ready_label
                in_comm, proof_r = self._is_committee_member(ready_label)
                if not in_comm:
                    return
                # Classic Bracha RBC: READY also carries the original message m.
                await send_ready(json.dumps({
                    "hash": h,
                    "from": self.my_id,
                    "proof": proof_r,
                    "payload": msg["payload"],
                }).encode())
                logger.info("[READY] node %d broadcast READY", self.my_id)

        # Edge case: 2t+1 READY already arrived before we got m — try output now
        if (not self._outputted
                and self._msg is not None
                and len(self._ready_recv_hash[h]) >= self.quorum_h
                and _hash(self._msg) == h):
            self._outputted = True
            logger.info("[OUTPUT] node %d delivered M (%d bytes) via late ECHO", self.my_id, len(self._msg))
            self._add_to_pending(self._msg)
            await self.output_queue.put((self._msg, self._hash))

        handle_echo_end = time.time()
        logger.info("[BENCHMARK] node %d handle_echo time: %f seconds", self.my_id, handle_echo_end - handle_echo_start)

    async def _handle_ready(self, transport_sender, msg):
        handle_ready_start = time.time()
        sender = int(msg.get("from", transport_sender))
        if sender != transport_sender:
            logger.warning("[READY] sender mismatch transport=%d claimed=%d - discarded",
                           transport_sender, sender)
            return

        ready_label = self.vrf_ready_label
        if not self._verify_vrf_proof(ready_label, msg["proof"], sender):
            logger.warning("[READY] invalid VRF proof from %d - discarded", sender)
            return

        if "payload" not in msg:
            logger.warning("[READY] missing payload from %d - discarded", sender)
            return

        m_candidate = bytes.fromhex(msg["payload"])
        h = _hash(m_candidate)
        if msg.get("hash") is not None and msg["hash"] != h:
            logger.warning("[READY] hash/payload mismatch from %d - discarded", sender)
            return
        if not self._payload_is_admissible(m_candidate):
            logger.warning("[READY] inadmissible payload from %d - discarded", sender)
            return

        if self._msg is None:
            self._msg = m_candidate
            self._hash = h
            logger.info("[READY] node %d recovered m from READY (%d bytes)", self.my_id, len(self._msg))
        elif _hash(self._msg) != h:
            logger.warning("[READY] conflicting payload from %d - discarded", sender)
            return

        if sender in self._ready_recv_hash[h]:
            return  # duplicate READY from same sender
        self._ready_recv_hash[h].add(sender)
        cnt = len(self._ready_recv_hash[h])
        logger.info(
            "[READY] node %d verified READY count for %s: %d/%d",
            self.my_id,
            h,
            cnt,
            self.quorum_h,
        )

        # Amplification: ceil(h/2)+1 READY -> broadcast READY if not sent yet.
        if cnt >= self.ready_amplify_threshold and not self._ready_sent:
            if self._hash is None:
                self._hash = h
            self._ready_sent = True
            ready_label2 = self.vrf_ready_label
            in_comm, proof_r = self._is_committee_member(ready_label2)
            if in_comm:
                send_ready = self.get_send(self._tag("READY"))
                await send_ready(json.dumps({
                    "hash": h,
                    "from": self.my_id,
                    "proof": proof_r,
                    "payload": msg["payload"],
                }).encode())
                logger.info("[READY] node %d amplified READY", self.my_id)

        # Output: h READY and we have the message cached from SEND/READY.
        if cnt >= self.quorum_h and not self._outputted:
            if self._msg is not None and _hash(self._msg) == h:
                self._outputted = True
                logger.info("[OUTPUT] node %d delivered M (%d bytes)", self.my_id, len(self._msg))
                self._add_to_pending(self._msg)
                await self.output_queue.put((self._msg, self._hash))

        handle_ready_end = time.time()
        logger.info("[BENCHMARK] node %d handle_ready time: %f seconds", self.my_id, handle_ready_end - handle_ready_start)

    # ------------------------------------------------------------------
    #  Context‑manager / cleanup utilities
    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # nothing special to clean up for this prototype
        return False

    def kill(self):
        # placeholder for symmetry with caller code
        pass

    # ..................................................................
    #  High‑level driver invoked by run_yosorbc.py
    # ..................................................................
    async def run_yosorbc(self, node_communicator):
        """
        Spawn the reactor and (if we are node 0) act as the sender
        with a dummy payload.  Completes once OUTPUT is delivered.
        """
        reactor_task = asyncio.create_task(self.reactor())

        # Give peers a moment to subscribe
        await asyncio.sleep(0.1)

        if self.my_id == 0:
            await self.sender(b"Hellofromnode0")  # P_s sends initial message

        # Wait until reactor terminates (it returns after OUTPUT)
        await reactor_task

        bytes_sent = node_communicator.bytes_sent
        for k,v in node_communicator.bytes_count.items():
            logger.info(f"[{self.my_id}] Bytes Sent: {k}:{v} which is {round((100*v)/bytes_sent,3)}%")
        logger.info(f"[{self.my_id}] Total bytes sent out aa: {bytes_sent}")

        # Gracefully shut down receive task and return so the process can exit
        self._subscribe_task.cancel()
        return
