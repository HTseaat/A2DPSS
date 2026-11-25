"""ACD+25 Algorithms 5–7, using existing field PVSS for a hybrid payload key."""
import asyncio
import secrets
import time
from beaver.g1_backend import G1Point, FIELD_ORDER, field_random
from beaver.pedersen_shamir import sample_poly, eval_poly, interpolate_at_zero
from beaver.pvss import PVSSTranscript, pvss_deal, pvss_verify, pvss_deliver
from .model import canonical, decode, node_set
from .crypto import encrypt_bytes, decrypt_bytes, encrypt_recipient, decrypt_recipient
from .transport import CommitteeBroadcast


class CommitteeAVSS:
    def __init__(self, runtime, view):
        self.r, self.view = runtime, view
        self.tag = runtime.p.role(view, "avss")
        self.setup_role = runtime.p.role(view, "key")
        self.share_role = runtime.p.role(view, "proposal")
        self.setup_rbc = CommitteeBroadcast(runtime, self.tag + "/setup", self.setup_role)
        # Proposal and rank share travel in ONE broadcast (Algorithm 10).
        self.share_rbc = CommitteeBroadcast(runtime, self.tag + "/share", self.share_role)
        self.setups, self.transcripts, self.payloads = {}, {}, {}
        self.keys, self.recovered = {}, {}
        self.rec_started = False
        self.setup_seen, self.share_seen, self.rec_sent = set(), set(), set()
        self.key_attempted, self.opened = set(), {}
        self.open_attempted = {}
        runtime.spawn(self._service())

    def aad(self, purpose, source):
        return canonical([self.r.session, self.tag, purpose, source])

    async def setup(self):
        if self.r.committees.member(self.setup_role)[0]:
            secret = field_random()
            polynomial = sample_poly(secret, self.r.p.degree)
            if self.r.p.mode == "fixed-size":
                recipients = secrets.SystemRandom().sample(range(self.r.p.population), self.r.p.m)
            else:
                recipients = [i for i in range(self.r.p.population)
                              if secrets.randbelow(1 << 53) < self.r.p.probability * (1 << 53)]
            if len(recipients) > self.r.p.m:
                raise RuntimeError("sampled AVSS recipient committee exceeds configured m")
            encrypted = []
            for recipient in recipients:
                share = {"recipient": recipient, "share": eval_poly(polynomial, recipient + 1)}
                signed = self.r.signer.attest(self.tag + "/key-share", share)
                encrypted.append(encrypt_recipient(
                    self.r.g, self.r.public[recipient]["encryption"],
                    canonical({"share": share, "signed": signed}), self.aad("setup", self.r.node)))
            payload = {"public": (G1Point.from_hex(self.r.g) * secret).to_hex(), "encrypted": encrypted}
            del secret, polynomial, recipients
            self.setup_rbc.broadcast(payload)
        await self.r.wait(lambda: len(self.setups) >= self.r.p.h)

    async def share(self, hidden, proposal):
        if not self.r.committees.member(self.share_role)[0]:
            return
        ids = sorted(self.setups)[:self.r.p.h]
        secret = field_random()
        start = time.perf_counter()
        transcript = pvss_deal(self.r.g, self.r.h, [self.setups[i]["public"] for i in ids],
                               secret, field_random(), self.r.p.degree, self.r.node, self.r.node + 1)
        self.r.metrics["election_pvss_deal_seconds"] += time.perf_counter() - start
        self.r.metrics["election_pvss_deals"] += 1
        ciphertext = encrypt_bytes(secret.to_bytes(32, "big"), canonical(hidden), self.aad("rank", self.r.node))
        del secret, hidden
        self.share_rbc.broadcast({"ids": ids, "transcript": transcript.to_obj(),
                                  "encrypted": ciphertext, "proposal": proposal})

    def reconstruct(self):
        if not self.rec_started:
            self.rec_started = True
            self.r.metrics["avss_rec_started"] += 1
            self.r.pulse()

    def _valid_setup(self, value):
        try:
            return (isinstance(value["encrypted"], list)
                    and len(value["encrypted"]) <= self.r.p.m
                    and not G1Point.from_hex(value["public"]).is_identity())
        except (KeyError, ValueError, TypeError):
            return False

    async def _service(self):
        while True:
            event = self.r.event
            for source, value in list(self.setup_rbc.delivered.items()):
                if source not in self.setup_seen:
                    self.setup_seen.add(source)
                    if self._valid_setup(value):
                        self.setups[source] = value
                        self.r.pulse()
            for source, value in list(self.share_rbc.delivered.items()):
                if source in self.share_seen:
                    continue
                try:
                    ids = value["ids"]
                    if not node_set(ids, self.r.p.population, self.r.p.h):
                        raise ValueError("bad PVSS recipient set")
                    if not set(ids).issubset(self.setups):
                        continue  # a referenced setup can arrive after this share
                    self.share_seen.add(source)
                    transcript = PVSSTranscript.from_bytes(canonical(value["transcript"]))
                    if transcript.dealer_id != source or transcript.dealer_eval_point != source + 1:
                        raise ValueError("wrong PVSS dealer")
                    start = time.perf_counter()
                    valid = pvss_verify(self.r.g, self.r.h, [self.setups[i]["public"] for i in ids],
                                        transcript, self.r.p.degree)
                    self.r.metrics["election_pvss_verify_seconds"] += time.perf_counter() - start
                    self.r.metrics["election_pvss_verifications"] += 1
                    if not valid:
                        raise ValueError("invalid PVSS")
                    self.transcripts[source] = (ids, transcript)
                    self.payloads[source] = value
                    self.r.pulse()
                except (KeyError, ValueError, TypeError, IndexError):
                    self.share_seen.add(source)
                    self.r.metrics["invalid_avss_shares"] += 1
                finally:
                    # Expose each newly validated payload to PE immediately;
                    # do not move its threshold delay into this upstream batch.
                    await asyncio.sleep(0)
            if self.rec_started:
                self._recover_keys()
                self._recover_payloads()
            await event.wait()

    def _recover_keys(self):
        for source, setup in list(self.setups.items()):
            topic = "rec/%s/%d" % (self.tag, source)
            if source not in self.rec_sent:
                self.rec_sent.add(source)
                for encrypted in setup["encrypted"]:
                    try:
                        share = decode(decrypt_recipient(self.r.private["encryption"], encrypted,
                                                          self.aad("setup", source)))
                        if self._valid_key_share(source, self.r.node, share):
                            self.r.multicast(topic, share)
                            break
                    except (ValueError, KeyError, TypeError):
                        continue
            if source in self.key_attempted:
                continue
            shares = {}
            for sender, value, _ in self.r.inbox(topic):
                if self._valid_key_share(source, sender, value):
                    shares.setdefault(sender + 1, value["share"]["share"])
            if len(shares) >= self.r.p.reconstruct:
                self.key_attempted.add(source)
                points = sorted(shares)[:self.r.p.reconstruct]
                secret = interpolate_at_zero(points, [shares[i] for i in points])
                if G1Point.from_hex(self.r.g) * secret == G1Point.from_hex(setup["public"]):
                    self.keys[source] = secret
                    self.r.pulse()

    def _valid_key_share(self, source, recipient, value):
        try:
            share, signed = value["share"], value["signed"]
            return (share["recipient"] == recipient and type(share["share"]) is int
                    and 0 <= share["share"] < FIELD_ORDER and signed["node"] == source
                    and self.r.verifier.verify(self.tag + "/key-share", share, signed))
        except (ValueError, KeyError, TypeError):
            return False

    def _recover_payloads(self):
        for source, (ids, transcript) in list(self.transcripts.items()):
            if source in self.recovered:
                continue
            shares = self.opened.setdefault(source, {})
            attempted = self.open_attempted.setdefault(source, set())
            for index, nominee in enumerate(ids):
                if nominee not in self.keys or index in attempted:
                    continue
                attempted.add(index)
                try:
                    pair = pvss_deliver(self.r.g, self.r.h, transcript, self.keys[nominee], index)
                    opening = G1Point.from_hex(self.r.g) * pair[0] + G1Point.from_hex(self.r.h) * pair[1]
                    if opening == G1Point.from_hex(transcript.commitment_vector[index + 1]):
                        shares[index + 1] = pair
                except (ValueError, KeyError, TypeError, IndexError):
                    self.r.metrics["invalid_avss_openings"] += 1
            if len(shares) >= self.r.p.reconstruct:
                points = sorted(shares)[:self.r.p.reconstruct]
                secret = interpolate_at_zero(points, [shares[i][0] for i in points])
                try:
                    value = decode(decrypt_bytes(secret.to_bytes(32, "big"),
                                                  self.payloads[source]["encrypted"], self.aad("rank", source)))
                except (ValueError, KeyError, TypeError):
                    value = None  # deterministic invalid rank, not a blocked reconstruction
                self.recovered[source] = value
                self.r.metrics["avss_reconstructed_payloads"] += 1
                self.r.pulse()
