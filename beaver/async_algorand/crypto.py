"""Real VRF/PVSS-compatible keys, ratcheted signatures and PoC encryption.

ECDH/HKDF/AEAD hides a ciphertext's recipient but is NOT a KM-NCE
construction. Erasure here means dropping the old Python key reference.
"""
import hashlib
import secrets
from Crypto.PublicKey import ECC
from Crypto.Signature import eddsa
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import HKDF
from Crypto.Hash import SHA256
from beaver.g1_backend import G1Point, field_random
from beaver.vrf import (vrf_backend_available, vrf_prove_sortition,
                        vrf_verify_sortition, fixed_size_committee, _r_from_beta)
from .model import canonical, digest


def raw_point(point):
    point = G1Point.from_hex(point).to_hex()
    if point == "infinity":
        raise ValueError("identity public key")
    x, y = point.strip("()").split(",")
    return "04" + x.zfill(96) + y.zfill(96)


def generate_identity(g):
    signing = ECC.generate(curve="Ed25519")
    vrf, encryption = field_random(), field_random()
    private = {"signing": signing.seed.hex(), "vrf": vrf, "encryption": encryption}
    public = {"signing": signing.public_key().export_key(format="raw").hex(),
              "vrf": raw_point((G1Point.from_hex(g) * vrf).to_hex()),
              "encryption": (G1Point.from_hex(g) * encryption).to_hex()}
    return private, public


class Signer:
    def __init__(self, session, node, seed):
        self.session, self.node = session, node
        self.key = ECC.construct(curve="Ed25519", seed=bytes.fromhex(seed))
        self.chain = []
        self.used = set()

    def attest(self, domain, value):
        data = [self.session, self.node, domain, value]
        return {"node": self.node, "chain": list(self.chain),
                "signature": eddsa.new(self.key, "rfc8032").sign(canonical(data)).hex()}

    def seal(self, topic, body):
        if topic in self.used:
            raise ValueError("honest participant may speak only once per role")
        self.used.add(topic)
        next_key = ECC.generate(curve="Ed25519")
        link = {"topic": topic, "hash": digest(body),
                "next": next_key.public_key().export_key(format="raw").hex()}
        signed = self.attest("envelope", link)
        envelope = {"topic": topic, "body": body, "link": link, "signed": signed}
        self.chain.append({"link": link, "signature": signed["signature"]})
        self.key = next_key
        return envelope


class Verifier:
    def __init__(self, session, public):
        self.session, self.public = session, public
        self.cache = {}
        self._verified_links = set()

    def _check_link(self, public, node, link, signature):
        # Reuse only the exact signature statement already checked under this
        # key. A public key or chain length alone cannot authenticate a prefix:
        # Byzantine senders may fork their ratchet or revisit a public key.
        cache_key = (self.session, node, public, canonical(link), signature)
        if cache_key not in self._verified_links:
            self._check(public, node, "envelope", link, signature)
            self._verified_links.add(cache_key)

    def verify(self, domain, value, signed):
        try:
            node = signed["node"]
            if type(node) is not int or not 0 <= node < len(self.public):
                return False
            chain = signed["chain"]
            if not isinstance(chain, list) or len(chain) > 100000:
                return False
            root = self.public[node]["signing"]
            cache_key = (self.session, node, root, digest(chain))
            current = self.cache.get(cache_key)
            if current is None:
                current = root
                used = set()
                for entry in chain:
                    link = entry["link"]
                    if link["topic"] in used:
                        return False
                    used.add(link["topic"])
                    self._check_link(current, node, link, entry["signature"])
                    current = link["next"]
                self.cache[cache_key] = current
            if domain == "envelope":
                # A successfully checked envelope is the next chain entry.
                # Detached attestations have different domains and cannot
                # authenticate a ratchet link.
                self._check_link(current, node, value, signed["signature"])
            else:
                self._check(current, node, domain, value, signed["signature"])
            return True
        except (KeyError, ValueError, TypeError, IndexError):
            return False

    def _check(self, public, node, domain, value, signature):
        key = eddsa.import_public_key(bytes.fromhex(public))
        eddsa.new(key, "rfc8032").verify(
            canonical([self.session, node, domain, value]), bytes.fromhex(signature))

    def envelope(self, envelope):
        try:
            link = envelope["link"]
            return (link["topic"] == envelope["topic"]
                    and link["hash"] == digest(envelope["body"])
                    and not any(entry["link"]["topic"] == envelope["topic"]
                                for entry in envelope["signed"]["chain"])
                    and self.verify("envelope", link, envelope["signed"]))
        except (KeyError, ValueError, TypeError):
            return False


class Committees:
    def __init__(self, session, node, parameters, private, public, g):
        if not vrf_backend_available():
            raise RuntimeError("Asynchronous Algorand requires the real VRF backend")
        self.session, self.node, self.p = session, node, parameters
        self.private, self.public, self.g = private, public, g
        self.local, self.checked = {}, {}

    def label(self, role):
        return canonical(["ACD25", self.session, role])

    def _fixture(self, role, node):
        return (self.p.mode != "fixed-size" or node in fixed_size_committee(
            label=self.label(role), population_size=self.p.population,
            committee_size=self.p.m, seed=self.p.seed))

    def member(self, role):
        if role not in self.local:
            probability = self.p.probability if self.p.mode == "bernoulli" else 1.0
            yes, proof = vrf_prove_sortition(label=self.label(role), vrf_sk=self.private["vrf"],
                                            g_hex=self.g, threshold_fraction=probability)
            self.local[role] = (yes and self._fixture(role, self.node), proof)
        return self.local[role]

    def verify(self, role, node, proof, rank=False):
        try:
            if type(node) is not int or not 0 <= node < self.p.population:
                return False
            key = (role, node, digest(proof), rank)
            if key not in self.checked:
                probability = 1.0 if rank or self.p.mode == "fixed-size" else self.p.probability
                valid = vrf_verify_sortition(
                    label=self.label(role), proof=proof, sender_id=node,
                    vrf_pks=[p["vrf"] for p in self.public],
                    g_hex=self.g, threshold_fraction=probability)
                self.checked[key] = valid and (rank or self._fixture(role, node))
            return self.checked[key]
        except (KeyError, ValueError, TypeError):
            return False

    def rank(self, role):
        _, proof = vrf_prove_sortition(label=self.label(role), vrf_sk=self.private["vrf"],
                                     g_hex=self.g, threshold_fraction=1.0)
        return {"proof": proof, "rank": _r_from_beta(proof["beta"])}

    def verify_rank(self, role, node, value):
        try:
            return (self.verify(role, node, value["proof"], rank=True)
                    and type(value["rank"]) is int
                    and value["rank"] == _r_from_beta(value["proof"]["beta"]))
        except (KeyError, ValueError, TypeError):
            return False


def encrypt_bytes(material, plaintext, aad):
    key = HKDF(material, 32, b"ACD25-PoC", SHA256, context=aad)
    cipher = AES.new(key, AES.MODE_GCM, nonce=secrets.token_bytes(12))
    cipher.update(aad)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return {"nonce": cipher.nonce.hex(), "ciphertext": ciphertext.hex(), "tag": tag.hex()}


def decrypt_bytes(material, encrypted, aad):
    key = HKDF(material, 32, b"ACD25-PoC", SHA256, context=aad)
    cipher = AES.new(key, AES.MODE_GCM, nonce=bytes.fromhex(encrypted["nonce"]))
    cipher.update(aad)
    return cipher.decrypt_and_verify(bytes.fromhex(encrypted["ciphertext"]),
                                     bytes.fromhex(encrypted["tag"]))


def encrypt_recipient(g, public, plaintext, aad):
    ephemeral = field_random()
    shared = (G1Point.from_hex(public) * ephemeral).to_hex().encode()
    result = encrypt_bytes(shared, plaintext, aad)
    result["ephemeral"] = (G1Point.from_hex(g) * ephemeral).to_hex()
    return result


def decrypt_recipient(private, encrypted, aad):
    ephemeral = G1Point.from_hex(encrypted["ephemeral"])
    if ephemeral.is_identity():
        raise ValueError("identity encryption key")
    return decrypt_bytes((ephemeral * private).to_hex().encode(), encrypted, aad)
