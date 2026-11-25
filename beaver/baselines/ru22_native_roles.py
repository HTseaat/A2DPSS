"""Real-VRF leader committee and one verifier committee per leader.

Trusted offline conditioning provides exact-n memberships for benchmarks. Role
signing keys are independent even when roles share a physical machine. This
models one-send roles; it does not implement pre-speech anonymous membership.
"""
import json
from Crypto.PublicKey import ECC

from beaver.baselines.ru22_certificate import canonical, digest, sign_statement, verify_statement
from beaver.baselines.ru22_native import NativeAuthority, NativeMetrics
from beaver.g1_backend import G1Point, field_random
from beaver.vrf import vrf_backend_available, vrf_prove_sortition, vrf_verify_sortition


CONTROL = {"ready", "ready-ack", "audit-state", "done", "done-ack"}


def raw_point(point):
    x, y = point.strip("()").split(",")
    return "04" + x.zfill(96) + y.zfill(96)


def role_label(context, role, attempt):
    return canonical(["RU22-native-role/v2", context.context_digest, role, attempt])


def public_authority(context, artifact):
    if (artifact.get("schema") != "ru22-native-roles/v2"
            or artifact.get("context") != context.context_digest
            or artifact.get("digest") != digest({k: v for k, v in artifact.items() if k != "digest"})):
        raise ValueError("invalid public native role artifact")
    roles = artifact["roles"]
    leaders = roles["leader"]["members"]
    if set(roles) != {"source", "leader"} | {f"verifier/{i}" for i in leaders}:
        raise ValueError("one verifier committee per leader is required")
    keys = lambda role: tuple((i, roles[role]["signing_keys"][str(i)]) for i in roles[role]["members"])
    return NativeAuthority(context.to_bytes(), keys("leader"),
        tuple((i, keys(f"verifier/{i}")) for i in leaders))


def build_roles(context, population):
    if not vrf_backend_available():
        raise RuntimeError("native RU22 requires real VRF backend")
    if type(population) is not int or population < context.resharing_n:
        raise ValueError("population smaller than role committee")
    public, private = [], []
    g = raw_point(context.g_hex)
    for _ in range(population):
        key, vrf = ECC.generate(curve="Ed25519"), field_random()
        public.append(dict(signing=key.public_key().export_key(format="raw").hex(),
                           vrf=raw_point((G1Point.from_hex(context.g_hex) * vrf).to_hex())))
        private.append(dict(signing=key.seed.hex(), vrf=str(vrf), role_signing={}))
    roles = {}

    def assign(role, members, attempt=None):
        keys = {}
        for i in members:
            key = ECC.generate(curve="Ed25519")
            keys[str(i)] = key.public_key().export_key(format="raw").hex()
            private[i]["role_signing"][role] = key.seed.hex()
        roles[role] = dict(members=list(members), attempt=attempt, signing_keys=keys)

    def select(role):
        for attempt in range(4096):
            label = role_label(context, role, attempt)
            chosen = [i for i, p in enumerate(private) if vrf_prove_sortition(
                label=label, vrf_sk=int(p["vrf"]), g_hex=g,
                threshold_fraction=context.resharing_n / population)[0]]
            if len(chosen) == context.resharing_n:
                assign(role, chosen, attempt)
                return chosen
        raise RuntimeError("unable to sample exact-n VRF committee")

    assign("source", sorted(context.source_role_indices))
    for leader in select("leader"):
        select(f"verifier/{leader}")
    artifact = dict(schema="ru22-native-roles/v2", context=context.context_digest,
                    selection="trusted-exact-n-conditioned-bernoulli", population=population,
                    committee_size=context.resharing_n, public=public, roles=roles)
    artifact["digest"] = digest(artifact)
    return artifact, private


class NativeAuthentication:
    def __init__(self, context, artifact, node, private):
        if not vrf_backend_available():
            raise RuntimeError("native RU22 requires real VRF backend")
        artifact = json.loads(canonical(artifact))
        material = {k: v for k, v in artifact.items() if k != "digest"}
        if (artifact.get("schema") != "ru22-native-roles/v2"
                or artifact.get("digest") != digest(material)
                or artifact["context"] != context.context_digest
                or artifact["committee_size"] != context.resharing_n
                or artifact["selection"] != "trusted-exact-n-conditioned-bernoulli"
                or type(artifact["population"]) is not int
                or artifact["population"] < context.resharing_n
                or artifact["population"] != len(artifact["public"])):
            raise ValueError("invalid native role artifact")
        population = artifact["population"]
        if type(node) is not int or not 0 <= node < population:
            raise ValueError("invalid native node")
        roles = artifact["roles"]
        expected_roles = {"source", "leader"} | {f"verifier/{i}" for i in roles["leader"]["members"]}
        if set(roles) != expected_roles:
            raise ValueError("one verifier committee per leader is required")
        all_keys = []
        for name, role in roles.items():
            members = role["members"]
            if (len(members) != context.resharing_n or len(set(members)) != len(members)
                    or any(type(i) is not int or not 0 <= i < population for i in members)
                    or members != sorted(members)
                    or set(role["signing_keys"]) != {str(i) for i in members}):
                raise ValueError("invalid native committee membership")
            if name == "source":
                if members != sorted(context.source_role_indices) or role["attempt"] is not None:
                    raise ValueError("invalid source committee")
            elif type(role["attempt"]) is not int or not 0 <= role["attempt"] < 4096:
                raise ValueError("invalid native VRF attempt")
            all_keys.extend(role["signing_keys"].values())
        if len(set(all_keys)) != len(all_keys):
            raise ValueError("role signing keys must be independent")
        public = artifact["public"][node]
        seed = bytes.fromhex(private["signing"])
        key = ECC.construct(curve="Ed25519", seed=seed)
        if (key.public_key().export_key(format="raw").hex() != public["signing"]
                or raw_point((G1Point.from_hex(context.g_hex) * int(private["vrf"])).to_hex()) != public["vrf"]):
            raise ValueError("native private identity mismatch")
        expected_private = {name for name, role in roles.items() if node in role["members"]}
        if set(private["role_signing"]) != expected_private:
            raise ValueError("native private role membership mismatch")
        self._role_seeds = {}
        for name, encoded in private["role_signing"].items():
            role_seed = bytes.fromhex(encoded)
            public_key = ECC.construct(curve="Ed25519", seed=role_seed).public_key().export_key(format="raw").hex()
            if roles[name]["signing_keys"][str(node)] != public_key:
                raise ValueError("native private role key mismatch")
            self._role_seeds[name] = bytearray(role_seed)
        self.context, self.artifact, self.node = context, artifact, node
        self._seed, self._vrf = bytearray(seed), int(private["vrf"])
        self.metrics = NativeMetrics()
        self.spent = set()
        self.closed = False
        self.authority = public_authority(context, artifact)

    def label(self, role):
        return role_label(self.context, role, self.artifact["roles"][role]["attempt"])

    def _message_role(self, phase, instance, body):
        if phase == "vote":
            if (type(instance) is not int
                    or instance not in self.artifact["roles"]["leader"]["members"]
                    or not isinstance(body, dict) or not isinstance(body.get("package"), dict)
                    or type(body["package"].get("leader")) is not int
                    or body["package"]["leader"] != instance):
                raise ValueError("vote does not match its leader instance")
            return f"verifier/{instance}"
        if instance is not None:
            raise ValueError("invalid native phase or instance")
        if phase in ("contribution", "combine"):
            return "source" if phase == "contribution" else "leader"
        if isinstance(phase, str) and phase in CONTROL:
            return None
        raise ValueError("invalid native phase or instance")

    def seal(self, phase, body, *, instance=None):
        if self.closed:
            raise RuntimeError("native identity is closed")
        role = self._message_role(phase, instance, body)
        if role:
            if role in self.spent:
                raise RuntimeError("native logical role already sent")
            if role not in self._role_seeds:
                raise ValueError("not an eligible role")
        proof = None
        if role and role != "source":
            with self.metrics.measure("vrf_proof_generation"):
                selected, proof = vrf_prove_sortition(label=self.label(role),
                    vrf_sk=self._vrf, g_hex=raw_point(self.context.g_hex),
                    threshold_fraction=self.context.resharing_n / self.artifact["population"])
            if not selected:
                raise ValueError("assigned role did not pass real VRF threshold")
        obj = dict(schema="ru22-native-envelope/v2", session=self.artifact["digest"],
                   sender=self.node, phase=phase, instance=instance, body=body, vrf=proof)
        signing_seed = self._role_seeds[role] if role else self._seed
        if role:
            self.spent.add(role)
        try:
            with self.metrics.measure("envelope_signing" if role else "control_envelope_signing"):
                obj["signature"] = sign_statement(signing_seed, obj)
        finally:
            if role:
                signing_seed[:] = b"\x00" * len(signing_seed)
                del self._role_seeds[role]
        return obj

    def open(self, envelope):
        obj = json.loads(canonical(envelope))
        if set(obj) != {"schema", "session", "sender", "phase", "instance", "body", "vrf", "signature"}:
            raise ValueError("invalid native envelope fields")
        sender = obj["sender"]
        if (obj["schema"] != "ru22-native-envelope/v2" or obj["session"] != self.artifact["digest"]
                or type(sender) is not int or not 0 <= sender < self.artifact["population"]):
            raise ValueError("invalid native envelope identity")
        phase = obj["phase"]
        role = self._message_role(phase, obj["instance"], obj["body"])
        if role:
            keys = self.artifact["roles"][role]["signing_keys"]
            if str(sender) not in keys:
                raise ValueError("sender outside selected committee")
            public_key = keys[str(sender)]
        else:
            public_key = self.artifact["public"][sender]["signing"]
        sig = obj.pop("signature")
        with self.metrics.measure("envelope_verification" if role else "control_envelope_verification"):
            if not verify_statement(public_key, obj, sig):
                raise ValueError("native envelope signature mismatch")
        if role and role != "source":
            with self.metrics.measure("vrf_proof_verification"):
                valid = vrf_verify_sortition(label=self.label(role), proof=obj["vrf"],
                    sender_id=sender, vrf_pks=[p["vrf"] for p in self.artifact["public"]],
                    g_hex=raw_point(self.context.g_hex),
                    threshold_fraction=self.context.resharing_n / self.artifact["population"])
            if not valid:
                raise ValueError("invalid native role VRF proof")
        elif obj["vrf"] is not None:
            raise ValueError("unexpected native VRF proof")
        return sender, phase, obj["body"]

    def close(self):
        self._seed[:] = b"\x00" * len(self._seed)
        for seed in self._role_seeds.values():
            seed[:] = b"\x00" * len(seed)
        self._role_seeds.clear()
        self._vrf = None
        self.closed = True
