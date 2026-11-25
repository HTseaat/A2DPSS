"""Signature-cache equivalence, adversarial chains and work-count regressions."""

import copy

import pytest
from Crypto.PublicKey import ECC

from beaver.async_algorand.crypto import Signer, Verifier


def identities(count=1, session="cache-test"):
    keys = [ECC.generate(curve="Ed25519") for _ in range(count)]
    public = [{"signing": key.public_key().export_key(format="raw").hex()}
              for key in keys]
    signers = [Signer(session, node, key.seed.hex())
               for node, key in enumerate(keys)]
    return signers, Verifier(session, public)


def count_checks(verifier, monkeypatch):
    calls = []
    check = verifier._check

    def counted(*args):
        calls.append(args)
        return check(*args)

    monkeypatch.setattr(verifier, "_check", counted)
    return calls


@pytest.mark.parametrize("count", [20, 40, 80])
@pytest.mark.parametrize("reverse", [False, True])
def test_ratchet_extensions_verify_each_signature_once(count, reverse, monkeypatch):
    signers, verifier = identities()
    messages = [signers[0].seal("role-%d" % i, {"value": i}) for i in range(count)]
    calls = count_checks(verifier, monkeypatch)
    for message in messages[::(-1 if reverse else 1)]:
        assert verifier.envelope(message)
    assert len(calls) == count
    # Re-delivery remains valid without redoing any Ed25519 verification.
    for message in messages:
        assert verifier.envelope(message)
    assert len(calls) == count


@pytest.mark.parametrize("field", ["next", "hash", "topic", "signature", "extra"])
def test_cached_prefix_does_not_authorize_a_modified_link(field):
    signers, verifier = identities()
    messages = [signers[0].seal("role-%d" % i, {"value": i}) for i in range(3)]
    assert all(verifier.envelope(message) for message in messages)
    forged = copy.deepcopy(messages[-1])
    entry = forged["signed"]["chain"][0]
    if field == "signature":
        entry[field] = "00" * 64
    elif field == "extra":
        entry["link"][field] = "not-signed"
    else:
        entry["link"][field] = "00" * 32 if field != "topic" else "different-role"
    assert not verifier.envelope(forged)
    assert verifier.envelope(messages[-1])


def test_failed_signature_does_not_poison_link_cache(monkeypatch):
    signers, verifier = identities()
    message = signers[0].seal("first", {"value": 1})
    forged = copy.deepcopy(message)
    forged["signed"]["signature"] = "00" * 64
    calls = count_checks(verifier, monkeypatch)
    assert not verifier.envelope(forged)
    assert not verifier._verified_links
    assert verifier.envelope(message)
    assert len(calls) == 2
    assert not verifier.envelope(forged)
    assert len(calls) == 3


def test_cached_links_still_reject_duplicate_roles_and_retroactive_envelopes():
    signers, verifier = identities()
    signer = signers[0]
    first = signer.seal("same-role", {"value": 1})
    assert verifier.envelope(first)
    # A Byzantine signer can sign a second chain link for a consumed role.
    # Its detached signature is valid; the complete chain must still fail.
    signer.used.remove("same-role")
    repeated = signer.seal("same-role", {"value": 2})
    assert verifier.verify("envelope", repeated["link"], repeated["signed"])
    assert not verifier.envelope(repeated)
    tail = signer.seal("tail", {"value": 3})
    assert not verifier.envelope(tail)


def test_authenticated_forks_are_kept_but_cross_branch_splices_are_rejected():
    signers, verifier = identities()
    left = signers[0]
    head = left.seal("head", {"value": 0})
    right = Signer(left.session, left.node, left.key.seed.hex())
    right.chain = copy.deepcopy(left.chain)
    right.used = set(left.used)
    left_branch = left.seal("branch", {"value": "left"})
    right_branch = right.seal("branch", {"value": "right"})
    left_tail = left.seal("tail", {"value": "left"})
    right_tail = right.seal("tail", {"value": "right"})
    # The original verifier accepts independently authenticated forks, which
    # the transport keeps as equivocation evidence. Caching must preserve it.
    for message in (left_tail, right_tail, head, right_branch, left_branch):
        assert verifier.envelope(message)
    forged = copy.deepcopy(left_tail)
    forged["signed"]["chain"][1] = copy.deepcopy(right_tail["signed"]["chain"][1])
    assert not verifier.envelope(forged)


@pytest.mark.parametrize("change", ["reorder", "truncate"])
def test_authenticated_links_cannot_be_reordered_or_truncated(change):
    signers, verifier = identities()
    messages = [signers[0].seal("role-%d" % i, {"value": i}) for i in range(4)]
    assert verifier.envelope(messages[-1])
    forged = copy.deepcopy(messages[-1])
    if change == "reorder":
        forged["signed"]["chain"].reverse()
    else:
        forged["signed"]["chain"].pop()
    assert not verifier.envelope(forged)


def test_cache_is_bound_to_node_session_and_root_key():
    signers, verifier = identities(2)
    message = signers[0].seal("first", {"value": 1})
    assert verifier.envelope(message)
    # Even identical public-key entries do not make different node statements
    # interchangeable: node identity is part of every signed message.
    verifier.public[1] = dict(verifier.public[0])
    forged = copy.deepcopy(message)
    forged["signed"]["node"] = 1
    assert not verifier.envelope(forged)
    verifier.session = "another-session"
    assert not verifier.envelope(message)
    verifier.session = "cache-test"
    assert verifier.envelope(message)
    verifier.public[0] = {"signing": ECC.generate(curve="Ed25519").public_key().export_key(format="raw").hex()}
    assert not verifier.envelope(message)


def test_detached_attestation_domain_cannot_prime_an_envelope_link():
    signers, verifier = identities()
    signer = signers[0]
    message = signer.seal("first", {"value": 1})
    assert verifier.envelope(message)
    assert not verifier.verify("nwa-vote", message["link"], message["signed"])
    other = signer.seal("second", {"value": 2})
    # Sign the very same link in another domain at the current ratchet key.
    attestation = signer.attest("nwa-vote", other["link"])
    assert verifier.verify("nwa-vote", other["link"], attestation)
    assert not verifier.verify("envelope", other["link"], attestation)


def test_bad_outer_payload_cannot_populate_signature_link_cache():
    signers, verifier = identities()
    message = signers[0].seal("first", {"value": 1})
    forged = copy.deepcopy(message)
    forged["body"]["value"] = 2
    assert not verifier.envelope(forged)
    assert not verifier._verified_links
    assert verifier.envelope(message)
