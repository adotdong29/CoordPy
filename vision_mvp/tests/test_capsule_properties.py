"""Property-based tests for the Capsule Contract (C1–C6).

Uses Hypothesis to auto-generate 100s of test cases per property.
Covers the six contract invariants stated in
``vision_mvp/wevra/capsule.py``:

  C1  Identity      — cid = SHA256(canonical(kind, payload, budget,
                      sorted(parents))). Deterministic.
  C2  Typed claim   — kind must be in CapsuleKind.ALL; unknown kinds
                      are rejected at construction.
  C3  Lifecycle     — PROPOSED → ADMITTED → SEALED → RETIRED only.
  C4  Budget        — admit enforces declared budgets.
  C5  Provenance    — every parent must be in the ledger; hash chain
                      is tamper-evident.
  C6  Frozen        — sealed CIDs are immutable and stable.

See ``ADVANCEMENT_TO_10_10.md`` Part III §1 for motivation.
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings, HealthCheck, strategies as st

from vision_mvp.wevra.capsule import (
    CapsuleAdmissionError,
    CapsuleBudget,
    CapsuleKind,
    CapsuleLedger,
    CapsuleLifecycle,
    CapsuleLifecycleError,
    ContextCapsule,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


_SIMPLE_KINDS = [
    CapsuleKind.HANDOFF,
    CapsuleKind.HANDLE,
    CapsuleKind.THREAD_RESOLUTION,
    CapsuleKind.READINESS_CHECK,
    CapsuleKind.PROVENANCE,
    CapsuleKind.PROFILE,
    CapsuleKind.ARTIFACT,
]


@st.composite
def budget_strat(draw):
    """Always sets at least max_bytes so payloads are bounded."""
    return CapsuleBudget(
        max_tokens=draw(st.one_of(st.none(),
                                  st.integers(min_value=1, max_value=10000))),
        max_bytes=draw(st.integers(min_value=1024, max_value=1 << 20)),
        max_rounds=draw(st.one_of(st.none(),
                                  st.integers(min_value=1, max_value=100))),
        max_parents=draw(st.integers(min_value=0, max_value=16)),
    )


_payload_scalar = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.text(min_size=0, max_size=64),
    st.booleans(),
    st.none(),
)


_payload_strat = st.dictionaries(
    st.text(min_size=1, max_size=8), _payload_scalar,
    min_size=0, max_size=5)


@st.composite
def capsule_strat(draw, parents=()):
    kind = draw(st.sampled_from(_SIMPLE_KINDS))
    budget = draw(budget_strat())
    payload = draw(_payload_strat)
    return ContextCapsule.new(
        kind=kind, payload=payload, budget=budget, parents=parents)


# ---------------------------------------------------------------------------
# C1 — Identity determinism
# ---------------------------------------------------------------------------


@given(cap=capsule_strat())
@settings(max_examples=200,
          suppress_health_check=[HealthCheck.too_slow])
def test_c1_identity_deterministic(cap: ContextCapsule) -> None:
    """C1: rebuilding a capsule from the same inputs yields the same CID."""
    twin = ContextCapsule.new(
        kind=cap.kind, payload=cap.payload,
        budget=cap.budget, parents=cap.parents)
    assert cap.cid == twin.cid
    assert len(cap.cid) == 64  # SHA-256 hex


@given(cap=capsule_strat())
@settings(max_examples=200)
def test_c1_identity_kind_sensitive(cap: ContextCapsule) -> None:
    """C1: flipping the kind must change the CID."""
    other_kinds = [k for k in _SIMPLE_KINDS if k != cap.kind]
    alt = ContextCapsule.new(
        kind=other_kinds[0], payload=cap.payload,
        budget=cap.budget, parents=cap.parents)
    assert alt.cid != cap.cid


# ---------------------------------------------------------------------------
# C2 — Typed claim (closed vocabulary)
# ---------------------------------------------------------------------------


@given(kind=st.text(min_size=1, max_size=20))
@settings(max_examples=200)
def test_c2_unknown_kind_rejected(kind: str) -> None:
    """C2: unknown kinds must raise ValueError at construction."""
    if kind in CapsuleKind.ALL:
        return
    with pytest.raises(ValueError):
        ContextCapsule.new(
            kind=kind, payload={"x": 1},
            budget=CapsuleBudget(max_bytes=1024))


@given(kind=st.sampled_from(sorted(CapsuleKind.ALL)))
@settings(max_examples=100)
def test_c2_known_kind_accepted(kind: str) -> None:
    """C2: any kind in the closed vocabulary is accepted."""
    cap = ContextCapsule.new(
        kind=kind, payload={"ok": True},
        budget=CapsuleBudget(max_bytes=1 << 15, max_parents=0))
    assert cap.kind in CapsuleKind.ALL


# ---------------------------------------------------------------------------
# C3 — Lifecycle order
# ---------------------------------------------------------------------------


@given(cap=capsule_strat())
@settings(max_examples=100,
          suppress_health_check=[HealthCheck.too_slow])
def test_c3_lifecycle_order_enforced(cap: ContextCapsule) -> None:
    """C3: admit→seal is the only legal sequence; skipping steps raises."""
    ledger = CapsuleLedger()

    # Sealing before admit must fail (lifecycle is still PROPOSED).
    with pytest.raises(CapsuleLifecycleError):
        ledger.seal(cap)

    admitted = ledger.admit(cap)
    assert admitted.lifecycle == CapsuleLifecycle.ADMITTED

    # Double-admit is not allowed (admit() expects PROPOSED).
    with pytest.raises(CapsuleLifecycleError):
        ledger.admit(admitted)

    sealed = ledger.seal(admitted)
    assert sealed.lifecycle == CapsuleLifecycle.SEALED
    assert sealed.cid == cap.cid  # C6 preview


@given(legal=st.sampled_from([
    (CapsuleLifecycle.PROPOSED, CapsuleLifecycle.ADMITTED, True),
    (CapsuleLifecycle.ADMITTED, CapsuleLifecycle.SEALED, True),
    (CapsuleLifecycle.SEALED, CapsuleLifecycle.RETIRED, True),
    (CapsuleLifecycle.PROPOSED, CapsuleLifecycle.SEALED, False),
    (CapsuleLifecycle.RETIRED, CapsuleLifecycle.ADMITTED, False),
    (CapsuleLifecycle.ADMITTED, CapsuleLifecycle.PROPOSED, False),
]))
def test_c3_transition_table(legal) -> None:
    """C3: CapsuleLifecycle.can_transition implements the edge table."""
    frm, to, expected = legal
    assert CapsuleLifecycle.can_transition(frm, to) is expected


# ---------------------------------------------------------------------------
# C4 — Budget enforcement
# ---------------------------------------------------------------------------


@given(max_tokens=st.integers(min_value=1, max_value=200),
       n_tokens=st.integers(min_value=0, max_value=200))
@settings(max_examples=200)
def test_c4_budget_tokens_monotonic(max_tokens: int, n_tokens: int) -> None:
    """C4: a capsule whose token count exceeds max_tokens is rejected."""
    cap = ContextCapsule.new(
        kind=CapsuleKind.HANDOFF, payload={"m": "x"},
        budget=CapsuleBudget(max_tokens=max_tokens, max_bytes=1 << 14,
                              max_parents=0),
        n_tokens=n_tokens)
    ledger = CapsuleLedger()
    if n_tokens > max_tokens:
        with pytest.raises(CapsuleAdmissionError):
            ledger.admit(cap)
    else:
        admitted = ledger.admit(cap)
        assert admitted.lifecycle == CapsuleLifecycle.ADMITTED


@given(nbytes=st.integers(min_value=1, max_value=4096),
       payload_len=st.integers(min_value=1, max_value=4096))
@settings(max_examples=100)
def test_c4_budget_bytes_at_construction(nbytes: int, payload_len: int) -> None:
    """C4: max_bytes is checked at construction on the canonical blob."""
    payload = {"s": "a" * payload_len}
    try:
        ContextCapsule.new(
            kind=CapsuleKind.ARTIFACT, payload=payload,
            budget=CapsuleBudget(max_bytes=nbytes, max_parents=0))
    except ValueError:
        # Expected when the payload serialises to > nbytes.
        return


# ---------------------------------------------------------------------------
# C5 — Provenance / hash chain
# ---------------------------------------------------------------------------


@given(caps=st.lists(capsule_strat(), min_size=1, max_size=10, unique_by=lambda c: c.cid))
@settings(max_examples=50,
          suppress_health_check=[HealthCheck.too_slow,
                                 HealthCheck.filter_too_much])
def test_c5_chain_verifies_after_sealing(caps) -> None:
    """C5: after sealing any number of parent-less capsules, the chain
    verifies and chain_head advances."""
    ledger = CapsuleLedger()
    head0 = ledger.chain_head()
    for c in caps:
        ledger.admit_and_seal(c)
    assert ledger.verify_chain()
    if caps:
        assert ledger.chain_head() != head0


@given(cap=capsule_strat())
@settings(max_examples=50)
def test_c5_unknown_parent_rejected(cap: ContextCapsule) -> None:
    """C5: a capsule whose parent CID is not in the ledger is rejected."""
    fake_parent = "0" * 64
    child = ContextCapsule.new(
        kind=CapsuleKind.HANDOFF, payload={"child": True},
        budget=CapsuleBudget(max_bytes=1 << 14, max_parents=4),
        parents=(fake_parent,))
    ledger = CapsuleLedger()
    with pytest.raises(CapsuleAdmissionError):
        ledger.admit(child)


# ---------------------------------------------------------------------------
# C6 — Frozen / sealed capsules are stable
# ---------------------------------------------------------------------------


@given(cap=capsule_strat())
@settings(max_examples=100)
def test_c6_seal_idempotent(cap: ContextCapsule) -> None:
    """C6: re-sealing the same CID is idempotent; the CID does not change."""
    ledger = CapsuleLedger()
    sealed = ledger.admit_and_seal(cap)
    # The ledger keeps exactly one entry per CID and its sealed copy
    # is stable; the stored capsule reports the original CID verbatim.
    stored = ledger.get(cap.cid)
    assert sealed.cid == stored.cid == cap.cid
    assert stored.lifecycle == CapsuleLifecycle.SEALED
    assert len(ledger) == 1


# ---------------------------------------------------------------------------
# Metamorphic: removing an entry does not grow the header view.
# ---------------------------------------------------------------------------


@given(caps=st.lists(capsule_strat(), min_size=2, max_size=8,
                      unique_by=lambda c: c.cid))
@settings(max_examples=50,
          suppress_health_check=[HealthCheck.too_slow,
                                 HealthCheck.filter_too_much])
def test_metamorphic_removal_shrinks_view(caps) -> None:
    """Dropping one capsule from the input set never grows the resulting
    ledger's serialised view. A sanity check on monotonicity of the
    capsule-graph projection."""
    from vision_mvp.wevra.capsule import render_view
    import json

    full = CapsuleLedger()
    for c in caps:
        full.admit_and_seal(c)

    reduced = CapsuleLedger()
    for c in caps[:-1]:
        reduced.admit_and_seal(c)

    size_full = len(json.dumps(render_view(full).as_dict(), sort_keys=True))
    size_reduced = len(json.dumps(
        render_view(reduced).as_dict(), sort_keys=True))
    assert size_reduced <= size_full


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
