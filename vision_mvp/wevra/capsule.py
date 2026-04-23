"""Context Capsules — Wevra's load-bearing abstraction.

A **Context Capsule** is a typed, content-addressed, lifecycle-bounded,
budget-bounded, provenance-carrying unit of coordination. Every piece
of "context" that crosses a role boundary, a layer boundary, or a run
boundary in Wevra is a capsule. Handles (Phase 19), typed handoffs
(Phase 31), escalation thread resolutions (Phase 35), adaptive
subscription edges (Phase 36), sweep cells, and the product report
itself all conform to the Capsule Contract — capsules are the thing
that was always there but was never named.

The Capsule Contract — six invariants
--------------------------------------

  C1  **Identity.**       A capsule has a stable content address
                          ``cid`` (SHA-256) derived from its
                          ``(kind, payload, budget, parents)``. Two
                          capsules with the same content collapse to
                          one CID.
  C2  **Typed claim.**    A capsule carries a ``kind`` — a short
                          enumerated semantic type (``HANDOFF``,
                          ``HANDLE``, ``THREAD_RESOLUTION``,
                          ``SWEEP_CELL``, ``PROVENANCE``,
                          ``RUN_REPORT``, …). Untyped capsules are
                          illegal.
  C3  **Lifecycle.**      Every capsule traverses an explicit
                          lifecycle: ``PROPOSED → ADMITTED → SEALED``
                          (+ optional ``RETIRED``). ``SEALED`` is
                          terminal on identity — a sealed capsule's
                          CID is fixed for all time. Transitions are
                          recorded.
  C4  **Budget.**         A capsule carries an explicit
                          ``CapsuleBudget`` (tokens, bytes, rounds,
                          witnesses). Admission enforces the budget
                          — a capsule that would exceed is rejected.
  C5  **Provenance.**     A capsule records its parent CIDs,
                          forming an audit DAG. The ledger maintains
                          a hash chain so the sequence of capsules
                          is tamper-evident (any retroactive insert
                          breaks the chain).
  C6  **Frozen.**         A sealed capsule is immutable. Its
                          ``cid`` is the only proof of authenticity
                          the SDK offers; if the bytes change, the
                          CID must change.

What "every piece of context is a capsule" actually means
----------------------------------------------------------

In a traditional agent / eval harness:
  * a *prompt string* crosses from caller to LLM.
  * a *function-call dict* crosses from LLM to tool.
  * a *retrieved chunk* crosses from index to caller.
  * a *run result* crosses from runtime to reporter.
These are raw strings or raw dicts; they have no identity, no type,
no lifecycle, no budget, no provenance.

In Wevra v3, every one of those becomes a ``ContextCapsule``. The
prompt LLM receives is the concatenation of typed capsules, not a
bag of bytes. The tool output is sealed as a capsule with a parent
link to the prompt capsule. The run report is a meta-capsule whose
parents are every capsule that crossed a boundary during the run.
The resulting on-disk artifact is not just a JSON file — it is a
**capsule graph** that can be audited, replayed, and traced.

What this module provides
-------------------------

  * ``CapsuleKind``        — enumerated semantic kinds.
  * ``CapsuleLifecycle``   — the four lifecycle states.
  * ``CapsuleBudget``      — an explicit budget record.
  * ``ContextCapsule``     — the frozen capsule type.
  * ``CapsuleLedger``      — an append-only, hash-chained
                              container. Admits capsules subject
                              to their budgets; seals them; exposes
                              a capsule graph.
  * ``CapsuleView``        — a capsule graph slice, safe to
                              serialise into a product report.
  * ``capsule_from_handle`` — adapter from ``context_ledger.Handle``.
  * ``capsule_from_handoff`` — adapter from ``role_handoff.TypedHandoff``.
  * ``capsule_from_report``  — adapter from a ``product_report.json``
                               dict (produces a ``RUN_REPORT`` capsule
                               whose parents are every boundary-
                               crossing capsule in the run).

Scope discipline
----------------

This module is the SDK's central abstraction but it does NOT replace
any substrate primitive. ``TypedHandoff`` stays ``TypedHandoff``;
``Handle`` stays ``Handle``; the Phase-31 ``HandoffRouter`` is byte-
for-byte unchanged. What this module adds is a *shared contract*
those primitives can all be viewed under — and a ledger that
actually collects them.

Honest originality
------------------

The individual primitives are inherited, not invented:
  * content-addressing from Merkle/Git/IPFS;
  * hash-chained logs from tamper-evident-log research;
  * typed claim kinds from actor / event-sourcing systems;
  * capability-style typed references from KeyKOS / seL4;
  * lifecycle states from session-typed protocols.
What is new in Wevra is the unification: one contract every piece
of coordination context in the SDK must satisfy, at the product
surface. The claim is that "context-as-object" is the correct
top-level abstraction for LLM-agent-team runtimes — and that it
is actually implemented here, not just asserted.

Theoretical anchor: ``docs/RESULTS_WEVRA_CAPSULE.md``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from typing import Any, Iterable


# =============================================================================
# Kinds — closed vocabulary for capsule semantic types
# =============================================================================


class CapsuleKind:
    """Canonical capsule kinds. Closed vocabulary — unknown kinds
    are rejected at capsule construction. Adding a kind is an
    SDK-version bump signal."""

    HANDOFF = "HANDOFF"
    """An inter-role typed handoff. Adapter: ``capsule_from_handoff``.
    Maps to ``role_handoff.TypedHandoff`` (Phase 31)."""

    HANDLE = "HANDLE"
    """An exact-memory handle into the context ledger. Adapter:
    ``capsule_from_handle``. Maps to ``context_ledger.Handle``
    (Phase 19)."""

    THREAD_RESOLUTION = "THREAD_RESOLUTION"
    """The single public output of an escalation thread
    (Phase 35 ``dynamic_comm``). The thread-internal reply chain is
    its parent DAG."""

    ADAPTIVE_EDGE = "ADAPTIVE_EDGE"
    """A bounded, TTL-expiring subscription edit (Phase 36
    ``adaptive_sub``). Budget carries the TTL."""

    SWEEP_CELL = "SWEEP_CELL"
    """One cell of a unified-runtime sweep — (parser_mode,
    apply_mode, n_distractors, pooled results, sandbox). Parent:
    the SweepSpec that produced it."""

    SWEEP_SPEC = "SWEEP_SPEC"
    """A frozen ``wevra.runtime.SweepSpec``. Parent: the profile
    capsule."""

    READINESS_CHECK = "READINESS_CHECK"
    """One ``phase44_public_readiness`` check's verdict
    (schema / adapter / parser / matcher / test_runner)."""

    PROVENANCE = "PROVENANCE"
    """The provenance manifest (``wevra.provenance.v1``) — git SHA,
    package version, JSONL SHA-256, argv, artifact list. One per
    run."""

    RUN_REPORT = "RUN_REPORT"
    """The top-level product report capsule. Parents are every
    other capsule in the run. One per run. The run-report capsule's
    CID is the durable identifier for a Wevra run."""

    PROFILE = "PROFILE"
    """A resolved profile dict (name, schema, readiness cfg, sweep
    cfg, trust tag)."""

    ARTIFACT = "ARTIFACT"
    """An on-disk artifact descriptor (path + SHA-256). Lets
    downstream consumers prove the file they are reading is the
    file the run produced."""

    COHORT = "COHORT"
    """A bounded-membership container capsule. Its ``parents`` are
    the CIDs of its members; its ``max_parents`` budget is the
    cardinality cap on the cohort. Phase 47 addition — the
    resolution of the W3-C3 honest-falsification: table-level
    bounded-context invariants (e.g. Phase-36 AdaptiveEdge's
    ``max_active_edges``) are subsumed by emitting a COHORT
    capsule whose parent set is the active members and whose
    ``max_parents`` is the table-level cap. See
    ``docs/CAPSULE_FORMALISM.md`` § 4 (Theorems W3-14 / W3-15 /
    W3-16) for the formal statement and the honest relational
    limitation that cohort lifting still does NOT close."""

    ALL = frozenset({
        HANDOFF, HANDLE, THREAD_RESOLUTION, ADAPTIVE_EDGE,
        SWEEP_CELL, SWEEP_SPEC, READINESS_CHECK, PROVENANCE,
        RUN_REPORT, PROFILE, ARTIFACT, COHORT,
    })


# =============================================================================
# Lifecycle
# =============================================================================


class CapsuleLifecycle:
    """The four capsule lifecycle states.

    A capsule is born ``PROPOSED``. Admission checks the budget and
    moves it to ``ADMITTED``. Sealing freezes the CID and moves to
    ``SEALED``. A capsule may later be marked ``RETIRED`` (e.g.
    on TTL expiry for ``ADAPTIVE_EDGE``) — ``RETIRED`` does NOT
    remove the capsule from the ledger; it only annotates that the
    capsule no longer contributes to the active context.
    """

    PROPOSED = "PROPOSED"
    ADMITTED = "ADMITTED"
    SEALED = "SEALED"
    RETIRED = "RETIRED"

    ALL = (PROPOSED, ADMITTED, SEALED, RETIRED)

    # Legal forward transitions.
    _EDGES = {
        PROPOSED: frozenset({ADMITTED}),
        ADMITTED: frozenset({SEALED}),
        SEALED: frozenset({RETIRED}),
        RETIRED: frozenset(),
    }

    @classmethod
    def can_transition(cls, frm: str, to: str) -> bool:
        return to in cls._EDGES.get(frm, frozenset())


# =============================================================================
# Budget
# =============================================================================


@dataclasses.dataclass(frozen=True)
class CapsuleBudget:
    """Explicit resource budget carried by every capsule.

    The ledger's ``admit`` step rejects any capsule that would push
    one of its declared limits past its budget. Budgets that don't
    apply to a given kind are left as ``None`` — unbounded on that
    axis. At least one axis must be set for a capsule to be admitted
    (Contract invariant C4); the class refuses a fully-null budget.

    The common axes are:

      * ``max_tokens``   — payload token estimate. Applies to
                            ``HANDOFF`` / ``HANDLE`` /
                            ``THREAD_RESOLUTION`` etc.
      * ``max_bytes``    — serialised payload byte budget.
                            Applies to ``ARTIFACT`` / ``RUN_REPORT``.
      * ``max_rounds``   — coordination-round TTL. Applies to
                            ``ADAPTIVE_EDGE`` / ``THREAD_RESOLUTION``.
      * ``max_witnesses`` — max witness tokens per reply (Phase 35).
      * ``max_parents``   — max parent capsules in the DAG. A
                            sanity cap; a capsule whose parent set
                            exceeds this is almost certainly a
                            design error.
    """

    max_tokens: int | None = None
    max_bytes: int | None = None
    max_rounds: int | None = None
    max_witnesses: int | None = None
    max_parents: int | None = None

    def __post_init__(self) -> None:
        axes = (self.max_tokens, self.max_bytes, self.max_rounds,
                self.max_witnesses, self.max_parents)
        if all(a is None for a in axes):
            raise ValueError(
                "CapsuleBudget must set at least one axis "
                "(Capsule Contract invariant C4). Pass e.g. "
                "CapsuleBudget(max_bytes=1<<16).")
        for name, v in (
                ("max_tokens", self.max_tokens),
                ("max_bytes", self.max_bytes),
                ("max_rounds", self.max_rounds),
                ("max_witnesses", self.max_witnesses),
                ("max_parents", self.max_parents)):
            if v is not None and v < 0:
                raise ValueError(f"{name} must be ≥ 0, got {v}")

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in dataclasses.asdict(self).items()
                if v is not None}


def _default_budget_for(kind: str) -> CapsuleBudget:
    """Return a sensible default budget for a capsule kind.

    Defaults are conservative but permissive — they exist so
    callers can construct a capsule without naming every axis, and
    so an admission failure is a loud signal, not a surprise.
    """
    if kind == CapsuleKind.HANDOFF:
        return CapsuleBudget(max_tokens=256, max_parents=16)
    if kind == CapsuleKind.HANDLE:
        return CapsuleBudget(max_bytes=1 << 20, max_parents=8)
    if kind == CapsuleKind.THREAD_RESOLUTION:
        return CapsuleBudget(max_tokens=512, max_rounds=8,
                              max_witnesses=64, max_parents=32)
    if kind == CapsuleKind.ADAPTIVE_EDGE:
        return CapsuleBudget(max_rounds=4, max_parents=8)
    if kind == CapsuleKind.SWEEP_CELL:
        return CapsuleBudget(max_bytes=1 << 16, max_parents=8)
    if kind == CapsuleKind.SWEEP_SPEC:
        return CapsuleBudget(max_bytes=4096, max_parents=4)
    if kind == CapsuleKind.READINESS_CHECK:
        return CapsuleBudget(max_bytes=1 << 15, max_parents=4)
    if kind == CapsuleKind.PROVENANCE:
        return CapsuleBudget(max_bytes=1 << 16, max_parents=4)
    if kind == CapsuleKind.RUN_REPORT:
        return CapsuleBudget(max_bytes=1 << 22, max_parents=1024)
    if kind == CapsuleKind.PROFILE:
        return CapsuleBudget(max_bytes=1 << 14, max_parents=0)
    if kind == CapsuleKind.ARTIFACT:
        return CapsuleBudget(max_bytes=1 << 30, max_parents=4)
    if kind == CapsuleKind.COHORT:
        # Default cohort cap: a conservative 64. The cohort-cap
        # ``max_parents`` is the operational form of the Phase-36
        # ``max_active_edges`` invariant; every cohort-using caller
        # sets it explicitly and a 64 default means an unconfigured
        # cohort is loud but usable.
        return CapsuleBudget(max_parents=64, max_bytes=1 << 17)
    raise ValueError(f"no default budget for kind {kind!r}")


# =============================================================================
# ContextCapsule
# =============================================================================


def _canonical(obj: Any) -> bytes:
    """Canonical JSON encoding — deterministic across runs."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       default=str).encode("utf-8")


def _capsule_cid(kind: str, payload: Any, budget: CapsuleBudget,
                 parents: Iterable[str]) -> str:
    """Content address = SHA-256 over (kind, payload, budget,
    sorted parents). Parents are sorted so two capsules with the
    same parent set collapse regardless of insertion order."""
    blob = _canonical({
        "kind": kind,
        "payload": payload,
        "budget": budget.as_dict(),
        "parents": sorted(parents),
    })
    return hashlib.sha256(blob).hexdigest()


@dataclasses.dataclass(frozen=True)
class ContextCapsule:
    """A typed, content-addressed, lifecycle-bounded,
    budget-bounded, provenance-carrying unit of coordination.

    See module docstring for the Capsule Contract (C1..C6). The
    ``cid`` is computed once at construction and is the capsule's
    identity for all subsequent operations.
    """

    cid: str
    kind: str
    payload: Any
    budget: CapsuleBudget
    parents: tuple[str, ...]
    lifecycle: str = CapsuleLifecycle.PROPOSED
    # Best-effort token / byte counts. The ledger uses them for
    # admission; if unset, the ledger falls back to measuring the
    # canonical payload encoding.
    n_tokens: int | None = None
    n_bytes: int | None = None
    emitted_at: float = 0.0
    metadata: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def new(cls, *, kind: str, payload: Any,
             budget: CapsuleBudget | None = None,
             parents: Iterable[str] = (),
             n_tokens: int | None = None,
             n_bytes: int | None = None,
             metadata: dict[str, Any] | None = None,
             ) -> "ContextCapsule":
        """Build a fresh ``PROPOSED`` capsule.

        This is the public constructor. It enforces C1 (CID
        derived from content) and C2 (kind must be in
        ``CapsuleKind.ALL``). Admission / sealing is the
        ledger's job (C3, C4).
        """
        if kind not in CapsuleKind.ALL:
            raise ValueError(
                f"unknown capsule kind {kind!r}; "
                f"must be one of {sorted(CapsuleKind.ALL)}")
        bd = budget if budget is not None else _default_budget_for(kind)
        ps = tuple(parents)
        if (bd.max_parents is not None
                and len(ps) > bd.max_parents):
            raise ValueError(
                f"capsule has {len(ps)} parents but budget.max_parents="
                f"{bd.max_parents}")
        # We canonicalise to measure bytes; cheap and lets the
        # budget check at admit time use a truthful number.
        blob = _canonical(payload)
        nb = n_bytes if n_bytes is not None else len(blob)
        if (bd.max_bytes is not None and nb > bd.max_bytes):
            raise ValueError(
                f"capsule payload is {nb} bytes but "
                f"budget.max_bytes={bd.max_bytes}")
        cid = _capsule_cid(kind, payload, bd, ps)
        md = tuple(sorted((metadata or {}).items()))
        return cls(
            cid=cid, kind=kind, payload=payload, budget=bd,
            parents=ps, lifecycle=CapsuleLifecycle.PROPOSED,
            n_tokens=n_tokens, n_bytes=nb,
            emitted_at=time.time(), metadata=md,
        )

    def metadata_dict(self) -> dict[str, Any]:
        return dict(self.metadata)

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe projection for serialisation into the product
        report. The payload is included verbatim; callers who want
        a lighter view should use ``as_header_dict``."""
        return {
            "cid": self.cid,
            "kind": self.kind,
            "payload": self.payload,
            "budget": self.budget.as_dict(),
            "parents": list(self.parents),
            "lifecycle": self.lifecycle,
            "n_tokens": self.n_tokens,
            "n_bytes": self.n_bytes,
            "emitted_at": round(self.emitted_at, 3),
            "metadata": self.metadata_dict(),
        }

    def as_header_dict(self) -> dict[str, Any]:
        """Lightweight projection — CID, kind, parents, lifecycle,
        sizes. No payload. The shape the ``CapsuleView`` default
        renders into the product report: readers who want the
        payload look it up by CID."""
        return {
            "cid": self.cid,
            "kind": self.kind,
            "parents": list(self.parents),
            "lifecycle": self.lifecycle,
            "n_tokens": self.n_tokens,
            "n_bytes": self.n_bytes,
        }


# =============================================================================
# CapsuleLedger — append-only, hash-chained
# =============================================================================


class CapsuleAdmissionError(Exception):
    """Raised when a capsule's budget is exceeded at admit time,
    or when a capsule's declared parents are unknown to the
    ledger (Contract invariant C5)."""


class CapsuleLifecycleError(Exception):
    """Raised when a lifecycle transition violates
    ``CapsuleLifecycle._EDGES`` (Contract invariant C3)."""


@dataclasses.dataclass
class _LedgerEntry:
    capsule: ContextCapsule
    chain_hash: str
    prev_chain_hash: str


def _chain_step(prev: str, capsule: ContextCapsule) -> str:
    blob = _canonical({
        "prev": prev, "cid": capsule.cid,
        "kind": capsule.kind, "lifecycle": capsule.lifecycle,
    })
    return hashlib.sha256(blob).hexdigest()


class CapsuleLedger:
    """Append-only, hash-chained ledger of context capsules.

    The ledger is the concrete realisation of Contract invariants
    C3 (lifecycle), C4 (budget), C5 (provenance/chain), and C6
    (frozen). Usage:

        lg = CapsuleLedger()
        c = ContextCapsule.new(kind=CapsuleKind.HANDOFF,
                                payload={"msg": "hi"},
                                budget=CapsuleBudget(max_tokens=64))
        c = lg.admit(c)     # checks budget + parents
        c = lg.seal(c)      # C6 seal; CID is now fixed forever

    ``admit`` returns a new capsule (lifecycle bumped); ``seal``
    likewise. Callers hold the returned object; the ledger keeps
    its own copy keyed by CID.

    Hash-chain invariant: every admit+seal step extends a chain
    rooted at ``GENESIS``. ``verify_chain`` recomputes every link
    and returns False on the first divergence — tamper / truncation
    detector.
    """

    GENESIS = "GENESIS"

    def __init__(self) -> None:
        self._entries: list[_LedgerEntry] = []
        self._by_cid: dict[str, ContextCapsule] = {}
        self._chain_head: str = self.GENESIS
        self._n_admitted: int = 0
        self._n_sealed: int = 0
        self._n_retired: int = 0
        self._n_rejected: int = 0

    # ------------------------------------------------------------
    # Core transitions
    # ------------------------------------------------------------

    def admit(self, capsule: ContextCapsule) -> ContextCapsule:
        """Admit a ``PROPOSED`` capsule. Enforces C4 (budget) and
        C5 (every parent is known). Returns an ``ADMITTED`` copy.

        Idempotent on CID: re-admitting the same CID returns the
        existing ledger entry's capsule rather than double-counting.
        """
        if capsule.lifecycle != CapsuleLifecycle.PROPOSED:
            raise CapsuleLifecycleError(
                f"admit() expects PROPOSED, got {capsule.lifecycle}")
        # Idempotent on CID.
        if capsule.cid in self._by_cid:
            return self._by_cid[capsule.cid]
        # C5: every parent must be in the ledger.
        for p in capsule.parents:
            if p not in self._by_cid:
                self._n_rejected += 1
                raise CapsuleAdmissionError(
                    f"capsule parent {p[:12]}… is not in the "
                    f"ledger (Contract invariant C5)")
        # C4: re-check byte / token budget against the measured
        # payload size. The constructor already checked max_bytes
        # and max_parents; here we cover max_tokens.
        b = capsule.budget
        if (b.max_tokens is not None and capsule.n_tokens is not None
                and capsule.n_tokens > b.max_tokens):
            self._n_rejected += 1
            raise CapsuleAdmissionError(
                f"capsule has {capsule.n_tokens} tokens but "
                f"budget.max_tokens={b.max_tokens}")
        admitted = dataclasses.replace(
            capsule, lifecycle=CapsuleLifecycle.ADMITTED)
        self._n_admitted += 1
        return admitted

    def seal(self, capsule: ContextCapsule) -> ContextCapsule:
        """Seal an ``ADMITTED`` capsule. C6: after this point the
        capsule's CID is fixed for all time; the ledger stores the
        sealed copy and appends a chain-hash link.

        Idempotent on CID.
        """
        if capsule.lifecycle != CapsuleLifecycle.ADMITTED:
            raise CapsuleLifecycleError(
                f"seal() expects ADMITTED, got {capsule.lifecycle}")
        if capsule.cid in self._by_cid:
            return self._by_cid[capsule.cid]
        sealed = dataclasses.replace(
            capsule, lifecycle=CapsuleLifecycle.SEALED)
        prev = self._chain_head
        ch = _chain_step(prev, sealed)
        self._entries.append(_LedgerEntry(
            capsule=sealed, chain_hash=ch, prev_chain_hash=prev))
        self._by_cid[sealed.cid] = sealed
        self._chain_head = ch
        self._n_sealed += 1
        return sealed

    def retire(self, cid: str) -> ContextCapsule:
        """Mark a sealed capsule as ``RETIRED``. The capsule stays
        in the ledger (C6 preserves the CID's historical meaning);
        only the lifecycle state is annotated. Returns the retired
        copy."""
        if cid not in self._by_cid:
            raise KeyError(cid)
        cap = self._by_cid[cid]
        if cap.lifecycle == CapsuleLifecycle.RETIRED:
            return cap
        if not CapsuleLifecycle.can_transition(
                cap.lifecycle, CapsuleLifecycle.RETIRED):
            raise CapsuleLifecycleError(
                f"retire() cannot transition from "
                f"{cap.lifecycle} to RETIRED")
        retired = dataclasses.replace(
            cap, lifecycle=CapsuleLifecycle.RETIRED)
        self._by_cid[cid] = retired
        # Rewrite the entry in place — the hash chain is computed
        # over the sealed state, not the retirement annotation, so
        # retiring does not break ``verify_chain``.
        for i, e in enumerate(self._entries):
            if e.capsule.cid == cid:
                self._entries[i] = _LedgerEntry(
                    capsule=retired,
                    chain_hash=e.chain_hash,
                    prev_chain_hash=e.prev_chain_hash,
                )
                break
        self._n_retired += 1
        return retired

    def admit_and_seal(self, capsule: ContextCapsule) -> ContextCapsule:
        """Convenience: admit then seal. The common path."""
        return self.seal(self.admit(capsule))

    # ------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------

    def get(self, cid: str) -> ContextCapsule:
        return self._by_cid[cid]

    def __contains__(self, cid: str) -> bool:
        return cid in self._by_cid

    def __len__(self) -> int:
        return len(self._entries)

    def all_capsules(self) -> list[ContextCapsule]:
        return [e.capsule for e in self._entries]

    def by_kind(self, kind: str) -> list[ContextCapsule]:
        return [e.capsule for e in self._entries
                 if e.capsule.kind == kind]

    def parents_of(self, cid: str) -> list[ContextCapsule]:
        cap = self._by_cid[cid]
        return [self._by_cid[p] for p in cap.parents
                 if p in self._by_cid]

    def children_of(self, cid: str) -> list[ContextCapsule]:
        """Capsules that declare ``cid`` as a parent. O(n)
        scan — the ledger is expected to be small."""
        return [e.capsule for e in self._entries
                 if cid in e.capsule.parents]

    def ancestors_of(self, cid: str) -> list[ContextCapsule]:
        """BFS walk of the parent DAG. Useful for "what derived
        this RUN_REPORT?"."""
        if cid not in self._by_cid:
            raise KeyError(cid)
        seen = {cid}
        out: list[ContextCapsule] = []
        frontier = [cid]
        while frontier:
            nxt: list[str] = []
            for c in frontier:
                for p in self._by_cid[c].parents:
                    if p in seen or p not in self._by_cid:
                        continue
                    seen.add(p)
                    out.append(self._by_cid[p])
                    nxt.append(p)
            frontier = nxt
        return out

    # ------------------------------------------------------------
    # Chain verification
    # ------------------------------------------------------------

    def chain_head(self) -> str:
        return self._chain_head

    def verify_chain(self) -> bool:
        prev = self.GENESIS
        for e in self._entries:
            # Recompute from the sealed capsule's durable fields
            # (CID + kind + SEALED lifecycle marker). Retirement
            # annotations do not enter the chain — they are an
            # audit overlay, not part of identity.
            durable = dataclasses.replace(
                e.capsule, lifecycle=CapsuleLifecycle.SEALED)
            expected = _chain_step(prev, durable)
            if expected != e.chain_hash:
                return False
            if e.prev_chain_hash != prev:
                return False
            prev = e.chain_hash
        return True

    def stats(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        by_lifecycle: dict[str, int] = {}
        for e in self._entries:
            by_kind[e.capsule.kind] = (
                by_kind.get(e.capsule.kind, 0) + 1)
            by_lifecycle[e.capsule.lifecycle] = (
                by_lifecycle.get(e.capsule.lifecycle, 0) + 1)
        return {
            "n_entries": len(self._entries),
            "n_admitted": self._n_admitted,
            "n_sealed": self._n_sealed,
            "n_retired": self._n_retired,
            "n_rejected": self._n_rejected,
            "by_kind": by_kind,
            "by_lifecycle": by_lifecycle,
            "chain_head": self._chain_head,
            "chain_ok": self.verify_chain(),
        }


# =============================================================================
# CapsuleView — serialisable DAG slice
# =============================================================================


CAPSULE_VIEW_SCHEMA = "wevra.capsule_view.v1"


@dataclasses.dataclass(frozen=True)
class CapsuleView:
    """A serialisable slice of a ``CapsuleLedger``.

    The shape a Wevra product report embeds under
    ``report["capsules"]``:

        {
          "schema":        "wevra.capsule_view.v1",
          "chain_head":    "<64-hex>",
          "chain_ok":      true,
          "stats":         { ... },
          "capsules":      [ <header-dict>, ... ],
          "root_cid":      "<cid of the RUN_REPORT capsule>",
        }

    Default ``include_payload=False`` so the view is compact in the
    on-disk artifact. Callers can opt into the full payload-carrying
    view for debugging.
    """

    schema: str
    chain_head: str
    chain_ok: bool
    stats: dict[str, Any]
    capsules: list[dict[str, Any]]
    root_cid: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "chain_head": self.chain_head,
            "chain_ok": self.chain_ok,
            "stats": self.stats,
            "root_cid": self.root_cid,
            "capsules": self.capsules,
        }


def render_view(ledger: CapsuleLedger,
                 *, include_payload: bool = False,
                 root_cid: str | None = None,
                 ) -> CapsuleView:
    """Render a ``CapsuleView`` from a ledger.

    ``include_payload=False`` is the default because payloads for
    ``ARTIFACT`` / ``RUN_REPORT`` / ``SWEEP_CELL`` can dominate the
    view's footprint; they are still recoverable from the ledger
    object if the SDK is used in-process. When the view is the only
    thing serialised to disk, consumers fetch payloads by CID from
    whichever other artifact (``sweep_result.json``,
    ``provenance.json``) still carries the full shape.
    """
    caps = []
    for cap in ledger.all_capsules():
        if include_payload:
            caps.append(cap.as_dict())
        else:
            caps.append(cap.as_header_dict())
    return CapsuleView(
        schema=CAPSULE_VIEW_SCHEMA,
        chain_head=ledger.chain_head(),
        chain_ok=ledger.verify_chain(),
        stats=ledger.stats(),
        capsules=caps,
        root_cid=root_cid,
    )


# =============================================================================
# Adapters — turn existing substrate primitives into capsules
# =============================================================================


def capsule_from_handle(handle: Any,
                         *,
                         parents: Iterable[str] = (),
                         ) -> ContextCapsule:
    """Build a ``HANDLE`` capsule from a
    ``vision_mvp.core.context_ledger.Handle``.

    Payload carries the handle's CID, span, fingerprint, and
    metadata — enough for a downstream consumer to reproduce the
    fetch without the ledger object being in scope. The capsule's
    own CID is NOT the handle's CID (it is a fresh content hash
    over the capsule shape), but the handle's CID is recorded in
    ``metadata["handle_cid"]`` for lookup.
    """
    md: dict[str, Any]
    payload: dict[str, Any]
    try:
        payload = {
            "handle_cid": handle.cid,
            "span": list(handle.span) if handle.span else None,
            "fingerprint": handle.fingerprint,
        }
        md = {"handle_cid": handle.cid}
        md.update(handle.metadata_dict())
    except AttributeError as ex:
        raise TypeError(
            f"capsule_from_handle() expects a context_ledger.Handle, "
            f"got {type(handle).__name__}: {ex}")
    return ContextCapsule.new(
        kind=CapsuleKind.HANDLE,
        payload=payload,
        parents=parents,
        metadata=md,
        n_tokens=len((handle.fingerprint or "").split()) or 1,
    )


def capsule_from_handoff(handoff: Any,
                          *,
                          parents: Iterable[str] = (),
                          ) -> ContextCapsule:
    """Build a ``HANDOFF`` capsule from a
    ``vision_mvp.core.role_handoff.TypedHandoff``.

    The capsule carries the handoff's full ``as_dict`` so downstream
    reasoning about the role-cast + claim-kind + payload is
    reconstructible from the capsule alone. The Phase-31 hash-chain
    and the Capsule-ledger hash-chain remain independent (each
    substrate signs its own history) but the capsule's ``metadata``
    records the handoff's ``chain_hash`` so the two can be cross-
    audited.
    """
    try:
        d = handoff.as_dict()
    except AttributeError as ex:
        raise TypeError(
            f"capsule_from_handoff() expects a TypedHandoff-like "
            f"object, got {type(handoff).__name__}: {ex}")
    payload = d
    md = {
        "source_role": d.get("source_role"),
        "to_role": d.get("to_role"),
        "claim_kind": d.get("claim_kind"),
        "handoff_id": d.get("handoff_id"),
        "handoff_chain_hash": d.get("chain_hash"),
    }
    return ContextCapsule.new(
        kind=CapsuleKind.HANDOFF,
        payload=payload,
        parents=parents,
        metadata=md,
        n_tokens=getattr(handoff, "n_tokens", None)
            or len(str(d.get("payload", "")).split()) or 1,
    )


def capsule_from_provenance(manifest: dict[str, Any],
                             *,
                             parents: Iterable[str] = (),
                             ) -> ContextCapsule:
    """Build a ``PROVENANCE`` capsule from a
    ``wevra.provenance.v1`` manifest dict."""
    return ContextCapsule.new(
        kind=CapsuleKind.PROVENANCE,
        payload=manifest,
        parents=parents,
        metadata={"schema": manifest.get("schema", "?")},
    )


def capsule_from_sweep_cell(cell: dict[str, Any],
                             *,
                             spec_cid: str | None = None,
                             ) -> ContextCapsule:
    """Build a ``SWEEP_CELL`` capsule from a ``wevra.sweep.v2``
    cell dict (parser_mode / apply_mode / n_distractors /
    pooled / n_instances)."""
    parents = (spec_cid,) if spec_cid else ()
    return ContextCapsule.new(
        kind=CapsuleKind.SWEEP_CELL,
        payload=cell,
        parents=parents,
        metadata={
            "parser_mode": cell.get("parser_mode"),
            "apply_mode": cell.get("apply_mode"),
            "n_distractors": cell.get("n_distractors"),
        },
    )


def capsule_from_sweep_spec(spec: Any,
                             *,
                             parents: Iterable[str] = (),
                             ) -> ContextCapsule:
    """Build a ``SWEEP_SPEC`` capsule from a ``SweepSpec``
    dataclass (or a dict of its fields)."""
    if dataclasses.is_dataclass(spec):
        payload = dataclasses.asdict(spec)
    elif isinstance(spec, dict):
        payload = dict(spec)
    else:
        raise TypeError(
            f"capsule_from_sweep_spec() expects SweepSpec or dict, "
            f"got {type(spec).__name__}")
    return ContextCapsule.new(
        kind=CapsuleKind.SWEEP_SPEC,
        payload=payload,
        parents=parents,
        metadata={"mode": payload.get("mode"),
                   "sandbox": payload.get("sandbox")},
    )


def capsule_from_profile(profile_name: str,
                          profile_dict: dict[str, Any] | None,
                          ) -> ContextCapsule:
    """Build a ``PROFILE`` capsule from a resolved profile. The
    profile is a root node in the capsule DAG — no parents."""
    payload: dict[str, Any] = {"name": profile_name}
    if profile_dict is not None:
        # Drop any non-JSON-canonicalisable scraps defensively.
        for k in ("description", "trust", "readiness", "sweep"):
            if k in profile_dict:
                payload[k] = profile_dict[k]
    return ContextCapsule.new(
        kind=CapsuleKind.PROFILE,
        payload=payload,
        parents=(),
        metadata={"profile": profile_name},
    )


def capsule_from_readiness(readiness: dict[str, Any],
                            *,
                            parents: Iterable[str] = (),
                            ) -> ContextCapsule:
    """Build one aggregated ``READINESS_CHECK`` capsule summarising
    a ``run_readiness`` verdict."""
    return ContextCapsule.new(
        kind=CapsuleKind.READINESS_CHECK,
        payload=readiness,
        parents=parents,
        metadata={
            "ready": bool(readiness.get("ready")),
            "n": readiness.get("n"),
            "n_passed_all": readiness.get("n_passed_all"),
        },
    )


def capsule_from_artifact(path: str,
                           *,
                           sha256: str | None = None,
                           parents: Iterable[str] = (),
                           ) -> ContextCapsule:
    """Build an ``ARTIFACT`` capsule describing an on-disk file.
    Does NOT open the file; the SHA is passed in by the caller so
    the ledger stays a pure in-memory object."""
    payload = {"path": path, "sha256": sha256}
    return ContextCapsule.new(
        kind=CapsuleKind.ARTIFACT,
        payload=payload,
        parents=parents,
        metadata={"path": path, "sha256": sha256 or ""},
    )


def capsule_from_cohort(*, cohort_tag: str,
                         member_cids: Iterable[str],
                         max_members: int,
                         predicate_note: str = "",
                         extra_payload: dict[str, Any] | None = None,
                         ) -> ContextCapsule:
    """Build a ``COHORT`` capsule — a bounded-membership container.

    The cohort's ``parents`` are the member CIDs. The cohort's
    ``max_parents`` budget is the cardinality cap on the members;
    admission fails at construction if more than ``max_members`` CIDs
    are passed in.

    This is the operational form of the Phase-47 cohort-lifting
    result (Theorem W3-15 in ``docs/CAPSULE_FORMALISM.md``):
    table-level bounded-context invariants of the shape
    ``|{c ∈ admitted : φ(c)}| ≤ N`` subsume to one cohort capsule
    with ``max_parents = N`` whose parent set is exactly
    ``{c : φ(c)}``.

    Arguments:
      * ``cohort_tag`` — short identifier for what the cohort
        represents, e.g. ``"adaptive_edge_table"``,
        ``"thread_replies"``. Goes into the payload so downstream
        consumers can discriminate.
      * ``member_cids`` — the CIDs of the capsule members (which
        must all already be admitted into the ledger that will
        admit this cohort, because every parent CID must be
        present).
      * ``max_members`` — the cardinality cap. Mapped to
        ``CapsuleBudget.max_parents`` so the existing admission
        rule enforces it.
      * ``predicate_note`` — optional human-readable note about
        the membership predicate (for audit / replay). Not
        machine-interpreted.
      * ``extra_payload`` — optional extra fields for the cohort's
        payload. For example, ``{"tick": 5}`` records the round
        at which the snapshot was taken.

    Raises ``ValueError`` if ``len(member_cids) > max_members``
    (the cohort-cardinality bound is enforced at construction,
    before ledger admission, via the ``max_parents`` axis).
    """
    members = tuple(member_cids)
    payload: dict[str, Any] = {
        "cohort_tag": cohort_tag,
        "n_members": len(members),
        "max_members": max_members,
        "predicate_note": predicate_note,
    }
    if extra_payload:
        for k, v in extra_payload.items():
            if k in payload:
                continue  # protect the canonical fields
            payload[k] = v
    return ContextCapsule.new(
        kind=CapsuleKind.COHORT,
        payload=payload,
        budget=CapsuleBudget(max_parents=int(max_members),
                              max_bytes=1 << 17),
        parents=members,
        metadata={"cohort_tag": cohort_tag,
                   "n_members": len(members),
                   "max_members": int(max_members)},
    )


def capsule_from_adaptive_sub_table(table: Any,
                                     *,
                                     tick: int | None = None,
                                     edge_cids: Iterable[str] | None = None,
                                     ) -> ContextCapsule:
    """Build a ``COHORT`` capsule witnessing the Phase-36
    ``AdaptiveSubscriptionTable`` invariant
    ``|active_edges| ≤ max_active_edges`` at a single tick.

    This is the adapter that closes the Phase-46 unification-audit
    PARTIAL verdict on AdaptiveEdge (``verdict == PARTIAL`` → now
    FULL via cohort lift). Phase 46's audit_adaptive_edge only
    lifted *one* edge; the table-level cap was not subsumable at
    that time. After Phase 47's COHORT addition, the table-level
    invariant IS subsumable: every active edge is lifted to an
    ADAPTIVE_EDGE capsule and admitted into a ledger; the table
    itself is then a COHORT capsule whose parent set is those
    edge CIDs and whose ``max_parents`` is
    ``table.max_active_edges``. Admission of the cohort fails
    iff the table-level cardinality bound is violated.

    The caller is responsible for having already admitted the
    edge capsules; if the individual edge CIDs are known, pass
    them as ``edge_cids``, otherwise this function re-derives
    them from ``table.active_edges()`` by constructing fresh
    ADAPTIVE_EDGE capsules (and the caller must admit those into
    their ledger).
    """
    edges = list(table.active_edges())
    resolved_cids: list[str]
    if edge_cids is not None:
        resolved_cids = [str(c) for c in edge_cids]
    else:
        resolved_cids = []
        for e in edges:
            cap = ContextCapsule.new(
                kind=CapsuleKind.ADAPTIVE_EDGE,
                payload=e.as_dict(),
                budget=CapsuleBudget(
                    max_rounds=int(getattr(e, "ttl_rounds", 1) or 1),
                    max_parents=8,
                ),
                metadata={
                    "edge_id": e.edge_id,
                    "source_role": e.source_role,
                    "claim_kind": e.claim_kind,
                },
            )
            resolved_cids.append(cap.cid)
    max_active = int(getattr(table, "max_active_edges", len(edges)))
    extra: dict[str, Any] = {}
    if tick is not None:
        extra["tick"] = int(tick)
    extra["n_active_edges"] = len(edges)
    return capsule_from_cohort(
        cohort_tag="adaptive_edge_table",
        member_cids=resolved_cids,
        max_members=max_active,
        predicate_note=("member ≡ AdaptiveEdge active at this tick;"
                         " enforced by max_parents == max_active_edges"),
        extra_payload=extra,
    )


def capsule_from_report(report_headers: dict[str, Any],
                         *,
                         parents: Iterable[str],
                         ) -> ContextCapsule:
    """Build a ``RUN_REPORT`` capsule — the root of the run's
    capsule graph. Parents are every other sealed capsule in the
    ledger (profile, provenance, readiness, sweep spec + cells,
    artifacts). The ``payload`` is the *headers* of the product
    report (profile, readiness-ready bool, sandbox, wall_seconds),
    not the full dict — to avoid circular payloads. Consumers who
    want the full report read ``product_report.json`` directly and
    verify its CID matches."""
    return ContextCapsule.new(
        kind=CapsuleKind.RUN_REPORT,
        payload=report_headers,
        parents=parents,
        metadata={
            "profile": report_headers.get("profile"),
            "ready": report_headers.get("ready"),
            "wall_seconds": report_headers.get("wall_seconds"),
        },
    )


# =============================================================================
# build_report_ledger — the runner's end-of-run entry point
# =============================================================================


def build_report_ledger(product_report: dict[str, Any],
                         *,
                         profile_dict: dict[str, Any] | None = None,
                         ) -> tuple[CapsuleLedger, str]:
    """Fold a finished ``product_report.json`` dict into a
    ``CapsuleLedger`` + the root ``RUN_REPORT`` CID.

    This is the convenience the product runner calls to materialise
    the capsule graph for an already-completed run. Every
    boundary-crossing artefact the runner knows about becomes a
    sealed capsule:

      * profile           → PROFILE       (root of the DAG)
      * readiness         → READINESS_CHECK (parent: profile)
      * sweep.spec-shape  → SWEEP_SPEC    (parent: profile)
      * sweep.cells[i]    → SWEEP_CELL    (parent: sweep_spec)
      * provenance        → PROVENANCE    (parent: profile)
      * artifacts         → ARTIFACT      (parent: profile)
      * run_report        → RUN_REPORT    (parents: all of the above)

    The function is intentionally conservative: if a section is
    missing or malformed, the run-report capsule simply has fewer
    parents. No section is load-bearing for the rest of the DAG.

    Returns ``(ledger, run_report_cid)``. The caller embeds
    ``render_view(ledger, root_cid=run_report_cid)`` in the report
    under ``report["capsules"]``.
    """
    ledger = CapsuleLedger()
    profile_name = product_report.get("profile", "unknown")
    prof_cap = capsule_from_profile(profile_name, profile_dict)
    prof_cap = ledger.admit_and_seal(prof_cap)

    parent_cids: list[str] = [prof_cap.cid]

    rd = product_report.get("readiness")
    if isinstance(rd, dict):
        rd_cap = capsule_from_readiness(rd, parents=(prof_cap.cid,))
        rd_cap = ledger.admit_and_seal(rd_cap)
        parent_cids.append(rd_cap.cid)

    sweep = product_report.get("sweep")
    sweep_spec_cid: str | None = None
    if isinstance(sweep, dict) and sweep.get("schema") == "wevra.sweep.v2":
        spec_payload = {
            "mode": sweep.get("mode"),
            "sandbox": sweep.get("sandbox"),
            "jsonl": sweep.get("jsonl"),
            "model": sweep.get("model"),
            "endpoint": sweep.get("endpoint"),
            "executed_in_process": sweep.get("executed_in_process"),
        }
        spec_cap = capsule_from_sweep_spec(
            spec_payload, parents=(prof_cap.cid,))
        spec_cap = ledger.admit_and_seal(spec_cap)
        sweep_spec_cid = spec_cap.cid
        parent_cids.append(spec_cap.cid)
        for cell in (sweep.get("cells") or []):
            if not isinstance(cell, dict):
                continue
            cell_cap = capsule_from_sweep_cell(
                cell, spec_cid=spec_cap.cid)
            cell_cap = ledger.admit_and_seal(cell_cap)
            parent_cids.append(cell_cap.cid)

    manifest = product_report.get("provenance")
    if isinstance(manifest, dict):
        prov_cap = capsule_from_provenance(
            manifest, parents=(prof_cap.cid,))
        prov_cap = ledger.admit_and_seal(prov_cap)
        parent_cids.append(prov_cap.cid)

    for art in (product_report.get("artifacts") or []):
        if not isinstance(art, str):
            continue
        art_cap = capsule_from_artifact(art, parents=(prof_cap.cid,))
        art_cap = ledger.admit_and_seal(art_cap)
        parent_cids.append(art_cap.cid)

    # Cap parent count at the RUN_REPORT's budget. The default is
    # 1024 (see ``_default_budget_for``), which covers a very large
    # sweep (~1000 cells); cap anyway so a runaway run does not
    # synthesise an illegal capsule.
    budget = _default_budget_for(CapsuleKind.RUN_REPORT)
    if (budget.max_parents is not None
            and len(parent_cids) > budget.max_parents):
        parent_cids = parent_cids[: budget.max_parents]

    headers = {
        "profile": profile_name,
        "schema": product_report.get("schema"),
        "wall_seconds": product_report.get("wall_seconds"),
        "ready": bool((product_report.get("readiness") or {})
                       .get("ready")),
        "executed_in_process": bool(
            (product_report.get("sweep") or {})
            .get("executed_in_process")),
    }
    run_cap = capsule_from_report(headers, parents=tuple(parent_cids))
    run_cap = ledger.admit_and_seal(run_cap)
    return ledger, run_cap.cid


__all__ = [
    # Primitives
    "CapsuleKind", "CapsuleLifecycle", "CapsuleBudget",
    "ContextCapsule",
    # Ledger
    "CapsuleLedger", "CapsuleAdmissionError", "CapsuleLifecycleError",
    # View
    "CapsuleView", "CAPSULE_VIEW_SCHEMA", "render_view",
    # Adapters
    "capsule_from_handle", "capsule_from_handoff",
    "capsule_from_provenance", "capsule_from_sweep_cell",
    "capsule_from_sweep_spec", "capsule_from_profile",
    "capsule_from_readiness", "capsule_from_artifact",
    "capsule_from_report", "build_report_ledger",
    # Phase-47 cohort subsumption
    "capsule_from_cohort", "capsule_from_adaptive_sub_table",
]
