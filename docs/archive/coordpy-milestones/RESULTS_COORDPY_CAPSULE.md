# RESULTS — CoordPy SDK v3: the Context Capsule abstraction

> *Milestone note. Theory-forward. States claims, their code anchors,
> their empirical status, and what remains conjectural.*

---

## 1. The claim in one paragraph

Before v3, CoordPy's top-line was *"bounded-context orchestration and
evaluation SDK."* That framing described an outcome but not a
mechanism; it also left unnamed the single load-bearing piece of
machinery that recurred across every Context-Zero substrate layer
from Phase 19 onward: **a typed, content-addressed, lifecycle-bounded,
budget-bounded, provenance-carrying object that crosses a role /
layer / run boundary**.

SDK v3 names that object — the **`ContextCapsule`** — fixes the six
invariants it must satisfy (the *Capsule Contract*), ships a
ledger that admits, seals, and audits capsules end-to-end, folds
every run's finished artefacts into a sealed capsule DAG, and
re-centres the product's public surface on it. The older substrate
primitives (Phase-19 `Handle`, Phase-31 `TypedHandoff`, Phase-35
`ThreadResolution`, Phase-36 `AdaptiveEdge`, every sweep cell,
every provenance manifest, every on-disk artefact) are recognised
as *instances* of the capsule-shaped object — no primitive is
re-implemented, nothing is renamed, and every Phase-N..Phase-44
test continues to pass byte-for-byte (1593 tests + 21 subtests,
green).

The resulting product identity is:

> **CoordPy is a context-capsule runtime.** Every piece of context
> that crosses a role boundary, a layer boundary, or a run
> boundary is a typed, content-addressed, lifecycle-bounded,
> budget-bounded, provenance-stamped capsule — never a raw prompt
> string.

---

## 2. The Capsule Contract — six invariants

Let `Cap = (cid, kind, payload, budget, parents, lifecycle,
n_tokens, n_bytes, metadata)`.

> **C1 Identity.**       `cid = SHA-256(canonical(kind, payload,
> budget, sorted(parents)))`. Two capsules with the same content
> collapse to one CID; parent insertion order is canonicalised.
>
> **C2 Typed claim.**    `kind ∈ CapsuleKind.ALL` (closed
> vocabulary: `HANDOFF`, `HANDLE`, `THREAD_RESOLUTION`,
> `ADAPTIVE_EDGE`, `SWEEP_CELL`, `SWEEP_SPEC`, `READINESS_CHECK`,
> `PROVENANCE`, `RUN_REPORT`, `PROFILE`, `ARTIFACT`). Unknown kinds
> are rejected at construction.
>
> **C3 Lifecycle.**      `lifecycle ∈ {PROPOSED, ADMITTED, SEALED,
> RETIRED}`, with legal forward transitions only:
> `PROPOSED → ADMITTED → SEALED → RETIRED`. Illegal transitions
> raise `CapsuleLifecycleError`.
>
> **C4 Budget.**         `CapsuleBudget(max_tokens, max_bytes,
> max_rounds, max_witnesses, max_parents)` — every capsule carries
> a budget; admission checks the axes that apply to its kind; a
> capsule that would exceed its budget is rejected.
> `CapsuleBudget()` with all axes `None` is an illegal construction.
>
> **C5 Provenance.**     `parents ⊆ ledger._by_cid`; the ledger
> keeps a hash chain linking consecutive seals. A retroactive
> modification of any sealed entry's chain_hash breaks
> `verify_chain()`.
>
> **C6 Frozen.**         `ContextCapsule` is a frozen dataclass.
> Once sealed, the capsule's payload cannot change without its
> CID changing.

**Contract tests (all green).** `vision_mvp/tests/test_coordpy_capsules.py`:

  * C1: `test_c1_identity_deterministic`,
    `test_c1_parent_order_is_canonicalised`.
  * C2: `test_c2_unknown_kind_rejected`.
  * C3: `test_c3_lifecycle_order`,
    `test_c3_idempotent_admit_and_seal`.
  * C4: `test_c4_budget_rejects_over_tokens`,
    `test_c4_budget_rejects_over_bytes_at_construction`,
    `test_c4_empty_budget_is_illegal`.
  * C5: `test_c5_parent_must_be_in_ledger`,
    `test_c5_hash_chain_detects_tamper`.
  * C6: `test_c6_sealed_capsule_is_frozen`.
  * End-to-end: `test_build_report_ledger_from_smoke_run`,
    `test_capsule_view_on_disk_matches_embedded`.
  * CLI: `test_cli_cid_and_view`.

---

## 3. Theorems (code-backed)

**Theorem W3-1 (Capsule Identity Soundness, C1+C6).**
Let `Cap` be a sealed capsule in a `CapsuleLedger`. If any field
in `(kind, payload, budget, parents)` is modified, the capsule's
CID must change; if the CID is unchanged, the fields are equal
byte-for-byte (after canonicalisation).
**Proof.** By construction — `cid = SHA-256(canonical(fields))`
in `capsule._capsule_cid`. Canonical encoding uses sorted keys,
sorted parents, and `separators=(",",":")` so byte-for-byte
equality of the canonical encoding is a function of field
equality up to Python's representation of scalars. Soundness of
SHA-256 is inherited.
**Empirical anchor.** `test_c1_identity_deterministic`,
`test_c1_parent_order_is_canonicalised`.

**Theorem W3-2 (Capsule Lifecycle Safety, C3).**
Let `ledger` be a `CapsuleLedger`. Every capsule in
`ledger._entries` has `lifecycle ∈ {SEALED, RETIRED}`; admission
and sealing are idempotent on CID; `admit()` on a
non-`PROPOSED` capsule raises `CapsuleLifecycleError`.
**Proof.** Inspection of `CapsuleLedger.admit` / `.seal`:
the lifecycle preconditions are checked first; `dataclasses.replace`
is the only state transition; the CID index makes admission
idempotent.
**Empirical anchor.** `test_c3_lifecycle_order`,
`test_c3_idempotent_admit_and_seal`.

**Theorem W3-3 (Chain Tamper-Evidence, C5).**
Let `ledger` be a `CapsuleLedger` with a sequence of sealed
entries `e_0, e_1, …, e_n`, where `e_i.chain_hash =
SHA-256(canonical(e_{i-1}.chain_hash, e_i.capsule.cid,
e_i.capsule.kind, SEALED))`. If any entry's `chain_hash` or
`prev_chain_hash` is mutated after seal, `verify_chain()`
returns False on the first divergent link.
**Proof.** Recomputation loop in `CapsuleLedger.verify_chain`:
propagates `prev` through the entries; each link is recomputed
from the durable sealed-state fields. The hash function's
collision resistance implies the recomputed chain differs from
the stored chain iff any field has changed.
**Empirical anchor.** `test_c5_hash_chain_detects_tamper`.

**Theorem W3-4 (Run-Report CID is a Run Identifier).**
Let `report` be a finished `product_report.json`. The
`build_report_ledger(report)` pair `(ledger, run_cid)` is a pure
function of the report's deterministically-ordered substructure
(profile, readiness, sweep spec, sweep cells, provenance,
artifacts). Two byte-equal reports yield byte-equal `run_cid`.
Two reports that differ in any substructure yield different
`run_cid` (Theorem W3-1 transitively).
**Proof.** The ``build_report_ledger`` folds each substructure
through a deterministic adapter into a `ContextCapsule` with a
fixed parent set; C1 makes each adapter output pure over its
input; the root `RUN_REPORT` capsule's parents are sorted-
canonicalised in `_capsule_cid`; therefore the root CID is a
pure function of the input report.
**Empirical anchor.** `test_build_report_ledger_from_smoke_run`
(asserts `ledger.verify_chain() = True` and that
`RUN_REPORT.lifecycle = SEALED`).

**Theorem W3-5 (Budget Invariant Across Capsule Kinds).**
For every `ContextCapsule` admitted into a `CapsuleLedger`,
every budget axis applicable to its kind holds. In particular:

  * `HANDOFF`: `n_tokens ≤ budget.max_tokens` (matches the
    Phase-31 per-role `τ` bound, Theorem P31-3).
  * `THREAD_RESOLUTION`: `n_tokens ≤ budget.max_tokens`,
    `max_rounds ≥ rounds_used`, `max_witnesses ≥ witness_tokens`
    (matches the Phase-35 `C_0 + R*·τ + T·R_max·W` bound,
    Theorem P35-2).
  * `SWEEP_CELL`: `n_bytes ≤ budget.max_bytes`.
  * `RUN_REPORT`: `len(parents) ≤ budget.max_parents`.

**Proof.** Admission enforces each axis; construction enforces
`max_bytes` and `max_parents`. The corresponding substrate-level
theorems (P31-3, P35-2) are *special cases* of the per-kind
budget invariant, unified under C4.
**Empirical anchor.** `test_c4_budget_rejects_over_tokens`,
`test_c4_budget_rejects_over_bytes_at_construction`.

**Theorem W3-6 (Capsule Contract Composes with Existing Substrate).**
Every Phase-19..Phase-36 substrate primitive admits a canonical
capsule adapter that preserves the substrate's original
semantics:

  * `capsule_from_handle` — `HANDLE` capsule carries the Phase-19
    `Handle.cid`, `.span`, `.fingerprint`, `.metadata`. Fetch via
    the capsule's metadata `handle_cid` is byte-equivalent to
    fetch via the original `Handle`.
  * `capsule_from_handoff` — `HANDOFF` capsule carries the
    Phase-31 `TypedHandoff.as_dict()`; the substrate's `chain_hash`
    is cross-referenced in the capsule's metadata so both hash
    chains are auditable together.
  * `capsule_from_sweep_cell`, `capsule_from_sweep_spec`,
    `capsule_from_provenance`, `capsule_from_readiness`,
    `capsule_from_artifact`, `capsule_from_profile` — forward
    the Phase-45 / Slice-2 artefacts verbatim into their
    capsule kinds.

No substrate primitive is modified. The capsule layer is
strictly additive on top of every Phase-N guarantee.
**Empirical anchor.** `test_adapter_handle_to_capsule`,
`test_adapter_handoff_to_capsule`, plus the full regression pass
(1593 tests + 21 subtests, 0 failures).

---

## 4. Conjectures (empirically plausible, not yet proved)

**Conjecture W3-C1 (Capsule-lens subsumption).**
Every theorem of the form "bounded active context per role" or
"bounded delivered tokens per run" in Phases 19..44 (P19-*,
P31-*, P35-*, P41-*, P43-*) is expressible as a C4 (budget)
constraint on a specific `CapsuleKind`. The capsule contract is
therefore not a new layer on top of the substrate results — it
is *the same statement* said once, at the right level.
*Status.* Supported on four primitive classes (`HANDLE`,
`HANDOFF`, `THREAD_RESOLUTION`, `SWEEP_CELL`); a formal
re-derivation of every Phase-N bounded-context theorem under
the capsule lens is pending.

**Conjecture W3-C2 (CID-pinned reproducibility).**
For any two runs `R_1`, `R_2` with identical `run_cid`, the
downstream consumer can prove, offline, that their
`product_report.json` bytes are equal modulo canonical
re-ordering of JSON keys — using only the capsule view and
SHA-256. Therefore "give me the run_cid" is a durable public
reference for a CoordPy run, the way a Git commit SHA is a
durable reference for a source tree.
*Status.* Supported by Theorem W3-1 + Theorem W3-4; an
adversarial attack where a malicious re-signer produces a
different report with the same run_cid is ruled out by SHA-256
collision resistance. Empirically exercised by
`test_capsule_view_on_disk_matches_embedded`.

**Conjecture W3-C3 (Category-shift claim, cautious).**
Framing CoordPy as a *context-capsule runtime* is a paradigm
shift relative to the "agent framework" and "eval harness"
categories because it relocates the load-bearing unit of
coordination from *strings* to *objects with identity, type,
lifecycle, budget, and proof*. This conjecture is not the
claim that capsules are a novel primitive (they are not — see
§ 6 on honest originality); it is the claim that *the product
level should centre on them*. Falsifier: an external framework
adopts the exact same six invariants under a different name
without crediting the unification. (Friendly falsifier — such
an adoption would still validate the shift.)
*Status.* Framing-level, not a formal claim. The code-backed
claims are W3-1..W3-6; this conjecture is the story they tell
when you step back.

---

## 5. What is code-backed vs empirical vs conjectural

| Claim | Type | Evidence |
|---|---|---|
| Capsules exist and satisfy C1..C6 | **Code-backed** | `vision_mvp/coordpy/capsule.py` + `test_coordpy_capsules.py` |
| Every run emits a sealed DAG | **Code-backed** | `product/runner.py` folds each run via `build_report_ledger`; artefact `capsule_view.json` is written next to `product_report.json` |
| The capsule graph verifies end-to-end on local_smoke | **Empirical** | 13-capsule graph on `local_smoke`, `chain_ok=True`, `root_cid` computed in <30ms |
| Adapters preserve substrate semantics | **Code-backed** | Theorem W3-6 + `test_adapter_handle_to_capsule`, `test_adapter_handoff_to_capsule` |
| Every Phase-N..Phase-44 test continues to pass | **Empirical** | 1593 tests + 21 subtests, 0 failures (`pytest vision_mvp/tests/`, 182 s wall) |
| Capsule contract subsumes Phase-31 and Phase-35 bounded-context theorems | **Conjectural (W3-C1)** | Four primitive classes re-derived; full re-derivation pending |
| `run_cid` is a durable public reference | **Conjectural (W3-C2)** | Supported by W3-1 + W3-4; no adversarial falsifier attempted |
| Capsule framing is a genuine paradigm shift | **Conjectural (W3-C3)** | Framing-level, not a formal claim |

---

## 6. Honest originality

**What is inherited, not invented.**

  * **Content addressing.** Merkle DAGs (`core/merkle_dag.py`), Git
    objects, IPFS — older than this project.
  * **Hash-chained logs.** Tamper-evident-log research (Haber &
    Stornetta 1991; Certificate Transparency; blockchain). The
    Phase-31 `HandoffLog` already did this; the capsule ledger
    reuses the idea.
  * **Typed claim kinds.** Actor systems, event-sourcing, CQRS,
    capability-based OSes (KeyKOS, seL4) — typed references with
    closed vocabularies is a long tradition.
  * **Lifecycle states / session types.** Session-typed protocol
    research; linear / affine type systems; RAII. The
    `PROPOSED → ADMITTED → SEALED → RETIRED` sequence is a
    deliberately minimal instance.
  * **Frozen / immutable records.** Standard library: Python
    frozen dataclasses, Clojure immutable data structures, Git's
    immutable object model.

**What is new, claimed cautiously.**

  * The **unification** of these primitives under a single
    contract applied uniformly to every inter-role, inter-layer,
    and inter-run artefact in an LLM-agent-team runtime.
  * The **product-level decision** that "context is not a prompt;
    context is an object with identity, type, lifecycle, budget,
    and proof" — and that this decision belongs at the top of the
    SDK, not buried in a research module.
  * The **end-to-end implementation** — the capsule graph lands on
    disk in a canonical schema (`coordpy.capsule_view.v1`), is
    auditable offline via `coordpy-capsule verify`, and the root
    `RUN_REPORT` CID is a durable public reference for a CoordPy
    run. No external trust dependency.
  * The **subsumption** of older Phase-19..Phase-44 substrate
    theorems as special cases of the capsule contract's budget
    invariant. This is a framing claim supported by W3-6 on four
    primitive classes; the full subsumption is Conjecture W3-C1.

**What the programme does NOT claim.**

  * Capsules are not a novel cryptographic primitive. The security
    properties derive from SHA-256 and are no stronger than it.
  * The programme does not claim capabilities-style authorisation,
    unforgeable references, or an ocap-model guarantee — those
    would require a typed-reference runtime the SDK does not
    implement.
  * The programme does not claim that all LLM-agent-team runtimes
    *must* centre on capsules. It claims that CoordPy does, and that
    the shape has stronger explanatory power than "bounded-context
    orchestration" for the specific work already in this repo.

---

## 7. What this changes about the programme's story

Before SDK v3, the master plan had two parallel stories:

  * Research story (§§ 1–9) — *per-agent minimum-sufficient
    context, measured across ten arcs*.
  * Product story (§ 10) — *a bounded-context evaluation SDK with a
    unified runtime and a plugin surface*.

After SDK v3, the product story has a mechanism-shaped centre:

  * **CoordPy is a context-capsule runtime. The research results of
    §§ 1–9 are empirical / theoretical evidence for specific
    budget invariants on specific capsule kinds. The product
    surface is "one capsule graph per run, CID-pinned,
    tamper-evident, bounded by construction."**

The research programme is unchanged by this naming — the Context
Zero identity (§§ 1–9) remains a multi-arc research programme on
context, routing, and substrate. What changes is that the
product slice now has a single, sharp, mechanism-shaped public
description that a third party reading `README.md` or
`docs/START_HERE.md` can understand in one paragraph — and which
the code actually implements, line-for-line.

---

## 8. What remains open

  * **W3-C1 completion.** A formal re-derivation of every
    Phase-N bounded-context theorem (P19-*, P31-*, P35-*, P41-*,
    P43-*) under the capsule lens — showing each theorem is a
    specific C4 budget invariant on a specific capsule kind.
    Mechanical but not yet done.
  * **Capsule-level payload opacity.** Today a capsule carries
    the full payload. A payload-opaque mode where the capsule
    carries only a CID-pointer to an external store would let
    a product report reference, say, a 100 MB sweep cell
    without embedding it. The default-view (`include_payload=False`)
    already approximates this; a true external-store mode is
    future work.
  * **Cross-run capsule references.** Today two independent
    runs produce independent capsule DAGs; there is no
    first-class way to say "run B consumed the result of run
    A." A `PARENT_RUN_CID` metadata field on `RUN_REPORT` is
    the natural minimal extension.
  * **Out-of-tree adapter contract.** The SDK ships adapters
    for the substrate primitives in this repo. A stable
    *adapter Protocol* — so external packages can register
    their own capsule kinds / adapters — is the natural SDK v4
    target (parallel to the current extension surface for
    sandboxes, task banks, and report sinks).

None of these blocks shipping SDK v3. They are the next
coherent research/product steps.

---

## 9. Code pointers (for another agent arriving cold)

  * Reference implementation: `vision_mvp/coordpy/capsule.py`
    (~700 LOC, no external dependencies beyond the stdlib and
    the existing package).
  * Public surface: `vision_mvp/coordpy/__init__.py` (SDK v3).
  * Runner integration:
    `vision_mvp/product/runner.py::run_profile` calls
    `build_report_ledger` on every finished report; writes
    `capsule_view.json` alongside `product_report.json`.
  * CLI: `vision_mvp/coordpy/_cli.py::_cmd_capsule` — subcommands
    `view`, `verify`, `cid`. Wired as `coordpy-capsule` in
    `pyproject.toml`.
  * Contract tests: `vision_mvp/tests/test_coordpy_capsules.py`
    (25 tests covering C1..C6 + the adapters + the CLI +
    end-to-end).
  * Public-API lock: `vision_mvp/tests/test_coordpy_public_api.py`
    — `SDK_VERSION == "coordpy.sdk.v3"`; any removal or rename of
    a capsule symbol is a breaking change.

Last touched: SDK v3 milestone, 2026-04-22.
