# Start Here

One-pass orientation for this repository. If you read only one doc, read
this one. Everything else in the repo should make sense after this page.

---

## What this repo is — in one line

**Wevra is a context-capsule runtime.** Every piece of context that
crosses a role boundary, a layer boundary, or a run boundary is a
typed, content-addressed, lifecycle-bounded, budget-bounded,
provenance-stamped **capsule** — never a raw prompt string. As of
SDK v3.3 (April 2026), capsules drive execution **one further
structural layer past the cell boundary**: each (task, strategy)
parse→patch→verdict transition is THREE sealed capsules
(PARSE_OUTCOME → PATCH_PROPOSAL → TEST_VERDICT) with parent-CID
gating, the parser-axis transition is a typed lifecycle step
(Theorem W3-39), and a runtime-checkable
``CapsuleLifecycleAudit`` mechanically verifies the lifecycle
correspondence on every finished run (Theorem W3-40).
Deterministic-mode replay (``RunSpec(deterministic=True)``)
collapses the full capsule DAG byte-for-byte across runs of the
same logical input (Theorem W3-41). Meta-artefacts (the report
itself) are authenticated by a *detached* META_MANIFEST in a
secondary ledger — the rendering circularity (impossible to seal
a report whose bytes encode the seal) is a sharp theorem
(W3-36) with a constructive boundary witness. ``wevra-capsule
verify`` recomputes the chain from on-disk header bytes and
re-hashes every artefact. See *"What capsules do at runtime
now"* below. Context Zero is the research programme that
produced it.

## The load-bearing abstraction — Context Capsule

A **`ContextCapsule`** is an immutable object with:

  * **`cid`**        — SHA-256 content address over
                        `(kind, payload, budget, parents)`.
  * **`kind`**       — closed-vocabulary semantic type (`HANDOFF`,
                        `HANDLE`, `THREAD_RESOLUTION`, `SWEEP_CELL`,
                        `PROVENANCE`, `RUN_REPORT`, `PROFILE`,
                        `ARTIFACT`, `READINESS_CHECK`, `SWEEP_SPEC`,
                        `ADAPTIVE_EDGE`).
  * **`lifecycle`**  — `PROPOSED → ADMITTED → SEALED` (+ optional
                        `RETIRED`).
  * **`budget`**     — `CapsuleBudget(max_tokens, max_bytes,
                        max_rounds, max_witnesses, max_parents)`.
  * **`parents`**    — tuple of parent CIDs (the capsule DAG).

Capsules live in a **`CapsuleLedger`** — append-only, hash-chained,
budget-enforcing, provenance-auditing. Every Wevra run emits a sealed
capsule DAG rooted at a `RUN_REPORT` capsule; the root CID is the
durable identifier for the run.

This abstraction *subsumes and re-centers* everything Wevra already
did. Handles (Phase 19), typed handoffs (Phase 31), thread resolutions
(Phase 35), sweep cells, and the provenance manifest were already
capsules — they just weren't named. SDK v3 names them.

See [`RESULTS_WEVRA_CAPSULE.md`](RESULTS_WEVRA_CAPSULE.md) for the
theorem-style statement of the Capsule Contract (invariants C1..C6)
and why it is a better top-level description of the product than
"bounded-context orchestration." See
[`RESULTS_WEVRA_CAPSULE_NATIVE.md`](RESULTS_WEVRA_CAPSULE_NATIVE.md)
for the SDK v3.1 milestone in which capsules become the runtime's
typed execution contract (W3-32 / W3-33 / W3-34 / W3-35).

---

## What capsules do at runtime now (SDK v3.2)

As of April 2026, when you run Wevra with the default
``RunSpec(capsule_native=True)``, capsules **drive** the run, not
just describe it. Specifically:

  * **Profile is sealed first.** Every other capsule's parent CID
    chain ends at the PROFILE capsule. You cannot seal a
    READINESS_CHECK or SWEEP_SPEC without a sealed PROFILE.
  * **Each runtime stage seals its capsule before the next stage
    can read its result.** Readiness verdict → READINESS_CHECK
    capsule. Sweep spec → SWEEP_SPEC capsule. Each cell's results
    → SWEEP_CELL capsule (sealed *as soon as the cell completes*,
    not after the whole sweep). Provenance manifest → PROVENANCE
    capsule. The RUN_REPORT capsule's parents are every other
    sealed capsule.
  * **Substantive artifacts are content-addressed at write time.**
    ``readiness_verdict.json`` / ``sweep_result.json`` /
    ``provenance.json`` are written via
    ``ctx.seal_and_write_artifact``: SHA-256 is computed in
    memory, baked into a sealed ARTIFACT capsule, then bytes hit
    disk, then re-read and re-hashed to verify. The on-disk
    file's hash matches the sealed CID by construction
    (Theorem W3-33).
  * **Mid-run failure is a typed witness, not a state-bag.** A
    stage that fails (admission rejection, missing parent CID,
    over-budget capsule) leaves a typed entry in the in-flight
    register that never reaches the ledger. The runtime exposes
    ``ctx.in_flight_failures()`` listing every kind / cid /
    failure for forensic inspection.
  * **Capsule view tags itself with construction mode.** The
    ``report["capsules"]`` block now carries
    ``"construction": "in_flight"`` so a downstream consumer can
    tell whether the ledger was built during the run (capsule-
    native) or folded after the fact (legacy post-hoc).

## What changed in SDK v3.2 (intra-cell + detached witness)

Two new structural moves were made:

  * **Intra-cell capsules** (``PATCH_PROPOSAL`` and
    ``TEST_VERDICT``). Each (task, strategy) inside a sweep
    cell now seals two capsules in flight: a PATCH_PROPOSAL
    when the generator returns a patch (parent: SWEEP_SPEC,
    payload: coordinates + content hash + bounded rationale),
    and a TEST_VERDICT when the sandbox returns a result
    (parent: PATCH_PROPOSAL, payload: WorkspaceResult fields).
    The ``patch → verdict`` ordering is enforced at the
    capsule layer (W3-32-extended). On the bundled
    ``local_smoke`` profile this seals 48 PATCH_PROPOSAL +
    48 TEST_VERDICT capsules per run.
  * **Detached META_MANIFEST witness for meta-artefacts.**
    The runtime now writes a fourth file, ``meta_manifest.json``,
    whose payload carries the on-disk SHA-256s of
    ``product_report.json``, ``capsule_view.json``, and
    ``product_summary.txt`` plus the primary ``root_cid``. The
    manifest sits in a *secondary* ledger disjoint from the
    primary — the rendering circularity (Theorem W3-36) makes
    it impossible to seal an ARTIFACT for a report whose bytes
    encode the seal. The manifest is the one-hop trust unit
    beyond the primary view; ``wevra-capsule verify`` now
    re-hashes every meta-artefact and primary artefact at
    audit time (W3-37 / W3-38).

## What changed in SDK v3.3 (deeper intra-cell + audit + determinism)

Three additive moves:

  * **Sub-intra-cell ``PARSE_OUTCOME`` capsule.** Each
    (task, strategy) now seals THREE capsules: a PARSE_OUTCOME
    *before* the PATCH_PROPOSAL (parent: SWEEP_SPEC, payload:
    coordinates + parser ``ok`` boolean + closed-vocabulary
    ``failure_kind`` + ``recovery`` label + bounded ``detail``).
    The PATCH_PROPOSAL's parents now include both SWEEP_SPEC and
    the PARSE_OUTCOME's CID, so the parse → patch → verdict
    chain is a typed witness on the DAG (Theorem W3-39). On
    ``local_smoke`` this seals 48 PARSE_OUTCOME capsules per
    run (all ``failure_kind == "oracle"`` because the bundled
    profile uses the deterministic_oracle path); under real-LLM
    profiles, ``failure_kind`` distributes over the parser's
    closed vocabulary (``ok``, ``unclosed_new``, ``prose_only``,
    ``parse_failed``, ``recovery=closed_at_eos`` etc.) so the
    parser-axis attribution is now lifecycle-governed.
  * **Runtime-checkable lifecycle audit** (``audit_capsule_lifecycle``,
    ``audit_capsule_lifecycle_from_view``). Eight invariants
    L-1..L-8 are checked mechanically on a finished run; the
    audit returns OK/BAD/EMPTY plus typed counterexamples
    (Theorem W3-40). The audit is also runnable from a
    forensic ``capsule_view.json`` alone — auditors do not need
    the runtime ctx that produced it.
  * **Deterministic-mode replay** (``RunSpec(deterministic=True)``).
    Strips per-run timestamps / wall-clock fields / host-local
    paths from PROVENANCE / RUN_REPORT / READINESS_CHECK
    payloads and ARTIFACT capsule paths so two runs of the same
    deterministic profile (mock mode, frozen JSONL,
    ``in_process``/``subprocess`` sandbox) produce byte-identical
    full-DAG CIDs and identical chain head (Theorem W3-41).

## What remains post-hoc / audit only

The capsule layer is *substantially* load-bearing in execution
now, but a few axes are still post-hoc / not capsule-tracked:

  * **Sub-step bytes still outside the slice.** The LLM prompt
    that the patch generator sends and the raw LLM response
    bytes are still plain Python. SDK v3.3 captures the
    *parse outcome* (the parser's structured verdict), not the
    LLM bytes that triggered it. The next sub-intra-cell slice
    would name them as ``PROMPT`` and ``LLM_RESPONSE`` capsules
    (Conjecture W3-C5).
  * **Sandbox stdout / stderr / test trace** remain plain bytes.
  * **The post-hoc ``build_report_ledger`` adapter** is retained
    as the third-party-facing path for code that has a
    ``product_report`` dict from somewhere outside the runtime
    (disk, an HTTP API, another tool). The two paths produce
    CID-equivalent ledgers on the spine kinds (Theorem W3-34
    preserved under SDK v3.2's intra-cell extension and SDK
    v3.3's sub-intra-cell extension); they differ on ARTIFACT
    (real SHA vs None) and transitively on RUN_REPORT.
  * **Re-execution determinism is opt-in only.** Without
    ``deterministic=True`` two runs produce different
    PROVENANCE / RUN_REPORT CIDs (timestamp / wall-clock variance).
    With the flag, the full DAG is reproducible (W3-41); the
    underlying profile must be deterministic (mock mode,
    frozen JSONL).
  * **META_MANIFEST authentication** is a one-hop trust unit.
    Theorem W3-36 establishes that authenticating the manifest
    *itself* within the primary ledger is impossible without
    structural circularity. Cryptographic signing of the
    manifest is orthogonal and out of scope.

## Quick check: which path is my run on?

```python
from vision_mvp.wevra import RunSpec, run, CONSTRUCTION_IN_FLIGHT
report = run(RunSpec(profile="local_smoke", out_dir="/tmp/x"))
assert report["capsules"]["construction"] == CONSTRUCTION_IN_FLIGHT
# in_flight_stats: every proposed capsule sealed.
assert report["capsules"]["in_flight_stats"]["n_failed"] == 0
```

---

## What this repo is

This repository is the home of two coupled things:

1. **Context Zero** — a research programme on *per-agent minimum-sufficient
   context* in multi-agent LLM systems. Theorems, phase shards, an
   EXTENDED_MATH survey, and ~1500 tests of substrate behaviour.
2. **Wevra** — the first shipped product from that programme. A
   **context-capsule runtime**: one `RunSpec` in, one reproducible,
   provenance-stamped, sealed-capsule-DAG report out.

Neither identity subsumes the other. Context Zero is the body of work;
Wevra is the load-bearing product slice of it that a third party can
install, run, and rely on. The capsule abstraction is Wevra's centre
of gravity; the substrate primitives (CASR router, typed handoffs,
escalation threads, adaptive subscriptions) are Wevra's *instances*
of capsule-shaped objects, not its identity.

If you came here because you want to **use** something, you want Wevra.
If you came here because you want to **extend the theory**, you want
Context Zero (`PROOFS.md`, `EXTENDED_MATH_*.md`, the RESULTS phase
notes, and the master plan in `docs/context_zero_master_plan.md`).

---

## One-sentence summary per layer

| Layer | One sentence | Stability |
|---|---|---|
| **Context Capsule primitives** (`wevra.capsule`) | `ContextCapsule` + `CapsuleLedger` + `CapsuleView`: the load-bearing SDK abstraction. Every cross-boundary artefact is a typed, content-addressed, lifecycle-bounded, budget-bounded, provenance-carrying capsule. | **Stable v1** (contract C1..C6). |
| **Capsule-native runtime** (`wevra.capsule_runtime`) | `CapsuleNativeRunContext` + `seal_and_write_artifact` + intra-cell `seal_patch_proposal` / `seal_test_verdict` + sub-intra-cell `seal_parse_outcome` (SDK v3.3) + detached `seal_meta_manifest`: capsules drive runtime stage transitions at run boundaries AND inside the inner sweep loop AND on the parser axis; substantive artefacts are content-addressed at write time and re-verifiable at audit time. The capsule layer is the runtime's typed execution contract on the spine and now extends two structural layers past the cell boundary. | **Stable v3** (theorems W3-32 / W3-33 / W3-34 / W3-35 / W3-32-extended / W3-36 / W3-37 / W3-38 / W3-39 / W3-40 / W3-41). |
| **Lifecycle audit** (`wevra.lifecycle_audit`) | `CapsuleLifecycleAudit` + `audit_capsule_lifecycle_from_view`: mechanically verifies eight lifecycle invariants L-1..L-8 over a finished run (Theorem W3-40). Returns OK / BAD / EMPTY plus typed counterexamples. Runnable from runtime ctx OR from on-disk `capsule_view.json` alone. | **Stable v1** (SDK v3.3). |
| **Wevra SDK** (`vision_mvp.wevra`) | Profile-driven context-capsule runtime for SWE-bench-Lite-shape banks; `RunSpec` → provenance-stamped report whose root is a sealed `RUN_REPORT` capsule + a detached `meta_manifest.json` witness. ``RunSpec.deterministic=True`` opt-in collapses CIDs across runs (Theorem W3-41). | **Stable v3.3** — public contract. |
| **Wevra console scripts** | `wevra`, `wevra-import`, `wevra-ci`, `wevra-capsule` — installed by `pip install wevra`. | **Stable v3**. |
| **Wevra extension protocols** (`wevra.extensions`) | `SandboxBackend`, `TaskBankLoader`, `ReportSink` — runtime-checkable Protocols, discovered via `importlib.metadata.entry_points`. | **Stable v1**. |
| **Unified runtime** (`wevra.runtime`) | `SweepSpec` + `run_sweep`: one code path for mock and real-LLM runs, with an explicit `acknowledge_heavy` cost gate. Every sweep cell becomes a `SWEEP_CELL` capsule. | **Stable v1**. |
| **Legacy product path** (`vision_mvp.product`) | Pre-Wevra import path. Still works; re-exported by `wevra`. | **Deprecated-compat** — do not import in new code. |
| **Core substrate** (`vision_mvp.core`) | CASR routing, hierarchical router, context ledger, exact_ops, typed role-handoff, dynamic_comm, adaptive_sub. Research primitives Wevra rests on; each is adapter-able into the capsule surface (`capsule_from_handle`, `capsule_from_handoff`, …). | **Settled, but research API** — no SDK guarantees. |
| **Research shards** (`vision_mvp.experiments`, `vision_mvp.tasks`, `RESULTS_PHASE*.md`, `EXTENDED_MATH_*.md`) | 45+ phases of falsifiability experiments, 72-framework theory survey, proofs. | **Research-grade** — empirical/proved per shard; no product-API guarantee. |
| **Boundary / next-slice** | Docker-first-by-default for public/untrusted JSONLs; first real out-of-tree plugin exemplar; release-on-tag firing. | **Declared, not fired** — see master plan § 10.5. |

For the full living stability matrix, see
[`context_zero_master_plan.md` § 10.1](context_zero_master_plan.md#101-stability-matrix-living).

---

## Minimal mental model

```
    Context Zero (research programme)
    ├── Theory: PROOFS.md, EXTENDED_MATH_[1-7].md, OPEN_QUESTIONS.md
    ├── Substrate: vision_mvp/core/*  (CASR router, exact memory,
    │                                   typed handoff, runtime calibration)
    ├── Research shards: vision_mvp/experiments/*, vision_mvp/tasks/*,
    │                    RESULTS_PHASE*.md (empirical diary, per phase)
    │
    └── Wevra (shipped product slice)
        ├── SDK:     vision_mvp/wevra/          (stable contract)
        ├── CLI:     wevra / wevra-import / wevra-ci
        ├── Runtime: wevra.runtime.run_sweep    (mock + real, one path)
        ├── Plugins: wevra.extensions           (3 Protocols + registry)
        ├── Schemas: phase45.product_report.v2, wevra.provenance.v1, …
        └── Legacy:  vision_mvp/product/*       (deprecated-compat)
```

The rule of thumb: **anything imported from `vision_mvp.wevra` is
product; anything else is research substrate or research shard.**

---

## What Wevra is — and what it is not

**Wevra IS:** a drop-in SDK for profile-driven evaluation runs on
SWE-bench-Lite-shape task banks, with a stable report schema, a CI
gate, a provenance manifest on every run, and a plugin surface for
sandboxes / task banks / report sinks.

**Wevra is NOT:**
- The whole Context Zero research programme.
- A universal multi-agent platform.
- A replacement for SWE-bench harnesses on arbitrary-shape tasks.
- An orchestrator for training runs or long-lived agent services.

The distinction matters: Wevra is deliberately narrow so that what it
does, it does with proofs and provenance. Scope creep is resisted on
purpose.

---

## Fastest path from zero to a real report

```bash
git clone <this-repo>
cd context-zero
pip install -e .[docker]            # Docker extra optional, recommended for public JSONLs
wevra --profile local_smoke --out-dir /tmp/wevra-smoke
wevra-ci       --report /tmp/wevra-smoke/product_report.json --min-pass-at-1 1.0
wevra-capsule  view   --report /tmp/wevra-smoke/product_report.json
wevra-capsule  verify --report /tmp/wevra-smoke/product_report.json
```

Four files of interest land in `/tmp/wevra-smoke/`:

- `product_report.json`   — machine-readable report (`phase45.product_report.v2`),
  includes a `capsules` block (`wevra.capsule_view.v1`).
- `capsule_view.json`     — the sealed capsule DAG on disk.
- `provenance.json`       — reproducibility manifest (`wevra.provenance.v1`).
- `product_summary.txt`   — human summary with a capsule-kind histogram,
  `chain_ok` flag, and the RUN_REPORT capsule's root CID.

Send someone the `root_cid` plus the JSONL SHA-256 (already recorded
in `provenance.json`) and they have everything needed to reproduce,
audit, and verify the run — no out-of-band trust required.

For a real-LLM sweep, set `WEVRA_OLLAMA_URL` and add `--acknowledge-heavy`.

---

## Where to go from here

- **I want to use Wevra** → [`README.md § Wevra SDK quick start`](../README.md) and
  [`vision_mvp/wevra/__init__.py`](../vision_mvp/wevra/__init__.py) (module docstring
  lists the full public surface).
- **I want to extend Wevra** → [`vision_mvp/wevra/extensions/`](../vision_mvp/wevra/extensions/)
  and the `examples/out_of_tree_plugin/` folder (minimal standalone package
  demonstrating the `entry_points` path).
- **I want to understand the substrate** → [`ARCHITECTURE.md`](../ARCHITECTURE.md)
  (skip the per-phase callouts on first read) and
  [`context_zero_master_plan.md` § 3](context_zero_master_plan.md).
- **I want the theory** → [`PROOFS.md`](../PROOFS.md), `EXTENDED_MATH_[1-7].md`,
  and [`context_zero_master_plan.md` § 1–2](context_zero_master_plan.md).
- **I want the research diary** → `vision_mvp/RESULTS_PHASE*.md` in
  order; `RESULTS_WEVRA_SLICE2.md` for the most recent SDK milestone.

---

*This document is the canonical orientation. If it and any other file
disagree on identity or scope, this document is right and the other
file is stale.*
