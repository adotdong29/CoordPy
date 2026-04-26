# RESULTS — Capsule research milestone (post-SDK-v3)

> *Theory-forward results note. The SDK-v3 milestone
> (`docs/RESULTS_WEVRA_CAPSULE.md`, 2026-04-22) named the
> capsule abstraction. This note moves it from a product label
> to a **research center** by adding (a) a formal mathematical
> model, (b) an ML problem with a real held-out result, (c) an
> empirical unification audit, and (d) honest scope on what is
> and is not subsumed.* Last touched: 2026-04-22.

---

## What is new here, scientifically?

A skeptical technical reviewer asking "what is genuinely new in
this milestone?" should read this section first.

### Inherited (NOT new)

* **Content addressing.** Merkle/Git/IPFS — older than this
  project.
* **Hash-chained logs.** Tamper-evident-log research (Haber &
  Stornetta 1991; Certificate Transparency).
* **Typed claim kinds.** Actor systems, event-sourcing, CQRS.
* **Lifecycle states.** Session-typed protocols, RAII, linear
  types.
* **Frozen dataclasses.** Standard-library shape; Phase 19
  Handle, Phase 31 TypedHandoff, Phase 35 ThreadResolution all
  use it.
* **Bounded-context theorems on substrate primitives.** Phase 19
  L2, Phase 31 P31-3, Phase 35 P35-2, Phase 41/43 P41-1 each
  exist on their own.

### The new unification (this milestone, code-backed)

* **A formal capsule space** $\mathcal{C}$ with explicit
  algebraic structure: capsule space, lifecycle automaton,
  ledger transition system, capsule DAG, **budget tropical-min
  monoid** $(\mathbb{B}, \oplus)$ with admissibility predicate
  $\mathcal{A}_b$ proven *monotone under tightening* (Theorem
  W3-8, Lemma 3.2). Source:
  `docs/CAPSULE_FORMALISM.md` § 1–3.
* **Theorem W3-11 (capsule subsumption, partial).** Four prior
  bounded-context theorems are *operationally* a special case
  of one capsule-budget invariant, with the reduction tuple
  $(k_T, b_T)$ written down explicitly:

  | $T$            | Primitive       | $k_T$               | $b_T$                                                      |
  |---             |---              |---                  |---                                                         |
  | L2 (P19)       | Handle          | HANDLE              | $b_t = B_{\rm worker}$                                     |
  | P31-3          | TypedHandoff    | HANDOFF             | $b_t = \tau$                                               |
  | P35-2          | EscalationThread/ThreadResolution | THREAD_RESOLUTION | $b_t = \tau$, $b_r = R_{\max}$, $b_w = W$ |
  | P41-1 / P43-1  | SubstratePromptCell | SWEEP_CELL      | $b_b = \beta_{\rm cell}$                                   |

  Verified operationally in
  `vision_mvp/experiments/phase46_unification_audit.py` —
  4/5 FULL, 1/5 PARTIAL across the audited primitives.
* **Falsifiable conjecture W3-C1 (full subsumption).** Stated
  with sharp falsifier: a Phase-N theorem whose substrate
  guarantee depends on a quantity *with no representation* in
  the five capsule budget axes. AdaptiveEdge (Phase 36) is the
  near-falsifier — it admits only a partial reduction.

### The ML problem opened by capsules (this milestone, code-backed)

* **Capsule admission as a learning problem.** Section 5 of
  `docs/CAPSULE_FORMALISM.md` (Conjecture W3-C4) and Section 4
  of `docs/RESULTS_CAPSULE_LEARNING.md` — admission is
  **strictly learnable** in the per-capsule classification
  sense:

  * Theorem-style claim **P46-1**: on Phase-31 incident triage
    under noisy extractors, a logistic-regression admission
    policy on capsule headers strictly beats the strongest
    non-learned heuristic on admit-precision at every tight
    budget cell. At budget 16 tokens: 0.796 vs 0.634 (FIFO),
    +16.2 pp. Held-out by-seed split, $n_{\rm test} = 40$
    instances, ~9 s wall.
  * Three new conjectures (P46-C1 bundle-aware admission;
    P46-C2 cross-domain transfer; P46-C3 rate-distortion
    optimality of header features) form the explicit Phase 47
    agenda.

* **The full-bundle decoder ceiling** (0.225) is a *real,
  honest negative*: per-capsule learning lifts admit precision
  but cannot rescue a priority decoder against
  bundle-poisoning. This is what the next ML question is
  about, not a hidden failure.

### What is still conjectural (named with falsifiers)

* **W3-C1 (full subsumption).** Settled on 4 primitive classes;
  open on Phases 28, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 44.
* **W3-C2 (CID-pinned reproducibility).** Stated in milestone
  note; proof obligations under SHA-256 second-preimage
  resistance.
* **W3-C3 (kind-alphabet completeness).** 11 kinds is a
  snapshot, not universal. Cross-run references and
  out-of-tree adapters are named candidate twelfth-kind risks.
* **W3-C4 (admission learnability).** First positive evidence
  via P46-1; further conjectures P46-C1..C3 are open.

### What this is NOT a claim of

* **Not a paradigm shift on its own.** The milestone produces
  one formalisation, one ML experiment, one audit, and four
  honest negatives. It strengthens the existing claim that
  "context is an object" by adding mathematical and ML weight
  to it; it does *not* prove the claim is universal.
* **Not a replacement for the substrate's own theorems.**
  P31-5 (expressivity separation), P35-1 (priority-inversion),
  P31-4 (correctness preservation) are *NOT* subsumed by the
  capsule contract — they are properties of *role topology*
  and *extractor soundness*, not capsule budgets. The
  formalisation is explicit about this in Theorem W3-11's
  "what is NOT subsumed" clause.
* **Not a category-theoretic restatement.** The formalisation
  uses DAGs and a min-monoid; categorical machinery would buy
  nothing the present view doesn't already provide.
* **Not an authentication / authorization claim.** The
  tamper-evidence is forensic under SHA-256, not a defence
  against malicious re-publishers.

---

## 1. Summary by part of the milestone

### PART A — Mathematical formalisation

Source: `docs/CAPSULE_FORMALISM.md` (~580 lines, ~10 KB).

* Defined capsule space $\mathcal{C}$, identity map
  $\mathit{cid}$, admissibility predicate $\mathcal{A}_b$,
  budget monoid $(\mathbb{B}, \oplus, \bot^5)$, capsule DAG
  $G(\mathcal{L})$.
* Stated W3-7 (CID is a homomorphism on parent canonicalisation),
  W3-8 (admissibility monotone under tightening, Lemma 3.2
  promoted), W3-9 (ledger DAG acyclic, append order is
  topological), W3-10 (chain tamper-evidence under SHA-256),
  W3-11 (capsule subsumption — the four-row table), W3-12 (view
  is a faithful header projection), W3-13 (DAG height bounded by
  4 on canonical run).
* Distinguished the *proven sub-class* of W3-11 from the open
  Conjecture W3-C1.

### PART B — Capsule learning problem

Source: `vision_mvp/wevra/capsule_policy.py` (~430 lines),
`vision_mvp/experiments/phase46_capsule_learning.py` (~340
lines), `docs/RESULTS_CAPSULE_LEARNING.md` (~330 lines),
`vision_mvp/tests/test_capsule_policy.py` (~190 lines).

* Defined `AdmissionPolicy` interface.
* Implemented four heuristic baselines (`FIFOPolicy`,
  `KindPriorityPolicy(cutoff=4|8)`, `SmallestFirstPolicy`)
  and one learned policy (`LearnedAdmissionPolicy`,
  logistic regression over a closed feature vocab).
* Implemented `BudgetedAdmissionLedger` — `CapsuleLedger` +
  global token budget + policy.
* Implemented `train_admission_policy` — full-batch GD,
  numpy-free, deterministic in `seed`.
* Built a real benchmark on Phase-31 incident-triage with
  Phase-32 noisy extractors, by-seed train/test split, 200
  scenario instances, 5 143 capsules.
* **Headline numbers** (test set, $n = 40$): learned policy
  beats every non-learned baseline on admit-precision at
  every budget $B \in \{16, 32, 48, 64\}$. At $B = 16$:
  precision 0.796 vs 0.634 (best heuristic), $+16.2$ pp.
* 6 contract tests pass; the experiment runs in ~9 s wall.

### PART C — Unification stress test

Source: `vision_mvp/experiments/phase46_unification_audit.py`
(~270 lines), `vision_mvp/tests/test_capsule_subsumption.py`
(~290 lines).

* Lifted each substrate primitive (Phase-19 Handle, Phase-31
  TypedHandoff, Phase-35 ThreadResolution, Phase-36
  AdaptiveEdge, end-to-end ProductReport) to a capsule via
  the canonical adapter.
* Verified `(k_T, b_T)` reduction holds operationally.
* **Result**: 4/5 FULL, 1/5 PARTIAL.
  * FULL: Handle, TypedHandoff, ThreadResolution, ProductReport.
  * PARTIAL: AdaptiveEdge — its `max_active_edges` bound is a
    table-level invariant, not a per-capsule invariant.
* 10 subsumption-stress tests pass, including 2 explicit
  *negative* tests that document where the capsule contract
  is silent (role topology, extractor soundness).

### PART D — Research narrative

This document.

### PART E — Minimal product/runtime adjustments

Source: `vision_mvp/wevra/__init__.py` (3 lines added).

* Re-exported `BudgetedAdmissionLedger` and the four policy
  classes as part of the public SDK surface, so external
  callers can drop in a capsule policy without crossing
  internal module boundaries. SDK_VERSION unchanged
  ("wevra.sdk.v3"); the public surface is *additive*.
* No runtime contract change. The unbudgeted
  `CapsuleLedger.admit_and_seal` path is byte-for-byte
  unchanged.

---

## 2. Theorem-style claims with explicit status

| Claim   | What it asserts                                                                     | Status                      |
|-------- |------------------------------------------------------------------------------------ |-----------------------------|
| W3-1..6 | (Milestone note) Capsule contract C1..C6 enforced.                                   | Code-backed.                |
| W3-7    | CID is a homomorphism on parent canonicalisation.                                    | Proved (§ 4 of formalism).  |
| W3-8    | Admissibility monotone under budget tightening.                                      | Proved (Lemma 3.2).         |
| W3-9    | Ledger DAG is acyclic; append order is topological.                                  | Proved (§ 4).               |
| W3-10   | Chain tamper-evidence under SHA-256 collision resistance.                            | Proved (forensic; § 4).     |
| W3-11   | 4-element subsumption sub-class.                                                     | Proved on 4 primitives.     |
| W3-12   | Capsule view is faithful header projection.                                          | Proved.                     |
| W3-13   | Run-pattern DAG height ≤ 4.                                                          | Proved by inspection.       |
| W3-C1   | Full subsumption across all Phase-19..44 bounded-context theorems.                   | **Conjectural.** Open.      |
| W3-C2   | CID-pinned reproducibility under SHA-256.                                            | Conjectural; W3-1 + W3-4.   |
| W3-C3   | 11 capsule kinds are a complete generating set.                                      | Conjectural; falsifiers named. |
| W3-C4   | Admission policy is learnable.                                                       | **Empirically supported by P46-1.** |
| **P46-1**| At budget 16, learned admit-precision = 0.796 ≫ best heuristic 0.634.               | Code-backed (`phase46_capsule_learning.py`). |
| P46-C1  | Bundle-aware admission lifts the noise-poisoning ceiling.                            | Conjectural; falsifier named.|
| P46-C2  | Cross-domain capsule policy transfer is non-trivial.                                  | Conjectural; falsifier named.|
| P46-C3  | Linear-in-headers learned policy is rate-distortion optimal on this distribution.     | Conjectural; falsifier named.|

---

## 3. Why this does — and does not — amount to a paradigm shift

### Why it does (cautious case)

* The capsule contract was previously a *product framing*
  ("context is an object"). After this milestone, it is
  *also*: (a) a formal mathematical structure; (b) an ML
  problem with a real held-out result; (c) an audit
  surface across four prior phases.
* The distinction matters for a reviewer asking "what's
  new here?": before, the answer was "the unification". After,
  the answer is "the unification, plus the formal model that
  proves the unification's reduction tuple, plus the
  learning result that opens admission as a research
  problem".
* The W3-11 reduction is *load-bearing for future
  research*: if a new substrate primitive ships, the
  question "does it fit the capsule contract?" is now a
  precise, falsifiable check rather than a framing
  judgement.

### Why it does not (the strict case)

* The four reductions are *operational* — each prior
  bounded-context theorem still has its own proof in its
  own phase note. The capsule contract subsumes them by
  exhibiting an adapter and a budget tuple, not by
  re-proving them under a unified machinery.
* The full subsumption (W3-C1) is not yet done. Phases 32,
  33, 34, 36, 37, 38, 39, 41, 42, 43, 44 each have
  bounded-context-shape theorems that have not been
  written out as capsule reductions.
* The ML result (P46-1) lifts admit-precision but does not
  yet lift downstream decoder accuracy — the noise-poisoning
  ceiling at 0.225 is unchanged. A paradigm shift in
  agent-team runtimes would require a downstream lift on
  end-to-end task accuracy, not just on a per-capsule
  classification metric.
* AdaptiveEdge is a partial fit. The "every coordination
  artifact is a capsule" claim has at least one near-
  falsifier in the existing substrate.

**The honest summary**: the milestone moves the capsule
abstraction from "product framing" to "research center" — but
"research center" is much weaker than "paradigm shift". The
shift, if there is one, will be earned by
(a) closing W3-C1 across the rest of the substrate,
(b) lifting decoder accuracy via P46-C1, and
(c) demonstrating cross-domain transfer via P46-C2. The
capsule milestone is the *precondition* for those, not the
result.

---

## 4. Files changed in this milestone

### New files (research artifacts)

* `docs/CAPSULE_FORMALISM.md` — formal mathematical model.
* `docs/RESULTS_CAPSULE_LEARNING.md` — Phase-46 ML result.
* `docs/RESULTS_CAPSULE_RESEARCH_MILESTONE.md` — this note.
* `vision_mvp/wevra/capsule_policy.py` — `AdmissionPolicy`
  framework, three heuristics, one learned policy,
  budgeted-admission ledger, training routine.
* `vision_mvp/experiments/phase46_capsule_learning.py` —
  benchmark driver.
* `vision_mvp/experiments/phase46_unification_audit.py` —
  per-primitive audit driver.
* `vision_mvp/tests/test_capsule_policy.py` — 6 contract
  tests on the policy framework.
* `vision_mvp/tests/test_capsule_subsumption.py` — 10
  subsumption-stress tests including 2 explicit negative
  cases.

### Modified files (additive only)

* `vision_mvp/wevra/__init__.py` — re-export
  `capsule_policy.*` symbols.
* `docs/context_zero_master_plan.md` — § 4.10 + § 4.11 + § 5
  add capsule-research-center section, ML/research agenda,
  and frontier triple.

### NOT modified (deliberate)

* `vision_mvp/wevra/capsule.py` — *byte-for-byte unchanged*.
  The capsule contract C1..C6 is the precondition of this
  milestone, not its product.
* All Phase-19..44 substrate primitives. The unification is
  *additive on top of* every Phase-N guarantee; no
  substrate primitive is modified.
* `vision_mvp/wevra/run.py`, `runtime.py`, `provenance.py`.
  No runtime contract change.

### Tests

* New: 16 (6 capsule-policy + 10 subsumption-stress).
* Pre-existing capsule tests (`test_wevra_capsules.py`,
  `test_wevra_public_api.py`): unchanged, all pass.
* Total new test code: ~480 lines.

---

## 5. Reproducibility

```bash
# Run the ML benchmark.
python -m vision_mvp.experiments.phase46_capsule_learning \
    --out-dir /tmp/wevra_phase46

# Run the unification audit.
python -m vision_mvp.experiments.phase46_unification_audit \
    --out-dir /tmp/wevra_phase46

# Run the new contract tests.
python -m pytest vision_mvp/tests/test_capsule_policy.py \
                  vision_mvp/tests/test_capsule_subsumption.py -v

# Run the full regression to confirm no Phase-N..Phase-44
# substrate test broke.
python -m pytest vision_mvp/tests/ -q --tb=no
```

Wall times on a 2024 macbook:

* Capsule-learning sweep: ~9 s.
* Unification audit: ~0.06 s.
* New contract tests: ~0.5 s.
* Full regression: see `tests/` (matches Phase-44 baseline).

---

## 6. Where to look next

1. **W3-C1 follow-up**: write capsule reductions for the 11
   remaining Phase-N bounded-context theorems. Mechanical;
   would close the conjecture entirely or surface a second
   honest non-fit.
2. **P46-C1 follow-up**: bundle-aware admission. The natural
   model is a per-bundle scoring function (e.g. a
   permutation-invariant set encoder) that operates on the
   *full proposed set* and returns a subset. Should beat the
   per-capsule learned policy on the noise-poisoning
   ceiling.
3. **P46-C2 follow-up**: cross-domain transfer. Train on
   incident-triage capsules, test on compliance-review or
   security-escalation capsules. Tests whether the
   header-level features generalise across domains.
4. **Adapter Protocol** (from milestone § 8 "what remains
   open"): a stable Protocol so external packages can
   register their own capsule kinds. Natural SDK v4 target.
