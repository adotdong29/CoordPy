# RESULTS W140 — tutor-compiler design enumerations (Lane α brainstorm)

**Status: design lane, pre-build.** The operator-required broad enumeration that justified the locked TC1–TC5 slate (RUNBOOK_W140 §4b). Killed ideas are explicit. The two shared families: FAMILY A `count_pairs_sum_le_t` (COMPLEXITY; naive O(N²) TLEs at N=50000; technique = sort + two-pointer; `algo_sig sort_two_pointer_pairsum`) and FAMILY B `subarrays_sum_and_range` (HIDDEN_EDGE; naive drops the (max−min)≤R constraint; technique = sliding window + max-deque + min-deque; `algo_sig two_pointer_two_constraint_deques`).

## (1) ≥10 tutor formats (ranked best-for-weak-model first; leak risk noted)

1. **Holed pseudocode skeleton** — the technique scaffold with the DISCRIMINATING predicate/aggregation blanked. Highest lift (hands the 8B the loop structure + the block-count trick it never invents). Leak: MED → guarded by hole-substance + discriminator-absence. **[→ TC2 skeleton]**
2. **Naive-vs-efficient contrast + failure reason** — the 8B already writes the naive; seeing the efficient diff is the lesson (esp. FAMILY B's dropped constraint). Leak: MED-HIGH → de-identified, discriminator blanked.
3. **Named-technique card + budget fact + key move** — lowest-leak high-lift; the spine. Leak: LOW. **[→ TC1]**
4. **Decision tree: signal → technique** — externalizes algorithm SELECTION (the 8B's real deficit). Leak: LOW.
5. **Common-bug warning list** — pre-empts the family's known anti-patterns (drop the range check / double-loop). Leak: LOW. **[→ TC1 bug_warnings]**
6. **Worked micro-trace on the PUBLIC sample** — grounds the abstract loop in observed state; MUST use the public sample only. Leak: MED → `public_only_literals`. **[→ TC5 stage 3]**
7. **Invariant / precondition checklist** — tells the 8B when to stop adjusting pointers (#1 off-by-one source). Leak: LOW. **[→ TC1 invariants]**
8. **Minimal one-line rule ("the single fact")** — smallest dose; for when longer text derails the 8B. Leak: LOW for multi-step families. **[→ TC3]**
9. **Staged / progressive disclosure** — dose-matched escalation under execution feedback. Leak: LOW→MED, monotone in stage. **[→ TC5]**
10. **Counterexample-by-shape** ("fails on a few large outliers among small values") — SHAPE words, never a concrete input. Leak: MED → `discriminator_shape_only`.
11. **Primitive-availability hint** ("deque has O(1) ends; use two") — unblocks FAMILY B's sub-skill. Leak: LOW. **[→ TC1 primitive_hint]**
12. **Reflexion self-check question** ("does your code use BOTH constraints?") — a verification trigger. Leak: LOW; weak alone, good closer.

**Killed:** full annotated reference solution (collapses to a paste leak + the 8B copies, learns nothing); correctness proof (8B can't consume it; pure token cost); long prose essay (triggers the long-text instruction breakdown that is a leading cause of the raw-witness 0/5); textbook/Leetcode-pattern citation (8B can't dereference; one-liner-leak risk for trivial families).

## (2) ≥8 ways a strong-model explanation fails on a weak model (+ the mitigation)

1. **Names a technique the 8B can't implement** → bind every name to a holed skeleton + primitive hint (never a bare name).
2. **Too abstract / no executable skeleton** → require a compilable shape (TC2).
3. **Assumes a sub-skill (deque maintenance / the `cnt += j-i` block trick)** → expand the sub-skill inline as its own micro-scaffold.
4. **Introduces a NEW bug while fixing the old** (mirrors W128 selection cap) → ship invariants + bug-warnings targeting the fix's own failure mode + a self-check.
5. **Overfits to the public sample** → keep worked traces on the public sample only + warn "do not hardcode the sample output"; counterexample-by-shape counter-programs it.
6. **Long-text instruction breakdown** (a direct cause of the raw-witness 0/5) → hard token budget (TC3) + staged disclosure (TC5).
7. **Reverts to naive under uncertainty** → supply a trust anchor (public-sample micro-trace) + invariants so it stops second-guessing.
8. **Symbol/notation mismatch** (text uses l,r,S,R; problem says low/high/budget) → family-generic variable names + an explicit constraint→variable mapping.
9. **Misallocated emphasis** (buries the crux) → crux-first ordering (key move first).
10. **No failure localization** (can't map symptom → root cause) → TC2 binds the observed witness to the named root cause + the specific rewrite (the biggest lever over the bare diagnostic).

## (3) ≥6 ways a tutor can leak the answer while looking "abstract" (+ the machine guard)

1. **Reference source pasted as "pseudocode"** → `holes_are_substantive` (a hole-free skeleton is rejected) + `no_reference_paste` (bounded contiguous run vs ref).
2. **Worked example numbers == a hidden test case** → `public_only_literals` (every literal byte-equals a public sample; no secret-only literal).
3. **Closed-form answer formula** → `is_procedure_not_answer` (the tutor carries control flow, not a terminal answer expression).
4. **Trivial-holes skeleton** (the solution with cosmetic blanks) → `holes_are_substantive` (the trivially-stubbed skeleton must FAIL public ⇒ the holes carry the answer-logic).
5. **Technique name == the whole algorithm for a one-liner family** → `one_liner_family_guard` (cards disabled where the reference is ≤1 statement / no branching; both W140 families certified multi-step).
6. **Secret-case discriminator input in a "fails on…" description** → `discriminator_shape_only` (no concrete array literal; shape words only).
7. **Sub-skill expansion reconstructs the reference** → the sub-skill micro-scaffold itself passes `no_reference_paste` + leaves the constraint predicates as holes.
8. **Train↔test collision** → `train_test_disjoint` (TRAIN seed base `140_1×10^5` byte-disjoint from dev/eval/frontier; no secret-case literal in the tutor) + the DECISIVE behavioral guarantee (grading on the disjoint hidden bank).

**Decisive principle (from the leakage research, §RESULTS_W140_RESEARCH_V1 B):** text leak-detectors are defeatable, so the answer-presence check is the DISCRIMINATOR (`spec.correct_fill`, a ≥5-token contiguous run) — always a HOLE in a legit tutor — and the ground truth is behavioral (disjoint hidden grade). The contiguous-run-vs-ref cap is a secondary gross-paste tripwire only; a legit holed scaffold may share the I/O boilerplate + the standard monotonic-deque idiom with the reference (a teachable primitive), which is reported transparently (`longest_ref_run`), not penalized.

## Recommended slate → locked TC1–TC5 + T6

| Arm | Object | Why | Leak guard |
|---|---|---|---|
| **TC1** (T1) | family card: name + budget + key move + primitive + bug warnings + invariants (deterministic from `headroom_note`+`algo_sig`) | lowest-leak, names the technique | discriminator-absence + one-liner guard |
| **TC2** (T2/T4) | TC1 + witness→rewrite routing + holed skeleton | failure-localization + worked scaffold (the highest-lift object; carries the oracle) | hole-substance (stubbed fails public) + discriminator blanked |
| **TC3** (T3) | smallest sufficient: one-line rule + key move + 1 warning, token-budgeted | the dose-matched winner if long text hurts the 8B | same as TC1, smaller surface |
| **TC4** (T4, LEAD) | the W139 capability-matched controller, fed TC2, routed by tutor-usability | turns cross-tier SAFETY into USEFULNESS; KEEP ⇒ non-negativity | routes only gated tutors |
| **TC5** (T5) | progressive disclosure (stage 0 TC3 → stage 3 +trace/self-check) | dose minimal, escalate on failure | per-stage gating |
| **T6** | content-free do-better instruction | the fake-different probe | must be non-genuine (it is) |
