# Research status — canonical, current

> Single-source-of-truth for the *active* research position of the
> Context Zero programme. If this file disagrees with any other
> doc on what is *true now*, this file is right and the other file
> is stale. For *theorem-by-theorem* status, see
> `docs/THEOREM_REGISTRY.md`. For *what may be claimed*, see
> `docs/HOW_NOT_TO_OVERSTATE.md`. Last touched: SDK v3.5,
> 2026-04-26.

## TL;DR

The programme now has **four** coupled research axes, each with a
sharp status:

1. **Capsule contract / runtime** — *active, advancing*. The
   contract (C1..C6) is settled. SDK v3.4 pushes capsule-native
   execution to the LLM byte boundary inside one Wevra run
   (W3-42..W3-45). The lifecycle audit covers L-1..L-11.
2. **Multi-agent capsule coordination** — *active, new (SDK
   v3.5)*. Capsule-native team coordination via TEAM_HANDOFF /
   ROLE_VIEW / TEAM_DECISION capsules. ``TeamCoordinator`` drives
   one round; ``audit_team_lifecycle`` mechanically verifies
   invariants T-1..T-7 (Theorem W4-1). Coverage-implies-
   correctness (W4-2) and local-view limitation (W4-3) hold on
   the Phase-52 incident-triage bench. A learned per-role
   admission policy admits **strictly fewer handoffs** (12/12
   train seeds, deterministic in direction) and improves pooled
   team-decision accuracy *on most seeds* (gap_full > 0 in 11/12
   seeds, mean +0.054; gap_root_cause > 0 in 8/12 seeds, mean
   +0.032) over the strongest fixed baseline (coverage-guided)
   on the Phase-52 default config — but the accuracy advantage
   reverses at higher noise (W4-C1 honest reading). This is the
   team-level slice of the original Context-Zero "solve context
   for multi-agent teams" thesis — the first slice that runs the
   capsule abstraction *between* agents, not just inside one run.
3. **Decoder frontier** — *open, with sharp limitation theorems*.
   The strict pre-Phase-50 paradigm-shift bar (W3-C7 strict) is
   **retracted** (W3-26, W3-27). The defensible reading is
   W3-C9 (Phase-49 candidate at $n=80$, gap reading at zero-shot).
   The next research direction is the relational decoder at
   higher level (Phase 51, W3-30 / W3-31 / W3-C10).
4. **Substrate primitives** — *settled*. CASR routing, exact
   memory, typed handoffs, escalation threads, adaptive
   subscriptions. ~1500 substrate tests, no active development on
   substrate primitives themselves.

## Current frontier (SDK v3.5, 2026-04-26)

### Active moves (SDK v3.5 — multi-agent capsule coordination)

- **Capsule-native multi-agent team coordination
  (W4 family).** Three new closed-vocabulary capsule kinds
  (TEAM_HANDOFF, ROLE_VIEW, TEAM_DECISION) make capsules
  load-bearing *between* agents. ``TeamCoordinator`` drives one
  coordination round end-to-end; ``audit_team_lifecycle``
  mechanically verifies T-1..T-7 (Theorem W4-1).
- **Coverage-implies-correctness** (W4-2, proved-conditional) and
  **Local-view limitation** (W4-3, proved-negative) anchor the
  team-level mechanism in the formal layer.
- **Learned per-role admission policy** (``team_policy.py``)
  strictly improves pooled team-decision accuracy over the
  strongest fixed baseline at matched per-role budgets on the
  Phase-52 incident-triage bench (W4-C1 positive empirical;
  conjectural at smaller train scales).
- **Phase-52 reference benchmark**
  (``vision_mvp/experiments/phase52_team_coord.py``) compares
  substrate / capsule_fifo / capsule_priority / capsule_coverage
  / capsule_learned head-to-head and reports
  ``audit_ok_rate = 1.000`` for every capsule strategy.

### Active moves (SDK v3.4 — still in force)

- **Capsule-native execution one further structural layer past
  v3.3.** PROMPT capsule sealed for every LLM call's prompt
  bytes; LLM_RESPONSE capsule sealed for every response bytes.
  PROMPT.parents = (SWEEP_SPEC,) (Theorem W3-42); LLM_RESPONSE
  parent = sealed PROMPT (Theorem W3-43); PARSE_OUTCOME may
  parent on (SWEEP_SPEC, LLM_RESPONSE) so the
  prompt → response → parse → patch → verdict chain is a
  typed DAG witness end-to-end (Theorem W3-44).
- **Lifecycle audit extended to L-9 / L-10 / L-11** (Theorem
  W3-45). Soundness: ``audit_capsule_lifecycle(ctx).verdict ==
  "OK"`` iff the ledger satisfies the eleven invariants.
- **Synthetic-LLM mode for CI-runnable end-to-end exercise.**
  ``SweepSpec(mode="synthetic", synthetic_model_tag=<tag>)``
  uses a deterministic in-process synthetic LLM client; no
  network. The full prompt/response/parse/patch/verdict chain
  seals end-to-end on every (task, strategy).
- **Cross-model parser-boundary research (W3-C6, empirical).**
  ``vision_mvp.experiments.parser_boundary_cross_model``
  reports cross-distribution PARSE_OUTCOME failure-kind TVD up
  to 1.000 across the synthetic distribution library, and
  strict→robust parser-mode shift up to 1.000 on
  ``synthetic.unclosed``.

### Active moves (SDK v3.3 — still in force)

- **PARSE_OUTCOME lifecycle gate.** Theorem W3-39.
- **Runtime-checkable lifecycle audit.** Theorem W3-40 / W3-45.
- **Deterministic-mode replay.** Theorem W3-41.

### Sharp limitation theorems we hold

- **W3-14** (negative): per-capsule budgets cannot enforce
  table-level cardinality invariants.
- **W3-16** (negative): cohort-lifting cannot enforce relational
  invariants.
- **W3-17** (conditional): admission rules cannot exceed the
  priority-decoder ceiling under ceiling-forcing spurious
  injection.
- **W3-21** (negative): linear class-agnostic decoders cannot
  achieve symmetric zero-shot transfer when gold-conditional
  feature signs flip across domains.
- **W3-29** (lower bound): magnitude-monoid linear family cannot
  cross the Bayes-divergence zero-shot risk lower bound.
- **W3-36** (sharp impossibility): the primary capsule ledger
  cannot authenticate its own rendering's bytes.
- **W4-3 (SDK v3.5)** (proved-negative): per-role budget below
  the role's causal-share floor admits sound runs that fail the
  team gate; no admission policy (FIFO, priority, coverage,
  learned) can recover. The natural next move is a
  cohort-lifted role view (W4-C2, conjectural).

### Active conjectures

- **W3-C1**: every Phase-N bounded-context theorem subsumes under
  the capsule contract. Conjectural (the four-case subsumption is
  proved; the general statement is open).
- **W3-C5 (legacy SDK v3.3)**: a sub-intra-cell PROMPT /
  LLM_RESPONSE capsule slice closes the inner-loop boundary
  without breaking W3-34 spine equivalence. **DISCHARGED in
  SDK v3.4** by Theorems W3-42 / W3-43 / W3-44 / W3-45.
- **W3-C6 (new in SDK v3.4)**: synthetic-LLM cross-distribution
  PARSE_OUTCOME failure-kind TVD ≥ 0.5 across the calibrated
  distribution library. **Empirical** (reproducible; not a
  proof — the distribution library is synthetic, not real
  cross-LLM).
- **W3-C9**: refined paradigm-shift reading (Phase-49 candidate at
  $n=80$ point-estimate, zero-shot gap reading).
- **W3-C10**: relational decoder level-ceiling.
- **W4-C1 (SDK v3.5)**: learned per-role admission policy
  admits strictly fewer handoffs (12/12 seeds, robust direction)
  and improves pooled team-decision accuracy on most train seeds
  (gap_full > 0 in 11/12 seeds, mean +0.054; gap_root_cause
  > 0 in 8/12 seeds, mean +0.032) over the strongest fixed
  admission baseline (coverage-guided) on the Phase-52 default
  config — but the accuracy advantage reverses at higher noise
  (spurious=0.50). Empirical: budget-efficiency dominance is
  robust per-seed; accuracy advantage is mean-positive, not
  strict per-seed.
- **W4-C2 (SDK v3.5)**: cohort-lifted role view closes W4-3 on a
  sub-class of scenarios.
- **W4-C3 (SDK v3.5)**: capsule-layer admission rule subsumes
  the Phase-36 ``AdaptiveSubscriptionTable`` route-edit primitive.

### Active retractions

- **W3-C7 (strict reading) is retracted.** "Point-estimate
  Gate 1 at $\hat p \ge 0.400$ AND strict zero-shot Gate 2 with
  per-direction penalty ≤ 5pp" was falsified at $n=320$ (W3-26,
  W3-27). Do not reintroduce the strict bar.
- **W3-C3** is retracted in favour of W3-15 cohort lift.
- **The earlier W3-C4** (now reused for SDK-v3.3
  PARSE_OUTCOME conjecture) named a candidate decoder paradigm
  shift; the strict reading is folded into W3-C7 retraction.

## What we are NOT actively claiming

- **Not** "we solved context."
- **Not** "we solved multi-agent context." SDK v3.5 makes the
  capsule abstraction load-bearing at the **team boundary** on
  one synthetic benchmark family (Phase-52 incident-triage) under
  a deterministic team decoder. The W4-2 result is
  proved-conditional (premises: faithful decoder + sound
  admission); the W4-C1 learned-policy advantage is empirical-
  positive on the default config and conjectural on smaller
  training scales. External validity to real production
  multi-agent teams is open.
- **Not** "the runtime is fully capsule-native." Specifically not
  capsule-native: sandbox stdout/stderr, sub-step parser-internal
  objects (regex match objects, recovery heuristic intermediate
  state), and on-the-wire LLM streaming chunks. PROMPT bytes and
  LLM_RESPONSE bytes ARE now capsule-tracked under SDK v3.4 (the
  prior "not capsule-native: LLM prompt bytes, raw LLM response
  bytes" line is **superseded** by Theorems W3-42 / W3-43).
- **Not** "Wevra is a universal multi-agent platform."
- **Not** "the decoder beat the Phase-31 ceiling by 22.5 pp."
  The sharper reading is W3-19 ($+15$pp at $n=80$, FIFO admission).
- **Not** "deterministic mode means the entire run is
  reproducible." It means the *capsule DAG* is reproducible under
  a frozen JSONL + a deterministic profile. Wall-clock and
  host-local fields are stripped from CIDs but live on disk.
- **Not** "the synthetic-LLM cross-distribution study is a real
  cross-LLM study." The distributions are calibrated synthetic
  (see ``synthetic_llm.SYNTHETIC_MODEL_PROFILES``), not
  measurements of ``gemma2:9b`` / ``qwen2.5:7b`` outputs. The
  empirical claim is about the parser's failure-kind closed
  vocabulary's *resolving power*, not about LLM output
  distributions in the wild.

## How to update this document

1. When a phase ships, add one line to the "Active moves" list and
   move any superseded line to "Sharp limitation theorems we hold"
   or "Active retractions" as appropriate.
2. When a conjecture is proved or falsified, move it to the
   correct section and update `THEOREM_REGISTRY.md`.
3. When a milestone note ships, add a one-line cross-link in this
   doc's relevant section.
4. Always update the "Last touched" date at the top.

## Cross-references

- Formal model (run-boundary, W3 family): `docs/CAPSULE_FORMALISM.md`
- Formal model (team-boundary, W4 family): `docs/CAPSULE_TEAM_FORMALISM.md`
- Theorem registry: `docs/THEOREM_REGISTRY.md`
- How-not-to-overstate rules: `docs/HOW_NOT_TO_OVERSTATE.md`
- Master plan: `docs/context_zero_master_plan.md`
- Milestone notes: `docs/RESULTS_WEVRA_*.md`,
  `docs/RESULTS_CAPSULE_*.md`,
  `docs/RESULTS_WEVRA_DEEP_INTRA_CELL.md` (SDK v3.3),
  `docs/RESULTS_WEVRA_INNER_LOOP.md` (SDK v3.4),
  `docs/RESULTS_WEVRA_TEAM_COORD.md` (SDK v3.5 — this milestone)
- Paper draft: `papers/wevra_capsule_native_runtime.md`
- Tests: `vision_mvp/tests/test_wevra_capsule_native*.py`,
  `test_wevra_capsule_native_deeper.py`,
  `test_wevra_capsule_native_inner_loop.py` (SDK v3.4),
  `test_wevra_team_coord.py` (SDK v3.5 — multi-agent slice),
  `test_capsule_*.py`
- Cross-model parser-boundary experiment:
  `vision_mvp/experiments/parser_boundary_cross_model.py`
- Multi-agent team coordination benchmark:
  `vision_mvp/experiments/phase52_team_coord.py`
