# Frontier-relevance audit — W134 (deployable complexity witness)

Extends the W132 + W133 audits; all prior classifications remain in force. W134's distinguishing
contribution is the **distillation of the exact-oracle EW2 complexity witness into a deployable,
oracle-free public-signal witness** + a dedicated **complexity-only** held-out corpus. This audit
classifies the new assets and re-states what W134 does and does not entitle.

## Active frontier arsenal (load-bearing / reusable / push-button)

* **`coordpy.deployable_complexity_witness_v1`** — the W134 advance. A deployable, oracle-free
  complexity witness: DW1 constraint-derived budget (`derive_budget_fact_v1`, bridging
  `public_signal_selection_oracle_v1.parse_max_constraint_v1`), DW2 public-format stress-growth
  ladder + log-log exponent fit + extrapolation (`build_ladder_v1` / `measure_growth_v1`), DW3
  rewrite prompt, DW4 KEEP/REWRITE/ABSTAIN controller (bridging
  `stronger_generator_slate_v1.complexity_admissible_v1` / `COMPLEXITY_OPS_BUDGET`), and the
  same-budget arm `run_deployable_witness_arm_v1` that consumes ONLY `(code, statement, public
  samples)` — no reference/naive/brute/secret. Reusable + push-button on any complexity corpus.
  **Lane-β verdict (held-out dev, 70B):** the $0 naive/ref separation establishes it is faithful to
  EW2 at the PROGRAM level (45/45), and at the MODEL it is REAL (D1/D2 +5.56pp over B0, MLB-2 90%+)
  but SUB-ORACLE (~half of C0's +11.11pp) and single-family ⇒ the §7a dev gate FAILS ⇒ $0 eval /
  $0 frontier (`W134-L-DEPLOYABLE-COMPLEXITY-WITNESS-DEV-CAP`). It STANDS as a reusable push-button
  instrument, not a confirmed frontier mechanism.
* **`coordpy.complexity_only_corpus_v1`** — the dedicated, deterministic, seed-disjoint
  COMPLEXITY-only train/dev/eval/frontier corpus (9 families × multi-seed). Locked eval split CID
  + frontier 30-slice CID; push-button. The first CoordPy corpus that isolates the one
  feedback-fixable sub-mode.
* **`scripts/run_w134_deployable_witness_bench_v1.py`** — dev/eval/frontier bench with the
  A0/A1/B0/C0(exact-oracle upper bound)/D1/D2/D3 arms, eval-CID + frontier-CID-guarded,
  `--lead` swappable. Same-budget; reuses the verbatim W108 evaluator.
* Consumed UNCHANGED (no mechanism drift): `exact_oracle_witness_v1` (EW2 = the C0 upper bound),
  `icpc_reflexion_bench_v1` (A0/A1/B0 + the blind reflexion fallback the deployable controller
  defers to), the audited grader `coordpy_icpc_battlefield_v1`, the W132 battlefield +
  `resistant_by_construction_slate_v1` (the 9 `cb_*` complexity templates), the verbatim W108
  evaluator. The mined public-signal modules `public_signal_selection_oracle_v1` (parse-max-
  constraint) + `stronger_generator_slate_v1` (complexity gate) are now bridged to the witness path.

## Useful baseline-only (diagnostic; not a frontier mechanism by itself)

* DW2's raw single-large-input TLE check WITHOUT the growth curve — that collapses to B0's blind
  "rejected/too-slow" bit (the W129 `derive_stress_cases_v1` cap: a generic format-preserving
  scale-up "never falsifies" on its own). The witness's value is the GROWTH CURVE + exponent +
  constraint-derived budget + target class, which the `deployable_witness_is_genuinely_new_v1`
  test gates on. A TLE-only signal is baseline, not the mechanism.
* The exact-oracle EW2 witness (`C0`) as a DEPLOYABLE mechanism — it is the UPPER BOUND and a
  faithfulness yardstick, NOT deployable (it executes a reference). It stays an evaluation anchor.
* `naive_source` / `ref_source` programs — used ONLY in the $0 instrument-validation (the
  naive/ref separation gate), never on any model-facing or deployed path.

## Dead directions (do NOT revive)

* Selling the exact-oracle complexity gain as deployable as-is (it consumes the oracle's timing;
  W134 is exactly the deployable-distillation milestone).
* A deployable witness that is only "your code timed out" with no measured curve / no exponent /
  no constraint-derived budget — that is B0's reject bit re-stated (fake-different prompt
  decoration); the genuinely-new test rejects it.
* Treating an empirically-fitted exponent as a ground-truth complexity oracle — the literature
  (SwiftSolve arXiv:2510.22626; GuessCompx arXiv:1911.01420) documents slope-fit instability at
  small n; W134 uses a guarded R² confidence gate and ABSTAINS to blind reflexion on a
  low-confidence fit (a calibrated diagnostic, not an oracle).
* Constructing problem-specific adversarial worst-case inputs (≈ solving the problem; the W129
  cap). W134 uses ONLY a fixed, pre-committed shape set (random / descending / constant); a family
  whose worst case is outside that set is a documented limitation, not a tuning knob.

## Anti-patterns (NOT dead — explicit anti-patterns, reaffirmed)

Bounded-context / compaction / prose-summary / "cram less / truncate better" remain anti-patterns,
explicitly NOT pursued. W134 is an exact-oracle-to-deployable witness-distillation + held-out-bench
milestone, NOT a context-compression milestone.

## Do not claim (W134 additions)

* That the deployable complexity witness is a confirmed third-retirement mechanism — W89 (+5.56)
  and W105 (+7.00) remain the only two retirements; a frontier rerun is single-seed and is NOT a
  retirement even on a pass (it would motivate a W135 multi-seed confirmation).
* That the deployable witness equals the exact-oracle witness in general — Lane β measures the gap
  (the C0 − D arm delta) on held-out complexity tasks; the earn rule requires staying within 2 pp
  of C0, and a miss is registered as a cap, not hand-waved.
* That a gain on the complexity-only field generalises to the value/algorithm capability-bound
  sub-modes (W133 showed those are capability-bound; W134 is scoped to COMPLEXITY_BLIND only).
* That the corpus's 9 families × multi-seed instances are 36 independent algorithm families — they
  are 9 distinct families × fresh seeds; the ≥3-family earn condition guards against a single-family
  fluke, and "seed families" is stated explicitly.
* That the empirical exponent is hardware-independent — it is a measured diagnostic on this host;
  the verdict (TLE / super-linear / over-budget) is robust, the exact wall-times are not.
