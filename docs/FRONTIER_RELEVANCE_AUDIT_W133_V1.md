# Frontier-relevance audit — W133 (supplement to the W132 V1 audit)

Extends the W132 V1 audit; all prior classifications remain in force. W133 classifies the
exact-oracle witness artifacts.

## Active frontier arsenal (load-bearing for the code line)

* **`coordpy.exact_oracle_witness_v1`** — the exact-oracle WITNESS instrument: the EW1
  (minimal counterexample, with a deterministic shrink trace) / EW2 (complexity-timing) /
  EW3 (edge) / EW4 (search-enum) slate, the content-addressed `WitnessProbeSetV1` API (fresh
  `witness_seed`, leakage-clean by witness type), the **same-budget** witness arm
  `run_witness_arm_v1` (a strict swap of the W120/W132 B-arm feedback object), and the
  `witness_is_genuinely_new_v1` fake-different test. **This is the W133 advance** — it turns
  the owned battlefield into a teacher and localises the W132 cap. The **EW2 complexity-timing
  witness is the load-bearing component** (+6.06 pp over B0 on held-out dev); EW1/EW3/EW4
  counterexamples are real + leakage-clean but add 0 over blind reflexion at 70B (capability
  ceiling). Reusable + push-button on the curriculum.
* **`coordpy.witness_curriculum_corpus_v1`** — the deterministic seed-disjoint train/dev/eval
  curriculum over `RBC_SLATE_V1` (99 admitted, content-CID + seed disjoint, family-balanced).
  A reusable held-out corpus; the eval split (`88b9b79a…`) is locked + push-button for an
  operator-greenlit confirmation run.
* **`scripts/run_w133_witness_bench_v1.py`** — the held-out witness bench (dev / eval /
  frontier modes; eval-CID-guarded; `--lead` swappable). EXECUTED on dev
  (`meta/llama-3.3-70b-instruct`, Maverick infra-down) ⇒ single-mode complexity gain; the eval
  + frontier modes are push-button if the cap is later revisited.
* The validated W120 bench (`icpc_reflexion_bench_v1`) + the audited grader
  (`coordpy_icpc_battlefield_v1`) + the W132 battlefield (`resistant_by_construction_*_v1`) +
  the verbatim W108 evaluator — consumed UNCHANGED; the witness arm bridges onto them (no
  mechanism drift; "C − A1" scored byte-identically to "B − A1").

## Useful baseline-only

* **EW1/EW3/EW4 counterexample witnesses** — REAL, leakage-clean, genuinely-new, and they fire
  on every value-bug trap, but at 70B they add **+0.00 pp over blind reflexion** (the
  wrong-algorithm capability ceiling). Diagnostic of WHERE the cap is capability-bound; not a
  gain at this scale. (A stronger model is the untested axis; gate CLOSED.)
* **`naive_source` programs** — the canonical trap candidates the witness self-tests fire on.
  Diagnostic only.

## Historical artifacts (unchanged)

W120–W132 ICPC battlefield + atlas + generator/selector arsenal + the resistant-by-construction
battlefield unchanged. The W132 battlefield is the substrate the W133 witness instrument teaches
over.

## Dead directions (do NOT revive)

* Treating W132's +3.33 pp as a uniform "capability-bound" cap — W133 LOCALISES it: the
  complexity component is feedback-fixable (EW2), the value/algorithm component is
  capability-bound (EW1 +0.00).
* Selling the EW1 counterexample channel as a repair gain at 70B — it is REAL + clean but adds
  nothing over blind reflexion (machine-checked: C1 − B0 = +0.00, 0 rescues vs B0).
* Selling a single-mode gain as a multi-mode earn — the pre-committed ≥ 2-mode bar exists
  precisely to prevent a single-family blip (however large) from being oversold as a frontier
  earn; the §7a dev gate enforced it.

## Anti-patterns (REMAIN explicit anti-patterns)

Bounded-context / compaction / prose-summary / "cram less / truncate better" remain
anti-patterns, explicitly NOT pursued. W133 is an exact-oracle-feedback instrument +
held-out-bench milestone, not a context-compression milestone.

## Do not claim (W133)

* That W133 retired anything (it did not; W89 + W105 remain the only two).
* That the witness mechanism beats A1 on resistant code (it beats A1 ONLY via EW2 on the
  COMPLEXITY mode; EW1 adds 0 over blind reflexion; the gain is single-mode and earns no
  frontier rerun).
* That eval / frontier / Maverick were run ($0 eval, $0 frontier per the §7a dev gate; Maverick
  infra-down ⇒ bench on the W105 model Llama-3.3-70B).
* That the complexity-witness gain is deployable as-is (it uses the oracle's TIMING; a
  constraint-derived deployable approximation is the W134 lever).
* That the witness leaks the graded hidden tests (EW1 disjoint-from-graded; EW2 size/timing
  only; graded on a disjoint bank = generalisation test).
