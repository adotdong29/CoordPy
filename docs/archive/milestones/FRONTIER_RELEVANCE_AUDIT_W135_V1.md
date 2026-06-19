# FRONTIER RELEVANCE AUDIT — W135 (solution-structure witness) — V1

Supplement to the W133/W134 audits; all prior classifications remain in force. This audit classifies
every W135 artifact as **active frontier arsenal**, **useful baseline-only**, **dead direction**, or
**anti-pattern**, so later milestones reuse what is load-bearing and avoid what is not.

## Active frontier arsenal

* **`coordpy.solution_structure_witness_v1`** — the oracle-derived solution-structure witness slate
  (SW1 greedy-failure certificate / SW2 optimal-substructure ladder / SW3 search-frontier exact-count
  / SW4 structure-to-rewrite controller). The FIRST module to feed oracle-derived ATTRIBUTED,
  minimal, sub-value-ladder structure (not just a counterexample) to a frozen LLM at inference, with
  a machine-checked genuinely-new-vs-EW1 guard + a leakage guard (disjoint probe + sub-instances, no
  solver source, no recurrence/state). A reusable, deterministic, content-addressed teaching asset on
  the owned battlefield — independent of the W135 β verdict.
* **`coordpy.noncomplexity_structure_corpus_v1`** — the dedicated NON-complexity (WRONG_ALGORITHM +
  SEARCH_ENUM) seed-disjoint train/dev/eval/frontier corpus (16 families × 5 seeds/split; locked
  CIDs; held-out integrity). The push-button battlefield for any future wrong-algorithm / search
  mechanism (eval + frontier slices locked + cached).
* **The structure-witness bench** (`scripts/run_w135_structure_witness_bench_v1.py`) — same-budget
  A0/A1/B0/C1/D0/S1–S4 with the C1-beating dev/earn gates + the structural-rescue audit, scored by
  the verbatim W108 evaluator. The reusable harness for the "structure vs counterexample" question.
* **The C1-as-flat-baseline comparison + the ≥+5-pp-over-C1 earn rule** — the methodological lever
  that makes "structure beats a bare counterexample" falsifiable (W133 left C1 as the +0.00 anchor;
  W135 makes beating it the bar).

## Useful baseline-only

* **C1 (EW1 counterexample, `exact_oracle_witness_v1.ARM_C1_COUNTEREXAMPLE`)** — retained as the FLAT
  baseline (W133: +0.00 over B0 on these modes); not a frontier mechanism itself, but the
  indispensable control the structure arms must beat.
* **D0 (W134 deployable complexity witness)** — retained as the NEGATIVE control on non-complexity
  (expected ≈ B0; the complexity witness almost never fires on output-mismatch traps) — the clean
  structure-vs-complexity dissociation.
* **B0 blind reflexion** — the validated W120/W132 stack; the same-budget anchor.

## Dead directions

* **HIDDEN_EDGE_STATE_MISS structure witness (SW4-invariant)** — NOT built: a HIDDEN_EDGE witness
  collapses to a concrete corner-case counterexample = exactly the EW1/C1 channel W133 proved flat,
  with no DISTINCT structure to render. Excluded as a main target (documented in RUNBOOK §2), not
  silently dropped.
* **Oracle-derived structure to break the wrong-algorithm ceiling at 70B (FINALISED — DEAD at 70B).**
  The held-out dev bench is flat: A0 = A1 = B0 = C1 = S4 = 81.25 %; S4 − C1 = S4 − B0 = +0.00 pp; 0
  rescues over C1 (0 modes / 0 families); the same 3 problems are capability-bound for every arm. The
  oracle-derived attributed structure witness (greedy-failure cert + optimal-substructure ladder +
  search-frontier exact counts) adds NOTHING over a bare counterexample or blind reflexion on WA/SE at
  70B ⇒ the wrong-algorithm ceiling is CAPABILITY-bound, not feedback-form-bound. The
  structure-feedback line at 70B is dead; the remaining levers are a code-competent local model
  (better GENERATION), a primary-KNOWN stronger model when the §3 gate opens, the Maverick
  cross-scale push-button, or a genuinely different mechanism axis. (The instrument + corpus STAND as
  reusable assets — the DEAD classification is of the 70B-feedback hypothesis, not the tooling.)

## Anti-patterns (unchanged, reinforced)

* **Bounded-context / compaction / prose-summary / "cram less / truncate better"** remain explicit
  anti-patterns — NOT pursued in W135. The structure witness is the opposite move: it adds a precise,
  oracle-derived, structurally-richer feedback object, not a compressed one.
* **Answer leakage sold as a mechanism** — a structure witness that disclosed the recurrence/state,
  or revealed an optimal value of a graded hidden case, would be leakage, not coordination; the
  no-leakage rule (disjoint probe + sub-instances, no solver source, no recurrence, grading on the
  disjoint hidden bank) is the guard, and any witness that collapses into leakage is killed.
* **Dirty synthetic benchmark / official-task paraphrase sold as resistant** — the corpus is minted
  from gate-validated W132 templates with within-seed novelty filtering; it is NOT an official-task
  paraphrase and NOT a hand-tuned toy.

## Stronger-model gate (W135 γ recheck)

Re-derived `NO_CERTIFIABLE_STRONGER_MODEL`, decision CID `258b6ed7` invariant `{KNOWN:1, UNKNOWN:4}`
(Qwen3-Coder-480B / DeepSeek-V4-Pro / Mistral-Small-4-119B-2603 / GLM-5 all primary-UNDISCLOSED;
the new entrant MiniMax-M3 [Jun-2026] has no published card/cutoff). Frontier target stays
`meta/llama-3.3-70b-instruct` (primary-KNOWN Dec-2023). No 405B.

Anchors: `docs/RUNBOOK_W135.md`; `docs/RESULTS_W135_STRUCTURE_WITNESS_BENCH_V1.md` (filled on
completion); `results/w135/`.
