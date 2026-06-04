# RESULTS — W136: machine-structured ALGORITHM-STATE TRACE instrument + root-cause of the "wrong-algorithm ceiling" (an I/O-FORMAT CONFOUND) + execution-grounded I/O repair

Executes the pre-committed `docs/RUNBOOK_W136.md` (locked before any NIM). COO-9 sibling (COO-61).
`coordpy.__version__ == "0.5.20"` · `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"` · no PyPI ·
`coordpy/__init__.py` untouched · `ultracode` OFF.

## One line

W135 proved the PROSE structure witness S4 ties the counterexample C1 and blind reflexion B0 (+0.00) on
the non-complexity dev field. W136 built the machine-structured algorithm-state TRACE (the dual
optimal/naive trajectory + the full 2-D subproblem-state DP table + a recurrence-derivation scaffold) and
found that it too cracks **0/3** traps — **then root-caused WHY, and the answer overturns the W132–W135
interpretation: the apparent "wrong-algorithm capability ceiling" on the generated WA/SE battlefield is an
I/O-FORMAT CONFOUND, not algorithm capability.** Llama-3.3-70B writes the CORRECT algorithms (2-D knapsack
DP, blocked-lattice DP, weighted-interval DP) but the W132 battlefield emits input WHITESPACE-FLATTENED
(grid rows / pairs / triples on one line; the reference reads `sys.stdin.read().split()`), while the model
assumes one-structure-per-line, so it crashes/misparses even the public samples. **Four independent
confirmations**, capped by: with STANDARD one-per-line I/O, A0 single-shot (no feedback, no mechanism)
**one-shots 3/3 traps**. The instrument is real; the field was confounded. **W89 (+5.56) + W105 (+7.00)
STAND** (they are on REAL HumanEval, untouched); W136 retires none and earns no frontier rerun (the
I/O-repair gains are parsing-only — §7b condition 4 correctly excludes them).

## Lane α — machine-structured trace instrument + fresh-seed corpus ($0 NIM) — **SUCCESS**

New modules (explicit-import only): `coordpy.algorithm_state_trace_v1` (the AT1 decision-path / AT2
subproblem-state / AT3 search-frontier trace slate + the content-addressed `AlgorithmStateTraceV1` typed
capsule [AT5] + the **full 2-D subproblem-state grid** `StateGridV1` assembled from black-box oracle calls
+ the family-aware `_typed_subinstances_v1` + the same-budget arms `run_trace_arm_v1` [T1/T2] + the
execution-grounded `run_io_grounded_trace_arm_v1` [T_IO] + the `trace_is_genuinely_new_vs_structure_v1`
guard) + `coordpy.algorithm_state_trace_corpus_v1` (the fresh-seed 136_0xx corpus). AT4
(invariant/internal-state) EXCLUDED for leakage (it would render the recurrence).

**Decisive construction finding.** The W135 generic 1-D slicer produced MALFORMED sub-instances on exactly
the 3 structured-input traps (knapsack `N W`+pairs, weighted-interval `N`+triples, lattice `R C`+grid), so
a generic output trace structurally could not reach them. W136's `_typed_subinstances_v1` (stride-detected
tuple/array prefixes + top-left `r'×c'` sub-grids, oracle-validated) + the 2-D `StateGridV1` give all 3
traps a genuine machine-structured trace — e.g. the knapsack capsule is the textbook items×capacity DP
table with the divergence cells marked:

```
subproblem-state table (optimal value for each (items, capacity)):
      capacity:      25       50       75      100
  items=1       120      120      120      120
  items=2       120      200      320      320
  items=3       120      200      320      440
your_approach diverges at (2,50): optimal 200 vs yours 120 ...
+ "Derive the recurrence that reproduces this table, then implement it."
```

**Corpus** `coordpy.algorithm_state_trace_corpus_v1` — 8 `wa_*` + 8 `se_*` × 5 FRESH seed-disjoint W136
seeds/split (136_0xx). Admitted **train 80 / dev 79 / eval 79 / frontier 78** (slice 30 = 14 SE + 16 WA);
all ≥ 36/36/36/30; held-out integrity TRUE. **LOCKED CIDs** ($0 build, predating all β NIM):
`corpus_cid ce1a6bc6…`, `eval_split_cid 13519353…`, `frontier_slice_cid 3f75b302…`. **Self-tests** ($0):
trace fires genuinely-new on **70/75** admitted train problems (the one thin family = `longest_common_subseq`,
string-pair input — a documented, machine-checkable limitation), **75/75** ref-silent (positive control);
reproducibility / deterministic typed sub-instances / leakage (no solver source in the capsule; sub-instances
disjoint from the graded bank) / negative control (complexity naive ⇒ NONE) all pass via the **18/18**
`tests/test_w136_algorithm_state_trace_v1.py` suite. ⇒ `W136-T-ALGORITHM-STATE-TRACE-INSTRUMENT-MINTABLE`.

## Lane β — held-out trace bench + learned-memory check + ROOT CAUSE

**Learned-memory/controller applicability (honest re-open, $0).** `differentiable_memory_substrate_v1` /
`composed_learned_memory_v1` / `live_composed_memory_training_v1` — **KILLED** (random-until-trained nets
benched only on synthetic `rng.standard_normal` recall data; the live one hard-requires GPU+transformers).
`constrained_policy_optimisation_v1` — **KILLED** (learned MLP + simulator reward). The ONLY honestly-usable
controller — the forward-only, weightless `controller_native_code_mechanism_v1` digest-router — IS exercised
as **T2**. Verdict: real traces do NOT rescue the trainable nets at inference (corroborates W124's
`TOO_SYNTHETIC_NOT_WARRANTED`).

**The decisive sequence (dev bench = the W135 dev problems; reuse A0=A1=B0=C1=S4=81.25% = 13/16, the SAME 3
capability-bound traps `se_lattice_paths_blocked` / `wa_knapsack_01` / `wa_weighted_interval_scheduling`).**

1. **T1 (machine-structured trace, incl. the full 2-D DP table + recurrence scaffold): cracks 0/3 traps.**
   The trace fires genuinely-new + leakage-clean on every trap (richer than S4 — it fires on knapsack
   where S4's flat ladder did NOT), yet the model produces no passing program. So *algorithm feedback in
   any form* — counterexample (C1), prose structure (S4), 1-D trace, and the full 2-D DP table — fails
   identically. ⇒ `W136-T-MACHINE-STRUCTURED-STATE-FEEDBACK-DOES-NOT-LIFT-APPARENT-CEILING`.

2. **Root-cause diagnostic (the W136 discovery).** Captured the model's ACTUAL code + the execution diff vs
   the oracle. The model writes CORRECT algorithms — a nested-loop 2-D knapsack DP, a grid DP, an
   end-sorted weighted-interval DP — but **misparses the input**: lattice reads `[list(input()) for _ in
   range(R)]` (R separate lines) and CRASHES on the space-separated grid `"4 4\n.... .... .... ...."`;
   weighted-interval prints `"Invalid input"` on the flattened triples; knapsack misreads the interleaved
   pairs. The W132 battlefield's reference uses `sys.stdin.read().split()` (format-agnostic), so it emits
   every structure whitespace-flattened onto one line, but the model assumes the STANDARD one-per-line
   competitive-programming layout.

3. **Four independent confirmations that I/O format — not the algorithm — is the trap discriminator:**
   * **Corpus-wide ($0):** the easy/hard split is EXACTLY the standard/non-standard I/O split — all 13
     easy families are `SINGLE_INT` / `SINGLE_ARRAY` (standard → model one-shots); all 3 traps are
     `GRID(flattened-rows)` / `MULTI_TUPLE(stride=2)` / `MULTI_TUPLE(stride=3)` (the only flattened families).
   * **Same-algorithm ($0):** the IDENTICAL correct 0/1-knapsack DP passes **7/7** secret cases with
     `sys.stdin.read().split()` parsing and fails **7/7** with `input()`-per-line parsing.
   * **Execution-grounded I/O repair (NIM, T_IO):** when the model's own code fails a valid public sample,
     prepend a generic (no-leakage) whitespace-parsing directive → cracks **3/3 traps** (both modes;
     controls pass); implied T_IO = 100 % vs 81.25 % (+18.75 pp).
   * **Standard-I/O A0 (NIM):** present the SAME problems with standard one-per-line I/O (expected outputs
     unchanged; `ref_still_passes_reformatted=True`) and run A0 single-shot (NO feedback, NO mechanism) →
     **one-shots 3/3 traps.**

   ⇒ `W136-T-WRONG-ALGORITHM-CEILING-ON-GENERATED-WA-SE-IS-IO-FORMAT-CONFOUND` +
   `W136-T-EXECUTION-GROUNDED-IO-REPAIR-CRACKS-ALL-TRAPS` +
   `W136-T-STANDARD-IO-A0-ONESHOTS-ALL-TRAPS`.

**Earn discipline.** T_IO clears the §7a/§7b numeric margins (+18.75 pp, ≥2 modes), but its rescues are
**parsing fixes**, which §7b condition 4 explicitly excludes ("a formatting-only or parsing-only gain is
NOT an earn"). So **the frontier rerun is NOT earned** — and rightly so: this is not algorithmic
superiority, it is the removal of a benchmark artifact. **$0 frontier NIM.** ⇒
`W136-L-GENERATED-WA-SE-BATTLEFIELD-IS-IO-CONFOUNDED-AND-ALGORITHM-LOW-HEADROOM`.

## Lane γ — research + stronger-model gate + frontier

Primary-source research (wired into the FORM, not summarized): the oracle-derived correct-algorithm STATE
trace is genuinely new vs Self-Debugging (arXiv:2304.05128) / LDB (arXiv:2402.16906) / Scratchpad
(arXiv:2112.00114), which feed the model its OWN buggy trace; NAR/CLRS (arXiv:2205.15659) + TransNAR
(arXiv:2406.09308) model correct-algorithm state only by TRAINING (justifies the T3 kill); NTM
(arXiv:1410.5401) is gradient-trained (corroborates the kill); CEGIS (arXiv:2502.07786) needs an SMT solver
and feeds a counterexample, not state; trace-repair (arXiv:2505.04441, longer-traces-hurt) ⇒ the bounded
≤6-row / ≤4×4-grid discipline. The decisive empirical lesson, however, is the one no algorithm-trace paper
anticipated here: the field's failures were **I/O parsing**, so algorithm-state feedback addressed the wrong
bug — an execution-grounded I/O check (à la LDB, but flagging a parse failure on a valid public input) is
what converts the traps.

Stronger-model gate re-derived `NO_CERTIFIABLE_STRONGER_MODEL`, **decision CID `258b6ed7` invariant**
(Qwen3-Coder-480B / DeepSeek-V4-Pro / Mistral-Small-4-119B-2603 / GLM-5 all primary-UNDISCLOSED). Gate
CLOSED; frontier target stays `meta/llama-3.3-70b-instruct`. **Frontier: NOT launched** (not earned; $0).
graphify START + END refreshed; the new `algorithm_state_trace_v1` is a REAL 1-hop bridge to
`exact_oracle_witness_v1`, `solution_structure_witness_v1`, `resistant_by_construction_battlefield_v1`,
`icpc_reflexion_bench_v1`, and (T2/T_IO) `executor_grounded_patcher_v1` / `controller_native_code_mechanism_v1`.

## Net — what changes, what stands

W136 LANDS a real machine-structured algorithm-state trace instrument (incl. the full 2-D DP state table)
and then, refusing to accept the apparent null, **root-caused the W132–W135 "wrong-algorithm capability
ceiling" and found it is an I/O-FORMAT CONFOUND** — quadruply confirmed, capped by A0 single-shot one-shotting
all 3 traps under standard I/O. This **REVISES** the interpretation of W133 (`EW1 +0.00 = wrong-algorithm
ceiling`), W135 (`structure-unliftable at 70B`), and W136's own T1 result: the feedback never helped because
it addressed the algorithm while the bug was I/O; the 70B model's algorithms were correct throughout. The
generated WA/SE battlefield is therefore (a) I/O-confounded and (b) once I/O-normalised, low-algorithm-headroom
(one-shot) — so it does NOT test hard algorithm capability and cannot demonstrate coordination superiority.

**W89 (+5.56) + W105 (+7.00) STAND as the only two retirements** — they are on REAL HumanEval(+) with
human-written I/O, entirely outside this confound. W136 retires none, earns no frontier rerun (parsing-only
gains), and demonstrates no coordination superiority — but it corrects a four-milestone misinterpretation
and lands the execution-grounded I/O-repair mechanism as a tested asset. `COO-9` stays lead. No version bump
(0.5.20 / coordpy.sdk.v3.43); no PyPI; `coordpy/__init__.py` untouched.

W136 ⇒ **W137** = rebuild the generated battlefield with STANDARD one-per-line I/O (or harden the harness to
present standard I/O) for a CLEAN algorithm-capability test; recognise the generated WA/SE field is
low-algorithm-headroom; pursue genuinely hard algorithm battlefields (the official contamination-resistant
ICPC/LiveCodeBench lines remain the real ones) / a primary-KNOWN stronger model when the gate opens.

## Carry-forwards added

* `W136-T-ALGORITHM-STATE-TRACE-INSTRUMENT-MINTABLE` — the AT1/AT2/AT3/AT5 trace slate + full 2-D
  subproblem-state DP table + family-aware typed sub-instances + fresh-seed corpus (80/79/79/78 ≥ floors;
  CIDs `ce1a6bc6…` / `13519353…` / `3f75b302…`); 70/75 train fire genuinely-new, 75/75 ref-silent; 18 tests; $0.
* `W136-T-MACHINE-STRUCTURED-STATE-FEEDBACK-DOES-NOT-LIFT-APPARENT-CEILING` — T1 (incl. the full 2-D DP table
  + recurrence scaffold) cracks 0/3 traps, identically to prose S4 and counterexample C1.
* `W136-T-WRONG-ALGORITHM-CEILING-ON-GENERATED-WA-SE-IS-IO-FORMAT-CONFOUND` — quadruple-confirmed: corpus
  easy/hard = standard/non-standard I/O; same DP 7/7 robust vs 7/7-fail per-line; I/O-repair cracks 3/3;
  A0+standard-I/O one-shots 3/3. The model's algorithms are correct; the battlefield's flattened I/O is the
  sole discriminator.
* `W136-T-EXECUTION-GROUNDED-IO-REPAIR-CRACKS-ALL-TRAPS` — the weightless, no-leakage `run_io_grounded_trace_arm_v1`
  (fires the whitespace-parse directive only when the model's own code fails a valid public sample) cracks
  3/3 traps (+18.75 pp, 2 modes); parsing-only ⇒ §7b cond 4 excludes it from a frontier earn.
* `W136-T-STANDARD-IO-A0-ONESHOTS-ALL-TRAPS` — with standard one-per-line I/O, A0 single-shot (no feedback)
  one-shots 3/3 traps ⇒ the "ceiling" vanishes with correct I/O.
* `W136-L-GENERATED-WA-SE-BATTLEFIELD-IS-IO-CONFOUNDED-AND-ALGORITHM-LOW-HEADROOM` — the generated battlefield
  does NOT test hard algorithm capability (I/O-confounded; I/O-fixed it is one-shot); the W133/W135
  "capability ceiling" conclusions on this field are CONFOUNDED and revised; W89+W105 (real HumanEval) STAND.
