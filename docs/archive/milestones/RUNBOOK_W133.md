# RUNBOOK W133 — exact-oracle witness curriculum + held-out witness-guided mechanism bench + conditional frontier rerun

**Status: LOCKED PRE-REGISTRATION.** Written BEFORE any W133 NIM (calibration, dev, eval,
or frontier) call and BEFORE any β result is interpreted. The Lane-α curriculum CONSTRUCTION
and the witness SELF-TESTS (§ 1, § 4, § 5) are **$0-NIM** (deterministic minting + the
answer-key / oracle subprocess; no model inference) and are allowed before this lock — the
W129/W130/W131/W132 $0-construction discipline. They are emitted by
`scripts/run_w133_build_curriculum_and_witness_selftest_v1.py` to
`results/w133/curriculum/`. Fill `docs/RESULTS_W133_*` ONLY from emitted verdict JSON
(`feedback_never_prewrite_results_before_data`). The pre-committed rules below are the branch
authority, not any prior or hope.

`ultracode` OFF. W133 is a bounded exact-oracle-feedback / conditional-frontier-rerun
milestone, not a repo-wide dynamic-workflow job. It turns ON only if the work unexpectedly
expands into a genuine dynamic-workflow problem (multiple witness families all earn live
reruns at once / repo-wide oracle-curriculum integration / broad multi-surface external
verification at once) — and only after an explicit mode-change note.

Stable boundary (unchanged, asserted in tests): `coordpy.__version__ == "0.5.20"`,
`coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`, no PyPI publish, `coordpy/__init__.py`
untouched, advanced work explicit-import only.

---

## § 0 — Why W133 is NOT a supply hunt / selector retry / prompt-only generator lap

W132 closed the "wrong-test" escape: on a CoordPy-minted resistant-by-construction field, at
the exact W105 retirement model (Llama-3.3-70B), the same-budget mechanism got B − A1 =
+3.33 pp (FAIL; MLB sub-floor; exactly ONE complexity rescue, six capability-bound traps
unsolved by all arms). So the battlefield is no longer the blocker, official supply is no
longer the blocker, disclosure is no longer the blocker for the existence of a resistant
field. **The next honest lever is richer SUPERVISION from the field we own, not another
blind hidden-test reject bit.** W132's reflexion feedback (B) is a BLIND reject bit: the model
sees only the judge verdict, the executor stderr, and the all-pass public-sample results — it
is never told WHICH input breaks it or WHAT the answer should be. W133 turns the battlefield
into a TEACHER: because every minted problem ships an exact `ref_source` oracle, we can
compute an **exact-oracle witness** — a concrete minimal counterexample (input + correct
output + the candidate's wrong output), or a complexity/timing witness for too-slow
programs — as the sanctioned feedback object, WITHOUT leaking the graded hidden cases.
Bounded-context / compaction / summarization remain explicit anti-patterns, NOT the path.

This is the inference-time CEGIS / self-debug / execution-feedback paradigm (see § 9), with a
sharp pre-committed interpretation either way: if the witness EARNS, the W132 cap was
**feedback-richness-bound, not capability-bound** (the model CAN fix these problems given a
good counterexample) ⇒ the mechanism story reopens and a frontier rerun is earned; if it
FAILS, the cap is **capability-bound** (the model cannot use even a perfect oracle
counterexample) ⇒ the strongest possible statement of the bounded ceiling.

---

## § 1 — α / β / γ branch logic (pre-committed)

* **Lane α — exact-oracle witness instrument + train/dev/eval curriculum (MAIN construction,
  $0).** Build `coordpy.exact_oracle_witness_v1` (the witness slate + the same-budget
  witness arm) + `coordpy.witness_curriculum_corpus_v1` (the seed-disjoint train/dev/eval
  curriculum over `RBC_SLATE_V1`). Run ALL § 5 self-tests + the § 3 no-leakage checks + the
  W132 regression fixtures. Lane α SUCCEEDS iff it admits **≥ 32 clean problems per split
  (≥ 96 total)**, the splits are content- and graded-secret-input disjoint, and the witness
  fires (leakage-clean, genuinely-new, deterministic) on every admitted problem's
  `naive_source` AND on all six W132 traps. If it cannot hit 96 honestly, land the instrument
  anyway and make the blocker machine-checkable.
* **Lane β — held-out witness-guided mechanism bench (MAIN spend).** Lock the arms (§ 6)
  BEFORE any NIM. Run the full arm slate on the **dev** split (the go/no-go signal, § 7a);
  only if the dev gate clears, run the lead arm + baselines on the **LOCKED eval** split
  (the earn measurement, § 7b). Frontier rerun (§ 8) only if eval earns.
* **Lane γ — primary-source research + frontier gate / graphify / truth (MANDATORY, $0
  except the earned frontier rerun).** Use primary sources to confirm the mechanism is
  executable-here (§ 9); re-derive the stronger-model cutoff gate (§ 8); refresh graphify
  START + END (§ 10); register carry-forward (§ 11); land executable code, not docs only.

Branch order: α (build + self-tests, $0) → γ-gate + research ($0) → β-dev (NIM) →
β-dev-gate → β-eval (NIM iff dev gate clears) → β-eval-earn → γ-frontier (NIM iff earned) →
γ-graphify/truth.

---

## § 2 — Frontier-target rule (Lane γ, LOCKED PRE-SPEND)

Machine-checked this session (2026-06-02, same as W132 §8d): `GET /v1/models` 200 in 0.18 s
lists all targets; **`meta/llama-3.3-70b-instruct` is reachable and fast (~0.38 s for an
8-token chat)**; **`meta/llama-4-maverick-17b-128e-instruct` is STILL infra-down (60 s
timeout, 0 bytes)**. Therefore:

* the **default frontier target is `meta/llama-3.3-70b-instruct`** — the EXACT model that
  earned the W105 retirement (primary-KNOWN cutoff ~Dec-2023 per the Llama model card; the
  2026-06-02 mint date strictly post-dates it ⇒ resistant by date AND construction), and the
  W132 anchor model, so W133 is the most on-target transfer test;
* **Maverick is the optional push-button CROSS-SCALE check** when its deployment recovers
  (`--model meta/llama-4-maverick-17b-128e-instruct` on the same slice); **W133 does NOT block
  on Maverick** (operator §γ.7);
* no stronger-than-Maverick model is frontier-eligible (§ 8 gate CLOSED).

---

## § 3 — No-leakage / witness-API rule (LOCKED, enforced in code)

1. The model under test sees ONLY `statement` + the PUBLIC `samples` + the **witness block**.
   `ref_source` / `naive_source` / `brute_source` and the graded `secret_cases` are NEVER
   placed in any model-facing path. The witness block carries an oracle OUTPUT (`expected`),
   never the oracle PROGRAM (`WitnessV1.to_prompt_block`).
2. Witness probe inputs are drawn from a FRESH `witness_seed = 999133` stream via the
   template's OWN generators. The leakage rule is by witness type, because they reveal
   different things: a **COUNTEREXAMPLE (EW1/EW3/EW4)** reveals `(input, expected = ref(input),
   observed)` — so its small probe input MUST be byte-disjoint from every graded secret case
   (no teaching-to-the-test); any small probe colliding with a secret input is dropped, and
   `leakage_clean` asserts the reported counterexample input is not a graded case. A
   **COMPLEXITY witness (EW2)** reveals ONLY a timing fact + the input SIZE — never the input
   bytes and never an expected output (`to_prompt_block`) — so its big stress input may
   coincide with a graded worst-case without leaking any answer; EW2 is therefore structurally
   `leakage_clean` (it discloses no graded `(input, output)` pair). The unifying invariant:
   **no witness ever discloses a graded secret case's answer to the model.**
3. Grading is the audited `grade_on_secret_v1` / `grade_icpc_candidate_case_v1` on the
   problem's `secret_cases` ONLY — a DISJOINT hidden bank the model never saw. The witness
   therefore tests **generalisation, not memorisation** (the literature-endorsed guard
   against overfitting-to-the-shown-test; § 9). A witness arm that special-cases the shown
   input still fails the disjoint hidden bank.
4. Every witness is reproducible from the content-addressed witness API (fixed `witness_seed`
   + deterministic probe builder; `WitnessProbeSetV1.cid`).

---

## § 4 — Witness slate (LOCKED, § 5-validated before any NIM)

* **EW1 — minimal failing counterexample** — the smallest fresh public-style input `X` where
  the candidate disagrees with the reference, reported as `(X, expected = ref(X),
  observed = candidate(X))` with a deterministic shrink trace (`shrink_steps`). For
  WRONG_ALGORITHM (a named-but-wrong technique) and the generic value-bug case.
* **EW2 — complexity witness** — for a candidate that is value-correct but asymptotically too
  slow (COMPLEXITY_BLIND), a size-growth witness: "did NOT finish within `T_probe = 2.0 s` on
  an input of size N≈`big_n` while the reference finishes in `ref_runtime` s — the algorithm
  is too slow." No hidden case is leaked; only the structured timing fact from the public scale.
* **EW3 — hidden-edge / invariant witness** — EW1 specialised to the HIDDEN_EDGE probe
  distribution (the family's own corner-case generators at the fresh witness seed), isolating
  the overlooked corner (wrap / overlap / inclusivity / tie / sign).
* **EW4 — search-enum witness** — EW1 with the small-n exact oracle (the family's small-n
  exhaustive cross-check) producing a concrete disagreement for SEARCH_ENUM miscounts.

Implementation: EW1/EW3/EW4 are one counterexample search over the family's own fresh-seed
generator distribution (which already produces the corner / small-n traps); EW2 is the
timing witness. The C-arm controller routes from OBSERVED behaviour (value disagreement →
counterexample; else value-correct-but-slow → complexity), never from a leaked mode label.

---

## § 5 — Oracle / witness self-test rule (LOCKED; $0; gate before NIM)

The witness instrument is admitted iff, on EVERY admitted curriculum problem (all 3 splits):

* **fires** — `select_witness_v1` (C3) produces a `COUNTEREXAMPLE` or `COMPLEXITY` witness on
  the canonical `naive_source` (the admissible-wrong / too-slow trap = the failure the model
  reproduces): the witness genuinely **flips the candidate while the reference stays correct**;
* **minimal / shrunk** — counterexamples carry a deterministic shrink trace; the reported
  input is the smallest failing one found;
* **leakage-clean** — the reported probe input is not a graded secret case (§ 3);
* **genuinely-new** — `witness_is_genuinely_new_v1` confirms the witness carries an input that
  is NOT a public sample AND an oracle output (so it is not "judge bit plus more words");
* **deterministic** — re-building the probe set at the same `witness_seed` yields an identical
  `WitnessProbeSetV1.cid`;
* **family-balance** — reported per split (mode histogram) and per W132 trap.

§ 5 also REQUIRES the W132 regression fixtures to pass: the witness fires (clean, new) on each
of the **six** W132 capability-bound traps (`cb_distinct_in_windows`, `cb_pairs_sum_eq_t`,
`cb_pairs_sum_le_t`, `he_interval_union_length`, `se_lattice_paths_blocked`, `wa_knapsack_01`)
and the one B-unique complexity rescue (`cb_pairs_absdiff_le_d`) on the W132 anchor (seed 132).

---

## § 6 — Same-budget evaluation rule (LOCKED before NIM)

Arms (the *already-validated* W88/W89 three-arm same-budget mechanism + the W133 witness
arms), all at `K = 5`, 1 seed per split (`133101` dev / `133102` eval / `132001` frontier),
`sampling_temperature = 0.7`, `max_tokens_per_call = 1536`, `executor_timeout_s = 8.0`, same
model on every arm:

* **A0** — single-shot (1 call, temperature 0).
* **A1** — same-budget self-consistency: K i.i.d. samples, pass iff ANY passes (oracle pass@K).
* **B0** — the W132/W120 sequential reflexion (K attempts; feedback = judge verdict bit +
  stderr tail + public-sample results). **The blind baseline.**
* **C1** — witness reflexion, EW1 only (counterexample feedback).
* **C2** — witness reflexion, EW2 only (complexity feedback).
* **C3** — witness reflexion, controller (EW1 else EW2; the designed LEAD).
* **C4** — constrained witness-action policy (`constrained_policy_optimisation_v1`):
  **fired ONLY if** the dev bench yields ≥ 30 witness-invoked reflexion decision points with
  non-degenerate outcome variance (a real labelled action dataset). Otherwise NOT fired
  (hope-funded; the W129 β1 discipline). On n≈33 with few invoked failures this is expected
  NOT to fire; if not fired, it is recorded as `C4_NOT_FIRED_DATA_SUPPORT`.

**Same-budget is exact.** Every B/C arm has the IDENTICAL structure to B0 (attempt-0 = the
standard initial prompt; K attempts; one model call per attempt; no early stop, no selective
retry); the ONLY difference is the between-attempt feedback object. The witness GENERATION is
$0 (the oracle + executor, not model calls), so no calls are added; if a witness step were
ever model-facing it would be reallocated from the K budget (it is not). The witness block is
a strict SUPERSET of B0's feedback (judge bit + stderr + public samples + the witness), so any
C − B0 gain is attributable to the witness object. Per problem: 1 + 5 + 5 + 5 + 5 + 5 = 26
calls (dev full slate); 1 + 5 + 5 + 5 = 16 calls (eval / frontier lead + baselines). Scored by
the verbatim W108 `_mlb_rates` + `_evaluate_phase2_gates` (the SAME code that scored
W89/W105/W120/W132), with each witness arm placed in the "B" slot so "C − A1" is computed
byte-identically to "B − A1".

---

## § 7 — Dev gate + the held-out earn rule (LOCKED before NIM)

**Lead arm selection (pre-committed):** the lead witness arm = `argmax` over {C1, C2, C3} of
`(arm − B0)` on the dev split among arms whose dev rescues span ≥ 2 distinct modes; ties
broken by `(arm − A1)`, then by arm order C3 > C1 > C2 (C3 is the designed lead).

**§ 7a — Dev gate (the eval-spend trigger).** Eval spend is EARNED iff on the dev split the
lead witness arm: (1) beats B0 by ≥ +3.33 pp, AND (2) its dev rescues (problems the lead
passes that B0 fails) span ≥ 2 distinct failure modes. If the dev gate FAILS ⇒ register the
witness-feedback cap, **$0 eval, $0 frontier** (do not fund a dead mechanism). The dev bench
is where we are allowed to look; the mechanism is NOT tuned on it (it is fully pre-committed
here), so dev is a clean go/no-go, not a design surface.

**§ 7b — Held-out earn rule (the frontier-rerun trigger; operator-locked).** A witness arm
earns the frontier rerun iff, on the LOCKED held-out **eval** slice (CID fixed before eval
spend), the lead arm:

* beats **A1 by ≥ +5.00 pp**, AND
* beats **B0 by > +3.33 pp**, AND
* the eval gains (problems the lead passes that A1 fails) span **≥ 2 distinct failure modes**,
* and the gain is NOT driven only by trivial formatting / parsing repair (a per-rescue audit
  classifies each as ALGORITHMIC vs FORMATTING; a formatting-only gain does NOT earn).

A weak or single-family blip is NOT an earn. The eval slice is never used for mechanism design.

---

## § 8 — Frontier rerun + stronger-model gate (LOCKED)

**§ 8a — Frontier rerun (earned iff § 7b passes).** Run the cheapest honest rerun first on the
**W132 LOCKED core 30-slice** (re-mint seed 132 → `select_core_slice_v1` → assert
`core_slice_cid` prefix `f6a2ebed`; seed 132001), arms = A0 / A1 / B0 / lead-witness-arm, SAME
exact-oracle grader (`grade_on_secret_v1`), pass-fail-only, scored by the verbatim W108
evaluator. This confirms whether the witness mechanism lifts the W132 anchor's +3.33 pp.

**§ 8b — Frontier outcome (pre-committed; filled ONLY from JSON):**
* `WITNESS_PASS_MECHANISM_DRIVEN` (lead − A1 ≥ +5 pp AND MLB-1 ≥ 33% AND MLB-2 ≥ 33% on the
  frontier slice) ⇒ a clean witness-driven superiority, single seed ⇒ **W134 = operator-
  greenlit multi-seed confirmation toward W89/W105 retirement-grade** (a retirement is
  registered ONLY on a clean multi-seed pass, NEVER on a single seed).
* `WITNESS_PASS_NON_MECHANISM_DRIVEN` (margin present, MLB sub-gate fails) ⇒ register bounded.
* `WITNESS_FAIL` ⇒ the witness mechanism does not transfer to the anchor ⇒ register the cap.

**§ 8c — Stronger-model gate (re-derived, $0).** `decide_certification_v1` re-derives
`NO_CERTIFIABLE_STRONGER_MODEL`, decision CID `258b6ed7…` invariant, {KNOWN:1, UNKNOWN:4}
(Maverick KNOWN Aug-2024 already-settled; Qwen3-Coder-480B / DeepSeek-V4-pro /
Mistral-Small-4-119B-2603 / GLM-5 primary-UNDISCLOSED — re-verified from primary sources §9).
No 405B run unless reachability changes and a pre-committed gate clears. W123→W132 caps stay
closed unless new evidence genuinely changes them.

---

## § 9 — Primary-source research rule (Lane γ, LOCKED)

Restrict to primary sources (arXiv / OpenReview / official venue pages / official model cards).
Use the literature ONLY if it changes the mechanism (no literature-summary-as-output). The
mechanism class — concrete counterexample / execution-feedback / oracle-grounded repair at a
fixed call budget — is confirmed EXECUTABLE-HERE (inference-time oracle + executor; NO
training/RL) by: CEGIS (Jha & Seshia, SYNT 2014, arXiv:1407.5397 — counterexample CHOICE
affects convergence; motivates the shrink-to-minimal); counterexample-guided repair (Morvalho
et al., AAAI 2025); Self-Debugging (Chen et al., ICLR 2024, arXiv:2304.05128 — feedback
richness "matches or beats >10× sampling", i.e. a same-budget win); Reflexion (Shinn et al.,
NeurIPS 2023 — = the B0 baseline); LDB (ACL 2024, arXiv:2402.16906); output-masked self-
inference (arXiv:2309.16120 — show input + correct output Z); PBT shortest-counterexample
(arXiv:2506.18315 — minimal input is the more informative repair signal); Self-Edit (ACL 2023).
Two documented LIMITS are carried as honest caveats: (a) **overfitting-to-the-shown-test**
(Ahmed et al., arXiv:2511.16858; UTGen/UTDebug, COLM 2025) ⇒ the § 3 disjoint-hidden-bank
grading is the literature-endorsed guard; (b) the **wrong-algorithm capability ceiling**
(oracle outputs help wrong-output / edge / complexity bugs but do not by themselves repair a
fundamentally wrong algorithm) ⇒ gains are PREDICTED to concentrate on COMPLEXITY +
HIDDEN_EDGE and be bounded on WRONG_ALGORITHM (consistent with W129/W130). Self-play / trained
test-generators (e.g. UTGen generator, arXiv:2502.14948, arXiv:2502.01619) are
TRAINING_DEPENDENT and are NOT used.

---

## § 10 — graphify deliverables (LOCKED)

* Refresh `graphify update .` at START (built from HEAD `66c0b38`; captured) and END (record
  END HEAD after material changes).
* `graphify explain` on the new `exact_oracle_witness_v1` + `witness_curriculum_corpus_v1`
  and the mined arsenal (`resistant_by_construction_battlefield_v1`,
  `resistant_by_construction_slate_v1`, `icpc_reflexion_bench_v1`,
  `coordpy_icpc_battlefield_v1`, `public_signal_selection_oracle_v1`,
  `stronger_generator_slate_v1`, `generator_failure_atlas_v1`,
  `constrained_policy_optimisation_v1`). `graphify path` + `graphify query` secondary.
* The new `exact_oracle_witness_v1` must create the FIRST semantic bridge from an
  oracle-WITNESS generator to BOTH the minted battlefield (`resistant_by_construction_
  battlefield_v1`, `MintedProblemV1`) AND the validated bench scaffolds
  (`icpc_reflexion_bench_v1`) AND the audited grader (`judge_icpc_output_v1`). The END graph
  must show these edges.

---

## § 11 — Carry-forward registration (LOCKED shape; filled ONLY from JSON)

* **W89 (+5.56) + W105 (+7.00)** remain the only two confirmed retirements unless a later
  clean multi-seed `WITNESS_PASS_MECHANISM_DRIVEN` is earned. W133 retires none on a single seed.
* Always add `W133-T-EXACT-ORACLE-WITNESS-INSTRUMENT-MINTABLE` (the instrument + the
  ≥96-instance seed-disjoint curriculum exist, witnesses fire leakage-clean on all traps —
  the construction advance regardless of the bench outcome).
* On dev-gate FAIL or eval-earn FAIL: register `W133-L-WITNESS-FEEDBACK-CAP` — even exact
  oracle counterexamples + complexity witnesses at the same budget do not lift the same-budget
  mechanism over A1 on the held-out resistant-by-construction field ⇒ the cap is
  CAPABILITY-bound, not feedback-bound; the bounded contamination-EXPOSED-HumanEval-family-
  at-70B ceiling STANDS and is STRENGTHENED (the oracle gave the answer and the model could not
  use it). Record the per-mode lead−B0 / lead−A1 distribution + the rescue audit.
* On eval EARN but frontier FAIL: register the held-out-only witness gain + the frontier cap.
* On frontier `WITNESS_PASS_MECHANISM_DRIVEN`: register the single-seed clean signal + the
  W134 multi-seed decision (NOT a retirement by itself); the cap is FEEDBACK-bound (a
  deployable witness-approximation becomes the next lever).
* Carry forward W123→W132 caps unchanged; decision CID `258b6ed7…` invariant.

---

## § 12 — W134 branch logic (pre-committed)

* Dev-gate or eval-earn `FAIL` ⇒ W134 = accept the registered `W133-L-WITNESS-FEEDBACK-CAP`
  (the witness instrument + curriculum STAND as reusable assets); the remaining levers are a
  genuinely different axis, the Maverick CROSS-SCALE push-button (same slice when it recovers),
  or a primary-KNOWN stronger-than-Maverick model when the § 8c gate opens. Bounded-context /
  compaction remain anti-patterns.
* eval EARN + frontier `WITNESS_FAIL` ⇒ W134 = strengthen the witness mechanism on the anchor
  or accept the held-out-only bounded gain.
* frontier `WITNESS_PASS_MECHANISM_DRIVEN` (single seed) ⇒ W134 = operator-greenlit multi-seed
  same-budget confirmation on the W132 core slice (≥ 3 seeds) + the Maverick cross-scale check;
  retire iff a clean multi-seed mechanism-driven +5.00 pp margin (NEVER on a single seed). This
  would also motivate a DEPLOYABLE witness-approximation (property-based / differential test
  oracle) as the genuinely-new deployable mechanism.
* `COO-9` stays the lead path unless the evidence genuinely forces a different code-line move.
