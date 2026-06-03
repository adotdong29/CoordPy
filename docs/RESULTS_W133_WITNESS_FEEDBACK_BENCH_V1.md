# RESULTS W133 — exact-oracle witness curriculum + held-out witness-guided mechanism bench (Llama-3.3-70B) — a REAL but SINGLE-MODE complexity-witness gain; frontier rerun NOT earned

**One line.** W132 closed the "wrong-test" escape (on the minted resistant-by-construction
field, at the W105 retirement model, the blind-reflexion mechanism got only B − A1 = +3.33 pp)
and left one honest lever un-pulled: **its reflexion feedback is a BLIND hidden-test reject
bit.** W133 turned the battlefield we OWN into a TEACHER — an **exact-oracle witness**
(a minimal counterexample, or a complexity/timing witness) computed from each problem's
`ref_source`, served as the *sanctioned* between-attempt feedback object WITHOUT leaking the
graded hidden cases — and ran the held-out generated-family bench. The result is sharp and
matches the literature's two documented limits exactly: **the complexity witness (EW2) is a
REAL, load-bearing mechanism gain** (on the held-out dev split the controller arm C3 beats the
blind W132 stack B0 by **+6.06 pp** and A1 by **+12.12 pp**, MLB-2 = 60 %, cracking the O(N²)
`COMPLEXITY_BLIND` traps that B0 and a counterexample-only arm cannot), **but the gain is
ENTIRELY single-mode**: the counterexample witness (EW1) adds **+0.00 pp over blind reflexion**
(zero rescues over B0) on the HIDDEN_EDGE / WRONG_ALGORITHM / SEARCH_ENUM modes — the
wrong-algorithm capability ceiling. Because the gain does not span ≥ 2 failure modes, the
**pre-committed dev gate FAILS** ⇒ **$0 eval, $0 frontier** (the frontier rerun is genuinely
not earned). **W89 (+5.56) + W105 (+7.00) remain the only two retirements; W133 retires none.**
The advance is a **localisation of the W132 cap**: its +3.33 pp is part feedback-fixable
(complexity, liftable by an oracle-grounded timing witness) and part capability-bound
(value/algorithm, unliftable even with a perfect counterexample).

All numbers below are filled ONLY from emitted verdict JSON (`results/w133/**`).

---

## Lane α — exact-oracle witness instrument + train/dev/eval curriculum ($0 NIM) — **SUCCESS**

New modules (explicit-import only): `coordpy.exact_oracle_witness_v1` (the EW1–EW4 witness
slate, the deterministic content-addressed probe API, the same-budget witness arm
`run_witness_arm_v1`, the `witness_is_genuinely_new_v1` fake-different test) +
`coordpy.witness_curriculum_corpus_v1` (the seed-disjoint train/dev/eval curriculum over the
W132 `RBC_SLATE_V1`).

**Curriculum build (`results/w133/curriculum/curriculum_build_v1.json`; re-derivable
byte-identically):**

| field | value |
|---|---|
| admitted train / dev / eval | **33 / 33 / 33 = 99 (≥ 96)** ✓ |
| seeds | 133001 / 133002 / 133003 (distinct from the W132 mint seed 132 ⇒ fresh instances) |
| family balance (each split) | COMPLEXITY_BLIND 9 / HIDDEN_EDGE 8 / SEARCH_ENUM 8 / WRONG_ALGORITHM 8 |
| `curriculum_cid` | `e5e44a4393d7a241…` |
| eval split CID (LOCKED before any eval spend) | `88b9b79a7389711b…` |
| held-out integrity | **content-CID pairwise-disjoint ✓ + seed-disjoint ✓** |
| witness self-tests (per split) | fired **33/33**, leakage-clean **33/33**, genuinely-new **33/33**, deterministic **33/33** |
| W132 regression fixtures | the **6/6** capability-bound traps + the 1 B-unique rescue all produce a firing, leakage-clean, genuinely-new witness on the W132 anchor (seed 132) |

**Held-out integrity is whole-problem content-CID disjointness + seed disjointness** (all
pairwise content-CID overlaps = 0). The residual per-secret-INPUT overlaps across splits are
ONLY seed-independent canonical boundary/stress constructions
(`results/w133/curriculum/secret_overlap_characterization_v1.json`): the O(N²) decreasing
worst-case stress (`cb_nearest_smaller_left`), the canonical mismatched bracket `[(])`
(`he_balanced_brackets`), and the smallest-n boundary `2` (five SEARCH_ENUM families). These
carry no memorisable answer signal — the model never sees secret cases, the splits are graded
by independent runs, and the mechanism is pre-committed (never tuned on any split).

**No-leakage (LOCKED, enforced + tested):** the model sees ONLY the statement + public samples
+ the witness block. `ref_source`/`naive_source`/`brute_source` are NEVER in any model-facing
path. A **COUNTEREXAMPLE** (EW1) discloses `(input, expected = ref(input), observed)` so its
probe input is strictly byte-disjoint from the graded secret cases (no teaching-to-the-test); a
**COMPLEXITY** witness (EW2) discloses ONLY a timing fact + input SIZE (never the input bytes
or an expected output), so it is structurally leakage-clean even when the stress input
coincides with a graded worst-case. Grading is the audited `grade_on_secret_v1` on a DISJOINT
hidden bank the model never saw — so the witness tests **generalisation, not memorisation** (the
literature-endorsed guard against overfitting-to-the-shown-test). 9/9 unit tests pass, incl. a
**positive control** (the correct `ref_source` yields NO witness) and a same-budget assertion
(the witness arm makes exactly K model calls).

---

## Lane β — held-out witness-guided mechanism bench (Llama-3.3-70B; Maverick infra-down) — **REAL but SINGLE-MODE; dev gate FAILS**

**Frontier target (RUNBOOK §2, machine-checked this session 2026-06-02, same as W132 §8d):**
`meta/llama-3.3-70b-instruct` reachable + fast (~0.38 s/8-token chat); **Maverick still
infra-down** (60 s timeout, 0 bytes). Target = Llama-3.3-70B (the W105 retirement model, KNOWN
cutoff ~Dec-2023; the minted field is resistant by date AND construction). Maverick stays the
optional push-button CROSS-SCALE check; W133 does not block on it.

**Same-budget arms (RUNBOOK §6):** A0 / A1 / B0 (the *already-validated* W120/W132 blind
reflexion) + the witness arms C1 (EW1 counterexample) / C2 (EW2 complexity) / C3 (controller:
EW1 else EW2). Every B/C arm is byte-identical to B0 (attempt-0 = the standard initial prompt;
K = 5 attempts; one model call per attempt; no early stop) — the ONLY change is the
between-attempt feedback object (the witness is a strict SUPERSET of B0's feedback). Witness
GENERATION is $0 (oracle + executor, not model calls). Scored by the verbatim W108 evaluator
with each witness arm in the "B" slot, so "C − A1" is computed byte-identically to "B − A1".

**Executed dev bench (33 held-out generated-family problems, 1 seed × K=5, 858 NIM calls,
wall 14 452 s with heavy 429-backoff; `results/w133/dev/.../w133_dev_report.json`):**

| arm | pass@1 | − A1 | − B0 | rescues vs A1 (modes) | MLB-2 |
|---|---|---|---|---|---|
| A0 | 72.73 % | — | — | — | — |
| A1 | 75.76 % | — | — | — | — |
| **B0** (blind reflexion) | **81.82 %** | +6.06 | — | (B0 − A1 = +6.06) | 33.3 % |
| **C1** (EW1 counterexample) | 81.82 % | +6.06 | **+0.00** | 2 (**COMPLEXITY only**); 0 vs B0 | 40 % |
| **C2** (EW2 complexity) | **87.88 %** | +12.12 | +6.06 | 4 (**COMPLEXITY only**) | 60 % |
| **C3** (controller, LEAD) | **87.88 %** | +12.12 | +6.06 | 4 (**COMPLEXITY only**); 2 vs B0 | 60 % |

**Two clean findings:**

1. **The complexity witness (EW2, in C2/C3) is REAL and load-bearing.** C3 beats B0 by
   **+6.06 pp** and A1 by **+12.12 pp**, with **MLB-2 = 60 %** (reflexion genuinely
   load-bearing), and all four of its A1-rescues are **algorithmic** (not formatting:
   `formatting_only=False`). The "your program is too slow at N≈80000 while a correct reference
   finishes in 0.2 s" signal systematically lets the 70B switch O(N²) → an efficient algorithm
   on `COMPLEXITY_BLIND` problems (`cb_distinct_in_windows`, `cb_pairs_sum_eq_t`,
   `cb_subarrays_sum_eq_k`, `cb_pairs_absdiff_le_d`) that the blind stack B0 and the
   counterexample-only arm C1 cannot crack. This is a genuine mechanism gain over W132's blind
   reject bit.
2. **The counterexample witness (EW1, in C1) adds NOTHING over blind reflexion.** C1 − B0 =
   **+0.00 pp**, **zero** rescues over B0. Showing a 70B a concrete failing input + the correct
   output for the HIDDEN_EDGE / WRONG_ALGORITHM / SEARCH_ENUM traps does not convert to rescues
   beyond what blind reflexion already achieves — the **wrong-algorithm capability ceiling** the
   literature documents (a named counterexample ≠ the correct algorithm), consistent with
   W129/W130.

**The gain is SINGLE-MODE.** Every witness rescue — vs A1 and vs B0, for every arm — is in
`COMPLEXITY_BLIND`. The lead arm C3 passes two of the three pre-committed earn conditions
(− A1 = +12.12 ≥ +5 ✓; − B0 = +6.06 > +3.33 ✓) but **FAILS the ≥ 2-distinct-modes condition**
(1 mode). Per the LOCKED **§7a dev gate** (beats B0 by ≥ +3.33 pp **AND** dev rescues span
≥ 2 modes), the dev gate **FAILS** ⇒ **$0 eval, $0 frontier**. The eval slice
(`88b9b79a…`, locked before spend) was deliberately **not** run: a single-mode gain cannot
clear the operator-locked ≥ 2-mode frontier-earn bar (§7b) on any slice, so funding it would
be spend beyond the pre-registration. **No retirement; W89 + W105 STAND.**

---

## Lane γ — primary-source research + stronger-model gate + graphify ($0) — gate CLOSED

**Primary-source research (RUNBOOK §9):** the mechanism class — concrete counterexample /
execution-feedback / oracle-grounded repair at a fixed call budget — is confirmed
**EXECUTABLE-HERE** (inference-time oracle + executor, NO training/RL) by CEGIS (Jha & Seshia,
SYNT 2014), counterexample-guided repair (Morvalho et al., AAAI 2025), Self-Debugging (Chen et
al., ICLR 2024 — feedback richness "matches or beats > 10× sampling", a same-budget win),
Reflexion (Shinn et al., NeurIPS 2023 — = the B0 baseline), LDB (ACL 2024), output-masked
self-inference (arXiv:2309.16120), and PBT shortest-counterexample (arXiv:2506.18315 — which
motivated the shrink-to-minimal witness). Two LIMITS the literature documents were **predicted
and then observed**: (a) **overfitting-to-the-shown-test** (Ahmed et al., arXiv:2511.16858;
UTGen/UTDebug, COLM 2025) ⇒ the disjoint-hidden-bank grading is the literature-endorsed guard
(and the witness arms must GENERALISE, which the complexity witness does and the counterexample
witness does not beyond B0); (b) the **wrong-algorithm capability ceiling** ⇒ the EW1 +0.00 pp
over B0. The literature **changed the design** (shrink-to-minimal counterexamples) but added no
new executable mechanism beyond the witness; trained test-generators / self-play are
TRAINING_DEPENDENT and were not used.

**Stronger-model gate (`results/w133/stronger_model_gate/gate_recheck_v1.json`):** re-derived
`NO_CERTIFIABLE_STRONGER_MODEL`, **decision CID `258b6ed7…` invariant** (byte-identical
W114→W133), {KNOWN:1, UNKNOWN:4}. Re-verified from primary sources: only Maverick discloses a
primary-KNOWN cutoff (Aug-2024, already-settled at W113); Qwen3-Coder-480B / DeepSeek-V4-pro
(NIM card: "Training Data Collection: Undisclosed") / Mistral-Small-4-119B-2603 / GLM-5 are all
primary-UNDISCLOSED. Gate **CLOSED**; no 405B run.

**graphify:** refreshed at START (HEAD `66c0b38`) and END; the new `exact_oracle_witness_v1`
creates the first semantic bridge from an oracle-WITNESS generator to the minted battlefield
(`MintedProblemV1`), the validated bench scaffolds (`icpc_reflexion_bench_v1`), and the audited
grader (`judge_icpc_output_v1`).

---

## Net + carry-forward

* **W89 (+5.56) + W105 (+7.00) remain the only two confirmed retirements.** W133 retires none
  (no frontier rerun was earned; the held-out gain is single-mode).
* **NEW `W133-T-EXACT-ORACLE-WITNESS-INSTRUMENT-MINTABLE`** — CoordPy can mint an exact-oracle
  witness instrument + a ≥ 96-instance seed-disjoint train/dev/eval curriculum over the owned
  battlefield, with witnesses that fire leakage-clean + genuinely-new on every admitted problem
  and on all six W132 traps. A reusable, deterministic, content-addressed teaching asset.
* **NEW `W133-T-COMPLEXITY-WITNESS-IS-REAL-AND-LOAD-BEARING-ON-HELD-OUT-DEV`** — on the held-out
  dev split the EW2 complexity witness (C3) beats the blind W132 stack B0 by +6.06 pp and A1 by
  +12.12 pp (MLB-2 = 60 %, 4 algorithmic complexity rescues): exact-oracle feedback richness IS
  a real, load-bearing same-budget gain — **for the COMPLEXITY mode**. (Single held-out dev
  seed; the locked eval slice was not run per the §7a dev gate.)
* **NEW `W133-L-WITNESS-FEEDBACK-SINGLE-MODE-CAP`** — the witness gain is COMPLEXITY-only; the
  EW1 counterexample channel adds +0.00 pp over blind reflexion (0 rescues vs B0) on the
  value/algorithm modes = the wrong-algorithm capability ceiling. The gain does not span ≥ 2
  modes ⇒ the pre-committed frontier-earn bar is not met ⇒ frontier rerun NOT earned ($0 eval,
  $0 frontier). The bounded contamination-EXPOSED-HumanEval-family-at-70B ceiling STANDS, now
  **LOCALISED**: the W132 +3.33 pp cap is part **feedback-fixable** (complexity, liftable by an
  oracle-grounded timing witness) + part **capability-bound** (value/algorithm, unliftable even
  with a perfect counterexample).
* W123→W132 caps carried forward unchanged; decision CID `258b6ed7…` invariant.

**W134** (per RUNBOOK §12): the dev gate failed (single-mode), so W134 = accept the registered
`W133-L-WITNESS-FEEDBACK-SINGLE-MODE-CAP`; the live levers are (a) an operator-greenlit
held-out **eval confirmation** of the single-mode complexity-witness gain (the eval slice
`88b9b79a…` is locked + push-button), (b) a **DEPLOYABLE complexity-timing witness** — the EW2
gain came from a "too slow at scale" signal whose oracle dependency (ref-is-fast) can be
replaced by a *constraint-derived* time budget (no oracle answer needed), making it the
genuinely-new deployable mechanism, (c) a genuinely different axis, the Maverick cross-scale
push-button, or a primary-KNOWN stronger-than-Maverick model when the §8c gate opens.
Bounded-context / compaction remain anti-patterns. COO-9 stays lead.

**No version bump (0.5.20 / coordpy.sdk.v3.43); no PyPI; `coordpy/__init__.py` untouched.**
