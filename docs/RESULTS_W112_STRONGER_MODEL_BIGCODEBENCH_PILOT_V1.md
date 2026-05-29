# RESULTS — W112 Lane α: stronger-model (Llama-4-Maverick) BigCodeBench pilot ⇒ +10.00 pp PASS_NON_MECHANISM_DRIVEN, but model-relative-contamination-CONFOUNDED (exposed-column), NOT a clean resistant reopening

**Verdict: the earned stronger-model pilot returned B − A1 = +10.00 pp, 9/9 core
Phase-2 gates, MLB-2 = 37.5 % (PASS) but MLB-1 = 26.67 % (FAIL) ⇒
`PASS_NON_MECHANISM_DRIVEN`. The margin REOPENED at the stronger model where 70B
was flat — BUT BigCodeBench (2024-06) is contamination-RESISTANT only relative to
Llama-3.3-70B's ~2024-01 cutoff; Llama-4-Maverick's pretraining cutoff is AUGUST
2024, which POST-DATES the benchmark, so BigCodeBench is plausibly
contamination-EXPOSED for Maverick. The result's structural signature is a
near-exact twin of the W109 APPS contamination-EXPOSED control (A0 = 73.33 %
IDENTICAL; PASS_NON_MECHANISM_DRIVEN; MLB-1-fail/MLB-2-pass). So this is most
parsimoniously an EXPOSED-column result, NOT a clean reopening of
contamination-RESISTANT superiority. W112 adds NO retirement; the two confirmed
retirements (W89, W105) STAND; the contamination-confound is STRENGTHENED again
(first WITHIN-benchmark resistant→exposed flip).**

---

## 1. Run identity (audit chain)

| Field | Value |
|---|---|
| Model | `meta/llama-4-maverick-17b-128e-instruct` (frontier MoE 400B total / 17B active; **Aug-2024 pretraining cutoff**; released 2025-04) |
| Why this model | the LOCKED § 1α tier-1 target: Llama-FAMILY cross-generation-UP (3.3 → 4); strictly stronger than 70B on code; non-reasoning (C-S3); reachable (HTTP 200) where 405B is 404×6. Selected by `scripts/run_w112_stronger_model_reachability_sweep_v1.py` (decision CID `a654956b…`); canary-confirmed code path. |
| Mechanism | sequential reflexion **B** (the § 1α-mech default; the best-measured 70B resistant mechanism); A0 + A1 baselines byte-identical to W110 |
| Benchmark / slice | the EXACT W110 fair 30-problem BigCodeBench gold-green slice (slice CID `b69bf3a0…` **re-derived + matched** ⇒ G1 holds; only the model differs from W110) |
| Corpus | `bigcode/bigcodebench` v0.1.4; JSONL SHA `ca4f352e…` (re-verified at load); release 2024-06 |
| Budget | 1 seed (110001) × 30 × K=5; byte-exact; no early-stop; ~330 NIM calls; wall 2350.5 s (~39 min) |
| Executor | `bigcodebench_executor_v1` — fresh `-I` subprocess `unittest` oracle, headless `Agg`, bcb_venv; NO LLM judge |
| Driver | `scripts/run_w110_bigcodebench_pilot.py` reused VERBATIM (only `--model` + `--out-dir` changed) ⇒ byte-identical harness/mechanism, clean cross-model comparison |

---

## 2. Empirical result — vs W110 70B on the SAME slice

| Arm | W110 `llama-3.3-70b` | W112 `llama-4-maverick` | Δ |
|---|---|---|---|
| A0 (single-shot T=0) | 63.33 % | **73.33 %** (22/30) | +10.00 pp |
| A1 (first-pass-among-K=5) | 70.00 % | **73.33 %** (22/30) | +3.33 pp |
| B (sequential reflexion K=5) | 70.00 % | **83.33 %** (25/30) | +13.33 pp |
| **B − A1** | **+0.00 pp** | **+10.00 pp** | **+10.00 pp** |
| MLB-1 (invocation rate) | 40.00 % (12/30) PASS | **26.67 % (8/30) FAIL** | |
| MLB-2 (rescue rate) | 25.00 % (3/12) FAIL | **37.50 % (3/8) PASS** | |
| Core Phase-2 gates | 7/9 | **9/9** | |
| Verdict | `FAIL` | **`PASS_NON_MECHANISM_DRIVEN`** | |

**The +10 pp margin = exactly 3 clean reflexion rescues with ZERO regressions:**
B rescued `BigCodeBench/15`, `/26`, `/51` (all on the FIRST reflexion turn,
`first_pass_idx=1`); no A1-pass→B-fail regressions (vs W110's net-0 from 3 rescues
cancelled by 3 regressions). So the margin is rescue-driven (MLB-2 = 37.5 %
healthy) — but the LABEL is `PASS_NON_MECHANISM_DRIVEN` because **MLB-1 = 26.67 %
< 33 %**: the stronger model solves more on attempt-0, so reflexion is invoked on
only 8/30 problems, below the load-bearing-invocation floor.

---

## 3. The decisive caveat — contamination-resistance is MODEL-CUTOFF-RELATIVE

BigCodeBench's contamination-resistance was established **relative to
Llama-3.3-70B's cutoff** (`docs/RESULTS_W110_*`: "post Llama-3.x ~2024-01
cutoff"). That property does NOT transfer to a newer model:

* **Llama-4-Maverick pretraining cutoff = August 2024** ([Meta Llama 4 model
  card](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4/); [AI
  Analysis](https://artificialanalysis.ai/models/llama-4-maverick)).
* **BigCodeBench v0.1.4 was publicly released ~June 2024** (HuggingFace + GitHub).
* August 2024 **post-dates** June 2024 ⇒ BigCodeBench is within Maverick's
  training window ⇒ **plausibly contamination-EXPOSED for Maverick** (the burden
  of proving RESISTANCE for the tested model is UNMET).

**The structural signature confirms the exposed reading.** W112 Maverick is a
near-exact twin of the W109 APPS contamination-EXPOSED control:

| | W109 APPS (EXPOSED 2021) | W112 Maverick BigCodeBench (exposed-for-Llama-4) |
|---|---|---|
| A0 single-shot | 73.33 % | **73.33 % (identical)** |
| B − A1 | +16.67 pp | +10.00 pp |
| MLB-1 | 23.33 % FAIL | 26.67 % FAIL |
| MLB-2 | 57.14 % PASS | 37.50 % PASS |
| Verdict | PASS_NON_MECHANISM_DRIVEN | **PASS_NON_MECHANISM_DRIVEN** |

The two contamination-RESISTANT points (at the model the benchmark is resistant
FOR) stay flat/negative: W108 LCB-2025 / Llama-3.3 = −3.33 pp; W110
BigCodeBench-2024 / Llama-3.3 = +0.00 pp (both MLB-2 = 25 % FAIL). The
contamination-EXPOSED points (incl. for-Llama-4-exposed BigCodeBench) all show a
large margin: W89 +5.56 / W105 +7.00 / W109 +16.67 / **W112 +10.00**. The SAME
slice flips +0.00 pp → +10.00 pp as the model's cutoff crosses the benchmark's
release date — the cleanest contamination dissociation in the programme, and the
FIRST within-benchmark one.

Corroboration from the Lane β census: the 3 problems Maverick "rescued"
(`/15`,`/26`,`/51`) were classified UNREACHABLE-in-the-fair-regime at 70B
(mock-coupled / no-contract). A stronger model fixing them from the SAME stderr
is consistent with either (a) genuine capability or (b) recall of a benchmark in
its training window — and the single confounded pilot CANNOT separate the two.

---

## 4. Honest interpretation (what it IS / IS NOT)

**What it IS.** The earned cross-scale-UP probe RAN on a genuinely stronger,
same-budget-comparable code model and produced a large (+10 pp), 9/9-gate,
zero-regression, MLB-2-healthy margin where 70B was flat. That is a real,
audited empirical datum and the honest aggressive move the milestone demanded —
the resistant ceiling at 70B is NOT a hard universal wall; a stronger model moves
the numbers materially.

**What it is NOT — a clean reopening of contamination-RESISTANT superiority.**
(1) BigCodeBench is NOT verified resistant for Maverick (Aug-2024 cutoff
post-dates the 2024-06 release) ⇒ the +10 pp is most parsimoniously an
EXPOSED-column result, structurally identical to the W109 APPS control. (2) The
verdict is `PASS_NON_MECHANISM_DRIVEN` (MLB-1 sub-floor), not a clean
mechanism-driven PASS. (3) Single-seed cheap pilot ⇒ NOT a retirement (Phase-3
multi-seed required). (4) The capability-vs-contamination ambiguity is
unresolved by this pilot.

**What it sharpens.** The contamination-confound is STRENGTHENED a third time —
and for the first time WITHIN a single benchmark: BigCodeBench's +0.00 pp (for
the model it is resistant for) → +10.00 pp (for the model it is exposed for) is
a vintage-by-cutoff dissociation on the identical slice. Still NOT proven
(single-seed; capability not fully excluded), but the confound now has a cleaner
mechanism: **contamination-resistance is model-cutoff-relative**, so "resistant
benchmark" claims must be re-qualified per model tested. The two confirmed
retirements remain contamination-EXPOSED-HumanEval-family-specific at 70B.

**The auto-computed interpreter** (`interpret_second_resistant_result_v1`, built
for the same-model W110 question) labels the confound "UNCHANGED /
SUPPORTED-not-proven" because it cannot see the model-cutoff flip. The
milestone-level reading SUPERSEDES it: the within-benchmark flip STRENGTHENS the
confound. Both are recorded; the superseding reading is the honest one.

---

## 5. Carry-forwards

**Added:**
* `W112-T-STRONGER-MODEL-BIGCODEBENCH-MARGIN-REOPENS-BUT-MODEL-EXPOSED` —
  ESTABLISHED (empirical; 330 NIM calls; bench Merkle in the report). On the EXACT
  W110 30-slice, `meta/llama-4-maverick` gives B − A1 = +10.00 pp (9/9 gates;
  MLB-2 = 37.5 % PASS; MLB-1 = 26.67 % FAIL; `PASS_NON_MECHANISM_DRIVEN`; 3 clean
  rescues `/15`,`/26`,`/51`, 0 regressions). The margin reopened vs 70B's
  +0.00 pp.
* `W112-T-CONTAMINATION-RESISTANCE-IS-MODEL-CUTOFF-RELATIVE` — ESTABLISHED
  (methodology + grounded dates). BigCodeBench 2024-06 is resistant for
  Llama-3.3-70B (~2024-01 cutoff) but plausibly EXPOSED for Llama-4-Maverick
  (Aug-2024 cutoff). The +0.00 pp → +10.00 pp flip on the identical slice is a
  within-benchmark resistant→exposed dissociation. STRENGTHENS the
  contamination-confound (not proof). "Resistant benchmark" must be re-qualified
  per model cutoff.
* `W112-L-STRONGER-MODEL-RESISTANT-SUPERIORITY-NOT-CLEANLY-DEMONSTRATED-CAP` —
  CAP. The W112 +10 pp does NOT cleanly reopen contamination-RESISTANT
  superiority: the benchmark is unverified-resistant (likely exposed) for the
  tested model; the verdict is non-mechanism-driven (MLB-1 sub-floor); single
  seed; capability-vs-contamination unresolved. NOT a retirement.

**Not retired:** the two confirmed retirements (W89, W105) — unchanged. W112
adds NO retirement.

---

## 6. What W113 becomes (refined by the cutoff discovery)

The locked RUNBOOK § 8 lumped `PASS_NON_MECHANISM_DRIVEN` with FAIL as "ceiling
persists." The DISCOVERED model-cutoff confound refines that branch (a
legitimate in-analysis refinement, not a post-hoc target-shift): the +10 pp is a
CONFOUNDED positive that must be DISAMBIGUATED, not dismissed. **W113 = a
benchmark VERIFIABLY contamination-resistant FOR Llama-4-Maverick** — i.e. with
problem/contest dates AFTER Aug 2024 (a date-filtered LiveCodeBench slice is the
natural instrument; the § 1α-bench secondary cross-check, now sharpened with a
post-Aug-2024 date filter). If the +10 pp HOLDS on a benchmark resistant for
Maverick ⇒ scale genuinely reopens RESISTANT superiority (the new frontier; then
a Phase-3 bench). If it COLLAPSES (like 70B) ⇒ the +10 pp was contamination
exposure and the bounded claim is reinforced. The other reachable tier-2 targets
(Qwen3-Coder-480B, DeepSeek-V4-pro) likely share the post-2024-06 cutoff confound
on BigCodeBench, so the DATE-FILTERED resistant benchmark — not merely another
model — is the load-bearing W113 instrument. `COO-9` stays lead.

---

## 7. Stable boundary preserved

`coordpy.__version__ == 0.5.20`; `coordpy.SDK_VERSION == coordpy.sdk.v3.43`;
no PyPI publish; `coordpy/__init__.py` untouched. ZERO new `coordpy.*` modules
(the pilot reused the W110 driver + bench + the mechanism-agnostic 9-gate
evaluator verbatim). The one earned expensive run was the 330-call Maverick
pilot; $0 on 405B (404×6 in the sweep), $0 on the other tier-2 targets, $0 on a
second pilot (W113 pre-committed), $0 on APPS / Llama-3.1 / 70B reflexion.
