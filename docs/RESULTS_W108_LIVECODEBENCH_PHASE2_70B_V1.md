# RESULTS — W108 LiveCodeBench functional-subset Phase-2 cheap pilot (70B)

**Verdict: `FAIL` (clean, NON-mechanism-driven). 7/9 Phase-2 gates; B − A1 = −3.33 pp; MLB-2 = 25 % < 33 %.**

This is the first empirical attack on the LiveCodeBench battlefield that W107
preflighted, executed end-to-end after the W108 real-data bug-fix. It is a
clean Phase-2 FAIL — the W89 sequential-reflexion mechanism does NOT beat
same-budget self-consistency on the contamination-resistant 2025 functional
subset at 70B. The two confirmed retirements (W89, W105) are UNCHANGED. W108
adds NO retirement.

---

## 1. Run identity (audit chain)

| Field | Value |
|---|---|
| Model | `meta/llama-3.3-70b-instruct` (the W89/W105 retirement class) |
| Corpus | `livecodebench/code_generation_lite` `release_v6` (`test6.jsonl`) |
| Corpus SHA-256 | `bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5` |
| Functional subset | 63 problems (of 175); all plain-arg; all dated 2025-01-11…2025-04-05 (post Llama-3.x cutoff) |
| Pilot slice | 30 problems; difficulty 8 easy / 12 medium / 10 hard; dates 2025-01-11…2025-02-15 |
| Slice CID | `2afc318cb9a24d9a52b8914082cfbddaa8e941ef85d77be5382981621f43aa82` |
| Preflight verdict CID | `61b9961c46943a5db8dbfd66b75844bfa2d092eac5543df2f4a613556e06bf33` |
| Seed | 108001 (single-seed cheap pilot) |
| K (A1 and B) | 5, byte-exact; no early-stop |
| NIM calls | 330 (A0 1 + A1 5 + B 5 per problem × 30) |
| Wall | 4 603.9 s (~77 min; 444 HTTP-429 retry events — heavy throttling, survived) |
| Bench Merkle root | `51876af221b028cab0a0bd77…` |

A0/A1/B are byte-identical in mechanism to W89/W103/W105; only the corpus,
executor (V2 newline decoder) and the prompt (produce a complete
`class Solution`) differ. NO LLM-as-judge — executor truth is the subprocess
exit code over the public functional tests.

---

## 2. Empirical result

| Arm | pass@1 |
|---|---|
| A0 (single-shot T=0) | **43.33 %** (13/30) |
| A1 (first-pass-among-K=5, T=0.7) | **63.33 %** (19/30) |
| B (sequential-reflexion-K=5, T=0.7) | **60.00 %** (18/30) |
| **B − A1** | **−3.33 pp** |
| B − A0 | +16.67 pp |

Per-problem A1-vs-B transition surface (30 problems):

| both pass | A1-only (B regressed) | B-only (B rescued past A1) | neither |
|---|---|---|---|
| 14 | 5 | 4 | 7 |

The net difference is exactly **one problem** (A1 solved 19, B solved 18).
Reflexion rescued 4 problems past A1 (qids 3701, 3692, 3763, 3771) but
regressed on 5 — a net −1.

---

## 3. The 9 Phase-2 gates + MLB sub-gates

| Gate | Pass | Value |
|---|---|---|
| G1 slice pre-committed | ✅ | slice CID pinned, reproduced by `--dry-run` |
| G2 A1 < 90 % | ✅ | 63.33 % (non-saturated — real headroom existed; the preflight C1/S2 was right) |
| G3 B > A1 | ❌ | 60.00 % < 63.33 % |
| G4 (B − A1) ≥ +5 pp | ❌ | −3.33 pp |
| G5 (B − A0) ≥ +5 pp | ✅ | +16.67 pp |
| G6 per-problem majority (≥ 16/30) | ✅ | 25/30 (B did not regress vs A1 on 25) |
| G7 budget byte-exact | ✅ | A1, B both K=5 |
| G8 audit chain re-derives | ✅ | per-call CIDs + per-seed/bench Merkle |
| G9 executor clean | ✅ | no-LLM-judge subprocess |
| MLB-1 invocation ≥ 33 % | ✅ | **53.33 %** (16/30 problems triggered reflexion) |
| MLB-2 rescue ≥ 33 % | ❌ | **25.00 %** (4/16 invocations rescued) |

**7/9 gates pass; the two failures are G3 (B > A1) and G4 (margin), plus the
MLB-2 load-bearing sub-gate. Verdict = `FAIL` (NON-mechanism-driven).**

---

## 4. Honest interpretation (what this is, and is NOT)

**What it IS — a clean, well-formed FAIL.** Reflexion was invoked at a healthy
rate (MLB-1 = 53.33 %, mirroring HumanEval+'s ~53–57 % invocation), so the
slice was genuinely hard enough to exercise the mechanism. But its rescue rate
was **25 %** — roughly HALF the HumanEval-family rescue rate (W103 47.06 % /
W105 55.62 %) — and below the 33 % load-bearing floor. Same-budget self-
consistency (A1) edged out sequential reflexion (B) by one problem. This is the
**W102 MBPP+ V2 shape** (FAIL with weak MLB-2 = 22 %), NOT the W103/W105
HumanEval shape (PASS with MLB-2 ≈ 47–56 %).

**The single most important fact:** this is the **first time the W89 mechanism
has been tested on a contamination-resistant benchmark**, and it FAILed. The
two confirmed retirements are on **contamination-EXPOSED** problems — base
HumanEval (2021) and HumanEval+ (the same 2021 problems with extra tests), both
inside the Llama-3.x training corpus. W108's LiveCodeBench slice is entirely
**post-cutoff 2025** problems the model has not seen. On that clean surface,
the reflexion advantage that retired on HumanEval-family did NOT replicate.

**What it is NOT — proof of a contamination confound.** With n = 30, 1 seed,
the −3.33 pp margin is a 1-problem effect and MLB-2's 25 % (4/16) is also
small-sample. W108 RAISES a **contamination-confound hypothesis** — that the
W89/W105 reflexion-superiority may be partly linked to benchmark familiarity —
but a single cheap pilot does not establish it. The FAIL is equally consistent
with (a) benchmark-family difficulty, (b) single-seed cheap-pilot noise, or
(c) a genuine contamination confound. It is registered as a HYPOTHESIS, not a
finding.

**What it does NOT change.** The two confirmed retirements (W89 +5.56 pp; W105
+7.00 pp, 6/6 bars, MLB-2 55.62 %) STAND — they are real same-budget results on
their benchmarks. W108 adds NO third retirement and retires NO research
carry-forward. It does add an explicit boundary: **the retirements have not
been demonstrated on contamination-resistant data, and the one attempt cleanly
failed.** This makes the programme's claim MORE carefully bounded, not weaker.

---

## 5. Carry-forwards

**Added (theorem / infrastructure anchors):**
* `W108-T-LIVECODEBENCH-REAL-DATA-BUGFIX-METADATA-JSON-STRING` — diagnosed +
  fixed the partial-scaffold gold-path smoke (metadata-as-JSON-string →
  `func_name=""` → ENTRY_NOT_FOUND); 19 regression tests lock it.
* `W108-T-LIVECODEBENCH-REAL-DATA-PREFLIGHT-EARNED` — real-corpus preflight
  PASS (P1∧P2∧P3∧P4); the cheap pilot was genuinely earned before NIM spend.

**Added (caps):**
* `W108-L-LIVECODEBENCH-REFLEXION-PHASE2-70B-CAP` — LiveCodeBench functional
  subset, `meta/llama-3.3-70b-instruct`, 1 seed × 30 × K=5: B − A1 = −3.33 pp;
  7/9 gates; MLB-2 = 25 % < 33 %. Clean FAIL, NON-mechanism-driven.
* `W108-L-REFLEXION-NOT-DEMONSTRATED-ON-CONTAMINATION-RESISTANT-BENCH-CAP` —
  the W89/W105 reflexion-superiority retirements are on contamination-EXPOSED
  HumanEval-family benchmarks; the first contamination-resistant attempt
  (LiveCodeBench 2025) FAILed (B − A1 = −3.33 pp; MLB-2 = 25 %). A
  contamination-confound hypothesis is OPEN, not established.

**Discharged (infrastructure caps confirmed on real data — NOT research
retirements):**
* `W107-L-LIVECODEBENCH-LOADER-V1-SCHEMA-CONFIRM-AT-FETCH-CAP` — DISCHARGED:
  the real `release_v6` schema is confirmed (metadata + test-cases are JSON
  strings; newline-per-arg input) and the loader binds it correctly.
* `W107-L-LIVECODEBENCH-RESIDUAL-PUBLISHED-BASELINE-GRADE-CAP` — DISCHARGED for
  the pilot slice: A1@K=5 = 63.33 % is now a LIVE re-executed measurement, no
  longer a published-baseline estimate.

**Stands:**
* `W107-L-LIVECODEBENCH-FUNCTIONAL-SUBSET-ONLY-CAP` — still functional-only.

**NOT retired:** the two confirmed retirements (W89, W105) — unchanged.

---

## 6. What W109 becomes

Per `docs/RUNBOOK_W108.md` § 8 (FAIL branch): register this cap, then decide
between an APPS cheap pilot or an honest no-go. The W108 result makes the
**APPS contaminated-control contrast** the most scientifically valuable next
move: APPS (2021 vintage, contamination-EXPOSED) and LiveCodeBench (2025,
contamination-RESISTANT) probe the same call-based functional shape with the
same mechanism + budget. If APPS PASSes where LiveCodeBench FAILed, that is
direct evidence the W89/W105 advantage is contamination-linked; if APPS also
FAILs, the advantage is HumanEval-family-specific. Either outcome sharpens the
programme's honest boundary. (A multi-seed LiveCodeBench re-check to de-noise
the 1-problem margin is the alternative.) The APPS scaffolding is already built
and pivot-ready (Lane γ). `COO-9` stays the lead path.
