# RUNBOOK — W113

**Clean contamination-resistant-FOR-Llama-4 benchmark construction + earned
Maverick pilot + tier-2 readiness.**

> Locked **2026-05-29**, BEFORE any NIM call (incl. any reachability re-probe
> and the cheap pilot), per the W93–W112 preflight/earn discipline. A
> gated branch milestone with THREE lanes — **NOT a new benchmark tournament,
> NOT another confounded exposed-column rerun, NOT a 70B rerun.** `ultracode`
> stays OFF. `COO-9` stays the lead path.

---

## The one question W113 answers

W112 reopened a **+10.00 pp** Llama-4-Maverick reflexion margin on BigCodeBench
— but BigCodeBench 2024-06 is contamination-**EXPOSED** for Maverick (Meta-stated
**August-2024** cutoff > release), so it was an EXPOSED-column result, not a clean
resistant reopening (`W112-T-CONTAMINATION-RESISTANCE-IS-MODEL-CUTOFF-RELATIVE`).

> **Does the +10 pp SURVIVE on a benchmark VERIFIABLY contamination-resistant for
> Llama-4-Maverick?** (problem dates strictly after August 2024.)

This completes a clean 2×2 (model scale × slice resistance) on the SAME mechanism
and the SAME pinned corpus:

|                       |  70B (Llama-3.3)        |  Maverick (Llama-4)        |
|-----------------------|-------------------------|----------------------------|
| **EXPOSED**  (BigCodeBench 2024-06) | +0.00 pp  (W110) | **+10.00 pp** (W112)  |
| **RESISTANT** (LiveCodeBench 2025)  | **−3.33 pp** (W108, FAIL) | **??? ← W113**  |

The bar is a **clean** resistant result (`PASS_MECHANISM_DRIVEN`) or an honest
failure. A close / `PASS_NON_MECHANISM_DRIVEN` / confounded blip is **NOT** a win.

---

## Linear

* **`COO-9`** (High, Todo) — "Build a second code benchmark battlefield with
  lower ceiling pressure" — stays the **lead path**. Parent epic **`COO-6`**.
  W113 directly executes the COO-9 charter DoD (pick a family, build the
  loader/evaluator with fairness discipline, specify A0/A1/B before running,
  produce a runbook).
* **`COO-36`** = W112 (Done). Its COO-9 summary comment pre-commits
  **W113 = a benchmark verifiably resistant FOR Llama-4 (dates > Aug-2024;
  date-filtered LiveCodeBench)**, and flags that the tier-2 reachable models
  likely share the post-cutoff confound — exactly W113's design.
* **W113** = a NEW sub-issue under `COO-6` (sibling of `COO-9`), created at
  milestone end with full results + a COO-9 summary comment (the W105→W112
  pattern). `linear_github_mapping.json` updated + `sync_linear_github_v1.py`
  validated as part of the close.

---

## What is NOT in scope (anti-drift)

* **No** reopening MBPP+ V2 (W102 cap).
* **No** reopening the frozen cross-modal lines (RealWorldQA frozen at 11B).
* **No** reopening the closed Llama-3.1 rescue branch (W106 NO-GO).
* **No** APPS main-lane NIM (APPS stays the exposed control only).
* **No** 70B reflexion de-noise on resistant code (W109 rule; a −3.33/+0.00
  weak-MLB-2 point cannot be de-noised into a PASS).
* **No** 405B expensive run unless reachability changes AND a pre-committed gate
  clears (405B is 404×6).
* **No** bounded-context / compaction / token-compression / "truncate better"
  drift — those remain anti-patterns, not the frontier path.
* **No** version bump, **no** PyPI publish, `coordpy/__init__.py` untouched.
* `ultracode` stays OFF (this is a bounded disambiguation milestone, not a
  repo-wide dynamic-workflow job). Threshold to reconsider: multiple post-cutoff
  instruments built in parallel / a repo-wide cutoff-audit migration / broad
  multi-surface verification at once — none of which W113 requires. If crossed,
  say so explicitly before changing modes.

---

## Operational state (pre-W113 facts, held constant)

* **Two confirmed retirements STAND** — W89 (base HumanEval, +5.56 pp) + W105
  (HumanEval+, +7.00 pp), both `meta/llama-3.3-70b-instruct` @ 70B,
  contamination-EXPOSED HumanEval-family. W113 must not weaken these.
* **Resistant superiority = 0 clean demonstrations** (reflexion 0/2 at 70B; M3
  sub-floor; the stronger-model +10 pp is on a for-it-EXPOSED benchmark).
* **Fixed priors** (NOT inputs the pilot can move): W108 70B resistant LCB
  −3.33 pp (FAIL); W112 Maverick exposed BigCodeBench +10.00 pp.
* Reachability (W112 sweep, 2026-05-29): `meta/llama-4-maverick-17b-128e-instruct`
  = HTTP 200 (tier-1, selected); Qwen3-Coder-480B / DeepSeek-V4-pro /
  Mistral-Small-4-119B = reachable tier-2; `meta/llama-3.1-405b-instruct` = 404×6.

---

## § 1 — α / β / γ branch logic (LOCKED)

* **Lane α (main, LIVE):** build the post-Aug-2024 resistant slice → run the
  NIM-free preflight → if EARNED, the ONE earned expensive run is the cheapest
  honest Maverick Phase-2 pilot on the resistant slice. The verdict is mapped
  by the pre-committed `cross_scale_resistant_interpretation_v1` rule (§ 8).
* **Lane β (mandatory, NIM-free):** lock the tier-2 ranking + the
  same-filtered-slice applicability rule + the spend/no-spend rule
  (`tier2_readiness_v1`). NO tier-2 NIM in W113 (the rule shows none is even
  eligible — see § 6).
* **Lane γ (mandatory, NIM-free):** graphify refresh-at-start (done, HEAD
  `00210b7`) → use for file selection + dependency checks → refresh-at-end;
  tighten the claim surface so the model-cutoff-relativity lesson is defensible
  whatever the pilot returns.

---

## § 2 — post-Aug-2024 contamination-resistance rule for Llama-4 (LOCKED)

A LiveCodeBench problem is **RESISTANT-FOR Llama-4-Maverick** iff its
`contest_date` is **STRICTLY AFTER August 2024**:

```
boundary(Maverick) = "2024-08-31"   (last day of the stated cutoff month)
resistant(problem) ⟺ normalize(problem.contest_date) > "2024-08-31"
                   ⟺ contest_date day ∈ {2024-09-01, …}
```

Rationale (LOCKED): the Meta-stated cutoff is published at **month** granularity
("August 2024"), so a problem dated *in* August 2024 cannot be **certified**
strictly-after the cutoff. The conservative, defensible rule therefore
**EXCLUDES the entire August-2024 window**. Resistance is **model-cutoff-relative**
(`W112-T-…`): it is a property of the PAIR `(slice, model cutoff)`, encoded in the
`MODEL_TRAINING_CUTOFFS` registry (`coordpy.livecodebench_resistant_slice_v1`).
Cutoff entries carry a confidence tag — `KNOWN` (Maverick, Llama-3.3),
`ESTIMATED`, `UNKNOWN` — and resistance can only be **certified** against a
`KNOWN` cutoff (you cannot certify resistance against a cutoff you cannot verify).

---

## § 3 — date-filtering / ambiguity-exclusion / slice-construction rule (LOCKED)

1. **Normalize** `contest_date` → `YYYY-MM-DD` via
   `normalize_contest_date_v1`: take the leading ISO date, validate month∈[1,12]
   & day∈[1,31]; **return None** (→ EXCLUDE) on missing / blank / non-ISO /
   out-of-range. Comparison is at **day** granularity; the upstream field has no
   tz offset and the slice is months past the cutoff, so intra-day/tz
   normalization is immaterial and recorded out of scope
   (`W113-L-CONTEST-DATE-DAY-GRANULARITY-CAP`).
2. **Partition** the functional subset (`partition_resistant_v1`) into RESISTANT
   (day > boundary) and EXCLUDED, with a TYPED breakdown:
   `excluded_missing_date` / `excluded_unparseable_date` /
   `excluded_not_after_cutoff`. An ambiguous/missing date is **never** counted
   resistant.
3. **Slice** the resistant subset with the existing deterministic, OUTCOME-BLIND
   `select_livecodebench_functional_slice_v1` (difficulty-stratified
   largest-remainder, `(contest_date, question_id)`-ordered). Pin the
   `resistant_slice_cid`.
4. **Source field:** `contest_date` (one per upstream row; loader carries it
   verbatim). **Release pin:** `release_v6`, JSONL SHA-256
   `bb4c364f…`, 175 rows → 63 functional. **No cross-version mixing.**

---

## § 4 — filtered-slice preflight rules (LOCKED — the EARN artifact)

`scripts/run_w113_resistant_slice_preflight.py`, NIM-free. `pilot_earned` ⟺
`overall_pass` ⟺ **P1 ∧ P2 ∧ P3 ∧ P5 ∧ P6**:

* **P1** corpus integrity — SHA-pinned load (`bb4c364f…`); refuses on missing /
  SHA-mismatch / schema-mismatch.
* **P2** executor_V2 self-test — synthetic gold/wrong/loop + REAL gold zigzag
  (reused verbatim from the W108 preflight).
* **P3** loader real-data self-test — all `func_name` resolved; plain-arg mix;
  difficulty mix (reused verbatim).
* **P5** **resistant-partition integrity** (the W113 load-bearing check): cutoff
  is `KNOWN`-grade; resistant subset ≥ **30** (`MIN_RESISTANT_SLICE`); **zero**
  unparseable and **zero** missing dates; resistant min strictly after the
  boundary.
* **P6** **resistant-slice selection + equivalence**: select the 30-slice from
  the resistant subset; pin its CID; **ASSERT it equals the W108 slice CID
  `2afc318c…`**. Equivalence proves the date filter did not perturb the problem
  set, so the Maverick pilot runs on the EXACT problems 70B ran (W108) — the
  ONLY variable is model scale (clean cross-scale).

**W113 preflight result (2026-05-29, NIM-free):** P1∧P2∧P3∧P5∧P6 **PASS**.
63/63 functional resistant for Maverick (boundary 2024-08-31, KNOWN); 0 excluded
(0 missing / 0 unparseable / 0 in-August); resistant dates 2025-01-11..2025-04-05;
30-slice CID `2afc318c…` **==** W108 slice CID. **Maverick pilot EARNED.**
Verdict CID `6f30990c042593cd6c26290f54ec254472a369d7887b21ca86fed04c797f6ac8`.

---

## § 5 — cheap-pilot gates (LOCKED — only if § 4 EARNS)

Driver: `scripts/run_w113_resistant_pilot.py`
(`--model meta/llama-4-maverick-17b-128e-instruct`), 1 seed × 30 × K=5 = **330
NIM calls**, max_tokens 1024, seed **113001**, on the pinned resistant slice
(CID `2afc318c…`). Mechanism byte-identical to W89/W103/W105/W108/W110/W112.
A canary (2 problems ≈ 22 calls) confirms reachability + the plain code path
before the full run. The **same 9 Phase-2 gates + MLB-1/MLB-2** as W108/W110/W112,
via the canonical `evaluate_phase2_gates_v1`:

| Gate | Rule |
|------|------|
| G1 | slice CID pre-committed (`2afc318c…`) |
| G2 | A1@K=5 < 90 % (non-saturated) |
| G3 | B > A1 (strict) |
| G4 | (B − A1) ≥ +5.0 pp |
| G5 | (B − A0) ≥ +5.0 pp |
| G6 | B not worse than A1 on ≥ 16 / 30 |
| G7 | A1/B budget byte-exact |
| G8 | per-call CIDs + Merkle re-derive |
| G9 | no-LLM-judge subprocess executor |
| MLB-1 | reflexion invoked on ≥ 33 % of problems |
| MLB-2 | of invocations, ≥ 33 % rescued |

**Verdict label** (canonical): `PASS_MECHANISM_DRIVEN` (9/9 core ∧ MLB-1 ∧ MLB-2)
/ `PASS_NON_MECHANISM_DRIVEN` (9/9 core, MLB fails) / `FAIL` (any core gate fails).

**§ 1α CLEAN-REOPENING BAR (LOCKED):** a clean resistant reopening requires
**`PASS_MECHANISM_DRIVEN`**. A `PASS_NON_MECHANISM_DRIVEN` margin (the W112
exposed shape) is **explicitly NOT** a clean reopening; a `FAIL` confirms the
W112 +10 pp was exposure.

---

## § 6 — tier-2 ranking + no-spend / spend rule (LOCKED — Lane β)

`coordpy.tier2_readiness_v1`. **Ranking** (carried from the W112 sweep
`rank_tier`): tier-2 preference = (1) `qwen/qwen3-coder-480b-a35b-instruct`
(code-specialized, largest), (2) `deepseek-ai/deepseek-v4-pro`,
(3) `mistralai/mistral-small-4-119b-2603`.

**Same-filtered-slice applicability rule:** a tier-2 model may be tested on a
resistant slice ONLY if the slice is **CERTIFIABLY** resistant for it — its
cutoff is `KNOWN` AND the slice min date is strictly after the boundary.

**Spend rule:** tier-2 NIM is spent iff **(a)** the W113 main-lane verdict earns
escalation (a clean reopening → replicate; an exposure-confirm → localize) **AND
(b)** ≥ 1 tier-2 model has a certifiably-resistant slice (≥ 30 problems) **AND
(c)** the same cheap K=5 single-seed budget.

**W113 verdict (LOCKED):** all three tier-2 cutoffs are **UNKNOWN** (Qwen3-Coder
released 2025-07, cutoff undisclosed; DeepSeek-V4-pro 2025+; Mistral-Small-4
"2603" = 2026-03 release) and plausibly overlap / post-date the test6 slice
(2025-01..04), so **NONE is certifiably resistant on the pinned corpus** ⇒
**tier-2 spend BLOCKED on a missing instrument, independent of the main-lane
outcome.** `$0` tier-2 NIM in W113. The next instrument for any tier-2 follow-up
is a **LATER date-filtered LiveCodeBench slice (release_v7+)** with problems
strictly after that model's first-KNOWN cutoff, operator-fetched + SHA-pinned.

---

## § 7 — graphify deliverables (LOCKED — Lane γ)

* Refresh at start from HEAD (`graphify update .` + `cluster-only --no-viz` to
  bring the report stamp to HEAD `00210b7`; **0 token cost**). **DONE.**
* Use concretely: `explain` on `run_livecodebench_reflexion_bench_v1` +
  `run_bigcodebench_reflexion_bench_v1`; `path` bigcodebench→livecodebench;
  `affected run_livecodebench_reflexion_bench_v1`; `explain` on the new W113
  modules/scripts; `query` for where the resistant-bound / contamination-bound
  claims live.
* Refresh at end after all code/doc changes; confirm the graph is built from the
  W113 HEAD.
* Tighten the claim surface (registry / status / honesty / consolidated
  narrative / contamination framing / CHANGELOG) so the model-cutoff-relativity
  lesson is defensible after W113.

---

## § 8 — W114 branch logic (LOCKED — pre-committed by verdict label)

Selected purely by the pilot's `verdict_label` via
`interpret_cross_scale_resistant_result_v1`:

* **`PASS_MECHANISM_DRIVEN` → `RESISTANT_SUPERIORITY_REOPENS`** (clean):
  scale GENUINELY reopens resistant superiority; the +10 pp was NOT mere
  exposure. **W114 = Maverick × resistant-LiveCodeBench Phase-3 retirement bench**
  (3 seeds × ~100 × K=5) — the earned path to a contamination-RESISTANT THIRD
  retirement (a genuinely new frontier). Entitled to a STRONGER superiority
  claim. Load W114 immediately; do not stop at the headline.
* **`PASS_NON_MECHANISM_DRIVEN` → `RESISTANT_MARGIN_NON_MECHANISM`** (ambiguous):
  margin survives but mechanism not load-bearing — NOT a clean reopening.
  **W114 weighs a multi-seed resistant de-noise vs accepting the bounded claim.**
  No Phase-3 entitlement; NOT entitled to a stronger superiority claim.
* **`FAIL` → `EXPOSURE_CONFIRMED`** (decisive negative): the W112 +10 pp WAS
  exposure; within the SAME model + mechanism the margin flips +10.00 pp
  (exposed) → FAIL (resistant) purely on slice resistance — the sharpest
  contamination dissociation yet, confirming model-cutoff-relativity at scale.
  **W114 = accept the bounded contamination-EXPOSED claim as the honest code
  ceiling and pursue a genuinely DIFFERENT axis** (not another exposed rerun, not
  another same-scale resistant reflexion pilot); tier-2 follow-up ONLY if a
  per-model-resistant slice is fetched + certified (§ 6). Confound STRENGTHENS
  (still not proof — single-seed cells; capability-scale not excluded).

In ALL branches: **W89 + W105 STAND**; `COO-9` stays lead unless the evidence
forces a different code-line move.

---

## § 9 — Stable boundary preservation (LOCKED)

* `coordpy.__version__ == "0.5.20"`; `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`;
  **no PyPI**; `coordpy/__init__.py` untouched.
* Advanced work explicit-import-only: 3 new modules
  (`livecodebench_resistant_slice_v1`, `cross_scale_resistant_interpretation_v1`,
  `tier2_readiness_v1`) + 2 scripts (preflight + pilot). The pilot reuses the
  canonical `evaluate_phase2_gates_v1` + the W108 NIM generator (namespace
  import; no duplication).
* 23rd consecutive preflight/earn-discipline validation (W93–W113).

---

## Honest framing

W113 does **multiple** substantial things in one push: a clean
resistant-FOR-Llama-4 slice construction with proven date integrity; a real
NIM-free preflight that EARNS the pilot; the earned cheapest-honest Maverick
pilot on the EXACT W108 slice; real tier-2 readiness with a locked spend gate
that the missing-instrument finding dominates; graphify refresh/use/refresh; and
the Linear/GitHub sync. The main move is **not** "try Maverick again" — it is to
build a benchmark actually resistant for Maverick and test it honestly. If the
pilot is `PASS_MECHANISM_DRIVEN`, say so strongly and load W114 immediately. If
it FAILs, say so sharply and harden the boundary. A close or
non-mechanism-driven edge is **not** a win.
