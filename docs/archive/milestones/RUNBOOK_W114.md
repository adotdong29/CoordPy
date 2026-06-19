# RUNBOOK — W114

**Bounded-ceiling registration + newer-certifiable-post-cutoff-instrument
attempt + earned pilot ONLY if a stronger model is honestly certifiable.**

> Locked **2026-05-29**, BEFORE any NIM call (incl. any reachability re-probe
> and any pilot/canary), per the W93–W113 preflight/earn discipline. A gated
> branch milestone with THREE lanes — **NOT another exposed rerun, NOT another
> same-scale resistant reflexion rerun on the same instrument, NOT a bounded-
> context / compaction / token-compression job.** `ultracode` stays OFF.
> `COO-9` stays the lead path.

---

## The one question W114 answers

W113 confirmed the W112 +10 pp was contamination EXPOSURE: on a benchmark
verifiably resistant for Llama-4-Maverick (date-filtered LiveCodeBench
`release_v6`, all 2025-01..04 ≫ the Aug-2024 cutoff, the EXACT W108 slice), the
W89 mechanism gives **B − A1 = +0.00 pp (FAIL)** at Maverick scale, collapsing
exactly as 70B did (W108 −3.33). Resistant superiority is now **0 clean across
BOTH scales**. The two retirements (W89 + W105) STAND, both contamination-
EXPOSED-HumanEval-family at 70B.

W114 does NOT ask "can we find another borderline positive?". It asks:

> **Can we construct a NEW instrument that is certifiably contamination-resistant
> for a reachable STRONGER model (not just Maverick), and earn one clean
> empirical shot on it — from the latest REAL release and OFFICIAL model
> cutoffs?**

If yes → run the cheapest honest pilot. If only Maverick is certifiable →
decide whether a second Maverick resistant pilot has verdict-changing power (it
does not; § 6). If no stronger model is certifiable from the latest real data →
**stop honestly and make the missing-instrument blocker load-bearing** (§ 7).
A close / confounded / exposed edge is **NOT** a win.

---

## Linear

* **`COO-9`** (High, Todo) — "Build a second code benchmark battlefield with
  lower ceiling pressure" — stays the **lead path**. Parent epic **`COO-6`**.
* **`COO-37`** = W113 (Done). Its close pre-commits **W114 = accept the bounded
  contamination-EXPOSED-HumanEval-family-at-70B claim as the honest code ceiling
  and pursue a GENUINELY DIFFERENT axis; tier-2 ONLY if a per-model-resistant
  slice is fetched + certified** (`docs/RUNBOOK_W113.md` § 8 FAIL branch).
* **W114** = a NEW sub-issue under `COO-6` (sibling of `COO-9`), created at
  milestone end with full results + a COO-9 summary comment (the W105→W113
  pattern). `linear_github_mapping.json` updated + `sync_linear_github_v1.py`
  validated as part of the close.

---

## What is NOT in scope (anti-drift)

* **No** reopening MBPP+ V2 (W102 cap).
* **No** reopening the frozen cross-modal lines (RealWorldQA frozen at 11B).
* **No** reopening the closed Llama-3.1 rescue branch (W106 NO-GO).
* **No** APPS main-lane NIM (APPS stays the exposed control only).
* **No** 70B resistant reflexion de-noise (W109 rule; a −3.33/+0.00 weak-MLB-2
  point cannot be de-noised into a PASS).
* **No** second Maverick resistant reflexion rerun on the same instrument
  (redundant; § 6).
* **No** 405B expensive run unless reachability changes AND a pre-committed gate
  clears (405B is 404×6).
* **No** bounded-context / compaction / token-compression / "truncate better"
  drift — those remain anti-patterns, not the frontier path.
* **No** version bump, **no** PyPI publish, `coordpy/__init__.py` untouched.
* `ultracode` stays OFF (this is a bounded registration + certification-supply
  milestone, not a repo-wide dynamic-workflow job). Threshold to reconsider:
  multiple new dated instruments built in parallel / a repo-wide cutoff-audit
  migration / broad multi-surface verification at once — none of which W114
  requires. If crossed, say so explicitly before changing modes.

---

## Operational state (pre-W114 facts, held constant)

* **Two confirmed retirements STAND** — W89 (base HumanEval, +5.56 pp) + W105
  (HumanEval+, +7.00 pp), both `meta/llama-3.3-70b-instruct` @ 70B,
  contamination-EXPOSED HumanEval-family. W114 must not weaken these.
* **Resistant superiority = 0 clean across BOTH scales** (70B −3.33 / +0.00;
  Maverick +0.00; M3 sub-floor at 70B).
* Reachability (W112 sweep, 2026-05-29, held as a fixed prior — NOT re-probed
  in W114 because reachability is not the binding gate; § 2):
  `meta/llama-4-maverick-17b-128e-instruct` = 200 (tier-1);
  `qwen/qwen3-coder-480b-a35b-instruct` / `deepseek-ai/deepseek-v4-pro` /
  `mistralai/mistral-small-4-119b-2603` = reachable tier-2;
  `meta/llama-3.1-405b-instruct` = 404×6.

---

## § 1 — α / β / γ branch logic (LOCKED)

* **Lane α (mandatory, NIM-free, EARLY but not the whole milestone):**
  register the W113 result as the **honest code ceiling** across every canonical
  truth surface (README / RESEARCH_STATUS / THEOREM_REGISTRY / consolidated
  narrative / HOW_NOT_TO_OVERSTATE / new W114 contamination framing / CHANGELOG).
  Boundedness impossible to miss: contamination-EXPOSED-HumanEval-family at 70B;
  exactly two retirements; **0 clean resistant superiority at EITHER 70B or
  Maverick scale.** This is the **truth floor the new axis must beat**, NOT a
  "we gave up" write-up.
* **Lane β (main, possibly LIVE):** verify the latest LCB release + official
  model cutoffs from PRIMARY sources → build the per-model certification rule
  (§ 4) → try to construct a slice certifiably resistant for a STRONGER model
  (§ 5). The ONE earned expensive run (if any) is the cheapest honest Phase-2
  pilot on a certifiably-resistant slice for the strongest honest target (§ 6).
  If no stronger model is certifiable from the latest real data → no-go (§ 7).
* **Lane γ (mandatory, NIM-free):** graphify refresh-at-start (done, HEAD
  `eb8d3ce`) → use for file selection + dependency checks → refresh-at-end;
  extend the W113 tier-2 readiness into a real **per-model certification /
  readiness layer** (`stronger_model_cutoff_certification_v1`); tighten the
  claim surface so the "bounded ceiling unless a NEW certified instrument
  reopens it" position is defensible.

---

## § 2 — latest-release / official-cutoff verification rule (LOCKED)

**Sources are PRIMARY only** (no guessing from memory):

1. **Latest LCB release:** the Hugging Face dataset file tree of
   `livecodebench/code_generation_lite` (the authoritative `testN.jsonl` set)
   + the dataset card. The latest release = the highest-numbered `testN.jsonl`.
2. **Model cutoffs:** official model cards (HF / vendor) + vendor blogs / release
   notes + dataset metadata ONLY. A cutoff is **KNOWN** iff a primary source
   states it explicitly (at month granularity or finer); else **ESTIMATED**
   (inferable from a primary release date with a documented rationale) or
   **UNKNOWN** (no primary source). The certification rule (§ 4) refuses to
   certify resistance against a non-KNOWN cutoff (the W112/W113 lesson).
3. **Reachability is NOT re-probed.** Reachability is not the binding gate — the
   binding gate is `(KNOWN cutoff) ∧ (instrument has ≥30 functional problems
   strictly after it)`. Re-probing reachability has no verdict-changing power
   (it cannot manufacture a KNOWN cutoff or a newer instrument), so it is not
   bought (same discipline as the redundant-pilot no-buy, § 6). The W112
   reachability facts are carried as a fixed prior.

**W114 verification pass (2026-05-29, primary sources):**

| Surface | Primary source | Finding |
|---|---|---|
| Latest LCB release | HF dataset file tree (`test.jsonl`..`test6.jsonl`; highest = `test6.jsonl`) | **`release_v6` is the latest**; no `test7`+. Full release May 2023–Apr 2025; the **functional/lite subset is 63 problems, all 2025-01-11..2025-04-05** (instrument frontier date **2025-04-05**). |
| Llama-4-Maverick cutoff | Official Llama 4 model card (llama.com / Meta GitHub `MODEL_CARD.md`) | **August 2024 — KNOWN.** |
| Qwen3-Coder-480B cutoff | Official HF model card + Qwen blog (`qwenlm.github.io/blog/qwen3-coder`) | **NO CUTOFF STATED — UNKNOWN.** Released 2025-07. |
| DeepSeek-V4-pro cutoff | Official sources | **Not disclosed — UNKNOWN** (V3 ≈ Jul-2024 per non-official extraction; V4 post-dates it). |
| Mistral-Small-4-2603 cutoff | Official Mistral docs / HF model card | **NO CUTOFF STATED — UNKNOWN.** Released 2026-03-16 (post-dates the entire release_v6 window). |

---

## § 3 — instrument-frontier fact (LOCKED, corpus-grounded)

From the SHA-pinned local `release_v6` (`test6.jsonl`; functional subset = 63),
computed NIM-free:

* functional `contest_date` span: **2025-01-11 .. 2025-04-05**.
* month histogram (resistant-for-Maverick, > 2024-08-31): 2025-01 = 14,
  2025-02 = 20, 2025-03 = 27, 2025-04 = 2 (total 63).
* **≥ 30 functional resistant problems require a KNOWN cutoff ≤ ~2025-01-31**
  (cutoff > 2025-01-31 → 49; > 2025-02-28 → 29 < 30; > 2025-03-31 → 2;
  > 2025-04-30 → 0). So the latest functional instrument can certify a
  pilot-grade (≥ 30) resistant slice ONLY for a model whose KNOWN cutoff is
  **January 2025 or earlier**.

---

## § 4 — per-model certification rule (LOCKED — the Lane β / γ instrument)

`coordpy.stronger_model_cutoff_certification_v1` (NIM-free, deterministic,
explicit-import-only; imports the W113 registry + rule, no duplication). For a
candidate `(model, latest-instrument)` pair, `CERTIFIABLE_RESISTANT` ⟺ ALL of:

* **C1** cutoff confidence is `KNOWN` (primary-source-stated). ESTIMATED /
  UNKNOWN ⇒ NOT certifiable (refuse to certify against an unverifiable cutoff).
* **C2** the instrument has ≥ `MIN_RESISTANT_SLICE` (= 30) FUNCTIONAL problems
  with `contest_date` strictly after the KNOWN boundary (ambiguity / missing
  date EXCLUDED, per `partition_resistant_v1`).
* **C3** the model is reachable (fixed prior, § 2) AND strictly stronger than
  the 70B baseline AND same-budget-comparable (non-reasoning plain path).
* **C4** the model is not already settled on the same instrument (a model with a
  recorded resistant verdict on the same slice is not re-certified for a new
  pilot — Maverick, W113).

The **strongest honest target** = the highest-ranked candidate that is
`CERTIFIABLE_RESISTANT`. If none, the certification verdict is
`NO_CERTIFIABLE_STRONGER_MODEL` and Lane β is a no-go (§ 7).

---

## § 5 — newer-instrument slice-construction + exclusion rule (LOCKED)

IF a candidate clears § 4 on the latest available instrument:

1. Pin the release + JSONL SHA-256 (operator-fetch discipline; cross-version
   mixing refused — `livecodebench_loader_v1`).
2. `partition_resistant_v1` against the model's KNOWN boundary; EXCLUDE
   missing / unparseable / not-after-cutoff (typed breakdown).
3. Select the deterministic, outcome-blind difficulty-stratified slice
   (`select_livecodebench_functional_slice_v1`); pin its CID.
4. Run the NIM-free preflight (corpus integrity + executor self-test + loader
   self-test + resistant-partition integrity); `pilot_earned` ⟺ all pass.

A newer release (`release_v7`+) is **only** admitted after the operator fetches
+ SHA-pins it AND the loader's `LIVECODEBENCH_KNOWN_RELEASES` is extended; W114
does NOT fabricate a release that does not exist on the real source.

---

## § 6 — pilot-earning rule if a model becomes certifiably eligible (LOCKED)

* If a STRONGER-than-Maverick model is `CERTIFIABLE_RESISTANT` (§ 4) AND the
  instrument is fetched + preflight-passed (§ 5): the ONE earned expensive run
  is the cheapest honest Phase-2 pilot (1 seed × 30 × K=5 = 330 calls) on its
  resistant slice, mechanism byte-identical to W89/W108/W113, scored by the
  canonical `evaluate_phase2_gates_v1` + MLB-1/MLB-2. A canary (≈ 22 calls)
  precedes the full run. Verdict mapped by the pre-committed cross-scale interp
  rule.
* **Only Maverick certifiable ⇒ NO buy.** Maverick already has a CLEAN resistant
  verdict on the only available functional instrument (W113 +0.00 pp FAIL,
  `EXPOSURE_CONFIRMED`). A second Maverick resistant pilot on the SAME (only)
  instrument has **no verdict-changing power** — it would re-measure a settled
  cell. Redundant ⇒ not bought (W106 margin-cap / "do not buy a redundant run"
  discipline).
* **No stronger model certifiable ⇒ NO buy** (§ 7).

---

## § 7 — no-go rule if no certifiable slice exists (LOCKED — the load-bearing branch)

If § 4 returns `NO_CERTIFIABLE_STRONGER_MODEL` on the latest real data, W114
STOPS honestly with **$0 NIM** and records the blocker as a hard, dated spend
gate (NOT surrender):

* The latest resistant FUNCTIONAL instrument (LCB `release_v6`, functional
  2025-01..04, frontier 2025-04-05) does NOT post-date a single reachable
  stronger-than-Maverick model's verifiable cutoff, and a ≥ 30 functional
  resistant slice requires a KNOWN cutoff ≤ ~Jan-2025 (§ 3).
* Every reachable stronger-than-Maverick frontier model
  (Qwen3-Coder-480B 2025-07, DeepSeek-V4-pro 2025+, Mistral-Small-4 2026-03)
  has an OFFICIALLY UNDISCLOSED cutoff (UNKNOWN ⇒ C1 fails) AND, where
  estimable, a cutoff that meets/post-dates the Apr-2025 frontier (C2 fails) —
  the two gaps COMPOUND.
* Carry-forward (added): `W114-L-RESISTANT-INSTRUMENT-FRONTIER-LAGS-MODEL-
  FRONTIER-CAP` + `W114-T-STRONGER-MODEL-CUTOFFS-OFFICIALLY-UNDISCLOSED`.
* This is **not** "give up" — it is the genuinely-different-axis finding
  (certification-supply analysis) that converts the W113 tier-2 blocker into a
  precise, per-model, dated spend gate, and names exactly what W115 needs.

---

## § 8 — graphify deliverables (LOCKED — Lane γ)

* Refresh at start from HEAD (`graphify update .`; HEAD `eb8d3ce`; **0 token
  cost** — no topology change). **DONE.**
* Use concretely: `explain run_livecodebench_reflexion_bench_v1` /
  `partition_resistant_v1` / `assess_tier2_applicability_v1`;
  `path run_bigcodebench_reflexion_bench_v1 run_livecodebench_reflexion_bench_v1`;
  `affected run_livecodebench_reflexion_bench_v1`; `explain` on the new W114
  certification module/script; `query` for the resistant-bound / ceiling-bound
  claim surfaces.
* Refresh at end after all code/doc changes; confirm the graph is built from the
  W114 HEAD.

---

## § 9 — W115 branch logic (LOCKED — pre-committed)

Selected by the W114 certification verdict:

* **`CERTIFIABLE_STRONGER_MODEL` → pilot ran:** W115 is dictated by the pilot
  verdict via the cross-scale interp rule (PASS_MECHANISM_DRIVEN → a
  contamination-RESISTANT Phase-3 retirement bench at the stronger scale, a
  genuinely new frontier; else de-noise vs accept).
* **`NO_CERTIFIABLE_STRONGER_MODEL` (the expected branch):** the resistant-code
  superiority question is **INSTRUMENT-BLOCKED**, not closed. W115 fires when
  **either** (a) the operator fetches + SHA-pins a newer resistant functional
  instrument (LCB `release_v7`+ or equivalent) with ≥ 30 functional problems
  dated strictly after a reachable frontier model's **KNOWN** cutoff — then run
  the pre-committed cheapest-honest pilot on the strongest such target; **or**
  (b) a reachable frontier model discloses a KNOWN cutoff ≤ the latest
  instrument's functional frontier — then certify + pilot. Until one holds, the
  registered bounded ceiling STANDS and any further resistant-code NIM is
  BLOCKED on the missing instrument. (A genuinely different non-code superiority
  axis may be selected instead; the frozen / closed lines stay closed.)

In ALL branches: **W89 + W105 STAND**; `COO-9` stays lead unless the evidence
forces a different code-line move.

---

## § 10 — Stable boundary preservation (LOCKED)

* `coordpy.__version__ == "0.5.20"`; `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`;
  **no PyPI**; `coordpy/__init__.py` untouched.
* Advanced work explicit-import-only: 1 new module
  (`stronger_model_cutoff_certification_v1`) + 1 script
  (`run_w114_stronger_model_certification_v1.py`); the certification reuses the
  W113 registry + `partition_resistant_v1` + `tier2_readiness_v1` ranking
  (namespace import; no duplication).
* 24th consecutive preflight/earn-discipline validation (W93–W114): runbook
  locked before any NIM; the no-go branch is pre-committed by the rule, so the
  $0 spend is discipline, not omission.

---

## Honest framing

W114 does **multiple** substantial things in one push: it registers the hardened
bounded ceiling as the truth floor (Lane α); it verifies the latest LCB release
and official model cutoffs from PRIMARY sources and builds a real per-model
certification layer that mechanically decides eligibility from real data (Lane
β/γ); and it lands the honest verdict — **no stronger-than-Maverick model is
certifiably resistant on the latest real instrument**, because the resistant
FUNCTIONAL instrument frontier (Apr-2025) has aged out relative to the reachable
frontier-model class whose cutoffs are officially undisclosed. The genuinely
different axis is the certification-supply analysis, not another exposed rerun
or resistant reflexion pilot. If a stronger model were certifiable, W114 would
run the cheapest honest pilot and load W115 immediately; since none is, W114
makes the missing-instrument blocker load-bearing and dated so W115 is real, not
hopeful. A close / confounded / exposed edge is not a win, and the bounded claim
is the truth floor, not surrender.
