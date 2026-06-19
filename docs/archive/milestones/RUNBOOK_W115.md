# RUNBOOK — W115

**External-frontier refresh (LIVE) + durable future-fire certification/instrument
pipeline + one clean pilot ONLY if a certifiably-resistant frontier instrument
actually exists.**

> Locked **2026-05-29**, BEFORE any NIM call (incl. any reachability re-probe and
> any pilot/canary), per the W93–W114 preflight/earn discipline. A gated branch
> milestone with THREE lanes — **NOT another exposed rerun, NOT another same-scale
> resistant reflexion rerun on the same `release_v6` slice, NOT a bounded-context /
> compaction / token-compression job.** `ultracode` stays OFF. `COO-9` stays the
> lead path.

---

## The one question W115 answers

W114 established (from primary sources, 2026-05-29) that NO reachable model
stronger than Llama-4-Maverick is certifiably contamination-resistant on the
latest real instrument: the resistant FUNCTIONAL instrument frontier (LCB
`release_v6`, Apr-2025) has aged out relative to the reachable frontier-model
class whose cutoffs are officially undisclosed. Verdict
`NO_CERTIFIABLE_STRONGER_MODEL`, $0 NIM. The current blocker is **instrument /
cutoff supply**, not "run another seed".

W115 does NOT ask "can we find a dirty benchmark and squeeze out a margin?" It
asks the time-sensitive supply question:

> **Has the external world changed enough — a newer official LiveCodeBench release,
> or a newly-disclosed KNOWN cutoff for a reachable stronger model — that a clean,
> certifiably-resistant pilot for a model STRONGER than Maverick is now possible?**

If yes → run the cheapest honest pilot on the strongest honest target. If only
Maverick gets a newer certifiable slice → decide whether a new Maverick pilot has
verdict-changing power (it does not unless the slice is genuinely new; § 6). If
no certifiably-resistant slice exists from the latest official frontier → **stop
honestly, make the missing-instrument/cutoff blocker sharper (not vaguer), and do
NOT buy a dirty run** (§ 7). A close / confounded / exposed edge is **NOT** a win.

This milestone ALSO discharges the supply side itself: it operationalises the W114
certification layer into a durable **future-fire pipeline** so W116 is push-button
the moment external state changes.

---

## Linear

* **`COO-9`** (High, Todo) — "Build a second code benchmark battlefield with lower
  ceiling pressure" — stays the **lead path**. Parent epic **`COO-6`**.
* **`COO-38`** = W114 (Done). Its close pre-commits **W115 = fires ONLY when a
  resistant FUNCTIONAL instrument with ≥30 problems dated strictly after a
  reachable frontier model's KNOWN cutoff exists** (a future LCB `release_v7`+ with
  post-Apr-2025 functional problems AND a frontier model disclosing a KNOWN cutoff
  < those problems); until then the bounded ceiling STANDS and resistant-code NIM
  is BLOCKED.
* **W115** = a NEW sub-issue under `COO-6` (sibling of `COO-9`), created at
  milestone end with full results + a COO-9 summary comment (the W105→W114
  pattern). `linear_github_mapping.json` updated + `sync_linear_github_v1.py`
  validated as part of the close.

---

## What is NOT in scope (anti-drift)

* **No** reopening MBPP+ V2 (W102 cap).
* **No** reopening the frozen cross-modal lines (RealWorldQA frozen at 11B).
* **No** reopening the closed Llama-3.1 rescue branch (W106 NO-GO).
* **No** APPS main-lane NIM (APPS stays the exposed control only).
* **No** 70B resistant reflexion de-noise (W109 rule).
* **No** second Maverick resistant reflexion rerun on the SAME `release_v6`
  instrument (redundant; § 6).
* **No** 405B expensive run unless reachability changes AND a pre-committed gate
  clears (405B is 404×6).
* **No** dirty / contamination-EXPOSED benchmark sold as a frontier win.
* **No** bounded-context / compaction / token-compression / "truncate better"
  drift — those remain anti-patterns, not the frontier path.
* **No** version bump, **no** PyPI publish, `coordpy/__init__.py` untouched.
* `ultracode` stays OFF (this is a bounded frontier-refresh + certification-supply
  milestone, not a repo-wide dynamic-workflow job). Threshold to reconsider:
  multiple official new instruments arriving at once / a repo-wide
  cutoff-certification migration / broad multi-surface external verification at
  once — none of which W115 requires. If crossed, say so explicitly before
  changing modes.

---

## Operational state (pre-W115 facts, held constant)

* **Two confirmed retirements STAND** — W89 (base HumanEval, +5.56 pp) + W105
  (HumanEval+, +7.00 pp), both `meta/llama-3.3-70b-instruct` @ 70B,
  contamination-EXPOSED HumanEval-family. W115 must not weaken these.
* **Resistant superiority = 0 clean across BOTH scales** (70B −3.33 / +0.00;
  Maverick +0.00; M3 sub-floor at 70B). REGISTERED ceiling (W114).
* Reachability (W112 sweep, 2026-05-29, carried as a fixed prior — NOT re-probed;
  § 2): Maverick = 200 (tier-1); Qwen3-Coder-480B / DeepSeek-V4-pro /
  Mistral-Small-4 = reachable tier-2; `meta/llama-3.1-405b-instruct` = 404×6.

---

## § 1 — α / β / γ branch logic (LOCKED)

* **Lane α (main, LIVE):** re-verify the latest LCB release surface + official
  model cutoffs from PRIMARY sources (§ 2) → re-run the per-model certification
  (§ 4) → if a certifiably-resistant slice for a stronger-than-Maverick model now
  exists, earn + run the cheapest honest Phase-2 pilot (§ 5/§ 6); else no-go (§ 7).
* **Lane β (mandatory, NIM-free):** generalise the W114 certification layer into a
  durable future-fire pipeline (`coordpy.frontier_certification_pipeline_v1`):
  latest-official-release detector + frontier-date histogram/summary + certifiable-
  slice candidate builder + per-model go/no-go matrix with exact blocker reasons,
  driven by a `FrontierSnapshotV1` (the external state as DATA). The pipeline runs
  on the live-verified W115 snapshot in this milestone, and makes W116 push-button.
* **Lane γ (mandatory, NIM-free):** graphify refresh-at-start (done, HEAD
  `f8b085d`) → use for file selection + dependency checks → refresh-at-end; tighten
  the claim surface so "the bounded ceiling stands unless a NEW certified
  instrument reopens it" is defensible AND the supply blocker is operational, not
  narrative.

---

## § 2 — official-source verification rule (LOCKED)

**Sources are PRIMARY only** (no guessing from memory; no third-party aggregator
as authority — aggregators corroborate at most):

1. **Latest LCB release:** the Hugging Face dataset file tree of
   `livecodebench/code_generation_lite` (the authoritative `testN.jsonl` set) +
   the dataset card / release commits. The latest release = the highest-numbered
   `testN.jsonl` present in the tree.
2. **Model cutoffs:** official model cards (HF / vendor) + vendor blogs / release
   notes + dataset metadata ONLY. A cutoff is **KNOWN** iff a primary source
   states it explicitly (month granularity or finer); else **ESTIMATED** (inferable
   from a primary release date with a documented rationale) or **UNKNOWN** (no
   primary source). The certification rule (§ 4) REFUSES to certify resistance
   against a non-KNOWN cutoff (the W112/W113 lesson).
3. **Reachability is NOT re-probed.** Reachability is not the binding gate — the
   binding gate is `(KNOWN cutoff) ∧ (instrument has ≥30 functional problems
   strictly after it)`. Re-probing reachability cannot manufacture a KNOWN cutoff
   or a newer instrument, so it has no verdict-changing power and is not bought
   (same discipline as the redundant-pilot no-buy, § 6). The W112 reachability
   facts are carried as a fixed prior.
4. **Tooling:** the live verification uses WebSearch/WebFetch against the official
   HF dataset/model cards + vendor docs/PDFs — the appropriate research primitives
   for fetching official documentation. (The chrome browser-automation MCP is
   reserved for `/browse` per the global guidance and is NOT used.)

**W115 LIVE verification pass (2026-05-29, primary sources):**

| Surface | Primary source | Finding (vs W114) |
|---|---|---|
| Latest LCB release | HF dataset file tree of `livecodebench/code_generation_lite` (`test.jsonl`..`test6.jsonl`; **highest = `test6.jsonl`, 134 MB**; latest commit "add v6" ~1 yr ago; no `test7.jsonl`+) | **`release_v6` STILL latest; no `test7`+. UNCHANGED.** Functional subset 63 problems, 2025-01-11..2025-04-05 (frontier **2025-04-05**). |
| Llama-4-Maverick cutoff | Official Llama 4 model card (model-cards docs; corroborated by multiple sources) | **August 2024 — KNOWN.** UNCHANGED. Already SETTLED on `release_v6` (W113 resistant FAIL). |
| Qwen3-Coder-480B cutoff | Official HF model card (`Qwen/Qwen3-Coder-480B-A35B-Instruct`) — fetched live | **NO CUTOFF STATED — UNKNOWN.** UNCHANGED. Released 2025-07-22; an estimable cutoff is ~2025 (at/after the Apr-2025 frontier ⇒ C2-exposed even if disclosed). |
| DeepSeek-V4-pro cutoff | **Official DeepSeek V4 model card PDF** (`fe-static.deepseek.com/.../deepseek-V4-model-card-EN.pdf`; **published 2026-04-27**; Pro = 1.6T params / 49B activated) — fetched + text-extracted live | **NO "cutoff" string anywhere; no training-data date stated — UNKNOWN.** SHARPENED: W114 said "no card published"; the card now EXISTS (2026-04-27) and STILL discloses no cutoff. A 2026-04 release ⇒ real cutoff ≥2025 ⇒ C2-exposed even if disclosed. |
| Mistral-Small-4-2603 cutoff | Official Mistral docs / HF (real line = Mistral-Small-3.2-2506) | **NO CUTOFF STATED for the candidate — UNKNOWN.** The real reachable Mistral line (Small 3.2, 2025-06) is weaker than Maverick and not the candidate; the 2026-03 "2603" tag post-dates the whole `release_v6` window ⇒ C2-exposed regardless. |
| Any NEW reachable stronger code model w/ KNOWN cutoff ≤ ~Jan-2025 | Broad official-source sweep | **NONE.** Stronger-than-Maverick models surfaced (DeepSeek V4, Qwen 3.5/3.6) are NEWER ⇒ later cutoffs ⇒ C2-exposed; none discloses a KNOWN cutoff ≤ ~Jan-2025. |

**Net:** both binding conditions still fail — no newer instrument, and no reachable
stronger model with a KNOWN cutoff ≤ ~Jan-2025. The external frontier has NOT
changed in a verdict-relevant way since W114.

---

## § 3 — instrument-frontier fact (LOCKED, corpus-grounded)

From the SHA-pinned local `release_v6` (`test6.jsonl`; SHA `bb4c364f…`; functional
subset = 63), re-verified by the pipeline script (`sha_ok` + `histogram_match`):

* functional `contest_date` span **2025-01-11 .. 2025-04-05**.
* month histogram (`YYYY-MM → count`): 2025-01 = 14, 2025-02 = 20, 2025-03 = 27,
  2025-04 = 2 (total 63).
* **≥30 functional resistant problems require a KNOWN cutoff month ≤ 2025-01**
  (> 2025-01 → 49; > 2025-02 → 29 < 30; > 2025-03 → 2; > 2025-04 → 0). So the
  latest functional instrument can certify a pilot-grade (≥30) resistant slice
  ONLY for a model whose KNOWN cutoff is **January 2025 or earlier**.

---

## § 4 — latest-release detection + per-model certification rule (LOCKED)

The W115 pipeline (`coordpy.frontier_certification_pipeline_v1`, NIM-free,
deterministic, explicit-import-only) imports the W113 registry + rule, the W114
`certify_model_v1` / `decide_certification_v1` / instrument, and the loader's
`LIVECODEBENCH_KNOWN_RELEASES`, and adds the supply-chain layer:

* **Latest-release detector** (`detect_latest_release_v1`): given the observed
  release list (from the live HF file tree, recorded in the snapshot) and the
  loader's admitted releases, returns the latest ADMITTED release, the latest
  OBSERVED release, and `newer_release_available` (observed > admitted, i.e. an
  operator-fetch is required to admit it). A newer release is admitted to the
  certification matrix ONLY after the operator fetches + SHA-pins it AND extends
  the loader (§ 5); the pipeline never fabricates a release.
* **Frontier-date summary** (`frontier_date_summary_v1`): the month histogram +
  frontier date + the threshold table (`min cutoff month for ≥N resistant`),
  generalised over ANY instrument's histogram.
* **Per-model certification** (reuses `certify_model_v1`): for each candidate,
  `CERTIFIABLE_RESISTANT` ⟺ C1 (KNOWN cutoff) ∧ C2 (≥`MIN_RESISTANT_SLICE`=30
  functional problems strictly after it on the instrument) ∧ C3 (reachable ∧
  strictly-stronger-than-70B ∧ same-budget-comparable) ∧ C4 (not already settled
  on this instrument).
* **Go/no-go matrix** (`run_frontier_certification_v1(snapshot)`): the full per-
  model table + verdict + exact per-model blocker + the W116 fire condition,
  driven by the `FrontierSnapshotV1` (external state as data). The model
  disclosures in the W115 snapshot MUST match the W113/W114 registry confidences
  (a consistency guard); divergence is flagged.

The **strongest honest target** = the highest-ranked candidate that is
`CERTIFIABLE_RESISTANT`. If none, the verdict is `NO_CERTIFIABLE_STRONGER_MODEL`
and Lane α is a no-go (§ 7).

---

## § 5 — newer-instrument slice-construction + exclusion rule (LOCKED)

IF a candidate clears § 4 on the latest available instrument:

1. Pin the release + JSONL SHA-256 (operator-fetch discipline; cross-version
   mixing refused — `livecodebench_loader_v1`). A newer release (`release_v7`+) is
   admitted ONLY after the operator fetches + SHA-pins it AND
   `LIVECODEBENCH_KNOWN_RELEASES` is extended; W115 does NOT fabricate a release
   that does not exist on the real source.
2. `partition_resistant_v1` against the model's KNOWN boundary; EXCLUDE missing /
   unparseable / not-after-cutoff (typed breakdown).
3. Select the deterministic, outcome-blind difficulty-stratified slice
   (`select_livecodebench_functional_slice_v1`); pin its CID.
4. Run the NIM-free preflight (corpus integrity + executor self-test + loader
   self-test + resistant-partition integrity); `pilot_earned` ⟺ all pass.

---

## § 6 — pilot-earning rule if a model becomes certifiably eligible (LOCKED)

* If a STRONGER-than-Maverick model is `CERTIFIABLE_RESISTANT` (§ 4) AND the
  instrument is fetched + preflight-passed (§ 5): the ONE earned expensive run is
  the cheapest honest Phase-2 pilot (1 seed × 30 × K=5 = 330 calls), mechanism
  byte-identical to W89/W108/W113, scored by the canonical
  `evaluate_phase2_gates_v1` + MLB-1/MLB-2, verdict mapped by
  `interpret_cross_scale_resistant_result_v1`. A canary (≈22 calls) precedes the
  full run.
* **Only Maverick certifiable on the SAME (only) instrument ⇒ NO buy.** Maverick
  already has a CLEAN resistant verdict on `release_v6` (W113 +0.00 pp FAIL,
  `EXPOSURE_CONFIRMED`). A second Maverick pilot on the SAME instrument re-measures
  a settled cell ⇒ no verdict-changing power ⇒ not bought (W106 redundant-run
  discipline). A Maverick pilot is bought ONLY if a GENUINELY NEW instrument
  (a different `release_v7`+ resistant slice it has never run) exists.
* **No stronger model certifiable ⇒ NO buy** (§ 7).

---

## § 7 — no-go rule if no certifiable slice exists (LOCKED — the load-bearing branch)

If § 4 returns `NO_CERTIFIABLE_STRONGER_MODEL` on the LIVE-verified latest real
data, W115 STOPS honestly with **$0 NIM** and records the blocker as a hard, dated,
per-model spend gate (NOT surrender):

* The latest resistant FUNCTIONAL instrument (LCB `release_v6`, functional
  2025-01..04, frontier 2025-04-05) is UNCHANGED — no `release_v7`+ exists on the
  official source as of the live re-check — and does NOT post-date a single
  reachable stronger-than-Maverick model's VERIFIABLE cutoff; a ≥30 functional
  resistant slice requires a KNOWN cutoff ≤ ~Jan-2025 (§ 3).
* Every reachable stronger-than-Maverick frontier model (Qwen3-Coder-480B 2025-07,
  DeepSeek-V4-pro 2026-04 card published but cutoff undisclosed, Mistral-Small-4
  2026-03) has an OFFICIALLY UNDISCLOSED cutoff (UNKNOWN ⇒ C1 fails) AND, where
  estimable, a cutoff that meets/post-dates the Apr-2025 frontier (C2 fails) — the
  gaps COMPOUND.
* Maverick (Aug-2024 KNOWN) is the only reachable model with a KNOWN cutoff and is
  already SETTLED on `release_v6` (C4 fails); no new instrument exists for it.
* Carry-forward (re-affirmed, sharpened): the W114 caps STAND
  (`W114-L-RESISTANT-INSTRUMENT-FRONTIER-LAGS-MODEL-FRONTIER-CAP` +
  `W114-T-STRONGER-MODEL-CUTOFFS-OFFICIALLY-UNDISCLOSED`); W115 ADDS the
  live-re-verification + the future-fire pipeline as the operational discharge
  (`W115-L-EXTERNAL-FRONTIER-UNCHANGED-NO-CERTIFIABLE-SLICE-REVERIFIED-CAP` +
  `W115-T-FUTURE-FIRE-CERTIFICATION-PIPELINE-SHIPS`).
* This is **not** "give up" — it is the honest aggressive supply-side move: the
  live re-check confirms the blocker is real and current (incl. against the
  brand-new DeepSeek V4 card), and the pipeline turns the W116 trigger into a
  push-button operation.

---

## § 8 — graphify deliverables (LOCKED — Lane γ)

* Refresh at start from HEAD (`graphify update .`; HEAD `f8b085d`; **0 token cost**
  — no topology change). **DONE.**
* Use concretely: `explain run_livecodebench_reflexion_bench_v1` /
  `partition_resistant_v1` / `assess_tier2_applicability_v1` /
  `certify_model_v1` (the W114 gate); `path run_w113_resistant_pilot.py
  run_livecodebench_reflexion_bench_v1`; `affected run_livecodebench_reflexion_
  bench_v1`; `explain` on the new W115 pipeline module/script; `query` for the
  ceiling-bound / certification-bound claim surfaces.
* Refresh at end after all code/doc changes; confirm the graph is built from the
  W115 HEAD.

---

## § 9 — W116 branch logic (LOCKED — pre-committed)

Selected by the W115 certification verdict (and the push-button pipeline makes the
re-evaluation cheap):

* **`CERTIFIABLE_STRONGER_MODEL` → pilot ran:** W116 is dictated by the pilot
  verdict via `interpret_cross_scale_resistant_result_v1`
  (`PASS_MECHANISM_DRIVEN` → a contamination-RESISTANT Phase-3 retirement bench at
  the stronger scale, a genuinely new frontier; `PASS_NON_MECHANISM_DRIVEN` →
  de-noise vs accept; `FAIL` → harden the boundary at the stronger scale).
* **`NO_CERTIFIABLE_STRONGER_MODEL` (the expected branch):** the resistant-code
  superiority question is **INSTRUMENT/CUTOFF-SUPPLY-BLOCKED**, not closed. W116
  fires the moment the pipeline's `newer_release_available` flips true (operator
  fetches + SHA-pins + admits a `release_v7`+ with ≥30 post-Apr-2025 functional
  problems for a model with a KNOWN cutoff < those problems) OR a reachable
  stronger-than-Maverick model discloses a KNOWN cutoff ≤ the instrument's
  functional frontier. Re-run `run_frontier_certification_v1` against the updated
  snapshot → if any model certifies, run the pre-committed cheapest-honest pilot on
  the strongest such target. Until one holds, the registered bounded ceiling STANDS
  and resistant-code NIM is BLOCKED on the missing instrument. (A genuinely
  different non-code superiority axis may be selected instead; the frozen / closed
  lines stay closed.)

In ALL branches: **W89 + W105 STAND**; `COO-9` stays lead unless the evidence
forces a different code-line move.

---

## § 10 — Stable boundary preservation (LOCKED)

* `coordpy.__version__ == "0.5.20"`; `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`;
  **no PyPI**; `coordpy/__init__.py` untouched.
* Advanced work explicit-import-only: 1 new module
  (`frontier_certification_pipeline_v1`) + 1 script
  (`run_w115_frontier_certification_v1.py`); the pipeline reuses the W113 registry +
  `partition_resistant_v1` + the W114 `certify_model_v1` / `decide_certification_v1`
  / instrument + the loader's `LIVECODEBENCH_KNOWN_RELEASES` (namespace import; no
  duplication).
* 25th consecutive preflight/earn-discipline validation (W93–W115): runbook locked
  before any NIM; the no-go branch is pre-committed by the rule, so the $0 spend is
  discipline, not omission.

---

## Honest framing

W115 does **multiple** substantial things in one push: it RE-VERIFIES the external
frontier LIVE from primary sources (latest LCB release + four model cutoffs incl.
the brand-new DeepSeek V4 card), confirming the supply blocker is real and current,
not a stale carry-forward; it OPERATIONALISES the certification supply chain into a
durable future-fire pipeline that makes W116 push-button; and it lands the honest
verdict — **no stronger-than-Maverick model is certifiably resistant on the latest
real instrument, and the external frontier has not moved in a verdict-relevant way
since W114.** If a stronger model were certifiable, W115 would run the cheapest
honest pilot and load W116 immediately; since none is, W115 makes the
missing-instrument/cutoff blocker operational and dated so W116 is real, not
hopeful. A close / confounded / exposed edge is not a win, and the bounded claim is
the registered truth floor, not surrender. `ultracode` stays OFF; `COO-9` stays
lead.
