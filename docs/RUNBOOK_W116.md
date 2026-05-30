# RUNBOOK — W116

**Upstream instrument-supply ATTACK (not passive waiting) + primary-source
model-cutoff ATTACK + durable upstream-ADMISSION pipeline + one clean pilot ONLY
if a genuinely certifiable new instrument actually exists.**

> Locked **2026-05-30**, BEFORE any NIM call (incl. any reachability re-probe and
> any pilot/canary), per the W93–W115 preflight/earn discipline. A gated branch
> milestone with THREE lanes — **NOT another exposed rerun, NOT another same-scale
> resistant reflexion rerun on the same `release_v6` slice, NOT a bounded-context /
> compaction / token-compression job.** `ultracode` stays OFF. `COO-9` stays the
> lead path.

---

## The one question W116 answers

W115 (live, 2026-05-29) re-verified that the external frontier had not moved: LCB
`release_v6` was still the latest official release and no reachable
stronger-than-Maverick model disclosed a KNOWN cutoff ≤ ~Jan-2025. It shipped a
future-fire pipeline that makes the re-check push-button, and STOPPED at "$0 NIM,
the blocker is supply".

W116 does **not** just re-run that snapshot checker and wait. It asks the active
supply-side question:

> **Can we actively CONSTRUCT the next certifiable post-cutoff functional
> instrument from official UPSTREAM sources — going one level upstream of the
> pinned `release_v6` — instead of passively waiting for someone to publish
> `release_v7`? And has any reachable stronger-than-Maverick model disclosed a
> primary-source KNOWN cutoff since W115?**

If a new official/upstream-admissible instrument exists AND a reachable
stronger-than-Maverick model now has a primary-KNOWN cutoff earlier than that
instrument's problems → build the ingestion/admission path, construct the
certifiable resistant slice, and run the cheapest honest pilot (§ 6). If only
Maverick certifies on a GENUINELY NEW instrument → verdict-changing-power test
(§ 6). If no admissible new supply exists → STOP honestly, make the exact missing
upstream-release / missing primary-cutoff / missing certifiable-slice condition
load-bearing, and STILL land the ingestion/admission pipeline so the next supply
change is immediately usable (§ 7). A close / rumored / aggregator-only / website-
only edge is **NOT** an admissible instrument and **NOT** a win.

---

## Linear

* **`COO-9`** (High, Todo) — "Build a second code benchmark battlefield with lower
  ceiling pressure" — stays the **lead path**. Parent epic **`COO-6`**.
* **`COO-39`** = W115 (Done). Its close pre-commits **W116 fires the moment the
  pipeline trigger flips**: a newer admitted `release_v7`+ with ≥30 post-frontier
  functional problems for a KNOWN-cutoff stronger model, OR a reachable
  stronger-than-Maverick model disclosing a KNOWN cutoff month ≤ 2025-01.
* **W116** = a NEW sub-issue under `COO-6` (sibling of `COO-9`), created at
  milestone end with full results + a COO-9 summary comment (the W105→W115
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
* **No** non-primary / aggregator-only / website-only / rumored instrument or
  cutoff treated as admissible (the W116 admissibility rule § 3 REFUSES them).
* **No** bounded-context / compaction / token-compression / "truncate better"
  drift — those remain anti-patterns, not the frontier path.
* **No** version bump, **no** PyPI publish, `coordpy/__init__.py` untouched.
* `ultracode` stays OFF (a bounded upstream-supply + certification milestone, not a
  repo-wide dynamic-workflow job). Threshold to reconsider: multiple admissible new
  instruments arriving at once / a repo-wide cutoff-certification migration / broad
  multi-surface external verification that cannot be done sequentially in one
  context — none of which W116 requires (the four-surface upstream sweep + four
  model cards were sequential and bounded). If crossed, say so explicitly before
  changing modes.

---

## Operational state (pre-W116 facts, held constant)

* **Two confirmed retirements STAND** — W89 (base HumanEval, +5.56 pp) + W105
  (HumanEval+, +7.00 pp), both `meta/llama-3.3-70b-instruct` @ 70B,
  contamination-EXPOSED HumanEval-family. W116 must not weaken these.
* **Resistant superiority = 0 clean across BOTH scales** (70B −3.33 / +0.00;
  Maverick +0.00; M3 sub-floor at 70B). REGISTERED ceiling (W114).
* Instrument frontier (W113/W114, corpus-grounded): LCB `release_v6` functional
  subset = 63 problems, 2025-01-11..2025-04-05; month histogram 2025-01=14 /
  2025-02=20 / 2025-03=27 / 2025-04=2; **a ≥30 functional resistant slice requires
  a KNOWN cutoff month ≤ 2025-01.** Decision CID `258b6ed7` (W114=W115).
* Reachability (W112 sweep, carried as a fixed prior — NOT re-probed): Maverick =
  200 (tier-1); Qwen3-Coder-480B / DeepSeek-V4-pro / Mistral-Small-4 = reachable
  tier-2; `meta/llama-3.1-405b-instruct` = 404×6.

---

## § 1 — α / β / γ branch logic (LOCKED)

* **Lane α (main, LIVE — upstream instrument-supply ATTACK):** go one level
  upstream of `release_v6`. Re-verify the LCB ecosystem from PRIMARY sources at
  MULTIPLE surfaces (§ 2): the lite dataset file tree, the loader's authoritative
  `ALLOWED_FILES` version list + `release_latest` resolution, the full
  `code_generation` dataset + its README frontier, and the GitHub repo. If an
  admissible NEW post-cutoff functional instrument exists (§ 3) → build the
  ingestion/admission path + construct the certifiable slice (§ 5) → if it certifies
  a stronger-than-Maverick model with a KNOWN cutoff (§ 4) earn + run the cheapest
  honest pilot (§ 6); else no-go (§ 7), but STILL land the admission pipeline.
* **Lane β (mandatory, LIVE — primary-source model-cutoff ATTACK):** re-check
  official cutoff disclosures for Qwen3-Coder-480B, DeepSeek-V4-pro,
  Mistral-Small-4-2603, Maverick, and ANY newly-reachable stronger model from
  PRIMARY sources (§ 2). Build the four-way disclosure-status matrix (§ 4b:
  KNOWN / ESTIMATED-but-unusable / UNKNOWN / contradictory-or-stale). If any
  stronger-than-Maverick model now has a truly KNOWN cutoff ≤ 2025-01, integrate
  it into the certification matrix; else sharpen the blocker.
* **Lane γ (mandatory, NIM-free — admission pipeline + graphify + truth):**
  extend the W115 snapshot-checker into a real upstream-ADMISSION pipeline
  (`coordpy.upstream_instrument_admission_v1`): a pre-committed admissibility rule
  (§ 3), a multi-surface upstream supply snapshot, an upstream-change detector, a
  certifiable-slice builder, the disclosure-status matrix, and an exact W117 fire
  condition — reusing the W113 registry + W114 gate + W115 pipeline (explicit-
  import-only, no duplication). graphify refreshed at start + close; used for file
  selection + dependency checks.

---

## § 2 — official-source + upstream-source verification rule (LOCKED)

**Sources are PRIMARY / upstream-authoritative only** (no guessing from memory; no
third-party aggregator, mirror, leaderboard-site intro, or rumor as authority —
they corroborate at most). Tooling: WebSearch/WebFetch against the official HF
dataset trees + loader scripts + model cards + vendor docs/PDFs + the official LCB
GitHub repo (the documented W113/W114/W115 convention; the chrome browser MCP is
reserved for `/browse` and is NOT used).

1. **Latest instrument supply — checked at FOUR upstream surfaces** (not just the
   lite file tree W115 checked):
   1. the HF file tree of `livecodebench/code_generation_lite` (the authoritative
      `testN.jsonl` set);
   2. the loader script `code_generation_lite.py` `ALLOWED_FILES` (the authoritative
      definition of which `release_vN` exist + what `release_latest` resolves to);
   3. the FULL `livecodebench/code_generation` dataset + its README frontier;
   4. the official LCB GitHub repo (README version list + tags/releases).
   The latest admissible release = the highest `release_vN` that exists as a real
   SHA-pinnable artifact at an authoritative source.
2. **Model cutoffs:** official model cards (HF / vendor) + vendor blogs / release
   notes / official PDFs + dataset metadata ONLY. **KNOWN** iff a PRIMARY source
   states it explicitly (month granularity or finer); **ESTIMATED** iff inferable
   from a primary release date with a documented rationale (NOT certification-
   grade); **UNKNOWN** iff no primary source states it; **CONTRADICTORY/STALE** iff
   only non-primary sources state mutually-inconsistent or carried-over values. The
   certification rule (§ 4) REFUSES to certify resistance against anything but a
   primary-KNOWN cutoff (the W112/W113 lesson).
3. **Reachability is NOT re-probed** (not the binding gate; the binding gate is
   `(primary-KNOWN cutoff) ∧ (admissible instrument has ≥30 functional problems
   strictly after it)`; W112 facts carried).

**W116 LIVE verification pass (2026-05-30, primary sources):**

| Surface | Primary source | Finding (vs W115) |
|---|---|---|
| Lite dataset file tree | HF API tree `livecodebench/code_generation_lite` (`test.jsonl`..`test6.jsonl`; **highest = `test6.jsonl`, 134 MB**; lastModified **2025-06-05**; commit `0fe84c39`) | **`release_v6` STILL latest; no `test7`+. UNCHANGED.** |
| Loader version definition | HF raw `code_generation_lite.py` `ALLOWED_FILES` (`v_list=[v1..v6]`; **`release_latest` → the SAME files as `release_v6`**; `DEFAULT_CONFIG_NAME="release_latest"`) | **No `release_v7`; the upstream "latest" alias resolves to v6. NEW SURFACE — confirms there is no hidden newer-than-v6 supply.** |
| Full dataset | HF `livecodebench/code_generation` tree + README | Single `test.jsonl` (9.4 GB); README **`release_v6` = May 2023–Apr 2025, 1055 problems** (frontier **2025-04-05**). **NEW SURFACE — the full set's frontier is also Apr-2025; no newer-dated problems.** |
| GitHub repo | `LiveCodeBench/LiveCodeBench` README + tags | README tops out at `release_v6` (Apr-2025, 1055); `/tags` shows no releases. **No `release_v7`.** |
| LCB website intro | livecodebench.github.io | Intro text STALE ("May 2023–Feb 2024"); **NOT authoritative** for the dataset frontier ⇒ ignored as a frontier source (the HF dataset + loader are the authority). |
| "Planned v7" search hint | non-primary WebSearch summary | A summary mentioned a "planned v7" through late-2025/early-2026; **contradicted by ALL four primary surfaces; INADMISSIBLE** (no artifact, no SHA, no authoritative confirmation) — recorded as the W117 watch signal, not as supply. |
| Llama-4-Maverick cutoff | Official Llama 4 model card (reconfirmed) | **August 2024 — KNOWN.** Already SETTLED on `release_v6` (W113 ⇒ C4). UNCHANGED. |
| Qwen3-Coder-480B cutoff | Official HF model card | **NO CUTOFF STATED — UNKNOWN.** UNCHANGED. |
| DeepSeek-V4-pro cutoff | Official HF card + DeepSeek V4 card PDF | **NO CUTOFF STATED — UNKNOWN.** 1.6T/49B, 2026. UNCHANGED. |
| Mistral-Small-4-119B-2603 cutoff | **Official Mistral docs model card + official Mistral announcement** (`mistral.ai/news/mistral-small-4`) | **NO CUTOFF STATED (PRIMARY) — UNKNOWN.** SHARPENED: the candidate is now CONFIRMED REAL (119B MoE, released 2026-03-16); the only cutoff figure is an aggregator (OpenRouter) "2025-06", which (a) is non-primary and (b) post-dates the Apr-2025 frontier ⇒ C2-exposed even if it were primary. |
| Mistral-Small-3.2-24B cutoff | aggregator/HF discussion | **KNOWN ~Oct-2023 but 24B (sub-70B) ⇒ C3-blocked** (not a stronger model; deprecated 2026-04-30, replaced by Small 4). Recorded for completeness. |
| Any NEW reachable stronger code model w/ primary-KNOWN cutoff ≤ ~Jan-2025 | Broad official-source sweep | **NONE.** Jan-2025-cutoff models that surfaced (e.g. a Gemini Flash-Lite) are closed/not-reachable/not in the candidate set; reachable open stronger-than-70B models with a KNOWN cutoff ≤ 2025-01 = Maverick only (settled). |

**Net:** all three binding conditions still fail — no newer admissible instrument
(verified at four upstream surfaces, incl. the `release_latest` alias), no reachable
stronger model with a primary-KNOWN cutoff ≤ 2025-01, and the only KNOWN-cutoff
reachable model (Maverick) is settled. The external/upstream frontier has NOT moved
in a verdict-relevant way since W115; the model-disclosure blocker is SHARPER
(Mistral Small 4 confirmed real + primary-no-cutoff).

---

## § 3 — upstream-admissible-instrument rule (LOCKED — the W116 supply gate)

An upstream instrument is **W116-ADMISSIBLE** iff ALL FIVE hold (else REFUSED):

* **A1 authoritative source** — an official HF `livecodebench/*` dataset OR the
  official LCB GitHub repo. An aggregator, mirror, leaderboard-site intro, blog
  rumor, or "planned" announcement is NOT authoritative.
* **A2 dated problems** — each problem carries a `contest_date` (the time-anchor).
* **A3 functional/code-generation-compatible** — has a `starter_code` FUNCTIONAL
  subset the W89 read→solve→execute→reflect mechanism can attack.
* **A4 machine-checkable provenance** — a SHA-256-pinnable JSONL artifact that can
  be operator-fetched + pinned + ADMITTED to `LIVECODEBENCH_KNOWN_RELEASES` (the
  loader refuses cross-version mixing / unpinned corpora).
* **A5 reproducible date histogram** — the functional month histogram can be
  re-derived from the pinned bytes (so the resistant-slice count is verifiable).

A NEW admissible instrument is one that is admissible AND newer than the currently
admitted `release_v6` (a higher `release_vN`, or `release_latest` re-pointing past
v6, or a distinct upstream functional dataset with post-2025-04 dated problems). On
the W116 snapshot: `release_v6` is admissible but ALREADY ADMITTED (not new); the
"planned v7" is REFUSED (A1 + A4 fail); ⇒ **0 admissible NEW instruments.**

---

## § 4 — latest-release detection + per-model certification rule (LOCKED)

Reuse the W115 `run_frontier_certification_v1` (which reuses the W114
`certify_model_v1` / `decide_certification_v1` gate, no duplication) on the
unchanged model/instrument/disclosure state ⇒ the certification decision must
re-derive **byte-identically** (decision CID `258b6ed7`). Per candidate,
`CERTIFIABLE_RESISTANT` ⟺ C1 (primary-KNOWN cutoff) ∧ C2 (≥30 functional problems
strictly after it on an admissible instrument) ∧ C3 (reachable ∧ strictly-stronger-
than-70B ∧ same-budget-comparable) ∧ C4 (not already settled).

### § 4b — per-model disclosure-status rule (LOCKED — Lane β deliverable)

Each candidate is classed by its PRIMARY-source disclosure:

* **KNOWN** — primary source states a cutoff (month or finer). Only KNOWN ≤ 2025-01
  + not-settled + stronger-than-70B can certify on `release_v6`.
* **ESTIMATED-but-unusable** — only a release-date-derived estimate exists; NOT
  certification-grade (C1 fails); if the estimate ≥ Apr-2025, also C2-exposed.
* **UNKNOWN** — no primary source states a cutoff (C1 fails).
* **CONTRADICTORY/STALE** — only non-primary sources, mutually inconsistent or
  carried-over (treated as UNKNOWN for certification; recorded for the audit).

The **strongest honest target** = the highest-ranked `CERTIFIABLE_RESISTANT`
candidate on an admissible instrument. If none → `NO_CERTIFIABLE_STRONGER_MODEL`,
Lane α is a no-go (§ 7).

---

## § 5 — newer-instrument slice-construction + exclusion rule (LOCKED)

IF an admissible NEW instrument (§ 3) clears § 4 for some candidate:

1. Operator-fetch + SHA-256-pin the release + ADMIT it to
   `LIVECODEBENCH_KNOWN_RELEASES` (cross-version mixing refused). W116 does NOT
   fabricate a release that does not exist on the real source.
2. `partition_resistant_v1` against the model's primary-KNOWN boundary; EXCLUDE
   missing / unparseable / not-after-cutoff (typed breakdown).
3. Select the deterministic, outcome-blind difficulty-stratified slice; pin its CID.
4. Run the NIM-free preflight (corpus integrity + executor self-test + loader
   self-test + resistant-partition integrity); `pilot_earned` ⟺ all pass AND
   the resistant slice has ≥ `MIN_RESISTANT_SLICE`=30 problems.

---

## § 6 — pilot-earning rule (LOCKED)

A pilot is earned ONLY if BOTH are true:

1. **Lane α** yields an admissible NEW instrument that is certifiably resistant for
   at least one reachable model (§ 3 + § 5); AND
2. **Lane β** yields a stronger-than-Maverick model with a primary-KNOWN cutoff
   that certifies on that instrument and is NOT redundant / already settled (§ 4).

Then the ONE earned expensive run is the cheapest honest Phase-2 pilot
(1 seed × 30 × K=5 = 330 calls; a ≈22-call canary first), mechanism byte-identical
to W89/W108/W113, scored by `evaluate_phase2_gates_v1` + MLB-1/MLB-2, verdict
mapped by `interpret_cross_scale_resistant_result_v1`.

* **Only Maverick certifiable on a GENUINELY NEW instrument** (a `release_v7`+
  resistant slice it never ran) ⇒ decide verdict-changing power: a Maverick pilot
  is bought ONLY if the NEW instrument is a different resistant slice (W113 settled
  Maverick on `release_v6` ⇒ a SAME-`release_v6` Maverick rerun has no
  verdict-changing power and is NOT bought, per W106 redundant-run discipline).
* **No stronger model certifiable AND/OR no admissible NEW instrument** ⇒ NO buy
  (§ 7).

---

## § 7 — no-go rule if no admissible certifiable slice exists (LOCKED — the load-bearing branch)

If § 3 returns 0 admissible NEW instruments OR § 4 returns
`NO_CERTIFIABLE_STRONGER_MODEL` on the LIVE-verified latest real data, W116 STOPS
honestly with **$0 NIM** and records the blocker as a hard, dated, per-surface +
per-model spend gate (NOT surrender):

* **Instrument side**: the latest admissible functional instrument is UNCHANGED
  (`release_v6`, functional 2025-01..04, frontier 2025-04-05), confirmed at FOUR
  upstream surfaces incl. the `release_latest` alias and the full dataset; no
  `release_v7`+ exists as a SHA-pinnable artifact at any authoritative source; the
  "planned v7" is non-primary and INADMISSIBLE (A1 + A4 fail).
* **Model side**: every reachable stronger-than-Maverick frontier model
  (Qwen3-Coder-480B, DeepSeek-V4-pro, Mistral-Small-4-119B-2603 — the last now
  CONFIRMED REAL with a primary card carrying NO cutoff) has an officially
  UNDISCLOSED cutoff (C1 fails) AND, where estimable, a cutoff at/after the
  Apr-2025 frontier (C2 fails) — the gaps COMPOUND. Maverick (Aug-2024 KNOWN) is
  the only reachable KNOWN cutoff and is already SETTLED on `release_v6` (C4 fails).
* **Carry-forward** (re-affirmed, sharpened): the W114/W115 caps STAND; W116 ADDS
  the four-surface upstream re-verification + the confirmed-real Mistral-Small-4
  finding + the durable upstream-ADMISSION pipeline as the operational discharge.
* This is **not** "give up" — it is the honest aggressive supply-side move: the
  live attack confirms the blocker is real and current at four upstream surfaces
  and on the model side (incl. the now-real Mistral Small 4), and the admission
  pipeline turns the W117 trigger into a push-button operation.

---

## § 8 — graphify deliverables (LOCKED — Lane γ)

* Refresh at start from HEAD (`graphify update .`; HEAD `5b3f75d`; **0 token cost**
  — no topology change). **DONE.**
* Use concretely: `explain run_livecodebench_reflexion_bench_v1` /
  `partition_resistant_v1` / `assess_tier2_applicability_v1` / `certify_model_v1`
  / `run_frontier_certification_v1`; `path run_w113_resistant_pilot.py
  run_livecodebench_reflexion_bench_v1`; `affected run_livecodebench_reflexion_
  bench_v1`; `explain` on the new W116 module/script; `query` for the ceiling-
  bound / certification-bound claim surfaces. **DONE at start.**
* Refresh at end after all code/doc changes; confirm the graph is built from the
  W116 HEAD.

---

## § 9 — W117 branch logic (LOCKED — pre-committed)

Selected by the W116 admission/certification verdict (the upstream-admission
pipeline makes the re-evaluation cheap):

* **`CERTIFIABLE_STRONGER_MODEL` on an admissible NEW instrument → pilot ran:**
  W117 is dictated by the pilot verdict via
  `interpret_cross_scale_resistant_result_v1` (`PASS_MECHANISM_DRIVEN` → a
  contamination-RESISTANT Phase-3 retirement bench at the stronger scale;
  `PASS_NON_MECHANISM_DRIVEN` → de-noise vs accept; `FAIL` → harden the boundary).
* **`NO_CERTIFIABLE_STRONGER_MODEL` (the expected branch):** the resistant-code
  superiority question is **UPSTREAM-SUPPLY + CUTOFF-DISCLOSURE-BLOCKED**, not
  closed. W117 fires the moment the admission pipeline's `detect_upstream_change_v1`
  flags an admissible change — a newer admitted `release_v7`+ (or `release_latest`
  re-pointing past v6, or a new upstream functional dataset) with ≥30 functional
  problems dated strictly after a reachable stronger-than-Maverick model's
  primary-KNOWN cutoff — OR a reachable stronger-than-Maverick model disclosing a
  primary-KNOWN cutoff month ≤ 2025-01. Re-run `run_upstream_admission_v1` against
  the updated snapshot → if any model certifies on an admissible instrument, run the
  pre-committed cheapest-honest pilot on the strongest such target. Until one holds,
  the registered bounded ceiling STANDS and resistant-code NIM is BLOCKED on the
  missing admissible instrument / missing primary cutoff. (A genuinely different
  non-code superiority axis may be selected instead; the frozen / closed lines stay
  closed.)

In ALL branches: **W89 + W105 STAND**; `COO-9` stays lead unless the evidence
forces a different code-line move.

---

## § 10 — Stable boundary preservation (LOCKED)

* `coordpy.__version__ == "0.5.20"`; `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`;
  **no PyPI**; `coordpy/__init__.py` untouched.
* Advanced work explicit-import-only: 1 new module
  (`upstream_instrument_admission_v1`) + 1 script
  (`run_w116_upstream_admission_v1.py`); the module reuses the W113 registry +
  `partition_resistant_v1`-adjacent helpers + the W114 `certify_model_v1` /
  instrument + the W115 `run_frontier_certification_v1` / `frontier_date_summary_v1`
  / `FrontierSnapshotV1` + the loader's `LIVECODEBENCH_KNOWN_RELEASES` (namespace
  import; no duplication).
* 26th consecutive preflight/earn-discipline validation (W93–W116): runbook locked
  before any NIM; the no-go branch is pre-committed by the rule, so the $0 spend is
  discipline, not omission.

---

## Honest framing

W116 does **multiple** substantial things in one push: it ATTACKS the upstream
instrument supply at FOUR authoritative surfaces (lite tree + loader `ALLOWED_FILES`
+ `release_latest` resolution + full dataset + GitHub) rather than passively waiting
for a `release_v7`; it ATTACKS the model-disclosure side from PRIMARY sources and
confirms the last hypothesized tier-2 candidate (Mistral-Small-4-119B-2603) is now a
REAL model whose official card discloses NO cutoff; it OPERATIONALISES the next clean
shot by shipping a durable upstream-ADMISSION pipeline with a pre-committed
admissibility rule, a multi-surface upstream-change detector, a certifiable-slice
builder, a four-way disclosure-status matrix, and an exact W117 fire condition; and
it lands the honest verdict — **no admissible NEW instrument exists and no reachable
stronger-than-Maverick model is certifiably resistant on the latest real data, so $0
NIM.** If a stronger model were certifiable on an admissible instrument, W116 would
run the cheapest honest pilot and load W117 immediately; since none is, W116 makes
the exact missing-upstream-release / missing-primary-cutoff / missing-certifiable-
slice condition load-bearing and turns the W117 trigger into a push-button
operation. A close / rumored / aggregator-only / website-only edge is not an
admissible instrument and not a win; the bounded claim is the registered truth
floor, not surrender. `ultracode` stays OFF; `COO-9` stays lead.
