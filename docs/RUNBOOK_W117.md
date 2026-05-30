# RUNBOOK — W117

**Upstream-DERIVED instrument CONSTRUCTION attack (not just packaged-release
detection) + deeper primary-source model-cutoff attack + durable
construction/admission pipeline + one clean pilot ONLY if a genuinely
construction-admissible new instrument is actually certifiable.**

> Locked **2026-05-30**, BEFORE any NIM call (incl. any reachability re-probe and
> any pilot/canary), per the W93–W116 preflight/earn discipline. A gated branch
> milestone with THREE lanes — **NOT another exposed rerun, NOT another same-scale
> resistant reflexion rerun on the same `release_v6` slice, NOT a "watch-and-wait"
> snapshot re-check, NOT a bounded-context / compaction / token-compression job.**
> `ultracode` stays OFF. `COO-9` stays the lead path.

---

## The one question W117 answers

W116 (live, 2026-05-30) ATTACKED the upstream instrument SUPPLY at FOUR authoritative
surfaces and the model-disclosure side from PRIMARY sources, shipped a durable
upstream-ADMISSION pipeline (`coordpy.upstream_instrument_admission_v1`), and STOPPED
honestly at "$0 NIM — no packaged `release_v7` exists and no reachable stronger model
has a primary-KNOWN cutoff ≤ 2025-01". W116 answered **"is there an admissible
*packaged* release_v7?"** (No).

W117 asks the harder, more aggressive **construction** question:

> **Even before a numbered `release_v7` is packaged, can we CONSTRUCT an
> upstream-authoritative, machine-checkable, certifiable post-cutoff functional
> instrument from official upstream PROVENANCE — the HF dataset revision/commit/
> discussion history, the official LCB GitHub data-generation pipeline, and the
> actual upstream contest provenance LCB draws from? And has any reachable
> stronger-than-Maverick model disclosed a primary-source KNOWN cutoff since W116
> when probed DEEPER than a single card (official PDFs / release notes / vendor
> docs)?**

If an upstream-DERIVED construction-admissible instrument can be built AND a
reachable stronger-than-Maverick model has a primary-KNOWN cutoff earlier than its
problems → build the construction/ingestion path, construct the certifiable
resistant slice, and run the cheapest honest pilot (§ 6). If only Maverick certifies
on a GENUINELY NEW instrument → verdict-changing-power test (§ 6). If no
construction-admissible supply exists → STOP honestly, make the **exact missing
upstream provenance artifact** load-bearing, and STILL land the
construction/admission machinery so the next provenance change is immediately usable
(§ 7). A raw-contest hand-assembly / aggregator / rumor / website-only edge is **NOT
a construction-admissible instrument** and **NOT a win** (§ 3 REFUSES it).

This lane is **not complete if it only says "still no v7."** It must either construct
an upstream-derived candidate or prove **precisely** why that is not yet possible.

---

## Linear

* **`COO-9`** (High, Todo) — "Build a second code benchmark battlefield with lower
  ceiling pressure" — stays the **lead path**. Parent epic **`COO-6`**.
* **`COO-40`** = W116 (Done). Its close pre-commits **W117 fires the moment
  `detect_upstream_change_v1` flags an admissible change**: a newer admitted
  `release_v7`+ (or `release_latest` re-point / new upstream functional dataset) with
  ≥30 post-frontier functional problems for a KNOWN-cutoff stronger model, OR a
  reachable stronger-than-Maverick model disclosing a primary-KNOWN cutoff ≤ 2025-01.
* **W117** = a NEW sub-issue under `COO-6` (sibling of `COO-9`), created at milestone
  end with full results + a COO-9 summary comment (the W105→W116 pattern).
  `linear_github_mapping.json` updated + `sync_linear_github_v1.py` validated as part
  of the close.

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
* **No** raw-contest hand-assembled / aggregator-only / website-only / rumored
  instrument or cutoff treated as construction-admissible (the W117 construction rule
  § 3 REFUSES them via B1 + B2).
* **No** bounded-context / compaction / token-compression / "truncate better" drift —
  those remain anti-patterns, not the frontier path.
* **No** version bump, **no** PyPI publish, `coordpy/__init__.py` untouched.
* `ultracode` stays OFF (a bounded upstream construction + certification milestone,
  not a repo-wide dynamic-workflow job). Threshold to reconsider: multiple
  construction-admissible candidate instruments arriving at once / a repo-wide
  cutoff-certification migration / broad multi-surface external verification that
  cannot be done sequentially in one context — none of which W117 requires (the
  eight-surface provenance sweep + four model cards were sequential and bounded). If
  crossed, say so explicitly before changing modes.

---

## Operational state (pre-W117 facts, held constant)

* **Two confirmed retirements STAND** — W89 (base HumanEval, +5.56 pp) + W105
  (HumanEval+, +7.00 pp), both `meta/llama-3.3-70b-instruct` @ 70B,
  contamination-EXPOSED HumanEval-family. W117 must not weaken these.
* **Resistant superiority = 0 clean across BOTH scales** (70B −3.33 / +0.00;
  Maverick +0.00; M3 sub-floor at 70B). REGISTERED ceiling (W114).
* Instrument frontier (W113/W114, corpus-grounded): LCB `release_v6` functional
  subset = 63 problems, 2025-01-11..2025-04-05; month histogram 2025-01=14 /
  2025-02=20 / 2025-03=27 / 2025-04=2; **a ≥30 functional resistant slice requires a
  KNOWN cutoff month ≤ 2025-01.** Decision CID `258b6ed7` (W114=W115=W116).
* Reachability (W112 sweep, carried as a fixed prior — NOT re-probed): Maverick =
  reachable tier-1; Qwen3-Coder-480B / DeepSeek-V4-pro / Mistral-Small-4 = reachable
  tier-2; `meta/llama-3.1-405b-instruct` = 404×6.

---

## § 1 — α / β / γ branch logic (LOCKED)

* **Lane α (main, LIVE — upstream-DERIVED instrument CONSTRUCTION attack):** go
  BEYOND release labels. Examine the actual upstream PROVENANCE supply (§ 2): the HF
  dataset revision/commit log, the HF refs (branches/tags), the HF discussions/PRs,
  the official LCB GitHub commit history + tags, the GitHub repo's data-generation
  pipeline structure, the dataset README provenance description, and the runner's
  data-loading path. Pre-commit the upstream-DERIVED construction rule (§ 3). Attempt
  to CONSTRUCT a shadow/post-v6 functional candidate instrument from authoritative
  provenance. If a construction-admissible candidate exists → build the
  ingestion/construction path + compute the date frontier + month histogram +
  construct the certifiable slice (§ 5) → if it certifies a stronger-than-Maverick
  model with a KNOWN cutoff (§ 4) earn + run the cheapest honest pilot (§ 6); else
  no-go (§ 7), but STILL land the construction/admission pipeline AND record the EXACT
  missing upstream provenance artifact.
* **Lane β (mandatory, LIVE — deeper primary-source model-cutoff attack):** re-check
  official cutoff disclosures for Qwen3-Coder-480B, DeepSeek-V4-pro,
  Mistral-Small-4-2603, Maverick, and ANY newly-reachable stronger model from PRIMARY
  sources, searching DEEPER than a single card where needed (official model cards +
  release notes + official PDFs + vendor docs/blogs). Build the sharpened
  disclosure-status matrix (§ 4b: KNOWN / ESTIMATED-but-unusable / UNKNOWN /
  contradictory-or-stale / newly-disclosed-since-W116). If any stronger-than-Maverick
  model now has a truly primary-KNOWN cutoff ≤ 2025-01, integrate it into the
  certification matrix immediately; else sharpen the blocker.
* **Lane γ (mandatory, NIM-free — construction/admission pipeline + graphify +
  truth):** extend the W116 admission pipeline into a real upstream-DERIVED
  CONSTRUCTION path (`coordpy.upstream_derived_instrument_construction_v1`): a
  pre-committed construction-provenance rule (§ 3), a multi-surface provenance
  snapshot, a candidate-instrument constructor, a provenance validator, a
  certifiable-slice builder, the per-model disclosure matrix, and an exact W118 fire
  condition — reusing the W113 registry + W114 gate + W115 pipeline + W116 admission
  layer (explicit-import-only, no duplication). graphify refreshed at start + close;
  used for file selection + dependency checks. **Land executable code/script assets,
  not just docs.**

---

## § 2 — official-source + upstream-PROVENANCE verification rule (LOCKED)

**Sources are PRIMARY / upstream-authoritative only** (no guessing from memory; no
third-party aggregator, mirror, leaderboard-site intro, or rumor as authority — they
corroborate at most). Tooling: WebSearch/WebFetch against the official HF dataset
trees + commit/ref/discussion APIs + loader scripts + model cards + vendor docs/PDFs +
the official LCB GitHub repo (the documented W113–W116 convention; the chrome browser
MCP is reserved for `/browse` and is NOT used).

**The W117 escalation over W116:** W116 verified the latest *packaged release* at four
surfaces (lite file tree / loader `ALLOWED_FILES` / full dataset README / GitHub
README). W117 attacks the *construction provenance* at EIGHT surfaces — the revision
history and collection mechanism, not just the release label:

| # | Provenance surface | Primary source | W117 LIVE finding (2026-05-30) |
|---|---|---|---|
| 1 | HF dataset commit/revision log | `api/datasets/livecodebench/code_generation_lite/commits/main` | 20 commits; latest data-bearing = **"add v6" 2025-04-21**; HEAD = "fix typos (#4)" 2025-06-05. **No post-v6 data commit; no `test7`.** |
| 2 | HF refs (branches/tags) | `…/refs` | **1 branch (`main`), 0 tags.** No staging / preview / v7 branch. |
| 3 | HF discussions / PRs | dataset `/discussions` | Newest = "LCB pull request" #14 + "Clarification on v6 size (454 vs 175)" #13. **No v7 / newer-data thread.** |
| 4 | LCB GitHub commit history | `api/.../commits` | Newest **2025-07-16** ("fix: typo in Explorer URL"); recent activity = runner maintenance (model infos, gemini, ERRATA). **No data-collection commit; no v7.** |
| 5 | LCB GitHub tags/releases | `api/.../tags` | **Empty (0 tags).** No `release_v7`. |
| 6 | GitHub repo data-pipeline structure | repo root contents | Top-level = `lcb_runner/` + `assets/` + configs. **NO scraper / data / collect / contest directory** ⇒ the public repo is the runner/eval harness, NOT a published collection pipeline. |
| 7 | Dataset README provenance | dataset `raw/main/README.md` | Provenance = LeetCode / AtCoder / Codeforces contests; each `release_vN` a temporal snapshot. **Documents ONLY loading published releases — NO generation tool / script / manifest** for new problems. |
| 8 | Runner data-loading path | `lcb_runner/benchmarks/code_generation.py` | Loads **exclusively** via `load_dataset("livecodebench/code_generation_lite", version_tag=…)`; **no local scraping / construction.** |

**Model cutoffs (Lane β, DEEPER pass):** official model cards (HF / vendor) + vendor
release notes / blogs / **official PDFs** + dataset metadata ONLY. **KNOWN** iff a
PRIMARY source states it explicitly (month granularity or finer); **ESTIMATED** iff
inferable from a primary release date (NOT certification-grade); **UNKNOWN** iff no
primary source states it; **CONTRADICTORY/STALE** iff only non-primary sources state
mutually-inconsistent or carried-over values. The certification rule (§ 4) REFUSES to
certify resistance against anything but a primary-KNOWN cutoff (the W112/W113 lesson).

**W117 LIVE Lane β pass (2026-05-30, primary sources):**

| Model | Primary source (verbatim probe) | Disclosure | Blocker |
|---|---|---|---|
| `meta/llama-4-maverick-17b-128e-instruct` | Meta `llama-models/.../llama4/MODEL_CARD.md` — **"August 2024"** (verbatim) | **KNOWN** | **C4** (settled on `release_v6`; W113) |
| `qwen/qwen3-coder-480b-a35b-instruct` | Official HF card raw README — **NO CUTOFF STATED** | **UNKNOWN** | C1 |
| `deepseek-ai/deepseek-v4-pro` | **Official DeepSeek V4 model-card PDF** (`fe-static.deepseek.com/…/deepseek-V4-model-card-EN.pdf`) — **NO CUTOFF STATED**; aggregator "Apr 2026" is non-primary | **UNKNOWN** (primary) | C1; aggregator figure (Apr-2026) post-dates frontier ⇒ C2-exposed |
| `mistralai/mistral-small-4-119b-2603` | Official Mistral docs models overview — **NO CUTOFF STATED**; aggregator "2025-06" non-primary | **UNKNOWN** (primary) | C1; aggregator figure post-dates frontier ⇒ C2-exposed |
| (broad sweep) any reachable stronger-than-70B open model w/ primary-KNOWN cutoff ≤ 2025-01 | official-source sweep (Qwen3.5 / GLM-5 / Kimi / Gemma 4 / DeepSeek R1) | **NONE** | listicle/aggregator only; none reachable+stronger with primary-KNOWN ≤ 2025-01 |

**Net:** all three binding conditions still fail — no construction-admissible
instrument (verified at EIGHT provenance surfaces incl. the revision log, refs,
discussions, and the absent collection pipeline), no reachable stronger model with a
primary-KNOWN cutoff ≤ 2025-01, and the only KNOWN-cutoff reachable model (Maverick)
is settled. The DeepSeek probe is SHARPENED: the official V4 PDF re-checked at primary
still states no cutoff (the "Apr 2026" figure is aggregator-only AND C2-exposed) — so
DeepSeek and Mistral now share the identical "UNKNOWN-from-primary +
C2-exposed-aggregator-figure" pattern.

---

## § 3 — upstream-DERIVED construction-admissible-instrument rule (LOCKED — the W117 supply gate)

W116's A1..A5 governs admitting a *packaged* release. W117 adds the CONSTRUCTION
dimension: when there is no packaged release, may a candidate be DERIVED from
authoritative upstream provenance? A constructed candidate is
**W117-CONSTRUCTION-ADMISSIBLE** iff ALL of A1..A5 (reused, unchanged) AND BOTH B1, B2
hold, AND it post-dates the admitted instrument:

* **A1 authoritative source** — official HF `livecodebench/*` dataset OR official LCB
  GitHub repo (reused).
* **A2 dated problems** — each carries a `contest_date` (reused).
* **A3 functional/code-generation-compatible** — has a `starter_code` FUNCTIONAL
  subset the W89 mechanism can attack (reused).
* **A4 machine-checkable provenance** — a SHA-256-pinnable JSONL artifact that can be
  operator-fetched + pinned + admitted to `LIVECODEBENCH_KNOWN_RELEASES` (reused).
* **A5 reproducible date histogram** — the functional month histogram re-derivable
  from the pinned bytes (reused).
* **B1 authoritative construction PROVENANCE** — the constructed candidate's problem
  set is fully defined by an LCB-PUBLISHED provenance artifact: a dataset
  revision/commit/PR that adds the problems, OR the LCB repo's PUBLISHED
  data-generation pipeline + a PUBLISHED problem-id/URL manifest. A raw-contest-site
  scrape, an operator hand-selection, or any non-LCB-published assembly is NOT
  authoritative construction provenance.
* **B2 no operator curation (reproducibility)** — the selection AND ordering of
  problems is fully determined by the published provenance, so anyone re-running it
  obtains the byte-identical set; the operator contributes NO discretionary choice.
  This is the anti-cherry-pick / no-"vibes-based" criterion.

A NEW construction-admissible instrument is one that is construction-admissible AND
newer than the currently admitted `release_v6` (post-2025-04 functional problems).

**On the W117 LIVE provenance snapshot (§ 2):** NO surface carries a post-v6
LCB-published artifact (no post-v6 dataset commit/revision/ref/PR; no published
collection pipeline; no problem-id manifest). The ONLY post-v6 path is a raw-contest
hand-assembly, which is **CONSTRUCTION-INADMISSIBLE** (fails B1 + B2 even though a
hand-assembled JSONL could be made dated/functional/SHA-pinnable). ⇒ **0
construction-admissible NEW instruments.** The exact missing artifact is load-bearing:
*an LCB-published post-v6 provenance artifact (a dataset revision/commit/PR adding
post-2025-04 functional problems, OR a published collection pipeline + problem-id
manifest enabling reproducible construction).*

---

## § 4 — latest-release detection + per-model certification rule (LOCKED)

Reuse the W116 `run_upstream_admission_v1` (which reuses W115
`run_frontier_certification_v1` → W114 `certify_model_v1` / `decide_certification_v1`,
no duplication) on the unchanged model/instrument/disclosure state ⇒ the certification
decision must re-derive **byte-identically** (decision CID `258b6ed7`). Per candidate,
`CERTIFIABLE_RESISTANT` ⟺ C1 (primary-KNOWN cutoff) ∧ C2 (≥30 functional problems
strictly after it on an admissible instrument) ∧ C3 (reachable ∧ strictly-stronger-
than-70B ∧ same-budget-comparable) ∧ C4 (not already settled).

### § 4b — per-model disclosure-status rule (LOCKED — Lane β deliverable)

Each candidate is classed by its PRIMARY-source disclosure: **KNOWN** (primary states
a month-or-finer cutoff; only KNOWN ≤ 2025-01 + not-settled + stronger-than-70B can
certify on `release_v6`); **ESTIMATED-but-unusable** (release-date-derived only; C1
fails; if ≥ Apr-2025, also C2-exposed); **UNKNOWN** (no primary cutoff; C1 fails);
**CONTRADICTORY/STALE** (only non-primary, mutually inconsistent / carried-over —
treated as UNKNOWN for certification, recorded for audit); **NEWLY-DISCLOSED-SINCE-
W116** (a primary cutoff that appeared since W116 — integrate immediately).

The **strongest honest target** = the highest-ranked `CERTIFIABLE_RESISTANT` candidate
on a construction-admissible instrument. If none → `NO_CERTIFIABLE_STRONGER_MODEL`,
Lane α is a no-go (§ 7).

---

## § 5 — construction + slice-construction + exclusion rule (LOCKED)

IF a construction-admissible NEW instrument (§ 3) clears § 4 for some candidate:

1. **Construct/ingest** the candidate from its LCB-published provenance artifact;
   operator-fetch + SHA-256-pin + ADMIT it to `LIVECODEBENCH_KNOWN_RELEASES`
   (cross-version mixing refused). W117 does NOT fabricate problems and does NOT
   hand-assemble from raw contest sites.
2. Compute the **date frontier + functional month histogram** from the pinned bytes
   (A5); re-derive the resistant count.
3. `partition_resistant_v1` against the model's primary-KNOWN boundary; EXCLUDE
   missing / unparseable / not-after-cutoff (typed breakdown).
4. Select the deterministic, outcome-blind difficulty-stratified slice; pin its CID.
5. Run the NIM-free preflight (corpus integrity + executor self-test + loader
   self-test + resistant-partition integrity); `pilot_earned` ⟺ all pass AND the
   resistant slice has ≥ `MIN_RESISTANT_SLICE`=30 problems.

---

## § 6 — pilot-earning rule (LOCKED)

A pilot is earned ONLY if BOTH are true:

1. **Lane α** yields a construction-admissible NEW instrument that is certifiably
   resistant for at least one reachable model (§ 3 + § 5); AND
2. **Lane β** yields a stronger-than-Maverick model with a primary-KNOWN cutoff that
   certifies on that instrument and is NOT redundant / already settled (§ 4).

Then the ONE earned expensive run is the cheapest honest Phase-2 pilot
(1 seed × 30 × K=5 = 330 calls; a ≈22-call canary first), mechanism byte-identical to
W89/W108/W113, scored by `evaluate_phase2_gates_v1` + MLB-1/MLB-2, verdict mapped by
`interpret_cross_scale_resistant_result_v1`.

* **Only Maverick certifiable on a GENUINELY NEW instrument** (a constructed post-v6
  resistant slice it never ran) ⇒ decide verdict-changing power: a Maverick pilot is
  bought ONLY if the NEW instrument is a different resistant slice (W113 settled
  Maverick on `release_v6` ⇒ a SAME-`release_v6` Maverick rerun has no
  verdict-changing power and is NOT bought, per W106 redundant-run discipline).
* **No stronger model certifiable AND/OR no construction-admissible NEW instrument**
  ⇒ NO buy (§ 7).

---

## § 7 — no-go rule if no construction-admissible certifiable slice exists (LOCKED — the load-bearing branch)

If § 3 returns 0 construction-admissible NEW instruments OR § 4 returns
`NO_CERTIFIABLE_STRONGER_MODEL` on the LIVE-verified latest real data, W117 STOPS
honestly with **$0 NIM** and records the blocker as a hard, dated, per-surface +
per-model spend gate (NOT surrender):

* **Instrument/construction side**: no post-v6 LCB-published provenance artifact
  exists at ANY of the EIGHT surfaces (revision log, refs, discussions, GitHub
  commits, GitHub tags, repo pipeline structure, README provenance, runner loader);
  the only post-v6 path (raw-contest hand-assembly) is CONSTRUCTION-INADMISSIBLE
  (B1 + B2 fail). The exact missing artifact is named (§ 3).
* **Model side**: every reachable stronger-than-Maverick frontier model
  (Qwen3-Coder-480B, DeepSeek-V4-pro, Mistral-Small-4-119B-2603) has an officially
  UNDISCLOSED cutoff from PRIMARY sources (C1 fails) AND, where an aggregator figure
  exists, a figure at/after the Apr-2025 frontier (C2-exposed). Maverick (Aug-2024
  KNOWN, primary-reconfirmed verbatim) is the only reachable KNOWN cutoff and is
  already SETTLED on `release_v6` (C4 fails).
* **Carry-forward** (re-affirmed, sharpened): the W114/W115/W116 caps STAND; W117 ADDS
  the eight-surface construction-provenance verification + the DeepSeek primary-PDF
  re-confirmation + the durable construction/admission pipeline + the construction
  rule (B1 + B2) as the operational discharge.
* This is **not** "give up" — it is the honest aggressive construction-side move: the
  live attack confirms a post-v6 instrument cannot be *constructed* from authoritative
  provenance (only hand-curated, which is refused), and the construction pipeline
  turns the W118 trigger into a push-button operation.

---

## § 8 — graphify deliverables (LOCKED — Lane γ)

* Refresh at start from HEAD (`graphify update .`; HEAD `dcec243`; **0 token cost** —
  no topology change). **DONE.**
* Use concretely: `explain assess_instrument_admissibility_v1 / detect_upstream_change_v1
  / run_upstream_admission_v1 / certify_model_v1 / partition_resistant_v1`;
  `path run_w113_resistant_pilot.py run_livecodebench_reflexion_bench_v1`;
  `affected run_livecodebench_reflexion_bench_v1`; `explain` on the new W117
  construction module/script; `query` for the ceiling-bound / construction-bound claim
  surfaces. **DONE at start.**
* Refresh at end after all code/doc changes; confirm the graph is built from the W117
  HEAD.

---

## § 9 — W118 branch logic (LOCKED — pre-committed)

Selected by the W117 construction/admission verdict (the construction pipeline makes
the re-evaluation cheap):

* **`CERTIFIABLE_STRONGER_MODEL` on a construction-admissible NEW instrument → pilot
  ran:** W118 is dictated by the pilot verdict via
  `interpret_cross_scale_resistant_result_v1` (`PASS_MECHANISM_DRIVEN` → a
  contamination-RESISTANT Phase-3 retirement bench at the stronger scale;
  `PASS_NON_MECHANISM_DRIVEN` → de-noise vs accept; `FAIL` → harden the boundary).
* **`NO_CERTIFIABLE_STRONGER_MODEL` (the expected branch):** the resistant-code
  superiority question is **UPSTREAM-CONSTRUCTION-PROVENANCE + CUTOFF-DISCLOSURE-
  BLOCKED**, not closed. W118 fires the moment the construction pipeline's detector
  flags an admissible change — (a) a packaged `release_v7`+ admitted to the loader, OR
  (b) an LCB-PUBLISHED post-v6 construction provenance (a dataset revision/commit/PR
  adding post-2025-04 functional problems, OR a published collection pipeline +
  problem-id manifest) enabling a B1∧B2-admissible ≥30 functional post-cutoff slice,
  OR (c) a reachable stronger-than-Maverick model disclosing a primary-KNOWN cutoff
  month ≤ 2025-01. Re-run `run_upstream_construction_v1` against the updated snapshot →
  if any model certifies on a construction-admissible instrument, run the pre-committed
  cheapest-honest pilot on the strongest such target. Until one holds, the registered
  bounded ceiling STANDS and resistant-code NIM is BLOCKED on the missing
  construction-provenance artifact / missing primary cutoff. (A genuinely different
  non-code superiority axis may be selected instead; the frozen / closed lines stay
  closed.)

In ALL branches: **W89 + W105 STAND**; `COO-9` stays lead unless the evidence forces a
different code-line move.

---

## § 10 — Stable boundary preservation (LOCKED)

* `coordpy.__version__ == "0.5.20"`; `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`;
  **no PyPI**; `coordpy/__init__.py` untouched.
* Advanced work explicit-import-only: 1 new module
  (`upstream_derived_instrument_construction_v1`) + 1 script
  (`run_w117_upstream_construction_v1.py`); the module reuses the W113 registry +
  `partition_resistant_v1` + the W114 `certify_model_v1` / instrument + the W115
  `run_frontier_certification_v1` + the W116 `run_upstream_admission_v1` /
  `assess_instrument_admissibility_v1` / `UpstreamInstrumentObservationV1` /
  `AdmissibilityRuleV1` + the loader's `LIVECODEBENCH_KNOWN_RELEASES` (namespace
  import; no duplication).
* 27th consecutive preflight/earn-discipline validation (W93–W117): runbook locked
  before any NIM; the no-go branch is pre-committed by the rule, so the $0 spend is
  discipline, not omission.

---

## Honest framing

W117 does **multiple** substantial things in one push: it ATTACKS the upstream
CONSTRUCTION provenance at EIGHT authoritative surfaces (revision/commit log + refs +
discussions + GitHub commits + GitHub tags + repo pipeline structure + README
provenance + runner loader) rather than only checking for a packaged `release_v7`; it
ATTACKS the model-disclosure side DEEPER from PRIMARY sources (re-confirming the
DeepSeek V4 model-card PDF states no cutoff and Maverick's "August 2024" verbatim); it
SHIPS a durable upstream-DERIVED construction/admission pipeline with a pre-committed
construction rule (A1..A5 ∧ B1 ∧ B2 — which REFUSES raw-contest hand-assembly), a
multi-surface provenance snapshot, a candidate-instrument constructor, a provenance
validator, a certifiable-slice builder, the disclosure matrix, and an exact W118 fire
condition; and it lands the honest verdict — **no post-v6 instrument can be
CONSTRUCTED from authoritative provenance (only hand-curated, which is refused) and no
reachable stronger-than-Maverick model is certifiably resistant on the latest real
data, so $0 NIM.** If a stronger model were certifiable on a construction-admissible
instrument, W117 would run the cheapest honest pilot and load W118 immediately; since
none is, W117 makes the EXACT missing-construction-provenance / missing-primary-cutoff
/ missing-certifiable-slice condition load-bearing and turns the W118 trigger into a
push-button operation. A raw-contest / rumored / aggregator-only / website-only edge
is not a construction-admissible instrument and not a win; the bounded claim is the
registered truth floor, not surrender. `ultracode` stays OFF; `COO-9` stays lead.
