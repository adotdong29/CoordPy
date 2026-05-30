# RUNBOOK — W118

**CoordPy-OWNED post-v6 functional-instrument CONSTRUCTION (build it from official
sources, do not wait for packaged `release_v7`) + deeper primary-source model-cutoff
attack + durable construction/admission/pilot-readiness pipeline + one clean pilot
ONLY if the constructed instrument is actually admissible AND certifiable.**

> Locked **2026-05-30**, BEFORE any NIM call (incl. any reachability re-probe and any
> pilot/canary), per the W93–W117 preflight/earn discipline. A gated milestone with
> THREE lanes — **NOT another exposed rerun, NOT another same-scale resistant reflexion
> rerun, NOT a "check if v7 exists yet" snapshot re-check, NOT a bounded-context /
> compaction / token-compression job.** `ultracode` stays OFF. `COO-9` stays the lead
> path.

---

## The one question W118 answers

W117 (live, 2026-05-30) attacked the upstream CONSTRUCTION provenance at EIGHT
authoritative surfaces and proved that **no post-v6 instrument can be *inherited* from
LiveCodeBench's published provenance** (LCB publishes only packaged releases — no
collection pipeline, no forward problem-id manifest), so its `B1` criterion
(authoritative LCB-published provenance) refused every post-v6 path. W117 answered
**"can a post-v6 instrument be *inherited* from LCB?"** (No), and STOPPED at "$0 NIM —
waiting on a packaged `release_v7` or an LCB-published construction provenance."

**W118 stops waiting and asks the strictly harder, more aggressive question:**

> **Can CoordPy itself CONSTRUCT a reproducible, machine-checkable, OFFICIAL-SOURCE
> post-v6 functional instrument — built directly from the official contest source
> family LCB already names (Codeforces / AtCoder / LeetCode), with provenance STRICTER
> than LCB's currently published post-v6 story — without pretending it is an LCB
> release? And has any reachable stronger-than-Maverick model disclosed a primary-KNOWN
> cutoff since W117 when probed DEEPER?**

This is a NEW, explicitly **CoordPy-OWNED** instrument line
(`coordpy_frontier_functional_v1`). It is **not** "LCB v7" and it does **not** smuggle
hand-curated problems in under "upstream-derived". The lane is **not complete if it
only says "still no v7."** It must either build a real CoordPy-owned post-v6 instrument
or prove **precisely**, machine-checkably, why the official-source family still cannot
yield a *pilot-runnable* one.

---

## Linear

* **`COO-9`** (High, Todo) — "Build a second code benchmark battlefield with lower
  ceiling pressure" — stays the **lead path**. Parent epic **`COO-6`**.
* **`COO-41`** = W117 (Done). Its close pre-commits **W118 fires the moment** a packaged
  `release_v7`+ is admitted, OR an LCB-PUBLISHED post-v6 construction provenance enables
  a B1∧B2 ≥30 slice, OR a reachable stronger-than-Maverick model discloses a
  primary-KNOWN cutoff ≤ 2025-01 — **but W118 escalates past passive waiting and
  attempts CoordPy-owned construction directly.**
* **W118** = a NEW sub-issue under `COO-6` (sibling of `COO-9`), created at milestone
  end with full results + a COO-9 summary comment (the W105→W117 pattern).
  `linear_github_mapping.json` updated + `sync_linear_github_v1.py` validated.

---

## What is NOT in scope (anti-drift)

* **No** reopening MBPP+ V2 (W102 cap).
* **No** reopening the frozen cross-modal lines (RealWorldQA frozen at 11B).
* **No** reopening the closed Llama-3.1 rescue branch (W106 NO-GO).
* **No** APPS main-lane NIM (APPS stays the exposed control only).
* **No** 70B resistant reflexion de-noise (W109 rule).
* **No** second Maverick resistant reflexion rerun on the SAME `release_v6` instrument
  (redundant; § 6).
* **No** 405B expensive run unless reachability changes AND a pre-committed gate clears.
* **No** dirty / contamination-EXPOSED / sample-only-graded benchmark sold as a frontier
  win.
* **No** raw-contest hand-assembled / aggregator-only / website-only / rumored
  instrument or cutoff treated as admissible (the O5/B2 no-operator-curation criterion
  REFUSES them).
* **No** operator-SYNTHESISED hidden tests (a reference-solution-and-generator harness
  written by the operator is operator curation — refused by O5 — and injects a
  correctness confound that would make any B−A1 uninterpretable).
* **No** bounded-context / compaction / token-compression / "truncate better" drift.
* **No** version bump, **no** PyPI publish, `coordpy/__init__.py` untouched.
* `ultracode` stays OFF (a bounded construction + certification milestone, not a
  repo-wide dynamic-workflow job). Threshold to reconsider: multiple candidate
  instrument BUILDERS live at once / a repo-wide certification migration / broad
  multi-surface external verification that cannot be done sequentially — none of which
  W118 requires (the official-source family is three named sources, swept sequentially).
  If crossed, say so explicitly before changing modes.

---

## Operational state (pre-W118 facts, held constant)

* **Two confirmed retirements STAND** — W89 (base HumanEval, +5.56 pp) + W105
  (HumanEval+, +7.00 pp), both `meta/llama-3.3-70b-instruct` @ 70B,
  contamination-EXPOSED HumanEval-family. W118 must not weaken these.
* **Resistant superiority = 0 clean across BOTH scales** (70B −3.33 / +0.00; Maverick
  +0.00; M3 sub-floor at 70B). REGISTERED ceiling (W114).
* Admitted instrument frontier (W113/W114): LCB `release_v6` functional subset = 63
  problems, 2025-01-11..2025-04-05; a ≥30 functional resistant slice on it requires a
  KNOWN cutoff ≤ 2025-01. Decision CID `258b6ed7` (W114=W115=W116=W117).
* Reachability (W112 sweep, fixed prior — NOT re-probed): Maverick = reachable tier-1;
  Qwen3-Coder-480B / DeepSeek-V4-pro / Mistral-Small-4 = reachable tier-2;
  `meta/llama-3.1-405b-instruct` = 404×6.

---

## § 1 — α / β / γ branch logic (LOCKED)

* **Lane α (main, LIVE — CoordPy-OWNED post-v6 instrument CONSTRUCTION):** stop waiting
  for a packaged release. Stay inside the official source family LCB names
  (Codeforces / AtCoder / LeetCode), at each source's OFFICIAL surface. Pre-commit the
  CoordPy-owned instrument rule (§ 3, O1..O7). Build a REAL constructor (fetch official
  metadata → normalise dates → deterministic inclusion/exclusion → machine-generated
  manifest → SHA pin + CID + date histogram). If the constructor yields ≥30 admitted
  problems AND an official executable grader (O7) → build the slice, run the full
  real-data preflight, and if it certifies a stronger-than-Maverick model (or a
  verdict-changing Maverick on a genuinely-new instrument) earn the cheapest honest
  pilot (§ 6); else no-go (§ 7), but STILL land the constructor/manifest/admission
  machinery AND make the EXACT missing official-source artifact machine-checkable.
* **Lane β (mandatory, LIVE — deeper primary-source model-cutoff attack):** re-check
  official cutoff disclosures for Qwen3-Coder-480B, DeepSeek-V4-pro,
  Mistral-Small-4-2603, Maverick, AND scan for any newly-reachable stronger model, from
  PRIMARY sources, DEEPER than a single card (official cards + release notes + official
  PDFs + vendor docs). Build the sharpened disclosure matrix (§ 4b). If any
  stronger-than-Maverick model now has a primary-KNOWN cutoff ≤ the constructed
  instrument's frontier, integrate immediately; else sharpen the blocker.
* **Lane γ (mandatory, NIM-free — construction/admission/pilot-readiness pipeline +
  graphify + truth):** land `coordpy.coordpy_frontier_functional_v1` (the constructor +
  the O1..O7 rule + the official-source-family grader registry + the manifest validator
  + the reused C1..C4 certification on the manifest + the O7 grader gate + the W119 fire
  condition), reusing the W113 registry + W114 `certify_model_v1` + W117
  `run_upstream_construction_v1` (explicit-import-only, no duplication; the LCB-inherited
  decision CID re-derives byte-identically = `258b6ed7`). graphify refreshed start +
  close. **Land executable code/script assets, not just docs.**

---

## § 2 — official-source family + provenance rule (LOCKED)

**Sources are the official contest family LCB names, at each source's OFFICIAL surface
only** — no random mirror, no third-party aggregator/scraper as authority (they
corroborate at most). Tooling: a real reproducible fetch (the official Codeforces JSON
API for the LIVE manifest) + WebSearch/WebFetch against the official source surfaces +
model cards/vendor docs/PDFs for Lane β (the documented W113–W117 convention; the chrome
browser MCP is reserved for `/browse` and is NOT used).

**The W118 LIVE official-source-family pass (2026-05-30):**

| # | Source (LCB-named) | Official surface | IDENTITY metadata + dates? | Official EXECUTABLE grader (O7)? | Status |
|---|---|---|---|---|---|
| 1 | **Codeforces** | `codeforces.com/api` (`contest.list` + `problemset.problems`) | **YES** — clean official JSON API; 11,223 problems w/ `type` + tags + contest dates | **NO** — the API record carries no test field; no official test-case endpoint; hidden judge tests never published | `NONE` |
| 2 | **AtCoder** | `atcoder.jp` problem pages + Dropbox system-test folder | partial — dates on contest pages; no official problems JSON API (kenkoooo = 3rd-party) | **NO via clean API** — system tests exist but only on a **Dropbox shared folder** (not an official machine-checkable API; automation discouraged) | `DROPBOX_NON_API` |
| 3 | **LeetCode** | `leetcode.com/graphql` (semi-official) | YES — problem content via GraphQL | **NO** — hidden tests are "not public — even premium users cannot access them" (deliberate) | `NONE` |

**Net (family-wide):** the official source family yields the post-v6 problem **IDENTITY
+ dates** at scale (Codeforces API: clean, machine-checkable), but **NO source publishes
a clean, official, reproducible EXECUTABLE per-problem test suite (the functional
GRADER)**. AtCoder is the closest (it *does* publish system tests) but via Dropbox, not
a clean official API. The grader blocker is a **property of the source family**, not a
Codeforces quirk.

**Model cutoffs (Lane β, DEEPER pass, 2026-05-30, primary sources):**

| Model | Primary source (verbatim probe) | Disclosure | Blocker |
|---|---|---|---|
| `meta/llama-4-maverick-17b-128e-instruct` | Meta `llama-models/.../llama4/MODEL_CARD.md` re-fetched — **"Knowledge cutoff: August 2024"** + **"pretraining data has a cutoff of August 2024"** (verbatim) | **KNOWN** | **C4** on `release_v6` (settled, W113); on `coordpy_frontier_functional_v1` identity-CERTIFIABLE but **O7-grader-blocked** |
| `qwen/qwen3-coder-480b-a35b-instruct` | Official HF card raw README — **NO CUTOFF STATED** | **UNKNOWN** | C1 |
| `deepseek-ai/deepseek-v4-pro` | **Official DeepSeek V4 model-card PDF re-fetched DIRECTLY** — **NO CUTOFF STATED**; aggregator "Apr 2026" non-primary | **UNKNOWN** (primary) | C1; aggregator figure post-frontier ⇒ C2-exposed |
| `mistralai/mistral-small-4-119b-2603` | Official Mistral docs `models_overview` — lists **"Mistral Small 4" v26.03, NO CUTOFF STATED** | **UNKNOWN** (primary) | C1; aggregator figure post-frontier ⇒ C2-exposed |
| `zai-org/glm-5` (**newly noted**) | Z.ai GLM-5 docs / GitHub — **NO PRIMARY CUTOFF**; listicle "training ~Feb 2026" non-primary | **UNKNOWN** (primary) | C1 + C2 (post-frontier figure) + **reachability UNVERIFIED** |

**Net:** no reachable stronger-than-Maverick model has a primary-KNOWN cutoff ≤ the
constructed frontier; **nothing newly primary-disclosed since W117** (GLM-5 is newly
*noted* but UNKNOWN-from-primary + C2-exposed + reachability-unverified).

---

## § 3 — CoordPy-OWNED post-v6 functional-instrument rule (LOCKED — the W118 supply gate)

A constructed instrument is **IDENTITY-ADMISSIBLE** iff O1..O6 hold and it carries ≥
`MIN_RESISTANT_SLICE` = 30 admitted problems; it is **PILOT-ADMISSIBLE** iff it is also
**GRADER-ADMISSIBLE (O7)**:

* **O1 official source family** — a source LCB names (Codeforces / AtCoder / LeetCode),
  at its OFFICIAL surface; NOT a mirror/aggregator/scraper, NOT presented as an LCB
  release.
* **O2 dated problems** — each carries an official contest start date (time-anchor).
* **O3 post-v6** — contest date STRICTLY AFTER the `release_v6` functional frontier
  `2025-04-05` (the entire frontier day excluded — the W113 strict-boundary
  conservatism).
* **O4 functional/code-generation-compatible** — a programming problem the W89
  mechanism can attack.
* **O5 deterministic inclusion/exclusion (no operator curation)** — the selection AND
  ordering are fully determined by a total machine rule over the official payload, so
  anyone re-running it on the same bytes obtains the byte-identical set; NO hand-picking
  / vibes; NO operator-synthesised content.
* **O6 machine-generated manifest** — reproducible fetch + a SHA-256 pin of the official
  source bytes + a content-addressed manifest CID + a re-derivable date histogram.
* **O7 official executable functional GRADER artifact** — a reproducible, OFFICIAL,
  machine-checkable per-problem hidden-test suite to GRADE generated code. Sample-only
  tests are insufficient (high hidden-test false-pass ⇒ an uninterpretable B−A1);
  operator-synthesised tests are operator curation (refused by O5). **This is the
  binding gate.**

**On the W118 LIVE pass (§ 2):** the official Codeforces API yields **894 admitted
problems** (O1..O6 hold; 894 ≥ 30) — the IDENTITY tier is SOLVED — but **O7 FAILS**
family-wide (no official executable grader). ⇒ the instrument is **IDENTITY-ADMISSIBLE
but NOT PILOT-ADMISSIBLE**. The exact missing artifact is load-bearing: *a reproducible,
official, machine-checkable per-problem executable test suite for ≥30 of the post-v6
functional problems.*

---

## § 4 — per-model certification rule on the constructed manifest (LOCKED)

Reuse the W114 `certify_model_v1` C1..C4 gate (NO duplication) against the constructed
manifest's date histogram (the manifest adapts to `LatestResistantInstrumentV1` via
`as_resistant_instrument()`). C4 is instrument-specific: the CoordPy-owned manifest is a
GENUINELY NEW instrument none of the candidates has run, so `already_settled_on_
instrument` resets to False (Maverick's W113 settlement was on `release_v6`, a different
slice). A candidate is **PILOT-ADMISSIBLE** iff it is C1∧C2∧C3∧C4 identity-certifiable
**AND** the manifest is GRADER-admissible (O7).

**On the W118 LIVE pass:** Maverick (KNOWN Aug-2024) has **all 894 problems resistant**
on this genuinely-new instrument ⇒ **identity-CERTIFIABLE (C1∧C2∧C3∧C4)**; the ONLY
blocker is the missing grader (O7). The three tier-2 models are C1-blocked (UNKNOWN
cutoff). ⇒ **0 pilot-admissible models** (the grader, not the instrument identities or
the cutoffs, is the binding gate for Maverick).

### § 4b — per-model disclosure-status rule (LOCKED — Lane β deliverable)

Each candidate is classed by its PRIMARY-source disclosure: **KNOWN** / **ESTIMATED-but-
unusable** / **UNKNOWN** / **CONTRADICTORY-or-stale** / **NEWLY-DISCLOSED-since-W117**.
Only a primary-KNOWN cutoff ≤ the manifest frontier, on a reachable stronger-than-70B
not-settled model, can certify. None qualifies (§ 2).

---

## § 5 — construction + slice-construction + exclusion rule (LOCKED)

IF the constructor yields an IDENTITY-admissible manifest AND O7 holds (an official
executable grader exists) AND § 4 certifies some candidate:

1. **Construct** the manifest from the official source (deterministic inclusion;
   machine-generated; SHA-pinned; CID-pinned; date histogram). DONE this milestone (894
   problems; manifest CID `fb4185a6…`; raw-fetch SHA `b6342fd1…`).
2. **Fetch the official grader** for the admitted problems; build the runnable slice;
   pin its CID. **BLOCKED** — no official grader exists (O7 fails).
3. `partition` the resistant slice for the target model's primary-KNOWN boundary;
   EXCLUDE missing/unparseable/not-after-cutoff (typed breakdown).
4. Select the deterministic, outcome-blind difficulty-stratified slice; pin its CID.
5. Run the NIM-free preflight (manifest integrity + grader self-test + slice integrity);
   `pilot_earned` ⟺ all pass AND the slice has ≥30 problems WITH a working grader.

Steps 2–5 are unreachable this milestone (O7 fails at step 2).

---

## § 6 — pilot-earning rule (LOCKED)

A pilot is earned ONLY if BOTH are true:

1. **Lane α** yields a constructed instrument that is PILOT-ADMISSIBLE (O1..O7) and
   certifiably resistant for at least one reachable model (§ 3 + § 4 + § 5); AND
2. **Lane β** yields a stronger-than-Maverick model with a primary-KNOWN cutoff that
   certifies on it AND is NOT redundant — OR Maverick on a GENUINELY NEW instrument it
   never ran (a verdict-changing run: it tests whether the W113 Maverick FAIL was
   `release_v6`-specific or general).

Then the ONE earned expensive run is the cheapest honest Phase-2 pilot (1 seed × 30 ×
K=5 = 330 calls; a ≈22-call canary first), mechanism byte-identical to W89/W108/W113,
scored by `evaluate_phase2_gates_v1` + MLB-1/MLB-2, verdict mapped by
`interpret_cross_scale_resistant_result_v1`.

**On the W118 LIVE pass:** Maverick is identity-certifiable on a genuinely-new instrument
(condition 2 met for a verdict-changing Maverick run) BUT the instrument is NOT
pilot-admissible (O7 grader absent ⇒ condition 1 fails). ⇒ **NO pilot earned; $0 NIM.**
This is discipline: a sample-only or operator-synthesised grader would make B−A1
uninterpretable and is refused.

---

## § 7 — no-go rule if no pilot-admissible certifiable slice exists (LOCKED — the load-bearing branch)

If § 3 returns no PILOT-admissible instrument OR § 4 certifies no usable model, W118
STOPS honestly with **$0 NIM** and records the blocker as a hard, dated, per-surface +
per-model + per-source spend gate (NOT surrender):

* **Instrument/construction side**: the IDENTITY tier is **SOLVED** (894 official
  post-v6 functional problems, 2025-04-07..2026-05-30, 130 contests, Codeforces API,
  deterministic, SHA+CID-pinned) — but the GRADER tier (O7) is **ABSENT family-wide**
  (Codeforces API: no test field/endpoint; LeetCode: hidden tests deliberately private;
  AtCoder: system tests Dropbox-only, no official API). The exact missing artifact is
  named (§ 3).
* **Model side**: every reachable stronger-than-Maverick model has a primary-UNDISCLOSED
  cutoff (C1) and, where an aggregator figure exists, one post-dating the frontier
  (C2-exposed). Maverick (Aug-2024 KNOWN) is identity-certifiable on the new instrument
  but O7-grader-blocked.
* **Carry-forward** (re-affirmed, sharpened): the W114/W115/W116/W117 caps STAND; W118
  ADDS the live CoordPy-owned construction (894 problems), the family-wide grader
  blocker, the GLM-5 newly-noted entry, and the durable constructor/admission pipeline.
* This is **not** "give up" — it is the honest aggressive construction-side move: the
  blocker has MOVED from "no post-v6 problem identities can be constructed" (W117) to
  "post-v6 identities are abundantly constructible (894, official, dated, deterministic),
  but no official source publishes the executable functional GRADER" (W118). The blocker
  is narrowed to ONE specific, named, machine-checkable artifact.

---

## § 8 — graphify deliverables (LOCKED — Lane γ)

* Refresh at start from HEAD (`graphify update .`; HEAD `32d3498`; no topology change).
  **DONE.**
* Use concretely: `explain run_livecodebench_reflexion_bench_v1 / partition_resistant_v1
  / certify_model_v1 / assess_construction_admissibility_v1 / run_upstream_construction_v1`;
  `path run_w113_resistant_pilot.py run_livecodebench_reflexion_bench_v1`;
  `affected run_livecodebench_reflexion_bench_v1`; `explain` on the new W118 module.
  **DONE at start.**
* Refresh at end after all code/doc changes; confirm the graph is built from the W118
  HEAD; `explain run_frontier_functional_construction_v1` shows the reuse edge to
  `run_upstream_construction_v1`.

---

## § 9 — W119 branch logic (LOCKED — pre-committed)

Selected by the W118 construction/certification verdict (the constructor makes the
re-evaluation cheap):

* **PILOT-ADMISSIBLE + CERTIFIABLE → pilot ran:** W119 is dictated by the pilot verdict
  via `interpret_cross_scale_resistant_result_v1`.
* **`NO_CERTIFIABLE_STRONGER_MODEL` (the expected branch):** the resistant-code
  superiority question is **OFFICIAL-GRADER-BLOCKED** (no longer instrument-identity- or
  cutoff-blocked on the identity tier). W119 fires the moment:
  - **(a) grader trigger** — an OFFICIAL, reproducible, machine-checkable executable
    per-problem test suite for ≥30 post-v6 functional problems appears on a clean
    official surface of a source LCB names (making `coordpy_frontier_functional_v1`
    pilot-admissible — Maverick is ALREADY identity-certifiable on it, so a grader alone
    unlocks the cheapest honest verdict-changing pilot); OR
  - **(b) packaged/construction trigger** — a packaged LCB `release_v7`+ is admitted, OR
    an LCB-published post-v6 construction provenance appears (the inherited W118
    triggers; re-run `run_upstream_construction_v1`); OR
  - **(c) cutoff trigger** — a reachable stronger-than-Maverick model discloses a
    primary-KNOWN cutoff ≤ the manifest frontier (gets a ≥30 resistant slice on the
    CoordPy-owned manifest); combined with a grader, run it.
  Re-run `run_frontier_functional_construction_v1` against the updated inputs → if a
  model becomes pilot-admissible, run the pre-committed cheapest-honest pilot on the
  strongest such target. Until one holds, the registered bounded ceiling STANDS and
  resistant-code NIM is BLOCKED on the missing official grader artifact. (A genuinely
  different non-code superiority axis may be selected instead; frozen/closed lines stay
  closed.)

In ALL branches: **W89 + W105 STAND**; `COO-9` stays lead unless the evidence forces a
different code-line move.

---

## § 10 — Stable boundary preservation (LOCKED)

* `coordpy.__version__ == "0.5.20"`; `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`;
  **no PyPI**; `coordpy/__init__.py` untouched.
* Advanced work explicit-import-only: 1 new module (`coordpy_frontier_functional_v1`) +
  1 script (`run_w118_frontier_functional_construction_v1.py`); the module reuses the
  W113 registry + `normalize_contest_date_v1` + `MIN_RESISTANT_SLICE` + the W114
  `certify_model_v1` / `LatestResistantInstrumentV1` / `STRONGER_MODEL_CANDIDATES` + the
  W117 `run_upstream_construction_v1` + the W116 disclosure types (namespace import; no
  duplication). The LCB-inherited decision CID re-derives byte-identically (`258b6ed7`).
* 28th consecutive preflight/earn-discipline validation (W93–W118): runbook locked
  before any NIM; the no-go branch is pre-committed by the rule, so the $0 spend is
  discipline, not omission.

---

## Honest framing

W118 does **multiple** substantial things in one push: it STOPS waiting for a packaged
`release_v7` and BUILDS a real CoordPy-owned constructor that runs LIVE against the
official Codeforces API and produces a reproducible, machine-checkable, SHA+CID-pinned
post-v6 functional-IDENTITY manifest of **894 problems** (2025-04-07..2026-05-30, 130
contests) — proving the date/identity axis is officially constructible at scale; it
sweeps the official-source family (Codeforces / AtCoder / LeetCode) and proves the
EXECUTABLE GRADER is **absent family-wide** (the exact, named, machine-checkable
blocker); it attacks the model-disclosure side DEEPER from PRIMARY sources (re-fetching
the DeepSeek V4 PDF directly, re-confirming Maverick's "August 2024" verbatim, and
adding the newly-noted GLM-5 as UNKNOWN-from-primary + C2-exposed); it SHIPS a durable
constructor/admission/pilot-readiness pipeline (O1..O7 rule, source-family grader
registry, manifest validator, reused C1..C4 + O7 gate, W119 fire condition) with a
falsifiability test; and it lands the honest verdict — **the post-v6 functional
IDENTITY is constructible (894, official) and Maverick is even identity-CERTIFIABLE on
it, but no official source publishes the executable functional GRADER, so the instrument
is not pilot-runnable ⇒ $0 NIM.** The blocker has MOVED (sharper than W117): from "no
post-v6 identities" to "abundant official identities, no official grader." If an
official grader (or a primary-KNOWN stronger cutoff) appeared, W118's pipeline would
make the pilot push-button and load W119 immediately; since none does, W118 makes the
EXACT missing official-grader artifact load-bearing. A sample-only / operator-synthesised
/ aggregator-only / rumored edge is not a pilot-admissible instrument and not a win; the
bounded claim is the registered truth floor, not surrender. `ultracode` stays OFF;
`COO-9` stays lead.
