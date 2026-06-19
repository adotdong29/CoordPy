# RESULTS — W118 CoordPy-OWNED post-v6 functional-instrument construction ($0 NIM)

**Date:** 2026-05-30 · **Lead path:** `COO-9` · **Issue:** W118 (sub-issue of `COO-6`)
· **Mechanism spend:** **$0 NIM** (no pilot earned) · **Stable boundary:**
`coordpy.__version__ == "0.5.20"`, `SDK_VERSION == "coordpy.sdk.v3.43"`, no PyPI,
`coordpy/__init__.py` untouched.

---

## One line

W118 stopped waiting for a packaged `release_v7` and **built a real CoordPy-OWNED
constructor** that runs LIVE against the official Codeforces API and produces a
reproducible, machine-checkable, SHA+CID-pinned post-v6 functional-IDENTITY manifest of
**894 problems** (2025-04-07 .. 2026-05-30, 130 contests) — proving the date/identity
axis is **officially constructible at scale** — then proved that the EXECUTABLE
functional GRADER is **absent family-wide** across the three official sources LCB names
(Codeforces / AtCoder / LeetCode), so the instrument is **identity-admissible but NOT
pilot-admissible**. The blocker has MOVED (sharper than W117): from "no post-v6 problem
identities can be constructed" to **"abundant official post-v6 identities, no official
executable grader."** Maverick is even **identity-CERTIFIABLE** on this genuinely-new
instrument (all 894 problems resistant for its KNOWN Aug-2024 cutoff) — the ONLY thing
blocking a verdict-changing pilot is the missing grader. **No pilot earned; $0 NIM. W89
+ W105 remain the only two retirements.**

---

## The three lanes

### Lane α — CoordPy-OWNED post-v6 instrument CONSTRUCTION (LIVE; the main empirical lane)

W117 proved no post-v6 instrument can be *inherited* from LCB's published provenance.
W118 attempts the construction directly from the official source family — and **succeeds
at the identity tier**:

* **Real constructor, real fetch.** `coordpy.coordpy_frontier_functional_v1`
  (`fetch_codeforces_official_v1` → `build_frontier_manifest_from_codeforces_v1`)
  LIVE-fetched the official Codeforces API (`contest.list` + `problemset.problems`;
  raw-fetch SHA `b6342fd1…`) and applied a total, deterministic inclusion rule
  (PROGRAMMING-type ∧ FINISHED ∧ contest date **strictly after** the `release_v6`
  frontier 2025-04-05).
* **Yield (real data):** **894 admitted** from 11,223 candidates; excluded
  not-after-frontier 10,329 (incl. the entire 2025-04-05 frontier day, the W113
  strict-boundary conservatism), not-programming 0, not-finished 0, missing-date 0.
* **Coverage:** **2025-04-07 .. 2026-05-30, 130 distinct contests**, month histogram
  45–88/month (2025-04 61 / 05 51 / 06 49 / 07 62 / 08 47 / 09 88 / 10 76 / 11 79 / 12
  70 / 2026-01 62 / 02 64 / 03 68 / 04 72 / 05 45). Manifest CID `fb4185a6…`.
* **894 ≫ MIN_RESISTANT_SLICE = 30** ⇒ the **IDENTITY tier (O1..O6) is SOLVED**.

### Lane α (cont.) — the grader tier (O7) is the blocker, and it is a SOURCE-FAMILY property

A *runnable* functional instrument needs an executable hidden-test suite per problem to
GRADE generated code. The official-source-family sweep (2026-05-30, primary surfaces):

| Source | Clean identity API? | Official executable grader (O7)? | Status |
|---|---|---|---|
| **Codeforces** | YES (JSON API; 11,223 problems + dates) | **NO** — API record carries no test field; no test-case endpoint; hidden tests never published | `NONE` |
| **AtCoder** | no official problems JSON API (kenkoooo = 3rd-party) | **NO via clean API** — system tests exist only on a **Dropbox** shared folder; automation discouraged | `DROPBOX_NON_API` |
| **LeetCode** | YES (GraphQL) | **NO** — hidden tests "not public — even premium users cannot access them" (deliberate) | `NONE` |

`any_source_has_official_grader = False`; `any_source_has_clean_identity_api = True`. The
grader is absent **family-wide**. Sample-only grading is non-credible (high hidden-test
false-pass ⇒ an uninterpretable B−A1) and operator-synthesised tests are operator
curation (refused by O5). ⇒ **O7 FAILS**; the instrument is **identity-admissible but
NOT pilot-admissible**.

**Per-model certification on the manifest** (reused W114 C1..C4 gate + the O7 grader
gate; C4 reset because the manifest is a genuinely-new instrument):

| Model | n_resistant on manifest | identity-certifiable (C1..C4) | pilot-admissible (+ O7) | blocker |
|---|---|---|---|---|
| `meta/llama-4-maverick-…` | **894** | **YES** | **NO** | **GRADER_BLOCKED (O7)** |
| `qwen/qwen3-coder-480b-…` | 671 | NO | NO | C1 (UNKNOWN cutoff) |
| `deepseek-ai/deepseek-v4-pro` | 894 | NO | NO | C1 (UNKNOWN cutoff) |
| `mistralai/mistral-small-4-…` | 117 | NO | NO | C1 (UNKNOWN cutoff) |

**Maverick is identity-CERTIFIABLE** (KNOWN Aug-2024 cutoff + 894 resistant + reachable
/ stronger / same-budget + a genuinely-new instrument it never ran — C4 reset from the
`release_v6` settlement) — **the ONLY blocker is the missing official grader (O7).**

### Lane β — deeper primary-source model-cutoff attack (LIVE)

Re-checked primary sources DEEPER than W117 and scanned for new models:

* **Maverick** — Meta `llama4/MODEL_CARD.md` re-fetched: **"Knowledge cutoff: August
  2024"** + **"pretraining data has a cutoff of August 2024"** (verbatim). KNOWN;
  C4-settled on `release_v6`; identity-certifiable-but-grader-blocked on the new
  instrument.
* **Qwen3-Coder-480B** — official HF card raw README: **NO CUTOFF STATED**. UNKNOWN.
* **DeepSeek-V4-pro** — official V4 model-card PDF **re-fetched DIRECTLY**: **NO CUTOFF
  STATED**; only a non-primary aggregator "Apr 2026" (C2-exposed). UNKNOWN (primary).
* **Mistral-Small-4-119B-2603** — official Mistral docs `models_overview`: lists
  **"Mistral Small 4" v26.03, NO CUTOFF STATED**. UNKNOWN (primary).
* **GLM-5 (`zai-org/glm-5`) — NEWLY NOTED** (strong 2026 open coder, SWE-bench 77.8%):
  no primary cutoff; only a listicle "training ~Feb 2026" (non-primary, ~10 months past
  the frontier ⇒ C2-exposed) and **reachability UNVERIFIED** (not in the W112 reachable
  NIM catalogue). UNKNOWN-from-primary.

**Net:** disclosure counts `{KNOWN: 1, UNKNOWN: 4}`; **nothing newly primary-disclosed
since W117**; no reachable stronger-than-Maverick model has a primary-KNOWN cutoff ≤ the
manifest frontier. GLM-5 is recorded as `newly_noted_uncertifiable`.

### Lane γ — constructor / admission / pilot-readiness pipeline (NIM-free)

`coordpy.coordpy_frontier_functional_v1` (1 module, explicit-import-only; reuses the
W113 registry + `normalize_contest_date_v1` + `MIN_RESISTANT_SLICE`, the W114
`certify_model_v1` / `LatestResistantInstrumentV1` / `STRONGER_MODEL_CANDIDATES`, the
W117 `run_upstream_construction_v1`, and the W116 disclosure types — NO duplication):

* `FrontierFunctionalInstrumentRuleV1` (O1..O7) + `assess_frontier_functional_
  admissibility_v1` (identity tier vs grader tier).
* `OFFICIAL_SOURCE_FAMILY` + `source_family_grader_summary_v1` (the family-wide O7
  registry).
* `build_frontier_manifest_from_codeforces_v1` (deterministic, pure) + the thin
  `fetch_codeforces_official_v1` (the only network I/O) + `FrontierManifestV1`
  (`manifest_cid` + `as_resistant_instrument`).
* `certify_models_on_manifest_v1` (reused C1..C4 + O7 gate).
* `W118_DISCLOSURE_MATRIX` + `disclosure_delta_since_w117_v1` (Lane β).
* `W119FireConditionV1` + `run_frontier_functional_construction_v1` (push-button) +
  `FrontierFunctionalConstructionResultV1` (`.cid()`).

Driver: `scripts/run_w118_frontier_functional_construction_v1.py` (live fetch, writes
`results/w118/frontier_functional/`). **The LCB-inherited certification decision
re-derives byte-identically: decision CID `258b6ed7…` (= W114/W115/W116/W117).** Result
CID `3ab0d186…`.

---

## Pilot decision

A pilot needs a PILOT-admissible instrument (O1..O7) AND a certifiable model. The
instrument is **identity-admissible but grader-blocked (O7)**, and Maverick — the one
identity-certifiable target — is grader-blocked too ⇒ **NO pilot earned; $0 NIM**
(discipline, not omission). A sample-only or operator-synthesised grader is refused
(non-credible / operator curation).

## Stable boundary

1 new explicit-import-only module (`coordpy_frontier_functional_v1`) + 1 script
(`run_w118_frontier_functional_construction_v1.py`) + 15 new tests (incl. two
falsifiability tests: an official grader DOES unlock a Maverick pilot + fire W119; a
grader without ≥30 does not). 96 tests pass across W113–W118. 28th consecutive
preflight/earn-discipline validation (W93–W118).

## Carry-forwards

* **Added**: `W118-T-COORDPY-OWNED-POST-V6-FUNCTIONAL-IDENTITY-CONSTRUCTIBLE`
  (894 official problems, identity tier solved), `W118-L-OFFICIAL-SOURCE-FAMILY-NO-
  EXECUTABLE-GRADER-CAP` (the binding grader blocker), `W118-T-MAVERICK-IDENTITY-
  CERTIFIABLE-GRADER-BLOCKED`, `W118-T-FRONTIER-FUNCTIONAL-CONSTRUCTION-PIPELINE-SHIPS`,
  `W118-T-GLM5-NEWLY-NOTED-UNCERTIFIABLE`.
* **Retired**: none. W89 + W105 STAND; the W114/W115/W116/W117 caps STAND.

## W119 (loaded)

Fires on (a) an OFFICIAL executable per-problem test suite for ≥30 post-v6 functional
problems appearing on a clean official surface (making `coordpy_frontier_functional_v1`
pilot-admissible — Maverick is ALREADY identity-certifiable on it, so a grader alone
unlocks the cheapest honest verdict-changing pilot), OR (b) a packaged `release_v7`+ /
LCB-published post-v6 construction provenance (the inherited triggers), OR (c) a
reachable stronger-than-Maverick model disclosing a primary-KNOWN cutoff ≤ the manifest
frontier. `COO-9` stays lead.
