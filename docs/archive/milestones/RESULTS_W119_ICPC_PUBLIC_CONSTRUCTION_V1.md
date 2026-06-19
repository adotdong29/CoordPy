# RESULTS — W119 official ICPC public-package post-cutoff construction ($0 NIM)

**Date:** 2026-05-30 · **Lead path:** `COO-9` · **Issue:** W119 (sub-issue of `COO-6`)
· **Mechanism spend:** **$0 NIM** (no pilot earned) · **Stable boundary:**
`coordpy.__version__ == "0.5.20"`, `SDK_VERSION == "coordpy.sdk.v3.43"`, no PyPI,
`coordpy/__init__.py` untouched.

---

## One line

W119 made the aggressive pivot W118 set up — **off the grader-LESS LiveCodeBench source
family onto the official ICPC problem-package family that already ships the executable
grader** — and PROVED it: a real constructor enumerated the official `github.com/icpc`
org, verified the post-cutoff Rocky Mountain Regional 2024-2025 + 2025-2026 packages ship
**real (non-LFS) secret test data + output validators + accepted reference solutions**,
and a NIM-free grader self-test ran official accepted solutions against the official
secret cases = **16/16 cases PASS** ⇒ **the W118 family-wide grader blocker is
DISSOLVED**. The honest verdict: the official ICPC grader EXISTS and self-test-passes on
a genuinely-new resistant battlefield, but the post-cutoff resistant pass-fail count from
the cleanest single official surface is **24 < `MIN_RESISTANT_SLICE` = 30**, which blocks
BOTH slice-admissibility AND the reused C2 certification gate ⇒ **no pilot-admissible
certifiable slice ⇒ $0 NIM. W89 + W105 remain the only two retirements.**

---

## The blocker MOVED (the W119 advance)

| Milestone | Question | Answer |
|---|---|---|
| W116 | Admissible *packaged* `release_v7`? | No (four surfaces). |
| W117 | Post-v6 instrument *inheritable* from LCB provenance? | No (eight surfaces). |
| W118 | Can CoordPy *construct* post-v6 IDENTITIES from official sources? | **Yes (894); but executable GRADER absent family-wide.** |
| **W119** | **Can we move to an official family that already SHIPS the grader, and build a clean resistant battlefield?** | **GRADER yes (present + self-test 16/16); resistant pass-fail SLICE = 24 < 30.** |

W118's blocker was "abundant official identities, no official executable grader." W119
**dissolves that** — the official ICPC package family ships real, self-test-passing
executable graders — and **narrows the blocker to ONE specific, named, machine-checkable
artifact**: +6 post-cutoff resistant pass-fail problems with a shipped grader.

---

## The three lanes

### Lane α — official ICPC public-package construction (LIVE; the main empirical lane)

W118 proved no grader can be extracted from the LCB source family. W119 pivots to the
official ICPC package family and **succeeds at the grader tier**:

* **Real constructor, real fetch.** `coordpy.coordpy_icpc_public_functional_v1`
  (`fetch_icpc_package_listing_v1` → `build_icpc_manifest_v1`) enumerated the official
  `github.com/icpc` org (33 repos) via the GitHub API and applied a total deterministic
  inclusion rule (official ICPC package ∧ dated ∧ contest date **strictly after**
  Maverick's Aug-2024 cutoff ∧ NOT interactive ∧ ships a usable grader).
* **Grader present + REAL bytes.** Two post-cutoff repos ship complete packages:
  - `icpc/na-rocky-mountain-2025-2026-public` (2025-11-13): 13 problems, **558 secret
    `.in` + 558 `.ans`**, **119 accepted reference solutions** (41 py / 32 cpp / 18 cc /
    15 java), **8 output validators**.
  - `icpc/na-rocky-mountain-2024-2025-public` (2024-12-03): 13 problems, **548 secret
    `.in`/`.ans`** (no reference solutions shipped).
  - LFS-vs-real check: **REAL TEST BYTES** (`airfaregrants/.../corner-1.in` = `1\n10\n`;
    `adriftatsea/.../001-N-N.in` = `N\nN`), not Git-LFS pointers.
* **Grader self-test (NIM-free; P8).** The official accepted Python solutions for
  `videogames` (8/8) and `whattimeisitmrfox` (8/8) reproduce the official `.ans` under a
  fresh-subprocess token-normalized diff oracle = **16/16 cases PASS** ⇒ the grader is a
  REAL EXECUTABLE ORACLE. (A third probe, `draftlottery`, is a high-precision
  floating-point problem graded by a float-tolerance validator, NOT the default diff
  oracle — its accepted solution does not match under naive diff, so it is honestly
  EXCLUDED from the self-test rather than counted as a pass.)
* **Manifest yield (real data):** **24 admitted** post-cutoff resistant pass-fail problems
  from 26 candidates; excluded interactive 1 (`poetictournament`), custom-without-validator
  1 (`alwaysknowwhereyourtowelis`), pre-cutoff 0. Months: 2024-12 (12) + 2025-11 (12).
  Manifest CID `2b337377…`.
* **`24 < MIN_RESISTANT_SLICE = 30`** ⇒ the GRADER tier (P7∧P8) is SOLVED but the slice
  count is short by 6.

### Lane α (cont.) — the slice count is the binding blocker, at BOTH levels

The 24-problem count is **load-bearing twice**: it blocks slice-admissibility (need ≥30)
AND the reused W114 C2 certification gate (which requires ≥30 resistant problems after the
cutoff). So on the actual 24-problem slice **even Maverick (KNOWN Aug-2024) is NOT
identity-certifiable** (C2: 24 < 30), and the tier-2 models are C1-blocked (UNKNOWN
cutoff):

| Model | identity-certifiable (C1..C4) | grader-admissible (P7∧P8) | slice-admissible (≥30) | pilot | blocker |
|---|---|---|---|---|---|
| `meta/llama-4-maverick-…` | **NO** | YES | NO | NO | **C2: 24 < 30** |
| `qwen/qwen3-coder-480b-…` | NO | YES | NO | NO | C1 (UNKNOWN cutoff) |
| `deepseek-ai/deepseek-v4-pro` | NO | YES | NO | NO | C1 (UNKNOWN cutoff) |
| `mistralai/mistral-small-4-…` | NO | YES | NO | NO | C1 (UNKNOWN cutoff) |

⇒ **0 certifiable models.** A NWERC-2024 official static-package second surface was probed
to aggregate toward 30 but the package zip 404s.

### Lane β — primary-source stronger-model cutoff attack (reused W118 matrix)

Maverick KNOWN "August 2024" (boundary 2024-08-31); Qwen3-Coder-480B / DeepSeek-V4-pro /
Mistral-Small-4 v26.03 / GLM-5 all **NO CUTOFF STATED** (primary). Disclosure
`{KNOWN: 1, UNKNOWN: 4}`; **nothing newly primary-disclosed since W118.** `NVIDIA_API_KEY`
IS present in the run env, so the no-pilot rests purely on the count gate, not reachability.

### Lane γ — package-to-pilot pipeline (NIM-free)

`coordpy.coordpy_icpc_public_functional_v1` (1 module, explicit-import-only; reuses the
W113 registry + `normalize_contest_date_v1` + `MIN_RESISTANT_SLICE`, the W114
`certify_model_v1` / `LatestResistantInstrumentV1` / `STRONGER_MODEL_CANDIDATES`, the W117
`run_upstream_construction_v1`, the W116 disclosure types, and the W118 disclosure matrix —
NO duplication):

* `IcpcPublicInstrumentRuleV1` (P1..P8) + `assess_icpc_admissibility_v1` (identity vs
  grader vs slice).
* `OFFICIAL_ICPC_PACKAGE_FAMILY` + `icpc_family_grader_summary_v1` (the official source-
  family grader registry).
* `build_icpc_manifest_v1` (deterministic, pure) + the thin `fetch_icpc_package_listing_v1`
  (gh-api listing) + `IcpcManifestV1` (`manifest_cid` + `as_resistant_instrument`).
* `run_icpc_stdin_executor_v1` — a real fresh-subprocess stdin/stdout code executor (the
  grader-execution path; NO model inference).
* `W119_GRADER_SELFTEST_V1` + `grader_selftest_summary_v1` (the P8 evidence).
* `certify_models_on_icpc_manifest_v1` (reused C1..C4 + grader + slice gates).
* `W120FireConditionV1` + `run_icpc_public_construction_v1` (push-button) +
  `IcpcPublicConstructionResultV1` (`.cid()`).

Driver: `scripts/run_w119_icpc_public_construction_v1.py` (writes
`results/w119/icpc_public/`). **The LCB-inherited certification decision re-derives
byte-identically: decision CID `258b6ed7…` (= W114/W115/W116/W117/W118).** Result CID
`577f7633…`.

---

## Pilot decision

A pilot needs a PILOT-admissible instrument (P1..P8 + ≥30 slice) AND a certifiable model.
The grader is admissible (self-test 16/16), but the post-cutoff resistant pass-fail count
is 24 < 30 — which blocks BOTH slice-admissibility AND the C2 gate, so no model certifies
⇒ **NO pilot earned; $0 NIM** (discipline, not omission). A sub-threshold 24-task pilot is
refused; `NVIDIA_API_KEY` is present, so this is a clean count-gate no-go.

## Stable boundary

1 new explicit-import-only module (`coordpy_icpc_public_functional_v1`) + 1 script
(`run_w119_icpc_public_construction_v1.py`) + 18 new tests (incl. 2 falsifiability tests:
a synthetic ≥30 grader-clean slice DOES make Maverick certifiable + pilot-admissible +
fires W120; a ≥30 slice WITHOUT a self-test-passing grader does NOT). 110 tests pass
across W113–W119. 29th consecutive preflight/earn-discipline validation (W93–W119).

## Carry-forwards

* **Added**: `W119-T-OFFICIAL-ICPC-PACKAGE-FAMILY-SHIPS-EXECUTABLE-GRADER` (the W118
  blocker dissolved), `W119-T-ICPC-GRADER-SELF-TEST-PASSES` (16/16 official cases),
  `W119-L-OFFICIAL-ICPC-RESISTANT-PASSFAIL-SLICE-COUNT-CAP` (24 < 30, the binding
  blocker), `W119-T-ICPC-PUBLIC-FUNCTIONAL-CONSTRUCTION-PIPELINE-SHIPS`.
* **Retired**: none. W89 + W105 STAND; the W114/W115/W116/W117/W118 caps STAND.

## W120 (loaded)

Fires on (a) the post-cutoff resistant pass-fail count reaching 30 on a clean official
surface (the next official ICPC regional package drop, OR one clean official
second-surface aggregation — the grader is ALREADY present + self-test-passing, so a slice
alone unlocks the cheapest honest verdict-changing Maverick pilot), OR (b) a reachable
stronger-than-Maverick model disclosing a primary-KNOWN cutoff. Re-run
`run_icpc_public_construction_v1` against the updated inputs. `COO-9` stays lead.
