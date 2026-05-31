# W119 — Milestone summary (official ICPC public-package pivot DISSOLVES the W118 grader blocker; slice count 24 < 30 blocks the pilot; $0 NIM)

**One line:** W119 made the aggressive pivot W118 set up — off the grader-LESS
LiveCodeBench source family (Codeforces/AtCoder/LeetCode) onto the **official ICPC
problem-package family** (`github.com/icpc`) that already ships the executable grader —
ran a REAL constructor LIVE against the official ICPC GitHub org, verified the post-cutoff
Rocky Mountain Regional 2024-2025 + 2025-2026 packages ship **real (non-LFS) secret test
data + output validators + accepted reference solutions**, PROVED the grader executable
with a NIM-free self-test (**16/16 official cases PASS**), and re-confirmed from primary
sources that no reachable stronger-than-Maverick model has a primary-KNOWN cutoff. The
LCB-inherited verdict re-derives `NO_CERTIFIABLE_STRONGER_MODEL` (decision CID
`258b6ed7…`, byte-identical to W114→W118). **The W118 family-wide grader blocker is
DISSOLVED; the new blocker is the SLICE COUNT (24 post-cutoff resistant pass-fail problems
< 30). No pilot earned; $0 NIM. W89 + W105 remain the only two retirements.**

---

## The blocker MOVED (the W119 advance)

| Milestone | Question | Answer |
|---|---|---|
| W117 | Inherit a post-v6 instrument from LCB? | No (eight surfaces). |
| W118 | Construct post-v6 IDENTITIES from official sources? | Yes (894); GRADER absent family-wide. |
| **W119** | **Pivot to an official family that SHIPS the grader?** | **GRADER present + self-test 16/16; resistant pass-fail SLICE = 24 < 30.** |

---

## The three lanes

### Lane α — official ICPC public-package construction (LIVE; real GitHub org)

* **24 admitted** post-cutoff resistant pass-fail problems (from 26 candidates) across 2
  official repos — RMRC-2025-2026 (2025-11-13; 12 admitted: 11 pass-fail + 1
  custom-with-validator; 1 interactive excluded) + RMRC-2024-2025 (2024-12-03; 12 pass-fail;
  1 custom-without-validator excluded). Months 2024-12 (12) + 2025-11 (12). Deterministic;
  manifest CID `2b337377…`.
* **GRADER PRESENT + EXECUTABLE** (the W118 blocker DISSOLVED): the official ICPC packages
  ship `data/secret/*.in`+`*.ans` (REAL bytes, not LFS) + 8 output validators + 119
  accepted reference solutions; a NIM-free grader self-test on `videogames` + `whattimeis-
  itmrfox` = **16/16 cases PASS** ⇒ a real executable oracle.
* **The SLICE COUNT is the binding blocker, at BOTH levels**: 24 < 30 blocks slice-
  admissibility AND the reused C2 cert gate (needs ≥30 resistant after cutoff), so even
  Maverick is NOT identity-certifiable on the 24-slice ⇒ 0 certifiable models. NWERC-2024
  static-package second surface 404s.

### Lane β — primary-source cutoff attack (reused W118 matrix)

Maverick KNOWN Aug-2024; Qwen3-Coder-480B / DeepSeek-V4-pro / Mistral-Small-4 / GLM-5 all
NO CUTOFF STATED (primary). `{KNOWN: 1, UNKNOWN: 4}`; nothing newly disclosed since W118.
`NVIDIA_API_KEY` present ⇒ the no-pilot is a clean count-gate no-go, not reachability.

### Lane γ — package-to-pilot pipeline (NIM-free)

`coordpy.coordpy_icpc_public_functional_v1` ships the P1..P8 rule, the official ICPC
source-family grader registry, the deterministic manifest constructor + thin live fetch, a
real fresh-subprocess stdin/stdout executor, the grader self-test summary, the reused
C1..C4 + grader + slice certification, the W118 disclosure matrix, and a structured W120
fire condition — reusing the W113/W114/W116/W117/W118 chain (LCB decision CID `258b6ed7…`
re-derives byte-identically). 18 new tests (incl. 2 falsifiability) + driver script.

## Pilot decision

Needs a PILOT-admissible instrument (P1..P8 + ≥30) AND a certifiable model — the grader is
admissible but the slice is 24 < 30 ⇒ no model certifies ⇒ **NO pilot earned; $0 NIM**.

## Stable boundary

`coordpy.__version__ == "0.5.20"`; `SDK_VERSION == "coordpy.sdk.v3.43"`; no PyPI;
`coordpy/__init__.py` untouched. 1 module + 1 script + 18 tests (110 across W113–W119).
29th consecutive preflight/earn-discipline validation (W93–W119).

## Carry-forwards

* **Added**: `W119-T-OFFICIAL-ICPC-PACKAGE-FAMILY-SHIPS-EXECUTABLE-GRADER`,
  `W119-T-ICPC-GRADER-SELF-TEST-PASSES`,
  `W119-L-OFFICIAL-ICPC-RESISTANT-PASSFAIL-SLICE-COUNT-CAP`,
  `W119-T-ICPC-PUBLIC-FUNCTIONAL-CONSTRUCTION-PIPELINE-SHIPS`.
* **Retired**: none. W89 + W105 STAND; the W114→W118 caps STAND.

## W120 (loaded)

Fires on (a) the post-cutoff resistant pass-fail count reaching 30 on a clean official
surface (the next official ICPC regional drop / one clean official second-surface
aggregation — grader already present, so a slice alone unlocks the cheapest honest
verdict-changing Maverick pilot), OR (b) a reachable stronger-than-Maverick model
disclosing a primary-KNOWN cutoff. `COO-9` stays lead.
