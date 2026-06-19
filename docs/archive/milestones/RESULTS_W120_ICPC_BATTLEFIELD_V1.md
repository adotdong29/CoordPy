# RESULTS — W120: official-ICPC multi-surface resistant battlefield (count-gap closure) + earned Maverick pilot

Milestone W120 / `COO-9`. `ultracode` OFF. No version bump; no PyPI; `coordpy/__init__.py`
untouched. Graph refreshed at start (HEAD `106428c`) and at end (see milestone summary).

## Headline

The W119 blocker was the COUNT (24 < 30), not the grader. W120 **closed the count gap on
official ICPC surfaces only**, **certified Maverick** on the resulting ≥30 battlefield, and
**ran the earned pilot**.

* **Lane α (count-gap closure): SUCCESS.** Combined official-ICPC resistant battlefield =
  **45 tier-1 pure pass-fail** (+3 float +1 custom-validator = 49 gradeable), from the
  `github.com/icpc` org, boundary 2024-08-31, dates 2024-11-11..2025-11-13. **45 ≥ 30 with
  margin, using the strictest tier alone — no loosening.**
* **Lane β (certification): Maverick KNOWN Aug-2024, reachable, certifiable.** Matrix
  `{KNOWN:1, UNKNOWN:4}`; nothing newly primary-disclosed since W119.
* **Lane γ (executable infra): shipped** — 2 modules + 2 scripts + 1 test file; the count
  gap is closed by code, not prose.
* **Pilot: EARNED and RUN** (see `## Pilot result` below).

## Lane α — count-gap closure

### Route α1 — RMRC exclusion audit (machine-checkable, problem-by-problem)

Re-derived all 26 RMRC problems directly from each official `problem.yaml` + tree:

| RMRC | pure pass-fail | float | custom+validator | interactive | custom-no-validator |
|---|---|---|---|---|---|
| 2025-2026 | 10 | 1 (`draftlottery`) | 1 (`corporateretreat`) | 1 (`poetictournament`) | 0 |
| 2024-2025 | 12 | 0 | 0 | 0 | 1 (`alwaysknowwhereyourtowelis`) |

**Clean correction:** W119 labeled `draftlottery` as pure pass-fail; it is actually
`float_relative_tolerance 1e-6` (tier-2 float). RMRC gradeable stays 24 (W119's headline
holds) but the composition is now exact: 22 pure + 1 float + 1 custom-with-validator.
Exclusions confirmed load-bearing: `poetictournament` interactive (P4); `alwaysknowwhereyourtowelis`
custom WITHOUT a shipped validator.

### Route α2 — ECNA multi-surface aggregation (NEW official surface)

`icpc/na-ecna-archive` (official NA-East-Division/East-Central-NA regional archive) ships
per-problem `.zip` Kattis packages by year. Post-cutoff years admitted:

| ECNA | pure pass-fail | float | interactive | custom-no-validator | scoring |
|---|---|---|---|---|---|
| 2024-2025 (2024-11-11) | 12 | 1 (`fencesmakegoodneighbors`) | 0 | 0 | 0 |
| 2025-2026 (2025-11-10) | 11 | 1 (`valleygulls`) | 0 | 0 | 0 |

**ECNA grader self-test (NIM-free, R8):** shipped accepted Python solutions vs official
secret cases in a fresh subprocess — **6/6 Python-self-testable problems, 149/149 cases
PASS** (incl. `valleygulls` 40/40 under the deterministic float oracle). The grader is a
real executable oracle on the new surface too.

### Combined battlefield (`coordpy_icpc_battlefield_v1`)

* **51 seen → 49 admitted → 45 tier-1 pure pass-fail** (+3 float +1 custom-validator).
* Excluded: 1 interactive + 1 custom-no-validator (typed, machine-checkable).
* Surfaces: ECNA + RMRC; dates 2024-11-11..2025-11-13; month histogram
  `{2024-11:13, 2024-12:12, 2025-11:24}`.
* `raw_classification_sha256 = b212866f…`; `manifest_cid = bf55bb6c…`;
  `core_slice_cid (30) = 01bf9ef8…`.
* Grader self-test combined: RMRC 16/16 + ECNA 149/149 = 165/165, each surface PASS.

## Lane β — stronger-model cutoff / certification

Re-checked PRIMARY sources 2026-05-31:

| model | primary cutoff | status | NIM-reachable | certifiable on the ≥30 battlefield |
|---|---|---|---|---|
| Llama-4-Maverick | **August 2024** (Meta llama4 MODEL_CARD.md, verbatim ×3) | KNOWN | ✅ | **YES** (C1✓ C2✓[45≥30] C3✓ C4✓) |
| Qwen3-Coder-480B | none stated | UNKNOWN | ✅ | no (C1) |
| DeepSeek-V4-pro | none stated (V3 sys-prompt "Jul-2024" is non-primary) | UNKNOWN | ✅ | no (C1) |
| Mistral-Small-4-119B | none stated | UNKNOWN | ✅ | no (C1) |
| GLM-5 | none primary | UNKNOWN | ❌ | no (C1+reach) |

`{KNOWN:1, UNKNOWN:4}`; nothing newly disclosed since W119. **Maverick is the sole
certifiable + reachable target** — and the count gap closing makes it certifiable for the
first time on this family (C2 flips 24→45).

## Lane γ — executable battlefield-to-pilot infrastructure

* `coordpy/coordpy_icpc_battlefield_v1.py` — multi-surface aggregator + tiered admission
  rule (R1..R8) + exclusion audit + combined manifest + deterministic float oracle +
  per-model certification (reuses W114 `certify_model_v1`) + core-slice selector +
  W121 fire condition. Reuses W119 executor + W117 `run_upstream_construction_v1` (LCB
  decision CID `258b6ed7…` re-derives byte-identically).
* `coordpy/icpc_reflexion_bench_v1.py` — stdin/stdout A0/A1/B sequential-reflexion bench;
  report-shape compatible with the verbatim W108 gate evaluator.
* `scripts/run_w120_icpc_battlefield_v1.py` — NIM-free construction/preflight + verdict.
* `scripts/run_w120_icpc_pilot.py` — earned-pilot driver (canary + full; dry-run loader).
* `tests/test_w120_icpc_battlefield_v1.py` — 13 tests incl. 2 falsifiability tests.

## Pilot result (W120-α) — EARNED, RAN, clean FAIL

Runbook `docs/RUNBOOK_W120.md` LOCKED before any NIM call; canary (2 problems, ~22 calls)
validated the harness end-to-end (reflexion fired + rescued). Full run:

* **Model** `meta/llama-4-maverick-17b-128e-instruct`; **instrument** the deterministic
  tier-1 core 30-slice (CID `01bf9ef8…`; 15 ECNA + 15 RMRC; dates 2024-11-11..2025-11-13;
  every problem post-Aug-2024 resistant); 1 seed × 30 × K=5 = **330 NIM calls; WALL 3249 s
  (~54 min)**.
* **Grader** = official secret cases (token-diff oracle; exit-0-iff-every-secret-case; NO
  LLM judge). **Reflexion feedback** = public samples + judge verdict bit + stderr only.
* **Gates** = the VERBATIM W108 `_evaluate_phase2_gates` + `_mlb_rates`.

| metric | value |
|---|---|
| A0 pass@1 | **20.00 %** (6/30) |
| A1 pass@1 (first-pass-among-K=5) | **23.33 %** (7/30) |
| B pass@1 (sequential reflexion K=5) | **23.33 %** (7/30) |
| **B − A1** | **+0.00 pp** |
| B − A0 | +3.33 pp |
| MLB-1 invocation | 83.33 % (25/30) **PASS** |
| MLB-2 rescue | 8.00 % (2/25) **FAIL** |
| Phase-2 gates | **6/9** (G3/G4/G5 fail) |
| **verdict** | **`FAIL`** → `BOUNDED_CEILING_HOLDS_ON_RESISTANT_ICPC` |

The mechanism was **genuinely invoked** (reflexion fired on 25/30 problems A1 missed at
attempt 0) but **did not transfer**: 2 rescues (`averagesubstringvalue` first-pass at
attempt 2, `fractionalsequence` at attempt 1) were offset by 1 regression
(`garagedoorcode`: A1 solved it, B did not) ⇒ **net +0.00 pp**. This is a clean negative,
not a confounded edge: the battlefield reached 30, the pilot RAN, and the same-budget
multi-agent mechanism tied the single agent on contamination-RESISTANT official ICPC code
at the certified Maverick scale.

### What this means

* **No third retirement.** W89 (base HumanEval +5.56pp) + W105 (HumanEval+ +7.00pp),
  both Llama-3.3-70B @ 70B contamination-EXPOSED-HumanEval-family, **remain the only two**.
* **Resistant superiority is 0 clean across FOUR settings:** W108 (70B LCB −3.33), W110
  (70B BigCodeBench +0.00), W113 (Maverick LCB +0.00), **W120 (Maverick official-ICPC ≥30
  +0.00)**.
* **The W114–W119 escape is CLOSED.** Those milestones could not test the resistant column
  for lack of a certifiable ≥30 grader-clean instrument. W120 BUILT it, certified Maverick,
  ran the pilot, and the mechanism still did not transfer — a strictly stronger statement
  of the bounded ceiling than "untestable".
* **Contamination-confound STRENGTHENED, not proven.** Cleanest resistant null yet (a ≥30
  grader-clean OFFICIAL post-cutoff battlefield at the certified scale). NOT proven: single
  seed (120001); ICPC-regional difficulty vs HumanEval difficulty not separated; a
  Python-under-time-limit floor depresses absolute pass rates (binds all arms equally, so
  it cannot manufacture the +0.00 — a conservative bias).

### W121

Pre-committed FAIL branch (`RUNBOOK_W120` §10): accept the now-tested bounded resistant
ceiling and pursue a genuinely different axis (NOT another exposed rerun). Optional cheap
multi-seed confirmation closes the single-seed caveat. Cross-cutting: a reachable
stronger-than-Maverick model with a primary-KNOWN cutoff would re-open the pilot on this
same ≥30 battlefield (prefer the strongest honest target). `COO-9` stays lead.

Artifacts: `results/w120/icpc_pilot/w120_icpc_pilot_meta_llama-4-maverick-17b-128e-instruct_*/`
(`icpc_reflexion_bench_report.json` + `provenance.json` + `icpc_reflexion_calls.jsonl`
sidecar with per-call prompt/response SHAs).
