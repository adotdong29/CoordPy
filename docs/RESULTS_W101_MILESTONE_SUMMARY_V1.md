# W101 — Second-code-benchmark battlefield tournament + MBPP+ lead selection + cheap NIM-free preflight milestone summary V1

> 2026-05-25.  Post-W100 code-pivot infrastructure milestone.
> Per the pre-committed Part H of `docs/RUNBOOK_W100.md`,
> `COO-9` (second code benchmark battlefield) was promoted to
> the lead path when W100 confirmed BOTH B2 (frontier lead) and
> B5 (baseline-only ceiling reference) FAIL the +5 pp Phase 2
> bar at 90B on RealWorldQA.  W101 executes the COO-9 charter
> end-to-end: it ranks 5 candidate code benchmark families
> against pre-committed criteria, mines the W89 / W91 sidecars
> offline, builds the MBPP+ loader + executor + bench + preflight
> infrastructure, and lands the empirical preflight verdict.
>
> **No NIM call is made in W101.**  The empirical preflight
> verdict (6 / 8 probes PASS; 2 DEFERRED on operator MBPP+
> fetch) licenses the *next* operator step — fetching the
> EvalPlus MBPP+ release artifact + recording its SHA pin — and
> the cheap NIM pilot is conditional on the preflight re-run
> clearing P1 + P2.  W101's deliverable is the discipline +
> infrastructure + ranked battlefield slate + empirically-
> grounded preflight verdict; the cheap pilot is the *next*
> milestone (W101+ Phase 2) gated on the operator authorisation.

## Inputs

| Field | Value |
|---|---|
| Lead path source | `COO-9` (promoted by W100 Part H code-pivot) |
| Battlefield slate | {MBPP+, HumanEval+, APPS, LiveCodeBench, SWE-bench-lite} (COO-9 charter) |
| LEAD chosen | **MBPP+** (EvalPlus's hardened MBPP) |
| BACKUP chosen | **HumanEval+** (build conditional on MBPP+ preflight FAIL) |
| Out-of-scope (W101) | APPS, LiveCodeBench, SWE-bench-lite |
| Arsenal mining source | W88 70B HumanEval reflexion + W91 5-seed 70B MBPP reflexion sidecars |
| Total sidecar calls mined | 2,640 (990 W89 + 1,650 W91) |
| Total subprocess re-executions | 2,640 |
| New NIM calls | 0 |
| New `coordpy.*` modules | 4 (explicit-import only) |
| New scripts | 3 |
| New tests | 24 (all unit-level; no NIM) |
| Total new code | ~1,800 LoC (loader + executor + bench + preflight + 3 scripts + tests) |
| Stable boundary preserved | `coordpy.__version__ == 0.5.20`; `SDK_VERSION == coordpy.sdk.v3.43`; no PyPI publish; `coordpy/__init__.py` untouched |

## Per-step verdicts

### Battlefield selection (PASS)

5-candidate × 8-criterion ranking matrix in
`docs/RESULTS_W101_BATTLEFIELD_SELECTION_V1.md`.

| Candidate | Verdict | Reason |
|---|---|---|
| **MBPP+** | **LEAD** | A-grade on every criterion except 1; surgically attacks the live `W91-L-MBPP-REFLEXION-V2-5SEED-PARTIAL-CAP` via EvalPlus's ~35× more hidden tests per problem |
| **HumanEval+** | **BACKUP** | Same EvalPlus family; trivial port from W88 HumanEval; smaller residual surface than MBPP+ |
| APPS | Out of scope | C-grade stack on infra cost; reserved for post-MBPP+ |
| LiveCodeBench | Out of scope | Time-anchored harness complexity above W101 envelope |
| SWE-bench-lite | Out of scope | F-grade on decomposition fit + cheap-pilot cost |

### Arsenal mining (PASS — empirically grounded)

`scripts/run_w101_arsenal_mining.py` re-executed 2,640 candidate
responses offline against the canonical W86 / W90 subprocess
executors.

| Bench | A0 | A1@K=5 | B | B−A1 (pp) | b-only / total B wins | Rescue fraction |
|---|---:|---:|---:|---:|---:|---:|
| W89 HumanEval-70B (3×30) | 46.67 % | 85.56 % | 91.11 % | **+5.56** | 8 / 82 | **9.76 %** |
| W91 MBPP-70B (5×30) | 75.33 % | 82.67 % | 84.00 % | **+1.33** | 5 / 126 | **3.97 %** |

Per-seed re-execution numbers match the published W89 + W91
result docs byte-for-byte.  Two structural findings:

1. **Reflexion-rescue surface is 2.5× richer on HumanEval than
   on base MBPP** (9.76 % vs 3.97 %) — the empirical symptom of
   the ceiling-saturation cap MBPP+ relieves.
2. **Hard-cluster size scales with ceiling-control** — MBPP's
   shared-fails cluster (21 / 150) is 2.5× larger than W89's
   (5 / 90); many MBPP "passes" would FAIL on EvalPlus extra
   tests, surfacing them to the failure-residual B can attack.

Full per-seed table in
`docs/RESULTS_W101_ARSENAL_MINING_V1.md`.

### Preflight (6 / 8 PASS; 2 DEFERRED; cheap pilot NOT YET earned)

`scripts/run_w101_mbpp_plus_preflight.py` writes verdict at
`results/w101/mbpp_plus_preflight/<RUN>/verdict.json`.

| Probe | Verdict (2026-05-25) |
|---|---|
| P1 MBPP+ corpus integrity | **DEFERRED** (cache absent; operator must fetch + record SHA pin) |
| P2 Executor self-test on canonical solutions | **DEFERRED** (same gate as P1) |
| P3 A1@K=5 failure-residual estimate | **PASS** (predicted MBPP+ A1@K=5 = **69.97 %**; saturation margin **20.03 pp**) |
| P4 Decomposition argument | **PASS** (1727 chars; W89 retirement as precedent) |
| AddrW101-P1 Mechanism-load-bearing prior | **PASS** (W89 rescue 9.76 %; threshold ≥ 5 % on at least one bench) |
| AddrW101-P2 Per-problem cluster structure | **PASS** (both partitions well-formed) |
| AddrW101-P3 Cross-bench failure-residual stability | **PASS** (margin 20.03 pp; threshold 10 pp) |
| AddrW101-P4 Anti-pattern guard | **PASS** (no `bounded_window` / `compaction` / `summary` tokens in W101 bench module) |

**Decision (per pre-committed runbook branch 1)**: cheap NIM
pilot NOT YET earned.  Operator step required: fetch MBPP+ +
record SHA pin + re-run preflight.  When P1+P2 PASS at re-run,
launch `scripts/run_w101_mbpp_plus_pilot.py` at 1 seed × 30
problems × K=5 (~330 NIM calls; ~2-3 h wall at 70B).

Full per-probe verdict in
`docs/RESULTS_W101_PREFLIGHT_V1.md`.

## Pre-committed Phase 2 gates (locked BEFORE any NIM call)

W101 inherits the W95 / W96-A / W96-C / W97 / W98 / W99 / W100
9-gate shape verbatim plus W101-specific MLB sub-gates:

1. Slice pre-committed (seed 101_001; 30 problems).
2. A1@K=5 < 90 %.
3. B > A1.
4. B − A1 ≥ +5 pp.
5. B > A0 by ≥ +5 pp.
6. Per-problem majority: B ≥ A1 on ≥ 16 of 30.
7. Budget exact (1 + 5 + 5 = 11 calls per problem).
8. Audit chain re-derives offline.
9. Executor stays clean.

**MLB sub-gates** (B only; locks the mechanism's load-bearingness
per the W100 lesson):

* MLB-1 Reflexion-cycle invocation rate ≥ 33 % (≥ 10 / 30
  problems where attempt 0 FAILs and reflexion is exercised).
* MLB-2 Reflexion rescue rate ≥ 33 % of invocations (≥ 1 in 3
  reflexion-exercised problems end up PASSing).

A B PASS with MLB-2 < 33 % is downgraded to
`PASS_NON_MECHANISM_DRIVEN` and does NOT entitle the W102
cross-scale confirmation milestone.

## Carry-forwards

### Added (this milestone)

**None this milestone.**  W101 is infrastructure + preflight; no
empirical NIM result has been produced, so no Phase 2 carry-
forward exists yet.  The W101 result docs explicitly list the
*candidate* carry-forward IDs that will be added by the cheap
pilot in W101+ Phase 2:

* `W101-L-MBPP-PLUS-REFLEXION-PHASE2-70B-CAP` (added if the
  cheap pilot FAILs gate 4 at 70B)
* `W101-L-MBPP-PLUS-PREFLIGHT-P3-SATURATION-CAP` (added if
  preflight P3 FAILs at re-run with cached MBPP+ data)
* `W101-L-MBPP-PLUS-MLB-2-NON-MECHANISM-DRIVEN-CAP` (added if
  cheap pilot PASSes gate 4 but MLB-2 FAILs)

### Retired

**None.**  W89 70B HumanEval K=5 remains the only confirmed
multi-seed same-budget multi-agent superiority retirement.

### Frontier-audit reclassifications

* **Active frontier (NEWLY added)**: `coordpy.mbpp_plus_loader_v1`,
  `coordpy.mbpp_plus_executor_v1`, `coordpy.mbpp_plus_reflexion_bench_v1`,
  `coordpy.mbpp_plus_preflight_v1`, the W101 arsenal-mining
  script, and the W101 multi-candidate-tournament discipline.
* **Active frontier (NEWLY promoted)**: `COO-9` → MBPP+ lead.
* **Baseline-only (NEWLY classified)**: base MBPP via
  `coordpy.mbpp_reflexion_bench_v1` (the W91 cap is empirical;
  MBPP+ is the structural fix).  Stays in-repo for regression /
  audit / cross-bench comparison.
* **Dead direction (NEWLY classified)**: base MBPP at K=5 same-
  budget at 70B for retirement.
* **Dead direction (unchanged)**: cross-modal RealWorldQA at the
  +5 pp Phase 2 bar at any scale (frozen at 11B).
* **Anti-pattern column**: UNCHANGED VERBATIM from W100.

## Discipline status

Preflight-first + cross-scale + multi-candidate-tournament-
then-confirm + mechanism-load-bearingness discipline validated
**ELEVEN consecutive times**: W93 / W94 / W95 / W96-A / W96-C /
W96-D / W97 / W98 / W99 / W100 / **W101**.

W101's distinguishing addition is the **codified battlefield-
selection matrix + offline-sidecar-re-execution arsenal-mining
infrastructure + W93 5-gate preflight extended with the W101-
specific AddrW101 probes**.  This is the first milestone to
ship a fully-documented battlefield-selection rubric (the
W94 / W96-D rubric was implicit; W101 makes it explicit).

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* All 4 new `coordpy.*` modules are explicit-import only.
* The 3 new driver scripts are NOT library modules.

## Cross-references

* `docs/RUNBOOK_W101.md` — pre-commit contract.
* `docs/RESULTS_W101_BATTLEFIELD_SELECTION_V1.md` — 5-candidate
  × 8-criterion ranking matrix.
* `docs/RESULTS_W101_ARSENAL_MINING_V1.md` — sidecar mining +
  cluster surface.
* `docs/RESULTS_W101_PREFLIGHT_V1.md` — preflight verdict.
* `docs/FRONTIER_RELEVANCE_AUDIT_W101_V1.md` — frontier
  classification supplement.
* `coordpy/mbpp_plus_loader_v1.py` — MBPP+ corpus loader.
* `coordpy/mbpp_plus_executor_v1.py` — extra-tests-aware executor.
* `coordpy/mbpp_plus_reflexion_bench_v1.py` — A0/A1/B bench.
* `coordpy/mbpp_plus_preflight_v1.py` — NIM-free preflight.
* `scripts/run_w101_arsenal_mining.py` — offline sidecar
  re-executor.
* `scripts/run_w101_mbpp_plus_preflight.py` — preflight runner.
* `scripts/run_w101_mbpp_plus_pilot.py` — conditional cheap-
  pilot driver.
* `tests/test_w101_mbpp_plus_v1.py` — 24 unit tests; all PASS.
* `linear_github_mapping.json` — W101 entry appended.
* `COO-25` (W101; new) — Linear parent issue.
* `COO-6` (post-W96-A frontier backlog; hub).
* `COO-9` (second code benchmark battlefield; this milestone
  delivers charter items 1-6).
* `COO-24` (W100; promoted COO-9 via Part H).
