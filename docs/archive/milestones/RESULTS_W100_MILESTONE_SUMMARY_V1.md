# W100 — RealWorldQA cross-scale 90B Phase 2 confirmation milestone summary V1

> 2026-05-25.  Cross-scale 90B confirmation of the W99 winners
> (B2 frontier lead + B5 baseline-only ceiling reference) on
> ``lmms-lab/RealWorldQA`` test.  Pre-committed in
> ``docs/RUNBOOK_W100.md``: candidate slate FROZEN by W99; no
> tournament; no new candidates; B4 + typed-extract sub-family
> remain dead.  Both 90B Phase 2 pilots ran + landed.  The
> empirical verdicts (delivered ~ 20 min apart, ~ 1.1 h total
> NIM wall) are:
>
> * **B2 90B Phase 2 — FAIL by −3.33 pp** with mechanism-load-
>   bearingness sub-gate MLB-2 ALSO FAILing (final-VLM rescue
>   rate 1 / 9 = 11.11 % vs 11B's 3 / 3 = 100 %).  Clean
>   cross-scale collapse pattern matching W96-C C1 verifier.
>   ``PHASE_2_FAIL``.  Pre-committed Part H code-pivot
>   contingency TRIGGERED.
> * **B5 90B Phase 2 — FAIL by +1.67 pp short of bar**
>   (B − A1 = +3.33 pp; gate 4 misses +5 pp threshold).
>   Gates 3 + 6 PASS (B > A1; per-problem 29 / 30 = 97 %).
>   Informative-not-claim verdict; B5 stays classified baseline-
>   only ceiling reference.
>
> Per the pre-committed Part H + Part C of the W100 brief:
>
> 1. **``COO-9`` (second code benchmark) PROMOTED to lead path.**
>    The cross-modal RealWorldQA arc is now structurally
>    restricted to the 11B regime.
> 2. **Phase 3 retirement bench NOT launched.**  B2's cross-
>    scale collapse forecloses retirement-grade evidence on
>    RealWorldQA without a structurally new mechanism.
> 3. **W99 11B carry-forwards STAND unchanged.**  W100 adds
>    cross-scale-bound qualifier carry-forwards but does NOT
>    erase the 11B truth.
> 4. **No version bump.  No PyPI publish.**  W100 ships zero
>    new ``coordpy.*`` modules; B2 + B5 reuse W98 / W99 modules
>    unchanged.

## Inputs

| Field | Value |
|---|---|
| Battlefield | RealWorldQA test (``lmms-lab/RealWorldQA``) |
| Slice | seed=96_504_002; n=30; **same as W97 / W98 / W99** |
| Candidate model | ``meta/llama-3.2-90b-vision-instruct`` |
| Sampling | T=0.7 (A1, B2-text-solver, B5-D2-B0-solver); T=0.0 (A0, B2-reader, B2-final-VLM, B5-D2-B0-reader) |
| K | 5 (byte-exact on A1 and every B route) |
| W99 11B baseline (same slice) | A0=36.67 / A1=93.33 / B2=100.00 / B5=100.00 / B4=76.67 |
| W100 90B verdicts | A0=46.67 / A1=76.67 (B2 run) / A1=80.00 (B5 run) / B2=73.33 / B5=83.33 |

## Per-candidate 90B Phase 2 verdicts

| Candidate | Mechanism | A0 | A1@K=5 | B | B−A1 | Gates | MLB sub-gates (B2) | Verdict |
|---|---|---:|---:|---:|---:|---:|---|---|
| **B2** (direct-vision final-turn) | image-at-decision-boundary | 46.67 % | 76.67 % | **73.33 %** | **−3.33 pp** | 7 / 9 | MLB-1 PASS / MLB-2 **FAIL** | ``PHASE_2_FAIL`` (mechanism collapse) |
| **B5** (switch baseline) | per-question routing | 46.67 % | 80.00 % | **83.33 %** | **+3.33 pp** | 8 / 9 | N/A | ``PHASE_2_FAIL`` (narrow ceiling miss) |

(Per-candidate result docs:
``docs/RESULTS_W100_REALWORLDQA_B2_PHASE2_90B_V1.md``;
``docs/RESULTS_W100_REALWORLDQA_B5_PHASE2_90B_V1.md``.)

## Cross-scale shift summary

| Arm | W99 11B (B − A1) | W100 90B (B − A1) | Δ 11B → 90B |
|---|---:|---:|---:|
| B2 | +6.67 pp (PASS structurally) | **−3.33 pp** (FAIL) | **−10.00 pp** |
| B5 | +6.67 pp (PASS structurally) | **+3.33 pp** (FAIL) | **−3.33 pp** |

Both arms degrade cross-scale, but B5 degrades MORE gracefully
than B2 (−3.33 pp vs −10.00 pp).  **At 90B, the simple switch
baseline outperforms the structural-mechanism candidate by
+10.00 pp** (B5 83.33 % vs B2 73.33 %).  This is the empirical
inversion that pre-committed cross-scale + MLB sub-gates were
designed to catch — and they caught it.

## Mechanism-load-bearingness (B2 only; new W100 sub-gates)

| Sub-gate | 11B | 90B | Threshold | Verdict |
|---|---:|---:|---|---|
| MLB-1 invocation rate ≤ 50 % | 10.00 % | 30.00 % | ≤ 50 % | PASS |
| MLB-2 rescue rate ≥ 33 % | 100.00 % | **11.11 %** | ≥ 33.33 % | **FAIL** |

The final-VLM at 90B is invoked 3x more often AND PASSes 9x
less often per invocation.  This is the W96-C C1 cross-scale-
collapse pattern at a structurally distinct mechanism.

## Per-problem mining (cross-scale comparison)

### B2: 11B PASS = 30 / 30 → 90B PASS = 22 / 30

* 22 / 30 both-pass (B2 keeps W99 11B wins on 22 problems).
* 0 new wins at 90B vs 11B.
* **8 new losses** at 90B vs 11B (5 multi-choice + 2 numeric +
  1 yes_no including the `000615` residual viewer-pov that
  B2's 11B final-VLM uniquely solved).
* All 8 regressions fall through to final-VLM at 90B; final-
  VLM rescues only 1 / 9 invocations.

### B5: 11B PASS = 30 / 30 → 90B PASS = 25 / 30

* 23 / 30 both-pass.
* 0 new wins at 90B vs 11B.
* 5 new losses (4 multi-choice routed to D2-B0 + 1 numeric
  routed to A1).
* D2-B0 on multi-choice: 18 / 18 = 100 % at 11B → 14 / 18 =
  77.8 % at 90B (W95-B0-shape cross-scale fragility on multi-
  choice extraction).
* A1 on yes_no + numeric + short_text: 12 / 12 = 100 % at 11B →
  11 / 12 = 91.7 % at 90B (basically flat).

## NIM-free preflight stayed honest

W100 added TWO new NIM-free cross-scale-stability probes; both
PASSed before any 90B NIM call:

* **AddrW100-B2-P5 cross-scale rescue-prior stability**: PASS
  (W96-D 90B residual 20.51 pp ⇒ expected unique-A1-rescues
  at 90B ≥ 6; threshold ≥ 3).
* **AddrW100-B5-P4 cross-scale route-mass stability**: PASS
  (deterministic parser route distribution + per-pid routing
  byte-identical to W99 11B on-disk).

The probes correctly licensed the NIM spend.  They could NOT
predict mechanism-load-bearingness at 90B — only an empirical
pilot can resolve that.  Hence the W100 MLB sub-gates are
**load-bearing safety equipment** for cross-scale confirmation
of mechanism-driven candidates.

## Honest framing of the milestone

* **What W100 earned**: a clean empirical cross-scale verdict
  on the W99 winners.  B2's mechanism does NOT generalize
  cross-scale; B5's routing-ceiling at 90B is narrower than
  the +5 pp Phase 2 bar.  The W100 verdict closes the cross-
  modal RealWorldQA arc at the 11B regime.
* **What W100 did NOT earn**: multi-agent context superiority
  on RealWorldQA at 90B; or anywhere cross-scale.
* **What W100 confirmed about W99**: the 11B PASS claims STAND
  but their generality is qualified.  W99-L-B2-PHASE2-11B-
  STRUCTURAL-PASS is preserved unchanged; W100 adds the
  cross-scale-bound qualifier W100-L-B2-PHASE2-90B-CAP.
* **What W100 confirmed about the discipline**: the
  preflight-first + cross-scale + MLB sub-gates + multi-
  candidate-tournament-then-confirm operating system caught
  a clean cross-scale collapse and triggered the pre-committed
  code-pivot contingency without drama.  This is the 10th
  consecutive validation.

## COO-9 promotion decision (resolved)

**B2 90B FAILed Phase 2 cleanly with mechanism collapse.**
Therefore:

* The W95-B0-family REPAIR via B2 mechanism is restricted to
  the 11B regime; cross-scale generalisation is NOT earned.
* **``COO-9`` (second code benchmark) PROMOTED to lead path**
  per ``docs/RUNBOOK_W100.md`` Part H code-pivot contingency.
  Status: High priority (was already High).  Description +
  comments updated to reflect lead-path status.
* The cross-modal RealWorldQA arc is **frozen at 11B** in the
  W100 frontier audit.  Future RealWorldQA work requires a
  structurally new mechanism, not another tweak to the W95-B0
  family.
* B5 stays classified baseline-only ceiling reference;
  unchanged.
* B4 + typed-extract-then-text-reason sub-family remain dead;
  unchanged.

## Phase 3 decision (resolved)

**B2 90B FAIL with MLB-2 FAIL ⇒ Phase 3 NOT launched.**

Pre-committed Phase 3 entitlement required BOTH 11B AND 90B
Phase 2 PASS with mechanism-load-bearingness sub-gates clearing.
The 11B side cleared via Option A; the 90B side did NOT clear
on its own merits AND MLB-2 FAILed.  Per ``RUNBOOK_W100`` §
"Phase 3 decision logic" branch 1, Phase 3 NOT launched.

## Carry-forwards

### Added (this milestone)

* ``W100-L-REALWORLDQA-B2-DIRECT-VISION-FINAL-TURN-PHASE2-90B-CAP``
  (cross-scale FAIL: B2 − A1 = −3.33 pp at 90B; MLB-2 FAIL)
* ``W100-L-REALWORLDQA-B2-CROSS-SCALE-COLLAPSE-MECHANISM-NON-LOAD-BEARING-AT-90B-CAP``
  (mechanism-level: final-VLM rescue rate 1 / 9 = 11.11 %
  vs 11B's 3 / 3 = 100 %)
* ``W100-L-REALWORLDQA-B5-SWITCH-BASELINE-90B-NARROW-MISS-CAP``
  (ceiling-reference: B5 − A1 = +3.33 pp at 90B; misses
  +5 pp bar by 1.67 pp; 8 / 9 gates PASS)
* ``W100-L-REALWORLDQA-W95-B0-D2-B0-MULTI-CHOICE-EXTRACTION-DEGRADES-CROSS-SCALE-CAP``
  (W95-B0-shape: D2-B0 multi-choice 18 / 18 at 11B → 14 / 18
  at 90B; consistent with the MathVista + W97 11B cross-scale
  fragility pattern)

### Retired

**None.**  W89 70B-HumanEval K=5 remains the only confirmed
multi-seed same-budget multi-agent superiority retirement.

### Frontier-audit reclassifications

* **Active frontier (status updated)**: B2 (direct-vision
  final-turn answerer) — image-at-decision-boundary mechanism
  load-bearing at 11B; **non-load-bearing at 90B**.  Cross-
  scale generalisation NOT earned.  B2 retains its W99 11B
  status but with the cross-scale-bound qualifier.
* **Baseline-only ceiling (status updated)**: B5 (question-
  type router) — per-question routing ceiling exists cross-
  scale but at 90B is narrower than +5 pp Phase 2 bar.  B5
  stays classified baseline-only.
* **Active frontier (NEWLY promoted)**: ``COO-9`` second-code-
  benchmark battlefield (MBPP+ / HumanEval+ / APPS /
  LiveCodeBench / SWE-bench-lite per the COO-9 charter).
  W101 pre-commit work begins.
* **Dead direction (unchanged)**: typed-extract-then-text-
  reason sub-family of W95-B0 (D2-B0 + W98 B1 + W99 B4 all
  capped at ≤ −6.67 pp at 11B; W99 B4 worsened to −16.67 pp).
* **NEW dead direction**: image-at-decision-boundary mechanism
  WITHIN the W95-B0 family AT 90B (this milestone's W100 B2
  90B FAIL).

## Discipline status

Preflight-first + cross-scale + multi-candidate-tournament-
then-confirm discipline validated TEN consecutive times: W93 /
W94 / W95 / W96-A / W96-C / W96-D / W97 / W98 / W99 / **W100**.

W100's distinguishing addition is the **mechanism-load-
bearingness sub-gates (MLB-1 + MLB-2)** for cross-scale
confirmation of mechanism-driven candidates.  These caught the
B2 90B mechanism collapse cleanly even when the gate-4 margin
miss alone would have already FAILed — they provide a
*mechanism-level explanation* for the FAIL, not just a margin-
level verdict.

## Stable boundary preservation

* ``coordpy.__version__`` unchanged at ``0.5.20``.
* ``coordpy.SDK_VERSION`` unchanged at ``coordpy.sdk.v3.43``.
* No PyPI publish.
* ``coordpy/__init__.py`` untouched.
* **Zero new ``coordpy.*`` modules introduced by W100.**  B2 +
  B5 reuse ``coordpy.realworldqa_bench_v3`` (W98) and
  ``coordpy.realworldqa_bench_v5`` (W99) unchanged.
* The only new code is the W100 pilot driver
  ``scripts/run_w100_realworldqa_pilot.py`` (a script, not a
  library module; does NOT enter the public API).

## Cross-references

* ``docs/RUNBOOK_W100.md`` — pre-commit contract.
* ``docs/FRONTIER_RELEVANCE_AUDIT_W100_V1.md`` — frontier
  classification supplement (anti-drift contract).
* ``docs/RESULTS_W100_REALWORLDQA_B2_PHASE2_90B_V1.md`` — B2
  90B verdict + mechanism-collapse mining.
* ``docs/RESULTS_W100_REALWORLDQA_B5_PHASE2_90B_V1.md`` — B5
  90B verdict + routing-ceiling decomposition.
* ``linear_github_mapping.json`` (W100 entry to be appended
  after the W100 milestone lands).
* ``COO-24`` (W100) parent ``COO-6``.
* ``COO-9`` (PROMOTED to lead path).
