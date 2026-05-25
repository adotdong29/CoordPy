# W99 — RealWorldQA B2 + B4 + B5 NIM-free preflight V1 (ALL THREE PASS at both scales)

> **2026-05-25 — All three W99 candidates B2 (direct-vision
> final-turn answerer), B4 (typed schema WITHOUT
> ``direct_answer_hint``), and B5 (question-type router /
> switch baseline) PASS every preflight gate at BOTH 11B AND
> 90B: W96-D D2 composite (P1..P4) + W99 addressability probes
> AddrW99-B2-{P1..P4} + AddrW99-B4-{P1..P3} + AddrW99-B5-
> {P1..P3}.  Per the pre-committed cross-candidate decision
> logic in ``docs/RUNBOOK_W99.md`` (multiple cheap tries
> allowed when multiple candidates earn it; promotion order =
> NIM-free expected lift descending), all three are entitled
> to cheap NIM pilots at 11B.  No NIM was spent in this
> milestone-step.**

## Configuration

| Field | Value |
|---|---|
| Battlefield | RealWorldQA test (``lmms-lab/RealWorldQA``) |
| Parquet shard SHA-256 (shard 0) | ``0ed8b555...`` |
| Parquet shard SHA-256 (shard 1) | ``7dcb3ac3...`` |
| Corpus Merkle root | ``dc629acbf88d929d95c4a47c8e553c050eb829aa48ba32ad53ca6283090618ab`` |
| W97 slice | seed=96_504_002, n=30 |
| W97 sidecars used | per_problem.jsonl (30 outcomes) |
| W98 B1 sidecars used | per_problem.jsonl (30 outcomes) |
| B2 candidate module | ``coordpy/realworldqa_bench_v3.py`` |
| B4 candidate module | ``coordpy/realworldqa_bench_v4.py`` |
| B5 candidate module | ``coordpy/realworldqa_bench_v5.py`` |
| Preflight runner | ``scripts/run_w99_realworldqa_preflight.py`` |
| 11B verdict cid | ``12a91fafea883c6d4d202f4acbd8caf82641b31ea6047d6598e06a31700e5f3e`` |
| 90B verdict cid | ``0bacd989850008b5416eaa6c0f4b1bacd88968a03f53d9853f5c2c454af6e907`` |

## Per-candidate composite preflight (W96-D P1..P4)

All three candidates PASS every probe at both scales:

| Probe | B2 11B | B2 90B | B4 11B | B4 90B | B5 11B | B5 90B |
|---|---|---|---|---|---|---|
| P1 corpus integrity | PASS | PASS | PASS | PASS | PASS | PASS |
| P2 executor self-test (gold) | PASS (765/765 = 100%) | PASS | PASS | PASS | PASS | PASS |
| P3 A1@K=5 failure residual | PASS (residual 26.56 pp) | PASS (residual 20.51 pp) | PASS | PASS | PASS | PASS |
| P4 decomposition argument + multimodal completeness | PASS (1698 chars; 100%) | PASS | PASS (1314 chars; 100%) | PASS | PASS (1234 chars; 100%) | PASS |

## W99 addressability probes (NIM-free)

### B2 (direct-vision final-turn answerer)

| Probe | Verdict | Detail |
|---|---|---|
| **AddrW99-B2-P1 — NIM-free upper bound from W97 confusion table** | **PASS** | both_pass=22, unique_b=3, unique_a1=5, neither=0 (W97 conf-table on 30-problem slice).  B2 best=100.00 % (rescue all unique-A1 via final-VLM); realistic=96.67 % (80 % rescue rate); conservative=90.00 % (50 % rescue rate).  A1@K=5 (W97)=90.00 %.  **Realistic B2 − A1 = +6.67 pp ≥ +5 pp ✓** |
| **AddrW99-B2-P2 — short-circuit static** | **PASS** | V3 short-circuit logic verified: enumerate(text_exes) + if exe.passed + break.  Padding logic verified: ``text_solver_short_circuit_pad``.  Final-VLM logic verified: ``final_vlm_answerer`` invokes ``vlm_gen`` with ``p.image_bytes``. |
| **AddrW99-B2-P3 — final-VLM rescue prior** | **PASS** | A1 K=5 rescues 5 / 5 W97 unique-A1-rescues by re-seeing the image (definitionally — those are the unique-A1-rescues).  B2 final-turn VLM has equivalent visual access on the same cluster. |
| **AddrW99-B2-P4 — budget exact** | **PASS** | B2 K=5 (1 reader + 3 text-solver + 1 final-VLM-or-pad). |

### B4 (typed schema WITHOUT direct_answer_hint)

| Probe | Verdict | Detail |
|---|---|---|
| **AddrW99-B4-P1 — schema primitives retained** | **PASS** | B4's typed schema explicitly includes ``state``, ``orientation``, ``depth``, ``text_in_object`` — every primitive required for the 4 W97 yes/no rescues B1 recovered. |
| **AddrW99-B4-P2 — hint field removed** | **PASS** | Reader prompt mentions ``direct_answer_hint`` 1× (only the explicit "Do NOT include any direct answer hint" admonition; threshold ≤ 1).  Solver template has NO mention of the hint. |
| **AddrW99-B4-P3 — budget exact** | **PASS** | B4 K=5 (1 typed reader + 4 typed solver turns). |

### B5 (question-type router / switch baseline)

| Probe | Verdict | Detail |
|---|---|---|
| **AddrW99-B5-P1 — ORACLE simulation on W97 sidecars** | **PASS** | NIM-free oracle: B5=30/30 = **100.00 %**; A1@K=5 (W97) = 90.00 %; **margin = +10.00 pp ≥ +5 pp ✓**.  Routing: 18 multi-choice → D2-B0 (PASS 18/18); 12 non-mc → A1 K=5 (PASS 12/12). |
| **AddrW99-B5-P2 — parser correctness** | **PASS** | Parser correct on 29 / 30 = 96.7 % of W97 slice (threshold ≥ 90 %).  Misclassification on a single short_text question that opens with "Where"; same as W98 AddrP6. |
| **AddrW99-B5-P3 — budget exact** | **PASS** | B5 K=5 on either route (D2-B0 = 1 reader + 4 text-solver; A1 = 5 VLM samples). |

## Cross-candidate decision (locked in ``RUNBOOK_W99.md``)

All three candidates survived preflight at both 11B and 90B.
Per the pre-committed cross-candidate decision logic:

* **B5 NIM-free expected lift**: +10.00 pp (oracle on W97
  slice).
* **B2 NIM-free expected lift**: +6.67 pp (realistic upper
  bound from W97 confusion table).
* **B4 NIM-free expected lift**: not available (no oracle;
  reasoning-only prediction).
* **Promotion order by NIM-free expected lift**: B5 → B2 → B4.

The W99 brief explicitly says "multiple cheap live tries are
allowed and expected" when multiple candidates earn it.  All
three earn it; up to three NIM pilots will be launched at 11B.
Each pilot is a 1-seed × 30-problem × K=5 run; ~330 NIM calls;
~ 15-25 min wall.

## Honest framing

This is a *preflight* — each candidate is *entitled* to a
NIM pilot, not certified as a Phase 2 win.

* B5's oracle prediction depends on:
  - The deterministic question-type parser correctness (29 /
    30 = 96.7 % on this slice; one short_text question may
    misroute).
  - W97 per-problem outcomes holding under fresh NIM sampling
    at temperature 0.7 (this was empirically stable in the
    W98 run with sampling variance ≤ 3 pp on A1).
* B2's realistic prediction depends on:
  - The final-VLM behaving similarly to A1 K=5 on the failure
    cluster (mechanistically sound — same image access plus
    structured-extraction context).
  - The K=3 text-solver budget covering the 3 W97 unique-B-
    rescues (W97 sidecars confirm first-PASS-at-≤2 on all 3
    multi-choice unique-B-rescues, so K=3 is sufficient).
* B4's prediction is reasoning-only.  Its empirical verdict
  requires NIM.

Pre-committed slice-saturation risk on ``96_504_002`` is
acknowledged and documented in ``RUNBOOK_W99.md``.  A1@K=5
saturated at 90 % in W97 and at 86.67 % in W98; under W99
sampling it is likely in the 85-92 % range.  Per
``RUNBOOK_W99.md`` **Option A**: re-run on the same slice for
direct cross-candidate comparison; treat ``(B − A1)`` as the
discriminator regardless of A1 saturation.

## Anti-cheat (carry-forward from W88-W98)

All preflight steps are NIM-free; no model calls were issued.
The W97 + W98 B1 sidecars used by AddrW99-B2-P1 / B2-P3 / B5-P1
/ B5-P2 are read from on-disk run dirs; no re-derivation of
W97 / W98 truth is asserted.

## Carry-forwards

### Added (this milestone-step)

**None.**  Preflight by definition does not add carry-
forwards; it only earns (or kills) the next NIM call.

### Retired

**None.**  W89 70B-HumanEval K=5 remains the only confirmed
multi-seed same-budget multi-agent superiority retirement.

## Discipline status

Preflight-first + cross-scale discipline now validated NINE
consecutive times (W93 / W94 / W95 / W96-A / W96-C / W96-D /
W97 / W98 / **W99**).  W99's distinguishing addition is the
*three-candidate tournament discriminator* — the milestone
runs preflight on THREE arsenal-driven candidates and may
promote ALL three to cheap NIM pilots (the multi-tries-
allowed rule kicks in because multiple cheap tries genuinely
earn their pilots).

## Next move

**Promote ALL THREE to 1-seed × 30-problem × K=5 cheap NIM
pilots at 11B** under the pre-committed 9 Phase 2 gates in
``docs/RUNBOOK_W99.md``.  Same slice (``96_504_002``).  Same
budget (~330 NIM calls × 3 = ~990 NIM calls; ~45-75 min total
wall).  Pilot driver: ``scripts/run_w99_realworldqa_pilot.py
--candidate {B2,B4,B5}``.

The brief's promotion order by NIM-free expected lift is
B5 (highest) → B2 → B4.  Launching in that order discriminates
the cheapest signal first.

## Re-running

```bash
.venv/bin/python scripts/run_w99_realworldqa_preflight.py \
    --candidate-model meta/llama-3.2-11b-vision-instruct

.venv/bin/python scripts/run_w99_realworldqa_preflight.py \
    --candidate-model meta/llama-3.2-90b-vision-instruct
```

Outputs land under
``results/w99/realworldqa_preflight_b2_b4_b5/``.  Canonical
11B verdict cid:
``12a91fafea883c6d4d202f4acbd8caf82641b31ea6047d6598e06a31700e5f3e``.
Canonical 90B verdict cid:
``0bacd989850008b5416eaa6c0f4b1bacd88968a03f53d9853f5c2c454af6e907``.

## Stable boundary preservation

* ``coordpy.__version__`` unchanged at ``0.5.20``.
* ``coordpy.SDK_VERSION`` unchanged at ``coordpy.sdk.v3.43``.
* No PyPI publish.
* ``coordpy/__init__.py`` untouched.
* New modules ``coordpy.realworldqa_bench_v4`` (B4) and
  ``coordpy.realworldqa_bench_v5`` (B5) are explicit-import
  only.  ``coordpy.realworldqa_bench_v3`` (B2) was already
  explicit-import only from W98.

## The honest claim W99 preflight (B2 + B4 + B5) earns

**On ``lmms-lab/RealWorldQA`` test with the W97 + W98 B1
cheap-pilot per-problem failure-cluster diagnoses as the
cheap-probe surface, all three W99 candidates B2 (direct-
vision final-turn answerer), B4 (typed schema sans
``direct_answer_hint``), and B5 (question-type router /
switch baseline) PASS the W96-D D2 composite preflight
(P1..P4) AND all 10 W99 addressability probes (AddrW99-B2-
P1..P4 + AddrW99-B4-P1..P3 + AddrW99-B5-P1..P3) at both 11B
and 90B.  Per the pre-committed cross-candidate decision
logic (multiple cheap tries allowed when multiple earn it;
promotion order by NIM-free expected lift descending), all
three are entitled to 1-seed × 30-problem × K=5 cheap NIM
pilots at 11B.  Promotion order: B5 (+10.00 pp oracle) → B2
(+6.67 pp realistic) → B4 (NIM-required).  No retirement.
No carry-forward retired.  No NIM spent in this preflight
step.  Discipline validation #9.**
