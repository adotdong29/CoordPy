# W104 — Milestone summary V1

> **2026-05-26.  Three-lane milestone: cross-scale HumanEval+
> Phase 2 cheap pilot + cross-scale-discipline hardening +
> W105 entitlement / fallback pre-commit.  Pre-locked primary
> target `meta/llama-3.1-405b-instruct` returned HTTP 404 on
> the NIM reachability smoke probe; pre-locked backup
> `meta/llama-3.1-70b-instruct` was applied per the W104
> RUNBOOK § Target-model selection rule.  Verdict:
> `PASS_MECHANISM_DRIVEN` at the cross-generation form (Llama
> 3.1 vs Llama 3.3 at 70B; B − A1 = +10.00 pp; MLB-1 56.67 %
> + MLB-2 35.29 % both PASS).**

## Headline

* **Verdict label**: `PASS_MECHANISM_DRIVEN` (9/9 Phase 2
  gates PASS + MLB-1 + MLB-2 both clear).
* **Cross-generation B − A1**: +10.00 pp (W103 70B Llama-3.3
  was +20.00 pp; cross-generation shift = **−10.00 pp**).
* **Cross-generation MLB-2 rescue rate**: 35.29 % (W103 70B
  Llama-3.3 was 47.06 %; cross-generation shift =
  **−11.77 pp**); STILL above the 33 % floor.
* **Cross-scale form achieved**: cross-GENERATION at same
  parameter scale (Llama 3.1 vs Llama 3.3 at 70B), NOT
  cross-scale-UP (70B → 405B) as the pre-locked primary
  intended.
* **Branch applied**: Branch A of the pre-committed W104
  RUNBOOK § Planning lane.  W105 = HumanEval+ Phase 3
  retirement bench entitled (slice pack pre-built at pack CID
  `8be55f3bf1650df3...`).

## What W104 delivered

### Lane 1 — Lead lane (cross-scale cheap pilot)

* W103 helper-anchored 30-problem slice reused BYTE-FOR-BYTE
  (slice CID `c35155956ece605c...` byte-equal verified at run
  start; corpus SHA `908377f1daf28dcb...` byte-equal verified
  at run start; pilot REFUSES to run on either mismatch).
* Reachability smoke probe applied per the pre-locked RUNBOOK
  § Target-model selection rule.  Primary
  `meta/llama-3.1-405b-instruct` → HTTP 404 on NIM (not
  hosted).  Backup `meta/llama-3.1-70b-instruct` → reachable;
  used.
* 1 seed × 30 problems × K=5 = 330 NIM calls at backup
  target; 7506.9 s (125.1 min) wall; modest 429 throttling
  near the end of the run.
* A0 = 46.67 % / A1@K=5 = 53.33 % / B = 63.33 %.
* 9 of 9 Phase 2 gates PASS; MLB-1 = 56.67 % PASS; MLB-2 =
  35.29 % PASS.
* Per-problem cluster (B vs A1): 5 b_only_wins (rescues);
  14 shared_wins; 2 a1_only_wins (regressions); 9 shared_fails.
* Cross-scale comparator emitted: cluster shifts vs W103 70B
  Llama-3.3 = **8 stayed / 11 improved / 11 regressed / 0
  flipped**; B − A1 cross-generation shift = −10.00 pp; MLB-2
  cross-generation shift = −11.77 pp.

### Lane 2 — Hardening lane (cross-scale discipline)

Four durable guardrails landed:

* `coordpy/cross_scale_comparator_v1.py` — single new
  explicit-import-only module (NOT added to
  `coordpy/__init__.py`); schema/provenance-guarded; refuses
  to run on slice / corpus / schema / MLB mismatch; emits
  per-problem cluster-shift classification.
* `scripts/run_w104_humaneval_plus_cross_scale_pilot.py` —
  pilot driver with byte-equal slice reuse from W103,
  reachability smoke probe BEFORE NIM spend, resume-from-
  sidecar capability for socket hangs / 429 storms, automatic
  cross-scale comparator emit on success.
* `scripts/_w104_emit_pilot_result_doc.py` — verdict-doc +
  comparator-doc emitter; reads the bench report + comparator
  JSON and templates the two markdown verdict docs (one for
  the W104 verdict, one for the comparator narrative).
* `tests/test_w104_cross_scale_discipline_v1.py` — 14 PASSing
  unit tests codifying (a) cross-scale comparator
  schema/provenance refuse-to-run paths, (b) cluster-shift
  correctness on synthetic pairs, (c) cross-scale margin
  shift arithmetic, (d) target-selection-rule determinism
  (anti-pattern token guard, primary-distinct-from-W103,
  primary-is-405B), (e) W103 slice CID equality with W103
  on-disk provenance, (f) sidecar resume-from-disk
  correctness with malformed-trailing-line handling.

### Lane 3 — Planning lane (W105 entitlement OR fallback)

* `docs/RESULTS_W104_HELPER_W105_PLANNING_V1.md` documents
  BOTH the W105 Phase 3 slice pack (Branch A; pre-built in
  this milestone) AND the Branch C fallback dispatch table.
* `scripts/run_w104_w105_phase3_slice_pack.py` is the slice-
  pack pre-builder.  Second real load-bearing downstream
  consumption of `coordpy.code_slice_selector_v1` (first was
  W103).
* `data/w105/phase3_slice_pack/w105_phase3_slice_pack_20260526T215647Z/slice_pack.json`
  is the locked W105 Phase 3 slice (100 problems; pack CID
  `8be55f3bf1650df3...`; W103 30-problem inner kernel
  preserved at the head + 45 mid-shell helper extension + 25
  corpus-fill; cluster mix 30 % rescue-surface concentrated +
  25 % broad corpus coverage + 45 % shared-wins extension).
* The Branch C dispatch JSON is also pre-committed in the
  planning artifact so a future FAIL outcome is execution-
  ready under any FAIL shape.

## Branch decision applied per pre-locked logic

Per the W104 RUNBOOK § Lead lane § Decision branch, the
empirical result triggers:

* **Branch A — full PASS**: 9/9 Phase 2 gates PASS + MLB-1
  PASS + MLB-2 PASS at the backup target
  `meta/llama-3.1-70b-instruct`.
* Carry-forward added: `W104-L-HUMANEVAL-PLUS-REFLEXION-
  PHASE2-CROSS-GENERATION-70B-LLAMA31-PASS` (single-seed
  cross-generation cheap-pilot PASS at Llama-3.1-70B-Instruct
  on the W103 helper-anchored slice; NOT a multi-scale
  retirement and explicitly NOT a 70B → 405B cross-scale-UP
  result).
* Carry-forward added: `W104-L-HUMANEVAL-PLUS-CROSS-SCALE-UP-
  PRIMARY-TARGET-405B-UNREACHABLE-ON-NIM-CAP` (the pre-locked
  primary 405B target was unreachable on NIM at the W104 run
  window; future cross-scale-UP attempts depend on 405B
  becoming hosted).
* **W105 = HumanEval+ Phase 3 retirement bench** ENTITLED.
  Slice pack pre-built; 3 seeds (105 001 / 105 002 / 105 003)
  × 100 problems × K=5 × 2 scales (Llama-3.3-70B-Instruct +
  Llama-3.1-70B-Instruct in the absence of 405B) = 6 600 NIM
  calls.  If 405B becomes reachable before W105 launches, the
  W105 RUNBOOK upgrades the cross-scale axis to (70B,
  405B).  Either way the slice is the locked pack CID
  `8be55f3bf1650df3...`.

`COO-9` REMAINS the lead path.  Programme entitlement after
W104: the W89 sequential-reflexion mechanism extends to
HumanEval+ across **two model classes at the cheap-pilot Phase
2 quality** (Llama-3.3-70B + Llama-3.1-70B), with mechanism
load-bearingness preserved (MLB-2 47 % → 35 %, still above the
33 % floor).  This is a structurally honest cross-generation
result, NOT a cross-scale-UP result.  Retirement-grade
generalisation still requires W105 Phase 3 multi-seed.

## Cross-scale comparator headline

Computed against the W103 70B Llama-3.3 result on the byte-
equal slice:

| Field | W103 70B Llama-3.3 | W104 70B Llama-3.1 |
|---|---:|---:|
| A0 | 50.00 % | 46.67 % |
| A1@K=5 | 50.00 % | 53.33 % |
| B (sequential reflexion K=5) | 70.00 % | 63.33 % |
| B − A1 | +20.00 pp | +10.00 pp |
| MLB-1 invocation | 56.67 % | 56.67 % |
| MLB-2 rescue | 47.06 % | 35.29 % |
| Per-problem b_only_wins | 7 | 5 |
| Per-problem shared_wins | 14 | 14 |
| Per-problem shared_fails | 8 | 9 |
| Per-problem a1_only_wins | 1 | 2 |

* Cross-generation shift on B − A1: **−10.00 pp** (W103 +20 →
  W104 +10).  The mechanism keeps half its W103 margin at the
  cross-generation target.
* Cross-generation shift on MLB-2: **−11.77 pp** (47.06 % →
  35.29 %).  Still above the 33 % floor; the reflexion
  mechanism is still load-bearing on Llama-3.1-70B-Instruct,
  but less so than on Llama-3.3-70B-Instruct.
* Cluster transitions on the 30-problem slice: 8 stayed; 11
  improved; 11 regressed; 0 flipped.  Symmetric improve /
  regress count is consistent with cross-generation sampling
  variance (no ambiguous "flipped" transitions; the comparator
  is structurally clean).

## Empirical event log

| Step | Outcome |
|---|---|
| W104 RUNBOOK locked BEFORE any NIM call | YES |
| Hardening lane code shipped + 14 unit tests PASS BEFORE any NIM call | YES |
| W105 planning lane (Phase 3 slice pack + Branch C dispatch) shipped BEFORE any NIM call | YES |
| Smoke probe on primary `meta/llama-3.1-405b-instruct` | FAIL (HTTP 404 on NIM) |
| Smoke probe on backup `meta/llama-3.1-70b-instruct` | PASS |
| Backup-target selection per pre-locked RUNBOOK | APPLIED |
| W103 slice CID equality at pilot start | VERIFIED byte-equal |
| HumanEval+ corpus SHA pin equality vs W103 | VERIFIED byte-equal |
| Cheap pilot empirical outcome | `PASS_MECHANISM_DRIVEN`; 9/9 gates + MLB-1 + MLB-2 PASS |
| Cross-scale comparator emitted | YES (after permission-outage retry; slice / corpus / schema all match) |
| Decision branch applied per pre-locked logic | Branch A |
| Carry-forwards added | 2 (cross-generation PASS + 405B-unreachable-on-NIM cap) |
| Carry-forwards retired | 0 (W89 70B-HumanEval-K=5 remains the only multi-seed retirement) |
| Discipline validation # | 14th (W93 / W94 / W95 / W96-A / W96-C / W96-D / W97 / W98 / W99 / W100 / W101 / W102 / W103 / W104) |
| Stable boundary preserved | YES (`coordpy.__version__=0.5.20`; `SDK_VERSION=coordpy.sdk.v3.43`; no PyPI; `coordpy/__init__.py` untouched) |
| New coordpy.* modules in W104 | 1 (`coordpy.cross_scale_comparator_v1`; explicit-import only) |
| Tests added in W104 | 14 (`tests/test_w104_cross_scale_discipline_v1.py`) |
| Pilot wall | 7506.9 s (125.1 min) |
| Pilot 429 retries | YES (modest; near end of run; all recovered cleanly) |
| Pilot mid-run permission outage | YES (macOS TCC briefly revoked Desktop access; pilot's own subprocess kept writing; observer shell + comparator emit recovered after permission was restored) |

## What W104 does NOT claim

* Multi-benchmark same-budget retirement on HumanEval+ —
  Phase 3 multi-seed (W105) required.
* Cross-scale-UP generalisation (70B → 405B) — the 405B
  primary was UNREACHABLE on NIM; the cross-generation result
  is structurally weaker.
* Generalisation to MBPP-family at 70B —
  `W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-CAP` stands.
* Generalisation to RealWorldQA — frozen at 11B per W100.
* Anything about "multi-agent context being solved".
* Cross-scale collapse risk eliminated — W96-A / W96-C / W100
  cross-modal cross-scale-UP collapse patterns remain
  structurally untested on the code line because 405B was
  unreachable on NIM at the W104 run window.

## What W104 IS entitled to claim

* The W89 sequential-reflexion mechanism extends to HumanEval+
  on TWO different Llama-3.x 70B model classes at Phase 2
  cheap-pilot quality.
* The W103 +20 pp result was NOT pure model-class luck —
  half the margin (+10 pp) survives the cross-generation
  swap.
* Mechanism load-bearingness (MLB-2) remains above the 33 %
  floor at the second model class (35.29 %), confirming the
  W89 mechanism is doing real work and not riding a
  sampling-variance lottery at the W103 result.
* The W93 – W104 preflight-first + cross-scale + multi-
  candidate-tournament-then-confirm + mechanism-load-
  bearingness + silent-degradation-anti-pattern-guard +
  arsenal-mining-prior-anti-pattern-guard discipline now has
  FOURTEEN consecutive validations.
* W105 Phase 3 retirement bench is entitled and pre-built.

## Anchors

* `docs/RUNBOOK_W104.md` — pre-commit contract.
* `docs/RESULTS_W104_HUMANEVAL_PLUS_PHASE2_405B_V1.md` —
  W104 pilot verdict (with cross-scale comparator block).
* `docs/RESULTS_W104_CROSS_SCALE_COMPARATOR_V1.md` —
  standalone comparator narrative.
* `docs/RESULTS_W104_HELPER_W105_PLANNING_V1.md` — W105
  Phase 3 slice pack + Branch C dispatch table.
* `docs/FRONTIER_RELEVANCE_AUDIT_W104_V1.md` — frontier
  audit supplement (14th preflight-discipline validation).
* `coordpy/cross_scale_comparator_v1.py` — comparator
  module.
* `scripts/run_w104_humaneval_plus_cross_scale_pilot.py` —
  pilot driver.
* `scripts/run_w104_w105_phase3_slice_pack.py` — W105 slice-
  pack pre-builder.
* `scripts/_w104_emit_pilot_result_doc.py` — verdict-doc +
  comparator-doc emitter.
* `tests/test_w104_cross_scale_discipline_v1.py` — 14
  PASSing hardening tests.
* `results/w104/humaneval_plus_cross_scale_pilot/w104_humaneval_plus_cross_scale_pilot_meta_llama-3.1-70b-instruct_20260526T215829Z/humaneval_plus_reflexion_bench_report.json` —
  bench report (rep CID embedded; bench Merkle re-derivable).
* `results/w104/humaneval_plus_cross_scale_pilot/w104_humaneval_plus_cross_scale_pilot_meta_llama-3.1-70b-instruct_20260526T215829Z/cross_scale_comparator_report.json` —
  cross-scale comparator JSON.
* `data/w105/phase3_slice_pack/w105_phase3_slice_pack_20260526T215647Z/slice_pack.json` —
  W105 Phase 3 slice pack (locked pack CID
  `8be55f3bf1650df3...`).
