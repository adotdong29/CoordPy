# W103 — HumanEval+ NIM-free preflight re-confirmation V1

> **2026-05-25.  7 of 7 PASS, deterministically identical to
> W102's 2026-05-25 verdict (same verdict CID).  The cheap
> HumanEval+ pilot at 70B is genuinely earned, NOT bought from
> hope.**

## Why re-confirm

The W102 HumanEval+ preflight ran on 2026-05-25 immediately
before the W102 MBPP+ V2 cheap pilot was launched.  W102's
verdict cid is
`4f57a2cf60ae6a1bbecf15a3ae6e0a9d68a1f9f52d07abb1eb7c2de72e25f7a4`.
For W103 (a separate milestone), the W103 RUNBOOK requires a
fresh preflight run against the same corpus + cache to verify:

1. The HumanEval+ corpus cache is still present + SHA-anchored.
2. The HumanEval+ executor still passes the canonical-solution
   self-test on the first 30 corpus rows.
3. The W93 5-gate + AddrW102-Hplus probes still PASS without
   regression.

If any probe degrades, the cheap pilot is NOT entitled to run.
This is the W93 preflight-first discipline applied verbatim.

## Inputs

| Field | Value |
|---|---|
| Candidate mechanism | B (W89 sequential reflexion on HumanEval+ at K=5) |
| Target model | `meta/llama-3.3-70b-instruct` |
| HumanEval+ corpus | `~/.cache/coordpy/humaneval-plus.jsonl` (164 rows; SHA-256 `908377f1...` matches HF LFS oid) |
| Preflight script | `scripts/run_w102_humaneval_plus_preflight.py` (re-used; no W103-specific changes) |
| Preflight module | `coordpy.humaneval_plus_preflight_v1` |
| W102 verdict cid (2026-05-25) | `4f57a2cf60ae6a1bbecf15a3ae6e0a9d68a1f9f52d07abb1eb7c2de72e25f7a4` |
| W103 re-confirm verdict cid (2026-05-25) | **`4f57a2cf60ae6a1bbecf15a3ae6e0a9d68a1f9f52d07abb1eb7c2de72e25f7a4`** (byte-identical) |
| W103 re-confirm output dir | `results/w102/humaneval_plus_preflight/w102_humaneval_plus_preflight_20260526T015421Z/` |

The verdict cid is byte-identical because the corpus + executor
+ probe definitions are all SHA-pinned and deterministic — the
re-confirm proves the W102 verdict still applies, exactly as
intended by the preflight discipline.

## Per-probe verdicts (re-confirmed)

| Probe | Verdict | Summary |
|---|---|---|
| **P1** corpus integrity | **PASS** | 164 problems; 164 with `test`; 164 with `entry_point`; HF LFS SHA matches. |
| **P2** executor self-test on canonical solutions | **PASS** | 30 / 30 = 100.00 % canonical pass under the V1 executor. |
| **P3** A1@K=5 failure-residual estimate | **PASS** | Predicted HumanEval+ A1@K=5 = 72.86 % (W88 70B HumanEval A1 mean 85.56 % − published EvalPlus Hoeffding lower-bound drop 12.7 pp); saturation margin 17.14 pp. |
| **P4** decomposition argument | **PASS** | 1528 chars; W89 retirement on base HumanEval-70B is the empirical precedent. |
| **P5** extra-test-surface integrity | **PASS** | 164 / 164 = 100.00 % rows carry `def check(` (floor 95 %). |
| **AddrW102-Hplus-AntiPattern** | **PASS** | No `bounded_window` / `compaction` / `summary` tokens in `coordpy/humaneval_plus_reflexion_bench_v1.py`. |
| **AddrW102-Hplus-W89-Rescue** | **PASS** | W89 base-HumanEval rescue fraction 9.76 % ≥ 5 % threshold. |

**7 of 7 PASS, deterministically identical to W102.**

## What this re-confirm does + does NOT do

Does:

* Verify the corpus + cache + SHA pin are still valid for the
  W103 cheap pilot.
* Verify the executor canonical-solution surface is still
  clean.
* Verify the W93 5-gate + AddrW102-Hplus probes still PASS
  without regression.
* Lock the verdict CID + the corpus SHA into the W103
  pilot driver's provenance fields (see Hardening lane in
  `docs/RUNBOOK_W103.md` §).

Does NOT:

* Predict the W103 cheap-pilot outcome with high confidence —
  only the cheap pilot can do that.
* Earn the +5 pp Phase 2 margin bar — that bar is earned by
  fresh-K=5 sampling at seed 103_001 on the helper-anchored
  slice.
* Substitute for the W93 multi-candidate tournament discipline.
* Bump `coordpy.__version__` or `SDK_VERSION`.
* Publish to PyPI.

## Anchors

* `docs/RUNBOOK_W103.md` — pre-commit contract.
* `docs/RESULTS_W103_HELPER_CONSUMPTION_V1.md` — slice
  attestation.
* `scripts/run_w102_humaneval_plus_preflight.py` — preflight
  runner (re-used verbatim for W103).
* `coordpy/humaneval_plus_preflight_v1.py` — 7-probe preflight
  module.
* `results/w102/humaneval_plus_preflight/w102_humaneval_plus_preflight_20260526T015421Z/verdict.json` —
  fresh W103 re-confirm artifact.
