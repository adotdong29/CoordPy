# W91 — Cross-modal VLM-in-loop V2 at `all_docstring` (Post-W90 empirical superiority wave V4)

> **2026-05-23 — CARRY-FORWARD STAYS.  W91 P2 (3 seeds) showed
> B +2.78 pp mean over A1_vlm (5 / 6 retirement bars met).
> Per the W91 runbook's pre-committed conditional, the
> follow-up W91 P2b (7 seeds, same architecture, same regime)
> was launched immediately.  W91 P2b CLEANLY DISCONFIRMS the
> P2 signal: B 77.4 % < A1_vlm 84.5 % by **−7.14 pp**; B wins
> only **2 / 7 seeds**.  The W91 P2 +2.78 pp was variance-
> driven (a 3-seed favorable draw), not structural.  At larger
> seed count the architecture's mean returns to NEGATIVE.
> `W88-L-CROSS-MODAL-CODE-V1-SPLIT-NOT-LOAD-BEARING-CAP`
> STAYS, with the stronger negative evidence now from 7 seeds.**

## TL;DR — Both runs, side by side

| Run | Seeds | Problems | A0_text | A1_vlm | B_vlm_loop | B−A1_vlm | B>A1 seeds | Outcome |
|---|---|---|---:|---:|---:|---:|---:|---|
| **W91 P2** | 3 | 12 | 44.4 % | 83.3 % | 86.1 % | **+2.78 pp** | 2/3 | 5/6 bars met; +5pp margin FAILS |
| **W91 P2b** | 7 | 12 | 42.9 % | 84.5 % | 77.4 % | **−7.14 pp** | 2/7 | 3/6 bars met; signal disconfirmed |

The +2.78 pp at 3 seeds did NOT generalise.  At 7 seeds the
mean drops to −7.14 pp.  The honest reading: **VLM-in-loop at
all_docstring + 90B-Vision + K=5 does not systematically beat
unified VLM at fair budget.**  The W91 P2 positive direction was a
sampling fluke.

## W91 P2 (3 seeds) — the favorable draw

3 seeds × 12 problems × 3 arms.  K=5; VLM = `meta/llama-3.2-90b-
vision-instruct`; text-LM (A0_text floor) = `meta/llama-3.1-8b-
instruct`; strip_mode = `all_docstring`.  Wall 6148 s
(~1 h 42 min); 36 text + 360 VLM calls.  Bench Merkle
`e335a129db6030c7…`.  Audit verifier 4/4 audit PASS;
retirement bars 5/6 (only +5 pp margin fails).

Result:
* A0_text mean 44.4 % (image strictly necessary at all_docstring;
  text-only floor very low).
* A1_vlm mean 83.3 % (K=5 first-pass; less ceiling-saturated
  than W90 P2's 91.7 % on doctest_only).
* B_vlm_loop mean **86.1 %** (B − A1_vlm = **+2.78 pp**).
* Per-seed: B beats A1_vlm on 2/3 seeds (+8.33 / +8.33 / −8.33).
* Per-seed seed 3: A1_vlm hit 100 % ceiling; B at 91.7 %
  unavoidably loses by −8.33 pp.

Five of six retirement bars met; the +5 pp margin bar failed
(+2.78 pp < +5.0 pp).  Per the W91 runbook pre-committed
conditional, **W91 P2b was launched immediately** to test
whether the gap was variance-driven or genuine.

## W91 P2b (7 seeds) — the disconfirmation

7 seeds × 12 problems × 3 arms.  Same VLM, same architecture,
same regime, same anti-cheat discipline.  Wall 10708 s
(~2 h 58 min); 84 text + 840 VLM calls.  Bench Merkle
`12dee027ec865214…`.  Audit verifier 4/4 audit PASS;
retirement bars 3/6.

Result:
* A0_text mean 42.9 % (consistent with P2).
* A1_vlm mean 84.5 % (consistent with P2; small variance).
* B_vlm_loop mean **77.4 %** (DROPS from P2's 86.1 % by
  −8.7 pp).
* B − A1_vlm = **−7.14 pp** (NEGATIVE; loses).
* Per-seed B beats A1_vlm: (False, False, True, False, True,
  False, False) — **2 of 7 seeds**.
* The first 3 seeds of P2b are NOT the same data as P2 — at
  T=0.7 sampling, fresh draws produce different sample sets
  even at the same numeric seed.

The P2 result of "B wins on seeds 1 & 2 by +8.33 pp each" did
NOT replicate at P2b.  At P2b seeds 1, 2, 4, 6, 7 B loses; at
P2b seeds 3 & 5 B wins.

## Why the disconfirmation matters

The W91 P2 +2.78 pp at 3 seeds had two warning signs that the
P2b 7-seed run confirmed:

1. **Per-seed variance dominated the mean delta.**  Each
   per-seed B−A1 was ±8.33 pp; the +2.78 pp mean was the
   average of (+8.33, +8.33, −8.33) — entirely driven by the
   2:1 sign split.  At 7 seeds the sign split shifted to 2:5
   and the mean went negative.
2. **Margin bar (+5 pp) failed correctly at P2.**  The
   pre-committed +5 pp margin threshold from W88/W89/W90 was
   the right discipline: it rejected the marginal P2 result
   before P2b had a chance to disconfirm it.

W91 P2b is the cleanest possible disconfirmation under the
W88/W89/W90 anti-cheat discipline.  No seed cherry-picking, no
prompt-fishing, no baseline-weakening.

## Cumulative cross-modal evidence (W88 → W89 → W90 → W91)

| Run | VLM | Code-LM | Strip mode | Arm shape | Seeds × Probs | A1 | B | B−A1 | B>A1 seeds | Retire? |
|---|---|---|---|---|---|---:|---:|---:|---:|---|
| W88 V1 | 11B-V | 8B | doctest_only | VLM-extract+code-LM | 3×12 | 86.1 % | 80.6 % | −5.56 pp | 0/3 | No |
| W89 P2 | 90B-V | 8B | all_docstring | VLM-extract+code-LM | 3×12 | 86.1 % | 58.3 % | −27.78 pp | 0/3 | No |
| W89 P3 | 90B-V | 70B | doctest_only | VLM-extract+code-LM | 3×12 | 91.7 % | 86.1 % | −5.56 pp | 0/3 | No |
| W90 P2 | 90B-V | (8B floor) | doctest_only | VLM-in-loop | 3×12 | 91.7 % | 91.7 % | +0.00 pp | 1/3 | No |
| **W91 P2** | **90B-V** | (8B floor) | **all_docstring** | **VLM-in-loop** | **3×12** | **83.3 %** | **86.1 %** | **+2.78 pp** | **2/3** | **No (margin bar fails)** |
| **W91 P2b** | **90B-V** | (8B floor) | **all_docstring** | **VLM-in-loop** | **7×12** | **84.5 %** | **77.4 %** | **−7.14 pp** | **2/7** | **No (disconfirms P2)** |

**The cumulative evidence at the 7-seed scale** is decisive:
across 5 cross-modal configurations spanning multiple model
scales, architectures, and corpus regimes, the cross-modal
team has NOT been demonstrated to strictly beat the unified
VLM at fair K=5 budget by ≥ +5 pp with per-seed majority.  The
W90 P2 doctest_only result tied at +0.00 pp.  The W91 P2 +2.78
pp at 3 seeds was variance.  The W91 P2b 7-seed extension
showed the underlying mean is NEGATIVE (−7.14 pp).

## What W91 retires

Nothing.

## What W91 contributes

* **`W91-L-CROSS-MODAL-VLM-LOOP-V2-3-SEED-VARIANCE-CAP`** —
  at 3 seeds × 12 problems × `all_docstring` × VLM-in-loop,
  the mean B−A1_vlm landed at +2.78 pp (5/6 bars met) but the
  follow-up 7-seed run showed the underlying mean is
  approximately −7.14 pp.  The 3-seed result was variance-
  driven.  Future cross-modal benches should use ≥ 5 seeds to
  reduce this variance risk.
* **`W91-L-CROSS-MODAL-VLM-LOOP-V2-DECISIVE-NEGATIVE-CAP`** —
  at 7 seeds × 12 problems × `all_docstring` × VLM-in-loop,
  B 77.4 % loses to A1_vlm 84.5 % by 7.14 pp; B wins only 2 of
  7 seeds.  This is the **most decisive negative evidence on
  the cross-modal team-organisation question to date** — the
  W88/W89 W90 W91 cumulative finding is that cross-modal team
  organisation is NOT load-bearing-better than unified VLM at
  fair K=5 budget across the configurations tested.  Future
  attempts must change something more fundamental: K budget,
  benchmark choice, or substrate-level cross-modal injection
  (the W87 carry-forward direction).
* **Pre-commit discipline validated.**  The +5 pp margin bar
  correctly rejected the P2 marginal result before P2b had a
  chance to disconfirm it.  The W88 / W89 / W90 retirement bars
  ARE the right discipline: they prevented W91 P2 from being
  prematurely declared a retirement.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* W91 reuses `coordpy.cross_modal_vlm_loop_bench_v1` unchanged
  (no new modules; no new tests; the existing audit chain
  surfaces handled both runs unchanged).

## Anti-cheat re-statement

All W88/W89/W90 anti-cheat clauses carry forward.  W91-specific:

* W91 P2 (3 seeds) is committed AS-IS, even though the
  follow-up disconfirmed it.  This is the honest pre-commit
  +5 pp margin bar successfully discriminating
  variance-driven signal from structural signal.
* W91 P2b uses 7 fresh seeds (90_046_001 through 90_046_007).
  Same VLM, same architecture, same K=5 budget, same
  retry policy.
* No selective retries; both runs are committed as their full
  Merkle-rooted reports.

## Re-running

```bash
# W91 P2 (3 seeds)
python scripts/run_w90_cross_modal_vlm_loop_bench.py \
    --vlm-model meta/llama-3.2-90b-vision-instruct \
    --text-model meta/llama-3.1-8b-instruct \
    --n-problems 12 --n-seeds 3 \
    --strip-mode all_docstring \
    --out-dir results/w91/cross_modal_vlm_loop_all_docstring

# W91 P2b (7 seeds)
python scripts/run_w90_cross_modal_vlm_loop_bench.py \
    --vlm-model meta/llama-3.2-90b-vision-instruct \
    --text-model meta/llama-3.1-8b-instruct \
    --n-problems 12 --n-seeds 7 \
    --strip-mode all_docstring \
    --out-dir results/w91/cross_modal_vlm_loop_all_docstring_7seeds

python scripts/verify_w90_cross_modal_vlm_loop_audit_chain.py \
    --run-dir <run-dir>
```

NIM provider-side sampling at T=0.7 carries variance; the
per-seed deltas may differ on a fresh re-run.  The conclusion
("VLM-in-loop at all_docstring does NOT strictly beat A1_vlm
at K=5 with +5 pp margin") is the stable claim and is robust
to expected sampling variance at 7 seeds.

## The honest claim this run earns

**W91 P2 + W91 P2b together rule out the "VLM-in-loop at
all_docstring beats unified VLM at fair K=5 budget" claim at
the +5 pp margin bar.**  Specifically:

* At 7 seeds × 12 problems × Llama-3.2-90B-Vision-Instruct ×
  K=5 × `strip_mode=all_docstring`, B sequential reflexion
  mean pass@1 is **77.4 %**, below A1_vlm 84.5 % by **−7.14 pp**.
* The cross-modal carry-forward
  `W88-L-CROSS-MODAL-CODE-V1-SPLIT-NOT-LOAD-BEARING-CAP` is
  **strongly preserved** by W91 P2b's evidence.
* The W90 P2 +0.00 pp tie (doctest_only) and the W91 P2 +2.78
  pp (3-seed at all_docstring) are now seen as the upper edge
  of variance, not real signal.
* **Image-load-bearing remains PROVEN** across all 6
  configurations (W88 + W89 P2 + W89 P3 + W90 P2 + W91 P2 +
  W91 P2b) — B beats A0_text by +13.9 / +16.7 / +52.8 /
  +16.7 / +41.7 / +34.5 pp.  The W87 multi-modal substrate
  carries real load-bearing information; the cross-modal
  TEAM organisation does not (at the K=5 budget on
  HumanEval-Visual on the VLMs tested).
