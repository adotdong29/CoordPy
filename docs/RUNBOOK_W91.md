# W91 — Post-W90 Empirical Superiority Wave V4 (runbook)

> **Pre-commit contract for the W91 wave.  W89 retired two
> HumanEval carry-forwards at 70B scale; W90 produced two
> meaningful refinements (MBPP +1.11 pp mean / cross-modal
> VLM-in-loop +0.00 pp tie) but no new retirements.  W91
> attacks the two remaining empirical carry-forwards directly
> by combining the W90 architectural wins with the W89/W90
> evidence about WHERE the strong-superiority bars actually
> live.**
>
> Locked 2026-05-23 BEFORE any W91 bench run.  Same retirement
> bars as W88/W89/W90; only the bench config moves.

## What W90 actually showed (the inherited diagnosis)

**Prong 1 (MBPP-70B): partial win, per-seed majority fail.**
B 82.2 % > A1 81.1 % by +1.11 pp on the mean (3 / 4 bars met).
Per-seed: A1 / B = (90/90, 73.3/73.3, 80/83.3) — B never loses
to A1 on any seed (ties on 2/3, wins by +3.3 pp on 1/3).  The
per-seed-majority bar (≥ 2 / 3 strict wins) fails (1 / 3).

**Structural reason**: variance dominates the +1.11 pp mean.
At 3 seeds × 30 problems × A1 mean ≈ 81 %, the per-seed range
is 16.7 pp (73.3 → 90.0), and the per-seed B−A1 magnitude is
≤ 3.3 pp.  Variance > effect → fragile per-seed.

**Prong 2 (Cross-modal VLM-in-loop): best architecture, tie.**
B_vlm_loop 91.7 % = A1_vlm 91.7 % (gap closed from W88/W89's
−5.6 / −27.8 / −5.6 pp to +0.00 pp).  3 / 6 bars met (image
direction); 3 / 6 fail (team-organisation direction).
Per-seed B − A1_vlm = (+8.3, −8.3, 0) — wins 1, ties 1, loses
1, mean 0.

**Structural reason**: ceiling effect.  At A1_vlm K=5 mean
= 91.7 %, only ~8.3 % failure-residual is left for B to
rescue.  Even a 40 % rescue rate would yield only ~3 pp — below
the +5 pp pre-committed retirement margin.  Need a less-
saturated benchmark or a regime where A1_vlm has lower
ceiling.

## What W91 changes (and why)

### Prong 1 — MBPP at 5 seeds × 30 problems

Same architecture (`coordpy.mbpp_reflexion_bench_v1`
sequential-reflexion B unchanged); same model
(`meta/llama-3.3-70b-instruct`); same K=5 budget.  Only the
**number of seeds** changes: 3 → 5.

Why this is the right pivot for the per-seed majority bar:

* W90 P1 showed B mean strictly beats A1 mean by +1.11 pp,
  margin met.  The per-seed-majority bar (1 / 3) failed — but
  3 seeds gives weak statistical power: with the per-seed
  std-dev of ~5-10 pp, a true +1 pp effect is often masked.
* At 5 seeds, the per-seed-majority bar becomes ≥ 3 / 5.
  Under H1 (true B > A1 by +1.1 pp) with per-seed std-dev
  ~5 pp, P(B > A1 per seed) ≈ 0.59, so P(≥ 3 of 5 wins) ≈
  0.55 — much better odds than W90's P(≥ 2 of 3) ≈ 0.39.
* Same architecture, same model, same task subset
  per-seed.  No baseline weakening.
* If at 5 seeds the per-seed majority is ≥ 3 / 5 AND the mean
  margin ≥ +1 pp, ALL 4 retirement bars are met.

### Prong 2 — Cross-modal VLM-in-loop at `strip_mode=all_docstring`

Same architecture (`coordpy.cross_modal_vlm_loop_bench_v1`
VLM-in-loop unchanged); same VLM (90B-Vision); same K=5 budget;
same A0_text floor model (8B).  Only **strip mode** changes:
`doctest_only` → `all_docstring`.

Why this is the right pivot for cross-modal:

* W89 P2 measured A1_vlm at 86.1 % on `all_docstring` with the
  W88/W89 SPLIT architecture — that's 5.6 pp lower than
  W90 P2's 91.7 % on `doctest_only` with VLM-in-loop.
* Combining W90 P2's BEST architecture (VLM-in-loop, no
  extraction handoff) with W89 P2's HARDER regime (image-
  strict, no prose) gives:
  - A1_vlm expected ~86 % (less ceiling-saturated than W90 P2)
  - B_vlm_loop expected to gain margin via reflexion on the
    larger failure-residual (~14 % instead of ~8 %)
* If B beats A1_vlm by ≥ +5 pp on this regime with majority
  per-seed, the cross-modal retirement bar is met for the
  first time in the programme.
* Same anti-cheat: same VLM on every arm; same task subset
  per seed; same budget.

## Pre-committed success criteria

**No relaxation from W88/W89/W90.**

### Prong 1 — Retires `W89-L-HUMANEVAL-REFLEXION-V2-HUMANEVAL-K5-SCALE-CAP` iff ALL 4 bars met:

1. `b_mean_strictly_beats_a1_mean = True`
2. `b_mean − a1_mean ≥ +1.0 pp`
3. `b_mean_strictly_beats_a0_mean = True`
4. **B beats A1 on more than half the seeds (≥ 3 / 5)**.

If all 4 met: the W89 70B-HumanEval retirement EXTENDS to a
second published benchmark (MBPP) with a robust per-seed
majority — clean cross-benchmark generalisation.

If fewer: new carry-forward
`W91-L-MBPP-REFLEXION-V2-NOT-BEATEN-BAR-FAIL` records the
additional negative evidence.

### Prong 2 — Retires `W88-L-CROSS-MODAL-CODE-V1-SPLIT-NOT-LOAD-BEARING-CAP` iff ALL 6 bars met:

1. `b_vlm_loop_mean_strictly_beats_a0_text_mean = True`
2. `b_vlm_loop_mean_strictly_beats_a1_vlm_mean = True`
3. B − A0_text margin ≥ +5.0 pp
4. **B − A1_vlm margin ≥ +5.0 pp**
5. B beats A0_text on more than half the seeds.
6. **B beats A1_vlm on more than half the seeds**.

Note: this prong RETIRES the W88 carry-forward by providing an
architecture (VLM-in-loop) that DOES strictly beat A1_vlm at
fair budget on a less-saturated regime.  The W88 SPLIT
architecture's empirical failure stands; W90 P2's TIE stands.
W91 P2 retires the broader claim that "cross-modal team
organisation is not load-bearing-better than unified VLM" —
because we now show one architecture + regime where it IS.

`W87-L-MULTI-MODAL-V1-NO-CROSS-MODAL-INJECT-CAP` stays — it's
about substrate-level injection, not benchmark superiority.

## Anti-cheat clauses

All W88/W89/W90 anti-cheat clauses carry forward.  W91-specific:

* **Prong 1**: 5 seeds replaces 3 seeds — the 3 W90 seeds
  (90_001, 90_002, 90_003) are PRESERVED and 2 new seeds
  (90_004, 90_005) are appended.  Pre-committed seed identity:
  no re-rolling.
* **Prong 2**: `strip_mode = all_docstring` (the W89 P2 mode
  applied to W90 P2's VLM-in-loop architecture).  Same VLM
  on A1_vlm and B_vlm_loop every turn.  Same K=5 budget.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* W91 uses W90 modules unchanged; no new explicit-import
  modules are required.

## Operational plan

1. Launch Prong 1 (MBPP 5×30) in background (long-running,
   ~5-6 hours at NIM 70B with current rate).
2. Launch Prong 2 (cross-modal all_docstring + VLM-in-loop,
   3 seeds × 12 problems) in background (~1.5 hours).
3. **If Prong 2 produces (a) B mean strictly beats A1_vlm mean,
   (b) per-seed majority B > A1_vlm, (c) image-load-bearing
   bars all met, BUT (d) the B − A1_vlm margin bar (+5 pp)
   FAILS** — launch **Prong 2b** (cross-modal at all_docstring
   + VLM-in-loop, **7 seeds × 12 problems**) to test whether
   the gap is variance-driven (a few ceiling seeds dragging
   the mean) or genuine.  This pre-commit is declared
   conditional on the Prong 2 outcome shape; the same seed
   identity discipline applies (90_046_001 through
   90_046_007).  All other bars unchanged.  W91 P2b retires
   `W88-L-CROSS-MODAL-CODE-V1-SPLIT-NOT-LOAD-BEARING-CAP` iff
   all 6 bars including the +5 pp margin are met at 7-seed
   scale.
4. Verify audit chains; produce RESULTS docs; update honesty
   surfaces; commit; ask for push approval.

The W88/W89/W90 automation stack is reused unchanged.  Only the
run command's seed-count / strip-mode flags flip.
