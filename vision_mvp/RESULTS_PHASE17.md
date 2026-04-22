# Phase 17 — Task-Generality Probe

Two runs of the same harness on `tasks/numeric_ledger.py` (12 agents
coordinating on numerical conventions: rounding, scale, NaN, overflow,
signed encoding). The first run used the stock event trigger (behavior
fingerprint); the second forced refinement on every higher-tier agent to
isolate the routing mechanism from the trigger.

## Headline

**Split verdict. The *routing* mechanism generalizes. The *trigger* does
not — at least not without task-specific tuning.**

- With the trigger active: all three legs scored identical to round-1
  (0.520). C3 and C7 both fail — no advantage visible.
- With the trigger bypassed (`--force-refine`): CASR beats ablation by
  **+0.160** on total score and **+0.267** on per-pair pass fraction.
  C3 and C7 hold, and by strong margins.

This isolates the generality question cleanly. Causal routing transfers
to a new coordination surface (numerical conventions, not dict schemas).
The event trigger does not transfer unaided — its task-specific probe
battery has to be engineered per surface.

## Run 1 — default trigger (behavior fingerprint)

| Leg | Score | Refined / Skipped | Tokens |
|---|---:|---:|---:|
| round-1 | 0.520 | — | — |
| full | 0.520 | 2 / 5 | 4 423 |
| **casr** | **0.520** | 2 / 5 | 4 003 |
| ablation | 0.520 | **0 / 7** | 3 175 |

All three legs scored identically to round-1. The behavior trigger
skipped 5 of 7 higher-tier agents in full and casr, and *all 7* in
ablation. In ablation the bulletin contains random producers, so the
pair-match lookup in the trigger's probe table never succeeds — the
trigger has no signal and correctly defaults to "skip". But the same
defaulting also happened on 5 of the 7 legitimate casr bulletins
(because LLMs tend to pick cents/half-up/skip/saturate by default, so
many producer/consumer pairs already agree at round-1 and the trigger
rightly sees no drift to correct).

Net effect: CASR had no opportunity to help because refinement almost
never fired.

### Claims at default trigger

| | Claim | Verdict |
|---|---|---|
| C1 | casr < full tok | ✅ −9.5 % |
| C2 | casr ≥ full − 0.05 | ✅ Δ = +0.000 |
| C3 | casr > ablation + 0.05 | ❌ Δ = +0.000 |
| C4 | full ≥ round-1 | ✅ (trivially) |
| C5 | trigger fired ≥ 1 skip | ✅ 5 skips |
| C7 | per-pair CASR vs ablation ≥ +0.20 | ❌ Δ = +0.000 |

## Run 2 — same harness, `--force-refine`

Added a single flag that makes every tier-1 agent call the LLM
regardless of trigger output. Same task, same model, same ablation
seed.

| Leg | Score | Refined / Skipped | Tokens |
|---|---:|---:|---:|
| round-1 | 0.600 | — | — |
| full | **0.640** | 7 / 0 | 6 915 |
| **casr** | **0.640** | 7 / 0 | 5 625 |
| ablation | **0.480** | 7 / 0 | 5 472 |

Ablation *dropped below round-1* (0.60 → 0.48). When the trigger
is bypassed, mirroring the *wrong* producer's convention is strictly
worse than not coordinating at all — a direct measurement of "random
refinement is misinformation." CASR retains full-leg quality at **18.7 %**
fewer tokens.

### Claims at forced refinement

| | Claim | Verdict |
|---|---|---|
| C1 | casr < full tok | ✅ −18.7 % |
| C2 | casr ≥ full − 0.05 | ✅ Δ = +0.000 |
| **C3** | casr > ablation + 0.05 | ✅ Δ = **+0.160** |
| C4 | full ≥ round-1 | ✅ Δ = +0.040 |
| C5 | trigger fired skips | ❌ (forced off) |
| **C7** | per-pair CASR vs ablation ≥ +0.20 | ✅ Δ = **+0.267** |

## Per-pair pass fractions (forced-refine run)

| pair | CASR | ablation |
|---|---:|---:|
| rounding | **1.00** | **0.00** |
| scale | 1.00 | 1.00 |
| nan | 0.67 | 0.67 |
| overflow | 1.00 | 0.67 |
| signed | 0.00 | 0.00 |

- **Rounding** is the clean proof of convention propagation. CASR sent
  the real `round_amount` to `check_rounded`, which mirrored it
  exactly; ablation sent a random producer, `check_rounded` mirrored
  that convention, and the actual `round_amount` used the original,
  unchanged. Every rounding test fails.
- **Overflow** shows the same effect less starkly: CASR 1.00 vs
  ablation 0.67.
- **Scale** and **NaN** converged because the LLM's default is the
  same across both producer and consumer drafts (cents / skip-NaN).
  Neither leg is helped or hurt by routing.
- **Signed encoding** stays at 0.00 in both legs — this convention is
  too structural for `qwen2.5-coder:7b` to mirror reliably from a
  short source preview. An interesting weakness of the model, not the
  routing.

## What the two runs together say

The commercial claim of task generality depends on which of the two
axes you care about:

1. **Causal routing alone is general.** When you force refinement,
   CASR propagates correct conventions on pairs where the model is
   capable of mirroring them (rounding, overflow). Ablation makes
   things worse. The mechanism is not ProtocolKit-specific.

2. **The event trigger is not general.** The Jaccard-on-dict-keys
   trigger from Phase 14 has literally nothing to measure on
   NumericLedger (no dict keys). The behavior-fingerprint trigger I
   built for Phase 17 has the right shape but is too conservative:
   it correctly detects disagreement on a small probe battery per
   pair, but it has no integrator probes at all, and it skips
   agents whose round-1 defaults happen to already match their
   producers (good) but therefore can't distinguish "nothing to fix"
   from "don't know how to check".

## What this means for Phase 18+

The factorization now is:

- **Routing layer** (CASR): transfer-tested. Consider this shipping
  infrastructure.
- **Trigger layer**: task-specific. Each new coordination surface
  needs its own disagreement signal — dict-key Jaccard for schemas,
  behavior probes for numerics, type signatures for functions,
  prefix-token divergence for free text, etc.

A design tax that should surface in the product story: every new
customer task needs ~half a day of trigger engineering. The routing
part they get for free.

A natural Phase 18 would be to take *one* trigger-general approach
(e.g., LLM-as-a-judge: ask a small LLM whether two draft snippets
"agree") and measure whether it matches the per-task triggers at
acceptable cost.

## What this does not say

- Only one model (`qwen2.5-coder:7b`) was tried. Different model
  families might handle signed-encoding conventions better.
- Only one task surface (numerical) was added. A third surface
  (e.g., error-handling conventions, async patterns, formatting)
  would further calibrate the routing-generalization claim.
- The forced-refinement run is an adversarial worst case for the
  trigger; it's not what a real CASR deployment would do. Real
  deployments need a working trigger.

## Reproduce

```bash
# Default trigger (C3/C7 fail):
python3 -m vision_mvp.experiments.phase17_generality \
    --out vision_mvp/results_phase17.json

# Forced refinement — isolate routing from trigger (C3/C7 hold):
python3 -m vision_mvp.experiments.phase17_generality --force-refine \
    --out vision_mvp/results_phase17_forced.json
```

Runtime: ~5 min per run (12 agents × 3 legs × ~10 s/call).
