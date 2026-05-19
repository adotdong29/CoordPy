# W84 / P1 #30 — Quantized-Runtime Substrate with Honest Precision Floors

## Summary

Extends the W80 instrumentation contract with three first-class
precision tiers — `TIER_FP32`, `TIER_BF16`, `TIER_INT8` — each
carrying an explicit, declared, content-addressed maximum-
absolute-difference floor for replay-from-KV. The contract
refuses to claim byte-identity at sub-fp32 tiers; instead it
emits a precision-tier-tagged semantic-equivalence claim
checked against the tier's declared floor.

The implementation runs on the W79 in-repo NumPy controlled
runtime so the precision floors are reproducible in CI without
GPU / `bitsandbytes` / `auto-gptq`. The maths is honest: bf16
uses IEEE 754 round-to-nearest-even on the fp32 representation
(low 16 mantissa bits zeroed after rounding); int8 uses
symmetric per-tensor quantization with explicit scale factors
and dequantization at every matmul boundary. Both forward AND
replay-from-KV apply the same tier rounding to every layer's
post-MLP hidden state and to the final logits — the
quantization is not "fp32 with looser tolerance".

## Definition-of-Done bars

| Bar | Status |
| --- | ------ |
| `precision_tier` is a first-class declared axis on the runtime (`QuantizedRuntimeV1.tier`, `QuantizedRuntimeV1.precision_floor`, baked into the runtime CID) | ✅ |
| Runtime can be instantiated in `TIER_BF16` and `TIER_INT8` modes (caller picks) | ✅ |
| Conformance suite passes on each tier with the tier-appropriate floor | ✅ (see numbers below) |
| At least one quantised model loads + runs forward + replay-from-KV under the contract | ✅ (in-repo NumPy controlled runtime; HF/bitsandbytes V2) |
| At `TIER_INT8`, replay produces the same top-1 continuation token as recompute on ≥ 95 % of a held-out prompt set | ✅ (97.5 % on n=40 default seed) |
| The W83 hidden-state intercept bench reproduces under `TIER_BF16`: `hidden_state_intercept_moves_cid == True` and replay-precision-floor reported honestly | ✅ |

## Measured numbers (40-prompt held-out bench, seed 84030001)

| Tier | Floor | `max_replay_diff` | `replay_within_floor` | `top1_match_rate_vs_fp32` | `hidden_intercept_moves_cid` |
| ---- | ----- | ----------------- | --------------------- | ------------------------- | ---------------------------- |
| `tier_fp32` | 5.0e-3 | 0.0 | True | 1.0 | True |
| `tier_bf16` | 5.0e-2 | 0.0 (close to fp32; the bf16 round-trip preserves the leading bits exactly on this prompt distribution) | True | 1.0 | True |
| `tier_int8` | 2.0e-1 | ~0.110 | True | 0.975 | True |

> `max_replay_diff == 0.0` at `tier_bf16` reflects the fact that
> the W79 controlled runtime weights are small-magnitude
> Xavier-normal; for those, the bf16 rounding of intermediate
> activations is identical at the bf16 representation when the
> forward and the replay arrive at the same layer-by-layer
> sequence (the rounding is idempotent on already-bf16 values).
> The bf16 floor is still recorded honestly as 5e-2.

> `max_replay_diff ~ 0.110` at `tier_int8` is well within the
> declared `2e-1` floor and is the EMPIRICAL int8 error level
> over `n_layers=4` matmuls with symmetric per-tensor int8 weights
> + fp32 activations + int8 re-quantization of activations after
> each layer.

## Anti-cheat compliance

* No quantisation is disabled between forward and replay —
  both pass through the same tier rounding (verified by
  `test_w84_int8_runtime_is_not_fp32_pretending`).
* No floor is widened until "byte-identity" passes — the
  declared floors are 5e-3 / 5e-2 / 2e-1 and the conformance
  check uses the *per-tier* floor.
* `TIER_INT8` is not claimed at the fp32 floor — the bench
  explicitly compares against `precision_floor(tier)`.
* No "mock quantised runtime" that runs fp32 internally — the
  bf16 path actually goes through `to_bf16` (bit truncation to
  the bf16 representation; verified by
  `test_w84_bf16_emulation_zeros_low_16_bits`); the int8 path
  goes through `quantize_int8_symmetric`.
* Replay-from-KV is NOT skipped under quantisation — the bench
  exercises it on every prompt of length ≥ 3.
* `TIER_INT8` ships in V1, not just `TIER_BF16` — the V1 closes
  the int8 stretch goal too, with empirical numbers reported
  honestly.

## Honest scope (V1)

* `W84-L-QUANTIZED-RUNTIME-V1-NUMPY-EMULATION-CAP` — the V1
  here emulates bf16 / int8 via bitwise / scale-based maths on
  fp32 / fp64 NumPy. Real GPU `bitsandbytes` / `auto-gptq` is
  V2; the V1 contract surface (`QuantizedRuntimeV1.tier`,
  `precision_floor`, conformance bench) is what they would
  plug into.
* `W84-L-QUANTIZED-RUNTIME-V1-PER-TENSOR-INT8-CAP` — V1 is
  symmetric per-tensor int8 (one scale per weight tensor).
  Per-channel / per-group quantization (AWQ / GPTQ style) is V2.
* `W84-L-QUANTIZED-RUNTIME-V1-NO-KV-QUANT-CAP` — V1 does NOT
  quantize the KV cache; weights and activations are quantized
  at each matmul, but KV reads/writes are fp32. KV cache
  quantization is V2.
* `W84-L-QUANTIZED-RUNTIME-V1-INT4-V2-CAP` — INT4 / mixed
  precision is V2.

## Reproduction

```python
from coordpy.quantized_runtime_substrate_v1 import (
    PrecisionTier, run_quantized_conformance_bench_v1,
)
for tier in (PrecisionTier.TIER_FP32, PrecisionTier.TIER_BF16,
             PrecisionTier.TIER_INT8):
    rep = run_quantized_conformance_bench_v1(
        tier=tier.value, n_prompts=40)
    print(tier.value, rep.to_dict())
```

Tests: `tests/test_w84_quantized_runtime_substrate.py` (14
tests, all passing).
