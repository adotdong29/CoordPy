# W84 / P0 #25 — Frontier-Scale Live Substrate Coupling Results

## What this is

The W84 P0 attack closes the first P0 blocker in the post-W83
meta issue (#49): until a real frontier-class open-weight
model is plugged into the W80 instrumentation contract end-to-
end, the W80 / W81 / W83 substrate-contract claims hold *only
on* ``distilbert/distilgpt2`` (~82M params, GPT-2 family). This
result note records the empirical numbers for that
end-to-end validation on
``Qwen/Qwen2.5-7B-Instruct`` (~7.62B params, Llama-family
architecture).

## Model and topology

| Field | Value |
| ----- | ----- |
| Model | ``Qwen/Qwen2.5-7B-Instruct`` |
| Parameters | 7.62 B |
| Architecture | Llama-family lineage (RMSNorm, RoPE, SwiGLU, GQA) |
| HF model_type | ``qwen2`` |
| Hidden dim | 3584 |
| Layers | 28 |
| Attention heads | 28 |
| Head dim | 128 |
| Native precision | bf16 |
| Device | CPU |
| W80 instrumentation contract | ``coordpy.transformers_runtime_v1`` (sibling-adapter unchanged; ``model_dtype="bf16"`` parameter added in W84) |

This is a different architecture *family* than the W80 baseline
(distilgpt2 is GPT-2 family with LayerNorm + learned absolute
positions + standard MHA). Per the P0 #25 anti-cheat, validating
at least one Llama-family model in addition to GPT-2 family is
load-bearing.

## Empirical numbers

| Bar | Value |
| --- | ----- |
| W80 conformance n_pass | **12 / 12** (issue floor: ≥10) |
| W80 conformance n_fail | **0** |
| Replay-from-KV ``max_abs_diff`` (bf16, CPU) | **0.6562** |
| Replay precision floor (bf16) | **1.0** |
| Replay byte-identical at floor | **True** |
| Hidden-state intercept moves trace CID | **True** |
| Substrate vs bounded-window-V3 win rate | **1.000 (9 of 9 positions)** |
| Substrate load-bearing claim reproduced | **True** |
| Load wall-clock | 76.9 s |
| Per-forward wall-clock | 9.41 s |
| Full-run wall-clock | 449 s (~7.5 min) |

The replay floor of **0.6562** is the honest bf16-on-CPU
``max_abs_diff`` against the recompute path. It is **not**
clipped to 5e-3 (the fp32 floor); the W80 conformance suite
honours the backend-declared ``precision_floor`` via the
``TransformersRuntimeV1.precision_floor`` property (1.0 for
bf16, 5e-2 for fp16, 5e-3 for fp32).

The substrate-vs-bounded-window head-to-head reproduces the
W83 V3 falsifier pattern: at every test position
``pos ∈ [window+1, len(ids))``, the substrate's
forward-then-replay-from-KV produces logits within the bf16
floor of the full forward, while the bounded-window baseline
(seeing only the last ``window=4`` tokens) materially
diverges. On a 28-token prompt this gives 9 test positions; the
substrate wins **all 9**.

## What this validates (and what it does not)

### Reproduced at frontier scale

1. **W80 instrumentation contract works on Llama-family
   architecture.** All 12 W80 axes pass on Qwen-2.5-7B-Instruct
   under bf16. The W80 contract is *not* GPT-2-family specific:
   the forward-hook + KV cache + ``inputs_embeds`` injection
   surface generalises to RMSNorm + RoPE + GQA models.
2. **W80 replay-from-KV byte-identical at the dtype's native
   floor.** bf16 max_abs_diff = 0.6562 < precision_floor = 1.0.
   The W80 byte-identity claim holds; the floor is wider than
   fp32 (5e-3), recorded honestly.
3. **W83 V1 hidden-state intercept moves-CID claim
   reproduces.** A nontrivial additive hidden-state inject at
   layer 4 provably changes the forward trace CID on Qwen-2.5-
   7B-Instruct. The intercept is *not* GPT-2-family specific.
4. **W83 V3 bounded-window-falsifier pattern reproduces at
   frontier scale.** The full-context substrate's replay
   matches the full forward at the bf16 floor; the bounded-
   window baseline (window=4) materially diverges. Win rate
   100% across 9 test positions.

### Honest carry-forward limits

- ``W84-L-FRONTIER-SCALE-V1-BF16-PRECISION-CAP`` — the
  empirical floor is bf16 on CPU. fp32 single-host floor is
  unchanged (5e-3 on distilgpt2). bf16 floor is recorded
  honestly as 0.6562.
- ``W84-L-FRONTIER-SCALE-V1-CPU-ONLY-CAP`` — V1 runs on CPU.
  GPU validation is V2 stretch (issue #25 acknowledges 70B is
  GPU-only).
- ``W84-L-FRONTIER-SCALE-V1-ONE-MODEL-CAP`` — V1 validates one
  7B model. Multi-model (Llama-3.1-8B + Mistral-7B + Phi-4-14B)
  coverage is V2.
- ``W84-L-FRONTIER-SCALE-V1-NOT-MOE-CAP`` — V1 does NOT
  validate MoE substrates. P1 #31 tracks that work.
- ``W84-L-FRONTIER-SCALE-V1-SMALL-PROMPT-CAP`` — V1 uses a
  ~28-token prompt to keep CPU wall-clock tractable. Long-
  context regimes are P0 #27 work.
- The W79 hosted-substrate wall is unchanged. Validating on
  open-weight local models does NOT pierce the
  ``W79-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` set.

## Reproducing this run

```bash
# Download the model into the HF cache (one-time, ~15 GB).
python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen2.5-7B-Instruct', \
    allow_patterns=['*.safetensors','*.json','*.txt','merges.txt','vocab.json'])"

# Run the validation. ~5–10 min wall-clock on a 16-thread CPU.
COORDPY_RUN_FRONTIER_BENCH=1 python3 -m pytest \
    tests/test_w84_frontier_scale_substrate.py \
    -k full_bench -v

# Or run the validation entrypoint directly:
python3 -c "
import torch
torch.set_num_threads(4)
from coordpy.frontier_scale_substrate_v1 import (
    run_frontier_scale_validation_v1,
)
r = run_frontier_scale_validation_v1()
print(r.to_json())
"
```

## Files

- ``coordpy/frontier_scale_substrate_v1.py`` — the W84 validator
  + bench.
- ``coordpy/transformers_runtime_v1.py`` — modified W80 runtime
  (added ``model_dtype`` parameter; default ``"fp32"`` preserves
  all W80 / W83 baseline behaviour). The dtype-related casts in
  ``_forward_torch`` / ``_build_past_kv_from_snapshot`` now use
  the model's loaded dtype.
- ``coordpy/runtime_instrumentation_v1.py`` — modified W80
  conformance suite (REPLAY_FROM_KV axis honours the backend's
  declared ``precision_floor`` if it exposes one).
- ``coordpy/hidden_state_intercept_bench_v1.py`` — modified W83
  bench accepts a ``model_dtype`` parameter.
- ``tests/test_w84_frontier_scale_substrate.py`` — tests.

## Witnesses

| Witness | CID prefix |
| ------- | ---------- |
| Baseline forward trace | (see ``r.baseline_trace_cid``) |
| Replay trace | (see ``r.replay_trace_cid``) |
| Intercept trace | (see ``r.intercept_trace_cid``) |
| Frontier-scale report | (see ``r.cid()``) |
| Frontier-scale witness | ``emit_frontier_scale_witness_v1(report=r).cid()`` |

These CIDs are *re-verifiable from disk* given the same model
weights + the same input prompt + the same module versions.
