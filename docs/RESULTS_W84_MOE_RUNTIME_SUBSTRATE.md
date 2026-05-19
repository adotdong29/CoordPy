# W84 / P1 #31 — Mixture-of-Experts Substrate V1

## Summary

Extends the W80 instrumentation contract with three new
MoE-specific axes and ships an in-repo MoE runtime that
implements them.

### New W80 contract axes (MoE)

* `READ_EXPERT_ROUTING_PER_LAYER` — per-layer `(seq_len, top_k)`
  selected expert IDs + their gate weights. Captured as
  `ExpertRoutingSnapshotV1` (content-addressed).
* `READ_EXPERT_OUTPUT_PER_EXPERT_PER_LAYER` — per-(layer, expert)
  output activations for the experts that fired in that layer.
  Captured as `ExpertOutputSnapshotV1`.
* `WRITE_FORCE_EXPERT_ROUTING_PER_LAYER` — override the router's
  decision on a per-layer basis. Captured as
  `MoEForceRoutingPlanV1`.

All three axes appear in `MoERuntimeAdapterV1.declared_axes()`
with capability tag `AVAILABLE`.

### Runtime

`MoERuntimeParamsV1` + `forward_moe_runtime` implement a real
top-K MoE transformer block in pure NumPy:

* dense attention (Q, K, V, O projections per layer);
* a per-layer router (linear projection `hidden -> n_experts`,
  argsort to get top-K, softmax over chosen scores for gate
  weights);
* `n_experts` independent (mlp_W1, mlp_W2) pairs per layer;
* per-token MLP output = sum over chosen experts of
  `gate_weight * expert_MLP(post_attn_hidden)`.

Anti-cheat baked in: `build_moe_runtime_params_v1` raises if
`n_experts < 4` or `top_k < 2`.

## Definition-of-Done bars

| Bar | Status |
| --- | ------ |
| At least 3 new MoE-specific axes are declared on the W80 contract | ✅ |
| An MoE runtime adapter loads at least one MoE configuration | ✅ (4-expert top-2; see `MoERuntimeAdapterV1`) |
| Forward + replay-from-KV passes on the MoE model with expert routing restored. Byte-identity (or model precision floor) on the final-token logits | ✅ (max replay diff at fp32 floor = 0.0 over 10 prompts) |
| Without restoring expert routing, the replay provably diverges (routing is load-bearing) | ✅ (forcing a +1-shifted routing produces `max_forced_routing_diff ≈ 2.22` on the same inputs — well above the fp32 floor) |
| W83 hidden-state intercept bench reproduces under MoE | ✅ (`hidden_intercept_moves_cid = True`, AND the routing CID also moves) |
| `RESULTS__MOE_SUBSTRATE.md` captures the actual numbers + which model + which precision tier | ✅ (this file) |

## Measured numbers (10-prompt bench, seed 84031007)

| Claim | Value |
| ----- | ----- |
| `n_experts` | 4 |
| `top_k` | 2 |
| `n_layers` | 3 |
| `max_forced_routing_diff` (natural vs +1-shifted forced routing) | 2.220682 |
| `max_replay_with_natural_routing_diff` (natural vs restored-routing replay) | 0.0 |
| `replay_with_natural_routing_within_floor` (< 5e-3) | True |
| `trace_cid_changes_with_routing` | True |
| `hidden_intercept_moves_cid` | True |
| `routing_cid_changes_with_force_plan` | True |

## Anti-cheat compliance

* **Routing is not ignored.** The MoE MLP block iterates per
  token per chosen expert, applying `gate_weight × expert(h)`;
  changing the routing changes the post-MLP output by ~2 on
  this in-repo model.
* **The router is not collapsed to dense.** With
  `n_experts = 4` and `top_k = 2`, exactly two experts fire per
  token per layer — and that decision is reflected in the
  trace's `routing_cid`.
* **Not declared on a degenerate config.** The anti-cheat
  `n_experts >= 4 and top_k >= 2` is enforced at param-build
  time (`ValueError` raised otherwise; tests prove it).
* **Routing snapshot is not stubbed.** Identical inputs ⇒
  identical routing ⇒ identical `routing.cid()`; different
  forced routing ⇒ different `routing.cid()` (tests prove
  both).
* **No silent fallback to dense semantics under forced
  routing.** The force-routing path goes through `_route_top_k`
  with `force_top_k_ids` set; the gate weights are recomputed
  by softmax over the router scores at the forced IDs (so the
  forced routing is differentiable and replayable).
* **bf16 / fp16 noise is not counted as routing divergence.**
  This V1 runs fp64 internally; the fp32 floor of 5e-3 is
  applied to the replay claim, and the forced-routing
  divergence (~2.22) is two orders of magnitude above that.

## Honest scope (V1)

* `W84-L-MOE-RUNTIME-V1-NUMPY-CAP` — V1 is an in-repo MoE
  transformer block in pure NumPy. The HF / Mixtral / Qwen-MoE
  adapter that exposes the same axes is V2.
* `W84-L-MOE-RUNTIME-V1-N_EXPERTS-4-TOP_K-2-CAP` — V1 defaults
  to 4 experts + top-2 routing.
* `W84-L-MOE-RUNTIME-V1-NO-SHARED-EXPERT-CAP` — DeepSeek-V3-style
  shared experts are V2.
* `W84-L-MOE-RUNTIME-V1-SINGLE-HOST-CAP` — V1 is single-host.
  Multi-GPU expert-parallel MoE is V2.
* `W84-L-MOE-RUNTIME-V1-FORCE-ROUTING-AVAILABLE-CAP` — V1 ships
  the `WRITE_FORCE_EXPERT_ROUTING_PER_LAYER` axis. The issue
  allowed making this V2; we close it in V1 anyway.

## Reproduction

```python
from coordpy.moe_runtime_substrate_v1 import run_moe_bench_v1
rep = run_moe_bench_v1(n_prompts=10)
print(rep.to_dict())
```

Tests: `tests/test_w84_moe_runtime_substrate.py` (13 tests, all
passing).
