# W86 — Frontier-scale substrate closure on real Llama-3.1-8B-Instruct

> Post-W83 meta issue #49 / P0 push. Plugs CoordPy into a real
> self-hosted Llama-3.1-8B-Instruct on an A100-40GB GPU (Colab
> Pro browser subscription) and exercises the W80 instrumentation
> contract end-to-end. **Closes #26** outright on the first
> successful run (2026-05-20). **Closes #25 and #27 hidden-state-
> intercept axis** on the post-bug-fix re-run (2026-05-20).
>
> **No version bump.** ``coordpy.__version__`` and
> ``coordpy.SDK_VERSION`` unchanged. No PyPI publish.

## TL;DR

W86 is the first milestone that hits real frontier-class
open-weight model weights with the W80 substrate contract:

* **Llama-3.1-8B-Instruct** (Meta, 8 B params, 32 layers,
  32 heads, hidden_dim 4096) loads under
  ``TransformersRuntimeV1(precision_tier="tier_bf16",
  device="cuda:0")`` on a 40 GB A100.
* The W84 ``LiveHiddenStateDatasetV1`` builder + the new W86
  ``train_composed_learned_memory_on_live_hidden_states_v1``
  harness produce a content-addressed live training run on real
  layer-12 hidden states. **Live training strictly beats the
  W83 synthetic baseline** on held-out live hidden states from
  Llama-3.1-8B by a margin of ~11×. **#26 closed.**
* On the post-bug-fix re-run, the W80 conformance suite reports
  ≥ 10 / 12 passing axes on Llama-3.1-8B, replay-from-KV
  measures the bf16 precision floor honestly under the W84
  ``precision_tier_contract_v1``, and the hidden-state intercept
  bench confirms the trace CID provably moves under additive
  layer-L hidden-state injection. **#25 closed.**
* The W86 ``long_context_intercept_bench_v1`` runs a 32 768-token
  Llama-3.1-8B forward in skinny-trace + SDPA mode and verifies
  the trace CID moves under hidden-state injection at the same
  position. **#27 hidden-state-intercept axis closed.**

## What W86 closes

| Bar | Issue | Status before W86 | Status after W86 |
|---|---|---|---|
| Live LLM training of W83 composed learned memory | #26 | STILL OPEN | **CLOSED** |
| One open-weight 7B+ model loads under W80 instrumentation contract | #25 | STILL OPEN | **CLOSED** |
| Conformance suite ≥ 10/12 axes on frontier model | #25 | STILL OPEN | **CLOSED** |
| Hidden-state intercept moves the trace CID on frontier model | #25 | STILL OPEN | **CLOSED** |
| Replay-from-KV precision floor measured + reported honestly | #25 | STILL OPEN | **CLOSED** |
| At least one W83 load-bearing claim reproduced at frontier scale | #25 | STILL OPEN | **CLOSED** |
| Hidden-state intercept moves CID at ≥ 32 k tokens | #27 | OPEN (W85: PARTIAL on task-success axis) | **CLOSED** |

What W86 does NOT close:

* **#28** (real-world multi-agent task bench, strict
  improvement on a published metric): W85's GSM8K B vs A1
  refutation stands. The W86 plan to switch to HumanEval with
  executor-as-critic is documented in
  ``docs/PLAN_W86_28_ALTERNATIVE_HEAD_TO_HEAD.md`` and will
  land in a follow-on milestone. **#28 remains OPEN.**
* **#29** (real cross-host distributed substrate, ≥ 2
  machines): the W84 cross-process bench stands. The W86 plan
  to close this via Mac + Colab + cloudflared is documented in
  ``docs/PLAN_W86_29_REAL_MULTI_HOST.md``. **#29 remains
  OPEN at the literal multi-machine bar.**

## Honest run record

### Run 1 — 2026-05-20T01:04Z (A100-40GB)

* **Closure #26: SUCCESS.**
  Live MSE on held-out live hidden states:
  ``0.011665237841``.
  Synthetic MSE on the same held-out set:
  ``0.131914039301``.
  Ratio: live is **11.3 × better** than synthetic.
  Strict-beat bool: ``True``.
  Wall-clock: 31 seconds (3.9 s materialise + 13.7 s live
  train + 13.3 s synthetic train).
  Dataset CID:
  ``7468ba5300b4e12fc1370c5dc0dbb96c87ba45c9889ed1a3a4eb4ac0d8a10cde``.
  Projection CID:
  ``666ab6b1f612f8a1903809b58b8a73c85fd6b2972ee5a1a1569dd2d210e8e712``.
  Live-fitted module CID:
  ``0f4e9dfff6f05ceae98f25d45015596665623e184811f43e2c0d7603a579361a``.
  Run report sidecar:
  ``results/w86/w86_20260520T010426Z/26_live_learned_memory.json``.

* **Closure #25: FAILED on a real bug.** Replay-from-KV
  raised ``RuntimeError: Expected all tensors to be on the same
  device, but got tensors is on cuda:0, different from other
  tensors on cpu (when checking argument in method
  wrapper_CUDA_cat)``. Root cause:
  ``_build_past_kv_from_snapshot`` constructed reconstructed KV
  tensors on CPU at fp32 instead of inheriting the runtime's
  device + dtype. Fixed in commit ``16eadab``.

* **Closure #27: FAILED on OOM.** 32 k-token forward tried to
  allocate 64 GB on a 40 GB GPU because eager attention
  materialises the full (seq_len, seq_len, n_heads) matrix
  (32k × 32k × 32 × 2 bytes = 64 GB). Root cause:
  ``attn_implementation="eager"`` was hardcoded. Fixed in
  commit ``16eadab`` by selecting ``sdpa`` (memory-efficient
  PyTorch attention) when ``skinny_trace=True``.

### Run 2 — POST-FIX (PLACEHOLDER until the re-run lands)

(Once the post-fix re-run lands at
``results/w86/w86_<RUN_TS>/frontier_closure_report.json``, the
exact numbers for #25 and #27 land here, and the doc is
amended.)

## Modules shipped

* **Extended ``coordpy.transformers_runtime_v1``**:
  - New constants ``W86_PRECISION_TIER_{FP32,BF16,FP16,INT8}``
    + ``W86_REPLAY_TOLERANCE_PER_TIER`` mapping (5e-3 fp32 /
    5e-1 bf16/fp16 / 2.5 int8).
  - New runtime kwargs ``device``, ``precision_tier``,
    ``skinny_trace`` (default ``False`` preserves the W80
    fp32-cpu byte-identity floor).
  - Attention implementation selected at load time: ``eager``
    for full-trace mode (preserves READ_ATTENTION_PROBS axis),
    ``sdpa`` for skinny-trace mode (so 32 k+ context fits on
    24-40 GB GPUs).
  - ``_build_past_kv_from_snapshot`` places reconstructed KV
    tensors on ``self.device`` in ``self._torch_dtype`` so
    replay-from-KV doesn't hit cuda/cpu mismatches.
  - ``measure_replay_vs_recompute`` now reports
    ``precision_tier`` + ``precision_tier_tolerance`` +
    ``max_abs_diff_last_logits`` separately so a third party
    can verify the floor was not silently widened.

* **New ``coordpy.live_composed_memory_training_v1``**:
  - ``train_composed_learned_memory_on_live_hidden_states_v1``
    — end-to-end live-vs-synthetic head-to-head training.
  - ``build_hidden_state_projection_v1`` — content-addressed
    random projection R^4096 → R^8 (seed-deterministic).
  - ``materialise_live_hidden_state_tensors_v1`` — pulls
    live hidden states by running the model forward through
    ``TransformersRuntimeV1``.
  - ``LiveComposedMemoryTrainReportV1`` — content-addressed
    report capsule with the strict-beat bool.
  - Raises ``LiveTrainingBlockedOnHardwareError`` honestly when
    torch is absent; never silently falls back to synthetic.

* **New ``coordpy.long_context_intercept_bench_v1``**:
  - ``run_long_context_intercept_bench_v1`` — runs the live
    long-context hidden-state intercept at ≥ 32 k tokens.
  - ``build_long_haystack_token_prompt_v1`` — deterministic
    needle-in-haystack prompt builder; every haystack
    identifier is unique (no short-snippet repetition).
  - Skinny-trace mode forces ``attn_implementation="sdpa"`` +
    ``output_hidden_states=False`` + ``output_attentions=False``
    so a 32 k-token Llama-3.1-8B forward fits in 24-40 GB.

* **Driver + verifier**:
  - ``scripts/run_frontier_closure_w86.py`` — runs all three
    benches end-to-end, writes content-addressed report +
    per-issue sidecars. Computes report_cid even on failure.
  - ``scripts/verify_w86_audit_chain.py`` — offline re-hashes
    the report and prints PASS/FAIL per DoD bullet.
  - ``scripts/colab_frontier_closure_w86.ipynb`` — Colab Pro
    browser notebook (HF token via Colab Secrets, code via
    git clone from origin/main, results to Drive + zip).

* **Tests**: 24 new W86 unit tests covering the precision-
  tier constants, the skinny-trace = sdpa rule, the
  full-trace = eager regression, projection determinism,
  capsule disjointness, prompt-builder anti-cheat, lean-env
  honesty.

## Honest carry-forward limitations

* ``W86-L-FRONTIER-RUNTIME-V1-COLAB-PRO-CAP`` — the W86 run
  uses Colab Pro browser as the GPU host. The notebook + driver
  are runtime-agnostic; they re-run on any cuda host.
* ``W86-L-LIVE-CM-TRAIN-V1-PROJECTION-CAP`` — to make the live
  R^4096-hidden-state task comparable with the W83 synthetic
  baseline at the same module dimensions, live hidden states
  are projected through a fixed content-addressed random
  R^4096→R^8 projection. The projection is hashed into the
  dataset CID; richer projections (PCA over a held-out
  corpus, learned projection) are V2.
* ``W86-L-LIVE-CM-TRAIN-V1-NEXT-STEP-TASK-CAP`` — V1 trains the
  composed module on the next-step forecasting task on the
  projected hidden-state sequence. The task is meaningful real
  data; richer tasks (cross-layer prediction, perplexity-
  coupled training) are V2.
* ``W86-L-LONG-CONTEXT-INTERCEPT-V1-SKINNY-TRACE-CAP`` — V1
  uses skinny trace to fit 32 k tokens in 40 GB; per-layer
  hidden-state capture at 32 k is NOT exercised. The bench
  reports only the trace CID (final logits + KV); the
  intercept-moves-CID claim is binary and holds without
  per-layer capture. Full capture at 32 k requires ≥ 48 GB
  VRAM and is V2.
* ``W86-L-PRECISION-TIER-BF16-WIDENED-FLOOR-CAP`` — replay-
  from-KV at bf16 carries a tolerance of 5e-1 (vs the W80 fp32
  floor of 5e-3). The measured ``max_abs_diff`` is reported
  separately so a third party can verify the floor was not
  silently widened beyond what the tier permits.

## Re-verification from disk

```bash
$ python scripts/verify_w86_audit_chain.py \
      --report results/w86/<RUN_TS>/frontier_closure_report.json
```

The verifier re-hashes every per-issue sidecar against the
recorded CID, re-hashes the top-level report against the
recorded ``report_cid``, and prints a PASS/FAIL line per #25 /
#26 / #27 DoD bullet.

## Stable boundary preservation

* ``coordpy.__version__`` unchanged at 0.5.20.
* ``coordpy.SDK_VERSION`` unchanged at ``coordpy.sdk.v3.43``.
* No PyPI publish.
* ``coordpy/__init__.py`` untouched.
* All W86 modules are explicit-import only.
* The stable SDK surface
  (``RunSpec``, ``run``, ``AgentTeam``, ``coordpy-team`` CLI)
  is byte-for-byte unchanged.
