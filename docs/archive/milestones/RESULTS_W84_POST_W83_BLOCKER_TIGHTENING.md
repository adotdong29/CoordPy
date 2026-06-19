# W84 — Post-W83 Blocker Audit & In-Repo Tightening (result note)

> Post-W83 research milestone. Lands on `main` after the W80
> P0-blocker, W81 P1-blocker, W82 P2-blocker, and W83 composed
> frontier substrate / learned memory waves. Last touched:
> 2026-05-19.

## What W84 ships

W84 is the **strongest honest audit + tightening pass** on the
new post-W83 blocker backlog (meta issue #49). It does NOT
attempt to close the literal multi-machine, real-frontier-
model, or live-LLM-training bars — those remain blocked on
hardware this host does not have. Instead, it:

1. Performs a strict audit of every P0 (#25–#29) and P1
   (#30–#37) child issue against its own Definition of Done
   and *How NOT to close this* anti-cheat clauses
   (`docs/AUDIT_POST_W83_BLOCKERS.md`).
2. Implements every P1 tightening the host *can* do honestly
   (#29 partial, #32, #33, #34, #35, #36, #37 partial).
3. Ships audit infrastructure for the hardware-blocked
   P0 line (#25, #26, #27 partial, #28, #30) so a future GPU
   host runs the bench without re-implementing.

## Architecture split

W84 advances honestly partition into three planes:

* **In-repo tightening plane.** Modules that close meaningful
  protocol gaps without needing external resources:
  budget enforcement, tool substrate, constrained Lagrangian
  RL, streaming substrate intercept, capacity bench,
  cross-process distributed substrate.
* **Audit infrastructure plane.** Modules that codify the
  contract for hardware-blocked issues without mocking the
  underlying capability: frontier capability probe, live
  hidden-state dataset, real-task bench adapter (plan-only),
  precision-tier contract, long-context substrate bench.
* **Analytical theory plane.** Four written proofs of W81 /
  W82 / W83 load-bearing claims, plus empirical sanity
  checks.

## Mechanism advances (11)

| # | Module | Anchor | Issue | Verdict |
|---|--------|--------|-------|---------|
| M1  | ``budget_enforcement_v1``                          | `RunBudgetSpecV1` + `BudgetEnforcerV1` + content-addressed cost model + breach audits + stress bench | #37 | PARTIAL |
| M2  | ``tool_call_substrate_v1``                         | `ToolCallSchemaV1` + `ToolResultSchemaV1` + sandbox + idempotency contract + audit-replayable team bench | #33 | PARTIAL |
| M3  | ``constrained_policy_optimisation_v1``             | `LagrangianRefinementV1` + projection + violation log + 10-seed floor-recovery bench + price of safety | #34 | PARTIAL |
| M4  | ``streaming_substrate_intercept_v1``               | `forward_stream` on the controlled runtime + SSE endpoint + mid-stream injection + bench (equivalence/divergence/replay) | #32 | PARTIAL |
| M5  | ``capacity_bench_harness_v1``                      | three-axis scaling + indexed-query remediation + measured 5–7× cliff move | #36 | PARTIAL |
| M6  | ``cross_process_distributed_substrate_v1``         | two real OS subprocesses + mTLS-shaped HMAC + `PartitionProxyV1` + ±5 s skew + idempotent apply across real net | #29 | PARTIAL |
| M7  | ``frontier_capability_probe_v1``                   | honest hardware probe; raises `FrontierBlockedOnHardwareError` | #25 | INFRASTRUCTURE |
| M8  | ``live_hidden_state_dataset_v1``                   | `LiveHiddenStateDatasetCapsuleV1` + held-out disjointness; raises `LiveTrainingBlockedOnHardwareError` | #26 | INFRASTRUCTURE |
| M9  | ``long_context_substrate_bench_v1``                | needle-in-haystack corpus + controlled-runtime vs bounded-V3 at {2k, 8k, 32k} | #27 | PARTIAL (substrate side) |
| M10 | ``real_task_bench_adapter_v1``                     | `RealTaskBenchAdapterV1` + SWE-bench-Lite JSONL ingest + `RealTaskBenchPlanChainV1` Merkle root | #28 | INFRASTRUCTURE |
| M11 | ``precision_tier_contract_v1``                     | three first-class precision tiers with declared floors; refuses widened floors | #30 | INFRASTRUCTURE |

Four proofs land under `papers/proofs/`:

* `w84_proof_trust_weighted_consensus_error_bound.md`
* `w84_proof_integrity_drop_does_not_increase_error.md`
* `w84_proof_lhr_slot_capacity_bound.md`
* `w84_proof_replay_from_kv_exact.md`

## Audit verdicts (one-line summary)

| Issue | DoD floor | Verdict |
|------|-----------|---------|
| #25 P0 Frontier 7B+ | one 7B+ open-weight model under W80 contract | STILL OPEN — blocked on hardware. W84 ships capability probe + harness gate. |
| #26 P0 Live training | live-trained memory strictly beats synthetic on held-out live eval | STILL OPEN — blocked on hardware. W84 ships dataset builder + held-out contract; refuses to mock. |
| #27 P0 Long-context ≥ 32k live | composed pipeline beats V3 on live task success at 32k | STILL OPEN — live-LLM 32k blocked on hardware. W84 ships controlled-runtime 32k bench (substrate dominates V3). |
| #28 P0 Real-world bench | one bench adapter; head-to-head strict improvement; 3 seeds | STILL OPEN — blocked on a real model. W84 ships plan-only adapter + Merkle-rooted plan chain. |
| #29 P0 Real cross-host | ≥ 2 hosts; mTLS; partition; skew; idempotent apply | PARTIAL — W84 ships ≥ 2 OS subprocesses + mTLS-shaped HMAC + partition + ±5 s skew + idempotency over real network (loopback). Literal multi-machine still open. |
| #30 P1 Quantized | bf16 + int8 with honest floors; ≥ 95 % top-1 at int8 | STILL OPEN — int8 load blocked on bitsandbytes + CUDA. W84 ships precision-tier contract + capability probe. |
| #31 P1 MoE | MoE-routing axes + adapter + replay divergence without routing | STILL OPEN — blocked on Mixtral / Qwen-MoE weights + GPU. W84 does not advance. |
| #32 P1 Streaming | `forward_stream` + SSE + mid-stream injection + per-token chunks | PARTIAL — W84 ships all three on the controlled runtime. HF streaming + openai SDK integration test are V2. |
| #33 P1 Tool substrate | content-addressed call/result + sandbox + idempotency + 5-agent merged audit | PARTIAL — W84 ships the contract + sandbox + 5-agent bench. RAG-index state + DB transactions are V2. |
| #34 P1 Constrained policy | Lagrangian + analytical gradients + floor recovery + 10 seeds + price of safety | PARTIAL — W84 ships Lagrangian + projection + 10-seed bench (Lagrangian strictly beats REINFORCE on floor). TRPO / PPO-clip are V2. |
| #35 P1 Analytical bounds | ≥ 3 proofs + empirical sanity + soundness review | PARTIAL — W84 ships 4 proofs + empirical sanity tests. Lean / Coq formalisation is the separate #48 issue. |
| #36 P1 Capacity scaling | 3 axes + cliff + remediation moves cliff ≥ 1 OoM | PARTIAL — W84 ships 3-axis bench + indexed-query remediation; measured ~5–7× speedup at Q=100 N=50k. Full OoM remains future work. |
| #37 P1 Budget enforcement | `RunBudgetSpecV1` + pre-action enforcer + cost model + stress | PARTIAL — W84 ships all of those. Per-tenant + dynamic-pricing are V2. |

## Stable boundary preservation

* ``coordpy.__version__`` unchanged at **0.5.20**.
* ``coordpy.SDK_VERSION`` unchanged at
  ``coordpy.sdk.v3.43``.
* No PyPI publish.
* ``coordpy/__init__.py`` untouched.
* All W84 modules are explicit-import only.
* The stable SDK surface (``RunSpec``, ``run``, ``AgentTeam``,
  ``coordpy-team`` CLI) is byte-for-byte unchanged.
* W83, W82, W81, W80, W79 baselines remain green.

## Files added

Modules under ``coordpy/``:

* ``budget_enforcement_v1.py``
* ``tool_call_substrate_v1.py``
* ``constrained_policy_optimisation_v1.py``
* ``streaming_substrate_intercept_v1.py``
* ``capacity_bench_harness_v1.py``
* ``cross_process_distributed_substrate_v1.py``
* ``frontier_capability_probe_v1.py``
* ``live_hidden_state_dataset_v1.py``
* ``long_context_substrate_bench_v1.py``
* ``real_task_bench_adapter_v1.py``
* ``precision_tier_contract_v1.py``

Tests under ``tests/``:

* ``test_w84_budget_enforcement.py`` (12 tests)
* ``test_w84_tool_call_substrate.py`` (16 tests)
* ``test_w84_constrained_policy_optimisation.py`` (12 tests)
* ``test_w84_streaming_substrate.py`` (11 tests)
* ``test_w84_capacity_bench.py`` (8 tests)
* ``test_w84_cross_process_distributed.py`` (10 tests)
* ``test_w84_analytical_bounds.py`` (4 tests)
* ``test_w84_audit_infrastructure.py`` (15 tests)

**Total: +88 new W84 tests, all passing.**

Proofs under ``papers/proofs/``:

* ``w84_proof_trust_weighted_consensus_error_bound.md``
* ``w84_proof_integrity_drop_does_not_increase_error.md``
* ``w84_proof_lhr_slot_capacity_bound.md``
* ``w84_proof_replay_from_kv_exact.md``

Docs:

* ``docs/AUDIT_POST_W83_BLOCKERS.md``
* ``docs/RESULTS_W84_POST_W83_BLOCKER_TIGHTENING.md`` (this
  file)
* ``docs/SUCCESS_CRITERION_W84_POST_W83_BLOCKER_TIGHTENING.md``
* ``docs/RESEARCH_STATUS.md`` (amended with W84 block)
* ``docs/THEOREM_REGISTRY.md`` (amended with W84-T-/W84-L-
  block)
* ``docs/HOW_NOT_TO_OVERSTATE.md`` (amended with W84 caveats)
* ``docs/context_zero_master_plan.md`` (W84 milestone added)
* ``CHANGELOG.md`` (W84 entry)

## What W84 does NOT advance

* Does NOT close any post-W83 P0 issue (#25–#29 literal bars).
  The hardware-blocked items get audit infrastructure only.
* Does NOT solve context for hosted-model substrate; the
  hosted wall is unchanged.
* Does NOT close #31 MoE substrate (no Mixtral or Qwen-MoE
  weights on this host).
* Does NOT promote any limitation tag to retired — every
  `W84-L-*-CAP` documents what V1 explicitly does NOT cover.

## Cheating modes resisted

For full traceability, every anti-cheat clause from the meta
issue #49 children is honored:

* #25 — refuses to mock a frontier model; probe says
  "blocked-on-hardware" honestly.
* #26 — refuses to fall back to synthetic data when
  transformers / torch absent; raises
  `LiveTrainingBlockedOnHardwareError`.
* #27 — needle-in-haystack corpus places needle at unique
  position; no short-snippet repetition; substrate-side bar
  separated from live-LLM bar.
* #28 — adapter refuses to run harness without a real model
  client; refuses to substitute synthetic data.
* #29 — partition test passes (503 during window); mTLS-shaped
  HMAC is on by default; idempotency replay-N-times produces
  a stable digest.
* #30 — precision-tier contract refuses to claim byte-identity
  at sub-fp32 tiers; the floor must match the canonical per-
  tier value.
* #32 — streaming forward emits per-token chunks (never buffer-
  and-emit-once); mid-stream injection diverges the post-N
  CIDs from baseline; the equivalence floor is fp64 round-off,
  not a widened budget.
* #33 — non-idempotent calls without a token are refused;
  sandbox enforces wall-time (the V1 enforced limit).
* #34 — Lagrangian uses analytical gradients (no autodiff
  library); constraints are content-addressed in the policy
  CID; ≥ 10 seeds reported; price of safety reported.
* #35 — four written proofs ship; each proof has a stated
  assumptions section; empirical sanity tests do NOT violate
  the bounds.
* #36 — three axes measured + remediation patches; cliff
  measured at ~5–7× (NOT a full OoM — reported honestly).
* #37 — every refusal emits a `BudgetBreachAuditV1` (no silent
  drop); cost model is monotone-in-tokens by construction;
  the disabled flag is part of the spec CID so a third party
  can detect a disabled enforcer.
