# Success criterion — W84 Post-W83 Blocker Audit & Tightening

> Pre-committed, falsifiable, code-backed bar for the W84
> milestone. Set on **2026-05-19** before the W84 line landed.
> Failing any of these bars retracts W84 to "partial".

W84 is a *strict honesty pass* over the post-W83 blocker
backlog (meta issue #49). W84 does NOT attempt to close any
issue that requires hardware that this host does not have.
It DOES:

1. Audit every P0 (#25–#29) and P1 (#30–#37) child against
   its own DoD + anti-cheat clauses.
2. Implement every P1 tightening achievable on a CPU + NumPy
   host without external resources.
3. Ship audit infrastructure (probes, contracts, adapter
   shapes) so hardware-blocked issues are configured for
   re-running on a GPU host without re-implementation.

## Milestone framing

* Carry forward all 19 W79 + 1 W83 (= 20) regimes unchanged.
* Carry forward every W80 / W81 / W82 / W83 module unchanged.
* Add eleven new W84 modules + four proofs. All explicit-
  import only.
* No new MASC regime. No change to MASC V15.

## Required honesty bars (load-bearing)

| Bar | Requirement | Where verified |
| --- | --- | --- |
| **Audit honesty** | Every P0 / P1 child has a precise audit verdict in `docs/AUDIT_POST_W83_BLOCKERS.md`. | `docs/AUDIT_POST_W83_BLOCKERS.md` |
| **Frontier probe honesty (#25)** | Probe reports `ready_for_frontier_bench=False` on this host; bench raises `FrontierBlockedOnHardwareError`; never mocks. | `coordpy.frontier_capability_probe_v1`, `tests/test_w84_audit_infrastructure.py::test_w84_frontier_probe_does_not_mock` |
| **Live-training honesty (#26)** | Builder enforces held-out disjointness; raises `LiveTrainingBlockedOnHardwareError` if transformers/torch absent; never synthetic. | `coordpy.live_hidden_state_dataset_v1` |
| **Long-context substrate-side (#27)** | Controlled-runtime substrate strictly beats bounded-V3 at 8k and 32k positions (V3 abstains past summary coverage). | `coordpy.long_context_substrate_bench_v1`, `tests/test_w84_audit_infrastructure.py::test_w84_long_context_bench_substrate_beats_v3_at_32k` |
| **Real-task adapter honesty (#28)** | `RealTaskBenchAdapterV1.run_harness` refuses without a model client; plan-only path emits a Merkle-rooted plan chain. | `coordpy.real_task_bench_adapter_v1` |
| **Cross-process distributed (#29)** | Two real OS subprocesses; mTLS-shaped HMAC mutual auth; partition drops 100% during window; ±5 s skew tolerated; idempotent apply across real network; sender/receiver post-roots match. | `coordpy.cross_process_distributed_substrate_v1`, `tests/test_w84_cross_process_distributed.py` |
| **Precision-tier contract (#30)** | Contract refuses widened-floor declarations; tier floors are content-addressed. | `coordpy.precision_tier_contract_v1`, `tests/test_w84_audit_infrastructure.py::test_w84_precision_tier_contract_refuses_widening` |
| **Streaming substrate (#32)** | Per-token streaming forward; streaming final-hidden CID equivalent at fp64 precision floor; mid-stream injection diverges post-step; streaming chain replayable; real SSE on the gateway. | `coordpy.streaming_substrate_intercept_v1`, `tests/test_w84_streaming_substrate.py` |
| **Tool substrate (#33)** | Identical inputs → identical CIDs; non-idempotent without token refused; sandbox enforces wall-time; 5-agent bench produces Merkle-rooted audit chain replayable from disk. | `coordpy.tool_call_substrate_v1`, `tests/test_w84_tool_call_substrate.py` |
| **Constrained policy (#34)** | Lagrangian strictly beats unconstrained REINFORCE on floor respect across 10 seeds; price of safety reported; constraints content-addressed; bootstrap CI on violation rate. | `coordpy.constrained_policy_optimisation_v1`, `tests/test_w84_constrained_policy_optimisation.py` |
| **Analytical bounds (#35)** | Four written proofs in `papers/proofs/`; each proof has an empirical sanity test that does NOT violate the bound. | `papers/proofs/*.md`, `tests/test_w84_analytical_bounds.py` |
| **Capacity scaling (#36)** | Three axes measured; cliff identified; remediation moves cliff ≥ 3× (honest report; not full OoM). | `coordpy.capacity_bench_harness_v1`, `tests/test_w84_capacity_bench.py::test_w84_speedup_factor_at_50k_q100_at_least_3x` |
| **Budget enforcement (#37)** | Over-budget regime → 0 commits + N abstains; under-budget → identical to no-enforcer; cost model monotone-in-tokens; disabled flag in spec CID. | `coordpy.budget_enforcement_v1`, `tests/test_w84_budget_enforcement.py` |
| **Stable-boundary preservation** | Every W84 module is explicit-import only; `coordpy.__version__` unchanged at 0.5.20; `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`; no PyPI release; `coordpy/__init__.py` untouched. | `coordpy/_version.py`, `coordpy/__init__.py` |

## Explicit non-claims

* W84 does NOT close any post-W83 P0 issue (#25–#29 literal
  bars). All five remain open in the meta issue.
* W84 does NOT close #31 MoE substrate (no Mixtral / Qwen-
  MoE weights, no GPU).
* W84 does NOT promote any limitation tag to retired.
* W84's cross-process distributed work is two subprocesses
  on one machine; literal multi-machine remains open
  (`W84-L-CROSS-PROCESS-DISTRIBUTED-V1-SAME-HOST-CAP`).
* W84's analytical bounds are math-readable, not formally
  verified; Lean / Coq is the separate P3 #48 issue.

## Strong, partial, failure

* **Strong success:** every honesty bar above passes.
* **Partial:** one or two bars fall short (e.g., the
  capacity-bench cliff move is 5× rather than the 10× the
  issue body asks for); the milestone ships with explicit
  caveats in the result note.
* **Failure:** any honesty bar is missed in a way that misses
  the spirit of "no cheating" (e.g., a frontier-mock slips
  in, or a synthetic dataset is silently labelled live).
  The milestone is retracted and the underlying module is
  fixed before re-landing.

## No version bump / no PyPI release

* ``coordpy.__version__`` stays at 0.5.20.
* ``coordpy.SDK_VERSION`` stays at ``coordpy.sdk.v3.43``.
* No PyPI publish.
* ``coordpy/__init__.py`` untouched.
* All W84 modules are explicit-import only.
* Stable SDK surface byte-for-byte unchanged.
