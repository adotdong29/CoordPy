# RESULTS — W86 P2 closures

> **Status (2026-05-20):** 7 of 8 P2 sub-issues of meta-#49
> TRULY CLOSED locally; #44 GPU/TPU substrate has its contract
> + bench infrastructure shipped (CPU CI green) with the
> empirical GPU evidence awaiting a single Colab Pro run.
>
> All eight modules are explicit-import only. No SDK version
> bump, no PyPI publish. `coordpy/__init__.py` untouched.

## Summary table

| Issue | Title (short)                           | Verdict     | Tests | Bench evidence                                          |
|------:|-----------------------------------------|-------------|------:|---------------------------------------------------------|
|  #38  | Byzantine Fault Tolerance (PBFT V1)     | **CLOSED**  |   22  | `results/w86/bft/<TS>/bft_v1_suite_report.json` (suite_cid `3ff0e1797c2b…`) |
|  #45  | Memory Garbage Collection (V1)          | **CLOSED**  |   16  | `results/w86/gc/<TS>/gc_v1_bench_report.json` (report_cid `c93a906690c4…`) |
|  #41  | Schema Evolution V1                     | **CLOSED**  |   17  | `results/w86/schema_evolution/<TS>/schema_evolution_v1_bench_report.json` |
|  #43  | Multi-Tenancy Isolation V1              | **CLOSED**  |   14  | `results/w86/tenancy/<TS>/multi_tenancy_v1_bench_report.json` |
|  #39  | Differential Privacy V1                 | **CLOSED**  |   18  | `results/w86/dp/<TS>/dp_v1_bench_report.json` (with proof in `papers/proofs/w86_proof_dp_v1.md`) |
|  #40  | MPC / Secret-Sharing V1                 | **CLOSED**  |   15  | `results/w86/mpc/<TS>/mpc_v1_bench_report.json` |
|  #42  | State Drift Across Model-Weight Updates | **CLOSED**  |   13  | `results/w86/drift/<TS>/drift_v1_bench_report.json` |
|  #44  | GPU/TPU Substrate Deterministic Replay  | **PARTIAL** |   11  | Contract + bench infrastructure + Colab notebook shipped; CPU CI green. Live GPU evidence awaits a single Colab Pro A100 run with `scripts/colab_gpu_deterministic_substrate_w86.ipynb`. |

Total: **126 new W86-P2 tests, all passing on Python 3.14 with NumPy + cryptography.**

## Per-issue DoD ↔ evidence

### #38 — Byzantine Fault Tolerance (PBFT V1)

| DoD bullet                                              | Evidence |
|---------------------------------------------------------|---------|
| `ByzantineWitnessV1` with crypto sigs over value        | `coordpy.byzantine_fault_tolerance_v1.ByzantineWitnessV1` + `tests/test_w86_byzantine_fault_tolerance_v1.py::test_byzantine_witness_signs_value_cid_and_verifies` |
| PBFT-style 3-phase protocol                             | `run_pbft_consensus_round_v1` — pre_prepare → prepare → commit, Ed25519-signed at every step |
| Collusion at f = ⌊(n−1)/3⌋ commits μ                    | n=7, f=2: `committed_value=1.0`, `committed_error=0.0` |
| f > (n−1)/3 refuses to commit                           | n=4, f=2: `verdict=refused_quorum_not_reached` |
| Equivocation evidence is independently verifiable       | `ByzantineEquivocationEvidenceV1.independently_verify` re-derives `conclusively_byzantine=True` |
| Safety + liveness proofs                                | `papers/proofs/w86_proof_byzantine_v1.md` (4 theorems: safety, equivocation detection, safety above bound, liveness) |
| `RESULTS_*_BYZANTINE_V1.md`                             | `docs/RESULTS_W86_BYZANTINE_V1.md` |

**Run:**
```
python3 scripts/run_w86_bft_v1_bench.py
python3 scripts/verify_w86_bft_v1_audit_chain.py --report results/w86/bft/<TS>/bft_v1_suite_report.json
```
Two consecutive runs at seed 86 038 produce byte-identical reports (`suite_cid = 3ff0e1797c2b7c7c…`).

### #45 — Memory Garbage Collection V1

| DoD bullet                                       | Evidence |
|--------------------------------------------------|---------|
| `GCPolicyV1` content-addressed                   | `coordpy.event_graph_garbage_collection_v1.GCPolicyV1.cid()` |
| `GCEventV1` emitted per pass                     | `GCEventV1` capsule with `purged_event_cids`, `policy_cid`, `gc_timestamp_ns` |
| Mark-and-sweep preserves load-bearing roots      | `mark_reachable_v1` walks parent_event_ids from declared roots + critical kinds + genesis |
| Grace period works                               | `restore_event_from_grace_v1` recovers soft-deleted events |
| 100k events, ≥80% memory reduction               | bench: `n_events_generated=100000`, `memory_reduction_fraction=0.9992` (99.92%) |
| Chain re-verifies end-to-end                     | `verify_chain_across_gc_v1.chain_verifies=True` after the 100k purge |
| Persistent-store sketch                          | `JSONLPersistentStoreV1` (JSON-Lines append-only) — sketch as DoD permits |
| `RESULTS_*_GC_V1.md`                             | This file |

**Run:**
```
python3 scripts/run_w86_gc_v1_bench.py
python3 scripts/verify_w86_gc_v1_audit_chain.py --report results/w86/gc/<TS>/gc_v1_bench_report.json
```
Two consecutive runs produce byte-identical reports (`report_cid = c93a906690c49421…`).

### #41 — Schema Evolution V1

| DoD bullet                                       | Evidence |
|--------------------------------------------------|---------|
| `SchemaRegistryV2` content-addressed             | `coordpy.schema_evolution_v1.SchemaRegistryV2` |
| `MigrationFnV1` for at least one schema pair     | `MigrationPlanV1` from `coordpy.migration_envelope.v1` → `coordpy.migration_envelope.v2` (renames `arrival_delay` (float seconds) → `arrival_delay_ns` (int_ns); adds `forwarded_from`) |
| Deterministic migration                          | bench: `deterministic_migration=True` (re-run produces byte-identical new payload CID) |
| In-flight upgrade, chain verifies                | bench: `chain_verifies_across_migration=True` via `MigrationEventV1` bridges |
| Deprecated-but-readable                          | bench: `deprecated_payload_readable=True`, `deprecation_warning_emitted=True` |
| `RESULTS_*_SCHEMA_EVOLUTION.md`                  | This file |

### #43 — Multi-Tenancy Isolation V1

| DoD bullet                                       | Evidence |
|--------------------------------------------------|---------|
| `TenantIdentityV1` content-addressed             | `coordpy.multi_tenancy_isolation_v1.TenantIdentityV1` |
| Per-tenant event graphs                          | physical: each tenant has its own `EventGraphV1` instance (NOT shared dict + tenant_id filter) |
| Cross-tenant queries rejected                    | bench: `cross_tenant_read_refused=True` with `CrossTenantAccessDeniedEventV1` audit |
| Per-tenant budgets                               | bench: A drains its $1 budget; B's `spent_cost_usd=0.0` untouched (`budget_isolation_holds=True`) |
| Per-tenant audit anchors                         | bench: Tenant A Merkle root ≠ Tenant B Merkle root (`audit_anchors_distinct=True`) |
| Tenant-token crypto binding                      | bench: A bad token signed with B's key but claiming A's identity is REFUSED (`token_swap_refused=True`) |
| No B byte in A's chain                           | bench: `no_b_bytes_in_a_chain=True` |
| `RESULTS_*_MULTI_TENANCY.md`                     | This file |

### #39 — Differential Privacy V1

| DoD bullet                                       | Evidence |
|--------------------------------------------------|---------|
| `DPCapsuleV1` Laplace / Gaussian                 | `DPMechanismParamsV1` with `noise_scale = sensitivity / ε` (Laplace) or `sensitivity · √(2 ln(1.25/δ)) / ε` (Gaussian) |
| `PIIRedactor` ≥ 5 PII patterns                   | bench: `pii_redaction_pattern_count=5` (email, ssn, credit_card_16, phone_us, ip_v4) + 6 redactions made |
| `DPBudgetTrackerV1` cumulative                   | bench: `budget_breach_refused=True` — exhausted budget refuses further calls |
| DP-aware composed pipeline                       | `run_dp_composed_pipeline_v1` emits both DP capsule CID + Merkle integrity anchor CID |
| Demonstrably private (DP + integrity proof)      | `papers/proofs/w86_proof_dp_v1.md` (4 theorems: Laplace ε-DP, basic composition, anchor preserves DP via post-processing theorem, redactor span-CID non-leaky) |
| Utility-vs-privacy curve                         | bench: 5 ε points × 1000 samples; `utility_curve_is_monotonic=True` |
| `RESULTS_*_DIFFERENTIAL_PRIVACY.md`              | This file |

### #40 — MPC / Secret-Sharing V1

| DoD bullet                                       | Evidence |
|--------------------------------------------------|---------|
| `SecretShareCapsuleV1` Shamir k-of-n             | `coordpy.mpc_secret_sharing_v1.split_secret_v1` over GF(p) with 521-bit Mersenne prime |
| `ThresholdReconstructorV1`                       | k-of-n recovers; <k raises `ValueError` (test: `test_shamir_below_threshold_does_not_recover`) |
| MPC-Average primitive                            | `run_mpc_average_v1` — no party learns others' values (only summed shares are published) |
| Pedersen commitment + Schnorr proof              | `make_schnorr_proof_v1` / `verify_schnorr_proof_v1`; forged commitment is REJECTED (`forged_share_rejected=True`) |
| Cross-org bench                                  | `run_cross_org_mpc_bench_v1`: 2 orgs × 3 parties each, threshold 4. `sum_matches=True`, `no_cleartext_secrets_crossed_orgs=True` |
| Drop-out k < n test                              | bench: `drop_out_test_works=True` (n-1 parties suffice with threshold k ≤ n-1) |
| `RESULTS_*_MPC_V1.md`                            | This file |

### #42 — State Drift Across Model-Weight Updates V1

| DoD bullet                                       | Evidence |
|--------------------------------------------------|---------|
| `backend_runtime_id` + model-weights CID         | `ModelWeightsCIDV1` aggregates every weight tensor's SHA-256 |
| `DriftDetectorV1`                                | `run_drift_detector_v1` replays a prompt corpus under both checkpoints; mean L2 of (h_new - h_old) |
| Fires when changed, not when unchanged           | bench: `drift_score_unchanged=0.0`, `drift_score_changed=0.218`, `threshold=0.015` |
| Re-training pipeline beats stale on hold-out     | bench: `stale_holdout_mse=2.98e-3`, `new_holdout_mse=3.19e-4` (9.3× strict improvement) |
| Stale-capsule invalidation + fallback to recompute | bench: `stale_verdict_marks_old_capsule_stale=True`, `fallback_recommendation_is_recompute_for_stale=True` |
| Principled threshold (NOT bench-tuned)           | `threshold = fp64_floor × safety_margin = 5e-3 × 3 = 1.5e-2`; documented in `DriftDetectorConfigV1` |
| `RESULTS_*_STATE_DRIFT_V1.md`                    | This file |

### #44 — GPU/TPU Substrate with Deterministic Replay (PARTIAL)

**What's shipped:**

| DoD bullet                                       | Status |
|--------------------------------------------------|--------|
| `transformers_runtime_v1` supports `device='cuda'` end-to-end | ✓ (already in W86 #25 closure) |
| Determinism wrapper ON by default                | ✓ `coordpy.gpu_deterministic_substrate_v1.apply_determinism_wrapper_v1` |
| Replay-from-KV byte-identity at GPU precision floor | infrastructure ✓; **live A100 numbers pending Colab run** |
| Hidden-state intercept moves CID on GPU          | infrastructure ✓; **live A100 numbers pending Colab run** |
| Negative test: deterministic-off breaks byte-identity | infrastructure ✓; **live A100 numbers pending Colab run** |
| Tensor-parallel readback                         | V1 pass-through contract shipped (`TensorParallelReadbackV1`); multi-GPU run is V2 stretch (explicitly permitted by issue scope) |

**To finalise closure:** open
[`scripts/colab_gpu_deterministic_substrate_w86.ipynb`](../scripts/colab_gpu_deterministic_substrate_w86.ipynb)
on Colab Pro A100 with the `hf_token` secret set, *Run all*. The notebook
takes ~6 min and produces a content-addressed JSON report
that the offline verifier
(`scripts/verify_w86_gpu_substrate_v1_audit_chain.py`)
re-derives end-to-end.

**Why not closed locally:** this host has no CUDA. The 11 CI
tests exercise the wrapper code paths + capsule shapes
(`tests/test_w86_gpu_deterministic_substrate_v1.py`); the
contract-check capsule reports `wrapper_active=False` on this
CPU host honestly, and `wrapper_active=True` on any CUDA
host after `apply_determinism_wrapper_v1()`. The Colab
notebook is the **only** missing step.

## Honest scope

All eight modules carry `W86-L-*-V1-*` carry-forward
limitation rows tracked in
`docs/THEOREM_REGISTRY.md` and
`docs/HOW_NOT_TO_OVERSTATE.md`. The key ones:

* `W86-L-BYZANTINE-V1-IN-PROCESS-CAP` — V1 PBFT runs in-process;
  wire-level (multi-host) is V2.
* `W86-L-BYZANTINE-V1-NO-VIEW-CHANGE-CAP` — V1 has no view-change;
  a Byzantine primary kills liveness, not safety.
* `W86-L-GC-V1-IN-MEMORY-FALLBACK-CAP` — persistent store is a
  JSONL sketch; LSM-tree / RocksDB is V2.
* `W86-L-DP-V1-BASIC-COMPOSITION-CAP` — V1 uses basic
  (sequential) composition; Rényi DP / advanced composition
  is V2.
* `W86-L-MPC-V1-AVERAGE-ONLY-CAP` — V1 MPC primitive is
  additive (sum / average); general MPC-multiply via garbled
  circuits is V2.
* `W86-L-MPC-V1-2-ORG-CAP` — V1 cross-org bench is 2 orgs; n-org
  is V2.
* `W86-L-DRIFT-V1-CONTROLLED-RUNTIME-CAP` — V1 exercises the
  drift mechanism on `controlled_runtime_substrate_v1` (pure
  NumPy fp64). The detector + threshold + re-training contract
  is runtime-agnostic; the same bench runs on a real fine-tuned
  HF model when one is available.
* `W86-L-TENANCY-V1-TWO-TENANTS-CAP` — bench is 2 tenants; n-tenant V2.
* `W86-L-GPU-V1-COLAB-PRO-CAP` — V1 closure run is on Colab Pro
  A100 / L4. Contract is hardware-agnostic.
* `W86-L-SCHEMA-EVOLUTION-V1-ONE-PAIR-CAP` — V1 ships one
  example migration (V1 → V2); full migration matrix V2.

## Reproducibility

Each closure ships:

* `scripts/run_w86_<closure>_v1_bench.py` — bench driver
* `scripts/verify_w86_<closure>_v1_audit_chain.py` — offline
  verifier (re-derives every CID, prints PASS/FAIL per DoD)
* `coordpy/<closure>_v1.py` — module (explicit-import only)
* `tests/test_w86_<closure>_v1.py` — CI tests
* `results/w86/<closure>/<TS>/<closure>_v1_bench_report.json` —
  canonical evidence

Run any closure twice at the default seed and the two reports
are byte-identical (driven by the seed → all RNG state →
determinism end-to-end).

## Stable boundary preservation

* `coordpy.__version__ == "0.5.20"` — byte-for-byte unchanged.
* `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"` — byte-for-byte
  unchanged.
* `coordpy/__init__.py` — byte-for-byte unchanged.
* No PyPI publish.
* All W86 P2 modules are explicit-import only.

## Files added

### Modules (under `coordpy/`)

* `byzantine_fault_tolerance_v1.py`
* `event_graph_garbage_collection_v1.py`
* `schema_evolution_v1.py`
* `multi_tenancy_isolation_v1.py`
* `differential_privacy_v1.py`
* `mpc_secret_sharing_v1.py`
* `state_drift_detection_v1.py`
* `gpu_deterministic_substrate_v1.py`

### Drivers + verifiers (under `scripts/`)

* `run_w86_bft_v1_bench.py` + `verify_w86_bft_v1_audit_chain.py`
* `run_w86_gc_v1_bench.py` + `verify_w86_gc_v1_audit_chain.py`
* `run_w86_schema_evolution_v1_bench.py` + `verify_w86_schema_evolution_v1_audit_chain.py`
* `run_w86_multi_tenancy_v1_bench.py` + `verify_w86_multi_tenancy_v1_audit_chain.py`
* `run_w86_dp_v1_bench.py` + `verify_w86_dp_v1_audit_chain.py`
* `run_w86_mpc_v1_bench.py` + `verify_w86_mpc_v1_audit_chain.py`
* `run_w86_drift_v1_bench.py` + `verify_w86_drift_v1_audit_chain.py`
* `run_w86_gpu_substrate_v1_bench.py` + `verify_w86_gpu_substrate_v1_audit_chain.py`
* `colab_gpu_deterministic_substrate_w86.ipynb` (Colab notebook for #44)

### Proofs (under `papers/proofs/`)

* `w86_proof_byzantine_v1.md` (Safety, Liveness, Equivocation,
  Safety-above-bound)
* `w86_proof_dp_v1.md` (Laplace ε-DP, Basic Composition,
  DP-Integrity Composition, Redaction Span-CID Non-Leaky)

### Tests (under `tests/`)

* `test_w86_byzantine_fault_tolerance_v1.py` — 22 tests
* `test_w86_event_graph_garbage_collection_v1.py` — 16 tests
* `test_w86_schema_evolution_v1.py` — 17 tests
* `test_w86_multi_tenancy_isolation_v1.py` — 14 tests
* `test_w86_differential_privacy_v1.py` — 18 tests
* `test_w86_mpc_secret_sharing_v1.py` — 15 tests
* `test_w86_state_drift_detection_v1.py` — 13 tests
* `test_w86_gpu_deterministic_substrate_v1.py` — 11 tests
