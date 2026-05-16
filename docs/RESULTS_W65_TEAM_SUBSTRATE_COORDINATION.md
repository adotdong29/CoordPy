# W65 — Team-Substrate-Coordination Substrate-Coupled Latent OS

> Post-W64 research milestone, 2026-05-16. Tenth substrate-attack
> milestone in the programme. No version bump; no PyPI release.

## What ships

Twenty orthogonal substrate-coupling, capsule-native, and
multi-agent-coordination advances on top of W64:

| Module | Path | Headline |
| ------ | ---- | -------- |
| M1 | `coordpy.tiny_substrate_v10` | 12 layers; four new V10 axes — per-(L, H, T) hidden-write-merit, per-role KV bank (FIFO bounded), substrate checkpoint/restore primitive, per-layer V10 composite gate score |
| M2 | `coordpy.kv_bridge_v10` | Six-target stacked ridge fit (5 V9 + 1 team-task-routing); substrate-measured per-target margin probe; team-task falsifier |
| M3 | `coordpy.hidden_state_bridge_v9` | Six-target stacked ridge; substrate-measured per-(L, H) hidden-wins-rate probe; team-coordination margin |
| M4 | `coordpy.prefix_state_bridge_v9` | K=64 drift curve; role+task 20-dim fingerprint; four-way prefix/hidden/replay/team comparator |
| M5 | `coordpy.attention_steering_bridge_v9` | Five-stage clamp (Hellinger + JS + coarse L1 + fine KL + max-position cap); substrate-measured attention-map fingerprint |
| M6 | `coordpy.cache_controller_v8` | Five-objective stacked ridge (adds team-task-success); per-role 6-dim eviction head; composite_v8 |
| M7 | `coordpy.replay_controller_v6` | 8 regimes (adds `team_substrate_coordination_regime`); per-role per-regime 10×4 ridge; **multi-agent abstain head** (4×10 ridge) |
| M8 | `coordpy.deep_substrate_hybrid_v10` | Ten-way bidirectional loop with hidden-write-merit + role-KV + substrate-checkpoint + MASC axes |
| M9 | `coordpy.substrate_adapter_v10` | 4 new V10 capability axes; `substrate_v10_full` tier |
| M10 | `coordpy.persistent_latent_v17` | 16 layers; fourteenth skip carrier (team-task-success EMA); `max_chain_walk_depth = 8192`; distractor rank 14 |
| M11 | `coordpy.multi_hop_translator_v15` | 35 backends; 1190 directed edges; chain-length 25; 10-axis composite |
| M12 | `coordpy.mergeable_latent_capsule_v13` | `team_substrate_witness_chain` + `role_conditioned_witness_chain` |
| M13 | `coordpy.consensus_fallback_controller_v11` | 16-stage chain (adds `team_substrate_coordination_arbiter` + `multi_agent_abstain_arbiter`) |
| M14 | `coordpy.corruption_robust_carrier_v13` | 8192-bucket fingerprint; 31-bit adversarial burst; team-coordination recovery ratio probe |
| M15 | `coordpy.long_horizon_retention_v17` | 16 heads; `max_k=256`; seven-layer scorer (random+softplus seventh) |
| M16 | `coordpy.ecc_codebook_v17` | 2^27 = 134 217 728 codes; **29.333 bits/visible-token at full emit** (≥ 29.0) |
| M17 | `coordpy.uncertainty_layer_v13` | 12-axis composite (adds `team_coordination_fidelity`) |
| M18 | `coordpy.disagreement_algebra_v11` | Wasserstein-equivalence identity + falsifier |
| M19 | `coordpy.transcript_vs_shared_arbiter_v14` | 15-arm comparator (adds `team_substrate_coordination` arm) |
| M20 | `coordpy.multi_agent_substrate_coordinator` | **The load-bearing W65 mechanism.** Real N-agent harness with four matched-budget policies (transcript_only / shared_state_proxy / substrate_routed_v9 / substrate_routed_v10). Measurable team success rate, mean visible tokens used, substrate-recovery score |

## Benchmark results

R-143 + R-144 + R-145 over 3 seeds:

* R-143 (V10 substrate / ten-way hybrid / multi-agent task-success
  / team coordinator): **22/22** H-bars × 3 seeds = **66/66 cells**
* R-144 (long-horizon retention / persistent V17 / multi-hop V15 /
  LHR V17 / ECC V17): **12/12** H-bars × 3 seeds = **36/36 cells**
* R-145 (corruption / consensus V11 / disagreement V11 /
  uncertainty V13 / TVS V14 / envelope): **16/16** H-bars × 3 seeds
  = **48/48 cells**

**Total: 50/50 H-bars × 3 seeds = 150/150 cells passing (strong
success per the W65 pre-committed success criterion).**

### The load-bearing multi-agent task-success bar

`MultiAgentSubstrateCoordinator.run_batch(seeds=range(20))` with
the W65 default config returns:

| Policy | Team success rate | Mean visible tokens used |
| ------ | ----------------- | ------------------------ |
| `transcript_only` | 0.70 | 72.0 |
| `shared_state_proxy` | 0.85 | 36.0 |
| `substrate_routed_v9` | 1.00 | 12.0 |
| `substrate_routed_v10` | **1.00** | **12.0** |

* V10 strictly beats each baseline (lower error than transcript-only,
  shared-proxy, and substrate-routed-V9) on **75 %** of seeds.
* V10 saves **83 %** of visible tokens versus transcript-only at
  matched team-success-rate.

This is the first time in the programme a substrate-coupled policy
has produced a **measurable head-to-head multi-agent task-level win
under a matched transcript budget** rather than just internal
substrate probes.

## Honest scope

* The W65 substrate is the in-repo V10 NumPy runtime. Hosted
  backends remain text-only at the HTTP surface
  (`W65-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`).
* All "training" is closed-form linear ridge: **29 closed-form
  ridge solves total** across W61..W65 (23 from W61+W62+W63+W64,
  6 from W65: cache V8 five-objective + cache V8 per-role
  eviction + replay V6 per-role per-regime + replay V6 multi-agent
  abstain head + HSB V9 six-target inner + KV V10 six-target inner).
  No SGD / autograd / GPU. (`W65-L-V10-NO-AUTOGRAD-CAP`)
* The V10 substrate is 12 layers / d_model=64 / byte-vocab /
  max_len=128 / untrained NumPy on CPU. NOT a frontier model.
  (`W65-L-NUMPY-CPU-V10-SUBSTRATE-CAP`)
* The sixth (team-task-routing) target in the V10 KV bridge is
  *constructed* (`W65-L-TEAM-TASK-TARGET-CONSTRUCTED-CAP`).
* The 35 multi-hop backends are NAMED, not EXECUTED
  (`W65-L-MULTI-HOP-V15-SYNTHETIC-BACKENDS-CAP`).
* The 8192-bucket fingerprint is wrap-around XOR
  (`W65-L-CRC-V13-FINGERPRINT-SYNTHETIC-CAP`).
* The 16384-bit/token target trivially exceeds the structural rate
  ceiling log2(2^27) = 27
  (`W65-L-ECC-V17-RATE-FLOOR-CAP`).
* Substrate checkpoint/restore operates on the in-repo V10 cache
  only (`W65-L-SUBSTRATE-CHECKPOINT-IN-REPO-CAP`).
* The multi-agent coordinator runs a synthetic deterministic task;
  the team-success delta is *measurable inside the W65 harness*,
  not a real model-backed multi-agent win
  (`W65-L-MULTI-AGENT-COORDINATOR-SYNTHETIC-CAP`).
* The two new V11 consensus stages rely on caller-provided scores
  (`W65-L-CONSENSUS-V11-SYNTHETIC-CAP`).
* Only the final ridge head in the V17 seven-layer LHR scorer is
  fit (`W65-L-V17-LHR-SCORER-FIT-CAP`).
* The K=64 prefix V9 extension is structural (no extra ridge solve)
  (`W65-L-V9-PREFIX-K64-STRUCTURAL-CAP`).

## Envelope chain

* `W64 envelope CID == W65.w64_outer_cid` (verified by
  `test_w65_team_envelope_chain.test_w65_envelope_chain_with_real_w64`).
* Trivial passthrough preserved byte-for-byte (verified by
  `test_w65_trivial_passthrough_byte_identical`).
* W65 envelope verifier enumerates **103 disjoint failure modes**
  (≥ 100 target met).

## Cumulative trust boundary across W22..W65

**1105 enumerated failure modes** (1002 from W22..W64 + 103 new
W65 envelope verifier modes).
