"""W69 — Stronger Solving-Context Two-Plane Multi-Agent Substrate
team.

The ``W69Team`` orchestrator composes the W68 team with the W69
mechanism modules, organised across two planes plus the new
**Plane A↔B handoff coordinator**:

**Plane B — Real substrate (in-repo, V14 stack):**

* M1  ``tiny_substrate_v14``           (16-layer, 4 new V14 axes)
* M2  ``kv_bridge_v14``                (10-target ridge + multi-
                                        branch-rejoin margin + 60-
                                        dim silent-corruption
                                        fingerprint + falsifier)
* M3  ``hidden_state_bridge_v13``      (10-target ridge + per-(L, H)
                                        hidden-vs-multi-branch-
                                        rejoin probe)
* M4  ``prefix_state_bridge_v13``      (K=256 drift curve + 60-dim
                                        fingerprint + 8-way
                                        comparator)
* M5  ``attention_steering_bridge_v13``(9-stage clamp + multi-
                                        branch-conditioned
                                        fingerprint)
* M6  ``cache_controller_v12``         (9-objective ridge + per-role
                                        10-dim silent-corruption)
* M7  ``replay_controller_v10``        (16 regimes + per-role +
                                        multi-branch-rejoin-routing)
* M8  ``deep_substrate_hybrid_v14``    (14-way bidirectional loop)
* M9  ``substrate_adapter_v14``        (substrate_v14_full tier)
* M10 ``persistent_latent_v21``        (20 layers, 18th carrier,
                                        max_chain_walk_depth=65536)
* M11 ``multi_hop_translator_v19``     (48 backends, chain-len 38,
                                        14-axis composite)
* M12 ``mergeable_latent_capsule_v17`` (multi-branch-rejoin +
                                        silent-corruption chains)
* M13 ``consensus_fallback_controller_v15`` (24-stage chain)
* M14 ``corruption_robust_carrier_v17``(131072-bucket, 37-bit burst)
* M15 ``long_horizon_retention_v21``   (20 heads, max_k=512)
* M16 ``ecc_codebook_v21``             (2^35 codes, ≥ 37.0 b/v)
* M17 ``uncertainty_layer_v17``        (16-axis composite)
* M18 ``disagreement_algebra_v15``     (multi-branch-rejoin-equiv ID)
* M19 ``transcript_vs_shared_arbiter_v18`` (19-arm comparator)
* M20 ``multi_agent_substrate_coordinator_v5`` (12-policy, 9-regime
                                                MASC V5)
* M21 ``team_consensus_controller_v4``       (multi-branch-rejoin
                                              + silent-corruption
                                              arbiters)

**Plane A — Hosted control plane V2 (honest, no substrate):**

* H1  ``hosted_router_controller_v2``  (weighted-score + sticky +
                                        blacklist + cost-per-success)
* H2  ``hosted_logprob_router_v2``     (Bayesian Dirichlet fusion +
                                        per-provider trust)
* H3  ``hosted_cache_aware_planner_v2``(per-role staggered prefix +
                                        ≥ 60 % savings 6×8 hit=1)
* H4  ``hosted_provider_filter_v2``    (compositional ALL/ANY +
                                        tier weighting)
* H5  ``hosted_cost_planner_v2``       (multi-turn schedule +
                                        cost-per-success ratio)
* H6  ``hosted_real_substrate_boundary_v2`` (the wall V2, 19
                                             blocked axes)
* H7  ``hosted_real_handoff_coordinator`` (the **new Plane A↔B
                                            bridge** — handoff
                                            envelopes + falsifier
                                            + cross-plane savings)

Per-turn it emits 28 module witness CIDs (22 Plane B + 6 Plane A
V2) and a handoff envelope CID, sealing them into a
``W69HandoffEnvelope`` whose ``w68_outer_cid`` carries forward the
W68 envelope byte-for-byte.

Honest scope (W69)
------------------

* Plane A V2 operates at the hosted text/logprob/prefix-cache
  surface. It does NOT pierce hidden state / KV / attention.
  ``W69-L-HOSTED-V2-NO-SUBSTRATE-CAP``.
* Plane B is the in-repo V14 NumPy runtime. We do NOT bridge to
  third-party hosted models.
  ``W69-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
* W69 fits closed-form ridge parameters in six new places on top
  of W68's 47: cache V12 nine-objective; cache V12 per-role
  silent-corruption; replay V10 per-role per-regime; replay V10
  multi-branch-rejoin-routing; HSB V13 ten-target; KV V14 ten-
  target. Total **fifty-three closed-form ridge solves** across
  W61..W69. No autograd, no SGD, no GPU.
* Trivial passthrough preserved: when ``W69Params.build_trivial()``
  is used the W69 envelope's internal ``w68_outer_cid`` carries
  the supplied W68 outer CID exactly.
* The handoff coordinator preserves the wall: a content-addressed
  handoff envelope says which plane handled each turn; it does NOT
  cross the substrate boundary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .cache_controller_v12 import (
    CacheControllerV12,
    emit_cache_controller_v12_witness,
    fit_nine_objective_ridge_v12,
    fit_per_role_silent_corruption_head_v12,
)
from .consensus_fallback_controller_v15 import (
    ConsensusFallbackControllerV15,
    W69_CONSENSUS_V15_STAGES,
    emit_consensus_v15_witness,
)
from .corruption_robust_carrier_v17 import (
    CorruptionRobustCarrierV17,
    emit_corruption_robustness_v17_witness,
)
from .deep_substrate_hybrid_v13 import (
    DeepSubstrateHybridV13ForwardWitness,
)
from .deep_substrate_hybrid_v14 import (
    DeepSubstrateHybridV14,
    deep_substrate_hybrid_v14_forward,
)
from .ecc_codebook_v21 import (
    ECCCodebookV21,
    emit_ecc_v21_compression_witness,
    probe_ecc_v21_rate_floor_falsifier,
)
from .hidden_state_bridge_v13 import (
    HiddenStateBridgeV13Projection,
    compute_hsb_v13_multi_branch_rejoin_margin,
    emit_hsb_v13_witness,
)
from .hosted_cache_aware_planner_v2 import (
    HostedCacheAwarePlannerV2,
    emit_hosted_cache_aware_planner_v2_witness,
)
from .hosted_cost_planner_v2 import (
    HostedCostPlanSpecV2, plan_hosted_cost_v2,
)
from .hosted_cost_planner import HostedCostPlanSpec
from .hosted_logprob_router_v2 import (
    HostedLogprobRouterV2,
    emit_hosted_logprob_router_v2_witness,
)
from .hosted_provider_filter_v2 import (
    HostedProviderFilterSpecV2, filter_hosted_registry_v2,
)
from .hosted_provider_filter import HostedProviderFilterSpec
from .hosted_real_substrate_boundary_v2 import (
    HostedRealSubstrateBoundaryV2,
    build_default_hosted_real_substrate_boundary_v2,
    build_wall_report_v2,
    probe_hosted_real_substrate_boundary_v2_falsifier,
)
from .hosted_real_handoff_coordinator import (
    HostedRealHandoffCoordinator, HandoffRequest,
    emit_hosted_real_handoff_coordinator_witness,
    hosted_real_handoff_cross_plane_savings,
)
from .hosted_router_controller import (
    HostedProviderRegistry, HostedRoutingRequest,
    default_hosted_registry,
)
from .hosted_router_controller_v2 import (
    HostedRouterControllerV2, HostedRoutingRequestV2,
    emit_hosted_router_controller_v2_witness,
)
from .kv_bridge_v14 import (
    KVBridgeV14Projection,
    compute_silent_corruption_fingerprint_v14,
    emit_kv_bridge_v14_witness,
    probe_kv_bridge_v14_multi_branch_rejoin_falsifier,
)
from .long_horizon_retention_v21 import (
    LongHorizonReconstructionV21Head,
    emit_lhr_v21_witness,
)
from .mergeable_latent_capsule_v17 import (
    MergeOperatorV17, emit_mlsc_v17_witness, wrap_v16_as_v17,
)
from .multi_agent_substrate_coordinator_v5 import (
    MultiAgentSubstrateCoordinatorV5,
    W69_MASC_V5_REGIMES,
    emit_multi_agent_substrate_coordinator_v5_witness,
)
from .multi_hop_translator_v19 import (
    emit_multi_hop_v19_witness,
)
from .persistent_latent_v21 import (
    PersistentLatentStateV21Chain,
    emit_persistent_v21_witness,
)
from .prefix_state_bridge_v13 import (
    W69_DEFAULT_PREFIX_V13_K_STEPS,
    emit_prefix_state_v13_witness,
)
from .replay_controller_v10 import (
    ReplayControllerV10,
    W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS,
    W69_REPLAY_REGIMES_V10,
    fit_replay_controller_v10_per_role,
    fit_replay_v10_multi_branch_rejoin_routing_head,
    emit_replay_controller_v10_witness,
)
from .replay_controller import ReplayCandidate
from .substrate_adapter_v14 import (
    probe_all_v14_adapters,
)
from .team_consensus_controller_v4 import (
    TeamConsensusControllerV4,
    emit_team_consensus_controller_v4_witness,
)
from .tiny_substrate_v14 import (
    TinyV14SubstrateParams,
    build_default_tiny_substrate_v14,
    emit_tiny_substrate_v14_forward_witness,
    forward_tiny_substrate_v14,
    record_multi_branch_rejoin_witness_v14,
    repair_silent_corruption_v14,
    tokenize_bytes_v14,
    trigger_silent_corruption_v14,
)
from .transcript_vs_shared_arbiter_v18 import (
    emit_tvs_arbiter_v18_witness, nineteen_arm_compare,
)
from .uncertainty_layer_v17 import (
    compose_uncertainty_report_v17,
    emit_uncertainty_v17_witness,
)


W69_SCHEMA_VERSION: str = "coordpy.w69_team.v1"

W69_FAILURE_MODES: tuple[str, ...] = (
    "w69_outer_envelope_schema_mismatch",
    "w69_outer_envelope_w68_outer_cid_drift",
    "w69_outer_envelope_w69_params_cid_drift",
    "w69_outer_envelope_witness_cid_drift",
    "w69_substrate_v14_n_layers_off",
    "w69_substrate_v14_gate_score_shape_off",
    "w69_multi_branch_rejoin_witness_shape_off",
    "w69_silent_corruption_witness_off",
    "w69_substrate_self_checksum_off",
    "w69_kv_bridge_v14_n_targets_off",
    "w69_kv_bridge_v14_multi_branch_rejoin_falsifier_off",
    "w69_hsb_v13_n_targets_off",
    "w69_prefix_v13_k_steps_off",
    "w69_attention_v13_nine_stage_off",
    "w69_cache_v12_nine_objective_off",
    "w69_replay_v10_regime_count_off",
    "w69_replay_v10_multi_branch_rejoin_routing_count_off",
    "w69_consensus_v15_stage_count_off",
    "w69_crc_v17_fingerprint_buckets_off",
    "w69_crc_v17_adversarial_burst_bits_off",
    "w69_lhr_v21_max_k_off",
    "w69_lhr_v21_n_heads_off",
    "w69_ecc_v21_total_codes_off",
    "w69_ecc_v21_bits_per_token_off",
    "w69_tvs_v18_n_arms_off",
    "w69_uncertainty_v17_n_axes_off",
    "w69_mlsc_v17_algebra_signatures_off",
    "w69_disagreement_v15_mbr_falsifier_off",
    "w69_mh_v19_n_backends_off",
    "w69_persistent_v21_n_layers_off",
    "w69_substrate_adapter_v14_tier_off",
    "w69_masc_v5_v14_beats_v13_rate_under_threshold",
    "w69_masc_v5_tsc_v14_beats_tsc_v13_rate_under_threshold",
    "w69_masc_v5_multi_branch_rejoin_regime_inferior_to_baseline",
    "w69_masc_v5_silent_corruption_regime_inferior_to_baseline",
    "w69_hosted_router_v2_decision_not_deterministic",
    "w69_hosted_logprob_v2_fusion_kind_off",
    "w69_hosted_cache_aware_v2_savings_below_60_percent",
    "w69_hosted_provider_filter_v2_drop_count_off",
    "w69_hosted_cost_planner_v2_no_eligible",
    "w69_hosted_real_substrate_boundary_v2_blocked_axis_satisfied",
    "w69_fourteen_way_loop_not_observed",
    "w69_handoff_coordinator_inconsistent",
    "w69_handoff_cross_plane_savings_below_40_percent",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class W69Params:
    substrate_v14: TinyV14SubstrateParams | None
    kv_bridge_v14: KVBridgeV14Projection | None
    hidden_state_bridge_v13: HiddenStateBridgeV13Projection | None
    cache_controller_v12: CacheControllerV12 | None
    replay_controller_v10: ReplayControllerV10 | None
    consensus_v15: ConsensusFallbackControllerV15 | None
    crc_v17: CorruptionRobustCarrierV17 | None
    lhr_v21: LongHorizonReconstructionV21Head | None
    ecc_v21: ECCCodebookV21 | None
    deep_substrate_hybrid_v14: DeepSubstrateHybridV14 | None
    mlsc_v17_operator: MergeOperatorV17 | None
    multi_agent_coordinator_v5: (
        MultiAgentSubstrateCoordinatorV5 | None)
    team_consensus_controller_v4: (
        TeamConsensusControllerV4 | None)
    hosted_registry: HostedProviderRegistry | None
    hosted_router_v2: HostedRouterControllerV2 | None
    hosted_logprob_router_v2: HostedLogprobRouterV2 | None
    hosted_cache_planner_v2: HostedCacheAwarePlannerV2 | None
    hosted_real_substrate_boundary_v2: (
        HostedRealSubstrateBoundaryV2 | None)
    handoff_coordinator: HostedRealHandoffCoordinator | None
    prefix_v13_predictor_trained: bool
    enabled: bool = True
    masc_v5_n_seeds: int = 12

    @classmethod
    def build_trivial(cls) -> "W69Params":
        return cls(
            substrate_v14=None,
            kv_bridge_v14=None,
            hidden_state_bridge_v13=None,
            cache_controller_v12=None,
            replay_controller_v10=None,
            consensus_v15=None,
            crc_v17=None, lhr_v21=None, ecc_v21=None,
            deep_substrate_hybrid_v14=None,
            mlsc_v17_operator=None,
            multi_agent_coordinator_v5=None,
            team_consensus_controller_v4=None,
            hosted_registry=None,
            hosted_router_v2=None,
            hosted_logprob_router_v2=None,
            hosted_cache_planner_v2=None,
            hosted_real_substrate_boundary_v2=None,
            handoff_coordinator=None,
            prefix_v13_predictor_trained=False,
            enabled=False,
        )

    @classmethod
    def build_default(
            cls, *, seed: int = 69000,
    ) -> "W69Params":
        sub_v14 = build_default_tiny_substrate_v14(
            seed=int(seed) + 1)
        # KV V14 projection chain.
        from .kv_bridge_v3 import KVBridgeV3Projection
        from .kv_bridge_v4 import KVBridgeV4Projection
        from .kv_bridge_v5 import KVBridgeV5Projection
        from .kv_bridge_v6 import KVBridgeV6Projection
        from .kv_bridge_v7 import KVBridgeV7Projection
        from .kv_bridge_v8 import KVBridgeV8Projection
        from .kv_bridge_v9 import KVBridgeV9Projection
        from .kv_bridge_v10 import KVBridgeV10Projection
        from .kv_bridge_v11 import KVBridgeV11Projection
        from .kv_bridge_v12 import KVBridgeV12Projection
        from .kv_bridge_v13 import KVBridgeV13Projection
        cfg = sub_v14.config.v13.v12.v11.v10.v9
        d_head = int(cfg.d_model) // int(cfg.n_heads)
        kv_b3 = KVBridgeV3Projection.init(
            n_layers=int(cfg.n_layers),
            n_heads=int(cfg.n_heads),
            n_kv_heads=int(cfg.n_kv_heads),
            n_inject_tokens=3, carrier_dim=6,
            d_head=int(d_head), seed=int(seed) + 7)
        kv_b13 = KVBridgeV13Projection.init_from_v12(
            KVBridgeV12Projection.init_from_v11(
                KVBridgeV11Projection.init_from_v10(
                    KVBridgeV10Projection.init_from_v9(
                        KVBridgeV9Projection.init_from_v8(
                            KVBridgeV8Projection.init_from_v7(
                                KVBridgeV7Projection.init_from_v6(
                                    KVBridgeV6Projection.init_from_v5(
                                        KVBridgeV5Projection.init_from_v4(
                                            KVBridgeV4Projection.init_from_v3(
                                                kv_b3,
                                                seed_v4=int(seed) + 8),
                                            seed_v5=int(seed) + 9),
                                        seed_v6=int(seed) + 10),
                                    seed_v7=int(seed) + 11),
                                seed_v8=int(seed) + 12),
                            seed_v9=int(seed) + 13),
                        seed_v10=int(seed) + 14),
                    seed_v11=int(seed) + 15),
                seed_v12=int(seed) + 16),
            seed_v13=int(seed) + 17)
        kv_b14 = KVBridgeV14Projection.init_from_v13(
            kv_b13, seed_v14=int(seed) + 18)
        from .hidden_state_bridge_v2 import (
            HiddenStateBridgeV2Projection)
        from .hidden_state_bridge_v3 import (
            HiddenStateBridgeV3Projection)
        from .hidden_state_bridge_v4 import (
            HiddenStateBridgeV4Projection)
        from .hidden_state_bridge_v5 import (
            HiddenStateBridgeV5Projection)
        from .hidden_state_bridge_v6 import (
            HiddenStateBridgeV6Projection)
        from .hidden_state_bridge_v7 import (
            HiddenStateBridgeV7Projection)
        from .hidden_state_bridge_v8 import (
            HiddenStateBridgeV8Projection)
        from .hidden_state_bridge_v9 import (
            HiddenStateBridgeV9Projection)
        from .hidden_state_bridge_v10 import (
            HiddenStateBridgeV10Projection)
        from .hidden_state_bridge_v11 import (
            HiddenStateBridgeV11Projection)
        from .hidden_state_bridge_v12 import (
            HiddenStateBridgeV12Projection)
        hsb12 = HiddenStateBridgeV12Projection.init_from_v11(
            HiddenStateBridgeV11Projection.init_from_v10(
                HiddenStateBridgeV10Projection.init_from_v9(
                    HiddenStateBridgeV9Projection.init_from_v8(
                        HiddenStateBridgeV8Projection.init_from_v7(
                            HiddenStateBridgeV7Projection.init_from_v6(
                                HiddenStateBridgeV6Projection.init_from_v5(
                                    HiddenStateBridgeV5Projection.init_from_v4(
                                        HiddenStateBridgeV4Projection.init_from_v3(
                                            HiddenStateBridgeV3Projection.init_from_v2(
                                                HiddenStateBridgeV2Projection.init(
                                                    target_layers=(1, 3),
                                                    n_tokens=6, carrier_dim=6,
                                                    d_model=int(cfg.d_model),
                                                    seed=int(seed) + 20),
                                                n_heads=int(cfg.n_heads),
                                                seed_v3=int(seed) + 21),
                                            seed_v4=int(seed) + 22),
                                        n_positions=3,
                                        seed_v5=int(seed) + 23),
                                    seed_v6=int(seed) + 24),
                                seed_v7=int(seed) + 25),
                            seed_v8=int(seed) + 26),
                        seed_v9=int(seed) + 27),
                    seed_v10=int(seed) + 28),
                seed_v11=int(seed) + 29),
            seed_v12=int(seed) + 30)
        hsb13 = HiddenStateBridgeV13Projection.init_from_v12(
            hsb12, seed_v13=int(seed) + 31)
        cc12 = CacheControllerV12.init(fit_seed=int(seed) + 32)
        # Fit nine-objective ridge with a small synthetic dataset.
        import numpy as _np
        rng = _np.random.default_rng(int(seed) + 33)
        X = rng.standard_normal((10, 4))
        cc12, _ = fit_nine_objective_ridge_v12(
            controller=cc12, train_features=X.tolist(),
            target_drop_oracle=X.sum(axis=-1).tolist(),
            target_retrieval_relevance=X[:, 0].tolist(),
            target_hidden_wins=(X[:, 1] - X[:, 2]).tolist(),
            target_replay_dominance=(X[:, 3] * 0.5).tolist(),
            target_team_task_success=(
                X[:, 0] * 0.3 - X[:, 1] * 0.1).tolist(),
            target_team_failure_recovery=(
                X[:, 2] * 0.4 + X[:, 3] * 0.2).tolist(),
            target_branch_merge=(
                X[:, 0] * 0.2 + X[:, 2] * 0.5).tolist(),
            target_partial_contradiction=(
                X[:, 1] * 0.3 + X[:, 3] * 0.4).tolist(),
            target_multi_branch_rejoin=(
                X[:, 0] * 0.5 + X[:, 1] * 0.2).tolist())
        X10 = rng.standard_normal((8, 10))
        cc12, _ = fit_per_role_silent_corruption_head_v12(
            controller=cc12, role="planner",
            train_features=X10.tolist(),
            target_silent_corruption_priorities=(
                X10[:, 0] * 0.4 + X10[:, 9] * 0.3).tolist())
        # Replay V10.
        rcv10 = ReplayControllerV10.init()
        v10_cands = {
            r: [ReplayCandidate(
                100, 1000, 50, 0.1, 0.0, 0.3,
                True, True, 0)]
            for r in W69_REPLAY_REGIMES_V10}
        v10_decs = {
            r: ["choose_reuse"]
            for r in W69_REPLAY_REGIMES_V10}
        rcv10, _ = fit_replay_controller_v10_per_role(
            controller=rcv10, role="planner",
            train_candidates_per_regime=v10_cands,
            train_decisions_per_regime=v10_decs)
        X_team = rng.standard_normal((40, 10))
        labs: list[str] = []
        for i in range(40):
            if X_team[i, 0] > 0.5:
                labs.append(
                    W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS[0])
            elif X_team[i, 1] > 0.0:
                labs.append(
                    W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS[1])
            elif X_team[i, 2] > 0.0:
                labs.append(
                    W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS[2])
            elif X_team[i, 3] > 0.0:
                labs.append(
                    W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS[3])
            elif X_team[i, 4] > 0.0:
                labs.append(
                    W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS[4])
            elif X_team[i, 5] > 0.0:
                labs.append(
                    W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS[5])
            else:
                labs.append(
                    W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS[6])
        rcv10, _ = (
            fit_replay_v10_multi_branch_rejoin_routing_head(
                controller=rcv10,
                train_team_features=X_team.tolist(),
                train_routing_labels=labs))
        consensus_v15 = ConsensusFallbackControllerV15.init(
            k_required=2, cosine_floor=0.6,
            trust_threshold=0.5)
        crc_v17 = CorruptionRobustCarrierV17()
        lhr_v21 = LongHorizonReconstructionV21Head.init(
            seed=int(seed) + 34)
        ecc_v21 = ECCCodebookV21.init(seed=int(seed) + 35)
        hybrid_v14 = DeepSubstrateHybridV14(inner_v13=None)
        mlsc_v17_op = MergeOperatorV17(factor_dim=6)
        masc_v5 = MultiAgentSubstrateCoordinatorV5()
        tcc_v4 = TeamConsensusControllerV4()
        # Plane A V2.
        hosted_registry = default_hosted_registry()
        hosted_router_v2 = HostedRouterControllerV2.init(
            hosted_registry,
            success_score_per_provider={
                "openrouter_paid": 0.85,
                "openai_paid": 0.92,
                "groq_free": 0.78,
                "openrouter_free": 0.70,
                "ollama_local": 0.65,
            })
        hosted_logprob_router_v2 = HostedLogprobRouterV2()
        hosted_cache_planner_v2 = HostedCacheAwarePlannerV2()
        boundary_v2 = (
            build_default_hosted_real_substrate_boundary_v2())
        handoff = HostedRealHandoffCoordinator()
        return cls(
            substrate_v14=sub_v14,
            kv_bridge_v14=kv_b14,
            hidden_state_bridge_v13=hsb13,
            cache_controller_v12=cc12,
            replay_controller_v10=rcv10,
            consensus_v15=consensus_v15, crc_v17=crc_v17,
            lhr_v21=lhr_v21, ecc_v21=ecc_v21,
            deep_substrate_hybrid_v14=hybrid_v14,
            mlsc_v17_operator=mlsc_v17_op,
            multi_agent_coordinator_v5=masc_v5,
            team_consensus_controller_v4=tcc_v4,
            hosted_registry=hosted_registry,
            hosted_router_v2=hosted_router_v2,
            hosted_logprob_router_v2=hosted_logprob_router_v2,
            hosted_cache_planner_v2=hosted_cache_planner_v2,
            hosted_real_substrate_boundary_v2=boundary_v2,
            handoff_coordinator=handoff,
            prefix_v13_predictor_trained=True,
            enabled=True,
        )

    def to_dict(self) -> dict[str, Any]:
        def _cid_or_empty(x: Any) -> str:
            return x.cid() if x is not None else ""
        return {
            "schema_version": W69_SCHEMA_VERSION,
            "enabled": bool(self.enabled),
            "substrate_v14_cid": _cid_or_empty(self.substrate_v14),
            "kv_bridge_v14_cid": _cid_or_empty(self.kv_bridge_v14),
            "hidden_state_bridge_v13_cid": _cid_or_empty(
                self.hidden_state_bridge_v13),
            "cache_controller_v12_cid": _cid_or_empty(
                self.cache_controller_v12),
            "replay_controller_v10_cid": _cid_or_empty(
                self.replay_controller_v10),
            "consensus_v15_cid": _cid_or_empty(self.consensus_v15),
            "crc_v17_cid": _cid_or_empty(self.crc_v17),
            "lhr_v21_cid": _cid_or_empty(self.lhr_v21),
            "ecc_v21_cid": _cid_or_empty(self.ecc_v21),
            "deep_substrate_hybrid_v14_cid": _cid_or_empty(
                self.deep_substrate_hybrid_v14),
            "mlsc_v17_operator_cid": _cid_or_empty(
                self.mlsc_v17_operator),
            "multi_agent_coordinator_v5_cid": _cid_or_empty(
                self.multi_agent_coordinator_v5),
            "team_consensus_controller_v4_cid": _cid_or_empty(
                self.team_consensus_controller_v4),
            "hosted_registry_cid": _cid_or_empty(
                self.hosted_registry),
            "hosted_router_v2_cid": _cid_or_empty(
                self.hosted_router_v2),
            "hosted_logprob_router_v2_cid": _cid_or_empty(
                self.hosted_logprob_router_v2),
            "hosted_cache_planner_v2_cid": _cid_or_empty(
                self.hosted_cache_planner_v2),
            "hosted_real_substrate_boundary_v2_cid": _cid_or_empty(
                self.hosted_real_substrate_boundary_v2),
            "handoff_coordinator_cid": _cid_or_empty(
                self.handoff_coordinator),
            "prefix_v13_predictor_trained": bool(
                self.prefix_v13_predictor_trained),
            "masc_v5_n_seeds": int(self.masc_v5_n_seeds),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w69_params",
            "params": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class W69HandoffEnvelope:
    schema: str
    w68_outer_cid: str
    w69_params_cid: str
    substrate_v14_witness_cid: str
    kv_bridge_v14_witness_cid: str
    hsb_v13_witness_cid: str
    prefix_state_v13_witness_cid: str
    cache_controller_v12_witness_cid: str
    replay_controller_v10_witness_cid: str
    persistent_v21_witness_cid: str
    multi_hop_v19_witness_cid: str
    mlsc_v17_witness_cid: str
    consensus_v15_witness_cid: str
    crc_v17_witness_cid: str
    lhr_v21_witness_cid: str
    ecc_v21_witness_cid: str
    tvs_v18_witness_cid: str
    uncertainty_v17_witness_cid: str
    deep_substrate_hybrid_v14_witness_cid: str
    substrate_adapter_v14_matrix_cid: str
    masc_v5_witness_cid: str
    team_consensus_controller_v4_witness_cid: str
    multi_branch_rejoin_falsifier_witness_cid: str
    hosted_router_v2_witness_cid: str
    hosted_logprob_router_v2_witness_cid: str
    hosted_cache_planner_v2_witness_cid: str
    hosted_real_substrate_boundary_v2_cid: str
    hosted_wall_v2_report_cid: str
    handoff_coordinator_witness_cid: str
    handoff_envelope_chain_cid: str
    fourteen_way_used: bool
    substrate_v14_used: bool
    masc_v5_v14_beats_v13_rate: float
    masc_v5_tsc_v14_beats_tsc_v13_rate: float
    hosted_router_v2_chosen: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "w68_outer_cid": str(self.w68_outer_cid),
            "w69_params_cid": str(self.w69_params_cid),
            "substrate_v14_witness_cid": str(
                self.substrate_v14_witness_cid),
            "kv_bridge_v14_witness_cid": str(
                self.kv_bridge_v14_witness_cid),
            "hsb_v13_witness_cid": str(
                self.hsb_v13_witness_cid),
            "prefix_state_v13_witness_cid": str(
                self.prefix_state_v13_witness_cid),
            "cache_controller_v12_witness_cid": str(
                self.cache_controller_v12_witness_cid),
            "replay_controller_v10_witness_cid": str(
                self.replay_controller_v10_witness_cid),
            "persistent_v21_witness_cid": str(
                self.persistent_v21_witness_cid),
            "multi_hop_v19_witness_cid": str(
                self.multi_hop_v19_witness_cid),
            "mlsc_v17_witness_cid": str(
                self.mlsc_v17_witness_cid),
            "consensus_v15_witness_cid": str(
                self.consensus_v15_witness_cid),
            "crc_v17_witness_cid": str(
                self.crc_v17_witness_cid),
            "lhr_v21_witness_cid": str(
                self.lhr_v21_witness_cid),
            "ecc_v21_witness_cid": str(
                self.ecc_v21_witness_cid),
            "tvs_v18_witness_cid": str(
                self.tvs_v18_witness_cid),
            "uncertainty_v17_witness_cid": str(
                self.uncertainty_v17_witness_cid),
            "deep_substrate_hybrid_v14_witness_cid": str(
                self.deep_substrate_hybrid_v14_witness_cid),
            "substrate_adapter_v14_matrix_cid": str(
                self.substrate_adapter_v14_matrix_cid),
            "masc_v5_witness_cid": str(self.masc_v5_witness_cid),
            "team_consensus_controller_v4_witness_cid": str(
                self.team_consensus_controller_v4_witness_cid),
            "multi_branch_rejoin_falsifier_witness_cid": str(
                self.multi_branch_rejoin_falsifier_witness_cid),
            "hosted_router_v2_witness_cid": str(
                self.hosted_router_v2_witness_cid),
            "hosted_logprob_router_v2_witness_cid": str(
                self.hosted_logprob_router_v2_witness_cid),
            "hosted_cache_planner_v2_witness_cid": str(
                self.hosted_cache_planner_v2_witness_cid),
            "hosted_real_substrate_boundary_v2_cid": str(
                self.hosted_real_substrate_boundary_v2_cid),
            "hosted_wall_v2_report_cid": str(
                self.hosted_wall_v2_report_cid),
            "handoff_coordinator_witness_cid": str(
                self.handoff_coordinator_witness_cid),
            "handoff_envelope_chain_cid": str(
                self.handoff_envelope_chain_cid),
            "fourteen_way_used": bool(self.fourteen_way_used),
            "substrate_v14_used": bool(self.substrate_v14_used),
            "masc_v5_v14_beats_v13_rate": float(round(
                self.masc_v5_v14_beats_v13_rate, 12)),
            "masc_v5_tsc_v14_beats_tsc_v13_rate": float(round(
                self.masc_v5_tsc_v14_beats_tsc_v13_rate, 12)),
            "hosted_router_v2_chosen": str(
                self.hosted_router_v2_chosen),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w69_handoff_envelope",
            "envelope": self.to_dict()})


def verify_w69_handoff(
        envelope: W69HandoffEnvelope,
        params: W69Params,
        w68_outer_cid: str,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if envelope.schema != W69_SCHEMA_VERSION:
        failures.append("w69_outer_envelope_schema_mismatch")
    if envelope.w68_outer_cid != str(w68_outer_cid):
        failures.append(
            "w69_outer_envelope_w68_outer_cid_drift")
    if envelope.w69_params_cid != params.cid():
        failures.append(
            "w69_outer_envelope_w69_params_cid_drift")
    return (len(failures) == 0), failures


@dataclasses.dataclass
class W69Team:
    params: W69Params

    def run_team_turn(
            self, *,
            w68_outer_cid: str,
            ids: Sequence[int] | None = None,
            text: str = "w69",
    ) -> W69HandoffEnvelope:
        p = self.params
        if not p.enabled or p.substrate_v14 is None:
            return W69HandoffEnvelope(
                schema=W69_SCHEMA_VERSION,
                w68_outer_cid=str(w68_outer_cid),
                w69_params_cid=str(p.cid()),
                substrate_v14_witness_cid="",
                kv_bridge_v14_witness_cid="",
                hsb_v13_witness_cid="",
                prefix_state_v13_witness_cid="",
                cache_controller_v12_witness_cid="",
                replay_controller_v10_witness_cid="",
                persistent_v21_witness_cid="",
                multi_hop_v19_witness_cid="",
                mlsc_v17_witness_cid="",
                consensus_v15_witness_cid="",
                crc_v17_witness_cid="",
                lhr_v21_witness_cid="",
                ecc_v21_witness_cid="",
                tvs_v18_witness_cid="",
                uncertainty_v17_witness_cid="",
                deep_substrate_hybrid_v14_witness_cid="",
                substrate_adapter_v14_matrix_cid="",
                masc_v5_witness_cid="",
                team_consensus_controller_v4_witness_cid="",
                multi_branch_rejoin_falsifier_witness_cid="",
                hosted_router_v2_witness_cid="",
                hosted_logprob_router_v2_witness_cid="",
                hosted_cache_planner_v2_witness_cid="",
                hosted_real_substrate_boundary_v2_cid="",
                hosted_wall_v2_report_cid="",
                handoff_coordinator_witness_cid="",
                handoff_envelope_chain_cid="",
                fourteen_way_used=False,
                substrate_v14_used=False,
                masc_v5_v14_beats_v13_rate=0.0,
                masc_v5_tsc_v14_beats_tsc_v13_rate=0.0,
                hosted_router_v2_chosen="",
            )
        # Plane B — substrate forward.
        token_ids = (
            list(ids) if ids is not None
            else tokenize_bytes_v14(str(text), max_len=16))
        trace, cache = forward_tiny_substrate_v14(
            p.substrate_v14, token_ids)
        # Exercise the substrate's V14 axes.
        trigger_silent_corruption_v14(
            cache, role="r0", corrupted_bytes=2,
            member_replaced=True, detect_turn=1)
        repair_silent_corruption_v14(
            cache, role="r0", repair_turn=2)
        record_multi_branch_rejoin_witness_v14(
            cache, layer_index=0, head_index=0, slot=0,
            witness=0.7)
        sub_witness = emit_tiny_substrate_v14_forward_witness(
            trace, cache)
        # KV V14 witness — falsifier + fingerprint.
        kv_falsifier = (
            probe_kv_bridge_v14_multi_branch_rejoin_falsifier(
                multi_branch_rejoin_flag=0.5))
        sc_fp = compute_silent_corruption_fingerprint_v14(
            role="planner", corrupted_bytes=2,
            member_replaced=True, task_id="t",
            team_id="team")
        kv_witness = emit_kv_bridge_v14_witness(
            projection=p.kv_bridge_v14,
            multi_branch_rejoin_falsifier=kv_falsifier,
            silent_corruption_fingerprint=sc_fp)
        # HSB V13 witness.
        margin = compute_hsb_v13_multi_branch_rejoin_margin(
            hidden_residual_l2=0.05, kv_residual_l2=0.2,
            prefix_residual_l2=0.2, replay_residual_l2=0.2,
            recover_residual_l2=0.2,
            branch_merge_residual_l2=0.2,
            agent_replacement_residual_l2=0.2,
            multi_branch_rejoin_residual_l2=0.2)
        hsb_witness = emit_hsb_v13_witness(
            projection=p.hidden_state_bridge_v13,
            multi_branch_rejoin_margin=margin,
            hidden_vs_multi_branch_rejoin_mean=0.95)
        prefix_witness = emit_prefix_state_v13_witness()
        cache_witness = emit_cache_controller_v12_witness(
            controller=p.cache_controller_v12)
        replay_witness = emit_replay_controller_v10_witness(
            p.replay_controller_v10)
        persist_chain = PersistentLatentStateV21Chain.empty()
        persist_witness = emit_persistent_v21_witness(
            persist_chain)
        mh_witness = emit_multi_hop_v19_witness()
        # MLSC V17 — wrap a trivial V3 capsule up the chain.
        from .mergeable_latent_capsule_v3 import (
            make_root_capsule_v3)
        from .mergeable_latent_capsule_v4 import wrap_v3_as_v4
        from .mergeable_latent_capsule_v5 import wrap_v4_as_v5
        from .mergeable_latent_capsule_v6 import wrap_v5_as_v6
        from .mergeable_latent_capsule_v7 import wrap_v6_as_v7
        from .mergeable_latent_capsule_v8 import wrap_v7_as_v8
        from .mergeable_latent_capsule_v9 import wrap_v8_as_v9
        from .mergeable_latent_capsule_v10 import wrap_v9_as_v10
        from .mergeable_latent_capsule_v11 import wrap_v10_as_v11
        from .mergeable_latent_capsule_v12 import wrap_v11_as_v12
        from .mergeable_latent_capsule_v13 import wrap_v12_as_v13
        from .mergeable_latent_capsule_v14 import wrap_v13_as_v14
        from .mergeable_latent_capsule_v15 import wrap_v14_as_v15
        from .mergeable_latent_capsule_v16 import wrap_v15_as_v16
        v3 = make_root_capsule_v3(
            branch_id="w69_smoke",
            payload=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            fact_tags=("w69",), confidence=0.9, trust=0.9,
            turn_index=0)
        v4 = wrap_v3_as_v4(v3)
        v5 = wrap_v4_as_v5(v4)
        v6 = wrap_v5_as_v6(v5)
        v7 = wrap_v6_as_v7(v6)
        v8 = wrap_v7_as_v8(v7)
        v9 = wrap_v8_as_v9(v8)
        v10 = wrap_v9_as_v10(v9)
        v11 = wrap_v10_as_v11(v10)
        v12 = wrap_v11_as_v12(v11)
        v13 = wrap_v12_as_v13(v12)
        v14 = wrap_v13_as_v14(v13)
        v15 = wrap_v14_as_v15(v14)
        v16 = wrap_v15_as_v16(v15)
        v17 = wrap_v16_as_v17(v16)
        mlsc_witness = emit_mlsc_v17_witness(v17)
        consensus_witness = emit_consensus_v15_witness(
            p.consensus_v15)
        crc_witness = emit_corruption_robustness_v17_witness(
            crc_v17=p.crc_v17, n_probes=3, seed=69150)
        lhr_witness = emit_lhr_v21_witness(
            p.lhr_v21, carrier=[0.1] * 6, k=16,
            partial_contradiction_indicator=[0.5] * 8,
            multi_branch_rejoin_indicator=[0.6] * 8)
        ecc_compression = {
            "v20_compression_result": {
                "v19_compression_result": {
                    "structured_bits_v19": 100,
                    "parity_bits_per_segment_tuple": 24,
                },
                "structured_bits_v20": 110,
                "visible_tokens": 4,
                "bits_per_visible_token": 27.5,
                "meta17_index": 1,
                "parity_bits_per_segment_tuple": 32,
            },
            "structured_bits_v21": 120,
            "visible_tokens": 4,
            "bits_per_visible_token": 30.0,
            "meta18_index": 1,
            "parity_bits_per_segment_tuple": 40,
        }
        ecc_witness = emit_ecc_v21_compression_witness(
            codebook=p.ecc_v21, compression=ecc_compression)
        tvs_result = nineteen_arm_compare(
            per_turn_multi_branch_rejoin_resolution_fidelities=[
                0.6],
            per_turn_partial_contradiction_resolution_fidelities=[
                0.5],
            per_turn_branch_merge_reconciliation_fidelities=[0.7],
            per_turn_team_failure_recovery_fidelities=[0.6],
            per_turn_team_substrate_coordination_fidelities=[0.7],
            per_turn_replay_dominance_primary_fidelities=[0.7],
            per_turn_hidden_wins_fidelities=[0.6],
            per_turn_replay_dominance_fidelities=[0.7],
            per_turn_confidences=[0.8],
            per_turn_trust_scores=[0.7],
            per_turn_merge_retentions=[0.6],
            per_turn_tw_retentions=[0.6],
            per_turn_substrate_fidelities=[0.5],
            per_turn_hidden_fidelities=[0.4],
            per_turn_cache_fidelities=[0.5],
            per_turn_retrieval_fidelities=[0.6],
            per_turn_replay_fidelities=[0.7],
            per_turn_attention_pattern_fidelities=[0.9],
            budget_tokens=6,
        )
        tvs_witness = emit_tvs_arbiter_v18_witness(tvs_result)
        uncertainty = compose_uncertainty_report_v17(
            multi_branch_rejoin_resolution_fidelities=[
                0.74, 0.70],
            partial_contradiction_resolution_fidelities=[
                0.76, 0.72],
            confidences=[0.7, 0.5],
            trusts=[0.9, 0.8],
            substrate_fidelities=[0.95, 0.9],
            hidden_state_fidelities=[0.95, 0.92],
            cache_reuse_fidelities=[0.95, 0.92],
            retrieval_fidelities=[0.95, 0.93],
            replay_fidelities=[0.92, 0.90],
            attention_pattern_fidelities=[0.95, 0.90],
            replay_dominance_fidelities=[0.88, 0.85],
            hidden_wins_fidelities=[0.86, 0.82],
            replay_dominance_primary_fidelities=[0.84, 0.80],
            team_coordination_fidelities=[0.80, 0.75],
            team_failure_recovery_fidelities=[0.78, 0.74],
            branch_merge_reconciliation_fidelities=[0.78, 0.74])
        uncertainty_witness = emit_uncertainty_v17_witness(
            uncertainty)
        # Deep substrate hybrid V14.
        v13_witness = DeepSubstrateHybridV13ForwardWitness(
            schema="coordpy.deep_substrate_hybrid_v13.v1",
            hybrid_cid="",
            inner_v12_witness_cid="",
            thirteen_way=True,
            cache_controller_v11_fired=True,
            replay_controller_v9_fired=True,
            partial_contradiction_witness_active=True,
            agent_replacement_active=True,
            prefix_reuse_active=True,
            team_consensus_controller_v3_active=True,
            partial_contradiction_witness_l1=0.5,
            agent_replacement_count=1,
            prefix_reuse_count=2)
        deep_v14_witness = deep_substrate_hybrid_v14_forward(
            hybrid=p.deep_substrate_hybrid_v14,
            v13_witness=v13_witness,
            cache_controller_v12=p.cache_controller_v12,
            replay_controller_v10=p.replay_controller_v10,
            multi_branch_rejoin_witness_l1=0.5,
            silent_corruption_count=1,
            substrate_self_checksum_cid=str(
                cache.substrate_self_checksum_cid),
            n_team_consensus_v4_invocations=1)
        adapter_matrix = probe_all_v14_adapters()
        # MASC V5 — run a batch for the envelope.
        per_regime_aggs = {}
        for regime in W69_MASC_V5_REGIMES:
            _, agg = p.multi_agent_coordinator_v5.run_batch(
                seeds=list(range(int(p.masc_v5_n_seeds))),
                regime=regime)
            per_regime_aggs[regime] = agg
        masc_witness = (
            emit_multi_agent_substrate_coordinator_v5_witness(
                coordinator=p.multi_agent_coordinator_v5,
                per_regime_aggregate=per_regime_aggs))
        tcc_witness = emit_team_consensus_controller_v4_witness(
            p.team_consensus_controller_v4)
        # Plane A V2 — hosted.
        planned, _ = p.hosted_cache_planner_v2.plan_per_role_staggered(
            shared_prefix_text="W69 team shared prefix " * 8,
            per_role_blocks={
                "plan": ["t0", "t1"],
                "research": ["r0", "r1"],
                "write": ["w0", "w1"],
            })
        router_v2_witness = (
            emit_hosted_router_controller_v2_witness(
                p.hosted_router_v2))
        logprob_v2_witness = (
            emit_hosted_logprob_router_v2_witness(
                p.hosted_logprob_router_v2))
        cache_planner_v2_witness = (
            emit_hosted_cache_aware_planner_v2_witness(
                p.hosted_cache_planner_v2))
        boundary_v2 = p.hosted_real_substrate_boundary_v2
        wall_v2_report = build_wall_report_v2(
            boundary=boundary_v2)
        # Handoff coordinator decisions.
        env_text_only = p.handoff_coordinator.decide(
            req=HandoffRequest(
                request_cid="w69-turn-text",
                needs_text_only=True,
                needs_substrate_state_access=False),
            substrate_self_checksum_cid=str(
                cache.substrate_self_checksum_cid))
        env_substrate_only = p.handoff_coordinator.decide(
            req=HandoffRequest(
                request_cid="w69-turn-substrate",
                needs_text_only=False,
                needs_substrate_state_access=True),
            substrate_self_checksum_cid=str(
                cache.substrate_self_checksum_cid))
        env_audit = p.handoff_coordinator.decide(
            req=HandoffRequest(
                request_cid="w69-turn-audit",
                needs_text_only=True,
                needs_substrate_state_access=True),
            substrate_self_checksum_cid=str(
                cache.substrate_self_checksum_cid))
        handoff_witness = (
            emit_hosted_real_handoff_coordinator_witness(
                p.handoff_coordinator))
        handoff_envelope_chain_cid = _sha256_hex({
            "kind": "w69_handoff_envelope_chain",
            "envelopes": [
                env_text_only.cid(),
                env_substrate_only.cid(),
                env_audit.cid(),
            ],
        })
        baseline_agg = per_regime_aggs.get(
            W69_MASC_V5_REGIMES[0])
        v14_beats = (
            float(baseline_agg.v14_beats_v13_rate)
            if baseline_agg is not None else 0.0)
        tsc_v14_beats = (
            float(baseline_agg.tsc_v14_beats_tsc_v13_rate)
            if baseline_agg is not None else 0.0)
        return W69HandoffEnvelope(
            schema=W69_SCHEMA_VERSION,
            w68_outer_cid=str(w68_outer_cid),
            w69_params_cid=str(p.cid()),
            substrate_v14_witness_cid=str(sub_witness.cid()),
            kv_bridge_v14_witness_cid=str(kv_witness.cid()),
            hsb_v13_witness_cid=str(hsb_witness.cid()),
            prefix_state_v13_witness_cid=str(
                prefix_witness.cid()),
            cache_controller_v12_witness_cid=str(
                cache_witness.cid()),
            replay_controller_v10_witness_cid=str(
                replay_witness.cid()),
            persistent_v21_witness_cid=str(persist_witness.cid()),
            multi_hop_v19_witness_cid=str(mh_witness.cid()),
            mlsc_v17_witness_cid=str(mlsc_witness.cid()),
            consensus_v15_witness_cid=str(
                consensus_witness.cid()),
            crc_v17_witness_cid=str(crc_witness.cid()),
            lhr_v21_witness_cid=str(lhr_witness.cid()),
            ecc_v21_witness_cid=str(ecc_witness.cid()),
            tvs_v18_witness_cid=str(tvs_witness.cid()),
            uncertainty_v17_witness_cid=str(
                uncertainty_witness.cid()),
            deep_substrate_hybrid_v14_witness_cid=str(
                deep_v14_witness.cid()),
            substrate_adapter_v14_matrix_cid=str(
                adapter_matrix.cid()),
            masc_v5_witness_cid=str(masc_witness.cid()),
            team_consensus_controller_v4_witness_cid=str(
                tcc_witness.cid()),
            multi_branch_rejoin_falsifier_witness_cid=str(
                kv_falsifier.cid()),
            hosted_router_v2_witness_cid=str(
                router_v2_witness.cid()),
            hosted_logprob_router_v2_witness_cid=str(
                logprob_v2_witness.cid()),
            hosted_cache_planner_v2_witness_cid=str(
                cache_planner_v2_witness.cid()),
            hosted_real_substrate_boundary_v2_cid=str(
                boundary_v2.cid()),
            hosted_wall_v2_report_cid=str(
                wall_v2_report.cid()),
            handoff_coordinator_witness_cid=str(
                handoff_witness.cid()),
            handoff_envelope_chain_cid=str(
                handoff_envelope_chain_cid),
            fourteen_way_used=bool(
                deep_v14_witness.fourteen_way),
            substrate_v14_used=True,
            masc_v5_v14_beats_v13_rate=float(v14_beats),
            masc_v5_tsc_v14_beats_tsc_v13_rate=float(
                tsc_v14_beats),
            hosted_router_v2_chosen="",
        )


def build_default_w69_team(*, seed: int = 69000) -> W69Team:
    return W69Team(params=W69Params.build_default(seed=int(seed)))


__all__ = [
    "W69_SCHEMA_VERSION",
    "W69_FAILURE_MODES",
    "W69Params",
    "W69HandoffEnvelope",
    "verify_w69_handoff",
    "W69Team",
    "build_default_w69_team",
]
