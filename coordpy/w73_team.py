"""W73 — Stronger Contradiction-Rejoin / Replacement / Delayed-
Repair Budget-Primary Two-Plane Multi-Agent Substrate team.

The ``W73Team`` orchestrator strictly wraps the ``W72Team`` and
adds the W73 mechanism modules organised across two planes plus
the new **replacement-aware Plane A↔B handoff coordinator V5**:

**Plane B — Real substrate (in-repo, V18 stack):**

* M1  ``tiny_substrate_v18``           (20-layer, 3 new V18 axes)
* M2  ``kv_bridge_v18``                (14-target ridge + 110-dim
                                        replacement-repair
                                        fingerprint + replacement-
                                        pressure falsifier)
* M3  ``cache_controller_v16``         (13-objective ridge + per-
                                        role 14-dim replacement-
                                        pressure head)
* M4  ``replay_controller_v14``        (21 regimes + 11-label
                                        replacement-aware routing
                                        head)
* M5  ``deep_substrate_hybrid_v18``    (18-way bidirectional loop)
* M6  ``substrate_adapter_v18``        (substrate_v18_full tier)
* M7  ``persistent_latent_v25``        (24 layers, 22nd carrier,
                                        max_chain_walk_depth=1048576)
* M8  ``long_horizon_retention_v25``   (24 heads, max_k=768)
* M9  ``mergeable_latent_capsule_v21`` (replacement-repair-
                                        trajectory chain +
                                        contradiction chain)
* M10 ``consensus_fallback_controller_v19`` (32-stage chain)
* M11 ``multi_agent_substrate_coordinator_v9`` (20-policy, 13-
                                                regime MASC V9)
* M12 ``team_consensus_controller_v8`` (replacement-pressure +
                                        replacement-after-CTR
                                        arbiters)

**Plane A — Hosted control plane V6 (honest, no substrate):**

* H1  ``hosted_router_controller_v6``  (replacement-pressure
                                        weighting + replacement-
                                        after-CTR match)
* H2  ``hosted_logprob_router_v6``     (replacement-aware abstain
                                        floor + per-budget+restart+
                                        rejoin+replacement
                                        tiebreak)
* H3  ``hosted_cache_aware_planner_v6``(four-layer rotated prefix +
                                        ≥ 85 % savings 14×8 hit=1)
* H4  ``hosted_cost_planner_v6``       (cost-per-replacement-
                                        rejoin-success-under-budget
                                        + abstain-when-replacement-
                                        pressure-violated)
* H5  ``hosted_real_substrate_boundary_v6`` (wall V6, 31 blocked
                                             axes)
* H6  ``hosted_real_handoff_coordinator_v5`` (the **new
                                              replacement-aware
                                              Plane A↔B bridge** —
                                              V5 envelopes +
                                              replacement falsifier
                                              + cross-plane savings)
* H7  ``hosted_provider_filter_v5``    (replacement-aware provider
                                        filter)

Per-turn it emits 19 W73 module witness CIDs (12 Plane B + 7 Plane
A V6) and a V5 handoff envelope CID, sealing them into a
``W73HandoffEnvelope`` whose ``w72_outer_cid`` carries forward the
W72 envelope byte-for-byte.

Honest scope (W73)
------------------

* Plane A V6 operates at the hosted text/logprob/prefix-cache
  surface. It does NOT pierce hidden state / KV / attention.
  ``W73-L-HOSTED-V6-NO-SUBSTRATE-CAP``.
* Plane B is the in-repo V18 NumPy runtime. We do NOT bridge to
  third-party hosted models.
  ``W73-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
* W73 fits closed-form ridge parameters in three new places on top
  of W72's 64: cache V16 thirteen-objective; cache V16 per-role
  replacement-pressure; replay V14 replacement-aware routing head;
  KV V18 fourteen-target. Total **67 closed-form ridge solves**
  across W61..W73. No autograd, no SGD, no GPU.
* Trivial passthrough preserved: when ``W73Params.build_trivial()``
  is used the W73 envelope's internal ``w72_outer_cid`` carries
  the supplied W72 outer CID exactly.
* The handoff coordinator V5 preserves the wall: a content-
  addressed V5 envelope says which plane handled each turn under
  the replacement-aware score; it does NOT cross the substrate
  boundary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .cache_controller_v16 import (
    CacheControllerV16, emit_cache_controller_v16_witness,
    fit_per_role_replacement_pressure_head_v16,
    fit_thirteen_objective_ridge_v16,
)
from .consensus_fallback_controller_v19 import (
    ConsensusFallbackControllerV19,
    emit_consensus_v19_witness,
)
from .deep_substrate_hybrid_v17 import (
    DeepSubstrateHybridV17ForwardWitness,
)
from .deep_substrate_hybrid_v18 import (
    DeepSubstrateHybridV18,
    deep_substrate_hybrid_v18_forward,
)
from .hosted_cache_aware_planner_v6 import (
    HostedCacheAwarePlannerV6,
    emit_hosted_cache_aware_planner_v6_witness,
    hosted_cache_aware_savings_v6_vs_recompute,
)
from .hosted_cost_planner_v6 import HostedCostPlanSpecV6
from .hosted_logprob_router_v6 import (
    HostedLogprobRouterV6,
    emit_hosted_logprob_router_v6_witness,
)
from .hosted_provider_filter_v5 import (
    HostedProviderFilterSpecV5,
    filter_hosted_registry_v5,
)
from .hosted_real_handoff_coordinator_v5 import (
    HandoffRequestV5, HostedRealHandoffCoordinatorV5,
    emit_hosted_real_handoff_coordinator_v5_witness,
    hosted_real_handoff_v5_replacement_aware_savings,
)
from .hosted_real_handoff_coordinator_v4 import HandoffRequestV4
from .hosted_real_handoff_coordinator_v3 import HandoffRequestV3
from .hosted_real_handoff_coordinator_v2 import HandoffRequestV2
from .hosted_real_handoff_coordinator import HandoffRequest
from .hosted_real_substrate_boundary_v6 import (
    HostedRealSubstrateBoundaryV6,
    build_default_hosted_real_substrate_boundary_v6,
    build_wall_report_v6,
    probe_hosted_real_substrate_boundary_v6_falsifier,
)
from .hosted_router_controller import (
    HostedProviderRegistry, HostedRoutingRequest,
    default_hosted_registry,
)
from .hosted_router_controller_v2 import HostedRoutingRequestV2
from .hosted_router_controller_v3 import HostedRoutingRequestV3
from .hosted_router_controller_v4 import HostedRoutingRequestV4
from .hosted_router_controller_v5 import HostedRoutingRequestV5
from .hosted_router_controller_v6 import (
    HostedRouterControllerV6, HostedRoutingRequestV6,
    emit_hosted_router_controller_v6_witness,
)
from .kv_bridge_v17 import KVBridgeV17Projection
from .kv_bridge_v18 import (
    KVBridgeV18Projection,
    compute_replacement_repair_fingerprint_v18,
    emit_kv_bridge_v18_witness,
    probe_kv_bridge_v18_replacement_pressure_falsifier,
)
from .long_horizon_retention_v25 import (
    LongHorizonReconstructionV25Head,
    emit_lhr_v25_witness,
)
from .mergeable_latent_capsule_v21 import (
    MergeOperatorV21, emit_mlsc_v21_witness, wrap_v20_as_v21,
)
from .multi_agent_substrate_coordinator_v9 import (
    MultiAgentSubstrateCoordinatorV9,
    W73_MASC_V9_REGIMES,
    emit_multi_agent_substrate_coordinator_v9_witness,
)
from .persistent_latent_v25 import (
    PersistentLatentStateV25Chain,
    emit_persistent_v25_witness,
)
from .replay_controller_v14 import (
    ReplayControllerV14,
    W73_REPLACEMENT_AWARE_ROUTING_LABELS,
    W73_REPLAY_REGIMES_V14,
    emit_replay_controller_v14_witness,
    fit_replay_controller_v14_per_role,
    fit_replay_v14_replacement_aware_routing_head,
)
from .replay_controller import ReplayCandidate
from .substrate_adapter_v18 import (
    W73_SUBSTRATE_TIER_SUBSTRATE_V18_FULL,
    probe_all_v18_adapters,
)
from .team_consensus_controller_v8 import (
    TeamConsensusControllerV8,
    emit_team_consensus_controller_v8_witness,
)
from .tiny_substrate_v18 import (
    TinyV18SubstrateParams,
    build_default_tiny_substrate_v18,
    emit_tiny_substrate_v18_forward_witness,
    forward_tiny_substrate_v18,
    record_contradiction_event_v18,
    record_replacement_event_v18,
    record_replacement_window_v18,
    tokenize_bytes_v18,
)
from .w72_team import (
    W72HandoffEnvelope, W72Params, W72Team,
)


W73_SCHEMA_VERSION: str = "coordpy.w73_team.v1"

W73_FAILURE_MODES: tuple[str, ...] = (
    "w73_outer_envelope_schema_mismatch",
    "w73_outer_envelope_w72_outer_cid_drift",
    "w73_outer_envelope_w73_params_cid_drift",
    "w73_outer_envelope_witness_cid_drift",
    "w73_substrate_v18_n_layers_off",
    "w73_substrate_v18_replacement_repair_trajectory_cid_off",
    "w73_substrate_v18_replacement_after_ctr_per_layer_shape_off",
    "w73_substrate_v18_replacement_pressure_gate_shape_off",
    "w73_kv_bridge_v18_n_targets_off",
    "w73_kv_bridge_v18_replacement_pressure_falsifier_off",
    "w73_cache_v16_thirteen_objective_off",
    "w73_replay_v14_regime_count_off",
    "w73_replay_v14_replacement_aware_routing_count_off",
    "w73_consensus_v19_stage_count_off",
    "w73_lhr_v25_max_k_off",
    "w73_lhr_v25_n_heads_off",
    "w73_persistent_v25_n_layers_off",
    "w73_substrate_adapter_v18_tier_off",
    "w73_masc_v9_v18_beats_v17_rate_under_threshold",
    "w73_masc_v9_tsc_v18_beats_tsc_v17_rate_under_threshold",
    "w73_masc_v9_compound_regime_inferior_to_baseline",
    "w73_hosted_router_v6_decision_not_deterministic",
    "w73_hosted_logprob_v6_abstain_floor_off",
    "w73_hosted_cache_aware_v6_savings_below_85_percent",
    "w73_hosted_cost_planner_v6_no_eligible",
    "w73_hosted_real_substrate_boundary_v6_blocked_axis_satisfied",
    "w73_eighteen_way_loop_not_observed",
    "w73_handoff_coordinator_v5_inconsistent",
    "w73_handoff_v5_cross_plane_savings_below_80_percent",
    "w73_team_consensus_v8_no_decisions",
    "w73_handoff_v5_replacement_alignment_off",
    "w73_handoff_envelope_v5_chain_cid_drift",
    "w73_inner_v72_envelope_invariant_off",
    "w73_handoff_v5_replacement_after_ctr_fallback_off",
    "w73_hosted_boundary_v6_blocked_axes_below_31",
    "w73_v18_substrate_self_checksum_cid_off",
    "w73_replacement_repair_trajectory_cid_drift",
    "w73_mlsc_v21_replacement_repair_trajectory_chain_off",
    "w73_v9_team_success_per_visible_token_below_floor",
    "w73_v9_visible_tokens_savings_below_65_percent",
    "w73_v9_compound_regime_v18_beats_v17_below_threshold",
    "w73_substrate_v18_replacement_repair_trajectory_chain_synthetic",
    "w73_inner_v18_falsifier_kind_off",
    "w73_handoff_v5_envelope_rp_alignment_off",
    "w73_hosted_router_v6_per_routing_cid_off",
    "w73_consensus_v19_replacement_pressure_arbiter_off",
    "w73_consensus_v19_replacement_after_ctr_arbiter_off",
    "w73_tcc_v8_replacement_pressure_arbiter_off",
    "w73_tcc_v8_replacement_after_ctr_arbiter_off",
    "w73_cache_v16_per_role_replacement_pressure_head_off",
    "w73_kv_bridge_v18_replacement_repair_fingerprint_off",
    "w73_substrate_v18_replacement_events_off",
    "w73_provider_filter_v5_pressure_drop_off",
    "w73_handoff_v5_dr_alignment_off",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class W73Params:
    substrate_v18: TinyV18SubstrateParams | None
    kv_bridge_v18: KVBridgeV18Projection | None
    cache_controller_v16: CacheControllerV16 | None
    replay_controller_v14: ReplayControllerV14 | None
    consensus_v19: ConsensusFallbackControllerV19 | None
    lhr_v25: LongHorizonReconstructionV25Head | None
    deep_substrate_hybrid_v18: DeepSubstrateHybridV18 | None
    mlsc_v21_operator: MergeOperatorV21 | None
    multi_agent_coordinator_v9: (
        MultiAgentSubstrateCoordinatorV9 | None)
    team_consensus_controller_v8: (
        TeamConsensusControllerV8 | None)
    hosted_registry: HostedProviderRegistry | None
    hosted_router_v6: HostedRouterControllerV6 | None
    hosted_logprob_router_v6: HostedLogprobRouterV6 | None
    hosted_cache_planner_v6: HostedCacheAwarePlannerV6 | None
    hosted_real_substrate_boundary_v6: (
        HostedRealSubstrateBoundaryV6 | None)
    handoff_coordinator_v5: (
        HostedRealHandoffCoordinatorV5 | None)
    hosted_provider_filter_v5: (
        HostedProviderFilterSpecV5 | None)
    w72_params: W72Params | None
    enabled: bool = True
    masc_v9_n_seeds: int = 6

    @classmethod
    def build_trivial(cls) -> "W73Params":
        return cls(
            substrate_v18=None,
            kv_bridge_v18=None,
            cache_controller_v16=None,
            replay_controller_v14=None,
            consensus_v19=None, lhr_v25=None,
            deep_substrate_hybrid_v18=None,
            mlsc_v21_operator=None,
            multi_agent_coordinator_v9=None,
            team_consensus_controller_v8=None,
            hosted_registry=None,
            hosted_router_v6=None,
            hosted_logprob_router_v6=None,
            hosted_cache_planner_v6=None,
            hosted_real_substrate_boundary_v6=None,
            handoff_coordinator_v5=None,
            hosted_provider_filter_v5=None,
            w72_params=None,
            enabled=False,
        )

    @classmethod
    def build_default(
            cls, *, seed: int = 73000,
    ) -> "W73Params":
        sub_v18 = build_default_tiny_substrate_v18(
            seed=int(seed) + 1)
        # KV V18 projection chain.
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
        from .kv_bridge_v14 import KVBridgeV14Projection
        from .kv_bridge_v15 import KVBridgeV15Projection
        from .kv_bridge_v16 import KVBridgeV16Projection
        cfg = (
            sub_v18.config.v17.v16.v15.v14.v13.v12.v11.v10.v9)
        d_head = int(cfg.d_model) // int(cfg.n_heads)
        kv_b3 = KVBridgeV3Projection.init(
            n_layers=int(cfg.n_layers),
            n_heads=int(cfg.n_heads),
            n_kv_heads=int(cfg.n_kv_heads),
            n_inject_tokens=3, carrier_dim=6,
            d_head=int(d_head), seed=int(seed) + 7)
        kv_b16 = KVBridgeV16Projection.init_from_v15(
            KVBridgeV15Projection.init_from_v14(
                KVBridgeV14Projection.init_from_v13(
                    KVBridgeV13Projection.init_from_v12(
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
                        seed_v13=int(seed) + 17),
                    seed_v14=int(seed) + 18),
                seed_v15=int(seed) + 19),
            seed_v16=int(seed) + 20)
        kv_b17 = KVBridgeV17Projection.init_from_v16(
            kv_b16, seed_v17=int(seed) + 21)
        kv_b18 = KVBridgeV18Projection.init_from_v17(
            kv_b17, seed_v18=int(seed) + 22)
        cc16 = CacheControllerV16.init(
            fit_seed=int(seed) + 32)
        import numpy as _np
        rng = _np.random.default_rng(int(seed) + 33)
        X = rng.standard_normal((10, 4))
        cc16, _ = fit_thirteen_objective_ridge_v16(
            controller=cc16, train_features=X.tolist(),
            target_drop_oracle=X.sum(axis=-1).tolist(),
            target_retrieval_relevance=X[:, 0].tolist(),
            target_hidden_wins=(
                X[:, 1] - X[:, 2]).tolist(),
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
                X[:, 0] * 0.5 + X[:, 1] * 0.2).tolist(),
            target_budget_primary=(
                X[:, 0] * 0.2 + X[:, 1] * 0.3
                + X[:, 2] * 0.4).tolist(),
            target_restart_dominance=(
                X[:, 3] * 0.4 + X[:, 0] * 0.2).tolist(),
            target_delayed_rejoin_after_restart=(
                X[:, 1] * 0.3 + X[:, 2] * 0.3
                + X[:, 3] * 0.3).tolist(),
            target_replacement_after_ctr=(
                X[:, 0] * 0.3 + X[:, 2] * 0.3
                + X[:, 3] * 0.3).tolist())
        X14 = rng.standard_normal((10, 14))
        cc16, _ = fit_per_role_replacement_pressure_head_v16(
            controller=cc16, role="planner",
            train_features=X14.tolist(),
            target_replacement_priorities=(
                X14[:, 0] * 0.3
                + X14[:, 13] * 0.4).tolist())
        # Replay V14.
        rcv14 = ReplayControllerV14.init()
        v14_cands = {
            r: [ReplayCandidate(
                100, 1000, 50, 0.1, 0.0, 0.3,
                True, True, 0)]
            for r in W73_REPLAY_REGIMES_V14}
        v14_decs = {
            r: ["choose_reuse"]
            for r in W73_REPLAY_REGIMES_V14}
        rcv14, _ = fit_replay_controller_v14_per_role(
            controller=rcv14, role="planner",
            train_candidates_per_regime=v14_cands,
            train_decisions_per_regime=v14_decs)
        X_team = rng.standard_normal((50, 10))
        labs: list[str] = []
        for i in range(50):
            lab_idx = (
                i % len(W73_REPLACEMENT_AWARE_ROUTING_LABELS))
            labs.append(
                W73_REPLACEMENT_AWARE_ROUTING_LABELS[lab_idx])
        rcv14, _ = fit_replay_v14_replacement_aware_routing_head(
            controller=rcv14,
            train_team_features=X_team.tolist(),
            train_routing_labels=labs)
        consensus_v19 = ConsensusFallbackControllerV19.init(
            k_required=2, cosine_floor=0.6,
            trust_threshold=0.5,
            multi_branch_rejoin_threshold=0.5,
            silent_corruption_threshold=0.5,
            repair_dominance_threshold=0.5,
            budget_primary_threshold=0.5,
            restart_aware_threshold=0.5,
            delayed_repair_threshold=0.5,
            rejoin_pressure_threshold=0.5,
            delayed_rejoin_threshold=0.5,
            replacement_pressure_threshold=0.5,
            replacement_after_ctr_threshold=0.5)
        lhr25 = LongHorizonReconstructionV25Head.init(
            seed=int(seed) + 40)
        deep_v18 = DeepSubstrateHybridV18()
        mlsc_v21_op = MergeOperatorV21()
        masc_v9 = MultiAgentSubstrateCoordinatorV9()
        tcc_v8 = TeamConsensusControllerV8()
        reg = default_hosted_registry()
        hosted_router_v6 = HostedRouterControllerV6.init(
            reg, {
                "openrouter_paid": 0.85,
                "openai_paid": 0.92,
            })
        hosted_logprob_router_v6 = HostedLogprobRouterV6()
        hosted_cache_planner_v6 = HostedCacheAwarePlannerV6()
        boundary_v6 = (
            build_default_hosted_real_substrate_boundary_v6())
        handoff_coord_v5 = HostedRealHandoffCoordinatorV5(
            boundary_v6=boundary_v6)
        # Provider filter V5 — pre-configured replacement-aware
        # spec.
        from .hosted_provider_filter_v4 import (
            HostedProviderFilterSpecV4,
        )
        from .hosted_provider_filter_v3 import (
            HostedProviderFilterSpecV3,
        )
        from .hosted_provider_filter_v2 import (
            HostedProviderFilterSpecV2,
        )
        from .hosted_provider_filter import (
            HostedProviderFilterSpec,
        )
        provider_filter_v5 = HostedProviderFilterSpecV5(
            inner_v4=HostedProviderFilterSpecV4(
                inner_v3=HostedProviderFilterSpecV3(
                    inner_v2=HostedProviderFilterSpecV2(
                        inner_specs=(
                            HostedProviderFilterSpec(
                                require_data_policy="no_log",
                                allowed_tiers=(
                                    "logprobs",
                                    "logprobs_and_prefix_cache",
                                    "prefix_cache",
                                    "text_only"),
                                max_p50_latency_ms=10_000.0,
                                max_cost_per_1k_output=100.0),
                        ),
                        combine="all",
                        tier_weights={
                            "logprobs_and_prefix_cache": 1.0,
                            "logprobs": 0.8,
                            "prefix_cache": 0.7,
                            "text_only": 0.5}),
                    restart_pressure=0.7,
                    restart_pressure_floor=0.5,
                    max_restart_noise_per_provider={
                        "openrouter_paid": 0.3,
                        "openai_paid": 1.0},
                    restart_tier_weights={
                        "logprobs_and_prefix_cache": 1.0,
                        "logprobs": 0.7,
                        "prefix_cache": 0.6,
                        "text_only": 0.4}),
                rejoin_pressure=0.7,
                rejoin_pressure_floor=0.5,
                max_rejoin_noise_per_provider={
                    "openrouter_paid": 0.25,
                    "openai_paid": 1.0},
                rejoin_tier_weights={
                    "logprobs_and_prefix_cache": 1.0,
                    "logprobs": 0.65,
                    "prefix_cache": 0.55,
                    "text_only": 0.35}),
            replacement_pressure=0.7,
            replacement_pressure_floor=0.5,
            max_replacement_noise_per_provider={
                "openrouter_paid": 0.20,
                "openai_paid": 1.0},
            replacement_tier_weights={
                "logprobs_and_prefix_cache": 1.0,
                "logprobs": 0.60,
                "prefix_cache": 0.50,
                "text_only": 0.30})
        # W72 inner params for envelope chaining.
        w72_params = W72Params.build_default(
            seed=int(seed) - 1000)
        return cls(
            substrate_v18=sub_v18,
            kv_bridge_v18=kv_b18,
            cache_controller_v16=cc16,
            replay_controller_v14=rcv14,
            consensus_v19=consensus_v19,
            lhr_v25=lhr25,
            deep_substrate_hybrid_v18=deep_v18,
            mlsc_v21_operator=mlsc_v21_op,
            multi_agent_coordinator_v9=masc_v9,
            team_consensus_controller_v8=tcc_v8,
            hosted_registry=reg,
            hosted_router_v6=hosted_router_v6,
            hosted_logprob_router_v6=hosted_logprob_router_v6,
            hosted_cache_planner_v6=hosted_cache_planner_v6,
            hosted_real_substrate_boundary_v6=boundary_v6,
            handoff_coordinator_v5=handoff_coord_v5,
            hosted_provider_filter_v5=provider_filter_v5,
            w72_params=w72_params,
            enabled=True,
            masc_v9_n_seeds=5,
        )

    def to_dict(self) -> dict[str, Any]:
        def _cid_or_empty(x: Any) -> str:
            return str(x.cid()) if x is not None else ""
        return {
            "schema": W73_SCHEMA_VERSION,
            "kind": "w73_params",
            "substrate_v18_cid": _cid_or_empty(
                self.substrate_v18),
            "kv_bridge_v18_cid": _cid_or_empty(
                self.kv_bridge_v18),
            "cache_controller_v16_cid": _cid_or_empty(
                self.cache_controller_v16),
            "replay_controller_v14_cid": _cid_or_empty(
                self.replay_controller_v14),
            "consensus_v19_cid": _cid_or_empty(
                self.consensus_v19),
            "lhr_v25_cid": _cid_or_empty(self.lhr_v25),
            "deep_substrate_hybrid_v18_cid": _cid_or_empty(
                self.deep_substrate_hybrid_v18),
            "mlsc_v21_operator_cid": _cid_or_empty(
                self.mlsc_v21_operator),
            "multi_agent_coordinator_v9_cid": _cid_or_empty(
                self.multi_agent_coordinator_v9),
            "team_consensus_controller_v8_cid": _cid_or_empty(
                self.team_consensus_controller_v8),
            "hosted_registry_cid": _cid_or_empty(
                self.hosted_registry),
            "hosted_router_v6_cid": _cid_or_empty(
                self.hosted_router_v6),
            "hosted_logprob_router_v6_cid": _cid_or_empty(
                self.hosted_logprob_router_v6),
            "hosted_cache_planner_v6_cid": _cid_or_empty(
                self.hosted_cache_planner_v6),
            "hosted_real_substrate_boundary_v6_cid":
                _cid_or_empty(
                    self.hosted_real_substrate_boundary_v6),
            "handoff_coordinator_v5_cid": _cid_or_empty(
                self.handoff_coordinator_v5),
            "hosted_provider_filter_v5_cid": _cid_or_empty(
                self.hosted_provider_filter_v5),
            "w72_params_cid": _cid_or_empty(self.w72_params),
            "enabled": bool(self.enabled),
            "masc_v9_n_seeds": int(self.masc_v9_n_seeds),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w73_params",
            "params": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class W73HandoffEnvelope:
    schema: str
    w72_outer_cid: str
    w73_params_cid: str
    substrate_v18_witness_cid: str
    kv_bridge_v18_witness_cid: str
    cache_controller_v16_witness_cid: str
    replay_controller_v14_witness_cid: str
    persistent_v25_witness_cid: str
    mlsc_v21_witness_cid: str
    consensus_v19_witness_cid: str
    lhr_v25_witness_cid: str
    deep_substrate_hybrid_v18_witness_cid: str
    substrate_adapter_v18_matrix_cid: str
    masc_v9_witness_cid: str
    team_consensus_controller_v8_witness_cid: str
    replacement_pressure_falsifier_witness_cid: str
    hosted_router_v6_witness_cid: str
    hosted_logprob_router_v6_witness_cid: str
    hosted_cache_planner_v6_witness_cid: str
    hosted_real_substrate_boundary_v6_cid: str
    hosted_wall_v6_report_cid: str
    handoff_coordinator_v5_witness_cid: str
    handoff_envelope_v5_chain_cid: str
    provider_filter_v5_report_cid: str
    eighteen_way_used: bool
    substrate_v18_used: bool
    masc_v9_v18_beats_v17_rate: float
    masc_v9_tsc_v18_beats_tsc_v17_rate: float
    masc_v9_team_success_per_visible_token: float
    hosted_router_v6_chosen: str
    replacement_repair_trajectory_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "w72_outer_cid": str(self.w72_outer_cid),
            "w73_params_cid": str(self.w73_params_cid),
            "substrate_v18_witness_cid": str(
                self.substrate_v18_witness_cid),
            "kv_bridge_v18_witness_cid": str(
                self.kv_bridge_v18_witness_cid),
            "cache_controller_v16_witness_cid": str(
                self.cache_controller_v16_witness_cid),
            "replay_controller_v14_witness_cid": str(
                self.replay_controller_v14_witness_cid),
            "persistent_v25_witness_cid": str(
                self.persistent_v25_witness_cid),
            "mlsc_v21_witness_cid": str(
                self.mlsc_v21_witness_cid),
            "consensus_v19_witness_cid": str(
                self.consensus_v19_witness_cid),
            "lhr_v25_witness_cid": str(
                self.lhr_v25_witness_cid),
            "deep_substrate_hybrid_v18_witness_cid": str(
                self.deep_substrate_hybrid_v18_witness_cid),
            "substrate_adapter_v18_matrix_cid": str(
                self.substrate_adapter_v18_matrix_cid),
            "masc_v9_witness_cid": str(
                self.masc_v9_witness_cid),
            "team_consensus_controller_v8_witness_cid": str(
                self.team_consensus_controller_v8_witness_cid),
            "replacement_pressure_falsifier_witness_cid": str(
                self.replacement_pressure_falsifier_witness_cid),
            "hosted_router_v6_witness_cid": str(
                self.hosted_router_v6_witness_cid),
            "hosted_logprob_router_v6_witness_cid": str(
                self.hosted_logprob_router_v6_witness_cid),
            "hosted_cache_planner_v6_witness_cid": str(
                self.hosted_cache_planner_v6_witness_cid),
            "hosted_real_substrate_boundary_v6_cid": str(
                self.hosted_real_substrate_boundary_v6_cid),
            "hosted_wall_v6_report_cid": str(
                self.hosted_wall_v6_report_cid),
            "handoff_coordinator_v5_witness_cid": str(
                self.handoff_coordinator_v5_witness_cid),
            "handoff_envelope_v5_chain_cid": str(
                self.handoff_envelope_v5_chain_cid),
            "provider_filter_v5_report_cid": str(
                self.provider_filter_v5_report_cid),
            "eighteen_way_used": bool(self.eighteen_way_used),
            "substrate_v18_used": bool(self.substrate_v18_used),
            "masc_v9_v18_beats_v17_rate": float(round(
                self.masc_v9_v18_beats_v17_rate, 12)),
            "masc_v9_tsc_v18_beats_tsc_v17_rate": float(round(
                self.masc_v9_tsc_v18_beats_tsc_v17_rate, 12)),
            "masc_v9_team_success_per_visible_token": float(
                round(
                    self
                    .masc_v9_team_success_per_visible_token,
                    12)),
            "hosted_router_v6_chosen": str(
                self.hosted_router_v6_chosen),
            "replacement_repair_trajectory_cid": str(
                self.replacement_repair_trajectory_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w73_handoff_envelope",
            "envelope": self.to_dict()})


def verify_w73_handoff(
        envelope: W73HandoffEnvelope,
        params: W73Params,
        w72_outer_cid: str,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if envelope.schema != W73_SCHEMA_VERSION:
        failures.append(
            "w73_outer_envelope_schema_mismatch")
    if envelope.w72_outer_cid != str(w72_outer_cid):
        failures.append(
            "w73_outer_envelope_w72_outer_cid_drift")
    if envelope.w73_params_cid != params.cid():
        failures.append(
            "w73_outer_envelope_w73_params_cid_drift")
    return (len(failures) == 0), failures


@dataclasses.dataclass
class W73Team:
    params: W73Params

    def run_team_turn(
            self, *,
            w72_outer_cid: str,
            ids: Sequence[int] | None = None,
            text: str = "w73",
    ) -> W73HandoffEnvelope:
        p = self.params
        if not p.enabled or p.substrate_v18 is None:
            return W73HandoffEnvelope(
                schema=W73_SCHEMA_VERSION,
                w72_outer_cid=str(w72_outer_cid),
                w73_params_cid=str(p.cid()),
                substrate_v18_witness_cid="",
                kv_bridge_v18_witness_cid="",
                cache_controller_v16_witness_cid="",
                replay_controller_v14_witness_cid="",
                persistent_v25_witness_cid="",
                mlsc_v21_witness_cid="",
                consensus_v19_witness_cid="",
                lhr_v25_witness_cid="",
                deep_substrate_hybrid_v18_witness_cid="",
                substrate_adapter_v18_matrix_cid="",
                masc_v9_witness_cid="",
                team_consensus_controller_v8_witness_cid="",
                replacement_pressure_falsifier_witness_cid="",
                hosted_router_v6_witness_cid="",
                hosted_logprob_router_v6_witness_cid="",
                hosted_cache_planner_v6_witness_cid="",
                hosted_real_substrate_boundary_v6_cid="",
                hosted_wall_v6_report_cid="",
                handoff_coordinator_v5_witness_cid="",
                handoff_envelope_v5_chain_cid="",
                provider_filter_v5_report_cid="",
                eighteen_way_used=False,
                substrate_v18_used=False,
                masc_v9_v18_beats_v17_rate=0.0,
                masc_v9_tsc_v18_beats_tsc_v17_rate=0.0,
                masc_v9_team_success_per_visible_token=0.0,
                hosted_router_v6_chosen="",
                replacement_repair_trajectory_cid="",
            )
        # Plane B — substrate V18 forward with full event chain.
        token_ids = (
            list(ids) if ids is not None
            else tokenize_bytes_v18(str(text), max_len=16))
        trace, cache = forward_tiny_substrate_v18(
            p.substrate_v18, token_ids,
            visible_token_budget=128.0,
            baseline_token_cost=512.0,
            restart_pressure=0.6,
            rejoin_pressure=0.6,
            replacement_pressure=0.6,
            contradiction_pressure=0.5)
        # Record restart + delay events on the inner V16 cache so the
        # V17 / V18 trajectory CIDs aggregate properly.
        from .tiny_substrate_v16 import (
            record_delay_window_v16, record_restart_event_v16,
        )
        from .tiny_substrate_v17 import (
            record_branch_pressure_window_v17,
            record_rejoin_event_v17,
        )
        record_restart_event_v16(
            cache.v17_cache.v16_cache, turn=1,
            restart_kind="agent_restart",
            role="planner")
        record_delay_window_v16(
            cache.v17_cache.v16_cache, restart_turn=1,
            repair_turn=4, delay_turns=3, role="planner")
        record_rejoin_event_v17(
            cache.v17_cache, turn=2,
            rejoin_kind="branch_rejoin",
            branch_id="main", role="planner")
        record_branch_pressure_window_v17(
            cache.v17_cache, restart_turn=1, rejoin_turn=5,
            rejoin_lag_turns=4, branch_id="main",
            role="planner")
        record_contradiction_event_v18(
            cache, turn=2,
            contradiction_kind="fact_contradiction",
            role="planner", branch_id="main")
        record_replacement_event_v18(
            cache, turn=3,
            replacement_kind="agent_replacement",
            role="planner", new_role="planner_fresh")
        record_replacement_window_v18(
            cache, contradiction_turn=2,
            replacement_turn=3, rejoin_turn=8,
            replacement_lag_turns=5, role="planner",
            branch_id="main")
        # Re-run forward to produce the replacement-repair CID
        # over the recorded events.
        trace, cache = forward_tiny_substrate_v18(
            p.substrate_v18, token_ids,
            v18_kv_cache=cache,
            visible_token_budget=128.0,
            baseline_token_cost=512.0,
            restart_pressure=0.6,
            rejoin_pressure=0.6,
            replacement_pressure=0.6,
            contradiction_pressure=0.5)
        sub_witness = emit_tiny_substrate_v18_forward_witness(
            trace, cache)
        # KV V18 witnesses.
        rep_falsifier = (
            probe_kv_bridge_v18_replacement_pressure_falsifier(
                replacement_pressure_flag=1))
        rrf = compute_replacement_repair_fingerprint_v18(
            role="planner",
            repair_trajectory_cid=str(
                cache.v17_cache.v16_cache.v15_cache
                .repair_trajectory_cid),
            delayed_repair_trajectory_cid=str(
                cache.v17_cache.v16_cache
                .delayed_repair_trajectory_cid),
            restart_repair_trajectory_cid=str(
                cache.v17_cache.restart_repair_trajectory_cid),
            replacement_repair_trajectory_cid=str(
                cache.replacement_repair_trajectory_cid),
            dominant_repair_label=1,
            restart_count=int(len(
                cache.v17_cache.v16_cache.restart_events)),
            rejoin_count=int(len(
                cache.v17_cache.rejoin_events)),
            replacement_count=int(len(
                cache.replacement_events)),
            contradiction_count=int(len(
                cache.contradiction_events)),
            visible_token_budget=128.0,
            baseline_cost=512.0,
            delay_turns=3,
            rejoin_lag_turns=4,
            replacement_lag_turns=5)
        kv_witness = emit_kv_bridge_v18_witness(
            projection=p.kv_bridge_v18,
            replacement_pressure_falsifier=rep_falsifier,
            replacement_repair_fingerprint=rrf)
        cache_witness = emit_cache_controller_v16_witness(
            controller=p.cache_controller_v16)
        replay_witness = emit_replay_controller_v14_witness(
            p.replay_controller_v14)
        persist_chain = (
            PersistentLatentStateV25Chain.empty())
        persist_witness = emit_persistent_v25_witness(
            persist_chain)
        # MLSC V21 — wrap a trivial V20 capsule up the chain.
        from .mergeable_latent_capsule_v3 import (
            make_root_capsule_v3)
        from .mergeable_latent_capsule_v4 import wrap_v3_as_v4
        from .mergeable_latent_capsule_v5 import wrap_v4_as_v5
        from .mergeable_latent_capsule_v6 import wrap_v5_as_v6
        from .mergeable_latent_capsule_v7 import wrap_v6_as_v7
        from .mergeable_latent_capsule_v8 import wrap_v7_as_v8
        from .mergeable_latent_capsule_v9 import wrap_v8_as_v9
        from .mergeable_latent_capsule_v10 import (
            wrap_v9_as_v10)
        from .mergeable_latent_capsule_v11 import (
            wrap_v10_as_v11)
        from .mergeable_latent_capsule_v12 import (
            wrap_v11_as_v12)
        from .mergeable_latent_capsule_v13 import (
            wrap_v12_as_v13)
        from .mergeable_latent_capsule_v14 import (
            wrap_v13_as_v14)
        from .mergeable_latent_capsule_v15 import (
            wrap_v14_as_v15)
        from .mergeable_latent_capsule_v16 import (
            wrap_v15_as_v16)
        from .mergeable_latent_capsule_v17 import (
            wrap_v16_as_v17)
        from .mergeable_latent_capsule_v18 import (
            wrap_v17_as_v18)
        from .mergeable_latent_capsule_v19 import (
            wrap_v18_as_v19)
        from .mergeable_latent_capsule_v20 import (
            wrap_v19_as_v20)
        v3 = make_root_capsule_v3(
            branch_id="w73_smoke",
            payload=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            fact_tags=("w73",), confidence=0.9, trust=0.9,
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
        v18 = wrap_v17_as_v18(v17)
        v19 = wrap_v18_as_v19(v18)
        v20 = wrap_v19_as_v20(
            v19,
            restart_repair_trajectory_chain=(
                str(cache.v17_cache
                    .restart_repair_trajectory_cid),),
            rejoin_pressure_chain=(
                f"rj_{int(len(cache.v17_cache.rejoin_events))}",
            ),
        )
        v21 = wrap_v20_as_v21(
            v20,
            replacement_repair_trajectory_chain=(
                str(cache.replacement_repair_trajectory_cid),),
            contradiction_chain=(
                f"ct_{int(len(cache.contradiction_events))}",),
        )
        mlsc_witness = emit_mlsc_v21_witness(v21)
        consensus_witness = emit_consensus_v19_witness(
            p.consensus_v19)
        lhr_witness = emit_lhr_v25_witness(
            p.lhr_v25, carrier=[0.1] * 6, k=16,
            partial_contradiction_indicator=[0.5] * 8,
            multi_branch_rejoin_indicator=[0.6] * 8,
            repair_dominance_indicator=[0.7] * 7,
            restart_indicator=[0.5] * 8,
            rejoin_indicator=[0.6] * 8,
            replacement_indicator=[0.7] * 8)
        # Deep substrate hybrid V18 — fold the V17 witness as a
        # pre-condition.
        v17_witness = DeepSubstrateHybridV17ForwardWitness(
            schema="coordpy.deep_substrate_hybrid_v17.v1",
            hybrid_cid="",
            inner_v16_witness_cid="",
            seventeen_way=True,
            cache_controller_v15_fired=True,
            replay_controller_v13_fired=True,
            restart_repair_trajectory_active=True,
            delayed_rejoin_after_restart_active=True,
            team_consensus_controller_v7_active=True,
            restart_repair_trajectory_cid=str(
                cache.v17_cache.restart_repair_trajectory_cid),
            delayed_rejoin_after_restart_l1=int(
                sub_witness.replacement_after_ctr_l1 + 1),
            rejoin_pressure_gate_mean=float(
                trace.v17_trace.rejoin_pressure_gate_per_layer
                .mean()),
        )
        deep_v18_witness = deep_substrate_hybrid_v18_forward(
            hybrid=p.deep_substrate_hybrid_v18,
            v17_witness=v17_witness,
            cache_controller_v16=p.cache_controller_v16,
            replay_controller_v14=p.replay_controller_v14,
            replacement_repair_trajectory_cid=str(
                cache.replacement_repair_trajectory_cid),
            replacement_after_ctr_l1=int(
                sub_witness.replacement_after_ctr_l1),
            replacement_pressure_gate_mean=float(
                trace.replacement_pressure_gate_per_layer
                .mean()),
            n_team_consensus_v8_invocations=1)
        adapter_matrix = probe_all_v18_adapters()
        # MASC V9 — run a batch for the envelope (all regimes).
        per_regime_aggs = {}
        for regime in W73_MASC_V9_REGIMES:
            _, agg = p.multi_agent_coordinator_v9.run_batch(
                seeds=list(range(int(p.masc_v9_n_seeds))),
                regime=regime)
            per_regime_aggs[regime] = agg
        masc_witness = (
            emit_multi_agent_substrate_coordinator_v9_witness(
                coordinator=p.multi_agent_coordinator_v9,
                per_regime_aggregate=per_regime_aggs))
        # TCC V8 — fire each new arbiter so the witness counts > 0.
        tcc_v8 = p.team_consensus_controller_v8
        tcc_v8.decide_v8(
            regime=(
                "replacement_after_contradiction_then_rejoin"),
            agent_guesses=[1.0, -1.0, 0.5, 0.2],
            agent_confidences=[0.8, 0.6, 0.7, 0.7],
            substrate_replay_trust=0.7,
            visible_token_budget_ratio=0.3,
            dominant_repair_label=1,
            agent_repair_labels=[1, 1, 0, 0],
            branch_assignments=[0, 1, 2, 0],
            restart_pressure=0.7,
            agent_restart_recovery_flags=[1, 0, 0, 0],
            delayed_repair_trajectory_cid=str(
                cache.v17_cache.v16_cache
                .delayed_repair_trajectory_cid),
            delay_turns=3,
            agent_delay_absorption_scores=[
                0.9, 0.5, 0.4, 0.3],
            rejoin_pressure=0.8,
            agent_rejoin_recovery_flags=[1, 1, 0, 0],
            restart_repair_trajectory_cid=str(
                cache.v17_cache
                .restart_repair_trajectory_cid),
            rejoin_lag_turns=4,
            agent_rejoin_absorption_scores=[
                0.95, 0.7, 0.5, 0.4],
            replacement_pressure=0.8,
            agent_replacement_recovery_flags=[1, 1, 0, 0],
            replacement_repair_trajectory_cid=str(
                cache.replacement_repair_trajectory_cid),
            replacement_lag_turns=5,
            agent_replacement_absorption_scores=[
                0.95, 0.6, 0.5, 0.4])
        tcc_v8.decide_v8(
            regime="baseline",
            agent_guesses=[0.5, 0.5, 0.4, 0.5],
            agent_confidences=[0.8, 0.6, 0.7, 0.7],
            substrate_replay_trust=0.7,
            visible_token_budget_ratio=0.9,
            dominant_repair_label=1,
            agent_repair_labels=[1, 1, 0, 0],
            replacement_pressure=0.7,
            agent_replacement_recovery_flags=[1, 0, 1, 0])
        tcc_witness = emit_team_consensus_controller_v8_witness(
            tcc_v8)
        # Plane A V6 — hosted.
        planned, _ = (
            p.hosted_cache_planner_v6
            .plan_per_role_four_layer_rotated(
                shared_prefix_text=(
                    "W73 team shared prefix " * 14),
                per_role_blocks={
                    "plan": ["t0", "t1"],
                    "research": ["r0", "r1"],
                    "write": ["w0", "w1"],
                    "review": ["v0", "v1"],
                }))
        # Router V6 — at least one decision so witness is non-empty.
        req_v6 = HostedRoutingRequestV6(
            inner_v5=HostedRoutingRequestV5(
                inner_v4=HostedRoutingRequestV4(
                    inner_v3=HostedRoutingRequestV3(
                        inner_v2=HostedRoutingRequestV2(
                            inner_v1=HostedRoutingRequest(
                                request_cid="w73-router-turn",
                                input_tokens=1000,
                                expected_output_tokens=300,
                                require_logprobs=True,
                                require_prefix_cache=True,
                                data_policy_required="no_log",
                                max_latency_ms=2000.0,
                                max_cost_usd=50.0),
                            weight_cost=1.0, weight_latency=0.5,
                            weight_success=0.3),
                        visible_token_budget=128,
                        baseline_token_cost=512,
                        repair_dominance_label=1),
                    restart_pressure=0.7,
                    weight_restart_pressure=0.6,
                    weight_delayed_repair_match=0.4),
                rejoin_pressure=0.7,
                weight_rejoin_pressure=0.6,
                weight_delayed_rejoin_match=0.4),
            replacement_pressure=0.7,
            weight_replacement_pressure=0.6,
            weight_replacement_after_ctr_match=0.4)
        router_dec = p.hosted_router_v6.decide_v6(req_v6)
        router_v6_witness = (
            emit_hosted_router_controller_v6_witness(
                p.hosted_router_v6))
        logprob_v6_witness = (
            emit_hosted_logprob_router_v6_witness(
                p.hosted_logprob_router_v6))
        cache_planner_v6_witness = (
            emit_hosted_cache_aware_planner_v6_witness(
                p.hosted_cache_planner_v6))
        boundary_v6 = p.hosted_real_substrate_boundary_v6
        wall_v6_report = build_wall_report_v6(
            boundary=boundary_v6)
        # Provider filter V5 — run once to seal a report CID.
        _, filter_report = filter_hosted_registry_v5(
            p.hosted_registry, p.hosted_provider_filter_v5,
            provider_restart_noise={
                "openrouter_paid": 0.5,
                "openai_paid": 0.1},
            provider_rejoin_noise={
                "openrouter_paid": 0.4,
                "openai_paid": 0.1},
            provider_replacement_noise={
                "openrouter_paid": 0.35,
                "openai_paid": 0.05})
        filter_report_cid = _sha256_hex({
            "kind": "w73_provider_filter_v5_report",
            "report": dict(filter_report),
        })
        # Handoff coordinator V5 decisions.
        env_text_only = p.handoff_coordinator_v5.decide_v5(
            req_v5=HandoffRequestV5(
                inner_v4=HandoffRequestV4(
                    inner_v3=HandoffRequestV3(
                        inner_v2=HandoffRequestV2(
                            inner_v1=HandoffRequest(
                                request_cid="w73-turn-text",
                                needs_text_only=True,
                                needs_substrate_state_access=(
                                    False)),
                            visible_token_budget=128,
                            baseline_token_cost=512,
                            dominant_repair_label=0),
                        restart_pressure=0.0,
                        delayed_repair_trajectory_cid="",
                        delay_turns=0,
                        expected_substrate_trust=0.7),
                    rejoin_pressure=0.0,
                    restart_repair_trajectory_cid="",
                    rejoin_lag_turns=0,
                    expected_substrate_trust_v4=0.7),
                replacement_pressure=0.0,
                replacement_repair_trajectory_cid="",
                replacement_lag_turns=0,
                expected_substrate_trust_v5=0.7))
        env_replacement_promoted = (
            p.handoff_coordinator_v5.decide_v5(
                req_v5=HandoffRequestV5(
                    inner_v4=HandoffRequestV4(
                        inner_v3=HandoffRequestV3(
                            inner_v2=HandoffRequestV2(
                                inner_v1=HandoffRequest(
                                    request_cid=(
                                        "w73-turn-rep"),
                                    needs_text_only=True,
                                    needs_substrate_state_access=(
                                        False)),
                                visible_token_budget=128,
                                baseline_token_cost=512,
                                dominant_repair_label=0),
                            restart_pressure=0.0,
                            delayed_repair_trajectory_cid="",
                            delay_turns=0,
                            expected_substrate_trust=0.7),
                        rejoin_pressure=0.0,
                        restart_repair_trajectory_cid="",
                        rejoin_lag_turns=0,
                        expected_substrate_trust_v4=0.7),
                    replacement_pressure=0.8,
                    replacement_repair_trajectory_cid=str(
                        cache
                        .replacement_repair_trajectory_cid),
                    replacement_lag_turns=5,
                    expected_substrate_trust_v5=0.7)))
        env_rep_fallback = (
            p.handoff_coordinator_v5.decide_v5(
                req_v5=HandoffRequestV5(
                    inner_v4=HandoffRequestV4(
                        inner_v3=HandoffRequestV3(
                            inner_v2=HandoffRequestV2(
                                inner_v1=HandoffRequest(
                                    request_cid="w73-turn-rep-f",
                                    needs_text_only=True,
                                    needs_substrate_state_access=(
                                        False)),
                                visible_token_budget=128,
                                baseline_token_cost=512,
                                dominant_repair_label=0),
                            restart_pressure=0.0,
                            delayed_repair_trajectory_cid="",
                            delay_turns=0,
                            expected_substrate_trust=0.7),
                        rejoin_pressure=0.0,
                        restart_repair_trajectory_cid="",
                        rejoin_lag_turns=0,
                        expected_substrate_trust_v4=0.7),
                    replacement_pressure=0.0,
                    replacement_repair_trajectory_cid=str(
                        cache
                        .replacement_repair_trajectory_cid),
                    replacement_lag_turns=5,
                    expected_substrate_trust_v5=0.7)))
        env_substrate_only = (
            p.handoff_coordinator_v5.decide_v5(
                req_v5=HandoffRequestV5(
                    inner_v4=HandoffRequestV4(
                        inner_v3=HandoffRequestV3(
                            inner_v2=HandoffRequestV2(
                                inner_v1=HandoffRequest(
                                    request_cid=(
                                        "w73-turn-substrate"),
                                    needs_text_only=False,
                                    needs_substrate_state_access=(
                                        True)),
                                visible_token_budget=128,
                                baseline_token_cost=512,
                                dominant_repair_label=1),
                            restart_pressure=0.3,
                            expected_substrate_trust=0.7),
                        rejoin_pressure=0.0,
                        expected_substrate_trust_v4=0.7),
                    replacement_pressure=0.0,
                    expected_substrate_trust_v5=0.7)))
        handoff_v5_witness = (
            emit_hosted_real_handoff_coordinator_v5_witness(
                p.handoff_coordinator_v5))
        handoff_envelope_chain_cid = _sha256_hex({
            "kind": "w73_handoff_envelope_v5_chain",
            "envelopes": [
                env_text_only.cid(),
                env_replacement_promoted.cid(),
                env_rep_fallback.cid(),
                env_substrate_only.cid(),
            ],
        })
        baseline_agg = per_regime_aggs.get("baseline")
        v18_beats = (
            float(baseline_agg.v18_beats_v17_rate)
            if baseline_agg is not None else 0.0)
        tsc_v18_beats = (
            float(baseline_agg.tsc_v18_beats_tsc_v17_rate)
            if baseline_agg is not None else 0.0)
        ts_per_vt = (
            float(
                baseline_agg.team_success_per_visible_token_v18)
            if baseline_agg is not None else 0.0)
        return W73HandoffEnvelope(
            schema=W73_SCHEMA_VERSION,
            w72_outer_cid=str(w72_outer_cid),
            w73_params_cid=str(p.cid()),
            substrate_v18_witness_cid=str(sub_witness.cid()),
            kv_bridge_v18_witness_cid=str(kv_witness.cid()),
            cache_controller_v16_witness_cid=str(
                cache_witness.cid()),
            replay_controller_v14_witness_cid=str(
                replay_witness.cid()),
            persistent_v25_witness_cid=str(
                persist_witness.cid()),
            mlsc_v21_witness_cid=str(mlsc_witness.cid()),
            consensus_v19_witness_cid=str(
                consensus_witness.cid()),
            lhr_v25_witness_cid=str(lhr_witness.cid()),
            deep_substrate_hybrid_v18_witness_cid=str(
                deep_v18_witness.cid()),
            substrate_adapter_v18_matrix_cid=str(
                adapter_matrix.cid()),
            masc_v9_witness_cid=str(masc_witness.cid()),
            team_consensus_controller_v8_witness_cid=str(
                tcc_witness.cid()),
            replacement_pressure_falsifier_witness_cid=str(
                rep_falsifier.cid()),
            hosted_router_v6_witness_cid=str(
                router_v6_witness.cid()),
            hosted_logprob_router_v6_witness_cid=str(
                logprob_v6_witness.cid()),
            hosted_cache_planner_v6_witness_cid=str(
                cache_planner_v6_witness.cid()),
            hosted_real_substrate_boundary_v6_cid=str(
                boundary_v6.cid()),
            hosted_wall_v6_report_cid=str(
                wall_v6_report.cid()),
            handoff_coordinator_v5_witness_cid=str(
                handoff_v5_witness.cid()),
            handoff_envelope_v5_chain_cid=str(
                handoff_envelope_chain_cid),
            provider_filter_v5_report_cid=str(
                filter_report_cid),
            eighteen_way_used=bool(
                deep_v18_witness.eighteen_way),
            substrate_v18_used=True,
            masc_v9_v18_beats_v17_rate=float(v18_beats),
            masc_v9_tsc_v18_beats_tsc_v17_rate=float(
                tsc_v18_beats),
            masc_v9_team_success_per_visible_token=float(
                ts_per_vt),
            hosted_router_v6_chosen=str(
                router_dec.chosen_provider or ""),
            replacement_repair_trajectory_cid=str(
                cache.replacement_repair_trajectory_cid),
        )


def build_default_w73_team(*, seed: int = 73000) -> W73Team:
    return W73Team(params=W73Params.build_default(seed=int(seed)))


__all__ = [
    "W73_SCHEMA_VERSION",
    "W73_FAILURE_MODES",
    "W73Params",
    "W73HandoffEnvelope",
    "verify_w73_handoff",
    "W73Team",
    "build_default_w73_team",
]
