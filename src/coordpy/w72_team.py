"""W72 — Stronger Delayed-Rejoin-After-Restart / Restart-Repair-
Trajectory Two-Plane Multi-Agent Substrate team.

The ``W72Team`` orchestrator strictly wraps the ``W71Team`` and
adds the W72 mechanism modules organised across two planes plus
the new **rejoin-aware Plane A↔B handoff coordinator V4**:

**Plane B — Real substrate (in-repo, V17 stack):**

* M1  ``tiny_substrate_v17``           (19-layer, 3 new V17 axes)
* M2  ``kv_bridge_v17``                (13-target ridge + 100-dim
                                        restart-repair fingerprint +
                                        rejoin-pressure falsifier)
* M3  ``cache_controller_v15``         (12-objective ridge + per-
                                        role 13-dim rejoin-pressure
                                        head)
* M4  ``replay_controller_v13``        (20 regimes + 10-label
                                        rejoin-aware routing head)
* M5  ``deep_substrate_hybrid_v17``    (17-way bidirectional loop)
* M6  ``substrate_adapter_v17``        (substrate_v17_full tier)
* M7  ``persistent_latent_v24``        (23 layers, 21st carrier,
                                        max_chain_walk_depth=524288)
* M8  ``long_horizon_retention_v24``   (23 heads, max_k=704)
* M9  ``mergeable_latent_capsule_v20`` (restart-repair-trajectory
                                        chain + rejoin-pressure
                                        chain)
* M10 ``consensus_fallback_controller_v18`` (30-stage chain)
* M11 ``multi_agent_substrate_coordinator_v8`` (18-policy, 12-regime
                                                MASC V8)
* M12 ``team_consensus_controller_v7`` (rejoin-pressure + delayed-
                                        rejoin arbiters)

**Plane A — Hosted control plane V5 (honest, no substrate):**

* H1  ``hosted_router_controller_v5``  (rejoin-pressure weighting
                                        + delayed-rejoin match)
* H2  ``hosted_logprob_router_v5``     (rejoin-aware abstain floor
                                        + per-budget+restart+rejoin
                                        tiebreak)
* H3  ``hosted_cache_aware_planner_v5``(three-layer rotated prefix +
                                        ≥ 80 % savings 12×8 hit=1)
* H4  ``hosted_cost_planner_v5``       (cost-per-rejoin-success-
                                        under-budget +
                                        abstain-when-rejoin-
                                        pressure-violated)
* H5  ``hosted_real_substrate_boundary_v5`` (wall V5, 28 blocked
                                             axes)
* H6  ``hosted_real_handoff_coordinator_v4`` (the **new rejoin-
                                              aware Plane A↔B
                                              bridge** — V4
                                              envelopes + delayed-
                                              rejoin falsifier +
                                              cross-plane savings)
* H7  ``hosted_provider_filter_v4``    (rejoin-aware provider
                                        filter)

Per-turn it emits 19 W72 module witness CIDs (12 Plane B + 7 Plane
A V5) and a V4 handoff envelope CID, sealing them into a
``W72HandoffEnvelope`` whose ``w71_outer_cid`` carries forward the
W71 envelope byte-for-byte.

Honest scope (W72)
------------------

* Plane A V5 operates at the hosted text/logprob/prefix-cache
  surface. It does NOT pierce hidden state / KV / attention.
  ``W72-L-HOSTED-V5-NO-SUBSTRATE-CAP``.
* Plane B is the in-repo V17 NumPy runtime. We do NOT bridge to
  third-party hosted models.
  ``W72-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
* W72 fits closed-form ridge parameters in three new places on top
  of W71's 61: cache V15 twelve-objective; cache V15 per-role
  rejoin-pressure; replay V13 rejoin-aware routing head; KV V17
  thirteen-target. Total **64 closed-form ridge solves** across
  W61..W72. No autograd, no SGD, no GPU.
* Trivial passthrough preserved: when ``W72Params.build_trivial()``
  is used the W72 envelope's internal ``w71_outer_cid`` carries
  the supplied W71 outer CID exactly.
* The handoff coordinator V4 preserves the wall: a content-
  addressed V4 envelope says which plane handled each turn under
  the rejoin-aware score; it does NOT cross the substrate
  boundary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .cache_controller_v15 import (
    CacheControllerV15, emit_cache_controller_v15_witness,
    fit_per_role_rejoin_pressure_head_v15,
    fit_twelve_objective_ridge_v15,
)
from .consensus_fallback_controller_v18 import (
    ConsensusFallbackControllerV18,
    emit_consensus_v18_witness,
)
from .deep_substrate_hybrid_v16 import (
    DeepSubstrateHybridV16ForwardWitness,
)
from .deep_substrate_hybrid_v17 import (
    DeepSubstrateHybridV17,
    deep_substrate_hybrid_v17_forward,
)
from .hosted_cache_aware_planner_v5 import (
    HostedCacheAwarePlannerV5,
    emit_hosted_cache_aware_planner_v5_witness,
)
from .hosted_cost_planner_v5 import HostedCostPlanSpecV5
from .hosted_logprob_router_v5 import (
    HostedLogprobRouterV5,
    emit_hosted_logprob_router_v5_witness,
)
from .hosted_provider_filter_v4 import (
    HostedProviderFilterSpecV4,
    filter_hosted_registry_v4,
)
from .hosted_provider_filter_v3 import HostedProviderFilterSpecV3
from .hosted_provider_filter_v2 import HostedProviderFilterSpecV2
from .hosted_provider_filter import HostedProviderFilterSpec
from .hosted_real_handoff_coordinator_v4 import (
    HandoffRequestV4, HostedRealHandoffCoordinatorV4,
    emit_hosted_real_handoff_coordinator_v4_witness,
    hosted_real_handoff_v4_rejoin_aware_savings,
)
from .hosted_real_handoff_coordinator_v3 import HandoffRequestV3
from .hosted_real_handoff_coordinator_v2 import HandoffRequestV2
from .hosted_real_handoff_coordinator import HandoffRequest
from .hosted_real_substrate_boundary_v5 import (
    HostedRealSubstrateBoundaryV5,
    build_default_hosted_real_substrate_boundary_v5,
    build_wall_report_v5,
    probe_hosted_real_substrate_boundary_v5_falsifier,
)
from .hosted_router_controller import (
    HostedProviderRegistry, HostedRoutingRequest,
    default_hosted_registry,
)
from .hosted_router_controller_v2 import HostedRoutingRequestV2
from .hosted_router_controller_v3 import HostedRoutingRequestV3
from .hosted_router_controller_v4 import HostedRoutingRequestV4
from .hosted_router_controller_v5 import (
    HostedRouterControllerV5, HostedRoutingRequestV5,
    emit_hosted_router_controller_v5_witness,
)
from .kv_bridge_v16 import KVBridgeV16Projection
from .kv_bridge_v17 import (
    KVBridgeV17Projection,
    compute_restart_repair_fingerprint_v17,
    emit_kv_bridge_v17_witness,
    probe_kv_bridge_v17_rejoin_pressure_falsifier,
)
from .long_horizon_retention_v24 import (
    LongHorizonReconstructionV24Head,
    emit_lhr_v24_witness,
)
from .mergeable_latent_capsule_v20 import (
    MergeOperatorV20, emit_mlsc_v20_witness, wrap_v19_as_v20,
)
from .multi_agent_substrate_coordinator_v8 import (
    MultiAgentSubstrateCoordinatorV8,
    W72_MASC_V8_REGIMES,
    emit_multi_agent_substrate_coordinator_v8_witness,
)
from .persistent_latent_v24 import (
    PersistentLatentStateV24Chain,
    emit_persistent_v24_witness,
)
from .replay_controller_v13 import (
    ReplayControllerV13,
    W72_REJOIN_AWARE_ROUTING_LABELS,
    W72_REPLAY_REGIMES_V13,
    emit_replay_controller_v13_witness,
    fit_replay_controller_v13_per_role,
    fit_replay_v13_rejoin_aware_routing_head,
)
from .replay_controller import ReplayCandidate
from .substrate_adapter_v17 import (
    W72_SUBSTRATE_TIER_SUBSTRATE_V17_FULL,
    probe_all_v17_adapters,
)
from .team_consensus_controller_v7 import (
    TeamConsensusControllerV7,
    emit_team_consensus_controller_v7_witness,
)
from .tiny_substrate_v17 import (
    TinyV17SubstrateParams,
    build_default_tiny_substrate_v17,
    emit_tiny_substrate_v17_forward_witness,
    forward_tiny_substrate_v17,
    record_branch_pressure_window_v17,
    record_rejoin_event_v17,
    tokenize_bytes_v17,
)
from .w71_team import (
    W71HandoffEnvelope, W71Params, W71Team,
)


W72_SCHEMA_VERSION: str = "coordpy.w72_team.v1"

W72_FAILURE_MODES: tuple[str, ...] = (
    "w72_outer_envelope_schema_mismatch",
    "w72_outer_envelope_w71_outer_cid_drift",
    "w72_outer_envelope_w72_params_cid_drift",
    "w72_outer_envelope_witness_cid_drift",
    "w72_substrate_v17_n_layers_off",
    "w72_substrate_v17_restart_repair_trajectory_cid_off",
    "w72_substrate_v17_delayed_rejoin_per_layer_shape_off",
    "w72_substrate_v17_rejoin_pressure_gate_shape_off",
    "w72_kv_bridge_v17_n_targets_off",
    "w72_kv_bridge_v17_rejoin_pressure_falsifier_off",
    "w72_cache_v15_twelve_objective_off",
    "w72_replay_v13_regime_count_off",
    "w72_replay_v13_rejoin_aware_routing_count_off",
    "w72_consensus_v18_stage_count_off",
    "w72_lhr_v24_max_k_off",
    "w72_lhr_v24_n_heads_off",
    "w72_persistent_v24_n_layers_off",
    "w72_substrate_adapter_v17_tier_off",
    "w72_masc_v8_v17_beats_v16_rate_under_threshold",
    "w72_masc_v8_tsc_v17_beats_tsc_v16_rate_under_threshold",
    "w72_masc_v8_compound_regime_inferior_to_baseline",
    "w72_hosted_router_v5_decision_not_deterministic",
    "w72_hosted_logprob_v5_abstain_floor_off",
    "w72_hosted_cache_aware_v5_savings_below_80_percent",
    "w72_hosted_cost_planner_v5_no_eligible",
    "w72_hosted_real_substrate_boundary_v5_blocked_axis_satisfied",
    "w72_seventeen_way_loop_not_observed",
    "w72_handoff_coordinator_v4_inconsistent",
    "w72_handoff_v4_cross_plane_savings_below_78_percent",
    "w72_team_consensus_v7_no_decisions",
    "w72_handoff_v4_rejoin_alignment_off",
    "w72_handoff_envelope_v4_chain_cid_drift",
    "w72_inner_v71_envelope_invariant_off",
    "w72_handoff_v4_delayed_rejoin_fallback_off",
    "w72_hosted_boundary_v5_blocked_axes_below_28",
    "w72_v17_substrate_self_checksum_cid_off",
    "w72_restart_repair_trajectory_cid_drift",
    "w72_mlsc_v20_restart_repair_trajectory_chain_off",
    "w72_v8_team_success_per_visible_token_below_floor",
    "w72_v8_visible_tokens_savings_below_65_percent",
    "w72_v8_compound_regime_v17_beats_v16_below_threshold",
    "w72_substrate_v17_restart_repair_trajectory_chain_synthetic",
    "w72_inner_v17_falsifier_kind_off",
    "w72_handoff_v4_envelope_rj_alignment_off",
    "w72_hosted_router_v5_per_routing_cid_off",
    "w72_consensus_v18_rejoin_pressure_arbiter_off",
    "w72_consensus_v18_delayed_rejoin_arbiter_off",
    "w72_tcc_v7_rejoin_pressure_arbiter_off",
    "w72_tcc_v7_delayed_rejoin_arbiter_off",
    "w72_cache_v15_per_role_rejoin_pressure_head_off",
    "w72_kv_bridge_v17_restart_repair_fingerprint_off",
    "w72_substrate_v17_rejoin_events_off",
    "w72_provider_filter_v4_pressure_drop_off",
    "w72_handoff_v4_dr_alignment_off",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class W72Params:
    substrate_v17: TinyV17SubstrateParams | None
    kv_bridge_v17: KVBridgeV17Projection | None
    cache_controller_v15: CacheControllerV15 | None
    replay_controller_v13: ReplayControllerV13 | None
    consensus_v18: ConsensusFallbackControllerV18 | None
    lhr_v24: LongHorizonReconstructionV24Head | None
    deep_substrate_hybrid_v17: DeepSubstrateHybridV17 | None
    mlsc_v20_operator: MergeOperatorV20 | None
    multi_agent_coordinator_v8: (
        MultiAgentSubstrateCoordinatorV8 | None)
    team_consensus_controller_v7: (
        TeamConsensusControllerV7 | None)
    hosted_registry: HostedProviderRegistry | None
    hosted_router_v5: HostedRouterControllerV5 | None
    hosted_logprob_router_v5: HostedLogprobRouterV5 | None
    hosted_cache_planner_v5: HostedCacheAwarePlannerV5 | None
    hosted_real_substrate_boundary_v5: (
        HostedRealSubstrateBoundaryV5 | None)
    handoff_coordinator_v4: (
        HostedRealHandoffCoordinatorV4 | None)
    hosted_provider_filter_v4: (
        HostedProviderFilterSpecV4 | None)
    w71_params: W71Params | None
    enabled: bool = True
    masc_v8_n_seeds: int = 8

    @classmethod
    def build_trivial(cls) -> "W72Params":
        return cls(
            substrate_v17=None,
            kv_bridge_v17=None,
            cache_controller_v15=None,
            replay_controller_v13=None,
            consensus_v18=None, lhr_v24=None,
            deep_substrate_hybrid_v17=None,
            mlsc_v20_operator=None,
            multi_agent_coordinator_v8=None,
            team_consensus_controller_v7=None,
            hosted_registry=None,
            hosted_router_v5=None,
            hosted_logprob_router_v5=None,
            hosted_cache_planner_v5=None,
            hosted_real_substrate_boundary_v5=None,
            handoff_coordinator_v4=None,
            hosted_provider_filter_v4=None,
            w71_params=None,
            enabled=False,
        )

    @classmethod
    def build_default(
            cls, *, seed: int = 72000,
    ) -> "W72Params":
        sub_v17 = build_default_tiny_substrate_v17(
            seed=int(seed) + 1)
        # KV V17 projection chain.
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
        cfg = (
            sub_v17.config.v16.v15.v14.v13.v12.v11.v10.v9)
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
        cc15 = CacheControllerV15.init(
            fit_seed=int(seed) + 32)
        import numpy as _np
        rng = _np.random.default_rng(int(seed) + 33)
        X = rng.standard_normal((10, 4))
        cc15, _ = fit_twelve_objective_ridge_v15(
            controller=cc15, train_features=X.tolist(),
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
                + X[:, 3] * 0.3).tolist())
        X13 = rng.standard_normal((10, 13))
        cc15, _ = fit_per_role_rejoin_pressure_head_v15(
            controller=cc15, role="planner",
            train_features=X13.tolist(),
            target_rejoin_priorities=(
                X13[:, 0] * 0.3
                + X13[:, 12] * 0.4).tolist())
        # Replay V13.
        rcv13 = ReplayControllerV13.init()
        v13_cands = {
            r: [ReplayCandidate(
                100, 1000, 50, 0.1, 0.0, 0.3,
                True, True, 0)]
            for r in W72_REPLAY_REGIMES_V13}
        v13_decs = {
            r: ["choose_reuse"]
            for r in W72_REPLAY_REGIMES_V13}
        rcv13, _ = fit_replay_controller_v13_per_role(
            controller=rcv13, role="planner",
            train_candidates_per_regime=v13_cands,
            train_decisions_per_regime=v13_decs)
        X_team = rng.standard_normal((50, 10))
        labs: list[str] = []
        for i in range(50):
            lab_idx = (
                i % len(W72_REJOIN_AWARE_ROUTING_LABELS))
            labs.append(
                W72_REJOIN_AWARE_ROUTING_LABELS[lab_idx])
        rcv13, _ = fit_replay_v13_rejoin_aware_routing_head(
            controller=rcv13,
            train_team_features=X_team.tolist(),
            train_routing_labels=labs)
        consensus_v18 = ConsensusFallbackControllerV18.init(
            k_required=2, cosine_floor=0.6,
            trust_threshold=0.5,
            multi_branch_rejoin_threshold=0.5,
            silent_corruption_threshold=0.5,
            repair_dominance_threshold=0.5,
            budget_primary_threshold=0.5,
            restart_aware_threshold=0.5,
            delayed_repair_threshold=0.5,
            rejoin_pressure_threshold=0.5,
            delayed_rejoin_threshold=0.5)
        lhr24 = LongHorizonReconstructionV24Head.init(
            seed=int(seed) + 40)
        deep_v17 = DeepSubstrateHybridV17()
        mlsc_v20_op = MergeOperatorV20()
        masc_v8 = MultiAgentSubstrateCoordinatorV8()
        tcc_v7 = TeamConsensusControllerV7()
        reg = default_hosted_registry()
        hosted_router_v5 = HostedRouterControllerV5.init(
            reg, {
                "openrouter_paid": 0.85,
                "openai_paid": 0.92,
            })
        hosted_logprob_router_v5 = HostedLogprobRouterV5()
        hosted_cache_planner_v5 = HostedCacheAwarePlannerV5()
        boundary_v5 = (
            build_default_hosted_real_substrate_boundary_v5())
        handoff_coord_v4 = HostedRealHandoffCoordinatorV4(
            boundary_v5=boundary_v5)
        # Provider filter V4 — pre-configured rejoin-aware spec.
        provider_filter_v4 = HostedProviderFilterSpecV4(
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
                "text_only": 0.35})
        # W71 inner params for envelope chaining.
        w71_params = W71Params.build_default(
            seed=int(seed) - 1000)
        return cls(
            substrate_v17=sub_v17,
            kv_bridge_v17=kv_b17,
            cache_controller_v15=cc15,
            replay_controller_v13=rcv13,
            consensus_v18=consensus_v18,
            lhr_v24=lhr24,
            deep_substrate_hybrid_v17=deep_v17,
            mlsc_v20_operator=mlsc_v20_op,
            multi_agent_coordinator_v8=masc_v8,
            team_consensus_controller_v7=tcc_v7,
            hosted_registry=reg,
            hosted_router_v5=hosted_router_v5,
            hosted_logprob_router_v5=hosted_logprob_router_v5,
            hosted_cache_planner_v5=hosted_cache_planner_v5,
            hosted_real_substrate_boundary_v5=boundary_v5,
            handoff_coordinator_v4=handoff_coord_v4,
            hosted_provider_filter_v4=provider_filter_v4,
            w71_params=w71_params,
            enabled=True,
            masc_v8_n_seeds=6,
        )

    def to_dict(self) -> dict[str, Any]:
        def _cid_or_empty(x: Any) -> str:
            return str(x.cid()) if x is not None else ""
        return {
            "schema": W72_SCHEMA_VERSION,
            "kind": "w72_params",
            "substrate_v17_cid": _cid_or_empty(
                self.substrate_v17),
            "kv_bridge_v17_cid": _cid_or_empty(
                self.kv_bridge_v17),
            "cache_controller_v15_cid": _cid_or_empty(
                self.cache_controller_v15),
            "replay_controller_v13_cid": _cid_or_empty(
                self.replay_controller_v13),
            "consensus_v18_cid": _cid_or_empty(
                self.consensus_v18),
            "lhr_v24_cid": _cid_or_empty(self.lhr_v24),
            "deep_substrate_hybrid_v17_cid": _cid_or_empty(
                self.deep_substrate_hybrid_v17),
            "mlsc_v20_operator_cid": _cid_or_empty(
                self.mlsc_v20_operator),
            "multi_agent_coordinator_v8_cid": _cid_or_empty(
                self.multi_agent_coordinator_v8),
            "team_consensus_controller_v7_cid": _cid_or_empty(
                self.team_consensus_controller_v7),
            "hosted_registry_cid": _cid_or_empty(
                self.hosted_registry),
            "hosted_router_v5_cid": _cid_or_empty(
                self.hosted_router_v5),
            "hosted_logprob_router_v5_cid": _cid_or_empty(
                self.hosted_logprob_router_v5),
            "hosted_cache_planner_v5_cid": _cid_or_empty(
                self.hosted_cache_planner_v5),
            "hosted_real_substrate_boundary_v5_cid":
                _cid_or_empty(
                    self.hosted_real_substrate_boundary_v5),
            "handoff_coordinator_v4_cid": _cid_or_empty(
                self.handoff_coordinator_v4),
            "hosted_provider_filter_v4_cid": _cid_or_empty(
                self.hosted_provider_filter_v4),
            "w71_params_cid": _cid_or_empty(self.w71_params),
            "enabled": bool(self.enabled),
            "masc_v8_n_seeds": int(self.masc_v8_n_seeds),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w72_params",
            "params": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class W72HandoffEnvelope:
    schema: str
    w71_outer_cid: str
    w72_params_cid: str
    substrate_v17_witness_cid: str
    kv_bridge_v17_witness_cid: str
    cache_controller_v15_witness_cid: str
    replay_controller_v13_witness_cid: str
    persistent_v24_witness_cid: str
    mlsc_v20_witness_cid: str
    consensus_v18_witness_cid: str
    lhr_v24_witness_cid: str
    deep_substrate_hybrid_v17_witness_cid: str
    substrate_adapter_v17_matrix_cid: str
    masc_v8_witness_cid: str
    team_consensus_controller_v7_witness_cid: str
    rejoin_pressure_falsifier_witness_cid: str
    hosted_router_v5_witness_cid: str
    hosted_logprob_router_v5_witness_cid: str
    hosted_cache_planner_v5_witness_cid: str
    hosted_real_substrate_boundary_v5_cid: str
    hosted_wall_v5_report_cid: str
    handoff_coordinator_v4_witness_cid: str
    handoff_envelope_v4_chain_cid: str
    provider_filter_v4_report_cid: str
    seventeen_way_used: bool
    substrate_v17_used: bool
    masc_v8_v17_beats_v16_rate: float
    masc_v8_tsc_v17_beats_tsc_v16_rate: float
    masc_v8_team_success_per_visible_token: float
    hosted_router_v5_chosen: str
    restart_repair_trajectory_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "w71_outer_cid": str(self.w71_outer_cid),
            "w72_params_cid": str(self.w72_params_cid),
            "substrate_v17_witness_cid": str(
                self.substrate_v17_witness_cid),
            "kv_bridge_v17_witness_cid": str(
                self.kv_bridge_v17_witness_cid),
            "cache_controller_v15_witness_cid": str(
                self.cache_controller_v15_witness_cid),
            "replay_controller_v13_witness_cid": str(
                self.replay_controller_v13_witness_cid),
            "persistent_v24_witness_cid": str(
                self.persistent_v24_witness_cid),
            "mlsc_v20_witness_cid": str(
                self.mlsc_v20_witness_cid),
            "consensus_v18_witness_cid": str(
                self.consensus_v18_witness_cid),
            "lhr_v24_witness_cid": str(
                self.lhr_v24_witness_cid),
            "deep_substrate_hybrid_v17_witness_cid": str(
                self.deep_substrate_hybrid_v17_witness_cid),
            "substrate_adapter_v17_matrix_cid": str(
                self.substrate_adapter_v17_matrix_cid),
            "masc_v8_witness_cid": str(
                self.masc_v8_witness_cid),
            "team_consensus_controller_v7_witness_cid": str(
                self.team_consensus_controller_v7_witness_cid),
            "rejoin_pressure_falsifier_witness_cid": str(
                self.rejoin_pressure_falsifier_witness_cid),
            "hosted_router_v5_witness_cid": str(
                self.hosted_router_v5_witness_cid),
            "hosted_logprob_router_v5_witness_cid": str(
                self.hosted_logprob_router_v5_witness_cid),
            "hosted_cache_planner_v5_witness_cid": str(
                self.hosted_cache_planner_v5_witness_cid),
            "hosted_real_substrate_boundary_v5_cid": str(
                self.hosted_real_substrate_boundary_v5_cid),
            "hosted_wall_v5_report_cid": str(
                self.hosted_wall_v5_report_cid),
            "handoff_coordinator_v4_witness_cid": str(
                self.handoff_coordinator_v4_witness_cid),
            "handoff_envelope_v4_chain_cid": str(
                self.handoff_envelope_v4_chain_cid),
            "provider_filter_v4_report_cid": str(
                self.provider_filter_v4_report_cid),
            "seventeen_way_used": bool(self.seventeen_way_used),
            "substrate_v17_used": bool(self.substrate_v17_used),
            "masc_v8_v17_beats_v16_rate": float(round(
                self.masc_v8_v17_beats_v16_rate, 12)),
            "masc_v8_tsc_v17_beats_tsc_v16_rate": float(round(
                self.masc_v8_tsc_v17_beats_tsc_v16_rate, 12)),
            "masc_v8_team_success_per_visible_token": float(
                round(
                    self
                    .masc_v8_team_success_per_visible_token,
                    12)),
            "hosted_router_v5_chosen": str(
                self.hosted_router_v5_chosen),
            "restart_repair_trajectory_cid": str(
                self.restart_repair_trajectory_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w72_handoff_envelope",
            "envelope": self.to_dict()})


def verify_w72_handoff(
        envelope: W72HandoffEnvelope,
        params: W72Params,
        w71_outer_cid: str,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if envelope.schema != W72_SCHEMA_VERSION:
        failures.append(
            "w72_outer_envelope_schema_mismatch")
    if envelope.w71_outer_cid != str(w71_outer_cid):
        failures.append(
            "w72_outer_envelope_w71_outer_cid_drift")
    if envelope.w72_params_cid != params.cid():
        failures.append(
            "w72_outer_envelope_w72_params_cid_drift")
    return (len(failures) == 0), failures


@dataclasses.dataclass
class W72Team:
    params: W72Params

    def run_team_turn(
            self, *,
            w71_outer_cid: str,
            ids: Sequence[int] | None = None,
            text: str = "w72",
    ) -> W72HandoffEnvelope:
        p = self.params
        if not p.enabled or p.substrate_v17 is None:
            return W72HandoffEnvelope(
                schema=W72_SCHEMA_VERSION,
                w71_outer_cid=str(w71_outer_cid),
                w72_params_cid=str(p.cid()),
                substrate_v17_witness_cid="",
                kv_bridge_v17_witness_cid="",
                cache_controller_v15_witness_cid="",
                replay_controller_v13_witness_cid="",
                persistent_v24_witness_cid="",
                mlsc_v20_witness_cid="",
                consensus_v18_witness_cid="",
                lhr_v24_witness_cid="",
                deep_substrate_hybrid_v17_witness_cid="",
                substrate_adapter_v17_matrix_cid="",
                masc_v8_witness_cid="",
                team_consensus_controller_v7_witness_cid="",
                rejoin_pressure_falsifier_witness_cid="",
                hosted_router_v5_witness_cid="",
                hosted_logprob_router_v5_witness_cid="",
                hosted_cache_planner_v5_witness_cid="",
                hosted_real_substrate_boundary_v5_cid="",
                hosted_wall_v5_report_cid="",
                handoff_coordinator_v4_witness_cid="",
                handoff_envelope_v4_chain_cid="",
                provider_filter_v4_report_cid="",
                seventeen_way_used=False,
                substrate_v17_used=False,
                masc_v8_v17_beats_v16_rate=0.0,
                masc_v8_tsc_v17_beats_tsc_v16_rate=0.0,
                masc_v8_team_success_per_visible_token=0.0,
                hosted_router_v5_chosen="",
                restart_repair_trajectory_cid="",
            )
        # Plane B — substrate V17 forward.
        token_ids = (
            list(ids) if ids is not None
            else tokenize_bytes_v17(str(text), max_len=16))
        trace, cache = forward_tiny_substrate_v17(
            p.substrate_v17, token_ids,
            visible_token_budget=128.0,
            baseline_token_cost=512.0,
            restart_pressure=0.6,
            rejoin_pressure=0.6)
        # Record restart + delay events on the inner V16 cache.
        from .tiny_substrate_v16 import (
            record_delay_window_v16, record_restart_event_v16,
        )
        record_restart_event_v16(
            cache.v16_cache, turn=1,
            restart_kind="agent_restart",
            role="planner")
        record_delay_window_v16(
            cache.v16_cache, restart_turn=1, repair_turn=4,
            delay_turns=3, role="planner")
        # Record rejoin + branch-pressure events on the V17 cache.
        record_rejoin_event_v17(
            cache, turn=2, rejoin_kind="branch_rejoin",
            branch_id="main", role="planner")
        record_branch_pressure_window_v17(
            cache, restart_turn=1, rejoin_turn=5,
            rejoin_lag_turns=4, branch_id="main",
            role="planner")
        # Re-run forward to produce the restart-repair CID over the
        # recorded events.
        trace, cache = forward_tiny_substrate_v17(
            p.substrate_v17, token_ids,
            v17_kv_cache=cache,
            visible_token_budget=128.0,
            baseline_token_cost=512.0,
            restart_pressure=0.6,
            rejoin_pressure=0.6)
        sub_witness = emit_tiny_substrate_v17_forward_witness(
            trace, cache)
        # KV V17 witnesses.
        rj_falsifier = (
            probe_kv_bridge_v17_rejoin_pressure_falsifier(
                rejoin_pressure_flag=1))
        rrf = compute_restart_repair_fingerprint_v17(
            role="planner",
            repair_trajectory_cid=str(
                cache.v16_cache.v15_cache.repair_trajectory_cid),
            delayed_repair_trajectory_cid=str(
                cache.v16_cache.delayed_repair_trajectory_cid),
            restart_repair_trajectory_cid=str(
                cache.restart_repair_trajectory_cid),
            dominant_repair_label=1,
            restart_count=int(len(
                cache.v16_cache.restart_events)),
            rejoin_count=int(len(cache.rejoin_events)),
            visible_token_budget=128.0,
            baseline_cost=512.0,
            delay_turns=3,
            rejoin_lag_turns=4)
        kv_witness = emit_kv_bridge_v17_witness(
            projection=p.kv_bridge_v17,
            rejoin_pressure_falsifier=rj_falsifier,
            restart_repair_fingerprint=rrf)
        cache_witness = emit_cache_controller_v15_witness(
            controller=p.cache_controller_v15)
        replay_witness = emit_replay_controller_v13_witness(
            p.replay_controller_v13)
        persist_chain = (
            PersistentLatentStateV24Chain.empty())
        persist_witness = emit_persistent_v24_witness(
            persist_chain)
        # MLSC V20 — wrap a trivial V19 capsule up the chain.
        from .mergeable_latent_capsule_v3 import (
            make_root_capsule_v3)
        from .mergeable_latent_capsule_v4 import wrap_v3_as_v4
        from .mergeable_latent_capsule_v5 import wrap_v4_as_v5
        from .mergeable_latent_capsule_v6 import wrap_v5_as_v6
        from .mergeable_latent_capsule_v7 import wrap_v6_as_v7
        from .mergeable_latent_capsule_v8 import wrap_v7_as_v8
        from .mergeable_latent_capsule_v9 import wrap_v8_as_v9
        from .mergeable_latent_capsule_v10 import wrap_v9_as_v10
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
        v3 = make_root_capsule_v3(
            branch_id="w72_smoke",
            payload=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            fact_tags=("w72",), confidence=0.9, trust=0.9,
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
        v18 = wrap_v17_as_v18(
            v17,
            repair_trajectory_chain=(
                str(cache.v16_cache.v15_cache.repair_trajectory_cid),),
            budget_primary_chain=(
                f"bp_{int(trace.v16_trace.v15_trace.budget_primary_gate_per_layer.mean() * 1000)}",),
        )
        v19 = wrap_v18_as_v19(
            v18,
            delayed_repair_trajectory_chain=(
                str(cache.v16_cache.delayed_repair_trajectory_cid),),
            restart_dominance_chain=(
                f"rst_{int(len(cache.v16_cache.restart_events))}",),
        )
        v20 = wrap_v19_as_v20(
            v19,
            restart_repair_trajectory_chain=(
                str(cache.restart_repair_trajectory_cid),),
            rejoin_pressure_chain=(
                f"rj_{int(len(cache.rejoin_events))}",),
        )
        mlsc_witness = emit_mlsc_v20_witness(v20)
        consensus_witness = emit_consensus_v18_witness(
            p.consensus_v18)
        lhr_witness = emit_lhr_v24_witness(
            p.lhr_v24, carrier=[0.1] * 6, k=16,
            partial_contradiction_indicator=[0.5] * 8,
            multi_branch_rejoin_indicator=[0.6] * 8,
            repair_dominance_indicator=[0.7] * 7,
            restart_indicator=[0.5] * 8,
            rejoin_indicator=[0.6] * 8)
        # Deep substrate hybrid V17 — fold the V16 witness as a
        # pre-condition.
        v16_witness = DeepSubstrateHybridV16ForwardWitness(
            schema="coordpy.deep_substrate_hybrid_v16.v1",
            hybrid_cid="",
            inner_v15_witness_cid="",
            sixteen_way=True,
            cache_controller_v14_fired=True,
            replay_controller_v12_fired=True,
            delayed_repair_trajectory_active=True,
            restart_dominance_active=True,
            team_consensus_controller_v6_active=True,
            delayed_repair_trajectory_cid=str(
                cache.v16_cache.delayed_repair_trajectory_cid),
            restart_dominance_l1=int(
                sub_witness.delayed_rejoin_after_restart_l1
                + 1),
            delayed_repair_gate_mean=float(
                trace.v16_trace.delayed_repair_gate_per_layer
                .mean()),
        )
        deep_v17_witness = deep_substrate_hybrid_v17_forward(
            hybrid=p.deep_substrate_hybrid_v17,
            v16_witness=v16_witness,
            cache_controller_v15=p.cache_controller_v15,
            replay_controller_v13=p.replay_controller_v13,
            restart_repair_trajectory_cid=str(
                cache.restart_repair_trajectory_cid),
            delayed_rejoin_after_restart_l1=int(
                sub_witness.delayed_rejoin_after_restart_l1),
            rejoin_pressure_gate_mean=float(
                trace.rejoin_pressure_gate_per_layer.mean()),
            n_team_consensus_v7_invocations=1)
        adapter_matrix = probe_all_v17_adapters()
        # MASC V8 — run a batch for the envelope (all regimes).
        per_regime_aggs = {}
        for regime in W72_MASC_V8_REGIMES:
            _, agg = p.multi_agent_coordinator_v8.run_batch(
                seeds=list(range(int(p.masc_v8_n_seeds))),
                regime=regime)
            per_regime_aggs[regime] = agg
        masc_witness = (
            emit_multi_agent_substrate_coordinator_v8_witness(
                coordinator=p.multi_agent_coordinator_v8,
                per_regime_aggregate=per_regime_aggs))
        # TCC V7 — fire each new arbiter so the witness counts > 0.
        tcc_v7 = p.team_consensus_controller_v7
        tcc_v7.decide_v7(
            regime=(
                "delayed_rejoin_after_restart_under_budget"),
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
                cache.v16_cache.delayed_repair_trajectory_cid),
            delay_turns=3,
            agent_delay_absorption_scores=[
                0.9, 0.5, 0.4, 0.3],
            rejoin_pressure=0.8,
            agent_rejoin_recovery_flags=[1, 1, 0, 0],
            restart_repair_trajectory_cid=str(
                cache.restart_repair_trajectory_cid),
            rejoin_lag_turns=4,
            agent_rejoin_absorption_scores=[
                0.95, 0.7, 0.5, 0.4])
        tcc_v7.decide_v7(
            regime="baseline",
            agent_guesses=[0.5, 0.5, 0.4, 0.5],
            agent_confidences=[0.8, 0.6, 0.7, 0.7],
            substrate_replay_trust=0.7,
            visible_token_budget_ratio=0.9,
            dominant_repair_label=1,
            agent_repair_labels=[1, 1, 0, 0],
            rejoin_pressure=0.7,
            agent_rejoin_recovery_flags=[1, 0, 1, 0])
        tcc_v7.decide_v7(
            regime="baseline",
            agent_guesses=[0.5, 0.5, 0.4, 0.5],
            agent_confidences=[0.8, 0.6, 0.7, 0.7],
            substrate_replay_trust=0.7,
            visible_token_budget_ratio=0.9,
            dominant_repair_label=0,
            restart_repair_trajectory_cid=str(
                cache.restart_repair_trajectory_cid),
            rejoin_lag_turns=4,
            agent_rejoin_absorption_scores=[
                0.8, 0.5, 0.4, 0.7])
        tcc_witness = emit_team_consensus_controller_v7_witness(
            tcc_v7)
        # Plane A V5 — hosted.
        planned, _ = (
            p.hosted_cache_planner_v5
            .plan_per_role_three_layer_rotated(
                shared_prefix_text=(
                    "W72 team shared prefix " * 12),
                per_role_blocks={
                    "plan": ["t0", "t1"],
                    "research": ["r0", "r1"],
                    "write": ["w0", "w1"],
                    "review": ["v0", "v1"],
                }))
        # Router V5 — at least one decision so witness is non-empty.
        req_v5 = HostedRoutingRequestV5(
            inner_v4=HostedRoutingRequestV4(
                inner_v3=HostedRoutingRequestV3(
                    inner_v2=HostedRoutingRequestV2(
                        inner_v1=HostedRoutingRequest(
                            request_cid="w72-router-turn",
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
            weight_delayed_rejoin_match=0.4)
        router_dec = p.hosted_router_v5.decide_v5(req_v5)
        router_v5_witness = (
            emit_hosted_router_controller_v5_witness(
                p.hosted_router_v5))
        logprob_v5_witness = (
            emit_hosted_logprob_router_v5_witness(
                p.hosted_logprob_router_v5))
        cache_planner_v5_witness = (
            emit_hosted_cache_aware_planner_v5_witness(
                p.hosted_cache_planner_v5))
        boundary_v5 = p.hosted_real_substrate_boundary_v5
        wall_v5_report = build_wall_report_v5(
            boundary=boundary_v5)
        # Provider filter V4 — run once to seal a report CID.
        _, filter_report = filter_hosted_registry_v4(
            p.hosted_registry, p.hosted_provider_filter_v4,
            provider_restart_noise={
                "openrouter_paid": 0.5,
                "openai_paid": 0.1},
            provider_rejoin_noise={
                "openrouter_paid": 0.4,
                "openai_paid": 0.1})
        filter_report_cid = _sha256_hex({
            "kind": "w72_provider_filter_v4_report",
            "report": dict(filter_report),
        })
        # Handoff coordinator V4 decisions.
        env_text_only = p.handoff_coordinator_v4.decide_v4(
            req_v4=HandoffRequestV4(
                inner_v3=HandoffRequestV3(
                    inner_v2=HandoffRequestV2(
                        inner_v1=HandoffRequest(
                            request_cid="w72-turn-text",
                            needs_text_only=True,
                            needs_substrate_state_access=False),
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
            substrate_self_checksum_cid=str(
                cache.v16_cache.v15_cache.v14_cache
                .substrate_self_checksum_cid))
        env_rejoin_promoted = (
            p.handoff_coordinator_v4.decide_v4(
                req_v4=HandoffRequestV4(
                    inner_v3=HandoffRequestV3(
                        inner_v2=HandoffRequestV2(
                            inner_v1=HandoffRequest(
                                request_cid=(
                                    "w72-turn-rejoin"),
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
                    rejoin_pressure=0.8,
                    restart_repair_trajectory_cid=str(
                        cache.restart_repair_trajectory_cid),
                    rejoin_lag_turns=4,
                    expected_substrate_trust_v4=0.7),
                substrate_self_checksum_cid=str(
                    cache.v16_cache.v15_cache.v14_cache
                    .substrate_self_checksum_cid)))
        env_rj_fallback = (
            p.handoff_coordinator_v4.decide_v4(
                req_v4=HandoffRequestV4(
                    inner_v3=HandoffRequestV3(
                        inner_v2=HandoffRequestV2(
                            inner_v1=HandoffRequest(
                                request_cid="w72-turn-rj",
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
                    restart_repair_trajectory_cid=str(
                        cache.restart_repair_trajectory_cid),
                    rejoin_lag_turns=4,
                    expected_substrate_trust_v4=0.7),
                substrate_self_checksum_cid=str(
                    cache.v16_cache.v15_cache.v14_cache
                    .substrate_self_checksum_cid)))
        env_substrate_only = (
            p.handoff_coordinator_v4.decide_v4(
                req_v4=HandoffRequestV4(
                    inner_v3=HandoffRequestV3(
                        inner_v2=HandoffRequestV2(
                            inner_v1=HandoffRequest(
                                request_cid=(
                                    "w72-turn-substrate"),
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
                substrate_self_checksum_cid=str(
                    cache.v16_cache.v15_cache.v14_cache
                    .substrate_self_checksum_cid)))
        handoff_v4_witness = (
            emit_hosted_real_handoff_coordinator_v4_witness(
                p.handoff_coordinator_v4))
        handoff_envelope_chain_cid = _sha256_hex({
            "kind": "w72_handoff_envelope_v4_chain",
            "envelopes": [
                env_text_only.cid(),
                env_rejoin_promoted.cid(),
                env_rj_fallback.cid(),
                env_substrate_only.cid(),
            ],
        })
        baseline_agg = per_regime_aggs.get("baseline")
        v17_beats = (
            float(baseline_agg.v17_beats_v16_rate)
            if baseline_agg is not None else 0.0)
        tsc_v17_beats = (
            float(baseline_agg.tsc_v17_beats_tsc_v16_rate)
            if baseline_agg is not None else 0.0)
        ts_per_vt = (
            float(
                baseline_agg.team_success_per_visible_token_v17)
            if baseline_agg is not None else 0.0)
        return W72HandoffEnvelope(
            schema=W72_SCHEMA_VERSION,
            w71_outer_cid=str(w71_outer_cid),
            w72_params_cid=str(p.cid()),
            substrate_v17_witness_cid=str(sub_witness.cid()),
            kv_bridge_v17_witness_cid=str(kv_witness.cid()),
            cache_controller_v15_witness_cid=str(
                cache_witness.cid()),
            replay_controller_v13_witness_cid=str(
                replay_witness.cid()),
            persistent_v24_witness_cid=str(
                persist_witness.cid()),
            mlsc_v20_witness_cid=str(mlsc_witness.cid()),
            consensus_v18_witness_cid=str(
                consensus_witness.cid()),
            lhr_v24_witness_cid=str(lhr_witness.cid()),
            deep_substrate_hybrid_v17_witness_cid=str(
                deep_v17_witness.cid()),
            substrate_adapter_v17_matrix_cid=str(
                adapter_matrix.cid()),
            masc_v8_witness_cid=str(masc_witness.cid()),
            team_consensus_controller_v7_witness_cid=str(
                tcc_witness.cid()),
            rejoin_pressure_falsifier_witness_cid=str(
                rj_falsifier.cid()),
            hosted_router_v5_witness_cid=str(
                router_v5_witness.cid()),
            hosted_logprob_router_v5_witness_cid=str(
                logprob_v5_witness.cid()),
            hosted_cache_planner_v5_witness_cid=str(
                cache_planner_v5_witness.cid()),
            hosted_real_substrate_boundary_v5_cid=str(
                boundary_v5.cid()),
            hosted_wall_v5_report_cid=str(
                wall_v5_report.cid()),
            handoff_coordinator_v4_witness_cid=str(
                handoff_v4_witness.cid()),
            handoff_envelope_v4_chain_cid=str(
                handoff_envelope_chain_cid),
            provider_filter_v4_report_cid=str(
                filter_report_cid),
            seventeen_way_used=bool(
                deep_v17_witness.seventeen_way),
            substrate_v17_used=True,
            masc_v8_v17_beats_v16_rate=float(v17_beats),
            masc_v8_tsc_v17_beats_tsc_v16_rate=float(
                tsc_v17_beats),
            masc_v8_team_success_per_visible_token=float(
                ts_per_vt),
            hosted_router_v5_chosen=str(
                router_dec.chosen_provider or ""),
            restart_repair_trajectory_cid=str(
                cache.restart_repair_trajectory_cid),
        )


def build_default_w72_team(*, seed: int = 72000) -> W72Team:
    return W72Team(params=W72Params.build_default(seed=int(seed)))


__all__ = [
    "W72_SCHEMA_VERSION",
    "W72_FAILURE_MODES",
    "W72Params",
    "W72HandoffEnvelope",
    "verify_w72_handoff",
    "W72Team",
    "build_default_w72_team",
]
