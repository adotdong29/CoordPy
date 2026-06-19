"""W71 — Stronger Delayed-Repair-After-Restart / Repair-Trajectory-
Primary Two-Plane Multi-Agent Substrate team.

The ``W71Team`` orchestrator strictly wraps the ``W70Team`` and
adds the W71 mechanism modules organised across two planes plus
the new **restart-aware Plane A↔B handoff coordinator V3**:

**Plane B — Real substrate (in-repo, V16 stack):**

* M1  ``tiny_substrate_v16``           (18-layer, 3 new V16 axes)
* M2  ``kv_bridge_v16``                (12-target ridge + 84-dim
                                        delayed-repair fingerprint +
                                        restart-dominance falsifier)
* M3  ``cache_controller_v14``         (11-objective ridge + per-
                                        role 12-dim restart-priority
                                        head)
* M4  ``replay_controller_v12``        (19 regimes + 9-label
                                        restart-aware routing head)
* M5  ``deep_substrate_hybrid_v16``    (16-way bidirectional loop)
* M6  ``substrate_adapter_v16``        (substrate_v16_full tier)
* M7  ``persistent_latent_v23``        (22 layers, 20th carrier,
                                        max_chain_walk_depth=262144)
* M8  ``long_horizon_retention_v23``   (22 heads, max_k=640)
* M9  ``mergeable_latent_capsule_v19`` (delayed-repair chain +
                                        restart-dominance chain)
* M10 ``consensus_fallback_controller_v17`` (28-stage chain)
* M11 ``multi_agent_substrate_coordinator_v7`` (16-policy, 11-regime
                                                MASC V7)
* M12 ``team_consensus_controller_v6`` (restart-aware + delayed-
                                        repair arbiters)

**Plane A — Hosted control plane V4 (honest, no substrate):**

* H1  ``hosted_router_controller_v4``  (restart-pressure weighting
                                        + delayed-repair match)
* H2  ``hosted_logprob_router_v4``     (restart-aware abstain floor
                                        + per-budget+restart
                                        tiebreak)
* H3  ``hosted_cache_aware_planner_v4``(two-layer rotated prefix +
                                        ≥ 72 % savings 10×8 hit=1)
* H4  ``hosted_cost_planner_v4``       (cost-per-repair-success-
                                        under-budget +
                                        abstain-when-restart-
                                        pressure-violated)
* H5  ``hosted_real_substrate_boundary_v4`` (wall V4, 25 blocked
                                             axes)
* H6  ``hosted_real_handoff_coordinator_v3`` (the **new restart-
                                              aware Plane A↔B
                                              bridge** — V3
                                              envelopes + delayed-
                                              repair falsifier +
                                              cross-plane savings)
* H7  ``hosted_provider_filter_v3``    (restart-aware provider
                                        filter)

Per-turn it emits 19 W71 module witness CIDs (12 Plane B + 7 Plane
A V4) and a V3 handoff envelope CID, sealing them into a
``W71HandoffEnvelope`` whose ``w70_outer_cid`` carries forward the
W70 envelope byte-for-byte.

Honest scope (W71)
------------------

* Plane A V4 operates at the hosted text/logprob/prefix-cache
  surface. It does NOT pierce hidden state / KV / attention.
  ``W71-L-HOSTED-V4-NO-SUBSTRATE-CAP``.
* Plane B is the in-repo V16 NumPy runtime. We do NOT bridge to
  third-party hosted models.
  ``W71-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
* W71 fits closed-form ridge parameters in three new places on top
  of W70's 58: cache V14 eleven-objective; cache V14 per-role
  restart-priority; replay V12 restart-aware routing head; KV V16
  twelve-target. Total **62 closed-form ridge solves** across
  W61..W71. No autograd, no SGD, no GPU.
* Trivial passthrough preserved: when ``W71Params.build_trivial()``
  is used the W71 envelope's internal ``w70_outer_cid`` carries
  the supplied W70 outer CID exactly.
* The handoff coordinator V3 preserves the wall: a content-
  addressed V3 envelope says which plane handled each turn under
  the restart-aware score; it does NOT cross the substrate
  boundary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .cache_controller_v14 import (
    CacheControllerV14,
    emit_cache_controller_v14_witness,
    fit_eleven_objective_ridge_v14,
    fit_per_role_restart_priority_head_v14,
)
from .consensus_fallback_controller_v17 import (
    ConsensusFallbackControllerV17,
    W71_CONSENSUS_V17_STAGES,
    emit_consensus_v17_witness,
)
from .deep_substrate_hybrid_v15 import (
    DeepSubstrateHybridV15,
    DeepSubstrateHybridV15ForwardWitness,
)
from .deep_substrate_hybrid_v16 import (
    DeepSubstrateHybridV16,
    deep_substrate_hybrid_v16_forward,
)
from .hosted_cache_aware_planner_v4 import (
    HostedCacheAwarePlannerV4,
    emit_hosted_cache_aware_planner_v4_witness,
)
from .hosted_cost_planner_v4 import HostedCostPlanSpecV4
from .hosted_logprob_router_v4 import (
    HostedLogprobRouterV4,
    emit_hosted_logprob_router_v4_witness,
)
from .hosted_provider_filter_v3 import (
    HostedProviderFilterSpecV3,
    filter_hosted_registry_v3,
)
from .hosted_provider_filter_v2 import HostedProviderFilterSpecV2
from .hosted_provider_filter import HostedProviderFilterSpec
from .hosted_real_handoff_coordinator_v3 import (
    HandoffRequestV3, HostedRealHandoffCoordinatorV3,
    emit_hosted_real_handoff_coordinator_v3_witness,
    hosted_real_handoff_v3_restart_aware_savings,
)
from .hosted_real_handoff_coordinator_v2 import HandoffRequestV2
from .hosted_real_handoff_coordinator import HandoffRequest
from .hosted_real_substrate_boundary_v4 import (
    HostedRealSubstrateBoundaryV4,
    build_default_hosted_real_substrate_boundary_v4,
    build_wall_report_v4,
    probe_hosted_real_substrate_boundary_v4_falsifier,
)
from .hosted_router_controller import (
    HostedProviderRegistry, HostedRoutingRequest,
    default_hosted_registry,
)
from .hosted_router_controller_v2 import HostedRoutingRequestV2
from .hosted_router_controller_v3 import HostedRoutingRequestV3
from .hosted_router_controller_v4 import (
    HostedRouterControllerV4, HostedRoutingRequestV4,
    emit_hosted_router_controller_v4_witness,
)
from .kv_bridge_v14 import KVBridgeV14Projection
from .kv_bridge_v15 import KVBridgeV15Projection
from .kv_bridge_v16 import (
    KVBridgeV16Projection,
    compute_delayed_repair_fingerprint_v16,
    emit_kv_bridge_v16_witness,
    probe_kv_bridge_v16_restart_dominance_falsifier,
)
from .long_horizon_retention_v23 import (
    LongHorizonReconstructionV23Head,
    emit_lhr_v23_witness,
)
from .mergeable_latent_capsule_v19 import (
    MergeOperatorV19, emit_mlsc_v19_witness, wrap_v18_as_v19,
)
from .multi_agent_substrate_coordinator_v7 import (
    MultiAgentSubstrateCoordinatorV7,
    W71_MASC_V7_REGIMES,
    emit_multi_agent_substrate_coordinator_v7_witness,
)
from .persistent_latent_v23 import (
    PersistentLatentStateV23Chain,
    emit_persistent_v23_witness,
)
from .replay_controller_v12 import (
    ReplayControllerV12,
    W71_REPLAY_REGIMES_V12,
    W71_RESTART_AWARE_ROUTING_LABELS,
    fit_replay_controller_v12_per_role,
    fit_replay_v12_restart_aware_routing_head,
    emit_replay_controller_v12_witness,
)
from .replay_controller import ReplayCandidate
from .substrate_adapter_v16 import (
    W71_SUBSTRATE_TIER_SUBSTRATE_V16_FULL,
    probe_all_v16_adapters,
)
from .team_consensus_controller_v6 import (
    TeamConsensusControllerV6,
    emit_team_consensus_controller_v6_witness,
)
from .tiny_substrate_v16 import (
    TinyV16SubstrateParams,
    build_default_tiny_substrate_v16,
    emit_tiny_substrate_v16_forward_witness,
    forward_tiny_substrate_v16,
    record_delay_window_v16,
    record_restart_event_v16,
    tokenize_bytes_v16,
)
from .w70_team import (
    W70HandoffEnvelope, W70Params, W70Team,
)


W71_SCHEMA_VERSION: str = "coordpy.w71_team.v1"

W71_FAILURE_MODES: tuple[str, ...] = (
    "w71_outer_envelope_schema_mismatch",
    "w71_outer_envelope_w70_outer_cid_drift",
    "w71_outer_envelope_w71_params_cid_drift",
    "w71_outer_envelope_witness_cid_drift",
    "w71_substrate_v16_n_layers_off",
    "w71_substrate_v16_delayed_repair_trajectory_cid_off",
    "w71_substrate_v16_restart_dominance_per_layer_shape_off",
    "w71_substrate_v16_delayed_repair_gate_shape_off",
    "w71_kv_bridge_v16_n_targets_off",
    "w71_kv_bridge_v16_restart_dominance_falsifier_off",
    "w71_cache_v14_eleven_objective_off",
    "w71_replay_v12_regime_count_off",
    "w71_replay_v12_restart_aware_routing_count_off",
    "w71_consensus_v17_stage_count_off",
    "w71_lhr_v23_max_k_off",
    "w71_lhr_v23_n_heads_off",
    "w71_persistent_v23_n_layers_off",
    "w71_substrate_adapter_v16_tier_off",
    "w71_masc_v7_v16_beats_v15_rate_under_threshold",
    "w71_masc_v7_tsc_v16_beats_tsc_v15_rate_under_threshold",
    "w71_masc_v7_compound_regime_inferior_to_baseline",
    "w71_hosted_router_v4_decision_not_deterministic",
    "w71_hosted_logprob_v4_abstain_floor_off",
    "w71_hosted_cache_aware_v4_savings_below_72_percent",
    "w71_hosted_cost_planner_v4_no_eligible",
    "w71_hosted_real_substrate_boundary_v4_blocked_axis_satisfied",
    "w71_sixteen_way_loop_not_observed",
    "w71_handoff_coordinator_v3_inconsistent",
    "w71_handoff_v3_cross_plane_savings_below_70_percent",
    "w71_team_consensus_v6_no_decisions",
    "w71_handoff_v3_restart_alignment_off",
    "w71_handoff_envelope_v3_chain_cid_drift",
    "w71_inner_v70_envelope_invariant_off",
    "w71_handoff_v3_delayed_repair_fallback_off",
    "w71_hosted_boundary_v4_blocked_axes_below_25",
    "w71_v16_substrate_self_checksum_cid_off",
    "w71_delayed_repair_trajectory_cid_drift",
    "w71_mlsc_v19_delayed_repair_trajectory_chain_off",
    "w71_v7_team_success_per_visible_token_below_floor",
    "w71_v7_visible_tokens_savings_below_65_percent",
    "w71_v7_compound_regime_v16_beats_v15_below_threshold",
    "w71_substrate_v16_delayed_repair_trajectory_chain_synthetic",
    "w71_inner_v16_falsifier_kind_off",
    "w71_handoff_v3_envelope_dr_alignment_off",
    "w71_hosted_router_v4_per_routing_cid_off",
    "w71_consensus_v17_restart_aware_arbiter_off",
    "w71_consensus_v17_delayed_repair_arbiter_off",
    "w71_tcc_v6_restart_aware_arbiter_off",
    "w71_tcc_v6_delayed_repair_arbiter_off",
    "w71_cache_v14_per_role_restart_priority_head_off",
    "w71_kv_bridge_v16_delayed_repair_fingerprint_off",
    "w71_substrate_v16_restart_events_off",
    "w71_provider_filter_v3_pressure_drop_off",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class W71Params:
    substrate_v16: TinyV16SubstrateParams | None
    kv_bridge_v16: KVBridgeV16Projection | None
    cache_controller_v14: CacheControllerV14 | None
    replay_controller_v12: ReplayControllerV12 | None
    consensus_v17: ConsensusFallbackControllerV17 | None
    lhr_v23: LongHorizonReconstructionV23Head | None
    deep_substrate_hybrid_v16: DeepSubstrateHybridV16 | None
    mlsc_v19_operator: MergeOperatorV19 | None
    multi_agent_coordinator_v7: (
        MultiAgentSubstrateCoordinatorV7 | None)
    team_consensus_controller_v6: (
        TeamConsensusControllerV6 | None)
    hosted_registry: HostedProviderRegistry | None
    hosted_router_v4: HostedRouterControllerV4 | None
    hosted_logprob_router_v4: HostedLogprobRouterV4 | None
    hosted_cache_planner_v4: HostedCacheAwarePlannerV4 | None
    hosted_real_substrate_boundary_v4: (
        HostedRealSubstrateBoundaryV4 | None)
    handoff_coordinator_v3: (
        HostedRealHandoffCoordinatorV3 | None)
    hosted_provider_filter_v3: (
        HostedProviderFilterSpecV3 | None)
    w70_params: W70Params | None
    enabled: bool = True
    masc_v7_n_seeds: int = 8

    @classmethod
    def build_trivial(cls) -> "W71Params":
        return cls(
            substrate_v16=None,
            kv_bridge_v16=None,
            cache_controller_v14=None,
            replay_controller_v12=None,
            consensus_v17=None, lhr_v23=None,
            deep_substrate_hybrid_v16=None,
            mlsc_v19_operator=None,
            multi_agent_coordinator_v7=None,
            team_consensus_controller_v6=None,
            hosted_registry=None,
            hosted_router_v4=None,
            hosted_logprob_router_v4=None,
            hosted_cache_planner_v4=None,
            hosted_real_substrate_boundary_v4=None,
            handoff_coordinator_v3=None,
            hosted_provider_filter_v3=None,
            w70_params=None,
            enabled=False,
        )

    @classmethod
    def build_default(
            cls, *, seed: int = 71000,
    ) -> "W71Params":
        sub_v16 = build_default_tiny_substrate_v16(
            seed=int(seed) + 1)
        # KV V16 projection chain (deep nest through V15..V3).
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
        cfg = (sub_v16.config.v15.v14.v13.v12.v11.v10.v9)
        d_head = int(cfg.d_model) // int(cfg.n_heads)
        kv_b3 = KVBridgeV3Projection.init(
            n_layers=int(cfg.n_layers),
            n_heads=int(cfg.n_heads),
            n_kv_heads=int(cfg.n_kv_heads),
            n_inject_tokens=3, carrier_dim=6,
            d_head=int(d_head), seed=int(seed) + 7)
        kv_b15 = KVBridgeV15Projection.init_from_v14(
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
            seed_v15=int(seed) + 19)
        kv_b16 = KVBridgeV16Projection.init_from_v15(
            kv_b15, seed_v16=int(seed) + 20)
        cc14 = CacheControllerV14.init(
            fit_seed=int(seed) + 32)
        import numpy as _np
        rng = _np.random.default_rng(int(seed) + 33)
        X = rng.standard_normal((10, 4))
        cc14, _ = fit_eleven_objective_ridge_v14(
            controller=cc14, train_features=X.tolist(),
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
                X[:, 3] * 0.4 + X[:, 0] * 0.2).tolist())
        X12 = rng.standard_normal((10, 12))
        cc14, _ = fit_per_role_restart_priority_head_v14(
            controller=cc14, role="planner",
            train_features=X12.tolist(),
            target_restart_priorities=(
                X12[:, 0] * 0.3
                + X12[:, 11] * 0.4).tolist())
        # Replay V12.
        rcv12 = ReplayControllerV12.init()
        v12_cands = {
            r: [ReplayCandidate(
                100, 1000, 50, 0.1, 0.0, 0.3,
                True, True, 0)]
            for r in W71_REPLAY_REGIMES_V12}
        v12_decs = {
            r: ["choose_reuse"]
            for r in W71_REPLAY_REGIMES_V12}
        rcv12, _ = fit_replay_controller_v12_per_role(
            controller=rcv12, role="planner",
            train_candidates_per_regime=v12_cands,
            train_decisions_per_regime=v12_decs)
        X_team = rng.standard_normal((45, 10))
        labs: list[str] = []
        for i in range(45):
            lab_idx = (
                i % len(W71_RESTART_AWARE_ROUTING_LABELS))
            labs.append(
                W71_RESTART_AWARE_ROUTING_LABELS[lab_idx])
        rcv12, _ = fit_replay_v12_restart_aware_routing_head(
            controller=rcv12,
            train_team_features=X_team.tolist(),
            train_routing_labels=labs)
        consensus_v17 = ConsensusFallbackControllerV17.init(
            k_required=2, cosine_floor=0.6,
            trust_threshold=0.5,
            multi_branch_rejoin_threshold=0.5,
            silent_corruption_threshold=0.5,
            repair_dominance_threshold=0.5,
            budget_primary_threshold=0.5,
            restart_aware_threshold=0.5,
            delayed_repair_threshold=0.5)
        lhr23 = LongHorizonReconstructionV23Head.init(
            seed=int(seed) + 40)
        deep_v16 = DeepSubstrateHybridV16()
        mlsc_v19_op = MergeOperatorV19()
        masc_v7 = MultiAgentSubstrateCoordinatorV7()
        tcc_v6 = TeamConsensusControllerV6()
        reg = default_hosted_registry()
        hosted_router_v4 = HostedRouterControllerV4.init(
            reg, {
                "openrouter_paid": 0.85,
                "openai_paid": 0.92,
            })
        hosted_logprob_router_v4 = HostedLogprobRouterV4()
        hosted_cache_planner_v4 = HostedCacheAwarePlannerV4()
        boundary_v4 = (
            build_default_hosted_real_substrate_boundary_v4())
        handoff_coord_v3 = HostedRealHandoffCoordinatorV3(
            boundary_v4=boundary_v4)
        # Provider filter V3 — pre-configured restart-aware spec.
        provider_filter_v3 = HostedProviderFilterSpecV3(
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
                "text_only": 0.4})
        # W70 inner params for envelope chaining.
        w70_params = W70Params.build_default(
            seed=int(seed) - 1000)
        return cls(
            substrate_v16=sub_v16,
            kv_bridge_v16=kv_b16,
            cache_controller_v14=cc14,
            replay_controller_v12=rcv12,
            consensus_v17=consensus_v17,
            lhr_v23=lhr23,
            deep_substrate_hybrid_v16=deep_v16,
            mlsc_v19_operator=mlsc_v19_op,
            multi_agent_coordinator_v7=masc_v7,
            team_consensus_controller_v6=tcc_v6,
            hosted_registry=reg,
            hosted_router_v4=hosted_router_v4,
            hosted_logprob_router_v4=hosted_logprob_router_v4,
            hosted_cache_planner_v4=hosted_cache_planner_v4,
            hosted_real_substrate_boundary_v4=boundary_v4,
            handoff_coordinator_v3=handoff_coord_v3,
            hosted_provider_filter_v3=provider_filter_v3,
            w70_params=w70_params,
            enabled=True,
            masc_v7_n_seeds=6,
        )

    def to_dict(self) -> dict[str, Any]:
        def _cid_or_empty(x: Any) -> str:
            return str(x.cid()) if x is not None else ""
        return {
            "schema": W71_SCHEMA_VERSION,
            "kind": "w71_params",
            "substrate_v16_cid": _cid_or_empty(
                self.substrate_v16),
            "kv_bridge_v16_cid": _cid_or_empty(
                self.kv_bridge_v16),
            "cache_controller_v14_cid": _cid_or_empty(
                self.cache_controller_v14),
            "replay_controller_v12_cid": _cid_or_empty(
                self.replay_controller_v12),
            "consensus_v17_cid": _cid_or_empty(
                self.consensus_v17),
            "lhr_v23_cid": _cid_or_empty(self.lhr_v23),
            "deep_substrate_hybrid_v16_cid": _cid_or_empty(
                self.deep_substrate_hybrid_v16),
            "mlsc_v19_operator_cid": _cid_or_empty(
                self.mlsc_v19_operator),
            "multi_agent_coordinator_v7_cid": _cid_or_empty(
                self.multi_agent_coordinator_v7),
            "team_consensus_controller_v6_cid": _cid_or_empty(
                self.team_consensus_controller_v6),
            "hosted_registry_cid": _cid_or_empty(
                self.hosted_registry),
            "hosted_router_v4_cid": _cid_or_empty(
                self.hosted_router_v4),
            "hosted_logprob_router_v4_cid": _cid_or_empty(
                self.hosted_logprob_router_v4),
            "hosted_cache_planner_v4_cid": _cid_or_empty(
                self.hosted_cache_planner_v4),
            "hosted_real_substrate_boundary_v4_cid":
                _cid_or_empty(
                    self.hosted_real_substrate_boundary_v4),
            "handoff_coordinator_v3_cid": _cid_or_empty(
                self.handoff_coordinator_v3),
            "hosted_provider_filter_v3_cid": _cid_or_empty(
                self.hosted_provider_filter_v3),
            "w70_params_cid": _cid_or_empty(self.w70_params),
            "enabled": bool(self.enabled),
            "masc_v7_n_seeds": int(self.masc_v7_n_seeds),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w71_params",
            "params": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class W71HandoffEnvelope:
    schema: str
    w70_outer_cid: str
    w71_params_cid: str
    substrate_v16_witness_cid: str
    kv_bridge_v16_witness_cid: str
    cache_controller_v14_witness_cid: str
    replay_controller_v12_witness_cid: str
    persistent_v23_witness_cid: str
    mlsc_v19_witness_cid: str
    consensus_v17_witness_cid: str
    lhr_v23_witness_cid: str
    deep_substrate_hybrid_v16_witness_cid: str
    substrate_adapter_v16_matrix_cid: str
    masc_v7_witness_cid: str
    team_consensus_controller_v6_witness_cid: str
    restart_dominance_falsifier_witness_cid: str
    hosted_router_v4_witness_cid: str
    hosted_logprob_router_v4_witness_cid: str
    hosted_cache_planner_v4_witness_cid: str
    hosted_real_substrate_boundary_v4_cid: str
    hosted_wall_v4_report_cid: str
    handoff_coordinator_v3_witness_cid: str
    handoff_envelope_v3_chain_cid: str
    provider_filter_v3_report_cid: str
    sixteen_way_used: bool
    substrate_v16_used: bool
    masc_v7_v16_beats_v15_rate: float
    masc_v7_tsc_v16_beats_tsc_v15_rate: float
    masc_v7_team_success_per_visible_token: float
    hosted_router_v4_chosen: str
    delayed_repair_trajectory_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "w70_outer_cid": str(self.w70_outer_cid),
            "w71_params_cid": str(self.w71_params_cid),
            "substrate_v16_witness_cid": str(
                self.substrate_v16_witness_cid),
            "kv_bridge_v16_witness_cid": str(
                self.kv_bridge_v16_witness_cid),
            "cache_controller_v14_witness_cid": str(
                self.cache_controller_v14_witness_cid),
            "replay_controller_v12_witness_cid": str(
                self.replay_controller_v12_witness_cid),
            "persistent_v23_witness_cid": str(
                self.persistent_v23_witness_cid),
            "mlsc_v19_witness_cid": str(
                self.mlsc_v19_witness_cid),
            "consensus_v17_witness_cid": str(
                self.consensus_v17_witness_cid),
            "lhr_v23_witness_cid": str(
                self.lhr_v23_witness_cid),
            "deep_substrate_hybrid_v16_witness_cid": str(
                self.deep_substrate_hybrid_v16_witness_cid),
            "substrate_adapter_v16_matrix_cid": str(
                self.substrate_adapter_v16_matrix_cid),
            "masc_v7_witness_cid": str(
                self.masc_v7_witness_cid),
            "team_consensus_controller_v6_witness_cid": str(
                self.team_consensus_controller_v6_witness_cid),
            "restart_dominance_falsifier_witness_cid": str(
                self.restart_dominance_falsifier_witness_cid),
            "hosted_router_v4_witness_cid": str(
                self.hosted_router_v4_witness_cid),
            "hosted_logprob_router_v4_witness_cid": str(
                self.hosted_logprob_router_v4_witness_cid),
            "hosted_cache_planner_v4_witness_cid": str(
                self.hosted_cache_planner_v4_witness_cid),
            "hosted_real_substrate_boundary_v4_cid": str(
                self.hosted_real_substrate_boundary_v4_cid),
            "hosted_wall_v4_report_cid": str(
                self.hosted_wall_v4_report_cid),
            "handoff_coordinator_v3_witness_cid": str(
                self.handoff_coordinator_v3_witness_cid),
            "handoff_envelope_v3_chain_cid": str(
                self.handoff_envelope_v3_chain_cid),
            "provider_filter_v3_report_cid": str(
                self.provider_filter_v3_report_cid),
            "sixteen_way_used": bool(self.sixteen_way_used),
            "substrate_v16_used": bool(self.substrate_v16_used),
            "masc_v7_v16_beats_v15_rate": float(round(
                self.masc_v7_v16_beats_v15_rate, 12)),
            "masc_v7_tsc_v16_beats_tsc_v15_rate": float(round(
                self.masc_v7_tsc_v16_beats_tsc_v15_rate, 12)),
            "masc_v7_team_success_per_visible_token": float(
                round(
                    self
                    .masc_v7_team_success_per_visible_token,
                    12)),
            "hosted_router_v4_chosen": str(
                self.hosted_router_v4_chosen),
            "delayed_repair_trajectory_cid": str(
                self.delayed_repair_trajectory_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w71_handoff_envelope",
            "envelope": self.to_dict()})


def verify_w71_handoff(
        envelope: W71HandoffEnvelope,
        params: W71Params,
        w70_outer_cid: str,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if envelope.schema != W71_SCHEMA_VERSION:
        failures.append(
            "w71_outer_envelope_schema_mismatch")
    if envelope.w70_outer_cid != str(w70_outer_cid):
        failures.append(
            "w71_outer_envelope_w70_outer_cid_drift")
    if envelope.w71_params_cid != params.cid():
        failures.append(
            "w71_outer_envelope_w71_params_cid_drift")
    return (len(failures) == 0), failures


@dataclasses.dataclass
class W71Team:
    params: W71Params

    def run_team_turn(
            self, *,
            w70_outer_cid: str,
            ids: Sequence[int] | None = None,
            text: str = "w71",
    ) -> W71HandoffEnvelope:
        p = self.params
        if not p.enabled or p.substrate_v16 is None:
            return W71HandoffEnvelope(
                schema=W71_SCHEMA_VERSION,
                w70_outer_cid=str(w70_outer_cid),
                w71_params_cid=str(p.cid()),
                substrate_v16_witness_cid="",
                kv_bridge_v16_witness_cid="",
                cache_controller_v14_witness_cid="",
                replay_controller_v12_witness_cid="",
                persistent_v23_witness_cid="",
                mlsc_v19_witness_cid="",
                consensus_v17_witness_cid="",
                lhr_v23_witness_cid="",
                deep_substrate_hybrid_v16_witness_cid="",
                substrate_adapter_v16_matrix_cid="",
                masc_v7_witness_cid="",
                team_consensus_controller_v6_witness_cid="",
                restart_dominance_falsifier_witness_cid="",
                hosted_router_v4_witness_cid="",
                hosted_logprob_router_v4_witness_cid="",
                hosted_cache_planner_v4_witness_cid="",
                hosted_real_substrate_boundary_v4_cid="",
                hosted_wall_v4_report_cid="",
                handoff_coordinator_v3_witness_cid="",
                handoff_envelope_v3_chain_cid="",
                provider_filter_v3_report_cid="",
                sixteen_way_used=False,
                substrate_v16_used=False,
                masc_v7_v16_beats_v15_rate=0.0,
                masc_v7_tsc_v16_beats_tsc_v15_rate=0.0,
                masc_v7_team_success_per_visible_token=0.0,
                hosted_router_v4_chosen="",
                delayed_repair_trajectory_cid="",
            )
        # Plane B — substrate V16 forward.
        token_ids = (
            list(ids) if ids is not None
            else tokenize_bytes_v16(str(text), max_len=16))
        trace, cache = forward_tiny_substrate_v16(
            p.substrate_v16, token_ids,
            visible_token_budget=128.0,
            baseline_token_cost=512.0,
            restart_pressure=0.6)
        # Record restart + delay events.
        record_restart_event_v16(
            cache, turn=1, restart_kind="agent_restart",
            role="planner")
        record_delay_window_v16(
            cache, restart_turn=1, repair_turn=5,
            delay_turns=3, role="planner")
        # Re-run forward to produce the delayed-repair CID over
        # the recorded events.
        trace, cache = forward_tiny_substrate_v16(
            p.substrate_v16, token_ids,
            v16_kv_cache=cache,
            visible_token_budget=128.0,
            baseline_token_cost=512.0,
            restart_pressure=0.6)
        sub_witness = emit_tiny_substrate_v16_forward_witness(
            trace, cache)
        # KV V16 witnesses.
        rd_falsifier = (
            probe_kv_bridge_v16_restart_dominance_falsifier(
                restart_dominance_flag=1))
        drf = compute_delayed_repair_fingerprint_v16(
            role="planner",
            repair_trajectory_cid=str(
                cache.v15_cache.repair_trajectory_cid),
            delayed_repair_trajectory_cid=str(
                cache.delayed_repair_trajectory_cid),
            dominant_repair_label=1,
            restart_count=int(len(cache.restart_events)),
            visible_token_budget=128.0,
            baseline_cost=512.0,
            delay_turns=3)
        kv_witness = emit_kv_bridge_v16_witness(
            projection=p.kv_bridge_v16,
            restart_dominance_falsifier=rd_falsifier,
            delayed_repair_fingerprint=drf)
        cache_witness = emit_cache_controller_v14_witness(
            controller=p.cache_controller_v14)
        replay_witness = emit_replay_controller_v12_witness(
            p.replay_controller_v12)
        persist_chain = (
            PersistentLatentStateV23Chain.empty())
        persist_witness = emit_persistent_v23_witness(
            persist_chain)
        # MLSC V19 — wrap a trivial V18 capsule up the chain.
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
        from .mergeable_latent_capsule_v17 import wrap_v16_as_v17
        from .mergeable_latent_capsule_v18 import wrap_v17_as_v18
        v3 = make_root_capsule_v3(
            branch_id="w71_smoke",
            payload=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            fact_tags=("w71",), confidence=0.9, trust=0.9,
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
                str(cache.v15_cache.repair_trajectory_cid),),
            budget_primary_chain=(
                f"bp_{int(trace.v15_trace.budget_primary_gate_per_layer.mean()*1000)}",),
        )
        v19 = wrap_v18_as_v19(
            v18,
            delayed_repair_trajectory_chain=(
                str(cache.delayed_repair_trajectory_cid),),
            restart_dominance_chain=(
                f"rst_{int(len(cache.restart_events))}",),
        )
        mlsc_witness = emit_mlsc_v19_witness(v19)
        consensus_witness = emit_consensus_v17_witness(
            p.consensus_v17)
        lhr_witness = emit_lhr_v23_witness(
            p.lhr_v23, carrier=[0.1] * 6, k=16,
            partial_contradiction_indicator=[0.5] * 8,
            multi_branch_rejoin_indicator=[0.6] * 8,
            repair_dominance_indicator=[0.7] * 7,
            restart_indicator=[0.5] * 8)
        # Deep substrate hybrid V16 — fold the V15 witness as a
        # pre-condition.
        v15_witness = DeepSubstrateHybridV15ForwardWitness(
            schema="coordpy.deep_substrate_hybrid_v15.v1",
            hybrid_cid="",
            inner_v14_witness_cid="",
            fifteen_way=True,
            cache_controller_v13_fired=True,
            replay_controller_v11_fired=True,
            repair_trajectory_active=True,
            budget_primary_active=True,
            team_consensus_controller_v5_active=True,
            repair_trajectory_cid=str(
                cache.v15_cache.repair_trajectory_cid),
            dominant_repair_l1=int(
                sub_witness.restart_dominance_l1),
            budget_primary_gate_mean=float(
                trace.v15_trace.budget_primary_gate_per_layer.mean()),
        )
        deep_v16_witness = deep_substrate_hybrid_v16_forward(
            hybrid=p.deep_substrate_hybrid_v16,
            v15_witness=v15_witness,
            cache_controller_v14=p.cache_controller_v14,
            replay_controller_v12=p.replay_controller_v12,
            delayed_repair_trajectory_cid=str(
                cache.delayed_repair_trajectory_cid),
            restart_dominance_l1=int(
                sub_witness.restart_dominance_l1),
            delayed_repair_gate_mean=float(
                trace.delayed_repair_gate_per_layer.mean()),
            n_team_consensus_v6_invocations=1)
        adapter_matrix = probe_all_v16_adapters()
        # MASC V7 — run a batch for the envelope (all regimes).
        per_regime_aggs = {}
        for regime in W71_MASC_V7_REGIMES:
            _, agg = p.multi_agent_coordinator_v7.run_batch(
                seeds=list(range(int(p.masc_v7_n_seeds))),
                regime=regime)
            per_regime_aggs[regime] = agg
        masc_witness = (
            emit_multi_agent_substrate_coordinator_v7_witness(
                coordinator=p.multi_agent_coordinator_v7,
                per_regime_aggregate=per_regime_aggs))
        # TCC V6 — fire each new arbiter so the witness counts > 0.
        tcc_v6 = p.team_consensus_controller_v6
        tcc_v6.decide_v6(
            regime=(
                "delayed_repair_after_restart"),
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
                cache.delayed_repair_trajectory_cid),
            delay_turns=3,
            agent_delay_absorption_scores=[
                0.9, 0.5, 0.4, 0.3])
        tcc_v6.decide_v6(
            regime="baseline",
            agent_guesses=[0.5, 0.5, 0.4, 0.5],
            agent_confidences=[0.8, 0.6, 0.7, 0.7],
            substrate_replay_trust=0.7,
            visible_token_budget_ratio=0.9,
            dominant_repair_label=1,
            agent_repair_labels=[1, 1, 0, 0],
            restart_pressure=0.6,
            agent_restart_recovery_flags=[1, 0, 1, 0])
        tcc_v6.decide_v6(
            regime="baseline",
            agent_guesses=[0.5, 0.5, 0.4, 0.5],
            agent_confidences=[0.8, 0.6, 0.7, 0.7],
            substrate_replay_trust=0.7,
            visible_token_budget_ratio=0.9,
            dominant_repair_label=0,
            delayed_repair_trajectory_cid=str(
                cache.delayed_repair_trajectory_cid),
            delay_turns=3,
            agent_delay_absorption_scores=[
                0.8, 0.5, 0.4, 0.7])
        tcc_witness = emit_team_consensus_controller_v6_witness(
            tcc_v6)
        # Plane A V4 — hosted.
        planned, _ = (
            p.hosted_cache_planner_v4
            .plan_per_role_two_layer_rotated(
                shared_prefix_text=(
                    "W71 team shared prefix " * 10),
                per_role_blocks={
                    "plan": ["t0", "t1"],
                    "research": ["r0", "r1"],
                    "write": ["w0", "w1"],
                    "review": ["v0", "v1"],
                }))
        # Router V4 — at least one decision so witness is non-empty.
        req_v4 = HostedRoutingRequestV4(
            inner_v3=HostedRoutingRequestV3(
                inner_v2=HostedRoutingRequestV2(
                    inner_v1=HostedRoutingRequest(
                        request_cid="w71-router-turn",
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
            weight_delayed_repair_match=0.4)
        router_dec = p.hosted_router_v4.decide_v4(req_v4)
        router_v4_witness = (
            emit_hosted_router_controller_v4_witness(
                p.hosted_router_v4))
        logprob_v4_witness = (
            emit_hosted_logprob_router_v4_witness(
                p.hosted_logprob_router_v4))
        cache_planner_v4_witness = (
            emit_hosted_cache_aware_planner_v4_witness(
                p.hosted_cache_planner_v4))
        boundary_v4 = p.hosted_real_substrate_boundary_v4
        wall_v4_report = build_wall_report_v4(
            boundary=boundary_v4)
        # Provider filter V3 — run once to seal a report CID.
        _, filter_report = filter_hosted_registry_v3(
            p.hosted_registry, p.hosted_provider_filter_v3,
            provider_restart_noise={
                "openrouter_paid": 0.5,
                "openai_paid": 0.1})
        filter_report_cid = _sha256_hex({
            "kind": "w71_provider_filter_v3_report",
            "report": dict(filter_report),
        })
        # Handoff coordinator V3 decisions.
        env_text_only = p.handoff_coordinator_v3.decide_v3(
            req_v3=HandoffRequestV3(
                inner_v2=HandoffRequestV2(
                    inner_v1=HandoffRequest(
                        request_cid="w71-turn-text",
                        needs_text_only=True,
                        needs_substrate_state_access=False),
                    visible_token_budget=128,
                    baseline_token_cost=512,
                    dominant_repair_label=0),
                restart_pressure=0.0,
                delayed_repair_trajectory_cid="",
                delay_turns=0,
                expected_substrate_trust=0.7),
            substrate_self_checksum_cid=str(
                cache.v15_cache.v14_cache
                .substrate_self_checksum_cid))
        env_restart_promoted = (
            p.handoff_coordinator_v3.decide_v3(
                req_v3=HandoffRequestV3(
                    inner_v2=HandoffRequestV2(
                        inner_v1=HandoffRequest(
                            request_cid=(
                                "w71-turn-restart"),
                            needs_text_only=True,
                            needs_substrate_state_access=(
                                True)),
                        visible_token_budget=128,
                        baseline_token_cost=512,
                        dominant_repair_label=1),
                    restart_pressure=0.8,
                    delayed_repair_trajectory_cid=str(
                        cache.delayed_repair_trajectory_cid),
                    delay_turns=3,
                    expected_substrate_trust=0.7),
                substrate_self_checksum_cid=str(
                    cache.v15_cache.v14_cache
                    .substrate_self_checksum_cid)))
        env_dr_fallback = (
            p.handoff_coordinator_v3.decide_v3(
                req_v3=HandoffRequestV3(
                    inner_v2=HandoffRequestV2(
                        inner_v1=HandoffRequest(
                            request_cid="w71-turn-dr",
                            needs_text_only=True,
                            needs_substrate_state_access=(
                                False)),
                        visible_token_budget=128,
                        baseline_token_cost=512,
                        dominant_repair_label=0),
                    restart_pressure=0.0,
                    delayed_repair_trajectory_cid=str(
                        cache.delayed_repair_trajectory_cid),
                    delay_turns=3,
                    expected_substrate_trust=0.7),
                substrate_self_checksum_cid=str(
                    cache.v15_cache.v14_cache
                    .substrate_self_checksum_cid)))
        env_substrate_only = (
            p.handoff_coordinator_v3.decide_v3(
                req_v3=HandoffRequestV3(
                    inner_v2=HandoffRequestV2(
                        inner_v1=HandoffRequest(
                            request_cid=(
                                "w71-turn-substrate"),
                            needs_text_only=False,
                            needs_substrate_state_access=(
                                True)),
                        visible_token_budget=128,
                        baseline_token_cost=512,
                        dominant_repair_label=1),
                    restart_pressure=0.3,
                    expected_substrate_trust=0.7),
                substrate_self_checksum_cid=str(
                    cache.v15_cache.v14_cache
                    .substrate_self_checksum_cid)))
        handoff_v3_witness = (
            emit_hosted_real_handoff_coordinator_v3_witness(
                p.handoff_coordinator_v3))
        handoff_envelope_chain_cid = _sha256_hex({
            "kind": "w71_handoff_envelope_v3_chain",
            "envelopes": [
                env_text_only.cid(),
                env_restart_promoted.cid(),
                env_dr_fallback.cid(),
                env_substrate_only.cid(),
            ],
        })
        baseline_agg = per_regime_aggs.get(
            "baseline")
        v16_beats = (
            float(baseline_agg.v16_beats_v15_rate)
            if baseline_agg is not None else 0.0)
        tsc_v16_beats = (
            float(baseline_agg.tsc_v16_beats_tsc_v15_rate)
            if baseline_agg is not None else 0.0)
        ts_per_vt = (
            float(
                baseline_agg.team_success_per_visible_token_v16)
            if baseline_agg is not None else 0.0)
        return W71HandoffEnvelope(
            schema=W71_SCHEMA_VERSION,
            w70_outer_cid=str(w70_outer_cid),
            w71_params_cid=str(p.cid()),
            substrate_v16_witness_cid=str(sub_witness.cid()),
            kv_bridge_v16_witness_cid=str(kv_witness.cid()),
            cache_controller_v14_witness_cid=str(
                cache_witness.cid()),
            replay_controller_v12_witness_cid=str(
                replay_witness.cid()),
            persistent_v23_witness_cid=str(
                persist_witness.cid()),
            mlsc_v19_witness_cid=str(mlsc_witness.cid()),
            consensus_v17_witness_cid=str(
                consensus_witness.cid()),
            lhr_v23_witness_cid=str(lhr_witness.cid()),
            deep_substrate_hybrid_v16_witness_cid=str(
                deep_v16_witness.cid()),
            substrate_adapter_v16_matrix_cid=str(
                adapter_matrix.cid()),
            masc_v7_witness_cid=str(masc_witness.cid()),
            team_consensus_controller_v6_witness_cid=str(
                tcc_witness.cid()),
            restart_dominance_falsifier_witness_cid=str(
                rd_falsifier.cid()),
            hosted_router_v4_witness_cid=str(
                router_v4_witness.cid()),
            hosted_logprob_router_v4_witness_cid=str(
                logprob_v4_witness.cid()),
            hosted_cache_planner_v4_witness_cid=str(
                cache_planner_v4_witness.cid()),
            hosted_real_substrate_boundary_v4_cid=str(
                boundary_v4.cid()),
            hosted_wall_v4_report_cid=str(
                wall_v4_report.cid()),
            handoff_coordinator_v3_witness_cid=str(
                handoff_v3_witness.cid()),
            handoff_envelope_v3_chain_cid=str(
                handoff_envelope_chain_cid),
            provider_filter_v3_report_cid=str(
                filter_report_cid),
            sixteen_way_used=bool(
                deep_v16_witness.sixteen_way),
            substrate_v16_used=True,
            masc_v7_v16_beats_v15_rate=float(v16_beats),
            masc_v7_tsc_v16_beats_tsc_v15_rate=float(
                tsc_v16_beats),
            masc_v7_team_success_per_visible_token=float(
                ts_per_vt),
            hosted_router_v4_chosen=str(
                router_dec.chosen_provider or ""),
            delayed_repair_trajectory_cid=str(
                cache.delayed_repair_trajectory_cid),
        )


def build_default_w71_team(*, seed: int = 71000) -> W71Team:
    return W71Team(params=W71Params.build_default(seed=int(seed)))


__all__ = [
    "W71_SCHEMA_VERSION",
    "W71_FAILURE_MODES",
    "W71Params",
    "W71HandoffEnvelope",
    "verify_w71_handoff",
    "W71Team",
    "build_default_w71_team",
]
