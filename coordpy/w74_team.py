"""W74 — Stronger Compound-Repair / Replacement-After-Delayed-
Repair Budget-Primary Two-Plane Multi-Agent Substrate team.

The ``W74Team`` orchestrator strictly wraps the ``W73Team`` and
adds the W74 mechanism modules organised across two planes plus
the new **compound-aware Plane A↔B handoff coordinator V6**:

**Plane B — Real substrate (in-repo, V19 stack):**

* M1  ``tiny_substrate_v19``           (21-layer, 3 new V19 axes)
* M2  ``kv_bridge_v19``                (15-target ridge + 120-dim
                                        compound-repair fingerprint
                                        + compound-pressure
                                        falsifier)
* M3  ``cache_controller_v17``         (14-objective ridge + per-
                                        role 15-dim compound-
                                        pressure head)
* M4  ``replay_controller_v15``        (22 regimes + 12-label
                                        compound-aware routing
                                        head)
* M5  ``deep_substrate_hybrid_v19``    (19-way bidirectional loop)
* M6  ``substrate_adapter_v19``        (substrate_v19_full tier)
* M7  ``persistent_latent_v26``        (25 layers, 23rd carrier,
                                        max_chain_walk_depth=2097152)
* M8  ``long_horizon_retention_v26``   (25 heads, max_k=832)
* M9  ``mergeable_latent_capsule_v22`` (compound-repair-trajectory
                                        chain + delayed-repair
                                        chain)
* M10 ``consensus_fallback_controller_v20`` (34-stage chain)
* M11 ``multi_agent_substrate_coordinator_v10`` (22-policy, 14-
                                                 regime MASC V10)
* M12 ``team_consensus_controller_v9`` (compound-pressure +
                                        compound-repair-after-DRTR
                                        arbiters)

**Plane A — Hosted control plane V7 (honest, no substrate):**

* H1  ``hosted_router_controller_v7``  (compound-pressure
                                        weighting + compound-
                                        repair-after-DRTR match)
* H2  ``hosted_logprob_router_v7``     (compound-aware abstain
                                        floor + per-budget+restart+
                                        rejoin+replacement+compound
                                        tiebreak)
* H3  ``hosted_cache_aware_planner_v7``(five-layer rotated prefix +
                                        ≥ 85 % savings 16×8 hit=1)
* H4  ``hosted_cost_planner_v7``       (cost-per-compound-success-
                                        under-budget + abstain-
                                        when-compound-pressure-
                                        violated)
* H5  ``hosted_real_substrate_boundary_v7`` (wall V7, 34 blocked
                                             axes)
* H6  ``hosted_real_handoff_coordinator_v6`` (the **new compound-
                                              aware Plane A↔B
                                              bridge** — V6
                                              envelopes + compound
                                              falsifier + cross-
                                              plane savings)
* H7  ``hosted_provider_filter_v6``    (compound-aware provider
                                        filter)

Per-turn it emits 19 W74 module witness CIDs (12 Plane B + 7 Plane
A V7) and a V6 handoff envelope CID, sealing them into a
``W74HandoffEnvelope`` whose ``w73_outer_cid`` carries forward the
W73 envelope byte-for-byte.

Honest scope (W74)
------------------

* Plane A V7 operates at the hosted text/logprob/prefix-cache
  surface. It does NOT pierce hidden state / KV / attention.
  ``W74-L-HOSTED-V7-NO-SUBSTRATE-CAP``.
* Plane B is the in-repo V19 NumPy runtime. We do NOT bridge to
  third-party hosted models.
  ``W74-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
* W74 fits closed-form ridge parameters in three new places on top
  of W73's 67: cache V17 fourteen-objective; cache V17 per-role
  compound-pressure; replay V15 compound-aware routing head; KV
  V19 fifteen-target. Total **70 closed-form ridge solves** across
  W61..W74. No autograd, no SGD, no GPU.
* Trivial passthrough preserved: when ``W74Params.build_trivial()``
  is used the W74 envelope's internal ``w73_outer_cid`` carries
  the supplied W73 outer CID exactly.
* The handoff coordinator V6 preserves the wall: a content-
  addressed V6 envelope says which plane handled each turn under
  the compound-aware score; it does NOT cross the substrate
  boundary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .cache_controller_v17 import (
    CacheControllerV17, emit_cache_controller_v17_witness,
    fit_per_role_compound_pressure_head_v17,
    fit_fourteen_objective_ridge_v17,
)
from .consensus_fallback_controller_v20 import (
    ConsensusFallbackControllerV20,
    emit_consensus_v20_witness,
)
from .deep_substrate_hybrid_v18 import (
    DeepSubstrateHybridV18ForwardWitness,
)
from .deep_substrate_hybrid_v19 import (
    DeepSubstrateHybridV19,
    deep_substrate_hybrid_v19_forward,
)
from .hosted_cache_aware_planner_v7 import (
    HostedCacheAwarePlannerV7,
    emit_hosted_cache_aware_planner_v7_witness,
)
from .hosted_cost_planner_v7 import HostedCostPlanSpecV7
from .hosted_logprob_router_v7 import (
    HostedLogprobRouterV7,
    emit_hosted_logprob_router_v7_witness,
)
from .hosted_provider_filter_v6 import (
    HostedProviderFilterSpecV6,
    filter_hosted_registry_v6,
)
from .hosted_real_handoff_coordinator_v6 import (
    HandoffRequestV6, HostedRealHandoffCoordinatorV6,
    emit_hosted_real_handoff_coordinator_v6_witness,
    hosted_real_handoff_v6_compound_aware_savings,
)
from .hosted_real_handoff_coordinator_v5 import HandoffRequestV5
from .hosted_real_handoff_coordinator_v4 import HandoffRequestV4
from .hosted_real_handoff_coordinator_v3 import HandoffRequestV3
from .hosted_real_handoff_coordinator_v2 import HandoffRequestV2
from .hosted_real_handoff_coordinator import HandoffRequest
from .hosted_real_substrate_boundary_v7 import (
    HostedRealSubstrateBoundaryV7,
    build_default_hosted_real_substrate_boundary_v7,
    build_wall_report_v7,
)
from .hosted_router_controller import (
    HostedProviderRegistry, HostedRoutingRequest,
    default_hosted_registry,
)
from .hosted_router_controller_v2 import HostedRoutingRequestV2
from .hosted_router_controller_v3 import HostedRoutingRequestV3
from .hosted_router_controller_v4 import HostedRoutingRequestV4
from .hosted_router_controller_v5 import HostedRoutingRequestV5
from .hosted_router_controller_v6 import HostedRoutingRequestV6
from .hosted_router_controller_v7 import (
    HostedRouterControllerV7, HostedRoutingRequestV7,
    emit_hosted_router_controller_v7_witness,
)
from .kv_bridge_v18 import KVBridgeV18Projection
from .kv_bridge_v19 import (
    KVBridgeV19Projection,
    compute_compound_repair_fingerprint_v19,
    emit_kv_bridge_v19_witness,
    probe_kv_bridge_v19_compound_pressure_falsifier,
)
from .long_horizon_retention_v26 import (
    LongHorizonReconstructionV26Head,
    emit_lhr_v26_witness,
)
from .mergeable_latent_capsule_v22 import (
    MergeOperatorV22, emit_mlsc_v22_witness, wrap_v21_as_v22,
)
from .multi_agent_substrate_coordinator_v10 import (
    MultiAgentSubstrateCoordinatorV10,
    W74_MASC_V10_REGIMES,
    emit_multi_agent_substrate_coordinator_v10_witness,
)
from .persistent_latent_v26 import (
    PersistentLatentStateV26Chain,
    emit_persistent_v26_witness,
)
from .replay_controller_v15 import (
    ReplayControllerV15,
    W74_COMPOUND_AWARE_ROUTING_LABELS,
    W74_REPLAY_REGIMES_V15,
    emit_replay_controller_v15_witness,
    fit_replay_controller_v15_per_role,
    fit_replay_v15_compound_aware_routing_head,
)
from .replay_controller import ReplayCandidate
from .substrate_adapter_v19 import (
    W74_SUBSTRATE_TIER_SUBSTRATE_V19_FULL,
    probe_all_v19_adapters,
)
from .team_consensus_controller_v9 import (
    TeamConsensusControllerV9,
    emit_team_consensus_controller_v9_witness,
)
from .tiny_substrate_v19 import (
    TinyV19SubstrateParams,
    build_default_tiny_substrate_v19,
    emit_tiny_substrate_v19_forward_witness,
    forward_tiny_substrate_v19,
    record_compound_failure_window_v19,
    record_delayed_repair_event_v19,
    tokenize_bytes_v19,
)
from .w73_team import (
    W73HandoffEnvelope, W73Params, W73Team,
)


W74_SCHEMA_VERSION: str = "coordpy.w74_team.v1"

W74_FAILURE_MODES: tuple[str, ...] = (
    "w74_outer_envelope_schema_mismatch",
    "w74_outer_envelope_w73_outer_cid_drift",
    "w74_outer_envelope_w74_params_cid_drift",
    "w74_outer_envelope_witness_cid_drift",
    "w74_substrate_v19_n_layers_off",
    "w74_substrate_v19_compound_repair_trajectory_cid_off",
    "w74_substrate_v19_compound_repair_rate_per_layer_shape_off",
    "w74_substrate_v19_compound_pressure_gate_shape_off",
    "w74_kv_bridge_v19_n_targets_off",
    "w74_kv_bridge_v19_compound_pressure_falsifier_off",
    "w74_cache_v17_fourteen_objective_off",
    "w74_replay_v15_regime_count_off",
    "w74_replay_v15_compound_aware_routing_count_off",
    "w74_consensus_v20_stage_count_off",
    "w74_lhr_v26_max_k_off",
    "w74_lhr_v26_n_heads_off",
    "w74_persistent_v26_n_layers_off",
    "w74_substrate_adapter_v19_tier_off",
    "w74_masc_v10_v19_beats_v18_rate_under_threshold",
    "w74_masc_v10_tsc_v19_beats_tsc_v18_rate_under_threshold",
    "w74_masc_v10_compound_regime_inferior_to_baseline",
    "w74_hosted_router_v7_decision_not_deterministic",
    "w74_hosted_logprob_v7_abstain_floor_off",
    "w74_hosted_cache_aware_v7_savings_below_85_percent",
    "w74_hosted_cost_planner_v7_no_eligible",
    "w74_hosted_real_substrate_boundary_v7_blocked_axis_satisfied",
    "w74_nineteen_way_loop_not_observed",
    "w74_handoff_coordinator_v6_inconsistent",
    "w74_handoff_v6_cross_plane_savings_below_82_percent",
    "w74_team_consensus_v9_no_decisions",
    "w74_handoff_v6_compound_alignment_off",
    "w74_handoff_envelope_v6_chain_cid_drift",
    "w74_inner_v73_envelope_invariant_off",
    "w74_handoff_v6_compound_repair_drtr_fallback_off",
    "w74_hosted_boundary_v7_blocked_axes_below_34",
    "w74_v19_substrate_self_checksum_cid_off",
    "w74_compound_repair_trajectory_cid_drift",
    "w74_mlsc_v22_compound_repair_trajectory_chain_off",
    "w74_v10_team_success_per_visible_token_below_floor",
    "w74_v10_visible_tokens_savings_below_65_percent",
    "w74_v10_compound_regime_v19_beats_v18_below_threshold",
    "w74_substrate_v19_compound_repair_trajectory_chain_synthetic",
    "w74_inner_v19_falsifier_kind_off",
    "w74_handoff_v6_envelope_cmp_alignment_off",
    "w74_hosted_router_v7_per_routing_cid_off",
    "w74_consensus_v20_compound_pressure_arbiter_off",
    "w74_consensus_v20_compound_repair_drtr_arbiter_off",
    "w74_tcc_v9_compound_pressure_arbiter_off",
    "w74_tcc_v9_compound_repair_drtr_arbiter_off",
    "w74_cache_v17_per_role_compound_pressure_head_off",
    "w74_kv_bridge_v19_compound_repair_fingerprint_off",
    "w74_substrate_v19_delayed_repair_events_off",
    "w74_provider_filter_v6_pressure_drop_off",
    "w74_handoff_v6_cmp_alignment_off",
    "w74_handoff_v6_decision_label_off",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class W74Params:
    substrate_v19: TinyV19SubstrateParams | None
    kv_bridge_v19: KVBridgeV19Projection | None
    cache_controller_v17: CacheControllerV17 | None
    replay_controller_v15: ReplayControllerV15 | None
    consensus_v20: ConsensusFallbackControllerV20 | None
    lhr_v26: LongHorizonReconstructionV26Head | None
    deep_substrate_hybrid_v19: DeepSubstrateHybridV19 | None
    mlsc_v22_operator: MergeOperatorV22 | None
    multi_agent_coordinator_v10: (
        MultiAgentSubstrateCoordinatorV10 | None)
    team_consensus_controller_v9: (
        TeamConsensusControllerV9 | None)
    hosted_registry: HostedProviderRegistry | None
    hosted_router_v7: HostedRouterControllerV7 | None
    hosted_logprob_router_v7: HostedLogprobRouterV7 | None
    hosted_cache_planner_v7: HostedCacheAwarePlannerV7 | None
    hosted_real_substrate_boundary_v7: (
        HostedRealSubstrateBoundaryV7 | None)
    handoff_coordinator_v6: (
        HostedRealHandoffCoordinatorV6 | None)
    hosted_provider_filter_v6: (
        HostedProviderFilterSpecV6 | None)
    w73_params: W73Params | None
    enabled: bool = True
    masc_v10_n_seeds: int = 6

    @classmethod
    def build_trivial(cls) -> "W74Params":
        return cls(
            substrate_v19=None,
            kv_bridge_v19=None,
            cache_controller_v17=None,
            replay_controller_v15=None,
            consensus_v20=None, lhr_v26=None,
            deep_substrate_hybrid_v19=None,
            mlsc_v22_operator=None,
            multi_agent_coordinator_v10=None,
            team_consensus_controller_v9=None,
            hosted_registry=None,
            hosted_router_v7=None,
            hosted_logprob_router_v7=None,
            hosted_cache_planner_v7=None,
            hosted_real_substrate_boundary_v7=None,
            handoff_coordinator_v6=None,
            hosted_provider_filter_v6=None,
            w73_params=None,
            enabled=False,
        )

    @classmethod
    def build_default(
            cls, *, seed: int = 74000,
    ) -> "W74Params":
        sub_v19 = build_default_tiny_substrate_v19(
            seed=int(seed) + 1)
        # KV V19 projection chain.
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
        from .kv_bridge_v17 import KVBridgeV17Projection
        cfg = (
            sub_v19.config.v18.v17.v16.v15.v14.v13.v12.v11.v10.v9)
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
        kv_b19 = KVBridgeV19Projection.init_from_v18(
            kv_b18, seed_v19=int(seed) + 23)
        cc17 = CacheControllerV17.init(
            fit_seed=int(seed) + 32)
        import numpy as _np
        rng = _np.random.default_rng(int(seed) + 33)
        X = rng.standard_normal((10, 4))
        cc17, _ = fit_fourteen_objective_ridge_v17(
            controller=cc17, train_features=X.tolist(),
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
                + X[:, 3] * 0.3).tolist(),
            target_compound_repair=(
                X[:, 0] * 0.25 + X[:, 1] * 0.25
                + X[:, 2] * 0.25 + X[:, 3] * 0.25).tolist())
        X15 = rng.standard_normal((10, 15))
        cc17, _ = fit_per_role_compound_pressure_head_v17(
            controller=cc17, role="planner",
            train_features=X15.tolist(),
            target_compound_priorities=(
                X15[:, 0] * 0.3
                + X15[:, 14] * 0.4).tolist())
        # Replay V15.
        rcv15 = ReplayControllerV15.init()
        v15_cands = {
            r: [ReplayCandidate(
                100, 1000, 50, 0.1, 0.0, 0.3,
                True, True, 0)]
            for r in W74_REPLAY_REGIMES_V15}
        v15_decs = {
            r: ["choose_reuse"]
            for r in W74_REPLAY_REGIMES_V15}
        rcv15, _ = fit_replay_controller_v15_per_role(
            controller=rcv15, role="planner",
            train_candidates_per_regime=v15_cands,
            train_decisions_per_regime=v15_decs)
        X_team = rng.standard_normal((50, 10))
        labs: list[str] = []
        for i in range(50):
            lab_idx = (
                i % len(W74_COMPOUND_AWARE_ROUTING_LABELS))
            labs.append(
                W74_COMPOUND_AWARE_ROUTING_LABELS[lab_idx])
        rcv15, _ = fit_replay_v15_compound_aware_routing_head(
            controller=rcv15,
            train_team_features=X_team.tolist(),
            train_routing_labels=labs)
        consensus_v20 = ConsensusFallbackControllerV20.init(
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
            replacement_after_ctr_threshold=0.5,
            compound_repair_threshold=0.5,
            compound_repair_drtr_threshold=0.5)
        lhr26 = LongHorizonReconstructionV26Head.init(
            seed=int(seed) + 40)
        deep_v19 = DeepSubstrateHybridV19()
        mlsc_v22_op = MergeOperatorV22()
        masc_v10 = MultiAgentSubstrateCoordinatorV10()
        tcc_v9 = TeamConsensusControllerV9()
        reg = default_hosted_registry()
        hosted_router_v7 = HostedRouterControllerV7.init(
            reg, {
                "openrouter_paid": 0.85,
                "openai_paid": 0.92,
            })
        hosted_logprob_router_v7 = HostedLogprobRouterV7()
        hosted_cache_planner_v7 = HostedCacheAwarePlannerV7()
        boundary_v7 = (
            build_default_hosted_real_substrate_boundary_v7())
        handoff_coord_v6 = HostedRealHandoffCoordinatorV6(
            boundary_v7=boundary_v7)
        # Provider filter V6 — pre-configured compound-aware spec.
        from .hosted_provider_filter_v5 import (
            HostedProviderFilterSpecV5,
        )
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
        provider_filter_v6 = HostedProviderFilterSpecV6(
            inner_v5=HostedProviderFilterSpecV5(
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
                    "text_only": 0.30}),
            compound_pressure=0.7,
            compound_pressure_floor=0.5,
            max_compound_noise_per_provider={
                "openrouter_paid": 0.18,
                "openai_paid": 1.0},
            compound_tier_weights={
                "logprobs_and_prefix_cache": 1.0,
                "logprobs": 0.55,
                "prefix_cache": 0.45,
                "text_only": 0.25})
        # W73 inner params for envelope chaining.
        w73_params = W73Params.build_default(
            seed=int(seed) - 1000)
        return cls(
            substrate_v19=sub_v19,
            kv_bridge_v19=kv_b19,
            cache_controller_v17=cc17,
            replay_controller_v15=rcv15,
            consensus_v20=consensus_v20,
            lhr_v26=lhr26,
            deep_substrate_hybrid_v19=deep_v19,
            mlsc_v22_operator=mlsc_v22_op,
            multi_agent_coordinator_v10=masc_v10,
            team_consensus_controller_v9=tcc_v9,
            hosted_registry=reg,
            hosted_router_v7=hosted_router_v7,
            hosted_logprob_router_v7=hosted_logprob_router_v7,
            hosted_cache_planner_v7=hosted_cache_planner_v7,
            hosted_real_substrate_boundary_v7=boundary_v7,
            handoff_coordinator_v6=handoff_coord_v6,
            hosted_provider_filter_v6=provider_filter_v6,
            w73_params=w73_params,
            enabled=True,
            masc_v10_n_seeds=5,
        )

    def to_dict(self) -> dict[str, Any]:
        def _cid_or_empty(x: Any) -> str:
            return str(x.cid()) if x is not None else ""
        return {
            "schema": W74_SCHEMA_VERSION,
            "kind": "w74_params",
            "substrate_v19_cid": _cid_or_empty(
                self.substrate_v19),
            "kv_bridge_v19_cid": _cid_or_empty(
                self.kv_bridge_v19),
            "cache_controller_v17_cid": _cid_or_empty(
                self.cache_controller_v17),
            "replay_controller_v15_cid": _cid_or_empty(
                self.replay_controller_v15),
            "consensus_v20_cid": _cid_or_empty(
                self.consensus_v20),
            "lhr_v26_cid": _cid_or_empty(self.lhr_v26),
            "deep_substrate_hybrid_v19_cid": _cid_or_empty(
                self.deep_substrate_hybrid_v19),
            "mlsc_v22_operator_cid": _cid_or_empty(
                self.mlsc_v22_operator),
            "multi_agent_coordinator_v10_cid": _cid_or_empty(
                self.multi_agent_coordinator_v10),
            "team_consensus_controller_v9_cid": _cid_or_empty(
                self.team_consensus_controller_v9),
            "hosted_registry_cid": _cid_or_empty(
                self.hosted_registry),
            "hosted_router_v7_cid": _cid_or_empty(
                self.hosted_router_v7),
            "hosted_logprob_router_v7_cid": _cid_or_empty(
                self.hosted_logprob_router_v7),
            "hosted_cache_planner_v7_cid": _cid_or_empty(
                self.hosted_cache_planner_v7),
            "hosted_real_substrate_boundary_v7_cid":
                _cid_or_empty(
                    self.hosted_real_substrate_boundary_v7),
            "handoff_coordinator_v6_cid": _cid_or_empty(
                self.handoff_coordinator_v6),
            "hosted_provider_filter_v6_cid": _cid_or_empty(
                self.hosted_provider_filter_v6),
            "w73_params_cid": _cid_or_empty(self.w73_params),
            "enabled": bool(self.enabled),
            "masc_v10_n_seeds": int(self.masc_v10_n_seeds),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w74_params",
            "params": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class W74HandoffEnvelope:
    schema: str
    w73_outer_cid: str
    w74_params_cid: str
    substrate_v19_witness_cid: str
    kv_bridge_v19_witness_cid: str
    cache_controller_v17_witness_cid: str
    replay_controller_v15_witness_cid: str
    persistent_v26_witness_cid: str
    mlsc_v22_witness_cid: str
    consensus_v20_witness_cid: str
    lhr_v26_witness_cid: str
    deep_substrate_hybrid_v19_witness_cid: str
    substrate_adapter_v19_matrix_cid: str
    masc_v10_witness_cid: str
    team_consensus_controller_v9_witness_cid: str
    compound_pressure_falsifier_witness_cid: str
    hosted_router_v7_witness_cid: str
    hosted_logprob_router_v7_witness_cid: str
    hosted_cache_planner_v7_witness_cid: str
    hosted_real_substrate_boundary_v7_cid: str
    hosted_wall_v7_report_cid: str
    handoff_coordinator_v6_witness_cid: str
    handoff_envelope_v6_chain_cid: str
    provider_filter_v6_report_cid: str
    nineteen_way_used: bool
    substrate_v19_used: bool
    masc_v10_v19_beats_v18_rate: float
    masc_v10_tsc_v19_beats_tsc_v18_rate: float
    masc_v10_team_success_per_visible_token: float
    hosted_router_v7_chosen: str
    compound_repair_trajectory_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "w73_outer_cid": str(self.w73_outer_cid),
            "w74_params_cid": str(self.w74_params_cid),
            "substrate_v19_witness_cid": str(
                self.substrate_v19_witness_cid),
            "kv_bridge_v19_witness_cid": str(
                self.kv_bridge_v19_witness_cid),
            "cache_controller_v17_witness_cid": str(
                self.cache_controller_v17_witness_cid),
            "replay_controller_v15_witness_cid": str(
                self.replay_controller_v15_witness_cid),
            "persistent_v26_witness_cid": str(
                self.persistent_v26_witness_cid),
            "mlsc_v22_witness_cid": str(
                self.mlsc_v22_witness_cid),
            "consensus_v20_witness_cid": str(
                self.consensus_v20_witness_cid),
            "lhr_v26_witness_cid": str(
                self.lhr_v26_witness_cid),
            "deep_substrate_hybrid_v19_witness_cid": str(
                self.deep_substrate_hybrid_v19_witness_cid),
            "substrate_adapter_v19_matrix_cid": str(
                self.substrate_adapter_v19_matrix_cid),
            "masc_v10_witness_cid": str(
                self.masc_v10_witness_cid),
            "team_consensus_controller_v9_witness_cid": str(
                self.team_consensus_controller_v9_witness_cid),
            "compound_pressure_falsifier_witness_cid": str(
                self.compound_pressure_falsifier_witness_cid),
            "hosted_router_v7_witness_cid": str(
                self.hosted_router_v7_witness_cid),
            "hosted_logprob_router_v7_witness_cid": str(
                self.hosted_logprob_router_v7_witness_cid),
            "hosted_cache_planner_v7_witness_cid": str(
                self.hosted_cache_planner_v7_witness_cid),
            "hosted_real_substrate_boundary_v7_cid": str(
                self.hosted_real_substrate_boundary_v7_cid),
            "hosted_wall_v7_report_cid": str(
                self.hosted_wall_v7_report_cid),
            "handoff_coordinator_v6_witness_cid": str(
                self.handoff_coordinator_v6_witness_cid),
            "handoff_envelope_v6_chain_cid": str(
                self.handoff_envelope_v6_chain_cid),
            "provider_filter_v6_report_cid": str(
                self.provider_filter_v6_report_cid),
            "nineteen_way_used": bool(self.nineteen_way_used),
            "substrate_v19_used": bool(self.substrate_v19_used),
            "masc_v10_v19_beats_v18_rate": float(round(
                self.masc_v10_v19_beats_v18_rate, 12)),
            "masc_v10_tsc_v19_beats_tsc_v18_rate": float(round(
                self.masc_v10_tsc_v19_beats_tsc_v18_rate, 12)),
            "masc_v10_team_success_per_visible_token": float(
                round(
                    self
                    .masc_v10_team_success_per_visible_token,
                    12)),
            "hosted_router_v7_chosen": str(
                self.hosted_router_v7_chosen),
            "compound_repair_trajectory_cid": str(
                self.compound_repair_trajectory_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w74_handoff_envelope",
            "envelope": self.to_dict()})


def verify_w74_handoff(
        envelope: W74HandoffEnvelope,
        params: W74Params,
        w73_outer_cid: str,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if envelope.schema != W74_SCHEMA_VERSION:
        failures.append(
            "w74_outer_envelope_schema_mismatch")
    if envelope.w73_outer_cid != str(w73_outer_cid):
        failures.append(
            "w74_outer_envelope_w73_outer_cid_drift")
    if envelope.w74_params_cid != params.cid():
        failures.append(
            "w74_outer_envelope_w74_params_cid_drift")
    return (len(failures) == 0), failures


@dataclasses.dataclass
class W74Team:
    params: W74Params

    def run_team_turn(
            self, *,
            w73_outer_cid: str,
            ids: Sequence[int] | None = None,
            text: str = "w74",
    ) -> W74HandoffEnvelope:
        p = self.params
        if not p.enabled or p.substrate_v19 is None:
            return W74HandoffEnvelope(
                schema=W74_SCHEMA_VERSION,
                w73_outer_cid=str(w73_outer_cid),
                w74_params_cid=str(p.cid()),
                substrate_v19_witness_cid="",
                kv_bridge_v19_witness_cid="",
                cache_controller_v17_witness_cid="",
                replay_controller_v15_witness_cid="",
                persistent_v26_witness_cid="",
                mlsc_v22_witness_cid="",
                consensus_v20_witness_cid="",
                lhr_v26_witness_cid="",
                deep_substrate_hybrid_v19_witness_cid="",
                substrate_adapter_v19_matrix_cid="",
                masc_v10_witness_cid="",
                team_consensus_controller_v9_witness_cid="",
                compound_pressure_falsifier_witness_cid="",
                hosted_router_v7_witness_cid="",
                hosted_logprob_router_v7_witness_cid="",
                hosted_cache_planner_v7_witness_cid="",
                hosted_real_substrate_boundary_v7_cid="",
                hosted_wall_v7_report_cid="",
                handoff_coordinator_v6_witness_cid="",
                handoff_envelope_v6_chain_cid="",
                provider_filter_v6_report_cid="",
                nineteen_way_used=False,
                substrate_v19_used=False,
                masc_v10_v19_beats_v18_rate=0.0,
                masc_v10_tsc_v19_beats_tsc_v18_rate=0.0,
                masc_v10_team_success_per_visible_token=0.0,
                hosted_router_v7_chosen="",
                compound_repair_trajectory_cid="",
            )
        # Plane B — substrate V19 forward with full event chain.
        token_ids = (
            list(ids) if ids is not None
            else tokenize_bytes_v19(str(text), max_len=16))
        trace, cache = forward_tiny_substrate_v19(
            p.substrate_v19, token_ids,
            visible_token_budget=128.0,
            baseline_token_cost=512.0,
            restart_pressure=0.6,
            rejoin_pressure=0.6,
            replacement_pressure=0.6,
            contradiction_pressure=0.5,
            delayed_repair_pressure=0.6,
            compound_pressure=0.7)
        # Record full event chain on the inner caches.
        from .tiny_substrate_v16 import (
            record_delay_window_v16, record_restart_event_v16,
        )
        from .tiny_substrate_v17 import (
            record_branch_pressure_window_v17,
            record_rejoin_event_v17,
        )
        from .tiny_substrate_v18 import (
            record_contradiction_event_v18,
            record_replacement_event_v18,
            record_replacement_window_v18,
        )
        record_restart_event_v16(
            cache.v18_cache.v17_cache.v16_cache, turn=1,
            restart_kind="agent_restart",
            role="planner")
        record_delay_window_v16(
            cache.v18_cache.v17_cache.v16_cache, restart_turn=1,
            repair_turn=4, delay_turns=3, role="planner")
        record_rejoin_event_v17(
            cache.v18_cache.v17_cache, turn=2,
            rejoin_kind="branch_rejoin",
            branch_id="main", role="planner")
        record_branch_pressure_window_v17(
            cache.v18_cache.v17_cache, restart_turn=1,
            rejoin_turn=5, rejoin_lag_turns=4, branch_id="main",
            role="planner")
        record_contradiction_event_v18(
            cache.v18_cache, turn=2,
            contradiction_kind="fact_contradiction",
            role="planner", branch_id="main")
        record_replacement_event_v18(
            cache.v18_cache, turn=3,
            replacement_kind="agent_replacement",
            role="planner", new_role="planner_fresh")
        record_replacement_window_v18(
            cache.v18_cache, contradiction_turn=2,
            replacement_turn=3, rejoin_turn=8,
            replacement_lag_turns=5, role="planner",
            branch_id="main")
        record_delayed_repair_event_v19(
            cache, turn=2,
            delayed_kind="delayed_repair", role="planner")
        record_compound_failure_window_v19(
            cache, delayed_repair_turn=2,
            replacement_turn=5, rejoin_turn=12,
            compound_window_turns=10, role="planner",
            branch_id="main")
        # Re-run forward to produce the compound-repair CID over
        # the recorded events.
        trace, cache = forward_tiny_substrate_v19(
            p.substrate_v19, token_ids,
            v19_kv_cache=cache,
            visible_token_budget=128.0,
            baseline_token_cost=512.0,
            restart_pressure=0.6,
            rejoin_pressure=0.6,
            replacement_pressure=0.6,
            contradiction_pressure=0.5,
            delayed_repair_pressure=0.6,
            compound_pressure=0.7)
        sub_witness = emit_tiny_substrate_v19_forward_witness(
            trace, cache)
        # KV V19 witnesses.
        cmp_falsifier = (
            probe_kv_bridge_v19_compound_pressure_falsifier(
                compound_pressure_flag=1))
        crf = compute_compound_repair_fingerprint_v19(
            role="planner",
            repair_trajectory_cid=str(
                cache.v18_cache.v17_cache.v16_cache.v15_cache
                .repair_trajectory_cid),
            delayed_repair_trajectory_cid=str(
                cache.v18_cache.v17_cache.v16_cache
                .delayed_repair_trajectory_cid),
            restart_repair_trajectory_cid=str(
                cache.v18_cache.v17_cache
                .restart_repair_trajectory_cid),
            replacement_repair_trajectory_cid=str(
                cache.v18_cache.replacement_repair_trajectory_cid),
            compound_repair_trajectory_cid=str(
                cache.compound_repair_trajectory_cid),
            dominant_repair_label=1,
            restart_count=int(len(
                cache.v18_cache.v17_cache.v16_cache
                .restart_events)),
            rejoin_count=int(len(
                cache.v18_cache.v17_cache.rejoin_events)),
            replacement_count=int(len(
                cache.v18_cache.replacement_events)),
            contradiction_count=int(len(
                cache.v18_cache.contradiction_events)),
            delayed_repair_count=int(len(
                cache.delayed_repair_events)),
            visible_token_budget=128.0,
            baseline_cost=512.0,
            delay_turns=3,
            rejoin_lag_turns=4,
            replacement_lag_turns=5,
            compound_window_turns=10)
        kv_witness = emit_kv_bridge_v19_witness(
            projection=p.kv_bridge_v19,
            compound_pressure_falsifier=cmp_falsifier,
            compound_repair_fingerprint=crf)
        cache_witness = emit_cache_controller_v17_witness(
            controller=p.cache_controller_v17)
        replay_witness = emit_replay_controller_v15_witness(
            p.replay_controller_v15)
        persist_chain = (
            PersistentLatentStateV26Chain.empty())
        persist_witness = emit_persistent_v26_witness(
            persist_chain)
        # MLSC V22 — wrap a trivial V21 capsule up the chain.
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
        from .mergeable_latent_capsule_v21 import (
            wrap_v20_as_v21)
        v3 = make_root_capsule_v3(
            branch_id="w74_smoke",
            payload=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            fact_tags=("w74",), confidence=0.9, trust=0.9,
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
                str(cache.v18_cache.v17_cache
                    .restart_repair_trajectory_cid),),
            rejoin_pressure_chain=(
                f"rj_{int(len(cache.v18_cache.v17_cache.rejoin_events))}",
            ),
        )
        v21 = wrap_v20_as_v21(
            v20,
            replacement_repair_trajectory_chain=(
                str(cache.v18_cache
                    .replacement_repair_trajectory_cid),),
            contradiction_chain=(
                f"ct_{int(len(cache.v18_cache.contradiction_events))}",),
        )
        v22 = wrap_v21_as_v22(
            v21,
            compound_repair_trajectory_chain=(
                str(cache.compound_repair_trajectory_cid),),
            delayed_repair_chain=(
                f"dr_{int(len(cache.delayed_repair_events))}",),
        )
        mlsc_witness = emit_mlsc_v22_witness(v22)
        consensus_witness = emit_consensus_v20_witness(
            p.consensus_v20)
        lhr_witness = emit_lhr_v26_witness(
            p.lhr_v26, carrier=[0.1] * 6, k=16,
            partial_contradiction_indicator=[0.5] * 8,
            multi_branch_rejoin_indicator=[0.6] * 8,
            repair_dominance_indicator=[0.7] * 7,
            restart_indicator=[0.5] * 8,
            rejoin_indicator=[0.6] * 8,
            replacement_indicator=[0.7] * 8,
            compound_indicator=[0.8] * 8)
        # Deep substrate hybrid V19 — fold the V18 witness as a
        # pre-condition.
        v18_witness = DeepSubstrateHybridV18ForwardWitness(
            schema="coordpy.deep_substrate_hybrid_v18.v1",
            hybrid_cid="",
            inner_v17_witness_cid="",
            eighteen_way=True,
            cache_controller_v16_fired=True,
            replay_controller_v14_fired=True,
            replacement_repair_trajectory_active=True,
            replacement_after_ctr_active=True,
            team_consensus_controller_v8_active=True,
            replacement_repair_trajectory_cid=str(
                cache.v18_cache
                .replacement_repair_trajectory_cid),
            replacement_after_ctr_l1=int(
                sub_witness.compound_repair_l1 + 1),
            replacement_pressure_gate_mean=float(
                trace.v18_trace
                .replacement_pressure_gate_per_layer.mean()),
        )
        deep_v19_witness = deep_substrate_hybrid_v19_forward(
            hybrid=p.deep_substrate_hybrid_v19,
            v18_witness=v18_witness,
            cache_controller_v17=p.cache_controller_v17,
            replay_controller_v15=p.replay_controller_v15,
            compound_repair_trajectory_cid=str(
                cache.compound_repair_trajectory_cid),
            compound_repair_l1=int(
                sub_witness.compound_repair_l1),
            compound_pressure_gate_mean=float(
                trace.compound_pressure_gate_per_layer.mean()),
            n_team_consensus_v9_invocations=1)
        adapter_matrix = probe_all_v19_adapters()
        # MASC V10 — run a batch for the envelope (all regimes).
        per_regime_aggs = {}
        for regime in W74_MASC_V10_REGIMES:
            _, agg = p.multi_agent_coordinator_v10.run_batch(
                seeds=list(range(int(p.masc_v10_n_seeds))),
                regime=regime)
            per_regime_aggs[regime] = agg
        masc_witness = (
            emit_multi_agent_substrate_coordinator_v10_witness(
                coordinator=p.multi_agent_coordinator_v10,
                per_regime_aggregate=per_regime_aggs))
        # TCC V9 — fire each new arbiter so the witness counts > 0.
        tcc_v9 = p.team_consensus_controller_v9
        tcc_v9.decide_v9(
            regime=(
                "replacement_after_delayed_repair_under_budget"),
            agent_guesses=[1.0, -1.0, 0.5, 0.2],
            agent_confidences=[0.8, 0.6, 0.7, 0.7],
            substrate_replay_trust=0.7,
            compound_repair_trajectory_cid=str(
                cache.compound_repair_trajectory_cid),
            compound_window_turns=10,
            agent_compound_absorption_scores=[
                0.95, 0.6, 0.5, 0.4])
        tcc_v9.decide_v9(
            regime="baseline",
            agent_guesses=[0.5, 0.5, 0.4, 0.5],
            agent_confidences=[0.8, 0.6, 0.7, 0.7],
            substrate_replay_trust=0.7,
            compound_pressure=0.8,
            agent_compound_recovery_flags=[1, 0, 1, 0])
        tcc_witness = emit_team_consensus_controller_v9_witness(
            tcc_v9)
        # Plane A V7 — hosted.
        planned, _ = (
            p.hosted_cache_planner_v7
            .plan_per_role_five_layer_rotated(
                shared_prefix_text=(
                    "W74 team shared prefix " * 16),
                per_role_blocks={
                    "plan": ["t0", "t1"],
                    "research": ["r0", "r1"],
                    "write": ["w0", "w1"],
                    "review": ["v0", "v1"],
                }))
        # Router V7 — at least one decision so witness is non-empty.
        req_v7 = HostedRoutingRequestV7(
            inner_v6=HostedRoutingRequestV6(
                inner_v5=HostedRoutingRequestV5(
                    inner_v4=HostedRoutingRequestV4(
                        inner_v3=HostedRoutingRequestV3(
                            inner_v2=HostedRoutingRequestV2(
                                inner_v1=HostedRoutingRequest(
                                    request_cid="w74-router-turn",
                                    input_tokens=1000,
                                    expected_output_tokens=300,
                                    require_logprobs=True,
                                    require_prefix_cache=True,
                                    data_policy_required="no_log",
                                    max_latency_ms=2000.0,
                                    max_cost_usd=50.0),
                                weight_cost=1.0,
                                weight_latency=0.5,
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
                weight_replacement_after_ctr_match=0.4),
            compound_pressure=0.7,
            weight_compound_pressure=0.6,
            weight_compound_repair_drtr_match=0.4)
        router_dec = p.hosted_router_v7.decide_v7(req_v7)
        router_v7_witness = (
            emit_hosted_router_controller_v7_witness(
                p.hosted_router_v7))
        logprob_v7_witness = (
            emit_hosted_logprob_router_v7_witness(
                p.hosted_logprob_router_v7))
        cache_planner_v7_witness = (
            emit_hosted_cache_aware_planner_v7_witness(
                p.hosted_cache_planner_v7))
        boundary_v7 = p.hosted_real_substrate_boundary_v7
        wall_v7_report = build_wall_report_v7(
            boundary=boundary_v7)
        # Provider filter V6 — run once to seal a report CID.
        _, filter_report = filter_hosted_registry_v6(
            p.hosted_registry, p.hosted_provider_filter_v6,
            provider_restart_noise={
                "openrouter_paid": 0.5,
                "openai_paid": 0.1},
            provider_rejoin_noise={
                "openrouter_paid": 0.4,
                "openai_paid": 0.1},
            provider_replacement_noise={
                "openrouter_paid": 0.35,
                "openai_paid": 0.05},
            provider_compound_noise={
                "openrouter_paid": 0.30,
                "openai_paid": 0.05})
        filter_report_cid = _sha256_hex({
            "kind": "w74_provider_filter_v6_report",
            "report": dict(filter_report),
        })
        # Handoff coordinator V6 decisions.
        env_text_only = p.handoff_coordinator_v6.decide_v6(
            req_v6=HandoffRequestV6(
                inner_v5=HandoffRequestV5(
                    inner_v4=HandoffRequestV4(
                        inner_v3=HandoffRequestV3(
                            inner_v2=HandoffRequestV2(
                                inner_v1=HandoffRequest(
                                    request_cid="w74-turn-text",
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
                    expected_substrate_trust_v5=0.7),
                compound_pressure=0.0,
                compound_repair_trajectory_cid="",
                compound_window_turns=0,
                expected_substrate_trust_v6=0.7))
        env_compound_promoted = (
            p.handoff_coordinator_v6.decide_v6(
                req_v6=HandoffRequestV6(
                    inner_v5=HandoffRequestV5(
                        inner_v4=HandoffRequestV4(
                            inner_v3=HandoffRequestV3(
                                inner_v2=HandoffRequestV2(
                                    inner_v1=HandoffRequest(
                                        request_cid=(
                                            "w74-turn-cmp"),
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
                        expected_substrate_trust_v5=0.7),
                    compound_pressure=0.8,
                    compound_repair_trajectory_cid=str(
                        cache.compound_repair_trajectory_cid),
                    compound_window_turns=10,
                    expected_substrate_trust_v6=0.7)))
        env_cmp_fallback = (
            p.handoff_coordinator_v6.decide_v6(
                req_v6=HandoffRequestV6(
                    inner_v5=HandoffRequestV5(
                        inner_v4=HandoffRequestV4(
                            inner_v3=HandoffRequestV3(
                                inner_v2=HandoffRequestV2(
                                    inner_v1=HandoffRequest(
                                        request_cid="w74-turn-cmp-f",
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
                        expected_substrate_trust_v5=0.7),
                    compound_pressure=0.0,
                    compound_repair_trajectory_cid=str(
                        cache.compound_repair_trajectory_cid),
                    compound_window_turns=10,
                    expected_substrate_trust_v6=0.7)))
        env_substrate_only = (
            p.handoff_coordinator_v6.decide_v6(
                req_v6=HandoffRequestV6(
                    inner_v5=HandoffRequestV5(
                        inner_v4=HandoffRequestV4(
                            inner_v3=HandoffRequestV3(
                                inner_v2=HandoffRequestV2(
                                    inner_v1=HandoffRequest(
                                        request_cid=(
                                            "w74-turn-substrate"),
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
                        expected_substrate_trust_v5=0.7),
                    compound_pressure=0.0,
                    expected_substrate_trust_v6=0.7)))
        handoff_v6_witness = (
            emit_hosted_real_handoff_coordinator_v6_witness(
                p.handoff_coordinator_v6))
        handoff_envelope_chain_cid = _sha256_hex({
            "kind": "w74_handoff_envelope_v6_chain",
            "envelopes": [
                env_text_only.cid(),
                env_compound_promoted.cid(),
                env_cmp_fallback.cid(),
                env_substrate_only.cid(),
            ],
        })
        baseline_agg = per_regime_aggs.get("baseline")
        v19_beats = (
            float(baseline_agg.v19_beats_v18_rate)
            if baseline_agg is not None else 0.0)
        tsc_v19_beats = (
            float(baseline_agg.tsc_v19_beats_tsc_v18_rate)
            if baseline_agg is not None else 0.0)
        ts_per_vt = (
            float(
                baseline_agg.team_success_per_visible_token_v19)
            if baseline_agg is not None else 0.0)
        return W74HandoffEnvelope(
            schema=W74_SCHEMA_VERSION,
            w73_outer_cid=str(w73_outer_cid),
            w74_params_cid=str(p.cid()),
            substrate_v19_witness_cid=str(sub_witness.cid()),
            kv_bridge_v19_witness_cid=str(kv_witness.cid()),
            cache_controller_v17_witness_cid=str(
                cache_witness.cid()),
            replay_controller_v15_witness_cid=str(
                replay_witness.cid()),
            persistent_v26_witness_cid=str(
                persist_witness.cid()),
            mlsc_v22_witness_cid=str(mlsc_witness.cid()),
            consensus_v20_witness_cid=str(
                consensus_witness.cid()),
            lhr_v26_witness_cid=str(lhr_witness.cid()),
            deep_substrate_hybrid_v19_witness_cid=str(
                deep_v19_witness.cid()),
            substrate_adapter_v19_matrix_cid=str(
                adapter_matrix.cid()),
            masc_v10_witness_cid=str(masc_witness.cid()),
            team_consensus_controller_v9_witness_cid=str(
                tcc_witness.cid()),
            compound_pressure_falsifier_witness_cid=str(
                cmp_falsifier.cid()),
            hosted_router_v7_witness_cid=str(
                router_v7_witness.cid()),
            hosted_logprob_router_v7_witness_cid=str(
                logprob_v7_witness.cid()),
            hosted_cache_planner_v7_witness_cid=str(
                cache_planner_v7_witness.cid()),
            hosted_real_substrate_boundary_v7_cid=str(
                boundary_v7.cid()),
            hosted_wall_v7_report_cid=str(
                wall_v7_report.cid()),
            handoff_coordinator_v6_witness_cid=str(
                handoff_v6_witness.cid()),
            handoff_envelope_v6_chain_cid=str(
                handoff_envelope_chain_cid),
            provider_filter_v6_report_cid=str(
                filter_report_cid),
            nineteen_way_used=bool(
                deep_v19_witness.nineteen_way),
            substrate_v19_used=True,
            masc_v10_v19_beats_v18_rate=float(v19_beats),
            masc_v10_tsc_v19_beats_tsc_v18_rate=float(
                tsc_v19_beats),
            masc_v10_team_success_per_visible_token=float(
                ts_per_vt),
            hosted_router_v7_chosen=str(
                router_dec.chosen_provider or ""),
            compound_repair_trajectory_cid=str(
                cache.compound_repair_trajectory_cid),
        )


def build_default_w74_team(*, seed: int = 74000) -> W74Team:
    return W74Team(params=W74Params.build_default(seed=int(seed)))


__all__ = [
    "W74_SCHEMA_VERSION",
    "W74_FAILURE_MODES",
    "W74Params",
    "W74HandoffEnvelope",
    "verify_w74_handoff",
    "W74Team",
    "build_default_w74_team",
]
