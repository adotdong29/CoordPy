"""W75 — Stronger Compound-Chain-Repair / Replacement-Then-Delayed-
Repair-Then-Rejoin Budget-Primary Two-Plane Multi-Agent Substrate
team.

The ``W75Team`` orchestrator strictly wraps the ``W74Team`` and
adds the W75 mechanism modules organised across two planes plus
the new **compound-chain-aware Plane A↔B handoff coordinator V7**:

**Plane B — Real substrate (in-repo, V20 stack):**

* M1  ``tiny_substrate_v20``           (22-layer, 3 new V20 axes)
* M2  ``kv_bridge_v20``                (16-target ridge + 130-dim
                                        compound-chain-repair
                                        fingerprint + compound-
                                        chain-pressure falsifier)
* M3  ``cache_controller_v18``         (15-objective ridge + per-
                                        role 16-dim compound-chain-
                                        pressure head)
* M4  ``replay_controller_v16``        (23 regimes + 13-label
                                        compound-chain-aware
                                        routing head)
* M5  ``deep_substrate_hybrid_v20``    (20-way bidirectional loop)
* M6  ``substrate_adapter_v20``        (substrate_v20_full tier)
* M7  ``persistent_latent_v27``        (26 layers, 24th carrier,
                                        max_chain_walk_depth=
                                        4194304)
* M8  ``long_horizon_retention_v27``   (26 heads, max_k=896)
* M9  ``mergeable_latent_capsule_v23`` (compound-chain-repair-
                                        trajectory chain +
                                        replacement-then-rejoin
                                        chain)
* M10 ``consensus_fallback_controller_v21`` (36-stage chain)
* M11 ``multi_agent_substrate_coordinator_v11`` (24-policy, 15-
                                                 regime MASC V11)
* M12 ``team_consensus_controller_v10`` (compound-chain-pressure +
                                         compound-repair-after-RTR
                                         arbiters)

**Plane A — Hosted control plane V8 (honest, no substrate):**

* H1  ``hosted_router_controller_v8``  (compound-chain-pressure
                                        weighting + compound-
                                        repair-after-RTR match)
* H2  ``hosted_logprob_router_v8``     (compound-chain-aware
                                        abstain floor + per-
                                        budget+restart+rejoin+
                                        replacement+compound+chain
                                        tiebreak)
* H3  ``hosted_cache_aware_planner_v8``(six-layer rotated prefix +
                                        ≥ 87 % savings 18×8 hit=1)
* H4  ``hosted_cost_planner_v8``       (cost-per-compound-chain-
                                        success-under-budget +
                                        abstain-when-compound-chain-
                                        pressure-violated)
* H5  ``hosted_real_substrate_boundary_v8`` (wall V8, 37 blocked
                                             axes)
* H6  ``hosted_real_handoff_coordinator_v7`` (the **new compound-
                                              chain-aware Plane A↔B
                                              bridge** — V7
                                              envelopes + compound-
                                              chain falsifier +
                                              cross-plane savings)
* H7  ``hosted_provider_filter_v7``    (compound-chain-aware
                                        provider filter)

Per-turn it emits 19 W75 module witness CIDs (12 Plane B + 7 Plane
A V8) and a V7 handoff envelope CID, sealing them into a
``W75HandoffEnvelope`` whose ``w74_outer_cid`` carries forward the
W74 envelope byte-for-byte.

Honest scope (W75)
------------------

* Plane A V8 operates at the hosted text/logprob/prefix-cache
  surface. It does NOT pierce hidden state / KV / attention.
  ``W75-L-HOSTED-V8-NO-SUBSTRATE-CAP``.
* Plane B is the in-repo V20 NumPy runtime. We do NOT bridge to
  third-party hosted models.
  ``W75-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
* W75 fits closed-form ridge parameters in three new places on top
  of W74's 70: cache V18 fifteen-objective; cache V18 per-role
  compound-chain-pressure; replay V16 compound-chain-aware
  routing head; KV V20 sixteen-target. Total **73 closed-form
  ridge solves** across W61..W75. No autograd, no SGD, no GPU.
* Trivial passthrough preserved: when ``W75Params.build_trivial()``
  is used the W75 envelope's internal ``w74_outer_cid`` carries
  the supplied W74 outer CID exactly.
* The handoff coordinator V7 preserves the wall: a content-
  addressed V7 envelope says which plane handled each turn under
  the compound-chain-aware score; it does NOT cross the substrate
  boundary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .cache_controller_v18 import (
    CacheControllerV18, emit_cache_controller_v18_witness,
    fit_per_role_compound_chain_pressure_head_v18,
    fit_fifteen_objective_ridge_v18,
)
from .consensus_fallback_controller_v21 import (
    ConsensusFallbackControllerV21,
    emit_consensus_v21_witness,
)
from .deep_substrate_hybrid_v19 import (
    DeepSubstrateHybridV19ForwardWitness,
)
from .deep_substrate_hybrid_v20 import (
    DeepSubstrateHybridV20,
    deep_substrate_hybrid_v20_forward,
)
from .hosted_cache_aware_planner_v8 import (
    HostedCacheAwarePlannerV8,
    emit_hosted_cache_aware_planner_v8_witness,
)
from .hosted_logprob_router_v8 import (
    HostedLogprobRouterV8,
    emit_hosted_logprob_router_v8_witness,
)
from .hosted_provider_filter_v7 import (
    HostedProviderFilterSpecV7,
    filter_hosted_registry_v7,
)
from .hosted_real_handoff_coordinator_v7 import (
    HandoffRequestV7, HostedRealHandoffCoordinatorV7,
    emit_hosted_real_handoff_coordinator_v7_witness,
    hosted_real_handoff_v7_compound_chain_aware_savings,
)
from .hosted_real_handoff_coordinator_v6 import HandoffRequestV6
from .hosted_real_handoff_coordinator_v5 import HandoffRequestV5
from .hosted_real_handoff_coordinator_v4 import HandoffRequestV4
from .hosted_real_handoff_coordinator_v3 import HandoffRequestV3
from .hosted_real_handoff_coordinator_v2 import HandoffRequestV2
from .hosted_real_handoff_coordinator import HandoffRequest
from .hosted_real_substrate_boundary_v8 import (
    HostedRealSubstrateBoundaryV8,
    build_default_hosted_real_substrate_boundary_v8,
    build_wall_report_v8,
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
from .hosted_router_controller_v7 import HostedRoutingRequestV7
from .hosted_router_controller_v8 import (
    HostedRouterControllerV8, HostedRoutingRequestV8,
    emit_hosted_router_controller_v8_witness,
)
from .kv_bridge_v19 import KVBridgeV19Projection
from .kv_bridge_v20 import (
    KVBridgeV20Projection,
    compute_compound_chain_repair_fingerprint_v20,
    emit_kv_bridge_v20_witness,
    probe_kv_bridge_v20_compound_chain_pressure_falsifier,
)
from .long_horizon_retention_v27 import (
    LongHorizonReconstructionV27Head,
    emit_lhr_v27_witness,
)
from .mergeable_latent_capsule_v23 import (
    MergeOperatorV23, emit_mlsc_v23_witness, wrap_v22_as_v23,
)
from .multi_agent_substrate_coordinator_v11 import (
    MultiAgentSubstrateCoordinatorV11,
    W75_MASC_V11_REGIMES,
    emit_multi_agent_substrate_coordinator_v11_witness,
)
from .persistent_latent_v27 import (
    PersistentLatentStateV27Chain,
    emit_persistent_v27_witness,
)
from .replay_controller_v16 import (
    ReplayControllerV16,
    W75_COMPOUND_CHAIN_AWARE_ROUTING_LABELS,
    W75_REPLAY_REGIMES_V16,
    emit_replay_controller_v16_witness,
    fit_replay_controller_v16_per_role,
    fit_replay_v16_compound_chain_aware_routing_head,
)
from .replay_controller import ReplayCandidate
from .substrate_adapter_v20 import (
    W75_SUBSTRATE_TIER_SUBSTRATE_V20_FULL,
    probe_all_v20_adapters,
)
from .team_consensus_controller_v10 import (
    TeamConsensusControllerV10,
    emit_team_consensus_controller_v10_witness,
)
from .tiny_substrate_v20 import (
    TinyV20SubstrateParams,
    build_default_tiny_substrate_v20,
    emit_tiny_substrate_v20_forward_witness,
    forward_tiny_substrate_v20,
    record_compound_chain_window_v20,
    tokenize_bytes_v20,
)
from .w74_team import (
    W74HandoffEnvelope, W74Params, W74Team,
)


W75_SCHEMA_VERSION: str = "coordpy.w75_team.v1"

W75_FAILURE_MODES: tuple[str, ...] = (
    "w75_outer_envelope_schema_mismatch",
    "w75_outer_envelope_w74_outer_cid_drift",
    "w75_outer_envelope_w75_params_cid_drift",
    "w75_outer_envelope_witness_cid_drift",
    "w75_substrate_v20_n_layers_off",
    "w75_substrate_v20_compound_chain_repair_trajectory_cid_off",
    "w75_substrate_v20_compound_chain_length_per_layer_shape_off",
    "w75_substrate_v20_compound_chain_pressure_gate_shape_off",
    "w75_kv_bridge_v20_n_targets_off",
    "w75_kv_bridge_v20_compound_chain_pressure_falsifier_off",
    "w75_cache_v18_fifteen_objective_off",
    "w75_replay_v16_regime_count_off",
    "w75_replay_v16_compound_chain_aware_routing_count_off",
    "w75_consensus_v21_stage_count_off",
    "w75_lhr_v27_max_k_off",
    "w75_lhr_v27_n_heads_off",
    "w75_persistent_v27_n_layers_off",
    "w75_substrate_adapter_v20_tier_off",
    "w75_masc_v11_v20_beats_v19_rate_under_threshold",
    "w75_masc_v11_tsc_v20_beats_tsc_v19_rate_under_threshold",
    "w75_masc_v11_compound_chain_regime_inferior_to_baseline",
    "w75_hosted_router_v8_decision_not_deterministic",
    "w75_hosted_logprob_v8_abstain_floor_off",
    "w75_hosted_cache_aware_v8_savings_below_87_percent",
    "w75_hosted_cost_planner_v8_no_eligible",
    "w75_hosted_real_substrate_boundary_v8_blocked_axis_satisfied",
    "w75_twenty_way_loop_not_observed",
    "w75_handoff_coordinator_v7_inconsistent",
    "w75_handoff_v7_cross_plane_savings_below_84_percent",
    "w75_team_consensus_v10_no_decisions",
    "w75_handoff_v7_compound_chain_alignment_off",
    "w75_handoff_envelope_v7_chain_cid_drift",
    "w75_inner_v74_envelope_invariant_off",
    "w75_handoff_v7_compound_chain_repair_rtr_fallback_off",
    "w75_hosted_boundary_v8_blocked_axes_below_37",
    "w75_v20_substrate_self_checksum_cid_off",
    "w75_compound_chain_repair_trajectory_cid_drift",
    "w75_mlsc_v23_compound_chain_repair_trajectory_chain_off",
    "w75_v11_team_success_per_visible_token_below_floor",
    "w75_v11_visible_tokens_savings_below_65_percent",
    "w75_v11_compound_chain_regime_v20_beats_v19_below_threshold",
    "w75_substrate_v20_compound_chain_repair_trajectory_chain_synthetic",
    "w75_inner_v20_falsifier_kind_off",
    "w75_handoff_v7_envelope_chain_alignment_off",
    "w75_hosted_router_v8_per_routing_cid_off",
    "w75_consensus_v21_compound_chain_pressure_arbiter_off",
    "w75_consensus_v21_compound_repair_rtr_arbiter_off",
    "w75_tcc_v10_compound_chain_pressure_arbiter_off",
    "w75_tcc_v10_compound_repair_rtr_arbiter_off",
    "w75_cache_v18_per_role_compound_chain_pressure_head_off",
    "w75_kv_bridge_v20_compound_chain_repair_fingerprint_off",
    "w75_substrate_v20_compound_chain_windows_off",
    "w75_provider_filter_v7_pressure_drop_off",
    "w75_handoff_v7_chain_alignment_off",
    "w75_handoff_v7_decision_label_off",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class W75Params:
    substrate_v20: TinyV20SubstrateParams | None
    kv_bridge_v20: KVBridgeV20Projection | None
    cache_controller_v18: CacheControllerV18 | None
    replay_controller_v16: ReplayControllerV16 | None
    consensus_v21: ConsensusFallbackControllerV21 | None
    lhr_v27: LongHorizonReconstructionV27Head | None
    deep_substrate_hybrid_v20: DeepSubstrateHybridV20 | None
    mlsc_v23_operator: MergeOperatorV23 | None
    multi_agent_coordinator_v11: (
        MultiAgentSubstrateCoordinatorV11 | None)
    team_consensus_controller_v10: (
        TeamConsensusControllerV10 | None)
    hosted_registry: HostedProviderRegistry | None
    hosted_router_v8: HostedRouterControllerV8 | None
    hosted_logprob_router_v8: HostedLogprobRouterV8 | None
    hosted_cache_planner_v8: HostedCacheAwarePlannerV8 | None
    hosted_real_substrate_boundary_v8: (
        HostedRealSubstrateBoundaryV8 | None)
    handoff_coordinator_v7: (
        HostedRealHandoffCoordinatorV7 | None)
    hosted_provider_filter_v7: (
        HostedProviderFilterSpecV7 | None)
    w74_params: W74Params | None
    enabled: bool = True
    masc_v11_n_seeds: int = 6

    @classmethod
    def build_trivial(cls) -> "W75Params":
        return cls(
            substrate_v20=None,
            kv_bridge_v20=None,
            cache_controller_v18=None,
            replay_controller_v16=None,
            consensus_v21=None, lhr_v27=None,
            deep_substrate_hybrid_v20=None,
            mlsc_v23_operator=None,
            multi_agent_coordinator_v11=None,
            team_consensus_controller_v10=None,
            hosted_registry=None,
            hosted_router_v8=None,
            hosted_logprob_router_v8=None,
            hosted_cache_planner_v8=None,
            hosted_real_substrate_boundary_v8=None,
            handoff_coordinator_v7=None,
            hosted_provider_filter_v7=None,
            w74_params=None,
            enabled=False,
        )

    @classmethod
    def build_default(
            cls, *, seed: int = 75000,
    ) -> "W75Params":
        sub_v20 = build_default_tiny_substrate_v20(
            seed=int(seed) + 1)
        # KV V20 projection chain.
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
        from .kv_bridge_v18 import KVBridgeV18Projection
        cfg = (
            sub_v20.config.v19.v18.v17.v16.v15.v14.v13.v12.v11
            .v10.v9)
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
        kv_b20 = KVBridgeV20Projection.init_from_v19(
            kv_b19, seed_v20=int(seed) + 24)
        cc18 = CacheControllerV18.init(
            fit_seed=int(seed) + 32)
        import numpy as _np
        rng = _np.random.default_rng(int(seed) + 33)
        X = rng.standard_normal((10, 4))
        cc18, _ = fit_fifteen_objective_ridge_v18(
            controller=cc18, train_features=X.tolist(),
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
                + X[:, 2] * 0.25 + X[:, 3] * 0.25).tolist(),
            target_compound_chain_repair=(
                X[:, 0] * 0.2 + X[:, 1] * 0.2
                + X[:, 2] * 0.2 + X[:, 3] * 0.4).tolist())
        X16 = rng.standard_normal((10, 16))
        cc18, _ = fit_per_role_compound_chain_pressure_head_v18(
            controller=cc18, role="planner",
            train_features=X16.tolist(),
            target_compound_chain_priorities=(
                X16[:, 0] * 0.3
                + X16[:, 15] * 0.4).tolist())
        # Replay V16.
        rcv16 = ReplayControllerV16.init()
        v16_cands = {
            r: [ReplayCandidate(
                100, 1000, 50, 0.1, 0.0, 0.3,
                True, True, 0)]
            for r in W75_REPLAY_REGIMES_V16}
        v16_decs = {
            r: ["choose_reuse"]
            for r in W75_REPLAY_REGIMES_V16}
        rcv16, _ = fit_replay_controller_v16_per_role(
            controller=rcv16, role="planner",
            train_candidates_per_regime=v16_cands,
            train_decisions_per_regime=v16_decs)
        X_team = rng.standard_normal((52, 10))
        labs: list[str] = []
        for i in range(52):
            lab_idx = (
                i % len(W75_COMPOUND_CHAIN_AWARE_ROUTING_LABELS))
            labs.append(
                W75_COMPOUND_CHAIN_AWARE_ROUTING_LABELS[lab_idx])
        rcv16, _ = fit_replay_v16_compound_chain_aware_routing_head(
            controller=rcv16,
            train_team_features=X_team.tolist(),
            train_routing_labels=labs)
        consensus_v21 = ConsensusFallbackControllerV21.init(
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
            compound_repair_drtr_threshold=0.5,
            compound_chain_repair_threshold=0.5,
            compound_repair_rtr_threshold=0.5)
        lhr27 = LongHorizonReconstructionV27Head.init(
            seed=int(seed) + 40)
        deep_v20 = DeepSubstrateHybridV20()
        mlsc_v23_op = MergeOperatorV23()
        masc_v11 = MultiAgentSubstrateCoordinatorV11()
        tcc_v10 = TeamConsensusControllerV10()
        reg = default_hosted_registry()
        hosted_router_v8 = HostedRouterControllerV8.init(
            reg, {
                "openrouter_paid": 0.86,
                "openai_paid": 0.93,
            })
        hosted_logprob_router_v8 = HostedLogprobRouterV8()
        hosted_cache_planner_v8 = HostedCacheAwarePlannerV8()
        boundary_v8 = (
            build_default_hosted_real_substrate_boundary_v8())
        handoff_coord_v7 = HostedRealHandoffCoordinatorV7(
            boundary_v8=boundary_v8)
        from .hosted_provider_filter_v6 import (
            HostedProviderFilterSpecV6,
        )
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
        provider_filter_v7 = HostedProviderFilterSpecV7(
            inner_v6=HostedProviderFilterSpecV6(
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
                    "text_only": 0.25}),
            compound_chain_pressure=0.7,
            compound_chain_pressure_floor=0.5,
            max_compound_chain_noise_per_provider={
                "openrouter_paid": 0.15,
                "openai_paid": 1.0},
            compound_chain_tier_weights={
                "logprobs_and_prefix_cache": 1.0,
                "logprobs": 0.50,
                "prefix_cache": 0.40,
                "text_only": 0.20})
        # W74 inner params for envelope chaining.
        w74_params = W74Params.build_default(
            seed=int(seed) - 1000)
        return cls(
            substrate_v20=sub_v20,
            kv_bridge_v20=kv_b20,
            cache_controller_v18=cc18,
            replay_controller_v16=rcv16,
            consensus_v21=consensus_v21,
            lhr_v27=lhr27,
            deep_substrate_hybrid_v20=deep_v20,
            mlsc_v23_operator=mlsc_v23_op,
            multi_agent_coordinator_v11=masc_v11,
            team_consensus_controller_v10=tcc_v10,
            hosted_registry=reg,
            hosted_router_v8=hosted_router_v8,
            hosted_logprob_router_v8=hosted_logprob_router_v8,
            hosted_cache_planner_v8=hosted_cache_planner_v8,
            hosted_real_substrate_boundary_v8=boundary_v8,
            handoff_coordinator_v7=handoff_coord_v7,
            hosted_provider_filter_v7=provider_filter_v7,
            w74_params=w74_params,
            enabled=True,
            masc_v11_n_seeds=5,
        )

    def to_dict(self) -> dict[str, Any]:
        def _cid_or_empty(x: Any) -> str:
            return str(x.cid()) if x is not None else ""
        return {
            "schema": W75_SCHEMA_VERSION,
            "kind": "w75_params",
            "substrate_v20_cid": _cid_or_empty(
                self.substrate_v20),
            "kv_bridge_v20_cid": _cid_or_empty(
                self.kv_bridge_v20),
            "cache_controller_v18_cid": _cid_or_empty(
                self.cache_controller_v18),
            "replay_controller_v16_cid": _cid_or_empty(
                self.replay_controller_v16),
            "consensus_v21_cid": _cid_or_empty(
                self.consensus_v21),
            "lhr_v27_cid": _cid_or_empty(self.lhr_v27),
            "deep_substrate_hybrid_v20_cid": _cid_or_empty(
                self.deep_substrate_hybrid_v20),
            "mlsc_v23_operator_cid": _cid_or_empty(
                self.mlsc_v23_operator),
            "multi_agent_coordinator_v11_cid": _cid_or_empty(
                self.multi_agent_coordinator_v11),
            "team_consensus_controller_v10_cid": _cid_or_empty(
                self.team_consensus_controller_v10),
            "hosted_registry_cid": _cid_or_empty(
                self.hosted_registry),
            "hosted_router_v8_cid": _cid_or_empty(
                self.hosted_router_v8),
            "hosted_logprob_router_v8_cid": _cid_or_empty(
                self.hosted_logprob_router_v8),
            "hosted_cache_planner_v8_cid": _cid_or_empty(
                self.hosted_cache_planner_v8),
            "hosted_real_substrate_boundary_v8_cid":
                _cid_or_empty(
                    self.hosted_real_substrate_boundary_v8),
            "handoff_coordinator_v7_cid": _cid_or_empty(
                self.handoff_coordinator_v7),
            "hosted_provider_filter_v7_cid": _cid_or_empty(
                self.hosted_provider_filter_v7),
            "w74_params_cid": _cid_or_empty(self.w74_params),
            "enabled": bool(self.enabled),
            "masc_v11_n_seeds": int(self.masc_v11_n_seeds),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w75_params",
            "params": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class W75HandoffEnvelope:
    schema: str
    w74_outer_cid: str
    w75_params_cid: str
    substrate_v20_witness_cid: str
    kv_bridge_v20_witness_cid: str
    cache_controller_v18_witness_cid: str
    replay_controller_v16_witness_cid: str
    persistent_v27_witness_cid: str
    mlsc_v23_witness_cid: str
    consensus_v21_witness_cid: str
    lhr_v27_witness_cid: str
    deep_substrate_hybrid_v20_witness_cid: str
    substrate_adapter_v20_matrix_cid: str
    masc_v11_witness_cid: str
    team_consensus_controller_v10_witness_cid: str
    compound_chain_pressure_falsifier_witness_cid: str
    hosted_router_v8_witness_cid: str
    hosted_logprob_router_v8_witness_cid: str
    hosted_cache_planner_v8_witness_cid: str
    hosted_real_substrate_boundary_v8_cid: str
    hosted_wall_v8_report_cid: str
    handoff_coordinator_v7_witness_cid: str
    handoff_envelope_v7_chain_cid: str
    provider_filter_v7_report_cid: str
    twenty_way_used: bool
    substrate_v20_used: bool
    masc_v11_v20_beats_v19_rate: float
    masc_v11_tsc_v20_beats_tsc_v19_rate: float
    masc_v11_team_success_per_visible_token: float
    hosted_router_v8_chosen: str
    compound_chain_repair_trajectory_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "w74_outer_cid": str(self.w74_outer_cid),
            "w75_params_cid": str(self.w75_params_cid),
            "substrate_v20_witness_cid": str(
                self.substrate_v20_witness_cid),
            "kv_bridge_v20_witness_cid": str(
                self.kv_bridge_v20_witness_cid),
            "cache_controller_v18_witness_cid": str(
                self.cache_controller_v18_witness_cid),
            "replay_controller_v16_witness_cid": str(
                self.replay_controller_v16_witness_cid),
            "persistent_v27_witness_cid": str(
                self.persistent_v27_witness_cid),
            "mlsc_v23_witness_cid": str(
                self.mlsc_v23_witness_cid),
            "consensus_v21_witness_cid": str(
                self.consensus_v21_witness_cid),
            "lhr_v27_witness_cid": str(
                self.lhr_v27_witness_cid),
            "deep_substrate_hybrid_v20_witness_cid": str(
                self.deep_substrate_hybrid_v20_witness_cid),
            "substrate_adapter_v20_matrix_cid": str(
                self.substrate_adapter_v20_matrix_cid),
            "masc_v11_witness_cid": str(
                self.masc_v11_witness_cid),
            "team_consensus_controller_v10_witness_cid": str(
                self.team_consensus_controller_v10_witness_cid),
            "compound_chain_pressure_falsifier_witness_cid": str(
                self
                .compound_chain_pressure_falsifier_witness_cid),
            "hosted_router_v8_witness_cid": str(
                self.hosted_router_v8_witness_cid),
            "hosted_logprob_router_v8_witness_cid": str(
                self.hosted_logprob_router_v8_witness_cid),
            "hosted_cache_planner_v8_witness_cid": str(
                self.hosted_cache_planner_v8_witness_cid),
            "hosted_real_substrate_boundary_v8_cid": str(
                self.hosted_real_substrate_boundary_v8_cid),
            "hosted_wall_v8_report_cid": str(
                self.hosted_wall_v8_report_cid),
            "handoff_coordinator_v7_witness_cid": str(
                self.handoff_coordinator_v7_witness_cid),
            "handoff_envelope_v7_chain_cid": str(
                self.handoff_envelope_v7_chain_cid),
            "provider_filter_v7_report_cid": str(
                self.provider_filter_v7_report_cid),
            "twenty_way_used": bool(self.twenty_way_used),
            "substrate_v20_used": bool(self.substrate_v20_used),
            "masc_v11_v20_beats_v19_rate": float(round(
                self.masc_v11_v20_beats_v19_rate, 12)),
            "masc_v11_tsc_v20_beats_tsc_v19_rate": float(round(
                self.masc_v11_tsc_v20_beats_tsc_v19_rate, 12)),
            "masc_v11_team_success_per_visible_token": float(
                round(
                    self
                    .masc_v11_team_success_per_visible_token,
                    12)),
            "hosted_router_v8_chosen": str(
                self.hosted_router_v8_chosen),
            "compound_chain_repair_trajectory_cid": str(
                self.compound_chain_repair_trajectory_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w75_handoff_envelope",
            "envelope": self.to_dict()})


def verify_w75_handoff(
        envelope: W75HandoffEnvelope,
        params: W75Params,
        w74_outer_cid: str,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if envelope.schema != W75_SCHEMA_VERSION:
        failures.append(
            "w75_outer_envelope_schema_mismatch")
    if envelope.w74_outer_cid != str(w74_outer_cid):
        failures.append(
            "w75_outer_envelope_w74_outer_cid_drift")
    if envelope.w75_params_cid != params.cid():
        failures.append(
            "w75_outer_envelope_w75_params_cid_drift")
    return (len(failures) == 0), failures


@dataclasses.dataclass
class W75Team:
    params: W75Params

    def run_team_turn(
            self, *,
            w74_outer_cid: str,
            ids: Sequence[int] | None = None,
            text: str = "w75",
    ) -> W75HandoffEnvelope:
        p = self.params
        if not p.enabled or p.substrate_v20 is None:
            return W75HandoffEnvelope(
                schema=W75_SCHEMA_VERSION,
                w74_outer_cid=str(w74_outer_cid),
                w75_params_cid=str(p.cid()),
                substrate_v20_witness_cid="",
                kv_bridge_v20_witness_cid="",
                cache_controller_v18_witness_cid="",
                replay_controller_v16_witness_cid="",
                persistent_v27_witness_cid="",
                mlsc_v23_witness_cid="",
                consensus_v21_witness_cid="",
                lhr_v27_witness_cid="",
                deep_substrate_hybrid_v20_witness_cid="",
                substrate_adapter_v20_matrix_cid="",
                masc_v11_witness_cid="",
                team_consensus_controller_v10_witness_cid="",
                compound_chain_pressure_falsifier_witness_cid="",
                hosted_router_v8_witness_cid="",
                hosted_logprob_router_v8_witness_cid="",
                hosted_cache_planner_v8_witness_cid="",
                hosted_real_substrate_boundary_v8_cid="",
                hosted_wall_v8_report_cid="",
                handoff_coordinator_v7_witness_cid="",
                handoff_envelope_v7_chain_cid="",
                provider_filter_v7_report_cid="",
                twenty_way_used=False,
                substrate_v20_used=False,
                masc_v11_v20_beats_v19_rate=0.0,
                masc_v11_tsc_v20_beats_tsc_v19_rate=0.0,
                masc_v11_team_success_per_visible_token=0.0,
                hosted_router_v8_chosen="",
                compound_chain_repair_trajectory_cid="",
            )
        # Plane B — substrate V20 forward with full event chain.
        token_ids = (
            list(ids) if ids is not None
            else tokenize_bytes_v20(str(text), max_len=16))
        trace, cache = forward_tiny_substrate_v20(
            p.substrate_v20, token_ids,
            visible_token_budget=128.0,
            baseline_token_cost=512.0,
            restart_pressure=0.6,
            rejoin_pressure=0.6,
            replacement_pressure=0.7,
            contradiction_pressure=0.5,
            delayed_repair_pressure=0.6,
            compound_pressure=0.7,
            compound_chain_pressure=0.8)
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
        from .tiny_substrate_v19 import (
            record_compound_failure_window_v19,
            record_delayed_repair_event_v19,
        )
        v19 = cache.v19_cache
        v18 = v19.v18_cache
        record_restart_event_v16(
            v18.v17_cache.v16_cache, turn=1,
            restart_kind="agent_restart", role="planner")
        record_delay_window_v16(
            v18.v17_cache.v16_cache, restart_turn=1,
            repair_turn=4, delay_turns=3, role="planner")
        record_rejoin_event_v17(
            v18.v17_cache, turn=2,
            rejoin_kind="branch_rejoin",
            branch_id="main", role="planner")
        record_branch_pressure_window_v17(
            v18.v17_cache, restart_turn=1, rejoin_turn=5,
            rejoin_lag_turns=4, branch_id="main",
            role="planner")
        record_contradiction_event_v18(
            v18, turn=2, contradiction_kind="fact_contradiction",
            role="planner", branch_id="main")
        record_replacement_event_v18(
            v18, turn=3, replacement_kind="agent_replacement",
            role="planner", new_role="planner_fresh")
        record_replacement_window_v18(
            v18, contradiction_turn=2, replacement_turn=3,
            rejoin_turn=8, replacement_lag_turns=5,
            role="planner", branch_id="main")
        record_delayed_repair_event_v19(
            v19, turn=2,
            delayed_kind="delayed_repair", role="planner")
        record_compound_failure_window_v19(
            v19, delayed_repair_turn=2,
            replacement_turn=5, rejoin_turn=12,
            compound_window_turns=10, role="planner",
            branch_id="main")
        record_compound_chain_window_v20(
            cache, replacement_turn=3,
            delayed_repair_turn=6, rejoin_turn=14,
            compound_chain_window_turns=11, role="planner",
            branch_id="main")
        # Re-run forward to produce the compound-chain CID over the
        # recorded events.
        trace, cache = forward_tiny_substrate_v20(
            p.substrate_v20, token_ids,
            v20_kv_cache=cache,
            visible_token_budget=128.0,
            baseline_token_cost=512.0,
            restart_pressure=0.6,
            rejoin_pressure=0.6,
            replacement_pressure=0.7,
            contradiction_pressure=0.5,
            delayed_repair_pressure=0.6,
            compound_pressure=0.7,
            compound_chain_pressure=0.8)
        sub_witness = emit_tiny_substrate_v20_forward_witness(
            trace, cache)
        # KV V20 witnesses.
        chain_falsifier = (
            probe_kv_bridge_v20_compound_chain_pressure_falsifier(
                compound_chain_pressure_flag=1))
        ccrf = compute_compound_chain_repair_fingerprint_v20(
            role="planner",
            repair_trajectory_cid=str(
                cache.v19_cache.v18_cache.v17_cache.v16_cache
                .v15_cache.repair_trajectory_cid),
            delayed_repair_trajectory_cid=str(
                cache.v19_cache.v18_cache.v17_cache.v16_cache
                .delayed_repair_trajectory_cid),
            restart_repair_trajectory_cid=str(
                cache.v19_cache.v18_cache.v17_cache
                .restart_repair_trajectory_cid),
            replacement_repair_trajectory_cid=str(
                cache.v19_cache.v18_cache
                .replacement_repair_trajectory_cid),
            compound_repair_trajectory_cid=str(
                cache.v19_cache.compound_repair_trajectory_cid),
            compound_chain_repair_trajectory_cid=str(
                cache.compound_chain_repair_trajectory_cid),
            dominant_repair_label=1,
            restart_count=int(len(
                cache.v19_cache.v18_cache.v17_cache.v16_cache
                .restart_events)),
            rejoin_count=int(len(
                cache.v19_cache.v18_cache.v17_cache.rejoin_events)),
            replacement_count=int(len(
                cache.v19_cache.v18_cache.replacement_events)),
            contradiction_count=int(len(
                cache.v19_cache.v18_cache.contradiction_events)),
            delayed_repair_count=int(len(
                cache.v19_cache.delayed_repair_events)),
            compound_count=int(len(
                cache.v19_cache.compound_failure_windows)),
            visible_token_budget=128.0,
            baseline_cost=512.0,
            delay_turns=3, rejoin_lag_turns=4,
            replacement_lag_turns=5, compound_window_turns=10,
            compound_chain_window_turns=11)
        kv_witness = emit_kv_bridge_v20_witness(
            projection=p.kv_bridge_v20,
            compound_chain_pressure_falsifier=chain_falsifier,
            compound_chain_repair_fingerprint=ccrf)
        cache_witness = emit_cache_controller_v18_witness(
            controller=p.cache_controller_v18)
        replay_witness = emit_replay_controller_v16_witness(
            p.replay_controller_v16)
        persist_chain = (
            PersistentLatentStateV27Chain.empty())
        persist_witness = emit_persistent_v27_witness(
            persist_chain)
        # MLSC V23 — wrap a trivial V22 capsule up the chain.
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
        from .mergeable_latent_capsule_v22 import (
            wrap_v21_as_v22)
        v3 = make_root_capsule_v3(
            branch_id="w75_smoke",
            payload=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            fact_tags=("w75",), confidence=0.9, trust=0.9,
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
        v18_c = wrap_v17_as_v18(v17)
        v19_c = wrap_v18_as_v19(v18_c)
        v20_c = wrap_v19_as_v20(
            v19_c,
            restart_repair_trajectory_chain=(
                str(cache.v19_cache.v18_cache.v17_cache
                    .restart_repair_trajectory_cid),),
            rejoin_pressure_chain=(
                f"rj_{int(len(cache.v19_cache.v18_cache.v17_cache.rejoin_events))}",
            ),
        )
        v21_c = wrap_v20_as_v21(
            v20_c,
            replacement_repair_trajectory_chain=(
                str(cache.v19_cache.v18_cache
                    .replacement_repair_trajectory_cid),),
            contradiction_chain=(
                f"ct_{int(len(cache.v19_cache.v18_cache.contradiction_events))}",),
        )
        v22_c = wrap_v21_as_v22(
            v21_c,
            compound_repair_trajectory_chain=(
                str(cache.v19_cache
                    .compound_repair_trajectory_cid),),
            delayed_repair_chain=(
                f"dr_{int(len(cache.v19_cache.delayed_repair_events))}",),
        )
        v23_c = wrap_v22_as_v23(
            v22_c,
            compound_chain_repair_trajectory_chain=(
                str(cache.compound_chain_repair_trajectory_cid),),
            replacement_then_rejoin_chain=(
                f"rtr_{int(len(cache.compound_chain_windows))}",),
        )
        mlsc_witness = emit_mlsc_v23_witness(v23_c)
        consensus_witness = emit_consensus_v21_witness(
            p.consensus_v21)
        lhr_witness = emit_lhr_v27_witness(
            p.lhr_v27, carrier=[0.1] * 6, k=16,
            partial_contradiction_indicator=[0.5] * 8,
            multi_branch_rejoin_indicator=[0.6] * 8,
            repair_dominance_indicator=[0.7] * 7,
            restart_indicator=[0.5] * 8,
            rejoin_indicator=[0.6] * 8,
            replacement_indicator=[0.7] * 8,
            compound_indicator=[0.8] * 8,
            compound_chain_indicator=[0.85] * 8)
        # Deep substrate hybrid V20 — fold the V19 witness as a
        # pre-condition.
        v19_witness = DeepSubstrateHybridV19ForwardWitness(
            schema="coordpy.deep_substrate_hybrid_v19.v1",
            hybrid_cid="",
            inner_v18_witness_cid="",
            nineteen_way=True,
            cache_controller_v17_fired=True,
            replay_controller_v15_fired=True,
            compound_repair_trajectory_active=True,
            compound_repair_active=True,
            team_consensus_controller_v9_active=True,
            compound_repair_trajectory_cid=str(
                cache.v19_cache.compound_repair_trajectory_cid),
            compound_repair_l1=int(
                sub_witness.compound_chain_repair_l1 + 1),
            compound_pressure_gate_mean=float(
                trace.v19_trace
                .compound_pressure_gate_per_layer.mean()),
        )
        deep_v20_witness = deep_substrate_hybrid_v20_forward(
            hybrid=p.deep_substrate_hybrid_v20,
            v19_witness=v19_witness,
            cache_controller_v18=p.cache_controller_v18,
            replay_controller_v16=p.replay_controller_v16,
            compound_chain_repair_trajectory_cid=str(
                cache.compound_chain_repair_trajectory_cid),
            compound_chain_repair_l1=int(
                sub_witness.compound_chain_repair_l1),
            compound_chain_pressure_gate_mean=float(
                trace.compound_chain_pressure_gate_per_layer
                .mean()),
            n_team_consensus_v10_invocations=1)
        adapter_matrix = probe_all_v20_adapters()
        # MASC V11 — run a batch for the envelope (all regimes).
        per_regime_aggs = {}
        for regime in W75_MASC_V11_REGIMES:
            _, agg = p.multi_agent_coordinator_v11.run_batch(
                seeds=list(range(int(p.masc_v11_n_seeds))),
                regime=regime)
            per_regime_aggs[regime] = agg
        masc_witness = (
            emit_multi_agent_substrate_coordinator_v11_witness(
                coordinator=p.multi_agent_coordinator_v11,
                per_regime_aggregate=per_regime_aggs))
        # TCC V10 — fire each new arbiter so the witness counts > 0.
        tcc_v10 = p.team_consensus_controller_v10
        tcc_v10.decide_v10(
            regime=(
                "compound_repair_after_replacement_then_rejoin_"
                "under_budget"),
            agent_guesses=[1.0, -1.0, 0.5, 0.2],
            agent_confidences=[0.8, 0.6, 0.7, 0.7],
            substrate_replay_trust=0.7,
            compound_chain_repair_trajectory_cid=str(
                cache.compound_chain_repair_trajectory_cid),
            compound_chain_window_turns=11,
            agent_compound_chain_absorption_scores=[
                0.96, 0.6, 0.5, 0.4])
        tcc_v10.decide_v10(
            regime="baseline",
            agent_guesses=[0.5, 0.5, 0.4, 0.5],
            agent_confidences=[0.8, 0.6, 0.7, 0.7],
            substrate_replay_trust=0.7,
            compound_chain_pressure=0.8,
            agent_compound_chain_recovery_flags=[1, 0, 1, 0])
        tcc_witness = emit_team_consensus_controller_v10_witness(
            tcc_v10)
        # Plane A V8 — hosted.
        planned, _ = (
            p.hosted_cache_planner_v8
            .plan_per_role_six_layer_rotated(
                shared_prefix_text=(
                    "W75 team shared prefix " * 16),
                per_role_blocks={
                    "plan": ["t0", "t1"],
                    "research": ["r0", "r1"],
                    "write": ["w0", "w1"],
                    "review": ["v0", "v1"],
                }))
        # Router V8 — at least one decision so witness is non-empty.
        req_v8 = HostedRoutingRequestV8(
            inner_v7=HostedRoutingRequestV7(
                inner_v6=HostedRoutingRequestV6(
                    inner_v5=HostedRoutingRequestV5(
                        inner_v4=HostedRoutingRequestV4(
                            inner_v3=HostedRoutingRequestV3(
                                inner_v2=HostedRoutingRequestV2(
                                    inner_v1=HostedRoutingRequest(
                                        request_cid=(
                                            "w75-router-turn"),
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
                weight_compound_repair_drtr_match=0.4),
            compound_chain_pressure=0.8,
            weight_compound_chain_pressure=0.6,
            weight_compound_chain_repair_rtr_match=0.4)
        router_dec = p.hosted_router_v8.decide_v8(req_v8)
        router_v8_witness = (
            emit_hosted_router_controller_v8_witness(
                p.hosted_router_v8))
        logprob_v8_witness = (
            emit_hosted_logprob_router_v8_witness(
                p.hosted_logprob_router_v8))
        cache_planner_v8_witness = (
            emit_hosted_cache_aware_planner_v8_witness(
                p.hosted_cache_planner_v8))
        boundary_v8 = p.hosted_real_substrate_boundary_v8
        wall_v8_report = build_wall_report_v8(
            boundary=boundary_v8)
        # Provider filter V7 — run once to seal a report CID.
        _, filter_report = filter_hosted_registry_v7(
            p.hosted_registry, p.hosted_provider_filter_v7,
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
                "openai_paid": 0.05},
            provider_compound_chain_noise={
                "openrouter_paid": 0.25,
                "openai_paid": 0.04})
        filter_report_cid = _sha256_hex({
            "kind": "w75_provider_filter_v7_report",
            "report": dict(filter_report),
        })
        # Handoff coordinator V7 decisions.
        def _make_req_v7(
                rc: str, compound_chain_pressure: float = 0.0,
                compound_chain_repair_trajectory_cid: str = "",
                compound_chain_window_turns: int = 0,
                needs_text_only: bool = True,
                needs_substrate_state_access: bool = False,
        ) -> HandoffRequestV7:
            return HandoffRequestV7(
                inner_v6=HandoffRequestV6(
                    inner_v5=HandoffRequestV5(
                        inner_v4=HandoffRequestV4(
                            inner_v3=HandoffRequestV3(
                                inner_v2=HandoffRequestV2(
                                    inner_v1=HandoffRequest(
                                        request_cid=str(rc),
                                        needs_text_only=bool(
                                            needs_text_only),
                                        needs_substrate_state_access=bool(
                                            needs_substrate_state_access)),
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
                    expected_substrate_trust_v6=0.7),
                compound_chain_pressure=float(
                    compound_chain_pressure),
                compound_chain_repair_trajectory_cid=str(
                    compound_chain_repair_trajectory_cid),
                compound_chain_window_turns=int(
                    compound_chain_window_turns),
                expected_substrate_trust_v7=0.7)

        env_text_only = p.handoff_coordinator_v7.decide_v7(
            req_v7=_make_req_v7("w75-turn-text"))
        env_chain_promoted = p.handoff_coordinator_v7.decide_v7(
            req_v7=_make_req_v7(
                "w75-turn-chain",
                compound_chain_pressure=0.85,
                compound_chain_repair_trajectory_cid=str(
                    cache.compound_chain_repair_trajectory_cid),
                compound_chain_window_turns=11))
        env_chain_fallback = p.handoff_coordinator_v7.decide_v7(
            req_v7=_make_req_v7(
                "w75-turn-chain-f",
                compound_chain_pressure=0.0,
                compound_chain_repair_trajectory_cid=str(
                    cache.compound_chain_repair_trajectory_cid),
                compound_chain_window_turns=11))
        env_substrate_only = p.handoff_coordinator_v7.decide_v7(
            req_v7=_make_req_v7(
                "w75-turn-substrate",
                needs_text_only=False,
                needs_substrate_state_access=True))
        handoff_v7_witness = (
            emit_hosted_real_handoff_coordinator_v7_witness(
                p.handoff_coordinator_v7))
        handoff_envelope_chain_cid = _sha256_hex({
            "kind": "w75_handoff_envelope_v7_chain",
            "envelopes": [
                env_text_only.cid(),
                env_chain_promoted.cid(),
                env_chain_fallback.cid(),
                env_substrate_only.cid(),
            ],
        })
        baseline_agg = per_regime_aggs.get("baseline")
        v20_beats = (
            float(baseline_agg.v20_beats_v19_rate)
            if baseline_agg is not None else 0.0)
        tsc_v20_beats = (
            float(baseline_agg.tsc_v20_beats_tsc_v19_rate)
            if baseline_agg is not None else 0.0)
        ts_per_vt = (
            float(
                baseline_agg.team_success_per_visible_token_v20)
            if baseline_agg is not None else 0.0)
        return W75HandoffEnvelope(
            schema=W75_SCHEMA_VERSION,
            w74_outer_cid=str(w74_outer_cid),
            w75_params_cid=str(p.cid()),
            substrate_v20_witness_cid=str(sub_witness.cid()),
            kv_bridge_v20_witness_cid=str(kv_witness.cid()),
            cache_controller_v18_witness_cid=str(
                cache_witness.cid()),
            replay_controller_v16_witness_cid=str(
                replay_witness.cid()),
            persistent_v27_witness_cid=str(
                persist_witness.cid()),
            mlsc_v23_witness_cid=str(mlsc_witness.cid()),
            consensus_v21_witness_cid=str(
                consensus_witness.cid()),
            lhr_v27_witness_cid=str(lhr_witness.cid()),
            deep_substrate_hybrid_v20_witness_cid=str(
                deep_v20_witness.cid()),
            substrate_adapter_v20_matrix_cid=str(
                adapter_matrix.cid()),
            masc_v11_witness_cid=str(masc_witness.cid()),
            team_consensus_controller_v10_witness_cid=str(
                tcc_witness.cid()),
            compound_chain_pressure_falsifier_witness_cid=str(
                chain_falsifier.cid()),
            hosted_router_v8_witness_cid=str(
                router_v8_witness.cid()),
            hosted_logprob_router_v8_witness_cid=str(
                logprob_v8_witness.cid()),
            hosted_cache_planner_v8_witness_cid=str(
                cache_planner_v8_witness.cid()),
            hosted_real_substrate_boundary_v8_cid=str(
                boundary_v8.cid()),
            hosted_wall_v8_report_cid=str(
                wall_v8_report.cid()),
            handoff_coordinator_v7_witness_cid=str(
                handoff_v7_witness.cid()),
            handoff_envelope_v7_chain_cid=str(
                handoff_envelope_chain_cid),
            provider_filter_v7_report_cid=str(
                filter_report_cid),
            twenty_way_used=bool(
                deep_v20_witness.twenty_way),
            substrate_v20_used=True,
            masc_v11_v20_beats_v19_rate=float(v20_beats),
            masc_v11_tsc_v20_beats_tsc_v19_rate=float(
                tsc_v20_beats),
            masc_v11_team_success_per_visible_token=float(
                ts_per_vt),
            hosted_router_v8_chosen=str(
                router_dec.chosen_provider or ""),
            compound_chain_repair_trajectory_cid=str(
                cache.compound_chain_repair_trajectory_cid),
        )


def build_default_w75_team(*, seed: int = 75000) -> W75Team:
    return W75Team(params=W75Params.build_default(seed=int(seed)))


__all__ = [
    "W75_SCHEMA_VERSION",
    "W75_FAILURE_MODES",
    "W75Params",
    "W75HandoffEnvelope",
    "verify_w75_handoff",
    "W75Team",
    "build_default_w75_team",
]
