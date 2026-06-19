"""W76 — Stronger Restart-After-Compound-Chain-Repair /
Compound-Chain-Then-Restart Budget-Primary Two-Plane Multi-Agent
Substrate team.

The ``W76Team`` orchestrator strictly wraps the ``W75Team`` and
adds the W76 mechanism modules organised across two planes plus
the new **chain-then-restart-aware Plane A↔B handoff coordinator
V8**:

**Plane B — Real substrate (in-repo, V21 stack):**

* M1  ``tiny_substrate_v21``           (23-layer, 3 new V21 axes)
* M2  ``kv_bridge_v21``                (17-target ridge + 140-dim
                                        chain-then-restart
                                        fingerprint + chain-then-
                                        restart-pressure falsifier)
* M3  ``cache_controller_v19``         (16-objective ridge + per-
                                        role 17-dim chain-then-
                                        restart-pressure head)
* M4  ``replay_controller_v17``        (24 regimes + 14-label
                                        chain-then-restart-aware
                                        routing head)
* M5  ``deep_substrate_hybrid_v21``    (21-way bidirectional loop)
* M6  ``substrate_adapter_v21``        (substrate_v21_full tier)
* M7  ``persistent_latent_v28``        (27 layers, 25th carrier,
                                        max_chain_walk_depth=
                                        8388608)
* M8  ``long_horizon_retention_v28``   (27 heads, max_k=960)
* M9  ``mergeable_latent_capsule_v24`` (chain-then-restart-
                                        trajectory chain +
                                        post-compound-chain-
                                        restart chain)
* M10 ``consensus_fallback_controller_v22`` (38-stage chain)
* M11 ``multi_agent_substrate_coordinator_v12`` (26-policy, 16-
                                                 regime MASC V12)
* M12 ``team_consensus_controller_v11`` (chain-then-restart-
                                         pressure +
                                         post-compound-chain-
                                         restart-after-RTR
                                         arbiters)

**Plane A — Hosted control plane V9 (honest, no substrate):**

* H1  ``hosted_router_controller_v9``  (chain-then-restart-
                                        pressure weighting +
                                        chain-then-restart-after-
                                        RTR match)
* H2  ``hosted_logprob_router_v9``     (chain-then-restart-aware
                                        abstain floor + 7-way
                                        tiebreak)
* H3  ``hosted_cache_aware_planner_v9``(seven-layer rotated
                                        prefix; ≥ 88 % savings
                                        20×8 hit=1)
* H4  ``hosted_cost_planner_v9``       (cost-per-chain-then-
                                        restart-success-under-
                                        budget + abstain-when-
                                        chain-then-restart-
                                        pressure-violated)
* H5  ``hosted_real_substrate_boundary_v9`` (wall V9, 40 blocked
                                             axes)
* H6  ``hosted_real_handoff_coordinator_v8`` (the **new chain-
                                              then-restart-aware
                                              Plane A↔B bridge**)
* H7  ``hosted_provider_filter_v8``    (chain-then-restart-aware
                                        provider filter)

Per-turn it emits 19 W76 module witness CIDs (12 Plane B + 7
Plane A V9) and a V8 handoff envelope CID, sealing them into a
``W76HandoffEnvelope`` whose ``w75_outer_cid`` carries forward
the W75 envelope byte-for-byte.

Honest scope (W76)
------------------

* Plane A V9 operates at the hosted text/logprob/prefix-cache
  surface. It does NOT pierce hidden state / KV / attention.
  ``W76-L-HOSTED-V9-NO-SUBSTRATE-CAP``.
* Plane B is the in-repo V21 NumPy runtime. We do NOT bridge to
  third-party hosted models.
  ``W76-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
* W76 fits closed-form ridge parameters in three new places on
  top of W75's 73: cache V19 sixteen-objective; replay V17
  chain-then-restart-aware routing head; KV V21 seventeen-target.
  Total **76 closed-form ridge solves** across W61..W76. No
  autograd, no SGD, no GPU.
* Trivial passthrough preserved: when ``W76Params.build_trivial()``
  is used the W76 envelope's internal ``w75_outer_cid`` carries
  the supplied W75 outer CID exactly.
* The handoff coordinator V8 preserves the wall: a content-
  addressed V8 envelope says which plane handled each turn under
  the chain-then-restart-aware score; it does NOT cross the
  substrate boundary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .cache_controller_v19 import (
    CacheControllerV19, emit_cache_controller_v19_witness,
    fit_per_role_chain_then_restart_pressure_head_v19,
    fit_sixteen_objective_ridge_v19,
)
from .consensus_fallback_controller_v22 import (
    ConsensusFallbackControllerV22,
    emit_consensus_v22_witness,
)
from .deep_substrate_hybrid_v20 import (
    DeepSubstrateHybridV20ForwardWitness,
)
from .deep_substrate_hybrid_v21 import (
    DeepSubstrateHybridV21,
    deep_substrate_hybrid_v21_forward,
)
from .hosted_cache_aware_planner_v9 import (
    HostedCacheAwarePlannerV9,
    emit_hosted_cache_aware_planner_v9_witness,
)
from .hosted_logprob_router_v9 import (
    HostedLogprobRouterV9,
    emit_hosted_logprob_router_v9_witness,
)
from .hosted_provider_filter_v8 import (
    HostedProviderFilterSpecV8,
    filter_hosted_registry_v8,
)
from .hosted_provider_filter_v7 import HostedProviderFilterSpecV7
from .hosted_provider_filter_v6 import HostedProviderFilterSpecV6
from .hosted_provider_filter_v5 import HostedProviderFilterSpecV5
from .hosted_provider_filter_v4 import HostedProviderFilterSpecV4
from .hosted_provider_filter_v3 import HostedProviderFilterSpecV3
from .hosted_provider_filter_v2 import HostedProviderFilterSpecV2
from .hosted_provider_filter import HostedProviderFilterSpec
from .hosted_real_handoff_coordinator_v8 import (
    HandoffRequestV8, HostedRealHandoffCoordinatorV8,
    emit_hosted_real_handoff_coordinator_v8_witness,
)
from .hosted_real_handoff_coordinator_v7 import HandoffRequestV7
from .hosted_real_handoff_coordinator_v6 import HandoffRequestV6
from .hosted_real_handoff_coordinator_v5 import HandoffRequestV5
from .hosted_real_handoff_coordinator_v4 import HandoffRequestV4
from .hosted_real_handoff_coordinator_v3 import HandoffRequestV3
from .hosted_real_handoff_coordinator_v2 import HandoffRequestV2
from .hosted_real_handoff_coordinator import HandoffRequest
from .hosted_real_substrate_boundary_v9 import (
    HostedRealSubstrateBoundaryV9,
    build_default_hosted_real_substrate_boundary_v9,
    build_wall_report_v9,
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
from .hosted_router_controller_v8 import HostedRoutingRequestV8
from .hosted_router_controller_v9 import (
    HostedRouterControllerV9, HostedRoutingRequestV9,
    emit_hosted_router_controller_v9_witness,
)
from .kv_bridge_v20 import KVBridgeV20Projection
from .kv_bridge_v21 import (
    KVBridgeV21Projection,
    compute_chain_then_restart_fingerprint_v21,
    emit_kv_bridge_v21_witness,
    probe_kv_bridge_v21_chain_then_restart_falsifier,
)
from .long_horizon_retention_v28 import (
    LongHorizonReconstructionV28Head,
    emit_lhr_v28_witness,
)
from .mergeable_latent_capsule_v24 import (
    MergeOperatorV24, emit_mlsc_v24_witness, wrap_v23_as_v24,
)
from .multi_agent_substrate_coordinator_v12 import (
    MultiAgentSubstrateCoordinatorV12,
    W76_MASC_V12_REGIMES,
    emit_multi_agent_substrate_coordinator_v12_witness,
)
from .persistent_latent_v28 import (
    PersistentLatentStateV28Chain,
    emit_persistent_v28_witness,
)
from .replay_controller import ReplayCandidate
from .replay_controller_v17 import (
    ReplayControllerV17,
    W76_CHAIN_THEN_RESTART_AWARE_ROUTING_LABELS,
    W76_REPLAY_REGIMES_V17,
    emit_replay_controller_v17_witness,
    fit_replay_controller_v17_per_role,
    fit_replay_v17_chain_then_restart_aware_routing_head,
)
from .substrate_adapter_v21 import (
    W76_SUBSTRATE_TIER_SUBSTRATE_V21_FULL,
    probe_all_v21_adapters,
)
from .team_consensus_controller_v11 import (
    TeamConsensusControllerV11,
    emit_team_consensus_controller_v11_witness,
)
from .tiny_substrate_v21 import (
    TinyV21SubstrateParams,
    build_default_tiny_substrate_v21,
    emit_tiny_substrate_v21_forward_witness,
    forward_tiny_substrate_v21,
    record_post_compound_chain_restart_window_v21,
    tokenize_bytes_v21,
)
from .w75_team import (
    W75HandoffEnvelope, W75Params, W75Team,
)


W76_SCHEMA_VERSION: str = "coordpy.w76_team.v1"

W76_FAILURE_MODES: tuple[str, ...] = (
    "w76_outer_envelope_schema_mismatch",
    "w76_outer_envelope_w75_outer_cid_drift",
    "w76_outer_envelope_w76_params_cid_drift",
    "w76_outer_envelope_witness_cid_drift",
    "w76_substrate_v21_n_layers_off",
    "w76_substrate_v21_chain_then_restart_trajectory_cid_off",
    "w76_substrate_v21_chain_then_restart_length_per_layer_shape_off",
    "w76_substrate_v21_chain_then_restart_pressure_gate_shape_off",
    "w76_kv_bridge_v21_n_targets_off",
    "w76_kv_bridge_v21_chain_then_restart_pressure_falsifier_off",
    "w76_cache_v19_sixteen_objective_off",
    "w76_replay_v17_regime_count_off",
    "w76_replay_v17_chain_then_restart_aware_routing_count_off",
    "w76_consensus_v22_stage_count_off",
    "w76_lhr_v28_max_k_off",
    "w76_lhr_v28_n_heads_off",
    "w76_persistent_v28_n_layers_off",
    "w76_substrate_adapter_v21_tier_off",
    "w76_masc_v12_v21_beats_v20_rate_under_threshold",
    "w76_masc_v12_tsc_v21_beats_tsc_v20_rate_under_threshold",
    "w76_masc_v12_chain_then_restart_regime_inferior_to_baseline",
    "w76_hosted_router_v9_decision_not_deterministic",
    "w76_hosted_logprob_v9_abstain_floor_off",
    "w76_hosted_cache_aware_v9_savings_below_88_percent",
    "w76_hosted_cost_planner_v9_no_eligible",
    "w76_hosted_real_substrate_boundary_v9_blocked_axis_satisfied",
    "w76_twenty_one_way_loop_not_observed",
    "w76_handoff_coordinator_v8_inconsistent",
    "w76_handoff_v8_cross_plane_savings_below_85_percent",
    "w76_team_consensus_v11_no_decisions",
    "w76_handoff_v8_chain_then_restart_alignment_off",
    "w76_handoff_envelope_v8_chain_cid_drift",
    "w76_inner_v75_envelope_invariant_off",
    "w76_handoff_v8_chain_then_restart_fallback_off",
    "w76_hosted_boundary_v9_blocked_axes_below_40",
    "w76_v21_substrate_self_checksum_cid_off",
    "w76_chain_then_restart_trajectory_cid_drift",
    "w76_mlsc_v24_chain_then_restart_trajectory_chain_off",
    "w76_v12_team_success_per_visible_token_below_floor",
    "w76_v12_visible_tokens_savings_below_65_percent",
    "w76_v12_chain_then_restart_regime_v21_beats_v20_below_threshold",
    "w76_substrate_v21_chain_then_restart_trajectory_chain_synthetic",
    "w76_inner_v21_falsifier_kind_off",
    "w76_handoff_v8_envelope_chain_alignment_off",
    "w76_hosted_router_v9_per_routing_cid_off",
    "w76_consensus_v22_chain_then_restart_arbiter_off",
    "w76_consensus_v22_post_compound_chain_restart_arbiter_off",
    "w76_tcc_v11_chain_then_restart_pressure_arbiter_off",
    "w76_tcc_v11_post_compound_chain_restart_arbiter_off",
    "w76_cache_v19_per_role_chain_then_restart_pressure_head_off",
    "w76_kv_bridge_v21_chain_then_restart_fingerprint_off",
    "w76_substrate_v21_post_compound_chain_restart_windows_off",
    "w76_provider_filter_v8_pressure_drop_off",
    "w76_handoff_v8_chain_alignment_off",
    "w76_handoff_v8_decision_label_off",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class W76Params:
    substrate_v21: TinyV21SubstrateParams | None
    kv_bridge_v21: KVBridgeV21Projection | None
    cache_controller_v19: CacheControllerV19 | None
    replay_controller_v17: ReplayControllerV17 | None
    consensus_v22: ConsensusFallbackControllerV22 | None
    lhr_v28: LongHorizonReconstructionV28Head | None
    deep_substrate_hybrid_v21: DeepSubstrateHybridV21 | None
    mlsc_v24_operator: MergeOperatorV24 | None
    multi_agent_coordinator_v12: (
        MultiAgentSubstrateCoordinatorV12 | None)
    team_consensus_controller_v11: (
        TeamConsensusControllerV11 | None)
    hosted_registry: HostedProviderRegistry | None
    hosted_router_v9: HostedRouterControllerV9 | None
    hosted_logprob_router_v9: HostedLogprobRouterV9 | None
    hosted_cache_planner_v9: HostedCacheAwarePlannerV9 | None
    hosted_real_substrate_boundary_v9: (
        HostedRealSubstrateBoundaryV9 | None)
    handoff_coordinator_v8: (
        HostedRealHandoffCoordinatorV8 | None)
    hosted_provider_filter_v8: (
        HostedProviderFilterSpecV8 | None)
    w75_params: W75Params | None
    enabled: bool = True
    masc_v12_n_seeds: int = 5

    @classmethod
    def build_trivial(cls) -> "W76Params":
        return cls(
            substrate_v21=None,
            kv_bridge_v21=None,
            cache_controller_v19=None,
            replay_controller_v17=None,
            consensus_v22=None, lhr_v28=None,
            deep_substrate_hybrid_v21=None,
            mlsc_v24_operator=None,
            multi_agent_coordinator_v12=None,
            team_consensus_controller_v11=None,
            hosted_registry=None,
            hosted_router_v9=None,
            hosted_logprob_router_v9=None,
            hosted_cache_planner_v9=None,
            hosted_real_substrate_boundary_v9=None,
            handoff_coordinator_v8=None,
            hosted_provider_filter_v8=None,
            w75_params=None,
            enabled=False,
        )

    @classmethod
    def build_default(
            cls, *, seed: int = 76000,
    ) -> "W76Params":
        sub_v21 = build_default_tiny_substrate_v21(
            seed=int(seed) + 1)
        # KV V21 projection chain — wrap KV V20.
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
        from .kv_bridge_v19 import KVBridgeV19Projection
        cfg = (
            sub_v21.config.v20.v19.v18.v17.v16.v15.v14.v13.v12
            .v11.v10.v9)
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
        kv_b21 = KVBridgeV21Projection.init_from_v20(
            kv_b20, seed_v21=int(seed) + 25)
        cc19 = CacheControllerV19.init(
            fit_seed=int(seed) + 32)
        import numpy as _np
        rng = _np.random.default_rng(int(seed) + 33)
        X = rng.standard_normal((10, 4))
        cc19, _ = fit_sixteen_objective_ridge_v19(
            controller=cc19, train_features=X.tolist(),
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
                + X[:, 2] * 0.2 + X[:, 3] * 0.4).tolist(),
            target_chain_then_restart_repair=(
                X[:, 0] * 0.15 + X[:, 1] * 0.20
                + X[:, 2] * 0.30 + X[:, 3] * 0.35).tolist())
        X17 = rng.standard_normal((10, 17))
        cc19, _ = fit_per_role_chain_then_restart_pressure_head_v19(
            controller=cc19, role="planner",
            train_features=X17.tolist(),
            target_chain_then_restart_priorities=(
                X17[:, 0] * 0.2
                + X17[:, 16] * 0.5).tolist())
        # Replay V17.
        rcv17 = ReplayControllerV17.init()
        v17_cands = {
            r: [ReplayCandidate(
                100, 1000, 50, 0.1, 0.0, 0.3,
                True, True, 0)]
            for r in W76_REPLAY_REGIMES_V17}
        v17_decs = {
            r: ["choose_reuse"]
            for r in W76_REPLAY_REGIMES_V17}
        rcv17, _ = fit_replay_controller_v17_per_role(
            controller=rcv17, role="planner",
            train_candidates_per_regime=v17_cands,
            train_decisions_per_regime=v17_decs)
        X_team = rng.standard_normal((56, 10))
        labs: list[str] = []
        for i in range(56):
            lab_idx = (
                i % len(
                    W76_CHAIN_THEN_RESTART_AWARE_ROUTING_LABELS))
            labs.append(
                W76_CHAIN_THEN_RESTART_AWARE_ROUTING_LABELS[
                    lab_idx])
        rcv17, _ = (
            fit_replay_v17_chain_then_restart_aware_routing_head(
                controller=rcv17,
                train_team_features=X_team.tolist(),
                train_routing_labels=labs))
        consensus_v22 = ConsensusFallbackControllerV22.init(
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
            compound_repair_rtr_threshold=0.5,
            chain_then_restart_threshold=0.5,
            post_compound_chain_restart_threshold=0.5)
        lhr28 = LongHorizonReconstructionV28Head.init(
            seed=int(seed) + 40)
        deep_v21 = DeepSubstrateHybridV21()
        mlsc_v24_op = MergeOperatorV24()
        masc_v12 = MultiAgentSubstrateCoordinatorV12()
        tcc_v11 = TeamConsensusControllerV11()
        reg = default_hosted_registry()
        hosted_router_v9 = HostedRouterControllerV9.init(
            reg, {
                "openrouter_paid": 0.86,
                "openai_paid": 0.93,
            })
        hosted_logprob_router_v9 = HostedLogprobRouterV9()
        hosted_cache_planner_v9 = HostedCacheAwarePlannerV9()
        boundary_v9 = (
            build_default_hosted_real_substrate_boundary_v9())
        handoff_coord_v8 = HostedRealHandoffCoordinatorV8(
            boundary_v9=boundary_v9)
        provider_filter_v8 = HostedProviderFilterSpecV8(
            inner_v7=HostedProviderFilterSpecV7(
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
                    "text_only": 0.20}),
            chain_then_restart_pressure=0.8,
            chain_then_restart_pressure_floor=0.5,
            max_chain_then_restart_noise_per_provider={
                "openrouter_paid": 0.12,
                "openai_paid": 1.0},
            chain_then_restart_tier_weights={
                "logprobs_and_prefix_cache": 1.0,
                "logprobs": 0.45,
                "prefix_cache": 0.35,
                "text_only": 0.15})
        w75_params = W75Params.build_default(
            seed=int(seed) - 1000)
        return cls(
            substrate_v21=sub_v21,
            kv_bridge_v21=kv_b21,
            cache_controller_v19=cc19,
            replay_controller_v17=rcv17,
            consensus_v22=consensus_v22,
            lhr_v28=lhr28,
            deep_substrate_hybrid_v21=deep_v21,
            mlsc_v24_operator=mlsc_v24_op,
            multi_agent_coordinator_v12=masc_v12,
            team_consensus_controller_v11=tcc_v11,
            hosted_registry=reg,
            hosted_router_v9=hosted_router_v9,
            hosted_logprob_router_v9=hosted_logprob_router_v9,
            hosted_cache_planner_v9=hosted_cache_planner_v9,
            hosted_real_substrate_boundary_v9=boundary_v9,
            handoff_coordinator_v8=handoff_coord_v8,
            hosted_provider_filter_v8=provider_filter_v8,
            w75_params=w75_params,
            enabled=True,
            masc_v12_n_seeds=5,
        )

    def to_dict(self) -> dict[str, Any]:
        def _cid_or_empty(x: Any) -> str:
            return str(x.cid()) if x is not None else ""
        return {
            "schema": W76_SCHEMA_VERSION,
            "kind": "w76_params",
            "substrate_v21_cid": _cid_or_empty(
                self.substrate_v21),
            "kv_bridge_v21_cid": _cid_or_empty(
                self.kv_bridge_v21),
            "cache_controller_v19_cid": _cid_or_empty(
                self.cache_controller_v19),
            "replay_controller_v17_cid": _cid_or_empty(
                self.replay_controller_v17),
            "consensus_v22_cid": _cid_or_empty(
                self.consensus_v22),
            "lhr_v28_cid": _cid_or_empty(self.lhr_v28),
            "deep_substrate_hybrid_v21_cid": _cid_or_empty(
                self.deep_substrate_hybrid_v21),
            "mlsc_v24_operator_cid": _cid_or_empty(
                self.mlsc_v24_operator),
            "multi_agent_coordinator_v12_cid": _cid_or_empty(
                self.multi_agent_coordinator_v12),
            "team_consensus_controller_v11_cid": _cid_or_empty(
                self.team_consensus_controller_v11),
            "hosted_registry_cid": _cid_or_empty(
                self.hosted_registry),
            "hosted_router_v9_cid": _cid_or_empty(
                self.hosted_router_v9),
            "hosted_logprob_router_v9_cid": _cid_or_empty(
                self.hosted_logprob_router_v9),
            "hosted_cache_planner_v9_cid": _cid_or_empty(
                self.hosted_cache_planner_v9),
            "hosted_real_substrate_boundary_v9_cid":
                _cid_or_empty(
                    self.hosted_real_substrate_boundary_v9),
            "handoff_coordinator_v8_cid": _cid_or_empty(
                self.handoff_coordinator_v8),
            "hosted_provider_filter_v8_cid": _cid_or_empty(
                self.hosted_provider_filter_v8),
            "w75_params_cid": _cid_or_empty(self.w75_params),
            "enabled": bool(self.enabled),
            "masc_v12_n_seeds": int(self.masc_v12_n_seeds),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w76_params",
            "params": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class W76HandoffEnvelope:
    schema: str
    w75_outer_cid: str
    w76_params_cid: str
    substrate_v21_witness_cid: str
    kv_bridge_v21_witness_cid: str
    cache_controller_v19_witness_cid: str
    replay_controller_v17_witness_cid: str
    persistent_v28_witness_cid: str
    mlsc_v24_witness_cid: str
    consensus_v22_witness_cid: str
    lhr_v28_witness_cid: str
    deep_substrate_hybrid_v21_witness_cid: str
    substrate_adapter_v21_matrix_cid: str
    masc_v12_witness_cid: str
    team_consensus_controller_v11_witness_cid: str
    chain_then_restart_pressure_falsifier_witness_cid: str
    hosted_router_v9_witness_cid: str
    hosted_logprob_router_v9_witness_cid: str
    hosted_cache_planner_v9_witness_cid: str
    hosted_real_substrate_boundary_v9_cid: str
    hosted_wall_v9_report_cid: str
    handoff_coordinator_v8_witness_cid: str
    handoff_envelope_v8_chain_cid: str
    provider_filter_v8_report_cid: str
    twenty_one_way_used: bool
    substrate_v21_used: bool
    masc_v12_v21_beats_v20_rate: float
    masc_v12_tsc_v21_beats_tsc_v20_rate: float
    masc_v12_team_success_per_visible_token: float
    hosted_router_v9_chosen: str
    chain_then_restart_trajectory_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "w75_outer_cid": str(self.w75_outer_cid),
            "w76_params_cid": str(self.w76_params_cid),
            "substrate_v21_witness_cid": str(
                self.substrate_v21_witness_cid),
            "kv_bridge_v21_witness_cid": str(
                self.kv_bridge_v21_witness_cid),
            "cache_controller_v19_witness_cid": str(
                self.cache_controller_v19_witness_cid),
            "replay_controller_v17_witness_cid": str(
                self.replay_controller_v17_witness_cid),
            "persistent_v28_witness_cid": str(
                self.persistent_v28_witness_cid),
            "mlsc_v24_witness_cid": str(
                self.mlsc_v24_witness_cid),
            "consensus_v22_witness_cid": str(
                self.consensus_v22_witness_cid),
            "lhr_v28_witness_cid": str(
                self.lhr_v28_witness_cid),
            "deep_substrate_hybrid_v21_witness_cid": str(
                self.deep_substrate_hybrid_v21_witness_cid),
            "substrate_adapter_v21_matrix_cid": str(
                self.substrate_adapter_v21_matrix_cid),
            "masc_v12_witness_cid": str(
                self.masc_v12_witness_cid),
            "team_consensus_controller_v11_witness_cid": str(
                self.team_consensus_controller_v11_witness_cid),
            "chain_then_restart_pressure_falsifier_witness_cid":
                str(
                    self
                    .chain_then_restart_pressure_falsifier_witness_cid),
            "hosted_router_v9_witness_cid": str(
                self.hosted_router_v9_witness_cid),
            "hosted_logprob_router_v9_witness_cid": str(
                self.hosted_logprob_router_v9_witness_cid),
            "hosted_cache_planner_v9_witness_cid": str(
                self.hosted_cache_planner_v9_witness_cid),
            "hosted_real_substrate_boundary_v9_cid": str(
                self.hosted_real_substrate_boundary_v9_cid),
            "hosted_wall_v9_report_cid": str(
                self.hosted_wall_v9_report_cid),
            "handoff_coordinator_v8_witness_cid": str(
                self.handoff_coordinator_v8_witness_cid),
            "handoff_envelope_v8_chain_cid": str(
                self.handoff_envelope_v8_chain_cid),
            "provider_filter_v8_report_cid": str(
                self.provider_filter_v8_report_cid),
            "twenty_one_way_used": bool(
                self.twenty_one_way_used),
            "substrate_v21_used": bool(self.substrate_v21_used),
            "masc_v12_v21_beats_v20_rate": float(round(
                self.masc_v12_v21_beats_v20_rate, 12)),
            "masc_v12_tsc_v21_beats_tsc_v20_rate": float(round(
                self.masc_v12_tsc_v21_beats_tsc_v20_rate, 12)),
            "masc_v12_team_success_per_visible_token": float(
                round(
                    self
                    .masc_v12_team_success_per_visible_token,
                    12)),
            "hosted_router_v9_chosen": str(
                self.hosted_router_v9_chosen),
            "chain_then_restart_trajectory_cid": str(
                self.chain_then_restart_trajectory_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w76_handoff_envelope",
            "envelope": self.to_dict()})


def verify_w76_handoff(
        envelope: W76HandoffEnvelope,
        params: W76Params,
        w75_outer_cid: str,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if envelope.schema != W76_SCHEMA_VERSION:
        failures.append(
            "w76_outer_envelope_schema_mismatch")
    if envelope.w75_outer_cid != str(w75_outer_cid):
        failures.append(
            "w76_outer_envelope_w75_outer_cid_drift")
    if envelope.w76_params_cid != params.cid():
        failures.append(
            "w76_outer_envelope_w76_params_cid_drift")
    return (len(failures) == 0), failures


@dataclasses.dataclass
class W76Team:
    params: W76Params

    def run_team_turn(
            self, *,
            w75_outer_cid: str,
            ids: Sequence[int] | None = None,
            text: str = "w76",
    ) -> W76HandoffEnvelope:
        p = self.params
        if not p.enabled or p.substrate_v21 is None:
            return W76HandoffEnvelope(
                schema=W76_SCHEMA_VERSION,
                w75_outer_cid=str(w75_outer_cid),
                w76_params_cid=str(p.cid()),
                substrate_v21_witness_cid="",
                kv_bridge_v21_witness_cid="",
                cache_controller_v19_witness_cid="",
                replay_controller_v17_witness_cid="",
                persistent_v28_witness_cid="",
                mlsc_v24_witness_cid="",
                consensus_v22_witness_cid="",
                lhr_v28_witness_cid="",
                deep_substrate_hybrid_v21_witness_cid="",
                substrate_adapter_v21_matrix_cid="",
                masc_v12_witness_cid="",
                team_consensus_controller_v11_witness_cid="",
                chain_then_restart_pressure_falsifier_witness_cid="",
                hosted_router_v9_witness_cid="",
                hosted_logprob_router_v9_witness_cid="",
                hosted_cache_planner_v9_witness_cid="",
                hosted_real_substrate_boundary_v9_cid="",
                hosted_wall_v9_report_cid="",
                handoff_coordinator_v8_witness_cid="",
                handoff_envelope_v8_chain_cid="",
                provider_filter_v8_report_cid="",
                twenty_one_way_used=False,
                substrate_v21_used=False,
                masc_v12_v21_beats_v20_rate=0.0,
                masc_v12_tsc_v21_beats_tsc_v20_rate=0.0,
                masc_v12_team_success_per_visible_token=0.0,
                hosted_router_v9_chosen="",
                chain_then_restart_trajectory_cid="",
            )
        # Plane B — substrate V21 forward with full event chain.
        token_ids = (
            list(ids) if ids is not None
            else tokenize_bytes_v21(str(text), max_len=16))
        trace, cache = forward_tiny_substrate_v21(
            p.substrate_v21, token_ids,
            visible_token_budget=128.0,
            baseline_token_cost=512.0,
            restart_pressure=0.7,
            rejoin_pressure=0.6,
            replacement_pressure=0.7,
            contradiction_pressure=0.5,
            delayed_repair_pressure=0.6,
            compound_pressure=0.7,
            compound_chain_pressure=0.8,
            compound_chain_then_restart_pressure=0.85)
        # Record full event chain including the W76 post-chain
        # restart window.
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
        from .tiny_substrate_v20 import (
            record_compound_chain_window_v20,
        )
        v20 = cache.v20_cache
        v19 = v20.v19_cache
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
            v20, replacement_turn=3,
            delayed_repair_turn=6, rejoin_turn=14,
            compound_chain_window_turns=11, role="planner",
            branch_id="main")
        # Record W76 post-chain restart window — the new event.
        # The team has just completed compound-chain repair
        # (rejoin at turn 14); a fresh restart hits at turn 18.
        record_restart_event_v16(
            v18.v17_cache.v16_cache, turn=18,
            restart_kind="post_compound_chain_restart",
            role="planner_fresh")
        record_post_compound_chain_restart_window_v21(
            cache, compound_chain_repair_turn=14,
            restart_turn=18,
            post_compound_chain_restart_window_turns=12,
            role="planner_fresh", branch_id="main")
        # Re-run forward to produce the chain-then-restart CID
        # over the recorded events.
        trace, cache = forward_tiny_substrate_v21(
            p.substrate_v21, token_ids,
            v21_kv_cache=cache,
            visible_token_budget=128.0,
            baseline_token_cost=512.0,
            restart_pressure=0.7,
            rejoin_pressure=0.6,
            replacement_pressure=0.7,
            contradiction_pressure=0.5,
            delayed_repair_pressure=0.6,
            compound_pressure=0.7,
            compound_chain_pressure=0.8,
            compound_chain_then_restart_pressure=0.85)
        sub_witness = emit_tiny_substrate_v21_forward_witness(
            trace, cache)
        # KV V21 witnesses.
        ctr_falsifier = (
            probe_kv_bridge_v21_chain_then_restart_falsifier(
                chain_then_restart_pressure_flag=1))
        ctrf = compute_chain_then_restart_fingerprint_v21(
            role="planner",
            repair_trajectory_cid=str(
                cache.v20_cache.v19_cache.v18_cache.v17_cache
                .v16_cache.v15_cache.repair_trajectory_cid),
            delayed_repair_trajectory_cid=str(
                cache.v20_cache.v19_cache.v18_cache.v17_cache
                .v16_cache.delayed_repair_trajectory_cid),
            restart_repair_trajectory_cid=str(
                cache.v20_cache.v19_cache.v18_cache.v17_cache
                .restart_repair_trajectory_cid),
            replacement_repair_trajectory_cid=str(
                cache.v20_cache.v19_cache.v18_cache
                .replacement_repair_trajectory_cid),
            compound_repair_trajectory_cid=str(
                cache.v20_cache.v19_cache
                .compound_repair_trajectory_cid),
            compound_chain_repair_trajectory_cid=str(
                cache.v20_cache
                .compound_chain_repair_trajectory_cid),
            compound_chain_then_restart_trajectory_cid=str(
                cache
                .compound_chain_then_restart_trajectory_cid),
            dominant_repair_label=1,
            restart_count=int(len(
                cache.v20_cache.v19_cache.v18_cache.v17_cache
                .v16_cache.restart_events)),
            rejoin_count=int(len(
                cache.v20_cache.v19_cache.v18_cache.v17_cache
                .rejoin_events)),
            replacement_count=int(len(
                cache.v20_cache.v19_cache.v18_cache
                .replacement_events)),
            contradiction_count=int(len(
                cache.v20_cache.v19_cache.v18_cache
                .contradiction_events)),
            delayed_repair_count=int(len(
                cache.v20_cache.v19_cache.delayed_repair_events)),
            compound_count=int(len(
                cache.v20_cache.v19_cache.compound_failure_windows)),
            compound_chain_count=int(len(
                cache.v20_cache.compound_chain_windows)),
            visible_token_budget=128.0,
            baseline_cost=512.0,
            delay_turns=3, rejoin_lag_turns=4,
            replacement_lag_turns=5, compound_window_turns=10,
            compound_chain_window_turns=11,
            post_compound_chain_restart_window_turns=12)
        kv_witness = emit_kv_bridge_v21_witness(
            projection=p.kv_bridge_v21,
            chain_then_restart_falsifier=ctr_falsifier,
            chain_then_restart_fingerprint=ctrf)
        cache_witness = emit_cache_controller_v19_witness(
            controller=p.cache_controller_v19)
        replay_witness = emit_replay_controller_v17_witness(
            p.replay_controller_v17)
        persist_chain = (
            PersistentLatentStateV28Chain.empty())
        persist_witness = emit_persistent_v28_witness(
            persist_chain)
        # MLSC V24 — wrap a trivial V23 capsule up the chain.
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
        from .mergeable_latent_capsule_v23 import (
            wrap_v22_as_v23)
        v3 = make_root_capsule_v3(
            branch_id="w76_smoke",
            payload=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            fact_tags=("w76",), confidence=0.9, trust=0.9,
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
                str(cache.v20_cache.v19_cache.v18_cache.v17_cache
                    .restart_repair_trajectory_cid),),
            rejoin_pressure_chain=(
                f"rj_{int(len(cache.v20_cache.v19_cache.v18_cache.v17_cache.rejoin_events))}",
            ),
        )
        v21_c = wrap_v20_as_v21(
            v20_c,
            replacement_repair_trajectory_chain=(
                str(cache.v20_cache.v19_cache.v18_cache
                    .replacement_repair_trajectory_cid),),
            contradiction_chain=(
                f"ct_{int(len(cache.v20_cache.v19_cache.v18_cache.contradiction_events))}",),
        )
        v22_c = wrap_v21_as_v22(
            v21_c,
            compound_repair_trajectory_chain=(
                str(cache.v20_cache.v19_cache
                    .compound_repair_trajectory_cid),),
            delayed_repair_chain=(
                f"dr_{int(len(cache.v20_cache.v19_cache.delayed_repair_events))}",),
        )
        v23_c = wrap_v22_as_v23(
            v22_c,
            compound_chain_repair_trajectory_chain=(
                str(cache.v20_cache
                    .compound_chain_repair_trajectory_cid),),
            replacement_then_rejoin_chain=(
                f"rtr_{int(len(cache.v20_cache.compound_chain_windows))}",),
        )
        v24_c = wrap_v23_as_v24(
            v23_c,
            chain_then_restart_trajectory_chain=(
                str(cache
                    .compound_chain_then_restart_trajectory_cid),),
            post_compound_chain_restart_chain=(
                f"pccr_{int(len(cache.post_compound_chain_restart_windows))}",),
        )
        mlsc_witness = emit_mlsc_v24_witness(v24_c)
        consensus_witness = emit_consensus_v22_witness(
            p.consensus_v22)
        lhr_witness = emit_lhr_v28_witness(
            p.lhr_v28, carrier=[0.1] * 6, k=16,
            partial_contradiction_indicator=[0.5] * 8,
            multi_branch_rejoin_indicator=[0.6] * 8,
            repair_dominance_indicator=[0.7] * 7,
            restart_indicator=[0.5] * 8,
            rejoin_indicator=[0.6] * 8,
            replacement_indicator=[0.7] * 8,
            compound_indicator=[0.8] * 8,
            compound_chain_indicator=[0.85] * 8,
            chain_then_restart_indicator=[0.90] * 8)
        # Deep substrate hybrid V21 — fold the V20 witness as a
        # pre-condition.
        v20_witness = DeepSubstrateHybridV20ForwardWitness(
            schema="coordpy.deep_substrate_hybrid_v20.v1",
            hybrid_cid="",
            inner_v19_witness_cid="",
            twenty_way=True,
            cache_controller_v18_fired=True,
            replay_controller_v16_fired=True,
            compound_chain_repair_trajectory_active=True,
            compound_chain_repair_active=True,
            team_consensus_controller_v10_active=True,
            compound_chain_repair_trajectory_cid=str(
                cache.v20_cache
                .compound_chain_repair_trajectory_cid),
            compound_chain_repair_l1=int(
                sub_witness.compound_chain_then_restart_repair_l1
                + 1),
            compound_chain_pressure_gate_mean=float(
                trace.v20_trace
                .compound_chain_pressure_gate_per_layer.mean()),
        )
        deep_v21_witness = deep_substrate_hybrid_v21_forward(
            hybrid=p.deep_substrate_hybrid_v21,
            v20_witness=v20_witness,
            cache_controller_v19=p.cache_controller_v19,
            replay_controller_v17=p.replay_controller_v17,
            chain_then_restart_trajectory_cid=str(
                cache
                .compound_chain_then_restart_trajectory_cid),
            chain_then_restart_repair_l1=int(
                sub_witness.compound_chain_then_restart_repair_l1),
            chain_then_restart_pressure_gate_mean=float(
                trace
                .compound_chain_then_restart_pressure_gate_per_layer
                .mean()),
            n_team_consensus_v11_invocations=1)
        adapter_matrix = probe_all_v21_adapters()
        # MASC V12 — run a batch for the envelope (all regimes).
        per_regime_aggs = {}
        for regime in W76_MASC_V12_REGIMES:
            _, agg = p.multi_agent_coordinator_v12.run_batch(
                seeds=list(range(int(p.masc_v12_n_seeds))),
                regime=regime)
            per_regime_aggs[regime] = agg
        masc_witness = (
            emit_multi_agent_substrate_coordinator_v12_witness(
                coordinator=p.multi_agent_coordinator_v12,
                per_regime_aggregate=per_regime_aggs))
        # TCC V11 — fire each new arbiter so the witness counts > 0.
        tcc_v11 = p.team_consensus_controller_v11
        tcc_v11.decide_v11(
            regime=(
                "restart_after_compound_chain_repair_under_"
                "budget"),
            agent_guesses=[1.0, -1.0, 0.5, 0.2],
            agent_confidences=[0.8, 0.6, 0.7, 0.7],
            substrate_replay_trust=0.7,
            chain_then_restart_trajectory_cid=str(
                cache
                .compound_chain_then_restart_trajectory_cid),
            post_compound_chain_restart_window_turns=12,
            agent_chain_then_restart_absorption_scores=[
                0.97, 0.6, 0.5, 0.4])
        tcc_v11.decide_v11(
            regime="baseline",
            agent_guesses=[0.5, 0.5, 0.4, 0.5],
            agent_confidences=[0.8, 0.6, 0.7, 0.7],
            substrate_replay_trust=0.7,
            chain_then_restart_pressure=0.85,
            agent_chain_then_restart_recovery_flags=[1, 0, 1, 0])
        tcc_witness = emit_team_consensus_controller_v11_witness(
            tcc_v11)
        # Plane A V9 — hosted.
        planned, _ = (
            p.hosted_cache_planner_v9
            .plan_per_role_seven_layer_rotated(
                shared_prefix_text=(
                    "W76 team shared prefix " * 16),
                per_role_blocks={
                    "plan": ["t0", "t1"],
                    "research": ["r0", "r1"],
                    "write": ["w0", "w1"],
                    "review": ["v0", "v1"],
                }))
        # Router V9 — at least one decision so witness is non-empty.
        req_v9 = HostedRoutingRequestV9(
            inner_v8=HostedRoutingRequestV8(
                inner_v7=HostedRoutingRequestV7(
                    inner_v6=HostedRoutingRequestV6(
                        inner_v5=HostedRoutingRequestV5(
                            inner_v4=HostedRoutingRequestV4(
                                inner_v3=HostedRoutingRequestV3(
                                    inner_v2=HostedRoutingRequestV2(
                                        inner_v1=HostedRoutingRequest(
                                            request_cid=(
                                                "w76-router-turn"),
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
                weight_compound_chain_repair_rtr_match=0.4),
            compound_chain_then_restart_pressure=0.85,
            weight_compound_chain_then_restart_pressure=0.6,
            weight_chain_then_restart_after_rtr_match=0.4)
        router_dec = p.hosted_router_v9.decide_v9(req_v9)
        router_v9_witness = (
            emit_hosted_router_controller_v9_witness(
                p.hosted_router_v9))
        logprob_v9_witness = (
            emit_hosted_logprob_router_v9_witness(
                p.hosted_logprob_router_v9))
        cache_planner_v9_witness = (
            emit_hosted_cache_aware_planner_v9_witness(
                p.hosted_cache_planner_v9))
        boundary_v9 = p.hosted_real_substrate_boundary_v9
        wall_v9_report = build_wall_report_v9(
            boundary=boundary_v9)
        # Provider filter V8 — run once to seal a report CID.
        _, filter_report = filter_hosted_registry_v8(
            p.hosted_registry, p.hosted_provider_filter_v8,
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
                "openai_paid": 0.04},
            provider_chain_then_restart_noise={
                "openrouter_paid": 0.20,
                "openai_paid": 0.03})
        filter_report_cid = _sha256_hex({
            "kind": "w76_provider_filter_v8_report",
            "report": dict(filter_report),
        })
        # Handoff coordinator V8 decisions.
        def _make_req_v8(
                rc: str,
                compound_chain_then_restart_pressure: float = 0.0,
                compound_chain_then_restart_trajectory_cid: str = "",
                post_compound_chain_restart_window_turns: int = 0,
                needs_text_only: bool = True,
                needs_substrate_state_access: bool = False,
        ) -> HandoffRequestV8:
            return HandoffRequestV8(
                inner_v7=HandoffRequestV7(
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
                    compound_chain_pressure=0.0,
                    compound_chain_repair_trajectory_cid="",
                    compound_chain_window_turns=0,
                    expected_substrate_trust_v7=0.7),
                compound_chain_then_restart_pressure=float(
                    compound_chain_then_restart_pressure),
                compound_chain_then_restart_trajectory_cid=str(
                    compound_chain_then_restart_trajectory_cid),
                post_compound_chain_restart_window_turns=int(
                    post_compound_chain_restart_window_turns),
                expected_substrate_trust_v8=0.7)

        env_text_only = p.handoff_coordinator_v8.decide_v8(
            req_v8=_make_req_v8("w76-turn-text"))
        env_ctr_promoted = p.handoff_coordinator_v8.decide_v8(
            req_v8=_make_req_v8(
                "w76-turn-ctr",
                compound_chain_then_restart_pressure=0.9,
                compound_chain_then_restart_trajectory_cid=str(
                    cache
                    .compound_chain_then_restart_trajectory_cid),
                post_compound_chain_restart_window_turns=12))
        env_ctr_fallback = p.handoff_coordinator_v8.decide_v8(
            req_v8=_make_req_v8(
                "w76-turn-ctr-f",
                compound_chain_then_restart_pressure=0.0,
                compound_chain_then_restart_trajectory_cid=str(
                    cache
                    .compound_chain_then_restart_trajectory_cid),
                post_compound_chain_restart_window_turns=12))
        env_substrate_only = p.handoff_coordinator_v8.decide_v8(
            req_v8=_make_req_v8(
                "w76-turn-substrate",
                needs_text_only=False,
                needs_substrate_state_access=True))
        handoff_v8_witness = (
            emit_hosted_real_handoff_coordinator_v8_witness(
                p.handoff_coordinator_v8))
        handoff_envelope_chain_cid = _sha256_hex({
            "kind": "w76_handoff_envelope_v8_chain",
            "envelopes": [
                env_text_only.cid(),
                env_ctr_promoted.cid(),
                env_ctr_fallback.cid(),
                env_substrate_only.cid(),
            ],
        })
        baseline_agg = per_regime_aggs.get("baseline")
        v21_beats = (
            float(baseline_agg.v21_beats_v20_rate)
            if baseline_agg is not None else 0.0)
        tsc_v21_beats = (
            float(baseline_agg.tsc_v21_beats_tsc_v20_rate)
            if baseline_agg is not None else 0.0)
        ts_per_vt = (
            float(
                baseline_agg.team_success_per_visible_token_v21)
            if baseline_agg is not None else 0.0)
        return W76HandoffEnvelope(
            schema=W76_SCHEMA_VERSION,
            w75_outer_cid=str(w75_outer_cid),
            w76_params_cid=str(p.cid()),
            substrate_v21_witness_cid=str(sub_witness.cid()),
            kv_bridge_v21_witness_cid=str(kv_witness.cid()),
            cache_controller_v19_witness_cid=str(
                cache_witness.cid()),
            replay_controller_v17_witness_cid=str(
                replay_witness.cid()),
            persistent_v28_witness_cid=str(
                persist_witness.cid()),
            mlsc_v24_witness_cid=str(mlsc_witness.cid()),
            consensus_v22_witness_cid=str(
                consensus_witness.cid()),
            lhr_v28_witness_cid=str(lhr_witness.cid()),
            deep_substrate_hybrid_v21_witness_cid=str(
                deep_v21_witness.cid()),
            substrate_adapter_v21_matrix_cid=str(
                adapter_matrix.cid()),
            masc_v12_witness_cid=str(masc_witness.cid()),
            team_consensus_controller_v11_witness_cid=str(
                tcc_witness.cid()),
            chain_then_restart_pressure_falsifier_witness_cid=str(
                ctr_falsifier.cid()),
            hosted_router_v9_witness_cid=str(
                router_v9_witness.cid()),
            hosted_logprob_router_v9_witness_cid=str(
                logprob_v9_witness.cid()),
            hosted_cache_planner_v9_witness_cid=str(
                cache_planner_v9_witness.cid()),
            hosted_real_substrate_boundary_v9_cid=str(
                boundary_v9.cid()),
            hosted_wall_v9_report_cid=str(
                wall_v9_report.cid()),
            handoff_coordinator_v8_witness_cid=str(
                handoff_v8_witness.cid()),
            handoff_envelope_v8_chain_cid=str(
                handoff_envelope_chain_cid),
            provider_filter_v8_report_cid=str(
                filter_report_cid),
            twenty_one_way_used=bool(
                deep_v21_witness.twenty_one_way),
            substrate_v21_used=True,
            masc_v12_v21_beats_v20_rate=float(v21_beats),
            masc_v12_tsc_v21_beats_tsc_v20_rate=float(
                tsc_v21_beats),
            masc_v12_team_success_per_visible_token=float(
                ts_per_vt),
            hosted_router_v9_chosen=str(
                router_dec.chosen_provider or ""),
            chain_then_restart_trajectory_cid=str(
                cache
                .compound_chain_then_restart_trajectory_cid),
        )


def build_default_w76_team(*, seed: int = 76000) -> W76Team:
    return W76Team(params=W76Params.build_default(seed=int(seed)))


__all__ = [
    "W76_SCHEMA_VERSION",
    "W76_FAILURE_MODES",
    "W76Params",
    "W76HandoffEnvelope",
    "verify_w76_handoff",
    "W76Team",
    "build_default_w76_team",
]
