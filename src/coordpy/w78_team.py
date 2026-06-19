"""W78 — Stronger Less-Bounded Long-Horizon Reconstruction /
Bounded-Window-Falsifier Budget-Primary Two-Plane Multi-Agent
Substrate team.

The ``W78Team`` orchestrator strictly wraps the ``W77Team`` and
adds the W78 mechanism modules organised across two planes plus
the new **long-horizon-reconstruction-aware Plane A↔B handoff
coordinator V10** AND a new **bounded-window-falsifier** anti-
goal mechanism that explicitly benchmarks against fixed-k
transcript baselines.

**Plane B — Real substrate (in-repo, V23 stack):**

* M1  ``tiny_substrate_v23``           (inherits V22 + 1 new
                                        V23 substrate axis)
* M2  ``kv_bridge_v23``                (19-target ridge + 156-dim
                                        long-horizon-
                                        reconstruction fingerprint
                                        + falsifier)
* M3  ``cache_controller_v21``         (18-objective ridge + per-
                                        role 19-dim long-horizon-
                                        reconstruction-pressure
                                        head)
* M4  ``replay_controller_v19``        (26 regimes + 16-label
                                        long-horizon-
                                        reconstruction-aware
                                        routing head)
* M5  ``deep_substrate_hybrid_v23``    (23-way bidirectional loop)
* M6  ``substrate_adapter_v23``        (substrate_v23_full tier)
* M7  ``persistent_latent_v30``        (29 layers, 27th carrier,
                                        max_chain_walk_depth=
                                        33554432)
* M8  ``long_horizon_retention_v30``   (29 heads, max_k=2048)
* M9  ``mergeable_latent_capsule_v26`` (long-horizon-reconstruction-
                                        trajectory chain +
                                        reconstruction-request-
                                        window chain)
* M10 ``consensus_fallback_controller_v24`` (42-stage chain)
* M11 ``multi_agent_substrate_coordinator_v14`` (30-policy, 18-
                                                 regime MASC V14)
* M12 ``team_consensus_controller_v13`` (long-horizon-
                                         reconstruction-pressure
                                         + long-horizon-
                                         reconstruction-trajectory
                                         arbiters)
* M13 ``long_horizon_reconstruction_substrate_v1`` (NEW load-
                                                    bearing
                                                    reconstruction
                                                    substrate)
* M14 ``bounded_window_baseline_v1`` (NEW anti-goal falsifier)

**Plane A — Hosted control plane V11 (honest, no substrate):**

* H1  ``hosted_router_controller_v11``  (long-horizon-
                                         reconstruction-pressure
                                         weighting + LHR match)
* H2  ``hosted_logprob_router_v11``     (long-horizon-
                                         reconstruction-aware
                                         abstain floor + 9-way
                                         tiebreak)
* H3  ``hosted_cache_aware_planner_v11``(nine-layer rotated
                                         prefix; ≥ 89 % savings
                                         20×8 hit=1)
* H4  ``hosted_cost_planner_v11``       (cost-per-long-horizon-
                                         reconstruction-success-
                                         under-budget + abstain-
                                         when-lhr-violated)
* H5  ``hosted_real_substrate_boundary_v11`` (wall V11, 46 blocked
                                              axes)
* H6  ``hosted_real_handoff_coordinator_v10`` (the **new long-
                                               horizon-
                                               reconstruction-
                                               aware Plane A↔B
                                               bridge**)
* H7  ``hosted_provider_filter_v10``    (long-horizon-
                                         reconstruction-aware
                                         provider filter)

Per-turn it emits 21 W78 module witness CIDs (14 Plane B +
7 Plane A V11) plus a V10 handoff envelope CID, sealing them
into a ``W78HandoffEnvelope`` whose ``w77_outer_cid`` carries
forward the W77 envelope byte-for-byte.

Honest scope (W78)
------------------

* Plane A V11 operates at the hosted text/logprob/prefix-cache
  surface. It does NOT pierce hidden state / KV / attention.
  ``W78-L-HOSTED-V11-NO-SUBSTRATE-CAP``.
* Plane B is the in-repo V23 NumPy runtime. We do NOT bridge to
  third-party hosted models.
  ``W78-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
* W78 fits two new closed-form ridges on top of W77's 77: cache
  V21 eighteen-objective + KV V23 nineteen-target. Total **79
  closed-form ridge solves** across W61..W78. No autograd, no
  SGD, no GPU.
* Trivial passthrough preserved: when ``W78Params.build_trivial()``
  is used the W78 envelope's internal ``w77_outer_cid`` carries
  the supplied W77 outer CID exactly.
* The handoff coordinator V10 preserves the wall: a content-
  addressed V10 envelope says which plane handled each turn
  under the long-horizon-reconstruction-aware score; it does
  NOT cross the substrate boundary.
* The bounded-window-baseline V1 is a **load-bearing falsifier**
  module — it benchmarks W78 against fixed-k transcript baselines
  and proves bounded-window-insufficiency for long-horizon
  reconstruction queries.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .bounded_window_baseline_v1 import (
    BoundedWindowQuery, build_default_bounded_window_baselines,
    emit_bounded_window_baseline_witness,
    prove_bounded_window_insufficient,
    run_bounded_window_falsifier,
)
from .cache_controller_v21 import (
    CacheControllerV21, emit_cache_controller_v21_witness,
    fit_eighteen_objective_ridge_v21,
    fit_per_role_long_horizon_reconstruction_pressure_head_v21,
)
from .consensus_fallback_controller_v24 import (
    ConsensusFallbackControllerV24,
    emit_consensus_v24_witness,
)
from .deep_substrate_hybrid_v22 import (
    DeepSubstrateHybridV22ForwardWitness,
)
from .deep_substrate_hybrid_v23 import (
    DeepSubstrateHybridV23,
    deep_substrate_hybrid_v23_forward,
)
from .hosted_cache_aware_planner_v11 import (
    HostedCacheAwarePlannerV11,
    emit_hosted_cache_aware_planner_v11_witness,
)
from .hosted_logprob_router_v11 import (
    HostedLogprobRouterV11,
    emit_hosted_logprob_router_v11_witness,
)
from .hosted_provider_filter_v10 import (
    HostedProviderFilterSpecV10,
    filter_hosted_registry_v10,
)
from .hosted_provider_filter_v9 import HostedProviderFilterSpecV9
from .hosted_real_handoff_coordinator_v10 import (
    HandoffRequestV10, HostedRealHandoffCoordinatorV10,
    emit_hosted_real_handoff_coordinator_v10_witness,
)
from .hosted_real_handoff_coordinator_v9 import HandoffRequestV9
from .hosted_real_substrate_boundary_v11 import (
    HostedRealSubstrateBoundaryV11,
    build_default_hosted_real_substrate_boundary_v11,
    build_wall_report_v11,
)
from .hosted_router_controller import (
    HostedProviderRegistry, default_hosted_registry,
)
from .hosted_router_controller_v10 import HostedRoutingRequestV10
from .hosted_router_controller_v11 import (
    HostedRouterControllerV11, HostedRoutingRequestV11,
    emit_hosted_router_controller_v11_witness,
)
from .kv_bridge_v22 import KVBridgeV22Projection
from .kv_bridge_v23 import (
    KVBridgeV23Projection,
    compute_long_horizon_reconstruction_fingerprint_v23,
    emit_kv_bridge_v23_witness,
    probe_kv_bridge_v23_long_horizon_reconstruction_falsifier,
)
from .long_horizon_reconstruction_substrate_v1 import (
    LongHorizonReconstructionQuery,
    build_default_long_horizon_reconstruction_carrier,
    emit_long_horizon_reconstruction_witness,
    reconstruct_long_horizon_event,
    report_reconstruction_vs_recompute_economics,
)
from .long_horizon_retention_v30 import (
    LongHorizonReconstructionV30Head,
    emit_lhr_v30_witness,
)
from .mergeable_latent_capsule_v26 import (
    MergeOperatorV26, emit_mlsc_v26_witness, wrap_v25_as_v26,
)
from .multi_agent_substrate_coordinator_v14 import (
    MultiAgentSubstrateCoordinatorV14,
    W78_MASC_V14_REGIMES,
    emit_multi_agent_substrate_coordinator_v14_witness,
)
from .persistent_latent_v30 import (
    PersistentLatentStateV30Chain,
    emit_persistent_v30_witness,
)
from .replay_controller import ReplayCandidate
from .replay_controller_v19 import (
    ReplayControllerV19,
    W78_LONG_HORIZON_RECONSTRUCTION_AWARE_ROUTING_LABELS,
    W78_REPLAY_REGIMES_V19,
    emit_replay_controller_v19_witness,
    fit_replay_controller_v19_per_role,
    fit_replay_v19_long_horizon_reconstruction_aware_routing_head,
)
from .substrate_adapter_v23 import (
    W78_SUBSTRATE_TIER_SUBSTRATE_V23_FULL,
    probe_all_v23_adapters,
)
from .team_consensus_controller_v13 import (
    TeamConsensusControllerV13,
    emit_team_consensus_controller_v13_witness,
)
from .tiny_substrate_v23 import (
    TinyV23SubstrateParams,
    build_default_tiny_substrate_v23,
    emit_tiny_substrate_v23_forward_witness,
    forward_tiny_substrate_v23,
    record_long_horizon_reconstruction_window_v23,
    tokenize_bytes_v23,
)
from .w77_team import W77HandoffEnvelope, W77Params, W77Team


W78_SCHEMA_VERSION: str = "coordpy.w78_team.v1"

W78_FAILURE_MODES: tuple[str, ...] = (
    "w78_outer_envelope_schema_mismatch",
    "w78_outer_envelope_w77_outer_cid_drift",
    "w78_outer_envelope_w78_params_cid_drift",
    "w78_outer_envelope_witness_cid_drift",
    "w78_substrate_v23_n_layers_off",
    "w78_substrate_v23_long_horizon_reconstruction_trajectory_cid_off",
    "w78_substrate_v23_long_horizon_reconstruction_length_per_layer_shape_off",
    "w78_substrate_v23_long_horizon_reconstruction_pressure_gate_shape_off",
    "w78_kv_bridge_v23_n_targets_off",
    "w78_kv_bridge_v23_long_horizon_reconstruction_falsifier_off",
    "w78_cache_v21_eighteen_objective_off",
    "w78_replay_v19_regime_count_off",
    "w78_replay_v19_long_horizon_reconstruction_routing_count_off",
    "w78_consensus_v24_stage_count_off",
    "w78_lhr_v30_max_k_off",
    "w78_lhr_v30_n_heads_off",
    "w78_persistent_v30_n_layers_off",
    "w78_substrate_adapter_v23_tier_off",
    "w78_masc_v14_v23_beats_v22_rate_under_threshold",
    "w78_masc_v14_tsc_v23_beats_tsc_v22_rate_under_threshold",
    "w78_masc_v14_long_horizon_reconstruction_regime_inferior_to_baseline",
    "w78_hosted_router_v11_decision_not_deterministic",
    "w78_hosted_logprob_v11_abstain_floor_off",
    "w78_hosted_cache_aware_v11_savings_below_89_percent",
    "w78_hosted_cost_planner_v11_no_eligible",
    "w78_hosted_real_substrate_boundary_v11_blocked_axis_satisfied",
    "w78_twenty_three_way_loop_not_observed",
    "w78_handoff_coordinator_v10_inconsistent",
    "w78_handoff_v10_cross_plane_savings_below_87_percent",
    "w78_team_consensus_v13_no_decisions",
    "w78_handoff_v10_long_horizon_reconstruction_alignment_off",
    "w78_handoff_envelope_v10_chain_cid_drift",
    "w78_inner_v77_envelope_invariant_off",
    "w78_handoff_v10_long_horizon_reconstruction_fallback_off",
    "w78_hosted_boundary_v11_blocked_axes_below_46",
    "w78_v23_substrate_self_checksum_cid_off",
    "w78_long_horizon_reconstruction_trajectory_cid_drift",
    "w78_mlsc_v26_long_horizon_reconstruction_trajectory_chain_off",
    "w78_v14_team_success_per_visible_token_below_floor",
    "w78_v14_visible_tokens_savings_below_70_percent",
    "w78_v14_long_horizon_reconstruction_regime_v23_beats_v22_below_threshold",
    "w78_substrate_v23_long_horizon_reconstruction_trajectory_chain_synthetic",
    "w78_inner_v23_falsifier_kind_off",
    "w78_handoff_v10_envelope_chain_alignment_off",
    "w78_hosted_router_v11_per_routing_cid_off",
    "w78_consensus_v24_long_horizon_reconstruction_arbiter_off",
    "w78_consensus_v24_long_horizon_reconstruction_best_parent_arbiter_off",
    "w78_tcc_v13_long_horizon_reconstruction_pressure_arbiter_off",
    "w78_tcc_v13_long_horizon_reconstruction_trajectory_arbiter_off",
    "w78_cache_v21_per_role_long_horizon_reconstruction_pressure_head_off",
    "w78_kv_bridge_v23_long_horizon_reconstruction_fingerprint_off",
    "w78_substrate_v23_long_horizon_reconstruction_windows_off",
    "w78_provider_filter_v10_pressure_drop_off",
    "w78_handoff_v10_alignment_off",
    "w78_handoff_v10_decision_label_off",
    "w78_bounded_window_baseline_not_failing_outside_window",
    "w78_long_horizon_reconstruction_substrate_not_reconstructing",
    "w78_bounded_window_insufficiency_not_proven",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class W78Params:
    substrate_v23: TinyV23SubstrateParams | None
    kv_bridge_v23: KVBridgeV23Projection | None
    cache_controller_v21: CacheControllerV21 | None
    replay_controller_v19: ReplayControllerV19 | None
    consensus_v24: ConsensusFallbackControllerV24 | None
    lhr_v30: LongHorizonReconstructionV30Head | None
    deep_substrate_hybrid_v23: DeepSubstrateHybridV23 | None
    mlsc_v26_operator: MergeOperatorV26 | None
    multi_agent_coordinator_v14: (
        MultiAgentSubstrateCoordinatorV14 | None)
    team_consensus_controller_v13: (
        TeamConsensusControllerV13 | None)
    hosted_registry: HostedProviderRegistry | None
    hosted_router_v11: HostedRouterControllerV11 | None
    hosted_logprob_router_v11: HostedLogprobRouterV11 | None
    hosted_cache_planner_v11: HostedCacheAwarePlannerV11 | None
    hosted_real_substrate_boundary_v11: (
        HostedRealSubstrateBoundaryV11 | None)
    handoff_coordinator_v10: (
        HostedRealHandoffCoordinatorV10 | None)
    hosted_provider_filter_v10: (
        HostedProviderFilterSpecV10 | None)
    w77_params: W77Params | None
    enabled: bool = True
    masc_v14_n_seeds: int = 5

    @classmethod
    def build_trivial(cls) -> "W78Params":
        return cls(
            substrate_v23=None,
            kv_bridge_v23=None,
            cache_controller_v21=None,
            replay_controller_v19=None,
            consensus_v24=None, lhr_v30=None,
            deep_substrate_hybrid_v23=None,
            mlsc_v26_operator=None,
            multi_agent_coordinator_v14=None,
            team_consensus_controller_v13=None,
            hosted_registry=None,
            hosted_router_v11=None,
            hosted_logprob_router_v11=None,
            hosted_cache_planner_v11=None,
            hosted_real_substrate_boundary_v11=None,
            handoff_coordinator_v10=None,
            hosted_provider_filter_v10=None,
            w77_params=None,
            enabled=False,
        )

    @classmethod
    def build_default(
            cls, *, seed: int = 78000,
    ) -> "W78Params":
        sub_v23 = build_default_tiny_substrate_v23(
            seed=int(seed) + 1)
        from .w77_team import W77Params as _W77P
        w77_p = _W77P.build_default(seed=int(seed) - 1000)
        kv_b23 = KVBridgeV23Projection.init_from_v22(
            w77_p.kv_bridge_v22, seed_v23=int(seed) + 27)
        cc21 = CacheControllerV21.init(
            fit_seed=int(seed) + 34)
        import numpy as _np
        rng = _np.random.default_rng(int(seed) + 35)
        X = rng.standard_normal((10, 4))
        cc21, _ = fit_eighteen_objective_ridge_v21(
            controller=cc21, train_features=X.tolist(),
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
                + X[:, 2] * 0.30 + X[:, 3] * 0.35).tolist(),
            target_post_restart_replacement_repair=(
                X[:, 0] * 0.1 + X[:, 1] * 0.2
                + X[:, 2] * 0.3 + X[:, 3] * 0.4).tolist(),
            target_long_horizon_reconstruction_repair=(
                X[:, 0] * 0.05 + X[:, 1] * 0.15
                + X[:, 2] * 0.30 + X[:, 3] * 0.50).tolist())
        X19 = rng.standard_normal((10, 19))
        cc21, _ = (
            fit_per_role_long_horizon_reconstruction_pressure_head_v21(
                controller=cc21, role="planner",
                train_features=X19.tolist(),
                target_long_horizon_reconstruction_priorities=(
                    X19[:, 0] * 0.2
                    + X19[:, 18] * 0.5).tolist()))
        rcv19 = ReplayControllerV19.init()
        v19_cands = {
            r: [ReplayCandidate(
                100, 1000, 50, 0.1, 0.0, 0.3,
                True, True, 0)]
            for r in W78_REPLAY_REGIMES_V19}
        v19_decs = {
            r: ["choose_reuse"]
            for r in W78_REPLAY_REGIMES_V19}
        rcv19, _ = fit_replay_controller_v19_per_role(
            controller=rcv19, role="planner",
            train_candidates_per_regime=v19_cands,
            train_decisions_per_regime=v19_decs)
        X_team = rng.standard_normal((64, 10))
        labs: list[str] = []
        for i in range(64):
            lab_idx = (
                i % len(
                    W78_LONG_HORIZON_RECONSTRUCTION_AWARE_ROUTING_LABELS))
            labs.append(
                W78_LONG_HORIZON_RECONSTRUCTION_AWARE_ROUTING_LABELS[
                    lab_idx])
        rcv19, _ = (
            fit_replay_v19_long_horizon_reconstruction_aware_routing_head(
                controller=rcv19,
                train_team_features=X_team.tolist(),
                train_routing_labels=labs))
        consensus_v24 = ConsensusFallbackControllerV24.init(
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
            post_compound_chain_restart_threshold=0.5,
            post_restart_replacement_threshold=0.5,
            post_restart_replacement_best_parent_threshold=0.5,
            long_horizon_reconstruction_threshold=0.5,
            long_horizon_reconstruction_best_parent_threshold=0.5)
        lhr30 = LongHorizonReconstructionV30Head.init(
            seed=int(seed) + 42)
        deep_v23 = DeepSubstrateHybridV23()
        mlsc_v26_op = MergeOperatorV26()
        masc_v14 = MultiAgentSubstrateCoordinatorV14()
        tcc_v13 = TeamConsensusControllerV13()
        reg = default_hosted_registry()
        hosted_router_v11 = HostedRouterControllerV11.init(
            reg, {
                "openrouter_paid": 0.88,
                "openai_paid": 0.95,
            })
        hosted_logprob_router_v11 = HostedLogprobRouterV11()
        hosted_cache_planner_v11 = HostedCacheAwarePlannerV11()
        boundary_v11 = (
            build_default_hosted_real_substrate_boundary_v11())
        handoff_coord_v10 = HostedRealHandoffCoordinatorV10(
            boundary_v11=boundary_v11)
        provider_filter_v10 = HostedProviderFilterSpecV10(
            inner_v9=w77_p.hosted_provider_filter_v9,
            long_horizon_reconstruction_pressure=0.85,
            long_horizon_reconstruction_pressure_floor=0.5,
            max_long_horizon_reconstruction_noise_per_provider={
                "openrouter_paid": 0.08,
                "openai_paid": 1.0},
            long_horizon_reconstruction_tier_weights={
                "logprobs_and_prefix_cache": 1.0,
                "logprobs": 0.40,
                "prefix_cache": 0.30,
                "text_only": 0.10})
        return cls(
            substrate_v23=sub_v23,
            kv_bridge_v23=kv_b23,
            cache_controller_v21=cc21,
            replay_controller_v19=rcv19,
            consensus_v24=consensus_v24,
            lhr_v30=lhr30,
            deep_substrate_hybrid_v23=deep_v23,
            mlsc_v26_operator=mlsc_v26_op,
            multi_agent_coordinator_v14=masc_v14,
            team_consensus_controller_v13=tcc_v13,
            hosted_registry=reg,
            hosted_router_v11=hosted_router_v11,
            hosted_logprob_router_v11=hosted_logprob_router_v11,
            hosted_cache_planner_v11=hosted_cache_planner_v11,
            hosted_real_substrate_boundary_v11=boundary_v11,
            handoff_coordinator_v10=handoff_coord_v10,
            hosted_provider_filter_v10=provider_filter_v10,
            w77_params=w77_p,
            enabled=True,
            masc_v14_n_seeds=5,
        )

    def to_dict(self) -> dict[str, Any]:
        def _cid_or_empty(x: Any) -> str:
            return str(x.cid()) if x is not None else ""
        return {
            "schema": W78_SCHEMA_VERSION,
            "kind": "w78_params",
            "substrate_v23_cid": _cid_or_empty(
                self.substrate_v23),
            "kv_bridge_v23_cid": _cid_or_empty(
                self.kv_bridge_v23),
            "cache_controller_v21_cid": _cid_or_empty(
                self.cache_controller_v21),
            "replay_controller_v19_cid": _cid_or_empty(
                self.replay_controller_v19),
            "consensus_v24_cid": _cid_or_empty(
                self.consensus_v24),
            "lhr_v30_cid": _cid_or_empty(self.lhr_v30),
            "deep_substrate_hybrid_v23_cid": _cid_or_empty(
                self.deep_substrate_hybrid_v23),
            "mlsc_v26_operator_cid": _cid_or_empty(
                self.mlsc_v26_operator),
            "multi_agent_coordinator_v14_cid": _cid_or_empty(
                self.multi_agent_coordinator_v14),
            "team_consensus_controller_v13_cid": _cid_or_empty(
                self.team_consensus_controller_v13),
            "hosted_registry_cid": _cid_or_empty(
                self.hosted_registry),
            "hosted_router_v11_cid": _cid_or_empty(
                self.hosted_router_v11),
            "hosted_logprob_router_v11_cid": _cid_or_empty(
                self.hosted_logprob_router_v11),
            "hosted_cache_planner_v11_cid": _cid_or_empty(
                self.hosted_cache_planner_v11),
            "hosted_real_substrate_boundary_v11_cid":
                _cid_or_empty(
                    self.hosted_real_substrate_boundary_v11),
            "handoff_coordinator_v10_cid": _cid_or_empty(
                self.handoff_coordinator_v10),
            "hosted_provider_filter_v10_cid": _cid_or_empty(
                self.hosted_provider_filter_v10),
            "w77_params_cid": _cid_or_empty(self.w77_params),
            "enabled": bool(self.enabled),
            "masc_v14_n_seeds": int(self.masc_v14_n_seeds),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w78_params",
            "params": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class W78HandoffEnvelope:
    schema: str
    w77_outer_cid: str
    w78_params_cid: str
    substrate_v23_witness_cid: str
    kv_bridge_v23_witness_cid: str
    cache_controller_v21_witness_cid: str
    replay_controller_v19_witness_cid: str
    persistent_v30_witness_cid: str
    mlsc_v26_witness_cid: str
    consensus_v24_witness_cid: str
    lhr_v30_witness_cid: str
    deep_substrate_hybrid_v23_witness_cid: str
    substrate_adapter_v23_matrix_cid: str
    masc_v14_witness_cid: str
    team_consensus_controller_v13_witness_cid: str
    long_horizon_reconstruction_pressure_falsifier_witness_cid: (
        str)
    hosted_router_v11_witness_cid: str
    hosted_logprob_router_v11_witness_cid: str
    hosted_cache_planner_v11_witness_cid: str
    hosted_real_substrate_boundary_v11_cid: str
    hosted_wall_v11_report_cid: str
    handoff_coordinator_v10_witness_cid: str
    handoff_envelope_v10_chain_cid: str
    provider_filter_v10_report_cid: str
    long_horizon_reconstruction_substrate_witness_cid: str
    bounded_window_baseline_witness_cid: str
    bounded_window_insufficiency_proof_cid: str
    twenty_three_way_used: bool
    substrate_v23_used: bool
    masc_v14_v23_beats_v22_rate: float
    masc_v14_tsc_v23_beats_tsc_v22_rate: float
    masc_v14_team_success_per_visible_token: float
    hosted_router_v11_chosen: str
    long_horizon_reconstruction_trajectory_cid: str
    long_horizon_reconstruction_success_rate: float
    bounded_window_all_fixed_k_failed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "w77_outer_cid": str(self.w77_outer_cid),
            "w78_params_cid": str(self.w78_params_cid),
            "substrate_v23_witness_cid": str(
                self.substrate_v23_witness_cid),
            "kv_bridge_v23_witness_cid": str(
                self.kv_bridge_v23_witness_cid),
            "cache_controller_v21_witness_cid": str(
                self.cache_controller_v21_witness_cid),
            "replay_controller_v19_witness_cid": str(
                self.replay_controller_v19_witness_cid),
            "persistent_v30_witness_cid": str(
                self.persistent_v30_witness_cid),
            "mlsc_v26_witness_cid": str(
                self.mlsc_v26_witness_cid),
            "consensus_v24_witness_cid": str(
                self.consensus_v24_witness_cid),
            "lhr_v30_witness_cid": str(
                self.lhr_v30_witness_cid),
            "deep_substrate_hybrid_v23_witness_cid": str(
                self.deep_substrate_hybrid_v23_witness_cid),
            "substrate_adapter_v23_matrix_cid": str(
                self.substrate_adapter_v23_matrix_cid),
            "masc_v14_witness_cid": str(
                self.masc_v14_witness_cid),
            "team_consensus_controller_v13_witness_cid": str(
                self.team_consensus_controller_v13_witness_cid),
            "long_horizon_reconstruction_pressure_falsifier_witness_cid":
                str(
                    self
                    .long_horizon_reconstruction_pressure_falsifier_witness_cid),
            "hosted_router_v11_witness_cid": str(
                self.hosted_router_v11_witness_cid),
            "hosted_logprob_router_v11_witness_cid": str(
                self.hosted_logprob_router_v11_witness_cid),
            "hosted_cache_planner_v11_witness_cid": str(
                self.hosted_cache_planner_v11_witness_cid),
            "hosted_real_substrate_boundary_v11_cid": str(
                self.hosted_real_substrate_boundary_v11_cid),
            "hosted_wall_v11_report_cid": str(
                self.hosted_wall_v11_report_cid),
            "handoff_coordinator_v10_witness_cid": str(
                self.handoff_coordinator_v10_witness_cid),
            "handoff_envelope_v10_chain_cid": str(
                self.handoff_envelope_v10_chain_cid),
            "provider_filter_v10_report_cid": str(
                self.provider_filter_v10_report_cid),
            "long_horizon_reconstruction_substrate_witness_cid":
                str(
                    self
                    .long_horizon_reconstruction_substrate_witness_cid),
            "bounded_window_baseline_witness_cid": str(
                self.bounded_window_baseline_witness_cid),
            "bounded_window_insufficiency_proof_cid": str(
                self.bounded_window_insufficiency_proof_cid),
            "twenty_three_way_used": bool(
                self.twenty_three_way_used),
            "substrate_v23_used": bool(self.substrate_v23_used),
            "masc_v14_v23_beats_v22_rate": float(round(
                self.masc_v14_v23_beats_v22_rate, 12)),
            "masc_v14_tsc_v23_beats_tsc_v22_rate": float(round(
                self.masc_v14_tsc_v23_beats_tsc_v22_rate, 12)),
            "masc_v14_team_success_per_visible_token": float(
                round(
                    self
                    .masc_v14_team_success_per_visible_token,
                    12)),
            "hosted_router_v11_chosen": str(
                self.hosted_router_v11_chosen),
            "long_horizon_reconstruction_trajectory_cid": str(
                self.long_horizon_reconstruction_trajectory_cid),
            "long_horizon_reconstruction_success_rate": float(
                round(
                    self
                    .long_horizon_reconstruction_success_rate,
                    12)),
            "bounded_window_all_fixed_k_failed": bool(
                self.bounded_window_all_fixed_k_failed),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w78_handoff_envelope",
            "envelope": self.to_dict()})


def verify_w78_handoff(
        envelope: W78HandoffEnvelope,
        params: W78Params,
        w77_outer_cid: str,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if envelope.schema != W78_SCHEMA_VERSION:
        failures.append(
            "w78_outer_envelope_schema_mismatch")
    if envelope.w77_outer_cid != str(w77_outer_cid):
        failures.append(
            "w78_outer_envelope_w77_outer_cid_drift")
    if envelope.w78_params_cid != params.cid():
        failures.append(
            "w78_outer_envelope_w78_params_cid_drift")
    return (len(failures) == 0), failures


@dataclasses.dataclass
class W78Team:
    params: W78Params

    def run_team_turn(
            self, *,
            w77_outer_cid: str,
            ids: Sequence[int] | None = None,
            text: str = "w78",
    ) -> W78HandoffEnvelope:
        p = self.params
        if not p.enabled or p.substrate_v23 is None:
            return W78HandoffEnvelope(
                schema=W78_SCHEMA_VERSION,
                w77_outer_cid=str(w77_outer_cid),
                w78_params_cid=str(p.cid()),
                substrate_v23_witness_cid="",
                kv_bridge_v23_witness_cid="",
                cache_controller_v21_witness_cid="",
                replay_controller_v19_witness_cid="",
                persistent_v30_witness_cid="",
                mlsc_v26_witness_cid="",
                consensus_v24_witness_cid="",
                lhr_v30_witness_cid="",
                deep_substrate_hybrid_v23_witness_cid="",
                substrate_adapter_v23_matrix_cid="",
                masc_v14_witness_cid="",
                team_consensus_controller_v13_witness_cid="",
                long_horizon_reconstruction_pressure_falsifier_witness_cid="",
                hosted_router_v11_witness_cid="",
                hosted_logprob_router_v11_witness_cid="",
                hosted_cache_planner_v11_witness_cid="",
                hosted_real_substrate_boundary_v11_cid="",
                hosted_wall_v11_report_cid="",
                handoff_coordinator_v10_witness_cid="",
                handoff_envelope_v10_chain_cid="",
                provider_filter_v10_report_cid="",
                long_horizon_reconstruction_substrate_witness_cid="",
                bounded_window_baseline_witness_cid="",
                bounded_window_insufficiency_proof_cid="",
                twenty_three_way_used=False,
                substrate_v23_used=False,
                masc_v14_v23_beats_v22_rate=0.0,
                masc_v14_tsc_v23_beats_tsc_v22_rate=0.0,
                masc_v14_team_success_per_visible_token=0.0,
                hosted_router_v11_chosen="",
                long_horizon_reconstruction_trajectory_cid="",
                long_horizon_reconstruction_success_rate=0.0,
                bounded_window_all_fixed_k_failed=False,
            )
        # Plane B — substrate V23 forward with W78 event.
        token_ids = (
            list(ids) if ids is not None
            else tokenize_bytes_v23(str(text), max_len=16))
        trace, cache = forward_tiny_substrate_v23(
            p.substrate_v23, token_ids,
            visible_token_budget=128.0,
            baseline_token_cost=512.0,
            restart_pressure=0.7,
            rejoin_pressure=0.6,
            replacement_pressure=0.7,
            contradiction_pressure=0.5,
            delayed_repair_pressure=0.6,
            compound_pressure=0.7,
            compound_chain_pressure=0.8,
            compound_chain_then_restart_pressure=0.85,
            post_restart_replacement_pressure=0.9,
            long_horizon_reconstruction_pressure=0.95)
        # Record W78 long-horizon-reconstruction window.
        record_long_horizon_reconstruction_window_v23(
            cache,
            compound_chain_failure_turn=30,
            reconstruction_request_turn=88,
            long_horizon_blackout_window_turns=58,
            role="planner_lhr", branch_id="main")
        trace, cache = forward_tiny_substrate_v23(
            p.substrate_v23, token_ids,
            v23_kv_cache=cache,
            visible_token_budget=128.0,
            baseline_token_cost=512.0,
            restart_pressure=0.7,
            rejoin_pressure=0.6,
            replacement_pressure=0.7,
            contradiction_pressure=0.5,
            delayed_repair_pressure=0.6,
            compound_pressure=0.7,
            compound_chain_pressure=0.8,
            compound_chain_then_restart_pressure=0.85,
            post_restart_replacement_pressure=0.9,
            long_horizon_reconstruction_pressure=0.95)
        sub_witness = emit_tiny_substrate_v23_forward_witness(
            trace, cache)
        # KV V23 witnesses.
        lhr_falsifier = (
            probe_kv_bridge_v23_long_horizon_reconstruction_falsifier(
                long_horizon_reconstruction_pressure_flag=1))
        lhrf = (
            compute_long_horizon_reconstruction_fingerprint_v23(
                role="planner",
                post_restart_replacement_trajectory_cid=str(
                    cache.v22_cache
                    .replacement_after_restart_after_compound_chain_trajectory_cid),
                long_horizon_reconstruction_trajectory_cid=str(
                    cache
                    .long_horizon_reconstruction_trajectory_cid),
                dominant_repair_label=1,
                long_horizon_reconstruction_count=int(len(
                    cache.long_horizon_reconstruction_windows)),
                long_horizon_blackout_window_turns=58,
                visible_token_budget=128.0,
                baseline_cost=512.0))
        kv_witness = emit_kv_bridge_v23_witness(
            projection=p.kv_bridge_v23,
            long_horizon_reconstruction_falsifier=lhr_falsifier,
            long_horizon_reconstruction_fingerprint=lhrf)
        cache_witness = emit_cache_controller_v21_witness(
            controller=p.cache_controller_v21)
        replay_witness = emit_replay_controller_v19_witness(
            p.replay_controller_v19)
        persist_chain = (
            PersistentLatentStateV30Chain.empty())
        persist_witness = emit_persistent_v30_witness(
            persist_chain)
        # MLSC V26 — wrap a trivial V25 capsule.
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
        from .mergeable_latent_capsule_v19 import wrap_v18_as_v19
        from .mergeable_latent_capsule_v20 import wrap_v19_as_v20
        from .mergeable_latent_capsule_v21 import wrap_v20_as_v21
        from .mergeable_latent_capsule_v22 import wrap_v21_as_v22
        from .mergeable_latent_capsule_v23 import wrap_v22_as_v23
        from .mergeable_latent_capsule_v24 import wrap_v23_as_v24
        from .mergeable_latent_capsule_v25 import wrap_v24_as_v25
        v3 = make_root_capsule_v3(
            branch_id="w78_smoke",
            payload=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            fact_tags=("w78",), confidence=0.9, trust=0.9,
            turn_index=0)
        v_chain = v3
        for wrap in (
                wrap_v3_as_v4, wrap_v4_as_v5, wrap_v5_as_v6,
                wrap_v6_as_v7, wrap_v7_as_v8, wrap_v8_as_v9,
                wrap_v9_as_v10, wrap_v10_as_v11,
                wrap_v11_as_v12, wrap_v12_as_v13,
                wrap_v13_as_v14, wrap_v14_as_v15,
                wrap_v15_as_v16, wrap_v16_as_v17,
                wrap_v17_as_v18, wrap_v18_as_v19):
            v_chain = wrap(v_chain)
        v_chain = wrap_v19_as_v20(
            v_chain,
            restart_repair_trajectory_chain=(),
            rejoin_pressure_chain=())
        v_chain = wrap_v20_as_v21(
            v_chain,
            replacement_repair_trajectory_chain=(),
            contradiction_chain=())
        v_chain = wrap_v21_as_v22(
            v_chain,
            compound_repair_trajectory_chain=(),
            delayed_repair_chain=())
        v_chain = wrap_v22_as_v23(
            v_chain,
            compound_chain_repair_trajectory_chain=(),
            replacement_then_rejoin_chain=())
        v_chain = wrap_v23_as_v24(
            v_chain,
            chain_then_restart_trajectory_chain=(),
            post_compound_chain_restart_chain=())
        v_chain = wrap_v24_as_v25(
            v_chain,
            post_restart_replacement_trajectory_chain=(
                str(cache.v22_cache
                    .replacement_after_restart_after_compound_chain_trajectory_cid),),
            post_restart_replacement_window_chain=(
                f"prw_{int(len(cache.v22_cache.post_restart_replacement_windows))}",))
        v26_c = wrap_v25_as_v26(
            v_chain,
            long_horizon_reconstruction_trajectory_chain=(
                str(cache
                    .long_horizon_reconstruction_trajectory_cid),),
            reconstruction_request_window_chain=(
                f"rrw_{int(len(cache.long_horizon_reconstruction_windows))}",))
        mlsc_witness = emit_mlsc_v26_witness(v26_c)
        consensus_witness = emit_consensus_v24_witness(
            p.consensus_v24)
        lhr_witness = emit_lhr_v30_witness(
            p.lhr_v30, carrier=[0.1] * 6, k=16,
            long_horizon_reconstruction_indicator=[0.95] * 8,
            partial_contradiction_indicator=[0.5] * 8,
            multi_branch_rejoin_indicator=[0.6] * 8,
            repair_dominance_indicator=[0.7] * 7,
            restart_indicator=[0.5] * 8,
            rejoin_indicator=[0.6] * 8,
            replacement_indicator=[0.7] * 8,
            compound_indicator=[0.8] * 8,
            compound_chain_indicator=[0.85] * 8,
            chain_then_restart_indicator=[0.90] * 8,
            post_restart_replacement_indicator=[0.92] * 8)
        # Deep substrate hybrid V23.
        v22_witness = DeepSubstrateHybridV22ForwardWitness(
            schema="coordpy.deep_substrate_hybrid_v22.v1",
            hybrid_cid="",
            inner_v21_witness_cid="",
            twenty_two_way=True,
            cache_controller_v20_fired=True,
            replay_controller_v18_fired=True,
            post_restart_replacement_trajectory_active=True,
            post_restart_replacement_repair_active=True,
            team_consensus_controller_v12_active=True,
            post_restart_replacement_trajectory_cid=str(
                cache.v22_cache
                .replacement_after_restart_after_compound_chain_trajectory_cid),
            post_restart_replacement_repair_l1=1,
            post_restart_replacement_pressure_gate_mean=0.5,
        )
        deep_v23_witness = deep_substrate_hybrid_v23_forward(
            hybrid=p.deep_substrate_hybrid_v23,
            v22_witness=v22_witness,
            cache_controller_v21=p.cache_controller_v21,
            replay_controller_v19=p.replay_controller_v19,
            long_horizon_reconstruction_trajectory_cid=str(
                cache
                .long_horizon_reconstruction_trajectory_cid),
            long_horizon_reconstruction_repair_l1=int(
                sub_witness.long_horizon_reconstruction_l1),
            long_horizon_reconstruction_pressure_gate_mean=float(
                trace
                .long_horizon_reconstruction_pressure_gate_per_layer
                .mean()),
            n_team_consensus_v13_invocations=1)
        adapter_matrix = probe_all_v23_adapters()
        # MASC V14 — run a batch across all 18 regimes.
        per_regime_aggs = {}
        for regime in W78_MASC_V14_REGIMES:
            _, agg = p.multi_agent_coordinator_v14.run_batch(
                seeds=list(range(int(p.masc_v14_n_seeds))),
                regime=regime)
            per_regime_aggs[regime] = agg
        masc_witness = (
            emit_multi_agent_substrate_coordinator_v14_witness(
                coordinator=p.multi_agent_coordinator_v14,
                per_regime_aggregate=per_regime_aggs))
        # TCC V13 — fire each new arbiter.
        tcc_v13 = p.team_consensus_controller_v13
        tcc_v13.decide_v13(
            regime=(
                "long_delay_reconstruction_after_compound_chain_failure"),
            agent_guesses=[1.0, -1.0, 0.5, 0.2],
            agent_confidences=[0.8, 0.6, 0.7, 0.7],
            substrate_replay_trust=0.7,
            long_horizon_reconstruction_trajectory_cid=str(
                cache
                .long_horizon_reconstruction_trajectory_cid),
            long_horizon_blackout_window_turns=58,
            agent_long_horizon_reconstruction_absorption_scores=[
                0.98, 0.6, 0.5, 0.4])
        tcc_v13.decide_v13(
            regime="baseline",
            agent_guesses=[0.5, 0.5, 0.4, 0.5],
            agent_confidences=[0.8, 0.6, 0.7, 0.7],
            substrate_replay_trust=0.7,
            long_horizon_reconstruction_pressure=0.90,
            agent_long_horizon_reconstruction_recovery_flags=[
                1, 0, 1, 0])
        tcc_witness = emit_team_consensus_controller_v13_witness(
            tcc_v13)
        # Plane A V11 — hosted.
        planned, _ = (
            p.hosted_cache_planner_v11
            .plan_per_role_nine_layer_rotated(
                shared_prefix_text=(
                    "W78 team shared prefix " * 16),
                per_role_blocks={
                    "plan": ["t0", "t1"],
                    "research": ["r0", "r1"],
                    "write": ["w0", "w1"],
                    "review": ["v0", "v1"],
                }))
        # Router V11.
        from .hosted_router_controller import (
            HostedRoutingRequest as _R1,
        )
        from .hosted_router_controller_v2 import (
            HostedRoutingRequestV2 as _R2,
        )
        from .hosted_router_controller_v3 import (
            HostedRoutingRequestV3 as _R3,
        )
        from .hosted_router_controller_v4 import (
            HostedRoutingRequestV4 as _R4,
        )
        from .hosted_router_controller_v5 import (
            HostedRoutingRequestV5 as _R5,
        )
        from .hosted_router_controller_v6 import (
            HostedRoutingRequestV6 as _R6,
        )
        from .hosted_router_controller_v7 import (
            HostedRoutingRequestV7 as _R7,
        )
        from .hosted_router_controller_v8 import (
            HostedRoutingRequestV8 as _R8,
        )
        from .hosted_router_controller_v9 import (
            HostedRoutingRequestV9 as _R9,
        )
        req_v9 = _R9(
            inner_v8=_R8(
                inner_v7=_R7(
                    inner_v6=_R6(
                        inner_v5=_R5(
                            inner_v4=_R4(
                                inner_v3=_R3(
                                    inner_v2=_R2(
                                        inner_v1=_R1(
                                            request_cid=(
                                                "w78-router-turn"),
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
        req_v10 = HostedRoutingRequestV10(
            inner_v9=req_v9,
            post_restart_replacement_pressure=0.9,
            weight_post_restart_replacement_pressure=0.6,
            weight_post_restart_replacement_after_pcr_match=0.4)
        req_v11 = HostedRoutingRequestV11(
            inner_v10=req_v10,
            long_horizon_reconstruction_pressure=0.95,
            weight_long_horizon_reconstruction_pressure=0.6,
            weight_long_horizon_reconstruction_after_pcr_match=0.4)
        router_dec = p.hosted_router_v11.decide_v11(req_v11)
        router_v11_witness = (
            emit_hosted_router_controller_v11_witness(
                p.hosted_router_v11))
        logprob_v11_witness = (
            emit_hosted_logprob_router_v11_witness(
                p.hosted_logprob_router_v11))
        cache_planner_v11_witness = (
            emit_hosted_cache_aware_planner_v11_witness(
                p.hosted_cache_planner_v11))
        boundary_v11 = p.hosted_real_substrate_boundary_v11
        wall_v11_report = build_wall_report_v11(
            boundary=boundary_v11)
        # Provider filter V10.
        _, filter_report = filter_hosted_registry_v10(
            p.hosted_registry, p.hosted_provider_filter_v10,
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
                "openai_paid": 0.03},
            provider_post_restart_replacement_noise={
                "openrouter_paid": 0.15,
                "openai_paid": 0.02},
            provider_long_horizon_reconstruction_noise={
                "openrouter_paid": 0.12,
                "openai_paid": 0.01})
        filter_report_cid = _sha256_hex({
            "kind": "w78_provider_filter_v10_report",
            "report": dict(filter_report),
        })
        # Handoff coordinator V10 decisions.
        def _make_req_v10(
                rc: str,
                long_horizon_reconstruction_pressure: float = 0.0,
                long_horizon_reconstruction_trajectory_cid: str = (
                    ""),
                long_horizon_blackout_window_turns: int = 0,
                needs_text_only: bool = True,
                needs_substrate_state_access: bool = False,
        ) -> HandoffRequestV10:
            from .hosted_real_handoff_coordinator_v9 import (
                HandoffRequestV9,
            )
            from .hosted_real_handoff_coordinator_v8 import (
                HandoffRequestV8,
            )
            from .hosted_real_handoff_coordinator_v7 import (
                HandoffRequestV7,
            )
            from .hosted_real_handoff_coordinator_v6 import (
                HandoffRequestV6,
            )
            from .hosted_real_handoff_coordinator_v5 import (
                HandoffRequestV5,
            )
            from .hosted_real_handoff_coordinator_v4 import (
                HandoffRequestV4,
            )
            from .hosted_real_handoff_coordinator_v3 import (
                HandoffRequestV3,
            )
            from .hosted_real_handoff_coordinator_v2 import (
                HandoffRequestV2,
            )
            from .hosted_real_handoff_coordinator import (
                HandoffRequest,
            )
            return HandoffRequestV10(
                inner_v9=HandoffRequestV9(
                    inner_v8=HandoffRequestV8(
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
                        compound_chain_then_restart_pressure=0.0,
                        compound_chain_then_restart_trajectory_cid="",
                        post_compound_chain_restart_window_turns=0,
                        expected_substrate_trust_v8=0.7),
                    post_restart_replacement_pressure=0.0,
                    post_restart_replacement_trajectory_cid="",
                    post_restart_replacement_window_turns=0,
                    expected_substrate_trust_v9=0.7),
                long_horizon_reconstruction_pressure=float(
                    long_horizon_reconstruction_pressure),
                long_horizon_reconstruction_trajectory_cid=str(
                    long_horizon_reconstruction_trajectory_cid),
                long_horizon_blackout_window_turns=int(
                    long_horizon_blackout_window_turns),
                expected_substrate_trust_v10=0.7)
        env_text_only = p.handoff_coordinator_v10.decide_v10(
            req_v10=_make_req_v10("w78-turn-text"))
        env_lhr_promoted = p.handoff_coordinator_v10.decide_v10(
            req_v10=_make_req_v10(
                "w78-turn-lhr",
                long_horizon_reconstruction_pressure=0.95,
                long_horizon_reconstruction_trajectory_cid=str(
                    cache
                    .long_horizon_reconstruction_trajectory_cid),
                long_horizon_blackout_window_turns=58))
        env_lhr_fallback = p.handoff_coordinator_v10.decide_v10(
            req_v10=_make_req_v10(
                "w78-turn-lhr-f",
                long_horizon_reconstruction_pressure=0.0,
                long_horizon_reconstruction_trajectory_cid=str(
                    cache
                    .long_horizon_reconstruction_trajectory_cid),
                long_horizon_blackout_window_turns=58))
        env_substrate_only = p.handoff_coordinator_v10.decide_v10(
            req_v10=_make_req_v10(
                "w78-turn-substrate",
                needs_text_only=False,
                needs_substrate_state_access=True))
        handoff_v10_witness = (
            emit_hosted_real_handoff_coordinator_v10_witness(
                p.handoff_coordinator_v10))
        handoff_envelope_chain_cid = _sha256_hex({
            "kind": "w78_handoff_envelope_v10_chain",
            "envelopes": [
                env_text_only.cid(),
                env_lhr_promoted.cid(),
                env_lhr_fallback.cid(),
                env_substrate_only.cid(),
            ],
        })
        # Long-horizon-reconstruction substrate.
        carrier = (
            build_default_long_horizon_reconstruction_carrier(
                n_events=256, seed=int(seed_for_carrier(p))))
        recon_queries = [
            LongHorizonReconstructionQuery(
                query_id=f"q{i}",
                source_turn=int(i * 32),
                current_turn=int(220 + i * 5))
            for i in range(5)]
        recon_outcomes = [
            reconstruct_long_horizon_event(
                carrier=carrier, query=q,
                visible_tokens_used=4)
            for q in recon_queries]
        recon_economics = (
            report_reconstruction_vs_recompute_economics(
                query=recon_queries[-1], carrier=carrier))
        recon_witness = emit_long_horizon_reconstruction_witness(
            carrier=carrier, outcomes=recon_outcomes,
            economics=recon_economics)
        # Bounded-window baseline falsifier.
        baselines = build_default_bounded_window_baselines()
        bw_query = BoundedWindowQuery(
            query_id="bw_q1", current_turn=220,
            source_turn=30,
            expected_event_cid=str(
                carrier.entries[30].event_cid))
        _, bw_falsifier = run_bounded_window_falsifier(
            baselines=baselines, query=bw_query)
        bw_proof = prove_bounded_window_insufficient(
            query_horizon_turns=190,
            baselines=baselines)
        bw_witness = emit_bounded_window_baseline_witness(
            baselines=baselines, proof=bw_proof,
            falsifier=bw_falsifier)
        baseline_agg = per_regime_aggs.get("baseline")
        v23_beats = (
            float(baseline_agg.v23_beats_v22_rate)
            if baseline_agg is not None else 0.0)
        tsc_v23_beats = (
            float(baseline_agg.tsc_v23_beats_tsc_v22_rate)
            if baseline_agg is not None else 0.0)
        ts_per_vt = (
            float(
                baseline_agg.team_success_per_visible_token_v23)
            if baseline_agg is not None else 0.0)
        n_succ = int(sum(
            1 for o in recon_outcomes if o.success))
        succ_rate = (
            float(n_succ) / float(len(recon_outcomes))
            if recon_outcomes else 0.0)
        return W78HandoffEnvelope(
            schema=W78_SCHEMA_VERSION,
            w77_outer_cid=str(w77_outer_cid),
            w78_params_cid=str(p.cid()),
            substrate_v23_witness_cid=str(sub_witness.cid()),
            kv_bridge_v23_witness_cid=str(kv_witness.cid()),
            cache_controller_v21_witness_cid=str(
                cache_witness.cid()),
            replay_controller_v19_witness_cid=str(
                replay_witness.cid()),
            persistent_v30_witness_cid=str(
                persist_witness.cid()),
            mlsc_v26_witness_cid=str(mlsc_witness.cid()),
            consensus_v24_witness_cid=str(
                consensus_witness.cid()),
            lhr_v30_witness_cid=str(lhr_witness.cid()),
            deep_substrate_hybrid_v23_witness_cid=str(
                deep_v23_witness.cid()),
            substrate_adapter_v23_matrix_cid=str(
                adapter_matrix.cid()),
            masc_v14_witness_cid=str(masc_witness.cid()),
            team_consensus_controller_v13_witness_cid=str(
                tcc_witness.cid()),
            long_horizon_reconstruction_pressure_falsifier_witness_cid=(
                str(lhr_falsifier.cid())),
            hosted_router_v11_witness_cid=str(
                router_v11_witness.cid()),
            hosted_logprob_router_v11_witness_cid=str(
                logprob_v11_witness.cid()),
            hosted_cache_planner_v11_witness_cid=str(
                cache_planner_v11_witness.cid()),
            hosted_real_substrate_boundary_v11_cid=str(
                boundary_v11.cid()),
            hosted_wall_v11_report_cid=str(
                wall_v11_report.cid()),
            handoff_coordinator_v10_witness_cid=str(
                handoff_v10_witness.cid()),
            handoff_envelope_v10_chain_cid=str(
                handoff_envelope_chain_cid),
            provider_filter_v10_report_cid=str(
                filter_report_cid),
            long_horizon_reconstruction_substrate_witness_cid=(
                str(recon_witness.cid())),
            bounded_window_baseline_witness_cid=str(
                bw_witness.cid()),
            bounded_window_insufficiency_proof_cid=str(
                bw_proof.cid()),
            twenty_three_way_used=bool(
                deep_v23_witness.twenty_three_way),
            substrate_v23_used=True,
            masc_v14_v23_beats_v22_rate=float(v23_beats),
            masc_v14_tsc_v23_beats_tsc_v22_rate=float(
                tsc_v23_beats),
            masc_v14_team_success_per_visible_token=float(
                ts_per_vt),
            hosted_router_v11_chosen=str(
                router_dec.chosen_provider or ""),
            long_horizon_reconstruction_trajectory_cid=str(
                cache.long_horizon_reconstruction_trajectory_cid),
            long_horizon_reconstruction_success_rate=float(
                succ_rate),
            bounded_window_all_fixed_k_failed=bool(
                bw_falsifier.all_fixed_k_failed),
        )


def seed_for_carrier(p: W78Params) -> int:
    """Derive a deterministic carrier seed from the W78 params
    chain."""
    cid = str(p.cid())
    # Reduce hex digest to int via mod.
    h = int(cid[:8], 16) & 0x7FFFFFFF
    return int(h) ^ 0x78A78A78


def build_default_w78_team(*, seed: int = 78000) -> W78Team:
    return W78Team(params=W78Params.build_default(seed=int(seed)))


__all__ = [
    "W78_SCHEMA_VERSION",
    "W78_FAILURE_MODES",
    "W78Params",
    "W78HandoffEnvelope",
    "verify_w78_handoff",
    "W78Team",
    "build_default_w78_team",
]
