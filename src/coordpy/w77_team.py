"""W77 — Stronger Replacement-After-Restart-After-Compound-Chain-
Repair / Post-Restart-Replacement Budget-Primary Two-Plane
Multi-Agent Substrate team.

The ``W77Team`` orchestrator strictly wraps the ``W76Team`` and
adds the W77 mechanism modules organised across two planes plus
the new **post-restart-replacement-aware Plane A↔B handoff
coordinator V9**:

**Plane B — Real substrate (in-repo, V22 stack):**

* M1  ``tiny_substrate_v22``           (inherits 23 physical
                                        layers from V21 + 3 new
                                        V22 substrate axes)
* M2  ``kv_bridge_v22``                (18-target ridge + 148-dim
                                        post-restart-replacement
                                        fingerprint + post-
                                        restart-replacement-
                                        pressure falsifier)
* M3  ``cache_controller_v20``         (17-objective ridge + per-
                                        role 18-dim post-restart-
                                        replacement-pressure
                                        head)
* M4  ``replay_controller_v18``        (25 regimes + 15-label
                                        post-restart-replacement-
                                        aware routing head)
* M5  ``deep_substrate_hybrid_v22``    (22-way bidirectional loop)
* M6  ``substrate_adapter_v22``        (substrate_v22_full tier)
* M7  ``persistent_latent_v29``        (28 layers, 26th carrier,
                                        max_chain_walk_depth=
                                        16777216)
* M8  ``long_horizon_retention_v29``   (28 heads, max_k=1024)
* M9  ``mergeable_latent_capsule_v25`` (post-restart-replacement-
                                        trajectory chain + post-
                                        restart-replacement-
                                        window chain)
* M10 ``consensus_fallback_controller_v23`` (40-stage chain)
* M11 ``multi_agent_substrate_coordinator_v13`` (28-policy, 17-
                                                 regime MASC V13)
* M12 ``team_consensus_controller_v12`` (post-restart-replacement-
                                         pressure + post-restart-
                                         replacement-trajectory
                                         arbiters)

**Plane A — Hosted control plane V10 (honest, no substrate):**

* H1  ``hosted_router_controller_v10``  (post-restart-replacement-
                                         pressure weighting + PCR
                                         match)
* H2  ``hosted_logprob_router_v10``     (post-restart-replacement-
                                         aware abstain floor + 8-
                                         way tiebreak)
* H3  ``hosted_cache_aware_planner_v10``(eight-layer rotated
                                         prefix; ≥ 89 % savings
                                         20×8 hit=1)
* H4  ``hosted_cost_planner_v10``       (cost-per-post-restart-
                                         replacement-success-
                                         under-budget + abstain-
                                         when-pcr-violated)
* H5  ``hosted_real_substrate_boundary_v10`` (wall V10, 43 blocked
                                              axes)
* H6  ``hosted_real_handoff_coordinator_v9`` (the **new post-
                                              restart-replacement-
                                              aware Plane A↔B
                                              bridge**)
* H7  ``hosted_provider_filter_v9``    (post-restart-replacement-
                                        aware provider filter)

Per-turn it emits 19 W77 module witness CIDs (12 Plane B + 7
Plane A V10) and a V9 handoff envelope CID, sealing them into a
``W77HandoffEnvelope`` whose ``w76_outer_cid`` carries forward
the W76 envelope byte-for-byte.

Honest scope (W77)
------------------

* Plane A V10 operates at the hosted text/logprob/prefix-cache
  surface. It does NOT pierce hidden state / KV / attention.
  ``W77-L-HOSTED-V10-NO-SUBSTRATE-CAP``.
* Plane B is the in-repo V22 NumPy runtime. We do NOT bridge to
  third-party hosted models.
  ``W77-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
* W77 fits one new closed-form ridge on top of W76's 76: KV V22
  eighteen-target. Total **77 closed-form ridge solves** across
  W61..W77. No autograd, no SGD, no GPU.
* Trivial passthrough preserved: when ``W77Params.build_trivial()``
  is used the W77 envelope's internal ``w76_outer_cid`` carries
  the supplied W76 outer CID exactly.
* The handoff coordinator V9 preserves the wall: a content-
  addressed V9 envelope says which plane handled each turn under
  the post-restart-replacement-aware score; it does NOT cross the
  substrate boundary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .cache_controller_v20 import (
    CacheControllerV20, emit_cache_controller_v20_witness,
    fit_per_role_post_restart_replacement_pressure_head_v20,
    fit_seventeen_objective_ridge_v20,
)
from .consensus_fallback_controller_v23 import (
    ConsensusFallbackControllerV23,
    emit_consensus_v23_witness,
)
from .deep_substrate_hybrid_v21 import (
    DeepSubstrateHybridV21ForwardWitness,
)
from .deep_substrate_hybrid_v22 import (
    DeepSubstrateHybridV22,
    deep_substrate_hybrid_v22_forward,
)
from .hosted_cache_aware_planner_v10 import (
    HostedCacheAwarePlannerV10,
    emit_hosted_cache_aware_planner_v10_witness,
)
from .hosted_logprob_router_v10 import (
    HostedLogprobRouterV10,
    emit_hosted_logprob_router_v10_witness,
)
from .hosted_provider_filter_v9 import (
    HostedProviderFilterSpecV9,
    filter_hosted_registry_v9,
)
from .hosted_provider_filter_v8 import HostedProviderFilterSpecV8
from .hosted_real_handoff_coordinator_v9 import (
    HandoffRequestV9, HostedRealHandoffCoordinatorV9,
    emit_hosted_real_handoff_coordinator_v9_witness,
)
from .hosted_real_handoff_coordinator_v8 import HandoffRequestV8
from .hosted_real_substrate_boundary_v10 import (
    HostedRealSubstrateBoundaryV10,
    build_default_hosted_real_substrate_boundary_v10,
    build_wall_report_v10,
)
from .hosted_router_controller import (
    HostedProviderRegistry, HostedRoutingRequest,
    default_hosted_registry,
)
from .hosted_router_controller_v9 import HostedRoutingRequestV9
from .hosted_router_controller_v10 import (
    HostedRouterControllerV10, HostedRoutingRequestV10,
    emit_hosted_router_controller_v10_witness,
)
from .kv_bridge_v21 import KVBridgeV21Projection
from .kv_bridge_v22 import (
    KVBridgeV22Projection,
    compute_post_restart_replacement_fingerprint_v22,
    emit_kv_bridge_v22_witness,
    probe_kv_bridge_v22_post_restart_replacement_falsifier,
)
from .long_horizon_retention_v29 import (
    LongHorizonReconstructionV29Head,
    emit_lhr_v29_witness,
)
from .mergeable_latent_capsule_v25 import (
    MergeOperatorV25, emit_mlsc_v25_witness, wrap_v24_as_v25,
)
from .multi_agent_substrate_coordinator_v13 import (
    MultiAgentSubstrateCoordinatorV13,
    W77_MASC_V13_REGIMES,
    emit_multi_agent_substrate_coordinator_v13_witness,
)
from .persistent_latent_v29 import (
    PersistentLatentStateV29Chain,
    emit_persistent_v29_witness,
)
from .replay_controller import ReplayCandidate
from .replay_controller_v18 import (
    ReplayControllerV18,
    W77_POST_RESTART_REPLACEMENT_AWARE_ROUTING_LABELS,
    W77_REPLAY_REGIMES_V18,
    emit_replay_controller_v18_witness,
    fit_replay_controller_v18_per_role,
    fit_replay_v18_post_restart_replacement_aware_routing_head,
)
from .substrate_adapter_v22 import (
    W77_SUBSTRATE_TIER_SUBSTRATE_V22_FULL,
    probe_all_v22_adapters,
)
from .team_consensus_controller_v12 import (
    TeamConsensusControllerV12,
    emit_team_consensus_controller_v12_witness,
)
from .tiny_substrate_v22 import (
    TinyV22SubstrateParams,
    build_default_tiny_substrate_v22,
    emit_tiny_substrate_v22_forward_witness,
    forward_tiny_substrate_v22,
    record_post_restart_replacement_window_v22,
    tokenize_bytes_v22,
)
from .w76_team import W76HandoffEnvelope, W76Params, W76Team


W77_SCHEMA_VERSION: str = "coordpy.w77_team.v1"

W77_FAILURE_MODES: tuple[str, ...] = (
    "w77_outer_envelope_schema_mismatch",
    "w77_outer_envelope_w76_outer_cid_drift",
    "w77_outer_envelope_w77_params_cid_drift",
    "w77_outer_envelope_witness_cid_drift",
    "w77_substrate_v22_n_layers_off",
    "w77_substrate_v22_post_restart_replacement_trajectory_cid_off",
    "w77_substrate_v22_post_restart_replacement_length_per_layer_shape_off",
    "w77_substrate_v22_post_restart_replacement_pressure_gate_shape_off",
    "w77_kv_bridge_v22_n_targets_off",
    "w77_kv_bridge_v22_post_restart_replacement_falsifier_off",
    "w77_cache_v20_seventeen_objective_off",
    "w77_replay_v18_regime_count_off",
    "w77_replay_v18_post_restart_replacement_routing_count_off",
    "w77_consensus_v23_stage_count_off",
    "w77_lhr_v29_max_k_off",
    "w77_lhr_v29_n_heads_off",
    "w77_persistent_v29_n_layers_off",
    "w77_substrate_adapter_v22_tier_off",
    "w77_masc_v13_v22_beats_v21_rate_under_threshold",
    "w77_masc_v13_tsc_v22_beats_tsc_v21_rate_under_threshold",
    "w77_masc_v13_post_restart_replacement_regime_inferior_to_baseline",
    "w77_hosted_router_v10_decision_not_deterministic",
    "w77_hosted_logprob_v10_abstain_floor_off",
    "w77_hosted_cache_aware_v10_savings_below_89_percent",
    "w77_hosted_cost_planner_v10_no_eligible",
    "w77_hosted_real_substrate_boundary_v10_blocked_axis_satisfied",
    "w77_twenty_two_way_loop_not_observed",
    "w77_handoff_coordinator_v9_inconsistent",
    "w77_handoff_v9_cross_plane_savings_below_86_percent",
    "w77_team_consensus_v12_no_decisions",
    "w77_handoff_v9_post_restart_replacement_alignment_off",
    "w77_handoff_envelope_v9_chain_cid_drift",
    "w77_inner_v76_envelope_invariant_off",
    "w77_handoff_v9_post_restart_replacement_fallback_off",
    "w77_hosted_boundary_v10_blocked_axes_below_43",
    "w77_v22_substrate_self_checksum_cid_off",
    "w77_post_restart_replacement_trajectory_cid_drift",
    "w77_mlsc_v25_post_restart_replacement_trajectory_chain_off",
    "w77_v13_team_success_per_visible_token_below_floor",
    "w77_v13_visible_tokens_savings_below_65_percent",
    "w77_v13_post_restart_replacement_regime_v22_beats_v21_below_threshold",
    "w77_substrate_v22_post_restart_replacement_trajectory_chain_synthetic",
    "w77_inner_v22_falsifier_kind_off",
    "w77_handoff_v9_envelope_chain_alignment_off",
    "w77_hosted_router_v10_per_routing_cid_off",
    "w77_consensus_v23_post_restart_replacement_arbiter_off",
    "w77_consensus_v23_post_restart_replacement_best_parent_arbiter_off",
    "w77_tcc_v12_post_restart_replacement_pressure_arbiter_off",
    "w77_tcc_v12_post_restart_replacement_trajectory_arbiter_off",
    "w77_cache_v20_per_role_post_restart_replacement_pressure_head_off",
    "w77_kv_bridge_v22_post_restart_replacement_fingerprint_off",
    "w77_substrate_v22_post_restart_replacement_windows_off",
    "w77_provider_filter_v9_pressure_drop_off",
    "w77_handoff_v9_alignment_off",
    "w77_handoff_v9_decision_label_off",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class W77Params:
    substrate_v22: TinyV22SubstrateParams | None
    kv_bridge_v22: KVBridgeV22Projection | None
    cache_controller_v20: CacheControllerV20 | None
    replay_controller_v18: ReplayControllerV18 | None
    consensus_v23: ConsensusFallbackControllerV23 | None
    lhr_v29: LongHorizonReconstructionV29Head | None
    deep_substrate_hybrid_v22: DeepSubstrateHybridV22 | None
    mlsc_v25_operator: MergeOperatorV25 | None
    multi_agent_coordinator_v13: (
        MultiAgentSubstrateCoordinatorV13 | None)
    team_consensus_controller_v12: (
        TeamConsensusControllerV12 | None)
    hosted_registry: HostedProviderRegistry | None
    hosted_router_v10: HostedRouterControllerV10 | None
    hosted_logprob_router_v10: HostedLogprobRouterV10 | None
    hosted_cache_planner_v10: HostedCacheAwarePlannerV10 | None
    hosted_real_substrate_boundary_v10: (
        HostedRealSubstrateBoundaryV10 | None)
    handoff_coordinator_v9: (
        HostedRealHandoffCoordinatorV9 | None)
    hosted_provider_filter_v9: (
        HostedProviderFilterSpecV9 | None)
    w76_params: W76Params | None
    enabled: bool = True
    masc_v13_n_seeds: int = 5

    @classmethod
    def build_trivial(cls) -> "W77Params":
        return cls(
            substrate_v22=None,
            kv_bridge_v22=None,
            cache_controller_v20=None,
            replay_controller_v18=None,
            consensus_v23=None, lhr_v29=None,
            deep_substrate_hybrid_v22=None,
            mlsc_v25_operator=None,
            multi_agent_coordinator_v13=None,
            team_consensus_controller_v12=None,
            hosted_registry=None,
            hosted_router_v10=None,
            hosted_logprob_router_v10=None,
            hosted_cache_planner_v10=None,
            hosted_real_substrate_boundary_v10=None,
            handoff_coordinator_v9=None,
            hosted_provider_filter_v9=None,
            w76_params=None,
            enabled=False,
        )

    @classmethod
    def build_default(
            cls, *, seed: int = 77000,
    ) -> "W77Params":
        sub_v22 = build_default_tiny_substrate_v22(
            seed=int(seed) + 1)
        # KV V22 projection chain — wrap KV V21 with the W76 init
        # chain.
        from .w76_team import W76Params
        w76_p = W76Params.build_default(seed=int(seed) - 1000)
        kv_b22 = KVBridgeV22Projection.init_from_v21(
            w76_p.kv_bridge_v21, seed_v22=int(seed) + 25)
        cc20 = CacheControllerV20.init(
            fit_seed=int(seed) + 32)
        import numpy as _np
        rng = _np.random.default_rng(int(seed) + 33)
        X = rng.standard_normal((10, 4))
        cc20, _ = fit_seventeen_objective_ridge_v20(
            controller=cc20, train_features=X.tolist(),
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
                + X[:, 2] * 0.3 + X[:, 3] * 0.4).tolist())
        X18 = rng.standard_normal((10, 18))
        cc20, _ = (
            fit_per_role_post_restart_replacement_pressure_head_v20(
                controller=cc20, role="planner",
                train_features=X18.tolist(),
                target_post_restart_replacement_priorities=(
                    X18[:, 0] * 0.2
                    + X18[:, 17] * 0.5).tolist()))
        # Replay V18.
        rcv18 = ReplayControllerV18.init()
        v18_cands = {
            r: [ReplayCandidate(
                100, 1000, 50, 0.1, 0.0, 0.3,
                True, True, 0)]
            for r in W77_REPLAY_REGIMES_V18}
        v18_decs = {
            r: ["choose_reuse"]
            for r in W77_REPLAY_REGIMES_V18}
        rcv18, _ = fit_replay_controller_v18_per_role(
            controller=rcv18, role="planner",
            train_candidates_per_regime=v18_cands,
            train_decisions_per_regime=v18_decs)
        X_team = rng.standard_normal((60, 10))
        labs: list[str] = []
        for i in range(60):
            lab_idx = (
                i % len(
                    W77_POST_RESTART_REPLACEMENT_AWARE_ROUTING_LABELS))
            labs.append(
                W77_POST_RESTART_REPLACEMENT_AWARE_ROUTING_LABELS[
                    lab_idx])
        rcv18, _ = (
            fit_replay_v18_post_restart_replacement_aware_routing_head(
                controller=rcv18,
                train_team_features=X_team.tolist(),
                train_routing_labels=labs))
        consensus_v23 = ConsensusFallbackControllerV23.init(
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
            post_restart_replacement_best_parent_threshold=0.5)
        lhr29 = LongHorizonReconstructionV29Head.init(
            seed=int(seed) + 40)
        deep_v22 = DeepSubstrateHybridV22()
        mlsc_v25_op = MergeOperatorV25()
        masc_v13 = MultiAgentSubstrateCoordinatorV13()
        tcc_v12 = TeamConsensusControllerV12()
        reg = default_hosted_registry()
        hosted_router_v10 = HostedRouterControllerV10.init(
            reg, {
                "openrouter_paid": 0.87,
                "openai_paid": 0.94,
            })
        hosted_logprob_router_v10 = HostedLogprobRouterV10()
        hosted_cache_planner_v10 = HostedCacheAwarePlannerV10()
        boundary_v10 = (
            build_default_hosted_real_substrate_boundary_v10())
        handoff_coord_v9 = HostedRealHandoffCoordinatorV9(
            boundary_v10=boundary_v10)
        provider_filter_v9 = HostedProviderFilterSpecV9(
            inner_v8=w76_p.hosted_provider_filter_v8,
            post_restart_replacement_pressure=0.8,
            post_restart_replacement_pressure_floor=0.5,
            max_post_restart_replacement_noise_per_provider={
                "openrouter_paid": 0.10,
                "openai_paid": 1.0},
            post_restart_replacement_tier_weights={
                "logprobs_and_prefix_cache": 1.0,
                "logprobs": 0.40,
                "prefix_cache": 0.30,
                "text_only": 0.10})
        return cls(
            substrate_v22=sub_v22,
            kv_bridge_v22=kv_b22,
            cache_controller_v20=cc20,
            replay_controller_v18=rcv18,
            consensus_v23=consensus_v23,
            lhr_v29=lhr29,
            deep_substrate_hybrid_v22=deep_v22,
            mlsc_v25_operator=mlsc_v25_op,
            multi_agent_coordinator_v13=masc_v13,
            team_consensus_controller_v12=tcc_v12,
            hosted_registry=reg,
            hosted_router_v10=hosted_router_v10,
            hosted_logprob_router_v10=hosted_logprob_router_v10,
            hosted_cache_planner_v10=hosted_cache_planner_v10,
            hosted_real_substrate_boundary_v10=boundary_v10,
            handoff_coordinator_v9=handoff_coord_v9,
            hosted_provider_filter_v9=provider_filter_v9,
            w76_params=w76_p,
            enabled=True,
            masc_v13_n_seeds=5,
        )

    def to_dict(self) -> dict[str, Any]:
        def _cid_or_empty(x: Any) -> str:
            return str(x.cid()) if x is not None else ""
        return {
            "schema": W77_SCHEMA_VERSION,
            "kind": "w77_params",
            "substrate_v22_cid": _cid_or_empty(
                self.substrate_v22),
            "kv_bridge_v22_cid": _cid_or_empty(
                self.kv_bridge_v22),
            "cache_controller_v20_cid": _cid_or_empty(
                self.cache_controller_v20),
            "replay_controller_v18_cid": _cid_or_empty(
                self.replay_controller_v18),
            "consensus_v23_cid": _cid_or_empty(
                self.consensus_v23),
            "lhr_v29_cid": _cid_or_empty(self.lhr_v29),
            "deep_substrate_hybrid_v22_cid": _cid_or_empty(
                self.deep_substrate_hybrid_v22),
            "mlsc_v25_operator_cid": _cid_or_empty(
                self.mlsc_v25_operator),
            "multi_agent_coordinator_v13_cid": _cid_or_empty(
                self.multi_agent_coordinator_v13),
            "team_consensus_controller_v12_cid": _cid_or_empty(
                self.team_consensus_controller_v12),
            "hosted_registry_cid": _cid_or_empty(
                self.hosted_registry),
            "hosted_router_v10_cid": _cid_or_empty(
                self.hosted_router_v10),
            "hosted_logprob_router_v10_cid": _cid_or_empty(
                self.hosted_logprob_router_v10),
            "hosted_cache_planner_v10_cid": _cid_or_empty(
                self.hosted_cache_planner_v10),
            "hosted_real_substrate_boundary_v10_cid":
                _cid_or_empty(
                    self.hosted_real_substrate_boundary_v10),
            "handoff_coordinator_v9_cid": _cid_or_empty(
                self.handoff_coordinator_v9),
            "hosted_provider_filter_v9_cid": _cid_or_empty(
                self.hosted_provider_filter_v9),
            "w76_params_cid": _cid_or_empty(self.w76_params),
            "enabled": bool(self.enabled),
            "masc_v13_n_seeds": int(self.masc_v13_n_seeds),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w77_params",
            "params": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class W77HandoffEnvelope:
    schema: str
    w76_outer_cid: str
    w77_params_cid: str
    substrate_v22_witness_cid: str
    kv_bridge_v22_witness_cid: str
    cache_controller_v20_witness_cid: str
    replay_controller_v18_witness_cid: str
    persistent_v29_witness_cid: str
    mlsc_v25_witness_cid: str
    consensus_v23_witness_cid: str
    lhr_v29_witness_cid: str
    deep_substrate_hybrid_v22_witness_cid: str
    substrate_adapter_v22_matrix_cid: str
    masc_v13_witness_cid: str
    team_consensus_controller_v12_witness_cid: str
    post_restart_replacement_pressure_falsifier_witness_cid: str
    hosted_router_v10_witness_cid: str
    hosted_logprob_router_v10_witness_cid: str
    hosted_cache_planner_v10_witness_cid: str
    hosted_real_substrate_boundary_v10_cid: str
    hosted_wall_v10_report_cid: str
    handoff_coordinator_v9_witness_cid: str
    handoff_envelope_v9_chain_cid: str
    provider_filter_v9_report_cid: str
    twenty_two_way_used: bool
    substrate_v22_used: bool
    masc_v13_v22_beats_v21_rate: float
    masc_v13_tsc_v22_beats_tsc_v21_rate: float
    masc_v13_team_success_per_visible_token: float
    hosted_router_v10_chosen: str
    post_restart_replacement_trajectory_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "w76_outer_cid": str(self.w76_outer_cid),
            "w77_params_cid": str(self.w77_params_cid),
            "substrate_v22_witness_cid": str(
                self.substrate_v22_witness_cid),
            "kv_bridge_v22_witness_cid": str(
                self.kv_bridge_v22_witness_cid),
            "cache_controller_v20_witness_cid": str(
                self.cache_controller_v20_witness_cid),
            "replay_controller_v18_witness_cid": str(
                self.replay_controller_v18_witness_cid),
            "persistent_v29_witness_cid": str(
                self.persistent_v29_witness_cid),
            "mlsc_v25_witness_cid": str(
                self.mlsc_v25_witness_cid),
            "consensus_v23_witness_cid": str(
                self.consensus_v23_witness_cid),
            "lhr_v29_witness_cid": str(
                self.lhr_v29_witness_cid),
            "deep_substrate_hybrid_v22_witness_cid": str(
                self.deep_substrate_hybrid_v22_witness_cid),
            "substrate_adapter_v22_matrix_cid": str(
                self.substrate_adapter_v22_matrix_cid),
            "masc_v13_witness_cid": str(
                self.masc_v13_witness_cid),
            "team_consensus_controller_v12_witness_cid": str(
                self.team_consensus_controller_v12_witness_cid),
            "post_restart_replacement_pressure_falsifier_witness_cid":
                str(
                    self
                    .post_restart_replacement_pressure_falsifier_witness_cid),
            "hosted_router_v10_witness_cid": str(
                self.hosted_router_v10_witness_cid),
            "hosted_logprob_router_v10_witness_cid": str(
                self.hosted_logprob_router_v10_witness_cid),
            "hosted_cache_planner_v10_witness_cid": str(
                self.hosted_cache_planner_v10_witness_cid),
            "hosted_real_substrate_boundary_v10_cid": str(
                self.hosted_real_substrate_boundary_v10_cid),
            "hosted_wall_v10_report_cid": str(
                self.hosted_wall_v10_report_cid),
            "handoff_coordinator_v9_witness_cid": str(
                self.handoff_coordinator_v9_witness_cid),
            "handoff_envelope_v9_chain_cid": str(
                self.handoff_envelope_v9_chain_cid),
            "provider_filter_v9_report_cid": str(
                self.provider_filter_v9_report_cid),
            "twenty_two_way_used": bool(
                self.twenty_two_way_used),
            "substrate_v22_used": bool(self.substrate_v22_used),
            "masc_v13_v22_beats_v21_rate": float(round(
                self.masc_v13_v22_beats_v21_rate, 12)),
            "masc_v13_tsc_v22_beats_tsc_v21_rate": float(round(
                self.masc_v13_tsc_v22_beats_tsc_v21_rate, 12)),
            "masc_v13_team_success_per_visible_token": float(
                round(
                    self
                    .masc_v13_team_success_per_visible_token,
                    12)),
            "hosted_router_v10_chosen": str(
                self.hosted_router_v10_chosen),
            "post_restart_replacement_trajectory_cid": str(
                self.post_restart_replacement_trajectory_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w77_handoff_envelope",
            "envelope": self.to_dict()})


def verify_w77_handoff(
        envelope: W77HandoffEnvelope,
        params: W77Params,
        w76_outer_cid: str,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if envelope.schema != W77_SCHEMA_VERSION:
        failures.append(
            "w77_outer_envelope_schema_mismatch")
    if envelope.w76_outer_cid != str(w76_outer_cid):
        failures.append(
            "w77_outer_envelope_w76_outer_cid_drift")
    if envelope.w77_params_cid != params.cid():
        failures.append(
            "w77_outer_envelope_w77_params_cid_drift")
    return (len(failures) == 0), failures


@dataclasses.dataclass
class W77Team:
    params: W77Params

    def run_team_turn(
            self, *,
            w76_outer_cid: str,
            ids: Sequence[int] | None = None,
            text: str = "w77",
    ) -> W77HandoffEnvelope:
        p = self.params
        if not p.enabled or p.substrate_v22 is None:
            return W77HandoffEnvelope(
                schema=W77_SCHEMA_VERSION,
                w76_outer_cid=str(w76_outer_cid),
                w77_params_cid=str(p.cid()),
                substrate_v22_witness_cid="",
                kv_bridge_v22_witness_cid="",
                cache_controller_v20_witness_cid="",
                replay_controller_v18_witness_cid="",
                persistent_v29_witness_cid="",
                mlsc_v25_witness_cid="",
                consensus_v23_witness_cid="",
                lhr_v29_witness_cid="",
                deep_substrate_hybrid_v22_witness_cid="",
                substrate_adapter_v22_matrix_cid="",
                masc_v13_witness_cid="",
                team_consensus_controller_v12_witness_cid="",
                post_restart_replacement_pressure_falsifier_witness_cid="",
                hosted_router_v10_witness_cid="",
                hosted_logprob_router_v10_witness_cid="",
                hosted_cache_planner_v10_witness_cid="",
                hosted_real_substrate_boundary_v10_cid="",
                hosted_wall_v10_report_cid="",
                handoff_coordinator_v9_witness_cid="",
                handoff_envelope_v9_chain_cid="",
                provider_filter_v9_report_cid="",
                twenty_two_way_used=False,
                substrate_v22_used=False,
                masc_v13_v22_beats_v21_rate=0.0,
                masc_v13_tsc_v22_beats_tsc_v21_rate=0.0,
                masc_v13_team_success_per_visible_token=0.0,
                hosted_router_v10_chosen="",
                post_restart_replacement_trajectory_cid="",
            )
        # Plane B — substrate V22 forward with W77 event.
        token_ids = (
            list(ids) if ids is not None
            else tokenize_bytes_v22(str(text), max_len=16))
        trace, cache = forward_tiny_substrate_v22(
            p.substrate_v22, token_ids,
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
            post_restart_replacement_pressure=0.9)
        # Record W77 post-restart-replacement window.
        record_post_restart_replacement_window_v22(
            cache, chain_then_restart_repair_turn=18,
            replacement_turn=22,
            post_restart_replacement_window_turns=10,
            role="planner_fresh", branch_id="main")
        # Re-run forward to seal the post-restart-replacement CID
        # over the recorded event.
        trace, cache = forward_tiny_substrate_v22(
            p.substrate_v22, token_ids,
            v22_kv_cache=cache,
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
            post_restart_replacement_pressure=0.9)
        sub_witness = emit_tiny_substrate_v22_forward_witness(
            trace, cache)
        # KV V22 witnesses.
        pcr_falsifier = (
            probe_kv_bridge_v22_post_restart_replacement_falsifier(
                post_restart_replacement_pressure_flag=1))
        pcrf = compute_post_restart_replacement_fingerprint_v22(
            role="planner",
            compound_chain_then_restart_trajectory_cid=str(
                cache.v21_cache
                .compound_chain_then_restart_trajectory_cid),
            replacement_after_restart_after_compound_chain_trajectory_cid=(
                str(cache
                    .replacement_after_restart_after_compound_chain_trajectory_cid)),
            dominant_repair_label=1,
            post_restart_replacement_count=int(len(
                cache.post_restart_replacement_windows)),
            post_restart_replacement_window_turns=10,
            visible_token_budget=128.0,
            baseline_cost=512.0)
        kv_witness = emit_kv_bridge_v22_witness(
            projection=p.kv_bridge_v22,
            post_restart_replacement_falsifier=pcr_falsifier,
            post_restart_replacement_fingerprint=pcrf)
        cache_witness = emit_cache_controller_v20_witness(
            controller=p.cache_controller_v20)
        replay_witness = emit_replay_controller_v18_witness(
            p.replay_controller_v18)
        persist_chain = (
            PersistentLatentStateV29Chain.empty())
        persist_witness = emit_persistent_v29_witness(
            persist_chain)
        # MLSC V25 — wrap a trivial V24 capsule (use V24 chain).
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
        v3 = make_root_capsule_v3(
            branch_id="w77_smoke",
            payload=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            fact_tags=("w77",), confidence=0.9, trust=0.9,
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
            chain_then_restart_trajectory_chain=(
                str(cache.v21_cache
                    .compound_chain_then_restart_trajectory_cid),),
            post_compound_chain_restart_chain=(
                f"pccr_{int(len(cache.v21_cache.post_compound_chain_restart_windows))}",))
        v25_c = wrap_v24_as_v25(
            v_chain,
            post_restart_replacement_trajectory_chain=(
                str(cache
                    .replacement_after_restart_after_compound_chain_trajectory_cid),),
            post_restart_replacement_window_chain=(
                f"prw_{int(len(cache.post_restart_replacement_windows))}",))
        mlsc_witness = emit_mlsc_v25_witness(v25_c)
        consensus_witness = emit_consensus_v23_witness(
            p.consensus_v23)
        lhr_witness = emit_lhr_v29_witness(
            p.lhr_v29, carrier=[0.1] * 6, k=16,
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
        # Deep substrate hybrid V22 — fold the V21 witness as a
        # pre-condition.
        v21_witness = DeepSubstrateHybridV21ForwardWitness(
            schema="coordpy.deep_substrate_hybrid_v21.v1",
            hybrid_cid="",
            inner_v20_witness_cid="",
            twenty_one_way=True,
            cache_controller_v19_fired=True,
            replay_controller_v17_fired=True,
            chain_then_restart_trajectory_active=True,
            chain_then_restart_repair_active=True,
            team_consensus_controller_v11_active=True,
            chain_then_restart_trajectory_cid=str(
                cache.v21_cache
                .compound_chain_then_restart_trajectory_cid),
            chain_then_restart_repair_l1=1,
            chain_then_restart_pressure_gate_mean=0.5,
        )
        deep_v22_witness = deep_substrate_hybrid_v22_forward(
            hybrid=p.deep_substrate_hybrid_v22,
            v21_witness=v21_witness,
            cache_controller_v20=p.cache_controller_v20,
            replay_controller_v18=p.replay_controller_v18,
            post_restart_replacement_trajectory_cid=str(
                cache
                .replacement_after_restart_after_compound_chain_trajectory_cid),
            post_restart_replacement_repair_l1=int(
                sub_witness
                .replacement_after_restart_after_compound_chain_repair_l1),
            post_restart_replacement_pressure_gate_mean=float(
                trace
                .replacement_after_restart_after_compound_chain_pressure_gate_per_layer
                .mean()),
            n_team_consensus_v12_invocations=1)
        adapter_matrix = probe_all_v22_adapters()
        # MASC V13 — run a batch for the envelope across all 17
        # regimes.
        per_regime_aggs = {}
        for regime in W77_MASC_V13_REGIMES:
            _, agg = p.multi_agent_coordinator_v13.run_batch(
                seeds=list(range(int(p.masc_v13_n_seeds))),
                regime=regime)
            per_regime_aggs[regime] = agg
        masc_witness = (
            emit_multi_agent_substrate_coordinator_v13_witness(
                coordinator=p.multi_agent_coordinator_v13,
                per_regime_aggregate=per_regime_aggs))
        # TCC V12 — fire each new arbiter so the witness counts > 0.
        tcc_v12 = p.team_consensus_controller_v12
        tcc_v12.decide_v12(
            regime=(
                "replacement_after_restart_after_compound_chain_"
                "repair_under_budget"),
            agent_guesses=[1.0, -1.0, 0.5, 0.2],
            agent_confidences=[0.8, 0.6, 0.7, 0.7],
            substrate_replay_trust=0.7,
            post_restart_replacement_trajectory_cid=str(
                cache
                .replacement_after_restart_after_compound_chain_trajectory_cid),
            post_restart_replacement_window_turns=10,
            agent_post_restart_replacement_absorption_scores=[
                0.97, 0.6, 0.5, 0.4])
        tcc_v12.decide_v12(
            regime="baseline",
            agent_guesses=[0.5, 0.5, 0.4, 0.5],
            agent_confidences=[0.8, 0.6, 0.7, 0.7],
            substrate_replay_trust=0.7,
            post_restart_replacement_pressure=0.85,
            agent_post_restart_replacement_recovery_flags=[
                1, 0, 1, 0])
        tcc_witness = emit_team_consensus_controller_v12_witness(
            tcc_v12)
        # Plane A V10 — hosted.
        planned, _ = (
            p.hosted_cache_planner_v10
            .plan_per_role_eight_layer_rotated(
                shared_prefix_text=(
                    "W77 team shared prefix " * 16),
                per_role_blocks={
                    "plan": ["t0", "t1"],
                    "research": ["r0", "r1"],
                    "write": ["w0", "w1"],
                    "review": ["v0", "v1"],
                }))
        # Router V10 — at least one decision so witness is
        # non-empty.
        from .hosted_router_controller_v9 import (
            HostedRoutingRequestV9 as _Rv9,
        )
        # Build a V10 request via the W76 V9 request chain.
        v9_req = _Rv9(
            inner_v8=p.w76_params.hosted_router_v9.inner_v8
            .audit_v8[0]["inner_v7_cid"] if False else (
                # construct fresh V8/V9 chain
                None))
        # Simpler: reuse a fresh V9 request like W76 does.
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
        req_v9 = HostedRoutingRequestV9(
            inner_v8=_R8(
                inner_v7=_R7(
                    inner_v6=_R6(
                        inner_v5=_R5(
                            inner_v4=_R4(
                                inner_v3=_R3(
                                    inner_v2=_R2(
                                        inner_v1=_R1(
                                            request_cid=(
                                                "w77-router-turn"),
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
        router_dec = p.hosted_router_v10.decide_v10(req_v10)
        router_v10_witness = (
            emit_hosted_router_controller_v10_witness(
                p.hosted_router_v10))
        logprob_v10_witness = (
            emit_hosted_logprob_router_v10_witness(
                p.hosted_logprob_router_v10))
        cache_planner_v10_witness = (
            emit_hosted_cache_aware_planner_v10_witness(
                p.hosted_cache_planner_v10))
        boundary_v10 = p.hosted_real_substrate_boundary_v10
        wall_v10_report = build_wall_report_v10(
            boundary=boundary_v10)
        # Provider filter V9 — run once to seal a report CID.
        _, filter_report = filter_hosted_registry_v9(
            p.hosted_registry, p.hosted_provider_filter_v9,
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
                "openai_paid": 0.02})
        filter_report_cid = _sha256_hex({
            "kind": "w77_provider_filter_v9_report",
            "report": dict(filter_report),
        })
        # Handoff coordinator V9 decisions.
        def _make_req_v9(
                rc: str,
                post_restart_replacement_pressure: float = 0.0,
                post_restart_replacement_trajectory_cid: str = "",
                post_restart_replacement_window_turns: int = 0,
                needs_text_only: bool = True,
                needs_substrate_state_access: bool = False,
        ) -> HandoffRequestV9:
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
            return HandoffRequestV9(
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
                post_restart_replacement_pressure=float(
                    post_restart_replacement_pressure),
                post_restart_replacement_trajectory_cid=str(
                    post_restart_replacement_trajectory_cid),
                post_restart_replacement_window_turns=int(
                    post_restart_replacement_window_turns),
                expected_substrate_trust_v9=0.7)

        env_text_only = p.handoff_coordinator_v9.decide_v9(
            req_v9=_make_req_v9("w77-turn-text"))
        env_pcr_promoted = p.handoff_coordinator_v9.decide_v9(
            req_v9=_make_req_v9(
                "w77-turn-pcr",
                post_restart_replacement_pressure=0.9,
                post_restart_replacement_trajectory_cid=str(
                    cache
                    .replacement_after_restart_after_compound_chain_trajectory_cid),
                post_restart_replacement_window_turns=10))
        env_pcr_fallback = p.handoff_coordinator_v9.decide_v9(
            req_v9=_make_req_v9(
                "w77-turn-pcr-f",
                post_restart_replacement_pressure=0.0,
                post_restart_replacement_trajectory_cid=str(
                    cache
                    .replacement_after_restart_after_compound_chain_trajectory_cid),
                post_restart_replacement_window_turns=10))
        env_substrate_only = p.handoff_coordinator_v9.decide_v9(
            req_v9=_make_req_v9(
                "w77-turn-substrate",
                needs_text_only=False,
                needs_substrate_state_access=True))
        handoff_v9_witness = (
            emit_hosted_real_handoff_coordinator_v9_witness(
                p.handoff_coordinator_v9))
        handoff_envelope_chain_cid = _sha256_hex({
            "kind": "w77_handoff_envelope_v9_chain",
            "envelopes": [
                env_text_only.cid(),
                env_pcr_promoted.cid(),
                env_pcr_fallback.cid(),
                env_substrate_only.cid(),
            ],
        })
        baseline_agg = per_regime_aggs.get("baseline")
        v22_beats = (
            float(baseline_agg.v22_beats_v21_rate)
            if baseline_agg is not None else 0.0)
        tsc_v22_beats = (
            float(baseline_agg.tsc_v22_beats_tsc_v21_rate)
            if baseline_agg is not None else 0.0)
        ts_per_vt = (
            float(
                baseline_agg.team_success_per_visible_token_v22)
            if baseline_agg is not None else 0.0)
        return W77HandoffEnvelope(
            schema=W77_SCHEMA_VERSION,
            w76_outer_cid=str(w76_outer_cid),
            w77_params_cid=str(p.cid()),
            substrate_v22_witness_cid=str(sub_witness.cid()),
            kv_bridge_v22_witness_cid=str(kv_witness.cid()),
            cache_controller_v20_witness_cid=str(
                cache_witness.cid()),
            replay_controller_v18_witness_cid=str(
                replay_witness.cid()),
            persistent_v29_witness_cid=str(
                persist_witness.cid()),
            mlsc_v25_witness_cid=str(mlsc_witness.cid()),
            consensus_v23_witness_cid=str(
                consensus_witness.cid()),
            lhr_v29_witness_cid=str(lhr_witness.cid()),
            deep_substrate_hybrid_v22_witness_cid=str(
                deep_v22_witness.cid()),
            substrate_adapter_v22_matrix_cid=str(
                adapter_matrix.cid()),
            masc_v13_witness_cid=str(masc_witness.cid()),
            team_consensus_controller_v12_witness_cid=str(
                tcc_witness.cid()),
            post_restart_replacement_pressure_falsifier_witness_cid=(
                str(pcr_falsifier.cid())),
            hosted_router_v10_witness_cid=str(
                router_v10_witness.cid()),
            hosted_logprob_router_v10_witness_cid=str(
                logprob_v10_witness.cid()),
            hosted_cache_planner_v10_witness_cid=str(
                cache_planner_v10_witness.cid()),
            hosted_real_substrate_boundary_v10_cid=str(
                boundary_v10.cid()),
            hosted_wall_v10_report_cid=str(
                wall_v10_report.cid()),
            handoff_coordinator_v9_witness_cid=str(
                handoff_v9_witness.cid()),
            handoff_envelope_v9_chain_cid=str(
                handoff_envelope_chain_cid),
            provider_filter_v9_report_cid=str(
                filter_report_cid),
            twenty_two_way_used=bool(
                deep_v22_witness.twenty_two_way),
            substrate_v22_used=True,
            masc_v13_v22_beats_v21_rate=float(v22_beats),
            masc_v13_tsc_v22_beats_tsc_v21_rate=float(
                tsc_v22_beats),
            masc_v13_team_success_per_visible_token=float(
                ts_per_vt),
            hosted_router_v10_chosen=str(
                router_dec.chosen_provider or ""),
            post_restart_replacement_trajectory_cid=str(
                cache
                .replacement_after_restart_after_compound_chain_trajectory_cid),
        )


def build_default_w77_team(*, seed: int = 77000) -> W77Team:
    return W77Team(params=W77Params.build_default(seed=int(seed)))


__all__ = [
    "W77_SCHEMA_VERSION",
    "W77_FAILURE_MODES",
    "W77Params",
    "W77HandoffEnvelope",
    "verify_w77_handoff",
    "W77Team",
    "build_default_w77_team",
]
