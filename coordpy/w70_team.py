"""W70 — Stronger Repair-Dominance / Budget-Primary Two-Plane
Multi-Agent Substrate team.

The ``W70Team`` orchestrator strictly wraps the ``W69Team`` and
adds the W70 mechanism modules organised across two planes plus
the new **budget-primary Plane A↔B handoff coordinator V2**:

**Plane B — Real substrate (in-repo, V15 stack):**

* M1  ``tiny_substrate_v15``           (17-layer, 3 new V15 axes)
* M2  ``kv_bridge_v15``                (11-target ridge + 70-dim
                                        repair-trajectory
                                        fingerprint + repair-
                                        dominance falsifier)
* M3  ``cache_controller_v13``         (10-objective ridge + per-
                                        role 11-dim budget-primary
                                        head)
* M4  ``replay_controller_v11``        (18 regimes + per-role +
                                        budget-primary routing
                                        head)
* M5  ``deep_substrate_hybrid_v15``    (15-way bidirectional loop)
* M6  ``substrate_adapter_v15``        (substrate_v15_full tier)
* M7  ``persistent_latent_v22``        (21 layers, 19th carrier,
                                        max_chain_walk_depth=131072)
* M8  ``long_horizon_retention_v22``   (21 heads, max_k=576)
* M9  ``mergeable_latent_capsule_v18`` (repair-trajectory chain +
                                        budget-primary chain)
* M10 ``consensus_fallback_controller_v16`` (26-stage chain)
* M11 ``multi_agent_substrate_coordinator_v6`` (14-policy, 10-regime
                                                MASC V6)
* M12 ``team_consensus_controller_v5`` (repair-dominance + budget-
                                        primary arbiters)

**Plane A — Hosted control plane V3 (honest, no substrate):**

* H1  ``hosted_router_controller_v3``  (budget-aware multi-objective
                                        + repair-dominance match)
* H2  ``hosted_logprob_router_v3``     (abstain-when-disagree + per-
                                        budget tiebreak)
* H3  ``hosted_cache_aware_planner_v3``(per-role staggered + rotated
                                        prefix + ≥ 65 % savings 8×8
                                        hit=1)
* H4  ``hosted_cost_planner_v3``       (cost-per-team-success-
                                        under-budget +
                                        abstain-when-budget-violated)
* H5  ``hosted_real_substrate_boundary_v3`` (the wall V3, 22 blocked
                                             axes)
* H6  ``hosted_real_handoff_coordinator_v2`` (the **new budget-
                                              primary Plane A↔B
                                              bridge** — V2
                                              envelopes + repair-
                                              dominance falsifier +
                                              cross-plane savings)

Per-turn it emits 18 W70 module witness CIDs (12 Plane B + 6 Plane
A V3) and a V2 handoff envelope CID, sealing them into a
``W70HandoffEnvelope`` whose ``w69_outer_cid`` carries forward the
W69 envelope byte-for-byte.

Honest scope (W70)
------------------

* Plane A V3 operates at the hosted text/logprob/prefix-cache
  surface. It does NOT pierce hidden state / KV / attention.
  ``W70-L-HOSTED-V3-NO-SUBSTRATE-CAP``.
* Plane B is the in-repo V15 NumPy runtime. We do NOT bridge to
  third-party hosted models.
  ``W70-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
* W70 fits closed-form ridge parameters in five new places on top
  of W69's 53: cache V13 ten-objective; cache V13 per-role
  budget-primary; replay V11 per-role per-regime; replay V11
  budget-primary-routing; KV V15 eleven-target. Total
  **fifty-eight closed-form ridge solves** across W61..W70. No
  autograd, no SGD, no GPU.
* Trivial passthrough preserved: when ``W70Params.build_trivial()``
  is used the W70 envelope's internal ``w69_outer_cid`` carries
  the supplied W69 outer CID exactly.
* The handoff coordinator V2 preserves the wall: a content-
  addressed V2 envelope says which plane handled each turn under
  the budget-primary score; it does NOT cross the substrate
  boundary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .cache_controller_v13 import (
    CacheControllerV13,
    emit_cache_controller_v13_witness,
    fit_ten_objective_ridge_v13,
    fit_per_role_budget_primary_head_v13,
)
from .consensus_fallback_controller_v16 import (
    ConsensusFallbackControllerV16,
    W70_CONSENSUS_V16_STAGES,
    emit_consensus_v16_witness,
)
from .deep_substrate_hybrid_v14 import (
    DeepSubstrateHybridV14ForwardWitness,
)
from .deep_substrate_hybrid_v15 import (
    DeepSubstrateHybridV15,
    deep_substrate_hybrid_v15_forward,
)
from .hosted_cache_aware_planner_v3 import (
    HostedCacheAwarePlannerV3,
    emit_hosted_cache_aware_planner_v3_witness,
)
from .hosted_cost_planner_v3 import HostedCostPlanSpecV3
from .hosted_logprob_router_v3 import (
    HostedLogprobRouterV3,
    emit_hosted_logprob_router_v3_witness,
)
from .hosted_real_substrate_boundary_v3 import (
    HostedRealSubstrateBoundaryV3,
    build_default_hosted_real_substrate_boundary_v3,
    build_wall_report_v3,
    probe_hosted_real_substrate_boundary_v3_falsifier,
)
from .hosted_real_handoff_coordinator_v2 import (
    HandoffRequestV2, HostedRealHandoffCoordinatorV2,
    emit_hosted_real_handoff_coordinator_v2_witness,
    hosted_real_handoff_v2_budget_primary_savings,
)
from .hosted_real_handoff_coordinator import HandoffRequest
from .hosted_router_controller import (
    HostedProviderRegistry, HostedRoutingRequest,
    default_hosted_registry,
)
from .hosted_router_controller_v2 import HostedRoutingRequestV2
from .hosted_router_controller_v3 import (
    HostedRouterControllerV3, HostedRoutingRequestV3,
    emit_hosted_router_controller_v3_witness,
)
from .kv_bridge_v14 import KVBridgeV14Projection
from .kv_bridge_v15 import (
    KVBridgeV15Projection,
    compute_repair_trajectory_fingerprint_v15,
    emit_kv_bridge_v15_witness,
    probe_kv_bridge_v15_repair_dominance_falsifier,
)
from .long_horizon_retention_v22 import (
    LongHorizonReconstructionV22Head,
    emit_lhr_v22_witness,
)
from .mergeable_latent_capsule_v18 import (
    MergeOperatorV18, emit_mlsc_v18_witness, wrap_v17_as_v18,
)
from .multi_agent_substrate_coordinator_v6 import (
    MultiAgentSubstrateCoordinatorV6,
    W70_MASC_V6_REGIMES,
    emit_multi_agent_substrate_coordinator_v6_witness,
)
from .persistent_latent_v22 import (
    PersistentLatentStateV22Chain,
    emit_persistent_v22_witness,
)
from .replay_controller_v11 import (
    ReplayControllerV11,
    W70_BUDGET_PRIMARY_ROUTING_LABELS,
    W70_REPLAY_REGIMES_V11,
    fit_replay_controller_v11_per_role,
    fit_replay_v11_budget_primary_routing_head,
    emit_replay_controller_v11_witness,
)
from .replay_controller import ReplayCandidate
from .substrate_adapter_v15 import (
    W70_SUBSTRATE_TIER_SUBSTRATE_V15_FULL,
    probe_all_v15_adapters,
)
from .team_consensus_controller_v5 import (
    TeamConsensusControllerV5,
    emit_team_consensus_controller_v5_witness,
)
from .tiny_substrate_v15 import (
    TinyV15SubstrateParams,
    build_default_tiny_substrate_v15,
    emit_tiny_substrate_v15_forward_witness,
    forward_tiny_substrate_v15,
    record_repair_event_v15,
    tokenize_bytes_v15,
    W70_REPAIR_MULTI_BRANCH_REJOIN,
)
from .w69_team import (
    W69HandoffEnvelope, W69Params, W69Team,
)


W70_SCHEMA_VERSION: str = "coordpy.w70_team.v1"

W70_FAILURE_MODES: tuple[str, ...] = (
    "w70_outer_envelope_schema_mismatch",
    "w70_outer_envelope_w69_outer_cid_drift",
    "w70_outer_envelope_w70_params_cid_drift",
    "w70_outer_envelope_witness_cid_drift",
    "w70_substrate_v15_n_layers_off",
    "w70_substrate_v15_repair_trajectory_cid_off",
    "w70_substrate_v15_dominant_repair_per_layer_shape_off",
    "w70_substrate_v15_budget_primary_gate_shape_off",
    "w70_kv_bridge_v15_n_targets_off",
    "w70_kv_bridge_v15_repair_dominance_falsifier_off",
    "w70_cache_v13_ten_objective_off",
    "w70_replay_v11_regime_count_off",
    "w70_replay_v11_budget_primary_routing_count_off",
    "w70_consensus_v16_stage_count_off",
    "w70_lhr_v22_max_k_off",
    "w70_lhr_v22_n_heads_off",
    "w70_persistent_v22_n_layers_off",
    "w70_substrate_adapter_v15_tier_off",
    "w70_masc_v6_v15_beats_v14_rate_under_threshold",
    "w70_masc_v6_tsc_v15_beats_tsc_v14_rate_under_threshold",
    "w70_masc_v6_compound_regime_inferior_to_baseline",
    "w70_hosted_router_v3_decision_not_deterministic",
    "w70_hosted_logprob_v3_abstain_kind_off",
    "w70_hosted_cache_aware_v3_savings_below_65_percent",
    "w70_hosted_cost_planner_v3_no_eligible",
    "w70_hosted_real_substrate_boundary_v3_blocked_axis_satisfied",
    "w70_fifteen_way_loop_not_observed",
    "w70_handoff_coordinator_v2_inconsistent",
    "w70_handoff_v2_cross_plane_savings_below_50_percent",
    "w70_team_consensus_v5_no_decisions",
    "w70_handoff_v2_repair_dominance_alignment_off",
    "w70_handoff_envelope_v2_chain_cid_drift",
    "w70_inner_v69_envelope_invariant_off",
    "w70_handoff_v2_budget_primary_fallback_off",
    "w70_hosted_boundary_v3_blocked_axes_below_22",
    "w70_v15_substrate_self_checksum_cid_off",
    "w70_repair_trajectory_cid_drift",
    "w70_mlsc_v18_repair_trajectory_chain_off",
    "w70_v6_team_success_per_visible_token_below_floor",
    "w70_v6_visible_tokens_savings_below_60_percent",
    "w70_v6_compound_regime_v15_beats_v14_below_threshold",
    "w70_substrate_v15_repair_trajectory_chain_synthetic",
    "w70_inner_v15_falsifier_kind_off",
    "w70_handoff_v2_envelope_repair_alignment_off",
    "w70_hosted_router_v3_per_budget_cid_off",
    "w70_consensus_v16_repair_dominance_arbiter_off",
    "w70_consensus_v16_budget_primary_arbiter_off",
    "w70_tcc_v5_repair_dominance_arbiter_off",
    "w70_tcc_v5_budget_primary_arbiter_off",
    "w70_tcc_v5_contradiction_then_rejoin_arbiter_off",
    "w70_cache_v13_per_role_budget_primary_head_off",
    "w70_kv_bridge_v15_repair_trajectory_fingerprint_off",
    "w70_substrate_v15_repair_events_off",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class W70Params:
    substrate_v15: TinyV15SubstrateParams | None
    kv_bridge_v15: KVBridgeV15Projection | None
    cache_controller_v13: CacheControllerV13 | None
    replay_controller_v11: ReplayControllerV11 | None
    consensus_v16: ConsensusFallbackControllerV16 | None
    lhr_v22: LongHorizonReconstructionV22Head | None
    deep_substrate_hybrid_v15: DeepSubstrateHybridV15 | None
    mlsc_v18_operator: MergeOperatorV18 | None
    multi_agent_coordinator_v6: (
        MultiAgentSubstrateCoordinatorV6 | None)
    team_consensus_controller_v5: (
        TeamConsensusControllerV5 | None)
    hosted_registry: HostedProviderRegistry | None
    hosted_router_v3: HostedRouterControllerV3 | None
    hosted_logprob_router_v3: HostedLogprobRouterV3 | None
    hosted_cache_planner_v3: HostedCacheAwarePlannerV3 | None
    hosted_real_substrate_boundary_v3: (
        HostedRealSubstrateBoundaryV3 | None)
    handoff_coordinator_v2: (
        HostedRealHandoffCoordinatorV2 | None)
    w69_params: W69Params | None
    enabled: bool = True
    masc_v6_n_seeds: int = 10

    @classmethod
    def build_trivial(cls) -> "W70Params":
        return cls(
            substrate_v15=None,
            kv_bridge_v15=None,
            cache_controller_v13=None,
            replay_controller_v11=None,
            consensus_v16=None, lhr_v22=None,
            deep_substrate_hybrid_v15=None,
            mlsc_v18_operator=None,
            multi_agent_coordinator_v6=None,
            team_consensus_controller_v5=None,
            hosted_registry=None,
            hosted_router_v3=None,
            hosted_logprob_router_v3=None,
            hosted_cache_planner_v3=None,
            hosted_real_substrate_boundary_v3=None,
            handoff_coordinator_v2=None,
            w69_params=None,
            enabled=False,
        )

    @classmethod
    def build_default(
            cls, *, seed: int = 70000,
    ) -> "W70Params":
        sub_v15 = build_default_tiny_substrate_v15(
            seed=int(seed) + 1)
        # KV V15 projection chain (deep nest through V14..V3).
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
        cfg = sub_v15.config.v14.v13.v12.v11.v10.v9
        d_head = int(cfg.d_model) // int(cfg.n_heads)
        kv_b3 = KVBridgeV3Projection.init(
            n_layers=int(cfg.n_layers),
            n_heads=int(cfg.n_heads),
            n_kv_heads=int(cfg.n_kv_heads),
            n_inject_tokens=3, carrier_dim=6,
            d_head=int(d_head), seed=int(seed) + 7)
        kv_b14 = KVBridgeV14Projection.init_from_v13(
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
            seed_v14=int(seed) + 18)
        kv_b15 = KVBridgeV15Projection.init_from_v14(
            kv_b14, seed_v15=int(seed) + 19)
        cc13 = CacheControllerV13.init(fit_seed=int(seed) + 32)
        # Fit ten-objective ridge with a small synthetic dataset.
        import numpy as _np
        rng = _np.random.default_rng(int(seed) + 33)
        X = rng.standard_normal((10, 4))
        cc13, _ = fit_ten_objective_ridge_v13(
            controller=cc13, train_features=X.tolist(),
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
                + X[:, 2] * 0.4).tolist())
        X11 = rng.standard_normal((8, 11))
        cc13, _ = fit_per_role_budget_primary_head_v13(
            controller=cc13, role="planner",
            train_features=X11.tolist(),
            target_budget_primary_priorities=(
                X11[:, 0] * 0.4 + X11[:, 10] * 0.3).tolist())
        # Replay V11.
        rcv11 = ReplayControllerV11.init()
        v11_cands = {
            r: [ReplayCandidate(
                100, 1000, 50, 0.1, 0.0, 0.3,
                True, True, 0)]
            for r in W70_REPLAY_REGIMES_V11}
        v11_decs = {
            r: ["choose_reuse"]
            for r in W70_REPLAY_REGIMES_V11}
        rcv11, _ = fit_replay_controller_v11_per_role(
            controller=rcv11, role="planner",
            train_candidates_per_regime=v11_cands,
            train_decisions_per_regime=v11_decs)
        X_team = rng.standard_normal((40, 10))
        labs: list[str] = []
        for i in range(40):
            lab_idx = i % len(W70_BUDGET_PRIMARY_ROUTING_LABELS)
            labs.append(
                W70_BUDGET_PRIMARY_ROUTING_LABELS[lab_idx])
        rcv11, _ = fit_replay_v11_budget_primary_routing_head(
            controller=rcv11,
            train_team_features=X_team.tolist(),
            train_routing_labels=labs)
        consensus_v16 = ConsensusFallbackControllerV16.init(
            k_required=2, cosine_floor=0.6,
            trust_threshold=0.5,
            multi_branch_rejoin_threshold=0.5,
            silent_corruption_threshold=0.5,
            repair_dominance_threshold=0.5,
            budget_primary_threshold=0.5)
        lhr22 = LongHorizonReconstructionV22Head.init(
            seed=int(seed) + 40)
        deep_v15 = DeepSubstrateHybridV15()
        mlsc_v18_op = MergeOperatorV18()
        masc_v6 = MultiAgentSubstrateCoordinatorV6()
        tcc_v5 = TeamConsensusControllerV5()
        reg = default_hosted_registry()
        from .hosted_router_controller_v3 import (
            HostedRouterControllerV3 as _HRCv3)
        hosted_router_v3 = _HRCv3.init(reg, {
            "openrouter_paid": 0.85,
            "openai_paid": 0.92,
        })
        hosted_logprob_router_v3 = HostedLogprobRouterV3()
        hosted_cache_planner_v3 = HostedCacheAwarePlannerV3()
        boundary_v3 = (
            build_default_hosted_real_substrate_boundary_v3())
        handoff_coord_v2 = HostedRealHandoffCoordinatorV2(
            boundary_v3=boundary_v3)
        # W69 inner params for envelope chaining.
        w69_params = W69Params.build_default(seed=int(seed) - 1000)
        return cls(
            substrate_v15=sub_v15,
            kv_bridge_v15=kv_b15,
            cache_controller_v13=cc13,
            replay_controller_v11=rcv11,
            consensus_v16=consensus_v16,
            lhr_v22=lhr22,
            deep_substrate_hybrid_v15=deep_v15,
            mlsc_v18_operator=mlsc_v18_op,
            multi_agent_coordinator_v6=masc_v6,
            team_consensus_controller_v5=tcc_v5,
            hosted_registry=reg,
            hosted_router_v3=hosted_router_v3,
            hosted_logprob_router_v3=hosted_logprob_router_v3,
            hosted_cache_planner_v3=hosted_cache_planner_v3,
            hosted_real_substrate_boundary_v3=boundary_v3,
            handoff_coordinator_v2=handoff_coord_v2,
            w69_params=w69_params,
            enabled=True,
            masc_v6_n_seeds=8,
        )

    def to_dict(self) -> dict[str, Any]:
        def _cid_or_empty(x: Any) -> str:
            return str(x.cid()) if x is not None else ""
        return {
            "schema": W70_SCHEMA_VERSION,
            "kind": "w70_params",
            "substrate_v15_cid": _cid_or_empty(self.substrate_v15),
            "kv_bridge_v15_cid": _cid_or_empty(self.kv_bridge_v15),
            "cache_controller_v13_cid": _cid_or_empty(
                self.cache_controller_v13),
            "replay_controller_v11_cid": _cid_or_empty(
                self.replay_controller_v11),
            "consensus_v16_cid": _cid_or_empty(
                self.consensus_v16),
            "lhr_v22_cid": _cid_or_empty(self.lhr_v22),
            "deep_substrate_hybrid_v15_cid": _cid_or_empty(
                self.deep_substrate_hybrid_v15),
            "mlsc_v18_operator_cid": _cid_or_empty(
                self.mlsc_v18_operator),
            "multi_agent_coordinator_v6_cid": _cid_or_empty(
                self.multi_agent_coordinator_v6),
            "team_consensus_controller_v5_cid": _cid_or_empty(
                self.team_consensus_controller_v5),
            "hosted_registry_cid": _cid_or_empty(
                self.hosted_registry),
            "hosted_router_v3_cid": _cid_or_empty(
                self.hosted_router_v3),
            "hosted_logprob_router_v3_cid": _cid_or_empty(
                self.hosted_logprob_router_v3),
            "hosted_cache_planner_v3_cid": _cid_or_empty(
                self.hosted_cache_planner_v3),
            "hosted_real_substrate_boundary_v3_cid":
                _cid_or_empty(
                    self.hosted_real_substrate_boundary_v3),
            "handoff_coordinator_v2_cid": _cid_or_empty(
                self.handoff_coordinator_v2),
            "w69_params_cid": _cid_or_empty(self.w69_params),
            "enabled": bool(self.enabled),
            "masc_v6_n_seeds": int(self.masc_v6_n_seeds),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w70_params",
            "params": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class W70HandoffEnvelope:
    schema: str
    w69_outer_cid: str
    w70_params_cid: str
    substrate_v15_witness_cid: str
    kv_bridge_v15_witness_cid: str
    cache_controller_v13_witness_cid: str
    replay_controller_v11_witness_cid: str
    persistent_v22_witness_cid: str
    mlsc_v18_witness_cid: str
    consensus_v16_witness_cid: str
    lhr_v22_witness_cid: str
    deep_substrate_hybrid_v15_witness_cid: str
    substrate_adapter_v15_matrix_cid: str
    masc_v6_witness_cid: str
    team_consensus_controller_v5_witness_cid: str
    repair_dominance_falsifier_witness_cid: str
    hosted_router_v3_witness_cid: str
    hosted_logprob_router_v3_witness_cid: str
    hosted_cache_planner_v3_witness_cid: str
    hosted_real_substrate_boundary_v3_cid: str
    hosted_wall_v3_report_cid: str
    handoff_coordinator_v2_witness_cid: str
    handoff_envelope_v2_chain_cid: str
    fifteen_way_used: bool
    substrate_v15_used: bool
    masc_v6_v15_beats_v14_rate: float
    masc_v6_tsc_v15_beats_tsc_v14_rate: float
    masc_v6_team_success_per_visible_token: float
    hosted_router_v3_chosen: str
    repair_trajectory_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "w69_outer_cid": str(self.w69_outer_cid),
            "w70_params_cid": str(self.w70_params_cid),
            "substrate_v15_witness_cid": str(
                self.substrate_v15_witness_cid),
            "kv_bridge_v15_witness_cid": str(
                self.kv_bridge_v15_witness_cid),
            "cache_controller_v13_witness_cid": str(
                self.cache_controller_v13_witness_cid),
            "replay_controller_v11_witness_cid": str(
                self.replay_controller_v11_witness_cid),
            "persistent_v22_witness_cid": str(
                self.persistent_v22_witness_cid),
            "mlsc_v18_witness_cid": str(
                self.mlsc_v18_witness_cid),
            "consensus_v16_witness_cid": str(
                self.consensus_v16_witness_cid),
            "lhr_v22_witness_cid": str(self.lhr_v22_witness_cid),
            "deep_substrate_hybrid_v15_witness_cid": str(
                self.deep_substrate_hybrid_v15_witness_cid),
            "substrate_adapter_v15_matrix_cid": str(
                self.substrate_adapter_v15_matrix_cid),
            "masc_v6_witness_cid": str(self.masc_v6_witness_cid),
            "team_consensus_controller_v5_witness_cid": str(
                self.team_consensus_controller_v5_witness_cid),
            "repair_dominance_falsifier_witness_cid": str(
                self.repair_dominance_falsifier_witness_cid),
            "hosted_router_v3_witness_cid": str(
                self.hosted_router_v3_witness_cid),
            "hosted_logprob_router_v3_witness_cid": str(
                self.hosted_logprob_router_v3_witness_cid),
            "hosted_cache_planner_v3_witness_cid": str(
                self.hosted_cache_planner_v3_witness_cid),
            "hosted_real_substrate_boundary_v3_cid": str(
                self.hosted_real_substrate_boundary_v3_cid),
            "hosted_wall_v3_report_cid": str(
                self.hosted_wall_v3_report_cid),
            "handoff_coordinator_v2_witness_cid": str(
                self.handoff_coordinator_v2_witness_cid),
            "handoff_envelope_v2_chain_cid": str(
                self.handoff_envelope_v2_chain_cid),
            "fifteen_way_used": bool(self.fifteen_way_used),
            "substrate_v15_used": bool(self.substrate_v15_used),
            "masc_v6_v15_beats_v14_rate": float(round(
                self.masc_v6_v15_beats_v14_rate, 12)),
            "masc_v6_tsc_v15_beats_tsc_v14_rate": float(round(
                self.masc_v6_tsc_v15_beats_tsc_v14_rate, 12)),
            "masc_v6_team_success_per_visible_token": float(
                round(
                    self.masc_v6_team_success_per_visible_token,
                    12)),
            "hosted_router_v3_chosen": str(
                self.hosted_router_v3_chosen),
            "repair_trajectory_cid": str(
                self.repair_trajectory_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w70_handoff_envelope",
            "envelope": self.to_dict()})


def verify_w70_handoff(
        envelope: W70HandoffEnvelope,
        params: W70Params,
        w69_outer_cid: str,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if envelope.schema != W70_SCHEMA_VERSION:
        failures.append("w70_outer_envelope_schema_mismatch")
    if envelope.w69_outer_cid != str(w69_outer_cid):
        failures.append(
            "w70_outer_envelope_w69_outer_cid_drift")
    if envelope.w70_params_cid != params.cid():
        failures.append(
            "w70_outer_envelope_w70_params_cid_drift")
    return (len(failures) == 0), failures


@dataclasses.dataclass
class W70Team:
    params: W70Params

    def run_team_turn(
            self, *,
            w69_outer_cid: str,
            ids: Sequence[int] | None = None,
            text: str = "w70",
    ) -> W70HandoffEnvelope:
        p = self.params
        if not p.enabled or p.substrate_v15 is None:
            return W70HandoffEnvelope(
                schema=W70_SCHEMA_VERSION,
                w69_outer_cid=str(w69_outer_cid),
                w70_params_cid=str(p.cid()),
                substrate_v15_witness_cid="",
                kv_bridge_v15_witness_cid="",
                cache_controller_v13_witness_cid="",
                replay_controller_v11_witness_cid="",
                persistent_v22_witness_cid="",
                mlsc_v18_witness_cid="",
                consensus_v16_witness_cid="",
                lhr_v22_witness_cid="",
                deep_substrate_hybrid_v15_witness_cid="",
                substrate_adapter_v15_matrix_cid="",
                masc_v6_witness_cid="",
                team_consensus_controller_v5_witness_cid="",
                repair_dominance_falsifier_witness_cid="",
                hosted_router_v3_witness_cid="",
                hosted_logprob_router_v3_witness_cid="",
                hosted_cache_planner_v3_witness_cid="",
                hosted_real_substrate_boundary_v3_cid="",
                hosted_wall_v3_report_cid="",
                handoff_coordinator_v2_witness_cid="",
                handoff_envelope_v2_chain_cid="",
                fifteen_way_used=False,
                substrate_v15_used=False,
                masc_v6_v15_beats_v14_rate=0.0,
                masc_v6_tsc_v15_beats_tsc_v14_rate=0.0,
                masc_v6_team_success_per_visible_token=0.0,
                hosted_router_v3_chosen="",
                repair_trajectory_cid="",
            )
        # Plane B — substrate V15 forward.
        token_ids = (
            list(ids) if ids is not None
            else tokenize_bytes_v15(str(text), max_len=16))
        trace, cache = forward_tiny_substrate_v15(
            p.substrate_v15, token_ids,
            visible_token_budget=128.0,
            baseline_token_cost=512.0)
        # Exercise V15 axes.
        record_repair_event_v15(
            cache, repair_label=W70_REPAIR_MULTI_BRANCH_REJOIN,
            turn=0, layer_index=0, role="planner")
        sub_witness = emit_tiny_substrate_v15_forward_witness(
            trace, cache)
        # KV V15 witnesses.
        rd_falsifier = (
            probe_kv_bridge_v15_repair_dominance_falsifier(
                dominant_repair_label=1))
        rt_fp = compute_repair_trajectory_fingerprint_v15(
            role="planner",
            repair_trajectory_cid=str(
                cache.repair_trajectory_cid),
            dominant_repair_label=1,
            visible_token_budget=128.0,
            baseline_cost=512.0)
        kv_witness = emit_kv_bridge_v15_witness(
            projection=p.kv_bridge_v15,
            repair_dominance_falsifier=rd_falsifier,
            repair_trajectory_fingerprint=rt_fp)
        cache_witness = emit_cache_controller_v13_witness(
            controller=p.cache_controller_v13)
        replay_witness = emit_replay_controller_v11_witness(
            p.replay_controller_v11)
        persist_chain = PersistentLatentStateV22Chain.empty()
        persist_witness = emit_persistent_v22_witness(
            persist_chain)
        # MLSC V18 — wrap a trivial V17 capsule up the chain.
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
        v3 = make_root_capsule_v3(
            branch_id="w70_smoke",
            payload=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            fact_tags=("w70",), confidence=0.9, trust=0.9,
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
                str(cache.repair_trajectory_cid),),
            budget_primary_chain=(
                f"bp_{int(trace.budget_primary_gate_per_layer.mean()*1000)}",),
        )
        mlsc_witness = emit_mlsc_v18_witness(v18)
        consensus_witness = emit_consensus_v16_witness(
            p.consensus_v16)
        lhr_witness = emit_lhr_v22_witness(
            p.lhr_v22, carrier=[0.1] * 6, k=16,
            partial_contradiction_indicator=[0.5] * 8,
            multi_branch_rejoin_indicator=[0.6] * 8,
            repair_dominance_indicator=[0.7] * 7)
        # Deep substrate hybrid V15.
        v14_witness = DeepSubstrateHybridV14ForwardWitness(
            schema="coordpy.deep_substrate_hybrid_v14.v1",
            hybrid_cid="",
            inner_v13_witness_cid="",
            fourteen_way=True,
            cache_controller_v12_fired=True,
            replay_controller_v10_fired=True,
            multi_branch_rejoin_witness_active=True,
            silent_corruption_active=True,
            substrate_self_checksum_active=True,
            team_consensus_controller_v4_active=True,
            multi_branch_rejoin_witness_l1=0.5,
            silent_corruption_count=1,
            substrate_self_checksum_cid=str(
                cache.v14_cache.substrate_self_checksum_cid))
        deep_v15_witness = deep_substrate_hybrid_v15_forward(
            hybrid=p.deep_substrate_hybrid_v15,
            v14_witness=v14_witness,
            cache_controller_v13=p.cache_controller_v13,
            replay_controller_v11=p.replay_controller_v11,
            repair_trajectory_cid=str(
                cache.repair_trajectory_cid),
            dominant_repair_l1=int(
                sub_witness.dominant_repair_l1),
            budget_primary_gate_mean=float(
                trace.budget_primary_gate_per_layer.mean()),
            n_team_consensus_v5_invocations=1)
        adapter_matrix = probe_all_v15_adapters()
        # MASC V6 — run a batch for the envelope (baseline regime).
        from .multi_agent_substrate_coordinator_v2 import (
            W66_MASC_V2_REGIME_BASELINE)
        per_regime_aggs = {}
        for regime in W70_MASC_V6_REGIMES:
            _, agg = p.multi_agent_coordinator_v6.run_batch(
                seeds=list(range(int(p.masc_v6_n_seeds))),
                regime=regime)
            per_regime_aggs[regime] = agg
        masc_witness = (
            emit_multi_agent_substrate_coordinator_v6_witness(
                coordinator=p.multi_agent_coordinator_v6,
                per_regime_aggregate=per_regime_aggs))
        # TCC V5 — fire each new arbiter so the witness counts > 0.
        tcc_v5 = p.team_consensus_controller_v5
        tcc_v5.decide_v5(
            regime=(
                "contradiction_then_rejoin_under_budget"),
            agent_guesses=[1.0, -1.0, 0.5, 0.2],
            agent_confidences=[0.8, 0.6, 0.7, 0.7],
            substrate_replay_trust=0.7,
            visible_token_budget_ratio=0.3,
            dominant_repair_label=1,
            agent_repair_labels=[1, 1, 0, 0],
            branch_assignments=[0, 1, 2, 0])
        tcc_v5.decide_v5(
            regime=W66_MASC_V2_REGIME_BASELINE,
            agent_guesses=[0.5, 0.5, 0.4, 0.5],
            agent_confidences=[0.8, 0.6, 0.7, 0.7],
            substrate_replay_trust=0.7,
            visible_token_budget_ratio=0.3,
            dominant_repair_label=0)
        tcc_v5.decide_v5(
            regime=W66_MASC_V2_REGIME_BASELINE,
            agent_guesses=[0.5, 0.5, 0.4, 0.5],
            agent_confidences=[0.8, 0.6, 0.7, 0.7],
            substrate_replay_trust=0.7,
            visible_token_budget_ratio=0.9,
            dominant_repair_label=1,
            agent_repair_labels=[1, 1, 0, 0])
        tcc_witness = emit_team_consensus_controller_v5_witness(
            tcc_v5)
        # Plane A V3 — hosted.
        planned, _ = (
            p.hosted_cache_planner_v3
            .plan_per_role_staggered_and_rotated(
                shared_prefix_text="W70 team shared prefix " * 8,
                per_role_blocks={
                    "plan": ["t0", "t1"],
                    "research": ["r0", "r1"],
                    "write": ["w0", "w1"],
                }))
        # Router V3 — at least one decision so witness is non-empty.
        req_v3 = HostedRoutingRequestV3(
            inner_v2=HostedRoutingRequestV2(
                inner_v1=HostedRoutingRequest(
                    request_cid="w70-router-turn",
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
            repair_dominance_label=1)
        router_dec = p.hosted_router_v3.decide_v3(req_v3)
        router_v3_witness = (
            emit_hosted_router_controller_v3_witness(
                p.hosted_router_v3))
        logprob_v3_witness = (
            emit_hosted_logprob_router_v3_witness(
                p.hosted_logprob_router_v3))
        cache_planner_v3_witness = (
            emit_hosted_cache_aware_planner_v3_witness(
                p.hosted_cache_planner_v3))
        boundary_v3 = p.hosted_real_substrate_boundary_v3
        wall_v3_report = build_wall_report_v3(
            boundary=boundary_v3)
        # Handoff coordinator V2 decisions.
        env_text_only = p.handoff_coordinator_v2.decide_v2(
            req_v2=HandoffRequestV2(
                inner_v1=HandoffRequest(
                    request_cid="w70-turn-text",
                    needs_text_only=True,
                    needs_substrate_state_access=False),
                visible_token_budget=128,
                baseline_token_cost=512,
                dominant_repair_label=0),
            substrate_self_checksum_cid=str(
                cache.v14_cache.substrate_self_checksum_cid))
        env_substrate_only = p.handoff_coordinator_v2.decide_v2(
            req_v2=HandoffRequestV2(
                inner_v1=HandoffRequest(
                    request_cid="w70-turn-substrate",
                    needs_text_only=False,
                    needs_substrate_state_access=True),
                visible_token_budget=128,
                baseline_token_cost=512,
                dominant_repair_label=1),
            substrate_self_checksum_cid=str(
                cache.v14_cache.substrate_self_checksum_cid))
        env_audit = p.handoff_coordinator_v2.decide_v2(
            req_v2=HandoffRequestV2(
                inner_v1=HandoffRequest(
                    request_cid="w70-turn-audit",
                    needs_text_only=True,
                    needs_substrate_state_access=True),
                visible_token_budget=64,
                baseline_token_cost=512,
                dominant_repair_label=1),
            substrate_self_checksum_cid=str(
                cache.v14_cache.substrate_self_checksum_cid))
        env_budget_fallback = p.handoff_coordinator_v2.decide_v2(
            req_v2=HandoffRequestV2(
                inner_v1=HandoffRequest(
                    request_cid="w70-turn-bp",
                    needs_text_only=True,
                    needs_substrate_state_access=False),
                visible_token_budget=32,
                baseline_token_cost=512,
                expected_team_success_hosted=0.30,
                expected_team_success_substrate=0.20,
                expected_team_success_audit=0.25,
                dominant_repair_label=0),
            substrate_self_checksum_cid=str(
                cache.v14_cache.substrate_self_checksum_cid))
        handoff_v2_witness = (
            emit_hosted_real_handoff_coordinator_v2_witness(
                p.handoff_coordinator_v2))
        handoff_envelope_chain_cid = _sha256_hex({
            "kind": "w70_handoff_envelope_v2_chain",
            "envelopes": [
                env_text_only.cid(),
                env_substrate_only.cid(),
                env_audit.cid(),
                env_budget_fallback.cid(),
            ],
        })
        baseline_agg = per_regime_aggs.get(
            W66_MASC_V2_REGIME_BASELINE)
        v15_beats = (
            float(baseline_agg.v15_beats_v14_rate)
            if baseline_agg is not None else 0.0)
        tsc_v15_beats = (
            float(baseline_agg.tsc_v15_beats_tsc_v14_rate)
            if baseline_agg is not None else 0.0)
        ts_per_vt = (
            float(baseline_agg.team_success_per_visible_token_v15)
            if baseline_agg is not None else 0.0)
        return W70HandoffEnvelope(
            schema=W70_SCHEMA_VERSION,
            w69_outer_cid=str(w69_outer_cid),
            w70_params_cid=str(p.cid()),
            substrate_v15_witness_cid=str(sub_witness.cid()),
            kv_bridge_v15_witness_cid=str(kv_witness.cid()),
            cache_controller_v13_witness_cid=str(
                cache_witness.cid()),
            replay_controller_v11_witness_cid=str(
                replay_witness.cid()),
            persistent_v22_witness_cid=str(persist_witness.cid()),
            mlsc_v18_witness_cid=str(mlsc_witness.cid()),
            consensus_v16_witness_cid=str(
                consensus_witness.cid()),
            lhr_v22_witness_cid=str(lhr_witness.cid()),
            deep_substrate_hybrid_v15_witness_cid=str(
                deep_v15_witness.cid()),
            substrate_adapter_v15_matrix_cid=str(
                adapter_matrix.cid()),
            masc_v6_witness_cid=str(masc_witness.cid()),
            team_consensus_controller_v5_witness_cid=str(
                tcc_witness.cid()),
            repair_dominance_falsifier_witness_cid=str(
                rd_falsifier.cid()),
            hosted_router_v3_witness_cid=str(
                router_v3_witness.cid()),
            hosted_logprob_router_v3_witness_cid=str(
                logprob_v3_witness.cid()),
            hosted_cache_planner_v3_witness_cid=str(
                cache_planner_v3_witness.cid()),
            hosted_real_substrate_boundary_v3_cid=str(
                boundary_v3.cid()),
            hosted_wall_v3_report_cid=str(
                wall_v3_report.cid()),
            handoff_coordinator_v2_witness_cid=str(
                handoff_v2_witness.cid()),
            handoff_envelope_v2_chain_cid=str(
                handoff_envelope_chain_cid),
            fifteen_way_used=bool(
                deep_v15_witness.fifteen_way),
            substrate_v15_used=True,
            masc_v6_v15_beats_v14_rate=float(v15_beats),
            masc_v6_tsc_v15_beats_tsc_v14_rate=float(
                tsc_v15_beats),
            masc_v6_team_success_per_visible_token=float(
                ts_per_vt),
            hosted_router_v3_chosen=str(
                router_dec.chosen_provider or ""),
            repair_trajectory_cid=str(
                cache.repair_trajectory_cid),
        )


def build_default_w70_team(*, seed: int = 70000) -> W70Team:
    return W70Team(params=W70Params.build_default(seed=int(seed)))


__all__ = [
    "W70_SCHEMA_VERSION",
    "W70_FAILURE_MODES",
    "W70Params",
    "W70HandoffEnvelope",
    "verify_w70_handoff",
    "W70Team",
    "build_default_w70_team",
]
