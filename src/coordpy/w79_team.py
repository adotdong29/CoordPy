"""W79 — Stronger Direct-Blocker-Attack / Replacement-Then-Restart-
After-Long-Delay Budget-Primary Two-Plane Multi-Agent Substrate
team.

The ``W79Team`` orchestrator strictly wraps the ``W78Team`` and
adds the W79 mechanism modules across two planes plus the new
controlled-runtime substrate, the local OpenAI-compatible
façade, and the learned-consolidation head — the W79 direct-
blocker-attack pillars.

**Plane B — Real substrate (in-repo, V24 stack):**

* M1  ``tiny_substrate_v24``           (V23 + 1 new W79 axis)
* M2  ``kv_bridge_v24``                (20-target + 168-dim
                                        replacement-then-restart-
                                        after-long-delay
                                        fingerprint + falsifier)
* M3  ``cache_controller_v22``         (19-objective ridge)
* M4  ``replay_controller_v20``        (27 regimes + 17-label
                                        routing head)
* M5  ``deep_substrate_hybrid_v24``    (24-way bidirectional)
* M6  ``substrate_adapter_v24``        (v24_full + controlled_
                                        runtime_v1 tiers)
* M7  ``persistent_latent_v31``        (30 layers, 28th carrier,
                                        max_chain_walk_depth=
                                        67108864)
* M8  ``long_horizon_retention_v31``   (30 heads, max_k=4096)
* M9  ``mergeable_latent_capsule_v27`` (replacement-then-restart-
                                        after-long-delay chain)
* M10 ``consensus_fallback_controller_v25`` (44-stage chain)
* M11 ``multi_agent_substrate_coordinator_v15`` (32-policy 19-
                                                 regime MASC V15)
* M12 ``team_consensus_controller_v14`` (RTRLD pressure +
                                         trajectory arbiters)
* M13 ``long_horizon_reconstruction_substrate_v2`` (learned-
                                                    consolidation
                                                    slots +
                                                    replay-vs-
                                                    recompute
                                                    arbiter)
* M14 ``bounded_window_baseline_v2`` (k64 + cross-prompt summary
                                      falsifier)

**Plane A — Hosted control plane V12 (honest, no substrate):**

* H1  ``hosted_router_controller_v12``  (RTRLD pressure
                                         weighting + RTRLD match)
* H2  ``hosted_logprob_router_v12``     (RTRLD-aware abstain
                                         floor + 10-way tiebreak)
* H3  ``hosted_cache_aware_planner_v12``(ten-layer rotated
                                         prefix; ≥ 90 % savings
                                         20×8 hit=1)
* H4  ``hosted_cost_planner_v12``       (cost-per-RTRLD-success-
                                         under-budget + abstain-
                                         when-RTRLD-violated)
* H5  ``hosted_real_substrate_boundary_v12`` (wall V12: 56+
                                              blocked axes)
* H6  ``hosted_real_handoff_coordinator_v11`` (RTRLD + controlled-
                                               runtime bridge)
* H7  ``hosted_provider_filter_v11``    (RTRLD-aware filter)

**W79 direct-blocker-attack pillars:**

* D1  ``controlled_runtime_substrate_v1`` — a SECOND substrate
       runtime that we control end-to-end. Exposes hidden state /
       KV / attention / replay-from-KV honestly. Byte-stable
       under recompute.
* D2  ``local_openai_compatible_facade_v1`` — an OpenAI-shaped
       façade pointing to D1. Hosted-shaped requests get a
       substrate side channel.
* D3  ``learned_consolidation_v1`` — autograd-style learned
       consolidation head. Beats closed-form ridge on the W79
       nonlinear consolidation dataset.

Honest scope (W79)
------------------

* Plane A V12 operates at the hosted text/logprob/prefix-cache
  surface. It does NOT pierce hidden state / KV / attention.
  ``W79-L-HOSTED-V12-NO-SUBSTRATE-CAP``.
* Plane B is the in-repo V24 NumPy runtime. We do NOT bridge to
  third-party hosted models.
  ``W79-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
* The W79 direct-blocker-attack pillars D1+D2 are an in-process
  controlled substrate runtime + façade. They DO expose
  substrate axes. They do NOT pierce third-party hosted-model
  substrate.
  ``W79-L-CONTROLLED-RUNTIME-IN-REPO-CAP``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .bounded_window_baseline_v1 import BoundedWindowQuery
from .bounded_window_baseline_v2 import (
    build_default_bounded_window_baselines_v2,
    emit_bounded_window_baseline_witness_v2,
    prove_bounded_window_insufficient_v2,
    run_bounded_window_falsifier_v2,
)
from .cache_controller_v22 import (
    CacheControllerV22, emit_cache_controller_v22_witness,
)
from .consensus_fallback_controller_v25 import (
    ConsensusFallbackControllerV25,
    emit_consensus_v25_witness,
)
from .controlled_runtime_substrate_v1 import (
    ControlledRuntimeParamsV1,
    build_controlled_runtime_params_v1,
    emit_controlled_runtime_witness,
    forward_controlled_runtime,
    measure_controlled_runtime_replay_vs_recompute,
    tokenize_bytes_v79,
)
from .deep_substrate_hybrid_v23 import (
    DeepSubstrateHybridV23ForwardWitness,
)
from .deep_substrate_hybrid_v24 import (
    DeepSubstrateHybridV24, deep_substrate_hybrid_v24_forward,
)
from .hosted_cache_aware_planner_v12 import (
    HostedCacheAwarePlannerV12,
    emit_hosted_cache_aware_planner_v12_witness,
)
from .hosted_cost_planner_v12 import (
    HostedCostPlannerV12, emit_hosted_cost_planner_v12_witness,
)
from .hosted_logprob_router_v12 import (
    HostedLogprobRouterV12,
    emit_hosted_logprob_router_v12_witness,
)
from .hosted_provider_filter_v11 import (
    HostedProviderFilterSpecV11, filter_hosted_registry_v11,
)
from .hosted_real_handoff_coordinator_v11 import (
    HandoffEnvelopeV11, HandoffRequestV11,
    HostedRealHandoffCoordinatorV11,
    emit_hosted_real_handoff_coordinator_v11_witness,
    hosted_real_handoff_v11_replacement_then_restart_after_long_delay_aware_savings,
    probe_hosted_real_handoff_v11_replacement_then_restart_after_long_delay_falsifier,
)
from .hosted_real_substrate_boundary_v12 import (
    HostedRealSubstrateBoundaryV12,
    build_default_hosted_real_substrate_boundary_v12,
    build_wall_report_v12,
)
from .hosted_router_controller import (
    HostedProviderRegistry, default_hosted_registry,
)
from .hosted_router_controller_v12 import (
    HostedRouterControllerV12, HostedRoutingRequestV12,
    emit_hosted_router_controller_v12_witness,
)
from .kv_bridge_v24 import (
    KVBridgeV24Projection,
    compute_replacement_then_restart_after_long_delay_fingerprint_v24,
    emit_kv_bridge_v24_witness,
    probe_kv_bridge_v24_replacement_then_restart_after_long_delay_falsifier,
)
from .learned_consolidation_v1 import (
    LearnedConsolidationHeadV1,
    build_learned_consolidation_head_v1,
    build_nonlinear_consolidation_dataset,
    compare_learned_vs_closed_form,
    emit_learned_consolidation_witness,
    train_learned_consolidation_head,
)
from .local_openai_compatible_facade_v1 import (
    LocalOpenAIChatCompletionRequestV1, LocalOpenAIChatMessageV1,
    LocalOpenAICompatibleFacadeV1,
    compare_facade_vs_hosted_substrate,
    emit_local_openai_facade_witness,
)
from .long_horizon_reconstruction_substrate_v1 import (
    LongHorizonReconstructionQuery,
    report_reconstruction_vs_recompute_economics,
)
from .long_horizon_reconstruction_substrate_v2 import (
    LongHorizonReconstructionCarrierV2,
    arbitrate_replay_vs_recompute_v2,
    build_default_long_horizon_reconstruction_carrier_v2,
    emit_long_horizon_reconstruction_witness_v2,
    reconstruct_long_horizon_event_v2,
)
from .long_horizon_retention_v31 import (
    LongHorizonReconstructionV31Head, emit_lhr_v31_witness,
)
from .mergeable_latent_capsule_v27 import (
    MergeOperatorV27, emit_mlsc_v27_witness, wrap_v26_as_v27,
)
from .multi_agent_substrate_coordinator_v15 import (
    MultiAgentSubstrateCoordinatorV15, W79_MASC_V15_REGIMES,
    emit_multi_agent_substrate_coordinator_v15_witness,
)
from .persistent_latent_v31 import (
    PersistentLatentStateV31Chain, emit_persistent_v31_witness,
)
from .replay_controller_v20 import (
    ReplayControllerV20, W79_REPLAY_REGIMES_V20,
    emit_replay_controller_v20_witness,
)
from .substrate_adapter_v24 import probe_all_v24_adapters
from .team_consensus_controller_v14 import (
    TeamConsensusControllerV14,
    emit_team_consensus_controller_v14_witness,
)
from .tiny_substrate_v24 import (
    TinyV24SubstrateParams, build_default_tiny_substrate_v24,
    emit_tiny_substrate_v24_forward_witness,
    forward_tiny_substrate_v24,
    record_replacement_then_restart_after_long_delay_window_v24,
    tokenize_bytes_v24,
)
from .w78_team import W78HandoffEnvelope, W78Params, W78Team


W79_SCHEMA_VERSION: str = "coordpy.w79_team.v1"

W79_FAILURE_MODES: tuple[str, ...] = (
    "w79_outer_envelope_schema_mismatch",
    "w79_outer_envelope_w78_outer_cid_drift",
    "w79_outer_envelope_w79_params_cid_drift",
    "w79_outer_envelope_witness_cid_drift",
    "w79_substrate_v24_n_layers_off",
    "w79_substrate_v24_rtrld_trajectory_cid_off",
    "w79_kv_bridge_v24_n_targets_off",
    "w79_kv_bridge_v24_rtrld_falsifier_off",
    "w79_cache_v22_nineteen_objective_off",
    "w79_replay_v20_regime_count_off",
    "w79_consensus_v25_stage_count_off",
    "w79_lhr_v31_max_k_off",
    "w79_lhr_v31_n_heads_off",
    "w79_persistent_v31_n_layers_off",
    "w79_substrate_adapter_v24_tier_off",
    "w79_substrate_adapter_v24_controlled_runtime_missing",
    "w79_masc_v15_v24_beats_v23_rate_under_threshold",
    "w79_masc_v15_tsc_v24_beats_tsc_v23_rate_under_threshold",
    "w79_masc_v15_w79_regime_inferior_to_baseline",
    "w79_hosted_router_v12_decision_not_deterministic",
    "w79_hosted_logprob_v12_abstain_floor_off",
    "w79_hosted_cache_aware_v12_savings_below_90_percent",
    "w79_hosted_cost_planner_v12_no_eligible",
    "w79_hosted_real_substrate_boundary_v12_blocked_axis_satisfied",
    "w79_handoff_coordinator_v11_inconsistent",
    "w79_handoff_v11_cross_plane_savings_below_88_percent",
    "w79_handoff_v11_controlled_runtime_decision_off",
    "w79_team_consensus_v14_no_decisions",
    "w79_handoff_v11_rtrld_alignment_off",
    "w79_handoff_envelope_v11_chain_cid_drift",
    "w79_inner_v78_envelope_invariant_off",
    "w79_hosted_boundary_v12_blocked_axes_below_56",
    "w79_v24_substrate_self_checksum_cid_off",
    "w79_rtrld_trajectory_cid_drift",
    "w79_mlsc_v27_rtrld_trajectory_chain_off",
    "w79_v15_team_success_per_visible_token_below_floor",
    "w79_v15_visible_tokens_savings_below_70_percent",
    "w79_v15_w79_regime_v24_beats_v23_below_threshold",
    "w79_substrate_v24_rtrld_trajectory_chain_synthetic",
    "w79_inner_v24_falsifier_kind_off",
    "w79_handoff_v11_envelope_chain_alignment_off",
    "w79_hosted_router_v12_per_routing_cid_off",
    "w79_consensus_v25_rtrld_arbiter_off",
    "w79_consensus_v25_rtrld_best_parent_arbiter_off",
    "w79_tcc_v14_rtrld_pressure_arbiter_off",
    "w79_tcc_v14_rtrld_trajectory_arbiter_off",
    "w79_cache_v22_per_role_rtrld_pressure_head_off",
    "w79_kv_bridge_v24_rtrld_fingerprint_off",
    "w79_substrate_v24_rtrld_windows_off",
    "w79_provider_filter_v11_pressure_drop_off",
    "w79_handoff_v11_alignment_off",
    "w79_handoff_v11_decision_label_off",
    "w79_bounded_window_baseline_v2_k64_not_failing",
    "w79_bounded_window_baseline_v2_cross_prompt_not_failing",
    "w79_long_horizon_reconstruction_v2_substrate_not_reconstructing",
    "w79_bounded_window_insufficiency_v2_not_proven",
    "w79_controlled_runtime_replay_not_byte_identical",
    "w79_controlled_runtime_no_kv_cache_emitted",
    "w79_facade_substrate_side_channel_missing_when_requested",
    "w79_learned_consolidation_post_loss_not_lower_than_pre",
    "w79_learned_consolidation_not_strictly_beating_ridge",
    "w79_facade_vs_hosted_substrate_report_inconsistent",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class W79Params:
    substrate_v24: TinyV24SubstrateParams | None
    kv_bridge_v24: KVBridgeV24Projection | None
    cache_controller_v22: CacheControllerV22 | None
    replay_controller_v20: ReplayControllerV20 | None
    consensus_v25: ConsensusFallbackControllerV25 | None
    lhr_v31: LongHorizonReconstructionV31Head | None
    deep_substrate_hybrid_v24: DeepSubstrateHybridV24 | None
    multi_agent_coordinator_v15: (
        MultiAgentSubstrateCoordinatorV15 | None)
    team_consensus_controller_v14: (
        TeamConsensusControllerV14 | None)
    hosted_registry: HostedProviderRegistry | None
    hosted_router_v12: HostedRouterControllerV12 | None
    hosted_logprob_router_v12: HostedLogprobRouterV12 | None
    hosted_cache_planner_v12: HostedCacheAwarePlannerV12 | None
    hosted_cost_planner_v12: HostedCostPlannerV12 | None
    hosted_real_substrate_boundary_v12: (
        HostedRealSubstrateBoundaryV12 | None)
    handoff_coordinator_v11: (
        HostedRealHandoffCoordinatorV11 | None)
    hosted_provider_filter_v11: (
        HostedProviderFilterSpecV11 | None)
    controlled_runtime_params_v1: (
        ControlledRuntimeParamsV1 | None)
    local_openai_facade_v1: (
        LocalOpenAICompatibleFacadeV1 | None)
    learned_consolidation_head_v1: (
        LearnedConsolidationHeadV1 | None)
    w78_params: W78Params | None
    enabled: bool = True
    masc_v15_n_seeds: int = 3

    @classmethod
    def build_trivial(cls) -> "W79Params":
        return cls(
            substrate_v24=None, kv_bridge_v24=None,
            cache_controller_v22=None,
            replay_controller_v20=None,
            consensus_v25=None, lhr_v31=None,
            deep_substrate_hybrid_v24=None,
            multi_agent_coordinator_v15=None,
            team_consensus_controller_v14=None,
            hosted_registry=None,
            hosted_router_v12=None,
            hosted_logprob_router_v12=None,
            hosted_cache_planner_v12=None,
            hosted_cost_planner_v12=None,
            hosted_real_substrate_boundary_v12=None,
            handoff_coordinator_v11=None,
            hosted_provider_filter_v11=None,
            controlled_runtime_params_v1=None,
            local_openai_facade_v1=None,
            learned_consolidation_head_v1=None,
            w78_params=None,
            enabled=False,
        )

    @classmethod
    def build_default(
            cls, *, seed: int = 79000,
    ) -> "W79Params":
        from .w78_team import W78Params as _W78P
        w78_p = _W78P.build_default(seed=int(seed) - 1000)
        sub_v24 = build_default_tiny_substrate_v24(
            seed=int(seed) + 1)
        kv_b24 = KVBridgeV24Projection.init_from_v23(
            w78_p.kv_bridge_v23, seed_v24=int(seed) + 27)
        cc22 = CacheControllerV22.init(
            fit_seed=int(seed) + 34)
        rcv20 = ReplayControllerV20.init()
        consensus_v25 = ConsensusFallbackControllerV25.init(
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
            long_horizon_reconstruction_best_parent_threshold=0.5,
            replacement_then_restart_after_long_delay_threshold=0.5,
            replacement_then_restart_after_long_delay_best_parent_threshold=0.5)
        lhr31 = LongHorizonReconstructionV31Head.init(
            seed=int(seed) + 42)
        deep_v24 = DeepSubstrateHybridV24()
        masc_v15 = MultiAgentSubstrateCoordinatorV15()
        tcc_v14 = TeamConsensusControllerV14()
        reg = default_hosted_registry()
        hosted_router_v12 = HostedRouterControllerV12.init(
            reg, {
                "openrouter_paid": 0.88,
                "openai_paid": 0.95,
            })
        hosted_logprob_router_v12 = HostedLogprobRouterV12()
        hosted_cache_planner_v12 = HostedCacheAwarePlannerV12()
        hosted_cost_planner_v12 = HostedCostPlannerV12()
        boundary_v12 = (
            build_default_hosted_real_substrate_boundary_v12())
        handoff_coord_v11 = HostedRealHandoffCoordinatorV11(
            boundary_v12=boundary_v12)
        provider_filter_v11 = HostedProviderFilterSpecV11(
            inner_v10=w78_p.hosted_provider_filter_v10,
            replacement_then_restart_after_long_delay_pressure=0.85,
            replacement_then_restart_after_long_delay_pressure_floor=0.5,
            max_replacement_then_restart_after_long_delay_noise_per_provider={
                "openrouter_paid": 0.06,
                "openai_paid": 1.0},
            replacement_then_restart_after_long_delay_tier_weights={
                "logprobs_and_prefix_cache": 1.0,
                "logprobs": 0.40,
                "prefix_cache": 0.30,
                "text_only": 0.10})
        ctrl_runtime_p = build_controlled_runtime_params_v1(
            seed=int(seed) + 79)
        facade = LocalOpenAICompatibleFacadeV1(
            runtime_params=ctrl_runtime_p)
        lc_head = build_learned_consolidation_head_v1(
            seed=int(seed) + 13)
        return cls(
            substrate_v24=sub_v24,
            kv_bridge_v24=kv_b24,
            cache_controller_v22=cc22,
            replay_controller_v20=rcv20,
            consensus_v25=consensus_v25,
            lhr_v31=lhr31,
            deep_substrate_hybrid_v24=deep_v24,
            multi_agent_coordinator_v15=masc_v15,
            team_consensus_controller_v14=tcc_v14,
            hosted_registry=reg,
            hosted_router_v12=hosted_router_v12,
            hosted_logprob_router_v12=hosted_logprob_router_v12,
            hosted_cache_planner_v12=hosted_cache_planner_v12,
            hosted_cost_planner_v12=hosted_cost_planner_v12,
            hosted_real_substrate_boundary_v12=boundary_v12,
            handoff_coordinator_v11=handoff_coord_v11,
            hosted_provider_filter_v11=provider_filter_v11,
            controlled_runtime_params_v1=ctrl_runtime_p,
            local_openai_facade_v1=facade,
            learned_consolidation_head_v1=lc_head,
            w78_params=w78_p,
            enabled=True,
            masc_v15_n_seeds=3,
        )

    def to_dict(self) -> dict[str, Any]:
        def _cid_or_empty(x: Any) -> str:
            return str(x.cid()) if x is not None else ""
        return {
            "schema": W79_SCHEMA_VERSION,
            "kind": "w79_params",
            "substrate_v24_cid": _cid_or_empty(
                self.substrate_v24),
            "kv_bridge_v24_cid": _cid_or_empty(
                self.kv_bridge_v24),
            "cache_controller_v22_cid": _cid_or_empty(
                self.cache_controller_v22),
            "replay_controller_v20_cid": _cid_or_empty(
                self.replay_controller_v20),
            "consensus_v25_cid": _cid_or_empty(
                self.consensus_v25),
            "lhr_v31_cid": _cid_or_empty(self.lhr_v31),
            "deep_substrate_hybrid_v24_cid": _cid_or_empty(
                self.deep_substrate_hybrid_v24),
            "multi_agent_coordinator_v15_cid": _cid_or_empty(
                self.multi_agent_coordinator_v15),
            "team_consensus_controller_v14_cid": _cid_or_empty(
                self.team_consensus_controller_v14),
            "hosted_registry_cid": _cid_or_empty(
                self.hosted_registry),
            "hosted_router_v12_cid": _cid_or_empty(
                self.hosted_router_v12),
            "hosted_logprob_router_v12_cid": _cid_or_empty(
                self.hosted_logprob_router_v12),
            "hosted_cache_planner_v12_cid": _cid_or_empty(
                self.hosted_cache_planner_v12),
            "hosted_cost_planner_v12_cid": _cid_or_empty(
                self.hosted_cost_planner_v12),
            "hosted_real_substrate_boundary_v12_cid":
                _cid_or_empty(
                    self.hosted_real_substrate_boundary_v12),
            "handoff_coordinator_v11_cid": _cid_or_empty(
                self.handoff_coordinator_v11),
            "hosted_provider_filter_v11_cid": _cid_or_empty(
                self.hosted_provider_filter_v11),
            "controlled_runtime_params_v1_cid": _cid_or_empty(
                self.controlled_runtime_params_v1),
            "local_openai_facade_v1_cid": _cid_or_empty(
                self.local_openai_facade_v1),
            "learned_consolidation_head_v1_cid": _cid_or_empty(
                self.learned_consolidation_head_v1),
            "w78_params_cid": _cid_or_empty(self.w78_params),
            "enabled": bool(self.enabled),
            "masc_v15_n_seeds": int(self.masc_v15_n_seeds),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_params",
            "params": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class W79HandoffEnvelope:
    schema: str
    w78_outer_cid: str
    w79_params_cid: str
    substrate_v24_witness_cid: str
    kv_bridge_v24_witness_cid: str
    cache_controller_v22_witness_cid: str
    replay_controller_v20_witness_cid: str
    persistent_v31_witness_cid: str
    mlsc_v27_witness_cid: str
    consensus_v25_witness_cid: str
    lhr_v31_witness_cid: str
    deep_substrate_hybrid_v24_witness_cid: str
    substrate_adapter_v24_matrix_cid: str
    masc_v15_witness_cid: str
    team_consensus_controller_v14_witness_cid: str
    replacement_then_restart_after_long_delay_pressure_falsifier_witness_cid: (
        str)
    hosted_router_v12_witness_cid: str
    hosted_logprob_router_v12_witness_cid: str
    hosted_cache_planner_v12_witness_cid: str
    hosted_cost_planner_v12_witness_cid: str
    hosted_real_substrate_boundary_v12_cid: str
    hosted_wall_v12_report_cid: str
    handoff_coordinator_v11_witness_cid: str
    handoff_envelope_v11_chain_cid: str
    provider_filter_v11_report_cid: str
    long_horizon_reconstruction_v2_witness_cid: str
    bounded_window_baseline_v2_witness_cid: str
    bounded_window_insufficiency_proof_v2_cid: str
    controlled_runtime_witness_cid: str
    controlled_runtime_replay_vs_recompute_report_cid: str
    local_openai_facade_witness_cid: str
    local_openai_facade_response_cid: str
    learned_consolidation_witness_cid: str
    twenty_four_way_used: bool
    substrate_v24_used: bool
    controlled_runtime_used: bool
    masc_v15_v24_beats_v23_rate: float
    masc_v15_tsc_v24_beats_tsc_v23_rate: float
    masc_v15_team_success_per_visible_token: float
    hosted_router_v12_chosen: str
    replacement_then_restart_after_long_delay_trajectory_cid: str
    long_horizon_reconstruction_v2_success_rate: float
    bounded_window_v2_all_fixed_k_failed: bool
    learned_consolidation_beats_ridge: bool
    facade_substrate_side_channel_emitted: bool
    controlled_runtime_replay_byte_identical: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_handoff_envelope",
            "envelope": self.to_dict()})


def verify_w79_handoff(
        envelope: W79HandoffEnvelope,
        params: W79Params,
        w78_outer_cid: str,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if envelope.schema != W79_SCHEMA_VERSION:
        failures.append(
            "w79_outer_envelope_schema_mismatch")
    if envelope.w78_outer_cid != str(w78_outer_cid):
        failures.append(
            "w79_outer_envelope_w78_outer_cid_drift")
    if envelope.w79_params_cid != params.cid():
        failures.append(
            "w79_outer_envelope_w79_params_cid_drift")
    return (len(failures) == 0), failures


@dataclasses.dataclass
class W79Team:
    params: W79Params

    def run_team_turn(
            self, *,
            w78_outer_cid: str,
            ids: Sequence[int] | None = None,
            text: str = "w79",
    ) -> W79HandoffEnvelope:
        p = self.params
        if not p.enabled or p.substrate_v24 is None:
            return _trivial_envelope(p, w78_outer_cid)
        # ---- Plane B forward.
        token_ids = (
            list(ids) if ids is not None
            else tokenize_bytes_v24(str(text), max_len=16))
        trace_v24, cache_v24 = forward_tiny_substrate_v24(
            p.substrate_v24, token_ids,
            replacement_then_restart_after_long_delay_pressure=0.95,
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
        record_replacement_then_restart_after_long_delay_window_v24(
            cache_v24,
            long_delay_blackout_start_turn=30,
            replacement_turn=45,
            restart_turn=70,
            reconstruction_request_turn=95,
            role="planner_w79", branch_id="main")
        trace_v24, cache_v24 = forward_tiny_substrate_v24(
            p.substrate_v24, token_ids,
            v24_kv_cache=cache_v24,
            replacement_then_restart_after_long_delay_pressure=0.95,
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
        sub_witness = (
            emit_tiny_substrate_v24_forward_witness(
                p.substrate_v24, trace_v24, cache_v24))
        rtrld_falsifier = (
            probe_kv_bridge_v24_replacement_then_restart_after_long_delay_falsifier(
                replacement_then_restart_after_long_delay_pressure_flag=1))
        rtrld_fp = (
            compute_replacement_then_restart_after_long_delay_fingerprint_v24(
                role="planner",
                long_horizon_reconstruction_trajectory_cid=str(
                    cache_v24.v23_cache.long_horizon_reconstruction_trajectory_cid
                    if cache_v24.v23_cache is not None else ""),
                replacement_then_restart_after_long_delay_trajectory_cid=str(
                    cache_v24
                    .replacement_then_restart_after_long_delay_trajectory_cid),
                replacement_then_restart_after_long_delay_count=int(
                    len(
                        cache_v24
                        .replacement_then_restart_after_long_delay_windows)),
                long_delay_blackout_window_turns=58,
                visible_token_budget=128.0,
                baseline_cost=512.0))
        kv_witness = emit_kv_bridge_v24_witness(
            projection=p.kv_bridge_v24,
            replacement_then_restart_after_long_delay_falsifier=(
                rtrld_falsifier),
            replacement_then_restart_after_long_delay_fingerprint=(
                rtrld_fp))
        cache_witness = emit_cache_controller_v22_witness(
            controller=p.cache_controller_v22)
        replay_witness = emit_replay_controller_v20_witness(
            p.replay_controller_v20)
        persist_chain = PersistentLatentStateV31Chain.empty()
        persist_witness = emit_persistent_v31_witness(
            persist_chain)
        # MLSC V27 chain — wrap up to V27.
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
        from .mergeable_latent_capsule_v26 import wrap_v25_as_v26
        v3 = make_root_capsule_v3(
            branch_id="w79_smoke",
            payload=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            fact_tags=("w79",), confidence=0.9, trust=0.9,
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
            post_restart_replacement_trajectory_chain=(),
            post_restart_replacement_window_chain=())
        v_chain = wrap_v25_as_v26(
            v_chain,
            long_horizon_reconstruction_trajectory_chain=(),
            reconstruction_request_window_chain=())
        v27_c = wrap_v26_as_v27(
            v_chain,
            replacement_then_restart_after_long_delay_trajectory_chain=(
                str(cache_v24
                    .replacement_then_restart_after_long_delay_trajectory_cid),),
            replacement_then_restart_after_long_delay_window_chain=(
                f"rtw_{int(len(cache_v24.replacement_then_restart_after_long_delay_windows))}",))
        mlsc_witness = emit_mlsc_v27_witness(v27_c)
        consensus_witness = emit_consensus_v25_witness(
            p.consensus_v25)
        lhr_witness = emit_lhr_v31_witness(
            p.lhr_v31, carrier=[0.1] * 6, k=16,
            long_horizon_reconstruction_indicator=[0.95] * 8,
            replacement_then_restart_after_long_delay_indicator=[
                0.97] * 8,
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
        # Deep substrate hybrid V24.
        v23_witness = DeepSubstrateHybridV23ForwardWitness(
            schema="coordpy.deep_substrate_hybrid_v23.v1",
            hybrid_cid="",
            inner_v22_witness_cid="",
            twenty_three_way=True,
            cache_controller_v21_fired=True,
            replay_controller_v19_fired=True,
            long_horizon_reconstruction_trajectory_active=True,
            long_horizon_reconstruction_repair_active=True,
            team_consensus_controller_v13_active=True,
            long_horizon_reconstruction_trajectory_cid=str(
                cache_v24.v23_cache.long_horizon_reconstruction_trajectory_cid
                if cache_v24.v23_cache is not None else ""),
            long_horizon_reconstruction_repair_l1=1,
            long_horizon_reconstruction_pressure_gate_mean=0.5,
        )
        deep_v24_witness = deep_substrate_hybrid_v24_forward(
            hybrid=p.deep_substrate_hybrid_v24,
            v23_witness=v23_witness,
            replacement_then_restart_after_long_delay_trajectory_cid=str(
                cache_v24
                .replacement_then_restart_after_long_delay_trajectory_cid),
            replacement_then_restart_after_long_delay_repair_l1=int(
                trace_v24
                .replacement_then_restart_after_long_delay_l1),
            replacement_then_restart_after_long_delay_pressure_gate_mean=float(
                trace_v24
                .replacement_then_restart_after_long_delay_pressure_gate_mean),
            n_team_consensus_v14_invocations=1)
        adapter_matrix = probe_all_v24_adapters()
        # MASC V15 — run a batch across all 19 regimes.
        per_regime_aggs = {}
        for regime in W79_MASC_V15_REGIMES:
            _, agg = p.multi_agent_coordinator_v15.run_batch(
                seeds=list(range(int(p.masc_v15_n_seeds))),
                regime=regime)
            per_regime_aggs[regime] = agg
        masc_witness = (
            emit_multi_agent_substrate_coordinator_v15_witness(
                coordinator=p.multi_agent_coordinator_v15,
                per_regime_aggregate=per_regime_aggs))
        # TCC V14 — fire arbiters.
        tcc_v14 = p.team_consensus_controller_v14
        tcc_v14.decide_v14(
            regime=(
                "long_delay_reconstruction_after_replacement_then_restart"),
            agent_guesses=[1.0, -1.0, 0.5, 0.2],
            agent_confidences=[0.8, 0.6, 0.7, 0.7],
            substrate_replay_trust=0.7,
            replacement_then_restart_after_long_delay_trajectory_cid=str(
                cache_v24
                .replacement_then_restart_after_long_delay_trajectory_cid),
            long_delay_blackout_window_turns=58,
            agent_replacement_then_restart_after_long_delay_absorption_scores=[
                0.97, 0.6, 0.5, 0.4])
        tcc_v14.decide_v14(
            regime="baseline",
            agent_guesses=[0.5, 0.5, 0.4, 0.5],
            agent_confidences=[0.8, 0.6, 0.7, 0.7],
            substrate_replay_trust=0.7,
            replacement_then_restart_after_long_delay_pressure=0.9,
            agent_replacement_then_restart_after_long_delay_recovery_flags=[
                1, 0, 1, 0])
        tcc_witness = emit_team_consensus_controller_v14_witness(
            tcc_v14)
        # Plane A V12 — hosted.
        cache_planned, _ = (
            p.hosted_cache_planner_v12
            .plan_per_role_ten_layer_rotated(
                shared_prefix_text=(
                    "W79 team shared prefix " * 16),
                per_role_blocks={
                    "plan": ["t0", "t1"],
                    "research": ["r0", "r1"],
                    "write": ["w0", "w1"],
                    "review": ["v0", "v1"],
                }))
        # Router V12.
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
        from .hosted_router_controller_v10 import (
            HostedRoutingRequestV10 as _R10,
        )
        from .hosted_router_controller_v11 import (
            HostedRoutingRequestV11 as _R11,
        )
        req_v11 = _R11(
            inner_v10=_R10(
                inner_v9=_R9(
                    inner_v8=_R8(
                        inner_v7=_R7(
                            inner_v6=_R6(
                                inner_v5=_R5(
                                    inner_v4=_R4(
                                        inner_v3=_R3(
                                            inner_v2=_R2(
                                                inner_v1=_R1(
                                                    request_cid=(
                                                        "w79-router-turn"),
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
                    weight_chain_then_restart_after_rtr_match=0.4),
                post_restart_replacement_pressure=0.9,
                weight_post_restart_replacement_pressure=0.6,
                weight_post_restart_replacement_after_pcr_match=0.4),
            long_horizon_reconstruction_pressure=0.95,
            weight_long_horizon_reconstruction_pressure=0.6,
            weight_long_horizon_reconstruction_after_pcr_match=0.4)
        req_v12 = HostedRoutingRequestV12(
            inner_v11=req_v11,
            replacement_then_restart_after_long_delay_pressure=0.97,
            weight_replacement_then_restart_after_long_delay_pressure=0.6,
            weight_replacement_then_restart_after_long_delay_match=0.4)
        router_dec = p.hosted_router_v12.decide_v12(req_v12)
        router_v12_witness = (
            emit_hosted_router_controller_v12_witness(
                p.hosted_router_v12))
        logprob_v12_witness = (
            emit_hosted_logprob_router_v12_witness(
                p.hosted_logprob_router_v12))
        cache_planner_v12_witness = (
            emit_hosted_cache_aware_planner_v12_witness(
                p.hosted_cache_planner_v12))
        cost_planner_v12_witness = (
            emit_hosted_cost_planner_v12_witness(
                p.hosted_cost_planner_v12))
        boundary_v12 = p.hosted_real_substrate_boundary_v12
        wall_v12_report = build_wall_report_v12(
            boundary=boundary_v12)
        # Provider filter V11.
        _, filter_report = filter_hosted_registry_v11(
            p.hosted_registry, p.hosted_provider_filter_v11,
            provider_replacement_then_restart_after_long_delay_noise={
                "openrouter_paid": 0.5,
                "openai_paid": 0.1})
        filter_report_cid = _sha256_hex({
            "kind": "w79_provider_filter_v11_report",
            "report": dict(filter_report),
        })
        # Handoff coordinator V11 decisions.
        from .hosted_real_handoff_coordinator import (
            HandoffRequest,
        )
        from .hosted_real_handoff_coordinator_v2 import (
            HandoffRequestV2,
        )
        from .hosted_real_handoff_coordinator_v3 import (
            HandoffRequestV3,
        )
        from .hosted_real_handoff_coordinator_v4 import (
            HandoffRequestV4,
        )
        from .hosted_real_handoff_coordinator_v5 import (
            HandoffRequestV5,
        )
        from .hosted_real_handoff_coordinator_v6 import (
            HandoffRequestV6,
        )
        from .hosted_real_handoff_coordinator_v7 import (
            HandoffRequestV7,
        )
        from .hosted_real_handoff_coordinator_v8 import (
            HandoffRequestV8,
        )
        from .hosted_real_handoff_coordinator_v9 import (
            HandoffRequestV9,
        )
        from .hosted_real_handoff_coordinator_v10 import (
            HandoffRequestV10,
        )

        def _make_req_v11(
                rc: str,
                rtrld_pressure: float = 0.0,
                rtrld_trajectory_cid: str = "",
                long_delay_blackout: int = 0,
                needs_controlled_runtime: bool = False,
                needs_text_only: bool = True,
                needs_substrate_state_access: bool = False,
        ) -> HandoffRequestV11:
            base_v10 = HandoffRequestV10(
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
                                                    needs_text_only=bool(needs_text_only),
                                                    needs_substrate_state_access=bool(needs_substrate_state_access)),
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
                long_horizon_reconstruction_pressure=0.0,
                long_horizon_reconstruction_trajectory_cid="",
                long_horizon_blackout_window_turns=0,
                expected_substrate_trust_v10=0.7)
            return HandoffRequestV11(
                inner_v10=base_v10,
                replacement_then_restart_after_long_delay_pressure=float(
                    rtrld_pressure),
                replacement_then_restart_after_long_delay_trajectory_cid=str(
                    rtrld_trajectory_cid),
                long_delay_blackout_window_turns=int(
                    long_delay_blackout),
                needs_controlled_runtime=bool(
                    needs_controlled_runtime),
                expected_substrate_trust_v11=0.7)

        rtrld_traj = str(
            cache_v24
            .replacement_then_restart_after_long_delay_trajectory_cid)
        env_text_only = p.handoff_coordinator_v11.decide_v11(
            req_v11=_make_req_v11("w79-turn-text"))
        env_rtrld_promoted = p.handoff_coordinator_v11.decide_v11(
            req_v11=_make_req_v11(
                "w79-turn-rtrld",
                rtrld_pressure=0.97,
                rtrld_trajectory_cid=rtrld_traj,
                long_delay_blackout=58))
        env_rtrld_fallback = p.handoff_coordinator_v11.decide_v11(
            req_v11=_make_req_v11(
                "w79-turn-rtrld-fb",
                rtrld_pressure=0.0,
                rtrld_trajectory_cid=rtrld_traj,
                long_delay_blackout=100))
        env_ctrl_runtime = p.handoff_coordinator_v11.decide_v11(
            req_v11=_make_req_v11(
                "w79-turn-ctrl",
                needs_controlled_runtime=True))
        env_substrate_only = p.handoff_coordinator_v11.decide_v11(
            req_v11=_make_req_v11(
                "w79-turn-substrate",
                needs_text_only=False,
                needs_substrate_state_access=True))
        handoff_v11_witness = (
            emit_hosted_real_handoff_coordinator_v11_witness(
                p.handoff_coordinator_v11))
        handoff_envelope_chain_cid = _sha256_hex({
            "kind": "w79_handoff_envelope_v11_chain",
            "envelopes": [
                env_text_only.cid(),
                env_rtrld_promoted.cid(),
                env_rtrld_fallback.cid(),
                env_ctrl_runtime.cid(),
                env_substrate_only.cid(),
            ],
        })
        # Long-horizon-reconstruction substrate V2 (with learned
        # consolidation head).
        head_for_carrier = (
            p.learned_consolidation_head_v1)
        carrier_v2 = (
            build_default_long_horizon_reconstruction_carrier_v2(
                n_events=256, seed=_seed_for_carrier_v2(p),
                head=head_for_carrier))
        recon_queries = [
            LongHorizonReconstructionQuery(
                query_id=f"q{i}",
                source_turn=int(i * 30),
                current_turn=int(220 + i * 5))
            for i in range(5)]
        recon_outcomes = [
            reconstruct_long_horizon_event_v2(
                carrier_v2=carrier_v2, query=q,
                visible_tokens_used=4)
            for q in recon_queries]
        recon_economics = (
            report_reconstruction_vs_recompute_economics(
                query=recon_queries[-1],
                carrier=carrier_v2.inner_v1))
        arbitrations = [
            arbitrate_replay_vs_recompute_v2(
                carrier_v2=carrier_v2, query=q,
                controlled_runtime_params_cid=str(
                    p.controlled_runtime_params_v1.cid()),
                replacement_then_restart_pressure=0.95)
            for q in recon_queries]
        recon_v2_witness = (
            emit_long_horizon_reconstruction_witness_v2(
                carrier_v2=carrier_v2,
                outcomes=recon_outcomes,
                economics=recon_economics,
                arbitrations=arbitrations))
        # Bounded-window baseline V2 falsifier.
        baselines_v2 = (
            build_default_bounded_window_baselines_v2())
        from .bounded_window_baseline_v1 import (
            build_default_bounded_window_baselines,
            emit_bounded_window_baseline_witness,
            prove_bounded_window_insufficient,
            run_bounded_window_falsifier,
        )
        v1_baselines = build_default_bounded_window_baselines()
        bw_query = BoundedWindowQuery(
            query_id="bw_q1_w79", current_turn=220,
            source_turn=10,
            expected_event_cid=str(
                carrier_v2.inner_v1.entries[10].event_cid))
        _, bw_v1_fals = run_bounded_window_falsifier(
            baselines=v1_baselines, query=bw_query)
        bw_v1_proof = prove_bounded_window_insufficient(
            query_horizon_turns=210,
            baselines=v1_baselines)
        bw_v1_w = emit_bounded_window_baseline_witness(
            baselines=v1_baselines, proof=bw_v1_proof,
            falsifier=bw_v1_fals)
        _, bw_v2_falsifier = run_bounded_window_falsifier_v2(
            baselines_v2=baselines_v2, query=bw_query)
        bw_v2_proof = prove_bounded_window_insufficient_v2(
            query_horizon_turns=210,
            baselines_v2=baselines_v2)
        bw_v2_witness = (
            emit_bounded_window_baseline_witness_v2(
                baselines_v2=baselines_v2,
                proof_v2=bw_v2_proof,
                falsifier_v2=bw_v2_falsifier,
                inner_v1_witness=bw_v1_w))
        # ---- Direct-blocker-attack pillars.
        # D1: controlled runtime.
        ctrl_ids = tokenize_bytes_v79(
            "w79 controlled-runtime smoke",
            max_len=32)
        ctrl_trace, ctrl_cache = forward_controlled_runtime(
            params=p.controlled_runtime_params_v1,
            input_token_ids=ctrl_ids)
        ctrl_witness = emit_controlled_runtime_witness(
            params=p.controlled_runtime_params_v1,
            trace=ctrl_trace, kv_cache=ctrl_cache)
        ctrl_rvr = (
            measure_controlled_runtime_replay_vs_recompute(
                params=p.controlled_runtime_params_v1,
                old_token_ids=ctrl_ids[:8],
                new_token_ids=ctrl_ids[8:]))
        # D2: OpenAI-compatible façade with substrate side channel.
        facade_req = LocalOpenAIChatCompletionRequestV1(
            model="coordpy.controlled_runtime_substrate_v1",
            messages=(
                LocalOpenAIChatMessageV1(
                    role="user",
                    content=(
                        "Solve context for multi-agent teams")),),
            substrate_return_hidden_state=True,
            substrate_return_kv_cache_cid=True,
            substrate_return_attention_probs=True)
        facade_resp = (
            p.local_openai_facade_v1.chat_completions_create(
                request=facade_req))
        facade_witness = emit_local_openai_facade_witness(
            p.local_openai_facade_v1)
        compare_facade_vs_hosted_substrate(
            facade_response=facade_resp,
            hosted_blocked_axes=tuple(
                boundary_v12.blocked_axes))
        # D3: learned consolidation.
        X, Y = build_nonlinear_consolidation_dataset(
            n_samples=96, seed=_seed_for_learned(p))
        trained_head, train_rep = (
            train_learned_consolidation_head(
                head=p.learned_consolidation_head_v1,
                train_features=X.tolist(),
                train_targets=Y.tolist()))
        vs_ridge = compare_learned_vs_closed_form(
            head=trained_head,
            eval_features=X.tolist(),
            eval_targets=Y.tolist())
        lc_witness = emit_learned_consolidation_witness(
            head=trained_head, vs_closed_form=vs_ridge)
        # Aggregate stats from MASC V15 baseline regime for
        # headline.
        baseline_agg = per_regime_aggs.get("baseline")
        v24_beats = (
            float(baseline_agg.v24_beats_v23_rate)
            if baseline_agg is not None else 0.0)
        tsc_v24_beats = (
            float(baseline_agg.tsc_v24_beats_tsc_v23_rate)
            if baseline_agg is not None else 0.0)
        ts_per_vt = (
            float(
                baseline_agg.team_success_per_visible_token_v24)
            if baseline_agg is not None else 0.0)
        n_succ = int(sum(
            1 for o in recon_outcomes if o.success))
        succ_rate = (
            float(n_succ) / float(len(recon_outcomes))
            if recon_outcomes else 0.0)
        return W79HandoffEnvelope(
            schema=W79_SCHEMA_VERSION,
            w78_outer_cid=str(w78_outer_cid),
            w79_params_cid=str(p.cid()),
            substrate_v24_witness_cid=str(sub_witness.cid()),
            kv_bridge_v24_witness_cid=str(kv_witness.cid()),
            cache_controller_v22_witness_cid=str(
                cache_witness.cid()),
            replay_controller_v20_witness_cid=str(
                replay_witness.cid()),
            persistent_v31_witness_cid=str(
                persist_witness.cid()),
            mlsc_v27_witness_cid=str(mlsc_witness.cid()),
            consensus_v25_witness_cid=str(
                consensus_witness.cid()),
            lhr_v31_witness_cid=str(lhr_witness.cid()),
            deep_substrate_hybrid_v24_witness_cid=str(
                deep_v24_witness.cid()),
            substrate_adapter_v24_matrix_cid=str(
                adapter_matrix.cid()),
            masc_v15_witness_cid=str(masc_witness.cid()),
            team_consensus_controller_v14_witness_cid=str(
                tcc_witness.cid()),
            replacement_then_restart_after_long_delay_pressure_falsifier_witness_cid=(
                str(rtrld_falsifier.cid())),
            hosted_router_v12_witness_cid=str(
                router_v12_witness.cid()),
            hosted_logprob_router_v12_witness_cid=str(
                logprob_v12_witness.cid()),
            hosted_cache_planner_v12_witness_cid=str(
                cache_planner_v12_witness.cid()),
            hosted_cost_planner_v12_witness_cid=str(
                cost_planner_v12_witness.cid()),
            hosted_real_substrate_boundary_v12_cid=str(
                boundary_v12.cid()),
            hosted_wall_v12_report_cid=str(
                wall_v12_report.cid()),
            handoff_coordinator_v11_witness_cid=str(
                handoff_v11_witness.cid()),
            handoff_envelope_v11_chain_cid=str(
                handoff_envelope_chain_cid),
            provider_filter_v11_report_cid=str(
                filter_report_cid),
            long_horizon_reconstruction_v2_witness_cid=str(
                recon_v2_witness.cid()),
            bounded_window_baseline_v2_witness_cid=str(
                bw_v2_witness.cid()),
            bounded_window_insufficiency_proof_v2_cid=str(
                bw_v2_proof.cid()),
            controlled_runtime_witness_cid=str(
                ctrl_witness.cid()),
            controlled_runtime_replay_vs_recompute_report_cid=(
                str(ctrl_rvr.cid())),
            local_openai_facade_witness_cid=str(
                facade_witness.cid()),
            local_openai_facade_response_cid=str(
                facade_resp.cid()),
            learned_consolidation_witness_cid=str(
                lc_witness.cid()),
            twenty_four_way_used=bool(
                deep_v24_witness.twenty_four_way),
            substrate_v24_used=True,
            controlled_runtime_used=True,
            masc_v15_v24_beats_v23_rate=float(v24_beats),
            masc_v15_tsc_v24_beats_tsc_v23_rate=float(
                tsc_v24_beats),
            masc_v15_team_success_per_visible_token=float(
                ts_per_vt),
            hosted_router_v12_chosen=str(
                router_dec.chosen_provider or ""),
            replacement_then_restart_after_long_delay_trajectory_cid=str(
                cache_v24
                .replacement_then_restart_after_long_delay_trajectory_cid),
            long_horizon_reconstruction_v2_success_rate=float(
                succ_rate),
            bounded_window_v2_all_fixed_k_failed=bool(
                bw_v2_falsifier.all_fixed_k_failed_v2),
            learned_consolidation_beats_ridge=bool(
                vs_ridge.learned_strictly_beats_ridge),
            facade_substrate_side_channel_emitted=bool(
                facade_resp.substrate_side_channel is not None),
            controlled_runtime_replay_byte_identical=bool(
                ctrl_rvr.replay_byte_identical),
        )


def _trivial_envelope(
        p: W79Params, w78_outer_cid: str,
) -> W79HandoffEnvelope:
    return W79HandoffEnvelope(
        schema=W79_SCHEMA_VERSION,
        w78_outer_cid=str(w78_outer_cid),
        w79_params_cid=str(p.cid()),
        substrate_v24_witness_cid="",
        kv_bridge_v24_witness_cid="",
        cache_controller_v22_witness_cid="",
        replay_controller_v20_witness_cid="",
        persistent_v31_witness_cid="",
        mlsc_v27_witness_cid="",
        consensus_v25_witness_cid="",
        lhr_v31_witness_cid="",
        deep_substrate_hybrid_v24_witness_cid="",
        substrate_adapter_v24_matrix_cid="",
        masc_v15_witness_cid="",
        team_consensus_controller_v14_witness_cid="",
        replacement_then_restart_after_long_delay_pressure_falsifier_witness_cid="",
        hosted_router_v12_witness_cid="",
        hosted_logprob_router_v12_witness_cid="",
        hosted_cache_planner_v12_witness_cid="",
        hosted_cost_planner_v12_witness_cid="",
        hosted_real_substrate_boundary_v12_cid="",
        hosted_wall_v12_report_cid="",
        handoff_coordinator_v11_witness_cid="",
        handoff_envelope_v11_chain_cid="",
        provider_filter_v11_report_cid="",
        long_horizon_reconstruction_v2_witness_cid="",
        bounded_window_baseline_v2_witness_cid="",
        bounded_window_insufficiency_proof_v2_cid="",
        controlled_runtime_witness_cid="",
        controlled_runtime_replay_vs_recompute_report_cid="",
        local_openai_facade_witness_cid="",
        local_openai_facade_response_cid="",
        learned_consolidation_witness_cid="",
        twenty_four_way_used=False,
        substrate_v24_used=False,
        controlled_runtime_used=False,
        masc_v15_v24_beats_v23_rate=0.0,
        masc_v15_tsc_v24_beats_tsc_v23_rate=0.0,
        masc_v15_team_success_per_visible_token=0.0,
        hosted_router_v12_chosen="",
        replacement_then_restart_after_long_delay_trajectory_cid="",
        long_horizon_reconstruction_v2_success_rate=0.0,
        bounded_window_v2_all_fixed_k_failed=False,
        learned_consolidation_beats_ridge=False,
        facade_substrate_side_channel_emitted=False,
        controlled_runtime_replay_byte_identical=False,
    )


def _seed_for_carrier_v2(p: W79Params) -> int:
    cid = str(p.cid())
    h = int(cid[:8], 16) & 0x7FFFFFFF
    return int(h) ^ 0x79A79A79


def _seed_for_learned(p: W79Params) -> int:
    cid = str(p.cid())
    h = int(cid[8:16], 16) & 0x7FFFFFFF
    return int(h) ^ 0x79B79B79


def build_default_w79_team(*, seed: int = 79000) -> W79Team:
    return W79Team(params=W79Params.build_default(seed=int(seed)))


__all__ = [
    "W79_SCHEMA_VERSION",
    "W79_FAILURE_MODES",
    "W79Params",
    "W79HandoffEnvelope",
    "verify_w79_handoff",
    "W79Team",
    "build_default_w79_team",
]
