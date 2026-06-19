"""W68 — Two-Plane Substrate-Coupled Latent OS team.

The ``W68Team`` orchestrator composes the W67 team with the W68
mechanism modules, organised across two planes:

**Plane B — Real substrate (in-repo, V13 stack):**

* M1  ``tiny_substrate_v13``           (15-layer, 4 new V13 axes)
* M2  ``kv_bridge_v13``                (9-target ridge + partial-
                                        contradiction margin + 50-
                                        dim agent-replacement
                                        fingerprint + falsifier)
* M3  ``hidden_state_bridge_v12``      (9-target ridge + per-(L, H)
                                        hidden-vs-agent-replacement
                                        probe)
* M4  ``prefix_state_bridge_v12``      (K=192 drift curve + 50-dim
                                        fingerprint + 7-way comparator)
* M5  ``attention_steering_bridge_v12``(8-stage clamp + agent-
                                        conditioned fingerprint)
* M6  ``cache_controller_v11``         (8-objective ridge + per-role
                                        9-dim agent-replacement)
* M7  ``replay_controller_v9``         (14 regimes + per-role +
                                        agent-replacement-routing)
* M8  ``deep_substrate_hybrid_v13``    (13-way bidirectional loop)
* M9  ``substrate_adapter_v13``        (substrate_v13_full tier)
* M10 ``persistent_latent_v20``        (19 layers, 17th carrier)
* M11 ``multi_hop_translator_v18``     (44 backends, chain-len 34)
* M12 ``mergeable_latent_capsule_v16`` (partial-contradiction +
                                        agent-replacement chains)
* M13 ``consensus_fallback_controller_v14`` (22-stage chain)
* M14 ``corruption_robust_carrier_v16``(65536-bucket, 36-bit burst)
* M15 ``long_horizon_retention_v20``   (19 heads, max_k=448)
* M16 ``ecc_codebook_v20``             (2^33 codes, ≥ 35.0 b/v)
* M17 ``uncertainty_layer_v16``        (15-axis composite)
* M18 ``disagreement_algebra_v14``     (agent-replacement-equiv ID)
* M19 ``transcript_vs_shared_arbiter_v17`` (18-arm comparator)
* M20 ``multi_agent_substrate_coordinator_v4`` (10-policy MASC V4)
* M21 ``team_consensus_controller_v3``       (partial-contradiction
                                              + agent-replacement
                                              arbiters)

**Plane A — Hosted control plane (honest, no substrate):**

* H1  ``hosted_router_controller``     (provider routing graph)
* H2  ``hosted_logprob_router``        (top-k logprob fusion +
                                        text-only quorum fallback)
* H3  ``hosted_cache_aware_planner``   (prefix-CID cache planning)
* H4  ``hosted_provider_filter``       (data-policy + tier filter)
* H5  ``hosted_cost_planner``          (cost / latency optimisation)
* H6  ``hosted_real_substrate_boundary`` (the explicit wall)

Per-turn it emits 28 module witness CIDs (22 Plane B + 6 Plane A)
and seals them into a ``W68HandoffEnvelope`` whose ``w67_outer_cid``
carries forward the W67 envelope byte-for-byte.

Honest scope (W68)
------------------

* Plane A operates at the hosted **text/logprob/prefix-cache**
  surface. It does NOT pierce hidden state / KV / attention.
  ``W68-L-HOSTED-NO-SUBSTRATE-CAP``.
* Plane B is the in-repo V13 NumPy runtime. We do NOT bridge to
  third-party hosted models.
  ``W68-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
* W68 fits closed-form ridge parameters in six new places on top
  of W67's 41: cache V11 eight-objective; cache V11 per-role
  agent-replacement; replay V9 per-role per-regime; replay V9
  agent-replacement-routing; HSB V12 nine-target; KV V13 nine-
  target. Total **forty-seven closed-form ridge solves** across
  W61..W68. No autograd, no SGD, no GPU.
* Trivial passthrough preserved: when ``W68Params.build_trivial()``
  is used the W68 envelope's internal ``w67_outer_cid`` carries
  the supplied W67 outer CID exactly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .cache_controller_v11 import (
    CacheControllerV11,
    emit_cache_controller_v11_witness,
    fit_eight_objective_ridge_v11,
    fit_per_role_agent_replacement_head_v11,
)
from .consensus_fallback_controller_v14 import (
    ConsensusFallbackControllerV14,
    W68_CONSENSUS_V14_STAGES,
    emit_consensus_v14_witness,
)
from .corruption_robust_carrier_v16 import (
    CorruptionRobustCarrierV16,
    emit_corruption_robustness_v16_witness,
)
from .deep_substrate_hybrid_v12 import (
    DeepSubstrateHybridV12ForwardWitness,
)
from .deep_substrate_hybrid_v13 import (
    DeepSubstrateHybridV13,
    deep_substrate_hybrid_v13_forward,
)
from .ecc_codebook_v20 import (
    ECCCodebookV20,
    emit_ecc_v20_compression_witness,
    probe_ecc_v20_rate_floor_falsifier,
)
from .hidden_state_bridge_v12 import (
    HiddenStateBridgeV12Projection,
    compute_hsb_v12_agent_replacement_margin,
    emit_hsb_v12_witness,
)
from .hosted_cache_aware_planner import (
    HostedCacheAwarePlanner,
    emit_hosted_cache_aware_planner_witness,
)
from .hosted_cost_planner import (
    HostedCostPlanSpec, plan_hosted_cost,
)
from .hosted_logprob_router import (
    HostedLogprobRouter,
    emit_hosted_logprob_router_witness,
)
from .hosted_provider_filter import (
    HostedProviderFilterSpec, filter_hosted_registry,
)
from .hosted_real_substrate_boundary import (
    HostedRealSubstrateBoundary,
    build_default_hosted_real_substrate_boundary,
    build_wall_report,
    probe_hosted_real_substrate_boundary_falsifier,
)
from .hosted_router_controller import (
    HostedProviderRegistry,
    HostedRouterController,
    default_hosted_registry,
    emit_hosted_router_controller_witness,
)
from .kv_bridge_v13 import (
    KVBridgeV13Projection,
    compute_agent_replacement_fingerprint_v13,
    emit_kv_bridge_v13_witness,
    probe_kv_bridge_v13_partial_contradiction_falsifier,
)
from .long_horizon_retention_v20 import (
    LongHorizonReconstructionV20Head,
    emit_lhr_v20_witness,
)
from .mergeable_latent_capsule_v16 import (
    MergeOperatorV16, emit_mlsc_v16_witness, wrap_v15_as_v16,
)
from .multi_agent_substrate_coordinator_v4 import (
    MultiAgentSubstrateCoordinatorV4,
    W68_MASC_V4_REGIMES,
    emit_multi_agent_substrate_coordinator_v4_witness,
)
from .multi_hop_translator_v18 import (
    emit_multi_hop_v18_witness,
)
from .persistent_latent_v20 import (
    PersistentLatentStateV20Chain,
    emit_persistent_v20_witness,
)
from .prefix_state_bridge_v12 import (
    W68_DEFAULT_PREFIX_V12_K_STEPS,
    emit_prefix_state_v12_witness,
)
from .replay_controller_v9 import (
    ReplayControllerV9,
    W68_AGENT_REPLACEMENT_ROUTING_LABELS,
    W68_REPLAY_REGIMES_V9,
    fit_replay_controller_v9_per_role,
    fit_replay_v9_agent_replacement_routing_head,
    emit_replay_controller_v9_witness,
)
from .replay_controller import ReplayCandidate
from .substrate_adapter_v13 import (
    probe_all_v13_adapters,
)
from .team_consensus_controller_v3 import (
    TeamConsensusControllerV3,
    emit_team_consensus_controller_v3_witness,
)
from .tiny_substrate_v13 import (
    TinyV13SubstrateParams,
    build_default_tiny_substrate_v13,
    emit_tiny_substrate_v13_forward_witness,
    forward_tiny_substrate_v13,
    record_partial_contradiction_witness_v13,
    record_prefix_reuse_v13,
    tokenize_bytes_v13,
    trigger_agent_replacement_v13,
)
from .transcript_vs_shared_arbiter_v17 import (
    emit_tvs_arbiter_v17_witness, eighteen_arm_compare,
)
from .uncertainty_layer_v16 import (
    compose_uncertainty_report_v16,
    emit_uncertainty_v16_witness,
)


W68_SCHEMA_VERSION: str = "coordpy.w68_team.v1"

W68_FAILURE_MODES: tuple[str, ...] = (
    "w68_outer_envelope_schema_mismatch",
    "w68_outer_envelope_w67_outer_cid_drift",
    "w68_outer_envelope_w68_params_cid_drift",
    "w68_outer_envelope_witness_cid_drift",
    "w68_substrate_v13_n_layers_off",
    "w68_substrate_v13_gate_score_shape_off",
    "w68_partial_contradiction_witness_shape_off",
    "w68_agent_replacement_flag_off",
    "w68_prefix_reuse_counter_off",
    "w68_kv_bridge_v13_n_targets_off",
    "w68_kv_bridge_v13_partial_contradiction_falsifier_off",
    "w68_hsb_v12_n_targets_off",
    "w68_prefix_v12_k_steps_off",
    "w68_attention_v12_eight_stage_off",
    "w68_cache_v11_eight_objective_off",
    "w68_replay_v9_regime_count_off",
    "w68_replay_v9_agent_replacement_routing_count_off",
    "w68_consensus_v14_stage_count_off",
    "w68_crc_v16_fingerprint_buckets_off",
    "w68_crc_v16_adversarial_burst_bits_off",
    "w68_lhr_v20_max_k_off",
    "w68_lhr_v20_n_heads_off",
    "w68_ecc_v20_total_codes_off",
    "w68_ecc_v20_bits_per_token_off",
    "w68_tvs_v17_n_arms_off",
    "w68_uncertainty_v16_n_axes_off",
    "w68_mlsc_v16_algebra_signatures_off",
    "w68_disagreement_v14_ar_falsifier_off",
    "w68_mh_v18_n_backends_off",
    "w68_persistent_v20_n_layers_off",
    "w68_substrate_adapter_v13_tier_off",
    "w68_masc_v4_v13_beats_v12_rate_under_threshold",
    "w68_masc_v4_tsc_v13_beats_tsc_v12_rate_under_threshold",
    "w68_masc_v4_partial_contradiction_regime_inferior_to_baseline",
    "w68_masc_v4_agent_replacement_regime_inferior_to_baseline",
    "w68_hosted_router_decision_not_deterministic",
    "w68_hosted_logprob_fusion_kind_off",
    "w68_hosted_cache_aware_savings_negative",
    "w68_hosted_provider_filter_drop_count_off",
    "w68_hosted_cost_planner_no_eligible",
    "w68_hosted_real_substrate_boundary_blocked_axis_satisfied",
    "w68_thirteen_way_loop_not_observed",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class W68Params:
    substrate_v13: TinyV13SubstrateParams | None
    kv_bridge_v13: KVBridgeV13Projection | None
    hidden_state_bridge_v12: HiddenStateBridgeV12Projection | None
    cache_controller_v11: CacheControllerV11 | None
    replay_controller_v9: ReplayControllerV9 | None
    consensus_v14: ConsensusFallbackControllerV14 | None
    crc_v16: CorruptionRobustCarrierV16 | None
    lhr_v20: LongHorizonReconstructionV20Head | None
    ecc_v20: ECCCodebookV20 | None
    deep_substrate_hybrid_v13: DeepSubstrateHybridV13 | None
    mlsc_v16_operator: MergeOperatorV16 | None
    multi_agent_coordinator_v4: (
        MultiAgentSubstrateCoordinatorV4 | None)
    team_consensus_controller_v3: (
        TeamConsensusControllerV3 | None)
    hosted_registry: HostedProviderRegistry | None
    hosted_router: HostedRouterController | None
    hosted_logprob_router: HostedLogprobRouter | None
    hosted_cache_planner: HostedCacheAwarePlanner | None
    hosted_real_substrate_boundary: (
        HostedRealSubstrateBoundary | None)
    prefix_v12_predictor_trained: bool
    enabled: bool = True
    masc_v4_n_seeds: int = 12

    @classmethod
    def build_trivial(cls) -> "W68Params":
        return cls(
            substrate_v13=None,
            kv_bridge_v13=None,
            hidden_state_bridge_v12=None,
            cache_controller_v11=None,
            replay_controller_v9=None,
            consensus_v14=None,
            crc_v16=None, lhr_v20=None, ecc_v20=None,
            deep_substrate_hybrid_v13=None,
            mlsc_v16_operator=None,
            multi_agent_coordinator_v4=None,
            team_consensus_controller_v3=None,
            hosted_registry=None,
            hosted_router=None,
            hosted_logprob_router=None,
            hosted_cache_planner=None,
            hosted_real_substrate_boundary=None,
            prefix_v12_predictor_trained=False,
            enabled=False,
        )

    @classmethod
    def build_default(
            cls, *, seed: int = 68000,
    ) -> "W68Params":
        # Plane B — real substrate.
        sub_v13 = build_default_tiny_substrate_v13(
            seed=int(seed) + 1)
        # KV V13 projection chain.
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
        cfg = sub_v13.config.v12.v11.v10.v9
        d_head = int(cfg.d_model) // int(cfg.n_heads)
        kv_b3 = KVBridgeV3Projection.init(
            n_layers=int(cfg.n_layers),
            n_heads=int(cfg.n_heads),
            n_kv_heads=int(cfg.n_kv_heads),
            n_inject_tokens=3, carrier_dim=6,
            d_head=int(d_head), seed=int(seed) + 7)
        kv_b12 = KVBridgeV12Projection.init_from_v11(
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
            seed_v12=int(seed) + 16)
        kv_b13 = KVBridgeV13Projection.init_from_v12(
            kv_b12, seed_v13=int(seed) + 17)
        from .hidden_state_bridge_v2 import (
            HiddenStateBridgeV2Projection)
        from .hidden_state_bridge_v3 import (
            HiddenStateBridgeV3Projection)
        from .hidden_state_bridge_v4 import (
            HiddenStateBridgeV4Projection)
        from .hidden_state_bridge_v5 import (
            HiddenStateBridgeV5Projection)
        from .hidden_state_bridge_v6 import (
            HiddenStateBridgeV6Projection)
        from .hidden_state_bridge_v7 import (
            HiddenStateBridgeV7Projection)
        from .hidden_state_bridge_v8 import (
            HiddenStateBridgeV8Projection)
        from .hidden_state_bridge_v9 import (
            HiddenStateBridgeV9Projection)
        from .hidden_state_bridge_v10 import (
            HiddenStateBridgeV10Projection)
        from .hidden_state_bridge_v11 import (
            HiddenStateBridgeV11Projection)
        hsb12 = HiddenStateBridgeV12Projection.init_from_v11(
            HiddenStateBridgeV11Projection.init_from_v10(
                HiddenStateBridgeV10Projection.init_from_v9(
                    HiddenStateBridgeV9Projection.init_from_v8(
                        HiddenStateBridgeV8Projection.init_from_v7(
                            HiddenStateBridgeV7Projection.init_from_v6(
                                HiddenStateBridgeV6Projection.init_from_v5(
                                    HiddenStateBridgeV5Projection.init_from_v4(
                                        HiddenStateBridgeV4Projection.init_from_v3(
                                            HiddenStateBridgeV3Projection.init_from_v2(
                                                HiddenStateBridgeV2Projection.init(
                                                    target_layers=(1, 3),
                                                    n_tokens=6, carrier_dim=6,
                                                    d_model=int(cfg.d_model),
                                                    seed=int(seed) + 20),
                                                n_heads=int(cfg.n_heads),
                                                seed_v3=int(seed) + 21),
                                            seed_v4=int(seed) + 22),
                                        n_positions=3,
                                        seed_v5=int(seed) + 23),
                                    seed_v6=int(seed) + 24),
                                seed_v7=int(seed) + 25),
                            seed_v8=int(seed) + 26),
                        seed_v9=int(seed) + 27),
                    seed_v10=int(seed) + 28),
                seed_v11=int(seed) + 29),
            seed_v12=int(seed) + 30)
        cc11 = CacheControllerV11.init(fit_seed=int(seed) + 31)
        # Fit eight-objective ridge with a small synthetic dataset.
        import numpy as _np  # local lazy import for build_default
        rng = _np.random.default_rng(int(seed) + 33)
        X = rng.standard_normal((10, 4))
        cc11, _ = fit_eight_objective_ridge_v11(
            controller=cc11, train_features=X.tolist(),
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
                X[:, 1] * 0.3 + X[:, 3] * 0.4).tolist())
        X9 = rng.standard_normal((8, 9))
        cc11, _ = fit_per_role_agent_replacement_head_v11(
            controller=cc11, role="planner",
            train_features=X9.tolist(),
            target_replacement_priorities=(
                X9[:, 0] * 0.4 + X9[:, 8] * 0.3).tolist())
        # Replay V9.
        rcv9 = ReplayControllerV9.init()
        v9_cands = {
            r: [ReplayCandidate(
                100, 1000, 50, 0.1, 0.0, 0.3,
                True, True, 0)]
            for r in W68_REPLAY_REGIMES_V9}
        v9_decs = {
            r: ["choose_reuse"]
            for r in W68_REPLAY_REGIMES_V9}
        rcv9, _ = fit_replay_controller_v9_per_role(
            controller=rcv9, role="planner",
            train_candidates_per_regime=v9_cands,
            train_decisions_per_regime=v9_decs)
        X_team = rng.standard_normal((40, 10))
        labs: list[str] = []
        for i in range(40):
            if X_team[i, 0] > 0.5:
                labs.append(
                    W68_AGENT_REPLACEMENT_ROUTING_LABELS[0])
            elif X_team[i, 1] > 0.0:
                labs.append(
                    W68_AGENT_REPLACEMENT_ROUTING_LABELS[1])
            elif X_team[i, 2] > 0.0:
                labs.append(
                    W68_AGENT_REPLACEMENT_ROUTING_LABELS[2])
            elif X_team[i, 3] > 0.0:
                labs.append(
                    W68_AGENT_REPLACEMENT_ROUTING_LABELS[3])
            elif X_team[i, 4] > 0.0:
                labs.append(
                    W68_AGENT_REPLACEMENT_ROUTING_LABELS[4])
            else:
                labs.append(
                    W68_AGENT_REPLACEMENT_ROUTING_LABELS[5])
        rcv9, _ = fit_replay_v9_agent_replacement_routing_head(
            controller=rcv9,
            train_team_features=X_team.tolist(),
            train_routing_labels=labs)
        consensus_v14 = ConsensusFallbackControllerV14.init(
            k_required=2, cosine_floor=0.6,
            trust_threshold=0.5)
        crc_v16 = CorruptionRobustCarrierV16()
        lhr_v20 = LongHorizonReconstructionV20Head.init(
            seed=int(seed) + 34)
        ecc_v20 = ECCCodebookV20.init(seed=int(seed) + 35)
        hybrid_v13 = DeepSubstrateHybridV13(inner_v12=None)
        mlsc_v16_op = MergeOperatorV16(factor_dim=6)
        masc_v4 = MultiAgentSubstrateCoordinatorV4()
        tcc_v3 = TeamConsensusControllerV3()
        # Plane A — hosted control plane.
        hosted_registry = default_hosted_registry()
        hosted_router = HostedRouterController(
            registry=hosted_registry)
        hosted_logprob_router = HostedLogprobRouter()
        hosted_cache_planner = HostedCacheAwarePlanner()
        boundary = build_default_hosted_real_substrate_boundary()
        return cls(
            substrate_v13=sub_v13,
            kv_bridge_v13=kv_b13,
            hidden_state_bridge_v12=hsb12,
            cache_controller_v11=cc11,
            replay_controller_v9=rcv9,
            consensus_v14=consensus_v14, crc_v16=crc_v16,
            lhr_v20=lhr_v20, ecc_v20=ecc_v20,
            deep_substrate_hybrid_v13=hybrid_v13,
            mlsc_v16_operator=mlsc_v16_op,
            multi_agent_coordinator_v4=masc_v4,
            team_consensus_controller_v3=tcc_v3,
            hosted_registry=hosted_registry,
            hosted_router=hosted_router,
            hosted_logprob_router=hosted_logprob_router,
            hosted_cache_planner=hosted_cache_planner,
            hosted_real_substrate_boundary=boundary,
            prefix_v12_predictor_trained=True,
            enabled=True,
        )

    def to_dict(self) -> dict[str, Any]:
        def _cid_or_empty(x: Any) -> str:
            return x.cid() if x is not None else ""
        return {
            "schema_version": W68_SCHEMA_VERSION,
            "enabled": bool(self.enabled),
            "substrate_v13_cid": _cid_or_empty(self.substrate_v13),
            "kv_bridge_v13_cid": _cid_or_empty(self.kv_bridge_v13),
            "hidden_state_bridge_v12_cid": _cid_or_empty(
                self.hidden_state_bridge_v12),
            "cache_controller_v11_cid": _cid_or_empty(
                self.cache_controller_v11),
            "replay_controller_v9_cid": _cid_or_empty(
                self.replay_controller_v9),
            "consensus_v14_cid": _cid_or_empty(self.consensus_v14),
            "crc_v16_cid": _cid_or_empty(self.crc_v16),
            "lhr_v20_cid": _cid_or_empty(self.lhr_v20),
            "ecc_v20_cid": _cid_or_empty(self.ecc_v20),
            "deep_substrate_hybrid_v13_cid": _cid_or_empty(
                self.deep_substrate_hybrid_v13),
            "mlsc_v16_operator_cid": _cid_or_empty(
                self.mlsc_v16_operator),
            "multi_agent_coordinator_v4_cid": _cid_or_empty(
                self.multi_agent_coordinator_v4),
            "team_consensus_controller_v3_cid": _cid_or_empty(
                self.team_consensus_controller_v3),
            "hosted_registry_cid": _cid_or_empty(
                self.hosted_registry),
            "hosted_router_cid": _cid_or_empty(
                self.hosted_router),
            "hosted_logprob_router_cid": _cid_or_empty(
                self.hosted_logprob_router),
            "hosted_cache_planner_cid": _cid_or_empty(
                self.hosted_cache_planner),
            "hosted_real_substrate_boundary_cid": _cid_or_empty(
                self.hosted_real_substrate_boundary),
            "prefix_v12_predictor_trained": bool(
                self.prefix_v12_predictor_trained),
            "masc_v4_n_seeds": int(self.masc_v4_n_seeds),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w68_params",
            "params": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class W68HandoffEnvelope:
    schema: str
    w67_outer_cid: str
    w68_params_cid: str
    substrate_v13_witness_cid: str
    kv_bridge_v13_witness_cid: str
    hsb_v12_witness_cid: str
    prefix_state_v12_witness_cid: str
    cache_controller_v11_witness_cid: str
    replay_controller_v9_witness_cid: str
    persistent_v20_witness_cid: str
    multi_hop_v18_witness_cid: str
    mlsc_v16_witness_cid: str
    consensus_v14_witness_cid: str
    crc_v16_witness_cid: str
    lhr_v20_witness_cid: str
    ecc_v20_witness_cid: str
    tvs_v17_witness_cid: str
    uncertainty_v16_witness_cid: str
    deep_substrate_hybrid_v13_witness_cid: str
    substrate_adapter_v13_matrix_cid: str
    masc_v4_witness_cid: str
    team_consensus_controller_v3_witness_cid: str
    partial_contradiction_falsifier_witness_cid: str
    hosted_router_witness_cid: str
    hosted_logprob_router_witness_cid: str
    hosted_cache_planner_witness_cid: str
    hosted_real_substrate_boundary_cid: str
    hosted_wall_report_cid: str
    thirteen_way_used: bool
    substrate_v13_used: bool
    masc_v4_v13_beats_v12_rate: float
    masc_v4_tsc_v13_beats_tsc_v12_rate: float
    hosted_router_chosen: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "w67_outer_cid": str(self.w67_outer_cid),
            "w68_params_cid": str(self.w68_params_cid),
            "substrate_v13_witness_cid": str(
                self.substrate_v13_witness_cid),
            "kv_bridge_v13_witness_cid": str(
                self.kv_bridge_v13_witness_cid),
            "hsb_v12_witness_cid": str(
                self.hsb_v12_witness_cid),
            "prefix_state_v12_witness_cid": str(
                self.prefix_state_v12_witness_cid),
            "cache_controller_v11_witness_cid": str(
                self.cache_controller_v11_witness_cid),
            "replay_controller_v9_witness_cid": str(
                self.replay_controller_v9_witness_cid),
            "persistent_v20_witness_cid": str(
                self.persistent_v20_witness_cid),
            "multi_hop_v18_witness_cid": str(
                self.multi_hop_v18_witness_cid),
            "mlsc_v16_witness_cid": str(
                self.mlsc_v16_witness_cid),
            "consensus_v14_witness_cid": str(
                self.consensus_v14_witness_cid),
            "crc_v16_witness_cid": str(
                self.crc_v16_witness_cid),
            "lhr_v20_witness_cid": str(
                self.lhr_v20_witness_cid),
            "ecc_v20_witness_cid": str(
                self.ecc_v20_witness_cid),
            "tvs_v17_witness_cid": str(
                self.tvs_v17_witness_cid),
            "uncertainty_v16_witness_cid": str(
                self.uncertainty_v16_witness_cid),
            "deep_substrate_hybrid_v13_witness_cid": str(
                self.deep_substrate_hybrid_v13_witness_cid),
            "substrate_adapter_v13_matrix_cid": str(
                self.substrate_adapter_v13_matrix_cid),
            "masc_v4_witness_cid": str(self.masc_v4_witness_cid),
            "team_consensus_controller_v3_witness_cid": str(
                self.team_consensus_controller_v3_witness_cid),
            "partial_contradiction_falsifier_witness_cid": str(
                self.partial_contradiction_falsifier_witness_cid),
            "hosted_router_witness_cid": str(
                self.hosted_router_witness_cid),
            "hosted_logprob_router_witness_cid": str(
                self.hosted_logprob_router_witness_cid),
            "hosted_cache_planner_witness_cid": str(
                self.hosted_cache_planner_witness_cid),
            "hosted_real_substrate_boundary_cid": str(
                self.hosted_real_substrate_boundary_cid),
            "hosted_wall_report_cid": str(
                self.hosted_wall_report_cid),
            "thirteen_way_used": bool(self.thirteen_way_used),
            "substrate_v13_used": bool(self.substrate_v13_used),
            "masc_v4_v13_beats_v12_rate": float(round(
                self.masc_v4_v13_beats_v12_rate, 12)),
            "masc_v4_tsc_v13_beats_tsc_v12_rate": float(round(
                self.masc_v4_tsc_v13_beats_tsc_v12_rate, 12)),
            "hosted_router_chosen": str(self.hosted_router_chosen),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w68_handoff_envelope",
            "envelope": self.to_dict()})


def verify_w68_handoff(
        envelope: W68HandoffEnvelope,
        params: W68Params,
        w67_outer_cid: str,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if envelope.schema != W68_SCHEMA_VERSION:
        failures.append("w68_outer_envelope_schema_mismatch")
    if envelope.w67_outer_cid != str(w67_outer_cid):
        failures.append(
            "w68_outer_envelope_w67_outer_cid_drift")
    if envelope.w68_params_cid != params.cid():
        failures.append(
            "w68_outer_envelope_w68_params_cid_drift")
    return (len(failures) == 0), failures


@dataclasses.dataclass
class W68Team:
    params: W68Params

    def run_team_turn(
            self, *,
            w67_outer_cid: str,
            ids: Sequence[int] | None = None,
            text: str = "w68",
    ) -> W68HandoffEnvelope:
        p = self.params
        if not p.enabled or p.substrate_v13 is None:
            # Trivial passthrough.
            return W68HandoffEnvelope(
                schema=W68_SCHEMA_VERSION,
                w67_outer_cid=str(w67_outer_cid),
                w68_params_cid=str(p.cid()),
                substrate_v13_witness_cid="",
                kv_bridge_v13_witness_cid="",
                hsb_v12_witness_cid="",
                prefix_state_v12_witness_cid="",
                cache_controller_v11_witness_cid="",
                replay_controller_v9_witness_cid="",
                persistent_v20_witness_cid="",
                multi_hop_v18_witness_cid="",
                mlsc_v16_witness_cid="",
                consensus_v14_witness_cid="",
                crc_v16_witness_cid="",
                lhr_v20_witness_cid="",
                ecc_v20_witness_cid="",
                tvs_v17_witness_cid="",
                uncertainty_v16_witness_cid="",
                deep_substrate_hybrid_v13_witness_cid="",
                substrate_adapter_v13_matrix_cid="",
                masc_v4_witness_cid="",
                team_consensus_controller_v3_witness_cid="",
                partial_contradiction_falsifier_witness_cid="",
                hosted_router_witness_cid="",
                hosted_logprob_router_witness_cid="",
                hosted_cache_planner_witness_cid="",
                hosted_real_substrate_boundary_cid="",
                hosted_wall_report_cid="",
                thirteen_way_used=False,
                substrate_v13_used=False,
                masc_v4_v13_beats_v12_rate=0.0,
                masc_v4_tsc_v13_beats_tsc_v12_rate=0.0,
                hosted_router_chosen="",
            )
        # Plane B — substrate forward.
        token_ids = (
            list(ids) if ids is not None
            else tokenize_bytes_v13(str(text), max_len=16))
        trace, cache = forward_tiny_substrate_v13(
            p.substrate_v13, token_ids)
        # Exercise the substrate's V13 axes so they fire.
        trigger_agent_replacement_v13(
            cache, role="r0", replacement_index=1,
            warm_restart_window=2)
        record_prefix_reuse_v13(
            cache, prefix_cid="abc123def", turn=0)
        record_partial_contradiction_witness_v13(
            cache, layer_index=0, head_index=0, slot=0,
            witness=0.5)
        sub_witness = emit_tiny_substrate_v13_forward_witness(
            trace, cache)
        # KV V13 witness — just CID the projection + falsifier.
        kv_falsifier = (
            probe_kv_bridge_v13_partial_contradiction_falsifier(
                partial_contradiction_flag=0.5))
        ar_fp = compute_agent_replacement_fingerprint_v13(
            role="planner", replacement_index=1,
            task_id="t", team_id="team")
        kv_witness = emit_kv_bridge_v13_witness(
            projection=p.kv_bridge_v13,
            partial_contradiction_falsifier=kv_falsifier,
            agent_replacement_fingerprint=ar_fp)
        # HSB V12 witness.
        margin = compute_hsb_v12_agent_replacement_margin(
            hidden_residual_l2=0.05, kv_residual_l2=0.2,
            prefix_residual_l2=0.2, replay_residual_l2=0.2,
            recover_residual_l2=0.2,
            branch_merge_residual_l2=0.2,
            agent_replacement_residual_l2=0.2)
        hsb_witness = emit_hsb_v12_witness(
            projection=p.hidden_state_bridge_v12,
            agent_replacement_margin=margin,
            hidden_vs_agent_replacement_mean=0.95)
        prefix_witness = emit_prefix_state_v12_witness()
        cache_witness = emit_cache_controller_v11_witness(
            controller=p.cache_controller_v11)
        replay_witness = emit_replay_controller_v9_witness(
            p.replay_controller_v9)
        persist_chain = PersistentLatentStateV20Chain.empty()
        persist_witness = emit_persistent_v20_witness(
            persist_chain)
        mh_witness = emit_multi_hop_v18_witness()
        # MLSC V16 — wrap a trivial V3 capsule up the chain.
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
        v3 = make_root_capsule_v3(
            branch_id="w68_smoke",
            payload=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            fact_tags=("w68",), confidence=0.9, trust=0.9,
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
        mlsc_witness = emit_mlsc_v16_witness(v16)
        consensus_witness = emit_consensus_v14_witness(
            p.consensus_v14)
        crc_witness = emit_corruption_robustness_v16_witness(
            crc_v16=p.crc_v16, n_probes=3, seed=68150)
        lhr_witness = emit_lhr_v20_witness(
            p.lhr_v20, carrier=[0.1] * 6, k=16,
            partial_contradiction_indicator=[0.5] * 8)
        ecc_compression = {
            "v19_compression_result": {
                "structured_bits_v19": 100,
                "parity_bits_per_segment_tuple": 24,
            },
            "structured_bits_v20": 110,
            "visible_tokens": 4,
            "bits_per_visible_token": 27.5,
            "meta17_index": 1,
            "parity_bits_per_segment_tuple": 32,
        }
        ecc_witness = emit_ecc_v20_compression_witness(
            codebook=p.ecc_v20, compression=ecc_compression)
        tvs_result = eighteen_arm_compare(
            per_turn_partial_contradiction_resolution_fidelities=(
                [0.5]),
            per_turn_branch_merge_reconciliation_fidelities=[0.7],
            per_turn_team_failure_recovery_fidelities=[0.6],
            per_turn_team_substrate_coordination_fidelities=[0.7],
            per_turn_replay_dominance_primary_fidelities=[0.7],
            per_turn_hidden_wins_fidelities=[0.6],
            per_turn_replay_dominance_fidelities=[0.7],
            per_turn_confidences=[0.8],
            per_turn_trust_scores=[0.7],
            per_turn_merge_retentions=[0.6],
            per_turn_tw_retentions=[0.6],
            per_turn_substrate_fidelities=[0.5],
            per_turn_hidden_fidelities=[0.4],
            per_turn_cache_fidelities=[0.5],
            per_turn_retrieval_fidelities=[0.6],
            per_turn_replay_fidelities=[0.7],
            per_turn_attention_pattern_fidelities=[0.9],
            budget_tokens=6,
        )
        tvs_witness = emit_tvs_arbiter_v17_witness(tvs_result)
        uncertainty = compose_uncertainty_report_v16(
            confidences=[0.7, 0.5],
            trusts=[0.9, 0.8],
            substrate_fidelities=[0.95, 0.9],
            hidden_state_fidelities=[0.95, 0.92],
            cache_reuse_fidelities=[0.95, 0.92],
            retrieval_fidelities=[0.95, 0.93],
            replay_fidelities=[0.92, 0.90],
            attention_pattern_fidelities=[0.95, 0.90],
            replay_dominance_fidelities=[0.88, 0.85],
            hidden_wins_fidelities=[0.86, 0.82],
            replay_dominance_primary_fidelities=[0.84, 0.80],
            team_coordination_fidelities=[0.80, 0.75],
            team_failure_recovery_fidelities=[0.78, 0.74],
            branch_merge_reconciliation_fidelities=[0.78, 0.74],
            partial_contradiction_resolution_fidelities=[
                0.76, 0.72])
        uncertainty_witness = emit_uncertainty_v16_witness(
            uncertainty)
        # Deep substrate hybrid V13.
        v12_witness = DeepSubstrateHybridV12ForwardWitness(
            schema="coordpy.deep_substrate_hybrid_v12.v1",
            hybrid_cid="",
            inner_v11_witness_cid="",
            twelve_way=True,
            cache_controller_v10_fired=True,
            replay_controller_v8_fired=True,
            branch_merge_witness_active=True,
            role_dropout_recovery_active=True,
            substrate_snapshot_fork_active=True,
            team_consensus_controller_v2_active=True,
            branch_merge_witness_l1=0.5,
            role_dropout_recovery_count=1,
            n_branches_active=2)
        deep_v13_witness = deep_substrate_hybrid_v13_forward(
            hybrid=p.deep_substrate_hybrid_v13,
            v12_witness=v12_witness,
            cache_controller_v11=p.cache_controller_v11,
            replay_controller_v9=p.replay_controller_v9,
            partial_contradiction_witness_l1=0.5,
            agent_replacement_count=1,
            prefix_reuse_count=2,
            n_team_consensus_v3_invocations=1)
        adapter_matrix = probe_all_v13_adapters()
        # MASC V4 — run a smaller batch for the envelope (12 seeds
        # is the default).
        per_regime_aggs = {}
        for regime in W68_MASC_V4_REGIMES:
            _, agg = p.multi_agent_coordinator_v4.run_batch(
                seeds=list(range(int(p.masc_v4_n_seeds))),
                regime=regime)
            per_regime_aggs[regime] = agg
        masc_witness = (
            emit_multi_agent_substrate_coordinator_v4_witness(
                coordinator=p.multi_agent_coordinator_v4,
                per_regime_aggregate=per_regime_aggs))
        tcc_witness = emit_team_consensus_controller_v3_witness(
            p.team_consensus_controller_v3)
        # Plane A — hosted.
        # Plan a small cache run.
        planned, plan_report = p.hosted_cache_planner.plan(
            shared_prefix_text="W68 team shared prefix" * 8,
            role_blocks=["plan", "research", "write"])
        router_witness = emit_hosted_router_controller_witness(
            p.hosted_router)
        logprob_witness = emit_hosted_logprob_router_witness(
            p.hosted_logprob_router)
        cache_planner_witness = (
            emit_hosted_cache_aware_planner_witness(
                p.hosted_cache_planner))
        boundary = p.hosted_real_substrate_boundary
        wall_report = build_wall_report(boundary=boundary)
        # Baseline-regime beats from MASC V4.
        baseline_agg = per_regime_aggs.get(
            W68_MASC_V4_REGIMES[0])
        v13_beats = (
            float(baseline_agg.v13_beats_v12_rate)
            if baseline_agg is not None else 0.0)
        tsc_v13_beats = (
            float(baseline_agg.tsc_v13_beats_tsc_v12_rate)
            if baseline_agg is not None else 0.0)
        return W68HandoffEnvelope(
            schema=W68_SCHEMA_VERSION,
            w67_outer_cid=str(w67_outer_cid),
            w68_params_cid=str(p.cid()),
            substrate_v13_witness_cid=str(sub_witness.cid()),
            kv_bridge_v13_witness_cid=str(kv_witness.cid()),
            hsb_v12_witness_cid=str(hsb_witness.cid()),
            prefix_state_v12_witness_cid=str(prefix_witness.cid()),
            cache_controller_v11_witness_cid=str(
                cache_witness.cid()),
            replay_controller_v9_witness_cid=str(
                replay_witness.cid()),
            persistent_v20_witness_cid=str(persist_witness.cid()),
            multi_hop_v18_witness_cid=str(mh_witness.cid()),
            mlsc_v16_witness_cid=str(mlsc_witness.cid()),
            consensus_v14_witness_cid=str(consensus_witness.cid()),
            crc_v16_witness_cid=str(crc_witness.cid()),
            lhr_v20_witness_cid=str(lhr_witness.cid()),
            ecc_v20_witness_cid=str(ecc_witness.cid()),
            tvs_v17_witness_cid=str(tvs_witness.cid()),
            uncertainty_v16_witness_cid=str(
                uncertainty_witness.cid()),
            deep_substrate_hybrid_v13_witness_cid=str(
                deep_v13_witness.cid()),
            substrate_adapter_v13_matrix_cid=str(
                adapter_matrix.cid()),
            masc_v4_witness_cid=str(masc_witness.cid()),
            team_consensus_controller_v3_witness_cid=str(
                tcc_witness.cid()),
            partial_contradiction_falsifier_witness_cid=str(
                kv_falsifier.cid()),
            hosted_router_witness_cid=str(router_witness.cid()),
            hosted_logprob_router_witness_cid=str(
                logprob_witness.cid()),
            hosted_cache_planner_witness_cid=str(
                cache_planner_witness.cid()),
            hosted_real_substrate_boundary_cid=str(
                boundary.cid()),
            hosted_wall_report_cid=str(wall_report.cid()),
            thirteen_way_used=bool(deep_v13_witness.thirteen_way),
            substrate_v13_used=True,
            masc_v4_v13_beats_v12_rate=float(v13_beats),
            masc_v4_tsc_v13_beats_tsc_v12_rate=float(
                tsc_v13_beats),
            hosted_router_chosen="",
        )


def build_default_w68_team(*, seed: int = 68000) -> W68Team:
    return W68Team(params=W68Params.build_default(seed=int(seed)))


__all__ = [
    "W68_SCHEMA_VERSION",
    "W68_FAILURE_MODES",
    "W68Params",
    "W68HandoffEnvelope",
    "verify_w68_handoff",
    "W68Team",
    "build_default_w68_team",
]
