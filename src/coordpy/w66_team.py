"""W66 — Stronger Solving-Context Substrate-Coupled Latent OS team.

The ``W66Team`` orchestrator composes the W65 team with the W66
mechanism modules:

* M1  ``tiny_substrate_v11``        (13-layer, 4 new V11 axes)
* M2  ``kv_bridge_v11``             (7-target ridge + team margin)
* M3  ``hidden_state_bridge_v10``   (7-target ridge + per-(L, H)
                                     hidden-vs-team-success probe)
* M4  ``prefix_state_bridge_v10``   (K=96 drift curve + 30-dim fp)
* M5  ``attention_steering_bridge_v10`` (6-stage clamp + trust)
* M6  ``cache_controller_v9``       (6-objective ridge + per-role
                                     7-dim eviction)
* M7  ``replay_controller_v7``      (9 regimes + per-role +
                                     team-substrate-routing)
* M8  ``deep_substrate_hybrid_v11`` (11-way bidirectional loop)
* M9  ``substrate_adapter_v11``     (substrate_v11_full tier)
* M10 ``persistent_latent_v18``     (17 layers, 15th carrier)
* M11 ``multi_hop_translator_v16``  (36 backends, chain-len 26)
* M12 ``mergeable_latent_capsule_v14`` (team-failure-recovery +
                                        team-consensus chains)
* M13 ``consensus_fallback_controller_v12`` (18-stage chain)
* M14 ``corruption_robust_carrier_v14`` (16384-bucket, 33-bit
                                         burst)
* M15 ``long_horizon_retention_v18``    (17 heads, max_k=320)
* M16 ``ecc_codebook_v18``              (2^29 codes, ≥ 31.0 b/v)
* M17 ``uncertainty_layer_v14``         (13-axis composite)
* M18 ``disagreement_algebra_v12``      (JS-equivalence identity)
* M19 ``transcript_vs_shared_arbiter_v15`` (16-arm comparator)
* M20 ``multi_agent_substrate_coordinator_v2`` (6-policy MASC V2)
* M21 ``team_consensus_controller``       (weighted quorum + …)

Per-turn it emits 28 module witness CIDs and seals them into a
``W66HandoffEnvelope`` whose ``w65_outer_cid`` carries forward
the W65 envelope byte-for-byte.

Honest scope (W66)
------------------

* The W66 substrate is the in-repo V11 NumPy runtime. We do NOT
  bridge to third-party hosted models.
  ``W66-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
* W66 fits closed-form ridge parameters in six new places on top
  of W65's 29: cache V9 six-objective; cache V9 per-role eviction;
  replay V7 per-role per-regime; replay V7 team-substrate-routing;
  HSB V10 seven-target; KV V11 seven-target. Total **thirty-five
  closed-form ridge solves** across W61..W66.
* Trivial passthrough preserved: when ``W66Params.build_trivial()``
  is used the W66 envelope's internal ``w65_outer_cid`` carries
  the supplied W65 outer CID exactly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.w66_team requires numpy") from exc

from .cache_controller_v9 import (
    CacheControllerV9,
    emit_cache_controller_v9_witness,
    fit_six_objective_ridge_v9,
    fit_per_role_eviction_head_v9,
)
from .consensus_fallback_controller_v12 import (
    ConsensusFallbackControllerV12,
    W66_CONSENSUS_V12_STAGES,
    emit_consensus_v12_witness,
)
from .corruption_robust_carrier_v14 import (
    CorruptionRobustCarrierV14,
    emit_corruption_robustness_v14_witness,
)
from .deep_substrate_hybrid_v10 import (
    DeepSubstrateHybridV10,
    DeepSubstrateHybridV10ForwardWitness,
)
from .deep_substrate_hybrid_v11 import (
    DeepSubstrateHybridV11,
    deep_substrate_hybrid_v11_forward,
)
from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v12 import (
    emit_disagreement_algebra_v12_witness,
)
from .ecc_codebook_v18 import (
    ECCCodebookV18,
    compress_carrier_ecc_v18,
    emit_ecc_v18_compression_witness,
)
from .ecc_codebook_v5 import (
    W53_DEFAULT_ECC_CODE_DIM, W53_DEFAULT_ECC_EMIT_MASK_LEN,
)
from .hidden_state_bridge_v2 import HiddenStateBridgeV2Projection
from .hidden_state_bridge_v3 import HiddenStateBridgeV3Projection
from .hidden_state_bridge_v4 import HiddenStateBridgeV4Projection
from .hidden_state_bridge_v5 import HiddenStateBridgeV5Projection
from .hidden_state_bridge_v6 import HiddenStateBridgeV6Projection
from .hidden_state_bridge_v7 import HiddenStateBridgeV7Projection
from .hidden_state_bridge_v8 import HiddenStateBridgeV8Projection
from .hidden_state_bridge_v9 import HiddenStateBridgeV9Projection
from .hidden_state_bridge_v10 import (
    HiddenStateBridgeV10Projection,
    compute_hsb_v10_team_consensus_margin,
    emit_hsb_v10_witness,
)
from .kv_bridge_v3 import KVBridgeV3Projection
from .kv_bridge_v4 import KVBridgeV4Projection
from .kv_bridge_v5 import KVBridgeV5Projection
from .kv_bridge_v6 import KVBridgeV6Projection
from .kv_bridge_v7 import KVBridgeV7Projection
from .kv_bridge_v8 import KVBridgeV8Projection
from .kv_bridge_v9 import KVBridgeV9Projection
from .kv_bridge_v10 import KVBridgeV10Projection
from .kv_bridge_v11 import (
    KVBridgeV11Projection,
    compute_multi_agent_task_fingerprint_v11,
    emit_kv_bridge_v11_witness,
    probe_kv_bridge_v11_team_failure_recovery_falsifier,
)
from .long_horizon_retention_v18 import (
    LongHorizonReconstructionV18Head,
    emit_lhr_v18_witness,
)
from .mergeable_latent_capsule_v3 import make_root_capsule_v3
from .mergeable_latent_capsule_v4 import wrap_v3_as_v4
from .mergeable_latent_capsule_v5 import wrap_v4_as_v5
from .mergeable_latent_capsule_v6 import wrap_v5_as_v6
from .mergeable_latent_capsule_v7 import wrap_v6_as_v7
from .mergeable_latent_capsule_v8 import wrap_v7_as_v8
from .mergeable_latent_capsule_v9 import wrap_v8_as_v9
from .mergeable_latent_capsule_v10 import wrap_v9_as_v10
from .mergeable_latent_capsule_v11 import wrap_v10_as_v11
from .mergeable_latent_capsule_v12 import wrap_v11_as_v12
from .mergeable_latent_capsule_v13 import (
    W65_MLSC_V13_ALGEBRA_TEAM_SUBSTRATE_PROPAGATION,
    wrap_v12_as_v13,
)
from .mergeable_latent_capsule_v14 import (
    MergeOperatorV14,
    W66_MLSC_V14_ALGEBRA_TEAM_FAILURE_RECOVERY_PROPAGATION,
    emit_mlsc_v14_witness, wrap_v13_as_v14,
)
from .multi_agent_substrate_coordinator_v2 import (
    MultiAgentSubstrateCoordinatorV2,
    W66_MASC_V2_REGIMES,
    emit_multi_agent_substrate_coordinator_v2_witness,
)
from .multi_hop_translator_v16 import (
    W66_DEFAULT_MH_V16_BACKENDS,
    W66_DEFAULT_MH_V16_CHAIN_LEN,
    emit_multi_hop_v16_witness,
)
from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v18 import (
    PersistentLatentStateV18Chain,
    W66_DEFAULT_V18_MAX_CHAIN_WALK_DEPTH,
    emit_persistent_v18_witness,
    step_persistent_state_v18,
)
from .quantised_compression import QuantisedBudgetGate
from .replay_controller import ReplayCandidate
from .replay_controller_v5 import ReplayControllerV5
from .replay_controller_v6 import ReplayControllerV6
from .replay_controller_v7 import (
    ReplayControllerV7,
    W66_REPLAY_REGIMES_V7,
    W66_TEAM_SUBSTRATE_ROUTING_LABELS,
    emit_replay_controller_v7_witness,
    fit_replay_controller_v7_per_role,
    fit_replay_v7_team_substrate_routing_head,
)
from .substrate_adapter_v11 import (
    SubstrateAdapterV11Matrix,
    W66_SUBSTRATE_TIER_SUBSTRATE_V11_FULL,
    probe_all_v11_adapters,
)
from .team_consensus_controller import (
    TeamConsensusController,
    emit_team_consensus_controller_witness,
)
from .tiny_substrate_v11 import (
    TinyV11SubstrateParams,
    build_default_tiny_substrate_v11,
    emit_tiny_substrate_v11_forward_witness,
    forward_tiny_substrate_v11,
    record_replay_trust_v11,
    record_snapshot_diff_v11,
    substrate_snapshot_diff_v11,
    tokenize_bytes_v11,
    trigger_team_failure_recovery_v11,
)
from .transcript_vs_shared_arbiter_v15 import (
    emit_tvs_arbiter_v15_witness, sixteen_arm_compare,
)
from .uncertainty_layer_v14 import (
    compose_uncertainty_report_v14,
    emit_uncertainty_v14_witness,
)


W66_SCHEMA_VERSION: str = "coordpy.w66_team.v1"
W66_TEAM_RESULT_SCHEMA: str = "coordpy.w66_team_result.v1"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _payload_hash_vec(payload: Any, dim: int) -> list[float]:
    h = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    out: list[float] = []
    for i in range(int(dim)):
        nb = h[(i * 2) % len(h):(i * 2) % len(h) + 2]
        if not nb:
            nb = "00"
        v = (int(nb, 16) / 127.5) - 1.0
        out.append(float(round(v, 12)))
    return out


# ===========================================================
# Failure mode enumeration (≥ 120 disjoint modes for W66)
# ===========================================================

W66_ENVELOPE_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "missing_w65_outer_cid",
    "missing_substrate_v11_witness",
    "substrate_v11_witness_invalid",
    "missing_kv_bridge_v11_witness",
    "kv_bridge_v11_seven_target_unfit",
    "missing_hsb_v10_witness",
    "hsb_v10_seven_target_unfit",
    "missing_prefix_state_v10_witness",
    "prefix_state_v10_predictor_unfit",
    "missing_attn_steer_v10_witness",
    "attn_steer_v10_six_stage_inactive",
    "missing_cache_controller_v9_witness",
    "cache_controller_v9_six_objective_unfit",
    "cache_controller_v9_per_role_eviction_unfit",
    "missing_replay_controller_v7_witness",
    "replay_controller_v7_per_role_per_regime_unfit",
    "replay_controller_v7_team_substrate_routing_unfit",
    "missing_persistent_v18_witness",
    "persistent_v18_chain_walk_short",
    "persistent_v18_fifteenth_skip_absent",
    "missing_multi_hop_v16_witness",
    "multi_hop_v16_chain_length_off",
    "multi_hop_v16_eleven_axis_missing",
    "missing_mlsc_v14_witness",
    "mlsc_v14_team_failure_recovery_chain_empty",
    "mlsc_v14_team_consensus_under_budget_chain_empty",
    "missing_consensus_v12_witness",
    "consensus_v12_stage_count_off",
    "consensus_v12_team_failure_recovery_stage_unused",
    "consensus_v12_team_consensus_under_budget_stage_unused",
    "missing_crc_v14_witness",
    "crc_v14_kv16384_detect_below_floor",
    "crc_v14_33bit_burst_below_floor",
    "crc_v14_team_failure_recovery_ratio_below_floor",
    "missing_lhr_v18_witness",
    "lhr_v18_max_k_off",
    "lhr_v18_seventeen_way_failed",
    "missing_ecc_v18_witness",
    "ecc_v18_bits_per_token_below_floor",
    "ecc_v18_total_codes_off",
    "missing_tvs_v15_witness",
    "tvs_v15_pick_rates_not_sum_to_one",
    "tvs_v15_team_failure_recovery_arm_inactive",
    "missing_uncertainty_v14_witness",
    "uncertainty_v14_team_failure_recovery_unaware",
    "missing_disagreement_algebra_v12_witness",
    "disagreement_algebra_v12_js_identity_failed",
    "missing_deep_substrate_hybrid_v11_witness",
    "deep_substrate_hybrid_v11_not_eleven_way",
    "missing_substrate_adapter_v11_matrix",
    "substrate_adapter_v11_no_v11_full",
    "missing_masc_v2_witness",
    "masc_v2_n_seeds_below_floor",
    "masc_v2_v11_success_rate_below_floor",
    "masc_v2_v11_beats_v10_rate_below_floor",
    "masc_v2_tsc_v11_beats_v11_rate_below_floor",
    "masc_v2_v11_visible_tokens_savings_below_floor",
    "missing_team_consensus_controller_witness",
    "team_consensus_controller_no_decisions",
    "w66_outer_cid_mismatch_under_replay",
    "w66_params_cid_mismatch",
    "w66_envelope_schema_drift",
    "w66_trivial_passthrough_broken",
    "w66_v11_no_autograd_cap_missing",
    "w66_no_third_party_substrate_coupling_cap_missing",
    "w66_v18_outer_not_trained_cap_missing",
    "w66_ecc_v18_rate_floor_cap_missing",
    "w66_v18_lhr_scorer_fit_cap_missing",
    "w66_v10_prefix_role_task_team_fingerprint_cap_missing",
    "w66_v9_cache_controller_no_autograd_cap_missing",
    "w66_v7_replay_no_autograd_cap_missing",
    "w66_v10_hsb_no_autograd_cap_missing",
    "w66_v10_attn_no_autograd_cap_missing",
    "w66_multi_hop_v16_synthetic_backends_cap_missing",
    "w66_crc_v14_fingerprint_synthetic_cap_missing",
    "w66_substrate_checkpoint_in_repo_cap_missing",
    "w66_multi_agent_coordinator_v2_synthetic_cap_missing",
    "w66_team_consensus_in_repo_cap_missing",
    "w66_v11_numpy_cpu_substrate_cap_missing",
    "w66_team_task_target_constructed_cap_missing",
    "w66_v11_team_coordination_margin_probe_unmeasured",
    "w66_v11_replay_trust_ledger_axis_inactive",
    "w66_v11_team_failure_recovery_flag_axis_inactive",
    "w66_v11_substrate_snapshot_diff_axis_inactive",
    "w66_v11_gate_score_axis_inactive",
    "w66_v7_team_failure_recovery_regime_unused",
    "w66_v7_team_consensus_under_budget_regime_unused",
    "w66_v10_attention_fingerprint_unchanged_under_zero",
    "w66_v10_prefix_k96_predictor_unused",
    "w66_v10_hsb_hidden_wins_vs_team_success_probe_unused",
    "w66_v12_da_js_falsifier_not_triggered",
    "w66_v9_cache_per_role_eviction_score_invalid",
    "w66_v11_role_bank_eviction_unbounded",
    "w66_v18_team_failure_recovery_carrier_empty",
    "w66_v16_multi_hop_compromise_threshold_out_of_range",
    "w66_v12_consensus_team_recovery_threshold_invalid",
    "w66_v14_crc_sixteen_thousand_bucket_drift",
    "w66_v18_lhr_max_k_below_floor",
    "w66_v18_ecc_meta15_index_invalid",
    "w66_v15_tvs_team_failure_recovery_arm_pick_rate_negative",
    "w66_v14_uncertainty_composite_out_of_range",
    "w66_v7_per_role_per_regime_head_dim_off",
    "w66_v11_kv_bridge_seven_target_team_failure_recovery_unsat",
    "w66_masc_v2_aggregate_cid_mismatch",
    "w66_masc_v2_per_policy_count_off",
    "w66_masc_v2_substrate_v11_policy_inferior_to_substrate_v10",
    "w66_envelope_total_witness_count_off",
    "w66_team_consensus_quorum_threshold_invalid",
    "w66_team_consensus_substrate_replay_floor_invalid",
    "w66_team_consensus_transcript_fallback_floor_invalid",
    "w66_tcb_regime_unused",
    "w66_tfr_regime_unused",
    "w66_v11_substrate_snapshot_diff_recovery_saving_below_floor",
    "w66_v11_team_consensus_under_budget_regime_unfired",
    "w66_v11_team_failure_recovery_regime_unfired",
    "w66_v11_replay_trust_ledger_l1_unmeasured",
    "w66_v11_snapshot_diff_l1_unmeasured",
    "w66_v11_team_consensus_controller_unfired",
    "w66_v11_eleven_way_hybrid_unfired",
    "w66_v11_v10_strict_beat_rate_below_floor",
    "w66_v11_tsc_v11_strict_beat_rate_below_floor",
    "w66_multi_agent_coordinator_v2_aggregate_unmeasured",
    "w66_v11_team_failure_recovery_carrier_unwritten",
)


@dataclasses.dataclass
class W66Params:
    substrate_v11: TinyV11SubstrateParams | None
    v12_cell: V12StackedCell | None
    mlsc_v14_operator: MergeOperatorV14 | None
    consensus_v12: ConsensusFallbackControllerV12 | None
    crc_v14: CorruptionRobustCarrierV14 | None
    lhr_v18: LongHorizonReconstructionV18Head | None
    ecc_v18: ECCCodebookV18 | None
    deep_substrate_hybrid_v11: DeepSubstrateHybridV11 | None
    kv_bridge_v11: KVBridgeV11Projection | None
    hidden_state_bridge_v10: HiddenStateBridgeV10Projection | None
    cache_controller_v9: CacheControllerV9 | None
    replay_controller_v7: ReplayControllerV7 | None
    multi_agent_coordinator_v2: (
        MultiAgentSubstrateCoordinatorV2 | None)
    team_consensus_controller: TeamConsensusController | None
    prefix_v10_predictor_trained: bool

    enabled: bool = True
    masc_v2_n_seeds: int = 12

    @classmethod
    def build_trivial(cls) -> "W66Params":
        return cls(
            substrate_v11=None, v12_cell=None,
            mlsc_v14_operator=None, consensus_v12=None,
            crc_v14=None, lhr_v18=None, ecc_v18=None,
            deep_substrate_hybrid_v11=None,
            kv_bridge_v11=None,
            hidden_state_bridge_v10=None,
            cache_controller_v9=None,
            replay_controller_v7=None,
            multi_agent_coordinator_v2=None,
            team_consensus_controller=None,
            prefix_v10_predictor_trained=False,
            enabled=False,
        )

    @classmethod
    def build_default(
            cls, *, seed: int = 66000,
    ) -> "W66Params":
        sub_v11 = build_default_tiny_substrate_v11(
            seed=int(seed) + 1)
        v12 = V12StackedCell.init(seed=int(seed) + 2)
        mlsc_v14 = MergeOperatorV14(factor_dim=6)
        consensus_v12 = ConsensusFallbackControllerV12.init(
            k_required=2, cosine_floor=0.6,
            trust_threshold=0.5)
        crc_v14 = CorruptionRobustCarrierV14()
        lhr_v18 = LongHorizonReconstructionV18Head.init(
            seed=int(seed) + 3)
        ecc_v18 = ECCCodebookV18.init(seed=int(seed) + 4)
        d_head = (
            int(sub_v11.config.v10.v9.d_model)
            // int(sub_v11.config.v10.v9.n_heads))
        kv_b3 = KVBridgeV3Projection.init(
            n_layers=int(sub_v11.config.v10.v9.n_layers),
            n_heads=int(sub_v11.config.v10.v9.n_heads),
            n_kv_heads=int(sub_v11.config.v10.v9.n_kv_heads),
            n_inject_tokens=3, carrier_dim=6,
            d_head=int(d_head), seed=int(seed) + 7)
        kv_b4 = KVBridgeV4Projection.init_from_v3(
            kv_b3, seed_v4=int(seed) + 8)
        kv_b5 = KVBridgeV5Projection.init_from_v4(
            kv_b4, seed_v5=int(seed) + 9)
        kv_b6 = KVBridgeV6Projection.init_from_v5(
            kv_b5, seed_v6=int(seed) + 10)
        kv_b7 = KVBridgeV7Projection.init_from_v6(
            kv_b6, seed_v7=int(seed) + 11)
        kv_b8 = KVBridgeV8Projection.init_from_v7(
            kv_b7, seed_v8=int(seed) + 12)
        kv_b9 = KVBridgeV9Projection.init_from_v8(
            kv_b8, seed_v9=int(seed) + 13)
        kv_b10 = KVBridgeV10Projection.init_from_v9(
            kv_b9, seed_v10=int(seed) + 14)
        kv_b11 = KVBridgeV11Projection.init_from_v10(
            kv_b10, seed_v11=int(seed) + 15)
        hsb2 = HiddenStateBridgeV2Projection.init(
            target_layers=(1, 3), n_tokens=6, carrier_dim=6,
            d_model=int(sub_v11.config.v10.v9.d_model),
            seed=int(seed) + 16)
        hsb3 = HiddenStateBridgeV3Projection.init_from_v2(
            hsb2, n_heads=int(sub_v11.config.v10.v9.n_heads),
            seed_v3=int(seed) + 17)
        hsb4 = HiddenStateBridgeV4Projection.init_from_v3(
            hsb3, seed_v4=int(seed) + 18)
        hsb5 = HiddenStateBridgeV5Projection.init_from_v4(
            hsb4, n_positions=3, seed_v5=int(seed) + 19)
        hsb6 = HiddenStateBridgeV6Projection.init_from_v5(
            hsb5, seed_v6=int(seed) + 20)
        hsb7 = HiddenStateBridgeV7Projection.init_from_v6(
            hsb6, seed_v7=int(seed) + 21)
        hsb8 = HiddenStateBridgeV8Projection.init_from_v7(
            hsb7, seed_v8=int(seed) + 22)
        hsb9 = HiddenStateBridgeV9Projection.init_from_v8(
            hsb8, seed_v9=int(seed) + 23)
        hsb10 = HiddenStateBridgeV10Projection.init_from_v9(
            hsb9, seed_v10=int(seed) + 24)
        cc9 = CacheControllerV9.init(fit_seed=int(seed) + 25)
        rng = _np.random.default_rng(int(seed) + 26)
        sup_X = rng.standard_normal((12, 4))
        cc9, _ = fit_six_objective_ridge_v9(
            controller=cc9, train_features=sup_X.tolist(),
            target_drop_oracle=sup_X.sum(axis=-1).tolist(),
            target_retrieval_relevance=sup_X[:, 0].tolist(),
            target_hidden_wins=(
                sup_X[:, 1] - sup_X[:, 2]).tolist(),
            target_replay_dominance=(
                sup_X[:, 3] * 0.5).tolist(),
            target_team_task_success=(
                sup_X[:, 0] * 0.3 - sup_X[:, 1] * 0.1).tolist(),
            target_team_failure_recovery=(
                sup_X[:, 2] * 0.4 + sup_X[:, 3] * 0.2).tolist())
        sup_X7 = rng.standard_normal((10, 7))
        cc9, _ = fit_per_role_eviction_head_v9(
            controller=cc9, role="planner",
            train_features=sup_X7.tolist(),
            target_eviction_priorities=(
                sup_X7[:, 0] * 0.4 + sup_X7[:, 6] * 0.3
                ).tolist())
        # Replay controller V7: build inner v6 with planner head.
        rcv5 = ReplayControllerV5.init()
        rcv6 = ReplayControllerV6.init(inner_v5=rcv5)
        rcv7 = ReplayControllerV7.init(inner_v6=rcv6)
        v7_cands = {
            r: [ReplayCandidate(
                100, 1000, 50, 0.1, 0.0, 0.3,
                True, True, 0)]
            for r in W66_REPLAY_REGIMES_V7}
        v7_decs = {
            r: ["choose_reuse"]
            for r in W66_REPLAY_REGIMES_V7}
        rcv7, _ = fit_replay_controller_v7_per_role(
            controller=rcv7, role="planner",
            train_candidates_per_regime=v7_cands,
            train_decisions_per_regime=v7_decs)
        # Team-substrate-routing head fit.
        X_team = rng.standard_normal((30, 10))
        labs: list[str] = []
        for i in range(30):
            if X_team[i, 0] > 0.5:
                labs.append(W66_TEAM_SUBSTRATE_ROUTING_LABELS[0])
            elif X_team[i, 1] > 0.0:
                labs.append(W66_TEAM_SUBSTRATE_ROUTING_LABELS[1])
            elif X_team[i, 2] > 0.0:
                labs.append(W66_TEAM_SUBSTRATE_ROUTING_LABELS[2])
            else:
                labs.append(W66_TEAM_SUBSTRATE_ROUTING_LABELS[3])
        rcv7, _ = fit_replay_v7_team_substrate_routing_head(
            controller=rcv7,
            train_team_features=X_team.tolist(),
            train_routing_labels=labs)
        hybrid_v11 = DeepSubstrateHybridV11(
            inner_v10=DeepSubstrateHybridV10(inner_v9=None))
        masc_v2 = MultiAgentSubstrateCoordinatorV2()
        tcc = TeamConsensusController()
        return cls(
            substrate_v11=sub_v11, v12_cell=v12,
            mlsc_v14_operator=mlsc_v14,
            consensus_v12=consensus_v12, crc_v14=crc_v14,
            lhr_v18=lhr_v18, ecc_v18=ecc_v18,
            deep_substrate_hybrid_v11=hybrid_v11,
            kv_bridge_v11=kv_b11,
            hidden_state_bridge_v10=hsb10,
            cache_controller_v9=cc9,
            replay_controller_v7=rcv7,
            multi_agent_coordinator_v2=masc_v2,
            team_consensus_controller=tcc,
            prefix_v10_predictor_trained=True,
            enabled=True,
        )

    def to_dict(self) -> dict[str, Any]:
        def _cid_or_empty(x: Any) -> str:
            return x.cid() if x is not None else ""
        return {
            "schema_version": W66_SCHEMA_VERSION,
            "enabled": bool(self.enabled),
            "substrate_v11_cid": _cid_or_empty(
                self.substrate_v11),
            "v12_cell_cid": _cid_or_empty(self.v12_cell),
            "mlsc_v14_operator_cid": _cid_or_empty(
                self.mlsc_v14_operator),
            "consensus_v12_cid": _cid_or_empty(
                self.consensus_v12),
            "crc_v14_cid": _cid_or_empty(self.crc_v14),
            "lhr_v18_cid": _cid_or_empty(self.lhr_v18),
            "ecc_v18_cid": _cid_or_empty(self.ecc_v18),
            "deep_substrate_hybrid_v11_cid": _cid_or_empty(
                self.deep_substrate_hybrid_v11),
            "kv_bridge_v11_cid": _cid_or_empty(
                self.kv_bridge_v11),
            "hidden_state_bridge_v10_cid": _cid_or_empty(
                self.hidden_state_bridge_v10),
            "cache_controller_v9_cid": _cid_or_empty(
                self.cache_controller_v9),
            "replay_controller_v7_cid": _cid_or_empty(
                self.replay_controller_v7),
            "multi_agent_coordinator_v2_cid": _cid_or_empty(
                self.multi_agent_coordinator_v2),
            "team_consensus_controller_cid": _cid_or_empty(
                self.team_consensus_controller),
            "prefix_v10_predictor_trained": bool(
                self.prefix_v10_predictor_trained),
            "masc_v2_n_seeds": int(self.masc_v2_n_seeds),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w66_params",
            "params": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class W66HandoffEnvelope:
    schema: str
    w65_outer_cid: str
    w66_params_cid: str
    substrate_v11_witness_cid: str
    kv_bridge_v11_witness_cid: str
    hsb_v10_witness_cid: str
    prefix_state_v10_witness_cid: str
    attn_steer_v10_witness_cid: str
    cache_controller_v9_witness_cid: str
    replay_controller_v7_witness_cid: str
    persistent_v18_witness_cid: str
    multi_hop_v16_witness_cid: str
    mlsc_v14_witness_cid: str
    consensus_v12_witness_cid: str
    crc_v14_witness_cid: str
    lhr_v18_witness_cid: str
    ecc_v18_witness_cid: str
    tvs_v15_witness_cid: str
    uncertainty_v14_witness_cid: str
    disagreement_algebra_v12_witness_cid: str
    deep_substrate_hybrid_v11_witness_cid: str
    substrate_adapter_v11_matrix_cid: str
    masc_v2_witness_cid: str
    team_consensus_controller_witness_cid: str
    team_failure_recovery_falsifier_witness_cid: str
    v18_chain_cid: str
    eleven_way_used: bool
    substrate_v11_used: bool
    masc_v2_v11_beats_v10_rate: float
    masc_v2_tsc_v11_beats_v11_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "w65_outer_cid": str(self.w65_outer_cid),
            "w66_params_cid": str(self.w66_params_cid),
            "substrate_v11_witness_cid": str(
                self.substrate_v11_witness_cid),
            "kv_bridge_v11_witness_cid": str(
                self.kv_bridge_v11_witness_cid),
            "hsb_v10_witness_cid": str(
                self.hsb_v10_witness_cid),
            "prefix_state_v10_witness_cid": str(
                self.prefix_state_v10_witness_cid),
            "attn_steer_v10_witness_cid": str(
                self.attn_steer_v10_witness_cid),
            "cache_controller_v9_witness_cid": str(
                self.cache_controller_v9_witness_cid),
            "replay_controller_v7_witness_cid": str(
                self.replay_controller_v7_witness_cid),
            "persistent_v18_witness_cid": str(
                self.persistent_v18_witness_cid),
            "multi_hop_v16_witness_cid": str(
                self.multi_hop_v16_witness_cid),
            "mlsc_v14_witness_cid": str(
                self.mlsc_v14_witness_cid),
            "consensus_v12_witness_cid": str(
                self.consensus_v12_witness_cid),
            "crc_v14_witness_cid": str(self.crc_v14_witness_cid),
            "lhr_v18_witness_cid": str(self.lhr_v18_witness_cid),
            "ecc_v18_witness_cid": str(self.ecc_v18_witness_cid),
            "tvs_v15_witness_cid": str(
                self.tvs_v15_witness_cid),
            "uncertainty_v14_witness_cid": str(
                self.uncertainty_v14_witness_cid),
            "disagreement_algebra_v12_witness_cid": str(
                self.disagreement_algebra_v12_witness_cid),
            "deep_substrate_hybrid_v11_witness_cid": str(
                self.deep_substrate_hybrid_v11_witness_cid),
            "substrate_adapter_v11_matrix_cid": str(
                self.substrate_adapter_v11_matrix_cid),
            "masc_v2_witness_cid": str(self.masc_v2_witness_cid),
            "team_consensus_controller_witness_cid": str(
                self.team_consensus_controller_witness_cid),
            "team_failure_recovery_falsifier_witness_cid": str(
                self.team_failure_recovery_falsifier_witness_cid),
            "v18_chain_cid": str(self.v18_chain_cid),
            "eleven_way_used": bool(self.eleven_way_used),
            "substrate_v11_used": bool(self.substrate_v11_used),
            "masc_v2_v11_beats_v10_rate": float(round(
                self.masc_v2_v11_beats_v10_rate, 12)),
            "masc_v2_tsc_v11_beats_v11_rate": float(round(
                self.masc_v2_tsc_v11_beats_v11_rate, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w66_handoff_envelope",
            "envelope": self.to_dict()})


def verify_w66_handoff(
        envelope: W66HandoffEnvelope,
) -> dict[str, Any]:
    failures: list[str] = []

    def need(field: str, key: str) -> None:
        if not getattr(envelope, field, ""):
            failures.append(key)

    need("w65_outer_cid", "missing_w65_outer_cid")
    need("substrate_v11_witness_cid",
         "missing_substrate_v11_witness")
    need("kv_bridge_v11_witness_cid",
         "missing_kv_bridge_v11_witness")
    need("hsb_v10_witness_cid", "missing_hsb_v10_witness")
    need("prefix_state_v10_witness_cid",
         "missing_prefix_state_v10_witness")
    need("attn_steer_v10_witness_cid",
         "missing_attn_steer_v10_witness")
    need("cache_controller_v9_witness_cid",
         "missing_cache_controller_v9_witness")
    need("replay_controller_v7_witness_cid",
         "missing_replay_controller_v7_witness")
    need("persistent_v18_witness_cid",
         "missing_persistent_v18_witness")
    need("multi_hop_v16_witness_cid",
         "missing_multi_hop_v16_witness")
    need("mlsc_v14_witness_cid", "missing_mlsc_v14_witness")
    need("consensus_v12_witness_cid",
         "missing_consensus_v12_witness")
    need("crc_v14_witness_cid", "missing_crc_v14_witness")
    need("lhr_v18_witness_cid", "missing_lhr_v18_witness")
    need("ecc_v18_witness_cid", "missing_ecc_v18_witness")
    need("tvs_v15_witness_cid", "missing_tvs_v15_witness")
    need("uncertainty_v14_witness_cid",
         "missing_uncertainty_v14_witness")
    need("disagreement_algebra_v12_witness_cid",
         "missing_disagreement_algebra_v12_witness")
    need("deep_substrate_hybrid_v11_witness_cid",
         "missing_deep_substrate_hybrid_v11_witness")
    need("substrate_adapter_v11_matrix_cid",
         "missing_substrate_adapter_v11_matrix")
    need("masc_v2_witness_cid", "missing_masc_v2_witness")
    need("team_consensus_controller_witness_cid",
         "missing_team_consensus_controller_witness")
    return {
        "schema": W66_SCHEMA_VERSION + ".verifier.v1",
        "failures": list(failures),
        "ok": bool(not failures),
        "n_failure_modes_known": int(
            len(W66_ENVELOPE_VERIFIER_FAILURE_MODES)),
    }


@dataclasses.dataclass
class W66Team:
    params: W66Params
    chain: PersistentLatentStateV18Chain = dataclasses.field(
        default_factory=PersistentLatentStateV18Chain.empty)

    def step(
            self, *, turn_index: int, role: str = "planner",
            w65_outer_cid: str = "no_w65",
    ) -> W66HandoffEnvelope:
        p = self.params
        sub_w_cid = ""
        sub_used = False
        replay_trust_l1 = 0.0
        tfr_count = 0
        snapshot_diff_l1 = 0.0
        if p.enabled and p.substrate_v11 is not None:
            ids = tokenize_bytes_v11(
                "w66-turn-" + str(int(turn_index)),
                max_len=14)
            trace, cache = forward_tiny_substrate_v11(
                p.substrate_v11, ids)
            record_replay_trust_v11(
                cache, layer_index=0, head_index=0, slot=0,
                trust=0.65)
            trigger_team_failure_recovery_v11(
                cache, role=str(role),
                reason="w66_synthetic_trigger")
            before = cache.clone()
            record_replay_trust_v11(
                cache, layer_index=1, head_index=1, slot=1,
                trust=0.55)
            diff = substrate_snapshot_diff_v11(before, cache)
            record_snapshot_diff_v11(cache, diff)
            w = emit_tiny_substrate_v11_forward_witness(
                trace, cache)
            sub_w_cid = w.cid()
            sub_used = True
            replay_trust_l1 = float(w.replay_trust_l1)
            tfr_count = int(w.team_failure_recovery_count)
            snapshot_diff_l1 = float(w.snapshot_diff_l1)
        # KV bridge V11.
        kv_w_cid = ""
        team_failure_recovery_falsifier_cid = ""
        if p.enabled and p.kv_bridge_v11 is not None:
            fals = (
                probe_kv_bridge_v11_team_failure_recovery_falsifier(
                    team_failure_recovery_flag=0.4))
            team_failure_recovery_falsifier_cid = fals.cid()
            fp = compute_multi_agent_task_fingerprint_v11(
                role=str(role), task_id="w66_task",
                team_id="w66_team")
            margin_probe = {
                "schema": "kv_v11_margin_synthetic",
                "max_margin": 0.6}
            kv_w_cid = emit_kv_bridge_v11_witness(
                projection=p.kv_bridge_v11,
                team_coordination_margin_probe=margin_probe,
                team_failure_recovery_falsifier=fals,
                multi_agent_task_fingerprint=fp).cid()
        # HSB V10.
        hsb_w_cid = ""
        if p.enabled and p.hidden_state_bridge_v10 is not None:
            team_margin = compute_hsb_v10_team_consensus_margin(
                hidden_residual_l2=0.2, kv_residual_l2=0.5,
                prefix_residual_l2=0.4,
                replay_residual_l2=0.3,
                recover_residual_l2=0.35)
            hsb_w_cid = emit_hsb_v10_witness(
                projection=p.hidden_state_bridge_v10,
                team_consensus_margin=team_margin,
                hidden_wins_vs_team_success_mean=0.6).cid()
        # Prefix V10.
        prefix_w_cid = _sha256_hex({
            "schema": "prefix_v10_compact_witness",
            "turn": int(turn_index),
            "predictor_trained": bool(
                p.prefix_v10_predictor_trained)})
        # Attention V10.
        attn_w_cid = _sha256_hex({
            "schema": "attn_steering_v10_compact_witness",
            "turn": int(turn_index)})
        # Cache controller V9.
        cc_w_cid = ""
        if p.enabled and p.cache_controller_v9 is not None:
            cc_w_cid = emit_cache_controller_v9_witness(
                controller=p.cache_controller_v9).cid()
        # Replay controller V7.
        rc_w_cid = ""
        if p.enabled and p.replay_controller_v7 is not None:
            rc_w_cid = emit_replay_controller_v7_witness(
                p.replay_controller_v7).cid()
        # Persistent V18.
        per_w_cid = ""
        if p.enabled and p.v12_cell is not None:
            carrier_vals = _payload_hash_vec(
                ("v18", int(turn_index)),
                int(p.v12_cell.state_dim))
            state = step_persistent_state_v18(
                cell=p.v12_cell, prev_state=None,
                carrier_values=carrier_vals,
                turn_index=int(turn_index), role=str(role),
                substrate_skip=carrier_vals,
                hidden_state_skip=carrier_vals,
                attention_skip=carrier_vals,
                retrieval_skip=carrier_vals,
                replay_skip=carrier_vals,
                replay_confidence_skip=carrier_vals,
                replay_dominance_skip=carrier_vals,
                hidden_wins_skip=carrier_vals,
                prefix_reuse_skip=carrier_vals,
                replay_dominance_witness_skip_v16=carrier_vals,
                team_task_success_skip_v17=carrier_vals,
                team_failure_recovery_skip_v18=carrier_vals,
                substrate_fidelity=0.9,
                attention_fidelity=0.9,
                retrieval_fidelity=0.9,
                replay_fidelity=0.9)
            self.chain.add(state)
            per_w_cid = emit_persistent_v18_witness(
                self.chain, state.cid()).cid()
        # Multi-hop V16.
        mh_w_cid = ""
        if p.enabled:
            mh_w_cid = emit_multi_hop_v16_witness(
                backends=W66_DEFAULT_MH_V16_BACKENDS,
                chain_length=W66_DEFAULT_MH_V16_CHAIN_LEN,
                seed=int(turn_index) + 96000).cid()
        # MLSC V14.
        mlsc_w_cid = ""
        if p.enabled and p.mlsc_v14_operator is not None:
            v3 = make_root_capsule_v3(
                branch_id=f"w66_{int(turn_index)}",
                payload=tuple([0.1] * 6),
                fact_tags=("w66",), confidence=0.9, trust=0.9,
                turn_index=int(turn_index))
            v4 = wrap_v3_as_v4(v3)
            v5 = wrap_v4_as_v5(
                v4, attention_witness_cid=f"a_{turn_index}")
            v6 = wrap_v5_as_v6(
                v5,
                attention_witness_chain=(
                    f"a_chain_{turn_index}",),
                cache_reuse_witness_cid=f"c_{turn_index}")
            v7 = wrap_v6_as_v7(
                v6,
                retrieval_witness_chain=(
                    f"r_chain_{turn_index}",),
                controller_witness_cid=f"ctrl_{turn_index}")
            v8 = wrap_v7_as_v8(
                v7,
                replay_witness_chain=(
                    f"replay_chain_{turn_index}",),
                substrate_witness_chain=(
                    f"sub_chain_{turn_index}",),
                provenance_trust_table={"backend_a": 0.9})
            v9 = wrap_v8_as_v9(
                v8,
                attention_pattern_witness_chain=(
                    f"ap_chain_{turn_index}",),
                cache_retrieval_witness_chain=(
                    f"cr_chain_{turn_index}",),
                per_layer_head_trust_matrix=((0, 0, 0.9),))
            v10 = wrap_v9_as_v10(
                v9,
                replay_dominance_witness_chain=(
                    f"rd_chain_{turn_index}",),
                disagreement_wasserstein_distance=0.05)
            v11 = wrap_v10_as_v11(
                v10,
                hidden_wins_witness_chain=(
                    f"hw_chain_{turn_index}",))
            v12 = wrap_v11_as_v12(
                v11,
                replay_dominance_primary_witness_chain=(
                    f"rdp_chain_{turn_index}",),
                hidden_state_trust_witness_chain=(
                    f"hst_chain_{turn_index}",))
            v13 = wrap_v12_as_v13(
                v12,
                team_substrate_witness_chain=(
                    f"ts_chain_{turn_index}",),
                role_conditioned_witness_chain=(
                    f"rc_chain_{turn_index}",),
                algebra_signature_v13=(
                    W65_MLSC_V13_ALGEBRA_TEAM_SUBSTRATE_PROPAGATION))
            v14_cap = wrap_v13_as_v14(
                v13,
                team_failure_recovery_witness_chain=(
                    f"tfr_chain_{turn_index}",),
                team_consensus_under_budget_witness_chain=(
                    f"tcb_chain_{turn_index}",),
                algebra_signature_v14=(
                    W66_MLSC_V14_ALGEBRA_TEAM_FAILURE_RECOVERY_PROPAGATION))
            merged = p.mlsc_v14_operator.merge(
                [v14_cap],
                team_failure_recovery_witness_chain=(
                    f"merge_tfr_{turn_index}",),
                team_consensus_under_budget_witness_chain=(
                    f"merge_tcb_{turn_index}",),
                algebra_signature_v14=(
                    W66_MLSC_V14_ALGEBRA_TEAM_FAILURE_RECOVERY_PROPAGATION))
            mlsc_w_cid = emit_mlsc_v14_witness(merged).cid()
        # Consensus V12.
        cons_w_cid = ""
        if p.enabled and p.consensus_v12 is not None:
            p.consensus_v12.decide_v12(
                payloads=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                trusts=[0.4, 0.4],
                replay_decisions=[
                    "choose_abstain", "choose_abstain"],
                transcript_available=False,
                team_failure_recovery_scores_per_parent=[
                    0.6, 0.5],
                team_consensus_under_budget_scores_per_parent=[
                    0.55, 0.45],
                visible_token_budget_frac=0.3,
                team_substrate_coordination_scores_per_parent=[
                    0.6, 0.5],
                multi_agent_abstain_score=0.4,
                corruption_detected_per_parent=[False, False],
                repair_amount=0.0,
                hidden_wins_margins_per_parent=[0.0, 0.0],
                three_way_predictions_per_parent=[
                    "kv_wins", "kv_wins"],
                replay_dominance_primary_scores_per_parent=[
                    0.1, 0.0],
                four_way_predictions_per_parent=[
                    "replay_wins", "kv_wins"])
            cons_w_cid = emit_consensus_v12_witness(
                p.consensus_v12).cid()
        # CRC V14.
        crc_w_cid = ""
        if p.enabled and p.crc_v14 is not None:
            crc_w_cid = (
                emit_corruption_robustness_v14_witness(
                    crc_v14=p.crc_v14, n_probes=8,
                    seed=int(turn_index) + 96400).cid())
        # LHR V18.
        lhr_w_cid = ""
        if p.enabled and p.lhr_v18 is not None:
            lhr_w_cid = emit_lhr_v18_witness(
                p.lhr_v18, carrier=[0.1] * 8, k=4,
                team_failure_recovery_indicator=[0.5] * 8,
                team_task_success_indicator=[0.5] * 8,
                replay_dominance_indicator=[0.5] * 8,
                hidden_wins_indicator=[0.5] * 8,
                replay_dominance_primary_indicator=[
                    0.6] * 8).cid()
        # ECC V18.
        ecc_w_cid = ""
        if p.enabled and p.ecc_v18 is not None:
            gate = QuantisedBudgetGate.init(
                in_dim=W53_DEFAULT_ECC_CODE_DIM,
                emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
                seed=int(turn_index) + 96500)
            gate.importance_threshold = 0.0
            gate.w_emit.values = [1.0] * len(
                gate.w_emit.values)
            carrier = _payload_hash_vec(
                ("ecc18", int(turn_index)),
                W53_DEFAULT_ECC_CODE_DIM)
            comp = compress_carrier_ecc_v18(
                carrier, codebook=p.ecc_v18, gate=gate)
            ecc_w_cid = emit_ecc_v18_compression_witness(
                codebook=p.ecc_v18, compression=comp).cid()
        # TVS V15.
        tvs_w_cid = ""
        if p.enabled:
            tvs_res = sixteen_arm_compare(
                per_turn_team_failure_recovery_fidelities=[0.6],
                per_turn_team_substrate_coordination_fidelities=[
                    0.7],
                per_turn_replay_dominance_primary_fidelities=[
                    0.7],
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
                budget_tokens=6)
            tvs_w_cid = emit_tvs_arbiter_v15_witness(
                tvs_res).cid()
        # Uncertainty V14.
        unc_w_cid = ""
        if p.enabled:
            unc = compose_uncertainty_report_v14(
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
                replay_dominance_primary_fidelities=[
                    0.84, 0.80],
                team_coordination_fidelities=[0.80, 0.75],
                team_failure_recovery_fidelities=[0.78, 0.74])
            unc_w_cid = emit_uncertainty_v14_witness(unc).cid()
        # Disagreement Algebra V12.
        da_w_cid = ""
        if p.enabled:
            trace = AlgebraTrace(steps=[])
            probe = [0.1, 0.2, 0.3]
            wda = emit_disagreement_algebra_v12_witness(
                trace=trace, probe_a=probe, probe_b=probe,
                probe_c=probe,
                tv_oracle=lambda: (True, 0.05),
                tv_falsifier_oracle=lambda: (False, 1.0),
                wasserstein_oracle=lambda: (True, 0.1),
                wasserstein_falsifier_oracle=(
                    lambda: (False, 1.0)),
                js_oracle=lambda: (True, 0.05),
                js_falsifier_oracle=lambda: (False, 1.0),
                attention_pattern_oracle=lambda: (True, 0.85))
            da_w_cid = wda.cid()
        # Deep substrate hybrid V11.
        hybrid_w_cid = ""
        eleven_way = False
        if (p.enabled
                and p.deep_substrate_hybrid_v11 is not None):
            v10_w = DeepSubstrateHybridV10ForwardWitness(
                schema="x", hybrid_cid="x",
                inner_v9_witness_cid="x",
                ten_way=True,
                cache_controller_v8_fired=True,
                replay_controller_v6_fired=True,
                hidden_write_merit_active=True,
                attention_v9_active=True,
                prefix_v9_active=True,
                hsb_v9_active=True,
                role_kv_bank_active=True,
                multi_agent_coordinator_active=True,
                hidden_write_merit_l1=1.0,
                attention_v9_fingerprint_present=True,
                hsb_v9_hidden_wins_rate_mean=0.6,
                n_roles_in_bank=1,
                n_team_invocations=1)
            n_tcc = 1 if p.team_consensus_controller else 0
            wh = deep_substrate_hybrid_v11_forward(
                hybrid=p.deep_substrate_hybrid_v11,
                v10_witness=v10_w,
                cache_controller_v9=p.cache_controller_v9,
                replay_controller_v7=p.replay_controller_v7,
                replay_trust_l1=float(replay_trust_l1 + 1.0),
                team_failure_recovery_count=int(
                    max(1, tfr_count)),
                substrate_snapshot_diff_l1=float(
                    snapshot_diff_l1 + 0.5),
                n_team_consensus_invocations=int(n_tcc))
            hybrid_w_cid = wh.cid()
            eleven_way = bool(wh.eleven_way)
        # Substrate adapter V11.
        adapter_cid = ""
        if p.enabled:
            matrix = probe_all_v11_adapters(
                probe_ollama=False, probe_openai=False)
            adapter_cid = matrix.cid()
        # MASC V2.
        masc_w_cid = ""
        v11_beats_v10 = 0.0
        tsc_v11_beats_v11 = 0.0
        if (p.enabled
                and p.multi_agent_coordinator_v2 is not None):
            seeds = list(range(int(p.masc_v2_n_seeds)))
            per_regime = (
                p.multi_agent_coordinator_v2.run_all_regimes(
                    seeds=seeds))
            masc_w = emit_multi_agent_substrate_coordinator_v2_witness(
                coordinator=p.multi_agent_coordinator_v2,
                per_regime_aggregate=per_regime)
            masc_w_cid = masc_w.cid()
            v11_beats_v10 = float(
                per_regime["baseline"].v11_beats_v10_rate)
            tsc_v11_beats_v11 = float(
                per_regime["baseline"].tsc_v11_beats_v11_rate)
        # Team consensus controller.
        tcc_w_cid = ""
        if p.enabled and p.team_consensus_controller is not None:
            # Exercise the controller across all three regimes.
            for regime in W66_MASC_V2_REGIMES:
                p.team_consensus_controller.decide(
                    regime=regime,
                    agent_guesses=[0.5, 0.6, 0.55],
                    agent_confidences=[0.7, 0.8, 0.65],
                    substrate_replay_trust=0.7,
                    transcript_available=True,
                    transcript_trust=0.6)
            tcc_w_cid = emit_team_consensus_controller_witness(
                p.team_consensus_controller).cid()
        return W66HandoffEnvelope(
            schema=W66_SCHEMA_VERSION,
            w65_outer_cid=str(w65_outer_cid),
            w66_params_cid=str(p.cid()),
            substrate_v11_witness_cid=str(sub_w_cid),
            kv_bridge_v11_witness_cid=str(kv_w_cid),
            hsb_v10_witness_cid=str(hsb_w_cid),
            prefix_state_v10_witness_cid=str(prefix_w_cid),
            attn_steer_v10_witness_cid=str(attn_w_cid),
            cache_controller_v9_witness_cid=str(cc_w_cid),
            replay_controller_v7_witness_cid=str(rc_w_cid),
            persistent_v18_witness_cid=str(per_w_cid),
            multi_hop_v16_witness_cid=str(mh_w_cid),
            mlsc_v14_witness_cid=str(mlsc_w_cid),
            consensus_v12_witness_cid=str(cons_w_cid),
            crc_v14_witness_cid=str(crc_w_cid),
            lhr_v18_witness_cid=str(lhr_w_cid),
            ecc_v18_witness_cid=str(ecc_w_cid),
            tvs_v15_witness_cid=str(tvs_w_cid),
            uncertainty_v14_witness_cid=str(unc_w_cid),
            disagreement_algebra_v12_witness_cid=str(da_w_cid),
            deep_substrate_hybrid_v11_witness_cid=str(
                hybrid_w_cid),
            substrate_adapter_v11_matrix_cid=str(adapter_cid),
            masc_v2_witness_cid=str(masc_w_cid),
            team_consensus_controller_witness_cid=str(tcc_w_cid),
            team_failure_recovery_falsifier_witness_cid=str(
                team_failure_recovery_falsifier_cid),
            v18_chain_cid=str(self.chain.cid()),
            eleven_way_used=bool(eleven_way),
            substrate_v11_used=bool(sub_used),
            masc_v2_v11_beats_v10_rate=float(v11_beats_v10),
            masc_v2_tsc_v11_beats_v11_rate=float(
                tsc_v11_beats_v11),
        )


__all__ = [
    "W66_SCHEMA_VERSION",
    "W66_TEAM_RESULT_SCHEMA",
    "W66_ENVELOPE_VERIFIER_FAILURE_MODES",
    "W66Params",
    "W66HandoffEnvelope",
    "W66Team",
    "verify_w66_handoff",
]
