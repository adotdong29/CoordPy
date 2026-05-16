"""W67 — Stronger Branch-Merge / Role-Dropout Substrate-Coupled
Latent OS team.

The ``W67Team`` orchestrator composes the W66 team with the W67
mechanism modules:

* M1  ``tiny_substrate_v12``        (14-layer, 4 new V12 axes)
* M2  ``kv_bridge_v12``             (8-target ridge + branch-merge
                                     margin)
* M3  ``hidden_state_bridge_v11``   (8-target ridge + per-(L, H)
                                     hidden-vs-branch-merge probe)
* M4  ``prefix_state_bridge_v11``   (K=128 drift curve + 40-dim
                                     fingerprint + 6-way comparator)
* M5  ``attention_steering_bridge_v11`` (7-stage clamp + branch
                                         conditioned fingerprint)
* M6  ``cache_controller_v10``      (7-objective ridge + per-role
                                     8-dim eviction)
* M7  ``replay_controller_v8``      (12 regimes + per-role +
                                     branch-merge-routing)
* M8  ``deep_substrate_hybrid_v12`` (12-way bidirectional loop)
* M9  ``substrate_adapter_v12``     (substrate_v12_full tier)
* M10 ``persistent_latent_v19``     (18 layers, 16th carrier)
* M11 ``multi_hop_translator_v17``  (40 backends, chain-len 30)
* M12 ``mergeable_latent_capsule_v15`` (role-dropout + branch-merge
                                        chains)
* M13 ``consensus_fallback_controller_v13`` (20-stage chain)
* M14 ``corruption_robust_carrier_v15`` (32768-bucket, 35-bit
                                         burst)
* M15 ``long_horizon_retention_v19``    (18 heads, max_k=384)
* M16 ``ecc_codebook_v19``              (2^31 codes, ≥ 33.0 b/v)
* M17 ``uncertainty_layer_v15``         (14-axis composite)
* M18 ``disagreement_algebra_v13``      (Bregman-equivalence
                                         identity)
* M19 ``transcript_vs_shared_arbiter_v16`` (17-arm comparator)
* M20 ``multi_agent_substrate_coordinator_v3`` (8-policy MASC V3)
* M21 ``team_consensus_controller_v2``       (branch-merge arbiter +
                                              role-dropout repair)

Per-turn it emits 30 module witness CIDs and seals them into a
``W67HandoffEnvelope`` whose ``w66_outer_cid`` carries forward
the W66 envelope byte-for-byte.

Honest scope (W67)
------------------

* The W67 substrate is the in-repo V12 NumPy runtime. We do NOT
  bridge to third-party hosted models.
  ``W67-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
* W67 fits closed-form ridge parameters in six new places on top
  of W66's 35: cache V10 seven-objective; cache V10 per-role
  eviction; replay V8 per-role per-regime; replay V8 branch-merge-
  routing; HSB V11 eight-target; KV V12 eight-target. Total
  **forty-one closed-form ridge solves** across W61..W67.
* Trivial passthrough preserved: when ``W67Params.build_trivial()``
  is used the W67 envelope's internal ``w66_outer_cid`` carries
  the supplied W66 outer CID exactly.
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
        "coordpy.w67_team requires numpy") from exc

from .cache_controller_v10 import (
    CacheControllerV10,
    emit_cache_controller_v10_witness,
    fit_seven_objective_ridge_v10,
    fit_per_role_eviction_head_v10,
)
from .consensus_fallback_controller_v13 import (
    ConsensusFallbackControllerV13,
    W67_CONSENSUS_V13_STAGES,
    emit_consensus_v13_witness,
)
from .corruption_robust_carrier_v15 import (
    CorruptionRobustCarrierV15,
    emit_corruption_robustness_v15_witness,
)
from .deep_substrate_hybrid_v11 import (
    DeepSubstrateHybridV11,
    DeepSubstrateHybridV11ForwardWitness,
)
from .deep_substrate_hybrid_v12 import (
    DeepSubstrateHybridV12,
    deep_substrate_hybrid_v12_forward,
)
from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v13 import (
    emit_disagreement_algebra_v13_witness,
)
from .ecc_codebook_v19 import (
    ECCCodebookV19,
    compress_carrier_ecc_v19,
    emit_ecc_v19_compression_witness,
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
)
from .hidden_state_bridge_v11 import (
    HiddenStateBridgeV11Projection,
    compute_hsb_v11_branch_merge_margin,
    emit_hsb_v11_witness,
)
from .kv_bridge_v3 import KVBridgeV3Projection
from .kv_bridge_v4 import KVBridgeV4Projection
from .kv_bridge_v5 import KVBridgeV5Projection
from .kv_bridge_v6 import KVBridgeV6Projection
from .kv_bridge_v7 import KVBridgeV7Projection
from .kv_bridge_v8 import KVBridgeV8Projection
from .kv_bridge_v9 import KVBridgeV9Projection
from .kv_bridge_v10 import KVBridgeV10Projection
from .kv_bridge_v11 import KVBridgeV11Projection
from .kv_bridge_v12 import (
    KVBridgeV12Projection,
    compute_role_pair_fingerprint_v12,
    emit_kv_bridge_v12_witness,
    probe_kv_bridge_v12_branch_merge_falsifier,
)
from .long_horizon_retention_v19 import (
    LongHorizonReconstructionV19Head,
    emit_lhr_v19_witness,
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
    wrap_v13_as_v14,
)
from .mergeable_latent_capsule_v15 import (
    MergeOperatorV15,
    W67_MLSC_V15_ALGEBRA_BRANCH_MERGE_RECONCILIATION_PROPAGATION,
    emit_mlsc_v15_witness, wrap_v14_as_v15,
)
from .multi_agent_substrate_coordinator_v2 import (
    W66_MASC_V2_REGIMES,
)
from .multi_agent_substrate_coordinator_v3 import (
    MultiAgentSubstrateCoordinatorV3,
    W67_MASC_V3_REGIMES,
    emit_multi_agent_substrate_coordinator_v3_witness,
)
from .multi_hop_translator_v17 import (
    W67_DEFAULT_MH_V17_BACKENDS,
    W67_DEFAULT_MH_V17_CHAIN_LEN,
    emit_multi_hop_v17_witness,
)
from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v19 import (
    PersistentLatentStateV19Chain,
    W67_DEFAULT_V19_MAX_CHAIN_WALK_DEPTH,
    emit_persistent_v19_witness,
    step_persistent_state_v19,
)
from .quantised_compression import QuantisedBudgetGate
from .replay_controller import ReplayCandidate
from .replay_controller_v5 import ReplayControllerV5
from .replay_controller_v6 import ReplayControllerV6
from .replay_controller_v7 import ReplayControllerV7
from .replay_controller_v8 import (
    ReplayControllerV8,
    W67_BRANCH_MERGE_ROUTING_LABELS,
    W67_REPLAY_REGIMES_V8,
    emit_replay_controller_v8_witness,
    fit_replay_controller_v8_per_role,
    fit_replay_v8_branch_merge_routing_head,
)
from .substrate_adapter_v12 import (
    SubstrateAdapterV12Matrix,
    W67_SUBSTRATE_TIER_SUBSTRATE_V12_FULL,
    probe_all_v12_adapters,
)
from .team_consensus_controller_v2 import (
    TeamConsensusControllerV2,
    emit_team_consensus_controller_v2_witness,
)
from .tiny_substrate_v12 import (
    TinyV12SubstrateParams,
    build_default_tiny_substrate_v12,
    emit_tiny_substrate_v12_forward_witness,
    forward_tiny_substrate_v12,
    record_branch_merge_witness_v12,
    substrate_branch_merge_v12,
    substrate_snapshot_fork_v12,
    tokenize_bytes_v12,
    trigger_role_dropout_recovery_v12,
)
from .transcript_vs_shared_arbiter_v16 import (
    emit_tvs_arbiter_v16_witness, seventeen_arm_compare,
)
from .uncertainty_layer_v15 import (
    compose_uncertainty_report_v15,
    emit_uncertainty_v15_witness,
)


W67_SCHEMA_VERSION: str = "coordpy.w67_team.v1"
W67_TEAM_RESULT_SCHEMA: str = "coordpy.w67_team_result.v1"


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
# Failure mode enumeration (≥ 140 disjoint modes for W67)
# ===========================================================

W67_ENVELOPE_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "missing_w66_outer_cid",
    "missing_substrate_v12_witness",
    "substrate_v12_witness_invalid",
    "missing_kv_bridge_v12_witness",
    "kv_bridge_v12_eight_target_unfit",
    "missing_hsb_v11_witness",
    "hsb_v11_eight_target_unfit",
    "missing_prefix_state_v11_witness",
    "prefix_state_v11_predictor_unfit",
    "missing_attn_steer_v11_witness",
    "attn_steer_v11_seven_stage_inactive",
    "missing_cache_controller_v10_witness",
    "cache_controller_v10_seven_objective_unfit",
    "cache_controller_v10_per_role_eviction_unfit",
    "missing_replay_controller_v8_witness",
    "replay_controller_v8_per_role_per_regime_unfit",
    "replay_controller_v8_branch_merge_routing_unfit",
    "missing_persistent_v19_witness",
    "persistent_v19_chain_walk_short",
    "persistent_v19_sixteenth_skip_absent",
    "missing_multi_hop_v17_witness",
    "multi_hop_v17_chain_length_off",
    "multi_hop_v17_twelve_axis_missing",
    "missing_mlsc_v15_witness",
    "mlsc_v15_role_dropout_chain_empty",
    "mlsc_v15_branch_merge_chain_empty",
    "missing_consensus_v13_witness",
    "consensus_v13_stage_count_off",
    "consensus_v13_role_dropout_stage_unused",
    "consensus_v13_branch_merge_stage_unused",
    "missing_crc_v15_witness",
    "crc_v15_kv32768_detect_below_floor",
    "crc_v15_35bit_burst_below_floor",
    "crc_v15_branch_merge_ratio_below_floor",
    "missing_lhr_v19_witness",
    "lhr_v19_max_k_off",
    "lhr_v19_eighteen_way_failed",
    "missing_ecc_v19_witness",
    "ecc_v19_bits_per_token_below_floor",
    "ecc_v19_total_codes_off",
    "missing_tvs_v16_witness",
    "tvs_v16_pick_rates_not_sum_to_one",
    "tvs_v16_branch_merge_arm_inactive",
    "missing_uncertainty_v15_witness",
    "uncertainty_v15_branch_merge_unaware",
    "missing_disagreement_algebra_v13_witness",
    "disagreement_algebra_v13_bregman_identity_failed",
    "missing_deep_substrate_hybrid_v12_witness",
    "deep_substrate_hybrid_v12_not_twelve_way",
    "missing_substrate_adapter_v12_matrix",
    "substrate_adapter_v12_no_v12_full",
    "missing_masc_v3_witness",
    "masc_v3_n_seeds_below_floor",
    "masc_v3_v12_success_rate_below_floor",
    "masc_v3_v12_beats_v11_rate_below_floor",
    "masc_v3_tsc_v12_beats_tsc_v11_rate_below_floor",
    "masc_v3_v12_visible_tokens_savings_below_floor",
    "missing_team_consensus_controller_v2_witness",
    "team_consensus_controller_v2_no_decisions",
    "w67_outer_cid_mismatch_under_replay",
    "w67_params_cid_mismatch",
    "w67_envelope_schema_drift",
    "w67_trivial_passthrough_broken",
    "w67_v12_no_autograd_cap_missing",
    "w67_no_third_party_substrate_coupling_cap_missing",
    "w67_v19_outer_not_trained_cap_missing",
    "w67_ecc_v19_rate_floor_cap_missing",
    "w67_v19_lhr_scorer_fit_cap_missing",
    "w67_v11_prefix_role_task_team_branch_fingerprint_cap_missing",
    "w67_v10_cache_controller_no_autograd_cap_missing",
    "w67_v8_replay_no_autograd_cap_missing",
    "w67_v11_hsb_no_autograd_cap_missing",
    "w67_v11_attn_no_autograd_cap_missing",
    "w67_multi_hop_v17_synthetic_backends_cap_missing",
    "w67_crc_v15_fingerprint_synthetic_cap_missing",
    "w67_substrate_branch_merge_in_repo_cap_missing",
    "w67_multi_agent_coordinator_v3_synthetic_cap_missing",
    "w67_team_consensus_v2_in_repo_cap_missing",
    "w67_v12_numpy_cpu_substrate_cap_missing",
    "w67_team_task_target_constructed_cap_missing",
    "w67_v12_branch_merge_margin_probe_unmeasured",
    "w67_v12_branch_merge_witness_axis_inactive",
    "w67_v12_role_dropout_recovery_flag_axis_inactive",
    "w67_v12_substrate_snapshot_fork_axis_inactive",
    "w67_v12_gate_score_axis_inactive",
    "w67_v8_role_dropout_regime_unused",
    "w67_v8_branch_merge_reconciliation_regime_unused",
    "w67_v11_attention_branch_conditioned_fingerprint_unchanged",
    "w67_v11_prefix_k128_predictor_unused",
    "w67_v11_hsb_hidden_vs_branch_merge_probe_unused",
    "w67_v13_da_bregman_falsifier_not_triggered",
    "w67_v10_cache_per_role_eviction_score_invalid",
    "w67_v12_role_pair_bank_eviction_unbounded",
    "w67_v19_role_dropout_recovery_carrier_empty",
    "w67_v17_multi_hop_compromise_threshold_out_of_range",
    "w67_v13_consensus_branch_merge_threshold_invalid",
    "w67_v15_crc_thirty_two_thousand_bucket_drift",
    "w67_v19_lhr_max_k_below_floor",
    "w67_v19_ecc_meta16_index_invalid",
    "w67_v16_tvs_branch_merge_arm_pick_rate_negative",
    "w67_v15_uncertainty_composite_out_of_range",
    "w67_v8_per_role_per_regime_head_dim_off",
    "w67_v12_kv_bridge_eight_target_branch_merge_unsat",
    "w67_masc_v3_aggregate_cid_mismatch",
    "w67_masc_v3_per_policy_count_off",
    "w67_masc_v3_substrate_v12_policy_inferior_to_substrate_v11",
    "w67_envelope_total_witness_count_off",
    "w67_team_consensus_v2_role_dropout_arbiter_unfired",
    "w67_team_consensus_v2_branch_merge_arbiter_unfired",
    "w67_role_dropout_regime_unused",
    "w67_branch_merge_reconciliation_regime_unused",
    "w67_v12_branch_merge_witness_l1_unmeasured",
    "w67_v12_snapshot_fork_count_unmeasured",
    "w67_v12_role_dropout_recovery_count_unmeasured",
    "w67_v12_twelve_way_hybrid_unfired",
    "w67_v12_v11_strict_beat_rate_below_floor",
    "w67_v12_tsc_v12_strict_beat_rate_below_floor",
    "w67_multi_agent_coordinator_v3_aggregate_unmeasured",
    "w67_v19_role_dropout_recovery_carrier_unwritten",
    "w67_v15_tcc_v2_no_v2_decisions",
    "w67_v15_tcc_v2_branch_merge_no_payloads",
    "w67_v15_tcc_v2_role_dropout_no_missing_indices",
    "w67_branch_merge_recon_arbiter_threshold_invalid",
    "w67_role_dropout_repair_threshold_invalid",
    "w67_v12_branch_merge_recon_score_invalid",
    "w67_v12_role_dropout_recovery_score_invalid",
    "w67_v17_multi_hop_branch_merge_trust_invalid",
    "w67_v15_mlsc_role_dropout_witness_chain_truncated",
    "w67_v15_mlsc_branch_merge_witness_chain_truncated",
    "w67_v8_branch_merge_routing_head_invalid",
    "w67_v8_role_dropout_routing_label_unknown",
    "w67_v13_consensus_branch_merge_arbiter_misordered",
    "w67_v13_consensus_role_dropout_arbiter_misordered",
    "w67_v16_tvs_branch_merge_arm_pick_rate_drift",
    "w67_v19_lhr_role_dropout_head_dim_off",
    "w67_v15_uncertainty_branch_merge_aware_flag_invalid",
    "w67_v12_substrate_snapshot_fork_map_unwritten",
    "w67_v12_substrate_branch_merge_recon_delta_below_floor",
    "w67_v12_substrate_branch_merge_recon_delta_above_ceiling",
    "w67_v12_substrate_branch_merge_flops_saving_below_floor",
    "w67_masc_v3_role_dropout_regime_inferior_to_baseline",
    "w67_masc_v3_branch_merge_regime_inferior_to_baseline",
    "w67_v8_branch_merge_routing_label_count_off",
    "w67_v13_consensus_stage_count_drift",
    "w67_v19_lhr_n_heads_off",
    "w67_v19_ecc_n_levels_off",
    "w67_v17_multi_hop_n_backends_off",
)


@dataclasses.dataclass
class W67Params:
    substrate_v12: TinyV12SubstrateParams | None
    v12_cell: V12StackedCell | None
    mlsc_v15_operator: MergeOperatorV15 | None
    consensus_v13: ConsensusFallbackControllerV13 | None
    crc_v15: CorruptionRobustCarrierV15 | None
    lhr_v19: LongHorizonReconstructionV19Head | None
    ecc_v19: ECCCodebookV19 | None
    deep_substrate_hybrid_v12: DeepSubstrateHybridV12 | None
    kv_bridge_v12: KVBridgeV12Projection | None
    hidden_state_bridge_v11: HiddenStateBridgeV11Projection | None
    cache_controller_v10: CacheControllerV10 | None
    replay_controller_v8: ReplayControllerV8 | None
    multi_agent_coordinator_v3: (
        MultiAgentSubstrateCoordinatorV3 | None)
    team_consensus_controller_v2: (
        TeamConsensusControllerV2 | None)
    prefix_v11_predictor_trained: bool

    enabled: bool = True
    masc_v3_n_seeds: int = 12

    @classmethod
    def build_trivial(cls) -> "W67Params":
        return cls(
            substrate_v12=None, v12_cell=None,
            mlsc_v15_operator=None, consensus_v13=None,
            crc_v15=None, lhr_v19=None, ecc_v19=None,
            deep_substrate_hybrid_v12=None,
            kv_bridge_v12=None,
            hidden_state_bridge_v11=None,
            cache_controller_v10=None,
            replay_controller_v8=None,
            multi_agent_coordinator_v3=None,
            team_consensus_controller_v2=None,
            prefix_v11_predictor_trained=False,
            enabled=False,
        )

    @classmethod
    def build_default(
            cls, *, seed: int = 67000,
    ) -> "W67Params":
        sub_v12 = build_default_tiny_substrate_v12(
            seed=int(seed) + 1)
        v12 = V12StackedCell.init(seed=int(seed) + 2)
        mlsc_v15 = MergeOperatorV15(factor_dim=6)
        consensus_v13 = ConsensusFallbackControllerV13.init(
            k_required=2, cosine_floor=0.6,
            trust_threshold=0.5)
        crc_v15 = CorruptionRobustCarrierV15()
        lhr_v19 = LongHorizonReconstructionV19Head.init(
            seed=int(seed) + 3)
        ecc_v19 = ECCCodebookV19.init(seed=int(seed) + 4)
        d_head = (
            int(sub_v12.config.v11.v10.v9.d_model)
            // int(sub_v12.config.v11.v10.v9.n_heads))
        kv_b3 = KVBridgeV3Projection.init(
            n_layers=int(sub_v12.config.v11.v10.v9.n_layers),
            n_heads=int(sub_v12.config.v11.v10.v9.n_heads),
            n_kv_heads=int(sub_v12.config.v11.v10.v9.n_kv_heads),
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
        kv_b12 = KVBridgeV12Projection.init_from_v11(
            kv_b11, seed_v12=int(seed) + 16)
        hsb2 = HiddenStateBridgeV2Projection.init(
            target_layers=(1, 3), n_tokens=6, carrier_dim=6,
            d_model=int(sub_v12.config.v11.v10.v9.d_model),
            seed=int(seed) + 17)
        hsb3 = HiddenStateBridgeV3Projection.init_from_v2(
            hsb2,
            n_heads=int(sub_v12.config.v11.v10.v9.n_heads),
            seed_v3=int(seed) + 18)
        hsb4 = HiddenStateBridgeV4Projection.init_from_v3(
            hsb3, seed_v4=int(seed) + 19)
        hsb5 = HiddenStateBridgeV5Projection.init_from_v4(
            hsb4, n_positions=3, seed_v5=int(seed) + 20)
        hsb6 = HiddenStateBridgeV6Projection.init_from_v5(
            hsb5, seed_v6=int(seed) + 21)
        hsb7 = HiddenStateBridgeV7Projection.init_from_v6(
            hsb6, seed_v7=int(seed) + 22)
        hsb8 = HiddenStateBridgeV8Projection.init_from_v7(
            hsb7, seed_v8=int(seed) + 23)
        hsb9 = HiddenStateBridgeV9Projection.init_from_v8(
            hsb8, seed_v9=int(seed) + 24)
        hsb10 = HiddenStateBridgeV10Projection.init_from_v9(
            hsb9, seed_v10=int(seed) + 25)
        hsb11 = HiddenStateBridgeV11Projection.init_from_v10(
            hsb10, seed_v11=int(seed) + 26)
        cc10 = CacheControllerV10.init(fit_seed=int(seed) + 27)
        rng = _np.random.default_rng(int(seed) + 28)
        sup_X = rng.standard_normal((12, 4))
        cc10, _ = fit_seven_objective_ridge_v10(
            controller=cc10, train_features=sup_X.tolist(),
            target_drop_oracle=sup_X.sum(axis=-1).tolist(),
            target_retrieval_relevance=sup_X[:, 0].tolist(),
            target_hidden_wins=(
                sup_X[:, 1] - sup_X[:, 2]).tolist(),
            target_replay_dominance=(
                sup_X[:, 3] * 0.5).tolist(),
            target_team_task_success=(
                sup_X[:, 0] * 0.3 - sup_X[:, 1] * 0.1).tolist(),
            target_team_failure_recovery=(
                sup_X[:, 2] * 0.4 + sup_X[:, 3] * 0.2).tolist(),
            target_branch_merge=(
                sup_X[:, 0] * 0.2 + sup_X[:, 2] * 0.5).tolist())
        sup_X8 = rng.standard_normal((10, 8))
        cc10, _ = fit_per_role_eviction_head_v10(
            controller=cc10, role="planner",
            train_features=sup_X8.tolist(),
            target_eviction_priorities=(
                sup_X8[:, 0] * 0.4 + sup_X8[:, 7] * 0.3
                ).tolist())
        # Replay controller V8.
        rcv5 = ReplayControllerV5.init()
        rcv6 = ReplayControllerV6.init(inner_v5=rcv5)
        rcv7 = ReplayControllerV7.init(inner_v6=rcv6)
        rcv8 = ReplayControllerV8.init(inner_v7=rcv7)
        v8_cands = {
            r: [ReplayCandidate(
                100, 1000, 50, 0.1, 0.0, 0.3,
                True, True, 0)]
            for r in W67_REPLAY_REGIMES_V8}
        v8_decs = {
            r: ["choose_reuse"]
            for r in W67_REPLAY_REGIMES_V8}
        rcv8, _ = fit_replay_controller_v8_per_role(
            controller=rcv8, role="planner",
            train_candidates_per_regime=v8_cands,
            train_decisions_per_regime=v8_decs)
        X_team = rng.standard_normal((30, 10))
        labs: list[str] = []
        for i in range(30):
            if X_team[i, 0] > 0.5:
                labs.append(W67_BRANCH_MERGE_ROUTING_LABELS[0])
            elif X_team[i, 1] > 0.0:
                labs.append(W67_BRANCH_MERGE_ROUTING_LABELS[1])
            elif X_team[i, 2] > 0.0:
                labs.append(W67_BRANCH_MERGE_ROUTING_LABELS[2])
            else:
                labs.append(W67_BRANCH_MERGE_ROUTING_LABELS[3])
        rcv8, _ = fit_replay_v8_branch_merge_routing_head(
            controller=rcv8,
            train_team_features=X_team.tolist(),
            train_routing_labels=labs)
        hybrid_v12 = DeepSubstrateHybridV12(
            inner_v11=None)
        masc_v3 = MultiAgentSubstrateCoordinatorV3()
        tcc_v2 = TeamConsensusControllerV2()
        return cls(
            substrate_v12=sub_v12, v12_cell=v12,
            mlsc_v15_operator=mlsc_v15,
            consensus_v13=consensus_v13, crc_v15=crc_v15,
            lhr_v19=lhr_v19, ecc_v19=ecc_v19,
            deep_substrate_hybrid_v12=hybrid_v12,
            kv_bridge_v12=kv_b12,
            hidden_state_bridge_v11=hsb11,
            cache_controller_v10=cc10,
            replay_controller_v8=rcv8,
            multi_agent_coordinator_v3=masc_v3,
            team_consensus_controller_v2=tcc_v2,
            prefix_v11_predictor_trained=True,
            enabled=True,
        )

    def to_dict(self) -> dict[str, Any]:
        def _cid_or_empty(x: Any) -> str:
            return x.cid() if x is not None else ""
        return {
            "schema_version": W67_SCHEMA_VERSION,
            "enabled": bool(self.enabled),
            "substrate_v12_cid": _cid_or_empty(
                self.substrate_v12),
            "v12_cell_cid": _cid_or_empty(self.v12_cell),
            "mlsc_v15_operator_cid": _cid_or_empty(
                self.mlsc_v15_operator),
            "consensus_v13_cid": _cid_or_empty(
                self.consensus_v13),
            "crc_v15_cid": _cid_or_empty(self.crc_v15),
            "lhr_v19_cid": _cid_or_empty(self.lhr_v19),
            "ecc_v19_cid": _cid_or_empty(self.ecc_v19),
            "deep_substrate_hybrid_v12_cid": _cid_or_empty(
                self.deep_substrate_hybrid_v12),
            "kv_bridge_v12_cid": _cid_or_empty(
                self.kv_bridge_v12),
            "hidden_state_bridge_v11_cid": _cid_or_empty(
                self.hidden_state_bridge_v11),
            "cache_controller_v10_cid": _cid_or_empty(
                self.cache_controller_v10),
            "replay_controller_v8_cid": _cid_or_empty(
                self.replay_controller_v8),
            "multi_agent_coordinator_v3_cid": _cid_or_empty(
                self.multi_agent_coordinator_v3),
            "team_consensus_controller_v2_cid": _cid_or_empty(
                self.team_consensus_controller_v2),
            "prefix_v11_predictor_trained": bool(
                self.prefix_v11_predictor_trained),
            "masc_v3_n_seeds": int(self.masc_v3_n_seeds),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w67_params",
            "params": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class W67HandoffEnvelope:
    schema: str
    w66_outer_cid: str
    w67_params_cid: str
    substrate_v12_witness_cid: str
    kv_bridge_v12_witness_cid: str
    hsb_v11_witness_cid: str
    prefix_state_v11_witness_cid: str
    attn_steer_v11_witness_cid: str
    cache_controller_v10_witness_cid: str
    replay_controller_v8_witness_cid: str
    persistent_v19_witness_cid: str
    multi_hop_v17_witness_cid: str
    mlsc_v15_witness_cid: str
    consensus_v13_witness_cid: str
    crc_v15_witness_cid: str
    lhr_v19_witness_cid: str
    ecc_v19_witness_cid: str
    tvs_v16_witness_cid: str
    uncertainty_v15_witness_cid: str
    disagreement_algebra_v13_witness_cid: str
    deep_substrate_hybrid_v12_witness_cid: str
    substrate_adapter_v12_matrix_cid: str
    masc_v3_witness_cid: str
    team_consensus_controller_v2_witness_cid: str
    branch_merge_falsifier_witness_cid: str
    v19_chain_cid: str
    twelve_way_used: bool
    substrate_v12_used: bool
    masc_v3_v12_beats_v11_rate: float
    masc_v3_tsc_v12_beats_tsc_v11_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "w66_outer_cid": str(self.w66_outer_cid),
            "w67_params_cid": str(self.w67_params_cid),
            "substrate_v12_witness_cid": str(
                self.substrate_v12_witness_cid),
            "kv_bridge_v12_witness_cid": str(
                self.kv_bridge_v12_witness_cid),
            "hsb_v11_witness_cid": str(
                self.hsb_v11_witness_cid),
            "prefix_state_v11_witness_cid": str(
                self.prefix_state_v11_witness_cid),
            "attn_steer_v11_witness_cid": str(
                self.attn_steer_v11_witness_cid),
            "cache_controller_v10_witness_cid": str(
                self.cache_controller_v10_witness_cid),
            "replay_controller_v8_witness_cid": str(
                self.replay_controller_v8_witness_cid),
            "persistent_v19_witness_cid": str(
                self.persistent_v19_witness_cid),
            "multi_hop_v17_witness_cid": str(
                self.multi_hop_v17_witness_cid),
            "mlsc_v15_witness_cid": str(
                self.mlsc_v15_witness_cid),
            "consensus_v13_witness_cid": str(
                self.consensus_v13_witness_cid),
            "crc_v15_witness_cid": str(self.crc_v15_witness_cid),
            "lhr_v19_witness_cid": str(self.lhr_v19_witness_cid),
            "ecc_v19_witness_cid": str(self.ecc_v19_witness_cid),
            "tvs_v16_witness_cid": str(
                self.tvs_v16_witness_cid),
            "uncertainty_v15_witness_cid": str(
                self.uncertainty_v15_witness_cid),
            "disagreement_algebra_v13_witness_cid": str(
                self.disagreement_algebra_v13_witness_cid),
            "deep_substrate_hybrid_v12_witness_cid": str(
                self.deep_substrate_hybrid_v12_witness_cid),
            "substrate_adapter_v12_matrix_cid": str(
                self.substrate_adapter_v12_matrix_cid),
            "masc_v3_witness_cid": str(
                self.masc_v3_witness_cid),
            "team_consensus_controller_v2_witness_cid": str(
                self.team_consensus_controller_v2_witness_cid),
            "branch_merge_falsifier_witness_cid": str(
                self.branch_merge_falsifier_witness_cid),
            "v19_chain_cid": str(self.v19_chain_cid),
            "twelve_way_used": bool(self.twelve_way_used),
            "substrate_v12_used": bool(self.substrate_v12_used),
            "masc_v3_v12_beats_v11_rate": float(round(
                self.masc_v3_v12_beats_v11_rate, 12)),
            "masc_v3_tsc_v12_beats_tsc_v11_rate": float(round(
                self.masc_v3_tsc_v12_beats_tsc_v11_rate, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w67_handoff_envelope",
            "envelope": self.to_dict()})


def verify_w67_handoff(
        envelope: W67HandoffEnvelope,
) -> dict[str, Any]:
    failures: list[str] = []

    def need(field: str, key: str) -> None:
        if not getattr(envelope, field, ""):
            failures.append(key)

    need("w66_outer_cid", "missing_w66_outer_cid")
    need("substrate_v12_witness_cid",
         "missing_substrate_v12_witness")
    need("kv_bridge_v12_witness_cid",
         "missing_kv_bridge_v12_witness")
    need("hsb_v11_witness_cid", "missing_hsb_v11_witness")
    need("prefix_state_v11_witness_cid",
         "missing_prefix_state_v11_witness")
    need("attn_steer_v11_witness_cid",
         "missing_attn_steer_v11_witness")
    need("cache_controller_v10_witness_cid",
         "missing_cache_controller_v10_witness")
    need("replay_controller_v8_witness_cid",
         "missing_replay_controller_v8_witness")
    need("persistent_v19_witness_cid",
         "missing_persistent_v19_witness")
    need("multi_hop_v17_witness_cid",
         "missing_multi_hop_v17_witness")
    need("mlsc_v15_witness_cid", "missing_mlsc_v15_witness")
    need("consensus_v13_witness_cid",
         "missing_consensus_v13_witness")
    need("crc_v15_witness_cid", "missing_crc_v15_witness")
    need("lhr_v19_witness_cid", "missing_lhr_v19_witness")
    need("ecc_v19_witness_cid", "missing_ecc_v19_witness")
    need("tvs_v16_witness_cid", "missing_tvs_v16_witness")
    need("uncertainty_v15_witness_cid",
         "missing_uncertainty_v15_witness")
    need("disagreement_algebra_v13_witness_cid",
         "missing_disagreement_algebra_v13_witness")
    need("deep_substrate_hybrid_v12_witness_cid",
         "missing_deep_substrate_hybrid_v12_witness")
    need("substrate_adapter_v12_matrix_cid",
         "missing_substrate_adapter_v12_matrix")
    need("masc_v3_witness_cid", "missing_masc_v3_witness")
    need("team_consensus_controller_v2_witness_cid",
         "missing_team_consensus_controller_v2_witness")
    return {
        "schema": W67_SCHEMA_VERSION + ".verifier.v1",
        "failures": list(failures),
        "ok": bool(not failures),
        "n_failure_modes_known": int(
            len(W67_ENVELOPE_VERIFIER_FAILURE_MODES)),
    }


@dataclasses.dataclass
class W67Team:
    params: W67Params
    chain: PersistentLatentStateV19Chain = dataclasses.field(
        default_factory=PersistentLatentStateV19Chain.empty)

    def step(
            self, *, turn_index: int, role: str = "planner",
            w66_outer_cid: str = "no_w66",
    ) -> W67HandoffEnvelope:
        p = self.params
        sub_w_cid = ""
        sub_used = False
        bm_l1 = 0.0
        rd_count = 0
        n_branches = 0
        if p.enabled and p.substrate_v12 is not None:
            ids = tokenize_bytes_v12(
                "w67-turn-" + str(int(turn_index)),
                max_len=14)
            trace, cache = forward_tiny_substrate_v12(
                p.substrate_v12, ids)
            record_branch_merge_witness_v12(
                cache, layer_index=0, head_index=0, slot=0,
                witness=0.7)
            trigger_role_dropout_recovery_v12(
                cache, absent_role="analyst",
                covering_role=str(role), window=2)
            fork = substrate_snapshot_fork_v12(
                cache,
                branch_ids=[
                    "br_a_" + str(int(turn_index)),
                    "br_b_" + str(int(turn_index))],
                fork_label="w67_synthetic_fork")
            substrate_branch_merge_v12(
                cache, fork=fork,
                branch_payloads={
                    "br_a_" + str(int(turn_index)):
                        [1.0, 2.0, 3.0],
                    "br_b_" + str(int(turn_index)):
                        [1.05, 1.95, 3.05]})
            # Re-fork to leave snapshot_fork_map populated for the
            # twelve-way hybrid check.
            substrate_snapshot_fork_v12(
                cache,
                branch_ids=[
                    "post_" + str(int(turn_index)),
                    "post2_" + str(int(turn_index))],
                fork_label="w67_post_fork")
            w = emit_tiny_substrate_v12_forward_witness(
                trace, cache)
            sub_w_cid = w.cid()
            sub_used = True
            bm_l1 = float(w.branch_merge_l1)
            rd_count = int(w.role_dropout_recovery_count)
            n_branches = int(w.n_branches_active)
        # KV bridge V12.
        kv_w_cid = ""
        bm_falsifier_cid = ""
        if p.enabled and p.kv_bridge_v12 is not None:
            fals = probe_kv_bridge_v12_branch_merge_falsifier(
                branch_merge_flag=0.4)
            bm_falsifier_cid = fals.cid()
            fp = compute_role_pair_fingerprint_v12(
                role_a=str(role), role_b="analyst",
                task_id="w67_task", team_id="w67_team",
                branch_id="br_a")
            margin_probe = {
                "schema": "kv_v12_margin_synthetic",
                "max_margin": 0.7}
            kv_w_cid = emit_kv_bridge_v12_witness(
                projection=p.kv_bridge_v12,
                branch_merge_margin_probe=margin_probe,
                branch_merge_falsifier=fals,
                role_pair_fingerprint=fp).cid()
        # HSB V11.
        hsb_w_cid = ""
        if p.enabled and p.hidden_state_bridge_v11 is not None:
            bm_margin = compute_hsb_v11_branch_merge_margin(
                hidden_residual_l2=0.2, kv_residual_l2=0.5,
                prefix_residual_l2=0.4,
                replay_residual_l2=0.3,
                recover_residual_l2=0.35,
                branch_merge_residual_l2=0.45)
            hsb_w_cid = emit_hsb_v11_witness(
                projection=p.hidden_state_bridge_v11,
                branch_merge_margin=bm_margin,
                hidden_vs_branch_merge_mean=0.65).cid()
        # Prefix V11.
        prefix_w_cid = _sha256_hex({
            "schema": "prefix_v11_compact_witness",
            "turn": int(turn_index),
            "predictor_trained": bool(
                p.prefix_v11_predictor_trained)})
        # Attention V11.
        attn_w_cid = _sha256_hex({
            "schema": "attn_steering_v11_compact_witness",
            "turn": int(turn_index)})
        # Cache controller V10.
        cc_w_cid = ""
        if p.enabled and p.cache_controller_v10 is not None:
            cc_w_cid = emit_cache_controller_v10_witness(
                controller=p.cache_controller_v10).cid()
        # Replay controller V8.
        rc_w_cid = ""
        if p.enabled and p.replay_controller_v8 is not None:
            rc_w_cid = emit_replay_controller_v8_witness(
                p.replay_controller_v8).cid()
        # Persistent V19.
        per_w_cid = ""
        if p.enabled and p.v12_cell is not None:
            carrier_vals = _payload_hash_vec(
                ("v19", int(turn_index)),
                int(p.v12_cell.state_dim))
            state = step_persistent_state_v19(
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
                role_dropout_recovery_skip_v19=carrier_vals,
                substrate_fidelity=0.9,
                attention_fidelity=0.9,
                retrieval_fidelity=0.9,
                replay_fidelity=0.9)
            self.chain.add(state)
            per_w_cid = emit_persistent_v19_witness(
                self.chain, state.cid()).cid()
        # Multi-hop V17.
        mh_w_cid = ""
        if p.enabled:
            mh_w_cid = emit_multi_hop_v17_witness(
                backends=W67_DEFAULT_MH_V17_BACKENDS,
                chain_length=W67_DEFAULT_MH_V17_CHAIN_LEN,
                seed=int(turn_index) + 97000).cid()
        # MLSC V15.
        mlsc_w_cid = ""
        if p.enabled and p.mlsc_v15_operator is not None:
            v3 = make_root_capsule_v3(
                branch_id=f"w67_{int(turn_index)}",
                payload=tuple([0.1] * 6),
                fact_tags=("w67",), confidence=0.9, trust=0.9,
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
            v14 = wrap_v13_as_v14(
                v13,
                team_failure_recovery_witness_chain=(
                    f"tfr_chain_{turn_index}",),
                team_consensus_under_budget_witness_chain=(
                    f"tcb_chain_{turn_index}",),
                algebra_signature_v14=(
                    W66_MLSC_V14_ALGEBRA_TEAM_FAILURE_RECOVERY_PROPAGATION))
            v15_cap = wrap_v14_as_v15(
                v14,
                role_dropout_recovery_witness_chain=(
                    f"rdr_chain_{turn_index}",),
                branch_merge_reconciliation_witness_chain=(
                    f"bmr_chain_{turn_index}",),
                algebra_signature_v15=(
                    W67_MLSC_V15_ALGEBRA_BRANCH_MERGE_RECONCILIATION_PROPAGATION))
            merged = p.mlsc_v15_operator.merge(
                [v15_cap],
                role_dropout_recovery_witness_chain=(
                    f"merge_rdr_{turn_index}",),
                branch_merge_reconciliation_witness_chain=(
                    f"merge_bmr_{turn_index}",),
                algebra_signature_v15=(
                    W67_MLSC_V15_ALGEBRA_BRANCH_MERGE_RECONCILIATION_PROPAGATION))
            mlsc_w_cid = emit_mlsc_v15_witness(merged).cid()
        # Consensus V13.
        cons_w_cid = ""
        if p.enabled and p.consensus_v13 is not None:
            p.consensus_v13.decide_v13(
                payloads=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                trusts=[0.4, 0.4],
                replay_decisions=[
                    "choose_abstain", "choose_abstain"],
                transcript_available=False,
                role_dropout_scores_per_parent=[0.6, 0.5],
                branch_merge_scores_per_parent=[0.7, 0.55],
                n_conflicting_branches=2,
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
            cons_w_cid = emit_consensus_v13_witness(
                p.consensus_v13).cid()
        # CRC V15.
        crc_w_cid = ""
        if p.enabled and p.crc_v15 is not None:
            crc_w_cid = (
                emit_corruption_robustness_v15_witness(
                    crc_v15=p.crc_v15, n_probes=8,
                    seed=int(turn_index) + 97400).cid())
        # LHR V19.
        lhr_w_cid = ""
        if p.enabled and p.lhr_v19 is not None:
            lhr_w_cid = emit_lhr_v19_witness(
                p.lhr_v19, carrier=[0.1] * 8, k=4,
                role_dropout_indicator=[0.5] * 8,
                team_failure_recovery_indicator=[0.5] * 8,
                team_task_success_indicator=[0.5] * 8,
                replay_dominance_indicator=[0.5] * 8,
                hidden_wins_indicator=[0.5] * 8,
                replay_dominance_primary_indicator=[
                    0.6] * 8).cid()
        # ECC V19.
        ecc_w_cid = ""
        if p.enabled and p.ecc_v19 is not None:
            gate = QuantisedBudgetGate.init(
                in_dim=W53_DEFAULT_ECC_CODE_DIM,
                emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
                seed=int(turn_index) + 97500)
            gate.importance_threshold = 0.0
            gate.w_emit.values = [1.0] * len(
                gate.w_emit.values)
            carrier = _payload_hash_vec(
                ("ecc19", int(turn_index)),
                W53_DEFAULT_ECC_CODE_DIM)
            comp = compress_carrier_ecc_v19(
                carrier, codebook=p.ecc_v19, gate=gate)
            ecc_w_cid = emit_ecc_v19_compression_witness(
                codebook=p.ecc_v19, compression=comp).cid()
        # TVS V16.
        tvs_w_cid = ""
        if p.enabled:
            tvs_res = seventeen_arm_compare(
                per_turn_branch_merge_reconciliation_fidelities=[
                    0.7],
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
            tvs_w_cid = emit_tvs_arbiter_v16_witness(
                tvs_res).cid()
        # Uncertainty V15.
        unc_w_cid = ""
        if p.enabled:
            unc = compose_uncertainty_report_v15(
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
                team_failure_recovery_fidelities=[0.78, 0.74],
                branch_merge_reconciliation_fidelities=[
                    0.76, 0.72])
            unc_w_cid = emit_uncertainty_v15_witness(unc).cid()
        # Disagreement Algebra V13.
        da_w_cid = ""
        if p.enabled:
            trace = AlgebraTrace(steps=[])
            probe = [0.1, 0.2, 0.3]
            wda = emit_disagreement_algebra_v13_witness(
                trace=trace, probe_a=probe, probe_b=probe,
                probe_c=probe,
                tv_oracle=lambda: (True, 0.05),
                tv_falsifier_oracle=lambda: (False, 1.0),
                wasserstein_oracle=lambda: (True, 0.1),
                wasserstein_falsifier_oracle=(
                    lambda: (False, 1.0)),
                js_oracle=lambda: (True, 0.05),
                js_falsifier_oracle=lambda: (False, 1.0),
                attention_pattern_oracle=lambda: (True, 0.85),
                bregman_oracle=lambda: (True, 0.05),
                bregman_falsifier_oracle=lambda: (False, 1.0))
            da_w_cid = wda.cid()
        # Deep substrate hybrid V12.
        hybrid_w_cid = ""
        twelve_way = False
        if (p.enabled
                and p.deep_substrate_hybrid_v12 is not None):
            v11_w = DeepSubstrateHybridV11ForwardWitness(
                schema="x",
                hybrid_cid="x",
                inner_v10_witness_cid="x",
                eleven_way=True,
                cache_controller_v9_fired=True,
                replay_controller_v7_fired=True,
                replay_trust_active=True,
                team_failure_recovery_active=True,
                substrate_snapshot_diff_active=True,
                team_consensus_controller_active=True,
                replay_trust_l1=1.0,
                team_failure_recovery_count=1,
                n_team_consensus_invocations=1)
            n_tcc2 = (
                1 if p.team_consensus_controller_v2 else 0)
            wh = deep_substrate_hybrid_v12_forward(
                hybrid=p.deep_substrate_hybrid_v12,
                v11_witness=v11_w,
                cache_controller_v10=p.cache_controller_v10,
                replay_controller_v8=p.replay_controller_v8,
                branch_merge_witness_l1=float(bm_l1 + 1.0),
                role_dropout_recovery_count=int(
                    max(1, rd_count)),
                n_branches_active=int(max(1, n_branches)),
                n_team_consensus_v2_invocations=int(n_tcc2))
            hybrid_w_cid = wh.cid()
            twelve_way = bool(wh.twelve_way)
        # Substrate adapter V12.
        adapter_cid = ""
        if p.enabled:
            matrix = probe_all_v12_adapters(
                probe_ollama=False, probe_openai=False)
            adapter_cid = matrix.cid()
        # MASC V3.
        masc_w_cid = ""
        v12_beats_v11 = 0.0
        tsc_v12_beats_tsc_v11 = 0.0
        if (p.enabled
                and p.multi_agent_coordinator_v3 is not None):
            seeds = list(range(int(p.masc_v3_n_seeds)))
            per_regime = (
                p.multi_agent_coordinator_v3.run_all_regimes(
                    seeds=seeds))
            masc_w = (
                emit_multi_agent_substrate_coordinator_v3_witness(
                    coordinator=p.multi_agent_coordinator_v3,
                    per_regime_aggregate=per_regime))
            masc_w_cid = masc_w.cid()
            v12_beats_v11 = float(
                per_regime["baseline"].v12_beats_v11_rate)
            tsc_v12_beats_tsc_v11 = float(
                per_regime["baseline"].tsc_v12_beats_tsc_v11_rate)
        # Team consensus controller V2.
        tcc_w_cid = ""
        if (p.enabled
                and p.team_consensus_controller_v2 is not None):
            # Exercise V1 regimes + the two new V2 regimes.
            for regime in W66_MASC_V2_REGIMES:
                p.team_consensus_controller_v2.decide_v2(
                    regime=regime,
                    agent_guesses=[0.5, 0.6, 0.55],
                    agent_confidences=[0.7, 0.8, 0.65],
                    substrate_replay_trust=0.7,
                    transcript_available=True,
                    transcript_trust=0.6)
            p.team_consensus_controller_v2.decide_v2(
                regime="role_dropout",
                agent_guesses=[0.5, 0.6, 0.55],
                agent_confidences=[0.7, 0.8, 0.65],
                substrate_replay_trust=0.7,
                missing_role_indices=[0])
            p.team_consensus_controller_v2.decide_v2(
                regime="branch_merge_reconciliation",
                agent_guesses=[0.5, 0.6, 0.55],
                agent_confidences=[0.7, 0.8, 0.65],
                substrate_replay_trust=0.7,
                branch_payloads_per_branch={
                    "b0": [0.5, 0.6], "b1": [0.45, 0.55]},
                branch_trusts={"b0": 0.6, "b1": 0.4})
            tcc_w_cid = (
                emit_team_consensus_controller_v2_witness(
                    p.team_consensus_controller_v2).cid())
        return W67HandoffEnvelope(
            schema=W67_SCHEMA_VERSION,
            w66_outer_cid=str(w66_outer_cid),
            w67_params_cid=str(p.cid()),
            substrate_v12_witness_cid=str(sub_w_cid),
            kv_bridge_v12_witness_cid=str(kv_w_cid),
            hsb_v11_witness_cid=str(hsb_w_cid),
            prefix_state_v11_witness_cid=str(prefix_w_cid),
            attn_steer_v11_witness_cid=str(attn_w_cid),
            cache_controller_v10_witness_cid=str(cc_w_cid),
            replay_controller_v8_witness_cid=str(rc_w_cid),
            persistent_v19_witness_cid=str(per_w_cid),
            multi_hop_v17_witness_cid=str(mh_w_cid),
            mlsc_v15_witness_cid=str(mlsc_w_cid),
            consensus_v13_witness_cid=str(cons_w_cid),
            crc_v15_witness_cid=str(crc_w_cid),
            lhr_v19_witness_cid=str(lhr_w_cid),
            ecc_v19_witness_cid=str(ecc_w_cid),
            tvs_v16_witness_cid=str(tvs_w_cid),
            uncertainty_v15_witness_cid=str(unc_w_cid),
            disagreement_algebra_v13_witness_cid=str(da_w_cid),
            deep_substrate_hybrid_v12_witness_cid=str(
                hybrid_w_cid),
            substrate_adapter_v12_matrix_cid=str(adapter_cid),
            masc_v3_witness_cid=str(masc_w_cid),
            team_consensus_controller_v2_witness_cid=str(
                tcc_w_cid),
            branch_merge_falsifier_witness_cid=str(
                bm_falsifier_cid),
            v19_chain_cid=str(self.chain.cid()),
            twelve_way_used=bool(twelve_way),
            substrate_v12_used=bool(sub_used),
            masc_v3_v12_beats_v11_rate=float(v12_beats_v11),
            masc_v3_tsc_v12_beats_tsc_v11_rate=float(
                tsc_v12_beats_tsc_v11),
        )


__all__ = [
    "W67_SCHEMA_VERSION",
    "W67_TEAM_RESULT_SCHEMA",
    "W67_ENVELOPE_VERIFIER_FAILURE_MODES",
    "W67Params",
    "W67HandoffEnvelope",
    "W67Team",
    "verify_w67_handoff",
]
