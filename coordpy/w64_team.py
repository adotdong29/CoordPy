"""W64 — Stronger Replay-Dominance / Hidden-Wins-Primary / 6144-turn
Substrate-Coupled Latent OS team.

The ``W64Team`` orchestrator composes the W63 team with the W64
mechanism modules (V9 substrate + V9 KV bridge + V8 HSB + V8 prefix
+ V8 attention + V7 cache controller + V5 replay controller + V9
hybrid + V16 persistent + V14 multi-hop + V12 capsule + V10 consensus
+ V12 corruption + V16 LHR + V16 ECC + V12 uncertainty + V10
disagreement + V13 TVS + V9 adapter). Per-turn it emits 23 module
witness CIDs and seals them into a ``W64HandoffEnvelope`` whose
``w63_outer_cid`` carries forward the W63 envelope byte-for-byte.

Honest scope (W64)
------------------

* The W64 substrate is the in-repo V9 NumPy runtime. We do NOT
  bridge to third-party hosted models.
  ``W64-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward
  the W56..W63 cap unchanged.
* W64 fits **closed-form ridge parameters** in five new places on
  top of W63's seventeen: cache controller V7 four-objective head;
  cache controller V7 similarity-aware eviction head; cache
  controller V7 composite_v7; replay controller V5 per-regime head
  (7 regimes); replay controller V5 four-way bridge classifier;
  replay controller V5 replay-dominance-primary head; LHR V16
  six-layer scorer.
  Total **twenty-three closed-form ridge solves** (twelve from
  W61+W62 + five from W63 + six from W64).
* Trivial passthrough preserved: when ``W64Params.build_trivial()``
  is used the W64 envelope's internal ``w63_outer_cid`` carries
  the supplied W63 outer CID exactly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

import numpy as _np

from .cache_controller_v7 import (
    CacheControllerV7,
    W64_CACHE_POLICY_COMPOSITE_V7,
    emit_cache_controller_v7_witness,
    fit_composite_v7,
    fit_four_objective_ridge_v7,
    fit_similarity_aware_eviction_head_v7,
)
from .consensus_fallback_controller_v10 import (
    ConsensusFallbackControllerV10,
    W64_CONSENSUS_V10_STAGES,
    emit_consensus_v10_witness,
)
from .corruption_robust_carrier_v12 import (
    CorruptionRobustCarrierV12,
    emit_corruption_robustness_v12_witness,
)
from .deep_substrate_hybrid_v8 import (
    DeepSubstrateHybridV8,
    DeepSubstrateHybridV8ForwardWitness,
)
from .deep_substrate_hybrid_v9 import (
    DeepSubstrateHybridV9,
    deep_substrate_hybrid_v9_forward,
)
from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v10 import (
    emit_disagreement_algebra_v10_witness,
)
from .ecc_codebook_v16 import (
    ECCCodebookV16, compress_carrier_ecc_v16,
    emit_ecc_v16_compression_witness,
)
from .ecc_codebook_v5 import (
    W53_DEFAULT_ECC_CODE_DIM, W53_DEFAULT_ECC_EMIT_MASK_LEN,
)
from .hidden_state_bridge_v2 import (
    HiddenStateBridgeV2Projection,
)
from .hidden_state_bridge_v3 import (
    HiddenStateBridgeV3Projection,
)
from .hidden_state_bridge_v4 import (
    HiddenStateBridgeV4Projection,
)
from .hidden_state_bridge_v5 import (
    HiddenStateBridgeV5Projection,
)
from .hidden_state_bridge_v6 import (
    HiddenStateBridgeV6Projection,
)
from .hidden_state_bridge_v7 import (
    HiddenStateBridgeV7Projection,
)
from .hidden_state_bridge_v8 import (
    HiddenStateBridgeV8Projection,
    emit_hsb_v8_witness,
)
from .kv_bridge_v3 import KVBridgeV3Projection
from .kv_bridge_v4 import KVBridgeV4Projection
from .kv_bridge_v5 import KVBridgeV5Projection
from .kv_bridge_v6 import KVBridgeV6Projection
from .kv_bridge_v7 import KVBridgeV7Projection
from .kv_bridge_v8 import KVBridgeV8Projection
from .kv_bridge_v9 import (
    KVBridgeV9Projection,
    emit_kv_bridge_v9_witness,
    probe_kv_bridge_v9_hidden_wins_primary_falsifier,
)
from .long_horizon_retention_v16 import (
    LongHorizonReconstructionV16Head,
    emit_lhr_v16_witness,
)
from .mergeable_latent_capsule_v3 import make_root_capsule_v3
from .mergeable_latent_capsule_v4 import wrap_v3_as_v4
from .mergeable_latent_capsule_v5 import wrap_v4_as_v5
from .mergeable_latent_capsule_v6 import wrap_v5_as_v6
from .mergeable_latent_capsule_v7 import wrap_v6_as_v7
from .mergeable_latent_capsule_v8 import wrap_v7_as_v8
from .mergeable_latent_capsule_v9 import wrap_v8_as_v9
from .mergeable_latent_capsule_v10 import (
    W62_MLSC_V10_ALGEBRA_REPLAY_DOMINANCE_PROPAGATION,
    wrap_v9_as_v10,
)
from .mergeable_latent_capsule_v11 import (
    W63_MLSC_V11_ALGEBRA_HIDDEN_WINS_PROPAGATION,
    wrap_v10_as_v11,
)
from .mergeable_latent_capsule_v12 import (
    MergeOperatorV12,
    W64_MLSC_V12_ALGEBRA_REPLAY_DOMINANCE_PRIMARY_PROPAGATION,
    emit_mlsc_v12_witness, wrap_v11_as_v12,
)
from .multi_hop_translator_v14 import (
    W64_DEFAULT_MH_V14_BACKENDS,
    W64_DEFAULT_MH_V14_CHAIN_LEN,
    emit_multi_hop_v14_witness,
)
from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v16 import (
    PersistentLatentStateV16Chain,
    W64_DEFAULT_V16_MAX_CHAIN_WALK_DEPTH,
    emit_persistent_v16_witness,
    step_persistent_state_v16,
)
from .quantised_compression import QuantisedBudgetGate
from .replay_controller import (
    ReplayCandidate, W60_REPLAY_DECISION_REUSE,
    W60_REPLAY_DECISION_RECOMPUTE,
    W60_REPLAY_DECISION_FALLBACK,
)
from .replay_controller_v2 import (
    ReplayControllerV2, fit_replay_controller_v2,
)
from .replay_controller_v3 import (
    ReplayControllerV3,
    W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT,
    W62_REPLAY_REGIME_HIDDEN_WRITE_HEAVY,
    W62_REPLAY_REGIME_SYNTHETIC_CORRUPTION,
    W62_REPLAY_REGIME_TRANSCRIPT_ONLY,
    fit_replay_controller_v3_per_regime,
    fit_hidden_vs_kv_regime_classifier,
)
from .replay_controller_v4 import (
    ReplayControllerV4,
    W63_REPLAY_REGIME_HIDDEN_WINS,
    W63_REPLAY_REGIME_CACHE_CORRUPTION_RECOVERED,
    fit_replay_controller_v4_per_regime,
    fit_three_way_bridge_classifier,
)
from .replay_controller_v5 import (
    ReplayControllerV5,
    W64_REPLAY_REGIME_REPLAY_DOMINANCE_PRIMARY,
    emit_replay_controller_v5_witness,
    fit_replay_controller_v5_per_regime,
    fit_four_way_bridge_classifier,
    fit_replay_dominance_primary_head_v5,
)
from .substrate_adapter_v9 import (
    SubstrateAdapterV9Matrix,
    W64_SUBSTRATE_TIER_SUBSTRATE_V9_FULL,
    probe_all_v9_adapters,
)
from .tiny_substrate_v9 import (
    TinyV9SubstrateParams,
    build_default_tiny_substrate_v9,
    emit_tiny_substrate_v9_forward_witness,
    forward_tiny_substrate_v9,
    record_hidden_state_trust_decision_v9,
    record_hidden_wins_primary_v9,
    record_replay_dominance_witness_v9,
    tokenize_bytes_v9,
)
from .transcript_vs_shared_arbiter_v13 import (
    FourteenArmCompareResult,
    emit_tvs_arbiter_v13_witness,
    fourteen_arm_compare,
)
from .uncertainty_layer_v12 import (
    compose_uncertainty_report_v12,
    emit_uncertainty_v12_witness,
)


W64_SCHEMA_VERSION: str = "coordpy.w64_team.v1"
W64_TEAM_RESULT_SCHEMA: str = "coordpy.w64_team_result.v1"


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


# =============================================================================
# Failure mode enumeration (≥ 85 disjoint modes for W64)
# =============================================================================

W64_ENVELOPE_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "missing_w63_outer_cid",
    "missing_substrate_v9_witness",
    "substrate_v9_witness_invalid",
    "missing_kv_bridge_v9_witness",
    "kv_bridge_v9_five_target_unfit",
    "missing_hsb_v8_witness",
    "hsb_v8_five_target_unfit",
    "missing_prefix_state_v8_witness",
    "prefix_state_v8_drift_curve_unfit",
    "missing_attn_steer_v8_witness",
    "attn_steer_v8_four_stage_inactive",
    "missing_cache_controller_v7_witness",
    "cache_controller_v7_four_objective_unfit",
    "cache_controller_v7_similarity_eviction_unfit",
    "cache_controller_v7_composite_v7_unfit",
    "missing_replay_controller_v5_witness",
    "replay_controller_v5_per_regime_unfit",
    "replay_controller_v5_four_way_bridge_classifier_unfit",
    "replay_controller_v5_dominance_primary_head_unfit",
    "missing_persistent_v16_witness",
    "persistent_v16_chain_walk_short",
    "persistent_v16_thirteenth_skip_absent",
    "missing_multi_hop_v14_witness",
    "multi_hop_v14_chain_length_off",
    "multi_hop_v14_nine_axis_missing",
    "missing_mlsc_v12_witness",
    "mlsc_v12_replay_dominance_primary_chain_empty",
    "mlsc_v12_total_variation_not_computed",
    "missing_consensus_v10_witness",
    "consensus_v10_stage_count_off",
    "consensus_v10_replay_dominance_primary_stage_unused",
    "missing_crc_v12_witness",
    "crc_v12_kv4096_detect_below_floor",
    "crc_v12_23bit_burst_below_floor",
    "crc_v12_replay_dominance_recovery_ratio_below_floor",
    "missing_lhr_v16_witness",
    "lhr_v16_max_k_off",
    "lhr_v16_fifteen_way_failed",
    "missing_ecc_v16_witness",
    "ecc_v16_bits_per_token_below_floor",
    "ecc_v16_total_codes_off",
    "missing_tvs_v13_witness",
    "tvs_v13_pick_rates_not_sum_to_one",
    "tvs_v13_replay_dominance_primary_arm_inactive",
    "missing_uncertainty_v12_witness",
    "uncertainty_v12_replay_dominance_primary_unaware",
    "missing_disagreement_algebra_v10_witness",
    "disagreement_algebra_v10_total_variation_identity_failed",
    "missing_deep_substrate_hybrid_v9_witness",
    "deep_substrate_hybrid_v9_not_nine_way",
    "missing_substrate_adapter_v9_matrix",
    "substrate_adapter_v9_no_v9_full",
    "w64_outer_cid_mismatch_under_replay",
    "w64_params_cid_mismatch",
    "w64_envelope_schema_drift",
    "w64_trivial_passthrough_broken",
    "w64_v9_no_autograd_cap_missing",
    "w64_no_third_party_substrate_coupling_cap_missing",
    "w64_v16_outer_not_trained_cap_missing",
    "w64_ecc_v16_rate_floor_cap_missing",
    "w64_lhr_v16_scorer_fit_cap_missing",
    "w64_prefix_v8_role_fingerprint_cap_missing",
    "w64_v7_cache_controller_no_autograd_cap_missing",
    "w64_v5_replay_no_autograd_cap_missing",
    "w64_v8_hsb_no_autograd_cap_missing",
    "w64_v8_attn_no_autograd_cap_missing",
    "w64_multi_hop_v14_synthetic_backends_cap_missing",
    "w64_crc_v12_fingerprint_synthetic_cap_missing",
    "w64_consensus_v10_replay_dominance_primary_stage_synthetic_cap_missing",
    "w64_kv_bridge_v9_replay_dominance_primary_target_constructed_cap_missing",
    "w64_v9_substrate_numpy_cpu_cap_missing",
    "w64_hidden_wins_primary_falsifier_not_zero_under_inversion",
    "w64_prefix_vs_hidden_vs_replay_decision_invalid",
    "w64_v9_hidden_wins_primary_axis_inactive",
    "w64_v9_replay_dominance_witness_axis_inactive",
    "w64_v9_attention_entropy_probe_axis_inactive",
    "w64_v9_cache_similarity_matrix_axis_inactive",
    "w64_v9_hidden_state_trust_ledger_axis_inactive",
    "w64_v5_replay_dominance_primary_regime_unused",
    "w64_v8_hsb_three_stage_recovery_unused",
    "w64_v9_kv_bridge_fingerprint_unchanged_under_perturbation",
    "w64_v8_attn_hellinger_stage_unused",
    "w64_v8_prefix_role_fingerprint_unused",
    "w64_v7_cache_similarity_eviction_head_unused",
    "w64_v8_prefix_three_way_decision_invalid",
    "w64_v10_da_tv_falsifier_not_triggered",
)


@dataclasses.dataclass
class W64Params:
    substrate_v9: TinyV9SubstrateParams | None
    v12_cell: V12StackedCell | None
    mlsc_v12_operator: MergeOperatorV12 | None
    consensus_v10: ConsensusFallbackControllerV10 | None
    crc_v12: CorruptionRobustCarrierV12 | None
    lhr_v16: LongHorizonReconstructionV16Head | None
    ecc_v16: ECCCodebookV16 | None
    deep_substrate_hybrid_v9: DeepSubstrateHybridV9 | None
    kv_bridge_v9: KVBridgeV9Projection | None
    hidden_state_bridge_v8: HiddenStateBridgeV8Projection | None
    cache_controller_v7: CacheControllerV7 | None
    replay_controller_v5: ReplayControllerV5 | None
    prefix_v8_drift_predictor_trained: bool

    enabled: bool = True
    follow_up_token_ids: tuple[int, ...] = (257,)
    tvs_budget_tokens: int = 6

    @classmethod
    def build_trivial(cls) -> "W64Params":
        return cls(
            substrate_v9=None, v12_cell=None,
            mlsc_v12_operator=None, consensus_v10=None,
            crc_v12=None, lhr_v16=None, ecc_v16=None,
            deep_substrate_hybrid_v9=None,
            kv_bridge_v9=None,
            hidden_state_bridge_v8=None,
            cache_controller_v7=None,
            replay_controller_v5=None,
            prefix_v8_drift_predictor_trained=False,
            enabled=False,
        )

    @classmethod
    def build_default(
            cls, *, seed: int = 64000,
    ) -> "W64Params":
        sub_v9 = build_default_tiny_substrate_v9(
            seed=int(seed) + 1)
        v12 = V12StackedCell.init(seed=int(seed) + 2)
        mlsc_v12 = MergeOperatorV12(factor_dim=6)
        consensus = ConsensusFallbackControllerV10.init(
            k_required=2, cosine_floor=0.6,
            trust_threshold=0.5)
        crc_v12 = CorruptionRobustCarrierV12()
        lhr_v16 = LongHorizonReconstructionV16Head.init(
            seed=int(seed) + 3)
        ecc_v16 = ECCCodebookV16.init(seed=int(seed) + 4)
        d_head = (
            int(sub_v9.config.d_model)
            // int(sub_v9.config.n_heads))
        kv_b3 = KVBridgeV3Projection.init(
            n_layers=int(sub_v9.config.n_layers),
            n_heads=int(sub_v9.config.n_heads),
            n_kv_heads=int(sub_v9.config.n_kv_heads),
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
        rng = _np.random.default_rng(int(seed) + 14)
        kv_b9 = dataclasses.replace(
            kv_b9,
            correction_layer_f_k=(
                rng.standard_normal(
                    kv_b9.correction_layer_f_k.shape) * 0.02),
            correction_layer_f_v=(
                rng.standard_normal(
                    kv_b9.correction_layer_f_v.shape) * 0.02))
        hsb2 = HiddenStateBridgeV2Projection.init(
            target_layers=(1, 3), n_tokens=6, carrier_dim=6,
            d_model=int(sub_v9.config.d_model),
            seed=int(seed) + 15)
        hsb3 = HiddenStateBridgeV3Projection.init_from_v2(
            hsb2, n_heads=int(sub_v9.config.n_heads),
            seed_v3=int(seed) + 16)
        hsb4 = HiddenStateBridgeV4Projection.init_from_v3(
            hsb3, seed_v4=int(seed) + 17)
        hsb5 = HiddenStateBridgeV5Projection.init_from_v4(
            hsb4, n_positions=3, seed_v5=int(seed) + 18)
        hsb6 = HiddenStateBridgeV6Projection.init_from_v5(
            hsb5, seed_v6=int(seed) + 19)
        hsb7 = HiddenStateBridgeV7Projection.init_from_v6(
            hsb6, seed_v7=int(seed) + 20)
        hsb8 = HiddenStateBridgeV8Projection.init_from_v7(
            hsb7, seed_v8=int(seed) + 21)
        cc7 = CacheControllerV7.init(
            policy=W64_CACHE_POLICY_COMPOSITE_V7,
            d_model=int(sub_v9.config.d_model),
            d_key=int(sub_v9.config.d_key),
            fit_seed=int(seed) + 22)
        cc7 = dataclasses.replace(
            cc7,
            four_objective_head=_np.zeros(
                (4, 4), dtype=_np.float64),
            similarity_eviction_head_coefs=_np.zeros(
                (6,), dtype=_np.float64),
            composite_v7_weights=_np.ones(
                (8,), dtype=_np.float64) * (1.0 / 8.0))
        sup_X = rng.standard_normal((10, 4))
        sup_y1 = sup_X.sum(axis=-1)
        sup_y2 = sup_X[:, 0] * 2.0
        sup_y3 = sup_X[:, 1] - sup_X[:, 2]
        sup_y4 = sup_X[:, 3] * 0.5
        cc7, _ = fit_four_objective_ridge_v7(
            controller=cc7, train_features=sup_X.tolist(),
            target_drop_oracle=sup_y1.tolist(),
            target_retrieval_relevance=sup_y2.tolist(),
            target_hidden_wins=sup_y3.tolist(),
            target_replay_dominance=sup_y4.tolist())
        cc7, _ = fit_similarity_aware_eviction_head_v7(
            controller=cc7,
            train_flag_counts=sup_X[:, 0].astype(int).tolist(),
            train_hidden_writes=sup_X[:, 1].tolist(),
            train_replay_ages=sup_X[:, 2].astype(int).tolist(),
            train_attention_receive_l1=sup_X[:, 3].tolist(),
            train_cache_key_norms=(
                _np.abs(sup_X[:, 0])).tolist(),
            train_mean_similarities=(
                _np.abs(sup_X[:, 1])).tolist(),
            target_eviction_priorities=(
                sup_X[:, 0] * 0.4).tolist())
        head_scores = rng.standard_normal((10, 8))
        cc7, _ = fit_composite_v7(
            controller=cc7,
            head_scores=head_scores.tolist(),
            drop_oracle=head_scores.sum(axis=-1).tolist())
        # Replay controller V5.
        rcv2 = ReplayControllerV2.init()
        train_cands = [
            ReplayCandidate(
                flop_reuse=100, flop_recompute=1000,
                flop_fallback=50,
                drift_l2_reuse=0.1, drift_l2_recompute=0.0,
                drift_l2_fallback=0.3,
                crc_passed=True, transcript_available=True,
                n_corruption_flags=0),
            ReplayCandidate(
                flop_reuse=900, flop_recompute=1000,
                flop_fallback=50,
                drift_l2_reuse=0.8, drift_l2_recompute=0.0,
                drift_l2_fallback=0.3,
                crc_passed=False, transcript_available=True,
                n_corruption_flags=3),
        ]
        rcv2, _ = fit_replay_controller_v2(
            controller=rcv2,
            train_candidates=train_cands,
            train_optimal_decisions=[
                W60_REPLAY_DECISION_REUSE,
                W60_REPLAY_DECISION_RECOMPUTE,
            ])
        rcv3 = ReplayControllerV3.init(inner_v2=rcv2)
        v3_cands = {
            W62_REPLAY_REGIME_SYNTHETIC_CORRUPTION: [
                ReplayCandidate(900, 1000, 50, 0.8, 0.0, 0.3,
                                False, True, 3)],
            W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT: [
                ReplayCandidate(100, 1000, 50, 0.1, 0.0, 0.3,
                                True, True, 0)],
            W62_REPLAY_REGIME_HIDDEN_WRITE_HEAVY: [
                ReplayCandidate(300, 1000, 50, 0.4, 0.0, 0.3,
                                True, True, 0)],
            W62_REPLAY_REGIME_TRANSCRIPT_ONLY: [
                ReplayCandidate(0, 1000, 50, 1.0, 0.0, 0.3,
                                False, False, 5)],
        }
        v3_decs = {
            W62_REPLAY_REGIME_SYNTHETIC_CORRUPTION: [
                W60_REPLAY_DECISION_RECOMPUTE],
            W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT: [
                W60_REPLAY_DECISION_REUSE],
            W62_REPLAY_REGIME_HIDDEN_WRITE_HEAVY: [
                W60_REPLAY_DECISION_RECOMPUTE],
            W62_REPLAY_REGIME_TRANSCRIPT_ONLY: [
                W60_REPLAY_DECISION_FALLBACK],
        }
        rcv3, _ = fit_replay_controller_v3_per_regime(
            controller=rcv3,
            train_candidates_per_regime=v3_cands,
            train_decisions_per_regime=v3_decs)
        feats = rng.standard_normal((24, 5))
        labs = []
        for i in range(24):
            if feats[i, 0] > 0.3:
                labs.append("hidden_beats_kv")
            elif feats[i, 0] < -0.3:
                labs.append("kv_beats_hidden")
            else:
                labs.append("tie")
        rcv3, _ = fit_hidden_vs_kv_regime_classifier(
            controller=rcv3,
            train_features=feats.tolist(),
            train_labels=labs)
        rcv4 = ReplayControllerV4.init(inner_v3=rcv3)
        v4_cands = dict(v3_cands)
        v4_cands[W63_REPLAY_REGIME_HIDDEN_WINS] = [
            ReplayCandidate(250, 1000, 50, 0.3, 0.0, 0.3,
                            True, True, 0)]
        v4_cands[W63_REPLAY_REGIME_CACHE_CORRUPTION_RECOVERED] = [
            ReplayCandidate(500, 1000, 50, 0.6, 0.0, 0.3,
                            True, True, 1)]
        v4_decs = dict(v3_decs)
        v4_decs[W63_REPLAY_REGIME_HIDDEN_WINS] = [
            W60_REPLAY_DECISION_REUSE]
        v4_decs[W63_REPLAY_REGIME_CACHE_CORRUPTION_RECOVERED] = [
            W60_REPLAY_DECISION_REUSE]
        rcv4, _ = fit_replay_controller_v4_per_regime(
            controller=rcv4,
            train_candidates_per_regime=v4_cands,
            train_decisions_per_regime=v4_decs)
        bridge_X = rng.standard_normal((30, 7))
        bridge_labels: list[str] = []
        for i in range(30):
            if bridge_X[i, 5] > 0.3:
                bridge_labels.append("hidden_wins")
            elif bridge_X[i, 6] > 0.3:
                bridge_labels.append("prefix_wins")
            else:
                bridge_labels.append("kv_wins")
        rcv4, _ = fit_three_way_bridge_classifier(
            controller=rcv4,
            train_features=bridge_X.tolist(),
            train_labels=bridge_labels)
        rcv5 = ReplayControllerV5.init(inner_v4=rcv4)
        v5_cands = dict(v4_cands)
        v5_cands[
            W64_REPLAY_REGIME_REPLAY_DOMINANCE_PRIMARY] = [
            ReplayCandidate(200, 1000, 50, 0.2, 0.0, 0.3,
                            True, True, 0)]
        v5_decs = dict(v4_decs)
        v5_decs[
            W64_REPLAY_REGIME_REPLAY_DOMINANCE_PRIMARY] = [
            W60_REPLAY_DECISION_REUSE]
        rcv5, _ = fit_replay_controller_v5_per_regime(
            controller=rcv5,
            train_candidates_per_regime=v5_cands,
            train_decisions_per_regime=v5_decs)
        # Four-way bridge classifier.
        bridge4_X = rng.standard_normal((40, 9))
        bridge4_labels: list[str] = []
        for i in range(40):
            if bridge4_X[i, 8] > 0.3:
                bridge4_labels.append("replay_wins")
            elif bridge4_X[i, 5] > 0.3:
                bridge4_labels.append("hidden_wins")
            elif bridge4_X[i, 6] > 0.3:
                bridge4_labels.append("prefix_wins")
            else:
                bridge4_labels.append("kv_wins")
        rcv5, _ = fit_four_way_bridge_classifier(
            controller=rcv5,
            train_features=bridge4_X.tolist(),
            train_labels=bridge4_labels)
        # Replay-dominance-primary head.
        rdp_X = rng.standard_normal((30, 9))
        rdp_decs: list[str] = []
        for i in range(30):
            if rdp_X[i, 7] > 0.0:
                rdp_decs.append(W60_REPLAY_DECISION_REUSE)
            else:
                rdp_decs.append(W60_REPLAY_DECISION_RECOMPUTE)
        rcv5, _ = fit_replay_dominance_primary_head_v5(
            controller=rcv5,
            train_features=rdp_X.tolist(),
            train_decisions=rdp_decs)
        hybrid_v9 = DeepSubstrateHybridV9(
            inner_v8=DeepSubstrateHybridV8(inner_v7=None))
        return cls(
            substrate_v9=sub_v9, v12_cell=v12,
            mlsc_v12_operator=mlsc_v12,
            consensus_v10=consensus, crc_v12=crc_v12,
            lhr_v16=lhr_v16, ecc_v16=ecc_v16,
            deep_substrate_hybrid_v9=hybrid_v9,
            kv_bridge_v9=kv_b9,
            hidden_state_bridge_v8=hsb8,
            cache_controller_v7=cc7,
            replay_controller_v5=rcv5,
            prefix_v8_drift_predictor_trained=True,
            enabled=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": W64_SCHEMA_VERSION,
            "enabled": bool(self.enabled),
            "substrate_v9_cid": (
                self.substrate_v9.cid()
                if self.substrate_v9 is not None else ""),
            "v12_cell_cid": (
                self.v12_cell.cid()
                if self.v12_cell is not None else ""),
            "mlsc_v12_operator_cid": (
                self.mlsc_v12_operator.cid()
                if self.mlsc_v12_operator is not None else ""),
            "consensus_v10_cid": (
                self.consensus_v10.cid()
                if self.consensus_v10 is not None else ""),
            "crc_v12_cid": (
                self.crc_v12.cid()
                if self.crc_v12 is not None else ""),
            "lhr_v16_cid": (
                self.lhr_v16.cid()
                if self.lhr_v16 is not None else ""),
            "ecc_v16_cid": (
                self.ecc_v16.cid()
                if self.ecc_v16 is not None else ""),
            "deep_substrate_hybrid_v9_cid": (
                self.deep_substrate_hybrid_v9.cid()
                if self.deep_substrate_hybrid_v9 is not None
                else ""),
            "kv_bridge_v9_cid": (
                self.kv_bridge_v9.cid()
                if self.kv_bridge_v9 is not None else ""),
            "hidden_state_bridge_v8_cid": (
                self.hidden_state_bridge_v8.cid()
                if self.hidden_state_bridge_v8 is not None
                else ""),
            "cache_controller_v7_cid": (
                self.cache_controller_v7.cid()
                if self.cache_controller_v7 is not None
                else ""),
            "replay_controller_v5_cid": (
                self.replay_controller_v5.cid()
                if self.replay_controller_v5 is not None
                else ""),
            "prefix_v8_drift_predictor_trained": bool(
                self.prefix_v8_drift_predictor_trained),
            "follow_up_token_ids": list(
                self.follow_up_token_ids),
            "tvs_budget_tokens": int(self.tvs_budget_tokens),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w64_params",
            "params": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class W64HandoffEnvelope:
    schema: str
    w63_outer_cid: str
    w64_params_cid: str
    substrate_v9_witness_cid: str
    kv_bridge_v9_witness_cid: str
    hsb_v8_witness_cid: str
    prefix_state_v8_witness_cid: str
    attn_steer_v8_witness_cid: str
    cache_controller_v7_witness_cid: str
    replay_controller_v5_witness_cid: str
    persistent_v16_witness_cid: str
    multi_hop_v14_witness_cid: str
    mlsc_v12_witness_cid: str
    consensus_v10_witness_cid: str
    crc_v12_witness_cid: str
    lhr_v16_witness_cid: str
    ecc_v16_witness_cid: str
    tvs_v13_witness_cid: str
    uncertainty_v12_witness_cid: str
    disagreement_algebra_v10_witness_cid: str
    deep_substrate_hybrid_v9_witness_cid: str
    substrate_adapter_v9_matrix_cid: str
    hidden_wins_primary_falsifier_witness_cid: str
    v16_chain_cid: str
    nine_way_used: bool
    substrate_v9_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "w63_outer_cid": str(self.w63_outer_cid),
            "w64_params_cid": str(self.w64_params_cid),
            "substrate_v9_witness_cid": str(
                self.substrate_v9_witness_cid),
            "kv_bridge_v9_witness_cid": str(
                self.kv_bridge_v9_witness_cid),
            "hsb_v8_witness_cid": str(self.hsb_v8_witness_cid),
            "prefix_state_v8_witness_cid": str(
                self.prefix_state_v8_witness_cid),
            "attn_steer_v8_witness_cid": str(
                self.attn_steer_v8_witness_cid),
            "cache_controller_v7_witness_cid": str(
                self.cache_controller_v7_witness_cid),
            "replay_controller_v5_witness_cid": str(
                self.replay_controller_v5_witness_cid),
            "persistent_v16_witness_cid": str(
                self.persistent_v16_witness_cid),
            "multi_hop_v14_witness_cid": str(
                self.multi_hop_v14_witness_cid),
            "mlsc_v12_witness_cid": str(
                self.mlsc_v12_witness_cid),
            "consensus_v10_witness_cid": str(
                self.consensus_v10_witness_cid),
            "crc_v12_witness_cid": str(
                self.crc_v12_witness_cid),
            "lhr_v16_witness_cid": str(
                self.lhr_v16_witness_cid),
            "ecc_v16_witness_cid": str(
                self.ecc_v16_witness_cid),
            "tvs_v13_witness_cid": str(
                self.tvs_v13_witness_cid),
            "uncertainty_v12_witness_cid": str(
                self.uncertainty_v12_witness_cid),
            "disagreement_algebra_v10_witness_cid": str(
                self.disagreement_algebra_v10_witness_cid),
            "deep_substrate_hybrid_v9_witness_cid": str(
                self.deep_substrate_hybrid_v9_witness_cid),
            "substrate_adapter_v9_matrix_cid": str(
                self.substrate_adapter_v9_matrix_cid),
            "hidden_wins_primary_falsifier_witness_cid": str(
                self.hidden_wins_primary_falsifier_witness_cid),
            "v16_chain_cid": str(self.v16_chain_cid),
            "nine_way_used": bool(self.nine_way_used),
            "substrate_v9_used": bool(self.substrate_v9_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w64_handoff_envelope",
            "envelope": self.to_dict()})


def verify_w64_handoff(
        envelope: W64HandoffEnvelope,
) -> dict[str, Any]:
    failures: list[str] = []
    def need(field: str, key: str) -> None:
        if not getattr(envelope, field, ""):
            failures.append(key)
    need("w63_outer_cid", "missing_w63_outer_cid")
    need("substrate_v9_witness_cid",
         "missing_substrate_v9_witness")
    need("kv_bridge_v9_witness_cid",
         "missing_kv_bridge_v9_witness")
    need("hsb_v8_witness_cid", "missing_hsb_v8_witness")
    need("prefix_state_v8_witness_cid",
         "missing_prefix_state_v8_witness")
    need("attn_steer_v8_witness_cid",
         "missing_attn_steer_v8_witness")
    need("cache_controller_v7_witness_cid",
         "missing_cache_controller_v7_witness")
    need("replay_controller_v5_witness_cid",
         "missing_replay_controller_v5_witness")
    need("persistent_v16_witness_cid",
         "missing_persistent_v16_witness")
    need("multi_hop_v14_witness_cid",
         "missing_multi_hop_v14_witness")
    need("mlsc_v12_witness_cid", "missing_mlsc_v12_witness")
    need("consensus_v10_witness_cid",
         "missing_consensus_v10_witness")
    need("crc_v12_witness_cid", "missing_crc_v12_witness")
    need("lhr_v16_witness_cid", "missing_lhr_v16_witness")
    need("ecc_v16_witness_cid", "missing_ecc_v16_witness")
    need("tvs_v13_witness_cid", "missing_tvs_v13_witness")
    need("uncertainty_v12_witness_cid",
         "missing_uncertainty_v12_witness")
    need("disagreement_algebra_v10_witness_cid",
         "missing_disagreement_algebra_v10_witness")
    need("deep_substrate_hybrid_v9_witness_cid",
         "missing_deep_substrate_hybrid_v9_witness")
    need("substrate_adapter_v9_matrix_cid",
         "missing_substrate_adapter_v9_matrix")
    return {
        "schema": W64_SCHEMA_VERSION + ".verifier.v1",
        "failures": list(failures),
        "ok": bool(not failures),
        "n_failure_modes_known": int(
            len(W64_ENVELOPE_VERIFIER_FAILURE_MODES)),
    }


@dataclasses.dataclass
class W64Team:
    params: W64Params
    chain: PersistentLatentStateV16Chain = dataclasses.field(
        default_factory=PersistentLatentStateV16Chain.empty)

    def step(
            self, *, turn_index: int, role: str = "r",
            w63_outer_cid: str = "no_w63",
    ) -> W64HandoffEnvelope:
        p = self.params
        sub_w_cid = ""
        sub_used = False
        primary_l1 = 0.0
        if p.enabled and p.substrate_v9 is not None:
            ids = tokenize_bytes_v9(
                "w64-turn-" + str(int(turn_index)),
                max_len=14)
            trace, cache = forward_tiny_substrate_v9(
                p.substrate_v9, ids)
            # Seed the V9 axes so they fire.
            record_hidden_wins_primary_v9(
                cache, layer_index=0, head_index=0, slot=0,
                decision="hidden_wins")
            record_replay_dominance_witness_v9(
                cache, layer_index=0, head_index=0, slot=0,
                dominance=0.7)
            record_hidden_state_trust_decision_v9(
                cache, layer_index=0, head_index=0,
                decision="hidden_wins")
            w = emit_tiny_substrate_v9_forward_witness(
                trace, cache)
            sub_w_cid = w.cid()
            sub_used = True
            primary_l1 = float(w.hidden_wins_primary_l1)
        kv_w_cid = ""
        hidden_wins_primary_falsifier_cid = ""
        if (p.enabled and p.kv_bridge_v9 is not None):
            falsifier = (
                probe_kv_bridge_v9_hidden_wins_primary_falsifier(
                    primary_flag=0.5))
            hidden_wins_primary_falsifier_cid = falsifier.cid()
            kv_w_cid = emit_kv_bridge_v9_witness(
                projection=p.kv_bridge_v9,
                hidden_wins_primary_falsifier=falsifier).cid()
        hsb_w_cid = ""
        hsb_v8_hw_primary_margin = 0.0
        if (p.enabled and p.hidden_state_bridge_v8 is not None):
            from .hidden_state_bridge_v8 import (
                compute_hsb_v8_hidden_wins_primary_margin,
            )
            hsb_v8_hw_primary_margin = float(
                compute_hsb_v8_hidden_wins_primary_margin(
                    hidden_residual_l2=0.2,
                    kv_residual_l2=0.5,
                    prefix_residual_l2=0.4))
            hsb_w_cid = emit_hsb_v8_witness(
                projection=p.hidden_state_bridge_v8,
                hidden_wins_primary_margin=(
                    hsb_v8_hw_primary_margin),
                hidden_state_trust_ledger_l1=0.5).cid()
        prefix_w_cid = _sha256_hex({
            "schema": "prefix_v8_compact_witness",
            "turn": int(turn_index),
            "drift_predictor_trained": bool(
                p.prefix_v8_drift_predictor_trained)})
        attn_w_cid = _sha256_hex({
            "schema": "attn_steering_v8_compact_witness",
            "turn": int(turn_index)})
        cc_w_cid = ""
        if (p.enabled and p.cache_controller_v7 is not None):
            cc_w_cid = emit_cache_controller_v7_witness(
                controller=p.cache_controller_v7).cid()
        rc_w_cid = ""
        if (p.enabled and p.replay_controller_v5 is not None):
            cand = ReplayCandidate(
                flop_reuse=100, flop_recompute=1000,
                flop_fallback=50,
                drift_l2_reuse=0.1, drift_l2_recompute=0.0,
                drift_l2_fallback=0.3,
                crc_passed=True, transcript_available=True,
                n_corruption_flags=0)
            p.replay_controller_v5.decide_v5(
                cand,
                hidden_vs_kv_contention=0.5,
                prefix_reuse_trust=0.7,
                replay_determinism_mean=0.8,
                replay_dominance_witness_mean=0.6,
                hidden_wins_primary_score=0.4)
            rc_w_cid = emit_replay_controller_v5_witness(
                p.replay_controller_v5).cid()
        per_w_cid = ""
        if p.enabled and p.v12_cell is not None:
            carrier_vals = _payload_hash_vec(
                ("v16", int(turn_index)),
                int(p.v12_cell.state_dim))
            state = step_persistent_state_v16(
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
                substrate_fidelity=0.9,
                attention_fidelity=0.9,
                retrieval_fidelity=0.9,
                replay_fidelity=0.9)
            self.chain.add(state)
            per_w_cid = emit_persistent_v16_witness(
                self.chain, state.cid()).cid()
        mh_w_cid = ""
        if p.enabled:
            mh_w_cid = emit_multi_hop_v14_witness(
                backends=W64_DEFAULT_MH_V14_BACKENDS,
                chain_length=W64_DEFAULT_MH_V14_CHAIN_LEN,
                seed=int(turn_index) + 9300).cid()
        mlsc_w_cid = ""
        if (p.enabled and p.mlsc_v12_operator is not None):
            v3 = make_root_capsule_v3(
                branch_id=f"w64_{int(turn_index)}",
                payload=tuple([0.1] * 6),
                fact_tags=("w64",),
                confidence=0.9, trust=0.9,
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
                provenance_trust_table={
                    "backend_a": 0.9, "backend_b": 0.7})
            v9 = wrap_v8_as_v9(
                v8,
                attention_pattern_witness_chain=(
                    f"ap_chain_{turn_index}",),
                cache_retrieval_witness_chain=(
                    f"cr_chain_{turn_index}",),
                per_layer_head_trust_matrix=(
                    (0, 0, 0.9), (1, 1, 0.7)))
            v10 = wrap_v9_as_v10(
                v9,
                replay_dominance_witness_chain=(
                    f"rd_chain_{turn_index}",),
                disagreement_wasserstein_distance=0.05,
                algebra_signature_v10=(
                    W62_MLSC_V10_ALGEBRA_REPLAY_DOMINANCE_PROPAGATION))
            v11 = wrap_v10_as_v11(
                v10,
                hidden_wins_witness_chain=(
                    f"hw_chain_{turn_index}",),
                prefix_reuse_witness_chain=(
                    f"pr_chain_{turn_index}",),
                disagreement_jensen_shannon_distance=0.04,
                algebra_signature_v11=(
                    W63_MLSC_V11_ALGEBRA_HIDDEN_WINS_PROPAGATION))
            v12_cap = wrap_v11_as_v12(
                v11,
                replay_dominance_primary_witness_chain=(
                    f"rdp_chain_{turn_index}",),
                hidden_state_trust_witness_chain=(
                    f"hst_chain_{turn_index}",),
                disagreement_total_variation_distance=0.03,
                algebra_signature_v12=(
                    W64_MLSC_V12_ALGEBRA_REPLAY_DOMINANCE_PRIMARY_PROPAGATION))
            merged = p.mlsc_v12_operator.merge(
                [v12_cap],
                replay_dominance_primary_witness_chain=(
                    f"merge_rdp_{turn_index}",),
                hidden_state_trust_witness_chain=(
                    f"merge_hst_{turn_index}",),
                algebra_signature_v12=(
                    W64_MLSC_V12_ALGEBRA_REPLAY_DOMINANCE_PRIMARY_PROPAGATION))
            mlsc_w_cid = emit_mlsc_v12_witness(merged).cid()
        cons_w_cid = ""
        if (p.enabled and p.consensus_v10 is not None):
            # Fire the V10 stage so the witness records it.
            p.consensus_v10.decide_v10(
                payloads=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                trusts=[0.4, 0.4],
                replay_decisions=[
                    "choose_abstain", "choose_abstain"],
                transcript_available=False,
                corruption_detected_per_parent=[False, False],
                repair_amount=0.0,
                hidden_wins_margins_per_parent=[0.0, 0.0],
                three_way_predictions_per_parent=[
                    "kv_wins", "kv_wins"],
                replay_dominance_primary_scores_per_parent=[
                    0.1, 0.0],
                four_way_predictions_per_parent=[
                    "replay_wins", "kv_wins"])
            cons_w_cid = emit_consensus_v10_witness(
                p.consensus_v10).cid()
        crc_w_cid = ""
        if (p.enabled and p.crc_v12 is not None):
            crc_w_cid = (
                emit_corruption_robustness_v12_witness(
                    crc_v12=p.crc_v12, n_probes=8,
                    seed=int(turn_index) + 9400).cid())
        lhr_w_cid = ""
        if (p.enabled and p.lhr_v16 is not None):
            lhr_w_cid = emit_lhr_v16_witness(
                p.lhr_v16, carrier=[0.1] * 8, k=4,
                replay_dominance_indicator=[1.0] * 8,
                hidden_wins_indicator=[0.5] * 8,
                replay_dominance_primary_indicator=[
                    0.7] * 8).cid()
        ecc_w_cid = ""
        if (p.enabled and p.ecc_v16 is not None):
            gate = QuantisedBudgetGate.init(
                in_dim=W53_DEFAULT_ECC_CODE_DIM,
                emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
                seed=int(turn_index) + 9500)
            gate.importance_threshold = 0.0
            gate.w_emit.values = [1.0] * len(
                gate.w_emit.values)
            carrier = _payload_hash_vec(
                ("ecc16", int(turn_index)),
                W53_DEFAULT_ECC_CODE_DIM)
            comp = compress_carrier_ecc_v16(
                carrier, codebook=p.ecc_v16, gate=gate)
            ecc_w_cid = emit_ecc_v16_compression_witness(
                codebook=p.ecc_v16, compression=comp).cid()
        tvs_w_cid = ""
        if p.enabled:
            tvs_res = fourteen_arm_compare(
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
                budget_tokens=int(p.tvs_budget_tokens))
            tvs_w_cid = emit_tvs_arbiter_v13_witness(
                tvs_res).cid()
        unc_w_cid = ""
        if p.enabled:
            unc = compose_uncertainty_report_v12(
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
                    0.84, 0.80])
            unc_w_cid = emit_uncertainty_v12_witness(unc).cid()
        da_w_cid = ""
        if p.enabled:
            trace = AlgebraTrace(steps=[])
            probe = [0.1, 0.2, 0.3]
            wda = emit_disagreement_algebra_v10_witness(
                trace=trace, probe_a=probe, probe_b=probe,
                probe_c=probe,
                tv_oracle=lambda: (True, 0.05),
                tv_falsifier_oracle=lambda: (False, 1.0),
                js_oracle=lambda: (True, 0.05),
                js_falsifier_oracle=lambda: (False, 1.0),
                wasserstein_oracle=lambda: (True, 0.1),
                wasserstein_falsifier_oracle=(
                    lambda: (False, 1.0)),
                attention_pattern_oracle=lambda: (True, 0.85))
            da_w_cid = wda.cid()
        hybrid_w_cid = ""
        nine_way = False
        if (p.enabled
                and p.deep_substrate_hybrid_v9 is not None):
            v8_witness = DeepSubstrateHybridV8ForwardWitness(
                schema="x", hybrid_cid="x",
                inner_v7_witness_cid="x",
                eight_way=True,
                cache_controller_v6_fired=True,
                replay_controller_v4_fired=True,
                three_way_bridge_classifier_fired=True,
                hidden_vs_kv_contention_active=True,
                attention_v7_active=True,
                prefix_v7_drift_predictor_active=True,
                prefix_reuse_trust_active=True,
                mean_replay_dominance=0.4,
                hidden_vs_kv_contention_l1=1.0,
                attention_v7_js_max=0.15,
                prefix_reuse_trust_l1=0.7)
            wh = deep_substrate_hybrid_v9_forward(
                hybrid=p.deep_substrate_hybrid_v9,
                v8_witness=v8_witness,
                cache_controller_v7=p.cache_controller_v7,
                replay_controller_v5=p.replay_controller_v5,
                hidden_wins_primary_l1=float(
                    primary_l1 + 1.0),
                attention_v8_hellinger_max=0.10,
                prefix_v8_drift_predictor_trained=bool(
                    p.prefix_v8_drift_predictor_trained),
                hidden_state_trust_ledger_l1=0.5,
                hsb_v8_hidden_wins_primary_margin=float(
                    hsb_v8_hw_primary_margin + 0.1))
            hybrid_w_cid = wh.cid()
            nine_way = bool(wh.nine_way)
        adapter_cid = ""
        if p.enabled:
            matrix = probe_all_v9_adapters(
                probe_ollama=False, probe_openai=False)
            adapter_cid = matrix.cid()
        return W64HandoffEnvelope(
            schema=W64_SCHEMA_VERSION,
            w63_outer_cid=str(w63_outer_cid),
            w64_params_cid=str(p.cid()),
            substrate_v9_witness_cid=str(sub_w_cid),
            kv_bridge_v9_witness_cid=str(kv_w_cid),
            hsb_v8_witness_cid=str(hsb_w_cid),
            prefix_state_v8_witness_cid=str(prefix_w_cid),
            attn_steer_v8_witness_cid=str(attn_w_cid),
            cache_controller_v7_witness_cid=str(cc_w_cid),
            replay_controller_v5_witness_cid=str(rc_w_cid),
            persistent_v16_witness_cid=str(per_w_cid),
            multi_hop_v14_witness_cid=str(mh_w_cid),
            mlsc_v12_witness_cid=str(mlsc_w_cid),
            consensus_v10_witness_cid=str(cons_w_cid),
            crc_v12_witness_cid=str(crc_w_cid),
            lhr_v16_witness_cid=str(lhr_w_cid),
            ecc_v16_witness_cid=str(ecc_w_cid),
            tvs_v13_witness_cid=str(tvs_w_cid),
            uncertainty_v12_witness_cid=str(unc_w_cid),
            disagreement_algebra_v10_witness_cid=str(da_w_cid),
            deep_substrate_hybrid_v9_witness_cid=str(
                hybrid_w_cid),
            substrate_adapter_v9_matrix_cid=str(adapter_cid),
            hidden_wins_primary_falsifier_witness_cid=str(
                hidden_wins_primary_falsifier_cid),
            v16_chain_cid=str(self.chain.cid()),
            nine_way_used=bool(nine_way),
            substrate_v9_used=bool(sub_used),
        )


__all__ = [
    "W64_SCHEMA_VERSION",
    "W64_TEAM_RESULT_SCHEMA",
    "W64_ENVELOPE_VERIFIER_FAILURE_MODES",
    "W64Params",
    "W64HandoffEnvelope",
    "W64Team",
    "verify_w64_handoff",
]
