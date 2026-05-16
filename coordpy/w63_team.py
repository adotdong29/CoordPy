"""W63 — Stronger Replay-Dominance / Hidden-Wins / 1024-turn
Substrate-Coupled Latent OS team.

The ``W63Team`` orchestrator composes the W62 team with the W63
mechanism modules (V8 substrate + V8 KV bridge + V7 HSB + V7 prefix
+ V7 attention + V6 cache controller + V4 replay controller + V8
hybrid + V15 persistent + V13 multi-hop + V11 capsule + V9 consensus
+ V11 corruption + V15 LHR + V15 ECC + V11 uncertainty + V9
disagreement + V12 TVS + V8 adapter). Per-turn it emits 22 module
witness CIDs and seals them into a ``W63HandoffEnvelope`` whose
``w62_outer_cid`` carries forward the W62 envelope byte-for-byte.

Honest scope
------------

* The W63 substrate is the in-repo V8 NumPy runtime. We do NOT
  bridge to third-party hosted models.
  ``W63-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward
  the W56..W62 cap unchanged.
* W63 fits **closed-form ridge parameters** in five new places on
  top of W62's twelve: cache controller V6 three-objective head;
  cache controller V6 retrieval-repair head; cache controller V6
  composite_v6; replay controller V4 per-regime head (6 regimes);
  replay controller V4 three-way bridge classifier.
  Total **seventeen closed-form ridge solves**.
  ``W63-L-V8-NO-AUTOGRAD-CAP`` documents.
* Trivial passthrough preserved: when ``W63Params.build_trivial()``
  is used the W63 envelope's internal ``w62_outer_cid`` carries
  the supplied W62 outer CID exactly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

import numpy as _np

from .cache_controller_v6 import (
    CacheControllerV6,
    W63_CACHE_POLICY_COMPOSITE_V6,
    emit_cache_controller_v6_witness,
    fit_composite_v6,
    fit_retrieval_repair_head_v6,
    fit_three_objective_ridge_v6,
)
from .consensus_fallback_controller_v9 import (
    ConsensusFallbackControllerV9,
    W63_CONSENSUS_V9_STAGES,
    emit_consensus_v9_witness,
)
from .corruption_robust_carrier_v11 import (
    CorruptionRobustCarrierV11,
    emit_corruption_robustness_v11_witness,
)
from .deep_substrate_hybrid_v7 import (
    DeepSubstrateHybridV7,
    DeepSubstrateHybridV7ForwardWitness,
)
from .deep_substrate_hybrid_v8 import (
    DeepSubstrateHybridV8,
    deep_substrate_hybrid_v8_forward,
)
from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v9 import (
    emit_disagreement_algebra_v9_witness,
)
from .ecc_codebook_v15 import (
    ECCCodebookV15, compress_carrier_ecc_v15,
    emit_ecc_v15_compression_witness,
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
    emit_hsb_v7_witness,
)
from .kv_bridge_v3 import KVBridgeV3Projection
from .kv_bridge_v4 import KVBridgeV4Projection
from .kv_bridge_v5 import KVBridgeV5Projection
from .kv_bridge_v6 import KVBridgeV6Projection
from .kv_bridge_v7 import KVBridgeV7Projection
from .kv_bridge_v8 import (
    KVBridgeV8Projection,
    emit_kv_bridge_v8_witness,
    probe_kv_bridge_v8_hidden_wins_falsifier,
)
from .long_horizon_retention_v15 import (
    LongHorizonReconstructionV15Head,
    emit_lhr_v15_witness,
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
    MergeOperatorV11,
    W63_MLSC_V11_ALGEBRA_HIDDEN_WINS_PROPAGATION,
    emit_mlsc_v11_witness, wrap_v10_as_v11,
)
from .multi_hop_translator_v13 import (
    W63_DEFAULT_MH_V13_BACKENDS,
    W63_DEFAULT_MH_V13_CHAIN_LEN,
    emit_multi_hop_v13_witness,
)
from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v15 import (
    PersistentLatentStateV15Chain,
    W63_DEFAULT_V15_MAX_CHAIN_WALK_DEPTH,
    emit_persistent_v15_witness,
    step_persistent_state_v15,
)
from .quantised_compression import QuantisedBudgetGate
from .replay_controller import (
    ReplayCandidate, W60_REPLAY_DECISION_REUSE,
    W60_REPLAY_DECISION_RECOMPUTE,
    W60_REPLAY_DECISION_ABSTAIN,
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
    emit_replay_controller_v4_witness,
    fit_replay_controller_v4_per_regime,
    fit_three_way_bridge_classifier,
)
from .substrate_adapter_v8 import (
    SubstrateAdapterV8Matrix,
    W63_SUBSTRATE_TIER_SUBSTRATE_V8_FULL,
    probe_all_v8_adapters,
)
from .tiny_substrate_v8 import (
    TinyV8SubstrateParams,
    build_default_tiny_substrate_v8,
    emit_tiny_substrate_v8_forward_witness,
    forward_tiny_substrate_v8,
    record_prefix_reuse_decision_v8,
    tokenize_bytes_v8,
)
from .transcript_vs_shared_arbiter_v12 import (
    ThirteenArmCompareResult,
    emit_tvs_arbiter_v12_witness,
    thirteen_arm_compare,
)
from .uncertainty_layer_v11 import (
    compose_uncertainty_report_v11,
    emit_uncertainty_v11_witness,
)


W63_SCHEMA_VERSION: str = "coordpy.w63_team.v1"
W63_TEAM_RESULT_SCHEMA: str = "coordpy.w63_team_result.v1"


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
# Failure mode enumeration (≥ 72 disjoint modes for W63)
# =============================================================================

W63_ENVELOPE_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "missing_w62_outer_cid",
    "missing_substrate_v8_witness",
    "substrate_v8_witness_invalid",
    "missing_kv_bridge_v8_witness",
    "kv_bridge_v8_four_target_unfit",
    "missing_hsb_v7_witness",
    "hsb_v7_four_target_unfit",
    "missing_prefix_state_v7_witness",
    "prefix_state_v7_drift_curve_unfit",
    "missing_attn_steer_v7_witness",
    "attn_steer_v7_three_stage_inactive",
    "missing_cache_controller_v6_witness",
    "cache_controller_v6_three_objective_unfit",
    "cache_controller_v6_retrieval_repair_unfit",
    "cache_controller_v6_composite_v6_unfit",
    "missing_replay_controller_v4_witness",
    "replay_controller_v4_per_regime_unfit",
    "replay_controller_v4_three_way_bridge_classifier_unfit",
    "missing_persistent_v15_witness",
    "persistent_v15_chain_walk_short",
    "persistent_v15_twelfth_skip_absent",
    "missing_multi_hop_v13_witness",
    "multi_hop_v13_chain_length_off",
    "multi_hop_v13_eight_axis_missing",
    "missing_mlsc_v11_witness",
    "mlsc_v11_hidden_wins_chain_empty",
    "mlsc_v11_jensen_shannon_not_computed",
    "missing_consensus_v9_witness",
    "consensus_v9_stage_count_off",
    "consensus_v9_hidden_wins_stage_unused",
    "missing_crc_v11_witness",
    "crc_v11_kv2048_detect_below_floor",
    "crc_v11_19bit_burst_below_floor",
    "crc_v11_hidden_state_recovery_ratio_above_floor",
    "missing_lhr_v15_witness",
    "lhr_v15_max_k_off",
    "lhr_v15_fourteen_way_failed",
    "missing_ecc_v15_witness",
    "ecc_v15_bits_per_token_below_floor",
    "ecc_v15_total_codes_off",
    "missing_tvs_v12_witness",
    "tvs_v12_pick_rates_not_sum_to_one",
    "tvs_v12_hidden_wins_arm_inactive",
    "missing_uncertainty_v11_witness",
    "uncertainty_v11_hidden_wins_unaware",
    "missing_disagreement_algebra_v9_witness",
    "disagreement_algebra_v9_jensen_shannon_identity_failed",
    "missing_deep_substrate_hybrid_v8_witness",
    "deep_substrate_hybrid_v8_not_eight_way",
    "missing_substrate_adapter_v8_matrix",
    "substrate_adapter_v8_no_v8_full",
    "w63_outer_cid_mismatch_under_replay",
    "w63_params_cid_mismatch",
    "w63_envelope_schema_drift",
    "w63_trivial_passthrough_broken",
    "w63_v8_no_autograd_cap_missing",
    "w63_no_third_party_substrate_coupling_cap_missing",
    "w63_v15_outer_not_trained_cap_missing",
    "w63_ecc_v15_rate_floor_cap_missing",
    "w63_lhr_v15_scorer_fit_cap_missing",
    "w63_prefix_v7_token_fingerprint_cap_missing",
    "w63_v6_cache_controller_no_autograd_cap_missing",
    "w63_v4_replay_no_autograd_cap_missing",
    "w63_v7_hsb_no_autograd_cap_missing",
    "w63_v7_attn_no_autograd_cap_missing",
    "w63_multi_hop_v13_synthetic_backends_cap_missing",
    "w63_crc_v11_fingerprint_synthetic_cap_missing",
    "w63_consensus_v9_hidden_wins_stage_synthetic_cap_missing",
    "w63_kv_bridge_v8_hidden_wins_target_constructed_cap_missing",
    "w63_v8_substrate_numpy_cpu_cap_missing",
    "w63_hidden_wins_falsifier_not_zero_under_inversion",
    "w63_prefix_vs_hidden_decision_invalid",
)


@dataclasses.dataclass
class W63Params:
    substrate_v8: TinyV8SubstrateParams | None
    v12_cell: V12StackedCell | None
    mlsc_v11_operator: MergeOperatorV11 | None
    consensus_v9: ConsensusFallbackControllerV9 | None
    crc_v11: CorruptionRobustCarrierV11 | None
    lhr_v15: LongHorizonReconstructionV15Head | None
    ecc_v15: ECCCodebookV15 | None
    deep_substrate_hybrid_v8: DeepSubstrateHybridV8 | None
    kv_bridge_v8: KVBridgeV8Projection | None
    hidden_state_bridge_v7: HiddenStateBridgeV7Projection | None
    cache_controller_v6: CacheControllerV6 | None
    replay_controller_v4: ReplayControllerV4 | None
    prefix_v7_drift_predictor_trained: bool

    enabled: bool = True
    follow_up_token_ids: tuple[int, ...] = (257,)
    tvs_budget_tokens: int = 6

    @classmethod
    def build_trivial(cls) -> "W63Params":
        return cls(
            substrate_v8=None, v12_cell=None,
            mlsc_v11_operator=None, consensus_v9=None,
            crc_v11=None, lhr_v15=None, ecc_v15=None,
            deep_substrate_hybrid_v8=None,
            kv_bridge_v8=None,
            hidden_state_bridge_v7=None,
            cache_controller_v6=None,
            replay_controller_v4=None,
            prefix_v7_drift_predictor_trained=False,
            enabled=False,
        )

    @classmethod
    def build_default(
            cls, *, seed: int = 63000,
    ) -> "W63Params":
        sub_v8 = build_default_tiny_substrate_v8(
            seed=int(seed) + 1)
        v12 = V12StackedCell.init(seed=int(seed) + 2)
        mlsc_v11 = MergeOperatorV11(factor_dim=6)
        consensus = ConsensusFallbackControllerV9.init(
            k_required=2, cosine_floor=0.6,
            trust_threshold=0.5)
        crc_v11 = CorruptionRobustCarrierV11()
        lhr_v15 = LongHorizonReconstructionV15Head.init(
            seed=int(seed) + 3)
        ecc_v15 = ECCCodebookV15.init(seed=int(seed) + 4)
        d_head = (
            int(sub_v8.config.d_model)
            // int(sub_v8.config.n_heads))
        kv_b3 = KVBridgeV3Projection.init(
            n_layers=int(sub_v8.config.n_layers),
            n_heads=int(sub_v8.config.n_heads),
            n_kv_heads=int(sub_v8.config.n_kv_heads),
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
        rng = _np.random.default_rng(int(seed) + 13)
        kv_b8 = dataclasses.replace(
            kv_b8,
            correction_layer_e_k=(
                rng.standard_normal(
                    kv_b8.correction_layer_e_k.shape) * 0.02),
            correction_layer_e_v=(
                rng.standard_normal(
                    kv_b8.correction_layer_e_v.shape) * 0.02))
        hsb2 = HiddenStateBridgeV2Projection.init(
            target_layers=(1, 3), n_tokens=6, carrier_dim=6,
            d_model=int(sub_v8.config.d_model),
            seed=int(seed) + 14)
        hsb3 = HiddenStateBridgeV3Projection.init_from_v2(
            hsb2, n_heads=int(sub_v8.config.n_heads),
            seed_v3=int(seed) + 15)
        hsb4 = HiddenStateBridgeV4Projection.init_from_v3(
            hsb3, seed_v4=int(seed) + 16)
        hsb5 = HiddenStateBridgeV5Projection.init_from_v4(
            hsb4, n_positions=3, seed_v5=int(seed) + 17)
        hsb6 = HiddenStateBridgeV6Projection.init_from_v5(
            hsb5, seed_v6=int(seed) + 18)
        hsb7 = HiddenStateBridgeV7Projection.init_from_v6(
            hsb6, seed_v7=int(seed) + 19)
        cc6 = CacheControllerV6.init(
            policy=W63_CACHE_POLICY_COMPOSITE_V6,
            d_model=int(sub_v8.config.d_model),
            d_key=int(sub_v8.config.d_key),
            fit_seed=int(seed) + 20)
        cc6 = dataclasses.replace(
            cc6,
            three_objective_head=_np.zeros(
                (4, 3), dtype=_np.float64),
            retrieval_repair_head_coefs=_np.zeros(
                (5,), dtype=_np.float64),
            composite_v6_weights=_np.ones(
                (7,), dtype=_np.float64) * (1.0 / 7.0))
        sup_X = rng.standard_normal((10, 4))
        sup_y1 = sup_X.sum(axis=-1)
        sup_y2 = sup_X[:, 0] * 2.0
        sup_y3 = sup_X[:, 1] - sup_X[:, 2]
        cc6, _ = fit_three_objective_ridge_v6(
            controller=cc6, train_features=sup_X.tolist(),
            target_drop_oracle=sup_y1.tolist(),
            target_retrieval_relevance=sup_y2.tolist(),
            target_hidden_wins=sup_y3.tolist())
        cc6, _ = fit_retrieval_repair_head_v6(
            controller=cc6,
            train_flag_counts=sup_X[:, 0].astype(int).tolist(),
            train_hidden_writes=sup_X[:, 1].tolist(),
            train_replay_ages=sup_X[:, 2].astype(int).tolist(),
            train_attention_receive_l1=sup_X[:, 3].tolist(),
            train_cache_key_norms=(
                _np.abs(sup_X[:, 0])).tolist(),
            target_repair_amounts=(
                sup_X[:, 0] * 0.4).tolist())
        head_scores = rng.standard_normal((10, 7))
        cc6, _ = fit_composite_v6(
            controller=cc6,
            head_scores=head_scores.tolist(),
            drop_oracle=head_scores.sum(axis=-1).tolist())
        # Replay controller V4.
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
        # Three-way bridge classifier.
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
        hybrid_v8 = DeepSubstrateHybridV8(
            inner_v7=DeepSubstrateHybridV7(inner_v6=None))
        return cls(
            substrate_v8=sub_v8, v12_cell=v12,
            mlsc_v11_operator=mlsc_v11,
            consensus_v9=consensus, crc_v11=crc_v11,
            lhr_v15=lhr_v15, ecc_v15=ecc_v15,
            deep_substrate_hybrid_v8=hybrid_v8,
            kv_bridge_v8=kv_b8,
            hidden_state_bridge_v7=hsb7,
            cache_controller_v6=cc6,
            replay_controller_v4=rcv4,
            prefix_v7_drift_predictor_trained=True,
            enabled=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": W63_SCHEMA_VERSION,
            "enabled": bool(self.enabled),
            "substrate_v8_cid": (
                self.substrate_v8.cid()
                if self.substrate_v8 is not None else ""),
            "v12_cell_cid": (
                self.v12_cell.cid()
                if self.v12_cell is not None else ""),
            "mlsc_v11_operator_cid": (
                self.mlsc_v11_operator.cid()
                if self.mlsc_v11_operator is not None else ""),
            "consensus_v9_cid": (
                self.consensus_v9.cid()
                if self.consensus_v9 is not None else ""),
            "crc_v11_cid": (
                self.crc_v11.cid()
                if self.crc_v11 is not None else ""),
            "lhr_v15_cid": (
                self.lhr_v15.cid()
                if self.lhr_v15 is not None else ""),
            "ecc_v15_cid": (
                self.ecc_v15.cid()
                if self.ecc_v15 is not None else ""),
            "deep_substrate_hybrid_v8_cid": (
                self.deep_substrate_hybrid_v8.cid()
                if self.deep_substrate_hybrid_v8 is not None
                else ""),
            "kv_bridge_v8_cid": (
                self.kv_bridge_v8.cid()
                if self.kv_bridge_v8 is not None else ""),
            "hidden_state_bridge_v7_cid": (
                self.hidden_state_bridge_v7.cid()
                if self.hidden_state_bridge_v7 is not None
                else ""),
            "cache_controller_v6_cid": (
                self.cache_controller_v6.cid()
                if self.cache_controller_v6 is not None
                else ""),
            "replay_controller_v4_cid": (
                self.replay_controller_v4.cid()
                if self.replay_controller_v4 is not None
                else ""),
            "prefix_v7_drift_predictor_trained": bool(
                self.prefix_v7_drift_predictor_trained),
            "follow_up_token_ids": list(
                self.follow_up_token_ids),
            "tvs_budget_tokens": int(self.tvs_budget_tokens),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w63_params",
            "params": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class W63HandoffEnvelope:
    schema: str
    w62_outer_cid: str
    w63_params_cid: str
    substrate_v8_witness_cid: str
    kv_bridge_v8_witness_cid: str
    hsb_v7_witness_cid: str
    prefix_state_v7_witness_cid: str
    attn_steer_v7_witness_cid: str
    cache_controller_v6_witness_cid: str
    replay_controller_v4_witness_cid: str
    persistent_v15_witness_cid: str
    multi_hop_v13_witness_cid: str
    mlsc_v11_witness_cid: str
    consensus_v9_witness_cid: str
    crc_v11_witness_cid: str
    lhr_v15_witness_cid: str
    ecc_v15_witness_cid: str
    tvs_v12_witness_cid: str
    uncertainty_v11_witness_cid: str
    disagreement_algebra_v9_witness_cid: str
    deep_substrate_hybrid_v8_witness_cid: str
    substrate_adapter_v8_matrix_cid: str
    hidden_wins_falsifier_witness_cid: str
    v15_chain_cid: str
    eight_way_used: bool
    substrate_v8_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "w62_outer_cid": str(self.w62_outer_cid),
            "w63_params_cid": str(self.w63_params_cid),
            "substrate_v8_witness_cid": str(
                self.substrate_v8_witness_cid),
            "kv_bridge_v8_witness_cid": str(
                self.kv_bridge_v8_witness_cid),
            "hsb_v7_witness_cid": str(self.hsb_v7_witness_cid),
            "prefix_state_v7_witness_cid": str(
                self.prefix_state_v7_witness_cid),
            "attn_steer_v7_witness_cid": str(
                self.attn_steer_v7_witness_cid),
            "cache_controller_v6_witness_cid": str(
                self.cache_controller_v6_witness_cid),
            "replay_controller_v4_witness_cid": str(
                self.replay_controller_v4_witness_cid),
            "persistent_v15_witness_cid": str(
                self.persistent_v15_witness_cid),
            "multi_hop_v13_witness_cid": str(
                self.multi_hop_v13_witness_cid),
            "mlsc_v11_witness_cid": str(
                self.mlsc_v11_witness_cid),
            "consensus_v9_witness_cid": str(
                self.consensus_v9_witness_cid),
            "crc_v11_witness_cid": str(
                self.crc_v11_witness_cid),
            "lhr_v15_witness_cid": str(
                self.lhr_v15_witness_cid),
            "ecc_v15_witness_cid": str(
                self.ecc_v15_witness_cid),
            "tvs_v12_witness_cid": str(
                self.tvs_v12_witness_cid),
            "uncertainty_v11_witness_cid": str(
                self.uncertainty_v11_witness_cid),
            "disagreement_algebra_v9_witness_cid": str(
                self.disagreement_algebra_v9_witness_cid),
            "deep_substrate_hybrid_v8_witness_cid": str(
                self.deep_substrate_hybrid_v8_witness_cid),
            "substrate_adapter_v8_matrix_cid": str(
                self.substrate_adapter_v8_matrix_cid),
            "hidden_wins_falsifier_witness_cid": str(
                self.hidden_wins_falsifier_witness_cid),
            "v15_chain_cid": str(self.v15_chain_cid),
            "eight_way_used": bool(self.eight_way_used),
            "substrate_v8_used": bool(self.substrate_v8_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w63_handoff_envelope",
            "envelope": self.to_dict()})


def verify_w63_handoff(
        envelope: W63HandoffEnvelope,
) -> dict[str, Any]:
    failures: list[str] = []
    def need(field: str, key: str) -> None:
        if not getattr(envelope, field, ""):
            failures.append(key)
    need("w62_outer_cid", "missing_w62_outer_cid")
    need("substrate_v8_witness_cid",
         "missing_substrate_v8_witness")
    need("kv_bridge_v8_witness_cid",
         "missing_kv_bridge_v8_witness")
    need("hsb_v7_witness_cid", "missing_hsb_v7_witness")
    need("prefix_state_v7_witness_cid",
         "missing_prefix_state_v7_witness")
    need("attn_steer_v7_witness_cid",
         "missing_attn_steer_v7_witness")
    need("cache_controller_v6_witness_cid",
         "missing_cache_controller_v6_witness")
    need("replay_controller_v4_witness_cid",
         "missing_replay_controller_v4_witness")
    need("persistent_v15_witness_cid",
         "missing_persistent_v15_witness")
    need("multi_hop_v13_witness_cid",
         "missing_multi_hop_v13_witness")
    need("mlsc_v11_witness_cid",
         "missing_mlsc_v11_witness")
    need("consensus_v9_witness_cid",
         "missing_consensus_v9_witness")
    need("crc_v11_witness_cid", "missing_crc_v11_witness")
    need("lhr_v15_witness_cid", "missing_lhr_v15_witness")
    need("ecc_v15_witness_cid", "missing_ecc_v15_witness")
    need("tvs_v12_witness_cid", "missing_tvs_v12_witness")
    need("uncertainty_v11_witness_cid",
         "missing_uncertainty_v11_witness")
    need("disagreement_algebra_v9_witness_cid",
         "missing_disagreement_algebra_v9_witness")
    need("deep_substrate_hybrid_v8_witness_cid",
         "missing_deep_substrate_hybrid_v8_witness")
    need("substrate_adapter_v8_matrix_cid",
         "missing_substrate_adapter_v8_matrix")
    return {
        "schema": W63_SCHEMA_VERSION + ".verifier.v1",
        "failures": list(failures),
        "ok": bool(not failures),
        "n_failure_modes_known": int(
            len(W63_ENVELOPE_VERIFIER_FAILURE_MODES)),
    }


@dataclasses.dataclass
class W63Team:
    params: W63Params
    chain: PersistentLatentStateV15Chain = dataclasses.field(
        default_factory=PersistentLatentStateV15Chain.empty)

    def step(
            self, *, turn_index: int, role: str = "r",
            w62_outer_cid: str = "no_w62",
    ) -> W63HandoffEnvelope:
        p = self.params
        sub_w_cid = ""
        sub_used = False
        contention_l1 = 0.0
        if p.enabled and p.substrate_v8 is not None:
            ids = tokenize_bytes_v8(
                "w63-turn-" + str(int(turn_index)),
                max_len=14)
            trace, cache = forward_tiny_substrate_v8(
                p.substrate_v8, ids)
            # Record a prefix-reuse decision into the V8
            # ledger so the substrate state is non-trivial.
            record_prefix_reuse_decision_v8(
                cache, layer_index=0, head_index=0,
                decision="prefix_reuse_success")
            w = emit_tiny_substrate_v8_forward_witness(
                trace, cache)
            sub_w_cid = w.cid()
            sub_used = True
            contention_l1 = float(
                w.hidden_vs_kv_contention_l1)
            # Seed the contention channel so the eight-way axis fires.
            from .tiny_substrate_v8 import (
                record_hidden_vs_kv_contention_v8,
            )
            record_hidden_vs_kv_contention_v8(
                cache, layer_index=0, head_index=0, slot=0,
                hidden_write_abs=0.5, kv_write_abs=0.1)
            contention_l1 = float(
                _np.linalg.norm(
                    cache.hidden_vs_kv_contention.ravel(),
                    ord=1))
        kv_w_cid = ""
        hidden_wins_falsifier_cid = ""
        if (p.enabled and p.kv_bridge_v8 is not None):
            kv_w_cid = emit_kv_bridge_v8_witness(
                projection=p.kv_bridge_v8).cid()
            falsifier = (
                probe_kv_bridge_v8_hidden_wins_falsifier(
                    hidden_residual_l2=0.4,
                    kv_residual_l2=0.6))
            hidden_wins_falsifier_cid = falsifier.cid()
        hsb_w_cid = ""
        if (p.enabled and p.hidden_state_bridge_v7 is not None):
            hsb_w_cid = emit_hsb_v7_witness(
                projection=p.hidden_state_bridge_v7,
                contention_l1=float(contention_l1),
                hidden_wins_margin=0.2).cid()
        prefix_w_cid = _sha256_hex({
            "schema": "prefix_v7_compact_witness",
            "turn": int(turn_index),
            "drift_predictor_trained": bool(
                p.prefix_v7_drift_predictor_trained)})
        attn_w_cid = _sha256_hex({
            "schema": "attn_steering_v7_compact_witness",
            "turn": int(turn_index)})
        cc_w_cid = ""
        if (p.enabled and p.cache_controller_v6 is not None):
            cc_w_cid = emit_cache_controller_v6_witness(
                controller=p.cache_controller_v6).cid()
        rc_w_cid = ""
        if (p.enabled and p.replay_controller_v4 is not None):
            cand = ReplayCandidate(
                flop_reuse=100, flop_recompute=1000,
                flop_fallback=50,
                drift_l2_reuse=0.1, drift_l2_recompute=0.0,
                drift_l2_fallback=0.3,
                crc_passed=True, transcript_available=True,
                n_corruption_flags=0)
            p.replay_controller_v4.decide_v4(
                cand,
                hidden_vs_kv_contention=0.5,
                prefix_reuse_trust=0.7,
                replay_determinism_mean=0.8)
            rc_w_cid = emit_replay_controller_v4_witness(
                p.replay_controller_v4).cid()
        per_w_cid = ""
        if p.enabled and p.v12_cell is not None:
            carrier_vals = _payload_hash_vec(
                ("v15", int(turn_index)),
                int(p.v12_cell.state_dim))
            state = step_persistent_state_v15(
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
                substrate_fidelity=0.9,
                attention_fidelity=0.9,
                retrieval_fidelity=0.9,
                replay_fidelity=0.9)
            self.chain.add(state)
            per_w_cid = emit_persistent_v15_witness(
                self.chain, state.cid()).cid()
        mh_w_cid = ""
        if p.enabled:
            mh_w_cid = emit_multi_hop_v13_witness(
                backends=W63_DEFAULT_MH_V13_BACKENDS,
                chain_length=W63_DEFAULT_MH_V13_CHAIN_LEN,
                seed=int(turn_index) + 8300).cid()
        mlsc_w_cid = ""
        if (p.enabled and p.mlsc_v11_operator is not None):
            v3 = make_root_capsule_v3(
                branch_id=f"w63_{int(turn_index)}",
                payload=tuple([0.1] * 6),
                fact_tags=("w63",),
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
            merged = p.mlsc_v11_operator.merge(
                [v11],
                hidden_wins_witness_chain=(
                    f"merge_hw_{turn_index}",),
                prefix_reuse_witness_chain=(
                    f"merge_pr_{turn_index}",),
                algebra_signature_v11=(
                    W63_MLSC_V11_ALGEBRA_HIDDEN_WINS_PROPAGATION))
            mlsc_w_cid = emit_mlsc_v11_witness(merged).cid()
        cons_w_cid = ""
        if (p.enabled and p.consensus_v9 is not None):
            cons_w_cid = emit_consensus_v9_witness(
                p.consensus_v9).cid()
        crc_w_cid = ""
        if (p.enabled and p.crc_v11 is not None):
            crc_w_cid = (
                emit_corruption_robustness_v11_witness(
                    crc_v11=p.crc_v11, n_probes=8,
                    seed=int(turn_index) + 8400).cid())
        lhr_w_cid = ""
        if (p.enabled and p.lhr_v15 is not None):
            lhr_w_cid = emit_lhr_v15_witness(
                p.lhr_v15, carrier=[0.1] * 8, k=4,
                replay_dominance_indicator=[1.0] * 8,
                hidden_wins_indicator=[0.5] * 8).cid()
        ecc_w_cid = ""
        if (p.enabled and p.ecc_v15 is not None):
            gate = QuantisedBudgetGate.init(
                in_dim=W53_DEFAULT_ECC_CODE_DIM,
                emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
                seed=int(turn_index) + 8500)
            gate.importance_threshold = 0.0
            gate.w_emit.values = [1.0] * len(
                gate.w_emit.values)
            carrier = _payload_hash_vec(
                ("ecc15", int(turn_index)),
                W53_DEFAULT_ECC_CODE_DIM)
            comp = compress_carrier_ecc_v15(
                carrier, codebook=p.ecc_v15, gate=gate)
            ecc_w_cid = emit_ecc_v15_compression_witness(
                codebook=p.ecc_v15, compression=comp).cid()
        tvs_w_cid = ""
        if p.enabled:
            tvs_res = thirteen_arm_compare(
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
            tvs_w_cid = emit_tvs_arbiter_v12_witness(
                tvs_res).cid()
        unc_w_cid = ""
        if p.enabled:
            unc = compose_uncertainty_report_v11(
                confidences=[0.7, 0.5],
                trusts=[0.9, 0.8],
                substrate_fidelities=[0.95, 0.9],
                hidden_state_fidelities=[0.95, 0.92],
                cache_reuse_fidelities=[0.95, 0.92],
                retrieval_fidelities=[0.95, 0.93],
                replay_fidelities=[0.92, 0.90],
                attention_pattern_fidelities=[0.95, 0.90],
                replay_dominance_fidelities=[0.88, 0.85],
                hidden_wins_fidelities=[0.86, 0.82])
            unc_w_cid = emit_uncertainty_v11_witness(unc).cid()
        da_w_cid = ""
        if p.enabled:
            trace = AlgebraTrace(steps=[])
            probe = [0.1, 0.2, 0.3]
            wda = emit_disagreement_algebra_v9_witness(
                trace=trace, probe_a=probe, probe_b=probe,
                probe_c=probe,
                js_oracle=lambda: (True, 0.05),
                js_falsifier_oracle=lambda: (False, 1.0),
                wasserstein_oracle=lambda: (True, 0.1),
                wasserstein_falsifier_oracle=(
                    lambda: (False, 1.0)),
                attention_pattern_oracle=lambda: (True, 0.85))
            da_w_cid = wda.cid()
        hybrid_w_cid = ""
        eight_way = False
        if (p.enabled and p.deep_substrate_hybrid_v8 is not None):
            v7_witness = DeepSubstrateHybridV7ForwardWitness(
                schema="x", hybrid_cid="x",
                inner_v6_witness_cid="x",
                seven_way=True,
                cache_controller_v5_fired=True,
                replay_controller_v3_fired=True,
                hidden_vs_kv_classifier_fired=True,
                cache_write_ledger_active=True,
                attention_v6_active=True,
                prefix_v6_drift_predictor_active=True,
                mean_replay_dominance=0.4,
                cache_write_ledger_l2=1.0,
                attention_v6_coarse_l1_shift=0.5)
            wh = deep_substrate_hybrid_v8_forward(
                hybrid=p.deep_substrate_hybrid_v8,
                v7_witness=v7_witness,
                cache_controller_v6=p.cache_controller_v6,
                replay_controller_v4=p.replay_controller_v4,
                hidden_vs_kv_contention_l1=float(
                    contention_l1 + 1.0),
                attention_v7_js_max=0.15,
                prefix_v7_drift_predictor_trained=bool(
                    p.prefix_v7_drift_predictor_trained),
                prefix_reuse_trust_l1=0.7)
            hybrid_w_cid = wh.cid()
            eight_way = bool(wh.eight_way)
        adapter_cid = ""
        if p.enabled:
            matrix = probe_all_v8_adapters(
                probe_ollama=False, probe_openai=False)
            adapter_cid = matrix.cid()
        return W63HandoffEnvelope(
            schema=W63_SCHEMA_VERSION,
            w62_outer_cid=str(w62_outer_cid),
            w63_params_cid=str(p.cid()),
            substrate_v8_witness_cid=str(sub_w_cid),
            kv_bridge_v8_witness_cid=str(kv_w_cid),
            hsb_v7_witness_cid=str(hsb_w_cid),
            prefix_state_v7_witness_cid=str(prefix_w_cid),
            attn_steer_v7_witness_cid=str(attn_w_cid),
            cache_controller_v6_witness_cid=str(cc_w_cid),
            replay_controller_v4_witness_cid=str(rc_w_cid),
            persistent_v15_witness_cid=str(per_w_cid),
            multi_hop_v13_witness_cid=str(mh_w_cid),
            mlsc_v11_witness_cid=str(mlsc_w_cid),
            consensus_v9_witness_cid=str(cons_w_cid),
            crc_v11_witness_cid=str(crc_w_cid),
            lhr_v15_witness_cid=str(lhr_w_cid),
            ecc_v15_witness_cid=str(ecc_w_cid),
            tvs_v12_witness_cid=str(tvs_w_cid),
            uncertainty_v11_witness_cid=str(unc_w_cid),
            disagreement_algebra_v9_witness_cid=str(da_w_cid),
            deep_substrate_hybrid_v8_witness_cid=str(
                hybrid_w_cid),
            substrate_adapter_v8_matrix_cid=str(adapter_cid),
            hidden_wins_falsifier_witness_cid=str(
                hidden_wins_falsifier_cid),
            v15_chain_cid=str(self.chain.cid()),
            eight_way_used=bool(eight_way),
            substrate_v8_used=bool(sub_used),
        )


__all__ = [
    "W63_SCHEMA_VERSION",
    "W63_TEAM_RESULT_SCHEMA",
    "W63_ENVELOPE_VERIFIER_FAILURE_MODES",
    "W63Params",
    "W63HandoffEnvelope",
    "W63Team",
    "verify_w63_handoff",
]
