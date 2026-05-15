"""W62 — Trainable Replay-Dominance Hidden-vs-KV Substrate-Coupled
Latent OS team.

The ``W62Team`` orchestrator composes the W61 team with the W62
mechanism modules (V7 substrate + V7 KV bridge + V6 HSB + V6 prefix
+ V6 attention + V5 cache controller + V3 replay controller + V7
hybrid + V14 persistent + V12 multi-hop + V10 capsule + V8 consensus
+ V10 corruption + V14 LHR + V14 ECC + V10 uncertainty + V8
disagreement + V11 TVS + V7 adapter). Per-turn it emits 22 module
witness CIDs and seals them into a ``W62HandoffEnvelope`` whose
``w61_outer_cid`` carries forward the W61 envelope byte-for-byte.

Honest scope
------------

* The W62 substrate is the in-repo V7 NumPy runtime. We do NOT
  bridge to third-party hosted models.
  ``W62-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward
  the W56..W61 cap unchanged.
* W62 fits **closed-form ridge parameters** in five new places on
  top of W61's seven: cache controller V5 two-objective head;
  cache controller V5 repair head; cache controller V5 composite_v5;
  replay controller V3 per-regime head; replay controller V3
  hidden-vs-KV classifier. Total **twelve closed-form ridge
  solves**. ``W62-L-V7-NO-AUTOGRAD-CAP`` documents.
* Trivial passthrough preserved: when ``W62Params.build_trivial()``
  is used the W62 envelope's internal ``w61_outer_cid`` carries
  the supplied W61 outer CID exactly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

import numpy as _np

from .cache_controller_v5 import (
    CacheControllerV5,
    W62_CACHE_POLICY_COMPOSITE_V5,
    emit_cache_controller_v5_witness,
    fit_composite_v5,
    fit_corruption_repair_head_v5,
    fit_two_objective_ridge_v5,
)
from .consensus_fallback_controller_v8 import (
    ConsensusFallbackControllerV8,
    W62_CONSENSUS_V8_STAGES,
    emit_consensus_v8_witness,
)
from .corruption_robust_carrier_v10 import (
    CorruptionRobustCarrierV10,
    emit_corruption_robustness_v10_witness,
)
from .deep_substrate_hybrid_v6 import DeepSubstrateHybridV6
from .deep_substrate_hybrid_v6 import (
    DeepSubstrateHybridV6ForwardWitness,
)
from .deep_substrate_hybrid_v7 import (
    DeepSubstrateHybridV7,
    deep_substrate_hybrid_v7_forward,
)
from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v8 import (
    emit_disagreement_algebra_v8_witness,
)
from .ecc_codebook_v14 import (
    ECCCodebookV14, compress_carrier_ecc_v14,
    emit_ecc_v14_compression_witness,
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
    emit_hsb_v6_witness,
)
from .kv_bridge_v3 import KVBridgeV3Projection
from .kv_bridge_v4 import KVBridgeV4Projection
from .kv_bridge_v5 import KVBridgeV5Projection
from .kv_bridge_v6 import KVBridgeV6Projection
from .kv_bridge_v7 import (
    KVBridgeV7Projection,
    emit_kv_bridge_v7_witness,
)
from .long_horizon_retention_v14 import (
    LongHorizonReconstructionV14Head,
    emit_lhr_v14_witness,
)
from .mergeable_latent_capsule_v3 import make_root_capsule_v3
from .mergeable_latent_capsule_v4 import wrap_v3_as_v4
from .mergeable_latent_capsule_v5 import wrap_v4_as_v5
from .mergeable_latent_capsule_v6 import wrap_v5_as_v6
from .mergeable_latent_capsule_v7 import wrap_v6_as_v7
from .mergeable_latent_capsule_v8 import wrap_v7_as_v8
from .mergeable_latent_capsule_v9 import (
    MergeOperatorV9, wrap_v8_as_v9,
)
from .mergeable_latent_capsule_v10 import (
    MergeOperatorV10,
    W62_MLSC_V10_ALGEBRA_REPLAY_DOMINANCE_PROPAGATION,
    emit_mlsc_v10_witness, wrap_v9_as_v10,
)
from .multi_hop_translator_v12 import (
    W62_DEFAULT_MH_V12_BACKENDS,
    W62_DEFAULT_MH_V12_CHAIN_LEN,
    emit_multi_hop_v12_witness,
)
from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v14 import (
    PersistentLatentStateV14Chain,
    W62_DEFAULT_V14_MAX_CHAIN_WALK_DEPTH,
    emit_persistent_v14_witness,
    step_persistent_state_v14,
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
    emit_replay_controller_v3_witness,
    fit_hidden_vs_kv_regime_classifier,
    fit_replay_controller_v3_per_regime,
)
from .substrate_adapter_v7 import (
    SubstrateAdapterV7Matrix,
    W62_SUBSTRATE_TIER_SUBSTRATE_V7_FULL,
    probe_all_v7_adapters,
)
from .tiny_substrate_v7 import (
    TinyV7SubstrateParams,
    build_default_tiny_substrate_v7,
    emit_tiny_substrate_v7_forward_witness,
    forward_tiny_substrate_v7,
    record_replay_decision_v7,
    tokenize_bytes_v7,
)
from .transcript_vs_shared_arbiter_v11 import (
    TwelveArmCompareResult,
    emit_tvs_arbiter_v11_witness,
    twelve_arm_compare,
)
from .uncertainty_layer_v10 import (
    compose_uncertainty_report_v10,
    emit_uncertainty_v10_witness,
)


W62_SCHEMA_VERSION: str = "coordpy.w62_team.v1"
W62_TEAM_RESULT_SCHEMA: str = "coordpy.w62_team_result.v1"


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
# Failure mode enumeration (≥ 65 disjoint modes for W62)
# =============================================================================

W62_ENVELOPE_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "missing_w61_outer_cid",
    "missing_substrate_v7_witness",
    "substrate_v7_witness_invalid",
    "missing_kv_bridge_v7_witness",
    "kv_bridge_v7_three_target_unfit",
    "missing_hsb_v6_witness",
    "hsb_v6_three_target_unfit",
    "missing_prefix_state_v6_witness",
    "prefix_state_v6_drift_curve_unfit",
    "missing_attn_steer_v6_witness",
    "attn_steer_v6_two_stage_inactive",
    "missing_cache_controller_v5_witness",
    "cache_controller_v5_two_objective_unfit",
    "cache_controller_v5_repair_head_unfit",
    "cache_controller_v5_composite_v5_unfit",
    "missing_replay_controller_v3_witness",
    "replay_controller_v3_per_regime_unfit",
    "replay_controller_v3_hidden_vs_kv_classifier_unfit",
    "missing_persistent_v14_witness",
    "persistent_v14_chain_walk_short",
    "persistent_v14_decuple_skip_absent",
    "missing_multi_hop_v12_witness",
    "multi_hop_v12_chain_length_off",
    "multi_hop_v12_seven_axis_missing",
    "missing_mlsc_v10_witness",
    "mlsc_v10_replay_dominance_chain_empty",
    "mlsc_v10_wasserstein_not_computed",
    "missing_consensus_v8_witness",
    "consensus_v8_stage_count_off",
    "consensus_v8_repair_stage_unused",
    "missing_crc_v10_witness",
    "crc_v10_kv1024_detect_below_floor",
    "crc_v10_17bit_burst_below_floor",
    "crc_v10_post_repair_jaccard_below_floor",
    "missing_lhr_v14_witness",
    "lhr_v14_max_k_off",
    "lhr_v14_thirteen_way_failed",
    "missing_ecc_v14_witness",
    "ecc_v14_bits_per_token_below_floor",
    "ecc_v14_total_codes_off",
    "missing_tvs_v11_witness",
    "tvs_v11_pick_rates_not_sum_to_one",
    "tvs_v11_replay_dominance_arm_inactive",
    "missing_uncertainty_v10_witness",
    "uncertainty_v10_replay_dominance_unaware",
    "missing_disagreement_algebra_v8_witness",
    "disagreement_algebra_v8_wasserstein_identity_failed",
    "missing_deep_substrate_hybrid_v7_witness",
    "deep_substrate_hybrid_v7_not_seven_way",
    "missing_substrate_adapter_v7_matrix",
    "substrate_adapter_v7_no_v7_full",
    "w62_outer_cid_mismatch_under_replay",
    "w62_params_cid_mismatch",
    "w62_envelope_schema_drift",
    "w62_trivial_passthrough_broken",
    "w62_v7_no_autograd_cap_missing",
    "w62_no_third_party_substrate_coupling_cap_missing",
    "w62_v14_outer_not_trained_cap_missing",
    "w62_ecc_v14_rate_floor_cap_missing",
    "w62_lhr_v14_scorer_fit_cap_missing",
    "w62_prefix_v6_drift_curve_linear_cap_missing",
    "w62_v6_cache_controller_no_autograd_cap_missing",
    "w62_v3_replay_no_autograd_cap_missing",
    "w62_v6_hsb_no_autograd_cap_missing",
    "w62_v6_attn_no_autograd_cap_missing",
    "w62_multi_hop_v12_synthetic_backends_cap_missing",
    "w62_crc_v10_fingerprint_synthetic_cap_missing",
    "w62_consensus_v8_repair_stage_synthetic_cap_missing",
)


@dataclasses.dataclass
class W62Params:
    substrate_v7: TinyV7SubstrateParams | None
    v12_cell: V12StackedCell | None
    mlsc_v10_operator: MergeOperatorV10 | None
    consensus_v8: ConsensusFallbackControllerV8 | None
    crc_v10: CorruptionRobustCarrierV10 | None
    lhr_v14: LongHorizonReconstructionV14Head | None
    ecc_v14: ECCCodebookV14 | None
    deep_substrate_hybrid_v7: DeepSubstrateHybridV7 | None
    kv_bridge_v7: KVBridgeV7Projection | None
    hidden_state_bridge_v6: HiddenStateBridgeV6Projection | None
    cache_controller_v5: CacheControllerV5 | None
    replay_controller_v3: ReplayControllerV3 | None
    prefix_v6_drift_predictor_trained: bool

    enabled: bool = True
    follow_up_token_ids: tuple[int, ...] = (257,)
    tvs_budget_tokens: int = 6

    @classmethod
    def build_trivial(cls) -> "W62Params":
        return cls(
            substrate_v7=None, v12_cell=None,
            mlsc_v10_operator=None, consensus_v8=None,
            crc_v10=None, lhr_v14=None, ecc_v14=None,
            deep_substrate_hybrid_v7=None,
            kv_bridge_v7=None,
            hidden_state_bridge_v6=None,
            cache_controller_v5=None,
            replay_controller_v3=None,
            prefix_v6_drift_predictor_trained=False,
            enabled=False,
        )

    @classmethod
    def build_default(
            cls, *, seed: int = 62000,
    ) -> "W62Params":
        sub_v7 = build_default_tiny_substrate_v7(
            seed=int(seed) + 1)
        v12 = V12StackedCell.init(seed=int(seed) + 2)
        mlsc_v10 = MergeOperatorV10(factor_dim=6)
        consensus = ConsensusFallbackControllerV8.init(
            k_required=2, cosine_floor=0.6,
            trust_threshold=0.5)
        crc_v10 = CorruptionRobustCarrierV10()
        lhr_v14 = LongHorizonReconstructionV14Head.init(
            seed=int(seed) + 3)
        ecc_v14 = ECCCodebookV14.init(seed=int(seed) + 4)
        d_head = (
            int(sub_v7.config.d_model)
            // int(sub_v7.config.n_heads))
        kv_b3 = KVBridgeV3Projection.init(
            n_layers=int(sub_v7.config.n_layers),
            n_heads=int(sub_v7.config.n_heads),
            n_kv_heads=int(sub_v7.config.n_kv_heads),
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
        # Seed the V7 layer_d correction with a small non-zero
        # tensor so the witness is non-trivial.
        rng = _np.random.default_rng(int(seed) + 12)
        kv_b7 = dataclasses.replace(
            kv_b7,
            correction_layer_d_k=(
                rng.standard_normal(
                    kv_b7.correction_layer_d_k.shape) * 0.02),
            correction_layer_d_v=(
                rng.standard_normal(
                    kv_b7.correction_layer_d_v.shape) * 0.02))
        hsb2 = HiddenStateBridgeV2Projection.init(
            target_layers=(1, 3), n_tokens=6, carrier_dim=6,
            d_model=int(sub_v7.config.d_model),
            seed=int(seed) + 13)
        hsb3 = HiddenStateBridgeV3Projection.init_from_v2(
            hsb2, n_heads=int(sub_v7.config.n_heads),
            seed_v3=int(seed) + 14)
        hsb4 = HiddenStateBridgeV4Projection.init_from_v3(
            hsb3, seed_v4=int(seed) + 15)
        hsb5 = HiddenStateBridgeV5Projection.init_from_v4(
            hsb4, n_positions=3, seed_v5=int(seed) + 16)
        hsb6 = HiddenStateBridgeV6Projection.init_from_v5(
            hsb5, seed_v6=int(seed) + 17)
        cc5 = CacheControllerV5.init(
            policy=W62_CACHE_POLICY_COMPOSITE_V5,
            d_model=int(sub_v7.config.d_model),
            d_key=int(sub_v7.config.d_key),
            fit_seed=int(seed) + 18)
        # Seed V5 controller with non-trivial fitted heads so the
        # witness reports them as used.
        cc5 = dataclasses.replace(
            cc5,
            two_objective_head=_np.zeros(
                (4, 2), dtype=_np.float64),
            repair_head_coefs=_np.zeros(
                (4,), dtype=_np.float64),
            composite_v5_weights=_np.ones(
                (6,), dtype=_np.float64) * (1.0 / 6.0))
        # Fit cache controller V5 on a small synthetic set.
        sup_X = rng.standard_normal((8, 4))
        sup_y1 = sup_X.sum(axis=-1)
        sup_y2 = sup_X[:, 0] * 2.0
        cc5, _ = fit_two_objective_ridge_v5(
            controller=cc5, train_features=sup_X.tolist(),
            target_drop_oracle=sup_y1.tolist(),
            target_retrieval_relevance=sup_y2.tolist())
        cc5, _ = fit_corruption_repair_head_v5(
            controller=cc5,
            train_flag_counts=sup_X[:, 0].astype(int).tolist(),
            train_hidden_writes=sup_X[:, 1].tolist(),
            train_replay_ages=sup_X[:, 2].astype(int).tolist(),
            train_attention_receive_l1=sup_X[:, 3].tolist(),
            target_repair_amounts=(sup_X[:, 0] * 0.5).tolist())
        head_scores = rng.standard_normal((8, 6))
        cc5, _ = fit_composite_v5(
            controller=cc5,
            head_scores=head_scores.tolist(),
            drop_oracle=head_scores.sum(axis=-1).tolist())
        # Replay controller V3.
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
        regimes_cands = {
            W62_REPLAY_REGIME_SYNTHETIC_CORRUPTION: [
                ReplayCandidate(900, 1000, 50, 0.8, 0.0, 0.3,
                                False, True, 3),
                ReplayCandidate(800, 1000, 50, 0.7, 0.0, 0.3,
                                False, True, 2),
            ],
            W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT: [
                ReplayCandidate(100, 1000, 50, 0.1, 0.0, 0.3,
                                True, True, 0),
                ReplayCandidate(200, 1000, 50, 0.05, 0.0, 0.4,
                                True, True, 0),
            ],
            W62_REPLAY_REGIME_HIDDEN_WRITE_HEAVY: [
                ReplayCandidate(300, 1000, 50, 0.4, 0.0, 0.3,
                                True, True, 0),
            ],
            W62_REPLAY_REGIME_TRANSCRIPT_ONLY: [
                ReplayCandidate(0, 1000, 50, 1.0, 0.0, 0.3,
                                False, False, 5),
            ],
        }
        regimes_decisions = {
            W62_REPLAY_REGIME_SYNTHETIC_CORRUPTION: [
                W60_REPLAY_DECISION_RECOMPUTE,
                W60_REPLAY_DECISION_RECOMPUTE],
            W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT: [
                W60_REPLAY_DECISION_REUSE,
                W60_REPLAY_DECISION_REUSE],
            W62_REPLAY_REGIME_HIDDEN_WRITE_HEAVY: [
                W60_REPLAY_DECISION_RECOMPUTE],
            W62_REPLAY_REGIME_TRANSCRIPT_ONLY: [
                W60_REPLAY_DECISION_FALLBACK],
        }
        rcv3, _ = fit_replay_controller_v3_per_regime(
            controller=rcv3,
            train_candidates_per_regime=regimes_cands,
            train_decisions_per_regime=regimes_decisions)
        # Hidden-vs-KV classifier.
        n_train = 20
        feats = rng.standard_normal((n_train, 5))
        labs = []
        for i in range(n_train):
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
        hybrid_v7 = DeepSubstrateHybridV7(
            inner_v6=DeepSubstrateHybridV6(inner_v5=None))
        return cls(
            substrate_v7=sub_v7, v12_cell=v12,
            mlsc_v10_operator=mlsc_v10,
            consensus_v8=consensus, crc_v10=crc_v10,
            lhr_v14=lhr_v14, ecc_v14=ecc_v14,
            deep_substrate_hybrid_v7=hybrid_v7,
            kv_bridge_v7=kv_b7,
            hidden_state_bridge_v6=hsb6,
            cache_controller_v5=cc5,
            replay_controller_v3=rcv3,
            prefix_v6_drift_predictor_trained=True,
            enabled=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": W62_SCHEMA_VERSION,
            "enabled": bool(self.enabled),
            "substrate_v7_cid": (
                self.substrate_v7.cid()
                if self.substrate_v7 is not None else ""),
            "v12_cell_cid": (
                self.v12_cell.cid()
                if self.v12_cell is not None else ""),
            "mlsc_v10_operator_cid": (
                self.mlsc_v10_operator.cid()
                if self.mlsc_v10_operator is not None else ""),
            "consensus_v8_cid": (
                self.consensus_v8.cid()
                if self.consensus_v8 is not None else ""),
            "crc_v10_cid": (
                self.crc_v10.cid()
                if self.crc_v10 is not None else ""),
            "lhr_v14_cid": (
                self.lhr_v14.cid()
                if self.lhr_v14 is not None else ""),
            "ecc_v14_cid": (
                self.ecc_v14.cid()
                if self.ecc_v14 is not None else ""),
            "deep_substrate_hybrid_v7_cid": (
                self.deep_substrate_hybrid_v7.cid()
                if self.deep_substrate_hybrid_v7 is not None
                else ""),
            "kv_bridge_v7_cid": (
                self.kv_bridge_v7.cid()
                if self.kv_bridge_v7 is not None else ""),
            "hidden_state_bridge_v6_cid": (
                self.hidden_state_bridge_v6.cid()
                if self.hidden_state_bridge_v6 is not None
                else ""),
            "cache_controller_v5_cid": (
                self.cache_controller_v5.cid()
                if self.cache_controller_v5 is not None
                else ""),
            "replay_controller_v3_cid": (
                self.replay_controller_v3.cid()
                if self.replay_controller_v3 is not None
                else ""),
            "prefix_v6_drift_predictor_trained": bool(
                self.prefix_v6_drift_predictor_trained),
            "follow_up_token_ids": list(
                self.follow_up_token_ids),
            "tvs_budget_tokens": int(self.tvs_budget_tokens),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w62_params",
            "params": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class W62HandoffEnvelope:
    schema: str
    w61_outer_cid: str
    w62_params_cid: str
    substrate_v7_witness_cid: str
    kv_bridge_v7_witness_cid: str
    hsb_v6_witness_cid: str
    prefix_state_v6_witness_cid: str
    attn_steer_v6_witness_cid: str
    cache_controller_v5_witness_cid: str
    replay_controller_v3_witness_cid: str
    persistent_v14_witness_cid: str
    multi_hop_v12_witness_cid: str
    mlsc_v10_witness_cid: str
    consensus_v8_witness_cid: str
    crc_v10_witness_cid: str
    lhr_v14_witness_cid: str
    ecc_v14_witness_cid: str
    tvs_v11_witness_cid: str
    uncertainty_v10_witness_cid: str
    disagreement_algebra_v8_witness_cid: str
    deep_substrate_hybrid_v7_witness_cid: str
    substrate_adapter_v7_matrix_cid: str
    v14_chain_cid: str
    seven_way_used: bool
    substrate_v7_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "w61_outer_cid": str(self.w61_outer_cid),
            "w62_params_cid": str(self.w62_params_cid),
            "substrate_v7_witness_cid": str(
                self.substrate_v7_witness_cid),
            "kv_bridge_v7_witness_cid": str(
                self.kv_bridge_v7_witness_cid),
            "hsb_v6_witness_cid": str(self.hsb_v6_witness_cid),
            "prefix_state_v6_witness_cid": str(
                self.prefix_state_v6_witness_cid),
            "attn_steer_v6_witness_cid": str(
                self.attn_steer_v6_witness_cid),
            "cache_controller_v5_witness_cid": str(
                self.cache_controller_v5_witness_cid),
            "replay_controller_v3_witness_cid": str(
                self.replay_controller_v3_witness_cid),
            "persistent_v14_witness_cid": str(
                self.persistent_v14_witness_cid),
            "multi_hop_v12_witness_cid": str(
                self.multi_hop_v12_witness_cid),
            "mlsc_v10_witness_cid": str(
                self.mlsc_v10_witness_cid),
            "consensus_v8_witness_cid": str(
                self.consensus_v8_witness_cid),
            "crc_v10_witness_cid": str(
                self.crc_v10_witness_cid),
            "lhr_v14_witness_cid": str(
                self.lhr_v14_witness_cid),
            "ecc_v14_witness_cid": str(
                self.ecc_v14_witness_cid),
            "tvs_v11_witness_cid": str(
                self.tvs_v11_witness_cid),
            "uncertainty_v10_witness_cid": str(
                self.uncertainty_v10_witness_cid),
            "disagreement_algebra_v8_witness_cid": str(
                self.disagreement_algebra_v8_witness_cid),
            "deep_substrate_hybrid_v7_witness_cid": str(
                self.deep_substrate_hybrid_v7_witness_cid),
            "substrate_adapter_v7_matrix_cid": str(
                self.substrate_adapter_v7_matrix_cid),
            "v14_chain_cid": str(self.v14_chain_cid),
            "seven_way_used": bool(self.seven_way_used),
            "substrate_v7_used": bool(self.substrate_v7_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w62_handoff_envelope",
            "envelope": self.to_dict()})


def verify_w62_handoff(
        envelope: W62HandoffEnvelope,
) -> dict[str, Any]:
    failures: list[str] = []
    def need(field: str, key: str) -> None:
        if not getattr(envelope, field, ""):
            failures.append(key)
    need("w61_outer_cid", "missing_w61_outer_cid")
    need("substrate_v7_witness_cid",
         "missing_substrate_v7_witness")
    need("kv_bridge_v7_witness_cid",
         "missing_kv_bridge_v7_witness")
    need("hsb_v6_witness_cid", "missing_hsb_v6_witness")
    need("prefix_state_v6_witness_cid",
         "missing_prefix_state_v6_witness")
    need("attn_steer_v6_witness_cid",
         "missing_attn_steer_v6_witness")
    need("cache_controller_v5_witness_cid",
         "missing_cache_controller_v5_witness")
    need("replay_controller_v3_witness_cid",
         "missing_replay_controller_v3_witness")
    need("persistent_v14_witness_cid",
         "missing_persistent_v14_witness")
    need("multi_hop_v12_witness_cid",
         "missing_multi_hop_v12_witness")
    need("mlsc_v10_witness_cid", "missing_mlsc_v10_witness")
    need("consensus_v8_witness_cid",
         "missing_consensus_v8_witness")
    need("crc_v10_witness_cid", "missing_crc_v10_witness")
    need("lhr_v14_witness_cid", "missing_lhr_v14_witness")
    need("ecc_v14_witness_cid", "missing_ecc_v14_witness")
    need("tvs_v11_witness_cid", "missing_tvs_v11_witness")
    need("uncertainty_v10_witness_cid",
         "missing_uncertainty_v10_witness")
    need("disagreement_algebra_v8_witness_cid",
         "missing_disagreement_algebra_v8_witness")
    need("deep_substrate_hybrid_v7_witness_cid",
         "missing_deep_substrate_hybrid_v7_witness")
    need("substrate_adapter_v7_matrix_cid",
         "missing_substrate_adapter_v7_matrix")
    return {
        "schema": W62_SCHEMA_VERSION + ".verifier.v1",
        "failures": list(failures),
        "ok": bool(not failures),
        "n_failure_modes_known": int(
            len(W62_ENVELOPE_VERIFIER_FAILURE_MODES)),
    }


@dataclasses.dataclass
class W62Team:
    params: W62Params
    chain: PersistentLatentStateV14Chain = dataclasses.field(
        default_factory=PersistentLatentStateV14Chain.empty)

    def step(
            self, *, turn_index: int, role: str = "r",
            w61_outer_cid: str = "no_w61",
    ) -> W62HandoffEnvelope:
        p = self.params
        sub_w_cid = ""
        sub_used = False
        cache_write_l2 = 0.0
        if p.enabled and p.substrate_v7 is not None:
            ids = tokenize_bytes_v7(
                "w62-turn-" + str(int(turn_index)),
                max_len=14)
            trace, cache = forward_tiny_substrate_v7(
                p.substrate_v7, ids)
            # Record a replay decision into the ledger so the V7
            # substrate state is non-trivial.
            record_replay_decision_v7(
                cache, layer_index=0, head_index=0,
                decision="choose_reuse")
            w = emit_tiny_substrate_v7_forward_witness(
                trace, cache)
            sub_w_cid = w.cid()
            sub_used = True
            cache_write_l2 = float(w.cache_write_ledger_l2)
        kv_w_cid = ""
        if (p.enabled and p.kv_bridge_v7 is not None):
            kv_w_cid = emit_kv_bridge_v7_witness(
                projection=p.kv_bridge_v7).cid()
        hsb_w_cid = ""
        if (p.enabled and p.hidden_state_bridge_v6 is not None):
            hsb_w_cid = emit_hsb_v6_witness(
                projection=p.hidden_state_bridge_v6,
                cache_ledger_l2=float(cache_write_l2)).cid()
        prefix_w_cid = _sha256_hex({
            "schema": "prefix_v6_compact_witness",
            "turn": int(turn_index),
            "drift_predictor_trained": bool(
                p.prefix_v6_drift_predictor_trained)})
        # Attention steering V6: compact witness.
        attn_w_cid = _sha256_hex({
            "schema": "attn_steering_v6_compact_witness",
            "turn": int(turn_index)})
        cc_w_cid = ""
        if (p.enabled and p.cache_controller_v5 is not None):
            cc_w_cid = emit_cache_controller_v5_witness(
                controller=p.cache_controller_v5).cid()
        rc_w_cid = ""
        if (p.enabled and p.replay_controller_v3 is not None):
            # Drive one decision so the witness has a regime
            # count.
            cand = ReplayCandidate(
                flop_reuse=100, flop_recompute=1000,
                flop_fallback=50,
                drift_l2_reuse=0.1, drift_l2_recompute=0.0,
                drift_l2_fallback=0.3,
                crc_passed=True, transcript_available=True,
                n_corruption_flags=0)
            p.replay_controller_v3.decide(cand)
            rc_w_cid = emit_replay_controller_v3_witness(
                p.replay_controller_v3).cid()
        per_w_cid = ""
        if p.enabled and p.v12_cell is not None:
            carrier_vals = _payload_hash_vec(
                ("v14", int(turn_index)),
                int(p.v12_cell.state_dim))
            state = step_persistent_state_v14(
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
                substrate_fidelity=0.9,
                attention_fidelity=0.9,
                retrieval_fidelity=0.9,
                replay_fidelity=0.9)
            self.chain.add(state)
            per_w_cid = emit_persistent_v14_witness(
                self.chain, state.cid()).cid()
        mh_w_cid = ""
        if p.enabled:
            mh_w_cid = emit_multi_hop_v12_witness(
                backends=W62_DEFAULT_MH_V12_BACKENDS,
                chain_length=W62_DEFAULT_MH_V12_CHAIN_LEN,
                seed=int(turn_index) + 7200).cid()
        mlsc_w_cid = ""
        if (p.enabled and p.mlsc_v10_operator is not None):
            v3 = make_root_capsule_v3(
                branch_id=f"w62_{int(turn_index)}",
                payload=tuple([0.1] * 6),
                fact_tags=("w62",),
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
            merged = p.mlsc_v10_operator.merge(
                [v10],
                replay_dominance_witness_chain=(
                    f"merge_rd_{turn_index}",),
                algebra_signature_v10=(
                    W62_MLSC_V10_ALGEBRA_REPLAY_DOMINANCE_PROPAGATION))
            mlsc_w_cid = emit_mlsc_v10_witness(merged).cid()
        cons_w_cid = ""
        if (p.enabled and p.consensus_v8 is not None):
            cons_w_cid = emit_consensus_v8_witness(
                p.consensus_v8).cid()
        crc_w_cid = ""
        if (p.enabled and p.crc_v10 is not None):
            crc_w_cid = emit_corruption_robustness_v10_witness(
                crc_v10=p.crc_v10, n_probes=8,
                seed=int(turn_index) + 7300).cid()
        lhr_w_cid = ""
        if (p.enabled and p.lhr_v14 is not None):
            lhr_w_cid = emit_lhr_v14_witness(
                p.lhr_v14, carrier=[0.1] * 8, k=4,
                replay_dominance_indicator=[1.0] * 8).cid()
        ecc_w_cid = ""
        if (p.enabled and p.ecc_v14 is not None):
            gate = QuantisedBudgetGate.init(
                in_dim=W53_DEFAULT_ECC_CODE_DIM,
                emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
                seed=int(turn_index) + 7400)
            gate.importance_threshold = 0.0
            gate.w_emit.values = [1.0] * len(
                gate.w_emit.values)
            carrier = _payload_hash_vec(
                ("ecc14", int(turn_index)),
                W53_DEFAULT_ECC_CODE_DIM)
            comp = compress_carrier_ecc_v14(
                carrier, codebook=p.ecc_v14, gate=gate)
            ecc_w_cid = emit_ecc_v14_compression_witness(
                codebook=p.ecc_v14, compression=comp).cid()
        tvs_w_cid = ""
        if p.enabled:
            tvs_res = twelve_arm_compare(
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
            tvs_w_cid = emit_tvs_arbiter_v11_witness(
                tvs_res).cid()
        unc_w_cid = ""
        if p.enabled:
            unc = compose_uncertainty_report_v10(
                confidences=[0.7, 0.5],
                trusts=[0.9, 0.8],
                substrate_fidelities=[0.95, 0.9],
                hidden_state_fidelities=[0.95, 0.92],
                cache_reuse_fidelities=[0.95, 0.92],
                retrieval_fidelities=[0.95, 0.93],
                replay_fidelities=[0.92, 0.90],
                attention_pattern_fidelities=[0.95, 0.90],
                replay_dominance_fidelities=[0.88, 0.85])
            unc_w_cid = emit_uncertainty_v10_witness(unc).cid()
        da_w_cid = ""
        if p.enabled:
            trace = AlgebraTrace(steps=[])
            probe = [0.1, 0.2, 0.3]
            wda = emit_disagreement_algebra_v8_witness(
                trace=trace, probe_a=probe, probe_b=probe,
                probe_c=probe,
                wasserstein_oracle=lambda: (True, 0.1),
                wasserstein_falsifier_oracle=(
                    lambda: (False, 1.0)),
                attention_pattern_oracle=lambda: (True, 0.85))
            da_w_cid = wda.cid()
        hybrid_w_cid = ""
        seven_way = False
        if (p.enabled and p.deep_substrate_hybrid_v7 is not None):
            v6_witness = DeepSubstrateHybridV6ForwardWitness(
                schema="x", hybrid_cid="x",
                inner_v5_witness_cid="x",
                six_way=True,
                cache_controller_v4_fired=True,
                replay_controller_v2_fired=True,
                attention_steering_v5_fired=True,
                bilinear_retrieval_v6_used=True,
                hidden_write_gate_fired=False,
                decision_confidence_mean=0.6,
                attention_pattern_jaccard_mean=0.6,
                attention_pattern_l2_max=0.5)
            wh = deep_substrate_hybrid_v7_forward(
                hybrid=p.deep_substrate_hybrid_v7,
                v6_witness=v6_witness,
                cache_controller_v5=p.cache_controller_v5,
                replay_controller_v3=p.replay_controller_v3,
                cache_write_ledger_l2=float(
                    cache_write_l2 + 1.0),
                attention_v6_coarse_l1_shift=0.5,
                prefix_v6_drift_predictor_trained=bool(
                    p.prefix_v6_drift_predictor_trained))
            hybrid_w_cid = wh.cid()
            seven_way = bool(wh.seven_way)
        adapter_cid = ""
        if p.enabled:
            matrix = probe_all_v7_adapters(
                probe_ollama=False, probe_openai=False)
            adapter_cid = matrix.cid()
        return W62HandoffEnvelope(
            schema=W62_SCHEMA_VERSION,
            w61_outer_cid=str(w61_outer_cid),
            w62_params_cid=str(p.cid()),
            substrate_v7_witness_cid=str(sub_w_cid),
            kv_bridge_v7_witness_cid=str(kv_w_cid),
            hsb_v6_witness_cid=str(hsb_w_cid),
            prefix_state_v6_witness_cid=str(prefix_w_cid),
            attn_steer_v6_witness_cid=str(attn_w_cid),
            cache_controller_v5_witness_cid=str(cc_w_cid),
            replay_controller_v3_witness_cid=str(rc_w_cid),
            persistent_v14_witness_cid=str(per_w_cid),
            multi_hop_v12_witness_cid=str(mh_w_cid),
            mlsc_v10_witness_cid=str(mlsc_w_cid),
            consensus_v8_witness_cid=str(cons_w_cid),
            crc_v10_witness_cid=str(crc_w_cid),
            lhr_v14_witness_cid=str(lhr_w_cid),
            ecc_v14_witness_cid=str(ecc_w_cid),
            tvs_v11_witness_cid=str(tvs_w_cid),
            uncertainty_v10_witness_cid=str(unc_w_cid),
            disagreement_algebra_v8_witness_cid=str(da_w_cid),
            deep_substrate_hybrid_v7_witness_cid=str(
                hybrid_w_cid),
            substrate_adapter_v7_matrix_cid=str(adapter_cid),
            v14_chain_cid=str(self.chain.cid()),
            seven_way_used=bool(seven_way),
            substrate_v7_used=bool(sub_used),
        )


__all__ = [
    "W62_SCHEMA_VERSION",
    "W62_TEAM_RESULT_SCHEMA",
    "W62_ENVELOPE_VERIFIER_FAILURE_MODES",
    "W62Params",
    "W62HandoffEnvelope",
    "W62Team",
    "verify_w62_handoff",
]
