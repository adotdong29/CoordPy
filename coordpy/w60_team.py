"""W60 — Trainable Cache-Control Substrate-Coupled Latent OS.

The ``W60Team`` orchestrator composes the W59 team with the W60
mechanism modules. Per-turn it emits 19 module witness CIDs and
seals them into a ``W60HandoffEnvelope`` whose ``w59_outer_cid``
carries forward the W59 envelope.

Honest scope
------------

* The W60 substrate is the in-repo V5 NumPy runtime. We do NOT
  bridge to third-party hosted models. ``W60-L-NO-THIRD-PARTY-
  SUBSTRATE-COUPLING-CAP`` carries forward the W59 cap unchanged.
* W60 fits **closed-form ridge parameters** in five places: the
  KV bridge V5 multi-direction correction, the HSB V4 per-(layer,
  head) target-logit fit, the cache controller V3
  ``learned_attention_receive`` head, the cache controller V3
  ``trained_eviction`` head, and the LHR V12 two-layer retention
  scorer. NOT end-to-end backprop. ``W60-L-V5-NO-AUTOGRAD-CAP``
  is load-bearing.
* Trivial passthrough is preserved byte-for-byte: when
  ``W60Params.build_trivial()`` is used the W60 envelope's
  internal ``w59_outer_cid`` carries the supplied W59 outer CID
  exactly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

import numpy as _np

from .attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
)
from .attention_steering_bridge_v4 import (
    steer_attention_and_measure_v4,
)
from .cache_controller_v3 import (
    CacheControllerV3,
    W60_CACHE_POLICY_COMPOSITE_V3,
    W60_CACHE_POLICY_LEARNED_ATTENTION_RECEIVE,
    apply_cache_controller_v3_and_measure,
)
from .consensus_fallback_controller_v6 import (
    ConsensusFallbackControllerV6,
    W60_CONSENSUS_V6_STAGES,
    emit_consensus_v6_witness,
)
from .corruption_robust_carrier_v8 import (
    CorruptionRobustCarrierV8,
    emit_corruption_robustness_v8_witness,
)
from .deep_proxy_stack_v6 import DeepProxyStackV6
from .deep_substrate_hybrid_v5 import (
    DeepSubstrateHybridV5,
    DeepSubstrateHybridV5ForwardWitness,
    deep_substrate_hybrid_v5_forward,
)
from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v6 import (
    emit_disagreement_algebra_v6_witness,
)
from .ecc_codebook_v12 import (
    ECCCodebookV12, ECCCompressionV12Witness,
    compress_carrier_ecc_v12,
    emit_ecc_v12_compression_witness,
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
    bridge_hidden_state_and_measure_v4,
)
from .kv_bridge_v3 import KVBridgeV3Projection
from .kv_bridge_v4 import KVBridgeV4Projection
from .kv_bridge_v5 import (
    KVBridgeV5Projection,
    bridge_carrier_and_measure_v5,
)
from .long_horizon_retention_v12 import (
    LongHorizonReconstructionV12Head,
    emit_lhr_v12_witness,
)
from .mergeable_latent_capsule_v3 import make_root_capsule_v3
from .mergeable_latent_capsule_v4 import wrap_v3_as_v4
from .mergeable_latent_capsule_v5 import wrap_v4_as_v5
from .mergeable_latent_capsule_v6 import wrap_v5_as_v6
from .mergeable_latent_capsule_v7 import wrap_v6_as_v7
from .mergeable_latent_capsule_v8 import (
    MergeOperatorV8, emit_mlsc_v8_witness, wrap_v7_as_v8,
)
from .multi_hop_translator_v10 import (
    W60_DEFAULT_MH_V10_BACKENDS,
    W60_DEFAULT_MH_V10_CHAIN_LEN,
    emit_multi_hop_v10_witness,
)
from .persistent_latent_v12 import (
    V12StackedCell, PersistentLatentStateV12Chain,
    emit_persistent_v12_witness, step_persistent_state_v12,
)
from .prefix_state_bridge_v4 import (
    bridge_prefix_state_and_measure_v4,
)
from .quantised_compression import QuantisedBudgetGate
from .replay_controller import (
    ReplayCandidate, ReplayController,
    emit_replay_controller_witness,
    W60_REPLAY_DECISION_REUSE,
)
from .substrate_adapter_v5 import (
    SubstrateAdapterV5Matrix,
    probe_all_v5_adapters,
)
from .tiny_substrate_v3 import _sha256_hex as _v3_sha
from .tiny_substrate_v5 import (
    TinyV5SubstrateParams,
    W60_V5_SEGMENT_RECOMPUTE,
    W60_V5_SEGMENT_REUSE,
    build_default_tiny_substrate_v5,
    emit_tiny_substrate_v5_forward_witness,
    forward_tiny_substrate_v5,
    tokenize_bytes_v5,
)
from .transcript_vs_shared_arbiter_v9 import (
    ten_arm_compare, emit_tvs_arbiter_v9_witness,
)
from .uncertainty_layer_v8 import (
    compose_uncertainty_report_v8,
    emit_uncertainty_v8_witness,
)


W60_SCHEMA_VERSION: str = "coordpy.w60_team.v1"
W60_TEAM_RESULT_SCHEMA: str = "coordpy.w60_team_result.v1"


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
# Failure mode enumeration (≥ 50 disjoint modes)
# =============================================================================


W60_ENVELOPE_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "missing_w59_outer_cid",
    "missing_substrate_v5_witness",
    "substrate_v5_witness_invalid",
    "missing_kv_bridge_v5_witness",
    "kv_bridge_v5_witness_invalid",
    "missing_hsb_v4_witness",
    "hsb_v4_witness_invalid",
    "missing_prefix_state_v4_witness",
    "prefix_state_v4_drop_drift_negative",
    "prefix_state_v4_flop_saving_zero",
    "missing_attn_steer_v4_witness",
    "attn_steer_v4_per_query_budget_violated",
    "missing_cache_controller_v3_witness",
    "cache_controller_v3_negative_flop_savings",
    "missing_replay_controller_witness",
    "replay_controller_no_decisions",
    "missing_persistent_v12_witness",
    "persistent_v12_chain_walk_short",
    "missing_multi_hop_v10_witness",
    "multi_hop_v10_chain_length_off",
    "missing_mlsc_v8_witness",
    "mlsc_v8_replay_chain_empty",
    "missing_consensus_v6_witness",
    "consensus_v6_stage_count_off",
    "missing_crc_v8_witness",
    "crc_v8_kv256_detect_below_floor",
    "crc_v8_post_replay_topk_below_floor",
    "missing_lhr_v12_witness",
    "lhr_v12_max_k_off",
    "missing_ecc_v12_witness",
    "ecc_v12_bits_per_token_below_floor",
    "ecc_v12_total_codes_off",
    "missing_tvs_v9_witness",
    "tvs_v9_pick_rates_not_sum_to_one",
    "missing_uncertainty_v8_witness",
    "uncertainty_v8_bracket_violated",
    "missing_disagreement_algebra_v6_witness",
    "disagreement_algebra_v6_replay_identity_failed",
    "missing_deep_substrate_hybrid_v5_witness",
    "deep_substrate_hybrid_v5_not_five_way",
    "missing_substrate_adapter_v5_matrix",
    "substrate_adapter_v5_no_v5_full",
    "w60_outer_cid_mismatch_under_replay",
    "w60_params_cid_mismatch",
    "w60_envelope_schema_drift",
    "w60_trivial_passthrough_broken",
    "w60_v5_no_autograd_cap_missing",
    "w60_no_third_party_substrate_coupling_cap_missing",
    "w60_v12_outer_not_trained_cap_missing",
    "w60_ecc_v12_rate_floor_cap_missing",
    "w60_lhr_v12_scorer_fit_cap_missing",
    "w60_corruption_flag_channel_cap_missing",
)


# =============================================================================
# W60Params
# =============================================================================


@dataclasses.dataclass
class W60Params:
    substrate_v5: TinyV5SubstrateParams | None
    v12_cell: V12StackedCell | None
    mlsc_v8_operator: MergeOperatorV8 | None
    consensus_v6: ConsensusFallbackControllerV6 | None
    crc_v8: CorruptionRobustCarrierV8 | None
    lhr_v12: LongHorizonReconstructionV12Head | None
    ecc_v12: ECCCodebookV12 | None
    deep_substrate_hybrid_v5: DeepSubstrateHybridV5 | None
    kv_bridge_v5: KVBridgeV5Projection | None
    hidden_state_bridge_v4: HiddenStateBridgeV4Projection | None
    attention_steering_v2_proj: (
        AttentionSteeringV2Projection | None)
    cache_controller_v3: CacheControllerV3 | None
    replay_controller: ReplayController | None

    enabled: bool = True
    follow_up_token_ids: tuple[int, ...] = (257,)
    tvs_budget_tokens: int = 6
    kl_budget_per_query: float = 0.5

    @classmethod
    def build_trivial(cls) -> "W60Params":
        return cls(
            substrate_v5=None, v12_cell=None,
            mlsc_v8_operator=None, consensus_v6=None,
            crc_v8=None, lhr_v12=None, ecc_v12=None,
            deep_substrate_hybrid_v5=None,
            kv_bridge_v5=None,
            hidden_state_bridge_v4=None,
            attention_steering_v2_proj=None,
            cache_controller_v3=None,
            replay_controller=None,
            enabled=False,
        )

    @classmethod
    def build_default(
            cls, *, seed: int = 60000,
    ) -> "W60Params":
        sub_v5 = build_default_tiny_substrate_v5(
            seed=int(seed) + 1)
        v12 = V12StackedCell.init(seed=int(seed) + 2)
        mlsc_op = MergeOperatorV8(factor_dim=6)
        consensus = ConsensusFallbackControllerV6.init(
            k_required=2, cosine_floor=0.6,
            trust_threshold=0.5)
        crc = CorruptionRobustCarrierV8()
        lhr = LongHorizonReconstructionV12Head.init(
            seed=int(seed) + 3)
        ecc = ECCCodebookV12.init(seed=int(seed) + 4)
        deep_v6 = DeepProxyStackV6.init(seed=int(seed) + 5)
        d_head = (
            int(sub_v5.config.d_model)
            // int(sub_v5.config.n_heads))
        kv_b3 = KVBridgeV3Projection.init(
            n_layers=int(sub_v5.config.n_layers),
            n_heads=int(sub_v5.config.n_heads),
            n_kv_heads=int(sub_v5.config.n_kv_heads),
            n_inject_tokens=3,
            carrier_dim=int(
                deep_v6.inner_v5.inner_v4.factor_dim),
            d_head=int(d_head),
            seed=int(seed) + 7)
        kv_b4 = KVBridgeV4Projection.init_from_v3(
            kv_b3, seed_v4=int(seed) + 8)
        kv_b5 = KVBridgeV5Projection.init_from_v4(
            kv_b4, seed_v5=int(seed) + 9)
        hsb2 = HiddenStateBridgeV2Projection.init(
            target_layers=(1, 3),
            n_tokens=6,
            carrier_dim=int(
                deep_v6.inner_v5.inner_v4.factor_dim),
            d_model=int(sub_v5.config.d_model),
            seed=int(seed) + 10)
        hsb3 = HiddenStateBridgeV3Projection.init_from_v2(
            hsb2, n_heads=int(sub_v5.config.n_heads),
            seed_v3=int(seed) + 11)
        hsb4 = HiddenStateBridgeV4Projection.init_from_v3(
            hsb3, seed_v4=int(seed) + 12)
        attn_steer = AttentionSteeringV2Projection.init(
            n_layers=int(sub_v5.config.n_layers),
            n_heads=int(sub_v5.config.n_heads),
            n_query=4, n_key=8,
            carrier_dim=int(
                deep_v6.inner_v5.inner_v4.factor_dim),
            seed=int(seed) + 13)
        ctrl_v3 = CacheControllerV3.init(
            policy=W60_CACHE_POLICY_COMPOSITE_V3,
            d_model=int(sub_v5.config.d_model),
            fit_seed=int(seed) + 14)
        ctrl_v3.composite_weights = _np.array(
            [0.4, 0.2, 0.2, 0.2], dtype=_np.float64)
        ctrl_v3.attention_receive_scorer = _np.zeros(
            (int(sub_v5.config.n_layers)
              * int(sub_v5.config.n_heads),),
            dtype=_np.float64)
        replay = ReplayController()
        hybrid_v5 = DeepSubstrateHybridV5.init(
            deep_v6=deep_v6, substrate_v5=sub_v5,
            bridge_v5=kv_b5,
            cache_controller_v3=ctrl_v3,
            replay_controller=replay,
            substrate_back_inject_weight=0.10,
            cache_retention_ratio=0.7)
        return cls(
            substrate_v5=sub_v5,
            v12_cell=v12,
            mlsc_v8_operator=mlsc_op,
            consensus_v6=consensus,
            crc_v8=crc,
            lhr_v12=lhr,
            ecc_v12=ecc,
            deep_substrate_hybrid_v5=hybrid_v5,
            kv_bridge_v5=kv_b5,
            hidden_state_bridge_v4=hsb4,
            attention_steering_v2_proj=attn_steer,
            cache_controller_v3=ctrl_v3,
            replay_controller=replay,
            enabled=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": W60_SCHEMA_VERSION,
            "enabled": bool(self.enabled),
            "substrate_v5_cid": (
                self.substrate_v5.cid()
                if self.substrate_v5 is not None else ""),
            "v12_cell_cid": (
                self.v12_cell.cid()
                if self.v12_cell is not None else ""),
            "mlsc_v8_operator_cid": (
                self.mlsc_v8_operator.cid()
                if self.mlsc_v8_operator is not None else ""),
            "consensus_v6_cid": (
                self.consensus_v6.cid()
                if self.consensus_v6 is not None else ""),
            "crc_v8_cid": (
                self.crc_v8.cid()
                if self.crc_v8 is not None else ""),
            "lhr_v12_cid": (
                self.lhr_v12.cid()
                if self.lhr_v12 is not None else ""),
            "ecc_v12_cid": (
                self.ecc_v12.cid()
                if self.ecc_v12 is not None else ""),
            "deep_substrate_hybrid_v5_cid": (
                self.deep_substrate_hybrid_v5.cid()
                if self.deep_substrate_hybrid_v5 is not None
                else ""),
            "kv_bridge_v5_cid": (
                self.kv_bridge_v5.cid()
                if self.kv_bridge_v5 is not None else ""),
            "hidden_state_bridge_v4_cid": (
                self.hidden_state_bridge_v4.cid()
                if self.hidden_state_bridge_v4 is not None
                else ""),
            "attention_steering_v2_proj_cid": (
                self.attention_steering_v2_proj.cid()
                if self.attention_steering_v2_proj is not None
                else ""),
            "cache_controller_v3_cid": (
                self.cache_controller_v3.cid()
                if self.cache_controller_v3 is not None else ""),
            "replay_controller_cid": (
                self.replay_controller.cid()
                if self.replay_controller is not None else ""),
            "follow_up_token_ids": list(
                self.follow_up_token_ids),
            "tvs_budget_tokens": int(self.tvs_budget_tokens),
            "kl_budget_per_query": float(round(
                self.kl_budget_per_query, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w60_params",
            "params": self.to_dict()})


# =============================================================================
# W60HandoffEnvelope
# =============================================================================


@dataclasses.dataclass(frozen=True)
class W60HandoffEnvelope:
    schema: str
    w59_outer_cid: str
    w60_params_cid: str
    substrate_v5_witness_cid: str
    kv_bridge_v5_witness_cid: str
    hsb_v4_witness_cid: str
    prefix_state_v4_witness_cid: str
    attn_steer_v4_witness_cid: str
    cache_controller_v3_witness_cid: str
    replay_controller_witness_cid: str
    persistent_v12_witness_cid: str
    multi_hop_v10_witness_cid: str
    mlsc_v8_witness_cid: str
    consensus_v6_witness_cid: str
    crc_v8_witness_cid: str
    lhr_v12_witness_cid: str
    ecc_v12_witness_cid: str
    tvs_v9_witness_cid: str
    uncertainty_v8_witness_cid: str
    disagreement_algebra_v6_witness_cid: str
    deep_substrate_hybrid_v5_witness_cid: str
    substrate_adapter_v5_matrix_cid: str
    v12_chain_cid: str
    replay_aware_weighted_mean: float
    five_way_used: bool
    substrate_v5_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "w59_outer_cid": str(self.w59_outer_cid),
            "w60_params_cid": str(self.w60_params_cid),
            "substrate_v5_witness_cid": str(
                self.substrate_v5_witness_cid),
            "kv_bridge_v5_witness_cid": str(
                self.kv_bridge_v5_witness_cid),
            "hsb_v4_witness_cid": str(self.hsb_v4_witness_cid),
            "prefix_state_v4_witness_cid": str(
                self.prefix_state_v4_witness_cid),
            "attn_steer_v4_witness_cid": str(
                self.attn_steer_v4_witness_cid),
            "cache_controller_v3_witness_cid": str(
                self.cache_controller_v3_witness_cid),
            "replay_controller_witness_cid": str(
                self.replay_controller_witness_cid),
            "persistent_v12_witness_cid": str(
                self.persistent_v12_witness_cid),
            "multi_hop_v10_witness_cid": str(
                self.multi_hop_v10_witness_cid),
            "mlsc_v8_witness_cid": str(
                self.mlsc_v8_witness_cid),
            "consensus_v6_witness_cid": str(
                self.consensus_v6_witness_cid),
            "crc_v8_witness_cid": str(self.crc_v8_witness_cid),
            "lhr_v12_witness_cid": str(
                self.lhr_v12_witness_cid),
            "ecc_v12_witness_cid": str(
                self.ecc_v12_witness_cid),
            "tvs_v9_witness_cid": str(
                self.tvs_v9_witness_cid),
            "uncertainty_v8_witness_cid": str(
                self.uncertainty_v8_witness_cid),
            "disagreement_algebra_v6_witness_cid": str(
                self.disagreement_algebra_v6_witness_cid),
            "deep_substrate_hybrid_v5_witness_cid": str(
                self.deep_substrate_hybrid_v5_witness_cid),
            "substrate_adapter_v5_matrix_cid": str(
                self.substrate_adapter_v5_matrix_cid),
            "v12_chain_cid": str(self.v12_chain_cid),
            "replay_aware_weighted_mean": float(round(
                self.replay_aware_weighted_mean, 12)),
            "five_way_used": bool(self.five_way_used),
            "substrate_v5_used": bool(self.substrate_v5_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w60_handoff_envelope",
            "envelope": self.to_dict()})


# =============================================================================
# Verifier
# =============================================================================


def verify_w60_handoff(
        envelope: W60HandoffEnvelope,
) -> dict[str, Any]:
    failures: list[str] = []
    if not envelope.w59_outer_cid:
        failures.append("missing_w59_outer_cid")
    if not envelope.substrate_v5_witness_cid:
        failures.append("missing_substrate_v5_witness")
    if not envelope.kv_bridge_v5_witness_cid:
        failures.append("missing_kv_bridge_v5_witness")
    if not envelope.hsb_v4_witness_cid:
        failures.append("missing_hsb_v4_witness")
    if not envelope.prefix_state_v4_witness_cid:
        failures.append("missing_prefix_state_v4_witness")
    if not envelope.attn_steer_v4_witness_cid:
        failures.append("missing_attn_steer_v4_witness")
    if not envelope.cache_controller_v3_witness_cid:
        failures.append("missing_cache_controller_v3_witness")
    if not envelope.replay_controller_witness_cid:
        failures.append("missing_replay_controller_witness")
    if not envelope.persistent_v12_witness_cid:
        failures.append("missing_persistent_v12_witness")
    if not envelope.multi_hop_v10_witness_cid:
        failures.append("missing_multi_hop_v10_witness")
    if not envelope.mlsc_v8_witness_cid:
        failures.append("missing_mlsc_v8_witness")
    if not envelope.consensus_v6_witness_cid:
        failures.append("missing_consensus_v6_witness")
    if not envelope.crc_v8_witness_cid:
        failures.append("missing_crc_v8_witness")
    if not envelope.lhr_v12_witness_cid:
        failures.append("missing_lhr_v12_witness")
    if not envelope.ecc_v12_witness_cid:
        failures.append("missing_ecc_v12_witness")
    if not envelope.tvs_v9_witness_cid:
        failures.append("missing_tvs_v9_witness")
    if not envelope.uncertainty_v8_witness_cid:
        failures.append("missing_uncertainty_v8_witness")
    if not envelope.disagreement_algebra_v6_witness_cid:
        failures.append(
            "missing_disagreement_algebra_v6_witness")
    if not envelope.deep_substrate_hybrid_v5_witness_cid:
        failures.append(
            "missing_deep_substrate_hybrid_v5_witness")
    if not envelope.substrate_adapter_v5_matrix_cid:
        failures.append(
            "missing_substrate_adapter_v5_matrix")
    return {
        "schema": W60_SCHEMA_VERSION + ".verifier.v1",
        "failures": list(failures),
        "ok": bool(not failures),
        "n_failure_modes_known": int(
            len(W60_ENVELOPE_VERIFIER_FAILURE_MODES)),
    }


# =============================================================================
# W60 team
# =============================================================================


@dataclasses.dataclass
class W60Team:
    params: W60Params
    chain: PersistentLatentStateV12Chain = dataclasses.field(
        default_factory=PersistentLatentStateV12Chain.empty)

    def step(
            self, *, turn_index: int, role: str = "r",
            w59_outer_cid: str = "no_w59",
    ) -> W60HandoffEnvelope:
        """Run one W60 turn and emit a sealed envelope."""
        p = self.params
        # M1 — substrate V5 forward.
        sub_v5_w_cid = ""
        sub_v5_used = False
        if p.enabled and p.substrate_v5 is not None:
            ids = tokenize_bytes_v5(
                "w60-turn-" + str(int(turn_index)),
                max_len=14)
            trace, cache = forward_tiny_substrate_v5(
                p.substrate_v5, ids)
            w = emit_tiny_substrate_v5_forward_witness(
                trace, cache)
            sub_v5_w_cid = w.cid()
            sub_v5_used = True
        # M2 — KV bridge V5.
        kv_v5_w_cid = ""
        if (p.enabled and p.substrate_v5 is not None
                and p.kv_bridge_v5 is not None):
            carrier = _payload_hash_vec(
                ("kv_b5", int(turn_index)),
                int(p.kv_bridge_v5.carrier_dim))
            fu = list(p.follow_up_token_ids)
            try:
                wkv = bridge_carrier_and_measure_v5(
                    params=p.substrate_v5.v3_params,
                    carrier=carrier,
                    projection=p.kv_bridge_v5,
                    follow_up_token_ids=fu)
                kv_v5_w_cid = wkv.cid()
            except Exception:
                kv_v5_w_cid = ""
        # M3 — HSB V4.
        hsb4_w_cid = ""
        if (p.enabled and p.substrate_v5 is not None
                and p.hidden_state_bridge_v4 is not None):
            carrier = _payload_hash_vec(
                ("hsb4", int(turn_index)),
                int(p.hidden_state_bridge_v4.inner_v3.inner_v2.carrier_dim))
            target_n = int(
                p.hidden_state_bridge_v4.inner_v3.inner_v2.n_tokens)
            ids = list(tokenize_bytes_v5(
                "hsb-turn-" + str(int(turn_index)),
                max_len=64))[:target_n]
            while len(ids) < target_n:
                ids.append(0)
            try:
                target = _np.zeros(
                    int(p.substrate_v5.config.vocab_size),
                    dtype=_np.float64)
                target[
                    int(turn_index)
                    % int(p.substrate_v5.config.vocab_size)] = 1.0
                whsb = bridge_hidden_state_and_measure_v4(
                    params=p.substrate_v5.v3_params,
                    carrier=carrier,
                    projection=p.hidden_state_bridge_v4,
                    token_ids=ids,
                    target_delta_logits=target)
                hsb4_w_cid = whsb.cid()
            except Exception:
                hsb4_w_cid = ""
        # M4 — prefix state V4.
        ps4_w_cid = ""
        if p.enabled and p.substrate_v5 is not None:
            prompt = tokenize_bytes_v5(
                "prefix-" + str(int(turn_index)), max_len=14)
            fu = tokenize_bytes_v5("fu", max_len=4)
            try:
                segs = [
                    (0, max(1, len(prompt) // 2),
                     W60_V5_SEGMENT_REUSE),
                    (max(1, len(prompt) // 2), len(prompt),
                     W60_V5_SEGMENT_RECOMPUTE),
                ]
                wps = bridge_prefix_state_and_measure_v4(
                    params_v5=p.substrate_v5,
                    prompt_token_ids=prompt,
                    follow_up_chain=[fu],
                    segments=segs)
                ps4_w_cid = wps.cid()
            except Exception:
                ps4_w_cid = ""
        # M5 — attention steering V4.
        attn4_w_cid = ""
        if (p.enabled and p.substrate_v5 is not None
                and p.attention_steering_v2_proj is not None):
            carrier = _payload_hash_vec(
                ("attn4", int(turn_index)),
                int(p.attention_steering_v2_proj.carrier_dim))
            target_q = int(
                p.attention_steering_v2_proj.n_query)
            ids = list(tokenize_bytes_v5(
                "att-" + str(int(turn_index)),
                max_len=64))[:target_q]
            while len(ids) < target_q:
                ids.append(0)
            try:
                watt = steer_attention_and_measure_v4(
                    params=p.substrate_v5.v3_params,
                    carrier=carrier,
                    projection=p.attention_steering_v2_proj,
                    token_ids=ids,
                    kl_budget_per_query=float(
                        p.kl_budget_per_query))
                attn4_w_cid = watt.cid()
            except Exception:
                attn4_w_cid = ""
        # M6 — cache controller V3.
        cc3_w_cid = ""
        if (p.enabled and p.substrate_v5 is not None
                and p.cache_controller_v3 is not None):
            prompt = tokenize_bytes_v5(
                "ctrl-" + str(int(turn_index)), max_len=18)
            fu = tokenize_bytes_v5("fu", max_len=4)
            qv = list(_np.random.default_rng(
                int(turn_index) + 6000).standard_normal(
                    int(p.cache_controller_v3.d_model)))
            try:
                wctrl = apply_cache_controller_v3_and_measure(
                    p.substrate_v5.v3_params,
                    p.cache_controller_v3,
                    prompt_token_ids=prompt,
                    follow_up_token_ids=fu,
                    retention_ratio=0.7,
                    query_vector=qv)
                cc3_w_cid = wctrl.cid()
            except Exception:
                cc3_w_cid = ""
        # M7 — replay controller.
        replay_w_cid = ""
        if (p.enabled and p.replay_controller is not None):
            cands: list[ReplayCandidate] = []
            for j in range(4):
                cand = ReplayCandidate(
                    flop_reuse=100 + j * 50,
                    flop_recompute=1000 + j * 100,
                    flop_fallback=10,
                    drift_l2_reuse=0.05 + 0.05 * j,
                    drift_l2_recompute=0.0,
                    drift_l2_fallback=0.5,
                    crc_passed=(j != 1),
                    transcript_available=True,
                    n_corruption_flags=int(j))
                p.replay_controller.decide(cand)
                cands.append(cand)
            wrep = emit_replay_controller_witness(
                p.replay_controller, cands)
            replay_w_cid = wrep.cid()
        # M8 — persistent V12.
        per_v12_w_cid = ""
        if p.enabled and p.v12_cell is not None:
            carrier_vals = _payload_hash_vec(
                ("v12", int(turn_index)),
                int(p.v12_cell.state_dim))
            state = step_persistent_state_v12(
                cell=p.v12_cell,
                prev_state=None,
                carrier_values=carrier_vals,
                turn_index=int(turn_index), role=str(role),
                substrate_skip=carrier_vals,
                hidden_state_skip=carrier_vals,
                attention_skip=carrier_vals,
                retrieval_skip=carrier_vals,
                replay_skip=carrier_vals,
                substrate_fidelity=0.9,
                attention_fidelity=0.9,
                retrieval_fidelity=0.9,
                replay_fidelity=0.9)
            self.chain.add(state)
            wp = emit_persistent_v12_witness(
                cell=p.v12_cell, chain=self.chain,
                leaf_cid=state.cid())
            per_v12_w_cid = wp.cid()
        # M9 — multi-hop V10.
        mh_v10_w_cid = ""
        if p.enabled:
            wmh = emit_multi_hop_v10_witness(
                backends=W60_DEFAULT_MH_V10_BACKENDS,
                chain_length=W60_DEFAULT_MH_V10_CHAIN_LEN,
                seed=int(turn_index) + 6000)
            mh_v10_w_cid = wmh.cid()
        # M10 — MLSC V8.
        mlsc_v8_w_cid = ""
        if (p.enabled and p.mlsc_v8_operator is not None):
            v3 = make_root_capsule_v3(
                branch_id=f"w60_{int(turn_index)}",
                payload=tuple([0.1] * 6),
                fact_tags=("w60",),
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
            merged = p.mlsc_v8_operator.merge(
                [v8],
                replay_witness_chain=(
                    f"merge_replay_{turn_index}",),
                substrate_witness_chain=(
                    f"merge_sub_{turn_index}",),
                provenance_trust_table={
                    "backend_c": 0.85})
            wmlsc = emit_mlsc_v8_witness(merged)
            mlsc_v8_w_cid = wmlsc.cid()
        # M11 — consensus V6.
        cons_v6_w_cid = ""
        if (p.enabled and p.consensus_v6 is not None):
            d = 6
            p1 = _payload_hash_vec(
                ("c6", "p1", int(turn_index)), d)
            p2 = _payload_hash_vec(
                ("c6", "p2", int(turn_index)), d)
            q = _payload_hash_vec(
                ("c6", "q", int(turn_index)), d)
            p.consensus_v6.decide(
                parent_payloads=[p1, p2],
                parent_trusts=[0.9, 0.7],
                parent_cache_fingerprints=[(1, 2), (3, 4)],
                parent_retrieval_scores=[0.6, 0.7],
                parent_replay_decisions=[
                    "choose_reuse", "choose_recompute"],
                query_direction=q,
                transcript_payload=[0.0] * d)
            wcons = emit_consensus_v6_witness(p.consensus_v6)
            cons_v6_w_cid = wcons.cid()
        # M12 — CRC V8.
        crc_v8_w_cid = ""
        if (p.enabled and p.crc_v8 is not None):
            wcrc = emit_corruption_robustness_v8_witness(
                crc_v8=p.crc_v8, n_probes=8,
                seed=int(turn_index) + 600)
            crc_v8_w_cid = wcrc.cid()
        # M13 — LHR V12.
        lhr_v12_w_cid = ""
        if (p.enabled and p.lhr_v12 is not None):
            wlhr = emit_lhr_v12_witness(head=p.lhr_v12)
            lhr_v12_w_cid = wlhr.cid()
        # M14 — ECC V12.
        ecc_v12_w_cid = ""
        if (p.enabled and p.ecc_v12 is not None):
            gate = QuantisedBudgetGate.init(
                in_dim=W53_DEFAULT_ECC_CODE_DIM,
                emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
                seed=int(turn_index) + 601)
            gate.importance_threshold = 0.0
            gate.w_emit.values = (
                [1.0] * len(gate.w_emit.values))
            carrier = _payload_hash_vec(
                ("ecc_v12", int(turn_index)),
                W53_DEFAULT_ECC_CODE_DIM)
            comp = compress_carrier_ecc_v12(
                carrier, codebook=p.ecc_v12, gate=gate)
            wecc = emit_ecc_v12_compression_witness(
                codebook=p.ecc_v12, compression=comp)
            ecc_v12_w_cid = wecc.cid()
        # M15 — TVS V9.
        tvs_v9_w_cid = ""
        if p.enabled:
            res = ten_arm_compare(
                per_turn_confidences=[0.7],
                per_turn_trust_scores=[0.8],
                per_turn_merge_retentions=[0.6],
                per_turn_tw_retentions=[0.55],
                per_turn_substrate_fidelities=[0.55],
                per_turn_hidden_fidelities=[0.5],
                per_turn_cache_fidelities=[0.6],
                per_turn_retrieval_fidelities=[0.7],
                per_turn_replay_fidelities=[0.85],
                budget_tokens=int(p.tvs_budget_tokens))
            wtvs = emit_tvs_arbiter_v9_witness(result=res)
            tvs_v9_w_cid = wtvs.cid()
        # M16 — uncertainty V8.
        unc_v8_w_cid = ""
        composite_v8 = 1.0
        if p.enabled:
            comp = compose_uncertainty_report_v8(
                component_confidences={
                    "kv": 0.85, "hsb": 0.80, "attn": 0.75,
                    "cache": 0.78, "retr": 0.82,
                    "replay": 0.80,
                },
                trust_weights={
                    "kv": 0.9, "hsb": 0.85, "attn": 0.8,
                    "cache": 0.85, "retr": 0.88,
                    "replay": 0.86,
                },
                substrate_fidelities={
                    "kv": 0.85, "hsb": 0.8, "attn": 0.7,
                    "cache": 0.75, "retr": 0.78,
                    "replay": 0.8,
                },
                hidden_state_fidelities={
                    "kv": 0.7, "hsb": 0.9, "attn": 0.7,
                    "cache": 0.7, "retr": 0.72,
                    "replay": 0.75,
                },
                cache_reuse_fidelities={
                    "kv": 0.75, "hsb": 0.7, "attn": 0.7,
                    "cache": 0.95, "retr": 0.8,
                    "replay": 0.85,
                },
                retrieval_fidelities={
                    "kv": 0.8, "hsb": 0.75, "attn": 0.7,
                    "cache": 0.78, "retr": 0.97,
                    "replay": 0.82,
                },
                replay_fidelities={
                    "kv": 0.78, "hsb": 0.72, "attn": 0.68,
                    "cache": 0.82, "retr": 0.78,
                    "replay": 0.96,
                })
            wu = emit_uncertainty_v8_witness(composite=comp)
            unc_v8_w_cid = wu.cid()
            composite_v8 = float(comp.weighted_composite)
        # M17 — disagreement algebra V6.
        da_v6_w_cid = ""
        if p.enabled:
            trace = AlgebraTrace.empty()
            pa = _payload_hash_vec(
                ("da6", "a", int(turn_index)), 6)
            pb = _payload_hash_vec(
                ("da6", "b", int(turn_index)), 6)
            pc = _payload_hash_vec(
                ("da6", "c", int(turn_index)), 6)

            def replay_oracle():
                return (True, 0.0)

            def retrieval_oracle():
                return (True, 0.0)
            wda = emit_disagreement_algebra_v6_witness(
                trace=trace,
                probe_a=pa, probe_b=pb, probe_c=pc,
                retrieval_replay_oracle=retrieval_oracle,
                replay_controller_oracle=replay_oracle)
            da_v6_w_cid = wda.cid()
        # M18 — deep substrate hybrid V5.
        hyb_v5_w_cid = ""
        five_way_used = False
        if (p.enabled
                and p.deep_substrate_hybrid_v5 is not None):
            try:
                d_v6 = p.deep_substrate_hybrid_v5.deep_v6
                in_dim = int(d_v6.in_dim)
                fd = int(d_v6.inner_v5.inner_v4.factor_dim)
                qi = _payload_hash_vec(
                    ("hyb_v5_q", int(turn_index)), in_dim)
                ks = [_payload_hash_vec(
                    ("hyb_v5_k", int(turn_index), j), fd)
                    for j in range(2)]
                vs = [_payload_hash_vec(
                    ("hyb_v5_v", int(turn_index), j), fd)
                    for j in range(2)]
                _, whyb, _ = deep_substrate_hybrid_v5_forward(
                    hybrid=p.deep_substrate_hybrid_v5,
                    query_input=qi,
                    slot_keys=ks, slot_values=vs,
                    role_index=0, branch_index=0,
                    cycle_index=0,
                    trust_scalar=0.9,
                    uncertainty_scale=0.95,
                    substrate_query_token_ids=list(
                        p.follow_up_token_ids))
                hyb_v5_w_cid = whyb.cid()
                five_way_used = bool(whyb.five_way)
            except Exception:
                hyb_v5_w_cid = ""
        # M19 — substrate adapter V5.
        adapter_v5_cid = ""
        if p.enabled:
            mat = probe_all_v5_adapters(
                probe_ollama=False, probe_openai=False)
            adapter_v5_cid = mat.cid()
        return W60HandoffEnvelope(
            schema=W60_SCHEMA_VERSION + ".envelope.v1",
            w59_outer_cid=str(w59_outer_cid),
            w60_params_cid=str(p.cid()),
            substrate_v5_witness_cid=str(sub_v5_w_cid),
            kv_bridge_v5_witness_cid=str(kv_v5_w_cid),
            hsb_v4_witness_cid=str(hsb4_w_cid),
            prefix_state_v4_witness_cid=str(ps4_w_cid),
            attn_steer_v4_witness_cid=str(attn4_w_cid),
            cache_controller_v3_witness_cid=str(cc3_w_cid),
            replay_controller_witness_cid=str(replay_w_cid),
            persistent_v12_witness_cid=str(per_v12_w_cid),
            multi_hop_v10_witness_cid=str(mh_v10_w_cid),
            mlsc_v8_witness_cid=str(mlsc_v8_w_cid),
            consensus_v6_witness_cid=str(cons_v6_w_cid),
            crc_v8_witness_cid=str(crc_v8_w_cid),
            lhr_v12_witness_cid=str(lhr_v12_w_cid),
            ecc_v12_witness_cid=str(ecc_v12_w_cid),
            tvs_v9_witness_cid=str(tvs_v9_w_cid),
            uncertainty_v8_witness_cid=str(unc_v8_w_cid),
            disagreement_algebra_v6_witness_cid=str(
                da_v6_w_cid),
            deep_substrate_hybrid_v5_witness_cid=str(
                hyb_v5_w_cid),
            substrate_adapter_v5_matrix_cid=str(
                adapter_v5_cid),
            v12_chain_cid=str(self.chain.cid()),
            replay_aware_weighted_mean=float(composite_v8),
            five_way_used=bool(five_way_used),
            substrate_v5_used=bool(sub_v5_used),
        )


def build_w60_team(*, seed: int = 60000) -> W60Team:
    return W60Team(
        params=W60Params.build_default(seed=int(seed)))


def build_trivial_w60_envelope(
        *, w59_outer_cid: str = "no_w59",
) -> W60HandoffEnvelope:
    return W60HandoffEnvelope(
        schema=W60_SCHEMA_VERSION + ".envelope.v1",
        w59_outer_cid=str(w59_outer_cid),
        w60_params_cid=str(W60Params.build_trivial().cid()),
        substrate_v5_witness_cid="",
        kv_bridge_v5_witness_cid="",
        hsb_v4_witness_cid="",
        prefix_state_v4_witness_cid="",
        attn_steer_v4_witness_cid="",
        cache_controller_v3_witness_cid="",
        replay_controller_witness_cid="",
        persistent_v12_witness_cid="",
        multi_hop_v10_witness_cid="",
        mlsc_v8_witness_cid="",
        consensus_v6_witness_cid="",
        crc_v8_witness_cid="",
        lhr_v12_witness_cid="",
        ecc_v12_witness_cid="",
        tvs_v9_witness_cid="",
        uncertainty_v8_witness_cid="",
        disagreement_algebra_v6_witness_cid="",
        deep_substrate_hybrid_v5_witness_cid="",
        substrate_adapter_v5_matrix_cid="",
        v12_chain_cid="",
        replay_aware_weighted_mean=1.0,
        five_way_used=False,
        substrate_v5_used=False,
    )


__all__ = [
    "W60_SCHEMA_VERSION",
    "W60_TEAM_RESULT_SCHEMA",
    "W60_ENVELOPE_VERIFIER_FAILURE_MODES",
    "W60Params",
    "W60HandoffEnvelope",
    "W60Team",
    "build_w60_team",
    "build_trivial_w60_envelope",
    "verify_w60_handoff",
]
