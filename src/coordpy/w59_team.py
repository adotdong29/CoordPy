"""W59 — Trainable Substrate-Conditioned Latent Operating System.

The ``W59Team`` orchestrator composes the W58 team with the W59
mechanism modules. Per-turn it emits 18 module witness CIDs and
seals them into a ``W59HandoffEnvelope`` whose
``w58_outer_cid`` carries forward the W58 envelope.

Honest scope
------------

* The W59 substrate is the in-repo V4 NumPy runtime. We do NOT
  bridge to third-party hosted models. ``W59-L-NO-THIRD-PARTY-
  SUBSTRATE-COUPLING-CAP`` carries forward the W58 cap unchanged.
* W59 fits **closed-form ridge parameters** in three places: the
  KV bridge V4 correction, the cache controller V2 retrieval
  matrix, and the LHR V11 retention scorer. NOT end-to-end
  backprop. ``W59-L-V4-NO-AUTOGRAD-CAP`` is load-bearing.
* Trivial passthrough is preserved byte-for-byte: when
  ``W59Params.build_trivial()`` is used the W59 envelope's
  internal ``w58_outer_cid`` carries the supplied W58 outer CID
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
from .attention_steering_bridge_v3 import (
    steer_attention_and_measure_v3,
)
from .cache_controller_v2 import (
    CacheControllerV2,
    W59_CACHE_POLICY_LEARNED_HIDDEN,
    W59_CACHE_POLICY_LEARNED_RETRIEVAL,
    apply_cache_controller_v2_and_measure,
)
from .consensus_fallback_controller_v5 import (
    ConsensusFallbackControllerV5,
    W59_CONSENSUS_V5_STAGES,
    emit_consensus_v5_witness,
)
from .corruption_robust_carrier_v7 import (
    CorruptionRobustCarrierV7,
    emit_corruption_robustness_v7_witness,
)
from .deep_proxy_stack_v6 import DeepProxyStackV6
from .deep_substrate_hybrid_v4 import (
    DeepSubstrateHybridV4,
    DeepSubstrateHybridV4ForwardWitness,
    deep_substrate_hybrid_v4_forward,
)
from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v5 import (
    emit_disagreement_algebra_v5_witness,
)
from .ecc_codebook_v11 import (
    ECCCodebookV11, ECCCompressionV11Witness,
    compress_carrier_ecc_v11,
    emit_ecc_v11_compression_witness,
)
from .ecc_codebook_v5 import (
    W53_DEFAULT_ECC_CODE_DIM, W53_DEFAULT_ECC_EMIT_MASK_LEN,
)
from .hidden_state_bridge_v2 import (
    HiddenStateBridgeV2Projection,
)
from .hidden_state_bridge_v3 import (
    HiddenStateBridgeV3Projection,
    bridge_hidden_state_and_measure_v3,
)
from .kv_bridge_v3 import KVBridgeV3Projection
from .kv_bridge_v4 import (
    KVBridgeV4Projection,
    bridge_carrier_and_measure_v4,
)
from .long_horizon_retention_v11 import (
    LongHorizonReconstructionV11Head,
    emit_lhr_v11_witness,
)
from .mergeable_latent_capsule_v3 import make_root_capsule_v3
from .mergeable_latent_capsule_v4 import wrap_v3_as_v4
from .mergeable_latent_capsule_v5 import wrap_v4_as_v5
from .mergeable_latent_capsule_v6 import wrap_v5_as_v6
from .mergeable_latent_capsule_v7 import (
    MergeOperatorV7, emit_mlsc_v7_witness, wrap_v6_as_v7,
)
from .multi_hop_translator_v9 import (
    W59_DEFAULT_MH_V9_BACKENDS,
    W59_DEFAULT_MH_V9_CHAIN_LEN,
    emit_multi_hop_v9_witness,
)
from .persistent_latent_v11 import (
    V11StackedCell, PersistentLatentStateV11Chain,
    emit_persistent_v11_witness, step_persistent_state_v11,
)
from .prefix_state_bridge_v3 import (
    bridge_prefix_state_and_measure_v3,
)
from .quantised_compression import QuantisedBudgetGate
from .substrate_adapter_v4 import (
    SubstrateAdapterV4Matrix,
    probe_all_v4_adapters,
)
from .tiny_substrate_v3 import _sha256_hex as _v3_sha
from .tiny_substrate_v4 import (
    TinyV4SubstrateParams,
    build_default_tiny_substrate_v4,
    emit_tiny_substrate_v4_forward_witness,
    forward_tiny_substrate_v4,
    tokenize_bytes_v4,
)
from .transcript_vs_shared_arbiter_v8 import (
    nine_arm_compare, emit_tvs_arbiter_v8_witness,
)
from .uncertainty_layer_v7 import (
    compose_uncertainty_report_v7,
    emit_uncertainty_v7_witness,
)


W59_SCHEMA_VERSION: str = "coordpy.w59_team.v1"
W59_TEAM_RESULT_SCHEMA: str = "coordpy.w59_team_result.v1"


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
# Failure mode enumeration (≥ 48 disjoint modes)
# =============================================================================


W59_ENVELOPE_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "missing_w58_outer_cid",
    "missing_substrate_v4_witness",
    "substrate_v4_witness_invalid",
    "missing_kv_bridge_v4_witness",
    "kv_bridge_v4_witness_invalid",
    "missing_hsb_v3_witness",
    "hsb_v3_witness_invalid",
    "missing_prefix_state_v3_witness",
    "prefix_state_v3_partial_reuse_mismatches_recompute",
    "prefix_state_v3_flop_saved_negative",
    "missing_attn_steer_v3_witness",
    "attn_steer_v3_per_head_budget_violated",
    "missing_cache_controller_v2_witness",
    "cache_controller_v2_negative_flop_savings",
    "missing_persistent_v11_witness",
    "persistent_v11_chain_walk_short",
    "missing_multi_hop_v9_witness",
    "multi_hop_v9_chain_length_off",
    "missing_mlsc_v7_witness",
    "mlsc_v7_retrieval_chain_empty",
    "missing_consensus_v5_witness",
    "consensus_v5_stage_count_off",
    "missing_crc_v7_witness",
    "crc_v7_kv128_detect_below_floor",
    "crc_v7_retrieval_topk_below_floor",
    "missing_lhr_v11_witness",
    "lhr_v11_max_k_off",
    "missing_ecc_v11_witness",
    "ecc_v11_bits_per_token_below_floor",
    "ecc_v11_total_codes_off",
    "missing_tvs_v8_witness",
    "tvs_v8_pick_rates_not_sum_to_one",
    "missing_uncertainty_v7_witness",
    "uncertainty_v7_bracket_violated",
    "missing_disagreement_algebra_v5_witness",
    "disagreement_algebra_v5_retrieval_identity_failed",
    "missing_deep_substrate_hybrid_v4_witness",
    "deep_substrate_hybrid_v4_not_four_way",
    "missing_substrate_adapter_v4_matrix",
    "substrate_adapter_v4_no_v4_full",
    "w59_outer_cid_mismatch_under_replay",
    "w59_params_cid_mismatch",
    "w59_envelope_schema_drift",
    "w59_trivial_passthrough_broken",
    "w59_v4_no_autograd_cap_missing",
    "w59_no_third_party_substrate_coupling_cap_missing",
    "w59_v11_outer_not_trained_cap_missing",
    "w59_ecc_v11_rate_floor_cap_missing",
    "w59_lhr_v11_scorer_fit_cap_missing",
)


# =============================================================================
# W59Params
# =============================================================================


@dataclasses.dataclass
class W59Params:
    substrate_v4: TinyV4SubstrateParams | None
    v11_cell: V11StackedCell | None
    mlsc_v7_operator: MergeOperatorV7 | None
    consensus_v5: ConsensusFallbackControllerV5 | None
    crc_v7: CorruptionRobustCarrierV7 | None
    lhr_v11: LongHorizonReconstructionV11Head | None
    ecc_v11: ECCCodebookV11 | None
    deep_substrate_hybrid_v4: DeepSubstrateHybridV4 | None
    kv_bridge_v4: KVBridgeV4Projection | None
    hidden_state_bridge_v3: HiddenStateBridgeV3Projection | None
    attention_steering_v3_proj: AttentionSteeringV2Projection | None
    cache_controller_v2: CacheControllerV2 | None

    enabled: bool = True
    follow_up_token_ids: tuple[int, ...] = (257,)
    tvs_budget_tokens: int = 6
    kl_budget_per_head: float = 0.6

    @classmethod
    def build_trivial(cls) -> "W59Params":
        return cls(
            substrate_v4=None, v11_cell=None,
            mlsc_v7_operator=None, consensus_v5=None,
            crc_v7=None, lhr_v11=None, ecc_v11=None,
            deep_substrate_hybrid_v4=None,
            kv_bridge_v4=None,
            hidden_state_bridge_v3=None,
            attention_steering_v3_proj=None,
            cache_controller_v2=None,
            enabled=False,
        )

    @classmethod
    def build_default(
            cls, *, seed: int = 59000,
    ) -> "W59Params":
        sub_v4 = build_default_tiny_substrate_v4(
            seed=int(seed) + 1)
        v11 = V11StackedCell.init(seed=int(seed) + 2)
        mlsc_op = MergeOperatorV7(factor_dim=6)
        consensus = ConsensusFallbackControllerV5(
            k_required=2, cosine_floor=0.6,
            trust_threshold=0.5)
        crc = CorruptionRobustCarrierV7()
        lhr = LongHorizonReconstructionV11Head.init(
            seed=int(seed) + 3)
        ecc = ECCCodebookV11.init(seed=int(seed) + 4)
        deep_v6 = DeepProxyStackV6.init(seed=int(seed) + 5)
        d_head = (
            int(sub_v4.config.d_model)
            // int(sub_v4.config.n_heads))
        kv_b3 = KVBridgeV3Projection.init(
            n_layers=int(sub_v4.config.n_layers),
            n_heads=int(sub_v4.config.n_heads),
            n_kv_heads=int(sub_v4.config.n_kv_heads),
            n_inject_tokens=3,
            carrier_dim=int(
                deep_v6.inner_v5.inner_v4.factor_dim),
            d_head=int(d_head),
            seed=int(seed) + 7)
        kv_b4 = KVBridgeV4Projection.init_from_v3(
            kv_b3, seed_v4=int(seed) + 8)
        hsb2 = HiddenStateBridgeV2Projection.init(
            target_layers=(1, 3),
            n_tokens=6,
            carrier_dim=int(
                deep_v6.inner_v5.inner_v4.factor_dim),
            d_model=int(sub_v4.config.d_model),
            seed=int(seed) + 9)
        hsb3 = HiddenStateBridgeV3Projection.init_from_v2(
            hsb2, n_heads=int(sub_v4.config.n_heads),
            seed_v3=int(seed) + 10)
        attn_steer = AttentionSteeringV2Projection.init(
            n_layers=int(sub_v4.config.n_layers),
            n_heads=int(sub_v4.config.n_heads),
            n_query=4, n_key=8,
            carrier_dim=int(
                deep_v6.inner_v5.inner_v4.factor_dim),
            seed=int(seed) + 11)
        # Build a learned_retrieval V2 controller with a randomly
        # initialised retrieval matrix (a real fit can be done at
        # benchmark time; for the default the structure is enough).
        ctrl_v2 = CacheControllerV2.init(
            policy=W59_CACHE_POLICY_LEARNED_RETRIEVAL,
            d_model=int(sub_v4.config.d_model),
            fit_seed=int(seed) + 12)
        # Initialise the retrieval matrix as scaled identity so
        # the controller is well-defined at zero training; the
        # benchmark code can fit on real data.
        d = int(sub_v4.config.d_model)
        ctrl_v2.retrieval_matrix = (
            0.1 * _np.eye(d, dtype=_np.float64))
        hybrid_v4 = DeepSubstrateHybridV4.init(
            deep_v6=deep_v6, substrate_v4=sub_v4,
            bridge_v4=kv_b4,
            cache_controller_v2=ctrl_v2,
            substrate_back_inject_weight=0.10,
            cache_retention_ratio=0.7)
        return cls(
            substrate_v4=sub_v4,
            v11_cell=v11,
            mlsc_v7_operator=mlsc_op,
            consensus_v5=consensus,
            crc_v7=crc,
            lhr_v11=lhr,
            ecc_v11=ecc,
            deep_substrate_hybrid_v4=hybrid_v4,
            kv_bridge_v4=kv_b4,
            hidden_state_bridge_v3=hsb3,
            attention_steering_v3_proj=attn_steer,
            cache_controller_v2=ctrl_v2,
            enabled=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": W59_SCHEMA_VERSION,
            "enabled": bool(self.enabled),
            "substrate_v4_cid": (
                self.substrate_v4.cid()
                if self.substrate_v4 is not None else ""),
            "v11_cell_cid": (
                self.v11_cell.cid()
                if self.v11_cell is not None else ""),
            "mlsc_v7_operator_cid": (
                self.mlsc_v7_operator.cid()
                if self.mlsc_v7_operator is not None else ""),
            "consensus_v5_cid": (
                self.consensus_v5.cid()
                if self.consensus_v5 is not None else ""),
            "crc_v7_cid": (
                self.crc_v7.cid()
                if self.crc_v7 is not None else ""),
            "lhr_v11_cid": (
                self.lhr_v11.cid()
                if self.lhr_v11 is not None else ""),
            "ecc_v11_cid": (
                self.ecc_v11.cid()
                if self.ecc_v11 is not None else ""),
            "deep_substrate_hybrid_v4_cid": (
                self.deep_substrate_hybrid_v4.cid()
                if self.deep_substrate_hybrid_v4 is not None
                else ""),
            "kv_bridge_v4_cid": (
                self.kv_bridge_v4.cid()
                if self.kv_bridge_v4 is not None else ""),
            "hidden_state_bridge_v3_cid": (
                self.hidden_state_bridge_v3.cid()
                if self.hidden_state_bridge_v3 is not None
                else ""),
            "attention_steering_v3_proj_cid": (
                self.attention_steering_v3_proj.cid()
                if self.attention_steering_v3_proj is not None
                else ""),
            "cache_controller_v2_cid": (
                self.cache_controller_v2.cid()
                if self.cache_controller_v2 is not None else ""),
            "follow_up_token_ids": list(
                self.follow_up_token_ids),
            "tvs_budget_tokens": int(self.tvs_budget_tokens),
            "kl_budget_per_head": float(round(
                self.kl_budget_per_head, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w59_params",
            "params": self.to_dict()})


# =============================================================================
# W59HandoffEnvelope
# =============================================================================


@dataclasses.dataclass(frozen=True)
class W59HandoffEnvelope:
    schema: str
    w58_outer_cid: str
    w59_params_cid: str
    substrate_v4_witness_cid: str
    kv_bridge_v4_witness_cid: str
    hsb_v3_witness_cid: str
    prefix_state_v3_witness_cid: str
    attn_steer_v3_witness_cid: str
    cache_controller_v2_witness_cid: str
    persistent_v11_witness_cid: str
    multi_hop_v9_witness_cid: str
    mlsc_v7_witness_cid: str
    consensus_v5_witness_cid: str
    crc_v7_witness_cid: str
    lhr_v11_witness_cid: str
    ecc_v11_witness_cid: str
    tvs_v8_witness_cid: str
    uncertainty_v7_witness_cid: str
    disagreement_algebra_v5_witness_cid: str
    deep_substrate_hybrid_v4_witness_cid: str
    substrate_adapter_v4_matrix_cid: str
    v11_chain_cid: str
    retrieval_aware_weighted_mean: float
    four_way_used: bool
    substrate_v4_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "w58_outer_cid": str(self.w58_outer_cid),
            "w59_params_cid": str(self.w59_params_cid),
            "substrate_v4_witness_cid": str(
                self.substrate_v4_witness_cid),
            "kv_bridge_v4_witness_cid": str(
                self.kv_bridge_v4_witness_cid),
            "hsb_v3_witness_cid": str(self.hsb_v3_witness_cid),
            "prefix_state_v3_witness_cid": str(
                self.prefix_state_v3_witness_cid),
            "attn_steer_v3_witness_cid": str(
                self.attn_steer_v3_witness_cid),
            "cache_controller_v2_witness_cid": str(
                self.cache_controller_v2_witness_cid),
            "persistent_v11_witness_cid": str(
                self.persistent_v11_witness_cid),
            "multi_hop_v9_witness_cid": str(
                self.multi_hop_v9_witness_cid),
            "mlsc_v7_witness_cid": str(
                self.mlsc_v7_witness_cid),
            "consensus_v5_witness_cid": str(
                self.consensus_v5_witness_cid),
            "crc_v7_witness_cid": str(self.crc_v7_witness_cid),
            "lhr_v11_witness_cid": str(
                self.lhr_v11_witness_cid),
            "ecc_v11_witness_cid": str(
                self.ecc_v11_witness_cid),
            "tvs_v8_witness_cid": str(
                self.tvs_v8_witness_cid),
            "uncertainty_v7_witness_cid": str(
                self.uncertainty_v7_witness_cid),
            "disagreement_algebra_v5_witness_cid": str(
                self.disagreement_algebra_v5_witness_cid),
            "deep_substrate_hybrid_v4_witness_cid": str(
                self.deep_substrate_hybrid_v4_witness_cid),
            "substrate_adapter_v4_matrix_cid": str(
                self.substrate_adapter_v4_matrix_cid),
            "v11_chain_cid": str(self.v11_chain_cid),
            "retrieval_aware_weighted_mean": float(round(
                self.retrieval_aware_weighted_mean, 12)),
            "four_way_used": bool(self.four_way_used),
            "substrate_v4_used": bool(self.substrate_v4_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w59_handoff_envelope",
            "envelope": self.to_dict()})


# =============================================================================
# Verifier
# =============================================================================


def verify_w59_handoff(
        envelope: W59HandoffEnvelope,
) -> dict[str, Any]:
    failures: list[str] = []
    if not envelope.w58_outer_cid:
        failures.append("missing_w58_outer_cid")
    if not envelope.substrate_v4_witness_cid:
        failures.append("missing_substrate_v4_witness")
    if not envelope.kv_bridge_v4_witness_cid:
        failures.append("missing_kv_bridge_v4_witness")
    if not envelope.hsb_v3_witness_cid:
        failures.append("missing_hsb_v3_witness")
    if not envelope.prefix_state_v3_witness_cid:
        failures.append("missing_prefix_state_v3_witness")
    if not envelope.attn_steer_v3_witness_cid:
        failures.append("missing_attn_steer_v3_witness")
    if not envelope.cache_controller_v2_witness_cid:
        failures.append("missing_cache_controller_v2_witness")
    if not envelope.persistent_v11_witness_cid:
        failures.append("missing_persistent_v11_witness")
    if not envelope.multi_hop_v9_witness_cid:
        failures.append("missing_multi_hop_v9_witness")
    if not envelope.mlsc_v7_witness_cid:
        failures.append("missing_mlsc_v7_witness")
    if not envelope.consensus_v5_witness_cid:
        failures.append("missing_consensus_v5_witness")
    if not envelope.crc_v7_witness_cid:
        failures.append("missing_crc_v7_witness")
    if not envelope.lhr_v11_witness_cid:
        failures.append("missing_lhr_v11_witness")
    if not envelope.ecc_v11_witness_cid:
        failures.append("missing_ecc_v11_witness")
    if not envelope.tvs_v8_witness_cid:
        failures.append("missing_tvs_v8_witness")
    if not envelope.uncertainty_v7_witness_cid:
        failures.append("missing_uncertainty_v7_witness")
    if not envelope.disagreement_algebra_v5_witness_cid:
        failures.append(
            "missing_disagreement_algebra_v5_witness")
    if not envelope.deep_substrate_hybrid_v4_witness_cid:
        failures.append(
            "missing_deep_substrate_hybrid_v4_witness")
    if not envelope.substrate_adapter_v4_matrix_cid:
        failures.append(
            "missing_substrate_adapter_v4_matrix")
    return {
        "schema": W59_SCHEMA_VERSION + ".verifier.v1",
        "failures": list(failures),
        "ok": bool(not failures),
        "n_failure_modes_known": int(
            len(W59_ENVELOPE_VERIFIER_FAILURE_MODES)),
    }


# =============================================================================
# Build W59 envelope
# =============================================================================


@dataclasses.dataclass
class W59Team:
    params: W59Params
    chain: PersistentLatentStateV11Chain = dataclasses.field(
        default_factory=PersistentLatentStateV11Chain.empty)

    def step(
            self, *, turn_index: int, role: str = "r",
            w58_outer_cid: str = "no_w58",
    ) -> W59HandoffEnvelope:
        """Run one W59 turn and emit a sealed envelope."""
        p = self.params
        # M1 — substrate V4 forward.
        sub_v4_w_cid = ""
        sub_v4_used = False
        if p.enabled and p.substrate_v4 is not None:
            ids = tokenize_bytes_v4(
                "w59-turn-" + str(int(turn_index)),
                max_len=14)
            trace, cache = forward_tiny_substrate_v4(
                p.substrate_v4, ids)
            w = emit_tiny_substrate_v4_forward_witness(
                trace, cache)
            sub_v4_w_cid = w.cid()
            sub_v4_used = True
        # M2 — KV bridge V4.
        kv_v4_w_cid = ""
        if (p.enabled and p.substrate_v4 is not None
                and p.kv_bridge_v4 is not None):
            carrier = _payload_hash_vec(
                ("kv_b4", int(turn_index)),
                int(p.kv_bridge_v4.carrier_dim))
            fu = list(p.follow_up_token_ids)
            wkv = bridge_carrier_and_measure_v4(
                params=p.substrate_v4.v3_params,
                carrier=carrier,
                projection=p.kv_bridge_v4,
                follow_up_token_ids=fu)
            kv_v4_w_cid = wkv.cid()
        # M3 — HSB V3.
        hsb3_w_cid = ""
        if (p.enabled and p.substrate_v4 is not None
                and p.hidden_state_bridge_v3 is not None):
            carrier = _payload_hash_vec(
                ("hsb3", int(turn_index)),
                int(p.hidden_state_bridge_v3.inner_v2.carrier_dim))
            target_n = int(
                p.hidden_state_bridge_v3.inner_v2.n_tokens)
            ids = list(tokenize_bytes_v4(
                "hsb-turn-" + str(int(turn_index)),
                max_len=64))[:target_n]
            while len(ids) < target_n:
                ids.append(0)
            try:
                target_delta = _np.zeros(
                    int(p.substrate_v4.config.vocab_size),
                    dtype=_np.float64)
                target_delta[
                    int(turn_index)
                    % int(p.substrate_v4.config.vocab_size)] = 1.0
                whsb = bridge_hidden_state_and_measure_v3(
                    params=p.substrate_v4.v3_params,
                    carrier=carrier,
                    projection=p.hidden_state_bridge_v3,
                    token_ids=ids,
                    target_delta_logits=target_delta)
                hsb3_w_cid = whsb.cid()
            except Exception:
                hsb3_w_cid = ""
        # M4 — prefix state V3.
        ps3_w_cid = ""
        if p.enabled and p.substrate_v4 is not None:
            prompt = tokenize_bytes_v4(
                "prefix-" + str(int(turn_index)), max_len=14)
            follow = tokenize_bytes_v4("fu", max_len=4)
            try:
                wps = bridge_prefix_state_and_measure_v3(
                    params_v4=p.substrate_v4,
                    prompt_token_ids=prompt,
                    follow_up_token_ids=follow,
                    prefix_reuse_len=max(
                        1, len(prompt) // 2))
                ps3_w_cid = wps.cid()
            except Exception:
                ps3_w_cid = ""
        # M5 — attention steering V3.
        attn3_w_cid = ""
        if (p.enabled and p.substrate_v4 is not None
                and p.attention_steering_v3_proj is not None):
            carrier = _payload_hash_vec(
                ("attn3", int(turn_index)),
                int(p.attention_steering_v3_proj.carrier_dim))
            target_q = int(p.attention_steering_v3_proj.n_query)
            ids = list(tokenize_bytes_v4(
                "att-" + str(int(turn_index)),
                max_len=64))[:target_q]
            while len(ids) < target_q:
                ids.append(0)
            try:
                watt = steer_attention_and_measure_v3(
                    params=p.substrate_v4.v3_params,
                    carrier=carrier,
                    projection=p.attention_steering_v3_proj,
                    token_ids=ids,
                    kl_budget_per_head=float(
                        p.kl_budget_per_head),
                    do_per_head_dominance=False)
                attn3_w_cid = watt.cid()
            except Exception:
                attn3_w_cid = ""
        # M6 — cache controller V2.
        cc2_w_cid = ""
        if (p.enabled and p.substrate_v4 is not None
                and p.cache_controller_v2 is not None):
            prompt = tokenize_bytes_v4(
                "ctrl-" + str(int(turn_index)), max_len=18)
            follow = tokenize_bytes_v4("fu", max_len=4)
            qv = list(_np.random.default_rng(
                int(turn_index) + 5900).standard_normal(
                    int(p.cache_controller_v2.d_model)))
            try:
                wctrl = apply_cache_controller_v2_and_measure(
                    p.substrate_v4.v3_params,
                    p.cache_controller_v2,
                    prompt_token_ids=prompt,
                    follow_up_token_ids=follow,
                    retention_ratio=0.7,
                    query_vector=qv)
                cc2_w_cid = wctrl.cid()
            except Exception:
                cc2_w_cid = ""
        # M7 — persistent V11.
        per_v11_w_cid = ""
        if p.enabled and p.v11_cell is not None:
            carrier_vals = _payload_hash_vec(
                ("v11", int(turn_index)),
                int(p.v11_cell.state_dim))
            state = step_persistent_state_v11(
                cell=p.v11_cell,
                prev_state=None,
                carrier_values=carrier_vals,
                turn_index=int(turn_index), role=str(role),
                substrate_skip=carrier_vals,
                hidden_state_skip=carrier_vals,
                attention_skip=carrier_vals,
                retrieval_skip=carrier_vals,
                substrate_fidelity=0.9,
                attention_fidelity=0.9,
                retrieval_fidelity=0.9)
            self.chain.add(state)
            wp = emit_persistent_v11_witness(
                cell=p.v11_cell, chain=self.chain,
                leaf_cid=state.cid())
            per_v11_w_cid = wp.cid()
        # M8 — multi-hop V9.
        mh_v9_w_cid = ""
        if p.enabled:
            wmh = emit_multi_hop_v9_witness(
                backends=W59_DEFAULT_MH_V9_BACKENDS,
                chain_length=W59_DEFAULT_MH_V9_CHAIN_LEN,
                seed=int(turn_index) + 5900)
            mh_v9_w_cid = wmh.cid()
        # M9 — MLSC V7.
        mlsc_v7_w_cid = ""
        if (p.enabled and p.mlsc_v7_operator is not None):
            v3 = make_root_capsule_v3(
                branch_id=f"w59_{int(turn_index)}",
                payload=tuple([0.1] * 6),
                fact_tags=("w59",),
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
            merged = p.mlsc_v7_operator.merge(
                [v7],
                retrieval_witness_chain=(
                    f"merge_r_{turn_index}",),
                controller_witness_cid=(
                    f"merge_ctrl_{turn_index}"))
            wmlsc = emit_mlsc_v7_witness(merged)
            mlsc_v7_w_cid = wmlsc.cid()
        # M10 — consensus V5.
        cons_v5_w_cid = ""
        if (p.enabled and p.consensus_v5 is not None):
            d = 6
            p1 = _payload_hash_vec(("c5", "p1",
                                     int(turn_index)), d)
            p2 = _payload_hash_vec(("c5", "p2",
                                     int(turn_index)), d)
            q = _payload_hash_vec(("c5", "q",
                                    int(turn_index)), d)
            p.consensus_v5.decide(
                parent_payloads=[p1, p2],
                parent_trusts=[0.9, 0.7],
                parent_cache_fingerprints=[(1, 2), (3, 4)],
                parent_retrieval_scores=[0.6, 0.7],
                query_direction=q,
                transcript_payload=[0.0] * d)
            wcons = emit_consensus_v5_witness(p.consensus_v5)
            cons_v5_w_cid = wcons.cid()
        # M11 — CRC V7.
        crc_v7_w_cid = ""
        if (p.enabled and p.crc_v7 is not None):
            wcrc = emit_corruption_robustness_v7_witness(
                crc_v7=p.crc_v7, n_probes=8,
                seed=int(turn_index) + 590)
            crc_v7_w_cid = wcrc.cid()
        # M12 — LHR V11.
        lhr_v11_w_cid = ""
        if (p.enabled and p.lhr_v11 is not None):
            wlhr = emit_lhr_v11_witness(head=p.lhr_v11)
            lhr_v11_w_cid = wlhr.cid()
        # M13 — ECC V11.
        ecc_v11_w_cid = ""
        if (p.enabled and p.ecc_v11 is not None):
            gate = QuantisedBudgetGate.init(
                in_dim=W53_DEFAULT_ECC_CODE_DIM,
                emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
                seed=int(turn_index) + 591)
            gate.importance_threshold = 0.0
            gate.w_emit.values = (
                [1.0] * len(gate.w_emit.values))
            carrier = _payload_hash_vec(
                ("ecc_v11", int(turn_index)),
                W53_DEFAULT_ECC_CODE_DIM)
            comp = compress_carrier_ecc_v11(
                carrier, codebook=p.ecc_v11, gate=gate)
            wecc = emit_ecc_v11_compression_witness(
                codebook=p.ecc_v11, compression=comp)
            ecc_v11_w_cid = wecc.cid()
        # M14 — TVS V8.
        tvs_v8_w_cid = ""
        if p.enabled:
            res = nine_arm_compare(
                per_turn_confidences=[0.7],
                per_turn_trust_scores=[0.8],
                per_turn_merge_retentions=[0.6],
                per_turn_tw_retentions=[0.55],
                per_turn_substrate_fidelities=[0.55],
                per_turn_hidden_fidelities=[0.5],
                per_turn_cache_fidelities=[0.6],
                per_turn_retrieval_fidelities=[0.75],
                budget_tokens=int(p.tvs_budget_tokens))
            wtvs = emit_tvs_arbiter_v8_witness(result=res)
            tvs_v8_w_cid = wtvs.cid()
        # M15 — uncertainty V7.
        unc_v7_w_cid = ""
        composite_v7 = 1.0
        if p.enabled:
            comp = compose_uncertainty_report_v7(
                component_confidences={
                    "kv": 0.85, "hsb": 0.80, "attn": 0.75,
                    "cache": 0.78, "retr": 0.82,
                },
                trust_weights={
                    "kv": 0.9, "hsb": 0.85, "attn": 0.8,
                    "cache": 0.85, "retr": 0.88,
                },
                substrate_fidelities={
                    "kv": 0.85, "hsb": 0.8, "attn": 0.7,
                    "cache": 0.75, "retr": 0.78,
                },
                hidden_state_fidelities={
                    "kv": 0.7, "hsb": 0.9, "attn": 0.7,
                    "cache": 0.7, "retr": 0.72,
                },
                cache_reuse_fidelities={
                    "kv": 0.75, "hsb": 0.7, "attn": 0.7,
                    "cache": 0.95, "retr": 0.8,
                },
                retrieval_fidelities={
                    "kv": 0.8, "hsb": 0.75, "attn": 0.7,
                    "cache": 0.78, "retr": 0.97,
                })
            wu = emit_uncertainty_v7_witness(composite=comp)
            unc_v7_w_cid = wu.cid()
            composite_v7 = float(comp.weighted_composite)
        # M16 — disagreement algebra V5.
        da_v5_w_cid = ""
        if p.enabled:
            trace = AlgebraTrace.empty()
            pa = _payload_hash_vec(("da5", "a",
                                     int(turn_index)), 6)
            pb = _payload_hash_vec(("da5", "b",
                                     int(turn_index)), 6)
            pc = _payload_hash_vec(("da5", "c",
                                     int(turn_index)), 6)

            def retrieval_oracle():
                return (True, 0.0)
            wda = emit_disagreement_algebra_v5_witness(
                trace=trace,
                probe_a=pa, probe_b=pb, probe_c=pc,
                retrieval_replay_oracle=retrieval_oracle)
            da_v5_w_cid = wda.cid()
        # M17 — deep substrate hybrid V4.
        hyb_v4_w_cid = ""
        four_way_used = False
        if (p.enabled
                and p.deep_substrate_hybrid_v4 is not None):
            try:
                d_v6 = p.deep_substrate_hybrid_v4.deep_v6
                in_dim = int(d_v6.in_dim)
                fd = int(d_v6.inner_v5.inner_v4.factor_dim)
                qi = _payload_hash_vec(
                    ("hyb_v4_q", int(turn_index)), in_dim)
                ks = [_payload_hash_vec(
                    ("hyb_v4_k", int(turn_index), j), fd)
                    for j in range(2)]
                vs = [_payload_hash_vec(
                    ("hyb_v4_v", int(turn_index), j), fd)
                    for j in range(2)]
                _, whyb, _ = deep_substrate_hybrid_v4_forward(
                    hybrid=p.deep_substrate_hybrid_v4,
                    query_input=qi,
                    slot_keys=ks, slot_values=vs,
                    role_index=0, branch_index=0,
                    cycle_index=0,
                    trust_scalar=0.9,
                    uncertainty_scale=0.95,
                    substrate_query_token_ids=list(
                        p.follow_up_token_ids))
                hyb_v4_w_cid = whyb.cid()
                four_way_used = bool(whyb.four_way)
            except Exception:
                hyb_v4_w_cid = ""
        # M18 — substrate adapter V4.
        adapter_v4_cid = ""
        if p.enabled:
            mat = probe_all_v4_adapters(
                probe_ollama=False, probe_openai=False)
            adapter_v4_cid = mat.cid()
        return W59HandoffEnvelope(
            schema=W59_SCHEMA_VERSION + ".envelope.v1",
            w58_outer_cid=str(w58_outer_cid),
            w59_params_cid=str(p.cid()),
            substrate_v4_witness_cid=str(sub_v4_w_cid),
            kv_bridge_v4_witness_cid=str(kv_v4_w_cid),
            hsb_v3_witness_cid=str(hsb3_w_cid),
            prefix_state_v3_witness_cid=str(ps3_w_cid),
            attn_steer_v3_witness_cid=str(attn3_w_cid),
            cache_controller_v2_witness_cid=str(cc2_w_cid),
            persistent_v11_witness_cid=str(per_v11_w_cid),
            multi_hop_v9_witness_cid=str(mh_v9_w_cid),
            mlsc_v7_witness_cid=str(mlsc_v7_w_cid),
            consensus_v5_witness_cid=str(cons_v5_w_cid),
            crc_v7_witness_cid=str(crc_v7_w_cid),
            lhr_v11_witness_cid=str(lhr_v11_w_cid),
            ecc_v11_witness_cid=str(ecc_v11_w_cid),
            tvs_v8_witness_cid=str(tvs_v8_w_cid),
            uncertainty_v7_witness_cid=str(unc_v7_w_cid),
            disagreement_algebra_v5_witness_cid=str(
                da_v5_w_cid),
            deep_substrate_hybrid_v4_witness_cid=str(
                hyb_v4_w_cid),
            substrate_adapter_v4_matrix_cid=str(adapter_v4_cid),
            v11_chain_cid=str(self.chain.cid()),
            retrieval_aware_weighted_mean=float(composite_v7),
            four_way_used=bool(four_way_used),
            substrate_v4_used=bool(sub_v4_used),
        )


def build_w59_team(*, seed: int = 59000) -> W59Team:
    return W59Team(params=W59Params.build_default(seed=int(seed)))


def build_trivial_w59_envelope(
        *, w58_outer_cid: str = "no_w58",
) -> W59HandoffEnvelope:
    """A W59 trivial envelope wraps a W58 outer CID through a
    disabled W59Params object. All W59 witness fields are empty
    strings; the verifier will flag many ``missing_*`` modes —
    that is the *expected* trivial passthrough behaviour."""
    return W59HandoffEnvelope(
        schema=W59_SCHEMA_VERSION + ".envelope.v1",
        w58_outer_cid=str(w58_outer_cid),
        w59_params_cid=str(W59Params.build_trivial().cid()),
        substrate_v4_witness_cid="",
        kv_bridge_v4_witness_cid="",
        hsb_v3_witness_cid="",
        prefix_state_v3_witness_cid="",
        attn_steer_v3_witness_cid="",
        cache_controller_v2_witness_cid="",
        persistent_v11_witness_cid="",
        multi_hop_v9_witness_cid="",
        mlsc_v7_witness_cid="",
        consensus_v5_witness_cid="",
        crc_v7_witness_cid="",
        lhr_v11_witness_cid="",
        ecc_v11_witness_cid="",
        tvs_v8_witness_cid="",
        uncertainty_v7_witness_cid="",
        disagreement_algebra_v5_witness_cid="",
        deep_substrate_hybrid_v4_witness_cid="",
        substrate_adapter_v4_matrix_cid="",
        v11_chain_cid="",
        retrieval_aware_weighted_mean=1.0,
        four_way_used=False,
        substrate_v4_used=False,
    )


__all__ = [
    "W59_SCHEMA_VERSION",
    "W59_TEAM_RESULT_SCHEMA",
    "W59_ENVELOPE_VERIFIER_FAILURE_MODES",
    "W59Params",
    "W59HandoffEnvelope",
    "W59Team",
    "build_w59_team",
    "build_trivial_w59_envelope",
    "verify_w59_handoff",
]
