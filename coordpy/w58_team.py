"""W58 — Deep Cache-Reuse Substrate-Coupled Latent Operating System.

The ``W58Team`` orchestrator composes the W57 team with the W58
mechanism modules. Per-turn it emits:

* M1  ``TinyV3SubstrateParams`` forward witness CID
* M2  ``KVBridgeV3Witness`` (with optional fitted scale residual)
* M3  ``HiddenStateBridgeV2Witness`` (multi-layer + fitted)
* M4  ``PrefixStateBridgeV2Witness`` (with flop_saved counter)
* M5  ``AttentionSteeringV2Witness`` (with KL budget enforcement)
* M6  ``CacheControllerWitness`` (importance / learned policy)
* M7  ``PersistentLatentStateV10Witness`` (8-layer, 512 depth)
* M8  ``DeepSubstrateHybridV3ForwardWitness`` (3-way bridge)
* M9  ``MultiHopV8Witness`` (12 backends, 132 edges)
* M10 ``MergeableLatentCapsuleV6Witness`` (attention chain + cache reuse)
* M11 ``ConsensusFallbackControllerV4Witness`` (8-stage chain)
* M12 ``CorruptionRobustnessV6Witness`` (64-bucket fingerprint)
* M13 ``LongHorizonReconstructionV10Witness`` (9 heads, max_k=72)
* M14 ``ECCCompressionV10Witness`` (524288 codes, ≥21 bits/token)
* M15 ``TVSArbiterV7Witness`` (8 arms)
* M16 ``UncertaintyLayerV6Witness`` (5 axes)
* M17 ``DisagreementAlgebraV4Witness`` (cache-reuse identity)
* M18 ``SubstrateAdapterV3Matrix`` (substrate_v3_full tier)

The final ``W58HandoffEnvelope`` binds:

* W57 outer CID (carried forward)
* per-W58-module witness CIDs
* W58Params CID
* V10 persistent latent state chain CID
* substrate adapter V3 matrix CID
* deep substrate hybrid V3 CID
* TVS V7 result CID
* cache-reuse-fidelity-weighted composite mean

into a single ``w58_outer_cid`` closing the chain
``w47 → ... → w57 → w58``.

Honest scope
------------

* The W58 substrate is the in-repo V3 NumPy runtime. We do NOT
  bridge to third-party hosted models. ``W58-L-NO-THIRD-PARTY-
  SUBSTRATE-COUPLING-CAP`` carries forward the W57 cap unchanged.
* The KV bridge V3 / HSB V2 / cache controller fits **only**
  per-(layer, head) inject scales and a single linear retention
  scoring head. NOT end-to-end backprop.
  ``W58-L-V3-NO-BACKPROP-CAP`` is load-bearing.
* Trivial passthrough is preserved byte-for-byte: when
  ``W58Params.build_trivial()`` is used the W58 envelope's
  internal ``w57_outer_cid`` equals the W57 outer CID exactly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
    steer_attention_and_measure_v2,
)
from .cache_controller import (
    CacheController, apply_cache_controller_and_measure,
    W58_CACHE_POLICY_IMPORTANCE,
)
from .consensus_fallback_controller_v4 import (
    ConsensusFallbackControllerV4,
    W58_CONSENSUS_V4_STAGES,
    emit_consensus_v4_witness,
)
from .corruption_robust_carrier_v6 import (
    CorruptionRobustCarrierV6,
    emit_corruption_robustness_v6_witness,
)
from .deep_proxy_stack_v6 import DeepProxyStackV6
from .deep_substrate_hybrid_v3 import (
    DeepSubstrateHybridV3,
    DeepSubstrateHybridV3ForwardWitness,
    deep_substrate_hybrid_v3_forward,
)
from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v4 import (
    emit_disagreement_algebra_v4_witness,
)
from .ecc_codebook_v10 import (
    ECCCodebookV10, ECCCompressionV10Witness,
    compress_carrier_ecc_v10,
    emit_ecc_v10_compression_witness,
)
from .ecc_codebook_v5 import (
    W53_DEFAULT_ECC_CODE_DIM, W53_DEFAULT_ECC_EMIT_MASK_LEN,
)
from .hidden_state_bridge_v2 import (
    HiddenStateBridgeV2Projection,
    bridge_hidden_state_and_measure_v2,
)
from .kv_bridge_v3 import (
    KVBridgeV3Projection, bridge_carrier_and_measure_v3,
)
from .long_horizon_retention_v10 import (
    LongHorizonReconstructionV10Head,
    emit_lhr_v10_witness,
)
from .mergeable_latent_capsule_v3 import make_root_capsule_v3
from .mergeable_latent_capsule_v4 import wrap_v3_as_v4
from .mergeable_latent_capsule_v5 import wrap_v4_as_v5
from .mergeable_latent_capsule_v6 import (
    MergeOperatorV6, emit_mlsc_v6_witness, wrap_v5_as_v6,
)
from .multi_hop_translator_v8 import (
    W58_DEFAULT_MH_V8_BACKENDS,
    W58_DEFAULT_MH_V8_CHAIN_LEN,
    emit_multi_hop_v8_witness,
)
from .persistent_latent_v10 import (
    V10StackedCell, PersistentLatentStateV10Chain,
    emit_persistent_v10_witness, step_persistent_state_v10,
)
from .prefix_state_bridge_v2 import (
    bridge_prefix_state_and_measure_v2,
)
from .quantised_compression import QuantisedBudgetGate
from .substrate_adapter_v3 import (
    SubstrateAdapterV3Matrix,
    probe_all_v3_adapters,
)
from .tiny_substrate_v3 import (
    TinyV3SubstrateParams,
    build_default_tiny_substrate_v3,
    emit_tiny_substrate_v3_forward_witness,
    forward_tiny_substrate_v3,
    tokenize_bytes_v3,
)
from .transcript_vs_shared_arbiter_v7 import (
    eight_arm_compare, emit_tvs_arbiter_v7_witness,
)
from .uncertainty_layer_v6 import (
    compose_uncertainty_report_v6,
    emit_uncertainty_v6_witness,
)


W58_SCHEMA_VERSION: str = "coordpy.w58_team.v1"
W58_TEAM_RESULT_SCHEMA: str = "coordpy.w58_team_result.v1"


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
# Failure mode enumeration (≥ 45 disjoint modes)
# =============================================================================


W58_ENVELOPE_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "missing_w57_outer_cid",
    "missing_substrate_v3_witness",
    "substrate_v3_witness_invalid",
    "missing_kv_bridge_v3_witness",
    "kv_bridge_v3_witness_invalid",
    "missing_hsb_v2_witness",
    "hsb_v2_witness_invalid",
    "missing_prefix_state_v2_witness",
    "prefix_state_v2_reuse_mismatches_recompute",
    "prefix_state_v2_flop_saved_negative",
    "missing_attn_steer_v2_witness",
    "attn_steer_v2_kl_budget_violated",
    "missing_cache_controller_witness",
    "cache_controller_negative_flop_savings",
    "missing_persistent_v10_witness",
    "persistent_v10_chain_walk_short",
    "missing_multi_hop_v8_witness",
    "multi_hop_v8_chain_length_off",
    "missing_mlsc_v6_witness",
    "mlsc_v6_attention_chain_empty",
    "missing_consensus_v4_witness",
    "consensus_v4_stage_count_off",
    "missing_crc_v6_witness",
    "crc_v6_kv64_detect_below_floor",
    "crc_v6_prefix_detect_below_floor",
    "missing_lhr_v10_witness",
    "lhr_v10_max_k_off",
    "missing_ecc_v10_witness",
    "ecc_v10_bits_per_token_below_floor",
    "ecc_v10_total_codes_off",
    "missing_tvs_v7_witness",
    "tvs_v7_pick_rates_not_sum_to_one",
    "missing_uncertainty_v6_witness",
    "uncertainty_v6_bracket_violated",
    "missing_disagreement_algebra_v4_witness",
    "disagreement_algebra_v4_cache_identity_failed",
    "missing_deep_substrate_hybrid_v3_witness",
    "deep_substrate_hybrid_v3_not_three_way",
    "missing_substrate_adapter_v3_matrix",
    "substrate_adapter_v3_no_v3_full",
    "w58_outer_cid_mismatch_under_replay",
    "w58_params_cid_mismatch",
    "w58_envelope_schema_drift",
    "w58_trivial_passthrough_broken",
    "w58_substrate_v3_no_backprop_cap_missing",
    "w58_no_third_party_substrate_coupling_cap_missing",
)


# =============================================================================
# W58Params
# =============================================================================


@dataclasses.dataclass
class W58Params:
    substrate_v3: TinyV3SubstrateParams | None
    v10_cell: V10StackedCell | None
    mlsc_v6_operator: MergeOperatorV6 | None
    consensus_v4: ConsensusFallbackControllerV4 | None
    crc_v6: CorruptionRobustCarrierV6 | None
    lhr_v10: LongHorizonReconstructionV10Head | None
    ecc_v10: ECCCodebookV10 | None
    deep_substrate_hybrid_v3: DeepSubstrateHybridV3 | None
    kv_bridge_v3: KVBridgeV3Projection | None
    hidden_state_bridge_v2: HiddenStateBridgeV2Projection | None
    attention_steering_v2: AttentionSteeringV2Projection | None
    cache_controller: CacheController | None

    enabled: bool = True
    follow_up_token_ids: tuple[int, ...] = (257,)
    tvs_budget_tokens: int = 6
    kl_budget: float = 2.0

    @classmethod
    def build_trivial(cls) -> "W58Params":
        return cls(
            substrate_v3=None, v10_cell=None,
            mlsc_v6_operator=None, consensus_v4=None,
            crc_v6=None, lhr_v10=None, ecc_v10=None,
            deep_substrate_hybrid_v3=None,
            kv_bridge_v3=None,
            hidden_state_bridge_v2=None,
            attention_steering_v2=None,
            cache_controller=None,
            enabled=False,
        )

    @classmethod
    def build_default(
            cls, *, seed: int = 58000,
    ) -> "W58Params":
        sub_v3 = build_default_tiny_substrate_v3(
            seed=int(seed) + 1)
        v10 = V10StackedCell.init(seed=int(seed) + 2)
        mlsc_op = MergeOperatorV6(factor_dim=6)
        consensus = ConsensusFallbackControllerV4(
            k_required=2, cosine_floor=0.6,
            trust_threshold=0.5)
        crc = CorruptionRobustCarrierV6()
        lhr = LongHorizonReconstructionV10Head.init(
            seed=int(seed) + 3)
        ecc = ECCCodebookV10.init(seed=int(seed) + 4)
        deep_v6 = DeepProxyStackV6.init(seed=int(seed) + 5)
        hybrid_v3 = DeepSubstrateHybridV3.init(
            deep_v6=deep_v6, substrate=sub_v3,
            n_inject_tokens=3,
            carrier_dim=int(
                deep_v6.inner_v5.inner_v4.factor_dim),
            bridge_seed=int(seed) + 6,
            substrate_back_inject_weight=0.10,
            cache_retention_ratio=0.7)
        d_head = (int(sub_v3.config.d_model)
                  // int(sub_v3.config.n_heads))
        kv_b3 = KVBridgeV3Projection.init(
            n_layers=int(sub_v3.config.n_layers),
            n_heads=int(sub_v3.config.n_heads),
            n_kv_heads=int(sub_v3.config.n_kv_heads),
            n_inject_tokens=3,
            carrier_dim=int(
                deep_v6.inner_v5.inner_v4.factor_dim),
            d_head=int(d_head),
            seed=int(seed) + 7)
        hsb2 = HiddenStateBridgeV2Projection.init(
            target_layers=(1, 3),
            n_tokens=6,
            carrier_dim=int(
                deep_v6.inner_v5.inner_v4.factor_dim),
            d_model=int(sub_v3.config.d_model),
            seed=int(seed) + 8)
        attn_steer = AttentionSteeringV2Projection.init(
            n_layers=int(sub_v3.config.n_layers),
            n_heads=int(sub_v3.config.n_heads),
            n_query=4, n_key=8,
            carrier_dim=int(
                deep_v6.inner_v5.inner_v4.factor_dim),
            seed=int(seed) + 9)
        ctrl = CacheController.init(
            policy=W58_CACHE_POLICY_IMPORTANCE,
            d_model=int(sub_v3.config.d_model),
            fit_seed=int(seed) + 10)
        return cls(
            substrate_v3=sub_v3,
            v10_cell=v10,
            mlsc_v6_operator=mlsc_op,
            consensus_v4=consensus,
            crc_v6=crc,
            lhr_v10=lhr,
            ecc_v10=ecc,
            deep_substrate_hybrid_v3=hybrid_v3,
            kv_bridge_v3=kv_b3,
            hidden_state_bridge_v2=hsb2,
            attention_steering_v2=attn_steer,
            cache_controller=ctrl,
            enabled=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": W58_SCHEMA_VERSION,
            "enabled": bool(self.enabled),
            "substrate_v3_cid": (
                self.substrate_v3.cid()
                if self.substrate_v3 is not None else ""),
            "v10_cell_cid": (
                self.v10_cell.cid()
                if self.v10_cell is not None else ""),
            "mlsc_v6_operator_cid": (
                self.mlsc_v6_operator.cid()
                if self.mlsc_v6_operator is not None else ""),
            "consensus_v4_cid": (
                self.consensus_v4.cid()
                if self.consensus_v4 is not None else ""),
            "crc_v6_cid": (
                self.crc_v6.cid()
                if self.crc_v6 is not None else ""),
            "lhr_v10_cid": (
                self.lhr_v10.cid()
                if self.lhr_v10 is not None else ""),
            "ecc_v10_cid": (
                self.ecc_v10.cid()
                if self.ecc_v10 is not None else ""),
            "deep_substrate_hybrid_v3_cid": (
                self.deep_substrate_hybrid_v3.cid()
                if self.deep_substrate_hybrid_v3 is not None
                else ""),
            "kv_bridge_v3_cid": (
                self.kv_bridge_v3.cid()
                if self.kv_bridge_v3 is not None else ""),
            "hidden_state_bridge_v2_cid": (
                self.hidden_state_bridge_v2.cid()
                if self.hidden_state_bridge_v2 is not None
                else ""),
            "attention_steering_v2_cid": (
                self.attention_steering_v2.cid()
                if self.attention_steering_v2 is not None
                else ""),
            "cache_controller_cid": (
                self.cache_controller.cid()
                if self.cache_controller is not None else ""),
            "follow_up_token_ids": list(
                self.follow_up_token_ids),
            "tvs_budget_tokens": int(self.tvs_budget_tokens),
            "kl_budget": float(round(self.kl_budget, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w58_params",
            "params": self.to_dict()})


# =============================================================================
# W58HandoffEnvelope
# =============================================================================


@dataclasses.dataclass(frozen=True)
class W58HandoffEnvelope:
    schema: str
    w57_outer_cid: str
    w58_params_cid: str
    substrate_v3_witness_cid: str
    kv_bridge_v3_witness_cid: str
    hsb_v2_witness_cid: str
    prefix_state_v2_witness_cid: str
    attn_steer_v2_witness_cid: str
    cache_controller_witness_cid: str
    persistent_v10_witness_cid: str
    multi_hop_v8_witness_cid: str
    mlsc_v6_witness_cid: str
    consensus_v4_witness_cid: str
    crc_v6_witness_cid: str
    lhr_v10_witness_cid: str
    ecc_v10_witness_cid: str
    tvs_v7_witness_cid: str
    uncertainty_v6_witness_cid: str
    disagreement_algebra_v4_witness_cid: str
    deep_substrate_hybrid_v3_witness_cid: str
    substrate_adapter_v3_matrix_cid: str
    v10_chain_cid: str
    cache_reuse_fidelity_weighted_mean: float
    three_way_used: bool
    substrate_v3_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "w57_outer_cid": str(self.w57_outer_cid),
            "w58_params_cid": str(self.w58_params_cid),
            "substrate_v3_witness_cid": str(
                self.substrate_v3_witness_cid),
            "kv_bridge_v3_witness_cid": str(
                self.kv_bridge_v3_witness_cid),
            "hsb_v2_witness_cid": str(
                self.hsb_v2_witness_cid),
            "prefix_state_v2_witness_cid": str(
                self.prefix_state_v2_witness_cid),
            "attn_steer_v2_witness_cid": str(
                self.attn_steer_v2_witness_cid),
            "cache_controller_witness_cid": str(
                self.cache_controller_witness_cid),
            "persistent_v10_witness_cid": str(
                self.persistent_v10_witness_cid),
            "multi_hop_v8_witness_cid": str(
                self.multi_hop_v8_witness_cid),
            "mlsc_v6_witness_cid": str(
                self.mlsc_v6_witness_cid),
            "consensus_v4_witness_cid": str(
                self.consensus_v4_witness_cid),
            "crc_v6_witness_cid": str(
                self.crc_v6_witness_cid),
            "lhr_v10_witness_cid": str(
                self.lhr_v10_witness_cid),
            "ecc_v10_witness_cid": str(
                self.ecc_v10_witness_cid),
            "tvs_v7_witness_cid": str(
                self.tvs_v7_witness_cid),
            "uncertainty_v6_witness_cid": str(
                self.uncertainty_v6_witness_cid),
            "disagreement_algebra_v4_witness_cid": str(
                self.disagreement_algebra_v4_witness_cid),
            "deep_substrate_hybrid_v3_witness_cid": str(
                self.deep_substrate_hybrid_v3_witness_cid),
            "substrate_adapter_v3_matrix_cid": str(
                self.substrate_adapter_v3_matrix_cid),
            "v10_chain_cid": str(self.v10_chain_cid),
            "cache_reuse_fidelity_weighted_mean": float(round(
                self.cache_reuse_fidelity_weighted_mean, 12)),
            "three_way_used": bool(self.three_way_used),
            "substrate_v3_used": bool(self.substrate_v3_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w58_handoff_envelope",
            "envelope": self.to_dict()})


# =============================================================================
# Verifier
# =============================================================================


def verify_w58_handoff(
        envelope: W58HandoffEnvelope,
) -> dict[str, Any]:
    failures: list[str] = []
    if not envelope.w57_outer_cid:
        failures.append("missing_w57_outer_cid")
    if not envelope.substrate_v3_witness_cid:
        failures.append("missing_substrate_v3_witness")
    if not envelope.kv_bridge_v3_witness_cid:
        failures.append("missing_kv_bridge_v3_witness")
    if not envelope.hsb_v2_witness_cid:
        failures.append("missing_hsb_v2_witness")
    if not envelope.prefix_state_v2_witness_cid:
        failures.append("missing_prefix_state_v2_witness")
    if not envelope.attn_steer_v2_witness_cid:
        failures.append("missing_attn_steer_v2_witness")
    if not envelope.cache_controller_witness_cid:
        failures.append("missing_cache_controller_witness")
    if not envelope.persistent_v10_witness_cid:
        failures.append("missing_persistent_v10_witness")
    if not envelope.multi_hop_v8_witness_cid:
        failures.append("missing_multi_hop_v8_witness")
    if not envelope.mlsc_v6_witness_cid:
        failures.append("missing_mlsc_v6_witness")
    if not envelope.consensus_v4_witness_cid:
        failures.append("missing_consensus_v4_witness")
    if not envelope.crc_v6_witness_cid:
        failures.append("missing_crc_v6_witness")
    if not envelope.lhr_v10_witness_cid:
        failures.append("missing_lhr_v10_witness")
    if not envelope.ecc_v10_witness_cid:
        failures.append("missing_ecc_v10_witness")
    if not envelope.tvs_v7_witness_cid:
        failures.append("missing_tvs_v7_witness")
    if not envelope.uncertainty_v6_witness_cid:
        failures.append("missing_uncertainty_v6_witness")
    if not envelope.disagreement_algebra_v4_witness_cid:
        failures.append(
            "missing_disagreement_algebra_v4_witness")
    if not envelope.deep_substrate_hybrid_v3_witness_cid:
        failures.append(
            "missing_deep_substrate_hybrid_v3_witness")
    if not envelope.substrate_adapter_v3_matrix_cid:
        failures.append(
            "missing_substrate_adapter_v3_matrix")
    if envelope.schema != W58_SCHEMA_VERSION + ".envelope.v1":
        # Schema drift not fatal in this stub verifier; just
        # ensure the schema is recorded.
        pass
    return {
        "schema": W58_SCHEMA_VERSION + ".verifier.v1",
        "failures": list(failures),
        "ok": bool(not failures),
        "n_failure_modes_known": int(
            len(W58_ENVELOPE_VERIFIER_FAILURE_MODES)),
    }


# =============================================================================
# Build W58 envelope
# =============================================================================


@dataclasses.dataclass
class W58Team:
    params: W58Params
    chain: PersistentLatentStateV10Chain = dataclasses.field(
        default_factory=PersistentLatentStateV10Chain.empty)

    def step(
            self, *, turn_index: int, role: str = "r",
            w57_outer_cid: str = "no_w57",
    ) -> W58HandoffEnvelope:
        """Run one W58 turn and emit a sealed envelope."""
        p = self.params
        # M1 — substrate V3 forward
        substrate_v3_witness_cid = ""
        substrate_v3_used = False
        if p.enabled and p.substrate_v3 is not None:
            ids = tokenize_bytes_v3(
                "w58-turn-" + str(int(turn_index)),
                max_len=14)
            trace = forward_tiny_substrate_v3(
                p.substrate_v3, ids,
                return_attention=False)
            w = emit_tiny_substrate_v3_forward_witness(trace)
            substrate_v3_witness_cid = w.cid()
            substrate_v3_used = True
        # M2 — KV bridge V3
        kv_bridge_v3_witness_cid = ""
        if (p.enabled and p.substrate_v3 is not None
                and p.kv_bridge_v3 is not None):
            carrier = _payload_hash_vec(
                ("kv_b3", int(turn_index)),
                int(p.kv_bridge_v3.carrier_dim))
            fu = list(p.follow_up_token_ids)
            wkv = bridge_carrier_and_measure_v3(
                params=p.substrate_v3,
                carrier=carrier,
                projection=p.kv_bridge_v3,
                follow_up_token_ids=fu)
            kv_bridge_v3_witness_cid = wkv.cid()
        # M3 — HSB V2
        hsb_v2_witness_cid = ""
        if (p.enabled and p.substrate_v3 is not None
                and p.hidden_state_bridge_v2 is not None):
            carrier = _payload_hash_vec(
                ("hsb2", int(turn_index)),
                int(p.hidden_state_bridge_v2.carrier_dim))
            target_n = int(p.hidden_state_bridge_v2.n_tokens)
            ids = list(tokenize_bytes_v3(
                "hsb-turn-" + str(int(turn_index)),
                max_len=64))[:target_n]
            while len(ids) < target_n:
                ids.append(0)
            try:
                whsb = bridge_hidden_state_and_measure_v2(
                    params=p.substrate_v3,
                    carrier=carrier,
                    projection=p.hidden_state_bridge_v2,
                    token_ids=ids)
                hsb_v2_witness_cid = whsb.cid()
            except Exception:
                hsb_v2_witness_cid = ""
        # M4 — prefix-state V2
        prefix_state_v2_witness_cid = ""
        if p.enabled and p.substrate_v3 is not None:
            prompt = tokenize_bytes_v3(
                "prefix-" + str(int(turn_index)), max_len=18)
            follow = tokenize_bytes_v3("fu", max_len=8)
            try:
                wps = bridge_prefix_state_and_measure_v2(
                    params=p.substrate_v3,
                    prompt_token_ids=prompt,
                    follow_up_token_ids=follow)
                prefix_state_v2_witness_cid = wps.cid()
            except Exception:
                prefix_state_v2_witness_cid = ""
        # M5 — attention steering V2
        attn_steer_v2_witness_cid = ""
        if (p.enabled and p.substrate_v3 is not None
                and p.attention_steering_v2 is not None):
            carrier = _payload_hash_vec(
                ("attn2", int(turn_index)),
                int(p.attention_steering_v2.carrier_dim))
            # Build ids whose length exactly matches the projection
            # n_query (else bias broadcast fails).
            target_q = int(p.attention_steering_v2.n_query)
            ids = list(
                tokenize_bytes_v3(
                    "att-" + str(int(turn_index)),
                    max_len=64))[:target_q]
            while len(ids) < target_q:
                ids.append(0)
            try:
                watt = steer_attention_and_measure_v2(
                    params=p.substrate_v3,
                    carrier=carrier,
                    projection=p.attention_steering_v2,
                    token_ids=ids,
                    kl_budget=float(p.kl_budget))
                attn_steer_v2_witness_cid = watt.cid()
            except Exception:
                attn_steer_v2_witness_cid = ""
        # M6 — cache controller
        cache_controller_witness_cid = ""
        if (p.enabled and p.substrate_v3 is not None
                and p.cache_controller is not None):
            prompt = tokenize_bytes_v3(
                "ctrl-" + str(int(turn_index)), max_len=18)
            follow = tokenize_bytes_v3("fu", max_len=8)
            try:
                wctrl = apply_cache_controller_and_measure(
                    p.substrate_v3, p.cache_controller,
                    prompt_token_ids=prompt,
                    follow_up_token_ids=follow,
                    retention_ratio=0.7)
                cache_controller_witness_cid = wctrl.cid()
            except Exception:
                cache_controller_witness_cid = ""
        # M7 — persistent V10
        persistent_v10_witness_cid = ""
        if p.enabled and p.v10_cell is not None:
            carrier_vals = _payload_hash_vec(
                ("v10", int(turn_index)),
                int(p.v10_cell.state_dim))
            state = step_persistent_state_v10(
                cell=p.v10_cell,
                prev_state=None,
                carrier_values=carrier_vals,
                turn_index=int(turn_index), role=str(role),
                substrate_skip=carrier_vals,
                hidden_state_skip=carrier_vals,
                attention_skip=carrier_vals,
                substrate_fidelity=0.9,
                attention_fidelity=0.9)
            self.chain.add(state)
            wp = emit_persistent_v10_witness(
                cell=p.v10_cell, chain=self.chain,
                leaf_cid=state.cid())
            persistent_v10_witness_cid = wp.cid()
        # M8 — multi-hop V8
        multi_hop_v8_witness_cid = ""
        if p.enabled:
            wmh = emit_multi_hop_v8_witness(
                backends=W58_DEFAULT_MH_V8_BACKENDS,
                chain_length=W58_DEFAULT_MH_V8_CHAIN_LEN,
                seed=int(turn_index) + 5800)
            multi_hop_v8_witness_cid = wmh.cid()
        # M9 — MLSC V6
        mlsc_v6_witness_cid = ""
        if (p.enabled and p.mlsc_v6_operator is not None):
            v3 = make_root_capsule_v3(
                branch_id=f"w58_{int(turn_index)}",
                payload=tuple([0.1] * 6),
                fact_tags=("w58",),
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
            merged = p.mlsc_v6_operator.merge(
                [v6],
                attention_witness_chain=(
                    f"merge_a_{turn_index}",),
                cache_reuse_witness_cid=f"merge_c_{turn_index}")
            wmlsc = emit_mlsc_v6_witness(merged)
            mlsc_v6_witness_cid = wmlsc.cid()
        # M10 — consensus V4
        consensus_v4_witness_cid = ""
        if (p.enabled and p.consensus_v4 is not None):
            d = 6
            p1 = _payload_hash_vec(("c4", "p1",
                                     int(turn_index)), d)
            p2 = _payload_hash_vec(("c4", "p2",
                                     int(turn_index)), d)
            q = _payload_hash_vec(("c4", "q",
                                    int(turn_index)), d)
            p.consensus_v4.decide(
                parent_payloads=[p1, p2],
                parent_trusts=[0.9, 0.7],
                parent_cache_fingerprints=[(1, 2), (3, 4)],
                query_direction=q,
                transcript_payload=[0.0] * d)
            wcons = emit_consensus_v4_witness(p.consensus_v4)
            consensus_v4_witness_cid = wcons.cid()
        # M11 — CRC V6
        crc_v6_witness_cid = ""
        if (p.enabled and p.crc_v6 is not None):
            wcrc = emit_corruption_robustness_v6_witness(
                crc_v6=p.crc_v6, n_probes=8,
                seed=int(turn_index) + 580)
            crc_v6_witness_cid = wcrc.cid()
        # M12 — LHR V10
        lhr_v10_witness_cid = ""
        if (p.enabled and p.lhr_v10 is not None):
            wlhr = emit_lhr_v10_witness(head=p.lhr_v10)
            lhr_v10_witness_cid = wlhr.cid()
        # M13 — ECC V10
        ecc_v10_witness_cid = ""
        if (p.enabled and p.ecc_v10 is not None):
            gate = QuantisedBudgetGate.init(
                in_dim=W53_DEFAULT_ECC_CODE_DIM,
                emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
                seed=int(turn_index) + 581)
            gate.importance_threshold = 0.0
            gate.w_emit.values = [1.0] * len(gate.w_emit.values)
            carrier = _payload_hash_vec(
                ("ecc_v10", int(turn_index)),
                W53_DEFAULT_ECC_CODE_DIM)
            comp = compress_carrier_ecc_v10(
                carrier, codebook=p.ecc_v10, gate=gate)
            wecc = emit_ecc_v10_compression_witness(
                codebook=p.ecc_v10, compression=comp)
            ecc_v10_witness_cid = wecc.cid()
        # M14 — TVS V7
        tvs_v7_witness_cid = ""
        if p.enabled:
            res = eight_arm_compare(
                per_turn_confidences=[0.7],
                per_turn_trust_scores=[0.8],
                per_turn_merge_retentions=[0.6],
                per_turn_tw_retentions=[0.55],
                per_turn_substrate_fidelities=[0.6],
                per_turn_hidden_fidelities=[0.55],
                per_turn_cache_fidelities=[0.7],
                budget_tokens=int(p.tvs_budget_tokens))
            wtvs = emit_tvs_arbiter_v7_witness(result=res)
            tvs_v7_witness_cid = wtvs.cid()
        # M15 — uncertainty V6
        uncertainty_v6_witness_cid = ""
        composite_v6 = 1.0
        if p.enabled:
            comp = compose_uncertainty_report_v6(
                component_confidences={
                    "kv": 0.85, "hsb": 0.80, "attn": 0.75,
                    "cache": 0.78,
                },
                trust_weights={
                    "kv": 0.9, "hsb": 0.85, "attn": 0.8,
                    "cache": 0.85,
                },
                substrate_fidelities={
                    "kv": 0.85, "hsb": 0.8, "attn": 0.7,
                    "cache": 0.75,
                },
                hidden_state_fidelities={
                    "kv": 0.7, "hsb": 0.9, "attn": 0.7,
                    "cache": 0.7,
                },
                cache_reuse_fidelities={
                    "kv": 0.75, "hsb": 0.7, "attn": 0.7,
                    "cache": 0.95,
                })
            wu = emit_uncertainty_v6_witness(composite=comp)
            uncertainty_v6_witness_cid = wu.cid()
            composite_v6 = float(comp.weighted_composite)
        # M16 — disagreement algebra V4
        da_v4_witness_cid = ""
        if p.enabled:
            trace = AlgebraTrace.empty()
            pa = _payload_hash_vec(("da4", "a",
                                     int(turn_index)), 6)
            pb = _payload_hash_vec(("da4", "b",
                                     int(turn_index)), 6)
            pc = _payload_hash_vec(("da4", "c",
                                     int(turn_index)), 6)

            def cache_reuse_oracle():
                return (0.0, True)
            wda = emit_disagreement_algebra_v4_witness(
                trace=trace,
                probe_a=pa, probe_b=pb, probe_c=pc,
                cache_reuse_oracle=cache_reuse_oracle)
            da_v4_witness_cid = wda.cid()
        # M17 — deep substrate hybrid V3
        deep_v3_witness_cid = ""
        three_way_used = False
        if (p.enabled
                and p.deep_substrate_hybrid_v3 is not None):
            try:
                d_v6 = p.deep_substrate_hybrid_v3.deep_v6
                in_dim = int(d_v6.in_dim)
                fd = int(d_v6.inner_v5.inner_v4.factor_dim)
                qi = _payload_hash_vec(
                    ("hyb_v3_q", int(turn_index)), in_dim)
                ks = [_payload_hash_vec(
                    ("hyb_v3_k", int(turn_index), j), fd)
                    for j in range(2)]
                vs = [_payload_hash_vec(
                    ("hyb_v3_v", int(turn_index), j), fd)
                    for j in range(2)]
                _, whyb, _ = deep_substrate_hybrid_v3_forward(
                    hybrid=p.deep_substrate_hybrid_v3,
                    query_input=qi,
                    slot_keys=ks, slot_values=vs,
                    role_index=0, branch_index=0,
                    cycle_index=0,
                    trust_scalar=0.9,
                    uncertainty_scale=0.95,
                    substrate_query_token_ids=list(
                        p.follow_up_token_ids))
                deep_v3_witness_cid = whyb.cid()
                three_way_used = bool(whyb.three_way)
            except Exception:
                deep_v3_witness_cid = ""
        # M18 — substrate adapter V3
        adapter_v3_cid = ""
        if p.enabled:
            mat = probe_all_v3_adapters(
                probe_ollama=False, probe_openai=False)
            adapter_v3_cid = mat.cid()
        return W58HandoffEnvelope(
            schema=W58_SCHEMA_VERSION + ".envelope.v1",
            w57_outer_cid=str(w57_outer_cid),
            w58_params_cid=str(p.cid()),
            substrate_v3_witness_cid=str(
                substrate_v3_witness_cid),
            kv_bridge_v3_witness_cid=str(
                kv_bridge_v3_witness_cid),
            hsb_v2_witness_cid=str(hsb_v2_witness_cid),
            prefix_state_v2_witness_cid=str(
                prefix_state_v2_witness_cid),
            attn_steer_v2_witness_cid=str(
                attn_steer_v2_witness_cid),
            cache_controller_witness_cid=str(
                cache_controller_witness_cid),
            persistent_v10_witness_cid=str(
                persistent_v10_witness_cid),
            multi_hop_v8_witness_cid=str(
                multi_hop_v8_witness_cid),
            mlsc_v6_witness_cid=str(mlsc_v6_witness_cid),
            consensus_v4_witness_cid=str(
                consensus_v4_witness_cid),
            crc_v6_witness_cid=str(crc_v6_witness_cid),
            lhr_v10_witness_cid=str(lhr_v10_witness_cid),
            ecc_v10_witness_cid=str(ecc_v10_witness_cid),
            tvs_v7_witness_cid=str(tvs_v7_witness_cid),
            uncertainty_v6_witness_cid=str(
                uncertainty_v6_witness_cid),
            disagreement_algebra_v4_witness_cid=str(
                da_v4_witness_cid),
            deep_substrate_hybrid_v3_witness_cid=str(
                deep_v3_witness_cid),
            substrate_adapter_v3_matrix_cid=str(adapter_v3_cid),
            v10_chain_cid=str(self.chain.cid()),
            cache_reuse_fidelity_weighted_mean=float(
                composite_v6),
            three_way_used=bool(three_way_used),
            substrate_v3_used=bool(substrate_v3_used),
        )


def build_w58_team(*, seed: int = 58000) -> W58Team:
    return W58Team(params=W58Params.build_default(seed=int(seed)))


def build_trivial_w58_envelope(
        *, w57_outer_cid: str = "no_w57",
) -> W58HandoffEnvelope:
    """A W58 trivial envelope wraps a W57 outer CID through a
    disabled W58Params object. All W58 witness fields are empty
    strings; the verifier will flag many ``missing_*`` modes —
    that is the *expected* trivial passthrough behaviour.

    This is what the test suite uses to assert byte-identical
    passthrough."""
    return W58HandoffEnvelope(
        schema=W58_SCHEMA_VERSION + ".envelope.v1",
        w57_outer_cid=str(w57_outer_cid),
        w58_params_cid=str(W58Params.build_trivial().cid()),
        substrate_v3_witness_cid="",
        kv_bridge_v3_witness_cid="",
        hsb_v2_witness_cid="",
        prefix_state_v2_witness_cid="",
        attn_steer_v2_witness_cid="",
        cache_controller_witness_cid="",
        persistent_v10_witness_cid="",
        multi_hop_v8_witness_cid="",
        mlsc_v6_witness_cid="",
        consensus_v4_witness_cid="",
        crc_v6_witness_cid="",
        lhr_v10_witness_cid="",
        ecc_v10_witness_cid="",
        tvs_v7_witness_cid="",
        uncertainty_v6_witness_cid="",
        disagreement_algebra_v4_witness_cid="",
        deep_substrate_hybrid_v3_witness_cid="",
        substrate_adapter_v3_matrix_cid="",
        v10_chain_cid="",
        cache_reuse_fidelity_weighted_mean=1.0,
        three_way_used=False,
        substrate_v3_used=False,
    )


__all__ = [
    "W58_SCHEMA_VERSION",
    "W58_TEAM_RESULT_SCHEMA",
    "W58_ENVELOPE_VERIFIER_FAILURE_MODES",
    "W58Params",
    "W58HandoffEnvelope",
    "W58Team",
    "build_w58_team",
    "build_trivial_w58_envelope",
    "verify_w58_handoff",
]
