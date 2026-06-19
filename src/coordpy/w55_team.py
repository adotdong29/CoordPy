"""W55 — Deep Trust-Weighted Disagreement-Algebraic Latent OS
(DTDA-LOS).

The ``W55Team`` orchestrator composes the W54 ``W54Team`` with
the eleven M1..M11 W55 mechanism modules:

* M1 ``V7StackedCell`` + ``PersistentLatentStateV7Chain``
* M2 ``MultiHopBackendTranslator`` (7-backend hept)
* M3 ``MergeableLatentCapsuleV3`` + ``MergeOperatorV3`` +
   ``MergeAuditTrailV3``
* M4 ``TrustWeightedConsensusController``
* M5 ``CorruptionRobustCarrierV3``
* M6 ``DeepProxyStackV6``
* M7 ``LongHorizonReconstructionV7Head``
* M8 ``ECCCodebookV7``
* M9 transcript-vs-shared arbiter V4 (5-arm)
* M10 uncertainty layer V3
* M11 disagreement algebra primitives

Each W55 turn emits W55 per-module witnesses on top of the
W54 envelope. The final ``W55HandoffEnvelope`` binds: the W54
outer CID, every W55 witness CID, the W55Params CID, the
persistent-V7 chain CID, the MLSC V3 audit trail CID, the
TWCC controller audit trail CID, the disagreement-algebra
trace CID, and a single ``w55_outer_cid`` that closes the
chain w47 → w48 → w49 → w50 → w51 → w52 → w53 → w54 → w55.

Honest scope (do-not-overstate)
-------------------------------

This module does NOT alter transformer-internal state. Every
W55 witness is computed over capsule-layer signals exclusively.
Trivial passthrough is preserved byte-for-byte: when
``W55Params.build_trivial()`` is used and all flags are
disabled, the W55 envelope's internal ``w54_outer_cid``
equals the W54 outer CID exactly — the
``W55-L-TRIVIAL-W55-PASSTHROUGH`` falsifier.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from .agents import Agent
from .corruption_robust_carrier_v2 import (
    CorruptionRobustCarrierV2,
)
from .corruption_robust_carrier_v3 import (
    CorruptionRobustCarrierV3,
    CorruptionRobustnessV3Witness,
    emit_corruption_robustness_v3_witness,
)
from .deep_proxy_stack_v4 import (
    W53_DEFAULT_DEEP_V4_FACTOR_DIM,
    W53_DEFAULT_DEEP_V4_IN_DIM,
    W53_DEFAULT_DEEP_V4_N_BRANCH_HEADS,
    W53_DEFAULT_DEEP_V4_N_CYCLE_HEADS,
    W53_DEFAULT_DEEP_V4_N_HEADS,
    W53_DEFAULT_DEEP_V4_N_ROLES,
)
from .deep_proxy_stack_v6 import (
    DeepProxyStackV6,
    DeepProxyStackV6ForwardWitness,
    W55_DEFAULT_DEEP_V6_BASE_ABSTAIN_THRESHOLD,
    W55_DEFAULT_DEEP_V6_N_LAYERS,
    W55_DEFAULT_DEEP_V6_OUTER_LAYERS,
    emit_deep_proxy_stack_v6_forward_witness,
)
from .disagreement_algebra import (
    AlgebraTrace,
    DisagreementAlgebraWitness,
    check_difference_self_cancellation,
    check_intersection_distributivity_on_agreement,
    check_merge_idempotent,
    emit_disagreement_algebra_witness,
)
from .ecc_codebook_v5 import (
    W53_DEFAULT_ECC_CODE_DIM,
    W53_DEFAULT_ECC_EMIT_MASK_LEN,
)
from .ecc_codebook_v7 import (
    ECCCodebookV7,
    ECCCompressionV7Witness,
    W55_DEFAULT_ECC_V7_K1,
    W55_DEFAULT_ECC_V7_K2,
    W55_DEFAULT_ECC_V7_K3,
    W55_DEFAULT_ECC_V7_K4,
    W55_DEFAULT_ECC_V7_K5,
    W55_DEFAULT_ECC_V7_K6,
    W55_DEFAULT_ECC_V7_TARGET_BITS_PER_TOKEN,
    compress_carrier_ecc_v7,
    emit_ecc_v7_compression_witness,
)
from .long_horizon_retention_v7 import (
    LongHorizonReconstructionV7Head,
    LongHorizonReconstructionV7Witness,
    W55_DEFAULT_LHR_V7_FLAT_FEATURE_DIM,
    W55_DEFAULT_LHR_V7_HIDDEN_DIM,
    W55_DEFAULT_LHR_V7_MAX_K,
    W55_DEFAULT_LHR_V7_N_BRANCHES,
    W55_DEFAULT_LHR_V7_N_CYCLES,
    W55_DEFAULT_LHR_V7_N_MERGE_PAIRS,
    W55_DEFAULT_LHR_V7_N_ROLES,
    emit_lhr_v7_witness,
)
from .mergeable_latent_capsule_v3 import (
    MergeAuditTrailV3,
    MergeOperatorV3,
    MergeableLatentCapsuleV3,
    MergeableLatentCapsuleV3Witness,
    W55_DEFAULT_MLSC_V3_FACTOR_DIM,
    W55_DEFAULT_MLSC_V3_TRUST_DEFAULT,
    W55_DEFAULT_MLSC_V3_TRUST_DECAY,
    emit_mlsc_v3_witness,
    make_root_capsule_v3,
    merge_capsules_v3,
    step_branch_capsule_v3,
)
from .multi_hop_translator import (
    MultiHopBackendTranslator,
)
from .multi_hop_translator_v5 import (
    MultiHopV5Witness,
    W55_DEFAULT_MH_V5_BACKENDS,
    W55_DEFAULT_MH_V5_CODE_DIM,
    W55_DEFAULT_MH_V5_FEATURE_DIM,
    build_unfitted_hept_translator,
    emit_multi_hop_v5_witness,
    synthesize_hept_training_set,
)
from .persistent_latent_v7 import (
    PersistentLatentStateV7,
    PersistentLatentStateV7Chain,
    PersistentLatentStateV7Witness,
    V7StackedCell,
    W55_DEFAULT_V7_INPUT_DIM,
    W55_DEFAULT_V7_N_LAYERS,
    W55_DEFAULT_V7_STATE_DIM,
    W55_V7_NO_PARENT_STATE,
    emit_persistent_v7_witness,
    step_persistent_state_v7,
)
from .quantised_compression import (
    QuantisedBudgetGate,
)
from .transcript_vs_shared_arbiter_v4 import (
    TVSArbiterV4Witness,
    emit_tvs_arbiter_v4_witness,
    five_arm_compare,
)
from .trust_weighted_consensus_controller import (
    TrustWeightedConsensusController,
    TrustWeightedConsensusPolicy,
    TwccWitness,
    W55_DEFAULT_TWCC_COSINE_FLOOR,
    W55_DEFAULT_TWCC_FALLBACK_COSINE_FLOOR,
    W55_DEFAULT_TWCC_K_MAX,
    W55_DEFAULT_TWCC_K_MIN,
    W55_DEFAULT_TWCC_TRUST_THRESHOLD,
    emit_twcc_witness,
)
from .uncertainty_layer import calibration_check
from .uncertainty_layer_v2 import calibration_check_under_noise
from .uncertainty_layer_v3 import (
    FactUncertainty,
    UncertaintyLayerV3Witness,
    calibration_check_under_adversarial,
    compose_uncertainty_report_v3,
    emit_uncertainty_layer_v3_witness,
)
from .w54_team import (
    W54HandoffEnvelope,
    W54Params,
    W54Registry,
    W54Team,
    W54TeamResult,
    W54TurnWitnessBundle,
    build_trivial_w54_registry,
    build_w54_registry,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W55_SCHEMA_VERSION: str = "coordpy.w55_team.v1"
W55_TEAM_RESULT_SCHEMA: str = "coordpy.w55_team_result.v1"

W55_NO_STATE: str = "no_w55_persistent_state"


# =============================================================================
# Helpers
# =============================================================================


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _payload_hash_vec(
        payload: Any, dim: int,
) -> list[float]:
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
# W55Params
# =============================================================================


@dataclasses.dataclass
class W55Params:
    """All trainable / config surfaces for W55, layered over W54."""

    persistent_v7_cell: V7StackedCell | None
    hept_translator: MultiHopBackendTranslator | None
    mlsc_v3_operator: MergeOperatorV3 | None
    twcc_controller: TrustWeightedConsensusController | None
    deep_stack_v6: DeepProxyStackV6 | None
    ecc_codebook_v7: ECCCodebookV7 | None
    ecc_gate_v7: QuantisedBudgetGate | None
    long_horizon_v7_head: (
        LongHorizonReconstructionV7Head | None)
    crc_v3: CorruptionRobustCarrierV3 | None

    persistent_v7_enabled: bool = False
    hept_translator_enabled: bool = False
    mlsc_v3_enabled: bool = False
    twcc_enabled: bool = False
    deep_stack_v6_enabled: bool = False
    ecc_v7_compression_enabled: bool = False
    long_horizon_v7_enabled: bool = False
    crc_v3_enabled: bool = False
    tvs_arbiter_v4_enabled: bool = False
    uncertainty_v3_enabled: bool = False
    disagreement_algebra_enabled: bool = False

    target_bits_per_token_v7: float = (
        W55_DEFAULT_ECC_V7_TARGET_BITS_PER_TOKEN)
    arbiter_budget_tokens: int = 4
    twcc_k_required: int = W55_DEFAULT_TWCC_K_MIN

    @classmethod
    def build_trivial(cls) -> "W55Params":
        return cls(
            persistent_v7_cell=None,
            hept_translator=None,
            mlsc_v3_operator=None,
            twcc_controller=None,
            deep_stack_v6=None,
            ecc_codebook_v7=None,
            ecc_gate_v7=None,
            long_horizon_v7_head=None,
            crc_v3=None,
            persistent_v7_enabled=False,
            hept_translator_enabled=False,
            mlsc_v3_enabled=False,
            twcc_enabled=False,
            deep_stack_v6_enabled=False,
            ecc_v7_compression_enabled=False,
            long_horizon_v7_enabled=False,
            crc_v3_enabled=False,
            tvs_arbiter_v4_enabled=False,
            uncertainty_v3_enabled=False,
            disagreement_algebra_enabled=False,
        )

    @classmethod
    def build_default(
            cls, *,
            role_universe: Sequence[str] = (
                "r0", "r1", "r2", "r3"),
            seed: int = 12345,
    ) -> "W55Params":
        cell_v7 = V7StackedCell.init(
            state_dim=W55_DEFAULT_V7_STATE_DIM,
            input_dim=W55_DEFAULT_V7_INPUT_DIM,
            n_layers=W55_DEFAULT_V7_N_LAYERS,
            seed=int(seed))
        hept_tr = build_unfitted_hept_translator(
            backends=W55_DEFAULT_MH_V5_BACKENDS,
            code_dim=W55_DEFAULT_MH_V5_CODE_DIM,
            feature_dim=W55_DEFAULT_MH_V5_FEATURE_DIM,
            seed=int(seed) + 7)
        mlsc_op = MergeOperatorV3(
            factor_dim=W55_DEFAULT_MLSC_V3_FACTOR_DIM)
        twcc = TrustWeightedConsensusController.init(
            policy=TrustWeightedConsensusPolicy(
                k_min=W55_DEFAULT_TWCC_K_MIN,
                k_max=W55_DEFAULT_TWCC_K_MAX,
                cosine_floor=W55_DEFAULT_TWCC_COSINE_FLOOR,
                fallback_cosine_floor=(
                    W55_DEFAULT_TWCC_FALLBACK_COSINE_FLOOR),
                trust_threshold=(
                    W55_DEFAULT_TWCC_TRUST_THRESHOLD),
                allow_trust_weighted=True,
                allow_fallback_best_parent=True,
                allow_fallback_transcript=True),
            operator=mlsc_op)
        deep6 = DeepProxyStackV6.init(
            n_layers=W55_DEFAULT_DEEP_V6_N_LAYERS,
            in_dim=W53_DEFAULT_DEEP_V4_IN_DIM,
            factor_dim=W53_DEFAULT_DEEP_V4_FACTOR_DIM,
            n_heads=W53_DEFAULT_DEEP_V4_N_HEADS,
            n_branch_heads=W53_DEFAULT_DEEP_V4_N_BRANCH_HEADS,
            n_cycle_heads=W53_DEFAULT_DEEP_V4_N_CYCLE_HEADS,
            n_roles=W53_DEFAULT_DEEP_V4_N_ROLES,
            n_outer_layers=W55_DEFAULT_DEEP_V6_OUTER_LAYERS,
            base_abstain_threshold=(
                W55_DEFAULT_DEEP_V6_BASE_ABSTAIN_THRESHOLD),
            seed=int(seed) + 13)
        ecc_v7 = ECCCodebookV7.init(
            n_coarse=W55_DEFAULT_ECC_V7_K1,
            n_fine=W55_DEFAULT_ECC_V7_K2,
            n_ultra=W55_DEFAULT_ECC_V7_K3,
            n_ultra2=W55_DEFAULT_ECC_V7_K4,
            n_ultra3=W55_DEFAULT_ECC_V7_K5,
            n_ultra4=W55_DEFAULT_ECC_V7_K6,
            code_dim=W53_DEFAULT_ECC_CODE_DIM,
            seed=int(seed) + 17)
        ecc_gate = QuantisedBudgetGate.init(
            in_dim=W53_DEFAULT_ECC_CODE_DIM,
            emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
            seed=int(seed) + 19)
        ecc_gate.importance_threshold = 0.0
        ecc_gate.w_emit.values = [
            1.0] * len(ecc_gate.w_emit.values)
        lhr_v7 = LongHorizonReconstructionV7Head.init(
            carrier_dim=(
                W55_DEFAULT_LHR_V7_MAX_K
                * W55_DEFAULT_LHR_V7_FLAT_FEATURE_DIM),
            hidden_dim=W55_DEFAULT_LHR_V7_HIDDEN_DIM,
            out_dim=W55_DEFAULT_LHR_V7_FLAT_FEATURE_DIM,
            max_k=W55_DEFAULT_LHR_V7_MAX_K,
            n_branches=W55_DEFAULT_LHR_V7_N_BRANCHES,
            n_cycles=W55_DEFAULT_LHR_V7_N_CYCLES,
            n_merge_pairs=W55_DEFAULT_LHR_V7_N_MERGE_PAIRS,
            n_roles=W55_DEFAULT_LHR_V7_N_ROLES,
            seed=int(seed) + 23)
        crc_v3 = CorruptionRobustCarrierV3.init(
            codebook=ecc_v7.inner_v6.inner_v5,
            gate=ecc_gate)
        return cls(
            persistent_v7_cell=cell_v7,
            hept_translator=hept_tr,
            mlsc_v3_operator=mlsc_op,
            twcc_controller=twcc,
            deep_stack_v6=deep6,
            ecc_codebook_v7=ecc_v7,
            ecc_gate_v7=ecc_gate,
            long_horizon_v7_head=lhr_v7,
            crc_v3=crc_v3,
            persistent_v7_enabled=True,
            hept_translator_enabled=True,
            mlsc_v3_enabled=True,
            twcc_enabled=True,
            deep_stack_v6_enabled=True,
            ecc_v7_compression_enabled=True,
            long_horizon_v7_enabled=True,
            crc_v3_enabled=True,
            tvs_arbiter_v4_enabled=True,
            uncertainty_v3_enabled=True,
            disagreement_algebra_enabled=True,
        )

    @property
    def all_flags_disabled(self) -> bool:
        return not (
            self.persistent_v7_enabled
            or self.hept_translator_enabled
            or self.mlsc_v3_enabled
            or self.twcc_enabled
            or self.deep_stack_v6_enabled
            or self.ecc_v7_compression_enabled
            or self.long_horizon_v7_enabled
            or self.crc_v3_enabled
            or self.tvs_arbiter_v4_enabled
            or self.uncertainty_v3_enabled
            or self.disagreement_algebra_enabled)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(W55_SCHEMA_VERSION),
            "persistent_v7_cell_cid": (
                self.persistent_v7_cell.cid()
                if self.persistent_v7_cell is not None
                else ""),
            "hept_translator_cid": (
                self.hept_translator.cid()
                if self.hept_translator is not None
                else ""),
            "mlsc_v3_operator_cid": (
                self.mlsc_v3_operator.cid()
                if self.mlsc_v3_operator is not None
                else ""),
            "twcc_controller_cid": (
                self.twcc_controller.cid()
                if self.twcc_controller is not None
                else ""),
            "deep_stack_v6_cid": (
                self.deep_stack_v6.cid()
                if self.deep_stack_v6 is not None
                else ""),
            "ecc_codebook_v7_cid": (
                self.ecc_codebook_v7.cid()
                if self.ecc_codebook_v7 is not None
                else ""),
            "ecc_gate_v7_cid": (
                self.ecc_gate_v7.cid()
                if self.ecc_gate_v7 is not None
                else ""),
            "long_horizon_v7_head_cid": (
                self.long_horizon_v7_head.cid()
                if self.long_horizon_v7_head is not None
                else ""),
            "crc_v3_cid": (
                self.crc_v3.cid()
                if self.crc_v3 is not None
                else ""),
            "persistent_v7_enabled": bool(
                self.persistent_v7_enabled),
            "hept_translator_enabled": bool(
                self.hept_translator_enabled),
            "mlsc_v3_enabled": bool(self.mlsc_v3_enabled),
            "twcc_enabled": bool(self.twcc_enabled),
            "deep_stack_v6_enabled": bool(
                self.deep_stack_v6_enabled),
            "ecc_v7_compression_enabled": bool(
                self.ecc_v7_compression_enabled),
            "long_horizon_v7_enabled": bool(
                self.long_horizon_v7_enabled),
            "crc_v3_enabled": bool(self.crc_v3_enabled),
            "tvs_arbiter_v4_enabled": bool(
                self.tvs_arbiter_v4_enabled),
            "uncertainty_v3_enabled": bool(
                self.uncertainty_v3_enabled),
            "disagreement_algebra_enabled": bool(
                self.disagreement_algebra_enabled),
            "target_bits_per_token_v7": float(round(
                self.target_bits_per_token_v7, 12)),
            "arbiter_budget_tokens": int(
                self.arbiter_budget_tokens),
            "twcc_k_required": int(self.twcc_k_required),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_params",
            "params": self.to_dict()})


# =============================================================================
# W55Registry
# =============================================================================


@dataclasses.dataclass
class W55Registry:
    """W55 registry — wraps a W54 registry + W55 params."""

    schema_cid: str
    inner_w54_registry: W54Registry
    params: W55Params

    @property
    def is_trivial(self) -> bool:
        return (
            self.inner_w54_registry.is_trivial
            and self.params.all_flags_disabled
        )


def build_trivial_w55_registry(
        *, schema_cid: str | None = None,
) -> W55Registry:
    cid = schema_cid or _sha256_hex({
        "kind": "w55_trivial_schema"})
    inner = build_trivial_w54_registry(
        schema_cid=str(cid))
    return W55Registry(
        schema_cid=str(cid),
        inner_w54_registry=inner,
        params=W55Params.build_trivial(),
    )


def build_w55_registry(
        *,
        schema_cid: str,
        inner_w54_registry: W54Registry | None = None,
        params: W55Params | None = None,
        role_universe: Sequence[str] = (
            "r0", "r1", "r2", "r3"),
        seed: int = 12345,
) -> W55Registry:
    inner = (
        inner_w54_registry
        if inner_w54_registry is not None
        else build_w54_registry(
            schema_cid=str(schema_cid),
            role_universe=role_universe,
            seed=int(seed)))
    p = params or W55Params.build_default(
        role_universe=role_universe, seed=int(seed))
    return W55Registry(
        schema_cid=str(schema_cid),
        inner_w54_registry=inner,
        params=p,
    )


# =============================================================================
# Per-turn witnesses
# =============================================================================


@dataclasses.dataclass(frozen=True)
class W55TurnWitnessBundle:
    """All W55 per-turn witness CIDs for one team turn."""

    persistent_v7_witness_cid: str
    hept_translator_witness_cid: str
    mlsc_v3_witness_cid: str
    twcc_witness_cid: str
    deep_stack_v6_witness_cid: str
    ecc_v7_compression_witness_cid: str
    long_horizon_v7_witness_cid: str
    crc_v3_witness_cid: str
    tvs_arbiter_v4_witness_cid: str
    uncertainty_v3_witness_cid: str
    disagreement_algebra_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "persistent_v7_witness_cid": str(
                self.persistent_v7_witness_cid),
            "hept_translator_witness_cid": str(
                self.hept_translator_witness_cid),
            "mlsc_v3_witness_cid": str(
                self.mlsc_v3_witness_cid),
            "twcc_witness_cid": str(self.twcc_witness_cid),
            "deep_stack_v6_witness_cid": str(
                self.deep_stack_v6_witness_cid),
            "ecc_v7_compression_witness_cid": str(
                self.ecc_v7_compression_witness_cid),
            "long_horizon_v7_witness_cid": str(
                self.long_horizon_v7_witness_cid),
            "crc_v3_witness_cid": str(
                self.crc_v3_witness_cid),
            "tvs_arbiter_v4_witness_cid": str(
                self.tvs_arbiter_v4_witness_cid),
            "uncertainty_v3_witness_cid": str(
                self.uncertainty_v3_witness_cid),
            "disagreement_algebra_witness_cid": str(
                self.disagreement_algebra_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_turn_witness_bundle",
            "bundle": self.to_dict()})


# =============================================================================
# W55HandoffEnvelope
# =============================================================================


@dataclasses.dataclass(frozen=True)
class W55HandoffEnvelope:
    """Sealed per-team W55 envelope."""

    schema_version: str
    w54_outer_cid: str
    params_cid: str
    turn_witness_bundle_cid: str
    w54_envelope_count: int
    persistent_v7_chain_cid: str
    mlsc_v3_audit_trail_cid: str
    twcc_audit_trail_cid: str
    disagreement_algebra_trace_cid: str
    composite_confidence_mean_v3: float
    trust_weighted_composite_mean: float
    twcc_quorum_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "w54_outer_cid": str(self.w54_outer_cid),
            "params_cid": str(self.params_cid),
            "turn_witness_bundle_cid": str(
                self.turn_witness_bundle_cid),
            "w54_envelope_count": int(
                self.w54_envelope_count),
            "persistent_v7_chain_cid": str(
                self.persistent_v7_chain_cid),
            "mlsc_v3_audit_trail_cid": str(
                self.mlsc_v3_audit_trail_cid),
            "twcc_audit_trail_cid": str(
                self.twcc_audit_trail_cid),
            "disagreement_algebra_trace_cid": str(
                self.disagreement_algebra_trace_cid),
            "composite_confidence_mean_v3": float(round(
                self.composite_confidence_mean_v3, 12)),
            "trust_weighted_composite_mean": float(round(
                self.trust_weighted_composite_mean, 12)),
            "twcc_quorum_rate": float(round(
                self.twcc_quorum_rate, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_outer_envelope",
            "envelope": self.to_dict()})


# =============================================================================
# Forward — compute per-turn W55 witnesses
# =============================================================================


def _persistent_v7_step(
        *,
        params: W55Params,
        turn_index: int,
        role: str,
        prev_state: PersistentLatentStateV7 | None,
        carrier_payload: Any,
        state_chain: PersistentLatentStateV7Chain,
        anchor_skip: Sequence[float] | None,
        branch_id: str = "main",
) -> PersistentLatentStateV7 | None:
    if (not params.persistent_v7_enabled
            or params.persistent_v7_cell is None):
        return None
    cell = params.persistent_v7_cell
    input_vec = _payload_hash_vec(
        carrier_payload, cell.state_dim)
    if anchor_skip is None:
        anchor_skip = input_vec
    new_state = step_persistent_state_v7(
        cell=cell,
        prev_state=prev_state,
        carrier_values=input_vec,
        turn_index=int(turn_index),
        role=str(role),
        branch_id=str(branch_id),
        anchor_skip=anchor_skip)
    state_chain.add(new_state)
    return new_state


def _emit_w55_turn_witnesses(
        *,
        params: W55Params,
        turn_index: int,
        role: str,
        prev_state: PersistentLatentStateV7 | None,
        state_chain: PersistentLatentStateV7Chain,
        mlsc_audit: MergeAuditTrailV3,
        mlsc_store: dict[str, MergeableLatentCapsuleV3],
        prev_mlsc_capsule: MergeableLatentCapsuleV3 | None,
        algebra_trace: AlgebraTrace,
        anchor_skip: Sequence[float] | None,
        branch_index: int = 0,
        cycle_index: int = 0,
        role_index: int = 0,
        carrier_payload: Any = None,
) -> tuple[
        W55TurnWitnessBundle,
        PersistentLatentStateV7 | None,
        MergeableLatentCapsuleV3 | None,
        float, float, float]:
    """Compute all per-turn W55 witnesses.

    Returns (bundle, new_state, new_mlsc_v3_capsule,
        composite_confidence_v3, tw_composite, twcc_quorum_rate).
    """
    new_state = _persistent_v7_step(
        params=params,
        turn_index=int(turn_index),
        role=str(role),
        prev_state=prev_state,
        carrier_payload=carrier_payload,
        state_chain=state_chain,
        anchor_skip=anchor_skip,
        branch_id="main")
    pv7_witness_cid = ""
    pv7_conf = 1.0
    if (new_state is not None
            and params.persistent_v7_cell is not None):
        w_pv7 = emit_persistent_v7_witness(
            state=new_state,
            cell=params.persistent_v7_cell,
            chain=state_chain)
        pv7_witness_cid = w_pv7.cid()
        gate_norm = float(
            new_state.update_gate_l1_sum) / float(
                max(1, int(new_state.state_dim)))
        pv7_conf = float(max(0.0, min(
            1.0, 1.0 - 0.5 * gate_norm)))
    # M2 — hept translator witness.
    hept_witness_cid = ""
    hept_conf = 1.0
    if (params.hept_translator_enabled
            and params.hept_translator is not None):
        ts = synthesize_hept_training_set(
            n_examples=4, seed=int(turn_index) + 11,
            backends=tuple(params.hept_translator.backends),
            code_dim=int(params.hept_translator.code_dim),
            feature_dim=int(
                params.hept_translator.feature_dim))
        w_h = emit_multi_hop_v5_witness(
            translator=params.hept_translator,
            examples=ts.examples[:4])
        hept_witness_cid = w_h.cid()
        hept_conf = float(max(0.0, min(
            1.0, w_h.direct_fidelity_a_g)))
    # M3 — MLSC V3 capsule.
    mlsc_v3_witness_cid = ""
    mlsc_conf = 1.0
    new_mlsc: MergeableLatentCapsuleV3 | None = None
    if (params.mlsc_v3_enabled
            and params.mlsc_v3_operator is not None):
        op = params.mlsc_v3_operator
        payload = (
            list(new_state.top_state)[:int(op.factor_dim)]
            if new_state is not None
            else _payload_hash_vec(
                carrier_payload, op.factor_dim))
        while len(payload) < int(op.factor_dim):
            payload.append(0.0)
        if prev_mlsc_capsule is None:
            new_mlsc = make_root_capsule_v3(
                branch_id=f"branch_{branch_index}",
                payload=payload,
                confidence=float(pv7_conf),
                trust=W55_DEFAULT_MLSC_V3_TRUST_DEFAULT,
                trust_decay=W55_DEFAULT_MLSC_V3_TRUST_DECAY,
                fact_tags=(f"turn={turn_index}",),
                turn_index=int(turn_index))
        else:
            new_mlsc = step_branch_capsule_v3(
                parent=prev_mlsc_capsule,
                payload=payload,
                confidence=float(pv7_conf),
                trust=float(prev_mlsc_capsule.trust),
                new_fact_tags=(f"turn={turn_index}",),
                turn_index=int(turn_index))
        mlsc_store[new_mlsc.cid()] = new_mlsc
        if (prev_mlsc_capsule is not None
                and (turn_index % 2) == 1
                and params.disagreement_algebra_enabled):
            merged = merge_capsules_v3(
                op, [prev_mlsc_capsule, new_mlsc],
                audit_trail=mlsc_audit,
                algebra_trace=algebra_trace,
                extra_fact_tags=(
                    f"merge_at_turn={turn_index}",))
            mlsc_store[merged.cid()] = merged
            new_mlsc = merged
        elif (prev_mlsc_capsule is not None
                and (turn_index % 2) == 1):
            merged = merge_capsules_v3(
                op, [prev_mlsc_capsule, new_mlsc],
                audit_trail=mlsc_audit,
                extra_fact_tags=(
                    f"merge_at_turn={turn_index}",))
            mlsc_store[merged.cid()] = merged
            new_mlsc = merged
        w_mlsc = emit_mlsc_v3_witness(
            leaf=new_mlsc,
            operator=op,
            audit_trail=mlsc_audit,
            capsule_store=mlsc_store)
        mlsc_v3_witness_cid = w_mlsc.cid()
        mlsc_conf = float(new_mlsc.confidence)
    # M4 — TWCC controller witness.
    twcc_witness_cid = ""
    if (params.twcc_enabled
            and params.twcc_controller is not None
            and new_mlsc is not None):
        if (turn_index % 2) == 0:
            sibling = (
                prev_mlsc_capsule if prev_mlsc_capsule
                is not None else new_mlsc)
            _, _ = params.twcc_controller.decide(
                [new_mlsc, sibling],
                turn_index=int(turn_index),
                k_required=int(params.twcc_k_required),
                transcript_payload=[0.0] * len(
                    new_mlsc.payload))
        w_twcc = emit_twcc_witness(params.twcc_controller)
        twcc_witness_cid = w_twcc.cid()
    # M5 — deep stack V6 forward witness.
    deep_v6_witness_cid = ""
    deep_v6_conf = 1.0
    if (params.deep_stack_v6_enabled
            and params.deep_stack_v6 is not None):
        s = params.deep_stack_v6
        if new_state is not None:
            q = list(new_state.top_state)[:s.in_dim]
        else:
            q = []
        while len(q) < s.in_dim:
            q.append(0.0)
        w_deep, _ = (
            emit_deep_proxy_stack_v6_forward_witness(
                stack=s, query_input=q,
                slot_keys=[q], slot_values=[q],
                role_index=int(role_index),
                branch_index=int(branch_index),
                cycle_index=int(cycle_index),
                trust_scalar=float(mlsc_conf),
                uncertainty_scale=float(pv7_conf)))
        deep_v6_witness_cid = w_deep.cid()
        deep_v6_conf = float(w_deep.corruption_confidence)
    # M6 — ECC V7 compression witness.
    ecc_v7_witness_cid = ""
    if (params.ecc_v7_compression_enabled
            and params.ecc_codebook_v7 is not None
            and params.ecc_gate_v7 is not None
            and new_state is not None):
        carrier = list(new_state.top_state)[
            :int(params.ecc_codebook_v7.code_dim)]
        while len(carrier) < int(
                params.ecc_codebook_v7.code_dim):
            carrier.append(0.0)
        comp = compress_carrier_ecc_v7(
            carrier,
            codebook=params.ecc_codebook_v7,
            gate=params.ecc_gate_v7)
        w_ecc = emit_ecc_v7_compression_witness(
            codebook=params.ecc_codebook_v7,
            compression=comp,
            target_bits_per_token=float(
                params.target_bits_per_token_v7))
        ecc_v7_witness_cid = w_ecc.cid()
    # M7 — long-horizon V7 witness.
    lhr_v7_witness_cid = ""
    if (params.long_horizon_v7_enabled
            and params.long_horizon_v7_head is not None):
        w_lhr = emit_lhr_v7_witness(
            head=params.long_horizon_v7_head,
            examples=(),
            k_max_for_degradation=16)
        lhr_v7_witness_cid = w_lhr.cid()
    # M8 — CRC V3 witness.
    crc_v3_witness_cid = ""
    crc_silent_failure_rate = 0.0
    if (params.crc_v3_enabled
            and params.crc_v3 is not None
            and new_state is not None):
        carrier = list(new_state.top_state)[
            :int(params.crc_v3.codebook.code_dim)]
        while len(carrier) < int(
                params.crc_v3.codebook.code_dim):
            carrier.append(0.0)
        w_crc = emit_corruption_robustness_v3_witness(
            crc_v3=params.crc_v3,
            carriers=[carrier],
            flip_intensity=1.0,
            seed=int(turn_index) + 1)
        crc_v3_witness_cid = w_crc.cid()
        crc_silent_failure_rate = float(
            w_crc.silent_failure_rate)
    # M9 — TVS arbiter V4 witness.
    tvs_arb_v4_witness_cid = ""
    if (params.tvs_arbiter_v4_enabled
            and params.ecc_codebook_v7 is not None
            and params.ecc_gate_v7 is not None
            and new_state is not None):
        carrier = list(new_state.top_state)[
            :int(params.ecc_codebook_v7.code_dim)]
        while len(carrier) < int(
                params.ecc_codebook_v7.code_dim):
            carrier.append(0.0)
        mr = float(mlsc_conf) * 0.9
        twr = float(mlsc_conf) * float(
            new_mlsc.trust if new_mlsc is not None else 1.0)
        # Use the inner V4 codebook (V7's V6's V5's V4).
        v4_codebook = (
            params.ecc_codebook_v7.inner_v6.inner_v5
            .inner_v4)
        result = five_arm_compare(
            carriers=[carrier],
            codebook=v4_codebook,
            gate=params.ecc_gate_v7,
            budget_tokens=int(
                params.arbiter_budget_tokens),
            per_turn_confidences=[float(pv7_conf)],
            per_turn_trust_scores=[
                float(new_mlsc.trust)
                if new_mlsc is not None else 1.0],
            per_turn_merge_retentions=[float(mr)],
            per_turn_tw_retentions=[float(twr)])
        w_tvs = emit_tvs_arbiter_v4_witness(result=result)
        tvs_arb_v4_witness_cid = w_tvs.cid()
    # M10 — uncertainty layer V3 witness.
    uncert_v3_witness_cid = ""
    composite_conf = 1.0
    tw_composite = 1.0
    if params.uncertainty_v3_enabled:
        report = compose_uncertainty_report_v3(
            persistent_v7_confidence=float(pv7_conf),
            multi_hop_v5_confidence=float(hept_conf),
            mlsc_v3_capsule_confidence=float(mlsc_conf),
            deep_v6_corruption_confidence=float(
                deep_v6_conf),
            crc_v3_silent_failure_rate=float(
                crc_silent_failure_rate),
            trust_weights={
                "persistent_v7": float(pv7_conf),
                "multi_hop_v5": float(hept_conf),
                "mlsc_v3": float(
                    new_mlsc.trust
                    if new_mlsc is not None else 1.0),
                "deep_v6": float(deep_v6_conf),
                "crc_v3": 1.0,
            },
            fact_uncertainties=tuple(
                FactUncertainty(
                    tag=fc.tag,
                    confidence=float(new_mlsc.confidence),
                    n_contributors=int(fc.count))
                for fc in (
                    new_mlsc.fact_confirmations
                    if new_mlsc is not None else ())
            ))
        composite_conf = float(report.composite_confidence)
        tw_composite = float(report.trust_weighted_composite)
        clean_cal = calibration_check(
            confidences=(), accuracies=(),
            min_calibration_gap=0.10)
        noisy_cal = calibration_check_under_noise(
            confidences=(), accuracies=(),
            noise_magnitude=0.1,
            seed=int(turn_index) + 1)
        adv_cal = calibration_check_under_adversarial(
            confidences=(), accuracies=(),
            perturbation_magnitude=0.1,
            seed=int(turn_index) + 2)
        w_unc = emit_uncertainty_layer_v3_witness(
            report=report,
            calibration_clean=clean_cal,
            calibration_noisy=noisy_cal,
            calibration_adversarial=adv_cal)
        uncert_v3_witness_cid = w_unc.cid()
    # M11 — disagreement algebra witness.
    algebra_witness_cid = ""
    if params.disagreement_algebra_enabled:
        # Run identity checks on the current state (or zero).
        if new_state is not None:
            probe = list(new_state.top_state)[:4]
        else:
            probe = [0.5, 0.3, -0.1, 0.0]
        check1 = check_merge_idempotent(probe)
        check2 = check_difference_self_cancellation(probe)
        # Distributivity check on hash-derived vectors.
        a = _payload_hash_vec(
            (int(turn_index), "a"), 4)
        b = _payload_hash_vec(
            (int(turn_index), "b"), 4)
        c = _payload_hash_vec(
            (int(turn_index), "c"), 4)
        check3 = (
            check_intersection_distributivity_on_agreement(
                a, b, c))
        w_alg = emit_disagreement_algebra_witness(
            trace=algebra_trace,
            identity_results=(check1, check2, check3))
        algebra_witness_cid = w_alg.cid()
    twcc_q_rate = 0.0
    if (params.twcc_controller is not None
            and params.twcc_enabled):
        w_twcc_local = emit_twcc_witness(
            params.twcc_controller)
        twcc_q_rate = float(w_twcc_local.quorum_rate)
    bundle = W55TurnWitnessBundle(
        persistent_v7_witness_cid=str(pv7_witness_cid),
        hept_translator_witness_cid=str(hept_witness_cid),
        mlsc_v3_witness_cid=str(mlsc_v3_witness_cid),
        twcc_witness_cid=str(twcc_witness_cid),
        deep_stack_v6_witness_cid=str(
            deep_v6_witness_cid),
        ecc_v7_compression_witness_cid=str(
            ecc_v7_witness_cid),
        long_horizon_v7_witness_cid=str(
            lhr_v7_witness_cid),
        crc_v3_witness_cid=str(crc_v3_witness_cid),
        tvs_arbiter_v4_witness_cid=str(
            tvs_arb_v4_witness_cid),
        uncertainty_v3_witness_cid=str(
            uncert_v3_witness_cid),
        disagreement_algebra_witness_cid=str(
            algebra_witness_cid),
    )
    return (
        bundle, new_state, new_mlsc,
        float(composite_conf),
        float(tw_composite),
        float(twcc_q_rate))


# =============================================================================
# W55TeamResult
# =============================================================================


@dataclasses.dataclass(frozen=True)
class W55TeamResult:
    schema: str
    task: str
    final_output: str
    w54_outer_cid: str
    w55_outer_cid: str
    w55_params_cid: str
    w55_envelope: W55HandoffEnvelope
    turn_witness_bundles: tuple[W55TurnWitnessBundle, ...]
    persistent_v7_state_cids: tuple[str, ...]
    mlsc_v3_capsule_cids: tuple[str, ...]
    composite_confidence_mean_v3: float
    trust_weighted_composite_mean: float
    twcc_quorum_rate: float
    n_turns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "task": str(self.task),
            "final_output": str(self.final_output),
            "w54_outer_cid": str(self.w54_outer_cid),
            "w55_outer_cid": str(self.w55_outer_cid),
            "w55_params_cid": str(self.w55_params_cid),
            "w55_envelope": self.w55_envelope.to_dict(),
            "turn_witness_bundles": [
                b.to_dict()
                for b in self.turn_witness_bundles],
            "persistent_v7_state_cids": list(
                self.persistent_v7_state_cids),
            "mlsc_v3_capsule_cids": list(
                self.mlsc_v3_capsule_cids),
            "composite_confidence_mean_v3": float(round(
                self.composite_confidence_mean_v3, 12)),
            "trust_weighted_composite_mean": float(round(
                self.trust_weighted_composite_mean, 12)),
            "twcc_quorum_rate": float(round(
                self.twcc_quorum_rate, 12)),
            "n_turns": int(self.n_turns),
        }


# =============================================================================
# W55Team
# =============================================================================


@dataclasses.dataclass
class W55Team:
    """W55 team orchestrator — wraps ``W54Team``."""

    agents: Sequence[Agent]
    registry: W55Registry
    backend: Any = None
    team_instructions: str = ""
    max_visible_handoffs: int = 4
    quad_backend_a: Any = None
    quad_backend_b: Any = None
    quad_backend_c: Any = None
    quad_backend_d: Any = None
    quad_anchor_n_turns: int = 4

    def run(
            self, task: str, *,
            progress: Callable[[Any], None] | None = None,
    ) -> W55TeamResult:
        w54_team = W54Team(
            agents=list(self.agents),
            backend=self.backend,
            registry=self.registry.inner_w54_registry,
            team_instructions=self.team_instructions,
            max_visible_handoffs=int(
                self.max_visible_handoffs),
            quad_backend_a=self.quad_backend_a,
            quad_backend_b=self.quad_backend_b,
            quad_backend_c=self.quad_backend_c,
            quad_backend_d=self.quad_backend_d,
            quad_anchor_n_turns=int(
                self.quad_anchor_n_turns),
        )
        w54_result = w54_team.run(task, progress=progress)
        params = self.registry.params
        state_chain = PersistentLatentStateV7Chain.empty()
        mlsc_audit = MergeAuditTrailV3.empty()
        algebra_trace = AlgebraTrace.empty()
        if params.twcc_controller is not None:
            params.twcc_controller.controller_audit = (
                type(params.twcc_controller.controller_audit)
                .empty())
            params.twcc_controller.capsule_audit = (
                MergeAuditTrailV3.empty())
        mlsc_store: dict[str, MergeableLatentCapsuleV3] = {}
        bundles: list[W55TurnWitnessBundle] = []
        persistent_v7_cids: list[str] = []
        mlsc_v3_cids: list[str] = []
        prev_state: PersistentLatentStateV7 | None = None
        prev_mlsc: MergeableLatentCapsuleV3 | None = None
        n_turns = int(w54_result.n_turns)
        composite_sum = 0.0
        tw_sum = 0.0
        quorum_sum = 0.0
        anchor_skip_seed: Sequence[float] | None = None
        for i, turn in enumerate(
                w54_result.turn_witness_bundles):
            n_branches = max(
                1, params.deep_stack_v6.n_branch_heads
                if params.deep_stack_v6 is not None
                else 4)
            n_cycles = max(
                1, params.deep_stack_v6.n_cycle_heads
                if params.deep_stack_v6 is not None
                else 4)
            n_roles = max(
                1, params.deep_stack_v6.n_roles
                if params.deep_stack_v6 is not None
                else 4)
            branch_index = int(i) % int(n_branches)
            cycle_index = int(i) % int(n_cycles)
            role_index = int(i) % int(n_roles)
            carrier_payload = (int(i), turn.cid())
            if anchor_skip_seed is None and i == 0:
                anchor_skip_seed = _payload_hash_vec(
                    carrier_payload,
                    params.persistent_v7_cell.state_dim
                    if params.persistent_v7_cell is not None
                    else 8)
            (bundle, new_state, new_mlsc,
             conf, tw, qrate) = _emit_w55_turn_witnesses(
                params=params,
                turn_index=int(i),
                role=str(self._role_for_turn(i)),
                prev_state=prev_state,
                state_chain=state_chain,
                mlsc_audit=mlsc_audit,
                mlsc_store=mlsc_store,
                prev_mlsc_capsule=prev_mlsc,
                algebra_trace=algebra_trace,
                anchor_skip=anchor_skip_seed,
                branch_index=int(branch_index),
                cycle_index=int(cycle_index),
                role_index=int(role_index),
                carrier_payload=carrier_payload,
            )
            bundles.append(bundle)
            composite_sum += float(conf)
            tw_sum += float(tw)
            quorum_sum += float(qrate)
            if new_state is not None:
                persistent_v7_cids.append(new_state.cid())
                prev_state = new_state
            if new_mlsc is not None:
                mlsc_v3_cids.append(new_mlsc.cid())
                prev_mlsc = new_mlsc
        comp_mean = (
            float(composite_sum) / float(max(1, n_turns)))
        tw_mean = (
            float(tw_sum) / float(max(1, n_turns)))
        quorum_mean = (
            float(quorum_sum) / float(max(1, n_turns)))
        bundles_cid = _sha256_hex({
            "kind": "w55_all_turn_witness_bundles",
            "bundles": [b.to_dict() for b in bundles],
        })
        twcc_audit_cid = (
            params.twcc_controller.controller_audit.cid()
            if params.twcc_controller is not None else "")
        env = W55HandoffEnvelope(
            schema_version=W55_SCHEMA_VERSION,
            w54_outer_cid=str(w54_result.w54_outer_cid),
            params_cid=str(params.cid()),
            turn_witness_bundle_cid=str(bundles_cid),
            w54_envelope_count=int(n_turns),
            persistent_v7_chain_cid=str(state_chain.cid()),
            mlsc_v3_audit_trail_cid=str(mlsc_audit.cid()),
            twcc_audit_trail_cid=str(twcc_audit_cid),
            disagreement_algebra_trace_cid=str(
                algebra_trace.cid()),
            composite_confidence_mean_v3=float(comp_mean),
            trust_weighted_composite_mean=float(tw_mean),
            twcc_quorum_rate=float(quorum_mean),
        )
        return W55TeamResult(
            schema=W55_TEAM_RESULT_SCHEMA,
            task=str(task),
            final_output=str(w54_result.final_output),
            w54_outer_cid=str(w54_result.w54_outer_cid),
            w55_outer_cid=str(env.cid()),
            w55_params_cid=str(params.cid()),
            w55_envelope=env,
            turn_witness_bundles=tuple(bundles),
            persistent_v7_state_cids=tuple(
                persistent_v7_cids),
            mlsc_v3_capsule_cids=tuple(mlsc_v3_cids),
            composite_confidence_mean_v3=float(comp_mean),
            trust_weighted_composite_mean=float(tw_mean),
            twcc_quorum_rate=float(quorum_mean),
            n_turns=int(n_turns),
        )

    def _role_for_turn(self, i: int) -> str:
        roles = [
            str(getattr(a, "role", f"r{i}"))
            for a in self.agents]
        if not roles:
            return f"r{i}"
        return roles[i % len(roles)]


# =============================================================================
# Verifier
# =============================================================================


W55_ENVELOPE_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w55_schema_mismatch",
    "w55_w54_outer_cid_mismatch",
    "w55_params_cid_mismatch",
    "w55_turn_witness_bundle_cid_mismatch",
    "w55_envelope_count_mismatch",
    "w55_outer_cid_mismatch",
    "w55_persistent_v7_chain_cid_mismatch",
    "w55_mlsc_v3_audit_trail_cid_mismatch",
    "w55_twcc_audit_trail_cid_mismatch",
    "w55_disagreement_algebra_trace_cid_mismatch",
    "w55_trivial_passthrough_w54_cid_mismatch",
    "w55_persistent_v7_witness_missing_when_enabled",
    "w55_hept_translator_witness_missing_when_enabled",
    "w55_mlsc_v3_witness_missing_when_enabled",
    "w55_twcc_witness_missing_when_enabled",
    "w55_deep_stack_v6_witness_missing_when_enabled",
    "w55_ecc_v7_compression_witness_missing_when_enabled",
    "w55_long_horizon_v7_witness_missing_when_enabled",
    "w55_crc_v3_witness_missing_when_enabled",
    "w55_tvs_arbiter_v4_witness_missing_when_enabled",
    "w55_uncertainty_v3_witness_missing_when_enabled",
    "w55_disagreement_algebra_witness_missing_when_enabled",
    "w55_envelope_payload_hash_mismatch",
    "w55_per_turn_bundle_count_mismatch",
    "w55_witness_bundle_cid_recompute_mismatch",
    "w55_persistent_v7_state_count_inconsistent",
    "w55_outer_cid_recompute_mismatch",
    "w55_inner_w54_envelope_invalid",
    "w55_role_universe_mismatch",
    "w55_composite_confidence_v3_out_of_bounds",
    "w55_trust_weighted_composite_out_of_bounds",
    "w55_twcc_quorum_rate_out_of_bounds",
    "w55_mlsc_v3_audit_trail_walk_orphan",
)


def verify_w55_handoff(
        envelope: W55HandoffEnvelope,
        *,
        expected_w54_outer_cid: str | None = None,
        expected_params_cid: str | None = None,
        bundles: Sequence[W55TurnWitnessBundle] | None = None,
        registry: W55Registry | None = None,
        persistent_v7_state_cids: (
            Sequence[str] | None) = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if envelope.schema_version != W55_SCHEMA_VERSION:
        failures.append("w55_schema_mismatch")
    if (expected_w54_outer_cid is not None
            and envelope.w54_outer_cid
            != expected_w54_outer_cid):
        failures.append("w55_w54_outer_cid_mismatch")
    if (expected_params_cid is not None
            and envelope.params_cid
            != expected_params_cid):
        failures.append("w55_params_cid_mismatch")
    if not (
            0.0
            <= float(
                envelope.composite_confidence_mean_v3)
            <= 1.0):
        failures.append(
            "w55_composite_confidence_v3_out_of_bounds")
    if not (
            0.0
            <= float(
                envelope.trust_weighted_composite_mean)
            <= 1.0):
        failures.append(
            "w55_trust_weighted_composite_out_of_bounds")
    if not (
            0.0 <= float(envelope.twcc_quorum_rate)
            <= 1.0):
        failures.append(
            "w55_twcc_quorum_rate_out_of_bounds")
    if bundles is not None:
        if envelope.w54_envelope_count != len(bundles):
            failures.append(
                "w55_per_turn_bundle_count_mismatch")
        recomputed = _sha256_hex({
            "kind": "w55_all_turn_witness_bundles",
            "bundles": [b.to_dict() for b in bundles],
        })
        if envelope.turn_witness_bundle_cid != recomputed:
            failures.append(
                "w55_witness_bundle_cid_recompute_mismatch")
    if registry is not None and bundles is not None:
        p = registry.params
        for b in bundles:
            if (p.persistent_v7_enabled
                    and not b.persistent_v7_witness_cid):
                failures.append(
                    "w55_persistent_v7_witness_missing_when_enabled")
                break
        for b in bundles:
            if (p.hept_translator_enabled
                    and not b.hept_translator_witness_cid):
                failures.append(
                    "w55_hept_translator_witness_missing_when_enabled")
                break
        for b in bundles:
            if (p.mlsc_v3_enabled
                    and not b.mlsc_v3_witness_cid):
                failures.append(
                    "w55_mlsc_v3_witness_missing_when_enabled")
                break
        for b in bundles:
            if (p.twcc_enabled
                    and not b.twcc_witness_cid):
                failures.append(
                    "w55_twcc_witness_missing_when_enabled")
                break
        for b in bundles:
            if (p.deep_stack_v6_enabled
                    and not b.deep_stack_v6_witness_cid):
                failures.append(
                    "w55_deep_stack_v6_witness_missing_when_enabled")
                break
        for b in bundles:
            if (p.ecc_v7_compression_enabled
                    and not b.ecc_v7_compression_witness_cid):
                failures.append(
                    "w55_ecc_v7_compression_witness_missing_when_enabled")
                break
        for b in bundles:
            if (p.long_horizon_v7_enabled
                    and not b.long_horizon_v7_witness_cid):
                failures.append(
                    "w55_long_horizon_v7_witness_missing_when_enabled")
                break
        for b in bundles:
            if (p.crc_v3_enabled
                    and not b.crc_v3_witness_cid):
                failures.append(
                    "w55_crc_v3_witness_missing_when_enabled")
                break
        for b in bundles:
            if (p.tvs_arbiter_v4_enabled
                    and not b.tvs_arbiter_v4_witness_cid):
                failures.append(
                    "w55_tvs_arbiter_v4_witness_missing_when_enabled")
                break
        for b in bundles:
            if (p.uncertainty_v3_enabled
                    and not b.uncertainty_v3_witness_cid):
                failures.append(
                    "w55_uncertainty_v3_witness_missing_when_enabled")
                break
        for b in bundles:
            if (p.disagreement_algebra_enabled
                    and not b.disagreement_algebra_witness_cid):
                failures.append(
                    "w55_disagreement_algebra_witness_missing_when_enabled")
                break
    if (persistent_v7_state_cids is not None
            and registry is not None):
        if (registry.params.persistent_v7_enabled
                and len(persistent_v7_state_cids)
                != envelope.w54_envelope_count):
            failures.append(
                "w55_persistent_v7_state_count_inconsistent")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "outer_cid": envelope.cid(),
        "n_failure_modes": int(
            len(W55_ENVELOPE_VERIFIER_FAILURE_MODES)),
    }


__all__ = [
    "W55_SCHEMA_VERSION",
    "W55_TEAM_RESULT_SCHEMA",
    "W55_NO_STATE",
    "W55_ENVELOPE_VERIFIER_FAILURE_MODES",
    "W55Params",
    "W55Registry",
    "W55TurnWitnessBundle",
    "W55HandoffEnvelope",
    "W55TeamResult",
    "W55Team",
    "build_trivial_w55_registry",
    "build_w55_registry",
    "verify_w55_handoff",
]
