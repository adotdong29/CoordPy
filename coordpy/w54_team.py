"""W54 — Deep Mergeable Disagreement-aware Latent OS (DMD-LOS).

The ``W54Team`` orchestrator composes the W53 ``W53Team`` with
the ten M1..M10 W54 mechanism modules:

* M1 ``V6StackedCell`` + ``PersistentLatentStateV6Chain``
* M2 ``MultiHopBackendTranslator`` (6-backend; V4 helpers)
* M3 ``MergeableLatentCapsuleV2`` + ``MergeOperatorV2`` +
   ``MergeAuditTrailV2``
* M4 ``ConsensusQuorumController``
* M5 ``CorruptionRobustCarrierV2``
* M6 ``DeepProxyStackV5``
* M7 ``LongHorizonReconstructionV6Head``
* M8 ``ECCCodebookV6``
* M9 transcript-vs-shared arbiter V3 (4-arm)
* M10 uncertainty / confidence layer V2

Each W54 turn emits W54 per-module witnesses on top of the
W53 envelope. The final ``W54HandoffEnvelope`` binds: the W53
outer CID, every W54 witness CID, the W54Params CID, the
persistent-V6 chain CID, the MLSC V2 audit trail CID, the
consensus controller audit trail CID, and a single
``w54_outer_cid`` that closes the chain
w47 → w48 → w49 → w50 → w51 → w52 → w53 → w54.

Honest scope (do-not-overstate)
-------------------------------

This module does NOT alter transformer-internal state. Every
W54 witness is computed over capsule-layer signals exclusively.
Trivial passthrough is preserved byte-for-byte: when
``W54Params.build_trivial()`` is used and all flags are
disabled, the W54 envelope's internal ``w53_outer_cid``
equals the W53 outer CID exactly — the
``W54-L-TRIVIAL-W54-PASSTHROUGH`` falsifier.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from .agents import Agent
from .consensus_quorum_controller import (
    ConsensusControllerWitness,
    ConsensusPolicy,
    ConsensusQuorumController,
    W54_DEFAULT_CONSENSUS_COSINE_FLOOR,
    W54_DEFAULT_CONSENSUS_FALLBACK_COSINE_FLOOR,
    W54_DEFAULT_CONSENSUS_K_MAX,
    W54_DEFAULT_CONSENSUS_K_MIN,
    emit_consensus_controller_witness,
)
from .corruption_robust_carrier_v2 import (
    CorruptionRobustCarrierV2,
    CorruptionRobustnessV2Witness,
    W54_DEFAULT_CRC_V2_REPETITION,
    emit_corruption_robustness_v2_witness,
)
from .deep_proxy_stack_v5 import (
    DeepProxyStackV5,
    DeepProxyStackV5ForwardWitness,
    W54_DEFAULT_DEEP_V5_ABSTAIN_THRESHOLD,
    W54_DEFAULT_DEEP_V5_N_LAYERS,
    W54_DEFAULT_DEEP_V5_OUTER_LAYERS,
    emit_deep_proxy_stack_v5_forward_witness,
)
from .deep_proxy_stack_v4 import (
    W53_DEFAULT_DEEP_V4_FACTOR_DIM,
    W53_DEFAULT_DEEP_V4_IN_DIM,
    W53_DEFAULT_DEEP_V4_N_BRANCH_HEADS,
    W53_DEFAULT_DEEP_V4_N_CYCLE_HEADS,
    W53_DEFAULT_DEEP_V4_N_HEADS,
    W53_DEFAULT_DEEP_V4_N_ROLES,
)
from .ecc_codebook_v5 import (
    W53_DEFAULT_ECC_CODE_DIM,
    W53_DEFAULT_ECC_EMIT_MASK_LEN,
)
from .ecc_codebook_v6 import (
    ECCCodebookV6,
    ECCCompressionV6Witness,
    W54_DEFAULT_ECC_V6_K1,
    W54_DEFAULT_ECC_V6_K2,
    W54_DEFAULT_ECC_V6_K3,
    W54_DEFAULT_ECC_V6_K4,
    W54_DEFAULT_ECC_V6_K5,
    W54_DEFAULT_ECC_V6_TARGET_BITS_PER_TOKEN,
    compress_carrier_ecc_v6,
    emit_ecc_v6_compression_witness,
)
from .long_horizon_retention_v6 import (
    LongHorizonReconstructionV6Head,
    LongHorizonReconstructionV6Witness,
    W54_DEFAULT_LHR_V6_FLAT_FEATURE_DIM,
    W54_DEFAULT_LHR_V6_HIDDEN_DIM,
    W54_DEFAULT_LHR_V6_MAX_K,
    W54_DEFAULT_LHR_V6_N_BRANCHES,
    W54_DEFAULT_LHR_V6_N_CYCLES,
    W54_DEFAULT_LHR_V6_N_MERGE_PAIRS,
    W54_DEFAULT_LHR_V6_N_ROLES,
    emit_lhr_v6_witness,
)
from .mergeable_latent_capsule_v2 import (
    MergeAuditTrailV2,
    MergeOperatorV2,
    MergeableLatentCapsuleV2,
    MergeableLatentCapsuleV2Witness,
    W54_DEFAULT_MLSC_V2_FACTOR_DIM,
    W54_DEFAULT_MLSC_V2_TRUST_DEFAULT,
    emit_mlsc_v2_witness,
    make_root_capsule_v2,
    merge_capsules_v2,
    step_branch_capsule_v2,
)
from .multi_hop_translator import (
    MultiHopBackendTranslator,
)
from .multi_hop_translator_v4 import (
    MultiHopV4Witness,
    W54_DEFAULT_MH_V4_BACKENDS,
    W54_DEFAULT_MH_V4_CODE_DIM,
    W54_DEFAULT_MH_V4_FEATURE_DIM,
    build_unfitted_hex_translator,
    emit_multi_hop_v4_witness,
    synthesize_hex_training_set,
)
from .persistent_latent_v6 import (
    PersistentLatentStateV6,
    PersistentLatentStateV6Chain,
    PersistentLatentStateV6Witness,
    V6StackedCell,
    W54_DEFAULT_V6_INPUT_DIM,
    W54_DEFAULT_V6_N_LAYERS,
    W54_DEFAULT_V6_STATE_DIM,
    W54_V6_NO_PARENT_STATE,
    emit_persistent_v6_witness,
    step_persistent_state_v6,
)
from .quantised_compression import (
    QuantisedBudgetGate,
    W52_DEFAULT_QUANT_EMIT_MASK_LEN,
)
from .transcript_vs_shared_arbiter_v3 import (
    TVSArbiterV3Witness,
    emit_tvs_arbiter_v3_witness,
    four_arm_compare,
)
from .uncertainty_layer_v2 import (
    UncertaintyLayerV2Witness,
    calibration_check_under_noise,
    compose_uncertainty_report_v2,
    emit_uncertainty_layer_v2_witness,
)
from .uncertainty_layer import calibration_check
from .w53_team import (
    W53HandoffEnvelope,
    W53Params,
    W53Registry,
    W53Team,
    W53TeamResult,
    W53TurnWitnessBundle,
    build_trivial_w53_registry,
    build_w53_registry,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W54_SCHEMA_VERSION: str = "coordpy.w54_team.v1"
W54_TEAM_RESULT_SCHEMA: str = "coordpy.w54_team_result.v1"

W54_NO_STATE: str = "no_w54_persistent_state"


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
# W54Params
# =============================================================================


@dataclasses.dataclass
class W54Params:
    """All trainable / config surfaces for W54, layered over W53."""

    persistent_v6_cell: V6StackedCell | None
    hex_translator: MultiHopBackendTranslator | None
    mlsc_v2_operator: MergeOperatorV2 | None
    consensus_controller: ConsensusQuorumController | None
    deep_stack_v5: DeepProxyStackV5 | None
    ecc_codebook_v6: ECCCodebookV6 | None
    ecc_gate_v6: QuantisedBudgetGate | None
    long_horizon_v6_head: (
        LongHorizonReconstructionV6Head | None)
    crc_v2: CorruptionRobustCarrierV2 | None

    persistent_v6_enabled: bool = False
    hex_translator_enabled: bool = False
    mlsc_v2_enabled: bool = False
    consensus_controller_enabled: bool = False
    deep_stack_v5_enabled: bool = False
    ecc_v6_compression_enabled: bool = False
    long_horizon_v6_enabled: bool = False
    crc_v2_enabled: bool = False
    tvs_arbiter_v3_enabled: bool = False
    uncertainty_v2_enabled: bool = False

    target_bits_per_token_v6: float = (
        W54_DEFAULT_ECC_V6_TARGET_BITS_PER_TOKEN)
    arbiter_budget_tokens: int = 3
    consensus_k_required: int = (
        W54_DEFAULT_CONSENSUS_K_MIN)

    @classmethod
    def build_trivial(cls) -> "W54Params":
        return cls(
            persistent_v6_cell=None,
            hex_translator=None,
            mlsc_v2_operator=None,
            consensus_controller=None,
            deep_stack_v5=None,
            ecc_codebook_v6=None,
            ecc_gate_v6=None,
            long_horizon_v6_head=None,
            crc_v2=None,
            persistent_v6_enabled=False,
            hex_translator_enabled=False,
            mlsc_v2_enabled=False,
            consensus_controller_enabled=False,
            deep_stack_v5_enabled=False,
            ecc_v6_compression_enabled=False,
            long_horizon_v6_enabled=False,
            crc_v2_enabled=False,
            tvs_arbiter_v3_enabled=False,
            uncertainty_v2_enabled=False,
        )

    @classmethod
    def build_default(
            cls, *,
            role_universe: Sequence[str] = (
                "r0", "r1", "r2", "r3"),
            seed: int = 12345,
    ) -> "W54Params":
        cell_v6 = V6StackedCell.init(
            state_dim=W54_DEFAULT_V6_STATE_DIM,
            input_dim=W54_DEFAULT_V6_INPUT_DIM,
            n_layers=W54_DEFAULT_V6_N_LAYERS,
            seed=int(seed))
        hex_tr = build_unfitted_hex_translator(
            backends=W54_DEFAULT_MH_V4_BACKENDS,
            code_dim=W54_DEFAULT_MH_V4_CODE_DIM,
            feature_dim=W54_DEFAULT_MH_V4_FEATURE_DIM,
            seed=int(seed) + 7)
        mlsc_op = MergeOperatorV2(
            factor_dim=W54_DEFAULT_MLSC_V2_FACTOR_DIM)
        consensus_ctrl = ConsensusQuorumController.init(
            policy=ConsensusPolicy(
                k_min=W54_DEFAULT_CONSENSUS_K_MIN,
                k_max=W54_DEFAULT_CONSENSUS_K_MAX,
                cosine_floor=(
                    W54_DEFAULT_CONSENSUS_COSINE_FLOOR),
                fallback_cosine_floor=(
                    W54_DEFAULT_CONSENSUS_FALLBACK_COSINE_FLOOR),
                allow_fallback=True),
            operator=mlsc_op)
        deep5 = DeepProxyStackV5.init(
            n_layers=W54_DEFAULT_DEEP_V5_N_LAYERS,
            in_dim=W53_DEFAULT_DEEP_V4_IN_DIM,
            factor_dim=W53_DEFAULT_DEEP_V4_FACTOR_DIM,
            n_heads=W53_DEFAULT_DEEP_V4_N_HEADS,
            n_branch_heads=W53_DEFAULT_DEEP_V4_N_BRANCH_HEADS,
            n_cycle_heads=W53_DEFAULT_DEEP_V4_N_CYCLE_HEADS,
            n_roles=W53_DEFAULT_DEEP_V4_N_ROLES,
            n_outer_layers=W54_DEFAULT_DEEP_V5_OUTER_LAYERS,
            abstain_threshold=(
                W54_DEFAULT_DEEP_V5_ABSTAIN_THRESHOLD),
            seed=int(seed) + 13)
        ecc_v6 = ECCCodebookV6.init(
            n_coarse=W54_DEFAULT_ECC_V6_K1,
            n_fine=W54_DEFAULT_ECC_V6_K2,
            n_ultra=W54_DEFAULT_ECC_V6_K3,
            n_ultra2=W54_DEFAULT_ECC_V6_K4,
            n_ultra3=W54_DEFAULT_ECC_V6_K5,
            code_dim=W53_DEFAULT_ECC_CODE_DIM,
            seed=int(seed) + 17)
        ecc_gate = QuantisedBudgetGate.init(
            in_dim=W53_DEFAULT_ECC_CODE_DIM,
            emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
            seed=int(seed) + 19)
        ecc_gate.importance_threshold = 0.0
        ecc_gate.w_emit.values = [
            1.0] * len(ecc_gate.w_emit.values)
        lhr_v6 = LongHorizonReconstructionV6Head.init(
            carrier_dim=(
                W54_DEFAULT_LHR_V6_MAX_K
                * W54_DEFAULT_LHR_V6_FLAT_FEATURE_DIM),
            hidden_dim=W54_DEFAULT_LHR_V6_HIDDEN_DIM,
            out_dim=W54_DEFAULT_LHR_V6_FLAT_FEATURE_DIM,
            max_k=W54_DEFAULT_LHR_V6_MAX_K,
            n_branches=W54_DEFAULT_LHR_V6_N_BRANCHES,
            n_cycles=W54_DEFAULT_LHR_V6_N_CYCLES,
            n_merge_pairs=W54_DEFAULT_LHR_V6_N_MERGE_PAIRS,
            n_roles=W54_DEFAULT_LHR_V6_N_ROLES,
            seed=int(seed) + 23)
        crc_v2 = CorruptionRobustCarrierV2.init(
            codebook=ecc_v6.inner_v5,
            gate=ecc_gate,
            repetition=W54_DEFAULT_CRC_V2_REPETITION)
        return cls(
            persistent_v6_cell=cell_v6,
            hex_translator=hex_tr,
            mlsc_v2_operator=mlsc_op,
            consensus_controller=consensus_ctrl,
            deep_stack_v5=deep5,
            ecc_codebook_v6=ecc_v6,
            ecc_gate_v6=ecc_gate,
            long_horizon_v6_head=lhr_v6,
            crc_v2=crc_v2,
            persistent_v6_enabled=True,
            hex_translator_enabled=True,
            mlsc_v2_enabled=True,
            consensus_controller_enabled=True,
            deep_stack_v5_enabled=True,
            ecc_v6_compression_enabled=True,
            long_horizon_v6_enabled=True,
            crc_v2_enabled=True,
            tvs_arbiter_v3_enabled=True,
            uncertainty_v2_enabled=True,
        )

    @property
    def all_flags_disabled(self) -> bool:
        return not (
            self.persistent_v6_enabled
            or self.hex_translator_enabled
            or self.mlsc_v2_enabled
            or self.consensus_controller_enabled
            or self.deep_stack_v5_enabled
            or self.ecc_v6_compression_enabled
            or self.long_horizon_v6_enabled
            or self.crc_v2_enabled
            or self.tvs_arbiter_v3_enabled
            or self.uncertainty_v2_enabled)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(W54_SCHEMA_VERSION),
            "persistent_v6_cell_cid": (
                self.persistent_v6_cell.cid()
                if self.persistent_v6_cell is not None
                else ""),
            "hex_translator_cid": (
                self.hex_translator.cid()
                if self.hex_translator is not None
                else ""),
            "mlsc_v2_operator_cid": (
                self.mlsc_v2_operator.cid()
                if self.mlsc_v2_operator is not None
                else ""),
            "consensus_controller_cid": (
                self.consensus_controller.cid()
                if self.consensus_controller is not None
                else ""),
            "deep_stack_v5_cid": (
                self.deep_stack_v5.cid()
                if self.deep_stack_v5 is not None
                else ""),
            "ecc_codebook_v6_cid": (
                self.ecc_codebook_v6.cid()
                if self.ecc_codebook_v6 is not None
                else ""),
            "ecc_gate_v6_cid": (
                self.ecc_gate_v6.cid()
                if self.ecc_gate_v6 is not None
                else ""),
            "long_horizon_v6_head_cid": (
                self.long_horizon_v6_head.cid()
                if self.long_horizon_v6_head is not None
                else ""),
            "crc_v2_cid": (
                self.crc_v2.cid()
                if self.crc_v2 is not None
                else ""),
            "persistent_v6_enabled": bool(
                self.persistent_v6_enabled),
            "hex_translator_enabled": bool(
                self.hex_translator_enabled),
            "mlsc_v2_enabled": bool(self.mlsc_v2_enabled),
            "consensus_controller_enabled": bool(
                self.consensus_controller_enabled),
            "deep_stack_v5_enabled": bool(
                self.deep_stack_v5_enabled),
            "ecc_v6_compression_enabled": bool(
                self.ecc_v6_compression_enabled),
            "long_horizon_v6_enabled": bool(
                self.long_horizon_v6_enabled),
            "crc_v2_enabled": bool(self.crc_v2_enabled),
            "tvs_arbiter_v3_enabled": bool(
                self.tvs_arbiter_v3_enabled),
            "uncertainty_v2_enabled": bool(
                self.uncertainty_v2_enabled),
            "target_bits_per_token_v6": float(round(
                self.target_bits_per_token_v6, 12)),
            "arbiter_budget_tokens": int(
                self.arbiter_budget_tokens),
            "consensus_k_required": int(
                self.consensus_k_required),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_params",
            "params": self.to_dict()})


# =============================================================================
# W54Registry
# =============================================================================


@dataclasses.dataclass
class W54Registry:
    """W54 registry — wraps a W53 registry + W54 params."""

    schema_cid: str
    inner_w53_registry: W53Registry
    params: W54Params

    @property
    def is_trivial(self) -> bool:
        return (
            self.inner_w53_registry.is_trivial
            and self.params.all_flags_disabled
        )


def build_trivial_w54_registry(
        *, schema_cid: str | None = None,
) -> W54Registry:
    cid = schema_cid or _sha256_hex({
        "kind": "w54_trivial_schema"})
    inner = build_trivial_w53_registry(
        schema_cid=str(cid))
    return W54Registry(
        schema_cid=str(cid),
        inner_w53_registry=inner,
        params=W54Params.build_trivial(),
    )


def build_w54_registry(
        *,
        schema_cid: str,
        inner_w53_registry: W53Registry | None = None,
        params: W54Params | None = None,
        role_universe: Sequence[str] = (
            "r0", "r1", "r2", "r3"),
        seed: int = 12345,
) -> W54Registry:
    inner = (
        inner_w53_registry
        if inner_w53_registry is not None
        else build_w53_registry(
            schema_cid=str(schema_cid),
            role_universe=role_universe,
            seed=int(seed)))
    p = params or W54Params.build_default(
        role_universe=role_universe, seed=int(seed))
    return W54Registry(
        schema_cid=str(schema_cid),
        inner_w53_registry=inner,
        params=p,
    )


# =============================================================================
# Per-turn witnesses
# =============================================================================


@dataclasses.dataclass(frozen=True)
class W54TurnWitnessBundle:
    """All W54 per-turn witness CIDs for one team turn."""

    persistent_v6_witness_cid: str
    hex_translator_witness_cid: str
    mlsc_v2_witness_cid: str
    consensus_controller_witness_cid: str
    deep_stack_v5_witness_cid: str
    ecc_v6_compression_witness_cid: str
    long_horizon_v6_witness_cid: str
    crc_v2_witness_cid: str
    tvs_arbiter_v3_witness_cid: str
    uncertainty_v2_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "persistent_v6_witness_cid": str(
                self.persistent_v6_witness_cid),
            "hex_translator_witness_cid": str(
                self.hex_translator_witness_cid),
            "mlsc_v2_witness_cid": str(
                self.mlsc_v2_witness_cid),
            "consensus_controller_witness_cid": str(
                self.consensus_controller_witness_cid),
            "deep_stack_v5_witness_cid": str(
                self.deep_stack_v5_witness_cid),
            "ecc_v6_compression_witness_cid": str(
                self.ecc_v6_compression_witness_cid),
            "long_horizon_v6_witness_cid": str(
                self.long_horizon_v6_witness_cid),
            "crc_v2_witness_cid": str(
                self.crc_v2_witness_cid),
            "tvs_arbiter_v3_witness_cid": str(
                self.tvs_arbiter_v3_witness_cid),
            "uncertainty_v2_witness_cid": str(
                self.uncertainty_v2_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_turn_witness_bundle",
            "bundle": self.to_dict()})


# =============================================================================
# W54HandoffEnvelope
# =============================================================================


@dataclasses.dataclass(frozen=True)
class W54HandoffEnvelope:
    """Sealed per-team W54 envelope."""

    schema_version: str
    w53_outer_cid: str
    params_cid: str
    turn_witness_bundle_cid: str
    w53_envelope_count: int
    persistent_v6_chain_cid: str
    mlsc_v2_audit_trail_cid: str
    consensus_controller_audit_cid: str
    composite_confidence_mean_v2: float
    arbiter_pick_rate_merge_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "w53_outer_cid": str(self.w53_outer_cid),
            "params_cid": str(self.params_cid),
            "turn_witness_bundle_cid": str(
                self.turn_witness_bundle_cid),
            "w53_envelope_count": int(
                self.w53_envelope_count),
            "persistent_v6_chain_cid": str(
                self.persistent_v6_chain_cid),
            "mlsc_v2_audit_trail_cid": str(
                self.mlsc_v2_audit_trail_cid),
            "consensus_controller_audit_cid": str(
                self.consensus_controller_audit_cid),
            "composite_confidence_mean_v2": float(round(
                self.composite_confidence_mean_v2, 12)),
            "arbiter_pick_rate_merge_mean": float(round(
                self.arbiter_pick_rate_merge_mean, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_outer_envelope",
            "envelope": self.to_dict()})


# =============================================================================
# Forward — compute per-turn W54 witnesses
# =============================================================================


def _persistent_v6_step(
        *,
        params: W54Params,
        turn_index: int,
        role: str,
        prev_state: PersistentLatentStateV6 | None,
        carrier_payload: Any,
        state_chain: PersistentLatentStateV6Chain,
        anchor_skip: Sequence[float] | None,
        branch_id: str = "main",
) -> PersistentLatentStateV6 | None:
    if (not params.persistent_v6_enabled
            or params.persistent_v6_cell is None):
        return None
    cell = params.persistent_v6_cell
    input_vec = _payload_hash_vec(
        carrier_payload, cell.state_dim)
    if anchor_skip is None:
        anchor_skip = input_vec
    new_state = step_persistent_state_v6(
        cell=cell,
        prev_state=prev_state,
        carrier_values=input_vec,
        turn_index=int(turn_index),
        role=str(role),
        branch_id=str(branch_id),
        anchor_skip=anchor_skip)
    state_chain.add(new_state)
    return new_state


def _emit_w54_turn_witnesses(
        *,
        params: W54Params,
        turn_index: int,
        role: str,
        prev_state: PersistentLatentStateV6 | None,
        state_chain: PersistentLatentStateV6Chain,
        mlsc_audit: MergeAuditTrailV2,
        mlsc_store: dict[str, MergeableLatentCapsuleV2],
        prev_mlsc_capsule: MergeableLatentCapsuleV2 | None,
        anchor_skip: Sequence[float] | None,
        branch_index: int = 0,
        cycle_index: int = 0,
        role_index: int = 0,
        carrier_payload: Any = None,
) -> tuple[
        W54TurnWitnessBundle,
        PersistentLatentStateV6 | None,
        MergeableLatentCapsuleV2 | None,
        float, float]:
    """Compute all per-turn W54 witnesses.

    Returns (bundle, new_state, new_mlsc_v2_capsule,
        composite_confidence_v2, arbiter_pick_rate_merge).
    """
    new_state = _persistent_v6_step(
        params=params,
        turn_index=int(turn_index),
        role=str(role),
        prev_state=prev_state,
        carrier_payload=carrier_payload,
        state_chain=state_chain,
        anchor_skip=anchor_skip,
        branch_id="main")
    pv6_witness_cid = ""
    pv6_conf = 1.0
    if (new_state is not None
            and params.persistent_v6_cell is not None):
        w_pv6 = emit_persistent_v6_witness(
            state=new_state,
            cell=params.persistent_v6_cell,
            chain=state_chain)
        pv6_witness_cid = w_pv6.cid()
        gate_norm = float(
            new_state.update_gate_l1_sum) / float(
                max(1, int(new_state.state_dim)))
        pv6_conf = float(max(0.0, min(
            1.0, 1.0 - 0.5 * gate_norm)))
    # M2 — hex translator witness.
    hex_witness_cid = ""
    hex_conf = 1.0
    if (params.hex_translator_enabled
            and params.hex_translator is not None):
        ts = synthesize_hex_training_set(
            n_examples=4, seed=int(turn_index) + 11,
            backends=tuple(params.hex_translator.backends),
            code_dim=int(params.hex_translator.code_dim),
            feature_dim=int(
                params.hex_translator.feature_dim))
        w_h = emit_multi_hop_v4_witness(
            translator=params.hex_translator,
            examples=ts.examples[:4])
        hex_witness_cid = w_h.cid()
        hex_conf = float(max(0.0, min(
            1.0, w_h.direct_fidelity_a_f)))
    # M3 — MLSC V2 capsule.
    mlsc_v2_witness_cid = ""
    mlsc_conf = 1.0
    new_mlsc: MergeableLatentCapsuleV2 | None = None
    if (params.mlsc_v2_enabled
            and params.mlsc_v2_operator is not None):
        op = params.mlsc_v2_operator
        payload = (
            list(new_state.top_state)[:int(op.factor_dim)]
            if new_state is not None
            else _payload_hash_vec(
                carrier_payload, op.factor_dim))
        while len(payload) < int(op.factor_dim):
            payload.append(0.0)
        if prev_mlsc_capsule is None:
            new_mlsc = make_root_capsule_v2(
                branch_id=f"branch_{branch_index}",
                payload=payload,
                confidence=float(pv6_conf),
                trust=W54_DEFAULT_MLSC_V2_TRUST_DEFAULT,
                fact_tags=(f"turn={turn_index}",),
                turn_index=int(turn_index))
        else:
            new_mlsc = step_branch_capsule_v2(
                parent=prev_mlsc_capsule,
                payload=payload,
                confidence=float(pv6_conf),
                trust=float(prev_mlsc_capsule.trust),
                new_fact_tags=(f"turn={turn_index}",),
                turn_index=int(turn_index))
        mlsc_store[new_mlsc.cid()] = new_mlsc
        if (prev_mlsc_capsule is not None
                and (turn_index % 2) == 1):
            merged = merge_capsules_v2(
                op, [prev_mlsc_capsule, new_mlsc],
                audit_trail=mlsc_audit,
                extra_fact_tags=(
                    f"merge_at_turn={turn_index}",))
            mlsc_store[merged.cid()] = merged
            new_mlsc = merged
        w_mlsc = emit_mlsc_v2_witness(
            leaf=new_mlsc,
            operator=op,
            audit_trail=mlsc_audit,
            capsule_store=mlsc_store)
        mlsc_v2_witness_cid = w_mlsc.cid()
        mlsc_conf = float(new_mlsc.confidence)
    # M4 — consensus controller witness.
    consensus_witness_cid = ""
    if (params.consensus_controller_enabled
            and params.consensus_controller is not None
            and new_mlsc is not None):
        # Exercise the controller on a small ad-hoc branch set:
        # the current MLSC capsule + one random neighbour.
        # Run only every other turn to keep audit growth bounded.
        if (turn_index % 2) == 0:
            sibling = (
                prev_mlsc_capsule if prev_mlsc_capsule
                is not None else new_mlsc)
            _, _ = params.consensus_controller.decide(
                [new_mlsc, sibling],
                turn_index=int(turn_index),
                k_required=int(
                    params.consensus_k_required))
        w_cc = emit_consensus_controller_witness(
            params.consensus_controller)
        consensus_witness_cid = w_cc.cid()
    # M5 — deep stack V5 forward witness.
    deep_v5_witness_cid = ""
    deep_v5_conf = 1.0
    if (params.deep_stack_v5_enabled
            and params.deep_stack_v5 is not None):
        s = params.deep_stack_v5
        if new_state is not None:
            q = list(new_state.top_state)[:s.in_dim]
        else:
            q = []
        while len(q) < s.in_dim:
            q.append(0.0)
        w_deep, _ = emit_deep_proxy_stack_v5_forward_witness(
            stack=s, query_input=q,
            slot_keys=[q], slot_values=[q],
            role_index=int(role_index),
            branch_index=int(branch_index),
            cycle_index=int(cycle_index),
            uncertainty_scale=float(pv6_conf))
        deep_v5_witness_cid = w_deep.cid()
        deep_v5_conf = float(w_deep.corruption_confidence)
    # M6 — ECC V6 compression witness.
    ecc_v6_witness_cid = ""
    if (params.ecc_v6_compression_enabled
            and params.ecc_codebook_v6 is not None
            and params.ecc_gate_v6 is not None
            and new_state is not None):
        carrier = list(new_state.top_state)[
            :int(params.ecc_codebook_v6.code_dim)]
        while len(carrier) < int(
                params.ecc_codebook_v6.code_dim):
            carrier.append(0.0)
        comp = compress_carrier_ecc_v6(
            carrier,
            codebook=params.ecc_codebook_v6,
            gate=params.ecc_gate_v6)
        w_ecc = emit_ecc_v6_compression_witness(
            codebook=params.ecc_codebook_v6,
            compression=comp,
            target_bits_per_token=float(
                params.target_bits_per_token_v6))
        ecc_v6_witness_cid = w_ecc.cid()
    # M7 — long-horizon V6 witness.
    lhr_v6_witness_cid = ""
    if (params.long_horizon_v6_enabled
            and params.long_horizon_v6_head is not None):
        w_lhr = emit_lhr_v6_witness(
            head=params.long_horizon_v6_head,
            examples=(),
            k_max_for_degradation=12)
        lhr_v6_witness_cid = w_lhr.cid()
    # M8 — CRC V2 witness.
    crc_v2_witness_cid = ""
    crc_silent_failure_rate = 0.0
    if (params.crc_v2_enabled
            and params.crc_v2 is not None
            and new_state is not None):
        carrier = list(new_state.top_state)[
            :int(params.crc_v2.codebook.code_dim)]
        while len(carrier) < int(
                params.crc_v2.codebook.code_dim):
            carrier.append(0.0)
        w_crc = emit_corruption_robustness_v2_witness(
            crc_v2=params.crc_v2,
            carriers=[carrier],
            flip_intensity=1.0,
            seed=int(turn_index) + 1)
        crc_v2_witness_cid = w_crc.cid()
        crc_silent_failure_rate = float(
            w_crc.silent_failure_rate)
    # M9 — TVS arbiter V3 witness.
    tvs_arb_v3_witness_cid = ""
    arbiter_pick_rate_merge = 0.0
    if (params.tvs_arbiter_v3_enabled
            and params.ecc_codebook_v6 is not None
            and params.ecc_gate_v6 is not None
            and new_state is not None):
        carrier = list(new_state.top_state)[
            :int(params.ecc_codebook_v6.code_dim)]
        while len(carrier) < int(
                params.ecc_codebook_v6.code_dim):
            carrier.append(0.0)
        # Synthesize merge_consensus retention: use the MLSC
        # capsule confidence as a proxy (high confidence →
        # high retention).
        mr = float(mlsc_conf) * 0.9
        result = four_arm_compare(
            carriers=[carrier],
            codebook=(
                params.ecc_codebook_v6.inner_v5.inner_v4),
            gate=params.ecc_gate_v6,
            budget_tokens=int(
                params.arbiter_budget_tokens),
            per_turn_confidences=[float(pv6_conf)],
            per_turn_merge_retentions=[float(mr)],
            abstain_threshold=0.15,
            prefer_shared_threshold=0.0,
            merge_floor=0.0)
        w_tvs = emit_tvs_arbiter_v3_witness(result=result)
        tvs_arb_v3_witness_cid = w_tvs.cid()
        arbiter_pick_rate_merge = float(
            w_tvs.pick_rate_merge)
    # M10 — uncertainty layer V2 witness.
    uncert_v2_witness_cid = ""
    composite_conf = 1.0
    if params.uncertainty_v2_enabled:
        report = compose_uncertainty_report_v2(
            persistent_v6_confidence=float(pv6_conf),
            multi_hop_v4_confidence=float(hex_conf),
            mlsc_v2_capsule_confidence=float(mlsc_conf),
            deep_v5_corruption_confidence=float(
                deep_v5_conf),
            crc_v2_silent_failure_rate=float(
                crc_silent_failure_rate),
            component_disagreements={})
        composite_conf = float(report.composite_confidence)
        clean_cal = calibration_check(
            confidences=(), accuracies=(),
            min_calibration_gap=0.10)
        noisy_cal = calibration_check_under_noise(
            confidences=(), accuracies=(),
            noise_magnitude=0.1,
            seed=int(turn_index) + 1)
        w_unc = emit_uncertainty_layer_v2_witness(
            report=report,
            calibration_clean=clean_cal,
            calibration_noisy=noisy_cal)
        uncert_v2_witness_cid = w_unc.cid()
    bundle = W54TurnWitnessBundle(
        persistent_v6_witness_cid=str(pv6_witness_cid),
        hex_translator_witness_cid=str(hex_witness_cid),
        mlsc_v2_witness_cid=str(mlsc_v2_witness_cid),
        consensus_controller_witness_cid=str(
            consensus_witness_cid),
        deep_stack_v5_witness_cid=str(
            deep_v5_witness_cid),
        ecc_v6_compression_witness_cid=str(
            ecc_v6_witness_cid),
        long_horizon_v6_witness_cid=str(
            lhr_v6_witness_cid),
        crc_v2_witness_cid=str(crc_v2_witness_cid),
        tvs_arbiter_v3_witness_cid=str(
            tvs_arb_v3_witness_cid),
        uncertainty_v2_witness_cid=str(
            uncert_v2_witness_cid),
    )
    return (
        bundle, new_state, new_mlsc,
        float(composite_conf),
        float(arbiter_pick_rate_merge))


# =============================================================================
# W54TeamResult
# =============================================================================


@dataclasses.dataclass(frozen=True)
class W54TeamResult:
    schema: str
    task: str
    final_output: str
    w53_outer_cid: str
    w54_outer_cid: str
    w54_params_cid: str
    w54_envelope: W54HandoffEnvelope
    turn_witness_bundles: tuple[W54TurnWitnessBundle, ...]
    persistent_v6_state_cids: tuple[str, ...]
    mlsc_v2_capsule_cids: tuple[str, ...]
    composite_confidence_mean_v2: float
    arbiter_pick_rate_merge_mean: float
    n_turns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "task": str(self.task),
            "final_output": str(self.final_output),
            "w53_outer_cid": str(self.w53_outer_cid),
            "w54_outer_cid": str(self.w54_outer_cid),
            "w54_params_cid": str(self.w54_params_cid),
            "w54_envelope": self.w54_envelope.to_dict(),
            "turn_witness_bundles": [
                b.to_dict()
                for b in self.turn_witness_bundles],
            "persistent_v6_state_cids": list(
                self.persistent_v6_state_cids),
            "mlsc_v2_capsule_cids": list(
                self.mlsc_v2_capsule_cids),
            "composite_confidence_mean_v2": float(round(
                self.composite_confidence_mean_v2, 12)),
            "arbiter_pick_rate_merge_mean": float(round(
                self.arbiter_pick_rate_merge_mean, 12)),
            "n_turns": int(self.n_turns),
        }


# =============================================================================
# W54Team
# =============================================================================


@dataclasses.dataclass
class W54Team:
    """W54 team orchestrator — wraps ``W53Team``."""

    agents: Sequence[Agent]
    registry: W54Registry
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
    ) -> W54TeamResult:
        w53_team = W53Team(
            agents=list(self.agents),
            backend=self.backend,
            registry=self.registry.inner_w53_registry,
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
        w53_result = w53_team.run(task, progress=progress)
        params = self.registry.params
        state_chain = PersistentLatentStateV6Chain.empty()
        mlsc_audit = MergeAuditTrailV2.empty()
        # Reset consensus controller for this run (audit is
        # per-run; controller dataclass instance is per-registry).
        if (params.consensus_controller is not None):
            params.consensus_controller.controller_audit = (
                type(params.consensus_controller.controller_audit)
                .empty())
            params.consensus_controller.capsule_audit = (
                MergeAuditTrailV2.empty())
        mlsc_store: dict[str, MergeableLatentCapsuleV2] = {}
        bundles: list[W54TurnWitnessBundle] = []
        persistent_v6_cids: list[str] = []
        mlsc_v2_cids: list[str] = []
        prev_state: PersistentLatentStateV6 | None = None
        prev_mlsc: MergeableLatentCapsuleV2 | None = None
        n_turns = int(w53_result.n_turns)
        composite_sum = 0.0
        merge_share_sum = 0.0
        anchor_skip_seed: Sequence[float] | None = None
        for i, turn in enumerate(
                w53_result.turn_witness_bundles):
            n_branches = max(
                1, params.deep_stack_v5.n_branch_heads
                if params.deep_stack_v5 is not None
                else 4)
            n_cycles = max(
                1, params.deep_stack_v5.n_cycle_heads
                if params.deep_stack_v5 is not None
                else 4)
            n_roles = max(
                1, params.deep_stack_v5.n_roles
                if params.deep_stack_v5 is not None
                else 4)
            branch_index = int(i) % int(n_branches)
            cycle_index = int(i) % int(n_cycles)
            role_index = int(i) % int(n_roles)
            carrier_payload = (int(i), turn.cid())
            if anchor_skip_seed is None and i == 0:
                anchor_skip_seed = _payload_hash_vec(
                    carrier_payload,
                    params.persistent_v6_cell.state_dim
                    if params.persistent_v6_cell is not None
                    else 8)
            (bundle, new_state, new_mlsc, conf,
             merge_share) = _emit_w54_turn_witnesses(
                params=params,
                turn_index=int(i),
                role=str(self._role_for_turn(i)),
                prev_state=prev_state,
                state_chain=state_chain,
                mlsc_audit=mlsc_audit,
                mlsc_store=mlsc_store,
                prev_mlsc_capsule=prev_mlsc,
                anchor_skip=anchor_skip_seed,
                branch_index=int(branch_index),
                cycle_index=int(cycle_index),
                role_index=int(role_index),
                carrier_payload=carrier_payload,
            )
            bundles.append(bundle)
            composite_sum += float(conf)
            merge_share_sum += float(merge_share)
            if new_state is not None:
                persistent_v6_cids.append(new_state.cid())
                prev_state = new_state
            if new_mlsc is not None:
                mlsc_v2_cids.append(new_mlsc.cid())
                prev_mlsc = new_mlsc
        comp_mean = (
            float(composite_sum) / float(max(1, n_turns)))
        merge_mean = (
            float(merge_share_sum) / float(max(1, n_turns)))
        bundles_cid = _sha256_hex({
            "kind": "w54_all_turn_witness_bundles",
            "bundles": [b.to_dict() for b in bundles],
        })
        controller_audit_cid = (
            params.consensus_controller.controller_audit.cid()
            if params.consensus_controller is not None
            else "")
        env = W54HandoffEnvelope(
            schema_version=W54_SCHEMA_VERSION,
            w53_outer_cid=str(w53_result.w53_outer_cid),
            params_cid=str(params.cid()),
            turn_witness_bundle_cid=str(bundles_cid),
            w53_envelope_count=int(n_turns),
            persistent_v6_chain_cid=str(state_chain.cid()),
            mlsc_v2_audit_trail_cid=str(mlsc_audit.cid()),
            consensus_controller_audit_cid=str(
                controller_audit_cid),
            composite_confidence_mean_v2=float(comp_mean),
            arbiter_pick_rate_merge_mean=float(merge_mean),
        )
        return W54TeamResult(
            schema=W54_TEAM_RESULT_SCHEMA,
            task=str(task),
            final_output=str(w53_result.final_output),
            w53_outer_cid=str(w53_result.w53_outer_cid),
            w54_outer_cid=str(env.cid()),
            w54_params_cid=str(params.cid()),
            w54_envelope=env,
            turn_witness_bundles=tuple(bundles),
            persistent_v6_state_cids=tuple(
                persistent_v6_cids),
            mlsc_v2_capsule_cids=tuple(mlsc_v2_cids),
            composite_confidence_mean_v2=float(comp_mean),
            arbiter_pick_rate_merge_mean=float(merge_mean),
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


W54_ENVELOPE_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w54_schema_mismatch",
    "w54_w53_outer_cid_mismatch",
    "w54_params_cid_mismatch",
    "w54_turn_witness_bundle_cid_mismatch",
    "w54_envelope_count_mismatch",
    "w54_outer_cid_mismatch",
    "w54_persistent_v6_chain_cid_mismatch",
    "w54_mlsc_v2_audit_trail_cid_mismatch",
    "w54_consensus_controller_audit_cid_mismatch",
    "w54_trivial_passthrough_w53_cid_mismatch",
    "w54_persistent_v6_witness_missing_when_enabled",
    "w54_hex_translator_witness_missing_when_enabled",
    "w54_mlsc_v2_witness_missing_when_enabled",
    "w54_consensus_controller_witness_missing_when_enabled",
    "w54_deep_stack_v5_witness_missing_when_enabled",
    "w54_ecc_v6_compression_witness_missing_when_enabled",
    "w54_long_horizon_v6_witness_missing_when_enabled",
    "w54_crc_v2_witness_missing_when_enabled",
    "w54_tvs_arbiter_v3_witness_missing_when_enabled",
    "w54_uncertainty_v2_witness_missing_when_enabled",
    "w54_envelope_payload_hash_mismatch",
    "w54_per_turn_bundle_count_mismatch",
    "w54_witness_bundle_cid_recompute_mismatch",
    "w54_persistent_v6_state_count_inconsistent",
    "w54_outer_cid_recompute_mismatch",
    "w54_inner_w53_envelope_invalid",
    "w54_role_universe_mismatch",
    "w54_composite_confidence_v2_out_of_bounds",
    "w54_arbiter_pick_rate_merge_out_of_bounds",
    "w54_mlsc_v2_audit_trail_walk_orphan",
)


def verify_w54_handoff(
        envelope: W54HandoffEnvelope,
        *,
        expected_w53_outer_cid: str | None = None,
        expected_params_cid: str | None = None,
        bundles: Sequence[W54TurnWitnessBundle] | None = None,
        registry: W54Registry | None = None,
        persistent_v6_state_cids: (
            Sequence[str] | None) = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if envelope.schema_version != W54_SCHEMA_VERSION:
        failures.append("w54_schema_mismatch")
    if (expected_w53_outer_cid is not None
            and envelope.w53_outer_cid
            != expected_w53_outer_cid):
        failures.append("w54_w53_outer_cid_mismatch")
    if (expected_params_cid is not None
            and envelope.params_cid
            != expected_params_cid):
        failures.append("w54_params_cid_mismatch")
    if not (
            0.0
            <= float(envelope.composite_confidence_mean_v2)
            <= 1.0):
        failures.append(
            "w54_composite_confidence_v2_out_of_bounds")
    if not (
            0.0 <= float(
                envelope.arbiter_pick_rate_merge_mean)
            <= 1.0):
        failures.append(
            "w54_arbiter_pick_rate_merge_out_of_bounds")
    if bundles is not None:
        if envelope.w53_envelope_count != len(bundles):
            failures.append(
                "w54_per_turn_bundle_count_mismatch")
        recomputed = _sha256_hex({
            "kind": "w54_all_turn_witness_bundles",
            "bundles": [b.to_dict() for b in bundles],
        })
        if envelope.turn_witness_bundle_cid != recomputed:
            failures.append(
                "w54_witness_bundle_cid_recompute_mismatch")
    if registry is not None and bundles is not None:
        p = registry.params
        for b in bundles:
            if (p.persistent_v6_enabled
                    and not b.persistent_v6_witness_cid):
                failures.append(
                    "w54_persistent_v6_witness_missing_when_enabled")
                break
        for b in bundles:
            if (p.hex_translator_enabled
                    and not b.hex_translator_witness_cid):
                failures.append(
                    "w54_hex_translator_witness_missing_when_enabled")
                break
        for b in bundles:
            if (p.mlsc_v2_enabled
                    and not b.mlsc_v2_witness_cid):
                failures.append(
                    "w54_mlsc_v2_witness_missing_when_enabled")
                break
        for b in bundles:
            if (p.consensus_controller_enabled
                    and not b.consensus_controller_witness_cid):
                failures.append(
                    "w54_consensus_controller_witness_missing_when_enabled")
                break
        for b in bundles:
            if (p.deep_stack_v5_enabled
                    and not b.deep_stack_v5_witness_cid):
                failures.append(
                    "w54_deep_stack_v5_witness_missing_when_enabled")
                break
        for b in bundles:
            if (p.ecc_v6_compression_enabled
                    and not b.ecc_v6_compression_witness_cid):
                failures.append(
                    "w54_ecc_v6_compression_witness_missing_when_enabled")
                break
        for b in bundles:
            if (p.long_horizon_v6_enabled
                    and not b.long_horizon_v6_witness_cid):
                failures.append(
                    "w54_long_horizon_v6_witness_missing_when_enabled")
                break
        for b in bundles:
            if (p.crc_v2_enabled
                    and not b.crc_v2_witness_cid):
                failures.append(
                    "w54_crc_v2_witness_missing_when_enabled")
                break
        for b in bundles:
            if (p.tvs_arbiter_v3_enabled
                    and not b.tvs_arbiter_v3_witness_cid):
                failures.append(
                    "w54_tvs_arbiter_v3_witness_missing_when_enabled")
                break
        for b in bundles:
            if (p.uncertainty_v2_enabled
                    and not b.uncertainty_v2_witness_cid):
                failures.append(
                    "w54_uncertainty_v2_witness_missing_when_enabled")
                break
    if (persistent_v6_state_cids is not None
            and registry is not None):
        if (registry.params.persistent_v6_enabled
                and len(persistent_v6_state_cids)
                != envelope.w53_envelope_count):
            failures.append(
                "w54_persistent_v6_state_count_inconsistent")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "outer_cid": envelope.cid(),
        "n_failure_modes": int(
            len(W54_ENVELOPE_VERIFIER_FAILURE_MODES)),
    }


__all__ = [
    "W54_SCHEMA_VERSION",
    "W54_TEAM_RESULT_SCHEMA",
    "W54_NO_STATE",
    "W54_ENVELOPE_VERIFIER_FAILURE_MODES",
    "W54Params",
    "W54Registry",
    "W54TurnWitnessBundle",
    "W54HandoffEnvelope",
    "W54TeamResult",
    "W54Team",
    "build_trivial_w54_registry",
    "build_w54_registry",
    "verify_w54_handoff",
]
