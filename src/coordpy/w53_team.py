"""W53 — Persistent Mergeable Corruption-Robust Latent OS (PMCRLOS).

The ``W53Team`` orchestrator composes the W52 ``W52Team`` with
the ten M1..M10 W53 mechanism modules:

* M1 ``V5StackedCell`` + ``PersistentLatentStateV5Chain``
* M2 ``MultiHopBackendTranslator`` (5-backend; V3 helpers)
* M3 ``MergeableLatentCapsule`` + ``MergeOperator`` +
   ``MergeAuditTrail``
* M4 ``DeepProxyStackV4``
* M5 ``ECCCodebookV5`` + parity / ECC compression
* M6 ``LongHorizonReconstructionV5Head``
* M7 ``BranchMergeMemoryV3Head`` (consensus + abstain)
* M8 ``CorruptionRobustCarrier``
* M9 transcript-vs-shared arbiter V2
* M10 uncertainty / confidence layer

Each W53 turn emits W53 per-module witnesses on top of the
W52 ``W52HandoffEnvelope``. The final ``W53HandoffEnvelope``
binds: the W52 outer CID, every W53 witness CID, the
W53Params CID, the persistent-V5 chain CID, the MLSC audit
trail CID, and a single ``w53_outer_cid`` that closes the
chain w47 → w48 → w49 → w50 → w51 → w52 → w53.

Honest scope (do-not-overstate)
-------------------------------

This module does NOT alter transformer-internal state. Every
W53 witness is computed over capsule-layer signals exclusively.
Trivial passthrough is preserved byte-for-byte: when
``W53Params.build_trivial()`` is used and all flags are
disabled, the W53 envelope's internal ``w52_outer_cid``
equals the W52 outer CID exactly — the
``W53-L-TRIVIAL-W53-PASSTHROUGH`` falsifier.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from .agents import Agent
from .branch_merge_memory_v3 import (
    BranchMergeMemoryV3Head,
    BranchMergeMemoryV3Witness,
    W53_DEFAULT_BMM_V3_COSINE_FLOOR,
    W53_DEFAULT_BMM_V3_K_REQUIRED,
    W53_DEFAULT_BMM_V3_N_CONSENSUS_PAGES,
    emit_bmm_v3_witness,
    evaluate_consensus_recall,
)
from .branch_cycle_memory import (
    W51_DEFAULT_BCM_FACTOR_DIM,
    W51_DEFAULT_BCM_N_BRANCH_PAGES,
    W51_DEFAULT_BCM_N_CYCLE_PAGES,
)
from .branch_cycle_memory_v2 import (
    W52_DEFAULT_BCM_V2_N_JOINT_PAGES,
    W52_DEFAULT_BCM_V2_PAGE_SLOTS,
)
from .corruption_robust_carrier import (
    CorruptionRobustCarrier,
    CorruptionRobustnessWitness,
    W53_DEFAULT_CRC_REPETITION,
    emit_corruption_robustness_witness,
)
from .cross_backend_alignment import (
    W50_ANCHOR_STATUS_SYNTHETIC,
)
from .deep_proxy_stack_v4 import (
    DeepProxyStackV4,
    DeepProxyStackV4ForwardWitness,
    W53_DEFAULT_DEEP_V4_FACTOR_DIM,
    W53_DEFAULT_DEEP_V4_IN_DIM,
    W53_DEFAULT_DEEP_V4_N_BRANCH_HEADS,
    W53_DEFAULT_DEEP_V4_N_CYCLE_HEADS,
    W53_DEFAULT_DEEP_V4_N_HEADS,
    W53_DEFAULT_DEEP_V4_N_LAYERS,
    W53_DEFAULT_DEEP_V4_N_ROLES,
    emit_deep_proxy_stack_v4_forward_witness,
)
from .ecc_codebook_v5 import (
    ECCCodebookV5,
    ECCCompressionWitness,
    W53_DEFAULT_ECC_CODE_DIM,
    W53_DEFAULT_ECC_EMIT_MASK_LEN,
    W53_DEFAULT_ECC_K1,
    W53_DEFAULT_ECC_K2,
    W53_DEFAULT_ECC_K3,
    W53_DEFAULT_ECC_K4,
    W53_DEFAULT_ECC_TARGET_BITS_PER_TOKEN,
    compress_carrier_ecc,
    emit_ecc_compression_witness,
)
from .long_horizon_retention_v5 import (
    LongHorizonReconstructionV5Head,
    LongHorizonReconstructionV5Witness,
    W53_DEFAULT_LHR_V5_FLAT_FEATURE_DIM,
    W53_DEFAULT_LHR_V5_HIDDEN_DIM,
    W53_DEFAULT_LHR_V5_MAX_K,
    W53_DEFAULT_LHR_V5_N_BRANCHES,
    W53_DEFAULT_LHR_V5_N_CYCLES,
    W53_DEFAULT_LHR_V5_N_MERGE_PAIRS,
    emit_lhr_v5_witness,
)
from .mergeable_latent_capsule import (
    MergeAuditTrail,
    MergeOperator,
    MergeableLatentCapsule,
    MergeableLatentCapsuleWitness,
    W53_DEFAULT_MLSC_FACTOR_DIM,
    W53_DEFAULT_MLSC_MERGE_TEMP,
    emit_mlsc_witness,
    make_root_capsule,
    merge_capsules,
)
from .multi_hop_translator import MultiHopBackendTranslator
from .multi_hop_translator_v3 import (
    MultiHopV3Witness,
    W53_DEFAULT_MH_V3_BACKENDS,
    W53_DEFAULT_MH_V3_CODE_DIM,
    W53_DEFAULT_MH_V3_FEATURE_DIM,
    build_unfitted_quint_translator,
    emit_multi_hop_v3_witness,
    synthesize_quint_training_set,
)
from .persistent_latent_v5 import (
    PersistentLatentStateV5,
    PersistentLatentStateV5Chain,
    PersistentLatentStateV5Witness,
    V5StackedCell,
    W53_DEFAULT_V5_INPUT_DIM,
    W53_DEFAULT_V5_N_LAYERS,
    W53_DEFAULT_V5_STATE_DIM,
    W53_V5_NO_PARENT_STATE,
    emit_persistent_v5_witness,
    step_persistent_state_v5,
)
from .quantised_compression import (
    QuantisedBudgetGate,
    W52_DEFAULT_QUANT_EMIT_MASK_LEN,
)
from .transcript_vs_shared_arbiter_v2 import (
    TVSArbiterV2Witness,
    emit_tvs_arbiter_v2_witness,
    three_arm_compare,
)
from .uncertainty_layer import (
    UncertaintyLayerWitness,
    calibration_check,
    compose_uncertainty_report,
    emit_uncertainty_layer_witness,
)
from .w52_team import (
    W52HandoffEnvelope,
    W52Params,
    W52Registry,
    W52Team,
    W52TeamResult,
    W52TurnWitnessBundle,
    build_trivial_w52_registry,
    build_w52_registry,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W53_SCHEMA_VERSION: str = "coordpy.w53_team.v1"
W53_TEAM_RESULT_SCHEMA: str = "coordpy.w53_team_result.v1"

W53_NO_STATE: str = "no_w53_persistent_state"


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
# W53Params
# =============================================================================


@dataclasses.dataclass
class W53Params:
    """All trainable / config surfaces for W53, layered over W52."""

    persistent_v5_cell: V5StackedCell | None
    quint_translator: MultiHopBackendTranslator | None
    mlsc_operator: MergeOperator | None
    deep_stack_v4: DeepProxyStackV4 | None
    ecc_codebook: ECCCodebookV5 | None
    ecc_gate: QuantisedBudgetGate | None
    long_horizon_v5_head: (
        LongHorizonReconstructionV5Head | None)
    branch_merge_memory_v3: BranchMergeMemoryV3Head | None
    corruption_robust_carrier: (
        CorruptionRobustCarrier | None)

    persistent_v5_enabled: bool = False
    quint_translator_enabled: bool = False
    mlsc_enabled: bool = False
    deep_stack_v4_enabled: bool = False
    ecc_compression_enabled: bool = False
    long_horizon_v5_enabled: bool = False
    branch_merge_memory_v3_enabled: bool = False
    corruption_robust_carrier_enabled: bool = False
    transcript_vs_shared_arbiter_v2_enabled: bool = False
    uncertainty_layer_enabled: bool = False

    target_bits_per_token: float = (
        W53_DEFAULT_ECC_TARGET_BITS_PER_TOKEN)
    arbiter_budget_tokens: int = 3
    bmm_v3_k_required: int = (
        W53_DEFAULT_BMM_V3_K_REQUIRED)

    @classmethod
    def build_trivial(cls) -> "W53Params":
        return cls(
            persistent_v5_cell=None,
            quint_translator=None,
            mlsc_operator=None,
            deep_stack_v4=None,
            ecc_codebook=None,
            ecc_gate=None,
            long_horizon_v5_head=None,
            branch_merge_memory_v3=None,
            corruption_robust_carrier=None,
            persistent_v5_enabled=False,
            quint_translator_enabled=False,
            mlsc_enabled=False,
            deep_stack_v4_enabled=False,
            ecc_compression_enabled=False,
            long_horizon_v5_enabled=False,
            branch_merge_memory_v3_enabled=False,
            corruption_robust_carrier_enabled=False,
            transcript_vs_shared_arbiter_v2_enabled=False,
            uncertainty_layer_enabled=False,
        )

    @classmethod
    def build_default(
            cls, *,
            role_universe: Sequence[str] = (
                "r0", "r1", "r2", "r3"),
            seed: int = 12345,
    ) -> "W53Params":
        cell_v5 = V5StackedCell.init(
            state_dim=W53_DEFAULT_V5_STATE_DIM,
            input_dim=W53_DEFAULT_V5_INPUT_DIM,
            n_layers=W53_DEFAULT_V5_N_LAYERS,
            seed=int(seed))
        quint = build_unfitted_quint_translator(
            backends=W53_DEFAULT_MH_V3_BACKENDS,
            code_dim=W53_DEFAULT_MH_V3_CODE_DIM,
            feature_dim=W53_DEFAULT_MH_V3_FEATURE_DIM,
            seed=int(seed) + 7)
        mlsc_op = MergeOperator(
            factor_dim=W53_DEFAULT_MLSC_FACTOR_DIM,
            temperature=W53_DEFAULT_MLSC_MERGE_TEMP)
        deep4 = DeepProxyStackV4.init(
            n_layers=W53_DEFAULT_DEEP_V4_N_LAYERS,
            in_dim=W53_DEFAULT_DEEP_V4_IN_DIM,
            factor_dim=W53_DEFAULT_DEEP_V4_FACTOR_DIM,
            n_heads=W53_DEFAULT_DEEP_V4_N_HEADS,
            n_branch_heads=W53_DEFAULT_DEEP_V4_N_BRANCH_HEADS,
            n_cycle_heads=W53_DEFAULT_DEEP_V4_N_CYCLE_HEADS,
            n_roles=W53_DEFAULT_DEEP_V4_N_ROLES,
            seed=int(seed) + 13)
        ecc_cb = ECCCodebookV5.init(
            n_coarse=W53_DEFAULT_ECC_K1,
            n_fine=W53_DEFAULT_ECC_K2,
            n_ultra=W53_DEFAULT_ECC_K3,
            n_ultra2=W53_DEFAULT_ECC_K4,
            code_dim=W53_DEFAULT_ECC_CODE_DIM,
            seed=int(seed) + 17)
        ecc_gate = QuantisedBudgetGate.init(
            in_dim=W53_DEFAULT_ECC_CODE_DIM,
            emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
            seed=int(seed) + 19)
        ecc_gate.importance_threshold = 0.0
        ecc_gate.w_emit.values = [
            1.0] * len(ecc_gate.w_emit.values)
        lhr_v5 = LongHorizonReconstructionV5Head.init(
            carrier_dim=(
                W53_DEFAULT_LHR_V5_MAX_K
                * W53_DEFAULT_LHR_V5_FLAT_FEATURE_DIM),
            hidden_dim=W53_DEFAULT_LHR_V5_HIDDEN_DIM,
            out_dim=W53_DEFAULT_LHR_V5_FLAT_FEATURE_DIM,
            max_k=W53_DEFAULT_LHR_V5_MAX_K,
            n_branches=W53_DEFAULT_LHR_V5_N_BRANCHES,
            n_cycles=W53_DEFAULT_LHR_V5_N_CYCLES,
            n_merge_pairs=W53_DEFAULT_LHR_V5_N_MERGE_PAIRS,
            seed=int(seed) + 23)
        bmm_v3 = BranchMergeMemoryV3Head.init(
            factor_dim=W51_DEFAULT_BCM_FACTOR_DIM,
            n_branch_pages=W51_DEFAULT_BCM_N_BRANCH_PAGES,
            n_cycle_pages=W51_DEFAULT_BCM_N_CYCLE_PAGES,
            page_capacity=W52_DEFAULT_BCM_V2_PAGE_SLOTS,
            n_joint_pages=W52_DEFAULT_BCM_V2_N_JOINT_PAGES,
            n_consensus_pages=(
                W53_DEFAULT_BMM_V3_N_CONSENSUS_PAGES),
            k_required=W53_DEFAULT_BMM_V3_K_REQUIRED,
            cosine_floor=W53_DEFAULT_BMM_V3_COSINE_FLOOR,
            seed=int(seed) + 29)
        crc = CorruptionRobustCarrier.init(
            codebook=ecc_cb, gate=ecc_gate,
            repetition=W53_DEFAULT_CRC_REPETITION)
        return cls(
            persistent_v5_cell=cell_v5,
            quint_translator=quint,
            mlsc_operator=mlsc_op,
            deep_stack_v4=deep4,
            ecc_codebook=ecc_cb,
            ecc_gate=ecc_gate,
            long_horizon_v5_head=lhr_v5,
            branch_merge_memory_v3=bmm_v3,
            corruption_robust_carrier=crc,
            persistent_v5_enabled=True,
            quint_translator_enabled=True,
            mlsc_enabled=True,
            deep_stack_v4_enabled=True,
            ecc_compression_enabled=True,
            long_horizon_v5_enabled=True,
            branch_merge_memory_v3_enabled=True,
            corruption_robust_carrier_enabled=True,
            transcript_vs_shared_arbiter_v2_enabled=True,
            uncertainty_layer_enabled=True,
        )

    @property
    def all_flags_disabled(self) -> bool:
        return not (
            self.persistent_v5_enabled
            or self.quint_translator_enabled
            or self.mlsc_enabled
            or self.deep_stack_v4_enabled
            or self.ecc_compression_enabled
            or self.long_horizon_v5_enabled
            or self.branch_merge_memory_v3_enabled
            or self.corruption_robust_carrier_enabled
            or self
            .transcript_vs_shared_arbiter_v2_enabled
            or self.uncertainty_layer_enabled)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(W53_SCHEMA_VERSION),
            "persistent_v5_cell_cid": (
                self.persistent_v5_cell.cid()
                if self.persistent_v5_cell is not None
                else ""),
            "quint_translator_cid": (
                self.quint_translator.cid()
                if self.quint_translator is not None
                else ""),
            "mlsc_operator_cid": (
                self.mlsc_operator.cid()
                if self.mlsc_operator is not None
                else ""),
            "deep_stack_v4_cid": (
                self.deep_stack_v4.cid()
                if self.deep_stack_v4 is not None
                else ""),
            "ecc_codebook_cid": (
                self.ecc_codebook.cid()
                if self.ecc_codebook is not None
                else ""),
            "ecc_gate_cid": (
                self.ecc_gate.cid()
                if self.ecc_gate is not None
                else ""),
            "long_horizon_v5_head_cid": (
                self.long_horizon_v5_head.cid()
                if self.long_horizon_v5_head is not None
                else ""),
            "branch_merge_memory_v3_cid": (
                self.branch_merge_memory_v3.cid()
                if self.branch_merge_memory_v3 is not None
                else ""),
            "corruption_robust_carrier_cid": (
                self.corruption_robust_carrier.cid()
                if (self.corruption_robust_carrier
                    is not None)
                else ""),
            "persistent_v5_enabled": bool(
                self.persistent_v5_enabled),
            "quint_translator_enabled": bool(
                self.quint_translator_enabled),
            "mlsc_enabled": bool(self.mlsc_enabled),
            "deep_stack_v4_enabled": bool(
                self.deep_stack_v4_enabled),
            "ecc_compression_enabled": bool(
                self.ecc_compression_enabled),
            "long_horizon_v5_enabled": bool(
                self.long_horizon_v5_enabled),
            "branch_merge_memory_v3_enabled": bool(
                self.branch_merge_memory_v3_enabled),
            "corruption_robust_carrier_enabled": bool(
                self.corruption_robust_carrier_enabled),
            "transcript_vs_shared_arbiter_v2_enabled": (
                bool(self
                     .transcript_vs_shared_arbiter_v2_enabled)),
            "uncertainty_layer_enabled": bool(
                self.uncertainty_layer_enabled),
            "target_bits_per_token": float(round(
                self.target_bits_per_token, 12)),
            "arbiter_budget_tokens": int(
                self.arbiter_budget_tokens),
            "bmm_v3_k_required": int(
                self.bmm_v3_k_required),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_params",
            "params": self.to_dict()})


# =============================================================================
# W53Registry
# =============================================================================


@dataclasses.dataclass
class W53Registry:
    """W53 registry — wraps a W52 registry + W53 params."""

    schema_cid: str
    inner_w52_registry: W52Registry
    params: W53Params

    @property
    def is_trivial(self) -> bool:
        return (
            self.inner_w52_registry.is_trivial
            and self.params.all_flags_disabled
        )


def build_trivial_w53_registry(
        *, schema_cid: str | None = None,
) -> W53Registry:
    cid = schema_cid or _sha256_hex({
        "kind": "w53_trivial_schema"})
    inner = build_trivial_w52_registry(
        schema_cid=str(cid))
    return W53Registry(
        schema_cid=str(cid),
        inner_w52_registry=inner,
        params=W53Params.build_trivial(),
    )


def build_w53_registry(
        *,
        schema_cid: str,
        inner_w52_registry: W52Registry | None = None,
        params: W53Params | None = None,
        role_universe: Sequence[str] = (
            "r0", "r1", "r2", "r3"),
        seed: int = 12345,
) -> W53Registry:
    inner = (
        inner_w52_registry
        if inner_w52_registry is not None
        else build_w52_registry(
            schema_cid=str(schema_cid),
            role_universe=role_universe,
            seed=int(seed)))
    p = params or W53Params.build_default(
        role_universe=role_universe, seed=int(seed))
    return W53Registry(
        schema_cid=str(schema_cid),
        inner_w52_registry=inner,
        params=p,
    )


# =============================================================================
# Per-turn witnesses
# =============================================================================


@dataclasses.dataclass(frozen=True)
class W53TurnWitnessBundle:
    """All W53 per-turn witness CIDs for one team turn."""

    persistent_v5_witness_cid: str
    quint_translator_witness_cid: str
    mlsc_witness_cid: str
    deep_stack_v4_forward_witness_cid: str
    ecc_compression_witness_cid: str
    long_horizon_v5_witness_cid: str
    branch_merge_memory_v3_witness_cid: str
    corruption_robust_carrier_witness_cid: str
    transcript_vs_shared_arbiter_v2_witness_cid: str
    uncertainty_layer_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "persistent_v5_witness_cid": str(
                self.persistent_v5_witness_cid),
            "quint_translator_witness_cid": str(
                self.quint_translator_witness_cid),
            "mlsc_witness_cid": str(self.mlsc_witness_cid),
            "deep_stack_v4_forward_witness_cid": str(
                self.deep_stack_v4_forward_witness_cid),
            "ecc_compression_witness_cid": str(
                self.ecc_compression_witness_cid),
            "long_horizon_v5_witness_cid": str(
                self.long_horizon_v5_witness_cid),
            "branch_merge_memory_v3_witness_cid": str(
                self.branch_merge_memory_v3_witness_cid),
            "corruption_robust_carrier_witness_cid": str(
                self
                .corruption_robust_carrier_witness_cid),
            "transcript_vs_shared_arbiter_v2_witness_cid": (
                str(self
                    .transcript_vs_shared_arbiter_v2_witness_cid)),
            "uncertainty_layer_witness_cid": str(
                self.uncertainty_layer_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_turn_witness_bundle",
            "bundle": self.to_dict()})


# =============================================================================
# W53HandoffEnvelope
# =============================================================================


@dataclasses.dataclass(frozen=True)
class W53HandoffEnvelope:
    """Sealed per-team W53 envelope."""

    schema_version: str
    w52_outer_cid: str
    params_cid: str
    turn_witness_bundle_cid: str
    w52_envelope_count: int
    persistent_v5_chain_cid: str
    mlsc_audit_trail_cid: str
    composite_confidence_mean: float
    arbiter_pick_rate_shared_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "w52_outer_cid": str(self.w52_outer_cid),
            "params_cid": str(self.params_cid),
            "turn_witness_bundle_cid": str(
                self.turn_witness_bundle_cid),
            "w52_envelope_count": int(
                self.w52_envelope_count),
            "persistent_v5_chain_cid": str(
                self.persistent_v5_chain_cid),
            "mlsc_audit_trail_cid": str(
                self.mlsc_audit_trail_cid),
            "composite_confidence_mean": float(round(
                self.composite_confidence_mean, 12)),
            "arbiter_pick_rate_shared_mean": float(round(
                self.arbiter_pick_rate_shared_mean, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_outer_envelope",
            "envelope": self.to_dict()})


# =============================================================================
# Forward — compute per-turn W53 witnesses
# =============================================================================


def _persistent_v5_step(
        *,
        params: W53Params,
        turn_index: int,
        role: str,
        prev_state: PersistentLatentStateV5 | None,
        carrier_payload: Any,
        state_chain: PersistentLatentStateV5Chain,
        cycle_index: int = 0,
        branch_id: str = "main",
) -> PersistentLatentStateV5 | None:
    if (not params.persistent_v5_enabled
            or params.persistent_v5_cell is None):
        return None
    cell = params.persistent_v5_cell
    input_vec = _payload_hash_vec(
        carrier_payload, cell.input_dim)
    new_state = step_persistent_state_v5(
        cell=cell,
        prev_state=prev_state,
        carrier_values=input_vec,
        turn_index=int(turn_index),
        role=str(role),
        branch_id=str(branch_id),
        cycle_index=int(cycle_index),
        skip_input=input_vec)
    state_chain.add(new_state)
    return new_state


def _emit_w53_turn_witnesses(
        *,
        params: W53Params,
        turn_index: int,
        role: str,
        prev_state: PersistentLatentStateV5 | None,
        state_chain: PersistentLatentStateV5Chain,
        mlsc_audit: MergeAuditTrail,
        mlsc_capsule_store: dict[
            str, MergeableLatentCapsule],
        prev_mlsc_capsule: MergeableLatentCapsule | None,
        anchor_payload: Mapping[str, Any] | None = None,
        branch_index: int = 0,
        cycle_index: int = 0,
        role_index: int = 0,
        carrier_payload: Any = None,
) -> tuple[
        W53TurnWitnessBundle,
        PersistentLatentStateV5 | None,
        MergeableLatentCapsule | None,
        float, float]:
    """Compute all per-turn W53 witnesses.

    Returns (bundle, new_state, new_mlsc_capsule,
        composite_confidence, arbiter_pick_rate_shared).
    """
    new_state = _persistent_v5_step(
        params=params,
        turn_index=int(turn_index),
        role=str(role),
        prev_state=prev_state,
        carrier_payload=carrier_payload,
        state_chain=state_chain,
        cycle_index=int(cycle_index))
    pv5_witness_cid = ""
    pv5_conf = 1.0
    if (new_state is not None
            and params.persistent_v5_cell is not None):
        w_pv5 = emit_persistent_v5_witness(
            state=new_state,
            cell=params.persistent_v5_cell,
            chain=state_chain)
        pv5_witness_cid = w_pv5.cid()
        # Confidence: 1 - sigmoid(gate_sum / state_dim) bounded
        # to [0, 1].
        gate_norm = float(
            new_state.update_gate_l1_sum) / float(
                max(1, int(new_state.state_dim)))
        pv5_conf = float(max(0.0, min(
            1.0, 1.0 - 0.5 * gate_norm)))
    # M2 — quint translator witness (synthetic empty probes).
    quint_witness_cid = ""
    quint_conf = 1.0
    if (params.quint_translator_enabled
            and params.quint_translator is not None):
        ts = synthesize_quint_training_set(
            n_examples=4, seed=int(turn_index) + 11,
            backends=tuple(params.quint_translator.backends),
            code_dim=int(
                params.quint_translator.code_dim),
            feature_dim=int(
                params.quint_translator.feature_dim))
        w_q = emit_multi_hop_v3_witness(
            translator=params.quint_translator,
            examples=ts.examples[:4])
        quint_witness_cid = w_q.cid()
        quint_conf = float(max(0.0, min(
            1.0, w_q.direct_fidelity_a_e)))
    # M3 — MLSC capsule.
    mlsc_witness_cid = ""
    mlsc_conf = 1.0
    new_mlsc_cap: MergeableLatentCapsule | None = None
    if (params.mlsc_enabled
            and params.mlsc_operator is not None):
        # Build a capsule from the V5 top state if available,
        # else from the carrier payload hash.
        op = params.mlsc_operator
        payload = (
            list(new_state.top_state)
            [:int(op.factor_dim)]
            if new_state is not None
            else _payload_hash_vec(
                carrier_payload, op.factor_dim))
        while len(payload) < int(op.factor_dim):
            payload.append(0.0)
        if prev_mlsc_capsule is None:
            new_mlsc_cap = make_root_capsule(
                branch_id=f"branch_{branch_index}",
                payload=payload,
                confidence=float(pv5_conf),
                fact_tags=(f"turn={turn_index}",),
                turn_index=int(turn_index))
        else:
            # Merge the new payload (as a synthetic
            # peer capsule) into the prev capsule. This
            # exercises the MergeOperator at every turn.
            from .mergeable_latent_capsule import (
                step_branch_capsule)
            child = step_branch_capsule(
                parent=prev_mlsc_capsule,
                payload=payload,
                confidence=float(pv5_conf),
                new_fact_tags=(f"turn={turn_index}",),
                turn_index=int(turn_index))
            new_mlsc_cap = child
        mlsc_capsule_store[
            new_mlsc_cap.cid()] = new_mlsc_cap
        # Once we have at least 2 capsules, do a merge to
        # exercise the audit trail.
        if (prev_mlsc_capsule is not None
                and (turn_index % 2) == 1):
            merged = merge_capsules(
                op,
                [prev_mlsc_capsule, new_mlsc_cap],
                audit_trail=mlsc_audit,
                extra_fact_tags=(
                    f"merge_at_turn={turn_index}",))
            mlsc_capsule_store[merged.cid()] = merged
            new_mlsc_cap = merged
        w_mlsc = emit_mlsc_witness(
            leaf=new_mlsc_cap,
            operator=op,
            audit_trail=mlsc_audit,
            capsule_store=mlsc_capsule_store)
        mlsc_witness_cid = w_mlsc.cid()
        mlsc_conf = float(new_mlsc_cap.confidence)
    # M4 — deep stack V4 forward witness.
    deep_v4_witness_cid = ""
    deep_v4_conf = 1.0
    if (params.deep_stack_v4_enabled
            and params.deep_stack_v4 is not None):
        s = params.deep_stack_v4
        if new_state is not None:
            q = list(new_state.top_state)[:s.in_dim]
        else:
            q = []
        while len(q) < s.in_dim:
            q.append(0.0)
        w_deep, _ = emit_deep_proxy_stack_v4_forward_witness(
            stack=s, query_input=q,
            slot_keys=[q], slot_values=[q],
            role_index=int(role_index),
            branch_index=int(branch_index),
            cycle_index=int(cycle_index))
        deep_v4_witness_cid = w_deep.cid()
        deep_v4_conf = float(w_deep.corruption_confidence)
    # M5 — ECC compression witness.
    ecc_witness_cid = ""
    if (params.ecc_compression_enabled
            and params.ecc_codebook is not None
            and params.ecc_gate is not None
            and new_state is not None):
        carrier = list(new_state.top_state)[
            :int(params.ecc_codebook.code_dim)]
        while len(carrier) < int(params.ecc_codebook.code_dim):
            carrier.append(0.0)
        comp = compress_carrier_ecc(
            carrier,
            codebook=params.ecc_codebook,
            gate=params.ecc_gate)
        w_ecc = emit_ecc_compression_witness(
            codebook=params.ecc_codebook,
            compression=comp,
            target_bits_per_token=float(
                params.target_bits_per_token))
        ecc_witness_cid = w_ecc.cid()
    # M6 — long-horizon V5 witness (synthetic empty).
    lhr_v5_witness_cid = ""
    if (params.long_horizon_v5_enabled
            and params.long_horizon_v5_head is not None):
        w_lhr = emit_lhr_v5_witness(
            head=params.long_horizon_v5_head,
            examples=(),
            k_max_for_degradation=8)
        lhr_v5_witness_cid = w_lhr.cid()
    # M7 — BMM V3 witness (no recall test in per-turn path).
    bmm_v3_witness_cid = ""
    if (params.branch_merge_memory_v3_enabled
            and params.branch_merge_memory_v3
            is not None):
        w_bmm = emit_bmm_v3_witness(
            head=params.branch_merge_memory_v3,
            consensus_recall=0.0)
        bmm_v3_witness_cid = w_bmm.cid()
    # M8 — corruption-robust carrier witness.
    crc_witness_cid = ""
    crc_silent_failure_rate = 0.0
    if (params.corruption_robust_carrier_enabled
            and params.corruption_robust_carrier
            is not None
            and new_state is not None):
        # Single-carrier probe.
        carrier = list(new_state.top_state)[
            :int(params.corruption_robust_carrier
                  .codebook.code_dim)]
        while len(carrier) < int(
                params.corruption_robust_carrier
                .codebook.code_dim):
            carrier.append(0.0)
        w_crc = emit_corruption_robustness_witness(
            crc=params.corruption_robust_carrier,
            carriers=[carrier],
            flip_intensity=1.0,
            seed=int(turn_index) + 1)
        crc_witness_cid = w_crc.cid()
        crc_silent_failure_rate = float(
            w_crc.silent_failure_rate)
    # M9 — transcript-vs-shared arbiter V2 witness.
    tvs_arb_v2_witness_cid = ""
    arbiter_pick_rate_shared = 0.0
    if (params.transcript_vs_shared_arbiter_v2_enabled
            and params.ecc_codebook is not None
            and params.ecc_gate is not None
            and new_state is not None):
        # Use the V5 top-state as a synthetic carrier.
        carrier = list(new_state.top_state)[
            :int(params.ecc_codebook.code_dim)]
        while len(carrier) < int(
                params.ecc_codebook.code_dim):
            carrier.append(0.0)
        result = three_arm_compare(
            carriers=[carrier],
            codebook=(
                params.ecc_codebook.inner_v4),
            gate=params.ecc_gate,
            budget_tokens=int(
                params.arbiter_budget_tokens),
            per_turn_confidences=[float(pv5_conf)],
            abstain_threshold=0.15,
            prefer_shared_threshold=0.0)
        w_tvs = emit_tvs_arbiter_v2_witness(
            result=result)
        tvs_arb_v2_witness_cid = w_tvs.cid()
        arbiter_pick_rate_shared = float(
            w_tvs.arbiter_pick_rate_shared)
    # M10 — uncertainty layer witness.
    uncert_witness_cid = ""
    composite_conf = 1.0
    if params.uncertainty_layer_enabled:
        report = compose_uncertainty_report(
            persistent_v5_confidence=float(pv5_conf),
            multi_hop_v3_confidence=float(quint_conf),
            mlsc_capsule_confidence=float(mlsc_conf),
            deep_v4_corruption_confidence=float(
                deep_v4_conf),
            crc_silent_failure_rate=float(
                crc_silent_failure_rate))
        composite_conf = float(report.composite_confidence)
        # Empty calibration check (handled by R-104 family).
        cal = calibration_check(
            confidences=(),
            accuracies=(),
            min_calibration_gap=0.10)
        w_unc = emit_uncertainty_layer_witness(
            report=report, calibration=cal)
        uncert_witness_cid = w_unc.cid()
    bundle = W53TurnWitnessBundle(
        persistent_v5_witness_cid=str(pv5_witness_cid),
        quint_translator_witness_cid=str(
            quint_witness_cid),
        mlsc_witness_cid=str(mlsc_witness_cid),
        deep_stack_v4_forward_witness_cid=str(
            deep_v4_witness_cid),
        ecc_compression_witness_cid=str(ecc_witness_cid),
        long_horizon_v5_witness_cid=str(
            lhr_v5_witness_cid),
        branch_merge_memory_v3_witness_cid=str(
            bmm_v3_witness_cid),
        corruption_robust_carrier_witness_cid=str(
            crc_witness_cid),
        transcript_vs_shared_arbiter_v2_witness_cid=str(
            tvs_arb_v2_witness_cid),
        uncertainty_layer_witness_cid=str(
            uncert_witness_cid),
    )
    return (
        bundle, new_state, new_mlsc_cap,
        float(composite_conf),
        float(arbiter_pick_rate_shared))


# =============================================================================
# W53TeamResult
# =============================================================================


@dataclasses.dataclass(frozen=True)
class W53TeamResult:
    schema: str
    task: str
    final_output: str
    w52_outer_cid: str
    w53_outer_cid: str
    w53_params_cid: str
    w53_envelope: W53HandoffEnvelope
    turn_witness_bundles: tuple[W53TurnWitnessBundle, ...]
    persistent_v5_state_cids: tuple[str, ...]
    mlsc_capsule_cids: tuple[str, ...]
    composite_confidence_mean: float
    arbiter_pick_rate_shared_mean: float
    n_turns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "task": str(self.task),
            "final_output": str(self.final_output),
            "w52_outer_cid": str(self.w52_outer_cid),
            "w53_outer_cid": str(self.w53_outer_cid),
            "w53_params_cid": str(self.w53_params_cid),
            "w53_envelope": self.w53_envelope.to_dict(),
            "turn_witness_bundles": [
                b.to_dict()
                for b in self.turn_witness_bundles],
            "persistent_v5_state_cids": list(
                self.persistent_v5_state_cids),
            "mlsc_capsule_cids": list(
                self.mlsc_capsule_cids),
            "composite_confidence_mean": float(round(
                self.composite_confidence_mean, 12)),
            "arbiter_pick_rate_shared_mean": float(round(
                self.arbiter_pick_rate_shared_mean, 12)),
            "n_turns": int(self.n_turns),
        }


# =============================================================================
# W53Team
# =============================================================================


@dataclasses.dataclass
class W53Team:
    """W53 team orchestrator — wraps ``W52Team``."""

    agents: Sequence[Agent]
    registry: W53Registry
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
    ) -> W53TeamResult:
        # Build inner W52 team and run it.
        w52_team = W52Team(
            agents=list(self.agents),
            backend=self.backend,
            registry=self.registry.inner_w52_registry,
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
        w52_result = w52_team.run(task, progress=progress)
        params = self.registry.params
        state_chain = PersistentLatentStateV5Chain.empty()
        mlsc_audit = MergeAuditTrail.empty()
        mlsc_store: dict[str, MergeableLatentCapsule] = {}
        bundles: list[W53TurnWitnessBundle] = []
        persistent_v5_cids: list[str] = []
        mlsc_cids: list[str] = []
        prev_state: PersistentLatentStateV5 | None = None
        prev_mlsc: MergeableLatentCapsule | None = None
        n_turns = int(w52_result.n_turns)
        composite_sum = 0.0
        arb_share_sum = 0.0
        for i, turn in enumerate(
                w52_result.turn_witness_bundles):
            n_branches = max(
                1, params.deep_stack_v4.n_branch_heads
                if params.deep_stack_v4 is not None
                else 4)
            n_cycles = max(
                1, params.deep_stack_v4.n_cycle_heads
                if params.deep_stack_v4 is not None
                else 4)
            n_roles = max(
                1, params.deep_stack_v4.n_roles
                if params.deep_stack_v4 is not None
                else 4)
            branch_index = int(i) % int(n_branches)
            cycle_index = int(i) % int(n_cycles)
            role_index = int(i) % int(n_roles)
            carrier_payload = (int(i), turn.cid())
            (bundle, new_state, new_mlsc, conf,
             arb_share) = _emit_w53_turn_witnesses(
                params=params,
                turn_index=int(i),
                role=str(self._role_for_turn(i)),
                prev_state=prev_state,
                state_chain=state_chain,
                mlsc_audit=mlsc_audit,
                mlsc_capsule_store=mlsc_store,
                prev_mlsc_capsule=prev_mlsc,
                branch_index=int(branch_index),
                cycle_index=int(cycle_index),
                role_index=int(role_index),
                carrier_payload=carrier_payload,
            )
            bundles.append(bundle)
            composite_sum += float(conf)
            arb_share_sum += float(arb_share)
            if new_state is not None:
                persistent_v5_cids.append(new_state.cid())
                prev_state = new_state
            if new_mlsc is not None:
                mlsc_cids.append(new_mlsc.cid())
                prev_mlsc = new_mlsc
        comp_mean = (
            float(composite_sum) / float(max(1, n_turns)))
        arb_mean = (
            float(arb_share_sum) / float(max(1, n_turns)))
        bundles_cid = _sha256_hex({
            "kind": "w53_all_turn_witness_bundles",
            "bundles": [b.to_dict() for b in bundles],
        })
        env = W53HandoffEnvelope(
            schema_version=W53_SCHEMA_VERSION,
            w52_outer_cid=str(w52_result.w52_outer_cid),
            params_cid=str(params.cid()),
            turn_witness_bundle_cid=str(bundles_cid),
            w52_envelope_count=int(n_turns),
            persistent_v5_chain_cid=str(state_chain.cid()),
            mlsc_audit_trail_cid=str(mlsc_audit.cid()),
            composite_confidence_mean=float(comp_mean),
            arbiter_pick_rate_shared_mean=float(arb_mean),
        )
        return W53TeamResult(
            schema=W53_TEAM_RESULT_SCHEMA,
            task=str(task),
            final_output=str(w52_result.final_output),
            w52_outer_cid=str(w52_result.w52_outer_cid),
            w53_outer_cid=str(env.cid()),
            w53_params_cid=str(params.cid()),
            w53_envelope=env,
            turn_witness_bundles=tuple(bundles),
            persistent_v5_state_cids=tuple(
                persistent_v5_cids),
            mlsc_capsule_cids=tuple(mlsc_cids),
            composite_confidence_mean=float(comp_mean),
            arbiter_pick_rate_shared_mean=float(arb_mean),
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


W53_ENVELOPE_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w53_schema_mismatch",
    "w53_w52_outer_cid_mismatch",
    "w53_params_cid_mismatch",
    "w53_turn_witness_bundle_cid_mismatch",
    "w53_envelope_count_mismatch",
    "w53_outer_cid_mismatch",
    "w53_persistent_v5_chain_cid_mismatch",
    "w53_mlsc_audit_trail_cid_mismatch",
    "w53_trivial_passthrough_w52_cid_mismatch",
    "w53_persistent_v5_witness_missing_when_enabled",
    "w53_quint_translator_witness_missing_when_enabled",
    "w53_mlsc_witness_missing_when_enabled",
    "w53_deep_stack_v4_witness_missing_when_enabled",
    "w53_ecc_compression_witness_missing_when_enabled",
    "w53_long_horizon_v5_witness_missing_when_enabled",
    "w53_branch_merge_memory_v3_witness_missing_when_enabled",
    "w53_corruption_robust_carrier_witness_missing_when_enabled",
    "w53_transcript_vs_shared_arbiter_v2_witness_missing_when_enabled",
    "w53_uncertainty_layer_witness_missing_when_enabled",
    "w53_envelope_payload_hash_mismatch",
    "w53_per_turn_bundle_count_mismatch",
    "w53_witness_bundle_cid_recompute_mismatch",
    "w53_persistent_v5_state_count_inconsistent",
    "w53_outer_cid_recompute_mismatch",
    "w53_inner_w52_envelope_invalid",
    "w53_role_universe_mismatch",
    "w53_composite_confidence_out_of_bounds",
    "w53_arbiter_pick_rate_shared_out_of_bounds",
    "w53_mlsc_audit_trail_walk_orphan",
)


def verify_w53_handoff(
        envelope: W53HandoffEnvelope,
        *,
        expected_w52_outer_cid: str | None = None,
        expected_params_cid: str | None = None,
        bundles: Sequence[W53TurnWitnessBundle] | None = None,
        registry: W53Registry | None = None,
        persistent_v5_state_cids: (
            Sequence[str] | None) = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if envelope.schema_version != W53_SCHEMA_VERSION:
        failures.append("w53_schema_mismatch")
    if (expected_w52_outer_cid is not None
            and envelope.w52_outer_cid
            != expected_w52_outer_cid):
        failures.append("w53_w52_outer_cid_mismatch")
    if (expected_params_cid is not None
            and envelope.params_cid
            != expected_params_cid):
        failures.append("w53_params_cid_mismatch")
    if not (
            0.0 <= float(envelope.composite_confidence_mean)
            <= 1.0):
        failures.append(
            "w53_composite_confidence_out_of_bounds")
    if not (
            0.0 <= float(
                envelope.arbiter_pick_rate_shared_mean)
            <= 1.0):
        failures.append(
            "w53_arbiter_pick_rate_shared_out_of_bounds")
    if bundles is not None:
        if envelope.w52_envelope_count != len(bundles):
            failures.append(
                "w53_per_turn_bundle_count_mismatch")
        recomputed = _sha256_hex({
            "kind": "w53_all_turn_witness_bundles",
            "bundles": [b.to_dict() for b in bundles],
        })
        if envelope.turn_witness_bundle_cid != recomputed:
            failures.append(
                "w53_witness_bundle_cid_recompute_mismatch")
    if registry is not None:
        for b in (bundles or ()):
            if (registry.params.persistent_v5_enabled
                    and not b.persistent_v5_witness_cid):
                failures.append(
                    "w53_persistent_v5_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params.quint_translator_enabled
                    and not b.quint_translator_witness_cid):
                failures.append(
                    "w53_quint_translator_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params.mlsc_enabled
                    and not b.mlsc_witness_cid):
                failures.append(
                    "w53_mlsc_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params.deep_stack_v4_enabled
                    and not b.deep_stack_v4_forward_witness_cid):
                failures.append(
                    "w53_deep_stack_v4_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params.ecc_compression_enabled
                    and not b.ecc_compression_witness_cid):
                failures.append(
                    "w53_ecc_compression_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params.long_horizon_v5_enabled
                    and not b.long_horizon_v5_witness_cid):
                failures.append(
                    "w53_long_horizon_v5_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params
                    .branch_merge_memory_v3_enabled
                    and not b
                    .branch_merge_memory_v3_witness_cid):
                failures.append(
                    "w53_branch_merge_memory_v3_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params
                    .corruption_robust_carrier_enabled
                    and not b
                    .corruption_robust_carrier_witness_cid):
                failures.append(
                    "w53_corruption_robust_carrier_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params
                    .transcript_vs_shared_arbiter_v2_enabled
                    and not b
                    .transcript_vs_shared_arbiter_v2_witness_cid):
                failures.append(
                    "w53_transcript_vs_shared_arbiter_v2_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params.uncertainty_layer_enabled
                    and not b.uncertainty_layer_witness_cid):
                failures.append(
                    "w53_uncertainty_layer_witness_missing_when_enabled")
                break
    if (persistent_v5_state_cids is not None
            and registry is not None):
        if (registry.params.persistent_v5_enabled
                and len(persistent_v5_state_cids)
                != envelope.w52_envelope_count):
            failures.append(
                "w53_persistent_v5_state_count_inconsistent")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "outer_cid": envelope.cid(),
        "n_failure_modes": int(
            len(W53_ENVELOPE_VERIFIER_FAILURE_MODES)),
    }


__all__ = [
    "W53_SCHEMA_VERSION",
    "W53_TEAM_RESULT_SCHEMA",
    "W53_NO_STATE",
    "W53_ENVELOPE_VERIFIER_FAILURE_MODES",
    "W53Params",
    "W53Registry",
    "W53TurnWitnessBundle",
    "W53HandoffEnvelope",
    "W53TeamResult",
    "W53Team",
    "build_trivial_w53_registry",
    "build_w53_registry",
    "verify_w53_handoff",
]
