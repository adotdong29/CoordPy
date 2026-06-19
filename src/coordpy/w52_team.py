"""W52 — Quantised Persistent Multi-Hop Latent Coordination (QPMHLC).

The ``W52Team`` orchestrator composes the W51 ``W51Team`` with
the eight M1..M8 W52 mechanism modules:

* M1 ``V4StackedCell`` + ``PersistentLatentStateV4Chain``
* M2 ``MultiHopBackendTranslator``
* M3 ``DeepProxyStackV3``
* M4 ``QuantisedCodebookV4`` + ``QuantisedBudgetGate``
* M5 ``LongHorizonReconstructionV4Head``
* M6 ``BranchCycleMemoryV2Head``
* M7 ``RoleGraphMixer``
* M8 transcript-vs-shared-state comparator

Each W52 turn emits W52 per-module witnesses on top of the
W51 ``W51HandoffEnvelope``. The final ``W52HandoffEnvelope``
binds: the W51 outer CID, every W52 witness CID, the
W52Params CID, the persistent-V4 chain CID, the multi-hop
translator anchor payload CID, and a single
``w52_outer_cid`` that closes the chain
w47 → w48 → w49 → w50 → w51 → w52.

Honest scope (do-not-overstate)
-------------------------------

This module does NOT alter transformer-internal state. Every
W52 witness is computed over capsule-layer signals exclusively.
Trivial passthrough is preserved byte-for-byte: when
``W52Params.build_trivial()`` is used and all flags are
disabled, the W52 envelope's internal ``w51_outer_cid``
equals the W51 outer CID exactly — the
``W52-L-TRIVIAL-W52-PASSTHROUGH`` falsifier.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from .agents import Agent
from .branch_cycle_memory_v2 import (
    BranchCycleMemoryV2Head,
    BranchCycleMemoryV2Witness,
    BranchCycleMemoryV2TrainingTrace,
    W52_DEFAULT_BCM_V2_N_JOINT_PAGES,
    W52_DEFAULT_BCM_V2_PAGE_SLOTS,
    emit_branch_cycle_memory_v2_witness,
)
from .branch_cycle_memory import (
    W51_DEFAULT_BCM_FACTOR_DIM,
    W51_DEFAULT_BCM_N_BRANCH_PAGES,
    W51_DEFAULT_BCM_N_CYCLE_PAGES,
)
from .cross_backend_alignment import (
    W50_ANCHOR_STATUS_SYNTHETIC,
)
from .deep_proxy_stack_v3 import (
    DeepProxyStackV3,
    DeepProxyStackV3ForwardWitness,
    W52_DEFAULT_DEEP_V3_FACTOR_DIM,
    W52_DEFAULT_DEEP_V3_IN_DIM,
    W52_DEFAULT_DEEP_V3_N_BRANCH_HEADS,
    W52_DEFAULT_DEEP_V3_N_CYCLE_HEADS,
    W52_DEFAULT_DEEP_V3_N_HEADS,
    W52_DEFAULT_DEEP_V3_N_LAYERS,
    W52_DEFAULT_DEEP_V3_N_ROLES,
    emit_deep_proxy_stack_v3_forward_witness,
)
from .llm_backend import LLMBackend
from .long_horizon_retention_v4 import (
    LongHorizonReconstructionV4Head,
    LongHorizonReconstructionV4Witness,
    LongHorizonV4TrainingTrace,
    W52_DEFAULT_LHR_V4_FLAT_FEATURE_DIM,
    W52_DEFAULT_LHR_V4_HIDDEN_DIM,
    W52_DEFAULT_LHR_V4_MAX_K,
    W52_DEFAULT_LHR_V4_N_BRANCHES,
    W52_DEFAULT_LHR_V4_N_CYCLES,
    emit_long_horizon_v4_witness,
)
from .multi_hop_translator import (
    MultiHopBackendTranslator,
    MultiHopTrainingTrace,
    MultiHopTranslatorWitness,
    W52_DEFAULT_MH_CODE_DIM,
    W52_DEFAULT_MH_FEATURE_DIM,
    build_unfitted_multi_hop_translator,
    emit_multi_hop_translator_witness,
    run_multi_hop_realism_anchor_probe,
    synthesize_multi_hop_training_set,
)
from .persistent_latent_v4 import (
    PersistentLatentStateV4,
    PersistentLatentStateV4Chain,
    PersistentLatentStateV4Witness,
    V4StackedCell,
    W52_DEFAULT_V4_INPUT_DIM,
    W52_DEFAULT_V4_N_LAYERS,
    W52_DEFAULT_V4_STATE_DIM,
    W52_V4_NO_PARENT_STATE,
    emit_persistent_v4_witness,
    step_persistent_state_v4,
)
from .quantised_compression import (
    CrammingWitnessV4,
    QuantisedBudgetGate,
    QuantisedCodebookV4,
    QuantisedCompressionTrainingTrace,
    QuantisedCompressionWitness,
    W52_DEFAULT_QUANT_BITS_PAYLOAD_LEN,
    W52_DEFAULT_QUANT_CODE_DIM,
    W52_DEFAULT_QUANT_EMIT_MASK_LEN,
    W52_DEFAULT_QUANT_K1,
    W52_DEFAULT_QUANT_K2,
    W52_DEFAULT_QUANT_K3,
    W52_DEFAULT_QUANT_TARGET_BITS_PER_TOKEN,
    compress_carrier_quantised,
    emit_cramming_witness_v4,
    emit_quantised_compression_witness,
    probe_quantised_degradation_curve,
)
from .role_graph_transfer import (
    RoleGraphMixer,
    RoleGraphTrainingTrace,
    RoleGraphWitness,
    W52_DEFAULT_RG_STATE_DIM,
    build_unfitted_role_graph_mixer,
    emit_role_graph_witness,
)
from .transcript_vs_shared_state import (
    TranscriptVsSharedWitness,
    emit_transcript_vs_shared_witness,
)
from .w51_team import (
    W51HandoffEnvelope,
    W51Params,
    W51Registry,
    W51Team,
    W51TeamResult,
    W51TurnWitnessBundle,
    build_trivial_w51_registry,
    build_w51_registry,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W52_SCHEMA_VERSION: str = "coordpy.w52_team.v1"
W52_TEAM_RESULT_SCHEMA: str = "coordpy.w52_team_result.v1"

W52_NO_STATE: str = "no_w52_persistent_state"


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


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    import math
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        ai = float(a[i])
        bi = float(b[i])
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na <= 1e-30 or nb <= 1e-30:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))


# =============================================================================
# W52Params
# =============================================================================


@dataclasses.dataclass
class W52Params:
    """All trainable / config surfaces for W52, layered over W51."""

    persistent_v4_cell: V4StackedCell | None
    multi_hop_translator: MultiHopBackendTranslator | None
    deep_stack_v3: DeepProxyStackV3 | None
    quantised_codebook: QuantisedCodebookV4 | None
    quantised_gate: QuantisedBudgetGate | None
    long_horizon_v4_head: LongHorizonReconstructionV4Head | None
    branch_cycle_memory_v2: BranchCycleMemoryV2Head | None
    role_graph_mixer: RoleGraphMixer | None

    persistent_v4_enabled: bool = False
    multi_hop_enabled: bool = False
    deep_stack_v3_enabled: bool = False
    quantised_compression_enabled: bool = False
    long_horizon_v4_enabled: bool = False
    branch_cycle_memory_v2_enabled: bool = False
    role_graph_enabled: bool = False
    transcript_vs_shared_enabled: bool = False

    target_bits_per_token: float = (
        W52_DEFAULT_QUANT_TARGET_BITS_PER_TOKEN)
    transcript_comparator_budget_tokens: int = 3

    @classmethod
    def build_trivial(cls) -> "W52Params":
        return cls(
            persistent_v4_cell=None,
            multi_hop_translator=None,
            deep_stack_v3=None,
            quantised_codebook=None,
            quantised_gate=None,
            long_horizon_v4_head=None,
            branch_cycle_memory_v2=None,
            role_graph_mixer=None,
            persistent_v4_enabled=False,
            multi_hop_enabled=False,
            deep_stack_v3_enabled=False,
            quantised_compression_enabled=False,
            long_horizon_v4_enabled=False,
            branch_cycle_memory_v2_enabled=False,
            role_graph_enabled=False,
            transcript_vs_shared_enabled=False,
        )

    @classmethod
    def build_default(
            cls, *,
            role_universe: Sequence[str] = (
                "r0", "r1", "r2", "r3"),
            seed: int = 12345,
    ) -> "W52Params":
        cell_v4 = V4StackedCell.init(
            state_dim=W52_DEFAULT_V4_STATE_DIM,
            input_dim=W52_DEFAULT_V4_INPUT_DIM,
            n_layers=W52_DEFAULT_V4_N_LAYERS,
            seed=int(seed))
        mh = build_unfitted_multi_hop_translator(
            backends=("A", "B", "C", "D"),
            code_dim=W52_DEFAULT_MH_CODE_DIM,
            feature_dim=W52_DEFAULT_MH_FEATURE_DIM,
            seed=int(seed) + 7)
        deep = DeepProxyStackV3.init(
            n_layers=W52_DEFAULT_DEEP_V3_N_LAYERS,
            in_dim=W52_DEFAULT_DEEP_V3_IN_DIM,
            factor_dim=W52_DEFAULT_DEEP_V3_FACTOR_DIM,
            n_heads=W52_DEFAULT_DEEP_V3_N_HEADS,
            n_branch_heads=W52_DEFAULT_DEEP_V3_N_BRANCH_HEADS,
            n_cycle_heads=W52_DEFAULT_DEEP_V3_N_CYCLE_HEADS,
            n_roles=W52_DEFAULT_DEEP_V3_N_ROLES,
            seed=int(seed) + 13)
        cb = QuantisedCodebookV4.init(
            n_coarse=W52_DEFAULT_QUANT_K1,
            n_fine=W52_DEFAULT_QUANT_K2,
            n_ultra=W52_DEFAULT_QUANT_K3,
            code_dim=W52_DEFAULT_QUANT_CODE_DIM,
            seed=int(seed) + 19)
        gate = QuantisedBudgetGate.init(
            in_dim=W52_DEFAULT_QUANT_CODE_DIM,
            emit_mask_len=W52_DEFAULT_QUANT_EMIT_MASK_LEN,
            seed=int(seed) + 23)
        lhr_v4 = LongHorizonReconstructionV4Head.init(
            carrier_dim=(
                W52_DEFAULT_LHR_V4_MAX_K
                * W52_DEFAULT_LHR_V4_FLAT_FEATURE_DIM),
            hidden_dim=W52_DEFAULT_LHR_V4_HIDDEN_DIM,
            out_dim=W52_DEFAULT_LHR_V4_FLAT_FEATURE_DIM,
            max_k=W52_DEFAULT_LHR_V4_MAX_K,
            n_branches=W52_DEFAULT_LHR_V4_N_BRANCHES,
            n_cycles=W52_DEFAULT_LHR_V4_N_CYCLES,
            seed=int(seed) + 29)
        bcm_v2 = BranchCycleMemoryV2Head.init(
            factor_dim=W51_DEFAULT_BCM_FACTOR_DIM,
            n_branch_pages=W51_DEFAULT_BCM_N_BRANCH_PAGES,
            n_cycle_pages=W51_DEFAULT_BCM_N_CYCLE_PAGES,
            page_capacity=W52_DEFAULT_BCM_V2_PAGE_SLOTS,
            n_joint_pages=W52_DEFAULT_BCM_V2_N_JOINT_PAGES,
            seed=int(seed) + 31)
        rg = build_unfitted_role_graph_mixer(
            role_universe=role_universe,
            state_dim=W52_DEFAULT_RG_STATE_DIM,
            seed=int(seed) + 37)
        return cls(
            persistent_v4_cell=cell_v4,
            multi_hop_translator=mh,
            deep_stack_v3=deep,
            quantised_codebook=cb,
            quantised_gate=gate,
            long_horizon_v4_head=lhr_v4,
            branch_cycle_memory_v2=bcm_v2,
            role_graph_mixer=rg,
            persistent_v4_enabled=True,
            multi_hop_enabled=True,
            deep_stack_v3_enabled=True,
            quantised_compression_enabled=True,
            long_horizon_v4_enabled=True,
            branch_cycle_memory_v2_enabled=True,
            role_graph_enabled=True,
            transcript_vs_shared_enabled=True,
        )

    @property
    def all_flags_disabled(self) -> bool:
        return not (
            self.persistent_v4_enabled
            or self.multi_hop_enabled
            or self.deep_stack_v3_enabled
            or self.quantised_compression_enabled
            or self.long_horizon_v4_enabled
            or self.branch_cycle_memory_v2_enabled
            or self.role_graph_enabled
            or self.transcript_vs_shared_enabled)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(W52_SCHEMA_VERSION),
            "persistent_v4_cell_cid": (
                self.persistent_v4_cell.cid()
                if self.persistent_v4_cell is not None else ""),
            "multi_hop_translator_cid": (
                self.multi_hop_translator.cid()
                if self.multi_hop_translator is not None
                else ""),
            "deep_stack_v3_cid": (
                self.deep_stack_v3.cid()
                if self.deep_stack_v3 is not None else ""),
            "quantised_codebook_cid": (
                self.quantised_codebook.cid()
                if self.quantised_codebook is not None
                else ""),
            "quantised_gate_cid": (
                self.quantised_gate.cid()
                if self.quantised_gate is not None else ""),
            "long_horizon_v4_head_cid": (
                self.long_horizon_v4_head.cid()
                if self.long_horizon_v4_head is not None
                else ""),
            "branch_cycle_memory_v2_cid": (
                self.branch_cycle_memory_v2.cid()
                if self.branch_cycle_memory_v2 is not None
                else ""),
            "role_graph_mixer_cid": (
                self.role_graph_mixer.cid()
                if self.role_graph_mixer is not None else ""),
            "persistent_v4_enabled": bool(
                self.persistent_v4_enabled),
            "multi_hop_enabled": bool(self.multi_hop_enabled),
            "deep_stack_v3_enabled": bool(
                self.deep_stack_v3_enabled),
            "quantised_compression_enabled": bool(
                self.quantised_compression_enabled),
            "long_horizon_v4_enabled": bool(
                self.long_horizon_v4_enabled),
            "branch_cycle_memory_v2_enabled": bool(
                self.branch_cycle_memory_v2_enabled),
            "role_graph_enabled": bool(self.role_graph_enabled),
            "transcript_vs_shared_enabled": bool(
                self.transcript_vs_shared_enabled),
            "target_bits_per_token": float(round(
                self.target_bits_per_token, 12)),
            "transcript_comparator_budget_tokens": int(
                self.transcript_comparator_budget_tokens),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_params",
            "params": self.to_dict()})


# =============================================================================
# W52Registry
# =============================================================================


@dataclasses.dataclass
class W52Registry:
    """W52 registry — wraps a W51 registry + W52 params."""

    schema_cid: str
    inner_w51_registry: W51Registry
    params: W52Params

    @property
    def is_trivial(self) -> bool:
        return (
            self.inner_w51_registry.is_trivial
            and self.params.all_flags_disabled
        )


def build_trivial_w52_registry(
        *, schema_cid: str | None = None,
) -> W52Registry:
    """A W52 registry that reduces to W51 trivial byte-for-byte."""
    cid = schema_cid or _sha256_hex({
        "kind": "w52_trivial_schema"})
    inner = build_trivial_w51_registry(schema_cid=str(cid))
    return W52Registry(
        schema_cid=str(cid),
        inner_w51_registry=inner,
        params=W52Params.build_trivial(),
    )


def build_w52_registry(
        *,
        schema_cid: str,
        inner_w51_registry: W51Registry | None = None,
        params: W52Params | None = None,
        role_universe: Sequence[str] = ("r0", "r1", "r2", "r3"),
        seed: int = 12345,
) -> W52Registry:
    inner = (
        inner_w51_registry
        if inner_w51_registry is not None
        else build_w51_registry(
            schema_cid=str(schema_cid),
            role_universe=role_universe,
            seed=int(seed)))
    p = params or W52Params.build_default(
        role_universe=role_universe, seed=int(seed))
    return W52Registry(
        schema_cid=str(schema_cid),
        inner_w51_registry=inner,
        params=p,
    )


# =============================================================================
# Per-turn witnesses
# =============================================================================


@dataclasses.dataclass(frozen=True)
class W52TurnWitnessBundle:
    """All W52 per-turn witness CIDs for one team turn."""

    persistent_v4_witness_cid: str
    multi_hop_witness_cid: str
    deep_stack_v3_forward_witness_cid: str
    quantised_compression_witness_cid: str
    cramming_witness_v4_cid: str
    long_horizon_v4_witness_cid: str
    branch_cycle_memory_v2_witness_cid: str
    role_graph_witness_cid: str
    transcript_vs_shared_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "persistent_v4_witness_cid": str(
                self.persistent_v4_witness_cid),
            "multi_hop_witness_cid": str(
                self.multi_hop_witness_cid),
            "deep_stack_v3_forward_witness_cid": str(
                self.deep_stack_v3_forward_witness_cid),
            "quantised_compression_witness_cid": str(
                self.quantised_compression_witness_cid),
            "cramming_witness_v4_cid": str(
                self.cramming_witness_v4_cid),
            "long_horizon_v4_witness_cid": str(
                self.long_horizon_v4_witness_cid),
            "branch_cycle_memory_v2_witness_cid": str(
                self.branch_cycle_memory_v2_witness_cid),
            "role_graph_witness_cid": str(
                self.role_graph_witness_cid),
            "transcript_vs_shared_witness_cid": str(
                self.transcript_vs_shared_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_turn_witness_bundle",
            "bundle": self.to_dict()})


# =============================================================================
# W52HandoffEnvelope
# =============================================================================


@dataclasses.dataclass(frozen=True)
class W52HandoffEnvelope:
    """Sealed per-team W52 envelope."""

    schema_version: str
    w51_outer_cid: str
    params_cid: str
    turn_witness_bundle_cid: str
    w51_envelope_count: int
    persistent_v4_chain_cid: str
    multi_hop_anchor_payload_cid: str
    multi_hop_anchor_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "w51_outer_cid": str(self.w51_outer_cid),
            "params_cid": str(self.params_cid),
            "turn_witness_bundle_cid": str(
                self.turn_witness_bundle_cid),
            "w51_envelope_count": int(self.w51_envelope_count),
            "persistent_v4_chain_cid": str(
                self.persistent_v4_chain_cid),
            "multi_hop_anchor_payload_cid": str(
                self.multi_hop_anchor_payload_cid),
            "multi_hop_anchor_status": str(
                self.multi_hop_anchor_status),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_outer_envelope",
            "envelope": self.to_dict()})


# =============================================================================
# Forward — compute per-turn W52 witnesses
# =============================================================================


def _persistent_v4_step(
        *,
        params: W52Params,
        turn_index: int,
        role: str,
        prev_state: PersistentLatentStateV4 | None,
        carrier_payload: Any,
        state_chain: PersistentLatentStateV4Chain,
        cycle_index: int = 0,
) -> PersistentLatentStateV4 | None:
    if (not params.persistent_v4_enabled
            or params.persistent_v4_cell is None):
        return None
    payload_hex = hashlib.sha256(
        _canonical_bytes(carrier_payload or "")).hexdigest()
    input_dim = params.persistent_v4_cell.input_dim
    input_vec: list[float] = []
    for i in range(input_dim):
        nb = payload_hex[
            (i * 2) % len(payload_hex):
            (i * 2) % len(payload_hex) + 2]
        if not nb:
            nb = "00"
        v = (int(nb, 16) / 127.5) - 1.0
        input_vec.append(float(round(v, 12)))
    new_state = step_persistent_state_v4(
        cell=params.persistent_v4_cell,
        prev_state=prev_state,
        carrier_values=input_vec,
        turn_index=int(turn_index),
        role=str(role),
        cycle_index=int(cycle_index),
        skip_input=input_vec)
    state_chain.add(new_state)
    return new_state


def _emit_w52_turn_witnesses(
        *,
        params: W52Params,
        turn_index: int,
        role: str,
        prev_state: PersistentLatentStateV4 | None,
        state_chain: PersistentLatentStateV4Chain,
        anchor_payload: Mapping[str, Any] | None = None,
        branch_index: int = 0,
        cycle_index: int = 0,
        role_index: int = 0,
        carrier_payload: Any = None,
) -> tuple[W52TurnWitnessBundle,
           PersistentLatentStateV4 | None]:
    """Compute all per-turn W52 witnesses."""
    new_state = _persistent_v4_step(
        params=params,
        turn_index=int(turn_index),
        role=str(role),
        prev_state=prev_state,
        carrier_payload=carrier_payload,
        state_chain=state_chain,
        cycle_index=int(cycle_index))
    pv4_witness_cid = ""
    if (new_state is not None
            and params.persistent_v4_cell is not None):
        w_pv4 = emit_persistent_v4_witness(
            state=new_state,
            cell=params.persistent_v4_cell,
            chain=state_chain)
        pv4_witness_cid = w_pv4.cid()
    # M2 — multi-hop translator (synthetic empty witness).
    mh_witness_cid = ""
    if (params.multi_hop_enabled
            and params.multi_hop_translator is not None):
        ts = synthesize_multi_hop_training_set(
            n_examples=4, seed=int(turn_index) + 1,
            backends=params.multi_hop_translator.backends,
            feature_dim=int(
                params.multi_hop_translator.feature_dim),
            code_dim=int(
                params.multi_hop_translator.code_dim))
        empty_trace = MultiHopTrainingTrace(
            seed=0, n_steps=0, final_loss=0.0,
            final_grad_norm=0.0,
            loss_head=(), loss_tail=(),
            training_set_cid=str(ts.cid()),
            final_translator_cid=str(
                params.multi_hop_translator.cid()),
            transitivity_weight=0.0,
            diverged=False)
        w_mh = emit_multi_hop_translator_witness(
            translator=params.multi_hop_translator,
            training_trace=empty_trace,
            probes=ts.examples[:4],
            anchor_payload=(
                dict(anchor_payload) if anchor_payload else None))
        mh_witness_cid = w_mh.cid()
    # M3 — deep stack V3 forward witness.
    deep_v3_witness_cid = ""
    if (params.deep_stack_v3_enabled
            and params.deep_stack_v3 is not None):
        s = params.deep_stack_v3
        if new_state is not None:
            q = list(new_state.top_state)[:s.in_dim]
        else:
            q = []
        while len(q) < s.in_dim:
            q.append(0.0)
        w_deep, _ = emit_deep_proxy_stack_v3_forward_witness(
            stack=s, query_input=q,
            slot_keys=[q], slot_values=[q],
            role_index=int(role_index),
            branch_index=int(branch_index),
            cycle_index=int(cycle_index))
        deep_v3_witness_cid = w_deep.cid()
    # M4 — quantised compression
    quant_witness_cid = ""
    cram_v4_cid = ""
    if (params.quantised_compression_enabled
            and params.quantised_codebook is not None
            and params.quantised_gate is not None
            and new_state is not None):
        carrier = list(new_state.top_state)[
            :params.quantised_codebook.code_dim]
        while (len(carrier)
               < params.quantised_codebook.code_dim):
            carrier.append(0.0)
        compression = compress_carrier_quantised(
            carrier,
            codebook=params.quantised_codebook,
            gate=params.quantised_gate,
            bits_payload_len=(
                W52_DEFAULT_QUANT_BITS_PAYLOAD_LEN))
        cw4 = emit_cramming_witness_v4(
            compression=compression)
        cram_v4_cid = cw4.cid()
        decoded = params.quantised_codebook.decode(
            coarse=compression.coarse_code,
            fine=compression.fine_code,
            ultra=compression.ultra_code)
        retention = _cosine(carrier, decoded)
        dc = probe_quantised_degradation_curve(
            carrier,
            codebook=params.quantised_codebook,
            gate=params.quantised_gate)
        empty_trace = QuantisedCompressionTrainingTrace(
            seed=0, n_steps=0, final_loss=0.0,
            final_grad_norm=0.0,
            loss_head=(), loss_tail=(),
            training_set_cid="",
            final_codebook_cid=str(
                params.quantised_codebook.cid()),
            final_gate_cid=str(
                params.quantised_gate.cid()),
            diverged=False)
        qw = emit_quantised_compression_witness(
            codebook=params.quantised_codebook,
            gate=params.quantised_gate,
            training_trace=empty_trace,
            cramming=cw4,
            retention_cosine=float(retention),
            target_bits_per_token=float(
                params.target_bits_per_token),
            degradation_curve=dc)
        quant_witness_cid = qw.cid()
    # M5 — long-horizon V4 (synthetic empty witness)
    lhr_v4_witness_cid = ""
    if (params.long_horizon_v4_enabled
            and params.long_horizon_v4_head is not None):
        empty_trace = LongHorizonV4TrainingTrace(
            seed=0, n_steps=0, final_loss=0.0,
            final_grad_norm=0.0,
            loss_head=(), loss_tail=(),
            training_set_cid="",
            final_head_cid=str(
                params.long_horizon_v4_head.cid()),
            diverged=False)
        w_lhr = emit_long_horizon_v4_witness(
            head=params.long_horizon_v4_head,
            training_trace=empty_trace,
            examples=())
        lhr_v4_witness_cid = w_lhr.cid()
    # M6 — branch/cycle memory V2 (synthetic empty witness)
    bcm_v2_witness_cid = ""
    if (params.branch_cycle_memory_v2_enabled
            and params.branch_cycle_memory_v2 is not None):
        empty_trace = BranchCycleMemoryV2TrainingTrace(
            seed=0, n_steps=0, final_loss=0.0,
            final_grad_norm=0.0,
            loss_head=(), loss_tail=(),
            training_set_cid="",
            final_head_cid=str(
                params.branch_cycle_memory_v2.cid()),
            diverged=False)
        w_bcm = emit_branch_cycle_memory_v2_witness(
            head=params.branch_cycle_memory_v2,
            training_trace=empty_trace,
            examples=())
        bcm_v2_witness_cid = w_bcm.cid()
    # M7 — role-graph mixer (synthetic empty witness)
    rg_witness_cid = ""
    if (params.role_graph_enabled
            and params.role_graph_mixer is not None):
        empty_trace = RoleGraphTrainingTrace(
            seed=0, n_steps=0, final_loss=0.0,
            final_grad_norm=0.0,
            loss_head=(), loss_tail=(),
            training_set_cid="",
            final_mixer_cid=str(
                params.role_graph_mixer.cid()),
            diverged=False)
        w_rg = emit_role_graph_witness(
            mixer=params.role_graph_mixer,
            training_trace=empty_trace,
            examples=())
        rg_witness_cid = w_rg.cid()
    # M8 — transcript-vs-shared comparator (uses new_state carrier)
    tvs_witness_cid = ""
    if (params.transcript_vs_shared_enabled
            and params.quantised_codebook is not None
            and params.quantised_gate is not None
            and new_state is not None):
        carrier = list(new_state.top_state)[
            :params.quantised_codebook.code_dim]
        while (len(carrier)
               < params.quantised_codebook.code_dim):
            carrier.append(0.0)
        w_tvs = emit_transcript_vs_shared_witness(
            carriers=[carrier],
            codebook=params.quantised_codebook,
            gate=params.quantised_gate,
            budget_tokens=int(
                params.transcript_comparator_budget_tokens))
        tvs_witness_cid = w_tvs.cid()
    bundle = W52TurnWitnessBundle(
        persistent_v4_witness_cid=str(pv4_witness_cid),
        multi_hop_witness_cid=str(mh_witness_cid),
        deep_stack_v3_forward_witness_cid=str(
            deep_v3_witness_cid),
        quantised_compression_witness_cid=str(
            quant_witness_cid),
        cramming_witness_v4_cid=str(cram_v4_cid),
        long_horizon_v4_witness_cid=str(lhr_v4_witness_cid),
        branch_cycle_memory_v2_witness_cid=str(
            bcm_v2_witness_cid),
        role_graph_witness_cid=str(rg_witness_cid),
        transcript_vs_shared_witness_cid=str(tvs_witness_cid),
    )
    return bundle, new_state


# =============================================================================
# W52TeamResult
# =============================================================================


@dataclasses.dataclass(frozen=True)
class W52TeamResult:
    schema: str
    task: str
    final_output: str
    w51_outer_cid: str
    w52_outer_cid: str
    w52_params_cid: str
    w52_envelope: W52HandoffEnvelope
    turn_witness_bundles: tuple[W52TurnWitnessBundle, ...]
    persistent_v4_state_cids: tuple[str, ...]
    multi_hop_anchor_status: str
    n_turns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "task": str(self.task),
            "final_output": str(self.final_output),
            "w51_outer_cid": str(self.w51_outer_cid),
            "w52_outer_cid": str(self.w52_outer_cid),
            "w52_params_cid": str(self.w52_params_cid),
            "w52_envelope": self.w52_envelope.to_dict(),
            "turn_witness_bundles": [
                b.to_dict() for b in self.turn_witness_bundles],
            "persistent_v4_state_cids": list(
                self.persistent_v4_state_cids),
            "multi_hop_anchor_status": str(
                self.multi_hop_anchor_status),
            "n_turns": int(self.n_turns),
        }


# =============================================================================
# W52Team
# =============================================================================


@dataclasses.dataclass
class W52Team:
    """W52 team orchestrator — wraps ``W51Team``."""

    agents: Sequence[Agent]
    registry: W52Registry
    backend: Any = None
    team_instructions: str = ""
    max_visible_handoffs: int = 4
    quad_backend_a: LLMBackend | None = None
    quad_backend_b: LLMBackend | None = None
    quad_backend_c: LLMBackend | None = None
    quad_backend_d: LLMBackend | None = None
    quad_anchor_n_turns: int = 4

    def _run_quad_anchor(self) -> dict[str, Any]:
        return run_multi_hop_realism_anchor_probe(
            backend_a=self.quad_backend_a,
            backend_b=self.quad_backend_b,
            backend_c=self.quad_backend_c,
            backend_d=self.quad_backend_d,
            n_turns=int(self.quad_anchor_n_turns),
        )

    def run(
            self, task: str, *,
            progress: Callable[[Any], None] | None = None,
    ) -> W52TeamResult:
        w51_team = W51Team(
            agents=list(self.agents),
            backend=self.backend,
            registry=self.registry.inner_w51_registry,
            team_instructions=self.team_instructions,
            max_visible_handoffs=int(self.max_visible_handoffs),
        )
        w51_result = w51_team.run(task, progress=progress)
        state_chain = PersistentLatentStateV4Chain.empty()
        quad_anchor_payload = (
            self._run_quad_anchor()
            if self.registry.params.multi_hop_enabled
            else {
                "anchor_status": W50_ANCHOR_STATUS_SYNTHETIC,
                "skipped_ok": 1.0,
                "n_turns": 0,
                "direct_ab": 0.0,
                "direct_bc": 0.0,
                "direct_cd": 0.0,
                "direct_ad": 0.0,
                "chain_len3_a_b_c_d": 0.0,
                "transitivity_gap_len3": 0.0,
                "reason": "multi-hop disabled"})
        bundles: list[W52TurnWitnessBundle] = []
        persistent_v4_state_cids: list[str] = []
        prev_state: PersistentLatentStateV4 | None = None
        n_turns = int(w51_result.n_turns)
        for i, turn in enumerate(
                w51_result.turn_witness_bundles):
            # Default branch/cycle/role assignment.
            n_branches = 4
            n_cycles = 4
            n_roles = 4
            if (self.registry.params.deep_stack_v3
                    is not None):
                n_branches = max(
                    1,
                    self.registry.params.deep_stack_v3.n_branch_heads)
                n_cycles = max(
                    1,
                    self.registry.params.deep_stack_v3.n_cycle_heads)
                n_roles = max(
                    1, self.registry.params.deep_stack_v3.n_roles)
            branch_index = int(i) % int(n_branches)
            cycle_index = int(i) % int(n_cycles)
            role_index = int(i) % int(n_roles)
            carrier_payload = (int(i), turn.cid())
            bundle, new_state = _emit_w52_turn_witnesses(
                params=self.registry.params,
                turn_index=int(i),
                role=str(self._role_for_turn(i)),
                prev_state=prev_state,
                state_chain=state_chain,
                anchor_payload=quad_anchor_payload,
                branch_index=int(branch_index),
                cycle_index=int(cycle_index),
                role_index=int(role_index),
                carrier_payload=carrier_payload,
            )
            bundles.append(bundle)
            if new_state is not None:
                persistent_v4_state_cids.append(new_state.cid())
                prev_state = new_state
        bundles_payload = {
            "kind": "w52_all_turn_witness_bundles",
            "bundles": [b.to_dict() for b in bundles],
        }
        bundles_cid = _sha256_hex(bundles_payload)
        anchor_cid = _sha256_hex({
            "kind": "w52_multi_hop_anchor_payload",
            "payload": dict(quad_anchor_payload),
        })
        env = W52HandoffEnvelope(
            schema_version=W52_SCHEMA_VERSION,
            w51_outer_cid=str(w51_result.w51_outer_cid),
            params_cid=str(self.registry.params.cid()),
            turn_witness_bundle_cid=str(bundles_cid),
            w51_envelope_count=int(n_turns),
            persistent_v4_chain_cid=str(state_chain.cid()),
            multi_hop_anchor_payload_cid=str(anchor_cid),
            multi_hop_anchor_status=str(
                quad_anchor_payload.get(
                    "anchor_status",
                    W50_ANCHOR_STATUS_SYNTHETIC)),
        )
        return W52TeamResult(
            schema=W52_TEAM_RESULT_SCHEMA,
            task=str(task),
            final_output=str(w51_result.final_output),
            w51_outer_cid=str(w51_result.w51_outer_cid),
            w52_outer_cid=str(env.cid()),
            w52_params_cid=str(self.registry.params.cid()),
            w52_envelope=env,
            turn_witness_bundles=tuple(bundles),
            persistent_v4_state_cids=tuple(
                persistent_v4_state_cids),
            multi_hop_anchor_status=str(
                env.multi_hop_anchor_status),
            n_turns=int(n_turns),
        )

    def _role_for_turn(self, i: int) -> str:
        roles = [str(getattr(a, "role", f"r{i}"))
                 for a in self.agents]
        if not roles:
            return f"r{i}"
        return roles[i % len(roles)]


# =============================================================================
# Verifier
# =============================================================================


W52_ENVELOPE_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w52_schema_mismatch",
    "w52_w51_outer_cid_mismatch",
    "w52_params_cid_mismatch",
    "w52_turn_witness_bundle_cid_mismatch",
    "w52_envelope_count_mismatch",
    "w52_outer_cid_mismatch",
    "w52_anchor_status_invalid",
    "w52_persistent_v4_chain_cid_mismatch",
    "w52_multi_hop_anchor_payload_cid_mismatch",
    "w52_trivial_passthrough_w51_cid_mismatch",
    "w52_persistent_v4_witness_missing_when_enabled",
    "w52_multi_hop_witness_missing_when_enabled",
    "w52_deep_stack_v3_witness_missing_when_enabled",
    "w52_quantised_compression_witness_missing_when_enabled",
    "w52_long_horizon_v4_witness_missing_when_enabled",
    "w52_branch_cycle_memory_v2_witness_missing_when_enabled",
    "w52_cramming_witness_v4_missing_when_enabled",
    "w52_role_graph_witness_missing_when_enabled",
    "w52_transcript_vs_shared_witness_missing_when_enabled",
    "w52_envelope_payload_hash_mismatch",
    "w52_per_turn_bundle_count_mismatch",
    "w52_witness_bundle_cid_recompute_mismatch",
    "w52_persistent_v4_state_count_inconsistent",
    "w52_outer_cid_recompute_mismatch",
    "w52_inner_w51_envelope_invalid",
    "w52_role_universe_mismatch",
)


def verify_w52_handoff(
        envelope: W52HandoffEnvelope,
        *,
        expected_w51_outer_cid: str | None = None,
        expected_params_cid: str | None = None,
        bundles: Sequence[W52TurnWitnessBundle] | None = None,
        registry: W52Registry | None = None,
        persistent_v4_state_cids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Verify a sealed W52 envelope.

    Returns ``ok``, ``failures``, the outer CID, and the count of
    enumerated failure modes (26 at W52).
    """
    failures: list[str] = []
    if envelope.schema_version != W52_SCHEMA_VERSION:
        failures.append("w52_schema_mismatch")
    if (expected_w51_outer_cid is not None
            and envelope.w51_outer_cid
            != expected_w51_outer_cid):
        failures.append("w52_w51_outer_cid_mismatch")
    if (expected_params_cid is not None
            and envelope.params_cid != expected_params_cid):
        failures.append("w52_params_cid_mismatch")
    if envelope.multi_hop_anchor_status not in (
            "synthetic_only", "real_llm_anchor", "skipped"):
        failures.append("w52_anchor_status_invalid")
    if bundles is not None:
        if envelope.w51_envelope_count != len(bundles):
            failures.append("w52_per_turn_bundle_count_mismatch")
        recomputed = _sha256_hex({
            "kind": "w52_all_turn_witness_bundles",
            "bundles": [b.to_dict() for b in bundles],
        })
        if envelope.turn_witness_bundle_cid != recomputed:
            failures.append(
                "w52_witness_bundle_cid_recompute_mismatch")
    if registry is not None:
        for b in (bundles or ()):
            if (registry.params.persistent_v4_enabled
                    and not b.persistent_v4_witness_cid):
                failures.append(
                    "w52_persistent_v4_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params.multi_hop_enabled
                    and not b.multi_hop_witness_cid):
                failures.append(
                    "w52_multi_hop_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params.deep_stack_v3_enabled
                    and not b.deep_stack_v3_forward_witness_cid):
                failures.append(
                    "w52_deep_stack_v3_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params
                    .quantised_compression_enabled
                    and not b.quantised_compression_witness_cid):
                failures.append(
                    "w52_quantised_compression_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params
                    .quantised_compression_enabled
                    and not b.cramming_witness_v4_cid):
                failures.append(
                    "w52_cramming_witness_v4_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params.long_horizon_v4_enabled
                    and not b.long_horizon_v4_witness_cid):
                failures.append(
                    "w52_long_horizon_v4_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params.branch_cycle_memory_v2_enabled
                    and not b.branch_cycle_memory_v2_witness_cid):
                failures.append(
                    "w52_branch_cycle_memory_v2_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params.role_graph_enabled
                    and not b.role_graph_witness_cid):
                failures.append(
                    "w52_role_graph_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params.transcript_vs_shared_enabled
                    and not b.transcript_vs_shared_witness_cid):
                failures.append(
                    "w52_transcript_vs_shared_witness_missing_when_enabled")
                break
    if (persistent_v4_state_cids is not None
            and registry is not None):
        if (registry.params.persistent_v4_enabled
                and len(persistent_v4_state_cids)
                != envelope.w51_envelope_count):
            failures.append(
                "w52_persistent_v4_state_count_inconsistent")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "outer_cid": envelope.cid(),
        "n_failure_modes": int(
            len(W52_ENVELOPE_VERIFIER_FAILURE_MODES)),
    }


__all__ = [
    "W52_SCHEMA_VERSION",
    "W52_TEAM_RESULT_SCHEMA",
    "W52_NO_STATE",
    "W52_ENVELOPE_VERIFIER_FAILURE_MODES",
    "W52Params",
    "W52Registry",
    "W52TurnWitnessBundle",
    "W52HandoffEnvelope",
    "W52TeamResult",
    "W52Team",
    "build_trivial_w52_registry",
    "build_w52_registry",
    "verify_w52_handoff",
]
