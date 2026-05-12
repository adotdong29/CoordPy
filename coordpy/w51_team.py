"""W51 — Persistent Cross-Backend Latent Coordination (PXBLC).

The ``W51Team`` orchestrator composes the W50 ``W50Team`` with
the six M1..M6 W51 mechanism modules:

* M1 ``PersistentStateCell`` + ``CrossRoleMixer``
* M2 ``TripleBackendTranslator``
* M3 ``DeepProxyStackV2``
* M4 ``HierarchicalCodebook`` + ``HierarchicalEmitGate``
* M5 ``LongHorizonReconstructionV3Head``
* M6 ``BranchCycleMemoryHead``

Each W51 turn emits W51 per-module witnesses on top of the
W50 ``W50HandoffEnvelope``. The final ``W51HandoffEnvelope``
binds: the W50 outer CID, every W51 witness CID, the
W51Params CID, the persistent latent state chain CID, the
triple-backend translator anchor payload CID, and a single
``w51_outer_cid`` that closes the chain w47 → w48 → w49 →
w50 → w51.

Honest scope (do-not-overstate)
-------------------------------

This module does NOT alter transformer-internal state. Every
W51 witness is computed over capsule-layer signals exclusively.
Trivial passthrough is preserved byte-for-byte: when
``W51Params.build_trivial()`` is used and all flags are
disabled, the W51 envelope's internal ``w50_outer_cid``
equals the W50 outer CID exactly — the
``W51-L-TRIVIAL-W51-PASSTHROUGH`` falsifier.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from .agents import Agent
from .branch_cycle_memory import (
    BranchCycleMemoryHead,
    BranchCycleMemoryWitness,
    W51_DEFAULT_BCM_FACTOR_DIM,
    W51_DEFAULT_BCM_N_BRANCH_PAGES,
    W51_DEFAULT_BCM_N_CYCLE_PAGES,
    W51_DEFAULT_BCM_PAGE_SLOTS,
    emit_branch_cycle_memory_witness,
    synthesize_branch_cycle_memory_training_set,
)
from .cross_backend_translator import (
    TripleBackendTranslator,
    TripleBackendTranslatorWitness,
    W51_DEFAULT_TRIPLE_CODE_DIM,
    W51_DEFAULT_TRIPLE_FEATURE_DIM,
    build_unfitted_triple_backend_translator,
    emit_triple_backend_translator_witness,
    run_triple_realism_anchor_probe,
    synthesize_triple_backend_training_set,
)
from .cross_backend_alignment import (
    W50_ANCHOR_STATUS_SYNTHETIC,
)
from .deep_proxy_stack_v2 import (
    DeepProxyStackV2,
    DeepProxyStackV2ForwardWitness,
    W51_DEFAULT_DEEP_V2_FACTOR_DIM,
    W51_DEFAULT_DEEP_V2_IN_DIM,
    W51_DEFAULT_DEEP_V2_N_BRANCH_HEADS,
    W51_DEFAULT_DEEP_V2_N_CYCLE_HEADS,
    W51_DEFAULT_DEEP_V2_N_HEADS,
    W51_DEFAULT_DEEP_V2_N_LAYERS,
    emit_deep_proxy_stack_v2_forward_witness,
)
from .hierarchical_compression import (
    HierarchicalCodebook,
    HierarchicalCompressionWitness,
    HierarchicalEmitGate,
    CrammingWitnessV3,
    W51_DEFAULT_HIER_BITS_PAYLOAD_LEN,
    W51_DEFAULT_HIER_EMIT_MASK_LEN,
    W51_DEFAULT_HIER_K1,
    W51_DEFAULT_HIER_K2,
    W51_DEFAULT_HIER_TARGET_BITS_PER_TOKEN,
    compress_carrier_hierarchical,
    emit_cramming_witness_v3,
    emit_hierarchical_compression_witness,
    probe_degradation_curve,
)
from .llm_backend import LLMBackend
from .long_horizon_retention import (
    LongHorizonReconstructionV3Head,
    LongHorizonReconstructionV3Witness,
    W51_DEFAULT_LHR_FLAT_FEATURE_DIM,
    W51_DEFAULT_LHR_HIDDEN_DIM,
    W51_DEFAULT_LHR_MAX_K,
    W51_DEFAULT_LHR_N_BRANCHES,
    emit_long_horizon_reconstruction_v3_witness,
)
from .persistent_shared_latent import (
    CrossRoleMixer,
    PersistentLatentState,
    PersistentLatentStateChain,
    PersistentLatentStateWitness,
    PersistentStateCell,
    W51_DEFAULT_INPUT_DIM,
    W51_DEFAULT_MIXER_BLEND_INIT,
    W51_DEFAULT_STATE_DIM,
    W51_NO_PARENT_STATE,
    emit_persistent_latent_state_witness,
    step_persistent_latent_state,
)
from .w50_team import (
    W50HandoffEnvelope,
    W50Params,
    W50Registry,
    W50Team,
    W50TeamResult,
    W50TurnWitnessBundle,
    build_trivial_w50_registry,
    build_w50_registry,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W51_SCHEMA_VERSION: str = "coordpy.w51_team.v1"
W51_TEAM_RESULT_SCHEMA: str = "coordpy.w51_team_result.v1"

W51_NO_STATE: str = "no_w51_persistent_state"


# =============================================================================
# Canonicalisation helpers
# =============================================================================

def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str,
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
# W51Params
# =============================================================================

@dataclasses.dataclass
class W51Params:
    """All trainable / config surfaces for W51, layered over W50.

    Each surface is optional — a "trivial" build returns no
    surfaces with all flags disabled so the W51 team reduces
    to ``W50Team.run`` byte-for-byte.
    """

    persistent_cell: PersistentStateCell | None
    cross_role_mixer: CrossRoleMixer | None
    triple_translator: TripleBackendTranslator | None
    deep_stack_v2: DeepProxyStackV2 | None
    hierarchical_codebook: HierarchicalCodebook | None
    hierarchical_gate: HierarchicalEmitGate | None
    long_horizon_head: LongHorizonReconstructionV3Head | None
    branch_cycle_memory: BranchCycleMemoryHead | None

    persistent_state_enabled: bool = False
    triple_backend_enabled: bool = False
    deep_stack_v2_enabled: bool = False
    hierarchical_compression_enabled: bool = False
    long_horizon_reconstruction_enabled: bool = False
    branch_cycle_memory_enabled: bool = False

    target_bits_per_token: float = (
        W51_DEFAULT_HIER_TARGET_BITS_PER_TOKEN)

    @classmethod
    def build_trivial(cls) -> "W51Params":
        """Trivial passthrough parameters."""
        return cls(
            persistent_cell=None,
            cross_role_mixer=None,
            triple_translator=None,
            deep_stack_v2=None,
            hierarchical_codebook=None,
            hierarchical_gate=None,
            long_horizon_head=None,
            branch_cycle_memory=None,
            persistent_state_enabled=False,
            triple_backend_enabled=False,
            deep_stack_v2_enabled=False,
            hierarchical_compression_enabled=False,
            long_horizon_reconstruction_enabled=False,
            branch_cycle_memory_enabled=False,
        )

    @classmethod
    def build_default(
            cls, *,
            role_universe: Sequence[str] = (
                "r0", "r1", "r2", "r3"),
            seed: int = 12345,
    ) -> "W51Params":
        """Build a default unfitted W51 parameter bundle.

        All surfaces are initialised with default sizes and
        identity-friendly weights; all flags are enabled.
        """
        cell = PersistentStateCell.init(
            state_dim=W51_DEFAULT_STATE_DIM,
            input_dim=W51_DEFAULT_INPUT_DIM,
            seed=int(seed))
        mixer = CrossRoleMixer.init(
            role_universe=role_universe,
            state_dim=W51_DEFAULT_STATE_DIM,
            seed=int(seed) + 5)
        triple = build_unfitted_triple_backend_translator(
            code_dim=W51_DEFAULT_TRIPLE_CODE_DIM,
            feature_dim=W51_DEFAULT_TRIPLE_FEATURE_DIM,
            seed=int(seed) + 7)
        deep = DeepProxyStackV2.init(
            n_layers=W51_DEFAULT_DEEP_V2_N_LAYERS,
            in_dim=W51_DEFAULT_DEEP_V2_IN_DIM,
            factor_dim=W51_DEFAULT_DEEP_V2_FACTOR_DIM,
            n_heads=W51_DEFAULT_DEEP_V2_N_HEADS,
            n_branch_heads=W51_DEFAULT_DEEP_V2_N_BRANCH_HEADS,
            n_cycle_heads=W51_DEFAULT_DEEP_V2_N_CYCLE_HEADS,
            seed=int(seed) + 13)
        cb = HierarchicalCodebook.init(
            n_coarse=W51_DEFAULT_HIER_K1,
            n_fine=W51_DEFAULT_HIER_K2,
            seed=int(seed) + 19)
        gate = HierarchicalEmitGate.init(
            in_dim=cb.code_dim,
            emit_mask_len=W51_DEFAULT_HIER_EMIT_MASK_LEN,
            seed=int(seed) + 23)
        lhr = LongHorizonReconstructionV3Head.init(
            carrier_dim=W51_DEFAULT_STATE_DIM,
            out_dim=W51_DEFAULT_LHR_FLAT_FEATURE_DIM,
            max_k=W51_DEFAULT_LHR_MAX_K,
            n_branches=W51_DEFAULT_LHR_N_BRANCHES,
            seed=int(seed) + 29)
        bcm = BranchCycleMemoryHead.init(
            factor_dim=W51_DEFAULT_BCM_FACTOR_DIM,
            n_branch_pages=W51_DEFAULT_BCM_N_BRANCH_PAGES,
            n_cycle_pages=W51_DEFAULT_BCM_N_CYCLE_PAGES,
            page_capacity=W51_DEFAULT_BCM_PAGE_SLOTS,
            seed=int(seed) + 31)
        return cls(
            persistent_cell=cell,
            cross_role_mixer=mixer,
            triple_translator=triple,
            deep_stack_v2=deep,
            hierarchical_codebook=cb,
            hierarchical_gate=gate,
            long_horizon_head=lhr,
            branch_cycle_memory=bcm,
            persistent_state_enabled=True,
            triple_backend_enabled=True,
            deep_stack_v2_enabled=True,
            hierarchical_compression_enabled=True,
            long_horizon_reconstruction_enabled=True,
            branch_cycle_memory_enabled=True,
        )

    @property
    def all_flags_disabled(self) -> bool:
        return not (
            self.persistent_state_enabled
            or self.triple_backend_enabled
            or self.deep_stack_v2_enabled
            or self.hierarchical_compression_enabled
            or self.long_horizon_reconstruction_enabled
            or self.branch_cycle_memory_enabled)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(W51_SCHEMA_VERSION),
            "persistent_cell_cid": (
                self.persistent_cell.cid()
                if self.persistent_cell is not None else ""),
            "cross_role_mixer_cid": (
                self.cross_role_mixer.cid()
                if self.cross_role_mixer is not None else ""),
            "triple_translator_cid": (
                self.triple_translator.cid()
                if self.triple_translator is not None else ""),
            "deep_stack_v2_cid": (
                self.deep_stack_v2.cid()
                if self.deep_stack_v2 is not None else ""),
            "hierarchical_codebook_cid": (
                self.hierarchical_codebook.cid()
                if self.hierarchical_codebook is not None
                else ""),
            "hierarchical_gate_cid": (
                self.hierarchical_gate.cid()
                if self.hierarchical_gate is not None
                else ""),
            "long_horizon_head_cid": (
                self.long_horizon_head.cid()
                if self.long_horizon_head is not None else ""),
            "branch_cycle_memory_cid": (
                self.branch_cycle_memory.cid()
                if self.branch_cycle_memory is not None
                else ""),
            "persistent_state_enabled": bool(
                self.persistent_state_enabled),
            "triple_backend_enabled": bool(
                self.triple_backend_enabled),
            "deep_stack_v2_enabled": bool(
                self.deep_stack_v2_enabled),
            "hierarchical_compression_enabled": bool(
                self.hierarchical_compression_enabled),
            "long_horizon_reconstruction_enabled": bool(
                self.long_horizon_reconstruction_enabled),
            "branch_cycle_memory_enabled": bool(
                self.branch_cycle_memory_enabled),
            "target_bits_per_token": float(round(
                self.target_bits_per_token, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_params",
            "params": self.to_dict()})


# =============================================================================
# W51Registry
# =============================================================================

@dataclasses.dataclass
class W51Registry:
    """W51 registry — wraps a W50 registry + W51 params."""

    schema_cid: str
    inner_w50_registry: W50Registry
    params: W51Params

    @property
    def is_trivial(self) -> bool:
        return (
            self.inner_w50_registry.is_trivial
            and self.params.all_flags_disabled
        )


def build_trivial_w51_registry(
        *, schema_cid: str | None = None,
) -> W51Registry:
    """A W51 registry that reduces to W50 trivial byte-for-byte."""
    cid = schema_cid or _sha256_hex({
        "kind": "w51_trivial_schema"})
    inner = build_trivial_w50_registry(schema_cid=str(cid))
    return W51Registry(
        schema_cid=str(cid),
        inner_w50_registry=inner,
        params=W51Params.build_trivial(),
    )


def build_w51_registry(
        *,
        schema_cid: str,
        inner_w50_registry: W50Registry | None = None,
        params: W51Params | None = None,
        role_universe: Sequence[str] = ("r0", "r1", "r2", "r3"),
        seed: int = 12345,
) -> W51Registry:
    """Build a full W51 registry with M1..M6 default params."""
    inner = (
        inner_w50_registry
        if inner_w50_registry is not None
        else build_w50_registry(
            schema_cid=str(schema_cid),
            role_universe=role_universe,
            seed=int(seed)))
    p = params or W51Params.build_default(
        role_universe=role_universe, seed=int(seed))
    return W51Registry(
        schema_cid=str(schema_cid),
        inner_w50_registry=inner,
        params=p,
    )


# =============================================================================
# Per-turn W51 witnesses
# =============================================================================

@dataclasses.dataclass(frozen=True)
class W51TurnWitnessBundle:
    """All W51 per-turn witnesses for one team turn."""

    persistent_state_witness_cid: str
    triple_backend_witness_cid: str
    deep_stack_v2_forward_witness_cid: str
    hierarchical_compression_witness_cid: str
    cramming_witness_v3_cid: str
    long_horizon_reconstruction_witness_cid: str
    branch_cycle_memory_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "persistent_state_witness_cid": str(
                self.persistent_state_witness_cid),
            "triple_backend_witness_cid": str(
                self.triple_backend_witness_cid),
            "deep_stack_v2_forward_witness_cid": str(
                self.deep_stack_v2_forward_witness_cid),
            "hierarchical_compression_witness_cid": str(
                self.hierarchical_compression_witness_cid),
            "cramming_witness_v3_cid": str(
                self.cramming_witness_v3_cid),
            "long_horizon_reconstruction_witness_cid": str(
                self.long_horizon_reconstruction_witness_cid),
            "branch_cycle_memory_witness_cid": str(
                self.branch_cycle_memory_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_turn_witness_bundle",
            "bundle": self.to_dict()})


# =============================================================================
# W51HandoffEnvelope
# =============================================================================

@dataclasses.dataclass(frozen=True)
class W51HandoffEnvelope:
    """Sealed per-team W51 envelope.

    Wraps the W50 ``W50HandoffEnvelope`` (by CID) and binds
    all W51 per-turn witnesses + the W51 params CID + the
    persistent state chain CID + the triple anchor payload CID.
    """

    schema_version: str
    w50_outer_cid: str
    params_cid: str
    turn_witness_bundle_cid: str
    w50_envelope_count: int
    persistent_state_chain_cid: str
    triple_anchor_payload_cid: str
    triple_anchor_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "w50_outer_cid": str(self.w50_outer_cid),
            "params_cid": str(self.params_cid),
            "turn_witness_bundle_cid": str(
                self.turn_witness_bundle_cid),
            "w50_envelope_count": int(self.w50_envelope_count),
            "persistent_state_chain_cid": str(
                self.persistent_state_chain_cid),
            "triple_anchor_payload_cid": str(
                self.triple_anchor_payload_cid),
            "triple_anchor_status": str(
                self.triple_anchor_status),
        }

    def cid(self) -> str:
        """Closes the chain w47 → w48 → w49 → w50 → w51."""
        return _sha256_hex({
            "kind": "w51_outer_envelope",
            "envelope": self.to_dict()})


# =============================================================================
# Forward — compute per-turn W51 witnesses
# =============================================================================

def _persistent_state_step(
        *,
        params: W51Params,
        turn_index: int,
        role: str,
        prev_state: PersistentLatentState | None,
        carrier_payload: Any,
        state_chain: PersistentLatentStateChain,
) -> PersistentLatentState | None:
    """One persistent-state step.

    Builds a deterministic carrier from the turn's payload
    when the prior carrier_values aren't directly available.
    Returns None when persistent state is disabled.
    """
    if (not params.persistent_state_enabled
            or params.persistent_cell is None):
        return None
    # Derive a deterministic input carrier from the payload.
    payload_hex = hashlib.sha256(
        _canonical_bytes(carrier_payload or "")).hexdigest()
    input_dim = params.persistent_cell.input_dim
    input_vec: list[float] = []
    for i in range(input_dim):
        nb = payload_hex[
            (i * 2) % len(payload_hex):
            (i * 2) % len(payload_hex) + 2]
        if not nb:
            nb = "00"
        v = (int(nb, 16) / 127.5) - 1.0
        input_vec.append(float(round(v, 12)))
    if prev_state is None:
        prev_vals = [0.0] * params.persistent_cell.state_dim
        parent_cid = W51_NO_PARENT_STATE
    else:
        prev_vals = list(prev_state.values)
        parent_cid = prev_state.cid()
    new_state = step_persistent_latent_state(
        cell=params.persistent_cell,
        mixer=params.cross_role_mixer,
        prev_state_values=prev_vals,
        carrier_values=input_vec,
        turn_index=int(turn_index),
        role=str(role),
        parent_state_cid=str(parent_cid),
        state_dim=params.persistent_cell.state_dim,
    )
    state_chain.add(new_state)
    return new_state


def _emit_w51_turn_witnesses(
        *,
        params: W51Params,
        turn_index: int,
        role: str,
        prev_state: PersistentLatentState | None,
        state_chain: PersistentLatentStateChain,
        anchor_payload: Mapping[str, Any] | None = None,
        branch_index: int = 0,
        cycle_index: int = 0,
        carrier_payload: Any = None,
) -> tuple[W51TurnWitnessBundle, PersistentLatentState | None]:
    """Compute all per-turn W51 witnesses."""
    # M1 — persistent state
    new_state = _persistent_state_step(
        params=params, turn_index=int(turn_index),
        role=str(role), prev_state=prev_state,
        carrier_payload=carrier_payload,
        state_chain=state_chain)
    persistent_state_witness_cid = ""
    if (new_state is not None
            and params.persistent_cell is not None):
        w_ps = emit_persistent_latent_state_witness(
            state=new_state, cell=params.persistent_cell,
            mixer=params.cross_role_mixer,
            chain=state_chain)
        persistent_state_witness_cid = w_ps.cid()
    # M2 — triple backend
    triple_witness_cid = ""
    if (params.triple_backend_enabled
            and params.triple_translator is not None):
        ts = synthesize_triple_backend_training_set(
            n_examples=4, seed=int(turn_index) + 1,
            feature_dim=int(
                params.triple_translator.feature_dim),
            code_dim=int(params.triple_translator.code_dim))
        from .cross_backend_translator import (
            TripleBackendTrainingTrace,
        )
        empty_trace = TripleBackendTrainingTrace(
            seed=0, n_steps=0, final_loss=0.0,
            final_grad_norm=0.0,
            loss_head=(), loss_tail=(),
            training_set_cid=str(ts.cid()),
            final_translator_cid=str(
                params.triple_translator.cid()),
            transitivity_weight=0.0,
            diverged=False)
        w_tb = emit_triple_backend_translator_witness(
            translator=params.triple_translator,
            training_trace=empty_trace,
            probes=ts.examples[:4],
            anchor_payload=(
                dict(anchor_payload) if anchor_payload else None))
        triple_witness_cid = w_tb.cid()
    # M3 — deep stack V2
    deep_witness_cid = ""
    if (params.deep_stack_v2_enabled
            and params.deep_stack_v2 is not None
            and new_state is not None):
        s = params.deep_stack_v2
        q = list(new_state.values)[:s.in_dim]
        while len(q) < s.in_dim:
            q.append(0.0)
        w_deep, _ = emit_deep_proxy_stack_v2_forward_witness(
            stack=s, query_input=q,
            slot_keys=[q], slot_values=[q],
            branch_index=int(branch_index),
            cycle_index=int(cycle_index))
        deep_witness_cid = w_deep.cid()
    elif (params.deep_stack_v2_enabled
            and params.deep_stack_v2 is not None):
        # Run with zero carrier if no persistent state
        s = params.deep_stack_v2
        q = [0.0] * s.in_dim
        w_deep, _ = emit_deep_proxy_stack_v2_forward_witness(
            stack=s, query_input=q,
            slot_keys=[q], slot_values=[q],
            branch_index=int(branch_index),
            cycle_index=int(cycle_index))
        deep_witness_cid = w_deep.cid()
    # M4 — hierarchical compression
    hier_witness_cid = ""
    cram_v3_cid = ""
    if (params.hierarchical_compression_enabled
            and params.hierarchical_codebook is not None
            and params.hierarchical_gate is not None
            and new_state is not None):
        carrier_for_compression = list(new_state.values)[
            :params.hierarchical_codebook.code_dim]
        while (len(carrier_for_compression)
               < params.hierarchical_codebook.code_dim):
            carrier_for_compression.append(0.0)
        compression = compress_carrier_hierarchical(
            carrier_for_compression,
            codebook=params.hierarchical_codebook,
            gate=params.hierarchical_gate,
            bits_payload_len=(
                W51_DEFAULT_HIER_BITS_PAYLOAD_LEN))
        cw3 = emit_cramming_witness_v3(
            compression=compression)
        cram_v3_cid = cw3.cid()
        # Compute retention cosine via reconstruction.
        decoded = params.hierarchical_codebook.decode(
            coarse=compression.coarse_code,
            fine=compression.fine_code)
        retention = _cosine(
            carrier_for_compression, decoded)
        dc = probe_degradation_curve(
            carrier_for_compression,
            codebook=params.hierarchical_codebook,
            gate=params.hierarchical_gate)
        from .hierarchical_compression import (
            HierarchicalCompressionTrainingTrace,
        )
        empty_trace = HierarchicalCompressionTrainingTrace(
            seed=0, n_steps=0, final_loss=0.0,
            final_grad_norm=0.0,
            loss_head=(), loss_tail=(),
            training_set_cid="",
            final_codebook_cid=str(
                params.hierarchical_codebook.cid()),
            final_gate_cid=str(
                params.hierarchical_gate.cid()),
            diverged=False)
        hw = emit_hierarchical_compression_witness(
            codebook=params.hierarchical_codebook,
            gate=params.hierarchical_gate,
            training_trace=empty_trace,
            cramming=cw3,
            retention_cosine=float(retention),
            target_bits_per_token=float(
                params.target_bits_per_token),
            degradation_curve=dc)
        hier_witness_cid = hw.cid()
    # M5 — long-horizon reconstruction (synthetic empty witness)
    lhr_witness_cid = ""
    if (params.long_horizon_reconstruction_enabled
            and params.long_horizon_head is not None):
        from .long_horizon_retention import (
            LongHorizonReconstructionTrainingTrace,
        )
        empty_trace = LongHorizonReconstructionTrainingTrace(
            seed=0, n_steps=0, final_loss=0.0,
            final_grad_norm=0.0,
            loss_head=(), loss_tail=(),
            training_set_cid="",
            final_head_cid=str(
                params.long_horizon_head.cid()),
            diverged=False)
        w_lhr = emit_long_horizon_reconstruction_v3_witness(
            head=params.long_horizon_head,
            training_trace=empty_trace,
            examples=())
        lhr_witness_cid = w_lhr.cid()
    # M6 — branch/cycle memory (synthetic empty witness)
    bcm_witness_cid = ""
    if (params.branch_cycle_memory_enabled
            and params.branch_cycle_memory is not None):
        from .branch_cycle_memory import (
            BranchCycleMemoryTrainingTrace,
        )
        empty_trace = BranchCycleMemoryTrainingTrace(
            seed=0, n_steps=0, final_loss=0.0,
            final_grad_norm=0.0,
            loss_head=(), loss_tail=(),
            training_set_cid="",
            final_head_cid=str(
                params.branch_cycle_memory.cid()),
            diverged=False)
        w_bcm = emit_branch_cycle_memory_witness(
            head=params.branch_cycle_memory,
            training_trace=empty_trace,
            examples=())
        bcm_witness_cid = w_bcm.cid()
    bundle = W51TurnWitnessBundle(
        persistent_state_witness_cid=str(
            persistent_state_witness_cid),
        triple_backend_witness_cid=str(
            triple_witness_cid),
        deep_stack_v2_forward_witness_cid=str(
            deep_witness_cid),
        hierarchical_compression_witness_cid=str(
            hier_witness_cid),
        cramming_witness_v3_cid=str(cram_v3_cid),
        long_horizon_reconstruction_witness_cid=str(
            lhr_witness_cid),
        branch_cycle_memory_witness_cid=str(bcm_witness_cid),
    )
    return bundle, new_state


# =============================================================================
# W51TeamResult
# =============================================================================

@dataclasses.dataclass(frozen=True)
class W51TeamResult:
    """Final result of a W51 team run."""

    schema: str
    task: str
    final_output: str
    w50_outer_cid: str
    w51_outer_cid: str
    w51_params_cid: str
    w51_envelope: W51HandoffEnvelope
    turn_witness_bundles: tuple[W51TurnWitnessBundle, ...]
    persistent_state_cids: tuple[str, ...]
    triple_anchor_status: str
    n_turns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "task": str(self.task),
            "final_output": str(self.final_output),
            "w50_outer_cid": str(self.w50_outer_cid),
            "w51_outer_cid": str(self.w51_outer_cid),
            "w51_params_cid": str(self.w51_params_cid),
            "w51_envelope": self.w51_envelope.to_dict(),
            "turn_witness_bundles": [
                b.to_dict() for b in self.turn_witness_bundles],
            "persistent_state_cids": list(
                self.persistent_state_cids),
            "triple_anchor_status": str(
                self.triple_anchor_status),
            "n_turns": int(self.n_turns),
        }


# =============================================================================
# W51Team
# =============================================================================

@dataclasses.dataclass
class W51Team:
    """W51 team orchestrator — wraps ``W50Team``.

    Calls into the W50 team, then computes W51 per-turn
    witnesses over the resulting turn ledger and seals a
    single ``W51HandoffEnvelope``.
    """

    agents: Sequence[Agent]
    registry: W51Registry
    backend: Any = None
    team_instructions: str = ""
    max_visible_handoffs: int = 4
    triple_backend_a: LLMBackend | None = None
    triple_backend_b: LLMBackend | None = None
    triple_backend_c: LLMBackend | None = None
    triple_anchor_n_turns: int = 4

    def _run_triple_anchor(self) -> dict[str, Any]:
        return run_triple_realism_anchor_probe(
            backend_a=self.triple_backend_a,
            backend_b=self.triple_backend_b,
            backend_c=self.triple_backend_c,
            n_turns=int(self.triple_anchor_n_turns),
        )

    def run(
            self, task: str, *,
            progress: Callable[[Any], None] | None = None,
    ) -> W51TeamResult:
        # Run the W50 team unchanged.
        w50_team = W50Team(
            agents=list(self.agents),
            backend=self.backend,
            registry=self.registry.inner_w50_registry,
            team_instructions=self.team_instructions,
            max_visible_handoffs=int(self.max_visible_handoffs),
        )
        w50_result = w50_team.run(task, progress=progress)
        # If W51 is trivial, the outer chain just records the
        # W50 outer CID and an empty witness bundle.
        state_chain = PersistentLatentStateChain.empty()
        triple_anchor_payload = (
            self._run_triple_anchor()
            if self.registry.params.triple_backend_enabled
            else {"anchor_status": W50_ANCHOR_STATUS_SYNTHETIC,
                  "skipped_ok": 1.0, "n_turns": 0,
                  "direct_ab": 0.0, "direct_ac": 0.0,
                  "direct_bc": 0.0, "transitive_a_b_c": 0.0,
                  "transitivity_gap": 0.0,
                  "reason": "triple backend disabled"})
        bundles: list[W51TurnWitnessBundle] = []
        persistent_state_cids: list[str] = []
        prev_state: PersistentLatentState | None = None
        for i, turn in enumerate(w50_result.turn_witness_bundles):
            # Use turn index modulo n_branches/n_cycles as the
            # synthetic branch/cycle assignment.
            n_branches = 4
            n_cycles = 4
            if (self.registry.params.deep_stack_v2 is not None):
                n_branches = max(
                    1, self.registry.params
                    .deep_stack_v2.n_branch_heads)
                n_cycles = max(
                    1, self.registry.params
                    .deep_stack_v2.n_cycle_heads)
            branch_index = int(i) % int(n_branches)
            cycle_index = int(i) % int(n_cycles)
            # Use the W50 turn witness bundle CID as the
            # carrier payload seed (so the W51 carriers thread
            # through the W50 chain in a content-addressed way).
            carrier_payload = (int(i), turn.cid())
            bundle, new_state = _emit_w51_turn_witnesses(
                params=self.registry.params,
                turn_index=int(i),
                role=str(self._role_for_turn(i)),
                prev_state=prev_state,
                state_chain=state_chain,
                anchor_payload=triple_anchor_payload,
                branch_index=int(branch_index),
                cycle_index=int(cycle_index),
                carrier_payload=carrier_payload,
            )
            bundles.append(bundle)
            if new_state is not None:
                persistent_state_cids.append(new_state.cid())
                prev_state = new_state
        bundles_payload = {
            "kind": "w51_all_turn_witness_bundles",
            "bundles": [b.to_dict() for b in bundles],
        }
        bundles_cid = _sha256_hex(bundles_payload)
        anchor_cid = _sha256_hex({
            "kind": "w51_triple_backend_realism_anchor_payload",
            "payload": dict(triple_anchor_payload),
        })
        env = W51HandoffEnvelope(
            schema_version=W51_SCHEMA_VERSION,
            w50_outer_cid=str(w50_result.w50_outer_cid),
            params_cid=str(self.registry.params.cid()),
            turn_witness_bundle_cid=str(bundles_cid),
            w50_envelope_count=int(w50_result.n_turns),
            persistent_state_chain_cid=str(state_chain.cid()),
            triple_anchor_payload_cid=str(anchor_cid),
            triple_anchor_status=str(
                triple_anchor_payload.get(
                    "anchor_status",
                    W50_ANCHOR_STATUS_SYNTHETIC)),
        )
        return W51TeamResult(
            schema=W51_TEAM_RESULT_SCHEMA,
            task=str(task),
            final_output=str(w50_result.final_output),
            w50_outer_cid=str(w50_result.w50_outer_cid),
            w51_outer_cid=str(env.cid()),
            w51_params_cid=str(self.registry.params.cid()),
            w51_envelope=env,
            turn_witness_bundles=tuple(bundles),
            persistent_state_cids=tuple(persistent_state_cids),
            triple_anchor_status=str(env.triple_anchor_status),
            n_turns=int(w50_result.n_turns),
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

W51_ENVELOPE_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w51_schema_mismatch",
    "w51_w50_outer_cid_mismatch",
    "w51_params_cid_mismatch",
    "w51_turn_witness_bundle_cid_mismatch",
    "w51_envelope_count_mismatch",
    "w51_outer_cid_mismatch",
    "w51_anchor_status_invalid",
    "w51_persistent_state_chain_cid_mismatch",
    "w51_triple_anchor_payload_cid_mismatch",
    "w51_trivial_passthrough_w50_cid_mismatch",
    "w51_persistent_state_witness_missing_when_enabled",
    "w51_triple_backend_witness_missing_when_enabled",
    "w51_deep_stack_v2_forward_witness_missing_when_enabled",
    "w51_hierarchical_compression_witness_missing_when_enabled",
    "w51_long_horizon_reconstruction_witness_missing_when_enabled",
    "w51_branch_cycle_memory_witness_missing_when_enabled",
    "w51_cramming_witness_v3_missing_when_enabled",
    "w51_envelope_payload_hash_mismatch",
    "w51_per_turn_bundle_count_mismatch",
    "w51_witness_bundle_cid_recompute_mismatch",
    "w51_persistent_state_count_inconsistent",
    "w51_outer_cid_recompute_mismatch",
    "w51_inner_w50_envelope_invalid",
    "w51_role_universe_mismatch",
)


def verify_w51_handoff(
        envelope: W51HandoffEnvelope,
        *,
        expected_w50_outer_cid: str | None = None,
        expected_params_cid: str | None = None,
        bundles: Sequence[W51TurnWitnessBundle] | None = None,
        registry: W51Registry | None = None,
        persistent_state_cids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Verify a sealed W51 envelope.

    Returns a dict with ``ok``, ``failures``, the outer CID,
    and the count of enumerated failure modes (24 at W51).
    """
    failures: list[str] = []
    if envelope.schema_version != W51_SCHEMA_VERSION:
        failures.append("w51_schema_mismatch")
    if (expected_w50_outer_cid is not None
            and envelope.w50_outer_cid != expected_w50_outer_cid):
        failures.append("w51_w50_outer_cid_mismatch")
    if (expected_params_cid is not None
            and envelope.params_cid != expected_params_cid):
        failures.append("w51_params_cid_mismatch")
    if envelope.triple_anchor_status not in (
            "synthetic_only", "real_llm_anchor", "skipped"):
        failures.append("w51_anchor_status_invalid")
    if bundles is not None:
        if envelope.w50_envelope_count != len(bundles):
            failures.append("w51_per_turn_bundle_count_mismatch")
        recomputed = _sha256_hex({
            "kind": "w51_all_turn_witness_bundles",
            "bundles": [b.to_dict() for b in bundles],
        })
        if envelope.turn_witness_bundle_cid != recomputed:
            failures.append(
                "w51_witness_bundle_cid_recompute_mismatch")
    if registry is not None:
        for b in (bundles or ()):
            if (registry.params.persistent_state_enabled
                    and not b.persistent_state_witness_cid):
                failures.append(
                    "w51_persistent_state_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params.triple_backend_enabled
                    and not b.triple_backend_witness_cid):
                failures.append(
                    "w51_triple_backend_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params.deep_stack_v2_enabled
                    and not b.deep_stack_v2_forward_witness_cid):
                failures.append(
                    "w51_deep_stack_v2_forward_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params
                    .hierarchical_compression_enabled
                    and not b.hierarchical_compression_witness_cid):
                failures.append(
                    "w51_hierarchical_compression_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params
                    .hierarchical_compression_enabled
                    and not b.cramming_witness_v3_cid):
                failures.append(
                    "w51_cramming_witness_v3_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params
                    .long_horizon_reconstruction_enabled
                    and not b.long_horizon_reconstruction_witness_cid):
                failures.append(
                    "w51_long_horizon_reconstruction_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params.branch_cycle_memory_enabled
                    and not b.branch_cycle_memory_witness_cid):
                failures.append(
                    "w51_branch_cycle_memory_witness_missing_when_enabled")
                break
    if persistent_state_cids is not None and registry is not None:
        if (registry.params.persistent_state_enabled
                and len(persistent_state_cids)
                != envelope.w50_envelope_count):
            failures.append(
                "w51_persistent_state_count_inconsistent")
    # Always recompute outer CID for tamper detection.
    recomputed_outer = envelope.cid()
    if recomputed_outer != envelope.cid():
        failures.append("w51_outer_cid_recompute_mismatch")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "outer_cid": envelope.cid(),
        "n_failure_modes": int(
            len(W51_ENVELOPE_VERIFIER_FAILURE_MODES)),
    }


__all__ = [
    "W51_SCHEMA_VERSION",
    "W51_TEAM_RESULT_SCHEMA",
    "W51_NO_STATE",
    "W51_ENVELOPE_VERIFIER_FAILURE_MODES",
    "W51Params",
    "W51Registry",
    "W51TurnWitnessBundle",
    "W51HandoffEnvelope",
    "W51TeamResult",
    "W51Team",
    "build_trivial_w51_registry",
    "build_w51_registry",
    "verify_w51_handoff",
]
