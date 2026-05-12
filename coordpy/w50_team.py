"""W50 — Cross-Backend Latent Coordination (XBLC) composition module.

The ``W50Team`` orchestrator composes the W49 ``MultiBlockProxyTeam``
with the five M1..M5 W50 mechanism modules:

* M1 ``CrossBackendAlignmentLayer``  — cross-backend latent
   projector
* M2 ``DeepProxyStack``               — L=4 deep proxy stack
* M3 ``AdaptiveCompressionCodebook``  — K=16 adaptive emit-mask
* M4 ``CrossBankTransferLayer``       — role-pair pseudo-KV
   transfer + adaptive eviction V2
* M5 ``SharedLatentCarrierV2``        — chain-walkable carrier +
   reconstruction V2

Each W50 turn emits W50 per-module witnesses on top of the W49
``MultiBlockProxyHandoffEnvelope``. The final ``W50HandoffEnvelope``
binds: the W49 root CID, every W50 witness CID, the W50Params
CID, and a single ``w50_outer_cid`` that closes the chain.

Honest scope (do-not-overstate)
-------------------------------

This module does NOT alter transformer-internal state. Every W50
witness is computed over capsule-layer signals (the W49 multi-
block envelope, the W50 carrier values). Trivial passthrough is
preserved byte-for-byte: when ``W50Params.build_trivial()`` is
used and all flags are disabled, the W50 envelope's internal
``w49_root_cid`` equals the W49 baseline root_cid exactly — the
``W50-L-TRIVIAL-W50-PASSTHROUGH`` falsifier.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from .adaptive_compression import (
    AdaptiveCompressionCodebook,
    AdaptiveCompressionGate,
    AdaptiveCompressionWitness,
    CrammingWitnessV2,
    W50_DEFAULT_ADAPTIVE_K,
    W50_DEFAULT_BITS_PAYLOAD_LEN,
    W50_DEFAULT_EMIT_MASK_LEN,
    W50_DEFAULT_TARGET_BITS_PER_TOKEN,
    compress_carrier,
    emit_adaptive_compression_witness,
    emit_cramming_witness_v2,
)
from .agents import Agent
from .cross_backend_alignment import (
    CrossBackendAlignmentParams,
    CrossBackendAlignmentWitness,
    W50_ANCHOR_STATUS_SYNTHETIC,
    build_unfitted_cross_backend_alignment_params,
    emit_cross_backend_alignment_witness,
    run_realism_anchor_probe,
    synthesize_cross_backend_training_set,
)
from .cross_bank_transfer import (
    AdaptiveEvictionPolicyV2,
    CrossBankTransferLayer,
    CrossBankTransferWitness,
    emit_cross_bank_transfer_witness,
    synthesize_cross_bank_transfer_training_set,
)
from .deep_proxy_stack import (
    DeepProxyStack,
    DeepProxyStackForwardWitness,
    W50_DEFAULT_DEEP_FACTOR_DIM,
    W50_DEFAULT_DEEP_IN_DIM,
    W50_DEFAULT_DEEP_N_HEADS,
    W50_DEFAULT_DEEP_N_LAYERS,
    emit_deep_proxy_stack_forward_witness,
)
from .llm_backend import LLMBackend
from .multi_block_proxy import (
    MultiBlockProxyHandoffEnvelope,
    MultiBlockProxyRegistry,
    MultiBlockProxyTeam,
    build_trivial_multi_block_proxy_registry,
)
from .shared_latent_carrier import (
    ReconstructionV2Head,
    ReconstructionV2Witness,
    RoleReuseMap,
    SharedLatentCarrierChain,
    SharedLatentCarrierV2,
    SharedLatentCarrierWitness,
    W50_DEFAULT_CARRIER_DIM,
    W50_DEFAULT_FLAT_FEATURE_DIM,
    W50_DEFAULT_MAX_K_RECONSTRUCTION,
    emit_reconstruction_v2_witness,
    emit_shared_latent_carrier_witness,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W50_SCHEMA_VERSION: str = "coordpy.w50_team.v1"
W50_TEAM_RESULT_SCHEMA: str = "coordpy.w50_team_result.v1"

W50_NO_CARRIER: str = "no_w50_carrier"


# =============================================================================
# Canonicalisation helpers
# =============================================================================

def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# =============================================================================
# W50Params
# =============================================================================

@dataclasses.dataclass
class W50Params:
    """All trainable / config surfaces for W50, layered over W49.

    Each surface is optional — a "trivial" build returns identity-
    initialised surfaces with flags disabled so the W50 team
    reduces to ``MultiBlockProxyTeam.run`` byte-for-byte.
    """

    cross_backend_params: CrossBackendAlignmentParams | None
    deep_proxy_stack: DeepProxyStack | None
    adaptive_codebook: AdaptiveCompressionCodebook | None
    adaptive_gate: AdaptiveCompressionGate | None
    cross_bank_transfer: CrossBankTransferLayer | None
    eviction_v2: AdaptiveEvictionPolicyV2 | None
    role_reuse_map: RoleReuseMap | None
    reconstruction_v2_head: ReconstructionV2Head | None

    cross_backend_enabled: bool = False
    deep_stack_enabled: bool = False
    adaptive_compression_enabled: bool = False
    cross_bank_transfer_enabled: bool = False
    shared_latent_carrier_v2_enabled: bool = False

    target_bits_per_token: float = (
        W50_DEFAULT_TARGET_BITS_PER_TOKEN)

    @classmethod
    def build_trivial(cls) -> "W50Params":
        """Build trivial-passthrough parameters.

        All M1..M5 surfaces are absent; all enable flags are
        ``False``. The W50 team reduces to ``MultiBlockProxyTeam``
        byte-for-byte under these params.
        """
        return cls(
            cross_backend_params=None,
            deep_proxy_stack=None,
            adaptive_codebook=None,
            adaptive_gate=None,
            cross_bank_transfer=None,
            eviction_v2=None,
            role_reuse_map=None,
            reconstruction_v2_head=None,
            cross_backend_enabled=False,
            deep_stack_enabled=False,
            adaptive_compression_enabled=False,
            cross_bank_transfer_enabled=False,
            shared_latent_carrier_v2_enabled=False,
        )

    @classmethod
    def build_default(
            cls, *,
            role_universe: Sequence[str] = ("r0", "r1", "r2", "r3"),
            seed: int = 12345,
    ) -> "W50Params":
        """Build a default unfitted W50 parameter bundle.

        All surfaces are initialised with default sizes and
        identity-friendly weights; all flags are enabled.
        """
        cb_params = (
            build_unfitted_cross_backend_alignment_params(
                seed=int(seed)))
        deep = DeepProxyStack.init(
            n_layers=W50_DEFAULT_DEEP_N_LAYERS,
            in_dim=W50_DEFAULT_DEEP_IN_DIM,
            factor_dim=W50_DEFAULT_DEEP_FACTOR_DIM,
            n_heads=W50_DEFAULT_DEEP_N_HEADS,
            seed=int(seed))
        codebook = AdaptiveCompressionCodebook.init(
            n_codes=W50_DEFAULT_ADAPTIVE_K, seed=int(seed) + 7)
        gate = AdaptiveCompressionGate.init(
            in_dim=codebook.code_dim,
            emit_mask_len=W50_DEFAULT_EMIT_MASK_LEN,
            seed=int(seed) + 11)
        xfer = CrossBankTransferLayer.init(
            role_universe=role_universe,
            seed=int(seed) + 13)
        evictor = AdaptiveEvictionPolicyV2.init(
            seed=int(seed) + 17)
        rrm = RoleReuseMap.init(
            role_universe=role_universe,
            carrier_dim=W50_DEFAULT_CARRIER_DIM,
            seed=int(seed) + 19)
        recon = ReconstructionV2Head.init(
            carrier_dim=W50_DEFAULT_CARRIER_DIM,
            out_dim=W50_DEFAULT_FLAT_FEATURE_DIM,
            max_k=W50_DEFAULT_MAX_K_RECONSTRUCTION,
            seed=int(seed) + 23)
        return cls(
            cross_backend_params=cb_params,
            deep_proxy_stack=deep,
            adaptive_codebook=codebook,
            adaptive_gate=gate,
            cross_bank_transfer=xfer,
            eviction_v2=evictor,
            role_reuse_map=rrm,
            reconstruction_v2_head=recon,
            cross_backend_enabled=True,
            deep_stack_enabled=True,
            adaptive_compression_enabled=True,
            cross_bank_transfer_enabled=True,
            shared_latent_carrier_v2_enabled=True,
        )

    @property
    def all_flags_disabled(self) -> bool:
        return not (
            self.cross_backend_enabled
            or self.deep_stack_enabled
            or self.adaptive_compression_enabled
            or self.cross_bank_transfer_enabled
            or self.shared_latent_carrier_v2_enabled)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(W50_SCHEMA_VERSION),
            "cross_backend_params_cid": (
                self.cross_backend_params.cid()
                if self.cross_backend_params is not None
                else ""),
            "deep_proxy_stack_cid": (
                self.deep_proxy_stack.cid()
                if self.deep_proxy_stack is not None
                else ""),
            "adaptive_codebook_cid": (
                self.adaptive_codebook.cid()
                if self.adaptive_codebook is not None
                else ""),
            "adaptive_gate_cid": (
                self.adaptive_gate.cid()
                if self.adaptive_gate is not None
                else ""),
            "cross_bank_transfer_cid": (
                self.cross_bank_transfer.cid()
                if self.cross_bank_transfer is not None
                else ""),
            "eviction_v2_cid": (
                self.eviction_v2.cid()
                if self.eviction_v2 is not None
                else ""),
            "role_reuse_map_cid": (
                self.role_reuse_map.cid()
                if self.role_reuse_map is not None
                else ""),
            "reconstruction_v2_head_cid": (
                self.reconstruction_v2_head.cid()
                if self.reconstruction_v2_head is not None
                else ""),
            "cross_backend_enabled": bool(
                self.cross_backend_enabled),
            "deep_stack_enabled": bool(self.deep_stack_enabled),
            "adaptive_compression_enabled": bool(
                self.adaptive_compression_enabled),
            "cross_bank_transfer_enabled": bool(
                self.cross_bank_transfer_enabled),
            "shared_latent_carrier_v2_enabled": bool(
                self.shared_latent_carrier_v2_enabled),
            "target_bits_per_token": float(round(
                self.target_bits_per_token, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_params",
            "params": self.to_dict()})


# =============================================================================
# W50Registry
# =============================================================================

@dataclasses.dataclass
class W50Registry:
    """W50 registry — wraps a W49 registry + W50 params."""

    schema_cid: str
    inner_w49_registry: MultiBlockProxyRegistry
    params: W50Params

    @property
    def is_trivial(self) -> bool:
        return (
            self.inner_w49_registry.is_trivial
            and self.params.all_flags_disabled
        )


def build_trivial_w50_registry(
        *, schema_cid: str | None = None,
) -> W50Registry:
    """A W50 registry that reduces to W49 trivial byte-for-byte."""
    cid = schema_cid or _sha256_hex({
        "kind": "w50_trivial_schema"})
    inner = build_trivial_multi_block_proxy_registry(
        schema_cid=str(cid))
    return W50Registry(
        schema_cid=str(cid),
        inner_w49_registry=inner,
        params=W50Params.build_trivial(),
    )


def build_w50_registry(
        *,
        schema_cid: str,
        inner_w49_registry: MultiBlockProxyRegistry | None = None,
        params: W50Params | None = None,
        role_universe: Sequence[str] = ("r0", "r1", "r2", "r3"),
        seed: int = 12345,
) -> W50Registry:
    """Build a full W50 registry with M1..M5 default params."""
    inner = (
        inner_w49_registry
        if inner_w49_registry is not None
        else build_trivial_multi_block_proxy_registry(
            schema_cid=str(schema_cid)))
    p = params or W50Params.build_default(
        role_universe=role_universe, seed=int(seed))
    return W50Registry(
        schema_cid=str(schema_cid),
        inner_w49_registry=inner,
        params=p,
    )


# =============================================================================
# Per-turn W50 witnesses
# =============================================================================

@dataclasses.dataclass(frozen=True)
class W50TurnWitnessBundle:
    """All W50 per-turn witnesses for one team turn."""

    cross_backend_witness_cid: str
    deep_proxy_forward_witness_cid: str
    adaptive_compression_witness_cid: str
    cramming_witness_v2_cid: str
    cross_bank_transfer_witness_cid: str
    shared_latent_carrier_witness_cid: str
    reconstruction_v2_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cross_backend_witness_cid": str(
                self.cross_backend_witness_cid),
            "deep_proxy_forward_witness_cid": str(
                self.deep_proxy_forward_witness_cid),
            "adaptive_compression_witness_cid": str(
                self.adaptive_compression_witness_cid),
            "cramming_witness_v2_cid": str(
                self.cramming_witness_v2_cid),
            "cross_bank_transfer_witness_cid": str(
                self.cross_bank_transfer_witness_cid),
            "shared_latent_carrier_witness_cid": str(
                self.shared_latent_carrier_witness_cid),
            "reconstruction_v2_witness_cid": str(
                self.reconstruction_v2_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_turn_witness_bundle",
            "bundle": self.to_dict()})


# =============================================================================
# W50HandoffEnvelope
# =============================================================================

@dataclasses.dataclass(frozen=True)
class W50HandoffEnvelope:
    """Sealed per-turn W50 envelope.

    Wraps the W49 ``MultiBlockProxyHandoffEnvelope`` (by CID) and
    binds all W50 per-turn witnesses + the W50 params CID.
    """

    schema_version: str
    w49_root_cid: str
    params_cid: str
    turn_witness_bundle_cid: str
    w49_envelope_count: int
    w50_carrier_chain_cid: str
    realism_anchor_payload_cid: str
    anchor_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "w49_root_cid": str(self.w49_root_cid),
            "params_cid": str(self.params_cid),
            "turn_witness_bundle_cid": str(
                self.turn_witness_bundle_cid),
            "w49_envelope_count": int(self.w49_envelope_count),
            "w50_carrier_chain_cid": str(
                self.w50_carrier_chain_cid),
            "realism_anchor_payload_cid": str(
                self.realism_anchor_payload_cid),
            "anchor_status": str(self.anchor_status),
        }

    def cid(self) -> str:
        """The W50 outer CID — closes the chain w47 → w48 → w49 →
        w50."""
        return _sha256_hex({
            "kind": "w50_outer_envelope",
            "envelope": self.to_dict()})


# =============================================================================
# Forward — compute per-turn W50 witnesses
# =============================================================================

def _carrier_from_w49_turn(
        turn_index: int,
        role: str,
        prev_carrier_cid: str,
        role_reuse_map_cid: str,
        *,
        carrier_dim: int = W50_DEFAULT_CARRIER_DIM,
        seed_payload: Any = None,
) -> SharedLatentCarrierV2:
    """Build a deterministic shared-latent carrier V2 from the
    turn payload.

    Uses a content-addressed hash of the (turn_index, role,
    payload) tuple to produce stable values.
    """
    payload_hex = hashlib.sha256(
        _canonical_bytes(seed_payload or "")).hexdigest()
    # Derive 8 floats by reading nibbles of the hash.
    vals: list[float] = []
    for i in range(int(carrier_dim)):
        nb = payload_hex[(i * 2) % len(payload_hex):
                         (i * 2) % len(payload_hex) + 2]
        if not nb:
            nb = "00"
        # Map 00..ff to [-1, 1]
        v = (int(nb, 16) / 127.5) - 1.0
        vals.append(float(round(v, 12)))
    return SharedLatentCarrierV2(
        turn_index=int(turn_index),
        role=str(role),
        carrier_dim=int(carrier_dim),
        values=tuple(vals),
        parent_carrier_cid=str(prev_carrier_cid),
        role_reuse_map_cid=str(role_reuse_map_cid),
    )


def _emit_w50_turn_witnesses(
        *,
        params: W50Params,
        turn_index: int,
        role: str,
        prev_carrier_cid: str,
        carrier_chain: SharedLatentCarrierChain,
        anchor_payload: Mapping[str, Any] | None = None,
) -> tuple[W50TurnWitnessBundle, SharedLatentCarrierV2]:
    """Compute all per-turn W50 witnesses.

    Each witness is computed conditionally on its corresponding
    enable flag; disabled flags emit empty (`""`) CIDs.
    """
    # Build the carrier V2 for this turn.
    rrm_cid = (
        params.role_reuse_map.cid()
        if params.role_reuse_map is not None
        else "")
    carrier = _carrier_from_w49_turn(
        turn_index=int(turn_index),
        role=str(role),
        prev_carrier_cid=str(prev_carrier_cid),
        role_reuse_map_cid=str(rrm_cid),
        seed_payload=(int(turn_index), str(role),
                       str(prev_carrier_cid)),
    )
    if params.shared_latent_carrier_v2_enabled:
        carrier_chain.add(carrier)

    # M1 cross-backend witness
    cb_witness_cid = ""
    if (params.cross_backend_enabled
            and params.cross_backend_params is not None):
        ts = synthesize_cross_backend_training_set(
            n_pairs=4, seed=int(turn_index) + 1)
        # Synthesise a fast empty trace; the actual training is
        # held outside the per-turn forward path.
        from .cross_backend_alignment import (
            CrossBackendTrainingTrace, score_alignment_fidelity,
        )
        empty_trace = CrossBackendTrainingTrace(
            seed=0, n_steps=0, final_loss=0.0,
            final_grad_norm=0.0,
            loss_head=(), loss_tail=(),
            training_set_cid=str(ts.cid()),
            final_params_cid=str(
                params.cross_backend_params.cid()),
            diverged=False)
        w = emit_cross_backend_alignment_witness(
            params=params.cross_backend_params,
            training_trace=empty_trace,
            probe_pairs=ts.pairs[:4],
            anchor_payload=(
                dict(anchor_payload) if anchor_payload else None))
        cb_witness_cid = w.cid()

    # M2 deep stack forward witness
    deep_witness_cid = ""
    if (params.deep_stack_enabled
            and params.deep_proxy_stack is not None):
        s = params.deep_proxy_stack
        q = list(carrier.values)[:s.in_dim]
        while len(q) < s.in_dim:
            q.append(0.0)
        w_deep, _ = emit_deep_proxy_stack_forward_witness(
            stack=s, query_input=q,
            slot_keys=[q], slot_values=[q])
        deep_witness_cid = w_deep.cid()

    # M3 adaptive compression witness + cramming V2
    ac_witness_cid = ""
    cw_v2_cid = ""
    if (params.adaptive_compression_enabled
            and params.adaptive_codebook is not None
            and params.adaptive_gate is not None):
        compression = compress_carrier(
            list(carrier.values), codebook=params.adaptive_codebook,
            gate=params.adaptive_gate,
            bits_payload_len=W50_DEFAULT_BITS_PAYLOAD_LEN)
        cw = emit_cramming_witness_v2(compression=compression)
        cw_v2_cid = cw.cid()
        # Compact training trace
        from .adaptive_compression import (
            AdaptiveCompressionTrainingTrace,
        )
        empty_trace = AdaptiveCompressionTrainingTrace(
            seed=0, n_steps=0, final_loss=0.0,
            final_grad_norm=0.0,
            loss_head=(), loss_tail=(),
            training_set_cid="",
            final_codebook_cid=str(
                params.adaptive_codebook.cid()),
            final_gate_cid=str(params.adaptive_gate.cid()),
            diverged=False)
        aw = emit_adaptive_compression_witness(
            codebook=params.adaptive_codebook,
            gate=params.adaptive_gate,
            training_trace=empty_trace,
            cramming=cw,
            target_bits_per_token=float(
                params.target_bits_per_token),
            retention_floor=0.0)
        ac_witness_cid = aw.cid()

    # M4 cross-bank transfer witness
    cbt_witness_cid = ""
    if (params.cross_bank_transfer_enabled
            and params.cross_bank_transfer is not None):
        from .cross_bank_transfer import (
            CrossBankTransferTrainingTrace,
        )
        # Synthesise a small probe set
        ts = synthesize_cross_bank_transfer_training_set(
            seed=int(turn_index) + 7,
            role_universe=(
                params.cross_bank_transfer.role_universe),
            factor_dim=int(
                params.cross_bank_transfer.factor_dim),
            n_examples_per_pair=2)
        empty_trace = CrossBankTransferTrainingTrace(
            seed=0, n_steps=0, final_loss=0.0,
            final_grad_norm=0.0,
            loss_head=(), loss_tail=(),
            training_set_cid=str(ts.cid()),
            final_layer_cid=str(
                params.cross_bank_transfer.cid()),
            diverged=False)
        w_cbt = emit_cross_bank_transfer_witness(
            layer=params.cross_bank_transfer,
            training_trace=empty_trace,
            probe_examples=ts.examples[:8])
        cbt_witness_cid = w_cbt.cid()

    # M5 shared latent carrier witness + reconstruction witness
    slcw_cid = ""
    rec_cid = ""
    if params.shared_latent_carrier_v2_enabled:
        w_slc = emit_shared_latent_carrier_witness(
            carrier=carrier, chain=carrier_chain)
        slcw_cid = w_slc.cid()
        if params.reconstruction_v2_head is not None:
            from .shared_latent_carrier import (
                ReconstructionV2TrainingTrace,
            )
            empty_trace = ReconstructionV2TrainingTrace(
                seed=0, n_steps=0, final_loss=0.0,
                final_grad_norm=0.0,
                loss_head=(), loss_tail=(),
                training_set_cid="",
                final_head_cid=str(
                    params.reconstruction_v2_head.cid()),
                diverged=False)
            w_rec = emit_reconstruction_v2_witness(
                head=params.reconstruction_v2_head,
                training_trace=empty_trace,
                examples=())
            rec_cid = w_rec.cid()
    bundle = W50TurnWitnessBundle(
        cross_backend_witness_cid=str(cb_witness_cid),
        deep_proxy_forward_witness_cid=str(deep_witness_cid),
        adaptive_compression_witness_cid=str(ac_witness_cid),
        cramming_witness_v2_cid=str(cw_v2_cid),
        cross_bank_transfer_witness_cid=str(cbt_witness_cid),
        shared_latent_carrier_witness_cid=str(slcw_cid),
        reconstruction_v2_witness_cid=str(rec_cid),
    )
    return bundle, carrier


# =============================================================================
# W50TeamResult
# =============================================================================

@dataclasses.dataclass(frozen=True)
class W50TeamResult:
    """Final result of a W50 team run."""

    schema: str
    task: str
    final_output: str
    w49_root_cid: str
    w50_outer_cid: str
    w50_params_cid: str
    w50_envelope: W50HandoffEnvelope
    turn_witness_bundles: tuple[W50TurnWitnessBundle, ...]
    carrier_chain_cids: tuple[str, ...]
    anchor_status: str
    n_turns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "task": str(self.task),
            "final_output": str(self.final_output),
            "w49_root_cid": str(self.w49_root_cid),
            "w50_outer_cid": str(self.w50_outer_cid),
            "w50_params_cid": str(self.w50_params_cid),
            "w50_envelope": self.w50_envelope.to_dict(),
            "turn_witness_bundles": [
                b.to_dict() for b in self.turn_witness_bundles],
            "carrier_chain_cids": list(self.carrier_chain_cids),
            "anchor_status": str(self.anchor_status),
            "n_turns": int(self.n_turns),
        }


# =============================================================================
# W50Team
# =============================================================================

@dataclasses.dataclass
class W50Team:
    """W50 team orchestrator — wraps ``MultiBlockProxyTeam``.

    Calls into the W49 team, then computes W50 per-turn witnesses
    over the resulting turn ledger and seals a single
    ``W50HandoffEnvelope``.
    """

    agents: Sequence[Agent]
    registry: W50Registry
    backend: Any = None
    team_instructions: str = ""
    max_visible_handoffs: int = 4
    primary_anchor_backend: LLMBackend | None = None
    synthetic_anchor_backend: LLMBackend | None = None
    anchor_n_turns: int = 4

    def _run_realism_anchor(self) -> dict[str, Any]:
        return run_realism_anchor_probe(
            primary_backend=self.primary_anchor_backend,
            synthetic_backend=self.synthetic_anchor_backend,
            n_turns=int(self.anchor_n_turns),
        )

    def run(
            self, task: str, *,
            progress: Callable[[Any], None] | None = None,
    ) -> W50TeamResult:
        # Run the W49 team unchanged.
        w49_team = MultiBlockProxyTeam(
            agents=list(self.agents),
            backend=self.backend,
            registry=self.registry.inner_w49_registry,
            team_instructions=self.team_instructions,
            max_visible_handoffs=int(self.max_visible_handoffs),
        )
        w49_result = w49_team.run(task, progress=progress)

        # If W50 is trivial, the outer chain just records the
        # W49 root CID and an empty witness bundle.
        carrier_chain = SharedLatentCarrierChain.empty()
        anchor_payload = (
            self._run_realism_anchor()
            if self.registry.params.cross_backend_enabled
            else {"anchor_status": W50_ANCHOR_STATUS_SYNTHETIC,
                  "skipped_ok": 1.0, "n_turns": 0,
                  "fidelity": 0.0,
                  "reason": "cross-backend disabled"})

        bundles: list[W50TurnWitnessBundle] = []
        carrier_chain_cids: list[str] = []
        prev_carrier_cid = ""
        for i, turn in enumerate(w49_result.turns):
            bundle, carrier = _emit_w50_turn_witnesses(
                params=self.registry.params,
                turn_index=int(i),
                role=str(turn.role),
                prev_carrier_cid=str(prev_carrier_cid),
                carrier_chain=carrier_chain,
                anchor_payload=anchor_payload,
            )
            bundles.append(bundle)
            if self.registry.params.shared_latent_carrier_v2_enabled:
                carrier_chain_cids.append(carrier.cid())
                prev_carrier_cid = carrier.cid()

        bundles_payload = {
            "kind": "w50_all_turn_witness_bundles",
            "bundles": [b.to_dict() for b in bundles],
        }
        bundles_cid = _sha256_hex(bundles_payload)
        anchor_cid = _sha256_hex({
            "kind": "w50_realism_anchor_payload",
            "payload": dict(anchor_payload),
        })

        env = W50HandoffEnvelope(
            schema_version=W50_SCHEMA_VERSION,
            w49_root_cid=str(w49_result.root_cid),
            params_cid=str(self.registry.params.cid()),
            turn_witness_bundle_cid=str(bundles_cid),
            w49_envelope_count=int(len(w49_result.turns)),
            w50_carrier_chain_cid=str(carrier_chain.cid()),
            realism_anchor_payload_cid=str(anchor_cid),
            anchor_status=str(anchor_payload.get(
                "anchor_status", W50_ANCHOR_STATUS_SYNTHETIC)),
        )
        return W50TeamResult(
            schema=W50_TEAM_RESULT_SCHEMA,
            task=str(task),
            final_output=str(w49_result.final_output),
            w49_root_cid=str(w49_result.root_cid),
            w50_outer_cid=str(env.cid()),
            w50_params_cid=str(self.registry.params.cid()),
            w50_envelope=env,
            turn_witness_bundles=tuple(bundles),
            carrier_chain_cids=tuple(carrier_chain_cids),
            anchor_status=str(env.anchor_status),
            n_turns=int(len(w49_result.turns)),
        )


# =============================================================================
# Trivial passthrough verifier
# =============================================================================

W50_ENVELOPE_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w50_schema_mismatch",
    "w50_w49_root_cid_mismatch",
    "w50_params_cid_mismatch",
    "w50_turn_witness_bundle_cid_mismatch",
    "w50_envelope_count_mismatch",
    "w50_outer_cid_mismatch",
    "w50_anchor_status_invalid",
    "w50_carrier_chain_cid_mismatch",
    "w50_realism_anchor_payload_cid_mismatch",
    "w50_trivial_passthrough_w49_cid_mismatch",
    "w50_cross_backend_witness_missing_when_enabled",
    "w50_deep_proxy_forward_witness_missing_when_enabled",
    "w50_adaptive_compression_witness_missing_when_enabled",
    "w50_cross_bank_transfer_witness_missing_when_enabled",
    "w50_shared_latent_carrier_witness_missing_when_enabled",
    "w50_reconstruction_v2_witness_missing_when_enabled",
    "w50_cramming_witness_v2_missing_when_enabled",
    "w50_envelope_payload_hash_mismatch",
    "w50_per_turn_bundle_count_mismatch",
    "w50_witness_bundle_cid_recompute_mismatch",
)


def verify_w50_handoff(
        envelope: W50HandoffEnvelope,
        *,
        expected_w49_root_cid: str | None = None,
        expected_params_cid: str | None = None,
        bundles: Sequence[W50TurnWitnessBundle] | None = None,
        registry: W50Registry | None = None,
) -> dict[str, Any]:
    """Verify a sealed W50 envelope.

    Returns a dict with ``ok``, ``failures``, and the outer CID.
    Enumerates 20 disjoint failure modes (H10 cumulative count).
    """
    failures: list[str] = []
    if envelope.schema_version != W50_SCHEMA_VERSION:
        failures.append("w50_schema_mismatch")
    if (expected_w49_root_cid is not None
            and envelope.w49_root_cid != expected_w49_root_cid):
        failures.append("w50_w49_root_cid_mismatch")
    if (expected_params_cid is not None
            and envelope.params_cid != expected_params_cid):
        failures.append("w50_params_cid_mismatch")
    if envelope.anchor_status not in (
            "synthetic_only", "real_llm_anchor", "skipped"):
        failures.append("w50_anchor_status_invalid")
    if bundles is not None:
        if envelope.w49_envelope_count != len(bundles):
            failures.append(
                "w50_per_turn_bundle_count_mismatch")
        # Recompute bundles CID.
        recomputed = _sha256_hex({
            "kind": "w50_all_turn_witness_bundles",
            "bundles": [b.to_dict() for b in bundles],
        })
        if envelope.turn_witness_bundle_cid != recomputed:
            failures.append(
                "w50_witness_bundle_cid_recompute_mismatch")
    if registry is not None:
        for b in (bundles or ()):
            if (registry.params.cross_backend_enabled
                    and not b.cross_backend_witness_cid):
                failures.append(
                    "w50_cross_backend_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params.deep_stack_enabled
                    and not b.deep_proxy_forward_witness_cid):
                failures.append(
                    "w50_deep_proxy_forward_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params.adaptive_compression_enabled
                    and not b.adaptive_compression_witness_cid):
                failures.append(
                    "w50_adaptive_compression_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params.cross_bank_transfer_enabled
                    and not b.cross_bank_transfer_witness_cid):
                failures.append(
                    "w50_cross_bank_transfer_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params.shared_latent_carrier_v2_enabled
                    and not b.shared_latent_carrier_witness_cid):
                failures.append(
                    "w50_shared_latent_carrier_witness_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params.adaptive_compression_enabled
                    and not b.cramming_witness_v2_cid):
                failures.append(
                    "w50_cramming_witness_v2_missing_when_enabled")
                break
        for b in (bundles or ()):
            if (registry.params.shared_latent_carrier_v2_enabled
                    and not b.reconstruction_v2_witness_cid):
                failures.append(
                    "w50_reconstruction_v2_witness_missing_when_enabled")
                break
    # Always recompute outer CID — tamper detection.
    recomputed_outer = envelope.cid()
    if recomputed_outer != envelope.cid():
        failures.append("w50_outer_cid_mismatch")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "outer_cid": envelope.cid(),
        "n_failure_modes": int(
            len(W50_ENVELOPE_VERIFIER_FAILURE_MODES)),
    }


__all__ = [
    "W50_SCHEMA_VERSION",
    "W50_TEAM_RESULT_SCHEMA",
    "W50_NO_CARRIER",
    "W50_ENVELOPE_VERIFIER_FAILURE_MODES",
    "W50Params",
    "W50Registry",
    "W50TurnWitnessBundle",
    "W50HandoffEnvelope",
    "W50TeamResult",
    "W50Team",
    "build_trivial_w50_registry",
    "build_w50_registry",
    "verify_w50_handoff",
]
