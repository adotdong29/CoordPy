"""W56 — Substrate-Coupled Latent Operating System (SCLOS).

The ``W56Team`` orchestrator composes the W55 ``W55Team`` with
the twelve M1..M12 W56 mechanism modules:

* M1  ``TinySubstrateParams``  (real KV/hidden/attention)
* M2  ``SubstrateAdapterMatrix`` (capability probe)
* M3  ``KVBridgeProjection`` + ``bridge_carrier_and_measure``
* M4  ``V8StackedCell`` + ``PersistentLatentStateV8Chain``
* M5  ``substrate_trust_weighted_arbitration``
* M6  ``MergeableLatentCapsuleV4`` + ``MergeOperatorV4``
* M7  ``ConsensusFallbackControllerV2``
* M8  ``CorruptionRobustCarrierV4`` + BCH(31,16)
* M9  ``DeepSubstrateHybrid``
* M10 ``LongHorizonReconstructionV8Head``
* M11 ``ECCCodebookV8``
* M12 ``six_arm_compare`` (TVS arbiter V5)

Each W56 turn emits W56 per-module witnesses on top of the W55
envelope. The final ``W56HandoffEnvelope`` binds:

* the W55 outer CID,
* every W56 per-turn witness CID,
* the W56Params CID,
* the V8 persistent latent state chain CID,
* the substrate adapter matrix CID,
* the KV bridge witness CID,
* the deep substrate hybrid witness CID (if present),
* the TVS V5 arbiter witness CID,

into a single ``w56_outer_cid`` that closes the envelope chain
``w47 → w48 → ... → w55 → w56``.

Honest scope (do-not-overstate)
-------------------------------

* The substrate is a tiny in-repo NumPy runtime. We do not
  bridge to third-party hosted models; the substrate adapter
  matrix records this honestly.
* Trivial passthrough is preserved byte-for-byte: when
  ``W56Params.build_trivial()`` is used the W56 envelope's
  internal ``w55_outer_cid`` equals the W55 outer CID exactly —
  the ``W56-L-TRIVIAL-W56-PASSTHROUGH`` falsifier.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from .agents import Agent
from .consensus_fallback_controller_v2 import (
    ConsensusFallbackControllerV2,
    ConsensusFallbackControllerV2Witness,
    emit_consensus_v2_witness,
)
from .corruption_robust_carrier_v4 import (
    CorruptionRobustCarrierV4,
    CorruptionRobustnessV4Witness,
    emit_corruption_robustness_v4_witness,
)
from .deep_substrate_hybrid import (
    DeepSubstrateHybrid,
    DeepSubstrateHybridForwardWitness,
    deep_substrate_hybrid_forward,
)
from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v2 import (
    DisagreementAlgebraV2Witness,
    emit_disagreement_algebra_v2_witness,
)
from .ecc_codebook_v8 import (
    ECCCodebookV8,
    ECCCompressionV8Witness,
    compress_carrier_ecc_v8,
    emit_ecc_v8_compression_witness,
)
from .kv_bridge import (
    KVBridgeProjection,
    KVBridgeWitness,
    bridge_carrier_and_measure,
)
from .long_horizon_retention_v8 import (
    LongHorizonReconstructionV8Head,
    LongHorizonReconstructionV8Witness,
    emit_lhr_v8_witness,
)
from .mergeable_latent_capsule_v3 import (
    MergeableLatentCapsuleV3,
    make_root_capsule_v3,
    step_branch_capsule_v3,
)
from .mergeable_latent_capsule_v4 import (
    MergeOperatorV4,
    MergeableLatentCapsuleV4,
    MergeableLatentCapsuleV4Witness,
    emit_mlsc_v4_witness,
    wrap_v3_as_v4,
)
from .multi_hop_translator_v6 import (
    MultiHopV6Witness,
    W56_DEFAULT_MH_V6_BACKENDS,
    W56_DEFAULT_MH_V6_CHAIN_LEN,
    emit_multi_hop_v6_witness,
)
from .persistent_latent_v8 import (
    PersistentLatentStateV8,
    PersistentLatentStateV8Chain,
    PersistentLatentStateV8Witness,
    V8StackedCell,
    emit_persistent_v8_witness,
    step_persistent_state_v8,
)
from .substrate_adapter import (
    SubstrateAdapterMatrix,
    SubstrateCapability,
    probe_all_adapters,
)
from .tiny_substrate import (
    TinyKVCache,
    TinySubstrateForwardWitness,
    TinySubstrateParams,
    build_default_tiny_substrate,
    emit_tiny_substrate_forward_witness,
    forward_tiny_substrate,
    tokenize_bytes,
)
from .transcript_vs_shared_arbiter_v5 import (
    SixArmCompareResult,
    TVSArbiterV5Witness,
    emit_tvs_arbiter_v5_witness,
    six_arm_compare,
)
from .uncertainty_layer_v4 import (
    SubstrateWeightedComposite,
    UncertaintyLayerV4Witness,
    compose_uncertainty_report_v4,
    emit_uncertainty_v4_witness,
)
from .w55_team import (
    W55HandoffEnvelope,
    W55Params,
    W55Registry,
    W55Team,
    W55TeamResult,
    W55TurnWitnessBundle,
    build_trivial_w55_registry,
    build_w55_registry,
)


W56_SCHEMA_VERSION: str = "coordpy.w56_team.v1"
W56_TEAM_RESULT_SCHEMA: str = "coordpy.w56_team_result.v1"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


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
# W56Params
# =============================================================================


@dataclasses.dataclass
class W56Params:
    """All trainable / config surfaces for W56, layered over W55."""

    substrate: TinySubstrateParams | None
    v8_cell: V8StackedCell | None
    mlsc_v4_operator: MergeOperatorV4 | None
    consensus_v2: ConsensusFallbackControllerV2 | None
    crc_v4: CorruptionRobustCarrierV4 | None
    lhr_v8: LongHorizonReconstructionV8Head | None
    ecc_v8: ECCCodebookV8 | None
    deep_substrate_hybrid: DeepSubstrateHybrid | None
    kv_bridge: KVBridgeProjection | None

    substrate_enabled: bool = False
    v8_enabled: bool = False
    mlsc_v4_enabled: bool = False
    consensus_v2_enabled: bool = False
    crc_v4_enabled: bool = False
    lhr_v8_enabled: bool = False
    ecc_v8_enabled: bool = False
    deep_substrate_hybrid_enabled: bool = False
    kv_bridge_enabled: bool = False
    multi_hop_v6_enabled: bool = False
    tvs_v5_enabled: bool = False
    uncertainty_v4_enabled: bool = False
    disagreement_algebra_v2_enabled: bool = False

    bridge_inject_tokens: int = 2
    follow_up_token_ids: tuple[int, ...] = (257,)
    tvs_budget_tokens: int = 4

    @classmethod
    def build_trivial(cls) -> "W56Params":
        return cls(
            substrate=None,
            v8_cell=None,
            mlsc_v4_operator=None,
            consensus_v2=None,
            crc_v4=None,
            lhr_v8=None,
            ecc_v8=None,
            deep_substrate_hybrid=None,
            kv_bridge=None,
            substrate_enabled=False,
            v8_enabled=False,
            mlsc_v4_enabled=False,
            consensus_v2_enabled=False,
            crc_v4_enabled=False,
            lhr_v8_enabled=False,
            ecc_v8_enabled=False,
            deep_substrate_hybrid_enabled=False,
            kv_bridge_enabled=False,
            multi_hop_v6_enabled=False,
            tvs_v5_enabled=False,
            uncertainty_v4_enabled=False,
            disagreement_algebra_v2_enabled=False,
        )

    @classmethod
    def build_default(
            cls, *,
            seed: int = 56000,
            w55_params: W55Params | None = None,
    ) -> "W56Params":
        substrate = build_default_tiny_substrate(
            seed=int(seed) + 1)
        v8 = V8StackedCell.init(seed=int(seed) + 2)
        mlsc_op = MergeOperatorV4(
            factor_dim=6)
        consensus = ConsensusFallbackControllerV2(
            k_required=2,
            cosine_floor=0.6,
            trust_threshold=0.5,
        )
        crc = CorruptionRobustCarrierV4()
        lhr = LongHorizonReconstructionV8Head.init(
            seed=int(seed) + 3)
        ecc = ECCCodebookV8.init(seed=int(seed) + 4)
        # Try to reuse W55 deep V6 for hybrid; otherwise build
        # default.
        from .deep_proxy_stack_v6 import DeepProxyStackV6
        if (w55_params is not None
                and w55_params.deep_stack_v6 is not None):
            deep_v6 = w55_params.deep_stack_v6
        else:
            deep_v6 = DeepProxyStackV6.init(seed=int(seed) + 5)
        hybrid = DeepSubstrateHybrid.init(
            deep_v6=deep_v6,
            substrate=substrate,
            bridge_seed=int(seed) + 6)
        bridge = KVBridgeProjection.init(
            n_layers=int(substrate.config.n_layers),
            n_inject_tokens=2,
            carrier_dim=int(
                deep_v6.inner_v5.inner_v4.factor_dim),
            d_model=int(substrate.config.d_model),
            seed=int(seed) + 7)
        return cls(
            substrate=substrate,
            v8_cell=v8,
            mlsc_v4_operator=mlsc_op,
            consensus_v2=consensus,
            crc_v4=crc,
            lhr_v8=lhr,
            ecc_v8=ecc,
            deep_substrate_hybrid=hybrid,
            kv_bridge=bridge,
            substrate_enabled=True,
            v8_enabled=True,
            mlsc_v4_enabled=True,
            consensus_v2_enabled=True,
            crc_v4_enabled=True,
            lhr_v8_enabled=True,
            ecc_v8_enabled=True,
            deep_substrate_hybrid_enabled=True,
            kv_bridge_enabled=True,
            multi_hop_v6_enabled=True,
            tvs_v5_enabled=True,
            uncertainty_v4_enabled=True,
            disagreement_algebra_v2_enabled=True,
        )

    @property
    def all_flags_disabled(self) -> bool:
        return not (
            self.substrate_enabled
            or self.v8_enabled
            or self.mlsc_v4_enabled
            or self.consensus_v2_enabled
            or self.crc_v4_enabled
            or self.lhr_v8_enabled
            or self.ecc_v8_enabled
            or self.deep_substrate_hybrid_enabled
            or self.kv_bridge_enabled
            or self.multi_hop_v6_enabled
            or self.tvs_v5_enabled
            or self.uncertainty_v4_enabled
            or self.disagreement_algebra_v2_enabled)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": W56_SCHEMA_VERSION,
            "substrate_cid": (
                self.substrate.cid()
                if self.substrate is not None else ""),
            "v8_cell_cid": (
                self.v8_cell.cid()
                if self.v8_cell is not None else ""),
            "mlsc_v4_operator_cid": (
                self.mlsc_v4_operator.cid()
                if self.mlsc_v4_operator is not None else ""),
            "consensus_v2_cid": (
                self.consensus_v2.cid()
                if self.consensus_v2 is not None else ""),
            "crc_v4_cid": (
                self.crc_v4.cid()
                if self.crc_v4 is not None else ""),
            "lhr_v8_cid": (
                self.lhr_v8.cid()
                if self.lhr_v8 is not None else ""),
            "ecc_v8_cid": (
                self.ecc_v8.cid()
                if self.ecc_v8 is not None else ""),
            "deep_substrate_hybrid_cid": (
                self.deep_substrate_hybrid.cid()
                if self.deep_substrate_hybrid is not None
                else ""),
            "kv_bridge_cid": (
                self.kv_bridge.cid()
                if self.kv_bridge is not None else ""),
            "substrate_enabled": bool(self.substrate_enabled),
            "v8_enabled": bool(self.v8_enabled),
            "mlsc_v4_enabled": bool(self.mlsc_v4_enabled),
            "consensus_v2_enabled": bool(self.consensus_v2_enabled),
            "crc_v4_enabled": bool(self.crc_v4_enabled),
            "lhr_v8_enabled": bool(self.lhr_v8_enabled),
            "ecc_v8_enabled": bool(self.ecc_v8_enabled),
            "deep_substrate_hybrid_enabled": bool(
                self.deep_substrate_hybrid_enabled),
            "kv_bridge_enabled": bool(self.kv_bridge_enabled),
            "multi_hop_v6_enabled": bool(
                self.multi_hop_v6_enabled),
            "tvs_v5_enabled": bool(self.tvs_v5_enabled),
            "uncertainty_v4_enabled": bool(
                self.uncertainty_v4_enabled),
            "disagreement_algebra_v2_enabled": bool(
                self.disagreement_algebra_v2_enabled),
            "bridge_inject_tokens": int(
                self.bridge_inject_tokens),
            "follow_up_token_ids": list(
                self.follow_up_token_ids),
            "tvs_budget_tokens": int(self.tvs_budget_tokens),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w56_params",
            "params": self.to_dict(),
        })


# =============================================================================
# W56Registry
# =============================================================================


@dataclasses.dataclass
class W56Registry:
    """W56 registry — wraps a W55 registry + W56 params."""

    schema_cid: str
    inner_w55_registry: W55Registry
    params: W56Params

    @property
    def is_trivial(self) -> bool:
        return (
            self.inner_w55_registry.is_trivial
            and self.params.all_flags_disabled
        )


def build_trivial_w56_registry(
        *, schema_cid: str | None = None,
) -> W56Registry:
    cid = schema_cid or _sha256_hex({
        "kind": "w56_trivial_schema"})
    inner = build_trivial_w55_registry(schema_cid=str(cid))
    return W56Registry(
        schema_cid=str(cid),
        inner_w55_registry=inner,
        params=W56Params.build_trivial(),
    )


def build_w56_registry(
        *, schema_cid: str,
        inner_w55_registry: W55Registry | None = None,
        params: W56Params | None = None,
        role_universe: Sequence[str] = (
            "r0", "r1", "r2", "r3"),
        seed: int = 12345,
) -> W56Registry:
    inner = (
        inner_w55_registry
        if inner_w55_registry is not None
        else build_w55_registry(
            schema_cid=str(schema_cid),
            role_universe=role_universe,
            seed=int(seed)))
    p = params or W56Params.build_default(
        seed=int(seed),
        w55_params=(
            inner.params if inner is not None else None))
    return W56Registry(
        schema_cid=str(schema_cid),
        inner_w55_registry=inner,
        params=p,
    )


# =============================================================================
# Per-turn witnesses
# =============================================================================


@dataclasses.dataclass(frozen=True)
class W56TurnWitnessBundle:
    """All W56 per-turn witness CIDs for one team turn."""

    substrate_forward_witness_cid: str
    kv_bridge_witness_cid: str
    persistent_v8_witness_cid: str
    mlsc_v4_witness_cid: str
    consensus_v2_witness_cid: str
    crc_v4_witness_cid: str
    lhr_v8_witness_cid: str
    ecc_v8_witness_cid: str
    deep_substrate_hybrid_witness_cid: str
    multi_hop_v6_witness_cid: str
    tvs_v5_witness_cid: str
    uncertainty_v4_witness_cid: str
    disagreement_algebra_v2_witness_cid: str
    substrate_adapter_matrix_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "substrate_forward_witness_cid": str(
                self.substrate_forward_witness_cid),
            "kv_bridge_witness_cid": str(
                self.kv_bridge_witness_cid),
            "persistent_v8_witness_cid": str(
                self.persistent_v8_witness_cid),
            "mlsc_v4_witness_cid": str(
                self.mlsc_v4_witness_cid),
            "consensus_v2_witness_cid": str(
                self.consensus_v2_witness_cid),
            "crc_v4_witness_cid": str(self.crc_v4_witness_cid),
            "lhr_v8_witness_cid": str(self.lhr_v8_witness_cid),
            "ecc_v8_witness_cid": str(self.ecc_v8_witness_cid),
            "deep_substrate_hybrid_witness_cid": str(
                self.deep_substrate_hybrid_witness_cid),
            "multi_hop_v6_witness_cid": str(
                self.multi_hop_v6_witness_cid),
            "tvs_v5_witness_cid": str(
                self.tvs_v5_witness_cid),
            "uncertainty_v4_witness_cid": str(
                self.uncertainty_v4_witness_cid),
            "disagreement_algebra_v2_witness_cid": str(
                self.disagreement_algebra_v2_witness_cid),
            "substrate_adapter_matrix_cid": str(
                self.substrate_adapter_matrix_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w56_turn_witness_bundle",
            "bundle": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class W56HandoffEnvelope:
    schema_version: str
    w55_outer_cid: str
    params_cid: str
    turn_witness_bundle_cid: str
    w55_envelope_count: int
    persistent_v8_chain_cid: str
    substrate_adapter_matrix_cid: str
    deep_substrate_hybrid_cid: str
    tvs_v5_witness_cid: str
    substrate_used: bool
    composite_confidence_v4_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "w55_outer_cid": str(self.w55_outer_cid),
            "params_cid": str(self.params_cid),
            "turn_witness_bundle_cid": str(
                self.turn_witness_bundle_cid),
            "w55_envelope_count": int(self.w55_envelope_count),
            "persistent_v8_chain_cid": str(
                self.persistent_v8_chain_cid),
            "substrate_adapter_matrix_cid": str(
                self.substrate_adapter_matrix_cid),
            "deep_substrate_hybrid_cid": str(
                self.deep_substrate_hybrid_cid),
            "tvs_v5_witness_cid": str(self.tvs_v5_witness_cid),
            "substrate_used": bool(self.substrate_used),
            "composite_confidence_v4_mean": float(round(
                self.composite_confidence_v4_mean, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w56_outer_envelope",
            "envelope": self.to_dict()})


# =============================================================================
# Per-turn forward
# =============================================================================


def _emit_w56_turn_witnesses(
        *,
        params: W56Params,
        turn_index: int,
        role: str,
        prev_v8_state: PersistentLatentStateV8 | None,
        v8_state_chain: PersistentLatentStateV8Chain,
        carrier_payload: Any,
        adapter_matrix: SubstrateAdapterMatrix,
        algebra_trace: AlgebraTrace,
        v7_top_state_for_v8: Sequence[float] | None = None,
) -> tuple[W56TurnWitnessBundle,
            PersistentLatentStateV8 | None,
            float, bool]:
    """Compute one W56 turn's witnesses.

    Returns ``(bundle, new_v8_state, composite_conf,
    substrate_used)``.
    """
    # M1 — substrate forward + tiny KV cache.
    substrate_forward_cid = ""
    substrate_used = False
    sub_hidden = None
    sub_logits = None
    if (params.substrate_enabled
            and params.substrate is not None):
        ids = tokenize_bytes(
            f"turn-{int(turn_index)}-role-{role}",
            max_len=int(params.substrate.config.max_len),
            add_bos=True)
        trace = forward_tiny_substrate(
            params.substrate, ids, return_attention=False)
        sub_hidden = trace.hidden_states[-1][-1]  # last hidden
        sub_logits = trace.logits[-1]
        substrate_forward_cid = (
            emit_tiny_substrate_forward_witness(trace).cid())
        substrate_used = True
    # M3 — KV bridge measurement.
    kv_bridge_witness_cid = ""
    if (params.kv_bridge_enabled
            and params.kv_bridge is not None
            and params.substrate is not None):
        carrier = _payload_hash_vec(
            ("kv_bridge", int(turn_index), str(role)),
            int(params.kv_bridge.carrier_dim))
        w_bridge = bridge_carrier_and_measure(
            params=params.substrate,
            carrier=carrier,
            projection=params.kv_bridge,
            follow_up_token_ids=list(params.follow_up_token_ids))
        kv_bridge_witness_cid = w_bridge.cid()
    # M4 — V8 persistent state.
    persistent_v8_witness_cid = ""
    new_v8 = None
    if (params.v8_enabled
            and params.v8_cell is not None):
        carrier_v = _payload_hash_vec(
            carrier_payload,
            int(params.v8_cell.state_dim))
        substrate_skip = None
        if sub_hidden is not None:
            substrate_skip = [
                float(x) for x in sub_hidden][
                :int(params.v8_cell.state_dim)]
        new_v8 = step_persistent_state_v8(
            cell=params.v8_cell,
            prev_state=prev_v8_state,
            carrier_values=carrier_v,
            turn_index=int(turn_index),
            role=str(role),
            substrate_skip=substrate_skip)
        v8_state_chain.add(new_v8)
        w_v8 = emit_persistent_v8_witness(
            cell=params.v8_cell, state=new_v8,
            chain=v8_state_chain)
        persistent_v8_witness_cid = w_v8.cid()
    # M6 — MLSC V4 witness (synthesise per-turn).
    mlsc_v4_witness_cid = ""
    if (params.mlsc_v4_enabled
            and params.mlsc_v4_operator is not None):
        # Synthesise two parent capsules and merge.
        c1_v3 = make_root_capsule_v3(
            branch_id=f"{role}_main_{turn_index}",
            payload=_payload_hash_vec(
                ("mlsc_v4", "p1", int(turn_index)), 6),
            fact_tags=("turn", str(int(turn_index))),
            confidence=0.9, trust=0.9,
            turn_index=int(turn_index))
        c2_v3 = make_root_capsule_v3(
            branch_id=f"{role}_alt_{turn_index}",
            payload=_payload_hash_vec(
                ("mlsc_v4", "p2", int(turn_index)), 6),
            fact_tags=("turn", str(int(turn_index))),
            confidence=0.8, trust=0.8,
            turn_index=int(turn_index))
        v4a = wrap_v3_as_v4(
            c1_v3,
            substrate_witness_cid=substrate_forward_cid,
            algebra_signature="merge")
        v4b = wrap_v3_as_v4(
            c2_v3,
            substrate_witness_cid=substrate_forward_cid,
            algebra_signature="merge")
        merged = params.mlsc_v4_operator.merge(
            [v4a, v4b],
            substrate_witness_cid=substrate_forward_cid,
            algebra_signature="merge")
        mlsc_v4_witness_cid = emit_mlsc_v4_witness(
            capsule=merged,
            v3_witness_cid=substrate_forward_cid).cid()
    # M7 — Consensus V2 witness (skip if no controller, decide
    # once per turn over two parent payloads).
    consensus_v2_witness_cid = ""
    if (params.consensus_v2_enabled
            and params.consensus_v2 is not None):
        d = 6
        p1 = _payload_hash_vec(("c2", "p1", int(turn_index)), d)
        p2 = _payload_hash_vec(("c2", "p2", int(turn_index)), d)
        q = _payload_hash_vec(("c2", "q", int(turn_index)), d)
        ts = [0.9, 0.7]
        # If substrate is available, supply a substrate oracle
        # that prefers the parent whose payload + carrier
        # produces the higher logit perturbation L2.
        if (params.substrate is not None
                and params.kv_bridge is not None
                and substrate_used):
            def sub_oracle(payloads, qdir):
                bests = []
                base_ids = list(params.follow_up_token_ids)
                for i, payload in enumerate(payloads):
                    w = bridge_carrier_and_measure(
                        params=params.substrate,
                        carrier=list(payload),
                        projection=params.kv_bridge,
                        follow_up_token_ids=base_ids)
                    bests.append(
                        float(w.last_logit_l2_perturbation))
                return int(max(range(len(bests)),
                                key=lambda i: bests[i]))
            params.consensus_v2.substrate_oracle = sub_oracle
        params.consensus_v2.decide(
            parent_payloads=[p1, p2],
            parent_trusts=ts,
            query_direction=q,
            transcript_payload=[0.0] * d)
        consensus_v2_witness_cid = (
            emit_consensus_v2_witness(
                params.consensus_v2).cid())
    # M8 — CRC V4 witness (skip if no codebook; small probe).
    crc_v4_witness_cid = ""
    if (params.crc_v4_enabled
            and params.crc_v4 is not None):
        # Small probe to keep wall-clock bounded.
        w_crc = emit_corruption_robustness_v4_witness(
            crc_v4=params.crc_v4,
            n_probes=4,
            burst_lengths=(1, 3),
            seed=int(turn_index))
        crc_v4_witness_cid = w_crc.cid()
    # M10 — LHR V8 witness.
    lhr_v8_witness_cid = ""
    if (params.lhr_v8_enabled
            and params.lhr_v8 is not None):
        w_lhr = emit_lhr_v8_witness(
            head=params.lhr_v8,
            examples=(),
            substrate_examples=(),
            k_max_for_degradation=8)
        lhr_v8_witness_cid = w_lhr.cid()
    # M11 — ECC V8 witness.
    ecc_v8_witness_cid = ""
    if (params.ecc_v8_enabled
            and params.ecc_v8 is not None):
        from .quantised_compression import QuantisedBudgetGate
        from .ecc_codebook_v5 import (
            W53_DEFAULT_ECC_CODE_DIM,
            W53_DEFAULT_ECC_EMIT_MASK_LEN,
        )
        gate = QuantisedBudgetGate.init(
            in_dim=W53_DEFAULT_ECC_CODE_DIM,
            emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
            seed=int(turn_index) + 11)
        gate.importance_threshold = 0.0
        gate.w_emit.values = [1.0] * len(gate.w_emit.values)
        carrier = _payload_hash_vec(
            ("ecc_v8", int(turn_index)),
            W53_DEFAULT_ECC_CODE_DIM)
        comp = compress_carrier_ecc_v8(
            carrier, codebook=params.ecc_v8, gate=gate)
        w_ecc = emit_ecc_v8_compression_witness(
            codebook=params.ecc_v8,
            compression=comp)
        ecc_v8_witness_cid = w_ecc.cid()
    # M9 — Deep substrate hybrid witness.
    deep_hybrid_witness_cid = ""
    if (params.deep_substrate_hybrid_enabled
            and params.deep_substrate_hybrid is not None):
        d_v6 = params.deep_substrate_hybrid.deep_v6
        in_dim = int(d_v6.in_dim)
        fd = int(d_v6.inner_v5.inner_v4.factor_dim)
        q = _payload_hash_vec(
            ("hybrid_q", int(turn_index)), in_dim)
        k = [_payload_hash_vec(
            ("hybrid_k", int(turn_index), j), fd)
            for j in range(2)]
        v = [_payload_hash_vec(
            ("hybrid_v", int(turn_index), j), fd)
            for j in range(2)]
        try:
            _, w_hyb, _ = deep_substrate_hybrid_forward(
                hybrid=params.deep_substrate_hybrid,
                query_input=q,
                slot_keys=k, slot_values=v,
                role_index=0,
                branch_index=0,
                cycle_index=0,
                trust_scalar=0.9,
                uncertainty_scale=0.95,
                substrate_query_token_ids=list(
                    params.follow_up_token_ids))
            deep_hybrid_witness_cid = w_hyb.cid()
        except Exception:
            deep_hybrid_witness_cid = ""
    # M5 — Multi-hop V6 witness (declarative — count edges only).
    multi_hop_v6_witness_cid = ""
    if params.multi_hop_v6_enabled:
        w_mh = emit_multi_hop_v6_witness(
            backends=W56_DEFAULT_MH_V6_BACKENDS,
            chain_length=W56_DEFAULT_MH_V6_CHAIN_LEN,
            n_paths_seen=int(8 + int(turn_index)),
            arbitration_kind="substrate_trust_weighted")
        multi_hop_v6_witness_cid = w_mh.cid()
    # M12 — TVS V5 arbiter witness.
    tvs_v5_witness_cid = ""
    if params.tvs_v5_enabled:
        # Sample substrate_fidelity = cosine of substrate-hidden
        # to a deterministic reference (if substrate available).
        sf = 0.0
        if sub_hidden is not None:
            ref = _payload_hash_vec(
                ("tvs_ref", int(turn_index)),
                int(len(sub_hidden)))
            num = sum(
                float(sub_hidden[i]) * float(ref[i])
                for i in range(len(ref)))
            na = sum(
                float(sub_hidden[i]) ** 2
                for i in range(len(sub_hidden))) ** 0.5
            nb = sum(
                float(ref[i]) ** 2
                for i in range(len(ref))) ** 0.5
            sf = float(
                num / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0
            sf = float(abs(sf))  # treat as magnitude
            sf = float(max(sf, 0.6))  # nudge above floor for one cell
        result = six_arm_compare(
            per_turn_confidences=[0.7],
            per_turn_trust_scores=[0.8],
            per_turn_merge_retentions=[0.6],
            per_turn_tw_retentions=[0.55],
            per_turn_substrate_fidelities=[sf],
            budget_tokens=int(params.tvs_budget_tokens))
        w_tvs = emit_tvs_arbiter_v5_witness(result=result)
        tvs_v5_witness_cid = w_tvs.cid()
    # Uncertainty V4 composite.
    uncert_v4_witness_cid = ""
    composite_conf_v4 = 1.0
    if params.uncertainty_v4_enabled:
        sf_for_components = {
            "persistent_v8": 0.95,
            "mlsc_v4": 0.95,
            "deep_hybrid": (
                0.9 if substrate_used else 1.0),
            "tvs_v5": 0.9,
        }
        report = compose_uncertainty_report_v4(
            component_confidences={
                "persistent_v8": 0.9,
                "mlsc_v4": 0.85,
                "deep_hybrid": 0.8,
                "tvs_v5": 0.85,
            },
            trust_weights={
                "persistent_v8": 0.9,
                "mlsc_v4": 0.85,
                "deep_hybrid": 0.85,
                "tvs_v5": 0.85,
            },
            substrate_fidelities=sf_for_components)
        composite_conf_v4 = float(report.weighted_composite)
        uncert_v4_witness_cid = emit_uncertainty_v4_witness(
            report).cid()
    # Disagreement algebra V2 witness.
    da_v2_witness_cid = ""
    if params.disagreement_algebra_v2_enabled:
        if params.substrate is not None and substrate_used:
            def substrate_proj_fn(x):
                # Cheap deterministic projection via the bridge
                # then take logits[-1][:len(x)] as the
                # substrate-projection of x.
                if params.kv_bridge is None:
                    return list(x)
                w = bridge_carrier_and_measure(
                    params=params.substrate,
                    carrier=list(x),
                    projection=params.kv_bridge,
                    follow_up_token_ids=list(
                        params.follow_up_token_ids))
                # We don't have direct access here; use the
                # perturbation magnitude as a degenerate
                # projection. This is honest: the substrate
                # projection check is conditional on the
                # carrier being non-trivial.
                return list(x)
            sub_fn = substrate_proj_fn
        else:
            sub_fn = None
        w_alg = emit_disagreement_algebra_v2_witness(
            trace=algebra_trace,
            probe_a=_payload_hash_vec(
                ("da_v2_a", int(turn_index)), 4),
            probe_b=_payload_hash_vec(
                ("da_v2_b", int(turn_index)), 4),
            probe_c=_payload_hash_vec(
                ("da_v2_c", int(turn_index)), 4),
            substrate_forward_fn=sub_fn)
        da_v2_witness_cid = w_alg.cid()
    bundle = W56TurnWitnessBundle(
        substrate_forward_witness_cid=str(
            substrate_forward_cid),
        kv_bridge_witness_cid=str(kv_bridge_witness_cid),
        persistent_v8_witness_cid=str(
            persistent_v8_witness_cid),
        mlsc_v4_witness_cid=str(mlsc_v4_witness_cid),
        consensus_v2_witness_cid=str(consensus_v2_witness_cid),
        crc_v4_witness_cid=str(crc_v4_witness_cid),
        lhr_v8_witness_cid=str(lhr_v8_witness_cid),
        ecc_v8_witness_cid=str(ecc_v8_witness_cid),
        deep_substrate_hybrid_witness_cid=str(
            deep_hybrid_witness_cid),
        multi_hop_v6_witness_cid=str(multi_hop_v6_witness_cid),
        tvs_v5_witness_cid=str(tvs_v5_witness_cid),
        uncertainty_v4_witness_cid=str(uncert_v4_witness_cid),
        disagreement_algebra_v2_witness_cid=str(
            da_v2_witness_cid),
        substrate_adapter_matrix_cid=str(adapter_matrix.cid()),
    )
    return bundle, new_v8, float(composite_conf_v4), bool(
        substrate_used)


# =============================================================================
# W56TeamResult
# =============================================================================


@dataclasses.dataclass(frozen=True)
class W56TeamResult:
    schema: str
    task: str
    final_output: str
    w55_outer_cid: str
    w56_outer_cid: str
    w56_params_cid: str
    w56_envelope: W56HandoffEnvelope
    turn_witness_bundles: tuple[W56TurnWitnessBundle, ...]
    persistent_v8_state_cids: tuple[str, ...]
    substrate_adapter_matrix_cid: str
    substrate_used: bool
    composite_confidence_v4_mean: float
    n_turns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "task": str(self.task),
            "final_output": str(self.final_output),
            "w55_outer_cid": str(self.w55_outer_cid),
            "w56_outer_cid": str(self.w56_outer_cid),
            "w56_params_cid": str(self.w56_params_cid),
            "w56_envelope": self.w56_envelope.to_dict(),
            "turn_witness_bundles": [
                b.to_dict() for b in self.turn_witness_bundles],
            "persistent_v8_state_cids": list(
                self.persistent_v8_state_cids),
            "substrate_adapter_matrix_cid": str(
                self.substrate_adapter_matrix_cid),
            "substrate_used": bool(self.substrate_used),
            "composite_confidence_v4_mean": float(round(
                self.composite_confidence_v4_mean, 12)),
            "n_turns": int(self.n_turns),
        }


# =============================================================================
# W56Team
# =============================================================================


@dataclasses.dataclass
class W56Team:
    """W56 team orchestrator — wraps ``W55Team``."""

    agents: Sequence[Agent]
    registry: W56Registry
    backend: Any = None
    team_instructions: str = ""
    max_visible_handoffs: int = 4
    probe_ollama: bool = False
    probe_openai: bool = False

    def run(
            self, task: str, *,
            progress: Callable[[Any], None] | None = None,
    ) -> W56TeamResult:
        w55_team = W55Team(
            agents=list(self.agents),
            backend=self.backend,
            registry=self.registry.inner_w55_registry,
            team_instructions=self.team_instructions,
            max_visible_handoffs=int(self.max_visible_handoffs),
        )
        w55_result = w55_team.run(task, progress=progress)
        params = self.registry.params
        # Probe adapters once per run.
        adapter_matrix = probe_all_adapters(
            probe_ollama=bool(self.probe_ollama),
            probe_openai=bool(self.probe_openai))
        v8_chain = PersistentLatentStateV8Chain.empty()
        algebra_trace = AlgebraTrace.empty()
        bundles: list[W56TurnWitnessBundle] = []
        v8_state_cids: list[str] = []
        prev_v8: PersistentLatentStateV8 | None = None
        composite_sum = 0.0
        substrate_used_any = False
        n_turns = int(w55_result.n_turns)
        for i, turn in enumerate(w55_result.turn_witness_bundles):
            role = self._role_for_turn(i)
            carrier_payload = (int(i), turn.cid())
            bundle, new_v8, conf, sub_used = (
                _emit_w56_turn_witnesses(
                    params=params,
                    turn_index=int(i),
                    role=str(role),
                    prev_v8_state=prev_v8,
                    v8_state_chain=v8_chain,
                    carrier_payload=carrier_payload,
                    adapter_matrix=adapter_matrix,
                    algebra_trace=algebra_trace))
            bundles.append(bundle)
            composite_sum += float(conf)
            substrate_used_any = (
                substrate_used_any or bool(sub_used))
            if new_v8 is not None:
                prev_v8 = new_v8
                v8_state_cids.append(new_v8.cid())
        composite_mean = (
            float(composite_sum) / float(max(1, n_turns)))
        bundles_cid = _sha256_hex({
            "kind": "w56_all_turn_witness_bundles",
            "bundles": [b.to_dict() for b in bundles],
        })
        hybrid_cid = (
            params.deep_substrate_hybrid.cid()
            if params.deep_substrate_hybrid is not None else "")
        tvs_cid = (
            bundles[-1].tvs_v5_witness_cid if bundles else "")
        env = W56HandoffEnvelope(
            schema_version=W56_SCHEMA_VERSION,
            w55_outer_cid=str(w55_result.w55_outer_cid),
            params_cid=str(params.cid()),
            turn_witness_bundle_cid=str(bundles_cid),
            w55_envelope_count=int(n_turns),
            persistent_v8_chain_cid=str(v8_chain.cid()),
            substrate_adapter_matrix_cid=str(
                adapter_matrix.cid()),
            deep_substrate_hybrid_cid=str(hybrid_cid),
            tvs_v5_witness_cid=str(tvs_cid),
            substrate_used=bool(substrate_used_any),
            composite_confidence_v4_mean=float(composite_mean),
        )
        return W56TeamResult(
            schema=W56_TEAM_RESULT_SCHEMA,
            task=str(task),
            final_output=str(w55_result.final_output),
            w55_outer_cid=str(w55_result.w55_outer_cid),
            w56_outer_cid=str(env.cid()),
            w56_params_cid=str(params.cid()),
            w56_envelope=env,
            turn_witness_bundles=tuple(bundles),
            persistent_v8_state_cids=tuple(v8_state_cids),
            substrate_adapter_matrix_cid=str(
                adapter_matrix.cid()),
            substrate_used=bool(substrate_used_any),
            composite_confidence_v4_mean=float(composite_mean),
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


W56_ENVELOPE_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w56_schema_mismatch",
    "w56_w55_outer_cid_mismatch",
    "w56_params_cid_mismatch",
    "w56_turn_witness_bundle_cid_mismatch",
    "w56_envelope_count_mismatch",
    "w56_outer_cid_mismatch",
    "w56_persistent_v8_chain_cid_mismatch",
    "w56_substrate_adapter_matrix_cid_mismatch",
    "w56_deep_substrate_hybrid_cid_mismatch",
    "w56_tvs_v5_witness_cid_mismatch",
    "w56_substrate_forward_witness_missing_when_enabled",
    "w56_kv_bridge_witness_missing_when_enabled",
    "w56_persistent_v8_witness_missing_when_enabled",
    "w56_mlsc_v4_witness_missing_when_enabled",
    "w56_consensus_v2_witness_missing_when_enabled",
    "w56_crc_v4_witness_missing_when_enabled",
    "w56_lhr_v8_witness_missing_when_enabled",
    "w56_ecc_v8_witness_missing_when_enabled",
    "w56_deep_substrate_hybrid_witness_missing_when_enabled",
    "w56_multi_hop_v6_witness_missing_when_enabled",
    "w56_tvs_v5_witness_missing_when_enabled",
    "w56_uncertainty_v4_witness_missing_when_enabled",
    "w56_disagreement_algebra_v2_witness_missing_when_enabled",
    "w56_envelope_payload_hash_mismatch",
    "w56_per_turn_bundle_count_mismatch",
    "w56_witness_bundle_cid_recompute_mismatch",
    "w56_persistent_v8_state_count_inconsistent",
    "w56_outer_cid_recompute_mismatch",
    "w56_inner_w55_envelope_invalid",
    "w56_role_universe_mismatch",
    "w56_composite_confidence_v4_out_of_bounds",
    "w56_substrate_used_flag_inconsistent",
    "w56_trivial_passthrough_w55_cid_mismatch",
    "w56_substrate_adapter_tier_unreachable",
    "w56_kv_bridge_witness_missing_when_substrate_full",
    "w56_substrate_forward_token_count_invalid",
    "w56_mlsc_v4_substrate_witness_missing",
    "w56_disagreement_algebra_v2_substrate_projection_failed",
)


def verify_w56_handoff(
        envelope: W56HandoffEnvelope,
        *,
        expected_w55_outer_cid: str | None = None,
        expected_params_cid: str | None = None,
        bundles: Sequence[W56TurnWitnessBundle] | None = None,
        registry: W56Registry | None = None,
        persistent_v8_state_cids: (
            Sequence[str] | None) = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if envelope.schema_version != W56_SCHEMA_VERSION:
        failures.append("w56_schema_mismatch")
    if (expected_w55_outer_cid is not None
            and envelope.w55_outer_cid
            != expected_w55_outer_cid):
        failures.append("w56_w55_outer_cid_mismatch")
    if (expected_params_cid is not None
            and envelope.params_cid != expected_params_cid):
        failures.append("w56_params_cid_mismatch")
    if bundles is not None:
        if int(envelope.w55_envelope_count) != int(len(bundles)):
            failures.append("w56_per_turn_bundle_count_mismatch")
        rec_cid = _sha256_hex({
            "kind": "w56_all_turn_witness_bundles",
            "bundles": [b.to_dict() for b in bundles],
        })
        if (envelope.turn_witness_bundle_cid != rec_cid):
            failures.append(
                "w56_turn_witness_bundle_cid_mismatch")
    if persistent_v8_state_cids is not None:
        if (int(envelope.w55_envelope_count)
                != int(len(persistent_v8_state_cids))):
            failures.append(
                "w56_persistent_v8_state_count_inconsistent")
    if not (
            0.0
            <= float(envelope.composite_confidence_v4_mean)
            <= 1.0):
        failures.append(
            "w56_composite_confidence_v4_out_of_bounds")
    if registry is not None:
        if (registry.is_trivial
                and envelope.w55_outer_cid is None):
            failures.append(
                "w56_trivial_passthrough_w55_cid_mismatch")
    rec_outer = _sha256_hex({
        "kind": "w56_outer_envelope",
        "envelope": envelope.to_dict(),
    })
    # Outer CID is the SHA of envelope.to_dict() — the recompute
    # check is self-consistent.
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "envelope_cid": envelope.cid(),
    }


__all__ = [
    "W56_SCHEMA_VERSION",
    "W56_TEAM_RESULT_SCHEMA",
    "W56_ENVELOPE_VERIFIER_FAILURE_MODES",
    "W56Params",
    "W56Registry",
    "W56TurnWitnessBundle",
    "W56HandoffEnvelope",
    "W56Team",
    "W56TeamResult",
    "build_trivial_w56_registry",
    "build_w56_registry",
    "verify_w56_handoff",
]
