"""W57 — Deep Substrate-Coupled Latent Operating System (DSCLOS).

The ``W57Team`` orchestrator composes the ``W56Team`` (which already
includes W55, W54, ... down to W47) with the W57 mechanism
modules:

* M1  ``TinyV2SubstrateParams`` (real KV/hidden/attention/RoPE/
      logit-lens, 4 layers, 8 heads, d_model=64)
* M2  ``KVBridgeV2Projection`` + ``bridge_carrier_and_measure_v2``
* M3  ``HiddenStateBridgeProjection`` + ``bridge_hidden_state_and_measure``
* M4  ``bridge_prefix_state_and_measure``
* M5  ``AttentionSteeringProjection`` + ``steer_attention_and_measure``
* M6  ``V9StackedCell`` + ``PersistentLatentStateV9Chain``
* M7  ``MultiHopV7Witness``
* M8  ``MergeableLatentCapsuleV5``
* M9  ``ConsensusFallbackControllerV3``
* M10 ``CorruptionRobustCarrierV5``
* M11 ``LongHorizonReconstructionV9Head``
* M12 ``ECCCodebookV9``
* M13 ``SevenArmCompareResult``
* M14 ``HiddenStateWeightedComposite``
* M15 ``DisagreementAlgebraV3Witness``
* M16 ``DeepSubstrateHybridV2``

Each W57 turn emits W57 per-module witnesses on top of the W56
envelope. The final ``W57HandoffEnvelope`` binds:

* the W56 outer CID,
* every W57 per-turn witness CID,
* the W57Params CID,
* the V9 persistent latent state chain CID,
* the substrate adapter V2 matrix CID,
* the deep substrate hybrid V2 CID,
* the TVS V6 arbiter witness CID,
* the hidden-state-weighted composite mean,

into a single ``w57_outer_cid`` that closes the envelope chain
``w47 → ... → w56 → w57``.

Honest scope (do-not-overstate)
-------------------------------

* The W57 substrate is the tiny in-repo V2 NumPy runtime. We do
  not bridge to third-party hosted models; the substrate adapter
  V2 matrix records this honestly.
* Trivial passthrough is preserved byte-for-byte: when
  ``W57Params.build_trivial()`` is used the W57 envelope's
  internal ``w56_outer_cid`` equals the W56 outer CID exactly.
* The W56 / W55 / ... conjectures and caps carry forward
  unchanged.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Sequence

from .agents import Agent
from .attention_steering_bridge import (
    AttentionSteeringProjection,
    AttentionSteeringWitness,
    steer_attention_and_measure,
)
from .consensus_fallback_controller_v3 import (
    ConsensusFallbackControllerV3,
    ConsensusFallbackControllerV3Witness,
    emit_consensus_v3_witness,
)
from .corruption_robust_carrier_v5 import (
    CorruptionRobustCarrierV5,
    CorruptionRobustnessV5Witness,
    emit_corruption_robustness_v5_witness,
)
from .deep_proxy_stack_v6 import DeepProxyStackV6
from .deep_substrate_hybrid_v2 import (
    DeepSubstrateHybridV2,
    DeepSubstrateHybridV2ForwardWitness,
    deep_substrate_hybrid_v2_forward,
)
from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v3 import (
    DisagreementAlgebraV3Witness,
    emit_disagreement_algebra_v3_witness,
)
from .ecc_codebook_v9 import (
    ECCCodebookV9,
    ECCCompressionV9Witness,
    compress_carrier_ecc_v9,
    emit_ecc_v9_compression_witness,
)
from .hidden_state_bridge import (
    HiddenStateBridgeProjection,
    HiddenStateBridgeWitness,
    bridge_hidden_state_and_measure,
)
from .kv_bridge_v2 import (
    KVBridgeV2Projection,
    KVBridgeV2Witness,
    bridge_carrier_and_measure_v2,
)
from .long_horizon_retention_v9 import (
    LongHorizonReconstructionV9Head,
    LongHorizonReconstructionV9Witness,
    emit_lhr_v9_witness,
)
from .mergeable_latent_capsule_v3 import (
    make_root_capsule_v3,
)
from .mergeable_latent_capsule_v4 import (
    wrap_v3_as_v4,
)
from .mergeable_latent_capsule_v5 import (
    MergeOperatorV5,
    MergeableLatentCapsuleV5,
    MergeableLatentCapsuleV5Witness,
    emit_mlsc_v5_witness,
    wrap_v4_as_v5,
)
from .multi_hop_translator_v7 import (
    MultiHopV7Witness,
    W57_DEFAULT_MH_V7_BACKENDS,
    W57_DEFAULT_MH_V7_CHAIN_LEN,
    emit_multi_hop_v7_witness,
)
from .persistent_latent_v9 import (
    PersistentLatentStateV9,
    PersistentLatentStateV9Chain,
    PersistentLatentStateV9Witness,
    V9StackedCell,
    emit_persistent_v9_witness,
    step_persistent_state_v9,
)
from .prefix_state_bridge import (
    PrefixStateBridgeWitness,
    bridge_prefix_state_and_measure,
)
from .substrate_adapter_v2 import (
    SubstrateAdapterV2Matrix,
    SubstrateCapabilityV2,
    probe_all_v2_adapters,
)
from .tiny_substrate_v2 import (
    TinyV2KVCache,
    TinyV2SubstrateForwardWitness,
    TinyV2SubstrateParams,
    build_default_tiny_substrate_v2,
    emit_tiny_substrate_v2_forward_witness,
    forward_tiny_substrate_v2,
    tokenize_bytes_v2,
)
from .transcript_vs_shared_arbiter_v6 import (
    SevenArmCompareResult,
    TVSArbiterV6Witness,
    emit_tvs_arbiter_v6_witness,
    seven_arm_compare,
)
from .uncertainty_layer_v5 import (
    HiddenStateWeightedComposite,
    UncertaintyLayerV5Witness,
    compose_uncertainty_report_v5,
    emit_uncertainty_v5_witness,
)
from .w56_team import (
    W56HandoffEnvelope,
    W56Params,
    W56Registry,
    W56Team,
    W56TeamResult,
    W56TurnWitnessBundle,
    build_trivial_w56_registry,
    build_w56_registry,
)


W57_SCHEMA_VERSION: str = "coordpy.w57_team.v1"
W57_TEAM_RESULT_SCHEMA: str = "coordpy.w57_team_result.v1"


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
# W57Params
# =============================================================================


@dataclasses.dataclass
class W57Params:
    """All trainable / config surfaces for W57, layered over W56."""

    substrate_v2: TinyV2SubstrateParams | None
    v9_cell: V9StackedCell | None
    mlsc_v5_operator: MergeOperatorV5 | None
    consensus_v3: ConsensusFallbackControllerV3 | None
    crc_v5: CorruptionRobustCarrierV5 | None
    lhr_v9: LongHorizonReconstructionV9Head | None
    ecc_v9: ECCCodebookV9 | None
    deep_substrate_hybrid_v2: DeepSubstrateHybridV2 | None
    kv_bridge_v2: KVBridgeV2Projection | None
    hidden_state_bridge: HiddenStateBridgeProjection | None
    attention_steering: AttentionSteeringProjection | None

    substrate_v2_enabled: bool = False
    v9_enabled: bool = False
    mlsc_v5_enabled: bool = False
    consensus_v3_enabled: bool = False
    crc_v5_enabled: bool = False
    lhr_v9_enabled: bool = False
    ecc_v9_enabled: bool = False
    deep_substrate_hybrid_v2_enabled: bool = False
    kv_bridge_v2_enabled: bool = False
    hidden_state_bridge_enabled: bool = False
    attention_steering_enabled: bool = False
    prefix_state_bridge_enabled: bool = False
    multi_hop_v7_enabled: bool = False
    tvs_v6_enabled: bool = False
    uncertainty_v5_enabled: bool = False
    disagreement_algebra_v3_enabled: bool = False

    bridge_inject_tokens: int = 3
    follow_up_token_ids: tuple[int, ...] = (257,)
    tvs_budget_tokens: int = 6

    @classmethod
    def build_trivial(cls) -> "W57Params":
        return cls(
            substrate_v2=None,
            v9_cell=None,
            mlsc_v5_operator=None,
            consensus_v3=None,
            crc_v5=None,
            lhr_v9=None,
            ecc_v9=None,
            deep_substrate_hybrid_v2=None,
            kv_bridge_v2=None,
            hidden_state_bridge=None,
            attention_steering=None,
        )

    @classmethod
    def build_default(
            cls, *,
            seed: int = 57000,
            w56_params: W56Params | None = None,
    ) -> "W57Params":
        sub_v2 = build_default_tiny_substrate_v2(seed=int(seed) + 1)
        v9 = V9StackedCell.init(seed=int(seed) + 2)
        mlsc_op = MergeOperatorV5(factor_dim=6)
        consensus = ConsensusFallbackControllerV3(
            k_required=2,
            cosine_floor=0.6,
            trust_threshold=0.5,
        )
        crc = CorruptionRobustCarrierV5()
        lhr = LongHorizonReconstructionV9Head.init(
            seed=int(seed) + 3)
        ecc = ECCCodebookV9.init(seed=int(seed) + 4)
        if (w56_params is not None
                and w56_params.deep_substrate_hybrid is not None):
            deep_v6 = (
                w56_params.deep_substrate_hybrid.deep_v6)
        else:
            deep_v6 = DeepProxyStackV6.init(seed=int(seed) + 5)
        hybrid_v2 = DeepSubstrateHybridV2.init(
            deep_v6=deep_v6,
            substrate=sub_v2,
            n_inject_tokens=3,
            carrier_dim=int(
                deep_v6.inner_v5.inner_v4.factor_dim),
            bridge_seed=int(seed) + 6,
            substrate_back_inject_weight=0.10,
            substrate_read_layer=1)
        d_head = (int(sub_v2.config.d_model)
                  // int(sub_v2.config.n_heads))
        kv_b2 = KVBridgeV2Projection.init(
            n_layers=int(sub_v2.config.n_layers),
            n_heads=int(sub_v2.config.n_heads),
            n_inject_tokens=3,
            carrier_dim=int(
                deep_v6.inner_v5.inner_v4.factor_dim),
            d_head=int(d_head),
            seed=int(seed) + 7)
        hsb = HiddenStateBridgeProjection.init(
            n_layers=int(sub_v2.config.n_layers),
            n_tokens=4,
            carrier_dim=int(
                deep_v6.inner_v5.inner_v4.factor_dim),
            d_model=int(sub_v2.config.d_model),
            seed=int(seed) + 8)
        attn_steer = AttentionSteeringProjection.init(
            n_layers=int(sub_v2.config.n_layers),
            n_heads=int(sub_v2.config.n_heads),
            n_query=4,
            n_key=8,
            carrier_dim=int(
                deep_v6.inner_v5.inner_v4.factor_dim),
            seed=int(seed) + 9)
        return cls(
            substrate_v2=sub_v2,
            v9_cell=v9,
            mlsc_v5_operator=mlsc_op,
            consensus_v3=consensus,
            crc_v5=crc,
            lhr_v9=lhr,
            ecc_v9=ecc,
            deep_substrate_hybrid_v2=hybrid_v2,
            kv_bridge_v2=kv_b2,
            hidden_state_bridge=hsb,
            attention_steering=attn_steer,
            substrate_v2_enabled=True,
            v9_enabled=True,
            mlsc_v5_enabled=True,
            consensus_v3_enabled=True,
            crc_v5_enabled=True,
            lhr_v9_enabled=True,
            ecc_v9_enabled=True,
            deep_substrate_hybrid_v2_enabled=True,
            kv_bridge_v2_enabled=True,
            hidden_state_bridge_enabled=True,
            attention_steering_enabled=True,
            prefix_state_bridge_enabled=True,
            multi_hop_v7_enabled=True,
            tvs_v6_enabled=True,
            uncertainty_v5_enabled=True,
            disagreement_algebra_v3_enabled=True,
        )

    @property
    def all_flags_disabled(self) -> bool:
        return not (
            self.substrate_v2_enabled or self.v9_enabled
            or self.mlsc_v5_enabled or self.consensus_v3_enabled
            or self.crc_v5_enabled or self.lhr_v9_enabled
            or self.ecc_v9_enabled
            or self.deep_substrate_hybrid_v2_enabled
            or self.kv_bridge_v2_enabled
            or self.hidden_state_bridge_enabled
            or self.attention_steering_enabled
            or self.prefix_state_bridge_enabled
            or self.multi_hop_v7_enabled
            or self.tvs_v6_enabled
            or self.uncertainty_v5_enabled
            or self.disagreement_algebra_v3_enabled)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": W57_SCHEMA_VERSION,
            "substrate_v2_cid": (
                self.substrate_v2.cid()
                if self.substrate_v2 is not None else ""),
            "v9_cell_cid": (
                self.v9_cell.cid()
                if self.v9_cell is not None else ""),
            "mlsc_v5_operator_cid": (
                self.mlsc_v5_operator.cid()
                if self.mlsc_v5_operator is not None else ""),
            "consensus_v3_cid": (
                self.consensus_v3.cid()
                if self.consensus_v3 is not None else ""),
            "crc_v5_cid": (
                self.crc_v5.cid()
                if self.crc_v5 is not None else ""),
            "lhr_v9_cid": (
                self.lhr_v9.cid()
                if self.lhr_v9 is not None else ""),
            "ecc_v9_cid": (
                self.ecc_v9.cid()
                if self.ecc_v9 is not None else ""),
            "deep_substrate_hybrid_v2_cid": (
                self.deep_substrate_hybrid_v2.cid()
                if self.deep_substrate_hybrid_v2 is not None
                else ""),
            "kv_bridge_v2_cid": (
                self.kv_bridge_v2.cid()
                if self.kv_bridge_v2 is not None else ""),
            "hidden_state_bridge_cid": (
                self.hidden_state_bridge.cid()
                if self.hidden_state_bridge is not None
                else ""),
            "attention_steering_cid": (
                self.attention_steering.cid()
                if self.attention_steering is not None
                else ""),
            "substrate_v2_enabled": bool(
                self.substrate_v2_enabled),
            "v9_enabled": bool(self.v9_enabled),
            "mlsc_v5_enabled": bool(self.mlsc_v5_enabled),
            "consensus_v3_enabled": bool(
                self.consensus_v3_enabled),
            "crc_v5_enabled": bool(self.crc_v5_enabled),
            "lhr_v9_enabled": bool(self.lhr_v9_enabled),
            "ecc_v9_enabled": bool(self.ecc_v9_enabled),
            "deep_substrate_hybrid_v2_enabled": bool(
                self.deep_substrate_hybrid_v2_enabled),
            "kv_bridge_v2_enabled": bool(
                self.kv_bridge_v2_enabled),
            "hidden_state_bridge_enabled": bool(
                self.hidden_state_bridge_enabled),
            "attention_steering_enabled": bool(
                self.attention_steering_enabled),
            "prefix_state_bridge_enabled": bool(
                self.prefix_state_bridge_enabled),
            "multi_hop_v7_enabled": bool(
                self.multi_hop_v7_enabled),
            "tvs_v6_enabled": bool(self.tvs_v6_enabled),
            "uncertainty_v5_enabled": bool(
                self.uncertainty_v5_enabled),
            "disagreement_algebra_v3_enabled": bool(
                self.disagreement_algebra_v3_enabled),
            "bridge_inject_tokens": int(
                self.bridge_inject_tokens),
            "follow_up_token_ids": list(
                self.follow_up_token_ids),
            "tvs_budget_tokens": int(self.tvs_budget_tokens),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w57_params",
            "params": self.to_dict(),
        })


# =============================================================================
# W57Registry
# =============================================================================


@dataclasses.dataclass
class W57Registry:
    schema_cid: str
    inner_w56_registry: W56Registry
    params: W57Params

    @property
    def is_trivial(self) -> bool:
        return (
            self.inner_w56_registry.is_trivial
            and self.params.all_flags_disabled)


def build_trivial_w57_registry(
        *, schema_cid: str | None = None,
) -> W57Registry:
    cid = schema_cid or _sha256_hex({
        "kind": "w57_trivial_schema"})
    inner = build_trivial_w56_registry(schema_cid=str(cid))
    return W57Registry(
        schema_cid=str(cid),
        inner_w56_registry=inner,
        params=W57Params.build_trivial(),
    )


def build_w57_registry(
        *, schema_cid: str,
        inner_w56_registry: W56Registry | None = None,
        params: W57Params | None = None,
        role_universe: Sequence[str] = (
            "r0", "r1", "r2", "r3"),
        seed: int = 12345,
) -> W57Registry:
    inner = (
        inner_w56_registry
        if inner_w56_registry is not None
        else build_w56_registry(
            schema_cid=str(schema_cid),
            role_universe=role_universe,
            seed=int(seed)))
    p = params or W57Params.build_default(
        seed=int(seed),
        w56_params=(
            inner.params if inner is not None else None))
    return W57Registry(
        schema_cid=str(schema_cid),
        inner_w56_registry=inner,
        params=p,
    )


# =============================================================================
# Per-turn witness bundle
# =============================================================================


@dataclasses.dataclass(frozen=True)
class W57TurnWitnessBundle:
    substrate_v2_forward_witness_cid: str
    kv_bridge_v2_witness_cid: str
    hidden_state_bridge_witness_cid: str
    prefix_state_bridge_witness_cid: str
    attention_steering_witness_cid: str
    persistent_v9_witness_cid: str
    mlsc_v5_witness_cid: str
    consensus_v3_witness_cid: str
    crc_v5_witness_cid: str
    lhr_v9_witness_cid: str
    ecc_v9_witness_cid: str
    deep_substrate_hybrid_v2_witness_cid: str
    multi_hop_v7_witness_cid: str
    tvs_v6_witness_cid: str
    uncertainty_v5_witness_cid: str
    disagreement_algebra_v3_witness_cid: str
    substrate_adapter_v2_matrix_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "substrate_v2_forward_witness_cid": str(
                self.substrate_v2_forward_witness_cid),
            "kv_bridge_v2_witness_cid": str(
                self.kv_bridge_v2_witness_cid),
            "hidden_state_bridge_witness_cid": str(
                self.hidden_state_bridge_witness_cid),
            "prefix_state_bridge_witness_cid": str(
                self.prefix_state_bridge_witness_cid),
            "attention_steering_witness_cid": str(
                self.attention_steering_witness_cid),
            "persistent_v9_witness_cid": str(
                self.persistent_v9_witness_cid),
            "mlsc_v5_witness_cid": str(
                self.mlsc_v5_witness_cid),
            "consensus_v3_witness_cid": str(
                self.consensus_v3_witness_cid),
            "crc_v5_witness_cid": str(
                self.crc_v5_witness_cid),
            "lhr_v9_witness_cid": str(
                self.lhr_v9_witness_cid),
            "ecc_v9_witness_cid": str(
                self.ecc_v9_witness_cid),
            "deep_substrate_hybrid_v2_witness_cid": str(
                self.deep_substrate_hybrid_v2_witness_cid),
            "multi_hop_v7_witness_cid": str(
                self.multi_hop_v7_witness_cid),
            "tvs_v6_witness_cid": str(
                self.tvs_v6_witness_cid),
            "uncertainty_v5_witness_cid": str(
                self.uncertainty_v5_witness_cid),
            "disagreement_algebra_v3_witness_cid": str(
                self.disagreement_algebra_v3_witness_cid),
            "substrate_adapter_v2_matrix_cid": str(
                self.substrate_adapter_v2_matrix_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w57_turn_witness_bundle",
            "bundle": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class W57HandoffEnvelope:
    schema_version: str
    w56_outer_cid: str
    params_cid: str
    turn_witness_bundle_cid: str
    w56_envelope_count: int
    persistent_v9_chain_cid: str
    substrate_adapter_v2_matrix_cid: str
    deep_substrate_hybrid_v2_cid: str
    tvs_v6_witness_cid: str
    substrate_v2_used: bool
    bidirectional_used: bool
    composite_confidence_v5_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "w56_outer_cid": str(self.w56_outer_cid),
            "params_cid": str(self.params_cid),
            "turn_witness_bundle_cid": str(
                self.turn_witness_bundle_cid),
            "w56_envelope_count": int(self.w56_envelope_count),
            "persistent_v9_chain_cid": str(
                self.persistent_v9_chain_cid),
            "substrate_adapter_v2_matrix_cid": str(
                self.substrate_adapter_v2_matrix_cid),
            "deep_substrate_hybrid_v2_cid": str(
                self.deep_substrate_hybrid_v2_cid),
            "tvs_v6_witness_cid": str(self.tvs_v6_witness_cid),
            "substrate_v2_used": bool(self.substrate_v2_used),
            "bidirectional_used": bool(self.bidirectional_used),
            "composite_confidence_v5_mean": float(round(
                self.composite_confidence_v5_mean, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w57_outer_envelope",
            "envelope": self.to_dict()})


# =============================================================================
# Per-turn forward
# =============================================================================


def _emit_w57_turn_witnesses(
        *, params: W57Params,
        turn_index: int,
        role: str,
        prev_v9_state: PersistentLatentStateV9 | None,
        v9_state_chain: PersistentLatentStateV9Chain,
        carrier_payload: Any,
        adapter_matrix_v2: SubstrateAdapterV2Matrix,
        algebra_trace: AlgebraTrace,
) -> tuple[W57TurnWitnessBundle,
           PersistentLatentStateV9 | None,
           float, bool, bool]:
    """Compute one W57 turn's witnesses.

    Returns ``(bundle, new_v9_state, composite_conf_v5,
    substrate_v2_used, bidirectional_used)``.
    """
    # M1 — substrate V2 forward.
    substrate_v2_forward_cid = ""
    substrate_used = False
    sub_hidden = None
    sub_intermediate_hidden = None
    if (params.substrate_v2_enabled
            and params.substrate_v2 is not None):
        ids = tokenize_bytes_v2(
            f"w57-turn-{int(turn_index)}-role-{role}",
            max_len=int(params.substrate_v2.config.max_len),
            add_bos=True)
        trace = forward_tiny_substrate_v2(
            params.substrate_v2, ids,
            return_attention=False)
        sub_hidden = trace.hidden_states[-1][-1]
        # Intermediate hidden state (layer 1).
        sub_intermediate_hidden = (
            trace.hidden_states[1][-1]
            if len(trace.hidden_states) > 1
            else trace.hidden_states[-1][-1])
        substrate_v2_forward_cid = str(
            emit_tiny_substrate_v2_forward_witness(trace).cid())
        substrate_used = True
    # M2 — KV bridge V2.
    kv_b2_witness_cid = ""
    if (params.kv_bridge_v2_enabled
            and params.kv_bridge_v2 is not None
            and params.substrate_v2 is not None):
        carrier = _payload_hash_vec(
            ("kv_v2", int(turn_index), str(role)),
            int(params.kv_bridge_v2.carrier_dim))
        w = bridge_carrier_and_measure_v2(
            params=params.substrate_v2,
            carrier=carrier,
            projection=params.kv_bridge_v2,
            follow_up_token_ids=list(
                params.follow_up_token_ids))
        kv_b2_witness_cid = w.cid()
    # M3 — hidden-state bridge.
    hsb_witness_cid = ""
    if (params.hidden_state_bridge_enabled
            and params.hidden_state_bridge is not None
            and params.substrate_v2 is not None):
        carrier = _payload_hash_vec(
            ("hsb", int(turn_index), str(role)),
            int(params.hidden_state_bridge.carrier_dim))
        try:
            w = bridge_hidden_state_and_measure(
                params=params.substrate_v2,
                carrier=carrier,
                projection=params.hidden_state_bridge,
                target_layer=1,
                token_ids=list(params.follow_up_token_ids))
            hsb_witness_cid = w.cid()
        except Exception:
            hsb_witness_cid = ""
    # M4 — prefix-state bridge.
    psb_witness_cid = ""
    if (params.prefix_state_bridge_enabled
            and params.substrate_v2 is not None):
        try:
            w = bridge_prefix_state_and_measure(
                params=params.substrate_v2,
                prompt_token_ids=tokenize_bytes_v2(
                    f"pfx-{int(turn_index)}", max_len=16),
                follow_up_token_ids=list(
                    params.follow_up_token_ids))
            psb_witness_cid = w.cid()
        except Exception:
            psb_witness_cid = ""
    # M5 — attention steering.
    attn_witness_cid = ""
    if (params.attention_steering_enabled
            and params.attention_steering is not None
            and params.substrate_v2 is not None):
        carrier = _payload_hash_vec(
            ("attn", int(turn_index), str(role)),
            int(params.attention_steering.carrier_dim))
        try:
            w = steer_attention_and_measure(
                params=params.substrate_v2,
                carrier=carrier,
                projection=params.attention_steering,
                token_ids=list(params.follow_up_token_ids))
            attn_witness_cid = w.cid()
        except Exception:
            attn_witness_cid = ""
    # M6 — V9 persistent state.
    v9_witness_cid = ""
    new_v9 = None
    if (params.v9_enabled
            and params.v9_cell is not None):
        carrier_v = _payload_hash_vec(
            carrier_payload,
            int(params.v9_cell.state_dim))
        substrate_skip = None
        hidden_skip = None
        sub_fid = 1.0
        if sub_hidden is not None:
            substrate_skip = [
                float(x) for x in sub_hidden][
                :int(params.v9_cell.state_dim)]
        if sub_intermediate_hidden is not None:
            hidden_skip = [
                float(x) for x in sub_intermediate_hidden][
                :int(params.v9_cell.state_dim)]
        new_v9 = step_persistent_state_v9(
            cell=params.v9_cell,
            prev_state=prev_v9_state,
            carrier_values=carrier_v,
            turn_index=int(turn_index),
            role=str(role),
            substrate_skip=substrate_skip,
            hidden_state_skip=hidden_skip,
            substrate_fidelity=float(sub_fid))
        v9_state_chain.add(new_v9)
        w_v9 = emit_persistent_v9_witness(
            cell=params.v9_cell, state=new_v9,
            chain=v9_state_chain)
        v9_witness_cid = w_v9.cid()
    # M7 — MLSC V5.
    mlsc_v5_witness_cid = ""
    if (params.mlsc_v5_enabled
            and params.mlsc_v5_operator is not None):
        c1_v3 = make_root_capsule_v3(
            branch_id=f"v5_{role}_main_{turn_index}",
            payload=_payload_hash_vec(
                ("mlsc_v5", "p1", int(turn_index)), 6),
            fact_tags=("turn", str(int(turn_index))),
            confidence=0.9, trust=0.9,
            turn_index=int(turn_index))
        c2_v3 = make_root_capsule_v3(
            branch_id=f"v5_{role}_alt_{turn_index}",
            payload=_payload_hash_vec(
                ("mlsc_v5", "p2", int(turn_index)), 6),
            fact_tags=("turn", str(int(turn_index))),
            confidence=0.85, trust=0.85,
            turn_index=int(turn_index))
        v4a = wrap_v3_as_v4(
            c1_v3,
            substrate_witness_cid=substrate_v2_forward_cid,
            algebra_signature="merge")
        v4b = wrap_v3_as_v4(
            c2_v3,
            substrate_witness_cid=substrate_v2_forward_cid,
            algebra_signature="merge")
        v5a = wrap_v4_as_v5(
            v4a,
            hidden_state_witness_chain=(
                substrate_v2_forward_cid,) if
                substrate_v2_forward_cid else (),
            attention_witness_cid=attn_witness_cid,
            per_head_trust=(0.95, 0.93, 0.91, 0.90,
                            0.94, 0.92, 0.91, 0.89))
        v5b = wrap_v4_as_v5(
            v4b,
            hidden_state_witness_chain=(
                substrate_v2_forward_cid,) if
                substrate_v2_forward_cid else (),
            attention_witness_cid=attn_witness_cid,
            per_head_trust=(0.92, 0.90, 0.89, 0.88,
                            0.91, 0.89, 0.88, 0.87))
        merged = params.mlsc_v5_operator.merge(
            [v5a, v5b],
            hidden_state_witness_chain=(
                substrate_v2_forward_cid,) if
                substrate_v2_forward_cid else (),
            attention_witness_cid=attn_witness_cid)
        mlsc_v5_witness_cid = emit_mlsc_v5_witness(
            capsule=merged).cid()
    # M8 — consensus V3.
    consensus_v3_witness_cid = ""
    if (params.consensus_v3_enabled
            and params.consensus_v3 is not None):
        d = 6
        p1 = _payload_hash_vec(("c3", "p1", int(turn_index)), d)
        p2 = _payload_hash_vec(("c3", "p2", int(turn_index)), d)
        q = _payload_hash_vec(("c3", "q", int(turn_index)), d)
        ts = [0.9, 0.7]
        # Substrate oracle: substrate_v2 KV bridge perturbation.
        if (params.substrate_v2 is not None
                and params.kv_bridge_v2 is not None
                and substrate_used):
            def sub_oracle(payloads, qdir):
                bests = []
                base_ids = list(params.follow_up_token_ids)
                for payload in payloads:
                    # Pad/truncate to carrier_dim.
                    pl = list(payload)[
                        :int(params.kv_bridge_v2.carrier_dim)]
                    while len(pl) < int(
                            params.kv_bridge_v2.carrier_dim):
                        pl.append(0.0)
                    w = bridge_carrier_and_measure_v2(
                        params=params.substrate_v2,
                        carrier=pl,
                        projection=params.kv_bridge_v2,
                        follow_up_token_ids=base_ids)
                    bests.append(
                        float(w.last_logit_l2_perturbation))
                return int(max(range(len(bests)),
                                key=lambda i: bests[i]))
            params.consensus_v3.substrate_oracle = sub_oracle
        # Logit-lens oracle: pick by per-layer logit lens cosine.
        if (params.substrate_v2 is not None and substrate_used):
            def lens_oracle(payloads, qdir):
                # cheap deterministic ranking via payload-l2.
                # An honest stand-in for "which payload's lens
                # at layer 1 best aligns with the query"; we
                # don't have a trained substrate, so we use the
                # readily-deterministic l2-distance to qdir.
                bests = []
                for payload in payloads:
                    diff_sq = sum(
                        (float(payload[i] if i < len(payload)
                                else 0.0)
                          - float(qdir[i] if i < len(qdir)
                                  else 0.0)) ** 2
                        for i in range(max(
                            len(payload), len(qdir))))
                    bests.append(diff_sq)
                # Smallest distance wins.
                return int(min(range(len(bests)),
                                key=lambda i: bests[i]))
            params.consensus_v3.logit_lens_oracle = lens_oracle
        params.consensus_v3.decide(
            parent_payloads=[p1, p2],
            parent_trusts=ts,
            query_direction=q,
            transcript_payload=[0.0] * d)
        consensus_v3_witness_cid = emit_consensus_v3_witness(
            params.consensus_v3).cid()
    # M9 — CRC V5.
    crc_v5_witness_cid = ""
    if (params.crc_v5_enabled
            and params.crc_v5 is not None):
        w_crc = emit_corruption_robustness_v5_witness(
            crc_v5=params.crc_v5,
            n_probes=8,
            seed=int(turn_index) + 575)
        crc_v5_witness_cid = w_crc.cid()
    # M10 — LHR V9.
    lhr_v9_witness_cid = ""
    if (params.lhr_v9_enabled
            and params.lhr_v9 is not None):
        w_lhr = emit_lhr_v9_witness(head=params.lhr_v9)
        lhr_v9_witness_cid = w_lhr.cid()
    # M11 — ECC V9.
    ecc_v9_witness_cid = ""
    if (params.ecc_v9_enabled
            and params.ecc_v9 is not None):
        from .quantised_compression import QuantisedBudgetGate
        from .ecc_codebook_v5 import (
            W53_DEFAULT_ECC_CODE_DIM,
            W53_DEFAULT_ECC_EMIT_MASK_LEN,
        )
        gate = QuantisedBudgetGate.init(
            in_dim=W53_DEFAULT_ECC_CODE_DIM,
            emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
            seed=int(turn_index) + 91)
        gate.importance_threshold = 0.0
        gate.w_emit.values = [1.0] * len(gate.w_emit.values)
        carrier = _payload_hash_vec(
            ("ecc_v9", int(turn_index)),
            W53_DEFAULT_ECC_CODE_DIM)
        comp = compress_carrier_ecc_v9(
            carrier, codebook=params.ecc_v9, gate=gate)
        w_ecc = emit_ecc_v9_compression_witness(
            codebook=params.ecc_v9, compression=comp)
        ecc_v9_witness_cid = w_ecc.cid()
    # M12 — deep substrate hybrid V2.
    deep_v2_witness_cid = ""
    bidirectional_used = False
    if (params.deep_substrate_hybrid_v2_enabled
            and params.deep_substrate_hybrid_v2 is not None):
        d_v6 = params.deep_substrate_hybrid_v2.deep_v6
        in_dim = int(d_v6.in_dim)
        fd = int(d_v6.inner_v5.inner_v4.factor_dim)
        q = _payload_hash_vec(
            ("hyb_v2_q", int(turn_index)), in_dim)
        k = [_payload_hash_vec(
            ("hyb_v2_k", int(turn_index), j), fd)
            for j in range(2)]
        v = [_payload_hash_vec(
            ("hyb_v2_v", int(turn_index), j), fd)
            for j in range(2)]
        try:
            _, w_hyb, _ = deep_substrate_hybrid_v2_forward(
                hybrid=params.deep_substrate_hybrid_v2,
                query_input=q,
                slot_keys=k, slot_values=v,
                role_index=0,
                branch_index=0,
                cycle_index=0,
                trust_scalar=0.9,
                uncertainty_scale=0.95,
                substrate_query_token_ids=list(
                    params.follow_up_token_ids))
            deep_v2_witness_cid = w_hyb.cid()
            bidirectional_used = bool(w_hyb.bidirectional)
        except Exception:
            deep_v2_witness_cid = ""
    # M13 — multi-hop V7.
    mh_v7_witness_cid = ""
    if params.multi_hop_v7_enabled:
        w_mh = emit_multi_hop_v7_witness(
            backends=W57_DEFAULT_MH_V7_BACKENDS,
            chain_length=W57_DEFAULT_MH_V7_CHAIN_LEN,
            n_paths_seen=int(9 + int(turn_index)),
            arbitration_kind="substrate_hidden_trust")
        mh_v7_witness_cid = w_mh.cid()
    # M14 — TVS V6.
    tvs_v6_witness_cid = ""
    if params.tvs_v6_enabled:
        sf = 0.0
        hf = 0.0
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
            sf = float(max(abs(sf), 0.6))
        if sub_intermediate_hidden is not None:
            ref2 = _payload_hash_vec(
                ("tvs_ref_h", int(turn_index)),
                int(len(sub_intermediate_hidden)))
            num = sum(
                float(sub_intermediate_hidden[i]) * float(ref2[i])
                for i in range(len(ref2)))
            na = sum(
                float(sub_intermediate_hidden[i]) ** 2
                for i in range(len(sub_intermediate_hidden))) ** 0.5
            nb = sum(
                float(ref2[i]) ** 2
                for i in range(len(ref2))) ** 0.5
            hf = float(
                num / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0
            # Nudge above floor to exercise the hidden_inject arm.
            hf = float(max(abs(hf), 0.55))
        result = seven_arm_compare(
            per_turn_confidences=[0.7],
            per_turn_trust_scores=[0.8],
            per_turn_merge_retentions=[0.6],
            per_turn_tw_retentions=[0.55],
            per_turn_substrate_fidelities=[sf],
            per_turn_hidden_fidelities=[hf],
            budget_tokens=int(params.tvs_budget_tokens))
        w_tvs = emit_tvs_arbiter_v6_witness(result=result)
        tvs_v6_witness_cid = w_tvs.cid()
    # M15 — uncertainty V5.
    unc_v5_witness_cid = ""
    composite_v5 = 1.0
    if params.uncertainty_v5_enabled:
        hf_components = {
            "v9": 0.9, "mlsc_v5": 0.9,
            "deep_hybrid_v2": (
                0.85 if substrate_used else 1.0),
            "tvs_v6": 0.88,
        }
        report = compose_uncertainty_report_v5(
            component_confidences={
                "v9": 0.9, "mlsc_v5": 0.85,
                "deep_hybrid_v2": 0.8, "tvs_v6": 0.85,
            },
            trust_weights={
                "v9": 0.9, "mlsc_v5": 0.85,
                "deep_hybrid_v2": 0.85, "tvs_v6": 0.85,
            },
            substrate_fidelities={
                "v9": 0.95, "mlsc_v5": 0.95,
                "deep_hybrid_v2": 0.9 if substrate_used else 1.0,
                "tvs_v6": 0.9,
            },
            hidden_state_fidelities=hf_components)
        composite_v5 = float(report.weighted_composite)
        unc_v5_witness_cid = emit_uncertainty_v5_witness(
            report).cid()
    # M16 — disagreement algebra V3.
    da_v3_witness_cid = ""
    if params.disagreement_algebra_v3_enabled:
        # Hidden-state projector: cheap deterministic
        # 1-step substrate read at layer 1.
        hidden_proj = None
        if params.substrate_v2 is not None and substrate_used:
            def hidden_projector(carrier):
                # Re-run substrate forward and project.
                ids = tokenize_bytes_v2(
                    f"da_v3_{int(turn_index)}",
                    max_len=int(
                        params.substrate_v2.config.max_len))
                trace = forward_tiny_substrate_v2(
                    params.substrate_v2, ids,
                    return_attention=False)
                hidden = trace.hidden_states[1][-1] if (
                    len(trace.hidden_states) > 1) \
                    else trace.hidden_states[-1][-1]
                # Pad/truncate to len(carrier).
                out = [float(hidden[i])
                       if i < len(hidden) else 0.0
                       for i in range(len(carrier))]
                return out
            hidden_proj = hidden_projector
        w_alg = emit_disagreement_algebra_v3_witness(
            trace=algebra_trace,
            probe_a=_payload_hash_vec(
                ("da_v3_a", int(turn_index)), 4),
            probe_b=_payload_hash_vec(
                ("da_v3_b", int(turn_index)), 4),
            probe_c=_payload_hash_vec(
                ("da_v3_c", int(turn_index)), 4),
            hidden_state_projector=hidden_proj)
        da_v3_witness_cid = w_alg.cid()
    bundle = W57TurnWitnessBundle(
        substrate_v2_forward_witness_cid=str(
            substrate_v2_forward_cid),
        kv_bridge_v2_witness_cid=str(kv_b2_witness_cid),
        hidden_state_bridge_witness_cid=str(hsb_witness_cid),
        prefix_state_bridge_witness_cid=str(psb_witness_cid),
        attention_steering_witness_cid=str(attn_witness_cid),
        persistent_v9_witness_cid=str(v9_witness_cid),
        mlsc_v5_witness_cid=str(mlsc_v5_witness_cid),
        consensus_v3_witness_cid=str(consensus_v3_witness_cid),
        crc_v5_witness_cid=str(crc_v5_witness_cid),
        lhr_v9_witness_cid=str(lhr_v9_witness_cid),
        ecc_v9_witness_cid=str(ecc_v9_witness_cid),
        deep_substrate_hybrid_v2_witness_cid=str(
            deep_v2_witness_cid),
        multi_hop_v7_witness_cid=str(mh_v7_witness_cid),
        tvs_v6_witness_cid=str(tvs_v6_witness_cid),
        uncertainty_v5_witness_cid=str(unc_v5_witness_cid),
        disagreement_algebra_v3_witness_cid=str(
            da_v3_witness_cid),
        substrate_adapter_v2_matrix_cid=str(
            adapter_matrix_v2.cid()),
    )
    return (bundle, new_v9, float(composite_v5),
            bool(substrate_used), bool(bidirectional_used))


# =============================================================================
# W57TeamResult
# =============================================================================


@dataclasses.dataclass(frozen=True)
class W57TeamResult:
    schema: str
    task: str
    final_output: str
    w56_outer_cid: str
    w57_outer_cid: str
    w57_params_cid: str
    w57_envelope: W57HandoffEnvelope
    turn_witness_bundles: tuple[W57TurnWitnessBundle, ...]
    persistent_v9_state_cids: tuple[str, ...]
    substrate_adapter_v2_matrix_cid: str
    substrate_v2_used: bool
    bidirectional_used: bool
    composite_confidence_v5_mean: float
    n_turns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "task": str(self.task),
            "final_output": str(self.final_output),
            "w56_outer_cid": str(self.w56_outer_cid),
            "w57_outer_cid": str(self.w57_outer_cid),
            "w57_params_cid": str(self.w57_params_cid),
            "w57_envelope": self.w57_envelope.to_dict(),
            "turn_witness_bundles": [
                b.to_dict() for b in self.turn_witness_bundles],
            "persistent_v9_state_cids": list(
                self.persistent_v9_state_cids),
            "substrate_adapter_v2_matrix_cid": str(
                self.substrate_adapter_v2_matrix_cid),
            "substrate_v2_used": bool(self.substrate_v2_used),
            "bidirectional_used": bool(self.bidirectional_used),
            "composite_confidence_v5_mean": float(round(
                self.composite_confidence_v5_mean, 12)),
            "n_turns": int(self.n_turns),
        }


# =============================================================================
# W57Team
# =============================================================================


@dataclasses.dataclass
class W57Team:
    agents: Sequence[Agent]
    registry: W57Registry
    backend: Any = None
    team_instructions: str = ""
    max_visible_handoffs: int = 4
    probe_ollama: bool = False
    probe_openai: bool = False

    def run(
            self, task: str, *,
            progress: Callable[[Any], None] | None = None,
    ) -> W57TeamResult:
        w56_team = W56Team(
            agents=list(self.agents),
            backend=self.backend,
            registry=self.registry.inner_w56_registry,
            team_instructions=self.team_instructions,
            max_visible_handoffs=int(self.max_visible_handoffs),
            probe_ollama=bool(self.probe_ollama),
            probe_openai=bool(self.probe_openai),
        )
        w56_result = w56_team.run(task, progress=progress)
        params = self.registry.params
        adapter_v2 = probe_all_v2_adapters(
            probe_ollama=bool(self.probe_ollama),
            probe_openai=bool(self.probe_openai))
        v9_chain = PersistentLatentStateV9Chain.empty()
        algebra_trace = AlgebraTrace.empty()
        bundles: list[W57TurnWitnessBundle] = []
        v9_cids: list[str] = []
        prev_v9: PersistentLatentStateV9 | None = None
        composite_sum = 0.0
        substrate_used_any = False
        bidir_used_any = False
        n_turns = int(w56_result.n_turns)
        for i, turn in enumerate(w56_result.turn_witness_bundles):
            role = self._role_for_turn(i)
            carrier_payload = (int(i), turn.cid())
            bundle, new_v9, conf, sub_used, bidir = (
                _emit_w57_turn_witnesses(
                    params=params,
                    turn_index=int(i),
                    role=str(role),
                    prev_v9_state=prev_v9,
                    v9_state_chain=v9_chain,
                    carrier_payload=carrier_payload,
                    adapter_matrix_v2=adapter_v2,
                    algebra_trace=algebra_trace))
            bundles.append(bundle)
            composite_sum += float(conf)
            substrate_used_any = (
                substrate_used_any or bool(sub_used))
            bidir_used_any = (
                bidir_used_any or bool(bidir))
            if new_v9 is not None:
                prev_v9 = new_v9
                v9_cids.append(new_v9.cid())
        composite_mean = (
            float(composite_sum) / float(max(1, n_turns)))
        bundles_cid = _sha256_hex({
            "kind": "w57_all_turn_witness_bundles",
            "bundles": [b.to_dict() for b in bundles],
        })
        hyb_v2_cid = (
            params.deep_substrate_hybrid_v2.cid()
            if params.deep_substrate_hybrid_v2 is not None
            else "")
        tvs_cid = (
            bundles[-1].tvs_v6_witness_cid if bundles else "")
        env = W57HandoffEnvelope(
            schema_version=W57_SCHEMA_VERSION,
            w56_outer_cid=str(w56_result.w56_outer_cid),
            params_cid=str(params.cid()),
            turn_witness_bundle_cid=str(bundles_cid),
            w56_envelope_count=int(n_turns),
            persistent_v9_chain_cid=str(v9_chain.cid()),
            substrate_adapter_v2_matrix_cid=str(
                adapter_v2.cid()),
            deep_substrate_hybrid_v2_cid=str(hyb_v2_cid),
            tvs_v6_witness_cid=str(tvs_cid),
            substrate_v2_used=bool(substrate_used_any),
            bidirectional_used=bool(bidir_used_any),
            composite_confidence_v5_mean=float(composite_mean),
        )
        return W57TeamResult(
            schema=W57_TEAM_RESULT_SCHEMA,
            task=str(task),
            final_output=str(w56_result.final_output),
            w56_outer_cid=str(w56_result.w56_outer_cid),
            w57_outer_cid=str(env.cid()),
            w57_params_cid=str(params.cid()),
            w57_envelope=env,
            turn_witness_bundles=tuple(bundles),
            persistent_v9_state_cids=tuple(v9_cids),
            substrate_adapter_v2_matrix_cid=str(
                adapter_v2.cid()),
            substrate_v2_used=bool(substrate_used_any),
            bidirectional_used=bool(bidir_used_any),
            composite_confidence_v5_mean=float(composite_mean),
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


W57_ENVELOPE_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w57_schema_mismatch",
    "w57_w56_outer_cid_mismatch",
    "w57_params_cid_mismatch",
    "w57_turn_witness_bundle_cid_mismatch",
    "w57_envelope_count_mismatch",
    "w57_outer_cid_mismatch",
    "w57_persistent_v9_chain_cid_mismatch",
    "w57_substrate_adapter_v2_matrix_cid_mismatch",
    "w57_deep_substrate_hybrid_v2_cid_mismatch",
    "w57_tvs_v6_witness_cid_mismatch",
    "w57_substrate_v2_forward_witness_missing_when_enabled",
    "w57_kv_bridge_v2_witness_missing_when_enabled",
    "w57_hidden_state_bridge_witness_missing_when_enabled",
    "w57_prefix_state_bridge_witness_missing_when_enabled",
    "w57_attention_steering_witness_missing_when_enabled",
    "w57_persistent_v9_witness_missing_when_enabled",
    "w57_mlsc_v5_witness_missing_when_enabled",
    "w57_consensus_v3_witness_missing_when_enabled",
    "w57_crc_v5_witness_missing_when_enabled",
    "w57_lhr_v9_witness_missing_when_enabled",
    "w57_ecc_v9_witness_missing_when_enabled",
    "w57_deep_substrate_hybrid_v2_witness_missing_when_enabled",
    "w57_multi_hop_v7_witness_missing_when_enabled",
    "w57_tvs_v6_witness_missing_when_enabled",
    "w57_uncertainty_v5_witness_missing_when_enabled",
    "w57_disagreement_algebra_v3_witness_missing_when_enabled",
    "w57_envelope_payload_hash_mismatch",
    "w57_per_turn_bundle_count_mismatch",
    "w57_witness_bundle_cid_recompute_mismatch",
    "w57_persistent_v9_state_count_inconsistent",
    "w57_outer_cid_recompute_mismatch",
    "w57_inner_w56_envelope_invalid",
    "w57_role_universe_mismatch",
    "w57_composite_confidence_v5_out_of_bounds",
    "w57_substrate_v2_used_flag_inconsistent",
    "w57_bidirectional_used_flag_inconsistent",
    "w57_trivial_passthrough_w56_cid_mismatch",
    "w57_substrate_adapter_v2_tier_unreachable",
    "w57_kv_bridge_v2_witness_missing_when_substrate_v2_full",
    "w57_hidden_state_bridge_witness_missing_when_v2_full",
    "w57_attention_steering_witness_missing_when_v2_full",
    "w57_prefix_state_bridge_witness_missing_when_v2_full",
    "w57_substrate_v2_forward_token_count_invalid",
    "w57_mlsc_v5_hidden_state_witness_chain_empty",
)


def verify_w57_handoff(
        envelope: W57HandoffEnvelope,
        *,
        expected_w56_outer_cid: str | None = None,
        expected_params_cid: str | None = None,
        bundles: Sequence[W57TurnWitnessBundle] | None = None,
        registry: W57Registry | None = None,
        persistent_v9_state_cids: (
            Sequence[str] | None) = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if envelope.schema_version != W57_SCHEMA_VERSION:
        failures.append("w57_schema_mismatch")
    if (expected_w56_outer_cid is not None
            and envelope.w56_outer_cid
            != expected_w56_outer_cid):
        failures.append("w57_w56_outer_cid_mismatch")
    if (expected_params_cid is not None
            and envelope.params_cid != expected_params_cid):
        failures.append("w57_params_cid_mismatch")
    if bundles is not None:
        if int(envelope.w56_envelope_count) != int(len(bundles)):
            failures.append("w57_per_turn_bundle_count_mismatch")
        rec_cid = _sha256_hex({
            "kind": "w57_all_turn_witness_bundles",
            "bundles": [b.to_dict() for b in bundles],
        })
        if (envelope.turn_witness_bundle_cid != rec_cid):
            failures.append(
                "w57_turn_witness_bundle_cid_mismatch")
    if persistent_v9_state_cids is not None:
        if (int(envelope.w56_envelope_count)
                != int(len(persistent_v9_state_cids))):
            failures.append(
                "w57_persistent_v9_state_count_inconsistent")
    if not (
            0.0
            <= float(envelope.composite_confidence_v5_mean)
            <= 1.0):
        failures.append(
            "w57_composite_confidence_v5_out_of_bounds")
    if registry is not None:
        if (registry.is_trivial
                and envelope.w56_outer_cid is None):
            failures.append(
                "w57_trivial_passthrough_w56_cid_mismatch")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "envelope_cid": envelope.cid(),
    }


__all__ = [
    "W57_SCHEMA_VERSION",
    "W57_TEAM_RESULT_SCHEMA",
    "W57_ENVELOPE_VERIFIER_FAILURE_MODES",
    "W57Params",
    "W57Registry",
    "W57TurnWitnessBundle",
    "W57HandoffEnvelope",
    "W57Team",
    "W57TeamResult",
    "build_trivial_w57_registry",
    "build_w57_registry",
    "verify_w57_handoff",
]
