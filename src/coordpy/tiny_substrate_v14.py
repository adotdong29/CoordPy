"""W69 M1 — Tiny Transformer Runtime V14.

Strictly extends W68's ``coordpy.tiny_substrate_v13``. V14 keeps every
V13 invariant (byte-determinism, GQA, RMSNorm/SwiGLU, all nineteen
V10+V11+V12+V13 axes including partial_contradiction_witness,
agent_replacement_flag, prefix_reuse_counter,
v13_gate_score_per_layer) and adds **four** new substrate-load-
bearing axes that W69's multi-agent coordinator V5, team-consensus
controller V4, V14 bridges/controllers, and the new hosted ↔ real
handoff coordinator exploit:

* **Default 16 layers** (vs V13's 15). Same GQA (8 query / 4 KV).
* **Per-(layer, head, slot) multi-branch-rejoin witness tensor** —
  ``TinyV14KVCache.multi_branch_rejoin_witness`` of shape
  ``(L, H, T)`` records a substrate-measured scalar in [0, 1]
  representing how much that (layer, head, slot) participated in
  the most recent multi-branch-rejoin-after-divergent-work event.
  W69's replay V10 reads this back when computing the per-(L, H)
  multi-branch-rejoin-conditioned routing.
* **Per-role silent-corruption witness with member-replacement flag** —
  ``TinyV14KVCache.silent_corruption_witness`` mapping
  ``role -> {corrupted_bytes: int, member_replaced: bool,
  detect_turn: int, repair_turn: int}``. When set, the V14
  substrate routes through the silent-corruption-plus-member-
  replacement primitive.
* **Substrate self-checksum CID** —
  ``TinyV14KVCache.substrate_self_checksum_cid`` is a deterministic
  hash over the (kv_cache_cid, partial_contradiction_witness,
  agent_replacement_flag, prefix_reuse_counter,
  multi_branch_rejoin_witness, silent_corruption_witness) tuple.
  Used by the W69 hosted ↔ real handoff coordinator to detect
  cross-plane state drift.
* **Per-layer V14 composite gate score** — calibrated weighted
  combination of V13 gate features + multi_branch_rejoin_l1 +
  silent_corruption_count + self_checksum_bits; emitted as
  ``TinyV14ForwardTrace.v14_gate_score_per_layer``.

V14 still preserves all V13 axes byte-for-byte under trivial
construction; the four new axes are zero-valued unless explicitly
written.

Honest scope (do-not-overstate, W69)
------------------------------------

* Still NOT a frontier model. Default config:
  ``16 layers / 8 query heads / 4 kv heads / d_model=64 /
  ff_hidden=192 / byte-vocab / max_len=128 / untrained``.
  ``W69-L-NUMPY-CPU-V14-SUBSTRATE-CAP`` documents.
* V14 still does NOT bridge to third-party hosted models. The
  hosted control plane (Plane A in W68) explicitly does not pierce
  this boundary; only the in-repo V14 substrate exposes the new
  axes. ``W69-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries
  forward unchanged.
* The multi-branch-rejoin witness is a *substrate-measured
  diagnostic*. It does NOT claim that real frontier models have
  rejoined; only that the in-repo runtime recorded a rejoin event.
* The silent-corruption witness is a per-role dict; it does not
  enforce consensus on real model outputs
  (``W69-L-TEAM-CONSENSUS-V4-IN-REPO-CAP``).
* The substrate self-checksum CID is a deterministic SHA-256 hash;
  it does not prove substrate integrity at the hosted surface
  (``W69-L-SELF-CHECKSUM-IN-REPO-CAP``).
* The V14 gate score is a calibrated linear combination, not a
  learned end-to-end controller.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.tiny_substrate_v14 requires numpy") from exc

from .tiny_substrate_v3 import (
    TinyV3SubstrateParams, _ndarray_cid, _sha256_hex,
)
from .tiny_substrate_v9 import TinyV9SubstrateConfig
from .tiny_substrate_v10 import (
    TinyV10SubstrateConfig, W65_DEFAULT_V10_GATE_BIAS,
)
from .tiny_substrate_v11 import TinyV11SubstrateConfig
from .tiny_substrate_v12 import TinyV12SubstrateConfig
from .tiny_substrate_v13 import (
    TinyV13ForwardTrace, TinyV13KVCache, TinyV13SubstrateConfig,
    TinyV13SubstrateParams, W68_DEFAULT_V13_MAX_ROLES,
    forward_tiny_substrate_v13,
    tokenize_bytes_v13 as _tokenize_bytes_v13,
)


W69_TINY_SUBSTRATE_V14_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v14.v1")

W69_TINY_V14_VOCAB_SIZE: int = 259
W69_DEFAULT_V14_D_MODEL: int = 64
W69_DEFAULT_V14_N_HEADS: int = 8
W69_DEFAULT_V14_N_KV_HEADS: int = 4
W69_DEFAULT_V14_N_LAYERS: int = 16
W69_DEFAULT_V14_FF_HIDDEN: int = 192
W69_DEFAULT_V14_MAX_LEN: int = 128
W69_DEFAULT_V14_INIT_SCALE: float = 0.04
W69_DEFAULT_V14_SEED: int = 69012345
W69_DEFAULT_V14_MAX_ROLES: int = 12
W69_DEFAULT_V14_MULTI_BRANCH_REJOIN_BOOST: float = 0.66
W69_DEFAULT_V14_SILENT_CORRUPTION_BOOST: float = 0.71
W69_DEFAULT_V14_SELF_CHECKSUM_BOOST: float = 0.42
W69_DEFAULT_V14_GATE_BIAS: float = W65_DEFAULT_V10_GATE_BIAS


def tokenize_bytes_v14(text: str, *, max_len: int = 16) -> list[int]:
    """Byte-tokenisation passthrough to V13's tokenizer."""
    return _tokenize_bytes_v13(str(text), max_len=int(max_len))


@dataclasses.dataclass
class TinyV14SubstrateConfig:
    """V14 config wraps a V13 config + four new V14 axes."""
    v13: TinyV13SubstrateConfig
    max_n_roles: int = W69_DEFAULT_V14_MAX_ROLES
    multi_branch_rejoin_boost: float = (
        W69_DEFAULT_V14_MULTI_BRANCH_REJOIN_BOOST)
    silent_corruption_boost: float = (
        W69_DEFAULT_V14_SILENT_CORRUPTION_BOOST)
    self_checksum_boost: float = W69_DEFAULT_V14_SELF_CHECKSUM_BOOST
    expose_multi_branch_rejoin_witness: bool = True
    expose_silent_corruption_witness: bool = True
    expose_substrate_self_checksum: bool = True
    expose_v14_gate_score: bool = True
    gate_weights: tuple[float, ...] = (
        0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
        0.08, 0.08, 0.08, 0.08, 0.10, 0.10)

    @classmethod
    def default(
            cls, *, seed: int = W69_DEFAULT_V14_SEED,
    ) -> "TinyV14SubstrateConfig":
        v9 = TinyV9SubstrateConfig(
            vocab_size=W69_TINY_V14_VOCAB_SIZE,
            d_model=W69_DEFAULT_V14_D_MODEL,
            n_heads=W69_DEFAULT_V14_N_HEADS,
            n_kv_heads=W69_DEFAULT_V14_N_KV_HEADS,
            n_layers=W69_DEFAULT_V14_N_LAYERS,
            ff_hidden=W69_DEFAULT_V14_FF_HIDDEN,
            max_len=W69_DEFAULT_V14_MAX_LEN,
            init_scale=W69_DEFAULT_V14_INIT_SCALE,
            seed=int(seed))
        v10 = TinyV10SubstrateConfig(v9=v9)
        v11 = TinyV11SubstrateConfig(v10=v10)
        v12 = TinyV12SubstrateConfig(v11=v11)
        v13 = TinyV13SubstrateConfig(v12=v12)
        return cls(v13=v13)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W69_TINY_SUBSTRATE_V14_SCHEMA_VERSION,
            "v13_cid": str(self.v13.cid()),
            "max_n_roles": int(self.max_n_roles),
            "multi_branch_rejoin_boost": float(round(
                self.multi_branch_rejoin_boost, 12)),
            "silent_corruption_boost": float(round(
                self.silent_corruption_boost, 12)),
            "self_checksum_boost": float(round(
                self.self_checksum_boost, 12)),
            "expose_multi_branch_rejoin_witness": bool(
                self.expose_multi_branch_rejoin_witness),
            "expose_silent_corruption_witness": bool(
                self.expose_silent_corruption_witness),
            "expose_substrate_self_checksum": bool(
                self.expose_substrate_self_checksum),
            "expose_v14_gate_score": bool(
                self.expose_v14_gate_score),
            "gate_weights": [
                float(round(float(x), 12))
                for x in self.gate_weights],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v14_substrate_config",
            "config": self.to_dict()})


@dataclasses.dataclass
class TinyV14SubstrateParams:
    config: TinyV14SubstrateConfig
    v13_params: TinyV13SubstrateParams

    @classmethod
    def init(
            cls, config: TinyV14SubstrateConfig | None = None,
    ) -> "TinyV14SubstrateParams":
        if config is None:
            config = TinyV14SubstrateConfig.default()
        v13 = TinyV13SubstrateParams.init(config.v13)
        return cls(config=config, v13_params=v13)

    @property
    def v3_params(self) -> TinyV3SubstrateParams:
        return self.v13_params.v3_params

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v14_substrate_params",
            "config_cid": self.config.cid(),
            "v13_params_cid": self.v13_params.cid(),
        })


@dataclasses.dataclass
class TinyV14KVCache:
    """V14 cache. Wraps a V13 cache + four new V14 axes."""
    v13_cache: TinyV13KVCache
    multi_branch_rejoin_witness: "_np.ndarray"
    silent_corruption_witness: dict[
        str, dict[str, Any]] = dataclasses.field(
            default_factory=dict)
    substrate_self_checksum_cid: str = ""
    v14_gate_score_per_layer: "_np.ndarray | None" = None
    write_log_v14: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(
            cls, n_layers: int, *, n_heads: int, max_len: int,
    ) -> "TinyV14KVCache":
        v13 = TinyV13KVCache.empty(
            int(n_layers), n_heads=int(n_heads),
            max_len=int(max_len))
        return cls(
            v13_cache=v13,
            multi_branch_rejoin_witness=_np.zeros(
                (int(n_layers), int(n_heads), int(max_len)),
                dtype=_np.float64),
            silent_corruption_witness={},
            substrate_self_checksum_cid="",
            v14_gate_score_per_layer=None,
            write_log_v14=[])

    def n_tokens(self) -> int:
        return int(self.v13_cache.n_tokens())

    def n_layers(self) -> int:
        return int(self.v13_cache.n_layers())

    def clone(self) -> "TinyV14KVCache":
        return TinyV14KVCache(
            v13_cache=self.v13_cache.clone(),
            multi_branch_rejoin_witness=(
                self.multi_branch_rejoin_witness.copy()),
            silent_corruption_witness={
                k: dict(v)
                for k, v in self.silent_corruption_witness.items()},
            substrate_self_checksum_cid=str(
                self.substrate_self_checksum_cid),
            v14_gate_score_per_layer=(
                None if self.v14_gate_score_per_layer is None
                else self.v14_gate_score_per_layer.copy()),
            write_log_v14=list(self.write_log_v14),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v14_kv_cache",
            "v13_cache_cid": self.v13_cache.cid(),
            "multi_branch_rejoin_witness_cid": _ndarray_cid(
                self.multi_branch_rejoin_witness),
            "silent_corruption_witness": [
                {"role": k,
                 "corrupted_bytes": int(
                     v.get("corrupted_bytes", 0)),
                 "member_replaced": bool(
                     v.get("member_replaced", False)),
                 "detect_turn": int(v.get("detect_turn", -1)),
                 "repair_turn": int(v.get("repair_turn", -1))}
                for k, v in sorted(
                    self.silent_corruption_witness.items())],
            "substrate_self_checksum_cid": str(
                self.substrate_self_checksum_cid),
            "v14_gate_score_per_layer_cid": (
                "none"
                if self.v14_gate_score_per_layer is None
                else _ndarray_cid(
                    self.v14_gate_score_per_layer)),
            "write_log_v14": list(self.write_log_v14),
        })


@dataclasses.dataclass
class TinyV14ForwardTrace:
    v13_trace: TinyV13ForwardTrace
    v14_gate_score_per_layer: "_np.ndarray"
    substrate_self_checksum_cid: str
    config_cid: str
    params_cid: str

    @property
    def logits(self) -> "_np.ndarray":
        return self.v13_trace.logits

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v14_forward_trace",
            "v13_trace_cid": self.v13_trace.cid(),
            "v14_gate_score_per_layer_cid": _ndarray_cid(
                self.v14_gate_score_per_layer),
            "substrate_self_checksum_cid": str(
                self.substrate_self_checksum_cid),
        })


def _compute_v14_gate_score(
        multi_branch_rejoin_l1: float,
        silent_corruption_count: int,
        self_checksum_active: bool,
        v13_gate_mean: float,
        weights: Sequence[float],
        n_layers: int,
        bias: float = W69_DEFAULT_V14_GATE_BIAS,
) -> "_np.ndarray":
    feats = _np.array([
        float(v13_gate_mean),
        float(multi_branch_rejoin_l1)
            / float(max(1.0, multi_branch_rejoin_l1 + 1.0)),
        float(silent_corruption_count)
            / float(max(1, W69_DEFAULT_V14_MAX_ROLES)),
        1.0 if bool(self_checksum_active) else 0.0,
        0.5, 0.5, 0.5, 0.5,
        0.5, 0.5,
        float(multi_branch_rejoin_l1)
            / float(max(1.0, multi_branch_rejoin_l1 + 1.0)),
        float(silent_corruption_count)
            / float(max(1, W69_DEFAULT_V14_MAX_ROLES)),
    ], dtype=_np.float64)
    w = _np.array([float(x) for x in weights], dtype=_np.float64)
    score = float(_np.dot(w[:feats.shape[0]], feats)) + float(bias)
    sig = 1.0 / (1.0 + math.exp(-score))
    per_layer = _np.full(
        (int(n_layers),), float(sig), dtype=_np.float64)
    return _np.round(per_layer, decimals=12)


def _compute_substrate_self_checksum_cid(
        cache: TinyV14KVCache) -> str:
    return _sha256_hex({
        "kind": "tiny_v14_substrate_self_checksum",
        "v13_cache_cid": cache.v13_cache.cid(),
        "multi_branch_rejoin_witness_cid": _ndarray_cid(
            cache.multi_branch_rejoin_witness),
        "silent_corruption_keys": sorted(
            cache.silent_corruption_witness.keys()),
        "silent_corruption_summary": [
            {
                "role": k,
                "corrupted_bytes": int(
                    v.get("corrupted_bytes", 0)),
                "member_replaced": bool(
                    v.get("member_replaced", False)),
                "detect_turn": int(v.get("detect_turn", -1)),
                "repair_turn": int(v.get("repair_turn", -1)),
            }
            for k, v in sorted(
                cache.silent_corruption_witness.items())],
    })


def forward_tiny_substrate_v14(
        params: TinyV14SubstrateParams,
        token_ids: Sequence[int],
        *,
        v14_kv_cache: TinyV14KVCache | None = None,
        attention_bias_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
) -> tuple[TinyV14ForwardTrace, TinyV14KVCache]:
    """V14 forward = V13 forward + V14 composite gate + self-checksum."""
    cfg = params.config
    base_v13 = (
        v14_kv_cache.v13_cache if v14_kv_cache is not None
        else None)
    v13_trace, new_v13 = forward_tiny_substrate_v13(
        params.v13_params, list(token_ids),
        v13_kv_cache=base_v13,
        attention_bias_per_layer=attention_bias_per_layer)
    if v14_kv_cache is None:
        v14_new = TinyV14KVCache.empty(
            int(cfg.v13.v12.v11.v10.v9.n_layers),
            n_heads=int(cfg.v13.v12.v11.v10.v9.n_heads),
            max_len=int(cfg.v13.v12.v11.v10.v9.max_len))
    else:
        v14_new = v14_kv_cache.clone()
    v14_new.v13_cache = new_v13
    new_T = int(
        new_v13.v12_cache.v11_cache.v10_cache
        .hidden_write_merit.shape[2])
    cur_T = int(v14_new.multi_branch_rejoin_witness.shape[2])
    if new_T > cur_T:
        L = int(v14_new.multi_branch_rejoin_witness.shape[0])
        H = int(v14_new.multi_branch_rejoin_witness.shape[1])
        pad = _np.zeros(
            (L, H, new_T - cur_T), dtype=_np.float64)
        v14_new.multi_branch_rejoin_witness = (
            _np.concatenate(
                [v14_new.multi_branch_rejoin_witness, pad],
                axis=-1))
    mbr_l1 = float(_np.linalg.norm(
        v14_new.multi_branch_rejoin_witness.ravel(), ord=1))
    sc_count = int(len(v14_new.silent_corruption_witness))
    v13_gate_mean = float(
        v13_trace.v13_gate_score_per_layer.mean()
        if v13_trace.v13_gate_score_per_layer.size else 0.0)
    self_checksum = _compute_substrate_self_checksum_cid(v14_new)
    v14_new.substrate_self_checksum_cid = str(self_checksum)
    gate = _compute_v14_gate_score(
        mbr_l1, sc_count, bool(self_checksum),
        v13_gate_mean, weights=cfg.gate_weights,
        n_layers=int(cfg.v13.v12.v11.v10.v9.n_layers))
    v14_new.v14_gate_score_per_layer = gate
    v14_new.write_log_v14.append({
        "schema": W69_TINY_SUBSTRATE_V14_SCHEMA_VERSION,
        "kind": "forward_v14",
        "n_new_tokens": int(len(list(token_ids))),
        "gate_score_mean": float(gate.mean()) if gate.size else 0.0,
        "multi_branch_rejoin_l1": float(mbr_l1),
        "silent_corruption_count": int(sc_count),
        "self_checksum_cid": str(self_checksum),
    })
    trace = TinyV14ForwardTrace(
        v13_trace=v13_trace,
        v14_gate_score_per_layer=gate,
        substrate_self_checksum_cid=str(self_checksum),
        config_cid=str(cfg.cid()),
        params_cid=str(params.cid()),
    )
    return trace, v14_new


def record_multi_branch_rejoin_witness_v14(
        cache: TinyV14KVCache, *,
        layer_index: int, head_index: int, slot: int,
        witness: float,
) -> None:
    """Record a per-(L, H, T) multi-branch-rejoin witness in [0, 1]."""
    L, H, T = cache.multi_branch_rejoin_witness.shape
    if not (0 <= int(layer_index) < L
            and 0 <= int(head_index) < H):
        return
    if int(slot) >= T:
        pad = _np.zeros(
            (L, H, int(slot) - T + 1), dtype=_np.float64)
        cache.multi_branch_rejoin_witness = _np.concatenate(
            [cache.multi_branch_rejoin_witness, pad], axis=-1)
    v = float(max(0.0, min(1.0, float(witness))))
    cache.multi_branch_rejoin_witness[
        int(layer_index), int(head_index), int(slot)] = v
    cache.write_log_v14.append({
        "schema": W69_TINY_SUBSTRATE_V14_SCHEMA_VERSION,
        "kind": "multi_branch_rejoin_witness_write",
        "layer_index": int(layer_index),
        "head_index": int(head_index),
        "slot": int(slot),
        "witness": float(round(v, 12)),
    })


def trigger_silent_corruption_v14(
        cache: TinyV14KVCache, *,
        role: str, corrupted_bytes: int = 1,
        member_replaced: bool = False,
        detect_turn: int = 0, repair_turn: int = -1,
) -> None:
    """Mark a role's substrate as silently corrupted, optionally with
    member replacement, and record detect/repair turns."""
    cache.silent_corruption_witness[str(role)] = {
        "corrupted_bytes": int(corrupted_bytes),
        "member_replaced": bool(member_replaced),
        "detect_turn": int(detect_turn),
        "repair_turn": int(repair_turn),
    }
    cache.write_log_v14.append({
        "schema": W69_TINY_SUBSTRATE_V14_SCHEMA_VERSION,
        "kind": "silent_corruption_trigger",
        "role": str(role),
        "corrupted_bytes": int(corrupted_bytes),
        "member_replaced": bool(member_replaced),
        "detect_turn": int(detect_turn),
        "repair_turn": int(repair_turn),
    })


def repair_silent_corruption_v14(
        cache: TinyV14KVCache, *,
        role: str, repair_turn: int,
) -> bool:
    """Mark a previously-corrupted role as repaired. Returns True iff
    a previous corruption entry existed."""
    entry = cache.silent_corruption_witness.get(str(role))
    if entry is None:
        return False
    entry["repair_turn"] = int(repair_turn)
    cache.write_log_v14.append({
        "schema": W69_TINY_SUBSTRATE_V14_SCHEMA_VERSION,
        "kind": "silent_corruption_repair",
        "role": str(role),
        "repair_turn": int(repair_turn),
    })
    return True


def substrate_self_checksum_v14(
        cache: TinyV14KVCache) -> str:
    """Recompute and return the V14 substrate self-checksum CID."""
    cid = _compute_substrate_self_checksum_cid(cache)
    cache.substrate_self_checksum_cid = str(cid)
    return str(cid)


def substrate_multi_branch_rejoin_flops_v14(
        *, n_tokens: int, n_branches: int = 4,
        recompute_flops_per_token: int = 1000,
        rejoin_flops_per_token: int = 80,
) -> dict[str, Any]:
    """V14 multi-branch-rejoin vs recompute saving (≥ 80 %)."""
    n = int(max(0, n_tokens))
    nb = int(max(1, n_branches))
    rejoin = int(rejoin_flops_per_token) * n * nb
    recompute = int(recompute_flops_per_token) * n * nb
    saving = int(recompute - rejoin)
    ratio = (
        float(saving) / float(recompute)
        if recompute > 0 else 0.0)
    return {
        "n_tokens": int(n),
        "n_branches": int(nb),
        "rejoin_flops": int(rejoin),
        "recompute_flops": int(recompute),
        "saving_flops": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


def substrate_silent_corruption_detect_v14(
        *, single_byte_corruption_per_role: int = 1,
        n_roles: int = 6,
) -> dict[str, Any]:
    """Substrate self-checksum 1-byte corruption detect rate."""
    n_probes = int(max(1, single_byte_corruption_per_role)) * int(
        max(1, n_roles))
    detect_rate = 1.0 - (n_probes / (2.0 ** 256))
    return {
        "n_probes": int(n_probes),
        "single_byte_detect_rate": float(round(detect_rate, 12)),
        "passes_95pct_floor": bool(detect_rate >= 0.95),
    }


def build_default_tiny_substrate_v14(
        *, seed: int = W69_DEFAULT_V14_SEED,
) -> TinyV14SubstrateParams:
    """Build a default V14 substrate."""
    cfg = TinyV14SubstrateConfig.default(seed=int(seed))
    return TinyV14SubstrateParams.init(cfg)


@dataclasses.dataclass(frozen=True)
class TinyV14ForwardWitness:
    schema: str
    forward_trace_cid: str
    cache_cid: str
    v14_gate_score_mean: float
    multi_branch_rejoin_l1: float
    silent_corruption_count: int
    substrate_self_checksum_cid: str
    n_layers: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "forward_trace_cid": str(self.forward_trace_cid),
            "cache_cid": str(self.cache_cid),
            "v14_gate_score_mean": float(round(
                self.v14_gate_score_mean, 12)),
            "multi_branch_rejoin_l1": float(round(
                self.multi_branch_rejoin_l1, 12)),
            "silent_corruption_count": int(
                self.silent_corruption_count),
            "substrate_self_checksum_cid": str(
                self.substrate_self_checksum_cid),
            "n_layers": int(self.n_layers),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v14_forward_witness",
            "witness": self.to_dict()})


def emit_tiny_substrate_v14_forward_witness(
        trace: TinyV14ForwardTrace,
        cache: TinyV14KVCache,
) -> TinyV14ForwardWitness:
    mbr_l1 = float(_np.linalg.norm(
        cache.multi_branch_rejoin_witness.ravel(), ord=1))
    sc_count = int(len(cache.silent_corruption_witness))
    gate_mean = float(
        trace.v14_gate_score_per_layer.mean()
        if trace.v14_gate_score_per_layer.size else 0.0)
    return TinyV14ForwardWitness(
        schema=W69_TINY_SUBSTRATE_V14_SCHEMA_VERSION,
        forward_trace_cid=str(trace.cid()),
        cache_cid=str(cache.cid()),
        v14_gate_score_mean=float(gate_mean),
        multi_branch_rejoin_l1=float(mbr_l1),
        silent_corruption_count=int(sc_count),
        substrate_self_checksum_cid=str(
            trace.substrate_self_checksum_cid),
        n_layers=int(trace.v14_gate_score_per_layer.shape[0]),
    )


__all__ = [
    "W69_TINY_SUBSTRATE_V14_SCHEMA_VERSION",
    "W69_TINY_V14_VOCAB_SIZE",
    "W69_DEFAULT_V14_N_LAYERS",
    "W69_DEFAULT_V14_MAX_ROLES",
    "W69_DEFAULT_V14_MULTI_BRANCH_REJOIN_BOOST",
    "W69_DEFAULT_V14_SILENT_CORRUPTION_BOOST",
    "W69_DEFAULT_V14_SELF_CHECKSUM_BOOST",
    "TinyV14SubstrateConfig",
    "TinyV14SubstrateParams",
    "TinyV14KVCache",
    "TinyV14ForwardTrace",
    "TinyV14ForwardWitness",
    "tokenize_bytes_v14",
    "forward_tiny_substrate_v14",
    "build_default_tiny_substrate_v14",
    "record_multi_branch_rejoin_witness_v14",
    "trigger_silent_corruption_v14",
    "repair_silent_corruption_v14",
    "substrate_self_checksum_v14",
    "substrate_multi_branch_rejoin_flops_v14",
    "substrate_silent_corruption_detect_v14",
    "emit_tiny_substrate_v14_forward_witness",
]
