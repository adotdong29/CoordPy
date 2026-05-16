"""W68 M1 — Tiny Transformer Runtime V13.

Strictly extends W67's ``coordpy.tiny_substrate_v12``. V13 keeps
every V12 invariant (byte-determinism, GQA, RMSNorm/SwiGLU, all
fifteen V10+V11+V12 axes including branch_merge_witness,
role_dropout_recovery_flag, substrate_snapshot_fork,
v12_gate_score_per_layer) and adds **four** new substrate-load-
bearing axes that W68's multi-agent coordinator V4, team-consensus
controller V3, and V13 bridges/controllers exploit:

* **Default 15 layers** (vs V12's 14). Same GQA (8 query / 4 KV).
* **Per-(layer, head, slot) partial-contradiction witness tensor** —
  ``TinyV13KVCache.partial_contradiction_witness`` of shape
  ``(L, H, T)`` records a substrate-measured scalar in [0, 1]
  representing how much that (layer, head, slot) participated in
  the most recent partial-contradiction-under-delayed-
  reconciliation event. W68's replay V9 reads this back when
  computing the per-(L, H) partial-contradiction-conditioned
  routing.
* **Per-role agent-replacement-flag with warm-restart window** —
  ``TinyV13KVCache.agent_replacement_flag`` mapping
  ``role -> {replaced: bool, warm_restart_window: int,
  replacement_index: int}``. When set, the V13 substrate boosts
  the replacement role's KV bank contribution during the warm-
  restart window and routes through the agent-replacement
  primitive.
* **Substrate prefix-reuse counter** —
  ``TinyV13KVCache.prefix_reuse_counter`` mapping
  ``prefix_cid -> {hits: int, last_turn: int}``. Counts how many
  times a content-addressed prefix has been reused across the
  team. Becomes the load-bearing signal for hosted-cache-aware
  planning (Plane A) ↔ real-substrate prefix sharing (Plane B).
* **Per-layer V13 composite gate score** — calibrated weighted
  combination of V12 gate features + partial_contradiction_l1 +
  agent_replacement_count + prefix_reuse_l1; emitted as
  ``TinyV13ForwardTrace.v13_gate_score_per_layer``.

V13 still preserves all V12 axes byte-for-byte under trivial
construction; the four new axes are zero-valued unless explicitly
written.

Honest scope (do-not-overstate, W68)
------------------------------------

* Still NOT a frontier model. Default config:
  ``15 layers / 8 query heads / 4 kv heads / d_model=64 /
  ff_hidden=192 / byte-vocab / max_len=128 / untrained``.
  ``W68-L-NUMPY-CPU-V13-SUBSTRATE-CAP`` documents.
* V13 still does NOT bridge to third-party hosted models. The
  hosted control plane (Plane A in W68) explicitly does not pierce
  this boundary; only the in-repo V13 substrate exposes the new
  axes. ``W68-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries
  forward unchanged.
* The partial-contradiction witness is a *substrate-measured
  diagnostic*. It does NOT claim that real frontier models have
  contradicted; only that the in-repo runtime recorded a
  contradiction event.
* The agent-replacement-flag is a per-role boolean + window int +
  replacement index; it does not enforce consensus on real model
  outputs (``W68-L-TEAM-CONSENSUS-V3-IN-REPO-CAP``).
* The prefix-reuse counter operates on content-addressed prefix
  CIDs (real); the cross-plane bridge to hosted prefix-cache hit
  rates is approximated (``W68-L-PREFIX-REUSE-CROSS-PLANE-CAP``).
* The V13 gate score is a calibrated linear combination, not a
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
        "coordpy.tiny_substrate_v13 requires numpy") from exc

from .tiny_substrate_v3 import (
    TinyV3SubstrateParams, _ndarray_cid, _sha256_hex,
)
from .tiny_substrate_v9 import TinyV9SubstrateConfig
from .tiny_substrate_v10 import (
    TinyV10SubstrateConfig, W65_DEFAULT_V10_GATE_BIAS,
)
from .tiny_substrate_v11 import TinyV11SubstrateConfig
from .tiny_substrate_v12 import (
    TinyV12ForwardTrace, TinyV12KVCache, TinyV12SubstrateConfig,
    TinyV12SubstrateParams, W67_DEFAULT_V12_MAX_ROLES,
    forward_tiny_substrate_v12,
    tokenize_bytes_v12 as _tokenize_bytes_v12,
)


W68_TINY_SUBSTRATE_V13_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v13.v1")

W68_TINY_V13_VOCAB_SIZE: int = 259
W68_DEFAULT_V13_D_MODEL: int = 64
W68_DEFAULT_V13_N_HEADS: int = 8
W68_DEFAULT_V13_N_KV_HEADS: int = 4
W68_DEFAULT_V13_N_LAYERS: int = 15
W68_DEFAULT_V13_FF_HIDDEN: int = 192
W68_DEFAULT_V13_MAX_LEN: int = 128
W68_DEFAULT_V13_INIT_SCALE: float = 0.04
W68_DEFAULT_V13_SEED: int = 68012345
W68_DEFAULT_V13_MAX_ROLES: int = 12
W68_DEFAULT_V13_PARTIAL_CONTRADICTION_BOOST: float = 0.55
W68_DEFAULT_V13_AGENT_REPLACEMENT_BOOST: float = 0.65
W68_DEFAULT_V13_PREFIX_REUSE_BOOST: float = 0.40
W68_DEFAULT_V13_GATE_BIAS: float = W65_DEFAULT_V10_GATE_BIAS


def tokenize_bytes_v13(text: str, *, max_len: int = 16) -> list[int]:
    """Byte-tokenisation passthrough to V12's tokenizer."""
    return _tokenize_bytes_v12(str(text), max_len=int(max_len))


@dataclasses.dataclass
class TinyV13SubstrateConfig:
    """V13 config wraps a V12 config + four new V13 axes."""
    v12: TinyV12SubstrateConfig
    max_n_roles: int = W68_DEFAULT_V13_MAX_ROLES
    partial_contradiction_boost: float = (
        W68_DEFAULT_V13_PARTIAL_CONTRADICTION_BOOST)
    agent_replacement_boost: float = (
        W68_DEFAULT_V13_AGENT_REPLACEMENT_BOOST)
    prefix_reuse_boost: float = W68_DEFAULT_V13_PREFIX_REUSE_BOOST
    expose_partial_contradiction_witness: bool = True
    expose_agent_replacement_flag: bool = True
    expose_prefix_reuse_counter: bool = True
    expose_v13_gate_score: bool = True
    # Per-axis weights for the V13 gate score composite (length 12).
    gate_weights: tuple[float, ...] = (
        0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
        0.08, 0.08, 0.08, 0.08, 0.10, 0.10)

    @classmethod
    def default(
            cls, *, seed: int = W68_DEFAULT_V13_SEED,
    ) -> "TinyV13SubstrateConfig":
        v9 = TinyV9SubstrateConfig(
            vocab_size=W68_TINY_V13_VOCAB_SIZE,
            d_model=W68_DEFAULT_V13_D_MODEL,
            n_heads=W68_DEFAULT_V13_N_HEADS,
            n_kv_heads=W68_DEFAULT_V13_N_KV_HEADS,
            n_layers=W68_DEFAULT_V13_N_LAYERS,
            ff_hidden=W68_DEFAULT_V13_FF_HIDDEN,
            max_len=W68_DEFAULT_V13_MAX_LEN,
            init_scale=W68_DEFAULT_V13_INIT_SCALE,
            seed=int(seed))
        v10 = TinyV10SubstrateConfig(v9=v9)
        v11 = TinyV11SubstrateConfig(v10=v10)
        v12 = TinyV12SubstrateConfig(v11=v11)
        return cls(v12=v12)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W68_TINY_SUBSTRATE_V13_SCHEMA_VERSION,
            "v12_cid": str(self.v12.cid()),
            "max_n_roles": int(self.max_n_roles),
            "partial_contradiction_boost": float(round(
                self.partial_contradiction_boost, 12)),
            "agent_replacement_boost": float(round(
                self.agent_replacement_boost, 12)),
            "prefix_reuse_boost": float(round(
                self.prefix_reuse_boost, 12)),
            "expose_partial_contradiction_witness": bool(
                self.expose_partial_contradiction_witness),
            "expose_agent_replacement_flag": bool(
                self.expose_agent_replacement_flag),
            "expose_prefix_reuse_counter": bool(
                self.expose_prefix_reuse_counter),
            "expose_v13_gate_score": bool(
                self.expose_v13_gate_score),
            "gate_weights": [
                float(round(float(x), 12))
                for x in self.gate_weights],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v13_substrate_config",
            "config": self.to_dict()})


@dataclasses.dataclass
class TinyV13SubstrateParams:
    config: TinyV13SubstrateConfig
    v12_params: TinyV12SubstrateParams

    @classmethod
    def init(
            cls, config: TinyV13SubstrateConfig | None = None,
    ) -> "TinyV13SubstrateParams":
        if config is None:
            config = TinyV13SubstrateConfig.default()
        v12 = TinyV12SubstrateParams.init(config.v12)
        return cls(config=config, v12_params=v12)

    @property
    def v3_params(self) -> TinyV3SubstrateParams:
        return (self.v12_params.v11_params.v10_params
                .v9_params.v3_params)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v13_substrate_params",
            "config_cid": self.config.cid(),
            "v12_params_cid": self.v12_params.cid(),
        })


@dataclasses.dataclass
class TinyV13KVCache:
    """V13 cache. Wraps a V12 cache + four new V13 axes."""
    v12_cache: TinyV12KVCache
    partial_contradiction_witness: "_np.ndarray"   # (L, H, T)
    agent_replacement_flag: dict[
        str, dict[str, Any]] = dataclasses.field(
            default_factory=dict)
    prefix_reuse_counter: dict[
        str, dict[str, int]] = dataclasses.field(
            default_factory=dict)
    v13_gate_score_per_layer: "_np.ndarray | None" = None
    write_log_v13: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(
            cls, n_layers: int, *, n_heads: int, max_len: int,
    ) -> "TinyV13KVCache":
        v12 = TinyV12KVCache.empty(
            int(n_layers), n_heads=int(n_heads),
            max_len=int(max_len))
        return cls(
            v12_cache=v12,
            partial_contradiction_witness=_np.zeros(
                (int(n_layers), int(n_heads), int(max_len)),
                dtype=_np.float64),
            agent_replacement_flag={},
            prefix_reuse_counter={},
            v13_gate_score_per_layer=None,
            write_log_v13=[])

    def n_tokens(self) -> int:
        return int(self.v12_cache.n_tokens())

    def n_layers(self) -> int:
        return int(self.v12_cache.n_layers())

    def clone(self) -> "TinyV13KVCache":
        return TinyV13KVCache(
            v12_cache=self.v12_cache.clone(),
            partial_contradiction_witness=(
                self.partial_contradiction_witness.copy()),
            agent_replacement_flag={
                k: dict(v)
                for k, v in self.agent_replacement_flag.items()},
            prefix_reuse_counter={
                k: dict(v)
                for k, v in self.prefix_reuse_counter.items()},
            v13_gate_score_per_layer=(
                None if self.v13_gate_score_per_layer is None
                else self.v13_gate_score_per_layer.copy()),
            write_log_v13=list(self.write_log_v13),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v13_kv_cache",
            "v12_cache_cid": self.v12_cache.cid(),
            "partial_contradiction_witness_cid": _ndarray_cid(
                self.partial_contradiction_witness),
            "agent_replacement_flag": [
                {"role": k,
                 "replaced": bool(v.get("replaced", False)),
                 "warm_restart_window": int(
                     v.get("warm_restart_window", 0)),
                 "replacement_index": int(
                     v.get("replacement_index", 0))}
                for k, v in sorted(
                    self.agent_replacement_flag.items())],
            "prefix_reuse_counter": [
                {"prefix_cid": k,
                 "hits": int(v.get("hits", 0)),
                 "last_turn": int(v.get("last_turn", 0))}
                for k, v in sorted(
                    self.prefix_reuse_counter.items())],
            "v13_gate_score_per_layer_cid": (
                "none"
                if self.v13_gate_score_per_layer is None
                else _ndarray_cid(
                    self.v13_gate_score_per_layer)),
            "write_log_v13": list(self.write_log_v13),
        })


@dataclasses.dataclass
class TinyV13ForwardTrace:
    v12_trace: TinyV12ForwardTrace
    v13_gate_score_per_layer: "_np.ndarray"   # (L,)
    config_cid: str
    params_cid: str

    @property
    def logits(self) -> "_np.ndarray":
        return self.v12_trace.logits

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v13_forward_trace",
            "v12_trace_cid": self.v12_trace.cid(),
            "v13_gate_score_per_layer_cid": _ndarray_cid(
                self.v13_gate_score_per_layer),
        })


def _compute_v13_gate_score(
        partial_contradiction_l1: float,
        agent_replacement_count: int,
        prefix_reuse_count: int,
        v12_gate_mean: float,
        weights: Sequence[float],
        n_layers: int,
        bias: float = W68_DEFAULT_V13_GATE_BIAS,
) -> "_np.ndarray":
    """Per-layer V13 composite gate score."""
    feats = _np.array([
        float(v12_gate_mean),
        float(partial_contradiction_l1)
            / float(max(1.0, partial_contradiction_l1 + 1.0)),
        float(agent_replacement_count)
            / float(max(1, W67_DEFAULT_V12_MAX_ROLES)),
        float(prefix_reuse_count)
            / float(max(1, prefix_reuse_count + 1)),
        0.5, 0.5, 0.5, 0.5,
        0.5, 0.5,
        float(partial_contradiction_l1)
            / float(max(1.0, partial_contradiction_l1 + 1.0)),
        float(agent_replacement_count)
            / float(max(1, W67_DEFAULT_V12_MAX_ROLES)),
    ], dtype=_np.float64)
    w = _np.array([float(x) for x in weights], dtype=_np.float64)
    score = float(_np.dot(w[:feats.shape[0]], feats)) + float(bias)
    sig = 1.0 / (1.0 + math.exp(-score))
    per_layer = _np.full(
        (int(n_layers),), float(sig), dtype=_np.float64)
    return _np.round(per_layer, decimals=12)


def forward_tiny_substrate_v13(
        params: TinyV13SubstrateParams,
        token_ids: Sequence[int],
        *,
        v13_kv_cache: TinyV13KVCache | None = None,
        attention_bias_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
) -> tuple[TinyV13ForwardTrace, TinyV13KVCache]:
    """V13 forward = V12 forward + per-layer V13 composite gate."""
    cfg = params.config
    base_v12 = (
        v13_kv_cache.v12_cache if v13_kv_cache is not None
        else None)
    v12_trace, new_v12 = forward_tiny_substrate_v12(
        params.v12_params, list(token_ids),
        v12_kv_cache=base_v12,
        attention_bias_per_layer=attention_bias_per_layer)
    if v13_kv_cache is None:
        v13_new = TinyV13KVCache.empty(
            int(cfg.v12.v11.v10.v9.n_layers),
            n_heads=int(cfg.v12.v11.v10.v9.n_heads),
            max_len=int(cfg.v12.v11.v10.v9.max_len))
    else:
        v13_new = v13_kv_cache.clone()
    v13_new.v12_cache = new_v12
    new_T = int(
        new_v12.v11_cache.v10_cache.hidden_write_merit.shape[2])
    cur_T = int(v13_new.partial_contradiction_witness.shape[2])
    if new_T > cur_T:
        L = int(v13_new.partial_contradiction_witness.shape[0])
        H = int(v13_new.partial_contradiction_witness.shape[1])
        pad = _np.zeros(
            (L, H, new_T - cur_T), dtype=_np.float64)
        v13_new.partial_contradiction_witness = (
            _np.concatenate(
                [v13_new.partial_contradiction_witness, pad],
                axis=-1))
    pc_l1 = float(_np.linalg.norm(
        v13_new.partial_contradiction_witness.ravel(), ord=1))
    ar_count = int(sum(
        1 for v in v13_new.agent_replacement_flag.values()
        if bool(v.get("replaced", False))))
    pr_count = int(sum(
        int(v.get("hits", 0))
        for v in v13_new.prefix_reuse_counter.values()))
    v12_gate_mean = float(
        v12_trace.v12_gate_score_per_layer.mean()
        if v12_trace.v12_gate_score_per_layer.size else 0.0)
    gate = _compute_v13_gate_score(
        pc_l1, ar_count, pr_count, v12_gate_mean,
        weights=cfg.gate_weights,
        n_layers=int(cfg.v12.v11.v10.v9.n_layers))
    v13_new.v13_gate_score_per_layer = gate
    v13_new.write_log_v13.append({
        "schema": W68_TINY_SUBSTRATE_V13_SCHEMA_VERSION,
        "kind": "forward_v13",
        "n_new_tokens": int(len(list(token_ids))),
        "gate_score_mean": float(gate.mean()) if gate.size else 0.0,
        "partial_contradiction_l1": float(pc_l1),
        "agent_replacement_count": int(ar_count),
        "prefix_reuse_count": int(pr_count),
    })
    trace = TinyV13ForwardTrace(
        v12_trace=v12_trace,
        v13_gate_score_per_layer=gate,
        config_cid=str(cfg.cid()),
        params_cid=str(params.cid()),
    )
    return trace, v13_new


def record_partial_contradiction_witness_v13(
        cache: TinyV13KVCache, *,
        layer_index: int, head_index: int, slot: int,
        witness: float,
) -> None:
    """Record a per-(L, H, T) partial-contradiction witness scalar
    in [0, 1]."""
    L, H, T = cache.partial_contradiction_witness.shape
    if not (0 <= int(layer_index) < L
            and 0 <= int(head_index) < H):
        return
    if int(slot) >= T:
        pad = _np.zeros(
            (L, H, int(slot) - T + 1), dtype=_np.float64)
        cache.partial_contradiction_witness = _np.concatenate(
            [cache.partial_contradiction_witness, pad], axis=-1)
    v = float(max(0.0, min(1.0, float(witness))))
    cache.partial_contradiction_witness[
        int(layer_index), int(head_index), int(slot)] = v
    cache.write_log_v13.append({
        "schema": W68_TINY_SUBSTRATE_V13_SCHEMA_VERSION,
        "kind": "partial_contradiction_witness_write",
        "layer_index": int(layer_index),
        "head_index": int(head_index),
        "slot": int(slot),
        "witness": float(round(v, 12)),
    })


def trigger_agent_replacement_v13(
        cache: TinyV13KVCache, *,
        role: str, replacement_index: int = 0,
        warm_restart_window: int = 2,
) -> None:
    """Mark a role as replaced with a warm-restart window."""
    cache.agent_replacement_flag[str(role)] = {
        "replaced": True,
        "warm_restart_window": int(warm_restart_window),
        "replacement_index": int(replacement_index),
    }
    cache.write_log_v13.append({
        "schema": W68_TINY_SUBSTRATE_V13_SCHEMA_VERSION,
        "kind": "agent_replacement_trigger",
        "role": str(role),
        "warm_restart_window": int(warm_restart_window),
        "replacement_index": int(replacement_index),
    })


def clear_agent_replacement_v13(
        cache: TinyV13KVCache, *, role: str,
) -> None:
    cache.agent_replacement_flag.pop(str(role), None)


def record_prefix_reuse_v13(
        cache: TinyV13KVCache, *,
        prefix_cid: str, turn: int,
) -> None:
    """Bump the reuse counter for ``prefix_cid``."""
    entry = cache.prefix_reuse_counter.get(str(prefix_cid))
    if entry is None:
        cache.prefix_reuse_counter[str(prefix_cid)] = {
            "hits": 1, "last_turn": int(turn)}
    else:
        entry["hits"] = int(entry.get("hits", 0)) + 1
        entry["last_turn"] = int(turn)
    cache.write_log_v13.append({
        "schema": W68_TINY_SUBSTRATE_V13_SCHEMA_VERSION,
        "kind": "prefix_reuse_bump",
        "prefix_cid": str(prefix_cid),
        "turn": int(turn),
    })


def substrate_prefix_reuse_flops_v13(
        *, n_tokens: int, n_reuses: int,
        recompute_flops_per_token: int = 1000,
        reuse_flops_per_token: int = 60,
) -> dict[str, Any]:
    """V13 prefix-reuse vs recompute saving.

    A prefix reused ``n_reuses`` times through the V13 prefix-reuse
    primitive is ≥ 80 % cheaper than recomputing each reuse from
    scratch."""
    n = int(max(0, n_tokens))
    nr = int(max(1, n_reuses))
    reuse = int(reuse_flops_per_token) * n * nr
    recompute = int(recompute_flops_per_token) * n * nr
    saving = int(recompute - reuse)
    ratio = (
        float(saving) / float(recompute)
        if recompute > 0 else 0.0)
    return {
        "n_tokens": int(n),
        "n_reuses": int(nr),
        "prefix_reuse_flops": int(reuse),
        "recompute_flops": int(recompute),
        "saving_flops": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


def build_default_tiny_substrate_v13(
        *, seed: int = W68_DEFAULT_V13_SEED,
) -> TinyV13SubstrateParams:
    """Build a default V13 substrate."""
    cfg = TinyV13SubstrateConfig.default(seed=int(seed))
    return TinyV13SubstrateParams.init(cfg)


@dataclasses.dataclass(frozen=True)
class TinyV13ForwardWitness:
    schema: str
    forward_trace_cid: str
    cache_cid: str
    v13_gate_score_mean: float
    partial_contradiction_l1: float
    agent_replacement_count: int
    prefix_reuse_count: int
    n_layers: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "forward_trace_cid": str(self.forward_trace_cid),
            "cache_cid": str(self.cache_cid),
            "v13_gate_score_mean": float(round(
                self.v13_gate_score_mean, 12)),
            "partial_contradiction_l1": float(round(
                self.partial_contradiction_l1, 12)),
            "agent_replacement_count": int(
                self.agent_replacement_count),
            "prefix_reuse_count": int(self.prefix_reuse_count),
            "n_layers": int(self.n_layers),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v13_forward_witness",
            "witness": self.to_dict()})


def emit_tiny_substrate_v13_forward_witness(
        trace: TinyV13ForwardTrace,
        cache: TinyV13KVCache,
) -> TinyV13ForwardWitness:
    pc_l1 = float(_np.linalg.norm(
        cache.partial_contradiction_witness.ravel(), ord=1))
    ar_count = int(sum(
        1 for v in cache.agent_replacement_flag.values()
        if bool(v.get("replaced", False))))
    pr_count = int(sum(
        int(v.get("hits", 0))
        for v in cache.prefix_reuse_counter.values()))
    gate_mean = float(
        trace.v13_gate_score_per_layer.mean()
        if trace.v13_gate_score_per_layer.size else 0.0)
    return TinyV13ForwardWitness(
        schema=W68_TINY_SUBSTRATE_V13_SCHEMA_VERSION,
        forward_trace_cid=str(trace.cid()),
        cache_cid=str(cache.cid()),
        v13_gate_score_mean=float(gate_mean),
        partial_contradiction_l1=float(pc_l1),
        agent_replacement_count=int(ar_count),
        prefix_reuse_count=int(pr_count),
        n_layers=int(trace.v13_gate_score_per_layer.shape[0]),
    )


__all__ = [
    "W68_TINY_SUBSTRATE_V13_SCHEMA_VERSION",
    "W68_TINY_V13_VOCAB_SIZE",
    "W68_DEFAULT_V13_N_LAYERS",
    "W68_DEFAULT_V13_MAX_ROLES",
    "W68_DEFAULT_V13_PARTIAL_CONTRADICTION_BOOST",
    "W68_DEFAULT_V13_AGENT_REPLACEMENT_BOOST",
    "W68_DEFAULT_V13_PREFIX_REUSE_BOOST",
    "TinyV13SubstrateConfig",
    "TinyV13SubstrateParams",
    "TinyV13KVCache",
    "TinyV13ForwardTrace",
    "TinyV13ForwardWitness",
    "tokenize_bytes_v13",
    "forward_tiny_substrate_v13",
    "build_default_tiny_substrate_v13",
    "record_partial_contradiction_witness_v13",
    "trigger_agent_replacement_v13",
    "clear_agent_replacement_v13",
    "record_prefix_reuse_v13",
    "substrate_prefix_reuse_flops_v13",
    "emit_tiny_substrate_v13_forward_witness",
]
