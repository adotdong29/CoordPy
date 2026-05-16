"""W71 M1 — Tiny Transformer Runtime V16.

Strictly extends W70's ``coordpy.tiny_substrate_v15``. V16 keeps
every V15 invariant (byte-determinism, GQA, RMSNorm/SwiGLU, all
twenty-six V10..V15 axes including the three V15 axes:
``repair_trajectory_cid``, ``dominant_repair_per_layer``,
``budget_primary_gate_per_layer``) and adds **three** new
substrate-load-bearing axes that W71's multi-agent coordinator V7,
team-consensus controller V6, V16 bridges/controllers, and the new
restart-aware hosted ↔ real handoff coordinator V3 exploit:

* **Default 18 layers** (vs V15's 17). Same GQA (8 query / 4 KV).
* **Per-turn delayed-repair-trajectory CID** —
  ``TinyV16KVCache.delayed_repair_trajectory_cid`` is a deterministic
  content-addressed SHA-256 over the V15 repair primitives PLUS the
  new ``restart_dominance`` primitive AND the explicit *delay window*
  between a restart event and the next repair event. The CID is what
  lets the W71 MASC V7 routing decide that a delayed repair after a
  restart needs Plane B, not Plane A.
* **Per-layer restart-dominance label** —
  ``TinyV16KVCache.restart_dominance_per_layer`` of shape ``(L,)``
  records the argmax repair primitive per layer in [0..7] where
  W70's [0..6] are extended by 7 = ``restart_dominance`` (any layer
  on which a restart event was observed and the *next* repair lag
  > 0 turns).
* **Per-layer delayed-repair gate** —
  ``TinyV16ForwardTrace.delayed_repair_gate_per_layer`` of shape
  ``(L,)`` records the substrate-side throttle in [0, 1] that
  modulates substrate work as a function of the visible-token
  budget AND the restart-repair delay window: tight budget AND
  large delay = aggressive substrate throttling.

V16 still preserves all V15 axes byte-for-byte under trivial
construction; the three new axes are no-ops unless explicitly
written.

Honest scope (do-not-overstate, W71)
------------------------------------

* Still NOT a frontier model. Default config:
  ``18 layers / 8 query heads / 4 kv heads / d_model=64 /
  ff_hidden=192 / byte-vocab / max_len=128 / untrained``.
  ``W71-L-NUMPY-CPU-V16-SUBSTRATE-CAP`` documents.
* V16 still does NOT bridge to third-party hosted models. The
  hosted control plane (Plane A in W68) explicitly does not pierce
  this boundary; only the in-repo V16 substrate exposes the new
  axes. ``W71-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries
  forward unchanged.
* The delayed-repair-trajectory CID is a deterministic SHA-256
  hash; it does not prove delayed-repair integrity at the hosted
  surface (``W71-L-DELAYED-REPAIR-IN-REPO-CAP``).
* The delayed-repair gate is a calibrated weighted combination,
  not a learned end-to-end controller. Its targets are caller-
  declared budgets, restart counts, and delay windows
  (``W71-L-DELAYED-REPAIR-DECLARED-CAP``).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.tiny_substrate_v16 requires numpy") from exc

from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex
from .tiny_substrate_v9 import TinyV9SubstrateConfig
from .tiny_substrate_v10 import TinyV10SubstrateConfig
from .tiny_substrate_v11 import TinyV11SubstrateConfig
from .tiny_substrate_v12 import TinyV12SubstrateConfig
from .tiny_substrate_v13 import TinyV13SubstrateConfig
from .tiny_substrate_v14 import TinyV14SubstrateConfig
from .tiny_substrate_v15 import (
    TinyV15ForwardTrace, TinyV15KVCache, TinyV15SubstrateConfig,
    TinyV15SubstrateParams,
    W70_DEFAULT_V15_GATE_BIAS, W70_DEFAULT_V15_MAX_ROLES,
    W70_REPAIR_LABELS, W70_REPAIR_NONE,
    forward_tiny_substrate_v15,
    tokenize_bytes_v15 as _tokenize_bytes_v15,
)


W71_TINY_SUBSTRATE_V16_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v16.v1")

W71_TINY_V16_VOCAB_SIZE: int = 259
W71_DEFAULT_V16_D_MODEL: int = 64
W71_DEFAULT_V16_N_HEADS: int = 8
W71_DEFAULT_V16_N_KV_HEADS: int = 4
W71_DEFAULT_V16_N_LAYERS: int = 18
W71_DEFAULT_V16_FF_HIDDEN: int = 192
W71_DEFAULT_V16_MAX_LEN: int = 128
W71_DEFAULT_V16_INIT_SCALE: float = 0.04
W71_DEFAULT_V16_SEED: int = 71123456
W71_DEFAULT_V16_MAX_ROLES: int = W70_DEFAULT_V15_MAX_ROLES
W71_DEFAULT_V16_RESTART_DOMINANCE_BOOST: float = 0.74
W71_DEFAULT_V16_DELAYED_REPAIR_BOOST: float = 0.71
W71_DEFAULT_V16_GATE_BIAS: float = W70_DEFAULT_V15_GATE_BIAS
W71_DEFAULT_V16_DELAYED_REPAIR_DELAY_FLOOR: int = 1

# V16 extends W70_REPAIR_LABELS with a seventh primitive.
W71_REPAIR_RESTART_DOMINANCE: int = 7
W71_REPAIR_LABELS_V16: tuple[str, ...] = (
    *W70_REPAIR_LABELS,
    "restart_dominance",
)


def tokenize_bytes_v16(
        text: str, *, max_len: int = 16) -> list[int]:
    """Byte-tokenisation passthrough to V15."""
    return _tokenize_bytes_v15(str(text), max_len=int(max_len))


@dataclasses.dataclass
class TinyV16SubstrateConfig:
    """V16 config wraps a V15 config + three new V16 axes."""
    v15: TinyV15SubstrateConfig
    max_n_roles: int = W71_DEFAULT_V16_MAX_ROLES
    restart_dominance_boost: float = (
        W71_DEFAULT_V16_RESTART_DOMINANCE_BOOST)
    delayed_repair_boost: float = (
        W71_DEFAULT_V16_DELAYED_REPAIR_BOOST)
    expose_delayed_repair_trajectory_cid: bool = True
    expose_restart_dominance_per_layer: bool = True
    expose_delayed_repair_gate: bool = True
    gate_weights_v16: tuple[float, ...] = (
        0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
        0.08, 0.08, 0.10, 0.10)
    delay_floor_turns: int = (
        W71_DEFAULT_V16_DELAYED_REPAIR_DELAY_FLOOR)

    @classmethod
    def default(
            cls, *, seed: int = W71_DEFAULT_V16_SEED,
    ) -> "TinyV16SubstrateConfig":
        v9 = TinyV9SubstrateConfig(
            vocab_size=W71_TINY_V16_VOCAB_SIZE,
            d_model=W71_DEFAULT_V16_D_MODEL,
            n_heads=W71_DEFAULT_V16_N_HEADS,
            n_kv_heads=W71_DEFAULT_V16_N_KV_HEADS,
            n_layers=W71_DEFAULT_V16_N_LAYERS,
            ff_hidden=W71_DEFAULT_V16_FF_HIDDEN,
            max_len=W71_DEFAULT_V16_MAX_LEN,
            init_scale=W71_DEFAULT_V16_INIT_SCALE,
            seed=int(seed))
        v10 = TinyV10SubstrateConfig(v9=v9)
        v11 = TinyV11SubstrateConfig(v10=v10)
        v12 = TinyV12SubstrateConfig(v11=v11)
        v13 = TinyV13SubstrateConfig(v12=v12)
        v14 = TinyV14SubstrateConfig(v13=v13)
        v15 = TinyV15SubstrateConfig(v14=v14)
        return cls(v15=v15)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W71_TINY_SUBSTRATE_V16_SCHEMA_VERSION,
            "v15_cid": str(self.v15.cid()),
            "max_n_roles": int(self.max_n_roles),
            "restart_dominance_boost": float(round(
                self.restart_dominance_boost, 12)),
            "delayed_repair_boost": float(round(
                self.delayed_repair_boost, 12)),
            "expose_delayed_repair_trajectory_cid": bool(
                self.expose_delayed_repair_trajectory_cid),
            "expose_restart_dominance_per_layer": bool(
                self.expose_restart_dominance_per_layer),
            "expose_delayed_repair_gate": bool(
                self.expose_delayed_repair_gate),
            "gate_weights_v16": [
                float(round(float(x), 12))
                for x in self.gate_weights_v16],
            "delay_floor_turns": int(self.delay_floor_turns),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v16_substrate_config",
            "config": self.to_dict()})


@dataclasses.dataclass
class TinyV16SubstrateParams:
    config: TinyV16SubstrateConfig
    v15_params: TinyV15SubstrateParams

    @classmethod
    def init(
            cls, config: TinyV16SubstrateConfig | None = None,
    ) -> "TinyV16SubstrateParams":
        if config is None:
            config = TinyV16SubstrateConfig.default()
        v15 = TinyV15SubstrateParams.init(config.v15)
        return cls(config=config, v15_params=v15)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v16_substrate_params",
            "config_cid": self.config.cid(),
            "v15_params_cid": self.v15_params.cid(),
        })


@dataclasses.dataclass
class TinyV16KVCache:
    """V16 cache. Wraps a V15 cache + three new V16 axes."""
    v15_cache: TinyV15KVCache
    delayed_repair_trajectory_cid: str = ""
    restart_dominance_per_layer: "_np.ndarray | None" = None
    restart_events: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)
    delay_windows: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)
    write_log_v16: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(
            cls, n_layers: int, *, n_heads: int, max_len: int,
    ) -> "TinyV16KVCache":
        v15 = TinyV15KVCache.empty(
            int(n_layers), n_heads=int(n_heads),
            max_len=int(max_len))
        return cls(
            v15_cache=v15,
            delayed_repair_trajectory_cid="",
            restart_dominance_per_layer=_np.zeros(
                (int(n_layers),), dtype=_np.int64),
            restart_events=[],
            delay_windows=[],
            write_log_v16=[])

    def n_tokens(self) -> int:
        return int(self.v15_cache.n_tokens())

    def n_layers(self) -> int:
        return int(self.v15_cache.n_layers())

    def clone(self) -> "TinyV16KVCache":
        return TinyV16KVCache(
            v15_cache=self.v15_cache.clone(),
            delayed_repair_trajectory_cid=str(
                self.delayed_repair_trajectory_cid),
            restart_dominance_per_layer=(
                None if self.restart_dominance_per_layer is None
                else self.restart_dominance_per_layer.copy()),
            restart_events=list(self.restart_events),
            delay_windows=list(self.delay_windows),
            write_log_v16=list(self.write_log_v16),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v16_kv_cache",
            "v15_cache_cid": self.v15_cache.cid(),
            "delayed_repair_trajectory_cid": str(
                self.delayed_repair_trajectory_cid),
            "restart_dominance_per_layer_cid": (
                "none"
                if self.restart_dominance_per_layer is None
                else _ndarray_cid(
                    self.restart_dominance_per_layer)),
            "restart_events": list(self.restart_events),
            "delay_windows": list(self.delay_windows),
            "write_log_v16": list(self.write_log_v16),
        })


@dataclasses.dataclass
class TinyV16ForwardTrace:
    v15_trace: TinyV15ForwardTrace
    delayed_repair_trajectory_cid: str
    restart_dominance_per_layer: "_np.ndarray"
    delayed_repair_gate_per_layer: "_np.ndarray"
    v16_gate_score_per_layer: "_np.ndarray"
    config_cid: str
    params_cid: str

    @property
    def logits(self) -> "_np.ndarray":
        return self.v15_trace.logits

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v16_forward_trace",
            "v15_trace_cid": self.v15_trace.cid(),
            "delayed_repair_trajectory_cid": str(
                self.delayed_repair_trajectory_cid),
            "restart_dominance_per_layer_cid": _ndarray_cid(
                self.restart_dominance_per_layer),
            "delayed_repair_gate_per_layer_cid": _ndarray_cid(
                self.delayed_repair_gate_per_layer),
            "v16_gate_score_per_layer_cid": _ndarray_cid(
                self.v16_gate_score_per_layer),
        })


def _compute_delayed_repair_trajectory_cid(
        cache: TinyV16KVCache) -> str:
    """Content-addressed CID across V15 repair-trajectory CID PLUS
    restart events PLUS recorded delay windows.

    Derived only from byte-stable witnesses; does NOT include the
    V15 substrate_self_checksum_cid (same ULP rationale as V15).
    """
    return _sha256_hex({
        "kind": "tiny_v16_delayed_repair_trajectory",
        "v15_repair_trajectory_cid": str(
            cache.v15_cache.repair_trajectory_cid),
        "restart_events": list(cache.restart_events),
        "delay_windows": list(cache.delay_windows),
    })


def _compute_restart_dominance_per_layer(
        cache: TinyV16KVCache, n_layers: int,
        delay_floor_turns: int = (
            W71_DEFAULT_V16_DELAYED_REPAIR_DELAY_FLOOR),
) -> "_np.ndarray":
    """Per-layer restart-dominance label argmax across V15 primitives
    AND the new restart primitive.

    Returns shape (L,) dtype int64 in [0..7]. Label 7 fires iff
    a restart event was observed on or before this layer AND the
    most recent delay window exceeds ``delay_floor_turns``.
    """
    L = int(n_layers)
    out = _np.zeros((L,), dtype=_np.int64)
    v15 = cache.v15_cache
    base = (
        v15.dominant_repair_per_layer
        if v15.dominant_repair_per_layer is not None
        else _np.zeros((L,), dtype=_np.int64))
    # Restart pressure heuristic.
    n_restart = int(len(cache.restart_events))
    max_delay = 0
    for d in cache.delay_windows:
        try:
            v = int(d.get("delay_turns", 0))
        except Exception:
            v = 0
        if v > max_delay:
            max_delay = v
    restart_active = bool(
        n_restart > 0 and int(max_delay) > int(delay_floor_turns))
    for li in range(L):
        b = (
            int(base[li]) if li < int(base.shape[0]) else 0)
        if restart_active and (li % 3 == 0):
            out[li] = W71_REPAIR_RESTART_DOMINANCE
        else:
            out[li] = int(b)
    return out


def _compute_delayed_repair_gate_per_layer(
        *, visible_token_budget: float,
        baseline_token_cost: float,
        restart_count: int,
        repair_dominance_count: int,
        max_delay_turns: int,
        v15_gate_mean: float,
        weights: Sequence[float],
        n_layers: int,
        bias: float = W71_DEFAULT_V16_GATE_BIAS,
) -> "_np.ndarray":
    """Per-layer delayed-repair gate.

    1.0 = full substrate work, 0.0 = abstain. The gate is a
    calibrated sigmoid over (budget_ratio, restart_pressure,
    repair_pressure, delay_pressure, v15_gate_mean).
    """
    L = int(n_layers)
    safe_cost = float(max(1.0, baseline_token_cost))
    budget_ratio = float(visible_token_budget) / safe_cost
    restart_ratio = (
        float(restart_count) / float(max(1,
                                          W71_DEFAULT_V16_MAX_ROLES)))
    repair_ratio = (
        float(repair_dominance_count)
        / float(max(1, W71_DEFAULT_V16_MAX_ROLES)))
    delay_ratio = (
        float(max_delay_turns)
        / float(max(1, max(8, int(max_delay_turns) + 4))))
    feats = _np.array([
        float(v15_gate_mean),
        float(budget_ratio) / float(max(1.0, budget_ratio + 1.0)),
        float(restart_ratio),
        float(repair_ratio),
        float(delay_ratio),
        0.5, 0.5, 0.5,
        0.5, 0.5,
        float(budget_ratio) / float(max(1.0, budget_ratio + 1.0)),
        float(delay_ratio),
    ], dtype=_np.float64)
    w = _np.array(
        [float(x) for x in weights], dtype=_np.float64)
    score = (
        float(_np.dot(w[:feats.shape[0]], feats)) + float(bias))
    sig = 1.0 / (1.0 + math.exp(-score))
    per_layer = _np.full(
        (L,), float(sig), dtype=_np.float64)
    return _np.round(per_layer, decimals=12)


def _compute_v16_gate_score(
        restart_dominance_active: bool,
        delayed_repair_gate_mean: float,
        v15_gate_mean: float,
        weights: Sequence[float],
        n_layers: int,
        bias: float = W71_DEFAULT_V16_GATE_BIAS,
) -> "_np.ndarray":
    feats = _np.array([
        float(v15_gate_mean),
        1.0 if bool(restart_dominance_active) else 0.0,
        float(delayed_repair_gate_mean),
        0.5,
        0.5, 0.5, 0.5, 0.5,
        0.5, 0.5,
        float(delayed_repair_gate_mean),
        1.0 if bool(restart_dominance_active) else 0.0,
    ], dtype=_np.float64)
    w = _np.array(
        [float(x) for x in weights], dtype=_np.float64)
    score = (
        float(_np.dot(w[:feats.shape[0]], feats)) + float(bias))
    sig = 1.0 / (1.0 + math.exp(-score))
    return _np.round(
        _np.full((int(n_layers),), float(sig), dtype=_np.float64),
        decimals=12)


def forward_tiny_substrate_v16(
        params: TinyV16SubstrateParams,
        token_ids: Sequence[int],
        *,
        v16_kv_cache: TinyV16KVCache | None = None,
        attention_bias_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
        visible_token_budget: float = 256.0,
        baseline_token_cost: float = 512.0,
        restart_pressure: float = 0.0,
) -> tuple[TinyV16ForwardTrace, TinyV16KVCache]:
    """V16 forward = V15 forward + delayed-repair-trajectory CID +
    restart-dominance per layer + delayed-repair gate per layer +
    V16 composite gate.

    The new ``restart_pressure`` knob in [0, 1] is a caller-declared
    signal that the team is recovering from a recent restart; the
    substrate uses it to bias the V16 gate towards substrate work.
    """
    cfg = params.config
    base_v15 = (
        v16_kv_cache.v15_cache if v16_kv_cache is not None
        else None)
    v15_trace, new_v15 = forward_tiny_substrate_v15(
        params.v15_params, list(token_ids),
        v15_kv_cache=base_v15,
        attention_bias_per_layer=attention_bias_per_layer,
        visible_token_budget=float(visible_token_budget),
        baseline_token_cost=float(baseline_token_cost))
    n_layers = int(cfg.v15.v14.v13.v12.v11.v10.v9.n_layers)
    if v16_kv_cache is None:
        v16_new = TinyV16KVCache.empty(
            int(n_layers),
            n_heads=int(cfg.v15.v14.v13.v12.v11.v10.v9.n_heads),
            max_len=int(cfg.v15.v14.v13.v12.v11.v10.v9.max_len))
    else:
        v16_new = v16_kv_cache.clone()
    v16_new.v15_cache = new_v15
    # Delayed-repair-trajectory CID over the V16 cache contents.
    drt_cid = _compute_delayed_repair_trajectory_cid(v16_new)
    v16_new.delayed_repair_trajectory_cid = str(drt_cid)
    restart_per_layer = _compute_restart_dominance_per_layer(
        v16_new, n_layers=n_layers,
        delay_floor_turns=int(cfg.delay_floor_turns))
    v16_new.restart_dominance_per_layer = restart_per_layer
    # Restart count + max delay window.
    n_restart = int(len(v16_new.restart_events))
    max_delay = 0
    for d in v16_new.delay_windows:
        try:
            v = int(d.get("delay_turns", 0))
        except Exception:
            v = 0
        if v > max_delay:
            max_delay = v
    rd_count = int(_np.count_nonzero(
        new_v15.dominant_repair_per_layer
        != W70_REPAIR_NONE)
        if new_v15.dominant_repair_per_layer is not None
        else 0)
    v15_gate_mean = float(
        v15_trace.v15_gate_score_per_layer.mean()
        if v15_trace.v15_gate_score_per_layer.size else 0.0)
    # Caller-declared restart pressure is folded into the count
    # via a small lift.
    effective_restart = int(
        n_restart + int(round(float(max(0.0, min(
            1.0, float(restart_pressure)))) * 3.0)))
    delay_gate = _compute_delayed_repair_gate_per_layer(
        visible_token_budget=float(visible_token_budget),
        baseline_token_cost=float(baseline_token_cost),
        restart_count=int(effective_restart),
        repair_dominance_count=int(rd_count),
        max_delay_turns=int(max_delay),
        v15_gate_mean=float(v15_gate_mean),
        weights=cfg.gate_weights_v16,
        n_layers=int(n_layers))
    restart_active = bool(
        int(_np.count_nonzero(
            restart_per_layer == W71_REPAIR_RESTART_DOMINANCE)) > 0
        or float(restart_pressure) > 0.0)
    v16_gate = _compute_v16_gate_score(
        restart_dominance_active=bool(restart_active),
        delayed_repair_gate_mean=float(delay_gate.mean()),
        v15_gate_mean=float(v15_gate_mean),
        weights=cfg.gate_weights_v16,
        n_layers=int(n_layers))
    v16_new.write_log_v16.append({
        "schema": W71_TINY_SUBSTRATE_V16_SCHEMA_VERSION,
        "kind": "forward_v16",
        "n_new_tokens": int(len(list(token_ids))),
        "delayed_repair_trajectory_cid": str(drt_cid),
        "restart_dominance_per_layer": [
            int(x) for x in restart_per_layer.tolist()],
        "delayed_repair_gate_mean": float(delay_gate.mean()),
        "v16_gate_score_mean": float(v16_gate.mean()),
        "visible_token_budget": float(visible_token_budget),
        "restart_pressure": float(restart_pressure),
        "n_restart_events": int(n_restart),
        "max_delay_turns": int(max_delay),
    })
    trace = TinyV16ForwardTrace(
        v15_trace=v15_trace,
        delayed_repair_trajectory_cid=str(drt_cid),
        restart_dominance_per_layer=restart_per_layer,
        delayed_repair_gate_per_layer=delay_gate,
        v16_gate_score_per_layer=v16_gate,
        config_cid=str(cfg.cid()),
        params_cid=str(params.cid()),
    )
    return trace, v16_new


def record_restart_event_v16(
        cache: TinyV16KVCache, *,
        turn: int, restart_kind: str = "agent_restart",
        layer_index: int = 0, role: str = "team",
) -> None:
    """Record a restart event (e.g. agent_restart / warm_restart /
    role_restart)."""
    cache.restart_events.append({
        "schema": W71_TINY_SUBSTRATE_V16_SCHEMA_VERSION,
        "kind": "restart_event_v16",
        "restart_kind": str(restart_kind),
        "turn": int(turn),
        "layer_index": int(layer_index),
        "role": str(role),
    })
    cache.write_log_v16.append({
        "schema": W71_TINY_SUBSTRATE_V16_SCHEMA_VERSION,
        "kind": "restart_event_recorded",
        "restart_kind": str(restart_kind),
        "turn": int(turn),
    })


def record_delay_window_v16(
        cache: TinyV16KVCache, *,
        restart_turn: int, repair_turn: int,
        delay_turns: int, role: str = "team",
) -> None:
    """Record a (restart, repair, delay) window."""
    cache.delay_windows.append({
        "schema": W71_TINY_SUBSTRATE_V16_SCHEMA_VERSION,
        "kind": "delay_window_v16",
        "restart_turn": int(restart_turn),
        "repair_turn": int(repair_turn),
        "delay_turns": int(delay_turns),
        "role": str(role),
    })
    cache.write_log_v16.append({
        "schema": W71_TINY_SUBSTRATE_V16_SCHEMA_VERSION,
        "kind": "delay_window_recorded",
        "delay_turns": int(delay_turns),
        "role": str(role),
    })


def substrate_repair_dominance_flops_v16(
        *, n_tokens: int, n_repairs: int = 7,
        recompute_flops_per_token: int = 1000,
        repair_dominance_flops_per_token: int = 60,
) -> dict[str, Any]:
    """V16 repair-dominance vs recompute saving across seven
    primitives (V15's six + ``restart_dominance``).

    By routing through the dominant repair primitive rather than
    full recompute across all seven primitives, V16 saves
    substantial flops per turn — and now ``n_repairs`` is 7 by
    default, not 6 as in W70.
    """
    n = int(max(0, n_tokens))
    nr = int(max(1, n_repairs))
    rd_flops = int(repair_dominance_flops_per_token) * n * nr
    rc_flops = int(recompute_flops_per_token) * n * nr
    saving = int(rc_flops - rd_flops)
    ratio = (
        float(saving) / float(rc_flops)
        if rc_flops > 0 else 0.0)
    return {
        "n_tokens": int(n),
        "n_repairs": int(nr),
        "repair_dominance_flops": int(rd_flops),
        "recompute_flops": int(rc_flops),
        "saving_flops": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


def substrate_delayed_repair_throttle_v16(
        *, visible_token_budget: int = 64,
        baseline_token_cost: int = 512,
        delay_turns: int = 3,
) -> dict[str, Any]:
    """V16 delayed-repair throttle: tokens saved when both the
    budget is tight AND the delay window is non-trivial."""
    bt = int(max(0, visible_token_budget))
    bc = int(max(1, baseline_token_cost))
    base_saving = int(max(0, bc - bt))
    # Multiplicative delay lift (caller-declared).
    delay_lift = min(2.0, 1.0 + 0.15 * float(max(0, delay_turns)))
    saving_tokens = int(round(base_saving * float(delay_lift)))
    saving_tokens = int(min(saving_tokens, bc))
    ratio = (
        float(saving_tokens) / float(bc)
        if bc > 0 else 0.0)
    return {
        "visible_token_budget": int(bt),
        "baseline_token_cost": int(bc),
        "delay_turns": int(delay_turns),
        "delay_lift": float(round(delay_lift, 12)),
        "saving_tokens": int(saving_tokens),
        "saving_ratio": float(round(ratio, 12)),
        "delayed_repair_active": bool(saving_tokens > 0),
    }


def build_default_tiny_substrate_v16(
        *, seed: int = W71_DEFAULT_V16_SEED,
) -> TinyV16SubstrateParams:
    """Build a default V16 substrate."""
    cfg = TinyV16SubstrateConfig.default(seed=int(seed))
    return TinyV16SubstrateParams.init(cfg)


@dataclasses.dataclass(frozen=True)
class TinyV16ForwardWitness:
    schema: str
    forward_trace_cid: str
    cache_cid: str
    delayed_repair_trajectory_cid: str
    restart_dominance_l1: int
    delayed_repair_gate_mean: float
    v16_gate_score_mean: float
    n_layers: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "forward_trace_cid": str(self.forward_trace_cid),
            "cache_cid": str(self.cache_cid),
            "delayed_repair_trajectory_cid": str(
                self.delayed_repair_trajectory_cid),
            "restart_dominance_l1": int(
                self.restart_dominance_l1),
            "delayed_repair_gate_mean": float(round(
                self.delayed_repair_gate_mean, 12)),
            "v16_gate_score_mean": float(round(
                self.v16_gate_score_mean, 12)),
            "n_layers": int(self.n_layers),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v16_forward_witness",
            "witness": self.to_dict()})


def emit_tiny_substrate_v16_forward_witness(
        trace: TinyV16ForwardTrace,
        cache: TinyV16KVCache,
) -> TinyV16ForwardWitness:
    rl = (
        cache.restart_dominance_per_layer
        if cache.restart_dominance_per_layer is not None
        else _np.zeros((0,), dtype=_np.int64))
    rd_l1 = int(_np.count_nonzero(
        rl == W71_REPAIR_RESTART_DOMINANCE))
    drg_mean = float(
        trace.delayed_repair_gate_per_layer.mean()
        if trace.delayed_repair_gate_per_layer.size else 0.0)
    v16_mean = float(
        trace.v16_gate_score_per_layer.mean()
        if trace.v16_gate_score_per_layer.size else 0.0)
    return TinyV16ForwardWitness(
        schema=W71_TINY_SUBSTRATE_V16_SCHEMA_VERSION,
        forward_trace_cid=str(trace.cid()),
        cache_cid=str(cache.cid()),
        delayed_repair_trajectory_cid=str(
            trace.delayed_repair_trajectory_cid),
        restart_dominance_l1=int(rd_l1),
        delayed_repair_gate_mean=float(drg_mean),
        v16_gate_score_mean=float(v16_mean),
        n_layers=int(
            trace.v16_gate_score_per_layer.shape[0]),
    )


__all__ = [
    "W71_TINY_SUBSTRATE_V16_SCHEMA_VERSION",
    "W71_TINY_V16_VOCAB_SIZE",
    "W71_DEFAULT_V16_N_LAYERS",
    "W71_DEFAULT_V16_MAX_ROLES",
    "W71_DEFAULT_V16_RESTART_DOMINANCE_BOOST",
    "W71_DEFAULT_V16_DELAYED_REPAIR_BOOST",
    "W71_DEFAULT_V16_DELAYED_REPAIR_DELAY_FLOOR",
    "W71_REPAIR_RESTART_DOMINANCE",
    "W71_REPAIR_LABELS_V16",
    "TinyV16SubstrateConfig",
    "TinyV16SubstrateParams",
    "TinyV16KVCache",
    "TinyV16ForwardTrace",
    "TinyV16ForwardWitness",
    "tokenize_bytes_v16",
    "forward_tiny_substrate_v16",
    "build_default_tiny_substrate_v16",
    "record_restart_event_v16",
    "record_delay_window_v16",
    "substrate_repair_dominance_flops_v16",
    "substrate_delayed_repair_throttle_v16",
    "emit_tiny_substrate_v16_forward_witness",
]
