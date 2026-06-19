"""W72 M1 — Tiny Transformer Runtime V17.

Strictly extends W71's ``coordpy.tiny_substrate_v16``. V17 keeps
every V16 invariant (byte-determinism, GQA, RMSNorm/SwiGLU, and
the three V16 axes: ``delayed_repair_trajectory_cid``,
``restart_dominance_per_layer``, ``delayed_repair_gate_per_layer``)
and adds **three** new substrate-load-bearing axes that W72's
multi-agent coordinator V8, team-consensus controller V7, V17
bridges/controllers, and the new delayed-rejoin-after-restart
hosted ↔ real handoff coordinator V4 exploit:

* **Default 19 layers** (vs V16's 18). Same GQA (8 query / 4 KV).
* **Per-turn restart-repair-trajectory CID** —
  ``TinyV17KVCache.restart_repair_trajectory_cid`` is a deterministic
  content-addressed SHA-256 over the V16 delayed-repair-trajectory
  CID PLUS recorded *rejoin events* (branch divergence + rejoin)
  AND the explicit *branch-pressure window* between a restart event
  and the next rejoin event. The CID is what lets the W72 MASC V8
  routing decide that a delayed rejoin after a restart under tight
  budget needs Plane B, not Plane A.
* **Per-layer delayed-rejoin-after-restart label** —
  ``TinyV17KVCache.delayed_rejoin_after_restart_per_layer`` of shape
  ``(L,)`` records the argmax repair primitive per layer in [0..8]
  where V16's [0..7] are extended by 8 = ``delayed_rejoin_after_restart``
  (any layer on which a restart event was observed AND the *next*
  rejoin lag > 0 turns).
* **Per-layer rejoin-pressure gate** —
  ``TinyV17ForwardTrace.rejoin_pressure_gate_per_layer`` of shape
  ``(L,)`` records the substrate-side throttle in [0, 1] that
  modulates substrate work as a function of the visible-token
  budget AND the restart-rejoin pressure: tight budget AND
  high rejoin pressure = aggressive substrate throttling.

V17 still preserves all V16 axes byte-for-byte under trivial
construction; the three new axes are no-ops unless explicitly
written.

Honest scope (do-not-overstate, W72)
------------------------------------

* Still NOT a frontier model. Default config:
  ``19 layers / 8 query heads / 4 kv heads / d_model=64 /
  ff_hidden=192 / byte-vocab / max_len=128 / untrained``.
  ``W72-L-NUMPY-CPU-V17-SUBSTRATE-CAP`` documents.
* V17 still does NOT bridge to third-party hosted models. The
  hosted control plane (Plane A in W68) explicitly does not pierce
  this boundary; only the in-repo V17 substrate exposes the new
  axes. ``W72-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries
  forward unchanged.
* The restart-repair-trajectory CID is a deterministic SHA-256
  hash; it does not prove rejoin integrity at the hosted surface
  (``W72-L-RESTART-REPAIR-IN-REPO-CAP``).
* The rejoin-pressure gate is a calibrated weighted combination,
  not a learned end-to-end controller. Its targets are caller-
  declared budgets, restart counts, rejoin events, and branch
  windows (``W72-L-REJOIN-PRESSURE-DECLARED-CAP``).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.tiny_substrate_v17 requires numpy") from exc

from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex
from .tiny_substrate_v9 import TinyV9SubstrateConfig
from .tiny_substrate_v10 import TinyV10SubstrateConfig
from .tiny_substrate_v11 import TinyV11SubstrateConfig
from .tiny_substrate_v12 import TinyV12SubstrateConfig
from .tiny_substrate_v13 import TinyV13SubstrateConfig
from .tiny_substrate_v14 import TinyV14SubstrateConfig
from .tiny_substrate_v15 import TinyV15SubstrateConfig
from .tiny_substrate_v16 import (
    TinyV16ForwardTrace, TinyV16KVCache, TinyV16SubstrateConfig,
    TinyV16SubstrateParams,
    W71_DEFAULT_V16_GATE_BIAS, W71_DEFAULT_V16_MAX_ROLES,
    W71_REPAIR_LABELS_V16,
    forward_tiny_substrate_v16,
    tokenize_bytes_v16 as _tokenize_bytes_v16,
)


W72_TINY_SUBSTRATE_V17_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v17.v1")

W72_TINY_V17_VOCAB_SIZE: int = 259
W72_DEFAULT_V17_D_MODEL: int = 64
W72_DEFAULT_V17_N_HEADS: int = 8
W72_DEFAULT_V17_N_KV_HEADS: int = 4
W72_DEFAULT_V17_N_LAYERS: int = 19
W72_DEFAULT_V17_FF_HIDDEN: int = 192
W72_DEFAULT_V17_MAX_LEN: int = 128
W72_DEFAULT_V17_INIT_SCALE: float = 0.04
W72_DEFAULT_V17_SEED: int = 72123456
W72_DEFAULT_V17_MAX_ROLES: int = W71_DEFAULT_V16_MAX_ROLES
W72_DEFAULT_V17_REJOIN_PRESSURE_BOOST: float = 0.76
W72_DEFAULT_V17_DELAYED_REJOIN_BOOST: float = 0.73
W72_DEFAULT_V17_GATE_BIAS: float = W71_DEFAULT_V16_GATE_BIAS
W72_DEFAULT_V17_REJOIN_LAG_FLOOR: int = 1

# V17 extends W71_REPAIR_LABELS_V16 with an eighth primitive.
W72_REPAIR_DELAYED_REJOIN_AFTER_RESTART: int = 8
W72_REPAIR_LABELS_V17: tuple[str, ...] = (
    *W71_REPAIR_LABELS_V16,
    "delayed_rejoin_after_restart",
)


def tokenize_bytes_v17(
        text: str, *, max_len: int = 16) -> list[int]:
    """Byte-tokenisation passthrough to V16."""
    return _tokenize_bytes_v16(str(text), max_len=int(max_len))


@dataclasses.dataclass
class TinyV17SubstrateConfig:
    """V17 config wraps a V16 config + three new V17 axes."""
    v16: TinyV16SubstrateConfig
    max_n_roles: int = W72_DEFAULT_V17_MAX_ROLES
    rejoin_pressure_boost: float = (
        W72_DEFAULT_V17_REJOIN_PRESSURE_BOOST)
    delayed_rejoin_boost: float = (
        W72_DEFAULT_V17_DELAYED_REJOIN_BOOST)
    expose_restart_repair_trajectory_cid: bool = True
    expose_delayed_rejoin_after_restart_per_layer: bool = True
    expose_rejoin_pressure_gate: bool = True
    gate_weights_v17: tuple[float, ...] = (
        0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07,
        0.08, 0.08, 0.09, 0.10, 0.09)
    rejoin_lag_floor_turns: int = (
        W72_DEFAULT_V17_REJOIN_LAG_FLOOR)

    @classmethod
    def default(
            cls, *, seed: int = W72_DEFAULT_V17_SEED,
    ) -> "TinyV17SubstrateConfig":
        v9 = TinyV9SubstrateConfig(
            vocab_size=W72_TINY_V17_VOCAB_SIZE,
            d_model=W72_DEFAULT_V17_D_MODEL,
            n_heads=W72_DEFAULT_V17_N_HEADS,
            n_kv_heads=W72_DEFAULT_V17_N_KV_HEADS,
            n_layers=W72_DEFAULT_V17_N_LAYERS,
            ff_hidden=W72_DEFAULT_V17_FF_HIDDEN,
            max_len=W72_DEFAULT_V17_MAX_LEN,
            init_scale=W72_DEFAULT_V17_INIT_SCALE,
            seed=int(seed))
        v10 = TinyV10SubstrateConfig(v9=v9)
        v11 = TinyV11SubstrateConfig(v10=v10)
        v12 = TinyV12SubstrateConfig(v11=v11)
        v13 = TinyV13SubstrateConfig(v12=v12)
        v14 = TinyV14SubstrateConfig(v13=v13)
        v15 = TinyV15SubstrateConfig(v14=v14)
        v16 = TinyV16SubstrateConfig(v15=v15)
        return cls(v16=v16)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W72_TINY_SUBSTRATE_V17_SCHEMA_VERSION,
            "v16_cid": str(self.v16.cid()),
            "max_n_roles": int(self.max_n_roles),
            "rejoin_pressure_boost": float(round(
                self.rejoin_pressure_boost, 12)),
            "delayed_rejoin_boost": float(round(
                self.delayed_rejoin_boost, 12)),
            "expose_restart_repair_trajectory_cid": bool(
                self.expose_restart_repair_trajectory_cid),
            "expose_delayed_rejoin_after_restart_per_layer": bool(
                self.expose_delayed_rejoin_after_restart_per_layer),
            "expose_rejoin_pressure_gate": bool(
                self.expose_rejoin_pressure_gate),
            "gate_weights_v17": [
                float(round(float(x), 12))
                for x in self.gate_weights_v17],
            "rejoin_lag_floor_turns": int(
                self.rejoin_lag_floor_turns),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v17_substrate_config",
            "config": self.to_dict()})


@dataclasses.dataclass
class TinyV17SubstrateParams:
    config: TinyV17SubstrateConfig
    v16_params: TinyV16SubstrateParams

    @classmethod
    def init(
            cls, config: TinyV17SubstrateConfig | None = None,
    ) -> "TinyV17SubstrateParams":
        if config is None:
            config = TinyV17SubstrateConfig.default()
        v16 = TinyV16SubstrateParams.init(config.v16)
        return cls(config=config, v16_params=v16)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v17_substrate_params",
            "config_cid": self.config.cid(),
            "v16_params_cid": self.v16_params.cid(),
        })


@dataclasses.dataclass
class TinyV17KVCache:
    """V17 cache. Wraps a V16 cache + three new V17 axes."""
    v16_cache: TinyV16KVCache
    restart_repair_trajectory_cid: str = ""
    delayed_rejoin_after_restart_per_layer: (
        "_np.ndarray | None") = None
    rejoin_events: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)
    branch_pressure_windows: list[dict[str, Any]] = (
        dataclasses.field(default_factory=list))
    write_log_v17: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(
            cls, n_layers: int, *, n_heads: int, max_len: int,
    ) -> "TinyV17KVCache":
        v16 = TinyV16KVCache.empty(
            int(n_layers), n_heads=int(n_heads),
            max_len=int(max_len))
        return cls(
            v16_cache=v16,
            restart_repair_trajectory_cid="",
            delayed_rejoin_after_restart_per_layer=_np.zeros(
                (int(n_layers),), dtype=_np.int64),
            rejoin_events=[],
            branch_pressure_windows=[],
            write_log_v17=[])

    def n_tokens(self) -> int:
        return int(self.v16_cache.n_tokens())

    def n_layers(self) -> int:
        return int(self.v16_cache.n_layers())

    def clone(self) -> "TinyV17KVCache":
        return TinyV17KVCache(
            v16_cache=self.v16_cache.clone(),
            restart_repair_trajectory_cid=str(
                self.restart_repair_trajectory_cid),
            delayed_rejoin_after_restart_per_layer=(
                None if (
                    self.delayed_rejoin_after_restart_per_layer
                    is None)
                else self
                .delayed_rejoin_after_restart_per_layer.copy()),
            rejoin_events=list(self.rejoin_events),
            branch_pressure_windows=list(
                self.branch_pressure_windows),
            write_log_v17=list(self.write_log_v17),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v17_kv_cache",
            "v16_cache_cid": self.v16_cache.cid(),
            "restart_repair_trajectory_cid": str(
                self.restart_repair_trajectory_cid),
            "delayed_rejoin_after_restart_per_layer_cid": (
                "none"
                if (self.delayed_rejoin_after_restart_per_layer
                    is None)
                else _ndarray_cid(
                    self.delayed_rejoin_after_restart_per_layer)),
            "rejoin_events": list(self.rejoin_events),
            "branch_pressure_windows": list(
                self.branch_pressure_windows),
            "write_log_v17": list(self.write_log_v17),
        })


@dataclasses.dataclass
class TinyV17ForwardTrace:
    v16_trace: TinyV16ForwardTrace
    restart_repair_trajectory_cid: str
    delayed_rejoin_after_restart_per_layer: "_np.ndarray"
    rejoin_pressure_gate_per_layer: "_np.ndarray"
    v17_gate_score_per_layer: "_np.ndarray"
    config_cid: str
    params_cid: str

    @property
    def logits(self) -> "_np.ndarray":
        return self.v16_trace.logits

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v17_forward_trace",
            "v16_trace_cid": self.v16_trace.cid(),
            "restart_repair_trajectory_cid": str(
                self.restart_repair_trajectory_cid),
            "delayed_rejoin_after_restart_per_layer_cid":
                _ndarray_cid(
                    self.delayed_rejoin_after_restart_per_layer),
            "rejoin_pressure_gate_per_layer_cid": _ndarray_cid(
                self.rejoin_pressure_gate_per_layer),
            "v17_gate_score_per_layer_cid": _ndarray_cid(
                self.v17_gate_score_per_layer),
        })


def _compute_restart_repair_trajectory_cid(
        cache: TinyV17KVCache) -> str:
    """Content-addressed CID across V16 delayed-repair-trajectory
    CID PLUS rejoin events PLUS recorded branch-pressure windows.

    Derived only from byte-stable witnesses; does NOT include the
    V15 substrate_self_checksum_cid (same ULP rationale as V16).
    """
    return _sha256_hex({
        "kind": "tiny_v17_restart_repair_trajectory",
        "v16_delayed_repair_trajectory_cid": str(
            cache.v16_cache.delayed_repair_trajectory_cid),
        "rejoin_events": list(cache.rejoin_events),
        "branch_pressure_windows": list(
            cache.branch_pressure_windows),
    })


def _compute_delayed_rejoin_after_restart_per_layer(
        cache: TinyV17KVCache, n_layers: int,
        rejoin_lag_floor_turns: int = (
            W72_DEFAULT_V17_REJOIN_LAG_FLOOR),
) -> "_np.ndarray":
    """Per-layer delayed-rejoin-after-restart label argmax across
    V16 primitives AND the new V17 rejoin primitive.

    Returns shape (L,) dtype int64 in [0..8]. Label 8 fires iff
    a restart event was observed on or before this layer AND the
    most recent rejoin-lag window exceeds ``rejoin_lag_floor_turns``.
    """
    L = int(n_layers)
    out = _np.zeros((L,), dtype=_np.int64)
    v16 = cache.v16_cache
    base = (
        v16.restart_dominance_per_layer
        if v16.restart_dominance_per_layer is not None
        else _np.zeros((L,), dtype=_np.int64))
    n_restart = int(len(v16.restart_events))
    n_rejoin = int(len(cache.rejoin_events))
    max_lag = 0
    for d in cache.branch_pressure_windows:
        try:
            v = int(d.get("rejoin_lag_turns", 0))
        except Exception:
            v = 0
        if v > max_lag:
            max_lag = v
    rejoin_active = bool(
        n_restart > 0 and n_rejoin > 0
        and int(max_lag) > int(rejoin_lag_floor_turns))
    for li in range(L):
        b = (
            int(base[li]) if li < int(base.shape[0]) else 0)
        if rejoin_active and (li % 4 == 0):
            out[li] = W72_REPAIR_DELAYED_REJOIN_AFTER_RESTART
        else:
            out[li] = int(b)
    return out


def _compute_rejoin_pressure_gate_per_layer(
        *, visible_token_budget: float,
        baseline_token_cost: float,
        restart_count: int,
        rejoin_count: int,
        repair_dominance_count: int,
        max_rejoin_lag_turns: int,
        v16_gate_mean: float,
        weights: Sequence[float],
        n_layers: int,
        bias: float = W72_DEFAULT_V17_GATE_BIAS,
) -> "_np.ndarray":
    """Per-layer rejoin-pressure gate.

    1.0 = full substrate work, 0.0 = abstain. The gate is a
    calibrated sigmoid over (budget_ratio, restart_pressure,
    rejoin_pressure, repair_pressure, rejoin_lag_pressure,
    v16_gate_mean).
    """
    L = int(n_layers)
    safe_cost = float(max(1.0, baseline_token_cost))
    budget_ratio = float(visible_token_budget) / safe_cost
    restart_ratio = (
        float(restart_count) / float(max(
            1, W72_DEFAULT_V17_MAX_ROLES)))
    rejoin_ratio = (
        float(rejoin_count) / float(max(
            1, W72_DEFAULT_V17_MAX_ROLES)))
    repair_ratio = (
        float(repair_dominance_count)
        / float(max(1, W72_DEFAULT_V17_MAX_ROLES)))
    lag_ratio = (
        float(max_rejoin_lag_turns)
        / float(max(1, max(8, int(max_rejoin_lag_turns) + 4))))
    feats = _np.array([
        float(v16_gate_mean),
        float(budget_ratio) / float(max(1.0, budget_ratio + 1.0)),
        float(restart_ratio),
        float(rejoin_ratio),
        float(repair_ratio),
        float(lag_ratio),
        0.5, 0.5,
        0.5, 0.5,
        float(budget_ratio) / float(max(1.0, budget_ratio + 1.0)),
        float(rejoin_ratio),
        float(lag_ratio),
    ], dtype=_np.float64)
    w = _np.array(
        [float(x) for x in weights], dtype=_np.float64)
    score = (
        float(_np.dot(w[:feats.shape[0]], feats)) + float(bias))
    sig = 1.0 / (1.0 + math.exp(-score))
    per_layer = _np.full(
        (L,), float(sig), dtype=_np.float64)
    return _np.round(per_layer, decimals=12)


def _compute_v17_gate_score(
        rejoin_active: bool,
        rejoin_pressure_gate_mean: float,
        v16_gate_mean: float,
        weights: Sequence[float],
        n_layers: int,
        bias: float = W72_DEFAULT_V17_GATE_BIAS,
) -> "_np.ndarray":
    feats = _np.array([
        float(v16_gate_mean),
        1.0 if bool(rejoin_active) else 0.0,
        float(rejoin_pressure_gate_mean),
        0.5,
        0.5, 0.5, 0.5, 0.5,
        0.5, 0.5,
        float(rejoin_pressure_gate_mean),
        1.0 if bool(rejoin_active) else 0.0,
        float(rejoin_pressure_gate_mean),
    ], dtype=_np.float64)
    w = _np.array(
        [float(x) for x in weights], dtype=_np.float64)
    score = (
        float(_np.dot(w[:feats.shape[0]], feats)) + float(bias))
    sig = 1.0 / (1.0 + math.exp(-score))
    return _np.round(
        _np.full((int(n_layers),), float(sig), dtype=_np.float64),
        decimals=12)


def forward_tiny_substrate_v17(
        params: TinyV17SubstrateParams,
        token_ids: Sequence[int],
        *,
        v17_kv_cache: TinyV17KVCache | None = None,
        attention_bias_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
        visible_token_budget: float = 256.0,
        baseline_token_cost: float = 512.0,
        restart_pressure: float = 0.0,
        rejoin_pressure: float = 0.0,
) -> tuple[TinyV17ForwardTrace, TinyV17KVCache]:
    """V17 forward = V16 forward + restart-repair-trajectory CID +
    delayed-rejoin-after-restart per layer + rejoin-pressure gate
    per layer + V17 composite gate.

    The new ``rejoin_pressure`` knob in [0, 1] is a caller-declared
    signal that the team is rejoining from divergent branches; the
    substrate uses it to bias the V17 gate towards substrate work.
    """
    cfg = params.config
    base_v16 = (
        v17_kv_cache.v16_cache if v17_kv_cache is not None
        else None)
    v16_trace, new_v16 = forward_tiny_substrate_v16(
        params.v16_params, list(token_ids),
        v16_kv_cache=base_v16,
        attention_bias_per_layer=attention_bias_per_layer,
        visible_token_budget=float(visible_token_budget),
        baseline_token_cost=float(baseline_token_cost),
        restart_pressure=float(restart_pressure))
    n_layers = int(cfg.v16.v15.v14.v13.v12.v11.v10.v9.n_layers)
    if v17_kv_cache is None:
        v17_new = TinyV17KVCache.empty(
            int(n_layers),
            n_heads=int(
                cfg.v16.v15.v14.v13.v12.v11.v10.v9.n_heads),
            max_len=int(
                cfg.v16.v15.v14.v13.v12.v11.v10.v9.max_len))
    else:
        v17_new = v17_kv_cache.clone()
    v17_new.v16_cache = new_v16
    # Restart-repair-trajectory CID over the V17 cache contents.
    rrt_cid = _compute_restart_repair_trajectory_cid(v17_new)
    v17_new.restart_repair_trajectory_cid = str(rrt_cid)
    rejoin_per_layer = (
        _compute_delayed_rejoin_after_restart_per_layer(
            v17_new, n_layers=n_layers,
            rejoin_lag_floor_turns=int(
                cfg.rejoin_lag_floor_turns)))
    v17_new.delayed_rejoin_after_restart_per_layer = (
        rejoin_per_layer)
    n_restart = int(len(new_v16.restart_events))
    n_rejoin = int(len(v17_new.rejoin_events))
    max_lag = 0
    for d in v17_new.branch_pressure_windows:
        try:
            v = int(d.get("rejoin_lag_turns", 0))
        except Exception:
            v = 0
        if v > max_lag:
            max_lag = v
    rd_count = int(
        _np.count_nonzero(
            new_v16.restart_dominance_per_layer != 0)
        if new_v16.restart_dominance_per_layer is not None
        else 0)
    v16_gate_mean = float(
        v16_trace.v16_gate_score_per_layer.mean()
        if v16_trace.v16_gate_score_per_layer.size else 0.0)
    # Caller-declared rejoin pressure is folded into the count
    # via a small lift.
    effective_rejoin = int(
        n_rejoin + int(round(float(max(0.0, min(
            1.0, float(rejoin_pressure)))) * 3.0)))
    rejoin_gate = _compute_rejoin_pressure_gate_per_layer(
        visible_token_budget=float(visible_token_budget),
        baseline_token_cost=float(baseline_token_cost),
        restart_count=int(n_restart),
        rejoin_count=int(effective_rejoin),
        repair_dominance_count=int(rd_count),
        max_rejoin_lag_turns=int(max_lag),
        v16_gate_mean=float(v16_gate_mean),
        weights=cfg.gate_weights_v17,
        n_layers=int(n_layers))
    rejoin_active = bool(
        int(_np.count_nonzero(
            rejoin_per_layer
            == W72_REPAIR_DELAYED_REJOIN_AFTER_RESTART)) > 0
        or float(rejoin_pressure) > 0.0)
    v17_gate = _compute_v17_gate_score(
        rejoin_active=bool(rejoin_active),
        rejoin_pressure_gate_mean=float(rejoin_gate.mean()),
        v16_gate_mean=float(v16_gate_mean),
        weights=cfg.gate_weights_v17,
        n_layers=int(n_layers))
    v17_new.write_log_v17.append({
        "schema": W72_TINY_SUBSTRATE_V17_SCHEMA_VERSION,
        "kind": "forward_v17",
        "n_new_tokens": int(len(list(token_ids))),
        "restart_repair_trajectory_cid": str(rrt_cid),
        "delayed_rejoin_after_restart_per_layer": [
            int(x) for x in rejoin_per_layer.tolist()],
        "rejoin_pressure_gate_mean": float(rejoin_gate.mean()),
        "v17_gate_score_mean": float(v17_gate.mean()),
        "visible_token_budget": float(visible_token_budget),
        "restart_pressure": float(restart_pressure),
        "rejoin_pressure": float(rejoin_pressure),
        "n_restart_events": int(n_restart),
        "n_rejoin_events": int(n_rejoin),
        "max_rejoin_lag_turns": int(max_lag),
    })
    trace = TinyV17ForwardTrace(
        v16_trace=v16_trace,
        restart_repair_trajectory_cid=str(rrt_cid),
        delayed_rejoin_after_restart_per_layer=rejoin_per_layer,
        rejoin_pressure_gate_per_layer=rejoin_gate,
        v17_gate_score_per_layer=v17_gate,
        config_cid=str(cfg.cid()),
        params_cid=str(params.cid()),
    )
    return trace, v17_new


def record_rejoin_event_v17(
        cache: TinyV17KVCache, *,
        turn: int, rejoin_kind: str = "branch_rejoin",
        branch_id: str = "main", role: str = "team",
) -> None:
    """Record a branch-rejoin event (e.g. branch_rejoin /
    contradiction_rejoin / multi_branch_rejoin)."""
    cache.rejoin_events.append({
        "schema": W72_TINY_SUBSTRATE_V17_SCHEMA_VERSION,
        "kind": "rejoin_event_v17",
        "rejoin_kind": str(rejoin_kind),
        "turn": int(turn),
        "branch_id": str(branch_id),
        "role": str(role),
    })
    cache.write_log_v17.append({
        "schema": W72_TINY_SUBSTRATE_V17_SCHEMA_VERSION,
        "kind": "rejoin_event_recorded",
        "rejoin_kind": str(rejoin_kind),
        "turn": int(turn),
    })


def record_branch_pressure_window_v17(
        cache: TinyV17KVCache, *,
        restart_turn: int, rejoin_turn: int,
        rejoin_lag_turns: int, branch_id: str = "main",
        role: str = "team",
) -> None:
    """Record a (restart, rejoin, lag) branch-pressure window."""
    cache.branch_pressure_windows.append({
        "schema": W72_TINY_SUBSTRATE_V17_SCHEMA_VERSION,
        "kind": "branch_pressure_window_v17",
        "restart_turn": int(restart_turn),
        "rejoin_turn": int(rejoin_turn),
        "rejoin_lag_turns": int(rejoin_lag_turns),
        "branch_id": str(branch_id),
        "role": str(role),
    })
    cache.write_log_v17.append({
        "schema": W72_TINY_SUBSTRATE_V17_SCHEMA_VERSION,
        "kind": "branch_pressure_window_recorded",
        "rejoin_lag_turns": int(rejoin_lag_turns),
        "branch_id": str(branch_id),
    })


def substrate_rejoin_dominance_flops_v17(
        *, n_tokens: int, n_repairs: int = 8,
        recompute_flops_per_token: int = 1000,
        rejoin_dominance_flops_per_token: int = 55,
) -> dict[str, Any]:
    """V17 rejoin-dominance vs recompute saving across eight
    primitives (V16's seven + ``delayed_rejoin_after_restart``).

    By routing through the dominant rejoin primitive rather than
    full recompute across all eight primitives, V17 saves
    substantial flops per turn — and now ``n_repairs`` is 8 by
    default, not 7 as in W71.
    """
    n = int(max(0, n_tokens))
    nr = int(max(1, n_repairs))
    rd_flops = int(rejoin_dominance_flops_per_token) * n * nr
    rc_flops = int(recompute_flops_per_token) * n * nr
    saving = int(rc_flops - rd_flops)
    ratio = (
        float(saving) / float(rc_flops)
        if rc_flops > 0 else 0.0)
    return {
        "n_tokens": int(n),
        "n_repairs": int(nr),
        "rejoin_dominance_flops": int(rd_flops),
        "recompute_flops": int(rc_flops),
        "saving_flops": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


def substrate_rejoin_pressure_throttle_v17(
        *, visible_token_budget: int = 64,
        baseline_token_cost: int = 512,
        rejoin_lag_turns: int = 3,
) -> dict[str, Any]:
    """V17 rejoin-pressure throttle: tokens saved when both the
    budget is tight AND the rejoin lag is non-trivial."""
    bt = int(max(0, visible_token_budget))
    bc = int(max(1, baseline_token_cost))
    base_saving = int(max(0, bc - bt))
    # Multiplicative lag lift (caller-declared).
    lag_lift = min(2.0, 1.0 + 0.18 * float(
        max(0, rejoin_lag_turns)))
    saving_tokens = int(round(base_saving * float(lag_lift)))
    saving_tokens = int(min(saving_tokens, bc))
    ratio = (
        float(saving_tokens) / float(bc)
        if bc > 0 else 0.0)
    return {
        "visible_token_budget": int(bt),
        "baseline_token_cost": int(bc),
        "rejoin_lag_turns": int(rejoin_lag_turns),
        "lag_lift": float(round(lag_lift, 12)),
        "saving_tokens": int(saving_tokens),
        "saving_ratio": float(round(ratio, 12)),
        "rejoin_pressure_active": bool(saving_tokens > 0),
    }


def build_default_tiny_substrate_v17(
        *, seed: int = W72_DEFAULT_V17_SEED,
) -> TinyV17SubstrateParams:
    """Build a default V17 substrate."""
    cfg = TinyV17SubstrateConfig.default(seed=int(seed))
    return TinyV17SubstrateParams.init(cfg)


@dataclasses.dataclass(frozen=True)
class TinyV17ForwardWitness:
    schema: str
    forward_trace_cid: str
    cache_cid: str
    restart_repair_trajectory_cid: str
    delayed_rejoin_after_restart_l1: int
    rejoin_pressure_gate_mean: float
    v17_gate_score_mean: float
    n_layers: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "forward_trace_cid": str(self.forward_trace_cid),
            "cache_cid": str(self.cache_cid),
            "restart_repair_trajectory_cid": str(
                self.restart_repair_trajectory_cid),
            "delayed_rejoin_after_restart_l1": int(
                self.delayed_rejoin_after_restart_l1),
            "rejoin_pressure_gate_mean": float(round(
                self.rejoin_pressure_gate_mean, 12)),
            "v17_gate_score_mean": float(round(
                self.v17_gate_score_mean, 12)),
            "n_layers": int(self.n_layers),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v17_forward_witness",
            "witness": self.to_dict()})


def emit_tiny_substrate_v17_forward_witness(
        trace: TinyV17ForwardTrace,
        cache: TinyV17KVCache,
) -> TinyV17ForwardWitness:
    rl = (
        cache.delayed_rejoin_after_restart_per_layer
        if cache.delayed_rejoin_after_restart_per_layer
        is not None
        else _np.zeros((0,), dtype=_np.int64))
    rj_l1 = int(_np.count_nonzero(
        rl == W72_REPAIR_DELAYED_REJOIN_AFTER_RESTART))
    rpg_mean = float(
        trace.rejoin_pressure_gate_per_layer.mean()
        if trace.rejoin_pressure_gate_per_layer.size else 0.0)
    v17_mean = float(
        trace.v17_gate_score_per_layer.mean()
        if trace.v17_gate_score_per_layer.size else 0.0)
    return TinyV17ForwardWitness(
        schema=W72_TINY_SUBSTRATE_V17_SCHEMA_VERSION,
        forward_trace_cid=str(trace.cid()),
        cache_cid=str(cache.cid()),
        restart_repair_trajectory_cid=str(
            trace.restart_repair_trajectory_cid),
        delayed_rejoin_after_restart_l1=int(rj_l1),
        rejoin_pressure_gate_mean=float(rpg_mean),
        v17_gate_score_mean=float(v17_mean),
        n_layers=int(
            trace.v17_gate_score_per_layer.shape[0]),
    )


__all__ = [
    "W72_TINY_SUBSTRATE_V17_SCHEMA_VERSION",
    "W72_TINY_V17_VOCAB_SIZE",
    "W72_DEFAULT_V17_N_LAYERS",
    "W72_DEFAULT_V17_MAX_ROLES",
    "W72_DEFAULT_V17_REJOIN_PRESSURE_BOOST",
    "W72_DEFAULT_V17_DELAYED_REJOIN_BOOST",
    "W72_DEFAULT_V17_REJOIN_LAG_FLOOR",
    "W72_REPAIR_DELAYED_REJOIN_AFTER_RESTART",
    "W72_REPAIR_LABELS_V17",
    "TinyV17SubstrateConfig",
    "TinyV17SubstrateParams",
    "TinyV17KVCache",
    "TinyV17ForwardTrace",
    "TinyV17ForwardWitness",
    "tokenize_bytes_v17",
    "forward_tiny_substrate_v17",
    "build_default_tiny_substrate_v17",
    "record_rejoin_event_v17",
    "record_branch_pressure_window_v17",
    "substrate_rejoin_dominance_flops_v17",
    "substrate_rejoin_pressure_throttle_v17",
    "emit_tiny_substrate_v17_forward_witness",
]
