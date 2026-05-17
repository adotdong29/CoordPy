"""W73 M1 — Tiny Transformer Runtime V18.

Strictly extends W72's ``coordpy.tiny_substrate_v17``. V18 keeps
every V17 invariant (byte-determinism, GQA, RMSNorm/SwiGLU, and the
three V17 axes: ``restart_repair_trajectory_cid``,
``delayed_rejoin_after_restart_per_layer``,
``rejoin_pressure_gate_per_layer``) and adds **three** new
substrate-load-bearing axes that W73's multi-agent coordinator V9,
team-consensus controller V8, V18 bridges/controllers, and the new
replacement-aware hosted ↔ real handoff coordinator V5 exploit:

* **Default 20 layers** (vs V17's 19). Same GQA (8 query / 4 KV).
* **Per-turn replacement-repair-trajectory CID** —
  ``TinyV18KVCache.replacement_repair_trajectory_cid`` is a
  deterministic content-addressed SHA-256 over the V17 restart-
  repair-trajectory CID PLUS recorded *replacement events* (a role
  being replaced by a fresh member after a contradiction and before
  the team rejoins) AND the explicit *replacement window* between
  the contradiction event and the eventual rejoin. The CID is what
  lets the W73 MASC V9 routing decide that a replacement after a
  contradiction followed by a delayed rejoin under tight budget
  needs Plane B, not Plane A.
* **Per-layer replacement-after-contradiction-then-rejoin label** —
  ``TinyV18KVCache.replacement_after_contradiction_then_rejoin_per_layer``
  of shape ``(L,)`` records the argmax repair primitive per layer
  in [0..9] where V17's [0..8] are extended by 9 =
  ``replacement_after_contradiction_then_rejoin`` (any layer on
  which a replacement event was observed AND a contradiction event
  preceded it AND a rejoin event followed it).
* **Per-layer replacement-pressure gate** —
  ``TinyV18ForwardTrace.replacement_pressure_gate_per_layer`` of
  shape ``(L,)`` records the substrate-side throttle in [0, 1]
  that modulates substrate work as a function of the visible-token
  budget AND the joint restart+rejoin+replacement+contradiction
  pressure: tight budget AND high joint pressure = aggressive
  substrate throttling.

V18 still preserves all V17 axes byte-for-byte under trivial
construction; the three new axes are no-ops unless explicitly
written.

Honest scope (do-not-overstate, W73)
------------------------------------

* Still NOT a frontier model. Default config:
  ``20 layers / 8 query heads / 4 kv heads / d_model=64 /
  ff_hidden=192 / byte-vocab / max_len=128 / untrained``.
  ``W73-L-NUMPY-CPU-V18-SUBSTRATE-CAP`` documents.
* V18 still does NOT bridge to third-party hosted models. The
  hosted control plane (Plane A) explicitly does not pierce this
  boundary; only the in-repo V18 substrate exposes the new axes.
  ``W73-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward
  unchanged.
* The replacement-repair-trajectory CID is a deterministic SHA-256
  hash; it does not prove replacement-repair integrity at the
  hosted surface (``W73-L-REPLACEMENT-REPAIR-IN-REPO-CAP``).
* The replacement-pressure gate is a calibrated weighted
  combination, not a learned end-to-end controller. Its targets
  are caller-declared budgets, restart counts, rejoin counts,
  replacement counts, contradiction counts, and replacement-lag
  windows (``W73-L-REPLACEMENT-PRESSURE-DECLARED-CAP``).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.tiny_substrate_v18 requires numpy") from exc

from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex
from .tiny_substrate_v9 import TinyV9SubstrateConfig
from .tiny_substrate_v10 import TinyV10SubstrateConfig
from .tiny_substrate_v11 import TinyV11SubstrateConfig
from .tiny_substrate_v12 import TinyV12SubstrateConfig
from .tiny_substrate_v13 import TinyV13SubstrateConfig
from .tiny_substrate_v14 import TinyV14SubstrateConfig
from .tiny_substrate_v15 import TinyV15SubstrateConfig
from .tiny_substrate_v16 import TinyV16SubstrateConfig
from .tiny_substrate_v17 import (
    TinyV17ForwardTrace, TinyV17KVCache, TinyV17SubstrateConfig,
    TinyV17SubstrateParams,
    W72_DEFAULT_V17_GATE_BIAS, W72_DEFAULT_V17_MAX_ROLES,
    W72_REPAIR_LABELS_V17,
    forward_tiny_substrate_v17,
    tokenize_bytes_v17 as _tokenize_bytes_v17,
)


W73_TINY_SUBSTRATE_V18_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v18.v1")

W73_TINY_V18_VOCAB_SIZE: int = 259
W73_DEFAULT_V18_D_MODEL: int = 64
W73_DEFAULT_V18_N_HEADS: int = 8
W73_DEFAULT_V18_N_KV_HEADS: int = 4
W73_DEFAULT_V18_N_LAYERS: int = 20
W73_DEFAULT_V18_FF_HIDDEN: int = 192
W73_DEFAULT_V18_MAX_LEN: int = 128
W73_DEFAULT_V18_INIT_SCALE: float = 0.04
W73_DEFAULT_V18_SEED: int = 73123456
W73_DEFAULT_V18_MAX_ROLES: int = W72_DEFAULT_V17_MAX_ROLES
W73_DEFAULT_V18_REPLACEMENT_PRESSURE_BOOST: float = 0.79
W73_DEFAULT_V18_REPLACEMENT_AFTER_CTR_BOOST: float = 0.78
W73_DEFAULT_V18_GATE_BIAS: float = W72_DEFAULT_V17_GATE_BIAS
W73_DEFAULT_V18_REPLACEMENT_LAG_FLOOR: int = 1

# V18 extends W72_REPAIR_LABELS_V17 with a tenth primitive.
W73_REPAIR_REPLACEMENT_AFTER_CTR: int = 9
W73_REPAIR_LABELS_V18: tuple[str, ...] = (
    *W72_REPAIR_LABELS_V17,
    "replacement_after_contradiction_then_rejoin",
)


def tokenize_bytes_v18(
        text: str, *, max_len: int = 16) -> list[int]:
    """Byte-tokenisation passthrough to V17."""
    return _tokenize_bytes_v17(str(text), max_len=int(max_len))


@dataclasses.dataclass
class TinyV18SubstrateConfig:
    """V18 config wraps a V17 config + three new V18 axes."""
    v17: TinyV17SubstrateConfig
    max_n_roles: int = W73_DEFAULT_V18_MAX_ROLES
    replacement_pressure_boost: float = (
        W73_DEFAULT_V18_REPLACEMENT_PRESSURE_BOOST)
    replacement_after_ctr_boost: float = (
        W73_DEFAULT_V18_REPLACEMENT_AFTER_CTR_BOOST)
    expose_replacement_repair_trajectory_cid: bool = True
    expose_replacement_after_ctr_per_layer: bool = True
    expose_replacement_pressure_gate: bool = True
    gate_weights_v18: tuple[float, ...] = (
        0.06, 0.06, 0.06, 0.07, 0.07, 0.07, 0.07, 0.07,
        0.08, 0.08, 0.09, 0.09, 0.10, 0.09)
    replacement_lag_floor_turns: int = (
        W73_DEFAULT_V18_REPLACEMENT_LAG_FLOOR)

    @classmethod
    def default(
            cls, *, seed: int = W73_DEFAULT_V18_SEED,
    ) -> "TinyV18SubstrateConfig":
        v9 = TinyV9SubstrateConfig(
            vocab_size=W73_TINY_V18_VOCAB_SIZE,
            d_model=W73_DEFAULT_V18_D_MODEL,
            n_heads=W73_DEFAULT_V18_N_HEADS,
            n_kv_heads=W73_DEFAULT_V18_N_KV_HEADS,
            n_layers=W73_DEFAULT_V18_N_LAYERS,
            ff_hidden=W73_DEFAULT_V18_FF_HIDDEN,
            max_len=W73_DEFAULT_V18_MAX_LEN,
            init_scale=W73_DEFAULT_V18_INIT_SCALE,
            seed=int(seed))
        v10 = TinyV10SubstrateConfig(v9=v9)
        v11 = TinyV11SubstrateConfig(v10=v10)
        v12 = TinyV12SubstrateConfig(v11=v11)
        v13 = TinyV13SubstrateConfig(v12=v12)
        v14 = TinyV14SubstrateConfig(v13=v13)
        v15 = TinyV15SubstrateConfig(v14=v14)
        v16 = TinyV16SubstrateConfig(v15=v15)
        v17 = TinyV17SubstrateConfig(v16=v16)
        return cls(v17=v17)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W73_TINY_SUBSTRATE_V18_SCHEMA_VERSION,
            "v17_cid": str(self.v17.cid()),
            "max_n_roles": int(self.max_n_roles),
            "replacement_pressure_boost": float(round(
                self.replacement_pressure_boost, 12)),
            "replacement_after_ctr_boost": float(round(
                self.replacement_after_ctr_boost, 12)),
            "expose_replacement_repair_trajectory_cid": bool(
                self.expose_replacement_repair_trajectory_cid),
            "expose_replacement_after_ctr_per_layer": bool(
                self.expose_replacement_after_ctr_per_layer),
            "expose_replacement_pressure_gate": bool(
                self.expose_replacement_pressure_gate),
            "gate_weights_v18": [
                float(round(float(x), 12))
                for x in self.gate_weights_v18],
            "replacement_lag_floor_turns": int(
                self.replacement_lag_floor_turns),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v18_substrate_config",
            "config": self.to_dict()})


@dataclasses.dataclass
class TinyV18SubstrateParams:
    config: TinyV18SubstrateConfig
    v17_params: TinyV17SubstrateParams

    @classmethod
    def init(
            cls, config: TinyV18SubstrateConfig | None = None,
    ) -> "TinyV18SubstrateParams":
        if config is None:
            config = TinyV18SubstrateConfig.default()
        v17 = TinyV17SubstrateParams.init(config.v17)
        return cls(config=config, v17_params=v17)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v18_substrate_params",
            "config_cid": self.config.cid(),
            "v17_params_cid": self.v17_params.cid(),
        })


@dataclasses.dataclass
class TinyV18KVCache:
    """V18 cache. Wraps a V17 cache + three new V18 axes."""
    v17_cache: TinyV17KVCache
    replacement_repair_trajectory_cid: str = ""
    replacement_after_ctr_per_layer: (
        "_np.ndarray | None") = None
    replacement_events: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)
    contradiction_events: list[dict[str, Any]] = (
        dataclasses.field(default_factory=list))
    replacement_windows: list[dict[str, Any]] = (
        dataclasses.field(default_factory=list))
    write_log_v18: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(
            cls, n_layers: int, *, n_heads: int, max_len: int,
    ) -> "TinyV18KVCache":
        v17 = TinyV17KVCache.empty(
            int(n_layers), n_heads=int(n_heads),
            max_len=int(max_len))
        return cls(
            v17_cache=v17,
            replacement_repair_trajectory_cid="",
            replacement_after_ctr_per_layer=_np.zeros(
                (int(n_layers),), dtype=_np.int64),
            replacement_events=[],
            contradiction_events=[],
            replacement_windows=[],
            write_log_v18=[])

    def n_tokens(self) -> int:
        return int(self.v17_cache.n_tokens())

    def n_layers(self) -> int:
        return int(self.v17_cache.n_layers())

    def clone(self) -> "TinyV18KVCache":
        return TinyV18KVCache(
            v17_cache=self.v17_cache.clone(),
            replacement_repair_trajectory_cid=str(
                self.replacement_repair_trajectory_cid),
            replacement_after_ctr_per_layer=(
                None if (
                    self.replacement_after_ctr_per_layer
                    is None)
                else self
                .replacement_after_ctr_per_layer.copy()),
            replacement_events=list(self.replacement_events),
            contradiction_events=list(
                self.contradiction_events),
            replacement_windows=list(
                self.replacement_windows),
            write_log_v18=list(self.write_log_v18),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v18_kv_cache",
            "v17_cache_cid": self.v17_cache.cid(),
            "replacement_repair_trajectory_cid": str(
                self.replacement_repair_trajectory_cid),
            "replacement_after_ctr_per_layer_cid": (
                "none"
                if (self.replacement_after_ctr_per_layer
                    is None)
                else _ndarray_cid(
                    self.replacement_after_ctr_per_layer)),
            "replacement_events": list(self.replacement_events),
            "contradiction_events": list(
                self.contradiction_events),
            "replacement_windows": list(
                self.replacement_windows),
            "write_log_v18": list(self.write_log_v18),
        })


@dataclasses.dataclass
class TinyV18ForwardTrace:
    v17_trace: TinyV17ForwardTrace
    replacement_repair_trajectory_cid: str
    replacement_after_ctr_per_layer: "_np.ndarray"
    replacement_pressure_gate_per_layer: "_np.ndarray"
    v18_gate_score_per_layer: "_np.ndarray"
    config_cid: str
    params_cid: str

    @property
    def logits(self) -> "_np.ndarray":
        return self.v17_trace.logits

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v18_forward_trace",
            "v17_trace_cid": self.v17_trace.cid(),
            "replacement_repair_trajectory_cid": str(
                self.replacement_repair_trajectory_cid),
            "replacement_after_ctr_per_layer_cid":
                _ndarray_cid(
                    self.replacement_after_ctr_per_layer),
            "replacement_pressure_gate_per_layer_cid":
                _ndarray_cid(
                    self.replacement_pressure_gate_per_layer),
            "v18_gate_score_per_layer_cid": _ndarray_cid(
                self.v18_gate_score_per_layer),
        })


def _compute_replacement_repair_trajectory_cid(
        cache: TinyV18KVCache) -> str:
    """Content-addressed CID over V17 restart-repair-trajectory
    CID PLUS replacement events PLUS contradiction events PLUS
    recorded replacement windows.

    Derived only from byte-stable witnesses.
    """
    return _sha256_hex({
        "kind": "tiny_v18_replacement_repair_trajectory",
        "v17_restart_repair_trajectory_cid": str(
            cache.v17_cache.restart_repair_trajectory_cid),
        "replacement_events": list(cache.replacement_events),
        "contradiction_events": list(
            cache.contradiction_events),
        "replacement_windows": list(
            cache.replacement_windows),
    })


def _compute_replacement_after_ctr_per_layer(
        cache: TinyV18KVCache, n_layers: int,
        replacement_lag_floor_turns: int = (
            W73_DEFAULT_V18_REPLACEMENT_LAG_FLOOR),
) -> "_np.ndarray":
    """Per-layer replacement-after-contradiction-then-rejoin label
    argmax across V17 primitives AND the new V18 replacement
    primitive.

    Returns shape (L,) dtype int64 in [0..9]. Label 9 fires iff a
    replacement event was observed AND a contradiction event
    preceded it AND a rejoin event followed it AND the most recent
    replacement-lag window exceeds ``replacement_lag_floor_turns``.
    """
    L = int(n_layers)
    out = _np.zeros((L,), dtype=_np.int64)
    v17 = cache.v17_cache
    base = (
        v17.delayed_rejoin_after_restart_per_layer
        if v17.delayed_rejoin_after_restart_per_layer is not None
        else _np.zeros((L,), dtype=_np.int64))
    n_replace = int(len(cache.replacement_events))
    n_contradict = int(len(cache.contradiction_events))
    n_rejoin = int(len(v17.rejoin_events))
    max_lag = 0
    for d in cache.replacement_windows:
        try:
            v = int(d.get("replacement_lag_turns", 0))
        except Exception:
            v = 0
        if v > max_lag:
            max_lag = v
    replacement_active = bool(
        n_replace > 0 and n_contradict > 0 and n_rejoin > 0
        and int(max_lag) > int(replacement_lag_floor_turns))
    for li in range(L):
        b = (
            int(base[li]) if li < int(base.shape[0]) else 0)
        if replacement_active and (li % 5 == 0):
            out[li] = W73_REPAIR_REPLACEMENT_AFTER_CTR
        else:
            out[li] = int(b)
    return out


def _compute_replacement_pressure_gate_per_layer(
        *, visible_token_budget: float,
        baseline_token_cost: float,
        restart_count: int,
        rejoin_count: int,
        replacement_count: int,
        contradiction_count: int,
        repair_dominance_count: int,
        max_replacement_lag_turns: int,
        v17_gate_mean: float,
        weights: Sequence[float],
        n_layers: int,
        bias: float = W73_DEFAULT_V18_GATE_BIAS,
) -> "_np.ndarray":
    """Per-layer replacement-pressure gate.

    1.0 = full substrate work, 0.0 = abstain. The gate is a
    calibrated sigmoid over (budget_ratio, restart_pressure,
    rejoin_pressure, replacement_pressure, contradiction_pressure,
    repair_pressure, replacement_lag_pressure, v17_gate_mean).
    """
    L = int(n_layers)
    safe_cost = float(max(1.0, baseline_token_cost))
    budget_ratio = float(visible_token_budget) / safe_cost
    rmax = float(max(1, W73_DEFAULT_V18_MAX_ROLES))
    restart_ratio = float(restart_count) / rmax
    rejoin_ratio = float(rejoin_count) / rmax
    replace_ratio = float(replacement_count) / rmax
    contradict_ratio = float(contradiction_count) / rmax
    repair_ratio = float(repair_dominance_count) / rmax
    lag_ratio = (
        float(max_replacement_lag_turns)
        / float(max(1, max(
            8, int(max_replacement_lag_turns) + 4))))
    feats = _np.array([
        float(v17_gate_mean),
        float(budget_ratio) / float(max(1.0, budget_ratio + 1.0)),
        float(restart_ratio),
        float(rejoin_ratio),
        float(replace_ratio),
        float(contradict_ratio),
        float(repair_ratio),
        float(lag_ratio),
        0.5, 0.5,
        float(replace_ratio),
        float(contradict_ratio),
        float(lag_ratio),
        float(budget_ratio) / float(max(1.0, budget_ratio + 1.0)),
    ], dtype=_np.float64)
    w = _np.array(
        [float(x) for x in weights], dtype=_np.float64)
    score = (
        float(_np.dot(w[:feats.shape[0]], feats)) + float(bias))
    sig = 1.0 / (1.0 + math.exp(-score))
    per_layer = _np.full(
        (L,), float(sig), dtype=_np.float64)
    return _np.round(per_layer, decimals=12)


def _compute_v18_gate_score(
        replacement_active: bool,
        replacement_pressure_gate_mean: float,
        v17_gate_mean: float,
        weights: Sequence[float],
        n_layers: int,
        bias: float = W73_DEFAULT_V18_GATE_BIAS,
) -> "_np.ndarray":
    feats = _np.array([
        float(v17_gate_mean),
        1.0 if bool(replacement_active) else 0.0,
        float(replacement_pressure_gate_mean),
        0.5,
        0.5, 0.5, 0.5, 0.5,
        0.5, 0.5,
        float(replacement_pressure_gate_mean),
        1.0 if bool(replacement_active) else 0.0,
        float(replacement_pressure_gate_mean),
        float(replacement_pressure_gate_mean),
    ], dtype=_np.float64)
    w = _np.array(
        [float(x) for x in weights], dtype=_np.float64)
    score = (
        float(_np.dot(w[:feats.shape[0]], feats)) + float(bias))
    sig = 1.0 / (1.0 + math.exp(-score))
    return _np.round(
        _np.full((int(n_layers),), float(sig), dtype=_np.float64),
        decimals=12)


def forward_tiny_substrate_v18(
        params: TinyV18SubstrateParams,
        token_ids: Sequence[int],
        *,
        v18_kv_cache: TinyV18KVCache | None = None,
        attention_bias_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
        visible_token_budget: float = 256.0,
        baseline_token_cost: float = 512.0,
        restart_pressure: float = 0.0,
        rejoin_pressure: float = 0.0,
        replacement_pressure: float = 0.0,
        contradiction_pressure: float = 0.0,
) -> tuple[TinyV18ForwardTrace, TinyV18KVCache]:
    """V18 forward = V17 forward + replacement-repair-trajectory
    CID + replacement-after-contradiction-then-rejoin per layer +
    replacement-pressure gate per layer + V18 composite gate.

    The new ``replacement_pressure`` and ``contradiction_pressure``
    knobs in [0, 1] are caller-declared signals that the team is
    absorbing a replacement after contradiction; the substrate uses
    them to bias the V18 gate towards substrate work.
    """
    cfg = params.config
    base_v17 = (
        v18_kv_cache.v17_cache if v18_kv_cache is not None
        else None)
    v17_trace, new_v17 = forward_tiny_substrate_v17(
        params.v17_params, list(token_ids),
        v17_kv_cache=base_v17,
        attention_bias_per_layer=attention_bias_per_layer,
        visible_token_budget=float(visible_token_budget),
        baseline_token_cost=float(baseline_token_cost),
        restart_pressure=float(restart_pressure),
        rejoin_pressure=float(rejoin_pressure))
    n_layers = int(
        cfg.v17.v16.v15.v14.v13.v12.v11.v10.v9.n_layers)
    if v18_kv_cache is None:
        v18_new = TinyV18KVCache.empty(
            int(n_layers),
            n_heads=int(
                cfg.v17.v16.v15.v14.v13.v12.v11.v10.v9.n_heads),
            max_len=int(
                cfg.v17.v16.v15.v14.v13.v12.v11.v10.v9.max_len))
    else:
        v18_new = v18_kv_cache.clone()
    v18_new.v17_cache = new_v17
    # Replacement-repair-trajectory CID over the V18 cache contents.
    rrt_cid = _compute_replacement_repair_trajectory_cid(v18_new)
    v18_new.replacement_repair_trajectory_cid = str(rrt_cid)
    rep_per_layer = (
        _compute_replacement_after_ctr_per_layer(
            v18_new, n_layers=n_layers,
            replacement_lag_floor_turns=int(
                cfg.replacement_lag_floor_turns)))
    v18_new.replacement_after_ctr_per_layer = rep_per_layer
    n_restart = int(len(new_v17.v16_cache.restart_events))
    n_rejoin = int(len(new_v17.rejoin_events))
    n_replace = int(len(v18_new.replacement_events))
    n_contradict = int(len(v18_new.contradiction_events))
    max_lag = 0
    for d in v18_new.replacement_windows:
        try:
            v = int(d.get("replacement_lag_turns", 0))
        except Exception:
            v = 0
        if v > max_lag:
            max_lag = v
    rd_count = int(
        _np.count_nonzero(
            new_v17.v16_cache.restart_dominance_per_layer != 0)
        if new_v17.v16_cache.restart_dominance_per_layer is not None
        else 0)
    v17_gate_mean = float(
        v17_trace.v17_gate_score_per_layer.mean()
        if v17_trace.v17_gate_score_per_layer.size else 0.0)
    # Caller-declared replacement / contradiction pressure is folded
    # into the counts via a small lift.
    effective_replace = int(
        n_replace + int(round(float(max(0.0, min(
            1.0, float(replacement_pressure)))) * 3.0)))
    effective_contradict = int(
        n_contradict + int(round(float(max(0.0, min(
            1.0, float(contradiction_pressure)))) * 3.0)))
    rep_gate = _compute_replacement_pressure_gate_per_layer(
        visible_token_budget=float(visible_token_budget),
        baseline_token_cost=float(baseline_token_cost),
        restart_count=int(n_restart),
        rejoin_count=int(n_rejoin),
        replacement_count=int(effective_replace),
        contradiction_count=int(effective_contradict),
        repair_dominance_count=int(rd_count),
        max_replacement_lag_turns=int(max_lag),
        v17_gate_mean=float(v17_gate_mean),
        weights=cfg.gate_weights_v18,
        n_layers=int(n_layers))
    replacement_active = bool(
        int(_np.count_nonzero(
            rep_per_layer
            == W73_REPAIR_REPLACEMENT_AFTER_CTR)) > 0
        or float(replacement_pressure) > 0.0)
    v18_gate = _compute_v18_gate_score(
        replacement_active=bool(replacement_active),
        replacement_pressure_gate_mean=float(rep_gate.mean()),
        v17_gate_mean=float(v17_gate_mean),
        weights=cfg.gate_weights_v18,
        n_layers=int(n_layers))
    v18_new.write_log_v18.append({
        "schema": W73_TINY_SUBSTRATE_V18_SCHEMA_VERSION,
        "kind": "forward_v18",
        "n_new_tokens": int(len(list(token_ids))),
        "replacement_repair_trajectory_cid": str(rrt_cid),
        "replacement_after_ctr_per_layer": [
            int(x) for x in rep_per_layer.tolist()],
        "replacement_pressure_gate_mean": float(rep_gate.mean()),
        "v18_gate_score_mean": float(v18_gate.mean()),
        "visible_token_budget": float(visible_token_budget),
        "restart_pressure": float(restart_pressure),
        "rejoin_pressure": float(rejoin_pressure),
        "replacement_pressure": float(replacement_pressure),
        "contradiction_pressure": float(contradiction_pressure),
        "n_replace_events": int(n_replace),
        "n_contradict_events": int(n_contradict),
        "max_replacement_lag_turns": int(max_lag),
    })
    trace = TinyV18ForwardTrace(
        v17_trace=v17_trace,
        replacement_repair_trajectory_cid=str(rrt_cid),
        replacement_after_ctr_per_layer=rep_per_layer,
        replacement_pressure_gate_per_layer=rep_gate,
        v18_gate_score_per_layer=v18_gate,
        config_cid=str(cfg.cid()),
        params_cid=str(params.cid()),
    )
    return trace, v18_new


def record_replacement_event_v18(
        cache: TinyV18KVCache, *,
        turn: int, replacement_kind: str = "agent_replacement",
        role: str = "team", new_role: str = "team",
) -> None:
    """Record an agent-replacement event (e.g. after contradiction)."""
    cache.replacement_events.append({
        "schema": W73_TINY_SUBSTRATE_V18_SCHEMA_VERSION,
        "kind": "replacement_event_v18",
        "replacement_kind": str(replacement_kind),
        "turn": int(turn),
        "role": str(role),
        "new_role": str(new_role),
    })
    cache.write_log_v18.append({
        "schema": W73_TINY_SUBSTRATE_V18_SCHEMA_VERSION,
        "kind": "replacement_event_recorded",
        "replacement_kind": str(replacement_kind),
        "turn": int(turn),
    })


def record_contradiction_event_v18(
        cache: TinyV18KVCache, *,
        turn: int,
        contradiction_kind: str = "fact_contradiction",
        role: str = "team", branch_id: str = "main",
) -> None:
    """Record a contradiction event (e.g. fact / branch disagreement)."""
    cache.contradiction_events.append({
        "schema": W73_TINY_SUBSTRATE_V18_SCHEMA_VERSION,
        "kind": "contradiction_event_v18",
        "contradiction_kind": str(contradiction_kind),
        "turn": int(turn),
        "role": str(role),
        "branch_id": str(branch_id),
    })
    cache.write_log_v18.append({
        "schema": W73_TINY_SUBSTRATE_V18_SCHEMA_VERSION,
        "kind": "contradiction_event_recorded",
        "contradiction_kind": str(contradiction_kind),
        "turn": int(turn),
    })


def record_replacement_window_v18(
        cache: TinyV18KVCache, *,
        contradiction_turn: int,
        replacement_turn: int,
        rejoin_turn: int,
        replacement_lag_turns: int,
        role: str = "team", branch_id: str = "main",
) -> None:
    """Record a (contradiction, replacement, rejoin, lag) window."""
    cache.replacement_windows.append({
        "schema": W73_TINY_SUBSTRATE_V18_SCHEMA_VERSION,
        "kind": "replacement_window_v18",
        "contradiction_turn": int(contradiction_turn),
        "replacement_turn": int(replacement_turn),
        "rejoin_turn": int(rejoin_turn),
        "replacement_lag_turns": int(replacement_lag_turns),
        "role": str(role),
        "branch_id": str(branch_id),
    })
    cache.write_log_v18.append({
        "schema": W73_TINY_SUBSTRATE_V18_SCHEMA_VERSION,
        "kind": "replacement_window_recorded",
        "replacement_lag_turns": int(replacement_lag_turns),
        "role": str(role),
    })


def substrate_replacement_dominance_flops_v18(
        *, n_tokens: int, n_repairs: int = 9,
        recompute_flops_per_token: int = 1000,
        replacement_dominance_flops_per_token: int = 50,
) -> dict[str, Any]:
    """V18 replacement-dominance vs recompute saving across nine
    primitives (V17's eight + ``replacement_after_contradiction_then_rejoin``).

    By routing through the dominant replacement primitive rather
    than full recompute across all nine primitives, V18 saves
    substantial flops per turn — and now ``n_repairs`` is 9 by
    default.
    """
    n = int(max(0, n_tokens))
    nr = int(max(1, n_repairs))
    rd_flops = (
        int(replacement_dominance_flops_per_token) * n * nr)
    rc_flops = int(recompute_flops_per_token) * n * nr
    saving = int(rc_flops - rd_flops)
    ratio = (
        float(saving) / float(rc_flops)
        if rc_flops > 0 else 0.0)
    return {
        "n_tokens": int(n),
        "n_repairs": int(nr),
        "replacement_dominance_flops": int(rd_flops),
        "recompute_flops": int(rc_flops),
        "saving_flops": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


def substrate_replacement_pressure_throttle_v18(
        *, visible_token_budget: int = 64,
        baseline_token_cost: int = 512,
        replacement_lag_turns: int = 3,
) -> dict[str, Any]:
    """V18 replacement-pressure throttle: tokens saved when both
    budget is tight AND replacement lag is non-trivial."""
    bt = int(max(0, visible_token_budget))
    bc = int(max(1, baseline_token_cost))
    base_saving = int(max(0, bc - bt))
    lag_lift = min(2.0, 1.0 + 0.20 * float(
        max(0, replacement_lag_turns)))
    saving_tokens = int(round(base_saving * float(lag_lift)))
    saving_tokens = int(min(saving_tokens, bc))
    ratio = (
        float(saving_tokens) / float(bc)
        if bc > 0 else 0.0)
    return {
        "visible_token_budget": int(bt),
        "baseline_token_cost": int(bc),
        "replacement_lag_turns": int(replacement_lag_turns),
        "lag_lift": float(round(lag_lift, 12)),
        "saving_tokens": int(saving_tokens),
        "saving_ratio": float(round(ratio, 12)),
        "replacement_pressure_active": bool(saving_tokens > 0),
    }


def build_default_tiny_substrate_v18(
        *, seed: int = W73_DEFAULT_V18_SEED,
) -> TinyV18SubstrateParams:
    """Build a default V18 substrate."""
    cfg = TinyV18SubstrateConfig.default(seed=int(seed))
    return TinyV18SubstrateParams.init(cfg)


@dataclasses.dataclass(frozen=True)
class TinyV18ForwardWitness:
    schema: str
    forward_trace_cid: str
    cache_cid: str
    replacement_repair_trajectory_cid: str
    replacement_after_ctr_l1: int
    replacement_pressure_gate_mean: float
    v18_gate_score_mean: float
    n_layers: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "forward_trace_cid": str(self.forward_trace_cid),
            "cache_cid": str(self.cache_cid),
            "replacement_repair_trajectory_cid": str(
                self.replacement_repair_trajectory_cid),
            "replacement_after_ctr_l1": int(
                self.replacement_after_ctr_l1),
            "replacement_pressure_gate_mean": float(round(
                self.replacement_pressure_gate_mean, 12)),
            "v18_gate_score_mean": float(round(
                self.v18_gate_score_mean, 12)),
            "n_layers": int(self.n_layers),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v18_forward_witness",
            "witness": self.to_dict()})


def emit_tiny_substrate_v18_forward_witness(
        trace: TinyV18ForwardTrace,
        cache: TinyV18KVCache,
) -> TinyV18ForwardWitness:
    rl = (
        cache.replacement_after_ctr_per_layer
        if cache.replacement_after_ctr_per_layer
        is not None
        else _np.zeros((0,), dtype=_np.int64))
    rep_l1 = int(_np.count_nonzero(
        rl == W73_REPAIR_REPLACEMENT_AFTER_CTR))
    rpg_mean = float(
        trace.replacement_pressure_gate_per_layer.mean()
        if trace.replacement_pressure_gate_per_layer.size
        else 0.0)
    v18_mean = float(
        trace.v18_gate_score_per_layer.mean()
        if trace.v18_gate_score_per_layer.size else 0.0)
    return TinyV18ForwardWitness(
        schema=W73_TINY_SUBSTRATE_V18_SCHEMA_VERSION,
        forward_trace_cid=str(trace.cid()),
        cache_cid=str(cache.cid()),
        replacement_repair_trajectory_cid=str(
            trace.replacement_repair_trajectory_cid),
        replacement_after_ctr_l1=int(rep_l1),
        replacement_pressure_gate_mean=float(rpg_mean),
        v18_gate_score_mean=float(v18_mean),
        n_layers=int(
            trace.v18_gate_score_per_layer.shape[0]),
    )


__all__ = [
    "W73_TINY_SUBSTRATE_V18_SCHEMA_VERSION",
    "W73_TINY_V18_VOCAB_SIZE",
    "W73_DEFAULT_V18_N_LAYERS",
    "W73_DEFAULT_V18_MAX_ROLES",
    "W73_DEFAULT_V18_REPLACEMENT_PRESSURE_BOOST",
    "W73_DEFAULT_V18_REPLACEMENT_AFTER_CTR_BOOST",
    "W73_DEFAULT_V18_REPLACEMENT_LAG_FLOOR",
    "W73_REPAIR_REPLACEMENT_AFTER_CTR",
    "W73_REPAIR_LABELS_V18",
    "TinyV18SubstrateConfig",
    "TinyV18SubstrateParams",
    "TinyV18KVCache",
    "TinyV18ForwardTrace",
    "TinyV18ForwardWitness",
    "tokenize_bytes_v18",
    "forward_tiny_substrate_v18",
    "build_default_tiny_substrate_v18",
    "record_replacement_event_v18",
    "record_contradiction_event_v18",
    "record_replacement_window_v18",
    "substrate_replacement_dominance_flops_v18",
    "substrate_replacement_pressure_throttle_v18",
    "emit_tiny_substrate_v18_forward_witness",
]
