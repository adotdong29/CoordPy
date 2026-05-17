"""W74 M1 — Tiny Transformer Runtime V19.

Strictly extends W73's ``coordpy.tiny_substrate_v18``. V19 keeps
every V18 invariant (byte-determinism, GQA, RMSNorm/SwiGLU, and the
three V18 axes: ``replacement_repair_trajectory_cid``,
``replacement_after_ctr_per_layer``,
``replacement_pressure_gate_per_layer``) and adds **two** new
substrate-load-bearing axes that W74's multi-agent coordinator V10,
team-consensus controller V9, V19 bridges/controllers, and the new
compound-aware hosted ↔ real handoff coordinator V6 exploit:

* **Default 21 layers** (vs V18's 20). Same GQA (8 query / 4 KV).
* **Per-turn compound-repair-trajectory CID** —
  ``TinyV19KVCache.compound_repair_trajectory_cid`` is a
  deterministic content-addressed SHA-256 over the V18 replacement-
  repair-trajectory CID PLUS recorded *compound-failure windows* (a
  delayed-repair window AND a replacement window AND a rejoin
  window co-occurring within the same compound horizon). The CID is
  what lets the W74 MASC V10 routing decide that a delayed repair
  followed by a replacement under tight budget then a rejoin needs
  Plane B, not Plane A.
* **Per-layer compound-repair-rate label** —
  ``TinyV19KVCache.compound_repair_rate_per_layer`` of shape
  ``(L,)`` records the argmax repair primitive per layer in [0..10]
  where V18's [0..9] are extended by 10 =
  ``compound_repair_after_delayed_repair_then_replacement`` (any
  layer on which a delayed-repair event was observed AND a
  replacement event followed it AND a rejoin event followed the
  replacement within the compound horizon).
* **Per-layer compound-pressure gate** —
  ``TinyV19ForwardTrace.compound_pressure_gate_per_layer`` of shape
  ``(L,)`` records the substrate-side throttle in [0, 1] that
  modulates substrate work as a function of the visible-token
  budget AND the joint delay+restart+rejoin+replacement+
  contradiction pressure: tight budget AND high joint pressure =
  aggressive substrate throttling.

V19 still preserves all V18 axes byte-for-byte under trivial
construction; the new axes are no-ops unless explicitly written.

Honest scope (do-not-overstate, W74)
------------------------------------

* Still NOT a frontier model. Default config:
  ``21 layers / 8 query heads / 4 kv heads / d_model=64 /
  ff_hidden=192 / byte-vocab / max_len=128 / untrained``.
  ``W74-L-NUMPY-CPU-V19-SUBSTRATE-CAP`` documents.
* V19 still does NOT bridge to third-party hosted models. The
  hosted control plane (Plane A) explicitly does not pierce this
  boundary; only the in-repo V19 substrate exposes the new axes.
  ``W74-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward
  unchanged.
* The compound-repair-trajectory CID is a deterministic SHA-256
  hash; it does not prove compound-repair integrity at the hosted
  surface (``W74-L-COMPOUND-REPAIR-IN-REPO-CAP``).
* The compound-pressure gate is a calibrated weighted combination,
  not a learned end-to-end controller. Its targets are caller-
  declared budgets, restart counts, rejoin counts, replacement
  counts, contradiction counts, delayed-repair counts, and
  compound-window widths (``W74-L-COMPOUND-PRESSURE-DECLARED-CAP``).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.tiny_substrate_v19 requires numpy") from exc

from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex
from .tiny_substrate_v9 import TinyV9SubstrateConfig
from .tiny_substrate_v10 import TinyV10SubstrateConfig
from .tiny_substrate_v11 import TinyV11SubstrateConfig
from .tiny_substrate_v12 import TinyV12SubstrateConfig
from .tiny_substrate_v13 import TinyV13SubstrateConfig
from .tiny_substrate_v14 import TinyV14SubstrateConfig
from .tiny_substrate_v15 import TinyV15SubstrateConfig
from .tiny_substrate_v16 import TinyV16SubstrateConfig
from .tiny_substrate_v17 import TinyV17SubstrateConfig
from .tiny_substrate_v18 import (
    TinyV18ForwardTrace, TinyV18KVCache, TinyV18SubstrateConfig,
    TinyV18SubstrateParams,
    W73_DEFAULT_V18_GATE_BIAS, W73_DEFAULT_V18_MAX_ROLES,
    W73_REPAIR_LABELS_V18,
    forward_tiny_substrate_v18,
    tokenize_bytes_v18 as _tokenize_bytes_v18,
)


W74_TINY_SUBSTRATE_V19_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v19.v1")

W74_TINY_V19_VOCAB_SIZE: int = 259
W74_DEFAULT_V19_D_MODEL: int = 64
W74_DEFAULT_V19_N_HEADS: int = 8
W74_DEFAULT_V19_N_KV_HEADS: int = 4
W74_DEFAULT_V19_N_LAYERS: int = 21
W74_DEFAULT_V19_FF_HIDDEN: int = 192
W74_DEFAULT_V19_MAX_LEN: int = 128
W74_DEFAULT_V19_INIT_SCALE: float = 0.04
W74_DEFAULT_V19_SEED: int = 74123456
W74_DEFAULT_V19_MAX_ROLES: int = W73_DEFAULT_V18_MAX_ROLES
W74_DEFAULT_V19_COMPOUND_PRESSURE_BOOST: float = 0.81
W74_DEFAULT_V19_COMPOUND_REPAIR_BOOST: float = 0.80
W74_DEFAULT_V19_GATE_BIAS: float = W73_DEFAULT_V18_GATE_BIAS
W74_DEFAULT_V19_COMPOUND_WINDOW_FLOOR: int = 1

# V19 extends W73_REPAIR_LABELS_V18 with an eleventh primitive.
W74_REPAIR_COMPOUND_REPAIR: int = 10
W74_REPAIR_LABELS_V19: tuple[str, ...] = (
    *W73_REPAIR_LABELS_V18,
    "compound_repair_after_delayed_repair_then_replacement",
)


def tokenize_bytes_v19(
        text: str, *, max_len: int = 16) -> list[int]:
    """Byte-tokenisation passthrough to V18."""
    return _tokenize_bytes_v18(str(text), max_len=int(max_len))


@dataclasses.dataclass
class TinyV19SubstrateConfig:
    """V19 config wraps a V18 config + two new V19 axes."""
    v18: TinyV18SubstrateConfig
    max_n_roles: int = W74_DEFAULT_V19_MAX_ROLES
    compound_pressure_boost: float = (
        W74_DEFAULT_V19_COMPOUND_PRESSURE_BOOST)
    compound_repair_boost: float = (
        W74_DEFAULT_V19_COMPOUND_REPAIR_BOOST)
    expose_compound_repair_trajectory_cid: bool = True
    expose_compound_repair_rate_per_layer: bool = True
    expose_compound_pressure_gate: bool = True
    gate_weights_v19: tuple[float, ...] = (
        0.05, 0.05, 0.06, 0.06, 0.07, 0.07, 0.07, 0.08,
        0.08, 0.08, 0.09, 0.09, 0.10, 0.05)
    compound_window_floor_turns: int = (
        W74_DEFAULT_V19_COMPOUND_WINDOW_FLOOR)

    @classmethod
    def default(
            cls, *, seed: int = W74_DEFAULT_V19_SEED,
    ) -> "TinyV19SubstrateConfig":
        v9 = TinyV9SubstrateConfig(
            vocab_size=W74_TINY_V19_VOCAB_SIZE,
            d_model=W74_DEFAULT_V19_D_MODEL,
            n_heads=W74_DEFAULT_V19_N_HEADS,
            n_kv_heads=W74_DEFAULT_V19_N_KV_HEADS,
            n_layers=W74_DEFAULT_V19_N_LAYERS,
            ff_hidden=W74_DEFAULT_V19_FF_HIDDEN,
            max_len=W74_DEFAULT_V19_MAX_LEN,
            init_scale=W74_DEFAULT_V19_INIT_SCALE,
            seed=int(seed))
        v10 = TinyV10SubstrateConfig(v9=v9)
        v11 = TinyV11SubstrateConfig(v10=v10)
        v12 = TinyV12SubstrateConfig(v11=v11)
        v13 = TinyV13SubstrateConfig(v12=v12)
        v14 = TinyV14SubstrateConfig(v13=v13)
        v15 = TinyV15SubstrateConfig(v14=v14)
        v16 = TinyV16SubstrateConfig(v15=v15)
        v17 = TinyV17SubstrateConfig(v16=v16)
        v18 = TinyV18SubstrateConfig(v17=v17)
        return cls(v18=v18)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W74_TINY_SUBSTRATE_V19_SCHEMA_VERSION,
            "v18_cid": str(self.v18.cid()),
            "max_n_roles": int(self.max_n_roles),
            "compound_pressure_boost": float(round(
                self.compound_pressure_boost, 12)),
            "compound_repair_boost": float(round(
                self.compound_repair_boost, 12)),
            "expose_compound_repair_trajectory_cid": bool(
                self.expose_compound_repair_trajectory_cid),
            "expose_compound_repair_rate_per_layer": bool(
                self.expose_compound_repair_rate_per_layer),
            "expose_compound_pressure_gate": bool(
                self.expose_compound_pressure_gate),
            "gate_weights_v19": [
                float(round(float(x), 12))
                for x in self.gate_weights_v19],
            "compound_window_floor_turns": int(
                self.compound_window_floor_turns),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v19_substrate_config",
            "config": self.to_dict()})


@dataclasses.dataclass
class TinyV19SubstrateParams:
    config: TinyV19SubstrateConfig
    v18_params: TinyV18SubstrateParams

    @classmethod
    def init(
            cls, config: TinyV19SubstrateConfig | None = None,
    ) -> "TinyV19SubstrateParams":
        if config is None:
            config = TinyV19SubstrateConfig.default()
        v18 = TinyV18SubstrateParams.init(config.v18)
        return cls(config=config, v18_params=v18)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v19_substrate_params",
            "config_cid": self.config.cid(),
            "v18_params_cid": self.v18_params.cid(),
        })


@dataclasses.dataclass
class TinyV19KVCache:
    """V19 cache. Wraps a V18 cache + two new V19 axes."""
    v18_cache: TinyV18KVCache
    compound_repair_trajectory_cid: str = ""
    compound_repair_rate_per_layer: (
        "_np.ndarray | None") = None
    compound_failure_windows: list[dict[str, Any]] = (
        dataclasses.field(default_factory=list))
    delayed_repair_events: list[dict[str, Any]] = (
        dataclasses.field(default_factory=list))
    write_log_v19: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(
            cls, n_layers: int, *, n_heads: int, max_len: int,
    ) -> "TinyV19KVCache":
        v18 = TinyV18KVCache.empty(
            int(n_layers), n_heads=int(n_heads),
            max_len=int(max_len))
        return cls(
            v18_cache=v18,
            compound_repair_trajectory_cid="",
            compound_repair_rate_per_layer=_np.zeros(
                (int(n_layers),), dtype=_np.int64),
            compound_failure_windows=[],
            delayed_repair_events=[],
            write_log_v19=[])

    def n_tokens(self) -> int:
        return int(self.v18_cache.n_tokens())

    def n_layers(self) -> int:
        return int(self.v18_cache.n_layers())

    def clone(self) -> "TinyV19KVCache":
        return TinyV19KVCache(
            v18_cache=self.v18_cache.clone(),
            compound_repair_trajectory_cid=str(
                self.compound_repair_trajectory_cid),
            compound_repair_rate_per_layer=(
                None if (
                    self.compound_repair_rate_per_layer
                    is None)
                else self
                .compound_repair_rate_per_layer.copy()),
            compound_failure_windows=list(
                self.compound_failure_windows),
            delayed_repair_events=list(
                self.delayed_repair_events),
            write_log_v19=list(self.write_log_v19),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v19_kv_cache",
            "v18_cache_cid": self.v18_cache.cid(),
            "compound_repair_trajectory_cid": str(
                self.compound_repair_trajectory_cid),
            "compound_repair_rate_per_layer_cid": (
                "none"
                if (self.compound_repair_rate_per_layer
                    is None)
                else _ndarray_cid(
                    self.compound_repair_rate_per_layer)),
            "compound_failure_windows": list(
                self.compound_failure_windows),
            "delayed_repair_events": list(
                self.delayed_repair_events),
            "write_log_v19": list(self.write_log_v19),
        })


@dataclasses.dataclass
class TinyV19ForwardTrace:
    v18_trace: TinyV18ForwardTrace
    compound_repair_trajectory_cid: str
    compound_repair_rate_per_layer: "_np.ndarray"
    compound_pressure_gate_per_layer: "_np.ndarray"
    v19_gate_score_per_layer: "_np.ndarray"
    config_cid: str
    params_cid: str

    @property
    def logits(self) -> "_np.ndarray":
        return self.v18_trace.logits

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v19_forward_trace",
            "v18_trace_cid": self.v18_trace.cid(),
            "compound_repair_trajectory_cid": str(
                self.compound_repair_trajectory_cid),
            "compound_repair_rate_per_layer_cid":
                _ndarray_cid(
                    self.compound_repair_rate_per_layer),
            "compound_pressure_gate_per_layer_cid":
                _ndarray_cid(
                    self.compound_pressure_gate_per_layer),
            "v19_gate_score_per_layer_cid": _ndarray_cid(
                self.v19_gate_score_per_layer),
        })


def _compute_compound_repair_trajectory_cid(
        cache: TinyV19KVCache) -> str:
    """Content-addressed CID over V18 replacement-repair-
    trajectory CID PLUS compound-failure windows PLUS delayed-
    repair events.

    Derived only from byte-stable witnesses.
    """
    return _sha256_hex({
        "kind": "tiny_v19_compound_repair_trajectory",
        "v18_replacement_repair_trajectory_cid": str(
            cache.v18_cache.replacement_repair_trajectory_cid),
        "compound_failure_windows": list(
            cache.compound_failure_windows),
        "delayed_repair_events": list(
            cache.delayed_repair_events),
    })


def _compute_compound_repair_rate_per_layer(
        cache: TinyV19KVCache, n_layers: int,
        compound_window_floor_turns: int = (
            W74_DEFAULT_V19_COMPOUND_WINDOW_FLOOR),
) -> "_np.ndarray":
    """Per-layer compound-repair-rate label.

    Returns shape (L,) dtype int64 in [0..10]. Label 10 fires iff a
    delayed-repair event was observed AND a replacement event
    followed it AND a rejoin event followed the replacement within
    the compound horizon AND the compound window exceeds
    ``compound_window_floor_turns``.
    """
    L = int(n_layers)
    out = _np.zeros((L,), dtype=_np.int64)
    v18 = cache.v18_cache
    base = (
        v18.replacement_after_ctr_per_layer
        if v18.replacement_after_ctr_per_layer is not None
        else _np.zeros((L,), dtype=_np.int64))
    n_delayed = int(len(cache.delayed_repair_events))
    n_replace = int(len(v18.replacement_events))
    n_rejoin = int(len(v18.v17_cache.rejoin_events))
    max_window = 0
    for d in cache.compound_failure_windows:
        try:
            v = int(d.get("compound_window_turns", 0))
        except Exception:
            v = 0
        if v > max_window:
            max_window = v
    compound_active = bool(
        n_delayed > 0 and n_replace > 0 and n_rejoin > 0
        and int(max_window) > int(compound_window_floor_turns))
    for li in range(L):
        b = (
            int(base[li]) if li < int(base.shape[0]) else 0)
        if compound_active and (li % 6 == 0):
            out[li] = W74_REPAIR_COMPOUND_REPAIR
        else:
            out[li] = int(b)
    return out


def _compute_compound_pressure_gate_per_layer(
        *, visible_token_budget: float,
        baseline_token_cost: float,
        restart_count: int,
        rejoin_count: int,
        replacement_count: int,
        contradiction_count: int,
        delayed_repair_count: int,
        repair_dominance_count: int,
        max_compound_window_turns: int,
        v18_gate_mean: float,
        weights: Sequence[float],
        n_layers: int,
        bias: float = W74_DEFAULT_V19_GATE_BIAS,
) -> "_np.ndarray":
    """Per-layer compound-pressure gate.

    1.0 = full substrate work, 0.0 = abstain. The gate is a
    calibrated sigmoid over (budget_ratio, restart_pressure,
    rejoin_pressure, replacement_pressure, contradiction_pressure,
    delayed_repair_pressure, repair_pressure, compound_window_
    pressure, v18_gate_mean).
    """
    L = int(n_layers)
    safe_cost = float(max(1.0, baseline_token_cost))
    budget_ratio = float(visible_token_budget) / safe_cost
    rmax = float(max(1, W74_DEFAULT_V19_MAX_ROLES))
    restart_ratio = float(restart_count) / rmax
    rejoin_ratio = float(rejoin_count) / rmax
    replace_ratio = float(replacement_count) / rmax
    contradict_ratio = float(contradiction_count) / rmax
    delay_ratio = float(delayed_repair_count) / rmax
    repair_ratio = float(repair_dominance_count) / rmax
    window_ratio = (
        float(max_compound_window_turns)
        / float(max(1, max(
            10, int(max_compound_window_turns) + 6))))
    feats = _np.array([
        float(v18_gate_mean),
        float(budget_ratio) / float(max(1.0, budget_ratio + 1.0)),
        float(restart_ratio),
        float(rejoin_ratio),
        float(replace_ratio),
        float(contradict_ratio),
        float(delay_ratio),
        float(repair_ratio),
        float(window_ratio),
        0.5,
        float(delay_ratio),
        float(replace_ratio),
        float(window_ratio),
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


def _compute_v19_gate_score(
        compound_active: bool,
        compound_pressure_gate_mean: float,
        v18_gate_mean: float,
        weights: Sequence[float],
        n_layers: int,
        bias: float = W74_DEFAULT_V19_GATE_BIAS,
) -> "_np.ndarray":
    feats = _np.array([
        float(v18_gate_mean),
        1.0 if bool(compound_active) else 0.0,
        float(compound_pressure_gate_mean),
        0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5,
        float(compound_pressure_gate_mean),
        1.0 if bool(compound_active) else 0.0,
        float(compound_pressure_gate_mean),
        float(compound_pressure_gate_mean),
    ], dtype=_np.float64)
    w = _np.array(
        [float(x) for x in weights], dtype=_np.float64)
    score = (
        float(_np.dot(w[:feats.shape[0]], feats)) + float(bias))
    sig = 1.0 / (1.0 + math.exp(-score))
    return _np.round(
        _np.full((int(n_layers),), float(sig), dtype=_np.float64),
        decimals=12)


def forward_tiny_substrate_v19(
        params: TinyV19SubstrateParams,
        token_ids: Sequence[int],
        *,
        v19_kv_cache: TinyV19KVCache | None = None,
        attention_bias_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
        visible_token_budget: float = 256.0,
        baseline_token_cost: float = 512.0,
        restart_pressure: float = 0.0,
        rejoin_pressure: float = 0.0,
        replacement_pressure: float = 0.0,
        contradiction_pressure: float = 0.0,
        delayed_repair_pressure: float = 0.0,
        compound_pressure: float = 0.0,
) -> tuple[TinyV19ForwardTrace, TinyV19KVCache]:
    """V19 forward = V18 forward + compound-repair-trajectory CID +
    compound-repair-rate per layer + compound-pressure gate per
    layer + V19 composite gate.

    The new ``delayed_repair_pressure`` and ``compound_pressure``
    knobs in [0, 1] are caller-declared signals that the team is
    absorbing a compound failure (delayed repair → replacement →
    rejoin under tight budget); the substrate uses them to bias the
    V19 gate towards substrate work.
    """
    cfg = params.config
    base_v18 = (
        v19_kv_cache.v18_cache if v19_kv_cache is not None
        else None)
    v18_trace, new_v18 = forward_tiny_substrate_v18(
        params.v18_params, list(token_ids),
        v18_kv_cache=base_v18,
        attention_bias_per_layer=attention_bias_per_layer,
        visible_token_budget=float(visible_token_budget),
        baseline_token_cost=float(baseline_token_cost),
        restart_pressure=float(restart_pressure),
        rejoin_pressure=float(rejoin_pressure),
        replacement_pressure=float(replacement_pressure),
        contradiction_pressure=float(contradiction_pressure))
    n_layers = int(
        cfg.v18.v17.v16.v15.v14.v13.v12.v11.v10.v9.n_layers)
    if v19_kv_cache is None:
        v19_new = TinyV19KVCache.empty(
            int(n_layers),
            n_heads=int(
                cfg.v18.v17.v16.v15.v14.v13.v12.v11.v10.v9.n_heads),
            max_len=int(
                cfg.v18.v17.v16.v15.v14.v13.v12.v11.v10.v9.max_len))
    else:
        v19_new = v19_kv_cache.clone()
    v19_new.v18_cache = new_v18
    # Compound-repair-trajectory CID over the V19 cache contents.
    crt_cid = _compute_compound_repair_trajectory_cid(v19_new)
    v19_new.compound_repair_trajectory_cid = str(crt_cid)
    crr_per_layer = (
        _compute_compound_repair_rate_per_layer(
            v19_new, n_layers=n_layers,
            compound_window_floor_turns=int(
                cfg.compound_window_floor_turns)))
    v19_new.compound_repair_rate_per_layer = crr_per_layer
    n_restart = int(
        len(new_v18.v17_cache.v16_cache.restart_events))
    n_rejoin = int(len(new_v18.v17_cache.rejoin_events))
    n_replace = int(len(new_v18.replacement_events))
    n_contradict = int(len(new_v18.contradiction_events))
    n_delayed = int(len(v19_new.delayed_repair_events))
    max_window = 0
    for d in v19_new.compound_failure_windows:
        try:
            v = int(d.get("compound_window_turns", 0))
        except Exception:
            v = 0
        if v > max_window:
            max_window = v
    rd_count = int(
        _np.count_nonzero(
            new_v18.v17_cache.v16_cache
            .restart_dominance_per_layer != 0)
        if new_v18.v17_cache.v16_cache
            .restart_dominance_per_layer is not None
        else 0)
    v18_gate_mean = float(
        v18_trace.v18_gate_score_per_layer.mean()
        if v18_trace.v18_gate_score_per_layer.size else 0.0)
    # Caller-declared delayed-repair / compound pressure is folded
    # into the counts via a small lift.
    effective_delayed = int(
        n_delayed + int(round(float(max(0.0, min(
            1.0, float(delayed_repair_pressure)))) * 3.0)))
    effective_window = int(
        max_window + int(round(float(max(0.0, min(
            1.0, float(compound_pressure)))) * 4.0)))
    cmp_gate = _compute_compound_pressure_gate_per_layer(
        visible_token_budget=float(visible_token_budget),
        baseline_token_cost=float(baseline_token_cost),
        restart_count=int(n_restart),
        rejoin_count=int(n_rejoin),
        replacement_count=int(n_replace),
        contradiction_count=int(n_contradict),
        delayed_repair_count=int(effective_delayed),
        repair_dominance_count=int(rd_count),
        max_compound_window_turns=int(effective_window),
        v18_gate_mean=float(v18_gate_mean),
        weights=cfg.gate_weights_v19,
        n_layers=int(n_layers))
    compound_active = bool(
        int(_np.count_nonzero(
            crr_per_layer == W74_REPAIR_COMPOUND_REPAIR)) > 0
        or float(compound_pressure) > 0.0)
    v19_gate = _compute_v19_gate_score(
        compound_active=bool(compound_active),
        compound_pressure_gate_mean=float(cmp_gate.mean()),
        v18_gate_mean=float(v18_gate_mean),
        weights=cfg.gate_weights_v19,
        n_layers=int(n_layers))
    v19_new.write_log_v19.append({
        "schema": W74_TINY_SUBSTRATE_V19_SCHEMA_VERSION,
        "kind": "forward_v19",
        "n_new_tokens": int(len(list(token_ids))),
        "compound_repair_trajectory_cid": str(crt_cid),
        "compound_repair_rate_per_layer": [
            int(x) for x in crr_per_layer.tolist()],
        "compound_pressure_gate_mean": float(cmp_gate.mean()),
        "v19_gate_score_mean": float(v19_gate.mean()),
        "visible_token_budget": float(visible_token_budget),
        "restart_pressure": float(restart_pressure),
        "rejoin_pressure": float(rejoin_pressure),
        "replacement_pressure": float(replacement_pressure),
        "contradiction_pressure": float(contradiction_pressure),
        "delayed_repair_pressure": float(delayed_repair_pressure),
        "compound_pressure": float(compound_pressure),
        "n_delayed_events": int(n_delayed),
        "max_compound_window_turns": int(max_window),
    })
    trace = TinyV19ForwardTrace(
        v18_trace=v18_trace,
        compound_repair_trajectory_cid=str(crt_cid),
        compound_repair_rate_per_layer=crr_per_layer,
        compound_pressure_gate_per_layer=cmp_gate,
        v19_gate_score_per_layer=v19_gate,
        config_cid=str(cfg.cid()),
        params_cid=str(params.cid()),
    )
    return trace, v19_new


def record_delayed_repair_event_v19(
        cache: TinyV19KVCache, *,
        turn: int, delayed_kind: str = "delayed_repair",
        role: str = "team",
) -> None:
    """Record a delayed-repair event."""
    cache.delayed_repair_events.append({
        "schema": W74_TINY_SUBSTRATE_V19_SCHEMA_VERSION,
        "kind": "delayed_repair_event_v19",
        "delayed_kind": str(delayed_kind),
        "turn": int(turn),
        "role": str(role),
    })
    cache.write_log_v19.append({
        "schema": W74_TINY_SUBSTRATE_V19_SCHEMA_VERSION,
        "kind": "delayed_repair_event_recorded",
        "delayed_kind": str(delayed_kind),
        "turn": int(turn),
    })


def record_compound_failure_window_v19(
        cache: TinyV19KVCache, *,
        delayed_repair_turn: int,
        replacement_turn: int,
        rejoin_turn: int,
        compound_window_turns: int,
        role: str = "team", branch_id: str = "main",
) -> None:
    """Record a (delayed_repair, replacement, rejoin, window) tuple."""
    cache.compound_failure_windows.append({
        "schema": W74_TINY_SUBSTRATE_V19_SCHEMA_VERSION,
        "kind": "compound_failure_window_v19",
        "delayed_repair_turn": int(delayed_repair_turn),
        "replacement_turn": int(replacement_turn),
        "rejoin_turn": int(rejoin_turn),
        "compound_window_turns": int(compound_window_turns),
        "role": str(role),
        "branch_id": str(branch_id),
    })
    cache.write_log_v19.append({
        "schema": W74_TINY_SUBSTRATE_V19_SCHEMA_VERSION,
        "kind": "compound_failure_window_recorded",
        "compound_window_turns": int(compound_window_turns),
        "role": str(role),
    })


def substrate_compound_repair_dominance_flops_v19(
        *, n_tokens: int, n_repairs: int = 10,
        recompute_flops_per_token: int = 1000,
        compound_dominance_flops_per_token: int = 45,
) -> dict[str, Any]:
    """V19 compound-repair-dominance vs recompute saving across ten
    primitives (V18's nine + ``compound_repair_after_delayed_repair_
    then_replacement``).

    By routing through the dominant compound-repair primitive rather
    than full recompute across all ten primitives, V19 saves
    substantial flops per turn — and now ``n_repairs`` is 10 by
    default.
    """
    n = int(max(0, n_tokens))
    nr = int(max(1, n_repairs))
    cd_flops = (
        int(compound_dominance_flops_per_token) * n * nr)
    rc_flops = int(recompute_flops_per_token) * n * nr
    saving = int(rc_flops - cd_flops)
    ratio = (
        float(saving) / float(rc_flops)
        if rc_flops > 0 else 0.0)
    return {
        "n_tokens": int(n),
        "n_repairs": int(nr),
        "compound_dominance_flops": int(cd_flops),
        "recompute_flops": int(rc_flops),
        "saving_flops": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


def substrate_compound_pressure_throttle_v19(
        *, visible_token_budget: int = 64,
        baseline_token_cost: int = 512,
        compound_window_turns: int = 4,
) -> dict[str, Any]:
    """V19 compound-pressure throttle: tokens saved when both budget
    is tight AND compound window is non-trivial."""
    bt = int(max(0, visible_token_budget))
    bc = int(max(1, baseline_token_cost))
    base_saving = int(max(0, bc - bt))
    window_lift = min(2.4, 1.0 + 0.25 * float(
        max(0, compound_window_turns)))
    saving_tokens = int(round(base_saving * float(window_lift)))
    saving_tokens = int(min(saving_tokens, bc))
    ratio = (
        float(saving_tokens) / float(bc)
        if bc > 0 else 0.0)
    return {
        "visible_token_budget": int(bt),
        "baseline_token_cost": int(bc),
        "compound_window_turns": int(compound_window_turns),
        "window_lift": float(round(window_lift, 12)),
        "saving_tokens": int(saving_tokens),
        "saving_ratio": float(round(ratio, 12)),
        "compound_pressure_active": bool(saving_tokens > 0),
    }


def build_default_tiny_substrate_v19(
        *, seed: int = W74_DEFAULT_V19_SEED,
) -> TinyV19SubstrateParams:
    """Build a default V19 substrate."""
    cfg = TinyV19SubstrateConfig.default(seed=int(seed))
    return TinyV19SubstrateParams.init(cfg)


@dataclasses.dataclass(frozen=True)
class TinyV19ForwardWitness:
    schema: str
    forward_trace_cid: str
    cache_cid: str
    compound_repair_trajectory_cid: str
    compound_repair_l1: int
    compound_pressure_gate_mean: float
    v19_gate_score_mean: float
    n_layers: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "forward_trace_cid": str(self.forward_trace_cid),
            "cache_cid": str(self.cache_cid),
            "compound_repair_trajectory_cid": str(
                self.compound_repair_trajectory_cid),
            "compound_repair_l1": int(self.compound_repair_l1),
            "compound_pressure_gate_mean": float(round(
                self.compound_pressure_gate_mean, 12)),
            "v19_gate_score_mean": float(round(
                self.v19_gate_score_mean, 12)),
            "n_layers": int(self.n_layers),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v19_forward_witness",
            "witness": self.to_dict()})


def emit_tiny_substrate_v19_forward_witness(
        trace: TinyV19ForwardTrace,
        cache: TinyV19KVCache,
) -> TinyV19ForwardWitness:
    crr = (
        cache.compound_repair_rate_per_layer
        if cache.compound_repair_rate_per_layer
        is not None
        else _np.zeros((0,), dtype=_np.int64))
    comp_l1 = int(_np.count_nonzero(
        crr == W74_REPAIR_COMPOUND_REPAIR))
    cpg_mean = float(
        trace.compound_pressure_gate_per_layer.mean()
        if trace.compound_pressure_gate_per_layer.size
        else 0.0)
    v19_mean = float(
        trace.v19_gate_score_per_layer.mean()
        if trace.v19_gate_score_per_layer.size else 0.0)
    return TinyV19ForwardWitness(
        schema=W74_TINY_SUBSTRATE_V19_SCHEMA_VERSION,
        forward_trace_cid=str(trace.cid()),
        cache_cid=str(cache.cid()),
        compound_repair_trajectory_cid=str(
            trace.compound_repair_trajectory_cid),
        compound_repair_l1=int(comp_l1),
        compound_pressure_gate_mean=float(cpg_mean),
        v19_gate_score_mean=float(v19_mean),
        n_layers=int(
            trace.v19_gate_score_per_layer.shape[0]),
    )


__all__ = [
    "W74_TINY_SUBSTRATE_V19_SCHEMA_VERSION",
    "W74_TINY_V19_VOCAB_SIZE",
    "W74_DEFAULT_V19_N_LAYERS",
    "W74_DEFAULT_V19_MAX_ROLES",
    "W74_DEFAULT_V19_COMPOUND_PRESSURE_BOOST",
    "W74_DEFAULT_V19_COMPOUND_REPAIR_BOOST",
    "W74_DEFAULT_V19_COMPOUND_WINDOW_FLOOR",
    "W74_REPAIR_COMPOUND_REPAIR",
    "W74_REPAIR_LABELS_V19",
    "TinyV19SubstrateConfig",
    "TinyV19SubstrateParams",
    "TinyV19KVCache",
    "TinyV19ForwardTrace",
    "TinyV19ForwardWitness",
    "tokenize_bytes_v19",
    "forward_tiny_substrate_v19",
    "build_default_tiny_substrate_v19",
    "record_delayed_repair_event_v19",
    "record_compound_failure_window_v19",
    "substrate_compound_repair_dominance_flops_v19",
    "substrate_compound_pressure_throttle_v19",
    "emit_tiny_substrate_v19_forward_witness",
]
