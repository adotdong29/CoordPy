"""W75 M1 — Tiny Transformer Runtime V20.

Strictly extends W74's ``coordpy.tiny_substrate_v19``. V20 keeps
every V19 invariant (byte-determinism, GQA, RMSNorm/SwiGLU, and the
three V19 axes: ``compound_repair_trajectory_cid``,
``compound_repair_rate_per_layer``,
``compound_pressure_gate_per_layer``) and adds **three** new
substrate-load-bearing axes that W75's multi-agent coordinator V11,
team-consensus controller V10, V20 bridges/controllers, and the new
compound-chain-aware hosted ↔ real handoff coordinator V7 exploit:

* **Default 22 layers** (vs V19's 21). Same GQA (8 query / 4 KV).
* **Per-turn compound-chain repair-trajectory CID** —
  ``TinyV20KVCache.compound_chain_repair_trajectory_cid`` is a
  deterministic content-addressed SHA-256 over the V19 compound-
  repair-trajectory CID PLUS all eleven recorded primitive event
  chains (restart, rejoin, contradiction, replacement, delayed-
  repair, compound-failure-windows, plus the new V20
  replacement-then-rejoin compound-chain windows). The CID is what
  lets the W75 MASC V11 routing decide that a *replacement first*
  followed by a *delayed repair* then a *rejoin* under tight budget
  is a distinct compound chain that needs Plane B, not Plane A.
* **Per-layer compound-chain-length label** —
  ``TinyV20KVCache.compound_chain_length_per_layer`` of shape
  ``(L,)`` records the maximum simultaneous primitive-chain depth
  per layer in [0..11] where V19's [0..10] are extended by 11 =
  ``compound_repair_after_replacement_then_rejoin`` (any layer on
  which a replacement event was observed AND a delayed-repair event
  followed it AND a rejoin event followed the delayed-repair within
  the compound-chain horizon).
* **Per-layer compound-chain-pressure gate** —
  ``TinyV20ForwardTrace.compound_chain_pressure_gate_per_layer`` of
  shape ``(L,)`` records the substrate-side throttle in [0, 1] that
  modulates substrate work as a function of the visible-token
  budget AND the joint chain depth across all eleven primitive
  pressures: tight budget AND high joint chain depth = aggressive
  substrate throttling.

V20 still preserves all V19 axes byte-for-byte under trivial
construction; the new axes are no-ops unless explicitly written.

Honest scope (do-not-overstate, W75)
------------------------------------

* Still NOT a frontier model. Default config:
  ``22 layers / 8 query heads / 4 kv heads / d_model=64 /
  ff_hidden=192 / byte-vocab / max_len=128 / untrained``.
  ``W75-L-NUMPY-CPU-V20-SUBSTRATE-CAP`` documents.
* V20 still does NOT bridge to third-party hosted models. The
  hosted control plane (Plane A) explicitly does not pierce this
  boundary; only the in-repo V20 substrate exposes the new axes.
  ``W75-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward
  unchanged.
* The compound-chain repair-trajectory CID is a deterministic SHA-
  256 hash; it does not prove compound-chain integrity at the
  hosted surface (``W75-L-COMPOUND-CHAIN-REPAIR-IN-REPO-CAP``).
* The compound-chain-pressure gate is a calibrated weighted
  combination, not a learned end-to-end controller. Its targets are
  caller-declared budgets, restart counts, rejoin counts,
  replacement counts, contradiction counts, delayed-repair counts,
  compound-window widths, AND compound-chain-window widths
  (``W75-L-COMPOUND-CHAIN-PRESSURE-DECLARED-CAP``).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.tiny_substrate_v20 requires numpy") from exc

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
from .tiny_substrate_v18 import TinyV18SubstrateConfig
from .tiny_substrate_v19 import (
    TinyV19ForwardTrace, TinyV19KVCache, TinyV19SubstrateConfig,
    TinyV19SubstrateParams,
    W74_DEFAULT_V19_COMPOUND_WINDOW_FLOOR,
    W74_DEFAULT_V19_GATE_BIAS, W74_DEFAULT_V19_MAX_ROLES,
    W74_REPAIR_LABELS_V19,
    forward_tiny_substrate_v19,
    tokenize_bytes_v19 as _tokenize_bytes_v19,
)


W75_TINY_SUBSTRATE_V20_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v20.v1")

W75_TINY_V20_VOCAB_SIZE: int = 259
W75_DEFAULT_V20_D_MODEL: int = 64
W75_DEFAULT_V20_N_HEADS: int = 8
W75_DEFAULT_V20_N_KV_HEADS: int = 4
W75_DEFAULT_V20_N_LAYERS: int = 22
W75_DEFAULT_V20_FF_HIDDEN: int = 192
W75_DEFAULT_V20_MAX_LEN: int = 128
W75_DEFAULT_V20_INIT_SCALE: float = 0.04
W75_DEFAULT_V20_SEED: int = 75123456
W75_DEFAULT_V20_MAX_ROLES: int = W74_DEFAULT_V19_MAX_ROLES
W75_DEFAULT_V20_COMPOUND_CHAIN_PRESSURE_BOOST: float = 0.82
W75_DEFAULT_V20_COMPOUND_CHAIN_REPAIR_BOOST: float = 0.82
W75_DEFAULT_V20_GATE_BIAS: float = W74_DEFAULT_V19_GATE_BIAS
W75_DEFAULT_V20_COMPOUND_CHAIN_WINDOW_FLOOR: int = 1

# V20 extends W74_REPAIR_LABELS_V19 with a twelfth primitive.
W75_REPAIR_COMPOUND_CHAIN_REPAIR: int = 11
W75_REPAIR_LABELS_V20: tuple[str, ...] = (
    *W74_REPAIR_LABELS_V19,
    "compound_repair_after_replacement_then_rejoin",
)


def tokenize_bytes_v20(
        text: str, *, max_len: int = 16) -> list[int]:
    """Byte-tokenisation passthrough to V19."""
    return _tokenize_bytes_v19(str(text), max_len=int(max_len))


@dataclasses.dataclass
class TinyV20SubstrateConfig:
    """V20 config wraps a V19 config + three new V20 axes."""
    v19: TinyV19SubstrateConfig
    max_n_roles: int = W75_DEFAULT_V20_MAX_ROLES
    compound_chain_pressure_boost: float = (
        W75_DEFAULT_V20_COMPOUND_CHAIN_PRESSURE_BOOST)
    compound_chain_repair_boost: float = (
        W75_DEFAULT_V20_COMPOUND_CHAIN_REPAIR_BOOST)
    expose_compound_chain_repair_trajectory_cid: bool = True
    expose_compound_chain_length_per_layer: bool = True
    expose_compound_chain_pressure_gate: bool = True
    gate_weights_v20: tuple[float, ...] = (
        0.05, 0.05, 0.05, 0.06, 0.06, 0.07, 0.07, 0.07,
        0.08, 0.08, 0.08, 0.09, 0.09, 0.10, 0.05)
    compound_chain_window_floor_turns: int = (
        W75_DEFAULT_V20_COMPOUND_CHAIN_WINDOW_FLOOR)

    @classmethod
    def default(
            cls, *, seed: int = W75_DEFAULT_V20_SEED,
    ) -> "TinyV20SubstrateConfig":
        v9 = TinyV9SubstrateConfig(
            vocab_size=W75_TINY_V20_VOCAB_SIZE,
            d_model=W75_DEFAULT_V20_D_MODEL,
            n_heads=W75_DEFAULT_V20_N_HEADS,
            n_kv_heads=W75_DEFAULT_V20_N_KV_HEADS,
            n_layers=W75_DEFAULT_V20_N_LAYERS,
            ff_hidden=W75_DEFAULT_V20_FF_HIDDEN,
            max_len=W75_DEFAULT_V20_MAX_LEN,
            init_scale=W75_DEFAULT_V20_INIT_SCALE,
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
        v19 = TinyV19SubstrateConfig(v18=v18)
        return cls(v19=v19)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W75_TINY_SUBSTRATE_V20_SCHEMA_VERSION,
            "v19_cid": str(self.v19.cid()),
            "max_n_roles": int(self.max_n_roles),
            "compound_chain_pressure_boost": float(round(
                self.compound_chain_pressure_boost, 12)),
            "compound_chain_repair_boost": float(round(
                self.compound_chain_repair_boost, 12)),
            "expose_compound_chain_repair_trajectory_cid": bool(
                self.expose_compound_chain_repair_trajectory_cid),
            "expose_compound_chain_length_per_layer": bool(
                self.expose_compound_chain_length_per_layer),
            "expose_compound_chain_pressure_gate": bool(
                self.expose_compound_chain_pressure_gate),
            "gate_weights_v20": [
                float(round(float(x), 12))
                for x in self.gate_weights_v20],
            "compound_chain_window_floor_turns": int(
                self.compound_chain_window_floor_turns),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v20_substrate_config",
            "config": self.to_dict()})


@dataclasses.dataclass
class TinyV20SubstrateParams:
    config: TinyV20SubstrateConfig
    v19_params: TinyV19SubstrateParams

    @classmethod
    def init(
            cls, config: TinyV20SubstrateConfig | None = None,
    ) -> "TinyV20SubstrateParams":
        if config is None:
            config = TinyV20SubstrateConfig.default()
        v19 = TinyV19SubstrateParams.init(config.v19)
        return cls(config=config, v19_params=v19)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v20_substrate_params",
            "config_cid": self.config.cid(),
            "v19_params_cid": self.v19_params.cid(),
        })


@dataclasses.dataclass
class TinyV20KVCache:
    """V20 cache. Wraps a V19 cache + three new V20 axes."""
    v19_cache: TinyV19KVCache
    compound_chain_repair_trajectory_cid: str = ""
    compound_chain_length_per_layer: (
        "_np.ndarray | None") = None
    compound_chain_windows: list[dict[str, Any]] = (
        dataclasses.field(default_factory=list))
    write_log_v20: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(
            cls, n_layers: int, *, n_heads: int, max_len: int,
    ) -> "TinyV20KVCache":
        v19 = TinyV19KVCache.empty(
            int(n_layers), n_heads=int(n_heads),
            max_len=int(max_len))
        return cls(
            v19_cache=v19,
            compound_chain_repair_trajectory_cid="",
            compound_chain_length_per_layer=_np.zeros(
                (int(n_layers),), dtype=_np.int64),
            compound_chain_windows=[],
            write_log_v20=[])

    def n_tokens(self) -> int:
        return int(self.v19_cache.n_tokens())

    def n_layers(self) -> int:
        return int(self.v19_cache.n_layers())

    def clone(self) -> "TinyV20KVCache":
        return TinyV20KVCache(
            v19_cache=self.v19_cache.clone(),
            compound_chain_repair_trajectory_cid=str(
                self.compound_chain_repair_trajectory_cid),
            compound_chain_length_per_layer=(
                None if (
                    self.compound_chain_length_per_layer
                    is None)
                else self
                .compound_chain_length_per_layer.copy()),
            compound_chain_windows=list(
                self.compound_chain_windows),
            write_log_v20=list(self.write_log_v20),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v20_kv_cache",
            "v19_cache_cid": self.v19_cache.cid(),
            "compound_chain_repair_trajectory_cid": str(
                self.compound_chain_repair_trajectory_cid),
            "compound_chain_length_per_layer_cid": (
                "none"
                if (self.compound_chain_length_per_layer
                    is None)
                else _ndarray_cid(
                    self.compound_chain_length_per_layer)),
            "compound_chain_windows": list(
                self.compound_chain_windows),
            "write_log_v20": list(self.write_log_v20),
        })


@dataclasses.dataclass
class TinyV20ForwardTrace:
    v19_trace: TinyV19ForwardTrace
    compound_chain_repair_trajectory_cid: str
    compound_chain_length_per_layer: "_np.ndarray"
    compound_chain_pressure_gate_per_layer: "_np.ndarray"
    v20_gate_score_per_layer: "_np.ndarray"
    config_cid: str
    params_cid: str

    @property
    def logits(self) -> "_np.ndarray":
        return self.v19_trace.logits

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v20_forward_trace",
            "v19_trace_cid": self.v19_trace.cid(),
            "compound_chain_repair_trajectory_cid": str(
                self.compound_chain_repair_trajectory_cid),
            "compound_chain_length_per_layer_cid":
                _ndarray_cid(
                    self.compound_chain_length_per_layer),
            "compound_chain_pressure_gate_per_layer_cid":
                _ndarray_cid(
                    self.compound_chain_pressure_gate_per_layer),
            "v20_gate_score_per_layer_cid": _ndarray_cid(
                self.v20_gate_score_per_layer),
        })


def _compute_compound_chain_repair_trajectory_cid(
        cache: TinyV20KVCache) -> str:
    """Content-addressed CID over V19 compound-repair-trajectory
    CID + all eleven recorded primitive event chains + V20 compound-
    chain windows.

    Derived only from byte-stable witnesses.
    """
    v19 = cache.v19_cache
    v18 = v19.v18_cache
    return _sha256_hex({
        "kind": "tiny_v20_compound_chain_repair_trajectory",
        "v19_compound_repair_trajectory_cid": str(
            v19.compound_repair_trajectory_cid),
        "v18_replacement_repair_trajectory_cid": str(
            v18.replacement_repair_trajectory_cid),
        "v18_replacement_events": list(
            v18.replacement_events),
        "v18_contradiction_events": list(
            v18.contradiction_events),
        "v17_rejoin_events": list(
            v18.v17_cache.rejoin_events),
        "v16_restart_events": list(
            v18.v17_cache.v16_cache.restart_events),
        "v19_delayed_repair_events": list(
            v19.delayed_repair_events),
        "v19_compound_failure_windows": list(
            v19.compound_failure_windows),
        "v20_compound_chain_windows": list(
            cache.compound_chain_windows),
    })


def _compute_compound_chain_length_per_layer(
        cache: TinyV20KVCache, n_layers: int,
        compound_chain_window_floor_turns: int = (
            W75_DEFAULT_V20_COMPOUND_CHAIN_WINDOW_FLOOR),
) -> "_np.ndarray":
    """Per-layer compound-chain-length label.

    Returns shape (L,) dtype int64 in [0..11]. Label 11 fires iff a
    replacement event was observed AND a delayed-repair event
    followed it AND a rejoin event followed the delayed-repair
    within the compound-chain horizon AND the compound-chain window
    exceeds ``compound_chain_window_floor_turns``.
    """
    L = int(n_layers)
    out = _np.zeros((L,), dtype=_np.int64)
    v19 = cache.v19_cache
    v18 = v19.v18_cache
    base = (
        v19.compound_repair_rate_per_layer
        if v19.compound_repair_rate_per_layer is not None
        else _np.zeros((L,), dtype=_np.int64))
    n_replace = int(len(v18.replacement_events))
    n_delayed = int(len(v19.delayed_repair_events))
    n_rejoin = int(len(v18.v17_cache.rejoin_events))
    max_window = 0
    for d in cache.compound_chain_windows:
        try:
            v = int(d.get("compound_chain_window_turns", 0))
        except Exception:
            v = 0
        if v > max_window:
            max_window = v
    chain_active = bool(
        n_replace > 0 and n_delayed > 0 and n_rejoin > 0
        and int(max_window)
            > int(compound_chain_window_floor_turns))
    for li in range(L):
        b = (
            int(base[li]) if li < int(base.shape[0]) else 0)
        if chain_active and (li % 7 == 0):
            out[li] = W75_REPAIR_COMPOUND_CHAIN_REPAIR
        else:
            out[li] = int(b)
    return out


def _compute_compound_chain_pressure_gate_per_layer(
        *, visible_token_budget: float,
        baseline_token_cost: float,
        restart_count: int,
        rejoin_count: int,
        replacement_count: int,
        contradiction_count: int,
        delayed_repair_count: int,
        compound_count: int,
        repair_dominance_count: int,
        max_compound_chain_window_turns: int,
        v19_gate_mean: float,
        weights: Sequence[float],
        n_layers: int,
        bias: float = W75_DEFAULT_V20_GATE_BIAS,
) -> "_np.ndarray":
    """Per-layer compound-chain-pressure gate.

    1.0 = full substrate work, 0.0 = abstain. The gate is a
    calibrated sigmoid over (budget_ratio, restart_pressure,
    rejoin_pressure, replacement_pressure, contradiction_pressure,
    delayed_repair_pressure, compound_pressure, repair_pressure,
    chain_window_pressure, v19_gate_mean).
    """
    L = int(n_layers)
    safe_cost = float(max(1.0, baseline_token_cost))
    budget_ratio = float(visible_token_budget) / safe_cost
    rmax = float(max(1, W75_DEFAULT_V20_MAX_ROLES))
    restart_ratio = float(restart_count) / rmax
    rejoin_ratio = float(rejoin_count) / rmax
    replace_ratio = float(replacement_count) / rmax
    contradict_ratio = float(contradiction_count) / rmax
    delay_ratio = float(delayed_repair_count) / rmax
    compound_ratio = float(compound_count) / rmax
    repair_ratio = float(repair_dominance_count) / rmax
    window_ratio = (
        float(max_compound_chain_window_turns)
        / float(max(1, max(
            10, int(max_compound_chain_window_turns) + 8))))
    feats = _np.array([
        float(v19_gate_mean),
        float(budget_ratio) / float(max(1.0, budget_ratio + 1.0)),
        float(restart_ratio),
        float(rejoin_ratio),
        float(replace_ratio),
        float(contradict_ratio),
        float(delay_ratio),
        float(compound_ratio),
        float(repair_ratio),
        float(window_ratio),
        0.5,
        float(compound_ratio),
        float(window_ratio),
        float(budget_ratio) / float(max(1.0, budget_ratio + 1.0)),
        float(window_ratio),
    ], dtype=_np.float64)
    w = _np.array(
        [float(x) for x in weights], dtype=_np.float64)
    score = (
        float(_np.dot(w[:feats.shape[0]], feats)) + float(bias))
    sig = 1.0 / (1.0 + math.exp(-score))
    per_layer = _np.full(
        (L,), float(sig), dtype=_np.float64)
    return _np.round(per_layer, decimals=12)


def _compute_v20_gate_score(
        chain_active: bool,
        compound_chain_pressure_gate_mean: float,
        v19_gate_mean: float,
        weights: Sequence[float],
        n_layers: int,
        bias: float = W75_DEFAULT_V20_GATE_BIAS,
) -> "_np.ndarray":
    feats = _np.array([
        float(v19_gate_mean),
        1.0 if bool(chain_active) else 0.0,
        float(compound_chain_pressure_gate_mean),
        0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5,
        float(compound_chain_pressure_gate_mean),
        1.0 if bool(chain_active) else 0.0,
        float(compound_chain_pressure_gate_mean),
        float(compound_chain_pressure_gate_mean),
        float(compound_chain_pressure_gate_mean),
    ], dtype=_np.float64)
    w = _np.array(
        [float(x) for x in weights], dtype=_np.float64)
    score = (
        float(_np.dot(w[:feats.shape[0]], feats)) + float(bias))
    sig = 1.0 / (1.0 + math.exp(-score))
    return _np.round(
        _np.full((int(n_layers),), float(sig), dtype=_np.float64),
        decimals=12)


def forward_tiny_substrate_v20(
        params: TinyV20SubstrateParams,
        token_ids: Sequence[int],
        *,
        v20_kv_cache: TinyV20KVCache | None = None,
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
        compound_chain_pressure: float = 0.0,
) -> tuple[TinyV20ForwardTrace, TinyV20KVCache]:
    """V20 forward = V19 forward + compound-chain repair-trajectory
    CID + compound-chain-length per layer + compound-chain-pressure
    gate per layer + V20 composite gate.

    The new ``compound_chain_pressure`` knob in [0, 1] is a caller-
    declared signal that the team is absorbing a compound CHAIN of
    failures (replacement → delayed repair → rejoin under tight
    budget); the substrate uses it to bias the V20 gate towards
    substrate work.
    """
    cfg = params.config
    base_v19 = (
        v20_kv_cache.v19_cache if v20_kv_cache is not None
        else None)
    v19_trace, new_v19 = forward_tiny_substrate_v19(
        params.v19_params, list(token_ids),
        v19_kv_cache=base_v19,
        attention_bias_per_layer=attention_bias_per_layer,
        visible_token_budget=float(visible_token_budget),
        baseline_token_cost=float(baseline_token_cost),
        restart_pressure=float(restart_pressure),
        rejoin_pressure=float(rejoin_pressure),
        replacement_pressure=float(replacement_pressure),
        contradiction_pressure=float(contradiction_pressure),
        delayed_repair_pressure=float(delayed_repair_pressure),
        compound_pressure=float(compound_pressure))
    n_layers = int(
        cfg.v19.v18.v17.v16.v15.v14.v13.v12.v11.v10.v9.n_layers)
    if v20_kv_cache is None:
        v20_new = TinyV20KVCache.empty(
            int(n_layers),
            n_heads=int(
                cfg.v19.v18.v17.v16.v15.v14.v13.v12.v11.v10.v9.n_heads),
            max_len=int(
                cfg.v19.v18.v17.v16.v15.v14.v13.v12.v11.v10.v9.max_len))
    else:
        v20_new = v20_kv_cache.clone()
    v20_new.v19_cache = new_v19
    crt_cid = _compute_compound_chain_repair_trajectory_cid(
        v20_new)
    v20_new.compound_chain_repair_trajectory_cid = str(crt_cid)
    ccl_per_layer = (
        _compute_compound_chain_length_per_layer(
            v20_new, n_layers=n_layers,
            compound_chain_window_floor_turns=int(
                cfg.compound_chain_window_floor_turns)))
    v20_new.compound_chain_length_per_layer = ccl_per_layer
    v18 = new_v19.v18_cache
    n_restart = int(len(v18.v17_cache.v16_cache.restart_events))
    n_rejoin = int(len(v18.v17_cache.rejoin_events))
    n_replace = int(len(v18.replacement_events))
    n_contradict = int(len(v18.contradiction_events))
    n_delayed = int(len(new_v19.delayed_repair_events))
    n_compound = int(len(new_v19.compound_failure_windows))
    max_chain_window = 0
    for d in v20_new.compound_chain_windows:
        try:
            v = int(d.get("compound_chain_window_turns", 0))
        except Exception:
            v = 0
        if v > max_chain_window:
            max_chain_window = v
    rd_count = int(
        _np.count_nonzero(
            v18.v17_cache.v16_cache
            .restart_dominance_per_layer != 0)
        if v18.v17_cache.v16_cache
            .restart_dominance_per_layer is not None
        else 0)
    v19_gate_mean = float(
        v19_trace.v19_gate_score_per_layer.mean()
        if v19_trace.v19_gate_score_per_layer.size else 0.0)
    effective_chain_window = int(
        max_chain_window + int(round(float(max(0.0, min(
            1.0, float(compound_chain_pressure)))) * 5.0)))
    cc_gate = _compute_compound_chain_pressure_gate_per_layer(
        visible_token_budget=float(visible_token_budget),
        baseline_token_cost=float(baseline_token_cost),
        restart_count=int(n_restart),
        rejoin_count=int(n_rejoin),
        replacement_count=int(n_replace),
        contradiction_count=int(n_contradict),
        delayed_repair_count=int(n_delayed),
        compound_count=int(n_compound),
        repair_dominance_count=int(rd_count),
        max_compound_chain_window_turns=int(effective_chain_window),
        v19_gate_mean=float(v19_gate_mean),
        weights=cfg.gate_weights_v20,
        n_layers=int(n_layers))
    chain_active = bool(
        int(_np.count_nonzero(
            ccl_per_layer
            == W75_REPAIR_COMPOUND_CHAIN_REPAIR)) > 0
        or float(compound_chain_pressure) > 0.0)
    v20_gate = _compute_v20_gate_score(
        chain_active=bool(chain_active),
        compound_chain_pressure_gate_mean=float(cc_gate.mean()),
        v19_gate_mean=float(v19_gate_mean),
        weights=cfg.gate_weights_v20,
        n_layers=int(n_layers))
    v20_new.write_log_v20.append({
        "schema": W75_TINY_SUBSTRATE_V20_SCHEMA_VERSION,
        "kind": "forward_v20",
        "n_new_tokens": int(len(list(token_ids))),
        "compound_chain_repair_trajectory_cid": str(crt_cid),
        "compound_chain_length_per_layer": [
            int(x) for x in ccl_per_layer.tolist()],
        "compound_chain_pressure_gate_mean": float(
            cc_gate.mean()),
        "v20_gate_score_mean": float(v20_gate.mean()),
        "visible_token_budget": float(visible_token_budget),
        "restart_pressure": float(restart_pressure),
        "rejoin_pressure": float(rejoin_pressure),
        "replacement_pressure": float(replacement_pressure),
        "contradiction_pressure": float(contradiction_pressure),
        "delayed_repair_pressure": float(delayed_repair_pressure),
        "compound_pressure": float(compound_pressure),
        "compound_chain_pressure": float(compound_chain_pressure),
        "n_delayed_events": int(n_delayed),
        "n_compound_failure_windows": int(n_compound),
        "max_compound_chain_window_turns": int(max_chain_window),
    })
    trace = TinyV20ForwardTrace(
        v19_trace=v19_trace,
        compound_chain_repair_trajectory_cid=str(crt_cid),
        compound_chain_length_per_layer=ccl_per_layer,
        compound_chain_pressure_gate_per_layer=cc_gate,
        v20_gate_score_per_layer=v20_gate,
        config_cid=str(cfg.cid()),
        params_cid=str(params.cid()),
    )
    return trace, v20_new


def record_compound_chain_window_v20(
        cache: TinyV20KVCache, *,
        replacement_turn: int,
        delayed_repair_turn: int,
        rejoin_turn: int,
        compound_chain_window_turns: int,
        role: str = "team", branch_id: str = "main",
) -> None:
    """Record a (replacement, delayed_repair, rejoin, chain-window)
    tuple."""
    cache.compound_chain_windows.append({
        "schema": W75_TINY_SUBSTRATE_V20_SCHEMA_VERSION,
        "kind": "compound_chain_window_v20",
        "replacement_turn": int(replacement_turn),
        "delayed_repair_turn": int(delayed_repair_turn),
        "rejoin_turn": int(rejoin_turn),
        "compound_chain_window_turns": int(
            compound_chain_window_turns),
        "role": str(role),
        "branch_id": str(branch_id),
    })
    cache.write_log_v20.append({
        "schema": W75_TINY_SUBSTRATE_V20_SCHEMA_VERSION,
        "kind": "compound_chain_window_recorded",
        "compound_chain_window_turns": int(
            compound_chain_window_turns),
        "role": str(role),
    })


def substrate_compound_chain_repair_dominance_flops_v20(
        *, n_tokens: int, n_repairs: int = 11,
        recompute_flops_per_token: int = 1000,
        chain_dominance_flops_per_token: int = 42,
) -> dict[str, Any]:
    """V20 compound-chain-repair-dominance vs recompute saving
    across eleven primitives (V19's ten + ``compound_repair_after_
    replacement_then_rejoin``).

    By routing through the dominant compound-chain-repair primitive
    rather than full recompute across all eleven primitives, V20
    saves substantial flops per turn — and now ``n_repairs`` is 11
    by default.
    """
    n = int(max(0, n_tokens))
    nr = int(max(1, n_repairs))
    cd_flops = (
        int(chain_dominance_flops_per_token) * n * nr)
    rc_flops = int(recompute_flops_per_token) * n * nr
    saving = int(rc_flops - cd_flops)
    ratio = (
        float(saving) / float(rc_flops)
        if rc_flops > 0 else 0.0)
    return {
        "n_tokens": int(n),
        "n_repairs": int(nr),
        "chain_dominance_flops": int(cd_flops),
        "recompute_flops": int(rc_flops),
        "saving_flops": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


def substrate_compound_chain_pressure_throttle_v20(
        *, visible_token_budget: int = 64,
        baseline_token_cost: int = 512,
        compound_chain_window_turns: int = 4,
) -> dict[str, Any]:
    """V20 compound-chain-pressure throttle: tokens saved when both
    budget is tight AND compound-chain window is non-trivial."""
    bt = int(max(0, visible_token_budget))
    bc = int(max(1, baseline_token_cost))
    base_saving = int(max(0, bc - bt))
    window_lift = min(2.6, 1.05 + 0.25 * float(
        max(0, compound_chain_window_turns)))
    saving_tokens = int(round(base_saving * float(window_lift)))
    saving_tokens = int(min(saving_tokens, bc))
    ratio = (
        float(saving_tokens) / float(bc)
        if bc > 0 else 0.0)
    return {
        "visible_token_budget": int(bt),
        "baseline_token_cost": int(bc),
        "compound_chain_window_turns": int(
            compound_chain_window_turns),
        "window_lift": float(round(window_lift, 12)),
        "saving_tokens": int(saving_tokens),
        "saving_ratio": float(round(ratio, 12)),
        "compound_chain_pressure_active": bool(
            saving_tokens > 0),
    }


def build_default_tiny_substrate_v20(
        *, seed: int = W75_DEFAULT_V20_SEED,
) -> TinyV20SubstrateParams:
    """Build a default V20 substrate."""
    cfg = TinyV20SubstrateConfig.default(seed=int(seed))
    return TinyV20SubstrateParams.init(cfg)


@dataclasses.dataclass(frozen=True)
class TinyV20ForwardWitness:
    schema: str
    forward_trace_cid: str
    cache_cid: str
    compound_chain_repair_trajectory_cid: str
    compound_chain_repair_l1: int
    compound_chain_pressure_gate_mean: float
    v20_gate_score_mean: float
    n_layers: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "forward_trace_cid": str(self.forward_trace_cid),
            "cache_cid": str(self.cache_cid),
            "compound_chain_repair_trajectory_cid": str(
                self.compound_chain_repair_trajectory_cid),
            "compound_chain_repair_l1": int(
                self.compound_chain_repair_l1),
            "compound_chain_pressure_gate_mean": float(round(
                self.compound_chain_pressure_gate_mean, 12)),
            "v20_gate_score_mean": float(round(
                self.v20_gate_score_mean, 12)),
            "n_layers": int(self.n_layers),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v20_forward_witness",
            "witness": self.to_dict()})


def emit_tiny_substrate_v20_forward_witness(
        trace: TinyV20ForwardTrace,
        cache: TinyV20KVCache,
) -> TinyV20ForwardWitness:
    ccl = (
        cache.compound_chain_length_per_layer
        if cache.compound_chain_length_per_layer
        is not None
        else _np.zeros((0,), dtype=_np.int64))
    chain_l1 = int(_np.count_nonzero(
        ccl == W75_REPAIR_COMPOUND_CHAIN_REPAIR))
    cpg_mean = float(
        trace.compound_chain_pressure_gate_per_layer.mean()
        if trace.compound_chain_pressure_gate_per_layer.size
        else 0.0)
    v20_mean = float(
        trace.v20_gate_score_per_layer.mean()
        if trace.v20_gate_score_per_layer.size else 0.0)
    return TinyV20ForwardWitness(
        schema=W75_TINY_SUBSTRATE_V20_SCHEMA_VERSION,
        forward_trace_cid=str(trace.cid()),
        cache_cid=str(cache.cid()),
        compound_chain_repair_trajectory_cid=str(
            trace.compound_chain_repair_trajectory_cid),
        compound_chain_repair_l1=int(chain_l1),
        compound_chain_pressure_gate_mean=float(cpg_mean),
        v20_gate_score_mean=float(v20_mean),
        n_layers=int(
            trace.v20_gate_score_per_layer.shape[0]),
    )


__all__ = [
    "W75_TINY_SUBSTRATE_V20_SCHEMA_VERSION",
    "W75_TINY_V20_VOCAB_SIZE",
    "W75_DEFAULT_V20_N_LAYERS",
    "W75_DEFAULT_V20_MAX_ROLES",
    "W75_DEFAULT_V20_COMPOUND_CHAIN_PRESSURE_BOOST",
    "W75_DEFAULT_V20_COMPOUND_CHAIN_REPAIR_BOOST",
    "W75_DEFAULT_V20_COMPOUND_CHAIN_WINDOW_FLOOR",
    "W75_REPAIR_COMPOUND_CHAIN_REPAIR",
    "W75_REPAIR_LABELS_V20",
    "TinyV20SubstrateConfig",
    "TinyV20SubstrateParams",
    "TinyV20KVCache",
    "TinyV20ForwardTrace",
    "tokenize_bytes_v20",
    "forward_tiny_substrate_v20",
    "record_compound_chain_window_v20",
    "substrate_compound_chain_repair_dominance_flops_v20",
    "substrate_compound_chain_pressure_throttle_v20",
    "build_default_tiny_substrate_v20",
    "TinyV20ForwardWitness",
    "emit_tiny_substrate_v20_forward_witness",
]
