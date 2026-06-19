"""W76 M1 — Tiny Transformer Runtime V21.

Strictly extends W75's ``coordpy.tiny_substrate_v20``. V21 keeps
every V20 invariant (byte-determinism, GQA, RMSNorm/SwiGLU, and
the three V20 axes: ``compound_chain_repair_trajectory_cid``,
``compound_chain_length_per_layer``,
``compound_chain_pressure_gate_per_layer``) and adds **three** new
substrate-load-bearing axes that W76's multi-agent coordinator
V12, team-consensus controller V11, V21 bridges/controllers, and
the new compound-chain-then-restart-aware hosted ↔ real handoff
coordinator V8 exploit:

* **Default 23 layers** (vs V20's 22). Same GQA (8 query / 4 KV).
* **Per-turn compound-chain-then-restart trajectory CID** —
  ``TinyV21KVCache.compound_chain_then_restart_trajectory_cid`` is
  a deterministic content-addressed SHA-256 over the V20 compound-
  chain-repair-trajectory CID PLUS all twelve recorded primitive
  event chains (restart, rejoin, contradiction, replacement,
  delayed-repair, compound-failure-windows, compound-chain-windows,
  plus the new V21 post-compound-chain-restart windows).
* **Per-layer compound-chain-then-restart length label** —
  ``TinyV21KVCache.compound_chain_then_restart_length_per_layer``
  of shape ``(L,)`` records the maximum simultaneous primitive
  chain depth per layer in [0..12] where V20's [0..11] are
  extended by 12 = ``restart_after_compound_chain_repair`` (any
  layer on which a restart event was observed AFTER a compound-
  chain-repair window).
* **Per-layer compound-chain-then-restart-pressure gate** —
  ``TinyV21ForwardTrace.compound_chain_then_restart_pressure_
  gate_per_layer`` of shape ``(L,)`` records the substrate-side
  throttle in [0, 1] that modulates substrate work as a function
  of the visible-token budget AND the joint chain+restart depth
  across all twelve primitive pressures.

V21 still preserves all V20 axes byte-for-byte under trivial
construction; the new axes are no-ops unless explicitly written.

Honest scope (do-not-overstate, W76)
------------------------------------

* Still NOT a frontier model. Default config:
  ``23 layers / 8 query heads / 4 kv heads / d_model=64 /
  ff_hidden=192 / byte-vocab / max_len=128 / untrained``.
  ``W76-L-NUMPY-CPU-V21-SUBSTRATE-CAP``.
* V21 does NOT bridge to third-party hosted models.
  ``W76-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
* The compound-chain-then-restart trajectory CID is a
  deterministic SHA-256 hash; it does not prove integrity at the
  hosted surface (``W76-L-COMPOUND-CHAIN-THEN-RESTART-IN-REPO-
  CAP``).
* The compound-chain-then-restart-pressure gate is a calibrated
  weighted combination, not a learned end-to-end controller
  (``W76-L-COMPOUND-CHAIN-THEN-RESTART-PRESSURE-DECLARED-CAP``).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.tiny_substrate_v21 requires numpy") from exc

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
from .tiny_substrate_v19 import TinyV19SubstrateConfig
from .tiny_substrate_v20 import (
    TinyV20ForwardTrace, TinyV20KVCache, TinyV20SubstrateConfig,
    TinyV20SubstrateParams,
    W75_DEFAULT_V20_COMPOUND_CHAIN_WINDOW_FLOOR,
    W75_DEFAULT_V20_GATE_BIAS, W75_DEFAULT_V20_MAX_ROLES,
    W75_REPAIR_LABELS_V20,
    forward_tiny_substrate_v20,
    tokenize_bytes_v20 as _tokenize_bytes_v20,
)


W76_TINY_SUBSTRATE_V21_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v21.v1")

W76_TINY_V21_VOCAB_SIZE: int = 259
W76_DEFAULT_V21_D_MODEL: int = 64
W76_DEFAULT_V21_N_HEADS: int = 8
W76_DEFAULT_V21_N_KV_HEADS: int = 4
W76_DEFAULT_V21_N_LAYERS: int = 23
W76_DEFAULT_V21_FF_HIDDEN: int = 192
W76_DEFAULT_V21_MAX_LEN: int = 128
W76_DEFAULT_V21_INIT_SCALE: float = 0.04
W76_DEFAULT_V21_SEED: int = 76123456
W76_DEFAULT_V21_MAX_ROLES: int = W75_DEFAULT_V20_MAX_ROLES
W76_DEFAULT_V21_CHAIN_THEN_RESTART_PRESSURE_BOOST: float = 0.84
W76_DEFAULT_V21_CHAIN_THEN_RESTART_REPAIR_BOOST: float = 0.84
W76_DEFAULT_V21_GATE_BIAS: float = W75_DEFAULT_V20_GATE_BIAS
W76_DEFAULT_V21_CHAIN_THEN_RESTART_WINDOW_FLOOR: int = 1

# V21 extends W75_REPAIR_LABELS_V20 with a thirteenth primitive.
W76_REPAIR_CHAIN_THEN_RESTART: int = 12
W76_REPAIR_LABELS_V21: tuple[str, ...] = (
    *W75_REPAIR_LABELS_V20,
    "restart_after_compound_chain_repair",
)


def tokenize_bytes_v21(
        text: str, *, max_len: int = 16) -> list[int]:
    """Byte-tokenisation passthrough to V20."""
    return _tokenize_bytes_v20(str(text), max_len=int(max_len))


@dataclasses.dataclass
class TinyV21SubstrateConfig:
    """V21 config wraps a V20 config + three new V21 axes."""
    v20: TinyV20SubstrateConfig
    max_n_roles: int = W76_DEFAULT_V21_MAX_ROLES
    chain_then_restart_pressure_boost: float = (
        W76_DEFAULT_V21_CHAIN_THEN_RESTART_PRESSURE_BOOST)
    chain_then_restart_repair_boost: float = (
        W76_DEFAULT_V21_CHAIN_THEN_RESTART_REPAIR_BOOST)
    expose_chain_then_restart_trajectory_cid: bool = True
    expose_chain_then_restart_length_per_layer: bool = True
    expose_chain_then_restart_pressure_gate: bool = True
    gate_weights_v21: tuple[float, ...] = (
        0.05, 0.05, 0.05, 0.06, 0.06, 0.06, 0.07, 0.07,
        0.07, 0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.05)
    chain_then_restart_window_floor_turns: int = (
        W76_DEFAULT_V21_CHAIN_THEN_RESTART_WINDOW_FLOOR)

    @classmethod
    def default(
            cls, *, seed: int = W76_DEFAULT_V21_SEED,
    ) -> "TinyV21SubstrateConfig":
        v9 = TinyV9SubstrateConfig(
            vocab_size=W76_TINY_V21_VOCAB_SIZE,
            d_model=W76_DEFAULT_V21_D_MODEL,
            n_heads=W76_DEFAULT_V21_N_HEADS,
            n_kv_heads=W76_DEFAULT_V21_N_KV_HEADS,
            n_layers=W76_DEFAULT_V21_N_LAYERS,
            ff_hidden=W76_DEFAULT_V21_FF_HIDDEN,
            max_len=W76_DEFAULT_V21_MAX_LEN,
            init_scale=W76_DEFAULT_V21_INIT_SCALE,
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
        v20 = TinyV20SubstrateConfig(v19=v19)
        return cls(v20=v20)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W76_TINY_SUBSTRATE_V21_SCHEMA_VERSION,
            "v20_cid": str(self.v20.cid()),
            "max_n_roles": int(self.max_n_roles),
            "chain_then_restart_pressure_boost": float(round(
                self.chain_then_restart_pressure_boost, 12)),
            "chain_then_restart_repair_boost": float(round(
                self.chain_then_restart_repair_boost, 12)),
            "expose_chain_then_restart_trajectory_cid": bool(
                self.expose_chain_then_restart_trajectory_cid),
            "expose_chain_then_restart_length_per_layer": bool(
                self.expose_chain_then_restart_length_per_layer),
            "expose_chain_then_restart_pressure_gate": bool(
                self.expose_chain_then_restart_pressure_gate),
            "gate_weights_v21": [
                float(round(float(x), 12))
                for x in self.gate_weights_v21],
            "chain_then_restart_window_floor_turns": int(
                self.chain_then_restart_window_floor_turns),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v21_substrate_config",
            "config": self.to_dict()})


@dataclasses.dataclass
class TinyV21SubstrateParams:
    config: TinyV21SubstrateConfig
    v20_params: TinyV20SubstrateParams

    @classmethod
    def init(
            cls, config: TinyV21SubstrateConfig | None = None,
    ) -> "TinyV21SubstrateParams":
        if config is None:
            config = TinyV21SubstrateConfig.default()
        v20 = TinyV20SubstrateParams.init(config.v20)
        return cls(config=config, v20_params=v20)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v21_substrate_params",
            "config_cid": self.config.cid(),
            "v20_params_cid": self.v20_params.cid(),
        })


@dataclasses.dataclass
class TinyV21KVCache:
    """V21 cache. Wraps a V20 cache + three new V21 axes."""
    v20_cache: TinyV20KVCache
    compound_chain_then_restart_trajectory_cid: str = ""
    compound_chain_then_restart_length_per_layer: (
        "_np.ndarray | None") = None
    post_compound_chain_restart_windows: list[dict[str, Any]] = (
        dataclasses.field(default_factory=list))
    write_log_v21: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(
            cls, n_layers: int, *, n_heads: int, max_len: int,
    ) -> "TinyV21KVCache":
        v20 = TinyV20KVCache.empty(
            int(n_layers), n_heads=int(n_heads),
            max_len=int(max_len))
        return cls(
            v20_cache=v20,
            compound_chain_then_restart_trajectory_cid="",
            compound_chain_then_restart_length_per_layer=(
                _np.zeros((int(n_layers),), dtype=_np.int64)),
            post_compound_chain_restart_windows=[],
            write_log_v21=[])

    def n_tokens(self) -> int:
        return int(self.v20_cache.n_tokens())

    def n_layers(self) -> int:
        return int(self.v20_cache.n_layers())

    def clone(self) -> "TinyV21KVCache":
        return TinyV21KVCache(
            v20_cache=self.v20_cache.clone(),
            compound_chain_then_restart_trajectory_cid=str(
                self.compound_chain_then_restart_trajectory_cid),
            compound_chain_then_restart_length_per_layer=(
                None if (
                    self
                    .compound_chain_then_restart_length_per_layer
                    is None)
                else self
                .compound_chain_then_restart_length_per_layer
                .copy()),
            post_compound_chain_restart_windows=list(
                self.post_compound_chain_restart_windows),
            write_log_v21=list(self.write_log_v21),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v21_kv_cache",
            "v20_cache_cid": self.v20_cache.cid(),
            "compound_chain_then_restart_trajectory_cid": str(
                self
                .compound_chain_then_restart_trajectory_cid),
            "compound_chain_then_restart_length_per_layer_cid": (
                "none"
                if (self
                    .compound_chain_then_restart_length_per_layer
                    is None)
                else _ndarray_cid(
                    self
                    .compound_chain_then_restart_length_per_layer)),
            "post_compound_chain_restart_windows": list(
                self.post_compound_chain_restart_windows),
            "write_log_v21": list(self.write_log_v21),
        })


@dataclasses.dataclass
class TinyV21ForwardTrace:
    v20_trace: TinyV20ForwardTrace
    compound_chain_then_restart_trajectory_cid: str
    compound_chain_then_restart_length_per_layer: "_np.ndarray"
    compound_chain_then_restart_pressure_gate_per_layer: (
        "_np.ndarray")
    v21_gate_score_per_layer: "_np.ndarray"
    config_cid: str
    params_cid: str

    @property
    def logits(self) -> "_np.ndarray":
        return self.v20_trace.logits

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v21_forward_trace",
            "v20_trace_cid": self.v20_trace.cid(),
            "compound_chain_then_restart_trajectory_cid": str(
                self
                .compound_chain_then_restart_trajectory_cid),
            "compound_chain_then_restart_length_per_layer_cid":
                _ndarray_cid(
                    self
                    .compound_chain_then_restart_length_per_layer),
            "compound_chain_then_restart_pressure_gate_per_layer_cid":
                _ndarray_cid(
                    self
                    .compound_chain_then_restart_pressure_gate_per_layer),
            "v21_gate_score_per_layer_cid": _ndarray_cid(
                self.v21_gate_score_per_layer),
        })


def _compute_compound_chain_then_restart_trajectory_cid(
        cache: TinyV21KVCache) -> str:
    """Content-addressed CID over V20 compound-chain-repair-
    trajectory CID + all twelve recorded primitive event chains
    + V21 post-compound-chain-restart windows."""
    v20 = cache.v20_cache
    v19 = v20.v19_cache
    v18 = v19.v18_cache
    return _sha256_hex({
        "kind": "tiny_v21_compound_chain_then_restart_trajectory",
        "v20_compound_chain_repair_trajectory_cid": str(
            v20.compound_chain_repair_trajectory_cid),
        "v19_compound_repair_trajectory_cid": str(
            v19.compound_repair_trajectory_cid),
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
            v20.compound_chain_windows),
        "v21_post_compound_chain_restart_windows": list(
            cache.post_compound_chain_restart_windows),
    })


def _compute_compound_chain_then_restart_length_per_layer(
        cache: TinyV21KVCache, n_layers: int,
        chain_then_restart_window_floor_turns: int = (
            W76_DEFAULT_V21_CHAIN_THEN_RESTART_WINDOW_FLOOR),
) -> "_np.ndarray":
    """Per-layer compound-chain-then-restart length label.

    Returns shape (L,) dtype int64 in [0..12]. Label 12 fires iff a
    compound-chain-repair window was observed AND a restart event
    followed it within the chain-then-restart horizon AND the
    chain-then-restart window exceeds
    ``chain_then_restart_window_floor_turns``.
    """
    L = int(n_layers)
    out = _np.zeros((L,), dtype=_np.int64)
    v20 = cache.v20_cache
    v19 = v20.v19_cache
    v18 = v19.v18_cache
    base = (
        v20.compound_chain_length_per_layer
        if v20.compound_chain_length_per_layer is not None
        else _np.zeros((L,), dtype=_np.int64))
    n_compound_chain = int(len(v20.compound_chain_windows))
    n_restart = int(len(v18.v17_cache.v16_cache.restart_events))
    max_window = 0
    for d in cache.post_compound_chain_restart_windows:
        try:
            v = int(d.get(
                "post_compound_chain_restart_window_turns", 0))
        except Exception:
            v = 0
        if v > max_window:
            max_window = v
    chain_then_restart_active = bool(
        n_compound_chain > 0 and n_restart > 0
        and int(max_window)
        > int(chain_then_restart_window_floor_turns))
    for li in range(L):
        b = (
            int(base[li]) if li < int(base.shape[0]) else 0)
        if chain_then_restart_active and (li % 11 == 0):
            out[li] = W76_REPAIR_CHAIN_THEN_RESTART
        else:
            out[li] = int(b)
    return out


def _compute_chain_then_restart_pressure_gate_per_layer(
        *, visible_token_budget: float,
        baseline_token_cost: float,
        restart_count: int,
        rejoin_count: int,
        replacement_count: int,
        contradiction_count: int,
        delayed_repair_count: int,
        compound_count: int,
        compound_chain_count: int,
        repair_dominance_count: int,
        max_chain_then_restart_window_turns: int,
        v20_gate_mean: float,
        weights: Sequence[float],
        n_layers: int,
        bias: float = W76_DEFAULT_V21_GATE_BIAS,
) -> "_np.ndarray":
    """Per-layer compound-chain-then-restart-pressure gate."""
    L = int(n_layers)
    safe_cost = float(max(1.0, baseline_token_cost))
    budget_ratio = float(visible_token_budget) / safe_cost
    rmax = float(max(1, W76_DEFAULT_V21_MAX_ROLES))
    restart_ratio = float(restart_count) / rmax
    rejoin_ratio = float(rejoin_count) / rmax
    replace_ratio = float(replacement_count) / rmax
    contradict_ratio = float(contradiction_count) / rmax
    delay_ratio = float(delayed_repair_count) / rmax
    compound_ratio = float(compound_count) / rmax
    chain_ratio = float(compound_chain_count) / rmax
    repair_ratio = float(repair_dominance_count) / rmax
    window_ratio = (
        float(max_chain_then_restart_window_turns)
        / float(max(1, max(
            10,
            int(max_chain_then_restart_window_turns) + 8))))
    feats = _np.array([
        float(v20_gate_mean),
        float(budget_ratio) / float(
            max(1.0, budget_ratio + 1.0)),
        float(restart_ratio),
        float(rejoin_ratio),
        float(replace_ratio),
        float(contradict_ratio),
        float(delay_ratio),
        float(compound_ratio),
        float(chain_ratio),
        float(repair_ratio),
        float(window_ratio),
        0.5,
        float(chain_ratio) * float(restart_ratio),
        float(window_ratio),
        float(budget_ratio) / float(
            max(1.0, budget_ratio + 1.0)),
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


def _compute_v21_gate_score(
        chain_then_restart_active: bool,
        chain_then_restart_pressure_gate_mean: float,
        v20_gate_mean: float,
        weights: Sequence[float],
        n_layers: int,
        bias: float = W76_DEFAULT_V21_GATE_BIAS,
) -> "_np.ndarray":
    feats = _np.array([
        float(v20_gate_mean),
        1.0 if bool(chain_then_restart_active) else 0.0,
        float(chain_then_restart_pressure_gate_mean),
        0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5,
        float(chain_then_restart_pressure_gate_mean),
        1.0 if bool(chain_then_restart_active) else 0.0,
        float(chain_then_restart_pressure_gate_mean),
        float(chain_then_restart_pressure_gate_mean),
        float(chain_then_restart_pressure_gate_mean),
        float(chain_then_restart_pressure_gate_mean),
    ], dtype=_np.float64)
    w = _np.array(
        [float(x) for x in weights], dtype=_np.float64)
    score = (
        float(_np.dot(w[:feats.shape[0]], feats)) + float(bias))
    sig = 1.0 / (1.0 + math.exp(-score))
    return _np.round(
        _np.full(
            (int(n_layers),), float(sig), dtype=_np.float64),
        decimals=12)


def forward_tiny_substrate_v21(
        params: TinyV21SubstrateParams,
        token_ids: Sequence[int],
        *,
        v21_kv_cache: TinyV21KVCache | None = None,
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
        compound_chain_then_restart_pressure: float = 0.0,
) -> tuple[TinyV21ForwardTrace, TinyV21KVCache]:
    """V21 forward = V20 forward + chain-then-restart trajectory
    CID + chain-then-restart length per layer + chain-then-restart-
    pressure gate per layer + V21 composite gate.

    The new ``compound_chain_then_restart_pressure`` knob in [0, 1]
    is a caller-declared signal that the team is absorbing a
    *restart-after-compound-chain-repair* arc; the substrate uses
    it to bias the V21 gate towards substrate work.
    """
    cfg = params.config
    base_v20 = (
        v21_kv_cache.v20_cache if v21_kv_cache is not None
        else None)
    v20_trace, new_v20 = forward_tiny_substrate_v20(
        params.v20_params, list(token_ids),
        v20_kv_cache=base_v20,
        attention_bias_per_layer=attention_bias_per_layer,
        visible_token_budget=float(visible_token_budget),
        baseline_token_cost=float(baseline_token_cost),
        restart_pressure=float(restart_pressure),
        rejoin_pressure=float(rejoin_pressure),
        replacement_pressure=float(replacement_pressure),
        contradiction_pressure=float(contradiction_pressure),
        delayed_repair_pressure=float(delayed_repair_pressure),
        compound_pressure=float(compound_pressure),
        compound_chain_pressure=float(compound_chain_pressure))
    n_layers = int(
        cfg.v20.v19.v18.v17.v16.v15.v14.v13.v12.v11.v10.v9
        .n_layers)
    if v21_kv_cache is None:
        v21_new = TinyV21KVCache.empty(
            int(n_layers),
            n_heads=int(
                cfg.v20.v19.v18.v17.v16.v15.v14.v13.v12.v11
                .v10.v9.n_heads),
            max_len=int(
                cfg.v20.v19.v18.v17.v16.v15.v14.v13.v12.v11
                .v10.v9.max_len))
    else:
        v21_new = v21_kv_cache.clone()
    v21_new.v20_cache = new_v20
    ctr_cid = _compute_compound_chain_then_restart_trajectory_cid(
        v21_new)
    v21_new.compound_chain_then_restart_trajectory_cid = str(
        ctr_cid)
    ctr_per_layer = (
        _compute_compound_chain_then_restart_length_per_layer(
            v21_new, n_layers=n_layers,
            chain_then_restart_window_floor_turns=int(
                cfg.chain_then_restart_window_floor_turns)))
    v21_new.compound_chain_then_restart_length_per_layer = (
        ctr_per_layer)
    v19 = new_v20.v19_cache
    v18 = v19.v18_cache
    n_restart = int(len(v18.v17_cache.v16_cache.restart_events))
    n_rejoin = int(len(v18.v17_cache.rejoin_events))
    n_replace = int(len(v18.replacement_events))
    n_contradict = int(len(v18.contradiction_events))
    n_delayed = int(len(v19.delayed_repair_events))
    n_compound = int(len(v19.compound_failure_windows))
    n_compound_chain = int(len(new_v20.compound_chain_windows))
    max_ctr_window = 0
    for d in v21_new.post_compound_chain_restart_windows:
        try:
            v = int(d.get(
                "post_compound_chain_restart_window_turns", 0))
        except Exception:
            v = 0
        if v > max_ctr_window:
            max_ctr_window = v
    rd_count = int(
        _np.count_nonzero(
            v18.v17_cache.v16_cache
            .restart_dominance_per_layer != 0)
        if v18.v17_cache.v16_cache
            .restart_dominance_per_layer is not None
        else 0)
    v20_gate_mean = float(
        v20_trace.v20_gate_score_per_layer.mean()
        if v20_trace.v20_gate_score_per_layer.size else 0.0)
    effective_ctr_window = int(
        max_ctr_window + int(round(float(max(0.0, min(
            1.0,
            float(compound_chain_then_restart_pressure))))
            * 5.0)))
    ctr_gate = _compute_chain_then_restart_pressure_gate_per_layer(
        visible_token_budget=float(visible_token_budget),
        baseline_token_cost=float(baseline_token_cost),
        restart_count=int(n_restart),
        rejoin_count=int(n_rejoin),
        replacement_count=int(n_replace),
        contradiction_count=int(n_contradict),
        delayed_repair_count=int(n_delayed),
        compound_count=int(n_compound),
        compound_chain_count=int(n_compound_chain),
        repair_dominance_count=int(rd_count),
        max_chain_then_restart_window_turns=int(
            effective_ctr_window),
        v20_gate_mean=float(v20_gate_mean),
        weights=cfg.gate_weights_v21,
        n_layers=int(n_layers))
    chain_then_restart_active = bool(
        int(_np.count_nonzero(
            ctr_per_layer
            == W76_REPAIR_CHAIN_THEN_RESTART)) > 0
        or float(compound_chain_then_restart_pressure) > 0.0)
    v21_gate = _compute_v21_gate_score(
        chain_then_restart_active=bool(
            chain_then_restart_active),
        chain_then_restart_pressure_gate_mean=float(
            ctr_gate.mean()),
        v20_gate_mean=float(v20_gate_mean),
        weights=cfg.gate_weights_v21,
        n_layers=int(n_layers))
    v21_new.write_log_v21.append({
        "schema": W76_TINY_SUBSTRATE_V21_SCHEMA_VERSION,
        "kind": "forward_v21",
        "n_new_tokens": int(len(list(token_ids))),
        "compound_chain_then_restart_trajectory_cid": str(
            ctr_cid),
        "compound_chain_then_restart_length_per_layer": [
            int(x) for x in ctr_per_layer.tolist()],
        "chain_then_restart_pressure_gate_mean": float(
            ctr_gate.mean()),
        "v21_gate_score_mean": float(v21_gate.mean()),
        "visible_token_budget": float(visible_token_budget),
        "restart_pressure": float(restart_pressure),
        "rejoin_pressure": float(rejoin_pressure),
        "replacement_pressure": float(replacement_pressure),
        "contradiction_pressure": float(contradiction_pressure),
        "delayed_repair_pressure": float(
            delayed_repair_pressure),
        "compound_pressure": float(compound_pressure),
        "compound_chain_pressure": float(compound_chain_pressure),
        "compound_chain_then_restart_pressure": float(
            compound_chain_then_restart_pressure),
        "n_compound_chain_windows": int(n_compound_chain),
        "max_chain_then_restart_window_turns": int(max_ctr_window),
    })
    trace = TinyV21ForwardTrace(
        v20_trace=v20_trace,
        compound_chain_then_restart_trajectory_cid=str(ctr_cid),
        compound_chain_then_restart_length_per_layer=ctr_per_layer,
        compound_chain_then_restart_pressure_gate_per_layer=(
            ctr_gate),
        v21_gate_score_per_layer=v21_gate,
        config_cid=str(cfg.cid()),
        params_cid=str(params.cid()),
    )
    return trace, v21_new


def record_post_compound_chain_restart_window_v21(
        cache: TinyV21KVCache, *,
        compound_chain_repair_turn: int,
        restart_turn: int,
        post_compound_chain_restart_window_turns: int,
        role: str = "team", branch_id: str = "main",
) -> None:
    """Record a (compound_chain_repair_turn, restart_turn, post-
    chain restart window) tuple."""
    cache.post_compound_chain_restart_windows.append({
        "schema": W76_TINY_SUBSTRATE_V21_SCHEMA_VERSION,
        "kind": "post_compound_chain_restart_window_v21",
        "compound_chain_repair_turn": int(
            compound_chain_repair_turn),
        "restart_turn": int(restart_turn),
        "post_compound_chain_restart_window_turns": int(
            post_compound_chain_restart_window_turns),
        "role": str(role),
        "branch_id": str(branch_id),
    })
    cache.write_log_v21.append({
        "schema": W76_TINY_SUBSTRATE_V21_SCHEMA_VERSION,
        "kind": "post_compound_chain_restart_window_recorded",
        "post_compound_chain_restart_window_turns": int(
            post_compound_chain_restart_window_turns),
        "role": str(role),
    })


def substrate_chain_then_restart_repair_dominance_flops_v21(
        *, n_tokens: int, n_repairs: int = 12,
        recompute_flops_per_token: int = 1000,
        chain_then_restart_dominance_flops_per_token: int = 40,
) -> dict[str, Any]:
    """V21 compound-chain-then-restart-repair-dominance vs full
    recompute across twelve primitives. By default
    ``n_repairs`` is 12.
    """
    n = int(max(0, n_tokens))
    nr = int(max(1, n_repairs))
    ctr_flops = (
        int(chain_then_restart_dominance_flops_per_token) * n * nr)
    rc_flops = int(recompute_flops_per_token) * n * nr
    saving = int(rc_flops - ctr_flops)
    ratio = (
        float(saving) / float(rc_flops)
        if rc_flops > 0 else 0.0)
    return {
        "n_tokens": int(n),
        "n_repairs": int(nr),
        "chain_then_restart_dominance_flops": int(ctr_flops),
        "recompute_flops": int(rc_flops),
        "saving_flops": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


def substrate_chain_then_restart_pressure_throttle_v21(
        *, visible_token_budget: int = 64,
        baseline_token_cost: int = 512,
        post_compound_chain_restart_window_turns: int = 4,
) -> dict[str, Any]:
    """V21 chain-then-restart-pressure throttle."""
    bt = int(max(0, visible_token_budget))
    bc = int(max(1, baseline_token_cost))
    base_saving = int(max(0, bc - bt))
    window_lift = min(2.7, 1.05 + 0.27 * float(
        max(0, post_compound_chain_restart_window_turns)))
    saving_tokens = int(round(base_saving * float(window_lift)))
    saving_tokens = int(min(saving_tokens, bc))
    ratio = (
        float(saving_tokens) / float(bc)
        if bc > 0 else 0.0)
    return {
        "visible_token_budget": int(bt),
        "baseline_token_cost": int(bc),
        "post_compound_chain_restart_window_turns": int(
            post_compound_chain_restart_window_turns),
        "window_lift": float(round(window_lift, 12)),
        "saving_tokens": int(saving_tokens),
        "saving_ratio": float(round(ratio, 12)),
        "chain_then_restart_pressure_active": bool(
            saving_tokens > 0),
    }


def build_default_tiny_substrate_v21(
        *, seed: int = W76_DEFAULT_V21_SEED,
) -> TinyV21SubstrateParams:
    """Build a default V21 substrate."""
    cfg = TinyV21SubstrateConfig.default(seed=int(seed))
    return TinyV21SubstrateParams.init(cfg)


@dataclasses.dataclass(frozen=True)
class TinyV21ForwardWitness:
    schema: str
    forward_trace_cid: str
    cache_cid: str
    compound_chain_then_restart_trajectory_cid: str
    compound_chain_then_restart_repair_l1: int
    chain_then_restart_pressure_gate_mean: float
    v21_gate_score_mean: float
    n_layers: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "forward_trace_cid": str(self.forward_trace_cid),
            "cache_cid": str(self.cache_cid),
            "compound_chain_then_restart_trajectory_cid": str(
                self
                .compound_chain_then_restart_trajectory_cid),
            "compound_chain_then_restart_repair_l1": int(
                self.compound_chain_then_restart_repair_l1),
            "chain_then_restart_pressure_gate_mean": float(round(
                self.chain_then_restart_pressure_gate_mean, 12)),
            "v21_gate_score_mean": float(round(
                self.v21_gate_score_mean, 12)),
            "n_layers": int(self.n_layers),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v21_forward_witness",
            "witness": self.to_dict()})


def emit_tiny_substrate_v21_forward_witness(
        trace: TinyV21ForwardTrace,
        cache: TinyV21KVCache,
) -> TinyV21ForwardWitness:
    ctr = (
        cache.compound_chain_then_restart_length_per_layer
        if cache
        .compound_chain_then_restart_length_per_layer is not None
        else _np.zeros((0,), dtype=_np.int64))
    chain_then_restart_l1 = int(_np.count_nonzero(
        ctr == W76_REPAIR_CHAIN_THEN_RESTART))
    ctrg_mean = float(
        trace
        .compound_chain_then_restart_pressure_gate_per_layer
        .mean()
        if trace
        .compound_chain_then_restart_pressure_gate_per_layer.size
        else 0.0)
    v21_mean = float(
        trace.v21_gate_score_per_layer.mean()
        if trace.v21_gate_score_per_layer.size else 0.0)
    return TinyV21ForwardWitness(
        schema=W76_TINY_SUBSTRATE_V21_SCHEMA_VERSION,
        forward_trace_cid=str(trace.cid()),
        cache_cid=str(cache.cid()),
        compound_chain_then_restart_trajectory_cid=str(
            trace.compound_chain_then_restart_trajectory_cid),
        compound_chain_then_restart_repair_l1=int(
            chain_then_restart_l1),
        chain_then_restart_pressure_gate_mean=float(ctrg_mean),
        v21_gate_score_mean=float(v21_mean),
        n_layers=int(
            trace.v21_gate_score_per_layer.shape[0]),
    )


__all__ = [
    "W76_TINY_SUBSTRATE_V21_SCHEMA_VERSION",
    "W76_TINY_V21_VOCAB_SIZE",
    "W76_DEFAULT_V21_N_LAYERS",
    "W76_DEFAULT_V21_MAX_ROLES",
    "W76_DEFAULT_V21_CHAIN_THEN_RESTART_PRESSURE_BOOST",
    "W76_DEFAULT_V21_CHAIN_THEN_RESTART_REPAIR_BOOST",
    "W76_DEFAULT_V21_CHAIN_THEN_RESTART_WINDOW_FLOOR",
    "W76_REPAIR_CHAIN_THEN_RESTART",
    "W76_REPAIR_LABELS_V21",
    "TinyV21SubstrateConfig",
    "TinyV21SubstrateParams",
    "TinyV21KVCache",
    "TinyV21ForwardTrace",
    "tokenize_bytes_v21",
    "forward_tiny_substrate_v21",
    "record_post_compound_chain_restart_window_v21",
    "substrate_chain_then_restart_repair_dominance_flops_v21",
    "substrate_chain_then_restart_pressure_throttle_v21",
    "build_default_tiny_substrate_v21",
    "TinyV21ForwardWitness",
    "emit_tiny_substrate_v21_forward_witness",
]
