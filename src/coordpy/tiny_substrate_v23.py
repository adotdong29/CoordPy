"""W78 M1 — Tiny Transformer Runtime V23.

Strictly extends W77's ``coordpy.tiny_substrate_v22``. V23 keeps
every V22 invariant byte-for-byte under trivial construction and
adds **one** new substrate-load-bearing axis: a per-turn
**long-horizon-reconstruction trajectory CID** that unifies the
W77 post-restart-replacement-trajectory CID with the recorded
W78 long-horizon-reconstruction event chain — the carrier the
W78 long-horizon-reconstruction substrate
(``coordpy.long_horizon_reconstruction_substrate_v1``) reads
from to *reconstruct events after a long visible-token blackout*.

* **Inherits V22's 23 physical layers**. V23 adds a fifth axis
  layer in the substrate cache (long-horizon-reconstruction
  axis); the physical transformer depth carries forward from V22
  byte-for-byte.
* **Per-turn long-horizon-reconstruction trajectory CID** —
  ``TinyV23KVCache.long_horizon_reconstruction_trajectory_cid``
  is a deterministic content-addressed SHA-256 over the V22
  post-restart-replacement-trajectory CID PLUS the recorded W78
  long-horizon-reconstruction events.
* **Per-layer long-horizon-reconstruction length label** — shape
  ``(L,)`` int64 in [0..14]. V22's [0..13] are extended by 14 =
  ``long_delay_reconstruction_after_compound_chain_failure``
  (any layer on which a reconstruction event was observed after
  a long-delay blackout window).
* **Per-layer long-horizon-reconstruction-pressure gate** —
  substrate-side throttle in [0, 1] that modulates substrate
  work as a function of: (visible-token budget, blackout window
  length, prior-chain depth, reconstruction-request count).

V23 strictly extends V22.

Honest scope (W78)
------------------

* Still NOT a frontier model. Default config inherited from V22.
  ``W78-L-NUMPY-CPU-V23-SUBSTRATE-CAP``.
* V23 does NOT bridge to third-party hosted models.
  ``W78-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.tiny_substrate_v23 requires numpy") from exc

from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex
from .tiny_substrate_v22 import (
    TinyV22ForwardTrace, TinyV22KVCache, TinyV22SubstrateConfig,
    TinyV22SubstrateParams, W77_DEFAULT_V22_GATE_BIAS,
    W77_DEFAULT_V22_MAX_ROLES, W77_REPAIR_LABELS_V22,
    forward_tiny_substrate_v22,
    tokenize_bytes_v22 as _tokenize_bytes_v22,
)


W78_TINY_SUBSTRATE_V23_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v23.v1")

W78_TINY_V23_VOCAB_SIZE: int = 259
W78_DEFAULT_V23_MAX_ROLES: int = W77_DEFAULT_V22_MAX_ROLES
W78_DEFAULT_V23_LONG_HORIZON_RECONSTRUCTION_PRESSURE_BOOST: float = (
    0.84)
W78_DEFAULT_V23_LONG_HORIZON_RECONSTRUCTION_REPAIR_BOOST: float = (
    0.84)
W78_DEFAULT_V23_GATE_BIAS: float = W77_DEFAULT_V22_GATE_BIAS
W78_DEFAULT_V23_LONG_HORIZON_RECONSTRUCTION_WINDOW_FLOOR: int = 50
W78_DEFAULT_V23_SEED: int = 78123456

# V23 extends W77_REPAIR_LABELS_V22 with a fifteenth primitive.
W78_REPAIR_LONG_DELAY_RECONSTRUCTION: int = 14
W78_REPAIR_LABELS_V23: tuple[str, ...] = (
    *W77_REPAIR_LABELS_V22,
    "long_delay_reconstruction_after_compound_chain_failure",
)


def tokenize_bytes_v23(
        text: str, *, max_len: int = 16) -> list[int]:
    return _tokenize_bytes_v22(str(text), max_len=int(max_len))


@dataclasses.dataclass
class TinyV23SubstrateConfig:
    v22: TinyV22SubstrateConfig
    max_n_roles: int = W78_DEFAULT_V23_MAX_ROLES
    long_horizon_reconstruction_pressure_boost: float = (
        W78_DEFAULT_V23_LONG_HORIZON_RECONSTRUCTION_PRESSURE_BOOST)
    long_horizon_reconstruction_repair_boost: float = (
        W78_DEFAULT_V23_LONG_HORIZON_RECONSTRUCTION_REPAIR_BOOST)
    expose_long_horizon_reconstruction_trajectory_cid: bool = True
    expose_long_horizon_reconstruction_length_per_layer: bool = (
        True)
    expose_long_horizon_reconstruction_pressure_gate: bool = True
    gate_weights_v23: tuple[float, ...] = (
        0.04, 0.05, 0.05, 0.06, 0.06, 0.07, 0.07, 0.08,
        0.08, 0.09, 0.09, 0.10, 0.10, 0.10, 0.11, 0.11)
    long_horizon_reconstruction_window_floor_turns: int = (
        W78_DEFAULT_V23_LONG_HORIZON_RECONSTRUCTION_WINDOW_FLOOR)

    @classmethod
    def default(
            cls, *, seed: int = W78_DEFAULT_V23_SEED,
    ) -> "TinyV23SubstrateConfig":
        v22 = TinyV22SubstrateConfig.default(seed=int(seed))
        return cls(v22=v22)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W78_TINY_SUBSTRATE_V23_SCHEMA_VERSION,
            "v22_cid": str(self.v22.cid()),
            "max_n_roles": int(self.max_n_roles),
            "long_horizon_reconstruction_pressure_boost": float(
                round(
                    self
                    .long_horizon_reconstruction_pressure_boost,
                    12)),
            "long_horizon_reconstruction_repair_boost": float(
                round(
                    self.long_horizon_reconstruction_repair_boost,
                    12)),
            "expose_long_horizon_reconstruction_trajectory_cid":
                bool(
                    self
                    .expose_long_horizon_reconstruction_trajectory_cid),
            "expose_long_horizon_reconstruction_length_per_layer":
                bool(
                    self
                    .expose_long_horizon_reconstruction_length_per_layer),
            "expose_long_horizon_reconstruction_pressure_gate":
                bool(
                    self
                    .expose_long_horizon_reconstruction_pressure_gate),
            "gate_weights_v23": [
                float(round(float(x), 12))
                for x in self.gate_weights_v23],
            "long_horizon_reconstruction_window_floor_turns": int(
                self
                .long_horizon_reconstruction_window_floor_turns),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v23_substrate_config",
            "config": self.to_dict()})


@dataclasses.dataclass
class TinyV23SubstrateParams:
    config: TinyV23SubstrateConfig
    v22_params: TinyV22SubstrateParams

    @classmethod
    def init(
            cls, config: TinyV23SubstrateConfig | None = None,
    ) -> "TinyV23SubstrateParams":
        if config is None:
            config = TinyV23SubstrateConfig.default()
        v22 = TinyV22SubstrateParams.init(config.v22)
        return cls(config=config, v22_params=v22)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v23_substrate_params",
            "config_cid": self.config.cid(),
            "v22_params_cid": self.v22_params.cid(),
        })


@dataclasses.dataclass
class TinyV23KVCache:
    v22_cache: TinyV22KVCache
    long_horizon_reconstruction_trajectory_cid: str = ""
    long_horizon_reconstruction_length_per_layer: (
        "_np.ndarray | None") = None
    long_horizon_reconstruction_windows: list[dict[str, Any]] = (
        dataclasses.field(default_factory=list))
    write_log_v23: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(
            cls, n_layers: int, *, n_heads: int, max_len: int,
    ) -> "TinyV23KVCache":
        v22 = TinyV22KVCache.empty(
            int(n_layers), n_heads=int(n_heads),
            max_len=int(max_len))
        return cls(
            v22_cache=v22,
            long_horizon_reconstruction_trajectory_cid="",
            long_horizon_reconstruction_length_per_layer=(
                _np.zeros((int(n_layers),), dtype=_np.int64)),
            long_horizon_reconstruction_windows=[],
            write_log_v23=[])

    def n_tokens(self) -> int:
        return int(self.v22_cache.n_tokens())

    def n_layers(self) -> int:
        return int(self.v22_cache.n_layers())

    def clone(self) -> "TinyV23KVCache":
        return TinyV23KVCache(
            v22_cache=self.v22_cache.clone(),
            long_horizon_reconstruction_trajectory_cid=str(
                self.long_horizon_reconstruction_trajectory_cid),
            long_horizon_reconstruction_length_per_layer=(
                None
                if (self
                    .long_horizon_reconstruction_length_per_layer
                    is None)
                else self
                .long_horizon_reconstruction_length_per_layer
                .copy()),
            long_horizon_reconstruction_windows=list(
                self.long_horizon_reconstruction_windows),
            write_log_v23=list(self.write_log_v23),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v23_kv_cache",
            "v22_cache_cid": self.v22_cache.cid(),
            "long_horizon_reconstruction_trajectory_cid": str(
                self.long_horizon_reconstruction_trajectory_cid),
            "long_horizon_reconstruction_length_per_layer_cid": (
                "none"
                if (self
                    .long_horizon_reconstruction_length_per_layer
                    is None)
                else _ndarray_cid(
                    self
                    .long_horizon_reconstruction_length_per_layer)),
            "long_horizon_reconstruction_windows": list(
                self.long_horizon_reconstruction_windows),
            "write_log_v23": list(self.write_log_v23),
        })


@dataclasses.dataclass
class TinyV23ForwardTrace:
    v22_trace: TinyV22ForwardTrace
    long_horizon_reconstruction_trajectory_cid: str
    long_horizon_reconstruction_length_per_layer: "_np.ndarray"
    long_horizon_reconstruction_pressure_gate_per_layer: (
        "_np.ndarray")
    v23_gate_score_per_layer: "_np.ndarray"
    config_cid: str
    params_cid: str

    @property
    def logits(self) -> "_np.ndarray":
        return self.v22_trace.logits

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v23_forward_trace",
            "v22_trace_cid": self.v22_trace.cid(),
            "long_horizon_reconstruction_trajectory_cid": str(
                self.long_horizon_reconstruction_trajectory_cid),
            "long_horizon_reconstruction_length_per_layer_cid":
                _ndarray_cid(
                    self
                    .long_horizon_reconstruction_length_per_layer),
            "long_horizon_reconstruction_pressure_gate_per_layer_cid":
                _ndarray_cid(
                    self
                    .long_horizon_reconstruction_pressure_gate_per_layer),
            "v23_gate_score_per_layer_cid": _ndarray_cid(
                self.v23_gate_score_per_layer),
        })


def _compute_long_horizon_reconstruction_trajectory_cid(
        cache: TinyV23KVCache) -> str:
    v22 = cache.v22_cache
    return _sha256_hex({
        "kind": "tiny_v23_long_horizon_reconstruction_trajectory",
        "v22_post_restart_replacement_trajectory_cid": str(
            v22
            .replacement_after_restart_after_compound_chain_trajectory_cid),
        "v23_long_horizon_reconstruction_windows": list(
            cache.long_horizon_reconstruction_windows),
    })


def _compute_long_horizon_reconstruction_length_per_layer(
        cache: TinyV23KVCache, n_layers: int,
        long_horizon_reconstruction_window_floor_turns: int = (
            W78_DEFAULT_V23_LONG_HORIZON_RECONSTRUCTION_WINDOW_FLOOR),
) -> "_np.ndarray":
    L = int(n_layers)
    out = _np.zeros((L,), dtype=_np.int64)
    v22 = cache.v22_cache
    base = (
        v22
        .replacement_after_restart_after_compound_chain_length_per_layer
        if v22
        .replacement_after_restart_after_compound_chain_length_per_layer
        is not None
        else _np.zeros((L,), dtype=_np.int64))
    n_pcr = int(
        1 if str(
            v22
            .replacement_after_restart_after_compound_chain_trajectory_cid
        ).strip() else 0)
    n_lhr = int(len(cache.long_horizon_reconstruction_windows))
    max_blackout = 0
    for d in cache.long_horizon_reconstruction_windows:
        try:
            v = int(d.get(
                "long_horizon_blackout_window_turns", 0))
        except Exception:
            v = 0
        if v > max_blackout:
            max_blackout = v
    lhr_active = bool(
        n_pcr > 0 and n_lhr > 0
        and int(max_blackout)
        > int(long_horizon_reconstruction_window_floor_turns))
    for li in range(L):
        b = (
            int(base[li]) if li < int(base.shape[0]) else 0)
        if lhr_active and (li % 13 == 0):
            out[li] = int(W78_REPAIR_LONG_DELAY_RECONSTRUCTION)
        else:
            out[li] = int(b)
    return out


def _compute_long_horizon_reconstruction_pressure_gate_per_layer(
        *, visible_token_budget: float,
        baseline_token_cost: float,
        v22_gate_mean: float,
        post_restart_replacement_pressure_gate_mean: float,
        n_long_horizon_reconstruction_windows: int,
        max_long_horizon_blackout_window_turns: int,
        weights: Sequence[float],
        n_layers: int,
        bias: float = W78_DEFAULT_V23_GATE_BIAS,
) -> "_np.ndarray":
    L = int(n_layers)
    safe_cost = float(max(1.0, baseline_token_cost))
    budget_ratio = float(visible_token_budget) / safe_cost
    rmax = float(max(1, W78_DEFAULT_V23_MAX_ROLES))
    window_ratio = (
        float(max_long_horizon_blackout_window_turns)
        / float(max(1, max(
            10,
            int(max_long_horizon_blackout_window_turns) + 16))))
    n_lhr_ratio = float(
        n_long_horizon_reconstruction_windows) / rmax
    feats = _np.array([
        float(v22_gate_mean),
        float(post_restart_replacement_pressure_gate_mean),
        float(budget_ratio) / float(
            max(1.0, budget_ratio + 1.0)),
        float(n_lhr_ratio),
        float(window_ratio),
        float(n_lhr_ratio) * float(window_ratio),
        float(window_ratio),
        float(post_restart_replacement_pressure_gate_mean)
        * float(n_lhr_ratio),
        float(window_ratio),
        float(n_lhr_ratio),
        float(v22_gate_mean),
        float(post_restart_replacement_pressure_gate_mean),
        float(budget_ratio) / float(
            max(1.0, budget_ratio + 1.0)),
        float(window_ratio),
        float(n_lhr_ratio),
        float(post_restart_replacement_pressure_gate_mean),
    ], dtype=_np.float64)
    w = _np.array(
        [float(x) for x in weights], dtype=_np.float64)
    score = (
        float(_np.dot(w[:feats.shape[0]], feats)) + float(bias))
    sig = 1.0 / (1.0 + math.exp(-score))
    per_layer = _np.full(
        (L,), float(sig), dtype=_np.float64)
    return _np.round(per_layer, decimals=12)


def _compute_v23_gate_score(
        long_horizon_reconstruction_active: bool,
        long_horizon_reconstruction_pressure_gate_mean: float,
        v22_gate_mean: float,
        weights: Sequence[float],
        n_layers: int,
        bias: float = W78_DEFAULT_V23_GATE_BIAS,
) -> "_np.ndarray":
    feats = _np.array([
        float(v22_gate_mean),
        1.0 if bool(long_horizon_reconstruction_active) else 0.0,
        float(long_horizon_reconstruction_pressure_gate_mean),
        0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5,
        float(long_horizon_reconstruction_pressure_gate_mean),
        1.0 if bool(long_horizon_reconstruction_active) else 0.0,
        float(long_horizon_reconstruction_pressure_gate_mean),
        float(long_horizon_reconstruction_pressure_gate_mean),
        float(long_horizon_reconstruction_pressure_gate_mean),
        float(long_horizon_reconstruction_pressure_gate_mean),
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


def forward_tiny_substrate_v23(
        params: TinyV23SubstrateParams,
        token_ids: Sequence[int],
        *,
        v23_kv_cache: TinyV23KVCache | None = None,
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
        post_restart_replacement_pressure: float = 0.0,
        long_horizon_reconstruction_pressure: float = 0.0,
) -> tuple[TinyV23ForwardTrace, TinyV23KVCache]:
    """V23 forward = V22 forward + long-horizon-reconstruction
    trajectory CID + long-horizon-reconstruction length per layer +
    long-horizon-reconstruction-pressure gate per layer + V23
    composite gate.
    """
    cfg = params.config
    base_v22 = (
        v23_kv_cache.v22_cache if v23_kv_cache is not None
        else None)
    v22_trace, new_v22 = forward_tiny_substrate_v22(
        params.v22_params, list(token_ids),
        v22_kv_cache=base_v22,
        attention_bias_per_layer=attention_bias_per_layer,
        visible_token_budget=float(visible_token_budget),
        baseline_token_cost=float(baseline_token_cost),
        restart_pressure=float(restart_pressure),
        rejoin_pressure=float(rejoin_pressure),
        replacement_pressure=float(replacement_pressure),
        contradiction_pressure=float(contradiction_pressure),
        delayed_repair_pressure=float(delayed_repair_pressure),
        compound_pressure=float(compound_pressure),
        compound_chain_pressure=float(compound_chain_pressure),
        compound_chain_then_restart_pressure=float(
            compound_chain_then_restart_pressure),
        post_restart_replacement_pressure=float(
            post_restart_replacement_pressure))
    n_layers = int(
        cfg.v22.v21.v20.v19.v18.v17.v16.v15.v14.v13.v12.v11.v10
        .v9.n_layers)
    if v23_kv_cache is None:
        v23_new = TinyV23KVCache.empty(
            int(n_layers),
            n_heads=int(
                cfg.v22.v21.v20.v19.v18.v17.v16.v15.v14.v13.v12
                .v11.v10.v9.n_heads),
            max_len=int(
                cfg.v22.v21.v20.v19.v18.v17.v16.v15.v14.v13.v12
                .v11.v10.v9.max_len))
    else:
        v23_new = v23_kv_cache.clone()
    v23_new.v22_cache = new_v22
    lhr_cid = _compute_long_horizon_reconstruction_trajectory_cid(
        v23_new)
    v23_new.long_horizon_reconstruction_trajectory_cid = str(
        lhr_cid)
    lhr_per_layer = (
        _compute_long_horizon_reconstruction_length_per_layer(
            v23_new, n_layers=n_layers,
            long_horizon_reconstruction_window_floor_turns=int(
                cfg
                .long_horizon_reconstruction_window_floor_turns)))
    v23_new.long_horizon_reconstruction_length_per_layer = (
        lhr_per_layer)
    max_lhr_blackout = 0
    for d in v23_new.long_horizon_reconstruction_windows:
        try:
            v = int(d.get(
                "long_horizon_blackout_window_turns", 0))
        except Exception:
            v = 0
        if v > max_lhr_blackout:
            max_lhr_blackout = v
    v22_gate_mean = float(
        v22_trace.v22_gate_score_per_layer.mean()
        if v22_trace.v22_gate_score_per_layer.size else 0.0)
    pcr_gate_mean = float(
        v22_trace
        .replacement_after_restart_after_compound_chain_pressure_gate_per_layer
        .mean()
        if v22_trace
        .replacement_after_restart_after_compound_chain_pressure_gate_per_layer
        .size
        else 0.0)
    effective_lhr_window = int(
        max_lhr_blackout + int(round(float(max(0.0, min(
            1.0,
            float(long_horizon_reconstruction_pressure))))
            * 10.0)))
    lhr_gate = (
        _compute_long_horizon_reconstruction_pressure_gate_per_layer(
            visible_token_budget=float(visible_token_budget),
            baseline_token_cost=float(baseline_token_cost),
            v22_gate_mean=float(v22_gate_mean),
            post_restart_replacement_pressure_gate_mean=float(
                pcr_gate_mean),
            n_long_horizon_reconstruction_windows=int(len(
                v23_new.long_horizon_reconstruction_windows)),
            max_long_horizon_blackout_window_turns=int(
                effective_lhr_window),
            weights=cfg.gate_weights_v23,
            n_layers=int(n_layers)))
    lhr_active = bool(
        int(_np.count_nonzero(
            lhr_per_layer == W78_REPAIR_LONG_DELAY_RECONSTRUCTION))
        > 0
        or float(long_horizon_reconstruction_pressure) > 0.0)
    v23_gate = _compute_v23_gate_score(
        long_horizon_reconstruction_active=bool(lhr_active),
        long_horizon_reconstruction_pressure_gate_mean=float(
            lhr_gate.mean()),
        v22_gate_mean=float(v22_gate_mean),
        weights=cfg.gate_weights_v23,
        n_layers=int(n_layers))
    v23_new.write_log_v23.append({
        "schema": W78_TINY_SUBSTRATE_V23_SCHEMA_VERSION,
        "kind": "forward_v23",
        "n_new_tokens": int(len(list(token_ids))),
        "long_horizon_reconstruction_trajectory_cid": str(
            lhr_cid),
        "long_horizon_reconstruction_length_per_layer": [
            int(x) for x in lhr_per_layer.tolist()],
        "long_horizon_reconstruction_pressure_gate_mean": float(
            lhr_gate.mean()),
        "v23_gate_score_mean": float(v23_gate.mean()),
        "visible_token_budget": float(visible_token_budget),
        "long_horizon_reconstruction_pressure": float(
            long_horizon_reconstruction_pressure),
        "n_long_horizon_reconstruction_windows": int(len(
            v23_new.long_horizon_reconstruction_windows)),
        "max_long_horizon_blackout_window_turns": int(
            max_lhr_blackout),
    })
    trace = TinyV23ForwardTrace(
        v22_trace=v22_trace,
        long_horizon_reconstruction_trajectory_cid=str(lhr_cid),
        long_horizon_reconstruction_length_per_layer=lhr_per_layer,
        long_horizon_reconstruction_pressure_gate_per_layer=(
            lhr_gate),
        v23_gate_score_per_layer=v23_gate,
        config_cid=str(cfg.cid()),
        params_cid=str(params.cid()),
    )
    return trace, v23_new


def record_long_horizon_reconstruction_window_v23(
        cache: TinyV23KVCache, *,
        compound_chain_failure_turn: int,
        reconstruction_request_turn: int,
        long_horizon_blackout_window_turns: int,
        role: str = "team", branch_id: str = "main",
) -> None:
    cache.long_horizon_reconstruction_windows.append({
        "schema": W78_TINY_SUBSTRATE_V23_SCHEMA_VERSION,
        "kind": "long_horizon_reconstruction_window_v23",
        "compound_chain_failure_turn": int(
            compound_chain_failure_turn),
        "reconstruction_request_turn": int(
            reconstruction_request_turn),
        "long_horizon_blackout_window_turns": int(
            long_horizon_blackout_window_turns),
        "role": str(role),
        "branch_id": str(branch_id),
    })
    cache.write_log_v23.append({
        "schema": W78_TINY_SUBSTRATE_V23_SCHEMA_VERSION,
        "kind": "long_horizon_reconstruction_window_recorded",
        "long_horizon_blackout_window_turns": int(
            long_horizon_blackout_window_turns),
        "role": str(role),
    })


def substrate_long_horizon_reconstruction_dominance_flops_v23(
        *, n_tokens: int, n_repairs: int = 14,
        recompute_flops_per_token: int = 1000,
        long_horizon_reconstruction_dominance_flops_per_token: (
            int) = 30,
) -> dict[str, Any]:
    n = int(max(0, n_tokens))
    nr = int(max(1, n_repairs))
    lhr_flops = (
        int(long_horizon_reconstruction_dominance_flops_per_token)
        * n * nr)
    rc_flops = int(recompute_flops_per_token) * n * nr
    saving = int(rc_flops - lhr_flops)
    ratio = (
        float(saving) / float(rc_flops)
        if rc_flops > 0 else 0.0)
    return {
        "n_tokens": int(n),
        "n_repairs": int(nr),
        "long_horizon_reconstruction_dominance_flops": int(
            lhr_flops),
        "recompute_flops": int(rc_flops),
        "saving_flops": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


def substrate_long_horizon_reconstruction_pressure_throttle_v23(
        *, visible_token_budget: int = 64,
        baseline_token_cost: int = 512,
        long_horizon_blackout_window_turns: int = 200,
) -> dict[str, Any]:
    bt = int(max(0, visible_token_budget))
    bc = int(max(1, baseline_token_cost))
    base_saving = int(max(0, bc - bt))
    # The longer the blackout, the more the substrate saves
    # relative to a recompute that has nothing visible to read.
    window_lift = min(3.2, 1.08 + 0.010 * float(
        max(0, long_horizon_blackout_window_turns)))
    saving_tokens = int(round(base_saving * float(window_lift)))
    saving_tokens = int(min(saving_tokens, bc))
    ratio = (
        float(saving_tokens) / float(bc)
        if bc > 0 else 0.0)
    return {
        "visible_token_budget": int(bt),
        "baseline_token_cost": int(bc),
        "long_horizon_blackout_window_turns": int(
            long_horizon_blackout_window_turns),
        "window_lift": float(round(window_lift, 12)),
        "saving_tokens": int(saving_tokens),
        "saving_ratio": float(round(ratio, 12)),
        "long_horizon_reconstruction_pressure_active": bool(
            saving_tokens > 0),
    }


def build_default_tiny_substrate_v23(
        *, seed: int = W78_DEFAULT_V23_SEED,
) -> TinyV23SubstrateParams:
    cfg = TinyV23SubstrateConfig.default(seed=int(seed))
    return TinyV23SubstrateParams.init(cfg)


@dataclasses.dataclass(frozen=True)
class TinyV23ForwardWitness:
    schema: str
    forward_trace_cid: str
    cache_cid: str
    long_horizon_reconstruction_trajectory_cid: str
    long_horizon_reconstruction_l1: int
    long_horizon_reconstruction_pressure_gate_mean: float
    v23_gate_score_mean: float
    n_layers: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "forward_trace_cid": str(self.forward_trace_cid),
            "cache_cid": str(self.cache_cid),
            "long_horizon_reconstruction_trajectory_cid": str(
                self.long_horizon_reconstruction_trajectory_cid),
            "long_horizon_reconstruction_l1": int(
                self.long_horizon_reconstruction_l1),
            "long_horizon_reconstruction_pressure_gate_mean":
                float(round(
                    self
                    .long_horizon_reconstruction_pressure_gate_mean,
                    12)),
            "v23_gate_score_mean": float(round(
                self.v23_gate_score_mean, 12)),
            "n_layers": int(self.n_layers),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v23_forward_witness",
            "witness": self.to_dict()})


def emit_tiny_substrate_v23_forward_witness(
        trace: TinyV23ForwardTrace,
        cache: TinyV23KVCache,
) -> TinyV23ForwardWitness:
    lhr = (
        cache.long_horizon_reconstruction_length_per_layer
        if cache.long_horizon_reconstruction_length_per_layer
        is not None
        else _np.zeros((0,), dtype=_np.int64))
    lhr_l1 = int(_np.count_nonzero(
        lhr == W78_REPAIR_LONG_DELAY_RECONSTRUCTION))
    lhrg_mean = float(
        trace.long_horizon_reconstruction_pressure_gate_per_layer
        .mean()
        if trace
        .long_horizon_reconstruction_pressure_gate_per_layer
        .size
        else 0.0)
    v23_mean = float(
        trace.v23_gate_score_per_layer.mean()
        if trace.v23_gate_score_per_layer.size else 0.0)
    return TinyV23ForwardWitness(
        schema=W78_TINY_SUBSTRATE_V23_SCHEMA_VERSION,
        forward_trace_cid=str(trace.cid()),
        cache_cid=str(cache.cid()),
        long_horizon_reconstruction_trajectory_cid=str(
            trace.long_horizon_reconstruction_trajectory_cid),
        long_horizon_reconstruction_l1=int(lhr_l1),
        long_horizon_reconstruction_pressure_gate_mean=float(
            lhrg_mean),
        v23_gate_score_mean=float(v23_mean),
        n_layers=int(trace.v23_gate_score_per_layer.shape[0]),
    )


__all__ = [
    "W78_TINY_SUBSTRATE_V23_SCHEMA_VERSION",
    "W78_TINY_V23_VOCAB_SIZE",
    "W78_DEFAULT_V23_MAX_ROLES",
    "W78_DEFAULT_V23_LONG_HORIZON_RECONSTRUCTION_PRESSURE_BOOST",
    "W78_DEFAULT_V23_LONG_HORIZON_RECONSTRUCTION_REPAIR_BOOST",
    "W78_DEFAULT_V23_LONG_HORIZON_RECONSTRUCTION_WINDOW_FLOOR",
    "W78_REPAIR_LONG_DELAY_RECONSTRUCTION",
    "W78_REPAIR_LABELS_V23",
    "TinyV23SubstrateConfig",
    "TinyV23SubstrateParams",
    "TinyV23KVCache",
    "TinyV23ForwardTrace",
    "tokenize_bytes_v23",
    "forward_tiny_substrate_v23",
    "record_long_horizon_reconstruction_window_v23",
    "substrate_long_horizon_reconstruction_dominance_flops_v23",
    "substrate_long_horizon_reconstruction_pressure_throttle_v23",
    "build_default_tiny_substrate_v23",
    "TinyV23ForwardWitness",
    "emit_tiny_substrate_v23_forward_witness",
]
