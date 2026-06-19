"""W77 M1 — Tiny Transformer Runtime V22.

Strictly extends W76's ``coordpy.tiny_substrate_v21``. V22 keeps
every V21 invariant byte-for-byte under trivial construction and
adds **three** new substrate-load-bearing axes that the new W77
multi-agent coordinator V13, team-consensus controller V12, V22
bridges/controllers, and the new replacement-after-restart-after-
compound-chain-repair-aware hosted ↔ real handoff coordinator V9
exploit:

* **Inherits V21's 23 layers**. V22 adds a fourth axis layer in
  the substrate cache (post-restart-replacement axis); the
  physical transformer depth carries forward from V21 byte-for-
  byte. Same GQA (8 query / 4 KV).
* **Per-turn replacement-after-restart-after-compound-chain
  trajectory CID** — ``TinyV22KVCache
  .replacement_after_restart_after_compound_chain_trajectory_cid``
  is a deterministic content-addressed SHA-256 over the V21
  compound-chain-then-restart-trajectory CID PLUS the recorded
  W77 post-chain-then-restart replacement event chain.
* **Per-layer replacement-after-restart-after-compound-chain
  length label** — shape ``(L,)`` int64 in [0..13]. V21's [0..12]
  are extended by 13 = ``replacement_after_restart_after_compound_
  chain_repair`` (any layer on which a replacement event was
  observed AFTER a post-compound-chain-restart window).
* **Per-layer replacement-after-restart-after-compound-chain
  pressure gate** — substrate-side throttle in [0, 1] that
  modulates substrate work as a function of the visible-token
  budget AND the joint chain+restart+post-restart-replacement
  depth across the thirteen primitive pressures.

V22 strictly extends V21: with no W77 events recorded the new
trajectory CID still differs from V21's by construction (it
includes the new chain length) but the new gate stays at the V21
gate mean — V22 reduces to V21 plus a deterministic V22-axis tag.

Honest scope (W77)
------------------

* Still NOT a frontier model. Default config:
  ``23 layers (inherited from V21) / 8q / 4kv / d_model=64 /
  ff_hidden=192 / byte-vocab / max_len=128 / untrained``.
  ``W77-L-NUMPY-CPU-V22-SUBSTRATE-CAP``.
* V22 does NOT bridge to third-party hosted models.
  ``W77-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.tiny_substrate_v22 requires numpy") from exc

from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex
from .tiny_substrate_v21 import (
    TinyV21ForwardTrace, TinyV21KVCache, TinyV21SubstrateConfig,
    TinyV21SubstrateParams,
    W76_DEFAULT_V21_GATE_BIAS, W76_DEFAULT_V21_MAX_ROLES,
    W76_REPAIR_LABELS_V21,
    forward_tiny_substrate_v21,
    tokenize_bytes_v21 as _tokenize_bytes_v21,
)


W77_TINY_SUBSTRATE_V22_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v22.v1")

W77_TINY_V22_VOCAB_SIZE: int = 259
W77_DEFAULT_V22_D_MODEL: int = 64
W77_DEFAULT_V22_N_HEADS: int = 8
W77_DEFAULT_V22_N_KV_HEADS: int = 4
W77_DEFAULT_V22_N_LAYERS: int = 23
W77_DEFAULT_V22_FF_HIDDEN: int = 192
W77_DEFAULT_V22_MAX_LEN: int = 128
W77_DEFAULT_V22_INIT_SCALE: float = 0.04
W77_DEFAULT_V22_SEED: int = 77123456
W77_DEFAULT_V22_MAX_ROLES: int = W76_DEFAULT_V21_MAX_ROLES
W77_DEFAULT_V22_POST_RESTART_REPLACEMENT_PRESSURE_BOOST: float = (
    0.82)
W77_DEFAULT_V22_POST_RESTART_REPLACEMENT_REPAIR_BOOST: float = 0.82
W77_DEFAULT_V22_GATE_BIAS: float = W76_DEFAULT_V21_GATE_BIAS
W77_DEFAULT_V22_POST_RESTART_REPLACEMENT_WINDOW_FLOOR: int = 1

# V22 extends W76_REPAIR_LABELS_V21 with a fourteenth primitive.
W77_REPAIR_REPLACEMENT_AFTER_RESTART_AFTER_COMPOUND_CHAIN: int = 13
W77_REPAIR_LABELS_V22: tuple[str, ...] = (
    *W76_REPAIR_LABELS_V21,
    "replacement_after_restart_after_compound_chain_repair",
)


def tokenize_bytes_v22(
        text: str, *, max_len: int = 16) -> list[int]:
    """Byte-tokenisation passthrough to V21."""
    return _tokenize_bytes_v21(str(text), max_len=int(max_len))


@dataclasses.dataclass
class TinyV22SubstrateConfig:
    """V22 config wraps a V21 config + three new V22 axes."""
    v21: TinyV21SubstrateConfig
    max_n_roles: int = W77_DEFAULT_V22_MAX_ROLES
    post_restart_replacement_pressure_boost: float = (
        W77_DEFAULT_V22_POST_RESTART_REPLACEMENT_PRESSURE_BOOST)
    post_restart_replacement_repair_boost: float = (
        W77_DEFAULT_V22_POST_RESTART_REPLACEMENT_REPAIR_BOOST)
    expose_post_restart_replacement_trajectory_cid: bool = True
    expose_post_restart_replacement_length_per_layer: bool = True
    expose_post_restart_replacement_pressure_gate: bool = True
    gate_weights_v22: tuple[float, ...] = (
        0.05, 0.05, 0.05, 0.06, 0.06, 0.06, 0.07, 0.07,
        0.07, 0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.10)
    post_restart_replacement_window_floor_turns: int = (
        W77_DEFAULT_V22_POST_RESTART_REPLACEMENT_WINDOW_FLOOR)

    @classmethod
    def default(
            cls, *, seed: int = W77_DEFAULT_V22_SEED,
    ) -> "TinyV22SubstrateConfig":
        v21 = TinyV21SubstrateConfig.default(seed=int(seed))
        return cls(v21=v21)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W77_TINY_SUBSTRATE_V22_SCHEMA_VERSION,
            "v21_cid": str(self.v21.cid()),
            "max_n_roles": int(self.max_n_roles),
            "post_restart_replacement_pressure_boost": float(round(
                self.post_restart_replacement_pressure_boost, 12)),
            "post_restart_replacement_repair_boost": float(round(
                self.post_restart_replacement_repair_boost, 12)),
            "expose_post_restart_replacement_trajectory_cid": bool(
                self.expose_post_restart_replacement_trajectory_cid),
            "expose_post_restart_replacement_length_per_layer": (
                bool(
                    self
                    .expose_post_restart_replacement_length_per_layer)),
            "expose_post_restart_replacement_pressure_gate": bool(
                self.expose_post_restart_replacement_pressure_gate),
            "gate_weights_v22": [
                float(round(float(x), 12))
                for x in self.gate_weights_v22],
            "post_restart_replacement_window_floor_turns": int(
                self.post_restart_replacement_window_floor_turns),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v22_substrate_config",
            "config": self.to_dict()})


@dataclasses.dataclass
class TinyV22SubstrateParams:
    config: TinyV22SubstrateConfig
    v21_params: TinyV21SubstrateParams

    @classmethod
    def init(
            cls, config: TinyV22SubstrateConfig | None = None,
    ) -> "TinyV22SubstrateParams":
        if config is None:
            config = TinyV22SubstrateConfig.default()
        v21 = TinyV21SubstrateParams.init(config.v21)
        return cls(config=config, v21_params=v21)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v22_substrate_params",
            "config_cid": self.config.cid(),
            "v21_params_cid": self.v21_params.cid(),
        })


@dataclasses.dataclass
class TinyV22KVCache:
    """V22 cache. Wraps a V21 cache + three new V22 axes."""
    v21_cache: TinyV21KVCache
    replacement_after_restart_after_compound_chain_trajectory_cid: (
        str) = ""
    replacement_after_restart_after_compound_chain_length_per_layer: (
        "_np.ndarray | None") = None
    post_restart_replacement_windows: list[dict[str, Any]] = (
        dataclasses.field(default_factory=list))
    write_log_v22: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(
            cls, n_layers: int, *, n_heads: int, max_len: int,
    ) -> "TinyV22KVCache":
        v21 = TinyV21KVCache.empty(
            int(n_layers), n_heads=int(n_heads),
            max_len=int(max_len))
        return cls(
            v21_cache=v21,
            replacement_after_restart_after_compound_chain_trajectory_cid=(
                ""),
            replacement_after_restart_after_compound_chain_length_per_layer=(
                _np.zeros((int(n_layers),), dtype=_np.int64)),
            post_restart_replacement_windows=[],
            write_log_v22=[])

    def n_tokens(self) -> int:
        return int(self.v21_cache.n_tokens())

    def n_layers(self) -> int:
        return int(self.v21_cache.n_layers())

    def clone(self) -> "TinyV22KVCache":
        return TinyV22KVCache(
            v21_cache=self.v21_cache.clone(),
            replacement_after_restart_after_compound_chain_trajectory_cid=(
                str(
                    self
                    .replacement_after_restart_after_compound_chain_trajectory_cid)),
            replacement_after_restart_after_compound_chain_length_per_layer=(
                None if (
                    self
                    .replacement_after_restart_after_compound_chain_length_per_layer
                    is None)
                else self
                .replacement_after_restart_after_compound_chain_length_per_layer
                .copy()),
            post_restart_replacement_windows=list(
                self.post_restart_replacement_windows),
            write_log_v22=list(self.write_log_v22),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v22_kv_cache",
            "v21_cache_cid": self.v21_cache.cid(),
            "replacement_after_restart_after_compound_chain_trajectory_cid":
                str(
                    self
                    .replacement_after_restart_after_compound_chain_trajectory_cid),
            "replacement_after_restart_after_compound_chain_length_per_layer_cid":
                ("none"
                 if (self
                     .replacement_after_restart_after_compound_chain_length_per_layer
                     is None)
                 else _ndarray_cid(
                     self
                     .replacement_after_restart_after_compound_chain_length_per_layer)),
            "post_restart_replacement_windows": list(
                self.post_restart_replacement_windows),
            "write_log_v22": list(self.write_log_v22),
        })


@dataclasses.dataclass
class TinyV22ForwardTrace:
    v21_trace: TinyV21ForwardTrace
    replacement_after_restart_after_compound_chain_trajectory_cid: (
        str)
    replacement_after_restart_after_compound_chain_length_per_layer: (
        "_np.ndarray")
    replacement_after_restart_after_compound_chain_pressure_gate_per_layer: (
        "_np.ndarray")
    v22_gate_score_per_layer: "_np.ndarray"
    config_cid: str
    params_cid: str

    @property
    def logits(self) -> "_np.ndarray":
        return self.v21_trace.logits

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v22_forward_trace",
            "v21_trace_cid": self.v21_trace.cid(),
            "replacement_after_restart_after_compound_chain_trajectory_cid":
                str(
                    self
                    .replacement_after_restart_after_compound_chain_trajectory_cid),
            "replacement_after_restart_after_compound_chain_length_per_layer_cid":
                _ndarray_cid(
                    self
                    .replacement_after_restart_after_compound_chain_length_per_layer),
            "replacement_after_restart_after_compound_chain_pressure_gate_per_layer_cid":
                _ndarray_cid(
                    self
                    .replacement_after_restart_after_compound_chain_pressure_gate_per_layer),
            "v22_gate_score_per_layer_cid": _ndarray_cid(
                self.v22_gate_score_per_layer),
        })


def _compute_post_restart_replacement_trajectory_cid(
        cache: TinyV22KVCache) -> str:
    """Content-addressed CID over V21 chain-then-restart-trajectory
    CID + W77 post-chain-then-restart replacement windows."""
    v21 = cache.v21_cache
    return _sha256_hex({
        "kind":
            "tiny_v22_replacement_after_restart_after_"
            "compound_chain_trajectory",
        "v21_compound_chain_then_restart_trajectory_cid": str(
            v21.compound_chain_then_restart_trajectory_cid),
        "v22_post_restart_replacement_windows": list(
            cache.post_restart_replacement_windows),
    })


def _compute_post_restart_replacement_length_per_layer(
        cache: TinyV22KVCache, n_layers: int,
        post_restart_replacement_window_floor_turns: int = (
            W77_DEFAULT_V22_POST_RESTART_REPLACEMENT_WINDOW_FLOOR),
) -> "_np.ndarray":
    """Per-layer replacement-after-restart-after-compound-chain
    length label.

    Returns shape (L,) dtype int64 in [0..13]. Label 13 fires iff a
    chain-then-restart trajectory CID has been observed AND a
    replacement event followed the post-chain-restart window AND
    that window exceeds
    ``post_restart_replacement_window_floor_turns``.
    """
    L = int(n_layers)
    out = _np.zeros((L,), dtype=_np.int64)
    v21 = cache.v21_cache
    base = (
        v21.compound_chain_then_restart_length_per_layer
        if v21.compound_chain_then_restart_length_per_layer
        is not None
        else _np.zeros((L,), dtype=_np.int64))
    n_chain_then_restart = int(
        1 if str(
            v21.compound_chain_then_restart_trajectory_cid
        ).strip() else 0)
    n_post = int(len(cache.post_restart_replacement_windows))
    max_window = 0
    for d in cache.post_restart_replacement_windows:
        try:
            v = int(d.get(
                "post_restart_replacement_window_turns", 0))
        except Exception:
            v = 0
        if v > max_window:
            max_window = v
    post_active = bool(
        n_chain_then_restart > 0 and n_post > 0
        and int(max_window)
        > int(post_restart_replacement_window_floor_turns))
    for li in range(L):
        b = (
            int(base[li]) if li < int(base.shape[0]) else 0)
        if post_active and (li % 12 == 0):
            out[li] = (
                W77_REPAIR_REPLACEMENT_AFTER_RESTART_AFTER_COMPOUND_CHAIN)
        else:
            out[li] = int(b)
    return out


def _compute_post_restart_replacement_pressure_gate_per_layer(
        *, visible_token_budget: float,
        baseline_token_cost: float,
        v21_gate_mean: float,
        chain_then_restart_pressure_gate_mean: float,
        n_post_restart_replacement_windows: int,
        max_post_restart_replacement_window_turns: int,
        weights: Sequence[float],
        n_layers: int,
        bias: float = W77_DEFAULT_V22_GATE_BIAS,
) -> "_np.ndarray":
    """Per-layer post-restart-replacement-pressure gate."""
    L = int(n_layers)
    safe_cost = float(max(1.0, baseline_token_cost))
    budget_ratio = float(visible_token_budget) / safe_cost
    rmax = float(max(1, W77_DEFAULT_V22_MAX_ROLES))
    window_ratio = (
        float(max_post_restart_replacement_window_turns)
        / float(max(1, max(
            10,
            int(max_post_restart_replacement_window_turns) + 8))))
    n_post_ratio = float(
        n_post_restart_replacement_windows) / rmax
    feats = _np.array([
        float(v21_gate_mean),
        float(chain_then_restart_pressure_gate_mean),
        float(budget_ratio) / float(
            max(1.0, budget_ratio + 1.0)),
        float(n_post_ratio),
        float(window_ratio),
        float(n_post_ratio) * float(window_ratio),
        float(window_ratio),
        float(chain_then_restart_pressure_gate_mean)
        * float(n_post_ratio),
        float(window_ratio),
        float(n_post_ratio),
        float(v21_gate_mean),
        float(chain_then_restart_pressure_gate_mean),
        float(budget_ratio) / float(
            max(1.0, budget_ratio + 1.0)),
        float(window_ratio),
        float(n_post_ratio),
        float(chain_then_restart_pressure_gate_mean),
    ], dtype=_np.float64)
    w = _np.array(
        [float(x) for x in weights], dtype=_np.float64)
    score = (
        float(_np.dot(w[:feats.shape[0]], feats)) + float(bias))
    sig = 1.0 / (1.0 + math.exp(-score))
    per_layer = _np.full(
        (L,), float(sig), dtype=_np.float64)
    return _np.round(per_layer, decimals=12)


def _compute_v22_gate_score(
        post_restart_replacement_active: bool,
        post_restart_replacement_pressure_gate_mean: float,
        v21_gate_mean: float,
        weights: Sequence[float],
        n_layers: int,
        bias: float = W77_DEFAULT_V22_GATE_BIAS,
) -> "_np.ndarray":
    feats = _np.array([
        float(v21_gate_mean),
        1.0 if bool(post_restart_replacement_active) else 0.0,
        float(post_restart_replacement_pressure_gate_mean),
        0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5,
        float(post_restart_replacement_pressure_gate_mean),
        1.0 if bool(post_restart_replacement_active) else 0.0,
        float(post_restart_replacement_pressure_gate_mean),
        float(post_restart_replacement_pressure_gate_mean),
        float(post_restart_replacement_pressure_gate_mean),
        float(post_restart_replacement_pressure_gate_mean),
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


def forward_tiny_substrate_v22(
        params: TinyV22SubstrateParams,
        token_ids: Sequence[int],
        *,
        v22_kv_cache: TinyV22KVCache | None = None,
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
) -> tuple[TinyV22ForwardTrace, TinyV22KVCache]:
    """V22 forward = V21 forward + post-restart-replacement
    trajectory CID + post-restart-replacement length per layer +
    post-restart-replacement-pressure gate per layer + V22
    composite gate.

    The new ``post_restart_replacement_pressure`` knob in [0, 1] is
    a caller-declared signal that the team is absorbing a
    *replacement-after-restart-after-compound-chain-repair* arc;
    the substrate uses it to bias the V22 gate towards substrate
    work.
    """
    cfg = params.config
    base_v21 = (
        v22_kv_cache.v21_cache if v22_kv_cache is not None
        else None)
    v21_trace, new_v21 = forward_tiny_substrate_v21(
        params.v21_params, list(token_ids),
        v21_kv_cache=base_v21,
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
            compound_chain_then_restart_pressure))
    n_layers = int(
        cfg.v21.v20.v19.v18.v17.v16.v15.v14.v13.v12.v11.v10.v9
        .n_layers)
    if v22_kv_cache is None:
        v22_new = TinyV22KVCache.empty(
            int(n_layers),
            n_heads=int(
                cfg.v21.v20.v19.v18.v17.v16.v15.v14.v13.v12.v11
                .v10.v9.n_heads),
            max_len=int(
                cfg.v21.v20.v19.v18.v17.v16.v15.v14.v13.v12.v11
                .v10.v9.max_len))
    else:
        v22_new = v22_kv_cache.clone()
    v22_new.v21_cache = new_v21
    pcr_cid = _compute_post_restart_replacement_trajectory_cid(
        v22_new)
    v22_new.replacement_after_restart_after_compound_chain_trajectory_cid = (
        str(pcr_cid))
    pcr_per_layer = (
        _compute_post_restart_replacement_length_per_layer(
            v22_new, n_layers=n_layers,
            post_restart_replacement_window_floor_turns=int(
                cfg.post_restart_replacement_window_floor_turns)))
    v22_new.replacement_after_restart_after_compound_chain_length_per_layer = (
        pcr_per_layer)
    max_pcr_window = 0
    for d in v22_new.post_restart_replacement_windows:
        try:
            v = int(d.get(
                "post_restart_replacement_window_turns", 0))
        except Exception:
            v = 0
        if v > max_pcr_window:
            max_pcr_window = v
    v21_gate_mean = float(
        v21_trace.v21_gate_score_per_layer.mean()
        if v21_trace.v21_gate_score_per_layer.size else 0.0)
    ctr_gate_mean = float(
        v21_trace
        .compound_chain_then_restart_pressure_gate_per_layer
        .mean()
        if v21_trace
        .compound_chain_then_restart_pressure_gate_per_layer.size
        else 0.0)
    effective_pcr_window = int(
        max_pcr_window + int(round(float(max(0.0, min(
            1.0,
            float(post_restart_replacement_pressure))))
            * 5.0)))
    pcr_gate = (
        _compute_post_restart_replacement_pressure_gate_per_layer(
            visible_token_budget=float(visible_token_budget),
            baseline_token_cost=float(baseline_token_cost),
            v21_gate_mean=float(v21_gate_mean),
            chain_then_restart_pressure_gate_mean=float(
                ctr_gate_mean),
            n_post_restart_replacement_windows=int(len(
                v22_new.post_restart_replacement_windows)),
            max_post_restart_replacement_window_turns=int(
                effective_pcr_window),
            weights=cfg.gate_weights_v22,
            n_layers=int(n_layers)))
    post_restart_replacement_active = bool(
        int(_np.count_nonzero(
            pcr_per_layer
            == W77_REPAIR_REPLACEMENT_AFTER_RESTART_AFTER_COMPOUND_CHAIN)) > 0
        or float(post_restart_replacement_pressure) > 0.0)
    v22_gate = _compute_v22_gate_score(
        post_restart_replacement_active=bool(
            post_restart_replacement_active),
        post_restart_replacement_pressure_gate_mean=float(
            pcr_gate.mean()),
        v21_gate_mean=float(v21_gate_mean),
        weights=cfg.gate_weights_v22,
        n_layers=int(n_layers))
    v22_new.write_log_v22.append({
        "schema": W77_TINY_SUBSTRATE_V22_SCHEMA_VERSION,
        "kind": "forward_v22",
        "n_new_tokens": int(len(list(token_ids))),
        "replacement_after_restart_after_compound_chain_trajectory_cid":
            str(pcr_cid),
        "replacement_after_restart_after_compound_chain_length_per_layer":
            [int(x) for x in pcr_per_layer.tolist()],
        "post_restart_replacement_pressure_gate_mean": float(
            pcr_gate.mean()),
        "v22_gate_score_mean": float(v22_gate.mean()),
        "visible_token_budget": float(visible_token_budget),
        "post_restart_replacement_pressure": float(
            post_restart_replacement_pressure),
        "n_post_restart_replacement_windows": int(len(
            v22_new.post_restart_replacement_windows)),
        "max_post_restart_replacement_window_turns": int(
            max_pcr_window),
    })
    trace = TinyV22ForwardTrace(
        v21_trace=v21_trace,
        replacement_after_restart_after_compound_chain_trajectory_cid=(
            str(pcr_cid)),
        replacement_after_restart_after_compound_chain_length_per_layer=(
            pcr_per_layer),
        replacement_after_restart_after_compound_chain_pressure_gate_per_layer=(
            pcr_gate),
        v22_gate_score_per_layer=v22_gate,
        config_cid=str(cfg.cid()),
        params_cid=str(params.cid()),
    )
    return trace, v22_new


def record_post_restart_replacement_window_v22(
        cache: TinyV22KVCache, *,
        chain_then_restart_repair_turn: int,
        replacement_turn: int,
        post_restart_replacement_window_turns: int,
        role: str = "team", branch_id: str = "main",
) -> None:
    """Record a (chain_then_restart_repair_turn, replacement_turn,
    post-restart-replacement window) tuple."""
    cache.post_restart_replacement_windows.append({
        "schema": W77_TINY_SUBSTRATE_V22_SCHEMA_VERSION,
        "kind": "post_restart_replacement_window_v22",
        "chain_then_restart_repair_turn": int(
            chain_then_restart_repair_turn),
        "replacement_turn": int(replacement_turn),
        "post_restart_replacement_window_turns": int(
            post_restart_replacement_window_turns),
        "role": str(role),
        "branch_id": str(branch_id),
    })
    cache.write_log_v22.append({
        "schema": W77_TINY_SUBSTRATE_V22_SCHEMA_VERSION,
        "kind": "post_restart_replacement_window_recorded",
        "post_restart_replacement_window_turns": int(
            post_restart_replacement_window_turns),
        "role": str(role),
    })


def substrate_post_restart_replacement_repair_dominance_flops_v22(
        *, n_tokens: int, n_repairs: int = 13,
        recompute_flops_per_token: int = 1000,
        post_restart_replacement_dominance_flops_per_token: int = (
            35),
) -> dict[str, Any]:
    """V22 post-restart-replacement-repair-dominance vs full
    recompute across thirteen primitives. By default
    ``n_repairs`` is 13.
    """
    n = int(max(0, n_tokens))
    nr = int(max(1, n_repairs))
    pcr_flops = (
        int(post_restart_replacement_dominance_flops_per_token)
        * n * nr)
    rc_flops = int(recompute_flops_per_token) * n * nr
    saving = int(rc_flops - pcr_flops)
    ratio = (
        float(saving) / float(rc_flops)
        if rc_flops > 0 else 0.0)
    return {
        "n_tokens": int(n),
        "n_repairs": int(nr),
        "post_restart_replacement_dominance_flops": int(pcr_flops),
        "recompute_flops": int(rc_flops),
        "saving_flops": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


def substrate_post_restart_replacement_pressure_throttle_v22(
        *, visible_token_budget: int = 64,
        baseline_token_cost: int = 512,
        post_restart_replacement_window_turns: int = 4,
) -> dict[str, Any]:
    """V22 post-restart-replacement-pressure throttle."""
    bt = int(max(0, visible_token_budget))
    bc = int(max(1, baseline_token_cost))
    base_saving = int(max(0, bc - bt))
    window_lift = min(2.8, 1.06 + 0.28 * float(
        max(0, post_restart_replacement_window_turns)))
    saving_tokens = int(round(base_saving * float(window_lift)))
    saving_tokens = int(min(saving_tokens, bc))
    ratio = (
        float(saving_tokens) / float(bc)
        if bc > 0 else 0.0)
    return {
        "visible_token_budget": int(bt),
        "baseline_token_cost": int(bc),
        "post_restart_replacement_window_turns": int(
            post_restart_replacement_window_turns),
        "window_lift": float(round(window_lift, 12)),
        "saving_tokens": int(saving_tokens),
        "saving_ratio": float(round(ratio, 12)),
        "post_restart_replacement_pressure_active": bool(
            saving_tokens > 0),
    }


def build_default_tiny_substrate_v22(
        *, seed: int = W77_DEFAULT_V22_SEED,
) -> TinyV22SubstrateParams:
    """Build a default V22 substrate."""
    cfg = TinyV22SubstrateConfig.default(seed=int(seed))
    return TinyV22SubstrateParams.init(cfg)


@dataclasses.dataclass(frozen=True)
class TinyV22ForwardWitness:
    schema: str
    forward_trace_cid: str
    cache_cid: str
    replacement_after_restart_after_compound_chain_trajectory_cid: (
        str)
    replacement_after_restart_after_compound_chain_repair_l1: int
    post_restart_replacement_pressure_gate_mean: float
    v22_gate_score_mean: float
    n_layers: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "forward_trace_cid": str(self.forward_trace_cid),
            "cache_cid": str(self.cache_cid),
            "replacement_after_restart_after_compound_chain_trajectory_cid":
                str(
                    self
                    .replacement_after_restart_after_compound_chain_trajectory_cid),
            "replacement_after_restart_after_compound_chain_repair_l1":
                int(
                    self
                    .replacement_after_restart_after_compound_chain_repair_l1),
            "post_restart_replacement_pressure_gate_mean": float(
                round(
                    self
                    .post_restart_replacement_pressure_gate_mean,
                    12)),
            "v22_gate_score_mean": float(round(
                self.v22_gate_score_mean, 12)),
            "n_layers": int(self.n_layers),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v22_forward_witness",
            "witness": self.to_dict()})


def emit_tiny_substrate_v22_forward_witness(
        trace: TinyV22ForwardTrace,
        cache: TinyV22KVCache,
) -> TinyV22ForwardWitness:
    pcr = (
        cache
        .replacement_after_restart_after_compound_chain_length_per_layer
        if cache
        .replacement_after_restart_after_compound_chain_length_per_layer
        is not None
        else _np.zeros((0,), dtype=_np.int64))
    pcr_l1 = int(_np.count_nonzero(
        pcr
        == W77_REPAIR_REPLACEMENT_AFTER_RESTART_AFTER_COMPOUND_CHAIN))
    pcrg_mean = float(
        trace
        .replacement_after_restart_after_compound_chain_pressure_gate_per_layer
        .mean()
        if trace
        .replacement_after_restart_after_compound_chain_pressure_gate_per_layer.size
        else 0.0)
    v22_mean = float(
        trace.v22_gate_score_per_layer.mean()
        if trace.v22_gate_score_per_layer.size else 0.0)
    return TinyV22ForwardWitness(
        schema=W77_TINY_SUBSTRATE_V22_SCHEMA_VERSION,
        forward_trace_cid=str(trace.cid()),
        cache_cid=str(cache.cid()),
        replacement_after_restart_after_compound_chain_trajectory_cid=(
            str(trace
                .replacement_after_restart_after_compound_chain_trajectory_cid)),
        replacement_after_restart_after_compound_chain_repair_l1=(
            int(pcr_l1)),
        post_restart_replacement_pressure_gate_mean=float(
            pcrg_mean),
        v22_gate_score_mean=float(v22_mean),
        n_layers=int(
            trace.v22_gate_score_per_layer.shape[0]),
    )


__all__ = [
    "W77_TINY_SUBSTRATE_V22_SCHEMA_VERSION",
    "W77_TINY_V22_VOCAB_SIZE",
    "W77_DEFAULT_V22_N_LAYERS",
    "W77_DEFAULT_V22_MAX_ROLES",
    "W77_DEFAULT_V22_POST_RESTART_REPLACEMENT_PRESSURE_BOOST",
    "W77_DEFAULT_V22_POST_RESTART_REPLACEMENT_REPAIR_BOOST",
    "W77_DEFAULT_V22_POST_RESTART_REPLACEMENT_WINDOW_FLOOR",
    "W77_REPAIR_REPLACEMENT_AFTER_RESTART_AFTER_COMPOUND_CHAIN",
    "W77_REPAIR_LABELS_V22",
    "TinyV22SubstrateConfig",
    "TinyV22SubstrateParams",
    "TinyV22KVCache",
    "TinyV22ForwardTrace",
    "tokenize_bytes_v22",
    "forward_tiny_substrate_v22",
    "record_post_restart_replacement_window_v22",
    "substrate_post_restart_replacement_repair_dominance_flops_v22",
    "substrate_post_restart_replacement_pressure_throttle_v22",
    "build_default_tiny_substrate_v22",
    "TinyV22ForwardWitness",
    "emit_tiny_substrate_v22_forward_witness",
]
