"""W64 M1 — Tiny Transformer Runtime V9.

Strictly extends W63's ``coordpy.tiny_substrate_v8``. V9 keeps every
V8 invariant (byte-determinism under same params, GQA, RMSNorm /
SwiGLU, content-addressable cache_keys axis, hidden_write_trace
channel, replay_age channel, forward_count integer, cross-layer
coupling, cache-write ledger, logit-lens probe, attention-receive
delta, replay-trust ledger, hidden-vs-KV contention tensor,
hidden-state confidence probe, replay-determinism channel,
prefix-reuse trust ledger, cross-layer-head coupling matrix) and
adds **five** new substrate-load-bearing axes that W64's trainable
controllers and bridges exploit:

* **Default 11 layers** (vs V8's 10). Same GQA (8 query / 4 KV).
* **Per-(layer, head, slot) hidden-wins-primary tensor** —
  ``TinyV9KVCache.hidden_wins_primary`` of shape ``(L, H, T)``
  records a *signed* primary-decision flag in {-1, 0, +1} where
  +1 = hidden wins, -1 = KV wins, 0 = tie / unevaluated. The
  hidden-wins-primary regime is "primary" when the EWMA over slots
  exceeds a configurable threshold.
* **Per-(layer, head, slot) replay-dominance witness channel** —
  ``TinyV9KVCache.replay_dominance_witness`` of shape ``(L, H, T)``
  records the per-slot replay-dominance scalar (max softmax prob
  margin over the second-best decision). The W64 replay
  controller V5 reads this as a regime feature.
* **Per-layer attention-entropy probe** — emit a per-layer Shannon
  entropy of the *attention probability distribution* (vs V8's
  logit-lens entropy on the *output* distribution).
* **Per-(layer, head, slot, slot) cache-similarity matrix** —
  per-layer per-head pairwise cosine of the cache_keys axis. The
  W64 retrieval head reads this for cache-eviction scoring.
* **Per-(layer, head) hidden-state-trust ledger** — EMA over per
  (layer, head) hidden_state_bridge V8 decisions, mirror of V8's
  prefix-reuse trust ledger.

V9 still preserves the V8 multi-segment partial-prefix reuse and
extends the write log with a ``schema=v9`` tag plus the new axis
deltas.

Honest scope (do-not-overstate, W64)
------------------------------------

* Still NOT a frontier model. Default config:
  ``11 layers / 8 query heads / 4 kv heads / d_model=64 /
  ff_hidden=192 / byte-vocab / max_len=128 / untrained``. V9 is
  richer than V8 but still a research substrate in pure NumPy on
  CPU. ``W64-L-NUMPY-CPU-V9-SUBSTRATE-CAP`` documents.
* V9 still does NOT bridge to third-party hosted models.
  ``W64-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries
  forward unchanged.
* The hidden-wins-primary tensor is a *diagnostic decision flag*,
  not a frontier-model ground truth.
* The attention-entropy probe is computed deterministically; it is
  NOT a calibrated information-theoretic measure.
* The cache-similarity matrix is per-(layer, head) over the
  cache_keys axis only.
* The hidden-state-trust ledger updates by an external setter; the
  substrate does NOT itself emit hidden-state decisions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover - numpy required.
    raise ImportError(
        "coordpy.tiny_substrate_v9 requires numpy") from exc

from .tiny_substrate_v3 import (
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
)
from .tiny_substrate_v6 import (
    W61_DEFAULT_V6_D_KEY,
)
from .tiny_substrate_v8 import (
    TinyV8ForwardTrace, TinyV8KVCache,
    TinyV8SubstrateConfig, TinyV8SubstrateParams,
    W63_DEFAULT_V8_D_KEY, W63_DEFAULT_V8_TRUST_EMA,
    W63_DEFAULT_V8_REPLAY_DETERMINISM_TOL,
    forward_tiny_substrate_v8,
)


W64_TINY_SUBSTRATE_V9_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v9.v1")

W64_TINY_V9_VOCAB_SIZE: int = 259
W64_DEFAULT_V9_D_MODEL: int = 64
W64_DEFAULT_V9_N_HEADS: int = 8
W64_DEFAULT_V9_N_KV_HEADS: int = 4
W64_DEFAULT_V9_N_LAYERS: int = 11
W64_DEFAULT_V9_FF_HIDDEN: int = 192
W64_DEFAULT_V9_MAX_LEN: int = 128
W64_DEFAULT_V9_INIT_SCALE: float = 0.04
W64_DEFAULT_V9_SEED: int = 64012345
W64_DEFAULT_V9_ROPE_BASE: float = 10000.0
W64_DEFAULT_V9_D_KEY: int = 8
W64_DEFAULT_V9_TRUST_EMA: float = 0.5
W64_DEFAULT_V9_REPLAY_DETERMINISM_TOL: float = 1e-6
W64_DEFAULT_V9_HIDDEN_WINS_PRIMARY_THRESHOLD: float = 0.20

W64_HIDDEN_DECISION_WIN_FLAG: float = 1.0
W64_HIDDEN_DECISION_LOSE_FLAG: float = -1.0
W64_HIDDEN_DECISION_TIE_FLAG: float = 0.0

_HIDDEN_DECISION_TO_FLAG: dict[str, float] = {
    "hidden_wins": W64_HIDDEN_DECISION_WIN_FLAG,
    "hidden_loses": W64_HIDDEN_DECISION_LOSE_FLAG,
    "kv_wins": W64_HIDDEN_DECISION_LOSE_FLAG,
    "tie": W64_HIDDEN_DECISION_TIE_FLAG,
}


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


@dataclasses.dataclass
class TinyV9SubstrateConfig:
    vocab_size: int = W64_TINY_V9_VOCAB_SIZE
    d_model: int = W64_DEFAULT_V9_D_MODEL
    n_heads: int = W64_DEFAULT_V9_N_HEADS
    n_kv_heads: int = W64_DEFAULT_V9_N_KV_HEADS
    n_layers: int = W64_DEFAULT_V9_N_LAYERS
    ff_hidden: int = W64_DEFAULT_V9_FF_HIDDEN
    max_len: int = W64_DEFAULT_V9_MAX_LEN
    init_scale: float = W64_DEFAULT_V9_INIT_SCALE
    seed: int = W64_DEFAULT_V9_SEED
    rope_base: float = W64_DEFAULT_V9_ROPE_BASE
    use_rope: bool = True
    use_rmsnorm: bool = True
    use_swiglu: bool = True
    d_key: int = W64_DEFAULT_V9_D_KEY
    trust_ema: float = W64_DEFAULT_V9_TRUST_EMA
    replay_determinism_tol: float = (
        W64_DEFAULT_V9_REPLAY_DETERMINISM_TOL)
    hidden_wins_primary_threshold: float = (
        W64_DEFAULT_V9_HIDDEN_WINS_PRIMARY_THRESHOLD)
    expose_hidden_wins_primary: bool = True
    expose_replay_dominance_witness: bool = True
    expose_attention_entropy_probe: bool = True
    expose_cache_similarity_matrix: bool = True
    expose_hidden_state_trust_ledger: bool = True

    def to_v8_config(self) -> TinyV8SubstrateConfig:
        return TinyV8SubstrateConfig(
            vocab_size=int(self.vocab_size),
            d_model=int(self.d_model),
            n_heads=int(self.n_heads),
            n_kv_heads=int(self.n_kv_heads),
            n_layers=int(self.n_layers),
            ff_hidden=int(self.ff_hidden),
            max_len=int(self.max_len),
            init_scale=float(self.init_scale),
            seed=int(self.seed),
            rope_base=float(self.rope_base),
            use_rope=bool(self.use_rope),
            use_rmsnorm=bool(self.use_rmsnorm),
            use_swiglu=bool(self.use_swiglu),
            d_key=int(self.d_key),
            trust_ema=float(self.trust_ema),
            replay_determinism_tol=float(
                self.replay_determinism_tol))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W64_TINY_SUBSTRATE_V9_SCHEMA_VERSION,
            "vocab_size": int(self.vocab_size),
            "d_model": int(self.d_model),
            "n_heads": int(self.n_heads),
            "n_kv_heads": int(self.n_kv_heads),
            "n_layers": int(self.n_layers),
            "ff_hidden": int(self.ff_hidden),
            "max_len": int(self.max_len),
            "init_scale": float(self.init_scale),
            "seed": int(self.seed),
            "rope_base": float(self.rope_base),
            "use_rope": bool(self.use_rope),
            "use_rmsnorm": bool(self.use_rmsnorm),
            "use_swiglu": bool(self.use_swiglu),
            "d_key": int(self.d_key),
            "trust_ema": float(round(self.trust_ema, 12)),
            "replay_determinism_tol": float(round(
                self.replay_determinism_tol, 12)),
            "hidden_wins_primary_threshold": float(round(
                self.hidden_wins_primary_threshold, 12)),
            "expose_hidden_wins_primary": bool(
                self.expose_hidden_wins_primary),
            "expose_replay_dominance_witness": bool(
                self.expose_replay_dominance_witness),
            "expose_attention_entropy_probe": bool(
                self.expose_attention_entropy_probe),
            "expose_cache_similarity_matrix": bool(
                self.expose_cache_similarity_matrix),
            "expose_hidden_state_trust_ledger": bool(
                self.expose_hidden_state_trust_ledger),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v9_substrate_config",
            "config": self.to_dict()})


@dataclasses.dataclass
class TinyV9SubstrateParams:
    config: TinyV9SubstrateConfig
    v8_params: TinyV8SubstrateParams
    # Per-layer affine [slope, bias] for the attention-entropy probe.
    attention_entropy_calibration: "_np.ndarray"   # (L, 2)

    @classmethod
    def init(
            cls, config: TinyV9SubstrateConfig | None = None,
    ) -> "TinyV9SubstrateParams":
        if config is None:
            config = TinyV9SubstrateConfig()
        v8 = TinyV8SubstrateParams.init(config.to_v8_config())
        rng = _np.random.default_rng(
            int(config.seed) ^ 0xCAFEBABE_64)
        cal = rng.standard_normal(
            (int(config.n_layers), 2)) * 0.2
        cal[:, 0] = -0.25
        cal[:, 1] = 0.4
        return cls(
            config=config, v8_params=v8,
            attention_entropy_calibration=cal.astype(_np.float64))

    @property
    def v3_params(self) -> TinyV3SubstrateParams:
        return self.v8_params.v3_params

    @property
    def v7_params(self):
        return self.v8_params.v7_params

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v9_substrate_params",
            "config_cid": self.config.cid(),
            "v8_params_cid": self.v8_params.cid(),
            "attention_entropy_calibration_cid": _ndarray_cid(
                self.attention_entropy_calibration),
        })


@dataclasses.dataclass
class TinyV9KVCache:
    """V9 cache. Wraps a V8 cache + five new substrate-internal
    axes."""
    v8_cache: TinyV8KVCache
    hidden_wins_primary: "_np.ndarray"  # (L, H, T)
    replay_dominance_witness: "_np.ndarray"  # (L, H, T)
    cache_similarity_matrix: "_np.ndarray"  # (L, H, T, T)
    hidden_state_trust_ledger: "_np.ndarray"  # (L, H)
    write_log_v9: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(
            cls, n_layers: int, *, n_heads: int, max_len: int,
            d_key: int = W61_DEFAULT_V6_D_KEY,
    ) -> "TinyV9KVCache":
        v8 = TinyV8KVCache.empty(
            int(n_layers), n_heads=int(n_heads),
            max_len=int(max_len), d_key=int(d_key))
        return cls(
            v8_cache=v8,
            hidden_wins_primary=_np.zeros(
                (int(n_layers), int(n_heads), int(max_len)),
                dtype=_np.float64),
            replay_dominance_witness=_np.zeros(
                (int(n_layers), int(n_heads), int(max_len)),
                dtype=_np.float64),
            cache_similarity_matrix=_np.zeros(
                (int(n_layers), int(n_heads),
                 int(max_len), int(max_len)),
                dtype=_np.float64),
            hidden_state_trust_ledger=_np.zeros(
                (int(n_layers), int(n_heads)),
                dtype=_np.float64),
            write_log_v9=[])

    def n_tokens(self) -> int:
        return int(self.v8_cache.n_tokens())

    def n_layers(self) -> int:
        return int(self.v8_cache.n_layers())

    def clone(self) -> "TinyV9KVCache":
        return TinyV9KVCache(
            v8_cache=self.v8_cache.clone(),
            hidden_wins_primary=(
                self.hidden_wins_primary.copy()),
            replay_dominance_witness=(
                self.replay_dominance_witness.copy()),
            cache_similarity_matrix=(
                self.cache_similarity_matrix.copy()),
            hidden_state_trust_ledger=(
                self.hidden_state_trust_ledger.copy()),
            write_log_v9=list(self.write_log_v9),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v9_kv_cache",
            "v8_cache_cid": self.v8_cache.cid(),
            "hidden_wins_primary_cid": _ndarray_cid(
                self.hidden_wins_primary),
            "replay_dominance_witness_cid": _ndarray_cid(
                self.replay_dominance_witness),
            "cache_similarity_matrix_cid": _ndarray_cid(
                self.cache_similarity_matrix),
            "hidden_state_trust_ledger_cid": _ndarray_cid(
                self.hidden_state_trust_ledger),
            "write_log_v9": list(self.write_log_v9),
        })


@dataclasses.dataclass
class TinyV9ForwardTrace:
    v8_trace: TinyV8ForwardTrace
    attention_entropy_per_layer: "_np.ndarray"  # (L,)
    config_cid: str
    params_cid: str

    @property
    def logits(self) -> "_np.ndarray":
        return self.v8_trace.logits

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v9_forward_trace",
            "v8_trace_cid": self.v8_trace.cid(),
            "attention_entropy_cid": _ndarray_cid(
                self.attention_entropy_per_layer),
        })


def _compute_attention_entropy(
        v8_trace: TinyV8ForwardTrace,
        calibration: "_np.ndarray",
) -> "_np.ndarray":
    """Per-layer attention entropy. Computed from the
    attention-receive distribution per layer; then a calibrated
    affine map; sigmoid to [0, 1]."""
    L = int(calibration.shape[0])
    out = _np.zeros((L,), dtype=_np.float64)
    ar = (v8_trace.v7_trace.v6_trace.v5_trace
          .attention_receive_per_layer)
    if not ar:
        return out
    n = int(min(L, len(ar)))
    for li in range(n):
        a = _np.asarray(ar[li], dtype=_np.float64)
        if a.size == 0:
            out[li] = 0.0
            continue
        # Normalize over keys per (head, query).
        v = a.ravel()
        v = _np.where(v > 0.0, v, 0.0)
        s = float(v.sum())
        if s <= 0.0:
            out[li] = 0.0
            continue
        p = v / s
        ent = float(-(p[p > 0.0]
                       * _np.log(p[p > 0.0])).sum())
        x = (
            float(calibration[li, 0]) * float(ent)
            + float(calibration[li, 1]))
        out[li] = 1.0 / (1.0 + math.exp(-x))
    return _np.round(out, decimals=12)


def _compute_cache_similarity_matrix(
        v8_cache: TinyV8KVCache,
) -> "_np.ndarray":
    """Per-layer pairwise cosine over the cache_keys axis. The
    cache_keys axis is per-(layer)-(token, d_key) — there is no
    head dim in v6's cache_keys, so we tile across heads.
    Output: (L, H, T, T)."""
    keys = v8_cache.v7_cache.v6_cache.cache_keys
    if keys is None or len(keys) == 0:
        return _np.zeros((0, 0, 0, 0), dtype=_np.float64)
    L = int(len(keys))
    arrs = [_np.asarray(k, dtype=_np.float64) for k in keys]
    T_max = int(max(a.shape[0] for a in arrs)
                 if arrs else 0)
    H = int(v8_cache.v7_cache.cache_write_ledger.shape[1])
    out = _np.zeros((L, H, T_max, T_max), dtype=_np.float64)
    for li, mat in enumerate(arrs):
        T = int(mat.shape[0])
        for ti in range(T):
            vi = mat[ti]
            ni = float(_np.linalg.norm(vi))
            for tj in range(T):
                vj = mat[tj]
                nj = float(_np.linalg.norm(vj))
                if ni > 0.0 and nj > 0.0:
                    val = float(
                        float(_np.dot(vi, vj)) / (ni * nj))
                elif ti == tj:
                    val = 1.0
                else:
                    val = 0.0
                # Tile across heads.
                for hi in range(H):
                    out[li, hi, ti, tj] = val
    return _np.round(out, decimals=12)


def forward_tiny_substrate_v9(
        params: TinyV9SubstrateParams,
        token_ids: Sequence[int],
        *,
        v9_kv_cache: TinyV9KVCache | None = None,
        attention_bias_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
) -> tuple[TinyV9ForwardTrace, TinyV9KVCache]:
    cfg = params.config
    base_v8 = (
        v9_kv_cache.v8_cache if v9_kv_cache is not None else None)
    v8_trace, new_v8 = forward_tiny_substrate_v8(
        params.v8_params, list(token_ids),
        v8_kv_cache=base_v8,
        attention_bias_per_layer=attention_bias_per_layer)
    if v9_kv_cache is None:
        v9_new = TinyV9KVCache.empty(
            int(cfg.n_layers), n_heads=int(cfg.n_heads),
            max_len=int(cfg.max_len), d_key=int(cfg.d_key))
    else:
        v9_new = v9_kv_cache.clone()
    v9_new.v8_cache = new_v8
    # Cache-similarity matrix.
    sim = _compute_cache_similarity_matrix(new_v8)
    if sim.size > 0 and sim.shape[2] > int(
            v9_new.cache_similarity_matrix.shape[2]):
        # Reshape to absorb new T dimension.
        new_T = int(sim.shape[2])
        v9_new.cache_similarity_matrix = sim.copy()
    else:
        # Pad sim into existing matrix.
        if sim.size > 0:
            L, H, T, _ = sim.shape
            v9_new.cache_similarity_matrix[
                :L, :H, :T, :T] = sim
    # Attention-entropy probe.
    ent = _compute_attention_entropy(
        v8_trace, params.attention_entropy_calibration)
    # Pad hidden_wins_primary, replay_dominance_witness if T grew.
    new_T = int(new_v8.v7_cache.cache_write_ledger.shape[2])
    cur_T = int(v9_new.hidden_wins_primary.shape[2])
    if new_T > cur_T:
        L = int(v9_new.hidden_wins_primary.shape[0])
        H = int(v9_new.hidden_wins_primary.shape[1])
        pad = _np.zeros(
            (L, H, new_T - cur_T), dtype=_np.float64)
        v9_new.hidden_wins_primary = _np.concatenate(
            [v9_new.hidden_wins_primary, pad], axis=-1)
        v9_new.replay_dominance_witness = _np.concatenate(
            [v9_new.replay_dominance_witness, pad.copy()],
            axis=-1)
    v9_new.write_log_v9.append({
        "schema": W64_TINY_SUBSTRATE_V9_SCHEMA_VERSION,
        "kind": "forward_v9",
        "n_new_tokens": int(len(list(token_ids))),
        "attention_entropy_mean": float(ent.mean())
            if ent.size else 0.0,
        "cache_similarity_mean": float(sim.mean())
            if sim.size else 0.0,
    })
    trace = TinyV9ForwardTrace(
        v8_trace=v8_trace,
        attention_entropy_per_layer=ent,
        config_cid=str(cfg.cid()),
        params_cid=str(params.cid()),
    )
    return trace, v9_new


def record_hidden_wins_primary_v9(
        cache: TinyV9KVCache, *,
        layer_index: int, head_index: int, slot: int,
        decision: str = "hidden_wins",
) -> None:
    """Record a per-(layer, head, slot) hidden-wins decision flag.
    decision ∈ {hidden_wins, hidden_loses, kv_wins, tie}."""
    L, H, T = cache.hidden_wins_primary.shape
    if not (0 <= int(layer_index) < L
            and 0 <= int(head_index) < H):
        return
    if int(slot) >= T:
        pad = _np.zeros(
            (L, H, int(slot) - T + 1), dtype=_np.float64)
        cache.hidden_wins_primary = _np.concatenate(
            [cache.hidden_wins_primary, pad], axis=-1)
    flag = float(_HIDDEN_DECISION_TO_FLAG.get(
        str(decision),
        W64_HIDDEN_DECISION_TIE_FLAG))
    cache.hidden_wins_primary[
        int(layer_index), int(head_index),
        int(slot)] = float(flag)
    cache.write_log_v9.append({
        "schema": W64_TINY_SUBSTRATE_V9_SCHEMA_VERSION,
        "kind": "hidden_wins_primary_write",
        "layer_index": int(layer_index),
        "head_index": int(head_index),
        "slot": int(slot),
        "decision": str(decision),
        "flag": float(flag),
    })


def record_replay_dominance_witness_v9(
        cache: TinyV9KVCache, *,
        layer_index: int, head_index: int, slot: int,
        dominance: float,
) -> None:
    """Record per-(layer, head, slot) replay-dominance scalar."""
    L, H, T = cache.replay_dominance_witness.shape
    if not (0 <= int(layer_index) < L
            and 0 <= int(head_index) < H):
        return
    if int(slot) >= T:
        pad = _np.zeros(
            (L, H, int(slot) - T + 1), dtype=_np.float64)
        cache.replay_dominance_witness = _np.concatenate(
            [cache.replay_dominance_witness, pad], axis=-1)
    cache.replay_dominance_witness[
        int(layer_index), int(head_index),
        int(slot)] = float(dominance)
    cache.write_log_v9.append({
        "schema": W64_TINY_SUBSTRATE_V9_SCHEMA_VERSION,
        "kind": "replay_dominance_witness_write",
        "layer_index": int(layer_index),
        "head_index": int(head_index),
        "slot": int(slot),
        "dominance": float(round(dominance, 12)),
    })


def record_hidden_state_trust_decision_v9(
        cache: TinyV9KVCache, *,
        layer_index: int, head_index: int,
        decision: str,
        trust_ema: float = W64_DEFAULT_V9_TRUST_EMA,
) -> None:
    """EMA-update the hidden-state-trust ledger for (layer, head)."""
    L = int(cache.hidden_state_trust_ledger.shape[0])
    H = int(cache.hidden_state_trust_ledger.shape[1])
    if not (0 <= int(layer_index) < L
            and 0 <= int(head_index) < H):
        return
    flag = float(_HIDDEN_DECISION_TO_FLAG.get(
        str(decision),
        W64_HIDDEN_DECISION_TIE_FLAG))
    alpha = float(max(0.0, min(1.0, float(trust_ema))))
    prev = float(cache.hidden_state_trust_ledger[
        int(layer_index), int(head_index)])
    cache.hidden_state_trust_ledger[
        int(layer_index), int(head_index)] = (
        alpha * flag + (1.0 - alpha) * prev)
    cache.write_log_v9.append({
        "schema": W64_TINY_SUBSTRATE_V9_SCHEMA_VERSION,
        "kind": "hidden_state_trust_decision",
        "layer_index": int(layer_index),
        "head_index": int(head_index),
        "decision": str(decision),
        "flag": float(round(flag, 12)),
    })


def hidden_wins_primary_summary(
        cache: TinyV9KVCache,
) -> dict[str, float]:
    """Summary of the hidden-wins-primary tensor."""
    h = _np.asarray(
        cache.hidden_wins_primary, dtype=_np.float64)
    n_pos = float((h > 0.0).sum())
    n_neg = float((h < 0.0).sum())
    n_tie = float((h == 0.0).sum())
    margin = float((h > 0.0).sum() - (h < 0.0).sum())
    return {
        "hidden_win_count": float(n_pos),
        "hidden_lose_count": float(n_neg),
        "tie_count": float(n_tie),
        "primary_margin": float(margin),
    }


def is_hidden_wins_primary_regime(
        cache: TinyV9KVCache, *,
        threshold: float = (
            W64_DEFAULT_V9_HIDDEN_WINS_PRIMARY_THRESHOLD),
) -> bool:
    """Returns True when the hidden-wins-primary regime is active.
    Defined as: mean(hidden_wins_primary > 0) - mean(< 0) >=
    threshold."""
    h = _np.asarray(
        cache.hidden_wins_primary, dtype=_np.float64)
    if h.size == 0:
        return False
    pos_rate = float((h > 0.0).sum()) / float(h.size)
    neg_rate = float((h < 0.0).sum()) / float(h.size)
    return bool((pos_rate - neg_rate) >= float(threshold))


# ---------------------------------------------------------------------------
# Witness
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TinyV9SubstrateForwardWitness:
    config_cid: str
    params_cid: str
    token_ids: tuple[int, ...]
    v8_witness_cid: str
    attention_entropy_cid: str
    attention_entropy_mean: float
    cache_similarity_matrix_cid: str
    cache_similarity_mean: float
    hidden_wins_primary_l1: float
    replay_dominance_witness_l1: float
    hidden_state_trust_ledger_l1: float
    hidden_wins_primary_active: bool
    n_layers: int
    n_heads: int
    forward_count: int
    schema: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "config_cid": str(self.config_cid),
            "params_cid": str(self.params_cid),
            "token_ids": list(self.token_ids),
            "v8_witness_cid": str(self.v8_witness_cid),
            "attention_entropy_cid": str(
                self.attention_entropy_cid),
            "attention_entropy_mean": float(round(
                self.attention_entropy_mean, 12)),
            "cache_similarity_matrix_cid": str(
                self.cache_similarity_matrix_cid),
            "cache_similarity_mean": float(round(
                self.cache_similarity_mean, 12)),
            "hidden_wins_primary_l1": float(round(
                self.hidden_wins_primary_l1, 12)),
            "replay_dominance_witness_l1": float(round(
                self.replay_dominance_witness_l1, 12)),
            "hidden_state_trust_ledger_l1": float(round(
                self.hidden_state_trust_ledger_l1, 12)),
            "hidden_wins_primary_active": bool(
                self.hidden_wins_primary_active),
            "n_layers": int(self.n_layers),
            "n_heads": int(self.n_heads),
            "forward_count": int(self.forward_count),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v9_substrate_forward_witness",
            "witness": self.to_dict()})


def emit_tiny_substrate_v9_forward_witness(
        trace: TinyV9ForwardTrace,
        cache: TinyV9KVCache,
) -> TinyV9SubstrateForwardWitness:
    from .tiny_substrate_v8 import (
        emit_tiny_substrate_v8_forward_witness,
    )
    v8w = emit_tiny_substrate_v8_forward_witness(
        trace.v8_trace, cache.v8_cache)
    return TinyV9SubstrateForwardWitness(
        schema=W64_TINY_SUBSTRATE_V9_SCHEMA_VERSION,
        config_cid=str(trace.config_cid),
        params_cid=str(trace.params_cid),
        token_ids=tuple(int(t) for t in v8w.token_ids),
        v8_witness_cid=str(v8w.cid()),
        attention_entropy_cid=_ndarray_cid(
            _np.asarray(
                trace.attention_entropy_per_layer,
                dtype=_np.float64)),
        attention_entropy_mean=float(
            trace.attention_entropy_per_layer.mean())
            if trace.attention_entropy_per_layer.size else 0.0,
        cache_similarity_matrix_cid=_ndarray_cid(
            _np.asarray(
                cache.cache_similarity_matrix,
                dtype=_np.float64)),
        cache_similarity_mean=float(
            cache.cache_similarity_matrix.mean())
            if cache.cache_similarity_matrix.size else 0.0,
        hidden_wins_primary_l1=float(_np.linalg.norm(
            cache.hidden_wins_primary.ravel(), ord=1)),
        replay_dominance_witness_l1=float(_np.linalg.norm(
            cache.replay_dominance_witness.ravel(), ord=1)),
        hidden_state_trust_ledger_l1=float(_np.linalg.norm(
            cache.hidden_state_trust_ledger.ravel(), ord=1)),
        hidden_wins_primary_active=bool(
            is_hidden_wins_primary_regime(cache)),
        n_layers=int(cache.n_layers()),
        n_heads=int(cache.hidden_state_trust_ledger.shape[1]),
        forward_count=int(
            cache.v8_cache.v7_cache.v6_cache.forward_count),
    )


def build_default_tiny_substrate_v9(
        *, seed: int = W64_DEFAULT_V9_SEED,
) -> TinyV9SubstrateParams:
    return TinyV9SubstrateParams.init(
        TinyV9SubstrateConfig(seed=int(seed)))


def tokenize_bytes_v9(
        text: str, *,
        max_len: int = W64_DEFAULT_V9_MAX_LEN,
        add_bos: bool = True,
) -> list[int]:
    from .tiny_substrate_v7 import tokenize_bytes_v7
    return tokenize_bytes_v7(
        text, max_len=int(max_len), add_bos=bool(add_bos))


def detokenize_bytes_v9(token_ids: Sequence[int]) -> str:
    from .tiny_substrate_v7 import detokenize_bytes_v7
    return detokenize_bytes_v7(list(token_ids))


__all__ = [
    "W64_TINY_SUBSTRATE_V9_SCHEMA_VERSION",
    "W64_TINY_V9_VOCAB_SIZE",
    "W64_DEFAULT_V9_D_MODEL",
    "W64_DEFAULT_V9_N_HEADS",
    "W64_DEFAULT_V9_N_KV_HEADS",
    "W64_DEFAULT_V9_N_LAYERS",
    "W64_DEFAULT_V9_FF_HIDDEN",
    "W64_DEFAULT_V9_MAX_LEN",
    "W64_DEFAULT_V9_INIT_SCALE",
    "W64_DEFAULT_V9_SEED",
    "W64_DEFAULT_V9_ROPE_BASE",
    "W64_DEFAULT_V9_D_KEY",
    "W64_DEFAULT_V9_TRUST_EMA",
    "W64_DEFAULT_V9_REPLAY_DETERMINISM_TOL",
    "W64_DEFAULT_V9_HIDDEN_WINS_PRIMARY_THRESHOLD",
    "W64_HIDDEN_DECISION_WIN_FLAG",
    "W64_HIDDEN_DECISION_LOSE_FLAG",
    "W64_HIDDEN_DECISION_TIE_FLAG",
    "TinyV9SubstrateConfig",
    "TinyV9SubstrateParams",
    "TinyV9KVCache",
    "TinyV9ForwardTrace",
    "TinyV9SubstrateForwardWitness",
    "forward_tiny_substrate_v9",
    "record_hidden_wins_primary_v9",
    "record_replay_dominance_witness_v9",
    "record_hidden_state_trust_decision_v9",
    "hidden_wins_primary_summary",
    "is_hidden_wins_primary_regime",
    "emit_tiny_substrate_v9_forward_witness",
    "build_default_tiny_substrate_v9",
    "tokenize_bytes_v9", "detokenize_bytes_v9",
]
