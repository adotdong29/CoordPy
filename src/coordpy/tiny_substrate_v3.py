"""W58 M1 — Tiny Transformer Runtime V3.

Strictly extends W57's ``coordpy.tiny_substrate_v2``. V3 keeps every
V2 invariant (byte-determinism, strict causal mask, RoPE,
prefix-state, attention-bias hook, logit lens, KV eviction) and
adds five new substrate-load-bearing pieces:

* **Grouped-Query Attention (GQA)** — ``n_kv_heads`` can be smaller
  than ``n_heads``. KV groups are formed by integer division
  ``head -> kv_head = head // group_size`` where
  ``group_size = n_heads // n_kv_heads``. The cache is stored
  per-kv-head, not per-head, so it occupies less memory; multiple
  query heads attend to the same K/V slot. Defaults to
  ``n_kv_heads=4`` with ``n_heads=8`` (group of 2).
* **RMSNorm** — optional RMS LayerNorm replaces V2's mean-var
  LayerNorm at every layer when ``use_rmsnorm=True``.
* **SwiGLU** feed-forward — optional gated FF with
  ``swish(x W_gate) * (x W_up) W_down`` when ``use_swiglu=True``.
* **Per-token KV importance tracking** — every layer's forward
  computes ``attention_received[t] = sum over (query, head) of
  attn_weights[head, q, t]``. The trace exposes this as
  ``kv_importance_per_layer`` (one ``(n_tokens,)`` vector per
  layer). The W58 cache controller uses this to score retention.
* **Real flop counter** — every forward records a fp64 operation
  estimate. Used by the prefix-state V2 reuse-vs-recompute bar
  (H100).

Plus:

* **Partial forward** — ``partial_forward_layers_l_to_end`` runs
  only the suffix of layers starting from layer ``l``, given a
  pre-computed residual stream at the boundary. This is the
  building block for layer-wise interventions.
* **KV fingerprint** — every cache carries a 64-bucket
  Reed-Solomon-style XOR fingerprint. The fingerprint is part of
  the cache CID and is what the CRC V6 corruption detector checks.

Honest scope (do-not-overstate, W58)
------------------------------------

* This is still NOT a frontier model. Default config is
  ``5 layers / 8 query heads / 4 kv heads / d_model=64 /
  byte-vocab / max_len=120 / untrained``. The runtime is richer
  than V2 but still a research substrate in pure NumPy on CPU.
  ``W58-L-NUMPY-CPU-V3-SUBSTRATE-CAP`` documents this.
* The V3 runtime still does NOT bridge to third-party hosted
  models. Ollama / OpenAI / hosted APIs remain text-only at the
  HTTP surface. ``W58-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``
  carries forward the W57 cap unchanged.
* The default config is untrained. The W58 KV bridge V3 *fits*
  per-head inject scales by finite differences against a fixed
  target perturbation, which is a small, scoped, gradient-free
  fit on the inject scale only — NOT end-to-end backprop. The
  ``W58-L-V3-NO-BACKPROP-CAP`` documents the boundary.
* GQA / RMSNorm / SwiGLU are standard. We claim a richer
  substrate, not a novel architecture.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover - numpy is required.
    raise ImportError(
        "coordpy.tiny_substrate_v3 requires numpy") from exc


W58_TINY_SUBSTRATE_V3_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v3.v1")

W58_TINY_V3_VOCAB_SIZE: int = 259  # byte vocab + PAD + BOS + EOS.
W58_TINY_V3_PAD_TOKEN: int = 256
W58_TINY_V3_BOS_TOKEN: int = 257
W58_TINY_V3_EOS_TOKEN: int = 258

W58_DEFAULT_V3_D_MODEL: int = 64
W58_DEFAULT_V3_N_HEADS: int = 8
W58_DEFAULT_V3_N_KV_HEADS: int = 4
W58_DEFAULT_V3_N_LAYERS: int = 5
W58_DEFAULT_V3_FF_HIDDEN: int = 160
W58_DEFAULT_V3_MAX_LEN: int = 120
W58_DEFAULT_V3_INIT_SCALE: float = 0.04
W58_DEFAULT_V3_SEED: int = 58012345
W58_DEFAULT_V3_ROPE_BASE: float = 10000.0


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _ndarray_cid(arr: "_np.ndarray") -> str:
    a = _np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(repr(a.shape).encode("utf-8"))
    h.update(b"|")
    h.update(str(a.dtype).encode("utf-8"))
    h.update(b"|")
    h.update(a.tobytes())
    return h.hexdigest()


def _seeded_rng(seed: int) -> "_np.random.Generator":
    return _np.random.default_rng(int(seed))


# ---------------------------------------------------------------------------
# 64-bucket Reed-Solomon-style XOR fingerprint of a flat byte stream
# ---------------------------------------------------------------------------


def _kv_fingerprint_64(arr: "_np.ndarray") -> tuple[int, ...]:
    """Return a 64-bucket XOR fingerprint of an array's raw bytes."""
    raw = _np.ascontiguousarray(arr).tobytes()
    buckets = [0] * 64
    for i, b in enumerate(raw):
        buckets[i % 64] ^= int(b)
    return tuple(int(b) for b in buckets)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def _rope_freqs_v3(
        d_head: int, max_len: int,
        *, base: float = W58_DEFAULT_V3_ROPE_BASE,
) -> "_np.ndarray":
    half = int(d_head) // 2
    if half == 0:
        return _np.zeros((int(max_len), int(d_head)),
                          dtype=_np.float64)
    inv_freq = 1.0 / (
        float(base) ** (_np.arange(0, half, dtype=_np.float64)
                        * 2.0 / float(d_head)))
    positions = _np.arange(int(max_len), dtype=_np.float64)
    angles = positions[:, None] * inv_freq[None, :]
    sin_a = _np.sin(angles)
    cos_a = _np.cos(angles)
    out = _np.zeros((int(max_len), int(d_head)), dtype=_np.float64)
    out[:, 0::2] = cos_a[:, : int(d_head) // 2]
    out[:, 1::2] = sin_a[:, : int(d_head) // 2]
    return out


def _apply_rope_v3(
        x: "_np.ndarray", positions: "_np.ndarray",
        rope_table: "_np.ndarray",
) -> "_np.ndarray":
    h, t, dh = x.shape
    if dh < 2:
        return x.copy()
    rope = rope_table[positions]
    cos = rope[:, 0::2]
    sin = rope[:, 1::2]
    x_even = x[:, :, 0::2]
    x_odd = x[:, :, 1::2]
    rot_even = x_even * cos[None, :, :] - x_odd * sin[None, :, :]
    rot_odd = x_odd * cos[None, :, :] + x_even * sin[None, :, :]
    out = _np.empty_like(x)
    out[:, :, 0::2] = rot_even
    out[:, :, 1::2] = rot_odd
    return out


# ---------------------------------------------------------------------------
# Parameters (GQA-aware)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TinyV3AttentionParams:
    w_q: "_np.ndarray"
    w_k: "_np.ndarray"
    w_v: "_np.ndarray"
    w_o: "_np.ndarray"
    n_heads: int
    n_kv_heads: int

    @classmethod
    def init(cls, *, d_model: int, n_heads: int, n_kv_heads: int,
             seed: int, init_scale: float,
             ) -> "TinyV3AttentionParams":
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} not divisible by n_heads={n_heads}")
        if n_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_heads={n_heads} not divisible by "
                f"n_kv_heads={n_kv_heads}")
        d_head = d_model // n_heads
        d_kv = int(n_kv_heads) * int(d_head)
        rng = _seeded_rng(seed)
        s = float(init_scale)
        return cls(
            w_q=rng.standard_normal((d_model, d_model)) * s,
            w_k=rng.standard_normal((d_model, d_kv)) * s,
            w_v=rng.standard_normal((d_model, d_kv)) * s,
            w_o=rng.standard_normal((d_model, d_model)) * s,
            n_heads=int(n_heads),
            n_kv_heads=int(n_kv_heads),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v3_attn_params",
            "w_q_cid": _ndarray_cid(self.w_q),
            "w_k_cid": _ndarray_cid(self.w_k),
            "w_v_cid": _ndarray_cid(self.w_v),
            "w_o_cid": _ndarray_cid(self.w_o),
            "n_heads": int(self.n_heads),
            "n_kv_heads": int(self.n_kv_heads),
        })


@dataclasses.dataclass
class TinyV3FFParams:
    use_swiglu: bool
    w_1: "_np.ndarray"
    b_1: "_np.ndarray"
    w_2: "_np.ndarray"
    b_2: "_np.ndarray"
    w_gate: "_np.ndarray"  # only used if use_swiglu

    @classmethod
    def init(cls, *, d_model: int, ff_hidden: int,
             seed: int, init_scale: float, use_swiglu: bool,
             ) -> "TinyV3FFParams":
        rng = _seeded_rng(seed)
        s = float(init_scale)
        return cls(
            use_swiglu=bool(use_swiglu),
            w_1=rng.standard_normal((d_model, ff_hidden)) * s,
            b_1=_np.zeros(ff_hidden, dtype=_np.float64),
            w_2=rng.standard_normal((ff_hidden, d_model)) * s,
            b_2=_np.zeros(d_model, dtype=_np.float64),
            w_gate=rng.standard_normal((d_model, ff_hidden)) * s,
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v3_ff_params",
            "use_swiglu": bool(self.use_swiglu),
            "w_1_cid": _ndarray_cid(self.w_1),
            "b_1_cid": _ndarray_cid(self.b_1),
            "w_2_cid": _ndarray_cid(self.w_2),
            "b_2_cid": _ndarray_cid(self.b_2),
            "w_gate_cid": _ndarray_cid(self.w_gate),
        })


@dataclasses.dataclass
class TinyV3LayerNormParams:
    use_rms: bool
    gamma: "_np.ndarray"
    beta: "_np.ndarray"

    @classmethod
    def init(cls, *, d_model: int, use_rms: bool
             ) -> "TinyV3LayerNormParams":
        return cls(
            use_rms=bool(use_rms),
            gamma=_np.ones(d_model, dtype=_np.float64),
            beta=_np.zeros(d_model, dtype=_np.float64),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v3_ln_params",
            "use_rms": bool(self.use_rms),
            "gamma_cid": _ndarray_cid(self.gamma),
            "beta_cid": _ndarray_cid(self.beta),
        })


@dataclasses.dataclass
class TinyV3LayerParams:
    attn: TinyV3AttentionParams
    ff: TinyV3FFParams
    ln1: TinyV3LayerNormParams
    ln2: TinyV3LayerNormParams

    @classmethod
    def init(cls, *, d_model: int, n_heads: int, n_kv_heads: int,
             ff_hidden: int, seed: int, init_scale: float,
             use_rmsnorm: bool, use_swiglu: bool,
             ) -> "TinyV3LayerParams":
        return cls(
            attn=TinyV3AttentionParams.init(
                d_model=d_model, n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                seed=int(seed) + 1, init_scale=init_scale),
            ff=TinyV3FFParams.init(
                d_model=d_model, ff_hidden=ff_hidden,
                seed=int(seed) + 2, init_scale=init_scale,
                use_swiglu=bool(use_swiglu)),
            ln1=TinyV3LayerNormParams.init(
                d_model=d_model, use_rms=bool(use_rmsnorm)),
            ln2=TinyV3LayerNormParams.init(
                d_model=d_model, use_rms=bool(use_rmsnorm)),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v3_layer_params",
            "attn_cid": self.attn.cid(),
            "ff_cid": self.ff.cid(),
            "ln1_cid": self.ln1.cid(),
            "ln2_cid": self.ln2.cid(),
        })


@dataclasses.dataclass
class TinyV3SubstrateConfig:
    vocab_size: int = W58_TINY_V3_VOCAB_SIZE
    d_model: int = W58_DEFAULT_V3_D_MODEL
    n_heads: int = W58_DEFAULT_V3_N_HEADS
    n_kv_heads: int = W58_DEFAULT_V3_N_KV_HEADS
    n_layers: int = W58_DEFAULT_V3_N_LAYERS
    ff_hidden: int = W58_DEFAULT_V3_FF_HIDDEN
    max_len: int = W58_DEFAULT_V3_MAX_LEN
    init_scale: float = W58_DEFAULT_V3_INIT_SCALE
    seed: int = W58_DEFAULT_V3_SEED
    rope_base: float = W58_DEFAULT_V3_ROPE_BASE
    use_rope: bool = True
    use_rmsnorm: bool = True
    use_swiglu: bool = True
    track_kv_importance: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W58_TINY_SUBSTRATE_V3_SCHEMA_VERSION,
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
            "track_kv_importance": bool(self.track_kv_importance),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v3_substrate_config",
            "config": self.to_dict()})


@dataclasses.dataclass
class TinyV3SubstrateParams:
    config: TinyV3SubstrateConfig
    embed: "_np.ndarray"
    pos_embed: "_np.ndarray"
    rope_table: "_np.ndarray"
    layers: list[TinyV3LayerParams]
    ln_f: TinyV3LayerNormParams
    unembed: "_np.ndarray"

    @classmethod
    def init(
            cls, config: TinyV3SubstrateConfig | None = None,
    ) -> "TinyV3SubstrateParams":
        if config is None:
            config = TinyV3SubstrateConfig()
        rng = _seeded_rng(config.seed)
        s = float(config.init_scale)
        embed = rng.standard_normal(
            (config.vocab_size, config.d_model)) * s
        pos_embed = rng.standard_normal(
            (config.max_len, config.d_model)) * s
        d_head = int(config.d_model) // int(config.n_heads)
        rope = _rope_freqs_v3(d_head=int(d_head),
                               max_len=int(config.max_len),
                               base=float(config.rope_base))
        layers: list[TinyV3LayerParams] = []
        for i in range(config.n_layers):
            layers.append(TinyV3LayerParams.init(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                ff_hidden=config.ff_hidden,
                seed=int(config.seed) + 1000 * (i + 1),
                init_scale=s,
                use_rmsnorm=bool(config.use_rmsnorm),
                use_swiglu=bool(config.use_swiglu)))
        ln_f = TinyV3LayerNormParams.init(
            d_model=config.d_model, use_rms=bool(config.use_rmsnorm))
        unembed = rng.standard_normal(
            (config.d_model, config.vocab_size)) * s
        return cls(
            config=config,
            embed=embed,
            pos_embed=pos_embed,
            rope_table=rope,
            layers=layers,
            ln_f=ln_f,
            unembed=unembed,
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v3_substrate_params",
            "config_cid": self.config.cid(),
            "embed_cid": _ndarray_cid(self.embed),
            "pos_embed_cid": _ndarray_cid(self.pos_embed),
            "rope_cid": _ndarray_cid(self.rope_table),
            "layer_cids": [l.cid() for l in self.layers],
            "ln_f_cid": self.ln_f.cid(),
            "unembed_cid": _ndarray_cid(self.unembed),
        })


# ---------------------------------------------------------------------------
# KV cache V3 (GQA-aware + importance + fingerprint)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TinyV3KVCache:
    keys: list["_np.ndarray"]      # (T, d_kv) per layer
    values: list["_np.ndarray"]    # (T, d_kv) per layer
    importance: list["_np.ndarray"]  # (T,) per layer
    write_log: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(cls, n_layers: int) -> "TinyV3KVCache":
        return cls(
            keys=[_np.zeros((0, 0), dtype=_np.float64)
                  for _ in range(int(n_layers))],
            values=[_np.zeros((0, 0), dtype=_np.float64)
                    for _ in range(int(n_layers))],
            importance=[_np.zeros((0,), dtype=_np.float64)
                         for _ in range(int(n_layers))],
            write_log=[],
        )

    def n_tokens(self) -> int:
        if not self.keys:
            return 0
        return int(self.keys[0].shape[0])

    def clone(self) -> "TinyV3KVCache":
        return TinyV3KVCache(
            keys=[k.copy() for k in self.keys],
            values=[v.copy() for v in self.values],
            importance=[i.copy() for i in self.importance],
            write_log=list(self.write_log),
        )

    def evict_lru(self, k: int) -> "TinyV3KVCache":
        if int(k) <= 0:
            return self.clone()
        out = self.clone()
        n = out.n_tokens()
        drop = min(int(k), int(n))
        for i in range(len(out.keys)):
            if out.keys[i].size and out.keys[i].shape[0] > drop:
                out.keys[i] = out.keys[i][drop:].copy()
                out.values[i] = out.values[i][drop:].copy()
                out.importance[i] = (
                    out.importance[i][drop:].copy())
            elif out.keys[i].size:
                out.keys[i] = out.keys[i][:0].copy()
                out.values[i] = out.values[i][:0].copy()
                out.importance[i] = out.importance[i][:0].copy()
        out.write_log.append({
            "schema": W58_TINY_SUBSTRATE_V3_SCHEMA_VERSION,
            "kind": "evict_lru",
            "dropped": int(drop),
        })
        return out

    def evict_weighted(
            self, weights: Sequence[float], keep: int,
    ) -> "TinyV3KVCache":
        out = self.clone()
        n = out.n_tokens()
        if n == 0 or int(keep) >= n:
            return out
        w = list(weights)[:n]
        while len(w) < n:
            w.append(0.0)
        order = sorted(range(n), key=lambda i: -float(w[i]))
        keep_set = sorted(order[: int(keep)])
        idx = _np.asarray(keep_set, dtype=_np.int64)
        for i in range(len(out.keys)):
            if out.keys[i].size:
                out.keys[i] = out.keys[i][idx].copy()
                out.values[i] = out.values[i][idx].copy()
                out.importance[i] = out.importance[i][idx].copy()
        out.write_log.append({
            "schema": W58_TINY_SUBSTRATE_V3_SCHEMA_VERSION,
            "kind": "evict_weighted",
            "kept": list(keep_set),
            "weights_total": float(round(sum(w), 12)),
        })
        return out

    def evict_by_importance(self, keep: int) -> "TinyV3KVCache":
        """Use the *intrinsic* importance vectors (sum across layers)
        as the eviction signal. This is the canonical W58 reuse path.
        """
        n = self.n_tokens()
        if n == 0:
            return self.clone()
        # Aggregate importance across layers.
        total = _np.zeros((n,), dtype=_np.float64)
        for imp in self.importance:
            if imp.size and imp.shape[0] == n:
                total = total + imp
        return self.evict_weighted(total.tolist(), int(keep))

    def fingerprint(self) -> tuple[int, ...]:
        """64-bucket XOR fingerprint over all layers' K, V bytes."""
        h = [0] * 64
        for k, v in zip(self.keys, self.values):
            for src in (k, v):
                fp = _kv_fingerprint_64(src)
                for i, b in enumerate(fp):
                    h[i] ^= int(b)
        return tuple(int(b) for b in h)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v3_kv_cache",
            "n_tokens": int(self.n_tokens()),
            "n_layers": int(len(self.keys)),
            "keys_cids": [_ndarray_cid(k) for k in self.keys],
            "values_cids": [_ndarray_cid(v) for v in self.values],
            "importance_cids": [
                _ndarray_cid(i) for i in self.importance],
            "fingerprint": list(self.fingerprint()),
            "write_log": list(self.write_log),
        })


# ---------------------------------------------------------------------------
# Prefix state V2 (with redundancy, optional)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TinyV3PrefixState:
    prefix_len: int
    keys: tuple["_np.ndarray", ...]
    values: tuple["_np.ndarray", ...]
    importance: tuple["_np.ndarray", ...]
    source_params_cid: str
    redundant_copy_cid: str

    def to_cache(self) -> TinyV3KVCache:
        return TinyV3KVCache(
            keys=[k.copy() for k in self.keys],
            values=[v.copy() for v in self.values],
            importance=[i.copy() for i in self.importance],
            write_log=[{
                "schema": W58_TINY_SUBSTRATE_V3_SCHEMA_VERSION,
                "kind": "prefix_state_v3_load",
                "prefix_len": int(self.prefix_len),
                "source_params_cid": str(self.source_params_cid),
            }],
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v3_prefix_state",
            "prefix_len": int(self.prefix_len),
            "keys_cids": [_ndarray_cid(k) for k in self.keys],
            "values_cids": [_ndarray_cid(v) for v in self.values],
            "importance_cids": [
                _ndarray_cid(i) for i in self.importance],
            "source_params_cid": str(self.source_params_cid),
            "redundant_copy_cid": str(self.redundant_copy_cid),
        })


def extract_prefix_state_v3(
        cache: TinyV3KVCache, *,
        prefix_len: int,
        source_params_cid: str,
) -> TinyV3PrefixState:
    n = cache.n_tokens()
    L = min(int(prefix_len), int(n))
    keys: list["_np.ndarray"] = []
    values: list["_np.ndarray"] = []
    importance: list["_np.ndarray"] = []
    for i in range(len(cache.keys)):
        if cache.keys[i].size:
            keys.append(cache.keys[i][:L].copy())
            values.append(cache.values[i][:L].copy())
            importance.append(
                cache.importance[i][:L].copy()
                if cache.importance[i].size
                else _np.zeros((L,), dtype=_np.float64))
        else:
            keys.append(cache.keys[i].copy())
            values.append(cache.values[i].copy())
            importance.append(_np.zeros((0,), dtype=_np.float64))
    redundant = _sha256_hex({
        "keys_cids": [_ndarray_cid(k) for k in keys],
        "values_cids": [_ndarray_cid(v) for v in values],
    })
    return TinyV3PrefixState(
        prefix_len=int(L),
        keys=tuple(keys),
        values=tuple(values),
        importance=tuple(importance),
        source_params_cid=str(source_params_cid),
        redundant_copy_cid=str(redundant),
    )


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def _swish(x: "_np.ndarray") -> "_np.ndarray":
    return x * (1.0 / (1.0 + _np.exp(-x)))


def _gelu(x: "_np.ndarray") -> "_np.ndarray":
    return 0.5 * x * (1.0 + _np.tanh(
        math.sqrt(2.0 / math.pi)
        * (x + 0.044715 * (x ** 3))))


def _norm_v3(
        x: "_np.ndarray", params: TinyV3LayerNormParams,
        *, eps: float = 1e-5,
) -> "_np.ndarray":
    if params.use_rms:
        rms = _np.sqrt(
            _np.mean(x * x, axis=-1, keepdims=True) + eps)
        y = x / rms
    else:
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        y = (x - mu) / _np.sqrt(var + eps)
    return y * params.gamma + params.beta


def _softmax(x: "_np.ndarray", axis: int = -1) -> "_np.ndarray":
    m = _np.max(x, axis=axis, keepdims=True)
    z = _np.exp(x - m)
    return z / _np.sum(z, axis=axis, keepdims=True)


def _split_heads_q(
        x: "_np.ndarray", n_heads: int,
) -> "_np.ndarray":
    n_tokens, d_model = x.shape
    d_head = d_model // n_heads
    return x.reshape(n_tokens, n_heads, d_head).transpose(1, 0, 2)


def _split_heads_kv(
        x: "_np.ndarray", n_kv_heads: int,
) -> "_np.ndarray":
    n_tokens, d_kv = x.shape
    d_head = d_kv // n_kv_heads
    return x.reshape(
        n_tokens, n_kv_heads, d_head).transpose(1, 0, 2)


def _merge_heads(x: "_np.ndarray") -> "_np.ndarray":
    n_heads, n_tokens, d_head = x.shape
    return x.transpose(1, 0, 2).reshape(n_tokens, n_heads * d_head)


def _gqa_broadcast_k_v(
        kv_h: "_np.ndarray", n_heads: int, n_kv_heads: int,
) -> "_np.ndarray":
    """Broadcast ``(n_kv_heads, T, d_head)`` to
    ``(n_heads, T, d_head)`` by repeating each kv head
    ``n_heads // n_kv_heads`` times."""
    if n_heads == n_kv_heads:
        return kv_h
    group = n_heads // n_kv_heads
    return _np.repeat(kv_h, repeats=int(group), axis=0)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TinyV3ForwardTrace:
    logits: "_np.ndarray"
    hidden_states: list["_np.ndarray"]
    attn_weights_per_layer: list["_np.ndarray"]
    kv_cache: TinyV3KVCache
    per_layer_logit_lens: list["_np.ndarray"]
    kv_importance_per_layer: list["_np.ndarray"]
    flop_count: int
    flop_count_per_layer: list[int]
    token_ids: tuple[int, ...]
    config_cid: str
    params_cid: str

    def hidden_state_at_layer(self, layer: int) -> "_np.ndarray":
        i = int(layer)
        if i < 0:
            i = len(self.hidden_states) + i
        return self.hidden_states[int(i)]

    def logit_lens_at_layer(self, layer: int) -> "_np.ndarray":
        i = int(layer)
        if i < 0:
            i = len(self.per_layer_logit_lens) + i
        return self.per_layer_logit_lens[int(i)]

    def hidden_state_cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v3_forward_hidden",
            "config_cid": self.config_cid,
            "params_cid": self.params_cid,
            "token_ids": list(self.token_ids),
            "hidden_state_cids": [
                _ndarray_cid(h) for h in self.hidden_states],
        })

    def logits_cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v3_forward_logits",
            "logits_cid": _ndarray_cid(self.logits),
        })

    def attention_cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v3_forward_attn",
            "attn_weights_cids": [
                _ndarray_cid(a)
                for a in self.attn_weights_per_layer],
        })

    def per_layer_logit_lens_cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v3_forward_logit_lens",
            "lens_cids": [
                _ndarray_cid(l)
                for l in self.per_layer_logit_lens],
        })

    def importance_cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v3_forward_importance",
            "importance_cids": [
                _ndarray_cid(i)
                for i in self.kv_importance_per_layer],
        })

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v3_forward_trace",
            "hidden_state_cid": self.hidden_state_cid(),
            "logits_cid": self.logits_cid(),
            "attention_cid": self.attention_cid(),
            "logit_lens_cid": self.per_layer_logit_lens_cid(),
            "importance_cid": self.importance_cid(),
            "kv_cache_cid": self.kv_cache.cid(),
            "flop_count": int(self.flop_count),
            "token_ids": list(self.token_ids),
        })


def _attention_layer_forward_v3(
        x: "_np.ndarray",
        params: TinyV3AttentionParams,
        *,
        kv_keys_prev: "_np.ndarray",
        kv_values_prev: "_np.ndarray",
        positions_new: "_np.ndarray",
        rope_table: "_np.ndarray",
        use_rope: bool,
        attention_bias: "_np.ndarray | None",
) -> tuple["_np.ndarray", "_np.ndarray",
           "_np.ndarray", "_np.ndarray", int]:
    """Multi-head causal self-attention with GQA + KV cache + RoPE
    + optional pre-softmax bias.

    Returns ``(layer_out, k_all_kv_flat, v_all_kv_flat,
                attn_weights, flop_estimate)``.
    """
    n_tokens = int(x.shape[0])
    d_model = int(x.shape[1])
    n_heads = int(params.n_heads)
    n_kv_heads = int(params.n_kv_heads)
    d_head = d_model // n_heads
    d_kv = n_kv_heads * d_head

    q = x @ params.w_q                    # (T, d_model)
    k_new = x @ params.w_k                # (T, d_kv)
    v_new = x @ params.w_v                # (T, d_kv)

    q_h = _split_heads_q(q, n_heads)        # (H, T, d_head)
    k_new_h = _split_heads_kv(
        k_new, n_kv_heads)                  # (KH, T, d_head)
    v_new_h = _split_heads_kv(
        v_new, n_kv_heads)                  # (KH, T, d_head)

    if use_rope:
        q_h = _apply_rope_v3(q_h, positions_new, rope_table)
        k_new_h = _apply_rope_v3(
            k_new_h, positions_new, rope_table)

    if kv_keys_prev.size == 0:
        k_all_h = k_new_h
        v_all_h = v_new_h
    else:
        prev_k_h = _split_heads_kv(kv_keys_prev, n_kv_heads)
        prev_v_h = _split_heads_kv(kv_values_prev, n_kv_heads)
        k_all_h = _np.concatenate(
            [prev_k_h, k_new_h], axis=1)
        v_all_h = _np.concatenate(
            [prev_v_h, v_new_h], axis=1)

    # Broadcast kv heads to query heads.
    k_all_h_bcast = _gqa_broadcast_k_v(
        k_all_h, n_heads, n_kv_heads)
    v_all_h_bcast = _gqa_broadcast_k_v(
        v_all_h, n_heads, n_kv_heads)

    scores = _np.einsum(
        "htd,hsd->hts",
        q_h, k_all_h_bcast) / math.sqrt(float(d_head))

    if attention_bias is not None:
        scores = scores + _np.asarray(
            attention_bias, dtype=_np.float64)

    n_prev = (int(kv_keys_prev.shape[0])
              if kv_keys_prev.size else 0)
    mask = _np.full(
        (n_tokens, n_prev + n_tokens), -1e9, dtype=_np.float64)
    for i in range(n_tokens):
        mask[i, : n_prev + i + 1] = 0.0
    scores = scores + mask[None, :, :]

    attn = _softmax(scores, axis=-1)
    out_h = _np.einsum(
        "hts,hsd->htd", attn, v_all_h_bcast)
    out = _merge_heads(out_h) @ params.w_o

    # Merge kv heads back to (T_all, d_kv) for cache storage.
    k_all_kv_flat = _merge_heads(k_all_h)
    v_all_kv_flat = _merge_heads(v_all_h)

    # Flop estimate (rough, order-of-magnitude): two matmuls
    # T*d*d for QKVO + 2*H*T*T_all*d_head for attention.
    T_all = n_prev + n_tokens
    flops = (
        4 * n_tokens * d_model * d_model
        + 2 * n_heads * n_tokens * T_all * d_head)
    return (out, k_all_kv_flat, v_all_kv_flat, attn, int(flops))


def forward_tiny_substrate_v3(
        params: TinyV3SubstrateParams,
        token_ids: Sequence[int],
        *,
        kv_cache: TinyV3KVCache | None = None,
        return_attention: bool = True,
        attention_bias_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
) -> TinyV3ForwardTrace:
    """Real forward over the V3 substrate."""
    cfg = params.config
    if any(int(t) < 0 or int(t) >= cfg.vocab_size
           for t in token_ids):
        raise ValueError(
            f"token_ids must be in [0,{cfg.vocab_size})")
    n_tokens = len(token_ids)
    if n_tokens == 0:
        raise ValueError("token_ids must be non-empty")
    ids = _np.asarray(token_ids, dtype=_np.int64)
    n_prev = 0
    if kv_cache is not None and kv_cache.n_tokens() > 0:
        n_prev = int(kv_cache.n_tokens())
    if n_prev + n_tokens > cfg.max_len:
        raise ValueError(
            f"total tokens {n_prev + n_tokens} exceeds "
            f"max_len {cfg.max_len}")
    pos = _np.arange(n_prev, n_prev + n_tokens, dtype=_np.int64)
    x = params.embed[ids] + params.pos_embed[pos]
    hidden_states = [x.copy()]
    attn_weights: list["_np.ndarray"] = []
    importance: list["_np.ndarray"] = []
    flop_per_layer: list[int] = []
    if kv_cache is None:
        new_cache = TinyV3KVCache.empty(cfg.n_layers)
    else:
        new_cache = kv_cache.clone()
    if len(new_cache.keys) != cfg.n_layers:
        new_cache = TinyV3KVCache.empty(cfg.n_layers)
    biases = (
        list(attention_bias_per_layer)
        if attention_bias_per_layer is not None
        else [None] * cfg.n_layers)
    while len(biases) < cfg.n_layers:
        biases.append(None)
    total_flops = 0
    n_prev_local = int(new_cache.n_tokens())
    d_kv = (int(cfg.d_model) // int(cfg.n_heads)
            * int(cfg.n_kv_heads))
    for layer_idx, layer in enumerate(params.layers):
        x_norm = _norm_v3(x, layer.ln1)
        kv_k = new_cache.keys[layer_idx]
        kv_v = new_cache.values[layer_idx]
        if kv_k.size == 0:
            kv_k = _np.zeros((0, d_kv), dtype=_np.float64)
            kv_v = _np.zeros((0, d_kv), dtype=_np.float64)
        attn_out, k_all, v_all, attn, flops = (
            _attention_layer_forward_v3(
                x_norm, layer.attn,
                kv_keys_prev=kv_k,
                kv_values_prev=kv_v,
                positions_new=pos,
                rope_table=params.rope_table,
                use_rope=bool(cfg.use_rope),
                attention_bias=biases[layer_idx]))
        new_cache.keys[layer_idx] = k_all
        new_cache.values[layer_idx] = v_all
        attn_weights.append(attn)
        flop_per_layer.append(int(flops))
        total_flops += int(flops)
        # KV importance per token: sum over query positions and
        # heads of the attention weights *received* by each key
        # position. Shape: (T_all,).
        T_all = int(attn.shape[-1])
        if bool(cfg.track_kv_importance):
            imp = _np.einsum("hts->s", attn)
        else:
            imp = _np.zeros((T_all,), dtype=_np.float64)
        # Concatenate the previous importance (already in cache)
        # with the freshly observed importance for new keys.
        prev_imp = new_cache.importance[layer_idx]
        # We *replace* prev_imp with the cumulative importance
        # observed in this forward across all key positions.
        new_cache.importance[layer_idx] = imp.astype(_np.float64)
        # Also produce a *new-token-only* importance vector for
        # the trace.
        importance.append(imp.astype(_np.float64).copy())
        x = x + attn_out
        x_norm2 = _norm_v3(x, layer.ln2)
        if bool(layer.ff.use_swiglu):
            gate = _swish(x_norm2 @ layer.ff.w_gate)
            up = x_norm2 @ layer.ff.w_1 + layer.ff.b_1
            ff_h = gate * up
            ff_out = ff_h @ layer.ff.w_2 + layer.ff.b_2
        else:
            ff_h = _gelu(
                x_norm2 @ layer.ff.w_1 + layer.ff.b_1)
            ff_out = ff_h @ layer.ff.w_2 + layer.ff.b_2
        x = x + ff_out
        # Add FF flops to layer total.
        ff_flops = (
            2 * n_tokens * int(cfg.d_model) * int(cfg.ff_hidden))
        flop_per_layer[layer_idx] += int(ff_flops)
        total_flops += int(ff_flops)
        hidden_states.append(x.copy())
    final = _norm_v3(x, params.ln_f)
    logits = final @ params.unembed
    per_layer_lens: list["_np.ndarray"] = []
    for hs in hidden_states:
        per_layer_lens.append(hs @ params.unembed)
    new_cache.write_log.append({
        "schema": W58_TINY_SUBSTRATE_V3_SCHEMA_VERSION,
        "kind": "forward",
        "n_new_tokens": int(n_tokens),
        "n_prev_tokens": int(n_prev_local),
        "flops": int(total_flops),
    })
    return TinyV3ForwardTrace(
        logits=logits,
        hidden_states=hidden_states,
        attn_weights_per_layer=attn_weights,
        kv_cache=new_cache,
        per_layer_logit_lens=per_layer_lens,
        kv_importance_per_layer=importance,
        flop_count=int(total_flops),
        flop_count_per_layer=list(flop_per_layer),
        token_ids=tuple(int(t) for t in token_ids),
        config_cid=cfg.cid(),
        params_cid=params.cid(),
    )


# ---------------------------------------------------------------------------
# Partial forward (suffix of layers)
# ---------------------------------------------------------------------------


def partial_forward_layers_l_to_end_v3(
        params: TinyV3SubstrateParams,
        *,
        residual_at_layer_l: "_np.ndarray",
        start_layer: int,
        positions: "_np.ndarray",
        kv_cache: TinyV3KVCache | None = None,
        attention_bias_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
) -> TinyV3ForwardTrace:
    """Run only layers ``[start_layer .. n_layers)`` given the
    residual stream at the input of layer ``start_layer``.

    Useful for hidden-state-bridge V2 experiments where we want to
    inject a perturbation at the input of one specific layer and
    re-run only the suffix.
    """
    cfg = params.config
    n_tokens = int(residual_at_layer_l.shape[0])
    x = residual_at_layer_l.astype(_np.float64).copy()
    hidden_states = [x.copy()]
    attn_weights: list["_np.ndarray"] = []
    importance: list["_np.ndarray"] = []
    flop_per_layer: list[int] = []
    if kv_cache is None:
        new_cache = TinyV3KVCache.empty(cfg.n_layers)
    else:
        new_cache = kv_cache.clone()
    biases = (
        list(attention_bias_per_layer)
        if attention_bias_per_layer is not None
        else [None] * cfg.n_layers)
    while len(biases) < cfg.n_layers:
        biases.append(None)
    total_flops = 0
    d_kv = (int(cfg.d_model) // int(cfg.n_heads)
            * int(cfg.n_kv_heads))
    for layer_idx in range(int(start_layer), int(cfg.n_layers)):
        layer = params.layers[layer_idx]
        x_norm = _norm_v3(x, layer.ln1)
        kv_k = new_cache.keys[layer_idx]
        kv_v = new_cache.values[layer_idx]
        if kv_k.size == 0:
            kv_k = _np.zeros((0, d_kv), dtype=_np.float64)
            kv_v = _np.zeros((0, d_kv), dtype=_np.float64)
        attn_out, k_all, v_all, attn, flops = (
            _attention_layer_forward_v3(
                x_norm, layer.attn,
                kv_keys_prev=kv_k,
                kv_values_prev=kv_v,
                positions_new=positions,
                rope_table=params.rope_table,
                use_rope=bool(cfg.use_rope),
                attention_bias=biases[layer_idx]))
        new_cache.keys[layer_idx] = k_all
        new_cache.values[layer_idx] = v_all
        attn_weights.append(attn)
        flop_per_layer.append(int(flops))
        total_flops += int(flops)
        T_all = int(attn.shape[-1])
        imp = (
            _np.einsum("hts->s", attn)
            if bool(cfg.track_kv_importance)
            else _np.zeros((T_all,), dtype=_np.float64))
        new_cache.importance[layer_idx] = imp.astype(_np.float64)
        importance.append(imp.copy())
        x = x + attn_out
        x_norm2 = _norm_v3(x, layer.ln2)
        if bool(layer.ff.use_swiglu):
            gate = _swish(x_norm2 @ layer.ff.w_gate)
            up = x_norm2 @ layer.ff.w_1 + layer.ff.b_1
            ff_h = gate * up
            ff_out = ff_h @ layer.ff.w_2 + layer.ff.b_2
        else:
            ff_h = _gelu(
                x_norm2 @ layer.ff.w_1 + layer.ff.b_1)
            ff_out = ff_h @ layer.ff.w_2 + layer.ff.b_2
        x = x + ff_out
        ff_flops = (
            2 * n_tokens * int(cfg.d_model) * int(cfg.ff_hidden))
        flop_per_layer[-1] += int(ff_flops)
        total_flops += int(ff_flops)
        hidden_states.append(x.copy())
    final = _norm_v3(x, params.ln_f)
    logits = final @ params.unembed
    per_layer_lens: list["_np.ndarray"] = []
    for hs in hidden_states:
        per_layer_lens.append(hs @ params.unembed)
    return TinyV3ForwardTrace(
        logits=logits,
        hidden_states=hidden_states,
        attn_weights_per_layer=attn_weights,
        kv_cache=new_cache,
        per_layer_logit_lens=per_layer_lens,
        kv_importance_per_layer=importance,
        flop_count=int(total_flops),
        flop_count_per_layer=list(flop_per_layer),
        token_ids=tuple(),
        config_cid=cfg.cid(),
        params_cid=params.cid(),
    )


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def tokenize_bytes_v3(
        text: str, *,
        max_len: int = W58_DEFAULT_V3_MAX_LEN,
        add_bos: bool = True,
) -> list[int]:
    raw = text.encode("utf-8")
    ids: list[int] = []
    if add_bos:
        ids.append(int(W58_TINY_V3_BOS_TOKEN))
    ids.extend(int(b) for b in raw)
    if len(ids) > int(max_len):
        ids = ids[: int(max_len)]
    return ids


def detokenize_bytes_v3(token_ids: Sequence[int]) -> str:
    buf = bytearray()
    for t in token_ids:
        ti = int(t)
        if ti == W58_TINY_V3_PAD_TOKEN:
            continue
        if ti == W58_TINY_V3_BOS_TOKEN:
            continue
        if ti == W58_TINY_V3_EOS_TOKEN:
            break
        if 0 <= ti < 256:
            buf.append(ti)
    try:
        return buf.decode("utf-8", errors="replace")
    except UnicodeDecodeError:
        return buf.decode("latin-1")


# ---------------------------------------------------------------------------
# Witness
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TinyV3SubstrateForwardWitness:
    config_cid: str
    params_cid: str
    token_ids: tuple[int, ...]
    hidden_state_cid: str
    logits_cid: str
    attention_cid: str
    logit_lens_cid: str
    importance_cid: str
    kv_cache_cid: str
    forward_trace_cid: str
    flop_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W58_TINY_SUBSTRATE_V3_SCHEMA_VERSION,
            "config_cid": str(self.config_cid),
            "params_cid": str(self.params_cid),
            "token_ids": list(self.token_ids),
            "hidden_state_cid": str(self.hidden_state_cid),
            "logits_cid": str(self.logits_cid),
            "attention_cid": str(self.attention_cid),
            "logit_lens_cid": str(self.logit_lens_cid),
            "importance_cid": str(self.importance_cid),
            "kv_cache_cid": str(self.kv_cache_cid),
            "forward_trace_cid": str(self.forward_trace_cid),
            "flop_count": int(self.flop_count),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v3_substrate_forward_witness",
            "witness": self.to_dict(),
        })


def emit_tiny_substrate_v3_forward_witness(
        trace: TinyV3ForwardTrace,
) -> TinyV3SubstrateForwardWitness:
    return TinyV3SubstrateForwardWitness(
        config_cid=str(trace.config_cid),
        params_cid=str(trace.params_cid),
        token_ids=tuple(trace.token_ids),
        hidden_state_cid=str(trace.hidden_state_cid()),
        logits_cid=str(trace.logits_cid()),
        attention_cid=str(trace.attention_cid()),
        logit_lens_cid=str(trace.per_layer_logit_lens_cid()),
        importance_cid=str(trace.importance_cid()),
        kv_cache_cid=str(trace.kv_cache.cid()),
        forward_trace_cid=str(trace.cid()),
        flop_count=int(trace.flop_count),
    )


def build_default_tiny_substrate_v3(
        *, seed: int = W58_DEFAULT_V3_SEED,
) -> TinyV3SubstrateParams:
    return TinyV3SubstrateParams.init(
        TinyV3SubstrateConfig(seed=int(seed)))


__all__ = [
    "W58_TINY_SUBSTRATE_V3_SCHEMA_VERSION",
    "W58_TINY_V3_VOCAB_SIZE",
    "W58_TINY_V3_PAD_TOKEN",
    "W58_TINY_V3_BOS_TOKEN",
    "W58_TINY_V3_EOS_TOKEN",
    "W58_DEFAULT_V3_D_MODEL",
    "W58_DEFAULT_V3_N_HEADS",
    "W58_DEFAULT_V3_N_KV_HEADS",
    "W58_DEFAULT_V3_N_LAYERS",
    "W58_DEFAULT_V3_FF_HIDDEN",
    "W58_DEFAULT_V3_MAX_LEN",
    "W58_DEFAULT_V3_INIT_SCALE",
    "W58_DEFAULT_V3_SEED",
    "W58_DEFAULT_V3_ROPE_BASE",
    "TinyV3AttentionParams",
    "TinyV3FFParams",
    "TinyV3LayerNormParams",
    "TinyV3LayerParams",
    "TinyV3SubstrateConfig",
    "TinyV3SubstrateParams",
    "TinyV3KVCache",
    "TinyV3PrefixState",
    "TinyV3ForwardTrace",
    "TinyV3SubstrateForwardWitness",
    "forward_tiny_substrate_v3",
    "partial_forward_layers_l_to_end_v3",
    "extract_prefix_state_v3",
    "tokenize_bytes_v3",
    "detokenize_bytes_v3",
    "emit_tiny_substrate_v3_forward_witness",
    "build_default_tiny_substrate_v3",
]
