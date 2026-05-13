"""W57 M1 — Tiny Transformer Runtime V2.

Deeper, richer than W56's ``coordpy.tiny_substrate``. The W57
substrate program treats the in-repo transformer as the
load-bearing object. W56 proved a partial substrate breach via a
2-layer / 4-head / d_model=32 NumPy runtime. W57 pushes the
substrate further:

* **4 layers / 8 heads / d_model=64** by default (vs W56's 2/4/32).
  Per-step cost stays ``O(L·H·n_tokens·d_head)`` in NumPy on CPU;
  the runtime is still bounded, still inspectable, still
  research-scale, but materially richer.
* **Real RoPE-style rotary positional embeddings** alongside
  W56's learned positional table. The substrate exposes both
  ``pos_embed`` (learned) and ``rope_freqs`` (rotary). The forward
  path uses the rotary on Q/K *before* dot-product, matching the
  modern transformer convention. The CID covers both.
* **Logit lens** — ``logit_lens`` reads the unembedded logits at
  any intermediate layer (not just the final). Useful for the
  substrate-conditioned consensus tiebreaker and for hidden-state
  bridge measurements.
* **Per-head attention bias hook** — every per-layer attention
  call accepts an optional ``(n_heads, T_new, T_all)`` bias tensor
  added pre-softmax. This is what the new attention-steering
  bridge writes through; the substrate honours it byte-for-byte.
* **Cache-eviction policy** — the KV cache supports two extra
  operations:

    - ``evict_lru(k)`` drops the ``k`` oldest tokens from every
      layer's cache;
    - ``evict_weighted(weights, keep)`` keeps the top-``keep``
      tokens by a caller-supplied importance vector.

  Eviction emits a content-addressed write_log entry so the
  resulting cache CID is replay-deterministic.
* **Prefix-state extraction** — a forward pass can return a
  packed *prefix state* object (per-layer K, V tensors trimmed
  to a prefix length) that can be saved, hashed, and re-fed into
  a future forward as ``kv_cache=...``. The prefix state is its
  own first-class object with a CID.
* **Multi-layer hidden-state extraction** — the forward trace
  reports per-layer hidden states (W56 did too), but V2 adds a
  ``hidden_state_at_layer(l, pos)`` helper plus a
  ``per_layer_logit_lens`` cache so the rest of the W57 stack
  doesn't have to recompute.

Honest scope (do-not-overstate)
-------------------------------

* This is still NOT a frontier model. It is a richer research
  runtime. ``W57-L-NUMPY-CPU-V2-SUBSTRATE-CAP`` documents this.
* This still does NOT bridge to third-party hosted models.
  ``W57-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward
  the W56 cap unchanged: Ollama / OpenAI / hosted APIs remain
  text-only at the HTTP surface.
* The default config is still untrained. A finite-difference
  fitter exists (W56's ``fit_substrate_next_token``); V2 inherits
  the same toy training surface but does not expand it.
* RoPE here is the standard formulation
  (``rotate_half`` over even/odd halves of each head dim), not a
  novel variant.

What it gets you
----------------

* All W56 substrate uses still work — V2 strictly extends V1.
* A richer surface for the W57 bridges:

    - ``kv_bridge_v2`` writes into a deeper, multi-head, RoPE-aware
      cache;
    - ``hidden_state_bridge`` reads / writes per-layer hidden
      states at any layer;
    - ``prefix_state_bridge`` saves / reuses prefix states across
      turns;
    - ``attention_steering_bridge`` writes attention bias tensors
      that the substrate honours.
* A logit lens that the consensus controller V3 uses as a
  substrate-logit tiebreaker oracle.
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
        "coordpy.tiny_substrate_v2 requires numpy") from exc


W57_TINY_SUBSTRATE_V2_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v2.v1")

W57_TINY_V2_VOCAB_SIZE: int = 259  # same byte-level vocab as W56
W57_TINY_V2_PAD_TOKEN: int = 256
W57_TINY_V2_BOS_TOKEN: int = 257
W57_TINY_V2_EOS_TOKEN: int = 258

W57_DEFAULT_V2_D_MODEL: int = 64
W57_DEFAULT_V2_N_HEADS: int = 8
W57_DEFAULT_V2_N_LAYERS: int = 4
W57_DEFAULT_V2_FF_HIDDEN: int = 128
W57_DEFAULT_V2_MAX_LEN: int = 96
W57_DEFAULT_V2_INIT_SCALE: float = 0.04
W57_DEFAULT_V2_SEED: int = 57012345
W57_DEFAULT_V2_ROPE_BASE: float = 10000.0


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


# =============================================================================
# RoPE
# =============================================================================


def _rope_freqs(
        d_head: int, max_len: int,
        *, base: float = W57_DEFAULT_V2_ROPE_BASE,
) -> "_np.ndarray":
    """Real rotary position frequencies.

    Returns ``(max_len, d_head)`` of cos/sin pairs flattened over
    even/odd indices: positions ``2k`` carry ``cos(theta)`` and
    positions ``2k+1`` carry ``sin(theta)``.
    """
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


def _apply_rope(
        x: "_np.ndarray", positions: "_np.ndarray",
        rope_table: "_np.ndarray",
) -> "_np.ndarray":
    """Apply rotary embedding to a ``(H, T, D_h)`` tensor in-place
    semantics (returns a new array).

    ``positions`` is ``(T,)`` integer positions; ``rope_table`` is
    ``(max_len, D_h)`` with cos at even indices and sin at odd
    indices. We pair up consecutive dims (``d=2k`` and ``d=2k+1``)
    and rotate them.
    """
    h, t, dh = x.shape
    if dh < 2:
        return x.copy()
    rope = rope_table[positions]              # (T, D_h)
    cos = rope[:, 0::2]                       # (T, D_h/2)
    sin = rope[:, 1::2]                       # (T, D_h/2)
    x_even = x[:, :, 0::2]                    # (H, T, D_h/2)
    x_odd = x[:, :, 1::2]
    rot_even = x_even * cos[None, :, :] - x_odd * sin[None, :, :]
    rot_odd = x_odd * cos[None, :, :] + x_even * sin[None, :, :]
    out = _np.empty_like(x)
    out[:, :, 0::2] = rot_even
    out[:, :, 1::2] = rot_odd
    return out


# =============================================================================
# Parameters
# =============================================================================


@dataclasses.dataclass
class TinyV2AttentionParams:
    w_q: "_np.ndarray"
    w_k: "_np.ndarray"
    w_v: "_np.ndarray"
    w_o: "_np.ndarray"
    n_heads: int

    @classmethod
    def init(cls, *, d_model: int, n_heads: int,
             seed: int, init_scale: float,
             ) -> "TinyV2AttentionParams":
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} not divisible by n_heads={n_heads}")
        rng = _seeded_rng(seed)
        s = float(init_scale)
        return cls(
            w_q=rng.standard_normal((d_model, d_model)) * s,
            w_k=rng.standard_normal((d_model, d_model)) * s,
            w_v=rng.standard_normal((d_model, d_model)) * s,
            w_o=rng.standard_normal((d_model, d_model)) * s,
            n_heads=int(n_heads),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v2_attn_params",
            "w_q_cid": _ndarray_cid(self.w_q),
            "w_k_cid": _ndarray_cid(self.w_k),
            "w_v_cid": _ndarray_cid(self.w_v),
            "w_o_cid": _ndarray_cid(self.w_o),
            "n_heads": int(self.n_heads),
        })


@dataclasses.dataclass
class TinyV2FFParams:
    w_1: "_np.ndarray"
    b_1: "_np.ndarray"
    w_2: "_np.ndarray"
    b_2: "_np.ndarray"

    @classmethod
    def init(cls, *, d_model: int, ff_hidden: int,
             seed: int, init_scale: float,
             ) -> "TinyV2FFParams":
        rng = _seeded_rng(seed)
        s = float(init_scale)
        return cls(
            w_1=rng.standard_normal((d_model, ff_hidden)) * s,
            b_1=_np.zeros(ff_hidden, dtype=_np.float64),
            w_2=rng.standard_normal((ff_hidden, d_model)) * s,
            b_2=_np.zeros(d_model, dtype=_np.float64),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v2_ff_params",
            "w_1_cid": _ndarray_cid(self.w_1),
            "b_1_cid": _ndarray_cid(self.b_1),
            "w_2_cid": _ndarray_cid(self.w_2),
            "b_2_cid": _ndarray_cid(self.b_2),
        })


@dataclasses.dataclass
class TinyV2LayerNormParams:
    gamma: "_np.ndarray"
    beta: "_np.ndarray"

    @classmethod
    def init(cls, *, d_model: int) -> "TinyV2LayerNormParams":
        return cls(
            gamma=_np.ones(d_model, dtype=_np.float64),
            beta=_np.zeros(d_model, dtype=_np.float64),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v2_ln_params",
            "gamma_cid": _ndarray_cid(self.gamma),
            "beta_cid": _ndarray_cid(self.beta),
        })


@dataclasses.dataclass
class TinyV2LayerParams:
    attn: TinyV2AttentionParams
    ff: TinyV2FFParams
    ln1: TinyV2LayerNormParams
    ln2: TinyV2LayerNormParams

    @classmethod
    def init(cls, *, d_model: int, n_heads: int, ff_hidden: int,
             seed: int, init_scale: float,
             ) -> "TinyV2LayerParams":
        return cls(
            attn=TinyV2AttentionParams.init(
                d_model=d_model, n_heads=n_heads,
                seed=int(seed) + 1, init_scale=init_scale),
            ff=TinyV2FFParams.init(
                d_model=d_model, ff_hidden=ff_hidden,
                seed=int(seed) + 2, init_scale=init_scale),
            ln1=TinyV2LayerNormParams.init(d_model=d_model),
            ln2=TinyV2LayerNormParams.init(d_model=d_model),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v2_layer_params",
            "attn_cid": self.attn.cid(),
            "ff_cid": self.ff.cid(),
            "ln1_cid": self.ln1.cid(),
            "ln2_cid": self.ln2.cid(),
        })


@dataclasses.dataclass
class TinyV2SubstrateConfig:
    vocab_size: int = W57_TINY_V2_VOCAB_SIZE
    d_model: int = W57_DEFAULT_V2_D_MODEL
    n_heads: int = W57_DEFAULT_V2_N_HEADS
    n_layers: int = W57_DEFAULT_V2_N_LAYERS
    ff_hidden: int = W57_DEFAULT_V2_FF_HIDDEN
    max_len: int = W57_DEFAULT_V2_MAX_LEN
    init_scale: float = W57_DEFAULT_V2_INIT_SCALE
    seed: int = W57_DEFAULT_V2_SEED
    rope_base: float = W57_DEFAULT_V2_ROPE_BASE
    use_rope: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W57_TINY_SUBSTRATE_V2_SCHEMA_VERSION,
            "vocab_size": int(self.vocab_size),
            "d_model": int(self.d_model),
            "n_heads": int(self.n_heads),
            "n_layers": int(self.n_layers),
            "ff_hidden": int(self.ff_hidden),
            "max_len": int(self.max_len),
            "init_scale": float(self.init_scale),
            "seed": int(self.seed),
            "rope_base": float(self.rope_base),
            "use_rope": bool(self.use_rope),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v2_substrate_config",
            "config": self.to_dict()})


@dataclasses.dataclass
class TinyV2SubstrateParams:
    config: TinyV2SubstrateConfig
    embed: "_np.ndarray"
    pos_embed: "_np.ndarray"
    rope_table: "_np.ndarray"
    layers: list[TinyV2LayerParams]
    ln_f: TinyV2LayerNormParams
    unembed: "_np.ndarray"

    @classmethod
    def init(
            cls, config: TinyV2SubstrateConfig | None = None,
    ) -> "TinyV2SubstrateParams":
        if config is None:
            config = TinyV2SubstrateConfig()
        rng = _seeded_rng(config.seed)
        s = float(config.init_scale)
        embed = rng.standard_normal(
            (config.vocab_size, config.d_model)) * s
        pos_embed = rng.standard_normal(
            (config.max_len, config.d_model)) * s
        d_head = int(config.d_model) // int(config.n_heads)
        rope = _rope_freqs(d_head=int(d_head),
                           max_len=int(config.max_len),
                           base=float(config.rope_base))
        layers: list[TinyV2LayerParams] = []
        for i in range(config.n_layers):
            layers.append(TinyV2LayerParams.init(
                d_model=config.d_model,
                n_heads=config.n_heads,
                ff_hidden=config.ff_hidden,
                seed=int(config.seed) + 1000 * (i + 1),
                init_scale=s))
        ln_f = TinyV2LayerNormParams.init(d_model=config.d_model)
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
            "kind": "tiny_v2_substrate_params",
            "config_cid": self.config.cid(),
            "embed_cid": _ndarray_cid(self.embed),
            "pos_embed_cid": _ndarray_cid(self.pos_embed),
            "rope_cid": _ndarray_cid(self.rope_table),
            "layer_cids": [l.cid() for l in self.layers],
            "ln_f_cid": self.ln_f.cid(),
            "unembed_cid": _ndarray_cid(self.unembed),
        })


# =============================================================================
# KV cache V2 (with eviction)
# =============================================================================


@dataclasses.dataclass
class TinyV2KVCache:
    keys: list["_np.ndarray"]
    values: list["_np.ndarray"]
    write_log: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(cls, n_layers: int) -> "TinyV2KVCache":
        return cls(
            keys=[_np.zeros((0, 0), dtype=_np.float64)
                  for _ in range(int(n_layers))],
            values=[_np.zeros((0, 0), dtype=_np.float64)
                    for _ in range(int(n_layers))],
            write_log=[],
        )

    def n_tokens(self) -> int:
        if not self.keys:
            return 0
        return int(self.keys[0].shape[0])

    def clone(self) -> "TinyV2KVCache":
        return TinyV2KVCache(
            keys=[k.copy() for k in self.keys],
            values=[v.copy() for v in self.values],
            write_log=list(self.write_log),
        )

    def evict_lru(self, k: int) -> "TinyV2KVCache":
        """Drop the ``k`` oldest tokens from every layer."""
        if int(k) <= 0:
            return self.clone()
        out = self.clone()
        n = out.n_tokens()
        drop = min(int(k), int(n))
        for i in range(len(out.keys)):
            if out.keys[i].size and out.keys[i].shape[0] > drop:
                out.keys[i] = out.keys[i][drop:].copy()
                out.values[i] = out.values[i][drop:].copy()
            elif out.keys[i].size:
                out.keys[i] = out.keys[i][:0].copy()
                out.values[i] = out.values[i][:0].copy()
        out.write_log.append({
            "schema": W57_TINY_SUBSTRATE_V2_SCHEMA_VERSION,
            "kind": "evict_lru",
            "dropped": int(drop),
        })
        return out

    def evict_weighted(
            self, weights: Sequence[float], keep: int,
    ) -> "TinyV2KVCache":
        """Keep the top-``keep`` tokens by weight; drop the rest."""
        out = self.clone()
        n = out.n_tokens()
        if n == 0 or int(keep) >= n:
            return out
        w = list(weights)[:n]
        while len(w) < n:
            w.append(0.0)
        # Sort indices by weight desc; keep first ``keep``.
        order = sorted(range(n),
                        key=lambda i: -float(w[i]))
        keep_set = sorted(order[: int(keep)])
        idx = _np.asarray(keep_set, dtype=_np.int64)
        for i in range(len(out.keys)):
            if out.keys[i].size:
                out.keys[i] = out.keys[i][idx].copy()
                out.values[i] = out.values[i][idx].copy()
        out.write_log.append({
            "schema": W57_TINY_SUBSTRATE_V2_SCHEMA_VERSION,
            "kind": "evict_weighted",
            "kept": list(keep_set),
        })
        return out

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v2_kv_cache",
            "n_tokens": int(self.n_tokens()),
            "n_layers": int(len(self.keys)),
            "keys_cids": [_ndarray_cid(k) for k in self.keys],
            "values_cids": [_ndarray_cid(v) for v in self.values],
            "write_log": list(self.write_log),
        })


# =============================================================================
# Prefix state (extracted, replayable)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class TinyV2PrefixState:
    """A snapshot of a KV cache up to a prefix length.

    Useful for prefix-state-reuse experiments (W57 M4 bridge):
    the prefix state is a first-class, content-addressed object
    that can be saved and re-fed into a future forward.
    """

    prefix_len: int
    keys: tuple["_np.ndarray", ...]
    values: tuple["_np.ndarray", ...]
    source_params_cid: str

    def to_cache(self) -> TinyV2KVCache:
        return TinyV2KVCache(
            keys=[k.copy() for k in self.keys],
            values=[v.copy() for v in self.values],
            write_log=[{
                "schema": W57_TINY_SUBSTRATE_V2_SCHEMA_VERSION,
                "kind": "prefix_state_load",
                "prefix_len": int(self.prefix_len),
                "source_params_cid": str(self.source_params_cid),
            }],
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v2_prefix_state",
            "prefix_len": int(self.prefix_len),
            "keys_cids": [_ndarray_cid(k) for k in self.keys],
            "values_cids": [_ndarray_cid(v) for v in self.values],
            "source_params_cid": str(self.source_params_cid),
        })


def extract_prefix_state(
        cache: TinyV2KVCache, *,
        prefix_len: int,
        source_params_cid: str,
) -> TinyV2PrefixState:
    """Trim a cache to the first ``prefix_len`` tokens."""
    n = cache.n_tokens()
    L = min(int(prefix_len), int(n))
    keys = []
    values = []
    for i in range(len(cache.keys)):
        if cache.keys[i].size:
            keys.append(cache.keys[i][:L].copy())
            values.append(cache.values[i][:L].copy())
        else:
            keys.append(cache.keys[i].copy())
            values.append(cache.values[i].copy())
    return TinyV2PrefixState(
        prefix_len=int(L),
        keys=tuple(keys),
        values=tuple(values),
        source_params_cid=str(source_params_cid),
    )


# =============================================================================
# Numerics
# =============================================================================


def _gelu(x: "_np.ndarray") -> "_np.ndarray":
    return 0.5 * x * (1.0 + _np.tanh(
        math.sqrt(2.0 / math.pi)
        * (x + 0.044715 * (x ** 3))))


def _layer_norm(
        x: "_np.ndarray", params: TinyV2LayerNormParams,
        *, eps: float = 1e-5,
) -> "_np.ndarray":
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    y = (x - mu) / _np.sqrt(var + eps)
    return y * params.gamma + params.beta


def _softmax(x: "_np.ndarray", axis: int = -1) -> "_np.ndarray":
    m = _np.max(x, axis=axis, keepdims=True)
    z = _np.exp(x - m)
    return z / _np.sum(z, axis=axis, keepdims=True)


def _split_heads(
        x: "_np.ndarray", n_heads: int,
) -> "_np.ndarray":
    n_tokens, d_model = x.shape
    d_head = d_model // n_heads
    return x.reshape(n_tokens, n_heads, d_head).transpose(1, 0, 2)


def _merge_heads(x: "_np.ndarray") -> "_np.ndarray":
    n_heads, n_tokens, d_head = x.shape
    return x.transpose(1, 0, 2).reshape(n_tokens, n_heads * d_head)


# =============================================================================
# Forward
# =============================================================================


@dataclasses.dataclass
class TinyV2ForwardTrace:
    logits: "_np.ndarray"
    hidden_states: list["_np.ndarray"]
    attn_weights_per_layer: list["_np.ndarray"]
    kv_cache: TinyV2KVCache
    per_layer_logit_lens: list["_np.ndarray"]
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
            "kind": "tiny_v2_forward_hidden",
            "config_cid": self.config_cid,
            "params_cid": self.params_cid,
            "token_ids": list(self.token_ids),
            "hidden_state_cids": [
                _ndarray_cid(h) for h in self.hidden_states],
        })

    def logits_cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v2_forward_logits",
            "logits_cid": _ndarray_cid(self.logits),
        })

    def attention_cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v2_forward_attn",
            "attn_weights_cids": [
                _ndarray_cid(a)
                for a in self.attn_weights_per_layer],
        })

    def per_layer_logit_lens_cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v2_forward_logit_lens",
            "lens_cids": [
                _ndarray_cid(l)
                for l in self.per_layer_logit_lens],
        })

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v2_forward_trace",
            "hidden_state_cid": self.hidden_state_cid(),
            "logits_cid": self.logits_cid(),
            "attention_cid": self.attention_cid(),
            "logit_lens_cid": self.per_layer_logit_lens_cid(),
            "kv_cache_cid": self.kv_cache.cid(),
            "token_ids": list(self.token_ids),
        })


def _attention_layer_forward_v2(
        x: "_np.ndarray",
        params: TinyV2AttentionParams,
        *,
        kv_keys_prev: "_np.ndarray",
        kv_values_prev: "_np.ndarray",
        positions_new: "_np.ndarray",
        rope_table: "_np.ndarray",
        use_rope: bool,
        attention_bias: "_np.ndarray | None",
) -> tuple["_np.ndarray", "_np.ndarray",
           "_np.ndarray", "_np.ndarray"]:
    """Multi-head causal self-attention with KV cache + RoPE +
    optional pre-softmax attention bias.

    The bias, if supplied, is shape
    ``(n_heads, n_tokens_new, n_prev + n_tokens_new)`` and is
    added to the scaled scores BEFORE the causal mask + softmax.
    The attention-steering bridge writes through this hook.
    """
    n_tokens = int(x.shape[0])
    d_model = int(x.shape[1])
    n_heads = int(params.n_heads)
    d_head = d_model // n_heads

    q = x @ params.w_q
    k_new = x @ params.w_k
    v_new = x @ params.w_v

    q_h = _split_heads(q, n_heads)
    k_new_h = _split_heads(k_new, n_heads)
    v_new_h = _split_heads(v_new, n_heads)

    if use_rope:
        q_h = _apply_rope(q_h, positions_new, rope_table)
        k_new_h = _apply_rope(k_new_h, positions_new, rope_table)

    # Concatenate the rotary-prepared new K with the existing
    # cache. Note: cache stores keys already RoPE-applied for
    # their source positions, so concatenation is causal-correct.
    if kv_keys_prev.size == 0:
        k_all_h = k_new_h
        v_all_h = v_new_h
    else:
        prev_k_h = _split_heads(kv_keys_prev, n_heads)
        prev_v_h = _split_heads(kv_values_prev, n_heads)
        k_all_h = _np.concatenate(
            [prev_k_h, k_new_h], axis=1)
        v_all_h = _np.concatenate(
            [prev_v_h, v_new_h], axis=1)

    scores = _np.einsum(
        "htd,hsd->hts", q_h, k_all_h) / math.sqrt(float(d_head))

    if attention_bias is not None:
        scores = scores + _np.asarray(attention_bias,
                                       dtype=_np.float64)

    n_prev = (int(kv_keys_prev.shape[0])
              if kv_keys_prev.size else 0)
    mask = _np.full(
        (n_tokens, n_prev + n_tokens), -1e9, dtype=_np.float64)
    for i in range(n_tokens):
        mask[i, : n_prev + i + 1] = 0.0
    scores = scores + mask[None, :, :]

    attn = _softmax(scores, axis=-1)
    out_h = _np.einsum("hts,hsd->htd", attn, v_all_h)
    out = _merge_heads(out_h) @ params.w_o
    # Re-merge K_all and V_all back to (T_all, d_model) for cache
    # storage.
    k_all_flat = _merge_heads(k_all_h)
    v_all_flat = _merge_heads(v_all_h)
    return out, k_all_flat, v_all_flat, attn


def forward_tiny_substrate_v2(
        params: TinyV2SubstrateParams,
        token_ids: Sequence[int],
        *,
        kv_cache: TinyV2KVCache | None = None,
        return_attention: bool = True,
        attention_bias_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
) -> TinyV2ForwardTrace:
    """Real forward over the V2 substrate.

    Computes embeddings → per-layer (LN1 → attention (RoPE) →
    residual → LN2 → FF → residual) → final LN → unembedding.
    Also computes a per-layer logit lens (unembed of each layer's
    post-residual hidden state, after the final LN gamma/beta is
    NOT applied — that's intentional: the lens is "what would this
    layer say if we stopped here").

    ``attention_bias_per_layer`` is an optional per-layer list of
    ``(n_heads, n_tokens_new, n_prev + n_tokens_new)`` tensors
    added pre-softmax. ``None`` entries skip biasing for that
    layer.
    """
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
    if kv_cache is None:
        new_cache = TinyV2KVCache.empty(cfg.n_layers)
    else:
        new_cache = kv_cache.clone()
    if len(new_cache.keys) != cfg.n_layers:
        new_cache = TinyV2KVCache.empty(cfg.n_layers)
    biases = attention_bias_per_layer or [None] * cfg.n_layers
    while len(biases) < cfg.n_layers:
        biases = list(biases) + [None]
    for layer_idx, layer in enumerate(params.layers):
        x_norm = _layer_norm(x, layer.ln1)
        kv_k = new_cache.keys[layer_idx]
        kv_v = new_cache.values[layer_idx]
        if kv_k.size == 0:
            kv_k = _np.zeros((0, cfg.d_model), dtype=_np.float64)
            kv_v = _np.zeros((0, cfg.d_model), dtype=_np.float64)
        attn_out, k_all, v_all, attn = _attention_layer_forward_v2(
            x_norm, layer.attn,
            kv_keys_prev=kv_k,
            kv_values_prev=kv_v,
            positions_new=pos,
            rope_table=params.rope_table,
            use_rope=bool(cfg.use_rope),
            attention_bias=biases[layer_idx])
        new_cache.keys[layer_idx] = k_all
        new_cache.values[layer_idx] = v_all
        attn_weights.append(attn)
        x = x + attn_out
        x_norm2 = _layer_norm(x, layer.ln2)
        ff_h = _gelu(x_norm2 @ layer.ff.w_1 + layer.ff.b_1)
        ff_out = ff_h @ layer.ff.w_2 + layer.ff.b_2
        x = x + ff_out
        hidden_states.append(x.copy())
    final = _layer_norm(x, params.ln_f)
    logits = final @ params.unembed
    # Per-layer logit lens: each layer's hidden state, unembedded.
    per_layer_lens: list["_np.ndarray"] = []
    for hs in hidden_states:
        per_layer_lens.append(hs @ params.unembed)
    return TinyV2ForwardTrace(
        logits=logits,
        hidden_states=hidden_states,
        attn_weights_per_layer=attn_weights,
        kv_cache=new_cache,
        per_layer_logit_lens=per_layer_lens,
        token_ids=tuple(int(t) for t in token_ids),
        config_cid=cfg.cid(),
        params_cid=params.cid(),
    )


# =============================================================================
# Decode
# =============================================================================


def decode_greedy_tiny_substrate_v2(
        params: TinyV2SubstrateParams,
        prompt_token_ids: Sequence[int],
        *,
        max_new_tokens: int = 8,
        kv_cache: TinyV2KVCache | None = None,
        stop_token: int = W57_TINY_V2_EOS_TOKEN,
) -> tuple[list[int], TinyV2ForwardTrace]:
    cfg = params.config
    cur_cache = (kv_cache.clone() if kv_cache is not None
                 else TinyV2KVCache.empty(cfg.n_layers))
    trace = forward_tiny_substrate_v2(
        params, list(prompt_token_ids), kv_cache=cur_cache,
        return_attention=False)
    cur_cache = trace.kv_cache
    generated: list[int] = []
    for _ in range(int(max_new_tokens)):
        nxt = int(_np.argmax(trace.logits[-1]))
        generated.append(int(nxt))
        if nxt == int(stop_token):
            break
        trace = forward_tiny_substrate_v2(
            params, [int(nxt)], kv_cache=cur_cache,
            return_attention=False)
        cur_cache = trace.kv_cache
    return generated, trace


# =============================================================================
# Tokenisation (same as W56 byte-level)
# =============================================================================


def tokenize_bytes_v2(
        text: str, *,
        max_len: int = W57_DEFAULT_V2_MAX_LEN,
        add_bos: bool = True,
) -> list[int]:
    raw = text.encode("utf-8")
    ids: list[int] = []
    if add_bos:
        ids.append(int(W57_TINY_V2_BOS_TOKEN))
    ids.extend(int(b) for b in raw)
    if len(ids) > int(max_len):
        ids = ids[: int(max_len)]
    return ids


def detokenize_bytes_v2(token_ids: Sequence[int]) -> str:
    buf = bytearray()
    for t in token_ids:
        ti = int(t)
        if ti == W57_TINY_V2_PAD_TOKEN:
            continue
        if ti == W57_TINY_V2_BOS_TOKEN:
            continue
        if ti == W57_TINY_V2_EOS_TOKEN:
            break
        if 0 <= ti < 256:
            buf.append(ti)
    try:
        return buf.decode("utf-8", errors="replace")
    except UnicodeDecodeError:
        return buf.decode("latin-1")


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class TinyV2SubstrateForwardWitness:
    config_cid: str
    params_cid: str
    token_ids: tuple[int, ...]
    hidden_state_cid: str
    logits_cid: str
    attention_cid: str
    logit_lens_cid: str
    kv_cache_cid: str
    forward_trace_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W57_TINY_SUBSTRATE_V2_SCHEMA_VERSION,
            "config_cid": str(self.config_cid),
            "params_cid": str(self.params_cid),
            "token_ids": list(self.token_ids),
            "hidden_state_cid": str(self.hidden_state_cid),
            "logits_cid": str(self.logits_cid),
            "attention_cid": str(self.attention_cid),
            "logit_lens_cid": str(self.logit_lens_cid),
            "kv_cache_cid": str(self.kv_cache_cid),
            "forward_trace_cid": str(self.forward_trace_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v2_substrate_forward_witness",
            "witness": self.to_dict(),
        })


def emit_tiny_substrate_v2_forward_witness(
        trace: TinyV2ForwardTrace,
) -> TinyV2SubstrateForwardWitness:
    return TinyV2SubstrateForwardWitness(
        config_cid=str(trace.config_cid),
        params_cid=str(trace.params_cid),
        token_ids=tuple(trace.token_ids),
        hidden_state_cid=str(trace.hidden_state_cid()),
        logits_cid=str(trace.logits_cid()),
        attention_cid=str(trace.attention_cid()),
        logit_lens_cid=str(trace.per_layer_logit_lens_cid()),
        kv_cache_cid=str(trace.kv_cache.cid()),
        forward_trace_cid=str(trace.cid()),
    )


def build_default_tiny_substrate_v2(
        *, seed: int = W57_DEFAULT_V2_SEED,
) -> TinyV2SubstrateParams:
    return TinyV2SubstrateParams.init(
        TinyV2SubstrateConfig(seed=int(seed)))


__all__ = [
    "W57_TINY_SUBSTRATE_V2_SCHEMA_VERSION",
    "W57_TINY_V2_VOCAB_SIZE",
    "W57_TINY_V2_PAD_TOKEN",
    "W57_TINY_V2_BOS_TOKEN",
    "W57_TINY_V2_EOS_TOKEN",
    "W57_DEFAULT_V2_D_MODEL",
    "W57_DEFAULT_V2_N_HEADS",
    "W57_DEFAULT_V2_N_LAYERS",
    "W57_DEFAULT_V2_FF_HIDDEN",
    "W57_DEFAULT_V2_MAX_LEN",
    "W57_DEFAULT_V2_INIT_SCALE",
    "W57_DEFAULT_V2_SEED",
    "W57_DEFAULT_V2_ROPE_BASE",
    "TinyV2AttentionParams",
    "TinyV2FFParams",
    "TinyV2LayerNormParams",
    "TinyV2LayerParams",
    "TinyV2SubstrateConfig",
    "TinyV2SubstrateParams",
    "TinyV2KVCache",
    "TinyV2PrefixState",
    "TinyV2ForwardTrace",
    "TinyV2SubstrateForwardWitness",
    "forward_tiny_substrate_v2",
    "decode_greedy_tiny_substrate_v2",
    "extract_prefix_state",
    "tokenize_bytes_v2",
    "detokenize_bytes_v2",
    "emit_tiny_substrate_v2_forward_witness",
    "build_default_tiny_substrate_v2",
]
