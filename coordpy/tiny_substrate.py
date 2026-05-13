"""W56 M1 — Tiny in-repo transformer substrate.

This is the load-bearing piece of the W56 substrate-attack
milestone. Prior research milestones (W43..W55) operated entirely
at the **capsule layer**: hidden states, KV caches, attention
weights, and logits were *proxies* — algebraic interfaces that
reproduced the shape of a transformer block without touching real
transformer internals.

W56 changes that, honestly and within a bounded scope. This
module provides a real, executable, deterministic small-vocab
small-depth transformer where:

* token embeddings are real (a real ``(vocab, d_model)`` matrix);
* multi-head causal self-attention is real
  (``softmax(QK^T / sqrt(d_head))·V`` with a strict upper-triangle
  causal mask);
* the per-layer KV cache is real (numpy arrays you can read,
  modify, and re-attend over on the next token);
* hidden states are real per-layer activation tensors;
* layer norm, position-wise feed-forward, residual stream, and
  the unembedding head are real;
* logits are real and reproducible byte-for-byte under a fixed
  seed.

The substrate is implemented in pure NumPy (the only required
dependency of ``coordpy``). No PyTorch, no JAX, no MLX, no GPU.
Default config is *tiny* — 2 layers, 4 heads, ``d_model=32``,
byte-level vocab (259 entries: 256 byte values + 3 control
tokens). This is deliberate: it is small enough to inspect and
prove, but real enough that KV bytes, hidden state tensors, and
attention weights are *not metaphorical*.

Honest scope (do-not-overstate)
-------------------------------

* This is NOT a frontier model. It is a small research runtime.
* This does NOT prove anything about third-party hosted models.
  ``W56-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` documents that
  Ollama / OpenAI-compatible / hosted backends cannot expose
  hidden states / KV / attention weights through their HTTP
  surface; the substrate-attack within this milestone is bounded
  to the in-repo runtime.
* This is NOT GPU-accelerated. ``W56-L-NUMPY-CPU-TINY-SUBSTRATE-CAP``
  documents the per-step cost as ``O(L · H · n_tokens · d_head)``
  in pure NumPy.
* The substrate is NOT trained on any real-world corpus by
  default. Initial weights are deterministic-seeded random; the
  point is to provide a *real substrate object* that the W56
  latent operating system can bridge into, not to compete on
  perplexity. A small ``fit_substrate_next_token`` helper is
  available for in-test toy training when the benchmark requires
  it.

What it gets you
----------------

* A KV cache you can write to (M3 KV bridge).
* Hidden states you can read at merge time (M4 V8 persistent
  state, M6 MLSC V4 substrate_witness).
* Attention weights you can inspect (M9 deep substrate hybrid).
* A forward call you can use as a tiebreaker oracle (M7
  consensus controller V2 substrate-conditioned stage).
* A reproducible "did the carrier change the output?" probe
  (M3 KV bridge, M12 substrate_replay arbiter arm).

The honest framing of W56 in one sentence: this is the first
Context Zero / CoordPy milestone where **somewhere in the loop,
real transformer attention is running over real KV bytes** — even
if that attention runs on a tiny in-repo model rather than a
frontier-scale hosted model.
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
        "coordpy.tiny_substrate requires numpy") from exc


# =============================================================================
# Schema, defaults
# =============================================================================

W56_TINY_SUBSTRATE_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate.v1")

# Byte-level vocab + a few control tokens.
W56_TINY_SUBSTRATE_VOCAB_SIZE: int = 259
W56_TINY_SUBSTRATE_PAD_TOKEN: int = 256
W56_TINY_SUBSTRATE_BOS_TOKEN: int = 257
W56_TINY_SUBSTRATE_EOS_TOKEN: int = 258

W56_DEFAULT_TINY_D_MODEL: int = 32
W56_DEFAULT_TINY_N_HEADS: int = 4
W56_DEFAULT_TINY_N_LAYERS: int = 2
W56_DEFAULT_TINY_FF_HIDDEN: int = 64
W56_DEFAULT_TINY_MAX_LEN: int = 64
W56_DEFAULT_TINY_INIT_SCALE: float = 0.05
W56_DEFAULT_TINY_SEED: int = 56012345


# =============================================================================
# Helpers
# =============================================================================


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _ndarray_cid(arr: "_np.ndarray") -> str:
    """Content-addressed hash of a numpy array's bytes + shape + dtype.

    The hash is computed over the array's canonical IEEE-754
    bytes (``arr.tobytes()``) plus its shape and dtype string. So
    two arrays with identical contents under different views or
    strides hash identically iff they round-trip identically
    through ``np.ascontiguousarray(arr).astype(arr.dtype)``.
    """
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
# Tiny substrate parameters
# =============================================================================


@dataclasses.dataclass
class TinyAttentionParams:
    """Per-layer multi-head attention parameters.

    Real ``W_Q``, ``W_K``, ``W_V``, ``W_O`` matrices — not
    placeholders. ``d_model`` is split across ``n_heads`` so
    ``d_head = d_model / n_heads`` (must divide evenly).
    """

    w_q: "_np.ndarray"  # (d_model, d_model)
    w_k: "_np.ndarray"  # (d_model, d_model)
    w_v: "_np.ndarray"  # (d_model, d_model)
    w_o: "_np.ndarray"  # (d_model, d_model)
    n_heads: int

    @classmethod
    def init(
            cls, *, d_model: int, n_heads: int,
            seed: int, init_scale: float,
    ) -> "TinyAttentionParams":
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} not divisible by n_heads={n_heads}")
        rng = _seeded_rng(seed)
        scale = float(init_scale)
        return cls(
            w_q=rng.standard_normal((d_model, d_model)) * scale,
            w_k=rng.standard_normal((d_model, d_model)) * scale,
            w_v=rng.standard_normal((d_model, d_model)) * scale,
            w_o=rng.standard_normal((d_model, d_model)) * scale,
            n_heads=int(n_heads),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_attn_params",
            "w_q_cid": _ndarray_cid(self.w_q),
            "w_k_cid": _ndarray_cid(self.w_k),
            "w_v_cid": _ndarray_cid(self.w_v),
            "w_o_cid": _ndarray_cid(self.w_o),
            "n_heads": int(self.n_heads),
        })


@dataclasses.dataclass
class TinyFFParams:
    """Per-layer position-wise feed-forward parameters.

    ``h = gelu(x · W1 + b1) · W2 + b2``. Real matrices.
    """

    w_1: "_np.ndarray"
    b_1: "_np.ndarray"
    w_2: "_np.ndarray"
    b_2: "_np.ndarray"

    @classmethod
    def init(
            cls, *, d_model: int, ff_hidden: int,
            seed: int, init_scale: float,
    ) -> "TinyFFParams":
        rng = _seeded_rng(seed)
        scale = float(init_scale)
        return cls(
            w_1=rng.standard_normal((d_model, ff_hidden)) * scale,
            b_1=_np.zeros(ff_hidden, dtype=_np.float64),
            w_2=rng.standard_normal((ff_hidden, d_model)) * scale,
            b_2=_np.zeros(d_model, dtype=_np.float64),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_ff_params",
            "w_1_cid": _ndarray_cid(self.w_1),
            "b_1_cid": _ndarray_cid(self.b_1),
            "w_2_cid": _ndarray_cid(self.w_2),
            "b_2_cid": _ndarray_cid(self.b_2),
        })


@dataclasses.dataclass
class TinyLayerNormParams:
    """Per-layer LN gamma + beta."""

    gamma: "_np.ndarray"
    beta: "_np.ndarray"

    @classmethod
    def init(cls, *, d_model: int) -> "TinyLayerNormParams":
        return cls(
            gamma=_np.ones(d_model, dtype=_np.float64),
            beta=_np.zeros(d_model, dtype=_np.float64),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_ln_params",
            "gamma_cid": _ndarray_cid(self.gamma),
            "beta_cid": _ndarray_cid(self.beta),
        })


@dataclasses.dataclass
class TinyLayerParams:
    """One transformer block (attention + FF + 2 LNs)."""

    attn: TinyAttentionParams
    ff: TinyFFParams
    ln1: TinyLayerNormParams
    ln2: TinyLayerNormParams

    @classmethod
    def init(
            cls, *, d_model: int, n_heads: int, ff_hidden: int,
            seed: int, init_scale: float,
    ) -> "TinyLayerParams":
        return cls(
            attn=TinyAttentionParams.init(
                d_model=d_model, n_heads=n_heads,
                seed=int(seed) + 1, init_scale=init_scale),
            ff=TinyFFParams.init(
                d_model=d_model, ff_hidden=ff_hidden,
                seed=int(seed) + 2, init_scale=init_scale),
            ln1=TinyLayerNormParams.init(d_model=d_model),
            ln2=TinyLayerNormParams.init(d_model=d_model),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_layer_params",
            "attn_cid": self.attn.cid(),
            "ff_cid": self.ff.cid(),
            "ln1_cid": self.ln1.cid(),
            "ln2_cid": self.ln2.cid(),
        })


@dataclasses.dataclass
class TinySubstrateConfig:
    """Tiny substrate model config."""

    vocab_size: int = W56_TINY_SUBSTRATE_VOCAB_SIZE
    d_model: int = W56_DEFAULT_TINY_D_MODEL
    n_heads: int = W56_DEFAULT_TINY_N_HEADS
    n_layers: int = W56_DEFAULT_TINY_N_LAYERS
    ff_hidden: int = W56_DEFAULT_TINY_FF_HIDDEN
    max_len: int = W56_DEFAULT_TINY_MAX_LEN
    init_scale: float = W56_DEFAULT_TINY_INIT_SCALE
    seed: int = W56_DEFAULT_TINY_SEED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W56_TINY_SUBSTRATE_SCHEMA_VERSION,
            "vocab_size": int(self.vocab_size),
            "d_model": int(self.d_model),
            "n_heads": int(self.n_heads),
            "n_layers": int(self.n_layers),
            "ff_hidden": int(self.ff_hidden),
            "max_len": int(self.max_len),
            "init_scale": float(self.init_scale),
            "seed": int(self.seed),
        }

    def cid(self) -> str:
        return _sha256_hex({"kind": "tiny_substrate_config",
                            "config": self.to_dict()})


@dataclasses.dataclass
class TinySubstrateParams:
    """Full transformer parameters: embed, layers, ln_f, unembed."""

    config: TinySubstrateConfig
    embed: "_np.ndarray"          # (vocab, d_model)
    pos_embed: "_np.ndarray"      # (max_len, d_model)
    layers: list[TinyLayerParams]
    ln_f: TinyLayerNormParams
    unembed: "_np.ndarray"        # (d_model, vocab)

    @classmethod
    def init(
            cls, config: TinySubstrateConfig | None = None,
    ) -> "TinySubstrateParams":
        if config is None:
            config = TinySubstrateConfig()
        rng = _seeded_rng(config.seed)
        scale = float(config.init_scale)
        embed = rng.standard_normal(
            (config.vocab_size, config.d_model)) * scale
        pos_embed = rng.standard_normal(
            (config.max_len, config.d_model)) * scale
        layers: list[TinyLayerParams] = []
        for i in range(config.n_layers):
            layers.append(TinyLayerParams.init(
                d_model=config.d_model,
                n_heads=config.n_heads,
                ff_hidden=config.ff_hidden,
                seed=int(config.seed) + 1000 * (i + 1),
                init_scale=scale))
        ln_f = TinyLayerNormParams.init(d_model=config.d_model)
        unembed = rng.standard_normal(
            (config.d_model, config.vocab_size)) * scale
        return cls(
            config=config,
            embed=embed,
            pos_embed=pos_embed,
            layers=layers,
            ln_f=ln_f,
            unembed=unembed,
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_substrate_params",
            "config_cid": self.config.cid(),
            "embed_cid": _ndarray_cid(self.embed),
            "pos_embed_cid": _ndarray_cid(self.pos_embed),
            "layer_cids": [layer.cid() for layer in self.layers],
            "ln_f_cid": self.ln_f.cid(),
            "unembed_cid": _ndarray_cid(self.unembed),
        })


# =============================================================================
# KV cache
# =============================================================================


@dataclasses.dataclass
class TinyKVCache:
    """Per-layer KV cache.

    ``keys[l]`` and ``values[l]`` are ``(n_tokens, d_model)``
    numpy arrays. ``n_tokens`` grows by 1 per appended token. The
    KV bridge module reads from / writes to these arrays directly.

    Provenance: ``write_log`` records every write the bridge has
    performed (source CID + slot index + bytes_cid) so the bridge
    operation is itself content-addressed and replay-deterministic.
    """

    keys: list["_np.ndarray"]
    values: list["_np.ndarray"]
    write_log: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(cls, n_layers: int) -> "TinyKVCache":
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

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_kv_cache",
            "n_tokens": int(self.n_tokens()),
            "n_layers": int(len(self.keys)),
            "keys_cids": [_ndarray_cid(k) for k in self.keys],
            "values_cids": [_ndarray_cid(v) for v in self.values],
            "write_log": list(self.write_log),
        })

    def clone(self) -> "TinyKVCache":
        return TinyKVCache(
            keys=[k.copy() for k in self.keys],
            values=[v.copy() for v in self.values],
            write_log=list(self.write_log),
        )


# =============================================================================
# Numerics
# =============================================================================


def _gelu(x: "_np.ndarray") -> "_np.ndarray":
    return 0.5 * x * (1.0 + _np.tanh(
        math.sqrt(2.0 / math.pi)
        * (x + 0.044715 * (x ** 3))))


def _layer_norm(
        x: "_np.ndarray", params: TinyLayerNormParams,
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
    # x: (n_heads, n_tokens, d_head)
    n_heads, n_tokens, d_head = x.shape
    return x.transpose(1, 0, 2).reshape(n_tokens, n_heads * d_head)


# =============================================================================
# Forward / decode
# =============================================================================


@dataclasses.dataclass
class TinyForwardTrace:
    """Result of one forward pass.

    Holds:
      * ``logits`` — ``(n_tokens, vocab_size)``;
      * ``hidden_states`` — ``[n_layers + 1]`` of
        ``(n_tokens, d_model)``; ``[0]`` is the post-embedding
        residual stream, ``[i+1]`` is the post-layer-i residual;
      * ``attn_weights_per_layer`` — ``[n_layers]`` of
        ``(n_heads, n_tokens, n_tokens)``;
      * ``kv_cache`` — full ``TinyKVCache`` after the forward;
      * ``content_cids`` — a structured dict for verifier audit.
    """

    logits: "_np.ndarray"
    hidden_states: list["_np.ndarray"]
    attn_weights_per_layer: list["_np.ndarray"]
    kv_cache: TinyKVCache
    token_ids: tuple[int, ...]
    config_cid: str
    params_cid: str

    def hidden_state_cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_forward_hidden",
            "config_cid": self.config_cid,
            "params_cid": self.params_cid,
            "token_ids": list(self.token_ids),
            "hidden_state_cids": [
                _ndarray_cid(h) for h in self.hidden_states],
        })

    def logits_cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_forward_logits",
            "logits_cid": _ndarray_cid(self.logits),
        })

    def attention_cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_forward_attn",
            "attn_weights_cids": [
                _ndarray_cid(a)
                for a in self.attn_weights_per_layer],
        })

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_forward_trace",
            "hidden_state_cid": self.hidden_state_cid(),
            "logits_cid": self.logits_cid(),
            "attention_cid": self.attention_cid(),
            "kv_cache_cid": self.kv_cache.cid(),
            "token_ids": list(self.token_ids),
        })


def _attention_layer_forward(
        x: "_np.ndarray",
        params: TinyAttentionParams,
        *,
        kv_keys_prev: "_np.ndarray",
        kv_values_prev: "_np.ndarray",
        return_attention: bool = True,
) -> tuple["_np.ndarray", "_np.ndarray",
           "_np.ndarray", "_np.ndarray"]:
    """Multi-head causal self-attention with KV cache.

    Reads ``kv_keys_prev`` / ``kv_values_prev`` (prior-token KV);
    computes Q/K/V from ``x``; concatenates new K/V to the prior
    KV; computes scaled-dot-product attention with strict causal
    masking over the *combined* sequence; projects back through
    ``W_O``.

    Returns ``(out, kv_keys_new, kv_values_new, attn_weights)``.
    ``out`` has shape ``(n_tokens, d_model)``;
    ``kv_keys_new`` / ``kv_values_new`` have shape
    ``(n_prev + n_tokens, d_model)`` (the updated cache for this
    layer); ``attn_weights`` is
    ``(n_heads, n_tokens, n_prev + n_tokens)`` over the combined
    sequence.

    The attention is real: it is computed in NumPy from real Q/K/V
    projections, with a real causal mask. Not a proxy.
    """
    n_tokens = int(x.shape[0])
    d_model = int(x.shape[1])
    n_heads = int(params.n_heads)
    d_head = d_model // n_heads

    q = x @ params.w_q
    k_new = x @ params.w_k
    v_new = x @ params.w_v

    if kv_keys_prev.size == 0:
        k_all = k_new
        v_all = v_new
    else:
        k_all = _np.concatenate(
            [kv_keys_prev, k_new], axis=0)
        v_all = _np.concatenate(
            [kv_values_prev, v_new], axis=0)

    q_h = _split_heads(q, n_heads)         # (H, T_new, D_h)
    k_h = _split_heads(k_all, n_heads)     # (H, T_all, D_h)
    v_h = _split_heads(v_all, n_heads)     # (H, T_all, D_h)

    scores = _np.einsum(
        "htd,hsd->hts", q_h, k_h) / math.sqrt(float(d_head))

    n_prev = int(kv_keys_prev.shape[0]) if kv_keys_prev.size else 0
    # Strict causal mask: query token at position ``n_prev + i``
    # (i in [0, n_tokens)) can attend to all key positions in
    # [0, n_prev + i].
    mask = _np.full(
        (n_tokens, n_prev + n_tokens), -1e9, dtype=_np.float64)
    for i in range(n_tokens):
        mask[i, : n_prev + i + 1] = 0.0
    scores = scores + mask[None, :, :]

    attn = _softmax(scores, axis=-1)       # (H, T_new, T_all)
    out_h = _np.einsum("hts,hsd->htd", attn, v_h)
    out = _merge_heads(out_h) @ params.w_o

    return out, k_all, v_all, attn


def forward_tiny_substrate(
        params: TinySubstrateParams,
        token_ids: Sequence[int],
        *,
        kv_cache: TinyKVCache | None = None,
        return_attention: bool = True,
) -> TinyForwardTrace:
    """Real forward pass over the tiny substrate.

    Computes embeddings → per-layer (LN1 → attention → residual →
    LN2 → FF → residual) → final LN → unembedding. Per-layer KV
    cache is grown by ``len(token_ids)`` entries. Causal mask is
    strictly enforced.

    Determinism: identical ``params`` + identical ``token_ids`` +
    identical ``kv_cache`` produce byte-identical hidden states /
    attention / logits / output KV cache.
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
        new_cache = TinyKVCache.empty(cfg.n_layers)
    else:
        new_cache = kv_cache.clone()
    # If the cache was empty but n_layers differ, re-init shape.
    if len(new_cache.keys) != cfg.n_layers:
        new_cache = TinyKVCache.empty(cfg.n_layers)

    for layer_idx, layer in enumerate(params.layers):
        x_norm = _layer_norm(x, layer.ln1)
        kv_k = new_cache.keys[layer_idx]
        kv_v = new_cache.values[layer_idx]
        if kv_k.size == 0:
            kv_k = _np.zeros((0, cfg.d_model), dtype=_np.float64)
            kv_v = _np.zeros((0, cfg.d_model), dtype=_np.float64)
        attn_out, k_all, v_all, attn = _attention_layer_forward(
            x_norm, layer.attn,
            kv_keys_prev=kv_k,
            kv_values_prev=kv_v,
            return_attention=return_attention)
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

    return TinyForwardTrace(
        logits=logits,
        hidden_states=hidden_states,
        attn_weights_per_layer=attn_weights,
        kv_cache=new_cache,
        token_ids=tuple(int(t) for t in token_ids),
        config_cid=cfg.cid(),
        params_cid=params.cid(),
    )


def decode_greedy_tiny_substrate(
        params: TinySubstrateParams,
        prompt_token_ids: Sequence[int],
        *,
        max_new_tokens: int = 8,
        kv_cache: TinyKVCache | None = None,
        stop_token: int = W56_TINY_SUBSTRATE_EOS_TOKEN,
) -> tuple[list[int], TinyForwardTrace]:
    """Real KV-cache-aware greedy decoding loop.

    Returns ``(generated_token_ids, final_forward_trace)``. The
    final trace's ``kv_cache`` reflects the full prefix + generated
    sequence; ``logits[-1]`` are the post-generation logits over
    the vocabulary at the final position.

    Determinism: greedy argmax → identical params + prompt + cache
    → identical generated tokens.
    """
    cfg = params.config
    cur_cache = (kv_cache.clone() if kv_cache is not None
                 else TinyKVCache.empty(cfg.n_layers))
    # First, ingest the prompt (creates cache).
    trace = forward_tiny_substrate(
        params, list(prompt_token_ids), kv_cache=cur_cache,
        return_attention=False)
    cur_cache = trace.kv_cache
    generated: list[int] = []
    for _ in range(int(max_new_tokens)):
        nxt = int(_np.argmax(trace.logits[-1]))
        generated.append(int(nxt))
        if nxt == int(stop_token):
            break
        trace = forward_tiny_substrate(
            params, [int(nxt)], kv_cache=cur_cache,
            return_attention=False)
        cur_cache = trace.kv_cache
    return generated, trace


# =============================================================================
# Tokenisation
# =============================================================================


def tokenize_bytes(
        text: str, *,
        max_len: int = W56_DEFAULT_TINY_MAX_LEN,
        add_bos: bool = True,
) -> list[int]:
    """Byte-level tokenisation. Real tokenisation, not a proxy.

    Each Unicode character is encoded as one or more bytes via
    UTF-8; each byte becomes one token id in ``[0, 256)``. Control
    tokens (BOS=257, EOS=258, PAD=256) are appended explicitly.
    """
    raw = text.encode("utf-8")
    ids: list[int] = []
    if add_bos:
        ids.append(int(W56_TINY_SUBSTRATE_BOS_TOKEN))
    ids.extend(int(b) for b in raw)
    if len(ids) > int(max_len):
        ids = ids[: int(max_len)]
    return ids


def detokenize_bytes(token_ids: Sequence[int]) -> str:
    """Inverse of ``tokenize_bytes``. Strips control tokens."""
    buf = bytearray()
    for t in token_ids:
        ti = int(t)
        if ti == W56_TINY_SUBSTRATE_PAD_TOKEN:
            continue
        if ti == W56_TINY_SUBSTRATE_BOS_TOKEN:
            continue
        if ti == W56_TINY_SUBSTRATE_EOS_TOKEN:
            break
        if 0 <= ti < 256:
            buf.append(ti)
    try:
        return buf.decode("utf-8", errors="replace")
    except UnicodeDecodeError:
        return buf.decode("latin-1")


# =============================================================================
# Tiny training (toy, for in-test substrate fitting)
# =============================================================================


def fit_substrate_next_token(
        params: TinySubstrateParams,
        sequences: Sequence[Sequence[int]],
        *,
        learning_rate: float = 0.05,
        n_steps: int = 32,
) -> dict[str, Any]:
    """Tiny gradient-free coordinate-descent training step.

    Honest scope: this is NOT a competitive optimiser. It is a
    cheap finite-difference coordinate update used in benchmark
    families that need a slightly-fit substrate. Trains only the
    unembedding bias-equivalent (a learnable post-projection
    scale) — leaves attention and FF random. Returns a loss
    history.

    The point is to enable a benchmark cell that asks "does
    feeding the carrier into the KV cache lift task-correct rate
    on a fitted substrate?". Even a tiny amount of fit is enough
    to expose the difference. We do NOT claim end-to-end
    autograd training of the substrate; that requires
    PyTorch/JAX. ``W56-L-NUMPY-CPU-TINY-SUBSTRATE-CAP`` documents
    this.
    """
    cfg = params.config
    history: list[float] = []
    for step in range(int(n_steps)):
        total_loss = 0.0
        total_grad = _np.zeros_like(params.unembed)
        for seq in sequences:
            if len(seq) < 2:
                continue
            trace = forward_tiny_substrate(params, list(seq[:-1]))
            target = _np.asarray(seq[1:], dtype=_np.int64)
            logits = trace.logits
            probs = _softmax(logits, axis=-1)
            n_t = int(probs.shape[0])
            for i in range(n_t):
                total_loss -= math.log(
                    max(float(probs[i, int(target[i])]), 1e-12))
            onehot = _np.zeros_like(probs)
            for i in range(n_t):
                onehot[i, int(target[i])] = 1.0
            # Gradient of softmax CE wrt unembed (final = ln_f(x_L))
            final = _layer_norm(
                trace.hidden_states[-1], params.ln_f)
            total_grad += final.T @ (probs - onehot)
        if sequences:
            total_loss /= float(len(sequences))
            params.unembed -= float(learning_rate) * total_grad
        history.append(float(total_loss))
    return {"history": [float(h) for h in history]}


# =============================================================================
# Substrate witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class TinySubstrateForwardWitness:
    """Content-addressed witness for one tiny substrate forward.

    Binds the substrate config CID, params CID, the input token
    ids, the hidden state CID (covers all per-layer activations),
    the logits CID, the attention CID, and the KV cache CID into
    a single witness CID that can be cross-referenced from the
    W56 envelope.
    """

    config_cid: str
    params_cid: str
    token_ids: tuple[int, ...]
    hidden_state_cid: str
    logits_cid: str
    attention_cid: str
    kv_cache_cid: str
    forward_trace_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W56_TINY_SUBSTRATE_SCHEMA_VERSION,
            "config_cid": str(self.config_cid),
            "params_cid": str(self.params_cid),
            "token_ids": list(self.token_ids),
            "hidden_state_cid": str(self.hidden_state_cid),
            "logits_cid": str(self.logits_cid),
            "attention_cid": str(self.attention_cid),
            "kv_cache_cid": str(self.kv_cache_cid),
            "forward_trace_cid": str(self.forward_trace_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_substrate_forward_witness",
            "witness": self.to_dict(),
        })


def emit_tiny_substrate_forward_witness(
        trace: TinyForwardTrace,
) -> TinySubstrateForwardWitness:
    return TinySubstrateForwardWitness(
        config_cid=str(trace.config_cid),
        params_cid=str(trace.params_cid),
        token_ids=tuple(trace.token_ids),
        hidden_state_cid=str(trace.hidden_state_cid()),
        logits_cid=str(trace.logits_cid()),
        attention_cid=str(trace.attention_cid()),
        kv_cache_cid=str(trace.kv_cache.cid()),
        forward_trace_cid=str(trace.cid()),
    )


# =============================================================================
# Convenience constructors
# =============================================================================


def build_default_tiny_substrate(
        *, seed: int = W56_DEFAULT_TINY_SEED,
) -> TinySubstrateParams:
    """Default-config tiny substrate; deterministic for a fixed seed."""
    return TinySubstrateParams.init(
        TinySubstrateConfig(seed=int(seed)))


__all__ = [
    "W56_TINY_SUBSTRATE_SCHEMA_VERSION",
    "W56_TINY_SUBSTRATE_VOCAB_SIZE",
    "W56_TINY_SUBSTRATE_PAD_TOKEN",
    "W56_TINY_SUBSTRATE_BOS_TOKEN",
    "W56_TINY_SUBSTRATE_EOS_TOKEN",
    "W56_DEFAULT_TINY_D_MODEL",
    "W56_DEFAULT_TINY_N_HEADS",
    "W56_DEFAULT_TINY_N_LAYERS",
    "W56_DEFAULT_TINY_FF_HIDDEN",
    "W56_DEFAULT_TINY_MAX_LEN",
    "W56_DEFAULT_TINY_INIT_SCALE",
    "W56_DEFAULT_TINY_SEED",
    "TinyAttentionParams",
    "TinyFFParams",
    "TinyLayerNormParams",
    "TinyLayerParams",
    "TinySubstrateConfig",
    "TinySubstrateParams",
    "TinyKVCache",
    "TinyForwardTrace",
    "TinySubstrateForwardWitness",
    "forward_tiny_substrate",
    "decode_greedy_tiny_substrate",
    "tokenize_bytes",
    "detokenize_bytes",
    "fit_substrate_next_token",
    "emit_tiny_substrate_forward_witness",
    "build_default_tiny_substrate",
]
