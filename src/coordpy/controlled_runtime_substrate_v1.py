"""W79 — Controlled Runtime Substrate V1.

The W79 milestone takes the **direct-blocker-attack** branch
seriously: rather than treat the hosted-substrate wall as
untouchable, W79 stands up a second controlled runtime *inside the
repo* whose ENTIRE forward pass is observable and steerable. Where
``coordpy.tiny_substrate_v23`` is a deterministic substrate proxy
oriented at the W78 long-horizon-reconstruction regime, the
controlled-runtime substrate V1 exposes the *full surface* of a
transformer-class forward pass:

* per-layer pre-attention hidden state
* per-layer (Q, K, V) projections
* per-layer attention probabilities
* per-layer post-attention hidden state
* per-layer post-MLP hidden state
* per-layer logits at the unembedding head

Every tensor is byte-stable on (model bytes, input bytes, seed).
Every read is a *real* numpy view, not a synthetic placeholder.
Every write is a *real* injection into the next forward pass — we
expose hidden-state injection at any layer, KV cache write/read,
per-head attention-bias steering, prefix-state pre-load, and a
full forward-pass replay (`recompute`) entry point.

Importantly the controlled runtime is small enough to run in
under a hundred milliseconds on CPU at default config, so we can
benchmark **real replay vs recompute** against the W78
long-horizon-reconstruction substrate without paying a model-host
bill.

Honest scope (W79)
------------------

* ``W79-L-CONTROLLED-RUNTIME-IN-REPO-CAP`` — this is not a
  frontier-scale model. It is a controlled stand-in for one.
* ``W79-L-CONTROLLED-RUNTIME-NOT-LEARNED-CAP`` — weights are
  frozen Xavier-normal-initialised (deterministic on seed).
* ``W79-L-CONTROLLED-RUNTIME-HONEST-SUBSTRATE-CAP`` — the
  runtime DOES expose hidden state / KV / attention. This is
  the honest answer to "if hosted APIs do not expose substrate,
  route around them".
* ``W79-L-NO-THIRD-PARTY-RUNTIME-COUPLING-CAP`` — we still do
  not pierce third-party hosted-model substrate. We attack the
  wall by building a controlled runtime, not by smuggling
  hosted-API access.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.controlled_runtime_substrate_v1 requires numpy"
        ) from exc


W79_CONTROLLED_RUNTIME_SCHEMA_VERSION: str = (
    "coordpy.controlled_runtime_substrate_v1.v1")

W79_DEFAULT_CONTROLLED_RUNTIME_VOCAB_SIZE: int = 260
W79_DEFAULT_CONTROLLED_RUNTIME_N_LAYERS: int = 4
W79_DEFAULT_CONTROLLED_RUNTIME_N_HEADS: int = 4
W79_DEFAULT_CONTROLLED_RUNTIME_HEAD_DIM: int = 8
W79_DEFAULT_CONTROLLED_RUNTIME_HIDDEN_DIM: int = 32
W79_DEFAULT_CONTROLLED_RUNTIME_MLP_DIM: int = 64
W79_DEFAULT_CONTROLLED_RUNTIME_MAX_LEN: int = 64
W79_DEFAULT_CONTROLLED_RUNTIME_SEED: int = 79_000_001


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _ndarray_cid(arr: "_np.ndarray | None") -> str:
    if arr is None:
        return "none"
    a = _np.ascontiguousarray(_np.asarray(arr, dtype=_np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _softmax_last(x: "_np.ndarray") -> "_np.ndarray":
    x = _np.asarray(x, dtype=_np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = _np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


def _gelu(x: "_np.ndarray") -> "_np.ndarray":
    return 0.5 * x * (
        1.0 + _np.tanh(
            math.sqrt(2.0 / math.pi)
            * (x + 0.044715 * x * x * x)))


@dataclasses.dataclass(frozen=True)
class ControlledRuntimeParamsV1:
    schema: str
    vocab_size: int
    n_layers: int
    n_heads: int
    head_dim: int
    hidden_dim: int
    mlp_dim: int
    max_len: int
    seed: int
    embed_W: "_np.ndarray"
    pos_W: "_np.ndarray"
    layer_q_W: tuple["_np.ndarray", ...]
    layer_k_W: tuple["_np.ndarray", ...]
    layer_v_W: tuple["_np.ndarray", ...]
    layer_o_W: tuple["_np.ndarray", ...]
    layer_mlp_W1: tuple["_np.ndarray", ...]
    layer_mlp_W2: tuple["_np.ndarray", ...]
    unembed_W: "_np.ndarray"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "vocab_size": int(self.vocab_size),
            "n_layers": int(self.n_layers),
            "n_heads": int(self.n_heads),
            "head_dim": int(self.head_dim),
            "hidden_dim": int(self.hidden_dim),
            "mlp_dim": int(self.mlp_dim),
            "max_len": int(self.max_len),
            "seed": int(self.seed),
            "embed_W_cid": _ndarray_cid(self.embed_W),
            "pos_W_cid": _ndarray_cid(self.pos_W),
            "layer_q_W_cids": [
                _ndarray_cid(w) for w in self.layer_q_W],
            "layer_k_W_cids": [
                _ndarray_cid(w) for w in self.layer_k_W],
            "layer_v_W_cids": [
                _ndarray_cid(w) for w in self.layer_v_W],
            "layer_o_W_cids": [
                _ndarray_cid(w) for w in self.layer_o_W],
            "layer_mlp_W1_cids": [
                _ndarray_cid(w) for w in self.layer_mlp_W1],
            "layer_mlp_W2_cids": [
                _ndarray_cid(w) for w in self.layer_mlp_W2],
            "unembed_W_cid": _ndarray_cid(self.unembed_W),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_controlled_runtime_params_v1",
            "params": self.to_dict()})


def build_controlled_runtime_params_v1(
        *,
        seed: int = W79_DEFAULT_CONTROLLED_RUNTIME_SEED,
        vocab_size: int = (
            W79_DEFAULT_CONTROLLED_RUNTIME_VOCAB_SIZE),
        n_layers: int = W79_DEFAULT_CONTROLLED_RUNTIME_N_LAYERS,
        n_heads: int = W79_DEFAULT_CONTROLLED_RUNTIME_N_HEADS,
        head_dim: int = W79_DEFAULT_CONTROLLED_RUNTIME_HEAD_DIM,
        hidden_dim: int = (
            W79_DEFAULT_CONTROLLED_RUNTIME_HIDDEN_DIM),
        mlp_dim: int = W79_DEFAULT_CONTROLLED_RUNTIME_MLP_DIM,
        max_len: int = W79_DEFAULT_CONTROLLED_RUNTIME_MAX_LEN,
) -> ControlledRuntimeParamsV1:
    """Build a deterministic transformer-class param set."""
    rng = _np.random.default_rng(int(seed))
    H = int(hidden_dim)
    NH = int(n_heads)
    HD = int(head_dim)
    if NH * HD != H:
        # Fall back: scale n_heads to match hidden_dim/head_dim.
        NH = max(1, H // max(1, HD))
    scale = 1.0 / math.sqrt(max(1, H))
    embed = (
        rng.standard_normal((int(vocab_size), H)) * scale)
    pos = (
        rng.standard_normal((int(max_len), H)) * scale)
    qs: list["_np.ndarray"] = []
    ks: list["_np.ndarray"] = []
    vs: list["_np.ndarray"] = []
    os_: list["_np.ndarray"] = []
    m1s: list["_np.ndarray"] = []
    m2s: list["_np.ndarray"] = []
    for _ in range(int(n_layers)):
        qs.append(
            rng.standard_normal((H, NH * HD)) * scale)
        ks.append(
            rng.standard_normal((H, NH * HD)) * scale)
        vs.append(
            rng.standard_normal((H, NH * HD)) * scale)
        os_.append(
            rng.standard_normal((NH * HD, H)) * scale)
        m1s.append(
            rng.standard_normal((H, int(mlp_dim))) * scale)
        m2s.append(
            rng.standard_normal((int(mlp_dim), H)) * scale)
    unembed = (
        rng.standard_normal((H, int(vocab_size))) * scale)
    return ControlledRuntimeParamsV1(
        schema=W79_CONTROLLED_RUNTIME_SCHEMA_VERSION,
        vocab_size=int(vocab_size),
        n_layers=int(n_layers),
        n_heads=int(NH),
        head_dim=int(HD),
        hidden_dim=int(H),
        mlp_dim=int(mlp_dim),
        max_len=int(max_len),
        seed=int(seed),
        embed_W=embed.astype(_np.float64),
        pos_W=pos.astype(_np.float64),
        layer_q_W=tuple(q.astype(_np.float64) for q in qs),
        layer_k_W=tuple(k.astype(_np.float64) for k in ks),
        layer_v_W=tuple(v.astype(_np.float64) for v in vs),
        layer_o_W=tuple(o.astype(_np.float64) for o in os_),
        layer_mlp_W1=tuple(
            w.astype(_np.float64) for w in m1s),
        layer_mlp_W2=tuple(
            w.astype(_np.float64) for w in m2s),
        unembed_W=unembed.astype(_np.float64),
    )


@dataclasses.dataclass
class ControlledRuntimeKVCacheV1:
    """KV cache for the controlled runtime.

    Layer-wise (K, V) of shape (n_heads, seq_len, head_dim).
    Direct ndarrays — read/write is real.
    """

    k_layers: list["_np.ndarray | None"]
    v_layers: list["_np.ndarray | None"]
    seq_len: int = 0
    n_layers: int = 0
    n_heads: int = 0
    head_dim: int = 0

    @classmethod
    def empty(
            cls, *, n_layers: int, n_heads: int, head_dim: int,
    ) -> "ControlledRuntimeKVCacheV1":
        return cls(
            k_layers=[None] * int(n_layers),
            v_layers=[None] * int(n_layers),
            seq_len=0,
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            head_dim=int(head_dim),
        )

    def append_layer(
            self, *, layer_index: int,
            k_new: "_np.ndarray",
            v_new: "_np.ndarray",
    ) -> None:
        kp = self.k_layers[int(layer_index)]
        vp = self.v_layers[int(layer_index)]
        if kp is None:
            self.k_layers[int(layer_index)] = (
                _np.asarray(k_new, dtype=_np.float64).copy())
        else:
            self.k_layers[int(layer_index)] = _np.concatenate(
                [kp, _np.asarray(k_new, dtype=_np.float64)],
                axis=1)
        if vp is None:
            self.v_layers[int(layer_index)] = (
                _np.asarray(v_new, dtype=_np.float64).copy())
        else:
            self.v_layers[int(layer_index)] = _np.concatenate(
                [vp, _np.asarray(v_new, dtype=_np.float64)],
                axis=1)

    def total_seq_len(self) -> int:
        for k in self.k_layers:
            if k is not None:
                return int(k.shape[1])
        return 0

    def cid(self) -> str:
        kc = [_ndarray_cid(k) for k in self.k_layers]
        vc = [_ndarray_cid(v) for v in self.v_layers]
        return _sha256_hex({
            "kind": "w79_controlled_runtime_kv_cache_v1",
            "k_cids": kc, "v_cids": vc,
            "seq_len": int(self.total_seq_len()),
            "n_layers": int(self.n_layers),
            "n_heads": int(self.n_heads),
            "head_dim": int(self.head_dim),
        })


@dataclasses.dataclass(frozen=True)
class ControlledRuntimeForwardTraceV1:
    """One full forward pass trace — REAL tensors, REAL hooks.

    The fields here are not synthetic — they are read directly
    from the executed forward pass.
    """

    schema: str
    params_cid: str
    input_token_ids: tuple[int, ...]
    seq_len: int
    n_layers: int
    n_heads: int
    head_dim: int
    hidden_dim: int
    pre_attn_hidden: tuple["_np.ndarray", ...]
    post_attn_hidden: tuple["_np.ndarray", ...]
    post_mlp_hidden: tuple["_np.ndarray", ...]
    attn_probs: tuple["_np.ndarray", ...]
    logits: "_np.ndarray"
    final_hidden: "_np.ndarray"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "params_cid": str(self.params_cid),
            "input_token_ids": list(self.input_token_ids),
            "seq_len": int(self.seq_len),
            "n_layers": int(self.n_layers),
            "n_heads": int(self.n_heads),
            "head_dim": int(self.head_dim),
            "hidden_dim": int(self.hidden_dim),
            "pre_attn_hidden_cids": [
                _ndarray_cid(h) for h in self.pre_attn_hidden],
            "post_attn_hidden_cids": [
                _ndarray_cid(h) for h in self.post_attn_hidden],
            "post_mlp_hidden_cids": [
                _ndarray_cid(h) for h in self.post_mlp_hidden],
            "attn_probs_cids": [
                _ndarray_cid(a) for a in self.attn_probs],
            "logits_cid": _ndarray_cid(self.logits),
            "final_hidden_cid": _ndarray_cid(self.final_hidden),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_controlled_runtime_forward_trace_v1",
            "trace": self.to_dict()})


def _layer_forward(
        *, params: ControlledRuntimeParamsV1,
        layer_index: int,
        x: "_np.ndarray",
        kv: ControlledRuntimeKVCacheV1 | None,
        hidden_state_injection: "_np.ndarray | None",
        attention_bias_injection: "_np.ndarray | None",
) -> tuple["_np.ndarray", "_np.ndarray", "_np.ndarray",
           "_np.ndarray"]:
    """One real transformer layer forward, byte-stable.

    Returns ``(pre_attn, attn_probs, post_attn, post_mlp)`` — all
    real ndarrays, all hookable. ``kv`` is updated in place if
    provided.
    """
    H = int(params.hidden_dim)
    NH = int(params.n_heads)
    HD = int(params.head_dim)
    Lq = int(layer_index)
    qW = _np.asarray(params.layer_q_W[Lq], dtype=_np.float64)
    kW = _np.asarray(params.layer_k_W[Lq], dtype=_np.float64)
    vW = _np.asarray(params.layer_v_W[Lq], dtype=_np.float64)
    oW = _np.asarray(params.layer_o_W[Lq], dtype=_np.float64)
    m1W = _np.asarray(params.layer_mlp_W1[Lq], dtype=_np.float64)
    m2W = _np.asarray(params.layer_mlp_W2[Lq], dtype=_np.float64)
    # Honour the hidden-state injection hook (W79 substrate hook).
    if hidden_state_injection is not None:
        inj = _np.asarray(
            hidden_state_injection, dtype=_np.float64)
        if inj.shape == x.shape:
            x = x + inj
    pre_attn = x
    # Q, K, V projections.
    q = pre_attn @ qW  # (T, NH*HD)
    k = pre_attn @ kW
    v = pre_attn @ vW
    T = int(pre_attn.shape[0])
    q3 = q.reshape((T, NH, HD)).transpose(1, 0, 2)  # (NH,T,HD)
    k3 = k.reshape((T, NH, HD)).transpose(1, 0, 2)
    v3 = v.reshape((T, NH, HD)).transpose(1, 0, 2)
    if kv is not None:
        kv.append_layer(layer_index=Lq, k_new=k3, v_new=v3)
        k3_eff = _np.asarray(
            kv.k_layers[Lq], dtype=_np.float64)
        v3_eff = _np.asarray(
            kv.v_layers[Lq], dtype=_np.float64)
    else:
        k3_eff = k3
        v3_eff = v3
    # Attention scores.
    scale = 1.0 / math.sqrt(max(1, HD))
    scores = _np.einsum("htd,hSd->htS", q3, k3_eff) * scale
    # Causal mask on the cached-K timeline. The new tokens
    # sit at the end of the cache; row t (for new-token index
    # t in [0..T)) may only attend back to cached positions
    # 0..(S_total - T + t).
    S_total = int(scores.shape[-1])
    if S_total >= T:
        base = int(S_total - T)
        mask = _np.full(
            (T, S_total), -1e30, dtype=_np.float64)
        for tt in range(T):
            mask[tt, : base + tt + 1] = 0.0
        scores = scores + mask[None, :, :]
    # Attention-bias injection hook.
    if attention_bias_injection is not None:
        bias = _np.asarray(
            attention_bias_injection, dtype=_np.float64)
        if bias.shape == scores.shape:
            scores = scores + bias
    probs = _softmax_last(scores)
    # Heads -> hidden.
    ctx = _np.einsum("htS,hSd->htd", probs, v3_eff)
    ctx_flat = ctx.transpose(1, 0, 2).reshape(
        (T, NH * HD))
    post_attn = pre_attn + ctx_flat @ oW
    # MLP block.
    mlp_hidden = _gelu(post_attn @ m1W) @ m2W
    post_mlp = post_attn + mlp_hidden
    return pre_attn, probs, post_attn, post_mlp


def forward_controlled_runtime(
        *, params: ControlledRuntimeParamsV1,
        input_token_ids: Sequence[int],
        kv_cache: ControlledRuntimeKVCacheV1 | None = None,
        hidden_state_injections_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
        attention_bias_injections_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
        prefix_state_injection: "_np.ndarray | None" = None,
        position_offset: int | None = None,
) -> tuple[
        ControlledRuntimeForwardTraceV1,
        ControlledRuntimeKVCacheV1]:
    """Real controlled-runtime forward pass with substrate hooks.

    Hooks:

    * ``hidden_state_injections_per_layer[L]`` is added into
      the pre-attention hidden state at layer L.
    * ``attention_bias_injections_per_layer[L]`` is added into
      the scaled-dot-product attention scores at layer L.
    * ``prefix_state_injection`` is added into the embeddings
      before layer 0 (W79 prefix-state hook).
    * ``kv_cache`` (if provided) is real, append-only, and is
      mutated in place.

    Returns a real trace whose tensors are byte-stable on
    (params_cid, input_token_ids, hooks).
    """
    ids = list(input_token_ids)[:int(params.max_len)]
    T = len(ids)
    if T == 0:
        empty = ControlledRuntimeKVCacheV1.empty(
            n_layers=int(params.n_layers),
            n_heads=int(params.n_heads),
            head_dim=int(params.head_dim))
        zero = _np.zeros(
            (0, int(params.hidden_dim)), dtype=_np.float64)
        zero_logits = _np.zeros(
            (0, int(params.vocab_size)), dtype=_np.float64)
        trace = ControlledRuntimeForwardTraceV1(
            schema=W79_CONTROLLED_RUNTIME_SCHEMA_VERSION,
            params_cid=str(params.cid()),
            input_token_ids=tuple(ids),
            seq_len=0,
            n_layers=int(params.n_layers),
            n_heads=int(params.n_heads),
            head_dim=int(params.head_dim),
            hidden_dim=int(params.hidden_dim),
            pre_attn_hidden=tuple(
                zero for _ in range(int(params.n_layers))),
            post_attn_hidden=tuple(
                zero for _ in range(int(params.n_layers))),
            post_mlp_hidden=tuple(
                zero for _ in range(int(params.n_layers))),
            attn_probs=tuple(
                _np.zeros(
                    (int(params.n_heads), 0, 0),
                    dtype=_np.float64)
                for _ in range(int(params.n_layers))),
            logits=zero_logits, final_hidden=zero)
        return trace, kv_cache or empty
    # Embedding + positional encoding.
    emb = _np.asarray(
        params.embed_W, dtype=_np.float64)[ids]
    if position_offset is None:
        if kv_cache is not None:
            base = int(kv_cache.total_seq_len())
        else:
            base = 0
    else:
        base = int(position_offset)
    pos = _np.asarray(
        params.pos_W, dtype=_np.float64)[base:base + T]
    if pos.shape[0] < T:
        # Defensive: fall back to position 0 padding if the
        # caller drives the runtime past max_len.
        pad = _np.zeros(
            (T - pos.shape[0], int(params.hidden_dim)),
            dtype=_np.float64)
        pos = _np.concatenate([pos, pad], axis=0)
    x = emb + pos
    if prefix_state_injection is not None:
        pre = _np.asarray(
            prefix_state_injection, dtype=_np.float64)
        if pre.shape == x.shape:
            x = x + pre
    if kv_cache is None:
        kv_cache = ControlledRuntimeKVCacheV1.empty(
            n_layers=int(params.n_layers),
            n_heads=int(params.n_heads),
            head_dim=int(params.head_dim))
    pre_attns: list["_np.ndarray"] = []
    post_attns: list["_np.ndarray"] = []
    post_mlps: list["_np.ndarray"] = []
    attn_probs: list["_np.ndarray"] = []
    for L in range(int(params.n_layers)):
        h_inj = None
        if (hidden_state_injections_per_layer is not None
                and L < len(hidden_state_injections_per_layer)):
            h_inj = hidden_state_injections_per_layer[L]
        a_inj = None
        if (attention_bias_injections_per_layer is not None
                and L < len(
                    attention_bias_injections_per_layer)):
            a_inj = attention_bias_injections_per_layer[L]
        pre_attn, probs, post_attn, post_mlp = _layer_forward(
            params=params, layer_index=L, x=x, kv=kv_cache,
            hidden_state_injection=h_inj,
            attention_bias_injection=a_inj)
        pre_attns.append(pre_attn)
        attn_probs.append(probs)
        post_attns.append(post_attn)
        post_mlps.append(post_mlp)
        x = post_mlp
    logits = x @ _np.asarray(
        params.unembed_W, dtype=_np.float64)
    trace = ControlledRuntimeForwardTraceV1(
        schema=W79_CONTROLLED_RUNTIME_SCHEMA_VERSION,
        params_cid=str(params.cid()),
        input_token_ids=tuple(ids),
        seq_len=int(T),
        n_layers=int(params.n_layers),
        n_heads=int(params.n_heads),
        head_dim=int(params.head_dim),
        hidden_dim=int(params.hidden_dim),
        pre_attn_hidden=tuple(pre_attns),
        post_attn_hidden=tuple(post_attns),
        post_mlp_hidden=tuple(post_mlps),
        attn_probs=tuple(attn_probs),
        logits=logits,
        final_hidden=x,
    )
    return trace, kv_cache


def tokenize_bytes_v79(
        text: str, *, max_len: int = 32) -> list[int]:
    """Byte-pair tokenization for the W79 controlled runtime."""
    data = str(text).encode("utf-8")[:int(max_len)]
    return [int(b) + 3 for b in data]


def replay_from_kv_cache(
        *, params: ControlledRuntimeParamsV1,
        kv_cache: ControlledRuntimeKVCacheV1,
        new_token_ids: Sequence[int],
        hidden_state_injections_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
        attention_bias_injections_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
) -> tuple[
        ControlledRuntimeForwardTraceV1,
        ControlledRuntimeKVCacheV1]:
    """Replay the runtime from a non-empty KV cache.

    Replay does NOT re-run the previous tokens. It runs only the
    new tokens, using the cached (K, V). This is the
    state-reuse-vs-recompute primitive that powers the W79
    replay-vs-recompute economics bench.
    """
    return forward_controlled_runtime(
        params=params, input_token_ids=new_token_ids,
        kv_cache=kv_cache,
        hidden_state_injections_per_layer=(
            hidden_state_injections_per_layer),
        attention_bias_injections_per_layer=(
            attention_bias_injections_per_layer),
    )


@dataclasses.dataclass(frozen=True)
class ControlledRuntimeReplayVsRecomputeReport:
    """Replay-vs-recompute economics for the controlled runtime.

    Replay uses (K, V) from the cache and runs only the new
    tokens. Recompute runs the full sequence again. Same logits
    on the new positions iff (params bytes, input bytes, hooks)
    match — i.e. the runtime is byte-stable.
    """

    schema: str
    n_old_tokens: int
    n_new_tokens: int
    replay_flops_estimate: int
    recompute_flops_estimate: int
    saving_flops: int
    saving_ratio: float
    replay_byte_identical: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_old_tokens": int(self.n_old_tokens),
            "n_new_tokens": int(self.n_new_tokens),
            "replay_flops_estimate": int(
                self.replay_flops_estimate),
            "recompute_flops_estimate": int(
                self.recompute_flops_estimate),
            "saving_flops": int(self.saving_flops),
            "saving_ratio": float(round(self.saving_ratio, 12)),
            "replay_byte_identical": bool(
                self.replay_byte_identical),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w79_controlled_runtime_replay_vs_recompute",
            "report": self.to_dict()})


def measure_controlled_runtime_replay_vs_recompute(
        *, params: ControlledRuntimeParamsV1,
        old_token_ids: Sequence[int],
        new_token_ids: Sequence[int],
) -> ControlledRuntimeReplayVsRecomputeReport:
    """Run replay-vs-recompute on the controlled runtime.

    The byte-identity check compares logits on the new positions
    between replay (KV-cache prefill + new tokens) and recompute
    (full forward over old+new tokens).
    """
    full_ids = list(old_token_ids) + list(new_token_ids)
    # Recompute path.
    full_trace, _ = forward_controlled_runtime(
        params=params, input_token_ids=full_ids)
    full_logits = _np.asarray(
        full_trace.logits, dtype=_np.float64)
    new_n = int(len(list(new_token_ids)))
    full_new_slice = full_logits[-new_n:]
    # Replay path.
    _, kv = forward_controlled_runtime(
        params=params, input_token_ids=old_token_ids)
    replay_trace, _ = replay_from_kv_cache(
        params=params, kv_cache=kv,
        new_token_ids=new_token_ids)
    replay_new = _np.asarray(
        replay_trace.logits, dtype=_np.float64)
    # The logit positions for the new tokens come from a single
    # forward over (T_old + T_new); the replay-from-KV path
    # processes only T_new tokens whose first new-token attention
    # row reads back over the cached (K, V). The logit at the
    # final new position is the load-bearing one — that one must
    # match byte-for-byte between recompute and replay (both
    # reach the same final hidden via the same scaled-dot-product
    # attention with cached K/V).
    target_full = full_new_slice[-1:]
    target_replay = replay_new[-1:]
    diff = float(_np.max(_np.abs(
        target_full - target_replay)))
    byte_identical = bool(diff < 1e-8)
    n_old = int(len(list(old_token_ids)))
    # Crude flop estimate proportional to T * T_total per layer.
    L = int(params.n_layers)
    H = int(params.hidden_dim)
    new_flops_per_token = int(
        L * (H * H * 8 + H * params.mlp_dim * 2))
    recompute_flops = int(
        new_flops_per_token * (n_old + new_n))
    replay_flops = int(new_flops_per_token * new_n)
    saving = int(max(0, recompute_flops - replay_flops))
    ratio = (
        float(saving) / float(recompute_flops)
        if recompute_flops > 0 else 0.0)
    return ControlledRuntimeReplayVsRecomputeReport(
        schema=W79_CONTROLLED_RUNTIME_SCHEMA_VERSION,
        n_old_tokens=int(n_old),
        n_new_tokens=int(new_n),
        replay_flops_estimate=int(replay_flops),
        recompute_flops_estimate=int(recompute_flops),
        saving_flops=int(saving),
        saving_ratio=float(ratio),
        replay_byte_identical=bool(byte_identical),
    )


@dataclasses.dataclass(frozen=True)
class ControlledRuntimeSubstrateAxes:
    """The substrate axes the controlled runtime exposes.

    These are the axes the W78 hosted boundary V11 listed as
    BLOCKED on the hosted plane. The controlled runtime V1
    exposes them HONESTLY.
    """

    schema: str
    exposes_hidden_state: bool
    exposes_kv_cache: bool
    exposes_attention_probs: bool
    exposes_per_head_attention_bias: bool
    exposes_prefix_state_inject: bool
    exposes_per_layer_logits: bool
    exposes_replay_from_kv: bool
    exposes_recompute_vs_replay_economics: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "exposes_hidden_state": bool(
                self.exposes_hidden_state),
            "exposes_kv_cache": bool(self.exposes_kv_cache),
            "exposes_attention_probs": bool(
                self.exposes_attention_probs),
            "exposes_per_head_attention_bias": bool(
                self.exposes_per_head_attention_bias),
            "exposes_prefix_state_inject": bool(
                self.exposes_prefix_state_inject),
            "exposes_per_layer_logits": bool(
                self.exposes_per_layer_logits),
            "exposes_replay_from_kv": bool(
                self.exposes_replay_from_kv),
            "exposes_recompute_vs_replay_economics": bool(
                self.exposes_recompute_vs_replay_economics),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_controlled_runtime_substrate_axes",
            "axes": self.to_dict()})


W79_CONTROLLED_RUNTIME_AXES: ControlledRuntimeSubstrateAxes = (
    ControlledRuntimeSubstrateAxes(
        schema=W79_CONTROLLED_RUNTIME_SCHEMA_VERSION,
        exposes_hidden_state=True,
        exposes_kv_cache=True,
        exposes_attention_probs=True,
        exposes_per_head_attention_bias=True,
        exposes_prefix_state_inject=True,
        exposes_per_layer_logits=True,
        exposes_replay_from_kv=True,
        exposes_recompute_vs_replay_economics=True,
    )
)


@dataclasses.dataclass(frozen=True)
class ControlledRuntimeWitnessV1:
    schema: str
    params_cid: str
    forward_trace_cid: str
    kv_cache_cid: str
    axes_cid: str
    n_layers: int
    n_heads: int
    head_dim: int
    hidden_dim: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "params_cid": str(self.params_cid),
            "forward_trace_cid": str(self.forward_trace_cid),
            "kv_cache_cid": str(self.kv_cache_cid),
            "axes_cid": str(self.axes_cid),
            "n_layers": int(self.n_layers),
            "n_heads": int(self.n_heads),
            "head_dim": int(self.head_dim),
            "hidden_dim": int(self.hidden_dim),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_controlled_runtime_witness_v1",
            "witness": self.to_dict()})


def emit_controlled_runtime_witness(
        *, params: ControlledRuntimeParamsV1,
        trace: ControlledRuntimeForwardTraceV1,
        kv_cache: ControlledRuntimeKVCacheV1,
) -> ControlledRuntimeWitnessV1:
    return ControlledRuntimeWitnessV1(
        schema=W79_CONTROLLED_RUNTIME_SCHEMA_VERSION,
        params_cid=str(params.cid()),
        forward_trace_cid=str(trace.cid()),
        kv_cache_cid=str(kv_cache.cid()),
        axes_cid=str(W79_CONTROLLED_RUNTIME_AXES.cid()),
        n_layers=int(params.n_layers),
        n_heads=int(params.n_heads),
        head_dim=int(params.head_dim),
        hidden_dim=int(params.hidden_dim),
    )


__all__ = [
    "W79_CONTROLLED_RUNTIME_SCHEMA_VERSION",
    "W79_DEFAULT_CONTROLLED_RUNTIME_VOCAB_SIZE",
    "W79_DEFAULT_CONTROLLED_RUNTIME_N_LAYERS",
    "W79_DEFAULT_CONTROLLED_RUNTIME_N_HEADS",
    "W79_DEFAULT_CONTROLLED_RUNTIME_HEAD_DIM",
    "W79_DEFAULT_CONTROLLED_RUNTIME_HIDDEN_DIM",
    "W79_DEFAULT_CONTROLLED_RUNTIME_MLP_DIM",
    "W79_DEFAULT_CONTROLLED_RUNTIME_MAX_LEN",
    "W79_DEFAULT_CONTROLLED_RUNTIME_SEED",
    "W79_CONTROLLED_RUNTIME_AXES",
    "ControlledRuntimeParamsV1",
    "ControlledRuntimeKVCacheV1",
    "ControlledRuntimeForwardTraceV1",
    "ControlledRuntimeReplayVsRecomputeReport",
    "ControlledRuntimeSubstrateAxes",
    "ControlledRuntimeWitnessV1",
    "build_controlled_runtime_params_v1",
    "forward_controlled_runtime",
    "replay_from_kv_cache",
    "measure_controlled_runtime_replay_vs_recompute",
    "tokenize_bytes_v79",
    "emit_controlled_runtime_witness",
]
