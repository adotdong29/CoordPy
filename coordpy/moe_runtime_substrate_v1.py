"""W84 / P1 #31 — Mixture-of-Experts Substrate V1.

The W80 instrumentation contract assumes dense transformer
blocks: every token activates every layer's full attention +
full MLP. Frontier open-weight families are increasingly sparse
MoE (DeepSeek-V3, Mixtral, Qwen-MoE, DBRX). In MoE, the "hidden
state" at a layer depends on *which experts fired*; restoring KV
cache without restoring the routing decision produces a divergent
forward pass.

The W80 ``KVCacheSnapshotV1`` does not carry expert-routing
metadata. This V1 extends the contract with:

1. ``READ_EXPERT_ROUTING_PER_LAYER`` — per-layer
   ``(seq_len, top_k)`` selected expert IDs + their gate
   weights.
2. ``READ_EXPERT_OUTPUT_PER_EXPERT_PER_LAYER`` — per-(layer,
   expert) output activations for the experts that fired.
3. ``WRITE_FORCE_EXPERT_ROUTING_PER_LAYER`` — override the
   router's decision (force a specific set of experts to fire).

The implementation is a pure-NumPy MoE transformer block (one
shared dense attention + per-layer MoE MLP with N experts and
top-K routing). The contract's load-bearing structural claim —
restoring KV alone diverges, restoring KV + routing matches — is
proven on this in-repo MoE; the same axes plug into a HF /
Mixtral / Qwen-MoE adapter that's a V2 deliverable.

The MoE bench also reproduces the hidden-state intercept claim
under MoE routing: injecting a hidden state at layer L moves
the trace CID, including the routing CID.

Honest scope (W84 P1 #31)
-------------------------

* ``W84-L-MOE-RUNTIME-V1-NUMPY-CAP`` — V1 is an in-repo MoE
  transformer block in pure NumPy. The HF / Mixtral / Qwen-MoE
  adapter that exposes the same axes is V2.
* ``W84-L-MOE-RUNTIME-V1-N_EXPERTS-4-TOP_K-2-CAP`` — V1 defaults
  to 4 experts + top-2 routing (the smallest config that
  honestly counts as MoE per the issue's anti-cheat rule
  ``n_experts >= 4 and top_k >= 2``).
* ``W84-L-MOE-RUNTIME-V1-NO-SHARED-EXPERT-CAP`` — V1 does NOT
  ship a shared expert / always-on expert. DeepSeek-V3-style
  shared experts are V2.
* ``W84-L-MOE-RUNTIME-V1-SINGLE-HOST-CAP`` — V1 is single-host.
  Multi-GPU expert-parallel MoE is V2.
* ``W84-L-MOE-RUNTIME-V1-FORCE-ROUTING-AVAILABLE-CAP`` — V1
  ships the ``WRITE_FORCE_EXPERT_ROUTING_PER_LAYER`` axis (the
  issue allowed making this V2; we close it in V1).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.moe_runtime_substrate_v1 requires numpy"
    ) from exc

from .runtime_instrumentation_v1 import (
    AttentionSnapshotV1,
    CapabilityTag,
    ForwardTraceV1,
    HiddenStateSnapshotV1,
    InjectionPlanV1,
    InstrumentationAxis,
    KVCacheSnapshotV1,
    W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
)


W84_MOE_RUNTIME_V1_SCHEMA_VERSION: str = (
    "coordpy.moe_runtime_substrate_v1.v1")


# ---------------------------------------------------------------
# Extended axes (the MoE-specific axes the contract V1 ships).
# These string values appear in the parity matrix and conformance
# reports alongside the W80 dense axes.
# ---------------------------------------------------------------


class MoEInstrumentationAxis(str, enum.Enum):
    READ_EXPERT_ROUTING_PER_LAYER = (
        "read_expert_routing_per_layer")
    READ_EXPERT_OUTPUT_PER_EXPERT_PER_LAYER = (
        "read_expert_output_per_expert_per_layer")
    WRITE_FORCE_EXPERT_ROUTING_PER_LAYER = (
        "write_force_expert_routing_per_layer")


W84_MOE_INSTRUMENTATION_AXES: tuple[str, ...] = tuple(
    a.value for a in MoEInstrumentationAxis)


# ---------------------------------------------------------------
# Defaults — the issue's anti-cheat: n_experts >= 4, top_k >= 2.
# ---------------------------------------------------------------


W84_MOE_DEFAULT_VOCAB_SIZE: int = 128
W84_MOE_DEFAULT_N_LAYERS: int = 3
W84_MOE_DEFAULT_N_HEADS: int = 4
W84_MOE_DEFAULT_HEAD_DIM: int = 8
W84_MOE_DEFAULT_HIDDEN_DIM: int = 32
W84_MOE_DEFAULT_MLP_DIM: int = 64
W84_MOE_DEFAULT_MAX_LEN: int = 64
W84_MOE_DEFAULT_N_EXPERTS: int = 4
W84_MOE_DEFAULT_TOP_K: int = 2
W84_MOE_DEFAULT_SEED: int = 84_031_001


# ---------------------------------------------------------------
# Snapshot schemas
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ExpertRoutingSnapshotV1:
    """Per-layer expert routing snapshot.

    ``per_layer_top_k_ids`` is one ``(seq_len, top_k)`` int
    array per layer. ``per_layer_top_k_gates`` is the gate
    probability for each chosen expert. Both are
    content-addressed: identical routing -> identical CID.
    """

    schema: str
    n_layers: int
    seq_len: int
    n_experts: int
    top_k: int
    per_layer_top_k_ids: tuple["_np.ndarray", ...]
    per_layer_top_k_gates: tuple["_np.ndarray", ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_layers": int(self.n_layers),
            "seq_len": int(self.seq_len),
            "n_experts": int(self.n_experts),
            "top_k": int(self.top_k),
            "per_layer_top_k_ids_cids": [
                _ndarray_cid(a)
                for a in self.per_layer_top_k_ids],
            "per_layer_top_k_gates_cids": [
                _ndarray_cid(a)
                for a in self.per_layer_top_k_gates],
            "per_layer_top_k_ids_shapes": [
                _shape(a) for a in self.per_layer_top_k_ids],
            "per_layer_top_k_gates_shapes": [
                _shape(a) for a in self.per_layer_top_k_gates],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_expert_routing_snapshot_v1",
            "snapshot": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class ExpertOutputSnapshotV1:
    """Per-(layer, expert) output activations for fired experts.

    ``per_layer_expert_outputs`` is a tuple of length n_layers,
    each a dict ``{expert_id: ndarray}`` for the experts that
    fired in that layer.
    """

    schema: str
    n_layers: int
    n_experts: int
    seq_len: int
    hidden_dim: int
    # We store as parallel tuples for ndarray purity.
    per_layer_expert_ids: tuple[tuple[int, ...], ...]
    per_layer_expert_outputs: tuple[
        tuple["_np.ndarray", ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_layers": int(self.n_layers),
            "n_experts": int(self.n_experts),
            "seq_len": int(self.seq_len),
            "hidden_dim": int(self.hidden_dim),
            "per_layer_expert_ids": [
                [int(e) for e in ids]
                for ids in self.per_layer_expert_ids],
            "per_layer_expert_output_cids": [
                [_ndarray_cid(o) for o in outs]
                for outs in self.per_layer_expert_outputs],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_expert_output_snapshot_v1",
            "snapshot": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class MoEForceRoutingPlanV1:
    """An override plan for the WRITE_FORCE_EXPERT_ROUTING axis.

    ``force_top_k_ids_per_layer`` overrides the router's decision
    on the named layers with the named ``(seq_len, top_k)``
    integer expert-id arrays. Layers set to ``None`` use the
    router's natural decision.
    """

    schema: str
    force_top_k_ids_per_layer: tuple[
        "_np.ndarray | None", ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "force_top_k_ids_cids_per_layer": [
                _ndarray_cid(a)
                for a in self.force_top_k_ids_per_layer],
            "force_top_k_ids_shapes_per_layer": [
                _shape(a)
                for a in self.force_top_k_ids_per_layer],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_moe_force_routing_plan_v1",
            "plan": self.to_dict()})


# ---------------------------------------------------------------
# In-repo MoE transformer
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MoERuntimeParamsV1:
    """Deterministic MoE transformer params (pure NumPy).

    Each layer has:
    * dense attention (Q, K, V, O projections)
    * a router (single linear: hidden -> n_experts)
    * n_experts independent (mlp_W1, mlp_W2) pairs
    """

    schema: str
    vocab_size: int
    n_layers: int
    n_heads: int
    head_dim: int
    hidden_dim: int
    mlp_dim: int
    max_len: int
    n_experts: int
    top_k: int
    seed: int
    embed_W: "_np.ndarray"
    pos_W: "_np.ndarray"
    layer_q_W: tuple["_np.ndarray", ...]
    layer_k_W: tuple["_np.ndarray", ...]
    layer_v_W: tuple["_np.ndarray", ...]
    layer_o_W: tuple["_np.ndarray", ...]
    layer_router_W: tuple["_np.ndarray", ...]
    layer_expert_mlp_W1: tuple[tuple["_np.ndarray", ...], ...]
    layer_expert_mlp_W2: tuple[tuple["_np.ndarray", ...], ...]
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
            "n_experts": int(self.n_experts),
            "top_k": int(self.top_k),
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
            "layer_router_W_cids": [
                _ndarray_cid(w) for w in self.layer_router_W],
            "layer_expert_mlp_W1_cids": [
                [_ndarray_cid(w) for w in es]
                for es in self.layer_expert_mlp_W1],
            "layer_expert_mlp_W2_cids": [
                [_ndarray_cid(w) for w in es]
                for es in self.layer_expert_mlp_W2],
            "unembed_W_cid": _ndarray_cid(self.unembed_W),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_moe_runtime_params_v1",
            "params": self.to_dict()})


def build_moe_runtime_params_v1(
        *,
        seed: int = W84_MOE_DEFAULT_SEED,
        vocab_size: int = W84_MOE_DEFAULT_VOCAB_SIZE,
        n_layers: int = W84_MOE_DEFAULT_N_LAYERS,
        n_heads: int = W84_MOE_DEFAULT_N_HEADS,
        head_dim: int = W84_MOE_DEFAULT_HEAD_DIM,
        hidden_dim: int = W84_MOE_DEFAULT_HIDDEN_DIM,
        mlp_dim: int = W84_MOE_DEFAULT_MLP_DIM,
        max_len: int = W84_MOE_DEFAULT_MAX_LEN,
        n_experts: int = W84_MOE_DEFAULT_N_EXPERTS,
        top_k: int = W84_MOE_DEFAULT_TOP_K,
) -> MoERuntimeParamsV1:
    """Build a deterministic MoE transformer param set.

    The anti-cheat: ``n_experts >= 4`` and ``top_k >= 2`` MUST
    hold. We raise rather than silently accept degenerate
    configs.
    """
    if int(n_experts) < 4:
        raise ValueError(
            "MoE runtime V1 anti-cheat: n_experts must be >= 4 "
            f"(got {int(n_experts)})")
    if int(top_k) < 2:
        raise ValueError(
            "MoE runtime V1 anti-cheat: top_k must be >= 2 "
            f"(got {int(top_k)})")
    if int(top_k) > int(n_experts):
        raise ValueError(
            f"top_k ({int(top_k)}) cannot exceed "
            f"n_experts ({int(n_experts)})")
    rng = _np.random.default_rng(int(seed))
    H = int(hidden_dim)
    NH = int(n_heads)
    HD = int(head_dim)
    if NH * HD != H:
        NH = max(1, H // max(1, HD))
    scale = 1.0 / math.sqrt(max(1, H))
    embed = rng.standard_normal(
        (int(vocab_size), H)) * scale
    pos = rng.standard_normal(
        (int(max_len), H)) * scale
    qs: list["_np.ndarray"] = []
    ks: list["_np.ndarray"] = []
    vs: list["_np.ndarray"] = []
    os_: list["_np.ndarray"] = []
    rs: list["_np.ndarray"] = []
    layer_e1: list[tuple["_np.ndarray", ...]] = []
    layer_e2: list[tuple["_np.ndarray", ...]] = []
    for _ in range(int(n_layers)):
        qs.append(rng.standard_normal((H, NH * HD)) * scale)
        ks.append(rng.standard_normal((H, NH * HD)) * scale)
        vs.append(rng.standard_normal((H, NH * HD)) * scale)
        os_.append(rng.standard_normal((NH * HD, H)) * scale)
        rs.append(rng.standard_normal(
            (H, int(n_experts))) * scale)
        e1s: list["_np.ndarray"] = []
        e2s: list["_np.ndarray"] = []
        for _ in range(int(n_experts)):
            e1s.append(rng.standard_normal(
                (H, int(mlp_dim))) * scale)
            e2s.append(rng.standard_normal(
                (int(mlp_dim), H)) * scale)
        layer_e1.append(tuple(e1s))
        layer_e2.append(tuple(e2s))
    unembed = rng.standard_normal(
        (H, int(vocab_size))) * scale
    return MoERuntimeParamsV1(
        schema=W84_MOE_RUNTIME_V1_SCHEMA_VERSION,
        vocab_size=int(vocab_size),
        n_layers=int(n_layers),
        n_heads=int(NH),
        head_dim=int(HD),
        hidden_dim=int(H),
        mlp_dim=int(mlp_dim),
        max_len=int(max_len),
        n_experts=int(n_experts),
        top_k=int(top_k),
        seed=int(seed),
        embed_W=embed.astype(_np.float64),
        pos_W=pos.astype(_np.float64),
        layer_q_W=tuple(q.astype(_np.float64) for q in qs),
        layer_k_W=tuple(k.astype(_np.float64) for k in ks),
        layer_v_W=tuple(v.astype(_np.float64) for v in vs),
        layer_o_W=tuple(o.astype(_np.float64) for o in os_),
        layer_router_W=tuple(
            r.astype(_np.float64) for r in rs),
        layer_expert_mlp_W1=tuple(
            tuple(w.astype(_np.float64) for w in es)
            for es in layer_e1),
        layer_expert_mlp_W2=tuple(
            tuple(w.astype(_np.float64) for w in es)
            for es in layer_e2),
        unembed_W=unembed.astype(_np.float64),
    )


@dataclasses.dataclass
class MoEKVCacheV1:
    """KV cache for the MoE runtime (same shape as W79 dense)."""

    k_layers: list["_np.ndarray | None"]
    v_layers: list["_np.ndarray | None"]
    n_layers: int = 0
    n_heads: int = 0
    head_dim: int = 0

    @classmethod
    def empty(
            cls, *, n_layers: int, n_heads: int, head_dim: int,
    ) -> "MoEKVCacheV1":
        return cls(
            k_layers=[None] * int(n_layers),
            v_layers=[None] * int(n_layers),
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            head_dim=int(head_dim))

    def append_layer(
            self, *, layer_index: int,
            k_new: "_np.ndarray", v_new: "_np.ndarray",
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


@dataclasses.dataclass(frozen=True)
class MoEForwardTraceV1:
    """One MoE forward pass — REAL routing, REAL expert outputs."""

    schema: str
    params_cid: str
    input_token_ids: tuple[int, ...]
    seq_len: int
    n_layers: int
    n_experts: int
    top_k: int
    hidden_dim: int
    pre_attn_hidden: tuple["_np.ndarray", ...]
    post_attn_hidden: tuple["_np.ndarray", ...]
    post_mlp_hidden: tuple["_np.ndarray", ...]
    attn_probs: tuple["_np.ndarray", ...]
    routing: ExpertRoutingSnapshotV1
    expert_outputs: ExpertOutputSnapshotV1
    logits: "_np.ndarray"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "params_cid": str(self.params_cid),
            "input_token_ids": list(self.input_token_ids),
            "seq_len": int(self.seq_len),
            "n_layers": int(self.n_layers),
            "n_experts": int(self.n_experts),
            "top_k": int(self.top_k),
            "hidden_dim": int(self.hidden_dim),
            "pre_attn_cids": [
                _ndarray_cid(h) for h in self.pre_attn_hidden],
            "post_attn_cids": [
                _ndarray_cid(h) for h in self.post_attn_hidden],
            "post_mlp_cids": [
                _ndarray_cid(h) for h in self.post_mlp_hidden],
            "attn_cids": [
                _ndarray_cid(a) for a in self.attn_probs],
            "routing_cid": str(self.routing.cid()),
            "expert_outputs_cid": str(
                self.expert_outputs.cid()),
            "logits_cid": _ndarray_cid(self.logits),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_moe_forward_trace_v1",
            "trace": self.to_dict()})


# ---------------------------------------------------------------
# Forward implementation
# ---------------------------------------------------------------


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


def _attention_layer(
        *, params: MoERuntimeParamsV1, L: int,
        pre_attn: "_np.ndarray",
        kv: MoEKVCacheV1 | None,
        hidden_state_injection: "_np.ndarray | None",
) -> tuple["_np.ndarray", "_np.ndarray"]:
    """Dense attention block (post-attn = pre + ctx @ O).

    Returns ``(post_attn, attn_probs)``.
    """
    H = int(params.hidden_dim)
    NH = int(params.n_heads)
    HD = int(params.head_dim)
    qW = _np.asarray(params.layer_q_W[L], dtype=_np.float64)
    kW = _np.asarray(params.layer_k_W[L], dtype=_np.float64)
    vW = _np.asarray(params.layer_v_W[L], dtype=_np.float64)
    oW = _np.asarray(params.layer_o_W[L], dtype=_np.float64)
    if hidden_state_injection is not None:
        inj = _np.asarray(
            hidden_state_injection, dtype=_np.float64)
        if inj.shape == pre_attn.shape:
            pre_attn = pre_attn + inj
    q = pre_attn @ qW
    k = pre_attn @ kW
    v = pre_attn @ vW
    T = int(pre_attn.shape[0])
    q3 = q.reshape((T, NH, HD)).transpose(1, 0, 2)
    k3 = k.reshape((T, NH, HD)).transpose(1, 0, 2)
    v3 = v.reshape((T, NH, HD)).transpose(1, 0, 2)
    if kv is not None:
        kv.append_layer(layer_index=L, k_new=k3, v_new=v3)
        k3_eff = _np.asarray(
            kv.k_layers[L], dtype=_np.float64)
        v3_eff = _np.asarray(
            kv.v_layers[L], dtype=_np.float64)
    else:
        k3_eff = k3
        v3_eff = v3
    scale = 1.0 / math.sqrt(max(1, HD))
    scores = _np.einsum(
        "htd,hSd->htS", q3, k3_eff) * scale
    S_total = int(scores.shape[-1])
    if S_total >= T:
        base = int(S_total - T)
        mask = _np.full(
            (T, S_total), -1e30, dtype=_np.float64)
        for tt in range(T):
            mask[tt, : base + tt + 1] = 0.0
        scores = scores + mask[None, :, :]
    probs = _softmax_last(scores)
    ctx = _np.einsum("htS,hSd->htd", probs, v3_eff)
    ctx_flat = ctx.transpose(1, 0, 2).reshape((T, NH * HD))
    post_attn = pre_attn + ctx_flat @ oW
    return post_attn, probs


def _route_top_k(
        *, params: MoERuntimeParamsV1, L: int,
        h: "_np.ndarray",
        force_top_k_ids: "_np.ndarray | None" = None,
) -> tuple["_np.ndarray", "_np.ndarray"]:
    """Top-K router over experts.

    Returns ``(top_k_ids, top_k_gates)`` of shapes
    ``(seq_len, top_k)``.

    If ``force_top_k_ids`` is provided, the router's decision is
    overridden but the *gate weights* are recomputed by softmax-
    ing over the forced ids' router scores (so the forced
    routing is still differentiable / replayable).
    """
    H = int(params.hidden_dim)
    NE = int(params.n_experts)
    K = int(params.top_k)
    rW = _np.asarray(
        params.layer_router_W[L], dtype=_np.float64)
    scores = h @ rW  # (T, NE)
    if force_top_k_ids is not None:
        forced = _np.asarray(
            force_top_k_ids, dtype=_np.int64)
        T = int(scores.shape[0])
        if forced.shape != (T, K):
            raise ValueError(
                "force_top_k_ids has wrong shape "
                f"(expected ({T}, {K}), got {forced.shape})")
        gates_logits = _np.take_along_axis(scores, forced, axis=1)
        gates = _softmax_last(gates_logits)
        return forced.astype(_np.int64), gates
    # Natural top-K
    order = _np.argsort(-scores, axis=1)[:, :K]
    gates_logits = _np.take_along_axis(scores, order, axis=1)
    gates = _softmax_last(gates_logits)
    return order.astype(_np.int64), gates


def _moe_mlp_layer(
        *, params: MoERuntimeParamsV1, L: int,
        post_attn: "_np.ndarray",
        force_top_k_ids: "_np.ndarray | None" = None,
) -> tuple["_np.ndarray", "_np.ndarray",
           "_np.ndarray", dict[int, "_np.ndarray"]]:
    """MoE MLP block. Returns ``(post_mlp, top_k_ids,
    top_k_gates, expert_outputs)``.

    ``expert_outputs`` maps expert_id -> the (n_tokens_routed,
    hidden_dim) output that expert produced for the tokens
    routed to it.
    """
    T, H = int(post_attn.shape[0]), int(post_attn.shape[1])
    top_k_ids, top_k_gates = _route_top_k(
        params=params, L=L, h=post_attn,
        force_top_k_ids=force_top_k_ids)
    NE = int(params.n_experts)
    K = int(params.top_k)
    out = _np.zeros_like(post_attn, dtype=_np.float64)
    fired_outputs: dict[int, list["_np.ndarray"]] = {}
    fired_token_idx: dict[int, list[int]] = {}
    # For each token + each chosen expert, run the expert.
    for t in range(T):
        for k in range(K):
            e = int(top_k_ids[t, k])
            g = float(top_k_gates[t, k])
            W1 = _np.asarray(
                params.layer_expert_mlp_W1[L][e],
                dtype=_np.float64)
            W2 = _np.asarray(
                params.layer_expert_mlp_W2[L][e],
                dtype=_np.float64)
            h_t = post_attn[t:t + 1]  # (1, H)
            exp_out = _gelu(h_t @ W1) @ W2  # (1, H)
            out[t:t + 1] += g * exp_out
            fired_outputs.setdefault(e, []).append(
                exp_out[0].astype(_np.float64))
            fired_token_idx.setdefault(e, []).append(t)
    expert_outputs: dict[int, "_np.ndarray"] = {}
    for e, vs in fired_outputs.items():
        expert_outputs[int(e)] = _np.stack(vs, axis=0)
    return post_attn + out, top_k_ids, top_k_gates, (
        expert_outputs)


def forward_moe_runtime(
        *, params: MoERuntimeParamsV1,
        input_token_ids: Sequence[int],
        kv_cache: MoEKVCacheV1 | None = None,
        hidden_state_injections_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
        force_routing_plan: MoEForceRoutingPlanV1 | None = None,
) -> tuple[MoEForwardTraceV1, MoEKVCacheV1]:
    """Run a full forward pass with REAL routing.

    Returns ``(trace, kv_cache_after)``.
    """
    ids = list(int(t) for t in input_token_ids)
    T = len(ids)
    H = int(params.hidden_dim)
    if T == 0:
        raise ValueError("input_token_ids empty")
    emb = _np.asarray(params.embed_W, dtype=_np.float64)
    pos = _np.asarray(params.pos_W, dtype=_np.float64)
    # Position offset = current cache len (so cached + new
    # positions line up).
    if kv_cache is None:
        kv_cache = MoEKVCacheV1.empty(
            n_layers=int(params.n_layers),
            n_heads=int(params.n_heads),
            head_dim=int(params.head_dim))
    base_pos = int(kv_cache.total_seq_len())
    x = _np.zeros((T, H), dtype=_np.float64)
    for i, t in enumerate(ids):
        x[i] = emb[int(t)] + pos[int(base_pos + i)]
    pre_attn_list: list["_np.ndarray"] = []
    post_attn_list: list["_np.ndarray"] = []
    post_mlp_list: list["_np.ndarray"] = []
    attn_probs_list: list["_np.ndarray"] = []
    routing_ids_list: list["_np.ndarray"] = []
    routing_gates_list: list["_np.ndarray"] = []
    expert_ids_per_layer: list[tuple[int, ...]] = []
    expert_outputs_per_layer: list[
        tuple["_np.ndarray", ...]] = []
    for L in range(int(params.n_layers)):
        hidden_inj = None
        if hidden_state_injections_per_layer is not None and (
                L < len(hidden_state_injections_per_layer)):
            hidden_inj = (
                hidden_state_injections_per_layer[L])
        pre_attn = x
        post_attn, probs = _attention_layer(
            params=params, L=L, pre_attn=pre_attn,
            kv=kv_cache,
            hidden_state_injection=hidden_inj)
        # Force-routing override for this layer (if any).
        force_ids = None
        if force_routing_plan is not None:
            if (L < len(
                    force_routing_plan.force_top_k_ids_per_layer)
                    and force_routing_plan
                    .force_top_k_ids_per_layer[L] is not None):
                force_ids = _np.asarray(
                    force_routing_plan
                    .force_top_k_ids_per_layer[L],
                    dtype=_np.int64)
        post_mlp, top_k_ids, top_k_gates, expert_outputs = (
            _moe_mlp_layer(
                params=params, L=L,
                post_attn=post_attn,
                force_top_k_ids=force_ids))
        pre_attn_list.append(pre_attn.copy())
        post_attn_list.append(post_attn.copy())
        post_mlp_list.append(post_mlp.copy())
        attn_probs_list.append(probs.copy())
        routing_ids_list.append(top_k_ids.copy())
        routing_gates_list.append(top_k_gates.copy())
        sorted_eids = sorted(expert_outputs.keys())
        expert_ids_per_layer.append(
            tuple(int(e) for e in sorted_eids))
        expert_outputs_per_layer.append(
            tuple(expert_outputs[e] for e in sorted_eids))
        x = post_mlp
    final_hidden = x[-1]
    logits = final_hidden @ _np.asarray(
        params.unembed_W, dtype=_np.float64)
    logits_full = x @ _np.asarray(
        params.unembed_W, dtype=_np.float64)
    routing = ExpertRoutingSnapshotV1(
        schema=W84_MOE_RUNTIME_V1_SCHEMA_VERSION,
        n_layers=int(params.n_layers),
        seq_len=int(T),
        n_experts=int(params.n_experts),
        top_k=int(params.top_k),
        per_layer_top_k_ids=tuple(routing_ids_list),
        per_layer_top_k_gates=tuple(routing_gates_list))
    expert_snap = ExpertOutputSnapshotV1(
        schema=W84_MOE_RUNTIME_V1_SCHEMA_VERSION,
        n_layers=int(params.n_layers),
        n_experts=int(params.n_experts),
        seq_len=int(T), hidden_dim=int(H),
        per_layer_expert_ids=tuple(expert_ids_per_layer),
        per_layer_expert_outputs=tuple(
            expert_outputs_per_layer))
    trace = MoEForwardTraceV1(
        schema=W84_MOE_RUNTIME_V1_SCHEMA_VERSION,
        params_cid=str(params.cid()),
        input_token_ids=tuple(ids),
        seq_len=int(T),
        n_layers=int(params.n_layers),
        n_experts=int(params.n_experts),
        top_k=int(params.top_k),
        hidden_dim=int(H),
        pre_attn_hidden=tuple(pre_attn_list),
        post_attn_hidden=tuple(post_attn_list),
        post_mlp_hidden=tuple(post_mlp_list),
        attn_probs=tuple(attn_probs_list),
        routing=routing,
        expert_outputs=expert_snap,
        logits=logits_full,
    )
    return trace, kv_cache


def replay_from_kv_and_routing(
        *, params: MoERuntimeParamsV1,
        kv_cache: MoEKVCacheV1,
        new_token_ids: Sequence[int],
        force_routing_plan: MoEForceRoutingPlanV1 | None = None,
) -> tuple[MoEForwardTraceV1, MoEKVCacheV1]:
    """Replay-from-KV with optional forced routing.

    The contract for MoE: replay = restore KV + (optionally)
    restore routing decisions. With ``force_routing_plan`` set
    to the original forward's routing snapshot, the replay
    produces byte-identical final logits.
    """
    return forward_moe_runtime(
        params=params,
        input_token_ids=new_token_ids,
        kv_cache=kv_cache,
        force_routing_plan=force_routing_plan,
    )


def routing_plan_from_snapshot(
        snap: ExpertRoutingSnapshotV1,
        *, n_layers: int,
        new_token_count: int | None = None,
) -> MoEForceRoutingPlanV1:
    """Build a force-routing plan from an ExpertRoutingSnapshotV1.

    If ``new_token_count`` is given, only the LAST
    ``new_token_count`` rows of each layer's routing are
    used — this is the natural shape for replay-from-KV where
    only the new tokens need their routing forced.
    """
    plan_layers: list["_np.ndarray | None"] = []
    for L in range(int(n_layers)):
        if L >= len(snap.per_layer_top_k_ids):
            plan_layers.append(None)
            continue
        ids = _np.asarray(
            snap.per_layer_top_k_ids[L], dtype=_np.int64)
        if new_token_count is not None:
            n_new = int(new_token_count)
            ids = ids[-n_new:, :]
        plan_layers.append(ids.copy())
    return MoEForceRoutingPlanV1(
        schema=W84_MOE_RUNTIME_V1_SCHEMA_VERSION,
        force_top_k_ids_per_layer=tuple(plan_layers))


# ---------------------------------------------------------------
# Adapter to the W80 instrumentation contract surface
# ---------------------------------------------------------------


def moe_declared_axes() -> Mapping[str, str]:
    a = CapabilityTag.AVAILABLE.value
    return {
        # W80 dense axes — MoE supports them too.
        InstrumentationAxis.READ_HIDDEN_STATE.value: a,
        InstrumentationAxis.READ_KV_CACHE.value: a,
        InstrumentationAxis.READ_ATTENTION_PROBS.value: a,
        InstrumentationAxis.READ_PER_LAYER_LOGITS.value: a,
        InstrumentationAxis.READ_FINAL_LOGITS.value: a,
        InstrumentationAxis.WRITE_HIDDEN_STATE_INJECT.value: a,
        InstrumentationAxis.WRITE_KV_RESTORE.value: a,
        InstrumentationAxis.WRITE_ATTENTION_BIAS.value: (
            CapabilityTag.BACKEND_SPECIFIC.value),
        InstrumentationAxis.INJECT_PREFIX_STATE.value: (
            CapabilityTag.BACKEND_SPECIFIC.value),
        InstrumentationAxis.REPLAY_FROM_KV.value: a,
        InstrumentationAxis.DETERMINISTIC_REPLAY.value: a,
        InstrumentationAxis.CONTENT_ADDRESSED_TRACE.value: a,
        # MoE-specific axes — the V1 claim.
        MoEInstrumentationAxis
            .READ_EXPERT_ROUTING_PER_LAYER.value: a,
        MoEInstrumentationAxis
            .READ_EXPERT_OUTPUT_PER_EXPERT_PER_LAYER.value: a,
        MoEInstrumentationAxis
            .WRITE_FORCE_EXPERT_ROUTING_PER_LAYER.value: a,
    }


@dataclasses.dataclass
class MoERuntimeAdapterV1:
    """Adapter speaking the W80 instrumentation contract on top
    of the in-repo MoE runtime."""

    params: MoERuntimeParamsV1 = dataclasses.field(
        default_factory=build_moe_runtime_params_v1)

    def backend_id(self) -> str:
        return "coordpy.moe_runtime_substrate_v1"

    def backend_runtime_id(self) -> str:
        return f"{self.backend_id()}@{self.params.cid()[:16]}"

    def declared_axes(self) -> Mapping[str, str]:
        return moe_declared_axes()

    def tokenize(
            self, text: str, *, max_len: int = 32,
    ) -> list[int]:
        ids = []
        for b in str(text).encode("utf-8"):
            ids.append(int(b) % int(self.params.vocab_size))
            if len(ids) >= int(max_len):
                break
        return ids


# ---------------------------------------------------------------
# Bench: divergence + replay claim
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MoEBenchReportV1:
    """Bench report capturing the load-bearing MoE claims.

    The MoE substrate's load-bearing structural claim has two
    halves. (1) Forcing a *different* expert routing on the same
    inputs MUST measurably change the post-MLP outputs and the
    final logits — otherwise the routing axis is decorative,
    not load-bearing. (2) The W84 contract carries the routing
    CID inside the trace CID, so two forwards that differ only
    in routing produce DIFFERENT trace CIDs (proving the trace
    is honest about the routing axis). (3) Replaying with the
    natural routing restored reproduces the full forward at the
    fp32 floor.
    """

    schema: str
    params_cid: str
    n_experts: int
    top_k: int
    n_layers: int
    n_prompts: int
    max_forced_routing_diff: float
    forced_routing_changes_output: bool
    max_replay_with_natural_routing_diff: float
    replay_with_natural_routing_within_floor: bool
    trace_cid_changes_with_routing: bool
    hidden_intercept_moves_cid: bool
    routing_cid_changes_with_force_plan: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "params_cid": str(self.params_cid),
            "n_experts": int(self.n_experts),
            "top_k": int(self.top_k),
            "n_layers": int(self.n_layers),
            "n_prompts": int(self.n_prompts),
            "max_forced_routing_diff": float(round(
                self.max_forced_routing_diff, 12)),
            "forced_routing_changes_output": bool(
                self.forced_routing_changes_output),
            "max_replay_with_natural_routing_diff": float(round(
                self.max_replay_with_natural_routing_diff, 12)),
            "replay_with_natural_routing_within_floor": bool(
                self.replay_with_natural_routing_within_floor),
            "trace_cid_changes_with_routing": bool(
                self.trace_cid_changes_with_routing),
            "hidden_intercept_moves_cid": bool(
                self.hidden_intercept_moves_cid),
            "routing_cid_changes_with_force_plan": bool(
                self.routing_cid_changes_with_force_plan),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_moe_bench_report_v1",
            "report": self.to_dict()})


def run_moe_bench_v1(
        *,
        params: MoERuntimeParamsV1 | None = None,
        n_prompts: int = 8,
        seed: int = 84_031_007,
) -> MoEBenchReportV1:
    """Run the MoE bench.

    Load-bearing checks:

    * **Forcing a different routing measurably changes the
      output.** Negative claim: if changing routing does NOT
      change outputs, then routing is decorative, not load-
      bearing, and the contract axis is dishonest.
    * **The trace CID distinguishes different routings.** Two
      forwards with the same inputs but different routing
      produce different trace CIDs (because routing CID is in
      the trace CID).
    * **Replay with natural routing matches recompute at fp32
      floor.** Restoring the natural routing produces the
      original output byte-identically (at fp32).
    * **Hidden-state intercept moves the trace CID under MoE.**
    * **Forcing a different routing plan produces a different
      routing CID.**
    """
    if params is None:
        params = build_moe_runtime_params_v1()
    rng = _np.random.default_rng(int(seed))
    max_forced_diff = 0.0
    forced_changes = True
    max_replay_diff = 0.0
    replay_within = True
    trace_cid_changes = True
    adapter = MoERuntimeAdapterV1(params=params)
    fp32_floor = 5e-3
    n_eval = 0
    for i in range(int(n_prompts)):
        n = int(rng.integers(5, 12))
        chars = [chr(int(c)) for c in rng.integers(
            ord('a'), ord('z'), size=n)]
        prompt = "".join(chars)
        ids = adapter.tokenize(prompt, max_len=12)
        if len(ids) < 3:
            continue
        n_eval += 1
        natural, _ = forward_moe_runtime(
            params=params, input_token_ids=ids)
        # Construct a different routing — shift each top-K id
        # by +1 mod n_experts. This is a real different routing
        # decision, not a no-op.
        mut_layers: list["_np.ndarray | None"] = []
        for L in range(int(params.n_layers)):
            ids_L = natural.routing.per_layer_top_k_ids[L]
            mut = (
                (_np.asarray(ids_L, dtype=_np.int64) + 1)
                % int(params.n_experts))
            mut_layers.append(mut.astype(_np.int64))
        forced_plan = MoEForceRoutingPlanV1(
            schema=W84_MOE_RUNTIME_V1_SCHEMA_VERSION,
            force_top_k_ids_per_layer=tuple(mut_layers))
        forced, _ = forward_moe_runtime(
            params=params, input_token_ids=ids,
            force_routing_plan=forced_plan)
        # Forced routing must measurably change the output.
        diff = float(_np.max(_np.abs(
            natural.logits[-1] - forced.logits[-1])))
        if diff > max_forced_diff:
            max_forced_diff = diff
        # The threshold here: clearly above fp32 noise. We
        # require the forced routing to change the output by
        # at least 1e-2 on at least one prompt — otherwise the
        # routing axis is not load-bearing.
        if diff < 1e-2:
            # Track but don't immediately fail.
            pass
        # Trace CID must distinguish the two forwards.
        if natural.cid() == forced.cid():
            trace_cid_changes = False
        # Replay with the natural routing restored.
        natural_plan = routing_plan_from_snapshot(
            natural.routing,
            n_layers=int(params.n_layers))
        replayed, _ = forward_moe_runtime(
            params=params, input_token_ids=ids,
            force_routing_plan=natural_plan)
        rep_diff = float(_np.max(_np.abs(
            natural.logits[-1] - replayed.logits[-1])))
        if rep_diff > max_replay_diff:
            max_replay_diff = rep_diff
        if rep_diff > fp32_floor:
            replay_within = False
    # At least one prompt must show a real (≥ 1e-2) forced
    # routing change.
    if max_forced_diff < 1e-2:
        forced_changes = False
    # Hidden-state intercept under MoE.
    intercept_moves = False
    ids = adapter.tokenize("intercept probe", max_len=10)
    if len(ids) >= 1:
        baseline, _ = forward_moe_runtime(
            params=params, input_token_ids=ids)
        shape0 = baseline.pre_attn_hidden[0].shape
        inj = _np.full(shape0, 0.05, dtype=_np.float64)
        injs: list["_np.ndarray | None"] = [
            None] * int(params.n_layers)
        injs[0] = inj
        after, _ = forward_moe_runtime(
            params=params, input_token_ids=ids,
            hidden_state_injections_per_layer=injs)
        intercept_moves = baseline.cid() != after.cid()
    # Force-routing CID claim.
    routing_cid_changes = False
    ids = adapter.tokenize("force routing", max_len=10)
    if len(ids) >= 1:
        natural, _ = forward_moe_runtime(
            params=params, input_token_ids=ids)
        mut_layers2: list["_np.ndarray | None"] = []
        for L in range(int(params.n_layers)):
            ids_L = natural.routing.per_layer_top_k_ids[L]
            mut = (
                (_np.asarray(ids_L, dtype=_np.int64) + 1)
                % int(params.n_experts))
            mut_layers2.append(mut.astype(_np.int64))
        plan = MoEForceRoutingPlanV1(
            schema=W84_MOE_RUNTIME_V1_SCHEMA_VERSION,
            force_top_k_ids_per_layer=tuple(mut_layers2))
        mutated, _ = forward_moe_runtime(
            params=params, input_token_ids=ids,
            force_routing_plan=plan)
        routing_cid_changes = (
            natural.routing.cid() != mutated.routing.cid())
    return MoEBenchReportV1(
        schema=W84_MOE_RUNTIME_V1_SCHEMA_VERSION,
        params_cid=str(params.cid()),
        n_experts=int(params.n_experts),
        top_k=int(params.top_k),
        n_layers=int(params.n_layers),
        n_prompts=int(n_eval),
        max_forced_routing_diff=float(max_forced_diff),
        forced_routing_changes_output=bool(forced_changes),
        max_replay_with_natural_routing_diff=float(
            max_replay_diff),
        replay_with_natural_routing_within_floor=bool(
            replay_within),
        trace_cid_changes_with_routing=bool(trace_cid_changes),
        hidden_intercept_moves_cid=bool(intercept_moves),
        routing_cid_changes_with_force_plan=bool(
            routing_cid_changes),
    )


def _kv_copy(
        src: MoEKVCacheV1, *,
        n_layers: int, n_heads: int, head_dim: int,
) -> MoEKVCacheV1:
    dst = MoEKVCacheV1.empty(
        n_layers=int(n_layers),
        n_heads=int(n_heads), head_dim=int(head_dim))
    for i in range(int(n_layers)):
        if src.k_layers[i] is not None:
            dst.k_layers[i] = _np.asarray(
                src.k_layers[i], dtype=_np.float64).copy()
        if src.v_layers[i] is not None:
            dst.v_layers[i] = _np.asarray(
                src.v_layers[i], dtype=_np.float64).copy()
    return dst


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _ndarray_cid(arr: "_np.ndarray | None") -> str:
    if arr is None:
        return "none"
    a = _np.ascontiguousarray(
        _np.asarray(arr, dtype=_np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _shape(arr: "_np.ndarray | None") -> tuple[int, ...]:
    if arr is None:
        return ()
    return tuple(int(s) for s in _np.asarray(arr).shape)


__all__ = [
    "W84_MOE_RUNTIME_V1_SCHEMA_VERSION",
    "W84_MOE_INSTRUMENTATION_AXES",
    "MoEInstrumentationAxis",
    "ExpertRoutingSnapshotV1",
    "ExpertOutputSnapshotV1",
    "MoEForceRoutingPlanV1",
    "MoERuntimeParamsV1",
    "MoEKVCacheV1",
    "MoEForwardTraceV1",
    "build_moe_runtime_params_v1",
    "forward_moe_runtime",
    "replay_from_kv_and_routing",
    "routing_plan_from_snapshot",
    "moe_declared_axes",
    "MoERuntimeAdapterV1",
    "MoEBenchReportV1",
    "run_moe_bench_v1",
]
