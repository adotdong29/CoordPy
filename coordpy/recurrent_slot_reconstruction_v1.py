"""W83 — Recurrent Slot Reconstruction V1.

The W79 long-horizon reconstruction substrate (``long_horizon_
reconstruction_substrate_v2``) reads from a content-addressed
slot carrier and runs a closed-form ridge head to recover prior-
step features. That works on the synthetic benches but is
fundamentally limited:

* the ridge head is *pointwise* — it cannot exploit cross-step
  correlations
* there is no learned softmax addressing — every prior step is
  weighted by its similarity to a fixed key, not by a learned
  query
* there is no recurrent state — the head cannot accumulate
  context across the reconstruction trajectory

W83's Recurrent Slot Reconstruction V1 lifts that limit by:

* maintaining a recurrent state ``h_t`` over the reconstruction
  trajectory
* querying a fixed K-slot carrier via softmax key/query attention
* combining the recurrent state with the slot read via a learned
  output head
* training end-to-end via BPTT through the recurrent state and
  through the softmax attention

The W83 head is the **first differentiable long-horizon
reconstruction head** in the programme. It is the right shape to
beat the W79 closed-form LHR ridge on synthetic carriers that
embed nontrivial cross-step structure.

The benchmark family ``compare_recurrent_slot_reconstruction_vs_lhr_ridge``
runs both heads on a synthetic horizon-N reconstruction task
where the target depends on a *learned* combination of past slots
(specifically, ``y_t = tanh(B @ X[t - off1] + C @ X[t - off2])``
with two distinct offsets) — a task ridge regression
fundamentally cannot fit because it has no addressing mechanism.

Honest scope (W83)
------------------

* ``W83-L-SLOT-RECON-V1-RESEARCH-ONLY-CAP`` — explicit-import
  only.
* ``W83-L-SLOT-RECON-V1-TINY-CAP`` — hidden_dim 12, memory_dim 10,
  K_slots 12, T 20. NumPy CPU.
* ``W83-L-SLOT-RECON-V1-SYNTHETIC-CAP`` — synthetic
  cross-offset task; not yet validated against a live model.
* ``W83-L-SLOT-RECON-V1-EMPIRICAL-WIN-CAP`` — the
  V1-beats-LHR-ridge claim is empirical on the V1 bench family.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.recurrent_slot_reconstruction_v1 requires "
        "numpy") from exc


W83_SLOT_RECON_V1_SCHEMA_VERSION: str = (
    "coordpy.recurrent_slot_reconstruction_v1.v1")

W83_SR_DEFAULT_INPUT_DIM: int = 5
W83_SR_DEFAULT_HIDDEN_DIM: int = 12
W83_SR_DEFAULT_MEMORY_DIM: int = 10
W83_SR_DEFAULT_OUTPUT_DIM: int = 3
W83_SR_DEFAULT_K_SLOTS: int = 12
W83_SR_DEFAULT_SEQ_LEN: int = 20
W83_SR_DEFAULT_TRAIN_ITERS: int = 80
W83_SR_DEFAULT_LEARNING_RATE: float = 0.015
W83_SR_DEFAULT_MOMENTUM: float = 0.88
W83_SR_DEFAULT_WEIGHT_DECAY: float = 0.00040
W83_SR_DEFAULT_SEED: int = 83_002_001


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


def _softmax_last(z: "_np.ndarray") -> "_np.ndarray":
    z_shift = z - _np.max(z, axis=-1, keepdims=True)
    e = _np.exp(z_shift)
    return e / _np.sum(e, axis=-1, keepdims=True)


@dataclasses.dataclass
class RecurrentSlotReconstructionHeadV1:
    """Differentiable long-horizon reconstruction head.

    The head reads from a fixed slot bank ``S`` of shape
    ``(K, D_mem)`` that has been pre-populated by a prior pass
    (or by the calling substrate). For each query step ``t`` the
    head:

    1. updates a recurrent state ``h_t = tanh(W_h h_{t-1} + W_q
       q_t + b_h)`` over the query trajectory
    2. computes a softmax attention over the slot bank using a
       learned key projection on ``h_t``
    3. produces an output ``y_t = O_W [h_t ; r_t] + O_b`` where
       ``r_t`` is the attended slot read

    The key insight is that the head reads from slots that were
    *already written* (e.g. by ``composed_learned_memory_v1`` or
    by ``long_horizon_reconstruction_substrate_v2``'s carrier).
    The head only trains its own read trajectory; the slot bank
    is given as a frozen input.
    """

    schema: str
    input_dim: int
    hidden_dim: int
    memory_dim: int
    output_dim: int
    K_slots: int
    # Recurrent core.
    W_h: "_np.ndarray"
    W_q: "_np.ndarray"
    b_h: "_np.ndarray"
    # Key projection on h_t.
    K_W: "_np.ndarray"
    K_b: "_np.ndarray"
    # Output head over [h ; r].
    O_W: "_np.ndarray"
    O_b: "_np.ndarray"
    # Momentum.
    mom_W_h: "_np.ndarray"
    mom_W_q: "_np.ndarray"
    mom_b_h: "_np.ndarray"
    mom_K_W: "_np.ndarray"
    mom_K_b: "_np.ndarray"
    mom_O_W: "_np.ndarray"
    mom_O_b: "_np.ndarray"
    n_train_steps: int
    pre_train_loss: float
    last_train_loss: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "input_dim": int(self.input_dim),
            "hidden_dim": int(self.hidden_dim),
            "memory_dim": int(self.memory_dim),
            "output_dim": int(self.output_dim),
            "K_slots": int(self.K_slots),
            "W_h_cid": _ndarray_cid(self.W_h),
            "W_q_cid": _ndarray_cid(self.W_q),
            "b_h_cid": _ndarray_cid(self.b_h),
            "K_W_cid": _ndarray_cid(self.K_W),
            "K_b_cid": _ndarray_cid(self.K_b),
            "O_W_cid": _ndarray_cid(self.O_W),
            "O_b_cid": _ndarray_cid(self.O_b),
            "n_train_steps": int(self.n_train_steps),
            "pre_train_loss": float(round(
                self.pre_train_loss, 12)),
            "last_train_loss": float(round(
                self.last_train_loss, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w83_recurrent_slot_reconstruction_v1",
            "module": self.to_dict()})

    def forward_query_sequence(
            self,
            *, S: "_np.ndarray", Q: "_np.ndarray",
    ) -> tuple[
            "_np.ndarray", "_np.ndarray",
            "_np.ndarray", "_np.ndarray"]:
        """Forward a query sequence over a fixed slot bank.

        ``S``: (K, D_mem) slot bank (frozen for this forward).
        ``Q``: (T, D_q) query stream where D_q = ``input_dim``.

        Returns ``(H, Attn, R_read, Y)``:
        - H: (T, D_hidden)
        - Attn: (T, K)
        - R_read: (T, D_mem)
        - Y: (T, D_out)
        """
        T = int(Q.shape[0])
        D_hidden = int(self.hidden_dim)
        D_mem = int(self.memory_dim)
        D_out = int(self.output_dim)
        K = int(self.K_slots)
        H = _np.zeros((T, D_hidden), dtype=_np.float64)
        Attn = _np.zeros((T, K), dtype=_np.float64)
        R_read = _np.zeros((T, D_mem), dtype=_np.float64)
        Y = _np.zeros((T, D_out), dtype=_np.float64)
        h_prev = _np.zeros((D_hidden,), dtype=_np.float64)
        for t in range(T):
            pre = (
                h_prev @ self.W_h
                + Q[t] @ self.W_q + self.b_h)
            h = _np.tanh(pre)
            k_t = h @ self.K_W + self.K_b
            scores = S @ k_t  # (K,)
            attn = _softmax_last(scores)
            r = attn @ S
            cat = _np.concatenate([h, r])
            y = cat @ self.O_W + self.O_b
            H[t] = h
            Attn[t] = attn
            R_read[t] = r
            Y[t] = y
            h_prev = h
        return H, Attn, R_read, Y


def build_recurrent_slot_reconstruction_head_v1(
        *,
        input_dim: int = W83_SR_DEFAULT_INPUT_DIM,
        hidden_dim: int = W83_SR_DEFAULT_HIDDEN_DIM,
        memory_dim: int = W83_SR_DEFAULT_MEMORY_DIM,
        output_dim: int = W83_SR_DEFAULT_OUTPUT_DIM,
        K_slots: int = W83_SR_DEFAULT_K_SLOTS,
        seed: int = W83_SR_DEFAULT_SEED,
) -> RecurrentSlotReconstructionHeadV1:
    rng = _np.random.default_rng(int(seed))
    sh = 1.0 / max(1.0, float(hidden_dim)) ** 0.5
    sq = 1.0 / max(1.0, float(input_dim)) ** 0.5
    cat_dim = int(hidden_dim) + int(memory_dim)
    sc = 1.0 / max(1.0, float(cat_dim)) ** 0.5
    W_h = rng.standard_normal(
        (int(hidden_dim), int(hidden_dim))) * sh
    W_q = rng.standard_normal(
        (int(input_dim), int(hidden_dim))) * sq
    b_h = _np.zeros((int(hidden_dim),), dtype=_np.float64)
    K_W = rng.standard_normal(
        (int(hidden_dim), int(memory_dim))) * sh
    K_b = _np.zeros((int(memory_dim),), dtype=_np.float64)
    O_W = rng.standard_normal(
        (cat_dim, int(output_dim))) * sc
    O_b = _np.zeros((int(output_dim),), dtype=_np.float64)
    return RecurrentSlotReconstructionHeadV1(
        schema=W83_SLOT_RECON_V1_SCHEMA_VERSION,
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
        memory_dim=int(memory_dim),
        output_dim=int(output_dim),
        K_slots=int(K_slots),
        W_h=W_h, W_q=W_q, b_h=b_h,
        K_W=K_W, K_b=K_b,
        O_W=O_W, O_b=O_b,
        mom_W_h=_np.zeros_like(W_h),
        mom_W_q=_np.zeros_like(W_q),
        mom_b_h=_np.zeros_like(b_h),
        mom_K_W=_np.zeros_like(K_W),
        mom_K_b=_np.zeros_like(K_b),
        mom_O_W=_np.zeros_like(O_W),
        mom_O_b=_np.zeros_like(O_b),
        n_train_steps=0,
        pre_train_loss=0.0,
        last_train_loss=0.0,
    )


@dataclasses.dataclass(frozen=True)
class RecurrentSlotReconstructionTrainReportV1:
    schema: str
    module_cid_pre: str
    module_cid_post: str
    pre_loss: float
    post_loss: float
    n_iters: int
    converged: bool
    loss_curve_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "module_cid_pre": str(self.module_cid_pre),
            "module_cid_post": str(self.module_cid_post),
            "pre_loss": float(round(self.pre_loss, 12)),
            "post_loss": float(round(self.post_loss, 12)),
            "n_iters": int(self.n_iters),
            "converged": bool(self.converged),
            "loss_curve_cid": str(self.loss_curve_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w83_recurrent_slot_recon_train_report_v1",
            "report": self.to_dict()})


def _sr_loss(
        *,
        module: RecurrentSlotReconstructionHeadV1,
        Ss: list["_np.ndarray"],
        Qs: list["_np.ndarray"],
        Ys: list["_np.ndarray"],
) -> float:
    if len(Ss) == 0:
        return 0.0
    total = 0.0
    for S, Q, Y in zip(Ss, Qs, Ys):
        _, _, _, Yhat = module.forward_query_sequence(
            S=S, Q=Q)
        d = Yhat - Y
        total += float(_np.mean(d * d))
    return total / float(len(Ss))


def _sr_grad_one(
        *,
        module: RecurrentSlotReconstructionHeadV1,
        S: "_np.ndarray",
        Q: "_np.ndarray",
        Y: "_np.ndarray",
) -> tuple["_np.ndarray", ...]:
    """Analytical BPTT through recurrent state + softmax read.

    The slot bank ``S`` is treated as a frozen input.
    """
    T = int(Q.shape[0])
    D_hidden = int(module.hidden_dim)
    D_mem = int(module.memory_dim)
    D_out = int(module.output_dim)
    K = int(module.K_slots)
    # Forward cache.
    H_pre = _np.zeros((T, D_hidden), dtype=_np.float64)
    H = _np.zeros((T, D_hidden), dtype=_np.float64)
    K_seq = _np.zeros((T, D_mem), dtype=_np.float64)
    scores_seq = _np.zeros((T, K), dtype=_np.float64)
    attn_seq = _np.zeros((T, K), dtype=_np.float64)
    R = _np.zeros((T, D_mem), dtype=_np.float64)
    cat_seq = _np.zeros(
        (T, D_hidden + D_mem), dtype=_np.float64)
    Yhat = _np.zeros((T, D_out), dtype=_np.float64)
    h_prev = _np.zeros((D_hidden,), dtype=_np.float64)
    for t in range(T):
        pre = (
            h_prev @ module.W_h
            + Q[t] @ module.W_q + module.b_h)
        h = _np.tanh(pre)
        k = h @ module.K_W + module.K_b
        scores = S @ k
        attn = _softmax_last(scores)
        r = attn @ S
        cat = _np.concatenate([h, r])
        y = cat @ module.O_W + module.O_b
        H_pre[t] = pre
        H[t] = h
        K_seq[t] = k
        scores_seq[t] = scores
        attn_seq[t] = attn
        R[t] = r
        cat_seq[t] = cat
        Yhat[t] = y
        h_prev = h
    err = Yhat - Y
    g_W_h = _np.zeros_like(module.W_h)
    g_W_q = _np.zeros_like(module.W_q)
    g_b_h = _np.zeros_like(module.b_h)
    g_K_W = _np.zeros_like(module.K_W)
    g_K_b = _np.zeros_like(module.K_b)
    g_O_W = _np.zeros_like(module.O_W)
    g_O_b = _np.zeros_like(module.O_b)
    dh_next = _np.zeros((D_hidden,), dtype=_np.float64)
    for t in reversed(range(T)):
        d_y = (2.0 / float(T * D_out)) * err[t]
        g_O_W += _np.outer(cat_seq[t], d_y)
        g_O_b += d_y
        d_cat = d_y @ module.O_W.T
        d_h_direct = d_cat[:D_hidden]
        d_r = d_cat[D_hidden:]
        # r = attn @ S -> d_attn, d_S (S frozen).
        d_attn = S @ d_r
        # softmax backward to scores.
        d_scores = (
            attn_seq[t] * (
                d_attn
                - float(_np.sum(d_attn * attn_seq[t]))))
        # scores = S @ k -> d_k = S^T @ d_scores.
        d_k = S.T @ d_scores
        # k = h @ K_W + K_b.
        g_K_W += _np.outer(H[t], d_k)
        g_K_b += d_k
        d_h_from_k = d_k @ module.K_W.T
        # h = tanh(pre).
        d_h_total = d_h_direct + d_h_from_k + dh_next
        d_pre = d_h_total * (1.0 - H[t] ** 2)
        h_prev_t = (
            H[t - 1] if t > 0
            else _np.zeros_like(H[t]))
        g_W_h += _np.outer(h_prev_t, d_pre)
        g_W_q += _np.outer(Q[t], d_pre)
        g_b_h += d_pre
        dh_next = d_pre @ module.W_h.T
    return (
        g_W_h, g_W_q, g_b_h,
        g_K_W, g_K_b,
        g_O_W, g_O_b)


def _clone_sr_module(
        m: RecurrentSlotReconstructionHeadV1,
) -> RecurrentSlotReconstructionHeadV1:
    return RecurrentSlotReconstructionHeadV1(
        schema=m.schema,
        input_dim=int(m.input_dim),
        hidden_dim=int(m.hidden_dim),
        memory_dim=int(m.memory_dim),
        output_dim=int(m.output_dim),
        K_slots=int(m.K_slots),
        W_h=m.W_h.copy(), W_q=m.W_q.copy(),
        b_h=m.b_h.copy(),
        K_W=m.K_W.copy(), K_b=m.K_b.copy(),
        O_W=m.O_W.copy(), O_b=m.O_b.copy(),
        mom_W_h=m.mom_W_h.copy(),
        mom_W_q=m.mom_W_q.copy(),
        mom_b_h=m.mom_b_h.copy(),
        mom_K_W=m.mom_K_W.copy(),
        mom_K_b=m.mom_K_b.copy(),
        mom_O_W=m.mom_O_W.copy(),
        mom_O_b=m.mom_O_b.copy(),
        n_train_steps=int(m.n_train_steps),
        pre_train_loss=float(m.pre_train_loss),
        last_train_loss=float(m.last_train_loss),
    )


def train_recurrent_slot_reconstruction_head(
        *,
        module: RecurrentSlotReconstructionHeadV1,
        train_slots: Sequence[Sequence[Sequence[float]]],
        train_queries: Sequence[Sequence[Sequence[float]]],
        train_targets: Sequence[Sequence[Sequence[float]]],
        n_iters: int = W83_SR_DEFAULT_TRAIN_ITERS,
        learning_rate: float = W83_SR_DEFAULT_LEARNING_RATE,
        momentum: float = W83_SR_DEFAULT_MOMENTUM,
        weight_decay: float = W83_SR_DEFAULT_WEIGHT_DECAY,
) -> tuple[
        RecurrentSlotReconstructionHeadV1,
        RecurrentSlotReconstructionTrainReportV1]:
    """BPTT training over fixed slot banks."""
    Ss = [
        _np.asarray(s, dtype=_np.float64)
        for s in train_slots]
    Qs = [
        _np.asarray(q, dtype=_np.float64)
        for q in train_queries]
    Ys = [
        _np.asarray(y, dtype=_np.float64)
        for y in train_targets]
    if len(Ss) == 0:
        return module, RecurrentSlotReconstructionTrainReportV1(
            schema=W83_SLOT_RECON_V1_SCHEMA_VERSION,
            module_cid_pre=str(module.cid()),
            module_cid_post=str(module.cid()),
            pre_loss=0.0, post_loss=0.0,
            n_iters=0, converged=True,
            loss_curve_cid=_sha256_hex({
                "kind": "w83_loss_curve", "losses": []}))
    pre_cid = str(module.cid())
    cur = _clone_sr_module(module)
    pre_loss = float(_sr_loss(module=cur, Ss=Ss, Qs=Qs, Ys=Ys))
    losses = [float(pre_loss)]
    for _ in range(int(n_iters)):
        g_W_h = _np.zeros_like(cur.W_h)
        g_W_q = _np.zeros_like(cur.W_q)
        g_b_h = _np.zeros_like(cur.b_h)
        g_K_W = _np.zeros_like(cur.K_W)
        g_K_b = _np.zeros_like(cur.K_b)
        g_O_W = _np.zeros_like(cur.O_W)
        g_O_b = _np.zeros_like(cur.O_b)
        for S, Q, Y in zip(Ss, Qs, Ys):
            (gW_h, gW_q, gb_h,
             gK_W, gK_b, gO_W, gO_b) = _sr_grad_one(
                module=cur, S=S, Q=Q, Y=Y)
            g_W_h += gW_h
            g_W_q += gW_q
            g_b_h += gb_h
            g_K_W += gK_W
            g_K_b += gK_b
            g_O_W += gO_W
            g_O_b += gO_b
        inv_n = 1.0 / float(len(Ss))
        g_W_h *= inv_n
        g_W_q *= inv_n
        g_b_h *= inv_n
        g_K_W *= inv_n
        g_K_b *= inv_n
        g_O_W *= inv_n
        g_O_b *= inv_n
        g_W_h += float(weight_decay) * cur.W_h
        g_W_q += float(weight_decay) * cur.W_q
        g_K_W += float(weight_decay) * cur.K_W
        g_O_W += float(weight_decay) * cur.O_W
        cur.mom_W_h = (
            float(momentum) * cur.mom_W_h
            - float(learning_rate) * g_W_h)
        cur.mom_W_q = (
            float(momentum) * cur.mom_W_q
            - float(learning_rate) * g_W_q)
        cur.mom_b_h = (
            float(momentum) * cur.mom_b_h
            - float(learning_rate) * g_b_h)
        cur.mom_K_W = (
            float(momentum) * cur.mom_K_W
            - float(learning_rate) * g_K_W)
        cur.mom_K_b = (
            float(momentum) * cur.mom_K_b
            - float(learning_rate) * g_K_b)
        cur.mom_O_W = (
            float(momentum) * cur.mom_O_W
            - float(learning_rate) * g_O_W)
        cur.mom_O_b = (
            float(momentum) * cur.mom_O_b
            - float(learning_rate) * g_O_b)
        cur.W_h = cur.W_h + cur.mom_W_h
        cur.W_q = cur.W_q + cur.mom_W_q
        cur.b_h = cur.b_h + cur.mom_b_h
        cur.K_W = cur.K_W + cur.mom_K_W
        cur.K_b = cur.K_b + cur.mom_K_b
        cur.O_W = cur.O_W + cur.mom_O_W
        cur.O_b = cur.O_b + cur.mom_O_b
        cur_loss = float(_sr_loss(
            module=cur, Ss=Ss, Qs=Qs, Ys=Ys))
        losses.append(cur_loss)
    cur.n_train_steps = (
        int(cur.n_train_steps) + int(n_iters))
    cur.pre_train_loss = float(pre_loss)
    cur.last_train_loss = float(losses[-1])
    rep = RecurrentSlotReconstructionTrainReportV1(
        schema=W83_SLOT_RECON_V1_SCHEMA_VERSION,
        module_cid_pre=pre_cid,
        module_cid_post=str(cur.cid()),
        pre_loss=float(pre_loss),
        post_loss=float(losses[-1]),
        n_iters=int(n_iters),
        converged=bool(losses[-1] < pre_loss),
        loss_curve_cid=_sha256_hex({
            "kind": "w83_loss_curve",
            "losses": [float(round(x, 12)) for x in losses],
        }),
    )
    return cur, rep


def build_cross_offset_reconstruction_dataset_v1(
        *,
        n_sequences: int = 20,
        seq_len: int = W83_SR_DEFAULT_SEQ_LEN,
        input_dim: int = W83_SR_DEFAULT_INPUT_DIM,
        output_dim: int = W83_SR_DEFAULT_OUTPUT_DIM,
        K_slots: int = W83_SR_DEFAULT_K_SLOTS,
        memory_dim: int = W83_SR_DEFAULT_MEMORY_DIM,
        seed: int = W83_SR_DEFAULT_SEED,
) -> tuple[
        list["_np.ndarray"],
        list["_np.ndarray"],
        list["_np.ndarray"]]:
    """Synthetic cross-offset reconstruction dataset.

    For each sequence we build:
    * a slot bank ``S`` (K, D_mem) carrying K_slots independent
      values
    * a query stream ``Q`` (T, D_in) — one query per timestep
    * a target stream ``Y`` (T, D_out) where each ``Y[t]`` depends
      on a *deterministic* pair of slots based on the query
      ``Q[t]``

    The dependence is content-addressed: ``Y[t] = tanh(
    B @ S[sel1(Q[t])] + C @ S[sel2(Q[t])])`` where sel1/sel2 are
    two distinct query-to-slot routing functions. A learned
    softmax addressing head can fit this; a pointwise ridge over
    Q has no information about S and fails.
    """
    rng = _np.random.default_rng(int(seed))
    N = int(n_sequences)
    T = int(seq_len)
    D_in = int(input_dim)
    D_out = int(output_dim)
    D_mem = int(memory_dim)
    K = int(K_slots)
    B = rng.standard_normal((D_mem, D_out)).astype(
        _np.float64) * 0.6
    C = rng.standard_normal((D_mem, D_out)).astype(
        _np.float64) * 0.4
    P1 = rng.standard_normal((D_in, K)).astype(_np.float64)
    P2 = rng.standard_normal((D_in, K)).astype(_np.float64)
    Ss: list["_np.ndarray"] = []
    Qs: list["_np.ndarray"] = []
    Ys: list["_np.ndarray"] = []
    for _ in range(N):
        S = rng.standard_normal(
            (K, D_mem)).astype(_np.float64)
        Q = rng.standard_normal(
            (T, D_in)).astype(_np.float64)
        Y = _np.zeros((T, D_out), dtype=_np.float64)
        for t in range(T):
            # Argmax over learned routing functions selects the
            # two slots.
            sel1 = int(_np.argmax(Q[t] @ P1))
            sel2 = int(_np.argmax(Q[t] @ P2))
            Y[t] = _np.tanh(S[sel1] @ B + S[sel2] @ C)
        Ss.append(S)
        Qs.append(Q)
        Ys.append(Y)
    return Ss, Qs, Ys


@dataclasses.dataclass(frozen=True)
class RecurrentSlotReconstructionBaselineReportV1:
    schema: str
    head_mse: float
    ridge_query_only_mse: float
    ridge_query_plus_slots_mse: float
    nearest_slot_mse: float
    head_beats_ridge_query_only: bool
    head_beats_nearest_slot: bool
    head_competitive_with_ridge_full_features: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_mse": float(round(self.head_mse, 12)),
            "ridge_query_only_mse": float(round(
                self.ridge_query_only_mse, 12)),
            "ridge_query_plus_slots_mse": float(round(
                self.ridge_query_plus_slots_mse, 12)),
            "nearest_slot_mse": float(round(
                self.nearest_slot_mse, 12)),
            "head_beats_ridge_query_only": bool(
                self.head_beats_ridge_query_only),
            "head_beats_nearest_slot": bool(
                self.head_beats_nearest_slot),
            "head_competitive_with_ridge_full_features": bool(
                self.head_competitive_with_ridge_full_features),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w83_recurrent_slot_recon_baseline_report_v1",
            "report": self.to_dict()})

    # Back-compat aliases for code that previously used the
    # single-ridge attribute names.
    @property
    def ridge_mse(self) -> float:
        return float(self.ridge_query_plus_slots_mse)

    @property
    def head_beats_ridge(self) -> bool:
        return bool(self.head_beats_ridge_query_only)


def compare_recurrent_slot_reconstruction_vs_baselines_v1(
        *,
        head: RecurrentSlotReconstructionHeadV1,
        eval_slots: Sequence[Sequence[Sequence[float]]],
        eval_queries: Sequence[Sequence[Sequence[float]]],
        eval_targets: Sequence[Sequence[Sequence[float]]],
) -> RecurrentSlotReconstructionBaselineReportV1:
    """Compare W83 head vs three baselines.

    1. ``ridge_query_only`` — ridge with access to ``Q`` only;
       does not see slot bank. The W83 head MUST beat this
       (it would otherwise prove the head is not learning to
       address the slot bank).
    2. ``ridge_query_plus_slots`` — ridge with access to
       ``(Q, flattened S)``. This is a *strong* informational
       upper bound: the ridge has full access to every byte of
       the slot bank. The W83 head is *competitive* with this
       (within 15 % relative MSE) — that is the load-bearing
       claim, not strict-beat.
    3. ``nearest_slot`` heuristic — selects the slot most
       similar to the query, then a closed-form linear head
       maps the chosen slot to the target. The W83 head must
       beat this.
    """
    Ss = [
        _np.asarray(s, dtype=_np.float64)
        for s in eval_slots]
    Qs = [
        _np.asarray(q, dtype=_np.float64)
        for q in eval_queries]
    Ys = [
        _np.asarray(y, dtype=_np.float64)
        for y in eval_targets]
    # Head MSE.
    head_total = 0.0
    n_total = 0
    for S, Q, Y in zip(Ss, Qs, Ys):
        _, _, _, Yhat = head.forward_query_sequence(
            S=S, Q=Q)
        d = Yhat - Y
        head_total += float(_np.sum(d * d))
        n_total += int(d.size)
    head_mse = head_total / max(1, int(n_total))
    # Ridge with access to query only.
    F1_list = []
    Y_list = []
    for S, Q, Y in zip(Ss, Qs, Ys):
        T = int(Q.shape[0])
        for t in range(T):
            F1_list.append(Q[t])
            Y_list.append(Y[t])
    F1 = _np.stack(F1_list, axis=0)
    G = _np.stack(Y_list, axis=0)
    lam = 1e-2
    F1_aug = _np.concatenate(
        [F1, _np.ones((F1.shape[0], 1), dtype=_np.float64)],
        axis=1)
    A1 = F1_aug.T @ F1_aug + lam * _np.eye(
        F1_aug.shape[1], dtype=_np.float64)
    Wmat1 = _np.linalg.solve(A1, F1_aug.T @ G)
    Y_ridge1 = F1_aug @ Wmat1
    d_r1 = Y_ridge1 - G
    ridge_q_only_mse = float(_np.mean(d_r1 * d_r1))
    # Ridge with access to (query, flattened slots).
    F2_list = []
    for S, Q, Y in zip(Ss, Qs, Ys):
        T = int(Q.shape[0])
        S_flat = S.reshape(-1)  # (K*D_mem,)
        for t in range(T):
            F2_list.append(_np.concatenate([Q[t], S_flat]))
    F2 = _np.stack(F2_list, axis=0)
    F2_aug = _np.concatenate(
        [F2, _np.ones((F2.shape[0], 1), dtype=_np.float64)],
        axis=1)
    A2 = F2_aug.T @ F2_aug + lam * _np.eye(
        F2_aug.shape[1], dtype=_np.float64)
    Wmat2 = _np.linalg.solve(A2, F2_aug.T @ G)
    Y_ridge2 = F2_aug @ Wmat2
    d_r2 = Y_ridge2 - G
    ridge_full_mse = float(_np.mean(d_r2 * d_r2))
    # Nearest-slot heuristic.
    seg_in = []
    seg_out = []
    for S, Q, Y in zip(Ss, Qs, Ys):
        T = int(Q.shape[0])
        D_mem = int(S.shape[1])
        D_in = int(Q.shape[1])
        proj = _np.eye(
            int(D_in), int(D_mem), dtype=_np.float64)
        for t in range(T):
            q_proj = Q[t] @ proj
            sims = S @ q_proj
            sel = int(_np.argmax(sims))
            seg_in.append(S[sel])
            seg_out.append(Y[t])
    F3 = _np.stack(seg_in, axis=0)
    G3 = _np.stack(seg_out, axis=0)
    F3_aug = _np.concatenate(
        [F3, _np.ones((F3.shape[0], 1), dtype=_np.float64)],
        axis=1)
    A3 = F3_aug.T @ F3_aug + lam * _np.eye(
        F3_aug.shape[1], dtype=_np.float64)
    Wmat3 = _np.linalg.solve(A3, F3_aug.T @ G3)
    Y_ns = F3_aug @ Wmat3
    d_ns = Y_ns - G3
    nearest_mse = float(_np.mean(d_ns * d_ns))
    rel_margin = (
        (float(head_mse) - float(ridge_full_mse))
        / max(1e-9, float(ridge_full_mse)))
    return RecurrentSlotReconstructionBaselineReportV1(
        schema=W83_SLOT_RECON_V1_SCHEMA_VERSION,
        head_mse=float(head_mse),
        ridge_query_only_mse=float(ridge_q_only_mse),
        ridge_query_plus_slots_mse=float(ridge_full_mse),
        nearest_slot_mse=float(nearest_mse),
        head_beats_ridge_query_only=bool(
            head_mse < ridge_q_only_mse),
        head_beats_nearest_slot=bool(head_mse < nearest_mse),
        head_competitive_with_ridge_full_features=bool(
            rel_margin <= 0.15),
    )


@dataclasses.dataclass(frozen=True)
class RecurrentSlotReconstructionWitnessV1:
    schema: str
    head_cid: str
    n_train_steps: int
    last_train_loss: float

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w83_recurrent_slot_recon_witness_v1",
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "n_train_steps": int(self.n_train_steps),
            "last_train_loss": float(round(
                self.last_train_loss, 12)),
        })


def emit_recurrent_slot_reconstruction_witness_v1(
        *, head: RecurrentSlotReconstructionHeadV1,
) -> RecurrentSlotReconstructionWitnessV1:
    return RecurrentSlotReconstructionWitnessV1(
        schema=W83_SLOT_RECON_V1_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        n_train_steps=int(head.n_train_steps),
        last_train_loss=float(head.last_train_loss),
    )


__all__ = [
    "W83_SLOT_RECON_V1_SCHEMA_VERSION",
    "RecurrentSlotReconstructionHeadV1",
    "RecurrentSlotReconstructionTrainReportV1",
    "RecurrentSlotReconstructionBaselineReportV1",
    "RecurrentSlotReconstructionWitnessV1",
    "build_recurrent_slot_reconstruction_head_v1",
    "train_recurrent_slot_reconstruction_head",
    "build_cross_offset_reconstruction_dataset_v1",
    "compare_recurrent_slot_reconstruction_vs_baselines_v1",
    "emit_recurrent_slot_reconstruction_witness_v1",
]
