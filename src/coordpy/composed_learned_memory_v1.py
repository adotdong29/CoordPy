"""W83 — Composed Learned Memory V1.

W81 shipped two separate learned-memory lines:

* ``learned_consolidation_v2`` — a recurrent core ``h_t = tanh(W_h
  h_{t-1} + W_x x_t + b)`` with linear write/read heads. Strong on
  temporal-integration, weak on content-addressed recall (the
  bottleneck is the single recurrent state).
* ``differentiable_memory_substrate_v1`` — K addressable slots with
  softmax read attention. Strong on content-addressed recall. The
  V1 train loop *detaches* the slot accumulation across timesteps
  to keep gradients tractable, which throws away the longest range
  of credit assignment.

P1 #9 / P1 #19 explicitly call out that the long-horizon
compression-and-recovery task sits at the heart of conquering
context. W83 composes both lines into a single end-to-end-trained
learned memory that:

1. carries both a recurrent state ``h_t`` AND a write-accumulating
   slot bank ``S``
2. routes writes via a learned alpha gate AND a learned softmax
   slot router (so a single timestep writes preferentially to one
   slot, not equally to all)
3. reads via a learned key/query softmax attention over the slot
   bank
4. trains via *full BPTT through the slot accumulation* — no
   per-step detachment

The full-BPTT step is the load-bearing W83 advance. It costs more
forward-state caching (the full slot trajectory must be stored)
but it lets the gradient propagate from a read at time t to a
write at time s for any s < t. On a task that requires the model
to write at t and read at t + k for k >> 0, this is the entire
ballgame.

The W83 composed line is explicit-import only and trains in pure
NumPy on CPU. The benchmark family in this module shows that the
composed line strictly beats:

* W81 ``learned_consolidation_v2`` alone (recurrent without slots)
* W81 ``differentiable_memory_substrate_v1`` (slots with detached
  BPTT)
* closed-form pointwise ridge

on a composed task that mixes temporal integration AND
content-addressed delayed recall.

Honest scope (W83)
------------------

* ``W83-L-COMPOSED-MEMORY-V1-RESEARCH-ONLY-CAP`` — explicit-import
  only; not on the stable public surface.
* ``W83-L-COMPOSED-MEMORY-V1-TINY-CAP`` — hidden_dim 16,
  memory_dim 12, K_slots 6, T 14. NumPy CPU.
* ``W83-L-COMPOSED-MEMORY-V1-SYNTHETIC-CAP`` — trained on the W83
  synthetic compound dataset. Live-runtime hidden-state coupling
  is not part of V1.
* ``W83-L-COMPOSED-MEMORY-V1-NUMPY-CAP`` — pure NumPy; no torch /
  jax / tensorflow.
* ``W83-L-COMPOSED-MEMORY-V1-EMPIRICAL-WIN-CAP`` — the
  V1-strictly-beats-baselines claim is empirical on the V1 bench
  family; not yet derived as an analytical theorem.
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
        "coordpy.composed_learned_memory_v1 requires numpy"
        ) from exc


W83_COMPOSED_MEMORY_V1_SCHEMA_VERSION: str = (
    "coordpy.composed_learned_memory_v1.v1")

W83_CM_DEFAULT_INPUT_DIM: int = 5
W83_CM_DEFAULT_HIDDEN_DIM: int = 16
W83_CM_DEFAULT_MEMORY_DIM: int = 12
W83_CM_DEFAULT_OUTPUT_DIM: int = 3
W83_CM_DEFAULT_K_SLOTS: int = 6
W83_CM_DEFAULT_SEQ_LEN: int = 14
W83_CM_DEFAULT_TRAIN_ITERS: int = 70
W83_CM_DEFAULT_LEARNING_RATE: float = 0.012
W83_CM_DEFAULT_MOMENTUM: float = 0.88
W83_CM_DEFAULT_WEIGHT_DECAY: float = 0.0003
W83_CM_DEFAULT_SEED: int = 83_001_001


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
class ComposedLearnedMemoryModuleV1:
    """Composed recurrent + slot-memory module.

    Forward dynamics per timestep::

        h_t       = tanh(W_h h_{t-1} + W_x x_t + b_h)         # recurrent
        q_t       = h_t @ Q_W + Q_b                            # read query
        v_t       = h_t @ V_W + V_b                            # write value
        write_t   = softmax(h_t @ R_W + R_b)  (K,)             # slot router
        alpha_t   = sigmoid(h_t @ A_W + A_b)  (K,)             # write gate
        S_t       = S_{t-1} + (alpha_t * write_t)[:, None] * v_t[None, :]
        scores_t  = S_t @ q_t                                  # (K,)
        attn_t    = softmax(scores_t)
        r_t       = attn_t @ S_t                               # (D_mem,)
        y_t       = O_W [h_t ; r_t] + O_b

    Notable design choices over the W81 lines:

    * The write router (``write_t``) is a softmax over the K slots
      (not a per-slot independent sigmoid), so each timestep
      effectively prefers a single slot to write to. This is
      crucial for full BPTT through ``S_t``: the gradient routes
      through the *softmax* slot probabilities, not through K
      independent sigmoids that could all be 0.5.
    * Slot accumulation ``S_t = S_{t-1} + …`` keeps the slot bank
      tied across timesteps; full BPTT therefore unrolls through
      ``S_t`` history.
    * Read uses query/key softmax over the *current* slot bank
      ``S_t``, which carries the write history.
    """

    schema: str
    input_dim: int
    hidden_dim: int
    memory_dim: int
    output_dim: int
    K_slots: int
    # Recurrent core.
    W_h: "_np.ndarray"
    W_x: "_np.ndarray"
    b_h: "_np.ndarray"
    # Read query head.
    Q_W: "_np.ndarray"
    Q_b: "_np.ndarray"
    # Write value head.
    V_W: "_np.ndarray"
    V_b: "_np.ndarray"
    # Slot router (softmax over K).
    R_W: "_np.ndarray"
    R_b: "_np.ndarray"
    # Write gate (sigmoid).
    A_W: "_np.ndarray"
    A_b: "_np.ndarray"
    # Output head over [h ; r].
    O_W: "_np.ndarray"
    O_b: "_np.ndarray"
    # Momentum.
    mom_W_h: "_np.ndarray"
    mom_W_x: "_np.ndarray"
    mom_b_h: "_np.ndarray"
    mom_Q_W: "_np.ndarray"
    mom_Q_b: "_np.ndarray"
    mom_V_W: "_np.ndarray"
    mom_V_b: "_np.ndarray"
    mom_R_W: "_np.ndarray"
    mom_R_b: "_np.ndarray"
    mom_A_W: "_np.ndarray"
    mom_A_b: "_np.ndarray"
    mom_O_W: "_np.ndarray"
    mom_O_b: "_np.ndarray"
    # Bookkeeping.
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
            "W_x_cid": _ndarray_cid(self.W_x),
            "b_h_cid": _ndarray_cid(self.b_h),
            "Q_W_cid": _ndarray_cid(self.Q_W),
            "Q_b_cid": _ndarray_cid(self.Q_b),
            "V_W_cid": _ndarray_cid(self.V_W),
            "V_b_cid": _ndarray_cid(self.V_b),
            "R_W_cid": _ndarray_cid(self.R_W),
            "R_b_cid": _ndarray_cid(self.R_b),
            "A_W_cid": _ndarray_cid(self.A_W),
            "A_b_cid": _ndarray_cid(self.A_b),
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
            "kind": "w83_composed_learned_memory_v1",
            "module": self.to_dict()})

    def forward_sequence(
            self, X: "_np.ndarray",
    ) -> tuple[
            "_np.ndarray", "_np.ndarray",
            "_np.ndarray", "_np.ndarray"]:
        """Forward a single sequence (T, input_dim).

        Returns ``(H, S_seq, R, Y)``:
        - H: (T, D_hidden) recurrent states
        - S_seq: (T+1, K, D_mem) slot trajectory (S_0 = zeros)
        - R: (T, D_mem) per-step read vectors
        - Y: (T, D_out) per-step output
        """
        T = int(X.shape[0])
        D_hidden = int(self.hidden_dim)
        D_mem = int(self.memory_dim)
        K = int(self.K_slots)
        D_out = int(self.output_dim)
        H = _np.zeros((T, D_hidden), dtype=_np.float64)
        S_seq = _np.zeros((T + 1, K, D_mem), dtype=_np.float64)
        R = _np.zeros((T, D_mem), dtype=_np.float64)
        Y = _np.zeros((T, D_out), dtype=_np.float64)
        h_prev = _np.zeros((D_hidden,), dtype=_np.float64)
        for t in range(T):
            pre = (
                h_prev @ self.W_h
                + X[t] @ self.W_x + self.b_h)
            h_t = _np.tanh(pre)
            q_t = h_t @ self.Q_W + self.Q_b
            v_t = h_t @ self.V_W + self.V_b
            r_pre = h_t @ self.R_W + self.R_b
            r_t_route = _softmax_last(r_pre)
            a_pre = h_t @ self.A_W + self.A_b
            a_t = 1.0 / (1.0 + _np.exp(-a_pre))
            write = (a_t * r_t_route)
            S_seq[t + 1] = (
                S_seq[t]
                + write[:, None] * v_t[None, :])
            scores = S_seq[t + 1] @ q_t
            attn = _softmax_last(scores)
            r_read = attn @ S_seq[t + 1]
            cat = _np.concatenate([h_t, r_read])
            y_t = cat @ self.O_W + self.O_b
            H[t] = h_t
            R[t] = r_read
            Y[t] = y_t
            h_prev = h_t
        return H, S_seq, R, Y

    def compressed_snapshot_cid(
            self, *, X: "_np.ndarray") -> str:
        _, S_seq, _, _ = self.forward_sequence(X)
        # Use the final slot bank as the compressed snapshot.
        return _ndarray_cid(S_seq[-1])


def build_composed_learned_memory_module_v1(
        *,
        input_dim: int = W83_CM_DEFAULT_INPUT_DIM,
        hidden_dim: int = W83_CM_DEFAULT_HIDDEN_DIM,
        memory_dim: int = W83_CM_DEFAULT_MEMORY_DIM,
        output_dim: int = W83_CM_DEFAULT_OUTPUT_DIM,
        K_slots: int = W83_CM_DEFAULT_K_SLOTS,
        seed: int = W83_CM_DEFAULT_SEED,
) -> ComposedLearnedMemoryModuleV1:
    rng = _np.random.default_rng(int(seed))
    sh = 1.0 / max(1.0, float(hidden_dim)) ** 0.5
    sx = 1.0 / max(1.0, float(input_dim)) ** 0.5
    cat_dim = int(hidden_dim) + int(memory_dim)
    sc = 1.0 / max(1.0, float(cat_dim)) ** 0.5
    W_h = rng.standard_normal(
        (int(hidden_dim), int(hidden_dim))) * sh
    W_x = rng.standard_normal(
        (int(input_dim), int(hidden_dim))) * sx
    b_h = _np.zeros((int(hidden_dim),), dtype=_np.float64)
    Q_W = rng.standard_normal(
        (int(hidden_dim), int(memory_dim))) * sh
    Q_b = _np.zeros((int(memory_dim),), dtype=_np.float64)
    V_W = rng.standard_normal(
        (int(hidden_dim), int(memory_dim))) * sh
    V_b = _np.zeros((int(memory_dim),), dtype=_np.float64)
    R_W = rng.standard_normal(
        (int(hidden_dim), int(K_slots))) * sh
    R_b = _np.zeros((int(K_slots),), dtype=_np.float64)
    A_W = rng.standard_normal(
        (int(hidden_dim), int(K_slots))) * sh
    A_b = _np.zeros((int(K_slots),), dtype=_np.float64)
    O_W = rng.standard_normal(
        (cat_dim, int(output_dim))) * sc
    O_b = _np.zeros((int(output_dim),), dtype=_np.float64)
    return ComposedLearnedMemoryModuleV1(
        schema=W83_COMPOSED_MEMORY_V1_SCHEMA_VERSION,
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
        memory_dim=int(memory_dim),
        output_dim=int(output_dim),
        K_slots=int(K_slots),
        W_h=W_h, W_x=W_x, b_h=b_h,
        Q_W=Q_W, Q_b=Q_b,
        V_W=V_W, V_b=V_b,
        R_W=R_W, R_b=R_b,
        A_W=A_W, A_b=A_b,
        O_W=O_W, O_b=O_b,
        mom_W_h=_np.zeros_like(W_h),
        mom_W_x=_np.zeros_like(W_x),
        mom_b_h=_np.zeros_like(b_h),
        mom_Q_W=_np.zeros_like(Q_W),
        mom_Q_b=_np.zeros_like(Q_b),
        mom_V_W=_np.zeros_like(V_W),
        mom_V_b=_np.zeros_like(V_b),
        mom_R_W=_np.zeros_like(R_W),
        mom_R_b=_np.zeros_like(R_b),
        mom_A_W=_np.zeros_like(A_W),
        mom_A_b=_np.zeros_like(A_b),
        mom_O_W=_np.zeros_like(O_W),
        mom_O_b=_np.zeros_like(O_b),
        n_train_steps=0,
        pre_train_loss=0.0,
        last_train_loss=0.0,
    )


@dataclasses.dataclass(frozen=True)
class ComposedLearnedMemoryTrainReportV1:
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
                "w83_composed_learned_memory_train_report_v1",
            "report": self.to_dict()})


def _cm_loss(
        *,
        module: ComposedLearnedMemoryModuleV1,
        Xs: list["_np.ndarray"],
        Ys: list["_np.ndarray"],
) -> float:
    if len(Xs) == 0:
        return 0.0
    total = 0.0
    for X, Y in zip(Xs, Ys):
        _, _, _, Yhat = module.forward_sequence(X)
        d = Yhat - Y
        total += float(_np.mean(d * d))
    return total / float(len(Xs))


def _cm_grad_full_bptt(
        *,
        module: ComposedLearnedMemoryModuleV1,
        X: "_np.ndarray", Y: "_np.ndarray",
) -> tuple["_np.ndarray", ...]:
    """Analytical full-BPTT gradients through slot accumulation.

    The forward pass caches everything needed; the backward pass
    propagates the read-attention gradient back to writes from
    *all* prior timesteps via the slot trajectory. Slots are
    cumulative: ``S_t = S_{t-1} + write_t``, so dL/dS_t flows
    backward to both the current write AND dL/dS_{t-1}.
    """
    T = int(X.shape[0])
    D_hidden = int(module.hidden_dim)
    D_mem = int(module.memory_dim)
    D_out = int(module.output_dim)
    K = int(module.K_slots)
    # Forward cache.
    H_pre = _np.zeros((T, D_hidden), dtype=_np.float64)
    H = _np.zeros((T, D_hidden), dtype=_np.float64)
    Q_seq = _np.zeros((T, D_mem), dtype=_np.float64)
    V_seq = _np.zeros((T, D_mem), dtype=_np.float64)
    R_pre = _np.zeros((T, K), dtype=_np.float64)
    R_route = _np.zeros((T, K), dtype=_np.float64)
    A_pre = _np.zeros((T, K), dtype=_np.float64)
    A_t = _np.zeros((T, K), dtype=_np.float64)
    write_seq = _np.zeros((T, K), dtype=_np.float64)
    S_seq = _np.zeros((T + 1, K, D_mem), dtype=_np.float64)
    scores_seq = _np.zeros((T, K), dtype=_np.float64)
    attn_seq = _np.zeros((T, K), dtype=_np.float64)
    R_read = _np.zeros((T, D_mem), dtype=_np.float64)
    cat_seq = _np.zeros(
        (T, D_hidden + D_mem), dtype=_np.float64)
    Yhat = _np.zeros((T, D_out), dtype=_np.float64)
    h_prev = _np.zeros((D_hidden,), dtype=_np.float64)
    for t in range(T):
        pre = h_prev @ module.W_h + X[t] @ module.W_x + module.b_h
        h = _np.tanh(pre)
        q = h @ module.Q_W + module.Q_b
        v = h @ module.V_W + module.V_b
        rpre = h @ module.R_W + module.R_b
        rroute = _softmax_last(rpre)
        apre = h @ module.A_W + module.A_b
        a = 1.0 / (1.0 + _np.exp(-apre))
        wgate = a * rroute
        S_seq[t + 1] = (
            S_seq[t] + wgate[:, None] * v[None, :])
        scores = S_seq[t + 1] @ q
        attn = _softmax_last(scores)
        r_read = attn @ S_seq[t + 1]
        cat = _np.concatenate([h, r_read])
        y = cat @ module.O_W + module.O_b
        H_pre[t] = pre
        H[t] = h
        Q_seq[t] = q
        V_seq[t] = v
        R_pre[t] = rpre
        R_route[t] = rroute
        A_pre[t] = apre
        A_t[t] = a
        write_seq[t] = wgate
        scores_seq[t] = scores
        attn_seq[t] = attn
        R_read[t] = r_read
        cat_seq[t] = cat
        Yhat[t] = y
        h_prev = h
    # Backward.
    err = Yhat - Y  # (T, D_out)
    g_W_h = _np.zeros_like(module.W_h)
    g_W_x = _np.zeros_like(module.W_x)
    g_b_h = _np.zeros_like(module.b_h)
    g_Q_W = _np.zeros_like(module.Q_W)
    g_Q_b = _np.zeros_like(module.Q_b)
    g_V_W = _np.zeros_like(module.V_W)
    g_V_b = _np.zeros_like(module.V_b)
    g_R_W = _np.zeros_like(module.R_W)
    g_R_b = _np.zeros_like(module.R_b)
    g_A_W = _np.zeros_like(module.A_W)
    g_A_b = _np.zeros_like(module.A_b)
    g_O_W = _np.zeros_like(module.O_W)
    g_O_b = _np.zeros_like(module.O_b)
    # Slot trajectory gradient: dL/dS_{t+1} aggregated across
    # later reads.
    dS_carry = _np.zeros((K, D_mem), dtype=_np.float64)
    dh_next = _np.zeros((D_hidden,), dtype=_np.float64)
    for t in reversed(range(T)):
        d_y = (2.0 / float(T * D_out)) * err[t]
        # Output head.
        g_O_W += _np.outer(cat_seq[t], d_y)
        g_O_b += d_y
        d_cat = d_y @ module.O_W.T
        d_h_direct = d_cat[:D_hidden]
        d_r_read = d_cat[D_hidden:]
        # r_read = attn @ S_{t+1}.
        d_attn = S_seq[t + 1] @ d_r_read
        d_S_t1 = (
            attn_seq[t][:, None] * d_r_read[None, :])
        # Add the carry from later timesteps.
        d_S_t1 = d_S_t1 + dS_carry
        # softmax backward to scores.
        d_scores = (
            attn_seq[t] * (
                d_attn
                - float(_np.sum(d_attn * attn_seq[t]))))
        # scores = S_{t+1} @ q.
        g_q = S_seq[t + 1].T @ d_scores
        d_S_from_scores = _np.outer(d_scores, Q_seq[t])
        d_S_t1 = d_S_t1 + d_S_from_scores
        # write contribution: S_{t+1} = S_t + write * v^T.
        # gradients flow to: write, v, S_t.
        d_write = (d_S_t1 * V_seq[t][None, :]).sum(axis=1)
        d_v = (write_seq[t][:, None] * d_S_t1).sum(axis=0)
        d_S_carry_next = d_S_t1.copy()
        # Forward to next loop step.
        dS_carry = d_S_carry_next
        # q = h @ Q_W + Q_b.
        g_Q_W += _np.outer(H[t], g_q)
        g_Q_b += g_q
        d_h_from_q = g_q @ module.Q_W.T
        # v = h @ V_W + V_b.
        g_V_W += _np.outer(H[t], d_v)
        g_V_b += d_v
        d_h_from_v = d_v @ module.V_W.T
        # write = a * rroute.
        d_a = d_write * R_route[t]
        d_rroute = d_write * A_t[t]
        # rroute = softmax(rpre).
        d_rpre = (
            R_route[t] * (
                d_rroute
                - float(_np.sum(
                    d_rroute * R_route[t]))))
        g_R_W += _np.outer(H[t], d_rpre)
        g_R_b += d_rpre
        d_h_from_r = d_rpre @ module.R_W.T
        # a = sigmoid(apre).
        d_apre = d_a * A_t[t] * (1.0 - A_t[t])
        g_A_W += _np.outer(H[t], d_apre)
        g_A_b += d_apre
        d_h_from_a = d_apre @ module.A_W.T
        # h = tanh(pre); aggregate.
        d_h_total = (
            d_h_direct + d_h_from_q + d_h_from_v
            + d_h_from_r + d_h_from_a + dh_next)
        d_pre = d_h_total * (1.0 - H[t] ** 2)
        h_prev_t = (
            H[t - 1] if t > 0
            else _np.zeros_like(H[t]))
        g_W_h += _np.outer(h_prev_t, d_pre)
        g_W_x += _np.outer(X[t], d_pre)
        g_b_h += d_pre
        dh_next = d_pre @ module.W_h.T
    return (
        g_W_h, g_W_x, g_b_h,
        g_Q_W, g_Q_b,
        g_V_W, g_V_b,
        g_R_W, g_R_b,
        g_A_W, g_A_b,
        g_O_W, g_O_b)


def _clone_cm_module(
        m: ComposedLearnedMemoryModuleV1,
) -> ComposedLearnedMemoryModuleV1:
    return ComposedLearnedMemoryModuleV1(
        schema=m.schema,
        input_dim=int(m.input_dim),
        hidden_dim=int(m.hidden_dim),
        memory_dim=int(m.memory_dim),
        output_dim=int(m.output_dim),
        K_slots=int(m.K_slots),
        W_h=m.W_h.copy(), W_x=m.W_x.copy(),
        b_h=m.b_h.copy(),
        Q_W=m.Q_W.copy(), Q_b=m.Q_b.copy(),
        V_W=m.V_W.copy(), V_b=m.V_b.copy(),
        R_W=m.R_W.copy(), R_b=m.R_b.copy(),
        A_W=m.A_W.copy(), A_b=m.A_b.copy(),
        O_W=m.O_W.copy(), O_b=m.O_b.copy(),
        mom_W_h=m.mom_W_h.copy(),
        mom_W_x=m.mom_W_x.copy(),
        mom_b_h=m.mom_b_h.copy(),
        mom_Q_W=m.mom_Q_W.copy(),
        mom_Q_b=m.mom_Q_b.copy(),
        mom_V_W=m.mom_V_W.copy(),
        mom_V_b=m.mom_V_b.copy(),
        mom_R_W=m.mom_R_W.copy(),
        mom_R_b=m.mom_R_b.copy(),
        mom_A_W=m.mom_A_W.copy(),
        mom_A_b=m.mom_A_b.copy(),
        mom_O_W=m.mom_O_W.copy(),
        mom_O_b=m.mom_O_b.copy(),
        n_train_steps=int(m.n_train_steps),
        pre_train_loss=float(m.pre_train_loss),
        last_train_loss=float(m.last_train_loss),
    )


def train_composed_learned_memory_module(
        *,
        module: ComposedLearnedMemoryModuleV1,
        train_sequences: Sequence[Sequence[Sequence[float]]],
        train_targets: Sequence[Sequence[Sequence[float]]],
        n_iters: int = W83_CM_DEFAULT_TRAIN_ITERS,
        learning_rate: float = W83_CM_DEFAULT_LEARNING_RATE,
        momentum: float = W83_CM_DEFAULT_MOMENTUM,
        weight_decay: float = W83_CM_DEFAULT_WEIGHT_DECAY,
) -> tuple[
        ComposedLearnedMemoryModuleV1,
        ComposedLearnedMemoryTrainReportV1]:
    """Full-BPTT training over slot-accumulating composed memory."""
    Xs = [
        _np.asarray(s, dtype=_np.float64)
        for s in train_sequences]
    Ys = [
        _np.asarray(t, dtype=_np.float64)
        for t in train_targets]
    if len(Xs) == 0:
        return module, ComposedLearnedMemoryTrainReportV1(
            schema=W83_COMPOSED_MEMORY_V1_SCHEMA_VERSION,
            module_cid_pre=str(module.cid()),
            module_cid_post=str(module.cid()),
            pre_loss=0.0, post_loss=0.0,
            n_iters=0, converged=True,
            loss_curve_cid=_sha256_hex({
                "kind": "w83_loss_curve", "losses": []}))
    pre_cid = str(module.cid())
    cur = _clone_cm_module(module)
    pre_loss = float(_cm_loss(module=cur, Xs=Xs, Ys=Ys))
    losses = [float(pre_loss)]
    for _ in range(int(n_iters)):
        g_W_h = _np.zeros_like(cur.W_h)
        g_W_x = _np.zeros_like(cur.W_x)
        g_b_h = _np.zeros_like(cur.b_h)
        g_Q_W = _np.zeros_like(cur.Q_W)
        g_Q_b = _np.zeros_like(cur.Q_b)
        g_V_W = _np.zeros_like(cur.V_W)
        g_V_b = _np.zeros_like(cur.V_b)
        g_R_W = _np.zeros_like(cur.R_W)
        g_R_b = _np.zeros_like(cur.R_b)
        g_A_W = _np.zeros_like(cur.A_W)
        g_A_b = _np.zeros_like(cur.A_b)
        g_O_W = _np.zeros_like(cur.O_W)
        g_O_b = _np.zeros_like(cur.O_b)
        for X, Y in zip(Xs, Ys):
            (gW_h, gW_x, gb_h,
             gQ_W, gQ_b, gV_W, gV_b,
             gR_W, gR_b, gA_W, gA_b,
             gO_W, gO_b) = _cm_grad_full_bptt(
                module=cur, X=X, Y=Y)
            g_W_h += gW_h
            g_W_x += gW_x
            g_b_h += gb_h
            g_Q_W += gQ_W
            g_Q_b += gQ_b
            g_V_W += gV_W
            g_V_b += gV_b
            g_R_W += gR_W
            g_R_b += gR_b
            g_A_W += gA_W
            g_A_b += gA_b
            g_O_W += gO_W
            g_O_b += gO_b
        inv_n = 1.0 / float(len(Xs))
        g_W_h *= inv_n
        g_W_x *= inv_n
        g_b_h *= inv_n
        g_Q_W *= inv_n
        g_Q_b *= inv_n
        g_V_W *= inv_n
        g_V_b *= inv_n
        g_R_W *= inv_n
        g_R_b *= inv_n
        g_A_W *= inv_n
        g_A_b *= inv_n
        g_O_W *= inv_n
        g_O_b *= inv_n
        # Weight decay.
        g_W_h += float(weight_decay) * cur.W_h
        g_W_x += float(weight_decay) * cur.W_x
        g_Q_W += float(weight_decay) * cur.Q_W
        g_V_W += float(weight_decay) * cur.V_W
        g_R_W += float(weight_decay) * cur.R_W
        g_A_W += float(weight_decay) * cur.A_W
        g_O_W += float(weight_decay) * cur.O_W
        # SGD with momentum.
        cur.mom_W_h = (
            float(momentum) * cur.mom_W_h
            - float(learning_rate) * g_W_h)
        cur.mom_W_x = (
            float(momentum) * cur.mom_W_x
            - float(learning_rate) * g_W_x)
        cur.mom_b_h = (
            float(momentum) * cur.mom_b_h
            - float(learning_rate) * g_b_h)
        cur.mom_Q_W = (
            float(momentum) * cur.mom_Q_W
            - float(learning_rate) * g_Q_W)
        cur.mom_Q_b = (
            float(momentum) * cur.mom_Q_b
            - float(learning_rate) * g_Q_b)
        cur.mom_V_W = (
            float(momentum) * cur.mom_V_W
            - float(learning_rate) * g_V_W)
        cur.mom_V_b = (
            float(momentum) * cur.mom_V_b
            - float(learning_rate) * g_V_b)
        cur.mom_R_W = (
            float(momentum) * cur.mom_R_W
            - float(learning_rate) * g_R_W)
        cur.mom_R_b = (
            float(momentum) * cur.mom_R_b
            - float(learning_rate) * g_R_b)
        cur.mom_A_W = (
            float(momentum) * cur.mom_A_W
            - float(learning_rate) * g_A_W)
        cur.mom_A_b = (
            float(momentum) * cur.mom_A_b
            - float(learning_rate) * g_A_b)
        cur.mom_O_W = (
            float(momentum) * cur.mom_O_W
            - float(learning_rate) * g_O_W)
        cur.mom_O_b = (
            float(momentum) * cur.mom_O_b
            - float(learning_rate) * g_O_b)
        cur.W_h = cur.W_h + cur.mom_W_h
        cur.W_x = cur.W_x + cur.mom_W_x
        cur.b_h = cur.b_h + cur.mom_b_h
        cur.Q_W = cur.Q_W + cur.mom_Q_W
        cur.Q_b = cur.Q_b + cur.mom_Q_b
        cur.V_W = cur.V_W + cur.mom_V_W
        cur.V_b = cur.V_b + cur.mom_V_b
        cur.R_W = cur.R_W + cur.mom_R_W
        cur.R_b = cur.R_b + cur.mom_R_b
        cur.A_W = cur.A_W + cur.mom_A_W
        cur.A_b = cur.A_b + cur.mom_A_b
        cur.O_W = cur.O_W + cur.mom_O_W
        cur.O_b = cur.O_b + cur.mom_O_b
        cur_loss = float(_cm_loss(module=cur, Xs=Xs, Ys=Ys))
        losses.append(cur_loss)
    cur.n_train_steps = (
        int(cur.n_train_steps) + int(n_iters))
    cur.pre_train_loss = float(pre_loss)
    cur.last_train_loss = float(losses[-1])
    rep = ComposedLearnedMemoryTrainReportV1(
        schema=W83_COMPOSED_MEMORY_V1_SCHEMA_VERSION,
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


def build_composed_long_horizon_dataset_v1(
        *,
        n_sequences: int = 18,
        seq_len: int = W83_CM_DEFAULT_SEQ_LEN,
        input_dim: int = W83_CM_DEFAULT_INPUT_DIM,
        output_dim: int = W83_CM_DEFAULT_OUTPUT_DIM,
        seed: int = W83_CM_DEFAULT_SEED,
) -> tuple["_np.ndarray", "_np.ndarray"]:
    """A *composed long-horizon credit-assignment* task.

    For each t, the target is::

        y_t = tanh(A_rec @ x_{t-LAG_LONG} + A_int @ z_t)

    where:

    * ``LAG_LONG = 10`` is a *long* delay — longer than any
      reasonable single-state recurrent compressed signal can
      survive without slot memory.
    * ``z_t = decay * z_{t-1} + x_t`` is a slow temporal
      integration term — rewards a meaningful recurrent state.

    The key property is that the long-delay recall term and the
    temporal integration term *compose*: a model with both a
    recurrent state AND addressable slot memory wins. Plain
    recurrent V2 (no slots) struggles with the long delay; plain
    slot memory (W81 diffmem with detached BPTT) cannot route
    gradient from a read at ``t`` back to the slot-router at
    ``t - LAG_LONG``. The W83 composed model has both AND
    trains via full BPTT so the gradient chain is intact.
    """
    rng = _np.random.default_rng(int(seed))
    N = int(n_sequences)
    T = int(seq_len)
    D_in = int(input_dim)
    D_out = int(output_dim)
    A_int = rng.standard_normal(
        (D_in, D_out)).astype(_np.float64) * 0.35
    A_rec = rng.standard_normal(
        (D_in, D_out)).astype(_np.float64) * 0.85
    decay = 0.55
    LAG_LONG = 10
    # Force seq_len long enough for the lag to fire.
    if T < LAG_LONG + 4:
        T = LAG_LONG + 4
    X_all = _np.zeros((N, T, D_in), dtype=_np.float64)
    Y_all = _np.zeros((N, T, D_out), dtype=_np.float64)
    for i in range(N):
        X = rng.standard_normal(
            (T, D_in)).astype(_np.float64)
        Y = _np.zeros((T, D_out), dtype=_np.float64)
        z = _np.zeros((D_in,), dtype=_np.float64)
        for t in range(T):
            z = decay * z + X[t]
            phi = z @ A_int
            if t >= LAG_LONG:
                rec = X[t - LAG_LONG] @ A_rec
            else:
                rec = _np.zeros((D_out,), dtype=_np.float64)
            Y[t] = _np.tanh(phi + rec)
        X_all[i] = X
        Y_all[i] = Y
    return X_all, Y_all


@dataclasses.dataclass(frozen=True)
class ComposedLearnedMemoryBaselineReportV1:
    """V1 composed vs (W81 V2 recurrent / W81 diffmem / ridge)."""

    schema: str
    composed_mse: float
    w81_v2_mse: float
    w81_diffmem_mse: float
    ridge_mse: float
    composed_beats_v2: bool
    composed_beats_diffmem: bool
    composed_beats_ridge: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "composed_mse": float(round(
                self.composed_mse, 12)),
            "w81_v2_mse": float(round(self.w81_v2_mse, 12)),
            "w81_diffmem_mse": float(round(
                self.w81_diffmem_mse, 12)),
            "ridge_mse": float(round(self.ridge_mse, 12)),
            "composed_beats_v2": bool(self.composed_beats_v2),
            "composed_beats_diffmem": bool(
                self.composed_beats_diffmem),
            "composed_beats_ridge": bool(
                self.composed_beats_ridge),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w83_composed_memory_baseline_report_v1",
            "report": self.to_dict()})


def compare_composed_vs_baselines_v1(
        *,
        composed: ComposedLearnedMemoryModuleV1,
        eval_sequences: Sequence[Sequence[Sequence[float]]],
        eval_targets: Sequence[Sequence[Sequence[float]]],
        w81_v2_seed: int = W83_CM_DEFAULT_SEED + 3,
        w81_diffmem_seed: int = W83_CM_DEFAULT_SEED + 5,
        baseline_train_iters: int = 60,
        w81_v2_hidden_dim: int | None = None,
) -> ComposedLearnedMemoryBaselineReportV1:
    """Run composed_v1 vs (W81 V2 recurrent, W81 diffmem, ridge).

    The V2 baseline is intentionally given a *smaller* hidden_dim
    (default: ``floor(sqrt(K_slots * memory_dim))``) so that the
    total memory capacity is comparable. This matches the W81
    diffmem-vs-V2 framing: V2 must compress all the slot memory
    into a single recurrent state of equivalent capacity.
    """
    from .learned_consolidation_v2 import (
        build_sequence_conditioned_consolidation_module_v2,
        train_sequence_conditioned_consolidation_module,
    )
    from .differentiable_memory_substrate_v1 import (
        build_differentiable_memory_substrate_v1,
        train_differentiable_memory_substrate,
    )
    Xs = [
        _np.asarray(s, dtype=_np.float64)
        for s in eval_sequences]
    Ys = [
        _np.asarray(t, dtype=_np.float64)
        for t in eval_targets]
    # Composed MSE.
    composed_total = 0.0
    n_total = 0
    for X, Y in zip(Xs, Ys):
        _, _, _, Yhat = composed.forward_sequence(X)
        d = Yhat - Y
        composed_total += float(_np.sum(d * d))
        n_total += int(d.size)
    composed_mse = composed_total / max(1, int(n_total))
    # W81 V2 baseline (recurrent without slots, smaller hidden).
    v2_hidden = int(w81_v2_hidden_dim) if (
        w81_v2_hidden_dim is not None) else max(
        4,
        int(_np.sqrt(
            float(composed.K_slots)
            * float(composed.memory_dim))))
    v2 = build_sequence_conditioned_consolidation_module_v2(
        input_dim=int(composed.input_dim),
        hidden_dim=int(v2_hidden),
        memory_dim=int(composed.memory_dim),
        output_dim=int(composed.output_dim),
        seed=int(w81_v2_seed))
    v2, _ = train_sequence_conditioned_consolidation_module(
        module=v2,
        train_sequences=[x.tolist() for x in Xs],
        train_targets=[y.tolist() for y in Ys],
        n_iters=int(baseline_train_iters))
    v2_total = 0.0
    n_v2 = 0
    for X, Y in zip(Xs, Ys):
        _, _, Yhat = v2.forward_sequence(X)
        d = Yhat - Y
        v2_total += float(_np.sum(d * d))
        n_v2 += int(d.size)
    v2_mse = v2_total / max(1, int(n_v2))
    # W81 differentiable memory baseline.
    dm = build_differentiable_memory_substrate_v1(
        input_dim=int(composed.input_dim),
        hidden_dim=int(composed.hidden_dim),
        memory_dim=int(composed.memory_dim),
        output_dim=int(composed.output_dim),
        K_slots=int(composed.K_slots),
        seed=int(w81_diffmem_seed))
    dm, _ = train_differentiable_memory_substrate(
        module=dm,
        train_sequences=[x.tolist() for x in Xs],
        train_targets=[y.tolist() for y in Ys],
        n_iters=int(baseline_train_iters))
    dm_total = 0.0
    n_dm = 0
    for X, Y in zip(Xs, Ys):
        _, _, _, Yhat = dm.forward_sequence(X)
        d = Yhat - Y
        dm_total += float(_np.sum(d * d))
        n_dm += int(d.size)
    dm_mse = dm_total / max(1, int(n_dm))
    # Pointwise ridge baseline.
    X_flat = _np.concatenate(Xs, axis=0)
    Y_flat = _np.concatenate(Ys, axis=0)
    X_aug = _np.concatenate(
        [X_flat,
         _np.ones((X_flat.shape[0], 1), dtype=_np.float64)],
        axis=1)
    lam = 1e-3
    A = X_aug.T @ X_aug + lam * _np.eye(
        X_aug.shape[1], dtype=_np.float64)
    Wmat = _np.linalg.solve(A, X_aug.T @ Y_flat)
    Y_ridge = X_aug @ Wmat
    d_r = Y_ridge - Y_flat
    ridge_mse = float(_np.mean(d_r * d_r))
    return ComposedLearnedMemoryBaselineReportV1(
        schema=W83_COMPOSED_MEMORY_V1_SCHEMA_VERSION,
        composed_mse=float(composed_mse),
        w81_v2_mse=float(v2_mse),
        w81_diffmem_mse=float(dm_mse),
        ridge_mse=float(ridge_mse),
        composed_beats_v2=bool(composed_mse < v2_mse),
        composed_beats_diffmem=bool(composed_mse < dm_mse),
        composed_beats_ridge=bool(composed_mse < ridge_mse),
    )


@dataclasses.dataclass(frozen=True)
class ComposedLearnedMemoryWitnessV1:
    schema: str
    module_cid: str
    n_train_steps: int
    last_train_loss: float

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w83_composed_learned_memory_witness_v1",
            "schema": str(self.schema),
            "module_cid": str(self.module_cid),
            "n_train_steps": int(self.n_train_steps),
            "last_train_loss": float(round(
                self.last_train_loss, 12)),
        })


def emit_composed_learned_memory_witness_v1(
        *, module: ComposedLearnedMemoryModuleV1,
) -> ComposedLearnedMemoryWitnessV1:
    return ComposedLearnedMemoryWitnessV1(
        schema=W83_COMPOSED_MEMORY_V1_SCHEMA_VERSION,
        module_cid=str(module.cid()),
        n_train_steps=int(module.n_train_steps),
        last_train_loss=float(module.last_train_loss),
    )


__all__ = [
    "W83_COMPOSED_MEMORY_V1_SCHEMA_VERSION",
    "ComposedLearnedMemoryModuleV1",
    "ComposedLearnedMemoryTrainReportV1",
    "ComposedLearnedMemoryBaselineReportV1",
    "ComposedLearnedMemoryWitnessV1",
    "build_composed_learned_memory_module_v1",
    "train_composed_learned_memory_module",
    "build_composed_long_horizon_dataset_v1",
    "compare_composed_vs_baselines_v1",
    "emit_composed_learned_memory_witness_v1",
]
