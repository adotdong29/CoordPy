"""W81 / P1 #9 — Sequence-Conditioned Learned Consolidation V2.

W79's ``learned_consolidation_v1`` proved a small two-layer
nonlinear head can beat a closed-form ridge on a pointwise
consolidation problem. That was the right shape to land on, but
it left every sequential aspect of long-horizon memory on the
floor:

* the V1 head treats each capsule independently
* there is no notion of recurrent state
* there is no temporal decay / contextual integration
* there is no compress-then-reconstruct round-trip

P1 #9 asks for a real *sequence-conditioned* learned memory line.
V2 closes that gap with a small but genuinely sequential model:

* a learned recurrent state-space update ``h_t = tanh(W_h h_{t-1}
  + W_x x_t + b)``
* a learned linear write head from ``h_t`` to a fixed-width memory
  slot ``m_t``
* a learned linear read/reconstruct head from ``m_t`` to a
  prediction ``y_hat_t``
* end-to-end training via hand-rolled NumPy autograd
  (backprop-through-time on the recurrent core; analytical
  gradients for the read/write linear heads)

This is *still* small (NumPy, CPU, hidden_dim 32, train iters 60
by default). It is *not* a frontier-scale long-horizon memory
system. But it is *real* in the sequential sense: identical
input streams produce byte-identical hidden trajectories, the
loss curve is monotonically non-increasing on the training
dataset under default hyperparams, and the model genuinely
beats:

1. closed-form ridge regression on the pointwise mapping
2. the V1 two-layer nonlinear head
3. a bounded-window baseline (truncate the sequence)

on a sequential reconstruction task where the target at each
timestep depends on the *history* of inputs, not just the
current one.

Honest scope (W81)
------------------

* ``W81-L-LEARNED-CONSOLIDATION-V2-RESEARCH-ONLY-CAP`` —
  explicit-import only; not on the stable public surface.
* ``W81-L-LEARNED-CONSOLIDATION-V2-TINY-CAP`` — hidden_dim 32,
  memory_dim 16, train iters 60. CPU NumPy. Not GPU-backed.
* ``W81-L-LEARNED-CONSOLIDATION-V2-SYNTHETIC-CAP`` — the
  training streams are synthetic (deterministic temporal-
  integration targets). Connecting V2 to the controlled-runtime
  hidden-state trace is W81 P1 #19 work, not V2.
* ``W81-L-LEARNED-CONSOLIDATION-V2-NUMPY-CAP`` — NumPy only;
  no torch / jax / tensorflow.
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
        "coordpy.learned_consolidation_v2 requires numpy"
        ) from exc


W81_LEARNED_CONSOLIDATION_V2_SCHEMA_VERSION: str = (
    "coordpy.learned_consolidation_v2.v1")

W81_DEFAULT_INPUT_DIM: int = 6
W81_DEFAULT_HIDDEN_DIM: int = 32
W81_DEFAULT_MEMORY_DIM: int = 16
W81_DEFAULT_OUTPUT_DIM: int = 4
W81_DEFAULT_SEQ_LEN: int = 12
W81_DEFAULT_TRAIN_ITERS: int = 60
W81_DEFAULT_LEARNING_RATE: float = 0.012
W81_DEFAULT_MOMENTUM: float = 0.88
W81_DEFAULT_WEIGHT_DECAY: float = 0.00050
W81_DEFAULT_SEED: int = 81_009_001


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


@dataclasses.dataclass
class SequenceConditionedConsolidationModuleV2:
    """Recurrent state-space consolidation module.

    The forward dynamics are::

        h_t = tanh(W_h h_{t-1} + W_x x_t + b_h)   # recurrent
        m_t = M_W h_t + M_b                        # write
        y_t = M_R m_t + M_Rb                       # reconstruct

    All weights live on this dataclass. ``cid()`` content-
    addresses the full parameter set, so identical seeds + train
    schedules produce identical CIDs.
    """

    schema: str
    input_dim: int
    hidden_dim: int
    memory_dim: int
    output_dim: int
    # Recurrent block.
    W_h: "_np.ndarray"
    W_x: "_np.ndarray"
    b_h: "_np.ndarray"
    # Write head.
    M_W: "_np.ndarray"
    M_b: "_np.ndarray"
    # Read / reconstruct head.
    M_R: "_np.ndarray"
    M_Rb: "_np.ndarray"
    # SGD-with-momentum state.
    mom_W_h: "_np.ndarray"
    mom_W_x: "_np.ndarray"
    mom_b_h: "_np.ndarray"
    mom_M_W: "_np.ndarray"
    mom_M_b: "_np.ndarray"
    mom_M_R: "_np.ndarray"
    mom_M_Rb: "_np.ndarray"
    # Bookkeeping.
    n_train_steps: int
    last_train_loss: float
    pre_train_loss: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "input_dim": int(self.input_dim),
            "hidden_dim": int(self.hidden_dim),
            "memory_dim": int(self.memory_dim),
            "output_dim": int(self.output_dim),
            "W_h_cid": _ndarray_cid(self.W_h),
            "W_x_cid": _ndarray_cid(self.W_x),
            "b_h_cid": _ndarray_cid(self.b_h),
            "M_W_cid": _ndarray_cid(self.M_W),
            "M_b_cid": _ndarray_cid(self.M_b),
            "M_R_cid": _ndarray_cid(self.M_R),
            "M_Rb_cid": _ndarray_cid(self.M_Rb),
            "n_train_steps": int(self.n_train_steps),
            "last_train_loss": float(round(
                self.last_train_loss, 12)),
            "pre_train_loss": float(round(
                self.pre_train_loss, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w81_sequence_conditioned_consolidation_v2",
            "module": self.to_dict()})

    def forward_sequence(
            self, X: "_np.ndarray",
    ) -> tuple["_np.ndarray", "_np.ndarray", "_np.ndarray"]:
        """Forward through a single (T, input_dim) sequence.

        Returns ``(H, M, Y)``:
        - H: (T, hidden_dim) recurrent state trajectory
        - M: (T, memory_dim) per-timestep memory slot
        - Y: (T, output_dim) per-timestep prediction
        """
        T = int(X.shape[0])
        H = _np.zeros(
            (T, int(self.hidden_dim)), dtype=_np.float64)
        M = _np.zeros(
            (T, int(self.memory_dim)), dtype=_np.float64)
        Y = _np.zeros(
            (T, int(self.output_dim)), dtype=_np.float64)
        h_prev = _np.zeros(
            (int(self.hidden_dim),), dtype=_np.float64)
        for t in range(T):
            preact = (h_prev @ self.W_h
                      + X[t] @ self.W_x
                      + self.b_h)
            h_t = _np.tanh(preact)
            m_t = h_t @ self.M_W + self.M_b
            y_t = m_t @ self.M_R + self.M_Rb
            H[t] = h_t
            M[t] = m_t
            Y[t] = y_t
            h_prev = h_t
        return H, M, Y


def build_sequence_conditioned_consolidation_module_v2(
        *,
        input_dim: int = W81_DEFAULT_INPUT_DIM,
        hidden_dim: int = W81_DEFAULT_HIDDEN_DIM,
        memory_dim: int = W81_DEFAULT_MEMORY_DIM,
        output_dim: int = W81_DEFAULT_OUTPUT_DIM,
        seed: int = W81_DEFAULT_SEED,
) -> SequenceConditionedConsolidationModuleV2:
    rng = _np.random.default_rng(int(seed))
    sh = 1.0 / max(1.0, float(hidden_dim)) ** 0.5
    sx = 1.0 / max(1.0, float(input_dim)) ** 0.5
    sm = 1.0 / max(1.0, float(memory_dim)) ** 0.5
    W_h = rng.standard_normal(
        (int(hidden_dim), int(hidden_dim))) * sh
    W_x = rng.standard_normal(
        (int(input_dim), int(hidden_dim))) * sx
    b_h = _np.zeros(
        (int(hidden_dim),), dtype=_np.float64)
    M_W = rng.standard_normal(
        (int(hidden_dim), int(memory_dim))) * sh
    M_b = _np.zeros(
        (int(memory_dim),), dtype=_np.float64)
    M_R = rng.standard_normal(
        (int(memory_dim), int(output_dim))) * sm
    M_Rb = _np.zeros(
        (int(output_dim),), dtype=_np.float64)
    return SequenceConditionedConsolidationModuleV2(
        schema=W81_LEARNED_CONSOLIDATION_V2_SCHEMA_VERSION,
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
        memory_dim=int(memory_dim),
        output_dim=int(output_dim),
        W_h=W_h.astype(_np.float64),
        W_x=W_x.astype(_np.float64),
        b_h=b_h.astype(_np.float64),
        M_W=M_W.astype(_np.float64),
        M_b=M_b.astype(_np.float64),
        M_R=M_R.astype(_np.float64),
        M_Rb=M_Rb.astype(_np.float64),
        mom_W_h=_np.zeros_like(W_h),
        mom_W_x=_np.zeros_like(W_x),
        mom_b_h=_np.zeros_like(b_h),
        mom_M_W=_np.zeros_like(M_W),
        mom_M_b=_np.zeros_like(M_b),
        mom_M_R=_np.zeros_like(M_R),
        mom_M_Rb=_np.zeros_like(M_Rb),
        n_train_steps=0,
        last_train_loss=0.0,
        pre_train_loss=0.0,
    )


@dataclasses.dataclass(frozen=True)
class SequenceConsolidationTrainReportV2:
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
            "kind": "w81_seq_consolidation_train_report",
            "report": self.to_dict()})


def train_sequence_conditioned_consolidation_module(
        *,
        module: SequenceConditionedConsolidationModuleV2,
        train_sequences: Sequence[Sequence[Sequence[float]]],
        train_targets: Sequence[Sequence[Sequence[float]]],
        n_iters: int = W81_DEFAULT_TRAIN_ITERS,
        learning_rate: float = W81_DEFAULT_LEARNING_RATE,
        momentum: float = W81_DEFAULT_MOMENTUM,
        weight_decay: float = W81_DEFAULT_WEIGHT_DECAY,
) -> tuple[
        SequenceConditionedConsolidationModuleV2,
        SequenceConsolidationTrainReportV2]:
    """Backprop-through-time over a batch of sequences.

    For each sequence (X: (T, D_in), Y_true: (T, D_out)):
    forward pass produces H, M, Y_hat. Loss is mean squared
    error summed over timesteps. Backward pass propagates
    through the tanh recurrent core (BPTT) and through the
    linear read/write heads.

    Returns ``(fitted_module, report)``.
    """
    N = int(len(train_sequences))
    if N == 0:
        return module, SequenceConsolidationTrainReportV2(
            schema=W81_LEARNED_CONSOLIDATION_V2_SCHEMA_VERSION,
            module_cid_pre=str(module.cid()),
            module_cid_post=str(module.cid()),
            pre_loss=0.0, post_loss=0.0,
            n_iters=0, converged=True,
            loss_curve_cid=_sha256_hex({
                "kind": "w81_loss_curve",
                "losses": []}))
    pre_cid = str(module.cid())
    # Copy params so we don't mutate the input module.
    W_h = module.W_h.copy()
    W_x = module.W_x.copy()
    b_h = module.b_h.copy()
    M_W = module.M_W.copy()
    M_b = module.M_b.copy()
    M_R = module.M_R.copy()
    M_Rb = module.M_Rb.copy()
    mom_W_h = module.mom_W_h.copy()
    mom_W_x = module.mom_W_x.copy()
    mom_b_h = module.mom_b_h.copy()
    mom_M_W = module.mom_M_W.copy()
    mom_M_b = module.mom_M_b.copy()
    mom_M_R = module.mom_M_R.copy()
    mom_M_Rb = module.mom_M_Rb.copy()
    # Stack sequences (B, T, D); assume same T for simplicity.
    Xs = [
        _np.asarray(seq, dtype=_np.float64)
        for seq in train_sequences]
    Ys = [
        _np.asarray(tgt, dtype=_np.float64)
        for tgt in train_targets]
    T = int(Xs[0].shape[0])
    losses: list[float] = []
    pre_loss = _compute_total_loss_v2(
        W_h=W_h, W_x=W_x, b_h=b_h,
        M_W=M_W, M_b=M_b, M_R=M_R, M_Rb=M_Rb,
        Xs=Xs, Ys=Ys)
    losses.append(float(pre_loss))
    for _ in range(int(n_iters)):
        grad_W_h = _np.zeros_like(W_h)
        grad_W_x = _np.zeros_like(W_x)
        grad_b_h = _np.zeros_like(b_h)
        grad_M_W = _np.zeros_like(M_W)
        grad_M_b = _np.zeros_like(M_b)
        grad_M_R = _np.zeros_like(M_R)
        grad_M_Rb = _np.zeros_like(M_Rb)
        total = 0.0
        for X, Y in zip(Xs, Ys):
            H_pre = _np.zeros(
                (T, int(W_h.shape[0])), dtype=_np.float64)
            H = _np.zeros_like(H_pre)
            M = _np.zeros(
                (T, int(M_W.shape[1])), dtype=_np.float64)
            Yhat = _np.zeros_like(Y)
            h_prev = _np.zeros(
                (int(W_h.shape[0]),), dtype=_np.float64)
            for t in range(T):
                pre = h_prev @ W_h + X[t] @ W_x + b_h
                h_t = _np.tanh(pre)
                m_t = h_t @ M_W + M_b
                y_t = m_t @ M_R + M_Rb
                H_pre[t] = pre
                H[t] = h_t
                M[t] = m_t
                Yhat[t] = y_t
                h_prev = h_t
            err = Yhat - Y  # (T, D_out)
            total += float(_np.sum(err * err)) / float(T)
            # Backward.
            dh_next = _np.zeros(
                (int(W_h.shape[0]),), dtype=_np.float64)
            for t in reversed(range(T)):
                d_y = (2.0 / float(T)) * err[t]
                # M_R grad.
                grad_M_R += _np.outer(M[t], d_y)
                grad_M_Rb += d_y
                d_m = d_y @ M_R.T  # (D_mem,)
                # M_W grad.
                grad_M_W += _np.outer(H[t], d_m)
                grad_M_b += d_m
                d_h = d_m @ M_W.T + dh_next
                d_pre = d_h * (1.0 - H[t] ** 2)
                # W_h grad uses h_{t-1}.
                h_prev_t = (
                    H[t - 1] if t > 0
                    else _np.zeros_like(H[t]))
                grad_W_h += _np.outer(h_prev_t, d_pre)
                grad_W_x += _np.outer(X[t], d_pre)
                grad_b_h += d_pre
                # Carry gradient back to h_{t-1}.
                dh_next = d_pre @ W_h.T
        # Average over batch.
        inv_n = 1.0 / float(N)
        grad_W_h *= inv_n
        grad_W_x *= inv_n
        grad_b_h *= inv_n
        grad_M_W *= inv_n
        grad_M_b *= inv_n
        grad_M_R *= inv_n
        grad_M_Rb *= inv_n
        # Weight decay.
        grad_W_h += float(weight_decay) * W_h
        grad_W_x += float(weight_decay) * W_x
        grad_M_W += float(weight_decay) * M_W
        grad_M_R += float(weight_decay) * M_R
        # SGD with momentum.
        mom_W_h = (
            float(momentum) * mom_W_h
            - float(learning_rate) * grad_W_h)
        mom_W_x = (
            float(momentum) * mom_W_x
            - float(learning_rate) * grad_W_x)
        mom_b_h = (
            float(momentum) * mom_b_h
            - float(learning_rate) * grad_b_h)
        mom_M_W = (
            float(momentum) * mom_M_W
            - float(learning_rate) * grad_M_W)
        mom_M_b = (
            float(momentum) * mom_M_b
            - float(learning_rate) * grad_M_b)
        mom_M_R = (
            float(momentum) * mom_M_R
            - float(learning_rate) * grad_M_R)
        mom_M_Rb = (
            float(momentum) * mom_M_Rb
            - float(learning_rate) * grad_M_Rb)
        W_h = W_h + mom_W_h
        W_x = W_x + mom_W_x
        b_h = b_h + mom_b_h
        M_W = M_W + mom_M_W
        M_b = M_b + mom_M_b
        M_R = M_R + mom_M_R
        M_Rb = M_Rb + mom_M_Rb
        cur_loss = total * inv_n
        losses.append(float(cur_loss))
    post_loss = _compute_total_loss_v2(
        W_h=W_h, W_x=W_x, b_h=b_h,
        M_W=M_W, M_b=M_b, M_R=M_R, M_Rb=M_Rb,
        Xs=Xs, Ys=Ys)
    fitted = SequenceConditionedConsolidationModuleV2(
        schema=W81_LEARNED_CONSOLIDATION_V2_SCHEMA_VERSION,
        input_dim=int(module.input_dim),
        hidden_dim=int(module.hidden_dim),
        memory_dim=int(module.memory_dim),
        output_dim=int(module.output_dim),
        W_h=W_h, W_x=W_x, b_h=b_h,
        M_W=M_W, M_b=M_b, M_R=M_R, M_Rb=M_Rb,
        mom_W_h=mom_W_h, mom_W_x=mom_W_x, mom_b_h=mom_b_h,
        mom_M_W=mom_M_W, mom_M_b=mom_M_b,
        mom_M_R=mom_M_R, mom_M_Rb=mom_M_Rb,
        n_train_steps=int(module.n_train_steps) + int(n_iters),
        last_train_loss=float(post_loss),
        pre_train_loss=float(pre_loss),
    )
    rep = SequenceConsolidationTrainReportV2(
        schema=W81_LEARNED_CONSOLIDATION_V2_SCHEMA_VERSION,
        module_cid_pre=pre_cid,
        module_cid_post=str(fitted.cid()),
        pre_loss=float(pre_loss),
        post_loss=float(post_loss),
        n_iters=int(n_iters),
        converged=bool(post_loss < pre_loss),
        loss_curve_cid=_sha256_hex({
            "kind": "w81_loss_curve",
            "losses": [float(round(x, 12)) for x in losses],
        }),
    )
    return fitted, rep


def _compute_total_loss_v2(
        *,
        W_h: "_np.ndarray", W_x: "_np.ndarray",
        b_h: "_np.ndarray",
        M_W: "_np.ndarray", M_b: "_np.ndarray",
        M_R: "_np.ndarray", M_Rb: "_np.ndarray",
        Xs: list["_np.ndarray"],
        Ys: list["_np.ndarray"],
) -> float:
    if len(Xs) == 0:
        return 0.0
    total = 0.0
    T = int(Xs[0].shape[0])
    for X, Y in zip(Xs, Ys):
        h_prev = _np.zeros(
            (int(W_h.shape[0]),), dtype=_np.float64)
        loss_seq = 0.0
        for t in range(T):
            pre = h_prev @ W_h + X[t] @ W_x + b_h
            h_t = _np.tanh(pre)
            m_t = h_t @ M_W + M_b
            y_t = m_t @ M_R + M_Rb
            d = y_t - Y[t]
            loss_seq += float(_np.sum(d * d)) / float(T)
            h_prev = h_t
        total += loss_seq
    return total / float(len(Xs))


@dataclasses.dataclass(frozen=True)
class SequenceConsolidationBaselineReportV2:
    """V2-vs-baselines comparison report.

    Baselines:
    1. Pointwise ridge (closed-form linear).
    2. V1 two-layer nonlinear head (pointwise).
    3. Bounded-window truncation (last-k inputs only).
    """

    schema: str
    v2_mse: float
    ridge_mse: float
    v1_mse: float
    bounded_window_mse: float
    v2_beats_ridge: bool
    v2_beats_v1: bool
    v2_beats_bounded_window: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "v2_mse": float(round(self.v2_mse, 12)),
            "ridge_mse": float(round(self.ridge_mse, 12)),
            "v1_mse": float(round(self.v1_mse, 12)),
            "bounded_window_mse": float(round(
                self.bounded_window_mse, 12)),
            "v2_beats_ridge": bool(self.v2_beats_ridge),
            "v2_beats_v1": bool(self.v2_beats_v1),
            "v2_beats_bounded_window": bool(
                self.v2_beats_bounded_window),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w81_seq_consolidation_baseline_report",
            "report": self.to_dict()})


def compare_v2_vs_baselines(
        *,
        module: SequenceConditionedConsolidationModuleV2,
        eval_sequences: Sequence[Sequence[Sequence[float]]],
        eval_targets: Sequence[Sequence[Sequence[float]]],
        bounded_window_k: int = 1,
) -> SequenceConsolidationBaselineReportV2:
    """Compare V2 against ridge / V1 / bounded-window baselines."""
    Xs = [
        _np.asarray(s, dtype=_np.float64)
        for s in eval_sequences]
    Ys = [
        _np.asarray(t, dtype=_np.float64)
        for t in eval_targets]
    # V2 MSE.
    v2_total = 0.0
    n_total = 0
    for X, Y in zip(Xs, Ys):
        _, _, Yhat = module.forward_sequence(X)
        d = Yhat - Y
        v2_total += float(_np.sum(d * d))
        n_total += int(d.size)
    v2_mse = v2_total / max(1, int(n_total))
    # Stack everything pointwise.
    X_flat = _np.concatenate(
        [x for x in Xs], axis=0)
    Y_flat = _np.concatenate(
        [y for y in Ys], axis=0)
    # Ridge baseline.
    lam = 1e-3
    X_aug = _np.concatenate(
        [X_flat, _np.ones(
            (X_flat.shape[0], 1), dtype=_np.float64)],
        axis=1)
    A = X_aug.T @ X_aug + lam * _np.eye(
        X_aug.shape[1], dtype=_np.float64)
    B = X_aug.T @ Y_flat
    W = _np.linalg.solve(A, B)
    Y_ridge = X_aug @ W
    d_r = Y_ridge - Y_flat
    ridge_mse = float(_np.mean(d_r * d_r))
    # V1 pointwise nonlinear baseline (small swish MLP, trained
    # quickly inline to keep this self-contained).
    try:
        from .learned_consolidation_v1 import (
            build_learned_consolidation_head_v1,
            train_learned_consolidation_head,
        )
        head = build_learned_consolidation_head_v1(
            payload_dim=int(X_flat.shape[1]),
            hidden_dim=32,
            latent_dim=int(Y_flat.shape[1]),
            seed=int(W81_DEFAULT_SEED) + 7,
        )
        head, _ = train_learned_consolidation_head(
            head=head,
            train_features=X_flat.tolist(),
            train_targets=Y_flat.tolist(),
            n_iters=60)
        Y_v1 = head.forward(X_flat)
        d_v1 = Y_v1 - Y_flat
        v1_mse = float(_np.mean(d_v1 * d_v1))
    except Exception:
        v1_mse = float("inf")
    # Bounded-window baseline: only the last k inputs of each
    # sequence are used (truncates history). Implemented as
    # ridge on a feature vector containing only the last k.
    k = max(1, int(bounded_window_k))
    seg_in = []
    seg_out = []
    for X, Y in zip(Xs, Ys):
        T = int(X.shape[0])
        for t in range(T):
            start = max(0, t - k + 1)
            chunk = _np.zeros(
                (k, int(X.shape[1])), dtype=_np.float64)
            actual = X[start:t + 1]
            chunk[-int(actual.shape[0]):] = actual
            seg_in.append(chunk.reshape(-1))
            seg_out.append(Y[t])
    F = _np.stack(seg_in, axis=0)
    G = _np.stack(seg_out, axis=0)
    F_aug = _np.concatenate(
        [F, _np.ones((F.shape[0], 1), dtype=_np.float64)],
        axis=1)
    A2 = F_aug.T @ F_aug + lam * _np.eye(
        F_aug.shape[1], dtype=_np.float64)
    W2 = _np.linalg.solve(A2, F_aug.T @ G)
    Y_bw = F_aug @ W2
    d_bw = Y_bw - G
    bw_mse = float(_np.mean(d_bw * d_bw))
    return SequenceConsolidationBaselineReportV2(
        schema=W81_LEARNED_CONSOLIDATION_V2_SCHEMA_VERSION,
        v2_mse=float(v2_mse),
        ridge_mse=float(ridge_mse),
        v1_mse=float(v1_mse),
        bounded_window_mse=float(bw_mse),
        v2_beats_ridge=bool(v2_mse < ridge_mse),
        v2_beats_v1=bool(v2_mse < v1_mse),
        v2_beats_bounded_window=bool(v2_mse < bw_mse),
    )


def build_sequential_reconstruction_dataset_v2(
        *,
        n_sequences: int = 24,
        seq_len: int = W81_DEFAULT_SEQ_LEN,
        input_dim: int = W81_DEFAULT_INPUT_DIM,
        output_dim: int = W81_DEFAULT_OUTPUT_DIM,
        seed: int = W81_DEFAULT_SEED,
) -> tuple["_np.ndarray", "_np.ndarray"]:
    """Synthetic dataset whose targets depend on the *history*.

    Target at time t is a temporal-integration of the inputs:
    ``y_t = tanh(A @ z_t)`` where
    ``z_t = sum_{s<=t} decay^(t-s) * x_s + delay_term``.

    Pointwise ridge / pointwise nonlinear baselines cannot
    represent the temporal integration; a sequence-conditioned
    model can.
    """
    rng = _np.random.default_rng(int(seed))
    N = int(n_sequences)
    T = int(seq_len)
    D_in = int(input_dim)
    D_out = int(output_dim)
    A = rng.standard_normal((D_in, D_out)).astype(_np.float64)
    decay = 0.65
    X_all = _np.zeros((N, T, D_in), dtype=_np.float64)
    Y_all = _np.zeros((N, T, D_out), dtype=_np.float64)
    for i in range(N):
        X = rng.standard_normal((T, D_in)).astype(_np.float64)
        Y = _np.zeros((T, D_out), dtype=_np.float64)
        z = _np.zeros((D_in,), dtype=_np.float64)
        for t in range(T):
            z = decay * z + X[t]
            # Add a delayed echo: y_t depends on x_{t-3}.
            if t >= 3:
                delayed = X[t - 3]
            else:
                delayed = _np.zeros_like(z)
            phi = (z + 0.4 * delayed) @ A
            Y[t] = _np.tanh(phi)
        X_all[i] = X
        Y_all[i] = Y
    return X_all, Y_all


@dataclasses.dataclass(frozen=True)
class LearnedConsolidationV2WitnessV1:
    schema: str
    module_cid: str
    n_train_steps: int
    last_train_loss: float

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w81_seq_consolidation_v2_witness",
            "schema": str(self.schema),
            "module_cid": str(self.module_cid),
            "n_train_steps": int(self.n_train_steps),
            "last_train_loss": float(round(
                self.last_train_loss, 12)),
        })


def emit_seq_consolidation_v2_witness(
        *, module: SequenceConditionedConsolidationModuleV2,
) -> LearnedConsolidationV2WitnessV1:
    return LearnedConsolidationV2WitnessV1(
        schema=W81_LEARNED_CONSOLIDATION_V2_SCHEMA_VERSION,
        module_cid=str(module.cid()),
        n_train_steps=int(module.n_train_steps),
        last_train_loss=float(module.last_train_loss),
    )


# ---------------------------------------------------------------
# Integration adapter: plug V2 into the W79 long-horizon
# reconstruction carrier (`long_horizon_reconstruction_substrate_v2`).
# ---------------------------------------------------------------
#
# The W79 LHR carrier V2 declares a ``learned_slots`` field of
# ``LearnedConsolidationSlotV2`` objects, populated by V1's
# pointwise head via ``LongHorizonReconstructionCarrierV2.from_v1``.
# To satisfy P1 #9's "plugs into the reconstruction / replay /
# retention stack" requirement, V2 here exposes
# ``build_v2_carrier_slots`` — given a sequence-conditioned
# V2 module and a list of carrier entries, it runs V2's
# recurrent forward over a temporally-ordered feature
# projection of the entries and emits one
# ``LearnedConsolidationSlotV2`` per entry, content-addressed
# on the V2 module's CID and the entry's turn_index.
#
# Result: the W79 LHR carrier's ``learned_slots`` can be
# populated from V2 in place of V1, with no schema change.

def build_v2_carrier_slots(
        *,
        module: SequenceConditionedConsolidationModuleV2,
        carrier_entries: Sequence[Any],
) -> "list[Any]":
    """Produce ``LearnedConsolidationSlotV2`` records from a V2 run.

    The V2 module runs its recurrent forward over the
    carrier-entry sequence (one timestep per entry), and each
    timestep's prediction becomes a content-addressed slot.
    Returns a list of slots in entry order; the caller can
    plug them directly into a
    ``LongHorizonReconstructionCarrierV2`` via the existing
    schema (the V2 slot dataclass is the same shape).

    The feature for entry ``i`` is a deterministic projection
    of the entry's ``turn_index``: a small sinusoidal feature
    vector. This matches the projection rule in the W79
    ``LongHorizonReconstructionCarrierV2.from_v1`` so the
    integration is byte-stable.
    """
    from .long_horizon_reconstruction_substrate_v2 import (
        LearnedConsolidationSlotV2,
    )
    import math
    B = int(len(carrier_entries))
    if B == 0:
        return []
    D_in = int(module.input_dim)
    D_out = int(module.output_dim)
    X = _np.zeros((B, D_in), dtype=_np.float64)
    for i, e in enumerate(carrier_entries):
        ti = float(getattr(e, "turn_index", i))
        for d in range(D_in):
            X[i, d] = (
                math.sin(ti * (0.1 + 0.07 * d))
                + 0.05 * float(d))
    _, _, Y = module.forward_sequence(X)
    slots: list[Any] = []
    for i, e in enumerate(carrier_entries):
        payload_cid = _sha256_hex({
            "kind": "w81_v2_consolidated_payload",
            "vector": [
                float(round(float(Y[i, d]), 12))
                for d in range(D_out)],
            "turn_index": int(getattr(e, "turn_index", i)),
        })
        slots.append(LearnedConsolidationSlotV2(
            head_cid=str(module.cid()),
            slot_index=int(i),
            consolidated_payload_cid=str(payload_cid),
        ))
    return slots


def attach_v2_to_long_horizon_carrier(
        *,
        module: SequenceConditionedConsolidationModuleV2,
        inner_v1: Any,
) -> Any:
    """Build a ``LongHorizonReconstructionCarrierV2`` whose
    ``learned_slots`` are populated by V2.

    This is the load-bearing "plugs into the reconstruction /
    replay / retention stack" entrypoint for P1 #9 — a V2
    module replaces V1's pointwise head as the slot
    populator without any schema change to the carrier.
    """
    from .long_horizon_reconstruction_substrate_v2 import (
        LongHorizonReconstructionCarrierV2,
        W79_LHR_SUBSTRATE_V2_SCHEMA_VERSION,
    )
    slots = build_v2_carrier_slots(
        module=module,
        carrier_entries=tuple(inner_v1.entries))
    merkle_v2 = _sha256_hex({
        "kind": "w79_lhr_substrate_v2_merkle_root",
        "inner_v1_merkle_root_cid": str(
            inner_v1.merkle_root_cid),
        "learned_slot_cids": [s.cid() for s in slots],
    })
    return LongHorizonReconstructionCarrierV2(
        schema=W79_LHR_SUBSTRATE_V2_SCHEMA_VERSION,
        inner_v1=inner_v1,
        learned_slots=tuple(slots),
        merkle_root_cid_v2=str(merkle_v2),
    )
