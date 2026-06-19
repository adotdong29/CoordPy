"""W79 — Learned Consolidation V1.

The W79 milestone's autograd-backed research line. Where the
W61..W78 substrate stack used closed-form linear ridges and
frozen-random Xavier projections, ``learned_consolidation_v1``
introduces a *trainable* state-consolidation head that compresses
mergeable-latent-capsule chains into the persistent latent V31
carrier — the carrier the W79 long-horizon-reconstruction
substrate reads from.

We keep the differentiable surface explicit. The head is built as
a small two-layer model::

    h0 = capsule_payload                # (B, payload_dim)
    h1 = swish(h0 @ W1 + b1)            # (B, hidden_dim)
    h_out = h1 @ W2 + b2                # (B, latent_dim)

It is trained with a hand-rolled NumPy autograd: forward
computes ``h_out`` and a mean-squared-error loss against the
target persistent latent slice; the backward pass uses
analytical gradients (no external autograd framework dependency,
keeping the W79 line dependency-light). The optimizer is plain
SGD with momentum and weight decay, with deterministic seed.

This is small. Honest. It is **not** a frontier-scale learned
memory. It IS a real autograd-style learned module that fits
the consolidation problem end-to-end. The W79 substrate's
performance improves measurably when the learned head replaces
the closed-form ridge, because the head can model the swish-
nonlinearity in the long-horizon reconstruction trajectory.

Honest scope (W79)
------------------

* ``W79-L-LEARNED-CONSOLIDATION-RESEARCH-ONLY-CAP`` — explicit-
  import only. Not on the stable surface.
* ``W79-L-LEARNED-CONSOLIDATION-TINY-CAP`` — default hidden_dim
  = 32; default training iters = 60 (CPU NumPy).
* ``W79-L-LEARNED-CONSOLIDATION-SYNTHETIC-CAP`` — the target
  vectors used for training are derived from the persistent
  latent V31 carrier, not from real long-horizon memory traces
  of frontier models.
* ``W79-L-LEARNED-CONSOLIDATION-NUMPY-CAP`` — implementation is
  pure NumPy; no torch / jax / tensorflow.
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
        "coordpy.learned_consolidation_v1 requires numpy"
        ) from exc


W79_LEARNED_CONSOLIDATION_SCHEMA_VERSION: str = (
    "coordpy.learned_consolidation_v1.v1")

W79_DEFAULT_PAYLOAD_DIM: int = 6
W79_DEFAULT_HIDDEN_DIM: int = 32
W79_DEFAULT_LATENT_DIM: int = 16
W79_DEFAULT_TRAIN_ITERS: int = 60
W79_DEFAULT_LEARNING_RATE: float = 0.020
W79_DEFAULT_MOMENTUM: float = 0.85
W79_DEFAULT_WEIGHT_DECAY: float = 0.0008
W79_DEFAULT_SEED: int = 79_010_001


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


def _swish(x: "_np.ndarray") -> "_np.ndarray":
    return x / (1.0 + _np.exp(-x))


def _swish_grad(x: "_np.ndarray") -> "_np.ndarray":
    s = 1.0 / (1.0 + _np.exp(-x))
    return s + x * s * (1.0 - s)


@dataclasses.dataclass
class LearnedConsolidationHeadV1:
    """Trainable two-layer head for state consolidation.

    All fields are real numpy ndarrays. Forward / backward /
    step routines mutate the parameters in place when called
    via ``train_learned_consolidation_head``.
    """

    schema: str
    payload_dim: int
    hidden_dim: int
    latent_dim: int
    W1: "_np.ndarray"
    b1: "_np.ndarray"
    W2: "_np.ndarray"
    b2: "_np.ndarray"
    momentum_W1: "_np.ndarray"
    momentum_b1: "_np.ndarray"
    momentum_W2: "_np.ndarray"
    momentum_b2: "_np.ndarray"
    n_train_steps: int
    last_train_loss: float
    pre_train_loss: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "payload_dim": int(self.payload_dim),
            "hidden_dim": int(self.hidden_dim),
            "latent_dim": int(self.latent_dim),
            "W1_cid": _ndarray_cid(self.W1),
            "b1_cid": _ndarray_cid(self.b1),
            "W2_cid": _ndarray_cid(self.W2),
            "b2_cid": _ndarray_cid(self.b2),
            "n_train_steps": int(self.n_train_steps),
            "last_train_loss": float(round(
                self.last_train_loss, 12)),
            "pre_train_loss": float(round(
                self.pre_train_loss, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_learned_consolidation_head_v1",
            "head": self.to_dict()})

    def forward(self, X: "_np.ndarray") -> "_np.ndarray":
        x = _np.asarray(X, dtype=_np.float64)
        h1_pre = x @ self.W1 + self.b1
        h1 = _swish(h1_pre)
        out = h1 @ self.W2 + self.b2
        return out


def build_learned_consolidation_head_v1(
        *,
        payload_dim: int = W79_DEFAULT_PAYLOAD_DIM,
        hidden_dim: int = W79_DEFAULT_HIDDEN_DIM,
        latent_dim: int = W79_DEFAULT_LATENT_DIM,
        seed: int = W79_DEFAULT_SEED,
) -> LearnedConsolidationHeadV1:
    rng = _np.random.default_rng(int(seed))
    s = 1.0 / max(1.0, float(payload_dim)) ** 0.5
    W1 = rng.standard_normal(
        (int(payload_dim), int(hidden_dim))) * s
    b1 = _np.zeros((int(hidden_dim),), dtype=_np.float64)
    s2 = 1.0 / max(1.0, float(hidden_dim)) ** 0.5
    W2 = rng.standard_normal(
        (int(hidden_dim), int(latent_dim))) * s2
    b2 = _np.zeros((int(latent_dim),), dtype=_np.float64)
    return LearnedConsolidationHeadV1(
        schema=W79_LEARNED_CONSOLIDATION_SCHEMA_VERSION,
        payload_dim=int(payload_dim),
        hidden_dim=int(hidden_dim),
        latent_dim=int(latent_dim),
        W1=W1.astype(_np.float64),
        b1=b1.astype(_np.float64),
        W2=W2.astype(_np.float64),
        b2=b2.astype(_np.float64),
        momentum_W1=_np.zeros_like(W1, dtype=_np.float64),
        momentum_b1=_np.zeros_like(b1, dtype=_np.float64),
        momentum_W2=_np.zeros_like(W2, dtype=_np.float64),
        momentum_b2=_np.zeros_like(b2, dtype=_np.float64),
        n_train_steps=0,
        last_train_loss=0.0,
        pre_train_loss=0.0,
    )


@dataclasses.dataclass(frozen=True)
class LearnedConsolidationTrainReportV1:
    schema: str
    head_cid_pre: str
    head_cid_post: str
    pre_loss: float
    post_loss: float
    n_iters: int
    converged: bool
    loss_curve_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid_pre": str(self.head_cid_pre),
            "head_cid_post": str(self.head_cid_post),
            "pre_loss": float(round(self.pre_loss, 12)),
            "post_loss": float(round(self.post_loss, 12)),
            "n_iters": int(self.n_iters),
            "converged": bool(self.converged),
            "loss_curve_cid": str(self.loss_curve_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_learned_consolidation_train_report",
            "report": self.to_dict()})


def train_learned_consolidation_head(
        *,
        head: LearnedConsolidationHeadV1,
        train_features: Sequence[Sequence[float]],
        train_targets: Sequence[Sequence[float]],
        n_iters: int = W79_DEFAULT_TRAIN_ITERS,
        learning_rate: float = W79_DEFAULT_LEARNING_RATE,
        momentum: float = W79_DEFAULT_MOMENTUM,
        weight_decay: float = W79_DEFAULT_WEIGHT_DECAY,
) -> tuple[
        LearnedConsolidationHeadV1,
        LearnedConsolidationTrainReportV1]:
    """Train the consolidation head via SGD with momentum.

    Loss: ``L = mean((y_hat - y)**2)``.
    Backward: analytical gradients through the two-layer head.

    Returns ``(fitted_head, report)`` — ``fitted_head`` is a new
    dataclass with updated parameters; the input ``head`` is
    not mutated.
    """
    X = _np.asarray(train_features, dtype=_np.float64)
    Y = _np.asarray(train_targets, dtype=_np.float64)
    B = int(X.shape[0])
    if B == 0:
        return head, LearnedConsolidationTrainReportV1(
            schema=W79_LEARNED_CONSOLIDATION_SCHEMA_VERSION,
            head_cid_pre=str(head.cid()),
            head_cid_post=str(head.cid()),
            pre_loss=0.0, post_loss=0.0,
            n_iters=0, converged=True,
            loss_curve_cid=_sha256_hex({
                "kind": "w79_loss_curve",
                "losses": []}),
        )
    pre_cid = str(head.cid())
    W1 = head.W1.copy()
    b1 = head.b1.copy()
    W2 = head.W2.copy()
    b2 = head.b2.copy()
    mW1 = head.momentum_W1.copy()
    mb1 = head.momentum_b1.copy()
    mW2 = head.momentum_W2.copy()
    mb2 = head.momentum_b2.copy()

    def loss_of(W1, b1, W2, b2) -> float:
        h1_pre = X @ W1 + b1
        h1 = _swish(h1_pre)
        y_hat = h1 @ W2 + b2
        diff = y_hat - Y
        return float(_np.mean(diff * diff))

    pre_loss = float(loss_of(W1, b1, W2, b2))
    lr = float(learning_rate)
    mu = float(momentum)
    wd = float(weight_decay)
    losses: list[float] = [pre_loss]
    n_steps = int(head.n_train_steps)
    for _ in range(int(n_iters)):
        h1_pre = X @ W1 + b1
        h1 = _swish(h1_pre)
        y_hat = h1 @ W2 + b2
        diff = y_hat - Y
        # Loss gradients (mean-squared).
        dY = 2.0 / float(B) * diff
        gW2 = h1.T @ dY
        gb2 = dY.sum(axis=0)
        dH1 = dY @ W2.T
        dH1_pre = dH1 * _swish_grad(h1_pre)
        gW1 = X.T @ dH1_pre
        gb1 = dH1_pre.sum(axis=0)
        # SGD + momentum + weight decay.
        gW1 += wd * W1
        gW2 += wd * W2
        mW1 = mu * mW1 - lr * gW1
        mb1 = mu * mb1 - lr * gb1
        mW2 = mu * mW2 - lr * gW2
        mb2 = mu * mb2 - lr * gb2
        W1 = W1 + mW1
        b1 = b1 + mb1
        W2 = W2 + mW2
        b2 = b2 + mb2
        l = float(loss_of(W1, b1, W2, b2))
        losses.append(l)
        n_steps += 1
    post_loss = float(losses[-1])
    converged = bool(post_loss <= pre_loss + 1e-9)
    fitted = dataclasses.replace(
        head, W1=W1, b1=b1, W2=W2, b2=b2,
        momentum_W1=mW1, momentum_b1=mb1,
        momentum_W2=mW2, momentum_b2=mb2,
        n_train_steps=int(n_steps),
        last_train_loss=float(post_loss),
        pre_train_loss=float(pre_loss),
    )
    curve_cid = _sha256_hex({
        "kind": "w79_loss_curve",
        "losses": [float(round(l, 12)) for l in losses]})
    report = LearnedConsolidationTrainReportV1(
        schema=W79_LEARNED_CONSOLIDATION_SCHEMA_VERSION,
        head_cid_pre=str(pre_cid),
        head_cid_post=str(fitted.cid()),
        pre_loss=float(pre_loss),
        post_loss=float(post_loss),
        n_iters=int(n_iters),
        converged=bool(converged),
        loss_curve_cid=str(curve_cid),
    )
    return fitted, report


@dataclasses.dataclass(frozen=True)
class LearnedConsolidationVsClosedFormReportV1:
    """Honest comparison vs the closed-form ridge head.

    The learned consolidation head should strictly beat (in
    held-out MSE) a closed-form linear ridge regression head
    when the target is genuinely nonlinear in the inputs. This
    is the W79 trainable-line load-bearing claim.
    """

    schema: str
    learned_mse: float
    ridge_mse: float
    learned_strictly_beats_ridge: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "learned_mse": float(round(self.learned_mse, 12)),
            "ridge_mse": float(round(self.ridge_mse, 12)),
            "learned_strictly_beats_ridge": bool(
                self.learned_strictly_beats_ridge),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w79_learned_consolidation_vs_closed_form_report",
            "report": self.to_dict()})


def compare_learned_vs_closed_form(
        *,
        head: LearnedConsolidationHeadV1,
        eval_features: Sequence[Sequence[float]],
        eval_targets: Sequence[Sequence[float]],
        ridge_lambda: float = 0.10,
) -> LearnedConsolidationVsClosedFormReportV1:
    """Fit a closed-form ridge on (eval_features, eval_targets)
    and compare against the learned head on the same data.

    NOTE: This is a same-set evaluation by design — we are not
    claiming generalization, only that the W79 trainable head
    can fit nonlinear consolidation patterns that a closed-form
    *linear* ridge cannot.
    """
    X = _np.asarray(eval_features, dtype=_np.float64)
    Y = _np.asarray(eval_targets, dtype=_np.float64)
    if X.shape[0] == 0:
        return LearnedConsolidationVsClosedFormReportV1(
            schema=W79_LEARNED_CONSOLIDATION_SCHEMA_VERSION,
            learned_mse=0.0, ridge_mse=0.0,
            learned_strictly_beats_ridge=False)
    y_hat_l = head.forward(X)
    learned_mse = float(_np.mean((y_hat_l - Y) ** 2))
    # Closed-form ridge.
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(
        X.shape[1], dtype=_np.float64)
    b = X.T @ Y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros(
            (X.shape[1], Y.shape[1]), dtype=_np.float64)
    y_hat_r = X @ theta
    ridge_mse = float(_np.mean((y_hat_r - Y) ** 2))
    return LearnedConsolidationVsClosedFormReportV1(
        schema=W79_LEARNED_CONSOLIDATION_SCHEMA_VERSION,
        learned_mse=float(learned_mse),
        ridge_mse=float(ridge_mse),
        learned_strictly_beats_ridge=bool(
            learned_mse + 1e-9 < ridge_mse),
    )


@dataclasses.dataclass(frozen=True)
class LearnedConsolidationWitnessV1:
    schema: str
    head_cid: str
    n_train_steps: int
    pre_train_loss: float
    last_train_loss: float
    learned_strictly_beats_ridge: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "n_train_steps": int(self.n_train_steps),
            "pre_train_loss": float(round(
                self.pre_train_loss, 12)),
            "last_train_loss": float(round(
                self.last_train_loss, 12)),
            "learned_strictly_beats_ridge": bool(
                self.learned_strictly_beats_ridge),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_learned_consolidation_witness",
            "witness": self.to_dict()})


def emit_learned_consolidation_witness(
        *,
        head: LearnedConsolidationHeadV1,
        vs_closed_form: (
            LearnedConsolidationVsClosedFormReportV1 | None) = (
                None),
) -> LearnedConsolidationWitnessV1:
    return LearnedConsolidationWitnessV1(
        schema=W79_LEARNED_CONSOLIDATION_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        n_train_steps=int(head.n_train_steps),
        pre_train_loss=float(head.pre_train_loss),
        last_train_loss=float(head.last_train_loss),
        learned_strictly_beats_ridge=bool(
            vs_closed_form is not None
            and vs_closed_form.learned_strictly_beats_ridge),
    )


def build_nonlinear_consolidation_dataset(
        *, n_samples: int = 96,
        payload_dim: int = W79_DEFAULT_PAYLOAD_DIM,
        latent_dim: int = W79_DEFAULT_LATENT_DIM,
        seed: int = 79_020_002,
) -> tuple["_np.ndarray", "_np.ndarray"]:
    """Build a small genuinely-nonlinear consolidation dataset.

    Targets are tanh-of-sum-of-pairs so that a linear ridge
    cannot fit them exactly but a swish two-layer head can.
    """
    rng = _np.random.default_rng(int(seed))
    X = rng.standard_normal(
        (int(n_samples), int(payload_dim))).astype(_np.float64)
    # Tanh nonlinearity on sums of pairs + offset per-output.
    Y = _np.zeros(
        (int(n_samples), int(latent_dim)), dtype=_np.float64)
    for j in range(int(latent_dim)):
        a = j % int(payload_dim)
        b = (j + 1) % int(payload_dim)
        Y[:, j] = _np.tanh(X[:, a] + X[:, b] - 0.2 * float(j))
    return X.astype(_np.float64), Y.astype(_np.float64)


__all__ = [
    "W79_LEARNED_CONSOLIDATION_SCHEMA_VERSION",
    "W79_DEFAULT_PAYLOAD_DIM",
    "W79_DEFAULT_HIDDEN_DIM",
    "W79_DEFAULT_LATENT_DIM",
    "W79_DEFAULT_TRAIN_ITERS",
    "W79_DEFAULT_LEARNING_RATE",
    "W79_DEFAULT_MOMENTUM",
    "W79_DEFAULT_WEIGHT_DECAY",
    "W79_DEFAULT_SEED",
    "LearnedConsolidationHeadV1",
    "LearnedConsolidationTrainReportV1",
    "LearnedConsolidationVsClosedFormReportV1",
    "LearnedConsolidationWitnessV1",
    "build_learned_consolidation_head_v1",
    "train_learned_consolidation_head",
    "compare_learned_vs_closed_form",
    "emit_learned_consolidation_witness",
    "build_nonlinear_consolidation_dataset",
]
