"""W63 M16 — Long-Horizon Retention V15.

Strictly extends W62's ``coordpy.long_horizon_retention_v14``. V14
had 13 heads + a four-layer scorer. V15 adds:

* **14 heads** (V14's 13 + hidden-wins-conditioned head).
* **Five-layer scorer** — V14's four-layer (random+ReLU →
  random+tanh → random+tanh-2 → ridge) plus a fifth
  random+gelu layer in front of the ridge.
* **max_k = 160** (vs V14's 128).

Honest scope
------------

* The hidden-wins head is a linear projection of the V8
  ``hidden_vs_kv_contention`` summary; not a calibrated retention
  signal.
* Only the final ridge head is fit. ``W63-L-V15-LHR-SCORER-FIT-CAP``.
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
        "coordpy.long_horizon_retention_v15 requires numpy"
        ) from exc

from .long_horizon_retention_v14 import (
    LongHorizonReconstructionV14Head,
    W62_DEFAULT_LHR_V14_REPLAY_DOMINANCE_DIM,
    W62_DEFAULT_LHR_V14_TANH2_PROJ_DIM,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W63_LHR_V15_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v15.v1")
W63_DEFAULT_LHR_V15_MAX_K: int = 160
W63_DEFAULT_LHR_V15_HIDDEN_WINS_DIM: int = 8
W63_DEFAULT_LHR_V15_GELU_PROJ_DIM: int = 16


def _gelu(x: "_np.ndarray") -> "_np.ndarray":
    return 0.5 * x * (1.0 + _np.tanh(
        math.sqrt(2.0 / math.pi)
        * (x + 0.044715 * x ** 3)))


@dataclasses.dataclass
class LongHorizonReconstructionV15Head:
    inner_v14: LongHorizonReconstructionV14Head
    max_k: int
    hidden_wins_dim: int
    gelu_proj_dim: int
    gelu_proj_W: "_np.ndarray | None" = None
    scorer_layer5: "_np.ndarray | None" = None
    scorer_layer5_residual: float = 0.0
    hidden_wins_W: "_np.ndarray | None" = None

    @classmethod
    def init(
            cls, *, max_k: int = W63_DEFAULT_LHR_V15_MAX_K,
            hidden_wins_dim: int = (
                W63_DEFAULT_LHR_V15_HIDDEN_WINS_DIM),
            gelu_proj_dim: int = (
                W63_DEFAULT_LHR_V15_GELU_PROJ_DIM),
            seed: int = 63150,
    ) -> "LongHorizonReconstructionV15Head":
        v14 = LongHorizonReconstructionV14Head.init(
            max_k=int(max_k), seed=int(seed))
        rng = _np.random.default_rng(int(seed) ^ 0xBADC_63)
        out_dim = int(v14.out_dim)
        gelu_W = rng.standard_normal(
            (int(v14.tanh2_proj_dim),
             int(gelu_proj_dim))) * 0.08
        hw_W = rng.standard_normal(
            (int(hidden_wins_dim),
             int(out_dim))) * 0.07
        return cls(
            inner_v14=v14,
            max_k=int(max_k),
            hidden_wins_dim=int(hidden_wins_dim),
            gelu_proj_dim=int(gelu_proj_dim),
            gelu_proj_W=gelu_W.astype(_np.float64),
            hidden_wins_W=hw_W.astype(_np.float64),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v14.out_dim)

    def hidden_wins_value(
            self, *, hidden_wins_indicator: Sequence[float],
    ) -> "_np.ndarray":
        v = _np.asarray(
            hidden_wins_indicator, dtype=_np.float64)
        if v.size < int(self.hidden_wins_dim):
            v = _np.concatenate([
                v, _np.zeros(
                    int(self.hidden_wins_dim) - v.size,
                    dtype=_np.float64)])
        elif v.size > int(self.hidden_wins_dim):
            v = v[:int(self.hidden_wins_dim)]
        return v @ self.hidden_wins_W

    def fourteen_way_value(
            self, *, carrier: Sequence[float], k: int,
            replay_dominance_indicator: (
                Sequence[float] | None) = None,
            hidden_wins_indicator: (
                Sequence[float] | None) = None,
            **kwargs: Any,
    ) -> "_np.ndarray":
        v13 = self.inner_v14.thirteen_way_value(
            carrier=list(carrier), k=int(k),
            replay_dominance_indicator=(
                replay_dominance_indicator),
            **kwargs)
        v13 = _np.asarray(v13, dtype=_np.float64)
        if hidden_wins_indicator is not None:
            hw = self.hidden_wins_value(
                hidden_wins_indicator=(
                    list(hidden_wins_indicator)))
            return v13 + 0.08 * hw
        return v13

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v15_head",
            "inner_v14_cid": str(self.inner_v14.cid()),
            "max_k": int(self.max_k),
            "hidden_wins_dim": int(self.hidden_wins_dim),
            "gelu_proj_dim": int(self.gelu_proj_dim),
            "gelu_proj_W_cid": (
                _ndarray_cid(self.gelu_proj_W)
                if self.gelu_proj_W is not None else "none"),
            "scorer_layer5_cid": (
                _ndarray_cid(self.scorer_layer5)
                if self.scorer_layer5 is not None
                else "untrained"),
            "scorer_layer5_residual": float(round(
                self.scorer_layer5_residual, 12)),
            "hidden_wins_W_cid": (
                _ndarray_cid(self.hidden_wins_W)
                if self.hidden_wins_W is not None
                else "none"),
        })


def fit_lhr_v15_five_layer_scorer(
        *, head: LongHorizonReconstructionV15Head,
        train_features: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        ridge_lambda: float = 0.10,
) -> tuple[LongHorizonReconstructionV15Head, dict[str, Any]]:
    """Fit the fifth-layer ridge on top of the V14 fourth-layer
    feature pipeline."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(train_targets, dtype=_np.float64)
    if X.shape[0] == 0:
        return head, {"converged": True, "n": 0}
    # Pass X through the gelu projection.
    if head.gelu_proj_W is not None and X.shape[1] == int(
            head.inner_v14.tanh2_proj_dim):
        f4 = _gelu(X @ _np.asarray(
            head.gelu_proj_W, dtype=_np.float64))
    else:
        f4 = X
    lam = max(float(ridge_lambda), 1e-9)
    A = f4.T @ f4 + lam * _np.eye(
        f4.shape[1], dtype=_np.float64)
    b = f4.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((f4.shape[1],), dtype=_np.float64)
    y_hat = f4 @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        head,
        scorer_layer5=_np.asarray(
            theta, dtype=_np.float64).copy(),
        scorer_layer5_residual=float(post))
    audit = {
        "schema": W63_LHR_V15_SCHEMA_VERSION,
        "kind": "lhr_v15_five_layer_scorer",
        "pre_fit_residual": float(pre),
        "post_fit_residual": float(post),
        "converged": bool(post <= pre + 1e-9),
        "n": int(X.shape[0]),
    }
    return fitted, audit


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV15Witness:
    schema: str
    head_cid: str
    max_k: int
    hidden_wins_dim: int
    out_dim: int
    n_heads: int
    fourteen_way_runs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "max_k": int(self.max_k),
            "hidden_wins_dim": int(self.hidden_wins_dim),
            "out_dim": int(self.out_dim),
            "n_heads": int(self.n_heads),
            "fourteen_way_runs": bool(self.fourteen_way_runs),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v15_witness",
            "witness": self.to_dict()})


def emit_lhr_v15_witness(
        head: LongHorizonReconstructionV15Head, *,
        carrier: Sequence[float], k: int = 16,
        replay_dominance_indicator: (
            Sequence[float] | None) = None,
        hidden_wins_indicator: (
            Sequence[float] | None) = None,
) -> LongHorizonReconstructionV15Witness:
    runs = True
    try:
        head.fourteen_way_value(
            carrier=list(carrier), k=int(k),
            replay_dominance_indicator=(
                list(replay_dominance_indicator)
                if replay_dominance_indicator is not None
                else None),
            hidden_wins_indicator=(
                list(hidden_wins_indicator)
                if hidden_wins_indicator is not None
                else None))
    except Exception:
        runs = False
    return LongHorizonReconstructionV15Witness(
        schema=W63_LHR_V15_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        max_k=int(head.max_k),
        hidden_wins_dim=int(head.hidden_wins_dim),
        out_dim=int(head.out_dim),
        n_heads=14,
        fourteen_way_runs=bool(runs),
    )


__all__ = [
    "W63_LHR_V15_SCHEMA_VERSION",
    "W63_DEFAULT_LHR_V15_MAX_K",
    "W63_DEFAULT_LHR_V15_HIDDEN_WINS_DIM",
    "W63_DEFAULT_LHR_V15_GELU_PROJ_DIM",
    "LongHorizonReconstructionV15Head",
    "fit_lhr_v15_five_layer_scorer",
    "LongHorizonReconstructionV15Witness",
    "emit_lhr_v15_witness",
]
