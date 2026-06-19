"""W67 M15 — Long-Horizon Retention V19.

Strictly extends W66's ``coordpy.long_horizon_retention_v18``. V18
had 17 heads + an eight-layer scorer at max_k=320. V19 adds:

* **18 heads** (V18's 17 + role-dropout-recovery head).
* **Nine-layer scorer** — V18's eight layers + a ninth
  random+mish layer before the final ridge.
* **max_k = 384** (vs V18's 320).

Honest scope (W67)
------------------

* Only the final ridge head is fit. The first eight layers are
  frozen random projections. ``W67-L-V19-LHR-SCORER-FIT-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.long_horizon_retention_v19 requires numpy"
        ) from exc

from .long_horizon_retention_v18 import (
    LongHorizonReconstructionV18Head,
    W66_DEFAULT_LHR_V18_SWISH_PROJ_DIM,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W67_LHR_V19_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v19.v1")
W67_DEFAULT_LHR_V19_MAX_K: int = 384
W67_DEFAULT_LHR_V19_ROLE_DROPOUT_DIM: int = 8
W67_DEFAULT_LHR_V19_MISH_PROJ_DIM: int = 32


def _mish(x: "_np.ndarray") -> "_np.ndarray":
    return x * _np.tanh(_np.log1p(_np.exp(x)))


@dataclasses.dataclass
class LongHorizonReconstructionV19Head:
    inner_v18: LongHorizonReconstructionV18Head
    max_k: int
    role_dropout_dim: int
    mish_proj_dim: int
    mish_proj_W: "_np.ndarray | None" = None
    scorer_layer9: "_np.ndarray | None" = None
    scorer_layer9_residual: float = 0.0
    role_dropout_W: "_np.ndarray | None" = None

    @classmethod
    def init(
            cls, *, max_k: int = W67_DEFAULT_LHR_V19_MAX_K,
            role_dropout_dim: int = (
                W67_DEFAULT_LHR_V19_ROLE_DROPOUT_DIM),
            mish_proj_dim: int = (
                W67_DEFAULT_LHR_V19_MISH_PROJ_DIM),
            seed: int = 67190,
    ) -> "LongHorizonReconstructionV19Head":
        v18 = LongHorizonReconstructionV18Head.init(
            max_k=int(max_k), seed=int(seed))
        rng = _np.random.default_rng(int(seed) ^ 0xCAFE_67)
        out_dim = int(v18.out_dim)
        mi_W = rng.standard_normal(
            (int(W66_DEFAULT_LHR_V18_SWISH_PROJ_DIM),
             int(mish_proj_dim))) * 0.05
        rd_W = rng.standard_normal(
            (int(role_dropout_dim), int(out_dim))) * 0.05
        return cls(
            inner_v18=v18,
            max_k=int(max_k),
            role_dropout_dim=int(role_dropout_dim),
            mish_proj_dim=int(mish_proj_dim),
            mish_proj_W=mi_W.astype(_np.float64),
            role_dropout_W=rd_W.astype(_np.float64),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v18.out_dim)

    def role_dropout_value(
            self, *,
            role_dropout_indicator: Sequence[float],
    ) -> "_np.ndarray":
        v = _np.asarray(
            role_dropout_indicator, dtype=_np.float64)
        if v.size < int(self.role_dropout_dim):
            v = _np.concatenate([
                v, _np.zeros(
                    int(self.role_dropout_dim) - v.size,
                    dtype=_np.float64)])
        elif v.size > int(self.role_dropout_dim):
            v = v[:int(self.role_dropout_dim)]
        return v @ self.role_dropout_W

    def eighteen_way_value(
            self, *, carrier: Sequence[float], k: int,
            role_dropout_indicator: (
                Sequence[float] | None) = None,
            **kwargs: Any,
    ) -> "_np.ndarray":
        v17 = self.inner_v18.seventeen_way_value(
            carrier=list(carrier), k=int(k), **kwargs)
        v17 = _np.asarray(v17, dtype=_np.float64)
        if role_dropout_indicator is not None:
            rd = self.role_dropout_value(
                role_dropout_indicator=list(
                    role_dropout_indicator))
            return v17 + 0.05 * rd
        return v17

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v19_head",
            "inner_v18_cid": str(self.inner_v18.cid()),
            "max_k": int(self.max_k),
            "role_dropout_dim": int(self.role_dropout_dim),
            "mish_proj_dim": int(self.mish_proj_dim),
            "mish_proj_W_cid": (
                _ndarray_cid(self.mish_proj_W)
                if self.mish_proj_W is not None else "none"),
            "scorer_layer9_cid": (
                _ndarray_cid(self.scorer_layer9)
                if self.scorer_layer9 is not None
                else "untrained"),
            "scorer_layer9_residual": float(round(
                self.scorer_layer9_residual, 12)),
            "role_dropout_W_cid": (
                _ndarray_cid(self.role_dropout_W)
                if self.role_dropout_W is not None
                else "none"),
        })


def fit_lhr_v19_nine_layer_scorer(
        *, head: LongHorizonReconstructionV19Head,
        train_features: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        ridge_lambda: float = 0.10,
) -> tuple[LongHorizonReconstructionV19Head, dict[str, Any]]:
    """Fit the ninth-layer ridge on top of the V18 eight-layer
    pipeline (mish projection then ridge)."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(train_targets, dtype=_np.float64)
    if X.shape[0] == 0:
        return head, {"converged": True, "n": 0}
    if head.mish_proj_W is not None and X.shape[1] == int(
            W66_DEFAULT_LHR_V18_SWISH_PROJ_DIM):
        f8 = _mish(X @ _np.asarray(
            head.mish_proj_W, dtype=_np.float64))
    else:
        f8 = X
    lam = max(float(ridge_lambda), 1e-9)
    A = f8.T @ f8 + lam * _np.eye(
        f8.shape[1], dtype=_np.float64)
    b = f8.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((f8.shape[1],), dtype=_np.float64)
    y_hat = f8 @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        head,
        scorer_layer9=_np.asarray(
            theta, dtype=_np.float64).copy(),
        scorer_layer9_residual=float(post))
    audit = {
        "schema": W67_LHR_V19_SCHEMA_VERSION,
        "kind": "lhr_v19_nine_layer_scorer",
        "pre_fit_residual": float(pre),
        "post_fit_residual": float(post),
        "converged": bool(post <= pre + 1e-9),
        "n": int(X.shape[0]),
    }
    return fitted, audit


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV19Witness:
    schema: str
    head_cid: str
    max_k: int
    role_dropout_dim: int
    out_dim: int
    n_heads: int
    eighteen_way_runs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "max_k": int(self.max_k),
            "role_dropout_dim": int(self.role_dropout_dim),
            "out_dim": int(self.out_dim),
            "n_heads": int(self.n_heads),
            "eighteen_way_runs": bool(self.eighteen_way_runs),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v19_witness",
            "witness": self.to_dict()})


def emit_lhr_v19_witness(
        head: LongHorizonReconstructionV19Head, *,
        carrier: Sequence[float], k: int = 16,
        role_dropout_indicator: (
            Sequence[float] | None) = None,
        **kwargs: Any,
) -> LongHorizonReconstructionV19Witness:
    runs = True
    try:
        head.eighteen_way_value(
            carrier=list(carrier), k=int(k),
            role_dropout_indicator=(
                list(role_dropout_indicator)
                if role_dropout_indicator is not None
                else None),
            **kwargs)
    except Exception:
        runs = False
    return LongHorizonReconstructionV19Witness(
        schema=W67_LHR_V19_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        max_k=int(head.max_k),
        role_dropout_dim=int(head.role_dropout_dim),
        out_dim=int(head.out_dim),
        n_heads=18,
        eighteen_way_runs=bool(runs),
    )


__all__ = [
    "W67_LHR_V19_SCHEMA_VERSION",
    "W67_DEFAULT_LHR_V19_MAX_K",
    "W67_DEFAULT_LHR_V19_ROLE_DROPOUT_DIM",
    "W67_DEFAULT_LHR_V19_MISH_PROJ_DIM",
    "LongHorizonReconstructionV19Head",
    "fit_lhr_v19_nine_layer_scorer",
    "LongHorizonReconstructionV19Witness",
    "emit_lhr_v19_witness",
]
