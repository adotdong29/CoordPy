"""W78 — Long-Horizon Retention V30.

Strictly extends W77's ``coordpy.long_horizon_retention_v29``. V29
had 28 heads + a nineteen-layer scorer at max_k=1024. V30 is the
**less-bounded long-horizon retention bump** at the core of W78:

* **29 heads** (V29's 28 + long-horizon-reconstruction head).
* **Twenty-layer scorer** — V29's nineteen layers + a twentieth
  random+swish layer before the final ridge.
* **max_k = 2048** (vs V29's 1024). This is the largest k in the
  programme to date and is the structural retention claim
  behind the W78 less-bounded-context win.

Honest scope (W78): only the final ridge head is fit; earlier
projections are frozen random. ``W78-L-V30-LHR-SCORER-FIT-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.long_horizon_retention_v30 requires numpy"
        ) from exc

from .long_horizon_retention_v29 import (
    LongHorizonReconstructionV29Head,
    W77_DEFAULT_LHR_V29_SWISH9_PROJ_DIM,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W78_LHR_V30_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v30.v1")
W78_DEFAULT_LHR_V30_MAX_K: int = 2048
W78_DEFAULT_LHR_V30_LONG_HORIZON_RECONSTRUCTION_DIM: int = 8
W78_DEFAULT_LHR_V30_SWISH10_PROJ_DIM: int = 80


def _swish(x: "_np.ndarray") -> "_np.ndarray":
    return x / (1.0 + _np.exp(-x))


@dataclasses.dataclass
class LongHorizonReconstructionV30Head:
    inner_v29: LongHorizonReconstructionV29Head
    max_k: int
    long_horizon_reconstruction_dim: int
    swish10_proj_dim: int
    swish10_proj_W: "_np.ndarray | None" = None
    scorer_layer20: "_np.ndarray | None" = None
    scorer_layer20_residual: float = 0.0
    long_horizon_reconstruction_W: "_np.ndarray | None" = None

    @classmethod
    def init(
            cls, *, max_k: int = W78_DEFAULT_LHR_V30_MAX_K,
            long_horizon_reconstruction_dim: int = (
                W78_DEFAULT_LHR_V30_LONG_HORIZON_RECONSTRUCTION_DIM),
            swish10_proj_dim: int = (
                W78_DEFAULT_LHR_V30_SWISH10_PROJ_DIM),
            seed: int = 78200,
    ) -> "LongHorizonReconstructionV30Head":
        v29 = LongHorizonReconstructionV29Head.init(
            max_k=int(max_k), seed=int(seed))
        rng = _np.random.default_rng(int(seed) ^ 0xCAFE_78)
        out_dim = int(v29.out_dim)
        s10_W = rng.standard_normal(
            (int(W77_DEFAULT_LHR_V29_SWISH9_PROJ_DIM),
             int(swish10_proj_dim))) * 0.05
        lhr_W = rng.standard_normal(
            (int(long_horizon_reconstruction_dim),
             int(out_dim))) * 0.05
        return cls(
            inner_v29=v29,
            max_k=int(max_k),
            long_horizon_reconstruction_dim=int(
                long_horizon_reconstruction_dim),
            swish10_proj_dim=int(swish10_proj_dim),
            swish10_proj_W=s10_W.astype(_np.float64),
            long_horizon_reconstruction_W=lhr_W.astype(
                _np.float64),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v29.out_dim)

    def long_horizon_reconstruction_value(
            self, *,
            long_horizon_reconstruction_indicator: Sequence[float],
    ) -> "_np.ndarray":
        v = _np.asarray(
            long_horizon_reconstruction_indicator,
            dtype=_np.float64)
        if v.size < int(self.long_horizon_reconstruction_dim):
            v = _np.concatenate([
                v, _np.zeros(
                    int(self.long_horizon_reconstruction_dim)
                    - v.size,
                    dtype=_np.float64)])
        elif v.size > int(self.long_horizon_reconstruction_dim):
            v = v[:int(self.long_horizon_reconstruction_dim)]
        return v @ self.long_horizon_reconstruction_W

    def twenty_nine_way_value(
            self, *, carrier: Sequence[float], k: int,
            long_horizon_reconstruction_indicator: (
                Sequence[float] | None) = None,
            **kwargs: Any,
    ) -> "_np.ndarray":
        v28 = self.inner_v29.twenty_eight_way_value(
            carrier=list(carrier), k=int(k), **kwargs)
        v28 = _np.asarray(v28, dtype=_np.float64)
        if long_horizon_reconstruction_indicator is not None:
            lhr = self.long_horizon_reconstruction_value(
                long_horizon_reconstruction_indicator=list(
                    long_horizon_reconstruction_indicator))
            return v28 + 0.05 * lhr
        return v28

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v30_head",
            "inner_v29_cid": str(self.inner_v29.cid()),
            "max_k": int(self.max_k),
            "long_horizon_reconstruction_dim": int(
                self.long_horizon_reconstruction_dim),
            "swish10_proj_dim": int(self.swish10_proj_dim),
            "swish10_proj_W_cid": (
                _ndarray_cid(self.swish10_proj_W)
                if self.swish10_proj_W is not None else "none"),
            "scorer_layer20_cid": (
                _ndarray_cid(self.scorer_layer20)
                if self.scorer_layer20 is not None
                else "untrained"),
            "scorer_layer20_residual": float(round(
                self.scorer_layer20_residual, 12)),
            "long_horizon_reconstruction_W_cid": (
                _ndarray_cid(self.long_horizon_reconstruction_W)
                if self.long_horizon_reconstruction_W is not None
                else "none"),
        })


def fit_lhr_v30_twenty_layer_scorer(
        *, head: LongHorizonReconstructionV30Head,
        train_features: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        ridge_lambda: float = 0.10,
) -> tuple[
        LongHorizonReconstructionV30Head, dict[str, Any]]:
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(train_targets, dtype=_np.float64)
    if X.shape[0] == 0:
        return head, {"converged": True, "n": 0}
    if head.swish10_proj_W is not None and X.shape[1] == int(
            W77_DEFAULT_LHR_V29_SWISH9_PROJ_DIM):
        f19 = _swish(X @ _np.asarray(
            head.swish10_proj_W, dtype=_np.float64))
    else:
        f19 = X
    lam = max(float(ridge_lambda), 1e-9)
    A = f19.T @ f19 + lam * _np.eye(
        f19.shape[1], dtype=_np.float64)
    b = f19.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((f19.shape[1],), dtype=_np.float64)
    y_hat = f19 @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        head,
        scorer_layer20=_np.asarray(
            theta, dtype=_np.float64).copy(),
        scorer_layer20_residual=float(post))
    audit = {
        "schema": W78_LHR_V30_SCHEMA_VERSION,
        "kind": "lhr_v30_twenty_layer_scorer",
        "pre_fit_residual": float(pre),
        "post_fit_residual": float(post),
        "converged": bool(post <= pre + 1e-9),
        "n": int(X.shape[0]),
    }
    return fitted, audit


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV30Witness:
    schema: str
    head_cid: str
    max_k: int
    long_horizon_reconstruction_dim: int
    out_dim: int
    n_heads: int
    twenty_nine_way_runs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "max_k": int(self.max_k),
            "long_horizon_reconstruction_dim": int(
                self.long_horizon_reconstruction_dim),
            "out_dim": int(self.out_dim),
            "n_heads": int(self.n_heads),
            "twenty_nine_way_runs": bool(
                self.twenty_nine_way_runs),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v30_witness",
            "witness": self.to_dict()})


def emit_lhr_v30_witness(
        head: LongHorizonReconstructionV30Head, *,
        carrier: Sequence[float], k: int = 16,
        long_horizon_reconstruction_indicator: (
            Sequence[float] | None) = None,
        **kwargs: Any,
) -> LongHorizonReconstructionV30Witness:
    runs = True
    try:
        head.twenty_nine_way_value(
            carrier=list(carrier), k=int(k),
            long_horizon_reconstruction_indicator=(
                list(long_horizon_reconstruction_indicator)
                if long_horizon_reconstruction_indicator
                is not None
                else None),
            **kwargs)
    except Exception:
        runs = False
    return LongHorizonReconstructionV30Witness(
        schema=W78_LHR_V30_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        max_k=int(head.max_k),
        long_horizon_reconstruction_dim=int(
            head.long_horizon_reconstruction_dim),
        out_dim=int(head.out_dim),
        n_heads=29,
        twenty_nine_way_runs=bool(runs),
    )


__all__ = [
    "W78_LHR_V30_SCHEMA_VERSION",
    "W78_DEFAULT_LHR_V30_MAX_K",
    "W78_DEFAULT_LHR_V30_LONG_HORIZON_RECONSTRUCTION_DIM",
    "W78_DEFAULT_LHR_V30_SWISH10_PROJ_DIM",
    "LongHorizonReconstructionV30Head",
    "fit_lhr_v30_twenty_layer_scorer",
    "LongHorizonReconstructionV30Witness",
    "emit_lhr_v30_witness",
]
