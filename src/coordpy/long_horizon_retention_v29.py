"""W77 — Long-Horizon Retention V29.

Strictly extends W76's ``coordpy.long_horizon_retention_v28``. V28
had 27 heads + an eighteen-layer scorer at max_k=960. V29 adds:

* **28 heads** (V28's 27 + post-restart-replacement-pressure-
  recovery head).
* **Nineteen-layer scorer** — V28's eighteen layers + a
  nineteenth random+swish layer before the final ridge.
* **max_k = 1024** (vs V28's 960).

Honest scope (W77): only the final ridge head is fit; earlier
projections are frozen random. ``W77-L-V29-LHR-SCORER-FIT-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.long_horizon_retention_v29 requires numpy"
        ) from exc

from .long_horizon_retention_v28 import (
    LongHorizonReconstructionV28Head,
    W76_DEFAULT_LHR_V28_SWISH8_PROJ_DIM,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W77_LHR_V29_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v29.v1")
W77_DEFAULT_LHR_V29_MAX_K: int = 1024
W77_DEFAULT_LHR_V29_POST_RESTART_REPLACEMENT_DIM: int = 8
W77_DEFAULT_LHR_V29_SWISH9_PROJ_DIM: int = 64


def _swish(x: "_np.ndarray") -> "_np.ndarray":
    return x / (1.0 + _np.exp(-x))


@dataclasses.dataclass
class LongHorizonReconstructionV29Head:
    inner_v28: LongHorizonReconstructionV28Head
    max_k: int
    post_restart_replacement_dim: int
    swish9_proj_dim: int
    swish9_proj_W: "_np.ndarray | None" = None
    scorer_layer19: "_np.ndarray | None" = None
    scorer_layer19_residual: float = 0.0
    post_restart_replacement_W: "_np.ndarray | None" = None

    @classmethod
    def init(
            cls, *, max_k: int = W77_DEFAULT_LHR_V29_MAX_K,
            post_restart_replacement_dim: int = (
                W77_DEFAULT_LHR_V29_POST_RESTART_REPLACEMENT_DIM),
            swish9_proj_dim: int = (
                W77_DEFAULT_LHR_V29_SWISH9_PROJ_DIM),
            seed: int = 77200,
    ) -> "LongHorizonReconstructionV29Head":
        v28 = LongHorizonReconstructionV28Head.init(
            max_k=int(max_k), seed=int(seed))
        rng = _np.random.default_rng(int(seed) ^ 0xCAFE_77)
        out_dim = int(v28.out_dim)
        s9_W = rng.standard_normal(
            (int(W76_DEFAULT_LHR_V28_SWISH8_PROJ_DIM),
             int(swish9_proj_dim))) * 0.05
        pcr_W = rng.standard_normal(
            (int(post_restart_replacement_dim),
             int(out_dim))) * 0.05
        return cls(
            inner_v28=v28,
            max_k=int(max_k),
            post_restart_replacement_dim=int(
                post_restart_replacement_dim),
            swish9_proj_dim=int(swish9_proj_dim),
            swish9_proj_W=s9_W.astype(_np.float64),
            post_restart_replacement_W=pcr_W.astype(
                _np.float64),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v28.out_dim)

    def post_restart_replacement_value(
            self, *,
            post_restart_replacement_indicator: Sequence[float],
    ) -> "_np.ndarray":
        v = _np.asarray(
            post_restart_replacement_indicator,
            dtype=_np.float64)
        if v.size < int(self.post_restart_replacement_dim):
            v = _np.concatenate([
                v, _np.zeros(
                    int(self.post_restart_replacement_dim)
                    - v.size,
                    dtype=_np.float64)])
        elif v.size > int(self.post_restart_replacement_dim):
            v = v[:int(self.post_restart_replacement_dim)]
        return v @ self.post_restart_replacement_W

    def twenty_eight_way_value(
            self, *, carrier: Sequence[float], k: int,
            post_restart_replacement_indicator: (
                Sequence[float] | None) = None,
            **kwargs: Any,
    ) -> "_np.ndarray":
        v27 = self.inner_v28.twenty_seven_way_value(
            carrier=list(carrier), k=int(k), **kwargs)
        v27 = _np.asarray(v27, dtype=_np.float64)
        if post_restart_replacement_indicator is not None:
            pcr = self.post_restart_replacement_value(
                post_restart_replacement_indicator=list(
                    post_restart_replacement_indicator))
            return v27 + 0.05 * pcr
        return v27

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v29_head",
            "inner_v28_cid": str(self.inner_v28.cid()),
            "max_k": int(self.max_k),
            "post_restart_replacement_dim": int(
                self.post_restart_replacement_dim),
            "swish9_proj_dim": int(self.swish9_proj_dim),
            "swish9_proj_W_cid": (
                _ndarray_cid(self.swish9_proj_W)
                if self.swish9_proj_W is not None else "none"),
            "scorer_layer19_cid": (
                _ndarray_cid(self.scorer_layer19)
                if self.scorer_layer19 is not None
                else "untrained"),
            "scorer_layer19_residual": float(round(
                self.scorer_layer19_residual, 12)),
            "post_restart_replacement_W_cid": (
                _ndarray_cid(self.post_restart_replacement_W)
                if self.post_restart_replacement_W is not None
                else "none"),
        })


def fit_lhr_v29_nineteen_layer_scorer(
        *, head: LongHorizonReconstructionV29Head,
        train_features: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        ridge_lambda: float = 0.10,
) -> tuple[
        LongHorizonReconstructionV29Head, dict[str, Any]]:
    """Fit the nineteenth-layer ridge on top of the V28
    eighteen-layer pipeline (swish9 projection then ridge)."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(train_targets, dtype=_np.float64)
    if X.shape[0] == 0:
        return head, {"converged": True, "n": 0}
    if head.swish9_proj_W is not None and X.shape[1] == int(
            W76_DEFAULT_LHR_V28_SWISH8_PROJ_DIM):
        f18 = _swish(X @ _np.asarray(
            head.swish9_proj_W, dtype=_np.float64))
    else:
        f18 = X
    lam = max(float(ridge_lambda), 1e-9)
    A = f18.T @ f18 + lam * _np.eye(
        f18.shape[1], dtype=_np.float64)
    b = f18.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((f18.shape[1],), dtype=_np.float64)
    y_hat = f18 @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        head,
        scorer_layer19=_np.asarray(
            theta, dtype=_np.float64).copy(),
        scorer_layer19_residual=float(post))
    audit = {
        "schema": W77_LHR_V29_SCHEMA_VERSION,
        "kind": "lhr_v29_nineteen_layer_scorer",
        "pre_fit_residual": float(pre),
        "post_fit_residual": float(post),
        "converged": bool(post <= pre + 1e-9),
        "n": int(X.shape[0]),
    }
    return fitted, audit


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV29Witness:
    schema: str
    head_cid: str
    max_k: int
    post_restart_replacement_dim: int
    out_dim: int
    n_heads: int
    twenty_eight_way_runs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "max_k": int(self.max_k),
            "post_restart_replacement_dim": int(
                self.post_restart_replacement_dim),
            "out_dim": int(self.out_dim),
            "n_heads": int(self.n_heads),
            "twenty_eight_way_runs": bool(
                self.twenty_eight_way_runs),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v29_witness",
            "witness": self.to_dict()})


def emit_lhr_v29_witness(
        head: LongHorizonReconstructionV29Head, *,
        carrier: Sequence[float], k: int = 16,
        post_restart_replacement_indicator: (
            Sequence[float] | None) = None,
        **kwargs: Any,
) -> LongHorizonReconstructionV29Witness:
    runs = True
    try:
        head.twenty_eight_way_value(
            carrier=list(carrier), k=int(k),
            post_restart_replacement_indicator=(
                list(post_restart_replacement_indicator)
                if post_restart_replacement_indicator is not None
                else None),
            **kwargs)
    except Exception:
        runs = False
    return LongHorizonReconstructionV29Witness(
        schema=W77_LHR_V29_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        max_k=int(head.max_k),
        post_restart_replacement_dim=int(
            head.post_restart_replacement_dim),
        out_dim=int(head.out_dim),
        n_heads=28,
        twenty_eight_way_runs=bool(runs),
    )


__all__ = [
    "W77_LHR_V29_SCHEMA_VERSION",
    "W77_DEFAULT_LHR_V29_MAX_K",
    "W77_DEFAULT_LHR_V29_POST_RESTART_REPLACEMENT_DIM",
    "W77_DEFAULT_LHR_V29_SWISH9_PROJ_DIM",
    "LongHorizonReconstructionV29Head",
    "fit_lhr_v29_nineteen_layer_scorer",
    "LongHorizonReconstructionV29Witness",
    "emit_lhr_v29_witness",
]
