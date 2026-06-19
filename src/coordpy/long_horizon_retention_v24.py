"""W72 — Long-Horizon Retention V24.

Strictly extends W71's ``coordpy.long_horizon_retention_v23``. V23
had 22 heads + a thirteen-layer scorer at max_k=640. V24 adds:

* **23 heads** (V23's 22 + rejoin-pressure-recovery head).
* **Fourteen-layer scorer** — V23's thirteen layers + a fourteenth
  random+swish layer before the final ridge.
* **max_k = 704** (vs V23's 640).

Honest scope (W72): only the final ridge head is fit; earlier
projections are frozen random. ``W72-L-V24-LHR-SCORER-FIT-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.long_horizon_retention_v24 requires numpy"
        ) from exc

from .long_horizon_retention_v23 import (
    LongHorizonReconstructionV23Head,
    W71_DEFAULT_LHR_V23_SWISH3_PROJ_DIM,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W72_LHR_V24_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v24.v1")
W72_DEFAULT_LHR_V24_MAX_K: int = 704
W72_DEFAULT_LHR_V24_REJOIN_DIM: int = 8
W72_DEFAULT_LHR_V24_SWISH4_PROJ_DIM: int = 48


def _swish(x: "_np.ndarray") -> "_np.ndarray":
    return x / (1.0 + _np.exp(-x))


@dataclasses.dataclass
class LongHorizonReconstructionV24Head:
    inner_v23: LongHorizonReconstructionV23Head
    max_k: int
    rejoin_dim: int
    swish4_proj_dim: int
    swish4_proj_W: "_np.ndarray | None" = None
    scorer_layer14: "_np.ndarray | None" = None
    scorer_layer14_residual: float = 0.0
    rejoin_W: "_np.ndarray | None" = None

    @classmethod
    def init(
            cls, *, max_k: int = W72_DEFAULT_LHR_V24_MAX_K,
            rejoin_dim: int = W72_DEFAULT_LHR_V24_REJOIN_DIM,
            swish4_proj_dim: int = (
                W72_DEFAULT_LHR_V24_SWISH4_PROJ_DIM),
            seed: int = 72200,
    ) -> "LongHorizonReconstructionV24Head":
        v23 = LongHorizonReconstructionV23Head.init(
            max_k=int(max_k), seed=int(seed))
        rng = _np.random.default_rng(int(seed) ^ 0xCAFE_72)
        out_dim = int(v23.out_dim)
        s4_W = rng.standard_normal(
            (int(W71_DEFAULT_LHR_V23_SWISH3_PROJ_DIM),
             int(swish4_proj_dim))) * 0.05
        rj_W = rng.standard_normal(
            (int(rejoin_dim), int(out_dim))) * 0.05
        return cls(
            inner_v23=v23,
            max_k=int(max_k),
            rejoin_dim=int(rejoin_dim),
            swish4_proj_dim=int(swish4_proj_dim),
            swish4_proj_W=s4_W.astype(_np.float64),
            rejoin_W=rj_W.astype(_np.float64),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v23.out_dim)

    def rejoin_value(
            self, *, rejoin_indicator: Sequence[float],
    ) -> "_np.ndarray":
        v = _np.asarray(
            rejoin_indicator, dtype=_np.float64)
        if v.size < int(self.rejoin_dim):
            v = _np.concatenate([
                v, _np.zeros(
                    int(self.rejoin_dim) - v.size,
                    dtype=_np.float64)])
        elif v.size > int(self.rejoin_dim):
            v = v[:int(self.rejoin_dim)]
        return v @ self.rejoin_W

    def twenty_three_way_value(
            self, *, carrier: Sequence[float], k: int,
            partial_contradiction_indicator: (
                Sequence[float] | None) = None,
            multi_branch_rejoin_indicator: (
                Sequence[float] | None) = None,
            repair_dominance_indicator: (
                Sequence[float] | None) = None,
            restart_indicator: (
                Sequence[float] | None) = None,
            rejoin_indicator: (
                Sequence[float] | None) = None,
            **kwargs: Any,
    ) -> "_np.ndarray":
        v22 = self.inner_v23.twenty_two_way_value(
            carrier=list(carrier), k=int(k),
            partial_contradiction_indicator=(
                list(partial_contradiction_indicator)
                if partial_contradiction_indicator is not None
                else None),
            multi_branch_rejoin_indicator=(
                list(multi_branch_rejoin_indicator)
                if multi_branch_rejoin_indicator is not None
                else None),
            repair_dominance_indicator=(
                list(repair_dominance_indicator)
                if repair_dominance_indicator is not None
                else None),
            restart_indicator=(
                list(restart_indicator)
                if restart_indicator is not None
                else None),
            **kwargs)
        v22 = _np.asarray(v22, dtype=_np.float64)
        if rejoin_indicator is not None:
            rj = self.rejoin_value(
                rejoin_indicator=list(rejoin_indicator))
            return v22 + 0.05 * rj
        return v22

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v24_head",
            "inner_v23_cid": str(self.inner_v23.cid()),
            "max_k": int(self.max_k),
            "rejoin_dim": int(self.rejoin_dim),
            "swish4_proj_dim": int(self.swish4_proj_dim),
            "swish4_proj_W_cid": (
                _ndarray_cid(self.swish4_proj_W)
                if self.swish4_proj_W is not None else "none"),
            "scorer_layer14_cid": (
                _ndarray_cid(self.scorer_layer14)
                if self.scorer_layer14 is not None
                else "untrained"),
            "scorer_layer14_residual": float(round(
                self.scorer_layer14_residual, 12)),
            "rejoin_W_cid": (
                _ndarray_cid(self.rejoin_W)
                if self.rejoin_W is not None else "none"),
        })


def fit_lhr_v24_fourteen_layer_scorer(
        *, head: LongHorizonReconstructionV24Head,
        train_features: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        ridge_lambda: float = 0.10,
) -> tuple[
        LongHorizonReconstructionV24Head, dict[str, Any]]:
    """Fit the fourteenth-layer ridge on top of the V23 thirteen-
    layer pipeline (swish4 projection then ridge)."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(train_targets, dtype=_np.float64)
    if X.shape[0] == 0:
        return head, {"converged": True, "n": 0}
    if head.swish4_proj_W is not None and X.shape[1] == int(
            W71_DEFAULT_LHR_V23_SWISH3_PROJ_DIM):
        f13 = _swish(X @ _np.asarray(
            head.swish4_proj_W, dtype=_np.float64))
    else:
        f13 = X
    lam = max(float(ridge_lambda), 1e-9)
    A = f13.T @ f13 + lam * _np.eye(
        f13.shape[1], dtype=_np.float64)
    b = f13.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((f13.shape[1],), dtype=_np.float64)
    y_hat = f13 @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        head,
        scorer_layer14=_np.asarray(
            theta, dtype=_np.float64).copy(),
        scorer_layer14_residual=float(post))
    audit = {
        "schema": W72_LHR_V24_SCHEMA_VERSION,
        "kind": "lhr_v24_fourteen_layer_scorer",
        "pre_fit_residual": float(pre),
        "post_fit_residual": float(post),
        "converged": bool(post <= pre + 1e-9),
        "n": int(X.shape[0]),
    }
    return fitted, audit


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV24Witness:
    schema: str
    head_cid: str
    max_k: int
    rejoin_dim: int
    out_dim: int
    n_heads: int
    twenty_three_way_runs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "max_k": int(self.max_k),
            "rejoin_dim": int(self.rejoin_dim),
            "out_dim": int(self.out_dim),
            "n_heads": int(self.n_heads),
            "twenty_three_way_runs": bool(
                self.twenty_three_way_runs),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v24_witness",
            "witness": self.to_dict()})


def emit_lhr_v24_witness(
        head: LongHorizonReconstructionV24Head, *,
        carrier: Sequence[float], k: int = 16,
        partial_contradiction_indicator: (
            Sequence[float] | None) = None,
        multi_branch_rejoin_indicator: (
            Sequence[float] | None) = None,
        repair_dominance_indicator: (
            Sequence[float] | None) = None,
        restart_indicator: (
            Sequence[float] | None) = None,
        rejoin_indicator: (
            Sequence[float] | None) = None,
        **kwargs: Any,
) -> LongHorizonReconstructionV24Witness:
    runs = True
    try:
        head.twenty_three_way_value(
            carrier=list(carrier), k=int(k),
            partial_contradiction_indicator=(
                list(partial_contradiction_indicator)
                if partial_contradiction_indicator is not None
                else None),
            multi_branch_rejoin_indicator=(
                list(multi_branch_rejoin_indicator)
                if multi_branch_rejoin_indicator is not None
                else None),
            repair_dominance_indicator=(
                list(repair_dominance_indicator)
                if repair_dominance_indicator is not None
                else None),
            restart_indicator=(
                list(restart_indicator)
                if restart_indicator is not None
                else None),
            rejoin_indicator=(
                list(rejoin_indicator)
                if rejoin_indicator is not None
                else None),
            **kwargs)
    except Exception:
        runs = False
    return LongHorizonReconstructionV24Witness(
        schema=W72_LHR_V24_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        max_k=int(head.max_k),
        rejoin_dim=int(head.rejoin_dim),
        out_dim=int(head.out_dim),
        n_heads=23,
        twenty_three_way_runs=bool(runs),
    )


__all__ = [
    "W72_LHR_V24_SCHEMA_VERSION",
    "W72_DEFAULT_LHR_V24_MAX_K",
    "W72_DEFAULT_LHR_V24_REJOIN_DIM",
    "W72_DEFAULT_LHR_V24_SWISH4_PROJ_DIM",
    "LongHorizonReconstructionV24Head",
    "fit_lhr_v24_fourteen_layer_scorer",
    "LongHorizonReconstructionV24Witness",
    "emit_lhr_v24_witness",
]
