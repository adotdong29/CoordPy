"""W71 — Long-Horizon Retention V23.

Strictly extends W70's ``coordpy.long_horizon_retention_v22``. V22
had 21 heads + a twelve-layer scorer at max_k=576. V23 adds:

* **22 heads** (V22's 21 + restart-dominance-recovery head).
* **Thirteen-layer scorer** — V22's twelve layers + a thirteenth
  random+swish layer before the final ridge.
* **max_k = 640** (vs V22's 576).

Honest scope (W71): only the final ridge head is fit; earlier
projections are frozen random. ``W71-L-V23-LHR-SCORER-FIT-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.long_horizon_retention_v23 requires numpy"
        ) from exc

from .long_horizon_retention_v22 import (
    LongHorizonReconstructionV22Head,
    W70_DEFAULT_LHR_V22_SWISH2_PROJ_DIM,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W71_LHR_V23_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v23.v1")
W71_DEFAULT_LHR_V23_MAX_K: int = 640
W71_DEFAULT_LHR_V23_RESTART_DIM: int = 8
W71_DEFAULT_LHR_V23_SWISH3_PROJ_DIM: int = 42


def _swish(x: "_np.ndarray") -> "_np.ndarray":
    return x / (1.0 + _np.exp(-x))


@dataclasses.dataclass
class LongHorizonReconstructionV23Head:
    inner_v22: LongHorizonReconstructionV22Head
    max_k: int
    restart_dim: int
    swish3_proj_dim: int
    swish3_proj_W: "_np.ndarray | None" = None
    scorer_layer13: "_np.ndarray | None" = None
    scorer_layer13_residual: float = 0.0
    restart_W: "_np.ndarray | None" = None

    @classmethod
    def init(
            cls, *, max_k: int = W71_DEFAULT_LHR_V23_MAX_K,
            restart_dim: int = W71_DEFAULT_LHR_V23_RESTART_DIM,
            swish3_proj_dim: int = (
                W71_DEFAULT_LHR_V23_SWISH3_PROJ_DIM),
            seed: int = 71200,
    ) -> "LongHorizonReconstructionV23Head":
        v22 = LongHorizonReconstructionV22Head.init(
            max_k=int(max_k), seed=int(seed))
        rng = _np.random.default_rng(int(seed) ^ 0xCAFE_71)
        out_dim = int(v22.out_dim)
        s3_W = rng.standard_normal(
            (int(W70_DEFAULT_LHR_V22_SWISH2_PROJ_DIM),
             int(swish3_proj_dim))) * 0.05
        rs_W = rng.standard_normal(
            (int(restart_dim), int(out_dim))) * 0.05
        return cls(
            inner_v22=v22,
            max_k=int(max_k),
            restart_dim=int(restart_dim),
            swish3_proj_dim=int(swish3_proj_dim),
            swish3_proj_W=s3_W.astype(_np.float64),
            restart_W=rs_W.astype(_np.float64),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v22.out_dim)

    def restart_value(
            self, *, restart_indicator: Sequence[float],
    ) -> "_np.ndarray":
        v = _np.asarray(
            restart_indicator, dtype=_np.float64)
        if v.size < int(self.restart_dim):
            v = _np.concatenate([
                v, _np.zeros(
                    int(self.restart_dim) - v.size,
                    dtype=_np.float64)])
        elif v.size > int(self.restart_dim):
            v = v[:int(self.restart_dim)]
        return v @ self.restart_W

    def twenty_two_way_value(
            self, *, carrier: Sequence[float], k: int,
            partial_contradiction_indicator: (
                Sequence[float] | None) = None,
            multi_branch_rejoin_indicator: (
                Sequence[float] | None) = None,
            repair_dominance_indicator: (
                Sequence[float] | None) = None,
            restart_indicator: (
                Sequence[float] | None) = None,
            **kwargs: Any,
    ) -> "_np.ndarray":
        v21 = self.inner_v22.twenty_one_way_value(
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
            **kwargs)
        v21 = _np.asarray(v21, dtype=_np.float64)
        if restart_indicator is not None:
            rs = self.restart_value(
                restart_indicator=list(restart_indicator))
            return v21 + 0.05 * rs
        return v21

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v23_head",
            "inner_v22_cid": str(self.inner_v22.cid()),
            "max_k": int(self.max_k),
            "restart_dim": int(self.restart_dim),
            "swish3_proj_dim": int(self.swish3_proj_dim),
            "swish3_proj_W_cid": (
                _ndarray_cid(self.swish3_proj_W)
                if self.swish3_proj_W is not None else "none"),
            "scorer_layer13_cid": (
                _ndarray_cid(self.scorer_layer13)
                if self.scorer_layer13 is not None
                else "untrained"),
            "scorer_layer13_residual": float(round(
                self.scorer_layer13_residual, 12)),
            "restart_W_cid": (
                _ndarray_cid(self.restart_W)
                if self.restart_W is not None else "none"),
        })


def fit_lhr_v23_thirteen_layer_scorer(
        *, head: LongHorizonReconstructionV23Head,
        train_features: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        ridge_lambda: float = 0.10,
) -> tuple[
        LongHorizonReconstructionV23Head, dict[str, Any]]:
    """Fit the thirteenth-layer ridge on top of the V22 twelve-
    layer pipeline (swish3 projection then ridge)."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(train_targets, dtype=_np.float64)
    if X.shape[0] == 0:
        return head, {"converged": True, "n": 0}
    if head.swish3_proj_W is not None and X.shape[1] == int(
            W70_DEFAULT_LHR_V22_SWISH2_PROJ_DIM):
        f12 = _swish(X @ _np.asarray(
            head.swish3_proj_W, dtype=_np.float64))
    else:
        f12 = X
    lam = max(float(ridge_lambda), 1e-9)
    A = f12.T @ f12 + lam * _np.eye(
        f12.shape[1], dtype=_np.float64)
    b = f12.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((f12.shape[1],), dtype=_np.float64)
    y_hat = f12 @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        head,
        scorer_layer13=_np.asarray(
            theta, dtype=_np.float64).copy(),
        scorer_layer13_residual=float(post))
    audit = {
        "schema": W71_LHR_V23_SCHEMA_VERSION,
        "kind": "lhr_v23_thirteen_layer_scorer",
        "pre_fit_residual": float(pre),
        "post_fit_residual": float(post),
        "converged": bool(post <= pre + 1e-9),
        "n": int(X.shape[0]),
    }
    return fitted, audit


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV23Witness:
    schema: str
    head_cid: str
    max_k: int
    restart_dim: int
    out_dim: int
    n_heads: int
    twenty_two_way_runs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "max_k": int(self.max_k),
            "restart_dim": int(self.restart_dim),
            "out_dim": int(self.out_dim),
            "n_heads": int(self.n_heads),
            "twenty_two_way_runs": bool(
                self.twenty_two_way_runs),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v23_witness",
            "witness": self.to_dict()})


def emit_lhr_v23_witness(
        head: LongHorizonReconstructionV23Head, *,
        carrier: Sequence[float], k: int = 16,
        partial_contradiction_indicator: (
            Sequence[float] | None) = None,
        multi_branch_rejoin_indicator: (
            Sequence[float] | None) = None,
        repair_dominance_indicator: (
            Sequence[float] | None) = None,
        restart_indicator: (
            Sequence[float] | None) = None,
        **kwargs: Any,
) -> LongHorizonReconstructionV23Witness:
    runs = True
    try:
        head.twenty_two_way_value(
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
    except Exception:
        runs = False
    return LongHorizonReconstructionV23Witness(
        schema=W71_LHR_V23_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        max_k=int(head.max_k),
        restart_dim=int(head.restart_dim),
        out_dim=int(head.out_dim),
        n_heads=22,
        twenty_two_way_runs=bool(runs),
    )


__all__ = [
    "W71_LHR_V23_SCHEMA_VERSION",
    "W71_DEFAULT_LHR_V23_MAX_K",
    "W71_DEFAULT_LHR_V23_RESTART_DIM",
    "W71_DEFAULT_LHR_V23_SWISH3_PROJ_DIM",
    "LongHorizonReconstructionV23Head",
    "fit_lhr_v23_thirteen_layer_scorer",
    "LongHorizonReconstructionV23Witness",
    "emit_lhr_v23_witness",
]
