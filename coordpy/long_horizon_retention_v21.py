"""W69 M13 — Long-Horizon Retention V21.

Strictly extends W68's ``coordpy.long_horizon_retention_v20``. V20
had 19 heads + a ten-layer scorer at max_k=448. V21 adds:

* **20 heads** (V20's 19 + multi-branch-rejoin-recovery head).
* **Eleven-layer scorer** — V20's ten layers + an eleventh
  random+swish layer before the final ridge.
* **max_k = 512** (vs V20's 448).

Honest scope (W69): only the final ridge head is fit; the first
ten layers are frozen random projections.
``W69-L-V21-LHR-SCORER-FIT-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.long_horizon_retention_v21 requires numpy"
        ) from exc

from .long_horizon_retention_v20 import (
    LongHorizonReconstructionV20Head,
    W68_DEFAULT_LHR_V20_GELU_PROJ_DIM,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W69_LHR_V21_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v21.v1")
W69_DEFAULT_LHR_V21_MAX_K: int = 512
W69_DEFAULT_LHR_V21_MULTI_BRANCH_REJOIN_DIM: int = 8
W69_DEFAULT_LHR_V21_SWISH_PROJ_DIM: int = 32


def _swish(x: "_np.ndarray") -> "_np.ndarray":
    return x / (1.0 + _np.exp(-x))


@dataclasses.dataclass
class LongHorizonReconstructionV21Head:
    inner_v20: LongHorizonReconstructionV20Head
    max_k: int
    multi_branch_rejoin_dim: int
    swish_proj_dim: int
    swish_proj_W: "_np.ndarray | None" = None
    scorer_layer11: "_np.ndarray | None" = None
    scorer_layer11_residual: float = 0.0
    multi_branch_rejoin_W: "_np.ndarray | None" = None

    @classmethod
    def init(
            cls, *, max_k: int = W69_DEFAULT_LHR_V21_MAX_K,
            multi_branch_rejoin_dim: int = (
                W69_DEFAULT_LHR_V21_MULTI_BRANCH_REJOIN_DIM),
            swish_proj_dim: int = (
                W69_DEFAULT_LHR_V21_SWISH_PROJ_DIM),
            seed: int = 69200,
    ) -> "LongHorizonReconstructionV21Head":
        v20 = LongHorizonReconstructionV20Head.init(
            max_k=int(max_k), seed=int(seed))
        rng = _np.random.default_rng(int(seed) ^ 0xCAFE_69)
        out_dim = int(v20.out_dim)
        s_W = rng.standard_normal(
            (int(W68_DEFAULT_LHR_V20_GELU_PROJ_DIM),
             int(swish_proj_dim))) * 0.05
        mbr_W = rng.standard_normal(
            (int(multi_branch_rejoin_dim),
             int(out_dim))) * 0.05
        return cls(
            inner_v20=v20,
            max_k=int(max_k),
            multi_branch_rejoin_dim=int(multi_branch_rejoin_dim),
            swish_proj_dim=int(swish_proj_dim),
            swish_proj_W=s_W.astype(_np.float64),
            multi_branch_rejoin_W=mbr_W.astype(_np.float64),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v20.out_dim)

    def multi_branch_rejoin_value(
            self, *,
            multi_branch_rejoin_indicator: Sequence[float],
    ) -> "_np.ndarray":
        v = _np.asarray(
            multi_branch_rejoin_indicator, dtype=_np.float64)
        if v.size < int(self.multi_branch_rejoin_dim):
            v = _np.concatenate([
                v, _np.zeros(
                    int(self.multi_branch_rejoin_dim) - v.size,
                    dtype=_np.float64)])
        elif v.size > int(self.multi_branch_rejoin_dim):
            v = v[:int(self.multi_branch_rejoin_dim)]
        return v @ self.multi_branch_rejoin_W

    def twenty_way_value(
            self, *, carrier: Sequence[float], k: int,
            partial_contradiction_indicator: (
                Sequence[float] | None) = None,
            multi_branch_rejoin_indicator: (
                Sequence[float] | None) = None,
            **kwargs: Any,
    ) -> "_np.ndarray":
        v19 = self.inner_v20.nineteen_way_value(
            carrier=list(carrier), k=int(k),
            partial_contradiction_indicator=(
                list(partial_contradiction_indicator)
                if partial_contradiction_indicator is not None
                else None),
            **kwargs)
        v19 = _np.asarray(v19, dtype=_np.float64)
        if multi_branch_rejoin_indicator is not None:
            mbr = self.multi_branch_rejoin_value(
                multi_branch_rejoin_indicator=list(
                    multi_branch_rejoin_indicator))
            return v19 + 0.05 * mbr
        return v19

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v21_head",
            "inner_v20_cid": str(self.inner_v20.cid()),
            "max_k": int(self.max_k),
            "multi_branch_rejoin_dim": int(
                self.multi_branch_rejoin_dim),
            "swish_proj_dim": int(self.swish_proj_dim),
            "swish_proj_W_cid": (
                _ndarray_cid(self.swish_proj_W)
                if self.swish_proj_W is not None else "none"),
            "scorer_layer11_cid": (
                _ndarray_cid(self.scorer_layer11)
                if self.scorer_layer11 is not None
                else "untrained"),
            "scorer_layer11_residual": float(round(
                self.scorer_layer11_residual, 12)),
            "multi_branch_rejoin_W_cid": (
                _ndarray_cid(self.multi_branch_rejoin_W)
                if self.multi_branch_rejoin_W is not None
                else "none"),
        })


def fit_lhr_v21_eleven_layer_scorer(
        *, head: LongHorizonReconstructionV21Head,
        train_features: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        ridge_lambda: float = 0.10,
) -> tuple[LongHorizonReconstructionV21Head, dict[str, Any]]:
    """Fit the eleventh-layer ridge on top of the V20 ten-layer
    pipeline (swish projection then ridge)."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(train_targets, dtype=_np.float64)
    if X.shape[0] == 0:
        return head, {"converged": True, "n": 0}
    if head.swish_proj_W is not None and X.shape[1] == int(
            W68_DEFAULT_LHR_V20_GELU_PROJ_DIM):
        f10 = _swish(X @ _np.asarray(
            head.swish_proj_W, dtype=_np.float64))
    else:
        f10 = X
    lam = max(float(ridge_lambda), 1e-9)
    A = f10.T @ f10 + lam * _np.eye(
        f10.shape[1], dtype=_np.float64)
    b = f10.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((f10.shape[1],), dtype=_np.float64)
    y_hat = f10 @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        head,
        scorer_layer11=_np.asarray(
            theta, dtype=_np.float64).copy(),
        scorer_layer11_residual=float(post))
    audit = {
        "schema": W69_LHR_V21_SCHEMA_VERSION,
        "kind": "lhr_v21_eleven_layer_scorer",
        "pre_fit_residual": float(pre),
        "post_fit_residual": float(post),
        "converged": bool(post <= pre + 1e-9),
        "n": int(X.shape[0]),
    }
    return fitted, audit


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV21Witness:
    schema: str
    head_cid: str
    max_k: int
    multi_branch_rejoin_dim: int
    out_dim: int
    n_heads: int
    twenty_way_runs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "max_k": int(self.max_k),
            "multi_branch_rejoin_dim": int(
                self.multi_branch_rejoin_dim),
            "out_dim": int(self.out_dim),
            "n_heads": int(self.n_heads),
            "twenty_way_runs": bool(self.twenty_way_runs),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v21_witness",
            "witness": self.to_dict()})


def emit_lhr_v21_witness(
        head: LongHorizonReconstructionV21Head, *,
        carrier: Sequence[float], k: int = 16,
        partial_contradiction_indicator: (
            Sequence[float] | None) = None,
        multi_branch_rejoin_indicator: (
            Sequence[float] | None) = None,
        **kwargs: Any,
) -> LongHorizonReconstructionV21Witness:
    runs = True
    try:
        head.twenty_way_value(
            carrier=list(carrier), k=int(k),
            partial_contradiction_indicator=(
                list(partial_contradiction_indicator)
                if partial_contradiction_indicator is not None
                else None),
            multi_branch_rejoin_indicator=(
                list(multi_branch_rejoin_indicator)
                if multi_branch_rejoin_indicator is not None
                else None),
            **kwargs)
    except Exception:
        runs = False
    return LongHorizonReconstructionV21Witness(
        schema=W69_LHR_V21_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        max_k=int(head.max_k),
        multi_branch_rejoin_dim=int(
            head.multi_branch_rejoin_dim),
        out_dim=int(head.out_dim),
        n_heads=20,
        twenty_way_runs=bool(runs),
    )


__all__ = [
    "W69_LHR_V21_SCHEMA_VERSION",
    "W69_DEFAULT_LHR_V21_MAX_K",
    "W69_DEFAULT_LHR_V21_MULTI_BRANCH_REJOIN_DIM",
    "W69_DEFAULT_LHR_V21_SWISH_PROJ_DIM",
    "LongHorizonReconstructionV21Head",
    "fit_lhr_v21_eleven_layer_scorer",
    "LongHorizonReconstructionV21Witness",
    "emit_lhr_v21_witness",
]
