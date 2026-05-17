"""W73 — Long-Horizon Retention V25.

Strictly extends W72's ``coordpy.long_horizon_retention_v24``. V24
had 23 heads + a fourteen-layer scorer at max_k=704. V25 adds:

* **24 heads** (V24's 23 + replacement-pressure-recovery head).
* **Fifteen-layer scorer** — V24's fourteen layers + a fifteenth
  random+swish layer before the final ridge.
* **max_k = 768** (vs V24's 704).

Honest scope (W73): only the final ridge head is fit; earlier
projections are frozen random. ``W73-L-V25-LHR-SCORER-FIT-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.long_horizon_retention_v25 requires numpy"
        ) from exc

from .long_horizon_retention_v24 import (
    LongHorizonReconstructionV24Head,
    W72_DEFAULT_LHR_V24_SWISH4_PROJ_DIM,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W73_LHR_V25_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v25.v1")
W73_DEFAULT_LHR_V25_MAX_K: int = 768
W73_DEFAULT_LHR_V25_REPLACEMENT_DIM: int = 8
W73_DEFAULT_LHR_V25_SWISH5_PROJ_DIM: int = 52


def _swish(x: "_np.ndarray") -> "_np.ndarray":
    return x / (1.0 + _np.exp(-x))


@dataclasses.dataclass
class LongHorizonReconstructionV25Head:
    inner_v24: LongHorizonReconstructionV24Head
    max_k: int
    replacement_dim: int
    swish5_proj_dim: int
    swish5_proj_W: "_np.ndarray | None" = None
    scorer_layer15: "_np.ndarray | None" = None
    scorer_layer15_residual: float = 0.0
    replacement_W: "_np.ndarray | None" = None

    @classmethod
    def init(
            cls, *, max_k: int = W73_DEFAULT_LHR_V25_MAX_K,
            replacement_dim: int = (
                W73_DEFAULT_LHR_V25_REPLACEMENT_DIM),
            swish5_proj_dim: int = (
                W73_DEFAULT_LHR_V25_SWISH5_PROJ_DIM),
            seed: int = 73200,
    ) -> "LongHorizonReconstructionV25Head":
        v24 = LongHorizonReconstructionV24Head.init(
            max_k=int(max_k), seed=int(seed))
        rng = _np.random.default_rng(int(seed) ^ 0xCAFE_73)
        out_dim = int(v24.out_dim)
        s5_W = rng.standard_normal(
            (int(W72_DEFAULT_LHR_V24_SWISH4_PROJ_DIM),
             int(swish5_proj_dim))) * 0.05
        rep_W = rng.standard_normal(
            (int(replacement_dim), int(out_dim))) * 0.05
        return cls(
            inner_v24=v24,
            max_k=int(max_k),
            replacement_dim=int(replacement_dim),
            swish5_proj_dim=int(swish5_proj_dim),
            swish5_proj_W=s5_W.astype(_np.float64),
            replacement_W=rep_W.astype(_np.float64),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v24.out_dim)

    def replacement_value(
            self, *, replacement_indicator: Sequence[float],
    ) -> "_np.ndarray":
        v = _np.asarray(
            replacement_indicator, dtype=_np.float64)
        if v.size < int(self.replacement_dim):
            v = _np.concatenate([
                v, _np.zeros(
                    int(self.replacement_dim) - v.size,
                    dtype=_np.float64)])
        elif v.size > int(self.replacement_dim):
            v = v[:int(self.replacement_dim)]
        return v @ self.replacement_W

    def twenty_four_way_value(
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
            replacement_indicator: (
                Sequence[float] | None) = None,
            **kwargs: Any,
    ) -> "_np.ndarray":
        v23 = self.inner_v24.twenty_three_way_value(
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
        v23 = _np.asarray(v23, dtype=_np.float64)
        if replacement_indicator is not None:
            rep = self.replacement_value(
                replacement_indicator=list(replacement_indicator))
            return v23 + 0.05 * rep
        return v23

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v25_head",
            "inner_v24_cid": str(self.inner_v24.cid()),
            "max_k": int(self.max_k),
            "replacement_dim": int(self.replacement_dim),
            "swish5_proj_dim": int(self.swish5_proj_dim),
            "swish5_proj_W_cid": (
                _ndarray_cid(self.swish5_proj_W)
                if self.swish5_proj_W is not None else "none"),
            "scorer_layer15_cid": (
                _ndarray_cid(self.scorer_layer15)
                if self.scorer_layer15 is not None
                else "untrained"),
            "scorer_layer15_residual": float(round(
                self.scorer_layer15_residual, 12)),
            "replacement_W_cid": (
                _ndarray_cid(self.replacement_W)
                if self.replacement_W is not None else "none"),
        })


def fit_lhr_v25_fifteen_layer_scorer(
        *, head: LongHorizonReconstructionV25Head,
        train_features: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        ridge_lambda: float = 0.10,
) -> tuple[
        LongHorizonReconstructionV25Head, dict[str, Any]]:
    """Fit the fifteenth-layer ridge on top of the V24 fourteen-
    layer pipeline (swish5 projection then ridge)."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(train_targets, dtype=_np.float64)
    if X.shape[0] == 0:
        return head, {"converged": True, "n": 0}
    if head.swish5_proj_W is not None and X.shape[1] == int(
            W72_DEFAULT_LHR_V24_SWISH4_PROJ_DIM):
        f14 = _swish(X @ _np.asarray(
            head.swish5_proj_W, dtype=_np.float64))
    else:
        f14 = X
    lam = max(float(ridge_lambda), 1e-9)
    A = f14.T @ f14 + lam * _np.eye(
        f14.shape[1], dtype=_np.float64)
    b = f14.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((f14.shape[1],), dtype=_np.float64)
    y_hat = f14 @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        head,
        scorer_layer15=_np.asarray(
            theta, dtype=_np.float64).copy(),
        scorer_layer15_residual=float(post))
    audit = {
        "schema": W73_LHR_V25_SCHEMA_VERSION,
        "kind": "lhr_v25_fifteen_layer_scorer",
        "pre_fit_residual": float(pre),
        "post_fit_residual": float(post),
        "converged": bool(post <= pre + 1e-9),
        "n": int(X.shape[0]),
    }
    return fitted, audit


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV25Witness:
    schema: str
    head_cid: str
    max_k: int
    replacement_dim: int
    out_dim: int
    n_heads: int
    twenty_four_way_runs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "max_k": int(self.max_k),
            "replacement_dim": int(self.replacement_dim),
            "out_dim": int(self.out_dim),
            "n_heads": int(self.n_heads),
            "twenty_four_way_runs": bool(
                self.twenty_four_way_runs),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v25_witness",
            "witness": self.to_dict()})


def emit_lhr_v25_witness(
        head: LongHorizonReconstructionV25Head, *,
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
        replacement_indicator: (
            Sequence[float] | None) = None,
        **kwargs: Any,
) -> LongHorizonReconstructionV25Witness:
    runs = True
    try:
        head.twenty_four_way_value(
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
            replacement_indicator=(
                list(replacement_indicator)
                if replacement_indicator is not None
                else None),
            **kwargs)
    except Exception:
        runs = False
    return LongHorizonReconstructionV25Witness(
        schema=W73_LHR_V25_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        max_k=int(head.max_k),
        replacement_dim=int(head.replacement_dim),
        out_dim=int(head.out_dim),
        n_heads=24,
        twenty_four_way_runs=bool(runs),
    )


__all__ = [
    "W73_LHR_V25_SCHEMA_VERSION",
    "W73_DEFAULT_LHR_V25_MAX_K",
    "W73_DEFAULT_LHR_V25_REPLACEMENT_DIM",
    "W73_DEFAULT_LHR_V25_SWISH5_PROJ_DIM",
    "LongHorizonReconstructionV25Head",
    "fit_lhr_v25_fifteen_layer_scorer",
    "LongHorizonReconstructionV25Witness",
    "emit_lhr_v25_witness",
]
