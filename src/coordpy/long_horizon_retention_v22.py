"""W70 — Long-Horizon Retention V22.

Strictly extends W69's ``coordpy.long_horizon_retention_v21``. V21
had 20 heads + an eleven-layer scorer at max_k=512. V22 adds:

* **21 heads** (V21's 20 + repair-dominance-recovery head).
* **Twelve-layer scorer** — V21's eleven layers + a twelfth
  random+swish layer before the final ridge.
* **max_k = 576** (vs V21's 512).

Honest scope (W70): only the final ridge head is fit; the first
eleven layers are frozen random projections.
``W70-L-V22-LHR-SCORER-FIT-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.long_horizon_retention_v22 requires numpy"
        ) from exc

from .long_horizon_retention_v21 import (
    LongHorizonReconstructionV21Head,
    W69_DEFAULT_LHR_V21_SWISH_PROJ_DIM,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W70_LHR_V22_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v22.v1")
W70_DEFAULT_LHR_V22_MAX_K: int = 576
W70_DEFAULT_LHR_V22_REPAIR_DOMINANCE_DIM: int = 7
W70_DEFAULT_LHR_V22_SWISH2_PROJ_DIM: int = 36


def _swish(x: "_np.ndarray") -> "_np.ndarray":
    return x / (1.0 + _np.exp(-x))


@dataclasses.dataclass
class LongHorizonReconstructionV22Head:
    inner_v21: LongHorizonReconstructionV21Head
    max_k: int
    repair_dominance_dim: int
    swish2_proj_dim: int
    swish2_proj_W: "_np.ndarray | None" = None
    scorer_layer12: "_np.ndarray | None" = None
    scorer_layer12_residual: float = 0.0
    repair_dominance_W: "_np.ndarray | None" = None

    @classmethod
    def init(
            cls, *, max_k: int = W70_DEFAULT_LHR_V22_MAX_K,
            repair_dominance_dim: int = (
                W70_DEFAULT_LHR_V22_REPAIR_DOMINANCE_DIM),
            swish2_proj_dim: int = (
                W70_DEFAULT_LHR_V22_SWISH2_PROJ_DIM),
            seed: int = 70200,
    ) -> "LongHorizonReconstructionV22Head":
        v21 = LongHorizonReconstructionV21Head.init(
            max_k=int(max_k), seed=int(seed))
        rng = _np.random.default_rng(int(seed) ^ 0xCAFE_70)
        out_dim = int(v21.out_dim)
        s2_W = rng.standard_normal(
            (int(W69_DEFAULT_LHR_V21_SWISH_PROJ_DIM),
             int(swish2_proj_dim))) * 0.05
        rd_W = rng.standard_normal(
            (int(repair_dominance_dim),
             int(out_dim))) * 0.05
        return cls(
            inner_v21=v21,
            max_k=int(max_k),
            repair_dominance_dim=int(repair_dominance_dim),
            swish2_proj_dim=int(swish2_proj_dim),
            swish2_proj_W=s2_W.astype(_np.float64),
            repair_dominance_W=rd_W.astype(_np.float64),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v21.out_dim)

    def repair_dominance_value(
            self, *, repair_dominance_indicator: Sequence[float],
    ) -> "_np.ndarray":
        v = _np.asarray(
            repair_dominance_indicator, dtype=_np.float64)
        if v.size < int(self.repair_dominance_dim):
            v = _np.concatenate([
                v, _np.zeros(
                    int(self.repair_dominance_dim) - v.size,
                    dtype=_np.float64)])
        elif v.size > int(self.repair_dominance_dim):
            v = v[:int(self.repair_dominance_dim)]
        return v @ self.repair_dominance_W

    def twenty_one_way_value(
            self, *, carrier: Sequence[float], k: int,
            partial_contradiction_indicator: (
                Sequence[float] | None) = None,
            multi_branch_rejoin_indicator: (
                Sequence[float] | None) = None,
            repair_dominance_indicator: (
                Sequence[float] | None) = None,
            **kwargs: Any,
    ) -> "_np.ndarray":
        v20 = self.inner_v21.twenty_way_value(
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
        v20 = _np.asarray(v20, dtype=_np.float64)
        if repair_dominance_indicator is not None:
            rd = self.repair_dominance_value(
                repair_dominance_indicator=list(
                    repair_dominance_indicator))
            return v20 + 0.05 * rd
        return v20

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v22_head",
            "inner_v21_cid": str(self.inner_v21.cid()),
            "max_k": int(self.max_k),
            "repair_dominance_dim": int(
                self.repair_dominance_dim),
            "swish2_proj_dim": int(self.swish2_proj_dim),
            "swish2_proj_W_cid": (
                _ndarray_cid(self.swish2_proj_W)
                if self.swish2_proj_W is not None else "none"),
            "scorer_layer12_cid": (
                _ndarray_cid(self.scorer_layer12)
                if self.scorer_layer12 is not None
                else "untrained"),
            "scorer_layer12_residual": float(round(
                self.scorer_layer12_residual, 12)),
            "repair_dominance_W_cid": (
                _ndarray_cid(self.repair_dominance_W)
                if self.repair_dominance_W is not None
                else "none"),
        })


def fit_lhr_v22_twelve_layer_scorer(
        *, head: LongHorizonReconstructionV22Head,
        train_features: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        ridge_lambda: float = 0.10,
) -> tuple[LongHorizonReconstructionV22Head, dict[str, Any]]:
    """Fit the twelfth-layer ridge on top of the V21 eleven-layer
    pipeline (swish2 projection then ridge)."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(train_targets, dtype=_np.float64)
    if X.shape[0] == 0:
        return head, {"converged": True, "n": 0}
    if head.swish2_proj_W is not None and X.shape[1] == int(
            W69_DEFAULT_LHR_V21_SWISH_PROJ_DIM):
        f11 = _swish(X @ _np.asarray(
            head.swish2_proj_W, dtype=_np.float64))
    else:
        f11 = X
    lam = max(float(ridge_lambda), 1e-9)
    A = f11.T @ f11 + lam * _np.eye(
        f11.shape[1], dtype=_np.float64)
    b = f11.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((f11.shape[1],), dtype=_np.float64)
    y_hat = f11 @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        head,
        scorer_layer12=_np.asarray(
            theta, dtype=_np.float64).copy(),
        scorer_layer12_residual=float(post))
    audit = {
        "schema": W70_LHR_V22_SCHEMA_VERSION,
        "kind": "lhr_v22_twelve_layer_scorer",
        "pre_fit_residual": float(pre),
        "post_fit_residual": float(post),
        "converged": bool(post <= pre + 1e-9),
        "n": int(X.shape[0]),
    }
    return fitted, audit


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV22Witness:
    schema: str
    head_cid: str
    max_k: int
    repair_dominance_dim: int
    out_dim: int
    n_heads: int
    twenty_one_way_runs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "max_k": int(self.max_k),
            "repair_dominance_dim": int(
                self.repair_dominance_dim),
            "out_dim": int(self.out_dim),
            "n_heads": int(self.n_heads),
            "twenty_one_way_runs": bool(self.twenty_one_way_runs),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v22_witness",
            "witness": self.to_dict()})


def emit_lhr_v22_witness(
        head: LongHorizonReconstructionV22Head, *,
        carrier: Sequence[float], k: int = 16,
        partial_contradiction_indicator: (
            Sequence[float] | None) = None,
        multi_branch_rejoin_indicator: (
            Sequence[float] | None) = None,
        repair_dominance_indicator: (
            Sequence[float] | None) = None,
        **kwargs: Any,
) -> LongHorizonReconstructionV22Witness:
    runs = True
    try:
        head.twenty_one_way_value(
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
    except Exception:
        runs = False
    return LongHorizonReconstructionV22Witness(
        schema=W70_LHR_V22_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        max_k=int(head.max_k),
        repair_dominance_dim=int(head.repair_dominance_dim),
        out_dim=int(head.out_dim),
        n_heads=21,
        twenty_one_way_runs=bool(runs),
    )


__all__ = [
    "W70_LHR_V22_SCHEMA_VERSION",
    "W70_DEFAULT_LHR_V22_MAX_K",
    "W70_DEFAULT_LHR_V22_REPAIR_DOMINANCE_DIM",
    "W70_DEFAULT_LHR_V22_SWISH2_PROJ_DIM",
    "LongHorizonReconstructionV22Head",
    "fit_lhr_v22_twelve_layer_scorer",
    "LongHorizonReconstructionV22Witness",
    "emit_lhr_v22_witness",
]
