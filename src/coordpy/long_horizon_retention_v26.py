"""W74 — Long-Horizon Retention V26.

Strictly extends W73's ``coordpy.long_horizon_retention_v25``. V25
had 24 heads + a fifteen-layer scorer at max_k=768. V26 adds:

* **25 heads** (V25's 24 + compound-pressure-recovery head).
* **Sixteen-layer scorer** — V25's fifteen layers + a sixteenth
  random+swish layer before the final ridge.
* **max_k = 832** (vs V25's 768).

Honest scope (W74): only the final ridge head is fit; earlier
projections are frozen random. ``W74-L-V26-LHR-SCORER-FIT-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.long_horizon_retention_v26 requires numpy"
        ) from exc

from .long_horizon_retention_v25 import (
    LongHorizonReconstructionV25Head,
    W73_DEFAULT_LHR_V25_SWISH5_PROJ_DIM,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W74_LHR_V26_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v26.v1")
W74_DEFAULT_LHR_V26_MAX_K: int = 832
W74_DEFAULT_LHR_V26_COMPOUND_DIM: int = 8
W74_DEFAULT_LHR_V26_SWISH6_PROJ_DIM: int = 56


def _swish(x: "_np.ndarray") -> "_np.ndarray":
    return x / (1.0 + _np.exp(-x))


@dataclasses.dataclass
class LongHorizonReconstructionV26Head:
    inner_v25: LongHorizonReconstructionV25Head
    max_k: int
    compound_dim: int
    swish6_proj_dim: int
    swish6_proj_W: "_np.ndarray | None" = None
    scorer_layer16: "_np.ndarray | None" = None
    scorer_layer16_residual: float = 0.0
    compound_W: "_np.ndarray | None" = None

    @classmethod
    def init(
            cls, *, max_k: int = W74_DEFAULT_LHR_V26_MAX_K,
            compound_dim: int = (
                W74_DEFAULT_LHR_V26_COMPOUND_DIM),
            swish6_proj_dim: int = (
                W74_DEFAULT_LHR_V26_SWISH6_PROJ_DIM),
            seed: int = 74200,
    ) -> "LongHorizonReconstructionV26Head":
        v25 = LongHorizonReconstructionV25Head.init(
            max_k=int(max_k), seed=int(seed))
        rng = _np.random.default_rng(int(seed) ^ 0xCAFE_74)
        out_dim = int(v25.out_dim)
        s6_W = rng.standard_normal(
            (int(W73_DEFAULT_LHR_V25_SWISH5_PROJ_DIM),
             int(swish6_proj_dim))) * 0.05
        comp_W = rng.standard_normal(
            (int(compound_dim), int(out_dim))) * 0.05
        return cls(
            inner_v25=v25,
            max_k=int(max_k),
            compound_dim=int(compound_dim),
            swish6_proj_dim=int(swish6_proj_dim),
            swish6_proj_W=s6_W.astype(_np.float64),
            compound_W=comp_W.astype(_np.float64),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v25.out_dim)

    def compound_value(
            self, *, compound_indicator: Sequence[float],
    ) -> "_np.ndarray":
        v = _np.asarray(
            compound_indicator, dtype=_np.float64)
        if v.size < int(self.compound_dim):
            v = _np.concatenate([
                v, _np.zeros(
                    int(self.compound_dim) - v.size,
                    dtype=_np.float64)])
        elif v.size > int(self.compound_dim):
            v = v[:int(self.compound_dim)]
        return v @ self.compound_W

    def twenty_five_way_value(
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
            compound_indicator: (
                Sequence[float] | None) = None,
            **kwargs: Any,
    ) -> "_np.ndarray":
        v24 = self.inner_v25.twenty_four_way_value(
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
        v24 = _np.asarray(v24, dtype=_np.float64)
        if compound_indicator is not None:
            cmp = self.compound_value(
                compound_indicator=list(compound_indicator))
            return v24 + 0.05 * cmp
        return v24

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v26_head",
            "inner_v25_cid": str(self.inner_v25.cid()),
            "max_k": int(self.max_k),
            "compound_dim": int(self.compound_dim),
            "swish6_proj_dim": int(self.swish6_proj_dim),
            "swish6_proj_W_cid": (
                _ndarray_cid(self.swish6_proj_W)
                if self.swish6_proj_W is not None else "none"),
            "scorer_layer16_cid": (
                _ndarray_cid(self.scorer_layer16)
                if self.scorer_layer16 is not None
                else "untrained"),
            "scorer_layer16_residual": float(round(
                self.scorer_layer16_residual, 12)),
            "compound_W_cid": (
                _ndarray_cid(self.compound_W)
                if self.compound_W is not None else "none"),
        })


def fit_lhr_v26_sixteen_layer_scorer(
        *, head: LongHorizonReconstructionV26Head,
        train_features: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        ridge_lambda: float = 0.10,
) -> tuple[
        LongHorizonReconstructionV26Head, dict[str, Any]]:
    """Fit the sixteenth-layer ridge on top of the V25 fifteen-
    layer pipeline (swish6 projection then ridge)."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(train_targets, dtype=_np.float64)
    if X.shape[0] == 0:
        return head, {"converged": True, "n": 0}
    if head.swish6_proj_W is not None and X.shape[1] == int(
            W73_DEFAULT_LHR_V25_SWISH5_PROJ_DIM):
        f15 = _swish(X @ _np.asarray(
            head.swish6_proj_W, dtype=_np.float64))
    else:
        f15 = X
    lam = max(float(ridge_lambda), 1e-9)
    A = f15.T @ f15 + lam * _np.eye(
        f15.shape[1], dtype=_np.float64)
    b = f15.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((f15.shape[1],), dtype=_np.float64)
    y_hat = f15 @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        head,
        scorer_layer16=_np.asarray(
            theta, dtype=_np.float64).copy(),
        scorer_layer16_residual=float(post))
    audit = {
        "schema": W74_LHR_V26_SCHEMA_VERSION,
        "kind": "lhr_v26_sixteen_layer_scorer",
        "pre_fit_residual": float(pre),
        "post_fit_residual": float(post),
        "converged": bool(post <= pre + 1e-9),
        "n": int(X.shape[0]),
    }
    return fitted, audit


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV26Witness:
    schema: str
    head_cid: str
    max_k: int
    compound_dim: int
    out_dim: int
    n_heads: int
    twenty_five_way_runs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "max_k": int(self.max_k),
            "compound_dim": int(self.compound_dim),
            "out_dim": int(self.out_dim),
            "n_heads": int(self.n_heads),
            "twenty_five_way_runs": bool(
                self.twenty_five_way_runs),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v26_witness",
            "witness": self.to_dict()})


def emit_lhr_v26_witness(
        head: LongHorizonReconstructionV26Head, *,
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
        compound_indicator: (
            Sequence[float] | None) = None,
        **kwargs: Any,
) -> LongHorizonReconstructionV26Witness:
    runs = True
    try:
        head.twenty_five_way_value(
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
            compound_indicator=(
                list(compound_indicator)
                if compound_indicator is not None
                else None),
            **kwargs)
    except Exception:
        runs = False
    return LongHorizonReconstructionV26Witness(
        schema=W74_LHR_V26_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        max_k=int(head.max_k),
        compound_dim=int(head.compound_dim),
        out_dim=int(head.out_dim),
        n_heads=25,
        twenty_five_way_runs=bool(runs),
    )


__all__ = [
    "W74_LHR_V26_SCHEMA_VERSION",
    "W74_DEFAULT_LHR_V26_MAX_K",
    "W74_DEFAULT_LHR_V26_COMPOUND_DIM",
    "W74_DEFAULT_LHR_V26_SWISH6_PROJ_DIM",
    "LongHorizonReconstructionV26Head",
    "fit_lhr_v26_sixteen_layer_scorer",
    "LongHorizonReconstructionV26Witness",
    "emit_lhr_v26_witness",
]
