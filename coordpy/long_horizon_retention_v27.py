"""W75 — Long-Horizon Retention V27.

Strictly extends W74's ``coordpy.long_horizon_retention_v26``. V26
had 25 heads + a sixteen-layer scorer at max_k=832. V27 adds:

* **26 heads** (V26's 25 + compound-chain-pressure-recovery head).
* **Seventeen-layer scorer** — V26's sixteen layers + a seventeenth
  random+swish layer before the final ridge.
* **max_k = 896** (vs V26's 832).

Honest scope (W75): only the final ridge head is fit; earlier
projections are frozen random. ``W75-L-V27-LHR-SCORER-FIT-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.long_horizon_retention_v27 requires numpy"
        ) from exc

from .long_horizon_retention_v26 import (
    LongHorizonReconstructionV26Head,
    W74_DEFAULT_LHR_V26_SWISH6_PROJ_DIM,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W75_LHR_V27_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v27.v1")
W75_DEFAULT_LHR_V27_MAX_K: int = 896
W75_DEFAULT_LHR_V27_COMPOUND_CHAIN_DIM: int = 8
W75_DEFAULT_LHR_V27_SWISH7_PROJ_DIM: int = 60


def _swish(x: "_np.ndarray") -> "_np.ndarray":
    return x / (1.0 + _np.exp(-x))


@dataclasses.dataclass
class LongHorizonReconstructionV27Head:
    inner_v26: LongHorizonReconstructionV26Head
    max_k: int
    compound_chain_dim: int
    swish7_proj_dim: int
    swish7_proj_W: "_np.ndarray | None" = None
    scorer_layer17: "_np.ndarray | None" = None
    scorer_layer17_residual: float = 0.0
    compound_chain_W: "_np.ndarray | None" = None

    @classmethod
    def init(
            cls, *, max_k: int = W75_DEFAULT_LHR_V27_MAX_K,
            compound_chain_dim: int = (
                W75_DEFAULT_LHR_V27_COMPOUND_CHAIN_DIM),
            swish7_proj_dim: int = (
                W75_DEFAULT_LHR_V27_SWISH7_PROJ_DIM),
            seed: int = 75200,
    ) -> "LongHorizonReconstructionV27Head":
        v26 = LongHorizonReconstructionV26Head.init(
            max_k=int(max_k), seed=int(seed))
        rng = _np.random.default_rng(int(seed) ^ 0xCAFE_75)
        out_dim = int(v26.out_dim)
        s7_W = rng.standard_normal(
            (int(W74_DEFAULT_LHR_V26_SWISH6_PROJ_DIM),
             int(swish7_proj_dim))) * 0.05
        chain_W = rng.standard_normal(
            (int(compound_chain_dim), int(out_dim))) * 0.05
        return cls(
            inner_v26=v26,
            max_k=int(max_k),
            compound_chain_dim=int(compound_chain_dim),
            swish7_proj_dim=int(swish7_proj_dim),
            swish7_proj_W=s7_W.astype(_np.float64),
            compound_chain_W=chain_W.astype(_np.float64),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v26.out_dim)

    def compound_chain_value(
            self, *, compound_chain_indicator: Sequence[float],
    ) -> "_np.ndarray":
        v = _np.asarray(
            compound_chain_indicator, dtype=_np.float64)
        if v.size < int(self.compound_chain_dim):
            v = _np.concatenate([
                v, _np.zeros(
                    int(self.compound_chain_dim) - v.size,
                    dtype=_np.float64)])
        elif v.size > int(self.compound_chain_dim):
            v = v[:int(self.compound_chain_dim)]
        return v @ self.compound_chain_W

    def twenty_six_way_value(
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
            compound_chain_indicator: (
                Sequence[float] | None) = None,
            **kwargs: Any,
    ) -> "_np.ndarray":
        v25 = self.inner_v26.twenty_five_way_value(
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
        v25 = _np.asarray(v25, dtype=_np.float64)
        if compound_chain_indicator is not None:
            chain = self.compound_chain_value(
                compound_chain_indicator=list(
                    compound_chain_indicator))
            return v25 + 0.05 * chain
        return v25

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v27_head",
            "inner_v26_cid": str(self.inner_v26.cid()),
            "max_k": int(self.max_k),
            "compound_chain_dim": int(self.compound_chain_dim),
            "swish7_proj_dim": int(self.swish7_proj_dim),
            "swish7_proj_W_cid": (
                _ndarray_cid(self.swish7_proj_W)
                if self.swish7_proj_W is not None else "none"),
            "scorer_layer17_cid": (
                _ndarray_cid(self.scorer_layer17)
                if self.scorer_layer17 is not None
                else "untrained"),
            "scorer_layer17_residual": float(round(
                self.scorer_layer17_residual, 12)),
            "compound_chain_W_cid": (
                _ndarray_cid(self.compound_chain_W)
                if self.compound_chain_W is not None
                else "none"),
        })


def fit_lhr_v27_seventeen_layer_scorer(
        *, head: LongHorizonReconstructionV27Head,
        train_features: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        ridge_lambda: float = 0.10,
) -> tuple[
        LongHorizonReconstructionV27Head, dict[str, Any]]:
    """Fit the seventeenth-layer ridge on top of the V26 sixteen-
    layer pipeline (swish7 projection then ridge)."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(train_targets, dtype=_np.float64)
    if X.shape[0] == 0:
        return head, {"converged": True, "n": 0}
    if head.swish7_proj_W is not None and X.shape[1] == int(
            W74_DEFAULT_LHR_V26_SWISH6_PROJ_DIM):
        f16 = _swish(X @ _np.asarray(
            head.swish7_proj_W, dtype=_np.float64))
    else:
        f16 = X
    lam = max(float(ridge_lambda), 1e-9)
    A = f16.T @ f16 + lam * _np.eye(
        f16.shape[1], dtype=_np.float64)
    b = f16.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((f16.shape[1],), dtype=_np.float64)
    y_hat = f16 @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        head,
        scorer_layer17=_np.asarray(
            theta, dtype=_np.float64).copy(),
        scorer_layer17_residual=float(post))
    audit = {
        "schema": W75_LHR_V27_SCHEMA_VERSION,
        "kind": "lhr_v27_seventeen_layer_scorer",
        "pre_fit_residual": float(pre),
        "post_fit_residual": float(post),
        "converged": bool(post <= pre + 1e-9),
        "n": int(X.shape[0]),
    }
    return fitted, audit


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV27Witness:
    schema: str
    head_cid: str
    max_k: int
    compound_chain_dim: int
    out_dim: int
    n_heads: int
    twenty_six_way_runs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "max_k": int(self.max_k),
            "compound_chain_dim": int(self.compound_chain_dim),
            "out_dim": int(self.out_dim),
            "n_heads": int(self.n_heads),
            "twenty_six_way_runs": bool(
                self.twenty_six_way_runs),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v27_witness",
            "witness": self.to_dict()})


def emit_lhr_v27_witness(
        head: LongHorizonReconstructionV27Head, *,
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
        compound_chain_indicator: (
            Sequence[float] | None) = None,
        **kwargs: Any,
) -> LongHorizonReconstructionV27Witness:
    runs = True
    try:
        head.twenty_six_way_value(
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
            compound_chain_indicator=(
                list(compound_chain_indicator)
                if compound_chain_indicator is not None
                else None),
            **kwargs)
    except Exception:
        runs = False
    return LongHorizonReconstructionV27Witness(
        schema=W75_LHR_V27_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        max_k=int(head.max_k),
        compound_chain_dim=int(head.compound_chain_dim),
        out_dim=int(head.out_dim),
        n_heads=26,
        twenty_six_way_runs=bool(runs),
    )


__all__ = [
    "W75_LHR_V27_SCHEMA_VERSION",
    "W75_DEFAULT_LHR_V27_MAX_K",
    "W75_DEFAULT_LHR_V27_COMPOUND_CHAIN_DIM",
    "W75_DEFAULT_LHR_V27_SWISH7_PROJ_DIM",
    "LongHorizonReconstructionV27Head",
    "fit_lhr_v27_seventeen_layer_scorer",
    "LongHorizonReconstructionV27Witness",
    "emit_lhr_v27_witness",
]
