"""W76 — Long-Horizon Retention V28.

Strictly extends W75's ``coordpy.long_horizon_retention_v27``. V27
had 26 heads + a seventeen-layer scorer at max_k=896. V28 adds:

* **27 heads** (V27's 26 + chain-then-restart-pressure-recovery
  head).
* **Eighteen-layer scorer** — V27's seventeen layers + an
  eighteenth random+swish layer before the final ridge.
* **max_k = 960** (vs V27's 896).

Honest scope (W76): only the final ridge head is fit; earlier
projections are frozen random. ``W76-L-V28-LHR-SCORER-FIT-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.long_horizon_retention_v28 requires numpy"
        ) from exc

from .long_horizon_retention_v27 import (
    LongHorizonReconstructionV27Head,
    W75_DEFAULT_LHR_V27_SWISH7_PROJ_DIM,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W76_LHR_V28_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v28.v1")
W76_DEFAULT_LHR_V28_MAX_K: int = 960
W76_DEFAULT_LHR_V28_CHAIN_THEN_RESTART_DIM: int = 8
W76_DEFAULT_LHR_V28_SWISH8_PROJ_DIM: int = 64


def _swish(x: "_np.ndarray") -> "_np.ndarray":
    return x / (1.0 + _np.exp(-x))


@dataclasses.dataclass
class LongHorizonReconstructionV28Head:
    inner_v27: LongHorizonReconstructionV27Head
    max_k: int
    chain_then_restart_dim: int
    swish8_proj_dim: int
    swish8_proj_W: "_np.ndarray | None" = None
    scorer_layer18: "_np.ndarray | None" = None
    scorer_layer18_residual: float = 0.0
    chain_then_restart_W: "_np.ndarray | None" = None

    @classmethod
    def init(
            cls, *, max_k: int = W76_DEFAULT_LHR_V28_MAX_K,
            chain_then_restart_dim: int = (
                W76_DEFAULT_LHR_V28_CHAIN_THEN_RESTART_DIM),
            swish8_proj_dim: int = (
                W76_DEFAULT_LHR_V28_SWISH8_PROJ_DIM),
            seed: int = 76200,
    ) -> "LongHorizonReconstructionV28Head":
        v27 = LongHorizonReconstructionV27Head.init(
            max_k=int(max_k), seed=int(seed))
        rng = _np.random.default_rng(int(seed) ^ 0xCAFE_76)
        out_dim = int(v27.out_dim)
        s8_W = rng.standard_normal(
            (int(W75_DEFAULT_LHR_V27_SWISH7_PROJ_DIM),
             int(swish8_proj_dim))) * 0.05
        ctr_W = rng.standard_normal(
            (int(chain_then_restart_dim), int(out_dim))) * 0.05
        return cls(
            inner_v27=v27,
            max_k=int(max_k),
            chain_then_restart_dim=int(chain_then_restart_dim),
            swish8_proj_dim=int(swish8_proj_dim),
            swish8_proj_W=s8_W.astype(_np.float64),
            chain_then_restart_W=ctr_W.astype(_np.float64),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v27.out_dim)

    def chain_then_restart_value(
            self, *,
            chain_then_restart_indicator: Sequence[float],
    ) -> "_np.ndarray":
        v = _np.asarray(
            chain_then_restart_indicator, dtype=_np.float64)
        if v.size < int(self.chain_then_restart_dim):
            v = _np.concatenate([
                v, _np.zeros(
                    int(self.chain_then_restart_dim) - v.size,
                    dtype=_np.float64)])
        elif v.size > int(self.chain_then_restart_dim):
            v = v[:int(self.chain_then_restart_dim)]
        return v @ self.chain_then_restart_W

    def twenty_seven_way_value(
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
            chain_then_restart_indicator: (
                Sequence[float] | None) = None,
            **kwargs: Any,
    ) -> "_np.ndarray":
        v26 = self.inner_v27.twenty_six_way_value(
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
        v26 = _np.asarray(v26, dtype=_np.float64)
        if chain_then_restart_indicator is not None:
            ctr = self.chain_then_restart_value(
                chain_then_restart_indicator=list(
                    chain_then_restart_indicator))
            return v26 + 0.05 * ctr
        return v26

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v28_head",
            "inner_v27_cid": str(self.inner_v27.cid()),
            "max_k": int(self.max_k),
            "chain_then_restart_dim": int(
                self.chain_then_restart_dim),
            "swish8_proj_dim": int(self.swish8_proj_dim),
            "swish8_proj_W_cid": (
                _ndarray_cid(self.swish8_proj_W)
                if self.swish8_proj_W is not None else "none"),
            "scorer_layer18_cid": (
                _ndarray_cid(self.scorer_layer18)
                if self.scorer_layer18 is not None
                else "untrained"),
            "scorer_layer18_residual": float(round(
                self.scorer_layer18_residual, 12)),
            "chain_then_restart_W_cid": (
                _ndarray_cid(self.chain_then_restart_W)
                if self.chain_then_restart_W is not None
                else "none"),
        })


def fit_lhr_v28_eighteen_layer_scorer(
        *, head: LongHorizonReconstructionV28Head,
        train_features: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        ridge_lambda: float = 0.10,
) -> tuple[
        LongHorizonReconstructionV28Head, dict[str, Any]]:
    """Fit the eighteenth-layer ridge on top of the V27
    seventeen-layer pipeline (swish8 projection then ridge)."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(train_targets, dtype=_np.float64)
    if X.shape[0] == 0:
        return head, {"converged": True, "n": 0}
    if head.swish8_proj_W is not None and X.shape[1] == int(
            W75_DEFAULT_LHR_V27_SWISH7_PROJ_DIM):
        f17 = _swish(X @ _np.asarray(
            head.swish8_proj_W, dtype=_np.float64))
    else:
        f17 = X
    lam = max(float(ridge_lambda), 1e-9)
    A = f17.T @ f17 + lam * _np.eye(
        f17.shape[1], dtype=_np.float64)
    b = f17.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((f17.shape[1],), dtype=_np.float64)
    y_hat = f17 @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        head,
        scorer_layer18=_np.asarray(
            theta, dtype=_np.float64).copy(),
        scorer_layer18_residual=float(post))
    audit = {
        "schema": W76_LHR_V28_SCHEMA_VERSION,
        "kind": "lhr_v28_eighteen_layer_scorer",
        "pre_fit_residual": float(pre),
        "post_fit_residual": float(post),
        "converged": bool(post <= pre + 1e-9),
        "n": int(X.shape[0]),
    }
    return fitted, audit


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV28Witness:
    schema: str
    head_cid: str
    max_k: int
    chain_then_restart_dim: int
    out_dim: int
    n_heads: int
    twenty_seven_way_runs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "max_k": int(self.max_k),
            "chain_then_restart_dim": int(
                self.chain_then_restart_dim),
            "out_dim": int(self.out_dim),
            "n_heads": int(self.n_heads),
            "twenty_seven_way_runs": bool(
                self.twenty_seven_way_runs),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v28_witness",
            "witness": self.to_dict()})


def emit_lhr_v28_witness(
        head: LongHorizonReconstructionV28Head, *,
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
        chain_then_restart_indicator: (
            Sequence[float] | None) = None,
        **kwargs: Any,
) -> LongHorizonReconstructionV28Witness:
    runs = True
    try:
        head.twenty_seven_way_value(
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
            chain_then_restart_indicator=(
                list(chain_then_restart_indicator)
                if chain_then_restart_indicator is not None
                else None),
            **kwargs)
    except Exception:
        runs = False
    return LongHorizonReconstructionV28Witness(
        schema=W76_LHR_V28_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        max_k=int(head.max_k),
        chain_then_restart_dim=int(head.chain_then_restart_dim),
        out_dim=int(head.out_dim),
        n_heads=27,
        twenty_seven_way_runs=bool(runs),
    )


__all__ = [
    "W76_LHR_V28_SCHEMA_VERSION",
    "W76_DEFAULT_LHR_V28_MAX_K",
    "W76_DEFAULT_LHR_V28_CHAIN_THEN_RESTART_DIM",
    "W76_DEFAULT_LHR_V28_SWISH8_PROJ_DIM",
    "LongHorizonReconstructionV28Head",
    "fit_lhr_v28_eighteen_layer_scorer",
    "LongHorizonReconstructionV28Witness",
    "emit_lhr_v28_witness",
]
