"""W68 M13 — Long-Horizon Retention V20.

Strictly extends W67's ``coordpy.long_horizon_retention_v19``. V19
had 18 heads + a nine-layer scorer at max_k=384. V20 adds:

* **19 heads** (V19's 18 + partial-contradiction-recovery head).
* **Ten-layer scorer** — V19's nine layers + a tenth random+gelu
  layer before the final ridge.
* **max_k = 448** (vs V19's 384).

Honest scope (W68)
------------------

* Only the final ridge head is fit. The first nine layers are
  frozen random projections. ``W68-L-V20-LHR-SCORER-FIT-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.long_horizon_retention_v20 requires numpy"
        ) from exc

from .long_horizon_retention_v19 import (
    LongHorizonReconstructionV19Head,
    W67_DEFAULT_LHR_V19_MISH_PROJ_DIM,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W68_LHR_V20_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v20.v1")
W68_DEFAULT_LHR_V20_MAX_K: int = 448
W68_DEFAULT_LHR_V20_PARTIAL_CONTRADICTION_DIM: int = 8
W68_DEFAULT_LHR_V20_GELU_PROJ_DIM: int = 32


def _gelu(x: "_np.ndarray") -> "_np.ndarray":
    return 0.5 * x * (1.0 + _np.tanh(
        _np.sqrt(2.0 / _np.pi)
        * (x + 0.044715 * (x ** 3))))


@dataclasses.dataclass
class LongHorizonReconstructionV20Head:
    inner_v19: LongHorizonReconstructionV19Head
    max_k: int
    partial_contradiction_dim: int
    gelu_proj_dim: int
    gelu_proj_W: "_np.ndarray | None" = None
    scorer_layer10: "_np.ndarray | None" = None
    scorer_layer10_residual: float = 0.0
    partial_contradiction_W: "_np.ndarray | None" = None

    @classmethod
    def init(
            cls, *, max_k: int = W68_DEFAULT_LHR_V20_MAX_K,
            partial_contradiction_dim: int = (
                W68_DEFAULT_LHR_V20_PARTIAL_CONTRADICTION_DIM),
            gelu_proj_dim: int = (
                W68_DEFAULT_LHR_V20_GELU_PROJ_DIM),
            seed: int = 68200,
    ) -> "LongHorizonReconstructionV20Head":
        v19 = LongHorizonReconstructionV19Head.init(
            max_k=int(max_k), seed=int(seed))
        rng = _np.random.default_rng(int(seed) ^ 0xCAFE_68)
        out_dim = int(v19.out_dim)
        g_W = rng.standard_normal(
            (int(W67_DEFAULT_LHR_V19_MISH_PROJ_DIM),
             int(gelu_proj_dim))) * 0.05
        pc_W = rng.standard_normal(
            (int(partial_contradiction_dim),
             int(out_dim))) * 0.05
        return cls(
            inner_v19=v19,
            max_k=int(max_k),
            partial_contradiction_dim=int(
                partial_contradiction_dim),
            gelu_proj_dim=int(gelu_proj_dim),
            gelu_proj_W=g_W.astype(_np.float64),
            partial_contradiction_W=pc_W.astype(_np.float64),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v19.out_dim)

    def partial_contradiction_value(
            self, *,
            partial_contradiction_indicator: Sequence[float],
    ) -> "_np.ndarray":
        v = _np.asarray(
            partial_contradiction_indicator, dtype=_np.float64)
        if v.size < int(self.partial_contradiction_dim):
            v = _np.concatenate([
                v, _np.zeros(
                    int(self.partial_contradiction_dim) - v.size,
                    dtype=_np.float64)])
        elif v.size > int(self.partial_contradiction_dim):
            v = v[:int(self.partial_contradiction_dim)]
        return v @ self.partial_contradiction_W

    def nineteen_way_value(
            self, *, carrier: Sequence[float], k: int,
            partial_contradiction_indicator: (
                Sequence[float] | None) = None,
            **kwargs: Any,
    ) -> "_np.ndarray":
        v18 = self.inner_v19.eighteen_way_value(
            carrier=list(carrier), k=int(k), **kwargs)
        v18 = _np.asarray(v18, dtype=_np.float64)
        if partial_contradiction_indicator is not None:
            pc = self.partial_contradiction_value(
                partial_contradiction_indicator=list(
                    partial_contradiction_indicator))
            return v18 + 0.05 * pc
        return v18

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v20_head",
            "inner_v19_cid": str(self.inner_v19.cid()),
            "max_k": int(self.max_k),
            "partial_contradiction_dim": int(
                self.partial_contradiction_dim),
            "gelu_proj_dim": int(self.gelu_proj_dim),
            "gelu_proj_W_cid": (
                _ndarray_cid(self.gelu_proj_W)
                if self.gelu_proj_W is not None else "none"),
            "scorer_layer10_cid": (
                _ndarray_cid(self.scorer_layer10)
                if self.scorer_layer10 is not None
                else "untrained"),
            "scorer_layer10_residual": float(round(
                self.scorer_layer10_residual, 12)),
            "partial_contradiction_W_cid": (
                _ndarray_cid(self.partial_contradiction_W)
                if self.partial_contradiction_W is not None
                else "none"),
        })


def fit_lhr_v20_ten_layer_scorer(
        *, head: LongHorizonReconstructionV20Head,
        train_features: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        ridge_lambda: float = 0.10,
) -> tuple[LongHorizonReconstructionV20Head, dict[str, Any]]:
    """Fit the tenth-layer ridge on top of the V19 nine-layer
    pipeline (gelu projection then ridge)."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(train_targets, dtype=_np.float64)
    if X.shape[0] == 0:
        return head, {"converged": True, "n": 0}
    if head.gelu_proj_W is not None and X.shape[1] == int(
            W67_DEFAULT_LHR_V19_MISH_PROJ_DIM):
        f9 = _gelu(X @ _np.asarray(
            head.gelu_proj_W, dtype=_np.float64))
    else:
        f9 = X
    lam = max(float(ridge_lambda), 1e-9)
    A = f9.T @ f9 + lam * _np.eye(
        f9.shape[1], dtype=_np.float64)
    b = f9.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((f9.shape[1],), dtype=_np.float64)
    y_hat = f9 @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        head,
        scorer_layer10=_np.asarray(
            theta, dtype=_np.float64).copy(),
        scorer_layer10_residual=float(post))
    audit = {
        "schema": W68_LHR_V20_SCHEMA_VERSION,
        "kind": "lhr_v20_ten_layer_scorer",
        "pre_fit_residual": float(pre),
        "post_fit_residual": float(post),
        "converged": bool(post <= pre + 1e-9),
        "n": int(X.shape[0]),
    }
    return fitted, audit


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV20Witness:
    schema: str
    head_cid: str
    max_k: int
    partial_contradiction_dim: int
    out_dim: int
    n_heads: int
    nineteen_way_runs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "max_k": int(self.max_k),
            "partial_contradiction_dim": int(
                self.partial_contradiction_dim),
            "out_dim": int(self.out_dim),
            "n_heads": int(self.n_heads),
            "nineteen_way_runs": bool(self.nineteen_way_runs),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v20_witness",
            "witness": self.to_dict()})


def emit_lhr_v20_witness(
        head: LongHorizonReconstructionV20Head, *,
        carrier: Sequence[float], k: int = 16,
        partial_contradiction_indicator: (
            Sequence[float] | None) = None,
        **kwargs: Any,
) -> LongHorizonReconstructionV20Witness:
    runs = True
    try:
        head.nineteen_way_value(
            carrier=list(carrier), k=int(k),
            partial_contradiction_indicator=(
                list(partial_contradiction_indicator)
                if partial_contradiction_indicator is not None
                else None),
            **kwargs)
    except Exception:
        runs = False
    return LongHorizonReconstructionV20Witness(
        schema=W68_LHR_V20_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        max_k=int(head.max_k),
        partial_contradiction_dim=int(
            head.partial_contradiction_dim),
        out_dim=int(head.out_dim),
        n_heads=19,
        nineteen_way_runs=bool(runs),
    )


__all__ = [
    "W68_LHR_V20_SCHEMA_VERSION",
    "W68_DEFAULT_LHR_V20_MAX_K",
    "W68_DEFAULT_LHR_V20_PARTIAL_CONTRADICTION_DIM",
    "W68_DEFAULT_LHR_V20_GELU_PROJ_DIM",
    "LongHorizonReconstructionV20Head",
    "fit_lhr_v20_ten_layer_scorer",
    "LongHorizonReconstructionV20Witness",
    "emit_lhr_v20_witness",
]
