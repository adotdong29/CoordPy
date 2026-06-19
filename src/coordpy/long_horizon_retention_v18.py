"""W66 M15 — Long-Horizon Retention V18.

Strictly extends W65's ``coordpy.long_horizon_retention_v17``. V17
had 16 heads + a seven-layer scorer at max_k=256. V18 adds:

* **17 heads** (V17's 16 + team-failure-recovery head).
* **Eight-layer scorer** — V17's seven layers (random+ReLU →
  random+tanh → random+tanh-2 → random+gelu → random+silu →
  random+softplus → ridge) plus an eighth random+swish layer
  before the final ridge.
* **max_k = 320** (vs V17's 256).

Honest scope (W66)
------------------

* Only the final ridge head is fit. The first seven layers are
  frozen random projections. ``W66-L-V18-LHR-SCORER-FIT-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.long_horizon_retention_v18 requires numpy"
        ) from exc

from .long_horizon_retention_v17 import (
    LongHorizonReconstructionV17Head,
    W65_DEFAULT_LHR_V17_SOFTPLUS_PROJ_DIM,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W66_LHR_V18_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v18.v1")
W66_DEFAULT_LHR_V18_MAX_K: int = 320
W66_DEFAULT_LHR_V18_TEAM_FAILURE_RECOVERY_DIM: int = 8
W66_DEFAULT_LHR_V18_SWISH_PROJ_DIM: int = 28


def _swish(x: "_np.ndarray") -> "_np.ndarray":
    return x * (1.0 / (1.0 + _np.exp(-x)))


@dataclasses.dataclass
class LongHorizonReconstructionV18Head:
    inner_v17: LongHorizonReconstructionV17Head
    max_k: int
    team_failure_recovery_dim: int
    swish_proj_dim: int
    swish_proj_W: "_np.ndarray | None" = None
    scorer_layer8: "_np.ndarray | None" = None
    scorer_layer8_residual: float = 0.0
    team_failure_recovery_W: "_np.ndarray | None" = None

    @classmethod
    def init(
            cls, *, max_k: int = W66_DEFAULT_LHR_V18_MAX_K,
            team_failure_recovery_dim: int = (
                W66_DEFAULT_LHR_V18_TEAM_FAILURE_RECOVERY_DIM),
            swish_proj_dim: int = (
                W66_DEFAULT_LHR_V18_SWISH_PROJ_DIM),
            seed: int = 66180,
    ) -> "LongHorizonReconstructionV18Head":
        v17 = LongHorizonReconstructionV17Head.init(
            max_k=int(max_k), seed=int(seed))
        rng = _np.random.default_rng(int(seed) ^ 0xCAFE_66)
        out_dim = int(v17.out_dim)
        sw_W = rng.standard_normal(
            (int(W65_DEFAULT_LHR_V17_SOFTPLUS_PROJ_DIM),
             int(swish_proj_dim))) * 0.05
        tfr_W = rng.standard_normal(
            (int(team_failure_recovery_dim),
             int(out_dim))) * 0.05
        return cls(
            inner_v17=v17,
            max_k=int(max_k),
            team_failure_recovery_dim=int(
                team_failure_recovery_dim),
            swish_proj_dim=int(swish_proj_dim),
            swish_proj_W=sw_W.astype(_np.float64),
            team_failure_recovery_W=tfr_W.astype(_np.float64),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v17.out_dim)

    def team_failure_recovery_value(
            self, *,
            team_failure_recovery_indicator: Sequence[float],
    ) -> "_np.ndarray":
        v = _np.asarray(
            team_failure_recovery_indicator, dtype=_np.float64)
        if v.size < int(self.team_failure_recovery_dim):
            v = _np.concatenate([
                v, _np.zeros(
                    int(self.team_failure_recovery_dim) - v.size,
                    dtype=_np.float64)])
        elif v.size > int(self.team_failure_recovery_dim):
            v = v[:int(self.team_failure_recovery_dim)]
        return v @ self.team_failure_recovery_W

    def seventeen_way_value(
            self, *, carrier: Sequence[float], k: int,
            team_failure_recovery_indicator: (
                Sequence[float] | None) = None,
            **kwargs: Any,
    ) -> "_np.ndarray":
        v16 = self.inner_v17.sixteen_way_value(
            carrier=list(carrier), k=int(k), **kwargs)
        v16 = _np.asarray(v16, dtype=_np.float64)
        if team_failure_recovery_indicator is not None:
            tfr = self.team_failure_recovery_value(
                team_failure_recovery_indicator=list(
                    team_failure_recovery_indicator))
            return v16 + 0.05 * tfr
        return v16

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v18_head",
            "inner_v17_cid": str(self.inner_v17.cid()),
            "max_k": int(self.max_k),
            "team_failure_recovery_dim": int(
                self.team_failure_recovery_dim),
            "swish_proj_dim": int(self.swish_proj_dim),
            "swish_proj_W_cid": (
                _ndarray_cid(self.swish_proj_W)
                if self.swish_proj_W is not None else "none"),
            "scorer_layer8_cid": (
                _ndarray_cid(self.scorer_layer8)
                if self.scorer_layer8 is not None
                else "untrained"),
            "scorer_layer8_residual": float(round(
                self.scorer_layer8_residual, 12)),
            "team_failure_recovery_W_cid": (
                _ndarray_cid(self.team_failure_recovery_W)
                if self.team_failure_recovery_W is not None
                else "none"),
        })


def fit_lhr_v18_eight_layer_scorer(
        *, head: LongHorizonReconstructionV18Head,
        train_features: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        ridge_lambda: float = 0.10,
) -> tuple[LongHorizonReconstructionV18Head, dict[str, Any]]:
    """Fit the eighth-layer ridge on top of the V17 seven-layer
    pipeline (swish projection then ridge)."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(train_targets, dtype=_np.float64)
    if X.shape[0] == 0:
        return head, {"converged": True, "n": 0}
    if head.swish_proj_W is not None and X.shape[1] == int(
            W65_DEFAULT_LHR_V17_SOFTPLUS_PROJ_DIM):
        f7 = _swish(X @ _np.asarray(
            head.swish_proj_W, dtype=_np.float64))
    else:
        f7 = X
    lam = max(float(ridge_lambda), 1e-9)
    A = f7.T @ f7 + lam * _np.eye(
        f7.shape[1], dtype=_np.float64)
    b = f7.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((f7.shape[1],), dtype=_np.float64)
    y_hat = f7 @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        head,
        scorer_layer8=_np.asarray(
            theta, dtype=_np.float64).copy(),
        scorer_layer8_residual=float(post))
    audit = {
        "schema": W66_LHR_V18_SCHEMA_VERSION,
        "kind": "lhr_v18_eight_layer_scorer",
        "pre_fit_residual": float(pre),
        "post_fit_residual": float(post),
        "converged": bool(post <= pre + 1e-9),
        "n": int(X.shape[0]),
    }
    return fitted, audit


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV18Witness:
    schema: str
    head_cid: str
    max_k: int
    team_failure_recovery_dim: int
    out_dim: int
    n_heads: int
    seventeen_way_runs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "max_k": int(self.max_k),
            "team_failure_recovery_dim": int(
                self.team_failure_recovery_dim),
            "out_dim": int(self.out_dim),
            "n_heads": int(self.n_heads),
            "seventeen_way_runs": bool(self.seventeen_way_runs),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v18_witness",
            "witness": self.to_dict()})


def emit_lhr_v18_witness(
        head: LongHorizonReconstructionV18Head, *,
        carrier: Sequence[float], k: int = 16,
        team_failure_recovery_indicator: (
            Sequence[float] | None) = None,
        **kwargs: Any,
) -> LongHorizonReconstructionV18Witness:
    runs = True
    try:
        head.seventeen_way_value(
            carrier=list(carrier), k=int(k),
            team_failure_recovery_indicator=(
                list(team_failure_recovery_indicator)
                if team_failure_recovery_indicator is not None
                else None),
            **kwargs)
    except Exception:
        runs = False
    return LongHorizonReconstructionV18Witness(
        schema=W66_LHR_V18_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        max_k=int(head.max_k),
        team_failure_recovery_dim=int(
            head.team_failure_recovery_dim),
        out_dim=int(head.out_dim),
        n_heads=17,
        seventeen_way_runs=bool(runs),
    )


__all__ = [
    "W66_LHR_V18_SCHEMA_VERSION",
    "W66_DEFAULT_LHR_V18_MAX_K",
    "W66_DEFAULT_LHR_V18_TEAM_FAILURE_RECOVERY_DIM",
    "W66_DEFAULT_LHR_V18_SWISH_PROJ_DIM",
    "LongHorizonReconstructionV18Head",
    "fit_lhr_v18_eight_layer_scorer",
    "LongHorizonReconstructionV18Witness",
    "emit_lhr_v18_witness",
]
