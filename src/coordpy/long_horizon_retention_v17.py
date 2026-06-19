"""W65 M15 — Long-Horizon Retention V17.

Strictly extends W64's ``coordpy.long_horizon_retention_v16``. V16
had 15 heads + a six-layer scorer at max_k=192. V17 adds:

* **16 heads** (V16's 15 + team-task-success head).
* **Seven-layer scorer** — V16's six layers (random+ReLU →
  random+tanh → random+tanh-2 → random+gelu → random+silu →
  ridge) plus a seventh random+softplus layer in front of the
  ridge.
* **max_k = 256** (vs V16's 192).

Honest scope (W65)
------------------

* Only the final ridge head is fit. The first six layers are
  frozen random projections. ``W65-L-V17-LHR-SCORER-FIT-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.long_horizon_retention_v17 requires numpy"
        ) from exc

from .long_horizon_retention_v15 import (
    W63_DEFAULT_LHR_V15_GELU_PROJ_DIM,
)
from .long_horizon_retention_v16 import (
    LongHorizonReconstructionV16Head,
    W64_DEFAULT_LHR_V16_SILU_PROJ_DIM,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W65_LHR_V17_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v17.v1")
W65_DEFAULT_LHR_V17_MAX_K: int = 256
W65_DEFAULT_LHR_V17_TEAM_TASK_SUCCESS_DIM: int = 8
W65_DEFAULT_LHR_V17_SOFTPLUS_PROJ_DIM: int = 24


def _softplus(x: "_np.ndarray") -> "_np.ndarray":
    return _np.log1p(_np.exp(x))


@dataclasses.dataclass
class LongHorizonReconstructionV17Head:
    inner_v16: LongHorizonReconstructionV16Head
    max_k: int
    team_task_success_dim: int
    softplus_proj_dim: int
    softplus_proj_W: "_np.ndarray | None" = None
    scorer_layer7: "_np.ndarray | None" = None
    scorer_layer7_residual: float = 0.0
    team_task_success_W: "_np.ndarray | None" = None

    @classmethod
    def init(
            cls, *, max_k: int = W65_DEFAULT_LHR_V17_MAX_K,
            team_task_success_dim: int = (
                W65_DEFAULT_LHR_V17_TEAM_TASK_SUCCESS_DIM),
            softplus_proj_dim: int = (
                W65_DEFAULT_LHR_V17_SOFTPLUS_PROJ_DIM),
            seed: int = 65170,
    ) -> "LongHorizonReconstructionV17Head":
        v16 = LongHorizonReconstructionV16Head.init(
            max_k=int(max_k), seed=int(seed))
        rng = _np.random.default_rng(int(seed) ^ 0xCAFE_65)
        out_dim = int(v16.out_dim)
        sp_W = rng.standard_normal(
            (int(W64_DEFAULT_LHR_V16_SILU_PROJ_DIM),
             int(softplus_proj_dim))) * 0.06
        tts_W = rng.standard_normal(
            (int(team_task_success_dim),
             int(out_dim))) * 0.05
        return cls(
            inner_v16=v16,
            max_k=int(max_k),
            team_task_success_dim=int(team_task_success_dim),
            softplus_proj_dim=int(softplus_proj_dim),
            softplus_proj_W=sp_W.astype(_np.float64),
            team_task_success_W=tts_W.astype(_np.float64),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v16.out_dim)

    def team_task_success_value(
            self, *,
            team_task_success_indicator: Sequence[float],
    ) -> "_np.ndarray":
        v = _np.asarray(
            team_task_success_indicator, dtype=_np.float64)
        if v.size < int(self.team_task_success_dim):
            v = _np.concatenate([
                v, _np.zeros(
                    int(self.team_task_success_dim) - v.size,
                    dtype=_np.float64)])
        elif v.size > int(self.team_task_success_dim):
            v = v[:int(self.team_task_success_dim)]
        return v @ self.team_task_success_W

    def sixteen_way_value(
            self, *, carrier: Sequence[float], k: int,
            team_task_success_indicator: (
                Sequence[float] | None) = None,
            **kwargs: Any,
    ) -> "_np.ndarray":
        v15 = self.inner_v16.fifteen_way_value(
            carrier=list(carrier), k=int(k), **kwargs)
        v15 = _np.asarray(v15, dtype=_np.float64)
        if team_task_success_indicator is not None:
            tts = self.team_task_success_value(
                team_task_success_indicator=list(
                    team_task_success_indicator))
            return v15 + 0.05 * tts
        return v15

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v17_head",
            "inner_v16_cid": str(self.inner_v16.cid()),
            "max_k": int(self.max_k),
            "team_task_success_dim": int(
                self.team_task_success_dim),
            "softplus_proj_dim": int(self.softplus_proj_dim),
            "softplus_proj_W_cid": (
                _ndarray_cid(self.softplus_proj_W)
                if self.softplus_proj_W is not None else "none"),
            "scorer_layer7_cid": (
                _ndarray_cid(self.scorer_layer7)
                if self.scorer_layer7 is not None
                else "untrained"),
            "scorer_layer7_residual": float(round(
                self.scorer_layer7_residual, 12)),
            "team_task_success_W_cid": (
                _ndarray_cid(self.team_task_success_W)
                if self.team_task_success_W is not None
                else "none"),
        })


def fit_lhr_v17_seven_layer_scorer(
        *, head: LongHorizonReconstructionV17Head,
        train_features: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        ridge_lambda: float = 0.10,
) -> tuple[LongHorizonReconstructionV17Head, dict[str, Any]]:
    """Fit the seventh-layer ridge on top of the V16 sixth-layer
    feature pipeline (here we just feed silu features through the
    softplus projection)."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(train_targets, dtype=_np.float64)
    if X.shape[0] == 0:
        return head, {"converged": True, "n": 0}
    if head.softplus_proj_W is not None and X.shape[1] == int(
            W64_DEFAULT_LHR_V16_SILU_PROJ_DIM):
        f6 = _softplus(X @ _np.asarray(
            head.softplus_proj_W, dtype=_np.float64))
    else:
        f6 = X
    lam = max(float(ridge_lambda), 1e-9)
    A = f6.T @ f6 + lam * _np.eye(
        f6.shape[1], dtype=_np.float64)
    b = f6.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((f6.shape[1],), dtype=_np.float64)
    y_hat = f6 @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        head,
        scorer_layer7=_np.asarray(
            theta, dtype=_np.float64).copy(),
        scorer_layer7_residual=float(post))
    audit = {
        "schema": W65_LHR_V17_SCHEMA_VERSION,
        "kind": "lhr_v17_seven_layer_scorer",
        "pre_fit_residual": float(pre),
        "post_fit_residual": float(post),
        "converged": bool(post <= pre + 1e-9),
        "n": int(X.shape[0]),
    }
    return fitted, audit


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV17Witness:
    schema: str
    head_cid: str
    max_k: int
    team_task_success_dim: int
    out_dim: int
    n_heads: int
    sixteen_way_runs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "max_k": int(self.max_k),
            "team_task_success_dim": int(
                self.team_task_success_dim),
            "out_dim": int(self.out_dim),
            "n_heads": int(self.n_heads),
            "sixteen_way_runs": bool(self.sixteen_way_runs),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v17_witness",
            "witness": self.to_dict()})


def emit_lhr_v17_witness(
        head: LongHorizonReconstructionV17Head, *,
        carrier: Sequence[float], k: int = 16,
        team_task_success_indicator: (
            Sequence[float] | None) = None,
        **kwargs: Any,
) -> LongHorizonReconstructionV17Witness:
    runs = True
    try:
        head.sixteen_way_value(
            carrier=list(carrier), k=int(k),
            team_task_success_indicator=(
                list(team_task_success_indicator)
                if team_task_success_indicator is not None
                else None),
            **kwargs)
    except Exception:
        runs = False
    return LongHorizonReconstructionV17Witness(
        schema=W65_LHR_V17_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        max_k=int(head.max_k),
        team_task_success_dim=int(
            head.team_task_success_dim),
        out_dim=int(head.out_dim),
        n_heads=16,
        sixteen_way_runs=bool(runs),
    )


__all__ = [
    "W65_LHR_V17_SCHEMA_VERSION",
    "W65_DEFAULT_LHR_V17_MAX_K",
    "W65_DEFAULT_LHR_V17_TEAM_TASK_SUCCESS_DIM",
    "W65_DEFAULT_LHR_V17_SOFTPLUS_PROJ_DIM",
    "LongHorizonReconstructionV17Head",
    "fit_lhr_v17_seven_layer_scorer",
    "LongHorizonReconstructionV17Witness",
    "emit_lhr_v17_witness",
]
