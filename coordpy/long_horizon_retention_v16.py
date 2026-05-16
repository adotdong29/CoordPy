"""W64 M15 — Long-Horizon Retention V16.

Strictly extends W63's ``coordpy.long_horizon_retention_v15``. V15
had 14 heads + a five-layer scorer. V16 adds:

* **15 heads** (V15's 14 + replay-dominance-primary head).
* **Six-layer scorer** — V15's five layers (random+ReLU →
  random+tanh → random+tanh-2 → random+gelu → ridge) plus a sixth
  random+silu layer in front of the ridge.
* **max_k = 192** (vs V15's 160).

Honest scope (W64)
------------------

* The replay-dominance-primary head is a linear projection of the
  V9 ``replay_dominance_witness`` summary; not a calibrated
  retention signal.
* Only the final ridge head is fit. ``W64-L-V16-LHR-SCORER-FIT-CAP``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.long_horizon_retention_v16 requires numpy"
        ) from exc

from .long_horizon_retention_v15 import (
    LongHorizonReconstructionV15Head,
    W63_DEFAULT_LHR_V15_GELU_PROJ_DIM,
    W63_DEFAULT_LHR_V15_HIDDEN_WINS_DIM,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W64_LHR_V16_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v16.v1")
W64_DEFAULT_LHR_V16_MAX_K: int = 192
W64_DEFAULT_LHR_V16_REPLAY_DOMINANCE_PRIMARY_DIM: int = 8
W64_DEFAULT_LHR_V16_SILU_PROJ_DIM: int = 20


def _silu(x: "_np.ndarray") -> "_np.ndarray":
    return x / (1.0 + _np.exp(-x))


@dataclasses.dataclass
class LongHorizonReconstructionV16Head:
    inner_v15: LongHorizonReconstructionV15Head
    max_k: int
    replay_dominance_primary_dim: int
    silu_proj_dim: int
    silu_proj_W: "_np.ndarray | None" = None
    scorer_layer6: "_np.ndarray | None" = None
    scorer_layer6_residual: float = 0.0
    replay_dominance_primary_W: "_np.ndarray | None" = None

    @classmethod
    def init(
            cls, *, max_k: int = W64_DEFAULT_LHR_V16_MAX_K,
            replay_dominance_primary_dim: int = (
                W64_DEFAULT_LHR_V16_REPLAY_DOMINANCE_PRIMARY_DIM),
            silu_proj_dim: int = (
                W64_DEFAULT_LHR_V16_SILU_PROJ_DIM),
            seed: int = 64160,
    ) -> "LongHorizonReconstructionV16Head":
        v15 = LongHorizonReconstructionV15Head.init(
            max_k=int(max_k), seed=int(seed))
        rng = _np.random.default_rng(int(seed) ^ 0xBEEF_64)
        out_dim = int(v15.out_dim)
        silu_W = rng.standard_normal(
            (int(W63_DEFAULT_LHR_V15_GELU_PROJ_DIM),
             int(silu_proj_dim))) * 0.07
        rdp_W = rng.standard_normal(
            (int(replay_dominance_primary_dim),
             int(out_dim))) * 0.06
        return cls(
            inner_v15=v15,
            max_k=int(max_k),
            replay_dominance_primary_dim=int(
                replay_dominance_primary_dim),
            silu_proj_dim=int(silu_proj_dim),
            silu_proj_W=silu_W.astype(_np.float64),
            replay_dominance_primary_W=rdp_W.astype(_np.float64),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v15.out_dim)

    def replay_dominance_primary_value(
            self, *,
            replay_dominance_primary_indicator: Sequence[float],
    ) -> "_np.ndarray":
        v = _np.asarray(
            replay_dominance_primary_indicator,
            dtype=_np.float64)
        if v.size < int(self.replay_dominance_primary_dim):
            v = _np.concatenate([
                v, _np.zeros(
                    int(self.replay_dominance_primary_dim)
                        - v.size,
                    dtype=_np.float64)])
        elif v.size > int(self.replay_dominance_primary_dim):
            v = v[:int(self.replay_dominance_primary_dim)]
        return v @ self.replay_dominance_primary_W

    def fifteen_way_value(
            self, *, carrier: Sequence[float], k: int,
            replay_dominance_indicator: (
                Sequence[float] | None) = None,
            hidden_wins_indicator: (
                Sequence[float] | None) = None,
            replay_dominance_primary_indicator: (
                Sequence[float] | None) = None,
            **kwargs: Any,
    ) -> "_np.ndarray":
        v14 = self.inner_v15.fourteen_way_value(
            carrier=list(carrier), k=int(k),
            replay_dominance_indicator=(
                replay_dominance_indicator),
            hidden_wins_indicator=hidden_wins_indicator,
            **kwargs)
        v14 = _np.asarray(v14, dtype=_np.float64)
        if replay_dominance_primary_indicator is not None:
            rdp = self.replay_dominance_primary_value(
                replay_dominance_primary_indicator=(
                    list(replay_dominance_primary_indicator)))
            return v14 + 0.07 * rdp
        return v14

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v16_head",
            "inner_v15_cid": str(self.inner_v15.cid()),
            "max_k": int(self.max_k),
            "replay_dominance_primary_dim": int(
                self.replay_dominance_primary_dim),
            "silu_proj_dim": int(self.silu_proj_dim),
            "silu_proj_W_cid": (
                _ndarray_cid(self.silu_proj_W)
                if self.silu_proj_W is not None else "none"),
            "scorer_layer6_cid": (
                _ndarray_cid(self.scorer_layer6)
                if self.scorer_layer6 is not None
                else "untrained"),
            "scorer_layer6_residual": float(round(
                self.scorer_layer6_residual, 12)),
            "replay_dominance_primary_W_cid": (
                _ndarray_cid(self.replay_dominance_primary_W)
                if self.replay_dominance_primary_W is not None
                else "none"),
        })


def fit_lhr_v16_six_layer_scorer(
        *, head: LongHorizonReconstructionV16Head,
        train_features: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        ridge_lambda: float = 0.10,
) -> tuple[LongHorizonReconstructionV16Head, dict[str, Any]]:
    """Fit the sixth-layer ridge on top of the V15 fifth-layer
    feature pipeline."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(train_targets, dtype=_np.float64)
    if X.shape[0] == 0:
        return head, {"converged": True, "n": 0}
    if head.silu_proj_W is not None and X.shape[1] == int(
            W63_DEFAULT_LHR_V15_GELU_PROJ_DIM):
        f5 = _silu(X @ _np.asarray(
            head.silu_proj_W, dtype=_np.float64))
    else:
        f5 = X
    lam = max(float(ridge_lambda), 1e-9)
    A = f5.T @ f5 + lam * _np.eye(
        f5.shape[1], dtype=_np.float64)
    b = f5.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((f5.shape[1],), dtype=_np.float64)
    y_hat = f5 @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        head,
        scorer_layer6=_np.asarray(
            theta, dtype=_np.float64).copy(),
        scorer_layer6_residual=float(post))
    audit = {
        "schema": W64_LHR_V16_SCHEMA_VERSION,
        "kind": "lhr_v16_six_layer_scorer",
        "pre_fit_residual": float(pre),
        "post_fit_residual": float(post),
        "converged": bool(post <= pre + 1e-9),
        "n": int(X.shape[0]),
    }
    return fitted, audit


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV16Witness:
    schema: str
    head_cid: str
    max_k: int
    replay_dominance_primary_dim: int
    out_dim: int
    n_heads: int
    fifteen_way_runs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "max_k": int(self.max_k),
            "replay_dominance_primary_dim": int(
                self.replay_dominance_primary_dim),
            "out_dim": int(self.out_dim),
            "n_heads": int(self.n_heads),
            "fifteen_way_runs": bool(self.fifteen_way_runs),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v16_witness",
            "witness": self.to_dict()})


def emit_lhr_v16_witness(
        head: LongHorizonReconstructionV16Head, *,
        carrier: Sequence[float], k: int = 16,
        replay_dominance_indicator: (
            Sequence[float] | None) = None,
        hidden_wins_indicator: (
            Sequence[float] | None) = None,
        replay_dominance_primary_indicator: (
            Sequence[float] | None) = None,
) -> LongHorizonReconstructionV16Witness:
    runs = True
    try:
        head.fifteen_way_value(
            carrier=list(carrier), k=int(k),
            replay_dominance_indicator=(
                list(replay_dominance_indicator)
                if replay_dominance_indicator is not None
                else None),
            hidden_wins_indicator=(
                list(hidden_wins_indicator)
                if hidden_wins_indicator is not None
                else None),
            replay_dominance_primary_indicator=(
                list(replay_dominance_primary_indicator)
                if replay_dominance_primary_indicator
                    is not None else None))
    except Exception:
        runs = False
    return LongHorizonReconstructionV16Witness(
        schema=W64_LHR_V16_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        max_k=int(head.max_k),
        replay_dominance_primary_dim=int(
            head.replay_dominance_primary_dim),
        out_dim=int(head.out_dim),
        n_heads=15,
        fifteen_way_runs=bool(runs),
    )


__all__ = [
    "W64_LHR_V16_SCHEMA_VERSION",
    "W64_DEFAULT_LHR_V16_MAX_K",
    "W64_DEFAULT_LHR_V16_REPLAY_DOMINANCE_PRIMARY_DIM",
    "W64_DEFAULT_LHR_V16_SILU_PROJ_DIM",
    "LongHorizonReconstructionV16Head",
    "fit_lhr_v16_six_layer_scorer",
    "LongHorizonReconstructionV16Witness",
    "emit_lhr_v16_witness",
]
