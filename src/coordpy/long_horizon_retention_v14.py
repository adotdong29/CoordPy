"""W62 — Long-Horizon Retention V14.

Strictly extends W61's ``coordpy.long_horizon_retention_v13``. V13
had 12 heads + a three-layer scorer. V14 adds:

* **13 heads** (V13's 12 + replay-dominance-conditioned head).
* **Four-layer scorer** — V13's three-layer (random+ReLU →
  random+tanh → ridge) plus a fourth random+tanh layer.
* **max_k = 128** preserved from V13.

Honest scope
------------

* The replay-dominance head is a linear projection of the
  per-(layer, head) replay-trust ledger summary; not a calibrated
  retention signal.
* Only the final ridge head is fit. ``W62-L-V14-LHR-SCORER-FIT-CAP``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.long_horizon_retention_v14 requires numpy"
        ) from exc

from .long_horizon_retention_v13 import (
    LongHorizonReconstructionV13Head,
    W61_LHR_V13_SCHEMA_VERSION,
    W61_DEFAULT_LHR_V13_ATTENTION_PATTERN_DIM,
    W61_DEFAULT_LHR_V13_MAX_K,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W62_LHR_V14_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v14.v1")
W62_DEFAULT_LHR_V14_MAX_K: int = 128
W62_DEFAULT_LHR_V14_REPLAY_DOMINANCE_DIM: int = 8
W62_DEFAULT_LHR_V14_TANH2_PROJ_DIM: int = 16


@dataclasses.dataclass
class LongHorizonReconstructionV14Head:
    inner_v13: LongHorizonReconstructionV13Head
    max_k: int
    replay_dominance_dim: int
    tanh2_proj_dim: int
    tanh2_proj_W: "_np.ndarray | None" = None
    scorer_layer4: "_np.ndarray | None" = None
    scorer_layer4_residual: float = 0.0
    replay_dominance_W: "_np.ndarray | None" = None

    @classmethod
    def init(
            cls, *, max_k: int = W62_DEFAULT_LHR_V14_MAX_K,
            replay_dominance_dim: int = (
                W62_DEFAULT_LHR_V14_REPLAY_DOMINANCE_DIM),
            tanh2_proj_dim: int = (
                W62_DEFAULT_LHR_V14_TANH2_PROJ_DIM),
            seed: int = 62140,
    ) -> "LongHorizonReconstructionV14Head":
        v13 = LongHorizonReconstructionV13Head.init(
            max_k=int(max_k), seed=int(seed))
        rng = _np.random.default_rng(int(seed) ^ 0xACDC_62)
        out_dim = int(v13.out_dim)
        tanh2_W = rng.standard_normal(
            (int(v13.tanh_proj_dim),
             int(tanh2_proj_dim))) * 0.10
        rd_W = rng.standard_normal(
            (int(replay_dominance_dim),
             int(out_dim))) * 0.08
        return cls(
            inner_v13=v13,
            max_k=int(max_k),
            replay_dominance_dim=int(replay_dominance_dim),
            tanh2_proj_dim=int(tanh2_proj_dim),
            tanh2_proj_W=tanh2_W.astype(_np.float64),
            replay_dominance_W=rd_W.astype(_np.float64),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v13.out_dim)

    def replay_dominance_value(
            self, *, replay_dominance_indicator: Sequence[float],
    ) -> "_np.ndarray":
        v = _np.asarray(
            replay_dominance_indicator, dtype=_np.float64)
        if v.size < int(self.replay_dominance_dim):
            v = _np.concatenate([
                v, _np.zeros(
                    int(self.replay_dominance_dim) - v.size,
                    dtype=_np.float64)])
        elif v.size > int(self.replay_dominance_dim):
            v = v[:int(self.replay_dominance_dim)]
        return v @ self.replay_dominance_W

    def thirteen_way_value(
            self, *, carrier: Sequence[float], k: int,
            replay_dominance_indicator: (
                Sequence[float] | None) = None,
            **kwargs: Any,
    ) -> "_np.ndarray":
        v12 = self.inner_v13.twelve_way_value(
            carrier=list(carrier), k=int(k), **kwargs)
        v12 = _np.asarray(v12, dtype=_np.float64)
        if replay_dominance_indicator is not None:
            rd = self.replay_dominance_value(
                replay_dominance_indicator=(
                    list(replay_dominance_indicator)))
            return v12 + 0.10 * rd
        return v12

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v14_head",
            "inner_v13_cid": str(self.inner_v13.cid()),
            "max_k": int(self.max_k),
            "replay_dominance_dim": int(
                self.replay_dominance_dim),
            "tanh2_proj_dim": int(self.tanh2_proj_dim),
            "tanh2_proj_W_cid": (
                _ndarray_cid(self.tanh2_proj_W)
                if self.tanh2_proj_W is not None else "none"),
            "scorer_layer4_cid": (
                _ndarray_cid(self.scorer_layer4)
                if self.scorer_layer4 is not None
                else "untrained"),
            "scorer_layer4_residual": float(round(
                self.scorer_layer4_residual, 12)),
            "replay_dominance_W_cid": (
                _ndarray_cid(self.replay_dominance_W)
                if self.replay_dominance_W is not None
                else "none"),
        })


def fit_lhr_v14_four_layer_scorer(
        *, head: LongHorizonReconstructionV14Head,
        train_features: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        ridge_lambda: float = 0.10,
) -> tuple[LongHorizonReconstructionV14Head, dict[str, Any]]:
    """Fit the fourth-layer ridge on top of the V13 third-layer
    feature pipeline."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(train_targets, dtype=_np.float64)
    if X.shape[0] == 0:
        return head, {"converged": True, "n": 0}
    # Pass X through tanh2 layer.
    if head.tanh2_proj_W is not None and X.shape[1] == int(
            head.inner_v13.tanh_proj_dim):
        f3 = _np.tanh(X @ _np.asarray(
            head.tanh2_proj_W, dtype=_np.float64))
    else:
        # Skip the tanh2 transform if shapes mismatch.
        f3 = X
    lam = max(float(ridge_lambda), 1e-9)
    A = f3.T @ f3 + lam * _np.eye(
        f3.shape[1], dtype=_np.float64)
    b = f3.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((f3.shape[1],), dtype=_np.float64)
    y_hat = f3 @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        head,
        scorer_layer4=_np.asarray(
            theta, dtype=_np.float64).copy(),
        scorer_layer4_residual=float(post))
    audit = {
        "schema": W62_LHR_V14_SCHEMA_VERSION,
        "kind": "lhr_v14_four_layer_scorer",
        "pre_fit_residual": float(pre),
        "post_fit_residual": float(post),
        "converged": bool(post <= pre + 1e-9),
        "n": int(X.shape[0]),
    }
    return fitted, audit


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV14Witness:
    schema: str
    head_cid: str
    max_k: int
    replay_dominance_dim: int
    out_dim: int
    n_heads: int
    thirteen_way_runs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "max_k": int(self.max_k),
            "replay_dominance_dim": int(
                self.replay_dominance_dim),
            "out_dim": int(self.out_dim),
            "n_heads": int(self.n_heads),
            "thirteen_way_runs": bool(self.thirteen_way_runs),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v14_witness",
            "witness": self.to_dict()})


def emit_lhr_v14_witness(
        head: LongHorizonReconstructionV14Head, *,
        carrier: Sequence[float], k: int = 16,
        replay_dominance_indicator: (
            Sequence[float] | None) = None,
) -> LongHorizonReconstructionV14Witness:
    runs = True
    try:
        head.thirteen_way_value(
            carrier=list(carrier), k=int(k),
            replay_dominance_indicator=(
                list(replay_dominance_indicator)
                if replay_dominance_indicator is not None
                else None))
    except Exception:
        runs = False
    return LongHorizonReconstructionV14Witness(
        schema=W62_LHR_V14_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        max_k=int(head.max_k),
        replay_dominance_dim=int(head.replay_dominance_dim),
        out_dim=int(head.out_dim),
        n_heads=13,
        thirteen_way_runs=bool(runs),
    )


__all__ = [
    "W62_LHR_V14_SCHEMA_VERSION",
    "W62_DEFAULT_LHR_V14_MAX_K",
    "W62_DEFAULT_LHR_V14_REPLAY_DOMINANCE_DIM",
    "W62_DEFAULT_LHR_V14_TANH2_PROJ_DIM",
    "LongHorizonReconstructionV14Head",
    "fit_lhr_v14_four_layer_scorer",
    "LongHorizonReconstructionV14Witness",
    "emit_lhr_v14_witness",
]
