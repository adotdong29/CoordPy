"""W61 M14 — Long-Horizon Retention V13.

Strictly extends W60's ``coordpy.long_horizon_retention_v12``. V13
adds a *twelfth* head — **attention-pattern-conditioned
reconstruction** — and raises ``max_k`` to **128** (vs V12's 96).
The new head consumes the attention-steering-V5 attention pattern
top-K positions (as a sparse indicator vector) and projects it
into the reconstruction output dimension.

V13 also extends V12's two-layer retention scorer: V12's first
layer was random + frozen (ReLU activation), V13 adds an explicit
**three-layer scorer** (random+ReLU → random+tanh → ridge over the
post-tanh features). Still no autograd: the first two layers are
fixed projections.

V13 strictly extends V12: when ``attention_pattern_top_k = None``
and ``three_layer_scorer = False``, V13's twelve-way value reduces
to V12 byte-for-byte.

Honest scope
------------

* The three-layer scorer fits *only* the final ridge head; the
  first two layers are random + frozen.
  ``W61-L-V13-LHR-SCORER-FIT-CAP`` documents.
* The new head outputs ``out_dim`` floats; the attention top-K
  vector is mean-pooled, projected, then mixed.
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
        "coordpy.long_horizon_retention_v13 requires numpy"
        ) from exc

from .long_horizon_retention_v12 import (
    LongHorizonReconstructionV12Head,
    W60_DEFAULT_LHR_V12_SCORER_RIDGE_LAMBDA,
    _ndarray_cid,
)


W61_LHR_V13_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v13.v1")
W61_DEFAULT_LHR_V13_MAX_K: int = 128
W61_DEFAULT_LHR_V13_ATTENTION_PATTERN_DIM: int = 32
W61_DEFAULT_LHR_V13_TANH_PROJ_DIM: int = 24


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class LongHorizonReconstructionV13Head:
    inner_v12: LongHorizonReconstructionV12Head
    max_k: int
    attention_pattern_dim: int
    tanh_proj_dim: int
    tanh_proj_W: "_np.ndarray | None" = None
    scorer_layer3: "_np.ndarray | None" = None
    scorer_layer3_residual: float = 0.0
    attention_pattern_W: "_np.ndarray | None" = None

    @classmethod
    def init(
            cls, *,
            max_k: int = W61_DEFAULT_LHR_V13_MAX_K,
            attention_pattern_dim: int = (
                W61_DEFAULT_LHR_V13_ATTENTION_PATTERN_DIM),
            tanh_proj_dim: int = (
                W61_DEFAULT_LHR_V13_TANH_PROJ_DIM),
            seed: int = 61130,
    ) -> "LongHorizonReconstructionV13Head":
        v12 = LongHorizonReconstructionV12Head.init(
            max_k=W61_DEFAULT_LHR_V13_MAX_K,
            seed=int(seed))
        rng = _np.random.default_rng(int(seed) ^ 0xACED_61)
        out_dim = int(v12.out_dim)
        tanh_W = rng.standard_normal(
            (int(v12.hidden_proj_dim),
             int(tanh_proj_dim))) * 0.10
        ap_W = rng.standard_normal(
            (int(attention_pattern_dim),
             int(out_dim))) * 0.08
        return cls(
            inner_v12=v12,
            max_k=int(max_k),
            attention_pattern_dim=int(attention_pattern_dim),
            tanh_proj_dim=int(tanh_proj_dim),
            tanh_proj_W=tanh_W.astype(_np.float64),
            attention_pattern_W=ap_W.astype(_np.float64),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v12.out_dim)

    def attention_pattern_value(
            self, *, attention_top_k_indicator: Sequence[float],
    ) -> "_np.ndarray":
        """Project the attention-pattern top-K indicator vector
        into the reconstruction output dimension."""
        v = _np.asarray(
            attention_top_k_indicator, dtype=_np.float64)
        if v.size < int(self.attention_pattern_dim):
            v = _np.concatenate(
                [v, _np.zeros(
                    int(self.attention_pattern_dim) - v.size,
                    dtype=_np.float64)])
        elif v.size > int(self.attention_pattern_dim):
            v = v[:int(self.attention_pattern_dim)]
        return v @ self.attention_pattern_W

    def twelve_way_value(
            self, *, carrier: Sequence[float], k: int,
            replay_state: Sequence[float] | None = None,
            retrieval_state: Sequence[float] | None = None,
            attention_state: Sequence[float] | None = None,
            hidden_state: Sequence[float] | None = None,
            attention_top_k_indicator: (
                Sequence[float] | None) = None,
            substrate_state: Sequence[float] | None = None,
    ) -> "_np.ndarray":
        # 11-way value from V12.
        v11 = self.inner_v12.replay_conditioned_value(
            carrier=list(carrier), k=int(k),
            replay_state=(
                list(replay_state)
                if replay_state is not None else None),
            retrieval_state=(
                list(retrieval_state)
                if retrieval_state is not None else None),
            attention_state=(
                list(attention_state)
                if attention_state is not None else None),
            hidden_state=(
                list(hidden_state)
                if hidden_state is not None else None),
            substrate_state=(
                list(substrate_state)
                if substrate_state is not None else None),
        )
        v11 = _np.asarray(v11, dtype=_np.float64)
        # Add attention-pattern head contribution.
        if attention_top_k_indicator is not None:
            ap = self.attention_pattern_value(
                attention_top_k_indicator=(
                    list(attention_top_k_indicator)))
            return v11 + 0.10 * ap
        return v11

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W61_LHR_V13_SCHEMA_VERSION,
            "kind": "lhr_v13_head",
            "inner_v12_cid": _sha256_hex({
                "kind": "lhr_v12_head_proxy",
                "inner_v11": "ref"}),
            "max_k": int(self.max_k),
            "attention_pattern_dim": int(
                self.attention_pattern_dim),
            "tanh_proj_dim": int(self.tanh_proj_dim),
            "tanh_proj_W_cid": (
                _ndarray_cid(self.tanh_proj_W)
                if self.tanh_proj_W is not None else "u"),
            "scorer_layer3_cid": (
                _ndarray_cid(self.scorer_layer3)
                if self.scorer_layer3 is not None else "u"),
            "scorer_layer3_residual": float(round(
                self.scorer_layer3_residual, 12)),
            "attention_pattern_W_cid": (
                _ndarray_cid(self.attention_pattern_W)
                if self.attention_pattern_W is not None else "u"),
        })


def fit_lhr_v13_three_layer_scorer(
        head: LongHorizonReconstructionV13Head,
        *, train_carriers: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        ridge_lambda: float = (
            W60_DEFAULT_LHR_V12_SCORER_RIDGE_LAMBDA),
) -> tuple[LongHorizonReconstructionV13Head, float]:
    """Fit the third-layer scorer by closed-form ridge over the
    post-tanh features.

    Layer 1: V12's random + frozen hidden_proj_W (ReLU).
    Layer 2: V13's random + frozen tanh_proj_W (tanh).
    Layer 3: V13's ridge fit (linear).
    """
    n = len(train_carriers)
    if n == 0 or len(train_targets) != n:
        return head, 0.0
    inner_v12 = head.inner_v12
    if inner_v12.hidden_proj_W is None:
        # Default random projection.
        d_in = max(1, len(list(train_carriers[0])))
        rng = _np.random.default_rng(
            int(inner_v12.inner_v11.cid().__hash__()
                if hasattr(inner_v12.inner_v11, "cid")
                else 0) % (1 << 30))
        inner_v12.hidden_proj_W = rng.standard_normal(
            (d_in, int(inner_v12.hidden_proj_dim))) * 0.1
    X = _np.zeros((n, int(head.tanh_proj_dim)),
                   dtype=_np.float64)
    y = _np.asarray(train_targets, dtype=_np.float64)
    for i, c in enumerate(train_carriers):
        c_arr = _np.asarray(c, dtype=_np.float64)
        if c_arr.size < inner_v12.hidden_proj_W.shape[0]:
            c_arr = _np.concatenate(
                [c_arr, _np.zeros(
                    inner_v12.hidden_proj_W.shape[0]
                    - c_arr.size, dtype=_np.float64)])
        else:
            c_arr = c_arr[:inner_v12.hidden_proj_W.shape[0]]
        h = _np.maximum(c_arr @ inner_v12.hidden_proj_W, 0.0)
        h2 = _np.tanh(h @ head.tanh_proj_W)
        X[i] = h2
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(
        head.tanh_proj_dim, dtype=_np.float64)
    b = X.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros(
            (head.tanh_proj_dim,), dtype=_np.float64)
    head.scorer_layer3 = theta
    y_hat = X @ theta
    residual = float(_np.mean(_np.abs(y - y_hat)))
    head.scorer_layer3_residual = float(residual)
    return head, float(residual)


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV13Witness:
    schema: str
    head_cid: str
    max_k: int
    attention_pattern_dim: int
    tanh_proj_dim: int
    out_dim: int
    n_heads: int
    twelve_way_runs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "max_k": int(self.max_k),
            "attention_pattern_dim": int(
                self.attention_pattern_dim),
            "tanh_proj_dim": int(self.tanh_proj_dim),
            "out_dim": int(self.out_dim),
            "n_heads": int(self.n_heads),
            "twelve_way_runs": bool(self.twelve_way_runs),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v13_witness",
            "witness": self.to_dict()})


def emit_lhr_v13_witness(
        head: LongHorizonReconstructionV13Head, *,
        carrier: Sequence[float], k: int = 16,
        attention_top_k_indicator: (
            Sequence[float] | None) = None,
) -> LongHorizonReconstructionV13Witness:
    runs = True
    try:
        head.twelve_way_value(
            carrier=list(carrier), k=int(k),
            attention_top_k_indicator=(
                list(attention_top_k_indicator)
                if attention_top_k_indicator is not None
                else None))
    except Exception:
        runs = False
    return LongHorizonReconstructionV13Witness(
        schema=W61_LHR_V13_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        max_k=int(head.max_k),
        attention_pattern_dim=int(head.attention_pattern_dim),
        tanh_proj_dim=int(head.tanh_proj_dim),
        out_dim=int(head.out_dim),
        n_heads=12,
        twelve_way_runs=bool(runs),
    )


__all__ = [
    "W61_LHR_V13_SCHEMA_VERSION",
    "W61_DEFAULT_LHR_V13_MAX_K",
    "W61_DEFAULT_LHR_V13_ATTENTION_PATTERN_DIM",
    "W61_DEFAULT_LHR_V13_TANH_PROJ_DIM",
    "LongHorizonReconstructionV13Head",
    "LongHorizonReconstructionV13Witness",
    "fit_lhr_v13_three_layer_scorer",
    "emit_lhr_v13_witness",
]
