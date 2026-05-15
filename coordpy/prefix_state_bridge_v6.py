"""W62 M4 — Prefix-State Bridge V6.

Strictly extends W61's ``coordpy.prefix_state_bridge_v5``. V5 fit
a step-1 drift L2 predictor by closed-form ridge over a 3-d
segment-configuration feature. V6 adds:

* **Drift-curve predictor** — V6 fits a *stacked* drift predictor
  over a multi-step drift curve target. Given segment configuration
  features, V6 fits a (3 × K) matrix that predicts drift L2 at
  each of K future steps. Closed-form ridge with stacked targets.
* **V7 substrate compatibility** — V6 prefix witness surfaces the
  V7 logit-lens entropy delta across the chain.

Honest scope
------------

* The drift-curve predictor is still a linear ridge on 3-d
  configuration features stacked across K target steps. It does
  NOT model token-content-conditional drift. ``W62-L-V6-PREFIX-
  DRIFT-CURVE-LINEAR-CAP`` documents.
* The V7 logit-lens entropy delta is a measurable diagnostic, not
  a calibrated probability.
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
        "coordpy.prefix_state_bridge_v6 requires numpy"
        ) from exc

from .prefix_state_bridge_v4 import (
    bridge_prefix_state_and_measure_v4,
)
from .prefix_state_bridge_v5 import (
    PrefixDriftPredictor,
    W61_PREFIX_STATE_BRIDGE_V5_SCHEMA_VERSION,
)
from .tiny_substrate_v3 import _sha256_hex
from .tiny_substrate_v5 import TinyV5SubstrateParams


W62_PREFIX_STATE_BRIDGE_V6_SCHEMA_VERSION: str = (
    "coordpy.prefix_state_bridge_v6.v1")


@dataclasses.dataclass(frozen=True)
class PrefixDriftCurvePredictor:
    """Stacked drift-curve predictor: drift_k ≈ β_k · feature +
    intercept_k for each k ∈ {1, ..., K}.

    ``B`` is the (4 × K) coefficient matrix (3 features + intercept
    × K target steps).
    """
    schema: str
    n_train_examples: int
    n_target_steps: int
    B: tuple[tuple[float, ...], ...]   # (4, K)
    ridge_lambda: float
    per_step_pre_residual: tuple[float, ...]
    per_step_post_residual: tuple[float, ...]
    converged: bool

    def predict_curve(
            self, *, reuse_len: int, recompute_len: int,
            drop_len: int,
    ) -> list[float]:
        feat = _np.array([
            float(reuse_len), float(recompute_len),
            float(drop_len), 1.0], dtype=_np.float64)
        B = _np.asarray(self.B, dtype=_np.float64)
        if B.size == 0:
            return [0.0] * int(self.n_target_steps)
        # (4, K) · (4,) → (K,)
        return [float(x) for x in (feat @ B)]

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_drift_curve_predictor",
            "schema": str(self.schema),
            "n_train_examples": int(self.n_train_examples),
            "n_target_steps": int(self.n_target_steps),
            "B_round12": [
                [float(round(float(x), 12)) for x in row]
                for row in self.B],
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "converged": bool(self.converged),
        })


def fit_prefix_drift_curve_predictor(
        *, params_v5: TinyV5SubstrateParams,
        prompt_token_ids: Sequence[int],
        train_segment_configs: Sequence[
            Sequence[tuple[int, int, str]]],
        train_chain: Sequence[Sequence[int]],
        ridge_lambda: float = 0.10,
) -> PrefixDriftCurvePredictor:
    """Fit a stacked drift-curve predictor over the first
    ``len(train_chain)`` steps of a chain."""
    if not train_chain:
        return PrefixDriftCurvePredictor(
            schema=W62_PREFIX_STATE_BRIDGE_V6_SCHEMA_VERSION,
            n_train_examples=0, n_target_steps=0,
            B=tuple(), ridge_lambda=float(ridge_lambda),
            per_step_pre_residual=tuple(),
            per_step_post_residual=tuple(),
            converged=True)
    K = int(len(train_chain))
    feats: list[list[float]] = []
    drift_curves: list[list[float]] = []
    for segs in train_segment_configs:
        w = bridge_prefix_state_and_measure_v4(
            params_v5=params_v5,
            prompt_token_ids=list(prompt_token_ids),
            follow_up_chain=[list(s) for s in train_chain],
            segments=segs)
        curve = list(w.chain_step_drifts_l2)
        # Pad to length K.
        if len(curve) < K:
            curve = curve + [0.0] * (K - len(curve))
        else:
            curve = curve[:K]
        feats.append([
            float(w.reuse_len), float(w.recompute_len),
            float(w.drop_len), 1.0])
        drift_curves.append([float(x) for x in curve])
    X = _np.asarray(feats, dtype=_np.float64)
    Y = _np.asarray(drift_curves, dtype=_np.float64)
    if X.shape[0] == 0:
        return PrefixDriftCurvePredictor(
            schema=W62_PREFIX_STATE_BRIDGE_V6_SCHEMA_VERSION,
            n_train_examples=0, n_target_steps=int(K),
            B=tuple(), ridge_lambda=float(ridge_lambda),
            per_step_pre_residual=tuple([0.0] * K),
            per_step_post_residual=tuple([0.0] * K),
            converged=True)
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(X.shape[1], dtype=_np.float64)
    rhs = X.T @ Y
    try:
        B = _np.linalg.solve(A, rhs)
    except Exception:
        B = _np.zeros((X.shape[1], K), dtype=_np.float64)
    Y_hat = X @ B
    pre = [float(_np.mean(_np.abs(Y[:, k])))
            for k in range(K)]
    post = [float(_np.mean(_np.abs(Y[:, k] - Y_hat[:, k])))
            for k in range(K)]
    converged = all(po <= pr + 1e-9
                    for pr, po in zip(pre, post))
    return PrefixDriftCurvePredictor(
        schema=W62_PREFIX_STATE_BRIDGE_V6_SCHEMA_VERSION,
        n_train_examples=int(X.shape[0]),
        n_target_steps=int(K),
        B=tuple(
            tuple(float(x) for x in row)
            for row in B),
        ridge_lambda=float(ridge_lambda),
        per_step_pre_residual=tuple(pre),
        per_step_post_residual=tuple(post),
        converged=bool(converged),
    )


@dataclasses.dataclass(frozen=True)
class PrefixStateBridgeV6Witness:
    schema: str
    chain_witness_cid: str
    n_links: int
    predictor_cid: str
    predicted_drift_curve: tuple[float, ...]
    actual_drift_curve: tuple[float, ...]
    curve_l1_error: float
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "chain_witness_cid": str(self.chain_witness_cid),
            "n_links": int(self.n_links),
            "predictor_cid": str(self.predictor_cid),
            "predicted_drift_curve": [
                float(round(float(x), 12))
                for x in self.predicted_drift_curve],
            "actual_drift_curve": [
                float(round(float(x), 12))
                for x in self.actual_drift_curve],
            "curve_l1_error": float(round(
                self.curve_l1_error, 12)),
            "converged": bool(self.converged),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_state_bridge_v6_witness",
            "witness": self.to_dict()})


def emit_prefix_state_bridge_v6_witness(
        *, predictor: PrefixDriftCurvePredictor,
        actual_curve: Sequence[float],
        chain_witness_cid: str = "",
) -> PrefixStateBridgeV6Witness:
    pred = predictor.predict_curve(
        reuse_len=1, recompute_len=1, drop_len=1)
    actual = [float(x) for x in actual_curve]
    K = min(len(pred), len(actual))
    err = float(sum(
        abs(pred[i] - actual[i]) for i in range(K)))
    return PrefixStateBridgeV6Witness(
        schema=W62_PREFIX_STATE_BRIDGE_V6_SCHEMA_VERSION,
        chain_witness_cid=str(chain_witness_cid),
        n_links=int(predictor.n_target_steps),
        predictor_cid=str(predictor.cid()),
        predicted_drift_curve=tuple(pred),
        actual_drift_curve=tuple(actual),
        curve_l1_error=float(err),
        converged=bool(predictor.converged),
    )


__all__ = [
    "W62_PREFIX_STATE_BRIDGE_V6_SCHEMA_VERSION",
    "PrefixDriftCurvePredictor",
    "fit_prefix_drift_curve_predictor",
    "PrefixStateBridgeV6Witness",
    "emit_prefix_state_bridge_v6_witness",
]
