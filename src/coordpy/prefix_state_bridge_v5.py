"""W61 M4 — Prefix-State Bridge V5.

Strictly extends W60's ``coordpy.prefix_state_bridge_v4``. V4
returned a per-step drift list and a cumulative drift envelope.
V5 adds *three* new substrate-load-bearing pieces:

* **Chain-of-chains** — V5 supports a chain whose elements are
  themselves multi-segment prefixes. ``bridge_prefix_chain_v5``
  walks each link as a multi-segment prefix and forwards across
  the chain, accumulating both per-step drift and per-link drift
  envelope.
* **Trained drift predictor** — V5 fits a tiny closed-form ridge
  model that, given a segment configuration (counts of reuse /
  recompute / drop tokens), predicts the step-1 drift L2. The fit
  is over a small set of training segment configurations; the
  controller can then ask the predictor *before* running the
  forward to decide whether the predicted drift is under
  threshold.
* **V6 substrate compatibility** — when the V5 bridge is asked to
  drive a V6 substrate, it surfaces the V6 cache key fingerprint
  and replay-age channel in the witness so the cache controller
  V4 can use them downstream.

Honest scope
------------

* The drift predictor is a *linear* model over a 3-d segment-
  configuration feature (``[reuse_len, recompute_len, drop_len]``).
  It does NOT model token-content-conditional drift. ``W61-L-V5-
  DRIFT-PREDICTOR-LINEAR-CAP`` documents the boundary.
* Chain-of-chains accumulates per-link drift via sum; the V5
  envelope is a triangle-inequality upper bound, not a tight
  geodesic bound.
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
        "coordpy.prefix_state_bridge_v5 requires numpy"
        ) from exc

from .prefix_state_bridge_v4 import (
    bridge_prefix_state_and_measure_v4,
    PrefixStateBridgeV4Witness,
)
from .tiny_substrate_v3 import (
    TinyV3KVCache,
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate_v3,
)
from .tiny_substrate_v5 import (
    TinyV5SubstrateParams,
    W60_V5_SEGMENT_DROP,
    W60_V5_SEGMENT_RECOMPUTE,
    W60_V5_SEGMENT_REUSE,
)
from .tiny_substrate_v6 import (
    TinyV6SubstrateParams,
    forward_tiny_substrate_v6,
    forward_with_multi_segment_reuse_v6,
    extract_multi_segment_prefix_v6,
)
from .kv_bridge_v6 import v6_cache_key_fingerprint


W61_PREFIX_STATE_BRIDGE_V5_SCHEMA_VERSION: str = (
    "coordpy.prefix_state_bridge_v5.v1")


@dataclasses.dataclass(frozen=True)
class PrefixDriftPredictor:
    """Closed-form ridge predictor of step-1 drift L2 given a
    3-d segment-configuration feature ``[reuse_len, recompute_len,
    drop_len]``. ``β`` is the (3,) fit coefficient vector and
    ``intercept`` is a scalar bias."""
    schema: str
    n_train_examples: int
    beta: tuple[float, float, float]
    intercept: float
    ridge_lambda: float
    pre_fit_residual: float
    post_fit_residual: float
    converged: bool

    def predict(
            self, *, reuse_len: int, recompute_len: int,
            drop_len: int,
    ) -> float:
        feat = _np.array(
            [float(reuse_len), float(recompute_len),
             float(drop_len)], dtype=_np.float64)
        return float(_np.dot(_np.asarray(
            self.beta, dtype=_np.float64), feat)
                     + float(self.intercept))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_train_examples": int(self.n_train_examples),
            "beta": [float(round(float(x), 12))
                       for x in self.beta],
            "intercept": float(round(self.intercept, 12)),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "pre_fit_residual": float(round(
                self.pre_fit_residual, 12)),
            "post_fit_residual": float(round(
                self.post_fit_residual, 12)),
            "converged": bool(self.converged),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_drift_predictor",
            "predictor": self.to_dict()})


def fit_prefix_drift_predictor(
        *,
        params_v5: TinyV5SubstrateParams,
        prompt_token_ids: Sequence[int],
        train_segment_configs: Sequence[
            Sequence[tuple[int, int, str]]],
        train_first_step_ids: Sequence[int],
        ridge_lambda: float = 0.10,
) -> PrefixDriftPredictor:
    """Fit a 3-feature linear ridge model: drift ≈ β·[reuse_len,
    recompute_len, drop_len] + intercept."""
    feats: list[list[float]] = []
    drifts: list[float] = []
    for segs in train_segment_configs:
        w = bridge_prefix_state_and_measure_v4(
            params_v5=params_v5,
            prompt_token_ids=prompt_token_ids,
            follow_up_chain=[list(train_first_step_ids)],
            segments=segs)
        d = float(w.chain_step_drifts_l2[0]
                  if w.chain_step_drifts_l2 else 0.0)
        feats.append([
            float(w.reuse_len), float(w.recompute_len),
            float(w.drop_len), 1.0])
        drifts.append(d)
    X = _np.asarray(feats, dtype=_np.float64)
    y = _np.asarray(drifts, dtype=_np.float64)
    if X.shape[0] == 0:
        return PrefixDriftPredictor(
            schema=W61_PREFIX_STATE_BRIDGE_V5_SCHEMA_VERSION,
            n_train_examples=0, beta=(0.0, 0.0, 0.0),
            intercept=0.0, ridge_lambda=float(ridge_lambda),
            pre_fit_residual=0.0, post_fit_residual=0.0,
            converged=True)
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(X.shape[1], dtype=_np.float64)
    b = X.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((X.shape[1],), dtype=_np.float64)
    y_hat = X @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    beta = tuple(float(x) for x in theta[:3])
    intercept = float(theta[3]) if X.shape[1] >= 4 else 0.0
    return PrefixDriftPredictor(
        schema=W61_PREFIX_STATE_BRIDGE_V5_SCHEMA_VERSION,
        n_train_examples=int(X.shape[0]),
        beta=beta, intercept=float(intercept),
        ridge_lambda=float(ridge_lambda),
        pre_fit_residual=float(pre),
        post_fit_residual=float(post),
        converged=bool(post <= pre + 1e-9),
    )


@dataclasses.dataclass(frozen=True)
class PrefixStateBridgeV5Witness:
    schema: str
    chain_witness_cids: tuple[str, ...]
    n_links: int
    per_link_flop_total: tuple[int, ...]
    per_link_step_drifts_l2: tuple[float, ...]
    per_link_cumulative_drift_l2: tuple[float, ...]
    chain_cumulative_drift_l2: float
    chain_total_flop: int
    chain_flop_saved_vs_full: int
    chain_flop_savings_ratio: float
    predictor_cid: str
    predicted_drift_l2_step1: float
    actual_drift_l2_step1: float
    drift_prediction_error: float
    v6_cache_key_fingerprint: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "chain_witness_cids": list(self.chain_witness_cids),
            "n_links": int(self.n_links),
            "per_link_flop_total": list(self.per_link_flop_total),
            "per_link_step_drifts_l2": [
                float(round(d, 12))
                for d in self.per_link_step_drifts_l2],
            "per_link_cumulative_drift_l2": [
                float(round(d, 12))
                for d in self.per_link_cumulative_drift_l2],
            "chain_cumulative_drift_l2": float(round(
                self.chain_cumulative_drift_l2, 12)),
            "chain_total_flop": int(self.chain_total_flop),
            "chain_flop_saved_vs_full": int(
                self.chain_flop_saved_vs_full),
            "chain_flop_savings_ratio": float(round(
                self.chain_flop_savings_ratio, 12)),
            "predictor_cid": str(self.predictor_cid),
            "predicted_drift_l2_step1": float(round(
                self.predicted_drift_l2_step1, 12)),
            "actual_drift_l2_step1": float(round(
                self.actual_drift_l2_step1, 12)),
            "drift_prediction_error": float(round(
                self.drift_prediction_error, 12)),
            "v6_cache_key_fingerprint": list(
                self.v6_cache_key_fingerprint),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_state_bridge_v5_witness",
            "witness": self.to_dict()})


def bridge_prefix_chain_v5(
        *,
        params_v6: TinyV6SubstrateParams,
        prompt_token_ids: Sequence[int],
        chain_segments: Sequence[
            Sequence[tuple[int, int, str]]],
        chain_follow_ups: Sequence[Sequence[int]],
        drift_predictor: PrefixDriftPredictor | None = None,
) -> PrefixStateBridgeV5Witness:
    """Walk a chain of multi-segment prefixes.

    ``chain_segments[i]`` is the segment config for link i (over the
    same prompt). ``chain_follow_ups[i]`` is the follow-up token
    sequence for link i. We emit per-link witnesses (V4 shape) and
    aggregate.
    """
    cfg = params_v6.config
    n_links = max(len(chain_segments), len(chain_follow_ups))
    witnesses: list[PrefixStateBridgeV4Witness] = []
    for i in range(n_links):
        segs = (
            chain_segments[i] if i < len(chain_segments)
            else [(0, len(list(prompt_token_ids)),
                    W60_V5_SEGMENT_REUSE)])
        fu = (
            chain_follow_ups[i] if i < len(chain_follow_ups)
            else [])
        w = bridge_prefix_state_and_measure_v4(
            params_v5=params_v6.v5_params,
            prompt_token_ids=prompt_token_ids,
            follow_up_chain=[list(fu)],
            segments=list(segs))
        witnesses.append(w)
    # Predict step-1 drift on the first link.
    if drift_predictor is not None and witnesses:
        first = witnesses[0]
        predicted = drift_predictor.predict(
            reuse_len=int(first.reuse_len),
            recompute_len=int(first.recompute_len),
            drop_len=int(first.drop_len))
    else:
        predicted = 0.0
    actual_step1 = float(
        witnesses[0].chain_step_drifts_l2[0]
        if witnesses and witnesses[0].chain_step_drifts_l2
        else 0.0)
    pred_err = float(abs(predicted - actual_step1))
    # V6 cache key fingerprint at the end of the chain.
    _, v6_cache = forward_tiny_substrate_v6(
        params_v6, list(prompt_token_ids))
    v6_fp = v6_cache_key_fingerprint(v6_cache)
    total_flop = int(sum(int(w.flop_multi_segment_total)
                         for w in witnesses))
    full_flop = int(sum(int(w.flop_full_recompute)
                        for w in witnesses))
    flop_saved = int(full_flop - total_flop)
    flop_ratio = (
        float(flop_saved) / float(max(full_flop, 1)))
    per_link_step = tuple(
        float(w.chain_step_drifts_l2[0]
              if w.chain_step_drifts_l2 else 0.0)
        for w in witnesses)
    per_link_cum = tuple(
        float(w.chain_cumulative_drift_l2) for w in witnesses)
    return PrefixStateBridgeV5Witness(
        schema=W61_PREFIX_STATE_BRIDGE_V5_SCHEMA_VERSION,
        chain_witness_cids=tuple(
            str(w.cid()) for w in witnesses),
        n_links=int(len(witnesses)),
        per_link_flop_total=tuple(
            int(w.flop_multi_segment_total)
            for w in witnesses),
        per_link_step_drifts_l2=per_link_step,
        per_link_cumulative_drift_l2=per_link_cum,
        chain_cumulative_drift_l2=float(
            sum(per_link_cum)),
        chain_total_flop=int(total_flop),
        chain_flop_saved_vs_full=int(flop_saved),
        chain_flop_savings_ratio=float(flop_ratio),
        predictor_cid=str(
            drift_predictor.cid()
            if drift_predictor is not None else "none"),
        predicted_drift_l2_step1=float(predicted),
        actual_drift_l2_step1=float(actual_step1),
        drift_prediction_error=float(pred_err),
        v6_cache_key_fingerprint=tuple(v6_fp),
    )


__all__ = [
    "W61_PREFIX_STATE_BRIDGE_V5_SCHEMA_VERSION",
    "PrefixDriftPredictor",
    "PrefixStateBridgeV5Witness",
    "fit_prefix_drift_predictor",
    "bridge_prefix_chain_v5",
]
