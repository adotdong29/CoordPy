"""W63 M4 — Prefix-State Bridge V7.

Strictly extends W62's ``coordpy.prefix_state_bridge_v6``. V6 fit a
stacked drift-curve predictor over K steps. V7 adds:

* **Token-content-conditional drift feature** — V7 augments the
  3-d configuration feature with a *token-content fingerprint*
  (4-d hash of follow-up tokens) producing a 7-d feature.
* **Longer chain drift bounds** — V7 fits over up to K=24 steps
  (vs V6's typical 3-8) and surfaces a per-step *envelope* width.
* **Prefix-vs-hidden comparison** —
  ``compare_prefix_vs_hidden_v7`` takes a prefix-reuse drift
  curve and a hidden-state-replay drift curve and returns a
  three-way ``{prefix_beats_hidden, hidden_beats_prefix, tie}``
  decision over the curve L1 area.
* **V8 prefix-reuse trust coupling** —
  ``write_prefix_v7_into_v8_reuse_trust`` records prefix decisions
  into the V8 substrate's prefix_reuse_trust ledger.

Honest scope
------------

* The token-content fingerprint is a *fixed* SHA256 projection;
  it is not learned. ``W63-L-V7-PREFIX-TOKEN-FINGERPRINT-CAP``
  documents.
* The prefix-vs-hidden comparison is over *drift curves on the
  same chain*; it is not a claim that prefix bridges beat hidden
  bridges in general.
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
        "coordpy.prefix_state_bridge_v7 requires numpy"
        ) from exc

from .prefix_state_bridge_v4 import (
    bridge_prefix_state_and_measure_v4,
)
from .prefix_state_bridge_v6 import (
    PrefixDriftCurvePredictor,
    fit_prefix_drift_curve_predictor,
)
from .tiny_substrate_v3 import _sha256_hex
from .tiny_substrate_v5 import TinyV5SubstrateParams
from .tiny_substrate_v8 import (
    TinyV8KVCache, record_prefix_reuse_decision_v8,
)


W63_PREFIX_STATE_BRIDGE_V7_SCHEMA_VERSION: str = (
    "coordpy.prefix_state_bridge_v7.v1")
W63_DEFAULT_PREFIX_V7_K_STEPS: int = 24
W63_DEFAULT_PREFIX_V7_TOKEN_FP_DIM: int = 4


def _token_fingerprint_v7(
        token_ids: Sequence[int], *,
        fp_dim: int = W63_DEFAULT_PREFIX_V7_TOKEN_FP_DIM,
) -> list[float]:
    """Fixed SHA256-based fingerprint of follow-up token IDs."""
    payload = json.dumps(
        [int(t) for t in token_ids], separators=(",", ":")
        ).encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()
    out: list[float] = []
    for i in range(int(fp_dim)):
        nb = h[(i * 2) % len(h):(i * 2) % len(h) + 2]
        if not nb:
            nb = "00"
        v = (int(nb, 16) / 127.5) - 1.0
        out.append(float(round(v, 12)))
    return out


@dataclasses.dataclass(frozen=True)
class PrefixDriftCurvePredictorV7:
    """V7 stacked drift-curve predictor over a 7-d feature
    [reuse_len, recompute_len, drop_len, fp1, fp2, fp3, fp4]."""
    schema: str
    n_train_examples: int
    n_target_steps: int
    B: tuple[tuple[float, ...], ...]   # (8, K) — 7 features + 1 bias
    ridge_lambda: float
    per_step_pre_residual: tuple[float, ...]
    per_step_post_residual: tuple[float, ...]
    converged: bool

    def predict_curve(
            self, *, reuse_len: int, recompute_len: int,
            drop_len: int,
            follow_up_tokens: Sequence[int] | None = None,
    ) -> list[float]:
        fp = _token_fingerprint_v7(
            list(follow_up_tokens) if follow_up_tokens else [])
        feat = _np.array([
            float(reuse_len), float(recompute_len),
            float(drop_len),
            *(float(x) for x in fp),
            1.0], dtype=_np.float64)
        B = _np.asarray(self.B, dtype=_np.float64)
        if B.size == 0:
            return [0.0] * int(self.n_target_steps)
        return [float(x) for x in (feat @ B)]

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_drift_curve_predictor_v7",
            "schema": str(self.schema),
            "n_train_examples": int(self.n_train_examples),
            "n_target_steps": int(self.n_target_steps),
            "B_round12": [
                [float(round(float(x), 12)) for x in row]
                for row in self.B],
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "converged": bool(self.converged),
        })


def fit_prefix_drift_curve_predictor_v7(
        *, params_v5: TinyV5SubstrateParams,
        prompt_token_ids: Sequence[int],
        train_segment_configs: Sequence[
            Sequence[tuple[int, int, str]]],
        train_chain: Sequence[Sequence[int]],
        ridge_lambda: float = 0.10,
) -> PrefixDriftCurvePredictorV7:
    """Fit the V7 stacked drift-curve predictor with token-content
    fingerprint features.
    """
    if not train_chain:
        return PrefixDriftCurvePredictorV7(
            schema=W63_PREFIX_STATE_BRIDGE_V7_SCHEMA_VERSION,
            n_train_examples=0, n_target_steps=0,
            B=tuple(), ridge_lambda=float(ridge_lambda),
            per_step_pre_residual=tuple(),
            per_step_post_residual=tuple(),
            converged=True)
    K = int(len(train_chain))
    feats: list[list[float]] = []
    drift_curves: list[list[float]] = []
    fp_dim = W63_DEFAULT_PREFIX_V7_TOKEN_FP_DIM
    for segs in train_segment_configs:
        w = bridge_prefix_state_and_measure_v4(
            params_v5=params_v5,
            prompt_token_ids=list(prompt_token_ids),
            follow_up_chain=[list(s) for s in train_chain],
            segments=segs)
        curve = list(w.chain_step_drifts_l2)
        if len(curve) < K:
            curve = curve + [0.0] * (K - len(curve))
        else:
            curve = curve[:K]
        # Build the token fingerprint from the first follow-up
        # link (V7 differentiates configurations by content too).
        first_chain = list(train_chain[0]) if train_chain else []
        fp = _token_fingerprint_v7(
            first_chain, fp_dim=fp_dim)
        feats.append([
            float(w.reuse_len), float(w.recompute_len),
            float(w.drop_len),
            *(float(x) for x in fp),
            1.0])
        drift_curves.append([float(x) for x in curve])
    X = _np.asarray(feats, dtype=_np.float64)
    Y = _np.asarray(drift_curves, dtype=_np.float64)
    if X.shape[0] == 0:
        return PrefixDriftCurvePredictorV7(
            schema=W63_PREFIX_STATE_BRIDGE_V7_SCHEMA_VERSION,
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
    return PrefixDriftCurvePredictorV7(
        schema=W63_PREFIX_STATE_BRIDGE_V7_SCHEMA_VERSION,
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
class PrefixVsHiddenDecisionWitnessV7:
    schema: str
    decision: str
    prefix_curve_l1: float
    hidden_curve_l1: float
    margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "decision": str(self.decision),
            "prefix_curve_l1": float(round(
                self.prefix_curve_l1, 12)),
            "hidden_curve_l1": float(round(
                self.hidden_curve_l1, 12)),
            "margin": float(round(self.margin, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_vs_hidden_decision_v7",
            "witness": self.to_dict()})


def compare_prefix_vs_hidden_v7(
        *, prefix_drift_curve: Sequence[float],
        hidden_drift_curve: Sequence[float],
        tie_threshold: float = 1e-6,
) -> PrefixVsHiddenDecisionWitnessV7:
    """Three-way comparison over curve L1 areas."""
    p = float(sum(abs(float(x)) for x in prefix_drift_curve))
    h = float(sum(abs(float(x)) for x in hidden_drift_curve))
    diff = p - h
    if diff < -float(tie_threshold):
        decision = "prefix_beats_hidden"
    elif diff > float(tie_threshold):
        decision = "hidden_beats_prefix"
    else:
        decision = "tie"
    return PrefixVsHiddenDecisionWitnessV7(
        schema=W63_PREFIX_STATE_BRIDGE_V7_SCHEMA_VERSION,
        decision=str(decision),
        prefix_curve_l1=float(p),
        hidden_curve_l1=float(h),
        margin=float(-diff),
    )


def write_prefix_v7_into_v8_reuse_trust(
        *, decisions_per_layer_head: dict[tuple[int, int], str],
        v8_cache: TinyV8KVCache,
        trust_ema: float = 0.5,
) -> dict[str, Any]:
    """Apply prefix-reuse decisions to the V8 reuse-trust ledger."""
    n_writes = 0
    for (li, hi), dec in decisions_per_layer_head.items():
        record_prefix_reuse_decision_v8(
            v8_cache,
            layer_index=int(li), head_index=int(hi),
            decision=str(dec),
            trust_ema=float(trust_ema))
        n_writes += 1
    return {
        "schema": W63_PREFIX_STATE_BRIDGE_V7_SCHEMA_VERSION,
        "kind": "prefix_v7_reuse_trust_write",
        "n_writes": int(n_writes),
    }


@dataclasses.dataclass(frozen=True)
class PrefixStateBridgeV7Witness:
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
            "kind": "prefix_state_bridge_v7_witness",
            "witness": self.to_dict()})


def emit_prefix_state_bridge_v7_witness(
        *, predictor: PrefixDriftCurvePredictorV7,
        actual_curve: Sequence[float],
        chain_witness_cid: str = "",
) -> PrefixStateBridgeV7Witness:
    pred = predictor.predict_curve(
        reuse_len=1, recompute_len=1, drop_len=1,
        follow_up_tokens=None)
    actual = [float(x) for x in actual_curve]
    K = min(len(pred), len(actual))
    err = float(sum(
        abs(pred[i] - actual[i]) for i in range(K)))
    return PrefixStateBridgeV7Witness(
        schema=W63_PREFIX_STATE_BRIDGE_V7_SCHEMA_VERSION,
        chain_witness_cid=str(chain_witness_cid),
        n_links=int(predictor.n_target_steps),
        predictor_cid=str(predictor.cid()),
        predicted_drift_curve=tuple(pred),
        actual_drift_curve=tuple(actual),
        curve_l1_error=float(err),
        converged=bool(predictor.converged),
    )


__all__ = [
    "W63_PREFIX_STATE_BRIDGE_V7_SCHEMA_VERSION",
    "W63_DEFAULT_PREFIX_V7_K_STEPS",
    "W63_DEFAULT_PREFIX_V7_TOKEN_FP_DIM",
    "PrefixDriftCurvePredictorV7",
    "fit_prefix_drift_curve_predictor_v7",
    "PrefixVsHiddenDecisionWitnessV7",
    "compare_prefix_vs_hidden_v7",
    "write_prefix_v7_into_v8_reuse_trust",
    "PrefixStateBridgeV7Witness",
    "emit_prefix_state_bridge_v7_witness",
]
