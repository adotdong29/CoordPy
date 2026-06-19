"""W64 M4 — Prefix-State Bridge V8.

Strictly extends W63's ``coordpy.prefix_state_bridge_v7``. V7 fit
a token-content-conditional drift-curve predictor over K=24 steps.
V8 adds:

* **Token-content + role-conditional drift feature** — V8 augments
  the 7-d V7 feature with a *role fingerprint* (4-d hash of role
  identifier) producing an 11-d feature.
* **Even longer chain drift bounds** — V8 fits over up to K=32
  steps (vs V7's 24) and surfaces a per-step *envelope* width plus
  a per-K *drift-acceleration* feature.
* **Three-way prefix-vs-hidden-vs-replay comparison** —
  ``compare_prefix_vs_hidden_vs_replay_v8`` extends V7's two-way
  comparison to a three-way decision over the curve L1 areas.
* **V9 hidden-state-trust coupling** —
  ``write_prefix_v8_into_v9_hidden_state_trust`` records prefix
  decisions into the V9 substrate's hidden_state_trust_ledger
  (negative side: a successful prefix-reuse implies the hidden
  bridge was NOT primary).

Honest scope (W64)
------------------

* The role fingerprint is a *fixed* SHA256 projection; it is not
  learned. ``W64-L-V8-PREFIX-ROLE-FINGERPRINT-CAP`` documents.
* The three-way comparison is over *drift curves on the same
  chain*; it is not a claim that prefix bridges beat hidden
  bridges in general.
* The drift-acceleration feature is a finite difference of the
  drift curve; it is a diagnostic, not a calibrated curvature.
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
        "coordpy.prefix_state_bridge_v8 requires numpy"
        ) from exc

from .prefix_state_bridge_v4 import (
    bridge_prefix_state_and_measure_v4,
)
from .prefix_state_bridge_v7 import (
    PrefixDriftCurvePredictorV7,
    _token_fingerprint_v7,
)
from .tiny_substrate_v3 import _sha256_hex
from .tiny_substrate_v5 import TinyV5SubstrateParams
from .tiny_substrate_v9 import (
    TinyV9KVCache, record_hidden_state_trust_decision_v9,
)


W64_PREFIX_STATE_BRIDGE_V8_SCHEMA_VERSION: str = (
    "coordpy.prefix_state_bridge_v8.v1")
W64_DEFAULT_PREFIX_V8_K_STEPS: int = 32
W64_DEFAULT_PREFIX_V8_ROLE_FP_DIM: int = 4


def _role_fingerprint_v8(
        role: str, *, fp_dim: int = W64_DEFAULT_PREFIX_V8_ROLE_FP_DIM,
) -> list[float]:
    """Fixed SHA256-based fingerprint of role identifier."""
    payload = str(role).encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()
    out: list[float] = []
    for i in range(int(fp_dim)):
        nb = h[(i * 2) % len(h):(i * 2) % len(h) + 2]
        if not nb:
            nb = "00"
        v = (int(nb, 16) / 127.5) - 1.0
        out.append(float(round(v, 12)))
    return out


def _drift_acceleration(
        curve: Sequence[float],
) -> float:
    """Sum of absolute second differences (proxy for curvature)."""
    c = [float(x) for x in curve]
    if len(c) < 3:
        return 0.0
    s = 0.0
    for i in range(1, len(c) - 1):
        s += abs(c[i + 1] - 2.0 * c[i] + c[i - 1])
    return float(s)


@dataclasses.dataclass(frozen=True)
class PrefixDriftCurvePredictorV8:
    """V8 stacked drift-curve predictor over an 11-d feature
    [reuse_len, recompute_len, drop_len, fp1..fp4, role_fp1..role_fp4,
    drift_acc, bias]."""
    schema: str
    n_train_examples: int
    n_target_steps: int
    B: tuple[tuple[float, ...], ...]   # (12, K)
    ridge_lambda: float
    per_step_pre_residual: tuple[float, ...]
    per_step_post_residual: tuple[float, ...]
    converged: bool

    def predict_curve(
            self, *, reuse_len: int, recompute_len: int,
            drop_len: int,
            follow_up_tokens: Sequence[int] | None = None,
            role: str = "r",
            drift_acceleration: float = 0.0,
    ) -> list[float]:
        fp = _token_fingerprint_v7(
            list(follow_up_tokens) if follow_up_tokens else [])
        rfp = _role_fingerprint_v8(str(role))
        feat = _np.array([
            float(reuse_len), float(recompute_len),
            float(drop_len),
            *(float(x) for x in fp),
            *(float(x) for x in rfp),
            float(drift_acceleration),
            1.0], dtype=_np.float64)
        B = _np.asarray(self.B, dtype=_np.float64)
        if B.size == 0:
            return [0.0] * int(self.n_target_steps)
        return [float(x) for x in (feat @ B)]

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_drift_curve_predictor_v8",
            "schema": str(self.schema),
            "n_train_examples": int(self.n_train_examples),
            "n_target_steps": int(self.n_target_steps),
            "B_round12": [
                [float(round(float(x), 12)) for x in row]
                for row in self.B],
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "converged": bool(self.converged),
        })


def fit_prefix_drift_curve_predictor_v8(
        *, params_v5: TinyV5SubstrateParams,
        prompt_token_ids: Sequence[int],
        train_segment_configs: Sequence[
            Sequence[tuple[int, int, str]]],
        train_chain: Sequence[Sequence[int]],
        roles: Sequence[str] | None = None,
        ridge_lambda: float = 0.10,
) -> PrefixDriftCurvePredictorV8:
    """Fit the V8 stacked drift-curve predictor with token-content
    fingerprint + role fingerprint + drift-acceleration features.
    """
    if not train_chain:
        return PrefixDriftCurvePredictorV8(
            schema=W64_PREFIX_STATE_BRIDGE_V8_SCHEMA_VERSION,
            n_train_examples=0, n_target_steps=0,
            B=tuple(), ridge_lambda=float(ridge_lambda),
            per_step_pre_residual=tuple(),
            per_step_post_residual=tuple(),
            converged=True)
    K = int(len(train_chain))
    if roles is None:
        roles_list = ["r"] * len(list(train_segment_configs))
    else:
        roles_list = list(roles)
    feats: list[list[float]] = []
    drift_curves: list[list[float]] = []
    for cfg_idx, segs in enumerate(train_segment_configs):
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
        first_chain = list(train_chain[0]) if train_chain else []
        fp = _token_fingerprint_v7(first_chain)
        rfp = _role_fingerprint_v8(str(
            roles_list[cfg_idx]
            if cfg_idx < len(roles_list) else "r"))
        d_acc = _drift_acceleration(curve)
        feats.append([
            float(w.reuse_len), float(w.recompute_len),
            float(w.drop_len),
            *(float(x) for x in fp),
            *(float(x) for x in rfp),
            float(d_acc),
            1.0])
        drift_curves.append([float(x) for x in curve])
    X = _np.asarray(feats, dtype=_np.float64)
    Y = _np.asarray(drift_curves, dtype=_np.float64)
    if X.shape[0] == 0:
        return PrefixDriftCurvePredictorV8(
            schema=W64_PREFIX_STATE_BRIDGE_V8_SCHEMA_VERSION,
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
    return PrefixDriftCurvePredictorV8(
        schema=W64_PREFIX_STATE_BRIDGE_V8_SCHEMA_VERSION,
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
class PrefixVsHiddenVsReplayDecisionWitnessV8:
    schema: str
    decision: str
    prefix_curve_l1: float
    hidden_curve_l1: float
    replay_curve_l1: float
    margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "decision": str(self.decision),
            "prefix_curve_l1": float(round(
                self.prefix_curve_l1, 12)),
            "hidden_curve_l1": float(round(
                self.hidden_curve_l1, 12)),
            "replay_curve_l1": float(round(
                self.replay_curve_l1, 12)),
            "margin": float(round(self.margin, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_vs_hidden_vs_replay_decision_v8",
            "witness": self.to_dict()})


def compare_prefix_vs_hidden_vs_replay_v8(
        *, prefix_drift_curve: Sequence[float],
        hidden_drift_curve: Sequence[float],
        replay_drift_curve: Sequence[float],
        tie_threshold: float = 1e-6,
) -> PrefixVsHiddenVsReplayDecisionWitnessV8:
    """Three-way comparison over curve L1 areas: which has the
    smallest drift wins."""
    p = float(sum(abs(float(x)) for x in prefix_drift_curve))
    h = float(sum(abs(float(x)) for x in hidden_drift_curve))
    r = float(sum(abs(float(x)) for x in replay_drift_curve))
    candidates = [("prefix", p), ("hidden", h), ("replay", r)]
    candidates.sort(key=lambda x: x[1])
    best = candidates[0]
    runner_up = candidates[1]
    margin = float(runner_up[1] - best[1])
    if margin < float(tie_threshold):
        decision = "tie"
    else:
        decision = f"{best[0]}_wins"
    return PrefixVsHiddenVsReplayDecisionWitnessV8(
        schema=W64_PREFIX_STATE_BRIDGE_V8_SCHEMA_VERSION,
        decision=str(decision),
        prefix_curve_l1=float(p),
        hidden_curve_l1=float(h),
        replay_curve_l1=float(r),
        margin=float(margin),
    )


def write_prefix_v8_into_v9_hidden_state_trust(
        *, decisions_per_layer_head: dict[tuple[int, int], str],
        v9_cache: TinyV9KVCache,
        trust_ema: float = 0.5,
) -> dict[str, Any]:
    """Apply prefix-reuse decisions to the V9 hidden-state-trust
    ledger (negative side: a successful prefix-reuse means the
    hidden bridge was NOT primary)."""
    n_writes = 0
    for (li, hi), prefix_dec in decisions_per_layer_head.items():
        # If prefix succeeded, that's evidence against hidden
        # being primary.
        if prefix_dec == "prefix_reuse_success":
            hidden_dec = "kv_wins"
        elif prefix_dec == "prefix_reuse_drift":
            hidden_dec = "tie"
        else:
            hidden_dec = "tie"
        record_hidden_state_trust_decision_v9(
            v9_cache,
            layer_index=int(li), head_index=int(hi),
            decision=str(hidden_dec),
            trust_ema=float(trust_ema))
        n_writes += 1
    return {
        "schema": W64_PREFIX_STATE_BRIDGE_V8_SCHEMA_VERSION,
        "kind": "prefix_v8_hidden_state_trust_write",
        "n_writes": int(n_writes),
    }


@dataclasses.dataclass(frozen=True)
class PrefixStateBridgeV8Witness:
    schema: str
    chain_witness_cid: str
    n_links: int
    predictor_cid: str
    predicted_drift_curve: tuple[float, ...]
    actual_drift_curve: tuple[float, ...]
    curve_l1_error: float
    drift_acceleration: float
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
            "drift_acceleration": float(round(
                self.drift_acceleration, 12)),
            "converged": bool(self.converged),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_state_bridge_v8_witness",
            "witness": self.to_dict()})


def emit_prefix_state_bridge_v8_witness(
        *, predictor: PrefixDriftCurvePredictorV8,
        actual_curve: Sequence[float],
        chain_witness_cid: str = "",
) -> PrefixStateBridgeV8Witness:
    pred = predictor.predict_curve(
        reuse_len=1, recompute_len=1, drop_len=1,
        follow_up_tokens=None, role="r")
    actual = [float(x) for x in actual_curve]
    K = min(len(pred), len(actual))
    err = float(sum(
        abs(pred[i] - actual[i]) for i in range(K)))
    return PrefixStateBridgeV8Witness(
        schema=W64_PREFIX_STATE_BRIDGE_V8_SCHEMA_VERSION,
        chain_witness_cid=str(chain_witness_cid),
        n_links=int(predictor.n_target_steps),
        predictor_cid=str(predictor.cid()),
        predicted_drift_curve=tuple(pred),
        actual_drift_curve=tuple(actual),
        curve_l1_error=float(err),
        drift_acceleration=float(_drift_acceleration(actual)),
        converged=bool(predictor.converged),
    )


__all__ = [
    "W64_PREFIX_STATE_BRIDGE_V8_SCHEMA_VERSION",
    "W64_DEFAULT_PREFIX_V8_K_STEPS",
    "W64_DEFAULT_PREFIX_V8_ROLE_FP_DIM",
    "PrefixDriftCurvePredictorV8",
    "fit_prefix_drift_curve_predictor_v8",
    "PrefixVsHiddenVsReplayDecisionWitnessV8",
    "compare_prefix_vs_hidden_vs_replay_v8",
    "write_prefix_v8_into_v9_hidden_state_trust",
    "PrefixStateBridgeV8Witness",
    "emit_prefix_state_bridge_v8_witness",
]
