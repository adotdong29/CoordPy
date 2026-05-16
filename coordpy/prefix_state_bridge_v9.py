"""W65 M4 — Prefix-State Bridge V9.

Strictly extends W64's ``coordpy.prefix_state_bridge_v8``. V8
predicted a K=32 drift curve using a 12-feature stacked ridge with
token + role fingerprints. V9 adds:

* **K=64 drift curve** support — V9 simply re-uses V8's fit at
  K=32 and *extends* the predictor with a zero-padded curve of
  length K=64; we record the structural extension as a separate
  field so callers can compare drift curves at K=32 and K=64.
* **Role + task fingerprint (20-dim feature)** —
  ``compute_role_task_fingerprint_v9`` concatenates the 4-dim role
  fingerprint with a 16-dim task-name SHA256 fingerprint.
* **Substrate-measured drift probe** —
  ``substrate_measured_drift_v9`` returns the per-step L1 area of
  a drift curve (used as a substrate-side severity metric).
* **Four-way prefix vs hidden vs replay vs team comparator** —
  ``compare_prefix_vs_hidden_vs_replay_vs_team_v9`` extends V8's
  three-way comparator with the team-coordination curve.

Honest scope (W65)
------------------

* The K=64 extension does NOT fit additional ridge parameters; it
  uses the V8 fit's prediction extended with zeros. The K=64
  feature space is structural — the V8 fit is the only fitted
  predictor. ``W65-L-V9-PREFIX-K64-STRUCTURAL-CAP`` documents.
* The 20-dim role + task fingerprint is a fixed SHA256 projection;
  not a learned representation.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.prefix_state_bridge_v9 requires numpy") from exc

from .prefix_state_bridge_v8 import (
    PrefixDriftCurvePredictorV8,
    W64_DEFAULT_PREFIX_V8_K_STEPS,
    W64_DEFAULT_PREFIX_V8_ROLE_FP_DIM,
    _role_fingerprint_v8,
    compare_prefix_vs_hidden_vs_replay_v8,
    fit_prefix_drift_curve_predictor_v8,
)
from .tiny_substrate_v3 import _sha256_hex
from .tiny_substrate_v5 import TinyV5SubstrateParams


W65_PREFIX_V9_SCHEMA_VERSION: str = (
    "coordpy.prefix_state_bridge_v9.v1")
W65_DEFAULT_PREFIX_V9_K_STEPS: int = 64
W65_DEFAULT_PREFIX_V9_TASK_FP_DIM: int = 16


def _task_fingerprint_v9(
        task_name: str,
        *, fp_dim: int = W65_DEFAULT_PREFIX_V9_TASK_FP_DIM,
) -> list[float]:
    payload = str(task_name).encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()
    out: list[float] = []
    for i in range(int(fp_dim)):
        nb = h[(i * 4) % len(h):(i * 4) % len(h) + 4]
        if not nb:
            nb = "0000"
        v = (int(nb, 16) / 32767.5) - 1.0
        out.append(float(round(v, 12)))
    return out


def compute_role_task_fingerprint_v9(
        *, role: str, task_name: str,
) -> list[float]:
    role_fp = _role_fingerprint_v8(
        str(role), fp_dim=W64_DEFAULT_PREFIX_V8_ROLE_FP_DIM)
    task_fp = _task_fingerprint_v9(
        str(task_name),
        fp_dim=W65_DEFAULT_PREFIX_V9_TASK_FP_DIM)
    return list(role_fp) + list(task_fp)


@dataclasses.dataclass(frozen=True)
class PrefixDriftCurvePredictorV9:
    schema: str
    inner_v8_cid: str
    inner_v8: PrefixDriftCurvePredictorV8
    k_steps_v9: int

    def predict_curve_v9(
            self, *, reuse_len: int, recompute_len: int,
            drop_len: int,
            follow_up_tokens: Sequence[int] | None = None,
            role: str = "r",
            task_name: str = "default",
            drift_acceleration: float = 0.0,
    ) -> list[float]:
        v8_curve = self.inner_v8.predict_curve(
            reuse_len=int(reuse_len),
            recompute_len=int(recompute_len),
            drop_len=int(drop_len),
            follow_up_tokens=follow_up_tokens,
            role=str(role),
            drift_acceleration=float(drift_acceleration))
        out = list(v8_curve)
        # Pad with last-value extrapolation up to K=64. The task
        # fingerprint is mixed in additively (small bias).
        task_bias = float(sum(
            _task_fingerprint_v9(str(task_name))[:4])) / 4.0 * 1e-3
        last = float(out[-1]) if out else 0.0
        while len(out) < int(self.k_steps_v9):
            out.append(float(last + task_bias))
        return out[:int(self.k_steps_v9)]

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_drift_curve_predictor_v9",
            "schema": str(self.schema),
            "inner_v8_cid": str(self.inner_v8_cid),
            "k_steps_v9": int(self.k_steps_v9),
        })


def fit_prefix_drift_curve_predictor_v9(
        *, params_v5: TinyV5SubstrateParams,
        prompt_token_ids: Sequence[int],
        train_segment_configs: Sequence[
            Sequence[tuple[int, int, str]]],
        train_chain: Sequence[Sequence[int]],
        roles: Sequence[str] | None = None,
        k_steps_v9: int = W65_DEFAULT_PREFIX_V9_K_STEPS,
        ridge_lambda: float = 0.10,
) -> PrefixDriftCurvePredictorV9:
    """Fit the V8 inner predictor; V9 wraps it with K=64 support
    (zero-padded extrapolation; no new ridge solve)."""
    v8_pred = fit_prefix_drift_curve_predictor_v8(
        params_v5=params_v5,
        prompt_token_ids=list(prompt_token_ids),
        train_segment_configs=list(train_segment_configs),
        train_chain=list(train_chain),
        roles=roles,
        ridge_lambda=float(ridge_lambda))
    return PrefixDriftCurvePredictorV9(
        schema=W65_PREFIX_V9_SCHEMA_VERSION,
        inner_v8_cid=str(v8_pred.cid()),
        inner_v8=v8_pred,
        k_steps_v9=int(k_steps_v9),
    )


def substrate_measured_drift_v9(
        curve: Sequence[float]) -> float:
    """Per-step L1 area of a drift curve (substrate-side severity).
    """
    arr = _np.asarray(curve, dtype=_np.float64)
    if arr.size == 0:
        return 0.0
    return float(_np.linalg.norm(arr.ravel(), ord=1))


@dataclasses.dataclass(frozen=True)
class PrefixV9FourWayDecision:
    decision: str   # prefix_wins / hidden_wins / replay_wins /
                    # team_wins / tie
    prefix_l1: float
    hidden_l1: float
    replay_l1: float
    team_l1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": str(self.decision),
            "prefix_l1": float(round(self.prefix_l1, 12)),
            "hidden_l1": float(round(self.hidden_l1, 12)),
            "replay_l1": float(round(self.replay_l1, 12)),
            "team_l1": float(round(self.team_l1, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_v9_four_way_decision",
            "decision": self.to_dict()})


def compare_prefix_vs_hidden_vs_replay_vs_team_v9(
        *, prefix_drift_curve: Sequence[float],
        hidden_drift_curve: Sequence[float],
        replay_drift_curve: Sequence[float],
        team_drift_curve: Sequence[float],
) -> PrefixV9FourWayDecision:
    p = substrate_measured_drift_v9(prefix_drift_curve)
    h = substrate_measured_drift_v9(hidden_drift_curve)
    r = substrate_measured_drift_v9(replay_drift_curve)
    t = substrate_measured_drift_v9(team_drift_curve)
    pairs = [("prefix_wins", p), ("hidden_wins", h),
             ("replay_wins", r), ("team_wins", t)]
    best = min(pairs, key=lambda x: x[1])
    # Tie if any two are equal to best within tolerance.
    n_ties = sum(1 for _, v in pairs if abs(v - best[1]) < 1e-12)
    decision = "tie" if n_ties > 1 else best[0]
    return PrefixV9FourWayDecision(
        decision=str(decision),
        prefix_l1=float(p), hidden_l1=float(h),
        replay_l1=float(r), team_l1=float(t),
    )


@dataclasses.dataclass(frozen=True)
class PrefixStateV9Witness:
    schema: str
    predictor_cid: str
    k_steps_v9: int
    role_task_fingerprint_dim: int
    four_way_decision_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "predictor_cid": str(self.predictor_cid),
            "k_steps_v9": int(self.k_steps_v9),
            "role_task_fingerprint_dim": int(
                self.role_task_fingerprint_dim),
            "four_way_decision_cid": str(
                self.four_way_decision_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_v9_witness",
            "witness": self.to_dict()})


def emit_prefix_state_v9_witness(
        *, predictor: PrefixDriftCurvePredictorV9 | None = None,
        four_way_decision: (
            PrefixV9FourWayDecision | None) = None,
) -> PrefixStateV9Witness:
    return PrefixStateV9Witness(
        schema=W65_PREFIX_V9_SCHEMA_VERSION,
        predictor_cid=(
            predictor.cid() if predictor is not None else ""),
        k_steps_v9=int(
            predictor.k_steps_v9 if predictor is not None
            else W65_DEFAULT_PREFIX_V9_K_STEPS),
        role_task_fingerprint_dim=int(
            W64_DEFAULT_PREFIX_V8_ROLE_FP_DIM
            + W65_DEFAULT_PREFIX_V9_TASK_FP_DIM),
        four_way_decision_cid=(
            four_way_decision.cid()
            if four_way_decision is not None else ""),
    )


__all__ = [
    "W65_PREFIX_V9_SCHEMA_VERSION",
    "W65_DEFAULT_PREFIX_V9_K_STEPS",
    "W65_DEFAULT_PREFIX_V9_TASK_FP_DIM",
    "compute_role_task_fingerprint_v9",
    "PrefixDriftCurvePredictorV9",
    "fit_prefix_drift_curve_predictor_v9",
    "substrate_measured_drift_v9",
    "PrefixV9FourWayDecision",
    "compare_prefix_vs_hidden_vs_replay_vs_team_v9",
    "PrefixStateV9Witness",
    "emit_prefix_state_v9_witness",
]
