"""W66 M4 — Prefix-State Bridge V10.

Strictly extends W65's ``coordpy.prefix_state_bridge_v9``. V9
supported K=64 drift curves and a 20-dim role + task fingerprint.
V10 adds:

* **K=96 drift curve** — extends V9's K=64 with 32 additional
  zero-padded steps; no new ridge fit. ``W66-L-V10-PREFIX-K96-
  STRUCTURAL-CAP`` documents the structural extension.
* **Role + task + team fingerprint (30-dim)** — concatenates the
  4-dim role fingerprint with a 16-dim task fingerprint and a
  10-dim team-id SHA256 fingerprint.
* **Five-way prefix/hidden/replay/team/recover comparator** —
  extends V9's four-way comparator with a *recover* curve for the
  team-failure-recovery regime.

Honest scope (W66)
------------------

* The K=96 extension does NOT fit additional ridge parameters; it
  uses V9's prediction extended with last-value extrapolation.
* The 30-dim fingerprint is a fixed SHA256 projection.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.prefix_state_bridge_v10 requires numpy") from exc

from .prefix_state_bridge_v9 import (
    PrefixDriftCurvePredictorV9,
    W65_DEFAULT_PREFIX_V9_K_STEPS,
    W65_DEFAULT_PREFIX_V9_TASK_FP_DIM,
    _task_fingerprint_v9,
    compute_role_task_fingerprint_v9,
    fit_prefix_drift_curve_predictor_v9,
    substrate_measured_drift_v9,
)
from .prefix_state_bridge_v8 import (
    W64_DEFAULT_PREFIX_V8_ROLE_FP_DIM,
)
from .tiny_substrate_v3 import _sha256_hex
from .tiny_substrate_v5 import TinyV5SubstrateParams


W66_PREFIX_V10_SCHEMA_VERSION: str = (
    "coordpy.prefix_state_bridge_v10.v1")
W66_DEFAULT_PREFIX_V10_K_STEPS: int = 96
W66_DEFAULT_PREFIX_V10_TEAM_FP_DIM: int = 10


def _team_fingerprint_v10(
        team_id: str,
        *, fp_dim: int = W66_DEFAULT_PREFIX_V10_TEAM_FP_DIM,
) -> list[float]:
    payload = str(team_id).encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()
    out: list[float] = []
    for i in range(int(fp_dim)):
        nb = h[(i * 4) % len(h):(i * 4) % len(h) + 4]
        if not nb:
            nb = "0000"
        v = (int(nb, 16) / 32767.5) - 1.0
        out.append(float(round(v, 12)))
    return out


def compute_role_task_team_fingerprint_v10(
        *, role: str, task_name: str, team_id: str,
) -> list[float]:
    base = compute_role_task_fingerprint_v9(
        role=str(role), task_name=str(task_name))
    return list(base) + list(_team_fingerprint_v10(str(team_id)))


@dataclasses.dataclass(frozen=True)
class PrefixDriftCurvePredictorV10:
    schema: str
    inner_v9_cid: str
    inner_v9: PrefixDriftCurvePredictorV9
    k_steps_v10: int

    def predict_curve_v10(
            self, *, reuse_len: int, recompute_len: int,
            drop_len: int,
            follow_up_tokens: Sequence[int] | None = None,
            role: str = "r",
            task_name: str = "default",
            team_id: str = "team",
            drift_acceleration: float = 0.0,
    ) -> list[float]:
        v9_curve = self.inner_v9.predict_curve_v9(
            reuse_len=int(reuse_len),
            recompute_len=int(recompute_len),
            drop_len=int(drop_len),
            follow_up_tokens=follow_up_tokens,
            role=str(role), task_name=str(task_name),
            drift_acceleration=float(drift_acceleration))
        out = list(v9_curve)
        team_bias = float(sum(
            _team_fingerprint_v10(str(team_id))[:4])) / 4.0 * 1e-3
        last = float(out[-1]) if out else 0.0
        while len(out) < int(self.k_steps_v10):
            out.append(float(last + team_bias))
        return out[:int(self.k_steps_v10)]

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_drift_curve_predictor_v10",
            "schema": str(self.schema),
            "inner_v9_cid": str(self.inner_v9_cid),
            "k_steps_v10": int(self.k_steps_v10),
        })


def fit_prefix_drift_curve_predictor_v10(
        *, params_v5: TinyV5SubstrateParams,
        prompt_token_ids: Sequence[int],
        train_segment_configs: Sequence[
            Sequence[tuple[int, int, str]]],
        train_chain: Sequence[Sequence[int]],
        roles: Sequence[str] | None = None,
        k_steps_v10: int = W66_DEFAULT_PREFIX_V10_K_STEPS,
        ridge_lambda: float = 0.10,
) -> PrefixDriftCurvePredictorV10:
    v9 = fit_prefix_drift_curve_predictor_v9(
        params_v5=params_v5,
        prompt_token_ids=list(prompt_token_ids),
        train_segment_configs=list(train_segment_configs),
        train_chain=list(train_chain),
        roles=roles,
        ridge_lambda=float(ridge_lambda))
    return PrefixDriftCurvePredictorV10(
        schema=W66_PREFIX_V10_SCHEMA_VERSION,
        inner_v9_cid=str(v9.cid()),
        inner_v9=v9,
        k_steps_v10=int(k_steps_v10),
    )


@dataclasses.dataclass(frozen=True)
class PrefixV10FiveWayDecision:
    decision: str
    prefix_l1: float
    hidden_l1: float
    replay_l1: float
    team_l1: float
    recover_l1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": str(self.decision),
            "prefix_l1": float(round(self.prefix_l1, 12)),
            "hidden_l1": float(round(self.hidden_l1, 12)),
            "replay_l1": float(round(self.replay_l1, 12)),
            "team_l1": float(round(self.team_l1, 12)),
            "recover_l1": float(round(self.recover_l1, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_v10_five_way_decision",
            "decision": self.to_dict()})


def compare_prefix_vs_hidden_vs_replay_vs_team_vs_recover_v10(
        *, prefix_drift_curve: Sequence[float],
        hidden_drift_curve: Sequence[float],
        replay_drift_curve: Sequence[float],
        team_drift_curve: Sequence[float],
        recover_drift_curve: Sequence[float],
) -> PrefixV10FiveWayDecision:
    p = substrate_measured_drift_v9(prefix_drift_curve)
    h = substrate_measured_drift_v9(hidden_drift_curve)
    r = substrate_measured_drift_v9(replay_drift_curve)
    t = substrate_measured_drift_v9(team_drift_curve)
    rc = substrate_measured_drift_v9(recover_drift_curve)
    pairs = [
        ("prefix_wins", p), ("hidden_wins", h),
        ("replay_wins", r), ("team_wins", t),
        ("recover_wins", rc)]
    best = min(pairs, key=lambda x: x[1])
    n_ties = sum(1 for _, v in pairs if abs(v - best[1]) < 1e-12)
    decision = "tie" if n_ties > 1 else best[0]
    return PrefixV10FiveWayDecision(
        decision=str(decision),
        prefix_l1=float(p), hidden_l1=float(h),
        replay_l1=float(r), team_l1=float(t),
        recover_l1=float(rc),
    )


@dataclasses.dataclass(frozen=True)
class PrefixStateV10Witness:
    schema: str
    predictor_cid: str
    k_steps_v10: int
    role_task_team_fingerprint_dim: int
    five_way_decision_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "predictor_cid": str(self.predictor_cid),
            "k_steps_v10": int(self.k_steps_v10),
            "role_task_team_fingerprint_dim": int(
                self.role_task_team_fingerprint_dim),
            "five_way_decision_cid": str(
                self.five_way_decision_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_v10_witness",
            "witness": self.to_dict()})


def emit_prefix_state_v10_witness(
        *, predictor: PrefixDriftCurvePredictorV10 | None = None,
        five_way_decision: (
            PrefixV10FiveWayDecision | None) = None,
) -> PrefixStateV10Witness:
    return PrefixStateV10Witness(
        schema=W66_PREFIX_V10_SCHEMA_VERSION,
        predictor_cid=(
            predictor.cid() if predictor is not None else ""),
        k_steps_v10=int(
            predictor.k_steps_v10 if predictor is not None
            else W66_DEFAULT_PREFIX_V10_K_STEPS),
        role_task_team_fingerprint_dim=int(
            W64_DEFAULT_PREFIX_V8_ROLE_FP_DIM
            + W65_DEFAULT_PREFIX_V9_TASK_FP_DIM
            + W66_DEFAULT_PREFIX_V10_TEAM_FP_DIM),
        five_way_decision_cid=(
            five_way_decision.cid()
            if five_way_decision is not None else ""),
    )


__all__ = [
    "W66_PREFIX_V10_SCHEMA_VERSION",
    "W66_DEFAULT_PREFIX_V10_K_STEPS",
    "W66_DEFAULT_PREFIX_V10_TEAM_FP_DIM",
    "compute_role_task_team_fingerprint_v10",
    "PrefixDriftCurvePredictorV10",
    "fit_prefix_drift_curve_predictor_v10",
    "PrefixV10FiveWayDecision",
    "compare_prefix_vs_hidden_vs_replay_vs_team_vs_recover_v10",
    "PrefixStateV10Witness",
    "emit_prefix_state_v10_witness",
]
