"""W67 M4 — Prefix-State Bridge V11.

Strictly extends W66's ``coordpy.prefix_state_bridge_v10``. V10
supported K=96 drift curves and a 30-dim role+task+team fingerprint.
V11 adds:

* **K=128 drift curve** — extends V10's K=96 with 32 additional
  last-value-extrapolated steps; no new ridge fit.
  ``W67-L-V11-PREFIX-K128-STRUCTURAL-CAP`` documents.
* **Role + task + team + branch fingerprint (40-dim)** —
  concatenates V10's 30-dim with a 10-dim branch-id SHA256
  fingerprint.
* **Six-way prefix/hidden/replay/team/recover/branch comparator**
  — extends V10's five-way with a *branch* curve for the
  branch-merge-reconciliation regime.

Honest scope (W67)
------------------

* The K=128 extension does NOT fit additional ridge parameters; it
  uses V10's prediction extended with last-value extrapolation.
* The 40-dim fingerprint is a fixed SHA256 projection.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

from .prefix_state_bridge_v10 import (
    PrefixDriftCurvePredictorV10,
    W66_DEFAULT_PREFIX_V10_K_STEPS,
    W66_DEFAULT_PREFIX_V10_TEAM_FP_DIM,
    compute_role_task_team_fingerprint_v10,
    fit_prefix_drift_curve_predictor_v10,
)
from .prefix_state_bridge_v9 import substrate_measured_drift_v9
from .prefix_state_bridge_v8 import (
    W64_DEFAULT_PREFIX_V8_ROLE_FP_DIM,
)
from .prefix_state_bridge_v9 import (
    W65_DEFAULT_PREFIX_V9_TASK_FP_DIM,
)
from .tiny_substrate_v3 import _sha256_hex
from .tiny_substrate_v5 import TinyV5SubstrateParams


W67_PREFIX_V11_SCHEMA_VERSION: str = (
    "coordpy.prefix_state_bridge_v11.v1")
W67_DEFAULT_PREFIX_V11_K_STEPS: int = 128
W67_DEFAULT_PREFIX_V11_BRANCH_FP_DIM: int = 10


def _branch_fingerprint_v11(
        branch_id: str,
        *, fp_dim: int = W67_DEFAULT_PREFIX_V11_BRANCH_FP_DIM,
) -> list[float]:
    payload = str(branch_id).encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()
    out: list[float] = []
    for i in range(int(fp_dim)):
        nb = h[(i * 4) % len(h):(i * 4) % len(h) + 4]
        if not nb:
            nb = "0000"
        v = (int(nb, 16) / 32767.5) - 1.0
        out.append(float(round(v, 12)))
    return out


def compute_role_task_team_branch_fingerprint_v11(
        *, role: str, task_name: str, team_id: str, branch_id: str,
) -> list[float]:
    base = compute_role_task_team_fingerprint_v10(
        role=str(role), task_name=str(task_name),
        team_id=str(team_id))
    return list(base) + list(_branch_fingerprint_v11(str(branch_id)))


@dataclasses.dataclass(frozen=True)
class PrefixDriftCurvePredictorV11:
    schema: str
    inner_v10_cid: str
    inner_v10: PrefixDriftCurvePredictorV10
    k_steps_v11: int

    def predict_curve_v11(
            self, *, reuse_len: int, recompute_len: int,
            drop_len: int,
            follow_up_tokens: Sequence[int] | None = None,
            role: str = "r",
            task_name: str = "default",
            team_id: str = "team",
            branch_id: str = "main",
            drift_acceleration: float = 0.0,
    ) -> list[float]:
        v10_curve = self.inner_v10.predict_curve_v10(
            reuse_len=int(reuse_len),
            recompute_len=int(recompute_len),
            drop_len=int(drop_len),
            follow_up_tokens=follow_up_tokens,
            role=str(role), task_name=str(task_name),
            team_id=str(team_id),
            drift_acceleration=float(drift_acceleration))
        out = list(v10_curve)
        branch_bias = float(sum(
            _branch_fingerprint_v11(str(branch_id))[:4])) / 4.0 * 1e-3
        last = float(out[-1]) if out else 0.0
        while len(out) < int(self.k_steps_v11):
            out.append(float(last + branch_bias))
        return out[:int(self.k_steps_v11)]

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_drift_curve_predictor_v11",
            "schema": str(self.schema),
            "inner_v10_cid": str(self.inner_v10_cid),
            "k_steps_v11": int(self.k_steps_v11),
        })


def fit_prefix_drift_curve_predictor_v11(
        *, params_v5: TinyV5SubstrateParams,
        prompt_token_ids: Sequence[int],
        train_segment_configs: Sequence[
            Sequence[tuple[int, int, str]]],
        train_chain: Sequence[Sequence[int]],
        roles: Sequence[str] | None = None,
        k_steps_v11: int = W67_DEFAULT_PREFIX_V11_K_STEPS,
        ridge_lambda: float = 0.10,
) -> PrefixDriftCurvePredictorV11:
    v10 = fit_prefix_drift_curve_predictor_v10(
        params_v5=params_v5,
        prompt_token_ids=list(prompt_token_ids),
        train_segment_configs=list(train_segment_configs),
        train_chain=list(train_chain),
        roles=roles,
        ridge_lambda=float(ridge_lambda))
    return PrefixDriftCurvePredictorV11(
        schema=W67_PREFIX_V11_SCHEMA_VERSION,
        inner_v10_cid=str(v10.cid()),
        inner_v10=v10,
        k_steps_v11=int(k_steps_v11),
    )


@dataclasses.dataclass(frozen=True)
class PrefixV11SixWayDecision:
    decision: str
    prefix_l1: float
    hidden_l1: float
    replay_l1: float
    team_l1: float
    recover_l1: float
    branch_l1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": str(self.decision),
            "prefix_l1": float(round(self.prefix_l1, 12)),
            "hidden_l1": float(round(self.hidden_l1, 12)),
            "replay_l1": float(round(self.replay_l1, 12)),
            "team_l1": float(round(self.team_l1, 12)),
            "recover_l1": float(round(self.recover_l1, 12)),
            "branch_l1": float(round(self.branch_l1, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_v11_six_way_decision",
            "decision": self.to_dict()})


def compare_prefix_vs_hidden_vs_replay_vs_team_vs_recover_vs_branch_v11(
        *, prefix_drift_curve: Sequence[float],
        hidden_drift_curve: Sequence[float],
        replay_drift_curve: Sequence[float],
        team_drift_curve: Sequence[float],
        recover_drift_curve: Sequence[float],
        branch_drift_curve: Sequence[float],
) -> PrefixV11SixWayDecision:
    p = substrate_measured_drift_v9(prefix_drift_curve)
    h = substrate_measured_drift_v9(hidden_drift_curve)
    r = substrate_measured_drift_v9(replay_drift_curve)
    t = substrate_measured_drift_v9(team_drift_curve)
    rc = substrate_measured_drift_v9(recover_drift_curve)
    b = substrate_measured_drift_v9(branch_drift_curve)
    pairs = [
        ("prefix_wins", p), ("hidden_wins", h),
        ("replay_wins", r), ("team_wins", t),
        ("recover_wins", rc), ("branch_wins", b)]
    best = min(pairs, key=lambda x: x[1])
    n_ties = sum(1 for _, v in pairs if abs(v - best[1]) < 1e-12)
    decision = "tie" if n_ties > 1 else best[0]
    return PrefixV11SixWayDecision(
        decision=str(decision),
        prefix_l1=float(p), hidden_l1=float(h),
        replay_l1=float(r), team_l1=float(t),
        recover_l1=float(rc), branch_l1=float(b),
    )


@dataclasses.dataclass(frozen=True)
class PrefixStateV11Witness:
    schema: str
    predictor_cid: str
    k_steps_v11: int
    role_task_team_branch_fingerprint_dim: int
    six_way_decision_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "predictor_cid": str(self.predictor_cid),
            "k_steps_v11": int(self.k_steps_v11),
            "role_task_team_branch_fingerprint_dim": int(
                self.role_task_team_branch_fingerprint_dim),
            "six_way_decision_cid": str(self.six_way_decision_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_v11_witness",
            "witness": self.to_dict()})


def emit_prefix_state_v11_witness(
        *, predictor: PrefixDriftCurvePredictorV11 | None = None,
        six_way_decision: (
            PrefixV11SixWayDecision | None) = None,
) -> PrefixStateV11Witness:
    return PrefixStateV11Witness(
        schema=W67_PREFIX_V11_SCHEMA_VERSION,
        predictor_cid=(
            predictor.cid() if predictor is not None else ""),
        k_steps_v11=int(
            predictor.k_steps_v11 if predictor is not None
            else W67_DEFAULT_PREFIX_V11_K_STEPS),
        role_task_team_branch_fingerprint_dim=int(
            W64_DEFAULT_PREFIX_V8_ROLE_FP_DIM
            + W65_DEFAULT_PREFIX_V9_TASK_FP_DIM
            + W66_DEFAULT_PREFIX_V10_TEAM_FP_DIM
            + W67_DEFAULT_PREFIX_V11_BRANCH_FP_DIM),
        six_way_decision_cid=(
            six_way_decision.cid()
            if six_way_decision is not None else ""),
    )


__all__ = [
    "W67_PREFIX_V11_SCHEMA_VERSION",
    "W67_DEFAULT_PREFIX_V11_K_STEPS",
    "W67_DEFAULT_PREFIX_V11_BRANCH_FP_DIM",
    "compute_role_task_team_branch_fingerprint_v11",
    "PrefixDriftCurvePredictorV11",
    "fit_prefix_drift_curve_predictor_v11",
    "PrefixV11SixWayDecision",
    "compare_prefix_vs_hidden_vs_replay_vs_team_vs_recover_vs_branch_v11",
    "PrefixStateV11Witness",
    "emit_prefix_state_v11_witness",
]
