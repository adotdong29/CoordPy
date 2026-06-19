"""W68 M4 — Prefix-State Bridge V12.

Strictly extends W67's ``coordpy.prefix_state_bridge_v11``. V11
supported K=128 drift curves and a 40-dim role+task+team+branch
fingerprint. V12 adds:

* **K=192 drift curve** — extends V11's K=128 with 64 additional
  last-value-extrapolated steps; no new ridge fit.
  ``W68-L-V12-PREFIX-K192-STRUCTURAL-CAP`` documents.
* **Role + task + team + branch + agent fingerprint (50-dim)** —
  concatenates V11's 40-dim with a 10-dim agent-replacement SHA256
  fingerprint.
* **Seven-way prefix/hidden/replay/team/recover/branch/contradict
  comparator** — extends V11's six-way with a *contradict* curve
  for the partial-contradiction-under-delayed-reconciliation
  regime.

Honest scope (W68)
------------------

* The K=192 extension does NOT fit additional ridge parameters; it
  uses V11's prediction extended with last-value extrapolation.
* The 50-dim fingerprint is a fixed SHA256 projection.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

from .prefix_state_bridge_v11 import (
    PrefixDriftCurvePredictorV11,
    W67_DEFAULT_PREFIX_V11_K_STEPS,
    W67_DEFAULT_PREFIX_V11_BRANCH_FP_DIM,
    compute_role_task_team_branch_fingerprint_v11,
    fit_prefix_drift_curve_predictor_v11,
)
from .prefix_state_bridge_v9 import substrate_measured_drift_v9
from .prefix_state_bridge_v8 import (
    W64_DEFAULT_PREFIX_V8_ROLE_FP_DIM,
)
from .prefix_state_bridge_v9 import (
    W65_DEFAULT_PREFIX_V9_TASK_FP_DIM,
)
from .prefix_state_bridge_v10 import (
    W66_DEFAULT_PREFIX_V10_TEAM_FP_DIM,
)
from .tiny_substrate_v3 import _sha256_hex
from .tiny_substrate_v5 import TinyV5SubstrateParams


W68_PREFIX_V12_SCHEMA_VERSION: str = (
    "coordpy.prefix_state_bridge_v12.v1")
W68_DEFAULT_PREFIX_V12_K_STEPS: int = 192
W68_DEFAULT_PREFIX_V12_AGENT_FP_DIM: int = 10


def _agent_fingerprint_v12(
        agent_id: str,
        *, fp_dim: int = W68_DEFAULT_PREFIX_V12_AGENT_FP_DIM,
) -> list[float]:
    payload = str(agent_id).encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()
    out: list[float] = []
    for i in range(int(fp_dim)):
        nb = h[(i * 4) % len(h):(i * 4) % len(h) + 4]
        if not nb:
            nb = "0000"
        v = (int(nb, 16) / 32767.5) - 1.0
        out.append(float(round(v, 12)))
    return out


def compute_role_task_team_branch_agent_fingerprint_v12(
        *, role: str, task_name: str, team_id: str, branch_id: str,
        agent_id: str,
) -> list[float]:
    base = compute_role_task_team_branch_fingerprint_v11(
        role=str(role), task_name=str(task_name),
        team_id=str(team_id), branch_id=str(branch_id))
    return list(base) + list(_agent_fingerprint_v12(str(agent_id)))


@dataclasses.dataclass(frozen=True)
class PrefixDriftCurvePredictorV12:
    schema: str
    inner_v11_cid: str
    inner_v11: PrefixDriftCurvePredictorV11
    k_steps_v12: int

    def predict_curve_v12(
            self, *, reuse_len: int, recompute_len: int,
            drop_len: int,
            follow_up_tokens: Sequence[int] | None = None,
            role: str = "r",
            task_name: str = "default",
            team_id: str = "team",
            branch_id: str = "main",
            agent_id: str = "a0",
            drift_acceleration: float = 0.0,
    ) -> list[float]:
        v11_curve = self.inner_v11.predict_curve_v11(
            reuse_len=int(reuse_len),
            recompute_len=int(recompute_len),
            drop_len=int(drop_len),
            follow_up_tokens=follow_up_tokens,
            role=str(role), task_name=str(task_name),
            team_id=str(team_id), branch_id=str(branch_id),
            drift_acceleration=float(drift_acceleration))
        out = list(v11_curve)
        agent_bias = float(sum(
            _agent_fingerprint_v12(str(agent_id))[:4])) / 4.0 * 1e-3
        last = float(out[-1]) if out else 0.0
        while len(out) < int(self.k_steps_v12):
            out.append(float(last + agent_bias))
        return out[:int(self.k_steps_v12)]

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_drift_curve_predictor_v12",
            "schema": str(self.schema),
            "inner_v11_cid": str(self.inner_v11_cid),
            "k_steps_v12": int(self.k_steps_v12),
        })


def fit_prefix_drift_curve_predictor_v12(
        *, params_v5: TinyV5SubstrateParams,
        prompt_token_ids: Sequence[int],
        train_segment_configs: Sequence[
            Sequence[tuple[int, int, str]]],
        train_chain: Sequence[Sequence[int]],
        roles: Sequence[str] | None = None,
        k_steps_v12: int = W68_DEFAULT_PREFIX_V12_K_STEPS,
        ridge_lambda: float = 0.10,
) -> PrefixDriftCurvePredictorV12:
    v11 = fit_prefix_drift_curve_predictor_v11(
        params_v5=params_v5,
        prompt_token_ids=list(prompt_token_ids),
        train_segment_configs=list(train_segment_configs),
        train_chain=list(train_chain),
        roles=roles,
        ridge_lambda=float(ridge_lambda))
    return PrefixDriftCurvePredictorV12(
        schema=W68_PREFIX_V12_SCHEMA_VERSION,
        inner_v11_cid=str(v11.cid()),
        inner_v11=v11,
        k_steps_v12=int(k_steps_v12),
    )


@dataclasses.dataclass(frozen=True)
class PrefixV12SevenWayDecision:
    decision: str
    prefix_l1: float
    hidden_l1: float
    replay_l1: float
    team_l1: float
    recover_l1: float
    branch_l1: float
    contradict_l1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": str(self.decision),
            "prefix_l1": float(round(self.prefix_l1, 12)),
            "hidden_l1": float(round(self.hidden_l1, 12)),
            "replay_l1": float(round(self.replay_l1, 12)),
            "team_l1": float(round(self.team_l1, 12)),
            "recover_l1": float(round(self.recover_l1, 12)),
            "branch_l1": float(round(self.branch_l1, 12)),
            "contradict_l1": float(round(self.contradict_l1, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_v12_seven_way_decision",
            "decision": self.to_dict()})


def compare_prefix_vs_hidden_vs_replay_vs_team_vs_recover_vs_branch_vs_contradict_v12(
        *, prefix_drift_curve: Sequence[float],
        hidden_drift_curve: Sequence[float],
        replay_drift_curve: Sequence[float],
        team_drift_curve: Sequence[float],
        recover_drift_curve: Sequence[float],
        branch_drift_curve: Sequence[float],
        contradict_drift_curve: Sequence[float],
) -> PrefixV12SevenWayDecision:
    p = substrate_measured_drift_v9(prefix_drift_curve)
    h = substrate_measured_drift_v9(hidden_drift_curve)
    r = substrate_measured_drift_v9(replay_drift_curve)
    t = substrate_measured_drift_v9(team_drift_curve)
    rc = substrate_measured_drift_v9(recover_drift_curve)
    b = substrate_measured_drift_v9(branch_drift_curve)
    c = substrate_measured_drift_v9(contradict_drift_curve)
    pairs = [
        ("prefix_wins", p), ("hidden_wins", h),
        ("replay_wins", r), ("team_wins", t),
        ("recover_wins", rc), ("branch_wins", b),
        ("contradict_wins", c)]
    best = min(pairs, key=lambda x: x[1])
    n_ties = sum(1 for _, v in pairs if abs(v - best[1]) < 1e-12)
    decision = "tie" if n_ties > 1 else best[0]
    return PrefixV12SevenWayDecision(
        decision=str(decision),
        prefix_l1=float(p), hidden_l1=float(h),
        replay_l1=float(r), team_l1=float(t),
        recover_l1=float(rc), branch_l1=float(b),
        contradict_l1=float(c),
    )


@dataclasses.dataclass(frozen=True)
class PrefixStateV12Witness:
    schema: str
    predictor_cid: str
    k_steps_v12: int
    role_task_team_branch_agent_fingerprint_dim: int
    seven_way_decision_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "predictor_cid": str(self.predictor_cid),
            "k_steps_v12": int(self.k_steps_v12),
            "role_task_team_branch_agent_fingerprint_dim": int(
                self
                .role_task_team_branch_agent_fingerprint_dim),
            "seven_way_decision_cid": str(
                self.seven_way_decision_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_v12_witness",
            "witness": self.to_dict()})


def emit_prefix_state_v12_witness(
        *, predictor: PrefixDriftCurvePredictorV12 | None = None,
        seven_way_decision: (
            PrefixV12SevenWayDecision | None) = None,
) -> PrefixStateV12Witness:
    return PrefixStateV12Witness(
        schema=W68_PREFIX_V12_SCHEMA_VERSION,
        predictor_cid=(
            predictor.cid() if predictor is not None else ""),
        k_steps_v12=int(
            predictor.k_steps_v12 if predictor is not None
            else W68_DEFAULT_PREFIX_V12_K_STEPS),
        role_task_team_branch_agent_fingerprint_dim=int(
            W64_DEFAULT_PREFIX_V8_ROLE_FP_DIM
            + W65_DEFAULT_PREFIX_V9_TASK_FP_DIM
            + W66_DEFAULT_PREFIX_V10_TEAM_FP_DIM
            + W67_DEFAULT_PREFIX_V11_BRANCH_FP_DIM
            + W68_DEFAULT_PREFIX_V12_AGENT_FP_DIM),
        seven_way_decision_cid=(
            seven_way_decision.cid()
            if seven_way_decision is not None else ""),
    )


__all__ = [
    "W68_PREFIX_V12_SCHEMA_VERSION",
    "W68_DEFAULT_PREFIX_V12_K_STEPS",
    "W68_DEFAULT_PREFIX_V12_AGENT_FP_DIM",
    "compute_role_task_team_branch_agent_fingerprint_v12",
    "PrefixDriftCurvePredictorV12",
    "fit_prefix_drift_curve_predictor_v12",
    "PrefixV12SevenWayDecision",
    "compare_prefix_vs_hidden_vs_replay_vs_team_vs_recover_vs_branch_vs_contradict_v12",
    "PrefixStateV12Witness",
    "emit_prefix_state_v12_witness",
]
