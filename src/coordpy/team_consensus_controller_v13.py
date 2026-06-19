"""W78 — Team-Consensus Controller V13.

Strictly extends W77's ``coordpy.team_consensus_controller_v12``.
The V13 controller adds:

* **long-horizon-reconstruction-pressure arbiter** — when caller-
  declared long-horizon-reconstruction pressure crosses the V13
  floor and substrate-replay trust ≥ floor, the V13 controller
  picks the weighted-mean of confidence-aligned agents whose
  long-horizon-reconstruction-recovery flag is set.
* **long-horizon-reconstruction-trajectory arbiter** — when
  caller-declared ``long_horizon_reconstruction_trajectory_cid``
  is non-empty AND the long-horizon-blackout-window crosses the
  floor, the V13 controller pulls toward the agent that absorbed
  the long-horizon-reconstruction horizon the cleanest.

V13 preserves V12's decision semantics byte-for-byte under the
W77 regimes (with a wrapping V13 audit ledger).

Honest scope (W78)
------------------

* ``W78-L-TEAM-CONSENSUS-V13-IN-REPO-CAP`` — operates on in-repo
  MASC V14 outcomes only.
* The thresholds are pre-committed constants; the controller is
  not autograd-trained.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.team_consensus_controller_v13 requires numpy"
        ) from exc

from .team_consensus_controller_v12 import (
    TeamConsensusControllerV12, W77_TC_V12_DECISIONS,
)
from .tiny_substrate_v3 import _sha256_hex


W78_TEAM_CONSENSUS_CONTROLLER_V13_SCHEMA_VERSION: str = (
    "coordpy.team_consensus_controller_v13.v1")

W78_TC_V13_DECISION_LONG_HORIZON_RECONSTRUCTION_PRESSURE: str = (
    "long_horizon_reconstruction_pressure_arbiter")
W78_TC_V13_DECISION_LONG_HORIZON_RECONSTRUCTION_TRAJECTORY: str = (
    "long_horizon_reconstruction_trajectory_arbiter")
W78_TC_V13_DECISIONS: tuple[str, ...] = (
    *W77_TC_V12_DECISIONS,
    W78_TC_V13_DECISION_LONG_HORIZON_RECONSTRUCTION_PRESSURE,
    W78_TC_V13_DECISION_LONG_HORIZON_RECONSTRUCTION_TRAJECTORY,
)


@dataclasses.dataclass
class TeamConsensusControllerV13:
    inner_v12: TeamConsensusControllerV12 = dataclasses.field(
        default_factory=TeamConsensusControllerV12)
    long_horizon_reconstruction_pressure_substrate_trust_floor: (
        float) = 0.5
    long_horizon_reconstruction_pressure_floor: float = 0.5
    long_horizon_reconstruction_trajectory_substrate_trust_floor: (
        float) = 0.5
    long_horizon_reconstruction_window_floor: int = 50
    audit_v13: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W78_TEAM_CONSENSUS_CONTROLLER_V13_SCHEMA_VERSION,
            "kind": "team_consensus_controller_v13",
            "inner_v12_cid": str(self.inner_v12.cid()),
            "long_horizon_reconstruction_pressure_substrate_trust_floor":
                float(round(
                    self
                    .long_horizon_reconstruction_pressure_substrate_trust_floor,
                    12)),
            "long_horizon_reconstruction_pressure_floor": float(
                round(
                    self
                    .long_horizon_reconstruction_pressure_floor,
                    12)),
            "long_horizon_reconstruction_trajectory_substrate_trust_floor":
                float(round(
                    self
                    .long_horizon_reconstruction_trajectory_substrate_trust_floor,
                    12)),
            "long_horizon_reconstruction_window_floor": int(
                self.long_horizon_reconstruction_window_floor),
        })

    def decide_v13(
            self, *,
            regime: str,
            agent_guesses: Sequence[float],
            agent_confidences: Sequence[float],
            substrate_replay_trust: float = 0.0,
            long_horizon_reconstruction_pressure: float = 0.0,
            agent_long_horizon_reconstruction_recovery_flags: (
                Sequence[int] | None) = None,
            long_horizon_reconstruction_trajectory_cid: str = "",
            long_horizon_blackout_window_turns: int = 0,
            agent_long_horizon_reconstruction_absorption_scores: (
                Sequence[float] | None) = None,
            **v12_kwargs: Any,
    ) -> dict[str, Any]:
        """V13 decision routine.

        Routes through V12 for W77 regimes. For the W78
        long-horizon-reconstruction regime V13 applies a
        coordinated arbiter on top.
        """
        # V13 long-horizon-reconstruction-trajectory arbiter.
        if (str(long_horizon_reconstruction_trajectory_cid)
                and int(long_horizon_blackout_window_turns)
                >= int(
                    self.long_horizon_reconstruction_window_floor)
                and agent_long_horizon_reconstruction_absorption_scores
                is not None
                and len(
                    agent_long_horizon_reconstruction_absorption_scores)
                == len(agent_guesses)
                and float(substrate_replay_trust)
                >= float(
                    self
                    .long_horizon_reconstruction_trajectory_substrate_trust_floor)):
            scores = _np.asarray(
                list(
                    agent_long_horizon_reconstruction_absorption_scores),
                dtype=_np.float64)
            gs = _np.asarray(
                list(agent_guesses), dtype=_np.float64)
            denom = float(scores.sum())
            if denom > 0.0:
                replaced = float(_np.dot(gs, scores) / denom)
            else:
                replaced = float(gs.mean())
            self.audit_v13.append({
                "decision":
                    W78_TC_V13_DECISION_LONG_HORIZON_RECONSTRUCTION_TRAJECTORY,
                "regime": str(regime),
                "long_horizon_blackout_window_turns": int(
                    long_horizon_blackout_window_turns),
            })
            return {
                "decision":
                    W78_TC_V13_DECISION_LONG_HORIZON_RECONSTRUCTION_TRAJECTORY,
                "payload": float(replaced),
                "regime": str(regime),
                "rationale":
                    "long_horizon_reconstruction_trajectory_"
                    "arbiter_applied",
                "long_horizon_blackout_window_turns": int(
                    long_horizon_blackout_window_turns),
            }
        # V13 long-horizon-reconstruction-pressure arbiter.
        if (float(long_horizon_reconstruction_pressure)
                >= float(
                    self
                    .long_horizon_reconstruction_pressure_floor)
                and agent_long_horizon_reconstruction_recovery_flags
                is not None
                and len(
                    agent_long_horizon_reconstruction_recovery_flags)
                == len(agent_guesses)
                and float(substrate_replay_trust)
                >= float(
                    self
                    .long_horizon_reconstruction_pressure_substrate_trust_floor)):
            matching_idx = [
                i for i, flag in enumerate(
                    agent_long_horizon_reconstruction_recovery_flags)
                if int(flag) > 0]
            if matching_idx:
                gs = _np.asarray(
                    [agent_guesses[i]
                     for i in matching_idx],
                    dtype=_np.float64)
                cs = _np.asarray(
                    [agent_confidences[i]
                     for i in matching_idx],
                    dtype=_np.float64)
                denom = float(cs.sum())
                if denom > 0.0:
                    replaced = float(_np.dot(gs, cs) / denom)
                else:
                    replaced = float(gs.mean())
                self.audit_v13.append({
                    "decision":
                        W78_TC_V13_DECISION_LONG_HORIZON_RECONSTRUCTION_PRESSURE,
                    "regime": str(regime),
                    "n_matching_agents": int(
                        len(matching_idx)),
                    "long_horizon_reconstruction_pressure": float(
                        long_horizon_reconstruction_pressure),
                })
                return {
                    "decision":
                        W78_TC_V13_DECISION_LONG_HORIZON_RECONSTRUCTION_PRESSURE,
                    "payload": float(replaced),
                    "regime": str(regime),
                    "rationale":
                        "long_horizon_reconstruction_pressure_"
                        "arbiter_applied",
                    "long_horizon_reconstruction_pressure": float(
                        long_horizon_reconstruction_pressure),
                }
        # Fall back to V12.
        v12_decision = self.inner_v12.decide_v12(
            regime=str(regime),
            agent_guesses=list(agent_guesses),
            agent_confidences=list(agent_confidences),
            substrate_replay_trust=float(
                substrate_replay_trust),
            **v12_kwargs)
        self.audit_v13.append({
            "decision": str(v12_decision.get("decision", "")),
            "regime": str(regime),
            "v12_fallback": True,
        })
        return dict(v12_decision)


@dataclasses.dataclass(frozen=True)
class TeamConsensusControllerV13Witness:
    schema: str
    controller_v13_cid: str
    inner_v12_witness_cid: str
    n_decisions_v13: int
    decisions_summary_v13: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_v13_cid": str(self.controller_v13_cid),
            "inner_v12_witness_cid": str(
                self.inner_v12_witness_cid),
            "n_decisions_v13": int(self.n_decisions_v13),
            "decisions_summary_v13": {
                k: int(v) for k, v in sorted(
                    self.decisions_summary_v13.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "team_consensus_controller_v13_witness",
            "witness": self.to_dict()})


def emit_team_consensus_controller_v13_witness(
        controller: TeamConsensusControllerV13,
) -> TeamConsensusControllerV13Witness:
    from .team_consensus_controller_v12 import (
        emit_team_consensus_controller_v12_witness,
    )
    inner_w = emit_team_consensus_controller_v12_witness(
        controller.inner_v12)
    summary: dict[str, int] = {
        d: 0 for d in W78_TC_V13_DECISIONS}
    for entry in controller.audit_v13:
        dec = str(entry.get("decision", ""))
        if dec in summary:
            summary[dec] += 1
    return TeamConsensusControllerV13Witness(
        schema=W78_TEAM_CONSENSUS_CONTROLLER_V13_SCHEMA_VERSION,
        controller_v13_cid=str(controller.cid()),
        inner_v12_witness_cid=str(inner_w.cid()),
        n_decisions_v13=int(len(controller.audit_v13)),
        decisions_summary_v13=dict(summary),
    )


__all__ = [
    "W78_TEAM_CONSENSUS_CONTROLLER_V13_SCHEMA_VERSION",
    "W78_TC_V13_DECISION_LONG_HORIZON_RECONSTRUCTION_PRESSURE",
    "W78_TC_V13_DECISION_LONG_HORIZON_RECONSTRUCTION_TRAJECTORY",
    "W78_TC_V13_DECISIONS",
    "TeamConsensusControllerV13",
    "TeamConsensusControllerV13Witness",
    "emit_team_consensus_controller_v13_witness",
]
