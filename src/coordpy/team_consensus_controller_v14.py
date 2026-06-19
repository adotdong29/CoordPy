"""W79 — Team-Consensus Controller V14.

Strictly extends W78's ``coordpy.team_consensus_controller_v13``.
Adds:

* ``replacement_then_restart_after_long_delay_pressure_arbiter``
  — fires when caller-declared replacement-then-restart-after-
  long-delay pressure crosses the V14 floor and substrate-replay
  trust ≥ floor.
* ``replacement_then_restart_after_long_delay_trajectory_arbiter``
  — fires when caller-declared
  ``replacement_then_restart_after_long_delay_trajectory_cid`` is
  non-empty AND the long-delay blackout window crosses the V14
  floor.
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
        "coordpy.team_consensus_controller_v14 requires numpy"
        ) from exc

from .team_consensus_controller_v13 import (
    TeamConsensusControllerV13, W78_TC_V13_DECISIONS,
)


W79_TEAM_CONSENSUS_CONTROLLER_V14_SCHEMA_VERSION: str = (
    "coordpy.team_consensus_controller_v14.v1")

W79_TC_V14_DECISION_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_PRESSURE: (
    str) = (
    "replacement_then_restart_after_long_delay_pressure_arbiter")
W79_TC_V14_DECISION_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_TRAJECTORY: (
    str) = (
    "replacement_then_restart_after_long_delay_trajectory_arbiter")
W79_TC_V14_DECISIONS: tuple[str, ...] = (
    *W78_TC_V13_DECISIONS,
    W79_TC_V14_DECISION_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_PRESSURE,
    W79_TC_V14_DECISION_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_TRAJECTORY,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class TeamConsensusControllerV14:
    inner_v13: TeamConsensusControllerV13 = dataclasses.field(
        default_factory=TeamConsensusControllerV13)
    replacement_then_restart_after_long_delay_pressure_substrate_trust_floor: (
        float) = 0.5
    replacement_then_restart_after_long_delay_pressure_floor: (
        float) = 0.5
    replacement_then_restart_after_long_delay_trajectory_substrate_trust_floor: (
        float) = 0.5
    replacement_then_restart_after_long_delay_window_floor: int = 55
    audit_v14: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W79_TEAM_CONSENSUS_CONTROLLER_V14_SCHEMA_VERSION,
            "kind": "team_consensus_controller_v14",
            "inner_v13_cid": str(self.inner_v13.cid()),
            "replacement_then_restart_after_long_delay_pressure_substrate_trust_floor": float(
                round(
                    self
                    .replacement_then_restart_after_long_delay_pressure_substrate_trust_floor,
                    12)),
            "replacement_then_restart_after_long_delay_pressure_floor": float(
                round(
                    self
                    .replacement_then_restart_after_long_delay_pressure_floor,
                    12)),
            "replacement_then_restart_after_long_delay_trajectory_substrate_trust_floor": float(
                round(
                    self
                    .replacement_then_restart_after_long_delay_trajectory_substrate_trust_floor,
                    12)),
            "replacement_then_restart_after_long_delay_window_floor": int(
                self
                .replacement_then_restart_after_long_delay_window_floor),
        })

    def decide_v14(
            self, *,
            regime: str,
            agent_guesses: Sequence[float],
            agent_confidences: Sequence[float],
            substrate_replay_trust: float = 0.0,
            replacement_then_restart_after_long_delay_pressure: float = (
                0.0),
            agent_replacement_then_restart_after_long_delay_recovery_flags: (
                Sequence[int] | None) = None,
            replacement_then_restart_after_long_delay_trajectory_cid: (
                str) = "",
            long_delay_blackout_window_turns: int = 0,
            agent_replacement_then_restart_after_long_delay_absorption_scores: (
                Sequence[float] | None) = None,
            **v13_kwargs: Any,
    ) -> dict[str, Any]:
        """V14 decision routine. Falls back to V13 for everything
        else."""
        # V14 trajectory arbiter.
        if (str(replacement_then_restart_after_long_delay_trajectory_cid)
                and int(long_delay_blackout_window_turns)
                >= int(
                    self
                    .replacement_then_restart_after_long_delay_window_floor)
                and agent_replacement_then_restart_after_long_delay_absorption_scores
                is not None
                and len(
                    agent_replacement_then_restart_after_long_delay_absorption_scores)
                == len(agent_guesses)
                and float(substrate_replay_trust)
                >= float(
                    self
                    .replacement_then_restart_after_long_delay_trajectory_substrate_trust_floor)):
            scores = _np.asarray(
                list(
                    agent_replacement_then_restart_after_long_delay_absorption_scores),
                dtype=_np.float64)
            gs = _np.asarray(
                list(agent_guesses), dtype=_np.float64)
            denom = float(scores.sum())
            if denom > 0.0:
                replaced = float(_np.dot(gs, scores) / denom)
            else:
                replaced = float(gs.mean())
            self.audit_v14.append({
                "decision":
                    W79_TC_V14_DECISION_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_TRAJECTORY,
                "regime": str(regime),
                "long_delay_blackout_window_turns": int(
                    long_delay_blackout_window_turns),
            })
            return {
                "decision":
                    W79_TC_V14_DECISION_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_TRAJECTORY,
                "payload": float(replaced),
                "regime": str(regime),
                "rationale":
                    "replacement_then_restart_after_long_delay_trajectory_arbiter_applied",
            }
        # V14 pressure arbiter.
        if (float(replacement_then_restart_after_long_delay_pressure)
                >= float(
                    self
                    .replacement_then_restart_after_long_delay_pressure_floor)
                and agent_replacement_then_restart_after_long_delay_recovery_flags
                is not None
                and len(
                    agent_replacement_then_restart_after_long_delay_recovery_flags)
                == len(agent_guesses)
                and float(substrate_replay_trust)
                >= float(
                    self
                    .replacement_then_restart_after_long_delay_pressure_substrate_trust_floor)):
            matching_idx = [
                i for i, flag in enumerate(
                    agent_replacement_then_restart_after_long_delay_recovery_flags)
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
                self.audit_v14.append({
                    "decision":
                        W79_TC_V14_DECISION_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_PRESSURE,
                    "regime": str(regime),
                    "n_matching_agents": int(
                        len(matching_idx)),
                })
                return {
                    "decision":
                        W79_TC_V14_DECISION_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_PRESSURE,
                    "payload": float(replaced),
                    "regime": str(regime),
                    "rationale":
                        "replacement_then_restart_after_long_delay_pressure_arbiter_applied",
                }
        v13_decision = self.inner_v13.decide_v13(
            regime=str(regime),
            agent_guesses=list(agent_guesses),
            agent_confidences=list(agent_confidences),
            substrate_replay_trust=float(
                substrate_replay_trust),
            **v13_kwargs)
        self.audit_v14.append({
            "decision": str(v13_decision.get("decision", "")),
            "regime": str(regime),
            "v13_fallback": True,
        })
        return dict(v13_decision)


@dataclasses.dataclass(frozen=True)
class TeamConsensusControllerV14Witness:
    schema: str
    controller_v14_cid: str
    inner_v13_witness_cid: str
    n_decisions_v14: int
    decisions_summary_v14: dict[str, int]

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "team_consensus_controller_v14_witness",
            "schema": str(self.schema),
            "controller_v14_cid": str(self.controller_v14_cid),
            "inner_v13_witness_cid": str(
                self.inner_v13_witness_cid),
            "n_decisions_v14": int(self.n_decisions_v14),
            "decisions_summary_v14": {
                k: int(v) for k, v in sorted(
                    self.decisions_summary_v14.items())},
        })


def emit_team_consensus_controller_v14_witness(
        controller: TeamConsensusControllerV14,
) -> TeamConsensusControllerV14Witness:
    from .team_consensus_controller_v13 import (
        emit_team_consensus_controller_v13_witness,
    )
    inner_w = emit_team_consensus_controller_v13_witness(
        controller.inner_v13)
    summary: dict[str, int] = {
        d: 0 for d in W79_TC_V14_DECISIONS}
    for entry in controller.audit_v14:
        dec = str(entry.get("decision", ""))
        if dec in summary:
            summary[dec] += 1
    return TeamConsensusControllerV14Witness(
        schema=W79_TEAM_CONSENSUS_CONTROLLER_V14_SCHEMA_VERSION,
        controller_v14_cid=str(controller.cid()),
        inner_v13_witness_cid=str(inner_w.cid()),
        n_decisions_v14=int(len(controller.audit_v14)),
        decisions_summary_v14=dict(summary),
    )


__all__ = [
    "W79_TEAM_CONSENSUS_CONTROLLER_V14_SCHEMA_VERSION",
    "W79_TC_V14_DECISION_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_PRESSURE",
    "W79_TC_V14_DECISION_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_TRAJECTORY",
    "W79_TC_V14_DECISIONS",
    "TeamConsensusControllerV14",
    "TeamConsensusControllerV14Witness",
    "emit_team_consensus_controller_v14_witness",
]
