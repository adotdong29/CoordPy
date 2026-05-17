"""W72 M12 — Team-Consensus Controller V7.

Strictly extends W71's ``coordpy.team_consensus_controller_v6``.
The V7 controller adds:

* **rejoin-pressure arbiter** — when caller-declared rejoin
  pressure crosses the V7 rejoin-floor and substrate-replay trust
  ≥ floor, the V7 controller picks the weighted-mean of
  confidence-aligned agents whose rejoin-recovery flag is set.
* **delayed-rejoin-after-restart arbiter** — when caller-declared
  ``restart_repair_trajectory_cid`` is non-empty AND the rejoin-
  lag window crosses the floor, the V7 controller pulls toward the
  agent that absorbed the rejoin-lag the cleanest.
* **per-regime thresholds for the W72 regime** — adds
  ``delayed_rejoin_after_restart_under_budget`` to the V6 eleven
  W71 regimes.

V7 preserves V6's decision semantics byte-for-byte under the W71
regimes (with a wrapping V7 audit ledger).

Honest scope (W72)
------------------

* ``W72-L-TEAM-CONSENSUS-V7-IN-REPO-CAP`` — operates on in-repo
  MASC V8 outcomes only.
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
        "coordpy.team_consensus_controller_v7 requires numpy"
        ) from exc

from .multi_agent_substrate_coordinator_v8 import (
    W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET,
    W72_MASC_V8_REGIMES,
)
from .team_consensus_controller_v6 import (
    TeamConsensusControllerV6, W71_TC_V6_DECISIONS,
)
from .tiny_substrate_v3 import _sha256_hex


W72_TEAM_CONSENSUS_CONTROLLER_V7_SCHEMA_VERSION: str = (
    "coordpy.team_consensus_controller_v7.v1")

W72_TC_V7_DECISION_REJOIN_PRESSURE: str = (
    "rejoin_pressure_arbiter")
W72_TC_V7_DECISION_DELAYED_REJOIN: str = (
    "delayed_rejoin_after_restart_arbiter")
W72_TC_V7_DECISIONS: tuple[str, ...] = (
    *W71_TC_V6_DECISIONS,
    W72_TC_V7_DECISION_REJOIN_PRESSURE,
    W72_TC_V7_DECISION_DELAYED_REJOIN,
)


@dataclasses.dataclass
class TeamConsensusControllerV7:
    inner_v6: TeamConsensusControllerV6 = dataclasses.field(
        default_factory=TeamConsensusControllerV6)
    rejoin_pressure_substrate_trust_floor: float = 0.5
    rejoin_pressure_floor: float = 0.5
    delayed_rejoin_substrate_trust_floor: float = 0.5
    rejoin_lag_floor: int = 1
    audit_v7: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W72_TEAM_CONSENSUS_CONTROLLER_V7_SCHEMA_VERSION,
            "kind": "team_consensus_controller_v7",
            "inner_v6_cid": str(self.inner_v6.cid()),
            "rejoin_pressure_substrate_trust_floor": float(round(
                self.rejoin_pressure_substrate_trust_floor, 12)),
            "rejoin_pressure_floor": float(round(
                self.rejoin_pressure_floor, 12)),
            "delayed_rejoin_substrate_trust_floor": float(round(
                self.delayed_rejoin_substrate_trust_floor, 12)),
            "rejoin_lag_floor": int(self.rejoin_lag_floor),
        })

    def decide_v7(
            self, *,
            regime: str,
            agent_guesses: Sequence[float],
            agent_confidences: Sequence[float],
            substrate_replay_trust: float = 0.0,
            transcript_available: bool = False,
            transcript_trust: float = 0.0,
            visible_token_budget_ratio: float = 1.0,
            dominant_repair_label: int = 0,
            agent_repair_labels: (
                Sequence[int] | None) = None,
            branch_assignments: (
                Sequence[int] | None) = None,
            corrupted_role_indices: (
                Sequence[int] | None) = None,
            restart_pressure: float = 0.0,
            agent_restart_recovery_flags: (
                Sequence[int] | None) = None,
            delayed_repair_trajectory_cid: str = "",
            delay_turns: int = 0,
            agent_delay_absorption_scores: (
                Sequence[float] | None) = None,
            rejoin_pressure: float = 0.0,
            agent_rejoin_recovery_flags: (
                Sequence[int] | None) = None,
            restart_repair_trajectory_cid: str = "",
            rejoin_lag_turns: int = 0,
            agent_rejoin_absorption_scores: (
                Sequence[float] | None) = None,
            **v6_kwargs: Any,
    ) -> dict[str, Any]:
        """V7 decision routine.

        Routes through V6 for W71 regimes (and the V7 rejoin-
        pressure / delayed-rejoin arbiters fire on top when their
        conditions hold). For the W72-only compound regime V7
        applies a coordinated arbiter.
        """
        if regime not in W72_MASC_V8_REGIMES:
            raise ValueError(f"unknown regime {regime!r}")
        # W72 compound regime — coordinated delayed-rejoin arbiter.
        if (regime == (
                W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET)
                and agent_rejoin_absorption_scores is not None
                and len(agent_rejoin_absorption_scores)
                    == len(agent_guesses)
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .delayed_rejoin_substrate_trust_floor)):
            scores = _np.asarray(
                list(agent_rejoin_absorption_scores),
                dtype=_np.float64)
            gs = _np.asarray(
                list(agent_guesses), dtype=_np.float64)
            denom = float(scores.sum())
            if denom > 0.0:
                rejoined = float(_np.dot(gs, scores) / denom)
            else:
                rejoined = float(gs.mean())
            self.audit_v7.append({
                "decision":
                    W72_TC_V7_DECISION_DELAYED_REJOIN,
                "regime": str(regime),
                "n_agents": int(len(agent_guesses)),
                "rejoin_lag_turns": int(rejoin_lag_turns),
            })
            return {
                "decision":
                    W72_TC_V7_DECISION_DELAYED_REJOIN,
                "payload": float(rejoined),
                "regime": str(regime),
                "rationale":
                    "delayed_rejoin_after_restart_arbiter_"
                    "applied",
            }
        # V7 rejoin-pressure arbiter (any regime + high rejoin
        # pressure).
        if (float(rejoin_pressure)
                >= float(self.rejoin_pressure_floor)
                and agent_rejoin_recovery_flags is not None
                and len(agent_rejoin_recovery_flags)
                    == len(agent_guesses)
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .rejoin_pressure_substrate_trust_floor)):
            matching_idx = [
                i for i, flag in enumerate(
                    agent_rejoin_recovery_flags)
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
                    rejoined = float(_np.dot(gs, cs) / denom)
                else:
                    rejoined = float(gs.mean())
                self.audit_v7.append({
                    "decision":
                        W72_TC_V7_DECISION_REJOIN_PRESSURE,
                    "regime": str(regime),
                    "n_matching_agents": int(
                        len(matching_idx)),
                    "rejoin_pressure": float(rejoin_pressure),
                })
                return {
                    "decision":
                        W72_TC_V7_DECISION_REJOIN_PRESSURE,
                    "payload": float(rejoined),
                    "regime": str(regime),
                    "rationale":
                        "rejoin_pressure_arbiter_applied",
                    "rejoin_pressure": float(rejoin_pressure),
                }
        # V7 delayed-rejoin arbiter (any regime + non-empty
        # restart-repair-trajectory CID + rejoin-lag > floor).
        if (str(restart_repair_trajectory_cid)
                and int(rejoin_lag_turns)
                    >= int(self.rejoin_lag_floor)
                and agent_rejoin_absorption_scores is not None
                and len(agent_rejoin_absorption_scores)
                    == len(agent_guesses)
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .delayed_rejoin_substrate_trust_floor)):
            scores = _np.asarray(
                list(agent_rejoin_absorption_scores),
                dtype=_np.float64)
            gs = _np.asarray(
                list(agent_guesses), dtype=_np.float64)
            denom = float(scores.sum())
            if denom > 0.0:
                rejoined = float(_np.dot(gs, scores) / denom)
            else:
                rejoined = float(gs.mean())
            self.audit_v7.append({
                "decision":
                    W72_TC_V7_DECISION_DELAYED_REJOIN,
                "regime": str(regime),
                "rejoin_lag_turns": int(rejoin_lag_turns),
            })
            return {
                "decision":
                    W72_TC_V7_DECISION_DELAYED_REJOIN,
                "payload": float(rejoined),
                "regime": str(regime),
                "rationale":
                    "delayed_rejoin_after_restart_arbiter_"
                    "applied",
                "rejoin_lag_turns": int(rejoin_lag_turns),
            }
        # Fall back to V6.
        if regime in (
                W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET,):
            v6_decision = self.inner_v6.decide_v6(
                regime="baseline",
                agent_guesses=list(agent_guesses),
                agent_confidences=list(agent_confidences),
                substrate_replay_trust=float(
                    substrate_replay_trust),
                transcript_available=bool(transcript_available),
                transcript_trust=float(transcript_trust),
                visible_token_budget_ratio=float(
                    visible_token_budget_ratio),
                dominant_repair_label=int(
                    dominant_repair_label),
                agent_repair_labels=(
                    list(agent_repair_labels)
                    if agent_repair_labels is not None
                    else None),
                branch_assignments=(
                    list(branch_assignments)
                    if branch_assignments is not None else None),
                corrupted_role_indices=(
                    list(corrupted_role_indices)
                    if corrupted_role_indices is not None
                    else None),
                restart_pressure=float(restart_pressure),
                agent_restart_recovery_flags=(
                    list(agent_restart_recovery_flags)
                    if agent_restart_recovery_flags is not None
                    else None),
                delayed_repair_trajectory_cid=str(
                    delayed_repair_trajectory_cid),
                delay_turns=int(delay_turns),
                agent_delay_absorption_scores=(
                    list(agent_delay_absorption_scores)
                    if agent_delay_absorption_scores is not None
                    else None),
                **v6_kwargs)
        else:
            v6_decision = self.inner_v6.decide_v6(
                regime=str(regime),
                agent_guesses=list(agent_guesses),
                agent_confidences=list(agent_confidences),
                substrate_replay_trust=float(
                    substrate_replay_trust),
                transcript_available=bool(transcript_available),
                transcript_trust=float(transcript_trust),
                visible_token_budget_ratio=float(
                    visible_token_budget_ratio),
                dominant_repair_label=int(
                    dominant_repair_label),
                agent_repair_labels=(
                    list(agent_repair_labels)
                    if agent_repair_labels is not None
                    else None),
                branch_assignments=(
                    list(branch_assignments)
                    if branch_assignments is not None else None),
                corrupted_role_indices=(
                    list(corrupted_role_indices)
                    if corrupted_role_indices is not None
                    else None),
                restart_pressure=float(restart_pressure),
                agent_restart_recovery_flags=(
                    list(agent_restart_recovery_flags)
                    if agent_restart_recovery_flags is not None
                    else None),
                delayed_repair_trajectory_cid=str(
                    delayed_repair_trajectory_cid),
                delay_turns=int(delay_turns),
                agent_delay_absorption_scores=(
                    list(agent_delay_absorption_scores)
                    if agent_delay_absorption_scores is not None
                    else None),
                **v6_kwargs)
        self.audit_v7.append({
            "decision": str(v6_decision.get("decision", "")),
            "regime": str(regime),
            "v6_fallback": True,
        })
        return dict(v6_decision)


@dataclasses.dataclass(frozen=True)
class TeamConsensusControllerV7Witness:
    schema: str
    controller_v7_cid: str
    inner_v6_witness_cid: str
    n_decisions_v7: int
    decisions_summary_v7: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_v7_cid": str(self.controller_v7_cid),
            "inner_v6_witness_cid": str(
                self.inner_v6_witness_cid),
            "n_decisions_v7": int(self.n_decisions_v7),
            "decisions_summary_v7": {
                k: int(v) for k, v in sorted(
                    self.decisions_summary_v7.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "team_consensus_controller_v7_witness",
            "witness": self.to_dict()})


def emit_team_consensus_controller_v7_witness(
        controller: TeamConsensusControllerV7,
) -> TeamConsensusControllerV7Witness:
    from .team_consensus_controller_v6 import (
        emit_team_consensus_controller_v6_witness,
    )
    inner_w = emit_team_consensus_controller_v6_witness(
        controller.inner_v6)
    summary: dict[str, int] = {
        d: 0 for d in W72_TC_V7_DECISIONS}
    for entry in controller.audit_v7:
        dec = str(entry.get("decision", ""))
        if dec in summary:
            summary[dec] += 1
    return TeamConsensusControllerV7Witness(
        schema=W72_TEAM_CONSENSUS_CONTROLLER_V7_SCHEMA_VERSION,
        controller_v7_cid=str(controller.cid()),
        inner_v6_witness_cid=str(inner_w.cid()),
        n_decisions_v7=int(len(controller.audit_v7)),
        decisions_summary_v7=dict(summary),
    )


__all__ = [
    "W72_TEAM_CONSENSUS_CONTROLLER_V7_SCHEMA_VERSION",
    "W72_TC_V7_DECISION_REJOIN_PRESSURE",
    "W72_TC_V7_DECISION_DELAYED_REJOIN",
    "W72_TC_V7_DECISIONS",
    "TeamConsensusControllerV7",
    "TeamConsensusControllerV7Witness",
    "emit_team_consensus_controller_v7_witness",
]
