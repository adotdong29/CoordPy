"""W71 M12 — Team-Consensus Controller V6.

Strictly extends W70's ``coordpy.team_consensus_controller_v5``.
The V6 controller adds:

* **restart-aware arbiter** — when caller-declared restart pressure
  crosses the V6 restart-floor and substrate-replay trust ≥ floor,
  the V6 controller picks the weighted-mean of confidence-aligned
  agents whose restart-recovery flag is set.
* **delayed-repair-after-restart arbiter** — when caller-declared
  ``delayed_repair_trajectory_cid`` is non-empty AND the delay
  window crosses the floor, the V6 controller pulls toward the
  agent that absorbed the delay window the cleanest.
* **per-regime thresholds for the W71 regime** — adds
  ``delayed_repair_after_restart`` to the V5 ten W70 regimes.

V6 preserves V5's decision semantics byte-for-byte under the W70
regimes (with a wrapping V6 audit ledger).

Honest scope (W71)
------------------

* ``W71-L-TEAM-CONSENSUS-V6-IN-REPO-CAP`` — operates on in-repo
  MASC V7 outcomes only.
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
        "coordpy.team_consensus_controller_v6 requires numpy"
        ) from exc

from .multi_agent_substrate_coordinator_v7 import (
    W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART,
    W71_MASC_V7_REGIMES,
)
from .team_consensus_controller_v5 import (
    TeamConsensusControllerV5, W70_TC_V5_DECISIONS,
)
from .tiny_substrate_v3 import _sha256_hex


W71_TEAM_CONSENSUS_CONTROLLER_V6_SCHEMA_VERSION: str = (
    "coordpy.team_consensus_controller_v6.v1")

W71_TC_V6_DECISION_RESTART_AWARE: str = (
    "restart_aware_arbiter")
W71_TC_V6_DECISION_DELAYED_REPAIR: str = (
    "delayed_repair_after_restart_arbiter")
W71_TC_V6_DECISIONS: tuple[str, ...] = (
    *W70_TC_V5_DECISIONS,
    W71_TC_V6_DECISION_RESTART_AWARE,
    W71_TC_V6_DECISION_DELAYED_REPAIR,
)


@dataclasses.dataclass
class TeamConsensusControllerV6:
    inner_v5: TeamConsensusControllerV5 = dataclasses.field(
        default_factory=TeamConsensusControllerV5)
    restart_aware_substrate_trust_floor: float = 0.5
    restart_pressure_floor: float = 0.5
    delayed_repair_substrate_trust_floor: float = 0.5
    delayed_repair_delay_floor: int = 1
    audit_v6: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W71_TEAM_CONSENSUS_CONTROLLER_V6_SCHEMA_VERSION,
            "kind": "team_consensus_controller_v6",
            "inner_v5_cid": str(self.inner_v5.cid()),
            "restart_aware_substrate_trust_floor": float(round(
                self.restart_aware_substrate_trust_floor, 12)),
            "restart_pressure_floor": float(round(
                self.restart_pressure_floor, 12)),
            "delayed_repair_substrate_trust_floor": float(round(
                self.delayed_repair_substrate_trust_floor, 12)),
            "delayed_repair_delay_floor": int(
                self.delayed_repair_delay_floor),
        })

    def decide_v6(
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
            **v5_kwargs: Any,
    ) -> dict[str, Any]:
        """V6 decision routine.

        Routes through V5 for W70 regimes (and the V6 restart-aware
        / delayed-repair arbiters fire on top when their conditions
        hold). For the W71-only compound regime V6 applies a
        coordinated arbiter.
        """
        if regime not in W71_MASC_V7_REGIMES:
            raise ValueError(f"unknown regime {regime!r}")
        # W71 compound regime — coordinated delayed-repair arbiter.
        if (regime == (
                W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART)
                and agent_delay_absorption_scores is not None
                and len(agent_delay_absorption_scores)
                    == len(agent_guesses)
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .delayed_repair_substrate_trust_floor)):
            scores = _np.asarray(
                list(agent_delay_absorption_scores),
                dtype=_np.float64)
            gs = _np.asarray(
                list(agent_guesses), dtype=_np.float64)
            denom = float(scores.sum())
            if denom > 0.0:
                repaired = float(_np.dot(gs, scores) / denom)
            else:
                repaired = float(gs.mean())
            self.audit_v6.append({
                "decision":
                    W71_TC_V6_DECISION_DELAYED_REPAIR,
                "regime": str(regime),
                "n_agents": int(len(agent_guesses)),
                "delay_turns": int(delay_turns),
            })
            return {
                "decision":
                    W71_TC_V6_DECISION_DELAYED_REPAIR,
                "payload": float(repaired),
                "regime": str(regime),
                "rationale":
                    "delayed_repair_after_restart_arbiter_"
                    "applied",
            }
        # V6 restart-aware arbiter (any regime + high restart
        # pressure).
        if (float(restart_pressure)
                >= float(self.restart_pressure_floor)
                and agent_restart_recovery_flags is not None
                and len(agent_restart_recovery_flags)
                    == len(agent_guesses)
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .restart_aware_substrate_trust_floor)):
            matching_idx = [
                i for i, flag in enumerate(
                    agent_restart_recovery_flags)
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
                    repaired = float(_np.dot(gs, cs) / denom)
                else:
                    repaired = float(gs.mean())
                self.audit_v6.append({
                    "decision":
                        W71_TC_V6_DECISION_RESTART_AWARE,
                    "regime": str(regime),
                    "n_matching_agents": int(
                        len(matching_idx)),
                    "restart_pressure": float(
                        restart_pressure),
                })
                return {
                    "decision":
                        W71_TC_V6_DECISION_RESTART_AWARE,
                    "payload": float(repaired),
                    "regime": str(regime),
                    "rationale":
                        "restart_aware_arbiter_applied",
                    "restart_pressure": float(
                        restart_pressure),
                }
        # V6 delayed-repair arbiter (any regime + non-empty
        # delayed-repair-trajectory CID + delay > floor).
        if (str(delayed_repair_trajectory_cid)
                and int(delay_turns)
                    >= int(self.delayed_repair_delay_floor)
                and agent_delay_absorption_scores is not None
                and len(agent_delay_absorption_scores)
                    == len(agent_guesses)
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .delayed_repair_substrate_trust_floor)):
            scores = _np.asarray(
                list(agent_delay_absorption_scores),
                dtype=_np.float64)
            gs = _np.asarray(
                list(agent_guesses), dtype=_np.float64)
            denom = float(scores.sum())
            if denom > 0.0:
                repaired = float(_np.dot(gs, scores) / denom)
            else:
                repaired = float(gs.mean())
            self.audit_v6.append({
                "decision":
                    W71_TC_V6_DECISION_DELAYED_REPAIR,
                "regime": str(regime),
                "delay_turns": int(delay_turns),
            })
            return {
                "decision":
                    W71_TC_V6_DECISION_DELAYED_REPAIR,
                "payload": float(repaired),
                "regime": str(regime),
                "rationale":
                    "delayed_repair_after_restart_arbiter_"
                    "applied",
                "delay_turns": int(delay_turns),
            }
        # Fall back to V5.
        if regime in (
                W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART,):
            v5_decision = self.inner_v5.decide_v5(
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
                **v5_kwargs)
        else:
            v5_decision = self.inner_v5.decide_v5(
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
                **v5_kwargs)
        self.audit_v6.append({
            "decision": str(v5_decision.get("decision", "")),
            "regime": str(regime),
            "v5_fallback": True,
        })
        return dict(v5_decision)


@dataclasses.dataclass(frozen=True)
class TeamConsensusControllerV6Witness:
    schema: str
    controller_v6_cid: str
    inner_v5_witness_cid: str
    n_decisions_v6: int
    decisions_summary_v6: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_v6_cid": str(self.controller_v6_cid),
            "inner_v5_witness_cid": str(
                self.inner_v5_witness_cid),
            "n_decisions_v6": int(self.n_decisions_v6),
            "decisions_summary_v6": {
                k: int(v) for k, v in sorted(
                    self.decisions_summary_v6.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "team_consensus_controller_v6_witness",
            "witness": self.to_dict()})


def emit_team_consensus_controller_v6_witness(
        controller: TeamConsensusControllerV6,
) -> TeamConsensusControllerV6Witness:
    from .team_consensus_controller_v5 import (
        emit_team_consensus_controller_v5_witness,
    )
    inner_w = emit_team_consensus_controller_v5_witness(
        controller.inner_v5)
    summary: dict[str, int] = {
        d: 0 for d in W71_TC_V6_DECISIONS}
    for entry in controller.audit_v6:
        dec = str(entry.get("decision", ""))
        if dec in summary:
            summary[dec] += 1
    return TeamConsensusControllerV6Witness(
        schema=W71_TEAM_CONSENSUS_CONTROLLER_V6_SCHEMA_VERSION,
        controller_v6_cid=str(controller.cid()),
        inner_v5_witness_cid=str(inner_w.cid()),
        n_decisions_v6=int(len(controller.audit_v6)),
        decisions_summary_v6=dict(summary),
    )


__all__ = [
    "W71_TEAM_CONSENSUS_CONTROLLER_V6_SCHEMA_VERSION",
    "W71_TC_V6_DECISION_RESTART_AWARE",
    "W71_TC_V6_DECISION_DELAYED_REPAIR",
    "W71_TC_V6_DECISIONS",
    "TeamConsensusControllerV6",
    "TeamConsensusControllerV6Witness",
    "emit_team_consensus_controller_v6_witness",
]
