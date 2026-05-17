"""W73 M12 — Team-Consensus Controller V8.

Strictly extends W72's ``coordpy.team_consensus_controller_v7``.
The V8 controller adds:

* **replacement-pressure arbiter** — when caller-declared
  replacement pressure crosses the V8 replacement-floor and
  substrate-replay trust ≥ floor, the V8 controller picks the
  weighted-mean of confidence-aligned agents whose replacement-
  recovery flag is set.
* **replacement-after-contradiction-then-rejoin arbiter** — when
  caller-declared ``replacement_repair_trajectory_cid`` is non-
  empty AND the replacement-lag window crosses the floor, the V8
  controller pulls toward the agent that absorbed the replacement-
  lag the cleanest.
* **per-regime thresholds for the W73 regime** — adds
  ``replacement_after_contradiction_then_rejoin`` to the V7
  twelve W72 regimes.

V8 preserves V7's decision semantics byte-for-byte under the W72
regimes (with a wrapping V8 audit ledger).

Honest scope (W73)
------------------

* ``W73-L-TEAM-CONSENSUS-V8-IN-REPO-CAP`` — operates on in-repo
  MASC V9 outcomes only.
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
        "coordpy.team_consensus_controller_v8 requires numpy"
        ) from exc

from .multi_agent_substrate_coordinator_v9 import (
    W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR,
    W73_MASC_V9_REGIMES,
)
from .team_consensus_controller_v7 import (
    TeamConsensusControllerV7, W72_TC_V7_DECISIONS,
)
from .tiny_substrate_v3 import _sha256_hex


W73_TEAM_CONSENSUS_CONTROLLER_V8_SCHEMA_VERSION: str = (
    "coordpy.team_consensus_controller_v8.v1")

W73_TC_V8_DECISION_REPLACEMENT_PRESSURE: str = (
    "replacement_pressure_arbiter")
W73_TC_V8_DECISION_REPLACEMENT_AFTER_CTR: str = (
    "replacement_after_contradiction_then_rejoin_arbiter")
W73_TC_V8_DECISIONS: tuple[str, ...] = (
    *W72_TC_V7_DECISIONS,
    W73_TC_V8_DECISION_REPLACEMENT_PRESSURE,
    W73_TC_V8_DECISION_REPLACEMENT_AFTER_CTR,
)


@dataclasses.dataclass
class TeamConsensusControllerV8:
    inner_v7: TeamConsensusControllerV7 = dataclasses.field(
        default_factory=TeamConsensusControllerV7)
    replacement_pressure_substrate_trust_floor: float = 0.5
    replacement_pressure_floor: float = 0.5
    replacement_after_ctr_substrate_trust_floor: float = 0.5
    replacement_lag_floor: int = 1
    audit_v8: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W73_TEAM_CONSENSUS_CONTROLLER_V8_SCHEMA_VERSION,
            "kind": "team_consensus_controller_v8",
            "inner_v7_cid": str(self.inner_v7.cid()),
            "replacement_pressure_substrate_trust_floor":
                float(round(
                    self
                    .replacement_pressure_substrate_trust_floor,
                    12)),
            "replacement_pressure_floor": float(round(
                self.replacement_pressure_floor, 12)),
            "replacement_after_ctr_substrate_trust_floor":
                float(round(
                    self
                    .replacement_after_ctr_substrate_trust_floor,
                    12)),
            "replacement_lag_floor": int(
                self.replacement_lag_floor),
        })

    def decide_v8(
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
            replacement_pressure: float = 0.0,
            agent_replacement_recovery_flags: (
                Sequence[int] | None) = None,
            replacement_repair_trajectory_cid: str = "",
            replacement_lag_turns: int = 0,
            agent_replacement_absorption_scores: (
                Sequence[float] | None) = None,
            **v7_kwargs: Any,
    ) -> dict[str, Any]:
        """V8 decision routine.

        Routes through V7 for W72 regimes (and the V8 replacement-
        pressure / replacement-after-CTR arbiters fire on top when
        their conditions hold). For the W73-only compound regime
        V8 applies a coordinated arbiter.
        """
        if regime not in W73_MASC_V9_REGIMES:
            raise ValueError(f"unknown regime {regime!r}")
        # W73 compound regime — coordinated replacement-after-CTR
        # arbiter.
        if (regime == W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR
                and agent_replacement_absorption_scores is not None
                and len(agent_replacement_absorption_scores)
                    == len(agent_guesses)
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .replacement_after_ctr_substrate_trust_floor)):
            scores = _np.asarray(
                list(agent_replacement_absorption_scores),
                dtype=_np.float64)
            gs = _np.asarray(
                list(agent_guesses), dtype=_np.float64)
            denom = float(scores.sum())
            if denom > 0.0:
                replaced = float(_np.dot(gs, scores) / denom)
            else:
                replaced = float(gs.mean())
            self.audit_v8.append({
                "decision":
                    W73_TC_V8_DECISION_REPLACEMENT_AFTER_CTR,
                "regime": str(regime),
                "n_agents": int(len(agent_guesses)),
                "replacement_lag_turns": int(
                    replacement_lag_turns),
            })
            return {
                "decision":
                    W73_TC_V8_DECISION_REPLACEMENT_AFTER_CTR,
                "payload": float(replaced),
                "regime": str(regime),
                "rationale":
                    "replacement_after_contradiction_then_rejoin"
                    "_arbiter_applied",
            }
        # V8 replacement-pressure arbiter (any regime + high
        # replacement pressure).
        if (float(replacement_pressure)
                >= float(self.replacement_pressure_floor)
                and agent_replacement_recovery_flags is not None
                and len(agent_replacement_recovery_flags)
                    == len(agent_guesses)
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .replacement_pressure_substrate_trust_floor)):
            matching_idx = [
                i for i, flag in enumerate(
                    agent_replacement_recovery_flags)
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
                self.audit_v8.append({
                    "decision":
                        W73_TC_V8_DECISION_REPLACEMENT_PRESSURE,
                    "regime": str(regime),
                    "n_matching_agents": int(
                        len(matching_idx)),
                    "replacement_pressure": float(
                        replacement_pressure),
                })
                return {
                    "decision":
                        W73_TC_V8_DECISION_REPLACEMENT_PRESSURE,
                    "payload": float(replaced),
                    "regime": str(regime),
                    "rationale":
                        "replacement_pressure_arbiter_applied",
                    "replacement_pressure": float(
                        replacement_pressure),
                }
        # V8 replacement-after-CTR arbiter (any regime + non-empty
        # replacement-repair-trajectory CID + replacement-lag
        # > floor).
        if (str(replacement_repair_trajectory_cid)
                and int(replacement_lag_turns)
                    >= int(self.replacement_lag_floor)
                and agent_replacement_absorption_scores is not None
                and len(agent_replacement_absorption_scores)
                    == len(agent_guesses)
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .replacement_after_ctr_substrate_trust_floor)):
            scores = _np.asarray(
                list(agent_replacement_absorption_scores),
                dtype=_np.float64)
            gs = _np.asarray(
                list(agent_guesses), dtype=_np.float64)
            denom = float(scores.sum())
            if denom > 0.0:
                replaced = float(_np.dot(gs, scores) / denom)
            else:
                replaced = float(gs.mean())
            self.audit_v8.append({
                "decision":
                    W73_TC_V8_DECISION_REPLACEMENT_AFTER_CTR,
                "regime": str(regime),
                "replacement_lag_turns": int(
                    replacement_lag_turns),
            })
            return {
                "decision":
                    W73_TC_V8_DECISION_REPLACEMENT_AFTER_CTR,
                "payload": float(replaced),
                "regime": str(regime),
                "rationale":
                    "replacement_after_contradiction_then_rejoin"
                    "_arbiter_applied",
                "replacement_lag_turns": int(
                    replacement_lag_turns),
            }
        # Fall back to V7.
        if regime == W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR:
            # The V7 controller doesn't know about the V9 regime;
            # call it with the baseline regime so the W72 V7
            # arbiters can still fire on the underlying signals.
            v7_decision = self.inner_v7.decide_v7(
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
                rejoin_pressure=float(rejoin_pressure),
                agent_rejoin_recovery_flags=(
                    list(agent_rejoin_recovery_flags)
                    if agent_rejoin_recovery_flags is not None
                    else None),
                restart_repair_trajectory_cid=str(
                    restart_repair_trajectory_cid),
                rejoin_lag_turns=int(rejoin_lag_turns),
                agent_rejoin_absorption_scores=(
                    list(agent_rejoin_absorption_scores)
                    if agent_rejoin_absorption_scores is not None
                    else None),
                **v7_kwargs)
        else:
            v7_decision = self.inner_v7.decide_v7(
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
                rejoin_pressure=float(rejoin_pressure),
                agent_rejoin_recovery_flags=(
                    list(agent_rejoin_recovery_flags)
                    if agent_rejoin_recovery_flags is not None
                    else None),
                restart_repair_trajectory_cid=str(
                    restart_repair_trajectory_cid),
                rejoin_lag_turns=int(rejoin_lag_turns),
                agent_rejoin_absorption_scores=(
                    list(agent_rejoin_absorption_scores)
                    if agent_rejoin_absorption_scores is not None
                    else None),
                **v7_kwargs)
        self.audit_v8.append({
            "decision": str(v7_decision.get("decision", "")),
            "regime": str(regime),
            "v7_fallback": True,
        })
        return dict(v7_decision)


@dataclasses.dataclass(frozen=True)
class TeamConsensusControllerV8Witness:
    schema: str
    controller_v8_cid: str
    inner_v7_witness_cid: str
    n_decisions_v8: int
    decisions_summary_v8: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_v8_cid": str(self.controller_v8_cid),
            "inner_v7_witness_cid": str(
                self.inner_v7_witness_cid),
            "n_decisions_v8": int(self.n_decisions_v8),
            "decisions_summary_v8": {
                k: int(v) for k, v in sorted(
                    self.decisions_summary_v8.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "team_consensus_controller_v8_witness",
            "witness": self.to_dict()})


def emit_team_consensus_controller_v8_witness(
        controller: TeamConsensusControllerV8,
) -> TeamConsensusControllerV8Witness:
    from .team_consensus_controller_v7 import (
        emit_team_consensus_controller_v7_witness,
    )
    inner_w = emit_team_consensus_controller_v7_witness(
        controller.inner_v7)
    summary: dict[str, int] = {
        d: 0 for d in W73_TC_V8_DECISIONS}
    for entry in controller.audit_v8:
        dec = str(entry.get("decision", ""))
        if dec in summary:
            summary[dec] += 1
    return TeamConsensusControllerV8Witness(
        schema=W73_TEAM_CONSENSUS_CONTROLLER_V8_SCHEMA_VERSION,
        controller_v8_cid=str(controller.cid()),
        inner_v7_witness_cid=str(inner_w.cid()),
        n_decisions_v8=int(len(controller.audit_v8)),
        decisions_summary_v8=dict(summary),
    )


__all__ = [
    "W73_TEAM_CONSENSUS_CONTROLLER_V8_SCHEMA_VERSION",
    "W73_TC_V8_DECISION_REPLACEMENT_PRESSURE",
    "W73_TC_V8_DECISION_REPLACEMENT_AFTER_CTR",
    "W73_TC_V8_DECISIONS",
    "TeamConsensusControllerV8",
    "TeamConsensusControllerV8Witness",
    "emit_team_consensus_controller_v8_witness",
]
