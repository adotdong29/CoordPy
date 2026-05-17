"""W74 M12 — Team-Consensus Controller V9.

Strictly extends W73's ``coordpy.team_consensus_controller_v8``.
The V9 controller adds:

* **compound-pressure arbiter** — when caller-declared compound
  pressure crosses the V9 compound-floor and substrate-replay trust
  ≥ floor, the V9 controller picks the weighted-mean of confidence-
  aligned agents whose compound-recovery flag is set.
* **compound-repair-after-DR-then-replacement arbiter** — when
  caller-declared ``compound_repair_trajectory_cid`` is non-empty
  AND the compound-window crosses the floor, the V9 controller
  pulls toward the agent that absorbed the compound horizon the
  cleanest.
* **per-regime thresholds for the W74 regime** — adds
  ``replacement_after_delayed_repair_under_budget`` to the V8
  thirteen W73 regimes.

V9 preserves V8's decision semantics byte-for-byte under the W73
regimes (with a wrapping V9 audit ledger).

Honest scope (W74)
------------------

* ``W74-L-TEAM-CONSENSUS-V9-IN-REPO-CAP`` — operates on in-repo
  MASC V10 outcomes only.
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
        "coordpy.team_consensus_controller_v9 requires numpy"
        ) from exc

from .multi_agent_substrate_coordinator_v10 import (
    W74_MASC_V10_REGIME_REPLACEMENT_AFTER_DELAYED_REPAIR,
    W74_MASC_V10_REGIMES,
)
from .team_consensus_controller_v8 import (
    TeamConsensusControllerV8, W73_TC_V8_DECISIONS,
)
from .tiny_substrate_v3 import _sha256_hex


W74_TEAM_CONSENSUS_CONTROLLER_V9_SCHEMA_VERSION: str = (
    "coordpy.team_consensus_controller_v9.v1")

W74_TC_V9_DECISION_COMPOUND_PRESSURE: str = (
    "compound_pressure_arbiter")
W74_TC_V9_DECISION_COMPOUND_REPAIR_AFTER_DRTR: str = (
    "compound_repair_after_delayed_repair_then_replacement_arbiter")
W74_TC_V9_DECISIONS: tuple[str, ...] = (
    *W73_TC_V8_DECISIONS,
    W74_TC_V9_DECISION_COMPOUND_PRESSURE,
    W74_TC_V9_DECISION_COMPOUND_REPAIR_AFTER_DRTR,
)


@dataclasses.dataclass
class TeamConsensusControllerV9:
    inner_v8: TeamConsensusControllerV8 = dataclasses.field(
        default_factory=TeamConsensusControllerV8)
    compound_pressure_substrate_trust_floor: float = 0.5
    compound_pressure_floor: float = 0.5
    compound_repair_drtr_substrate_trust_floor: float = 0.5
    compound_window_floor: int = 1
    audit_v9: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W74_TEAM_CONSENSUS_CONTROLLER_V9_SCHEMA_VERSION,
            "kind": "team_consensus_controller_v9",
            "inner_v8_cid": str(self.inner_v8.cid()),
            "compound_pressure_substrate_trust_floor":
                float(round(
                    self
                    .compound_pressure_substrate_trust_floor,
                    12)),
            "compound_pressure_floor": float(round(
                self.compound_pressure_floor, 12)),
            "compound_repair_drtr_substrate_trust_floor":
                float(round(
                    self
                    .compound_repair_drtr_substrate_trust_floor,
                    12)),
            "compound_window_floor": int(
                self.compound_window_floor),
        })

    def decide_v9(
            self, *,
            regime: str,
            agent_guesses: Sequence[float],
            agent_confidences: Sequence[float],
            substrate_replay_trust: float = 0.0,
            compound_pressure: float = 0.0,
            agent_compound_recovery_flags: (
                Sequence[int] | None) = None,
            compound_repair_trajectory_cid: str = "",
            compound_window_turns: int = 0,
            agent_compound_absorption_scores: (
                Sequence[float] | None) = None,
            **v8_kwargs: Any,
    ) -> dict[str, Any]:
        """V9 decision routine.

        Routes through V8 for W73 regimes (and the V9 compound-
        pressure / compound-repair-after-DRTR arbiters fire on top
        when their conditions hold). For the W74-only compound
        regime V9 applies a coordinated arbiter.
        """
        if regime not in W74_MASC_V10_REGIMES:
            raise ValueError(f"unknown regime {regime!r}")
        # W74 compound regime — coordinated compound-repair-after-
        # DRTR arbiter.
        if (regime
                == W74_MASC_V10_REGIME_REPLACEMENT_AFTER_DELAYED_REPAIR
                and agent_compound_absorption_scores is not None
                and len(agent_compound_absorption_scores)
                    == len(agent_guesses)
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .compound_repair_drtr_substrate_trust_floor)):
            scores = _np.asarray(
                list(agent_compound_absorption_scores),
                dtype=_np.float64)
            gs = _np.asarray(
                list(agent_guesses), dtype=_np.float64)
            denom = float(scores.sum())
            if denom > 0.0:
                replaced = float(_np.dot(gs, scores) / denom)
            else:
                replaced = float(gs.mean())
            self.audit_v9.append({
                "decision":
                    W74_TC_V9_DECISION_COMPOUND_REPAIR_AFTER_DRTR,
                "regime": str(regime),
                "n_agents": int(len(agent_guesses)),
                "compound_window_turns": int(
                    compound_window_turns),
            })
            return {
                "decision":
                    W74_TC_V9_DECISION_COMPOUND_REPAIR_AFTER_DRTR,
                "payload": float(replaced),
                "regime": str(regime),
                "rationale":
                    "compound_repair_after_delayed_repair_"
                    "then_replacement_arbiter_applied",
            }
        # V9 compound-pressure arbiter (any regime + high compound
        # pressure).
        if (float(compound_pressure)
                >= float(self.compound_pressure_floor)
                and agent_compound_recovery_flags is not None
                and len(agent_compound_recovery_flags)
                    == len(agent_guesses)
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .compound_pressure_substrate_trust_floor)):
            matching_idx = [
                i for i, flag in enumerate(
                    agent_compound_recovery_flags)
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
                self.audit_v9.append({
                    "decision":
                        W74_TC_V9_DECISION_COMPOUND_PRESSURE,
                    "regime": str(regime),
                    "n_matching_agents": int(
                        len(matching_idx)),
                    "compound_pressure": float(
                        compound_pressure),
                })
                return {
                    "decision":
                        W74_TC_V9_DECISION_COMPOUND_PRESSURE,
                    "payload": float(replaced),
                    "regime": str(regime),
                    "rationale":
                        "compound_pressure_arbiter_applied",
                    "compound_pressure": float(
                        compound_pressure),
                }
        # V9 compound-repair-after-DRTR arbiter (any regime + non-
        # empty compound-repair-trajectory CID + compound-window >
        # floor).
        if (str(compound_repair_trajectory_cid)
                and int(compound_window_turns)
                    >= int(self.compound_window_floor)
                and agent_compound_absorption_scores is not None
                and len(agent_compound_absorption_scores)
                    == len(agent_guesses)
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .compound_repair_drtr_substrate_trust_floor)):
            scores = _np.asarray(
                list(agent_compound_absorption_scores),
                dtype=_np.float64)
            gs = _np.asarray(
                list(agent_guesses), dtype=_np.float64)
            denom = float(scores.sum())
            if denom > 0.0:
                replaced = float(_np.dot(gs, scores) / denom)
            else:
                replaced = float(gs.mean())
            self.audit_v9.append({
                "decision":
                    W74_TC_V9_DECISION_COMPOUND_REPAIR_AFTER_DRTR,
                "regime": str(regime),
                "compound_window_turns": int(
                    compound_window_turns),
            })
            return {
                "decision":
                    W74_TC_V9_DECISION_COMPOUND_REPAIR_AFTER_DRTR,
                "payload": float(replaced),
                "regime": str(regime),
                "rationale":
                    "compound_repair_after_delayed_repair_"
                    "then_replacement_arbiter_applied",
                "compound_window_turns": int(
                    compound_window_turns),
            }
        # Fall back to V8.
        if regime == (
                W74_MASC_V10_REGIME_REPLACEMENT_AFTER_DELAYED_REPAIR):
            # The V8 controller doesn't know about the V10 regime;
            # call it with baseline so V8's arbiters can still fire.
            v8_decision = self.inner_v8.decide_v8(
                regime="baseline",
                agent_guesses=list(agent_guesses),
                agent_confidences=list(agent_confidences),
                substrate_replay_trust=float(
                    substrate_replay_trust),
                **v8_kwargs)
        else:
            v8_decision = self.inner_v8.decide_v8(
                regime=str(regime),
                agent_guesses=list(agent_guesses),
                agent_confidences=list(agent_confidences),
                substrate_replay_trust=float(
                    substrate_replay_trust),
                **v8_kwargs)
        self.audit_v9.append({
            "decision": str(v8_decision.get("decision", "")),
            "regime": str(regime),
            "v8_fallback": True,
        })
        return dict(v8_decision)


@dataclasses.dataclass(frozen=True)
class TeamConsensusControllerV9Witness:
    schema: str
    controller_v9_cid: str
    inner_v8_witness_cid: str
    n_decisions_v9: int
    decisions_summary_v9: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_v9_cid": str(self.controller_v9_cid),
            "inner_v8_witness_cid": str(
                self.inner_v8_witness_cid),
            "n_decisions_v9": int(self.n_decisions_v9),
            "decisions_summary_v9": {
                k: int(v) for k, v in sorted(
                    self.decisions_summary_v9.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "team_consensus_controller_v9_witness",
            "witness": self.to_dict()})


def emit_team_consensus_controller_v9_witness(
        controller: TeamConsensusControllerV9,
) -> TeamConsensusControllerV9Witness:
    from .team_consensus_controller_v8 import (
        emit_team_consensus_controller_v8_witness,
    )
    inner_w = emit_team_consensus_controller_v8_witness(
        controller.inner_v8)
    summary: dict[str, int] = {
        d: 0 for d in W74_TC_V9_DECISIONS}
    for entry in controller.audit_v9:
        dec = str(entry.get("decision", ""))
        if dec in summary:
            summary[dec] += 1
    return TeamConsensusControllerV9Witness(
        schema=W74_TEAM_CONSENSUS_CONTROLLER_V9_SCHEMA_VERSION,
        controller_v9_cid=str(controller.cid()),
        inner_v8_witness_cid=str(inner_w.cid()),
        n_decisions_v9=int(len(controller.audit_v9)),
        decisions_summary_v9=dict(summary),
    )


__all__ = [
    "W74_TEAM_CONSENSUS_CONTROLLER_V9_SCHEMA_VERSION",
    "W74_TC_V9_DECISION_COMPOUND_PRESSURE",
    "W74_TC_V9_DECISION_COMPOUND_REPAIR_AFTER_DRTR",
    "W74_TC_V9_DECISIONS",
    "TeamConsensusControllerV9",
    "TeamConsensusControllerV9Witness",
    "emit_team_consensus_controller_v9_witness",
]
