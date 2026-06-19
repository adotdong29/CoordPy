"""W77 M12 — Team-Consensus Controller V12.

Strictly extends W76's ``coordpy.team_consensus_controller_v11``.
The V12 controller adds:

* **post-restart-replacement-pressure arbiter** — when caller-
  declared post-restart-replacement pressure crosses the V12 floor
  and substrate-replay trust ≥ floor, the V12 controller picks the
  weighted-mean of confidence-aligned agents whose post-restart-
  replacement-recovery flag is set.
* **post-restart-replacement-trajectory arbiter** — when caller-
  declared ``post_restart_replacement_trajectory_cid`` is non-
  empty AND the post-restart-replacement window crosses the floor,
  the V12 controller pulls toward the agent that absorbed the
  post-restart-replacement horizon the cleanest.
* **per-regime thresholds for the W77 regime** — adds
  ``replacement_after_restart_after_compound_chain_repair_under_budget``
  to the V11 sixteen W76 regimes.

V12 preserves V11's decision semantics byte-for-byte under the W76
regimes (with a wrapping V12 audit ledger).

Honest scope (W77)
------------------

* ``W77-L-TEAM-CONSENSUS-V12-IN-REPO-CAP`` — operates on in-repo
  MASC V13 outcomes only.
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
        "coordpy.team_consensus_controller_v12 requires numpy"
        ) from exc

from .multi_agent_substrate_coordinator_v13 import (
    W77_MASC_V13_REGIME_POST_RESTART_REPLACEMENT,
    W77_MASC_V13_REGIMES,
)
from .team_consensus_controller_v11 import (
    TeamConsensusControllerV11, W76_TC_V11_DECISIONS,
)
from .tiny_substrate_v3 import _sha256_hex


W77_TEAM_CONSENSUS_CONTROLLER_V12_SCHEMA_VERSION: str = (
    "coordpy.team_consensus_controller_v12.v1")

W77_TC_V12_DECISION_POST_RESTART_REPLACEMENT_PRESSURE: str = (
    "post_restart_replacement_pressure_arbiter")
W77_TC_V12_DECISION_POST_RESTART_REPLACEMENT_TRAJECTORY: str = (
    "post_restart_replacement_trajectory_arbiter")
W77_TC_V12_DECISIONS: tuple[str, ...] = (
    *W76_TC_V11_DECISIONS,
    W77_TC_V12_DECISION_POST_RESTART_REPLACEMENT_PRESSURE,
    W77_TC_V12_DECISION_POST_RESTART_REPLACEMENT_TRAJECTORY,
)


@dataclasses.dataclass
class TeamConsensusControllerV12:
    inner_v11: TeamConsensusControllerV11 = dataclasses.field(
        default_factory=TeamConsensusControllerV11)
    post_restart_replacement_pressure_substrate_trust_floor: (
        float) = 0.5
    post_restart_replacement_pressure_floor: float = 0.5
    post_restart_replacement_trajectory_substrate_trust_floor: (
        float) = 0.5
    post_restart_replacement_window_floor: int = 1
    audit_v12: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W77_TEAM_CONSENSUS_CONTROLLER_V12_SCHEMA_VERSION,
            "kind": "team_consensus_controller_v12",
            "inner_v11_cid": str(self.inner_v11.cid()),
            "post_restart_replacement_pressure_substrate_trust_floor":
                float(round(
                    self
                    .post_restart_replacement_pressure_substrate_trust_floor,
                    12)),
            "post_restart_replacement_pressure_floor": float(round(
                self.post_restart_replacement_pressure_floor, 12)),
            "post_restart_replacement_trajectory_substrate_trust_floor":
                float(round(
                    self
                    .post_restart_replacement_trajectory_substrate_trust_floor,
                    12)),
            "post_restart_replacement_window_floor": int(
                self.post_restart_replacement_window_floor),
        })

    def decide_v12(
            self, *,
            regime: str,
            agent_guesses: Sequence[float],
            agent_confidences: Sequence[float],
            substrate_replay_trust: float = 0.0,
            post_restart_replacement_pressure: float = 0.0,
            agent_post_restart_replacement_recovery_flags: (
                Sequence[int] | None) = None,
            post_restart_replacement_trajectory_cid: str = "",
            post_restart_replacement_window_turns: int = 0,
            agent_post_restart_replacement_absorption_scores: (
                Sequence[float] | None) = None,
            **v11_kwargs: Any,
    ) -> dict[str, Any]:
        """V12 decision routine.

        Routes through V11 for W76 regimes. For the W77 post-
        restart-replacement regime V12 applies a coordinated
        arbiter on top.
        """
        if regime not in W77_MASC_V13_REGIMES:
            raise ValueError(f"unknown regime {regime!r}")
        # W77 post-restart-replacement regime — coordinated
        # post-restart-replacement-trajectory arbiter.
        if (regime
                == W77_MASC_V13_REGIME_POST_RESTART_REPLACEMENT
                and agent_post_restart_replacement_absorption_scores
                is not None
                and len(
                    agent_post_restart_replacement_absorption_scores)
                == len(agent_guesses)
                and float(substrate_replay_trust)
                >= float(
                    self
                    .post_restart_replacement_trajectory_substrate_trust_floor)):
            scores = _np.asarray(
                list(
                    agent_post_restart_replacement_absorption_scores),
                dtype=_np.float64)
            gs = _np.asarray(
                list(agent_guesses), dtype=_np.float64)
            denom = float(scores.sum())
            if denom > 0.0:
                replaced = float(_np.dot(gs, scores) / denom)
            else:
                replaced = float(gs.mean())
            self.audit_v12.append({
                "decision":
                    W77_TC_V12_DECISION_POST_RESTART_REPLACEMENT_TRAJECTORY,
                "regime": str(regime),
                "n_agents": int(len(agent_guesses)),
                "post_restart_replacement_window_turns": int(
                    post_restart_replacement_window_turns),
            })
            return {
                "decision":
                    W77_TC_V12_DECISION_POST_RESTART_REPLACEMENT_TRAJECTORY,
                "payload": float(replaced),
                "regime": str(regime),
                "rationale":
                    "post_restart_replacement_trajectory_"
                    "arbiter_applied",
            }
        # V12 post-restart-replacement-pressure arbiter.
        if (float(post_restart_replacement_pressure)
                >= float(
                    self
                    .post_restart_replacement_pressure_floor)
                and agent_post_restart_replacement_recovery_flags
                is not None
                and len(
                    agent_post_restart_replacement_recovery_flags)
                == len(agent_guesses)
                and float(substrate_replay_trust)
                >= float(
                    self
                    .post_restart_replacement_pressure_substrate_trust_floor)):
            matching_idx = [
                i for i, flag in enumerate(
                    agent_post_restart_replacement_recovery_flags)
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
                self.audit_v12.append({
                    "decision":
                        W77_TC_V12_DECISION_POST_RESTART_REPLACEMENT_PRESSURE,
                    "regime": str(regime),
                    "n_matching_agents": int(
                        len(matching_idx)),
                    "post_restart_replacement_pressure": float(
                        post_restart_replacement_pressure),
                })
                return {
                    "decision":
                        W77_TC_V12_DECISION_POST_RESTART_REPLACEMENT_PRESSURE,
                    "payload": float(replaced),
                    "regime": str(regime),
                    "rationale":
                        "post_restart_replacement_pressure_"
                        "arbiter_applied",
                    "post_restart_replacement_pressure": float(
                        post_restart_replacement_pressure),
                }
        # V12 post-restart-replacement-trajectory arbiter for any
        # regime with non-empty trajectory CID + post-restart-
        # replacement-window > floor.
        if (str(post_restart_replacement_trajectory_cid)
                and int(post_restart_replacement_window_turns)
                >= int(
                    self.post_restart_replacement_window_floor)
                and agent_post_restart_replacement_absorption_scores
                is not None
                and len(
                    agent_post_restart_replacement_absorption_scores)
                == len(agent_guesses)
                and float(substrate_replay_trust)
                >= float(
                    self
                    .post_restart_replacement_trajectory_substrate_trust_floor)):
            scores = _np.asarray(
                list(
                    agent_post_restart_replacement_absorption_scores),
                dtype=_np.float64)
            gs = _np.asarray(
                list(agent_guesses), dtype=_np.float64)
            denom = float(scores.sum())
            if denom > 0.0:
                replaced = float(_np.dot(gs, scores) / denom)
            else:
                replaced = float(gs.mean())
            self.audit_v12.append({
                "decision":
                    W77_TC_V12_DECISION_POST_RESTART_REPLACEMENT_TRAJECTORY,
                "regime": str(regime),
                "post_restart_replacement_window_turns": int(
                    post_restart_replacement_window_turns),
            })
            return {
                "decision":
                    W77_TC_V12_DECISION_POST_RESTART_REPLACEMENT_TRAJECTORY,
                "payload": float(replaced),
                "regime": str(regime),
                "rationale":
                    "post_restart_replacement_trajectory_"
                    "arbiter_applied",
                "post_restart_replacement_window_turns": int(
                    post_restart_replacement_window_turns),
            }
        # Fall back to V11.
        if regime == W77_MASC_V13_REGIME_POST_RESTART_REPLACEMENT:
            v11_decision = self.inner_v11.decide_v11(
                regime="baseline",
                agent_guesses=list(agent_guesses),
                agent_confidences=list(agent_confidences),
                substrate_replay_trust=float(
                    substrate_replay_trust),
                **v11_kwargs)
        else:
            v11_decision = self.inner_v11.decide_v11(
                regime=str(regime),
                agent_guesses=list(agent_guesses),
                agent_confidences=list(agent_confidences),
                substrate_replay_trust=float(
                    substrate_replay_trust),
                **v11_kwargs)
        self.audit_v12.append({
            "decision": str(v11_decision.get("decision", "")),
            "regime": str(regime),
            "v11_fallback": True,
        })
        return dict(v11_decision)


@dataclasses.dataclass(frozen=True)
class TeamConsensusControllerV12Witness:
    schema: str
    controller_v12_cid: str
    inner_v11_witness_cid: str
    n_decisions_v12: int
    decisions_summary_v12: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_v12_cid": str(self.controller_v12_cid),
            "inner_v11_witness_cid": str(
                self.inner_v11_witness_cid),
            "n_decisions_v12": int(self.n_decisions_v12),
            "decisions_summary_v12": {
                k: int(v) for k, v in sorted(
                    self.decisions_summary_v12.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "team_consensus_controller_v12_witness",
            "witness": self.to_dict()})


def emit_team_consensus_controller_v12_witness(
        controller: TeamConsensusControllerV12,
) -> TeamConsensusControllerV12Witness:
    from .team_consensus_controller_v11 import (
        emit_team_consensus_controller_v11_witness,
    )
    inner_w = emit_team_consensus_controller_v11_witness(
        controller.inner_v11)
    summary: dict[str, int] = {
        d: 0 for d in W77_TC_V12_DECISIONS}
    for entry in controller.audit_v12:
        dec = str(entry.get("decision", ""))
        if dec in summary:
            summary[dec] += 1
    return TeamConsensusControllerV12Witness(
        schema=W77_TEAM_CONSENSUS_CONTROLLER_V12_SCHEMA_VERSION,
        controller_v12_cid=str(controller.cid()),
        inner_v11_witness_cid=str(inner_w.cid()),
        n_decisions_v12=int(len(controller.audit_v12)),
        decisions_summary_v12=dict(summary),
    )


__all__ = [
    "W77_TEAM_CONSENSUS_CONTROLLER_V12_SCHEMA_VERSION",
    "W77_TC_V12_DECISION_POST_RESTART_REPLACEMENT_PRESSURE",
    "W77_TC_V12_DECISION_POST_RESTART_REPLACEMENT_TRAJECTORY",
    "W77_TC_V12_DECISIONS",
    "TeamConsensusControllerV12",
    "TeamConsensusControllerV12Witness",
    "emit_team_consensus_controller_v12_witness",
]
