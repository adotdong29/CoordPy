"""W76 M12 — Team-Consensus Controller V11.

Strictly extends W75's ``coordpy.team_consensus_controller_v10``.
The V11 controller adds:

* **chain-then-restart-pressure arbiter** — when caller-declared
  chain-then-restart pressure crosses the V11 chain-then-restart
  floor and substrate-replay trust ≥ floor, the V11 controller
  picks the weighted-mean of confidence-aligned agents whose
  chain-then-restart-recovery flag is set.
* **post-compound-chain-restart-after-RTR arbiter** — when
  caller-declared ``chain_then_restart_trajectory_cid`` is non-
  empty AND the post-compound-chain-restart window crosses the
  floor, the V11 controller pulls toward the agent that absorbed
  the chain-then-restart horizon the cleanest.
* **per-regime thresholds for the W76 regime** — adds
  ``restart_after_compound_chain_repair_under_budget`` to the V10
  fifteen W75 regimes.

V11 preserves V10's decision semantics byte-for-byte under the
W75 regimes (with a wrapping V11 audit ledger).

Honest scope (W76)
------------------

* ``W76-L-TEAM-CONSENSUS-V11-IN-REPO-CAP`` — operates on in-repo
  MASC V12 outcomes only.
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
        "coordpy.team_consensus_controller_v11 requires numpy"
        ) from exc

from .multi_agent_substrate_coordinator_v12 import (
    W76_MASC_V12_REGIME_CHAIN_THEN_RESTART,
    W76_MASC_V12_REGIMES,
)
from .team_consensus_controller_v10 import (
    TeamConsensusControllerV10, W75_TC_V10_DECISIONS,
)
from .tiny_substrate_v3 import _sha256_hex


W76_TEAM_CONSENSUS_CONTROLLER_V11_SCHEMA_VERSION: str = (
    "coordpy.team_consensus_controller_v11.v1")

W76_TC_V11_DECISION_CHAIN_THEN_RESTART_PRESSURE: str = (
    "chain_then_restart_pressure_arbiter")
W76_TC_V11_DECISION_POST_COMPOUND_CHAIN_RESTART_RTR: str = (
    "post_compound_chain_restart_after_rtr_arbiter")
W76_TC_V11_DECISIONS: tuple[str, ...] = (
    *W75_TC_V10_DECISIONS,
    W76_TC_V11_DECISION_CHAIN_THEN_RESTART_PRESSURE,
    W76_TC_V11_DECISION_POST_COMPOUND_CHAIN_RESTART_RTR,
)


@dataclasses.dataclass
class TeamConsensusControllerV11:
    inner_v10: TeamConsensusControllerV10 = dataclasses.field(
        default_factory=TeamConsensusControllerV10)
    chain_then_restart_pressure_substrate_trust_floor: float = (
        0.5)
    chain_then_restart_pressure_floor: float = 0.5
    post_compound_chain_restart_rtr_substrate_trust_floor: (
        float) = 0.5
    post_compound_chain_restart_window_floor: int = 1
    audit_v11: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W76_TEAM_CONSENSUS_CONTROLLER_V11_SCHEMA_VERSION,
            "kind": "team_consensus_controller_v11",
            "inner_v10_cid": str(self.inner_v10.cid()),
            "chain_then_restart_pressure_substrate_trust_floor":
                float(round(
                    self
                    .chain_then_restart_pressure_substrate_trust_floor,
                    12)),
            "chain_then_restart_pressure_floor": float(round(
                self.chain_then_restart_pressure_floor, 12)),
            "post_compound_chain_restart_rtr_substrate_trust_floor":
                float(round(
                    self
                    .post_compound_chain_restart_rtr_substrate_trust_floor,
                    12)),
            "post_compound_chain_restart_window_floor": int(
                self.post_compound_chain_restart_window_floor),
        })

    def decide_v11(
            self, *,
            regime: str,
            agent_guesses: Sequence[float],
            agent_confidences: Sequence[float],
            substrate_replay_trust: float = 0.0,
            chain_then_restart_pressure: float = 0.0,
            agent_chain_then_restart_recovery_flags: (
                Sequence[int] | None) = None,
            chain_then_restart_trajectory_cid: str = "",
            post_compound_chain_restart_window_turns: int = 0,
            agent_chain_then_restart_absorption_scores: (
                Sequence[float] | None) = None,
            **v10_kwargs: Any,
    ) -> dict[str, Any]:
        """V11 decision routine.

        Routes through V10 for W75 regimes (with the V11 chain-
        then-restart-pressure arbiter firing on top when its
        condition holds). For the W76-only chain-then-restart
        regime V11 applies a coordinated arbiter.
        """
        if regime not in W76_MASC_V12_REGIMES:
            raise ValueError(f"unknown regime {regime!r}")
        # W76 chain-then-restart regime — coordinated post-
        # compound-chain-restart-after-RTR arbiter.
        if (regime
                == W76_MASC_V12_REGIME_CHAIN_THEN_RESTART
                and agent_chain_then_restart_absorption_scores
                is not None
                and len(
                    agent_chain_then_restart_absorption_scores)
                == len(agent_guesses)
                and float(substrate_replay_trust)
                >= float(
                    self
                    .post_compound_chain_restart_rtr_substrate_trust_floor)):
            scores = _np.asarray(
                list(
                    agent_chain_then_restart_absorption_scores),
                dtype=_np.float64)
            gs = _np.asarray(
                list(agent_guesses), dtype=_np.float64)
            denom = float(scores.sum())
            if denom > 0.0:
                replaced = float(_np.dot(gs, scores) / denom)
            else:
                replaced = float(gs.mean())
            self.audit_v11.append({
                "decision":
                    W76_TC_V11_DECISION_POST_COMPOUND_CHAIN_RESTART_RTR,
                "regime": str(regime),
                "n_agents": int(len(agent_guesses)),
                "post_compound_chain_restart_window_turns": int(
                    post_compound_chain_restart_window_turns),
            })
            return {
                "decision":
                    W76_TC_V11_DECISION_POST_COMPOUND_CHAIN_RESTART_RTR,
                "payload": float(replaced),
                "regime": str(regime),
                "rationale":
                    "post_compound_chain_restart_after_rtr_"
                    "arbiter_applied",
            }
        # V11 chain-then-restart-pressure arbiter.
        if (float(chain_then_restart_pressure)
                >= float(self.chain_then_restart_pressure_floor)
                and agent_chain_then_restart_recovery_flags
                is not None
                and len(agent_chain_then_restart_recovery_flags)
                == len(agent_guesses)
                and float(substrate_replay_trust)
                >= float(
                    self
                    .chain_then_restart_pressure_substrate_trust_floor)):
            matching_idx = [
                i for i, flag in enumerate(
                    agent_chain_then_restart_recovery_flags)
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
                self.audit_v11.append({
                    "decision":
                        W76_TC_V11_DECISION_CHAIN_THEN_RESTART_PRESSURE,
                    "regime": str(regime),
                    "n_matching_agents": int(
                        len(matching_idx)),
                    "chain_then_restart_pressure": float(
                        chain_then_restart_pressure),
                })
                return {
                    "decision":
                        W76_TC_V11_DECISION_CHAIN_THEN_RESTART_PRESSURE,
                    "payload": float(replaced),
                    "regime": str(regime),
                    "rationale":
                        "chain_then_restart_pressure_arbiter_"
                        "applied",
                    "chain_then_restart_pressure": float(
                        chain_then_restart_pressure),
                }
        # V11 post-compound-chain-restart-after-RTR arbiter (any
        # regime + non-empty chain-then-restart-trajectory CID +
        # post-compound-chain-restart-window > floor).
        if (str(chain_then_restart_trajectory_cid)
                and int(post_compound_chain_restart_window_turns)
                >= int(
                    self
                    .post_compound_chain_restart_window_floor)
                and agent_chain_then_restart_absorption_scores
                is not None
                and len(
                    agent_chain_then_restart_absorption_scores)
                == len(agent_guesses)
                and float(substrate_replay_trust)
                >= float(
                    self
                    .post_compound_chain_restart_rtr_substrate_trust_floor)):
            scores = _np.asarray(
                list(
                    agent_chain_then_restart_absorption_scores),
                dtype=_np.float64)
            gs = _np.asarray(
                list(agent_guesses), dtype=_np.float64)
            denom = float(scores.sum())
            if denom > 0.0:
                replaced = float(_np.dot(gs, scores) / denom)
            else:
                replaced = float(gs.mean())
            self.audit_v11.append({
                "decision":
                    W76_TC_V11_DECISION_POST_COMPOUND_CHAIN_RESTART_RTR,
                "regime": str(regime),
                "post_compound_chain_restart_window_turns": int(
                    post_compound_chain_restart_window_turns),
            })
            return {
                "decision":
                    W76_TC_V11_DECISION_POST_COMPOUND_CHAIN_RESTART_RTR,
                "payload": float(replaced),
                "regime": str(regime),
                "rationale":
                    "post_compound_chain_restart_after_rtr_"
                    "arbiter_applied",
                "post_compound_chain_restart_window_turns": int(
                    post_compound_chain_restart_window_turns),
            }
        # Fall back to V10.
        if regime == W76_MASC_V12_REGIME_CHAIN_THEN_RESTART:
            v10_decision = self.inner_v10.decide_v10(
                regime="baseline",
                agent_guesses=list(agent_guesses),
                agent_confidences=list(agent_confidences),
                substrate_replay_trust=float(
                    substrate_replay_trust),
                **v10_kwargs)
        else:
            v10_decision = self.inner_v10.decide_v10(
                regime=str(regime),
                agent_guesses=list(agent_guesses),
                agent_confidences=list(agent_confidences),
                substrate_replay_trust=float(
                    substrate_replay_trust),
                **v10_kwargs)
        self.audit_v11.append({
            "decision": str(v10_decision.get("decision", "")),
            "regime": str(regime),
            "v10_fallback": True,
        })
        return dict(v10_decision)


@dataclasses.dataclass(frozen=True)
class TeamConsensusControllerV11Witness:
    schema: str
    controller_v11_cid: str
    inner_v10_witness_cid: str
    n_decisions_v11: int
    decisions_summary_v11: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_v11_cid": str(self.controller_v11_cid),
            "inner_v10_witness_cid": str(
                self.inner_v10_witness_cid),
            "n_decisions_v11": int(self.n_decisions_v11),
            "decisions_summary_v11": {
                k: int(v) for k, v in sorted(
                    self.decisions_summary_v11.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "team_consensus_controller_v11_witness",
            "witness": self.to_dict()})


def emit_team_consensus_controller_v11_witness(
        controller: TeamConsensusControllerV11,
) -> TeamConsensusControllerV11Witness:
    from .team_consensus_controller_v10 import (
        emit_team_consensus_controller_v10_witness,
    )
    inner_w = emit_team_consensus_controller_v10_witness(
        controller.inner_v10)
    summary: dict[str, int] = {
        d: 0 for d in W76_TC_V11_DECISIONS}
    for entry in controller.audit_v11:
        dec = str(entry.get("decision", ""))
        if dec in summary:
            summary[dec] += 1
    return TeamConsensusControllerV11Witness(
        schema=W76_TEAM_CONSENSUS_CONTROLLER_V11_SCHEMA_VERSION,
        controller_v11_cid=str(controller.cid()),
        inner_v10_witness_cid=str(inner_w.cid()),
        n_decisions_v11=int(len(controller.audit_v11)),
        decisions_summary_v11=dict(summary),
    )


__all__ = [
    "W76_TEAM_CONSENSUS_CONTROLLER_V11_SCHEMA_VERSION",
    "W76_TC_V11_DECISION_CHAIN_THEN_RESTART_PRESSURE",
    "W76_TC_V11_DECISION_POST_COMPOUND_CHAIN_RESTART_RTR",
    "W76_TC_V11_DECISIONS",
    "TeamConsensusControllerV11",
    "TeamConsensusControllerV11Witness",
    "emit_team_consensus_controller_v11_witness",
]
