"""W75 M12 — Team-Consensus Controller V10.

Strictly extends W74's ``coordpy.team_consensus_controller_v9``.
The V10 controller adds:

* **compound-chain-pressure arbiter** — when caller-declared
  compound-chain pressure crosses the V10 compound-chain floor and
  substrate-replay trust ≥ floor, the V10 controller picks the
  weighted-mean of confidence-aligned agents whose compound-chain-
  recovery flag is set.
* **compound-repair-after-replacement-then-rejoin arbiter** — when
  caller-declared ``compound_chain_repair_trajectory_cid`` is non-
  empty AND the compound-chain-window crosses the floor, the V10
  controller pulls toward the agent that absorbed the compound-
  chain horizon the cleanest.
* **per-regime thresholds for the W75 regime** — adds
  ``compound_repair_after_replacement_then_rejoin_under_budget`` to
  the V9 fourteen W74 regimes.

V10 preserves V9's decision semantics byte-for-byte under the W74
regimes (with a wrapping V10 audit ledger).

Honest scope (W75)
------------------

* ``W75-L-TEAM-CONSENSUS-V10-IN-REPO-CAP`` — operates on in-repo
  MASC V11 outcomes only.
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
        "coordpy.team_consensus_controller_v10 requires numpy"
        ) from exc

from .multi_agent_substrate_coordinator_v11 import (
    W75_MASC_V11_REGIME_COMPOUND_CHAIN,
    W75_MASC_V11_REGIMES,
)
from .team_consensus_controller_v9 import (
    TeamConsensusControllerV9, W74_TC_V9_DECISIONS,
)
from .tiny_substrate_v3 import _sha256_hex


W75_TEAM_CONSENSUS_CONTROLLER_V10_SCHEMA_VERSION: str = (
    "coordpy.team_consensus_controller_v10.v1")

W75_TC_V10_DECISION_COMPOUND_CHAIN_PRESSURE: str = (
    "compound_chain_pressure_arbiter")
W75_TC_V10_DECISION_COMPOUND_REPAIR_AFTER_RTR: str = (
    "compound_repair_after_replacement_then_rejoin_arbiter")
W75_TC_V10_DECISIONS: tuple[str, ...] = (
    *W74_TC_V9_DECISIONS,
    W75_TC_V10_DECISION_COMPOUND_CHAIN_PRESSURE,
    W75_TC_V10_DECISION_COMPOUND_REPAIR_AFTER_RTR,
)


@dataclasses.dataclass
class TeamConsensusControllerV10:
    inner_v9: TeamConsensusControllerV9 = dataclasses.field(
        default_factory=TeamConsensusControllerV9)
    compound_chain_pressure_substrate_trust_floor: float = 0.5
    compound_chain_pressure_floor: float = 0.5
    compound_repair_rtr_substrate_trust_floor: float = 0.5
    compound_chain_window_floor: int = 1
    audit_v10: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W75_TEAM_CONSENSUS_CONTROLLER_V10_SCHEMA_VERSION,
            "kind": "team_consensus_controller_v10",
            "inner_v9_cid": str(self.inner_v9.cid()),
            "compound_chain_pressure_substrate_trust_floor":
                float(round(
                    self
                    .compound_chain_pressure_substrate_trust_floor,
                    12)),
            "compound_chain_pressure_floor": float(round(
                self.compound_chain_pressure_floor, 12)),
            "compound_repair_rtr_substrate_trust_floor":
                float(round(
                    self
                    .compound_repair_rtr_substrate_trust_floor,
                    12)),
            "compound_chain_window_floor": int(
                self.compound_chain_window_floor),
        })

    def decide_v10(
            self, *,
            regime: str,
            agent_guesses: Sequence[float],
            agent_confidences: Sequence[float],
            substrate_replay_trust: float = 0.0,
            compound_chain_pressure: float = 0.0,
            agent_compound_chain_recovery_flags: (
                Sequence[int] | None) = None,
            compound_chain_repair_trajectory_cid: str = "",
            compound_chain_window_turns: int = 0,
            agent_compound_chain_absorption_scores: (
                Sequence[float] | None) = None,
            **v9_kwargs: Any,
    ) -> dict[str, Any]:
        """V10 decision routine.

        Routes through V9 for W74 regimes (and the V10 compound-
        chain-pressure / compound-repair-after-RTR arbiters fire on
        top when their conditions hold). For the W75-only compound-
        chain regime V10 applies a coordinated arbiter.
        """
        if regime not in W75_MASC_V11_REGIMES:
            raise ValueError(f"unknown regime {regime!r}")
        # W75 compound-chain regime — coordinated compound-repair-
        # after-RTR arbiter.
        if (regime
                == W75_MASC_V11_REGIME_COMPOUND_CHAIN
                and agent_compound_chain_absorption_scores
                is not None
                and len(agent_compound_chain_absorption_scores)
                    == len(agent_guesses)
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .compound_repair_rtr_substrate_trust_floor)):
            scores = _np.asarray(
                list(agent_compound_chain_absorption_scores),
                dtype=_np.float64)
            gs = _np.asarray(
                list(agent_guesses), dtype=_np.float64)
            denom = float(scores.sum())
            if denom > 0.0:
                replaced = float(_np.dot(gs, scores) / denom)
            else:
                replaced = float(gs.mean())
            self.audit_v10.append({
                "decision":
                    W75_TC_V10_DECISION_COMPOUND_REPAIR_AFTER_RTR,
                "regime": str(regime),
                "n_agents": int(len(agent_guesses)),
                "compound_chain_window_turns": int(
                    compound_chain_window_turns),
            })
            return {
                "decision":
                    W75_TC_V10_DECISION_COMPOUND_REPAIR_AFTER_RTR,
                "payload": float(replaced),
                "regime": str(regime),
                "rationale":
                    "compound_repair_after_replacement_then_"
                    "rejoin_arbiter_applied",
            }
        # V10 compound-chain-pressure arbiter (any regime + high
        # compound-chain pressure).
        if (float(compound_chain_pressure)
                >= float(self.compound_chain_pressure_floor)
                and agent_compound_chain_recovery_flags is not None
                and len(agent_compound_chain_recovery_flags)
                    == len(agent_guesses)
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .compound_chain_pressure_substrate_trust_floor)):
            matching_idx = [
                i for i, flag in enumerate(
                    agent_compound_chain_recovery_flags)
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
                self.audit_v10.append({
                    "decision":
                        W75_TC_V10_DECISION_COMPOUND_CHAIN_PRESSURE,
                    "regime": str(regime),
                    "n_matching_agents": int(
                        len(matching_idx)),
                    "compound_chain_pressure": float(
                        compound_chain_pressure),
                })
                return {
                    "decision":
                        W75_TC_V10_DECISION_COMPOUND_CHAIN_PRESSURE,
                    "payload": float(replaced),
                    "regime": str(regime),
                    "rationale":
                        "compound_chain_pressure_arbiter_applied",
                    "compound_chain_pressure": float(
                        compound_chain_pressure),
                }
        # V10 compound-repair-after-RTR arbiter (any regime + non-
        # empty compound-chain-repair-trajectory CID + compound-
        # chain-window > floor).
        if (str(compound_chain_repair_trajectory_cid)
                and int(compound_chain_window_turns)
                    >= int(self.compound_chain_window_floor)
                and agent_compound_chain_absorption_scores
                is not None
                and len(agent_compound_chain_absorption_scores)
                    == len(agent_guesses)
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .compound_repair_rtr_substrate_trust_floor)):
            scores = _np.asarray(
                list(agent_compound_chain_absorption_scores),
                dtype=_np.float64)
            gs = _np.asarray(
                list(agent_guesses), dtype=_np.float64)
            denom = float(scores.sum())
            if denom > 0.0:
                replaced = float(_np.dot(gs, scores) / denom)
            else:
                replaced = float(gs.mean())
            self.audit_v10.append({
                "decision":
                    W75_TC_V10_DECISION_COMPOUND_REPAIR_AFTER_RTR,
                "regime": str(regime),
                "compound_chain_window_turns": int(
                    compound_chain_window_turns),
            })
            return {
                "decision":
                    W75_TC_V10_DECISION_COMPOUND_REPAIR_AFTER_RTR,
                "payload": float(replaced),
                "regime": str(regime),
                "rationale":
                    "compound_repair_after_replacement_then_"
                    "rejoin_arbiter_applied",
                "compound_chain_window_turns": int(
                    compound_chain_window_turns),
            }
        # Fall back to V9.
        if regime == W75_MASC_V11_REGIME_COMPOUND_CHAIN:
            # The V9 controller doesn't know about the V11 regime;
            # call it with baseline so V9's arbiters can still fire.
            v9_decision = self.inner_v9.decide_v9(
                regime="baseline",
                agent_guesses=list(agent_guesses),
                agent_confidences=list(agent_confidences),
                substrate_replay_trust=float(
                    substrate_replay_trust),
                **v9_kwargs)
        else:
            v9_decision = self.inner_v9.decide_v9(
                regime=str(regime),
                agent_guesses=list(agent_guesses),
                agent_confidences=list(agent_confidences),
                substrate_replay_trust=float(
                    substrate_replay_trust),
                **v9_kwargs)
        self.audit_v10.append({
            "decision": str(v9_decision.get("decision", "")),
            "regime": str(regime),
            "v9_fallback": True,
        })
        return dict(v9_decision)


@dataclasses.dataclass(frozen=True)
class TeamConsensusControllerV10Witness:
    schema: str
    controller_v10_cid: str
    inner_v9_witness_cid: str
    n_decisions_v10: int
    decisions_summary_v10: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_v10_cid": str(self.controller_v10_cid),
            "inner_v9_witness_cid": str(
                self.inner_v9_witness_cid),
            "n_decisions_v10": int(self.n_decisions_v10),
            "decisions_summary_v10": {
                k: int(v) for k, v in sorted(
                    self.decisions_summary_v10.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "team_consensus_controller_v10_witness",
            "witness": self.to_dict()})


def emit_team_consensus_controller_v10_witness(
        controller: TeamConsensusControllerV10,
) -> TeamConsensusControllerV10Witness:
    from .team_consensus_controller_v9 import (
        emit_team_consensus_controller_v9_witness,
    )
    inner_w = emit_team_consensus_controller_v9_witness(
        controller.inner_v9)
    summary: dict[str, int] = {
        d: 0 for d in W75_TC_V10_DECISIONS}
    for entry in controller.audit_v10:
        dec = str(entry.get("decision", ""))
        if dec in summary:
            summary[dec] += 1
    return TeamConsensusControllerV10Witness(
        schema=W75_TEAM_CONSENSUS_CONTROLLER_V10_SCHEMA_VERSION,
        controller_v10_cid=str(controller.cid()),
        inner_v9_witness_cid=str(inner_w.cid()),
        n_decisions_v10=int(len(controller.audit_v10)),
        decisions_summary_v10=dict(summary),
    )


__all__ = [
    "W75_TEAM_CONSENSUS_CONTROLLER_V10_SCHEMA_VERSION",
    "W75_TC_V10_DECISION_COMPOUND_CHAIN_PRESSURE",
    "W75_TC_V10_DECISION_COMPOUND_REPAIR_AFTER_RTR",
    "W75_TC_V10_DECISIONS",
    "TeamConsensusControllerV10",
    "TeamConsensusControllerV10Witness",
    "emit_team_consensus_controller_v10_witness",
]
