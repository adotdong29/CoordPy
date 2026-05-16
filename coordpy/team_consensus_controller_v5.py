"""W70 M12 — Team-Consensus Controller V5.

Strictly extends W69's ``coordpy.team_consensus_controller_v4``. The
V5 controller adds:

* **repair-dominance arbiter** — when a non-zero
  dominant_repair_label is supplied with substrate-replay trust
  ≥ floor, the V5 controller picks the weighted-mean of confidence-
  aligned agents whose repair label matches the dominant repair.
* **budget-primary arbiter** — when the visible-token budget ratio
  is below the V5 budget-primary threshold and substrate-replay
  trust ≥ floor, the V5 controller picks the highest-confidence
  agent and routes through Plane B (substrate). Saves visible
  tokens on tight-budget turns.
* **per-regime thresholds for the W70 regime** — adds
  ``contradiction_then_rejoin_under_budget`` to the V4 nine W69
  regimes.

V5 preserves V4's decision semantics byte-for-byte under the W69
regimes (with a wrapping V5 audit ledger).

Honest scope (W70)
------------------

* ``W70-L-TEAM-CONSENSUS-V5-IN-REPO-CAP`` — operates on in-repo
  MASC V6 outcomes only.
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
        "coordpy.team_consensus_controller_v5 requires numpy"
        ) from exc

from .multi_agent_substrate_coordinator_v6 import (
    W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET,
    W70_MASC_V6_REGIMES,
)
from .team_consensus_controller_v4 import (
    TeamConsensusControllerV4,
    W69_TC_V4_DECISIONS,
)
from .tiny_substrate_v3 import _sha256_hex


W70_TEAM_CONSENSUS_CONTROLLER_V5_SCHEMA_VERSION: str = (
    "coordpy.team_consensus_controller_v5.v1")

W70_TC_V5_DECISION_REPAIR_DOMINANCE: str = (
    "repair_dominance_arbiter")
W70_TC_V5_DECISION_BUDGET_PRIMARY: str = (
    "budget_primary_arbiter")
W70_TC_V5_DECISION_CONTRADICTION_THEN_REJOIN: str = (
    "contradiction_then_rejoin_under_budget_arbiter")
W70_TC_V5_DECISIONS: tuple[str, ...] = (
    *W69_TC_V4_DECISIONS,
    W70_TC_V5_DECISION_REPAIR_DOMINANCE,
    W70_TC_V5_DECISION_BUDGET_PRIMARY,
    W70_TC_V5_DECISION_CONTRADICTION_THEN_REJOIN,
)


@dataclasses.dataclass
class TeamConsensusControllerV5:
    inner_v4: TeamConsensusControllerV4 = dataclasses.field(
        default_factory=TeamConsensusControllerV4)
    repair_dominance_substrate_trust_floor: float = 0.5
    budget_primary_substrate_trust_floor: float = 0.5
    contradiction_then_rejoin_substrate_trust_floor: float = 0.5
    budget_primary_visible_token_ratio_floor: float = 0.5
    audit_v5: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W70_TEAM_CONSENSUS_CONTROLLER_V5_SCHEMA_VERSION,
            "kind": "team_consensus_controller_v5",
            "inner_v4_cid": str(self.inner_v4.cid()),
            "repair_dominance_substrate_trust_floor": float(
                round(
                    self
                    .repair_dominance_substrate_trust_floor,
                    12)),
            "budget_primary_substrate_trust_floor": float(round(
                self.budget_primary_substrate_trust_floor, 12)),
            "contradiction_then_rejoin_substrate_trust_floor":
                float(round(
                    self
                    .contradiction_then_rejoin_substrate_trust_floor,
                    12)),
            "budget_primary_visible_token_ratio_floor": float(
                round(
                    self
                    .budget_primary_visible_token_ratio_floor,
                    12)),
        })

    def decide_v5(
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
            **v4_kwargs: Any,
    ) -> dict[str, Any]:
        """V5 decision routine.

        Routes through V4 for the W69 regimes (and the V5 budget-
        primary / repair-dominance arbiters fire on top when their
        conditions hold). For the W70-only compound regime V5
        applies a coordinated arbiter."""
        if regime not in W70_MASC_V6_REGIMES:
            raise ValueError(f"unknown regime {regime!r}")
        # W70 compound regime — coordinated PC-then-MBR arbiter.
        if (regime == (
                W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET)
                and branch_assignments is not None
                and len(branch_assignments) == len(agent_guesses)
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .contradiction_then_rejoin_substrate_trust_floor)):
            conf = _np.asarray(
                list(agent_confidences), dtype=_np.float64)
            gs = _np.asarray(
                list(agent_guesses), dtype=_np.float64)
            denom = float(conf.sum())
            if denom > 0.0:
                repaired = float(_np.dot(gs, conf) / denom)
            else:
                repaired = float(gs.mean())
            self.audit_v5.append({
                "decision":
                    W70_TC_V5_DECISION_CONTRADICTION_THEN_REJOIN,
                "regime": str(regime),
                "n_agents": int(len(agent_guesses)),
            })
            return {
                "decision":
                    W70_TC_V5_DECISION_CONTRADICTION_THEN_REJOIN,
                "payload": float(repaired),
                "regime": str(regime),
                "rationale":
                    ("contradiction_then_rejoin_under_budget_"
                     "arbiter_applied"),
            }
        # V5 repair-dominance arbiter (any regime + non-zero label).
        if (int(dominant_repair_label) > 0
                and agent_repair_labels is not None
                and len(agent_repair_labels) == len(
                    agent_guesses)
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .repair_dominance_substrate_trust_floor)):
            matching_idx = [
                i for i, lab in enumerate(agent_repair_labels)
                if int(lab) == int(dominant_repair_label)]
            if matching_idx:
                gs = _np.asarray(
                    [agent_guesses[i] for i in matching_idx],
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
                self.audit_v5.append({
                    "decision":
                        W70_TC_V5_DECISION_REPAIR_DOMINANCE,
                    "regime": str(regime),
                    "n_matching_agents": int(len(matching_idx)),
                    "dominant_repair_label": int(
                        dominant_repair_label),
                })
                return {
                    "decision":
                        W70_TC_V5_DECISION_REPAIR_DOMINANCE,
                    "payload": float(repaired),
                    "regime": str(regime),
                    "rationale":
                        "repair_dominance_arbiter_applied",
                    "dominant_repair_label": int(
                        dominant_repair_label),
                }
        # V5 budget-primary arbiter (any regime + tight budget).
        if (float(visible_token_budget_ratio)
                < float(
                    self.budget_primary_visible_token_ratio_floor)
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .budget_primary_substrate_trust_floor)):
            confs = _np.asarray(
                list(agent_confidences), dtype=_np.float64)
            if confs.size > 0:
                top_idx = int(_np.argmax(confs))
                payload = float(agent_guesses[top_idx])
                self.audit_v5.append({
                    "decision":
                        W70_TC_V5_DECISION_BUDGET_PRIMARY,
                    "regime": str(regime),
                    "top_agent_idx": int(top_idx),
                    "visible_token_budget_ratio": float(
                        visible_token_budget_ratio),
                })
                return {
                    "decision":
                        W70_TC_V5_DECISION_BUDGET_PRIMARY,
                    "payload": float(payload),
                    "regime": str(regime),
                    "rationale":
                        "budget_primary_arbiter_applied",
                    "top_agent_idx": int(top_idx),
                }
        # Fall back to V4.
        if regime in (
                W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET,):
            v4_decision = self.inner_v4.decide_v4(
                regime="baseline",
                agent_guesses=list(agent_guesses),
                agent_confidences=list(agent_confidences),
                substrate_replay_trust=float(
                    substrate_replay_trust),
                transcript_available=bool(transcript_available),
                transcript_trust=float(transcript_trust),
                branch_assignments=(
                    list(branch_assignments)
                    if branch_assignments is not None else None),
                corrupted_role_indices=(
                    list(corrupted_role_indices)
                    if corrupted_role_indices is not None
                    else None),
                **v4_kwargs)
        else:
            v4_decision = self.inner_v4.decide_v4(
                regime=str(regime),
                agent_guesses=list(agent_guesses),
                agent_confidences=list(agent_confidences),
                substrate_replay_trust=float(
                    substrate_replay_trust),
                transcript_available=bool(transcript_available),
                transcript_trust=float(transcript_trust),
                branch_assignments=(
                    list(branch_assignments)
                    if branch_assignments is not None else None),
                corrupted_role_indices=(
                    list(corrupted_role_indices)
                    if corrupted_role_indices is not None
                    else None),
                **v4_kwargs)
        self.audit_v5.append({
            "decision": str(v4_decision.get("decision", "")),
            "regime": str(regime),
            "v4_fallback": True,
        })
        return dict(v4_decision)


@dataclasses.dataclass(frozen=True)
class TeamConsensusControllerV5Witness:
    schema: str
    controller_v5_cid: str
    inner_v4_witness_cid: str
    n_decisions_v5: int
    decisions_summary_v5: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_v5_cid": str(self.controller_v5_cid),
            "inner_v4_witness_cid": str(
                self.inner_v4_witness_cid),
            "n_decisions_v5": int(self.n_decisions_v5),
            "decisions_summary_v5": {
                k: int(v) for k, v in sorted(
                    self.decisions_summary_v5.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "team_consensus_controller_v5_witness",
            "witness": self.to_dict()})


def emit_team_consensus_controller_v5_witness(
        controller: TeamConsensusControllerV5,
) -> TeamConsensusControllerV5Witness:
    from .team_consensus_controller_v4 import (
        emit_team_consensus_controller_v4_witness,
    )
    inner_w = emit_team_consensus_controller_v4_witness(
        controller.inner_v4)
    summary: dict[str, int] = {
        d: 0 for d in W70_TC_V5_DECISIONS}
    for entry in controller.audit_v5:
        dec = str(entry.get("decision", ""))
        if dec in summary:
            summary[dec] += 1
    return TeamConsensusControllerV5Witness(
        schema=W70_TEAM_CONSENSUS_CONTROLLER_V5_SCHEMA_VERSION,
        controller_v5_cid=str(controller.cid()),
        inner_v4_witness_cid=str(inner_w.cid()),
        n_decisions_v5=int(len(controller.audit_v5)),
        decisions_summary_v5=dict(summary),
    )


__all__ = [
    "W70_TEAM_CONSENSUS_CONTROLLER_V5_SCHEMA_VERSION",
    "W70_TC_V5_DECISION_REPAIR_DOMINANCE",
    "W70_TC_V5_DECISION_BUDGET_PRIMARY",
    "W70_TC_V5_DECISION_CONTRADICTION_THEN_REJOIN",
    "W70_TC_V5_DECISIONS",
    "TeamConsensusControllerV5",
    "TeamConsensusControllerV5Witness",
    "emit_team_consensus_controller_v5_witness",
]
