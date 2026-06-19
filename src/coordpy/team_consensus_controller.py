"""W66 M21 — Team-Consensus Controller.

The first capsule-native CoordPy module dedicated to **multi-agent
team consensus** under tight visible-token budgets. Composes:

* **weighted quorum** — each agent's per-turn vote is weighted by
  its substrate-conditioned confidence;
* **abstain head** — an agent abstains when its confidence is
  below the per-regime threshold;
* **substrate-replay fallback** — when no agent reaches quorum and
  the substrate-replay channel is trustworthy, the controller
  replays the prior consensus from the V11 substrate snapshot-
  diff;
* **transcript fallback** — only used when substrate-replay is
  also untrustworthy.

The controller is **regime-aware**: it accepts a
``regime`` parameter and applies different thresholds for the
three W66 regimes (baseline / team-consensus-under-budget /
team-failure-recovery).

Honest scope (W66)
------------------

* ``W66-L-TEAM-CONSENSUS-IN-REPO-CAP`` — the team-consensus
  controller operates on **in-repo MASC outcomes** only. It does
  not enforce consensus on real model outputs.
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
        "coordpy.team_consensus_controller requires numpy"
        ) from exc

from .multi_agent_substrate_coordinator_v2 import (
    W66_MASC_V2_REGIMES,
    W66_MASC_V2_REGIME_BASELINE,
    W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
    W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
)
from .tiny_substrate_v3 import _sha256_hex


W66_TEAM_CONSENSUS_CONTROLLER_SCHEMA_VERSION: str = (
    "coordpy.team_consensus_controller.v1")

W66_TC_DECISION_QUORUM: str = "quorum_merge"
W66_TC_DECISION_ABSTAIN: str = "abstain"
W66_TC_DECISION_SUBSTRATE_REPLAY: str = "substrate_replay"
W66_TC_DECISION_TRANSCRIPT_FALLBACK: str = (
    "transcript_fallback")
W66_TC_DECISIONS: tuple[str, ...] = (
    W66_TC_DECISION_QUORUM,
    W66_TC_DECISION_ABSTAIN,
    W66_TC_DECISION_SUBSTRATE_REPLAY,
    W66_TC_DECISION_TRANSCRIPT_FALLBACK,
)


@dataclasses.dataclass
class TeamConsensusController:
    quorum_threshold_baseline: float = 0.5
    quorum_threshold_tcub: float = 0.4
    quorum_threshold_tfr: float = 0.35
    confidence_floor_baseline: float = 0.35
    confidence_floor_tcub: float = 0.40
    confidence_floor_tfr: float = 0.25
    substrate_replay_trust_floor: float = 0.55
    transcript_trust_floor: float = 0.40
    audit: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W66_TEAM_CONSENSUS_CONTROLLER_SCHEMA_VERSION,
            "kind": "team_consensus_controller",
            "quorum_threshold_baseline": float(round(
                self.quorum_threshold_baseline, 12)),
            "quorum_threshold_tcub": float(round(
                self.quorum_threshold_tcub, 12)),
            "quorum_threshold_tfr": float(round(
                self.quorum_threshold_tfr, 12)),
            "confidence_floor_baseline": float(round(
                self.confidence_floor_baseline, 12)),
            "confidence_floor_tcub": float(round(
                self.confidence_floor_tcub, 12)),
            "confidence_floor_tfr": float(round(
                self.confidence_floor_tfr, 12)),
            "substrate_replay_trust_floor": float(round(
                self.substrate_replay_trust_floor, 12)),
            "transcript_trust_floor": float(round(
                self.transcript_trust_floor, 12)),
        })

    def _thresholds(self, regime: str) -> tuple[float, float]:
        if regime == W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET:
            return (
                float(self.quorum_threshold_tcub),
                float(self.confidence_floor_tcub))
        if regime == W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY:
            return (
                float(self.quorum_threshold_tfr),
                float(self.confidence_floor_tfr))
        return (
            float(self.quorum_threshold_baseline),
            float(self.confidence_floor_baseline))

    def decide(
            self, *,
            regime: str,
            agent_guesses: Sequence[float],
            agent_confidences: Sequence[float],
            substrate_replay_trust: float = 0.0,
            transcript_available: bool = False,
            transcript_trust: float = 0.0,
    ) -> dict[str, Any]:
        """Return the team-consensus decision.

        ``decision`` ∈ ``W66_TC_DECISIONS``. ``payload`` is the
        consensus guess (or empty if abstaining). ``rationale``
        explains which rule fired."""
        if regime not in W66_MASC_V2_REGIMES:
            raise ValueError(f"unknown regime {regime!r}")
        guesses = _np.asarray(
            agent_guesses, dtype=_np.float64)
        confidences = _np.asarray(
            agent_confidences, dtype=_np.float64)
        if int(guesses.size) != int(confidences.size):
            raise ValueError(
                "agent_guesses and agent_confidences must agree "
                "in length")
        quorum_thr, conf_floor = self._thresholds(regime)
        n_total = int(guesses.size)
        if n_total == 0:
            return self._abstain_record(
                regime=regime,
                rationale="no_agents")
        # Filter low-confidence agents.
        mask = confidences >= float(conf_floor)
        n_active = int(mask.sum())
        active_frac = float(n_active) / float(n_total)
        if active_frac >= float(quorum_thr):
            active_guesses = guesses[mask]
            active_conf = confidences[mask]
            denom = float(active_conf.sum())
            if denom > 0.0:
                weighted = float(
                    _np.dot(active_guesses, active_conf)
                    / denom)
            else:
                weighted = float(active_guesses.mean())
            self.audit.append({
                "decision": W66_TC_DECISION_QUORUM,
                "regime": str(regime),
                "n_active": int(n_active),
                "n_total": int(n_total),
                "weighted_payload": float(round(weighted, 12)),
                "active_frac": float(round(active_frac, 12)),
            })
            return {
                "decision": W66_TC_DECISION_QUORUM,
                "payload": float(weighted),
                "regime": str(regime),
                "rationale": "weighted_quorum_above_threshold",
                "active_frac": float(active_frac),
            }
        # Quorum failed. Try substrate-replay fallback.
        if (float(substrate_replay_trust)
                >= float(self.substrate_replay_trust_floor)):
            self.audit.append({
                "decision": W66_TC_DECISION_SUBSTRATE_REPLAY,
                "regime": str(regime),
                "substrate_replay_trust": float(round(
                    substrate_replay_trust, 12)),
            })
            # Substrate-replay returns prior consensus payload
            # (synthetic: mean of guesses weighted by trust).
            payload = float(_np.dot(
                guesses, confidences) / float(
                    max(confidences.sum(), 1e-9)))
            return {
                "decision": W66_TC_DECISION_SUBSTRATE_REPLAY,
                "payload": float(payload),
                "regime": str(regime),
                "rationale": (
                    "substrate_replay_trust_above_floor"),
                "active_frac": float(active_frac),
            }
        # Substrate-replay too untrusted. Try transcript.
        if (transcript_available
                and float(transcript_trust)
                >= float(self.transcript_trust_floor)):
            self.audit.append({
                "decision": W66_TC_DECISION_TRANSCRIPT_FALLBACK,
                "regime": str(regime),
                "transcript_trust": float(round(
                    transcript_trust, 12)),
            })
            return {
                "decision": W66_TC_DECISION_TRANSCRIPT_FALLBACK,
                "payload": float(guesses.mean()),
                "regime": str(regime),
                "rationale": "transcript_trust_above_floor",
                "active_frac": float(active_frac),
            }
        return self._abstain_record(
            regime=regime,
            rationale=(
                "no_quorum_no_substrate_replay_no_transcript"))

    def _abstain_record(
            self, *, regime: str, rationale: str,
    ) -> dict[str, Any]:
        self.audit.append({
            "decision": W66_TC_DECISION_ABSTAIN,
            "regime": str(regime),
            "rationale": str(rationale),
        })
        return {
            "decision": W66_TC_DECISION_ABSTAIN,
            "payload": None,
            "regime": str(regime),
            "rationale": str(rationale),
        }


@dataclasses.dataclass(frozen=True)
class TeamConsensusControllerWitness:
    schema: str
    controller_cid: str
    n_decisions: int
    decisions_summary: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_decisions": int(self.n_decisions),
            "decisions_summary": {
                k: int(v) for k, v in sorted(
                    self.decisions_summary.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "team_consensus_controller_witness",
            "witness": self.to_dict()})


def emit_team_consensus_controller_witness(
        controller: TeamConsensusController,
) -> TeamConsensusControllerWitness:
    summary: dict[str, int] = {
        d: 0 for d in W66_TC_DECISIONS}
    for entry in controller.audit:
        dec = str(entry.get("decision", ""))
        if dec in summary:
            summary[dec] += 1
    return TeamConsensusControllerWitness(
        schema=W66_TEAM_CONSENSUS_CONTROLLER_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_decisions=int(len(controller.audit)),
        decisions_summary=dict(summary),
    )


__all__ = [
    "W66_TEAM_CONSENSUS_CONTROLLER_SCHEMA_VERSION",
    "W66_TC_DECISION_QUORUM",
    "W66_TC_DECISION_ABSTAIN",
    "W66_TC_DECISION_SUBSTRATE_REPLAY",
    "W66_TC_DECISION_TRANSCRIPT_FALLBACK",
    "W66_TC_DECISIONS",
    "TeamConsensusController",
    "TeamConsensusControllerWitness",
    "emit_team_consensus_controller_witness",
]
