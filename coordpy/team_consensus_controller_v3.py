"""W68 M21 — Team-Consensus Controller V3.

Strictly extends W67's ``coordpy.team_consensus_controller_v2``. The
V3 controller adds:

* **partial-contradiction arbiter** — when two or more agents'
  payloads contradict (sign-flipped guesses) the V3 controller
  reconciles by substrate-replay weighted-mean and pulls toward
  the most-confident non-contradicting subgroup.
* **agent-replacement-warm-restart arbiter** — when a role is
  replaced mid-run the V3 controller fills in from a
  substrate-prefix-reuse snapshot weighted by the surviving-agents
  trust.
* **per-regime thresholds for the W68 regimes** — adds
  ``partial_contradiction_under_delayed_reconciliation`` and
  ``agent_replacement_warm_restart`` to the V2 controller's five
  W67 regimes.

V3 preserves V2's decision semantics byte-for-byte under the V2
regimes (with a wrapping V3 audit ledger).

Honest scope (W68)
------------------

* ``W68-L-TEAM-CONSENSUS-V3-IN-REPO-CAP`` — operates on in-repo
  MASC V4 outcomes only.
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
        "coordpy.team_consensus_controller_v3 requires numpy"
        ) from exc

from .multi_agent_substrate_coordinator_v4 import (
    W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
    W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
    W68_MASC_V4_REGIMES,
)
from .team_consensus_controller_v2 import (
    TeamConsensusControllerV2,
    W67_TC_V2_DECISIONS,
)
from .tiny_substrate_v3 import _sha256_hex


W68_TEAM_CONSENSUS_CONTROLLER_V3_SCHEMA_VERSION: str = (
    "coordpy.team_consensus_controller_v3.v1")

W68_TC_V3_DECISION_PARTIAL_CONTRADICTION: str = (
    "partial_contradiction_arbiter")
W68_TC_V3_DECISION_AGENT_REPLACEMENT_WARM_RESTART: str = (
    "agent_replacement_warm_restart_arbiter")
W68_TC_V3_DECISIONS: tuple[str, ...] = (
    *W67_TC_V2_DECISIONS,
    W68_TC_V3_DECISION_PARTIAL_CONTRADICTION,
    W68_TC_V3_DECISION_AGENT_REPLACEMENT_WARM_RESTART,
)


@dataclasses.dataclass
class TeamConsensusControllerV3:
    inner_v2: TeamConsensusControllerV2 = dataclasses.field(
        default_factory=TeamConsensusControllerV2)
    quorum_threshold_partial_contradiction: float = 0.35
    quorum_threshold_agent_replacement: float = 0.35
    confidence_floor_partial_contradiction: float = 0.30
    confidence_floor_agent_replacement: float = 0.30
    partial_contradiction_substrate_trust_floor: float = 0.5
    agent_replacement_substrate_trust_floor: float = 0.5
    audit_v3: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W68_TEAM_CONSENSUS_CONTROLLER_V3_SCHEMA_VERSION,
            "kind": "team_consensus_controller_v3",
            "inner_v2_cid": str(self.inner_v2.cid()),
            "quorum_threshold_partial_contradiction": float(round(
                self.quorum_threshold_partial_contradiction, 12)),
            "quorum_threshold_agent_replacement": float(round(
                self.quorum_threshold_agent_replacement, 12)),
            "confidence_floor_partial_contradiction": float(round(
                self.confidence_floor_partial_contradiction, 12)),
            "confidence_floor_agent_replacement": float(round(
                self.confidence_floor_agent_replacement, 12)),
            "partial_contradiction_substrate_trust_floor": float(
                round(
                    self.partial_contradiction_substrate_trust_floor,
                    12)),
            "agent_replacement_substrate_trust_floor": float(round(
                self.agent_replacement_substrate_trust_floor, 12)),
        })

    def decide_v3(
            self, *,
            regime: str,
            agent_guesses: Sequence[float],
            agent_confidences: Sequence[float],
            substrate_replay_trust: float = 0.0,
            transcript_available: bool = False,
            transcript_trust: float = 0.0,
            contradiction_indicators: (
                Sequence[float] | None) = None,
            replaced_role_indices: (
                Sequence[int] | None) = None,
            **v2_kwargs: Any,
    ) -> dict[str, Any]:
        """V3 decision routine.

        Routes through V2 for the W67 regimes. For the W68-only
        regimes V3 applies the new arbiters first."""
        if regime not in W68_MASC_V4_REGIMES:
            raise ValueError(f"unknown regime {regime!r}")
        # W68 partial-contradiction regime: try arbiter first.
        if (regime
                == W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION
                and contradiction_indicators is not None
                and len(contradiction_indicators)
                    == len(agent_guesses)
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .partial_contradiction_substrate_trust_floor)):
            non_contradicting = [
                i for i, c in enumerate(contradiction_indicators)
                if float(c) < 0.5]
            if non_contradicting:
                guesses = _np.asarray(
                    [agent_guesses[i] for i in non_contradicting],
                    dtype=_np.float64)
                conf = _np.asarray(
                    [agent_confidences[i]
                     for i in non_contradicting],
                    dtype=_np.float64)
                denom = float(conf.sum())
                if denom > 0.0:
                    repaired = float(
                        _np.dot(guesses, conf) / denom)
                else:
                    repaired = float(guesses.mean())
                self.audit_v3.append({
                    "decision":
                        W68_TC_V3_DECISION_PARTIAL_CONTRADICTION,
                    "regime": str(regime),
                    "n_contradicting": int(
                        len(contradiction_indicators)
                        - len(non_contradicting)),
                    "n_kept": int(len(non_contradicting)),
                })
                return {
                    "decision":
                        W68_TC_V3_DECISION_PARTIAL_CONTRADICTION,
                    "payload": float(repaired),
                    "regime": str(regime),
                    "rationale":
                        "partial_contradiction_arbiter_applied",
                    "n_contradicting": int(
                        len(contradiction_indicators)
                        - len(non_contradicting)),
                }
        # W68 agent-replacement-warm-restart regime: try arbiter first.
        if (regime
                == W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART
                and replaced_role_indices is not None
                and len(replaced_role_indices) >= 1
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .agent_replacement_substrate_trust_floor)):
            present_idx = [
                i for i in range(len(agent_guesses))
                if i not in set(replaced_role_indices)]
            if present_idx:
                surviving = _np.asarray(
                    [agent_guesses[i] for i in present_idx],
                    dtype=_np.float64)
                surv_conf = _np.asarray(
                    [agent_confidences[i] for i in present_idx],
                    dtype=_np.float64)
                denom = float(surv_conf.sum())
                if denom > 0.0:
                    repaired = float(
                        _np.dot(surviving, surv_conf) / denom)
                else:
                    repaired = float(surviving.mean())
                self.audit_v3.append({
                    "decision": (
                        W68_TC_V3_DECISION_AGENT_REPLACEMENT_WARM_RESTART),
                    "regime": str(regime),
                    "n_replaced": int(len(replaced_role_indices)),
                    "n_present": int(len(present_idx)),
                })
                return {
                    "decision": (
                        W68_TC_V3_DECISION_AGENT_REPLACEMENT_WARM_RESTART),
                    "payload": float(repaired),
                    "regime": str(regime),
                    "rationale": (
                        "agent_replacement_warm_restart_arbiter_applied"),
                    "n_replaced": int(len(replaced_role_indices)),
                }
        # Fall back to V2 for the W67 regimes (and any W68 regime
        # where the V3 arbiter did not fire).
        if regime in (
                W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
                W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART):
            v2_decision = self.inner_v2.decide_v2(
                regime="baseline",
                agent_guesses=list(agent_guesses),
                agent_confidences=list(agent_confidences),
                substrate_replay_trust=float(
                    substrate_replay_trust),
                transcript_available=bool(transcript_available),
                transcript_trust=float(transcript_trust),
                **v2_kwargs)
        else:
            v2_decision = self.inner_v2.decide_v2(
                regime=str(regime),
                agent_guesses=list(agent_guesses),
                agent_confidences=list(agent_confidences),
                substrate_replay_trust=float(
                    substrate_replay_trust),
                transcript_available=bool(transcript_available),
                transcript_trust=float(transcript_trust),
                **v2_kwargs)
        self.audit_v3.append({
            "decision": str(v2_decision.get("decision", "")),
            "regime": str(regime),
            "v2_fallback": True,
        })
        return dict(v2_decision)


@dataclasses.dataclass(frozen=True)
class TeamConsensusControllerV3Witness:
    schema: str
    controller_v3_cid: str
    inner_v2_witness_cid: str
    n_decisions_v3: int
    decisions_summary_v3: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_v3_cid": str(self.controller_v3_cid),
            "inner_v2_witness_cid": str(
                self.inner_v2_witness_cid),
            "n_decisions_v3": int(self.n_decisions_v3),
            "decisions_summary_v3": {
                k: int(v) for k, v in sorted(
                    self.decisions_summary_v3.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "team_consensus_controller_v3_witness",
            "witness": self.to_dict()})


def emit_team_consensus_controller_v3_witness(
        controller: TeamConsensusControllerV3,
) -> TeamConsensusControllerV3Witness:
    from .team_consensus_controller_v2 import (
        emit_team_consensus_controller_v2_witness,
    )
    inner_w = emit_team_consensus_controller_v2_witness(
        controller.inner_v2)
    summary: dict[str, int] = {
        d: 0 for d in W68_TC_V3_DECISIONS}
    for entry in controller.audit_v3:
        dec = str(entry.get("decision", ""))
        if dec in summary:
            summary[dec] += 1
    return TeamConsensusControllerV3Witness(
        schema=W68_TEAM_CONSENSUS_CONTROLLER_V3_SCHEMA_VERSION,
        controller_v3_cid=str(controller.cid()),
        inner_v2_witness_cid=str(inner_w.cid()),
        n_decisions_v3=int(len(controller.audit_v3)),
        decisions_summary_v3=dict(summary),
    )


__all__ = [
    "W68_TEAM_CONSENSUS_CONTROLLER_V3_SCHEMA_VERSION",
    "W68_TC_V3_DECISION_PARTIAL_CONTRADICTION",
    "W68_TC_V3_DECISION_AGENT_REPLACEMENT_WARM_RESTART",
    "W68_TC_V3_DECISIONS",
    "TeamConsensusControllerV3",
    "TeamConsensusControllerV3Witness",
    "emit_team_consensus_controller_v3_witness",
]
