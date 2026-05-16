"""W69 M19 — Team-Consensus Controller V4.

Strictly extends W68's ``coordpy.team_consensus_controller_v3``. The
V4 controller adds:

* **multi-branch-rejoin arbiter** — when N agents fork into M
  branches with conflicting payloads and the team must rejoin, the
  V4 controller picks the weighted-mean of confidence-aligned
  branches.
* **silent-corruption-plus-member-replacement arbiter** — when a
  role's substrate is silently corrupted and the role is replaced
  with a fresh member, the V4 controller fills in from the
  surviving-agents weighted-mean.
* **per-regime thresholds for the W69 regimes** — adds
  ``multi_branch_rejoin_after_divergent_work`` and
  ``silent_corruption_plus_member_replacement`` to the V3
  controller's seven W68 regimes.

V4 preserves V3's decision semantics byte-for-byte under the W68
regimes (with a wrapping V4 audit ledger).

Honest scope (W69)
------------------

* ``W69-L-TEAM-CONSENSUS-V4-IN-REPO-CAP`` — operates on in-repo
  MASC V5 outcomes only.
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
        "coordpy.team_consensus_controller_v4 requires numpy"
        ) from exc

from .multi_agent_substrate_coordinator_v5 import (
    W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
    W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT,
    W69_MASC_V5_REGIMES,
)
from .team_consensus_controller_v3 import (
    TeamConsensusControllerV3,
    W68_TC_V3_DECISIONS,
)
from .tiny_substrate_v3 import _sha256_hex


W69_TEAM_CONSENSUS_CONTROLLER_V4_SCHEMA_VERSION: str = (
    "coordpy.team_consensus_controller_v4.v1")

W69_TC_V4_DECISION_MULTI_BRANCH_REJOIN: str = (
    "multi_branch_rejoin_arbiter")
W69_TC_V4_DECISION_SILENT_CORRUPTION_PLUS_REPLACEMENT: str = (
    "silent_corruption_plus_member_replacement_arbiter")
W69_TC_V4_DECISIONS: tuple[str, ...] = (
    *W68_TC_V3_DECISIONS,
    W69_TC_V4_DECISION_MULTI_BRANCH_REJOIN,
    W69_TC_V4_DECISION_SILENT_CORRUPTION_PLUS_REPLACEMENT,
)


@dataclasses.dataclass
class TeamConsensusControllerV4:
    inner_v3: TeamConsensusControllerV3 = dataclasses.field(
        default_factory=TeamConsensusControllerV3)
    quorum_threshold_multi_branch_rejoin: float = 0.35
    quorum_threshold_silent_corruption: float = 0.35
    confidence_floor_multi_branch_rejoin: float = 0.30
    confidence_floor_silent_corruption: float = 0.30
    multi_branch_rejoin_substrate_trust_floor: float = 0.5
    silent_corruption_substrate_trust_floor: float = 0.5
    audit_v4: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W69_TEAM_CONSENSUS_CONTROLLER_V4_SCHEMA_VERSION,
            "kind": "team_consensus_controller_v4",
            "inner_v3_cid": str(self.inner_v3.cid()),
            "quorum_threshold_multi_branch_rejoin": float(round(
                self.quorum_threshold_multi_branch_rejoin, 12)),
            "quorum_threshold_silent_corruption": float(round(
                self.quorum_threshold_silent_corruption, 12)),
            "confidence_floor_multi_branch_rejoin": float(round(
                self.confidence_floor_multi_branch_rejoin, 12)),
            "confidence_floor_silent_corruption": float(round(
                self.confidence_floor_silent_corruption, 12)),
            "multi_branch_rejoin_substrate_trust_floor": float(
                round(
                    self
                    .multi_branch_rejoin_substrate_trust_floor,
                    12)),
            "silent_corruption_substrate_trust_floor": float(
                round(
                    self.silent_corruption_substrate_trust_floor,
                    12)),
        })

    def decide_v4(
            self, *,
            regime: str,
            agent_guesses: Sequence[float],
            agent_confidences: Sequence[float],
            substrate_replay_trust: float = 0.0,
            transcript_available: bool = False,
            transcript_trust: float = 0.0,
            branch_assignments: (
                Sequence[int] | None) = None,
            corrupted_role_indices: (
                Sequence[int] | None) = None,
            **v3_kwargs: Any,
    ) -> dict[str, Any]:
        """V4 decision routine.

        Routes through V3 for the W68 regimes. For the W69-only
        regimes V4 applies the new arbiters first."""
        if regime not in W69_MASC_V5_REGIMES:
            raise ValueError(f"unknown regime {regime!r}")
        # W69 multi-branch-rejoin regime.
        if (regime == W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN
                and branch_assignments is not None
                and len(branch_assignments) == len(agent_guesses)
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .multi_branch_rejoin_substrate_trust_floor)):
            n_branches = len(set(int(b)
                                 for b in branch_assignments))
            if n_branches >= 2:
                # Weighted mean across all branches; favour higher-
                # confidence agents within each branch.
                conf = _np.asarray(
                    list(agent_confidences), dtype=_np.float64)
                gs = _np.asarray(
                    list(agent_guesses), dtype=_np.float64)
                denom = float(conf.sum())
                if denom > 0.0:
                    repaired = float(_np.dot(gs, conf) / denom)
                else:
                    repaired = float(gs.mean())
                self.audit_v4.append({
                    "decision":
                        W69_TC_V4_DECISION_MULTI_BRANCH_REJOIN,
                    "regime": str(regime),
                    "n_branches": int(n_branches),
                    "n_agents": int(len(agent_guesses)),
                })
                return {
                    "decision":
                        W69_TC_V4_DECISION_MULTI_BRANCH_REJOIN,
                    "payload": float(repaired),
                    "regime": str(regime),
                    "rationale":
                        "multi_branch_rejoin_arbiter_applied",
                    "n_branches": int(n_branches),
                }
        # W69 silent-corruption-plus-member-replacement regime.
        if (regime
                == W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT
                and corrupted_role_indices is not None
                and len(corrupted_role_indices) >= 1
                and float(substrate_replay_trust)
                    >= float(
                        self
                        .silent_corruption_substrate_trust_floor)):
            present_idx = [
                i for i in range(len(agent_guesses))
                if i not in set(corrupted_role_indices)]
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
                self.audit_v4.append({
                    "decision": (
                        W69_TC_V4_DECISION_SILENT_CORRUPTION_PLUS_REPLACEMENT),
                    "regime": str(regime),
                    "n_corrupted": int(
                        len(corrupted_role_indices)),
                    "n_present": int(len(present_idx)),
                })
                return {
                    "decision": (
                        W69_TC_V4_DECISION_SILENT_CORRUPTION_PLUS_REPLACEMENT),
                    "payload": float(repaired),
                    "regime": str(regime),
                    "rationale": (
                        "silent_corruption_plus_member_replacement_arbiter_applied"),
                    "n_corrupted": int(
                        len(corrupted_role_indices)),
                }
        # Fall back to V3.
        if regime in (
                W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
                W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT):
            v3_decision = self.inner_v3.decide_v3(
                regime="baseline",
                agent_guesses=list(agent_guesses),
                agent_confidences=list(agent_confidences),
                substrate_replay_trust=float(
                    substrate_replay_trust),
                transcript_available=bool(transcript_available),
                transcript_trust=float(transcript_trust),
                **v3_kwargs)
        else:
            v3_decision = self.inner_v3.decide_v3(
                regime=str(regime),
                agent_guesses=list(agent_guesses),
                agent_confidences=list(agent_confidences),
                substrate_replay_trust=float(
                    substrate_replay_trust),
                transcript_available=bool(transcript_available),
                transcript_trust=float(transcript_trust),
                **v3_kwargs)
        self.audit_v4.append({
            "decision": str(v3_decision.get("decision", "")),
            "regime": str(regime),
            "v3_fallback": True,
        })
        return dict(v3_decision)


@dataclasses.dataclass(frozen=True)
class TeamConsensusControllerV4Witness:
    schema: str
    controller_v4_cid: str
    inner_v3_witness_cid: str
    n_decisions_v4: int
    decisions_summary_v4: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_v4_cid": str(self.controller_v4_cid),
            "inner_v3_witness_cid": str(
                self.inner_v3_witness_cid),
            "n_decisions_v4": int(self.n_decisions_v4),
            "decisions_summary_v4": {
                k: int(v) for k, v in sorted(
                    self.decisions_summary_v4.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "team_consensus_controller_v4_witness",
            "witness": self.to_dict()})


def emit_team_consensus_controller_v4_witness(
        controller: TeamConsensusControllerV4,
) -> TeamConsensusControllerV4Witness:
    from .team_consensus_controller_v3 import (
        emit_team_consensus_controller_v3_witness,
    )
    inner_w = emit_team_consensus_controller_v3_witness(
        controller.inner_v3)
    summary: dict[str, int] = {
        d: 0 for d in W69_TC_V4_DECISIONS}
    for entry in controller.audit_v4:
        dec = str(entry.get("decision", ""))
        if dec in summary:
            summary[dec] += 1
    return TeamConsensusControllerV4Witness(
        schema=W69_TEAM_CONSENSUS_CONTROLLER_V4_SCHEMA_VERSION,
        controller_v4_cid=str(controller.cid()),
        inner_v3_witness_cid=str(inner_w.cid()),
        n_decisions_v4=int(len(controller.audit_v4)),
        decisions_summary_v4=dict(summary),
    )


__all__ = [
    "W69_TEAM_CONSENSUS_CONTROLLER_V4_SCHEMA_VERSION",
    "W69_TC_V4_DECISION_MULTI_BRANCH_REJOIN",
    "W69_TC_V4_DECISION_SILENT_CORRUPTION_PLUS_REPLACEMENT",
    "W69_TC_V4_DECISIONS",
    "TeamConsensusControllerV4",
    "TeamConsensusControllerV4Witness",
    "emit_team_consensus_controller_v4_witness",
]
