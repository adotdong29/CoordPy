"""W67 M21 — Team-Consensus Controller V2.

Strictly extends W66's ``coordpy.team_consensus_controller``. The
V2 controller adds:

* **branch-merge arbiter** — when two branches' guesses conflict
  the V2 controller reconciles by substrate-fork-and-merge
  (weighted mean over branch fingerprints + substrate-replay
  bias toward the higher-trust branch).
* **role-dropout-repair head** — when a role is missing for one
  or more turns the V2 controller fills in from a substrate-
  replay snapshot weighted by the surviving-roles trust.
* **per-regime thresholds for the W67 regimes** — adds
  ``role_dropout`` and ``branch_merge_reconciliation`` to the V1
  controller's three W66 regimes.

V2 preserves V1's decision semantics byte-for-byte under the V1
regimes (with a wrapping V2 audit ledger).

Honest scope (W67)
------------------

* ``W67-L-TEAM-CONSENSUS-V2-IN-REPO-CAP`` — operates on in-repo
  MASC V3 outcomes only.
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
        "coordpy.team_consensus_controller_v2 requires numpy"
        ) from exc

from .multi_agent_substrate_coordinator_v3 import (
    W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION,
    W67_MASC_V3_REGIME_ROLE_DROPOUT,
    W67_MASC_V3_REGIMES,
)
from .team_consensus_controller import (
    TeamConsensusController,
    W66_TC_DECISIONS,
    W66_TC_DECISION_ABSTAIN,
    W66_TC_DECISION_QUORUM,
    W66_TC_DECISION_SUBSTRATE_REPLAY,
    W66_TC_DECISION_TRANSCRIPT_FALLBACK,
)
from .tiny_substrate_v3 import _sha256_hex


W67_TEAM_CONSENSUS_CONTROLLER_V2_SCHEMA_VERSION: str = (
    "coordpy.team_consensus_controller_v2.v1")

W67_TC_V2_DECISION_BRANCH_MERGE_RECONCILIATION: str = (
    "branch_merge_reconciliation_merge")
W67_TC_V2_DECISION_ROLE_DROPOUT_REPAIR: str = (
    "role_dropout_repair")
W67_TC_V2_DECISIONS: tuple[str, ...] = (
    *W66_TC_DECISIONS,
    W67_TC_V2_DECISION_BRANCH_MERGE_RECONCILIATION,
    W67_TC_V2_DECISION_ROLE_DROPOUT_REPAIR,
)


@dataclasses.dataclass
class TeamConsensusControllerV2:
    inner_v1: TeamConsensusController = dataclasses.field(
        default_factory=TeamConsensusController)
    quorum_threshold_role_dropout: float = 0.30
    quorum_threshold_branch_merge: float = 0.40
    confidence_floor_role_dropout: float = 0.25
    confidence_floor_branch_merge: float = 0.30
    branch_merge_substrate_trust_floor: float = 0.5
    role_dropout_substrate_trust_floor: float = 0.5
    audit_v2: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W67_TEAM_CONSENSUS_CONTROLLER_V2_SCHEMA_VERSION,
            "kind": "team_consensus_controller_v2",
            "inner_v1_cid": str(self.inner_v1.cid()),
            "quorum_threshold_role_dropout": float(round(
                self.quorum_threshold_role_dropout, 12)),
            "quorum_threshold_branch_merge": float(round(
                self.quorum_threshold_branch_merge, 12)),
            "confidence_floor_role_dropout": float(round(
                self.confidence_floor_role_dropout, 12)),
            "confidence_floor_branch_merge": float(round(
                self.confidence_floor_branch_merge, 12)),
            "branch_merge_substrate_trust_floor": float(round(
                self.branch_merge_substrate_trust_floor, 12)),
            "role_dropout_substrate_trust_floor": float(round(
                self.role_dropout_substrate_trust_floor, 12)),
        })

    def _v2_thresholds(
            self, regime: str,
    ) -> tuple[float, float] | None:
        if regime == W67_MASC_V3_REGIME_ROLE_DROPOUT:
            return (
                float(self.quorum_threshold_role_dropout),
                float(self.confidence_floor_role_dropout))
        if regime == W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION:
            return (
                float(self.quorum_threshold_branch_merge),
                float(self.confidence_floor_branch_merge))
        return None

    def decide_v2(
            self, *,
            regime: str,
            agent_guesses: Sequence[float],
            agent_confidences: Sequence[float],
            substrate_replay_trust: float = 0.0,
            transcript_available: bool = False,
            transcript_trust: float = 0.0,
            branch_payloads_per_branch: (
                dict[str, Sequence[float]] | None) = None,
            branch_trusts: (
                dict[str, float] | None) = None,
            missing_role_indices: (
                Sequence[int] | None) = None,
    ) -> dict[str, Any]:
        """V2 decision routine.

        Routes through V1 for the W66 regimes. For the W67-only
        regimes (role_dropout, branch_merge_reconciliation) V2
        applies the new arbiters first."""
        if regime not in W67_MASC_V3_REGIMES:
            raise ValueError(f"unknown regime {regime!r}")
        # W67 branch-merge regime: try branch-merge arbiter first.
        if (regime
                == W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION
                and branch_payloads_per_branch is not None
                and len(branch_payloads_per_branch) >= 2
                and float(substrate_replay_trust)
                >= float(
                    self.branch_merge_substrate_trust_floor)):
            trusts = (
                branch_trusts if branch_trusts is not None
                else {b: 1.0
                      for b in branch_payloads_per_branch.keys()})
            bids = sorted(branch_payloads_per_branch.keys())
            payloads = [
                list(branch_payloads_per_branch[b]) for b in bids]
            weights = [
                float(trusts.get(b, 1.0)) for b in bids]
            if sum(weights) <= 0.0:
                weights = [1.0] * len(bids)
            sumw = float(sum(weights))
            Dim = max(len(p) for p in payloads)
            merged = [0.0] * Dim
            for pl, w in zip(payloads, weights):
                for i in range(min(Dim, len(pl))):
                    merged[i] += float(pl[i]) * float(w) / sumw
            self.audit_v2.append({
                "decision":
                    W67_TC_V2_DECISION_BRANCH_MERGE_RECONCILIATION,
                "regime": str(regime),
                "n_branches": int(len(bids)),
                "branch_ids": list(bids),
            })
            return {
                "decision":
                    W67_TC_V2_DECISION_BRANCH_MERGE_RECONCILIATION,
                "payload": (
                    float(merged[0]) if len(merged) == 1
                    else [float(x) for x in merged]),
                "regime": str(regime),
                "rationale": (
                    "branch_merge_reconciliation_merge_applied"),
                "n_branches": int(len(bids)),
            }
        # W67 role-dropout regime: try role-dropout-repair head first.
        if (regime == W67_MASC_V3_REGIME_ROLE_DROPOUT
                and missing_role_indices is not None
                and len(missing_role_indices) >= 1
                and float(substrate_replay_trust)
                >= float(
                    self.role_dropout_substrate_trust_floor)):
            present_idx = [
                i for i in range(len(agent_guesses))
                if i not in set(missing_role_indices)]
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
                self.audit_v2.append({
                    "decision":
                        W67_TC_V2_DECISION_ROLE_DROPOUT_REPAIR,
                    "regime": str(regime),
                    "n_missing": int(len(missing_role_indices)),
                    "n_present": int(len(present_idx)),
                })
                return {
                    "decision":
                        W67_TC_V2_DECISION_ROLE_DROPOUT_REPAIR,
                    "payload": float(repaired),
                    "regime": str(regime),
                    "rationale":
                        "role_dropout_repair_head_applied",
                    "n_missing": int(len(missing_role_indices)),
                }
        # Fall back to V1 for the W66 regimes.
        if regime in (
                W67_MASC_V3_REGIME_ROLE_DROPOUT,
                W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION):
            # No arbiter fired — apply V1 quorum/abstain/replay/
            # transcript chain treating the regime as baseline.
            v1_decision = self.inner_v1.decide(
                regime="baseline",
                agent_guesses=list(agent_guesses),
                agent_confidences=list(agent_confidences),
                substrate_replay_trust=float(
                    substrate_replay_trust),
                transcript_available=bool(transcript_available),
                transcript_trust=float(transcript_trust))
        else:
            v1_decision = self.inner_v1.decide(
                regime=str(regime),
                agent_guesses=list(agent_guesses),
                agent_confidences=list(agent_confidences),
                substrate_replay_trust=float(
                    substrate_replay_trust),
                transcript_available=bool(transcript_available),
                transcript_trust=float(transcript_trust))
        self.audit_v2.append({
            "decision": str(v1_decision.get("decision", "")),
            "regime": str(regime),
            "v1_fallback": True,
        })
        return dict(v1_decision)


@dataclasses.dataclass(frozen=True)
class TeamConsensusControllerV2Witness:
    schema: str
    controller_v2_cid: str
    inner_v1_witness_cid: str
    n_decisions_v2: int
    decisions_summary_v2: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_v2_cid": str(self.controller_v2_cid),
            "inner_v1_witness_cid": str(
                self.inner_v1_witness_cid),
            "n_decisions_v2": int(self.n_decisions_v2),
            "decisions_summary_v2": {
                k: int(v) for k, v in sorted(
                    self.decisions_summary_v2.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "team_consensus_controller_v2_witness",
            "witness": self.to_dict()})


def emit_team_consensus_controller_v2_witness(
        controller: TeamConsensusControllerV2,
) -> TeamConsensusControllerV2Witness:
    from .team_consensus_controller import (
        emit_team_consensus_controller_witness,
    )
    inner_w = emit_team_consensus_controller_witness(
        controller.inner_v1)
    summary: dict[str, int] = {
        d: 0 for d in W67_TC_V2_DECISIONS}
    for entry in controller.audit_v2:
        dec = str(entry.get("decision", ""))
        if dec in summary:
            summary[dec] += 1
    return TeamConsensusControllerV2Witness(
        schema=W67_TEAM_CONSENSUS_CONTROLLER_V2_SCHEMA_VERSION,
        controller_v2_cid=str(controller.cid()),
        inner_v1_witness_cid=str(inner_w.cid()),
        n_decisions_v2=int(len(controller.audit_v2)),
        decisions_summary_v2=dict(summary),
    )


__all__ = [
    "W67_TEAM_CONSENSUS_CONTROLLER_V2_SCHEMA_VERSION",
    "W67_TC_V2_DECISION_BRANCH_MERGE_RECONCILIATION",
    "W67_TC_V2_DECISION_ROLE_DROPOUT_REPAIR",
    "W67_TC_V2_DECISIONS",
    "TeamConsensusControllerV2",
    "TeamConsensusControllerV2Witness",
    "emit_team_consensus_controller_v2_witness",
]
