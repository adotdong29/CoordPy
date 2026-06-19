"""W54 M4 — Consensus / Quorum Controller.

A first-class K-of-N consensus controller wired on top of MLSC
V2's ``compute_consensus_quorum_v2``. Adds:

* **explicit policy** — ``K_min, K_max, cosine_floor,
  fallback_cosine_floor`` are explicit knobs
* **abstain-with-fallback semantics** — returns one of
  {quorum_merged, fallback_best_parent, abstain}
* **content-addressed K-of-N audit trail** — every consensus
  call appends a typed entry that records the K chosen, the N
  observed, the policy parameters, the resulting capsule CID,
  and the reason for the chosen branch

Honest scope: pure-Python only. Not a Byzantine fault-tolerant
distributed consensus protocol — only a capsule-layer agreement
mechanism (see W54-L-MLSC-V2-DISAGREEMENT-IS-NOT-CONFLICT-
RESOLUTION).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Mapping, Sequence

from .mergeable_latent_capsule_v2 import (
    ConsensusQuorumResultV2,
    MergeableLatentCapsuleV2,
    MergeAuditTrailV2,
    MergeOperatorV2,
    compute_consensus_quorum_v2,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W54_CONSENSUS_CONTROLLER_SCHEMA_VERSION: str = (
    "coordpy.consensus_quorum_controller.v1")

W54_CONSENSUS_DECISION_QUORUM: str = "quorum_merged"
W54_CONSENSUS_DECISION_FALLBACK: str = "fallback_best_parent"
W54_CONSENSUS_DECISION_ABSTAIN: str = "abstain"

W54_DEFAULT_CONSENSUS_K_MIN: int = 2
W54_DEFAULT_CONSENSUS_K_MAX: int = 8
W54_DEFAULT_CONSENSUS_COSINE_FLOOR: float = 0.5
W54_DEFAULT_CONSENSUS_FALLBACK_COSINE_FLOOR: float = 0.3


# =============================================================================
# Helpers
# =============================================================================


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# =============================================================================
# ConsensusPolicy
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ConsensusPolicy:
    """K-of-N policy parameters."""

    k_min: int
    k_max: int
    cosine_floor: float
    fallback_cosine_floor: float
    allow_fallback: bool

    @classmethod
    def default(cls) -> "ConsensusPolicy":
        return cls(
            k_min=W54_DEFAULT_CONSENSUS_K_MIN,
            k_max=W54_DEFAULT_CONSENSUS_K_MAX,
            cosine_floor=W54_DEFAULT_CONSENSUS_COSINE_FLOOR,
            fallback_cosine_floor=(
                W54_DEFAULT_CONSENSUS_FALLBACK_COSINE_FLOOR),
            allow_fallback=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W54_CONSENSUS_CONTROLLER_SCHEMA_VERSION),
            "k_min": int(self.k_min),
            "k_max": int(self.k_max),
            "cosine_floor": float(round(
                self.cosine_floor, 12)),
            "fallback_cosine_floor": float(round(
                self.fallback_cosine_floor, 12)),
            "allow_fallback": bool(self.allow_fallback),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_consensus_policy",
            "policy": self.to_dict()})


# =============================================================================
# ConsensusAuditEntry + Trail
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ConsensusControllerAuditEntry:
    """A single recorded consensus decision."""

    policy_cid: str
    n_branches: int
    k_chosen: int
    decision: str
    quorum_capsule_cid: str
    fallback_branch_id: str
    parent_cids: tuple[str, ...]
    selected_branch_ids: tuple[str, ...]
    reason: str
    turn_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_cid": str(self.policy_cid),
            "n_branches": int(self.n_branches),
            "k_chosen": int(self.k_chosen),
            "decision": str(self.decision),
            "quorum_capsule_cid": str(
                self.quorum_capsule_cid),
            "fallback_branch_id": str(
                self.fallback_branch_id),
            "parent_cids": list(self.parent_cids),
            "selected_branch_ids": list(
                self.selected_branch_ids),
            "reason": str(self.reason),
            "turn_index": int(self.turn_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_consensus_audit_entry",
            "entry": self.to_dict()})


@dataclasses.dataclass
class ConsensusControllerAuditTrail:
    entries: list[ConsensusControllerAuditEntry]

    @classmethod
    def empty(cls) -> "ConsensusControllerAuditTrail":
        return cls(entries=[])

    def add(
            self, entry: ConsensusControllerAuditEntry,
    ) -> None:
        self.entries.append(entry)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_consensus_audit_trail",
            "entries": [e.to_dict() for e in self.entries],
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W54_CONSENSUS_CONTROLLER_SCHEMA_VERSION),
            "n_entries": int(len(self.entries)),
            "audit_cid": str(self.cid()),
        }


# =============================================================================
# Controller
# =============================================================================


@dataclasses.dataclass
class ConsensusQuorumController:
    """K-of-N consensus + fallback controller."""

    policy: ConsensusPolicy
    operator: MergeOperatorV2
    capsule_audit: MergeAuditTrailV2
    controller_audit: ConsensusControllerAuditTrail

    @classmethod
    def init(
            cls, *,
            policy: ConsensusPolicy | None = None,
            operator: MergeOperatorV2,
            capsule_audit: MergeAuditTrailV2 | None = None,
            controller_audit: (
                ConsensusControllerAuditTrail | None) = None,
    ) -> "ConsensusQuorumController":
        return cls(
            policy=(
                policy if policy is not None
                else ConsensusPolicy.default()),
            operator=operator,
            capsule_audit=(
                capsule_audit if capsule_audit is not None
                else MergeAuditTrailV2.empty()),
            controller_audit=(
                controller_audit
                if controller_audit is not None
                else ConsensusControllerAuditTrail.empty()),
        )

    def decide(
            self,
            branches: Sequence[MergeableLatentCapsuleV2],
            *,
            turn_index: int = 0,
            k_required: int | None = None,
    ) -> tuple[
            ConsensusQuorumResultV2,
            ConsensusControllerAuditEntry]:
        n = len(branches)
        k = (
            int(k_required)
            if k_required is not None
            else int(max(
                self.policy.k_min,
                min(n, self.policy.k_max))))
        result = compute_consensus_quorum_v2(
            branches,
            operator=self.operator,
            audit_trail=self.capsule_audit,
            k_required=int(k),
            cosine_floor=float(self.policy.cosine_floor),
            allow_fallback=bool(self.policy.allow_fallback),
            fallback_cosine_floor=float(
                self.policy.fallback_cosine_floor),
        )
        if result.quorum_reached:
            decision = W54_CONSENSUS_DECISION_QUORUM
            reason = (
                f"k_clique_found:size={result.n_branches}"
                f",floor={result.consensus_cosine_floor:.4g}")
        elif result.fallback_used:
            decision = W54_CONSENSUS_DECISION_FALLBACK
            reason = (
                "fallback_to_best_parent:trust*conf"
                f",cosine_to_mean={result.consensus_cosine_floor:.4g}")
        else:
            decision = W54_CONSENSUS_DECISION_ABSTAIN
            reason = "no_quorum_no_fallback"
        entry = ConsensusControllerAuditEntry(
            policy_cid=str(self.policy.cid()),
            n_branches=int(n),
            k_chosen=int(k),
            decision=str(decision),
            quorum_capsule_cid=str(
                result.consensus_capsule_cid),
            fallback_branch_id=str(
                result.fallback_branch_id),
            parent_cids=tuple(
                str(b.cid()) for b in branches),
            selected_branch_ids=tuple(
                result.selected_branch_ids),
            reason=str(reason),
            turn_index=int(turn_index),
        )
        self.controller_audit.add(entry)
        return result, entry

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_consensus_controller",
            "policy_cid": str(self.policy.cid()),
            "operator_cid": str(self.operator.cid()),
            "capsule_audit_cid": str(
                self.capsule_audit.cid()),
            "controller_audit_cid": str(
                self.controller_audit.cid()),
        })


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ConsensusControllerWitness:
    controller_cid: str
    policy_cid: str
    n_decisions: int
    n_quorum: int
    n_fallback: int
    n_abstain: int
    quorum_rate: float
    fallback_rate: float
    abstain_rate: float
    capsule_audit_cid: str
    controller_audit_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller_cid": str(self.controller_cid),
            "policy_cid": str(self.policy_cid),
            "n_decisions": int(self.n_decisions),
            "n_quorum": int(self.n_quorum),
            "n_fallback": int(self.n_fallback),
            "n_abstain": int(self.n_abstain),
            "quorum_rate": float(round(
                self.quorum_rate, 12)),
            "fallback_rate": float(round(
                self.fallback_rate, 12)),
            "abstain_rate": float(round(
                self.abstain_rate, 12)),
            "capsule_audit_cid": str(
                self.capsule_audit_cid),
            "controller_audit_cid": str(
                self.controller_audit_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_consensus_controller_witness",
            "witness": self.to_dict()})


def emit_consensus_controller_witness(
        controller: ConsensusQuorumController,
) -> ConsensusControllerWitness:
    n = int(len(controller.controller_audit.entries))
    n_q = sum(
        1 for e in controller.controller_audit.entries
        if e.decision == W54_CONSENSUS_DECISION_QUORUM)
    n_f = sum(
        1 for e in controller.controller_audit.entries
        if e.decision == W54_CONSENSUS_DECISION_FALLBACK)
    n_a = sum(
        1 for e in controller.controller_audit.entries
        if e.decision == W54_CONSENSUS_DECISION_ABSTAIN)
    return ConsensusControllerWitness(
        controller_cid=str(controller.cid()),
        policy_cid=str(controller.policy.cid()),
        n_decisions=int(n),
        n_quorum=int(n_q),
        n_fallback=int(n_f),
        n_abstain=int(n_a),
        quorum_rate=float(n_q) / float(max(1, n)),
        fallback_rate=float(n_f) / float(max(1, n)),
        abstain_rate=float(n_a) / float(max(1, n)),
        capsule_audit_cid=str(
            controller.capsule_audit.cid()),
        controller_audit_cid=str(
            controller.controller_audit.cid()),
    )


# =============================================================================
# Verifier
# =============================================================================

W54_CONSENSUS_CONTROLLER_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w54_consensus_controller_cid_mismatch",
    "w54_consensus_controller_policy_cid_mismatch",
    "w54_consensus_controller_rates_invalid",
    "w54_consensus_controller_audit_walk_orphan",
    "w54_consensus_controller_quorum_rate_below_floor",
)


def verify_consensus_controller_witness(
        witness: ConsensusControllerWitness,
        *,
        expected_controller_cid: str | None = None,
        expected_policy_cid: str | None = None,
        min_quorum_rate: float | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_controller_cid is not None
            and witness.controller_cid
            != str(expected_controller_cid)):
        failures.append(
            "w54_consensus_controller_cid_mismatch")
    if (expected_policy_cid is not None
            and witness.policy_cid
            != str(expected_policy_cid)):
        failures.append(
            "w54_consensus_controller_policy_cid_mismatch")
    s = (
        float(witness.quorum_rate)
        + float(witness.fallback_rate)
        + float(witness.abstain_rate))
    if abs(s - 1.0) > 1e-6 and s != 0.0:
        failures.append(
            "w54_consensus_controller_rates_invalid")
    if (min_quorum_rate is not None
            and witness.quorum_rate < float(min_quorum_rate)):
        failures.append(
            "w54_consensus_controller_quorum_rate_below_floor")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W54_CONSENSUS_CONTROLLER_SCHEMA_VERSION",
    "W54_CONSENSUS_DECISION_QUORUM",
    "W54_CONSENSUS_DECISION_FALLBACK",
    "W54_CONSENSUS_DECISION_ABSTAIN",
    "W54_DEFAULT_CONSENSUS_K_MIN",
    "W54_DEFAULT_CONSENSUS_K_MAX",
    "W54_DEFAULT_CONSENSUS_COSINE_FLOOR",
    "W54_DEFAULT_CONSENSUS_FALLBACK_COSINE_FLOOR",
    "W54_CONSENSUS_CONTROLLER_VERIFIER_FAILURE_MODES",
    "ConsensusPolicy",
    "ConsensusControllerAuditEntry",
    "ConsensusControllerAuditTrail",
    "ConsensusQuorumController",
    "ConsensusControllerWitness",
    "emit_consensus_controller_witness",
    "verify_consensus_controller_witness",
]
