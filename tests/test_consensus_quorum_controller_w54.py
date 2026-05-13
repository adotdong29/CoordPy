"""W54 M4 — Consensus / Quorum Controller tests."""

from __future__ import annotations

import random

from coordpy.consensus_quorum_controller import (
    ConsensusPolicy,
    ConsensusQuorumController,
    W54_CONSENSUS_CONTROLLER_VERIFIER_FAILURE_MODES,
    W54_CONSENSUS_DECISION_ABSTAIN,
    W54_CONSENSUS_DECISION_FALLBACK,
    W54_CONSENSUS_DECISION_QUORUM,
    emit_consensus_controller_witness,
    verify_consensus_controller_witness,
)
from coordpy.mergeable_latent_capsule_v2 import (
    MergeOperatorV2,
    make_root_capsule_v2,
)


def test_controller_decides_quorum_on_consistent_branches() -> None:
    op = MergeOperatorV2(factor_dim=4)
    ctrl = ConsensusQuorumController.init(
        policy=ConsensusPolicy(
            k_min=2, k_max=4, cosine_floor=0.5,
            fallback_cosine_floor=0.0,
            allow_fallback=True),
        operator=op)
    target = [0.8, 0.3, -0.4, 0.2]
    branches = [
        make_root_capsule_v2(
            branch_id=f"b{i}",
            payload=[t + 0.02 * i for t in target],
            confidence=0.8, trust=0.9)
        for i in range(3)
    ]
    res, entry = ctrl.decide(
        branches, turn_index=0, k_required=2)
    assert entry.decision == W54_CONSENSUS_DECISION_QUORUM
    assert res.quorum_reached


def test_controller_decides_fallback_when_quorum_unmet() -> None:
    op = MergeOperatorV2(factor_dim=4)
    ctrl = ConsensusQuorumController.init(
        policy=ConsensusPolicy(
            k_min=5, k_max=5, cosine_floor=0.5,
            fallback_cosine_floor=-1.0,
            allow_fallback=True),
        operator=op)
    rng = random.Random(7)
    branches = [
        make_root_capsule_v2(
            branch_id=f"b{i}",
            payload=[
                rng.uniform(-1, 1) for _ in range(4)],
            confidence=0.7, trust=0.8)
        for i in range(4)
    ]
    res, entry = ctrl.decide(
        branches, turn_index=0, k_required=5)
    assert entry.decision == W54_CONSENSUS_DECISION_FALLBACK
    assert res.fallback_used


def test_controller_audit_records_parent_cids() -> None:
    op = MergeOperatorV2(factor_dim=4)
    ctrl = ConsensusQuorumController.init(
        policy=ConsensusPolicy.default(),
        operator=op)
    rng = random.Random(11)
    branches = [
        make_root_capsule_v2(
            branch_id=f"b{i}",
            payload=[
                rng.uniform(-1, 1) for _ in range(4)],
            confidence=0.8, trust=0.9)
        for i in range(3)
    ]
    _, entry = ctrl.decide(
        branches, turn_index=0, k_required=2)
    parent_cids = set(entry.parent_cids)
    for b in branches:
        assert b.cid() in parent_cids


def test_controller_witness_rates_sum_to_one() -> None:
    op = MergeOperatorV2(factor_dim=4)
    ctrl = ConsensusQuorumController.init(
        policy=ConsensusPolicy.default(),
        operator=op)
    rng = random.Random(23)
    for trial in range(4):
        branches = [
            make_root_capsule_v2(
                branch_id=f"b{trial}_{i}",
                payload=[
                    rng.uniform(-1, 1) for _ in range(4)],
                confidence=0.7, trust=0.8)
            for i in range(3)
        ]
        ctrl.decide(
            branches, turn_index=trial, k_required=2)
    w = emit_consensus_controller_witness(ctrl)
    s = (
        float(w.quorum_rate) + float(w.fallback_rate)
        + float(w.abstain_rate))
    assert abs(s - 1.0) < 1e-6


def test_controller_verifier_failure_modes_count() -> None:
    assert len(
        W54_CONSENSUS_CONTROLLER_VERIFIER_FAILURE_MODES) == 5
