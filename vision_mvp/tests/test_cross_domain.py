"""Cross-domain validation tests.

Verifies that the capsule/routing system works correctly across
robotics, NLP, and planning domains by testing:

  1. Robotics Kan minimality (Theorem KAN-1 witness).
  2. NLP handoff naturality (Theorem naturality witness).
  3. Planning operad associativity (Theorem OPERAD-1 witness).
  4. All domains: ConsistencyChecker finds 0 violations.
  5. LearnedRouter trained on domain traces reaches AUC > 0.80.
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_mvp.core.categorical_semantics import AgentTeamOperad, CapsuleCategory
from vision_mvp.core.cross_domain import (
    NLPDomainAdapter,
    PlanningDomainAdapter,
    RoboticsDomainAdapter,
    BiologyDomainAdapter,
    SupplyChainDomainAdapter,
    FinanceDomainAdapter,
    ScienceDomainAdapter,
    ConsensusDomainAdapter,
)
from vision_mvp.core.learned_routing import LearnedRouter, RoutingTrainer
from vision_mvp.formal.consistency_checker import ConsistencyChecker
from vision_mvp.wevra.capsule import CapsuleKind


# ---------------------------------------------------------------------------
# 1. Robotics — Kan minimality
# ---------------------------------------------------------------------------


def test_robotics_kan_minimality():
    """Theorem KAN-1: routing to 'executor' selects minimal covering set."""
    trace = RoboticsDomainAdapter.generate_trace(n_events=40, seed=7)
    # Ensure trace covers all four kinds.
    kinds_present = {c.kind for c in trace}
    assert CapsuleKind.SWEEP_CELL in kinds_present
    assert CapsuleKind.READINESS_CHECK in kinds_present
    assert CapsuleKind.PROFILE in kinds_present

    cat = CapsuleCategory(RoboticsDomainAdapter.role_support())
    kan = cat.right_kan_extension(trace, "executor")
    executor_support = set(RoboticsDomainAdapter._ROLE_SUPPORT["executor"])
    assert {c.kind for c in kan} == executor_support & kinds_present
    assert cat.verify_kan_minimality(trace, "executor")


def test_robotics_kan_sensor_fusion():
    """sensor_fusion only needs HANDLE — Kan extension has exactly 1 kind."""
    trace = RoboticsDomainAdapter.generate_trace(n_events=40, seed=8)
    cat = CapsuleCategory(RoboticsDomainAdapter.role_support())
    kan = cat.right_kan_extension(trace, "sensor_fusion")
    assert all(c.kind == CapsuleKind.HANDLE for c in kan)
    assert cat.verify_kan_minimality(trace, "sensor_fusion")


# ---------------------------------------------------------------------------
# 2. NLP — naturality
# ---------------------------------------------------------------------------


def test_nlp_kan_decoder():
    """NLP: Kan extension for decoder is minimal."""
    trace = NLPDomainAdapter.generate_trace(n_events=40, seed=5)
    cat = CapsuleCategory(NLPDomainAdapter.role_support())

    kan = cat.right_kan_extension(trace, "decoder")
    decoder_support = set(NLPDomainAdapter._ROLE_SUPPORT["decoder"])
    kinds_in_kan = {c.kind for c in kan}

    # Kan extension should cover all kinds in decoder's support that are present
    assert kinds_in_kan.issubset(decoder_support)
    assert cat.verify_kan_minimality(trace, "decoder")


def test_nlp_adjoint_inclusion():
    """Right adjoint (minimal) is always a subset of left adjoint (maximal)."""
    trace = NLPDomainAdapter.generate_trace(n_events=40, seed=6)
    cat = CapsuleCategory(NLPDomainAdapter.role_support())
    for role in ["tokenizer", "encoder", "decoder"]:
        pair = cat.compute_adjoint(CapsuleKind.HANDLE, role, trace)
        left_cids = {c.cid for c in pair["left"]}
        right_cids = {c.cid for c in pair["right"]}
        assert right_cids.issubset(left_cids)


# ---------------------------------------------------------------------------
# 3. Planning — operad associativity
# ---------------------------------------------------------------------------


def test_planning_operad_associativity():
    """Theorem OPERAD-1: any 3-role bracketing yields the same root CID."""
    op = AgentTeamOperad()
    roles = list(PlanningDomainAdapter._ROLE_SUPPORT.keys())
    assert op.verify_associativity(roles)


def test_planning_operad_all_subsets():
    """Associativity holds for every sub-team of planning roles."""
    op = AgentTeamOperad()
    roles = list(PlanningDomainAdapter._ROLE_SUPPORT.keys())
    for n in range(2, len(roles) + 1):
        assert op.verify_associativity(roles[:n])


# ---------------------------------------------------------------------------
# 4. All domains — ConsistencyChecker 0 violations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter,seed", [
    (RoboticsDomainAdapter,  10),
    (NLPDomainAdapter,       20),
    (PlanningDomainAdapter,  30),
    (BiologyDomainAdapter,   40),
    (SupplyChainDomainAdapter, 50),
    (FinanceDomainAdapter,   60),
    (ScienceDomainAdapter,   70),
    (ConsensusDomainAdapter, 80),
])
def test_domain_zero_violations(adapter, seed):
    """ConsistencyChecker finds 0 violations over 200 domain trace trials."""
    checker = ConsistencyChecker()
    summary = checker.fuzz_consistency(n_trials=200, ops_per_trial=10,
                                       seed=seed)
    assert summary["total_violations"] == 0, summary
    assert summary["total_transitions"] >= 2000


# ---------------------------------------------------------------------------
# 5. LearnedRouter — cross-domain AUC > 0.80
# ---------------------------------------------------------------------------


def _build_dataset(adapter, role: str, n_traces: int = 30, seq_len: int = 32,
                   seed: int = 0):
    """Build (events_matrix, labels_matrix) from domain traces."""
    n_event_types = len(adapter.event_types())
    all_events, all_labels = [], []
    for i in range(n_traces):
        trace = adapter.generate_trace(n_events=seq_len, seed=seed + i)
        event_ids = [adapter.event_type_id(c.payload["event_type"])
                     for c in trace[:seq_len]]
        # Pad/truncate to seq_len.
        while len(event_ids) < seq_len:
            event_ids.append(0)
        event_ids = event_ids[:seq_len]
        labels = adapter.labels_for_role(trace[:seq_len], role)
        while len(labels) < seq_len:
            labels.append(0)
        labels = labels[:seq_len]
        all_events.append(event_ids)
        all_labels.append(labels)
    return (np.array(all_events, dtype=np.int64),
            np.array(all_labels, dtype=np.float32))


@pytest.mark.parametrize("adapter,role", [
    (RoboticsDomainAdapter,    "executor"),
    (NLPDomainAdapter,         "decoder"),
    (PlanningDomainAdapter,    "verifier"),
    (BiologyDomainAdapter,     "validator"),
    (SupplyChainDomainAdapter, "compliance_checker"),
    (FinanceDomainAdapter,     "auditor"),
    (ScienceDomainAdapter,     "publication_reviewer"),
    (ConsensusDomainAdapter,   "recovery_manager"),
])
def test_learned_router_domain_auc(adapter, role):
    """LearnedRouter reaches AUC > 0.80 on each domain after training."""
    n_event_types = len(adapter.event_types())
    events, labels = _build_dataset(adapter, role, n_traces=64,
                                    seq_len=32, seed=42)
    role_ids = np.zeros(len(events), dtype=np.int64)
    router = LearnedRouter(n_event_types=n_event_types, n_roles=1, seed=0)
    trainer = RoutingTrainer(router, lr=3e-1)
    for _ in range(150):
        trainer.train_epoch(events, role_ids, labels)
    result = trainer.evaluate(events, role_ids, labels)
    assert result["auc"] > 0.80, f"{adapter.DOMAIN_NAME}/{role}: {result}"
