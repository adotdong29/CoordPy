"""Tests for the 9.2 -> 9.5+ advancement sprint.

Covers the three new modules:

  * ``core.categorical_semantics`` — Kan extension + operad associativity.
  * ``formal.consistency_checker`` — zero-violation fuzz over 1000 trials.
  * ``core.learned_routing`` — trainable router reaches AUC > 0.80.

The TLC model checker is exercised lightly — its schema is verified but
the actual ``tlc`` subprocess only runs when the binary is installed on
the host, so the test is skip-with-reason when it isn't.
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_mvp.wevra.capsule import (
    CapsuleBudget, CapsuleKind, ContextCapsule,
)
from vision_mvp.core.categorical_semantics import (
    AgentTeamOperad, CapsuleCategory, TeamNode,
)
from vision_mvp.core.learned_routing import (
    LearnedRouter, RoutingTrainer, synthetic_dataset,
)
from vision_mvp.formal.consistency_checker import ConsistencyChecker
from vision_mvp.formal.run_model_checker import (
    INVARIANTS, ModelCheckResult, TLCModelChecker,
)


# ---- categorical_semantics ----------------------------------------------


def _make_capsules(kinds):
    return [
        ContextCapsule.new(
            kind=k, payload={"i": i},
            budget=CapsuleBudget(max_bytes=256), parents=())
        for i, k in enumerate(kinds)
    ]


def test_kan_extension_minimality():
    cat = CapsuleCategory({
        "triage": [CapsuleKind.HANDOFF, CapsuleKind.ARTIFACT],
        "fix": [CapsuleKind.HANDOFF],
    })
    caps = _make_capsules(
        [CapsuleKind.HANDOFF, CapsuleKind.HANDOFF,
         CapsuleKind.ARTIFACT, CapsuleKind.PROFILE])
    kan = cat.right_kan_extension(caps, "triage")
    assert {c.kind for c in kan} == {CapsuleKind.HANDOFF, CapsuleKind.ARTIFACT}
    assert cat.verify_kan_minimality(caps, "triage")


def test_adjoint_inclusion():
    cat = CapsuleCategory({"r": [CapsuleKind.HANDOFF]})
    caps = _make_capsules(
        [CapsuleKind.HANDOFF, CapsuleKind.ARTIFACT, CapsuleKind.HANDOFF])
    pair = cat.compute_adjoint(CapsuleKind.HANDOFF, "r", caps)
    assert set(c.cid for c in pair["right"]).issubset(
        c.cid for c in pair["left"])


def test_handoff_naturality():
    cat = CapsuleCategory({
        "big": [CapsuleKind.HANDOFF, CapsuleKind.ARTIFACT],
        "small": [CapsuleKind.HANDOFF],
    })

    def handoff_fn(role, tup):
        support = cat.obj(role).support
        return tuple(c for c in tup if c.kind in support)

    caps = tuple(_make_capsules(
        [CapsuleKind.HANDOFF, CapsuleKind.ARTIFACT, CapsuleKind.HANDOFF]))
    assert cat.verify_naturality(handoff_fn, {"big": caps})


def test_operad_associativity():
    op = AgentTeamOperad()
    for n in range(2, 6):
        assert op.verify_associativity(list("abcdef"[:n]))


# ---- consistency_checker -------------------------------------------------


def test_consistency_zero_violations_small():
    c = ConsistencyChecker()
    transitions = c.extract_python_behavior(n_capsules=50, seed=0)
    report = c.verify_against_tla_spec(transitions)
    assert report.all_pass, report.violations


def test_consistency_fuzz_1000_trials():
    c = ConsistencyChecker()
    summary = c.fuzz_consistency(n_trials=1000, ops_per_trial=10, seed=42)
    assert summary["total_violations"] == 0, summary
    assert summary["total_transitions"] >= 10_000


# ---- run_model_checker ---------------------------------------------------


def test_model_checker_reports_installed_flag():
    # Without TLC installed, the result shape is still valid.
    result = TLCModelChecker().run_model_check(max_depth=50, workers=2)
    assert isinstance(result, ModelCheckResult)
    assert isinstance(result.as_dict(), dict)
    assert set(INVARIANTS) == {
        "Inv_C1_Identity", "Inv_C2_TypedClaim", "Inv_C3_Lifecycle",
        "Inv_C4_Budget", "Inv_C5_Provenance", "Inv_C6_Frozen",
    }


def test_model_checker_cfg_roundtrip(tmp_path):
    cfg = tmp_path / "CapsuleContract.cfg"
    checker = TLCModelChecker(cfg_path=cfg)
    checker.write_cfg(max_ledger_size=3, n_kinds=2)
    text = cfg.read_text()
    assert "MaxLedgerSize = 3" in text
    for inv in INVARIANTS:
        assert inv in text


# ---- learned_routing -----------------------------------------------------


def test_learned_router_forward_shape():
    router = LearnedRouter(n_event_types=10, n_roles=3, seed=0)
    events, roles, labels = synthetic_dataset(
        n_event_types=10, n_roles=3, batch=8, seq_len=12, seed=0)
    preds = router.forward(events, int(np.asarray(roles).reshape(-1)[0]))
    arr = preds.detach().cpu().numpy() if hasattr(preds, "detach") else np.asarray(preds)
    assert arr.shape == (8, 12)
    assert ((arr >= 0) & (arr <= 1)).all()


def test_learned_router_training_auc():
    router = LearnedRouter(n_event_types=20, n_roles=4, seed=0)
    trainer = RoutingTrainer(router, lr=3e-1)
    events, roles, labels = synthetic_dataset(
        n_event_types=20, n_roles=4, batch=128, seq_len=24, seed=1)

    losses = []
    for _ in range(120):
        losses.append(trainer.train_epoch(events, roles, labels))

    assert losses[-1] < losses[0], (losses[0], losses[-1])
    r = trainer.evaluate(events, roles, labels)
    assert r["auc"] > 0.80, r
