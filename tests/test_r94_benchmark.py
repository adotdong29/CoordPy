"""Tests for the R-94 benchmark family that exercise the W47
autograd manifold stack layer.

Each test covers one of H1..H12 of the W47 success criterion in
``docs/SUCCESS_CRITERION_W47_AUTOGRAD_MANIFOLD.md``.

The pure-Python autograd engine is correct but slow; we use a
small seed set (3 seeds) to keep total wall-clock under a few
minutes.
"""

from __future__ import annotations

import pytest

from coordpy.autograd_manifold import (
    W47_ALL_FAILURE_MODES,
    W47_AUTOGRAD_MANIFOLD_SCHEMA_VERSION,
    verify_autograd_manifold_handoff,
)
from coordpy.r94_benchmark import (
    R94_FAMILY_TABLE,
    family_autograd_compromise_cap,
    family_autograd_convergence,
    family_autograd_ctrl_aware_backend,
    family_autograd_envelope_verifier,
    family_autograd_gradient_check,
    family_nonlinear_separability,
    family_replay_determinism,
    family_trainable_dictionary,
    family_trainable_memory_head,
    family_trainable_packed_control,
    family_trainable_role_adapter,
    family_trivial_autograd_passthrough,
    render_text_report,
    run_all_families,
    run_family,
)


SEEDS = (0, 1, 2)


# =============================================================================
# H1 — Trivial autograd passthrough is byte-for-W46
# =============================================================================

class TestH1TrivialAutogradPassthrough:
    def test_h1_all_arms_pass(self):
        cmp_ = run_family(
            "r94_trivial_autograd_passthrough", seeds=SEEDS)
        for arm in (
                "baseline_team", "w43_closed_form",
                "w44_live_coupled", "w45_learned_coupled",
                "w46_memory_coupled", "w47_autograd"):
            agg = cmp_.get(arm)
            assert agg is not None
            assert agg.min == agg.max == 1.0


# =============================================================================
# H2 — Autograd engine matches finite-difference gradients
# =============================================================================

class TestH2AutogradGradientCheck:
    def test_h2_all_supported_ops_pass(self):
        cmp_ = run_family(
            "r94_autograd_gradient_check", seeds=SEEDS)
        agg = cmp_.get("w47_autograd")
        assert agg is not None
        assert agg.min == agg.max == 1.0


# =============================================================================
# H3 — Autograd stack converges on the linear regime
# =============================================================================

class TestH3AutogradConvergence:
    def test_h3_converges_on_linear_data(self):
        cmp_ = run_family(
            "r94_autograd_convergence", seeds=SEEDS)
        agg = cmp_.get("w47_autograd")
        assert agg is not None
        assert agg.min == 1.0


# =============================================================================
# H4 — Deep stack is trainable on a nonlinear regime
# =============================================================================

class TestH4NonlinearSeparability:
    def test_h4_deep_stack_trainable(self):
        cmp_ = run_family(
            "r94_nonlinear_separability", seeds=SEEDS)
        agg = cmp_.get("w47_autograd")
        assert agg is not None
        assert agg.min == 1.0


# =============================================================================
# H5 — Trainable dictionary is end-to-end-trainable
# =============================================================================

class TestH5TrainableDictionary:
    def test_h5_dict_trains(self):
        cmp_ = run_family(
            "r94_trainable_dictionary", seeds=SEEDS)
        agg = cmp_.get("w47_autograd")
        assert agg is not None
        assert agg.min == 1.0


# =============================================================================
# H6 — Trainable memory head beats cosine baseline
# =============================================================================

class TestH6TrainableMemoryHead:
    def test_h6_trained_head_beats_cosine(self):
        cmp_ = run_family(
            "r94_trainable_memory_head", seeds=SEEDS)
        agg = cmp_.get("w47_autograd")
        assert agg is not None
        assert agg.min == 1.0
        baseline = cmp_.get("w46_memory_coupled")
        assert baseline is not None
        assert baseline.max == 0.0


# =============================================================================
# H7 — Trainable packed control serializer is bijective + bounded
# =============================================================================

class TestH7TrainablePackedControl:
    def test_h7_round_trip_ok(self):
        cmp_ = run_family(
            "r94_trainable_packed_control", seeds=SEEDS)
        agg = cmp_.get("w47_autograd")
        assert agg is not None
        assert agg.min == agg.max == 1.0


# =============================================================================
# H8 — Trainable rank-r role adapter recovers role-shift
# =============================================================================

class TestH8TrainableRoleAdapter:
    def test_h8_rank2_adapter_works(self):
        cmp_ = run_family(
            "r94_trainable_role_adapter", seeds=SEEDS)
        agg = cmp_.get("w47_autograd")
        assert agg is not None
        assert agg.min == 1.0


# =============================================================================
# H9 — Replay determinism with frozen weights
# =============================================================================

class TestH9ReplayDeterminism:
    def test_h9_replay_is_deterministic(self):
        cmp_ = run_family(
            "r94_replay_determinism", seeds=SEEDS)
        agg = cmp_.get("w47_autograd")
        assert agg is not None
        assert agg.min == agg.max == 1.0


# =============================================================================
# H10 — Verifier soundness
# =============================================================================

class TestH10VerifierSoundness:
    def test_h10_verifier_rejects_forgeries(self):
        cmp_ = run_family(
            "r94_autograd_envelope_verifier", seeds=SEEDS)
        agg = cmp_.get("w47_autograd")
        assert agg is not None
        assert agg.min == agg.max == 1.0


# =============================================================================
# H11 — Compromise cap (limitation reproduces)
# =============================================================================

class TestH11AutogradCompromiseCap:
    def test_h11_limitation_reproduces(self):
        cmp_ = run_family(
            "r94_autograd_compromise_cap", seeds=SEEDS)
        agg = cmp_.get("w47_autograd")
        assert agg is not None
        # Honest claim: not majority-protective.
        assert agg.max <= 0.5
        # The W46 baseline is at zero protection.
        baseline = cmp_.get("w46_memory_coupled")
        assert baseline.mean == 0.0


# =============================================================================
# H12 — CTRL-aware backend behavioural lift
# =============================================================================

class TestH12CtrlAwareBackend:
    def test_h12_ctrl_full_lifts_task_correct_rate(self):
        cmp_ = run_family(
            "r94_autograd_ctrl_aware_backend", seeds=SEEDS)
        full = cmp_.get("w47_autograd")
        off = cmp_.get("w47_ctrl_off")
        base = cmp_.get("baseline_team")
        assert full is not None
        assert off is not None
        assert base is not None
        # Full ctrl must strictly lift over both off and baseline.
        assert full.mean > off.mean + 1e-9
        assert full.mean > base.mean + 1e-9
        # Reality should be 1.0 mean.
        assert full.mean >= 0.9


# =============================================================================
# Aggregate / regression
# =============================================================================

class TestAggregate:
    def test_all_families_registered(self):
        # Basic sanity that R94_FAMILY_TABLE is fully wired.
        assert len(R94_FAMILY_TABLE) >= 12
        for fam in R94_FAMILY_TABLE:
            assert fam.startswith("r94_")

    def test_text_report_renders(self):
        cmp_ = run_family(
            "r94_trivial_autograd_passthrough", seeds=(0,))
        out = render_text_report(
            {"r94_trivial_autograd_passthrough": cmp_})
        assert "R-94" in out
        assert "w47_autograd" in out


# =============================================================================
# Failure modes count
# =============================================================================

class TestFailureModes:
    def test_w47_failure_modes_ge_18(self):
        assert len(W47_ALL_FAILURE_MODES) >= 18

    def test_schema_version_v1(self):
        assert W47_AUTOGRAD_MANIFOLD_SCHEMA_VERSION == (
            "coordpy.autograd_manifold.v1")
