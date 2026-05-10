"""Tests for the R-92 benchmark family that exercise the W45
learned manifold controller layer.

Each test covers one of H1..H12 of the W45 success criterion in
``docs/SUCCESS_CRITERION_W45_LEARNED_MANIFOLD.md``.
"""

from __future__ import annotations

import pytest

from coordpy.learned_manifold import (
    W45_ALL_FAILURE_MODES,
    W45_LEARNED_MANIFOLD_SCHEMA_VERSION,
    verify_learned_manifold_handoff,
)
from coordpy.r92_benchmark import (
    R92_FAMILY_TABLE,
    family_attention_specialization,
    family_factoradic_hint_compression,
    family_learned_calibration_gain,
    family_model_facing_hint_response,
    family_replay_determinism,
    family_role_adapter_recovery,
    family_trivial_learned_passthrough,
    family_w45_compromise_cap,
    family_w45_falsifier,
    render_text_report,
    run_all_families,
    run_family,
)


SEEDS = (0, 1, 2, 3, 4)


# =============================================================================
# H1 — Trivial learned passthrough is byte-for-W44
# =============================================================================

class TestH1TrivialLearnedPassthrough:
    def test_h1_all_arms_pass(self):
        cmp_ = run_family(
            "r92_trivial_learned_passthrough", seeds=SEEDS)
        for arm in (
                "baseline_team", "w43_closed_form",
                "w44_live_coupled", "w45_learned_coupled"):
            agg = cmp_.get(arm)
            assert agg is not None
            assert agg.min == agg.max == 1.0


# =============================================================================
# H2 — Learned calibration gain strictly improves over W44
# =============================================================================

class TestH2LearnedCalibrationGain:
    def test_h2_learned_beats_w44(self):
        cmp_ = run_family(
            "r92_learned_calibration_gain", seeds=SEEDS)
        learned = cmp_.get("w45_learned_coupled")
        w44 = cmp_.get("w44_live_coupled")
        assert learned is not None and w44 is not None
        # Learned must be at least 0.20 above W44 mean.
        assert learned.mean - w44.mean >= 0.20
        # Learned hits 1.0 on every seed.
        assert learned.min == 1.0


# =============================================================================
# H3 — Attention routing specializes per signature
# =============================================================================

class TestH3AttentionSpecialization:
    def test_h3_attention_specialises(self):
        cmp_ = run_family(
            "r92_attention_specialization", seeds=SEEDS)
        learned = cmp_.get("w45_learned_coupled")
        assert learned is not None
        assert learned.min == 1.0
        assert learned.mean == 1.0


# =============================================================================
# H4 — Role-specific adapter strictly lifts role3 precision
# =============================================================================

class TestH4RoleAdapterRecovery:
    def test_h4_role_adapter_lifts_role3(self):
        cmp_ = run_family(
            "r92_role_adapter_recovery", seeds=SEEDS)
        with_adapter = cmp_.get("w45_learned_coupled")
        shared_only = cmp_.get("w45_shared_only")
        assert with_adapter is not None
        assert shared_only is not None
        # The with-adapter arm must beat the shared-only arm by
        # at least 0.20 on role3.
        assert with_adapter.mean - shared_only.mean >= 0.20
        # And the with-adapter arm must hit 1.0 on every seed.
        assert with_adapter.min == 1.0


# =============================================================================
# H5 — Factoradic + hint compression preserves route + adds
# bounded confidence channel
# =============================================================================

class TestH5FactoradicHintCompression:
    def test_h5_hint_round_trip_ok(self):
        cmp_ = run_family(
            "r92_factoradic_hint_compression", seeds=SEEDS)
        learned = cmp_.get("w45_learned_coupled")
        assert learned is not None
        assert learned.min == 1.0
        assert learned.mean == 1.0


# =============================================================================
# H6 — Model-facing hint response: hint-aware backend lifts the
# task-correct rate
# =============================================================================

class TestH6ModelFacingHintResponse:
    def test_h6_learned_lifts_task_correct(self):
        cmp_ = run_family(
            "r92_model_facing_hint_response", seeds=SEEDS)
        learned = cmp_.get("w45_learned_coupled")
        w44 = cmp_.get("w44_live_coupled")
        assert learned is not None and w44 is not None
        assert learned.mean - w44.mean >= 0.40
        assert learned.min == 1.0


# =============================================================================
# H7 — No false abstention on linear-flow falsifier
# =============================================================================

class TestH7Falsifier:
    def test_h7_no_false_abstain(self):
        cmp_ = run_family("r92_w45_falsifier", seeds=SEEDS)
        learned = cmp_.get("w45_learned_coupled")
        assert learned is not None
        assert learned.min == learned.max == 1.0


# =============================================================================
# H8 — Compromised-observation cap (limitation reproduces)
# =============================================================================

class TestH8CompromiseCap:
    def test_h8_compromise_cap_reproduces(self):
        cmp_ = run_family("r92_w45_compromise_cap", seeds=SEEDS)
        learned = cmp_.get("w45_learned_coupled")
        w44 = cmp_.get("w44_live_coupled")
        assert learned is not None and w44 is not None
        # Both should be 0.0; the limitation is the test passing.
        assert learned.mean == 0.0
        assert w44.mean == 0.0


# =============================================================================
# H9 — Envelope verifier enumerates >= 14 disjoint failure modes
# =============================================================================

class TestH9VerifierCoverage:
    def test_h9_n_failure_modes(self):
        assert len(W45_ALL_FAILURE_MODES) >= 14
        # Disjoint.
        assert (len(set(W45_ALL_FAILURE_MODES))
                == len(W45_ALL_FAILURE_MODES))


# =============================================================================
# H10 — Replay determinism: bit-perfect across two independent runs
# =============================================================================

class TestH10ReplayDeterminism:
    def test_h10_replay_determinism_ok(self):
        cmp_ = run_family(
            "r92_replay_determinism", seeds=SEEDS)
        learned = cmp_.get("w45_learned_coupled")
        assert learned is not None
        assert learned.min == learned.max == 1.0


# =============================================================================
# H11 — Cumulative trust boundary tracks the W45 modes
# =============================================================================

class TestH11CumulativeTrustBoundary:
    def test_h11_w45_modes_disjoint_from_w22_w44(self):
        # W22..W42 cumulative: 196 named modes; W43 adds 18 disjoint
        # modes; W44 adds 12 disjoint modes; W45 adds >= 14.
        # We check the W45 mode names start with "w45_" or are the
        # ``empty_w45_envelope`` sentinel.
        w45_prefix = sum(
            1 for m in W45_ALL_FAILURE_MODES
            if m.startswith("w45_"))
        # plus the empty sentinel.
        empty = sum(
            1 for m in W45_ALL_FAILURE_MODES
            if m == "empty_w45_envelope")
        assert w45_prefix + empty >= 14
        assert empty == 1


# =============================================================================
# H12 — Stable SDK contract preserved
# =============================================================================

class TestH12SdkContractPreserved:
    def test_h12_version_unchanged(self):
        import coordpy
        assert coordpy.__version__ == "0.5.20"
        assert coordpy.SDK_VERSION == "coordpy.sdk.v3.43"

    def test_h12_w45_module_not_in_experimental(self):
        import coordpy
        # The W45 surface must NOT be in __experimental__ at this
        # milestone.
        names = set(coordpy.__experimental__)
        for w45_name in (
                "LearnedManifoldTeam",
                "LearnedManifoldRegistry",
                "build_learned_manifold_registry",
                "fit_learned_controller",
        ):
            assert w45_name not in names

    def test_h12_w45_module_importable_explicit(self):
        # Must be reachable via explicit import only.
        from coordpy.learned_manifold import (
            LearnedManifoldTeam,
            build_learned_manifold_registry,
            fit_learned_controller,
        )
        assert LearnedManifoldTeam is not None
        assert build_learned_manifold_registry is not None
        assert fit_learned_controller is not None


# =============================================================================
# Aggregate sanity
# =============================================================================

class TestRunAllFamilies:
    def test_run_all_families_produces_each_family(self):
        results = run_all_families(seeds=SEEDS)
        # All 9 families present.
        assert len(results) == 9
        for f in (
                "r92_trivial_learned_passthrough",
                "r92_learned_calibration_gain",
                "r92_attention_specialization",
                "r92_role_adapter_recovery",
                "r92_factoradic_hint_compression",
                "r92_model_facing_hint_response",
                "r92_w45_falsifier",
                "r92_w45_compromise_cap",
                "r92_replay_determinism",
        ):
            assert f in results

    def test_render_text_report_includes_all_families(self):
        results = run_all_families(seeds=(0, 1))
        text = render_text_report(results)
        for f in R92_FAMILY_TABLE:
            assert f in text


# =============================================================================
# Per-seed determinism
# =============================================================================

class TestPerSeedDeterminism:
    def test_same_seed_produces_same_result(self):
        # Run each family twice at seed=0 and compare.
        for family, fn in R92_FAMILY_TABLE.items():
            r1 = fn(0)
            r2 = fn(0)
            assert (r1.keys() == r2.keys()), (
                f"family={family}: arm keys diverged")
            for arm in r1:
                assert (r1[arm].metric_value
                        == r2[arm].metric_value), (
                    f"{family}/{arm}: per-seed metric drifted")
