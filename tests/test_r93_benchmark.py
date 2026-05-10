"""Tests for the R-93 benchmark family that exercise the W46
manifold memory controller layer.

Each test covers one of H1..H12 of the W46 success criterion in
``docs/SUCCESS_CRITERION_W46_MANIFOLD_MEMORY.md``.
"""

from __future__ import annotations

import pytest

from coordpy.manifold_memory import (
    W46_ALL_FAILURE_MODES,
    W46_MANIFOLD_MEMORY_SCHEMA_VERSION,
    verify_manifold_memory_handoff,
)
from coordpy.r93_benchmark import (
    R93_FAMILY_TABLE,
    family_causal_mask_preservation,
    family_compressed_control_packing,
    family_cyclic_consensus_memory,
    family_dictionary_reconstruction,
    family_long_branching_memory,
    family_memory_facing_hint_response,
    family_replay_determinism,
    family_role_shift_adaptation,
    family_shared_prefix_reuse,
    family_trivial_memory_passthrough,
    family_w46_compromise_cap,
    family_w46_falsifier,
    render_text_report,
    run_all_families,
    run_family,
)


SEEDS = (0, 1, 2, 3, 4)


# =============================================================================
# H1 — Trivial memory passthrough is byte-for-W45
# =============================================================================

class TestH1TrivialMemoryPassthrough:
    def test_h1_all_arms_pass(self):
        cmp_ = run_family(
            "r93_trivial_memory_passthrough", seeds=SEEDS)
        for arm in (
                "baseline_team", "w43_closed_form",
                "w44_live_coupled", "w45_learned_coupled",
                "w46_memory_coupled"):
            agg = cmp_.get(arm)
            assert agg is not None
            assert agg.min == agg.max == 1.0


# =============================================================================
# H2 — Long-branching-memory: w46 strictly beats w45 on deep turns
# =============================================================================

class TestH2LongBranchingMemory:
    def test_h2_memory_beats_w45(self):
        cmp_ = run_family(
            "r93_long_branching_memory", seeds=SEEDS)
        mem = cmp_.get("w46_memory_coupled")
        w45 = cmp_.get("w45_learned_coupled")
        assert mem is not None and w45 is not None
        # Strict beat of >= 0.20 (the bar is wide; reality is 1.0).
        assert mem.mean - w45.mean >= 0.20
        # Reality: memory ratifies all deep turns.
        assert mem.min >= 0.8


# =============================================================================
# H3 — Cyclic-consensus preservation (no regression vs W45)
# =============================================================================

class TestH3CyclicConsensusPreservation:
    def test_h3_memory_preserves_w45(self):
        cmp_ = run_family(
            "r93_cyclic_consensus_memory", seeds=SEEDS)
        mem = cmp_.get("w46_memory_coupled")
        w45 = cmp_.get("w45_learned_coupled")
        assert mem is not None and w45 is not None
        assert mem.min == mem.max == 1.0
        assert mem.mean >= w45.mean


# =============================================================================
# H4 — Multi-rank role adapter strictly beats shared-only
# =============================================================================

class TestH4RoleShiftAdaptation:
    def test_h4_rank2_beats_shared_only(self):
        cmp_ = run_family(
            "r93_role_shift_adaptation", seeds=SEEDS)
        rank2 = cmp_.get("w46_rank2")
        shared = cmp_.get("w46_shared_only")
        assert rank2 is not None and shared is not None
        assert rank2.mean - shared.mean >= 0.20
        assert rank2.min == 1.0


# =============================================================================
# H5 — Packed control token surface is bijective + bounded
# =============================================================================

class TestH5CompressedControlPacking:
    def test_h5_round_trip_ok_and_bounded(self):
        cmp_ = run_family(
            "r93_compressed_control_packing", seeds=SEEDS)
        mem = cmp_.get("w46_memory_coupled")
        assert mem is not None
        assert mem.min == mem.max == 1.0


# =============================================================================
# H6 — Memory-facing hint response: w46 lifts task-correct rate
# =============================================================================

class TestH6MemoryFacingHintResponse:
    def test_h6_lifts_task_correct_rate(self):
        cmp_ = run_family(
            "r93_memory_facing_hint_response", seeds=SEEDS)
        mem = cmp_.get("w46_memory_coupled")
        w45 = cmp_.get("w45_learned_coupled")
        assert mem is not None and w45 is not None
        assert mem.mean - w45.mean >= 0.40
        assert mem.min == 1.0


# =============================================================================
# H7 — Causal mask preservation
# =============================================================================

class TestH7CausalMaskPreservation:
    def test_h7_future_inject_zero_delta(self):
        cmp_ = run_family(
            "r93_causal_mask_preservation", seeds=SEEDS)
        mem = cmp_.get("w46_memory_coupled")
        assert mem is not None
        assert mem.min == mem.max == 1.0


# =============================================================================
# H8 — Dictionary basis bijective round-trip
# =============================================================================

class TestH8DictionaryReconstruction:
    def test_h8_dictionary_round_trip_ok(self):
        cmp_ = run_family(
            "r93_dictionary_reconstruction", seeds=SEEDS)
        mem = cmp_.get("w46_memory_coupled")
        assert mem is not None
        assert mem.min == mem.max == 1.0


# =============================================================================
# H9 — Shared-prefix capsule reuses bytes across turns
# =============================================================================

class TestH9SharedPrefixReuse:
    def test_h9_prefix_reuse_ok(self):
        cmp_ = run_family(
            "r93_shared_prefix_reuse", seeds=SEEDS)
        mem = cmp_.get("w46_memory_coupled")
        assert mem is not None
        assert mem.min == mem.max == 1.0


# =============================================================================
# H10 — Envelope verifier enumerates >= 16 disjoint failure modes
# =============================================================================

class TestH10VerifierCoverage:
    def test_h10_n_failure_modes(self):
        assert len(W46_ALL_FAILURE_MODES) >= 16
        # Disjoint.
        assert (len(set(W46_ALL_FAILURE_MODES))
                == len(W46_ALL_FAILURE_MODES))

    def test_h10_disjoint_from_w22_w45(self):
        # All W46 modes must start with "w46_" or be the empty
        # sentinel.
        w46_prefix = sum(
            1 for m in W46_ALL_FAILURE_MODES
            if m.startswith("w46_"))
        empty = sum(
            1 for m in W46_ALL_FAILURE_MODES
            if m == "empty_w46_envelope")
        assert w46_prefix + empty == len(W46_ALL_FAILURE_MODES)
        assert empty == 1


# =============================================================================
# H11 — Replay determinism: bit-perfect across two runs
# =============================================================================

class TestH11ReplayDeterminism:
    def test_h11_replay_determinism_ok(self):
        cmp_ = run_family(
            "r93_replay_determinism", seeds=SEEDS)
        mem = cmp_.get("w46_memory_coupled")
        assert mem is not None
        assert mem.min == mem.max == 1.0


# =============================================================================
# H12 — Stable SDK contract preserved
# =============================================================================

class TestH12SdkContractPreserved:
    def test_h12_version_unchanged(self):
        import coordpy
        assert coordpy.__version__ == "0.5.20"
        assert coordpy.SDK_VERSION == "coordpy.sdk.v3.43"

    def test_h12_w46_module_not_in_experimental(self):
        import coordpy
        names = set(coordpy.__experimental__)
        for w46_name in (
                "ManifoldMemoryTeam",
                "ManifoldMemoryRegistry",
                "build_manifold_memory_registry",
                "fit_memory_controller",
        ):
            assert w46_name not in names

    def test_h12_w46_module_importable_explicit(self):
        from coordpy.manifold_memory import (
            ManifoldMemoryTeam,
            build_manifold_memory_registry,
            fit_memory_controller,
        )
        assert ManifoldMemoryTeam is not None
        assert build_manifold_memory_registry is not None
        assert fit_memory_controller is not None


# =============================================================================
# Falsifier + compromise-cap families (sanity + limitation)
# =============================================================================

class TestFalsifierAndCompromiseCap:
    def test_w46_falsifier_no_false_abstain(self):
        cmp_ = run_family("r93_w46_falsifier", seeds=SEEDS)
        mem = cmp_.get("w46_memory_coupled")
        assert mem is not None
        assert mem.min == mem.max == 1.0

    def test_w46_compromise_cap_reproduces(self):
        cmp_ = run_family("r93_w46_compromise_cap", seeds=SEEDS)
        mem = cmp_.get("w46_memory_coupled")
        w45 = cmp_.get("w45_learned_coupled")
        assert mem is not None and w45 is not None
        # Both should be 0 — the limitation reproduces honestly.
        assert mem.mean == 0.0
        assert w45.mean == 0.0


# =============================================================================
# Aggregate sanity
# =============================================================================

class TestRunAllFamilies:
    def test_run_all_families_produces_each_family(self):
        results = run_all_families(seeds=(0, 1))
        assert len(results) == 12
        for f in R93_FAMILY_TABLE:
            assert f in results

    def test_render_text_report_includes_all_families(self):
        results = run_all_families(seeds=(0, 1))
        text = render_text_report(results)
        for f in R93_FAMILY_TABLE:
            assert f in text


# =============================================================================
# Per-seed determinism: same seed -> same result
# =============================================================================

class TestPerSeedDeterminism:
    def test_same_seed_produces_same_result(self):
        for family, fn in R93_FAMILY_TABLE.items():
            r1 = fn(0)
            r2 = fn(0)
            assert r1.keys() == r2.keys(), (
                f"family={family}: arm keys diverged")
            for arm in r1:
                assert (r1[arm].metric_value
                        == r2[arm].metric_value), (
                    f"{family}/{arm}: per-seed metric drifted")
