"""Tests for the R-90 benchmark family (W43 product-manifold layer).

Confirms that each family produces the headline numbers documented in
``docs/RESULTS_COORDPY_W43_PRODUCT_MANIFOLD.md`` across the
pre-committed seed set ``(0, 1, 2, 3, 4)``.
"""

from __future__ import annotations

import math

from coordpy.r90_benchmark import (
    R90_FAMILY_TABLE,
    family_causal_violation,
    family_compact_state_transfer,
    family_consensus_cycle,
    family_linear_flow_falsifier,
    family_long_branch,
    family_routing_compression,
    family_subspace_drift,
    family_trivial_pmc,
    render_text_report,
    run_all_families,
    run_family,
)


_SEEDS = (0, 1, 2, 3, 4)


# =============================================================================
# H1..H10 of the W43 success criterion (per the success-criterion doc)
# =============================================================================

class TestR90SuccessBars:
    def test_h1_trivial_passthrough_clean(self):
        cmp_ = run_family("r90_trivial_pmc", seeds=_SEEDS)
        for arm in (
            "baseline_w42_passthrough",
            "baseline_w42_active",
            "w43_pmc_active",
        ):
            agg = cmp_.get(arm)
            assert agg is not None
            assert agg.min == 1.0 and agg.max == 1.0

    def test_h2_branch_round_trip_perfect(self):
        cmp_ = run_family("r90_long_branch", seeds=_SEEDS)
        agg = cmp_.get("w43_pmc_active")
        assert agg is not None
        assert agg.min == 1.0 and agg.max == 1.0

    def test_h3_consensus_cycle_strict_gain(self):
        cmp_ = run_family("r90_consensus_cycle", seeds=_SEEDS)
        pmc = cmp_.get("w43_pmc_active")
        w42 = cmp_.get("baseline_w42_active")
        assert pmc is not None and w42 is not None
        # PMC must be perfect; W42 must be strictly worse.
        assert pmc.min == 1.0 and pmc.max == 1.0
        assert w42.mean < 1.0
        assert pmc.mean - w42.mean >= 0.25

    def test_h4_compact_state_transfer_strict_bit_gain(self):
        cmp_ = run_family(
            "r90_compact_state_transfer", seeds=_SEEDS)
        pmc = cmp_.get("w43_pmc_active")
        w42 = cmp_.get("baseline_w42_active")
        assert pmc is not None and w42 is not None
        # PMC must add strictly more structured bits than W42 at
        # the same overhead.
        assert pmc.min - w42.mean >= 256

    def test_h5_causal_rejection_perfect(self):
        cmp_ = run_family("r90_causal_violation", seeds=_SEEDS)
        pmc = cmp_.get("w43_pmc_active")
        w42 = cmp_.get("baseline_w42_active")
        assert pmc is not None and w42 is not None
        # PMC must reject 100% of out-of-order cells; W42 cannot
        # detect them.
        assert pmc.min == 1.0 and pmc.max == 1.0
        assert w42.max == 0.0

    def test_h6_factoradic_bit_gain_at_zero_visible_cost(self):
        cmp_ = run_family(
            "r90_routing_compression", seeds=_SEEDS)
        pmc = cmp_.get("w43_pmc_active")
        assert pmc is not None
        # ceil(log2(8!)) = 16 bits per cell.
        assert pmc.min == 16.0 and pmc.max == 16.0

    def test_h7_no_false_abstain_on_linear_flow(self):
        cmp_ = run_family(
            "r90_linear_flow_falsifier", seeds=_SEEDS)
        pmc = cmp_.get("w43_pmc_active")
        assert pmc is not None
        # On the falsifier regime, PMC must NOT over-claim.
        assert pmc.min == 1.0 and pmc.max == 1.0

    def test_h8_subspace_drift_strict_gain(self):
        cmp_ = run_family("r90_subspace_drift", seeds=_SEEDS)
        pmc = cmp_.get("w43_pmc_active")
        w42 = cmp_.get("baseline_w42_active")
        assert pmc is not None and w42 is not None
        assert pmc.min == 1.0 and pmc.max == 1.0
        assert pmc.mean - w42.mean >= 0.25


# =============================================================================
# Render-and-aggregate sanity
# =============================================================================

class TestR90Aggregator:
    def test_run_all_families_returns_full_table(self):
        all_results = run_all_families(seeds=_SEEDS)
        assert set(all_results.keys()) == set(R90_FAMILY_TABLE)

    def test_render_text_report_smoke(self):
        all_results = run_all_families(seeds=_SEEDS)
        text = render_text_report(all_results)
        assert "R-90" in text
        for family in R90_FAMILY_TABLE:
            assert family in text

    def test_seed_aggregate_min_le_mean_le_max(self):
        cmp_ = run_family("r90_consensus_cycle", seeds=_SEEDS)
        for agg in cmp_.aggregates:
            assert agg.min <= agg.mean <= agg.max


# =============================================================================
# Channel-specific seeded round-trips
# =============================================================================

class TestSeededFamilyConsistency:
    def test_factoradic_channel_capacity_grows_with_n(self):
        # ceil(log2(n!)) is monotone in n.
        from coordpy.product_manifold import encode_factoradic_route
        prev = 0
        for n in range(2, 13):
            fac = encode_factoradic_route(tuple(range(n)))
            cur = fac.n_structured_bits()
            assert cur >= prev
            prev = cur

    def test_routing_compression_scales_with_n(self):
        # When we crank n_roles up, the factoradic gain grows.
        from coordpy.r90_benchmark import family_routing_compression
        small = family_routing_compression(0, n_roles=4)
        large = family_routing_compression(0, n_roles=12)
        sm = small["w43_pmc_active"].n_factoradic_bits
        la = large["w43_pmc_active"].n_factoradic_bits
        assert la > sm
