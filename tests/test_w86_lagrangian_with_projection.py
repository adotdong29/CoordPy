"""W86 #34 — Lagrangian + projection-fallback floor recovery tests.

The W84 ``LagrangianRefinementV1`` alone achieves 0/10 seeds
floor-respected on the W84 floor-recovery bench (high
Monte-Carlo variance on the reward-shaping dual estimator).
The W86 closure path combines Lagrangian refinement with the
projection-based fallback the issue body explicitly permits;
the combination achieves 10/10 seeds floor-respected.
"""
from __future__ import annotations

import pytest


def test_w86_lagrangian_proj_module_imports():
    from coordpy.lagrangian_with_projection_v1 import (
        LagrangianProjectionBenchPointV1,
        LagrangianProjectionBenchReportV1,
        W86_LAGRANGIAN_PROJ_V1_SCHEMA_VERSION,
        projected_controller_action_probs,
        run_lagrangian_with_projection_floor_recovery_v1,
    )
    assert W86_LAGRANGIAN_PROJ_V1_SCHEMA_VERSION == (
        "coordpy.lagrangian_with_projection_v1.v1")


def test_w86_proj_floor_respected_on_all_seeds():
    """Load-bearing #34 DoD bullet: the Lagrangian (here,
    Lagrangian + projection) refinement keeps the floor
    respected at the END of refinement, across ≥ 10 seeds."""
    from coordpy.lagrangian_with_projection_v1 import (
        run_lagrangian_with_projection_floor_recovery_v1,
    )
    r = run_lagrangian_with_projection_floor_recovery_v1(
        n_seeds=10)
    assert int(r.n_seeds) == 10
    # Every seed's projected policy respects the abstain floor.
    assert int(r.n_seeds_floor_respected_proj) == 10
    # And the projection strictly beats REINFORCE.
    assert bool(
        r.proj_strictly_beats_reinforce_on_floor) is True
    # The bootstrap CI on the projected violation rate must be
    # at zero (the projection guarantees no violation).
    assert float(
        r.bootstrap_proj_floor_violation_ci_hi) <= 0.001


def test_w86_proj_strict_beat_over_lagrangian_only():
    """Anti-cheat tightening: the W86 closure path must
    strictly beat the W84 Lagrangian-only refinement, NOT just
    tie it. Otherwise we'd be claiming a closure without real
    progress."""
    from coordpy.lagrangian_with_projection_v1 import (
        run_lagrangian_with_projection_floor_recovery_v1,
    )
    r = run_lagrangian_with_projection_floor_recovery_v1(
        n_seeds=10)
    assert int(
        r.n_seeds_floor_respected_proj) > int(
        r.n_seeds_floor_respected_lagrangian_only)
    assert bool(
        r.proj_strictly_beats_lagrangian_only_on_floor
    ) is True


def test_w86_proj_price_of_safety_reported():
    """The issue body's anti-cheat clause requires the price-
    of-safety to be reported with bootstrap CIs. Verify."""
    from coordpy.lagrangian_with_projection_v1 import (
        run_lagrangian_with_projection_floor_recovery_v1,
    )
    r = run_lagrangian_with_projection_floor_recovery_v1(
        n_seeds=10)
    d = r.to_dict()
    assert "mean_price_of_safety_proj_vs_lagrangian" in d
    assert "mean_price_of_safety_proj_vs_reinforce" in d
    assert (
        "bootstrap_proj_floor_violation_ci_lo" in d)
    assert (
        "bootstrap_proj_floor_violation_ci_hi" in d)


def test_w86_proj_bench_cid_deterministic():
    """Same config → same report CID (anti-cheat: content-
    addressed audit chain)."""
    from coordpy.lagrangian_with_projection_v1 import (
        run_lagrangian_with_projection_floor_recovery_v1,
    )
    r1 = run_lagrangian_with_projection_floor_recovery_v1(
        n_seeds=3)
    r2 = run_lagrangian_with_projection_floor_recovery_v1(
        n_seeds=3)
    assert r1.cid() == r2.cid()


def test_w86_projected_controller_probs_respect_floor():
    """Direct test of the projection helper: any softmax
    distribution projected onto a floor-constrained set must
    respect the floor by construction."""
    import numpy as np
    from coordpy.constrained_policy_optimisation_v1 import (
        ConstrainedPolicyConfigV1,
        W84_CONSTRAINED_POLICY_V1_SCHEMA_VERSION,
        project_onto_feasible_set_v1,
    )
    from coordpy.learned_economics_controller_v1 import (
        W81_ECONOMICS_ACTIONS,
    )
    config = ConstrainedPolicyConfigV1(
        schema=W84_CONSTRAINED_POLICY_V1_SCHEMA_VERSION,
        action_floors=(("abstain", 0.10),),
        action_ceilings=(),
        action_cost_ceilings=(),
        whitelist_actions=(),
    )
    # Adversarial distribution: abstain probability 0.0.
    probs = np.array([0.5, 0.3, 0.15, 0.05, 0.0],
                     dtype=np.float64)
    proj = project_onto_feasible_set_v1(
        probs, config=config)
    abstain_idx = W81_ECONOMICS_ACTIONS.index("abstain")
    assert proj[abstain_idx] >= 0.10 - 1e-9
    # Distribution is normalised.
    assert abs(float(np.sum(proj)) - 1.0) < 1e-9
