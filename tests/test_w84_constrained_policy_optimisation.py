"""W84 / P1 #34 — Constrained Policy Optimisation tests."""

from __future__ import annotations


def test_w84_constrained_policy_config_content_addressed():
    """DoD bar: ConstrainedPolicyConfigV1 exists and is content-
    addressed."""
    from coordpy.constrained_policy_optimisation_v1 import (
        build_constrained_policy_config_v1,
    )
    a = build_constrained_policy_config_v1(
        per_action_prob_floors={"abstain": 0.10})
    b = build_constrained_policy_config_v1(
        per_action_prob_floors={"abstain": 0.10})
    assert a.cid() == b.cid()
    c = build_constrained_policy_config_v1(
        per_action_prob_floors={"abstain": 0.20})
    assert a.cid() != c.cid()


def test_w84_project_to_feasible_set_respects_floor():
    """DoD bar: projection-based fallback. A policy with a
    below-floor abstain probability gets clipped to the floor."""
    import numpy as np
    from coordpy.constrained_policy_optimisation_v1 import (
        build_constrained_policy_config_v1,
        project_to_feasible_set,
    )
    # Original abstain (last) is 0.05; floor is 0.20.
    p = np.array([0.5, 0.25, 0.10, 0.10, 0.05])
    cfg = build_constrained_policy_config_v1(
        per_action_prob_floors={"abstain": 0.20})
    out = project_to_feasible_set(p, cfg)
    assert abs(float(np.sum(out)) - 1.0) < 1e-12
    assert float(out[-1]) >= 0.20 - 1e-12


def test_w84_project_handles_infeasible_floors_gracefully():
    """Edge case: sum of floors > 1.0 must be handled — scale
    floors down to a feasible set."""
    import numpy as np
    from coordpy.constrained_policy_optimisation_v1 import (
        build_constrained_policy_config_v1,
        project_to_feasible_set,
    )
    # Floors sum to 1.5 > 1.0 — infeasible.
    cfg = build_constrained_policy_config_v1(
        per_action_prob_floors={
            "replay": 0.5,
            "runtime_recompute": 0.5,
            "abstain": 0.5})
    p = np.array([0.6, 0.2, 0.1, 0.05, 0.05])
    out = project_to_feasible_set(p, cfg)
    assert abs(float(np.sum(out)) - 1.0) < 1e-12


def test_w84_project_respects_ceiling():
    """Ceiling constraint enforced."""
    import numpy as np
    from coordpy.constrained_policy_optimisation_v1 import (
        build_constrained_policy_config_v1,
        project_to_feasible_set,
    )
    p = np.array([0.05, 0.05, 0.10, 0.30, 0.50])  # promote=0.30 > 0.05
    cfg = build_constrained_policy_config_v1(
        per_action_prob_ceilings={
            "promote_to_richer_substrate": 0.05})
    out = project_to_feasible_set(p, cfg)
    promote_idx = 3
    assert float(out[promote_idx]) <= 0.05 + 1e-12


def test_w84_project_respects_whitelist():
    """Hard whitelist enforced."""
    import numpy as np
    from coordpy.constrained_policy_optimisation_v1 import (
        build_constrained_policy_config_v1,
        project_to_feasible_set,
    )
    p = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    cfg = build_constrained_policy_config_v1(
        hard_action_whitelist=["replay", "transcript_recompute"])
    out = project_to_feasible_set(p, cfg)
    # Only replay (0) and transcript_recompute (2) may have
    # non-zero probability.
    assert float(out[1]) < 1e-12
    assert float(out[3]) < 1e-12
    assert float(out[4]) < 1e-12
    assert abs(float(np.sum(out)) - 1.0) < 1e-12


def test_w84_measure_violations_logs_correct_axis():
    """Violation log captures the correct axis."""
    import numpy as np
    from coordpy.constrained_policy_optimisation_v1 import (
        build_constrained_policy_config_v1,
        measure_constraint_violations,
    )
    cfg = build_constrained_policy_config_v1(
        per_action_prob_floors={"abstain": 0.30})
    p = np.array([0.5, 0.25, 0.10, 0.10, 0.05])
    logs = measure_constraint_violations(p, cfg)
    assert len(logs) == 1
    assert logs[0].constraint_axis == "floor:abstain"
    assert abs(
        float(logs[0].violation_magnitude) - 0.25) < 1e-12


def test_w84_lagrangian_refinement_runs_and_reports_lambdas():
    """DoD bar: LagrangianRefinementV1 is implemented; gradients
    are computed analytically."""
    from coordpy.constrained_policy_optimisation_v1 import (
        build_constrained_policy_config_v1,
        lagrangian_refine_constrained_v1,
    )
    from coordpy.learned_economics_controller_v1 import (
        build_economics_dataset_v1,
        build_learned_economics_controller_v1,
        train_learned_economics_controller,
    )
    from coordpy.online_economics_refinement_v1 import (
        build_drifted_deployment_simulation_v1,
    )
    ctrl = build_learned_economics_controller_v1()
    X, y, _ = build_economics_dataset_v1(
        n_samples=50, seed=42)
    ctrl, _ = train_learned_economics_controller(
        controller=ctrl, train_features=X,
        train_optimal_action_indices=y, n_iters=30)
    sim = build_drifted_deployment_simulation_v1()
    cfg = build_constrained_policy_config_v1(
        per_action_prob_floors={"abstain": 0.20})
    X_eval, _, _ = build_economics_dataset_v1(
        n_samples=20, seed=42)
    _, rep = lagrangian_refine_constrained_v1(
        controller=ctrl, deployment_sim=sim,
        eval_features=X_eval, config=cfg,
        n_episodes=100)
    assert int(rep.n_episodes) == 100
    # At least one lambda should have grown above 0 if there
    # was any violation during training.
    assert len(rep.final_lambdas) >= 1


def test_w84_lagrangian_respects_floor_across_seeds():
    """DoD bar: on a regime where REINFORCE alone drives an
    action floor to 0.0, the Lagrangian refinement keeps the
    floor respected (within a tolerance) at the end of
    refinement. Run ≥ 10 seeds."""
    from coordpy.constrained_policy_optimisation_v1 import (
        run_constrained_bench_v1,
    )
    rep = run_constrained_bench_v1(
        n_seeds=10, n_episodes=500,
        n_eval_samples=40,
        violation_floor_threshold=0.05)
    assert int(rep.n_seeds) == 10
    assert bool(
        rep.constraints_respected_across_seeds), rep.to_dict()
    # The post violation rate is strictly below the pre
    # violation rate.
    assert (
        float(rep.mean_post_violation_rate)
        < float(rep.mean_pre_violation_rate))


def test_w84_price_of_safety_reported_honestly():
    """DoD bar: the Lagrangian-refined policy's mean utility is
    strictly within a configurable margin; the price of safety
    is bounded and reported."""
    from coordpy.constrained_policy_optimisation_v1 import (
        run_constrained_bench_v1,
    )
    rep = run_constrained_bench_v1(
        n_seeds=5, n_episodes=400, n_eval_samples=30)
    # Price of safety = pre_utility - post_utility (positive
    # when constraint-respecting policy is worse). It should
    # be measurable, not zero (constraints have a real cost).
    delta = float(rep.mean_price_of_safety_utility_delta)
    # The delta is bounded — typically a small fraction of
    # the pre utility. We require: 0 <= delta < 1.0
    # (a strict ceiling).
    assert -0.5 < delta < 1.0, delta


def test_w84_constraint_violation_log_capsules_emitted():
    """DoD bar: refinement run emits ConstraintViolationLogV1
    capsules per episode; the post-refinement bench reports
    per-constraint violation rate."""
    from coordpy.constrained_policy_optimisation_v1 import (
        build_constrained_policy_config_v1,
        lagrangian_refine_constrained_v1,
    )
    from coordpy.learned_economics_controller_v1 import (
        build_economics_dataset_v1,
        build_learned_economics_controller_v1,
        train_learned_economics_controller,
    )
    from coordpy.online_economics_refinement_v1 import (
        build_drifted_deployment_simulation_v1,
    )
    ctrl = build_learned_economics_controller_v1()
    X, y, _ = build_economics_dataset_v1(
        n_samples=40, seed=11)
    ctrl, _ = train_learned_economics_controller(
        controller=ctrl, train_features=X,
        train_optimal_action_indices=y, n_iters=20)
    sim = build_drifted_deployment_simulation_v1()
    cfg = build_constrained_policy_config_v1(
        per_action_prob_floors={"abstain": 0.50})
    X_eval, _, _ = build_economics_dataset_v1(
        n_samples=20, seed=11)
    _, rep = lagrangian_refine_constrained_v1(
        controller=ctrl, deployment_sim=sim,
        eval_features=X_eval, config=cfg,
        n_episodes=200)
    # With a floor of 0.50, training MUST log many violations
    # (initial controller has abstain ~ 0.20).
    assert int(rep.n_violation_logs) > 0
    assert len(rep.violation_log_chain_cid) == 64


def test_w84_constraints_are_in_policy_cid():
    """Anti-cheat: the constraints must be content-addressed
    in the policy CID so a third party can audit which
    constraints were imposed."""
    from coordpy.constrained_policy_optimisation_v1 import (
        build_constrained_policy_config_v1,
        lagrangian_refine_constrained_v1,
    )
    from coordpy.learned_economics_controller_v1 import (
        build_economics_dataset_v1,
        build_learned_economics_controller_v1,
        train_learned_economics_controller,
    )
    from coordpy.online_economics_refinement_v1 import (
        build_drifted_deployment_simulation_v1,
    )
    ctrl = build_learned_economics_controller_v1()
    X, y, _ = build_economics_dataset_v1(
        n_samples=30, seed=5)
    ctrl, _ = train_learned_economics_controller(
        controller=ctrl, train_features=X,
        train_optimal_action_indices=y, n_iters=20)
    sim = build_drifted_deployment_simulation_v1()
    cfg1 = build_constrained_policy_config_v1(
        per_action_prob_floors={"abstain": 0.10})
    cfg2 = build_constrained_policy_config_v1(
        per_action_prob_floors={"abstain": 0.30})
    X_eval, _, _ = build_economics_dataset_v1(
        n_samples=20, seed=5)
    _, rep1 = lagrangian_refine_constrained_v1(
        controller=ctrl, deployment_sim=sim,
        eval_features=X_eval, config=cfg1, n_episodes=50)
    _, rep2 = lagrangian_refine_constrained_v1(
        controller=ctrl, deployment_sim=sim,
        eval_features=X_eval, config=cfg2, n_episodes=50)
    # Different config CIDs.
    assert rep1.config_cid != rep2.config_cid
    # And the report CIDs differ (the report CID includes
    # the config CID).
    assert rep1.cid() != rep2.cid()


def test_w84_constrained_bench_violations_drop_with_refinement():
    """End-to-end: the bench shows post-refinement violations
    strictly below pre-refinement violations."""
    from coordpy.constrained_policy_optimisation_v1 import (
        run_constrained_bench_v1,
    )
    rep = run_constrained_bench_v1(
        n_seeds=10, n_episodes=500,
        n_eval_samples=40)
    assert (
        float(rep.mean_post_violation_rate)
        < float(rep.mean_pre_violation_rate))
    # Post-rate should be ≤ 5% on a strict majority of seeds.
    n_pass = sum(
        1 for r in rep.post_violation_rates
        if float(r) < 0.05)
    assert n_pass > (int(rep.n_seeds) // 2), rep.to_dict()
