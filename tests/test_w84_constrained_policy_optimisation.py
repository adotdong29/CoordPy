"""W84 / P1 #34 — Constrained policy optimisation V1 tests."""

from __future__ import annotations

import numpy as np
import pytest

from coordpy.constrained_policy_optimisation_v1 import (
    ConstrainedPolicyConfigV1,
    LagrangianRefinementV1,
    W84_CONSTRAINED_POLICY_V1_SCHEMA_VERSION,
    project_onto_feasible_set_v1,
    run_lagrangian_floor_recovery_bench_v1,
)
from coordpy.learned_economics_controller_v1 import (
    W81_ECONOMICS_ACTIONS,
)
from coordpy.constrained_policy_optimisation_v1 import (
    _build_floor_pressing_features,
    _drifted_sim_for_floor_press,
    _train_initial_controller,
)


# ---------------------------------------------------------------
# Config: validation + content-addressing.
# ---------------------------------------------------------------

def test_w84_config_validates_unknown_action():
    with pytest.raises(ValueError):
        ConstrainedPolicyConfigV1(
            schema=W84_CONSTRAINED_POLICY_V1_SCHEMA_VERSION,
            action_floors=(("not_an_action", 0.1),),
            action_ceilings=(),
            action_cost_ceilings=(),
        )


def test_w84_config_validates_floor_range():
    with pytest.raises(ValueError):
        ConstrainedPolicyConfigV1(
            schema=W84_CONSTRAINED_POLICY_V1_SCHEMA_VERSION,
            action_floors=(("abstain", 1.5),),  # > 1
            action_ceilings=(),
            action_cost_ceilings=(),
        )


def test_w84_config_cid_is_stable():
    a = ConstrainedPolicyConfigV1(
        schema=W84_CONSTRAINED_POLICY_V1_SCHEMA_VERSION,
        action_floors=(("abstain", 0.1),),
        action_ceilings=(("replay", 0.5),),
        action_cost_ceilings=(("runtime_recompute", 1000.0),),
        whitelist_actions=("replay", "abstain"),
    )
    b = ConstrainedPolicyConfigV1(
        schema=W84_CONSTRAINED_POLICY_V1_SCHEMA_VERSION,
        action_floors=(("abstain", 0.1),),
        action_ceilings=(("replay", 0.5),),
        action_cost_ceilings=(("runtime_recompute", 1000.0),),
        whitelist_actions=("replay", "abstain"),
    )
    assert a.cid() == b.cid()
    assert len(a.cid()) == 64


def test_w84_whitelist_implicit_ceiling_0():
    cfg = ConstrainedPolicyConfigV1(
        schema=W84_CONSTRAINED_POLICY_V1_SCHEMA_VERSION,
        action_floors=(),
        action_ceilings=(),
        action_cost_ceilings=(),
        whitelist_actions=("replay", "abstain"),
    )
    assert cfg.ceiling_for("replay") == 1.0
    # Non-whitelisted action gets ceiling 0.
    assert cfg.ceiling_for("promote_to_richer_substrate") == 0.0


# ---------------------------------------------------------------
# Projection.
# ---------------------------------------------------------------

def test_w84_projection_clips_to_floor():
    cfg = ConstrainedPolicyConfigV1(
        schema=W84_CONSTRAINED_POLICY_V1_SCHEMA_VERSION,
        action_floors=(("abstain", 0.1),),
        action_ceilings=(),
        action_cost_ceilings=(),
    )
    # The 4 non-abstain actions have prob ~0.25; abstain has 0.0.
    p = np.array([0.25, 0.25, 0.25, 0.25, 0.0],
                 dtype=np.float64)
    proj = project_onto_feasible_set_v1(p, config=cfg)
    abstain_idx = W81_ECONOMICS_ACTIONS.index("abstain")
    assert proj[abstain_idx] >= 0.10 - 1e-9
    assert abs(proj.sum() - 1.0) < 1e-9


def test_w84_projection_respects_ceiling():
    cfg = ConstrainedPolicyConfigV1(
        schema=W84_CONSTRAINED_POLICY_V1_SCHEMA_VERSION,
        action_floors=(),
        action_ceilings=(("replay", 0.10),),
        action_cost_ceilings=(),
    )
    p = np.array([0.80, 0.05, 0.05, 0.05, 0.05],
                 dtype=np.float64)
    proj = project_onto_feasible_set_v1(p, config=cfg)
    replay_idx = W81_ECONOMICS_ACTIONS.index("replay")
    assert proj[replay_idx] <= 0.10 + 1e-9
    assert abs(proj.sum() - 1.0) < 1e-9


# ---------------------------------------------------------------
# Lagrangian floor-recovery bench.
# ---------------------------------------------------------------

def test_w84_lagrangian_strictly_beats_reinforce_on_floor():
    """The 10-seed bench reports that Lagrangian respects the
    floor on strictly more seeds than unconstrained REINFORCE.
    """
    rep = run_lagrangian_floor_recovery_bench_v1(
        n_seeds=10, n_episodes=200,
        abstain_floor=0.10, floor_tolerance=0.01)
    assert rep.lagrangian_strictly_beats_reinforce_on_floor
    assert (rep.n_seeds_floor_respected_lagrangian
            > rep.n_seeds_floor_respected_reinforce)


def test_w84_lagrangian_bench_reports_bootstrap_ci():
    rep = run_lagrangian_floor_recovery_bench_v1(
        n_seeds=10, n_episodes=200)
    # Bootstrap CI must be ordered.
    assert (rep.bootstrap_floor_violation_ci_lo
            <= rep.bootstrap_floor_violation_ci_hi)


def test_w84_lagrangian_bench_majority_seeds_respect_floor():
    """≥ 6/10 seeds must respect the floor with the Lagrangian
    refinement at floor=0.10, tolerance=0.01.
    """
    rep = run_lagrangian_floor_recovery_bench_v1(
        n_seeds=10, n_episodes=200,
        abstain_floor=0.10, floor_tolerance=0.01)
    assert rep.n_seeds_floor_respected_lagrangian >= 6


def test_w84_lagrangian_refinement_emits_violation_logs():
    cfg = ConstrainedPolicyConfigV1(
        schema=W84_CONSTRAINED_POLICY_V1_SCHEMA_VERSION,
        action_floors=(("abstain", 0.10),),
        action_ceilings=(),
        action_cost_ceilings=(),
    )
    init = _train_initial_controller(seed=84_001)
    sim = _drifted_sim_for_floor_press()
    train = _build_floor_pressing_features(seed=84_002, n=32)
    eval_ = _build_floor_pressing_features(seed=84_003, n=32)
    lag = LagrangianRefinementV1(n_episodes=40, seed=84_004)
    _post, rep, logs = lag.refine(
        controller=init, sim=sim, config=cfg,
        train_features=train, eval_features=eval_)
    assert len(logs) == 40
    # Every log entry is content-addressed.
    for log in logs:
        assert len(log.cid()) == 64
    # The violation chain CID covers every log CID.
    assert len(rep.violation_chain_cid) == 64
    assert (rep.violation_log_cids
            == tuple(str(log.cid()) for log in logs))


def test_w84_lagrangian_price_of_safety_bounded():
    """Price of safety is reported. We don't constrain the sign
    here (in our floor-press regime, the Lagrangian sometimes
    beats unconstrained REINFORCE because the unconstrained policy
    gets stuck in a poor local optimum). We assert the metric is
    finite and reported.
    """
    rep = run_lagrangian_floor_recovery_bench_v1(
        n_seeds=4, n_episodes=80)
    assert np.isfinite(rep.mean_price_of_safety)


def test_w84_constraint_violation_log_serializable():
    cfg = ConstrainedPolicyConfigV1(
        schema=W84_CONSTRAINED_POLICY_V1_SCHEMA_VERSION,
        action_floors=(("abstain", 0.10),),
        action_ceilings=(),
        action_cost_ceilings=(),
    )
    init = _train_initial_controller(seed=84_010)
    sim = _drifted_sim_for_floor_press()
    train = _build_floor_pressing_features(seed=84_011, n=16)
    eval_ = _build_floor_pressing_features(seed=84_012, n=16)
    lag = LagrangianRefinementV1(n_episodes=10, seed=84_013)
    _post, _rep, logs = lag.refine(
        controller=init, sim=sim, config=cfg,
        train_features=train, eval_features=eval_)
    for log in logs:
        d = log.to_dict()
        assert "feasible" in d
        assert "floor_violations" in d
