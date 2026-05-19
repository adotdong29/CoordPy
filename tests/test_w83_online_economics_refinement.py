"""W83 — online economics refinement tests."""

from __future__ import annotations


def _train_offline_ctrl(seed: int = 12345):
    from coordpy.learned_economics_controller_v1 import (
        build_learned_economics_controller_v1,
        train_learned_economics_controller,
        build_economics_dataset_v1,
    )
    ctrl = build_learned_economics_controller_v1()
    X_train, y_train, _ = build_economics_dataset_v1(
        n_samples=200, seed=int(seed))
    ctrl, _ = train_learned_economics_controller(
        controller=ctrl,
        train_features=X_train,
        train_optimal_action_indices=y_train,
        n_iters=80)
    return ctrl


def test_w83_drifted_simulation_distinct_from_default():
    from coordpy.online_economics_refinement_v1 import (
        build_drifted_deployment_simulation_v1,
    )
    sim = build_drifted_deployment_simulation_v1()
    # The drifted sim has explicit multipliers.
    assert float(sim.replay_cost_multiplier) != 1.0
    assert float(
        sim.transcript_recompute_cost_multiplier) != 1.0
    assert len(sim.cid()) == 64


def test_w83_online_refinement_beats_offline_on_drifted_sim():
    import numpy as np
    from coordpy.learned_economics_controller_v1 import (
        build_economics_dataset_v1,
    )
    from coordpy.online_economics_refinement_v1 import (
        build_drifted_deployment_simulation_v1,
        online_refine_economics_controller_v1,
    )
    ctrl = _train_offline_ctrl(seed=12345)
    dep_sim = build_drifted_deployment_simulation_v1()
    X_ev, _, _ = build_economics_dataset_v1(
        n_samples=80, seed=98765)
    y_ev = np.zeros(
        (X_ev.shape[0],), dtype=np.int64)
    for i in range(int(X_ev.shape[0])):
        y_ev[i] = int(
            dep_sim.optimal_action_index(
                feature=X_ev[i]))
    _, rep = online_refine_economics_controller_v1(
        controller=ctrl,
        deployment_sim=dep_sim,
        eval_features=X_ev,
        eval_optimal_actions=y_ev,
        n_online_episodes=80)
    assert float(
        rep.post_eval_mean_utility) > float(
        rep.pre_eval_mean_utility), rep.to_dict()
    assert float(
        rep.post_eval_optimality_gap) < float(
        rep.pre_eval_optimality_gap), rep.to_dict()
    assert bool(rep.online_refinement_beats_offline)


def test_w83_online_refinement_episode_cids_unique_and_chained():
    import numpy as np
    from coordpy.learned_economics_controller_v1 import (
        build_economics_dataset_v1,
    )
    from coordpy.online_economics_refinement_v1 import (
        build_drifted_deployment_simulation_v1,
        online_refine_economics_controller_v1,
    )
    ctrl = _train_offline_ctrl(seed=22221)
    dep_sim = build_drifted_deployment_simulation_v1()
    X_ev, _, _ = build_economics_dataset_v1(
        n_samples=40, seed=44441)
    y_ev = np.zeros(
        (X_ev.shape[0],), dtype=np.int64)
    for i in range(int(X_ev.shape[0])):
        y_ev[i] = int(
            dep_sim.optimal_action_index(
                feature=X_ev[i]))
    _, rep = online_refine_economics_controller_v1(
        controller=ctrl,
        deployment_sim=dep_sim,
        eval_features=X_ev,
        eval_optimal_actions=y_ev,
        n_online_episodes=20)
    assert int(rep.n_online_episodes) == 20
    assert int(len(rep.episode_cids)) == 20
    assert len(set(rep.episode_cids)) == 20
    assert len(rep.episode_chain_cid) == 64


def test_w83_online_refinement_witness_emitted():
    import numpy as np
    from coordpy.learned_economics_controller_v1 import (
        build_economics_dataset_v1,
    )
    from coordpy.online_economics_refinement_v1 import (
        build_drifted_deployment_simulation_v1,
        emit_online_economics_refinement_witness_v1,
        online_refine_economics_controller_v1,
    )
    ctrl = _train_offline_ctrl(seed=33333)
    dep_sim = build_drifted_deployment_simulation_v1()
    X_ev, _, _ = build_economics_dataset_v1(
        n_samples=40, seed=66666)
    y_ev = np.zeros(
        (X_ev.shape[0],), dtype=np.int64)
    for i in range(int(X_ev.shape[0])):
        y_ev[i] = int(
            dep_sim.optimal_action_index(
                feature=X_ev[i]))
    _, rep = online_refine_economics_controller_v1(
        controller=ctrl,
        deployment_sim=dep_sim,
        eval_features=X_ev,
        eval_optimal_actions=y_ev,
        n_online_episodes=20)
    w = emit_online_economics_refinement_witness_v1(report=rep)
    assert len(w.cid()) == 64
