"""Phase 84 -- W37 anchor-cross-host basis-trajectory ratification tests."""

from __future__ import annotations

import dataclasses
import unittest

from vision_mvp.wevra.team_coord import (
    CrossHostBasisTrajectoryEntry,
    CrossHostBasisTrajectoryRatificationEnvelope,
    W37_CROSS_HOST_TRAJECTORY_SCHEMA_VERSION,
    W37_BRANCH_TRAJECTORY_REROUTED,
    W37_BRANCH_TRAJECTORY_NO_HISTORY,
    W37_BRANCH_TRAJECTORY_DISAGREEMENT,
    W37_BRANCH_TRAJECTORY_UNSAFE,
    W37_BRANCH_TRAJECTORY_NO_TRIGGER,
    select_cross_host_trajectory_projection,
    verify_cross_host_trajectory_ratification,
    _compute_cross_host_trajectory_state_cid,
    _compute_w37_trajectory_audit_cid,
    _compute_w37_trajectory_topology_cid,
    _compute_w37_manifest_v7_cid,
)
from vision_mvp.experiments.phase84_cross_host_basis_trajectory import (
    _stable_schema_capsule,
    run_phase84_seed_sweep,
)


def _entry(
        host_id: str,
        oracle_id: str,
        top_set: tuple[str, ...],
        ewma: float = 1.0,
        n_obs: int = 2,
        n_anchored: int = 2,
        anchored_hosts: tuple[str, ...] = ("anchor_a", "anchor_b"),
        last_cell_idx: int = 0,
) -> CrossHostBasisTrajectoryEntry:
    return CrossHostBasisTrajectoryEntry(
        host_id=host_id,
        oracle_id=oracle_id,
        top_set=top_set,
        ewma_anchored_match=ewma,
        n_observations=n_obs,
        n_anchored_observations=n_anchored,
        anchored_host_ids=anchored_hosts,
        last_cell_idx=last_cell_idx,
    )


def _clean_envelope() -> tuple[
        CrossHostBasisTrajectoryRatificationEnvelope, dict]:
    schema = _stable_schema_capsule()
    parent_w36_cid = "ab" * 32
    entries = (
        _entry("mac1", "service_graph", ("gold",)),
        _entry("mac1", "change_history", ("gold",)),
        _entry("mac1", "oncall_notes", ("gold",)),
    )
    state_cid = _compute_cross_host_trajectory_state_cid(
        trajectory_entries=entries)
    topology_cid = _compute_w37_trajectory_topology_cid(
        registered_host_ids=("mac1", "mac_remote", "mac_shadow",
                             "anchor_a", "anchor_b"),
        registered_anchor_host_ids=("anchor_a", "anchor_b"))
    audit_cid = _compute_w37_trajectory_audit_cid(
        projection_branch=W37_BRANCH_TRAJECTORY_REROUTED,
        selected_oracle_ids=(
            "service_graph", "change_history", "oncall_notes"),
        supporting_host_ids=("mac1",),
        trajectory_anchor_host_ids=("anchor_a", "anchor_b"),
        projection_top_set=("gold",),
        trajectory_score=1.0,
        trajectory_margin=1.0,
        trajectory_threshold=0.80,
        trajectory_margin_min=0.10,
        min_anchored_observations=2,
        min_trajectory_anchored_hosts=2,
    )
    manifest_cid = _compute_w37_manifest_v7_cid(
        parent_w36_cid=parent_w36_cid,
        cross_host_trajectory_state_cid=state_cid,
        trajectory_audit_cid=audit_cid,
        trajectory_topology_cid=topology_cid,
    )
    env = CrossHostBasisTrajectoryRatificationEnvelope(
        schema_version=W37_CROSS_HOST_TRAJECTORY_SCHEMA_VERSION,
        schema_cid=str(schema.cid),
        parent_w36_cid=parent_w36_cid,
        trajectory_entries=entries,
        cross_host_trajectory_state_cid=state_cid,
        trajectory_topology_cid=topology_cid,
        projection_branch=W37_BRANCH_TRAJECTORY_REROUTED,
        selected_oracle_ids=(
            "service_graph", "change_history", "oncall_notes"),
        supporting_host_ids=("mac1",),
        trajectory_anchor_host_ids=("anchor_a", "anchor_b"),
        projection_top_set=("gold",),
        trajectory_score=1.0,
        trajectory_margin=1.0,
        trajectory_threshold=0.80,
        trajectory_margin_min=0.10,
        min_anchored_observations=2,
        min_trajectory_anchored_hosts=2,
        trajectory_audit_cid=audit_cid,
        manifest_v7_cid=manifest_cid,
        cell_index=0,
        wire_required=True,
    )
    kwargs = dict(
        registered_schema=schema,
        registered_parent_w36_cid=parent_w36_cid,
        registered_oracle_ids=frozenset(
            {"service_graph", "change_history", "oncall_notes"}),
        registered_host_ids=frozenset(
            {"mac1", "mac_remote", "mac_shadow",
             "anchor_a", "anchor_b"}),
        registered_anchor_host_ids=frozenset({"anchor_a", "anchor_b"}),
        registered_trajectory_topology_cid=topology_cid,
    )
    return env, kwargs


class W37SelectorTests(unittest.TestCase):

    def test_single_host_recovery_with_anchored_history_reroutes(
            self) -> None:
        traj = (
            _entry("mac1", "service_graph", ("gold",), ewma=1.0,
                   n_anchored=2,
                   anchored_hosts=("mac_remote", "mac_shadow")),
        )
        current = (("mac1", "service_graph", ("gold",)),)
        top, oids, hosts, anchors, score, margin, branch = (
            select_cross_host_trajectory_projection(
                trajectory_entries=traj,
                current_basis_entries=current,
            ))
        self.assertEqual(("gold",), top)
        self.assertEqual(W37_BRANCH_TRAJECTORY_REROUTED, branch)
        self.assertEqual(("mac1",), hosts)
        self.assertEqual(("service_graph",), oids)
        self.assertEqual(("mac_remote", "mac_shadow"), anchors)

    def test_no_history_returns_no_history_branch(self) -> None:
        traj = ()
        current = (("mac1", "service_graph", ("gold",)),)
        _top, _o, _h, _a, _s, _m, branch = (
            select_cross_host_trajectory_projection(
                trajectory_entries=traj,
                current_basis_entries=current,
            ))
        self.assertEqual(W37_BRANCH_TRAJECTORY_NO_HISTORY, branch)

    def test_below_threshold_returns_unsafe(self) -> None:
        traj = (
            _entry("mac1", "service_graph", ("gold",), ewma=0.4),
        )
        current = (("mac1", "service_graph", ("gold",)),)
        _top, _o, _h, _a, _s, _m, branch = (
            select_cross_host_trajectory_projection(
                trajectory_entries=traj,
                current_basis_entries=current,
                trajectory_threshold=0.80,
            ))
        self.assertEqual(W37_BRANCH_TRAJECTORY_UNSAFE, branch)

    def test_single_host_anchoring_does_not_pass_min_anchor_hosts(
            self) -> None:
        # Only one anchor host -- below ``min_trajectory_anchored_hosts``.
        # The selector classifies this as UNSAFE (a candidate exists but
        # fails the anchored-host requirement) rather than NO_HISTORY,
        # which is reserved for cells where no current basis entry has
        # any matching trajectory.
        traj = (
            _entry("mac1", "service_graph", ("gold",), ewma=1.0,
                   n_anchored=2, anchored_hosts=("mac1",)),
        )
        current = (("mac1", "service_graph", ("gold",)),)
        _top, _o, _h, _a, _s, _m, branch = (
            select_cross_host_trajectory_projection(
                trajectory_entries=traj,
                current_basis_entries=current,
                min_trajectory_anchored_hosts=2,
            ))
        self.assertEqual(W37_BRANCH_TRAJECTORY_UNSAFE, branch)

    def test_two_competing_top_sets_yields_disagreement(self) -> None:
        traj = (
            _entry("mac1", "service_graph", ("a", "b"), ewma=1.0),
            _entry("mac1", "service_graph", ("c", "d"), ewma=1.0),
        )
        current = (
            ("mac1", "service_graph", ("a", "b")),
            ("mac1", "service_graph", ("c", "d")),
        )
        _top, _o, _h, _a, _s, _m, branch = (
            select_cross_host_trajectory_projection(
                trajectory_entries=traj,
                current_basis_entries=current,
                trajectory_margin_min=0.10,
            ))
        self.assertEqual(W37_BRANCH_TRAJECTORY_DISAGREEMENT, branch)


class W37VerifierTests(unittest.TestCase):

    def test_clean_envelope_passes(self) -> None:
        env, kwargs = _clean_envelope()
        outcome = verify_cross_host_trajectory_ratification(env, **kwargs)
        self.assertTrue(outcome.ok, outcome.reason)

    def test_empty_envelope_rejected(self) -> None:
        _env, kwargs = _clean_envelope()
        outcome = verify_cross_host_trajectory_ratification(None, **kwargs)
        self.assertEqual("empty_w37_envelope", outcome.reason)

    def test_schema_version_unknown_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, schema_version="wrong")
        outcome = verify_cross_host_trajectory_ratification(bad, **kwargs)
        self.assertEqual("w37_schema_version_unknown", outcome.reason)

    def test_schema_cid_mismatch_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, schema_cid="wrong")
        outcome = verify_cross_host_trajectory_ratification(bad, **kwargs)
        self.assertEqual("w37_schema_cid_mismatch", outcome.reason)

    def test_parent_w36_cid_mismatch_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, parent_w36_cid="cd" * 32)
        outcome = verify_cross_host_trajectory_ratification(bad, **kwargs)
        self.assertEqual("w36_parent_cid_mismatch", outcome.reason)

    def test_projection_branch_unknown_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, projection_branch="unknown")
        outcome = verify_cross_host_trajectory_ratification(bad, **kwargs)
        self.assertEqual("w37_projection_branch_unknown", outcome.reason)

    def test_unregistered_oracle_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        entries = list(env.trajectory_entries)
        entries[0] = dataclasses.replace(entries[0], oracle_id="rogue")
        bad = dataclasses.replace(env, trajectory_entries=tuple(entries))
        outcome = verify_cross_host_trajectory_ratification(bad, **kwargs)
        self.assertEqual(
            "w37_trajectory_entry_unregistered_oracle", outcome.reason)

    def test_unregistered_host_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        entries = list(env.trajectory_entries)
        entries[0] = dataclasses.replace(entries[0], host_id="rogue_host")
        bad = dataclasses.replace(env, trajectory_entries=tuple(entries))
        outcome = verify_cross_host_trajectory_ratification(bad, **kwargs)
        self.assertEqual(
            "w37_trajectory_entry_unregistered_host", outcome.reason)

    def test_ewma_out_of_range_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        entries = list(env.trajectory_entries)
        entries[0] = dataclasses.replace(
            entries[0], ewma_anchored_match=2.0)
        bad = dataclasses.replace(env, trajectory_entries=tuple(entries))
        outcome = verify_cross_host_trajectory_ratification(bad, **kwargs)
        self.assertEqual(
            "w37_trajectory_ewma_out_of_range", outcome.reason)

    def test_observation_count_invalid_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        entries = list(env.trajectory_entries)
        entries[0] = dataclasses.replace(
            entries[0], n_anchored_observations=99)
        bad = dataclasses.replace(env, trajectory_entries=tuple(entries))
        outcome = verify_cross_host_trajectory_ratification(bad, **kwargs)
        self.assertEqual(
            "w37_trajectory_observation_count_invalid", outcome.reason)

    def test_state_cid_mismatch_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(
            env, cross_host_trajectory_state_cid="00" * 32)
        outcome = verify_cross_host_trajectory_ratification(bad, **kwargs)
        self.assertEqual(
            "w37_trajectory_state_cid_mismatch", outcome.reason)

    def test_projection_top_set_unregistered_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, projection_top_set=("foreign",))
        outcome = verify_cross_host_trajectory_ratification(bad, **kwargs)
        self.assertEqual(
            "w37_projection_top_set_unregistered", outcome.reason)

    def test_trajectory_requirement_invalid_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, trajectory_threshold=2.0)
        outcome = verify_cross_host_trajectory_ratification(bad, **kwargs)
        self.assertEqual(
            "w37_trajectory_requirement_invalid", outcome.reason)

    def test_trajectory_topology_cid_mismatch_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, trajectory_topology_cid="00" * 32)
        outcome = verify_cross_host_trajectory_ratification(bad, **kwargs)
        self.assertEqual(
            "w37_trajectory_topology_cid_mismatch", outcome.reason)

    def test_manifest_mismatch_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, manifest_v7_cid="00" * 32)
        outcome = verify_cross_host_trajectory_ratification(bad, **kwargs)
        self.assertEqual("w37_manifest_v7_cid_mismatch", outcome.reason)

    def test_outer_cid_mismatch_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, w37_cid="00" * 32)
        outcome = verify_cross_host_trajectory_ratification(bad, **kwargs)
        self.assertEqual("w37_outer_cid_mismatch", outcome.reason)


class W37Phase84BenchmarkTests(unittest.TestCase):

    def test_single_host_trajectory_recover_correctness_gain_over_w36(
            self) -> None:
        result = run_phase84_seed_sweep(
            bank="single_host_trajectory_recover",
            n_eval=16,
            seeds=(11, 17, 23, 29, 31),
        )
        self.assertGreaterEqual(
            result["min_delta_correctness_w37_w36"], 0.20)
        self.assertEqual(1.0, result["min_trust_precision_w37"])
        self.assertLessEqual(result["max_overhead_w37_per_cell"], 1)

    def test_trivial_w37_byte_for_w36(self) -> None:
        result = run_phase84_seed_sweep(
            bank="trivial_w37",
            n_eval=16,
            seeds=(11, 17, 23, 29, 31),
        )
        self.assertTrue(result["all_byte_equivalent_w37_w36"])

    def test_no_trajectory_history_preserves_w36(self) -> None:
        result = run_phase84_seed_sweep(
            bank="no_trajectory_history",
            n_eval=16,
            seeds=(11, 17, 23, 29, 31),
        )
        self.assertEqual(0.0, result["min_delta_correctness_w37_w36"])
        self.assertEqual(0.0, result["max_delta_correctness_w37_w36"])
        self.assertEqual(1.0, result["min_trust_precision_w37"])

    def test_poisoned_trajectory_does_not_reroute(self) -> None:
        result = run_phase84_seed_sweep(
            bank="poisoned_trajectory",
            n_eval=16,
            seeds=(11, 17, 23, 29, 31),
        )
        self.assertEqual(0.0, result["min_delta_correctness_w37_w36"])
        self.assertEqual(0.0, result["max_delta_correctness_w37_w36"])
        self.assertEqual(1.0, result["min_trust_precision_w37"])

    def test_trajectory_disagreement_preserves_w36(self) -> None:
        result = run_phase84_seed_sweep(
            bank="trajectory_disagreement",
            n_eval=16,
            seeds=(11, 17, 23, 29, 31),
        )
        self.assertEqual(0.0, result["min_delta_correctness_w37_w36"])
        self.assertEqual(0.0, result["max_delta_correctness_w37_w36"])
        self.assertEqual(1.0, result["min_trust_precision_w37"])


if __name__ == "__main__":
    unittest.main()
