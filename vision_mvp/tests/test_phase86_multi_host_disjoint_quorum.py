"""Phase 86 -- W39 multi-host disjoint quorum consensus-reference tests."""

from __future__ import annotations

import dataclasses
import unittest

from vision_mvp.wevra.team_coord import (
    ConsensusReferenceProbe,
    MultiHostDisjointQuorumProbe,
    MultiHostDisjointQuorumRatificationEnvelope,
    MultiHostDisjointQuorumRegistry,
    MutuallyDisjointTopologyError,
    DisjointTopologyError,
    W39_MULTI_HOST_DISJOINT_QUORUM_SCHEMA_VERSION,
    W39_BRANCH_QUORUM_RATIFIED,
    W39_BRANCH_QUORUM_DIVERGENCE_ABSTAINED,
    W39_BRANCH_QUORUM_NO_REFERENCES,
    W39_BRANCH_QUORUM_NO_TRIGGER,
    W39_BRANCH_QUORUM_INSUFFICIENT,
    W39_BRANCH_QUORUM_SPLIT,
    W39_BRANCH_QUORUM_REFERENCE_WEAK,
    W39_BRANCH_QUORUM_REJECTED,
    W39_BRANCH_TRIVIAL_QUORUM_PASSTHROUGH,
    select_multi_host_disjoint_quorum_decision,
    verify_multi_host_disjoint_quorum_ratification,
    _compute_w39_quorum_state_cid,
    _compute_w39_quorum_topology_cid,
    _compute_w39_mutual_disjointness_cid,
    _compute_w39_quorum_decision_cid,
    _compute_w39_quorum_audit_cid,
    _compute_w39_manifest_v9_cid,
)
from vision_mvp.experiments.phase86_multi_host_disjoint_quorum import (
    _stable_schema_capsule,
    run_phase86_seed_sweep,
)


def _member(top, hosts=("mac_off_cluster_a",),
            oracles=("disjoint_quorum_oracle_a",),
            strength=1.0, cell_idx=0):
    return ConsensusReferenceProbe(
        top_set=tuple(sorted(top)),
        consensus_host_ids=tuple(hosts),
        consensus_oracle_ids=tuple(oracles),
        consensus_strength=float(strength),
        cell_idx=int(cell_idx),
    )


def _quorum_probe(member_tops, hosts_per_pool=None,
                  oracles_per_pool=None,
                  strengths=None,
                  quorum_min=2, min_quorum_probes=2, cell_idx=0):
    if hosts_per_pool is None:
        hosts_per_pool = [
            ("mac_off_cluster_a",), ("mac_off_cluster_b",)]
    if oracles_per_pool is None:
        oracles_per_pool = [
            ("disjoint_quorum_oracle_a",),
            ("disjoint_quorum_oracle_b",)]
    if strengths is None:
        strengths = [1.0] * len(member_tops)
    members = []
    for k, top in enumerate(member_tops):
        members.append(_member(
            top, hosts=hosts_per_pool[k],
            oracles=oracles_per_pool[k],
            strength=strengths[k], cell_idx=cell_idx))
    return MultiHostDisjointQuorumProbe(
        member_probes=tuple(members),
        quorum_min=int(quorum_min),
        min_quorum_probes=int(min_quorum_probes),
        cell_idx=int(cell_idx),
    )


class W39SelectorTests(unittest.TestCase):
    def test_no_probe_returns_no_references(self) -> None:
        branch, *_ = select_multi_host_disjoint_quorum_decision(
            quorum_probe=None,
            w38_candidate_top_set=("a", "b"),
            w38_rerouted=True)
        self.assertEqual(branch, W39_BRANCH_QUORUM_NO_REFERENCES)

    def test_w38_not_rerouted_returns_no_trigger(self) -> None:
        branch, *_ = select_multi_host_disjoint_quorum_decision(
            quorum_probe=_quorum_probe([("a", "b"), ("a", "b")]),
            w38_candidate_top_set=("a", "b"),
            w38_rerouted=False)
        self.assertEqual(branch, W39_BRANCH_QUORUM_NO_TRIGGER)

    def test_insufficient_quorum_falls_through(self) -> None:
        branch, *_ = select_multi_host_disjoint_quorum_decision(
            quorum_probe=_quorum_probe(
                [("a", "b")],
                hosts_per_pool=[("mac_off_cluster_a",)],
                oracles_per_pool=[("disjoint_quorum_oracle_a",)],
                strengths=[1.0]),
            w38_candidate_top_set=("a", "b"),
            w38_rerouted=True,
            quorum_min=2, min_quorum_probes=2)
        self.assertEqual(branch, W39_BRANCH_QUORUM_INSUFFICIENT)

    def test_all_weak_returns_reference_weak(self) -> None:
        branch, *_ = select_multi_host_disjoint_quorum_decision(
            quorum_probe=_quorum_probe(
                [("a", "b"), ("a", "b")],
                strengths=[0.0, 0.0]),
            w38_candidate_top_set=("a", "b"),
            w38_rerouted=True,
            consensus_strength_min=0.66)
        self.assertEqual(branch, W39_BRANCH_QUORUM_REFERENCE_WEAK)

    def test_quorum_agreement_returns_ratified(self) -> None:
        branch, n_agree, n_disagree, _, n_total, top, *_ = (
            select_multi_host_disjoint_quorum_decision(
                quorum_probe=_quorum_probe([("a", "b"), ("a", "b")]),
                w38_candidate_top_set=("a", "b"),
                w38_rerouted=True))
        self.assertEqual(branch, W39_BRANCH_QUORUM_RATIFIED)
        self.assertEqual(n_agree, 2)
        self.assertEqual(n_disagree, 0)
        self.assertEqual(n_total, 2)
        self.assertEqual(top, ("a", "b"))

    def test_quorum_disagreement_returns_divergence(self) -> None:
        branch, n_agree, n_disagree, *_ = (
            select_multi_host_disjoint_quorum_decision(
                quorum_probe=_quorum_probe(
                    [("c", "d"), ("c", "d")]),
                w38_candidate_top_set=("a", "b"),
                w38_rerouted=True))
        self.assertEqual(
            branch, W39_BRANCH_QUORUM_DIVERGENCE_ABSTAINED)
        self.assertEqual(n_agree, 0)
        self.assertEqual(n_disagree, 2)

    def test_split_returns_split_branch(self) -> None:
        branch, n_agree, n_disagree, *_ = (
            select_multi_host_disjoint_quorum_decision(
                quorum_probe=_quorum_probe(
                    [("a", "b"), ("c", "d")]),
                w38_candidate_top_set=("a", "b"),
                w38_rerouted=True,
                quorum_min=2))
        self.assertEqual(branch, W39_BRANCH_QUORUM_SPLIT)
        self.assertEqual(n_agree, 1)
        self.assertEqual(n_disagree, 1)


class W39RegistryTests(unittest.TestCase):
    def test_overlapping_trajectory_topology_raises(self) -> None:
        schema = _stable_schema_capsule()
        with self.assertRaises(DisjointTopologyError):
            MultiHostDisjointQuorumRegistry(
                schema=schema,
                registered_quorum_pool_host_ids=(
                    frozenset({"mac1"}),),
                registered_quorum_pool_oracle_ids=(
                    frozenset({"o1"}),),
                registered_trajectory_host_ids=frozenset({"mac1"}),
            )

    def test_non_mutually_disjoint_pools_raise(self) -> None:
        schema = _stable_schema_capsule()
        with self.assertRaises(MutuallyDisjointTopologyError):
            MultiHostDisjointQuorumRegistry(
                schema=schema,
                registered_quorum_pool_host_ids=(
                    frozenset({"mac_a", "mac_b"}),
                    frozenset({"mac_b", "mac_c"})),
                registered_quorum_pool_oracle_ids=(
                    frozenset({"o1"}), frozenset({"o2"})),
                registered_trajectory_host_ids=frozenset(
                    {"mac_traj"}),
            )

    def test_disjoint_topology_constructs(self) -> None:
        schema = _stable_schema_capsule()
        reg = MultiHostDisjointQuorumRegistry(
            schema=schema,
            registered_quorum_pool_host_ids=(
                frozenset({"mac_off_a"}),
                frozenset({"mac_off_b"})),
            registered_quorum_pool_oracle_ids=(
                frozenset({"o1"}), frozenset({"o2"})),
            registered_trajectory_host_ids=frozenset(
                {"mac_traj"}),
        )
        self.assertFalse(reg.is_trivial)


class W39VerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = _stable_schema_capsule()
        self.pool_hosts = (
            frozenset({"mac_off_cluster_a"}),
            frozenset({"mac_off_cluster_b"}),
        )
        self.pool_oracles = (
            frozenset({"disjoint_quorum_oracle_a"}),
            frozenset({"disjoint_quorum_oracle_b"}),
        )
        self.trajectory_hosts = frozenset(
            {"mac1", "mac_remote", "mac_shadow", "mac_consensus"})
        self.topology_cid = _compute_w39_quorum_topology_cid(
            registered_quorum_pool_host_ids=self.pool_hosts,
            registered_quorum_pool_oracle_ids=self.pool_oracles,
            registered_trajectory_host_ids=self.trajectory_hosts)
        self.mutual_cid = _compute_w39_mutual_disjointness_cid(
            registered_quorum_pool_host_ids=self.pool_hosts)
        self.parent_w38_cid = "f00d" * 16
        self.kwargs = dict(
            registered_schema=self.schema,
            registered_parent_w38_cid=self.parent_w38_cid,
            registered_quorum_pool_host_ids=self.pool_hosts,
            registered_quorum_pool_oracle_ids=self.pool_oracles,
            registered_trajectory_host_ids=self.trajectory_hosts,
            registered_quorum_topology_cid=self.topology_cid,
            registered_mutual_disjointness_cid=self.mutual_cid,
        )

    def _build_clean_envelope(self) -> (
            MultiHostDisjointQuorumRatificationEnvelope):
        probe = _quorum_probe([("a", "b"), ("a", "b")])
        state_cid = _compute_w39_quorum_state_cid(probe=probe)
        decision_cid = _compute_w39_quorum_decision_cid(
            projection_branch=W39_BRANCH_QUORUM_RATIFIED,
            n_agree=2, n_disagree=0, n_weak=0, n_total=2,
            decision_top_set=("a", "b"),
            per_probe_divergence_scores=(0.0, 0.0),
            per_probe_branches=(W39_BRANCH_QUORUM_RATIFIED,
                                W39_BRANCH_QUORUM_RATIFIED),
        )
        audit_cid = _compute_w39_quorum_audit_cid(
            projection_branch=W39_BRANCH_QUORUM_RATIFIED,
            w38_top_set=("a", "b"),
            decision_top_set=("a", "b"),
            n_agree=2, n_disagree=0, n_weak=0, n_total=2,
            quorum_min=2, min_quorum_probes=2,
            consensus_strength_min=0.66,
            divergence_margin_min=0.10,
        )
        manifest_cid = _compute_w39_manifest_v9_cid(
            parent_w38_cid=self.parent_w38_cid,
            quorum_state_cid=state_cid,
            quorum_audit_cid=audit_cid,
            quorum_topology_cid=self.topology_cid,
            quorum_decision_cid=decision_cid,
            mutual_disjointness_cid=self.mutual_cid,
        )
        return MultiHostDisjointQuorumRatificationEnvelope(
            schema_version=(
                W39_MULTI_HOST_DISJOINT_QUORUM_SCHEMA_VERSION),
            schema_cid=self.schema.cid,
            parent_w38_cid=self.parent_w38_cid,
            quorum_probe=probe,
            quorum_state_cid=state_cid,
            quorum_topology_cid=self.topology_cid,
            quorum_decision_cid=decision_cid,
            mutual_disjointness_cid=self.mutual_cid,
            projection_branch=W39_BRANCH_QUORUM_RATIFIED,
            w38_top_set=("a", "b"),
            decision_top_set=("a", "b"),
            n_agree=2, n_disagree=0, n_weak=0, n_total=2,
            quorum_min=2, min_quorum_probes=2,
            consensus_strength_min=0.66,
            divergence_margin_min=0.10,
            per_probe_divergence_scores=(0.0, 0.0),
            per_probe_branches=(
                W39_BRANCH_QUORUM_RATIFIED,
                W39_BRANCH_QUORUM_RATIFIED),
            quorum_audit_cid=audit_cid,
            manifest_v9_cid=manifest_cid,
            cell_index=0,
            wire_required=True,
        )

    def test_clean_envelope_passes(self) -> None:
        env = self._build_clean_envelope()
        outcome = verify_multi_host_disjoint_quorum_ratification(
            env, **self.kwargs)
        self.assertTrue(outcome.ok, msg=str(outcome.reason))
        self.assertEqual(outcome.reason, "ok")
        self.assertGreaterEqual(outcome.n_checks, 14)

    def test_empty_envelope_rejected(self) -> None:
        outcome = verify_multi_host_disjoint_quorum_ratification(
            None, **self.kwargs)
        self.assertEqual(outcome.reason, "empty_w39_envelope")

    def test_schema_version_unknown_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, schema_version="bogus", w39_cid="")
        outcome = verify_multi_host_disjoint_quorum_ratification(
            bad, **self.kwargs)
        self.assertEqual(
            outcome.reason, "w39_schema_version_unknown")

    def test_schema_cid_mismatch_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, schema_cid="bogus", w39_cid="")
        outcome = verify_multi_host_disjoint_quorum_ratification(
            bad, **self.kwargs)
        self.assertEqual(outcome.reason, "w39_schema_cid_mismatch")

    def test_parent_w38_cid_mismatch_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, parent_w38_cid="bogus", w39_cid="")
        outcome = verify_multi_host_disjoint_quorum_ratification(
            bad, **self.kwargs)
        self.assertEqual(outcome.reason, "w38_parent_cid_mismatch")

    def test_projection_branch_unknown_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, projection_branch="bogus", w39_cid="")
        outcome = verify_multi_host_disjoint_quorum_ratification(
            bad, **self.kwargs)
        self.assertEqual(
            outcome.reason, "w39_projection_branch_unknown")

    def test_unregistered_quorum_host_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad_probe = _quorum_probe(
            [("a", "b"), ("a", "b")],
            hosts_per_pool=[
                ("mac_unknown",), ("mac_off_cluster_b",)],
            oracles_per_pool=[
                ("disjoint_quorum_oracle_a",),
                ("disjoint_quorum_oracle_b",)])
        bad = dataclasses.replace(
            env, quorum_probe=bad_probe,
            quorum_state_cid=_compute_w39_quorum_state_cid(
                probe=bad_probe),
            w39_cid="")
        outcome = verify_multi_host_disjoint_quorum_ratification(
            bad, **self.kwargs)
        self.assertEqual(
            outcome.reason, "w39_quorum_probe_unregistered_host")

    def test_unregistered_quorum_oracle_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad_probe = _quorum_probe(
            [("a", "b"), ("a", "b")],
            oracles_per_pool=[
                ("oracle_unknown",),
                ("disjoint_quorum_oracle_b",)])
        bad = dataclasses.replace(
            env, quorum_probe=bad_probe,
            quorum_state_cid=_compute_w39_quorum_state_cid(
                probe=bad_probe),
            w39_cid="")
        outcome = verify_multi_host_disjoint_quorum_ratification(
            bad, **self.kwargs)
        self.assertEqual(
            outcome.reason, "w39_quorum_probe_unregistered_oracle")

    def test_quorum_disjoint_topology_violation_rejected(
            self) -> None:
        env = self._build_clean_envelope()
        # Verifier called with overlapping trajectory + pool: pool 0
        # contains ``mac_off_cluster_a`` AND we register it in the
        # trajectory hosts.
        bad_kwargs = dict(self.kwargs)
        bad_kwargs["registered_trajectory_host_ids"] = frozenset(
            {"mac_off_cluster_a", "mac1"})
        outcome = verify_multi_host_disjoint_quorum_ratification(
            env, **bad_kwargs)
        self.assertEqual(
            outcome.reason,
            "w39_quorum_disjoint_topology_violation")

    def test_quorum_mutual_disjointness_violation_rejected(
            self) -> None:
        env = self._build_clean_envelope()
        # Verifier called with overlapping pool 0 ∩ pool 1.  Build
        # a parallel kwargs with overlapping pool host IDs (the
        # registry would never construct such a config, but the
        # verifier itself must reject it as a defense-in-depth.)
        overlap_pools = (
            frozenset({"mac_off_cluster_a", "mac_shared"}),
            frozenset({"mac_off_cluster_b", "mac_shared"}),
        )
        bad_kwargs = dict(self.kwargs)
        bad_kwargs["registered_quorum_pool_host_ids"] = overlap_pools
        outcome = verify_multi_host_disjoint_quorum_ratification(
            env, **bad_kwargs)
        self.assertEqual(
            outcome.reason,
            "w39_quorum_mutual_disjointness_violation")

    def test_quorum_thresholds_invalid_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, consensus_strength_min=2.5, w39_cid="")
        outcome = verify_multi_host_disjoint_quorum_ratification(
            bad, **self.kwargs)
        self.assertEqual(
            outcome.reason, "w39_quorum_thresholds_invalid")

    def test_quorum_state_cid_mismatch_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, quorum_state_cid="bogus", w39_cid="")
        outcome = verify_multi_host_disjoint_quorum_ratification(
            bad, **self.kwargs)
        self.assertEqual(
            outcome.reason, "w39_quorum_state_cid_mismatch")

    def test_quorum_decision_cid_mismatch_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, quorum_decision_cid="bogus", w39_cid="")
        outcome = verify_multi_host_disjoint_quorum_ratification(
            bad, **self.kwargs)
        self.assertEqual(
            outcome.reason, "w39_quorum_decision_cid_mismatch")

    def test_quorum_topology_cid_mismatch_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, quorum_topology_cid="bogus", w39_cid="")
        outcome = verify_multi_host_disjoint_quorum_ratification(
            bad, **self.kwargs)
        self.assertEqual(
            outcome.reason, "w39_quorum_topology_cid_mismatch")

    def test_manifest_v9_cid_mismatch_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, manifest_v9_cid="bogus", w39_cid="")
        outcome = verify_multi_host_disjoint_quorum_ratification(
            bad, **self.kwargs)
        self.assertEqual(
            outcome.reason, "w39_manifest_v9_cid_mismatch")

    def test_outer_cid_mismatch_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(env, w39_cid="bogus")
        outcome = verify_multi_host_disjoint_quorum_ratification(
            bad, **self.kwargs)
        self.assertEqual(outcome.reason, "w39_outer_cid_mismatch")


class W39BankTests(unittest.TestCase):
    def test_trivial_w39_byte_for_w38(self) -> None:
        sweep = run_phase86_seed_sweep(
            bank="trivial_w39", n_eval=16,
            seeds=(11, 17, 23, 29, 31))
        self.assertTrue(
            sweep["all_byte_equivalent_w39_w38"],
            msg=f"trivial sweep is not byte-equivalent: {sweep}")
        self.assertEqual(sweep["max_overhead_w39_per_cell"], 0)
        self.assertEqual(
            sweep["min_delta_correctness_w39_w38"], 0.0)
        self.assertEqual(
            sweep["max_delta_correctness_w39_w38"], 0.0)
        self.assertEqual(
            sweep["min_delta_trust_precision_w39_w38"], 0.0)
        self.assertEqual(
            sweep["max_delta_trust_precision_w39_w38"], 0.0)

    def test_multi_host_colluded_consensus_trust_precision_gain(
            self) -> None:
        sweep = run_phase86_seed_sweep(
            bank="multi_host_colluded_consensus", n_eval=16,
            seeds=(11, 17, 23, 29, 31))
        self.assertGreaterEqual(
            sweep["min_delta_trust_precision_w39_w38"], 0.20,
            msg=f"colluded sweep gain too small: {sweep}")
        self.assertGreaterEqual(
            sweep["min_trust_precision_w39"], 0.95)
        self.assertLessEqual(
            sweep["max_overhead_w39_per_cell"], 1)

    def test_no_regression_quorum_agrees_no_regression(self) -> None:
        sweep = run_phase86_seed_sweep(
            bank="no_regression_quorum_agrees", n_eval=16,
            seeds=(11, 17, 23, 29, 31))
        self.assertGreaterEqual(
            sweep["min_delta_correctness_w39_w38"], 0.0,
            msg=f"no-regression sweep regressed: {sweep}")
        self.assertGreaterEqual(
            sweep["min_delta_trust_precision_w39_w38"], 0.0)

    def test_full_quorum_collusion_limitation_theorem(
            self) -> None:
        sweep = run_phase86_seed_sweep(
            bank="full_quorum_collusion", n_eval=16,
            seeds=(11, 17, 23, 29, 31))
        # W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP fires: W39 cannot
        # recover when every member probe is compromised in lock-step.
        self.assertEqual(
            sweep["min_delta_trust_precision_w39_w38"], 0.0,
            msg=("limitation theorem expected delta=0; "
                 f"got: {sweep}"))
        self.assertEqual(
            sweep["max_delta_trust_precision_w39_w38"], 0.0)

    def test_insufficient_quorum_preserves_w38(self) -> None:
        sweep = run_phase86_seed_sweep(
            bank="insufficient_quorum", n_eval=16,
            seeds=(11, 17, 23, 29, 31))
        self.assertEqual(
            sweep["min_delta_trust_precision_w39_w38"], 0.0)
        self.assertEqual(
            sweep["max_delta_trust_precision_w39_w38"], 0.0)


if __name__ == "__main__":
    unittest.main()
