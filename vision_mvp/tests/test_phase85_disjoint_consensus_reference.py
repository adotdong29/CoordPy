"""Phase 85 -- W38 disjoint cross-source consensus-reference tests."""

from __future__ import annotations

import dataclasses
import unittest

from vision_mvp.coordpy.team_coord import (
    ConsensusReferenceProbe,
    DisjointConsensusReferenceRatificationEnvelope,
    DisjointConsensusReferenceRegistry,
    DisjointTopologyError,
    W38_DISJOINT_CONSENSUS_SCHEMA_VERSION,
    W38_BRANCH_CONSENSUS_RATIFIED,
    W38_BRANCH_CONSENSUS_DIVERGENCE_ABSTAINED,
    W38_BRANCH_CONSENSUS_NO_REFERENCE,
    W38_BRANCH_CONSENSUS_NO_TRIGGER,
    W38_BRANCH_CONSENSUS_REFERENCE_WEAK,
    W38_BRANCH_CONSENSUS_REJECTED,
    select_disjoint_consensus_divergence,
    verify_disjoint_consensus_reference_ratification,
    _compute_w38_consensus_state_cid,
    _compute_w38_consensus_topology_cid,
    _compute_w38_consensus_probe_cid,
    _compute_w38_divergence_audit_cid,
    _compute_w38_manifest_v8_cid,
    _w38_top_set_divergence_score,
)
from vision_mvp.experiments.phase85_disjoint_consensus_reference import (
    _stable_schema_capsule,
    run_phase85_seed_sweep,
)


def _probe(top, hosts=("mac_consensus",),
           oracles=("disjoint_change_history",
                    "disjoint_oncall_notes"),
           strength=1.0, cell_idx=0):
    return ConsensusReferenceProbe(
        top_set=tuple(sorted(top)),
        consensus_host_ids=tuple(hosts),
        consensus_oracle_ids=tuple(oracles),
        consensus_strength=float(strength),
        cell_idx=int(cell_idx),
    )


class W38DivergenceScoreTests(unittest.TestCase):
    def test_identical_top_sets_score_zero(self) -> None:
        self.assertEqual(
            _w38_top_set_divergence_score(("a", "b"), ("b", "a")), 0.0)

    def test_disjoint_top_sets_score_one(self) -> None:
        self.assertEqual(
            _w38_top_set_divergence_score(("a", "b"), ("c", "d")), 1.0)

    def test_partial_overlap_score_in_between(self) -> None:
        score = _w38_top_set_divergence_score(("a", "b"), ("b", "c"))
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)


class W38SelectorTests(unittest.TestCase):
    def test_no_probe_returns_no_reference(self) -> None:
        branch, _, _ = select_disjoint_consensus_divergence(
            consensus_probe=None,
            w37_candidate_top_set=("a", "b"),
            w37_rerouted=True)
        self.assertEqual(branch, W38_BRANCH_CONSENSUS_NO_REFERENCE)

    def test_w37_not_rerouted_returns_no_trigger(self) -> None:
        branch, _, _ = select_disjoint_consensus_divergence(
            consensus_probe=_probe(("a", "b")),
            w37_candidate_top_set=("a", "b"),
            w37_rerouted=False)
        self.assertEqual(branch, W38_BRANCH_CONSENSUS_NO_TRIGGER)

    def test_weak_consensus_returns_reference_weak(self) -> None:
        branch, _, _ = select_disjoint_consensus_divergence(
            consensus_probe=_probe(("a", "b"), strength=0.0),
            w37_candidate_top_set=("a", "b"),
            w37_rerouted=True,
            consensus_strength_min=0.66)
        self.assertEqual(branch, W38_BRANCH_CONSENSUS_REFERENCE_WEAK)

    def test_agreement_returns_ratified(self) -> None:
        branch, score, top = select_disjoint_consensus_divergence(
            consensus_probe=_probe(("a", "b")),
            w37_candidate_top_set=("a", "b"),
            w37_rerouted=True)
        self.assertEqual(branch, W38_BRANCH_CONSENSUS_RATIFIED)
        self.assertEqual(score, 0.0)
        self.assertEqual(top, ("a", "b"))

    def test_disagreement_returns_divergence_abstained(self) -> None:
        branch, score, top = select_disjoint_consensus_divergence(
            consensus_probe=_probe(("a", "b")),
            w37_candidate_top_set=("c", "d"),
            w37_rerouted=True)
        self.assertEqual(branch, W38_BRANCH_CONSENSUS_DIVERGENCE_ABSTAINED)
        self.assertEqual(score, 1.0)


class W38VerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = _stable_schema_capsule()
        self.consensus_hosts = frozenset({"mac_consensus"})
        self.consensus_oracles = frozenset({
            "disjoint_change_history", "disjoint_oncall_notes"})
        self.trajectory_hosts = frozenset({
            "mac1", "mac_remote", "mac_shadow"})
        self.consensus_topology_cid = (
            _compute_w38_consensus_topology_cid(
                registered_consensus_host_ids=self.consensus_hosts,
                registered_consensus_oracle_ids=self.consensus_oracles,
                registered_trajectory_host_ids=self.trajectory_hosts))
        self.parent_w37_cid = "deadbeef" * 8
        self.kwargs = dict(
            registered_schema=self.schema,
            registered_parent_w37_cid=self.parent_w37_cid,
            registered_consensus_host_ids=self.consensus_hosts,
            registered_consensus_oracle_ids=self.consensus_oracles,
            registered_trajectory_host_ids=self.trajectory_hosts,
            registered_consensus_topology_cid=(
                self.consensus_topology_cid),
        )

    def _build_clean_envelope(self) -> (
            DisjointConsensusReferenceRatificationEnvelope):
        probe = _probe(("a", "b"))
        state_cid = _compute_w38_consensus_state_cid(probe=probe)
        probe_cid = _compute_w38_consensus_probe_cid(probe=probe)
        audit_cid = _compute_w38_divergence_audit_cid(
            projection_branch=W38_BRANCH_CONSENSUS_RATIFIED,
            w37_top_set=("a", "b"),
            consensus_top_set=("a", "b"),
            divergence_score=0.0,
            consensus_strength=1.0,
            consensus_strength_min=0.66,
            divergence_margin_min=0.10,
        )
        manifest_cid = _compute_w38_manifest_v8_cid(
            parent_w37_cid=self.parent_w37_cid,
            consensus_reference_state_cid=state_cid,
            divergence_audit_cid=audit_cid,
            consensus_topology_cid=self.consensus_topology_cid,
            consensus_probe_cid=probe_cid,
        )
        return DisjointConsensusReferenceRatificationEnvelope(
            schema_version=W38_DISJOINT_CONSENSUS_SCHEMA_VERSION,
            schema_cid=self.schema.cid,
            parent_w37_cid=self.parent_w37_cid,
            consensus_probe=probe,
            consensus_reference_state_cid=state_cid,
            consensus_topology_cid=self.consensus_topology_cid,
            consensus_probe_cid=probe_cid,
            projection_branch=W38_BRANCH_CONSENSUS_RATIFIED,
            w37_top_set=("a", "b"),
            consensus_top_set=("a", "b"),
            divergence_score=0.0,
            consensus_strength=1.0,
            consensus_strength_min=0.66,
            divergence_margin_min=0.10,
            divergence_audit_cid=audit_cid,
            manifest_v8_cid=manifest_cid,
            cell_index=0,
            wire_required=True,
        )

    def test_clean_envelope_passes(self) -> None:
        env = self._build_clean_envelope()
        outcome = verify_disjoint_consensus_reference_ratification(
            env, **self.kwargs)
        self.assertTrue(outcome.ok, msg=str(outcome.reason))
        self.assertEqual(outcome.reason, "ok")
        self.assertGreaterEqual(outcome.n_checks, 14)

    def test_empty_envelope_rejected(self) -> None:
        outcome = verify_disjoint_consensus_reference_ratification(
            None, **self.kwargs)
        self.assertEqual(outcome.reason, "empty_w38_envelope")

    def test_schema_version_unknown_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, schema_version="bogus", w38_cid="")
        outcome = verify_disjoint_consensus_reference_ratification(
            bad, **self.kwargs)
        self.assertEqual(outcome.reason, "w38_schema_version_unknown")

    def test_schema_cid_mismatch_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(env, schema_cid="bogus", w38_cid="")
        outcome = verify_disjoint_consensus_reference_ratification(
            bad, **self.kwargs)
        self.assertEqual(outcome.reason, "w38_schema_cid_mismatch")

    def test_parent_w37_cid_mismatch_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(env, parent_w37_cid="bogus", w38_cid="")
        outcome = verify_disjoint_consensus_reference_ratification(
            bad, **self.kwargs)
        self.assertEqual(outcome.reason, "w37_parent_cid_mismatch")

    def test_projection_branch_unknown_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, projection_branch="not_a_real_branch", w38_cid="")
        outcome = verify_disjoint_consensus_reference_ratification(
            bad, **self.kwargs)
        self.assertEqual(outcome.reason, "w38_projection_branch_unknown")

    def test_unregistered_consensus_host_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad_probe = ConsensusReferenceProbe(
            top_set=("a", "b"),
            consensus_host_ids=("mac_unregistered",),
            consensus_oracle_ids=("disjoint_change_history",),
            consensus_strength=1.0, cell_idx=0)
        bad = dataclasses.replace(
            env, consensus_probe=bad_probe, w38_cid="")
        outcome = verify_disjoint_consensus_reference_ratification(
            bad, **self.kwargs)
        self.assertEqual(outcome.reason,
                         "w38_consensus_host_unregistered")

    def test_unregistered_consensus_oracle_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad_probe = ConsensusReferenceProbe(
            top_set=("a", "b"),
            consensus_host_ids=("mac_consensus",),
            consensus_oracle_ids=("never_registered",),
            consensus_strength=1.0, cell_idx=0)
        bad = dataclasses.replace(
            env, consensus_probe=bad_probe, w38_cid="")
        outcome = verify_disjoint_consensus_reference_ratification(
            bad, **self.kwargs)
        self.assertEqual(outcome.reason,
                         "w38_consensus_oracle_unregistered")

    def test_disjoint_topology_violation_rejected(self) -> None:
        env = self._build_clean_envelope()
        kwargs = dict(self.kwargs)
        # Inject overlap: mac1 is in BOTH consensus and trajectory.
        kwargs["registered_consensus_host_ids"] = (
            frozenset({"mac_consensus", "mac1"}))
        outcome = verify_disjoint_consensus_reference_ratification(
            env, **kwargs)
        self.assertEqual(outcome.reason,
                         "w38_disjoint_topology_violation")

    def test_consensus_strength_out_of_range_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad_probe = ConsensusReferenceProbe(
            top_set=("a", "b"),
            consensus_host_ids=("mac_consensus",),
            consensus_oracle_ids=("disjoint_change_history",),
            consensus_strength=2.0, cell_idx=0)
        bad = dataclasses.replace(
            env, consensus_probe=bad_probe, w38_cid="")
        outcome = verify_disjoint_consensus_reference_ratification(
            bad, **self.kwargs)
        self.assertEqual(outcome.reason,
                         "w38_consensus_strength_out_of_range")

    def test_divergence_threshold_invalid_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, divergence_score=2.0, w38_cid="")
        outcome = verify_disjoint_consensus_reference_ratification(
            bad, **self.kwargs)
        self.assertEqual(outcome.reason,
                         "w38_divergence_threshold_invalid")

    def test_consensus_state_cid_mismatch_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, consensus_reference_state_cid="bogus", w38_cid="")
        outcome = verify_disjoint_consensus_reference_ratification(
            bad, **self.kwargs)
        self.assertEqual(outcome.reason,
                         "w38_consensus_state_cid_mismatch")

    def test_consensus_probe_cid_mismatch_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, consensus_probe_cid="bogus", w38_cid="")
        outcome = verify_disjoint_consensus_reference_ratification(
            bad, **self.kwargs)
        self.assertEqual(outcome.reason,
                         "w38_consensus_probe_cid_mismatch")

    def test_consensus_topology_cid_mismatch_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, consensus_topology_cid="bogus", w38_cid="")
        outcome = verify_disjoint_consensus_reference_ratification(
            bad, **self.kwargs)
        self.assertEqual(outcome.reason,
                         "w38_consensus_topology_cid_mismatch")

    def test_manifest_v8_cid_mismatch_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, manifest_v8_cid="bogus", w38_cid="")
        outcome = verify_disjoint_consensus_reference_ratification(
            bad, **self.kwargs)
        self.assertEqual(outcome.reason,
                         "w38_manifest_v8_cid_mismatch")

    def test_outer_cid_mismatch_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(env, w38_cid="bogus_outer")
        outcome = verify_disjoint_consensus_reference_ratification(
            bad, **self.kwargs)
        self.assertEqual(outcome.reason, "w38_outer_cid_mismatch")


class W38RegistryTests(unittest.TestCase):
    def test_overlapping_topology_raises(self) -> None:
        schema = _stable_schema_capsule()
        with self.assertRaises(DisjointTopologyError):
            DisjointConsensusReferenceRegistry(
                schema=schema,
                inner_w37_registry=None,
                registered_consensus_host_ids=frozenset({"mac1"}),
                registered_trajectory_host_ids=frozenset({"mac1"}),
            )

    def test_disjoint_topology_constructs(self) -> None:
        schema = _stable_schema_capsule()
        registry = DisjointConsensusReferenceRegistry(
            schema=schema,
            inner_w37_registry=None,
            registered_consensus_host_ids=frozenset({"mac_consensus"}),
            registered_trajectory_host_ids=frozenset(
                {"mac1", "mac_remote"}),
        )
        self.assertIsNotNone(registry.consensus_topology_cid)


class W38BankTests(unittest.TestCase):
    def test_trivial_w38_byte_for_w37(self) -> None:
        sweep = run_phase85_seed_sweep(
            bank="trivial_w38", n_eval=16,
            seeds=(11, 17, 23, 29, 31))
        self.assertTrue(sweep["all_byte_equivalent_w38_w37"],
                        msg=str(sweep))
        self.assertEqual(
            sweep["min_delta_correctness_w38_w37"], 0.0)
        self.assertEqual(
            sweep["max_overhead_w38_per_cell"], 0)

    def test_colluded_cross_host_trajectory_trust_precision_gain(
            self) -> None:
        sweep = run_phase85_seed_sweep(
            bank="colluded_cross_host_trajectory", n_eval=16,
            seeds=(11, 17, 23, 29, 31))
        self.assertGreaterEqual(
            sweep["min_delta_trust_precision_w38_w37"], 0.20,
            msg=str(sweep))
        self.assertEqual(
            sweep["min_trust_precision_w38"], 1.0,
            msg=str(sweep))
        self.assertLessEqual(
            sweep["max_overhead_w38_per_cell"], 1)

    def test_no_collusion_consensus_agrees_no_regression(self) -> None:
        sweep = run_phase85_seed_sweep(
            bank="no_collusion_consensus_agrees", n_eval=16,
            seeds=(11, 17, 23, 29, 31))
        self.assertGreaterEqual(
            sweep["min_delta_correctness_w38_w37"], 0.0)
        self.assertGreaterEqual(
            sweep["min_delta_trust_precision_w38_w37"], 0.0)

    def test_consensus_also_compromised_limitation_theorem(
            self) -> None:
        # W38-L-CONSENSUS-COLLUSION-CAP: when the disjoint consensus
        # reference is itself compromised in lock-step, W38 cannot
        # recover; the bank must show that trust precision does not
        # rise above W37's wrong-commit baseline.
        sweep = run_phase85_seed_sweep(
            bank="consensus_also_compromised", n_eval=16,
            seeds=(11, 17, 23, 29, 31))
        self.assertEqual(
            sweep["min_delta_trust_precision_w38_w37"], 0.0,
            msg=str(sweep))
        self.assertEqual(
            sweep["max_delta_trust_precision_w38_w37"], 0.0,
            msg=str(sweep))

    def test_no_consensus_reference_preserves_w37(self) -> None:
        sweep = run_phase85_seed_sweep(
            bank="no_consensus_reference", n_eval=16,
            seeds=(11, 17, 23, 29, 31))
        self.assertEqual(
            sweep["min_delta_correctness_w38_w37"], 0.0)
        self.assertEqual(
            sweep["min_delta_trust_precision_w38_w37"], 0.0)


if __name__ == "__main__":
    unittest.main()
