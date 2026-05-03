"""Phase 87 -- W40 cross-host response-signature heterogeneity tests."""

from __future__ import annotations

import dataclasses
import unittest

from vision_mvp.wevra.team_coord import (
    ResponseSignatureProbe,
    MultiHostResponseHeterogeneityProbe,
    CrossHostResponseHeterogeneityRatificationEnvelope,
    CrossHostResponseHeterogeneityRegistry,
    DisjointTopologyError,
    MutuallyDisjointTopologyError,
    W40_RESPONSE_HETEROGENEITY_SCHEMA_VERSION,
    W40_BRANCH_RESPONSE_SIGNATURE_DIVERSE,
    W40_BRANCH_RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED,
    W40_BRANCH_RESPONSE_SIGNATURE_NO_REFERENCES,
    W40_BRANCH_RESPONSE_SIGNATURE_NO_TRIGGER,
    W40_BRANCH_RESPONSE_SIGNATURE_INSUFFICIENT,
    W40_BRANCH_RESPONSE_SIGNATURE_INCOMPLETE,
    W40_BRANCH_RESPONSE_SIGNATURE_REJECTED,
    W40_BRANCH_TRIVIAL_RESPONSE_SIGNATURE_PASSTHROUGH,
    select_cross_host_response_heterogeneity_decision,
    verify_cross_host_response_heterogeneity_ratification,
    _w40_canonical_token_bag,
    _w40_compute_response_signature_cid,
    _w40_pairwise_jaccard_divergence,
    _compute_w40_response_signature_state_cid,
    _compute_w40_response_signature_topology_cid,
    _compute_w40_response_heterogeneity_witness_cid,
    _compute_w40_response_signature_decision_cid,
    _compute_w40_response_signature_audit_cid,
    _compute_w40_manifest_v10_cid,
)
from vision_mvp.experiments.phase87_cross_host_response_heterogeneity import (
    _stable_schema_capsule,
    run_phase87_seed_sweep,
)


def _member(idx, host, oracle, text, cell_idx=0):
    bag = _w40_canonical_token_bag(text)
    return ResponseSignatureProbe(
        member_index=int(idx),
        host_ids=(host,),
        oracle_ids=(oracle,),
        response_token_bag=bag,
        response_signature_cid=_w40_compute_response_signature_cid(
            response_text=text),
        cell_idx=int(cell_idx),
    )


def _response_probe(member_texts,
                    hosts_per_pool=None,
                    oracles_per_pool=None,
                    diversity_min=0.20,
                    min_probes=2, cell_idx=0):
    if hosts_per_pool is None:
        hosts_per_pool = [
            ("mac_off_cluster_a",), ("mac_off_cluster_b",)]
    if oracles_per_pool is None:
        oracles_per_pool = [
            ("disjoint_quorum_oracle_a",),
            ("disjoint_quorum_oracle_b",)]
    members = []
    for k, text in enumerate(member_texts):
        members.append(_member(
            idx=k, host=hosts_per_pool[k][0],
            oracle=oracles_per_pool[k][0], text=text,
            cell_idx=cell_idx))
    return MultiHostResponseHeterogeneityProbe(
        member_probes=tuple(members),
        response_text_diversity_min=float(diversity_min),
        min_response_signature_probes=int(min_probes),
        cell_idx=int(cell_idx),
    )


class W40HelperTests(unittest.TestCase):
    def test_canonical_token_bag_lowercases_and_dedupes(self) -> None:
        bag = _w40_canonical_token_bag("Hello WORLD hello world!")
        self.assertEqual(bag, ("hello", "world"))

    def test_canonical_token_bag_handles_empty(self) -> None:
        self.assertEqual(_w40_canonical_token_bag(""), ())
        self.assertEqual(_w40_canonical_token_bag(None), ())

    def test_pairwise_jaccard_divergence_extremes(self) -> None:
        # Identical bags -> 0.0 (full collapse).
        self.assertEqual(
            _w40_pairwise_jaccard_divergence(("a", "b"), ("a", "b")),
            0.0)
        # Disjoint bags -> 1.0 (full diversity).
        self.assertEqual(
            _w40_pairwise_jaccard_divergence(("a", "b"), ("c", "d")),
            1.0)
        # Half-overlap.
        jac = _w40_pairwise_jaccard_divergence(
            ("a", "b"), ("b", "c"))
        # |inter| = 1, |union| = 3 -> 1 - 1/3 = 0.6667
        self.assertAlmostEqual(jac, 1.0 - 1.0 / 3.0, places=4)

    def test_response_signature_cid_is_deterministic(self) -> None:
        cid_1 = _w40_compute_response_signature_cid(
            response_text="Hello World")
        cid_2 = _w40_compute_response_signature_cid(
            response_text="hello world")
        # Canonicalisation maps both to the same canonical bag.
        self.assertEqual(cid_1, cid_2)

    def test_response_signature_cid_distinguishes_distinct_bags(
            self) -> None:
        cid_1 = _w40_compute_response_signature_cid(
            response_text="alpha bravo")
        cid_2 = _w40_compute_response_signature_cid(
            response_text="charlie delta")
        self.assertNotEqual(cid_1, cid_2)


class W40SelectorTests(unittest.TestCase):
    def test_no_probe_returns_no_references(self) -> None:
        branch, *_ = (
            select_cross_host_response_heterogeneity_decision(
                response_probe=None,
                w39_ratified=True))
        self.assertEqual(
            branch, W40_BRANCH_RESPONSE_SIGNATURE_NO_REFERENCES)

    def test_w39_not_ratified_returns_no_trigger(self) -> None:
        probe = _response_probe(
            ["alpha bravo", "charlie delta"])
        branch, *_ = (
            select_cross_host_response_heterogeneity_decision(
                response_probe=probe,
                w39_ratified=False))
        self.assertEqual(
            branch, W40_BRANCH_RESPONSE_SIGNATURE_NO_TRIGGER)

    def test_insufficient_probes_falls_through(self) -> None:
        probe = _response_probe(
            ["alpha bravo"],
            hosts_per_pool=[("mac_off_cluster_a",)],
            oracles_per_pool=[("disjoint_quorum_oracle_a",)],
            min_probes=2)
        branch, *_ = (
            select_cross_host_response_heterogeneity_decision(
                response_probe=probe,
                w39_ratified=True,
                min_response_signature_probes=2))
        self.assertEqual(
            branch, W40_BRANCH_RESPONSE_SIGNATURE_INSUFFICIENT)

    def test_diverse_responses_returns_diverse(self) -> None:
        probe = _response_probe(
            ["alpha bravo charlie",
             "delta echo foxtrot golf hotel"])
        branch, n_div, n_col, n_pairs, n_total, mean_jac, *_ = (
            select_cross_host_response_heterogeneity_decision(
                response_probe=probe,
                w39_ratified=True))
        self.assertEqual(
            branch, W40_BRANCH_RESPONSE_SIGNATURE_DIVERSE)
        self.assertEqual(n_div, 1)
        self.assertEqual(n_col, 0)
        self.assertEqual(n_pairs, 1)
        self.assertEqual(n_total, 2)
        self.assertGreater(mean_jac, 0.5)

    def test_collapsed_responses_returns_collapse_abstained(
            self) -> None:
        probe = _response_probe(
            ["identical wrong consensus answer",
             "identical wrong consensus answer"])
        branch, n_div, n_col, n_pairs, n_total, mean_jac, *_ = (
            select_cross_host_response_heterogeneity_decision(
                response_probe=probe,
                w39_ratified=True))
        self.assertEqual(
            branch,
            W40_BRANCH_RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED)
        self.assertEqual(n_div, 0)
        self.assertEqual(n_col, 1)
        self.assertEqual(mean_jac, 0.0)

    def test_incomplete_response_signature_falls_through(
            self) -> None:
        # Force a member with empty signature_cid via dataclass replace.
        probe = _response_probe(
            ["alpha bravo", "charlie delta"])
        bad_member = dataclasses.replace(
            probe.member_probes[0], response_signature_cid="")
        bad_probe = MultiHostResponseHeterogeneityProbe(
            member_probes=(
                bad_member, probe.member_probes[1]),
            response_text_diversity_min=(
                probe.response_text_diversity_min),
            min_response_signature_probes=(
                probe.min_response_signature_probes),
            cell_idx=probe.cell_idx,
        )
        branch, *_ = (
            select_cross_host_response_heterogeneity_decision(
                response_probe=bad_probe,
                w39_ratified=True))
        self.assertEqual(
            branch, W40_BRANCH_RESPONSE_SIGNATURE_INCOMPLETE)


class W40RegistryTests(unittest.TestCase):
    def test_overlapping_trajectory_topology_raises(self) -> None:
        schema = _stable_schema_capsule()
        with self.assertRaises(DisjointTopologyError):
            CrossHostResponseHeterogeneityRegistry(
                schema=schema,
                registered_member_pool_host_ids=(
                    frozenset({"mac1"}),),
                registered_member_pool_oracle_ids=(
                    frozenset({"o1"}),),
                registered_trajectory_host_ids=frozenset({"mac1"}),
            )

    def test_non_mutually_disjoint_pools_raise(self) -> None:
        schema = _stable_schema_capsule()
        with self.assertRaises(MutuallyDisjointTopologyError):
            CrossHostResponseHeterogeneityRegistry(
                schema=schema,
                registered_member_pool_host_ids=(
                    frozenset({"mac_a", "mac_b"}),
                    frozenset({"mac_b", "mac_c"})),
                registered_member_pool_oracle_ids=(
                    frozenset({"o1"}), frozenset({"o2"})),
                registered_trajectory_host_ids=frozenset(
                    {"mac_traj"}),
            )

    def test_disjoint_topology_constructs(self) -> None:
        schema = _stable_schema_capsule()
        reg = CrossHostResponseHeterogeneityRegistry(
            schema=schema,
            registered_member_pool_host_ids=(
                frozenset({"mac_off_a"}),
                frozenset({"mac_off_b"})),
            registered_member_pool_oracle_ids=(
                frozenset({"o1"}), frozenset({"o2"})),
            registered_trajectory_host_ids=frozenset(
                {"mac_traj"}),
        )
        self.assertFalse(reg.is_trivial)


class W40VerifierTests(unittest.TestCase):
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
        self.topology_cid = (
            _compute_w40_response_signature_topology_cid(
                registered_member_pool_host_ids=self.pool_hosts,
                registered_member_pool_oracle_ids=(
                    self.pool_oracles),
                registered_trajectory_host_ids=(
                    self.trajectory_hosts)))
        self.witness_cid = (
            _compute_w40_response_heterogeneity_witness_cid(
                registered_member_pool_host_ids=self.pool_hosts))
        self.parent_w39_cid = "f00d" * 16
        self.kwargs = dict(
            registered_schema=self.schema,
            registered_parent_w39_cid=self.parent_w39_cid,
            registered_member_pool_host_ids=self.pool_hosts,
            registered_member_pool_oracle_ids=self.pool_oracles,
            registered_trajectory_host_ids=self.trajectory_hosts,
            registered_response_signature_topology_cid=(
                self.topology_cid),
            registered_response_heterogeneity_witness_cid=(
                self.witness_cid),
        )

    def _build_clean_envelope(self) -> (
            CrossHostResponseHeterogeneityRatificationEnvelope):
        probe = _response_probe(
            ["alpha bravo charlie",
             "delta echo foxtrot golf hotel"])
        (branch, n_div, n_col, n_pairs, n_total, mean_jac,
         per_pair_jacs, per_pair_brs) = (
            select_cross_host_response_heterogeneity_decision(
                response_probe=probe,
                w39_ratified=True))
        state_cid = _compute_w40_response_signature_state_cid(
            probe=probe)
        decision_cid = (
            _compute_w40_response_signature_decision_cid(
                projection_branch=branch,
                n_diverse_pairs=int(n_div),
                n_collapse_pairs=int(n_col),
                n_pairs=int(n_pairs),
                n_total=int(n_total),
                mean_pairwise_jaccard=float(mean_jac),
                per_pair_jaccards=per_pair_jacs,
                per_pair_branches=per_pair_brs,
            ))
        audit_cid = _compute_w40_response_signature_audit_cid(
            projection_branch=branch,
            w39_decision_top_set=("a", "b"),
            n_diverse_pairs=int(n_div),
            n_collapse_pairs=int(n_col),
            n_pairs=int(n_pairs),
            n_total=int(n_total),
            mean_pairwise_jaccard=float(mean_jac),
            response_text_diversity_min=0.20,
            min_response_signature_probes=2,
        )
        manifest_cid = _compute_w40_manifest_v10_cid(
            parent_w39_cid=self.parent_w39_cid,
            response_signature_state_cid=state_cid,
            response_signature_audit_cid=audit_cid,
            response_signature_topology_cid=self.topology_cid,
            response_signature_decision_cid=decision_cid,
            response_heterogeneity_witness_cid=self.witness_cid,
        )
        return CrossHostResponseHeterogeneityRatificationEnvelope(
            schema_version=(
                W40_RESPONSE_HETEROGENEITY_SCHEMA_VERSION),
            schema_cid=self.schema.cid,
            parent_w39_cid=self.parent_w39_cid,
            response_probe=probe,
            response_signature_state_cid=state_cid,
            response_signature_topology_cid=self.topology_cid,
            response_signature_decision_cid=decision_cid,
            response_heterogeneity_witness_cid=self.witness_cid,
            projection_branch=branch,
            w39_decision_top_set=("a", "b"),
            n_diverse_pairs=int(n_div),
            n_collapse_pairs=int(n_col),
            n_pairs=int(n_pairs),
            n_total=int(n_total),
            mean_pairwise_jaccard=float(mean_jac),
            response_text_diversity_min=0.20,
            min_response_signature_probes=2,
            per_pair_jaccards=per_pair_jacs,
            per_pair_branches=per_pair_brs,
            response_signature_audit_cid=audit_cid,
            manifest_v10_cid=manifest_cid,
            cell_index=0,
            wire_required=True,
        )

    def test_clean_envelope_passes(self) -> None:
        env = self._build_clean_envelope()
        outcome = (
            verify_cross_host_response_heterogeneity_ratification(
                env, **self.kwargs))
        self.assertTrue(outcome.ok, msg=str(outcome.reason))
        self.assertEqual(outcome.reason, "ok")
        self.assertGreaterEqual(outcome.n_checks, 14)

    def test_empty_envelope_rejected(self) -> None:
        outcome = (
            verify_cross_host_response_heterogeneity_ratification(
                None, **self.kwargs))
        self.assertEqual(outcome.reason, "empty_w40_envelope")

    def test_schema_version_unknown_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, schema_version="bogus", w40_cid="")
        outcome = (
            verify_cross_host_response_heterogeneity_ratification(
                bad, **self.kwargs))
        self.assertEqual(
            outcome.reason, "w40_schema_version_unknown")

    def test_schema_cid_mismatch_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, schema_cid="bogus", w40_cid="")
        outcome = (
            verify_cross_host_response_heterogeneity_ratification(
                bad, **self.kwargs))
        self.assertEqual(outcome.reason, "w40_schema_cid_mismatch")

    def test_parent_w39_cid_mismatch_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, parent_w39_cid="bogus", w40_cid="")
        outcome = (
            verify_cross_host_response_heterogeneity_ratification(
                bad, **self.kwargs))
        self.assertEqual(outcome.reason, "w39_parent_cid_mismatch")

    def test_projection_branch_unknown_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, projection_branch="bogus", w40_cid="")
        outcome = (
            verify_cross_host_response_heterogeneity_ratification(
                bad, **self.kwargs))
        self.assertEqual(
            outcome.reason, "w40_projection_branch_unknown")

    def test_unregistered_response_host_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad_probe = _response_probe(
            ["alpha bravo charlie", "delta echo foxtrot"],
            hosts_per_pool=[
                ("mac_unknown",), ("mac_off_cluster_b",)])
        bad = dataclasses.replace(
            env, response_probe=bad_probe,
            response_signature_state_cid=(
                _compute_w40_response_signature_state_cid(
                    probe=bad_probe)),
            w40_cid="")
        outcome = (
            verify_cross_host_response_heterogeneity_ratification(
                bad, **self.kwargs))
        self.assertEqual(
            outcome.reason,
            "w40_response_probe_unregistered_host")

    def test_unregistered_response_oracle_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad_probe = _response_probe(
            ["alpha bravo charlie", "delta echo foxtrot"],
            oracles_per_pool=[
                ("oracle_unknown",),
                ("disjoint_quorum_oracle_b",)])
        bad = dataclasses.replace(
            env, response_probe=bad_probe,
            response_signature_state_cid=(
                _compute_w40_response_signature_state_cid(
                    probe=bad_probe)),
            w40_cid="")
        outcome = (
            verify_cross_host_response_heterogeneity_ratification(
                bad, **self.kwargs))
        self.assertEqual(
            outcome.reason,
            "w40_response_probe_unregistered_oracle")

    def test_response_disjoint_topology_violation_rejected(
            self) -> None:
        env = self._build_clean_envelope()
        bad_kwargs = dict(self.kwargs)
        bad_kwargs["registered_trajectory_host_ids"] = frozenset(
            {"mac_off_cluster_a", "mac1"})
        outcome = (
            verify_cross_host_response_heterogeneity_ratification(
                env, **bad_kwargs))
        self.assertEqual(
            outcome.reason,
            "w40_response_disjoint_topology_violation")

    def test_response_mutual_disjointness_violation_rejected(
            self) -> None:
        env = self._build_clean_envelope()
        overlap_pools = (
            frozenset({"mac_off_cluster_a", "mac_shared"}),
            frozenset({"mac_off_cluster_b", "mac_shared"}),
        )
        bad_kwargs = dict(self.kwargs)
        bad_kwargs["registered_member_pool_host_ids"] = overlap_pools
        outcome = (
            verify_cross_host_response_heterogeneity_ratification(
                env, **bad_kwargs))
        self.assertEqual(
            outcome.reason,
            "w40_response_mutual_disjointness_violation")

    def test_response_thresholds_invalid_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, response_text_diversity_min=2.5, w40_cid="")
        outcome = (
            verify_cross_host_response_heterogeneity_ratification(
                bad, **self.kwargs))
        self.assertEqual(
            outcome.reason, "w40_response_thresholds_invalid")

    def test_response_state_cid_mismatch_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, response_signature_state_cid="bogus", w40_cid="")
        outcome = (
            verify_cross_host_response_heterogeneity_ratification(
                bad, **self.kwargs))
        self.assertEqual(
            outcome.reason, "w40_response_state_cid_mismatch")

    def test_response_decision_cid_mismatch_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, response_signature_decision_cid="bogus",
            w40_cid="")
        outcome = (
            verify_cross_host_response_heterogeneity_ratification(
                bad, **self.kwargs))
        self.assertEqual(
            outcome.reason, "w40_response_decision_cid_mismatch")

    def test_response_topology_cid_mismatch_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, response_signature_topology_cid="bogus",
            w40_cid="")
        outcome = (
            verify_cross_host_response_heterogeneity_ratification(
                bad, **self.kwargs))
        self.assertEqual(
            outcome.reason, "w40_response_topology_cid_mismatch")

    def test_response_heterogeneity_witness_cid_mismatch_rejected(
            self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, response_heterogeneity_witness_cid="bogus",
            w40_cid="")
        outcome = (
            verify_cross_host_response_heterogeneity_ratification(
                bad, **self.kwargs))
        # The mismatch is bucketed under the named mode for
        # mutual-disjointness violation defense-in-depth.
        self.assertEqual(
            outcome.reason,
            "w40_response_mutual_disjointness_violation")

    def test_manifest_v10_cid_mismatch_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(
            env, manifest_v10_cid="bogus", w40_cid="")
        outcome = (
            verify_cross_host_response_heterogeneity_ratification(
                bad, **self.kwargs))
        self.assertEqual(
            outcome.reason, "w40_manifest_v10_cid_mismatch")

    def test_outer_cid_mismatch_rejected(self) -> None:
        env = self._build_clean_envelope()
        bad = dataclasses.replace(env, w40_cid="bogus")
        outcome = (
            verify_cross_host_response_heterogeneity_ratification(
                bad, **self.kwargs))
        self.assertEqual(outcome.reason, "w40_outer_cid_mismatch")


class W40BankTests(unittest.TestCase):
    def test_trivial_w40_byte_for_w39(self) -> None:
        sweep = run_phase87_seed_sweep(
            bank="trivial_w40", n_eval=16,
            seeds=(11, 17, 23, 29, 31))
        self.assertTrue(
            sweep["all_byte_equivalent_w40_w39"],
            msg=f"trivial sweep is not byte-equivalent: {sweep}")
        self.assertEqual(sweep["max_overhead_w40_per_cell"], 0)
        self.assertEqual(
            sweep["min_delta_correctness_w40_w39"], 0.0)
        self.assertEqual(
            sweep["max_delta_correctness_w40_w39"], 0.0)
        self.assertEqual(
            sweep["min_delta_trust_precision_w40_w39"], 0.0)
        self.assertEqual(
            sweep["max_delta_trust_precision_w40_w39"], 0.0)

    def test_response_signature_collapse_trust_precision_gain(
            self) -> None:
        sweep = run_phase87_seed_sweep(
            bank="response_signature_collapse", n_eval=16,
            seeds=(11, 17, 23, 29, 31))
        self.assertGreaterEqual(
            sweep["min_delta_trust_precision_w40_w39"], 0.20,
            msg=f"collapse sweep gain too small: {sweep}")
        self.assertGreaterEqual(
            sweep["min_trust_precision_w40"], 0.95)
        self.assertLessEqual(
            sweep["max_overhead_w40_per_cell"], 1)

    def test_no_regression_diverse_agrees_no_regression(
            self) -> None:
        sweep = run_phase87_seed_sweep(
            bank="no_regression_diverse_agrees", n_eval=16,
            seeds=(11, 17, 23, 29, 31))
        self.assertGreaterEqual(
            sweep["min_delta_correctness_w40_w39"], 0.0,
            msg=f"no-regression sweep regressed: {sweep}")
        self.assertGreaterEqual(
            sweep["min_delta_trust_precision_w40_w39"], 0.0)

    def test_coordinated_diverse_response_limitation_theorem(
            self) -> None:
        sweep = run_phase87_seed_sweep(
            bank="coordinated_diverse_response", n_eval=16,
            seeds=(11, 17, 23, 29, 31))
        # W40-L-COORDINATED-DIVERSE-RESPONSE-CAP fires: W40 cannot
        # recover when the adversary diversifies response bytes
        # while holding the wrong top_set in lock-step.
        self.assertEqual(
            sweep["min_delta_trust_precision_w40_w39"], 0.0,
            msg=("limitation theorem expected delta=0; "
                 f"got: {sweep}"))
        self.assertEqual(
            sweep["max_delta_trust_precision_w40_w39"], 0.0)

    def test_insufficient_response_signature_preserves_w39(
            self) -> None:
        sweep = run_phase87_seed_sweep(
            bank="insufficient_response_signature", n_eval=16,
            seeds=(11, 17, 23, 29, 31))
        self.assertEqual(
            sweep["min_delta_trust_precision_w40_w39"], 0.0)
        self.assertEqual(
            sweep["max_delta_trust_precision_w40_w39"], 0.0)


if __name__ == "__main__":
    unittest.main()
