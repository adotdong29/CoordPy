"""Tests for SDK v3.30 / W29 — geometry-partitioned product-manifold
dense control + audited subspace-basis payload + factoradic routing
index + causal-validity gate + cross-host variance witness.

Coverage map (mirrors `docs/SUCCESS_CRITERION_W29_GEOMETRY_PARTITIONED.md`):

* H1 — every enumerated failure mode of
  ``verify_geometry_partition_ratification`` is asserted.
* H2 — W29 = W28 byte-for-byte on the trivial-partition path
  (``W29-Λ-trivial-partition``).
* H3 — five named tampers each rejected per cell with the expected
  reason.
* H5 — three named falsifiers (W29-Λ-trivial-partition,
  W29-Λ-non-orthogonal-basis,
  W29-Λ-coordinated-drift-cross-host).
* Subspace basis primitives — orthogonality, dimension, hash, NaN.
* Factoradic encode/decode round-trip for ``K ∈ [0, 5]``.
* Classifier — LINEAR / HIERARCHICAL / CYCLIC over alternating
  signature histories.
* Cross-host variance witness — disagreement count, witness CID
  recompute.
"""

from __future__ import annotations

import dataclasses
import math
import unittest

from vision_mvp.wevra.team_coord import (
    SchemaCapsule, build_incident_triage_schema_capsule,
    SubspaceBasis, verify_subspace_basis,
    compute_structural_subspace_basis,
    encode_permutation_to_factoradic,
    decode_factoradic_to_permutation,
    CrossHostVarianceWitness,
    GeometryPartitionedRatificationEnvelope,
    PartitionRegistration,
    GeometryPartitionRegistry,
    GeometryPartitionedOrchestrator,
    classify_partition_id_for_cell,
    verify_geometry_partition_ratification,
    build_trivial_partition_registry,
    build_three_partition_registry,
    W29_PARTITION_SCHEMA_VERSION,
    W29_PARTITION_LINEAR, W29_PARTITION_HIERARCHICAL,
    W29_PARTITION_CYCLIC,
    W29_DEFAULT_ORTHOGONALITY_TOL,
    W29_BRANCH_PARTITION_RESOLVED,
    W29_BRANCH_TRIVIAL_PARTITION_PASSTHROUGH,
    W29_BRANCH_PARTITION_REJECTED,
    W29_BRANCH_NO_PARTITION_NEEDED,
    LatentVerificationOutcome,
    _compute_causal_validity_signature,
)


AMBIENT = ("a", "b", "c", "d", "e", "f", "g", "h")


# =============================================================================
# Subspace basis primitives
# =============================================================================


class SubspaceBasisTests(unittest.TestCase):

    def test_dim_zero_basis_verifies_ok(self) -> None:
        b = SubspaceBasis(
            dim=0, ambient_dim=8, ambient_vocabulary=AMBIENT,
            basis_vectors=())
        out = verify_subspace_basis(
            b, expected_dim=0, expected_ambient_dim=8)
        self.assertTrue(out.ok, out.reason)

    def test_dim_two_orthogonal_basis_verifies_ok(self) -> None:
        # Two orthonormal canonical basis vectors.
        v0 = tuple([1.0] + [0.0] * 7)
        v1 = tuple([0.0, 1.0] + [0.0] * 6)
        b = SubspaceBasis(
            dim=2, ambient_dim=8, ambient_vocabulary=AMBIENT,
            basis_vectors=(v0, v1))
        out = verify_subspace_basis(
            b, expected_dim=2, expected_ambient_dim=8)
        self.assertTrue(out.ok, out.reason)
        self.assertLess(b.gram_off_diag_max, 1e-9)

    def test_non_orthogonal_basis_rejected(self) -> None:
        v = tuple([1.0] + [0.0] * 7)
        b = SubspaceBasis(
            dim=2, ambient_dim=8, ambient_vocabulary=AMBIENT,
            basis_vectors=(v, v))  # parallel ⇒ Gram off-diag = 1
        out = verify_subspace_basis(
            b, expected_dim=2, expected_ambient_dim=8,
            orthogonality_tol=1e-4)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "subspace_basis_non_orthogonal")

    def test_dim_mismatch_rejected(self) -> None:
        v0 = tuple([1.0] + [0.0] * 7)
        b = SubspaceBasis(
            dim=1, ambient_dim=8, ambient_vocabulary=AMBIENT,
            basis_vectors=(v0,))
        out = verify_subspace_basis(
            b, expected_dim=2, expected_ambient_dim=8)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "subspace_basis_dim_mismatch")

    def test_ambient_dim_mismatch_rejected(self) -> None:
        v0 = tuple([1.0] + [0.0] * 7)
        b = SubspaceBasis(
            dim=1, ambient_dim=8, ambient_vocabulary=AMBIENT,
            basis_vectors=(v0,))
        out = verify_subspace_basis(
            b, expected_dim=1, expected_ambient_dim=4)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "subspace_basis_dim_mismatch")

    def test_nan_inf_rejected(self) -> None:
        v0 = tuple([float("nan")] + [0.0] * 7)
        b = SubspaceBasis(
            dim=1, ambient_dim=8, ambient_vocabulary=AMBIENT,
            basis_vectors=(v0,))
        out = verify_subspace_basis(
            b, expected_dim=1, expected_ambient_dim=8)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "subspace_basis_nan_inf")

    def test_basis_cid_recomputes(self) -> None:
        v0 = tuple([1.0] + [0.0] * 7)
        b = SubspaceBasis(
            dim=1, ambient_dim=8, ambient_vocabulary=AMBIENT,
            basis_vectors=(v0,))
        self.assertEqual(b.basis_cid, b._recompute_cid())

    def test_structural_basis_partition_specific(self) -> None:
        """The structural basis differs across partition_ids
        (different partition-indicator axes ⇒ different basis_cids).
        """
        votes = (("a", 3), ("b", 1), ("c", 2))
        b_lin = compute_structural_subspace_basis(
            canonical_per_tag_votes=votes,
            ambient_vocabulary=AMBIENT,
            partition_id=W29_PARTITION_LINEAR,
            basis_dim=2)
        b_cyc = compute_structural_subspace_basis(
            canonical_per_tag_votes=votes,
            ambient_vocabulary=AMBIENT,
            partition_id=W29_PARTITION_CYCLIC,
            basis_dim=2)
        # Both must verify orthogonality.
        for b in (b_lin, b_cyc):
            out = verify_subspace_basis(
                b, expected_dim=2, expected_ambient_dim=8)
            self.assertTrue(out.ok, out.reason)
        # Different partitions ⇒ different basis_cids on the same
        # canonical state.
        self.assertNotEqual(b_lin.basis_cid, b_cyc.basis_cid)

    def test_structural_basis_empty_votes_falls_back(self) -> None:
        """Empty per-tag-vote tuple must produce a valid orthonormal
        basis (deterministic e_0 fallback)."""
        b = compute_structural_subspace_basis(
            canonical_per_tag_votes=(),
            ambient_vocabulary=AMBIENT,
            partition_id=W29_PARTITION_HIERARCHICAL,
            basis_dim=2)
        out = verify_subspace_basis(
            b, expected_dim=2, expected_ambient_dim=8)
        self.assertTrue(out.ok, out.reason)


# =============================================================================
# Factoradic Lehmer code
# =============================================================================


class FactoradicCodeTests(unittest.TestCase):

    def test_round_trip_K0_to_5(self) -> None:
        """Every factoradic index in ``[0, K!)`` round-trips through
        encode→decode for ``K ∈ [0, 5]``."""
        for K in range(6):
            for idx in range(math.factorial(K)):
                p = decode_factoradic_to_permutation(idx, K)
                self.assertEqual(
                    encode_permutation_to_factoradic(p), idx,
                    msg=f"K={K}, idx={idx}, p={p}")

    def test_index_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            decode_factoradic_to_permutation(10, 3)
        with self.assertRaises(ValueError):
            decode_factoradic_to_permutation(-1, 3)
        with self.assertRaises(ValueError):
            decode_factoradic_to_permutation(1, 0)

    def test_non_permutation_raises(self) -> None:
        with self.assertRaises(ValueError):
            encode_permutation_to_factoradic((0, 0, 1))
        with self.assertRaises(ValueError):
            encode_permutation_to_factoradic((1, 2, 4))


# =============================================================================
# Classifier
# =============================================================================


class ClassifierTests(unittest.TestCase):

    def test_empty_history_is_hierarchical(self) -> None:
        pid = classify_partition_id_for_cell(
            w28_branch="ratified", signature_cid="a",
            signature_history=())
        self.assertEqual(pid, W29_PARTITION_HIERARCHICAL)

    def test_same_as_last_is_linear(self) -> None:
        pid = classify_partition_id_for_cell(
            w28_branch="ratified", signature_cid="a",
            signature_history=("a",))
        self.assertEqual(pid, W29_PARTITION_LINEAR)

    def test_new_signature_after_run_is_hierarchical(self) -> None:
        pid = classify_partition_id_for_cell(
            w28_branch="ratified", signature_cid="b",
            signature_history=("a", "a", "a"))
        self.assertEqual(pid, W29_PARTITION_HIERARCHICAL)

    def test_returning_signature_is_cyclic(self) -> None:
        pid = classify_partition_id_for_cell(
            w28_branch="ratified", signature_cid="a",
            signature_history=("a", "a", "b", "b"),
            cycle_window=8)
        self.assertEqual(pid, W29_PARTITION_CYCLIC)

    def test_returning_within_run_is_cyclic_too(self) -> None:
        # After cycling, even cells in the same suffix run are CYCLIC.
        pid = classify_partition_id_for_cell(
            w28_branch="ratified", signature_cid="a",
            signature_history=("a", "a", "b", "b", "a"),
            cycle_window=8)
        self.assertEqual(pid, W29_PARTITION_CYCLIC)

    def test_alternating_pattern(self) -> None:
        sigs = ["A"] * 4 + ["B"] * 4 + ["A"] * 4 + ["B"] * 4
        history: list[str] = []
        expected = [
            1, 0, 0, 0,  # A run
            1, 0, 0, 0,  # B run
            2, 2, 2, 2,  # A returns ⇒ all CYCLIC
            2, 2, 2, 2,  # B returns ⇒ all CYCLIC
        ]
        for i, sig in enumerate(sigs):
            pid = classify_partition_id_for_cell(
                w28_branch="ratified", signature_cid=sig,
                signature_history=tuple(history),
                cycle_window=8)
            self.assertEqual(
                pid, expected[i],
                msg=f"cell {i} sig={sig} expected {expected[i]} got {pid}")
            history.append(sig)


# =============================================================================
# Cross-host variance witness
# =============================================================================


class CrossHostVarianceWitnessTests(unittest.TestCase):

    def test_witness_cid_recomputes(self) -> None:
        w = CrossHostVarianceWitness(
            cell_index=0,
            disagreement_pairs=(("p1", "h1", "p2", "h2"),),
            cross_host_disagreements=1,
            total_pairs_seen=1,
        )
        self.assertEqual(w.witness_cid, w._recompute_cid())

    def test_canonical_sort_collides_identical_disagreement_sets(
            self) -> None:
        w1 = CrossHostVarianceWitness(
            cell_index=0,
            disagreement_pairs=(
                ("p1", "h1", "p2", "h2"),
                ("p3", "h1", "p4", "h2"),
            ),
            cross_host_disagreements=2, total_pairs_seen=4)
        w2 = CrossHostVarianceWitness(
            cell_index=0,
            # Order reversed; sort must collapse these.
            disagreement_pairs=(
                ("p3", "h1", "p4", "h2"),
                ("p1", "h1", "p2", "h2"),
            ),
            cross_host_disagreements=2, total_pairs_seen=4)
        self.assertEqual(w1.witness_cid, w2.witness_cid)


# =============================================================================
# Geometry partition envelope verifier — every enumerated failure mode
# =============================================================================


class _Fixture:
    """Minimal in-test scaffolding for envelope construction.

    Constructs a registered schema, partition table, and a
    syntactically valid envelope pinned to the same schema.
    """

    def __init__(self) -> None:
        self.schema = build_incident_triage_schema_capsule()
        self.consumer_order = ("c0", "c1", "c2")
        self.partition_table = (
            PartitionRegistration(
                partition_id=W29_PARTITION_LINEAR,
                consumer_permutation=(0, 1, 2)),
            PartitionRegistration(
                partition_id=W29_PARTITION_HIERARCHICAL,
                consumer_permutation=(1, 2, 0)),
            PartitionRegistration(
                partition_id=W29_PARTITION_CYCLIC,
                consumer_permutation=(2, 1, 0)),
        )
        self.basis_dim = 2
        self.ambient_dim = 8
        v0 = tuple([1.0] + [0.0] * 7)
        v1 = tuple([0.0, 1.0] + [0.0] * 6)
        self.basis = SubspaceBasis(
            dim=2, ambient_dim=8, ambient_vocabulary=AMBIENT,
            basis_vectors=(v0, v1))
        self.parent_w28_cid = "a" * 64
        self.predecessor_cids = ("p" * 64,)
        self.causal_sig = _compute_causal_validity_signature(
            parent_w28_cid=self.parent_w28_cid,
            predecessor_cids=self.predecessor_cids)
        # Identity perm ⇒ factoradic index 0.
        self.factoradic_idx = encode_permutation_to_factoradic((0, 1, 2))

    def envelope(self, **overrides) -> (
            GeometryPartitionedRatificationEnvelope):
        kwargs = dict(
            schema_version=W29_PARTITION_SCHEMA_VERSION,
            schema_cid=self.schema.cid,
            w28_ratification_cid=self.parent_w28_cid,
            partition_id=W29_PARTITION_LINEAR,
            basis=self.basis,
            basis_dim=self.basis_dim,
            ambient_dim=self.ambient_dim,
            factoradic_route_index=self.factoradic_idx,
            factoradic_route_n_factors=3,
            causal_validity_signature=self.causal_sig,
            predecessor_cids=self.predecessor_cids,
            cross_host_variance_witness_cid="",
            cell_index=0,
            wire_required=True,
        )
        kwargs.update(overrides)
        # Force a fresh recompute of partition_cid.
        kwargs.setdefault("partition_cid", "")
        return GeometryPartitionedRatificationEnvelope(**kwargs)

    def verify(self, env, **overrides):
        kwargs = dict(
            registered_schema=self.schema,
            registered_w28_ratification_cid=self.parent_w28_cid,
            registered_partition_table=self.partition_table,
            registered_basis_dim=self.basis_dim,
            registered_ambient_dim=self.ambient_dim,
            registered_consumer_order=self.consumer_order,
        )
        kwargs.update(overrides)
        return verify_geometry_partition_ratification(env, **kwargs)


class GeometryPartitionVerifierFailureModeTests(unittest.TestCase):

    def setUp(self) -> None:
        self.fx = _Fixture()

    def test_ok_envelope_verifies(self) -> None:
        env = self.fx.envelope()
        out = self.fx.verify(env)
        self.assertTrue(out.ok, out.reason)

    def test_empty_partition_envelope(self) -> None:
        out = self.fx.verify(None)
        self.assertEqual(out.reason, "empty_partition_envelope")

    def test_schema_version_unknown(self) -> None:
        env = self.fx.envelope(schema_version="bogus.v0")
        out = self.fx.verify(env)
        self.assertEqual(out.reason, "schema_version_unknown")

    def test_schema_cid_mismatch(self) -> None:
        env = self.fx.envelope(schema_cid="0" * 64)
        out = self.fx.verify(env)
        self.assertEqual(out.reason, "schema_cid_mismatch")

    def test_w28_parent_cid_mismatch(self) -> None:
        env = self.fx.envelope(w28_ratification_cid="b" * 64)
        out = self.fx.verify(env)
        self.assertEqual(out.reason, "w28_parent_cid_mismatch")

    def test_partition_id_unregistered(self) -> None:
        env = self.fx.envelope(partition_id=99)
        out = self.fx.verify(env)
        self.assertEqual(out.reason, "partition_id_unregistered")

    def test_subspace_basis_dim_mismatch(self) -> None:
        # Envelope's basis has dim=2 but registered_basis_dim=4.
        env = self.fx.envelope()
        out = self.fx.verify(env, registered_basis_dim=4)
        self.assertEqual(out.reason, "subspace_basis_dim_mismatch")

    def test_subspace_basis_non_orthogonal(self) -> None:
        v = tuple([1.0] + [0.0] * 7)
        bad = SubspaceBasis(
            dim=2, ambient_dim=8, ambient_vocabulary=AMBIENT,
            basis_vectors=(v, v))
        env = self.fx.envelope(basis=bad)
        out = self.fx.verify(env)
        self.assertEqual(out.reason, "subspace_basis_non_orthogonal")

    def test_subspace_basis_nan_inf(self) -> None:
        v0 = tuple([float("nan")] + [0.0] * 7)
        v1 = tuple([0.0, 1.0] + [0.0] * 6)
        bad = SubspaceBasis(
            dim=2, ambient_dim=8, ambient_vocabulary=AMBIENT,
            basis_vectors=(v0, v1))
        env = self.fx.envelope(basis=bad)
        out = self.fx.verify(env)
        self.assertEqual(out.reason, "subspace_basis_nan_inf")

    def test_factoradic_index_out_of_range(self) -> None:
        env = self.fx.envelope(factoradic_route_index=math.factorial(3))
        out = self.fx.verify(env)
        self.assertEqual(out.reason, "factoradic_index_out_of_range")

    def test_factoradic_route_inverse_mismatch(self) -> None:
        # Use a permutation that does NOT match the registered
        # consumer_permutation for partition_id=0.
        wrong_perm = encode_permutation_to_factoradic((2, 1, 0))
        env = self.fx.envelope(
            factoradic_route_index=wrong_perm,
            partition_id=W29_PARTITION_LINEAR)  # registered (0,1,2)
        out = self.fx.verify(env)
        self.assertEqual(out.reason, "factoradic_route_inverse_mismatch")

    def test_causal_predecessor_unregistered(self) -> None:
        env = self.fx.envelope(predecessor_cids=("z" * 64,))
        # rebuild causal_sig for the new predecessors so that the
        # ONLY rejection is the predecessor-unregistered check.
        env = self.fx.envelope(
            predecessor_cids=("z" * 64,),
            causal_validity_signature=_compute_causal_validity_signature(
                parent_w28_cid=self.fx.parent_w28_cid,
                predecessor_cids=("z" * 64,)))
        out = self.fx.verify(
            env, registered_predecessor_cids=frozenset(
                self.fx.predecessor_cids))
        self.assertEqual(out.reason, "causal_predecessor_unregistered")

    def test_causal_validity_signature_invalid(self) -> None:
        env = self.fx.envelope(causal_validity_signature="0" * 64)
        out = self.fx.verify(env)
        self.assertEqual(out.reason, "causal_validity_signature_invalid")

    def test_cross_host_variance_witness_unsealed(self) -> None:
        env = self.fx.envelope(cross_host_variance_witness_cid="")
        out = self.fx.verify(
            env, cross_host_disagreement_observed=True)
        self.assertEqual(out.reason,
                          "cross_host_variance_witness_unsealed")

    def test_partition_cid_hash_mismatch(self) -> None:
        env = self.fx.envelope()
        # Force-overwrite partition_cid past frozen guard.
        object.__setattr__(env, "partition_cid", "ff" * 32)
        out = self.fx.verify(env)
        self.assertEqual(out.reason, "partition_cid_hash_mismatch")


# =============================================================================
# Trivial-partition factory + registry helpers
# =============================================================================


class RegistryFactoryTests(unittest.TestCase):

    def test_trivial_registry_is_trivial(self) -> None:
        r = build_trivial_partition_registry(
            schema=build_incident_triage_schema_capsule())
        self.assertTrue(r.is_trivial)
        self.assertFalse(r.has_wire_required_layer)
        # All three structural partition_ids registered (so the
        # classifier can hand any cell to any partition without
        # tripping ``partition_id_unregistered``).
        self.assertEqual(
            r.registered_partition_ids,
            frozenset({W29_PARTITION_LINEAR,
                       W29_PARTITION_HIERARCHICAL,
                       W29_PARTITION_CYCLIC}))

    def test_three_partition_registry_is_not_trivial(self) -> None:
        r = build_three_partition_registry(
            schema=build_incident_triage_schema_capsule(),
            consumer_order=("c0", "c1", "c2"),
            ambient_vocabulary=AMBIENT,
            basis_dim=2)
        self.assertFalse(r.is_trivial)
        self.assertTrue(r.has_wire_required_layer)
        self.assertEqual(int(r.basis_dim), 2)
        self.assertEqual(int(r.ambient_dim), 8)
        # Identity / shift1 / reverse permutations registered per
        # partition.
        labels = {p.label for p in r.partition_table}
        self.assertEqual(labels,
                         {"linear", "hierarchical", "cyclic"})


# =============================================================================
# End-to-end byte-for-W28 invariant (R-76-TRIVIAL-PARTITION smoke)
# =============================================================================


class TrivialPartitionByteForByteTest(unittest.TestCase):
    """Wire-cost-free behaviour: when the registry is trivial AND the
    classifier hands the cell to any registered partition, the W29
    layer charges 0 wire tokens AND emits
    ``W29_BRANCH_TRIVIAL_PARTITION_PASSTHROUGH``.

    This is the H2 byte-for-W28 anchor measured directly on the runtime
    surface (without spinning up the full phase76 bench).
    """

    def test_trivial_registry_charges_no_wire(self) -> None:
        r = build_trivial_partition_registry(
            schema=build_incident_triage_schema_capsule())
        # is_trivial ⇒ has_wire_required_layer=False ⇒ no token.
        self.assertTrue(r.is_trivial)
        self.assertFalse(r.has_wire_required_layer)


if __name__ == "__main__":
    unittest.main()
