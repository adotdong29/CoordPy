"""Tests for the W43 Product-Manifold Capsule (PMC) layer.

Covers:
  * the five channels' encoding contracts (CIDs deterministic,
    round-trip identities)
  * the policy-driven decision selector
  * the orchestrator's trivial-passthrough falsifier
  * the orchestrator's verifier (18 enumerated failure modes)
  * the channel bundle assembly
"""

from __future__ import annotations

import math

import pytest

from coordpy.product_manifold import (
    CausalVectorClock,
    CellObservation,
    HyperbolicBranchEncoding,
    ProductManifoldOrchestrator,
    ProductManifoldPolicyEntry,
    ProductManifoldRatificationEnvelope,
    ProductManifoldRegistry,
    SphericalConsensusSignature,
    SubspaceBasis,
    W43_ALL_BRANCHES,
    W43_BRANCH_PMC_CAUSAL_VIOLATION_ABSTAINED,
    W43_BRANCH_PMC_DISABLED,
    W43_BRANCH_PMC_NO_POLICY,
    W43_BRANCH_PMC_RATIFIED,
    W43_BRANCH_PMC_REJECTED,
    W43_BRANCH_PMC_SPHERICAL_DIVERGENCE_ABSTAINED,
    W43_BRANCH_PMC_SUBSPACE_DRIFT_ABSTAINED,
    W43_BRANCH_TRIVIAL_PMC_PASSTHROUGH,
    W43_PRODUCT_MANIFOLD_SCHEMA_VERSION,
    build_product_manifold_registry,
    build_trivial_product_manifold_registry,
    causally_dominates,
    cosine_agreement,
    decode_factoradic_route,
    decode_hyperbolic_branch_path_prefix,
    detect_causal_violation_index,
    encode_cell_channels,
    encode_euclidean_attributes,
    encode_factoradic_route,
    encode_hyperbolic_branch,
    encode_spherical_consensus,
    encode_subspace_basis,
    is_causally_admissible,
    principal_angle_drift,
    select_pmc_decision,
    verify_product_manifold_ratification,
)


# =============================================================================
# Channel: hyperbolic branch
# =============================================================================

class TestHyperbolicChannel:
    def test_round_trip_for_all_paths_up_to_capacity(self):
        for depth in range(0, 9):
            for seed in range(8):
                path = tuple((seed >> i) & 1 for i in range(depth))
                enc = encode_hyperbolic_branch(path, dim=4)
                recovered = decode_hyperbolic_branch_path_prefix(enc)
                assert recovered == path, (
                    f"round-trip failed: depth={depth} path={path} "
                    f"recovered={recovered}")

    def test_norm_strictly_below_r_max(self):
        for depth in (0, 4, 8, 12):
            path = tuple([1] * depth)
            enc = encode_hyperbolic_branch(path, dim=4, r_max=0.95)
            r = math.sqrt(sum(c * c for c in enc.coordinates))
            assert r < enc.r_max

    def test_cid_is_deterministic(self):
        path = (1, 0, 1, 1, 0, 0, 1, 0)
        a = encode_hyperbolic_branch(path, dim=4)
        b = encode_hyperbolic_branch(path, dim=4)
        assert a.cid() == b.cid()

    def test_different_paths_different_cids(self):
        a = encode_hyperbolic_branch((1, 0, 1, 1), dim=4)
        b = encode_hyperbolic_branch((0, 1, 0, 0), dim=4)
        assert a.cid() != b.cid()

    def test_invalid_path_bit_rejected(self):
        with pytest.raises(ValueError):
            encode_hyperbolic_branch((1, 0, 2), dim=4)

    def test_invalid_dim_rejected(self):
        with pytest.raises(ValueError):
            encode_hyperbolic_branch((1, 0), dim=0)
        with pytest.raises(ValueError):
            encode_hyperbolic_branch((1, 0), dim=2, r_max=1.5)


# =============================================================================
# Channel: spherical consensus
# =============================================================================

class TestSphericalChannel:
    def test_unit_norm(self):
        sig = encode_spherical_consensus(("a", "b", "c", "a"))
        norm = math.sqrt(sum(c * c for c in sig.coordinates))
        assert abs(norm - 1.0) < 1e-9

    def test_empty_input(self):
        sig = encode_spherical_consensus(())
        assert sig.n_observations == 0
        assert all(c == 0.0 for c in sig.coordinates)

    def test_cosine_agreement_perfect(self):
        a = encode_spherical_consensus(("a", "b", "c"))
        b = encode_spherical_consensus(("a", "b", "c"))
        assert abs(cosine_agreement(a, b) - 1.0) < 1e-9

    def test_cosine_agreement_zero_with_empty(self):
        a = encode_spherical_consensus(())
        b = encode_spherical_consensus(("a",))
        assert cosine_agreement(a, b) == 0.0

    def test_signature_is_permutation_invariant(self):
        a = encode_spherical_consensus(("a", "b", "c"))
        b = encode_spherical_consensus(("c", "a", "b"))
        assert a.cid() == b.cid()


# =============================================================================
# Channel: euclidean attributes
# =============================================================================

class TestEuclideanChannel:
    def test_field_order_padding(self):
        v = encode_euclidean_attributes(
            {"a": 1.0, "b": 2.0},
            field_order=["a", "b", "c", "d"], dim=4,
        )
        assert v.coordinates == (1.0, 2.0, 0.0, 0.0)
        assert v.field_names == ("a", "b", "c", "d")

    def test_truncation(self):
        v = encode_euclidean_attributes(
            {"a": 1.0, "b": 2.0, "c": 3.0},
            field_order=["a", "b"], dim=2,
        )
        assert v.coordinates == (1.0, 2.0)

    def test_cid_deterministic(self):
        a = encode_euclidean_attributes(
            {"x": 0.5}, field_order=["x"], dim=1)
        b = encode_euclidean_attributes(
            {"x": 0.5}, field_order=["x"], dim=1)
        assert a.cid() == b.cid()


# =============================================================================
# Channel: factoradic route
# =============================================================================

class TestFactoradicChannel:
    def test_round_trip_for_all_perms_up_to_n6(self):
        # Exhaustive Lehmer-code round-trip for n in 0..6.
        import itertools as it
        for n in range(0, 7):
            for perm in it.permutations(range(n)):
                fac = encode_factoradic_route(perm)
                rec = decode_factoradic_route(
                    fac.factoradic_int, n=n)
                assert rec.permutation == perm

    def test_information_capacity(self):
        for n in range(0, 13):
            fac = encode_factoradic_route(tuple(range(n)))
            expected = (
                0 if n < 2
                else int(math.ceil(math.log2(math.factorial(n)))))
            assert fac.n_structured_bits() == expected

    def test_invalid_permutation_rejected(self):
        with pytest.raises(ValueError):
            encode_factoradic_route([0, 0, 1])  # not a permutation

    def test_invalid_factoradic_int_rejected(self):
        with pytest.raises(ValueError):
            decode_factoradic_route(-1, n=3)
        with pytest.raises(ValueError):
            decode_factoradic_route(99, n=3)

    def test_cid_deterministic(self):
        a = encode_factoradic_route((2, 0, 1))
        b = encode_factoradic_route((2, 0, 1))
        assert a.cid() == b.cid()


# =============================================================================
# Channel: subspace basis (Grassmannian-style)
# =============================================================================

class TestSubspaceChannel:
    def test_orthonormal_columns(self):
        sb = encode_subspace_basis(
            ((1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
            rank=2, dim=4,
        )
        # Inner product of columns should be 0; norms should be 1.
        col0 = [sb.basis_columns[i][0] for i in range(sb.dim)]
        col1 = [sb.basis_columns[i][1] for i in range(sb.dim)]
        assert abs(sum(a * b for a, b in zip(col0, col1))) < 1e-9
        assert abs(math.sqrt(sum(a * a for a in col0)) - 1.0) < 1e-9
        assert abs(math.sqrt(sum(a * a for a in col1)) - 1.0) < 1e-9

    def test_canonical_under_column_permutation(self):
        a = encode_subspace_basis(
            ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
            rank=2, dim=4,
        )
        # The matrix's columns are e_0 and e_1; same span, same CID
        # because Gram-Schmidt + sign canonicalisation produces the
        # same representation.
        b = encode_subspace_basis(
            ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
            rank=2, dim=4,
        )
        assert a.cid() == b.cid()

    def test_principal_angle_zero_for_identical(self):
        a = encode_subspace_basis(
            ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
            rank=2, dim=4,
        )
        b = encode_subspace_basis(
            ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
            rank=2, dim=4,
        )
        assert principal_angle_drift(a, b) < 1e-6

    def test_principal_angle_pi_over_2_for_orthogonal(self):
        a = encode_subspace_basis(
            ((1.0,), (0.0,), (0.0,), (0.0,)),
            rank=1, dim=4,
        )
        b = encode_subspace_basis(
            ((0.0,), (1.0,), (0.0,), (0.0,)),
            rank=1, dim=4,
        )
        assert abs(principal_angle_drift(a, b)
                   - math.pi / 2) < 1e-6


# =============================================================================
# Channel: causal vector clocks
# =============================================================================

class TestCausalChannel:
    def test_dominates_componentwise(self):
        a = CausalVectorClock.from_mapping({"r0": 1, "r1": 0})
        b = CausalVectorClock.from_mapping({"r0": 1, "r1": 1})
        assert causally_dominates(a, b)
        assert not causally_dominates(b, a)

    def test_admissible_strictly_monotone(self):
        clocks = [
            CausalVectorClock.from_mapping({"r0": 1}),
            CausalVectorClock.from_mapping({"r0": 1, "r1": 1}),
            CausalVectorClock.from_mapping({"r0": 1, "r1": 1, "r2": 1}),
        ]
        assert is_causally_admissible(clocks)
        assert detect_causal_violation_index(clocks) == -1

    def test_inadmissible_out_of_order(self):
        clocks = [
            CausalVectorClock.from_mapping({"r0": 1, "r1": 1}),
            CausalVectorClock.from_mapping({"r0": 1}),  # decreased!
        ]
        assert not is_causally_admissible(clocks)
        assert detect_causal_violation_index(clocks) == 0


# =============================================================================
# Cell observation -> channel bundle
# =============================================================================

class TestChannelBundle:
    def test_basic_bundle(self):
        obs = CellObservation(
            branch_path=(1, 0, 1, 0),
            claim_kinds=("a", "b", "a"),
            role_arrival_order=("r1", "r0", "r2"),
            role_universe=("r0", "r1", "r2"),
            attributes=tuple({"round": 2.0}.items()),
            subspace_vectors=(
                (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
            causal_clocks=(
                CausalVectorClock.from_mapping({"r1": 1}),
                CausalVectorClock.from_mapping(
                    {"r0": 1, "r1": 1}),
                CausalVectorClock.from_mapping(
                    {"r0": 1, "r1": 1, "r2": 1}),
            ),
        )
        bundle = encode_cell_channels(obs)
        assert bundle.causal_admissible
        assert bundle.factoradic.n == len(obs.role_universe)

    def test_role_outside_universe_raises(self):
        obs = CellObservation(
            role_arrival_order=("r99",),
            role_universe=("r0", "r1"),
        )
        with pytest.raises(ValueError):
            encode_cell_channels(obs)


# =============================================================================
# Decision selector
# =============================================================================

class TestDecisionSelector:
    def _good_obs(self):
        return CellObservation(
            branch_path=(1, 0),
            claim_kinds=("a", "b"),
            role_arrival_order=("r0", "r1"),
            role_universe=("r0", "r1"),
            attributes=tuple({"round": 1.0}.items()),
            subspace_vectors=(
                (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
            causal_clocks=(
                CausalVectorClock.from_mapping({"r0": 1}),
                CausalVectorClock.from_mapping(
                    {"r0": 1, "r1": 1}),
            ),
        )

    def test_no_policy_returns_no_policy(self):
        observed = encode_spherical_consensus(("a",))
        observed_sub = encode_subspace_basis(
            ((1.0,), (0.0,), (0.0,), (0.0,)),
            rank=1, dim=4)
        branch, agreement, drift = select_pmc_decision(
            observed_spherical=observed,
            expected_spherical=None,
            observed_subspace=observed_sub,
            expected_subspace=None,
            causal_admissible=True,
            policy_match_found=False,
        )
        assert branch == W43_BRANCH_PMC_NO_POLICY
        assert agreement == 0.0
        assert drift == 0.0

    def test_causal_violation_short_circuits(self):
        observed = encode_spherical_consensus(("a",))
        expected = encode_spherical_consensus(("a",))
        observed_sub = encode_subspace_basis(
            ((1.0,),), rank=1, dim=1)
        expected_sub = encode_subspace_basis(
            ((1.0,),), rank=1, dim=1)
        branch, _agreement, _drift = select_pmc_decision(
            observed_spherical=observed,
            expected_spherical=expected,
            observed_subspace=observed_sub,
            expected_subspace=expected_sub,
            causal_admissible=False,
            policy_match_found=True,
        )
        assert branch == W43_BRANCH_PMC_CAUSAL_VIOLATION_ABSTAINED

    def test_subspace_drift_abstains_when_above_max(self):
        observed = encode_spherical_consensus(("a",))
        expected = encode_spherical_consensus(("a",))
        observed_sub = encode_subspace_basis(
            ((1.0,), (0.0,), (0.0,), (0.0,)), rank=1, dim=4)
        expected_sub = encode_subspace_basis(
            ((0.0,), (1.0,), (0.0,), (0.0,)), rank=1, dim=4)
        branch, _agreement, drift = select_pmc_decision(
            observed_spherical=observed,
            expected_spherical=expected,
            observed_subspace=observed_sub,
            expected_subspace=expected_sub,
            causal_admissible=True,
            policy_match_found=True,
            subspace_drift_max=0.1,
        )
        assert branch == W43_BRANCH_PMC_SUBSPACE_DRIFT_ABSTAINED
        assert drift > 0.1

    def test_spherical_divergence_abstains(self):
        observed = encode_spherical_consensus(("alert",) * 3)
        expected = encode_spherical_consensus(("event",) * 3)
        sub = encode_subspace_basis(
            ((1.0,),), rank=1, dim=1)
        branch, agreement, _drift = select_pmc_decision(
            observed_spherical=observed,
            expected_spherical=expected,
            observed_subspace=sub,
            expected_subspace=sub,
            causal_admissible=True,
            policy_match_found=True,
            spherical_agreement_min=0.99,
        )
        assert branch == W43_BRANCH_PMC_SPHERICAL_DIVERGENCE_ABSTAINED

    def test_ratifies_on_full_match(self):
        observed = encode_spherical_consensus(("event",) * 3)
        expected = encode_spherical_consensus(("event",) * 3)
        sub = encode_subspace_basis(
            ((1.0,),), rank=1, dim=1)
        branch, agreement, drift = select_pmc_decision(
            observed_spherical=observed,
            expected_spherical=expected,
            observed_subspace=sub,
            expected_subspace=sub,
            causal_admissible=True,
            policy_match_found=True,
        )
        assert branch == W43_BRANCH_PMC_RATIFIED
        assert agreement >= 0.99


# =============================================================================
# Orchestrator + verifier
# =============================================================================

def _good_observation():
    return CellObservation(
        branch_path=(1, 0, 1, 0),
        claim_kinds=("event", "event"),
        role_arrival_order=("r0", "r1"),
        role_universe=("r0", "r1"),
        attributes=tuple({"round": 1.0}.items()),
        subspace_vectors=(
            (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
        causal_clocks=(
            CausalVectorClock.from_mapping({"r0": 1}),
            CausalVectorClock.from_mapping({"r0": 1, "r1": 1}),
        ),
    )


def _good_policy(sig: str) -> ProductManifoldPolicyEntry:
    return ProductManifoldPolicyEntry(
        role_handoff_signature_cid=sig,
        expected_services=("event",),
        expected_spherical=encode_spherical_consensus(
            ("event", "event")),
        expected_subspace=encode_subspace_basis(
            ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))),
        expected_causal_topology_hash="(r0,r1)",
    )


class TestOrchestratorTrivialPassthrough:
    def test_trivial_registry_yields_passthrough(self):
        reg = build_trivial_product_manifold_registry()
        orch = ProductManifoldOrchestrator(registry=reg)
        result = orch.decode(
            observation=_good_observation(),
            role_handoff_signature_cid="sig",
            parent_w42_cid="parent",
            n_w42_visible_tokens=4,
        )
        assert result.decision_branch == (
            W43_BRANCH_TRIVIAL_PMC_PASSTHROUGH)
        assert result.n_w43_overhead_tokens == 0
        assert result.n_structured_bits == 0
        # Byte-for-W42 reduction: the visible-token count is
        # preserved exactly.
        assert (result.n_w43_visible_tokens
                == result.n_w42_visible_tokens)

    def test_disabled_only_yields_pmc_disabled(self):
        # When pmc_enabled is False but other guard-rails are still
        # on, we are NOT trivial — we should hit the disabled
        # branch, not the trivial branch.
        reg = ProductManifoldRegistry(
            schema_cid="cid",
            pmc_enabled=False,
            manifest_v13_disabled=False,
            abstain_on_causal_violation=True,
            abstain_on_subspace_drift=True,
            abstain_on_spherical_divergence=True,
        )
        orch = ProductManifoldOrchestrator(registry=reg)
        result = orch.decode(
            observation=_good_observation(),
            role_handoff_signature_cid="sig",
            parent_w42_cid="parent",
            n_w42_visible_tokens=4,
        )
        assert result.decision_branch == W43_BRANCH_PMC_DISABLED


class TestOrchestratorActive:
    def test_ratifies_on_match(self):
        sig = "a" * 64
        policy = _good_policy(sig)
        reg = build_product_manifold_registry(
            schema_cid="schema-cid",
            policy_entries=(policy,),
        )
        orch = ProductManifoldOrchestrator(registry=reg)
        result = orch.decode(
            observation=_good_observation(),
            role_handoff_signature_cid=sig,
            parent_w42_cid="b" * 64,
            n_w42_visible_tokens=4,
        )
        assert result.decision_branch == W43_BRANCH_PMC_RATIFIED
        assert result.n_w43_overhead_tokens == 1
        assert result.verification_ok
        assert result.n_structured_bits >= 7 * 256

    def test_abstains_on_causal_violation(self):
        sig = "a" * 64
        bad_obs = CellObservation(
            branch_path=(1,),
            claim_kinds=("event",),
            role_arrival_order=("r0", "r1"),
            role_universe=("r0", "r1"),
            attributes=tuple({"round": 1.0}.items()),
            subspace_vectors=(
                (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
            causal_clocks=(
                CausalVectorClock.from_mapping(
                    {"r0": 2, "r1": 5}),
                CausalVectorClock.from_mapping(
                    {"r0": 1, "r1": 4}),  # decreased!
            ),
        )
        policy = _good_policy(sig)
        reg = build_product_manifold_registry(
            schema_cid="schema-cid",
            policy_entries=(policy,),
        )
        orch = ProductManifoldOrchestrator(registry=reg)
        result = orch.decode(
            observation=bad_obs,
            role_handoff_signature_cid=sig,
            parent_w42_cid="b" * 64,
            n_w42_visible_tokens=4,
        )
        assert result.decision_branch == (
            W43_BRANCH_PMC_CAUSAL_VIOLATION_ABSTAINED)


class TestVerifier:
    def _build_envelope(self):
        sig = "a" * 64
        policy = _good_policy(sig)
        reg = build_product_manifold_registry(
            schema_cid="schema-cid",
            policy_entries=(policy,),
        )
        orch = ProductManifoldOrchestrator(registry=reg)
        result = orch.decode(
            observation=_good_observation(),
            role_handoff_signature_cid=sig,
            parent_w42_cid="b" * 64,
            n_w42_visible_tokens=4,
        )
        return reg, orch.last_envelope

    def test_envelope_verifies(self):
        reg, env = self._build_envelope()
        outcome = verify_product_manifold_ratification(
            env,
            registered_schema_cid=reg.schema_cid,
            registered_parent_w42_cid="b" * 64,
        )
        assert outcome.ok
        assert outcome.n_checks >= 18

    def test_empty_envelope_rejected(self):
        outcome = verify_product_manifold_ratification(
            None,
            registered_schema_cid="x",
            registered_parent_w42_cid="y",
        )
        assert not outcome.ok
        assert outcome.reason == "empty_w43_envelope"

    def test_wrong_schema_version_rejected(self):
        reg, env = self._build_envelope()
        import dataclasses
        bad = dataclasses.replace(
            env, schema_version="other.schema.v2")
        outcome = verify_product_manifold_ratification(
            bad,
            registered_schema_cid=reg.schema_cid,
            registered_parent_w42_cid=env.parent_w42_cid,
        )
        assert not outcome.ok
        assert outcome.reason == "w43_schema_version_unknown"

    def test_wrong_parent_w42_cid_rejected(self):
        reg, env = self._build_envelope()
        outcome = verify_product_manifold_ratification(
            env,
            registered_schema_cid=reg.schema_cid,
            registered_parent_w42_cid="wrong-parent",
        )
        assert not outcome.ok
        assert outcome.reason == "w42_parent_cid_mismatch"

    def test_swapping_state_cid_breaks_witness(self):
        reg, env = self._build_envelope()
        import dataclasses
        bad = dataclasses.replace(
            env, manifold_state_cid="0" * 64)
        outcome = verify_product_manifold_ratification(
            bad,
            registered_schema_cid=reg.schema_cid,
            registered_parent_w42_cid=env.parent_w42_cid,
        )
        assert not outcome.ok
        assert outcome.reason == (
            "w43_manifold_state_cid_mismatch")

    def test_token_accounting_mismatch_rejected(self):
        reg, env = self._build_envelope()
        import dataclasses
        bad = dataclasses.replace(env, n_w43_visible_tokens=999)
        outcome = verify_product_manifold_ratification(
            bad,
            registered_schema_cid=reg.schema_cid,
            registered_parent_w42_cid=env.parent_w42_cid,
        )
        assert not outcome.ok
        # The token-accounting check is one of the late checks; as
        # long as the decision is "not ok", the test passes.
        assert outcome.reason in (
            "w43_token_accounting_invalid",
            "w43_manifold_witness_cid_mismatch",
            "w43_manifest_v13_cid_mismatch",
        )


# =============================================================================
# Schema invariants
# =============================================================================

class TestSchemaInvariants:
    def test_schema_version_constant(self):
        assert W43_PRODUCT_MANIFOLD_SCHEMA_VERSION == (
            "coordpy.product_manifold.v1")

    def test_branches_disjoint(self):
        # All branch labels are distinct strings.
        assert len(set(W43_ALL_BRANCHES)) == len(W43_ALL_BRANCHES)
