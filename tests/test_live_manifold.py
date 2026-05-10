"""Tests for the W44 Live Manifold-Coupled Coordination layer.

Covers:
  * trivial passthrough is byte-for-AgentTeam
  * live spherical / subspace / causal gates substitute the abstain
    output for the agent's generate() call
  * factoradic compressor reduces visible-prompt tokens
  * the W44 envelope verifier enumerates 12 disjoint failure modes
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest

from coordpy import agent
from coordpy.agents import AgentTeam
from coordpy.live_manifold import (
    LiveManifoldHandoffEnvelope,
    LiveManifoldOrchestrator,
    LiveManifoldRegistry,
    LiveManifoldTeam,
    LiveObservationBuilderResult,
    LiveTurnContext,
    W44_ABSTAIN_BRANCHES,
    W44_ALL_BRANCHES,
    W44_ALL_ROUTE_MODES,
    W44_BRANCH_LIVE_CAUSAL_ABSTAIN,
    W44_BRANCH_LIVE_RATIFIED,
    W44_BRANCH_LIVE_SPHERICAL_ABSTAIN,
    W44_BRANCH_LIVE_SUBSPACE_ABSTAIN,
    W44_BRANCH_TRIVIAL_LIVE_PASSTHROUGH,
    W44_DEFAULT_ABSTAIN_OUTPUT,
    W44_LIVE_MANIFOLD_SCHEMA_VERSION,
    W44_ROUTE_MODE_FACTORADIC,
    W44_ROUTE_MODE_FACTORADIC_WITH_TEXTUAL,
    W44_ROUTE_MODE_TEXTUAL,
    build_live_manifold_registry,
    build_trivial_live_manifold_registry,
    default_live_observation_builder,
    verify_live_manifold_handoff,
)
from coordpy.product_manifold import (
    CausalVectorClock,
    CellObservation,
    ProductManifoldPolicyEntry,
    encode_spherical_consensus,
    encode_subspace_basis,
)
from coordpy.synthetic_llm import SyntheticLLMClient


# =============================================================================
# Fixtures
# =============================================================================

def _backend(default: str = "ok") -> SyntheticLLMClient:
    return SyntheticLLMClient(default_response=default)


def _agents(n: int):
    return [
        agent(f"role{i}", f"You are role{i}.",
              max_tokens=64, temperature=0.0)
        for i in range(n)
    ]


def _const_obs_builder_factory(
        *,
        signature: str,
        kinds_at_turn: callable,
        subspace_at_turn: callable | None = None,
        violate_at_turn: callable | None = None,
):
    """Build an observation builder that emits a fixed signature
    and per-turn kinds/subspace/violation overrides."""

    def _builder(ctx: LiveTurnContext) -> LiveObservationBuilderResult:
        # Build clean clocks first.
        snapshots = []
        walk = {r: 0 for r in ctx.role_universe}
        for r in ctx.role_arrival_order:
            walk[r] = walk.get(r, 0) + 1
            snapshots.append(
                CausalVectorClock.from_mapping(dict(walk)))
        if (violate_at_turn is not None
                and violate_at_turn(ctx.turn_index)
                and len(snapshots) >= 2):
            snapshots[-1] = CausalVectorClock.from_mapping({})
        kinds = kinds_at_turn(ctx.turn_index)
        subspace = (subspace_at_turn(ctx.turn_index)
                    if subspace_at_turn is not None
                    else ((1.0, 0.0), (0.0, 1.0),
                          (0.0, 0.0), (0.0, 0.0)))
        obs = CellObservation(
            branch_path=tuple(0 for _ in range(ctx.turn_index)),
            claim_kinds=tuple(kinds),
            role_arrival_order=tuple(ctx.role_arrival_order),
            role_universe=tuple(ctx.role_universe),
            attributes=tuple({"round": float(ctx.turn_index)}.items()),
            subspace_vectors=tuple(tuple(r) for r in subspace),
            causal_clocks=tuple(snapshots),
        )
        return LiveObservationBuilderResult(
            observation=obs, role_handoff_signature_cid=signature)
    return _builder


# =============================================================================
# Trivial passthrough
# =============================================================================

class TestTrivialLivePassthrough:
    def test_trivial_registry_is_trivial(self):
        reg = build_trivial_live_manifold_registry()
        assert reg.is_trivial
        assert not reg.live_enabled
        assert not reg.abstain_substitution_enabled
        assert reg.inline_route_mode == W44_ROUTE_MODE_TEXTUAL

    def test_trivial_run_matches_agent_team(self):
        a = _agents(3)
        backend = _backend("hello")
        baseline = AgentTeam(
            list(a), backend=backend,
            max_visible_handoffs=2,
            capture_capsules=True).run("task")

        backend2 = _backend("hello")
        reg = build_trivial_live_manifold_registry()
        live = LiveManifoldTeam(
            list(a), backend=backend2, registry=reg,
            max_visible_handoffs=2,
            capture_capsules=True).run("task")

        assert live.final_output == baseline.final_output
        assert len(live.turns) == len(baseline.turns)
        assert live.n_behavioral_changes == 0
        assert live.n_abstain_substitutions == 0
        for t in live.live_turns:
            assert (t.envelope.decision_branch
                    == W44_BRANCH_TRIVIAL_LIVE_PASSTHROUGH)


# =============================================================================
# Live gates
# =============================================================================

class TestLiveSphericalGate:
    def test_divergent_kinds_substitute_abstain(self):
        sig = hashlib.sha256(b"sph-test").hexdigest()
        expected_kinds = ("event", "event")
        divergent_kinds = ("alert", "alert")
        policy = ProductManifoldPolicyEntry(
            role_handoff_signature_cid=sig,
            expected_services=("sp",),
            expected_spherical=encode_spherical_consensus(
                expected_kinds),
            expected_subspace=encode_subspace_basis(
                ((1.0, 0.0), (0.0, 1.0),
                 (0.0, 0.0), (0.0, 0.0))),
            expected_causal_topology_hash="(...)",
        )
        schema = hashlib.sha256(b"sph-schema").hexdigest()
        reg = build_live_manifold_registry(
            schema_cid=schema, policy_entries=(policy,),
            abstain_substitution_enabled=True)

        builder = _const_obs_builder_factory(
            signature=sig,
            kinds_at_turn=lambda i: (
                expected_kinds if i == 0 else divergent_kinds),
        )
        team = LiveManifoldTeam(
            _agents(3), backend=_backend("real_output"),
            registry=reg, observation_builder=builder,
            max_visible_handoffs=2, capture_capsules=True)
        result = team.run("task")
        assert result.live_turns[0].envelope.decision_branch == (
            W44_BRANCH_LIVE_RATIFIED)
        assert result.live_turns[0].agent_turn.output == "real_output"
        for t in result.live_turns[1:]:
            assert t.envelope.decision_branch == (
                W44_BRANCH_LIVE_SPHERICAL_ABSTAIN)
            assert t.agent_turn.output == W44_DEFAULT_ABSTAIN_OUTPUT
            assert t.envelope.behavioral_change

    def test_audit_only_does_not_substitute(self):
        sig = hashlib.sha256(b"sph-audit-test").hexdigest()
        expected_kinds = ("event", "event")
        divergent_kinds = ("alert", "alert")
        policy = ProductManifoldPolicyEntry(
            role_handoff_signature_cid=sig,
            expected_services=("sp",),
            expected_spherical=encode_spherical_consensus(
                expected_kinds),
            expected_subspace=encode_subspace_basis(
                ((1.0, 0.0), (0.0, 1.0),
                 (0.0, 0.0), (0.0, 0.0))),
            expected_causal_topology_hash="(...)",
        )
        schema = hashlib.sha256(b"sph-audit-schema").hexdigest()
        reg = build_live_manifold_registry(
            schema_cid=schema, policy_entries=(policy,),
            abstain_substitution_enabled=False)
        builder = _const_obs_builder_factory(
            signature=sig,
            kinds_at_turn=lambda i: (
                expected_kinds if i == 0 else divergent_kinds),
        )
        team = LiveManifoldTeam(
            _agents(3), backend=_backend("real_output"),
            registry=reg, observation_builder=builder,
            max_visible_handoffs=2, capture_capsules=True)
        result = team.run("task")
        # Branches still flag divergence...
        assert (result.live_turns[1].envelope.decision_branch
                == W44_BRANCH_LIVE_SPHERICAL_ABSTAIN)
        # ...but the agent's output is the real synthetic backend
        # response, not the abstain marker.
        assert result.live_turns[1].agent_turn.output == "real_output"
        assert result.n_abstain_substitutions == 0


class TestLiveSubspaceGate:
    def test_drift_substitute_abstain(self):
        sig = hashlib.sha256(b"sub-test").hexdigest()
        expected_kinds = ("agent_output", "agent_output")
        clean_subspace = (
            (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))
        drift_subspace = (
            (0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, 0.0))
        policy = ProductManifoldPolicyEntry(
            role_handoff_signature_cid=sig,
            expected_services=("sub",),
            expected_spherical=encode_spherical_consensus(
                expected_kinds),
            expected_subspace=encode_subspace_basis(clean_subspace),
            expected_causal_topology_hash="(...)",
        )
        schema = hashlib.sha256(b"sub-schema").hexdigest()
        reg = build_live_manifold_registry(
            schema_cid=schema, policy_entries=(policy,),
            abstain_substitution_enabled=True)
        builder = _const_obs_builder_factory(
            signature=sig,
            kinds_at_turn=lambda i: expected_kinds,
            subspace_at_turn=lambda i: (
                clean_subspace if i == 0 else drift_subspace),
        )
        team = LiveManifoldTeam(
            _agents(3), backend=_backend("real_output"),
            registry=reg, observation_builder=builder,
            max_visible_handoffs=2, capture_capsules=True)
        result = team.run("task")
        for t in result.live_turns[1:]:
            assert (t.envelope.decision_branch
                    == W44_BRANCH_LIVE_SUBSPACE_ABSTAIN)
            assert t.agent_turn.output == W44_DEFAULT_ABSTAIN_OUTPUT


class TestLiveCausalGate:
    def test_violation_substitute_abstain(self):
        sig = hashlib.sha256(b"causal-test").hexdigest()
        expected_kinds = ("agent_output", "agent_output")
        clean_subspace = (
            (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))
        policy = ProductManifoldPolicyEntry(
            role_handoff_signature_cid=sig,
            expected_services=("causal",),
            expected_spherical=encode_spherical_consensus(
                expected_kinds),
            expected_subspace=encode_subspace_basis(clean_subspace),
            expected_causal_topology_hash="(...)",
        )
        schema = hashlib.sha256(b"causal-schema").hexdigest()
        reg = build_live_manifold_registry(
            schema_cid=schema, policy_entries=(policy,),
            abstain_substitution_enabled=True)
        builder = _const_obs_builder_factory(
            signature=sig,
            kinds_at_turn=lambda i: expected_kinds,
            violate_at_turn=lambda i: i >= 2,
        )
        team = LiveManifoldTeam(
            _agents(3), backend=_backend("real_output"),
            registry=reg, observation_builder=builder,
            max_visible_handoffs=2, capture_capsules=True)
        result = team.run("task")
        # Turn 2 has the violation injected.
        assert (result.live_turns[2].envelope.decision_branch
                == W44_BRANCH_LIVE_CAUSAL_ABSTAIN)
        assert (result.live_turns[2].agent_turn.output
                == W44_DEFAULT_ABSTAIN_OUTPUT)


# =============================================================================
# Factoradic compressor
# =============================================================================

class TestFactoradicCompressor:
    def test_factoradic_saves_visible_tokens(self):
        n = 6
        a = _agents(n)
        sig = hashlib.sha256(b"fac-test").hexdigest()
        expected_kinds = ("agent_output",) * (n - 1)
        clean_subspace = (
            (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))
        policy = ProductManifoldPolicyEntry(
            role_handoff_signature_cid=sig,
            expected_services=("fac",),
            expected_spherical=encode_spherical_consensus(
                expected_kinds),
            expected_subspace=encode_subspace_basis(clean_subspace),
            expected_causal_topology_hash="(...)",
        )
        schema = hashlib.sha256(b"fac-schema").hexdigest()
        reg_textual = build_live_manifold_registry(
            schema_cid=schema, policy_entries=(policy,),
            abstain_substitution_enabled=False,
            inline_route_mode=W44_ROUTE_MODE_TEXTUAL)
        reg_fac = build_live_manifold_registry(
            schema_cid=schema, policy_entries=(policy,),
            abstain_substitution_enabled=False,
            inline_route_mode=W44_ROUTE_MODE_FACTORADIC)
        builder = _const_obs_builder_factory(
            signature=sig,
            kinds_at_turn=lambda i: expected_kinds,
        )

        # Use a longer payload so the factoradic compression beats
        # the per-turn parsing cost.
        long_payload = (
            "agent output payload with several extra words "
            "to make rendering meaningful and longer per turn")
        team_t = LiveManifoldTeam(
            list(a), backend=SyntheticLLMClient(
                default_response=long_payload),
            registry=reg_textual, observation_builder=builder,
            max_visible_handoffs=4, capture_capsules=True)
        team_f = LiveManifoldTeam(
            list(a), backend=SyntheticLLMClient(
                default_response=long_payload),
            registry=reg_fac, observation_builder=builder,
            max_visible_handoffs=4, capture_capsules=True)
        rt = team_t.run("permutation task")
        rf = team_f.run("permutation task")
        assert rt.n_visible_tokens_saved_factoradic == 0
        assert rf.n_visible_tokens_saved_factoradic > 4

    def test_factoradic_with_textual_saves_less_than_pure(self):
        # Hybrid mode emits BOTH the factoradic header and the
        # textual rendering, so its saving (relative to pure
        # textual) is small; pure factoradic strictly dominates.
        n = 6
        a = _agents(n)
        sig = hashlib.sha256(b"fac-with-test").hexdigest()
        expected_kinds = ("agent_output",) * (n - 1)
        clean_subspace = (
            (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))
        policy = ProductManifoldPolicyEntry(
            role_handoff_signature_cid=sig,
            expected_services=("fac",),
            expected_spherical=encode_spherical_consensus(
                expected_kinds),
            expected_subspace=encode_subspace_basis(clean_subspace),
            expected_causal_topology_hash="(...)",
        )
        schema = hashlib.sha256(b"fac-schema-with").hexdigest()
        reg_hybrid = build_live_manifold_registry(
            schema_cid=schema, policy_entries=(policy,),
            abstain_substitution_enabled=False,
            inline_route_mode=(
                W44_ROUTE_MODE_FACTORADIC_WITH_TEXTUAL))
        reg_pure = build_live_manifold_registry(
            schema_cid=schema, policy_entries=(policy,),
            abstain_substitution_enabled=False,
            inline_route_mode=W44_ROUTE_MODE_FACTORADIC)
        builder = _const_obs_builder_factory(
            signature=sig,
            kinds_at_turn=lambda i: expected_kinds,
        )
        long = (
            "agent payload with many extra words to make rendering"
            " meaningful per turn for measurement")
        hyb = LiveManifoldTeam(
            list(a), backend=SyntheticLLMClient(default_response=long),
            registry=reg_hybrid, observation_builder=builder,
            max_visible_handoffs=4, capture_capsules=True).run("t")
        pure = LiveManifoldTeam(
            list(a), backend=SyntheticLLMClient(default_response=long),
            registry=reg_pure, observation_builder=builder,
            max_visible_handoffs=4, capture_capsules=True).run("t")
        assert (pure.n_visible_tokens_saved_factoradic
                > hyb.n_visible_tokens_saved_factoradic)


# =============================================================================
# Verifier
# =============================================================================

class TestVerifier:
    def _build_envelope_pair(self):
        # Build a real run, capture the first envelope.
        sig = hashlib.sha256(b"verify-test").hexdigest()
        expected_kinds = ("event", "event")
        policy = ProductManifoldPolicyEntry(
            role_handoff_signature_cid=sig,
            expected_services=("v",),
            expected_spherical=encode_spherical_consensus(
                expected_kinds),
            expected_subspace=encode_subspace_basis(
                ((1.0, 0.0), (0.0, 1.0),
                 (0.0, 0.0), (0.0, 0.0))),
            expected_causal_topology_hash="(...)",
        )
        schema = hashlib.sha256(b"verify-schema").hexdigest()
        reg = build_live_manifold_registry(
            schema_cid=schema, policy_entries=(policy,),
            abstain_substitution_enabled=True)
        builder = _const_obs_builder_factory(
            signature=sig,
            kinds_at_turn=lambda i: expected_kinds,
        )
        team = LiveManifoldTeam(
            _agents(3), backend=_backend("real"),
            registry=reg, observation_builder=builder,
            capture_capsules=True)
        result = team.run("verify task")
        return result.live_turns[0].envelope, schema

    def test_clean_envelope_verifies(self):
        env, schema = self._build_envelope_pair()
        outcome = verify_live_manifold_handoff(
            env, registered_schema_cid=schema)
        assert outcome.ok
        assert outcome.n_checks >= 12

    def test_empty_envelope(self):
        outcome = verify_live_manifold_handoff(
            None, registered_schema_cid="x")
        assert not outcome.ok
        assert outcome.reason == "empty_w44_envelope"

    def test_schema_version_unknown(self):
        env, schema = self._build_envelope_pair()
        bad = dataclasses.replace(env, schema_version="bogus")
        outcome = verify_live_manifold_handoff(
            bad, registered_schema_cid=schema)
        assert outcome.reason == "w44_schema_version_unknown"

    def test_schema_cid_mismatch(self):
        env, _ = self._build_envelope_pair()
        outcome = verify_live_manifold_handoff(
            env, registered_schema_cid="other")
        assert outcome.reason == "w44_schema_cid_mismatch"

    def test_decision_branch_unknown(self):
        env, schema = self._build_envelope_pair()
        bad = dataclasses.replace(env, decision_branch="bogus")
        outcome = verify_live_manifold_handoff(
            bad, registered_schema_cid=schema)
        assert outcome.reason == "w44_decision_branch_unknown"

    def test_route_mode_unknown(self):
        env, schema = self._build_envelope_pair()
        bad = dataclasses.replace(env, inline_route_mode="bogus")
        outcome = verify_live_manifold_handoff(
            bad, registered_schema_cid=schema)
        assert outcome.reason == "w44_route_mode_unknown"

    def test_signature_invalid(self):
        env, schema = self._build_envelope_pair()
        bad = dataclasses.replace(
            env, role_handoff_signature_cid="short")
        outcome = verify_live_manifold_handoff(
            bad, registered_schema_cid=schema)
        assert (outcome.reason
                == "w44_role_handoff_signature_cid_invalid")

    def test_construction_witness_mismatch(self):
        env, schema = self._build_envelope_pair()
        bad = dataclasses.replace(
            env, prompt_construction_witness_cid="0" * 64)
        outcome = verify_live_manifold_handoff(
            bad, registered_schema_cid=schema)
        assert (outcome.reason
                == "w44_prompt_construction_witness_cid_mismatch")

    def test_witness_cid_mismatch(self):
        env, schema = self._build_envelope_pair()
        bad = dataclasses.replace(env, live_witness_cid="0" * 64)
        outcome = verify_live_manifold_handoff(
            bad, registered_schema_cid=schema)
        assert outcome.reason == "w44_live_witness_cid_mismatch"

    def test_outer_cid_mismatch(self):
        env, schema = self._build_envelope_pair()
        bad = dataclasses.replace(env, live_outer_cid="0" * 64)
        outcome = verify_live_manifold_handoff(
            bad, registered_schema_cid=schema)
        assert outcome.reason == "w44_outer_cid_mismatch"

    def test_token_accounting_invalid(self):
        env, schema = self._build_envelope_pair()
        bad = dataclasses.replace(
            env, n_visible_prompt_tokens_saved=999)
        outcome = verify_live_manifold_handoff(
            bad, registered_schema_cid=schema)
        assert outcome.reason == "w44_token_accounting_invalid"

    def test_factoradic_bits_invalid(self):
        env, schema = self._build_envelope_pair()
        bad = dataclasses.replace(env, factoradic_int=-1)
        outcome = verify_live_manifold_handoff(
            bad, registered_schema_cid=schema)
        assert outcome.reason == "w44_factoradic_bits_invalid"


# =============================================================================
# Schema and contract sanity
# =============================================================================

class TestSchemaContract:
    def test_all_branches_unique(self):
        assert len(set(W44_ALL_BRANCHES)) == len(W44_ALL_BRANCHES)

    def test_abstain_branches_subset(self):
        assert W44_ABSTAIN_BRANCHES.issubset(set(W44_ALL_BRANCHES))

    def test_all_route_modes(self):
        assert W44_ROUTE_MODE_TEXTUAL in W44_ALL_ROUTE_MODES
        assert W44_ROUTE_MODE_FACTORADIC in W44_ALL_ROUTE_MODES
        assert (W44_ROUTE_MODE_FACTORADIC_WITH_TEXTUAL
                in W44_ALL_ROUTE_MODES)

    def test_schema_version_constant(self):
        assert (W44_LIVE_MANIFOLD_SCHEMA_VERSION
                == "coordpy.live_manifold.v1")
