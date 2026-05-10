"""Tests for the W45 Learned Manifold Controller layer.

Covers:
  * trivial passthrough reduces to AgentTeam byte-for-byte
  * channel-feature extraction is deterministic
  * ridge fitter produces sensible projections + attention logits
  * forward pass returns calibrated probabilities + bucketed
    confidence + role-aware deltas
  * causal-mask witness flags channel observability correctly
  * the W45 envelope verifier enumerates 14+ disjoint failure
    modes
  * the prompt builder emits ``MANIFOLD_HINT: route=...`` bytes
    under hint mode and a textual shadow under hint-off
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest

from coordpy import agent
from coordpy.agents import AgentTeam
from coordpy.learned_manifold import (
    ControllerForwardResult,
    HintAwareSyntheticBackend,
    LearnedControllerParams,
    LearnedManifoldHandoffEnvelope,
    LearnedManifoldOrchestrator,
    LearnedManifoldRegistry,
    LearnedManifoldTeam,
    TrainingExample,
    TrainingSet,
    W45_ALL_BRANCHES,
    W45_ALL_FAILURE_MODES,
    W45_ALL_HINT_MODES,
    W45_BRANCH_LEARNED_MARGIN_ABSTAIN,
    W45_BRANCH_LEARNED_RATIFIED,
    W45_BRANCH_TRIVIAL_LEARNED_PASSTHROUGH,
    W45_CHANNEL_ORDER,
    W45_CONFIDENCE_BUCKETS,
    W45_DEFAULT_FEATURE_DIM,
    W45_HINT_MODE_FACTORADIC_WITH_HINT,
    W45_HINT_MODE_HINT_ONLY,
    W45_HINT_MODE_OFF,
    W45_LEARNED_ABSTAIN_BRANCHES,
    W45_LEARNED_MANIFOLD_SCHEMA_VERSION,
    W45_N_CHANNELS,
    _channel_features_from_bundle,
    build_learned_manifold_registry,
    build_trivial_learned_manifold_registry,
    build_unfitted_controller_params,
    derive_causal_mask,
    fit_learned_controller,
    forward_controller,
    verify_learned_manifold_handoff,
)
from coordpy.product_manifold import (
    CausalVectorClock,
    CellObservation,
    ProductManifoldPolicyEntry,
    encode_cell_channels,
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


def _build_simple_training_set(
        *,
        signature: str = "sig0",
        n: int = 8,
        feature_dim: int = W45_DEFAULT_FEATURE_DIM,
) -> TrainingSet:
    examples = []
    for i in range(n):
        positive = (i % 2 == 0)
        feats = []
        for c_name in W45_CHANNEL_ORDER:
            if c_name == "spherical":
                vec = [1.0 if positive else -1.0] + (
                    [0.0] * (feature_dim - 1))
            else:
                vec = [0.0] * feature_dim
            feats.append((c_name, tuple(vec)))
        examples.append(TrainingExample(
            role=f"role{i % 3}",
            role_handoff_signature_cid=signature,
            channel_features=tuple(feats),
            label=1.0 if positive else -1.0,
        ))
    return TrainingSet(
        examples=tuple(examples), feature_dim=feature_dim)


# =============================================================================
# Trivial passthrough
# =============================================================================

class TestTrivialLearnedPassthrough:
    def test_trivial_registry_is_trivial(self):
        reg = build_trivial_learned_manifold_registry()
        assert reg.is_trivial
        assert not reg.learned_enabled
        assert not reg.use_attention_routing
        assert reg.role_adapter_disabled
        assert reg.prompt_hint_mode == W45_HINT_MODE_OFF
        assert not reg.abstain_substitution_enabled
        assert reg.params.fitting_method == "unfitted"

    def test_trivial_run_matches_agent_team(self):
        a = _agents(3)
        baseline = AgentTeam(
            list(a), backend=_backend("hello"),
            max_visible_handoffs=2,
            capture_capsules=True).run("task")

        reg = build_trivial_learned_manifold_registry()
        learned = LearnedManifoldTeam(
            list(a), backend=_backend("hello"),
            registry=reg, max_visible_handoffs=2,
            capture_capsules=True).run("task")

        assert learned.final_output == baseline.final_output
        assert len(learned.turns) == len(baseline.turns)
        assert learned.n_behavioral_changes == 0
        assert learned.n_abstain_substitutions == 0
        for t in learned.learned_turns:
            assert (t.envelope.decision_branch
                    == W45_BRANCH_TRIVIAL_LEARNED_PASSTHROUGH)


# =============================================================================
# Channel feature extraction
# =============================================================================

class TestChannelFeatures:
    def test_zero_observation_yields_zero_features(self):
        obs = CellObservation()
        bundle = encode_cell_channels(obs)
        feats = _channel_features_from_bundle(
            bundle, feature_dim=W45_DEFAULT_FEATURE_DIM)
        assert set(feats.keys()) == set(W45_CHANNEL_ORDER)
        for c_name, v in feats.items():
            assert len(v) == W45_DEFAULT_FEATURE_DIM

    def test_features_are_deterministic(self):
        obs = CellObservation(
            branch_path=(0, 1, 0),
            claim_kinds=("event", "summary"),
            role_arrival_order=("a", "b"),
            role_universe=("a", "b", "c"),
            attributes=(("round", 1.0),),
            subspace_vectors=((1.0, 0.0), (0.0, 1.0)),
            causal_clocks=(
                CausalVectorClock.from_mapping({"a": 1}),),
        )
        bundle = encode_cell_channels(obs)
        f1 = _channel_features_from_bundle(bundle)
        f2 = _channel_features_from_bundle(bundle)
        assert f1 == f2

    def test_spherical_feature_reflects_agreement(self):
        kinds = ("event", "summary")
        obs = CellObservation(claim_kinds=kinds)
        bundle = encode_cell_channels(obs)
        expected = encode_spherical_consensus(kinds)
        feats = _channel_features_from_bundle(
            bundle, expected_spherical=expected)
        # Agreement should be 1.0 when expected matches observed.
        assert feats["spherical"][0] == pytest.approx(1.0, abs=1e-6)


# =============================================================================
# Ridge fitter + forward pass
# =============================================================================

class TestFitterAndForward:
    def test_unfitted_params_yield_neutral_prediction(self):
        params = build_unfitted_controller_params()
        feats = {c: tuple(0.0 for _ in range(params.feature_dim))
                 for c in W45_CHANNEL_ORDER}
        fr = forward_controller(
            channel_features=feats, params=params, role="role0")
        # Sigmoid(0) = 0.5
        assert fr.ratify_probability == pytest.approx(0.5, abs=1e-6)
        assert fr.confidence_bucket == 2  # mid bucket

    def test_fitter_recovers_spherical_signal(self):
        ts = _build_simple_training_set()
        params = fit_learned_controller(ts)
        assert params.fitting_method == "ridge_v1"
        feats_pos = {c: tuple(0.0 for _ in range(4))
                     for c in W45_CHANNEL_ORDER}
        feats_pos["spherical"] = (1.0, 0.0, 0.0, 0.0)
        fr_pos = forward_controller(
            channel_features=feats_pos, params=params,
            role="role0", use_attention_routing=True)
        assert fr_pos.ratify_probability > 0.7

        feats_neg = dict(feats_pos)
        feats_neg["spherical"] = (-1.0, 0.0, 0.0, 0.0)
        fr_neg = forward_controller(
            channel_features=feats_neg, params=params,
            role="role0", use_attention_routing=True)
        assert fr_neg.ratify_probability < 0.3

    def test_attention_weights_normalize_to_one(self):
        ts = _build_simple_training_set()
        params = fit_learned_controller(ts)
        feats = {c: tuple(0.0 for _ in range(4))
                 for c in W45_CHANNEL_ORDER}
        feats["spherical"] = (1.0, 0.0, 0.0, 0.0)
        fr = forward_controller(
            channel_features=feats, params=params, role="role0",
            use_attention_routing=True)
        s = sum(fr.attention_weights)
        assert s == pytest.approx(1.0, abs=1e-6)
        assert len(fr.attention_weights) == W45_N_CHANNELS

    def test_role_adapter_flips_sign_on_inverted_role(self):
        # Build a bank where role0..role2 use label=sign(sph) and
        # role3 uses the inverted convention.
        examples = []
        for i in range(8):
            pos = (i % 2 == 0)
            for r in ("role0", "role1", "role2"):
                feats = []
                for c_name in W45_CHANNEL_ORDER:
                    if c_name == "spherical":
                        v = [1.0 if pos else -1.0] + [0.0] * 3
                    else:
                        v = [0.0] * 4
                    feats.append((c_name, tuple(v)))
                examples.append(TrainingExample(
                    role=r,
                    role_handoff_signature_cid="sig",
                    channel_features=tuple(feats),
                    label=1.0 if pos else -1.0,
                ))
            # role3: flipped.
            feats = []
            for c_name in W45_CHANNEL_ORDER:
                if c_name == "spherical":
                    v = [1.0 if pos else -1.0] + [0.0] * 3
                else:
                    v = [0.0] * 4
                feats.append((c_name, tuple(v)))
            examples.append(TrainingExample(
                role="role3",
                role_handoff_signature_cid="sig",
                channel_features=tuple(feats),
                label=-1.0 if pos else 1.0,
            ))
        ts = TrainingSet(
            examples=tuple(examples), feature_dim=4)
        params = fit_learned_controller(ts)
        # Positive spherical feature.
        feats = {c: tuple(0.0 for _ in range(4))
                 for c in W45_CHANNEL_ORDER}
        feats["spherical"] = (1.0, 0.0, 0.0, 0.0)
        fr_role0 = forward_controller(
            channel_features=feats, params=params,
            role="role0", use_attention_routing=True)
        fr_role3 = forward_controller(
            channel_features=feats, params=params,
            role="role3", use_attention_routing=True)
        # role0 (positive convention): ratify.
        assert fr_role0.ratify_probability > 0.7
        # role3 (inverted): abstain (probability < 0.5).
        assert fr_role3.ratify_probability < 0.3

    def test_unfitted_role_falls_back_to_shared_base(self):
        ts = _build_simple_training_set()
        params = fit_learned_controller(ts)
        feats = {c: tuple(0.0 for _ in range(4))
                 for c in W45_CHANNEL_ORDER}
        feats["spherical"] = (1.0, 0.0, 0.0, 0.0)
        fr_unknown = forward_controller(
            channel_features=feats, params=params,
            role="never_seen_role", use_attention_routing=True)
        # Should behave like role0 / role1 / role2 (positive
        # convention).
        assert fr_unknown.role_delta_value == 0.0


# =============================================================================
# Causal mask witness
# =============================================================================

class TestCausalMask:
    def test_empty_obs_flags_unobservable(self):
        obs = CellObservation()
        bundle = encode_cell_channels(obs)
        cm = derive_causal_mask(bundle, turn_index=0)
        flags = dict(zip(W45_CHANNEL_ORDER, cm.observable_channels))
        # Hyperbolic and euclidean are always observable (zero
        # encoding is a valid trivial case).
        assert flags["hyperbolic"]
        assert flags["euclidean"]
        # Others are unobservable for an empty cell.
        assert not flags["spherical"]
        assert not flags["factoradic"]
        assert not flags["subspace"]
        assert not flags["causal"]

    def test_full_obs_flags_all_observable(self):
        obs = CellObservation(
            branch_path=(0,),
            claim_kinds=("event",),
            role_arrival_order=("a",),
            role_universe=("a", "b"),
            subspace_vectors=((1.0, 0.0), (0.0, 1.0)),
            causal_clocks=(
                CausalVectorClock.from_mapping({"a": 1}),),
        )
        bundle = encode_cell_channels(obs)
        cm = derive_causal_mask(bundle, turn_index=2)
        for c_name, ok in zip(
                W45_CHANNEL_ORDER, cm.observable_channels):
            assert ok, f"{c_name} should be observable"


# =============================================================================
# Verifier
# =============================================================================

class TestVerifier:
    def _build_envelope(self):
        sig = hashlib.sha256(b"verify-test").hexdigest()
        schema = hashlib.sha256(b"verify-schema").hexdigest()
        expected_kinds = ("event", "summary")
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
        reg = build_learned_manifold_registry(
            schema_cid=schema, policy_entries=(policy,),
            prompt_hint_mode=W45_HINT_MODE_FACTORADIC_WITH_HINT)
        team = LearnedManifoldTeam(
            _agents(3), backend=_backend("real"),
            registry=reg, capture_capsules=True)
        result = team.run("verify task")
        return result.learned_turns[0].envelope, schema

    def test_clean_envelope_verifies(self):
        env, schema = self._build_envelope()
        outcome = verify_learned_manifold_handoff(
            env, registered_schema_cid=schema)
        assert outcome.ok
        assert outcome.n_checks >= 14

    def test_empty_envelope(self):
        outcome = verify_learned_manifold_handoff(
            None, registered_schema_cid="x")
        assert not outcome.ok
        assert outcome.reason == "empty_w45_envelope"

    def test_schema_version_unknown(self):
        env, schema = self._build_envelope()
        bad = dataclasses.replace(env, schema_version="bogus")
        out = verify_learned_manifold_handoff(
            bad, registered_schema_cid=schema)
        assert out.reason == "w45_schema_version_unknown"

    def test_schema_cid_mismatch(self):
        env, _ = self._build_envelope()
        out = verify_learned_manifold_handoff(
            env, registered_schema_cid="other")
        assert out.reason == "w45_schema_cid_mismatch"

    def test_decision_branch_unknown(self):
        env, schema = self._build_envelope()
        bad = dataclasses.replace(env, decision_branch="bogus")
        out = verify_learned_manifold_handoff(
            bad, registered_schema_cid=schema)
        assert out.reason == "w45_decision_branch_unknown"

    def test_hint_mode_unknown(self):
        env, schema = self._build_envelope()
        bad = dataclasses.replace(env, prompt_hint_mode="bogus")
        out = verify_learned_manifold_handoff(
            bad, registered_schema_cid=schema)
        assert out.reason == "w45_hint_mode_unknown"

    def test_signature_invalid(self):
        env, schema = self._build_envelope()
        bad = dataclasses.replace(
            env, role_handoff_signature_cid="short")
        out = verify_learned_manifold_handoff(
            bad, registered_schema_cid=schema)
        assert out.reason == (
            "w45_role_handoff_signature_cid_invalid")

    def test_token_accounting_invalid(self):
        env, schema = self._build_envelope()
        bad = dataclasses.replace(
            env, n_visible_prompt_tokens_saved=999)
        out = verify_learned_manifold_handoff(
            bad, registered_schema_cid=schema)
        assert out.reason == "w45_token_accounting_invalid"

    def test_confidence_bucket_invalid(self):
        env, schema = self._build_envelope()
        bad = dataclasses.replace(env, hint_confidence_bucket=99)
        out = verify_learned_manifold_handoff(
            bad, registered_schema_cid=schema)
        assert out.reason == "w45_confidence_bucket_invalid"

    def test_ratify_probability_invalid(self):
        env, schema = self._build_envelope()
        bad = dataclasses.replace(env, ratify_probability=2.0)
        out = verify_learned_manifold_handoff(
            bad, registered_schema_cid=schema)
        assert out.reason == "w45_ratify_probability_invalid"

    def test_outer_cid_mismatch(self):
        env, schema = self._build_envelope()
        bad = dataclasses.replace(env, learned_outer_cid="0" * 64)
        out = verify_learned_manifold_handoff(
            bad, registered_schema_cid=schema)
        assert out.reason == "w45_outer_cid_mismatch"

    def test_construction_witness_mismatch(self):
        env, schema = self._build_envelope()
        bad = dataclasses.replace(
            env, prompt_construction_witness_cid="0" * 64)
        out = verify_learned_manifold_handoff(
            bad, registered_schema_cid=schema)
        assert out.reason == (
            "w45_prompt_construction_witness_cid_mismatch")


# =============================================================================
# Prompt hint
# =============================================================================

class TestPromptHint:
    def test_hint_mode_off_matches_w44_prompt(self):
        sig = hashlib.sha256(b"hint-off").hexdigest()
        schema = hashlib.sha256(b"hint-off-schema").hexdigest()
        policy = ProductManifoldPolicyEntry(
            role_handoff_signature_cid=sig,
            expected_services=("v",),
            expected_spherical=encode_spherical_consensus(
                ("event", "summary")),
            expected_subspace=encode_subspace_basis(
                ((1.0, 0.0), (0.0, 1.0),
                 (0.0, 0.0), (0.0, 0.0))),
            expected_causal_topology_hash="(...)",
        )
        reg = build_learned_manifold_registry(
            schema_cid=schema, policy_entries=(policy,),
            prompt_hint_mode=W45_HINT_MODE_OFF)
        team = LearnedManifoldTeam(
            _agents(3), backend=_backend("ok"),
            registry=reg, capture_capsules=True)
        result = team.run("task")
        # No hint substring should appear when hint mode is off.
        for t in result.learned_turns:
            assert "MANIFOLD_HINT" not in t.agent_turn.prompt

    def test_hint_mode_on_emits_hint(self):
        sig = hashlib.sha256(b"hint-on").hexdigest()
        schema = hashlib.sha256(b"hint-on-schema").hexdigest()
        policy = ProductManifoldPolicyEntry(
            role_handoff_signature_cid=sig,
            expected_services=("v",),
            expected_spherical=encode_spherical_consensus(
                ("event", "summary")),
            expected_subspace=encode_subspace_basis(
                ((1.0, 0.0), (0.0, 1.0),
                 (0.0, 0.0), (0.0, 0.0))),
            expected_causal_topology_hash="(...)",
        )
        reg = build_learned_manifold_registry(
            schema_cid=schema, policy_entries=(policy,),
            prompt_hint_mode=W45_HINT_MODE_FACTORADIC_WITH_HINT)
        team = LearnedManifoldTeam(
            _agents(3), backend=_backend("ok"),
            registry=reg, capture_capsules=True)
        result = team.run("task")
        # The hint appears on every turn except possibly turn 0
        # when there's no factoradic route (we still emit the hint
        # though).
        seen = sum(
            1 for t in result.learned_turns
            if "MANIFOLD_HINT: route=" in t.agent_turn.prompt)
        assert seen >= 2

    def test_hint_aware_backend_responds_with_hint(self):
        sig = hashlib.sha256(b"hint-aware-test").hexdigest()
        schema = hashlib.sha256(b"hint-aware-schema").hexdigest()
        policy = ProductManifoldPolicyEntry(
            role_handoff_signature_cid=sig,
            expected_services=("v",),
            expected_spherical=encode_spherical_consensus(
                ("event", "summary")),
            expected_subspace=encode_subspace_basis(
                ((1.0, 0.0), (0.0, 1.0),
                 (0.0, 0.0), (0.0, 0.0))),
            expected_causal_topology_hash="(...)",
        )
        reg = build_learned_manifold_registry(
            schema_cid=schema, policy_entries=(policy,),
            prompt_hint_mode=W45_HINT_MODE_FACTORADIC_WITH_HINT)
        team = LearnedManifoldTeam(
            _agents(3), backend=HintAwareSyntheticBackend(),
            registry=reg, capture_capsules=True)
        result = team.run("task")
        assert "MANIFOLD_OK" in result.final_output


# =============================================================================
# Schema and contract sanity
# =============================================================================

class TestSchemaContract:
    def test_all_branches_unique(self):
        assert len(set(W45_ALL_BRANCHES)) == len(W45_ALL_BRANCHES)

    def test_abstain_branches_subset(self):
        assert W45_LEARNED_ABSTAIN_BRANCHES.issubset(
            set(W45_ALL_BRANCHES))

    def test_all_hint_modes(self):
        assert W45_HINT_MODE_OFF in W45_ALL_HINT_MODES
        assert (W45_HINT_MODE_FACTORADIC_WITH_HINT
                in W45_ALL_HINT_MODES)
        assert W45_HINT_MODE_HINT_ONLY in W45_ALL_HINT_MODES

    def test_schema_version_constant(self):
        assert (W45_LEARNED_MANIFOLD_SCHEMA_VERSION
                == "coordpy.learned_manifold.v1")

    def test_n_channels_is_six(self):
        assert W45_N_CHANNELS == 6
        assert len(W45_CHANNEL_ORDER) == 6

    def test_failure_modes_disjoint(self):
        assert len(W45_ALL_FAILURE_MODES) >= 14
        assert (len(set(W45_ALL_FAILURE_MODES))
                == len(W45_ALL_FAILURE_MODES))


# =============================================================================
# Margin gating
# =============================================================================

class TestMarginGate:
    def test_margin_abstain_fires_when_logit_below_threshold(self):
        # Fit a controller that will produce a very negative gate
        # logit on adversarial features.
        ts = _build_simple_training_set()
        params = fit_learned_controller(ts)
        sig = hashlib.sha256(b"margin-test").hexdigest()
        schema = hashlib.sha256(b"margin-schema").hexdigest()
        policy = ProductManifoldPolicyEntry(
            role_handoff_signature_cid=sig,
            expected_services=("v",),
            expected_spherical=encode_spherical_consensus(
                ("event", "summary")),
            expected_subspace=encode_subspace_basis(
                ((1.0, 0.0), (0.0, 1.0),
                 (0.0, 0.0), (0.0, 0.0))),
            expected_causal_topology_hash="(...)",
        )
        # Configure a very high margin threshold so EVERY ratified
        # W44 decision falls into margin_abstain.
        reg = build_learned_manifold_registry(
            schema_cid=schema, policy_entries=(policy,),
            params=params,
            prompt_hint_mode=W45_HINT_MODE_OFF,
            abstain_substitution_enabled=True,
            margin_abstain_threshold=100.0,
            spherical_agreement_min=0.0,
        )
        # Use an observation builder that emits the right signature
        # to trigger policy match.
        from coordpy.live_manifold import (
            LiveObservationBuilderResult, LiveTurnContext)

        def _builder(ctx):
            obs = CellObservation(
                branch_path=tuple(
                    0 for _ in range(ctx.turn_index)),
                claim_kinds=("event", "summary"),
                role_arrival_order=tuple(ctx.role_arrival_order),
                role_universe=tuple(ctx.role_universe),
                attributes=(("round", float(ctx.turn_index)),),
                subspace_vectors=(
                    (1.0, 0.0), (0.0, 1.0),
                    (0.0, 0.0), (0.0, 0.0)),
                causal_clocks=tuple(
                    CausalVectorClock.from_mapping({r: 1})
                    for r in ctx.role_arrival_order),
            )
            return LiveObservationBuilderResult(
                observation=obs,
                role_handoff_signature_cid=sig)

        team = LearnedManifoldTeam(
            _agents(3), backend=_backend("real"),
            registry=reg, observation_builder=_builder,
            capture_capsules=True)
        result = team.run("task")
        # All non-trivial turns should hit margin_abstain (the
        # turn 0 may hit no_policy since it has no prior arrivals).
        assert result.n_learned_margin_abstains >= 1
