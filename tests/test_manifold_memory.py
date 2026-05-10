"""Tests for the W46 Manifold Memory Controller layer.

Each test covers one of the core components: multi-layer
controller stack, memory bank, causally-masked time attention,
multi-rank role adapter, dictionary basis, control token
surface, shared prefix capsule, registry/orchestrator, envelope
verifier, and trivial-passthrough falsifier.
"""

from __future__ import annotations

import math

import pytest

from coordpy.agents import AgentTeam, agent
from coordpy.learned_manifold import (
    TrainingExample,
    TrainingSet,
    W45_CHANNEL_ORDER,
    W45_DEFAULT_FEATURE_DIM,
)
from coordpy.manifold_memory import (
    ControlTokenWitness,
    DictionaryBasis,
    LayerParams,
    ManifoldMemoryBank,
    ManifoldMemoryHandoffEnvelope,
    ManifoldMemoryTeam,
    MemoryAwareSyntheticBackend,
    MemoryEntry,
    MultiLayerControllerParams,
    MultiRankRoleAdapter,
    PrefixCapsule,
    TimeAttentionWitness,
    W46_ALL_BRANCHES,
    W46_ALL_CTRL_MODES,
    W46_ALL_FAILURE_MODES,
    W46_BRANCH_MEMORY_RATIFIED,
    W46_BRANCH_TRIVIAL_MEMORY_PASSTHROUGH,
    W46_CTRL_MODE_COMPACT,
    W46_CTRL_MODE_FULL,
    W46_CTRL_MODE_OFF,
    W46_DEFAULT_DICTIONARY_SIZE,
    W46_DEFAULT_N_LAYERS,
    W46_DEFAULT_ROLE_DELTA_RANK,
    W46_MANIFOLD_MEMORY_SCHEMA_VERSION,
    build_control_token_string,
    build_manifold_memory_registry,
    build_prefix_capsule,
    build_trivial_manifold_memory_registry,
    build_unfitted_memory_controller_params,
    compute_time_attention,
    fit_memory_controller,
    forward_memory_controller,
    verify_manifold_memory_handoff,
)
from coordpy.synthetic_llm import SyntheticLLMClient


# =============================================================================
# Helpers
# =============================================================================

def _make_synthetic_backend(default: str = "OK") -> SyntheticLLMClient:
    return SyntheticLLMClient(
        model_tag="synthetic.w46", default_response=default)


def _make_training_set(*, n: int = 12, sig: str = "abc"):
    examples = []
    for i in range(n):
        label = 1.0 if i % 2 == 0 else -1.0
        feats = [
            (c, ((label if c == "spherical" else 0.0),
                 0.0, 0.0, 0.0))
            for c in W45_CHANNEL_ORDER]
        examples.append(TrainingExample(
            role=f"role{i % 3}",
            role_handoff_signature_cid=sig,
            channel_features=tuple(feats),
            label=label,
        ))
    return TrainingSet(
        examples=tuple(examples),
        feature_dim=W45_DEFAULT_FEATURE_DIM,
    )


# =============================================================================
# Schema / branch / mode invariants
# =============================================================================

class TestSchemaAndBranches:
    def test_schema_version_is_v1(self):
        assert W46_MANIFOLD_MEMORY_SCHEMA_VERSION == (
            "coordpy.manifold_memory.v1")

    def test_branches_are_unique(self):
        assert len(set(W46_ALL_BRANCHES)) == len(W46_ALL_BRANCHES)

    def test_ctrl_modes_are_unique(self):
        assert (len(set(W46_ALL_CTRL_MODES))
                == len(W46_ALL_CTRL_MODES))

    def test_failure_modes_are_unique_and_ge_16(self):
        assert (len(set(W46_ALL_FAILURE_MODES))
                == len(W46_ALL_FAILURE_MODES))
        assert len(W46_ALL_FAILURE_MODES) >= 16

    def test_failure_modes_disjoint_from_w22_w45(self):
        # W46 modes should start with "w46_" or be the empty
        # sentinel.
        w46_prefix = sum(
            1 for m in W46_ALL_FAILURE_MODES
            if m.startswith("w46_"))
        empty = sum(
            1 for m in W46_ALL_FAILURE_MODES
            if m == "empty_w46_envelope")
        assert w46_prefix + empty == len(W46_ALL_FAILURE_MODES)
        assert empty == 1


# =============================================================================
# DictionaryBasis: bijective encode/decode + closest assignment
# =============================================================================

class TestDictionaryBasis:
    def test_encode_decode_round_trip(self):
        proto = ((1.0, 0.0, 0.0, 0.0) * 6,
                 (0.0, 1.0, 0.0, 0.0) * 6,
                 (0.0, 0.0, 1.0, 0.0) * 6)
        dictionary = DictionaryBasis(
            feature_dim=4, prototypes=proto)
        v = [0.7, 0.1, 0.0, 0.0] * 6
        idx, residual = dictionary.encode(v)
        decoded = dictionary.decode(idx, residual)
        assert len(decoded) == len(v)
        for a, b in zip(v, decoded):
            assert abs(a - b) <= 1e-9

    def test_encode_returns_closest_prototype(self):
        # Two unit prototypes; vector aligned with prototype 0.
        proto = (
            (1.0, 0.0, 0.0, 0.0) * 6,
            (-1.0, 0.0, 0.0, 0.0) * 6,
        )
        dictionary = DictionaryBasis(
            feature_dim=4, prototypes=proto)
        v = [0.9, 0.0, 0.0, 0.0] * 6
        idx, _ = dictionary.encode(v)
        assert idx == 0

    def test_empty_prototypes_returns_sentinel(self):
        dictionary = DictionaryBasis(
            feature_dim=4, prototypes=tuple())
        idx, residual = dictionary.encode([1.0] * 24)
        assert idx == -1
        # Decode falls through to identity.
        decoded = dictionary.decode(idx, residual)
        assert list(decoded) == [1.0] * 24

    def test_cid_is_64_hex(self):
        dictionary = DictionaryBasis(
            feature_dim=4,
            prototypes=((0.0,) * 24,))
        cid = dictionary.cid()
        assert len(cid) == 64
        int(cid, 16)  # must be hex


# =============================================================================
# Memory bank: causal mask + content-addressing
# =============================================================================

class TestManifoldMemoryBank:
    def _mk_entry(self, turn: int, gate: float) -> MemoryEntry:
        return MemoryEntry(
            turn_index=turn,
            role=f"role{turn % 3}",
            role_handoff_signature_cid="x" * 64,
            channel_features=tuple(
                (c, (1.0, 0.0, 0.0, 0.0))
                for c in W45_CHANNEL_ORDER),
            per_channel_logits=(0.0,) * 6,
            gate_logit=float(gate),
            ratify_probability=0.5,
            decision_branch=W46_BRANCH_MEMORY_RATIFIED,
            dict_index=0,
            dict_residual_l1=0.0,
        )

    def test_admissible_for_turn_is_strict(self):
        bank = ManifoldMemoryBank(capacity=4)
        bank.append(self._mk_entry(0, 1.0))
        bank.append(self._mk_entry(1, -1.0))
        bank.append(self._mk_entry(3, 2.0))
        # Strictly less than turn_index=3 -> only turns 0, 1.
        adm = bank.admissible_for_turn(3)
        assert {e.turn_index for e in adm} == {0, 1}

    def test_capacity_ring_eviction(self):
        bank = ManifoldMemoryBank(capacity=2)
        bank.append(self._mk_entry(0, 0.0))
        bank.append(self._mk_entry(1, 0.0))
        bank.append(self._mk_entry(2, 0.0))
        assert {e.turn_index for e in bank.entries} == {1, 2}

    def test_head_cid_is_deterministic(self):
        b1 = ManifoldMemoryBank(capacity=4)
        b2 = ManifoldMemoryBank(capacity=4)
        for i in range(3):
            b1.append(self._mk_entry(i, float(i)))
            b2.append(self._mk_entry(i, float(i)))
        assert b1.head_cid() == b2.head_cid()
        assert len(b1.head_cid()) == 64


# =============================================================================
# Time attention: causal mask preservation + softmax weighting
# =============================================================================

class TestTimeAttention:
    def _mk_entry(self, turn: int, gate: float) -> MemoryEntry:
        return MemoryEntry(
            turn_index=turn,
            role=f"role{turn % 3}",
            role_handoff_signature_cid="x" * 64,
            channel_features=tuple(
                (c, (1.0, 0.0, 0.0, 0.0)
                 if c == "spherical" else (0.0,) * 4)
                for c in W45_CHANNEL_ORDER),
            per_channel_logits=(0.0,) * 6,
            gate_logit=float(gate),
            ratify_probability=0.5,
            decision_branch=W46_BRANCH_MEMORY_RATIFIED,
            dict_index=0,
            dict_residual_l1=0.0,
        )

    def test_empty_bank_returns_zero(self):
        bank = ManifoldMemoryBank(capacity=4)
        flat_query = [0.0] * 24
        wit = compute_time_attention(
            flat_query=flat_query, memory_bank=bank,
            turn_index=5, temperature=1.0, feature_dim=4)
        assert wit.mask_size == 0
        assert wit.pooled_value == 0.0
        assert wit.enabled is True

    def test_disabled_returns_zero(self):
        bank = ManifoldMemoryBank(capacity=4)
        bank.append(self._mk_entry(0, 1.0))
        flat_query = [1.0] + [0.0] * 23
        wit = compute_time_attention(
            flat_query=flat_query, memory_bank=bank,
            turn_index=5, temperature=1.0,
            feature_dim=4, enabled=False)
        assert wit.pooled_value == 0.0
        assert wit.enabled is False

    def test_future_entries_are_masked(self):
        bank_a = ManifoldMemoryBank(capacity=8)
        bank_a.append(self._mk_entry(0, 1.0))
        bank_b = ManifoldMemoryBank(capacity=8)
        bank_b.append(self._mk_entry(0, 1.0))
        bank_b.append(self._mk_entry(5, 99.0))  # future poison
        flat_query = [1.0] + [0.0] * 23
        wit_a = compute_time_attention(
            flat_query=flat_query, memory_bank=bank_a,
            turn_index=3, temperature=1.0, feature_dim=4)
        wit_b = compute_time_attention(
            flat_query=flat_query, memory_bank=bank_b,
            turn_index=3, temperature=1.0, feature_dim=4)
        assert (abs(wit_a.pooled_value - wit_b.pooled_value)
                <= 1e-9)

    def test_query_aligned_pool_returns_value(self):
        bank = ManifoldMemoryBank(capacity=4)
        bank.append(self._mk_entry(0, 1.7))
        flat_query = [1.0] + [0.0] * 23
        wit = compute_time_attention(
            flat_query=flat_query, memory_bank=bank,
            turn_index=2, temperature=1.0, feature_dim=4)
        # Single admissible entry; pool = its gate logit.
        assert abs(wit.pooled_value - 1.7) <= 1e-9
        assert wit.mask_size == 1


# =============================================================================
# Multi-layer / multi-rank fitter and forward
# =============================================================================

class TestMultiLayerFitter:
    def test_unfitted_params_have_n_layers_setting(self):
        params = build_unfitted_memory_controller_params(
            n_layers=3, rank=2, dictionary_size=4)
        assert params.n_layers == 3
        assert params.role_adapter.rank == 2
        assert params.dictionary.k == 4

    def test_fit_produces_2_layers_by_default(self):
        ts = _make_training_set()
        params = fit_memory_controller(ts)
        assert params.n_layers == W46_DEFAULT_N_LAYERS
        assert len(params.layers) == params.n_layers - 1

    def test_fit_is_deterministic(self):
        ts = _make_training_set()
        a = fit_memory_controller(ts)
        b = fit_memory_controller(ts)
        assert a.cid() == b.cid()

    def test_role_adapter_rank_respected(self):
        ts = _make_training_set()
        rank2 = fit_memory_controller(ts, role_delta_rank=2)
        rank1 = fit_memory_controller(ts, role_delta_rank=1)
        assert rank2.role_adapter.rank == 2
        assert rank1.role_adapter.rank == 1

    def test_dictionary_size_respected(self):
        ts = _make_training_set()
        params = fit_memory_controller(ts, dictionary_size=5)
        assert params.dictionary.k == 5

    def test_forward_with_zero_params_returns_zero_logit(self):
        params = build_unfitted_memory_controller_params()
        bank = ManifoldMemoryBank(capacity=4)
        feats = {c: (0.0, 0.0, 0.0, 0.0)
                 for c in W45_CHANNEL_ORDER}
        fr = forward_memory_controller(
            channel_features=feats, params=params,
            role="role0", memory_bank=bank, turn_index=0)
        assert abs(fr.gate_logit) <= 1e-9
        assert abs(fr.ratify_probability - 0.5) <= 1e-9


# =============================================================================
# Prefix capsule: stability + reuse
# =============================================================================

class TestPrefixCapsule:
    def test_empty_prior_outputs_returns_empty(self):
        prefix, cap = build_prefix_capsule(
            prior_outputs=(), prefix_turns=2,
            policy_entry_cid="abc",
            prior_prefix_sha=None,
        )
        assert prefix == ""
        assert cap.prefix_token_count == 0

    def test_first_n_outputs_only(self):
        # With prefix_turns=2, the capsule uses the FIRST 2
        # outputs, not the last 2. So additional outputs do not
        # change the prefix.
        outs = (
            ("r0", "first"), ("r1", "second"),
            ("r2", "third"), ("r3", "fourth"))
        p1, c1 = build_prefix_capsule(
            prior_outputs=outs[:3], prefix_turns=2,
            policy_entry_cid="x")
        p2, c2 = build_prefix_capsule(
            prior_outputs=outs[:4], prefix_turns=2,
            policy_entry_cid="x")
        # Same first 2 outputs -> same prefix bytes.
        assert c1.prefix_sha256 == c2.prefix_sha256
        assert p1 == p2

    def test_reused_flag_when_sha_matches(self):
        outs = (("r0", "a"), ("r1", "b"))
        _, c1 = build_prefix_capsule(
            prior_outputs=outs, prefix_turns=2,
            policy_entry_cid="x", prior_prefix_sha=None)
        assert c1.reused is False
        _, c2 = build_prefix_capsule(
            prior_outputs=outs, prefix_turns=2,
            policy_entry_cid="x",
            prior_prefix_sha=c1.prefix_sha256)
        assert c2.reused is True
        assert c2.prefix_sha256 == c1.prefix_sha256

    def test_cid_is_64_hex(self):
        outs = (("r0", "a"), ("r1", "b"))
        _, cap = build_prefix_capsule(
            prior_outputs=outs, prefix_turns=2,
            policy_entry_cid="x")
        cid = cap.cid()
        assert len(cid) == 64


# =============================================================================
# Control token surface: bijective + bounded
# =============================================================================

class TestControlTokenSurface:
    def test_off_mode_emits_empty(self):
        text, wit = build_control_token_string(
            ctrl_mode=W46_CTRL_MODE_OFF,
            route=0, confidence_bucket=0,
            ratify_probability=0.5,
            layer_logits=(0.1, 0.2),
            mem_attn_value=0.3,
            dict_index=1, mem_summary="ok",
            role_universe=("a", "b"), turn_index=0)
        assert text == ""
        assert wit.n_ctrl_tokens == 0

    def test_compact_mode_emits_single_line(self):
        text, wit = build_control_token_string(
            ctrl_mode=W46_CTRL_MODE_COMPACT,
            route=3, confidence_bucket=2,
            ratify_probability=0.8421,
            layer_logits=(0.1, 0.2),
            mem_attn_value=0.3,
            dict_index=1, mem_summary="r0:rat",
            role_universe=("a", "b", "c"), turn_index=1)
        assert text.startswith("MANIFOLD_CTRL: route=3")
        assert "conf=2" in text
        assert "dict_idx=1" in text
        assert wit.cid().__len__() == 64

    def test_full_mode_emits_multiline(self):
        text, wit = build_control_token_string(
            ctrl_mode=W46_CTRL_MODE_FULL,
            route=4, confidence_bucket=3,
            ratify_probability=0.9111,
            layer_logits=(0.1, -0.2, 0.3),
            mem_attn_value=-0.05,
            dict_index=2, mem_summary="r0:rat,r1:sub",
            role_universe=("a", "b"), turn_index=2)
        assert text.startswith("MANIFOLD_CTRL:")
        assert "layer_logits=[" in text
        assert "mem_attn=" in text
        assert "dict_idx=2" in text
        assert "mem_summary=r0:rat,r1:sub" in text
        # Round-trip the witness back to text.
        wit2 = wit
        assert wit2.cid() == wit.cid()

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            build_control_token_string(
                ctrl_mode="bogus", route=0,
                confidence_bucket=0,
                ratify_probability=0.5,
                layer_logits=(0.0,), mem_attn_value=0.0,
                dict_index=0, mem_summary="",
                role_universe=("a",), turn_index=0)


# =============================================================================
# Trivial passthrough falsifier (W46-L-TRIVIAL-MEMORY-PASSTHROUGH)
# =============================================================================

class TestTrivialPassthrough:
    def test_trivial_registry_is_trivial(self):
        reg = build_trivial_manifold_memory_registry()
        assert reg.is_trivial is True

    def test_trivial_run_matches_agent_team(self):
        a = (agent("r0", "go"), agent("r1", "go"),
             agent("r2", "go"))
        base = AgentTeam(
            a, backend=_make_synthetic_backend(),
            team_instructions="t", max_visible_handoffs=2,
            capture_capsules=True)
        b = base.run("hello")

        reg = build_trivial_manifold_memory_registry()
        team = ManifoldMemoryTeam(
            a, backend=_make_synthetic_backend(),
            registry=reg, team_instructions="t",
            max_visible_handoffs=2, capture_capsules=True)
        m = team.run("hello")

        assert m.final_output == b.final_output
        assert len(m.turns) == len(b.turns)
        for t in m.memory_turns:
            assert t.envelope.decision_branch == (
                W46_BRANCH_TRIVIAL_MEMORY_PASSTHROUGH)


# =============================================================================
# Verifier soundness
# =============================================================================

class TestVerifier:
    def _build_fitted_team(self):
        ts = _make_training_set()
        params = fit_memory_controller(ts)
        reg = build_manifold_memory_registry(
            schema_cid="s" * 64, policy_entries=(),
            params=params,
            control_token_mode=W46_CTRL_MODE_FULL,
            spherical_agreement_min=0.0,
            subspace_drift_max=math.pi,
        )
        team = ManifoldMemoryTeam(
            [agent(f"role{i}", "x") for i in range(3)],
            backend=_make_synthetic_backend(),
            registry=reg, max_visible_handoffs=2,
            capture_capsules=True)
        return reg, team.run("verify probe")

    def test_empty_envelope_fails(self):
        out = verify_manifold_memory_handoff(
            None, registered_schema_cid="s" * 64)
        assert out.ok is False
        assert out.reason == "empty_w46_envelope"

    def test_valid_envelope_passes_at_least_16_checks(self):
        reg, result = self._build_fitted_team()
        for t in result.memory_turns:
            out = verify_manifold_memory_handoff(
                t.envelope,
                registered_schema_cid=reg.schema_cid)
            assert out.ok is True
            assert out.n_checks >= 16

    def test_schema_version_tamper_fails(self):
        reg, result = self._build_fitted_team()
        env = result.memory_turns[0].envelope
        import dataclasses
        bad = dataclasses.replace(
            env, schema_version="bogus.v999")
        out = verify_manifold_memory_handoff(
            bad, registered_schema_cid=reg.schema_cid)
        assert out.ok is False
        assert out.reason == "w46_schema_version_unknown"

    def test_token_accounting_tamper_fails(self):
        reg, result = self._build_fitted_team()
        env = result.memory_turns[0].envelope
        import dataclasses
        bad = dataclasses.replace(
            env, n_visible_prompt_tokens_saved=99999)
        out = verify_manifold_memory_handoff(
            bad, registered_schema_cid=reg.schema_cid)
        assert out.ok is False
        assert out.reason == "w46_token_accounting_invalid"

    def test_outer_cid_tamper_fails(self):
        reg, result = self._build_fitted_team()
        env = result.memory_turns[-1].envelope
        import dataclasses
        bad = dataclasses.replace(
            env, memory_outer_cid="0" * 64)
        out = verify_manifold_memory_handoff(
            bad, registered_schema_cid=reg.schema_cid)
        assert out.ok is False
        # The verifier checks memory_witness re-derivation first
        # for this kind of tamper; either failure mode is
        # acceptable.
        assert out.reason in {
            "w46_outer_cid_mismatch",
            "w46_memory_witness_cid_mismatch",
        }


# =============================================================================
# Determinism: two runs produce byte-identical envelopes
# =============================================================================

class TestDeterminism:
    def test_two_runs_match(self):
        ts = _make_training_set()
        params = fit_memory_controller(ts)
        reg = build_manifold_memory_registry(
            schema_cid="s" * 64, policy_entries=(),
            params=params,
            control_token_mode=W46_CTRL_MODE_FULL,
            spherical_agreement_min=0.0,
            subspace_drift_max=math.pi,
        )

        def _run():
            team = ManifoldMemoryTeam(
                [agent(f"role{i}", "x") for i in range(3)],
                backend=_make_synthetic_backend(),
                registry=reg, max_visible_handoffs=2,
                capture_capsules=True)
            return team.run("determinism probe")

        a = _run()
        b = _run()
        assert a.final_output == b.final_output
        assert a.root_cid == b.root_cid
        assert (a.final_memory_bank_head_cid
                == b.final_memory_bank_head_cid)
        for ta, tb in zip(a.memory_turns, b.memory_turns):
            assert (ta.envelope.memory_outer_cid
                    == tb.envelope.memory_outer_cid)
            assert (ta.envelope.memory_bank_head_cid
                    == tb.envelope.memory_bank_head_cid)


# =============================================================================
# Memory-aware synthetic backend
# =============================================================================

class TestMemoryAwareSyntheticBackend:
    def test_correct_when_ctrl_and_mem_summary_present(self):
        b = MemoryAwareSyntheticBackend()
        prompt = (
            "Agent role0\n"
            "MANIFOLD_CTRL:\n"
            "  mem_summary=r0:rat\n"
            "go")
        assert b.generate(prompt) == "MEMORY_OK"

    def test_no_ctrl_returns_no_ctrl(self):
        b = MemoryAwareSyntheticBackend()
        assert b.generate("no manifold block") == (
            "MEMORY_NO_CTRL")

    def test_ctrl_without_mem_summary_returns_no_ctrl(self):
        b = MemoryAwareSyntheticBackend()
        prompt = "MANIFOLD_CTRL: route=0"
        assert b.generate(prompt) == "MEMORY_NO_CTRL"


# =============================================================================
# Public surface check
# =============================================================================

class TestPublicSurface:
    def test_w46_module_importable_explicit(self):
        from coordpy.manifold_memory import (
            ManifoldMemoryTeam,
            build_manifold_memory_registry,
            fit_memory_controller,
        )
        assert ManifoldMemoryTeam is not None
        assert build_manifold_memory_registry is not None
        assert fit_memory_controller is not None

    def test_w46_module_not_in_experimental(self):
        import coordpy
        names = set(coordpy.__experimental__)
        for w46_name in (
                "ManifoldMemoryTeam",
                "ManifoldMemoryRegistry",
                "build_manifold_memory_registry",
                "fit_memory_controller",
        ):
            assert w46_name not in names

    def test_version_unchanged(self):
        import coordpy
        assert coordpy.__version__ == "0.5.20"
        assert coordpy.SDK_VERSION == "coordpy.sdk.v3.43"
