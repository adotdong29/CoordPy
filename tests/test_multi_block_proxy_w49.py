"""Tests for the W49 Multi-Block Cross-Bank Coordination layer.

Per-component unit coverage: multi-block proxy stack (depth +
residuals), multi-bank pseudo-KV (per-role + shared), bank router,
bank mix gate, learned eviction policy, retention head, dictionary
codebook compression, shared-latent capsule, cramming witness,
trivial-passthrough falsifier, envelope verifier, train
determinism, public surface.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math

import pytest

from coordpy.agents import AgentTeam, agent
from coordpy.autograd_manifold import Variable
from coordpy.learned_manifold import (
    W45_CHANNEL_ORDER,
    W45_DEFAULT_FEATURE_DIM,
)
from coordpy.multi_block_proxy import (
    BankMixGate,
    BankRouter,
    CrammingWitness,
    DictionaryCodebook,
    EvictionPolicy,
    FeedForwardBlock,
    LatentControlV2Witness,
    MultiBankPseudoKV,
    MultiBlockAwareSyntheticBackend,
    MultiBlockExample,
    MultiBlockProxyHandoffEnvelope,
    MultiBlockProxyParams,
    MultiBlockProxyStack,
    MultiBlockProxyTeam,
    MultiBlockTrainingSet,
    ProxyTransformerBlock,
    RetentionHead,
    SharedLatentCapsule,
    SharedLatentProjector,
    W49_ALL_BRANCHES,
    W49_ALL_FAILURE_MODES,
    W49_BRANCH_MULTI_BLOCK_RATIFIED,
    W49_BRANCH_TRIVIAL_MULTI_BLOCK_PASSTHROUGH,
    W49_DEFAULT_DICTIONARY_SIZE,
    W49_DEFAULT_N_BLOCKS,
    W49_MULTI_BLOCK_PROXY_SCHEMA_VERSION,
    build_cramming_witness,
    build_latent_control_v2_string,
    build_multi_block_proxy_registry,
    build_trivial_multi_block_proxy_registry,
    build_unfitted_multi_block_proxy_params,
    fit_multi_block_proxy,
    forward_multi_block_proxy,
    verify_multi_block_proxy_handoff,
)
from coordpy.product_manifold import (
    ProductManifoldPolicyEntry,
    encode_spherical_consensus,
    encode_subspace_basis,
)
from coordpy.shared_state_proxy import (
    PseudoKVBank,
    PseudoKVSlot,
    SharedStateProxyTeam,
    build_trivial_shared_state_proxy_registry,
)
from coordpy.synthetic_llm import SyntheticLLMClient


def _make_synthetic_backend(default: str = "OK") -> SyntheticLLMClient:
    return SyntheticLLMClient(
        model_tag="synthetic.w49", default_response=default)


def _make_training_set(
        *, n: int = 8,
        feature_dim: int = W45_DEFAULT_FEATURE_DIM,
) -> MultiBlockTrainingSet:
    examples = []
    for i in range(n):
        label = 1.0 if i % 2 == 0 else -1.0
        feats = [
            (c, ((label if c == "spherical" else 0.0),
                 0.0, 0.0, 0.0))
            for c in W45_CHANNEL_ORDER]
        examples.append(MultiBlockExample(
            role=f"role{i % 2}",
            channel_features=tuple(feats),
            branch_id=i % 2, cycle_id=0,
            label=label,
            retention_label=1.0 if label > 0 else 0.0,
            dictionary_target=i % 4,
            eviction_target=0.5,
            target_fact_hash=(label, 0.0, 0.0, 0.0),
        ))
    return MultiBlockTrainingSet(
        examples=tuple(examples),
        feature_dim=int(feature_dim))


# =============================================================================
# Schema, branches, failure modes
# =============================================================================

class TestSchemaAndBranches:
    def test_schema_version_is_v1(self):
        assert (W49_MULTI_BLOCK_PROXY_SCHEMA_VERSION
                == "coordpy.multi_block_proxy.v1")

    def test_all_branches_disjoint(self):
        assert len(W49_ALL_BRANCHES) == len(set(W49_ALL_BRANCHES))

    def test_failure_modes_disjoint(self):
        assert len(W49_ALL_FAILURE_MODES) == len(
            set(W49_ALL_FAILURE_MODES))

    def test_failure_modes_cover_at_least_18(self):
        # H10 success bar: ≥ 18 failure modes.
        assert len(W49_ALL_FAILURE_MODES) >= 18


# =============================================================================
# Feed-forward block + multi-block stack
# =============================================================================

class TestFeedForwardBlock:
    def test_forward_shape(self):
        ffn = FeedForwardBlock.init(
            in_dim=4, hidden_dim=8, seed=0)
        out = ffn.forward_value([1.0, 0.5, -0.5, 0.0])
        assert len(out) == 4

    def test_params_count(self):
        ffn = FeedForwardBlock.init(
            in_dim=4, hidden_dim=8, seed=0)
        # 4 ParamTensors: w1, b1, w2, b2.
        assert len(ffn.params()) == 4

    def test_cid_stable(self):
        ffn = FeedForwardBlock.init(
            in_dim=4, hidden_dim=8, seed=0)
        assert ffn.cid() == ffn.cid()


class TestProxyTransformerBlock:
    def test_forward_shape(self):
        blk = ProxyTransformerBlock.init(
            in_dim=8, factor_dim=4, n_heads=2,
            ffn_hidden_dim=8, seed=0)
        out = blk.forward_value(
            query_input=[1.0] * 8,
            slot_keys=[[1.0] * 8, [-1.0] * 8],
            slot_values=[[0.5] * 8, [-0.5] * 8],
        )
        assert len(out) == 8


class TestMultiBlockProxyStack:
    def test_two_blocks_compose(self):
        stack = MultiBlockProxyStack.init(
            n_blocks=2, in_dim=8, factor_dim=4, n_heads=2,
            seed=0)
        out = stack.forward_value(
            query_input=[1.0] * 8,
            slot_keys=[[1.0] * 8],
            slot_values=[[0.5] * 8],
        )
        assert len(out) == 8

    def test_distinct_block_cids(self):
        stack = MultiBlockProxyStack.init(
            n_blocks=2, in_dim=8, factor_dim=4, n_heads=2,
            seed=0)
        assert stack.blocks[0].cid() != stack.blocks[1].cid()


# =============================================================================
# Multi-bank pseudo-KV
# =============================================================================

class TestMultiBankPseudoKV:
    def test_role_banks_isolated(self):
        mb = MultiBankPseudoKV(
            role_capacity=4, shared_capacity=4, factor_dim=4)
        a = mb.get_or_init_role_bank("role0")
        b = mb.get_or_init_role_bank("role1")
        assert a is not b
        a.write(PseudoKVSlot(
            slot_index=0, turn_index=0, role="role0",
            key=(1.0,) * 4, value=(1.0,) * 4,
            write_gate_value=1.0,
            source_observation_cid="x"))
        assert a.size == 1
        assert b.size == 0

    def test_head_cid_changes_with_writes(self):
        mb = MultiBankPseudoKV(
            role_capacity=4, shared_capacity=4, factor_dim=4)
        h0 = mb.head_cid()
        mb.get_or_init_role_bank("role0").write(PseudoKVSlot(
            slot_index=0, turn_index=0, role="role0",
            key=(1.0,) * 4, value=(1.0,) * 4,
            write_gate_value=1.0,
            source_observation_cid="x"))
        assert mb.head_cid() != h0

    def test_reset_clears(self):
        mb = MultiBankPseudoKV(
            role_capacity=4, shared_capacity=4, factor_dim=4)
        mb.get_or_init_role_bank("role0").write(PseudoKVSlot(
            slot_index=0, turn_index=0, role="role0",
            key=(1.0,) * 4, value=(1.0,) * 4,
            write_gate_value=1.0,
            source_observation_cid="x"))
        mb.reset()
        assert mb.total_size() == 0


# =============================================================================
# Bank router + mix gate + eviction
# =============================================================================

class TestBankRouter:
    def test_sigmoid_in_unit_interval(self):
        r = BankRouter.init(in_dim=4, seed=0)
        v = r.forward_value([1.0, 0.5, -0.5, 0.0])
        assert 0.0 <= v <= 1.0


class TestBankMixGate:
    def test_sigmoid_in_unit_interval(self):
        g = BankMixGate.init(in_dim=4, seed=0)
        v = g.forward_value([1.0, 0.5, -0.5, 0.0])
        assert 0.0 <= v <= 1.0


class TestEvictionPolicy:
    def test_evict_picks_lowest_score(self):
        pol = EvictionPolicy.init(in_dim=3, seed=0)
        # Weight write_gate positive => slots with HIGH write_gate
        # get HIGH score, get kept; LOW write_gate evicted.
        pol.w_evict.update_values([0.0, 0.0, 4.0])
        bank = PseudoKVBank(capacity=3, factor_dim=4)
        bank.write(PseudoKVSlot(
            slot_index=0, turn_index=0, role="role0",
            key=(0.0,) * 4, value=(0.0,) * 4,
            write_gate_value=0.95,
            source_observation_cid="signal"))
        bank.write(PseudoKVSlot(
            slot_index=1, turn_index=1, role="role0",
            key=(0.0,) * 4, value=(0.0,) * 4,
            write_gate_value=0.05,
            source_observation_cid="noise"))
        # Lowest-score slot should be the noise one.
        idx = pol.evict_index(
            bank=bank, current_role="role0", current_turn=2)
        assert idx == 1

    def test_evict_empty_bank(self):
        pol = EvictionPolicy.init(in_dim=3, seed=0)
        empty = PseudoKVBank(capacity=3, factor_dim=4)
        idx = pol.evict_index(
            bank=empty, current_role="role0", current_turn=0)
        assert idx == -1


# =============================================================================
# Retention head
# =============================================================================

class TestRetentionHead:
    def test_forward_in_unit_interval(self):
        rh = RetentionHead.init(in_dim=8, hidden_dim=4, seed=0)
        v = rh.forward_value([0.5] * 8)
        assert 0.0 <= v <= 1.0


# =============================================================================
# Dictionary codebook
# =============================================================================

class TestDictionaryCodebook:
    def test_encode_returns_valid_code(self):
        cb = DictionaryCodebook.init(
            n_codes=8, code_dim=4, seed=0)
        code = cb.encode_value([0.5, 0.3, -0.2, 0.1])
        assert 0 <= code < 8

    def test_code_bits_log2(self):
        cb = DictionaryCodebook.init(
            n_codes=8, code_dim=4, seed=0)
        assert cb.code_bits() == 3

    def test_code_vector_returns_proto(self):
        cb = DictionaryCodebook.init(
            n_codes=4, code_dim=4, seed=0)
        v = cb.code_vector(2)
        assert len(v) == 4


class TestLatentControlV2:
    def test_round_trip(self):
        mask = (True, False, True)
        bits = (1, 0, 1)
        text, witness = build_latent_control_v2_string(
            ctrl_tag="LATENT_CTRL_V2",
            dictionary_code=5, code_bits=3,
            emit_mask=mask, bits_payload=bits,
            shared_latent_hash_short="abcd1234efef")
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert sha == witness.ctrl_bytes_sha256

    def test_compact_when_empty_mask(self):
        text, witness = build_latent_control_v2_string(
            ctrl_tag="LATENT_CTRL_V2",
            dictionary_code=5, code_bits=3,
            emit_mask=tuple(), bits_payload=tuple(),
            shared_latent_hash_short="abcd1234efef")
        assert "mask=" not in text
        assert "bits=" not in text
        assert "code=5/3b" in text


# =============================================================================
# Shared-latent capsule + projector
# =============================================================================

class TestSharedLatentCapsule:
    def test_evolving_cid(self):
        cap_a = SharedLatentCapsule(
            turn_index=0, role="role0", dim=4,
            values=(0.1, 0.2, 0.3, 0.4),
            parent_capsule_cid="")
        cap_b = SharedLatentCapsule(
            turn_index=1, role="role0", dim=4,
            values=(0.5, 0.6, 0.7, 0.8),
            parent_capsule_cid=cap_a.cid())
        assert cap_a.cid() != cap_b.cid()

    def test_hash_short_is_12_hex(self):
        cap = SharedLatentCapsule(
            turn_index=0, role="role0", dim=4,
            values=(0.1, 0.2, 0.3, 0.4),
            parent_capsule_cid="")
        h = cap.hash_short()
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)


class TestSharedLatentProjector:
    def test_forward_shape(self):
        proj = SharedLatentProjector.init(
            in_dim=8, out_dim=4, seed=0)
        out = proj.forward_value([0.5] * 8)
        assert len(out) == 4
        # Tanh output bounded.
        for v in out:
            assert -1.0 <= v <= 1.0


# =============================================================================
# Cramming witness
# =============================================================================

class TestCrammingWitness:
    def test_structured_bits_sum(self):
        cw = build_cramming_witness(
            dictionary_code=4, code_bits=3,
            emit_mask=(True, False, True),
            bits_payload=(1, 0, 1, 0),
            visible_ctrl_tokens=5,
            visible_latent_header_tokens=2,
            shared_latent_capsule_bytes=128,
        )
        # 3 code + 3 mask + 4 bits = 10
        assert cw.structured_bits == 10
        # bits per token = 10 / 7
        assert abs(cw.bits_per_visible_token - 10.0 / 7.0) < 1e-9

    def test_cid_stable(self):
        cw = build_cramming_witness(
            dictionary_code=4, code_bits=3,
            emit_mask=(True,), bits_payload=(1,),
            visible_ctrl_tokens=4,
            visible_latent_header_tokens=2,
            shared_latent_capsule_bytes=128,
        )
        assert cw.cid() == cw.cid()


# =============================================================================
# Trivial-passthrough falsifier
# =============================================================================

class TestTrivialPassthrough:
    def test_trivial_w49_reduces_to_w48_team(self):
        ag = [agent(f"role{i}", "instr", max_tokens=32,
                    temperature=0.0)
              for i in range(3)]
        w48 = SharedStateProxyTeam(
            ag, backend=_make_synthetic_backend(),
            registry=build_trivial_shared_state_proxy_registry(),
            max_visible_handoffs=2, capture_capsules=True,
        ).run("test")
        w49 = MultiBlockProxyTeam(
            ag, backend=_make_synthetic_backend(),
            registry=build_trivial_multi_block_proxy_registry(),
            max_visible_handoffs=2, capture_capsules=True,
        ).run("test")
        assert w49.final_output == w48.final_output
        assert len(w49.turns) == len(w48.turns)
        for t in w49.multi_block_turns:
            assert t.envelope.decision_branch == (
                W49_BRANCH_TRIVIAL_MULTI_BLOCK_PASSTHROUGH)


# =============================================================================
# Forward + fit
# =============================================================================

class TestForwardMultiBlockProxy:
    def test_forward_basic(self):
        ts = _make_training_set(n=4)
        params = build_unfitted_multi_block_proxy_params(
            feature_dim=ts.feature_dim)
        ex = ts.examples[0]
        bank = MultiBankPseudoKV(
            role_capacity=4, shared_capacity=4,
            factor_dim=int(params.inner_w48.factor_dim))
        fr, cap = forward_multi_block_proxy(
            channel_features=ex.channel_features_map,
            params=params, role=str(ex.role),
            multi_bank=bank, turn_index=0,
            target_fact_hash=ex.target_fact_hash,
        )
        assert isinstance(cap, SharedLatentCapsule)
        assert 0.0 <= fr.ratify_probability <= 1.0


class TestFitMultiBlockProxy:
    def test_fit_replay_determinism(self):
        ts = _make_training_set(n=6)
        a = fit_multi_block_proxy(ts, n_steps=10, seed=42)
        b = fit_multi_block_proxy(ts, n_steps=10, seed=42)
        assert a.cid() == b.cid()
        assert a.training_trace_cid == b.training_trace_cid


# =============================================================================
# Envelope verifier
# =============================================================================

class TestVerifier:
    def _build_run(self):
        SCHEMA = hashlib.sha256(
            b"w49.test.schema").hexdigest()
        sig = hashlib.sha256(b"sig").hexdigest()
        policy = ProductManifoldPolicyEntry(
            role_handoff_signature_cid=sig,
            expected_services=("memory",),
            expected_spherical=encode_spherical_consensus(
                ("event", "summary")),
            expected_subspace=encode_subspace_basis(
                ((1.0, 0.0), (0.0, 1.0),
                 (0.0, 0.0), (0.0, 0.0))),
            expected_causal_topology_hash="(...)",
        )
        reg = build_multi_block_proxy_registry(
            schema_cid=SCHEMA, policy_entries=(policy,),
            margin_abstain_threshold=-99.0,
            spherical_agreement_min=0.5,
            subspace_drift_max=math.pi,
        )
        ag = [agent(f"role{i}", "i", max_tokens=32,
                    temperature=0.0)
              for i in range(3)]
        team = MultiBlockProxyTeam(
            ag, backend=_make_synthetic_backend(),
            registry=reg, max_visible_handoffs=2,
            capture_capsules=True,
        )
        return reg, team.run("verifier probe")

    def test_clean_envelope_verifies(self):
        reg, r = self._build_run()
        env = r.multi_block_turns[-1].envelope
        out = verify_multi_block_proxy_handoff(
            env, registered_schema_cid=reg.schema_cid,
            registered_multi_block_params_cid=reg.params.cid())
        assert out.ok, out.reason

    def test_schema_mismatch_caught(self):
        reg, r = self._build_run()
        env = r.multi_block_turns[-1].envelope
        forged = dataclasses.replace(
            env, schema_cid="z" * 64)
        out = verify_multi_block_proxy_handoff(
            forged, registered_schema_cid=reg.schema_cid,
            registered_multi_block_params_cid=reg.params.cid())
        assert not out.ok

    def test_dictionary_cid_tamper_caught(self):
        reg, r = self._build_run()
        env = r.multi_block_turns[-1].envelope
        forged = dataclasses.replace(
            env, dictionary_cid="0" * 64)
        out = verify_multi_block_proxy_handoff(
            forged, registered_schema_cid=reg.schema_cid,
            registered_multi_block_params_cid=reg.params.cid())
        assert not out.ok

    def test_retention_head_cid_tamper_caught(self):
        reg, r = self._build_run()
        env = r.multi_block_turns[-1].envelope
        forged = dataclasses.replace(
            env, retention_head_cid="0" * 64)
        out = verify_multi_block_proxy_handoff(
            forged, registered_schema_cid=reg.schema_cid,
            registered_multi_block_params_cid=reg.params.cid())
        assert not out.ok

    def test_shared_latent_capsule_cid_tamper_caught(self):
        reg, r = self._build_run()
        env = r.multi_block_turns[-1].envelope
        forged = dataclasses.replace(
            env, shared_latent_capsule_cid="z" * 63)
        out = verify_multi_block_proxy_handoff(
            forged, registered_schema_cid=reg.schema_cid,
            registered_multi_block_params_cid=reg.params.cid())
        assert not out.ok


# =============================================================================
# Shared-latent chain walk
# =============================================================================

class TestSharedLatentChainWalk:
    def test_chain_walks_back_through_envelopes(self):
        SCHEMA = hashlib.sha256(
            b"w49.test.chain").hexdigest()
        sig = hashlib.sha256(b"sig").hexdigest()
        policy = ProductManifoldPolicyEntry(
            role_handoff_signature_cid=sig,
            expected_services=("memory",),
            expected_spherical=encode_spherical_consensus(
                ("event", "summary")),
            expected_subspace=encode_subspace_basis(
                ((1.0, 0.0), (0.0, 1.0),
                 (0.0, 0.0), (0.0, 0.0))),
            expected_causal_topology_hash="(...)",
        )
        reg = build_multi_block_proxy_registry(
            schema_cid=SCHEMA, policy_entries=(policy,),
            margin_abstain_threshold=-99.0,
            spherical_agreement_min=0.5,
            subspace_drift_max=math.pi,
        )
        ag = [agent(f"role{i}", "i", max_tokens=32,
                    temperature=0.0)
              for i in range(3)]
        team = MultiBlockProxyTeam(
            ag, backend=_make_synthetic_backend(),
            registry=reg, max_visible_handoffs=2,
            capture_capsules=True,
        )
        r = team.run("chain walk probe")
        # Walk: turn t's parent_cid must equal turn t-1's
        # shared_latent_capsule_cid.
        for i in range(1, len(r.multi_block_turns)):
            prev = r.multi_block_turns[
                i - 1].decision.shared_latent_capsule.cid()
            parent = r.multi_block_turns[
                i].envelope.shared_latent_parent_cid
            assert parent == prev


# =============================================================================
# Public surface
# =============================================================================

class TestPublicSurface:
    def test_all_exports(self):
        import coordpy.multi_block_proxy as m
        for name in (
                "MultiBlockProxyTeam", "MultiBlockProxyParams",
                "MultiBlockTrainingSet", "MultiBlockExample",
                "fit_multi_block_proxy",
                "forward_multi_block_proxy",
                "build_multi_block_proxy_registry",
                "build_trivial_multi_block_proxy_registry",
                "verify_multi_block_proxy_handoff",
                "W49_ALL_FAILURE_MODES",
                "MultiBlockAwareSyntheticBackend",
        ):
            assert hasattr(m, name), name

    def test_module_does_not_pollute_main_init(self):
        # The released SDK contract preservation: W49 NOT exported
        # from coordpy/__init__.py.
        import coordpy
        assert not hasattr(coordpy, "MultiBlockProxyTeam")
        assert not hasattr(coordpy, "build_multi_block_proxy_registry")
