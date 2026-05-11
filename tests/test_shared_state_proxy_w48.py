"""Tests for the W48 Shared-State Transformer-Proxy layer.

Per-component unit coverage: shared base state capsule, per-role
LoRA delta, pseudo-KV bank, multi-head proxy attention,
slot-memory write head, reconstruction decoder, branch/cycle
bias, latent control serializer, branch-history compressor,
training trace, orchestrator + team, envelope verifier,
trivial-passthrough falsifier.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math

import pytest

from coordpy.agents import AgentTeam, agent
from coordpy.autograd_manifold import (
    AutogradManifoldTeam,
    Variable,
    build_trivial_autograd_manifold_registry,
)
from coordpy.learned_manifold import (
    W45_CHANNEL_ORDER,
    W45_DEFAULT_FEATURE_DIM,
    W45_N_CHANNELS,
)
from coordpy.product_manifold import (
    ProductManifoldPolicyEntry,
    encode_spherical_consensus,
    encode_subspace_basis,
)
from coordpy.shared_state_proxy import (
    BranchCycleBias,
    BranchHistoryWitness,
    LatentControlSerializer,
    LatentControlWitness,
    MultiHeadProxyAttention,
    ProxyAttentionHead,
    PseudoKVBank,
    PseudoKVSlot,
    ReconstructionDecoder,
    RoleSharedStateDelta,
    SharedStateAwareSyntheticBackend,
    SharedStateCapsule,
    SharedStateExample,
    SharedStateProxyHandoffEnvelope,
    SharedStateProxyOrchestrator,
    SharedStateProxyParams,
    SharedStateProxyTeam,
    SharedStateTrainingSet,
    SlotMemoryWriteHead,
    W48_ALL_BRANCHES,
    W48_ALL_FAILURE_MODES,
    W48_BRANCH_PROXY_RATIFIED,
    W48_BRANCH_TRIVIAL_SHARED_STATE_PASSTHROUGH,
    W48_DEFAULT_FACTOR_DIM,
    W48_DEFAULT_LATENT_CTRL_BITS,
    W48_DEFAULT_N_BRANCHES,
    W48_DEFAULT_N_CYCLES,
    W48_DEFAULT_N_HEADS,
    W48_DEFAULT_PSEUDO_KV_SLOTS,
    W48_DEFAULT_SHARED_STATE_DIM,
    W48_SHARED_STATE_PROXY_SCHEMA_VERSION,
    build_latent_control_string,
    build_shared_state_proxy_registry,
    build_trivial_shared_state_proxy_registry,
    build_unfitted_shared_state_proxy_params,
    compress_branch_history,
    decompress_branch_history,
    fit_shared_state_proxy,
    forward_shared_state_proxy,
    verify_shared_state_proxy_handoff,
)
from coordpy.synthetic_llm import SyntheticLLMClient


def _make_synthetic_backend(default: str = "OK") -> SyntheticLLMClient:
    return SyntheticLLMClient(
        model_tag="synthetic.w48", default_response=default)


def _make_training_set(
        *, n: int = 8,
        feature_dim: int = W45_DEFAULT_FEATURE_DIM,
) -> SharedStateTrainingSet:
    examples = []
    for i in range(n):
        label = 1.0 if i % 2 == 0 else -1.0
        feats = [
            (c, ((label if c == "spherical" else 0.0),
                 0.0, 0.0, 0.0))
            for c in W45_CHANNEL_ORDER]
        examples.append(SharedStateExample(
            role=f"role{i % 3}",
            channel_features=tuple(feats),
            branch_id=i % W48_DEFAULT_N_BRANCHES,
            cycle_id=i % W48_DEFAULT_N_CYCLES,
            label=label,
            write_target=1.0 if label > 0 else 0.0,
        ))
    return SharedStateTrainingSet(
        examples=tuple(examples),
        feature_dim=int(feature_dim))


# =============================================================================
# Schema / branch invariants
# =============================================================================

class TestSchemaAndBranches:
    def test_schema_version_is_v1(self):
        assert W48_SHARED_STATE_PROXY_SCHEMA_VERSION == (
            "coordpy.shared_state_proxy.v1")

    def test_branches_are_unique(self):
        assert (len(set(W48_ALL_BRANCHES))
                == len(W48_ALL_BRANCHES))

    def test_failure_modes_ge_22(self):
        assert len(W48_ALL_FAILURE_MODES) >= 22

    def test_failure_modes_disjoint_from_w22_w47(self):
        w48_prefix = sum(
            1 for m in W48_ALL_FAILURE_MODES
            if m.startswith("w48_"))
        empty = sum(
            1 for m in W48_ALL_FAILURE_MODES
            if m == "empty_w48_envelope")
        assert w48_prefix + empty == len(W48_ALL_FAILURE_MODES)
        assert empty == 1


# =============================================================================
# Shared state capsule
# =============================================================================

class TestSharedStateCapsule:
    def test_init_shape(self):
        ss = SharedStateCapsule.init(dim=8, seed=0)
        assert ss.dim == 8
        assert len(ss.values) == 8

    def test_cid_stable_for_seed(self):
        a = SharedStateCapsule.init(dim=4, seed=0)
        b = SharedStateCapsule.init(dim=4, seed=0)
        assert a.cid() == b.cid()
        assert len(a.cid()) == 64

    def test_cid_differs_for_different_seeds(self):
        a = SharedStateCapsule.init(dim=4, seed=0)
        b = SharedStateCapsule.init(dim=4, seed=1)
        assert a.cid() != b.cid()

    def test_state_hash_short_12hex(self):
        ss = SharedStateCapsule.init(dim=4, seed=0)
        s = ss.state_hash_short()
        assert len(s) == 12
        int(s, 16)


# =============================================================================
# RoleSharedStateDelta
# =============================================================================

class TestRoleSharedStateDelta:
    def test_init_creates_per_role(self):
        rsd = RoleSharedStateDelta.init(
            roles=("a", "b"), rank=2, dim=8, seed=0)
        assert "a" in rsd.role_factors
        assert "b" in rsd.role_factors

    def test_unknown_role_returns_zero(self):
        rsd = RoleSharedStateDelta.init(
            roles=("a",), rank=2, dim=4, seed=0)
        d = rsd.forward_value(role="missing")
        assert d == (0.0, 0.0, 0.0, 0.0)

    def test_forward_vars_returns_dim_variables(self):
        rsd = RoleSharedStateDelta.init(
            roles=("a",), rank=2, dim=4, seed=0)
        vs = rsd.forward_vars(role="a")
        assert len(vs) == 4
        for v in vs:
            assert isinstance(v, Variable)

    def test_cid_64hex(self):
        rsd = RoleSharedStateDelta.init(
            roles=("a", "b"), rank=2, dim=4, seed=0)
        assert len(rsd.cid()) == 64


# =============================================================================
# Pseudo-KV bank
# =============================================================================

class TestPseudoKVBank:
    def test_write_appends_within_capacity(self):
        b = PseudoKVBank(capacity=3, factor_dim=4)
        for i in range(2):
            b.write(PseudoKVSlot(
                slot_index=i, turn_index=i, role="r",
                key=(1.0, 0.0, 0.0, 0.0),
                value=(0.0, 1.0, 0.0, 0.0),
                write_gate_value=0.7,
                source_observation_cid="x" * 64))
        assert b.size == 2

    def test_write_ring_buffer_when_full(self):
        b = PseudoKVBank(capacity=2, factor_dim=4)
        for i in range(3):
            b.write(PseudoKVSlot(
                slot_index=i, turn_index=i, role="r",
                key=(0.0,) * 4,
                value=(0.0,) * 4,
                write_gate_value=0.5,
                source_observation_cid="x" * 64))
        assert b.size == 2
        # First entry evicted; remaining indices renumbered.
        assert b.slots[0].turn_index == 1
        assert b.slots[1].turn_index == 2

    def test_admissible_for_turn_causal_mask(self):
        b = PseudoKVBank(capacity=4, factor_dim=4)
        for i in range(3):
            b.write(PseudoKVSlot(
                slot_index=i, turn_index=i, role="r",
                key=(0.0,) * 4,
                value=(0.0,) * 4,
                write_gate_value=0.5,
                source_observation_cid="x" * 64))
        adm = b.admissible_for_turn(2)
        assert len(adm) == 2
        for s in adm:
            assert s.turn_index < 2

    def test_head_cid_changes_when_slot_added(self):
        b = PseudoKVBank(capacity=3, factor_dim=4)
        h0 = b.head_cid()
        b.write(PseudoKVSlot(
            slot_index=0, turn_index=0, role="r",
            key=(0.0,) * 4,
            value=(0.0,) * 4,
            write_gate_value=0.5,
            source_observation_cid="x" * 64))
        h1 = b.head_cid()
        assert h0 != h1
        assert len(h1) == 64


# =============================================================================
# Multi-head proxy attention
# =============================================================================

class TestProxyAttention:
    def test_head_init_shapes(self):
        h = ProxyAttentionHead.init(
            in_dim=8, factor_dim=4, seed=0)
        assert h.w_query.shape == (4, 8)
        assert h.w_key.shape == (4, 8)
        assert h.w_value.shape == (4, 8)

    def test_multi_head_init_shapes(self):
        m = MultiHeadProxyAttention.init(
            in_dim=8, factor_dim=4, n_heads=2, seed=0)
        assert m.n_heads == 2
        assert len(m.heads) == 2
        assert m.w_output.shape == (8, 8)  # factor_dim * n_heads

    def test_empty_bank_returns_zero(self):
        m = MultiHeadProxyAttention.init(
            in_dim=4, factor_dim=2, n_heads=1, seed=0)
        out, attn = m.forward_value(
            query_input=[1.0, 0.0, 0.0, 0.0],
            slot_keys=[], slot_values=[])
        assert all(v == 0.0 for v in out)
        assert all(len(a) == 0 for a in attn)

    def test_attn_weights_sum_to_one(self):
        m = MultiHeadProxyAttention.init(
            in_dim=4, factor_dim=2, n_heads=2, seed=0)
        out, attn = m.forward_value(
            query_input=[1.0, 0.0, 0.0, 0.0],
            slot_keys=[[1.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0]],
            slot_values=[[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0]])
        for a in attn:
            assert abs(sum(a) - 1.0) < 1e-9


# =============================================================================
# Slot memory write head
# =============================================================================

class TestWriteHead:
    def test_init_shape(self):
        h = SlotMemoryWriteHead.init(in_dim=12, seed=0)
        assert h.w_gate.shape == (12,)

    def test_forward_value_in_zero_one(self):
        h = SlotMemoryWriteHead.init(in_dim=4, seed=0)
        v = h.forward_value([1.0, 0.0, 0.0, 0.0])
        assert 0.0 <= v <= 1.0


# =============================================================================
# Reconstruction decoder
# =============================================================================

class TestReconstructionDecoder:
    def test_init_shapes(self):
        d = ReconstructionDecoder.init(
            in_dim=8, hidden_dim=4, recon_dim=12, seed=0)
        assert d.w_hidden.shape == (4, 8)
        assert d.w_out.shape == (12, 4)

    def test_forward_returns_recon_dim(self):
        d = ReconstructionDecoder.init(
            in_dim=4, hidden_dim=4, recon_dim=6, seed=0)
        out = d.forward_value([0.5] * 4)
        assert len(out) == 6


# =============================================================================
# Branch-cycle bias
# =============================================================================

class TestBranchCycleBias:
    def test_init_shape(self):
        b = BranchCycleBias.init(
            n_branches=4, n_cycles=5, seed=0)
        assert b.bias.shape == (4, 5)
        assert len(b.bias.values) == 20

    def test_lookup_modular(self):
        b = BranchCycleBias.init(
            n_branches=2, n_cycles=2, seed=0)
        v00 = b.lookup_value(branch_id=0, cycle_id=0)
        v_overflow = b.lookup_value(branch_id=2, cycle_id=2)
        assert v00 == v_overflow  # modular


# =============================================================================
# Latent control serializer
# =============================================================================

class TestLatentControl:
    def test_default_emit_mask_is_emit_all(self):
        s = LatentControlSerializer.init(n_bits=4, seed=0)
        m = s.emit_mask()
        assert all(m)

    def test_build_latent_control_round_trip(self):
        body, witness = build_latent_control_string(
            ctrl_tag="LATENT_CTRL",
            emit_mask=(True, False, True, False),
            bits_payload=(1, 0, 1, 1),
            shared_state_hash_short="abcd1234efef")
        sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
        assert sha == witness.ctrl_bytes_sha256
        assert "SHARED_STATE_HASH=abcd1234efef" in body
        assert "mask=1010" in body
        assert "bits=1011" in body
        assert witness.n_ctrl_tokens >= 1


# =============================================================================
# Branch-history compressor
# =============================================================================

class TestBranchHistoryCompressor:
    def test_round_trip_exact(self):
        bp = (0, 1, 2, 3, 0, 2)
        cp = (1, 3, 0, 2, 0, 3)
        text, witness = compress_branch_history(
            branch_path=bp, cycle_path=cp,
            n_branches=4, n_cycles=4)
        bp_b, cp_b = decompress_branch_history(
            packed_integer=witness.packed_integer,
            n_pairs=len(bp), n_branches=4, n_cycles=4)
        assert bp_b == bp
        assert cp_b == cp

    def test_compressed_tokens_less_than_textual(self):
        bp = (0,) * 6
        cp = (1,) * 6
        text, witness = compress_branch_history(
            branch_path=bp, cycle_path=cp,
            n_branches=4, n_cycles=4)
        assert (witness.compressed_tokens
                < witness.textual_tokens)


# =============================================================================
# Trivial passthrough falsifier
# =============================================================================

class TestTrivialPassthrough:
    def test_trivial_w48_reduces_to_w47_team(self):
        agents = [
            agent("a", "instr a", max_tokens=64),
            agent("b", "instr b", max_tokens=64),
        ]
        be_w48 = SyntheticLLMClient(
            model_tag="t", default_response="output")
        reg_w48 = build_trivial_shared_state_proxy_registry()
        team_w48 = SharedStateProxyTeam(
            agents, backend=be_w48, registry=reg_w48,
            max_visible_handoffs=2, capture_capsules=True)
        r_w48 = team_w48.run("test task")

        be_w47 = SyntheticLLMClient(
            model_tag="t", default_response="output")
        reg_w47 = build_trivial_autograd_manifold_registry()
        team_w47 = AutogradManifoldTeam(
            agents, backend=be_w47, registry=reg_w47,
            max_visible_handoffs=2, capture_capsules=True)
        r_w47 = team_w47.run("test task")

        assert r_w48.final_output == r_w47.final_output
        assert len(r_w48.turns) == len(r_w47.turns)
        for t in r_w48.proxy_turns:
            assert t.envelope.decision_branch == (
                W48_BRANCH_TRIVIAL_SHARED_STATE_PASSTHROUGH)

    def test_trivial_registry_is_trivial(self):
        reg = build_trivial_shared_state_proxy_registry()
        assert reg.is_trivial


# =============================================================================
# Shared-state CID stability across turns
# =============================================================================

class TestSharedStateCIDStability:
    def test_shared_state_cid_same_every_turn(self):
        sig = hashlib.sha256(b"ss.cid.stab").hexdigest()
        schema = hashlib.sha256(b"ss.schema").hexdigest()
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
        reg = build_shared_state_proxy_registry(
            schema_cid=schema, policy_entries=(policy,),
            margin_abstain_threshold=-99.0,
            spherical_agreement_min=0.5,
            subspace_drift_max=math.pi,
        )
        agents = [agent(f"r{i}", f"i{i}", max_tokens=64)
                  for i in range(4)]
        be = SyntheticLLMClient(
            model_tag="t", default_response="output")
        team = SharedStateProxyTeam(
            agents, backend=be, registry=reg,
            max_visible_handoffs=2, capture_capsules=True)
        r = team.run("cid stability probe")
        cids = [t.envelope.shared_state_capsule_cid
                for t in r.proxy_turns]
        assert len(set(cids)) == 1


# =============================================================================
# Pseudo-KV bank writes across turns
# =============================================================================

class TestPseudoKVAcrossTurns:
    def test_pseudo_kv_writes_grow_or_stable(self):
        sig = hashlib.sha256(b"pkv.writes").hexdigest()
        schema = hashlib.sha256(b"pkv.schema").hexdigest()
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
        reg = build_shared_state_proxy_registry(
            schema_cid=schema, policy_entries=(policy,),
            margin_abstain_threshold=-99.0,
            spherical_agreement_min=0.5,
            subspace_drift_max=math.pi,
        )
        agents = [agent(f"r{i}", f"i{i}", max_tokens=64)
                  for i in range(4)]
        be = SyntheticLLMClient(
            model_tag="t", default_response="output")
        team = SharedStateProxyTeam(
            agents, backend=be, registry=reg,
            max_visible_handoffs=2, capture_capsules=True)
        r = team.run("pseudo-kv writes probe")
        # At least one pseudo-kv write should have happened.
        assert r.n_pseudo_kv_writes >= 1


# =============================================================================
# fit_shared_state_proxy
# =============================================================================

class TestFitSharedStateProxy:
    def test_fit_returns_finite_params(self):
        ts = _make_training_set()
        p = fit_shared_state_proxy(
            ts, n_steps=10, seed=0)
        assert p.fitting_method in (
            "shared_state_proxy_adam_v1",
            "shared_state_proxy_diverged")

    def test_fit_replay_determinism(self):
        ts = _make_training_set()
        a = fit_shared_state_proxy(ts, n_steps=10, seed=0)
        b = fit_shared_state_proxy(ts, n_steps=10, seed=0)
        assert a.cid() == b.cid()
        assert a.training_trace.cid() == b.training_trace.cid()

    def test_fit_different_seed_different_cid(self):
        ts = _make_training_set()
        a = fit_shared_state_proxy(ts, n_steps=10, seed=0)
        b = fit_shared_state_proxy(ts, n_steps=10, seed=1)
        assert a.cid() != b.cid()


# =============================================================================
# forward_shared_state_proxy
# =============================================================================

class TestForwardSharedStateProxy:
    def test_forward_returns_finite_logit(self):
        ts = _make_training_set()
        p = fit_shared_state_proxy(ts, n_steps=5, seed=0)
        bank = PseudoKVBank(
            capacity=W48_DEFAULT_PSEUDO_KV_SLOTS,
            factor_dim=W48_DEFAULT_FACTOR_DIM)
        ex = ts.examples[0]
        fr = forward_shared_state_proxy(
            channel_features=ex.channel_features_map,
            params=p, role=str(ex.role),
            pseudo_kv_bank=bank, turn_index=0,
            branch_id=int(ex.branch_id),
            cycle_id=int(ex.cycle_id),
        )
        assert isinstance(fr.gate_logit, float)
        assert 0.0 <= fr.ratify_probability <= 1.0
        assert fr.confidence_bucket in (0, 1, 2, 3)


# =============================================================================
# Verifier soundness
# =============================================================================

class TestVerifier:
    @pytest.fixture
    def env_and_params(self):
        sig = hashlib.sha256(b"verifier.sig").hexdigest()
        schema_cid = hashlib.sha256(
            b"verifier.schema").hexdigest()
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
        reg = build_shared_state_proxy_registry(
            schema_cid=schema_cid, policy_entries=(policy,),
            margin_abstain_threshold=-99.0,
            spherical_agreement_min=0.5,
            subspace_drift_max=math.pi,
        )
        agents_ = [agent(f"r{i}", f"i{i}", max_tokens=64)
                   for i in range(3)]
        be = SyntheticLLMClient(
            model_tag="t", default_response="output")
        team = SharedStateProxyTeam(
            agents_, backend=be, registry=reg,
            max_visible_handoffs=2, capture_capsules=True)
        r = team.run("verifier test")
        return (
            r.proxy_turns[-1].envelope,
            schema_cid,
            reg.params.cid(),
            reg.params.shared_state.cid(),
        )

    def test_base_envelope_verifies_ok(self, env_and_params):
        env, schema, params_cid, ss_cid = env_and_params
        out = verify_shared_state_proxy_handoff(
            env, registered_schema_cid=schema,
            registered_proxy_params_cid=params_cid,
            registered_shared_state_capsule_cid=ss_cid)
        assert out.ok
        assert out.reason == "ok"
        assert out.n_checks >= 22

    def test_empty_envelope_rejected(self):
        out = verify_shared_state_proxy_handoff(
            None, registered_schema_cid="anything")
        assert not out.ok
        assert out.reason == "empty_w48_envelope"

    def test_bad_schema_version(self, env_and_params):
        env, schema, _, _ = env_and_params
        forged = dataclasses.replace(env, schema_version="badver")
        out = verify_shared_state_proxy_handoff(
            forged, registered_schema_cid=schema)
        assert not out.ok
        assert out.reason == "w48_schema_version_unknown"

    def test_bad_outer_cid(self, env_and_params):
        env, schema, _, _ = env_and_params
        forged = dataclasses.replace(
            env, proxy_outer_cid="z" * 64)
        out = verify_shared_state_proxy_handoff(
            forged, registered_schema_cid=schema)
        assert not out.ok
        assert out.reason == "w48_outer_cid_mismatch"

    def test_bad_proxy_witness(self, env_and_params):
        env, schema, _, _ = env_and_params
        forged = dataclasses.replace(
            env, proxy_witness_cid="0" * 64)
        out = verify_shared_state_proxy_handoff(
            forged, registered_schema_cid=schema)
        assert not out.ok
        assert out.reason == "w48_proxy_witness_cid_mismatch"

    def test_bad_emit_mask(self, env_and_params):
        env, schema, _, _ = env_and_params
        forged = dataclasses.replace(
            env, latent_emit_mask=())
        out = verify_shared_state_proxy_handoff(
            forged, registered_schema_cid=schema)
        assert not out.ok
        assert out.reason == "w48_emit_mask_invalid"

    def test_bad_shared_state_capsule_cid(
            self, env_and_params):
        env, schema, params_cid, _ = env_and_params
        forged = dataclasses.replace(
            env, shared_state_capsule_cid="0" * 63)
        out = verify_shared_state_proxy_handoff(
            forged, registered_schema_cid=schema,
            registered_proxy_params_cid=params_cid)
        assert not out.ok
        assert out.reason == (
            "w48_shared_state_capsule_cid_invalid")

    def test_registered_params_mismatch(
            self, env_and_params):
        env, schema, _, ss_cid = env_and_params
        out = verify_shared_state_proxy_handoff(
            env, registered_schema_cid=schema,
            registered_proxy_params_cid="z" * 64)
        assert not out.ok
        assert out.reason == (
            "w48_proxy_params_cid_invalid")


# =============================================================================
# Replay determinism (run-level)
# =============================================================================

class TestReplayDeterminism:
    def test_two_runs_byte_identical(self):
        sig = hashlib.sha256(b"replay.sig").hexdigest()
        schema = hashlib.sha256(b"replay.schema").hexdigest()
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

        def _run():
            reg = build_shared_state_proxy_registry(
                schema_cid=schema, policy_entries=(policy,),
                margin_abstain_threshold=-99.0,
                spherical_agreement_min=0.5,
                subspace_drift_max=math.pi,
            )
            agents = [agent(f"r{i}", f"i{i}", max_tokens=64)
                      for i in range(3)]
            be = SyntheticLLMClient(
                model_tag="t", default_response="out")
            team = SharedStateProxyTeam(
                agents, backend=be, registry=reg,
                max_visible_handoffs=2,
                capture_capsules=True)
            return team.run("replay probe")
        a = _run()
        b = _run()
        assert a.final_output == b.final_output
        assert a.root_cid == b.root_cid
        for t1, t2 in zip(a.proxy_turns, b.proxy_turns):
            assert (t1.envelope.proxy_outer_cid
                    == t2.envelope.proxy_outer_cid)
            assert (t1.envelope.shared_state_capsule_cid
                    == t2.envelope.shared_state_capsule_cid)
            assert (t1.envelope.pseudo_kv_bank_head_cid
                    == t2.envelope.pseudo_kv_bank_head_cid)


# =============================================================================
# Shared-state-aware backend behavioural lift
# =============================================================================

class TestSharedStateAwareBackend:
    def test_backend_responds_to_shared_state_header(self):
        be = SharedStateAwareSyntheticBackend()
        assert be.generate("SHARED_STATE_HASH: abc") == (
            be.correct_with_shared_state)
        assert be.generate("no header here") == (
            be.answer_without_shared_state)


# =============================================================================
# Public surface sanity
# =============================================================================

class TestPublicSurface:
    def test_team_class_constructible(self):
        agents = [agent("x", "y", max_tokens=64)]
        be = SyntheticLLMClient(
            model_tag="t", default_response="out")
        reg = build_trivial_shared_state_proxy_registry()
        team = SharedStateProxyTeam(
            agents, backend=be, registry=reg,
            max_visible_handoffs=1, capture_capsules=True)
        assert team.schema_cid

    def test_unfitted_params_have_unfitted_method(self):
        p = build_unfitted_shared_state_proxy_params()
        assert p.fitting_method == "unfitted"
