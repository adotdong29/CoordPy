"""Tests for the W47 Autograd Manifold Stack layer.

Each test covers one of the core components: the Variable
autograd engine + finite-difference gradient checks, the
trainable multi-layer stack, the trainable rank-r role adapter,
the trainable dictionary, the trainable memory head, the
trainable packed control serializer, the orchestrator + team,
the envelope verifier (21 failure modes), and the trivial
passthrough falsifier.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math

import pytest

from coordpy.agents import AgentTeam, agent
from coordpy.autograd_manifold import (
    AdamOptimizer,
    AutogradControlSerializer,
    AutogradDictionary,
    AutogradManifoldHandoffEnvelope,
    AutogradManifoldOrchestrator,
    AutogradManifoldParams,
    AutogradManifoldStack,
    AutogradManifoldStack as _Stack,
    AutogradManifoldTeam,
    AutogradMemoryHead,
    AutogradRoleAdapter,
    AutogradStackLayer,
    CtrlAwareAutogradBackend,
    ParamTensor,
    TrainingTraceWitness,
    Variable,
    W47_ALL_BRANCHES,
    W47_ALL_FAILURE_MODES,
    W47_AUTOGRAD_MANIFOLD_SCHEMA_VERSION,
    W47_BRANCH_AUTOGRAD_RATIFIED,
    W47_BRANCH_TRIVIAL_AUTOGRAD_PASSTHROUGH,
    W47_DEFAULT_HIDDEN_DIM,
    W47_DEFAULT_N_LAYERS,
    W47_SUPPORTED_OPS,
    _DeterministicLCG,
    build_autograd_manifold_registry,
    build_trivial_autograd_manifold_registry,
    build_unfitted_autograd_params,
    fit_autograd_controller,
    forward_autograd_controller,
    gradient_check,
    vdot,
    vmean,
    vsoftmax,
    vsum,
    verify_autograd_manifold_handoff,
)
from coordpy.learned_manifold import (
    TrainingExample,
    TrainingSet,
    W45_CHANNEL_ORDER,
    W45_DEFAULT_FEATURE_DIM,
)
from coordpy.manifold_memory import (
    ManifoldMemoryBank,
    ManifoldMemoryTeam,
    W46_CTRL_MODE_FULL,
    W46_CTRL_MODE_OFF,
    build_trivial_manifold_memory_registry,
)
from coordpy.product_manifold import (
    ProductManifoldPolicyEntry,
    encode_spherical_consensus,
    encode_subspace_basis,
)
from coordpy.synthetic_llm import SyntheticLLMClient


# =============================================================================
# Helpers
# =============================================================================

def _make_synthetic_backend(default: str = "OK") -> SyntheticLLMClient:
    return SyntheticLLMClient(
        model_tag="synthetic.w47", default_response=default)


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
# Schema / branch invariants
# =============================================================================

class TestSchemaAndBranches:
    def test_schema_version_is_v1(self):
        assert W47_AUTOGRAD_MANIFOLD_SCHEMA_VERSION == (
            "coordpy.autograd_manifold.v1")

    def test_branches_are_unique(self):
        assert len(set(W47_ALL_BRANCHES)) == len(W47_ALL_BRANCHES)

    def test_failure_modes_are_unique_and_ge_18(self):
        assert (len(set(W47_ALL_FAILURE_MODES))
                == len(W47_ALL_FAILURE_MODES))
        assert len(W47_ALL_FAILURE_MODES) >= 18

    def test_failure_modes_disjoint_from_w22_w46(self):
        w47_prefix = sum(
            1 for m in W47_ALL_FAILURE_MODES
            if m.startswith("w47_"))
        empty = sum(
            1 for m in W47_ALL_FAILURE_MODES
            if m == "empty_w47_envelope")
        assert w47_prefix + empty == len(W47_ALL_FAILURE_MODES)
        assert empty == 1


# =============================================================================
# Variable autograd engine
# =============================================================================

class TestVariableEngine:
    def test_leaf_value_and_grad(self):
        v = Variable(2.5)
        assert v.value == 2.5
        assert v.grad == 0.0
        assert v.op == "leaf"

    def test_add_and_backward(self):
        a = Variable(1.0)
        b = Variable(2.0)
        c = a + b
        c.backward()
        assert c.value == 3.0
        assert a.grad == 1.0
        assert b.grad == 1.0

    def test_mul_chain_rule(self):
        a = Variable(3.0)
        b = Variable(4.0)
        c = a * b
        c.backward()
        assert c.value == 12.0
        assert a.grad == 4.0
        assert b.grad == 3.0

    def test_tanh_derivative(self):
        a = Variable(0.5)
        c = a.tanh()
        c.backward()
        # d/dx tanh(x) = 1 - tanh(x)^2
        expected = 1.0 - math.tanh(0.5) ** 2
        assert abs(a.grad - expected) < 1e-10

    def test_sigmoid_derivative(self):
        a = Variable(0.4)
        c = a.sigmoid()
        c.backward()
        s = 1.0 / (1.0 + math.exp(-0.4))
        expected = s * (1.0 - s)
        assert abs(a.grad - expected) < 1e-10

    def test_pow_derivative(self):
        a = Variable(2.0)
        c = a ** 3.0
        c.backward()
        # d/dx x^3 = 3x^2
        assert abs(a.grad - 12.0) < 1e-10

    def test_div_derivative(self):
        a = Variable(6.0)
        b = Variable(3.0)
        c = a / b
        c.backward()
        # d/da a/b = 1/b; d/db a/b = -a/b^2
        assert abs(a.grad - 1.0 / 3.0) < 1e-10
        assert abs(b.grad - (-6.0 / 9.0)) < 1e-10

    def test_log_and_exp_round_trip(self):
        a = Variable(1.5)
        c = a.exp().log()
        c.backward()
        # d/dx log(exp(x)) = 1
        assert abs(a.grad - 1.0) < 1e-10


# =============================================================================
# Gradient checks against finite differences
# =============================================================================

class TestGradientChecks:
    def test_gradient_check_linear(self):
        def f(vs):
            x, y = vs
            return x * Variable(2.0) + y * Variable(-3.0) + Variable(1.0)

        ok, err, _, _ = gradient_check(f, [0.5, -0.7])
        assert ok
        assert err < 1e-5

    def test_gradient_check_tanh(self):
        def f(vs):
            return (vs[0] * Variable(1.3)).tanh()

        ok, err, _, _ = gradient_check(f, [0.3])
        assert ok
        assert err < 1e-5

    def test_gradient_check_sigmoid_bce(self):
        def f(vs):
            s = vs[0].sigmoid()
            return -((s + 1e-9).log())

        ok, err, _, _ = gradient_check(f, [0.4])
        assert ok
        assert err < 1e-5

    def test_gradient_check_softmax_xent(self):
        def f(vs):
            soft = vsoftmax(vs)
            return -((soft[1] + 1e-9).log())

        ok, err, _, _ = gradient_check(f, [0.5, 0.7, -0.2])
        assert ok
        assert err < 1e-5

    def test_gradient_check_dot_product(self):
        def f(vs):
            return vdot(
                [vs[0], vs[1]], [vs[2], vs[3]]).tanh()

        ok, err, _, _ = gradient_check(
            f, [0.2, -0.4, 0.6, 0.5])
        assert ok
        assert err < 1e-5

    def test_supported_ops_count(self):
        # Sanity: the supported-ops list is the truth source.
        assert len(W47_SUPPORTED_OPS) >= 12


# =============================================================================
# DeterministicLCG seed reproducibility
# =============================================================================

class TestDeterministicLCG:
    def test_same_seed_same_stream(self):
        a = _DeterministicLCG(seed=42)
        b = _DeterministicLCG(seed=42)
        for _ in range(8):
            assert a.next_uniform() == b.next_uniform()

    def test_diff_seed_diff_stream(self):
        a = _DeterministicLCG(seed=1)
        b = _DeterministicLCG(seed=2)
        first_a = a.next_uniform()
        first_b = b.next_uniform()
        assert first_a != first_b

    def test_uniform_in_range(self):
        rng = _DeterministicLCG(seed=99)
        for _ in range(20):
            v = rng.next_uniform()
            assert 0.0 <= v <= 1.0


# =============================================================================
# ParamTensor + Adam
# =============================================================================

class TestParamTensorAndAdam:
    def test_param_tensor_init_seeded(self):
        p = ParamTensor(shape=(3, 4), values=[])
        p.init_seed(seed=7, scale=0.5)
        assert p.size == 12
        assert all(-0.5 <= v <= 0.5 for v in p.values)
        # Determinism: same seed -> same values.
        p2 = ParamTensor(shape=(3, 4), values=[])
        p2.init_seed(seed=7, scale=0.5)
        assert p.values == p2.values

    def test_adam_step_decreases_quadratic_loss(self):
        # Scalar quadratic: minimise (x - 5)^2 — gradient at x is
        # 2*(x - 5).
        p = ParamTensor(shape=(1,), values=[0.0])
        opt = AdamOptimizer(learning_rate=0.5)
        for _ in range(80):
            xs = p.make_vars()
            loss = (xs[0] - Variable(5.0)) * (xs[0] - Variable(5.0))
            loss.backward()
            opt.step([p])
        assert abs(p.values[0] - 5.0) < 0.5

    def test_adam_grad_clip(self):
        # With grad_clip = 0.1, step magnitude is capped.
        p = ParamTensor(shape=(1,), values=[0.0])
        opt = AdamOptimizer(learning_rate=1.0, grad_clip=0.1)
        xs = p.make_vars()
        loss = (xs[0] - Variable(100.0)) ** 2.0
        loss.backward()
        opt.step([p])
        # With LR=1 and grad ~ 200, but clipped to ~0.1, the
        # step should be order O(LR) = 1.
        assert abs(p.values[0]) < 5.0


# =============================================================================
# AutogradStackLayer / AutogradManifoldStack
# =============================================================================

class TestAutogradStackLayer:
    def test_layer_init_shapes(self):
        layer = AutogradStackLayer.init(
            in_dim=8, out_dim=4, seed=0)
        assert layer.in_dim == 8
        assert layer.out_dim == 4
        assert layer.weights.shape == (4, 8)
        assert layer.biases.shape == (4,)
        # Bias init = 0.0.
        assert all(v == 0.0 for v in layer.biases.values)

    def test_layer_forward_shapes(self):
        layer = AutogradStackLayer.init(
            in_dim=4, out_dim=2, seed=0, activation="tanh")
        out = layer.forward([Variable(1.0)] * 4)
        assert len(out) == 2
        # Tanh outputs are in (-1, 1).
        for v in out:
            assert -1.0 < v.value < 1.0


class TestAutogradManifoldStack:
    def test_stack_init_shapes(self):
        stack = AutogradManifoldStack.init(
            feature_dim=4, n_layers=3, hidden_dim=8, seed=0)
        # 3 layers: hidden, hidden, scalar.
        assert stack.n_layers == 3
        assert stack.layers[-1].out_dim == 1
        assert stack.layers[0].in_dim == 24  # 6 channels * 4

    def test_stack_forward_value_returns_scalar(self):
        stack = AutogradManifoldStack.init(
            feature_dim=4, n_layers=2, hidden_dim=4, seed=0)
        v = stack.forward_value([0.5] * 24)
        assert isinstance(v, float)

    def test_stack_cid_64hex(self):
        stack = AutogradManifoldStack.init(
            feature_dim=4, n_layers=2, hidden_dim=4, seed=0)
        cid = stack.cid()
        assert len(cid) == 64
        int(cid, 16)


# =============================================================================
# AutogradRoleAdapter
# =============================================================================

class TestAutogradRoleAdapter:
    def test_adapter_init_per_role(self):
        adapter = AutogradRoleAdapter.init(
            roles=("role0", "role1"), rank=2, in_dim=8, seed=0)
        assert "role0" in adapter.role_factors
        assert "role1" in adapter.role_factors
        a, b = adapter.role_factors["role0"]
        assert a.shape == (8, 2)
        assert b.shape == (2,)

    def test_adapter_unknown_role_returns_zero(self):
        adapter = AutogradRoleAdapter.init(
            roles=("role0",), rank=2, in_dim=4, seed=0)
        delta = adapter.forward_delta(
            role="role_missing",
            hidden=[Variable(1.0)] * 4)
        assert delta.value == 0.0


# =============================================================================
# AutogradDictionary
# =============================================================================

class TestAutogradDictionary:
    def test_encode_decode_round_trip(self):
        d = AutogradDictionary.init(
            feature_dim=4, k=3, seed=0, init_scale=0.5)
        v = [0.5] * 24
        idx, residual = d.encode_inference(v)
        decoded = d.decode(idx, residual)
        for a, b in zip(v, decoded):
            assert abs(a - b) <= 1e-9

    def test_encode_returns_closest_prototype(self):
        # Build a dict with two unit prototypes.
        d = AutogradDictionary.init(
            feature_dim=4, k=2, seed=0, init_scale=0.0)
        # Manually set prototype values.
        d.prototypes.values = [1.0] * 24 + [-1.0] * 24
        idx, _ = d.encode_inference([1.0] * 24)
        assert idx == 0

    def test_cid_64hex(self):
        d = AutogradDictionary.init(
            feature_dim=4, k=4, seed=0)
        cid = d.cid()
        assert len(cid) == 64
        int(cid, 16)


# =============================================================================
# AutogradMemoryHead
# =============================================================================

class TestAutogradMemoryHead:
    def test_head_init_shapes(self):
        head = AutogradMemoryHead.init(
            in_dim=12, head_dim=4, seed=0)
        assert head.w_query.shape == (4, 12)
        assert head.w_key.shape == (4, 12)
        assert head.w_value.shape == (1,)

    def test_empty_keys_returns_zero(self):
        head = AutogradMemoryHead.init(
            in_dim=4, head_dim=2, seed=0)
        pooled, attn = head.forward_attention_value(
            query_input=[1.0, 0.0, 0.0, 0.0],
            keys_inputs=[],
            entry_logits=[],
        )
        assert pooled == 0.0
        assert attn == []

    def test_attention_weights_sum_to_one(self):
        head = AutogradMemoryHead.init(
            in_dim=4, head_dim=2, seed=0)
        pooled, attn = head.forward_attention_value(
            query_input=[1.0, 0.0, 0.0, 0.0],
            keys_inputs=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            entry_logits=[1.0, -1.0, 0.5],
        )
        assert abs(sum(attn) - 1.0) < 1e-9


# =============================================================================
# AutogradControlSerializer
# =============================================================================

class TestAutogradControlSerializer:
    def test_default_emit_mask_is_emit_all(self):
        ser = AutogradControlSerializer.init(seed=0)
        mask = ser.emit_mask()
        assert all(mask)

    def test_train_to_target_mask(self):
        target = (True, False, True, False)
        ser = AutogradControlSerializer.init(seed=0)
        opt = AdamOptimizer(learning_rate=0.3)
        for _ in range(120):
            loss = ser.forward_loss_vars(target_mask=target)
            loss.backward()
            opt.step(ser.params())
        learned = ser.emit_mask()
        assert learned == target


# =============================================================================
# fit_autograd_controller end-to-end
# =============================================================================

class TestFitAutogradController:
    def test_fit_returns_finite_params(self):
        ts = _make_training_set()
        p = fit_autograd_controller(
            ts, n_layers=2, hidden_dim=4, n_steps=20, seed=0)
        assert p.fitting_method in (
            "autograd_adam_v1", "autograd_diverged")
        assert isinstance(p.training_trace.final_train_loss, float)

    def test_fit_replay_determinism(self):
        ts = _make_training_set()
        p1 = fit_autograd_controller(
            ts, n_layers=2, hidden_dim=4, n_steps=20, seed=0)
        p2 = fit_autograd_controller(
            ts, n_layers=2, hidden_dim=4, n_steps=20, seed=0)
        assert p1.cid() == p2.cid()
        assert p1.training_trace.cid() == p2.training_trace.cid()

    def test_fit_diff_seed_diff_params(self):
        ts = _make_training_set()
        p1 = fit_autograd_controller(
            ts, n_layers=2, hidden_dim=4, n_steps=10, seed=0)
        p2 = fit_autograd_controller(
            ts, n_layers=2, hidden_dim=4, n_steps=10, seed=1)
        assert p1.cid() != p2.cid()


# =============================================================================
# forward_autograd_controller
# =============================================================================

class TestForwardAutogradController:
    def test_forward_returns_scalar_logit(self):
        ts = _make_training_set()
        p = fit_autograd_controller(
            ts, n_layers=2, hidden_dim=4, n_steps=10, seed=0)
        ex = ts.examples[0]
        fr = forward_autograd_controller(
            channel_features=ex.channel_features_map,
            params=p, role=str(ex.role),
            memory_bank=ManifoldMemoryBank(capacity=1),
            turn_index=0, time_attention_enabled=False,
        )
        assert isinstance(fr.gate_logit, float)
        assert 0.0 <= fr.ratify_probability <= 1.0
        assert fr.confidence_bucket in (0, 1, 2, 3)

    def test_emit_mask_shape(self):
        ts = _make_training_set()
        p = fit_autograd_controller(
            ts, n_layers=2, hidden_dim=4, n_steps=5, seed=0)
        ex = ts.examples[0]
        fr = forward_autograd_controller(
            channel_features=ex.channel_features_map,
            params=p, role=str(ex.role),
            memory_bank=ManifoldMemoryBank(capacity=1),
            turn_index=0, time_attention_enabled=False,
        )
        assert len(fr.emit_mask) == 4
        assert all(isinstance(b, bool) for b in fr.emit_mask)


# =============================================================================
# Trivial passthrough falsifier
# =============================================================================

class TestTrivialPassthrough:
    def test_trivial_w47_reduces_to_agentteam(self):
        agents = [
            agent("a", "instr a", max_tokens=64),
            agent("b", "instr b", max_tokens=64),
        ]
        be = SyntheticLLMClient(
            model_tag="t", default_response="output")
        base = AgentTeam(
            agents, backend=be, max_visible_handoffs=2,
            capture_capsules=True)
        r_base = base.run("test task")

        be2 = SyntheticLLMClient(
            model_tag="t", default_response="output")
        reg = build_trivial_autograd_manifold_registry()
        team = AutogradManifoldTeam(
            agents, backend=be2, registry=reg,
            max_visible_handoffs=2, capture_capsules=True)
        r = team.run("test task")
        assert r.final_output == r_base.final_output
        assert len(r.turns) == len(r_base.turns)
        for t in r.autograd_turns:
            assert t.envelope.decision_branch == (
                W47_BRANCH_TRIVIAL_AUTOGRAD_PASSTHROUGH)

    def test_trivial_registry_is_trivial(self):
        reg = build_trivial_autograd_manifold_registry()
        assert reg.is_trivial


# =============================================================================
# Verifier soundness
# =============================================================================

class TestVerifier:
    @pytest.fixture
    def env_and_params(self):
        sig = hashlib.sha256(b"verifier.sig").hexdigest()
        ts = _make_training_set(sig=sig)
        p = fit_autograd_controller(
            ts, n_layers=2, hidden_dim=4, n_steps=10, seed=0)
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
        reg = build_autograd_manifold_registry(
            schema_cid=schema_cid,
            policy_entries=(policy,),
            params=p,
            margin_abstain_threshold=-99.0,
            spherical_agreement_min=0.5,
            subspace_drift_max=math.pi,
        )
        agents_ = [
            agent(f"r{i}", f"i{i}", max_tokens=64)
            for i in range(3)]
        be = SyntheticLLMClient(
            model_tag="t", default_response="output")
        team = AutogradManifoldTeam(
            agents_, backend=be, registry=reg,
            max_visible_handoffs=2, capture_capsules=True)
        r = team.run("verifier test")
        return r.autograd_turns[-1].envelope, schema_cid, p.cid()

    def test_base_envelope_verifies_ok(self, env_and_params):
        env, schema, params_cid = env_and_params
        out = verify_autograd_manifold_handoff(
            env, registered_schema_cid=schema,
            registered_autograd_params_cid=params_cid)
        assert out.ok
        assert out.reason == "ok"
        assert out.n_checks >= 18

    def test_empty_envelope_rejected(self):
        out = verify_autograd_manifold_handoff(
            None, registered_schema_cid="anything")
        assert not out.ok
        assert out.reason == "empty_w47_envelope"

    def test_bad_schema_version(self, env_and_params):
        env, schema, params_cid = env_and_params
        forged = dataclasses.replace(env, schema_version="badver")
        out = verify_autograd_manifold_handoff(
            forged, registered_schema_cid=schema)
        assert not out.ok
        assert out.reason == "w47_schema_version_unknown"

    def test_bad_outer_cid(self, env_and_params):
        env, schema, _ = env_and_params
        forged = dataclasses.replace(
            env, autograd_outer_cid="z" * 64)
        out = verify_autograd_manifold_handoff(
            forged, registered_schema_cid=schema)
        assert not out.ok
        assert out.reason == "w47_outer_cid_mismatch"

    def test_bad_emit_mask(self, env_and_params):
        env, schema, _ = env_and_params
        forged = dataclasses.replace(
            env, emit_mask=(True, True))
        out = verify_autograd_manifold_handoff(
            forged, registered_schema_cid=schema)
        assert not out.ok
        assert out.reason == "w47_emit_mask_invalid"

    def test_bad_witness_cid(self, env_and_params):
        env, schema, _ = env_and_params
        forged = dataclasses.replace(
            env, autograd_witness_cid="x" * 64)
        out = verify_autograd_manifold_handoff(
            forged, registered_schema_cid=schema)
        assert not out.ok
        assert out.reason == "w47_autograd_witness_cid_mismatch"


# =============================================================================
# Public surface invariants
# =============================================================================

class TestPublicSurface:
    def test_team_class_is_subclass_safe(self):
        # Sanity: AutogradManifoldTeam exists and is constructible.
        agents = [agent("x", "y", max_tokens=64)]
        be = SyntheticLLMClient(
            model_tag="t", default_response="out")
        reg = build_trivial_autograd_manifold_registry()
        team = AutogradManifoldTeam(
            agents, backend=be, registry=reg,
            max_visible_handoffs=1, capture_capsules=True)
        assert team.schema_cid

    def test_unfitted_params_have_unfitted_method(self):
        p = build_unfitted_autograd_params()
        assert p.fitting_method == "unfitted"

    def test_ctrl_aware_backend_responds_to_full_ctrl(self):
        be = CtrlAwareAutogradBackend()
        assert be.generate("MANIFOLD_CTRL: layer_logits=[]") == (
            be.correct_with_full_ctrl)
        assert be.generate("MANIFOLD_CTRL:") == (
            be.answer_with_partial_ctrl)
        assert be.generate("nothing here") == (
            be.answer_without_ctrl)
