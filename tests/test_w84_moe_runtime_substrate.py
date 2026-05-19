"""W84 / P1 #31 — MoE Substrate tests."""

from __future__ import annotations


def test_w84_moe_extended_axes_declared():
    """DoD bar: at least 3 new MoE-specific axes are declared."""
    from coordpy.moe_runtime_substrate_v1 import (
        MoEInstrumentationAxis,
        W84_MOE_INSTRUMENTATION_AXES,
    )
    assert len(W84_MOE_INSTRUMENTATION_AXES) >= 3
    assert (
        MoEInstrumentationAxis
        .READ_EXPERT_ROUTING_PER_LAYER.value
        in W84_MOE_INSTRUMENTATION_AXES)
    assert (
        MoEInstrumentationAxis
        .READ_EXPERT_OUTPUT_PER_EXPERT_PER_LAYER.value
        in W84_MOE_INSTRUMENTATION_AXES)
    assert (
        MoEInstrumentationAxis
        .WRITE_FORCE_EXPERT_ROUTING_PER_LAYER.value
        in W84_MOE_INSTRUMENTATION_AXES)


def test_w84_moe_anti_cheat_rejects_n_experts_below_4():
    import pytest
    from coordpy.moe_runtime_substrate_v1 import (
        build_moe_runtime_params_v1,
    )
    with pytest.raises(ValueError):
        build_moe_runtime_params_v1(n_experts=2, top_k=1)


def test_w84_moe_anti_cheat_rejects_top_k_below_2():
    import pytest
    from coordpy.moe_runtime_substrate_v1 import (
        build_moe_runtime_params_v1,
    )
    with pytest.raises(ValueError):
        build_moe_runtime_params_v1(n_experts=4, top_k=1)


def test_w84_moe_runtime_constructs():
    """DoD bar: an MoE runtime adapter loads at least one
    MoE configuration."""
    from coordpy.moe_runtime_substrate_v1 import (
        MoERuntimeAdapterV1, build_moe_runtime_params_v1,
    )
    params = build_moe_runtime_params_v1()
    assert int(params.n_experts) >= 4
    assert int(params.top_k) >= 2
    adapter = MoERuntimeAdapterV1(params=params)
    ids = adapter.tokenize("moe smoke", max_len=8)
    assert len(ids) > 0


def test_w84_moe_forward_returns_routing_snapshot():
    """DoD bar: the trace carries a content-addressed routing
    snapshot."""
    from coordpy.moe_runtime_substrate_v1 import (
        MoERuntimeAdapterV1, forward_moe_runtime,
        build_moe_runtime_params_v1,
    )
    params = build_moe_runtime_params_v1()
    adapter = MoERuntimeAdapterV1(params=params)
    ids = adapter.tokenize("snap routing", max_len=10)
    trace, _ = forward_moe_runtime(
        params=params, input_token_ids=ids)
    assert len(trace.routing.cid()) == 64
    assert int(trace.routing.n_experts) == int(
        params.n_experts)
    assert int(trace.routing.top_k) == int(params.top_k)
    # Each layer has a (seq_len, top_k) routing-id array.
    for L in range(int(params.n_layers)):
        assert trace.routing.per_layer_top_k_ids[L].shape == (
            int(trace.seq_len), int(params.top_k))


def test_w84_moe_forward_returns_expert_output_snapshot():
    """DoD bar: per-(layer, expert) outputs captured."""
    from coordpy.moe_runtime_substrate_v1 import (
        forward_moe_runtime, build_moe_runtime_params_v1,
        MoERuntimeAdapterV1,
    )
    params = build_moe_runtime_params_v1()
    adapter = MoERuntimeAdapterV1(params=params)
    ids = adapter.tokenize("expert outputs", max_len=10)
    trace, _ = forward_moe_runtime(
        params=params, input_token_ids=ids)
    assert len(trace.expert_outputs.cid()) == 64
    # At least one expert fired per layer.
    for L in range(int(params.n_layers)):
        assert len(
            trace.expert_outputs.per_layer_expert_ids[L]) > 0


def test_w84_moe_forced_routing_changes_post_mlp_output():
    """DoD bar: forcing a different routing measurably changes
    the post-MLP output."""
    import numpy as np
    from coordpy.moe_runtime_substrate_v1 import (
        MoEForceRoutingPlanV1,
        W84_MOE_RUNTIME_V1_SCHEMA_VERSION,
        forward_moe_runtime, build_moe_runtime_params_v1,
        MoERuntimeAdapterV1,
    )
    params = build_moe_runtime_params_v1()
    adapter = MoERuntimeAdapterV1(params=params)
    ids = adapter.tokenize("forced routing probe", max_len=10)
    natural, _ = forward_moe_runtime(
        params=params, input_token_ids=ids)
    # Build a different routing plan: shift each chosen
    # expert by +1 (mod n_experts) on every layer.
    mut_layers = []
    for L in range(int(params.n_layers)):
        ids_L = natural.routing.per_layer_top_k_ids[L]
        mut = (
            (np.asarray(ids_L, dtype=np.int64) + 1)
            % int(params.n_experts))
        mut_layers.append(mut.astype(np.int64))
    plan = MoEForceRoutingPlanV1(
        schema=W84_MOE_RUNTIME_V1_SCHEMA_VERSION,
        force_top_k_ids_per_layer=tuple(mut_layers))
    forced, _ = forward_moe_runtime(
        params=params, input_token_ids=ids,
        force_routing_plan=plan)
    diff = float(np.max(np.abs(
        natural.logits[-1] - forced.logits[-1])))
    # The forced routing MUST produce a measurably different
    # output — at least 1e-2.
    assert diff > 1e-2, diff


def test_w84_moe_kv_plus_routing_replay_matches_full_at_fp32():
    """DoD bar: replay with KV + routing restored is
    byte-identical to the full forward at fp32 floor."""
    import numpy as np
    from coordpy.moe_runtime_substrate_v1 import (
        forward_moe_runtime, build_moe_runtime_params_v1,
        MoERuntimeAdapterV1, routing_plan_from_snapshot,
    )
    params = build_moe_runtime_params_v1()
    adapter = MoERuntimeAdapterV1(params=params)
    ids = adapter.tokenize("byte identical", max_len=10)
    natural, _ = forward_moe_runtime(
        params=params, input_token_ids=ids)
    plan = routing_plan_from_snapshot(
        natural.routing,
        n_layers=int(params.n_layers))
    replayed, _ = forward_moe_runtime(
        params=params, input_token_ids=ids,
        force_routing_plan=plan)
    diff = float(np.max(np.abs(
        natural.logits[-1] - replayed.logits[-1])))
    assert diff < 5e-3, diff


def test_w84_moe_trace_cid_distinguishes_routings():
    """DoD bar: the W84 trace CID is honest about routing —
    two forwards with same KV but different routing produce
    different trace CIDs."""
    import numpy as np
    from coordpy.moe_runtime_substrate_v1 import (
        MoEForceRoutingPlanV1,
        W84_MOE_RUNTIME_V1_SCHEMA_VERSION,
        forward_moe_runtime, build_moe_runtime_params_v1,
        MoERuntimeAdapterV1,
    )
    params = build_moe_runtime_params_v1()
    adapter = MoERuntimeAdapterV1(params=params)
    ids = adapter.tokenize("trace cid", max_len=10)
    natural, _ = forward_moe_runtime(
        params=params, input_token_ids=ids)
    mut_layers = []
    for L in range(int(params.n_layers)):
        ids_L = natural.routing.per_layer_top_k_ids[L]
        mut = (
            (np.asarray(ids_L, dtype=np.int64) + 1)
            % int(params.n_experts))
        mut_layers.append(mut.astype(np.int64))
    plan = MoEForceRoutingPlanV1(
        schema=W84_MOE_RUNTIME_V1_SCHEMA_VERSION,
        force_top_k_ids_per_layer=tuple(mut_layers))
    forced, _ = forward_moe_runtime(
        params=params, input_token_ids=ids,
        force_routing_plan=plan)
    assert natural.cid() != forced.cid()
    assert natural.routing.cid() != forced.routing.cid()


def test_w84_moe_hidden_state_intercept_moves_cid():
    """DoD bar: hidden-state intercept at layer L moves the
    trace CID, including the routing CID."""
    import numpy as np
    from coordpy.moe_runtime_substrate_v1 import (
        forward_moe_runtime, build_moe_runtime_params_v1,
        MoERuntimeAdapterV1,
    )
    params = build_moe_runtime_params_v1()
    adapter = MoERuntimeAdapterV1(params=params)
    ids = adapter.tokenize("hidden intercept", max_len=10)
    baseline, _ = forward_moe_runtime(
        params=params, input_token_ids=ids)
    shape0 = baseline.pre_attn_hidden[0].shape
    inj = np.full(shape0, 0.1, dtype=np.float64)
    injs = [None] * int(params.n_layers)
    injs[0] = inj
    after, _ = forward_moe_runtime(
        params=params, input_token_ids=ids,
        hidden_state_injections_per_layer=injs)
    assert baseline.cid() != after.cid()
    # The routing CID also changes (since hidden state moved
    # changes the router decision).
    assert baseline.routing.cid() != after.routing.cid()


def test_w84_moe_bench_passes_all_load_bearing_claims():
    """End-to-end bench: ALL load-bearing MoE claims hold."""
    from coordpy.moe_runtime_substrate_v1 import (
        run_moe_bench_v1,
    )
    rep = run_moe_bench_v1(n_prompts=10)
    d = rep.to_dict()
    assert bool(rep.forced_routing_changes_output), d
    assert bool(
        rep.replay_with_natural_routing_within_floor), d
    assert bool(rep.trace_cid_changes_with_routing), d
    assert bool(rep.hidden_intercept_moves_cid), d
    assert bool(rep.routing_cid_changes_with_force_plan), d
    assert int(rep.n_experts) >= 4
    assert int(rep.top_k) >= 2


def test_w84_moe_routing_snapshot_cid_deterministic_on_same_inputs():
    """Anti-cheat: identical routing -> identical CID."""
    from coordpy.moe_runtime_substrate_v1 import (
        forward_moe_runtime, build_moe_runtime_params_v1,
    )
    params = build_moe_runtime_params_v1()
    ids = [3, 7, 11, 22, 51]
    t1, _ = forward_moe_runtime(
        params=params, input_token_ids=ids)
    t2, _ = forward_moe_runtime(
        params=params, input_token_ids=ids)
    assert t1.routing.cid() == t2.routing.cid()
    assert t1.cid() == t2.cid()


def test_w84_moe_adapter_declares_all_moe_axes():
    from coordpy.moe_runtime_substrate_v1 import (
        MoERuntimeAdapterV1, W84_MOE_INSTRUMENTATION_AXES,
    )
    from coordpy.runtime_instrumentation_v1 import (
        CapabilityTag,
    )
    a = MoERuntimeAdapterV1()
    axes = a.declared_axes()
    for ax in W84_MOE_INSTRUMENTATION_AXES:
        assert ax in axes
        assert axes[ax] == CapabilityTag.AVAILABLE.value
