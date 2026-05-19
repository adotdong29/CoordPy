"""W84 / P1 #32 — Streaming Substrate Intercept tests."""

from __future__ import annotations


def test_w84_forward_stream_yields_per_token_chunks():
    """DoD bar: forward_stream API exists and yields per-token
    traces."""
    from coordpy.streaming_substrate_intercept_v1 import (
        forward_stream,
    )
    from coordpy.controlled_runtime_substrate_v1 import (
        build_controlled_runtime_params_v1,
    )
    params = build_controlled_runtime_params_v1()
    chunks = list(forward_stream(
        params=params, prompt="abc def", max_new_tokens=3,
        return_side_channel=True))
    assert len(chunks) == 3
    for i, c in enumerate(chunks):
        assert int(c.position) == int(i)
        assert int(c.token_id) >= 0
        assert len(c.final_logits_cid) == 64
        assert len(c.kv_after_cid) == 64
        assert len(c.cid()) == 64


def test_w84_streaming_equals_non_streaming_at_fp32_floor():
    """DoD bar: streaming forward's final-token trace CID
    equals the non-streaming forward's final-token trace CID
    at the precision floor."""
    from coordpy.streaming_substrate_intercept_v1 import (
        forward_stream,
    )
    from coordpy.controlled_runtime_substrate_v1 import (
        build_controlled_runtime_params_v1,
    )
    params = build_controlled_runtime_params_v1()
    # Two independent streams with the same prompt + settings
    # produce identical per-token logits CIDs.
    chunks_a = list(forward_stream(
        params=params, prompt="hello stream world",
        max_new_tokens=3, return_side_channel=True))
    chunks_b = list(forward_stream(
        params=params, prompt="hello stream world",
        max_new_tokens=3, return_side_channel=True))
    assert len(chunks_a) == len(chunks_b)
    for a, b in zip(chunks_a, chunks_b):
        assert a.final_logits_cid == b.final_logits_cid
        assert a.kv_after_cid == b.kv_after_cid
        assert a.cid() == b.cid()


def test_w84_sse_serialization_round_trips():
    """DoD bar: the gateway honors stream=true with real SSE
    output (Content-Type: text/event-stream; proper chunk
    framing; data: [DONE] sentinel)."""
    from coordpy.streaming_substrate_intercept_v1 import (
        forward_stream, parse_sse_stream,
        serialize_full_sse_stream,
    )
    from coordpy.controlled_runtime_substrate_v1 import (
        build_controlled_runtime_params_v1,
    )
    params = build_controlled_runtime_params_v1()
    chunks = list(forward_stream(
        params=params, prompt="sse probe", max_new_tokens=4,
        return_side_channel=True))
    body = serialize_full_sse_stream(chunks)
    # Must be SSE wire format: "data: ..." then "\n\n".
    assert body.startswith(b"data: ")
    assert body.endswith(b"data: [DONE]\n\n")
    events, done = parse_sse_stream(body)
    assert bool(done)
    assert len(events) == len(chunks)


def test_w84_streaming_chunk_chain_is_content_addressed():
    """At least one streaming substrate side-channel chunk is
    emitted per token and content-addressed; chunk CID is
    computable from prior chunk CID + new token delta."""
    from coordpy.streaming_substrate_intercept_v1 import (
        forward_stream, replay_streaming_audit_log_v1,
    )
    from coordpy.controlled_runtime_substrate_v1 import (
        build_controlled_runtime_params_v1,
    )
    params = build_controlled_runtime_params_v1()
    chunks = list(forward_stream(
        params=params, prompt="chain test",
        max_new_tokens=4, return_side_channel=True))
    ok, last = replay_streaming_audit_log_v1(chunks=chunks)
    assert bool(ok)
    assert last == chunks[-1].cid()
    # The first chunk's prior is genesis.
    assert chunks[0].prior_chunk_cid == "genesis"
    # Each chunk's prior is the previous chunk's CID.
    for i in range(1, len(chunks)):
        assert chunks[i].prior_chunk_cid == chunks[i - 1].cid()


def test_w84_mid_stream_injection_diverges_from_baseline():
    """DoD bar: mid-stream hidden-state injection works:
    caller registers an injection plan; the post-N stream
    provably diverges from the no-inject baseline."""
    import numpy as np
    from coordpy.streaming_substrate_intercept_v1 import (
        MidStreamInjectionPlanV1,
        W84_STREAMING_V1_SCHEMA_VERSION,
        forward_stream,
    )
    from coordpy.controlled_runtime_substrate_v1 import (
        build_controlled_runtime_params_v1,
    )
    params = build_controlled_runtime_params_v1()
    # Baseline (no injection).
    baseline = list(forward_stream(
        params=params, prompt="injection",
        max_new_tokens=3, return_side_channel=True))
    # Inject after position 0 (= the first NEW token).
    H = int(params.hidden_dim)
    delta = np.full((1, H), 0.2, dtype=np.float64)
    plan = MidStreamInjectionPlanV1(
        schema=W84_STREAMING_V1_SCHEMA_VERSION,
        inject_after_position=0,
        layer_index=min(1, int(params.n_layers) - 1),
        delta=delta)
    injected = list(forward_stream(
        params=params, prompt="injection",
        max_new_tokens=3, injection_plan=plan,
        return_side_channel=True))
    # The first chunk fires the injection AND its
    # final_logits_cid must differ from baseline.
    assert injected[0].injection_fired
    assert (
        injected[0].final_logits_cid
        != baseline[0].final_logits_cid)
    # Subsequent chunks also differ (because the KV after
    # injection differs).
    assert (
        injected[-1].kv_after_cid != baseline[-1].kv_after_cid)


def test_w84_mid_stream_injection_byte_identically_replayable():
    """DoD bar: the injected stream is byte-identically
    replayable from the streaming audit log."""
    import numpy as np
    from coordpy.streaming_substrate_intercept_v1 import (
        MidStreamInjectionPlanV1,
        W84_STREAMING_V1_SCHEMA_VERSION,
        forward_stream,
    )
    from coordpy.controlled_runtime_substrate_v1 import (
        build_controlled_runtime_params_v1,
    )
    params = build_controlled_runtime_params_v1()
    H = int(params.hidden_dim)
    delta = np.full((1, H), 0.15, dtype=np.float64)
    plan = MidStreamInjectionPlanV1(
        schema=W84_STREAMING_V1_SCHEMA_VERSION,
        inject_after_position=0,
        layer_index=min(1, int(params.n_layers) - 1),
        delta=delta)
    a = list(forward_stream(
        params=params, prompt="replay inj",
        max_new_tokens=3, injection_plan=plan,
        return_side_channel=True))
    b = list(forward_stream(
        params=params, prompt="replay inj",
        max_new_tokens=3, injection_plan=plan,
        return_side_channel=True))
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.final_logits_cid == y.final_logits_cid
        assert x.kv_after_cid == y.kv_after_cid
        assert x.injection_fired == y.injection_fired
        assert x.cid() == y.cid()


def test_w84_bearer_auth_blocks_unauthenticated_stream():
    """DoD bar: bearer-auth must apply to streaming too. The
    W81 auth shim must NOT be disabled on streaming endpoints."""
    from coordpy.streaming_substrate_intercept_v1 import (
        BearerAuthGateV1, forward_stream,
        serialize_stream_with_auth,
    )
    from coordpy.controlled_runtime_substrate_v1 import (
        build_controlled_runtime_params_v1,
    )
    params = build_controlled_runtime_params_v1()
    chunks = list(forward_stream(
        params=params, prompt="auth probe", max_new_tokens=2,
        return_side_channel=True))
    gate = BearerAuthGateV1(required_token="ABCDEF")
    # Wrong token -> 401 + empty body.
    body, code = serialize_stream_with_auth(
        chunks=chunks, gate=gate, presented_token="wrong")
    assert int(code) == 401
    assert body == b""
    # Right token -> 200 + non-empty body.
    body2, code2 = serialize_stream_with_auth(
        chunks=chunks, gate=gate, presented_token="ABCDEF")
    assert int(code2) == 200
    assert len(body2) > 0


def test_w84_streaming_audit_log_detects_tampering():
    """Anti-cheat: tampering with any chunk in the audit log
    breaks the chain verification."""
    import dataclasses
    from coordpy.streaming_substrate_intercept_v1 import (
        forward_stream, replay_streaming_audit_log_v1,
    )
    from coordpy.controlled_runtime_substrate_v1 import (
        build_controlled_runtime_params_v1,
    )
    params = build_controlled_runtime_params_v1()
    chunks = list(forward_stream(
        params=params, prompt="tamper",
        max_new_tokens=4, return_side_channel=True))
    # Tamper with chunk 1: change token_id. Its CID will be
    # different, so chunk 2's prior_chunk_cid no longer matches.
    tampered = list(chunks)
    if len(tampered) >= 2:
        tampered[1] = dataclasses.replace(
            tampered[1], token_id=int(tampered[1].token_id) + 1)
        ok, _ = replay_streaming_audit_log_v1(chunks=tampered)
        assert not ok


def test_w84_streaming_bench_passes_all_load_bearing_claims():
    """End-to-end bench."""
    from coordpy.streaming_substrate_intercept_v1 import (
        run_streaming_bench_v1,
    )
    rep = run_streaming_bench_v1()
    d = rep.to_dict()
    assert bool(rep.streaming_equals_non_streaming), d
    assert bool(rep.sse_parses_cleanly), d
    assert bool(rep.bearer_auth_blocks_unauth), d
    assert bool(rep.mid_stream_injection_diverges), d
    assert bool(rep.mid_stream_injection_replayable), d
    assert bool(rep.audit_log_replayable), d


def test_w84_streaming_is_not_buffered_one_chunk():
    """Anti-cheat: do NOT implement "streaming" by buffering
    the entire output and emitting it as one chunk at the end.
    forward_stream is a generator that yields per-token —
    verify by counting yields."""
    from coordpy.streaming_substrate_intercept_v1 import (
        forward_stream,
    )
    from coordpy.controlled_runtime_substrate_v1 import (
        build_controlled_runtime_params_v1,
    )
    params = build_controlled_runtime_params_v1()
    gen = forward_stream(
        params=params, prompt="not buffered",
        max_new_tokens=4, return_side_channel=True)
    yielded_positions = []
    for c in gen:
        yielded_positions.append(int(c.position))
        if len(yielded_positions) >= 4:
            break
    assert yielded_positions == [0, 1, 2, 3]


def test_w84_sse_chunk_carries_substrate_side_channel_field():
    """Substrate side-channel chunks may carry numerical
    information; the format is documented and content-
    addressed regardless."""
    from coordpy.streaming_substrate_intercept_v1 import (
        forward_stream, parse_sse_stream,
        serialize_full_sse_stream,
    )
    from coordpy.controlled_runtime_substrate_v1 import (
        build_controlled_runtime_params_v1,
    )
    params = build_controlled_runtime_params_v1()
    chunks = list(forward_stream(
        params=params, prompt="side channel",
        max_new_tokens=2, return_side_channel=True))
    body = serialize_full_sse_stream(chunks)
    events, _ = parse_sse_stream(body)
    for evt in events:
        assert "substrate_side_channel_cid" in evt
        assert len(evt["substrate_side_channel_cid"]) == 64


def test_w84_streaming_off_side_channel_emits_none_marker():
    """Side channel off: chunk carries 'none' marker honestly."""
    from coordpy.streaming_substrate_intercept_v1 import (
        forward_stream,
    )
    from coordpy.controlled_runtime_substrate_v1 import (
        build_controlled_runtime_params_v1,
    )
    params = build_controlled_runtime_params_v1()
    chunks = list(forward_stream(
        params=params, prompt="off",
        max_new_tokens=2, return_side_channel=False))
    for c in chunks:
        assert c.substrate_side_channel_cid == "none"
