"""W84 / P1 #32 — Streaming substrate intercept tests."""

from __future__ import annotations

import json
import time as _time
import urllib.error
import urllib.request

import numpy as np
import pytest

from coordpy.controlled_runtime_substrate_v1 import (
    build_controlled_runtime_params_v1,
    tokenize_bytes_v79,
)
from coordpy.streaming_substrate_intercept_v1 import (
    StreamingHTTPServer,
    StreamingInjectionPlanV1,
    W84_STREAMING_EQUIVALENCE_FLOOR,
    W84_STREAMING_V1_SCHEMA_VERSION,
    forward_stream_controlled_runtime_v1,
    parse_sse_chunks,
    run_streaming_substrate_bench_v1,
)


# ---------------------------------------------------------------
# Bench bars.
# ---------------------------------------------------------------

def test_w84_streaming_equivalence_at_precision_floor():
    rep = run_streaming_substrate_bench_v1(
        prompt_text="context zero stream",
        n_tokens=8, injection_step=3)
    assert rep.final_hidden_equivalence_holds is True
    assert rep.final_logits_equivalence_holds is True
    # The actual difference is < W84_STREAMING_EQUIVALENCE_FLOOR.
    assert (rep.final_hidden_max_abs_diff
            < W84_STREAMING_EQUIVALENCE_FLOOR)
    assert (rep.final_logits_max_abs_diff
            < W84_STREAMING_EQUIVALENCE_FLOOR)


def test_w84_streaming_injection_diverges_post_step():
    rep = run_streaming_substrate_bench_v1(
        prompt_text="context zero stream",
        n_tokens=8, injection_step=3,
        injection_scale=0.20)
    assert rep.injection_diverges_post_step is True
    # The baseline and injected streaming chains must differ.
    assert (rep.streaming_chain_cid_baseline
            != rep.streaming_chain_cid_injected)


def test_w84_streaming_chain_is_deterministically_replayable():
    rep = run_streaming_substrate_bench_v1(
        prompt_text="context zero stream",
        n_tokens=8, injection_step=3)
    assert rep.streaming_chain_replayable is True


def test_w84_per_token_chunks_are_chained_cids():
    p = build_controlled_runtime_params_v1()
    ids = tokenize_bytes_v79("hello", max_len=5)
    rep = forward_stream_controlled_runtime_v1(
        params=p, input_token_ids=ids)
    # Each chunk's prev_chunk_cid is the prior chunk's CID
    # (or "genesis" for the first).
    chunks = rep.per_token_chunks
    assert chunks[0].prev_chunk_cid == "genesis"
    for i in range(1, len(chunks)):
        assert (chunks[i].prev_chunk_cid
                == chunks[i - 1].chunk_cid())


def test_w84_chunk_cid_changes_when_token_changes():
    p = build_controlled_runtime_params_v1()
    rep_a = forward_stream_controlled_runtime_v1(
        params=p,
        input_token_ids=tokenize_bytes_v79("abc", max_len=3))
    rep_b = forward_stream_controlled_runtime_v1(
        params=p,
        input_token_ids=tokenize_bytes_v79("xyz", max_len=3))
    assert (rep_a.streaming_chain_cid
            != rep_b.streaming_chain_cid)


# ---------------------------------------------------------------
# Mid-stream injection.
# ---------------------------------------------------------------

def test_w84_mid_stream_injection_fires_on_exact_step():
    p = build_controlled_runtime_params_v1()
    ids = tokenize_bytes_v79("inject test", max_len=6)
    per_layer = tuple(
        0.30 * np.ones((1, p.hidden_dim), dtype=np.float64)
        for _ in range(p.n_layers))
    plan = StreamingInjectionPlanV1(
        schema=W84_STREAMING_V1_SCHEMA_VERSION,
        fire_after_step=2,
        per_layer_delta=per_layer)
    rep = forward_stream_controlled_runtime_v1(
        params=p, input_token_ids=ids,
        injection_plan=plan)
    fired_steps = [
        c.step_index for c in rep.per_token_chunks
        if c.injection_fired_this_step]
    # The plan fires *between* step 2 and step 3, so step 3 has
    # the injection applied.
    assert fired_steps == [3]


def test_w84_injection_plan_is_content_addressed():
    p = build_controlled_runtime_params_v1()
    per_layer = tuple(
        0.30 * np.ones((1, p.hidden_dim), dtype=np.float64)
        for _ in range(p.n_layers))
    plan_a = StreamingInjectionPlanV1(
        schema=W84_STREAMING_V1_SCHEMA_VERSION,
        fire_after_step=2,
        per_layer_delta=per_layer)
    plan_b = StreamingInjectionPlanV1(
        schema=W84_STREAMING_V1_SCHEMA_VERSION,
        fire_after_step=2,
        per_layer_delta=per_layer)
    assert plan_a.cid() == plan_b.cid()
    assert len(plan_a.cid()) == 64


# ---------------------------------------------------------------
# SSE endpoint.
# ---------------------------------------------------------------

def _http_post(url: str, payload: dict, headers: dict = None) -> tuple[int, bytes]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url, data=data, method="POST",
        headers={"Content-Type": "application/json",
                 **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=4.0) as r:
            return int(r.status), r.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()


def test_w84_streaming_sse_endpoint_returns_real_sse():
    p = build_controlled_runtime_params_v1()
    server = StreamingHTTPServer(params=p, port=0)
    server.start()
    try:
        # Wait briefly for the server to be ready.
        _time.sleep(0.05)
        status, body = _http_post(
            url=f"http://127.0.0.1:{server.actual_port}"
            "/v1/substrate/forward_stream",
            payload={"prompt": "hello sse", "n_tokens": 4})
        assert status == 200
        chunks = parse_sse_chunks(body)
        # 4 token chunks (the [DONE] sentinel is filtered out).
        assert len(chunks) == 4
        assert all("step_index" in c for c in chunks)
        # Chain is well-formed.
        assert chunks[0]["prev_chunk_cid"] == "genesis"
    finally:
        server.stop()


def test_w84_streaming_sse_bearer_auth_required():
    p = build_controlled_runtime_params_v1()
    server = StreamingHTTPServer(
        params=p, port=0, bearer_token="secret-xyz")
    server.start()
    try:
        _time.sleep(0.05)
        # Without bearer → 401.
        status, body = _http_post(
            url=f"http://127.0.0.1:{server.actual_port}"
            "/v1/substrate/forward_stream",
            payload={"prompt": "hi", "n_tokens": 2})
        assert status == 401
        # With bearer → 200.
        status, body = _http_post(
            url=f"http://127.0.0.1:{server.actual_port}"
            "/v1/substrate/forward_stream",
            payload={"prompt": "hi", "n_tokens": 2},
            headers={"Authorization": "Bearer secret-xyz"})
        assert status == 200
    finally:
        server.stop()


def test_w84_streaming_sse_rejects_bad_paths():
    p = build_controlled_runtime_params_v1()
    server = StreamingHTTPServer(params=p, port=0)
    server.start()
    try:
        _time.sleep(0.05)
        status, _body = _http_post(
            url=f"http://127.0.0.1:{server.actual_port}"
            "/v1/wrong/path",
            payload={"prompt": "hi", "n_tokens": 2})
        assert status == 404
    finally:
        server.stop()


def test_w84_streaming_sse_chunks_match_in_process_chain():
    p = build_controlled_runtime_params_v1()
    server = StreamingHTTPServer(params=p, port=0)
    server.start()
    try:
        _time.sleep(0.05)
        status, body = _http_post(
            url=f"http://127.0.0.1:{server.actual_port}"
            "/v1/substrate/forward_stream",
            payload={"prompt": "stream", "n_tokens": 5})
        assert status == 200
        http_chunks = parse_sse_chunks(body)
        # In-process forward over the same params + prompt.
        ids = tokenize_bytes_v79("stream", max_len=5)
        rep = forward_stream_controlled_runtime_v1(
            params=p, input_token_ids=ids)
        assert len(http_chunks) == len(ids)
        # Verify the SSE chain is well-formed at the chunk level:
        # each prev_chunk_cid equals the preceding chunk's CID.
        for i in range(1, len(http_chunks)):
            prev_full = rep.per_token_chunks[i - 1].chunk_cid()
            assert http_chunks[i]["prev_chunk_cid"] == prev_full
        # Cross-check the chain at the token-id level.
        for h, ipc in zip(http_chunks, rep.per_token_chunks):
            assert int(h["token_id"]) == int(ipc.token_id)
            assert h["post_mlp_hidden_at_token_cid"] == (
                ipc.post_mlp_hidden_at_token_cid)
            assert h["logits_row_cid"] == ipc.logits_row_cid
    finally:
        server.stop()
