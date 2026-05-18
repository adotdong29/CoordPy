"""W81 / P1 #7 — Deployable substrate gateway tests.

Covers:
- programmatic dispatch through each route
- OpenAI-compatible chat completion shape
- substrate side-channel as an explicit option
- substrate forward / replay / conformance / capabilities
- bearer-token auth shim
- content addressing on request + response
- real HTTP transport via stdlib http.server
- end-to-end bench report
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest


def _build_gateway(*, bearer_token=None,
                   allow_side_channel=True):
    from coordpy.deployable_substrate_gateway_v1 import (
        DeployableSubstrateGatewayV1,
        GatewayConfigV1,
    )
    return DeployableSubstrateGatewayV1(
        config=GatewayConfigV1(
            bearer_token=bearer_token,
            allow_substrate_side_channel=allow_side_channel))


def test_w81_gateway_healthz_open():
    from coordpy.deployable_substrate_gateway_v1 import (
        W81_GATEWAY_PATH_HEALTHZ,
    )
    gw = _build_gateway()
    resp = gw.dispatch(
        path=W81_GATEWAY_PATH_HEALTHZ, body={})
    assert int(resp.status) == 200
    assert bool(resp.body["ok"])
    assert isinstance(resp.body["gateway_cid"], str)
    assert isinstance(resp.body["routes"], list)
    assert len(resp.body["routes"]) >= 6


def test_w81_gateway_chat_completion_openai_shape():
    from coordpy.deployable_substrate_gateway_v1 import (
        W81_GATEWAY_PATH_CHAT_COMPLETIONS,
    )
    gw = _build_gateway()
    resp = gw.dispatch(
        path=W81_GATEWAY_PATH_CHAT_COMPLETIONS,
        body={
            "model": "any",
            "messages": [
                {"role": "user", "content": "hi"}],
            "max_tokens": 4,
        })
    assert int(resp.status) == 200
    body = resp.body
    assert "id" in body
    assert body["object"] == "chat.completion"
    assert isinstance(body["choices"], list)
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert isinstance(body["usage"], dict)
    # By default we don't return the side channel.
    assert "substrate_side_channel" not in body
    # Streaming flag is explicit, not magical.
    assert body["streaming_supported"] is False


def test_w81_gateway_substrate_side_channel_explicit_opt_in():
    from coordpy.deployable_substrate_gateway_v1 import (
        W81_GATEWAY_PATH_CHAT_COMPLETIONS,
    )
    gw = _build_gateway()
    resp = gw.dispatch(
        path=W81_GATEWAY_PATH_CHAT_COMPLETIONS,
        body={
            "model": "any",
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 2,
            "substrate_options": {
                "return_side_channel": True,
            },
        })
    assert int(resp.status) == 200
    assert "substrate_side_channel" in resp.body
    sc = resp.body["substrate_side_channel"]
    assert "forward_trace_cid" in sc
    assert "kv_cache_cid" in sc
    assert "per_layer_attn_probs_cids" in sc
    assert len(sc["per_layer_attn_probs_cids"]) >= 1


def test_w81_gateway_substrate_side_channel_can_be_locked_down():
    from coordpy.deployable_substrate_gateway_v1 import (
        W81_GATEWAY_PATH_CHAT_COMPLETIONS,
    )
    gw = _build_gateway(allow_side_channel=False)
    resp = gw.dispatch(
        path=W81_GATEWAY_PATH_CHAT_COMPLETIONS,
        body={
            "model": "any",
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 2,
            "substrate_options": {
                "return_side_channel": True,
            },
        })
    assert int(resp.status) == 200
    # Even though caller asked, the gateway config blocked it.
    assert "substrate_side_channel" not in resp.body


def test_w81_gateway_substrate_forward_endpoint_returns_axes():
    from coordpy.deployable_substrate_gateway_v1 import (
        W81_GATEWAY_PATH_SUBSTRATE_FORWARD,
    )
    gw = _build_gateway()
    resp = gw.dispatch(
        path=W81_GATEWAY_PATH_SUBSTRATE_FORWARD,
        body={"prompt": "hello substrate"})
    assert int(resp.status) == 200
    body = resp.body
    assert isinstance(body["forward_trace_cid"], str)
    assert isinstance(body["kv_cache_cid"], str)
    assert isinstance(body["per_layer_attn_probs_cids"], list)
    assert len(body["per_layer_hidden_state_cids"]) >= 1
    assert isinstance(body["substrate_axes"], list)
    assert len(body["substrate_axes"]) >= 5


def test_w81_gateway_substrate_replay_is_byte_identical():
    from coordpy.deployable_substrate_gateway_v1 import (
        W81_GATEWAY_PATH_SUBSTRATE_REPLAY,
    )
    gw = _build_gateway()
    resp = gw.dispatch(
        path=W81_GATEWAY_PATH_SUBSTRATE_REPLAY,
        body={"prompt": "byte identical replay check",
              "split_at": 4})
    assert int(resp.status) == 200
    body = resp.body
    # In-repo NumPy controlled runtime is fully deterministic.
    assert float(body["final_logit_max_abs_diff"]) < 1e-8
    assert bool(body["byte_identical_within_1e_minus_8"])


def test_w81_gateway_substrate_replay_accepts_registered_cids():
    from coordpy.deployable_substrate_gateway_v1 import (
        W81_GATEWAY_PATH_SUBSTRATE_FORWARD,
        W81_GATEWAY_PATH_SUBSTRATE_REPLAY,
    )
    gw = _build_gateway()
    fwd = gw.dispatch(
        path=W81_GATEWAY_PATH_SUBSTRATE_FORWARD,
        body={"prompt": "cid backed replay"})
    assert int(fwd.status) == 200
    replay = gw.dispatch(
        path=W81_GATEWAY_PATH_SUBSTRATE_REPLAY,
        body={
            "forward_trace_cid": fwd.body["forward_trace_cid"],
            "kv_cache_cid": fwd.body["kv_cache_cid"],
        })
    assert int(replay.status) == 200
    body = replay.body
    assert body["replay_source"] == "registered_trace"
    assert bool(body["registered_forward_trace_match"])
    assert bool(body["registered_kv_cache_match"])
    assert float(body["final_logit_max_abs_diff"]) < 1e-8


def test_w81_gateway_substrate_conformance_passes_12_axes():
    from coordpy.deployable_substrate_gateway_v1 import (
        W81_GATEWAY_PATH_SUBSTRATE_CONFORMANCE,
    )
    gw = _build_gateway()
    resp = gw.dispatch(
        path=W81_GATEWAY_PATH_SUBSTRATE_CONFORMANCE, body={})
    assert int(resp.status) == 200
    body = resp.body
    assert int(body["n_pass"]) == 12
    assert int(body["n_fail"]) == 0


def test_w81_gateway_substrate_capabilities_is_honest():
    from coordpy.deployable_substrate_gateway_v1 import (
        W81_GATEWAY_PATH_SUBSTRATE_CAPABILITIES,
    )
    gw = _build_gateway()
    resp = gw.dispatch(
        path=W81_GATEWAY_PATH_SUBSTRATE_CAPABILITIES, body={})
    assert int(resp.status) == 200
    body = resp.body
    assert isinstance(body["substrate_axes"], list)
    assert isinstance(body["honest_caveats"], list)
    assert len(body["honest_caveats"]) >= 2
    # The caveats must mention the hosted-wall non-claim.
    joined = " ".join(body["honest_caveats"]).lower()
    assert "hosted" in joined


def test_w81_gateway_auth_rejects_missing_token():
    from coordpy.deployable_substrate_gateway_v1 import (
        W81_GATEWAY_PATH_CHAT_COMPLETIONS,
    )
    gw = _build_gateway(bearer_token="research-secret-abc")
    resp = gw.dispatch(
        path=W81_GATEWAY_PATH_CHAT_COMPLETIONS,
        body={
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 1,
        })
    assert int(resp.status) == 401


def test_w81_gateway_auth_accepts_valid_token():
    from coordpy.deployable_substrate_gateway_v1 import (
        W81_GATEWAY_PATH_CHAT_COMPLETIONS,
    )
    gw = _build_gateway(bearer_token="research-secret-abc")
    resp = gw.dispatch(
        path=W81_GATEWAY_PATH_CHAT_COMPLETIONS,
        body={
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 1,
        },
        auth_header="Bearer research-secret-abc")
    assert int(resp.status) == 200


def test_w81_gateway_unknown_route_404():
    gw = _build_gateway()
    resp = gw.dispatch(
        path="/nope", body={})
    assert int(resp.status) == 404
    assert "valid_routes" in resp.body["error"]


def test_w81_gateway_content_addressing():
    """Identical request -> identical response CID."""
    from coordpy.deployable_substrate_gateway_v1 import (
        W81_GATEWAY_PATH_CHAT_COMPLETIONS,
    )
    gw_a = _build_gateway()
    gw_b = _build_gateway()
    body = {
        "messages": [{"role": "user", "content": "deterministic"}],
        "max_tokens": 2,
    }
    r_a = gw_a.dispatch(
        path=W81_GATEWAY_PATH_CHAT_COMPLETIONS, body=body)
    r_b = gw_b.dispatch(
        path=W81_GATEWAY_PATH_CHAT_COMPLETIONS, body=body)
    assert r_a.request_cid == r_b.request_cid
    assert r_a.response_cid == r_b.response_cid


def test_w81_gateway_audit_log_grows_on_each_call():
    from coordpy.deployable_substrate_gateway_v1 import (
        W81_GATEWAY_PATH_HEALTHZ,
    )
    gw = _build_gateway()
    assert len(gw.audit) == 0
    gw.dispatch(path=W81_GATEWAY_PATH_HEALTHZ, body={})
    assert len(gw.audit) == 1
    gw.dispatch(path=W81_GATEWAY_PATH_HEALTHZ, body={})
    assert len(gw.audit) == 2
    assert int(gw.request_counter) == 2


def test_w81_gateway_serves_over_real_http_transport():
    """Bind a real TCP listener and round-trip a request."""
    from coordpy.deployable_substrate_gateway_v1 import (
        W81_GATEWAY_PATH_HEALTHZ,
    )
    gw = _build_gateway()
    server = gw.serve_forever(host="127.0.0.1", port=0)
    try:
        url = (
            f"http://127.0.0.1:{server.actual_port}"
            f"{W81_GATEWAY_PATH_HEALTHZ}")
        with urllib.request.urlopen(url, timeout=5) as resp:
            assert int(resp.status) == 200
            body = json.loads(resp.read().decode("utf-8"))
            assert body["ok"] is True
            # The gateway CID matches the in-process gateway.
            assert body["gateway_cid"] == str(gw.cid())
            # Audit headers are populated.
            assert resp.headers.get(
                "X-CoordPy-Gateway-Cid") is not None
            assert resp.headers.get(
                "X-CoordPy-Request-Cid") is not None
            assert resp.headers.get(
                "X-CoordPy-Response-Cid") is not None
    finally:
        server.stop()


def test_w81_gateway_http_chat_completion_round_trip():
    from coordpy.deployable_substrate_gateway_v1 import (
        W81_GATEWAY_PATH_CHAT_COMPLETIONS,
    )
    gw = _build_gateway()
    server = gw.serve_forever(host="127.0.0.1", port=0)
    try:
        url = (
            f"http://127.0.0.1:{server.actual_port}"
            f"{W81_GATEWAY_PATH_CHAT_COMPLETIONS}")
        payload = json.dumps({
            "model": "controlled",
            "messages": [
                {"role": "user", "content": "hello"}],
            "max_tokens": 2,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert int(resp.status) == 200
            body = json.loads(resp.read().decode("utf-8"))
            assert body["object"] == "chat.completion"
            assert len(body["choices"]) == 1
    finally:
        server.stop()


def test_w81_gateway_http_auth_blocks_unauthenticated():
    from coordpy.deployable_substrate_gateway_v1 import (
        W81_GATEWAY_PATH_CHAT_COMPLETIONS,
    )
    gw = _build_gateway(bearer_token="t-research-7")
    server = gw.serve_forever(host="127.0.0.1", port=0)
    try:
        url = (
            f"http://127.0.0.1:{server.actual_port}"
            f"{W81_GATEWAY_PATH_CHAT_COMPLETIONS}")
        payload = json.dumps({
            "messages": [
                {"role": "user", "content": "hello"}],
            "max_tokens": 1,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST")
        with pytest.raises(urllib.error.HTTPError) as einfo:
            urllib.request.urlopen(req, timeout=5)
        assert int(einfo.value.code) == 401
    finally:
        server.stop()


def test_w81_gateway_end_to_end_bench():
    from coordpy.deployable_substrate_gateway_v1 import (
        run_gateway_end_to_end_bench_v1,
    )
    rep = run_gateway_end_to_end_bench_v1()
    assert int(rep.conformance_n_pass) == 12
    assert int(rep.conformance_n_fail) == 0
    assert isinstance(rep.cid(), str)
    assert len(rep.cid()) == 64
    assert len(rep.routed_endpoints) == 4


def test_w81_gateway_stream_true_returns_non_streaming_body():
    """`stream=true` requests are answered with a single
    non-streaming JSON body; `streaming_supported` is false and
    `stream_requested` echoes the caller's intent."""
    from coordpy.deployable_substrate_gateway_v1 import (
        W81_GATEWAY_PATH_CHAT_COMPLETIONS,
    )
    gw = _build_gateway()
    resp = gw.dispatch(
        path=W81_GATEWAY_PATH_CHAT_COMPLETIONS,
        body={
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 2,
            "stream": True,
        })
    assert int(resp.status) == 200
    assert resp.body["streaming_supported"] is False
    assert resp.body["stream_requested"] is True
    # Body is a single complete chat-completion, not chunked.
    assert resp.body["object"] == "chat.completion"
    assert len(resp.body["choices"]) == 1


def test_w81_gateway_chat_completion_has_system_fingerprint():
    """OpenAI shape compatibility: response carries
    `system_fingerprint`, content-addressed against the
    runtime params CID so identical configs produce identical
    fingerprints."""
    from coordpy.deployable_substrate_gateway_v1 import (
        W81_GATEWAY_PATH_CHAT_COMPLETIONS,
    )
    gw_a = _build_gateway()
    gw_b = _build_gateway()
    body = {
        "messages": [{"role": "user", "content": "fp"}],
        "max_tokens": 1,
    }
    r_a = gw_a.dispatch(
        path=W81_GATEWAY_PATH_CHAT_COMPLETIONS, body=body)
    r_b = gw_b.dispatch(
        path=W81_GATEWAY_PATH_CHAT_COMPLETIONS, body=body)
    assert "system_fingerprint" in r_a.body
    assert isinstance(r_a.body["system_fingerprint"], str)
    assert r_a.body["system_fingerprint"].startswith("fp_")
    # Same runtime params -> same fingerprint.
    assert (r_a.body["system_fingerprint"]
            == r_b.body["system_fingerprint"])


def test_w81_gateway_chat_completion_logprobs_is_null_by_default():
    """OpenAI strict-spec compatibility: `choices[].logprobs`
    is null when not requested. The CoordPy research extension
    `x_coordpy_raw_logits_head` is namespaced with the `x_`
    prefix so strict SDKs ignore it."""
    from coordpy.deployable_substrate_gateway_v1 import (
        W81_GATEWAY_PATH_CHAT_COMPLETIONS,
    )
    gw = _build_gateway()
    resp = gw.dispatch(
        path=W81_GATEWAY_PATH_CHAT_COMPLETIONS,
        body={
            "messages": [{"role": "user", "content": "logp"}],
            "max_tokens": 1,
        })
    choice = resp.body["choices"][0]
    assert choice["logprobs"] is None
    # Raw research-extension is present under x_ prefix.
    assert "x_coordpy_raw_logits_head" in choice
    assert isinstance(
        choice["x_coordpy_raw_logits_head"], list)


def test_w81_gateway_openai_sdk_round_trip():
    """If `openai-python` is available, verify a real
    `openai.OpenAI` client can talk to the gateway over HTTP.
    This is the load-bearing "standard client can talk to it"
    DoD criterion for P1 #7."""
    openai = pytest.importorskip("openai")
    from coordpy.deployable_substrate_gateway_v1 import (
        W81_GATEWAY_PATH_CHAT_COMPLETIONS,
    )
    gw = _build_gateway()
    server = gw.serve_forever(host="127.0.0.1", port=0)
    try:
        client = openai.OpenAI(
            base_url=f"http://127.0.0.1:{server.actual_port}/v1",
            api_key="dummy")
        resp = client.chat.completions.create(
            model="controlled",
            messages=[
                {"role": "user", "content": "sdk"},
            ],
            max_tokens=2)
        assert resp.object == "chat.completion"
        assert len(resp.choices) == 1
        assert resp.choices[0].message.role == "assistant"
        # logprobs is null in our response — the SDK should
        # accept that.
        assert resp.choices[0].logprobs is None
    finally:
        server.stop()


def test_w81_gateway_witness_chains_state():
    from coordpy.deployable_substrate_gateway_v1 import (
        W81_GATEWAY_PATH_HEALTHZ,
        emit_gateway_witness_v1,
    )
    gw = _build_gateway()
    w1 = emit_gateway_witness_v1(gateway=gw)
    gw.dispatch(path=W81_GATEWAY_PATH_HEALTHZ, body={})
    w2 = emit_gateway_witness_v1(gateway=gw)
    assert w1.request_counter == 0
    assert w2.request_counter == 1
    assert w1.cid() != w2.cid()
    # Gateway-CID is stable across calls (it's structural, not
    # tied to request count).
    assert w1.gateway_cid == w2.gateway_cid
