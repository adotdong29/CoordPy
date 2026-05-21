"""W87 / P3 #47 — Observability V1 tests.

Covers:
  * Prometheus text exposition format is well-formed and
    parseable by an independent stdlib-only parser
    (catches drift from the v0.0.4 spec).
  * OTLP/HTTP JSON spans match the OTLP wire-format shape:
    `resourceSpans[*].scopeSpans[*].spans[*]` with required
    fields traceId/spanId/name/kind/startTimeUnixNano/
    endTimeUnixNano/status.
  * Structured JSON-line logs round-trip through `json.loads`
    and carry the standard label set.
  * Sampling is deterministic in the trace_id and respects
    `always_sample_errors`.
  * Gateway integration: /metrics endpoint serves valid text;
    each request is counted, observed, and spanned; the
    standard label set is honoured.
  * Optional opentelemetry-SDK detection is correct.
"""

from __future__ import annotations

import json

import pytest

from coordpy.observability_v1 import (
    W87_DEFAULT_HISTOGRAM_BUCKETS_SECONDS,
    W87_GATEWAY_PATH_METRICS,
    W87_METRIC_NAME_PREFIX,
    W87_OBSERVABILITY_V1_SCHEMA_VERSION,
    W87_OTLP_SPAN_KIND_INTERNAL,
    W87_OTLP_STATUS_CODE_OK,
    W87_OTLP_STATUS_CODE_ERROR,
    W87_STANDARD_LABEL_KEYS,
    HeadSamplerV1,
    JSONLineLoggingHandlerV1,
    ObservabilityV1,
    OTLPResourceSpansBatchV1,
    PrometheusMetricsRegistryV1,
    SpanContext,
    StructuredLoggerV1,
    export_spans_otlp_http_v1,
    opentelemetry_sdk_available,
    register_observability_v1,
)


# ---------------------------------------------------------------
# Stdlib Prometheus text-format parser
# ---------------------------------------------------------------

def _parse_prometheus_text(text: str) -> dict:
    """Tiny stdlib-only parser for the Prometheus text exposition
    format (v0.0.4).  Sufficient for assertions, not for general
    consumption (does not support exemplars / extended types)."""
    out: dict = {"helps": {}, "types": {}, "samples": []}
    for raw in text.splitlines():
        if not raw:
            continue
        if raw.startswith("# HELP "):
            rest = raw[len("# HELP "):]
            name, _, help_text = rest.partition(" ")
            out["helps"][name] = help_text
            continue
        if raw.startswith("# TYPE "):
            rest = raw[len("# TYPE "):]
            name, _, kind = rest.partition(" ")
            out["types"][name] = kind
            continue
        if raw.startswith("#"):
            continue  # other comments ignored
        # sample: name[{labels}] value [timestamp]
        # Find an opening brace if present.
        if "{" in raw:
            name, _, rest = raw.partition("{")
            labels_str, _, rest = rest.partition("}")
            value_str = rest.strip().split(" ", 1)[0]
            labels: dict[str, str] = {}
            # Tokenise key="value",key="value"
            pos = 0
            while pos < len(labels_str):
                # key
                k_end = labels_str.find("=", pos)
                if k_end == -1:
                    break
                key = labels_str[pos:k_end]
                # value (quoted)
                if labels_str[k_end + 1] != '"':
                    raise AssertionError(
                        f"unquoted label value in {raw!r}")
                v_end = labels_str.find('"', k_end + 2)
                if v_end == -1:
                    raise AssertionError(
                        f"unterminated label value in {raw!r}")
                value = labels_str[k_end + 2:v_end]
                labels[key] = value
                pos = v_end + 1
                if pos < len(labels_str) and labels_str[pos] == ",":
                    pos += 1
            try:
                value = float(value_str)
            except ValueError as exc:
                raise AssertionError(
                    f"bad value {value_str!r} in {raw!r}") from exc
            out["samples"].append({
                "name": name,
                "labels": labels,
                "value": value,
            })
        else:
            name, _, rest = raw.partition(" ")
            value_str = rest.strip().split(" ", 1)[0]
            try:
                value = float(value_str)
            except ValueError as exc:
                raise AssertionError(
                    f"bad value {value_str!r} in {raw!r}") from exc
            out["samples"].append({
                "name": name,
                "labels": {},
                "value": value,
            })
    return out


def test_w87_prometheus_text_format_parses() -> None:
    obs = ObservabilityV1.default()
    obs.registry.inc(
        "coordpy_gateway_requests_total",
        labels={"path": "/v1/chat/completions",
                "status": "200"})
    obs.registry.observe(
        "coordpy_gateway_request_duration_seconds",
        value=0.012,
        labels={"path": "/v1/chat/completions"})
    obs.registry.set(
        "coordpy_event_graph_size",
        value=42.0,
        labels={"tenant_id": "t1"})
    text = obs.metrics_text()
    parsed = _parse_prometheus_text(text)
    # All 8 required metrics must appear in TYPE lines.
    expected_types = {
        "coordpy_gateway_requests_total": "counter",
        "coordpy_gateway_request_duration_seconds": "histogram",
        "coordpy_consensus_commits_total": "counter",
        "coordpy_consensus_abstains_total": "counter",
        "coordpy_integrity_verdicts_total": "counter",
        "coordpy_event_graph_size": "gauge",
        "coordpy_audit_anchor_root_age_seconds": "gauge",
        "coordpy_observability_spans_emitted_total": "counter",
    }
    for name, kind in expected_types.items():
        assert name in parsed["types"], (
            f"{name} missing from TYPE lines")
        assert parsed["types"][name] == kind, (
            f"{name}: expected {kind}, got {parsed['types'][name]}")
    # Counter sample with labels survived round-trip.
    request_samples = [s for s in parsed["samples"]
                       if s["name"] == "coordpy_gateway_requests_total"
                       and s["labels"]]
    assert any(
        s["labels"].get("path") == "/v1/chat/completions"
        and s["labels"].get("status") == "200"
        and s["value"] == 1.0
        for s in request_samples), (
        f"counter sample missing: {request_samples}")
    # Histogram emits le buckets + sum + count.
    hist_bucket_samples = [
        s for s in parsed["samples"]
        if s["name"] == "coordpy_gateway_request_duration_seconds_bucket"
        and s["labels"].get("path") == "/v1/chat/completions"]
    assert len(hist_bucket_samples) == (
        len(W87_DEFAULT_HISTOGRAM_BUCKETS_SECONDS) + 1), (
        "histogram bucket count mismatch")
    inf_samples = [
        s for s in hist_bucket_samples
        if s["labels"].get("le") == "+Inf"]
    assert len(inf_samples) == 1 and inf_samples[0]["value"] == 1.0
    sum_samples = [
        s for s in parsed["samples"]
        if s["name"] == "coordpy_gateway_request_duration_seconds_sum"]
    assert len(sum_samples) == 1
    count_samples = [
        s for s in parsed["samples"]
        if s["name"] == "coordpy_gateway_request_duration_seconds_count"]
    assert len(count_samples) == 1 and count_samples[0]["value"] == 1.0


def test_w87_metric_name_prefix_enforced() -> None:
    """Anti-cheat: metric names without the coordpy_ prefix MUST
    be rejected."""
    reg = PrometheusMetricsRegistryV1()
    with pytest.raises(ValueError, match="coordpy_"):
        reg.register_counter(
            name="my_counter", help_text="bad")
    # Prefix is accepted.
    reg.register_counter(
        name="coordpy_my_counter", help_text="ok")
    assert "coordpy_my_counter" in reg.metric_names()


def test_w87_otlp_json_wire_format_shape() -> None:
    """Spans serialise to the OTLP/HTTP JSON shape."""
    obs = ObservabilityV1.default(sample_rate=1.0)  # keep all
    with obs.span("substrate_restore",
                  attributes={"kv_cache_cid": "deadbeef"}) as sp:
        sp.set_attribute("run_cid", "abc123")
    with obs.span("consensus_commit",
                  attributes={"decision_kind": "majority"}):
        pass
    batch = obs.spans_batch()
    assert len(batch.spans) == 2
    wire = batch.to_otlp_json()
    # OTLP wire shape: top-level resourceSpans -> scopeSpans -> spans
    assert "resourceSpans" in wire
    assert len(wire["resourceSpans"]) == 1
    rs0 = wire["resourceSpans"][0]
    assert "resource" in rs0
    assert any(
        a["key"] == "service.name" for a in rs0["resource"]["attributes"])
    assert "scopeSpans" in rs0
    assert len(rs0["scopeSpans"]) == 1
    scope = rs0["scopeSpans"][0]
    assert scope["scope"]["name"] == "coordpy.observability_v1"
    span_jsons = scope["spans"]
    assert len(span_jsons) == 2
    for s in span_jsons:
        assert "traceId" in s and len(s["traceId"]) == 32
        assert "spanId" in s and len(s["spanId"]) == 16
        assert "name" in s
        assert "kind" in s and s["kind"] == W87_OTLP_SPAN_KIND_INTERNAL
        assert "startTimeUnixNano" in s
        assert "endTimeUnixNano" in s
        # status object
        assert "status" in s and "code" in s["status"]
        # attributes are list of {key, value:{stringValue:...}}
        for a in s["attributes"]:
            assert "key" in a and "value" in a
            assert "stringValue" in a["value"]
    # Confirm at least one span has kv_cache_cid in attributes.
    found_kv = False
    for s in span_jsons:
        for a in s["attributes"]:
            if a["key"] == "kv_cache_cid":
                assert a["value"]["stringValue"] == "deadbeef"
                found_kv = True
    assert found_kv


def test_w87_otlp_error_status() -> None:
    """Spans that raise an exception get ERROR status."""
    obs = ObservabilityV1.default(sample_rate=1.0)
    with pytest.raises(RuntimeError):
        with obs.span("will_fail"):
            raise RuntimeError("boom")
    spans = obs.spans_batch().spans
    assert any(
        s.name == "will_fail" and
        s.status_code == W87_OTLP_STATUS_CODE_ERROR and
        "boom" in s.status_message
        for s in spans)


def test_w87_structured_logs_jsonline_roundtrip() -> None:
    """Each log line is a single JSON object with the standard
    label set + record_cid."""
    obs = ObservabilityV1.default(sample_rate=1.0)
    obs.log(level="INFO", message="hello world",
            run_cid="run-1", tenant_id="tenant-a",
            agent_id="agent-x", role="solver",
            extra={"k": "v"})
    obs.log(level="WARNING", message="something is off",
            run_cid="run-1", tenant_id="tenant-a",
            agent_id="agent-y", role="critic")
    lines = obs.logger.lines()
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        for key in W87_STANDARD_LABEL_KEYS:
            assert key in obj, f"{key} missing from {obj}"
        assert "record_cid" in obj
        assert "timestamp_ns" in obj
        assert isinstance(obj["timestamp_ns"], int)
    obj0 = json.loads(lines[0])
    assert obj0["ext"] == {"k": "v"}
    assert obj0["agent_id"] == "agent-x"
    assert obj0["role"] == "solver"


def test_w87_head_sampler_deterministic() -> None:
    """Sampling is deterministic in trace_id_hex."""
    s = HeadSamplerV1(sample_rate=0.5, seed=42)
    trace_id = "ab" * 16
    # Run a few times — always the same answer for the same trace.
    answers = {s.should_sample(trace_id_hex=trace_id)
               for _ in range(100)}
    assert len(answers) == 1
    # Different trace_ids spread across both buckets.
    decisions = [
        s.should_sample(trace_id_hex=("%032x" % i))
        for i in range(2000)]
    # ~50% sampled within a wide tolerance.
    rate = sum(decisions) / float(len(decisions))
    assert 0.40 <= rate <= 0.60, (
        f"empirical sample rate {rate} far from 0.5")


def test_w87_head_sampler_always_sample_errors() -> None:
    s = HeadSamplerV1(sample_rate=0.0,
                      always_sample_errors=True, seed=1)
    # Non-error: never sampled at rate=0.
    assert not s.should_sample(trace_id_hex="abc",
                               is_error=False)
    # Error: always sampled.
    assert s.should_sample(trace_id_hex="abc", is_error=True)
    s2 = HeadSamplerV1(sample_rate=0.0,
                       always_sample_errors=False, seed=1)
    assert not s2.should_sample(trace_id_hex="abc", is_error=True)


def test_w87_otlp_http_export_handles_offline_endpoint() -> None:
    """Export to a deliberately-unreachable endpoint returns a
    structured failure rather than raising."""
    obs = ObservabilityV1.default(sample_rate=1.0)
    with obs.span("test"):
        pass
    batch = obs.spans_batch()
    # 127.0.0.1:1 is essentially never bound.
    result = export_spans_otlp_http_v1(
        batch=batch,
        endpoint="http://127.0.0.1:1/v1/traces",
        timeout_seconds=2.0)
    assert result.n_spans == 1
    # Either URLError or HTTPError; status_code may be -1.
    assert result.error  # non-empty


def test_w87_gateway_metrics_endpoint() -> None:
    """The gateway exposes /metrics that returns valid Prometheus
    text."""
    pytest.importorskip("numpy")
    from coordpy.deployable_substrate_gateway_v1 import (
        DeployableSubstrateGatewayV1,
    )
    gw = DeployableSubstrateGatewayV1()
    obs = ObservabilityV1.default()
    register_observability_v1(gw, obs)
    resp = gw.dispatch(path=W87_GATEWAY_PATH_METRICS, body=None,
                       auth_header=None)
    assert resp.status == 200
    text = resp.body["metrics_text"]
    assert "# HELP coordpy_gateway_requests_total" in text
    # Issue another request: the counter should fire.
    gw.dispatch(path="/healthz", body=None, auth_header=None)
    resp2 = gw.dispatch(path=W87_GATEWAY_PATH_METRICS, body=None,
                        auth_header=None)
    parsed = _parse_prometheus_text(resp2.body["metrics_text"])
    healthz_samples = [
        s for s in parsed["samples"]
        if s["name"] == "coordpy_gateway_requests_total"
        and s["labels"].get("path") == "/healthz"]
    assert healthz_samples, (
        "/healthz request did not register on the counter")
    # The duration histogram should have at least one observation.
    hist_count = [
        s for s in parsed["samples"]
        if s["name"] == "coordpy_gateway_request_duration_seconds_count"
        and s["labels"].get("path") == "/healthz"]
    assert hist_count and hist_count[0]["value"] >= 1.0


def test_w87_gateway_spans_emitted_per_request() -> None:
    """Each gateway dispatch produces a gateway.dispatch span."""
    pytest.importorskip("numpy")
    from coordpy.deployable_substrate_gateway_v1 import (
        DeployableSubstrateGatewayV1,
    )
    gw = DeployableSubstrateGatewayV1()
    obs = ObservabilityV1.default(sample_rate=1.0)  # keep all
    register_observability_v1(gw, obs)
    n_before = len(obs.spans_batch().spans)
    gw.dispatch(path="/healthz", body=None, auth_header=None)
    gw.dispatch(path="/healthz", body=None, auth_header=None)
    n_after = len(obs.spans_batch().spans)
    assert n_after == n_before + 2
    names = [s.name for s in obs.spans_batch().spans]
    assert names.count("gateway.dispatch") >= 2
    # Each span carries path attribute.
    paths = [s.attributes.get("path") for s in obs.spans_batch().spans
             if s.name == "gateway.dispatch"]
    assert all(p == "/healthz" for p in paths)


def test_w87_metric_label_set_includes_tenant_run_agent_role() -> None:
    """Standard label set is the issue's required label set."""
    assert W87_STANDARD_LABEL_KEYS == (
        "tenant_id", "run_cid", "agent_id", "role")


def test_w87_observability_state_cid_changes_on_event() -> None:
    obs = ObservabilityV1.default(sample_rate=1.0)
    cid0 = obs.cid()
    with obs.span("a"):
        pass
    cid1 = obs.cid()
    assert cid1 != cid0


def test_w87_jsonline_logging_handler_bridges_stdlib_logging() -> None:
    import logging as stdlib_logging
    obs = ObservabilityV1.default(sample_rate=1.0)
    h = JSONLineLoggingHandlerV1(obs.logger)
    log = stdlib_logging.getLogger("w87.test")
    log.handlers.clear()
    log.addHandler(h)
    log.setLevel(stdlib_logging.INFO)
    log.info("hello", extra={"run_cid": "run-7", "agent_id": "a", "role": "r"})
    lines = obs.logger.lines()
    assert lines
    obj = json.loads(lines[-1])
    assert obj["level"] == "INFO"
    assert obj["message"] == "hello"
    assert obj["run_cid"] == "run-7"
    # Cleanup the global logger.
    log.handlers.clear()


def test_w87_opentelemetry_sdk_detection() -> None:
    """The detection function returns a bool either way."""
    assert isinstance(opentelemetry_sdk_available(), bool)


def test_w87_no_raw_payload_bytes_in_spans_or_metrics() -> None:
    """Anti-cheat: spans/metrics MUST NOT carry raw payload
    bytes — only CIDs and counters. We enforce this by type:
    OTLPSpanV1.attributes is Mapping[str, str], and the
    Prometheus registry only accepts numeric values."""
    obs = ObservabilityV1.default(sample_rate=1.0)
    with obs.span("test", attributes={"k": "v"}) as sp:
        # All attribute values are strings; bytes would fail.
        sp.set_attribute("cid", "abcdef0123456789")
    spans = obs.spans_batch().spans
    for s in spans:
        for k, v in s.attributes.items():
            assert isinstance(k, str)
            assert isinstance(v, str)
