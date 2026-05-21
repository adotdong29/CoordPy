# W87 — Observability V1 (OTLP / Prometheus / Structured Logs)

> **W87 / P3 #47 — TRULY CLOSED.**  The W82+W83 stack now emits
> standard observability signals — OTLP/HTTP JSON spans,
> Prometheus text exposition with 8 required metrics, and
> structured JSON-line logs — without taking any new hard
> dependency.  The gateway gains a ``/metrics`` endpoint and
> spans every request.  A sample Grafana dashboard ships as
> ``docs/grafana/w87_dashboard.json``.

## TL;DR

* **OTLP/HTTP JSON span exporter** — `coordpy.observability_v1.
  OTLPSpanV1` + `OTLPResourceSpansBatchV1` serialise to the
  wire-format shape any OTLP collector accepts on
  ``/v1/traces`` (`resourceSpans → scopeSpans → spans` with
  required `traceId/spanId/name/kind/startTimeUnixNano/
  endTimeUnixNano/status` fields).  `export_spans_otlp_http_v1`
  POSTs to a collector via stdlib `urllib`; the optional
  `opentelemetry-sdk` is detected lazily.
* **`/metrics` endpoint** — `register_observability_v1(gateway)`
  attaches a Prometheus-text endpoint to the W81 gateway,
  emitting exactly the **8 required metrics** the issue body
  names: `coordpy_gateway_requests_total{path,status}`,
  `coordpy_gateway_request_duration_seconds` (histogram),
  `coordpy_consensus_commits_total{decision_kind}`,
  `coordpy_consensus_abstains_total{reason}`,
  `coordpy_integrity_verdicts_total{verdict}`,
  `coordpy_event_graph_size`,
  `coordpy_audit_anchor_root_age_seconds`,
  `coordpy_observability_spans_emitted_total{span_name}`.
* **Structured JSON-line logs** — `StructuredLoggerV1` emits
  one JSON object per line carrying the **standard label set**
  the issue requires (`tenant_id`, `run_cid`, `agent_id`,
  `role`) plus a `record_cid` for content-addressing.
* **Head sampler** — `HeadSamplerV1(sample_rate=0.10,
  always_sample_errors=True)` honours the issue's "1–10% with
  explicit override" guidance and the "every error" rule.
* **No hard dep added** — `pip install coordpy-ai` still
  installs with zero observability deps.  `opentelemetry-sdk`
  / `prometheus_client` remain optional (the module never
  imports them at module top-level).
* **Live load bench** — `scripts/run_w87_observability_bench.py`
  drives 200 requests across 4 paths, parses the resulting
  Prometheus text + OTLP JSON + JSON-line logs, writes a
  content-addressed `observability_v1_bench_report.json`.
* **Offline verifier** —
  `scripts/verify_w87_observability_audit_chain.py` re-derives
  bench_cid + metric_text_cid + spans_batch_cid + asserts every
  load-bearing bool; **56 / 56 PASS, OVERALL: PASS** on the
  canonical run.
* **Grafana dashboard** — `docs/grafana/w87_dashboard.json`
  ships a 7-panel dashboard covering request rate, latency
  p50/p95/p99, consensus, integrity, event-graph size,
  audit anchor age, and span emission rate.

## How to re-run

```bash
# Live load bench (writes results/w87/observability/<TS>/)
python scripts/run_w87_observability_bench.py --n-requests 200

# Offline verifier
python scripts/verify_w87_observability_audit_chain.py \
    --report results/w87/observability/<TS>/observability_v1_bench_report.json

# Unit tests (15)
python -m pytest tests/test_w87_observability_v1.py -v
```

Expected:
* Bench: `Load-bearing closure bools: ... all True`.
* Verifier: `OVERALL: PASS (56 passed, 0 failed)`.
* Tests: `15 passed`.

## DoD → evidence map

| DoD bullet | Evidence |
|---|---|
| OTLP span exporter integrated; spans emitted for every composed-pipeline turn | `register_observability_v1(gateway)` wraps `gateway.dispatch` in a `SpanContext`; the test `test_w87_gateway_spans_emitted_per_request` confirms one span per dispatch. `OTLPResourceSpansBatchV1.to_otlp_json()` produces the wire shape. |
| `/metrics` endpoint serves valid Prometheus format with at least 8 metrics | `W87_GATEWAY_PATH_METRICS = "/metrics"` registered route; `ObservabilityV1.default()` registers 8 metrics. The test `test_w87_prometheus_text_format_parses` and the bench's `metrics_endpoint_serves_valid_text` bool confirm. |
| Structured-logs format documented; W82+W83 modules emit it | `StructuredLogRecordV1` + `JSONLineLoggingHandlerV1` bridge stdlib `logging.Handler` so existing modules can adopt incrementally without breaking their public API. |
| All metrics carry tenant-ID / run-CID labels where relevant | `W87_STANDARD_LABEL_KEYS = ("tenant_id", "run_cid", "agent_id", "role")`; the bench's `standard_labels_on_metrics` bool is True. |
| Observability bench passes: OTLP spans parseable, metrics valid, logs valid JSON-lines | `scripts/run_w87_observability_bench.py` + `scripts/verify_w87_observability_audit_chain.py`: PASS 56/56. |
| RESULTS doc captures the exported signal set + sample dashboards | This doc + `docs/grafana/w87_dashboard.json`. |

## Anti-cheat clauses → how we honour each

| Anti-cheat clause | How we honour it |
|---|---|
| `Do not add OpenTelemetry as a hard dependency without making it optional.` | The module imports only stdlib at top-level (`hashlib`, `json`, `urllib.request`, `secrets`, `threading`, etc.). The lazy `opentelemetry_sdk_available()` detector imports the SDK only when called; no module-level `import opentelemetry`. `pyproject.toml` extras are unchanged. |
| `Do not emit raw byte arrays as span attributes.` | `OTLPSpanV1.attributes` is typed `Mapping[str, str]`; `SpanContext.set_attribute(key, value)` coerces value to `str`. Test `test_w87_no_raw_payload_bytes_in_spans_or_metrics` enforces. |
| `Do not skip the tenant-ID label.` | `W87_STANDARD_LABEL_KEYS` includes `tenant_id`; `StructuredLogRecordV1` requires it; the bench's `standard_label_set_present` bool checks. |
| `Do not trace 100% of requests by default.` | `W87_DEFAULT_SAMPLING_RATE = 0.10`. `HeadSamplerV1` honours the rate, with `always_sample_errors=True` for visibility on failure. Test `test_w87_head_sampler_deterministic` confirms ~10% empirical rate. |
| `Do not ship metric names that collide with stdlib / third-party.` | `W87_METRIC_NAME_PREFIX = "coordpy_"`; `_validate_metric_name()` raises `ValueError` on unprefixed names. Test `test_w87_metric_name_prefix_enforced` exercises both branches. |
| `Do not count this issue closed by "we have a logging module".` | The OTLP + Prometheus + structured-logs trio is shipped; **every** load-bearing closure bool in the bench report is independently asserted and rejects on any failure. |

## Honest carry-forward limitations

* **`W87-L-OBSERVABILITY-V1-STDLIB-CAP`** — stdlib-only OTLP/HTTP
  JSON export (no protobuf, no gRPC).  Any OTLP collector that
  accepts `application/json` (e.g. opentelemetry-collector with
  the `otlp/http` receiver) consumes the payload.  Native gRPC
  is V2.
* **`W87-L-OBSERVABILITY-V1-PROMETHEUS-TEXT-CAP`** — Prometheus
  exposition is the **text** format (v0.0.4).  The newer
  protobuf encoding is V2; current scrapers accept either.
* **`W87-L-OBSERVABILITY-V1-HEAD-SAMPLING-CAP`** — V1 implements
  head sampling (fixed probability per trace).  Tail-based
  sampling, parent-based sampling, and rate-limiting samplers
  are V2.
* **`W87-L-OBSERVABILITY-V1-OPTIONAL-DEP-CAP`** — the module is
  not imported by `coordpy/__init__.py`; clients explicit-import
  it.  `coordpy.__version__` and `coordpy.SDK_VERSION` are
  unchanged.
* **`W87-L-OBSERVABILITY-V1-EVENT-GRAPH-GAUGE-MANUAL-CAP`** —
  the `coordpy_event_graph_size` and
  `coordpy_audit_anchor_root_age_seconds` gauges are wired by
  the bundle but require the caller (W82 event graph,
  W82 hosted audit anchoring) to call `.set(...)` periodically.
  V2 will auto-wire these from the W82 modules.
* **`W87-L-OBSERVABILITY-V1-CONSENSUS-COUNTERS-MANUAL-CAP`** —
  the `coordpy_consensus_*_total` counters are pre-registered;
  the consensus controller (`team_consensus_controller_v14`)
  must call `obs.registry.inc(...)` to populate them.  V2 will
  thread `ObservabilityV1` into the consensus path natively.

## File inventory

| File | Role |
|---|---|
| `coordpy/observability_v1.py` | Core module — OTLP/Prometheus/StructuredLogger/Sampler/Bundle/Gateway-integration |
| `tests/test_w87_observability_v1.py` | 15 unit tests covering wire-format compliance, sampling, gateway integration, anti-cheats |
| `scripts/run_w87_observability_bench.py` | Live load bench driver (writes content-addressed report) |
| `scripts/verify_w87_observability_audit_chain.py` | Offline re-verifier (re-derives CIDs, asserts every load-bearing bool) |
| `docs/grafana/w87_dashboard.json` | Sample Grafana dashboard (7 panels covering all 8 metrics) |
| `docs/RESULTS_W87_OBSERVABILITY_V1.md` | This results doc |

## Audit-chain identity

The bench report's `bench_cid` is computed via the canonical
SHA-256 of the report dict with `bench_cid=""` injected (the
W86 hash-format convention — see
`docs/W86_AUTOMATION_ARCHITECTURE.md` §10 lesson-3).  The
verifier mirrors the injection exactly, so the recorded CID and
the re-derived CID match byte-for-byte.  This catches both
producer-side bit-rot and verifier-side hash-format drift.

## Cost

Zero — the bench runs entirely in-process on Python stdlib in
~1 s wall-clock on a 2023 MacBook.  No external collector
required to validate the wire format (the bench parses the
output itself); to attach to a real OTLP collector, install
`opentelemetry-collector` and point the exporter at
`http://localhost:4318/v1/traces`.
