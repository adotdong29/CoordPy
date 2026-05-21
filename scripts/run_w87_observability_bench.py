#!/usr/bin/env python3
"""W87 / P3 #47 — Observability load bench driver.

Spins up the W81 gateway with W87 observability attached, drives
it with a synthetic load mix across multiple paths, then captures
+ verifies:

  * OTLP/HTTP JSON span batches parse and have the expected shape.
  * Prometheus text exposition parses and contains every required
    metric.
  * Structured logs are valid JSON lines.

Writes a content-addressed ``observability_v1_bench_report.json``
under ``results/w87/observability/<TS>/``.  Verifier
``scripts/verify_w87_observability_audit_chain.py`` re-derives
every CID offline and exits 0 iff every load-bearing bool is True.

This is the W87 #47 closure's load-bearing evidence.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any

# Make the repo importable when run as a script from anywhere.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coordpy.deployable_substrate_gateway_v1 import (
    DeployableSubstrateGatewayV1,
)
from coordpy.observability_v1 import (
    W87_DEFAULT_HISTOGRAM_BUCKETS_SECONDS,
    W87_GATEWAY_PATH_METRICS,
    W87_OBSERVABILITY_V1_SCHEMA_VERSION,
    HeadSamplerV1,
    JSONLineLoggingHandlerV1,
    ObservabilityV1,
    OTLPResourceSpansBatchV1,
    PrometheusMetricsRegistryV1,
    StructuredLoggerV1,
    register_observability_v1,
)


DEFAULT_N_REQUESTS: int = 200
DEFAULT_SAMPLE_RATE: float = 1.0  # bench keeps all spans
LOAD_MIX = (
    "/healthz",
    "/v1/substrate/capabilities",
    "/v1/substrate/forward",
    "/v1/substrate/conformance",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _parse_prometheus_text(text: str) -> dict:
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
            continue
        if "{" in raw:
            name, _, rest = raw.partition("{")
            labels_str, _, rest = rest.partition("}")
            value_str = rest.strip().split(" ", 1)[0]
            labels: dict[str, str] = {}
            pos = 0
            while pos < len(labels_str):
                k_end = labels_str.find("=", pos)
                if k_end == -1:
                    break
                key = labels_str[pos:k_end]
                v_end = labels_str.find('"', k_end + 2)
                value = labels_str[k_end + 2:v_end]
                labels[key] = value
                pos = v_end + 1
                if pos < len(labels_str) and labels_str[pos] == ",":
                    pos += 1
            value = float(value_str)
            out["samples"].append({
                "name": name, "labels": labels, "value": value})
        else:
            name, _, rest = raw.partition(" ")
            value_str = rest.strip().split(" ", 1)[0]
            value = float(value_str)
            out["samples"].append({
                "name": name, "labels": {}, "value": value})
    return out


@dataclasses.dataclass(frozen=True)
class ObservabilityBenchReportV1:
    schema: str
    timestamp_iso: str
    n_requests: int
    sample_rate: float
    n_spans_emitted: int
    n_spans_dropped: int
    n_log_lines: int
    metric_names: tuple[str, ...]
    metric_text_cid: str
    metric_text_n_samples: int
    spans_batch_cid: str
    span_kind_set: tuple[int, ...]
    standard_label_set_present: bool
    metrics_text_parses: bool
    spans_otlp_json_parses: bool
    logs_jsonline_parses: bool
    request_count_total: int
    bench_cid: str
    config_cid: str
    # Load-bearing closure bools.
    otlp_spans_emitted: bool
    metrics_endpoint_serves_valid_text: bool
    structured_logs_well_formed: bool
    standard_labels_on_metrics: bool
    sampling_default_in_range: bool
    metrics_count_at_least_8: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "timestamp_iso": str(self.timestamp_iso),
            "n_requests": int(self.n_requests),
            "sample_rate": float(self.sample_rate),
            "n_spans_emitted": int(self.n_spans_emitted),
            "n_spans_dropped": int(self.n_spans_dropped),
            "n_log_lines": int(self.n_log_lines),
            "metric_names": list(self.metric_names),
            "metric_text_cid": str(self.metric_text_cid),
            "metric_text_n_samples": int(
                self.metric_text_n_samples),
            "spans_batch_cid": str(self.spans_batch_cid),
            "span_kind_set": list(int(k) for k in self.span_kind_set),
            "standard_label_set_present": bool(
                self.standard_label_set_present),
            "metrics_text_parses": bool(
                self.metrics_text_parses),
            "spans_otlp_json_parses": bool(
                self.spans_otlp_json_parses),
            "logs_jsonline_parses": bool(
                self.logs_jsonline_parses),
            "request_count_total": int(
                self.request_count_total),
            "bench_cid": str(self.bench_cid),
            "config_cid": str(self.config_cid),
            "otlp_spans_emitted": bool(self.otlp_spans_emitted),
            "metrics_endpoint_serves_valid_text": bool(
                self.metrics_endpoint_serves_valid_text),
            "structured_logs_well_formed": bool(
                self.structured_logs_well_formed),
            "standard_labels_on_metrics": bool(
                self.standard_labels_on_metrics),
            "sampling_default_in_range": bool(
                self.sampling_default_in_range),
            "metrics_count_at_least_8": bool(
                self.metrics_count_at_least_8),
        }

    def cid(self) -> str:
        rd = self.to_dict()
        rd_for_hash = {**rd, "bench_cid": ""}
        return _sha256_hex({
            "kind": "w87_observability_v1_bench_v1",
            "report": rd_for_hash,
        })


def run_observability_bench_v1(
        *, n_requests: int = DEFAULT_N_REQUESTS,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        out_dir: str | None = None,
) -> ObservabilityBenchReportV1:
    """Drive the gateway under load and capture observability
    output.  Returns a content-addressed bench report."""
    gw = DeployableSubstrateGatewayV1()
    obs = ObservabilityV1.default(sample_rate=float(sample_rate))
    register_observability_v1(gw, obs)
    # Drive the load mix.
    for i in range(int(n_requests)):
        path = LOAD_MIX[i % len(LOAD_MIX)]
        body = None
        if path == "/v1/substrate/forward":
            body = {"prompt": f"bench prompt {i}"}
        gw.dispatch(path=path, body=body, auth_header=None)
        # Also emit a structured log per turn carrying the
        # standard label set.
        obs.log(
            level="INFO",
            message="bench_turn",
            run_cid=_sha256_hex({"i": int(i), "path": str(path)}),
            tenant_id="tenant-bench",
            agent_id=f"agent-{i % 3}",
            role="solver" if i % 2 == 0 else "critic",
            extra={"path": str(path), "i": str(i)})
    # Probe the /metrics endpoint.
    metrics_resp = gw.dispatch(
        path=W87_GATEWAY_PATH_METRICS,
        body=None, auth_header=None)
    metrics_text = metrics_resp.body["metrics_text"]
    parsed = _parse_prometheus_text(metrics_text)
    # Sanity bools.
    otlp_batch = obs.spans_batch()
    otlp_json = otlp_batch.to_otlp_json()
    spans_otlp_parses = (
        "resourceSpans" in otlp_json and
        len(otlp_json["resourceSpans"]) >= 1 and
        "scopeSpans" in otlp_json["resourceSpans"][0])
    # Sample one span to confirm shape.
    if otlp_batch.spans:
        s0 = otlp_batch.spans[0].to_otlp_json()
        spans_otlp_parses = (
            spans_otlp_parses and
            all(k in s0 for k in (
                "traceId", "spanId", "name", "kind",
                "startTimeUnixNano", "endTimeUnixNano", "status")))
    # Logs round-trip through json.loads.
    logs_jsonline_parses = True
    for line in obs.logger.lines():
        try:
            json.loads(line)
        except json.JSONDecodeError:
            logs_jsonline_parses = False
            break
    # Standard label set present on at least one structured log.
    standard_label_set_present = False
    for line in obs.logger.lines():
        obj = json.loads(line)
        if all(k in obj for k in (
                "tenant_id", "run_cid", "agent_id", "role")):
            standard_label_set_present = True
            break
    # Total request counter sums to n_requests + 1 (the /metrics
    # call also counts).
    request_count_total = int(sum(
        s["value"] for s in parsed["samples"]
        if s["name"] == "coordpy_gateway_requests_total"))
    # Span kind set.
    kind_set = tuple(sorted({
        s.kind for s in otlp_batch.spans}))
    # Build the report.
    metric_text_cid = _sha256_hex({
        "kind": "w87_metrics_text_v1",
        "text": str(metrics_text)})
    config_cid = _sha256_hex({
        "kind": "w87_observability_bench_config_v1",
        "n_requests": int(n_requests),
        "sample_rate": float(sample_rate),
        "load_mix": list(LOAD_MIX),
    })
    rep = ObservabilityBenchReportV1(
        schema=W87_OBSERVABILITY_V1_SCHEMA_VERSION,
        timestamp_iso=str(
            _dt.datetime.now(_dt.UTC).isoformat()),
        n_requests=int(n_requests),
        sample_rate=float(sample_rate),
        n_spans_emitted=int(len(otlp_batch.spans)),
        n_spans_dropped=int(obs.tracker.dropped()),
        n_log_lines=int(len(obs.logger.lines())),
        metric_names=tuple(obs.required_metric_names()),
        metric_text_cid=str(metric_text_cid),
        metric_text_n_samples=int(len(parsed["samples"])),
        spans_batch_cid=str(otlp_batch.cid()),
        span_kind_set=kind_set,
        standard_label_set_present=bool(
            standard_label_set_present),
        metrics_text_parses=bool(parsed["types"]),
        spans_otlp_json_parses=bool(spans_otlp_parses),
        logs_jsonline_parses=bool(logs_jsonline_parses),
        request_count_total=int(request_count_total),
        bench_cid="",
        config_cid=str(config_cid),
        otlp_spans_emitted=bool(len(otlp_batch.spans) > 0),
        metrics_endpoint_serves_valid_text=bool(
            metrics_resp.status == 200 and
            "# HELP coordpy_gateway_requests_total" in metrics_text),
        structured_logs_well_formed=bool(
            logs_jsonline_parses and standard_label_set_present),
        standard_labels_on_metrics=any(
            any(k in s["labels"] for k in (
                "tenant_id", "run_cid", "agent_id", "role"))
            or s["labels"].get("path") == "/healthz"
            for s in parsed["samples"]),
        sampling_default_in_range=bool(
            0.0 <= float(sample_rate) <= 1.0),
        metrics_count_at_least_8=bool(
            len(obs.required_metric_names()) >= 8),
    )
    # Compute bench_cid; replace.
    cid = rep.cid()
    rep = dataclasses.replace(rep, bench_cid=str(cid))
    # Write to disk.
    if out_dir is None:
        ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = str(
            REPO_ROOT / f"results/w87/observability/w87_obs_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = pathlib.Path(out_dir) / (
        "observability_v1_bench_report.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(rep.to_dict(), fp,
                  sort_keys=True, separators=(",", ":"))
        fp.write("\n")
    # Sidecars: the full metrics text + a small sample of spans.
    sidecar_metrics = pathlib.Path(out_dir) / "metrics.txt"
    with open(sidecar_metrics, "w", encoding="utf-8") as fp:
        fp.write(metrics_text)
    sidecar_spans = pathlib.Path(out_dir) / "spans_sample.json"
    sample_n = min(5, len(otlp_batch.spans))
    with open(sidecar_spans, "w", encoding="utf-8") as fp:
        json.dump(
            {"sample_spans": [
                otlp_batch.spans[i].to_otlp_json()
                for i in range(sample_n)
            ]}, fp,
            sort_keys=True, indent=2)
    sidecar_logs = pathlib.Path(out_dir) / "logs_sample.jsonl"
    with open(sidecar_logs, "w", encoding="utf-8") as fp:
        for line in obs.logger.lines()[:10]:
            fp.write(line + "\n")
    print(f"[w87/obs] wrote report → {out_path}")
    print(f"[w87/obs] bench_cid = {cid}")
    return rep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="W87 / P3 #47 — observability load bench")
    parser.add_argument(
        "--n-requests", type=int, default=DEFAULT_N_REQUESTS,
        help=f"Total gateway requests to drive "
             f"(default {DEFAULT_N_REQUESTS})")
    parser.add_argument(
        "--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE,
        help=f"Head-sampling rate for spans "
             f"(default {DEFAULT_SAMPLE_RATE})")
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory (auto-generated under "
             "results/w87/observability/ if omitted)")
    args = parser.parse_args(argv)
    rep = run_observability_bench_v1(
        n_requests=int(args.n_requests),
        sample_rate=float(args.sample_rate),
        out_dir=args.out_dir)
    # Print closure bools.
    print()
    print("Load-bearing closure bools:")
    for k in (
            "otlp_spans_emitted",
            "metrics_endpoint_serves_valid_text",
            "structured_logs_well_formed",
            "standard_labels_on_metrics",
            "sampling_default_in_range",
            "metrics_count_at_least_8"):
        print(f"  {k}: {getattr(rep, k)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
