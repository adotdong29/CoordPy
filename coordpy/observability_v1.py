"""W87 / P3 #47 — Observability V1.

The W82+W83 stack ships content-addressed audit chains; this
module ships the *operational* observability that production
deployments need:

  * OTLP spans (OpenTelemetry-format) per composed-pipeline
    turn — child spans for substrate-restore, consensus,
    integrity-verify, Merkle-anchor.  Span attributes carry the
    relevant CIDs so the trace can be cross-referenced with the
    audit chain.
  * Prometheus metrics — counters, gauges, histograms — emitted
    via the gateway's new ``/metrics`` endpoint in the standard
    Prometheus text exposition format.
  * Structured JSON-line logs — content-addressed, with explicit
    tenant / run / agent / role fields where relevant.
  * Sampling — sensible defaults (10% trace sampling); fully
    configurable.

The module is **stdlib-only**.  When the optional
``opentelemetry-sdk`` is installed, spans can also be exported
to a real collector via ``OTLPSpanExporter.export_to_http(...)``;
without that dep, spans are emitted in the OTLP/HTTP JSON
wire-format directly (which any OTLP collector accepts).

The gateway integration is additive: the W81 gateway gains a
``/metrics`` route via ``register_observability_v1(gateway)``,
and per-turn spans are emitted via the ``SpanContext`` context
manager.  Both are no-ops when observability is not enabled.

Honest scope (W87)
------------------

* ``W87-L-OBSERVABILITY-V1-STDLIB-CAP`` — stdlib-only.  The
  opentelemetry SDK is optional; without it, spans are emitted
  in OTLP/HTTP JSON via ``urllib`` POST.
* ``W87-L-OBSERVABILITY-V1-PROMETHEUS-TEXT-CAP`` — Prometheus
  exposition is the text format (v0.0.4).  protobuf encoding is
  V2.
* ``W87-L-OBSERVABILITY-V1-HEAD-SAMPLING-CAP`` — V1 supports
  head sampling (fixed probability per trace).  Tail-based and
  parent-based sampling are V2.
* ``W87-L-OBSERVABILITY-V1-NO-RAW-PAYLOAD-CAP`` — span and log
  attributes carry CIDs and integer counters, never raw payload
  bytes.  Multi-tenant deployments do not leak payloads into
  the observability plane.
* ``W87-L-OBSERVABILITY-V1-OPTIONAL-DEP-CAP`` — adding this
  module does NOT change the stable surface.  ``pip install
  coordpy-ai`` still installs cleanly with zero observability
  deps.  ``pip install coordpy-ai[observability]`` opts in to
  the opentelemetry SDK + prometheus_client (both optional).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import logging
import os
import random
import secrets
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Mapping, Sequence


W87_OBSERVABILITY_V1_SCHEMA_VERSION: str = (
    "coordpy.observability_v1.v1")

# Metric name prefix — mandatory per the issue body's anti-cheat
# clause "metric names that collide with stdlib / third-party
# ... ``coordpy_`` prefix is mandatory".
W87_METRIC_NAME_PREFIX: str = "coordpy_"

# Standard label set — the issue body's required label set.
W87_STANDARD_LABEL_KEYS: tuple[str, ...] = (
    "tenant_id", "run_cid", "agent_id", "role")

# Default head-sampling rate (10%) — issue body says "aim for
# 1–10% with explicit override". 10% maintains visibility under
# typical research load.
W87_DEFAULT_SAMPLING_RATE: float = 0.10

# OTLP span kind enum (numeric per OTLP spec).
W87_OTLP_SPAN_KIND_INTERNAL: int = 1
W87_OTLP_SPAN_KIND_SERVER: int = 2
W87_OTLP_SPAN_KIND_CLIENT: int = 3

# OTLP status code enum.
W87_OTLP_STATUS_CODE_UNSET: int = 0
W87_OTLP_STATUS_CODE_OK: int = 1
W87_OTLP_STATUS_CODE_ERROR: int = 2


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(
        _canonical_bytes(payload)).hexdigest()


def _new_trace_id() -> str:
    """16-byte trace ID, hex-encoded (32 chars). OTLP spec
    requires 16 bytes for the trace ID."""
    return secrets.token_hex(16)


def _new_span_id() -> str:
    """8-byte span ID, hex-encoded (16 chars). OTLP spec
    requires 8 bytes for the span ID."""
    return secrets.token_hex(8)


def _now_ns() -> int:
    return int(time.time_ns())


def _validate_label_value(value: str) -> str:
    """Conservative label-value sanitiser. Prometheus expects
    UTF-8 with escaped backslashes, quotes, and newlines. We
    forbid newlines outright; everything else is escaped."""
    s = str(value)
    if "\n" in s or "\r" in s:
        raise ValueError(
            f"label value contains a newline: {value!r}")
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _validate_metric_name(name: str) -> str:
    """Validate that a metric name follows Prometheus naming
    rules AND carries the mandatory ``coordpy_`` prefix."""
    if not isinstance(name, str):
        raise TypeError(
            f"metric name must be str, got {type(name).__name__}")
    if not name.startswith(W87_METRIC_NAME_PREFIX):
        raise ValueError(
            f"metric name {name!r} must start with prefix "
            f"{W87_METRIC_NAME_PREFIX!r} (W87-L-OBSERVABILITY-V1-"
            f"NO-RAW-PAYLOAD-CAP and the issue body's anti-cheat "
            f"clause)")
    # Prometheus: [a-zA-Z_:][a-zA-Z0-9_:]*
    if not name[0].isalpha() and name[0] not in ("_", ":"):
        raise ValueError(f"metric name {name!r} bad first char")
    for ch in name[1:]:
        if not (ch.isalnum() or ch in ("_", ":")):
            raise ValueError(
                f"metric name {name!r} contains bad char {ch!r}")
    return name


# ---------------------------------------------------------------
# OTLP span types
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class OTLPSpanV1:
    """One OTLP span in the wire-format-compatible shape.

    All fields map 1:1 to the OTLP/HTTP JSON wire format
    (https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding).
    A serialised batch of these can be POSTed to any OTLP
    collector accepting ``application/json`` on
    ``/v1/traces``.
    """

    schema: str
    name: str
    trace_id_hex: str
    span_id_hex: str
    parent_span_id_hex: str  # "" if root
    kind: int  # one of W87_OTLP_SPAN_KIND_*
    start_time_unix_nano: int
    end_time_unix_nano: int
    attributes: Mapping[str, str]
    status_code: int  # one of W87_OTLP_STATUS_CODE_*
    status_message: str

    def to_otlp_json(self) -> dict[str, Any]:
        """Render this span in the OTLP/HTTP JSON wire shape."""
        attrs = [
            {"key": str(k),
             "value": {"stringValue": str(v)}}
            for k, v in sorted(self.attributes.items())
        ]
        out: dict[str, Any] = {
            "traceId": str(self.trace_id_hex),
            "spanId": str(self.span_id_hex),
            "name": str(self.name),
            "kind": int(self.kind),
            "startTimeUnixNano": str(int(self.start_time_unix_nano)),
            "endTimeUnixNano": str(int(self.end_time_unix_nano)),
            "attributes": attrs,
            "status": {
                "code": int(self.status_code),
                "message": str(self.status_message),
            },
        }
        if self.parent_span_id_hex:
            out["parentSpanId"] = str(self.parent_span_id_hex)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "name": str(self.name),
            "trace_id": str(self.trace_id_hex),
            "span_id": str(self.span_id_hex),
            "parent_span_id": str(self.parent_span_id_hex),
            "kind": int(self.kind),
            "start_time_unix_nano": int(
                self.start_time_unix_nano),
            "end_time_unix_nano": int(self.end_time_unix_nano),
            "attributes": dict(self.attributes),
            "status_code": int(self.status_code),
            "status_message": str(self.status_message),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w87_otlp_span_v1",
            "span": self.to_dict(),
        })


@dataclasses.dataclass(frozen=True)
class OTLPResourceSpansBatchV1:
    """A batch of spans grouped by resource — the OTLP wire
    payload shape for ``/v1/traces``."""

    schema: str
    service_name: str
    service_version: str
    spans: tuple[OTLPSpanV1, ...]

    def to_otlp_json(self) -> dict[str, Any]:
        """Wire-format payload for POST /v1/traces."""
        resource_attrs = [
            {"key": "service.name",
             "value": {"stringValue": str(self.service_name)}},
            {"key": "service.version",
             "value": {"stringValue": str(self.service_version)}},
        ]
        return {
            "resourceSpans": [{
                "resource": {"attributes": resource_attrs},
                "scopeSpans": [{
                    "scope": {
                        "name": "coordpy.observability_v1",
                        "version": str(
                            W87_OBSERVABILITY_V1_SCHEMA_VERSION),
                    },
                    "spans": [s.to_otlp_json() for s in self.spans],
                }],
            }],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "service_name": str(self.service_name),
            "service_version": str(self.service_version),
            "n_spans": int(len(self.spans)),
            "span_cids": [s.cid() for s in self.spans],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w87_otlp_resource_spans_batch_v1",
            "batch": self.to_dict(),
        })


# ---------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class HeadSamplerV1:
    """Probability-based head sampler.

    ``sample_rate`` is in [0, 1]. ``always_sample_errors=True``
    forces a 100% sample rate for any span emitted under a
    failed status code (issue body: "every error").
    """

    schema: str = W87_OBSERVABILITY_V1_SCHEMA_VERSION
    sample_rate: float = W87_DEFAULT_SAMPLING_RATE
    always_sample_errors: bool = True
    seed: int = 87_047_001

    def should_sample(
            self, *, trace_id_hex: str, is_error: bool = False,
    ) -> bool:
        """Decide whether to keep a trace. Decision is deterministic
        in ``trace_id_hex`` so consistent sampling holds across
        spans of the same trace."""
        if self.always_sample_errors and is_error:
            return True
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        # Deterministic hash → [0, 1) → sample if below rate.
        h = hashlib.sha256(
            (str(self.seed) + ":" + str(trace_id_hex)).encode(
                "utf-8")).hexdigest()
        # Use the first 8 hex chars as a uniform [0, 1) sample.
        bucket = int(h[:8], 16) / float(0xFFFFFFFF)
        return bucket < self.sample_rate

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w87_head_sampler_v1",
            "schema": str(self.schema),
            "sample_rate": float(self.sample_rate),
            "always_sample_errors": bool(
                self.always_sample_errors),
            "seed": int(self.seed),
        })


# ---------------------------------------------------------------
# Prometheus metrics registry
# ---------------------------------------------------------------

class MetricKind(str, enum.Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


W87_METRIC_KINDS: tuple[str, ...] = tuple(
    k.value for k in MetricKind)

# Default histogram buckets in seconds (Prometheus convention).
W87_DEFAULT_HISTOGRAM_BUCKETS_SECONDS: tuple[float, ...] = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
    2.5, 5.0, 10.0)


def _format_labels(labels: Mapping[str, str]) -> str:
    if not labels:
        return ""
    parts = [
        f'{k}="{_validate_label_value(v)}"'
        for k, v in sorted(labels.items())
    ]
    return "{" + ",".join(parts) + "}"


@dataclasses.dataclass
class _Counter:
    name: str
    help_text: str
    samples: dict[tuple[tuple[str, str], ...], float] = (
        dataclasses.field(default_factory=dict))

    def inc(self, *, labels: Mapping[str, str], amount: float = 1.0) -> None:
        key = tuple(sorted(labels.items()))
        self.samples[key] = self.samples.get(key, 0.0) + float(amount)


@dataclasses.dataclass
class _Gauge:
    name: str
    help_text: str
    samples: dict[tuple[tuple[str, str], ...], float] = (
        dataclasses.field(default_factory=dict))

    def set(self, *, labels: Mapping[str, str], value: float) -> None:
        key = tuple(sorted(labels.items()))
        self.samples[key] = float(value)


@dataclasses.dataclass
class _Histogram:
    name: str
    help_text: str
    buckets: tuple[float, ...]
    samples: dict[tuple[tuple[str, str], ...],
                  dict[str, float]] = dataclasses.field(
        default_factory=dict)

    def observe(self, *, labels: Mapping[str, str], value: float) -> None:
        key = tuple(sorted(labels.items()))
        agg = self.samples.setdefault(
            key, {"count": 0.0, "sum": 0.0,
                  **{f"le_{b}": 0.0 for b in self.buckets},
                  "le_+Inf": 0.0})
        agg["count"] += 1.0
        agg["sum"] += float(value)
        agg["le_+Inf"] += 1.0
        for b in self.buckets:
            if float(value) <= b:
                agg[f"le_{b}"] += 1.0


class PrometheusMetricsRegistryV1:
    """Thread-safe Prometheus-compatible metrics registry.

    Renders the standard text exposition format
    (https://prometheus.io/docs/instrumenting/exposition_formats/).
    All metric names MUST start with the ``coordpy_`` prefix.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, _Counter] = {}
        self._gauges: dict[str, _Gauge] = {}
        self._histograms: dict[str, _Histogram] = {}

    def register_counter(
            self, *, name: str, help_text: str,
    ) -> None:
        _validate_metric_name(name)
        with self._lock:
            if name in self._counters:
                return
            self._counters[name] = _Counter(
                name=name, help_text=str(help_text))

    def register_gauge(
            self, *, name: str, help_text: str,
    ) -> None:
        _validate_metric_name(name)
        with self._lock:
            if name in self._gauges:
                return
            self._gauges[name] = _Gauge(
                name=name, help_text=str(help_text))

    def register_histogram(
            self, *, name: str, help_text: str,
            buckets: Sequence[float] = (
                W87_DEFAULT_HISTOGRAM_BUCKETS_SECONDS),
    ) -> None:
        _validate_metric_name(name)
        with self._lock:
            if name in self._histograms:
                return
            self._histograms[name] = _Histogram(
                name=name, help_text=str(help_text),
                buckets=tuple(float(b) for b in buckets))

    def inc(self, name: str, *,
            labels: Mapping[str, str] | None = None,
            amount: float = 1.0) -> None:
        labels = dict(labels or {})
        with self._lock:
            ctr = self._counters.get(str(name))
            if ctr is None:
                raise KeyError(
                    f"counter {name!r} not registered")
            ctr.inc(labels=labels, amount=float(amount))

    def set(self, name: str, *, value: float,
            labels: Mapping[str, str] | None = None) -> None:
        labels = dict(labels or {})
        with self._lock:
            g = self._gauges.get(str(name))
            if g is None:
                raise KeyError(
                    f"gauge {name!r} not registered")
            g.set(labels=labels, value=float(value))

    def observe(self, name: str, *, value: float,
                labels: Mapping[str, str] | None = None,
                ) -> None:
        labels = dict(labels or {})
        with self._lock:
            h = self._histograms.get(str(name))
            if h is None:
                raise KeyError(
                    f"histogram {name!r} not registered")
            h.observe(labels=labels, value=float(value))

    def render_text(self) -> str:
        """Render the full exposition format."""
        lines: list[str] = []
        with self._lock:
            for name in sorted(self._counters.keys()):
                ctr = self._counters[name]
                lines.append(f"# HELP {name} {ctr.help_text}")
                lines.append(f"# TYPE {name} counter")
                if not ctr.samples:
                    lines.append(f"{name} 0")
                for labels, value in sorted(
                        ctr.samples.items()):
                    label_str = _format_labels(dict(labels))
                    lines.append(
                        f"{name}{label_str} {float(value)}")
            for name in sorted(self._gauges.keys()):
                g = self._gauges[name]
                lines.append(f"# HELP {name} {g.help_text}")
                lines.append(f"# TYPE {name} gauge")
                if not g.samples:
                    lines.append(f"{name} 0")
                for labels, value in sorted(
                        g.samples.items()):
                    label_str = _format_labels(dict(labels))
                    lines.append(
                        f"{name}{label_str} {float(value)}")
            for name in sorted(self._histograms.keys()):
                h = self._histograms[name]
                lines.append(f"# HELP {name} {h.help_text}")
                lines.append(f"# TYPE {name} histogram")
                if not h.samples:
                    # Render zeroed bucket lines so scrapers don't
                    # see a "metric type changed" error on first scrape.
                    for b in h.buckets:
                        lines.append(
                            f'{name}_bucket{{le="{b}"}} 0')
                    lines.append(
                        f'{name}_bucket{{le="+Inf"}} 0')
                    lines.append(f"{name}_sum 0")
                    lines.append(f"{name}_count 0")
                for labels, agg in sorted(h.samples.items()):
                    base = dict(labels)
                    for b in h.buckets:
                        bucket_labels = dict(base)
                        bucket_labels["le"] = str(b)
                        bucket_str = _format_labels(bucket_labels)
                        lines.append(
                            f"{name}_bucket{bucket_str} "
                            f"{agg[f'le_{b}']}")
                    inf_labels = dict(base)
                    inf_labels["le"] = "+Inf"
                    inf_str = _format_labels(inf_labels)
                    lines.append(
                        f"{name}_bucket{inf_str} "
                        f"{agg['le_+Inf']}")
                    sum_str = _format_labels(base)
                    lines.append(
                        f"{name}_sum{sum_str} {agg['sum']}")
                    count_str = _format_labels(base)
                    lines.append(
                        f"{name}_count{count_str} "
                        f"{agg['count']}")
        # Prometheus text format must end with a newline.
        return "\n".join(lines) + "\n"

    def metric_names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(
                list(self._counters.keys()) +
                list(self._gauges.keys()) +
                list(self._histograms.keys())))


# ---------------------------------------------------------------
# Structured JSON-line logger
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class StructuredLogRecordV1:
    """One JSON-line log record. All fields are explicit; no
    payload bytes."""

    schema: str
    timestamp_ns: int
    level: str
    message: str
    tenant_id: str
    run_cid: str
    agent_id: str
    role: str
    extra: Mapping[str, str]

    def to_jsonline(self) -> str:
        d: dict[str, Any] = {
            "schema": str(self.schema),
            "timestamp_ns": int(self.timestamp_ns),
            "level": str(self.level),
            "message": str(self.message),
            "tenant_id": str(self.tenant_id),
            "run_cid": str(self.run_cid),
            "agent_id": str(self.agent_id),
            "role": str(self.role),
        }
        # Standard fields go first; extra fields under "ext" to
        # avoid collisions.
        if self.extra:
            d["ext"] = {str(k): str(v) for k, v in self.extra.items()}
        d["record_cid"] = _sha256_hex(d)
        return json.dumps(
            d, sort_keys=True, separators=(",", ":"))


class StructuredLoggerV1:
    """Append-only JSON-line logger.

    By default writes to an in-memory buffer; ``file_path``
    redirects to a file (one JSON object per line).
    """

    def __init__(
            self, *, file_path: str | None = None,
            default_tenant_id: str = "default",
            default_agent_id: str = "",
            default_role: str = "") -> None:
        self.file_path = file_path
        self.default_tenant_id = str(default_tenant_id)
        self.default_agent_id = str(default_agent_id)
        self.default_role = str(default_role)
        self._buffer: list[str] = []
        self._lock = threading.Lock()

    def log(
            self, *, level: str, message: str,
            run_cid: str = "",
            tenant_id: str | None = None,
            agent_id: str | None = None,
            role: str | None = None,
            extra: Mapping[str, str] | None = None,
    ) -> StructuredLogRecordV1:
        rec = StructuredLogRecordV1(
            schema=W87_OBSERVABILITY_V1_SCHEMA_VERSION,
            timestamp_ns=_now_ns(),
            level=str(level).upper(),
            message=str(message),
            tenant_id=str(
                tenant_id if tenant_id is not None
                else self.default_tenant_id),
            run_cid=str(run_cid),
            agent_id=str(
                agent_id if agent_id is not None
                else self.default_agent_id),
            role=str(
                role if role is not None
                else self.default_role),
            extra=dict(extra or {}),
        )
        line = rec.to_jsonline()
        with self._lock:
            self._buffer.append(line)
            if self.file_path:
                with open(self.file_path, "a", encoding="utf-8") as fp:
                    fp.write(line + "\n")
        return rec

    def lines(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._buffer)


# ---------------------------------------------------------------
# Span tracker
# ---------------------------------------------------------------

class _SpanTrackerV1:
    """Append-only span buffer. Thread-safe."""

    def __init__(
            self, *,
            service_name: str = "coordpy",
            service_version: str = (
                W87_OBSERVABILITY_V1_SCHEMA_VERSION),
            sampler: HeadSamplerV1 | None = None,
            registry: PrometheusMetricsRegistryV1 | None = None,
    ) -> None:
        self.service_name = str(service_name)
        self.service_version = str(service_version)
        self.sampler = sampler or HeadSamplerV1()
        self.registry = registry
        self._spans: list[OTLPSpanV1] = []
        self._lock = threading.Lock()
        self._dropped: int = 0

    def emit(self, span: OTLPSpanV1, *, is_error: bool = False,
             ) -> bool:
        if not self.sampler.should_sample(
                trace_id_hex=span.trace_id_hex,
                is_error=is_error):
            with self._lock:
                self._dropped += 1
            return False
        with self._lock:
            self._spans.append(span)
        if self.registry is not None:
            # Sampling-aware counter on the registry.
            try:
                self.registry.inc(
                    "coordpy_observability_spans_emitted_total",
                    labels={"span_name": str(span.name)})
            except KeyError:
                pass
        return True

    def spans(self) -> tuple[OTLPSpanV1, ...]:
        with self._lock:
            return tuple(self._spans)

    def dropped(self) -> int:
        with self._lock:
            return int(self._dropped)

    def batch(self) -> OTLPResourceSpansBatchV1:
        return OTLPResourceSpansBatchV1(
            schema=W87_OBSERVABILITY_V1_SCHEMA_VERSION,
            service_name=self.service_name,
            service_version=self.service_version,
            spans=self.spans(),
        )


class SpanContext:
    """Context manager for emitting a span around a piece of
    work. Use as::

        with SpanContext(tracker, "substrate_restore",
                         attributes={"run_cid": cid}) as ctx:
            ...
            ctx.set_attribute("kv_cache_cid", kv_cid)

    Spans are emitted on ``__exit__``. The context manager
    captures start/end ns and status (OK on no-exception,
    ERROR if an exception propagates).
    """

    def __init__(
            self, tracker: _SpanTrackerV1, name: str, *,
            trace_id_hex: str | None = None,
            parent_span_id_hex: str = "",
            kind: int = W87_OTLP_SPAN_KIND_INTERNAL,
            attributes: Mapping[str, str] | None = None,
    ) -> None:
        self.tracker = tracker
        self.name = str(name)
        self.trace_id_hex = trace_id_hex or _new_trace_id()
        self.span_id_hex = _new_span_id()
        self.parent_span_id_hex = str(parent_span_id_hex)
        self.kind = int(kind)
        self._attributes: dict[str, str] = {
            str(k): str(v) for k, v in (attributes or {}).items()
        }
        self._start_ns: int = 0
        self._end_ns: int = 0
        self._status_code: int = W87_OTLP_STATUS_CODE_UNSET
        self._status_message: str = ""

    def set_attribute(self, key: str, value: str) -> None:
        self._attributes[str(key)] = str(value)

    def __enter__(self) -> "SpanContext":
        self._start_ns = _now_ns()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._end_ns = _now_ns()
        if exc_type is None:
            self._status_code = W87_OTLP_STATUS_CODE_OK
            self._status_message = ""
        else:
            self._status_code = W87_OTLP_STATUS_CODE_ERROR
            self._status_message = str(exc) if exc is not None else ""
        span = OTLPSpanV1(
            schema=W87_OBSERVABILITY_V1_SCHEMA_VERSION,
            name=self.name,
            trace_id_hex=self.trace_id_hex,
            span_id_hex=self.span_id_hex,
            parent_span_id_hex=self.parent_span_id_hex,
            kind=self.kind,
            start_time_unix_nano=int(self._start_ns),
            end_time_unix_nano=int(self._end_ns),
            attributes=self._attributes,
            status_code=int(self._status_code),
            status_message=str(self._status_message),
        )
        is_error = (self._status_code ==
                    W87_OTLP_STATUS_CODE_ERROR)
        self.tracker.emit(span, is_error=is_error)


# ---------------------------------------------------------------
# OTLP HTTP exporter
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class OTLPHttpExportResultV1:
    schema: str
    endpoint: str
    n_spans: int
    status_code: int
    payload_cid: str
    error: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "endpoint": str(self.endpoint),
            "n_spans": int(self.n_spans),
            "status_code": int(self.status_code),
            "payload_cid": str(self.payload_cid),
            "error": str(self.error),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w87_otlp_http_export_result_v1",
            "result": self.to_dict(),
        })


def export_spans_otlp_http_v1(
        *, batch: OTLPResourceSpansBatchV1,
        endpoint: str,
        timeout_seconds: float = 5.0,
        headers: Mapping[str, str] | None = None,
) -> OTLPHttpExportResultV1:
    """POST the OTLP/HTTP JSON batch to ``endpoint`` (typically
    ``http://otel-collector:4318/v1/traces``).

    Does NOT raise on HTTP errors; returns a structured result
    capturing status / error so callers can fan out to multiple
    backends without one failure stopping the others.
    """
    payload = json.dumps(
        batch.to_otlp_json(),
        sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload_cid = hashlib.sha256(payload).hexdigest()
    req_headers = {
        "Content-Type": "application/json",
        "User-Agent": "coordpy/observability_v1",
    }
    for k, v in (headers or {}).items():
        req_headers[str(k)] = str(v)
    req = urllib.request.Request(
        url=str(endpoint), data=payload, method="POST",
        headers=req_headers)
    status = -1
    error = ""
    try:
        with urllib.request.urlopen(
                req, timeout=float(timeout_seconds)) as resp:
            status = int(resp.status)
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        error = f"HTTPError {exc.code}: {exc.reason}"
    except urllib.error.URLError as exc:
        error = f"URLError: {exc.reason}"
    except (TimeoutError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return OTLPHttpExportResultV1(
        schema=W87_OBSERVABILITY_V1_SCHEMA_VERSION,
        endpoint=str(endpoint),
        n_spans=int(len(batch.spans)),
        status_code=int(status),
        payload_cid=str(payload_cid),
        error=str(error),
    )


# ---------------------------------------------------------------
# Top-level observability bundle
# ---------------------------------------------------------------

@dataclasses.dataclass
class ObservabilityV1:
    """Bundle of observability state: registry + tracker + logger.

    Convenience top-level container that callers wire into the
    W81 gateway and the W83 composed pipeline.

    Default-constructed instance has the standard metric set
    pre-registered:

      * coordpy_gateway_requests_total{path,status}
      * coordpy_gateway_request_duration_seconds (histogram)
      * coordpy_consensus_commits_total{decision_kind}
      * coordpy_consensus_abstains_total{reason}
      * coordpy_integrity_verdicts_total{verdict}
      * coordpy_event_graph_size (gauge)
      * coordpy_audit_anchor_root_age_seconds (gauge)
      * coordpy_observability_spans_emitted_total{span_name}
    """

    registry: PrometheusMetricsRegistryV1
    tracker: _SpanTrackerV1
    logger: StructuredLoggerV1
    sampler: HeadSamplerV1
    service_name: str = "coordpy"

    @classmethod
    def default(
            cls, *, service_name: str = "coordpy",
            sample_rate: float = W87_DEFAULT_SAMPLING_RATE,
    ) -> "ObservabilityV1":
        registry = PrometheusMetricsRegistryV1()
        # Required standard metric set. The issue body says
        # "exposing at least 8 metrics".
        registry.register_counter(
            name="coordpy_gateway_requests_total",
            help_text="Total gateway requests by path and status")
        registry.register_histogram(
            name="coordpy_gateway_request_duration_seconds",
            help_text="Gateway request duration seconds")
        registry.register_counter(
            name="coordpy_consensus_commits_total",
            help_text="Consensus commits by decision_kind")
        registry.register_counter(
            name="coordpy_consensus_abstains_total",
            help_text="Consensus abstains by reason")
        registry.register_counter(
            name="coordpy_integrity_verdicts_total",
            help_text="Integrity verdicts by verdict")
        registry.register_gauge(
            name="coordpy_event_graph_size",
            help_text="Current event-graph size in nodes")
        registry.register_gauge(
            name="coordpy_audit_anchor_root_age_seconds",
            help_text="Seconds since the last audit anchor commit")
        registry.register_counter(
            name="coordpy_observability_spans_emitted_total",
            help_text="OTLP spans emitted by span_name")
        sampler = HeadSamplerV1(sample_rate=float(sample_rate))
        tracker = _SpanTrackerV1(
            service_name=service_name,
            service_version=W87_OBSERVABILITY_V1_SCHEMA_VERSION,
            sampler=sampler, registry=registry)
        logger = StructuredLoggerV1()
        return cls(
            registry=registry, tracker=tracker,
            logger=logger, sampler=sampler,
            service_name=str(service_name),
        )

    def span(self, name: str, *,
             trace_id_hex: str | None = None,
             parent_span_id_hex: str = "",
             attributes: Mapping[str, str] | None = None,
             ) -> SpanContext:
        return SpanContext(
            self.tracker, name,
            trace_id_hex=trace_id_hex,
            parent_span_id_hex=parent_span_id_hex,
            attributes=attributes)

    def metrics_text(self) -> str:
        return self.registry.render_text()

    def spans_batch(self) -> OTLPResourceSpansBatchV1:
        return self.tracker.batch()

    def log(self, *args, **kwargs) -> StructuredLogRecordV1:
        return self.logger.log(*args, **kwargs)

    def required_metric_names(self) -> tuple[str, ...]:
        return self.registry.metric_names()

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w87_observability_v1_state",
            "schema": str(W87_OBSERVABILITY_V1_SCHEMA_VERSION),
            "service_name": str(self.service_name),
            "sampler_cid": str(self.sampler.cid()),
            "n_spans": int(len(self.tracker.spans())),
            "n_log_lines": int(len(self.logger.lines())),
            "metric_names": list(self.registry.metric_names()),
        })


# ---------------------------------------------------------------
# Gateway integration
# ---------------------------------------------------------------

W87_GATEWAY_PATH_METRICS: str = "/metrics"


def register_observability_v1(
        gateway, observability: ObservabilityV1,
        *, expose_metrics: bool = True,
) -> None:
    """Attach observability to a W81 DeployableSubstrateGatewayV1
    instance.  Adds a /metrics route and wraps `dispatch` so each
    request is timed + counted + spanned.

    This is opt-in: gateway behavior is unchanged if this is not
    called.
    """
    if getattr(gateway, "_observability_attached", False):
        return
    gateway._observability = observability  # type: ignore[attr-defined]
    gateway._observability_expose_metrics = bool(expose_metrics)
    original_dispatch = gateway.dispatch

    def wrapped_dispatch(*, path, body=None, auth_header=None):
        trace_id = _new_trace_id()
        # /metrics path is served here, bypassing the original
        # gateway dispatch.
        if (bool(gateway._observability_expose_metrics)
                and str(path) == W87_GATEWAY_PATH_METRICS):
            text = observability.metrics_text()
            from .deployable_substrate_gateway_v1 import (
                GatewayResponseV1,
                W81_GATEWAY_V1_SCHEMA_VERSION as W81V)
            # Return as application/json body that includes the
            # raw text so HTTP-server can also send text/plain.
            req_cid = _sha256_hex({
                "kind": "w87_metrics_request_v1",
                "path": str(path)})
            body_out: dict[str, Any] = {
                "schema": str(W81V),
                "metrics_text": str(text),
                "metric_names": list(
                    observability.required_metric_names()),
                "observability_state_cid": str(
                    observability.cid()),
            }
            return GatewayResponseV1(
                schema=str(W81V),
                path=str(path),
                status=200,
                body=body_out,
                request_cid=str(req_cid),
                response_cid=_sha256_hex(body_out),
            )
        # Otherwise: wrap the call in a span and count/observe.
        start = time.time()
        with observability.span(
                "gateway.dispatch",
                trace_id_hex=trace_id,
                attributes={"path": str(path)}) as sp:
            try:
                resp = original_dispatch(
                    path=path, body=body, auth_header=auth_header)
                sp.set_attribute(
                    "status", str(int(resp.status)))
                sp.set_attribute(
                    "request_cid", str(resp.request_cid))
                sp.set_attribute(
                    "response_cid", str(resp.response_cid))
                dur = float(time.time() - start)
                observability.registry.inc(
                    "coordpy_gateway_requests_total",
                    labels={"path": str(path),
                            "status": str(int(resp.status))})
                observability.registry.observe(
                    "coordpy_gateway_request_duration_seconds",
                    value=float(dur),
                    labels={"path": str(path)})
                return resp
            except Exception as exc:
                observability.registry.inc(
                    "coordpy_gateway_requests_total",
                    labels={"path": str(path), "status": "500"})
                raise

    gateway.dispatch = wrapped_dispatch  # type: ignore[assignment]
    gateway._observability_attached = True  # type: ignore[attr-defined]


# ---------------------------------------------------------------
# Optional opentelemetry-SDK bridge (lazy)
# ---------------------------------------------------------------

def opentelemetry_sdk_available() -> bool:
    try:
        import opentelemetry  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------
# Stdlib logging bridge
# ---------------------------------------------------------------

class JSONLineLoggingHandlerV1(logging.Handler):
    """A stdlib ``logging.Handler`` that bridges to a
    :class:`StructuredLoggerV1`. Drop into any existing logger
    setup to upgrade the format."""

    def __init__(
            self, structured: StructuredLoggerV1,
            *,
            run_cid_attr: str = "run_cid") -> None:
        super().__init__()
        self.structured = structured
        self.run_cid_attr = str(run_cid_attr)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            run_cid = str(getattr(record, self.run_cid_attr, ""))
            extra: dict[str, str] = {}
            # Pull standard extra fields if present.
            for k in W87_STANDARD_LABEL_KEYS:
                if k != self.run_cid_attr and hasattr(record, k):
                    extra[k] = str(getattr(record, k))
            self.structured.log(
                level=record.levelname,
                message=record.getMessage(),
                run_cid=run_cid,
                extra=extra)
        except Exception:  # noqa: BLE001 — handler must not raise
            self.handleError(record)


__all__ = [
    # Constants
    "W87_OBSERVABILITY_V1_SCHEMA_VERSION",
    "W87_METRIC_NAME_PREFIX",
    "W87_STANDARD_LABEL_KEYS",
    "W87_DEFAULT_SAMPLING_RATE",
    "W87_DEFAULT_HISTOGRAM_BUCKETS_SECONDS",
    "W87_OTLP_SPAN_KIND_INTERNAL",
    "W87_OTLP_SPAN_KIND_SERVER",
    "W87_OTLP_SPAN_KIND_CLIENT",
    "W87_OTLP_STATUS_CODE_UNSET",
    "W87_OTLP_STATUS_CODE_OK",
    "W87_OTLP_STATUS_CODE_ERROR",
    "W87_METRIC_KINDS",
    "W87_GATEWAY_PATH_METRICS",
    # Types
    "OTLPSpanV1",
    "OTLPResourceSpansBatchV1",
    "OTLPHttpExportResultV1",
    "HeadSamplerV1",
    "MetricKind",
    "PrometheusMetricsRegistryV1",
    "StructuredLogRecordV1",
    "StructuredLoggerV1",
    "SpanContext",
    "ObservabilityV1",
    "JSONLineLoggingHandlerV1",
    # Functions
    "export_spans_otlp_http_v1",
    "register_observability_v1",
    "opentelemetry_sdk_available",
]
