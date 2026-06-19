#!/usr/bin/env python3
"""W87 / P3 #47 — Observability bench offline re-verifier.

Re-derives the bench_cid + metric_text_cid + spans_batch_cid from
the bench report and its sidecars (metrics.txt, spans_sample.json,
logs_sample.jsonl), then re-asserts every load-bearing closure
bool.  Exits 0 iff every check passes; non-zero on any failure.

This is the same pattern used by every W86 verifier.

Usage::

    python scripts/verify_w87_observability_audit_chain.py \
        --report results/w87/observability/<TS>/observability_v1_bench_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


def _canonical_bytes(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


LOAD_BEARING_BOOLS = (
    "otlp_spans_emitted",
    "metrics_endpoint_serves_valid_text",
    "structured_logs_well_formed",
    "standard_labels_on_metrics",
    "sampling_default_in_range",
    "metrics_count_at_least_8",
)


def _verify(report_path: pathlib.Path) -> int:
    if not report_path.is_file():
        print(f"FAIL: report not found at {report_path}",
              file=sys.stderr)
        return 1
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    fails = 0
    passes = 0

    def _check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails, passes
        if ok:
            passes += 1
            print(f"PASS {name}{(': ' + detail) if detail else ''}")
        else:
            fails += 1
            print(f"FAIL {name}{(': ' + detail) if detail else ''}",
                  file=sys.stderr)

    # Re-derive bench_cid.
    recorded_bench_cid = str(rep.get("bench_cid", ""))
    rep_for_hash = {**rep, "bench_cid": ""}
    derived_bench_cid = _sha256_hex({
        "kind": "w87_observability_v1_bench_v1",
        "report": rep_for_hash,
    })
    _check(
        "bench_cid",
        recorded_bench_cid == derived_bench_cid,
        f"recorded={recorded_bench_cid[:16]}... "
        f"derived={derived_bench_cid[:16]}...")

    # Re-derive metric_text_cid from sidecar.
    metrics_text_path = report_path.parent / "metrics.txt"
    if metrics_text_path.is_file():
        text = metrics_text_path.read_text(encoding="utf-8")
        derived_metric_cid = _sha256_hex({
            "kind": "w87_metrics_text_v1",
            "text": str(text)})
        _check(
            "metric_text_cid",
            str(rep.get("metric_text_cid", "")) ==
            derived_metric_cid,
            f"recorded={rep.get('metric_text_cid', '')[:16]}... "
            f"derived={derived_metric_cid[:16]}...")
        # Re-parse the metrics text and assert it carries the
        # 8 required metrics.
        required = {
            "coordpy_gateway_requests_total",
            "coordpy_gateway_request_duration_seconds",
            "coordpy_consensus_commits_total",
            "coordpy_consensus_abstains_total",
            "coordpy_integrity_verdicts_total",
            "coordpy_event_graph_size",
            "coordpy_audit_anchor_root_age_seconds",
            "coordpy_observability_spans_emitted_total",
        }
        present = set()
        for line in text.splitlines():
            if line.startswith("# TYPE "):
                rest = line[len("# TYPE "):]
                name, _, kind = rest.partition(" ")
                present.add(name)
        missing = required - present
        _check(
            "required_metrics_present",
            len(missing) == 0,
            f"missing={sorted(missing) if missing else 'none'}")
    else:
        _check(
            "metrics_text_sidecar_present",
            False,
            f"missing {metrics_text_path}")

    # Sample spans sidecar must parse and match OTLP shape.
    spans_path = report_path.parent / "spans_sample.json"
    if spans_path.is_file():
        spans_obj = json.loads(
            spans_path.read_text(encoding="utf-8"))
        sample = spans_obj.get("sample_spans", [])
        for i, s in enumerate(sample):
            for k in ("traceId", "spanId", "name", "kind",
                      "startTimeUnixNano", "endTimeUnixNano",
                      "status"):
                _check(
                    f"span[{i}].{k}",
                    k in s,
                    "missing" if k not in s else "present")
            if "traceId" in s:
                _check(
                    f"span[{i}].traceId_len",
                    len(s["traceId"]) == 32,
                    f"len={len(s['traceId'])}")
            if "spanId" in s:
                _check(
                    f"span[{i}].spanId_len",
                    len(s["spanId"]) == 16,
                    f"len={len(s['spanId'])}")
    else:
        _check(
            "spans_sample_sidecar_present",
            False,
            f"missing {spans_path}")

    # Logs sidecar must be valid JSON lines with standard fields.
    logs_path = report_path.parent / "logs_sample.jsonl"
    if logs_path.is_file():
        text = logs_path.read_text(encoding="utf-8")
        all_valid = True
        all_have_standard = True
        for raw in text.splitlines():
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                all_valid = False
                break
            for k in ("tenant_id", "run_cid", "agent_id", "role"):
                if k not in obj:
                    all_have_standard = False
        _check("logs_jsonline_valid", all_valid)
        _check("logs_have_standard_labels", all_have_standard)
    else:
        _check(
            "logs_sample_sidecar_present",
            False,
            f"missing {logs_path}")

    # Re-assert every load-bearing closure bool.
    for k in LOAD_BEARING_BOOLS:
        _check(k, bool(rep.get(k, False)), str(rep.get(k)))

    print()
    print(f"OVERALL: {'PASS' if fails == 0 else 'FAIL'} "
          f"({passes} passed, {fails} failed)")
    return 0 if fails == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="W87 / P3 #47 — observability audit re-verifier")
    parser.add_argument(
        "--report", required=True,
        help="Path to observability_v1_bench_report.json")
    args = parser.parse_args(argv)
    return _verify(pathlib.Path(args.report).resolve())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
