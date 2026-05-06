"""Phase-46 public-data import + schema-audit CLI.

A thin, turnkey path for a real public SWE-bench-Lite JSONL (or any
SWE-bench-Lite-shape bank). Three concerns, one invocation:

  * **Schema audit** — every row is inspected against the adapter's
    required + derivable key set. The per-row verdict distinguishes
    *native SWE-bench-Lite shape* (``patch`` / ``test_patch`` /
    ``problem_statement`` / ``repo`` / ``base_commit`` /
    ``instance_id``) from *hermetic shape* (``repo_files`` inline,
    ``buggy_file_relpath``, ``buggy_function``, ``test_source``) and
    names the missing keys if neither shape is satisfied.
  * **Row-level readiness** — delegates to
    ``phase44_public_readiness.run_readiness`` (the Theorem P44-3
    validator) and attaches the five-check verdict + per-check
    failures to the import report.
  * **Failure-mode enumeration** — a malformed file, a
    non-UTF-8 byte, an empty bank, an instance_id duplication, a
    row that is a JSON array instead of an object: each produces a
    specific, actionable message.

One command:

    coordpy.import_data \\
        --jsonl /path/to/swe_bench_lite.jsonl \\
        --out   /tmp/coordpy_run/public_lite_audit.json

Exit code is 0 iff both the schema audit and the readiness verdict
are clean. A non-zero exit carries the blocker list, suitable for a
CI gate. The operator's normal path is then:

    coordpy.import_data --jsonl X --out Y
    coordpy --profile public_jsonl \\
            --jsonl X --out-dir Z

If no public JSONL is on local disk, the runner still accepts a
placeholder path; the import CLI reports the specific blocker
(``file_not_found``) with exit 2 so the operator knows the path is
wrong rather than the pipeline.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Any

from coordpy._internal.experiments.phase44_public_readiness import run_readiness


IMPORT_SCHEMA = "phase46.import_audit.v1"

# Keys the ``SWEBenchAdapter.from_swe_bench_dict`` path recognises.
_NATIVE_REQUIRED = (
    "instance_id", "patch", "problem_statement", "repo", "base_commit")
_HERMETIC_REQUIRED = (
    "instance_id", "buggy_file_relpath", "buggy_function",
    "test_source", "patch")
# A row can satisfy either. Adapter accepts both with repo_files
# inline (hermetic) or fetches them (native + runtime resolution).


def _classify_row_shape(row: dict[str, Any]) -> tuple[str, list[str]]:
    """Return (shape_tag, missing_keys).

    shape_tag is one of ``native_lite``, ``hermetic``, ``ambiguous``,
    ``unusable``. ``missing_keys`` is empty iff one of the two
    shapes is satisfied.
    """
    native_missing = [k for k in _NATIVE_REQUIRED if k not in row]
    hermetic_missing = [k for k in _HERMETIC_REQUIRED if k not in row]
    if not native_missing and not hermetic_missing:
        return "ambiguous", []
    if not native_missing:
        return "native_lite", []
    if not hermetic_missing:
        return "hermetic", []
    # Prefer the shape with fewer missing keys for the error message.
    if len(native_missing) <= len(hermetic_missing):
        return "unusable", native_missing
    return "unusable", hermetic_missing


def audit_jsonl(jsonl_path: str,
                 *, limit: int | None = None,
                 run_readiness_check: bool = True,
                 sandbox_name: str = "subprocess",
                 ) -> dict[str, Any]:
    """Return a ``phase46.import_audit.v1`` report dict.

    On I/O or file-not-found errors the report carries
    ``error_kind`` and ``error_detail`` and marks ``ok = False``.
    """
    abs_path = os.path.abspath(jsonl_path)
    if not os.path.exists(abs_path):
        return {
            "schema": IMPORT_SCHEMA,
            "jsonl_path": abs_path,
            "ok": False,
            "error_kind": "file_not_found",
            "error_detail": f"no such file: {abs_path}",
            "rows": [], "readiness": None, "blockers": ["file_not_found"],
        }

    rows_out: list[dict[str, Any]] = []
    shape_counter: collections.Counter = collections.Counter()
    seen_ids: dict[str, int] = {}
    duplicates: list[tuple[str, int, int]] = []
    n_total = 0
    n_decode_errors = 0
    n_non_object = 0
    try:
        with open(abs_path, "r", encoding="utf-8") as fh:
            for (idx, raw) in enumerate(fh):
                raw = raw.strip()
                if not raw or raw.startswith("#"):
                    continue
                if limit is not None and n_total >= limit:
                    break
                n_total += 1
                row_info: dict[str, Any] = {"idx": idx}
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as ex:
                    n_decode_errors += 1
                    row_info.update({
                        "instance_id": None, "shape": "invalid_json",
                        "missing_keys": [], "reason": str(ex)})
                    rows_out.append(row_info)
                    continue
                if not isinstance(parsed, dict):
                    n_non_object += 1
                    row_info.update({
                        "instance_id": None,
                        "shape": "non_object",
                        "missing_keys": [],
                        "reason": f"row is {type(parsed).__name__}, expected object"})
                    rows_out.append(row_info)
                    continue
                iid = parsed.get("instance_id") or f"<anon-{idx}>"
                if iid in seen_ids:
                    duplicates.append((iid, seen_ids[iid], idx))
                else:
                    seen_ids[iid] = idx
                shape, missing = _classify_row_shape(parsed)
                shape_counter[shape] += 1
                row_info.update({
                    "instance_id": iid,
                    "shape": shape,
                    "missing_keys": missing,
                })
                rows_out.append(row_info)
    except UnicodeDecodeError as ex:
        return {
            "schema": IMPORT_SCHEMA,
            "jsonl_path": abs_path,
            "ok": False,
            "error_kind": "utf8_decode_error",
            "error_detail": str(ex),
            "rows": [], "readiness": None,
            "blockers": ["utf8_decode_error"],
        }

    # Aggregate blockers.
    blockers: list[str] = []
    if n_total == 0:
        blockers.append("empty_bank")
    if n_decode_errors:
        blockers.append(f"json_decode_errors:{n_decode_errors}")
    if n_non_object:
        blockers.append(f"non_object_rows:{n_non_object}")
    if shape_counter.get("unusable"):
        blockers.append(f"unusable_rows:{shape_counter['unusable']}")
    if duplicates:
        blockers.append(f"duplicate_instance_ids:{len(duplicates)}")

    readiness = None
    if run_readiness_check and not blockers:
        readiness = run_readiness(
            abs_path, limit=limit, sandbox_name=sandbox_name)
        if not readiness["ready"]:
            blockers.extend(
                [f"readiness:{b}" for b in readiness["blockers"]])
    elif run_readiness_check and blockers:
        # Do not run the expensive readiness check on a file that has
        # already failed schema; surface that explicitly.
        readiness = {
            "skipped": True,
            "reason": "schema_blockers_present",
            "blockers": blockers[:],
        }

    ok = not blockers
    return {
        "schema": IMPORT_SCHEMA,
        "jsonl_path": abs_path,
        "ok": ok,
        "n_rows": n_total,
        "shape_counts": dict(shape_counter),
        "decode_errors": n_decode_errors,
        "non_object_rows": n_non_object,
        "duplicate_instance_ids": [
            {"instance_id": iid, "first_idx": a, "dup_idx": b}
            for (iid, a, b) in duplicates],
        "rows": rows_out,
        "readiness": readiness,
        "blockers": blockers,
    }


def _render_summary(report: dict[str, Any]) -> str:
    lines = ["=== phase46 public-data import audit ==="]
    lines.append(f"jsonl: {report['jsonl_path']}")
    if not report["ok"] and report.get("error_kind"):
        lines.append(f"FATAL: {report['error_kind']} — "
                      f"{report.get('error_detail','')}")
        return "\n".join(lines) + "\n"
    lines.append(f"rows      : {report.get('n_rows', 0)}")
    lines.append(f"shapes    : {report.get('shape_counts', {})}")
    lines.append(f"decode_err: {report.get('decode_errors', 0)}")
    lines.append(f"non_object: {report.get('non_object_rows', 0)}")
    dups = report.get("duplicate_instance_ids", [])
    lines.append(f"duplicates: {len(dups)}"
                  + (f"  first: {dups[0]}" if dups else ""))
    rd = report.get("readiness")
    if rd is None:
        lines.append("readiness : (not run)")
    elif rd.get("skipped"):
        lines.append(f"readiness : SKIPPED ({rd.get('reason','?')})")
    else:
        lines.append(f"readiness : {'READY' if rd['ready'] else 'NOT READY'} "
                      f"n={rd.get('n','-')} "
                      f"passed={rd.get('n_passed_all','-')} "
                      f"wall={rd.get('wall_seconds','-')}s")
    if report.get("blockers"):
        lines.append(f"blockers  : {report['blockers']}")
    else:
        lines.append("blockers  : []")
    lines.append(f"verdict   : {'OK' if report['ok'] else 'BLOCKED'}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase-46 public-data import + schema audit.")
    ap.add_argument("--jsonl", required=True,
                     help="Path to a SWE-bench-Lite-shape JSONL.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sandbox",
                     choices=("in_process", "subprocess"),
                     default="subprocess")
    ap.add_argument("--skip-readiness", action="store_true",
                     help="Schema audit only; skip the P44-3 validator.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = audit_jsonl(
        args.jsonl, limit=args.limit,
        run_readiness_check=not args.skip_readiness,
        sandbox_name=args.sandbox)
    print(_render_summary(report))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"Wrote {args.out}")
    if report.get("error_kind") == "file_not_found":
        return 2
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
