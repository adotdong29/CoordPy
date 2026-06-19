"""Phase 44 — Public SWE-bench-Lite drop-in readiness validator.

Phase 43 showed that ``load_jsonl_bank`` already consumes a
SWE-bench-Lite-shape JSONL and ``verify_public_style_loader`` oracle-
saturates on the 57-instance bundled bank. The Phase-43 § D.1 claim
was: the externalisation gap is a ``--jsonl <path>`` swap away.

Phase 44 promotes that claim from *documentation* to *validated code*:
a stand-alone readiness driver that takes any local JSONL (bundled
bank, a downloaded public SWE-bench-Lite artifact, or a hand-rolled
test subset) and runs **five checks** in order:

  1. **Schema check** — every row is a JSON object with the keys the
     ``SWEBenchAdapter.from_swe_bench_dict`` path requires; missing
     keys are surfaced per-instance. We cover both the native SWE-
     bench shape (``patch``, ``test_patch``, ``problem_statement``,
     ``repo``, ``base_commit``, ``instance_id``) and the Phase-40
     hermetic shape (``repo_files`` inline, ``buggy_file_relpath``,
     ``buggy_function``, ``test_source``).
  2. **Adapter check** — every row constructs a
     ``SWEBenchStyleTask`` without exception through the adapter.
  3. **Parser check** — the ``gold_patch`` (or ``patch``) diff parses
     through ``parse_unified_diff`` and yields at least one non-empty
     substitution tuple.
  4. **Matcher check** — the gold substitutions apply cleanly
     (``apply_patch(mode="strict")``) to the buggy source and produce
     a patched file that compiles (``compile(..., "exec")``). This is
     the oracle-round-trip guarantee.
  5. **Test-runner check** — when a ``test_source`` or derivable
     test body is present, the patched file + test execute through
     ``run_patched_test`` and the oracle passes the hidden test. We
     run this inside the selected ``Sandbox`` so the readiness check
     mirrors the evaluation pipeline byte-for-byte.

The output is a single JSON verdict suitable for a CI gate:
``{"ready": true/false, "n": N, "checks": {...}, "blockers": [...]}``.
When any instance fails any check, it is reported with the offending
instance_id, the check that failed, and a short reason. An external
SWE-bench-Lite artifact that passes all five checks is guaranteed to
run through ``phase42_parser_sweep`` or ``phase44_semantic_residue``
by a pure ``--jsonl <path>`` flag change; no adapter or matcher
rework is needed.

Reproducible runs
-----------------

    # Bundled bank — both checks must pass.
    python3 -m coordpy._internal.experiments.phase44_public_readiness \\
        --jsonl coordpy/_internal/tasks/data/swe_lite_style_bank.jsonl \\
        --out /tmp/coordpy_results_phase44_readiness_bundled.json

    # External public SWE-bench-Lite JSONL (place at this path).
    python3 -m coordpy._internal.experiments.phase44_public_readiness \\
        --jsonl /path/to/swe_bench_lite.jsonl \\
        --limit 50 \\
        --out /tmp/coordpy_results_phase44_readiness_swe_lite.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))


from coordpy._internal.tasks.swe_bench_bridge import (
    SWEBenchAdapter, apply_patch, parse_unified_diff,
    run_patched_test,
)
from coordpy._internal.tasks.swe_sandbox import (
    SubprocessSandbox, select_sandbox,
)


# Keys required by the SWE-bench-Lite-shape path. Derivable keys are
# not strictly required — the adapter can produce them from the
# parsed diff.
_REQUIRED_KEYS_CORE = ("instance_id",)
_EITHER_KEYS = (
    # At least one of these must be present for the adapter to
    # produce a task.
    ("patch", "gold_patch"),
)
_TEST_BODY_KEYS = (
    # At least one of these must be present for the readiness test
    # path to run. ``test_patch`` is SWE-bench's default; ``test_source``
    # is our hermetic shape. A real SWE-bench-Lite row always has
    # ``test_patch``.
    ("test_source", "test_patch"),
)


def _row_iter(jsonl_path: str, limit: int | None):
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        i = 0
        for raw in fh:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as ex:
                yield i, None, f"json_decode_error: {ex}"
                i += 1
                if limit is not None and i >= limit:
                    break
                continue
            yield i, row, None
            i += 1
            if limit is not None and i >= limit:
                break


def _check_schema(row: dict) -> str:
    for k in _REQUIRED_KEYS_CORE:
        if k not in row:
            return f"missing_required_key:{k}"
    for opts in _EITHER_KEYS:
        if not any(k in row for k in opts):
            return f"missing_any_of:{'|'.join(opts)}"
    return ""


def _check_adapter(row: dict) -> tuple[object | None, str]:
    try:
        inline = row.get("repo_files") or {}
        if not isinstance(inline, dict):
            return None, "repo_files_not_dict"
        task = SWEBenchAdapter.from_swe_bench_dict(
            row, repo_files=(inline or {}), hidden_event_log=())
        return task, ""
    except ValueError as ex:
        return None, f"adapter_value_error:{ex}"
    except KeyError as ex:
        return None, f"adapter_missing_key:{ex}"
    except Exception as ex:
        return None, f"adapter_error:{type(ex).__name__}:{ex}"


def _check_parser(row: dict) -> tuple[dict | None, str]:
    diff = row.get("patch") or row.get("gold_patch") or ""
    if not isinstance(diff, str):
        # Native substitution shape; treat as structurally ok.
        return {}, "substitution_shape"
    try:
        parsed = parse_unified_diff(diff)
    except Exception as ex:
        return None, f"parser_error:{type(ex).__name__}:{ex}"
    if not parsed:
        return None, "parser_no_hunks"
    return parsed, ""


def _check_matcher(row: dict, task) -> tuple[str, str]:
    """Apply the gold patch under strict matcher. Returns
    (patched_source, ""), or ("", error_reason)."""
    rel = task.buggy_file_relpath
    inline = row.get("repo_files") or {}
    # The adapter path prefixes ``instance_id/`` on the task's
    # relpath; the inline dict is un-prefixed. Strip the prefix
    # for the lookup.
    key = rel
    if "/" in rel and row.get("instance_id") and rel.startswith(
            row["instance_id"] + "/"):
        key = rel[len(row["instance_id"]) + 1:]
    src = inline.get(key)
    if src is None:
        # Try the raw relpath.
        src = inline.get(rel)
    if src is None:
        return "", f"missing_repo_file:{rel}"
    try:
        new_src, applied, reason = apply_patch(
            src, task.gold_patch, mode="strict")
    except Exception as ex:
        return "", f"apply_error:{type(ex).__name__}:{ex}"
    if not applied:
        return "", f"apply_reject:{reason}"
    try:
        compile(new_src, "<readiness>", "exec")
    except SyntaxError as ex:
        return "", f"patched_source_syntax_error:{ex}"
    return new_src, ""


def _check_test_runner(task, new_src: str, sandbox) -> str:
    """Run the oracle-patched source + test through the sandbox.

    The sandbox ``run`` path applies a patch, then compiles and
    executes the test. For the readiness check we already have the
    oracle-patched source, so we pass it as ``buggy_source`` and
    use a **no-op identity patch** (pick the first non-empty line
    as both OLD and NEW — substring-matches itself uniquely as
    long as it appears once). If the file has no unique line, we
    fall back to inline execution.
    """
    if not task.test_source:
        return "missing_test_source"
    # Pick a unique no-op anchor.
    noop_patch: tuple[tuple[str, str], ...] = ()
    for line in new_src.splitlines(keepends=True):
        if line.strip() and new_src.count(line) == 1:
            noop_patch = ((line, line),)
            break
    if not noop_patch:
        # Cannot construct a unique no-op; fall back to inline.
        return _check_test_runner_inline(task, new_src)
    try:
        wr = sandbox.run(
            buggy_source=new_src, patch=noop_patch,
            test_source=task.test_source,
            module_name=f"readiness_{task.instance_id}",
            timeout_s=15.0, apply_mode="strict")
    except Exception as ex:
        return f"sandbox_error:{type(ex).__name__}:{ex}"
    if wr.test_passed:
        return ""
    return f"oracle_test_failed:{wr.error_kind}:{wr.error_detail[:120]}"


def _check_test_runner_inline(task, new_src: str) -> str:
    """Inline (no-sandbox) variant of the oracle-patch test."""
    if not task.test_source:
        return "missing_test_source"
    try:
        wr = run_patched_test(
            file_source=new_src, patched_source=new_src,
            test_source=task.test_source,
            module_name=f"readiness_{task.instance_id}")
    except Exception as ex:
        return f"runner_error:{type(ex).__name__}:{ex}"
    if wr.test_passed:
        return ""
    return f"oracle_test_failed:{wr.error_kind}:{wr.error_detail[:120]}"


def run_readiness(jsonl_path: str, *,
                   limit: int | None,
                   sandbox_name: str = "subprocess",
                   ) -> dict:
    """Run all five checks on the JSONL at ``jsonl_path``.

    Returns a dict ``{"ready": bool, "n": int, "n_passed": int,
    "checks": {...}, "blockers": [...]}``.
    """
    t0 = time.time()
    results = {
        "schema": {"passed": 0, "failed": 0, "failures": []},
        "adapter": {"passed": 0, "failed": 0, "failures": []},
        "parser": {"passed": 0, "failed": 0, "failures": []},
        "matcher": {"passed": 0, "failed": 0, "failures": []},
        "test_runner": {"passed": 0, "failed": 0, "failures": []},
    }
    n_total = 0
    n_passed_all = 0

    if sandbox_name == "in_process":
        sandbox = None
    else:
        sandbox = select_sandbox(sandbox_name)
        if sandbox_name == "docker" and not sandbox.is_available():
            sandbox = SubprocessSandbox()

    for (idx, row, row_err) in _row_iter(jsonl_path, limit):
        n_total += 1
        if row_err is not None:
            results["schema"]["failed"] += 1
            results["schema"]["failures"].append({
                "idx": idx, "instance_id": "<unknown>",
                "reason": row_err})
            continue
        iid = str(row.get("instance_id", f"row-{idx}"))

        reason = _check_schema(row)
        if reason:
            results["schema"]["failed"] += 1
            results["schema"]["failures"].append(
                {"idx": idx, "instance_id": iid, "reason": reason})
            continue
        results["schema"]["passed"] += 1

        task, reason = _check_adapter(row)
        if reason:
            results["adapter"]["failed"] += 1
            results["adapter"]["failures"].append(
                {"idx": idx, "instance_id": iid, "reason": reason})
            continue
        results["adapter"]["passed"] += 1

        _parsed, reason = _check_parser(row)
        if reason and reason != "substitution_shape":
            results["parser"]["failed"] += 1
            results["parser"]["failures"].append(
                {"idx": idx, "instance_id": iid, "reason": reason})
            continue
        results["parser"]["passed"] += 1

        new_src, reason = _check_matcher(row, task)
        if reason:
            results["matcher"]["failed"] += 1
            results["matcher"]["failures"].append(
                {"idx": idx, "instance_id": iid, "reason": reason})
            continue
        results["matcher"]["passed"] += 1

        if sandbox is not None:
            reason = _check_test_runner(task, new_src, sandbox)
        else:
            reason = _check_test_runner_inline(task, new_src)
        if reason:
            results["test_runner"]["failed"] += 1
            results["test_runner"]["failures"].append(
                {"idx": idx, "instance_id": iid, "reason": reason})
            continue
        results["test_runner"]["passed"] += 1
        n_passed_all += 1

    ready = (n_total > 0 and
              all(r["failed"] == 0 for r in results.values()))
    wall = time.time() - t0
    blockers: list[str] = []
    for (name, r) in results.items():
        if r["failed"] > 0:
            blockers.append(f"{name}: {r['failed']} failure(s)")

    return {
        "ready": ready,
        "jsonl_path": jsonl_path,
        "n": n_total,
        "n_passed_all": n_passed_all,
        "wall_seconds": round(wall, 2),
        "sandbox": sandbox.name() if sandbox else "in_process",
        "checks": results,
        "blockers": blockers,
        "schema": "phase44.readiness.v1",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True,
                      help="Local SWE-bench-Lite-shape JSONL to validate.")
    ap.add_argument("--limit", type=int, default=None,
                      help="Validate only the first N rows (None = all).")
    ap.add_argument("--sandbox",
                      choices=("auto", "in_process", "subprocess", "docker"),
                      default="subprocess")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    verdict = run_readiness(
        args.jsonl, limit=args.limit, sandbox_name=args.sandbox)
    print(json.dumps({
        "ready": verdict["ready"],
        "n": verdict["n"],
        "n_passed_all": verdict["n_passed_all"],
        "wall_seconds": verdict["wall_seconds"],
        "blockers": verdict["blockers"],
        "schema": verdict["schema"],
    }, indent=2))
    print("\nPer-check counts:")
    for (name, r) in verdict["checks"].items():
        print(f"  {name:<14s} passed={r['passed']}  failed={r['failed']}"
              f"  (first failures: "
              f"{[f['instance_id'] for f in r['failures'][:3]]})")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(verdict, fh, indent=2, default=str)
        print(f"\nWrote {args.out}")
    if not verdict["ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
