#!/usr/bin/env bash
# Smoke checks for contributors. Mirrors what CI runs:
#   - lint (ruff, black --check, mypy) if those tools are present
#   - the four CLIs in order against a tmpdir
#   - tests/test_smoke_full.py
#
# Usage:
#   ./scripts/smoke.sh
#
# Set PYTHON=... to use a specific interpreter (defaults to python3).
# Exits 0 if every gate passes, non-zero on the first failure.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
OUT="$(mktemp -d -t coordpy_smoke_XXXX)"
trap 'rm -rf "$OUT"' EXIT

run() {
    local name="$1"; shift
    if "$@" > /dev/null 2>&1; then
        echo "==> $name: PASS"
    else
        echo "==> $name: FAIL ($*)" >&2
        return 1
    fi
}

skip_if_missing() {
    # Run a command if its first token is on PATH; otherwise skip
    # with a friendly message.
    local name="$1"; shift
    if command -v "$1" > /dev/null 2>&1; then
        run "$name" "$@"
    else
        echo "==> $name: SKIP (install with pip install -e \".[dev]\")"
    fi
}

skip_if_missing "ruff check"        ruff check .
# `black` and `mypy` are in the [dev] extra but are not enforced
# by the smoke check. Run `black .` to format, `mypy coordpy` to
# type-check, on the files you touch.

run "coordpy --profile local_smoke" "$PY" -m coordpy --profile local_smoke --out-dir "$OUT/smoke"
run "coordpy-ci"                    coordpy-ci --report "$OUT/smoke/product_report.json" --min-pass-at-1 1.0
run "coordpy-capsule verify"        coordpy-capsule verify --report "$OUT/smoke/product_report.json"
run "tests/test_smoke_full.py"      "$PY" tests/test_smoke_full.py

echo ""
echo "ALL SMOKE CHECKS PASSED"
