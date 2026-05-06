#!/usr/bin/env bash
# Smoke checks for contributors. Mirrors what CI runs.
#
# Usage:
#   ./scripts/smoke.sh
#
# Exits 0 if every gate passes, non-zero otherwise.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
OUT="$(mktemp -d -t coordpy_smoke_XXXX)"

cleanup() { rm -rf "$OUT"; }
trap cleanup EXIT

echo "==> coordpy --profile local_smoke"
"$PY" -m coordpy --profile local_smoke --out-dir "$OUT/smoke" > /dev/null

echo "==> coordpy-ci"
"$PY" -m coordpy._cli --help > /dev/null 2>&1 || true
coordpy-ci --report "$OUT/smoke/product_report.json" --min-pass-at-1 1.0 > /dev/null

echo "==> coordpy-capsule verify"
coordpy-capsule verify --report "$OUT/smoke/product_report.json" > /dev/null

echo "==> tests/test_smoke_full.py"
"$PY" tests/test_smoke_full.py > /dev/null

echo "==> ALL SMOKE CHECKS PASSED"
