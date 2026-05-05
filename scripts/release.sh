#!/usr/bin/env bash
# Release helper for coordpy-ai.
#
# Usage:
#   ./scripts/release.sh build         # build sdist + wheel
#   ./scripts/release.sh check         # build, twine-check, run smoke driver
#   ./scripts/release.sh testpypi      # build + check + upload to TestPyPI
#   ./scripts/release.sh upload        # build + check + upload to real PyPI
#
# Requires: python -m build, twine, check-wheel-contents.
# Pre-flight: bump coordpy/_version.py, update CHANGELOG, commit.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CMD="${1:-check}"

PY="${PYTHON:-python3}"

clean() {
    echo "==> cleaning build artifacts"
    rm -rf dist build coordpy.egg-info coordpy_ai.egg-info
    find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
}

ensure_tooling() {
    "$PY" -m pip install --quiet --upgrade pip build twine check-wheel-contents
}

build() {
    clean
    ensure_tooling
    echo "==> building sdist + wheel"
    "$PY" -m build
}

verify() {
    echo "==> twine check"
    "$PY" -m twine check dist/*
    echo "==> check-wheel-contents"
    "$PY" -m check_wheel_contents dist/*.whl
}

smoke() {
    echo "==> install wheel into a throwaway venv and run the smoke driver"
    local venv
    venv="$(mktemp -d -t coordpy_release_XXXX)/venv"
    "$PY" -m venv "$venv"
    "$venv/bin/pip" install --quiet --upgrade pip
    "$venv/bin/pip" install --quiet dist/*.whl
    "$venv/bin/python" tests/test_smoke_full.py
    "$venv/bin/python" examples/build_with_coordpy.py
    rm -rf "$(dirname "$venv")"
}

case "$CMD" in
    clean)
        clean
        ;;
    build)
        build
        ;;
    check)
        build
        verify
        smoke
        echo "==> ALL GREEN — ready to upload"
        echo "    sdist:  $(ls dist/*.tar.gz)"
        echo "    wheel:  $(ls dist/*.whl)"
        ;;
    testpypi)
        build
        verify
        smoke
        echo "==> uploading to TestPyPI"
        "$PY" -m twine upload --repository testpypi dist/*
        ;;
    upload)
        build
        verify
        smoke
        echo "==> uploading to PyPI"
        echo "    target: https://pypi.org/project/coordpy-ai/"
        "$PY" -m twine upload dist/*
        ;;
    *)
        echo "usage: $0 {clean|build|check|testpypi|upload}" >&2
        exit 2
        ;;
esac
