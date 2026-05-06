#!/usr/bin/env bash
# Release helper for coordpy-ai.
#
# Usage:
#   ./scripts/release.sh build         # build sdist + wheel
#   ./scripts/release.sh check         # build, twine-check, run smoke driver
#   ./scripts/release.sh testpypi      # build + check + upload to TestPyPI
#   ./scripts/release.sh upload        # build + check + upload to real PyPI
#
# Build tooling (build, twine, check-wheel-contents) is installed
# into a private venv under .release-venv so this script works on
# distros where the system Python is managed (Debian/Ubuntu, etc.).
#
# Set PYTHON=... to pick the interpreter that creates the venv.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CMD="${1:-check}"

PY="${PYTHON:-python3}"
VENV="$ROOT/.release-venv"
VPY="$VENV/bin/python"

clean() {
    echo "==> cleaning build artifacts"
    rm -rf dist build coordpy.egg-info coordpy_ai.egg-info
    find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
}

ensure_tooling() {
    if [ ! -x "$VPY" ]; then
        echo "==> creating release venv at .release-venv"
        "$PY" -m venv "$VENV"
    fi
    "$VPY" -m pip install --quiet --upgrade pip
    "$VPY" -m pip install --quiet --upgrade build twine check-wheel-contents
}

build() {
    clean
    ensure_tooling
    echo "==> building sdist + wheel"
    "$VPY" -m build
}

verify() {
    echo "==> twine check"
    "$VPY" -m twine check dist/*
    echo "==> check-wheel-contents"
    "$VPY" -m check_wheel_contents dist/*.whl
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

verify_tag_matches_version() {
    local version
    version="$("$VPY" -c 'from coordpy._version import __version__; print(__version__)')"
    local tag
    tag="$(git describe --tags --exact-match HEAD 2>/dev/null || true)"
    if [ -n "$tag" ] && [ "$tag" != "v$version" ]; then
        echo "ERROR: HEAD tag $tag does not match coordpy/_version.py ($version)" >&2
        echo "       expected tag: v$version" >&2
        exit 3
    fi
    if [ -n "$(git status --porcelain)" ]; then
        echo "WARNING: working tree is dirty; commit before tagging." >&2
    fi
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
        "$VPY" -m twine upload --repository testpypi dist/*
        ;;
    upload)
        build
        verify
        smoke
        verify_tag_matches_version
        echo "==> uploading to PyPI"
        echo "    target: https://pypi.org/project/coordpy-ai/"
        "$VPY" -m twine upload dist/*
        ;;
    *)
        echo "usage: $0 {clean|build|check|testpypi|upload}" >&2
        exit 2
        ;;
esac
