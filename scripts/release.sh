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
    local root
    local venv
    local out_dir
    root="$(mktemp -d -t coordpy_release_XXXX)"
    venv="$root/venv"
    out_dir="$root/coordpy-smoke"
    "$PY" -m venv "$venv"
    "$venv/bin/pip" install --quiet --upgrade pip
    "$venv/bin/pip" install --quiet dist/*.whl
    "$venv/bin/python" tests/test_smoke_full.py
    "$venv/bin/coordpy" --profile local_smoke --out-dir "$out_dir"
    "$venv/bin/coordpy-capsule" verify --report "$out_dir/product_report.json"
    rm -rf "$root"
}

verify_tag_matches_version() {
    # Read the version directly from the file, without importing
    # ``coordpy`` (which would pull in numpy and the rest of the
    # runtime; the release venv intentionally only has build /
    # twine / check-wheel-contents installed).
    local version
    version="$(grep -E '^__version__' coordpy/_version.py \
        | head -n 1 \
        | sed -E 's/.*"([^"]+)".*/\1/')"
    if [ -z "$version" ]; then
        echo "ERROR: could not parse __version__ from coordpy/_version.py" >&2
        exit 3
    fi
    local tag
    tag="$(git describe --tags --exact-match HEAD 2>/dev/null || true)"
    if [ -z "$tag" ]; then
        echo "ERROR: HEAD has no exact-match git tag." >&2
        echo "       coordpy/_version.py is $version, so tag with:" >&2
        echo "         git tag -a v$version -m 'coordpy-ai v$version'" >&2
        exit 3
    fi
    if [ "$tag" != "v$version" ]; then
        echo "ERROR: HEAD tag $tag does not match coordpy/_version.py ($version)." >&2
        echo "       Expected tag: v$version" >&2
        exit 3
    fi
    if [ -n "$(git status --porcelain)" ]; then
        echo "ERROR: working tree is dirty; commit before tagging." >&2
        git status --short >&2
        exit 3
    fi
    echo "==> tag v$version matches coordpy/_version.py and tree is clean"
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
        # Run the tag + clean-tree gate first; no point building
        # something that would not be allowed to ship anyway.
        ensure_tooling
        verify_tag_matches_version
        build
        verify
        smoke
        echo "==> uploading to PyPI"
        echo "    target: https://pypi.org/project/coordpy-ai/"
        "$VPY" -m twine upload dist/*
        ;;
    *)
        echo "usage: $0 {clean|build|check|testpypi|upload}" >&2
        exit 2
        ;;
esac
