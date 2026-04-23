"""Wevra provenance manifest.

Every Wevra run emits a provenance block recording:
  * code identity (git SHA if available, package version, Python, platform)
  * profile identity (name, schema version)
  * model/endpoint/sandbox identity
  * input JSONL path + SHA-256 checksum (if resolvable)
  * invocation identity (argv, timestamp)
  * artifact paths written by the run

The manifest is a plain JSON-serializable dict; it is attached to
``product_report.json`` under the ``provenance`` key and also written
standalone as ``provenance.json`` alongside the report.

This module has no dependencies outside the standard library.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import platform as _platform
import subprocess
import sys
from typing import Any, Iterable

PROVENANCE_SCHEMA = "wevra.provenance.v1"


def _git_sha(repo_dir: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            check=False, capture_output=True, text=True, timeout=2.0)
        sha = out.stdout.strip()
        if out.returncode == 0 and len(sha) == 40:
            return sha
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    return None


def _git_dirty(repo_dir: str) -> bool | None:
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "status", "--porcelain"],
            check=False, capture_output=True, text=True, timeout=2.0)
        if out.returncode != 0:
            return None
        return bool(out.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _package_version() -> str:
    try:
        from vision_mvp import __version__
        return str(__version__)
    except Exception:
        return "unknown"


def _sha256_of_file(path: str, max_bytes: int | None = None) -> str | None:
    try:
        h = hashlib.sha256()
        read = 0
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                if max_bytes is not None and read + len(chunk) > max_bytes:
                    h.update(chunk[: max_bytes - read])
                    break
                h.update(chunk)
                read += len(chunk)
        return h.hexdigest()
    except OSError:
        return None


def build_manifest(*,
                    profile_name: str | None = None,
                    profile_schema: str | None = None,
                    jsonl_path: str | None = None,
                    model: str | None = None,
                    endpoint: str | None = None,
                    sandbox: str | None = None,
                    out_dir: str | None = None,
                    artifacts: Iterable[str] | None = None,
                    argv: Iterable[str] | None = None,
                    extra: dict[str, Any] | None = None,
                    repo_dir: str | None = None,
                    ) -> dict[str, Any]:
    """Build a Wevra provenance manifest dict.

    All fields are best-effort — unresolvable values are recorded as
    ``None`` rather than omitted, so downstream consumers can tell the
    difference between "not applicable" and "not collected".
    """
    if repo_dir is None:
        # Default: the repo root inferred from this file's location.
        repo_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", ".."))
    jsonl_abs = os.path.abspath(jsonl_path) if jsonl_path else None
    jsonl_sha = _sha256_of_file(jsonl_abs) if jsonl_abs else None
    jsonl_bytes = (
        os.path.getsize(jsonl_abs)
        if jsonl_abs and os.path.exists(jsonl_abs) else None)
    manifest = {
        "schema": PROVENANCE_SCHEMA,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "code": {
            "git_sha": _git_sha(repo_dir),
            "git_dirty": _git_dirty(repo_dir),
            "package": "vision_mvp",
            "package_version": _package_version(),
            "repo_dir": repo_dir,
        },
        "runtime": {
            "python_version": sys.version.split()[0],
            "python_implementation": _platform.python_implementation(),
            "platform": _platform.platform(),
            "machine": _platform.machine(),
            "system": _platform.system(),
        },
        "profile": {
            "name": profile_name,
            "schema": profile_schema,
        },
        "model": {
            "tag": model,
            "endpoint": endpoint,
        },
        "sandbox": sandbox,
        "input": {
            "jsonl_path": jsonl_abs,
            "jsonl_sha256": jsonl_sha,
            "jsonl_bytes": jsonl_bytes,
        },
        "invocation": {
            "argv": list(argv) if argv is not None else list(sys.argv),
            "cwd": os.getcwd(),
            "user": os.environ.get("USER") or os.environ.get("USERNAME"),
            "hostname": _platform.node(),
        },
        "output": {
            "out_dir": os.path.abspath(out_dir) if out_dir else None,
            "artifacts": sorted(artifacts) if artifacts else [],
        },
        "extra": dict(extra) if extra else {},
    }
    return manifest


__all__ = ["PROVENANCE_SCHEMA", "build_manifest"]
