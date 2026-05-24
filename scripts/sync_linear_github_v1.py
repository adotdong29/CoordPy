#!/usr/bin/env python3
"""Durable Linear ↔ GitHub sync helper.

Reads ``linear_github_mapping.json`` at repo root, validates that
every referenced commit + doc exists, and emits a Linear-ready
markdown summary that can be pasted into a Linear comment or
project document on end-of-milestone.

This is intentionally a SMALL, DURABLE script.  It does NOT call
Linear's MCP — Linear writes are still done explicitly via the
MCP tools so they remain auditable in conversation history.  This
script's job is to (a) make sure the mapping is honest and (b)
produce a copy-pasteable snapshot of the current truth surface
that future-you (or another agent) can re-sync from.

Usage::

    python scripts/sync_linear_github_v1.py validate
    python scripts/sync_linear_github_v1.py snapshot
    python scripts/sync_linear_github_v1.py snapshot --milestone W95
    python scripts/sync_linear_github_v1.py snapshot --markdown > snapshot.md

Exit code is non-zero IFF validation finds drift (missing
commits, missing docs, malformed entries).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MAPPING_PATH = REPO_ROOT / "linear_github_mapping.json"


def _load_mapping() -> dict:
    if not MAPPING_PATH.exists():
        raise SystemExit(
            f"mapping file not found at {MAPPING_PATH}")
    return json.loads(MAPPING_PATH.read_text())


def _git_commit_exists(sha: str) -> bool:
    if not sha:
        return False
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=REPO_ROOT, check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def _git_commit_subject(sha: str) -> str:
    try:
        r = subprocess.run(
            ["git", "log", "-1", "--format=%s", sha],
            cwd=REPO_ROOT, check=True,
            capture_output=True, text=True)
        return r.stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def _git_origin_has(sha: str) -> bool:
    if not sha:
        return False
    try:
        r = subprocess.run(
            ["git", "branch", "-r", "--contains", sha],
            cwd=REPO_ROOT, check=True,
            capture_output=True, text=True)
        return "origin/main" in r.stdout or "origin/master" in r.stdout
    except subprocess.CalledProcessError:
        return False


def _doc_exists(path: str) -> bool:
    return (REPO_ROOT / path).is_file()


def cmd_validate(args) -> int:
    mapping = _load_mapping()
    issues: list[str] = []
    for m in mapping.get("milestones", []):
        mid = m.get("id", "?")
        for sha in m.get("commits", []):
            if not _git_commit_exists(sha):
                issues.append(
                    f"{mid}: commit {sha} not present in repo")
            elif not _git_origin_has(sha):
                issues.append(
                    f"{mid}: commit {sha} not yet pushed to "
                    "origin/main")
        for d in m.get("docs", []):
            if not _doc_exists(d):
                issues.append(
                    f"{mid}: doc {d} not found at repo root")
    if not issues:
        print(
            f"OK: {len(mapping.get('milestones', []))} "
            "milestones, all commits + docs present.")
        return 0
    for x in issues:
        print(f"DRIFT: {x}")
    return 2


def _render_markdown(
        mapping: dict, *,
        only_milestone: str | None) -> str:
    out: list[str] = []
    out.append(
        f"# {mapping['team']} ↔ GitHub sync snapshot\n")
    out.append(
        f"Linear workspace: {mapping['linear_workspace_url']}  ")
    out.append(
        f"GitHub repo: `{mapping['github_repo']}`  \n")
    for m in mapping.get("milestones", []):
        if only_milestone and m["id"] != only_milestone:
            continue
        out.append(f"## {m['id']} — {m['title']}\n")
        out.append(f"**Outcome**: {m['outcome']}\n")
        commits = m.get("commits") or []
        if commits:
            out.append("**Commits**:")
            for sha in commits:
                subj = _git_commit_subject(sha)
                on_remote = (
                    " on origin/main"
                    if _git_origin_has(sha)
                    else " (LOCAL ONLY)")
                out.append(
                    f"* `{sha}` — {subj}{on_remote}")
            out.append("")
        docs = m.get("docs") or []
        if docs:
            out.append("**Repo docs**:")
            for d in docs:
                present = (
                    "" if _doc_exists(d) else " (MISSING)")
                out.append(f"* `{d}`{present}")
            out.append("")
        issues = m.get("linear_issues") or []
        if issues:
            out.append("**Linear issues**:")
            for iid in issues:
                out.append(
                    f"* `{iid}` — "
                    f"https://linear.app/coordpy/issue/{iid}")
            out.append("")
        retired = m.get("carry_forwards_retired") or []
        if retired:
            out.append(
                "**Carry-forwards retired**: "
                + ", ".join(f"`{r}`" for r in retired))
        added = m.get("carry_forwards_added") or []
        if added:
            out.append(
                "**Carry-forwards added**: "
                + ", ".join(f"`{a}`" for a in added))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def cmd_snapshot(args) -> int:
    mapping = _load_mapping()
    md = _render_markdown(
        mapping, only_milestone=args.milestone)
    sys.stdout.write(md)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub_v = sub.add_parser(
        "validate",
        help=("Verify every referenced commit + doc exists; "
              "exit non-zero on drift."))
    sub_v.set_defaults(func=cmd_validate)
    sub_s = sub.add_parser(
        "snapshot",
        help=("Emit a Linear-ready markdown snapshot of the "
              "current truth surface."))
    sub_s.add_argument(
        "--milestone", default=None,
        help="Only emit a single milestone (e.g. W95).")
    sub_s.add_argument(
        "--markdown", action="store_true",
        help=("[reserved] reserved for future formats; "
              "currently a no-op since markdown is the "
              "default."))
    sub_s.set_defaults(func=cmd_snapshot)
    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
