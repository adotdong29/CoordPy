"""Auto-generate ``docs/THEOREMS_AUTO.md`` from in-code theorem citations.

Scans every ``*.py`` file under ``vision_mvp/`` for docstring or
comment lines that name a theorem using the project's ``Wn-TAG-n``
identifier scheme (e.g. ``Theorem W3-8``, ``Theorem W3-30``). Each
citation is captured once per ``(file, lineno, id)`` triple and
emitted into a grouped markdown index so that readers of the paper
or the docs can jump straight from a theorem ID to the code that
references it.

Usage::

    python -m vision_mvp.scripts.generate_theorem_docs
    # writes: docs/THEOREMS_AUTO.md

The output is idempotent: running the script twice produces the
same bytes (entries are sorted by theorem ID, then file path).

See ``ADVANCEMENT_TO_10_10.md`` Part III §3 for the design note.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# Match the project's theorem id scheme: a letter-digit prefix such
# as ``W3`` followed by an optional short tag and a trailing number.
# Examples matched: ``W3-8``, ``W3-20``, ``W3-Cat-1``, ``W3-30``.
_THEOREM_RE = re.compile(
    r"Theorem\s+(?P<id>W\d+(?:-[A-Za-z]+)?-\d+)"
)


def _iter_py_files(root: Path):
    for p in root.rglob("*.py"):
        parts = p.parts
        if "__pycache__" in parts or "build" in parts or "dist" in parts:
            continue
        yield p


def _extract_from_string(text: str):
    """Yield (theorem_id, line_offset) for each match in ``text``."""
    for m in _THEOREM_RE.finditer(text):
        prefix = text[: m.start()]
        line_offset = prefix.count("\n")
        yield m.group("id"), line_offset


def extract_theorems(source_dir: Path) -> list[dict]:
    """Walk ``source_dir`` and return a list of citation records.

    Each record is ``{'id', 'file', 'lineno', 'context'}``. ``context``
    is a short excerpt of the surrounding text (up to 180 chars)
    so the auto-generated page is informative even without jumping
    into the file.
    """
    out: list[dict] = []
    seen: set[tuple[str, str, int]] = set()

    for path in _iter_py_files(source_dir):
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        # Fast string-level scan catches docstrings, comments, and
        # ordinary string literals equally — every "Theorem X" note,
        # whether in a docstring or a comment, ends up in the file
        # text verbatim.
        for theorem_id, line_offset in _extract_from_string(source):
            lineno = line_offset + 1
            key = (theorem_id, str(path), lineno)
            if key in seen:
                continue
            seen.add(key)
            lines = source.splitlines()
            ctx_start = max(0, line_offset - 0)
            ctx_end = min(len(lines), line_offset + 1)
            context = " ".join(
                line.strip() for line in lines[ctx_start:ctx_end])
            if len(context) > 180:
                context = context[:177] + "..."
            out.append({
                "id": theorem_id,
                "file": str(path),
                "lineno": lineno,
                "context": context,
            })

        # Also walk the AST so we can surface function-level theorem
        # declarations — i.e. functions named ``theorem_w3_<n>`` or
        # similar — with their full docstring. These are rare today
        # but the hook makes the generator future-proof.
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if not node.name.lower().startswith("theorem"):
                continue
            doc = ast.get_docstring(node) or ""
            m = _THEOREM_RE.search(doc) or _THEOREM_RE.search(node.name)
            if not m:
                continue
            key = (m.group("id"), str(path), node.lineno)
            if key in seen:
                continue
            seen.add(key)
            context = doc.strip().splitlines()[0] if doc else node.name
            if len(context) > 180:
                context = context[:177] + "..."
            out.append({
                "id": m.group("id"),
                "file": str(path),
                "lineno": node.lineno,
                "context": context,
            })

    return out


def render_markdown(theorems: list[dict], *, repo_root: Path) -> str:
    """Render a grouped markdown document.

    Theorems are grouped by prefix (``W3``, ``W4``, ...) and then
    sorted numerically by the trailing integer so ``W3-2`` comes
    before ``W3-10``.
    """

    def sort_key(t: dict):
        parts = t["id"].split("-")
        try:
            trailing = int(parts[-1])
        except ValueError:
            trailing = 0
        return (parts[0], trailing, t["id"], t["file"], t["lineno"])

    by_prefix: dict[str, list[dict]] = {}
    for t in sorted(theorems, key=sort_key):
        by_prefix.setdefault(t["id"].split("-")[0], []).append(t)

    lines: list[str] = [
        "# Theorems — auto-generated index",
        "",
        "This file is regenerated by "
        "`python -m vision_mvp.scripts.generate_theorem_docs`. ",
        "It enumerates every `Theorem W*-...-*` citation in the "
        "Python source tree, grouped by prefix.",
        "",
        f"Total citations: **{len(theorems)}** "
        f"across **{len({t['file'] for t in theorems})}** files.",
        "",
    ]

    for prefix in sorted(by_prefix):
        lines.append(f"## {prefix}")
        lines.append("")
        current_id: str | None = None
        for t in by_prefix[prefix]:
            if t["id"] != current_id:
                current_id = t["id"]
                lines.append(f"### {current_id}")
                lines.append("")
            rel = Path(t["file"]).resolve().relative_to(repo_root)
            lines.append(
                f"- `{rel}:{t['lineno']}` — {t['context']}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    repo_root = Path(__file__).resolve().parents[2]
    source_dir = repo_root / "vision_mvp"
    out_path = repo_root / "docs" / "archive" / "legacy-progress-notes" / "THEOREMS_AUTO.md"

    theorems = extract_theorems(source_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(theorems, repo_root=repo_root),
                         encoding="utf-8")
    print(f"wrote {out_path} ({len(theorems)} citations)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
