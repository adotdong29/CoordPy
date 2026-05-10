"""Tiny TTY-aware presentation helpers for CoordPy CLIs.

Stdlib-only. Detects whether stdout is a TTY and degrades gracefully
to plain ASCII when piped or redirected — so CI logs stay clean and
``> file.txt`` doesn't fill with escape codes.

The output style is deliberately understated:

* Unicode box-drawing for section headers.
* Aligned key/value blocks (``  key  : value``).
* A small palette: bold, dim, green, red, yellow, cyan.
* No emoji, no animation, no progress bars.

External callers should treat these helpers as internal-but-stable —
they are exercised across every CLI subcommand and the bundled
example, but their formatting is not part of any data contract.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------


def _is_color_safe() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("COORDPY_NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    term = os.environ.get("TERM", "")
    if term in ("dumb",):
        return False
    return True


def _is_unicode_safe() -> bool:
    # Matches Python's own approach: only emit non-ASCII glyphs when
    # the active stdout encoding can carry them.
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "utf" in enc


_COLOR = _is_color_safe()
_UNICODE = _is_unicode_safe()


# ---------------------------------------------------------------------------
# Style primitives
# ---------------------------------------------------------------------------


_RESET = "\x1b[0m"
_STYLES = {
    "bold":   "\x1b[1m",
    "dim":    "\x1b[2m",
    "green":  "\x1b[32m",
    "red":    "\x1b[31m",
    "yellow": "\x1b[33m",
    "cyan":   "\x1b[36m",
    "magenta": "\x1b[35m",
}


def style(text: str, *names: str) -> str:
    """Wrap ``text`` in the named ANSI styles when a TTY is detected.

    No-op when stdout is piped/redirected, ``NO_COLOR`` is set, or the
    terminal is non-color. Multiple names compose: ``style("hi",
    "bold", "green")``.
    """
    if not _COLOR or not names:
        return text
    prefix = "".join(_STYLES.get(n, "") for n in names)
    if not prefix:
        return text
    return f"{prefix}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Boxes / headers
# ---------------------------------------------------------------------------


_HORIZ = "─" if _UNICODE else "-"
_TL = "┌" if _UNICODE else "+"
_TR = "┐" if _UNICODE else "+"
_BL = "└" if _UNICODE else "+"
_BR = "┘" if _UNICODE else "+"
_VERT = "│" if _UNICODE else "|"


def header(title: str, *, width: int = 78) -> str:
    """A compact section header rendered as a single bold underlined
    line. Cleaner than full boxes when stacked in a CLI output."""
    bar = _HORIZ * max(8, width - len(title) - 2)
    line = f"{title} {bar}"
    return style(line, "bold")


def box(title: str, lines: Iterable[str], *, width: int = 78) -> str:
    """A boxed block. Use sparingly — usually for the final summary."""
    inner_lines = list(lines)
    inner_w = max(
        [len(title) + 2] + [len(_visible_strlen(line)) for line in inner_lines]
    )
    inner_w = max(inner_w, width - 4)
    out: list[str] = []
    title_str = f" {title} "
    pad = (inner_w - len(title_str))
    out.append(_TL + _HORIZ + style(title_str, "bold") + _HORIZ * max(0, pad) + _TR)
    for line in inner_lines:
        vis = _visible_strlen(line)
        out.append(_VERT + " " + line + " " * max(0, inner_w - vis - 1) + _VERT)
    out.append(_BL + _HORIZ * (inner_w + 2) + _BR)
    return "\n".join(out)


def _visible_strlen(text: str) -> int:
    """Length of ``text`` ignoring ANSI escape sequences."""
    out = 0
    i = 0
    while i < len(text):
        if text[i] == "\x1b":
            # Skip until 'm'.
            while i < len(text) and text[i] != "m":
                i += 1
            i += 1
        else:
            out += 1
            i += 1
    return out


# ---------------------------------------------------------------------------
# Aligned key/value blocks
# ---------------------------------------------------------------------------


def kv(items: Iterable[tuple[str, object]], *, indent: str = "  ",
       key_width: int | None = None) -> str:
    """Render ``items`` as an aligned ``  key  : value`` block.

    If ``key_width`` is None, computes it from the longest key.
    """
    pairs = list(items)
    if not pairs:
        return ""
    kw = key_width or max(len(str(k)) for k, _ in pairs)
    out = []
    for k, v in pairs:
        key = str(k).ljust(kw)
        out.append(f"{indent}{style(key, 'dim')}  {v}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def table(headers: list[str], rows: list[list[object]],
          *, align: list[str] | None = None) -> str:
    """A minimal aligned table.

    ``align`` is a list of ``"l"`` / ``"r"`` per column; default is
    left-align. The first row is rendered bold.
    """
    rows_str: list[list[str]] = [
        [str(c) for c in row] for row in rows
    ]
    cols = len(headers)
    if align is None:
        align = ["l"] * cols
    widths = [
        max([len(headers[i])] + [len(r[i]) for r in rows_str])
        for i in range(cols)
    ]
    def _fmt_row(cells: list[str], *, bold: bool = False) -> str:
        out = []
        for c, w, a in zip(cells, widths, align):
            cell = c.rjust(w) if a == "r" else c.ljust(w)
            out.append(style(cell, "bold") if bold else cell)
        return "  ".join(out)
    out: list[str] = []
    out.append(_fmt_row(headers, bold=True))
    out.append(style(_HORIZ * (sum(widths) + (cols - 1) * 2), "dim"))
    for r in rows_str:
        out.append(_fmt_row(r))
    return "\n".join(out)


def verdict_color(ok: bool) -> str:
    """Return the right ANSI style name for a pass/fail verdict."""
    return "green" if ok else "red"


__all__ = [
    "style",
    "header",
    "box",
    "kv",
    "table",
    "verdict_color",
]
