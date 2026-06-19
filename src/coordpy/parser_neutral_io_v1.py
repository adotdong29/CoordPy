"""W137 / COO-9 — parser-neutral I/O kernel + the HC1 parser-neutrality gate.

W136 root-caused the W132–W135 "wrong-algorithm ceiling": the generated WRONG_ALGORITHM /
SEARCH_ENUM battlefield was an **I/O-FORMAT CONFOUND**, not algorithm capability.  The W132
generators flatten every structure onto ONE body line (``_case`` joins the whole body with
spaces), so a grid / a list of (w,v) pairs / a list of (s,e,w) triples all collapse to a single
line.  The reference reads ``sys.stdin.read().split()`` (token-stream, format-agnostic) and is
correct either way, but a model that obeys the statement's "then N lines each with ..." writes a
PER-LINE reader that crashes on the flattened body.  The model's *algorithm* was correct; only its
*parser* broke (quadruple-confirmed in W136; standard-I/O A0 one-shots all three traps).

This module removes that confound at the SOURCE.  It defines a **canonical one-logical-item-per-line
normal form** and a machine-checkable **parser-neutrality gate (HC1)** that certifies, for every
minted case, that:

  (1) the input is in canonical normal form (line 0 = header; each subsequent line is exactly one
      logical item — a k-tuple is k tokens, a grid row is one string of length C, an array is one
      line), AND
  (2) a STRICT per-line reader (the parser a model writes from the statement) and a read-all-tokens
      reader recover the **byte-identical structured data** — i.e. the format is unambiguous, so the
      W136 confound provably cannot recur.

The kernel is *algorithm-free*: it parses to structured fields, not to an answer, so HC1 measures
I/O presentation only.  Templates render their inputs via :func:`render_normal_form_v1` and declare
an :class:`IoShapeV1`; the W137 build self-test runs :func:`parser_neutrality_gate_v1` on every
minted sample + secret case.

Primary-source grounding (Lane γ): the gate operationalises the FormatSpread / ReCode finding that
trivial I/O presentation can swing code-LLM pass-rates 40–76 pp (arXiv:2310.11324; arXiv:2411.10541;
arXiv:2212.10264) — so format-invariance must be a first-class, enforced property, not an assumption.

Pure / deterministic / no I/O / no model inference.  Explicit-import-only; ``coordpy/__init__.py``
is untouched.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import deque
from typing import Any, Optional, Sequence, Union

PARSER_NEUTRAL_IO_V1_SCHEMA_VERSION: str = "coordpy.parser_neutral_io_v1.v1"

# ---- I/O block kinds (a normal-form input is a sequence of these) -----------------------
BLOCK_SCALAR_LINE: str = "scalar_line"   # one line of `len(names)` integers, named
BLOCK_ARRAY_LINE: str = "array_line"     # one line of `count` integers (count = a prior scalar)
BLOCK_ROWS: str = "rows"                 # `count` lines, each exactly `len(fields)` integers
BLOCK_GRID: str = "grid"                 # `rows` lines, each one string of length `cols`


class NormalFormError(ValueError):
    """A per-line read failed because the input is NOT in canonical normal form."""


# ===================================================== I/O shape descriptor

@dataclasses.dataclass(frozen=True)
class IoBlockV1:
    """One block of a normal-form input.

    * ``BLOCK_SCALAR_LINE`` — ``names`` integers on one line (e.g. ``["N", "W"]``).
    * ``BLOCK_ARRAY_LINE``  — ``field`` = an array of length ``count_ref`` on one line
      (``count_ref`` names a previously-read scalar, or is an int literal).
    * ``BLOCK_ROWS``        — ``field`` = ``count_ref`` rows, each ``names`` integers, one row/line.
    * ``BLOCK_GRID``        — ``field`` = ``rows_ref`` strings each of length ``cols_ref``, one/line.
    """

    kind: str
    field: str = ""                    # output field name (array/rows/grid); scalars name themselves
    names: tuple[str, ...] = ()        # scalar names, or per-row field names
    count_ref: Union[str, int] = 0     # array length / row count (a scalar name or an int)
    rows_ref: Union[str, int] = 0      # grid row count
    cols_ref: Union[str, int] = 0      # grid column count (string length per row)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "field": self.field, "names": list(self.names),
                "count_ref": self.count_ref, "rows_ref": self.rows_ref,
                "cols_ref": self.cols_ref}


@dataclasses.dataclass(frozen=True)
class IoShapeV1:
    """An ordered sequence of :class:`IoBlockV1`.  Declares a problem family's canonical I/O."""

    blocks: tuple[IoBlockV1, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": PARSER_NEUTRAL_IO_V1_SCHEMA_VERSION,
                "blocks": [b.to_dict() for b in self.blocks]}

    def shape_cid(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


# convenience constructors (keep call-sites in slate-v2 readable) -------------------------

def scalar_line(*names: str) -> IoBlockV1:
    return IoBlockV1(kind=BLOCK_SCALAR_LINE, names=tuple(names))


def array_line(field: str, count_ref: Union[str, int]) -> IoBlockV1:
    return IoBlockV1(kind=BLOCK_ARRAY_LINE, field=field, count_ref=count_ref)


def rows(field: str, count_ref: Union[str, int], *names: str) -> IoBlockV1:
    return IoBlockV1(kind=BLOCK_ROWS, field=field, names=tuple(names), count_ref=count_ref)


def grid(field: str, rows_ref: Union[str, int], cols_ref: Union[str, int]) -> IoBlockV1:
    return IoBlockV1(kind=BLOCK_GRID, field=field, rows_ref=rows_ref, cols_ref=cols_ref)


def io_shape(*blocks: IoBlockV1) -> IoShapeV1:
    return IoShapeV1(blocks=tuple(blocks))


# ===================================================== rendering (templates emit this)

def _resolve(ref: Union[str, int], scope: dict[str, Any]) -> int:
    if isinstance(ref, int):
        return ref
    if ref not in scope:
        raise KeyError(f"count_ref {ref!r} not yet read")
    return int(scope[ref])


def render_normal_form_v1(shape: IoShapeV1, data: dict[str, Any]) -> str:
    """Render ``data`` to a canonical one-logical-item-per-line string.

    ``data`` maps field names to values: scalar names -> int; an array field -> a list of ints; a
    rows field -> a list of equal-length int tuples; a grid field -> a list of strings.  The output
    is the normal form: NO two distinct logical items ever share a line.
    """
    out_lines: list[str] = []
    scope: dict[str, Any] = {}
    for b in shape.blocks:
        if b.kind == BLOCK_SCALAR_LINE:
            vals = [int(data[n]) for n in b.names]
            for n, v in zip(b.names, vals):
                scope[n] = v
            out_lines.append(" ".join(str(v) for v in vals))
        elif b.kind == BLOCK_ARRAY_LINE:
            arr = list(data[b.field])
            n = _resolve(b.count_ref, scope)
            if len(arr) != n:
                raise NormalFormError(f"array {b.field!r} len {len(arr)} != count {n}")
            scope[b.field] = arr
            out_lines.append(" ".join(str(int(x)) for x in arr))
        elif b.kind == BLOCK_ROWS:
            rws = list(data[b.field])
            n = _resolve(b.count_ref, scope)
            if len(rws) != n:
                raise NormalFormError(f"rows {b.field!r} len {len(rws)} != count {n}")
            stride = len(b.names)
            for r in rws:
                if len(r) != stride:
                    raise NormalFormError(f"row in {b.field!r} has {len(r)} != stride {stride}")
                out_lines.append(" ".join(str(int(x)) for x in r))
            scope[b.field] = rws
        elif b.kind == BLOCK_GRID:
            g = list(data[b.field])
            r = _resolve(b.rows_ref, scope)
            c = _resolve(b.cols_ref, scope)
            if len(g) != r:
                raise NormalFormError(f"grid {b.field!r} rows {len(g)} != {r}")
            for row in g:
                if len(row) != c:
                    raise NormalFormError(f"grid row len {len(row)} != cols {c}")
            out_lines.extend(str(row) for row in g)
            scope[b.field] = g
        else:
            raise NormalFormError(f"unknown block kind {b.kind!r}")
    return "\n".join(out_lines) + "\n"


# ===================================================== the two canonical readers

def parse_per_line_v1(text: str, shape: IoShapeV1) -> dict[str, Any]:
    """STRICT per-line reader — the parser a model writes when it obeys the statement.

    Reads exactly one logical item per line; raises :class:`NormalFormError` if any body line has
    the wrong token count, the wrong row-string length, or is missing.  This is the reader the W132
    confound *broke* (the flattened body had too many tokens on one line / too few lines).
    """
    lines = text.split("\n")
    # drop a single trailing empty line from the final newline; keep internal blanks significant
    if lines and lines[-1] == "":
        lines = lines[:-1]
    it = iter(lines)
    scope: dict[str, Any] = {}

    def _next_line() -> str:
        try:
            return next(it)
        except StopIteration as ex:
            raise NormalFormError("ran out of lines (per-line reader)") from ex

    for b in shape.blocks:
        if b.kind == BLOCK_SCALAR_LINE:
            toks = _next_line().split()
            if len(toks) != len(b.names):
                raise NormalFormError(
                    f"scalar line has {len(toks)} tokens, expected {len(b.names)}")
            for n, t in zip(b.names, toks):
                scope[n] = int(t)
        elif b.kind == BLOCK_ARRAY_LINE:
            n = _resolve(b.count_ref, scope)
            toks = _next_line().split()
            if len(toks) != n:
                raise NormalFormError(f"array line has {len(toks)} tokens, expected {n}")
            scope[b.field] = [int(t) for t in toks]
        elif b.kind == BLOCK_ROWS:
            n = _resolve(b.count_ref, scope)
            stride = len(b.names)
            acc: list[tuple[int, ...]] = []
            for _ in range(n):
                toks = _next_line().split()
                if len(toks) != stride:
                    raise NormalFormError(
                        f"row has {len(toks)} tokens, expected stride {stride}")
                acc.append(tuple(int(t) for t in toks))
            scope[b.field] = acc
        elif b.kind == BLOCK_GRID:
            r = _resolve(b.rows_ref, scope)
            c = _resolve(b.cols_ref, scope)
            acc_g: list[str] = []
            for _ in range(r):
                ln = _next_line()
                if len(ln) != c:
                    raise NormalFormError(f"grid row len {len(ln)} != cols {c}")
                acc_g.append(ln)
            scope[b.field] = acc_g
        else:
            raise NormalFormError(f"unknown block kind {b.kind!r}")
    return scope


def parse_all_tokens_v1(text: str, shape: IoShapeV1) -> dict[str, Any]:
    """Read-all-whitespace-tokens reader — the format-agnostic parser the reference uses.

    For GRID blocks the row strings are the next ``rows`` whitespace tokens (this is exactly how the
    W132 refs read flattened grids).  This reader *cannot* fail on a flattened body — which is why,
    when it disagrees with :func:`parse_per_line_v1`, the input is confounded.
    """
    dq: deque[str] = deque(text.split())
    scope: dict[str, Any] = {}

    def _pop() -> str:
        if not dq:
            raise NormalFormError("ran out of tokens (all-tokens reader)")
        return dq.popleft()

    for b in shape.blocks:
        if b.kind == BLOCK_SCALAR_LINE:
            for n in b.names:
                scope[n] = int(_pop())
        elif b.kind == BLOCK_ARRAY_LINE:
            n = _resolve(b.count_ref, scope)
            scope[b.field] = [int(_pop()) for _ in range(n)]
        elif b.kind == BLOCK_ROWS:
            n = _resolve(b.count_ref, scope)
            stride = len(b.names)
            scope[b.field] = [tuple(int(_pop()) for _ in range(stride)) for _ in range(n)]
        elif b.kind == BLOCK_GRID:
            r = _resolve(b.rows_ref, scope)
            scope[b.field] = [str(_pop()) for _ in range(r)]
        else:
            raise NormalFormError(f"unknown block kind {b.kind!r}")
    return scope


# ===================================================== HC1 — parser-neutrality gate

@dataclasses.dataclass(frozen=True)
class ParserNeutralityRecordV1:
    n_cases: int
    n_normal_form: int          # cases whose stored input == its own re-rendered normal form
    n_dual_parser_agree: int    # cases where per-line == all-tokens structured data
    n_per_line_ok: int          # cases where the strict per-line reader did NOT raise
    is_parser_neutral: bool
    first_failure: str

    def to_dict(self) -> dict[str, Any]:
        return {"n_cases": self.n_cases, "n_normal_form": self.n_normal_form,
                "n_dual_parser_agree": self.n_dual_parser_agree,
                "n_per_line_ok": self.n_per_line_ok,
                "is_parser_neutral": bool(self.is_parser_neutral),
                "first_failure": self.first_failure}


def parser_neutrality_gate_v1(inputs: Sequence[str], shape: IoShapeV1,
                              ) -> ParserNeutralityRecordV1:
    """HC1.  An input set is parser-neutral iff, for EVERY case:

    * the strict per-line reader does not raise (the statement's per-line format actually works), AND
    * the per-line and all-tokens readers recover byte-identical structured data (unambiguous), AND
    * the input is already in canonical normal form (re-rendering it is a no-op — no flattening).

    A W132 confounded input (flattened body) fails the first two: per-line raises / disagrees.
    """
    n = len(inputs)
    nf = agree = pl_ok = 0
    first = ""
    for inp in inputs:
        try:
            d_all = parse_all_tokens_v1(inp, shape)
        except NormalFormError as ex:
            if not first:
                first = f"all-tokens reader raised: {ex}"
            continue
        # canonical normal form: re-render the all-tokens structure and require equality
        try:
            rerender = render_normal_form_v1(shape, d_all)
            same_nf = (rerender == inp if inp.endswith("\n") else rerender.rstrip("\n") == inp)
        except NormalFormError:
            same_nf = False
        if same_nf:
            nf += 1
        elif not first:
            first = "input is not in canonical normal form (re-render differs)"
        # strict per-line read + dual-parser agreement
        try:
            d_pl = parse_per_line_v1(inp, shape)
            pl_ok += 1
            if d_pl == d_all:
                agree += 1
            elif not first:
                first = "per-line and all-tokens disagree"
        except NormalFormError as ex:
            if not first:
                first = f"per-line reader raised (confound): {ex}"
    is_neutral = bool(n > 0 and nf == n and agree == n and pl_ok == n)
    return ParserNeutralityRecordV1(
        n_cases=n, n_normal_form=nf, n_dual_parser_agree=agree, n_per_line_ok=pl_ok,
        is_parser_neutral=is_neutral, first_failure=first)


__all__ = [
    "PARSER_NEUTRAL_IO_V1_SCHEMA_VERSION",
    "BLOCK_SCALAR_LINE", "BLOCK_ARRAY_LINE", "BLOCK_ROWS", "BLOCK_GRID",
    "NormalFormError", "IoBlockV1", "IoShapeV1",
    "scalar_line", "array_line", "rows", "grid", "io_shape",
    "render_normal_form_v1", "parse_per_line_v1", "parse_all_tokens_v1",
    "ParserNeutralityRecordV1", "parser_neutrality_gate_v1",
]
