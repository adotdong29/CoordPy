"""Exact operators over external memory — Phase 21.

The Phase-19/20 substrate guaranteed lossless STORAGE: every artifact
in the ledger is byte-equal to what was put in. The Phase-21 layer
guarantees lossless COMPUTATION over that storage: deterministic
operators that consume handles and produce results without ever
summarising or paraphrasing artifact bodies.

The operators are intentionally small and composable. There is no
"reasoning" inside an operator — only filtering, extracting, joining,
and reducing. An LLM may participate at the *boundaries* (parsing the
question into a plan; rendering the result back to natural language)
but never inside the reduction itself.

Why bother? Phase 20 surfaced two question classes the bounded
retrieval worker cannot answer well:

  1. **Aggregation queries** ("how many distinct vendors?", "list all
     Sev-1 incidents") need *every* relevant artifact, not the top-k.
     The retrieval-then-LLM pattern silently drops the long tail.
  2. **Cross-reference joins** ("which products use a vendor that also
     appears in X?") need explicit chained lookups, which the worker's
     regex-extraction multi-hop can't compose deeply.

Both classes have crisp closed-form solutions when computation is
*allowed to look at typed metadata or run regex over fetched bodies*
without an LLM in the inner loop. That's what this module provides.

Vocabulary used here (precise but small):

  Operator    : a callable that maps an input "stage" to an output
                stage. Stages flow:  ledger → handles → values → reduction.
  Stage       : the typed payload between operators. We use plain
                Python lists/dicts; an Operator carries its own input
                contract.
  Plan        : an ordered list of Operators. Executing the plan over
                a ledger produces a final value (str | int | list).
  Provenance  : every Operator records which CIDs it touched so the
                final answer carries the same provenance the
                bounded-context worker carries.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .context_ledger import ContextLedger, Handle


# =============================================================================
# Stage / result types
# =============================================================================


@dataclass
class StageHandles:
    """A list of handles, possibly with attached metadata extras."""
    handles: list[Handle]
    note: str = ""


@dataclass
class StageValues:
    """A list of (handle, value) pairs after extraction."""
    pairs: list[tuple[Handle, Any]]
    field_name: str = ""


@dataclass
class StageScalar:
    """A scalar reduction result (count, sum, min, max)."""
    value: Any
    field_name: str = ""
    op: str = ""
    contributing_cids: list[str] = field(default_factory=list)


@dataclass
class StageList:
    """An enumerated list result (list of values, list of strings, etc.)."""
    items: list[Any]
    field_name: str = ""
    contributing_cids: list[str] = field(default_factory=list)


@dataclass
class StageGroups:
    """A grouping result (group key → count or → list)."""
    groups: dict[Any, Any]
    field_name: str = ""
    op: str = ""
    contributing_cids: list[str] = field(default_factory=list)


Stage = StageHandles | StageValues | StageScalar | StageList | StageGroups


@dataclass
class OperatorTrace:
    """Per-operator audit record. Captured during execution."""
    name: str
    in_size: int
    out_size: int
    cids_touched: list[str] = field(default_factory=list)
    notes: dict = field(default_factory=dict)


# =============================================================================
# Operators
# =============================================================================


@dataclass
class Filter:
    """Restrict handles to those whose metadata satisfies `pred`.

    `pred` takes (metadata_dict, body_or_None) and returns bool.
    If `body_required=False` (default), only metadata is consulted —
    no `ledger.fetch` is called and we keep prompt-budget cost zero.
    If `body_required=True`, each handle's full body is fetched
    (still byte-equal — no summarisation) before the predicate runs.

    For the typical aggregation question ("how many Sev-1 incidents?"),
    metadata-only filtering is enough because the corpus indexes typed
    fields. For body-required predicates ("any incident mentioning
    'PCI'"), we pay the fetch cost honestly and surface it via
    `OperatorTrace.cids_touched`.
    """

    name: str
    pred: Callable[[dict, str | None], bool]
    body_required: bool = False

    def execute(self, ledger: ContextLedger,
                inp: StageHandles | None,
                trace: list[OperatorTrace]) -> StageHandles:
        # If no input given, this is a "scan all" op.
        candidates = inp.handles if inp is not None else ledger.all_handles()
        out: list[Handle] = []
        touched: list[str] = []
        for h in candidates:
            md = h.metadata_dict()
            body = ledger.fetch(h) if self.body_required else None
            if self.body_required:
                touched.append(h.cid)
            if self.pred(md, body):
                out.append(h)
        trace.append(OperatorTrace(
            name=f"Filter({self.name})",
            in_size=len(candidates), out_size=len(out),
            cids_touched=touched,
            notes={"body_required": self.body_required},
        ))
        return StageHandles(handles=out, note=self.name)


@dataclass
class Extract:
    """Pull a typed value from each handle.

    Two source modes:
      * `source="metadata"` — read `metadata_dict()[field]`. No fetch,
        no LLM, exact in O(N). This is the workhorse for typed corpora.
      * `source="regex"`    — fetch the body, apply `pattern` (compiled
        re.Pattern), take group `group` of the first match. Exact and
        deterministic; pays the fetch cost.

    Missing values are dropped (the pair list shrinks); we surface the
    drop count in the trace.
    """

    name: str
    field: str
    source: str = "metadata"     # "metadata" | "regex"
    pattern: re.Pattern | None = None
    group: int | str = 0
    coerce: Callable[[Any], Any] | None = None

    def execute(self, ledger: ContextLedger,
                inp: StageHandles | None,
                trace: list[OperatorTrace]) -> StageValues:
        # If used as the first operator in a plan, default to scanning
        # every handle in the ledger.
        if inp is None:
            inp = StageHandles(handles=ledger.all_handles())
        out: list[tuple[Handle, Any]] = []
        touched: list[str] = []
        n_missing = 0
        for h in inp.handles:
            v: Any = None
            if self.source == "metadata":
                v = h.metadata_dict().get(self.field)
            elif self.source == "regex":
                if self.pattern is None:
                    raise ValueError("regex source requires `pattern`")
                body = ledger.fetch(h)
                touched.append(h.cid)
                m = self.pattern.search(body)
                if m is not None:
                    try:
                        v = m.group(self.group)
                    except (IndexError, error if (error := getattr(re, "error", None)) else IndexError):
                        v = None
            else:
                raise ValueError(f"unknown source {self.source!r}")
            if v is None:
                n_missing += 1
                continue
            if self.coerce is not None:
                try:
                    v = self.coerce(v)
                except (ValueError, TypeError):
                    n_missing += 1
                    continue
            out.append((h, v))
        trace.append(OperatorTrace(
            name=f"Extract({self.field}, {self.source})",
            in_size=len(inp.handles), out_size=len(out),
            cids_touched=touched,
            notes={"missing": n_missing},
        ))
        return StageValues(pairs=out, field_name=self.field)


@dataclass
class Count:
    """Reduce values to a scalar count.

    `distinct=True` counts unique values; `distinct=False` counts pairs
    (i.e., the number of handles that produced any value).
    """

    distinct: bool = False

    def execute(self, ledger: ContextLedger,
                inp: StageValues,
                trace: list[OperatorTrace]) -> StageScalar:
        vals = [v for _h, v in inp.pairs]
        n = len(set(vals)) if self.distinct else len(vals)
        cids = [h.cid for h, _v in inp.pairs]
        trace.append(OperatorTrace(
            name=f"Count(distinct={self.distinct})",
            in_size=len(vals), out_size=1,
        ))
        return StageScalar(value=n, field_name=inp.field_name,
                           op="count_distinct" if self.distinct else "count",
                           contributing_cids=cids)


@dataclass
class Sum:
    """Sum extracted numeric values. Coerces non-numeric to 0."""

    def execute(self, ledger: ContextLedger,
                inp: StageValues,
                trace: list[OperatorTrace]) -> StageScalar:
        s = 0.0
        n = 0
        for _h, v in inp.pairs:
            try:
                s += float(v)
                n += 1
            except (ValueError, TypeError):
                continue
        # Return int when possible, else float.
        result = int(s) if s == int(s) else s
        cids = [h.cid for h, _v in inp.pairs]
        trace.append(OperatorTrace(
            name="Sum", in_size=len(inp.pairs), out_size=1,
            notes={"summed": n}))
        return StageScalar(value=result, field_name=inp.field_name,
                           op="sum", contributing_cids=cids)


@dataclass
class MinMax:
    """min or max over numeric values."""
    op: str = "min"     # "min" | "max"

    def execute(self, ledger: ContextLedger,
                inp: StageValues,
                trace: list[OperatorTrace]) -> StageScalar:
        nums: list[float] = []
        for _h, v in inp.pairs:
            try:
                nums.append(float(v))
            except (ValueError, TypeError):
                continue
        if not nums:
            result = None
        else:
            result = min(nums) if self.op == "min" else max(nums)
            if result == int(result):
                result = int(result)
        cids = [h.cid for h, _v in inp.pairs]
        trace.append(OperatorTrace(
            name=self.op.capitalize(), in_size=len(inp.pairs), out_size=1))
        return StageScalar(value=result, field_name=inp.field_name,
                           op=self.op, contributing_cids=cids)


@dataclass
class GroupCount:
    """Group values, return `{value: count_of_handles}` (or top-k by count)."""

    top_k: int | None = None

    def execute(self, ledger: ContextLedger,
                inp: StageValues,
                trace: list[OperatorTrace]) -> StageGroups:
        c = Counter(v for _h, v in inp.pairs)
        if self.top_k is not None:
            c = dict(c.most_common(self.top_k))
        else:
            c = dict(c)
        cids = [h.cid for h, _v in inp.pairs]
        trace.append(OperatorTrace(
            name=f"GroupCount(top_k={self.top_k})",
            in_size=len(inp.pairs), out_size=len(c)))
        return StageGroups(groups=c, field_name=inp.field_name,
                           op="group_count",
                           contributing_cids=cids)


@dataclass
class List_:
    """Enumerate the (handle, value) pairs as a value list (preserve order)."""

    sort: bool = False

    def execute(self, ledger: ContextLedger,
                inp: StageValues,
                trace: list[OperatorTrace]) -> StageList:
        items = [v for _h, v in inp.pairs]
        if self.sort:
            try:
                items = sorted(items)
            except TypeError:
                items = sorted(items, key=str)
        cids = [h.cid for h, _v in inp.pairs]
        trace.append(OperatorTrace(
            name=f"List(sort={self.sort})",
            in_size=len(inp.pairs), out_size=len(items)))
        return StageList(items=items, field_name=inp.field_name,
                         contributing_cids=cids)


@dataclass
class Unnest:
    """Flatten a list/tuple-valued field across handles.

    Used for code metadata where one file has many functions. Given a
    StageValues whose values are tuples (or lists), produces a new
    StageValues where each (handle, item) pair appears once per item.
    The handle is repeated; the item replaces the previous value.

    Example: a StageValues with [(h1, ("foo","bar")), (h2, ("baz",))]
    becomes [(h1, "foo"), (h1, "bar"), (h2, "baz")]. Counting `distinct`
    on the unnested list then gives the number of distinct elements
    across the whole corpus, not just per-file.
    """

    def execute(self, ledger: ContextLedger,
                inp: StageValues,
                trace: list[OperatorTrace]) -> StageValues:
        out: list[tuple[Handle, Any]] = []
        for h, v in inp.pairs:
            if isinstance(v, (list, tuple)):
                for item in v:
                    out.append((h, item))
            elif v is not None:
                out.append((h, v))
        trace.append(OperatorTrace(
            name="Unnest", in_size=len(inp.pairs), out_size=len(out)))
        return StageValues(pairs=out, field_name=inp.field_name)


@dataclass
class ArgExtreme:
    """Return the (handle, value) of the min or max numeric value.

    Like MinMax, but instead of producing a scalar, produces a single
    (handle, value) pair so the caller can read the *handle* whose
    field had the extreme. Used for "which file has the most
    functions" — we need the file path, not just the count."""

    op: str = "max"          # "min" | "max"

    def execute(self, ledger: ContextLedger,
                inp: StageValues,
                trace: list[OperatorTrace]) -> StageValues:
        nums: list[tuple[Handle, float]] = []
        for h, v in inp.pairs:
            try:
                nums.append((h, float(v)))
            except (ValueError, TypeError):
                continue
        if not nums:
            trace.append(OperatorTrace(
                name=f"ArgExtreme({self.op})",
                in_size=len(inp.pairs), out_size=0))
            return StageValues(pairs=[], field_name=inp.field_name)
        keyfn = (lambda x: -x[1]) if self.op == "max" else (lambda x: x[1])
        nums.sort(key=keyfn)
        h_best, v_best = nums[0]
        # Coerce float-ints back to int for cleaner output.
        if v_best == int(v_best):
            v_best = int(v_best)
        trace.append(OperatorTrace(
            name=f"ArgExtreme({self.op})",
            in_size=len(inp.pairs), out_size=1))
        return StageValues(pairs=[(h_best, v_best)], field_name=inp.field_name)


@dataclass
class ProjectMeta:
    """Replace the value in each pair with a different metadata field of
    the same handle. Used after `ArgExtreme` to turn `(handle, count)`
    into `(handle, file_path)` so the rendered answer is the file."""

    field: str

    def execute(self, ledger: ContextLedger,
                inp: StageValues,
                trace: list[OperatorTrace]) -> StageValues:
        out = [(h, h.metadata_dict().get(self.field)) for h, _v in inp.pairs]
        out = [(h, v) for h, v in out if v is not None]
        trace.append(OperatorTrace(
            name=f"ProjectMeta({self.field})",
            in_size=len(inp.pairs), out_size=len(out)))
        return StageValues(pairs=out, field_name=self.field)


@dataclass
class Join:
    """Follow a cross-reference field on each handle to a target handle.

    The left side has a *reference field* whose value is a metadata key
    of the right side (e.g., metadata `related_incident_id` →
    metadata `incident_id` on a different handle). Returns a new
    StageHandles containing the *right-side* handles, in left-side order
    and dedup-preserving.

    This is how chained queries like "for each NordAxis incident, what
    related incident's vendor is named?" become exact: no LLM, no
    paraphrasing, just dictionary lookups."""

    name: str
    left_ref_field: str
    right_match_field: str

    def execute(self, ledger: ContextLedger,
                inp: StageHandles,
                trace: list[OperatorTrace]) -> StageHandles:
        # Build right-side index over the whole ledger by metadata.
        right_index: dict[Any, Handle] = {}
        for h in ledger.all_handles():
            v = h.metadata_dict().get(self.right_match_field)
            if v is not None and v not in right_index:
                right_index[v] = h
        out: list[Handle] = []
        seen: set[str] = set()
        n_unmatched = 0
        for h in inp.handles:
            ref = h.metadata_dict().get(self.left_ref_field)
            if ref is None:
                n_unmatched += 1
                continue
            target = right_index.get(ref)
            if target is None:
                n_unmatched += 1
                continue
            if target.cid in seen:
                continue
            seen.add(target.cid)
            out.append(target)
        trace.append(OperatorTrace(
            name=f"Join({self.left_ref_field}→{self.right_match_field})",
            in_size=len(inp.handles), out_size=len(out),
            notes={"unmatched": n_unmatched},
        ))
        return StageHandles(handles=out, note=self.name)


# =============================================================================
# QueryPlan: a sequence of operators
# =============================================================================


@dataclass
class QueryPlan:
    """An ordered pipeline of operators. Stage types must compose."""

    ops: list[Any]              # Filter, Extract, Count, ...
    answer_template: str | None = None
    description: str = ""

    def execute(self, ledger: ContextLedger) -> tuple[Stage, list[OperatorTrace]]:
        trace: list[OperatorTrace] = []
        stage: Stage | None = None
        for i, op in enumerate(self.ops):
            if i == 0:
                # First op may take None (= scan all) or StageHandles.
                stage = op.execute(ledger, None, trace)   # type: ignore[arg-type]
            else:
                stage = op.execute(ledger, stage, trace)  # type: ignore[arg-type]
        return stage, trace                                # type: ignore[return-value]

    def render(self, final: Stage) -> str:
        """Format the final stage into a string answer.

        - Scalar → "<value>"
        - List   → "<v1>, <v2>, …"
        - Groups → "<k1>: <v1>, <k2>: <v2>, …" sorted by count desc
        - Handles → fingerprints (rare endpoint).

        If `answer_template` is set, it's `.format(value=…, items=…,
        groups=…)`-formatted instead.
        """
        if isinstance(final, StageScalar):
            value = final.value
            text = str(value) if value is not None else "no result"
        elif isinstance(final, StageList):
            text = ", ".join(str(x) for x in final.items)
        elif isinstance(final, StageGroups):
            sorted_pairs = sorted(final.groups.items(),
                                  key=lambda kv: -_as_count(kv[1]))
            text = ", ".join(f"{k}: {v}" for k, v in sorted_pairs)
        elif isinstance(final, StageHandles):
            text = ", ".join(h.fingerprint for h in final.handles)
        elif isinstance(final, StageValues):
            text = ", ".join(str(v) for _h, v in final.pairs)
        else:
            text = repr(final)

        if self.answer_template:
            try:
                return self.answer_template.format(
                    value=getattr(final, "value", None),
                    items=getattr(final, "items", None),
                    groups=getattr(final, "groups", None),
                    text=text,
                )
            except (KeyError, IndexError):
                return text
        return text


def _as_count(v: Any) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return len(v) if hasattr(v, "__len__") else 0
