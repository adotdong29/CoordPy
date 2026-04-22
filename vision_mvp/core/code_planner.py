"""Code-aware query planner — Phase 22 (extended in Phase 23, 24, 25).

Extends the Phase-21 `QueryPlanner` with patterns specific to source-
code corpora ingested via `core.code_index.CodeIndexer`. The base
planner's incident-review patterns are unchanged; this planner tries
the code patterns first, then falls through to the base planner.

Recognised code patterns:

  Phase 22:
    * `code_count_files`             "how many files / how many modules"
    * `code_count_functions_total`   "how many functions in the codebase"
    * `code_count_classes_total`     "how many classes / how many class definitions"
    * `code_count_test_files`        "how many test files"
    * `code_files_importing`         "list files importing X" / "which files import X"
    * `code_functions_returning_none` "list functions that return None"
    * `code_top_file_by_functions`   "which file has the most functions"
    * `code_largest_file`            "what is the largest file" / "biggest file"
    * `code_distinct_imports`        "how many distinct imports / unique modules imported"

  Phase 23 additions (three patterns that lift direct-exact coverage
  on multiple real Python corpora without bloating the table):
    * `code_count_methods_total`      "how many methods"
    * `code_count_files_with_docstrings` "how many files have docstrings"
    * `code_most_imported_module`     "which module is imported most often"

  Phase 24 additions — **conservative static-semantic patterns**. These
  read parallel semantic tuples populated by `core.code_semantics`:
    * `code_count_may_raise`          "how many functions may raise"
    * `code_list_may_raise`           "list functions that may raise"
    * `code_count_is_recursive`       "how many functions are recursive"
    * `code_list_is_recursive`        "list recursive functions"
    * `code_count_may_write_global`   "how many functions may write global state"
    * `code_list_may_write_global`    "list functions that may write module globals"
    * `code_count_calls_subprocess`   "how many functions call subprocess"
    * `code_list_calls_subprocess`    "list functions that invoke subprocess"
    * `code_count_calls_filesystem`   "how many functions call the filesystem"
    * `code_list_calls_filesystem`    "list functions that touch the filesystem"
    * `code_count_calls_network`      "how many functions make network calls"
    * `code_list_calls_network`       "list functions that make network calls"
    * `code_count_calls_external_io`  "how many functions have external side effects"
    * `code_list_calls_external_io`   "list functions with external side effects"

  Phase 25 additions — **interprocedural conservative-semantic patterns**.
  These read transitive-boolean tuples populated by `core.code_interproc`
  (via the post-pass in `core.code_index.CodeIndexer.index_into`):
    * `code_count_trans_may_raise`          "how many functions may transitively raise"
    * `code_list_trans_may_raise`           "list functions that may transitively raise"
    * `code_count_trans_calls_subprocess`   "how many functions transitively invoke subprocess"
    * `code_list_trans_calls_subprocess`    "list functions that transitively invoke subprocess"
    * `code_count_trans_calls_filesystem`   "how many functions transitively touch the filesystem"
    * `code_list_trans_calls_filesystem`    "list functions that transitively touch the filesystem"
    * `code_count_trans_calls_network`      "how many functions transitively make network calls"
    * `code_list_trans_calls_network`       "list functions that transitively make network calls"
    * `code_count_trans_calls_external_io`  "how many functions have transitive external side effects"
    * `code_count_trans_may_write_global`   "how many functions transitively mutate globals"
    * `code_list_trans_may_write_global`    "list functions with transitive global writes"
    * `code_count_participates_in_cycle`    "how many functions participate in a recursion cycle"
    * `code_list_participates_in_cycle`     "list functions in a mutual-recursion cycle"
    * `code_count_has_unresolved_callees`   "how many functions call unresolved helpers"
    * `code_list_has_unresolved_callees`    "list functions with unresolved callees"

All patterns are pure regex / keyword. Zero LLM in the planning step.
The result of `plan(question)` is the same `PlanResult` shape used by
the Phase-21 base planner — the caller can dispatch identically.
"""

from __future__ import annotations

import re

from .exact_ops import (
    ArgExtreme, Count, Extract, Filter, GroupCount, Join, List_, MinMax,
    ProjectMeta, QueryPlan, Sum, Unnest,
)
from .query_planner import PlanResult, QueryPlanner


class CodeQueryPlanner(QueryPlanner):
    """Try code patterns first, then the base planner's incident patterns."""

    # ------- code patterns ---------------------------------------------

    def _try_code_count_files(self, q: str) -> PlanResult | None:
        if not re.search(r"\bhow\s+many\b", q, re.IGNORECASE):
            return None
        if not re.search(r"\b(files|modules|python\s+files)\b",
                         q, re.IGNORECASE):
            return None
        if re.search(r"\b(distinct|unique)\b", q, re.IGNORECASE):
            return None    # let _try_code_distinct_imports / count_distinct
        if re.search(r"\b(functions|classes|tests|imports)\b",
                     q, re.IGNORECASE):
            return None    # other patterns own these
        plan = QueryPlan(
            ops=[Extract(name="paths", field="file_path", source="metadata"),
                 Count(distinct=False)],
            description="count files",
        )
        return PlanResult(plan=plan, pattern="code_count_files",
                          matched_groups={},
                          rationale="how many files/modules")

    def _try_code_count_functions_total(self, q: str) -> PlanResult | None:
        if not re.search(r"\bhow\s+many\s+functions\b",
                         q, re.IGNORECASE):
            return None
        if re.search(r"\b(distinct|unique|return|test)\b",
                     q, re.IGNORECASE):
            return None
        # Sum over per-file n_functions.
        plan = QueryPlan(
            ops=[Extract(name="n_fn", field="n_functions",
                         source="metadata", coerce=int),
                 Sum()],
            description="sum n_functions across all files",
        )
        return PlanResult(plan=plan, pattern="code_count_functions_total",
                          matched_groups={},
                          rationale="how many functions (total)")

    def _try_code_count_classes_total(self, q: str) -> PlanResult | None:
        if not re.search(r"\bhow\s+many\s+(classes|class\s+definitions)\b",
                         q, re.IGNORECASE):
            return None
        plan = QueryPlan(
            ops=[Extract(name="n_cls", field="n_classes",
                         source="metadata", coerce=int),
                 Sum()],
            description="sum n_classes across all files",
        )
        return PlanResult(plan=plan, pattern="code_count_classes_total",
                          matched_groups={},
                          rationale="how many classes")

    def _try_code_count_test_files(self, q: str) -> PlanResult | None:
        if not re.search(r"\bhow\s+many\s+test\s+files\b",
                         q, re.IGNORECASE):
            return None
        plan = QueryPlan(
            ops=[Filter(name="is_test_file",
                        pred=lambda md, _b: bool(md.get("is_test_file"))),
                 Extract(name="paths", field="file_path", source="metadata"),
                 Count(distinct=False)],
            description="count files where is_test_file=True",
        )
        return PlanResult(plan=plan, pattern="code_count_test_files",
                          matched_groups={},
                          rationale="how many test files")

    def _try_code_distinct_imports(self, q: str) -> PlanResult | None:
        if not re.search(r"\bhow\s+many\s+(?:distinct|unique)\b",
                         q, re.IGNORECASE):
            return None
        if not re.search(r"\b(imports|modules\s+imported|imported|modules)\b",
                         q, re.IGNORECASE):
            return None
        plan = QueryPlan(
            ops=[Extract(name="imports", field="imports", source="metadata"),
                 Unnest(),
                 Count(distinct=True)],
            description="count distinct imports across the corpus",
        )
        return PlanResult(plan=plan, pattern="code_distinct_imports",
                          matched_groups={},
                          rationale="how many distinct imports")

    def _try_code_files_importing(self, q: str) -> PlanResult | None:
        # "list files importing numpy", "which files import numpy",
        # "files that import numpy"
        m = re.search(
            r"\b(?:list|which|find|show|enumerate|name)\s+(?:files|modules|all\s+files)\b"
            r".*?\b(?:that\s+import|importing|import)\b\s+([A-Za-z_][A-Za-z0-9_.]*)",
            q, re.IGNORECASE)
        if not m:
            # Looser fallback: "files importing X"
            m = re.search(
                r"\b(?:files|modules)\s+importing\s+([A-Za-z_][A-Za-z0-9_.]*)",
                q, re.IGNORECASE)
        if not m:
            return None
        target = m.group(1)
        # Strip trailing period etc.
        target = target.rstrip(".,;:")

        # Match if `target` appears as either an exact import name or
        # as the dotted prefix of an `ImportFrom`-style entry.
        def pred(md: dict, _body) -> bool:
            imps = md.get("imports", ())
            for imp in imps:
                if imp == target or imp.startswith(target + "."):
                    return True
            return False
        plan = QueryPlan(
            ops=[Filter(name=f"imports {target}", pred=pred),
                 Extract(name="paths", field="file_path", source="metadata"),
                 List_(sort=True)],
            description=f"list file_path where 'imports' contains {target}",
        )
        return PlanResult(plan=plan, pattern="code_files_importing",
                          matched_groups={"target": target},
                          rationale=f"list files importing {target}")

    def _try_code_functions_returning_none(self, q: str) -> PlanResult | None:
        # "list functions that return None", "which functions return None",
        # "list functions returning None"
        if not re.search(
                r"\b(?:functions|methods)\b.*?\b(?:return|returning)\b.*?\bNone\b",
                q, re.IGNORECASE):
            return None
        # Use a small inline operator that walks the parallel
        # (function_names, function_returns_none) tuples on each file's
        # metadata and emits "module.name" rows for the qualifying
        # functions. The operator API supports plain duck-typed objects
        # that implement `execute(ledger, inp, trace)` so the inline
        # class works without subclassing.
        from .exact_ops import OperatorTrace, StageValues

        class _ExtractQualifiedReturnNone:
            def execute(self, ledger, inp, trace):
                out: list = []
                for h in inp.handles:
                    md = h.metadata_dict()
                    names = md.get("function_names", ())
                    flags = md.get("function_returns_none", ())
                    module = md.get("module_name", "?")
                    for n, ret in zip(names, flags):
                        if ret:
                            out.append((h, f"{module}.{n}"))
                trace.append(OperatorTrace(
                    name="ExtractQualifiedReturnsNone",
                    in_size=len(inp.handles), out_size=len(out)))
                return StageValues(pairs=out, field_name="qualified_name")

        plan = QueryPlan(
            ops=[Filter(name="any returns_none",
                        pred=lambda md, _b: any(md.get("function_returns_none", ()))),
                 _ExtractQualifiedReturnNone(),
                 List_(sort=True)],
            description="list module.func for functions provably returning None",
        )
        return PlanResult(plan=plan, pattern="code_functions_returning_none",
                          matched_groups={},
                          rationale="list functions returning None")

    def _try_code_top_file_by_functions(self, q: str) -> PlanResult | None:
        # "which file has the most functions", "file with the most functions"
        if not re.search(
                r"\b(which|what)\b.*?\bfile\b.*?\b(most|fewest|least|smallest|largest)\b.*?\bfunctions\b",
                q, re.IGNORECASE) and not re.search(
                r"\bfile\s+with\s+the\s+most\s+functions\b",
                q, re.IGNORECASE):
            return None
        most = re.search(r"\b(most|largest)\b", q, re.IGNORECASE) is not None
        plan = QueryPlan(
            ops=[Extract(name="n_fn", field="n_functions",
                         source="metadata", coerce=int),
                 ArgExtreme(op="max" if most else "min"),
                 ProjectMeta(field="file_path"),
                 List_()],
            description="argmax n_functions → file_path",
        )
        return PlanResult(plan=plan, pattern="code_top_file_by_functions",
                          matched_groups={"most": most},
                          rationale="which file has the most functions")

    def _try_code_largest_file(self, q: str) -> PlanResult | None:
        # "what is the largest file by lines", "biggest file"
        if not re.search(r"\b(largest|biggest|smallest|longest|shortest)\b",
                         q, re.IGNORECASE):
            return None
        if not re.search(r"\bfile\b", q, re.IGNORECASE):
            return None
        if re.search(r"\bfunctions\b", q, re.IGNORECASE):
            return None    # let _try_code_top_file_by_functions take it
        big = re.search(r"\b(largest|biggest|longest)\b",
                        q, re.IGNORECASE) is not None
        plan = QueryPlan(
            ops=[Extract(name="lines", field="line_count",
                         source="metadata", coerce=int),
                 ArgExtreme(op="max" if big else "min"),
                 ProjectMeta(field="file_path"),
                 List_()],
            description="argmax line_count → file_path",
        )
        return PlanResult(plan=plan, pattern="code_largest_file",
                          matched_groups={"largest": big},
                          rationale="largest/smallest file by line_count")

    # ------- Phase-23 additions ---------------------------------------

    def _try_code_count_methods_total(self, q: str) -> PlanResult | None:
        # "how many methods are defined", "how many methods in the corpus"
        if not re.search(r"\bhow\s+many\s+methods\b", q, re.IGNORECASE):
            return None
        # Exclude "how many methods return None" etc.
        if re.search(r"\b(return|returning|async)\b", q, re.IGNORECASE):
            return None
        plan = QueryPlan(
            ops=[Extract(name="n_meth", field="n_methods",
                         source="metadata", coerce=int),
                 Sum()],
            description="sum n_methods across all files",
        )
        return PlanResult(plan=plan, pattern="code_count_methods_total",
                          matched_groups={},
                          rationale="how many methods (total)")

    def _try_code_count_files_with_docstrings(self, q: str) -> PlanResult | None:
        # "how many files have docstrings", "how many modules have docstrings"
        # Also accepts "with a docstring" / "with docstrings".
        if not re.search(r"\bhow\s+many\s+(files|modules)\b",
                         q, re.IGNORECASE):
            return None
        if not re.search(r"\b(have|with|contain|carry)\b\s+"
                         r"(?:a\s+|the\s+)?docstrings?\b",
                         q, re.IGNORECASE):
            return None
        plan = QueryPlan(
            ops=[Filter(name="has_docstring",
                        pred=lambda md, _b: bool(md.get("has_docstring"))),
                 Extract(name="paths", field="file_path", source="metadata"),
                 Count(distinct=False)],
            description="count files where has_docstring=True",
        )
        return PlanResult(plan=plan, pattern="code_count_files_with_docstrings",
                          matched_groups={},
                          rationale="how many files have docstrings")

    # ------- Phase-24 semantic patterns -------------------------------

    # Shared matcher data: canonical predicate → (metadata n_fn field,
    # per-function bool field, natural-language triggers, kind tag).
    # The trigger list is parsed by `_match_semantic_topic`; each
    # trigger is a regex that must match somewhere in the query.
    _SEM_TOPICS = {
        "may_raise": {
            "field_bool": "function_may_raise",
            "field_count": "n_functions_may_raise",
            "triggers": (
                r"\bmay\s+raise\b",
                r"\braise\s+(?:an?\s+)?(?:exception|error)\b",
                r"\bcan\s+raise\b",
                r"\braise\s+exceptions?\b",
                r"\bmight\s+raise\b",
                r"\bthrows?\s+(?:an?\s+)?(?:exception|error)\b",
            ),
            "noun": "may_raise",
        },
        "is_recursive": {
            "field_bool": "function_is_recursive",
            "field_count": "n_functions_is_recursive",
            "triggers": (
                r"\brecursive\b",
                r"\brecurs(?:ion|es)\b",
                r"\bcall\s+themselves\b",
                r"\bself[-\s]recursive\b",
            ),
            "noun": "recursive",
        },
        "may_write_global": {
            "field_bool": "function_may_write_global",
            "field_count": "n_functions_may_write_global",
            "triggers": (
                r"\bwrite\s+(?:to\s+)?(?:a\s+)?global(?:s|\s+state)?\b",
                r"\bmutate\s+(?:a\s+)?(?:module[-\s])?globals?\b",
                r"\bmutate\s+(?:module\s+)?state\b",
                r"\bmodify\s+(?:a\s+)?globals?\b",
                r"\bassign\s+(?:to\s+)?(?:a\s+)?globals?\b",
                r"\bwrite\s+module\s+state\b",
                r"\bwrite\s+(?:a\s+)?module[-\s]level\b",
            ),
            "noun": "may_write_global",
        },
        "calls_subprocess": {
            "field_bool": "function_calls_subprocess",
            "field_count": "n_functions_calls_subprocess",
            "triggers": (
                r"\bsubprocess\b",
                r"\b(?:invoke|call|run)\s+(?:a\s+)?(?:shell|process)\b",
                r"\bshell\s+command\b",
                r"\bshell\s+out\b",
                r"\bos\.system\b",
                r"\bspawn\s+(?:a\s+)?process\b",
                r"\bchild\s+process(?:es)?\b",
            ),
            "noun": "calls_subprocess",
        },
        "calls_filesystem": {
            "field_bool": "function_calls_filesystem",
            "field_count": "n_functions_calls_filesystem",
            "triggers": (
                r"\bfile[-\s]?system\b",
                r"\btouch\s+(?:the\s+)?filesystem\b",
                r"\bread\s+(?:from\s+)?files?\b",
                r"\bwrite\s+(?:to\s+)?(?:the\s+)?(?:disk|files?|file\s+system)\b",
                r"\bopen\s+(?:a\s+)?file\b",
                r"\b(?:read|write)\s+disk\b",
                r"\bi/?o\s+on\s+disk\b",
                r"\b(?:path|file|disk)\s+i/?o\b",
            ),
            "noun": "calls_filesystem",
        },
        "calls_network": {
            "field_bool": "function_calls_network",
            "field_count": "n_functions_calls_network",
            "triggers": (
                r"\bnetwork\b",
                r"\bnetwork\s+calls?\b",
                r"\b(?:http|https)\s+(?:call|request)s?\b",
                r"\bsocket\b",
                r"\burl\s+open\b",
                r"\bmake\s+(?:an?\s+)?http\b",
                r"\bweb\s+(?:calls?|requests?)\b",
                r"\b(?:requests|httpx|urllib)\b",
            ),
            "noun": "calls_network",
        },
        "calls_external_io": {
            "field_bool": None,       # derived from union of three flags
            "field_count": "n_functions_calls_external_io",
            "triggers": (
                r"\bside[-\s]effects?\s+outside\s+local\s+scope\b",
                r"\bexternal\s+side[-\s]effects?\b",
                r"\bexternal\s+i/?o\b",
                r"\boutside\s+(?:scope|local\s+scope)\b",
                r"\bhave\s+side[-\s]effects?\b",
            ),
            "noun": "calls_external_io",
        },
    }

    def _match_semantic_topic(self, q: str) -> str | None:
        for topic, spec in self._SEM_TOPICS.items():
            for trig in spec["triggers"]:
                if re.search(trig, q, re.IGNORECASE):
                    return topic
        return None

    # ------- Phase-25 interprocedural topic table --------------------
    #
    # Each entry follows the same shape as `_SEM_TOPICS`, but the
    # triggers require one of a small set of *transitivity markers*
    # ("transitive/transitively/indirectly/through a helper/chain/
    # reach/reaches/reachable/via"). This keeps Phase-24 (intra-only)
    # phrasings routing to the Phase-24 patterns unchanged.
    #
    # `participates_in_cycle` is special: the phrasing "mutual
    # recursion" / "recursion cycle" / "call cycle" is unambiguously
    # interprocedural without any transitive marker, so its triggers
    # fire directly. The Phase-24 `is_recursive` pattern still fires
    # for "recursive" alone — the new pattern matches only when the
    # query explicitly asks about *cycle* or *mutual* recursion.

    _INTERPROC_TOPICS = {
        "trans_may_raise": {
            "field_bool": "function_trans_may_raise",
            "field_count": "n_functions_trans_may_raise",
            "triggers": (
                r"\btransitively\s+raise\b",
                r"\bmay\s+(?:transitively|indirectly)\s+raise\b",
                r"\b(?:indirectly|transitively)\s+(?:raise|throw)s?\b",
                r"\braise\s+(?:an?\s+)?(?:exception|error)\s+(?:transitively|through\s+(?:a\s+)?helpers?)\b",
                r"\braise\s+through\s+(?:a\s+)?helpers?\b",
                r"\b(?:chain|reach)(?:es|ing)?\s+(?:a\s+)?raise\b",
            ),
            "noun": "trans_may_raise",
        },
        "trans_may_write_global": {
            "field_bool": "function_trans_may_write_global",
            "field_count": "n_functions_trans_may_write_global",
            "triggers": (
                r"\btransitively\s+(?:write|mutate)\s+(?:to\s+)?(?:a\s+)?(?:module[-\s])?globals?\b",
                r"\btransitively\s+(?:write|mutate)\s+(?:module\s+)?state\b",
                r"\b(?:indirectly|transitively)\s+modify\s+(?:a\s+)?(?:module[-\s])?globals?\b",
                r"\bmutate\s+(?:module\s+)?globals?\s+through\s+(?:a\s+)?helpers?\b",
                r"\bwrite\s+(?:module\s+)?globals?\s+through\s+(?:a\s+)?helpers?\b",
                r"\b(?:indirect|transitive)\s+(?:module[-\s])?global\s+(?:writes?|mutations?)\b",
            ),
            "noun": "trans_may_write_global",
        },
        "trans_calls_subprocess": {
            "field_bool": "function_trans_calls_subprocess",
            "field_count": "n_functions_trans_calls_subprocess",
            "triggers": (
                r"\btransitively\s+(?:invoke|call|run)\s+subprocess\b",
                r"\btransitively\s+(?:invoke|call|run)\s+(?:a\s+)?(?:shell|process)\b",
                r"\btransitively\s+shell\s+out\b",
                r"\b(?:indirectly|transitively)\s+spawn\s+(?:a\s+)?process\b",
                r"\breach(?:es|ing)?\s+subprocess\b",
                r"\bsubprocess\s+(?:through|via)\s+(?:a\s+)?helpers?\b",
            ),
            "noun": "trans_calls_subprocess",
        },
        "trans_calls_filesystem": {
            "field_bool": "function_trans_calls_filesystem",
            "field_count": "n_functions_trans_calls_filesystem",
            "triggers": (
                r"\btransitively\s+(?:touch|read|write|access)\s+(?:the\s+)?(?:file[-\s]?system|disk)\b",
                r"\b(?:indirectly|transitively)\s+open\s+(?:a\s+)?file\b",
                r"\bfile[-\s]?system\s+(?:through|via)\s+(?:a\s+)?helpers?\b",
                r"\breach(?:es|ing)?\s+(?:the\s+)?(?:file[-\s]?system|disk)\b",
                r"\bdisk\s+(?:i/?o)\s+(?:through|via)\s+(?:a\s+)?helpers?\b",
            ),
            "noun": "trans_calls_filesystem",
        },
        "trans_calls_network": {
            "field_bool": "function_trans_calls_network",
            "field_count": "n_functions_trans_calls_network",
            "triggers": (
                r"\btransitively\s+(?:make|send|perform|issue)\s+(?:an?\s+)?(?:network\s+call|http\s+(?:call|request)|web\s+(?:call|request))s?\b",
                r"\btransitively\s+(?:call|invoke)\s+(?:the\s+)?network\b",
                r"\b(?:indirectly|transitively)\s+make\s+(?:an?\s+)?http\b",
                r"\b(?:indirectly|transitively)\s+(?:call|open)\s+(?:a\s+)?socket\b",
                r"\breach(?:es|ing)?\s+(?:the\s+)?network\b",
                r"\bnetwork\s+(?:through|via)\s+(?:a\s+)?helpers?\b",
            ),
            "noun": "trans_calls_network",
        },
        "trans_calls_external_io": {
            "field_bool": None,          # union
            "field_count": "n_functions_trans_calls_external_io",
            "triggers": (
                r"\btransitively\s+have\s+side[-\s]effects?\b",
                r"\btransitively\s+have\s+external\s+side[-\s]effects?\b",
                r"\btransitive\s+external\s+(?:i/?o|side[-\s]effects?)\b",
                r"\bside[-\s]effects?\s+through\s+(?:a\s+)?helpers?\b",
                r"\breach(?:es|ing)?\s+external\s+i/?o\b",
            ),
            "noun": "trans_calls_external_io",
        },
        "participates_in_cycle": {
            "field_bool": "function_participates_in_cycle",
            "field_count": "n_functions_participates_in_cycle",
            "triggers": (
                r"\bparticipate\s+in\s+(?:a\s+)?(?:recursion|call)\s+cycle\b",
                r"\b(?:recursion|call)\s+cycle\b",
                r"\bmutual(?:ly)?[-\s]recursive\b",
                r"\bmutual\s+recursion\b",
                r"\bin\s+(?:a\s+)?(?:recursion|call)\s+cycle\b",
                r"\bcycles?\s+of\s+(?:function\s+)?calls?\b",
            ),
            "noun": "participates_in_cycle",
        },
        "has_unresolved_callees": {
            "field_bool": "function_has_unresolved_callees",
            "field_count": "n_functions_has_unresolved_callees",
            "triggers": (
                r"\bunresolved\s+(?:call(?:ee)?s?|helpers?)\b",
                r"\bcall\s+(?:into\s+)?(?:an?\s+)?(?:unresolved|opaque)\s+helpers?\b",
                r"\b(?:statically[-\s])?unresolvable\s+(?:calls?|helpers?)\b",
                r"\bcall\s+(?:a\s+)?helper\s+(?:outside|not\s+in)\s+the\s+corpus\b",
            ),
            "noun": "has_unresolved_callees",
        },
    }

    def _match_interproc_topic(self, q: str) -> str | None:
        for topic, spec in self._INTERPROC_TOPICS.items():
            for trig in spec["triggers"]:
                if re.search(trig, q, re.IGNORECASE):
                    return topic
        return None

    def _try_code_interproc_count(self, q: str) -> PlanResult | None:
        if not self._is_count_question(q):
            return None
        if not re.search(r"\b(functions?|methods?|wrappers?)\b",
                         q, re.IGNORECASE):
            return None
        topic = self._match_interproc_topic(q)
        if topic is None:
            return None
        spec = self._INTERPROC_TOPICS[topic]
        plan = QueryPlan(
            ops=[Extract(name="n_ip", field=spec["field_count"],
                         source="metadata", coerce=int),
                 Sum()],
            description=f"sum {spec['field_count']} across all files",
        )
        return PlanResult(plan=plan,
                          pattern=f"code_count_{topic}",
                          matched_groups={"topic": topic},
                          rationale=f"how many functions {topic}")

    def _try_code_interproc_list(self, q: str) -> PlanResult | None:
        if not re.search(
                r"\b(list|which|enumerate|show|identify|find|name)\b.*?"
                r"\b(functions?|methods?|wrappers?)\b",
                q, re.IGNORECASE):
            return None
        topic = self._match_interproc_topic(q)
        if topic is None:
            return None
        spec = self._INTERPROC_TOPICS[topic]
        bool_field = spec["field_bool"]
        from .exact_ops import OperatorTrace, StageValues

        class _InterprocListExtractor:
            def __init__(self_inner, topic_inner: str,
                         bool_field_inner: str | None):
                self_inner.topic = topic_inner
                self_inner.bool_field = bool_field_inner

            def execute(self_inner, ledger, inp, trace):
                out: list = []
                for h in inp.handles:
                    md = h.metadata_dict()
                    names = md.get("semantic_function_names", ())
                    module = md.get("module_name", "?")
                    if self_inner.bool_field is not None:
                        flags = md.get(self_inner.bool_field, ())
                    else:
                        sp = md.get("function_trans_calls_subprocess", ())
                        fs = md.get("function_trans_calls_filesystem", ())
                        nw = md.get("function_trans_calls_network", ())
                        flags = tuple(
                            a or b or c for a, b, c in zip(sp, fs, nw)
                        )
                    for n, flag in zip(names, flags):
                        if flag:
                            out.append((h, f"{module}.{n}"))
                trace.append(OperatorTrace(
                    name=f"ExtractInterproc({self_inner.topic})",
                    in_size=len(inp.handles), out_size=len(out)))
                return StageValues(pairs=out,
                                   field_name=f"interproc_{self_inner.topic}")

        plan = QueryPlan(
            ops=[Filter(
                    name=f"any {topic}",
                    pred=_make_any_interproc_pred(topic, bool_field)),
                 _InterprocListExtractor(topic, bool_field),
                 List_(sort=True)],
            description=f"list qualified names where {topic}=True",
        )
        return PlanResult(plan=plan,
                          pattern=f"code_list_{topic}",
                          matched_groups={"topic": topic},
                          rationale=f"list functions {topic}")

    @staticmethod
    def _is_count_question(q: str) -> bool:
        return re.search(r"\bhow\s+many\b", q, re.IGNORECASE) is not None

    @staticmethod
    def _is_list_question(q: str) -> bool:
        return re.search(
            r"\b(list|which|enumerate|show|identify|find|name)\b.*?\b(functions?|methods?)\b",
            q, re.IGNORECASE,
        ) is not None

    def _try_code_semantic_count(self, q: str) -> PlanResult | None:
        if not self._is_count_question(q):
            return None
        # Require "functions" or "methods" as the counted noun.
        if not re.search(r"\b(functions?|methods?)\b", q, re.IGNORECASE):
            return None
        topic = self._match_semantic_topic(q)
        if topic is None:
            return None
        spec = self._SEM_TOPICS[topic]
        # Every semantic topic has a file-level aggregate count; Sum
        # over it gives the corpus-wide total exactly.
        plan = QueryPlan(
            ops=[Extract(name="n_sem", field=spec["field_count"],
                         source="metadata", coerce=int),
                 Sum()],
            description=f"sum {spec['field_count']} across all files",
        )
        return PlanResult(plan=plan, pattern=f"code_count_{topic}",
                          matched_groups={"topic": topic},
                          rationale=f"how many functions {topic}")

    def _try_code_semantic_list(self, q: str) -> PlanResult | None:
        if not self._is_list_question(q):
            return None
        topic = self._match_semantic_topic(q)
        if topic is None:
            return None
        spec = self._SEM_TOPICS[topic]
        bool_field = spec["field_bool"]
        # Shared walker: walks parallel `semantic_function_names` +
        # chosen boolean tuple and emits qualified-name strings for True
        # entries. For `calls_external_io` we combine three tuples.
        from .exact_ops import OperatorTrace, StageValues

        class _SemanticListExtractor:
            def __init__(self_inner, topic_inner: str,
                         bool_field_inner: str | None):
                self_inner.topic = topic_inner
                self_inner.bool_field = bool_field_inner

            def execute(self_inner, ledger, inp, trace):
                out: list = []
                for h in inp.handles:
                    md = h.metadata_dict()
                    names = md.get("semantic_function_names", ())
                    module = md.get("module_name", "?")
                    if self_inner.bool_field is not None:
                        flags = md.get(self_inner.bool_field, ())
                    else:
                        # calls_external_io: union of three tuples
                        sp = md.get("function_calls_subprocess", ())
                        fs = md.get("function_calls_filesystem", ())
                        nw = md.get("function_calls_network", ())
                        flags = tuple(
                            a or b or c for a, b, c in zip(sp, fs, nw)
                        )
                    for n, flag in zip(names, flags):
                        if flag:
                            out.append((h, f"{module}.{n}"))
                trace.append(OperatorTrace(
                    name=f"ExtractSemantic({self_inner.topic})",
                    in_size=len(inp.handles), out_size=len(out)))
                return StageValues(pairs=out,
                                   field_name=f"semantic_{self_inner.topic}")

        plan = QueryPlan(
            ops=[Filter(
                    name=f"any {topic}",
                    pred=_make_any_semantic_pred(topic, bool_field)),
                 _SemanticListExtractor(topic, bool_field),
                 List_(sort=True)],
            description=f"list qualified names where {topic}=True",
        )
        return PlanResult(plan=plan, pattern=f"code_list_{topic}",
                          matched_groups={"topic": topic},
                          rationale=f"list functions {topic}")

    # ------- continuation of Phase 23 patterns ------------------------

    def _try_code_most_imported_module(self, q: str) -> PlanResult | None:
        # "which module is imported most often",
        # "what is the most imported module",
        # "which import appears most often".
        has_subject = re.search(
            r"\b(module|import|package)s?\b", q, re.IGNORECASE) is not None
        if not has_subject:
            return None
        has_extremum = re.search(
            r"\b(most|often|frequently|popular|common)\b", q,
            re.IGNORECASE) is not None
        if not has_extremum:
            return None
        # Avoid colliding with `code_files_importing`: that pattern
        # requires a literal module name argument and one of the verbs
        # "import / importing / that import".
        if re.search(
                r"\b(?:importing|that\s+import|files\s+importing)\b",
                q, re.IGNORECASE):
            return None
        # Use the imports tuple + Unnest + GroupCount(top_k=1).
        plan = QueryPlan(
            ops=[Extract(name="imports", field="imports", source="metadata"),
                 Unnest(),
                 GroupCount(top_k=1)],
            description="most frequently imported module",
        )
        return PlanResult(plan=plan, pattern="code_most_imported_module",
                          matched_groups={},
                          rationale="most imported module")

    # ------- override dispatch ----------------------------------------

    def plan(self, question: str) -> PlanResult:
        for matcher in (
            # --- Phase 22 (most specific first) ---
            self._try_code_files_importing,
            self._try_code_functions_returning_none,
            self._try_code_top_file_by_functions,
            self._try_code_largest_file,
            self._try_code_count_test_files,
            self._try_code_distinct_imports,
            # --- Phase 25 interprocedural patterns — tried BEFORE
            #     Phase 24 so that "how many functions transitively
            #     call subprocess" routes to the trans predicate, not
            #     the intra one. The interproc trigger regexes require
            #     a transitivity marker, so Phase-24 phrasings
            #     ("how many functions call subprocess") still route
            #     to the Phase-24 patterns unchanged. ---
            self._try_code_interproc_list,
            self._try_code_interproc_count,
            # --- Phase 24 semantic patterns BEFORE the generic
            #     "count_functions_total" matcher so that a question
            #     like "how many functions are recursive?" routes to
            #     the recursive predicate rather than matching the
            #     general functions-count. Order: list-first (more
            #     specific), then count. ---
            self._try_code_semantic_list,
            self._try_code_semantic_count,
            self._try_code_count_functions_total,
            self._try_code_count_classes_total,
            # --- Phase 23 additions ---
            self._try_code_count_methods_total,
            self._try_code_count_files_with_docstrings,
            self._try_code_most_imported_module,
            # --- catch-all for "how many (python )?files" ---
            self._try_code_count_files,
        ):
            res = matcher(question)
            if res is not None:
                return res
        # Fall through to the incident-review base planner.
        return super().plan(question)


def _make_any_interproc_pred(topic: str, bool_field: str | None):
    """Filter predicate for the Phase-25 list patterns. Same shape as
    `_make_any_semantic_pred` but reads the trans-tuples."""
    if bool_field is None:
        def _pred(md, _body) -> bool:
            sp = md.get("function_trans_calls_subprocess", ())
            fs = md.get("function_trans_calls_filesystem", ())
            nw = md.get("function_trans_calls_network", ())
            return any(a or b or c for a, b, c in zip(sp, fs, nw))
        return _pred
    else:
        def _pred(md, _body) -> bool:
            return any(md.get(bool_field, ()))
        return _pred


def _make_any_semantic_pred(topic: str, bool_field: str | None):
    """Build a `Filter.pred` that returns True when the handle's
    metadata has at least one True entry in the chosen boolean tuple.

    For `calls_external_io` (`bool_field is None`), we OR the three
    underlying tuples. Used by `_try_code_semantic_list` to prune
    handles that contribute no qualifying function — purely a speed
    / noise reduction, correctness is preserved either way."""
    if bool_field is None:
        def _pred(md, _body) -> bool:
            sp = md.get("function_calls_subprocess", ())
            fs = md.get("function_calls_filesystem", ())
            nw = md.get("function_calls_network", ())
            return any(a or b or c for a, b, c in zip(sp, fs, nw))
        return _pred
    else:
        def _pred(md, _body) -> bool:
            return any(md.get(bool_field, ()))
        return _pred
