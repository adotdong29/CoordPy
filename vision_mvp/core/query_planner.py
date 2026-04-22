"""Query planner — natural-language → exact-operator pipeline.

Phase 21 layer that sits above retrieval. Given a question, attempts
to recognise it as one of a small set of canonical query patterns and
emit a deterministic `QueryPlan` over `core/exact_ops` operators. If no
pattern matches, the planner returns `None` and the caller is expected
to fall through to the standard `BoundedRetrievalWorker` (Phase
19/20).

Crucially, the parser is **pure regex / keyword** — no LLM in the
planning step. This preserves the "exact computation" property:
either the query is recognised and answered deterministically
without an LLM, or it isn't and the system uses the existing
LLM-backed retrieval path. Failure to parse is honest, not a
fallback to summarisation.

Recognised patterns (current Phase-21 vocabulary):

  * `count_distinct_field`   "how many distinct {field} are mentioned?"
  * `count_filter`            "how many {kind} had {filter}?"
  * `list_filter`             "list all {kind} in {city}"
  * `top_group`               "which {field} appears most/least often?"
  * `sum_field`               "what is the total {field} across {filter}?"
  * `min_max_field`           "what is the largest/smallest {field}?"
  * `join_via_ref`            "for the related incident of {ref}, what is its {field}?"

Each pattern compiles to one or two operators. The planner is
deliberately conservative — adding a pattern is a controlled change
to a regex table, not training a parser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .exact_ops import (
    Count, Extract, Filter, GroupCount, Join, List_, MinMax, QueryPlan, Sum,
)


# =============================================================================
# Field vocabulary — corpus-specific but tiny
# =============================================================================
#
# Every (synonym → canonical metadata field) row that a pattern can
# match. Adding a new field/synonym to a corpus means adding a row;
# the planner never guesses.

_FIELD_SYNONYMS: dict[str, str] = {
    # canonical name        → metadata key on a section/Handle
    "vendor":               "vendor",
    "vendors":              "vendor",
    "city":                 "city",
    "region":               "city",
    "product":              "product",
    "products":             "product",
    "incident":             "incident_id",
    "incidents":            "incident_id",
    "ticket":               "ticket_id",
    "tickets":              "ticket_id",
    "sla":                  "sla_minutes",
    "sla minute":           "sla_minutes",
    "mttd":                 "mttd_hours",
    "mttd hour":            "mttd_hours",
    "mttr":                 "mttr_hours",
    "mttr hour":            "mttr_hours",
    "severity":             "sev",
    "sev":                  "sev",
    "section":              "section_idx",
}

_NUMERIC_FIELDS = {"sla_minutes", "mttd_hours", "mttr_hours", "section_idx"}


# Sev keyword regex matches "Sev-1", "sev-2", "sev1", etc.
_SEV_RE = re.compile(r"\bsev[-\s]?(\d)\b", re.IGNORECASE)


# =============================================================================
# Planner
# =============================================================================


@dataclass
class PlanResult:
    """Output of `QueryPlanner.plan`."""
    plan: QueryPlan | None
    pattern: str
    matched_groups: dict
    rationale: str


class QueryPlanner:
    """Map natural-language questions to operator pipelines."""

    def __init__(self, field_synonyms: dict[str, str] | None = None):
        # Allow corpus-specific overrides without touching code.
        self.fields = dict(_FIELD_SYNONYMS)
        if field_synonyms:
            self.fields.update(field_synonyms)

    # --------- Helpers ------------------------------------------------

    def _match_field(
        self, text: str,
        allowed_canonical: set[str] | None = None,
    ) -> tuple[str, str] | None:
        """Return (canonical_field, matched_phrase) for the first synonym
        that appears in `text` and maps to a canonical field in
        `allowed_canonical` (if given). Longer synonyms tried first so
        "sla minute" beats "sla" when both apply."""
        low = text.lower()
        for syn in sorted(self.fields, key=lambda s: -len(s)):
            canon = self.fields[syn]
            if allowed_canonical is not None and canon not in allowed_canonical:
                continue
            if re.search(rf"\b{re.escape(syn)}\b", low):
                return canon, syn
        return None

    @staticmethod
    def _match_sev(text: str) -> str | None:
        m = _SEV_RE.search(text)
        if not m:
            return None
        return f"Sev-{m.group(1)}"

    @staticmethod
    def _match_in_city(text: str) -> str | None:
        # "in {City}" with capitalised first letter, allow accents.
        # Cities in the needle corpus include "São Paulo", "Cape Town",
        # "Reykjavík" — multiple words and unicode. We accept up to two
        # capitalised words (covers all corpus cities).
        m = re.search(
            r"\bin\s+([A-Z\u00C0-\u017F][\w\u00C0-\u017F]*(?:\s+[A-Z\u00C0-\u017F][\w\u00C0-\u017F]*)?)",
            text)
        return m.group(1) if m else None

    # --------- Pattern matchers ---------------------------------------

    def _try_count_distinct(self, q: str) -> PlanResult | None:
        # "how many distinct vendors", "how many unique products"
        m = re.search(
            r"\bhow\s+many\s+(?:distinct|unique|different)\s+(\w+)",
            q, re.IGNORECASE)
        if not m:
            return None
        word = m.group(1)
        target = self._match_field(word) or self._match_field(q)
        if not target:
            return None
        canon, syn = target
        plan = QueryPlan(
            ops=[Extract(name="extract", field=canon, source="metadata"),
                 Count(distinct=True)],
            answer_template="{value}",
            description=f"count distinct {canon}",
        )
        return PlanResult(plan=plan, pattern="count_distinct_field",
                          matched_groups={"field": canon, "synonym": syn},
                          rationale=f"matched 'how many distinct {syn}'")

    def _try_count_filter(self, q: str) -> PlanResult | None:
        # "how many Sev-1 incidents", "how many incidents in São Paulo"
        if not re.search(r"\bhow\s+many\b", q, re.IGNORECASE):
            return None
        if re.search(r"\bdistinct|\bunique|\bdifferent",
                     q, re.IGNORECASE):
            return None    # let count_distinct take it
        sev = self._match_sev(q)
        city = self._match_in_city(q)
        # Build predicate from any/all matched filter dims.
        if sev is None and city is None:
            return None
        def pred(md: dict, _body) -> bool:
            if sev is not None and md.get("sev") != sev:
                return False
            if city is not None and md.get("city") != city:
                return False
            return True
        # We count handles satisfying the predicate.
        plan = QueryPlan(
            ops=[Filter(name="pred", pred=pred),
                 Extract(name="ids", field="incident_id", source="metadata"),
                 Count(distinct=False)],
            description=f"count incidents matching sev={sev} city={city}",
        )
        return PlanResult(plan=plan, pattern="count_filter",
                          matched_groups={"sev": sev, "city": city},
                          rationale=f"matched 'how many' + sev/city filter")

    def _try_list_filter(self, q: str) -> PlanResult | None:
        # "list all incidents in Lyon" → list incident_ids; same for vendors.
        m = re.search(r"\blist\s+(?:all\s+|every\s+)?(\w+)", q, re.IGNORECASE)
        if not m:
            return None
        target = self._match_field(m.group(1))
        if not target:
            return None
        canon, _syn = target
        sev = self._match_sev(q)
        city = self._match_in_city(q)
        def pred(md: dict, _body) -> bool:
            if sev is not None and md.get("sev") != sev:
                return False
            if city is not None and md.get("city") != city:
                return False
            return True
        ops = [Filter(name="pred", pred=pred),
               Extract(name="vals", field=canon, source="metadata"),
               List_(sort=True)]
        plan = QueryPlan(
            ops=ops,
            description=f"list {canon} matching sev={sev} city={city}",
        )
        return PlanResult(plan=plan, pattern="list_filter",
                          matched_groups={"field": canon, "sev": sev,
                                          "city": city},
                          rationale="matched 'list ...'")

    def _try_top_group(self, q: str) -> PlanResult | None:
        # "which vendor appears most often", "which product had the most incidents"
        if not re.search(
                r"\b(which|what)\b.+\b(most|least|maximum|minimum)\b",
                q, re.IGNORECASE):
            return None
        target = self._match_field(q)
        if not target:
            return None
        canon, _syn = target
        most = re.search(r"\bmost|maximum\b", q, re.IGNORECASE) is not None
        plan = QueryPlan(
            ops=[Extract(name="vals", field=canon, source="metadata"),
                 GroupCount(top_k=1 if most else None)],
            description=f"top group by {canon} ({'most' if most else 'least'})",
        )
        return PlanResult(plan=plan, pattern="top_group",
                          matched_groups={"field": canon, "most": most},
                          rationale="matched 'which X most/least'")

    def _try_sum_field(self, q: str) -> PlanResult | None:
        # "what is the total MTTD hours across all PulseCore incidents"
        if not re.search(r"\b(total|sum)\b.+\b(mttd|mttr|sla)\b",
                         q, re.IGNORECASE):
            return None
        target = self._match_field(q, allowed_canonical=_NUMERIC_FIELDS)
        if not target:
            return None
        canon, _syn = target
        # Optional product filter, e.g., "across all PulseCore incidents".
        # Match against the known product vocabulary only — avoids
        # sentence-initial capitals like "What".
        known_products = ("PulseCore", "HelixQ", "NovaTrack", "AetherLogs",
                          "BluePeak", "Quanta-IO", "ShieldDB", "MagnetView")
        prod = None
        for p in known_products:
            if re.search(rf"\b{re.escape(p)}\b", q):
                prod = p
                break
        sev = self._match_sev(q)
        city = self._match_in_city(q)
        def pred(md: dict, _body) -> bool:
            if prod is not None and md.get("product") != prod:
                return False
            if sev is not None and md.get("sev") != sev:
                return False
            if city is not None and md.get("city") != city:
                return False
            return True
        plan = QueryPlan(
            ops=[Filter(name="pred", pred=pred),
                 Extract(name="vals", field=canon, source="metadata",
                         coerce=float),
                 Sum()],
            description=f"sum {canon} (filter prod={prod} sev={sev} city={city})",
        )
        return PlanResult(plan=plan, pattern="sum_field",
                          matched_groups={"field": canon, "product": prod,
                                          "sev": sev, "city": city},
                          rationale="matched 'total/sum X'")

    def _try_min_max(self, q: str) -> PlanResult | None:
        # "what is the largest/smallest MTTD hours"
        m = re.search(
            r"\b(largest|smallest|maximum|minimum|highest|lowest)\b",
            q, re.IGNORECASE)
        if not m:
            return None
        target = self._match_field(q, allowed_canonical=_NUMERIC_FIELDS)
        if not target:
            return None
        canon, _syn = target
        is_max = m.group(1).lower() in ("largest", "maximum", "highest")
        plan = QueryPlan(
            ops=[Extract(name="vals", field=canon, source="metadata",
                         coerce=float),
                 MinMax(op="max" if is_max else "min")],
            description=f"{'max' if is_max else 'min'} {canon}",
        )
        return PlanResult(plan=plan, pattern="min_max_field",
                          matched_groups={"field": canon, "max": is_max},
                          rationale=f"matched '{m.group(1)}'")

    def _try_join_via_ref(self, q: str) -> PlanResult | None:
        # "for incident OS-2026-0001, what is the related vendor"
        m = re.search(
            r"\b(?:for|of)\s+incident\s+(OS-\d{4}-\d{4})", q, re.IGNORECASE)
        if not m:
            return None
        if "related" not in q.lower():
            return None
        # What field on the related section? Scan ONLY the text after
        # the word "related" so a question like "for incident OS-…,
        # what is the related vendor?" resolves to the vendor of the
        # target, not to "incident" (which appears earlier).
        rel_pos = q.lower().index("related")
        tail = q[rel_pos:]
        # Prefer fields that aren't the incident_id itself (the incident
        # id was already consumed by the regex above).
        other_canon = {v for v in self.fields.values() if v != "incident_id"}
        target = self._match_field(tail, allowed_canonical=other_canon)
        if not target:
            return None
        canon, _syn = target
        incident_id = m.group(1)
        # Filter to the start incident, then join via related ref to
        # the right side, then extract the field there.
        def pred(md: dict, _body) -> bool:
            return md.get("incident_id") == incident_id
        plan = QueryPlan(
            ops=[Filter(name=f"start={incident_id}", pred=pred),
                 Join(name="related", left_ref_field="related_incident_id",
                      right_match_field="incident_id"),
                 Extract(name="vals", field=canon, source="metadata"),
                 List_()],
            description=f"join related of {incident_id} → {canon}",
        )
        return PlanResult(plan=plan, pattern="join_via_ref",
                          matched_groups={"start": incident_id,
                                          "field": canon},
                          rationale="matched 'for incident X ... related Y'")

    # --------- Public API ----------------------------------------------

    def plan(self, question: str) -> PlanResult:
        """Try every pattern in priority order. Returns the first match.

        If no pattern matches, returns `PlanResult(plan=None, ...)` so
        callers can fall back to the LLM-backed retrieval worker."""
        for matcher in (
            self._try_join_via_ref,        # most specific first
            self._try_sum_field,
            self._try_min_max,
            self._try_top_group,
            self._try_count_distinct,
            self._try_count_filter,
            self._try_list_filter,
        ):
            res = matcher(question)
            if res is not None:
                return res
        return PlanResult(plan=None, pattern="(unmatched)",
                          matched_groups={},
                          rationale="no Phase-21 pattern recognised")
