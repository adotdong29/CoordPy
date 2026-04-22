"""Tests for exact-computation operators (Phase 21)."""

from __future__ import annotations

import re
import unittest

import numpy as np

from vision_mvp.core.context_ledger import ContextLedger, hash_embedding
from vision_mvp.core.exact_ops import (
    Count, Extract, Filter, GroupCount, Join, List_, MinMax, QueryPlan,
    StageGroups, StageHandles, StageList, StageScalar, StageValues, Sum,
)


def _embed(text: str) -> np.ndarray:
    return hash_embedding(text, dim=32)


def _seed_ledger():
    """Mini incident corpus with 5 sections of typed metadata."""
    l = ContextLedger(embed_dim=32, embed_fn=_embed)
    rows = [
        ("OS-2026-0001", "NordAxis",  "Frankfurt",  "PulseCore", "Sev-1", 30, 12),
        ("OS-2026-0002", "NordAxis",  "Singapore",  "HelixQ",    "Sev-2", 60, 24),
        ("OS-2026-0003", "GridFlux",  "Lyon",       "PulseCore", "Sev-1", 15, 5),
        ("OS-2026-0004", "Cedarform", "Lyon",       "NovaTrack", "Sev-3", 90, 48),
        ("OS-2026-0005", "NordAxis",  "São Paulo",  "PulseCore", "Sev-2", 45, 18),
    ]
    handles = []
    for idx, (iid, vendor, city, prod, sev, sla, mttd) in enumerate(rows):
        h = l.put(
            f"Section {idx+1}. Incident {iid} — {prod} {sev} in {city}. "
            f"Vendor {vendor}. SLA {sla} minutes. MTTD {mttd} hours.",
            metadata={
                "section_idx": idx,
                "incident_id": iid,
                "vendor": vendor,
                "city": city,
                "product": prod,
                "sev": sev,
                "sla_minutes": sla,
                "mttd_hours": mttd,
            },
        )
        handles.append(h)
    return l, handles


class TestFilterAndExtract(unittest.TestCase):
    def setUp(self):
        self.ledger, self.handles = _seed_ledger()

    def test_filter_metadata_only(self):
        trace = []
        f = Filter(name="sev1", pred=lambda md, _b: md.get("sev") == "Sev-1")
        out = f.execute(self.ledger, None, trace)
        self.assertEqual(len(out.handles), 2)
        self.assertEqual(trace[-1].name, "Filter(sev1)")
        # Metadata-only filtering does NO ledger.fetch.
        self.assertEqual(trace[-1].cids_touched, [])

    def test_filter_body_required_fetches(self):
        trace = []
        f = Filter(name="vendor_in_body",
                   pred=lambda _md, body: "NordAxis" in (body or ""),
                   body_required=True)
        out = f.execute(self.ledger, None, trace)
        self.assertEqual(len(out.handles), 3)   # 3 NordAxis sections
        self.assertEqual(len(trace[-1].cids_touched), 5)  # all bodies fetched

    def test_extract_metadata(self):
        trace = []
        all_h = StageHandles(handles=self.handles)
        e = Extract(name="vendors", field="vendor", source="metadata")
        out = e.execute(self.ledger, all_h, trace)
        self.assertEqual([v for _h, v in out.pairs],
                         ["NordAxis", "NordAxis", "GridFlux",
                          "Cedarform", "NordAxis"])

    def test_extract_regex(self):
        trace = []
        all_h = StageHandles(handles=self.handles)
        # Pull the SLA value from the body via regex.
        e = Extract(name="sla_re", field="body_regex", source="regex",
                    pattern=re.compile(r"SLA\s+(\d+)\s+minutes"),
                    group=1, coerce=int)
        out = e.execute(self.ledger, all_h, trace)
        self.assertEqual([v for _h, v in out.pairs],
                         [30, 60, 15, 90, 45])
        # All bodies were fetched.
        self.assertEqual(len(trace[-1].cids_touched), 5)


class TestReductions(unittest.TestCase):
    def setUp(self):
        self.ledger, self.handles = _seed_ledger()

    def _vendors_stage(self):
        trace: list = []
        e = Extract(name="vendors", field="vendor", source="metadata")
        return e.execute(self.ledger, StageHandles(handles=self.handles),
                         trace), trace

    def test_count_total(self):
        s, _ = self._vendors_stage()
        out = Count(distinct=False).execute(self.ledger, s, [])
        self.assertEqual(out.value, 5)

    def test_count_distinct(self):
        s, _ = self._vendors_stage()
        out = Count(distinct=True).execute(self.ledger, s, [])
        self.assertEqual(out.value, 3)   # NordAxis, GridFlux, Cedarform

    def test_sum_numeric(self):
        trace = []
        e = Extract(name="mttd", field="mttd_hours", source="metadata",
                    coerce=int)
        s = e.execute(self.ledger, StageHandles(handles=self.handles), trace)
        out = Sum().execute(self.ledger, s, trace)
        self.assertEqual(out.value, 12 + 24 + 5 + 48 + 18)

    def test_min_max(self):
        trace = []
        e = Extract(name="sla", field="sla_minutes", source="metadata",
                    coerce=int)
        s = e.execute(self.ledger, StageHandles(handles=self.handles), trace)
        self.assertEqual(MinMax(op="min").execute(self.ledger, s, []).value, 15)
        self.assertEqual(MinMax(op="max").execute(self.ledger, s, []).value, 90)

    def test_group_count(self):
        s, _ = self._vendors_stage()
        out = GroupCount().execute(self.ledger, s, [])
        self.assertEqual(out.groups, {"NordAxis": 3, "GridFlux": 1,
                                       "Cedarform": 1})

    def test_group_count_top_k(self):
        s, _ = self._vendors_stage()
        out = GroupCount(top_k=1).execute(self.ledger, s, [])
        self.assertEqual(out.groups, {"NordAxis": 3})

    def test_list_sorted(self):
        s, _ = self._vendors_stage()
        out = List_(sort=True).execute(self.ledger, s, [])
        self.assertEqual(out.items,
                         ["Cedarform", "GridFlux", "NordAxis", "NordAxis",
                          "NordAxis"])


class TestQueryPlanCompose(unittest.TestCase):
    def setUp(self):
        self.ledger, _ = _seed_ledger()

    def test_count_distinct_vendors_pipeline(self):
        plan = QueryPlan(ops=[
            Extract(name="v", field="vendor", source="metadata"),
            Count(distinct=True),
        ])
        result, trace = plan.execute(self.ledger)
        self.assertIsInstance(result, StageScalar)
        self.assertEqual(result.value, 3)
        self.assertEqual(plan.render(result), "3")

    def test_count_filter_pipeline(self):
        plan = QueryPlan(ops=[
            Filter(name="lyon", pred=lambda md, _b: md.get("city") == "Lyon"),
            Extract(name="ids", field="incident_id", source="metadata"),
            Count(distinct=False),
        ])
        result, _ = plan.execute(self.ledger)
        self.assertEqual(result.value, 2)

    def test_max_mttd_pipeline(self):
        plan = QueryPlan(ops=[
            Extract(name="m", field="mttd_hours", source="metadata", coerce=int),
            MinMax(op="max"),
        ])
        result, _ = plan.execute(self.ledger)
        self.assertEqual(result.value, 48)

    def test_render_groups(self):
        plan = QueryPlan(ops=[
            Extract(name="v", field="vendor", source="metadata"),
            GroupCount(),
        ])
        result, _ = plan.execute(self.ledger)
        rendered = plan.render(result)
        # NordAxis should appear first (highest count).
        self.assertTrue(rendered.startswith("NordAxis: 3"))


class TestJoin(unittest.TestCase):
    def setUp(self):
        # Build a corpus where each section references the NEXT one.
        l = ContextLedger(embed_dim=32, embed_fn=_embed)
        for i in range(4):
            l.put(
                f"section {i} body referencing section {(i+1) % 4}",
                metadata={
                    "section_idx": i,
                    "incident_id": f"OS-{i:04d}",
                    "related_incident_id": f"OS-{(i+1) % 4:04d}",
                    "vendor": ["A", "B", "C", "D"][i],
                },
            )
        self.ledger = l

    def test_join_follows_reference(self):
        # Start at incident 0, join to its related, extract vendor.
        plan = QueryPlan(ops=[
            Filter(name="start",
                   pred=lambda md, _b: md.get("incident_id") == "OS-0000"),
            Join(name="rel", left_ref_field="related_incident_id",
                 right_match_field="incident_id"),
            Extract(name="vendor", field="vendor", source="metadata"),
            List_(),
        ])
        result, trace = plan.execute(self.ledger)
        self.assertEqual(result.items, ["B"])    # related of A is B
        # Three operators ran (Filter, Join, Extract, List_) → 4 traces
        self.assertEqual(len(trace), 4)


class TestPureExactness(unittest.TestCase):
    """An end-to-end sanity check: the operator pipeline never touches the
    LLM and never paraphrases."""

    def test_no_summarization_in_pipeline(self):
        ledger, _ = _seed_ledger()
        plan = QueryPlan(ops=[
            Extract(name="v", field="vendor", source="metadata"),
            Count(distinct=True),
        ])
        before_fetch = ledger.stats_dict()["n_fetch"]
        result, _ = plan.execute(ledger)
        after_fetch = ledger.stats_dict()["n_fetch"]
        # Metadata-only path: zero ledger fetches.
        self.assertEqual(after_fetch, before_fetch)
        self.assertEqual(result.value, 3)


if __name__ == "__main__":
    unittest.main()
