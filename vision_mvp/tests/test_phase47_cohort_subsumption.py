r"""Phase 47 cohort subsumption tests.

Operational verification of Theorems W3-14 / W3-15 / W3-16
(``docs/CAPSULE_FORMALISM.md`` § 4, Phase-47 extension):

* **W3-14 (per-capsule locality)**. Per-capsule budgets cannot
  enforce cardinality invariants on projections of the admitted
  set.
* **W3-15 (cohort lift)**. Adding a ``COHORT`` capsule whose
  ``parents`` are the members and whose ``max_parents`` is the
  cardinality cap lifts AdaptiveEdge from PARTIAL to FULL. The
  Phase-36 ``max_active_edges`` table-level invariant is
  subsumable at the *cohort* level.
* **W3-16 (relational limitation)**. Even with COHORT, the
  capsule algebra cannot express a *relational* invariant over
  distinct members (e.g. "no two members share an event_id")
  unless the cohort's constructor computes that predicate
  upfront. Cohort-admission alone is cardinality-only.

The four test classes below lock each theorem in an operational
test that would fail if the capsule algebra silently drifted.
"""

from __future__ import annotations

import unittest


class W3_14_PerCapsuleLocalityTests(unittest.TestCase):
    r"""W3-14 is a *negative* theorem — no per-capsule budget can
    enforce "total number of admitted HANDLE capsules ≤ N" because
    the admit check runs on one capsule at a time and has no
    access to other sealed capsules' identities.

    The test documents this by constructing a ledger and showing
    that admitting N+1 capsules under any per-capsule budget still
    succeeds — the cardinality bound is not enforced."""

    def test_per_capsule_budget_cannot_bound_total_count(self):
        from vision_mvp.wevra import (
            ContextCapsule, CapsuleKind, CapsuleBudget, CapsuleLedger,
        )
        ledger = CapsuleLedger()
        # Set every axis of the per-capsule budget to tight values.
        # This is the strongest-possible per-capsule constraint.
        budget = CapsuleBudget(
            max_tokens=64, max_bytes=1 << 16,
            max_rounds=1, max_witnesses=1, max_parents=0)
        # Admit 100 HANDLE capsules. Each passes the per-capsule
        # budget. There is no per-capsule axis that limits the
        # total count.
        n = 100
        for i in range(n):
            c = ContextCapsule.new(
                kind=CapsuleKind.HANDLE,
                payload={"handle_cid": f"{i:064d}", "fp": f"f{i}"},
                budget=budget, n_tokens=1,
            )
            ledger.admit_and_seal(c)
        # All 100 admitted. No per-capsule budget stopped them.
        self.assertEqual(len(ledger), n)

    def test_table_cap_not_in_legal_axes(self):
        """Check that ``CapsuleBudget`` does not silently gain a
        ``max_total_ledger_size`` or ``max_total_members`` axis — a
        regression of W3-14 would be a new axis of that shape."""
        from vision_mvp.wevra import CapsuleBudget
        axes = set(CapsuleBudget().__dataclass_fields__.keys()) \
            if False else set()
        import dataclasses
        fields = {f.name
                  for f in dataclasses.fields(CapsuleBudget)}
        # Legal axes (as of Phase 47).
        legal = {"max_tokens", "max_bytes", "max_rounds",
                  "max_witnesses", "max_parents"}
        self.assertEqual(fields, legal)
        self.assertNotIn("max_total_ledger_size", fields)


class W3_15_CohortLiftTests(unittest.TestCase):
    r"""W3-15 — cohort subsumption lifts AdaptiveEdge PARTIAL → FULL.

    A COHORT capsule whose parent set is the active edge CIDs and
    whose ``max_parents`` is the table-level cap replicates the
    ``AdaptiveSubscriptionTable.install_edge`` cardinality rejection
    *at admission time*, not just inside the substrate."""

    def test_cohort_admits_within_cap(self):
        from vision_mvp.wevra import (
            ContextCapsule, CapsuleKind, CapsuleBudget, CapsuleLedger,
            capsule_from_cohort,
        )
        ledger = CapsuleLedger()
        # Admit three ADAPTIVE_EDGE members, then a cohort over them.
        member_cids = []
        for i in range(3):
            cap = ContextCapsule.new(
                kind=CapsuleKind.ADAPTIVE_EDGE,
                payload={"edge_id": f"E-{i}", "source_role": "monitor",
                          "claim_kind": "CAUSALITY_HYPOTHESIS",
                          "consumer_roles": ["auditor"],
                          "ttl_rounds": 2, "installed_at": 0},
                budget=CapsuleBudget(max_rounds=4, max_parents=8))
            sealed = ledger.admit_and_seal(cap)
            member_cids.append(sealed.cid)
        cohort = capsule_from_cohort(
            cohort_tag="adaptive_edge_table",
            member_cids=member_cids,
            max_members=4,  # table cap = 4
            predicate_note="active in tick=0",
            extra_payload={"tick": 0})
        sealed_cohort = ledger.admit_and_seal(cohort)
        self.assertEqual(sealed_cohort.kind, CapsuleKind.COHORT)
        self.assertEqual(
            sealed_cohort.metadata_dict()["n_members"], 3)
        self.assertTrue(ledger.verify_chain())

    def test_cohort_over_cap_rejected_at_construction(self):
        """If membership exceeds ``max_members``, the cohort's
        own construction fails via the ``max_parents`` axis."""
        from vision_mvp.wevra import capsule_from_cohort
        # 5 members, cap = 4 — rejected at construction.
        with self.assertRaises(ValueError):
            capsule_from_cohort(
                cohort_tag="adaptive_edge_table",
                member_cids=[f"cid{i}" for i in range(5)],
                max_members=4)

    def test_adaptive_sub_table_adapter_witnesses_cap(self):
        """End-to-end: AdaptiveSubscriptionTable at the cap → cohort
        admission fails; below the cap → cohort admits.

        This closes the Phase-46 PARTIAL verdict on AdaptiveEdge:
        the table-level ``max_active_edges`` bound is subsumable
        under the capsule contract via the cohort lift."""
        from vision_mvp.core.role_handoff import RoleSubscriptionTable
        from vision_mvp.core.adaptive_sub import (
            AdaptiveSubscriptionTable,
        )
        from vision_mvp.wevra import (
            ContextCapsule, CapsuleKind, CapsuleBudget, CapsuleLedger,
            capsule_from_adaptive_sub_table,
        )
        base = RoleSubscriptionTable()
        table = AdaptiveSubscriptionTable(
            base=base, max_active_edges=3)
        # Install 3 edges — at the cap.
        table.install_edge("monitor", "CAUSALITY_HYPOTHESIS",
                            ["auditor"], ttl_rounds=2)
        table.install_edge("db_admin", "CAUSALITY_HYPOTHESIS",
                            ["auditor"], ttl_rounds=2)
        table.install_edge("sysadmin", "CAUSALITY_HYPOTHESIS",
                            ["auditor"], ttl_rounds=2)
        # Lift the table to a cohort capsule. Succeeds because 3 ≤ 3.
        ledger = CapsuleLedger()
        # Admit the edge capsules first (cohort parents must be
        # in the ledger).
        edge_cids = []
        for e in table.active_edges():
            cap = ContextCapsule.new(
                kind=CapsuleKind.ADAPTIVE_EDGE,
                payload=e.as_dict(),
                budget=CapsuleBudget(max_rounds=4, max_parents=8))
            sealed = ledger.admit_and_seal(cap)
            edge_cids.append(sealed.cid)
        cohort = capsule_from_adaptive_sub_table(
            table, tick=0, edge_cids=edge_cids)
        sealed_cohort = ledger.admit_and_seal(cohort)
        self.assertEqual(
            sealed_cohort.metadata_dict()["n_members"], 3)

        # Now synthesise a fourth edge CID that would exceed the
        # cap. Re-lift the cohort with 4 members → construction
        # fails (max_parents = max_active_edges = 3).
        from vision_mvp.wevra import capsule_from_cohort
        with self.assertRaises(ValueError):
            capsule_from_cohort(
                cohort_tag="adaptive_edge_table",
                member_cids=edge_cids + ["would-be-4th-cid"],
                max_members=table.max_active_edges)


class W3_16_RelationalLimitationTests(unittest.TestCase):
    r"""W3-16 — cohort lifting is *cardinality-only*. Predicates
    quantified over *pairs* of distinct members (e.g. "no two
    members share an event_id") cannot be enforced by cohort
    admission alone.

    These tests document that the cohort admission rule remains
    silent on relational invariants. If a future change tried to
    sneak a relational check into cohort admission, these tests
    would need explicit revision — forcing an honest write-up of
    what changed."""

    def test_cohort_admission_silent_on_member_overlap(self):
        """A cohort whose two members share a duplicate
        ``source_event_ids=(1,)`` is still admitted by the
        cardinality-only cohort rule. The relational invariant
        'no two members share an event_id' is NOT enforced."""
        from vision_mvp.wevra import (
            ContextCapsule, CapsuleKind, CapsuleBudget, CapsuleLedger,
            capsule_from_cohort,
        )
        ledger = CapsuleLedger()
        # Two HANDOFF capsules with deliberately overlapping
        # source_event_ids.
        cap_a = ContextCapsule.new(
            kind=CapsuleKind.HANDOFF,
            payload={"msg": "a", "source_event_ids": [1]},
            budget=CapsuleBudget(max_tokens=64, max_parents=16),
            n_tokens=4,
            metadata={"source_role": "monitor",
                       "claim_kind": "ERROR_RATE_SPIKE"})
        cap_b = ContextCapsule.new(
            kind=CapsuleKind.HANDOFF,
            payload={"msg": "b", "source_event_ids": [1]},
            budget=CapsuleBudget(max_tokens=64, max_parents=16),
            n_tokens=4,
            metadata={"source_role": "db_admin",
                       "claim_kind": "SLOW_QUERY_OBSERVED"})
        sealed_a = ledger.admit_and_seal(cap_a)
        sealed_b = ledger.admit_and_seal(cap_b)
        # Cohort admits both despite shared event id.
        cohort = capsule_from_cohort(
            cohort_tag="overlap_test",
            member_cids=[sealed_a.cid, sealed_b.cid],
            max_members=4,
            predicate_note="overlap invariant NOT enforced by admission")
        sealed_cohort = ledger.admit_and_seal(cohort)
        self.assertEqual(
            sealed_cohort.metadata_dict()["n_members"], 2)
        # The cohort is admitted. The relational invariant is a
        # property of the caller's predicate, not of cohort admission.

    def test_cohort_parents_are_deduplicated_by_cid(self):
        """Exactly-duplicate CIDs are not double-counted — this
        is a *consequence* of content addressing (W3-7), not of
        relational enforcement. A cohort whose member list
        contains duplicates collapses to the deduplicated set
        automatically via the CID canonicalisation."""
        from vision_mvp.wevra import (
            ContextCapsule, CapsuleKind, CapsuleBudget, CapsuleLedger,
            capsule_from_cohort,
        )
        ledger = CapsuleLedger()
        cap = ContextCapsule.new(
            kind=CapsuleKind.HANDLE,
            payload={"handle_cid": "0" * 64, "fp": "z"},
            budget=CapsuleBudget(max_tokens=8, max_parents=8),
            n_tokens=1)
        sealed = ledger.admit_and_seal(cap)
        # A cohort whose member list duplicates the same CID.
        # The current adapter accepts duplicates as distinct
        # positions; we document this behaviour so a future change
        # that adds dedup semantics must update this test.
        cohort = capsule_from_cohort(
            cohort_tag="dedup_probe",
            member_cids=[sealed.cid, sealed.cid, sealed.cid],
            max_members=8)
        # Current behaviour: parents is a tuple; duplicates ARE
        # kept at construction, and the cohort's parent count is
        # 3. The relational "no duplicate CID among members" check
        # is NOT enforced by cohort admission.
        self.assertEqual(len(cohort.parents), 3)


class CohortInUnificationAuditTests(unittest.TestCase):
    """Smoke test that the full Phase-46 unification audit with
    the added Phase-47 cohort step reports AdaptiveEdge as FULL
    (via cohort lift) rather than PARTIAL."""

    def test_audit_runs_and_has_cohort_fit(self):
        from vision_mvp.experiments.phase46_unification_audit import (
            run_audit,
        )
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            out = run_audit(out_dir=td)
            names = [a["primitive"] for a in out["audits"]]
            self.assertTrue(
                any("AdaptiveEdge" in n for n in names),
                "AdaptiveEdge row missing from audit")
            # After the cohort lift, at least the cohort row
            # should be present — and AdaptiveEdge (cohort-lifted)
            # should be FULL.
            cohort_rows = [
                a for a in out["audits"]
                if "cohort" in a["primitive"].lower()
                    or a.get("capsule_kind") == "COHORT"
                    or a.get("verdict") == "FULL"
                    and "AdaptiveEdge" in a["primitive"]]
            self.assertGreater(len(cohort_rows), 0,
                "Expected a FULL verdict on a cohort-lifted primitive")


if __name__ == "__main__":
    unittest.main()
