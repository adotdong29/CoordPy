"""SDK v3.3 contract tests — deeper capsule-native slice +
deterministic-mode CID determinism + lifecycle-invariant audit.

These tests lock four new claims:

  W3-39   PARSE_OUTCOME lifecycle correspondence. Each (task,
           strategy) inside a sweep cell drives a PARSE_OUTCOME
           capsule (parent: SWEEP_SPEC) BEFORE the corresponding
           PATCH_PROPOSAL. The PATCH_PROPOSAL parents on both
           SWEEP_SPEC and the PARSE_OUTCOME, giving the
           parse → patch → verdict chain a typed DAG witness.
           A PARSE_OUTCOME outside a sealed SWEEP_SPEC is
           rejected; a PATCH_PROPOSAL declaring a non-sealed
           PARSE_OUTCOME parent is rejected (C5).

  W3-40   Lifecycle-audit soundness. A finished
           ``CapsuleNativeRunContext`` whose audit returns
           ``verdict == "OK"`` satisfies the eight lifecycle
           invariants L-1..L-8 by construction. Counterexamples
           are surfaced as typed violations, not silent corruption.

  W3-41   Deterministic-mode CID determinism. Two runs of the
           same deterministic profile (mock mode, in_process
           sandbox, frozen JSONL) under
           ``RunSpec(deterministic=True)`` produce byte-identical
           full-DAG CIDs and chain head.

  W3-32-extended^2  W3-32 / W3-32-extended carry forward to the
           PARSE_OUTCOME slice: a SWEEP_SPEC must be sealed before
           any PARSE_OUTCOME; the PATCH_PROPOSAL is parented on
           the PARSE_OUTCOME's CID; the audit detects coordinate
           drift between PARSE_OUTCOME and PATCH_PROPOSAL.

The tests use the public surface only
(``from vision_mvp.wevra import ...``).
"""

from __future__ import annotations

import os
import tempfile
import unittest


class ParseOutcomeLifecycleTests(unittest.TestCase):
    """W3-39 — PARSE_OUTCOME lifecycle gate + DAG shape."""

    def test_parse_outcome_seals_under_spec(self):
        """A PARSE_OUTCOME capsule seals with parent =
        SWEEP_SPEC. Payload carries the (instance, strategy,
        parser_mode, apply_mode, n_distractors) coordinates plus
        the parser's ``ok`` boolean, ``failure_kind``, ``recovery``
        label, ``substitutions_count``, and bounded ``detail``."""
        from vision_mvp.wevra import (
            CapsuleNativeRunContext, CapsuleKind,
        )
        ctx = CapsuleNativeRunContext()
        ctx.start_run(profile_name="local_smoke", profile_dict={})
        ctx.seal_sweep_spec({
            "mode": "mock", "sandbox": "in_process",
            "jsonl": "x.jsonl", "model": None, "endpoint": None,
            "executed_in_process": True})
        cap = ctx.seal_parse_outcome(
            instance_id="mini-001", strategy="substrate",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6,
            ok=False, failure_kind="unclosed_new",
            recovery="", substitutions_count=0,
            detail="block-end missing")
        self.assertEqual(cap.kind, CapsuleKind.PARSE_OUTCOME)
        self.assertEqual(cap.parents, (ctx.spec_cap.cid,))
        p = cap.payload
        self.assertEqual(p["instance_id"], "mini-001")
        self.assertEqual(p["strategy"], "substrate")
        self.assertFalse(p["ok"])
        self.assertEqual(p["failure_kind"], "unclosed_new")
        self.assertEqual(p["recovery"], "")
        self.assertEqual(p["substitutions_count"], 0)
        self.assertEqual(p["detail"], "block-end missing")

    def test_parse_outcome_refuses_without_spec(self):
        """A PARSE_OUTCOME cannot precede SWEEP_SPEC — the W3-39
        lifecycle gate is enforced at the type level."""
        from vision_mvp.wevra import (
            CapsuleNativeRunContext, CapsuleLifecycleError,
        )
        ctx = CapsuleNativeRunContext()
        ctx.start_run(profile_name="local_smoke", profile_dict={})
        with self.assertRaises(CapsuleLifecycleError):
            ctx.seal_parse_outcome(
                instance_id="x", strategy="substrate",
                parser_mode="strict", apply_mode="strict",
                n_distractors=6,
                ok=True, failure_kind="ok")

    def test_patch_parent_includes_parse_outcome_when_passed(self):
        """When ``parse_outcome_cid`` is provided to
        ``seal_patch_proposal``, the PATCH_PROPOSAL's parents
        contain BOTH the SWEEP_SPEC and the PARSE_OUTCOME."""
        from vision_mvp.wevra import (
            CapsuleNativeRunContext, CapsuleKind,
        )
        ctx = CapsuleNativeRunContext()
        ctx.start_run(profile_name="local_smoke", profile_dict={})
        ctx.seal_sweep_spec({
            "mode": "mock", "sandbox": "in_process",
            "jsonl": "x.jsonl", "model": None, "endpoint": None,
            "executed_in_process": True})
        p = ctx.seal_parse_outcome(
            instance_id="m1", strategy="naive",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6,
            ok=True, failure_kind="ok")
        patch = ctx.seal_patch_proposal(
            instance_id="m1", strategy="naive",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6,
            substitutions=(("a", "b"),),
            parse_outcome_cid=p.cid)
        self.assertIn(ctx.spec_cap.cid, patch.parents)
        self.assertIn(p.cid, patch.parents)
        self.assertEqual(len(patch.parents), 2)

    def test_patch_refuses_unsealed_parse_outcome_parent(self):
        """A PATCH_PROPOSAL declaring a non-sealed
        ``parse_outcome_cid`` is rejected (C5)."""
        from vision_mvp.wevra import (
            CapsuleNativeRunContext, CapsuleLifecycleError,
        )
        ctx = CapsuleNativeRunContext()
        ctx.start_run(profile_name="local_smoke", profile_dict={})
        ctx.seal_sweep_spec({
            "mode": "mock", "sandbox": "in_process",
            "jsonl": "x.jsonl", "model": None, "endpoint": None,
            "executed_in_process": True})
        fake_parse_cid = "f" * 64
        with self.assertRaises(CapsuleLifecycleError):
            ctx.seal_patch_proposal(
                instance_id="m1", strategy="naive",
                parser_mode="strict", apply_mode="strict",
                n_distractors=6, substitutions=(),
                parse_outcome_cid=fake_parse_cid)

    def test_smoke_run_emits_parse_outcome_per_patch(self):
        """A capsule-native run on local_smoke emits one
        PARSE_OUTCOME per (task, strategy) — equal in count to
        PATCH_PROPOSAL and TEST_VERDICT."""
        from vision_mvp.wevra import RunSpec, run, CapsuleKind
        with tempfile.TemporaryDirectory() as td:
            r = run(RunSpec(profile="local_smoke", out_dir=td))
        view = r["capsules"]
        parses = [c for c in view["capsules"]
                   if c["kind"] == CapsuleKind.PARSE_OUTCOME]
        patches = [c for c in view["capsules"]
                    if c["kind"] == CapsuleKind.PATCH_PROPOSAL]
        verdicts = [c for c in view["capsules"]
                     if c["kind"] == CapsuleKind.TEST_VERDICT]
        self.assertGreater(len(parses), 0)
        self.assertEqual(len(parses), len(patches))
        self.assertEqual(len(parses), len(verdicts))
        # On local_smoke (deterministic_oracle path), every parse
        # outcome has failure_kind == "oracle".
        for p in parses:
            self.assertEqual(p["payload"]["failure_kind"], "oracle")
            self.assertTrue(p["payload"]["ok"])

    def test_smoke_run_patch_chains_on_parse(self):
        """Each PATCH_PROPOSAL's parent set contains exactly one
        PARSE_OUTCOME's CID, and the PARSE_OUTCOME's coordinates
        match the PATCH_PROPOSAL's coordinates."""
        from vision_mvp.wevra import RunSpec, run, CapsuleKind
        with tempfile.TemporaryDirectory() as td:
            r = run(RunSpec(profile="local_smoke", out_dir=td))
        view = r["capsules"]
        parses = {c["cid"]: c for c in view["capsules"]
                    if c["kind"] == CapsuleKind.PARSE_OUTCOME}
        patches = [c for c in view["capsules"]
                    if c["kind"] == CapsuleKind.PATCH_PROPOSAL]
        for patch in patches:
            parse_parents = [p for p in patch["parents"]
                              if p in parses]
            self.assertEqual(len(parse_parents), 1)
            parse = parses[parse_parents[0]]
            for k in ("instance_id", "strategy", "parser_mode",
                       "apply_mode", "n_distractors"):
                self.assertEqual(
                    parse["payload"][k], patch["payload"][k],
                    f"PARSE_OUTCOME / PATCH_PROPOSAL coordinate "
                    f"mismatch on field {k}")


class LifecycleAuditTests(unittest.TestCase):
    """W3-40 — runtime-checkable lifecycle audit."""

    def _make_clean_ctx(self):
        from vision_mvp.wevra import CapsuleNativeRunContext
        ctx = CapsuleNativeRunContext()
        ctx.start_run(profile_name="local_smoke", profile_dict={})
        ctx.seal_sweep_spec({
            "mode": "mock", "sandbox": "in_process",
            "jsonl": "x.jsonl", "model": None, "endpoint": None,
            "executed_in_process": True})
        p = ctx.seal_parse_outcome(
            instance_id="m1", strategy="naive",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6,
            ok=True, failure_kind="ok")
        patch = ctx.seal_patch_proposal(
            instance_id="m1", strategy="naive",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6,
            substitutions=(("a", "b"),),
            parse_outcome_cid=p.cid)
        ctx.seal_test_verdict(
            instance_id="m1", strategy="naive",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6,
            patch_proposal_cid=patch.cid,
            patch_applied=True, syntax_ok=True, test_passed=True)
        return ctx

    def test_audit_clean_run_is_ok(self):
        from vision_mvp.wevra import audit_capsule_lifecycle
        ctx = self._make_clean_ctx()
        report = audit_capsule_lifecycle(ctx)
        self.assertEqual(report.verdict, "OK")
        self.assertEqual(report.violations, [])
        self.assertEqual(
            len(report.rules_passed), len(report.rules_checked))

    def test_audit_smoke_run_is_ok(self):
        """The full local_smoke run produces a ledger that
        passes every lifecycle invariant."""
        from vision_mvp.wevra import (
            RunSpec, run, audit_capsule_lifecycle_from_view,
        )
        with tempfile.TemporaryDirectory() as td:
            r = run(RunSpec(profile="local_smoke", out_dir=td))
        report = audit_capsule_lifecycle_from_view(r["capsules"])
        self.assertEqual(report.verdict, "OK", report.violations)

    def test_audit_detects_coordinate_drift(self):
        """A PATCH_PROPOSAL whose coordinates diverge from its
        PARSE_OUTCOME parent's coordinates is flagged as L-7
        violation."""
        from vision_mvp.wevra import (
            CapsuleNativeRunContext, audit_capsule_lifecycle,
        )
        ctx = CapsuleNativeRunContext()
        ctx.start_run(profile_name="local_smoke", profile_dict={})
        ctx.seal_sweep_spec({
            "mode": "mock", "sandbox": "in_process",
            "jsonl": "x.jsonl", "model": None, "endpoint": None,
            "executed_in_process": True})
        p = ctx.seal_parse_outcome(
            instance_id="m1", strategy="naive",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6,
            ok=True, failure_kind="ok")
        # Drift instance_id
        ctx.seal_patch_proposal(
            instance_id="WRONG", strategy="naive",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6,
            substitutions=(("a", "b"),),
            parse_outcome_cid=p.cid)
        report = audit_capsule_lifecycle(ctx)
        self.assertEqual(report.verdict, "BAD")
        rules = {v["rule"] for v in report.violations}
        # L-6 catches the count-coords mismatch (no matching
        # patch coords for the parse), L-7 catches the per-CID
        # mismatch (this depends on how many violations bubble).
        self.assertTrue(
            "L-7_patch_coordinates_match_parse_outcome" in rules
            or "L-6_patch_has_matching_parse_and_verdict" in rules,
            f"expected L-6 or L-7 violation, got {rules}")

    def test_audit_returns_empty_on_no_capsules(self):
        from vision_mvp.wevra import (
            CapsuleNativeRunContext, audit_capsule_lifecycle,
        )
        ctx = CapsuleNativeRunContext()
        report = audit_capsule_lifecycle(ctx)
        self.assertEqual(report.verdict, "EMPTY")


class DeterministicModeTests(unittest.TestCase):
    """W3-41 — deterministic-mode CID determinism."""

    def _collect_cids(self, report):
        out: dict[str, list[str]] = {}
        for c in report["capsules"]["capsules"]:
            out.setdefault(c["kind"], []).append(c["cid"])
        return {k: sorted(v) for k, v in out.items()}

    def test_w3_41_two_runs_collapse_to_identical_cids(self):
        """Two runs of local_smoke under
        ``RunSpec(deterministic=True)`` produce byte-identical
        CIDs on every kind, including ARTIFACT and RUN_REPORT."""
        from vision_mvp.wevra import RunSpec, run
        with tempfile.TemporaryDirectory() as td1:
            r1 = run(RunSpec(
                profile="local_smoke", out_dir=td1,
                deterministic=True))
        with tempfile.TemporaryDirectory() as td2:
            r2 = run(RunSpec(
                profile="local_smoke", out_dir=td2,
                deterministic=True))
        c1 = self._collect_cids(r1)
        c2 = self._collect_cids(r2)
        self.assertEqual(set(c1), set(c2),
                          "kind sets must match across runs")
        for k in c1:
            self.assertEqual(
                c1[k], c2[k],
                f"CIDs of kind {k!r} differ across deterministic "
                f"runs")
        self.assertEqual(
            r1["capsules"]["chain_head"],
            r2["capsules"]["chain_head"],
            "chain heads must match across deterministic runs")
        self.assertEqual(
            r1["capsules"]["root_cid"],
            r2["capsules"]["root_cid"],
            "root CIDs must match across deterministic runs")

    def test_default_mode_is_not_deterministic(self):
        """Without the deterministic flag, RUN_REPORT and
        PROVENANCE CIDs differ between runs (timestamp /
        wall_seconds variance). This is the negative side of
        W3-41 — opt-in only."""
        from vision_mvp.wevra import RunSpec, run
        with tempfile.TemporaryDirectory() as td1:
            r1 = run(RunSpec(profile="local_smoke", out_dir=td1))
        with tempfile.TemporaryDirectory() as td2:
            r2 = run(RunSpec(profile="local_smoke", out_dir=td2))
        c1 = self._collect_cids(r1)
        c2 = self._collect_cids(r2)
        # PROVENANCE differs because it carries timestamp_utc.
        self.assertNotEqual(
            c1.get("PROVENANCE"), c2.get("PROVENANCE"),
            "PROVENANCE CIDs should differ in non-deterministic mode")

    def test_deterministic_mode_audit_still_passes(self):
        """The lifecycle audit holds under deterministic mode —
        canonicalisation does not break L-1..L-8."""
        from vision_mvp.wevra import (
            RunSpec, run, audit_capsule_lifecycle_from_view,
        )
        with tempfile.TemporaryDirectory() as td:
            r = run(RunSpec(
                profile="local_smoke", out_dir=td,
                deterministic=True))
        report = audit_capsule_lifecycle_from_view(r["capsules"])
        self.assertEqual(report.verdict, "OK", report.violations)


class ParseOutcomeRationaleMappingTests(unittest.TestCase):
    """The runtime maps ``ProposedPatch.rationale`` to a typed
    ``(ok, failure_kind, recovery, detail)`` tuple. The mapping is
    deterministic and covers every shape the substrate can
    produce (oracle, llm_proposed, llm_proposed:<recovery>,
    parse_failed:<kind>, gen_error)."""

    def test_oracle_rationale_maps_to_oracle_kind(self):
        from vision_mvp.wevra.runtime import (
            _parse_outcome_from_rationale,
        )
        ok, kind, rec, detail = _parse_outcome_from_rationale(
            "issue summary text", n_substitutions=1)
        self.assertTrue(ok)
        self.assertEqual(kind, "oracle")
        self.assertEqual(rec, "")

    def test_llm_proposed_clean_maps_to_ok(self):
        from vision_mvp.wevra.runtime import (
            _parse_outcome_from_rationale,
        )
        ok, kind, rec, _ = _parse_outcome_from_rationale(
            "llm_proposed", n_substitutions=1)
        self.assertTrue(ok)
        self.assertEqual(kind, "ok")
        self.assertEqual(rec, "")

    def test_llm_proposed_with_recovery_maps_recovery(self):
        from vision_mvp.wevra.runtime import (
            _parse_outcome_from_rationale,
        )
        ok, kind, rec, _ = _parse_outcome_from_rationale(
            "llm_proposed:closed_at_eos", n_substitutions=1)
        self.assertTrue(ok)
        self.assertEqual(kind, "ok")
        self.assertEqual(rec, "closed_at_eos")

    def test_parse_failed_maps_to_failure_kind(self):
        from vision_mvp.wevra.runtime import (
            _parse_outcome_from_rationale,
        )
        ok, kind, rec, _ = _parse_outcome_from_rationale(
            "parse_failed:unclosed_new", n_substitutions=0)
        self.assertFalse(ok)
        self.assertEqual(kind, "unclosed_new")
        self.assertEqual(rec, "")

    def test_gen_error_maps_to_typed_failure(self):
        from vision_mvp.wevra.runtime import (
            _parse_outcome_from_rationale,
        )
        ok, kind, rec, detail = _parse_outcome_from_rationale(
            "gen_error:RuntimeError", n_substitutions=0)
        self.assertFalse(ok)
        self.assertEqual(kind, "gen_error")
        self.assertEqual(rec, "")
        self.assertIn("RuntimeError", detail)


if __name__ == "__main__":
    unittest.main()
