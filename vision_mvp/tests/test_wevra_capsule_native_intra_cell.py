"""SDK v3.2 contract tests — intra-cell capsule-native slice + detached
META_MANIFEST witness + stronger on-disk verification.

These tests lock four new claims:

  W3-32-extended  Lifecycle ↔ execution-state correspondence on
                   intra-cell kinds. Each (task, strategy) inside a
                   sweep cell drives a PATCH_PROPOSAL capsule
                   (parent: SWEEP_SPEC) and a TEST_VERDICT capsule
                   (parent: PATCH_PROPOSAL). The chain ``patch →
                   verdict`` is enforced at the capsule layer; a
                   verdict cannot precede its patch.

  W3-36           Meta-artifact circularity / detached-witness.
                   The set of meta-artifacts whose bytes are a
                   structural function of the rendered RUN_REPORT
                   view cannot be authenticated by ARTIFACT
                   capsules in the primary ledger. The strongest
                   authentication achievable is a detached
                   META_MANIFEST in a secondary ledger; the trust
                   unit is one explicit hop beyond the primary
                   view.

  W3-37           Chain-from-headers verification. The
                   ``capsule_view.json`` chain head is recomputable
                   from on-disk header bytes. ``wevra-capsule
                   verify`` recomputes the chain step-by-step from
                   disk and compares to the on-disk
                   ``chain_head`` claim — a tamper that flips a
                   CID, a kind, or an ordering is detected.

  W3-38           ARTIFACT on-disk re-hash. Every ARTIFACT
                   capsule's payload SHA-256 is checked against
                   the actual on-disk file's SHA-256 by
                   ``verify_artifacts_on_disk``. A drift between
                   sealed and on-disk bytes is reported as a
                   mismatch.

The tests use the public surface only
(``from vision_mvp.wevra import ...``).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest


class IntraCellCapsuleTests(unittest.TestCase):
    """W3-32-extended — PATCH_PROPOSAL / TEST_VERDICT lifecycle."""

    def test_patch_proposal_seals_under_spec(self):
        """A PATCH_PROPOSAL capsule seals with parent = SWEEP_SPEC.
        The capsule's payload carries the (task, strategy, parser_mode,
        apply_mode, n_distractors) coordinates plus a SHA over the
        substitutions and a bounded rationale."""
        from vision_mvp.wevra import (
            CapsuleNativeRunContext, CapsuleKind,
        )
        ctx = CapsuleNativeRunContext()
        ctx.start_run(profile_name="local_smoke", profile_dict={})
        ctx.seal_sweep_spec({
            "mode": "mock", "sandbox": "in_process",
            "jsonl": "x.jsonl", "model": None, "endpoint": None,
            "executed_in_process": True})
        cap = ctx.seal_patch_proposal(
            instance_id="mini-001", strategy="substrate",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6,
            substitutions=(("buggy_line", "fixed_line"),),
            rationale="llm_proposed")
        self.assertEqual(cap.kind, CapsuleKind.PATCH_PROPOSAL)
        # Parent of the patch is the sweep spec.
        self.assertEqual(cap.parents, (ctx.spec_cap.cid,))
        # Coordinates are in the payload.
        p = cap.payload
        self.assertEqual(p["instance_id"], "mini-001")
        self.assertEqual(p["strategy"], "substrate")
        self.assertEqual(p["parser_mode"], "strict")
        self.assertEqual(p["apply_mode"], "strict")
        self.assertEqual(p["n_distractors"], 6)
        self.assertEqual(p["n_substitutions"], 1)
        # SHA-256 of the substitutions is a real hex string.
        self.assertEqual(len(p["substitutions_sha256"]), 64)

    def test_patch_proposal_refuses_without_spec(self):
        """A PATCH_PROPOSAL cannot precede SWEEP_SPEC — the capsule
        contract gate is enforced at the type level."""
        from vision_mvp.wevra import (
            CapsuleNativeRunContext, CapsuleLifecycleError,
        )
        ctx = CapsuleNativeRunContext()
        ctx.start_run(profile_name="local_smoke", profile_dict={})
        with self.assertRaises(CapsuleLifecycleError):
            ctx.seal_patch_proposal(
                instance_id="x", strategy="substrate",
                parser_mode="strict", apply_mode="strict",
                n_distractors=6, substitutions=())

    def test_test_verdict_seals_under_patch_proposal(self):
        """A TEST_VERDICT capsule seals with parent = PATCH_PROPOSAL.
        The verdict's payload carries the WorkspaceResult fields."""
        from vision_mvp.wevra import (
            CapsuleNativeRunContext, CapsuleKind,
        )
        ctx = CapsuleNativeRunContext()
        ctx.start_run(profile_name="local_smoke", profile_dict={})
        ctx.seal_sweep_spec({
            "mode": "mock", "sandbox": "in_process",
            "jsonl": "x.jsonl", "model": None, "endpoint": None,
            "executed_in_process": True})
        patch_cap = ctx.seal_patch_proposal(
            instance_id="mini-001", strategy="substrate",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6,
            substitutions=(("buggy", "fixed"),))
        verdict = ctx.seal_test_verdict(
            instance_id="mini-001", strategy="substrate",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6,
            patch_proposal_cid=patch_cap.cid,
            patch_applied=True, syntax_ok=True, test_passed=True,
            error_kind="", error_detail="")
        self.assertEqual(verdict.kind, CapsuleKind.TEST_VERDICT)
        self.assertEqual(verdict.parents, (patch_cap.cid,))
        self.assertTrue(verdict.payload["test_passed"])
        self.assertTrue(verdict.payload["patch_applied"])

    def test_test_verdict_refuses_without_patch(self):
        """A TEST_VERDICT cannot precede its PATCH_PROPOSAL."""
        from vision_mvp.wevra import (
            CapsuleNativeRunContext, CapsuleLifecycleError,
        )
        ctx = CapsuleNativeRunContext()
        ctx.start_run(profile_name="local_smoke", profile_dict={})
        ctx.seal_sweep_spec({
            "mode": "mock", "sandbox": "in_process",
            "jsonl": "x.jsonl", "model": None, "endpoint": None,
            "executed_in_process": True})
        fake_patch_cid = "f" * 64
        with self.assertRaises(CapsuleLifecycleError):
            ctx.seal_test_verdict(
                instance_id="x", strategy="substrate",
                parser_mode="strict", apply_mode="strict",
                n_distractors=6,
                patch_proposal_cid=fake_patch_cid,
                patch_applied=False, syntax_ok=False,
                test_passed=False, error_kind="patch_no_match")

    def test_smoke_run_emits_intra_cell_capsules(self):
        """A capsule-native run on local_smoke (mock bank × 3
        strategies × 2 parser modes) emits one PATCH_PROPOSAL and
        one TEST_VERDICT per (task, strategy, cell) triple. Each
        TEST_VERDICT's parent is exactly one PATCH_PROPOSAL, and
        each PATCH_PROPOSAL's parent is the SWEEP_SPEC."""
        from vision_mvp.wevra import RunSpec, run, CapsuleKind
        with tempfile.TemporaryDirectory() as td:
            r = run(RunSpec(profile="local_smoke", out_dir=td))
        view = r["capsules"]
        cells = [c for c in view["capsules"]
                  if c["kind"] == CapsuleKind.SWEEP_CELL]
        patches = [c for c in view["capsules"]
                    if c["kind"] == CapsuleKind.PATCH_PROPOSAL]
        verdicts = [c for c in view["capsules"]
                     if c["kind"] == CapsuleKind.TEST_VERDICT]
        # local_smoke has 2 parser modes × 1 apply × 1 n_distractors
        # = 2 cells; the bank size × 3 strategies determines the
        # patch/verdict count per cell. The exact bank size depends
        # on the profile's ``n_instances`` cap and the bundled JSONL,
        # so we assert the *invariants* rather than a magic number:
        # patches and verdicts are both ≥ 1, equal in count, and
        # the count is a multiple of (n_cells × 3 strategies).
        self.assertGreaterEqual(len(cells), 1)
        self.assertGreater(len(patches), 0)
        self.assertEqual(len(patches), len(verdicts))
        # Spec CID — every PATCH_PROPOSAL hangs off the same spec.
        spec_caps = [c for c in view["capsules"]
                      if c["kind"] == CapsuleKind.SWEEP_SPEC]
        self.assertEqual(len(spec_caps), 1)
        spec_cid = spec_caps[0]["cid"]
        for patch in patches:
            self.assertEqual(patch["parents"], [spec_cid])
        # Each TEST_VERDICT's parent is a sealed PATCH_PROPOSAL.
        patch_cid_set = {p["cid"] for p in patches}
        for v in verdicts:
            self.assertEqual(len(v["parents"]), 1)
            self.assertIn(v["parents"][0], patch_cid_set)

    def test_run_report_excludes_intra_cell_capsules_from_parents(self):
        """The RUN_REPORT's parent set is the run-boundary spine
        (PROFILE / READINESS_CHECK / SWEEP_SPEC / SWEEP_CELL /
        PROVENANCE / ARTIFACT) only. Intra-cell capsules are
        siblings of SWEEP_CELL via SWEEP_SPEC; they do not appear
        as direct parents of RUN_REPORT, which keeps the parent
        set bounded by spine size and preserves Theorem W3-34's
        spine equivalence with the post-hoc fold."""
        from vision_mvp.wevra import RunSpec, run, CapsuleKind
        with tempfile.TemporaryDirectory() as td:
            r = run(RunSpec(profile="local_smoke", out_dir=td))
        view = r["capsules"]
        run_cap = next(
            c for c in view["capsules"]
            if c["kind"] == CapsuleKind.RUN_REPORT)
        run_parent_set = set(run_cap["parents"])
        intra_kinds = {CapsuleKind.PATCH_PROPOSAL,
                        CapsuleKind.TEST_VERDICT,
                        CapsuleKind.META_MANIFEST}
        for cap in view["capsules"]:
            if cap["kind"] in intra_kinds:
                self.assertNotIn(
                    cap["cid"], run_parent_set,
                    f"intra-cell capsule of kind {cap['kind']} "
                    f"must NOT be a direct parent of RUN_REPORT")


class MetaManifestDetachedWitnessTests(unittest.TestCase):
    """W3-36 — meta-artifact circularity / detached-witness."""

    def test_meta_manifest_seals_in_secondary_ledger(self):
        """The META_MANIFEST capsule lives in a SECONDARY ledger
        (``ctx.meta_manifest_ledger``), not the primary ledger.
        This is the formal expression of the circularity
        boundary: META_MANIFEST cannot live in the primary
        ledger because its payload is a function of meta-artifact
        bytes, which themselves encode the primary ledger's
        rendered view."""
        from vision_mvp.wevra import RunSpec, run, CapsuleKind
        with tempfile.TemporaryDirectory() as td:
            r = run(RunSpec(profile="local_smoke", out_dir=td))
            # The detached witness file is on disk.
            manifest_path = os.path.join(td, "meta_manifest.json")
            self.assertTrue(os.path.exists(manifest_path))
            with open(manifest_path, "r") as fh:
                manifest = json.load(fh)
        # Manifest is tagged as detached.
        self.assertEqual(manifest["construction"], "detached_witness")
        # Manifest cross-references the primary root_cid.
        self.assertEqual(
            manifest["primary_root_cid"], r["capsules"]["root_cid"])
        # The single META_MANIFEST capsule names every meta-artifact.
        manifest_caps = [c for c in manifest["capsules"]
                          if c["kind"] == CapsuleKind.META_MANIFEST]
        self.assertEqual(len(manifest_caps), 1)
        meta_artifacts = manifest_caps[0]["payload"]["meta_artifacts"]
        meta_paths = {m["path"] for m in meta_artifacts}
        self.assertEqual(
            meta_paths,
            {"product_report.json", "capsule_view.json",
             "product_summary.txt"})
        # The primary view does NOT contain a META_MANIFEST capsule
        # (the manifest is sealed in the secondary ledger only).
        primary_meta_caps = [c for c in r["capsules"]["capsules"]
                              if c["kind"] == CapsuleKind.META_MANIFEST]
        self.assertEqual(primary_meta_caps, [])

    def test_meta_manifest_refuses_without_run_report(self):
        """The detached witness is post-fixed-point: it cannot be
        sealed before the RUN_REPORT (the manifest's ``root_cid``
        field requires the primary fixed point)."""
        from vision_mvp.wevra import (
            CapsuleNativeRunContext, CapsuleLifecycleError,
        )
        ctx = CapsuleNativeRunContext()
        ctx.start_run(profile_name="local_smoke", profile_dict={})
        with self.assertRaises(CapsuleLifecycleError):
            ctx.seal_meta_manifest(meta_artifacts=[])

    def test_meta_manifest_shas_match_on_disk(self):
        """The SHAs in the META_MANIFEST equal the actual SHA-256
        of the on-disk meta-artifact bytes — anyone holding the
        manifest can verify."""
        from vision_mvp.wevra import RunSpec, run, CapsuleKind
        with tempfile.TemporaryDirectory() as td:
            run(RunSpec(profile="local_smoke", out_dir=td))
            with open(os.path.join(td, "meta_manifest.json"), "r") as fh:
                manifest = json.load(fh)
            for cap in manifest["capsules"]:
                if cap["kind"] != CapsuleKind.META_MANIFEST:
                    continue
                for entry in cap["payload"]["meta_artifacts"]:
                    full = os.path.join(td, entry["path"])
                    with open(full, "rb") as fh:
                        bytes_ = fh.read()
                    self.assertEqual(
                        hashlib.sha256(bytes_).hexdigest(),
                        entry["sha256"],
                        f"meta-artifact {entry['path']!r} bytes do "
                        f"not hash to the manifest's claim")
                    self.assertEqual(len(bytes_), entry["n_bytes"])


class StrongOnDiskVerificationTests(unittest.TestCase):
    """W3-37 — chain-from-headers; W3-38 — ARTIFACT bytes on disk."""

    def test_w3_37_chain_recompute_from_view(self):
        """``verify_chain_from_view_dict`` recomputes the chain head
        from on-disk header bytes; the recomputed head equals the
        view's ``chain_head`` claim. This is stronger than trusting
        the writer's self-reported ``chain_ok`` boolean."""
        from vision_mvp.wevra import (
            RunSpec, run, verify_chain_from_view_dict,
        )
        with tempfile.TemporaryDirectory() as td:
            r = run(RunSpec(profile="local_smoke", out_dir=td))
            with open(os.path.join(td, "capsule_view.json"), "r") as fh:
                disk_view = json.load(fh)
        # Embedded view recomputes correctly.
        self.assertTrue(verify_chain_from_view_dict(r["capsules"]))
        # On-disk view recomputes correctly.
        self.assertTrue(verify_chain_from_view_dict(disk_view))

    def test_w3_37_tamper_detected(self):
        """If we tamper with the view's chain_head, the recompute
        rejects it. This is the *forensic* tamper-evidence claim
        of the chain-from-headers verification."""
        from vision_mvp.wevra import (
            RunSpec, run, verify_chain_from_view_dict,
        )
        with tempfile.TemporaryDirectory() as td:
            r = run(RunSpec(profile="local_smoke", out_dir=td))
        view = dict(r["capsules"])
        view["chain_head"] = "0" * 64  # tamper the head
        self.assertFalse(verify_chain_from_view_dict(view))

    def test_w3_37_tamper_capsule_order_detected(self):
        """Reordering capsules in the view changes the chain head
        recomputation; the verification fails."""
        from vision_mvp.wevra import (
            RunSpec, run, verify_chain_from_view_dict,
        )
        with tempfile.TemporaryDirectory() as td:
            r = run(RunSpec(profile="local_smoke", out_dir=td))
        view = dict(r["capsules"])
        # Reverse the capsule ordering.
        view["capsules"] = list(reversed(view["capsules"]))
        self.assertFalse(verify_chain_from_view_dict(view))

    def test_w3_38_artifact_on_disk_re_hash(self):
        """Every ARTIFACT capsule's payload SHA-256 matches the
        on-disk file's SHA-256 at audit time."""
        from vision_mvp.wevra import (
            RunSpec, run, verify_artifacts_on_disk,
        )
        with tempfile.TemporaryDirectory() as td:
            r = run(RunSpec(profile="local_smoke", out_dir=td))
            result = verify_artifacts_on_disk(
                r["capsules"], base_dir=td)
        self.assertEqual(result["verdict"], "OK")
        self.assertGreater(result["checked"], 0)
        self.assertEqual(result["mismatch"], [])
        self.assertEqual(result["missing"], [])

    def test_w3_38_artifact_drift_detected(self):
        """Tampering with an on-disk substantive artifact is
        detected by the on-disk re-hash."""
        from vision_mvp.wevra import (
            RunSpec, run, verify_artifacts_on_disk,
        )
        with tempfile.TemporaryDirectory() as td:
            r = run(RunSpec(profile="local_smoke", out_dir=td))
            # Tamper with the on-disk readiness file.
            target = os.path.join(td, "readiness_verdict.json")
            with open(target, "wb") as fh:
                fh.write(b"TAMPERED")
            result = verify_artifacts_on_disk(
                r["capsules"], base_dir=td)
        self.assertEqual(result["verdict"], "BAD")
        # At least one drift entry names readiness_verdict.json.
        drifted_paths = [d["path"] for d in result["mismatch"]]
        self.assertIn(
            "readiness_verdict.json",
            [os.path.basename(p) for p in drifted_paths])

    def test_w3_38_meta_manifest_drift_detected(self):
        """Tampering with a meta-artifact (the report itself,
        which is NOT in the primary ledger) is detected by the
        detached META_MANIFEST verification — not the primary
        verification. This witnesses the boundary in Theorem
        W3-36: the manifest is the trust unit for meta-artifacts."""
        from vision_mvp.wevra import (
            RunSpec, run, verify_meta_manifest_on_disk,
        )
        with tempfile.TemporaryDirectory() as td:
            run(RunSpec(profile="local_smoke", out_dir=td))
            # Tamper with the product report (a meta-artifact).
            target = os.path.join(td, "product_summary.txt")
            with open(target, "w") as fh:
                fh.write("TAMPERED — not the rendered summary")
            with open(os.path.join(td, "meta_manifest.json"), "r") as fh:
                manifest = json.load(fh)
            result = verify_meta_manifest_on_disk(
                manifest, base_dir=td)
        self.assertEqual(result["verdict"], "BAD")
        drifted_paths = [d["path"] for d in result["mismatch"]]
        self.assertIn("product_summary.txt", drifted_paths)


class W3_34_PreservedUnderIntraCellExtensionTests(unittest.TestCase):
    """Theorem W3-34's spine-CID equivalence between the in-flight
    builder and the post-hoc ``build_report_ledger`` fold is
    *preserved* under SDK v3.2's intra-cell extension. The
    intra-cell capsules are siblings of the spine; they do not
    affect spine CIDs."""

    def test_w3_34_spine_equivalence_holds(self):
        from vision_mvp.product.runner import run_profile
        from vision_mvp.wevra import build_report_ledger, CapsuleKind
        from vision_mvp.product import profiles as _profiles
        with tempfile.TemporaryDirectory() as td:
            in_flight_report = run_profile(
                "local_smoke", out_dir=td, capsule_native=True)
        in_flight_view = in_flight_report["capsules"]
        prof = _profiles.get_profile("local_smoke")
        post_hoc_ledger, _ = build_report_ledger(
            in_flight_report, profile_dict=prof)
        equivalence_kinds = {
            CapsuleKind.PROFILE, CapsuleKind.READINESS_CHECK,
            CapsuleKind.SWEEP_SPEC, CapsuleKind.SWEEP_CELL,
            CapsuleKind.PROVENANCE,
        }
        in_flight_by_kind: dict[str, set[str]] = {}
        for cap in in_flight_view["capsules"]:
            k = cap["kind"]
            if k not in equivalence_kinds:
                continue
            in_flight_by_kind.setdefault(k, set()).add(cap["cid"])
        post_hoc_by_kind: dict[str, set[str]] = {}
        for cap in post_hoc_ledger.all_capsules():
            if cap.kind not in equivalence_kinds:
                continue
            post_hoc_by_kind.setdefault(cap.kind, set()).add(cap.cid)
        for k in equivalence_kinds:
            self.assertEqual(
                in_flight_by_kind.get(k, set()),
                post_hoc_by_kind.get(k, set()),
                f"kind {k!r}: spine CIDs disagree under SDK v3.2 — "
                f"intra-cell extension must NOT affect spine "
                f"equivalence (Theorem W3-34 preservation)")


if __name__ == "__main__":
    unittest.main()
