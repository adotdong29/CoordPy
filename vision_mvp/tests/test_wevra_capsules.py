"""Context Capsule contract tests (SDK v3).

These tests lock in the six-invariant Capsule Contract (C1..C6) that
Wevra v3 centres on:

  C1  Identity       — CID is deterministic over (kind, payload,
                        budget, parents).
  C2  Typed claim    — every capsule has a known ``CapsuleKind``.
  C3  Lifecycle      — PROPOSED → ADMITTED → SEALED (+ optional
                        RETIRED); illegal transitions are rejected.
  C4  Budget         — admission rejects over-budget capsules.
  C5  Provenance     — parents must be in the ledger; retroactive
                        mutation breaks the hash chain.
  C6  Frozen         — once SEALED, CID and payload are fixed.

The tests use the public surface only (``from vision_mvp.wevra
import ...``) so they double as a demo of SDK v3.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest


class CapsuleContractTests(unittest.TestCase):

    def test_c1_identity_deterministic(self):
        from vision_mvp.wevra import (
            ContextCapsule, CapsuleKind, CapsuleBudget,
        )
        c1 = ContextCapsule.new(
            kind=CapsuleKind.HANDOFF,
            payload={"msg": "hi", "n": 1},
            budget=CapsuleBudget(max_tokens=64))
        c2 = ContextCapsule.new(
            kind=CapsuleKind.HANDOFF,
            payload={"n": 1, "msg": "hi"},  # key order swapped
            budget=CapsuleBudget(max_tokens=64))
        self.assertEqual(c1.cid, c2.cid)
        # Different content → different CID.
        c3 = ContextCapsule.new(
            kind=CapsuleKind.HANDOFF,
            payload={"msg": "bye", "n": 1},
            budget=CapsuleBudget(max_tokens=64))
        self.assertNotEqual(c1.cid, c3.cid)

    def test_c1_parent_order_is_canonicalised(self):
        from vision_mvp.wevra import (
            ContextCapsule, CapsuleKind, CapsuleBudget,
        )
        # Same parent set, different insertion order → same CID.
        a = ContextCapsule.new(
            kind=CapsuleKind.HANDOFF, payload={"k": 1},
            budget=CapsuleBudget(max_tokens=16), parents=())
        b = ContextCapsule.new(
            kind=CapsuleKind.HANDOFF, payload={"k": 2},
            budget=CapsuleBudget(max_tokens=16), parents=())
        child_ab = ContextCapsule.new(
            kind=CapsuleKind.THREAD_RESOLUTION,
            payload={"resolved": True},
            budget=CapsuleBudget(max_rounds=1),
            parents=(a.cid, b.cid))
        child_ba = ContextCapsule.new(
            kind=CapsuleKind.THREAD_RESOLUTION,
            payload={"resolved": True},
            budget=CapsuleBudget(max_rounds=1),
            parents=(b.cid, a.cid))
        self.assertEqual(child_ab.cid, child_ba.cid)

    def test_c2_unknown_kind_rejected(self):
        from vision_mvp.wevra import ContextCapsule, CapsuleBudget
        with self.assertRaises(ValueError):
            ContextCapsule.new(
                kind="NOT_A_REAL_KIND",
                payload={},
                budget=CapsuleBudget(max_bytes=64))

    def test_c3_lifecycle_order(self):
        from vision_mvp.wevra import (
            ContextCapsule, CapsuleKind, CapsuleBudget,
            CapsuleLedger, CapsuleLifecycle, CapsuleLifecycleError,
        )
        lg = CapsuleLedger()
        c = ContextCapsule.new(
            kind=CapsuleKind.HANDOFF, payload={"x": 1},
            budget=CapsuleBudget(max_tokens=16))
        self.assertEqual(c.lifecycle, CapsuleLifecycle.PROPOSED)
        c_adm = lg.admit(c)
        self.assertEqual(c_adm.lifecycle, CapsuleLifecycle.ADMITTED)
        # Admitting an already-ADMITTED capsule is illegal.
        with self.assertRaises(CapsuleLifecycleError):
            lg.admit(c_adm)
        c_sealed = lg.seal(c_adm)
        self.assertEqual(c_sealed.lifecycle, CapsuleLifecycle.SEALED)
        # Sealing a PROPOSED (or already SEALED) capsule is illegal.
        with self.assertRaises(CapsuleLifecycleError):
            lg.seal(c)
        # RETIRED is a legal annotation from SEALED only.
        c_ret = lg.retire(c_sealed.cid)
        self.assertEqual(c_ret.lifecycle, CapsuleLifecycle.RETIRED)

    def test_c3_idempotent_admit_and_seal(self):
        from vision_mvp.wevra import (
            ContextCapsule, CapsuleKind, CapsuleBudget, CapsuleLedger,
        )
        lg = CapsuleLedger()
        c = ContextCapsule.new(
            kind=CapsuleKind.HANDOFF, payload={"k": "v"},
            budget=CapsuleBudget(max_tokens=16))
        c_sealed = lg.admit_and_seal(c)
        # Admitting the same CID twice returns the sealed copy.
        c_again = lg.admit(c)
        self.assertEqual(c_again.cid, c_sealed.cid)
        # Ledger has only one entry.
        self.assertEqual(len(lg), 1)

    def test_c4_budget_rejects_over_tokens(self):
        from vision_mvp.wevra import (
            ContextCapsule, CapsuleKind, CapsuleBudget,
            CapsuleLedger, CapsuleAdmissionError,
        )
        lg = CapsuleLedger()
        c = ContextCapsule.new(
            kind=CapsuleKind.HANDOFF,
            payload={"msg": "x"},
            budget=CapsuleBudget(max_tokens=2),
            n_tokens=9)  # deliberately over budget
        with self.assertRaises(CapsuleAdmissionError):
            lg.admit(c)

    def test_c4_budget_rejects_over_bytes_at_construction(self):
        from vision_mvp.wevra import (
            ContextCapsule, CapsuleKind, CapsuleBudget,
        )
        # max_bytes is checked at construction (so we can't even
        # forge an over-budget capsule).
        with self.assertRaises(ValueError):
            ContextCapsule.new(
                kind=CapsuleKind.ARTIFACT,
                payload={"blob": "x" * 4096},
                budget=CapsuleBudget(max_bytes=128))

    def test_c4_empty_budget_is_illegal(self):
        from vision_mvp.wevra import CapsuleBudget
        with self.assertRaises(ValueError):
            CapsuleBudget()

    def test_c5_parent_must_be_in_ledger(self):
        from vision_mvp.wevra import (
            ContextCapsule, CapsuleKind, CapsuleBudget,
            CapsuleLedger, CapsuleAdmissionError,
        )
        lg = CapsuleLedger()
        fake_cid = "f" * 64
        c = ContextCapsule.new(
            kind=CapsuleKind.THREAD_RESOLUTION,
            payload={"resolved": True},
            budget=CapsuleBudget(max_rounds=1),
            parents=(fake_cid,))
        with self.assertRaises(CapsuleAdmissionError):
            lg.admit(c)

    def test_c5_hash_chain_detects_tamper(self):
        from vision_mvp.wevra import (
            ContextCapsule, CapsuleKind, CapsuleBudget, CapsuleLedger,
        )
        lg = CapsuleLedger()
        a = ContextCapsule.new(
            kind=CapsuleKind.HANDOFF, payload={"x": 1},
            budget=CapsuleBudget(max_tokens=16))
        b = ContextCapsule.new(
            kind=CapsuleKind.HANDOFF, payload={"x": 2},
            budget=CapsuleBudget(max_tokens=16))
        lg.admit_and_seal(a)
        lg.admit_and_seal(b)
        self.assertTrue(lg.verify_chain())
        # Tamper with the log by rewriting the second entry's
        # chain_hash.
        entries = lg._entries  # type: ignore[attr-defined]
        entries[1] = type(entries[1])(
            capsule=entries[1].capsule,
            chain_hash="0" * 64,
            prev_chain_hash=entries[1].prev_chain_hash,
        )
        self.assertFalse(lg.verify_chain())

    def test_c6_sealed_capsule_is_frozen(self):
        from vision_mvp.wevra import (
            ContextCapsule, CapsuleKind, CapsuleBudget, CapsuleLedger,
        )
        import dataclasses
        lg = CapsuleLedger()
        c = ContextCapsule.new(
            kind=CapsuleKind.HANDOFF, payload={"x": 1},
            budget=CapsuleBudget(max_tokens=16))
        c_sealed = lg.admit_and_seal(c)
        # The sealed dataclass is frozen — you can't mutate it.
        with self.assertRaises(dataclasses.FrozenInstanceError):
            c_sealed.payload = {"x": 999}  # type: ignore[misc]

    def test_view_schema_and_chain_ok(self):
        from vision_mvp.wevra import (
            ContextCapsule, CapsuleKind, CapsuleBudget,
            CapsuleLedger, render_view, CAPSULE_VIEW_SCHEMA,
        )
        lg = CapsuleLedger()
        for i in range(3):
            lg.admit_and_seal(ContextCapsule.new(
                kind=CapsuleKind.HANDOFF,
                payload={"i": i},
                budget=CapsuleBudget(max_tokens=16)))
        view = render_view(lg)
        self.assertEqual(view.schema, CAPSULE_VIEW_SCHEMA)
        self.assertTrue(view.chain_ok)
        self.assertEqual(len(view.capsules), 3)
        # Header-view by default has no payload leak.
        self.assertNotIn("payload", view.capsules[0])

    def test_adapter_handle_to_capsule(self):
        from vision_mvp.wevra import (
            capsule_from_handle, CapsuleKind, CapsuleLedger,
        )
        from vision_mvp.core.context_ledger import (
            ContextLedger, hash_embedding,
        )
        cl = ContextLedger(embed_dim=32, embed_fn=lambda s: hash_embedding(s, 32))
        h = cl.put("incident 42 — disk near full",
                     metadata={"doc_id": "i42"})
        cap = capsule_from_handle(h)
        self.assertEqual(cap.kind, CapsuleKind.HANDLE)
        self.assertEqual(cap.metadata_dict()["handle_cid"], h.cid)
        # Capsule can be sealed into a capsule ledger.
        lg = CapsuleLedger()
        sealed = lg.admit_and_seal(cap)
        self.assertEqual(sealed.kind, CapsuleKind.HANDLE)

    def test_adapter_handoff_to_capsule(self):
        from vision_mvp.wevra import (
            capsule_from_handoff, CapsuleKind, CapsuleLedger,
        )
        from vision_mvp.core.role_handoff import HandoffLog
        log = HandoffLog()
        h = log.emit(
            source_role="dba", source_agent_id=0,
            to_role="auditor", claim_kind="SLOW_QUERY_OBSERVED",
            payload="pg_stat_statements#12 mean_ms=4200",
            source_event_ids=[42, 43], round=1)
        cap = capsule_from_handoff(h)
        self.assertEqual(cap.kind, CapsuleKind.HANDOFF)
        self.assertEqual(
            cap.metadata_dict()["claim_kind"], "SLOW_QUERY_OBSERVED")
        # Substrate chain-hash is cross-referenced in the capsule's
        # metadata so the two logs can be audited together.
        self.assertEqual(
            cap.metadata_dict()["handoff_chain_hash"], h.chain_hash)
        lg = CapsuleLedger()
        sealed = lg.admit_and_seal(cap)
        self.assertEqual(sealed.kind, CapsuleKind.HANDOFF)

    def test_build_report_ledger_from_smoke_run(self):
        # The load-bearing end-to-end: a smoke run must produce a
        # sealed capsule graph rooted at a RUN_REPORT capsule.
        from vision_mvp.wevra import (
            RunSpec, run, build_report_ledger, CapsuleKind,
        )
        with tempfile.TemporaryDirectory() as td:
            report = run(RunSpec(profile="local_smoke", out_dir=td))
            ledger, run_cid = build_report_ledger(report)
            self.assertTrue(ledger.verify_chain())
            # RUN_REPORT is sealed and is the root CID.
            run_cap = ledger.get(run_cid)
            self.assertEqual(run_cap.kind, CapsuleKind.RUN_REPORT)
            self.assertEqual(run_cap.lifecycle, "SEALED")
            # Profile is an ancestor.
            kinds_up = {c.kind for c in ledger.ancestors_of(run_cid)}
            self.assertIn(CapsuleKind.PROFILE, kinds_up)
            self.assertIn(CapsuleKind.PROVENANCE, kinds_up)
            self.assertIn(CapsuleKind.READINESS_CHECK, kinds_up)

    def test_capsule_view_on_disk_matches_embedded(self):
        from vision_mvp.wevra import RunSpec, run
        with tempfile.TemporaryDirectory() as td:
            report = run(RunSpec(profile="local_smoke", out_dir=td))
            view_path = os.path.join(td, "capsule_view.json")
            with open(view_path, "r", encoding="utf-8") as fh:
                disk = json.load(fh)
            embedded = report["capsules"]
            self.assertEqual(disk["schema"], embedded["schema"])
            self.assertEqual(disk["chain_head"], embedded["chain_head"])
            self.assertEqual(disk["root_cid"], embedded["root_cid"])
            self.assertEqual(
                len(disk["capsules"]), len(embedded["capsules"]))


class CapsuleCLITests(unittest.TestCase):
    """Exercise the ``wevra-capsule`` CLI."""

    def test_cli_cid_and_view(self):
        from vision_mvp.wevra._cli import _cmd_capsule
        from vision_mvp.wevra import RunSpec, run
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as td:
            run(RunSpec(profile="local_smoke", out_dir=td))
            report_path = os.path.join(td, "product_report.json")
            # cid subcommand — prints the RUN_REPORT CID.
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = _cmd_capsule(["cid", "--report", report_path])
            self.assertEqual(rc, 0)
            cid = buf.getvalue().strip()
            self.assertEqual(len(cid), 64)
            # view subcommand — prints a summary.
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = _cmd_capsule(["view", "--report", report_path])
            self.assertEqual(rc, 0)
            self.assertIn("capsule graph", buf.getvalue())
            self.assertIn("chain_ok", buf.getvalue())
            # verify subcommand — returns 0 when every check passes.
            # SDK v3.2 strengthened verify to four independent
            # on-disk checks; the output prints each line plus a
            # final ``verdict = OK / BAD`` summary.
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = _cmd_capsule(["verify", "--report", report_path])
            self.assertEqual(rc, 0)
            output = buf.getvalue()
            # Final overall verdict is OK.
            self.assertRegex(output, r"verdict\s*=\s*OK")
            # Each named check is present.
            self.assertIn("chain_recompute_embedded", output)
            self.assertIn("chain_recompute_on_disk", output)
            self.assertIn("artifacts_on_disk", output)


if __name__ == "__main__":
    unittest.main()
