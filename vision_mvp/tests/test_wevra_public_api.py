"""Wevra SDK public-surface contract tests.

These tests LOCK the public API of ``vision_mvp.wevra``. Any symbol
removed or renamed here is a breaking SDK change and must bump the
``SDK_VERSION`` in ``vision_mvp/wevra/__init__.py``.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
import unittest


class WevraSurfaceTests(unittest.TestCase):
    """Lock the set of public symbols on the ``wevra`` package."""

    REQUIRED_SYMBOLS = (
        "RunSpec", "run",
        "SweepSpec", "run_sweep", "HeavyRunNotAcknowledged",
        # Capsule primitives (SDK v3 — the load-bearing abstraction).
        "ContextCapsule", "CapsuleKind", "CapsuleLifecycle",
        "CapsuleBudget", "CapsuleLedger", "CapsuleView",
        "CapsuleAdmissionError", "CapsuleLifecycleError",
        "CAPSULE_VIEW_SCHEMA", "render_view", "build_report_ledger",
        "WevraConfig",
        "PROVENANCE_SCHEMA", "build_manifest",
        "profiles", "report", "ci_gate", "import_data", "extensions",
        "__version__", "SDK_VERSION",
        "PRODUCT_REPORT_SCHEMA", "PRODUCT_REPORT_SCHEMA_V1",
        "CI_VERDICT_SCHEMA", "IMPORT_AUDIT_SCHEMA",
    )

    def test_top_level_importable(self):
        import vision_mvp.wevra as w
        for name in self.REQUIRED_SYMBOLS:
            self.assertTrue(hasattr(w, name), f"missing public symbol: {name}")

    def test_all_is_exhaustive(self):
        import vision_mvp.wevra as w
        for name in self.REQUIRED_SYMBOLS:
            self.assertIn(name, w.__all__, f"{name} must be in __all__")

    def test_schema_constants_are_strings(self):
        from vision_mvp.wevra import (
            PROVENANCE_SCHEMA, PRODUCT_REPORT_SCHEMA,
            CI_VERDICT_SCHEMA, IMPORT_AUDIT_SCHEMA, SDK_VERSION,
        )
        for v in (PROVENANCE_SCHEMA, PRODUCT_REPORT_SCHEMA,
                   CI_VERDICT_SCHEMA, IMPORT_AUDIT_SCHEMA, SDK_VERSION):
            self.assertIsInstance(v, str)
            self.assertTrue(len(v) > 0)

    def test_sdk_version_is_v3(self):
        from vision_mvp.wevra import SDK_VERSION
        # Slice 3 bump: the Context Capsule primitive becomes the
        # load-bearing abstraction. Additive on the v2 surface
        # (RunSpec / SweepSpec / run_sweep / extensions all
        # unchanged); the new surface is capsule.* + a ``capsules``
        # block on the product report.
        self.assertEqual(SDK_VERSION, "wevra.sdk.v3")

    def test_runspec_is_frozen_dataclass(self):
        from vision_mvp.wevra import RunSpec
        self.assertTrue(dataclasses.is_dataclass(RunSpec))
        self.assertTrue(
            RunSpec.__dataclass_params__.frozen,
            "RunSpec must be frozen so runtime cannot mutate it")
        fields = {f.name for f in dataclasses.fields(RunSpec)}
        self.assertSetEqual(fields, {
            "profile", "out_dir", "jsonl_override",
            "skip_sweep", "force_sweep",
            "acknowledge_heavy", "allow_unsafe_sandbox",
            "report_sinks", "config",
        })

    def test_wevra_config_frozen_and_env_overridable(self):
        from vision_mvp.wevra import WevraConfig
        cfg = WevraConfig()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg.model = "x"  # type: ignore[misc]
        # Invalid sandbox is rejected at construction.
        with self.assertRaises(ValueError):
            WevraConfig(sandbox="nonsense")

    def test_wevra_config_from_env(self):
        from vision_mvp.wevra import WevraConfig
        os.environ["WEVRA_MODEL"] = "test-model:1b"
        os.environ["WEVRA_SANDBOX"] = "in_process"
        try:
            cfg = WevraConfig.from_env()
            self.assertEqual(cfg.model, "test-model:1b")
            self.assertEqual(cfg.sandbox, "in_process")
            # kwargs override env.
            cfg2 = WevraConfig.from_env(model="override:7b")
            self.assertEqual(cfg2.model, "override:7b")
        finally:
            del os.environ["WEVRA_MODEL"]
            del os.environ["WEVRA_SANDBOX"]

    def test_profiles_are_re_exported(self):
        from vision_mvp.wevra import profiles
        names = profiles.list_profiles()
        for required in ("local_smoke", "bundled_57", "public_jsonl"):
            self.assertIn(required, names)

    def test_run_spec_round_trip(self):
        from vision_mvp.wevra import RunSpec, run, CAPSULE_VIEW_SCHEMA
        with tempfile.TemporaryDirectory() as td:
            report = run(RunSpec(profile="local_smoke", out_dir=td))
            self.assertTrue(report["readiness"]["ready"])
            self.assertIn("provenance", report)
            self.assertEqual(
                report["provenance"]["schema"], "wevra.provenance.v1")
            # Provenance file on disk.
            self.assertTrue(os.path.exists(
                os.path.join(td, "provenance.json")))
            # SDK-v3 capsule view is load-bearing: every run
            # ships one both in-report and on-disk.
            self.assertIn("capsules", report)
            self.assertEqual(
                report["capsules"]["schema"], CAPSULE_VIEW_SCHEMA)
            self.assertTrue(report["capsules"]["chain_ok"])
            self.assertIsNotNone(report["capsules"]["root_cid"])
            self.assertTrue(os.path.exists(
                os.path.join(td, "capsule_view.json")))

    def test_product_path_still_works_for_backcompat(self):
        # The ``vision_mvp.product`` path must still function —
        # consumers of the old surface are not broken by Slice 1.
        from vision_mvp.product.runner import run_profile
        with tempfile.TemporaryDirectory() as td:
            rep = run_profile("local_smoke", out_dir=td)
            self.assertTrue(rep["readiness"]["ready"])
            # And it gets provenance too.
            self.assertEqual(
                rep["provenance"]["schema"], "wevra.provenance.v1")


if __name__ == "__main__":
    unittest.main()
