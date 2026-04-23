"""Contract tests for the Wevra extension system.

Locks the public extension surface: three runtime-checkable
Protocols, each with an in-process registry, plus
``entry_points``-based discovery. These tests cover the end-to-end
path: register → list → resolve → invoke.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest


class SandboxExtensionTests(unittest.TestCase):

    def test_builtins_registered_lazily(self):
        from vision_mvp.wevra import extensions
        names = extensions.list_sandboxes()
        for required in ("in_process", "subprocess", "docker"):
            self.assertIn(required, names)

    def test_get_builtin_sandbox_satisfies_protocol(self):
        from vision_mvp.wevra.extensions import (
            SandboxBackend, get_sandbox,
        )
        inst = get_sandbox("in_process")
        self.assertIsInstance(inst, SandboxBackend)
        self.assertEqual(inst.name(), "in_process")
        self.assertTrue(inst.is_available())

    def test_register_custom_sandbox_roundtrip(self):
        from vision_mvp.wevra.extensions import (
            SandboxBackend, register_sandbox, get_sandbox,
            list_sandboxes,
        )

        class _CountingSandbox:
            def __init__(self) -> None:
                self.calls = 0

            def name(self) -> str:
                return "counting"

            def is_available(self) -> bool:
                return True

            def run(self, **kwargs):
                self.calls += 1
                return {"called": self.calls}

        register_sandbox("counting", _CountingSandbox, overwrite=True)
        self.assertIn("counting", list_sandboxes())
        inst = get_sandbox("counting")
        self.assertIsInstance(inst, SandboxBackend)
        self.assertEqual(inst.name(), "counting")

    def test_duplicate_registration_rejected(self):
        from vision_mvp.wevra.extensions import register_sandbox
        register_sandbox("dup", lambda **_: object(), overwrite=True)
        with self.assertRaises(ValueError):
            register_sandbox("dup", lambda **_: object())

    def test_unknown_sandbox_raises(self):
        from vision_mvp.wevra.extensions import get_sandbox
        with self.assertRaises(KeyError):
            get_sandbox("definitely-not-a-sandbox")


class TaskBankExtensionTests(unittest.TestCase):

    def test_builtin_jsonl_loader_exists(self):
        from vision_mvp.wevra.extensions import list_task_banks
        self.assertIn("jsonl", list_task_banks())

    def test_builtin_jsonl_loader_loads_bundled_bank(self):
        from vision_mvp.wevra.extensions import (
            TaskBankLoader, get_task_bank,
        )
        loader = get_task_bank("jsonl")
        self.assertIsInstance(loader, TaskBankLoader)
        bundled = os.path.join(
            os.path.dirname(__file__), "..", "tasks", "data",
            "swe_lite_style_bank.jsonl")
        bundle = loader.load(bundled, limit=3)
        self.assertEqual(bundle.schema, "wevra.task_bank.v1")
        self.assertEqual(bundle.source, bundled)
        self.assertIsInstance(bundle.repo_files, dict)
        # Tasks are materialized; bundled bank has at least 3 rows.
        tasks = list(bundle.tasks)
        self.assertGreaterEqual(len(tasks), 1)


class ReportSinkExtensionTests(unittest.TestCase):

    def test_builtins_registered(self):
        from vision_mvp.wevra.extensions import list_report_sinks
        names = list_report_sinks()
        for required in ("stdout", "jsonfile"):
            self.assertIn(required, names)

    def test_jsonfile_sink_writes_report(self):
        from vision_mvp.wevra.extensions import get_report_sink
        with tempfile.TemporaryDirectory() as td:
            sink = get_report_sink("jsonfile", path=os.path.join(td, "r.json"))
            out = sink.emit({"schema": "x", "profile": "y"})
            self.assertTrue(out["ok"])
            with open(out["wrote"], "r", encoding="utf-8") as fh:
                got = json.load(fh)
            self.assertEqual(got["profile"], "y")

    def test_worked_example_sink_end_to_end(self):
        from vision_mvp.wevra.extensions import register_report_sink
        from vision_mvp.wevra.extensions.examples.jsonl_report_sink import (
            JsonlWithMetaSink,
        )
        register_report_sink(
            "jsonl_with_meta", JsonlWithMetaSink, overwrite=True)
        from vision_mvp.wevra import RunSpec, run
        with tempfile.TemporaryDirectory() as td:
            sink_target = os.path.join(td, "external.json")
            # Register a sink factory that binds `path` to sink_target,
            # then drive it through RunSpec.report_sinks.
            register_report_sink(
                "jsonl_with_meta",
                (lambda **_kw: JsonlWithMetaSink(path=sink_target)),
                overwrite=True)
            rep = run(RunSpec(
                profile="local_smoke",
                out_dir=td,
                report_sinks=("jsonl_with_meta",)))
            # Sink ran.
            self.assertEqual(len(rep["sink_emissions"]), 1)
            emission = rep["sink_emissions"][0]
            self.assertEqual(emission["sink"], "jsonl_with_meta")
            self.assertTrue(emission["ok"])
            # Its output + its meta sidecar both exist on disk.
            self.assertTrue(os.path.exists(sink_target))
            self.assertTrue(os.path.exists(sink_target + ".meta.json"))
            with open(sink_target + ".meta.json", "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            self.assertEqual(meta["profile"], "local_smoke")
            self.assertEqual(
                meta["provenance_schema"], "wevra.provenance.v1")


class RegistryDiscoveryTests(unittest.TestCase):

    def test_all_extensions_reports_three_groups(self):
        from vision_mvp.wevra.extensions import all_extensions
        ext = all_extensions()
        self.assertIn("wevra.sandboxes", ext)
        self.assertIn("wevra.task_banks", ext)
        self.assertIn("wevra.report_sinks", ext)

    def test_discover_entry_points_returns_shape(self):
        # No third-party plugins are installed, but the discovery
        # machinery should still run clean and return the expected
        # shape.
        from vision_mvp.wevra.extensions import discover_entry_points
        result = discover_entry_points()
        self.assertIn("discovered", result)
        self.assertIn("errors", result)
        self.assertIn("wevra.sandboxes", result["discovered"])


if __name__ == "__main__":
    unittest.main()
