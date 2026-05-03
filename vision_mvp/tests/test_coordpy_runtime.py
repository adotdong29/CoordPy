"""Contract tests for the CoordPy unified runtime (Slice 2).

Covers ``SweepSpec`` validation, mock execution path, real-mode
staging (acknowledge_heavy=False), strict cost gate, env-var
endpoint overrides, and the v2 report schema.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
import unittest


class SweepSpecTests(unittest.TestCase):

    def test_sweepspec_is_frozen(self):
        from vision_mvp.coordpy import SweepSpec
        self.assertTrue(dataclasses.is_dataclass(SweepSpec))
        self.assertTrue(SweepSpec.__dataclass_params__.frozen)

    def test_real_mode_without_model_rejected(self):
        from vision_mvp.coordpy import SweepSpec
        with self.assertRaises(ValueError):
            SweepSpec(mode="real", jsonl="x.jsonl", model=None)

    def test_bad_mode_rejected(self):
        from vision_mvp.coordpy import SweepSpec
        with self.assertRaises(ValueError):
            SweepSpec(mode="fast", jsonl="x.jsonl")


class MockRunSweepTests(unittest.TestCase):

    BUNDLED = os.path.join(
        os.path.dirname(__file__), "..", "tasks", "data",
        "swe_lite_style_bank.jsonl")

    def test_mock_run_sweep_returns_v2_schema(self):
        from vision_mvp.coordpy import SweepSpec, run_sweep
        spec = SweepSpec(
            mode="mock", jsonl=self.BUNDLED,
            sandbox="in_process", parser_modes=("strict",),
            apply_modes=("strict",), n_distractors=(6,),
            n_instances=4)
        sw = run_sweep(spec)
        self.assertEqual(sw["schema"], "coordpy.sweep.v2")
        self.assertEqual(sw["mode"], "mock")
        self.assertTrue(sw["executed_in_process"])
        self.assertFalse(sw["requires_acknowledgement"])
        self.assertIsNone(sw["launch_cmd"])
        self.assertIsNotNone(sw["cells"])
        self.assertEqual(len(sw["cells"]), 1)
        self.assertEqual(sw["cells"][0]["parser_mode"], "strict")


class RealStagingTests(unittest.TestCase):

    BUNDLED = os.path.join(
        os.path.dirname(__file__), "..", "tasks", "data",
        "swe_lite_style_bank.jsonl")

    def test_real_staged_when_not_acknowledged(self):
        from vision_mvp.coordpy import SweepSpec, run_sweep
        spec = SweepSpec(
            mode="real", jsonl=self.BUNDLED, model="qwen2.5-coder:14b",
            endpoint="http://192.168.12.191:11434",
            acknowledge_heavy=False)
        sw = run_sweep(spec)
        self.assertFalse(sw["executed_in_process"])
        self.assertTrue(sw["requires_acknowledgement"])
        self.assertIsNone(sw["cells"])
        self.assertIsNotNone(sw["launch_cmd"])
        self.assertIn("qwen2.5-coder:14b", sw["launch_cmd"])
        self.assertIn("192.168.12.191", " ".join(sw["launch_cmd"]))

    def test_strict_cost_gate_raises(self):
        from vision_mvp.coordpy import (
            SweepSpec, run_sweep, HeavyRunNotAcknowledged,
        )
        spec = SweepSpec(
            mode="real", jsonl=self.BUNDLED, model="qwen2.5-coder:14b",
            endpoint="http://192.168.12.191:11434",
            acknowledge_heavy=False)
        with self.assertRaises(HeavyRunNotAcknowledged):
            run_sweep(spec, strict_cost_gate=True)


class RunnerV2ReportTests(unittest.TestCase):

    def test_runner_emits_v2_schema(self):
        from vision_mvp.coordpy import RunSpec, run, PRODUCT_REPORT_SCHEMA
        with tempfile.TemporaryDirectory() as td:
            rep = run(RunSpec(profile="local_smoke", out_dir=td))
        self.assertEqual(rep["schema"], PRODUCT_REPORT_SCHEMA)
        self.assertEqual(rep["schema"], "phase45.product_report.v2")
        self.assertEqual(rep["sweep"]["schema"], "coordpy.sweep.v2")

    def test_aspen_profile_stages_through_v2_path_by_default(self):
        from vision_mvp.coordpy import RunSpec, run
        with tempfile.TemporaryDirectory() as td:
            rep = run(RunSpec(
                profile="aspen_mac1_coder", out_dir=td,
                acknowledge_heavy=False))
        sw = rep["sweep"]
        self.assertEqual(sw["schema"], "coordpy.sweep.v2")
        self.assertFalse(sw["executed_in_process"])
        self.assertTrue(sw["requires_acknowledgement"])
        # Staged launch_cmd lives inside the v2 block.
        self.assertIn("phase42_parser_sweep", " ".join(sw["launch_cmd"]))

    def test_runspec_config_flows_into_real_profile(self):
        from vision_mvp.coordpy import CoordPyConfig, RunSpec, run
        cfg = CoordPyConfig(
            model="gpt-4o-mini",
            llm_backend="openai",
            llm_base_url="https://api.example.test",
            llm_api_key="secret",
        )
        with tempfile.TemporaryDirectory() as td:
            rep = run(RunSpec(
                profile="aspen_mac1_coder",
                out_dir=td,
                acknowledge_heavy=False,
                config=cfg,
            ))
        sw = rep["sweep"]
        self.assertEqual(sw["schema"], "coordpy.sweep.v2")
        self.assertFalse(sw["executed_in_process"])
        self.assertIn("gpt-4o-mini", " ".join(sw["launch_cmd"]))
        self.assertIn("https://api.example.test", " ".join(sw["launch_cmd"]))


class EnvEndpointOverrideTests(unittest.TestCase):

    def test_mac1_override_applies(self):
        from vision_mvp.coordpy.runtime import sweep_spec_from_profile
        os.environ["COORDPY_OLLAMA_URL_MAC1"] = "http://localhost:11434"
        try:
            spec = sweep_spec_from_profile("aspen_mac1_coder")
            self.assertEqual(spec.endpoint, "http://localhost:11434")
        finally:
            del os.environ["COORDPY_OLLAMA_URL_MAC1"]

    def test_generic_override_applies(self):
        from vision_mvp.coordpy.runtime import sweep_spec_from_profile
        os.environ["COORDPY_OLLAMA_URL"] = "http://example.internal:11434"
        try:
            spec = sweep_spec_from_profile("aspen_mac2_frontier")
            self.assertEqual(spec.endpoint, "http://example.internal:11434")
        finally:
            del os.environ["COORDPY_OLLAMA_URL"]

    def test_openai_config_override_applies(self):
        from vision_mvp.coordpy import CoordPyConfig
        from vision_mvp.coordpy.runtime import sweep_spec_from_profile
        cfg = CoordPyConfig(
            model="gpt-4o-mini",
            llm_backend="openai_compatible",
            llm_base_url="https://api.example.test",
        )
        spec = sweep_spec_from_profile(
            "aspen_mac1_coder",
            config=cfg,
        )
        self.assertEqual(spec.model, "gpt-4o-mini")
        self.assertEqual(spec.endpoint, "https://api.example.test")


if __name__ == "__main__":
    unittest.main()
