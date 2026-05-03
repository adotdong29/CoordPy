"""CoordPy provenance manifest tests.

Every CoordPy run must carry a provenance block with the fields that
let an operator reproduce it: git SHA, package version, python,
platform, profile, model, sandbox, JSONL path + checksum, argv, and
artifact list.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest


REQUIRED_TOP_LEVEL = (
    "schema", "timestamp_utc", "code", "runtime", "profile",
    "model", "sandbox", "input", "invocation", "output",
)

REQUIRED_CODE_FIELDS = (
    "git_sha", "git_dirty", "package", "package_version", "repo_dir")
REQUIRED_RUNTIME_FIELDS = (
    "python_version", "python_implementation", "platform",
    "machine", "system")
REQUIRED_INPUT_FIELDS = ("jsonl_path", "jsonl_sha256", "jsonl_bytes")
REQUIRED_INVOCATION_FIELDS = ("argv", "cwd", "user", "hostname")
REQUIRED_OUTPUT_FIELDS = ("out_dir", "artifacts")


class ProvenanceManifestTests(unittest.TestCase):

    def test_build_manifest_shape(self):
        from vision_mvp.coordpy.provenance import build_manifest
        m = build_manifest(
            profile_name="local_smoke",
            profile_schema="phase45.profile.v1",
            jsonl_path=None,
            model="qwen2.5-coder:14b",
            endpoint="http://localhost:11434",
            sandbox="in_process",
            out_dir="/tmp/example",
            artifacts=["product_report.json"],
            argv=["coordpy", "--profile", "local_smoke"],
        )
        for k in REQUIRED_TOP_LEVEL:
            self.assertIn(k, m, f"missing provenance key: {k}")
        self.assertEqual(m["schema"], "coordpy.provenance.v1")
        for k in REQUIRED_CODE_FIELDS:
            self.assertIn(k, m["code"])
        for k in REQUIRED_RUNTIME_FIELDS:
            self.assertIn(k, m["runtime"])
        for k in REQUIRED_INPUT_FIELDS:
            self.assertIn(k, m["input"])
        for k in REQUIRED_INVOCATION_FIELDS:
            self.assertIn(k, m["invocation"])
        for k in REQUIRED_OUTPUT_FIELDS:
            self.assertIn(k, m["output"])
        self.assertEqual(m["model"]["tag"], "qwen2.5-coder:14b")
        self.assertEqual(m["model"]["endpoint"], "http://localhost:11434")

    def test_jsonl_sha256_is_computed_when_file_exists(self):
        from vision_mvp.coordpy.provenance import build_manifest
        with tempfile.NamedTemporaryFile(
                "w", suffix=".jsonl", delete=False) as fh:
            fh.write('{"a":1}\n{"b":2}\n')
            path = fh.name
        try:
            m = build_manifest(jsonl_path=path)
            self.assertIsNotNone(m["input"]["jsonl_sha256"])
            self.assertEqual(len(m["input"]["jsonl_sha256"]), 64)
            self.assertEqual(m["input"]["jsonl_bytes"], os.path.getsize(path))
        finally:
            os.unlink(path)

    def test_jsonl_fields_none_when_path_missing(self):
        from vision_mvp.coordpy.provenance import build_manifest
        m = build_manifest(jsonl_path=None)
        self.assertIsNone(m["input"]["jsonl_path"])
        self.assertIsNone(m["input"]["jsonl_sha256"])

    def test_runner_emits_provenance_on_every_run(self):
        from vision_mvp.product.runner import run_profile
        with tempfile.TemporaryDirectory() as td:
            rep = run_profile("local_smoke", out_dir=td)
            # Attached to report.
            self.assertIn("provenance", rep)
            prov = rep["provenance"]
            self.assertEqual(prov["schema"], "coordpy.provenance.v1")
            self.assertEqual(prov["profile"]["name"], "local_smoke")
            self.assertEqual(prov["sandbox"], "in_process")
            # Written to disk.
            disk_path = os.path.join(td, "provenance.json")
            self.assertTrue(os.path.exists(disk_path))
            with open(disk_path, "r", encoding="utf-8") as fh:
                disk_prov = json.load(fh)
            self.assertEqual(
                disk_prov["schema"], "coordpy.provenance.v1")
            # JSONL checksum populated for the bundled bank.
            self.assertIsNotNone(disk_prov["input"]["jsonl_sha256"])
            # Artifact list includes provenance.json itself.
            self.assertIn("provenance.json", disk_prov["output"]["artifacts"])
            # And the product report lists it too.
            self.assertIn("provenance.json", rep["artifacts"])

    def test_provenance_captures_aspen_endpoint_for_real_profile(self):
        from vision_mvp.product.runner import run_profile
        with tempfile.TemporaryDirectory() as td:
            rep = run_profile("aspen_mac1_coder", out_dir=td)
            prov = rep["provenance"]
            self.assertEqual(prov["model"]["tag"], "qwen2.5-coder:14b")
            self.assertEqual(
                prov["model"]["endpoint"], "http://192.168.12.191:11434")

    def test_provenance_json_is_valid_json(self):
        # Regression guard: manifest must survive json.dump/load.
        from vision_mvp.coordpy.provenance import build_manifest
        m = build_manifest(
            profile_name="x", sandbox="subprocess",
            model=None, endpoint=None)
        s = json.dumps(m, default=str)
        round_tripped = json.loads(s)
        self.assertEqual(round_tripped["schema"], "coordpy.provenance.v1")


class CLITests(unittest.TestCase):
    """Smoke-test the console scripts via their Python entry points."""

    def test_coordpy_cli_version_flag(self):
        from vision_mvp.coordpy import _cli
        # --version is a pure stdout path, exit 0.
        rc = _cli._cmd_run(["--profile", "local_smoke",
                            "--out-dir", "/tmp/unused-coordpy-version",
                            "--version"])
        self.assertEqual(rc, 0)

    def test_coordpy_cli_runs_local_smoke(self):
        from vision_mvp.coordpy import _cli
        with tempfile.TemporaryDirectory() as td:
            rc = _cli._cmd_run(["--profile", "local_smoke",
                                "--out-dir", td])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(
                os.path.join(td, "product_report.json")))
            self.assertTrue(os.path.exists(
                os.path.join(td, "provenance.json")))

    def test_coordpy_ci_cli_on_fresh_report(self):
        from vision_mvp.coordpy import _cli
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(_cli._cmd_run(
                ["--profile", "local_smoke", "--out-dir", td]), 0)
            rc = _cli._cmd_ci(["--report",
                                os.path.join(td, "product_report.json")])
            self.assertEqual(rc, 0)

    def test_coordpy_import_cli_on_missing_file(self):
        from vision_mvp.coordpy import _cli
        rc = _cli._cmd_import(["--jsonl", "/tmp/definitely-missing.jsonl",
                                "--skip-readiness"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
