"""Unit tests for the round-1 cache module (experiments/round1_cache.py)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vision_mvp.experiments.round1_cache import save_round1, load_round1


_SAMPLE_ROUND1 = {
    "drafts": {
        "alpha": "def alpha(x): return x",
        "beta":  "def beta(x): return x + 1",
    },
    "tokens": {
        "alpha": {"prompt": 100, "completion": 50},
        "beta":  {"prompt": 110, "completion": 55},
    },
    "acceptance": {
        "alpha": {"accepted": True,  "attempts": 1},
        "beta":  {"accepted": False, "attempts": 2},
    },
}


class TestSaveLoad(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            save_round1(path, _SAMPLE_ROUND1, surface="testsurf", model="m1")
            r1, meta = load_round1(path)

        self.assertEqual(r1["drafts"], _SAMPLE_ROUND1["drafts"])
        self.assertEqual(r1["tokens"], _SAMPLE_ROUND1["tokens"])
        self.assertEqual(r1["acceptance"], _SAMPLE_ROUND1["acceptance"])
        self.assertEqual(meta["surface"], "testsurf")
        self.assertEqual(meta["model"], "m1")
        self.assertIn("timestamp", meta)

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "cache.json"
            save_round1(path, _SAMPLE_ROUND1, surface="s", model="m")
            self.assertTrue(path.exists())

    def test_file_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            save_round1(path, _SAMPLE_ROUND1, surface="s", model="m")
            with path.open() as fh:
                raw = json.load(fh)
        self.assertIn("round1", raw)
        self.assertIn("surface", raw)
        self.assertIn("model", raw)
        self.assertIn("timestamp", raw)


class TestSaveValidation(unittest.TestCase):
    def test_missing_drafts_raises(self):
        bad = {k: v for k, v in _SAMPLE_ROUND1.items() if k != "drafts"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            with self.assertRaises(ValueError) as ctx:
                save_round1(path, bad, surface="s", model="m")
        self.assertIn("drafts", str(ctx.exception))

    def test_missing_tokens_raises(self):
        bad = {k: v for k, v in _SAMPLE_ROUND1.items() if k != "tokens"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            with self.assertRaises(ValueError):
                save_round1(path, bad, surface="s", model="m")

    def test_missing_acceptance_raises(self):
        bad = {k: v for k, v in _SAMPLE_ROUND1.items() if k != "acceptance"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            with self.assertRaises(ValueError):
                save_round1(path, bad, surface="s", model="m")


class TestLoadValidation(unittest.TestCase):
    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_round1("/nonexistent/path/cache.json")

    def test_missing_top_level_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            with path.open("w") as fh:
                json.dump({"surface": "s", "model": "m", "timestamp": "t"}, fh)
            with self.assertRaises(ValueError) as ctx:
                load_round1(path)
        self.assertIn("round1", str(ctx.exception))

    def test_missing_round1_subkey(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            with path.open("w") as fh:
                json.dump({
                    "surface": "s", "model": "m", "timestamp": "t",
                    "round1": {"drafts": {}, "tokens": {}},  # missing acceptance
                }, fh)
            with self.assertRaises(ValueError) as ctx:
                load_round1(path)
        self.assertIn("acceptance", str(ctx.exception))

    def test_extra_keys_in_round1_are_preserved(self):
        """Fields beyond the three required keys pass through unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            with path.open("w") as fh:
                json.dump({
                    "surface": "s", "model": "m", "timestamp": "t",
                    "round1": {
                        "drafts": {}, "tokens": {}, "acceptance": {},
                        "extra_field": "hello",
                    },
                }, fh)
            r1, _ = load_round1(path)
        self.assertEqual(r1["extra_field"], "hello")


class TestMetadata(unittest.TestCase):
    def test_metadata_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            save_round1(path, _SAMPLE_ROUND1, surface="protocolkit", model="qwen2.5-coder:7b")
            _, meta = load_round1(path)
        self.assertEqual(set(meta.keys()), {"surface", "model", "timestamp"})

    def test_timestamp_is_iso8601(self):
        from datetime import datetime
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            save_round1(path, _SAMPLE_ROUND1, surface="s", model="m")
            _, meta = load_round1(path)
        # Should parse without error
        dt = datetime.fromisoformat(meta["timestamp"])
        self.assertIsNotNone(dt)


if __name__ == "__main__":
    unittest.main()
