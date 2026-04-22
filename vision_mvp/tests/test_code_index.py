"""Tests for the Python AST ingestion module — Phase 22, extended in Phase 23."""

from __future__ import annotations

import os
import tempfile
import textwrap
import unittest

import numpy as np

from vision_mvp.core.code_index import (
    CodeIndexer, CodeMetadata, IngestionStats,
    _function_returns_none, extract_metadata,
)
from vision_mvp.core.context_ledger import ContextLedger, hash_embedding


def _embed(text: str) -> np.ndarray:
    return hash_embedding(text, dim=32)


class TestExtractMetadata(unittest.TestCase):
    def test_simple_module(self):
        src = textwrap.dedent("""\
            \"\"\"A small module.\"\"\"
            import os
            from collections import Counter

            def foo(a, b):
                return a + b

            def bar() -> None:
                pass

            class Baz:
                def method(self):
                    pass
        """)
        md = extract_metadata("/tmp/sample.py", src, root="/tmp")
        self.assertEqual(md.module_name, "sample")
        self.assertEqual(md.n_functions, 2)
        self.assertEqual(md.n_classes, 1)
        self.assertEqual(md.n_methods, 1)
        self.assertIn("os", md.imports)
        self.assertIn("collections", md.imports)
        self.assertIn("collections.Counter", md.imports)
        self.assertEqual(md.function_names, ("foo", "bar"))
        self.assertEqual(md.class_names, ("Baz",))
        self.assertTrue(md.has_docstring)
        # foo returns a value → False; bar -> None → True
        self.assertFalse(md.function_returns_none[0])
        self.assertTrue(md.function_returns_none[1])

    def test_generator_not_returning_none(self):
        src = textwrap.dedent("""\
            def gen():
                yield 1
                yield 2
        """)
        md = extract_metadata("/tmp/g.py", src, root="/tmp")
        self.assertEqual(md.function_names, ("gen",))
        # Generator → False (not classified as None-returning)
        self.assertFalse(md.function_returns_none[0])

    def test_no_explicit_return_is_none(self):
        src = "def f(): pass"
        md = extract_metadata("/tmp/n.py", src, root="/tmp")
        self.assertTrue(md.function_returns_none[0])

    def test_test_file_detection(self):
        md = extract_metadata("/tmp/foo/test_module.py",
                               "def test_x(): pass", root="/tmp")
        self.assertTrue(md.is_test_file)
        self.assertTrue(md.function_is_test[0])

    def test_syntax_error_doesnt_raise(self):
        md = extract_metadata("/tmp/bad.py", "def def def", root="/tmp")
        self.assertEqual(md.n_functions, 0)
        self.assertEqual(md.imports, ())


class TestFunctionReturnsNoneStatic(unittest.TestCase):
    def _parse(self, src):
        import ast
        return ast.parse(src).body[0]

    def test_explicit_value_return(self):
        node = self._parse("def f():\n    return 1\n")
        self.assertFalse(_function_returns_none(node))

    def test_return_none_only(self):
        node = self._parse("def f():\n    return\n")
        self.assertTrue(_function_returns_none(node))

    def test_annotated_none(self):
        node = self._parse("def f() -> None:\n    return 1\n")
        self.assertTrue(_function_returns_none(node))   # annotation wins

    def test_yield_makes_generator(self):
        node = self._parse("def f():\n    yield 1\n")
        self.assertFalse(_function_returns_none(node))


class TestIndexerIntoLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        # Create three files.
        with open(os.path.join(self.root, "a.py"), "w") as f:
            f.write("import os\ndef foo(): return 1\n")
        with open(os.path.join(self.root, "b.py"), "w") as f:
            f.write("import os\nimport sys\ndef bar(): pass\nclass C: pass\n")
        os.makedirs(os.path.join(self.root, "tests"))
        with open(os.path.join(self.root, "tests", "test_a.py"), "w") as f:
            f.write("def test_one(): pass\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_index_ingests_all_files(self):
        ledger = ContextLedger(embed_dim=32, embed_fn=_embed,
                                max_artifact_chars=10_000)
        idx = CodeIndexer(root=self.root)
        handles = idx.index_into(ledger)
        self.assertEqual(len(handles), 3)
        self.assertEqual(len(ledger), 3)

    def test_metadata_survives_handle_round_trip(self):
        ledger = ContextLedger(embed_dim=32, embed_fn=_embed)
        idx = CodeIndexer(root=self.root)
        handles = idx.index_into(ledger)
        # Find b.py — has 1 function and 1 class.
        bh = [h for h in handles if h.metadata_dict()["file_path"].endswith("b.py")][0]
        md = bh.metadata_dict()
        self.assertEqual(md["n_functions"], 1)
        self.assertEqual(md["n_classes"], 1)
        self.assertIn("os", md["imports"])
        self.assertIn("sys", md["imports"])

    def test_test_file_classified(self):
        ledger = ContextLedger(embed_dim=32, embed_fn=_embed)
        idx = CodeIndexer(root=self.root)
        handles = idx.index_into(ledger)
        test_handles = [h for h in handles
                        if h.metadata_dict().get("is_test_file")]
        self.assertEqual(len(test_handles), 1)
        self.assertTrue(test_handles[0].metadata_dict()["file_path"].endswith("test_a.py"))

    def test_idempotent_reingest(self):
        ledger = ContextLedger(embed_dim=32, embed_fn=_embed)
        idx = CodeIndexer(root=self.root)
        h1 = idx.index_into(ledger)
        h2 = idx.index_into(ledger)
        # Re-ingesting the SAME files produces the SAME CIDs.
        self.assertEqual([h.cid for h in h1], [h.cid for h in h2])
        # Ledger unique-CIDs unchanged.
        self.assertEqual(len(ledger), 3)


class TestIngestionStats(unittest.TestCase):
    """Phase-23: coverage accounting per ingest pass."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        # 3 parse-OK files
        with open(os.path.join(self.root, "a.py"), "w") as f:
            f.write("import os\ndef foo(): return 1\n")
        with open(os.path.join(self.root, "b.py"), "w") as f:
            f.write('"""doc."""\nimport sys\nclass C: pass\n')
        os.makedirs(os.path.join(self.root, "tests"))
        with open(os.path.join(self.root, "tests", "test_a.py"), "w") as f:
            f.write("def test_one(): pass\n")
        # 1 trivial file (blank, no imports, no defs, no docstring)
        with open(os.path.join(self.root, "empty.py"), "w") as f:
            f.write("# a comment only\n")
        # 1 syntax-error file
        with open(os.path.join(self.root, "broken.py"), "w") as f:
            f.write("def def def\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_stats_populated(self):
        ledger = ContextLedger(embed_dim=32, embed_fn=_embed,
                                max_artifact_chars=10_000)
        idx = CodeIndexer(root=self.root)
        idx.index_into(ledger)
        s = idx.stats
        self.assertEqual(s.files_seen, 5)
        # 3 parse_ok (have import/def/class/docstring), 1 trivial, 1 error
        self.assertEqual(s.files_parsed_ok, 3)
        self.assertEqual(s.files_trivial, 1)
        self.assertEqual(s.files_syntax_error, 1)
        self.assertEqual(s.files_oversize_skipped, 0)

    def test_oversize_accounted(self):
        # A huge file trips the oversize path but not the syntax-error path.
        big_path = os.path.join(self.root, "big.py")
        with open(big_path, "w") as f:
            f.write("x = 1\n" * 5000)   # ~30 KB
        ledger = ContextLedger(embed_dim=32, embed_fn=_embed)
        idx = CodeIndexer(root=self.root, max_chars_per_file=1024)
        idx.index_into(ledger)
        self.assertEqual(idx.stats.files_oversize_skipped, 1)

    def test_coverage_ratios(self):
        ledger = ContextLedger(embed_dim=32, embed_fn=_embed)
        idx = CodeIndexer(root=self.root)
        idx.index_into(ledger)
        s = idx.stats
        # parse_coverage counts parsed_ok AND trivial (both parsed cleanly)
        self.assertAlmostEqual(s.parse_coverage, 4 / 5)
        # metadata_completeness counts only parsed_ok (structural signal)
        self.assertAlmostEqual(s.metadata_completeness, 3 / 5)

    def test_fresh_stats_per_call(self):
        ledger = ContextLedger(embed_dim=32, embed_fn=_embed)
        idx = CodeIndexer(root=self.root)
        idx.index_into(ledger)
        first = idx.stats.files_seen
        idx.index_into(ledger)
        # Second call resets stats rather than accumulating.
        self.assertEqual(idx.stats.files_seen, first)
        self.assertEqual(idx.stats.files_parsed_ok, 3)

    def test_as_dict_is_json_serialisable(self):
        import json
        ledger = ContextLedger(embed_dim=32, embed_fn=_embed)
        idx = CodeIndexer(root=self.root)
        idx.index_into(ledger)
        blob = json.dumps(idx.stats.as_dict())
        round_trip = json.loads(blob)
        self.assertEqual(round_trip["files_seen"], 5)
        self.assertEqual(round_trip["files_parsed_ok"], 3)


if __name__ == "__main__":
    unittest.main()
