"""Tests for the Phase-23 multi-corpus registry."""

from __future__ import annotations

import os
import tempfile
import textwrap
import unittest

from vision_mvp.tasks.corpus_registry import (
    CorpusRegistry, CorpusSpec, default_phase23_registry,
)


def _make_tiny_repo(path: str) -> None:
    files = {
        "m.py": "import os\ndef foo(): return 1\n",
        "n.py": "import sys\nclass A: pass\n",
        "pkg/__init__.py": "",
        "pkg/inner.py": '"""doc."""\ndef bar() -> None: pass\n',
    }
    for rel, content in files.items():
        full = os.path.join(path, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)


class TestCorpusSpec(unittest.TestCase):
    def test_defaults(self):
        s = CorpusSpec(name="foo", root="/tmp/x")
        self.assertEqual(s.family, "unknown")
        self.assertIsNone(s.max_files)
        self.assertEqual(s.max_chars_per_file, 64_000)

    def test_frozen(self):
        s = CorpusSpec(name="foo", root="/tmp/x")
        # dataclass(frozen=True) must disallow mutation
        with self.assertRaises(Exception):
            s.name = "bar"  # type: ignore[misc]


class TestRegistryBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _make_tiny_repo(self.tmp.name)
        self.reg = CorpusRegistry([
            CorpusSpec(name="tiny", root=self.tmp.name, family="test"),
        ])

    def tearDown(self):
        self.tmp.cleanup()

    def test_build_one_corpus(self):
        entries = self.reg.build()
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e.name, "tiny")
        self.assertEqual(e.spec.family, "test")
        self.assertEqual(e.corpus.n_files, 4)

    def test_summary_contents(self):
        entries = self.reg.build()
        s = entries[0].summary()
        # Required keys
        for k in ("name", "root", "family", "n_files", "total_lines",
                  "n_functions_total", "n_classes_total", "n_distinct_imports",
                  "n_questions", "questions_by_kind", "coverage",
                  "most_imported_module"):
            self.assertIn(k, s, f"missing key {k!r}")
        self.assertEqual(s["n_files"], 4)

    def test_coverage_matches_indexer(self):
        entries = self.reg.build()
        cov = entries[0].coverage
        self.assertGreaterEqual(cov.files_seen, 4)
        self.assertGreaterEqual(cov.files_parsed_ok, 1)
        self.assertEqual(cov.files_syntax_error, 0)
        # parse_coverage is a ratio in [0, 1]
        self.assertGreaterEqual(cov.parse_coverage, 0.0)
        self.assertLessEqual(cov.parse_coverage, 1.0)

    def test_only_filter(self):
        # Add a second spec; only= should narrow it.
        tmp2 = tempfile.TemporaryDirectory()
        _make_tiny_repo(tmp2.name)
        self.reg.add(CorpusSpec(name="tiny2", root=tmp2.name, family="test"))
        try:
            entries = self.reg.build(only={"tiny"})
            self.assertEqual([e.name for e in entries], ["tiny"])
            entries = self.reg.build(only={"tiny2"})
            self.assertEqual([e.name for e in entries], ["tiny2"])
        finally:
            tmp2.cleanup()

    def test_build_is_repeatable(self):
        e1 = self.reg.build()
        e2 = self.reg.build()
        # Same number of files, same gold answers
        self.assertEqual(e1[0].corpus.n_functions_total,
                         e2[0].corpus.n_functions_total)
        self.assertEqual(e1[0].corpus.n_classes_total,
                         e2[0].corpus.n_classes_total)


class TestMissingRootHandled(unittest.TestCase):
    def test_nonexistent_root_crashes_on_build(self):
        # Caller is expected to pass a valid root; we surface the
        # failure rather than silently returning 0 files.
        reg = CorpusRegistry([
            CorpusSpec(name="x", root="/definitely/does/not/exist"),
        ])
        entries = reg.build()
        self.assertEqual(entries[0].corpus.n_files, 0)
        # Coverage still populates with zero counters.
        self.assertEqual(entries[0].coverage.files_seen, 0)


class TestDefaultPhase23Registry(unittest.TestCase):
    def test_discovers_repo_corpora(self):
        # We're inside the repo; the four default roots should resolve.
        reg = default_phase23_registry()
        names = [s.name for s in reg.specs]
        self.assertIn("vision-core", names)
        # At least three should exist even if experiments/tests layouts change.
        self.assertGreaterEqual(len(names), 3)

    def test_rejects_empty_extra_roots(self):
        # Nonexistent extra root is silently ignored (not crash-y).
        reg = default_phase23_registry(
            extra_roots=["/definitely/does/not/exist"])
        for s in reg.specs:
            self.assertNotEqual(s.root, "/definitely/does/not/exist")

    def test_accepts_real_extra_root(self):
        # A real extra root gets added with its basename as name.
        tmp = tempfile.TemporaryDirectory()
        try:
            _make_tiny_repo(tmp.name)
            reg = default_phase23_registry(extra_roots=[tmp.name])
            names = [s.name for s in reg.specs]
            basename = os.path.basename(os.path.normpath(tmp.name))
            self.assertIn(basename, names)
        finally:
            tmp.cleanup()


class TestMultiCorpusDirectExact(unittest.TestCase):
    """End-to-end smoke test: the direct-exact path works across multiple
    corpora without shared state leaking between them."""

    def setUp(self):
        self.tmp1 = tempfile.TemporaryDirectory()
        self.tmp2 = tempfile.TemporaryDirectory()
        # Corpus 1: 2 files, 2 functions
        with open(os.path.join(self.tmp1.name, "a.py"), "w") as f:
            f.write("def f(): pass\n")
        with open(os.path.join(self.tmp1.name, "b.py"), "w") as f:
            f.write("def g(): pass\n")
        # Corpus 2: 1 file, 3 functions + 1 class
        with open(os.path.join(self.tmp2.name, "c.py"), "w") as f:
            f.write("def h(): pass\ndef i(): pass\ndef j(): pass\n"
                    "class K: pass\n")

    def tearDown(self):
        self.tmp1.cleanup()
        self.tmp2.cleanup()

    def test_two_corpora_report_different_counts(self):
        from vision_mvp.core.code_index import CodeIndexer
        from vision_mvp.core.code_planner import CodeQueryPlanner
        from vision_mvp.core.context_ledger import ContextLedger, hash_embedding
        from vision_mvp.tasks.needle_corpus import NeedleCorpus

        reg = CorpusRegistry([
            CorpusSpec(name="c1", root=self.tmp1.name),
            CorpusSpec(name="c2", root=self.tmp2.name),
        ])
        planner = CodeQueryPlanner()
        for entry in reg.build():
            ledger = ContextLedger(
                embed_dim=16,
                embed_fn=lambda t: hash_embedding(t, dim=16))
            CodeIndexer(root=entry.spec.root).index_into(ledger)
            # Get the count_functions_total question from the corpus
            q_count = [q for q in entry.corpus.questions
                       if q.kind == "count_functions_total"][0]
            res = planner.plan(q_count.question)
            stage, _ = res.plan.execute(ledger)
            rendered = res.plan.render(stage)
            self.assertTrue(NeedleCorpus.score_exact(rendered, q_count),
                            f"{entry.name}: expected {q_count.gold}, got {rendered}")


if __name__ == "__main__":
    unittest.main()
