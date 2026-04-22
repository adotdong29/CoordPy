"""Tests for the executable-snippet corpus — Phase 26.

Two kinds of claims:

  1. **Registry integrity.** Every snippet parses as valid Python,
     its target exists in the compiled module, and its ground_truth
     dict has an entry per runtime-decidable predicate.

  2. **Probe / ground-truth agreement.** For every snippet, running
     the runtime probes against the declared target produces the
     result the author claimed in `ground_truth`. If this test ever
     fails, either the snippet's behaviour changed or the probe
     drifted; this is the self-consistency boundary that keeps the
     calibration harness honest.

Note: these tests call `calibrate_snippet` on each registry entry
with a small fuzz budget. Each snippet runs in a sandboxed context
via the probe infrastructure — no subprocess ever spawns, no real
file ever opens outside the per-probe tempdir, no real socket ever
connects.
"""

from __future__ import annotations

import ast
import unittest

from vision_mvp.core.code_runtime_calibration import (
    RUNTIME_DECIDABLE_PREDICATES, SNIPPET_CORPUS_PREDICATES,
    calibrate_snippet,
    compute_static_flags_from_source,
)
from vision_mvp.tasks.executable_snippets import (
    default_snippet_registry, snippet_by_name, snippets_by_family,
)


class TestRegistryIntegrity(unittest.TestCase):

    def test_registry_is_non_empty(self):
        self.assertGreaterEqual(len(default_snippet_registry()), 15)

    def test_every_snippet_has_unique_name(self):
        names = [s.name for s in default_snippet_registry()]
        self.assertEqual(len(names), len(set(names)))

    def test_every_snippet_source_parses(self):
        import textwrap
        for s in default_snippet_registry():
            try:
                ast.parse(textwrap.dedent(s.source))
            except SyntaxError as e:
                self.fail(f"snippet {s.name!r} has SyntaxError: {e}")

    def test_every_snippet_has_full_ground_truth(self):
        # Phase-26 snippet corpus was authored for the six original
        # Phase-26 predicates. The Phase-28 `may_raise_explicit` /
        # `may_raise_implicit` split is derived from `may_raise` at
        # runtime and does not require a separate ground-truth entry
        # per snippet.
        for s in default_snippet_registry():
            for p in SNIPPET_CORPUS_PREDICATES:
                self.assertIn(p, s.ground_truth,
                               msg=f"{s.name} missing ground_truth[{p}]")

    def test_snippet_by_name_roundtrip(self):
        first = default_snippet_registry()[0]
        self.assertIs(snippet_by_name(first.name), first)

    def test_snippets_by_family_filters(self):
        # `direct` is a big family; assert at least one entry.
        direct = snippets_by_family("direct")
        self.assertGreaterEqual(len(direct), 3)


class TestProbeAgreesWithGroundTruth(unittest.TestCase):
    """For every snippet and every runtime-decidable predicate, the
    probe's observation must match the author's ground_truth.

    This is self-consistency — if it fails, either (a) the snippet
    is mis-declared or (b) the probe has drifted. Fix (a) in
    `tasks/executable_snippets.py`; fix (b) in
    `core/code_runtime_calibration.py`. The calibration against
    the ANALYZER (static vs runtime) is the separate axis measured
    by the experiment.
    """

    def test_every_snippet_agrees_per_predicate(self):
        # Phase-28 note: restrict this to the six Phase-26 predicates —
        # the snippet corpus's ground-truth dicts are keyed on them.
        # The new Phase-28 `may_raise_explicit` / `may_raise_implicit`
        # probes are covered independently in
        # `test_phase28_implicit_raise.py`.
        for spec in default_snippet_registry():
            with self.subTest(snippet=spec.name):
                res = calibrate_snippet(
                    spec,
                    predicates=SNIPPET_CORPUS_PREDICATES,
                    seeds=(17, 23),
                )
                for pred in SNIPPET_CORPUS_PREDICATES:
                    obs = res.runtime_observations[pred]
                    expected = spec.ground_truth[pred]
                    if not obs.applicable:
                        # Some snippets (e.g. mutual_recursion's
                        # cycle_entry) have targets that are simply
                        # not runtime-decidable for a predicate; skip.
                        continue
                    self.assertEqual(
                        obs.runtime_flag, expected,
                        msg=(f"{spec.name}:{pred}: "
                             f"probe={obs.runtime_flag} "
                             f"ground_truth={expected} "
                             f"witnesses={obs.witnesses}"),
                    )


class TestAnalyzerVsRuntime(unittest.TestCase):
    """Spot-check that specific snippets land where we expect on the
    calibration axis. These are the anchoring cases for Theorem
    P26-1 (analyzer-gold vs runtime-truth are separate axes).
    """

    def test_dead_raise_is_static_true_runtime_false(self):
        spec = snippet_by_name("dead_raise")
        static = compute_static_flags_from_source(
            spec.source, spec.target_qname)
        self.assertTrue(static["may_raise"])
        res = calibrate_snippet(
            spec, predicates=["may_raise"], seeds=(0,),
            static_flags=static)
        self.assertFalse(
            res.runtime_observations["may_raise"].runtime_flag)
        self.assertIn(("may_raise", "false_positive"), res.divergences())

    def test_hidden_subprocess_via_eval_is_static_false_runtime_true(self):
        spec = snippet_by_name("hidden_subprocess_via_eval")
        static = compute_static_flags_from_source(
            spec.source, spec.target_qname)
        self.assertFalse(static.get("calls_subprocess", False))
        res = calibrate_snippet(
            spec, predicates=["calls_subprocess"], seeds=(0,),
            static_flags=static)
        self.assertTrue(
            res.runtime_observations["calls_subprocess"].runtime_flag)
        self.assertIn(
            ("calls_subprocess", "false_negative"), res.divergences())

    def test_hidden_filesystem_via_getattr_is_static_false_runtime_true(self):
        spec = snippet_by_name("hidden_filesystem_via_getattr")
        static = compute_static_flags_from_source(
            spec.source, spec.target_qname)
        self.assertFalse(static.get("calls_filesystem", False))
        res = calibrate_snippet(
            spec, predicates=["calls_filesystem"], seeds=(0,),
            static_flags=static)
        self.assertTrue(
            res.runtime_observations["calls_filesystem"].runtime_flag)
        self.assertIn(
            ("calls_filesystem", "false_negative"), res.divergences())

    def test_wrapper_subprocess_trans_agrees_with_runtime(self):
        """Interprocedural analysis catches the wrapper; runtime
        observes. Both should agree on True."""
        spec = snippet_by_name("wrapper_subprocess")
        static = compute_static_flags_from_source(
            spec.source, spec.target_qname)
        self.assertTrue(
            static["calls_subprocess"],
            "Phase-25 trans flag should catch subprocess through helper")
        res = calibrate_snippet(
            spec, predicates=["calls_subprocess"], seeds=(0,),
            static_flags=static)
        self.assertTrue(
            res.runtime_observations["calls_subprocess"].runtime_flag)
        self.assertEqual(res.divergences(), [])

    def test_caught_raise_both_false(self):
        spec = snippet_by_name("caught_raise")
        static = compute_static_flags_from_source(
            spec.source, spec.target_qname)
        self.assertFalse(static["may_raise"],
                          "analyzer correctly recognises catch-all")
        res = calibrate_snippet(
            spec, predicates=["may_raise"], seeds=(0,),
            static_flags=static)
        self.assertFalse(
            res.runtime_observations["may_raise"].runtime_flag)
        self.assertEqual(res.divergences(), [])


if __name__ == "__main__":
    unittest.main()
