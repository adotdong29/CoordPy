"""Phase 30 — substrate-evaluation loop-harness tests.

Covers:

  1. ``MockAnswerLLM`` deterministic echo / UNKNOWN behaviour.
  2. ``grade_answer`` per-kind grading correctness on each family
     (count, list, top-file, open-vocab).
  3. ``_render_substrate_answer`` produces the right cue string per
     task kind.
  4. ``build_aggregator_prompt`` includes / excludes the right
     content under every strategy and truncates cleanly.
  5. ``run_loop`` end-to-end on a tiny synthetic corpus:
       * naive, routing, substrate_wrap with ``MockAnswerLLM`` — the
         substrate_wrap path reaches 100 % accuracy on matched
         kinds because the cue is echoed.
       * naive path with ``MockAnswerLLM`` returns UNKNOWN →
         accuracy drops on count/list kinds.
  6. ``compute_cross_strategy_deltas`` reports the expected
     accuracy and token-ratio ordering.
  7. Prompt-chars monotonicity: substrate ≪ routing ≪ naive
     (the product headline on a corpus with non-trivial events).

Tests are deterministic and do not depend on Ollama or any real LLM.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import textwrap
import unittest

from vision_mvp.tasks.python_corpus import PythonCorpus
from vision_mvp.tasks.swe_loop_harness import (
    MockAnswerLLM, build_aggregator_prompt,
    compute_cross_strategy_deltas, grade_answer,
    _render_substrate_answer, run_loop,
)
from vision_mvp.tasks.task_scale_swe import (
    KIND_COUNT_FUNCTIONS, KIND_COUNT_MAY_RAISE, KIND_LIST_MAY_RAISE,
    KIND_OPEN_VOCAB, KIND_TOP_FILE_BY_FUNCTIONS,
    Task, build_event_stream, build_task_bank,
)


def _write(root: str, rel: str, body: str) -> None:
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(body)


def _make_corpus(tmp: str) -> PythonCorpus:
    _write(tmp, "mod_a.py", textwrap.dedent('''
        """Module a."""
        import subprocess
        def runner(cmd: str) -> None:
            subprocess.run([cmd])
        def raiser():
            raise ValueError('x')
        def pure(x: int) -> int:
            return x + 1
    ''').strip() + "\n")
    _write(tmp, "mod_b.py", textwrap.dedent('''
        """Module b."""
        import os
        def touches(p: str) -> str:
            with open(p) as f:
                return f.read()
    ''').strip() + "\n")
    c = PythonCorpus(root=tmp, seed=30)
    c.build()
    return c


class TestMockAnswerLLM(unittest.TestCase):
    def test_echoes_substrate_cue(self):
        mock = MockAnswerLLM()
        prompt = ("...\n"
                  "SUBSTRATE_ANSWER: 42\n"
                  "...\n")
        self.assertEqual(mock(prompt), "42")
        self.assertEqual(mock.n_calls, 1)
        self.assertGreater(mock.total_prompt_chars, 0)

    def test_returns_unknown_when_no_cue(self):
        mock = MockAnswerLLM()
        self.assertEqual(mock("nothing here"), "UNKNOWN")


class TestGrading(unittest.TestCase):
    def test_count_query_first_int(self):
        t = Task(task_id=0, kind=KIND_COUNT_MAY_RAISE,
                 question="how many", gold=3)
        self.assertTrue(grade_answer(t, "3"))
        self.assertTrue(grade_answer(t, "The answer is 3."))
        self.assertFalse(grade_answer(t, "2"))
        self.assertFalse(grade_answer(t, "no idea"))

    def test_list_query_exact_set(self):
        t = Task(task_id=0, kind=KIND_LIST_MAY_RAISE, question="list",
                 gold=("mod_a.raiser", "mod_b.touches"))
        self.assertTrue(grade_answer(t, "mod_a.raiser\nmod_b.touches"))
        self.assertTrue(grade_answer(t,
                                      '["mod_a.raiser", "mod_b.touches"]'))
        self.assertFalse(grade_answer(t, "mod_a.raiser"))  # missing one
        self.assertFalse(grade_answer(
            t, "mod_a.raiser\nmod_b.touches\nmod_c.extra"))   # FP

    def test_top_file(self):
        t = Task(task_id=0, kind=KIND_TOP_FILE_BY_FUNCTIONS,
                 question="which", gold="vision_mvp/core/big.py")
        self.assertTrue(grade_answer(t,
                                      "The file vision_mvp/core/big.py has the most."))
        self.assertTrue(grade_answer(t, "big.py"))
        self.assertFalse(grade_answer(t, "other.py"))

    def test_open_vocab_substring(self):
        t = Task(task_id=0, kind=KIND_OPEN_VOCAB,
                 question="summary", gold="mod_a", target_arg="mod_a")
        self.assertTrue(grade_answer(t, "mod_a processes commands"))
        self.assertFalse(grade_answer(t, "unrelated text"))


class TestSubstrateCue(unittest.TestCase):
    def test_count_cue(self):
        t = Task(task_id=0, kind=KIND_COUNT_FUNCTIONS,
                 question="q", gold=12)
        cue = _render_substrate_answer(t, corpus=None)
        self.assertEqual(cue, "12")

    def test_list_cue(self):
        t = Task(task_id=0, kind=KIND_LIST_MAY_RAISE, question="q",
                 gold=("a.x", "b.y"))
        cue = _render_substrate_answer(t, corpus=None)
        self.assertIn("a.x", cue)
        self.assertIn("b.y", cue)


class TestPromptAssembly(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="p30_prompt_")
        self.corpus = _make_corpus(self.tmp)
        self.events = build_event_stream(self.corpus, seed=30)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_naive_contains_every_event_body(self):
        tasks = build_task_bank(self.corpus, seed=30)
        t = tasks[0]
        prompt, delivered = build_aggregator_prompt(
            t, self.events, self.corpus, "naive",
            max_events_in_prompt=10_000)
        self.assertEqual(len(delivered), len(self.events))
        self.assertIn("DELIVERED EVENTS:", prompt)

    def test_substrate_zero_content_events(self):
        tasks = build_task_bank(self.corpus, seed=30)
        t = [x for x in tasks if x.kind == KIND_COUNT_MAY_RAISE][0]
        cue = _render_substrate_answer(t, self.corpus)
        prompt, delivered = build_aggregator_prompt(
            t, self.events, self.corpus, "substrate",
            substrate_cue=cue)
        # Only fixed-point events delivered.
        self.assertTrue(all(ev.is_fixed_point for ev in delivered))
        self.assertIn(f"SUBSTRATE_ANSWER: {cue}", prompt)
        # The prompt does not embed per-function details.
        for ev in self.events:
            if ev.event_type == "FUNCTION_DEF":
                self.assertNotIn(ev.body, prompt)

    def test_routing_subset(self):
        tasks = build_task_bank(self.corpus, seed=30)
        t = tasks[0]
        prompt_naive, delivered_naive = build_aggregator_prompt(
            t, self.events, self.corpus, "naive",
            max_events_in_prompt=10_000)
        prompt_rout, delivered_rout = build_aggregator_prompt(
            t, self.events, self.corpus, "routing",
            max_events_in_prompt=10_000)
        self.assertLessEqual(len(delivered_rout), len(delivered_naive))
        # Routing preserves fixed-point + matching subscriptions.
        self.assertTrue(all(
            ev.event_type in ("FILE_OPEN", "FUNCTION_DEF",
                                "IMPORT_STMT", "CLASS_DEF",
                                "TASK_GOAL", "FINAL_ANSWER")
            for ev in delivered_rout))


class TestRunLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="p30_loop_")
        self.corpus = _make_corpus(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_mock_llm_substrate_wrap_reaches_100_on_matched(self):
        mock = MockAnswerLLM()
        rep = run_loop(
            "mini", self.corpus, mock,
            strategies=("substrate_wrap",), seed=30,
        )
        pooled = rep.pooled_summary()
        # substrate_wrap: the mock echoes the cue → perfect on kinds
        # where the cue is the full answer (count/list/top-file).
        # Open-vocab's cue is a metadata summary, and the grader
        # checks substring containment of the target_arg (module
        # name), which is in the cue — so it also passes.
        self.assertEqual(pooled["substrate_wrap"]["accuracy"], 1.0)

    def test_naive_with_mock_llm_fails(self):
        mock = MockAnswerLLM()
        rep = run_loop(
            "mini", self.corpus, mock,
            strategies=("naive",), seed=30,
        )
        pooled = rep.pooled_summary()
        # Mock returns UNKNOWN on naive → grader gives 0 on everything
        # except KIND_OPEN_VOCAB which happens to pass substring only
        # if the module name happens to be "UNKNOWN" (it won't).
        self.assertLessEqual(pooled["naive"]["accuracy"], 0.5)

    def test_prompt_tokens_ordering(self):
        mock = MockAnswerLLM()
        rep = run_loop(
            "mini", self.corpus, mock,
            strategies=("naive", "routing", "substrate_wrap"), seed=30,
        )
        pooled = rep.pooled_summary()
        self.assertGreater(
            pooled["naive"]["mean_prompt_tokens"],
            pooled["substrate_wrap"]["mean_prompt_tokens"])
        self.assertGreaterEqual(
            pooled["naive"]["mean_prompt_tokens"],
            pooled["routing"]["mean_prompt_tokens"])

    def test_cross_strategy_deltas(self):
        mock = MockAnswerLLM()
        rep = run_loop(
            "mini", self.corpus, mock,
            strategies=("naive", "substrate_wrap"), seed=30,
        )
        deltas = compute_cross_strategy_deltas(rep)
        self.assertTrue(deltas)
        # One pair: naive vs substrate_wrap.
        d = deltas[0]
        self.assertEqual(d.base, "naive")
        self.assertEqual(d.comp, "substrate_wrap")
        self.assertGreater(d.token_ratio, 1.0)   # naive uses more tokens
        self.assertGreater(d.accuracy_delta, 0.0)   # substrate wins

    def test_report_is_json_serialisable(self):
        mock = MockAnswerLLM()
        rep = run_loop(
            "mini", self.corpus, mock,
            strategies=("substrate_wrap",), seed=30,
        )
        import json
        s = json.dumps(rep.as_dict())
        self.assertIn("corpus_name", s)


if __name__ == "__main__":
    unittest.main()
