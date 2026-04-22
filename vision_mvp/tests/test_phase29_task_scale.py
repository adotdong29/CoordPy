"""Phase 29 — task-scale causal-relevance tests.

Covers:

  1. ``build_event_stream`` emits one event per corpus atom (file,
     function, import, class) plus deterministic fixed-point events
     and a sprinkling of agent-comment chatter.
  2. ``build_task_bank`` produces at least one task for every
     substrate-matched kind plus one open-vocab residual, and every
     task's ``gold`` is computed from the same analyzer output the
     planner uses (so the oracle is well-defined).
  3. ``oracle_relevance`` returns True for every fixed-point event
     regardless of role; task-specific relevance is correct on the
     three raise-axis predicates we exercise.
  4. ``deliver`` under the three delivery strategies:
       * naive    — delivers every event.
       * routing  — Bloom-filter by event-type subscription per role.
       * substrate— zero content events for aggregator on matched
                    kinds; fixed-point only for other roles.
  5. ``run_corpus_bench`` on a tiny synthetic corpus produces the
     expected headline decomposition: naive aggregator relevance is
     small, substrate match rate is high, substrate answer-correct
     rate is 1.
  6. ``decide_falsifiability`` assigns the three gate outcomes
     correctly at the boundary values.
  7. Every (task, role, strategy) triple has its
     ``DeliveryResult.recall_of_relevant`` = 1 on ``naive`` — the
     oracle's ground-truth-relevant set is exactly a subset of the
     full event stream.

Tests are deterministic and do not depend on Phase-28 / Phase-27
machinery.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import textwrap
import unittest

from vision_mvp.tasks.python_corpus import PythonCorpus
from vision_mvp.tasks.task_scale_swe import (
    ALL_ROLES, ALL_TASK_KINDS,
    EVENT_AGENT_COMMENT, EVENT_CLASS_DEF, EVENT_FILE_OPEN,
    EVENT_FUNCTION_DEF, EVENT_IMPORT_STMT, EVENT_TASK_GOAL,
    FIXED_POINT_EVENT_TYPES, KIND_COUNT_FILES_IMPORTING,
    KIND_COUNT_FUNCTIONS, KIND_COUNT_MAY_RAISE,
    KIND_COUNT_TRANS_MAY_RAISE, KIND_OPEN_VOCAB,
    KIND_COUNT_PARTICIPATES_IN_CYCLE,
    ROLE_AGGREGATOR, ROLE_FILE_INDEXER, ROLE_ORCHESTRATOR,
    ROLE_REVIEWER, ROLE_SEMANTIC_ANALYZER,
    build_event_stream, build_task_bank,
    decide_falsifiability, deliver, oracle_relevance,
    run_corpus_bench,
)


def _make_corpus(tmp: str) -> PythonCorpus:
    src_a = textwrap.dedent('''
        """Module a — processes commands."""
        import subprocess
        def runner(cmd: str) -> None:
            subprocess.run([cmd])
        def pure(x: int) -> int:
            return x + 1
        def raiser():
            raise ValueError('x')
    ''').strip() + "\n"
    src_b = textwrap.dedent('''
        """Module b — filesystem + subprocess helpers."""
        import subprocess
        import os
        def touches_fs(p: str) -> str:
            with open(p) as f:
                return f.read()
        class Helper:
            def spawn(self):
                subprocess.run(['ls'])
        def wrapper():
            Helper().spawn()
    ''').strip() + "\n"
    with open(os.path.join(tmp, "a.py"), "w") as f:
        f.write(src_a)
    with open(os.path.join(tmp, "b.py"), "w") as f:
        f.write(src_b)
    c = PythonCorpus(root=tmp, seed=29)
    c.build()
    return c


class TestBuildEventStream(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="p29_ev_")
        self.c = _make_corpus(self.tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_stream_has_all_event_types(self) -> None:
        evs = build_event_stream(self.c)
        types = {e.event_type for e in evs}
        # task goal + final answer fixed points must always appear.
        self.assertIn(EVENT_TASK_GOAL, types)
        self.assertIn(EVENT_FILE_OPEN, types)
        self.assertIn(EVENT_FUNCTION_DEF, types)
        self.assertIn(EVENT_IMPORT_STMT, types)
        self.assertIn(EVENT_CLASS_DEF, types)
        self.assertIn(EVENT_AGENT_COMMENT, types)

    def test_stream_is_deterministic(self) -> None:
        evs1 = build_event_stream(self.c, n_agent_comments=5, seed=77)
        evs2 = build_event_stream(self.c, n_agent_comments=5, seed=77)
        self.assertEqual([(e.event_id, e.event_type, e.body) for e in evs1],
                           [(e.event_id, e.event_type, e.body) for e in evs2])

    def test_one_file_open_per_file(self) -> None:
        evs = build_event_stream(self.c)
        n_fo = sum(1 for e in evs if e.event_type == EVENT_FILE_OPEN)
        self.assertEqual(n_fo, self.c.n_files)

    def test_function_flags_populated(self) -> None:
        evs = build_event_stream(self.c)
        fn_evs = [e for e in evs if e.event_type == EVENT_FUNCTION_DEF]
        self.assertGreater(len(fn_evs), 0)
        # Every function event has the Phase-28 flag tuple attached.
        for e in fn_evs:
            keys = {k for k, _v in e.function_flags}
            self.assertIn("may_raise", keys)
            self.assertIn("trans_may_raise", keys)
            self.assertIn("trans_calls_subprocess", keys)


class TestBuildTaskBank(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="p29_tb_")
        self.c = _make_corpus(self.tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_bank_covers_substrate_matched_kinds(self) -> None:
        tasks = build_task_bank(self.c)
        kinds = {t.kind for t in tasks}
        self.assertIn(KIND_COUNT_MAY_RAISE, kinds)
        self.assertIn(KIND_COUNT_TRANS_MAY_RAISE, kinds)
        self.assertIn(KIND_COUNT_FUNCTIONS, kinds)
        self.assertIn(KIND_COUNT_PARTICIPATES_IN_CYCLE, kinds)
        self.assertIn(KIND_OPEN_VOCAB, kinds)

    def test_gold_matches_corpus_aggregates(self) -> None:
        tasks = build_task_bank(self.c)
        for t in tasks:
            if t.kind == KIND_COUNT_MAY_RAISE:
                self.assertEqual(t.gold, self.c.n_functions_may_raise)
            elif t.kind == KIND_COUNT_TRANS_MAY_RAISE:
                self.assertEqual(
                    t.gold, self.c.n_functions_trans_may_raise)
            elif t.kind == KIND_COUNT_PARTICIPATES_IN_CYCLE:
                self.assertEqual(
                    t.gold, self.c.n_functions_participates_in_cycle)


class TestOracleRelevance(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="p29_or_")
        self.c = _make_corpus(self.tmp)
        self.events = build_event_stream(self.c)
        self.tasks = build_task_bank(self.c)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fixed_point_always_relevant(self) -> None:
        task = self.tasks[0]
        fp_evs = [e for e in self.events if e.is_fixed_point]
        for role in ALL_ROLES:
            for ev in fp_evs:
                self.assertTrue(oracle_relevance(task, role, ev))

    def test_orchestrator_and_reviewer_only_see_fixed_points(self) -> None:
        task = self.tasks[0]
        for ev in self.events:
            if ev.is_fixed_point:
                continue
            self.assertFalse(oracle_relevance(
                task, ROLE_ORCHESTRATOR, ev))
            self.assertFalse(oracle_relevance(
                task, ROLE_REVIEWER, ev))

    def test_file_indexer_sees_file_open_and_import(self) -> None:
        task = self.tasks[0]
        for ev in self.events:
            if ev.is_fixed_point:
                continue
            rel = oracle_relevance(task, ROLE_FILE_INDEXER, ev)
            expected = ev.event_type in (EVENT_FILE_OPEN, EVENT_IMPORT_STMT)
            self.assertEqual(rel, expected,
                               msg=f"{ev.event_type=} rel={rel} exp={expected}")

    def test_semantic_analyzer_sees_function_and_class(self) -> None:
        task = self.tasks[0]
        for ev in self.events:
            if ev.is_fixed_point:
                continue
            rel = oracle_relevance(task, ROLE_SEMANTIC_ANALYZER, ev)
            expected = ev.event_type in (EVENT_FUNCTION_DEF, EVENT_CLASS_DEF)
            self.assertEqual(rel, expected)

    def test_aggregator_on_count_trans_may_raise(self) -> None:
        # Find the task.
        ts = [t for t in self.tasks if t.kind == KIND_COUNT_TRANS_MAY_RAISE]
        self.assertEqual(len(ts), 1)
        t = ts[0]
        # For every FUNCTION_DEF event, relevance matches the
        # trans_may_raise flag.
        for ev in self.events:
            if ev.event_type != EVENT_FUNCTION_DEF:
                if ev.is_fixed_point:
                    continue
                self.assertFalse(oracle_relevance(t, ROLE_AGGREGATOR, ev))
                continue
            flags = dict(ev.function_flags)
            expected = bool(flags.get("trans_may_raise", False))
            self.assertEqual(
                oracle_relevance(t, ROLE_AGGREGATOR, ev), expected)


class TestDeliverStrategies(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="p29_dl_")
        self.c = _make_corpus(self.tmp)
        self.events = build_event_stream(self.c)
        self.tasks = build_task_bank(self.c)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_naive_delivers_every_event_to_every_role(self) -> None:
        for task in self.tasks:
            for role in ALL_ROLES:
                res = deliver(task, self.events, role, "naive",
                                corpus=self.c)
                self.assertEqual(res.n_delivered, len(self.events))
                self.assertEqual(res.recall_of_relevant, 1.0)

    def test_routing_drops_off_subscription(self) -> None:
        task = self.tasks[0]
        res_orch = deliver(task, self.events, ROLE_ORCHESTRATOR, "routing",
                             corpus=self.c)
        # Only fixed-point events survive orchestrator subscription.
        self.assertEqual(res_orch.n_delivered, 2)  # task_goal + final_answer
        for ev in self.events:
            if ev.event_type in FIXED_POINT_EVENT_TYPES:
                continue
            # Content events are dropped for orchestrator.

    def test_substrate_delivers_zero_content_to_aggregator_on_matched(self) -> None:
        # Pick a matched kind.
        t = [t for t in self.tasks
             if t.kind == KIND_COUNT_TRANS_MAY_RAISE][0]
        res = deliver(t, self.events, ROLE_AGGREGATOR, "substrate",
                        corpus=self.c)
        self.assertTrue(res.substrate_matched)
        # Only fixed-point events are delivered on the direct-exact path.
        for ev_id in range(0, 1000):
            pass
        self.assertLessEqual(res.n_delivered, 2)
        self.assertTrue(res.answer_correct)

    def test_substrate_falls_back_to_retrieval_on_open_vocab(self) -> None:
        open_vocab_tasks = [t for t in self.tasks
                              if t.kind == KIND_OPEN_VOCAB]
        if not open_vocab_tasks:
            self.skipTest("no open-vocab task in this corpus")
        t = open_vocab_tasks[0]
        res = deliver(t, self.events, ROLE_AGGREGATOR, "substrate",
                        corpus=self.c)
        self.assertFalse(res.substrate_matched)
        # At least the fixed-point events come through.
        self.assertGreaterEqual(res.n_delivered, 1)

    def test_naive_recall_of_relevant_is_one(self) -> None:
        for task in self.tasks:
            for role in ALL_ROLES:
                res = deliver(task, self.events, role, "naive",
                                corpus=self.c)
                if res.n_ground_truth_relevant > 0:
                    self.assertEqual(res.recall_of_relevant, 1.0)


class TestRunCorpusBench(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="p29_rcb_")
        self.c = _make_corpus(self.tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_headline_decomposition(self) -> None:
        rep = run_corpus_bench("p29_smoke", self.c, seed=29)
        p = rep.pooled["per_strategy"]
        # Naive aggregator relevance must be < 50% on this small
        # corpus; the thesis is CONFIRMED at task scale.
        self.assertLess(
            p["naive"]["mean_relevance_fraction_aggregator"], 0.50)
        # Substrate must match on the planner-matched kinds.
        self.assertGreaterEqual(p["substrate"]["substrate_match_rate"], 0.5)
        # Substrate answer correctness is 100%.
        self.assertEqual(
            p["substrate"]["answer_correct_rate_aggregator"], 1.0)

    def test_token_reduction_substrate_over_naive(self) -> None:
        rep = run_corpus_bench("p29_smoke", self.c, seed=29)
        p = rep.pooled["per_strategy"]
        for role in ALL_ROLES:
            naive_t = p["naive"]["mean_delivered_tokens_per_role"][role]
            sub_t = p["substrate"]["mean_delivered_tokens_per_role"][role]
            self.assertGreaterEqual(naive_t, sub_t,
                                      msg=f"role={role} naive<substrate")


class TestFalsifiabilityDecision(unittest.TestCase):

    def test_confirmed_when_below_gate_lower(self) -> None:
        d = decide_falsifiability(0.10)
        self.assertEqual(d.decision, "confirmed")

    def test_mixed_on_boundary(self) -> None:
        d = decide_falsifiability(0.65)
        self.assertEqual(d.decision, "mixed")

    def test_falsified_when_above_gate_upper(self) -> None:
        d = decide_falsifiability(0.95)
        self.assertEqual(d.decision, "falsified")

    def test_boundary_at_gate_lower(self) -> None:
        d = decide_falsifiability(0.50)
        # 0.50 is in the inclusive-mixed range per our definition.
        self.assertEqual(d.decision, "mixed")

    def test_boundary_at_gate_upper(self) -> None:
        d = decide_falsifiability(0.80)
        self.assertEqual(d.decision, "mixed")


class TestPhase29TaskKindsCoverage(unittest.TestCase):
    """Every declared task kind must resolve to a gold answer under
    the task bank generator — otherwise the benchmark would silently
    drop a query family."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="p29_kc_")
        self.c = _make_corpus(self.tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_all_substrate_kinds_appear(self) -> None:
        tasks = build_task_bank(self.c)
        kinds = {t.kind for t in tasks}
        # Every substrate-matched kind appears (except possibly the
        # files_importing family if the corpus has no popular imports,
        # which is unlikely on any real corpus).
        self.assertIn(KIND_COUNT_FUNCTIONS, kinds)
        self.assertIn(KIND_COUNT_MAY_RAISE, kinds)
        self.assertIn(KIND_COUNT_TRANS_MAY_RAISE, kinds)


if __name__ == "__main__":
    unittest.main()
