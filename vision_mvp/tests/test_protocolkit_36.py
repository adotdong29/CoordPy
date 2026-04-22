"""Sanity tests for the 36-agent ProtocolKit-36 task."""

from __future__ import annotations

import unittest

from vision_mvp.core.code_harness import run_sandboxed
from vision_mvp.tasks.protocolkit_36 import (
    CALL_GRAPH, FUNCTION_SPECS, SPEC_ORDER, TEST_RUNNER_SRC, TEST_WEIGHTS,
    _REF, agent_prompt, compose_module, score_tests,
)


class TestCatalog(unittest.TestCase):
    def test_36_agents(self):
        self.assertEqual(len(SPEC_ORDER), 36)
        self.assertEqual(len(FUNCTION_SPECS), 36)
        self.assertEqual(len(_REF), 36)

    def test_48_tests(self):
        self.assertEqual(len(TEST_WEIGHTS), 48)
        self.assertAlmostEqual(sum(TEST_WEIGHTS.values()), 1.0, places=9)

    def test_call_graph_has_entry_per_agent(self):
        self.assertEqual(set(CALL_GRAPH.keys()), set(SPEC_ORDER))

    def test_tier0_is_producers(self):
        # Producers have empty call-graph entries.
        tier0 = [s for s, deps in CALL_GRAPH.items() if not deps]
        self.assertEqual(len(tier0), 15)   # 15 producers

    def test_integrator_call_3_producers(self):
        integrators = [
            "audit_log_entry", "transfer_request", "position_update",
            "file_upload_result", "throttled_page", "priority_error",
        ]
        for i in integrators:
            self.assertEqual(len(CALL_GRAPH[i]), 3,
                              f"{i} should call 3 producers")


class TestReference(unittest.TestCase):
    def test_reference_passes_all_48(self):
        module = compose_module(_REF)
        result = run_sandboxed(module, TEST_RUNNER_SRC, timeout_s=20)
        score = score_tests(result.per_test)
        self.assertEqual(score["n_passed"], 48, result.stderr)
        self.assertAlmostEqual(score["weighted_score"], 1.0, places=3)


class TestPromptShape(unittest.TestCase):
    def test_prompt_mentions_36(self):
        p = agent_prompt("make_auth_token")
        self.assertIn("36-person", p)
        self.assertIn("make_auth_token", p)

    def test_prompt_includes_dependency_when_provided(self):
        p = agent_prompt("read_event_kind",
                          dependency_outputs={"make_event_header":
                                                "def make_event_header(k, t): return {'kind': k, 'ts': t}"})
        self.assertIn("make_event_header", p)


if __name__ == "__main__":
    unittest.main()
