"""Unit tests for ``vision_mvp.core.llm_thread_replier`` — the
Phase-36 Part-B LLM-driven thread reply path.

Coverage:
  * parse_llm_reply_json: well-formed, malformed, out-of-vocab.
  * Witness clamp to witness_token_cap.
  * DeterministicMockReplier returns an allowed reply_kind.
  * causality_extractor_from_replier yields the INDEPENDENT_ROOT
    / DOWNSTREAM_SYMPTOM_OF / UNCERTAIN string shape.
  * LLMThreadReplier.stats counts well-formed vs malformed
    vs out-of-vocab.
  * Cache returns cached replies without calling the LLM twice.
"""
from __future__ import annotations

import unittest

from vision_mvp.core.dynamic_comm import (
    REPLY_DOWNSTREAM_SYMPTOM, REPLY_INDEPENDENT_ROOT,
    REPLY_UNCERTAIN,
)
from vision_mvp.core.llm_thread_replier import (
    DEFAULT_REPLY_KINDS, LLMReplyConfig, LLMReplierStats,
    LLMThreadReplier, DeterministicMockReplier,
    causality_extractor_from_replier,
    parse_llm_reply_json,
)


class TestParser(unittest.TestCase):
    def test_well_formed(self):
        cfg = LLMReplyConfig()
        reply, witness, ok = parse_llm_reply_json(
            '{"reply_kind": "INDEPENDENT_ROOT", '
            '"witness": "deadlock orders"}', cfg)
        self.assertEqual(reply, REPLY_INDEPENDENT_ROOT)
        self.assertEqual(witness, "deadlock orders")
        self.assertTrue(ok)

    def test_malformed_falls_back(self):
        cfg = LLMReplyConfig(fallback_reply_kind=REPLY_UNCERTAIN)
        reply, witness, ok = parse_llm_reply_json(
            "I think the answer is probably DEADLOCK.", cfg)
        self.assertEqual(reply, REPLY_UNCERTAIN)
        self.assertEqual(witness, "")
        self.assertFalse(ok)

    def test_out_of_vocab_falls_back(self):
        cfg = LLMReplyConfig(
            allowed_reply_kinds=DEFAULT_REPLY_KINDS,
            fallback_reply_kind=REPLY_UNCERTAIN)
        reply, _witness, ok = parse_llm_reply_json(
            '{"reply_kind": "SOMETHING_ELSE", "witness": "foo"}',
            cfg)
        self.assertEqual(reply, REPLY_UNCERTAIN)
        self.assertFalse(ok)

    def test_witness_clamped(self):
        cfg = LLMReplyConfig(witness_token_cap=3)
        reply, witness, ok = parse_llm_reply_json(
            '{"reply_kind": "INDEPENDENT_ROOT", '
            '"witness": "one two three four five six"}', cfg)
        self.assertTrue(ok)
        self.assertEqual(len(witness.split()), 3)


class TestDeterministicMockReplier(unittest.TestCase):
    def test_mock_returns_expected_reply_kind(self):
        stub = DeterministicMockReplier(kind_replies={
            ("db_admin", "DEADLOCK"): REPLY_INDEPENDENT_ROOT,
        })
        cfg = LLMReplyConfig()
        replier = LLMThreadReplier(llm_call=stub, config=cfg)
        reply_kind, witness, ok = replier(
            scenario=object(),
            role="db_admin", kind="DEADLOCK",
            payload="deadlock relation=orders",
            other_candidates=(("sysadmin", "CRON", "cron ..."),),
            role_events=None,
        )
        self.assertTrue(ok)
        self.assertEqual(reply_kind, REPLY_INDEPENDENT_ROOT)

    def test_mock_malformed_degrades(self):
        stub = DeterministicMockReplier(
            kind_replies={("db_admin", "X"): REPLY_INDEPENDENT_ROOT},
            malformed_prob=1.0)
        cfg = LLMReplyConfig(fallback_reply_kind=REPLY_UNCERTAIN)
        replier = LLMThreadReplier(llm_call=stub, config=cfg)
        reply_kind, _witness, ok = replier(
            scenario=object(),
            role="db_admin", kind="X",
            payload="...")
        self.assertFalse(ok)
        self.assertEqual(reply_kind, REPLY_UNCERTAIN)


class TestCausalityExtractorAdaptor(unittest.TestCase):
    def test_independent_root_preserved(self):
        stub = DeterministicMockReplier(kind_replies={
            ("db_admin", "DEADLOCK"): REPLY_INDEPENDENT_ROOT,
        })
        cfg = LLMReplyConfig()
        replier = LLMThreadReplier(llm_call=stub, config=cfg)
        ext = causality_extractor_from_replier(replier)
        out = ext(object(), "db_admin", "DEADLOCK", "...")
        self.assertEqual(out, "INDEPENDENT_ROOT")

    def test_downstream_formatted(self):
        stub = DeterministicMockReplier(kind_replies={
            ("db_admin", "POOL"): REPLY_DOWNSTREAM_SYMPTOM,
        })
        cfg = LLMReplyConfig()
        replier = LLMThreadReplier(llm_call=stub, config=cfg)
        ext = causality_extractor_from_replier(replier)
        out = ext(object(), "db_admin", "POOL", "...")
        self.assertTrue(out.startswith("DOWNSTREAM_SYMPTOM_OF:"))
        self.assertTrue(out.endswith("POOL"))

    def test_uncertain_default(self):
        stub = DeterministicMockReplier(kind_replies={})
        cfg = LLMReplyConfig()
        replier = LLMThreadReplier(llm_call=stub, config=cfg)
        ext = causality_extractor_from_replier(replier)
        out = ext(object(), "sysadmin", "CRON", "...")
        self.assertEqual(out, "UNCERTAIN")


class TestStats(unittest.TestCase):
    def test_stats_counts_well_formed(self):
        stub = DeterministicMockReplier(
            kind_replies={("r", "k"): REPLY_INDEPENDENT_ROOT})
        cfg = LLMReplyConfig()
        replier = LLMThreadReplier(llm_call=stub, config=cfg)
        replier(object(), "r", "k", "payload")
        self.assertEqual(replier.stats.n_calls, 1)
        self.assertEqual(replier.stats.n_well_formed, 1)
        self.assertEqual(replier.stats.n_malformed, 0)

    def test_cache_prevents_double_call(self):
        stub = DeterministicMockReplier(
            kind_replies={("r", "k"): REPLY_INDEPENDENT_ROOT})
        cfg = LLMReplyConfig()
        cache: dict[str, str] = {}
        replier = LLMThreadReplier(
            llm_call=stub, config=cfg, cache=cache)

        class _Scenario:
            scenario_id = "sx"

        replier(_Scenario(), "r", "k", "payload")
        calls_after_first = replier.stats.n_calls
        replier(_Scenario(), "r", "k", "payload")
        # Cache hit: no new LLM call.
        self.assertEqual(replier.stats.n_calls, calls_after_first)
        self.assertEqual(replier.stats.n_cache_hits, 1)


if __name__ == "__main__":
    unittest.main()
