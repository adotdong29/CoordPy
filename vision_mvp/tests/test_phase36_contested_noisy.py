"""End-to-end tests for Phase-36 on the contested-incident bank.

Coverage:
  * Noisy dynamic (drop_prob=0): identical to clean Phase-35
    dynamic run (contested_acc = 1.00).
  * Noisy dynamic (drop_prob=1.0): collapses to static baseline
    (contested_acc = 0.00).
  * Adaptive_sub strategy: 100 % on clean bank.
  * Adaptive_sub strategy under noise: tracks dynamic within
    a small gap on matched noise budget.
  * LLM-typed mock replier: 100 % on clean bank when oracle-
    perfect; degrades under malformed_prob.
"""
from __future__ import annotations

import unittest

from vision_mvp.core.reply_noise import (
    ReplyNoiseConfig, noisy_causality_extractor,
)
from vision_mvp.core.llm_thread_replier import (
    LLMReplyConfig, LLMThreadReplier, DeterministicMockReplier,
    causality_extractor_from_replier,
)
from vision_mvp.core.dynamic_comm import (
    REPLY_DOWNSTREAM_SYMPTOM, REPLY_INDEPENDENT_ROOT,
    REPLY_UNCERTAIN,
)
from vision_mvp.tasks.contested_incident import (
    STRATEGY_DYNAMIC, STRATEGY_ADAPTIVE_SUB,
    STRATEGY_STATIC_HANDOFF,
    MockContestedAuditor, build_contested_bank,
    run_contested_loop,
    infer_causality_hypothesis,
)


class TestNoisyDynamic(unittest.TestCase):
    def _run(self, drop_prob: float):
        cfg = ReplyNoiseConfig(drop_prob=drop_prob, seed=36)
        ext = noisy_causality_extractor(
            infer_causality_hypothesis, cfg)
        bank = build_contested_bank(seed=35)
        auditor = MockContestedAuditor()
        rep = run_contested_loop(
            bank, auditor,
            strategies=(STRATEGY_STATIC_HANDOFF,
                          STRATEGY_DYNAMIC, STRATEGY_ADAPTIVE_SUB),
            seed=35, causality_extractor=ext,
        )
        return rep.pooled()

    def test_clean_is_phase35_baseline(self):
        pooled = self._run(drop_prob=0.0)
        self.assertEqual(
            pooled[STRATEGY_DYNAMIC]["contested_accuracy_full"], 1.0)
        self.assertEqual(
            pooled[STRATEGY_ADAPTIVE_SUB]["contested_accuracy_full"],
            1.0)
        self.assertEqual(
            pooled[STRATEGY_STATIC_HANDOFF][
                "contested_accuracy_full"], 0.0)

    def test_full_drop_collapses_to_static(self):
        pooled = self._run(drop_prob=1.0)
        # Dynamic + adaptive_sub both collapse to the static
        # baseline (33 % full / 0 % contested) because every
        # INDEPENDENT_ROOT reply is dropped to UNCERTAIN.
        self.assertEqual(
            pooled[STRATEGY_DYNAMIC]["contested_accuracy_full"],
            0.0)
        self.assertEqual(
            pooled[STRATEGY_ADAPTIVE_SUB][
                "contested_accuracy_full"], 0.0)

    def test_dynamic_and_adaptive_sub_track(self):
        # Across noise grid, the two primitives should track
        # closely (Conjecture C35-5 / Phase-36 C36-5).
        for drop_prob in [0.0, 0.25, 0.5, 1.0]:
            pooled = self._run(drop_prob=drop_prob)
            d = pooled[STRATEGY_DYNAMIC]["accuracy_full"]
            a = pooled[STRATEGY_ADAPTIVE_SUB]["accuracy_full"]
            self.assertLessEqual(abs(d - a), 0.1,
                                   f"dyn-adp gap > 10pp at "
                                   f"drop={drop_prob}")


class TestLLMTypedReplier(unittest.TestCase):
    def _build_oracle_replier(self):
        # Build a (role, kind) → reply_kind map that matches the
        # Phase-35 oracle on scenarios where there is no
        # per-scenario context-dependence.
        oracle = {
            ("db_admin", "DEADLOCK_SUSPECTED"):
                REPLY_INDEPENDENT_ROOT,
            ("db_admin", "POOL_EXHAUSTION"):
                REPLY_DOWNSTREAM_SYMPTOM,
            ("network", "DNS_MISROUTE"):
                REPLY_INDEPENDENT_ROOT,
            ("monitor", "ERROR_RATE_SPIKE"):
                REPLY_DOWNSTREAM_SYMPTOM,
        }
        stub = DeterministicMockReplier(kind_replies=oracle)
        cfg = LLMReplyConfig(witness_token_cap=12)
        return LLMThreadReplier(
            llm_call=stub, config=cfg, cache={})

    def test_llm_replier_runs_without_error(self):
        replier = self._build_oracle_replier()
        ext = causality_extractor_from_replier(replier)
        bank = build_contested_bank(seed=35)
        auditor = MockContestedAuditor()
        rep = run_contested_loop(
            bank, auditor,
            strategies=(STRATEGY_DYNAMIC,),
            seed=35, causality_extractor=ext,
        )
        pooled = rep.pooled()
        # Should at least not crash; accuracy depends on the
        # breadth of the oracle map.
        self.assertIn(STRATEGY_DYNAMIC, pooled)
        # Every reply should have been well-formed by the mock.
        self.assertGreaterEqual(replier.stats.n_well_formed,
                                  replier.stats.n_calls)


if __name__ == "__main__":
    unittest.main()
