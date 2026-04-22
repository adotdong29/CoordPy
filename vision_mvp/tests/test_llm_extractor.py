"""Unit tests for the Phase-33 LLM-driven extractor and its
empirical-noise calibration layer.

Covers:
  * Prompt construction determinism.
  * Claim parsing (well-formed + malformed replies).
  * ``LLMExtractor`` end-to-end with a deterministic mock LLM.
  * Cache semantics (hit counter, key stability).
  * ``ClaimComparator`` attribution on hand-crafted examples.
  * ``calibrate_extractor`` pipeline on the Phase-31 scenario bank
    with (i) the perfect regex extractor, (ii) a deterministic-drop
    mock LLM.
  * ``closest_synthetic_config`` matching against a small pooled dict.
"""

from __future__ import annotations

import unittest

from vision_mvp.core.extractor_calibration import (
    ClaimComparator, calibrate_extractor,
    closest_synthetic_config, compare_to_synthetic_curve,
    pool_comparisons,
)
from vision_mvp.core.extractor_noise import (
    incident_triage_known_kinds, compliance_review_known_kinds,
)
from vision_mvp.core.llm_extractor import (
    DeterministicCache, DeterministicMockExtractorLLM, LLMExtractor,
    LLMExtractorConfig, LLMExtractorStats,
    build_extractor_prompt, parse_llm_claims,
)
from vision_mvp.tasks.incident_triage import (
    ROLE_DB_ADMIN, ROLE_MONITOR, ROLE_NETWORK, ROLE_SYSADMIN,
    build_scenario_bank, extract_claims_for_role,
)


class TestPrompt(unittest.TestCase):

    def test_prompt_is_deterministic(self):
        s = build_scenario_bank(seed=31, distractors_per_role=4)[0]
        events = list(s.per_role_events[ROLE_SYSADMIN])
        cfg = LLMExtractorConfig()
        a = build_extractor_prompt(ROLE_SYSADMIN, events,
                                    ["DISK_FILL_CRITICAL"], cfg)
        b = build_extractor_prompt(ROLE_SYSADMIN, events,
                                    ["DISK_FILL_CRITICAL"], cfg)
        self.assertEqual(a, b)

    def test_prompt_includes_event_body(self):
        s = build_scenario_bank(seed=31, distractors_per_role=0)[0]
        events = list(s.per_role_events[ROLE_SYSADMIN])
        cfg = LLMExtractorConfig()
        p = build_extractor_prompt(ROLE_SYSADMIN, events,
                                    ["DISK_FILL_CRITICAL"], cfg)
        self.assertIn("used=99%", p)

    def test_prompt_disables_event_ids(self):
        s = build_scenario_bank(seed=31, distractors_per_role=0)[0]
        events = list(s.per_role_events[ROLE_SYSADMIN])
        cfg = LLMExtractorConfig(include_event_ids=False)
        p = build_extractor_prompt(ROLE_SYSADMIN, events,
                                    ["DISK_FILL_CRITICAL"], cfg)
        self.assertNotIn("id=0", p)


class TestParser(unittest.TestCase):

    def test_parse_single_claim(self):
        txt = '{"kind": "DISK_FILL_CRITICAL", ' \
              '"payload": "used=99%", "event_ids": [12]}'
        out = parse_llm_claims(txt)
        self.assertEqual(
            out, [("DISK_FILL_CRITICAL", "used=99%", (12,))])

    def test_parse_multiple_claims(self):
        txt = ('{"kind": "A", "payload": "x", "event_ids": [1]}\n'
               '{"kind": "B", "payload": "y", "event_ids": [2, 3]}')
        out = parse_llm_claims(txt)
        self.assertEqual(len(out), 2)

    def test_parse_ignores_surrounding_prose(self):
        txt = ('here is my answer:\n'
               '{"kind": "A", "payload": "x", "event_ids": [1]}\n'
               'thanks!')
        self.assertEqual(len(parse_llm_claims(txt)), 1)

    def test_parse_handles_malformed_gracefully(self):
        # malformed JSON still OK for other matches
        txt = ('{"kind": "A", "payload": "x", "event_ids": [1]}\n'
               '{"bad": "no_kind"}')
        self.assertEqual(len(parse_llm_claims(txt)), 1)


class TestLLMExtractor(unittest.TestCase):

    def _ext(self, drop_prob: float = 0.0,
             spurious_kind: str | None = None) -> LLMExtractor:
        mock = DeterministicMockExtractorLLM(
            keyword_to_kind={
                "used=99": "DISK_FILL_CRITICAL",
                "exit=137": "CRON_OVERRUN",
                "duration_s=5400": "CRON_OVERRUN",
                "oom_kill": "OOM_KILL",
            },
            drop_prob=drop_prob,
            spurious_body=("spurious")
            if spurious_kind else None,
            spurious_kind=spurious_kind)
        return LLMExtractor(
            llm_call=mock,
            known_kinds_by_role=incident_triage_known_kinds(),
            cache=DeterministicCache(model="mock"))

    def test_extractor_emits_known_kinds(self):
        s = build_scenario_bank(seed=31, distractors_per_role=0)[0]
        events = list(s.per_role_events[ROLE_SYSADMIN])
        ext = self._ext()
        out = ext(ROLE_SYSADMIN, events, s)
        kinds = {k for (k, _, _) in out}
        self.assertTrue(kinds.issubset(
            {"DISK_FILL_CRITICAL", "CRON_OVERRUN", "OOM_KILL"}))

    def test_extractor_cache_hit(self):
        s = build_scenario_bank(seed=31, distractors_per_role=4)[0]
        events = list(s.per_role_events[ROLE_SYSADMIN])
        ext = self._ext()
        _ = ext(ROLE_SYSADMIN, events, s)
        self.assertEqual(ext.stats.n_cache_hits, 0)
        _ = ext(ROLE_SYSADMIN, events, s)
        self.assertEqual(ext.stats.n_cache_hits, 1)

    def test_extractor_filters_unknown_kinds(self):
        s = build_scenario_bank(seed=31, distractors_per_role=0)[0]
        events = list(s.per_role_events[ROLE_SYSADMIN])
        ext = self._ext(spurious_kind="BOGUS_KIND_NOT_IN_POOL")
        out = ext(ROLE_SYSADMIN, events, s)
        # Spurious kind filtered; still have the two real emissions.
        kinds = {k for (k, _, _) in out}
        self.assertNotIn("BOGUS_KIND_NOT_IN_POOL", kinds)
        self.assertGreater(ext.stats.n_dropped_unknown_kind, 0)


class TestClaimComparator(unittest.TestCase):

    def setUp(self):
        self.cmp = ClaimComparator()

    def test_perfect_extractor_no_drops(self):
        gold = [
            ("r1", "K1", "payload1 a b c", (1,)),
            ("r1", "K2", "payload2 x y", (2, 3)),
        ]
        emissions = [
            ("K1", "payload1 a b c extra", (1,)),
            ("K2", "payload2 x y", (2,)),
        ]
        c = self.cmp.classify("r1", emissions, gold, [1, 2, 3])
        self.assertEqual(c["n_correct"], 2)
        self.assertEqual(c["n_dropped"], 0)
        self.assertEqual(c["n_mislabeled"], 0)
        self.assertEqual(c["n_spurious"], 0)

    def test_drop_counted(self):
        gold = [
            ("r1", "K1", "p", (1,)),
            ("r1", "K2", "p", (2,)),
        ]
        emissions = [("K1", "p", (1,))]
        c = self.cmp.classify("r1", emissions, gold, [1, 2])
        self.assertEqual(c["n_dropped"], 1)
        self.assertEqual(c["n_correct"], 1)

    def test_mislabel_on_causal_event(self):
        gold = [("r1", "K1", "p", (1,))]
        emissions = [("K2", "p", (1,))]
        c = self.cmp.classify("r1", emissions, gold, [1])
        self.assertEqual(c["n_dropped"], 1)
        self.assertEqual(c["n_mislabeled"], 1)

    def test_spurious_on_distractor(self):
        gold = [("r1", "K1", "p", (1,))]
        emissions = [("K1", "p", (1,)), ("K1", "p2", (99,))]
        c = self.cmp.classify("r1", emissions, gold, [1])
        self.assertEqual(c["n_correct"], 1)
        self.assertEqual(c["n_spurious"], 1)

    def test_payload_corruption_detects_missing_token(self):
        gold = [("r1", "K1", "alpha beta gamma", (1,))]
        emissions = [("K1", "alpha gamma", (1,))]  # missing "beta"
        c = self.cmp.classify("r1", emissions, gold, [1])
        self.assertEqual(c["n_payload_corrupted"], 1)


class TestCalibratePipeline(unittest.TestCase):

    def _getters(self):
        def role_events(scenario, role):
            return list(scenario.per_role_events.get(role, ()))

        def causal_ids(scenario, role):
            return [ev.event_id
                    for ev in scenario.per_role_events.get(role, ())
                    if ev.is_causal]

        def gold_chain(scenario):
            return scenario.causal_chain
        return role_events, causal_ids, gold_chain

    def test_regex_extractor_calibrates_near_perfect(self):
        scenarios = build_scenario_bank(seed=31,
                                          distractors_per_role=6)
        role_events, causal_ids, gold_chain = self._getters()
        audit = calibrate_extractor(
            extract_claims_for_role, scenarios,
            role_events, causal_ids, gold_chain,
            roles_to_probe=[
                ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN,
                ROLE_NETWORK],
            extractor_label="regex")
        self.assertEqual(audit.drop_rate, 0.0)
        self.assertLess(audit.spurious_per_event, 0.05)

    def test_synthetic_match_finds_closest(self):
        # Audit claims drop=0.10, spurious=0.02, mislabel=0, pc=0.
        audit = pool_comparisons(
            per_role_counts={
                "role_a": {"n_dropped": 1, "n_correct": 9,
                            "n_mislabeled": 0, "n_spurious": 2,
                            "n_payload_corrupted": 0,
                            "n_gold": 10, "n_emissions": 11,
                            "n_distractor_events": 100},
            },
            extractor_label="unit",
            n_scenarios=1)
        sweep = {
            "d_drop0.0_sp0.0_mis0.0_corr0.0": {
                "domain": "d",
                "drop_prob": 0.0, "spurious_prob": 0.0,
                "mislabel_prob": 0.0,
                "payload_corrupt_prob": 0.0,
                "accuracy_mean": 1.0, "recall_mean": 1.0,
                "precision_mean": 1.0, "tokens_mean": 200},
            "d_drop0.1_sp0.05_mis0.0_corr0.0": {
                "domain": "d",
                "drop_prob": 0.1, "spurious_prob": 0.05,
                "mislabel_prob": 0.0,
                "payload_corrupt_prob": 0.0,
                "accuracy_mean": 0.7, "recall_mean": 0.85,
                "precision_mean": 0.43, "tokens_mean": 264},
            "d_drop0.5_sp0.1_mis0.0_corr0.0": {
                "domain": "d",
                "drop_prob": 0.5, "spurious_prob": 0.1,
                "mislabel_prob": 0.0,
                "payload_corrupt_prob": 0.0,
                "accuracy_mean": 0.0, "recall_mean": 0.45,
                "precision_mean": 0.34, "tokens_mean": 245},
        }
        match = closest_synthetic_config(audit, sweep, domain="d")
        self.assertIsNotNone(match)
        self.assertEqual(match.drop_prob, 0.1)

    def test_compare_verdict(self):
        audit = pool_comparisons({
            "r": {"n_dropped": 0, "n_correct": 10, "n_mislabeled": 0,
                   "n_spurious": 0, "n_payload_corrupted": 0,
                   "n_gold": 10, "n_emissions": 10,
                   "n_distractor_events": 50}},
            extractor_label="x", n_scenarios=1)
        sweep = {
            "d_drop0.0_sp0.0_mis0.0_corr0.0": {
                "domain": "d", "drop_prob": 0.0, "spurious_prob": 0.0,
                "mislabel_prob": 0.0, "payload_corrupt_prob": 0.0,
                "accuracy_mean": 1.0, "recall_mean": 1.0,
                "precision_mean": 1.0, "tokens_mean": 200},
        }
        rep = compare_to_synthetic_curve(
            audit, sweep,
            real_measured_accuracy=1.0,
            real_measured_recall=1.0,
            real_measured_precision=1.0,
            domain="d")
        self.assertEqual(rep["verdict"], "approximates")


if __name__ == "__main__":
    unittest.main()
