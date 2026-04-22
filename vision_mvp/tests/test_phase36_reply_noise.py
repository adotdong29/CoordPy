"""Unit tests for ``vision_mvp.core.reply_noise`` — the Phase-36
reply-noise wrappers.

Coverage:
  * Identity wrapper passes through.
  * Drop flips INDEPENDENT_ROOT → UNCERTAIN with the configured
    probability (deterministic per seed).
  * Mislabel flips INDEPENDENT_ROOT → DOWNSTREAM_SYMPTOM_OF:kind
    when it fires.
  * Adversarial drop_root always hits the INDEPENDENT_ROOT reply.
  * Adversarial budget=1 is honoured per scenario.
  * Per-scenario budget is independent.
  * ReplyCorruptionReport records the axis that fired.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass

from vision_mvp.core.reply_noise import (
    ReplyNoiseConfig, ReplyCorruptionReport,
    noisy_causality_extractor,
    AdversarialReplyConfig, adversarial_reply_extractor,
    ADVERSARIAL_REPLY_MODE_DROP_ROOT,
    ADVERSARIAL_REPLY_MODE_FLIP_ROOT_TO_SYMPTOM,
    ADVERSARIAL_REPLY_MODE_INJECT_ROOT_ON_SYMPTOM,
    ADVERSARIAL_REPLY_MODE_COMBINED,
    CAUSALITY_INDEPENDENT_ROOT, CAUSALITY_UNCERTAIN,
    CAUSALITY_DOWNSTREAM_PREFIX,
)


@dataclass(frozen=True)
class _FakeScenario:
    scenario_id: str


def _oracle_extractor(oracle: dict):
    def _ex(scenario, role, kind, payload):
        return oracle.get((role, kind), CAUSALITY_UNCERTAIN)
    return _ex


class TestIdentityNoise(unittest.TestCase):
    def test_identity_passes_through(self):
        base = _oracle_extractor({
            ("db_admin", "DEADLOCK"): CAUSALITY_INDEPENDENT_ROOT,
        })
        cfg = ReplyNoiseConfig()
        self.assertTrue(cfg.is_identity())
        wrapped = noisy_causality_extractor(base, cfg)
        self.assertIs(wrapped, base)


class TestDrop(unittest.TestCase):
    def test_drop_prob_1_flips_independent_root(self):
        base = _oracle_extractor({
            ("db_admin", "DEADLOCK"): CAUSALITY_INDEPENDENT_ROOT,
        })
        cfg = ReplyNoiseConfig(drop_prob=1.0, seed=42)
        report = ReplyCorruptionReport()
        wrapped = noisy_causality_extractor(base, cfg, report=report)
        out = wrapped(_FakeScenario("s1"), "db_admin",
                      "DEADLOCK", "deadlock orders")
        self.assertEqual(out, CAUSALITY_UNCERTAIN)
        self.assertEqual(report.n_dropped, 1)

    def test_drop_prob_0_preserves(self):
        base = _oracle_extractor({
            ("db_admin", "DEADLOCK"): CAUSALITY_INDEPENDENT_ROOT,
        })
        cfg = ReplyNoiseConfig(drop_prob=0.0, seed=42)
        # cfg.is_identity() is True → pass-through wrapper.
        self.assertTrue(cfg.is_identity())

    def test_drop_deterministic_across_calls(self):
        base = _oracle_extractor({
            ("db_admin", "DEADLOCK"): CAUSALITY_INDEPENDENT_ROOT,
        })
        cfg = ReplyNoiseConfig(drop_prob=0.5, seed=7)
        wrapped = noisy_causality_extractor(base, cfg)
        out1 = wrapped(_FakeScenario("s1"), "db_admin",
                        "DEADLOCK", "deadlock orders")
        out2 = wrapped(_FakeScenario("s1"), "db_admin",
                        "DEADLOCK", "deadlock orders")
        self.assertEqual(out1, out2)


class TestMislabel(unittest.TestCase):
    def test_mislabel_prob_1_flips_to_downstream(self):
        base = _oracle_extractor({
            ("db_admin", "DEADLOCK"): CAUSALITY_INDEPENDENT_ROOT,
        })
        cfg = ReplyNoiseConfig(mislabel_prob=1.0, seed=42)
        report = ReplyCorruptionReport()
        wrapped = noisy_causality_extractor(base, cfg, report=report)
        out = wrapped(_FakeScenario("s1"), "db_admin",
                      "DEADLOCK", "...")
        # Oracle INDEPENDENT_ROOT → mislabelled to DOWNSTREAM.
        self.assertTrue(out.startswith(CAUSALITY_DOWNSTREAM_PREFIX))
        self.assertEqual(report.n_mislabeled, 1)

    def test_mislabel_flips_downstream_to_independent(self):
        base = _oracle_extractor({
            ("db_admin", "POOL"): CAUSALITY_DOWNSTREAM_PREFIX +
                                  "DEADLOCK",
        })
        cfg = ReplyNoiseConfig(mislabel_prob=1.0, seed=7)
        wrapped = noisy_causality_extractor(base, cfg)
        out = wrapped(_FakeScenario("s1"), "db_admin", "POOL", "...")
        self.assertEqual(out, CAUSALITY_INDEPENDENT_ROOT)


class TestAdversarial(unittest.TestCase):
    def test_drop_root_always_hits_independent(self):
        base = _oracle_extractor({
            ("db_admin", "DEADLOCK"): CAUSALITY_INDEPENDENT_ROOT,
        })
        adv = AdversarialReplyConfig(
            target_mode=ADVERSARIAL_REPLY_MODE_DROP_ROOT, budget=1)
        wrapped = adversarial_reply_extractor(base, adv)
        out = wrapped(_FakeScenario("s1"), "db_admin",
                      "DEADLOCK", "...")
        self.assertEqual(out, CAUSALITY_UNCERTAIN)

    def test_flip_root_to_symptom(self):
        base = _oracle_extractor({
            ("db_admin", "DEADLOCK"): CAUSALITY_INDEPENDENT_ROOT,
        })
        adv = AdversarialReplyConfig(
            target_mode=ADVERSARIAL_REPLY_MODE_FLIP_ROOT_TO_SYMPTOM,
            budget=1)
        wrapped = adversarial_reply_extractor(base, adv)
        out = wrapped(_FakeScenario("s1"), "db_admin",
                      "DEADLOCK", "...")
        self.assertTrue(out.startswith(CAUSALITY_DOWNSTREAM_PREFIX))

    def test_inject_root_on_symptom(self):
        base = _oracle_extractor({
            ("db_admin", "POOL"): CAUSALITY_DOWNSTREAM_PREFIX +
                                  "DEADLOCK",
        })
        adv = AdversarialReplyConfig(
            target_mode=ADVERSARIAL_REPLY_MODE_INJECT_ROOT_ON_SYMPTOM,
            budget=1)
        wrapped = adversarial_reply_extractor(base, adv)
        out = wrapped(_FakeScenario("s1"), "db_admin", "POOL", "...")
        self.assertEqual(out, CAUSALITY_INDEPENDENT_ROOT)

    def test_budget_honored_per_scenario(self):
        base = _oracle_extractor({
            ("db_admin", "DEADLOCK"): CAUSALITY_INDEPENDENT_ROOT,
            ("network", "TLS"): CAUSALITY_INDEPENDENT_ROOT,
        })
        adv = AdversarialReplyConfig(
            target_mode=ADVERSARIAL_REPLY_MODE_DROP_ROOT, budget=1)
        wrapped = adversarial_reply_extractor(base, adv)
        s = _FakeScenario("s1")
        out1 = wrapped(s, "db_admin", "DEADLOCK", "...")
        out2 = wrapped(s, "network", "TLS", "...")
        self.assertEqual(out1, CAUSALITY_UNCERTAIN)
        # Budget exhausted on first call.
        self.assertEqual(out2, CAUSALITY_INDEPENDENT_ROOT)

    def test_budget_independent_across_scenarios(self):
        base = _oracle_extractor({
            ("db_admin", "DEADLOCK"): CAUSALITY_INDEPENDENT_ROOT,
            ("network", "TLS"): CAUSALITY_INDEPENDENT_ROOT,
        })
        adv = AdversarialReplyConfig(
            target_mode=ADVERSARIAL_REPLY_MODE_DROP_ROOT, budget=1)
        wrapped = adversarial_reply_extractor(base, adv)
        out1 = wrapped(_FakeScenario("s1"), "db_admin",
                        "DEADLOCK", "...")
        out2 = wrapped(_FakeScenario("s2"), "network", "TLS", "...")
        self.assertEqual(out1, CAUSALITY_UNCERTAIN)
        self.assertEqual(out2, CAUSALITY_UNCERTAIN)

    def test_target_roles_filter(self):
        base = _oracle_extractor({
            ("db_admin", "DEADLOCK"): CAUSALITY_INDEPENDENT_ROOT,
            ("network", "TLS"): CAUSALITY_INDEPENDENT_ROOT,
        })
        adv = AdversarialReplyConfig(
            target_mode=ADVERSARIAL_REPLY_MODE_DROP_ROOT,
            target_roles=("db_admin",),
            budget=10)
        wrapped = adversarial_reply_extractor(base, adv)
        s = _FakeScenario("s1")
        self.assertEqual(
            wrapped(s, "db_admin", "DEADLOCK", "..."),
            CAUSALITY_UNCERTAIN)
        self.assertEqual(
            wrapped(s, "network", "TLS", "..."),
            CAUSALITY_INDEPENDENT_ROOT)


class TestCorruptionReport(unittest.TestCase):
    def test_report_counts(self):
        base = _oracle_extractor({
            ("r1", "k1"): CAUSALITY_INDEPENDENT_ROOT,
            ("r2", "k2"): CAUSALITY_UNCERTAIN,
        })
        cfg = ReplyNoiseConfig(drop_prob=1.0, seed=42)
        report = ReplyCorruptionReport()
        wrapped = noisy_causality_extractor(base, cfg, report=report)
        wrapped(_FakeScenario("s1"), "r1", "k1", "...")
        wrapped(_FakeScenario("s1"), "r2", "k2", "...")
        self.assertEqual(report.n_calls, 2)
        # One INDEPENDENT_ROOT was dropped, one UNCERTAIN was
        # unchanged.
        self.assertEqual(report.n_dropped, 1)
        self.assertEqual(report.n_unchanged, 1)


if __name__ == "__main__":
    unittest.main()
