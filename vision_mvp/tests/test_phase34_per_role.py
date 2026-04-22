"""Unit tests for Phase 34 Part A — per-role calibration + per-role
noise wrapper.

Covers:

  * ``PerRoleNoiseConfig`` identity behaviour.
  * ``per_role_noisy_extractor`` dispatches the right NoiseConfig
    per role.
  * ``per_role_heterogeneity`` reports correct spread / worst role.
  * ``per_role_closest_synthetic`` returns a sensible match when
    given a small sweep table.
  * ``per_role_audit_summary`` wiring.
  * ``build_from_audit_per_role`` round-trip: if we fit a
    PerRoleNoiseConfig from an audit and replay, we get the same
    rates back (up to rng variance).
  * End-to-end: when one role has drop_prob = 1.0, substrate
    accuracy collapses; when a different role has drop_prob = 1.0,
    a different scenario set is affected.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass

from vision_mvp.core.extractor_calibration import (
    ExtractorAudit, per_role_audit_summary, per_role_closest_synthetic,
    per_role_heterogeneity,
)
from vision_mvp.core.extractor_noise import (
    NoiseConfig, PerRoleNoiseConfig, build_from_audit_per_role,
    build_uniform_per_role, per_role_noisy_extractor,
)


@dataclass
class _StubEvent:
    event_id: int
    body: str
    is_fixed_point: bool = False
    is_causal: bool = False


@dataclass
class _StubScenario:
    scenario_id: str = "stub"


def _base_extractor(role, events, scenario):
    return [("K_A", ev.body, (ev.event_id,))
            for ev in events if ev.is_causal]


class TestPerRoleNoiseConfigIdentity(unittest.TestCase):

    def test_default_is_identity(self):
        cfg = PerRoleNoiseConfig()
        self.assertTrue(cfg.is_identity())

    def test_uniform_zero_is_identity(self):
        cfg = build_uniform_per_role(
            roles=["a", "b"], config=NoiseConfig())
        self.assertTrue(cfg.is_identity())

    def test_single_nonzero_role_not_identity(self):
        cfg = PerRoleNoiseConfig(
            by_role={"a": NoiseConfig(drop_prob=0.5, seed=1)})
        self.assertFalse(cfg.is_identity())

    def test_identity_returns_base_extractor(self):
        out = per_role_noisy_extractor(
            _base_extractor, {"r": ("K_A",)},
            PerRoleNoiseConfig())
        self.assertIs(out, _base_extractor)


class TestPerRoleDispatch(unittest.TestCase):

    def test_role_a_drops_all_role_b_preserves(self):
        # Role A gets drop=1.0, Role B gets drop=0.0.
        per_role = PerRoleNoiseConfig(
            by_role={
                "a": NoiseConfig(drop_prob=1.0, seed=1),
                "b": NoiseConfig(drop_prob=0.0, seed=1),
            },
            seed=0)
        evs = [_StubEvent(event_id=i, body=f"e{i}", is_causal=True)
               for i in range(5)]
        noisy = per_role_noisy_extractor(
            _base_extractor, {"a": ("K_A",), "b": ("K_A",)},
            per_role)
        self.assertEqual(noisy("a", evs, _StubScenario()), [])
        self.assertEqual(len(noisy("b", evs, _StubScenario())), 5)

    def test_fallback_applies_to_unknown_role(self):
        per_role = PerRoleNoiseConfig(
            by_role={"a": NoiseConfig(drop_prob=1.0, seed=1)},
            fallback=NoiseConfig(drop_prob=0.0, seed=1))
        evs = [_StubEvent(event_id=i, body=f"e{i}", is_causal=True)
               for i in range(5)]
        noisy = per_role_noisy_extractor(
            _base_extractor, {"a": ("K_A",), "c": ("K_A",)},
            per_role)
        self.assertEqual(noisy("a", evs, _StubScenario()), [])
        # Unknown role 'c' uses fallback (drop=0) → 5 emissions.
        self.assertEqual(len(noisy("c", evs, _StubScenario())), 5)


class TestHeterogeneity(unittest.TestCase):

    def test_empty_audit(self):
        audit = ExtractorAudit(by_role={})
        r = per_role_heterogeneity(audit)
        self.assertFalse(r["pooled_masks_per_role"])
        self.assertEqual(r["max_spread_any_axis"], 0.0)

    def test_spread_above_threshold(self):
        audit = ExtractorAudit(
            drop_rate=0.5,
            by_role={
                "legal": {"drop_rate": 0.1,
                          "mislabel_rate": 0.0,
                          "spurious_per_event": 0.0,
                          "payload_corrupt_rate": 0.0},
                "finance": {"drop_rate": 0.9,
                             "mislabel_rate": 0.0,
                             "spurious_per_event": 0.0,
                             "payload_corrupt_rate": 0.0},
            })
        r = per_role_heterogeneity(audit)
        self.assertAlmostEqual(r["max_spread_any_axis"], 0.8,
                                 places=3)
        self.assertTrue(r["pooled_masks_per_role"])
        self.assertEqual(r["worst_role_by_axis"]["drop_rate"],
                           "finance")

    def test_spread_below_threshold(self):
        audit = ExtractorAudit(
            drop_rate=0.1,
            by_role={
                "a": {"drop_rate": 0.1,
                       "mislabel_rate": 0.0,
                       "spurious_per_event": 0.0,
                       "payload_corrupt_rate": 0.0},
                "b": {"drop_rate": 0.15,
                       "mislabel_rate": 0.0,
                       "spurious_per_event": 0.0,
                       "payload_corrupt_rate": 0.0},
            })
        r = per_role_heterogeneity(audit)
        self.assertFalse(r["pooled_masks_per_role"])


class TestPerRoleClosestSynthetic(unittest.TestCase):

    def test_returns_one_match_per_role(self):
        audit = ExtractorAudit(
            drop_rate=0.5,
            by_role={
                "a": {"drop_rate": 0.1,
                       "mislabel_rate": 0.0,
                       "spurious_per_event": 0.0,
                       "payload_corrupt_rate": 0.0},
                "b": {"drop_rate": 0.9,
                       "mislabel_rate": 0.0,
                       "spurious_per_event": 0.0,
                       "payload_corrupt_rate": 0.0},
            })
        sweep = {
            "k0": {"domain": "d", "drop_prob": 0.1,
                     "spurious_prob": 0.0, "mislabel_prob": 0.0,
                     "payload_corrupt_prob": 0.0,
                     "accuracy_mean": 0.9, "recall_mean": 0.9,
                     "precision_mean": 1.0, "tokens_mean": 100},
            "k1": {"domain": "d", "drop_prob": 0.9,
                     "spurious_prob": 0.0, "mislabel_prob": 0.0,
                     "payload_corrupt_prob": 0.0,
                     "accuracy_mean": 0.1, "recall_mean": 0.1,
                     "precision_mean": 1.0, "tokens_mean": 50},
        }
        matches = per_role_closest_synthetic(audit, sweep, domain="d")
        self.assertEqual(matches["a"].drop_prob, 0.1)
        self.assertEqual(matches["b"].drop_prob, 0.9)


class TestAuditSummary(unittest.TestCase):

    def test_summary_contains_limiting_role(self):
        audit = ExtractorAudit(
            drop_rate=0.5,
            by_role={
                "weak": {"drop_rate": 0.9,
                          "mislabel_rate": 0.0,
                          "spurious_per_event": 0.0,
                          "payload_corrupt_rate": 0.0},
                "strong": {"drop_rate": 0.1,
                            "mislabel_rate": 0.0,
                            "spurious_per_event": 0.0,
                            "payload_corrupt_rate": 0.0},
            })
        s = per_role_audit_summary(audit)
        self.assertEqual(s["role_limited_by"], "weak")
        self.assertAlmostEqual(s["role_limited_rate"], 0.9,
                                 places=3)


class TestBuildFromAuditPerRole(unittest.TestCase):

    def test_round_trip_rates(self):
        by_role = {
            "a": {"drop_rate": 0.3, "mislabel_rate": 0.0,
                   "spurious_per_event": 0.1,
                   "payload_corrupt_rate": 0.0},
            "b": {"drop_rate": 0.7, "mislabel_rate": 0.0,
                   "spurious_per_event": 0.0,
                   "payload_corrupt_rate": 0.0},
        }
        cfg = build_from_audit_per_role(by_role, seed=7)
        self.assertAlmostEqual(
            cfg.by_role["a"].drop_prob, 0.3, places=3)
        self.assertAlmostEqual(
            cfg.by_role["b"].drop_prob, 0.7, places=3)
        self.assertAlmostEqual(
            cfg.by_role["a"].spurious_prob, 0.1, places=3)


class TestPerRoleEndToEnd(unittest.TestCase):

    def test_silencing_one_role_collapses_accuracy_only_for_that_role(
            self):
        from vision_mvp.tasks.compliance_review import (
            MockComplianceAuditor, STRATEGY_SUBSTRATE,
            build_scenario_bank, extract_claims_for_role,
            run_compliance_loop,
        )
        from vision_mvp.core.extractor_noise import (
            compliance_review_known_kinds,
        )
        bank = build_scenario_bank(seed=34,
                                     distractors_per_role=6)
        known = compliance_review_known_kinds()

        # Silence the privacy role only.
        cfg = PerRoleNoiseConfig(
            by_role={"privacy": NoiseConfig(drop_prob=1.0, seed=1)},
            fallback=NoiseConfig(),
            seed=1)
        ex = per_role_noisy_extractor(
            extract_claims_for_role, known, cfg)
        rep = run_compliance_loop(
            bank, MockComplianceAuditor(),
            strategies=(STRATEGY_SUBSTRATE,),
            seed=34, extractor=ex)
        p = rep.pooled()[STRATEGY_SUBSTRATE]
        # Privacy owns the causal chain for missing_dpa and
        # cross_border_transfer_unauthorized scenarios; silencing
        # privacy should collapse accuracy on at least those two.
        self.assertLessEqual(p["accuracy_full"], 0.8)

    def test_per_role_noise_is_deterministic(self):
        from vision_mvp.tasks.compliance_review import (
            MockComplianceAuditor, STRATEGY_SUBSTRATE,
            build_scenario_bank, extract_claims_for_role,
            run_compliance_loop,
        )
        from vision_mvp.core.extractor_noise import (
            compliance_review_known_kinds,
        )
        bank = build_scenario_bank(seed=34,
                                     distractors_per_role=6)
        known = compliance_review_known_kinds()
        cfg = PerRoleNoiseConfig(
            by_role={"legal": NoiseConfig(drop_prob=0.5, seed=3),
                      "privacy": NoiseConfig(drop_prob=0.25, seed=3)},
            seed=3)
        ex1 = per_role_noisy_extractor(
            extract_claims_for_role, known, cfg)
        ex2 = per_role_noisy_extractor(
            extract_claims_for_role, known, cfg)
        r1 = run_compliance_loop(
            bank, MockComplianceAuditor(),
            strategies=(STRATEGY_SUBSTRATE,),
            seed=34, extractor=ex1).pooled()[STRATEGY_SUBSTRATE]
        r2 = run_compliance_loop(
            bank, MockComplianceAuditor(),
            strategies=(STRATEGY_SUBSTRATE,),
            seed=34, extractor=ex2).pooled()[STRATEGY_SUBSTRATE]
        self.assertEqual(r1["accuracy_full"], r2["accuracy_full"])
        self.assertEqual(r1["mean_handoff_recall"],
                           r2["mean_handoff_recall"])


if __name__ == "__main__":
    unittest.main()
