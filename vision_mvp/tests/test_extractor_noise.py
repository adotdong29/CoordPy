"""Unit tests for ``vision_mvp.core.extractor_noise``.

The noise module is the Phase-32 instrument that exercises the
typed-handoff substrate's graceful-degradation bound (Theorem P32-2).
Tests here cover:

  * Identity noise config is a no-op.
  * Drop-prob reduces emitted claims in expectation; drop_prob = 1.0
    empties the output.
  * Spurious-prob only adds claims on non-causal events and never
    inserts claim kinds outside the role's known-kind pool.
  * Mislabel-prob preserves emission count but may change kinds.
  * Determinism: same seed → same output.
  * Noise is per-(scenario, role) — different seeds yield different
    output; changing the scenario id shifts the rng.
  * Wired end-to-end into both incident_triage and compliance_review.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass

from vision_mvp.core.extractor_noise import (
    NoiseConfig, noisy_extractor,
    compliance_review_known_kinds, incident_triage_known_kinds,
)


# A minimal stub extractor / event type so we can test behaviour
# without pulling in a whole scenario.
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
    # Emit a causal claim for every ``is_causal`` event (kind = K_A).
    return [("K_A", ev.body, (ev.event_id,))
            for ev in events if ev.is_causal]


class TestIdentity(unittest.TestCase):

    def test_default_config_is_identity(self):
        cfg = NoiseConfig()
        self.assertTrue(cfg.is_identity())

    def test_identity_returns_base_extractor(self):
        base = _base_extractor
        out = noisy_extractor(base, {"r": ("K_A", "K_B")},
                                NoiseConfig())
        self.assertIs(out, base)

    def test_noop_roundtrip(self):
        base = _base_extractor
        noisy = noisy_extractor(base, {"r": ("K_A",)}, NoiseConfig())
        evs = [_StubEvent(event_id=i, body=f"ev{i}", is_causal=True)
               for i in range(3)]
        self.assertEqual(base("r", evs, _StubScenario()),
                         noisy("r", evs, _StubScenario()))


class TestDrop(unittest.TestCase):

    def test_drop_all(self):
        n = NoiseConfig(drop_prob=1.0, seed=1)
        noisy = noisy_extractor(_base_extractor, {"r": ("K_A",)}, n)
        evs = [_StubEvent(event_id=i, body=f"b{i}", is_causal=True)
               for i in range(10)]
        self.assertEqual(noisy("r", evs, _StubScenario()), [])

    def test_drop_none(self):
        n = NoiseConfig(drop_prob=0.0, seed=1)
        noisy = noisy_extractor(_base_extractor, {"r": ("K_A",)}, n)
        evs = [_StubEvent(event_id=i, body=f"b{i}", is_causal=True)
               for i in range(5)]
        # With drop_prob=0.0 and all other knobs 0, still identity.
        out = noisy("r", evs, _StubScenario())
        self.assertEqual(len(out), 5)


class TestSpurious(unittest.TestCase):

    def test_spurious_full_emits_on_noncausal(self):
        n = NoiseConfig(spurious_prob=1.0, seed=1)
        noisy = noisy_extractor(_base_extractor, {"r": ("K_A", "K_B")},
                                  n)
        evs = [_StubEvent(event_id=i, body=f"b{i}", is_causal=False)
               for i in range(5)]
        out = noisy("r", evs, _StubScenario())
        self.assertEqual(len(out), 5)
        for (kind, _p, _e) in out:
            self.assertIn(kind, ("K_A", "K_B"))

    def test_spurious_respects_pool(self):
        n = NoiseConfig(spurious_prob=1.0, seed=1)
        noisy = noisy_extractor(_base_extractor, {"r": ("K_A", "K_B")},
                                  n)
        evs = [_StubEvent(event_id=i, body=f"b{i}", is_causal=False)
               for i in range(30)]
        out = noisy("r", evs, _StubScenario())
        kinds = {k for (k, _p, _e) in out}
        self.assertTrue(kinds.issubset({"K_A", "K_B"}))

    def test_spurious_skips_fixed_points(self):
        n = NoiseConfig(spurious_prob=1.0, seed=1)
        noisy = noisy_extractor(_base_extractor, {"r": ("K_A",)}, n)
        evs = [_StubEvent(event_id=i, body=f"b{i}",
                          is_causal=False, is_fixed_point=True)
               for i in range(5)]
        out = noisy("r", evs, _StubScenario())
        self.assertEqual(out, [])


class TestMislabel(unittest.TestCase):

    def test_mislabel_preserves_count(self):
        n = NoiseConfig(mislabel_prob=1.0, seed=1)
        noisy = noisy_extractor(_base_extractor,
                                  {"r": ("K_A", "K_B", "K_C")}, n)
        evs = [_StubEvent(event_id=i, body=f"b{i}", is_causal=True)
               for i in range(10)]
        out = noisy("r", evs, _StubScenario())
        self.assertEqual(len(out), 10)
        # Never outputs K_A when mislabel is forced.
        self.assertTrue(all(k != "K_A" for (k, _p, _e) in out))

    def test_mislabel_no_pool_change(self):
        # If pool has only one element, mislabel has no alternatives.
        n = NoiseConfig(mislabel_prob=1.0, seed=1)
        noisy = noisy_extractor(_base_extractor, {"r": ("K_A",)}, n)
        evs = [_StubEvent(event_id=i, body=f"b{i}", is_causal=True)
               for i in range(5)]
        out = noisy("r", evs, _StubScenario())
        # Length preserved, kinds unchanged (no alternative available).
        self.assertTrue(all(k == "K_A" for (k, _p, _e) in out))


class TestDeterminism(unittest.TestCase):

    def test_same_seed_same_output(self):
        n = NoiseConfig(drop_prob=0.4, spurious_prob=0.2, seed=42)
        evs = [_StubEvent(event_id=i, body=f"b{i}",
                          is_causal=(i % 2 == 0)) for i in range(20)]
        noisy = noisy_extractor(_base_extractor, {"r": ("K_A", "K_B")},
                                  n)
        a = noisy("r", evs, _StubScenario())
        b = noisy("r", evs, _StubScenario())
        self.assertEqual(a, b)

    def test_different_seed_different_output(self):
        evs = [_StubEvent(event_id=i, body=f"b{i}",
                          is_causal=(i % 2 == 0)) for i in range(20)]
        n1 = NoiseConfig(drop_prob=0.5, seed=1)
        n2 = NoiseConfig(drop_prob=0.5, seed=2)
        a = noisy_extractor(_base_extractor, {"r": ("K_A",)}, n1)(
            "r", evs, _StubScenario())
        b = noisy_extractor(_base_extractor, {"r": ("K_A",)}, n2)(
            "r", evs, _StubScenario())
        self.assertNotEqual(a, b)


class TestEndToEndCompliance(unittest.TestCase):

    def test_zero_noise_matches_gold(self):
        from vision_mvp.tasks.compliance_review import (
            build_scenario_bank, extract_claims_for_role,
            MockComplianceAuditor, run_compliance_loop,
            STRATEGY_SUBSTRATE,
        )
        bank = build_scenario_bank(seed=32, distractors_per_role=6)
        ex = noisy_extractor(extract_claims_for_role,
                              compliance_review_known_kinds(),
                              NoiseConfig(seed=32))
        aud = MockComplianceAuditor()
        rep = run_compliance_loop(bank, aud,
                                    strategies=(STRATEGY_SUBSTRATE,),
                                    seed=32, extractor=ex)
        p = rep.pooled()[STRATEGY_SUBSTRATE]
        self.assertEqual(p["accuracy_full"], 1.0)

    def test_high_noise_drops_accuracy(self):
        from vision_mvp.tasks.compliance_review import (
            build_scenario_bank, extract_claims_for_role,
            MockComplianceAuditor, run_compliance_loop,
            STRATEGY_SUBSTRATE,
        )
        bank = build_scenario_bank(seed=32, distractors_per_role=6)
        ex = noisy_extractor(extract_claims_for_role,
                              compliance_review_known_kinds(),
                              NoiseConfig(drop_prob=1.0, seed=32))
        aud = MockComplianceAuditor()
        rep = run_compliance_loop(bank, aud,
                                    strategies=(STRATEGY_SUBSTRATE,),
                                    seed=32, extractor=ex)
        p = rep.pooled()[STRATEGY_SUBSTRATE]
        # With drop_prob=1.0 every causal claim is dropped; the
        # substrate produces "approved" as default — hence accuracy
        # is 0 on blocked/conditional scenarios (the entire bank).
        self.assertLess(p["accuracy_full"], 0.25)

    def test_every_row_has_non_identity_marker(self):
        # Sanity: if the noise wrapper is used and the config is
        # non-identity, the wrapper is the callable returned (not the
        # base).
        from vision_mvp.tasks.compliance_review import (
            extract_claims_for_role,
        )
        out = noisy_extractor(extract_claims_for_role,
                                compliance_review_known_kinds(),
                                NoiseConfig(drop_prob=0.1, seed=0))
        self.assertIsNot(out, extract_claims_for_role)


class TestEndToEndIncident(unittest.TestCase):

    def test_identity_noise_preserves_accuracy(self):
        from vision_mvp.tasks.incident_triage import (
            build_scenario_bank, extract_claims_for_role,
            MockIncidentAuditor, run_incident_loop,
            STRATEGY_SUBSTRATE,
        )
        bank = build_scenario_bank(seed=31, distractors_per_role=6)
        ex = noisy_extractor(extract_claims_for_role,
                              incident_triage_known_kinds(),
                              NoiseConfig(seed=31))
        aud = MockIncidentAuditor()
        rep = run_incident_loop(bank, aud,
                                  strategies=(STRATEGY_SUBSTRATE,),
                                  seed=31, extractor=ex)
        self.assertEqual(
            rep.pooled()[STRATEGY_SUBSTRATE]["accuracy_full"], 1.0)

    def test_drop_all_collapses_accuracy(self):
        from vision_mvp.tasks.incident_triage import (
            build_scenario_bank, extract_claims_for_role,
            MockIncidentAuditor, run_incident_loop,
            STRATEGY_SUBSTRATE,
        )
        bank = build_scenario_bank(seed=31, distractors_per_role=6)
        ex = noisy_extractor(extract_claims_for_role,
                              incident_triage_known_kinds(),
                              NoiseConfig(drop_prob=1.0, seed=31))
        aud = MockIncidentAuditor()
        rep = run_incident_loop(bank, aud,
                                  strategies=(STRATEGY_SUBSTRATE,),
                                  seed=31, extractor=ex)
        self.assertEqual(
            rep.pooled()[STRATEGY_SUBSTRATE]["accuracy_full"], 0.0)


if __name__ == "__main__":
    unittest.main()
