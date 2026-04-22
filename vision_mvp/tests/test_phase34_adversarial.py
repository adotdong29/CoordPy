"""Unit tests for Phase 34 Part B — adversarial extractor noise.

Covers:

  * ``AdversarialConfig`` construction and rejection of unknown
    target_mode.
  * ``ADVERSARIAL_MODE_LOAD_BEARING_DROP`` drops emissions whose
    ``(role, kind)`` is in the scenario's causal chain, up to budget.
  * Priority-order is respected: higher-priority kinds dropped first.
  * ``ADVERSARIAL_MODE_ROLE_SILENCING`` drops every emission from
    targeted roles.
  * ``ADVERSARIAL_MODE_SEVERITY_ESCALATION`` injects a spurious claim
    on the first distractor event.
  * ``ADVERSARIAL_MODE_COMBINED`` applies all three passes.
  * End-to-end against compliance: load-bearing drop at budget 1
    collapses accuracy more than matched i.i.d. (Theorem P34-2).
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass

from vision_mvp.core.extractor_noise import (
    ADVERSARIAL_MODE_COMBINED, ADVERSARIAL_MODE_LOAD_BEARING_DROP,
    ADVERSARIAL_MODE_ROLE_SILENCING,
    ADVERSARIAL_MODE_SEVERITY_ESCALATION,
    AdversarialConfig, NoiseConfig,
    adversarial_extractor, noisy_extractor,
)


@dataclass
class _StubEvent:
    event_id: int
    body: str
    is_fixed_point: bool = False
    is_causal: bool = False


@dataclass
class _StubScenario:
    scenario_id: str
    causal_chain: tuple = ()


def _base(role, events, scenario):
    # Emit K_A on every causal event, K_B on every non-causal
    # event whose body starts with 'b'.
    out = []
    for ev in events:
        if ev.is_causal:
            out.append(("K_A", ev.body, (ev.event_id,)))
        elif ev.body.startswith("b"):
            out.append(("K_B", ev.body, (ev.event_id,)))
    return out


class TestAdversarialConfig(unittest.TestCase):

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            adversarial_extractor(
                _base, {}, AdversarialConfig(target_mode="unknown"))

    def test_default_is_load_bearing_drop(self):
        cfg = AdversarialConfig()
        self.assertEqual(cfg.target_mode,
                           ADVERSARIAL_MODE_LOAD_BEARING_DROP)


class TestLoadBearingDrop(unittest.TestCase):

    def test_drops_load_bearing_claim_only(self):
        # scenario has causal chain with (r, K_A) load-bearing.
        sc = _StubScenario(
            scenario_id="sc1",
            causal_chain=(("r", "K_A", "body1", (0,)),))
        events = [
            _StubEvent(event_id=0, body="body1", is_causal=True),
            _StubEvent(event_id=1, body="b_distractor",
                       is_causal=False),
        ]
        cfg = AdversarialConfig(
            target_mode=ADVERSARIAL_MODE_LOAD_BEARING_DROP,
            drop_budget=1, seed=0)
        noisy = adversarial_extractor(_base, {"r": ("K_A", "K_B")},
                                       cfg)
        out = noisy("r", events, sc)
        kinds = {k for (k, _p, _e) in out}
        # K_A load-bearing emission was dropped, K_B (non-causal,
        # non-load-bearing) preserved.
        self.assertNotIn("K_A", kinds)
        self.assertIn("K_B", kinds)

    def test_budget_of_zero_is_pass_through(self):
        sc = _StubScenario(
            scenario_id="sc1",
            causal_chain=(("r", "K_A", "body1", (0,)),))
        events = [
            _StubEvent(event_id=0, body="body1", is_causal=True),
        ]
        cfg = AdversarialConfig(
            target_mode=ADVERSARIAL_MODE_LOAD_BEARING_DROP,
            drop_budget=0, seed=0)
        noisy = adversarial_extractor(
            _base, {"r": ("K_A",)}, cfg)
        out = noisy("r", events, sc)
        self.assertEqual(len(out), 1)

    def test_priority_order_prefers_high_priority(self):
        # Two load-bearing claims; budget=1. With priority K_A first,
        # K_A should be dropped, K_B preserved.
        sc = _StubScenario(
            scenario_id="sc1",
            causal_chain=(("r", "K_A", "body1", (0,)),
                           ("r", "K_B", "body2", (1,))))

        def _two_kind_base(role, events, scenario):
            out = []
            for ev in events:
                if ev.event_id == 0:
                    out.append(("K_A", ev.body, (ev.event_id,)))
                elif ev.event_id == 1:
                    out.append(("K_B", ev.body, (ev.event_id,)))
            return out

        events = [
            _StubEvent(event_id=0, body="body1", is_causal=True),
            _StubEvent(event_id=1, body="body2", is_causal=True),
        ]
        cfg = AdversarialConfig(
            target_mode=ADVERSARIAL_MODE_LOAD_BEARING_DROP,
            drop_budget=1, priority_order=("K_A", "K_B"), seed=0)
        noisy = adversarial_extractor(
            _two_kind_base, {"r": ("K_A", "K_B")}, cfg)
        out = noisy("r", events, sc)
        kinds = {k for (k, _p, _e) in out}
        self.assertNotIn("K_A", kinds)
        self.assertIn("K_B", kinds)

    def test_budget_negative_drops_all(self):
        sc = _StubScenario(
            scenario_id="sc1",
            causal_chain=(("r", "K_A", "b1", (0,)),
                           ("r", "K_A", "b2", (1,))))
        events = [
            _StubEvent(event_id=0, body="b1", is_causal=True),
            _StubEvent(event_id=1, body="b2", is_causal=True),
        ]
        cfg = AdversarialConfig(
            target_mode=ADVERSARIAL_MODE_LOAD_BEARING_DROP,
            drop_budget=-1, seed=0)
        noisy = adversarial_extractor(
            _base, {"r": ("K_A",)}, cfg)
        out = noisy("r", events, sc)
        kinds = {k for (k, _p, _e) in out}
        self.assertNotIn("K_A", kinds)


class TestRoleSilencing(unittest.TestCase):

    def test_silences_target_role(self):
        events = [
            _StubEvent(event_id=0, body="body1", is_causal=True),
        ]
        sc = _StubScenario(
            scenario_id="sc1",
            causal_chain=(("r1", "K_A", "body1", (0,)),))
        cfg = AdversarialConfig(
            target_mode=ADVERSARIAL_MODE_ROLE_SILENCING,
            target_roles=("r1",), seed=0)
        noisy = adversarial_extractor(
            _base, {"r1": ("K_A",), "r2": ("K_A",)}, cfg)
        self.assertEqual(noisy("r1", events, sc), [])
        self.assertEqual(len(noisy("r2", events, sc)), 1)

    def test_non_target_role_passes_through(self):
        events = [
            _StubEvent(event_id=0, body="body1", is_causal=True),
        ]
        sc = _StubScenario(
            scenario_id="sc1",
            causal_chain=(("r1", "K_A", "body1", (0,)),))
        cfg = AdversarialConfig(
            target_mode=ADVERSARIAL_MODE_ROLE_SILENCING,
            target_roles=("r2",), seed=0)
        noisy = adversarial_extractor(
            _base, {"r1": ("K_A",)}, cfg)
        self.assertEqual(len(noisy("r1", events, sc)), 1)


class TestSeverityEscalation(unittest.TestCase):

    def test_injects_on_first_distractor(self):
        events = [
            _StubEvent(event_id=0, body="causal_body", is_causal=True),
            _StubEvent(event_id=1, body="distractor",
                       is_causal=False),
            _StubEvent(event_id=2, body="distractor2",
                       is_causal=False),
        ]
        sc = _StubScenario(
            scenario_id="sc1",
            causal_chain=(("r", "K_A", "causal_body", (0,)),))
        cfg = AdversarialConfig(
            target_mode=ADVERSARIAL_MODE_SEVERITY_ESCALATION,
            escalation_kinds=("K_HIGH",), seed=0)
        noisy = adversarial_extractor(
            _base, {"r": ("K_A", "K_HIGH")}, cfg)
        out = noisy("r", events, sc)
        # First base emission (K_A on event_id=0) preserved +
        # injected K_HIGH on event_id=1.
        kinds = [k for (k, _p, _e) in out]
        self.assertIn("K_A", kinds)
        self.assertIn("K_HIGH", kinds)

    def test_no_injection_if_escalation_kind_not_in_pool(self):
        events = [
            _StubEvent(event_id=0, body="body", is_causal=True),
            _StubEvent(event_id=1, body="distractor",
                       is_causal=False),
        ]
        sc = _StubScenario(
            scenario_id="sc1",
            causal_chain=(("r", "K_A", "body", (0,)),))
        cfg = AdversarialConfig(
            target_mode=ADVERSARIAL_MODE_SEVERITY_ESCALATION,
            escalation_kinds=("K_NOT_IN_POOL",), seed=0)
        noisy = adversarial_extractor(
            _base, {"r": ("K_A",)}, cfg)
        out = noisy("r", events, sc)
        # Falls back to first kind in pool = K_A.
        kinds = [k for (k, _p, _e) in out]
        self.assertEqual(kinds.count("K_A"), 2)


class TestCombined(unittest.TestCase):

    def test_combined_applies_all_three(self):
        # Role "r1" targeted for silencing; scenario has load-bearing
        # (r1, K_A) emission. Combined mode silences r1 entirely
        # (the load-bearing-drop pass also runs but on the now-empty
        # baseline), so output is only the severity escalation.
        events = [
            _StubEvent(event_id=0, body="body", is_causal=True),
            _StubEvent(event_id=1, body="distractor",
                       is_causal=False),
        ]
        sc = _StubScenario(
            scenario_id="sc1",
            causal_chain=(("r1", "K_A", "body", (0,)),))
        cfg = AdversarialConfig(
            target_mode=ADVERSARIAL_MODE_COMBINED,
            target_roles=("r1",), drop_budget=1,
            escalation_kinds=("K_HIGH",), seed=0)
        noisy = adversarial_extractor(
            _base, {"r1": ("K_A", "K_HIGH")}, cfg)
        out = noisy("r1", events, sc)
        kinds = [k for (k, _p, _e) in out]
        # K_A silenced, K_HIGH injected on distractor.
        self.assertNotIn("K_A", kinds)
        self.assertIn("K_HIGH", kinds)


class TestEndToEndCompliance(unittest.TestCase):

    def test_load_bearing_drop_beats_matched_iid(self):
        # Phase 34 Part B headline: at budget = 1 (one load-bearing
        # claim per scenario), the adversary's substrate accuracy is
        # strictly lower than matched-nominal-budget i.i.d.
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
        # Mean causal chain length on compliance bank is 2.
        mean_R = 2.0
        budget = 1

        adv_ex = adversarial_extractor(
            extract_claims_for_role, known,
            AdversarialConfig(
                target_mode=ADVERSARIAL_MODE_LOAD_BEARING_DROP,
                drop_budget=budget, seed=34))
        iid_ex = noisy_extractor(
            extract_claims_for_role, known,
            NoiseConfig(drop_prob=budget / mean_R, seed=34))
        adv_rep = run_compliance_loop(
            bank, MockComplianceAuditor(),
            strategies=(STRATEGY_SUBSTRATE,),
            seed=34, extractor=adv_ex)
        iid_rep = run_compliance_loop(
            bank, MockComplianceAuditor(),
            strategies=(STRATEGY_SUBSTRATE,),
            seed=34, extractor=iid_ex)
        adv_acc = adv_rep.pooled()[
            STRATEGY_SUBSTRATE]["accuracy_full"]
        iid_acc = iid_rep.pooled()[
            STRATEGY_SUBSTRATE]["accuracy_full"]
        # Adversarial should always be ≤ matched i.i.d. (the theorem
        # predicts strict inequality for budget < R*; at budget = 1
        # with R* = 2 the adversary destroys one of two required
        # claims every scenario — accuracy = 0).
        self.assertEqual(adv_acc, 0.0)
        self.assertLessEqual(adv_acc, iid_acc)


if __name__ == "__main__":
    unittest.main()
