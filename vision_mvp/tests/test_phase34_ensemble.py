"""Unit tests for Phase 34 Part C — ensemble (union) extractor.

Covers:

  * ``UnionExtractor`` dedups on ``(kind, sorted evids)``.
  * Primary-only / secondary-only / shared counts tracked.
  * End-to-end: on the compliance mixed bank (5 canonical + 5
    narrative), regex-only ≈ 50 %, LLM-narrative ≈ 0 %, ensemble
    reaches 100 % (Theorem P34-3).
  * Pooled drop rate satisfies the C33-4 bound δ_u ≤ δ_r · δ_l.
  * Ensemble spurious rate bounded by sum (ε_u ≤ ε_r + ε_l).
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass

from vision_mvp.core.ensemble_extractor import (
    EXTRACTOR_ENSEMBLE, EXTRACTOR_LLM, EXTRACTOR_REGEX,
    UnionExtractor, union_of,
)


@dataclass
class _StubEvent:
    event_id: int
    body: str
    is_fixed_point: bool = False
    is_causal: bool = False


@dataclass
class _StubScenario:
    scenario_id: str = "sc1"


def _ex_a(role, events, scenario):
    return [("K_A", "p_a", (ev.event_id,)) for ev in events
            if ev.event_id % 2 == 0]


def _ex_b(role, events, scenario):
    return [("K_A", "p_b", (ev.event_id,)) for ev in events
            if ev.event_id % 2 == 1]


def _ex_both(role, events, scenario):
    return [("K_A", "p_both", (ev.event_id,))
            for ev in events[:2]]


class TestUnionExtractor(unittest.TestCase):

    def test_disjoint_evids_produces_disjoint_union(self):
        events = [_StubEvent(event_id=i, body=f"e{i}")
                  for i in range(4)]
        ex = UnionExtractor(_ex_a, _ex_b,
                             label_primary="a", label_secondary="b")
        out = ex("r", events, _StubScenario())
        evids = sorted(e[0] for (_k, _p, e) in out)
        self.assertEqual(evids, [0, 1, 2, 3])
        self.assertEqual(ex.stats["n_primary_only"], 2)
        self.assertEqual(ex.stats["n_secondary_only"], 2)
        self.assertEqual(ex.stats["n_shared"], 0)

    def test_shared_claims_dedup(self):
        events = [_StubEvent(event_id=i, body=f"e{i}")
                  for i in range(4)]
        ex = UnionExtractor(_ex_a, _ex_both,
                             label_primary="a", label_secondary="both")
        out = ex("r", events, _StubScenario())
        # primary produces K_A@0, K_A@2; both produces K_A@0, K_A@1.
        # Dedup by (kind, sorted evids): shared = (K_A, (0,)).
        evids = sorted(e[0] for (_k, _p, e) in out)
        self.assertEqual(evids, [0, 1, 2])
        self.assertEqual(ex.stats["n_shared"], 1)

    def test_convenience_constructor(self):
        ex = union_of(_ex_a, _ex_b)
        self.assertEqual(ex.label_primary, EXTRACTOR_REGEX)
        self.assertEqual(ex.label_secondary, EXTRACTOR_LLM)


class TestMixedBankEndToEnd(unittest.TestCase):

    def test_ensemble_beats_either_single(self):
        from vision_mvp.experiments.phase34_ensemble_extractor import (
            build_mixed_bank, llm_narrative_extractor,
        )
        from vision_mvp.tasks.compliance_review import (
            MockComplianceAuditor, STRATEGY_SUBSTRATE,
            extract_claims_for_role,
            run_compliance_loop,
        )
        bank = build_mixed_bank(seed=34, distractors_per_role=6)
        # Sanity: 10 scenarios (5 canonical + 5 narrative).
        self.assertEqual(len(bank), 10)

        regex_rep = run_compliance_loop(
            bank, MockComplianceAuditor(),
            strategies=(STRATEGY_SUBSTRATE,),
            seed=34, extractor=extract_claims_for_role)
        llm_rep = run_compliance_loop(
            bank, MockComplianceAuditor(),
            strategies=(STRATEGY_SUBSTRATE,),
            seed=34, extractor=llm_narrative_extractor)
        union = union_of(extract_claims_for_role,
                           llm_narrative_extractor,
                           label_primary="regex",
                           label_secondary="llm")
        union_rep = run_compliance_loop(
            bank, MockComplianceAuditor(),
            strategies=(STRATEGY_SUBSTRATE,),
            seed=34, extractor=union)

        regex_acc = regex_rep.pooled()[
            STRATEGY_SUBSTRATE]["accuracy_full"]
        llm_acc = llm_rep.pooled()[
            STRATEGY_SUBSTRATE]["accuracy_full"]
        union_acc = union_rep.pooled()[
            STRATEGY_SUBSTRATE]["accuracy_full"]
        # Ensemble strictly dominates either single extractor.
        self.assertGreater(union_acc, regex_acc)
        self.assertGreater(union_acc, llm_acc)
        # Ensemble reaches the mock ceiling.
        self.assertEqual(union_acc, 1.0)

    def test_c33_4_drop_bound_satisfied(self):
        from vision_mvp.core.extractor_calibration import (
            calibrate_extractor,
        )
        from vision_mvp.experiments.phase34_ensemble_extractor import (
            build_mixed_bank, llm_narrative_extractor,
        )
        from vision_mvp.tasks.compliance_review import (
            extract_claims_for_role, ROLE_LEGAL, ROLE_PRIVACY,
            ROLE_SECURITY, ROLE_FINANCE,
        )
        bank = build_mixed_bank(seed=34, distractors_per_role=6)
        roles = [ROLE_LEGAL, ROLE_SECURITY, ROLE_PRIVACY,
                 ROLE_FINANCE]

        def _role_events(s, r):
            return list(s.per_role_docs.get(r, ()))

        def _causal_ids(s, r):
            return [d.doc_id for d in s.per_role_docs.get(r, ())
                    if d.is_causal]

        def _gold(s):
            return s.causal_chain

        regex_audit = calibrate_extractor(
            extract_claims_for_role, bank,
            _role_events, _causal_ids, _gold,
            roles_to_probe=roles, extractor_label="regex")
        llm_audit = calibrate_extractor(
            llm_narrative_extractor, bank,
            _role_events, _causal_ids, _gold,
            roles_to_probe=roles, extractor_label="llm")
        union = union_of(extract_claims_for_role,
                           llm_narrative_extractor)
        union_audit = calibrate_extractor(
            union, bank,
            _role_events, _causal_ids, _gold,
            roles_to_probe=roles, extractor_label="union")
        delta_r = regex_audit.drop_rate
        delta_l = llm_audit.drop_rate
        delta_u = union_audit.drop_rate
        # C33-4 product bound.
        self.assertLessEqual(delta_u, delta_r * delta_l + 1e-6)
        # C33-4 union spurious bound.
        eps_r = regex_audit.spurious_per_event
        eps_l = llm_audit.spurious_per_event
        eps_u = union_audit.spurious_per_event
        self.assertLessEqual(eps_u, eps_r + eps_l + 1e-6)


if __name__ == "__main__":
    unittest.main()
