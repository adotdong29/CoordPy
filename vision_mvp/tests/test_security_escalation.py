"""Unit tests for the Phase-33 security-audit-escalation benchmark.

Mirrors the shape of ``test_phase31_incident_triage`` and
``test_compliance_review``:

  * Scenario bank determinism, severity coverage.
  * Role-subscription wiring (every claim reaches the CISO).
  * Regex extractor correctness on causal events and distractors.
  * Handoff protocol end-to-end; hash-chain validity.
  * Oracle + handoff-relevance predicates.
  * Grader behaviour.
  * Failure attribution.
  * End-to-end invariants — substrate ceiling, routing starvation,
    substrate context bounded independent of distractors, and the
    *ordinal severity* decoder (new Phase-33 shape).
"""

from __future__ import annotations

import unittest

from vision_mvp.tasks.security_escalation import (
    ALL_CLAIMS, ALL_ROLES, ALL_STRATEGIES,
    CLAIM_DATA_STAGING, CLAIM_PERSISTENCE_INSTALLED,
    CLAIM_PHISHING_DETECTED, CLAIM_MALWARE_DETECTED,
    CLAIM_REGULATED_DATA_EXPOSED, CLAIM_IOC_KNOWN_BAD_IP,
    CLAIM_BRUTE_FORCE, CLAIM_CROSS_TENANT_LEAK,
    FAILURE_LLM_ERROR, FAILURE_MISSING_HANDOFF, FAILURE_NONE,
    FAILURE_RETRIEVAL_MISS, FAILURE_SPURIOUS_CLAIM,
    FAILURE_TRUNCATION,
    MockSecurityAuditor, ROLE_CISO, ROLE_SOC_ANALYST,
    ROLE_IR_ENGINEER, ROLE_THREAT_INTEL, ROLE_DATA_STEWARD,
    SEVERITY_CRITICAL, SEVERITY_HIGH, SEVERITY_MEDIUM,
    SEVERITY_INDEX,
    STRATEGY_NAIVE, STRATEGY_ROUTING, STRATEGY_SUBSTRATE,
    STRATEGY_SUBSTRATE_WRAP,
    attribute_failure, build_auditor_prompt,
    build_role_subscriptions, build_scenario_bank,
    decode_from_handoffs, extract_claims_for_role, grade_answer,
    handoff_is_relevant, naive_event_stream, oracle_relevance,
    parse_answer, run_security_loop, run_handoff_protocol,
)


class TestScenarioBank(unittest.TestCase):

    def test_bank_deterministic(self):
        a = build_scenario_bank(seed=33, distractors_per_role=10)
        b = build_scenario_bank(seed=33, distractors_per_role=10)
        for s1, s2 in zip(a, b):
            self.assertEqual(s1.scenario_id, s2.scenario_id)
            self.assertEqual(s1.gold_severity, s2.gold_severity)
            self.assertEqual(s1.gold_classification,
                             s2.gold_classification)

    def test_bank_has_five_scenarios(self):
        a = build_scenario_bank(seed=33)
        self.assertEqual(len(a), 5)
        ids = {s.scenario_id for s in a}
        self.assertEqual(ids, {
            "phishing_exfil", "ransomware_precursor",
            "supply_chain", "insider_threat",
            "brute_force_blocked"})

    def test_severity_ordinal_cover(self):
        sevs = {s.gold_severity
                for s in build_scenario_bank(seed=33)}
        self.assertIn(SEVERITY_CRITICAL, sevs)
        self.assertIn(SEVERITY_HIGH, sevs)
        self.assertIn(SEVERITY_MEDIUM, sevs)

    def test_each_scenario_has_causal_events(self):
        for s in build_scenario_bank(seed=33):
            causal = sum(1 for _r, ds in s.per_role_events.items()
                         for d in ds if d.is_causal)
            self.assertGreater(causal, 0,
                               msg=f"{s.scenario_id} has no causal")

    def test_distractors_scale(self):
        s1 = build_scenario_bank(seed=33, distractors_per_role=3)[0]
        s2 = build_scenario_bank(seed=33, distractors_per_role=50)[0]
        n1 = sum(len(ds) for ds in s1.per_role_events.values())
        n2 = sum(len(ds) for ds in s2.per_role_events.values())
        self.assertGreater(n2, n1)


class TestRoleSubscriptions(unittest.TestCase):

    def test_every_claim_reaches_ciso(self):
        subs = build_role_subscriptions()
        for (src, kind), consumers in subs._table.items():
            self.assertIn(ROLE_CISO, consumers,
                           msg=f"{(src, kind)} skips CISO")

    def test_subscription_table_has_all_claims(self):
        subs = build_role_subscriptions()
        self.assertEqual(len(subs.all_pairs()), len(ALL_CLAIMS))


class TestExtractorRegex(unittest.TestCase):

    def test_extractor_emits_expected_kinds(self):
        bank = build_scenario_bank(seed=33)
        for s in bank:
            for role in (ROLE_SOC_ANALYST, ROLE_IR_ENGINEER,
                          ROLE_THREAT_INTEL, ROLE_DATA_STEWARD):
                evs = list(s.per_role_events.get(role, ()))
                claims = extract_claims_for_role(role, evs, s)
                # Every claim kind MUST be in ALL_CLAIMS.
                for (kind, _, _) in claims:
                    self.assertIn(kind, ALL_CLAIMS)

    def test_extractor_recalls_causal_chain(self):
        bank = build_scenario_bank(seed=33)
        for s in bank:
            seen: set[tuple[str, str]] = set()
            for role in (ROLE_SOC_ANALYST, ROLE_IR_ENGINEER,
                          ROLE_THREAT_INTEL, ROLE_DATA_STEWARD):
                evs = list(s.per_role_events.get(role, ()))
                for (kind, _, _) in extract_claims_for_role(
                        role, evs, s):
                    seen.add((role, kind))
            for (role, kind, _, _) in s.causal_chain:
                self.assertIn((role, kind), seen,
                               msg=f"{s.scenario_id}: missing "
                                    f"{(role, kind)}")

    def test_extractor_no_spurious_on_distractors(self):
        bank = build_scenario_bank(seed=33, distractors_per_role=30)
        for s in bank:
            for role in (ROLE_SOC_ANALYST, ROLE_IR_ENGINEER,
                          ROLE_THREAT_INTEL, ROLE_DATA_STEWARD):
                evs = [ev for ev in
                        s.per_role_events.get(role, ())
                        if not ev.is_causal]
                claims = extract_claims_for_role(role, evs, s)
                self.assertEqual(
                    claims, [],
                    msg=f"{s.scenario_id}/{role}: spurious "
                         f"emissions {claims}")


class TestHandoffProtocol(unittest.TestCase):

    def test_handoff_chain_ok(self):
        s = build_scenario_bank(seed=33)[0]
        router = run_handoff_protocol(s)
        self.assertTrue(router.verify())

    def test_ciso_inbox_nonempty(self):
        bank = build_scenario_bank(seed=33)
        for s in bank:
            router = run_handoff_protocol(s)
            hs = tuple(router.inboxes[ROLE_CISO].peek())
            self.assertGreater(len(hs), 0)


class TestDecoder(unittest.TestCase):

    def test_decoded_matches_gold(self):
        bank = build_scenario_bank(seed=33)
        for s in bank:
            router = run_handoff_protocol(s)
            hs = tuple(router.inboxes[ROLE_CISO].peek())
            dec = decode_from_handoffs(hs)
            self.assertEqual(dec["severity"], s.gold_severity,
                              msg=s.scenario_id)
            self.assertEqual(dec["classification"],
                              s.gold_classification,
                              msg=s.scenario_id)
            self.assertEqual(dec["containment"],
                              s.gold_containment,
                              msg=s.scenario_id)
            self.assertEqual(dec["notify"], tuple(s.gold_notify),
                              msg=s.scenario_id)

    def test_severity_ordinal_non_monotone_under_spurious(self):
        """A spurious high-severity claim escalates a MEDIUM scenario.

        This is the Phase-33 ordinal-decoder analogue of Phase-32's
        strict-decoder regime: severity is a *max* over delivered
        claim kinds, so precision failures directly flip the verdict.
        """
        s = build_scenario_bank(seed=33)[4]
        self.assertEqual(s.gold_severity, SEVERITY_MEDIUM)
        router = run_handoff_protocol(s)
        hs = list(router.inboxes[ROLE_CISO].peek())
        # Inject a spurious HIGH claim.
        from vision_mvp.core.role_handoff import TypedHandoff
        hs.append(TypedHandoff(
            handoff_id=-1, source_role=ROLE_IR_ENGINEER,
            source_agent_id=0, to_role=ROLE_CISO,
            claim_kind=CLAIM_MALWARE_DETECTED,
            payload="spurious", source_event_ids=(), round=1,
            payload_cid="", prev_chain_hash="", chain_hash=""))
        dec = decode_from_handoffs(hs)
        self.assertEqual(dec["severity"], SEVERITY_HIGH)
        self.assertNotEqual(dec["severity"], SEVERITY_MEDIUM)


class TestOracle(unittest.TestCase):

    def test_fixed_point_relevant_to_every_role(self):
        s = build_scenario_bank(seed=33)[0]
        evs = naive_event_stream(s)
        for ev in evs:
            if not ev.is_fixed_point:
                continue
            for role in ALL_ROLES:
                self.assertTrue(oracle_relevance(ev, role, s))

    def test_handoff_relevance_from_chain(self):
        s = build_scenario_bank(seed=33)[0]
        router = run_handoff_protocol(s)
        hs = tuple(router.inboxes[ROLE_CISO].peek())
        for h in hs:
            if (h.source_role, h.claim_kind) in {
                    (r, k) for (r, k, _, _) in s.causal_chain}:
                self.assertTrue(handoff_is_relevant(h, s))


class TestGrader(unittest.TestCase):

    def test_correct_answer_parses_clean(self):
        s = build_scenario_bank(seed=33)[0]
        ans = (f"SEVERITY: {s.gold_severity}\n"
               f"CLASSIFICATION: {s.gold_classification}\n"
               f"CONTAINMENT: {s.gold_containment}\n"
               f"NOTIFY: {','.join(s.gold_notify)}\n")
        g = grade_answer(s, ans)
        self.assertTrue(g["severity_correct"])
        self.assertTrue(g["classification_correct"])
        self.assertTrue(g["containment_correct"])
        self.assertTrue(g["notify_correct"])
        self.assertTrue(g["full_correct"])

    def test_wrong_severity_flags_failure(self):
        s = build_scenario_bank(seed=33)[0]
        ans = ("SEVERITY: low\n"
               f"CLASSIFICATION: {s.gold_classification}\n"
               f"CONTAINMENT: {s.gold_containment}\n"
               f"NOTIFY: {','.join(s.gold_notify)}\n")
        g = grade_answer(s, ans)
        self.assertFalse(g["severity_correct"])
        self.assertFalse(g["full_correct"])


class TestEndToEnd(unittest.TestCase):

    def test_mock_substrate_ceiling_at_k6(self):
        bank = build_scenario_bank(seed=33, distractors_per_role=6)
        aud = MockSecurityAuditor()
        rep = run_security_loop(bank, aud,
                                  strategies=(STRATEGY_SUBSTRATE,))
        p = rep.pooled()[STRATEGY_SUBSTRATE]
        self.assertEqual(p["accuracy_full"], 1.0)

    def test_substrate_token_bound_flat_across_k(self):
        tokens = []
        for k in (6, 60, 120):
            bank = build_scenario_bank(seed=33,
                                         distractors_per_role=k)
            aud = MockSecurityAuditor()
            rep = run_security_loop(
                bank, aud, strategies=(STRATEGY_SUBSTRATE,))
            p = rep.pooled()[STRATEGY_SUBSTRATE]
            tokens.append(p["mean_prompt_tokens"])
        # Bounded and essentially flat.
        self.assertLess(max(tokens) - min(tokens), 50)

    def test_routing_starves_ciso(self):
        bank = build_scenario_bank(seed=33, distractors_per_role=6)
        aud = MockSecurityAuditor()
        rep = run_security_loop(bank, aud,
                                  strategies=(STRATEGY_ROUTING,))
        p = rep.pooled()[STRATEGY_ROUTING]
        self.assertEqual(p["accuracy_full"], 0.0)

    def test_naive_truncates_at_high_k(self):
        bank = build_scenario_bank(seed=33, distractors_per_role=120)
        aud = MockSecurityAuditor()
        rep = run_security_loop(bank, aud,
                                  strategies=(STRATEGY_NAIVE,))
        p = rep.pooled()[STRATEGY_NAIVE]
        self.assertGreater(p["truncated_count"], 0)
        self.assertLess(p["accuracy_full"], 1.0)


if __name__ == "__main__":
    unittest.main()
