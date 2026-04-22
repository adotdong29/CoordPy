"""Unit tests for the Phase-32 vendor-onboarding compliance-review
benchmark.

Mirrors the shape of ``test_phase31_incident_triage`` so the same
invariants are tested on the second non-code domain:

  * Scenario bank determinism and coverage.
  * Role-subscription wiring (every claim reaches the compliance
    officer).
  * Claim-extractor correctness on causal docs and distractors.
  * Handoff protocol end-to-end; hash-chain validity.
  * Oracle + handoff-relevance predicates.
  * Grader behaviour on well-formed / malformed answers.
  * Failure attribution — verdict / flags / remediation / spurious.
  * End-to-end invariants — substrate ceiling, routing starvation,
    substrate context bounded independent of distractors.
"""
from __future__ import annotations

import unittest

from vision_mvp.tasks.compliance_review import (
    ALL_CLAIMS, ALL_ROLES, ALL_STRATEGIES,
    CLAIM_DPA_MISSING, CLAIM_ENCRYPTION_AT_REST_MISSING,
    CLAIM_LIABILITY_CAP_MISSING, CLAIM_RETENTION_UNCAPPED,
    CLAIM_BUDGET_THRESHOLD_BREACH, CLAIM_AUTO_RENEWAL_UNFAVOURABLE,
    CLAIM_PAYMENT_TERMS_AGGRESSIVE,
    FAILURE_LLM_ERROR, FAILURE_MISSING_HANDOFF, FAILURE_NONE,
    FAILURE_RETRIEVAL_MISS, FAILURE_TRUNCATION, FAILURE_SPURIOUS_CLAIM,
    MockComplianceAuditor, ROLE_COMPLIANCE, ROLE_LEGAL, ROLE_PRIVACY,
    ROLE_SECURITY, ROLE_FINANCE,
    STRATEGY_NAIVE, STRATEGY_ROUTING, STRATEGY_SUBSTRATE,
    STRATEGY_SUBSTRATE_WRAP,
    VERDICT_APPROVED, VERDICT_BLOCKED, VERDICT_CONDITIONAL,
    attribute_failure, build_auditor_prompt,
    build_role_subscriptions, build_scenario_bank,
    decode_from_handoffs, extract_claims_for_role, grade_answer,
    handoff_is_relevant, naive_doc_stream, oracle_relevance,
    parse_answer, run_compliance_loop, run_handoff_protocol,
)


# -----------------------------------------------------------------------------
# Scenario bank
# -----------------------------------------------------------------------------


class TestScenarioBank(unittest.TestCase):

    def test_bank_deterministic(self):
        a = build_scenario_bank(seed=32, distractors_per_role=10)
        b = build_scenario_bank(seed=32, distractors_per_role=10)
        for s1, s2 in zip(a, b):
            self.assertEqual(s1.scenario_id, s2.scenario_id)
            self.assertEqual(s1.gold_verdict, s2.gold_verdict)
            self.assertEqual(s1.gold_flags, s2.gold_flags)
            self.assertEqual(s1.gold_remediation, s2.gold_remediation)

    def test_bank_has_five_scenarios(self):
        a = build_scenario_bank(seed=32)
        self.assertEqual(len(a), 5)
        ids = {s.scenario_id for s in a}
        self.assertEqual(ids, {
            "missing_dpa", "uncapped_liability",
            "weak_encryption", "cross_border_transfer_unauthorized",
            "budget_threshold_breach"})

    def test_each_scenario_has_causal_docs(self):
        for s in build_scenario_bank(seed=32):
            causal = sum(1 for _r, ds in s.per_role_docs.items()
                         for d in ds if d.is_causal)
            self.assertGreater(causal, 0,
                               msg=f"{s.scenario_id} has no causal docs")

    def test_distractors_scale(self):
        small = build_scenario_bank(seed=32, distractors_per_role=3)
        big = build_scenario_bank(seed=32, distractors_per_role=30)
        sm = sum(len(ds) for ds in small[0].per_role_docs.values())
        bg = sum(len(ds) for ds in big[0].per_role_docs.values())
        self.assertGreater(bg, sm)

    def test_verdicts_cover_all_three(self):
        verdicts = {s.gold_verdict
                    for s in build_scenario_bank(seed=32)}
        self.assertIn(VERDICT_BLOCKED, verdicts)
        self.assertIn(VERDICT_CONDITIONAL, verdicts)


# -----------------------------------------------------------------------------
# Role subscriptions
# -----------------------------------------------------------------------------


class TestRoleSubscriptions(unittest.TestCase):

    def test_every_claim_reaches_compliance(self):
        subs = build_role_subscriptions()
        for (src, kind), consumers in subs._table.items():
            self.assertIn(ROLE_COMPLIANCE, consumers,
                           msg=f"{(src, kind)} skips compliance officer")

    def test_every_claim_kind_covered(self):
        subs = build_role_subscriptions()
        covered = subs.all_claim_kinds()
        for k in ALL_CLAIMS:
            self.assertIn(k, covered,
                           msg=f"{k} has no subscription entry")


# -----------------------------------------------------------------------------
# Claim extractors
# -----------------------------------------------------------------------------


class TestExtractors(unittest.TestCase):

    def test_legal_extracts_liability_missing(self):
        bank = build_scenario_bank(seed=32, distractors_per_role=0)
        s = [x for x in bank if x.scenario_id == "uncapped_liability"][0]
        claims = extract_claims_for_role(ROLE_LEGAL,
                                          s.per_role_docs[ROLE_LEGAL], s)
        kinds = {k for (k, _p, _e) in claims}
        self.assertIn(CLAIM_LIABILITY_CAP_MISSING, kinds)

    def test_privacy_detects_dpa_missing(self):
        bank = build_scenario_bank(seed=32, distractors_per_role=0)
        s = [x for x in bank if x.scenario_id == "missing_dpa"][0]
        claims = extract_claims_for_role(ROLE_PRIVACY,
                                          s.per_role_docs[ROLE_PRIVACY], s)
        kinds = {k for (k, _p, _e) in claims}
        self.assertIn(CLAIM_DPA_MISSING, kinds)
        self.assertIn(CLAIM_RETENTION_UNCAPPED, kinds)

    def test_security_detects_encryption_missing(self):
        bank = build_scenario_bank(seed=32, distractors_per_role=0)
        s = [x for x in bank
             if x.scenario_id == "weak_encryption"][0]
        claims = extract_claims_for_role(ROLE_SECURITY,
                                          s.per_role_docs[ROLE_SECURITY],
                                          s)
        kinds = {k for (k, _p, _e) in claims}
        self.assertIn(CLAIM_ENCRYPTION_AT_REST_MISSING, kinds)

    def test_finance_detects_budget_breach_and_payment(self):
        bank = build_scenario_bank(seed=32, distractors_per_role=0)
        s = [x for x in bank
             if x.scenario_id == "budget_threshold_breach"][0]
        claims = extract_claims_for_role(ROLE_FINANCE,
                                          s.per_role_docs[ROLE_FINANCE],
                                          s)
        kinds = {k for (k, _p, _e) in claims}
        self.assertIn(CLAIM_BUDGET_THRESHOLD_BREACH, kinds)
        self.assertIn(CLAIM_PAYMENT_TERMS_AGGRESSIVE, kinds)

    def test_distractors_do_not_produce_claims(self):
        bank = build_scenario_bank(seed=32, distractors_per_role=40)
        for s in bank:
            for role, docs in s.per_role_docs.items():
                if role == ROLE_COMPLIANCE:
                    continue
                distractors = [d for d in docs if not d.is_causal]
                claims = extract_claims_for_role(role, distractors, s)
                self.assertEqual(
                    claims, [],
                    msg=(f"{s.scenario_id}/{role}: distractors produced "
                          f"{claims}"))


# -----------------------------------------------------------------------------
# Oracle + handoff relevance
# -----------------------------------------------------------------------------


class TestOracle(unittest.TestCase):

    def test_fixed_point_always_relevant(self):
        bank = build_scenario_bank(seed=32)
        stream = naive_doc_stream(bank[0])
        fixed = [d for d in stream if d.is_fixed_point]
        for d in fixed:
            for role in ALL_ROLES:
                self.assertTrue(oracle_relevance(d, role, bank[0]))

    def test_distractor_irrelevant_to_own_role(self):
        bank = build_scenario_bank(seed=32, distractors_per_role=3)
        s = bank[0]
        for role in (ROLE_LEGAL, ROLE_SECURITY, ROLE_PRIVACY,
                      ROLE_FINANCE):
            for d in s.per_role_docs[role]:
                if not d.is_causal:
                    self.assertFalse(
                        oracle_relevance(d, role, s),
                        msg=f"{role} distractor {d.doc_id} relevant?")

    def test_handoff_relevance_matches_chain(self):
        bank = build_scenario_bank(seed=32)
        s = bank[0]
        router = run_handoff_protocol(s)
        for h in router.inboxes[ROLE_COMPLIANCE].peek():
            self.assertTrue(handoff_is_relevant(h, s))


# -----------------------------------------------------------------------------
# Handoff protocol
# -----------------------------------------------------------------------------


class TestHandoffProtocol(unittest.TestCase):

    def test_chain_valid(self):
        bank = build_scenario_bank(seed=32)
        for s in bank:
            router = run_handoff_protocol(s)
            self.assertTrue(router.verify())
            self.assertGreater(router.log_length(), 0)

    def test_compliance_inbox_has_all_required_claims(self):
        bank = build_scenario_bank(seed=32)
        for s in bank:
            router = run_handoff_protocol(s)
            seen = {(h.source_role, h.claim_kind)
                    for h in router.inboxes[ROLE_COMPLIANCE].peek()}
            required = {(r, k)
                        for (r, k, _p, _e) in s.causal_chain}
            missing = required - seen
            self.assertFalse(
                missing, msg=f"{s.scenario_id} missing {missing}")


# -----------------------------------------------------------------------------
# Grader
# -----------------------------------------------------------------------------


class TestGrader(unittest.TestCase):

    def test_grader_accepts_gold(self):
        bank = build_scenario_bank(seed=32)
        s = bank[0]
        text = (f"VERDICT: {s.gold_verdict}\n"
                f"FLAGS: {','.join(s.gold_flags)}\n"
                f"REMEDIATION: {s.gold_remediation}\n")
        g = grade_answer(s, text)
        self.assertTrue(g["full_correct"])

    def test_grader_rejects_wrong_verdict(self):
        bank = build_scenario_bank(seed=32)
        s = bank[0]
        text = (f"VERDICT: approved\n"
                f"FLAGS: {','.join(s.gold_flags)}\n"
                f"REMEDIATION: {s.gold_remediation}\n")
        g = grade_answer(s, text)
        self.assertFalse(g["verdict_correct"])

    def test_parse_is_case_insensitive(self):
        p = parse_answer("Verdict: Blocked\n"
                         "Flags: dpa_missing,retention_uncapped\n"
                         "Remediation: Require_Signed_DPA\n")
        self.assertEqual(p["verdict"], "blocked")
        self.assertEqual(p["flags"],
                         ("dpa_missing", "retention_uncapped"))
        self.assertEqual(p["remediation"], "require_signed_dpa")


# -----------------------------------------------------------------------------
# Failure attribution
# -----------------------------------------------------------------------------


class TestFailureAttribution(unittest.TestCase):

    def test_none_on_success(self):
        bank = build_scenario_bank(seed=32)
        s = bank[0]
        router = run_handoff_protocol(s)
        handoffs = tuple(router.inboxes[ROLE_COMPLIANCE].peek())
        g = {"full_correct": True, "verdict_correct": True,
             "flags_correct": True, "remediation_correct": True,
             "parsed": {}}
        self.assertEqual(
            attribute_failure(s, g, STRATEGY_SUBSTRATE, handoffs, False),
            FAILURE_NONE)

    def test_truncation_under_naive(self):
        s = build_scenario_bank(seed=32)[0]
        g = {"full_correct": False, "verdict_correct": False,
             "flags_correct": False, "remediation_correct": False,
             "parsed": {}}
        self.assertEqual(
            attribute_failure(s, g, STRATEGY_NAIVE, (), True),
            FAILURE_TRUNCATION)

    def test_missing_handoff_under_substrate_empty_inbox(self):
        s = build_scenario_bank(seed=32)[0]
        g = {"full_correct": False, "verdict_correct": False,
             "flags_correct": False, "remediation_correct": False,
             "parsed": {}}
        self.assertEqual(
            attribute_failure(s, g, STRATEGY_SUBSTRATE, (), False),
            FAILURE_MISSING_HANDOFF)

    def test_retrieval_miss_under_routing(self):
        s = build_scenario_bank(seed=32)[0]
        g = {"full_correct": False, "verdict_correct": False,
             "flags_correct": False, "remediation_correct": False,
             "parsed": {}}
        self.assertEqual(
            attribute_failure(s, g, STRATEGY_ROUTING, (), False),
            FAILURE_RETRIEVAL_MISS)


# -----------------------------------------------------------------------------
# Prompt assembly
# -----------------------------------------------------------------------------


class TestPromptAssembly(unittest.TestCase):

    def test_substrate_prompt_contains_cue(self):
        bank = build_scenario_bank(seed=32)
        s = bank[0]
        stream = naive_doc_stream(s)
        cue = {"verdict": "blocked",
               "flags": ("dpa_missing", "retention_uncapped"),
               "remediation": "require_signed_dpa_and_cap_retention"}
        prompt, delivered, truncated = build_auditor_prompt(
            s, STRATEGY_SUBSTRATE, stream, handoffs=(),
            substrate_cue=cue)
        self.assertIn("SUBSTRATE_ANSWER:", prompt)
        self.assertIn("VERDICT: blocked", prompt)
        self.assertFalse(truncated)

    def test_naive_prompt_has_documents(self):
        bank = build_scenario_bank(seed=32)
        s = bank[0]
        stream = naive_doc_stream(s)
        prompt, _, _ = build_auditor_prompt(
            s, STRATEGY_NAIVE, stream, handoffs=(), substrate_cue=None)
        self.assertIn("DELIVERED DOCUMENTS:", prompt)
        self.assertNotIn("SUBSTRATE_ANSWER:", prompt)

    def test_truncation_flag(self):
        bank = build_scenario_bank(seed=32, distractors_per_role=100)
        s = bank[0]
        stream = naive_doc_stream(s)
        prompt, delivered, trunc = build_auditor_prompt(
            s, STRATEGY_NAIVE, stream, handoffs=(),
            substrate_cue=None, max_docs_in_prompt=50)
        self.assertTrue(trunc)
        self.assertEqual(len(delivered), 50)


# -----------------------------------------------------------------------------
# End-to-end invariants
# -----------------------------------------------------------------------------


class TestEndToEndInvariants(unittest.TestCase):

    def test_substrate_matches_gold_under_mock(self):
        bank = build_scenario_bank(seed=32, distractors_per_role=6)
        aud = MockComplianceAuditor()
        rep = run_compliance_loop(
            bank, aud, strategies=(STRATEGY_SUBSTRATE,
                                    STRATEGY_SUBSTRATE_WRAP), seed=32)
        for m in rep.measurements:
            self.assertTrue(
                m.grading["full_correct"],
                msg=f"{m.scenario_id}/{m.strategy} failed under mock")

    def test_routing_fails_no_content(self):
        bank = build_scenario_bank(seed=32, distractors_per_role=6)
        aud = MockComplianceAuditor()
        rep = run_compliance_loop(
            bank, aud, strategies=(STRATEGY_ROUTING,), seed=32)
        for m in rep.measurements:
            self.assertFalse(
                m.grading["full_correct"],
                msg=(f"{m.scenario_id} passed routing but compliance "
                      "officer has no content"))

    def test_substrate_tokens_independent_of_distractors(self):
        aud = MockComplianceAuditor()
        rep_s = run_compliance_loop(
            build_scenario_bank(seed=32, distractors_per_role=3),
            aud, strategies=(STRATEGY_SUBSTRATE,), seed=32)
        aud = MockComplianceAuditor()
        rep_b = run_compliance_loop(
            build_scenario_bank(seed=32, distractors_per_role=80),
            aud, strategies=(STRATEGY_SUBSTRATE,), seed=32)
        t_s = rep_s.pooled()[STRATEGY_SUBSTRATE]["mean_prompt_tokens"]
        t_b = rep_b.pooled()[STRATEGY_SUBSTRATE]["mean_prompt_tokens"]
        self.assertLess(abs(t_b - t_s), 1e-6,
                         msg=f"substrate tokens moved: {t_s} vs {t_b}")

    def test_naive_tokens_grow_with_distractors(self):
        aud = MockComplianceAuditor()
        rep_s = run_compliance_loop(
            build_scenario_bank(seed=32, distractors_per_role=3),
            aud, strategies=(STRATEGY_NAIVE,), seed=32)
        aud = MockComplianceAuditor()
        rep_b = run_compliance_loop(
            build_scenario_bank(seed=32, distractors_per_role=80),
            aud, strategies=(STRATEGY_NAIVE,), seed=32)
        t_s = rep_s.pooled()[STRATEGY_NAIVE]["mean_prompt_tokens"]
        t_b = rep_b.pooled()[STRATEGY_NAIVE]["mean_prompt_tokens"]
        self.assertGreater(t_b, t_s * 2)


# -----------------------------------------------------------------------------
# Decoder behaviour
# -----------------------------------------------------------------------------


class TestDecoder(unittest.TestCase):

    def test_empty_handoffs_yield_approved(self):
        dec = decode_from_handoffs([])
        self.assertEqual(dec["verdict"], VERDICT_APPROVED)
        self.assertEqual(dec["flags"], ())

    def test_blocking_kind_yields_blocked(self):
        from vision_mvp.core.role_handoff import TypedHandoff
        h = TypedHandoff(
            handoff_id=0, source_role=ROLE_PRIVACY, source_agent_id=0,
            to_role=ROLE_COMPLIANCE, claim_kind=CLAIM_DPA_MISSING,
            payload="data_processing_agreement=missing",
            source_event_ids=(1,), round=0,
            payload_cid="x", prev_chain_hash="", chain_hash="")
        dec = decode_from_handoffs([h])
        self.assertEqual(dec["verdict"], VERDICT_BLOCKED)

    def test_conditional_only_yields_conditional(self):
        from vision_mvp.core.role_handoff import TypedHandoff
        h = TypedHandoff(
            handoff_id=0, source_role=ROLE_LEGAL, source_agent_id=0,
            to_role=ROLE_COMPLIANCE,
            claim_kind=CLAIM_LIABILITY_CAP_MISSING,
            payload="limits=none", source_event_ids=(1,),
            round=0, payload_cid="x", prev_chain_hash="", chain_hash="")
        dec = decode_from_handoffs([h])
        self.assertEqual(dec["verdict"], VERDICT_CONDITIONAL)


if __name__ == "__main__":
    unittest.main()
