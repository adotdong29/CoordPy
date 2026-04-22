"""Unit tests for the Phase-31 multi-role incident-triage benchmark.

Covers:
  * Scenario bank determinism (same seed → same gold, same events).
  * Role-subscription wiring.
  * Claim-extractor correctness on causal events and distractors.
  * Handoff protocol end-to-end.
  * Oracle relevance + handoff-relevance predicates.
  * Grader behaviour on well-formed and malformed answers.
  * Failure attribution across strategies.
  * Full ``run_incident_loop`` cross-strategy invariants.
"""
from __future__ import annotations

import unittest

from vision_mvp.tasks.incident_triage import (
    ALL_CLAIMS, ALL_ROLES, ALL_STRATEGIES,
    CLAIM_DISK_FILL_CRITICAL, CLAIM_ERROR_RATE_SPIKE,
    CLAIM_POOL_EXHAUSTION, CLAIM_SLOW_QUERY_OBSERVED,
    EVENT_METRIC_SAMPLE, EVENT_OS_EVENT, EVENT_SQL_STAT,
    FAILURE_LLM_ERROR, FAILURE_MISSING_HANDOFF, FAILURE_NONE,
    FAILURE_RETRIEVAL_MISS, FAILURE_TRUNCATION,
    MockIncidentAuditor, ROLE_AUDITOR, ROLE_DB_ADMIN,
    ROLE_MONITOR, ROLE_NETWORK, ROLE_SYSADMIN,
    STRATEGY_NAIVE, STRATEGY_ROUTING, STRATEGY_SUBSTRATE,
    STRATEGY_SUBSTRATE_WRAP,
    attribute_failure, build_auditor_prompt, build_role_subscriptions,
    build_scenario_bank, extract_claims_for_role, grade_answer,
    handoff_is_relevant, naive_event_stream, oracle_relevance,
    parse_answer, run_handoff_protocol, run_incident_loop,
)


# -----------------------------------------------------------------------------
# Scenario bank
# -----------------------------------------------------------------------------


class TestScenarioBank(unittest.TestCase):

    def test_bank_deterministic(self):
        a = build_scenario_bank(seed=31, distractors_per_role=10)
        b = build_scenario_bank(seed=31, distractors_per_role=10)
        self.assertEqual(len(a), len(b))
        for s1, s2 in zip(a, b):
            self.assertEqual(s1.scenario_id, s2.scenario_id)
            self.assertEqual(s1.gold_root_cause, s2.gold_root_cause)
            self.assertEqual(s1.gold_services, s2.gold_services)
            self.assertEqual(s1.gold_remediation, s2.gold_remediation)

    def test_bank_has_five_scenarios(self):
        a = build_scenario_bank(seed=31)
        self.assertEqual(len(a), 5)
        ids = {s.scenario_id for s in a}
        self.assertEqual(ids, {
            "disk_fill_cron", "tls_expiry_healthcheck_loop",
            "dns_misroute_leak", "memory_leak_oom",
            "deadlock_pool_exhaustion"})

    def test_every_scenario_has_causal_events(self):
        for s in build_scenario_bank(seed=31):
            causal = sum(1 for _role, evs in s.per_role_events.items()
                         for ev in evs if ev.is_causal)
            self.assertGreater(causal, 0,
                               msg=f"{s.scenario_id} has no causal events")

    def test_distractors_scale(self):
        small = build_scenario_bank(seed=31, distractors_per_role=3)
        big = build_scenario_bank(seed=31, distractors_per_role=20)
        sm_ev = sum(len(evs) for evs in small[0].per_role_events.values())
        bg_ev = sum(len(evs) for evs in big[0].per_role_events.values())
        self.assertGreater(bg_ev, sm_ev)


# -----------------------------------------------------------------------------
# Role subscriptions
# -----------------------------------------------------------------------------


class TestRoleSubscriptions(unittest.TestCase):

    def test_every_claim_reaches_auditor(self):
        subs = build_role_subscriptions()
        # For every (source, kind) pair in the table, auditor must be
        # subscribed (or be a valid non-auditor target — we assert
        # auditor specifically, which is the programme invariant).
        for (src, kind), consumers in subs._table.items():
            self.assertIn(ROLE_AUDITOR, consumers,
                           msg=f"{(src, kind)} skips auditor")

    def test_every_claim_kind_has_at_least_one_subscription(self):
        subs = build_role_subscriptions()
        covered = subs.all_claim_kinds()
        for k in ALL_CLAIMS:
            self.assertIn(k, covered,
                           msg=f"{k} has no subscription")


# -----------------------------------------------------------------------------
# Claim extractors
# -----------------------------------------------------------------------------


class TestClaimExtractors(unittest.TestCase):

    def test_monitor_extracts_error_spike(self):
        bank = build_scenario_bank(seed=31, distractors_per_role=0)
        disk = bank[0]  # disk_fill_cron
        monitor_events = disk.per_role_events[ROLE_MONITOR]
        claims = extract_claims_for_role(ROLE_MONITOR,
                                          monitor_events, disk)
        kinds = {k for (k, _p, _e) in claims}
        self.assertIn(CLAIM_ERROR_RATE_SPIKE, kinds)

    def test_distractors_do_not_produce_claims(self):
        bank = build_scenario_bank(seed=31, distractors_per_role=40)
        disk = bank[0]
        # Take ONLY the distractor events (is_causal=False).
        distractors = [ev for ev in disk.per_role_events[ROLE_MONITOR]
                       if not ev.is_causal]
        claims = extract_claims_for_role(ROLE_MONITOR,
                                          distractors, disk)
        self.assertEqual(claims, [])

    def test_db_admin_detects_slow_query_and_pool(self):
        bank = build_scenario_bank(seed=31, distractors_per_role=0)
        disk = bank[0]
        db_events = disk.per_role_events[ROLE_DB_ADMIN]
        claims = extract_claims_for_role(ROLE_DB_ADMIN,
                                          db_events, disk)
        kinds = {k for (k, _p, _e) in claims}
        self.assertIn(CLAIM_SLOW_QUERY_OBSERVED, kinds)
        self.assertIn(CLAIM_POOL_EXHAUSTION, kinds)

    def test_sysadmin_disk_fill(self):
        bank = build_scenario_bank(seed=31, distractors_per_role=0)
        disk = bank[0]
        sys_events = disk.per_role_events[ROLE_SYSADMIN]
        claims = extract_claims_for_role(ROLE_SYSADMIN,
                                          sys_events, disk)
        kinds = {k for (k, _p, _e) in claims}
        self.assertIn(CLAIM_DISK_FILL_CRITICAL, kinds)


# -----------------------------------------------------------------------------
# Oracle relevance
# -----------------------------------------------------------------------------


class TestOracleRelevance(unittest.TestCase):

    def test_fixed_point_always_relevant(self):
        bank = build_scenario_bank(seed=31)
        stream = naive_event_stream(bank[0])
        fixed = [ev for ev in stream if ev.is_fixed_point]
        for ev in fixed:
            for role in ALL_ROLES:
                self.assertTrue(oracle_relevance(ev, role, bank[0]))

    def test_distractor_irrelevant_to_owning_role(self):
        bank = build_scenario_bank(seed=31, distractors_per_role=3)
        s = bank[0]
        for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN,
                      ROLE_NETWORK):
            for ev in s.per_role_events[role]:
                if not ev.is_causal:
                    self.assertFalse(oracle_relevance(ev, role, s),
                                      msg=f"role {role} ev {ev.event_id}"
                                           " relevant but distractor")

    def test_auditor_raw_events_only_causal(self):
        bank = build_scenario_bank(seed=31)
        s = bank[0]
        stream = naive_event_stream(s)
        for ev in stream:
            if ev.is_fixed_point:
                continue
            if ev.is_causal:
                self.assertTrue(oracle_relevance(ev, ROLE_AUDITOR, s))
            else:
                self.assertFalse(oracle_relevance(ev, ROLE_AUDITOR, s))

    def test_cross_role_event_irrelevant(self):
        """A monitor event is not observable by the network engineer
        under the role taxonomy, therefore never relevant to it."""
        bank = build_scenario_bank(seed=31)
        s = bank[0]
        for ev in s.per_role_events[ROLE_MONITOR]:
            self.assertFalse(
                oracle_relevance(ev, ROLE_NETWORK, s),
                msg="monitor event relevant to network — taxonomy bug")


# -----------------------------------------------------------------------------
# Handoff protocol
# -----------------------------------------------------------------------------


class TestHandoffProtocol(unittest.TestCase):

    def test_handoff_chain_valid(self):
        bank = build_scenario_bank(seed=31)
        router = run_handoff_protocol(bank[0])
        self.assertTrue(router.verify())
        self.assertGreater(router.log_length(), 0)

    def test_auditor_inbox_has_all_required_claims(self):
        bank = build_scenario_bank(seed=31)
        for s in bank:
            router = run_handoff_protocol(s)
            inbox = router.inboxes[ROLE_AUDITOR]
            seen = {(h.source_role, h.claim_kind)
                    for h in inbox.peek()}
            required = {(role, kind)
                        for (role, kind, _p, _e) in s.causal_chain}
            missing = required - seen
            self.assertFalse(
                missing, msg=f"{s.scenario_id}: missing handoffs "
                              f"{missing}")

    def test_handoff_is_relevant_matches_chain(self):
        bank = build_scenario_bank(seed=31)
        s = bank[0]
        router = run_handoff_protocol(s)
        for h in router.inboxes[ROLE_AUDITOR].peek():
            self.assertTrue(handoff_is_relevant(h, s))


# -----------------------------------------------------------------------------
# Grader
# -----------------------------------------------------------------------------


class TestGrader(unittest.TestCase):

    def test_grader_accepts_gold_format(self):
        bank = build_scenario_bank(seed=31)
        s = bank[0]
        text = (f"ROOT_CAUSE: {s.gold_root_cause}\n"
                f"SERVICES: {','.join(s.gold_services)}\n"
                f"REMEDIATION: {s.gold_remediation}\n")
        g = grade_answer(s, text)
        self.assertTrue(g["full_correct"])
        self.assertTrue(g["root_cause_correct"])
        self.assertTrue(g["services_correct"])
        self.assertTrue(g["remediation_correct"])

    def test_grader_rejects_wrong_root_cause(self):
        bank = build_scenario_bank(seed=31)
        s = bank[0]
        text = ("ROOT_CAUSE: wrong_label\nSERVICES: "
                + ",".join(s.gold_services)
                + f"\nREMEDIATION: {s.gold_remediation}\n")
        g = grade_answer(s, text)
        self.assertFalse(g["root_cause_correct"])

    def test_grader_rejects_wrong_services(self):
        bank = build_scenario_bank(seed=31)
        s = bank[0]
        text = (f"ROOT_CAUSE: {s.gold_root_cause}\nSERVICES: extra\n"
                f"REMEDIATION: {s.gold_remediation}\n")
        g = grade_answer(s, text)
        self.assertFalse(g["services_correct"])

    def test_parse_answer_case_insensitive(self):
        p = parse_answer("Root_Cause: X\nServices: a,b\n"
                         "Remediation: Y\n")
        self.assertEqual(p["root_cause"], "x")
        self.assertEqual(p["services"], ("a", "b"))
        self.assertEqual(p["remediation"], "y")


# -----------------------------------------------------------------------------
# Failure attribution
# -----------------------------------------------------------------------------


class TestFailureAttribution(unittest.TestCase):

    def test_none_on_success(self):
        bank = build_scenario_bank(seed=31)
        s = bank[0]
        router = run_handoff_protocol(s)
        handoffs = tuple(router.inboxes[ROLE_AUDITOR].peek())
        g = {"full_correct": True, "root_cause_correct": True,
             "services_correct": True, "remediation_correct": True,
             "parsed": {}}
        self.assertEqual(
            attribute_failure(s, g, STRATEGY_SUBSTRATE, handoffs, False),
            FAILURE_NONE)

    def test_truncation_attribution_under_naive(self):
        bank = build_scenario_bank(seed=31)
        s = bank[0]
        g = {"full_correct": False, "root_cause_correct": False,
             "services_correct": False, "remediation_correct": False,
             "parsed": {}}
        self.assertEqual(
            attribute_failure(s, g, STRATEGY_NAIVE, (), True),
            FAILURE_TRUNCATION)

    def test_missing_handoff_attribution_under_substrate(self):
        bank = build_scenario_bank(seed=31)
        s = bank[0]
        g = {"full_correct": False, "root_cause_correct": False,
             "services_correct": False, "remediation_correct": False,
             "parsed": {}}
        # No handoffs delivered → every required claim is missing.
        self.assertEqual(
            attribute_failure(s, g, STRATEGY_SUBSTRATE, (), False),
            FAILURE_MISSING_HANDOFF)

    def test_retrieval_miss_attribution_under_routing(self):
        bank = build_scenario_bank(seed=31)
        s = bank[0]
        g = {"full_correct": False, "root_cause_correct": False,
             "services_correct": False, "remediation_correct": False,
             "parsed": {}}
        self.assertEqual(
            attribute_failure(s, g, STRATEGY_ROUTING, (), False),
            FAILURE_RETRIEVAL_MISS)

    def test_llm_error_attribution_when_handoffs_present_but_wrong(self):
        bank = build_scenario_bank(seed=31)
        s = bank[0]
        router = run_handoff_protocol(s)
        handoffs = tuple(router.inboxes[ROLE_AUDITOR].peek())
        g = {"full_correct": False, "root_cause_correct": False,
             "services_correct": True, "remediation_correct": True,
             "parsed": {}}
        self.assertEqual(
            attribute_failure(s, g, STRATEGY_SUBSTRATE, handoffs, False),
            FAILURE_LLM_ERROR)


# -----------------------------------------------------------------------------
# Prompt assembly — truncation
# -----------------------------------------------------------------------------


class TestPromptAssembly(unittest.TestCase):

    def test_substrate_prompt_includes_cue(self):
        bank = build_scenario_bank(seed=31)
        s = bank[0]
        stream = naive_event_stream(s)
        cue = {"root_cause": "disk_fill",
               "services": ("api", "orders", "web"),
               "remediation": "rotate_logs_and_clear_backup"}
        prompt, delivered, truncated = build_auditor_prompt(
            s, STRATEGY_SUBSTRATE, stream, handoffs=(),
            substrate_cue=cue)
        self.assertIn("SUBSTRATE_ANSWER:", prompt)
        self.assertIn("ROOT_CAUSE: disk_fill", prompt)
        self.assertIn("api,orders,web", prompt)
        self.assertFalse(truncated)

    def test_naive_prompt_excludes_substrate_answer(self):
        bank = build_scenario_bank(seed=31)
        s = bank[0]
        stream = naive_event_stream(s)
        prompt, _, _ = build_auditor_prompt(
            s, STRATEGY_NAIVE, stream, handoffs=(), substrate_cue=None)
        self.assertIn("DELIVERED EVENTS:", prompt)
        self.assertNotIn("SUBSTRATE_ANSWER:", prompt)

    def test_truncation_flag_set_when_exceeded(self):
        bank = build_scenario_bank(seed=31, distractors_per_role=100)
        s = bank[0]
        stream = naive_event_stream(s)
        prompt, delivered, truncated = build_auditor_prompt(
            s, STRATEGY_NAIVE, stream, handoffs=(),
            substrate_cue=None, max_events_in_prompt=50)
        self.assertTrue(truncated)
        self.assertEqual(len(delivered), 50)


# -----------------------------------------------------------------------------
# End-to-end benchmark invariants
# -----------------------------------------------------------------------------


class TestEndToEndInvariants(unittest.TestCase):

    def test_substrate_always_matches_gold_under_mock(self):
        bank = build_scenario_bank(seed=31, distractors_per_role=6)
        auditor = MockIncidentAuditor()
        rep = run_incident_loop(
            bank, auditor,
            strategies=(STRATEGY_SUBSTRATE, STRATEGY_SUBSTRATE_WRAP),
            seed=31)
        for m in rep.measurements:
            self.assertTrue(
                m.grading["full_correct"],
                msg=f"{m.scenario_id}/{m.strategy} failed under mock")

    def test_routing_fails_under_mock_because_no_content(self):
        bank = build_scenario_bank(seed=31, distractors_per_role=6)
        auditor = MockIncidentAuditor()
        rep = run_incident_loop(
            bank, auditor, strategies=(STRATEGY_ROUTING,), seed=31)
        for m in rep.measurements:
            self.assertFalse(
                m.grading["full_correct"],
                msg=f"{m.scenario_id} passed under routing but auditor "
                     "should have no content")

    def test_substrate_tokens_bounded_independent_of_distractors(self):
        auditor = MockIncidentAuditor()
        bank_small = build_scenario_bank(seed=31,
                                           distractors_per_role=3)
        bank_big = build_scenario_bank(seed=31,
                                         distractors_per_role=80)
        rep_s = run_incident_loop(bank_small, auditor,
                                   strategies=(STRATEGY_SUBSTRATE,),
                                   seed=31)
        rep_b = run_incident_loop(bank_big, auditor,
                                   strategies=(STRATEGY_SUBSTRATE,),
                                   seed=31)
        tok_s = rep_s.pooled()[STRATEGY_SUBSTRATE]["mean_prompt_tokens"]
        tok_b = rep_b.pooled()[STRATEGY_SUBSTRATE]["mean_prompt_tokens"]
        # Substrate prompt is a constant cue + fixed-point events; its
        # token count must not grow with the distractor-driven event
        # stream.
        self.assertLess(abs(tok_b - tok_s), 1e-6,
                         msg=(f"substrate prompt token count depends on "
                              f"stream size: {tok_s} vs {tok_b}"))

    def test_naive_tokens_grow_with_distractors(self):
        auditor = MockIncidentAuditor()
        bank_small = build_scenario_bank(seed=31,
                                           distractors_per_role=3)
        bank_big = build_scenario_bank(seed=31,
                                         distractors_per_role=80)
        rep_s = run_incident_loop(bank_small, auditor,
                                   strategies=(STRATEGY_NAIVE,),
                                   seed=31)
        rep_b = run_incident_loop(bank_big, auditor,
                                   strategies=(STRATEGY_NAIVE,),
                                   seed=31)
        tok_s = rep_s.pooled()[STRATEGY_NAIVE]["mean_prompt_tokens"]
        tok_b = rep_b.pooled()[STRATEGY_NAIVE]["mean_prompt_tokens"]
        self.assertGreater(tok_b, tok_s * 2,
                            msg=("naive prompt should grow with event "
                                 f"stream size but did not: {tok_s} -> "
                                 f"{tok_b}"))

    def test_substrate_fails_gracefully_when_no_content(self):
        """If the auditor's inbox has NO handoffs, substrate delivery
        carries only fixed-point events and an empty cue — the
        answer cannot be constructed. Attribution should name
        ``missing_handoff`` rather than crashing."""
        # Synthesise a scenario whose per-role events have no
        # causal signal — the extractors emit nothing.
        from vision_mvp.tasks.incident_triage import (
            IncidentScenario, IncidentEvent,
            ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN,
            ROLE_NETWORK, ROLE_AUDITOR,
        )
        s = IncidentScenario(
            scenario_id="no_signal_test", description="no signal",
            gold_root_cause="disk_fill",
            gold_services=("api",),
            gold_remediation="rotate_logs_and_clear_backup",
            causal_chain=(
                (ROLE_SYSADMIN, CLAIM_DISK_FILL_CRITICAL,
                 "/var/log used=99%", (0,)),
            ),
            per_role_events={
                ROLE_MONITOR: (), ROLE_DB_ADMIN: (),
                ROLE_SYSADMIN: (), ROLE_NETWORK: (),
                ROLE_AUDITOR: (),
            },
        )
        auditor = MockIncidentAuditor()
        rep = run_incident_loop(
            [s], auditor, strategies=(STRATEGY_SUBSTRATE,), seed=31)
        m = rep.measurements[0]
        # The auditor produced an "unknown" root cause — graceful
        # failure, attribution is missing_handoff.
        self.assertFalse(m.grading["full_correct"])
        self.assertEqual(m.failure_kind, FAILURE_MISSING_HANDOFF)


if __name__ == "__main__":
    unittest.main()
