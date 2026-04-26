"""SDK v3.7 contract tests — Phase-53 model-scale vs capsule-structure.

Locks the integration boundary that lets the team-coordination
benchmark accept a real-LLM producer-role extractor, plus the
parser robustness that turns model-emitted byte salad into a
closed-vocabulary candidate-handoff stream:

  W6-1   The team-lifecycle audit (T-1..T-7) holds for every
          (regime, strategy, scenario) cell of the phase-53
          benchmark when driven by a deterministic stub backend
          (no real network). Mechanically-checked.

  W6-2   The capsule-native runtime accepts a duck-typed
          ``LLMBackend`` substitute as the producer-role extractor
          backend in the Phase-53 driver. The team-coord pipeline
          seals TEAM_HANDOFF / ROLE_VIEW / TEAM_DECISION capsules
          end-to-end against an arbitrary backend (no spine
          modification). Proved by inspection + exercise.

  W6-3   ``parse_role_response`` parses the closed-vocabulary
          claim grammar robustly: it accepts ``KIND | payload`` /
          ``KIND: payload`` / ``KIND - payload`` / ``KIND — payload``,
          rejects kinds outside the allowed list, deduplicates by
          kind (first wins), strips preamble noise, and treats the
          ``NONE`` sentinel as zero claims.

The tests use a deterministic in-process stub backend so the
suite is hermetic. Adding a real-LLM smoke is left out of the
unit suite (gated to operator runs against the Mac 1 endpoint).
"""

from __future__ import annotations

import dataclasses
import json
import unittest
from typing import Any, Sequence

from vision_mvp.experiments.phase53_scale_vs_structure import (
    LLMExtractorStats, build_candidate_handoff_stream_via_llm,
    extract_claims_for_role_via_llm, parse_role_response,
    run_phase53,
)
from vision_mvp.tasks.incident_triage import (
    ROLE_DB_ADMIN, ROLE_MONITOR, ROLE_NETWORK, ROLE_SYSADMIN,
    build_scenario_bank,
)
from vision_mvp.wevra.llm_backend import LLMBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _ScriptedBackend:
    """Deterministic stub LLM. Returns a per-role canned response.

    Conforms to ``vision_mvp.wevra.llm_backend.LLMBackend``: exposes
    ``model``, ``base_url``, ``generate``. Used to make the team-coord
    pipeline runnable without a real LLM endpoint.
    """

    responses_by_role: dict[str, str]
    model: str = "stub-llm"
    base_url: "str | None" = None
    n_calls: int = 0

    def generate(self, prompt: str, max_tokens: int = 80,
                 temperature: float = 0.0) -> str:
        self.n_calls += 1
        # Identify the role from the prompt's leading line.
        for role in self.responses_by_role:
            if f"You are the {role!r} agent" in prompt:
                return self.responses_by_role[role]
        return "NONE"


# ---------------------------------------------------------------------------
# W6-3 — parse_role_response robustness
# ---------------------------------------------------------------------------


class ParseRoleResponseRobustnessTests(unittest.TestCase):
    """Parser must absorb LLM byte salad cleanly."""

    def test_basic_pipe_separator(self) -> None:
        out = parse_role_response(
            "DISK_FILL_CRITICAL | /var used=99% service=archival",
            allowed_kinds=("DISK_FILL_CRITICAL", "OOM_KILL"))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], "DISK_FILL_CRITICAL")
        self.assertIn("99%", out[0][1])

    def test_colon_separator(self) -> None:
        out = parse_role_response(
            "TLS_EXPIRED: rule=allow service=api reason=expired",
            allowed_kinds=("TLS_EXPIRED",))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], "TLS_EXPIRED")

    def test_em_dash_separator(self) -> None:
        out = parse_role_response(
            "OOM_KILL — process killed for memory service=web",
            allowed_kinds=("OOM_KILL",))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], "OOM_KILL")
        self.assertIn("service=web", out[0][1])

    def test_unknown_kind_rejected(self) -> None:
        out = parse_role_response(
            "BOGUS_KIND | ignore me",
            allowed_kinds=("DISK_FILL_CRITICAL",))
        self.assertEqual(out, [])

    def test_lowercase_kind_rejected(self) -> None:
        out = parse_role_response(
            "disk_fill_critical | foo",
            allowed_kinds=("DISK_FILL_CRITICAL",))
        self.assertEqual(out, [])

    def test_dedup_first_wins(self) -> None:
        out = parse_role_response(
            "OOM_KILL | first\nOOM_KILL | second",
            allowed_kinds=("OOM_KILL",))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][1], "first")

    def test_none_sentinel_alone_yields_empty(self) -> None:
        self.assertEqual(parse_role_response("NONE", ("OOM_KILL",)), [])
        self.assertEqual(
            parse_role_response("none", ("OOM_KILL",)), [])

    def test_preamble_dropped(self) -> None:
        out = parse_role_response(
            "Sure! Here are my claims:\n"
            "TLS_EXPIRED | service=api expired\n"
            "DNS_MISROUTE: SERVFAIL on api.internal",
            allowed_kinds=("TLS_EXPIRED", "DNS_MISROUTE"))
        self.assertEqual({k for (k, _) in out},
                         {"TLS_EXPIRED", "DNS_MISROUTE"})

    def test_max_claims_cap(self) -> None:
        text = "\n".join([
            "ERROR_RATE_SPIKE | a", "LATENCY_SPIKE | b",
            "SLOW_QUERY_OBSERVED | c", "POOL_EXHAUSTION | d",
            "DEADLOCK_SUSPECTED | e",
        ])
        allowed = ("ERROR_RATE_SPIKE", "LATENCY_SPIKE",
                   "SLOW_QUERY_OBSERVED", "POOL_EXHAUSTION",
                   "DEADLOCK_SUSPECTED")
        out = parse_role_response(text, allowed, max_claims=3)
        self.assertEqual(len(out), 3)

    def test_payload_truncated(self) -> None:
        long = "x" * 1000
        out = parse_role_response(
            f"OOM_KILL | {long}", ("OOM_KILL",))
        self.assertEqual(len(out), 1)
        self.assertLessEqual(len(out[0][1]), 240)

    def test_mixed_none_and_real_claims(self) -> None:
        out = parse_role_response(
            "ERROR_RATE_SPIKE | foo\nNONE\nLATENCY_SPIKE | bar",
            allowed_kinds=("ERROR_RATE_SPIKE", "LATENCY_SPIKE"))
        # Both real claims kept; NONE skipped.
        self.assertEqual({k for (k, _) in out},
                         {"ERROR_RATE_SPIKE", "LATENCY_SPIKE"})

    def test_empty_response_yields_empty(self) -> None:
        self.assertEqual(parse_role_response("", ("OOM_KILL",)), [])
        self.assertEqual(parse_role_response("   \n  \n", ("OOM_KILL",)),
                         [])


# ---------------------------------------------------------------------------
# W6-2 — backend duck-typing in the LLM extractor
# ---------------------------------------------------------------------------


class LLMExtractorBackendDuckTypingTests(unittest.TestCase):
    """The LLM extractor must accept any duck-typed
    ``LLMBackend`` substitute and produce candidate-handoff tuples
    that the team-coord pipeline understands.
    """

    def test_backend_protocol_membership(self) -> None:
        b = _ScriptedBackend(responses_by_role={})
        self.assertIsInstance(b, LLMBackend)

    def test_extractor_records_stats(self) -> None:
        b = _ScriptedBackend(responses_by_role={
            "sysadmin": "DISK_FILL_CRITICAL | /var used=99%",
        })
        bank = build_scenario_bank(seed=31, distractors_per_role=8)
        sc = bank[0]   # disk_fill_cron
        stats = LLMExtractorStats(model_tag="stub-llm")
        out = extract_claims_for_role_via_llm(
            ROLE_SYSADMIN, sc.per_role_events.get(ROLE_SYSADMIN, ()),
            sc, backend=b, stats=stats)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], "DISK_FILL_CRITICAL")
        self.assertEqual(stats.n_calls, 1)
        self.assertEqual(stats.n_total_emitted_claims, 1)
        self.assertEqual(stats.n_failed_calls, 0)

    def test_extractor_handles_backend_exception(self) -> None:
        @dataclasses.dataclass
        class _BrokenBackend:
            model: str = "broken"
            base_url: "str | None" = None

            def generate(self, prompt: str, max_tokens: int = 80,
                         temperature: float = 0.0) -> str:
                raise RuntimeError("simulated network error")

        bank = build_scenario_bank(seed=31, distractors_per_role=8)
        sc = bank[0]
        stats = LLMExtractorStats(model_tag="broken")
        out = extract_claims_for_role_via_llm(
            ROLE_SYSADMIN, sc.per_role_events.get(ROLE_SYSADMIN, ()),
            sc, backend=_BrokenBackend(), stats=stats)
        self.assertEqual(out, [])
        self.assertEqual(stats.n_calls, 1)
        self.assertEqual(stats.n_failed_calls, 1)

    def test_per_scenario_stream_routes_via_subscriptions(self) -> None:
        b = _ScriptedBackend(responses_by_role={
            "monitor":  "ERROR_RATE_SPIKE | service=api err=0.4",
            "db_admin": "SLOW_QUERY_OBSERVED | mean_ms=4200",
            "sysadmin": "DISK_FILL_CRITICAL | used=99% service=archival\n"
                         "CRON_OVERRUN | exit=137 duration_s=5400",
            "network":  "NONE",
        })
        bank = build_scenario_bank(seed=31, distractors_per_role=8)
        sc = bank[0]
        cands = build_candidate_handoff_stream_via_llm(
            sc, backend=b)
        # DISK_FILL_CRITICAL is subscribed by both auditor and DBA;
        # the others go to the auditor only. So we expect 2 routings
        # for DISK_FILL_CRITICAL plus 1 each for the others = 5.
        kinds = [k for (_s, _t, k, _p, _e) in cands]
        self.assertIn("DISK_FILL_CRITICAL", kinds)
        self.assertIn("ERROR_RATE_SPIKE", kinds)
        self.assertIn("SLOW_QUERY_OBSERVED", kinds)
        self.assertIn("CRON_OVERRUN", kinds)
        self.assertEqual(b.n_calls, 4)   # one per producer role


# ---------------------------------------------------------------------------
# W6-1 — team-lifecycle audit holds end-to-end with a stub backend
# ---------------------------------------------------------------------------


class Phase53AuditOkGridTests(unittest.TestCase):
    """The phase-53 driver returns ``audit_ok=True`` for every
    (regime, capsule_strategy) cell when run with a deterministic
    stub backend. Substrate is by design ``audit_ok=False`` (it is
    not in the capsule ledger).
    """

    def test_synthetic_only_audit_ok_grid(self) -> None:
        # No LLM calls — just exercises the synthetic regime end-to-end.
        rep = run_phase53(
            model_tags=("synthetic",),
            n_eval=2, K_auditor=4, T_auditor=128,
            verbose=False)
        grid = rep["audit_ok_grid"]
        self.assertEqual(set(grid.keys()), {"synthetic"})
        for s in ("capsule_fifo", "capsule_priority",
                   "capsule_coverage", "capsule_learned"):
            self.assertTrue(grid["synthetic"][s],
                              f"audit_ok=False for {s!r}")
        # Substrate is intentionally False.
        self.assertFalse(grid["synthetic"]["substrate"])

    def test_decomposition_and_audit_ok_with_stub_backend(self) -> None:
        """Run a 1-regime, n_eval=2 phase53 with a stub LLM backend
        injected by monkey-patching the OllamaBackend factory inside
        the experiment module.

        Hermetic: zero network. Locks the W6-1 audit_ok property
        and the W6-2 backend integration in one go.
        """
        from vision_mvp.experiments import phase53_scale_vs_structure as P53

        scripted = _ScriptedBackend(responses_by_role={
            "monitor":  "ERROR_RATE_SPIKE | service=api",
            "db_admin": "SLOW_QUERY_OBSERVED | mean_ms=4200 service=orders",
            "sysadmin": "DISK_FILL_CRITICAL | used=99% service=archival",
            "network":  "NONE",
        })
        original = P53.OllamaBackend
        try:
            P53.OllamaBackend = lambda **kw: scripted   # type: ignore
            rep = run_phase53(
                model_tags=("stub-llm",),
                endpoint="http://stub.invalid",
                n_eval=2, K_auditor=4, T_auditor=128,
                verbose=False)
        finally:
            P53.OllamaBackend = original  # restore

        # All capsule strategies under the stub regime audit OK.
        grid = rep["audit_ok_grid"]
        for s in ("capsule_fifo", "capsule_priority",
                   "capsule_coverage", "capsule_learned"):
            self.assertTrue(grid["stub-llm"][s],
                              f"audit_ok=False for {s!r}")
        # Decomposition has structure_gain entry for the stub regime.
        self.assertIn("stub-llm", rep["decomposition"]["structure_gain"])
        # Schema name pinned.
        self.assertEqual(rep["schema"], "phase53.scale_vs_structure.v1")
        # Extractor stats recorded for the LLM regime.
        self.assertIn("stub-llm", rep["extractor_stats"])
        self.assertGreater(
            rep["extractor_stats"]["stub-llm"]["n_calls"], 0)
        self.assertEqual(
            rep["extractor_stats"]["stub-llm"]["n_failed_calls"], 0)


# ---------------------------------------------------------------------------
# Schema lock — the phase-53 report is JSON-serialisable end-to-end
# ---------------------------------------------------------------------------


class Phase53ReportSchemaTests(unittest.TestCase):

    def test_synthetic_report_round_trips_through_json(self) -> None:
        rep = run_phase53(
            model_tags=("synthetic",),
            n_eval=2, K_auditor=4, T_auditor=128,
            verbose=False)
        text = json.dumps(rep, sort_keys=True)
        parsed = json.loads(text)
        self.assertEqual(parsed["schema"], rep["schema"])
        self.assertIn("pooled_per_regime", parsed)
        self.assertIn("decomposition", parsed)
        self.assertIn("audit_ok_grid", parsed)


if __name__ == "__main__":
    unittest.main()
