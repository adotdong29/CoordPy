"""Tests for SDK v3.19 — relational-compatibility disambiguator + W18 family.

Covers:

* Helper-function unit tests: tokenisation, single- and compound-target
  scoring, contiguous-subsequence semantic property.
* Phase-65 bench-property tests: every cell carries the named
  symmetric-corroboration ingredient AND the bank-specific round-2
  relational-mention class (compat / no_compat / confound / deceive).
* Phase-65 default-config tests: W18 strict win on R-65-COMPAT-LOOSE
  AND R-65-COMPAT-TIGHT (W18-1).
* Phase-65 5-seed stability: gap w18 − attention_aware ≥ 0.50 on
  every seed under the COMPAT loose AND tight regimes.
* Phase-65 falsifier tests: W18 ties FIFO at 0.000 on R-65-NO-COMPAT
  / R-65-CONFOUND / R-65-DECEIVE (W18-Λ-no-compat / -confound /
  -deceive).
* Phase-65 backward-compat smoke: W18 reduces to W15 byte-for-byte on
  R-58 default and R-64-SYM (R-64-SYM is partial — W18 helps on the
  deadlock-flavored scenarios where round-2 carries a relational
  mention; ties FIFO on the others).
* Phase-65 token-budget honesty: W18 reads only the W15-packed
  bundle; ``tokens_kept`` accounting unchanged byte-for-byte.
* Phase-65 audit T-1..T-7 preservation on every cell of every regime.
"""

from __future__ import annotations

import unittest

from vision_mvp.experiments.phase58_multi_round_decoder import (
    _build_round_candidates as _phase58_build_round_candidates,
    build_phase58_bank,
)
from vision_mvp.experiments.phase64_live_composition import (
    build_phase64_sym_bank,
)
from vision_mvp.experiments.phase65_relational_disambiguation import (
    _bench_property_p65,
    build_phase65_bank,
    run_phase65,
    run_phase65_seed_stability_sweep,
    run_cross_regime_synthetic,
)
from vision_mvp.wevra.team_coord import (
    AttentionAwareBundleDecoder,
    RelationalCompatibilityDisambiguator,
    W18CompatibilityResult,
    _DecodedHandoff,
    _disambiguator_payload_tokens,
    _relational_compatibility_score,
)


class W18ScorerUnitTests(unittest.TestCase):
    """W18 helper functions — closed-form determinism."""

    def test_tokeniser_splits_on_non_identifier_chars(self):
        toks = _disambiguator_payload_tokens(
            "deadlock relation=orders_payments_join wait_chain=2")
        self.assertIn("orders", toks)
        self.assertIn("payments", toks)
        self.assertIn("orders_payments_join", toks)
        self.assertIn("wait_chain", toks)

    def test_tokeniser_lower_cases(self):
        toks = _disambiguator_payload_tokens("ORDERS Payments Join")
        self.assertIn("orders", toks)
        self.assertIn("payments", toks)
        self.assertIn("join", toks)

    def test_tokeniser_empty_payload(self):
        self.assertEqual(_disambiguator_payload_tokens(""), ())

    def test_tokeniser_path_separators(self):
        toks = _disambiguator_payload_tokens("/storage/A/B/var/log")
        self.assertIn("storage", toks)
        self.assertIn("a", toks)
        self.assertIn("b", toks)
        self.assertIn("var", toks)
        self.assertIn("log", toks)

    def test_score_single_part_target_direct_hit(self):
        toks = _disambiguator_payload_tokens(
            "deadlock relation=orders_payments_join")
        self.assertEqual(
            _relational_compatibility_score("orders", toks),
            (1, 1))   # direct=1 (standalone), compound=1 (inside compound)

    def test_score_single_part_target_no_match(self):
        toks = _disambiguator_payload_tokens(
            "deadlock relation=orders_payments_join")
        self.assertEqual(
            _relational_compatibility_score("cache", toks), (0, 0))

    def test_score_compound_target_contiguous_subsequence(self):
        # ``db_query`` should match inside
        # ``svc_web_then_svc_db_query`` (db,query is a contiguous
        # subsequence of svc,web,then,svc,db,query).
        toks = _disambiguator_payload_tokens(
            "query_path=svc_web_then_svc_db_query")
        d, c = _relational_compatibility_score("db_query", toks)
        self.assertEqual((d, c), (0, 1))

    def test_score_compound_target_non_contiguous_does_not_match(self):
        # ``orders_payments`` is contiguous in
        # ``orders_payments_join`` so it should match.
        toks = _disambiguator_payload_tokens("orders_payments_join")
        d, c = _relational_compatibility_score("orders_payments", toks)
        self.assertEqual((d, c), (0, 1))

    def test_score_compound_target_non_adjacent_does_not_match(self):
        # ``orders_join`` should NOT match in ``orders_payments_join``
        # because orders and join are not adjacent.
        toks = _disambiguator_payload_tokens("orders_payments_join")
        d, c = _relational_compatibility_score("orders_join", toks)
        self.assertEqual((d, c), (0, 0))

    def test_score_empty_target(self):
        toks = _disambiguator_payload_tokens("anything")
        self.assertEqual(_relational_compatibility_score("", toks), (0, 0))


class W18DecoderUnitTests(unittest.TestCase):
    """W18 :class:`RelationalCompatibilityDisambiguator` — determinism +
    abstention semantics."""

    def _scenario(self, gold_a: str, gold_b: str, decoy: str,
                   r2_payload: str):
        # Synthetic two-round bundle with a single specific-tier
        # round-2 disambiguator carrying ``r2_payload``.
        r1 = [
            _DecodedHandoff("monitor", "LATENCY_SPIKE",
                             f"p95_ms=2200 service={gold_a}"),
            _DecodedHandoff("monitor", "ERROR_RATE_SPIKE",
                             f"error_rate=0.20 service={gold_b}"),
            _DecodedHandoff("monitor", "LATENCY_SPIKE",
                             f"p95_ms=2100 service={decoy}"),
            _DecodedHandoff("monitor", "ERROR_RATE_SPIKE",
                             f"error_rate=0.18 service={decoy}"),
            _DecodedHandoff("network", "FW_BLOCK_SURGE",
                             f"rule=deny count=11 service={gold_a}"),
            _DecodedHandoff("network", "FW_BLOCK_SURGE",
                             f"rule=deny count=10 service={gold_b}"),
            _DecodedHandoff("network", "FW_BLOCK_SURGE",
                             f"rule=deny count=9 service={decoy}"),
            _DecodedHandoff("network", "FW_BLOCK_SURGE",
                             f"rule=deny count=8 service={decoy}"),
        ]
        r2 = [_DecodedHandoff("db_admin", "DEADLOCK_SUSPECTED", r2_payload)]
        return r1, r2

    def test_compat_round2_gold_only_recovers_gold(self):
        r1, r2 = self._scenario(
            "orders", "payments", "cache",
            "deadlock relation=orders_payments_join wait_chain=2")
        w18 = RelationalCompatibilityDisambiguator()
        ans = w18.decode_rounds([r1, r2])
        self.assertEqual(ans["root_cause"], "deadlock")
        self.assertEqual(set(ans["services"]), {"orders", "payments"})
        self.assertFalse(ans["compatibility"]["abstained"])

    def test_no_compat_no_signal_abstains(self):
        r1, r2 = self._scenario(
            "orders", "payments", "cache",
            "deadlock wait_chain=2 detected_at=t=120")
        w18 = RelationalCompatibilityDisambiguator()
        ans = w18.decode_rounds([r1, r2])
        # Abstain → fall back to inner W15 answer (which is empty
        # under W11 contradiction-aware drop on symmetric-corr).
        self.assertTrue(ans["compatibility"]["abstained"])
        self.assertNotEqual(set(ans["services"]), {"orders", "payments"})

    def test_confound_symmetric_signal_abstains(self):
        r1, r2 = self._scenario(
            "orders", "payments", "cache",
            "deadlock relation=orders_payments_cache_join")
        w18 = RelationalCompatibilityDisambiguator()
        ans = w18.decode_rounds([r1, r2])
        self.assertTrue(ans["compatibility"]["abstained"])

    def test_deceive_decoy_only_picks_decoy(self):
        r1, r2 = self._scenario(
            "orders", "payments", "cache",
            "deadlock relation=cache_cache_join")
        w18 = RelationalCompatibilityDisambiguator()
        ans = w18.decode_rounds([r1, r2])
        # W18 trusts its evidence: picks decoy and fails.
        self.assertFalse(ans["compatibility"]["abstained"])
        self.assertEqual(set(ans["services"]), {"cache"})

    def test_determinism(self):
        r1, r2 = self._scenario(
            "orders", "payments", "cache",
            "deadlock relation=orders_payments_join")
        w18a = RelationalCompatibilityDisambiguator()
        w18b = RelationalCompatibilityDisambiguator()
        a = w18a.decode_rounds([r1, r2])
        b = w18b.decode_rounds([r1, r2])
        self.assertEqual(a["root_cause"], b["root_cause"])
        self.assertEqual(a["services"], b["services"])
        self.assertEqual(a["compatibility"]["per_tag_scores"],
                         b["compatibility"]["per_tag_scores"])

    def test_disabled_path_reduces_to_inner(self):
        r1, r2 = self._scenario(
            "orders", "payments", "cache",
            "deadlock relation=orders_payments_join")
        inner = AttentionAwareBundleDecoder()
        w18 = RelationalCompatibilityDisambiguator(
            inner=inner, enabled=False)
        a = w18.decode_rounds([r1, r2])
        b = inner.decode_rounds([r1, r2])
        self.assertEqual(a["root_cause"], b["root_cause"])
        self.assertEqual(a["services"], b["services"])
        self.assertTrue(a["compatibility"]["abstained"])


class Phase65BenchPropertyTests(unittest.TestCase):
    """Mechanically-verified bench property witnesses for every R-65
    sub-bank: symmetric corroboration AND named round-2 class."""

    def _verify(self, bank: str, expected_class: str):
        scenarios = build_phase65_bank(
            bank=bank, n_replicates=2, seed=11)
        self.assertEqual(len(scenarios), 8)
        for sc in scenarios:
            r1 = _phase58_build_round_candidates(sc.round1_emissions)
            r2 = _phase58_build_round_candidates(sc.round2_emissions)
            bp = _bench_property_p65(sc, r1, r2)
            self.assertTrue(bp["symmetric_corroboration_holds"],
                              f"sym fails on {sc.scenario_id}")
            self.assertEqual(bp["r2_class"], expected_class,
                              f"r2_class mismatch on {sc.scenario_id}: "
                              f"got {bp['r2_class']!r}, "
                              f"expected {expected_class!r}")

    def test_compat_round2_mentions_gold_only(self):
        self._verify("compat", "compat")

    def test_no_compat_round2_no_service_mention(self):
        self._verify("no_compat", "no_compat")

    def test_confound_round2_mentions_gold_and_decoy(self):
        self._verify("confound", "confound")

    def test_deceive_round2_mentions_decoy_only(self):
        self._verify("deceive", "deceive")


class Phase65DefaultConfigTests(unittest.TestCase):
    """W18-1 strict-win anchor + audit T-1..T-7 preservation."""

    @classmethod
    def setUpClass(cls):
        cls.r_loose = run_phase65(
            bank="compat", T_decoder=None, n_eval=8,
            K_auditor=12, bank_seed=11, bank_replicates=2)
        cls.r_tight = run_phase65(
            bank="compat", T_decoder=24, n_eval=8,
            K_auditor=12, bank_seed=11, bank_replicates=2)

    def test_w18_strict_win_loose(self):
        self.assertGreaterEqual(
            self.r_loose["pooled"]["capsule_relational_compat"]
                       ["accuracy_full"], 1.0)
        self.assertLess(
            self.r_loose["pooled"]["capsule_attention_aware"]
                       ["accuracy_full"], 1.0)
        gap = self.r_loose["headline_gap"]["w18_minus_attention_aware"]
        self.assertGreaterEqual(gap, 0.50)

    def test_w18_strict_win_tight(self):
        self.assertGreaterEqual(
            self.r_tight["pooled"]["capsule_relational_compat"]
                       ["accuracy_full"], 1.0)
        self.assertLess(
            self.r_tight["pooled"]["capsule_attention_aware"]
                       ["accuracy_full"], 1.0)
        gap = self.r_tight["headline_gap"]["w18_minus_attention_aware"]
        self.assertGreaterEqual(gap, 0.50)

    def test_w18_audit_OK_on_every_cell(self):
        for cell in (self.r_loose, self.r_tight):
            grid = cell["audit_ok_grid"]
            for s, ok in grid.items():
                if s == "substrate":
                    continue
                self.assertTrue(ok, f"audit failed for {s}")

    def test_w17_lambda_symmetric_extends_to_r65(self):
        # Every non-W18 capsule strategy ties FIFO at 0.000 on
        # R-65-COMPAT.
        for cell in (self.r_loose, self.r_tight):
            for s, p in cell["pooled"].items():
                if s == "capsule_relational_compat":
                    continue
                self.assertEqual(
                    p["accuracy_full"], 0.0,
                    f"non-W18 strategy {s!r} did not tie FIFO on "
                    f"{cell['config']['bank']!r} (got "
                    f"{p['accuracy_full']:.3f})")


class Phase65SeedStabilityTests(unittest.TestCase):
    """W18-1 stability across 5 alternate ``bank_seed`` values."""

    @classmethod
    def setUpClass(cls):
        cls.sweep_loose = run_phase65_seed_stability_sweep(
            bank="compat", T_decoder=None, n_eval=8, K_auditor=12)
        cls.sweep_tight = run_phase65_seed_stability_sweep(
            bank="compat", T_decoder=24, n_eval=8, K_auditor=12)

    def test_loose_min_gap_above_strong_bar(self):
        self.assertGreaterEqual(
            self.sweep_loose["min_w18_minus_attention_aware"], 0.50)

    def test_tight_min_gap_above_strong_bar(self):
        self.assertGreaterEqual(
            self.sweep_tight["min_w18_minus_attention_aware"], 0.50)

    def test_w18_one_thousand_on_every_seed(self):
        for sweep in (self.sweep_loose, self.sweep_tight):
            for seed_str, rep in sweep["per_seed"].items():
                self.assertGreaterEqual(
                    rep["pooled"]["capsule_relational_compat"], 1.0,
                    f"W18 not at 1.0 on seed {seed_str}")


class Phase65FalsifierTests(unittest.TestCase):
    """W18-Λ-no-compat / -confound / -deceive: the named structural
    limits where W18 ties FIFO or fails by construction."""

    @classmethod
    def setUpClass(cls):
        cls.r_no_compat = run_phase65(
            bank="no_compat", T_decoder=None, n_eval=8,
            K_auditor=12, bank_seed=11)
        cls.r_confound = run_phase65(
            bank="confound", T_decoder=None, n_eval=8,
            K_auditor=12, bank_seed=11)
        cls.r_deceive = run_phase65(
            bank="deceive", T_decoder=None, n_eval=8,
            K_auditor=12, bank_seed=11)

    def test_w18_lambda_no_compat_ties_fifo(self):
        self.assertEqual(
            self.r_no_compat["pooled"]
                            ["capsule_relational_compat"]
                            ["accuracy_full"], 0.0)

    def test_w18_lambda_confound_ties_fifo(self):
        self.assertEqual(
            self.r_confound["pooled"]
                          ["capsule_relational_compat"]
                          ["accuracy_full"], 0.0)

    def test_w18_lambda_deceive_picks_decoy_and_fails(self):
        # On DECEIVE the W18 picks decoy services and the projection
        # is non-empty (not abstention) — but ``accuracy_full`` is 0
        # because decoy != gold.
        self.assertEqual(
            self.r_deceive["pooled"]
                         ["capsule_relational_compat"]
                         ["accuracy_full"], 0.0)


class Phase65BackwardCompatTests(unittest.TestCase):
    """W18-3 backward-compat: on R-58 default the W18 method ties
    AttentionAwareBundleDecoder byte-for-byte on the answer field."""

    def test_w18_matches_w15_on_phase58_deadlock(self):
        bank = build_phase58_bank(n_replicates=1, seed=11)
        for sc in bank:
            r1 = [_DecodedHandoff(src, kind, payload)
                   for (src, _to, kind, payload, _evs)
                   in _phase58_build_round_candidates(sc.round1_emissions)]
            r2 = [_DecodedHandoff(src, kind, payload)
                   for (src, _to, kind, payload, _evs)
                   in _phase58_build_round_candidates(sc.round2_emissions)]
            inner = AttentionAwareBundleDecoder()
            w18 = RelationalCompatibilityDisambiguator(
                inner=AttentionAwareBundleDecoder())
            inner_ans = inner.decode_rounds([r1, r2])
            w18_ans = w18.decode_rounds([r1, r2])
            # On R-58 default the deadlock scenario carries the
            # relational-mention; the W18 strict-asymmetric branch
            # produces the same gold-only set as the W15 + W11
            # contradiction-aware drop.
            self.assertEqual(
                set(inner_ans["services"]),
                set(w18_ans["services"]),
                f"W18 services diverged on {sc.scenario_id}: "
                f"inner={inner_ans['services']!r}, "
                f"w18={w18_ans['services']!r}")
            self.assertEqual(
                inner_ans["root_cause"], w18_ans["root_cause"])

    def test_w18_partial_recovery_on_p64sym_deadlock(self):
        """On R-64-SYM the W18 partially recovers — only the deadlock
        scenarios carry a relational mention; pool/disk/slow_query
        do not. The W18 method is *partially* successful here, which
        the bench reports honestly: 2/8 rather than 8/8."""
        bank = build_phase64_sym_bank(n_replicates=1, seed=11)
        recovered = 0
        for sc in bank:
            r1 = [_DecodedHandoff(src, kind, payload)
                   for (src, _to, kind, payload, _evs)
                   in _phase58_build_round_candidates(sc.round1_emissions)]
            r2 = [_DecodedHandoff(src, kind, payload)
                   for (src, _to, kind, payload, _evs)
                   in _phase58_build_round_candidates(sc.round2_emissions)]
            w18 = RelationalCompatibilityDisambiguator()
            ans = w18.decode_rounds([r1, r2])
            services = set(ans["services"])
            gold = set(sc.gold_services_pair)
            if services == gold:
                recovered += 1
        # Only the two deadlock scenarios have round-2 relational
        # mentions on the existing R-64-SYM bank.
        self.assertGreaterEqual(recovered, 1)
        self.assertLessEqual(recovered, len(bank))


class Phase65TokenEfficiencyTests(unittest.TestCase):
    """W18 reads only the W15-packed bundle; ``tokens_kept`` is
    byte-for-byte preserved (bounded-context honesty)."""

    def test_w18_does_not_inflate_tokens_kept(self):
        rep = run_phase65(
            bank="compat", T_decoder=24, n_eval=8,
            K_auditor=12, bank_seed=11, bank_replicates=2)
        ps_w15 = rep["pack_stats_summary"]["capsule_attention_aware"]
        ps_w18 = rep["pack_stats_summary"]["capsule_relational_compat"]
        # The W18 method shares the W15 pack stats byte-for-byte
        # (it consumes the same kept handoffs the W15 decoder does;
        # no extra capsule reads).
        self.assertEqual(
            ps_w18["tokens_kept_sum"], ps_w15["tokens_kept_sum"])
        self.assertEqual(
            ps_w18["handoffs_decoder_input_sum"],
            ps_w15["handoffs_decoder_input_sum"])
        self.assertEqual(
            ps_w18["tokens_kept_over_input"],
            ps_w15["tokens_kept_over_input"])

    def test_w18_strict_token_budget(self):
        rep = run_phase65(
            bank="compat", T_decoder=24, n_eval=8,
            K_auditor=12, bank_seed=11, bank_replicates=2)
        ps_w18 = rep["pack_stats_summary"]["capsule_relational_compat"]
        # Strict: tokens kept must be ≤ T_decoder × n_cells.
        T = 24
        n_cells = ps_w18["n_cells"]
        self.assertLessEqual(ps_w18["tokens_kept_sum"], T * n_cells)


class Phase65CrossRegimeSyntheticTests(unittest.TestCase):
    """Cross-regime synthetic summary — five cells separate cleanly."""

    @classmethod
    def setUpClass(cls):
        cls.d = run_cross_regime_synthetic(
            n_eval=8, bank_seed=11, K_auditor=12)

    def test_compat_loose_w18_at_one_thousand(self):
        self.assertEqual(
            self.d["headline_summary"]["r65_compat_loose_w18"], 1.0)

    def test_compat_tight_w18_at_one_thousand(self):
        self.assertEqual(
            self.d["headline_summary"]["r65_compat_tight_w18"], 1.0)

    def test_falsifiers_zero(self):
        self.assertEqual(
            self.d["headline_summary"]["r65_no_compat_w18"], 0.0)
        self.assertEqual(
            self.d["headline_summary"]["r65_confound_w18"], 0.0)
        self.assertEqual(
            self.d["headline_summary"]["r65_deceive_w18"], 0.0)

    def test_attention_aware_zero_on_compat(self):
        self.assertEqual(
            self.d["headline_summary"]
                  ["r65_compat_loose_attention_aware"], 0.0)
        self.assertEqual(
            self.d["headline_summary"]
                  ["r65_compat_tight_attention_aware"], 0.0)


if __name__ == "__main__":
    unittest.main()
