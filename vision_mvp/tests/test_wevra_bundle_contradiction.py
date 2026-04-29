"""Tests for SDK v3.20 — bundle-contradiction-aware trust-weighted
disambiguator + W19 family.

Covers:

* Helper-function unit tests: ``_w19_canonical_primary_index`` (canonical-
  role tiebreak), ``_w19_witness_counts`` (specific-tier witness counter
  excluding primary).
* W19 decoder unit tests: branch-by-branch determinism + abstention
  semantics (R-66 sub-banks).
* Phase-66 bench-property tests: every cell carries the named symmetric-
  corroboration ingredient AND the bank-specific (primary, secondary)
  shape.
* Phase-66 default-config tests: W19 strict win on R-66-DECEIVE-NAIVE
  (loose AND tight) AND R-66-CONFOUND-RESOLVABLE (W19-1).
* Phase-66 5-seed stability: gap w19 − w18 ≥ 0.50 on every seed under
  the deceive-naive AND confound-resolvable regimes.
* Phase-66 falsifier tests: W19 ties FIFO at 0.000 on R-66-DECEIVE-TOTAL
  (W19-Λ-total) AND R-66-OUTSIDE-REQUIRED (W19-Λ-outside).
* Phase-66 backward-compat smoke: W19 reduces to W18 byte-for-byte on
  R-58 default and on the R-65 default banks (R-65-COMPAT / R-65-DECEIVE
  / R-65-CONFOUND / R-65-NO-COMPAT).
* Phase-66 token-budget honesty: W19 reads only the W18-packed bundle
  (which itself reads only the W15-packed bundle); ``tokens_kept_sum``
  unchanged byte-for-byte from W18.
* Phase-66 audit T-1..T-7 preservation on every cell of every regime.
"""

from __future__ import annotations

import unittest

from vision_mvp.experiments.phase58_multi_round_decoder import (
    _build_round_candidates as _phase58_build_round_candidates,
    build_phase58_bank,
)
from vision_mvp.experiments.phase65_relational_disambiguation import (
    build_phase65_bank,
)
from vision_mvp.experiments.phase66_deceptive_ambiguity import (
    _bench_property_p66,
    build_phase66_bank,
    run_phase66,
    run_phase66_seed_stability_sweep,
    run_cross_regime_synthetic,
)
from vision_mvp.wevra.team_coord import (
    AttentionAwareBundleDecoder,
    BundleContradictionDisambiguator,
    RelationalCompatibilityDisambiguator,
    W18CompatibilityResult,
    W19TrustResult,
    W19_ALL_BRANCHES,
    W19_BRANCH_ABSTAINED_NO_SIGNAL,
    W19_BRANCH_ABSTAINED_SYMMETRIC,
    W19_BRANCH_CONFOUND_RESOLVED,
    W19_BRANCH_DISABLED,
    W19_BRANCH_INVERSION,
    W19_BRANCH_PRIMARY_TRUSTED,
    _DecodedHandoff,
    _INCIDENT_TRIAGE_CANONICAL_ROLE_FOR_KIND,
    _SPECIFIC_TIER_CLAIM_KINDS,
    _w19_canonical_primary_index,
    _w19_witness_counts,
)


def _r1_symmetric(A: str, B: str, decoy: str
                    ) -> list[_DecodedHandoff]:
    """Build the R-66 round-1 symmetric corroboration handoffs."""
    return [
        _DecodedHandoff("monitor", "LATENCY_SPIKE",
                         f"p95_ms=2200 service={A}"),
        _DecodedHandoff("monitor", "ERROR_RATE_SPIKE",
                         f"error_rate=0.20 service={B}"),
        _DecodedHandoff("monitor", "LATENCY_SPIKE",
                         f"p95_ms=2100 service={decoy}"),
        _DecodedHandoff("monitor", "ERROR_RATE_SPIKE",
                         f"error_rate=0.18 service={decoy}"),
        _DecodedHandoff("network", "FW_BLOCK_SURGE",
                         f"rule=deny count=11 service={A}"),
        _DecodedHandoff("network", "FW_BLOCK_SURGE",
                         f"rule=deny count=10 service={B}"),
        _DecodedHandoff("network", "FW_BLOCK_SURGE",
                         f"rule=deny count=9 service={decoy}"),
        _DecodedHandoff("network", "FW_BLOCK_SURGE",
                         f"rule=deny count=8 service={decoy}"),
    ]


class W19PrimaryIndexTests(unittest.TestCase):
    """``_w19_canonical_primary_index`` — canonical-role tiebreak."""

    def test_no_specific_tier_returns_minus_one(self):
        union = [
            _DecodedHandoff("monitor", "LATENCY_SPIKE",
                             "p95_ms=2100 service=cache"),
        ]
        round_hint = [1]
        self.assertEqual(
            _w19_canonical_primary_index(union, round_hint), -1)

    def test_single_specific_tier_returns_its_index(self):
        union = [
            _DecodedHandoff("monitor", "LATENCY_SPIKE",
                             "p95_ms=2100 service=cache"),
            _DecodedHandoff("db_admin", "DEADLOCK_SUSPECTED",
                             "deadlock relation=orders_payments_join"),
        ]
        round_hint = [1, 2]
        self.assertEqual(
            _w19_canonical_primary_index(union, round_hint), 1)

    def test_canonical_role_match_preferred(self):
        # Two specific-tier handoffs in round 2: db_admin is the
        # canonical role for DEADLOCK_SUSPECTED; monitor is not.
        # Even though monitor < db_admin lex, the canonical-role
        # tiebreak picks db_admin.
        union = [
            _DecodedHandoff("monitor", "DEADLOCK_SUSPECTED",
                             "deadlock relation=cache_cache_join"),
            _DecodedHandoff("db_admin", "DEADLOCK_SUSPECTED",
                             "deadlock relation=orders_payments_join"),
        ]
        round_hint = [2, 2]
        self.assertEqual(
            _w19_canonical_primary_index(union, round_hint), 1)

    def test_higher_round_preferred(self):
        # Round-2 specific-tier sorts before round-1 specific-tier.
        union = [
            _DecodedHandoff("db_admin", "DEADLOCK_SUSPECTED",
                             "deadlock relation=orders_payments_join"),
            _DecodedHandoff("monitor", "DEADLOCK_SUSPECTED",
                             "deadlock relation=cache_cache_join"),
        ]
        round_hint = [1, 2]
        # The monitor handoff (round 2) sorts before the db_admin
        # handoff (round 1) on the ``-ridx`` axis.
        self.assertEqual(
            _w19_canonical_primary_index(union, round_hint), 1)

    def test_canonical_role_table_completeness(self):
        # Every entry in the canonical-role table must reference a
        # kind that's actually in the specific-tier set.
        for kind in _INCIDENT_TRIAGE_CANONICAL_ROLE_FOR_KIND:
            self.assertIn(kind, _SPECIFIC_TIER_CLAIM_KINDS,
                            f"canonical-role table mentions {kind!r} "
                            f"but it's not in _SPECIFIC_TIER_CLAIM_KINDS")


class W19WitnessCountsTests(unittest.TestCase):
    """``_w19_witness_counts`` — specific-tier witness counter."""

    def test_excludes_primary(self):
        # Note: ``_w19_witness_counts`` only counts handoffs whose
        # ``claim_kind`` is in :data:`_SPECIFIC_TIER_CLAIM_KINDS` (the
        # canonical set). The full W19 pipeline normalises synonyms
        # to canonical via the W12 / W13 layered normaliser before
        # calling this helper; this unit test passes already-canonical
        # kinds to mirror that contract.
        union = [
            _DecodedHandoff("db_admin", "DEADLOCK_SUSPECTED",
                             "deadlock relation=cache_cache_join"),
            _DecodedHandoff("monitor", "DEADLOCK_SUSPECTED",
                             "deadlock relation=orders_payments_join"),
        ]
        # Pretend index 0 (db_admin) is primary.
        aw = _w19_witness_counts(
            union, primary_index=0,
            admitted_tags=["orders", "payments", "cache"])
        # Secondary (monitor) mentions orders + payments → 1 each.
        self.assertEqual(aw["orders"], 1)
        self.assertEqual(aw["payments"], 1)
        self.assertEqual(aw["cache"], 0)

    def test_no_witnesses_returns_zero(self):
        union = [
            _DecodedHandoff("db_admin", "DEADLOCK_SUSPECTED",
                             "deadlock relation=orders_payments_join"),
        ]
        aw = _w19_witness_counts(
            union, primary_index=0,
            admitted_tags=["orders", "payments", "cache"])
        self.assertEqual(aw["orders"], 0)
        self.assertEqual(aw["payments"], 0)
        self.assertEqual(aw["cache"], 0)

    def test_round1_noise_kinds_excluded(self):
        # Round-1 LATENCY_SPIKE / ERROR_RATE_SPIKE / FW_BLOCK_SURGE
        # are NOT in _SPECIFIC_TIER_CLAIM_KINDS, so they don't
        # count as asymmetric witnesses regardless of payload.
        union = [
            _DecodedHandoff("db_admin", "DEADLOCK_SUSPECTED",
                             "deadlock relation=cache_cache_join"),
            _DecodedHandoff("monitor", "LATENCY_SPIKE",
                             "p95_ms=2200 service=orders"),
            _DecodedHandoff("monitor", "LATENCY_SPIKE",
                             "p95_ms=2200 service=payments"),
        ]
        aw = _w19_witness_counts(
            union, primary_index=0,
            admitted_tags=["orders", "payments", "cache"])
        self.assertEqual(aw["orders"], 0)
        self.assertEqual(aw["payments"], 0)
        self.assertEqual(aw["cache"], 0)

    def test_dedup_byte_identical_witnesses(self):
        # Two byte-identical secondary handoffs collapse to one
        # witness count.
        union = [
            _DecodedHandoff("db_admin", "DEADLOCK_SUSPECTED",
                             "deadlock relation=cache_cache_join"),
            _DecodedHandoff("monitor", "DEADLOCK_SUSPECTED",
                             "deadlock relation=orders_payments_join"),
            _DecodedHandoff("monitor", "DEADLOCK_SUSPECTED",
                             "deadlock relation=orders_payments_join"),
        ]
        aw = _w19_witness_counts(
            union, primary_index=0,
            admitted_tags=["orders", "payments", "cache"])
        self.assertEqual(aw["orders"], 1)
        self.assertEqual(aw["payments"], 1)
        self.assertEqual(aw["cache"], 0)


class W19DecoderUnitTests(unittest.TestCase):
    """``BundleContradictionDisambiguator`` — branch-by-branch
    determinism + abstention semantics."""

    def _scenario(self, gold_a: str, gold_b: str, decoy: str,
                   primary_payload: str, secondary_payload: str | None,
                   primary_role: str = "db_admin",
                   primary_kind: str = "DEADLOCK_SUSPECTED",
                   secondary_role: str = "monitor",
                   secondary_kind: str = "DEADLOCK_DETECTED"):
        r1 = _r1_symmetric(gold_a, gold_b, decoy)
        r2 = [
            _DecodedHandoff(primary_role, primary_kind, primary_payload),
        ]
        if secondary_payload is not None:
            r2.append(_DecodedHandoff(secondary_role, secondary_kind,
                                        secondary_payload))
        return r1, r2

    def test_corroborated_primary_trusted_branch(self):
        # Primary names gold; secondary also names gold. Witness for
        # gold > 0; W18 strict-asymmetric branch fires. W19 confirms
        # (no inversion needed).
        r1, r2 = self._scenario(
            "orders", "payments", "cache",
            "deadlock relation=orders_payments_join wait_chain=2",
            "deadlock relation=orders_payments_join detected_at=t=120")
        w19 = BundleContradictionDisambiguator()
        ans = w19.decode_rounds([r1, r2])
        self.assertEqual(set(ans["services"]), {"orders", "payments"})
        self.assertEqual(
            ans["trust"]["decoder_branch"],
            W19_BRANCH_PRIMARY_TRUSTED)

    def test_deceive_naive_confound_resolved_branch(self):
        # Primary names decoy; secondary names gold. W18's full-text
        # scorer sees positive hits on every tag → abstain.
        # W19's witness counter (excluding primary) sees aw(gold) > 0,
        # aw(decoy) = 0 → confound-resolved branch picks gold.
        r1, r2 = self._scenario(
            "orders", "payments", "cache",
            "deadlock relation=cache_cache_join wait_chain=2",
            "deadlock relation=orders_payments_join detected_at=t=120")
        w19 = BundleContradictionDisambiguator()
        ans = w19.decode_rounds([r1, r2])
        self.assertEqual(set(ans["services"]), {"orders", "payments"})
        self.assertEqual(
            ans["trust"]["decoder_branch"],
            W19_BRANCH_CONFOUND_RESOLVED)

    def test_confound_resolvable_confound_resolved_branch(self):
        # Primary names all three; secondary names gold only. W18
        # abstains (full-set hit). W19 picks strict-max-aw subset = gold.
        r1, r2 = self._scenario(
            "orders", "payments", "cache",
            "deadlock relation=orders_payments_cache_join wait_chain=2",
            "deadlock relation=orders_payments_join detected_at=t=120")
        w19 = BundleContradictionDisambiguator()
        ans = w19.decode_rounds([r1, r2])
        self.assertEqual(set(ans["services"]), {"orders", "payments"})
        self.assertEqual(
            ans["trust"]["decoder_branch"],
            W19_BRANCH_CONFOUND_RESOLVED)

    def test_deceive_total_no_witness_falls_through_to_w18(self):
        # Primary names decoy; NO secondary. aw uniform 0; W19 cannot
        # fire inversion AND cannot fire confound. Falls through to
        # W18's strict-asymmetric pick of decoy.
        r1, r2 = self._scenario(
            "orders", "payments", "cache",
            "deadlock relation=cache_cache_join wait_chain=2",
            None)
        w19 = BundleContradictionDisambiguator()
        ans = w19.decode_rounds([r1, r2])
        # W19 follows W18 here — picks decoy (the named falsifier).
        self.assertEqual(set(ans["services"]), {"cache"})
        self.assertEqual(
            ans["trust"]["decoder_branch"],
            W19_BRANCH_PRIMARY_TRUSTED)

    def test_outside_required_symmetric_witness_abstains(self):
        # Primary names decoy; secondary names ALL three. aw uniform
        # across all admitted tags → W19 cannot find a strict-max
        # subset → abstain (W19-Λ-outside).
        r1, r2 = self._scenario(
            "orders", "payments", "cache",
            "deadlock relation=cache_cache_join wait_chain=2",
            "deadlock relation=orders_payments_cache_join detected_at=t=120")
        w19 = BundleContradictionDisambiguator()
        ans = w19.decode_rounds([r1, r2])
        # W18 abstains AND W19 abstains.
        self.assertEqual(
            ans["trust"]["decoder_branch"],
            W19_BRANCH_ABSTAINED_SYMMETRIC)

    def test_no_specific_tier_in_bundle_abstains(self):
        # Round 1 only — no specific-tier disambiguator anywhere.
        r1 = _r1_symmetric("orders", "payments", "cache")
        w19 = BundleContradictionDisambiguator()
        ans = w19.decode_rounds([r1, []])
        self.assertEqual(
            ans["trust"]["decoder_branch"],
            W19_BRANCH_ABSTAINED_NO_SIGNAL)

    def test_determinism(self):
        r1, r2 = self._scenario(
            "orders", "payments", "cache",
            "deadlock relation=cache_cache_join",
            "deadlock relation=orders_payments_join")
        a = BundleContradictionDisambiguator().decode_rounds([r1, r2])
        b = BundleContradictionDisambiguator().decode_rounds([r1, r2])
        self.assertEqual(a["root_cause"], b["root_cause"])
        self.assertEqual(a["services"], b["services"])
        self.assertEqual(
            a["trust"]["decoder_branch"], b["trust"]["decoder_branch"])
        self.assertEqual(
            a["trust"]["per_tag_witness_count"],
            b["trust"]["per_tag_witness_count"])

    def test_disabled_path_reduces_to_inner_w18(self):
        r1, r2 = self._scenario(
            "orders", "payments", "cache",
            "deadlock relation=cache_cache_join",
            "deadlock relation=orders_payments_join")
        inner = RelationalCompatibilityDisambiguator()
        w19 = BundleContradictionDisambiguator(
            inner=RelationalCompatibilityDisambiguator(),
            enabled=False)
        a = w19.decode_rounds([r1, r2])
        b = inner.decode_rounds([r1, r2])
        # When disabled, W19 returns the inner W18's answer.
        self.assertEqual(a["root_cause"], b["root_cause"])
        self.assertEqual(a["services"], b["services"])
        self.assertEqual(
            a["trust"]["decoder_branch"], W19_BRANCH_DISABLED)

    def test_branches_are_well_typed(self):
        # Every branch the decoder can return must be in the
        # registered set.
        r1 = _r1_symmetric("orders", "payments", "cache")
        for r2_payload, secondary in (
                ("deadlock relation=orders_payments_join", None),
                ("deadlock relation=cache_cache_join", None),
                ("deadlock relation=orders_payments_cache_join", None),
                ("deadlock wait_chain=2", None),
                ("deadlock relation=cache_cache_join",
                 "deadlock relation=orders_payments_join"),
        ):
            r2 = [_DecodedHandoff("db_admin", "DEADLOCK_SUSPECTED",
                                    r2_payload)]
            if secondary is not None:
                r2.append(_DecodedHandoff(
                    "monitor", "DEADLOCK_DETECTED", secondary))
            ans = BundleContradictionDisambiguator().decode_rounds(
                [r1, r2])
            self.assertIn(
                ans["trust"]["decoder_branch"], W19_ALL_BRANCHES)


class Phase66BenchPropertyTests(unittest.TestCase):
    """Mechanically-verified bench property witnesses for every R-66
    sub-bank: symmetric corroboration AND named (primary, secondary)
    shape."""

    def _verify(self, bank: str, expected_shape: tuple[str, str]):
        scenarios = build_phase66_bank(
            bank=bank, n_replicates=2, seed=11)
        self.assertEqual(len(scenarios), 8)
        from vision_mvp.experiments.phase66_deceptive_ambiguity import (
            _build_round_candidates_p66,
        )
        for sc in scenarios:
            r1 = _build_round_candidates_p66(sc.round1_emissions)
            r2 = _build_round_candidates_p66(sc.round2_emissions)
            bp = _bench_property_p66(sc, r1, r2)
            self.assertTrue(bp["symmetric_corroboration_holds"],
                              f"sym fails on {sc.scenario_id}")
            self.assertEqual(tuple(bp["shape"]), expected_shape,
                              f"shape mismatch on {sc.scenario_id}: "
                              f"got {tuple(bp['shape'])!r}, "
                              f"expected {expected_shape!r}")

    def test_corroborated_shape(self):
        self._verify("corroborated", ("gold_only", "gold_only"))

    def test_deceive_naive_shape(self):
        self._verify("deceive_naive", ("decoy_only", "gold_only"))

    def test_confound_resolvable_shape(self):
        self._verify("confound_resolvable", ("all_three", "gold_only"))

    def test_deceive_total_shape(self):
        self._verify("deceive_total", ("decoy_only", "absent"))

    def test_outside_required_shape(self):
        self._verify("outside_required", ("decoy_only", "all_three"))


class Phase66DefaultConfigTests(unittest.TestCase):
    """W19-1 strict-win anchor + audit T-1..T-7 preservation."""

    @classmethod
    def setUpClass(cls):
        cls.r_corr = run_phase66(
            bank="corroborated", T_decoder=None, n_eval=8,
            K_auditor=12, bank_seed=11, bank_replicates=2)
        cls.r_deceive_loose = run_phase66(
            bank="deceive_naive", T_decoder=None, n_eval=8,
            K_auditor=12, bank_seed=11, bank_replicates=2)
        cls.r_deceive_tight = run_phase66(
            bank="deceive_naive", T_decoder=24, n_eval=8,
            K_auditor=12, bank_seed=11, bank_replicates=2)
        cls.r_confound = run_phase66(
            bank="confound_resolvable", T_decoder=None, n_eval=8,
            K_auditor=12, bank_seed=11, bank_replicates=2)

    def test_w19_strict_win_deceive_naive_loose(self):
        rep = self.r_deceive_loose
        self.assertGreaterEqual(
            rep["pooled"]["capsule_bundle_contradiction"]
                ["accuracy_full"], 1.0)
        self.assertLess(
            rep["pooled"]["capsule_relational_compat"]
                ["accuracy_full"], 1.0)
        gap = rep["headline_gap"]["w19_minus_w18"]
        self.assertGreaterEqual(gap, 0.50)

    def test_w19_strict_win_deceive_naive_tight(self):
        rep = self.r_deceive_tight
        self.assertGreaterEqual(
            rep["pooled"]["capsule_bundle_contradiction"]
                ["accuracy_full"], 1.0)
        self.assertLess(
            rep["pooled"]["capsule_relational_compat"]
                ["accuracy_full"], 1.0)
        gap = rep["headline_gap"]["w19_minus_w18"]
        self.assertGreaterEqual(gap, 0.50)

    def test_w19_strict_win_confound_resolvable(self):
        rep = self.r_confound
        self.assertGreaterEqual(
            rep["pooled"]["capsule_bundle_contradiction"]
                ["accuracy_full"], 1.0)
        self.assertLess(
            rep["pooled"]["capsule_relational_compat"]
                ["accuracy_full"], 1.0)
        gap = rep["headline_gap"]["w19_minus_w18"]
        self.assertGreaterEqual(gap, 0.50)

    def test_w19_ratifies_w18_on_corroborated(self):
        # On the positive anchor where both W18 and W19 should hit
        # 1.000, W19 must not regress and audit must hold.
        rep = self.r_corr
        self.assertGreaterEqual(
            rep["pooled"]["capsule_bundle_contradiction"]
                ["accuracy_full"], 1.0)
        self.assertGreaterEqual(
            rep["pooled"]["capsule_relational_compat"]
                ["accuracy_full"], 1.0)

    def test_w19_audit_OK_on_every_cell(self):
        for cell in (self.r_corr, self.r_deceive_loose,
                       self.r_deceive_tight, self.r_confound):
            grid = cell["audit_ok_grid"]
            for s, ok in grid.items():
                if s == "substrate":
                    continue
                self.assertTrue(
                    ok, f"audit failed for {s} on bank "
                          f"{cell['config']['bank']!r}")


class Phase66SeedStabilityTests(unittest.TestCase):
    """W19-1 stability across 5 alternate ``bank_seed`` values."""

    @classmethod
    def setUpClass(cls):
        cls.sweep_deceive_loose = run_phase66_seed_stability_sweep(
            bank="deceive_naive", T_decoder=None,
            n_eval=8, K_auditor=12)
        cls.sweep_deceive_tight = run_phase66_seed_stability_sweep(
            bank="deceive_naive", T_decoder=24,
            n_eval=8, K_auditor=12)
        cls.sweep_confound = run_phase66_seed_stability_sweep(
            bank="confound_resolvable", T_decoder=None,
            n_eval=8, K_auditor=12)

    def test_deceive_loose_min_gap_above_strong_bar(self):
        self.assertGreaterEqual(
            self.sweep_deceive_loose["min_w19_minus_w18"], 0.50)

    def test_deceive_tight_min_gap_above_strong_bar(self):
        self.assertGreaterEqual(
            self.sweep_deceive_tight["min_w19_minus_w18"], 0.50)

    def test_confound_min_gap_above_strong_bar(self):
        self.assertGreaterEqual(
            self.sweep_confound["min_w19_minus_w18"], 0.50)

    def test_w19_one_thousand_on_every_seed(self):
        for sweep in (self.sweep_deceive_loose,
                       self.sweep_deceive_tight,
                       self.sweep_confound):
            for seed_str, rep in sweep["per_seed"].items():
                self.assertGreaterEqual(
                    rep["pooled"]["capsule_bundle_contradiction"], 1.0,
                    f"W19 not at 1.0 on seed {seed_str} of "
                    f"{sweep['config']['bank']!r}")


class Phase66FalsifierTests(unittest.TestCase):
    """W19-Λ-total / W19-Λ-outside: the named structural limits where
    W19 ties FIFO by construction."""

    @classmethod
    def setUpClass(cls):
        cls.r_total = run_phase66(
            bank="deceive_total", T_decoder=None, n_eval=8,
            K_auditor=12, bank_seed=11)
        cls.r_outside = run_phase66(
            bank="outside_required", T_decoder=None, n_eval=8,
            K_auditor=12, bank_seed=11)

    def test_w19_lambda_total_ties_fifo(self):
        # No witnesses → W19 reduces to W18; primary names decoy →
        # picks decoy → fails.
        self.assertEqual(
            self.r_total["pooled"]
                       ["capsule_bundle_contradiction"]
                       ["accuracy_full"], 0.0)
        # The branch must be primary-trusted (W19 fell through to W18).
        branches = self.r_total["w19_branch_counts"]
        self.assertGreater(
            branches.get(W19_BRANCH_PRIMARY_TRUSTED, 0), 0)

    def test_w19_lambda_outside_ties_fifo_via_abstention(self):
        # Symmetric witnesses → W19 abstains (refuses to choose).
        self.assertEqual(
            self.r_outside["pooled"]
                         ["capsule_bundle_contradiction"]
                         ["accuracy_full"], 0.0)
        # The branch must be abstained-symmetric.
        branches = self.r_outside["w19_branch_counts"]
        self.assertGreater(
            branches.get(W19_BRANCH_ABSTAINED_SYMMETRIC, 0), 0)


class Phase66BackwardCompatTests(unittest.TestCase):
    """W19-3 backward-compat: on R-58 default + every R-65 default
    bank, W19 ties W18 byte-for-byte on the answer field."""

    def _ties_per_scenario(self, r1_handoffs, r2_handoffs):
        w18 = RelationalCompatibilityDisambiguator()
        w19 = BundleContradictionDisambiguator()
        a18 = w18.decode_rounds([r1_handoffs, r2_handoffs])
        a19 = w19.decode_rounds([r1_handoffs, r2_handoffs])
        return (set(a18["services"]), set(a19["services"]),
                a18["root_cause"], a19["root_cause"])

    def test_w19_matches_w18_on_phase58_default(self):
        bank = build_phase58_bank(n_replicates=1, seed=11)
        for sc in bank:
            r1 = [_DecodedHandoff(src, kind, payload)
                   for (src, _to, kind, payload, _evs)
                   in _phase58_build_round_candidates(sc.round1_emissions)]
            r2 = [_DecodedHandoff(src, kind, payload)
                   for (src, _to, kind, payload, _evs)
                   in _phase58_build_round_candidates(sc.round2_emissions)]
            s18, s19, r18, r19 = self._ties_per_scenario(r1, r2)
            self.assertEqual(
                s18, s19,
                f"W19 services diverged on R-58 {sc.scenario_id}: "
                f"w18={s18!r}, w19={s19!r}")
            self.assertEqual(r18, r19)

    def test_w19_matches_w18_on_phase65_default_banks(self):
        for bank_name in ("compat", "no_compat", "confound", "deceive"):
            bank = build_phase65_bank(
                bank=bank_name, n_replicates=1, seed=11)
            for sc in bank:
                r1 = [_DecodedHandoff(src, kind, payload)
                       for (src, _to, kind, payload, _evs)
                       in _phase58_build_round_candidates(
                            sc.round1_emissions)]
                r2 = [_DecodedHandoff(src, kind, payload)
                       for (src, _to, kind, payload, _evs)
                       in _phase58_build_round_candidates(
                            sc.round2_emissions)]
                s18, s19, r18, r19 = self._ties_per_scenario(r1, r2)
                self.assertEqual(
                    s18, s19,
                    f"W19 services diverged on R-65-{bank_name} "
                    f"{sc.scenario_id}: w18={s18!r}, w19={s19!r}")
                self.assertEqual(r18, r19)


class Phase66TokenEfficiencyTests(unittest.TestCase):
    """W19 reads only the W18-packed bundle (which itself reads only
    the W15-packed bundle); ``tokens_kept`` is byte-for-byte preserved
    (bounded-context honesty)."""

    def test_w19_does_not_inflate_tokens_kept(self):
        rep = run_phase66(
            bank="deceive_naive", T_decoder=24, n_eval=8,
            K_auditor=12, bank_seed=11, bank_replicates=2)
        ps_w18 = rep["pack_stats_summary"]["capsule_relational_compat"]
        ps_w19 = rep["pack_stats_summary"]["capsule_bundle_contradiction"]
        # The W19 method shares the W18 pack stats byte-for-byte
        # (it consumes the same kept handoffs the W18 decoder does;
        # no extra capsule reads).
        self.assertEqual(
            ps_w19["tokens_kept_sum"], ps_w18["tokens_kept_sum"])
        self.assertEqual(
            ps_w19["handoffs_decoder_input_sum"],
            ps_w18["handoffs_decoder_input_sum"])
        self.assertEqual(
            ps_w19["tokens_kept_over_input"],
            ps_w18["tokens_kept_over_input"])

    def test_w19_strict_token_budget(self):
        rep = run_phase66(
            bank="deceive_naive", T_decoder=24, n_eval=8,
            K_auditor=12, bank_seed=11, bank_replicates=2)
        ps_w19 = rep["pack_stats_summary"]["capsule_bundle_contradiction"]
        # Strict: tokens kept must be ≤ T_decoder × n_cells.
        T = 24
        n_cells = ps_w19["n_cells"]
        self.assertLessEqual(ps_w19["tokens_kept_sum"], T * n_cells)


class Phase66CrossRegimeSyntheticTests(unittest.TestCase):
    """Cross-regime synthetic summary — six cells separate cleanly."""

    @classmethod
    def setUpClass(cls):
        cls.d = run_cross_regime_synthetic(
            n_eval=8, bank_seed=11, K_auditor=12)

    def test_corroborated_w19_at_one_thousand(self):
        self.assertEqual(
            self.d["headline_summary"]["r66_corroborated_w19"], 1.0)
        self.assertEqual(
            self.d["headline_summary"]["r66_corroborated_w18"], 1.0)

    def test_deceive_naive_loose_w19_at_one_thousand(self):
        self.assertEqual(
            self.d["headline_summary"]
                  ["r66_deceive_naive_loose_w19"], 1.0)
        self.assertEqual(
            self.d["headline_summary"]
                  ["r66_deceive_naive_loose_w18"], 0.0)

    def test_deceive_naive_tight_w19_at_one_thousand(self):
        self.assertEqual(
            self.d["headline_summary"]
                  ["r66_deceive_naive_tight_w19"], 1.0)
        self.assertEqual(
            self.d["headline_summary"]
                  ["r66_deceive_naive_tight_w18"], 0.0)

    def test_confound_resolvable_w19_at_one_thousand(self):
        self.assertEqual(
            self.d["headline_summary"]
                  ["r66_confound_resolvable_w19"], 1.0)
        self.assertEqual(
            self.d["headline_summary"]
                  ["r66_confound_resolvable_w18"], 0.0)

    def test_w19_lambda_total_zero(self):
        self.assertEqual(
            self.d["headline_summary"]["r66_deceive_total_w19"], 0.0)

    def test_w19_lambda_outside_zero(self):
        self.assertEqual(
            self.d["headline_summary"]["r66_outside_required_w19"], 0.0)

    def test_strict_gain_summary(self):
        for k in ("w19_minus_w18_deceive_naive_loose",
                   "w19_minus_w18_deceive_naive_tight",
                   "w19_minus_w18_confound_resolvable"):
            self.assertGreaterEqual(
                self.d["headline_summary"][k], 0.50,
                f"{k} below strong-bar threshold")


if __name__ == "__main__":
    unittest.main()
