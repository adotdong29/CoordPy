"""Tests for SDK v3.17 end-to-end W14+W15 composition + W16 family.

Covers:

* Replay extractor — :class:`OllamaReplayExtractor` parses recorded
  Phase-61 raw responses byte-for-byte and re-routes through the
  same role subscriptions; counters and fall-through are exact
  (W16-3 backward-compat anchor on the recorded LLM bytes).
* Phase-63 default config tests — pre-committed cell-by-cell
  ``accuracy_full`` and the W16-1 strict-gain anchor on the
  composed regime.
* Phase-63 5-seed stability — gap composed − fifo_pack ≥ 0.50 on
  every seed under structured prompt + tight T_decoder.
* Phase-63 cross-regime — W14-Λ × W15-Λ joint failure on the
  naive-tight cell, W14-only success on the loose cell, W16-1
  strict gain on the composed-tight cell, W16-Λ-degenerate on
  T_decoder=2.
* Phase-63 ollama-replay — strict +0.50 W14+W15 composition gain on
  the recorded qwen2.5:14b-32k bytes at T_decoder ∈ [13, 16] band.

The replay tests load
``docs/data/phase61_real_ollama_structured_qwen2_5_14b_n8.json``
(structured prompt, n=8 saturated, the W14-Λ-real anchor) and
``docs/data/phase61_real_ollama_naive_qwen2_5_14b_n8.json`` (naive
prompt, the W14-Λ-real falsifier). Both files were captured in the
SDK v3.15 milestone with 0 endpoint failures and the resulting raw
responses are byte-stable.
"""

from __future__ import annotations

import os
import unittest

from vision_mvp.experiments.phase63_composed_real_llm import (
    OllamaReplayExtractor, run_cross_regime_summary, run_phase63,
    run_phase63_seed_stability_sweep,
)
from vision_mvp.wevra.team_coord import (
    PRODUCER_PROMPT_NAIVE, PRODUCER_PROMPT_STRUCTURED,
)

# Repo-relative paths to recorded Phase-61 real-Ollama bytes.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_REPLAY_STRUCT = os.path.join(
    _REPO_ROOT, "docs", "data",
    "phase61_real_ollama_structured_qwen2_5_14b_n8.json")
_REPLAY_NAIVE = os.path.join(
    _REPO_ROOT, "docs", "data",
    "phase61_real_ollama_naive_qwen2_5_14b_n8.json")


class OllamaReplayExtractorTests(unittest.TestCase):
    """Replay extractor unit tests — byte-for-byte determinism over
    recorded Phase-61 LLM bytes."""

    def test_replay_loads_structured_capture(self):
        ext = OllamaReplayExtractor.from_phase61_report(_REPLAY_STRUCT)
        self.assertEqual(ext.prompt_mode, PRODUCER_PROMPT_STRUCTURED)
        self.assertEqual(len(ext.replay_scenarios()), 8)

    def test_replay_loads_naive_capture(self):
        ext = OllamaReplayExtractor.from_phase61_report(_REPLAY_NAIVE)
        self.assertEqual(ext.prompt_mode, PRODUCER_PROMPT_NAIVE)
        self.assertEqual(len(ext.replay_scenarios()), 8)

    def test_replay_rejects_missing_raw_block(self):
        # phase62 cross-regime has no raw_responses_per_scenario
        bad_path = os.path.join(_REPO_ROOT, "docs", "data",
                                "phase62_cross_regime.json")
        if not os.path.exists(bad_path):
            self.skipTest("phase62 cross-regime not present")
        with self.assertRaises(ValueError):
            OllamaReplayExtractor.from_phase61_report(bad_path)


class Phase63ComposedTightTests(unittest.TestCase):
    """W16-1 anchor — magnitude-filter structured + T_decoder=24,
    K_auditor=12, n_eval=8, bank_seed=11.
    """

    @classmethod
    def setUpClass(cls):
        cls.r_composed = run_phase63(
            n_eval=8, K_auditor=12, T_auditor=256,
            bank_seed=11, bank_replicates=2,
            T_decoder=24,
            extractor="magnitude_filter",
            prompt_mode=PRODUCER_PROMPT_STRUCTURED,
            verbose=False)
        cls.r_baseline = run_phase63(
            n_eval=8, K_auditor=12, T_auditor=256,
            bank_seed=11, bank_replicates=2,
            T_decoder=None,
            extractor="identity",
            prompt_mode=PRODUCER_PROMPT_NAIVE,
            verbose=False)
        cls.r_naive_tight = run_phase63(
            n_eval=8, K_auditor=12, T_auditor=256,
            bank_seed=11, bank_replicates=2,
            T_decoder=24,
            extractor="magnitude_filter",
            prompt_mode=PRODUCER_PROMPT_NAIVE,
            verbose=False)
        cls.r_w14_only = run_phase63(
            n_eval=8, K_auditor=12, T_auditor=256,
            bank_seed=11, bank_replicates=2,
            T_decoder=None,
            extractor="magnitude_filter",
            prompt_mode=PRODUCER_PROMPT_NAIVE,
            verbose=False)
        cls.r_w14_success = run_phase63(
            n_eval=8, K_auditor=12, T_auditor=256,
            bank_seed=11, bank_replicates=2,
            T_decoder=None,
            extractor="magnitude_filter",
            prompt_mode=PRODUCER_PROMPT_STRUCTURED,
            verbose=False)
        cls.r_degen_budget = run_phase63(
            n_eval=8, K_auditor=12, T_auditor=256,
            bank_seed=11, bank_replicates=2,
            T_decoder=2,
            extractor="magnitude_filter",
            prompt_mode=PRODUCER_PROMPT_STRUCTURED,
            verbose=False)

    def _gap(self, rep, a, b):
        return (rep["pooled"][a]["accuracy_full"]
                - rep["pooled"][b]["accuracy_full"])

    def test_composed_strict_win_over_fifo_pack(self):
        """W16-1: composed (W14+W15) strictly beats W14-only-budgeted
        (FIFO pack of structured-prompt stream) by ≥ 0.50."""
        gap = self._gap(self.r_composed,
                          "capsule_attention_aware",
                          "capsule_layered_fifo_packed")
        self.assertGreaterEqual(gap, 0.50,
            msg=f"W16-1: composed - fifo_packed_layered = {gap}; "
                 f"expected ≥ 0.50")

    def test_composed_strict_win_over_substrate_fifo(self):
        gap = self._gap(self.r_composed,
                          "capsule_attention_aware",
                          "capsule_fifo")
        self.assertGreaterEqual(gap, 0.50,
            msg=f"W16-1: composed - capsule_fifo = {gap}; "
                 f"expected ≥ 0.50")

    def test_composed_property_holds_8_of_8(self):
        """Under structured prompt the magnitude-filter producer
        preserves the bench property; W16-1 premise."""
        self.assertEqual(
            self.r_composed["bench_summary"]
                ["scenarios_with_property"], 8)

    def test_composed_keeps_round2_specific_claim(self):
        """The composed packer puts the round-2 specific claim at
        rank 0 in 8/8 cells."""
        ps = (self.r_composed["pack_stats_summary"]
              ["capsule_attention_aware"])
        self.assertEqual(ps["position_of_first_causal_claim_avg"], 0.0)
        self.assertEqual(ps["n_with_causal_claim_kept"], 8)

    def test_composed_fifo_pack_drops_round2_specific_claim(self):
        """The W14-only-budgeted baseline (FIFO pack) drops the
        disambiguator in 8/8 cells."""
        ps = (self.r_composed["pack_stats_summary"]
              ["capsule_layered_fifo_packed"])
        self.assertEqual(ps["n_with_causal_claim_kept"], 0)
        self.assertEqual(ps["position_of_first_causal_claim_avg"], -1.0)

    def test_baseline_loose_w15_ties_layered(self):
        """W16-3 backward-compat: with no producer compression and no
        budget pressure, composed = unpacked layered."""
        self.assertEqual(
            self.r_baseline["pooled"]["capsule_attention_aware"]
                ["accuracy_full"], 1.000)
        self.assertEqual(
            self.r_baseline["pooled"]["capsule_layered_multi_round"]
                ["accuracy_full"], 1.000)

    def test_w14_only_failure_naive_no_budget(self):
        """W14-Λ-prompt extends to R-62 multi-hypothesis events: under
        producer compression alone, every cross-round decoder fails."""
        self.assertEqual(
            self.r_w14_only["pooled"]["capsule_attention_aware"]
                ["accuracy_full"], 0.000)
        self.assertEqual(
            self.r_w14_only["pooled"]["capsule_layered_multi_round"]
                ["accuracy_full"], 0.000)
        self.assertEqual(
            self.r_w14_only["bench_summary"]
                ["scenarios_with_property"], 0)

    def test_naive_tight_w16_lambda_compose(self):
        """W16-Λ-compose: naive prompt + tight budget — every method
        ties FIFO at 0.000. Joint failure of W14-Λ-prompt and
        W15-Λ-budget on the same regime."""
        for s in ("capsule_fifo", "capsule_multi_round",
                   "capsule_robust_multi_round",
                   "capsule_layered_multi_round",
                   "capsule_layered_fifo_packed",
                   "capsule_attention_aware"):
            self.assertEqual(
                self.r_naive_tight["pooled"][s]["accuracy_full"], 0.000,
                msg=f"naive_tight: {s} = "
                     f"{self.r_naive_tight['pooled'][s]['accuracy_full']}; "
                     f"expected 0.000")

    def test_w14_success_loose_budget(self):
        """W14 alone restores the bench property under loose budget;
        unpacked layered = 1.000."""
        self.assertEqual(
            self.r_w14_success["pooled"]["capsule_layered_multi_round"]
                ["accuracy_full"], 1.000)
        self.assertEqual(
            self.r_w14_success["bench_summary"]
                ["scenarios_with_property"], 8)

    def test_degen_budget_w16_lambda_degenerate(self):
        """W16-Λ-degenerate: T_decoder=2 is below the round-2 specific
        claim's token cost — both packers collapse to 0.000."""
        self.assertEqual(
            self.r_degen_budget["pooled"]["capsule_attention_aware"]
                ["accuracy_full"], 0.000)
        self.assertEqual(
            self.r_degen_budget["pooled"]["capsule_layered_fifo_packed"]
                ["accuracy_full"], 0.000)

    def test_audit_OK_on_every_capsule_strategy(self):
        for rep_name, rep in (("composed", self.r_composed),
                                ("baseline", self.r_baseline),
                                ("naive_tight", self.r_naive_tight),
                                ("w14_success", self.r_w14_success),
                                ("degen_budget", self.r_degen_budget)):
            for s, ok in rep["audit_ok_grid"].items():
                if s == "substrate":
                    continue
                self.assertTrue(ok,
                    msg=f"{rep_name}/{s} failed audit")


class Phase63SeedStabilityTests(unittest.TestCase):
    """W16-1 stability: gap composed − fifo_pack ≥ 0.50 across 5/5
    seeds on the structured + tight regime."""

    @classmethod
    def setUpClass(cls):
        cls.sweep = run_phase63_seed_stability_sweep(
            seeds=(11, 17, 23, 29, 31),
            n_eval=8, K_auditor=12, T_auditor=256,
            T_decoder=24,
            extractor="magnitude_filter",
            prompt_mode=PRODUCER_PROMPT_STRUCTURED)

    def test_composed_one_thousand_on_every_seed(self):
        for seed, rep in self.sweep["per_seed"].items():
            v = (rep["pooled"]["capsule_attention_aware"]
                  ["accuracy_full"])
            self.assertEqual(v, 1.000,
                msg=f"seed={seed}: composed={v}, expected 1.000")

    def test_fifo_pack_zero_on_every_seed(self):
        for seed, rep in self.sweep["per_seed"].items():
            v = (rep["pooled"]["capsule_layered_fifo_packed"]
                  ["accuracy_full"])
            self.assertEqual(v, 0.000,
                msg=f"seed={seed}: fifo_pack={v}, expected 0.000")

    def test_gap_composed_minus_fifo_pack_holds_across_5_seeds(self):
        for seed in (11, 17, 23, 29, 31):
            gap = (self.sweep["per_seed"][seed]["headline_gap"]
                    ["composed_minus_fifo_packed"])
            self.assertGreaterEqual(gap, 0.50,
                msg=f"seed={seed}: composed - fifo_packed = {gap}")


class Phase63CrossRegimeTests(unittest.TestCase):
    """The seven canonical synthetic cells separate cleanly."""

    @classmethod
    def setUpClass(cls):
        cls.rep = run_cross_regime_summary(
            n_eval=8, bank_seed=11, K_auditor=12, T_auditor=256)

    def test_seven_synthetic_cells_separate_cleanly(self):
        d = self.rep
        # Sanity baseline: composed = 1.000.
        self.assertEqual(
            d["r63_baseline_loose"]["pooled"]["capsule_attention_aware"]
                ["accuracy_full"], 1.000)
        # W15-only: composed strict win over FIFO pack.
        self.assertEqual(
            d["r63_w15_only"]["pooled"]["capsule_attention_aware"]
                ["accuracy_full"], 1.000)
        self.assertEqual(
            d["r63_w15_only"]["pooled"]["capsule_layered_fifo_packed"]
                ["accuracy_full"], 0.000)
        # W14-only failure: bench broken upstream; everything fails.
        self.assertEqual(
            d["r63_w14_only"]["pooled"]["capsule_layered_multi_round"]
                ["accuracy_full"], 0.000)
        self.assertEqual(
            d["r63_w14_only"]["pooled"]["capsule_attention_aware"]
                ["accuracy_full"], 0.000)
        # Naive + tight: W16-Λ-compose joint failure.
        self.assertEqual(
            d["r63_naive_tight"]["pooled"]["capsule_attention_aware"]
                ["accuracy_full"], 0.000)
        # W14 success: structured + loose; cross-round wins.
        self.assertEqual(
            d["r63_w14_success"]["pooled"]["capsule_layered_multi_round"]
                ["accuracy_full"], 1.000)
        # Composed-tight: W16-1 anchor.
        self.assertEqual(
            d["r63_composed_tight"]["pooled"]["capsule_attention_aware"]
                ["accuracy_full"], 1.000)
        self.assertEqual(
            d["r63_composed_tight"]["pooled"]
                ["capsule_layered_fifo_packed"]
                ["accuracy_full"], 0.000)
        # Degen-budget: W16-Λ-degenerate.
        self.assertEqual(
            d["r63_degen_budget"]["pooled"]["capsule_attention_aware"]
                ["accuracy_full"], 0.000)


class Phase63OllamaReplayTests(unittest.TestCase):
    """The W16-Λ-real-replay anchor on the recorded qwen2.5:14b-32k bytes.

    The Phase-61 ollama-structured capture (n=8, 24 producer calls, 0
    endpoint failures) is byte-stable; the Phase-63 replay regime
    re-routes those bytes through the W14+W15 pipeline at K_auditor=8
    (Phase-61's anchor) and ``T_decoder=14`` (the centre of the budget
    band where the FIFO pack drops the round-2 disambiguator while the
    salience pack keeps it).
    """

    @classmethod
    def setUpClass(cls):
        if not (os.path.exists(_REPLAY_STRUCT)
                and os.path.exists(_REPLAY_NAIVE)):
            raise unittest.SkipTest(
                "Phase-61 real-Ollama replay sources missing")
        cls.r_loose = run_phase63(
            n_eval=8, K_auditor=8, T_auditor=256,
            bank_seed=11, bank_replicates=2,
            T_decoder=None,
            extractor="ollama_replay",
            replay_source=_REPLAY_STRUCT,
            verbose=False)
        cls.r_tight = run_phase63(
            n_eval=8, K_auditor=8, T_auditor=256,
            bank_seed=11, bank_replicates=2,
            T_decoder=14,
            extractor="ollama_replay",
            replay_source=_REPLAY_STRUCT,
            verbose=False)
        cls.r_naive_tight = run_phase63(
            n_eval=8, K_auditor=8, T_auditor=256,
            bank_seed=11, bank_replicates=2,
            T_decoder=14,
            extractor="ollama_replay",
            replay_source=_REPLAY_NAIVE,
            verbose=False)

    def test_replay_loose_matches_phase61_w14_anchor(self):
        """W16-3 anchor: the recorded structured-prompt bytes give
        accuracy_full = 0.500 at K_auditor=8, T_decoder=None — matches
        the Phase-61 W14-Λ-real anchor byte-for-byte."""
        for s in ("capsule_attention_aware",
                   "capsule_layered_multi_round",
                   "capsule_layered_fifo_packed",
                   "capsule_robust_multi_round",
                   "capsule_multi_round"):
            self.assertEqual(
                self.r_loose["pooled"][s]["accuracy_full"], 0.500,
                msg=f"replay_loose: {s} = "
                     f"{self.r_loose['pooled'][s]['accuracy_full']}, "
                     f"expected 0.500 (Phase-61 W14-Λ-real anchor)")

    def test_replay_tight_composed_strict_win(self):
        """W16-Λ-real-replay: at T_decoder=14, the W15 pack keeps the
        round-2 disambiguator while FIFO-pack drops it on the recorded
        qwen2.5:14b-32k bytes; strict gain = +0.500."""
        self.assertEqual(
            self.r_tight["pooled"]["capsule_attention_aware"]
                ["accuracy_full"], 0.500)
        self.assertEqual(
            self.r_tight["pooled"]["capsule_layered_fifo_packed"]
                ["accuracy_full"], 0.000)
        gap = (self.r_tight["headline_gap"]
                ["composed_minus_fifo_packed"])
        self.assertGreaterEqual(gap, 0.50,
            msg=f"replay_tight: composed - fifo_packed = {gap}")

    def test_replay_naive_tight_joint_failure(self):
        """The recorded naive-prompt bytes break the bench property
        upstream; even with the W15 packer the result is 0.000 (W16-Λ-
        compose anchor on real-LLM bytes)."""
        self.assertEqual(
            self.r_naive_tight["bench_summary"]
                ["scenarios_with_property"], 0)
        for s in ("capsule_attention_aware",
                   "capsule_layered_fifo_packed",
                   "capsule_layered_multi_round"):
            self.assertEqual(
                self.r_naive_tight["pooled"][s]["accuracy_full"], 0.000,
                msg=f"replay_naive_tight: {s} non-zero")

    def test_replay_audit_OK(self):
        for rep_name, rep in (("loose", self.r_loose),
                                ("tight", self.r_tight),
                                ("naive_tight", self.r_naive_tight)):
            for s, ok in rep["audit_ok_grid"].items():
                if s == "substrate":
                    continue
                self.assertTrue(ok,
                    msg=f"replay/{rep_name}/{s} failed audit")


if __name__ == "__main__":
    unittest.main()
