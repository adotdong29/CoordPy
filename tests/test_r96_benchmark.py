"""Tests for the R-96 benchmark family — H1..H10 hypotheses.

Each test runs one R-96 family on a small seed set and asserts
the pre-committed success bar of the corresponding hypothesis
from ``docs/SUCCESS_CRITERION_W49_MULTI_BLOCK_PROXY.md``.
"""

from __future__ import annotations

from coordpy.r96_benchmark import (
    run_family,
)


SEEDS = (0, 1, 2)


class TestH1TrivialPassthrough:
    def test_passthrough_ok_across_all_arms(self):
        comp = run_family(
            "r96_trivial_multi_block_passthrough", seeds=SEEDS)
        for arm in (
                "baseline_team", "w48_shared_state",
                "w49_multi_block"):
            a = comp.get(arm)
            assert a is not None, arm
            assert a.mean == 1.0, (arm, a.values)


class TestH2MultiBlockDepth:
    def test_depth_advantage(self):
        comp = run_family(
            "r96_multi_block_depth", seeds=SEEDS)
        w49 = comp.get("w49_multi_block")
        w48 = comp.get("w48_shared_state")
        assert w49 is not None and w48 is not None
        # Pre-committed bar: delta ≥ 0.10.
        assert w49.mean - w48.mean >= 0.10


class TestH3MultiBankRecall:
    def test_multi_bank_beats_single(self):
        comp = run_family("r96_multi_bank_recall", seeds=SEEDS)
        w49 = comp.get("w49_multi_block")
        w48 = comp.get("w48_shared_state")
        assert w49 is not None and w48 is not None
        assert w49.mean - w48.mean >= 0.20


class TestH4LearnedEviction:
    def test_learned_beats_fifo(self):
        comp = run_family("r96_learned_eviction", seeds=SEEDS)
        w49 = comp.get("w49_multi_block")
        w48 = comp.get("w48_shared_state")
        assert w49 is not None and w48 is not None
        assert w49.mean - w48.mean >= 0.20


class TestH5RetentionHead:
    def test_retention_head_correctness(self):
        comp = run_family("r96_retention_head", seeds=SEEDS)
        w49 = comp.get("w49_multi_block")
        assert w49 is not None
        assert w49.mean >= 0.85


class TestH6DictionaryCompression:
    def test_dictionary_saves_tokens(self):
        comp = run_family(
            "r96_dictionary_compression", seeds=SEEDS)
        w49 = comp.get("w49_multi_block")
        assert w49 is not None
        assert w49.mean >= 0.25


class TestH7SharedLatentCapsule:
    def test_chain_walks_back(self):
        comp = run_family(
            "r96_shared_latent_capsule", seeds=SEEDS)
        w49 = comp.get("w49_multi_block")
        assert w49 is not None
        assert w49.mean == 1.0


class TestH8CrossBankInterference:
    def test_zero_perturbation(self):
        comp = run_family(
            "r96_cross_bank_interference", seeds=SEEDS)
        w49 = comp.get("w49_multi_block")
        assert w49 is not None
        # Perturbation ≤ 0.05.
        assert w49.mean <= 0.05


class TestH9ReplayDeterminism:
    def test_replay_deterministic(self):
        comp = run_family("r96_replay_determinism", seeds=SEEDS)
        w49 = comp.get("w49_multi_block")
        assert w49 is not None
        assert w49.mean == 1.0


class TestH10EnvelopeVerifier:
    def test_verifier_sound(self):
        comp = run_family("r96_envelope_verifier", seeds=SEEDS)
        w49 = comp.get("w49_multi_block")
        assert w49 is not None
        assert w49.mean == 1.0
