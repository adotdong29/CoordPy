"""Per-module tests for the W52 transcript-vs-shared-state comparator."""

from __future__ import annotations

import random

import pytest

from coordpy.quantised_compression import (
    QuantisedBudgetGate,
    QuantisedCodebookV4,
    fit_quantised_compression,
    synthesize_quantised_compression_training_set,
)
from coordpy.transcript_vs_shared_state import (
    compare_transcript_vs_shared_state,
    emit_transcript_vs_shared_witness,
    verify_transcript_vs_shared_witness,
)


def _build_codebook_and_gate(seed: int, *, code_dim: int = 12):
    ts = synthesize_quantised_compression_training_set(
        n_examples=24, code_dim=code_dim, n_coarse=32,
        n_fine=16, n_ultra=8, emit_mask_len=16, seed=seed)
    cb, gate, _ = fit_quantised_compression(
        ts, n_steps=16, seed=seed)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    return cb, gate


def test_comparator_returns_both_arms_and_gap() -> None:
    cb, gate = _build_codebook_and_gate(seed=1)
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.0, -0.1, -0.2,
         -0.3, 0.0, 0.1, 0.2]
    res = compare_transcript_vs_shared_state(
        x, codebook=cb, gate=gate, budget_tokens=3)
    assert res.budget_tokens == 3
    assert 0.0 <= res.transcript_retention_cosine <= 1.0
    assert 0.0 <= res.shared_retention_cosine <= 1.0


def test_witness_round_trips() -> None:
    cb, gate = _build_codebook_and_gate(seed=2)
    rng = random.Random(2)
    carriers = [
        [rng.uniform(-1, 1) for _ in range(12)]
        for _ in range(8)
    ]
    w = emit_transcript_vs_shared_witness(
        carriers=carriers, codebook=cb, gate=gate,
        budget_tokens=3)
    v = verify_transcript_vs_shared_witness(
        w, expected_comparison_cid=w.comparison_cid,
        expected_budget_tokens=3,
        expected_n_probes=len(carriers))
    assert v["ok"] is True


def test_shared_arm_beats_transcript_at_tight_budget() -> None:
    cb, gate = _build_codebook_and_gate(seed=3)
    rng = random.Random(3)
    carriers = [
        [rng.uniform(-1, 1) for _ in range(12)]
        for _ in range(16)
    ]
    w = emit_transcript_vs_shared_witness(
        carriers=carriers, codebook=cb, gate=gate,
        budget_tokens=3)
    assert (
        w.shared_retention_cosine
        > w.transcript_retention_cosine)


def test_empty_carriers_returns_zero_witness() -> None:
    cb, gate = _build_codebook_and_gate(seed=4)
    w = emit_transcript_vs_shared_witness(
        carriers=[], codebook=cb, gate=gate,
        budget_tokens=3)
    assert w.n_probes == 0
    assert w.shared_retention_cosine == 0.0
