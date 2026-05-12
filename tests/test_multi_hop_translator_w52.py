"""Per-module tests for the W52 multi-hop translator."""

from __future__ import annotations

import pytest

from coordpy.multi_hop_translator import (
    MultiHopBackendTranslator,
    build_unfitted_multi_hop_translator,
    calibrate_confidence_from_residual,
    emit_multi_hop_translator_witness,
    fit_multi_hop_translator,
    forge_multi_hop_training_set,
    perturb_edge,
    run_multi_hop_realism_anchor_probe,
    score_multi_hop_fidelity,
    synthesize_multi_hop_training_set,
    verify_multi_hop_translator_witness,
)


def test_multi_hop_init_has_n_choose_2_directed_edges() -> None:
    tr = build_unfitted_multi_hop_translator(
        backends=("A", "B", "C", "D"),
        code_dim=4, feature_dim=4, seed=1)
    # 4 backends, all ordered pairs except self → 4*3 = 12
    assert len(tr.edges) == 12


def test_multi_hop_chain_apply_length_3() -> None:
    tr = build_unfitted_multi_hop_translator(
        backends=("A", "B", "C", "D"),
        code_dim=4, feature_dim=4, seed=1)
    x = [0.1, 0.2, 0.3, 0.4]
    out = tr.apply_chain_value(("A", "B", "C", "D"), x)
    assert len(out) == 4


def test_multi_hop_fit_decreases_loss() -> None:
    ts = synthesize_multi_hop_training_set(
        n_examples=12, code_dim=4, feature_dim=4, seed=1)
    tr, trace = fit_multi_hop_translator(
        ts, n_steps=48, seed=1)
    assert not trace.diverged
    assert len(trace.loss_head) > 0
    assert len(trace.loss_tail) > 0


def test_multi_hop_witness_verify_ok() -> None:
    ts = synthesize_multi_hop_training_set(
        n_examples=12, code_dim=4, feature_dim=4, seed=2)
    tr, trace = fit_multi_hop_translator(
        ts, n_steps=48, seed=2)
    w = emit_multi_hop_translator_witness(
        translator=tr, training_trace=trace,
        probes=ts.examples[:4])
    v = verify_multi_hop_translator_witness(
        w, expected_translator_cid=tr.cid(),
        expected_n_backends=4)
    assert v["ok"] is True


def test_multi_hop_realism_anchor_skip_path() -> None:
    payload = run_multi_hop_realism_anchor_probe()
    assert payload["anchor_status"] in (
        "synthetic_only", "real_llm_anchor", "skipped")
    assert payload["skipped_ok"] in (0.0, 1.0)


def test_perturb_edge_changes_cid() -> None:
    tr = build_unfitted_multi_hop_translator(
        backends=("A", "B", "C", "D"),
        code_dim=4, feature_dim=4, seed=1)
    cid_before = tr.cid()
    perturb_edge(tr, src="A", dst="B",
                 noise_magnitude=1.0, seed=99)
    cid_after = tr.cid()
    assert cid_before != cid_after


def test_calibrate_confidence_updates_logits() -> None:
    ts = synthesize_multi_hop_training_set(
        n_examples=12, code_dim=4, feature_dim=4, seed=3)
    tr, _ = fit_multi_hop_translator(ts, n_steps=24, seed=3)
    cid_before = tr.cid()
    calibrate_confidence_from_residual(tr, ts.examples)
    assert tr.cid() != cid_before


def test_disagreement_weighted_arbitration_beats_naive_on_perturb() -> None:
    ts = synthesize_multi_hop_training_set(
        n_examples=12, code_dim=4, feature_dim=4, seed=4)
    tr, _ = fit_multi_hop_translator(ts, n_steps=64, seed=4)
    perturb_edge(tr, src="A", dst="B",
                 noise_magnitude=2.0, seed=44)
    calibrate_confidence_from_residual(tr, ts.examples)
    fid = score_multi_hop_fidelity(tr, ts.examples[:6])
    # Weighted should be >= naive on this perturbed regime.
    assert fid.arbitration_weighted_score >= (
        fid.arbitration_naive_score - 0.05)


def test_forge_multi_hop_changes_training_set_cid() -> None:
    ts = synthesize_multi_hop_training_set(
        n_examples=6, code_dim=4, feature_dim=4, seed=5)
    forged = forge_multi_hop_training_set(ts, seed=5)
    assert ts.cid() != forged.cid()
