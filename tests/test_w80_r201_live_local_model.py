"""W80 / P0 #12 — R-201 live local-model benchmark tests."""

from __future__ import annotations

import pytest

try:
    import torch  # type: ignore  # noqa: F401
    import transformers  # type: ignore  # noqa: F401
    _HAS_HF = True
except Exception:  # noqa: BLE001
    _HAS_HF = False


def test_w80_r201_schema_versioned():
    from coordpy.r201_benchmark import R201_SCHEMA_VERSION
    assert R201_SCHEMA_VERSION == "coordpy.r201_benchmark.v1"


def test_w80_r201_returns_deterministic_unavailable_report_without_hf():
    """When transformers is missing, R-201 must produce a
    deterministic 'unavailable' report rather than silently
    passing or crashing."""
    if _HAS_HF:
        pytest.skip(
            "transformers available — this exercises the "
            "missing-deps path")
    from coordpy.r201_benchmark import run_r201
    r = run_r201()
    assert r["transformers_available"] is False
    assert bool(r["all_pass"]) is False
    assert r["cells"]["H1400"] is False


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed")
def test_w80_r201_all_pass_against_live_model():
    from coordpy.r201_benchmark import run_r201
    r = run_r201()
    assert r["transformers_available"] is True
    assert bool(r["all_pass"]), (
        f"r201 failures: "
        f"{dict((k, v) for k, v in r['cells'].items() if not v)}")
    # R-201 ships 32 H-bars (H1400..H1431). The first wave
    # (H1400..H1421) is the core live-runtime conformance set;
    # the second wave (H1422..H1431) covers the cross-backend
    # mechanism deltas (#6), task success + answer consistency
    # + visible-token / recompute cost (#12), and the
    # substrate adapter V25 + capability matrix V1
    # integration checks (#5 + #17).
    assert len(r["cells"]) == 32
    for h in range(1400, 1432):
        assert f"H{h}" in r["cells"]


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed")
def test_w80_r201_replay_is_byte_identical_within_fp32_tol():
    from coordpy.runtime_instrumentation_v1 import (
        W80_REPLAY_FROM_KV_MAX_ABS_DIFF,
    )
    from coordpy.r201_benchmark import run_r201
    r = run_r201()
    # The replay-vs-recompute summary should report byte-identity.
    summary = r["metrics"]["replay_vs_recompute_summary"]
    assert bool(summary["replay_byte_identical"])
    assert float(summary["max_abs_diff_last_logits"]) < float(
        W80_REPLAY_FROM_KV_MAX_ABS_DIFF)


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed")
def test_w80_r201_substrate_routing_saves_token_work():
    from coordpy.r201_benchmark import run_r201
    r = run_r201()
    metrics = r["metrics"]
    # Substrate routing must save strictly less token work than
    # transcript-only on a long-prompt scenario.
    assert int(metrics["token_work_substrate_routed"]) < int(
        metrics["token_work_transcript_only"])
    # And the saving ratio must be at least 30 % on the default
    # prompt.
    assert float(
        metrics["substrate_routed_token_saving_ratio"]) >= 0.3


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed")
def test_w80_r201_conformance_passes_on_live_runtime():
    from coordpy.r201_benchmark import run_r201
    r = run_r201()
    assert int(r["metrics"]["conformance_fail_count"]) == 0
    assert int(r["metrics"]["conformance_pass_count"]) == 12


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed")
def test_w80_r201_answer_consistency_deterministic():
    """Two greedy completions from the same prompt with the
    same seed produce identical token sequences — the live
    runtime is deterministic."""
    from coordpy.r201_benchmark import run_r201
    r = run_r201()
    ac = r["metrics"]["answer_consistency"]
    assert bool(ac["match"]), (
        f"answer consistency must be deterministic; "
        f"a={ac['gen_a_token_ids']} b={ac['gen_b_token_ids']}")
    assert int(ac["n_completion_tokens"]) > 0


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed")
def test_w80_r201_task_success_completion_unambiguous():
    """The live runtime can produce a non-empty greedy
    completion whose final-position top-1 logit beats top-2
    by ≥ 0.1 — i.e. the completion is unambiguous, not a
    near-tie."""
    from coordpy.r201_benchmark import run_r201
    r = run_r201()
    ts = r["metrics"]["task_success"]
    assert int(ts["completion_token_count"]) > 0
    assert float(ts["final_top1_minus_top2"]) > 0.1


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed")
def test_w80_r201_substrate_adapter_v25_carries_hf_backend():
    """The W80 substrate adapter V25 surfaces the HF runtime
    inside the existing substrate adapter line — closes the
    P0 #5 integration bar."""
    from coordpy.r201_benchmark import run_r201
    r = run_r201()
    sav25 = r["metrics"]["substrate_adapter_v25"]
    assert "transformers_runtime_v1" in sav25["backends"]
    assert int(sav25["n_controlled_runtimes"]) >= 2


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed")
def test_w80_r201_cross_backend_replay_delta():
    """The replay-from-KV mechanism produces small last-row
    logit divergence on BOTH backends — the W80 cross-backend
    mechanism transfer evidence."""
    from coordpy.runtime_instrumentation_v1 import (
        W80_REPLAY_FROM_KV_MAX_ABS_DIFF,
    )
    from coordpy.r201_benchmark import run_r201
    r = run_r201()
    delta = r["metrics"]["xback_replay_delta"]
    assert "numpy_diff" in delta
    assert "transformers_diff" in delta
    # NumPy: byte-identical replay (< 1e-8).
    assert float(delta["numpy_diff"]) < 1e-8
    # Transformers: byte-identical-in-fp32 replay (< 5e-3).
    assert float(delta["transformers_diff"]) < float(
        W80_REPLAY_FROM_KV_MAX_ABS_DIFF)
