"""W83 — live hidden-state intercept bench tests.

These tests skip the live-runtime check when transformers/torch
are not installed; in that case we verify only that the bench's
skip-friendly contract is honored.
"""

from __future__ import annotations

import pytest


def test_w83_hidden_intercept_bench_emits_report_with_or_without_hf():
    from coordpy.hidden_state_intercept_bench_v1 import (
        run_hidden_state_intercept_bench_v1,
    )
    rep = run_hidden_state_intercept_bench_v1()
    assert len(rep.cid()) == 64
    # Schema is stable.
    assert rep.schema == (
        "coordpy.hidden_state_intercept_bench_v1.v1")
    # Either transformers is available (then live checks run)
    # or it isn't (then skip-friendly fields are populated).


def test_w83_hidden_intercept_bench_witness_emitted():
    from coordpy.hidden_state_intercept_bench_v1 import (
        emit_hidden_state_intercept_bench_witness_v1,
        run_hidden_state_intercept_bench_v1,
    )
    rep = run_hidden_state_intercept_bench_v1()
    w = emit_hidden_state_intercept_bench_witness_v1(
        bench=rep)
    assert len(w.cid()) == 64
    assert bool(w.transformers_available) == bool(
        rep.transformers_available)


def test_w83_hidden_intercept_bench_live_when_hf_available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        pytest.skip(
            "transformers / torch not installed; skipping "
            "live HF intercept test")
    from coordpy.hidden_state_intercept_bench_v1 import (
        run_hidden_state_intercept_bench_v1,
    )
    rep = run_hidden_state_intercept_bench_v1()
    assert bool(rep.transformers_available)
    assert bool(rep.replay_byte_identical)
    assert bool(rep.hidden_state_intercept_moves_cid)
    assert len(rep.full_trace_cid) == 64
    assert len(rep.replay_trace_cid) == 64
    assert len(rep.hidden_inject_trace_cid) == 64
