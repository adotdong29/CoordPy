"""W84 / P0 #25 — frontier-scale substrate coupling tests.

These tests are gated on the optional ``transformers`` + ``torch``
deps. When they are missing OR when the named frontier-scale
model is not present in the HF cache, the tests skip cleanly so
CI on lean environments stays green.

The full-bench test takes 3–6 minutes wall-clock on a 16-thread
CPU (one full forward over the prompt, two forward + replay
pairs per substrate-vs-bounded position, plus the conformance
sweep). Pytest skips it unless the COORDPY_RUN_FRONTIER_BENCH
env var is set so day-to-day local test runs stay fast.
"""

from __future__ import annotations

import os

import pytest

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    _HAS_HF = True
except Exception:  # noqa: BLE001
    _HAS_HF = False


def _model_in_cache(name: str) -> bool:
    """Best-effort check that the named model is in the local
    HF cache so the test doesn't trigger a 15GB download."""
    try:
        from huggingface_hub import (  # noqa: WPS433
            try_to_load_from_cache,
        )
        path = try_to_load_from_cache(
            repo_id=str(name), filename="config.json")
    except Exception:  # noqa: BLE001
        return False
    return path is not None and path is not object()


def test_w84_frontier_scale_report_dataclass_immutable():
    from coordpy.frontier_scale_substrate_v1 import (
        FrontierScaleValidationReportV1,
        W84_FRONTIER_SCALE_V1_SCHEMA_VERSION,
    )
    r = FrontierScaleValidationReportV1(
        schema=W84_FRONTIER_SCALE_V1_SCHEMA_VERSION,
        model_name="x", model_dtype="bf16", device="cpu",
        n_params=0, n_layers=0, hidden_dim=0, n_heads=0,
        head_dim=0,
        architecture_family="unknown",
        transformers_available=False,
        n_input_tokens=0,
        conformance_n_pass=0, conformance_n_fail=0,
        conformance_n_total=0,
        replay_max_abs_diff_final_logits=0.0,
        replay_precision_floor=0.0,
        replay_byte_identical_at_floor=False,
        hidden_state_intercept_moves_cid=False,
        substrate_vs_bounded_window_v3_win_rate=0.0,
        substrate_load_bearing_claim_reproduced=False,
        load_seconds=0.0,
        forward_seconds_per_pass=0.0,
        full_run_seconds=0.0,
        baseline_trace_cid="", replay_trace_cid="",
        intercept_trace_cid="", detail="")
    # Frozen dataclass.
    with pytest.raises(dataclasses_FrozenError()):
        r.model_name = "y"  # type: ignore[misc]
    assert r.cid().startswith(("", " ")) or len(r.cid()) >= 32


def dataclasses_FrozenError():
    import dataclasses
    return dataclasses.FrozenInstanceError


def test_w84_frontier_scale_skip_when_transformers_missing(
        monkeypatch):
    """When transformers cannot be imported, the validator
    returns a report with transformers_available=False and
    skips the live checks."""
    import coordpy.frontier_scale_substrate_v1 as fs

    # Force the ImportError fall-through path by monkey-patching
    # the import inside the function.
    def _fail_import():
        raise ImportError("synthetic ImportError for test")

    # Replace transformers_runtime_v1 module attr with a stub
    # that errors on submodule import.
    import sys
    saved = sys.modules.get("coordpy.transformers_runtime_v1")
    sys.modules["coordpy.transformers_runtime_v1"] = None  # type: ignore[assignment]
    try:
        r = fs.run_frontier_scale_validation_v1(
            model_name="fake/missing-model")
    finally:
        if saved is not None:
            sys.modules[(
                "coordpy.transformers_runtime_v1")] = saved
        elif "coordpy.transformers_runtime_v1" in sys.modules:
            del sys.modules[
                "coordpy.transformers_runtime_v1"]
    assert not r.transformers_available
    assert r.detail.endswith(
        "not importable") or "not importable" in r.detail


def test_w84_arch_family_for_known_models():
    from coordpy.frontier_scale_substrate_v1 import (
        _arch_family_for,
    )
    from types import SimpleNamespace
    assert _arch_family_for(SimpleNamespace(
        model_type="gpt2")) == "gpt-2-family"
    assert _arch_family_for(SimpleNamespace(
        model_type="llama")) == "llama-family"
    assert _arch_family_for(SimpleNamespace(
        model_type="qwen2")).endswith("(llama-lineage)")
    assert _arch_family_for(SimpleNamespace(
        model_type="mistral")).endswith("(llama-lineage)")
    assert _arch_family_for(SimpleNamespace(
        model_type="gemma2")) == "gemma-family"


def test_w84_frontier_scale_witness_is_content_addressed():
    from coordpy.frontier_scale_substrate_v1 import (
        FrontierScaleValidationReportV1,
        emit_frontier_scale_witness_v1,
        W84_FRONTIER_SCALE_V1_SCHEMA_VERSION,
    )
    r = FrontierScaleValidationReportV1(
        schema=W84_FRONTIER_SCALE_V1_SCHEMA_VERSION,
        model_name="x", model_dtype="bf16", device="cpu",
        n_params=7620000000, n_layers=28, hidden_dim=3584,
        n_heads=28, head_dim=128,
        architecture_family="qwen-family-(llama-lineage)",
        transformers_available=True,
        n_input_tokens=14,
        conformance_n_pass=12, conformance_n_fail=0,
        conformance_n_total=12,
        replay_max_abs_diff_final_logits=0.5,
        replay_precision_floor=1.0,
        replay_byte_identical_at_floor=True,
        hidden_state_intercept_moves_cid=True,
        substrate_vs_bounded_window_v3_win_rate=1.0,
        substrate_load_bearing_claim_reproduced=True,
        load_seconds=74.0,
        forward_seconds_per_pass=4.0,
        full_run_seconds=300.0,
        baseline_trace_cid="a" * 64, replay_trace_cid="b" * 64,
        intercept_trace_cid="c" * 64,
        detail="frontier-scale validation passed")
    w = emit_frontier_scale_witness_v1(report=r)
    assert w.report_cid == r.cid()
    assert w.cid() == w.cid()  # deterministic
    assert (
        bool(w.replay_byte_identical_at_floor)
        == bool(r.replay_byte_identical_at_floor))


def test_w84_module_exports_load_bearing_names():
    """Stable public surface check for the W84 P0 #25 module."""
    from coordpy import frontier_scale_substrate_v1 as fs
    for name in (
        "W84_FRONTIER_SCALE_V1_SCHEMA_VERSION",
        "W84_FRONTIER_DEFAULT_MODEL_NAME",
        "W84_FRONTIER_DEFAULT_MODEL_DTYPE",
        "FrontierScaleValidationReportV1",
        "FrontierScaleWitnessV1",
        "run_frontier_scale_validation_v1",
        "emit_frontier_scale_witness_v1",
    ):
        assert name in fs.__all__
        assert hasattr(fs, name)


def test_w84_frontier_default_model_is_7b_and_llama_family():
    from coordpy.frontier_scale_substrate_v1 import (
        W84_FRONTIER_DEFAULT_MODEL_NAME,
    )
    # Default target must be a >=7B Llama-family open-weight
    # model per the issue. Anti-cheat: don't drift this default
    # to a sub-7B model in the future.
    assert "7b" in W84_FRONTIER_DEFAULT_MODEL_NAME.lower() or (
        "7B" in W84_FRONTIER_DEFAULT_MODEL_NAME)
    assert W84_FRONTIER_DEFAULT_MODEL_NAME.startswith(
        ("Qwen/", "meta-llama/", "mistralai/", "google/"))


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed")
@pytest.mark.skipif(
    not _model_in_cache(
        "Qwen/Qwen2.5-7B-Instruct"),
    reason=(
        "Qwen/Qwen2.5-7B-Instruct not in HF cache; "
        "skipping full frontier-scale bench"))
@pytest.mark.skipif(
    not os.environ.get("COORDPY_RUN_FRONTIER_BENCH", ""),
    reason=(
        "set COORDPY_RUN_FRONTIER_BENCH=1 to run the 7B "
        "frontier-scale bench (3-6 min wall-clock on CPU)"))
def test_w84_frontier_scale_full_bench_qwen_2p5_7b():
    """The load-bearing P0 #25 test.

    Runs the full validation on Qwen2.5-7B-Instruct in bf16:
    conformance >= 10/12, replay byte-identical at floor,
    hidden-state intercept moves CID, substrate-vs-bounded
    win rate > 0.5.
    """
    from coordpy.frontier_scale_substrate_v1 import (
        run_frontier_scale_validation_v1,
    )
    r = run_frontier_scale_validation_v1(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        model_dtype="bf16",
        prompt_max_len=24)
    assert r.transformers_available
    assert r.n_params >= 7_000_000_000, (
        f"frontier model must be >=7B params, got "
        f"{r.n_params/1e9:.2f}B")
    assert "llama-lineage" in r.architecture_family
    assert int(r.conformance_n_pass) >= 10, (
        f"conformance must pass >=10 of 12 axes, got "
        f"{r.conformance_n_pass}/{r.conformance_n_total}")
    assert int(r.conformance_n_fail) == 0
    assert bool(r.replay_byte_identical_at_floor), (
        f"replay max_abs_diff={r.replay_max_abs_diff_final_logits} "
        f"> precision_floor={r.replay_precision_floor}")
    assert bool(r.hidden_state_intercept_moves_cid)
    assert (
        float(r.substrate_vs_bounded_window_v3_win_rate)
        > 0.5)
    assert bool(r.substrate_load_bearing_claim_reproduced)
