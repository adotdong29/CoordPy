"""W84 — audit-infrastructure tests.

Covers the W84 modules that ship the contract / probe / adapter
for issues that are honestly hardware-blocked:

* ``coordpy.frontier_capability_probe_v1`` (#25)
* ``coordpy.live_hidden_state_dataset_v1`` (#26)
* ``coordpy.long_context_substrate_bench_v1`` (#27)
* ``coordpy.real_task_bench_adapter_v1`` (#28)
* ``coordpy.precision_tier_contract_v1`` (#30)
"""

from __future__ import annotations

import os
import tempfile

import pytest

from coordpy.frontier_capability_probe_v1 import (
    FrontierBlockedOnHardwareError,
    W84_FRONTIER_OPEN_WEIGHT_CANDIDATES,
    probe_frontier_capability_v1,
    run_frontier_substrate_bench_v1,
)
from coordpy.live_hidden_state_dataset_v1 import (
    LiveHiddenStateDatasetCapsuleV1,
    LiveTrainingBlockedOnHardwareError,
    W84_LIVE_DATASET_V1_SCHEMA_VERSION,
    build_live_hidden_state_dataset_v1,
    materialise_live_hidden_state_dataset_v1,
)
from coordpy.long_context_substrate_bench_v1 import (
    NeedleHaystackPromptV1,
    W84_LONG_CONTEXT_V1_SCHEMA_VERSION,
    build_needle_haystack_corpus_v1,
    run_long_context_substrate_bench_v1,
)
from coordpy.precision_tier_contract_v1 import (
    PrecisionTier,
    PrecisionTierContractV1,
    W84_PRECISION_TIER_FLOORS,
    W84_PRECISION_TIER_V1_SCHEMA_VERSION,
    build_precision_tier_contract_v1,
    precision_tier_floor_for,
    probe_precision_tier_capability_v1,
)
from coordpy.real_task_bench_adapter_v1 import (
    RealTaskBenchAdapterV1,
    RealTaskBenchBlockedOnModelError,
    W84_REAL_TASK_BENCH_V1_SCHEMA_VERSION,
)


# ---------------------------------------------------------------
# Frontier probe (#25).
# ---------------------------------------------------------------

def test_w84_frontier_probe_does_not_mock():
    rep = probe_frontier_capability_v1()
    # On this host, the probe MUST honestly say not-ready.
    assert rep.ready_for_frontier_bench is False
    assert len(rep.blocked_reason) > 0
    assert rep.cache_models_found == ()


def test_w84_frontier_probe_is_content_addressed():
    r1 = probe_frontier_capability_v1()
    r2 = probe_frontier_capability_v1()
    # The probe's CID is deterministic on host state; identical
    # within a single test run.
    assert r1.cid() == r2.cid()
    assert len(r1.cid()) == 64


def test_w84_frontier_bench_raises_when_blocked():
    with pytest.raises(FrontierBlockedOnHardwareError):
        run_frontier_substrate_bench_v1()


def test_w84_frontier_candidates_list_includes_llama_and_qwen():
    # Sanity: the candidate list covers the issue body's
    # accepted targets.
    names = " ".join(W84_FRONTIER_OPEN_WEIGHT_CANDIDATES)
    assert "Llama" in names
    assert "Qwen" in names


# ---------------------------------------------------------------
# Live hidden-state dataset (#26).
# ---------------------------------------------------------------

def test_w84_live_dataset_enforces_held_out_disjointness():
    with pytest.raises(ValueError):
        build_live_hidden_state_dataset_v1(
            prompts_train=["hello world"],
            prompts_eval=["hello world"],  # overlap!
            model_name="distilgpt2")


def test_w84_live_dataset_capsule_is_content_addressed():
    cap = build_live_hidden_state_dataset_v1(
        prompts_train=["train prompt 1", "train prompt 2"],
        prompts_eval=["eval prompt 1", "eval prompt 2"],
        model_name="distilgpt2",
        layer_index=4)
    cap2 = build_live_hidden_state_dataset_v1(
        prompts_train=["train prompt 1", "train prompt 2"],
        prompts_eval=["eval prompt 1", "eval prompt 2"],
        model_name="distilgpt2",
        layer_index=4)
    assert cap.cid() == cap2.cid()
    assert len(cap.cid()) == 64


def test_w84_materialise_refuses_without_transformers():
    cap = build_live_hidden_state_dataset_v1(
        prompts_train=["a"], prompts_eval=["b"],
        model_name="distilgpt2")
    # When transformers is not installed, must raise the
    # structured BlockedOnHardware error — never a synthetic
    # fallback.
    try:
        import transformers  # type: ignore  # noqa: F401
        pytest.skip(
            "transformers is available; this test only "
            "exercises the no-transformers refusal path")
    except Exception:
        pass
    with pytest.raises(LiveTrainingBlockedOnHardwareError):
        materialise_live_hidden_state_dataset_v1(
            capsule=cap,
            prompts_train=["a"], prompts_eval=["b"],
            model_name="distilgpt2")


# ---------------------------------------------------------------
# Long-context substrate bench (#27).
# ---------------------------------------------------------------

def test_w84_needle_haystack_prompts_are_deterministic():
    p1 = NeedleHaystackPromptV1(
        schema=W84_LONG_CONTEXT_V1_SCHEMA_VERSION,
        prompt_id="t", total_positions=64,
        needle_position=32, needle_value=251, seed=1)
    p2 = NeedleHaystackPromptV1(
        schema=W84_LONG_CONTEXT_V1_SCHEMA_VERSION,
        prompt_id="t", total_positions=64,
        needle_position=32, needle_value=251, seed=1)
    assert p1.cid() == p2.cid()
    tokens = p1.materialise_positions()
    assert tokens == p2.materialise_positions()
    assert tokens[32] == 251


def test_w84_long_context_bench_substrate_beats_v3_at_32k():
    rep = run_long_context_substrate_bench_v1(
        horizons=(2_000, 8_000, 32_000), n_per_horizon=3)
    # The substrate dominates V3 on horizons past V3's coverage
    # (k=256 + 4*256 summary = 1280 positions). At 2k the needle
    # at position 1000 IS inside V3's summary coverage (1280),
    # so V3 succeeds. At 8k and 32k it does not — V3 fails.
    # Substrate succeeds at every horizon.
    pt_8k = next(
        p for p in rep.per_horizon_points if p.horizon == 8000)
    pt_32k = next(
        p for p in rep.per_horizon_points if p.horizon == 32000)
    assert pt_8k.substrate_strictly_beats_v3
    assert pt_32k.substrate_strictly_beats_v3
    assert pt_32k.substrate_task_success_rate == 1.0
    assert pt_32k.bounded_v3_task_success_rate == 0.0
    assert pt_8k.bounded_v3_task_success_rate == 0.0


# ---------------------------------------------------------------
# Precision tier contract (#30).
# ---------------------------------------------------------------

def test_w84_precision_tier_contract_refuses_widening():
    """Anti-cheat: the contract refuses to be built with a
    floor that does NOT match the canonical per-tier floor."""
    with pytest.raises(ValueError):
        PrecisionTierContractV1(
            schema=W84_PRECISION_TIER_V1_SCHEMA_VERSION,
            declared_tier=PrecisionTier.TIER_INT8.value,
            max_abs_diff_floor=5e-3,  # fp32 floor — wrong tier!
            semantic_equivalence_floor=0.95,
            runtime_id="x")


def test_w84_precision_tier_floor_lookup():
    assert (precision_tier_floor_for(tier=PrecisionTier.TIER_FP32)
            == 5e-3)
    assert (precision_tier_floor_for(tier=PrecisionTier.TIER_BF16)
            == 5e-2)
    assert (precision_tier_floor_for(tier=PrecisionTier.TIER_INT8)
            == 2e-1)


def test_w84_precision_tier_capability_probe_reports_fp32_always():
    probe = probe_precision_tier_capability_v1()
    assert probe.fp32_available is True


def test_w84_precision_tier_contract_cid_stable():
    c1 = build_precision_tier_contract_v1(
        tier=PrecisionTier.TIER_BF16,
        runtime_id="test_runtime")
    c2 = build_precision_tier_contract_v1(
        tier=PrecisionTier.TIER_BF16,
        runtime_id="test_runtime")
    assert c1.cid() == c2.cid()
    assert len(c1.cid()) == 64


# ---------------------------------------------------------------
# Real-task bench adapter (#28).
# ---------------------------------------------------------------

def test_w84_real_task_adapter_refuses_to_run_without_model():
    adapter = RealTaskBenchAdapterV1(
        schema=W84_REAL_TASK_BENCH_V1_SCHEMA_VERSION)
    # Need a stub JSONL.
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"instance_id": "t1", '
                '"problem_statement": "x"}\n')
        path = f.name
    try:
        with pytest.raises(RealTaskBenchBlockedOnModelError):
            adapter.run_harness(
                jsonl_path=path, model_client=None,
                composed_pipeline_config_cid="cfg-cid")
    finally:
        os.unlink(path)


def test_w84_real_task_adapter_ingest_swe_lite_emits_plan_chain():
    adapter = RealTaskBenchAdapterV1(
        schema=W84_REAL_TASK_BENCH_V1_SCHEMA_VERSION)
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"instance_id": "task-1", '
                '"problem_statement": "Fix the bug"}\n')
        f.write('{"instance_id": "task-2", '
                '"problem_statement": "Add the test"}\n')
        path = f.name
    try:
        chain = adapter.plan_from_swe_lite_jsonl(
            jsonl_path=path,
            composed_pipeline_config_cid="cfg-cid")
        assert chain.n_tasks == 2
        assert len(chain.chain_root_cid) == 64
        assert len(chain.plan_cids) == 2
    finally:
        os.unlink(path)
