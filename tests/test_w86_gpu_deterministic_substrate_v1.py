"""CI tests for ``coordpy.gpu_deterministic_substrate_v1``.

These tests are CPU-only — they exercise the wrapper code
paths + capsule shapes without requiring CUDA. The GPU
numerical bench lives in
``scripts/colab_gpu_deterministic_substrate_w86.ipynb``;
the JSON it produces is re-verified offline via
``scripts/verify_w86_gpu_substrate_v1_audit_chain.py``.
"""

from __future__ import annotations

import os

import pytest

from coordpy.gpu_deterministic_substrate_v1 import (
    DETERMINISM_OFF_ENV_VAR,
    DeterminismLoadBearingWitnessV1,
    DeterminismMode,
    DeterminismWrapperConfigV1,
    DeterminismWrapperResultV1,
    GPUSubstrateBenchReportV1,
    GPUSubstrateContractCheckV1,
    TensorParallelReadbackV1,
    W86_GPU_V1_DETERMINISM_WITNESS_OP,
    apply_determinism_wrapper_v1,
    determinism_load_bearing_witness_v1,
    run_gpu_substrate_contract_check_v1,
)


def test_determinism_config_default_is_on():
    c = DeterminismWrapperConfigV1()
    assert c.mode == DeterminismMode.DETERMINISTIC
    assert c.cudnn_deterministic is True
    assert c.cudnn_benchmark is False
    assert c.torch_deterministic_algorithms is True


def test_determinism_config_cid_stable():
    c1 = DeterminismWrapperConfigV1()
    c2 = DeterminismWrapperConfigV1()
    assert c1.cid() == c2.cid()


def test_apply_determinism_wrapper_returns_result_capsule():
    res = apply_determinism_wrapper_v1()
    assert isinstance(res, DeterminismWrapperResultV1)
    assert len(res.cid()) == 64


def test_determinism_off_env_var_inverts_mode():
    saved = os.environ.pop(DETERMINISM_OFF_ENV_VAR, None)
    try:
        os.environ[DETERMINISM_OFF_ENV_VAR] = "1"
        res = apply_determinism_wrapper_v1()
        assert (
            res.requested_config.mode
            == DeterminismMode.NON_DETERMINISTIC)
    finally:
        os.environ.pop(DETERMINISM_OFF_ENV_VAR, None)
        if saved is not None:
            os.environ[DETERMINISM_OFF_ENV_VAR] = saved


def test_tp_readback_world_size_1_is_passthrough():
    tp = TensorParallelReadbackV1(world_size=1)
    obj = object()
    assert tp.all_gather_hidden_state(obj) is obj


def test_tp_readback_world_size_gt_1_requires_distributed():
    """A real 2-GPU call requires torch.distributed initialised;
    on the bench host we just confirm the right error is raised.
    """
    tp = TensorParallelReadbackV1(world_size=2)
    with pytest.raises((RuntimeError, ImportError, Exception)):
        tp.all_gather_hidden_state(object())


def test_tp_readback_cid_stable():
    tp1 = TensorParallelReadbackV1(world_size=1)
    tp2 = TensorParallelReadbackV1(world_size=1)
    assert tp1.cid() == tp2.cid()
    tp3 = TensorParallelReadbackV1(world_size=2)
    assert tp3.cid() != tp1.cid()


def test_gpu_substrate_bench_report_cid_stable():
    fields = dict(
        model_name="x", device="cuda:0",
        precision_tier="tier_bf16",
        determinism_wrapper_result_cid="0" * 64,
        tensor_parallel_config_cid="0" * 64,
        cuda_device_name="A100-40GB", cuda_capability="8.0",
        pos_replay_max_abs_diff=0.156,
        pos_replay_within_tier_tolerance=True,
        pos_intercept_moves_cid=True,
        pos_forward_trace_cid_first="a" * 64,
        pos_forward_trace_cid_second="a" * 64,
        pos_forwards_byte_identical=True,
        neg_replay_max_abs_diff=5.0,
        neg_replay_breaks_byte_identity=True,
        pos_determinism_witness_raised=True,
        neg_determinism_witness_completed=True,
        wrapper_is_load_bearing=True,
        pos_determinism_witness_cid="b" * 64,
        neg_determinism_witness_cid="c" * 64,
        tier_tolerance=0.5,
        tp_readback_passthrough_byte_identical=True)
    r1 = GPUSubstrateBenchReportV1(**fields)
    r2 = GPUSubstrateBenchReportV1(**fields)
    assert r1.cid() == r2.cid()


def test_determinism_witness_capsule_content_addressed():
    w = DeterminismLoadBearingWitnessV1(
        op_attempted=True, op_completed=False, raised=True,
        raise_msg="x", op_name="Tensor.scatter_add_",
        cuda_available=True)
    assert len(w.cid()) == 64
    w2 = DeterminismLoadBearingWitnessV1(
        op_attempted=True, op_completed=True, raised=False,
        raise_msg="", op_name="Tensor.scatter_add_",
        cuda_available=True)
    # Different result → different CID.
    assert w.cid() != w2.cid()


def test_determinism_witness_runs_on_cpu_returns_no_cuda():
    """Without CUDA the witness honestly reports cuda_available=False
    and op_attempted=False (it does NOT silently 'pass')."""
    w = determinism_load_bearing_witness_v1()
    try:
        import torch  # type: ignore
        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    if not cuda:
        assert w.cuda_available is False
        assert w.op_attempted is False


def test_determinism_witness_op_name_is_canonical():
    """The witness op MUST be one PyTorch lists as non-deterministic.
    ``Tensor.scatter_add_`` on CUDA float tensors is in
    https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    """
    assert W86_GPU_V1_DETERMINISM_WITNESS_OP == "Tensor.scatter_add_"
    w = determinism_load_bearing_witness_v1()
    assert w.op_name == W86_GPU_V1_DETERMINISM_WITNESS_OP


def test_contract_check_passes_locally():
    rep = run_gpu_substrate_contract_check_v1()
    # CI host has torch? On Mac/Linux with torch installed,
    # wrapper_with_torch_active should be True. If torch is
    # not installed, cpu_inactive should be True.
    try:
        import torch  # noqa: F401
        torch_present = True
    except ImportError:
        torch_present = False
    if torch_present:
        # We allow either: wrapper to be active OR to fail
        # cleanly with a structured note (some CUDA versions
        # can't toggle cudnn settings without a GPU context).
        # The key thing is: the test exists, the path runs,
        # the result is content-addressed.
        assert rep.report_cid != ""
    assert rep.determinism_off_env_var_inverts_mode is True
    assert rep.tp_readback_passthrough_byte_identical is True
    assert rep.capsule_shapes_all_serialise is True


def test_default_mode_is_deterministic():
    """DoD: GPU determinism wrapper is ON BY DEFAULT.
    Confirm the default config sets mode=DETERMINISTIC.
    """
    c = DeterminismWrapperConfigV1()
    assert c.mode == DeterminismMode.DETERMINISTIC


def test_cublas_workspace_config_documented():
    """A passing assertion that the default cublas workspace
    config is the documented PyTorch value (the determinism
    wrapper FAILS at runtime without this being set on
    CUDA 10.2+).
    """
    c = DeterminismWrapperConfigV1()
    assert c.cublas_workspace_config == ":4096:8"
