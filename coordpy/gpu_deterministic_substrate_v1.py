"""W86+ / P2 #44 — GPU/TPU Substrate with Deterministic Replay V1.

Issue #44 asks for a GPU substrate path on top of W80's
``transformers_runtime_v1`` with explicit determinism guarantees:

1. ``device='cuda'`` end-to-end (already shipped in W86's
   `transformers_runtime_v1`).
2. GPU determinism wrapper ON BY DEFAULT (this module).
3. Replay-from-KV byte-identity at the GPU precision floor —
   measured on a real GPU + bench evidence.
4. Hidden-state intercept moves CID on GPU.
5. Negative test: ``deterministic_off`` breaks byte-identity.
6. Tensor-parallel readback contract (single-GPU
   pass-through V1; multi-GPU is V2 stretch).

This module ships the **determinism wrapper contract** and the
**bench harness shape**.  Because Colab Pro browser is the only
GPU surface in this environment, the actual GPU run lives in
``scripts/colab_gpu_deterministic_substrate_w86.ipynb`` (which
generates a content-addressed JSON report verifiable offline
via ``scripts/verify_w86_gpu_substrate_v1_audit_chain.py``).

Honest scope (V1)
-----------------

* ``W86-L-GPU-V1-RESEARCH-ONLY-CAP``
* ``W86-L-GPU-V1-COLAB-PRO-CAP`` — the V1 closure run is on
  Colab Pro A100 / L4. The contract is hardware-agnostic; the
  same bench runs on any CUDA host.
* ``W86-L-GPU-V1-DETERMINISM-WRAPPER-DEFAULT-ON-CAP`` — the
  wrapper sets `torch.use_deterministic_algorithms(True)` AND
  `torch.backends.cudnn.deterministic = True` AND
  `torch.backends.cudnn.benchmark = False` by default. The
  caller can opt out (see DETERMINISM_OFF_ENV_VAR) and the
  negative arm of the bench does this to demonstrate the
  determinism is load-bearing.
* ``W86-L-GPU-V1-PRECISION-TIER-BF16-CAP`` — V1 GPU
  measurements are at bf16 (tier_bf16); the W86 #25 closure
  established the bf16 GPU floor at ≤ 0.5 max_abs_diff.
* ``W86-L-GPU-V1-TENSOR-PARALLEL-V2-CAP`` — V1 ships the
  all-gather contract as a single-GPU pass-through (the
  function exists, has the right shape, and is byte-identical
  when ``world_size = 1``); the multi-GPU run is V2.
* ``W86-L-GPU-V1-NVIDIA-PYTORCH-CAP`` — V1 covers NVIDIA +
  PyTorch. Apple MPS and AMD ROCm are V2 stretch.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import os
from typing import Any, Mapping, Optional, Sequence


W86_GPU_V1_SCHEMA_VERSION: str = (
    "coordpy.gpu_deterministic_substrate_v1.v1"
)

DETERMINISM_OFF_ENV_VAR: str = "W86_GPU_DETERMINISM_OFF"


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            default=str).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------
# Determinism wrapper
# ---------------------------------------------------------------------


class DeterminismMode(enum.Enum):
    """Determinism wrapper state."""

    DETERMINISTIC = "deterministic"
    """Default. `torch.use_deterministic_algorithms(True)`,
    `cudnn.deterministic = True`, `cudnn.benchmark = False`."""

    NON_DETERMINISTIC = "non_deterministic"
    """Explicit opt-out — sets `cudnn.benchmark = True` and
    `cudnn.deterministic = False`. This is what triggers the
    *negative arm* of the bench: GPU runs deterministically
    off, byte-identity fails."""


@dataclasses.dataclass(frozen=True)
class DeterminismWrapperConfigV1:
    """Content-addressed determinism config."""

    mode: DeterminismMode = DeterminismMode.DETERMINISTIC
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    torch_deterministic_algorithms: bool = True
    cublas_workspace_config: str = ":4096:8"
    """Required by CUDA 10.2+ to set
    `torch.use_deterministic_algorithms(True)` cleanly. Standard
    PyTorch documented workspace size."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W86_GPU_V1_SCHEMA_VERSION,
            "mode": self.mode.value,
            "cudnn_deterministic": bool(self.cudnn_deterministic),
            "cudnn_benchmark": bool(self.cudnn_benchmark),
            "torch_deterministic_algorithms": bool(
                self.torch_deterministic_algorithms),
            "cublas_workspace_config": str(
                self.cublas_workspace_config),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_determinism_wrapper_config_v1",
            "config": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class DeterminismWrapperResultV1:
    """Result of applying the determinism wrapper.

    Records the actual settings observed after enabling, so the
    audit chain can confirm the wrapper actually took effect
    (rather than silently failing).
    """

    requested_config: DeterminismWrapperConfigV1
    observed_cudnn_deterministic: bool
    observed_cudnn_benchmark: bool
    observed_torch_deterministic_algorithms: bool
    observed_cublas_workspace_config: str
    wrapper_active: bool
    """True iff every observed setting matches the requested
    config."""

    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W86_GPU_V1_SCHEMA_VERSION,
            "requested_config_cid": (
                self.requested_config.cid()),
            "observed_cudnn_deterministic": bool(
                self.observed_cudnn_deterministic),
            "observed_cudnn_benchmark": bool(
                self.observed_cudnn_benchmark),
            "observed_torch_deterministic_algorithms": bool(
                self.observed_torch_deterministic_algorithms),
            "observed_cublas_workspace_config": str(
                self.observed_cublas_workspace_config),
            "wrapper_active": bool(self.wrapper_active),
            "notes": list(self.notes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_determinism_wrapper_result_v1",
            "result": self.to_dict()})


def apply_determinism_wrapper_v1(
        config: Optional[DeterminismWrapperConfigV1] = None
        ) -> DeterminismWrapperResultV1:
    """Apply the determinism wrapper.

    Imports torch and sets the determinism flags. Returns a
    content-addressed result capsule. If ``torch`` is not
    installed, returns a result with ``wrapper_active=False``
    and an explanatory note — this is the honest CI path.

    Honors ``DETERMINISM_OFF_ENV_VAR``: when the env var is set
    to any truthy value, this function applies the *negative*
    configuration regardless of ``config``. Used by the bench
    negative arm.
    """
    config = config or DeterminismWrapperConfigV1()
    if os.environ.get(DETERMINISM_OFF_ENV_VAR, "").lower() in (
            "1", "true", "yes", "on"):
        config = dataclasses.replace(
            config,
            mode=DeterminismMode.NON_DETERMINISTIC,
            cudnn_deterministic=False,
            cudnn_benchmark=True,
            torch_deterministic_algorithms=False)

    notes: list[str] = []
    try:
        import torch  # type: ignore
    except ImportError:
        return DeterminismWrapperResultV1(
            requested_config=config,
            observed_cudnn_deterministic=False,
            observed_cudnn_benchmark=False,
            observed_torch_deterministic_algorithms=False,
            observed_cublas_workspace_config="",
            wrapper_active=False,
            notes=("torch not installed",))

    if config.mode == DeterminismMode.DETERMINISTIC:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
            config.cublas_workspace_config)
        try:
            torch.use_deterministic_algorithms(
                bool(
                    config.torch_deterministic_algorithms))
        except RuntimeError as e:
            notes.append(
                f"use_deterministic_algorithms raised: {e!r}")
    else:
        # Opt-out: unset deterministic.
        try:
            torch.use_deterministic_algorithms(False)
        except RuntimeError as e:
            notes.append(
                f"use_deterministic_algorithms(False) raised: {e!r}")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ""

    try:
        torch.backends.cudnn.deterministic = bool(
            config.cudnn_deterministic)
        torch.backends.cudnn.benchmark = bool(
            config.cudnn_benchmark)
    except Exception as e:
        notes.append(f"cudnn flags failed: {e!r}")

    observed_cudnn_det = bool(
        getattr(torch.backends.cudnn, "deterministic", False))
    observed_cudnn_bench = bool(
        getattr(torch.backends.cudnn, "benchmark", False))
    # torch.are_deterministic_algorithms_enabled() exists.
    try:
        observed_torch_det = bool(
            torch.are_deterministic_algorithms_enabled())
    except Exception:
        observed_torch_det = False
    observed_cublas = os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG", "")
    wrapper_active = (
        observed_cudnn_det == config.cudnn_deterministic
        and observed_cudnn_bench == config.cudnn_benchmark
        and observed_torch_det == (
            config.torch_deterministic_algorithms))

    return DeterminismWrapperResultV1(
        requested_config=config,
        observed_cudnn_deterministic=observed_cudnn_det,
        observed_cudnn_benchmark=observed_cudnn_bench,
        observed_torch_deterministic_algorithms=observed_torch_det,
        observed_cublas_workspace_config=observed_cublas,
        wrapper_active=wrapper_active,
        notes=tuple(notes))


# ---------------------------------------------------------------------
# Tensor-parallel readback contract
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TensorParallelReadbackV1:
    """All-gather contract for reading hidden states under TP.

    ``world_size = 1`` is the V1 default — a pass-through that
    returns the tensor unchanged.  ``world_size > 1`` invokes
    ``torch.distributed.all_gather`` if available; the V1
    contract documents this as the lift-and-shift path for V2
    when 2-GPU runs become reachable.

    The function shape + audit-chain entry is identical between
    V1 (pass-through) and V2 (real all-gather) so the bench
    re-runs without code changes on a 2-GPU host.
    """

    world_size: int = 1
    backend: str = "nccl"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W86_GPU_V1_SCHEMA_VERSION,
            "world_size": int(self.world_size),
            "backend": str(self.backend),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_tp_readback_v1",
            "config": self.to_dict()})

    def all_gather_hidden_state(self, local_h):
        """Pass-through V1.

        On a 2-GPU host with `torch.distributed` initialised,
        an upgrade replaces the V1 pass-through with a real
        `dist.all_gather` and concatenation.
        """
        if self.world_size <= 1:
            return local_h
        try:
            import torch  # type: ignore
            from torch import distributed as dist  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "TensorParallelReadbackV1 with world_size > 1 "
                "requires torch + torch.distributed") from exc
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialised for "
                "world_size > 1")
        gathered: list = [
            torch.zeros_like(local_h) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_h)
        return torch.cat(gathered, dim=-1)


# ---------------------------------------------------------------------
# Bench report schema (the actual closure JSON)
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GPUSubstrateBenchReportV1:
    """W86 #44 closure bench report schema.

    Populated by ``scripts/colab_gpu_deterministic_substrate_w86.ipynb``
    or any equivalent CUDA-enabled driver, written to disk as
    JSON, and re-verified offline via
    ``scripts/verify_w86_gpu_substrate_v1_audit_chain.py``.
    """

    model_name: str
    device: str
    precision_tier: str
    determinism_wrapper_result_cid: str
    tensor_parallel_config_cid: str
    cuda_device_name: str
    cuda_capability: str
    """E.g. '8.0' for A100."""

    # Positive arm: determinism ON.
    pos_replay_max_abs_diff: float
    pos_replay_within_tier_tolerance: bool
    pos_intercept_moves_cid: bool
    pos_forward_trace_cid_first: str
    pos_forward_trace_cid_second: str
    pos_forwards_byte_identical: bool

    # Negative arm: determinism OFF.
    neg_replay_max_abs_diff: float
    neg_replay_breaks_byte_identity: bool
    """Anti-cheat: when determinism is OFF, byte-identity at
    bf16 tier must NOT hold."""

    # Tier tolerance — single source of truth.
    tier_tolerance: float

    # Tensor-parallel V1 pass-through bool: returns the tensor
    # unchanged when world_size=1.
    tp_readback_passthrough_byte_identical: bool

    report_cid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W86_GPU_V1_SCHEMA_VERSION,
            "model_name": str(self.model_name),
            "device": str(self.device),
            "precision_tier": str(self.precision_tier),
            "determinism_wrapper_result_cid": str(
                self.determinism_wrapper_result_cid),
            "tensor_parallel_config_cid": str(
                self.tensor_parallel_config_cid),
            "cuda_device_name": str(self.cuda_device_name),
            "cuda_capability": str(self.cuda_capability),
            "pos_replay_max_abs_diff": float(round(
                self.pos_replay_max_abs_diff, 12)),
            "pos_replay_within_tier_tolerance": bool(
                self.pos_replay_within_tier_tolerance),
            "pos_intercept_moves_cid": bool(
                self.pos_intercept_moves_cid),
            "pos_forward_trace_cid_first": str(
                self.pos_forward_trace_cid_first),
            "pos_forward_trace_cid_second": str(
                self.pos_forward_trace_cid_second),
            "pos_forwards_byte_identical": bool(
                self.pos_forwards_byte_identical),
            "neg_replay_max_abs_diff": float(round(
                self.neg_replay_max_abs_diff, 12)),
            "neg_replay_breaks_byte_identity": bool(
                self.neg_replay_breaks_byte_identity),
            "tier_tolerance": float(round(
                self.tier_tolerance, 12)),
            "tp_readback_passthrough_byte_identical": bool(
                self.tp_readback_passthrough_byte_identical),
            "report_cid": str(self.report_cid),
        }

    def cid(self) -> str:
        d = self.to_dict()
        d["report_cid"] = ""
        return _sha256_hex({
            "kind": "w86_gpu_substrate_bench_report_v1",
            "report": d})


# ---------------------------------------------------------------------
# CPU-only sanity bench
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GPUSubstrateContractCheckV1:
    """A CPU-only contract check that exercises every wrapper +
    capsule shape without requiring CUDA.

    The actual GPU numbers come from the Colab notebook; this
    contract check is what runs in CI to guarantee the wrapper
    code paths + capsule shapes haven't drifted.
    """

    determinism_wrapper_cpu_inactive: bool
    """Without torch the wrapper reports ``wrapper_active=False``
    + a structured note."""

    determinism_wrapper_with_torch_active: bool
    """With torch installed, applying the default config makes
    the wrapper active."""

    determinism_off_env_var_inverts_mode: bool
    """When the env var is set, mode flips to
    NON_DETERMINISTIC."""

    tp_readback_passthrough_byte_identical: bool
    """world_size=1 returns the tensor unchanged."""

    capsule_shapes_all_serialise: bool
    """All capsule classes round-trip via to_dict()."""

    report_cid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W86_GPU_V1_SCHEMA_VERSION,
            "determinism_wrapper_cpu_inactive": bool(
                self.determinism_wrapper_cpu_inactive),
            "determinism_wrapper_with_torch_active": bool(
                self.determinism_wrapper_with_torch_active),
            "determinism_off_env_var_inverts_mode": bool(
                self.determinism_off_env_var_inverts_mode),
            "tp_readback_passthrough_byte_identical": bool(
                self.tp_readback_passthrough_byte_identical),
            "capsule_shapes_all_serialise": bool(
                self.capsule_shapes_all_serialise),
            "report_cid": str(self.report_cid),
        }

    def cid(self) -> str:
        d = self.to_dict()
        d["report_cid"] = ""
        return _sha256_hex({
            "kind": "w86_gpu_substrate_contract_check_v1",
            "report": d})


def run_gpu_substrate_contract_check_v1() -> (
        GPUSubstrateContractCheckV1):
    """CI-friendly contract check (no GPU required).

    Exercises the wrapper code paths; on a host with torch
    available, applies the wrapper and confirms it took effect.
    """
    # Determinism off env var inverts mode.
    saved = os.environ.pop(DETERMINISM_OFF_ENV_VAR, None)
    try:
        os.environ[DETERMINISM_OFF_ENV_VAR] = "1"
        with_off = apply_determinism_wrapper_v1()
        off_mode_correct = (
            with_off.requested_config.mode
            == DeterminismMode.NON_DETERMINISTIC)
    finally:
        os.environ.pop(DETERMINISM_OFF_ENV_VAR, None)
        if saved is not None:
            os.environ[DETERMINISM_OFF_ENV_VAR] = saved

    # Default (on) — depends on torch presence.
    cpu_inactive = False
    torch_active = False
    try:
        import torch  # noqa: F401  # type: ignore
        on = apply_determinism_wrapper_v1()
        torch_active = on.wrapper_active
    except ImportError:
        # Without torch the wrapper is inactive.
        on = apply_determinism_wrapper_v1()
        cpu_inactive = (not on.wrapper_active)

    # TP pass-through byte-identity (numpy / list path).
    tp = TensorParallelReadbackV1(world_size=1)
    sentinel = object()
    result = tp.all_gather_hidden_state(sentinel)
    tp_passthrough = (result is sentinel)

    # Capsule shapes round-trip.
    shapes_ok = True
    try:
        _ = DeterminismWrapperConfigV1().cid()
        _ = TensorParallelReadbackV1().cid()
        _ = GPUSubstrateBenchReportV1(
            model_name="x", device="cuda:0",
            precision_tier="tier_bf16",
            determinism_wrapper_result_cid="0" * 64,
            tensor_parallel_config_cid="0" * 64,
            cuda_device_name="x", cuda_capability="x",
            pos_replay_max_abs_diff=0.0,
            pos_replay_within_tier_tolerance=True,
            pos_intercept_moves_cid=True,
            pos_forward_trace_cid_first="0" * 64,
            pos_forward_trace_cid_second="0" * 64,
            pos_forwards_byte_identical=True,
            neg_replay_max_abs_diff=99.0,
            neg_replay_breaks_byte_identity=True,
            tier_tolerance=0.5,
            tp_readback_passthrough_byte_identical=True).cid()
    except Exception:
        shapes_ok = False

    rep = GPUSubstrateContractCheckV1(
        determinism_wrapper_cpu_inactive=bool(
            cpu_inactive),
        determinism_wrapper_with_torch_active=bool(torch_active),
        determinism_off_env_var_inverts_mode=bool(
            off_mode_correct),
        tp_readback_passthrough_byte_identical=bool(
            tp_passthrough),
        capsule_shapes_all_serialise=bool(shapes_ok))
    rep = dataclasses.replace(rep, report_cid=rep.cid())
    return rep


__all__ = [
    "W86_GPU_V1_SCHEMA_VERSION",
    "DETERMINISM_OFF_ENV_VAR",
    "DeterminismMode",
    "DeterminismWrapperConfigV1",
    "DeterminismWrapperResultV1",
    "TensorParallelReadbackV1",
    "GPUSubstrateBenchReportV1",
    "GPUSubstrateContractCheckV1",
    "apply_determinism_wrapper_v1",
    "run_gpu_substrate_contract_check_v1",
]
