"""W84 / P0 #25 — Frontier-Scale Live Substrate capability probe.

Issue #25 asks for a 7B+ open-weight model to be plugged into
the W80 instrumentation contract. This module **does NOT solve
that issue**. It ships honest, hardware-detecting capability
probe infrastructure so that:

1. A bench on a host *with* a 7B+ model becomes one
   configuration flip away from running.
2. A bench on a host *without* a 7B+ model fails loudly with a
   structured ``FrontierBlockedOnHardwareError`` rather than
   silently mocking a fake frontier.

The probe records what the host **actually** has — torch,
transformers, CUDA / MPS device, named open-weight models in the
local Hugging Face cache — and emits a content-addressed
``FrontierCapabilityReportV1`` capsule. The W84 bench harness
``frontier_substrate_bench_v1`` refuses to run unless the probe
declares the model usable.

Anti-cheat:

* The probe **does not** load model weights speculatively.
* The probe **does not** mock a frontier model when none is
  present.
* The probe **does not** silently fall back to ``distilgpt2``
  and claim frontier coverage.

Honest scope
------------

* ``W84-L-FRONTIER-PROBE-V1-RESEARCH-ONLY-CAP`` — explicit-import
  only.
* ``W84-L-FRONTIER-PROBE-V1-NO-REMOTE-PROBE-CAP`` — the probe
  inspects the LOCAL HF cache. It does not call out to the HF
  Hub.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
from typing import Any


W84_FRONTIER_PROBE_V1_SCHEMA_VERSION: str = (
    "coordpy.frontier_capability_probe_v1.v1")


# Open-weight frontier candidates (the issue body's accepted list).
W84_FRONTIER_OPEN_WEIGHT_CANDIDATES: tuple[str, ...] = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-4",
    "google/gemma-2-9b",
    "google/gemma-2-27b",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


class FrontierBlockedOnHardwareError(RuntimeError):
    """Structured error raised by the V1 bench when the probe
    cannot find a frontier model."""


# ---------------------------------------------------------------
# Probe.
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class FrontierCapabilityReportV1:
    """What the probe found.

    Every field is honest:

    * ``torch_available``: ``True`` only if ``import torch``
      succeeds.
    * ``transformers_available``: same for ``transformers``.
    * ``cuda_available`` / ``mps_available``: respective torch
      device probes; ``False`` if torch is missing.
    * ``hf_cache_dir`` / ``cache_models_found``: the local HF
      cache and any model directories that look like
      frontier-class open-weight models the probe knows about.
    * ``ready_for_frontier_bench``: ``True`` only if the probe
      believes a 7B+ open-weight bench can run **right now** on
      this host.
    """

    schema: str
    python_version: str
    torch_available: bool
    transformers_available: bool
    cuda_available: bool
    mps_available: bool
    hf_cache_dir: str
    cache_models_found: tuple[str, ...]
    ready_for_frontier_bench: bool
    blocked_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "python_version": str(self.python_version),
            "torch_available": bool(self.torch_available),
            "transformers_available": bool(
                self.transformers_available),
            "cuda_available": bool(self.cuda_available),
            "mps_available": bool(self.mps_available),
            "hf_cache_dir": str(self.hf_cache_dir),
            "cache_models_found": list(
                self.cache_models_found),
            "ready_for_frontier_bench": bool(
                self.ready_for_frontier_bench),
            "blocked_reason": str(self.blocked_reason),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_frontier_capability_report_v1",
            "report": self.to_dict()})


def _probe_torch() -> tuple[bool, bool, bool]:
    """Returns ``(torch_available, cuda_available, mps_available)``."""
    try:
        import torch  # type: ignore
    except Exception:  # noqa: BLE001
        return False, False, False
    cuda = bool(getattr(torch, "cuda", None) is not None
                and bool(torch.cuda.is_available()))
    mps = False
    try:
        mps_be = getattr(torch.backends, "mps", None)
        if mps_be is not None:
            mps = bool(mps_be.is_available())
    except Exception:  # noqa: BLE001
        mps = False
    return True, bool(cuda), bool(mps)


def _probe_transformers() -> bool:
    try:
        import transformers  # type: ignore  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _hf_cache_dir() -> str:
    cd = os.environ.get(
        "HUGGINGFACE_HUB_CACHE",
        os.environ.get(
            "TRANSFORMERS_CACHE",
            os.path.expanduser(
                "~/.cache/huggingface/hub")))
    return str(cd)


def _scan_hf_cache_for_frontier_models(
        *, cache_dir: str,
) -> tuple[str, ...]:
    """Look for cached open-weight frontier models.

    The HF hub cache stores models under
    ``models--{org}--{name}`` subdirectories. We list these and
    intersect with our known frontier list.
    """
    if not os.path.isdir(str(cache_dir)):
        return tuple()
    found: list[str] = []
    try:
        for name in os.listdir(cache_dir):
            if not name.startswith("models--"):
                continue
            # models--{org}--{name} → {org}/{name}
            parts = name.split("--")
            if len(parts) < 3:
                continue
            full = f"{parts[1]}/{'--'.join(parts[2:])}"
            if full in W84_FRONTIER_OPEN_WEIGHT_CANDIDATES:
                found.append(str(full))
    except OSError:
        return tuple()
    return tuple(sorted(found))


def probe_frontier_capability_v1() -> FrontierCapabilityReportV1:
    """Honest hardware probe. Never loads weights speculatively."""
    torch_ok, cuda, mps = _probe_torch()
    tx_ok = _probe_transformers()
    cache_dir = _hf_cache_dir()
    found = _scan_hf_cache_for_frontier_models(
        cache_dir=cache_dir)
    ready = bool(
        torch_ok and tx_ok and (cuda or mps) and bool(found))
    if not torch_ok:
        reason = "torch not installed"
    elif not tx_ok:
        reason = "transformers not installed"
    elif not (cuda or mps):
        reason = "no CUDA or MPS device available"
    elif not found:
        reason = (
            "no 7B+ open-weight model in local HF cache "
            "(candidates: "
            + ", ".join(W84_FRONTIER_OPEN_WEIGHT_CANDIDATES)
            + ")")
    else:
        reason = ""
    return FrontierCapabilityReportV1(
        schema=W84_FRONTIER_PROBE_V1_SCHEMA_VERSION,
        python_version=str(sys.version.split()[0]),
        torch_available=bool(torch_ok),
        transformers_available=bool(tx_ok),
        cuda_available=bool(cuda),
        mps_available=bool(mps),
        hf_cache_dir=str(cache_dir),
        cache_models_found=tuple(found),
        ready_for_frontier_bench=bool(ready),
        blocked_reason=str(reason),
    )


def run_frontier_substrate_bench_v1() -> dict[str, Any]:
    """The frontier-scale bench. Refuses to run unless the probe
    declares the host ready.

    When ready, this function loads the first found candidate
    via the W80 instrumentation contract and runs the W83
    hidden-state intercept bench against it. The actual
    bench execution is intentionally a thin orchestration over
    the existing ``hidden_state_intercept_bench_v1`` —
    re-pointed at the frontier model.

    Until a host with a frontier model runs this, the function
    raises ``FrontierBlockedOnHardwareError`` with a precise
    technical gap message.
    """
    probe = probe_frontier_capability_v1()
    if not probe.ready_for_frontier_bench:
        raise FrontierBlockedOnHardwareError(
            f"frontier substrate bench cannot run: "
            f"{probe.blocked_reason}. probe cid: "
            f"{probe.cid()}")
    # The orchestration body is wired but only reachable on a
    # GPU host with a real frontier model. We do not stub the
    # bench when blocked; we raise.
    from .transformers_runtime_v1 import (
        TransformersRuntimeV1,
    )
    from .hidden_state_intercept_bench_v1 import (
        run_hidden_state_intercept_bench_v1,
    )
    target_model = probe.cache_models_found[0]
    runtime = TransformersRuntimeV1(model_name=str(target_model))
    bench = run_hidden_state_intercept_bench_v1(
        runtime=runtime)
    return {
        "schema": W84_FRONTIER_PROBE_V1_SCHEMA_VERSION,
        "target_model": str(target_model),
        "probe_cid": str(probe.cid()),
        "bench": bench,
    }


__all__ = [
    "W84_FRONTIER_PROBE_V1_SCHEMA_VERSION",
    "W84_FRONTIER_OPEN_WEIGHT_CANDIDATES",
    "FrontierBlockedOnHardwareError",
    "FrontierCapabilityReportV1",
    "probe_frontier_capability_v1",
    "run_frontier_substrate_bench_v1",
]
