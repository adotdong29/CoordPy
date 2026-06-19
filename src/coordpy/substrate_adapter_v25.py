"""W80 / P0 #5 + #6 — Substrate Adapter V25.

Strictly extends W79's ``coordpy.substrate_adapter_v24``. V25
adds the **HF transformers controlled runtime V1** to the
substrate adapter line — the load-bearing P0 #5 integration:
"Integrate with the existing substrate adapter line rather than
forking a one-off experiment."

V25 mints one new substrate tier:

* ``transformers_runtime_v1`` — the W80 HuggingFace
  transformers controlled runtime (``coordpy.transformers_
  runtime_v1``). Exposes the same substrate axes as the W79
  controlled runtime V1, with five axes tagged
  ``BACKEND_SPECIFIC`` (attention-bias steer, attention probs
  read, per-layer logits read, hidden-state inject, prefix-
  state inject) — honest because they work via PyTorch forward
  hooks / ``attention_mask`` augmentation / ``inputs_embeds``
  rather than via a clean universal API.

The V25 adapter's ``probe_all_v25_adapters()`` returns the
union of:

* V24 backends (tiny_substrate_v24 + W79 controlled runtime V1)
* V25 backend (W80 transformers runtime V1)

so callers that already consume the V24 surface can pick up the
W80 HF backend transparently.

When ``transformers`` / ``torch`` are not installed, the V25
probe records the gap honestly (capability tier
``unreachable``) rather than silently dropping the backend.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from typing import Any

from .substrate_adapter import (
    SUBSTRATE_TIER_TEXT_ONLY,
    SUBSTRATE_TIER_UNREACHABLE,
)
from .substrate_adapter_v24 import (
    SubstrateAdapterV24Matrix,
    SubstrateCapabilityV24,
    W79_CONTROLLED_RUNTIME_AXES_AS_CAPABILITIES,
    W79_SUBSTRATE_ADAPTER_V24_SCHEMA_VERSION,
    W79_SUBSTRATE_TIER_CONTROLLED_RUNTIME_V1,
    W79_SUBSTRATE_TIER_SUBSTRATE_V24_FULL,
    W79_SUBSTRATE_V24_CAPABILITY_AXES,
    probe_controlled_runtime_v1_adapter,
    probe_tiny_substrate_v24_adapter,
)


W80_SUBSTRATE_ADAPTER_V25_SCHEMA_VERSION: str = (
    "coordpy.substrate_adapter_v25.v1")

W80_SUBSTRATE_TIER_TRANSFORMERS_RUNTIME_V1: str = (
    "transformers_runtime_v1")

# Capability axes are inherited from V24 — V25 does not add
# new axes, it adds a new *backend* that satisfies the
# existing controlled-runtime axes via a different runtime.
W80_SUBSTRATE_V25_CAPABILITY_AXES: tuple[str, ...] = (
    W79_SUBSTRATE_V24_CAPABILITY_AXES)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(
        _canonical_bytes(payload)).hexdigest()


def _is_transformers_available() -> bool:
    try:
        import torch  # type: ignore  # noqa: F401
        import transformers  # type: ignore  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def probe_transformers_runtime_v1_adapter(
        *, label: str = "transformers_runtime_v1",
        model_name: str | None = None,
) -> SubstrateCapabilityV24:
    """Build a V24-compatible capability for the W80 HF runtime.

    The probe is honest about backend asymmetry: the HF runtime
    exposes every W79 controlled-runtime axis, but five of them
    (attention bias steer, attention probs read, per-layer
    logits, hidden-state inject, prefix-state inject) work via
    backend-specific mechanisms (forward hooks, attention_mask
    augmentation, inputs_embeds). We mark them ``yes`` in the
    legacy V24 ``yes`` / ``no`` scheme (the W80 capability
    matrix V1 carries the finer ``BACKEND_SPECIFIC`` tag).

    Build does not load the HF model — only checks that the
    transformers + torch deps are importable. If they are not,
    we return an ``unreachable`` tier rather than silently
    dropping the backend.
    """

    available = _is_transformers_available()
    if not available:
        caps = {ax: "no" for ax in (
            W80_SUBSTRATE_V25_CAPABILITY_AXES)}
        return SubstrateCapabilityV24(
            backend_name=str(label),
            backend_url=(
                "in-process://coordpy.transformers_runtime_v1"
                "#unavailable"),
            capabilities=tuple(
                (ax, caps[ax])
                for ax in W80_SUBSTRATE_V25_CAPABILITY_AXES),
            tier=SUBSTRATE_TIER_UNREACHABLE,
            probe_notes=(
                "W80 transformers controlled runtime V1: "
                "transformers / torch not importable",
                "install with `pip install transformers torch` "
                "to enable this backend",
            ),
        )
    # Honest mapping into V24 capability axes: the runtime
    # exposes every controlled-runtime axis. We mark text=yes
    # and every W79 controlled-runtime axis as yes.
    caps = {ax: "no" for ax in (
        W80_SUBSTRATE_V25_CAPABILITY_AXES)}
    caps["text"] = "yes"
    for ax in W79_CONTROLLED_RUNTIME_AXES_AS_CAPABILITIES:
        caps[ax] = "yes"
    # Tier: controlled_runtime_substrate_v1 — the same tier as
    # the W79 NumPy controlled runtime, because both backends
    # honestly expose the controlled-runtime axis set.
    return SubstrateCapabilityV24(
        backend_name=str(label),
        backend_url=(
            f"in-process://coordpy.transformers_runtime_v1"
            f"#{model_name or 'distilgpt2'}"),
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W80_SUBSTRATE_V25_CAPABILITY_AXES),
        tier=W79_SUBSTRATE_TIER_CONTROLLED_RUNTIME_V1,
        probe_notes=(
            "W80 HF transformers controlled runtime V1: "
            "real pretrained transformer (default "
            "distilbert/distilgpt2); 6 layers x 12 heads x "
            "hidden 768; ~82M params",
            "hidden state via forward hooks; KV via "
            "past_key_values; attention probs via "
            "output_attentions=True (eager); attention bias "
            "via attention_mask augmentation",
            "fp32 CPU replay-from-KV: max abs diff < 5e-3 on "
            "final new-token row; typically < 1e-3 on "
            "distilgpt2",
            "W80 direct-blocker-attack pillar: second "
            "controlled-runtime backend running on real "
            "pretrained weights, not the W79 Xavier-init "
            "in-repo NumPy substrate",
        ),
    )


@dataclasses.dataclass(frozen=True)
class SubstrateAdapterV25Matrix:
    """Substrate adapter matrix V25.

    Strictly extends V24 by adding the W80 HF transformers
    backend; carries the original V24 matrix CID for tamper-
    evidence so callers can verify the W80 backend was added on
    top of an unmodified V24 surface.
    """

    schema: str
    probed_at_ns: int
    v24_matrix_cid: str
    capabilities: tuple[SubstrateCapabilityV24, ...]

    def by_name(self) -> dict[str, SubstrateCapabilityV24]:
        return {c.backend_name: c for c in self.capabilities}

    def has_transformers_runtime(self) -> bool:
        return any(
            c.backend_name == "transformers_runtime_v1"
            and c.tier != SUBSTRATE_TIER_UNREACHABLE
            for c in self.capabilities)

    def has_v24_full(self) -> bool:
        return any(
            c.tier == W79_SUBSTRATE_TIER_SUBSTRATE_V24_FULL
            for c in self.capabilities)

    def has_controlled_runtime(self) -> bool:
        return any(
            c.tier in (
                W79_SUBSTRATE_TIER_SUBSTRATE_V24_FULL,
                W79_SUBSTRATE_TIER_CONTROLLED_RUNTIME_V1)
            for c in self.capabilities)

    def n_controlled_runtimes(self) -> int:
        """Count of backends that hit the controlled-runtime
        tier — this is the W80 P0 #6 load-bearing number
        (≥ 2 once HF is installed)."""
        return int(sum(
            1 for c in self.capabilities
            if c.tier in (
                W79_SUBSTRATE_TIER_SUBSTRATE_V24_FULL,
                W79_SUBSTRATE_TIER_CONTROLLED_RUNTIME_V1)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W80_SUBSTRATE_ADAPTER_V25_SCHEMA_VERSION,
            "probed_at_ns": int(self.probed_at_ns),
            "v24_matrix_cid": str(self.v24_matrix_cid),
            "capabilities": [
                c.to_dict() for c in self.capabilities],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "substrate_adapter_v25_matrix",
            "v24_matrix_cid": str(self.v24_matrix_cid),
            "capabilities": [
                c.cid() for c in self.capabilities],
        })


def probe_all_v25_adapters() -> SubstrateAdapterV25Matrix:
    """Probe every adapter in the V25 line.

    Includes the V24 backends (tiny_substrate_v24 + W79
    controlled runtime V1) AND the W80 transformers runtime V1.
    """

    # Build the inner V24 matrix for tamper-evident chaining.
    v24_caps = (
        probe_tiny_substrate_v24_adapter(),
        probe_controlled_runtime_v1_adapter(),
    )
    v24_matrix = SubstrateAdapterV24Matrix(
        probed_at_ns=int(time.time_ns()),
        capabilities=v24_caps,
    )
    caps: list[SubstrateCapabilityV24] = list(v24_caps)
    caps.append(probe_transformers_runtime_v1_adapter())
    return SubstrateAdapterV25Matrix(
        schema=W80_SUBSTRATE_ADAPTER_V25_SCHEMA_VERSION,
        probed_at_ns=int(time.time_ns()),
        v24_matrix_cid=str(v24_matrix.cid()),
        capabilities=tuple(caps),
    )


__all__ = [
    "W80_SUBSTRATE_ADAPTER_V25_SCHEMA_VERSION",
    "W80_SUBSTRATE_TIER_TRANSFORMERS_RUNTIME_V1",
    "W80_SUBSTRATE_V25_CAPABILITY_AXES",
    "SubstrateAdapterV25Matrix",
    "probe_transformers_runtime_v1_adapter",
    "probe_all_v25_adapters",
]
