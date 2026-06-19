"""W79 — Substrate Adapter V24.

Strictly extends W78's ``coordpy.substrate_adapter_v23``. V24
adds three new capability axes that the W79 V24 substrate (and
the controlled-runtime substrate V1) satisfy and hosted backends
do not:

* ``replacement_then_restart_after_long_delay_trajectory_cid``
* ``replacement_then_restart_after_long_delay_length_per_layer``
* ``replacement_then_restart_after_long_delay_pressure_gate_per_layer``

V24 adds two new top tiers:

* ``substrate_v24_full`` — the in-repo W79 V24 substrate.
* ``controlled_runtime_substrate_v1`` — the W79 controlled
  runtime substrate (real hidden state / KV / attention).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from typing import Any

from .substrate_adapter import (
    SUBSTRATE_TIER_EMBEDDINGS_ONLY,
    SUBSTRATE_TIER_LOGITS_ONLY,
    SUBSTRATE_TIER_TEXT_ONLY,
    SUBSTRATE_TIER_UNREACHABLE,
)
from .substrate_adapter_v23 import (
    SubstrateCapabilityV23, W78_SUBSTRATE_TIER_SUBSTRATE_V23_FULL,
    W78_SUBSTRATE_V23_CAPABILITY_AXES,
    W78_SUBSTRATE_V23_NEW_AXES,
)


W79_SUBSTRATE_ADAPTER_V24_SCHEMA_VERSION: str = (
    "coordpy.substrate_adapter_v24.v1")

W79_SUBSTRATE_TIER_SUBSTRATE_V24_FULL: str = "substrate_v24_full"
W79_SUBSTRATE_TIER_CONTROLLED_RUNTIME_V1: str = (
    "controlled_runtime_substrate_v1")

W79_SUBSTRATE_V24_NEW_AXES: tuple[str, ...] = (
    "replacement_then_restart_after_long_delay_trajectory_cid",
    "replacement_then_restart_after_long_delay_length_per_layer",
    "replacement_then_restart_after_long_delay_pressure_gate_per_layer",
)

W79_CONTROLLED_RUNTIME_AXES_AS_CAPABILITIES: tuple[str, ...] = (
    "controlled_runtime_hidden_state",
    "controlled_runtime_kv_cache",
    "controlled_runtime_attention_probs",
    "controlled_runtime_per_head_attention_bias",
    "controlled_runtime_prefix_state_inject",
    "controlled_runtime_per_layer_logits",
    "controlled_runtime_replay_from_kv",
)

W79_SUBSTRATE_V24_CAPABILITY_AXES: tuple[str, ...] = (
    *W78_SUBSTRATE_V23_CAPABILITY_AXES,
    *W79_SUBSTRATE_V24_NEW_AXES,
    *W79_CONTROLLED_RUNTIME_AXES_AS_CAPABILITIES,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class SubstrateCapabilityV24:
    backend_name: str
    backend_url: str
    capabilities: tuple[tuple[str, str], ...]
    tier: str
    probe_notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W79_SUBSTRATE_ADAPTER_V24_SCHEMA_VERSION,
            "backend_name": str(self.backend_name),
            "backend_url": str(self.backend_url),
            "capabilities": [
                [str(ax), str(val)]
                for ax, val in self.capabilities],
            "tier": str(self.tier),
            "probe_notes": [str(n) for n in self.probe_notes],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "substrate_capability_v24",
            "capability": self.to_dict()})


def _decide_tier_v24(caps: dict[str, str]) -> str:
    if caps.get("text") != "yes":
        return SUBSTRATE_TIER_UNREACHABLE
    has_v24 = all(
        caps.get(ax) == "yes"
        for ax in W79_SUBSTRATE_V24_NEW_AXES)
    has_v23 = all(
        caps.get(ax) == "yes"
        for ax in W78_SUBSTRATE_V23_NEW_AXES)
    has_ctrl = all(
        caps.get(ax) == "yes"
        for ax in W79_CONTROLLED_RUNTIME_AXES_AS_CAPABILITIES)
    if has_ctrl and has_v24 and has_v23:
        return W79_SUBSTRATE_TIER_SUBSTRATE_V24_FULL
    if has_ctrl:
        return W79_SUBSTRATE_TIER_CONTROLLED_RUNTIME_V1
    if has_v23:
        return W78_SUBSTRATE_TIER_SUBSTRATE_V23_FULL
    if caps.get("logits") == "yes":
        return SUBSTRATE_TIER_LOGITS_ONLY
    if caps.get("embeddings") == "yes":
        return SUBSTRATE_TIER_EMBEDDINGS_ONLY
    return SUBSTRATE_TIER_TEXT_ONLY


def probe_tiny_substrate_v24_adapter(
        *, label: str = "tiny_substrate_v24",
) -> SubstrateCapabilityV24:
    caps = {ax: "yes" for ax in W79_SUBSTRATE_V24_CAPABILITY_AXES}
    return SubstrateCapabilityV24(
        backend_name=str(label),
        backend_url="in-process://coordpy.tiny_substrate_v24",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W79_SUBSTRATE_V24_CAPABILITY_AXES),
        tier=_decide_tier_v24(caps),
        probe_notes=(
            "tiny in-repo numpy V24 substrate + per-turn "
            "replacement-then-restart-after-long-delay trajectory "
            "CID + per-layer length + pressure gate. Plus the W79 "
            "controlled runtime substrate (hidden / KV / "
            "attention).",
            "still NOT a frontier model; still does NOT prove "
            "third-party substrate access — but the W79 "
            "controlled runtime IS a real second substrate "
            "backend that we control end-to-end.",
        ),
    )


def probe_controlled_runtime_v1_adapter(
        *, label: str = "controlled_runtime_substrate_v1",
) -> SubstrateCapabilityV24:
    caps = {ax: "no" for ax in W79_SUBSTRATE_V24_CAPABILITY_AXES}
    caps["text"] = "yes"
    for ax in W79_CONTROLLED_RUNTIME_AXES_AS_CAPABILITIES:
        caps[ax] = "yes"
    return SubstrateCapabilityV24(
        backend_name=str(label),
        backend_url=(
            "in-process://coordpy.controlled_runtime_substrate_v1"),
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W79_SUBSTRATE_V24_CAPABILITY_AXES),
        tier=_decide_tier_v24(caps),
        probe_notes=(
            "controlled-runtime substrate V1: real transformer-"
            "class forward with hidden state, KV cache, attention "
            "probabilities, per-head attention bias, prefix state, "
            "per-layer logits, byte-stable replay-from-KV.",
            "W79 direct-blocker-attack pillar 1: we built a "
            "second substrate runtime we control.",
        ),
    )


def probe_v23_substrate_adapter_as_v24(
        cap: SubstrateCapabilityV23,
) -> SubstrateCapabilityV24:
    base = {ax: val for ax, val in cap.capabilities}
    for ax in W79_SUBSTRATE_V24_NEW_AXES:
        base.setdefault(ax, "no")
    for ax in W79_CONTROLLED_RUNTIME_AXES_AS_CAPABILITIES:
        base.setdefault(ax, "no")
    tier = _decide_tier_v24(base)
    return SubstrateCapabilityV24(
        backend_name=str(cap.backend_name),
        backend_url=str(cap.backend_url),
        capabilities=tuple(
            (ax, base.get(ax, "no"))
            for ax in W79_SUBSTRATE_V24_CAPABILITY_AXES),
        tier=str(tier),
        probe_notes=tuple(cap.probe_notes) + (
            "wrapped from W78 substrate adapter V23",),
    )


@dataclasses.dataclass(frozen=True)
class SubstrateAdapterV24Matrix:
    probed_at_ns: int
    capabilities: tuple[SubstrateCapabilityV24, ...]

    def by_name(self) -> dict[str, SubstrateCapabilityV24]:
        return {c.backend_name: c for c in self.capabilities}

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W79_SUBSTRATE_ADAPTER_V24_SCHEMA_VERSION,
            "probed_at_ns": int(self.probed_at_ns),
            "capabilities": [
                c.to_dict() for c in self.capabilities],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "substrate_adapter_v24_matrix",
            "capabilities": [
                c.cid() for c in self.capabilities],
        })


def probe_all_v24_adapters() -> SubstrateAdapterV24Matrix:
    caps: list[SubstrateCapabilityV24] = []
    caps.append(probe_tiny_substrate_v24_adapter())
    caps.append(probe_controlled_runtime_v1_adapter())
    return SubstrateAdapterV24Matrix(
        probed_at_ns=int(time.time_ns()),
        capabilities=tuple(caps))


__all__ = [
    "W79_SUBSTRATE_ADAPTER_V24_SCHEMA_VERSION",
    "W79_SUBSTRATE_TIER_SUBSTRATE_V24_FULL",
    "W79_SUBSTRATE_TIER_CONTROLLED_RUNTIME_V1",
    "W79_SUBSTRATE_V24_NEW_AXES",
    "W79_SUBSTRATE_V24_CAPABILITY_AXES",
    "W79_CONTROLLED_RUNTIME_AXES_AS_CAPABILITIES",
    "SubstrateCapabilityV24",
    "probe_tiny_substrate_v24_adapter",
    "probe_controlled_runtime_v1_adapter",
    "probe_v23_substrate_adapter_as_v24",
    "SubstrateAdapterV24Matrix",
    "probe_all_v24_adapters",
]
