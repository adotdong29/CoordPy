"""W78 — Substrate Adapter V23.

Strictly extends W77's ``coordpy.substrate_adapter_v22``. V23 adds
three new capability axes that the W78 V23 substrate satisfies and
hosted backends do not:

  * ``long_horizon_reconstruction_trajectory_cid``
  * ``long_horizon_reconstruction_length_per_layer``
  * ``long_horizon_reconstruction_pressure_gate_per_layer``

V23 adds a new top tier:

  * ``substrate_v23_full`` — only the W78 V23 in-repo runtime
    satisfies every axis.

``W78-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from .substrate_adapter import (
    SUBSTRATE_TIER_EMBEDDINGS_ONLY,
    SUBSTRATE_TIER_LOGITS_ONLY,
    SUBSTRATE_TIER_TEXT_ONLY,
    SUBSTRATE_TIER_UNREACHABLE,
    probe_ollama_adapter,
    probe_openai_compatible_adapter,
)
from .substrate_adapter_v21 import SubstrateCapabilityV21
from .substrate_adapter_v22 import (
    SubstrateCapabilityV22,
    W77_SUBSTRATE_TIER_SUBSTRATE_V22_FULL,
    W77_SUBSTRATE_V22_CAPABILITY_AXES,
    W77_SUBSTRATE_V22_NEW_AXES,
    probe_v21_substrate_adapter_as_v22,
)
from .tiny_substrate_v3 import _sha256_hex


W78_SUBSTRATE_ADAPTER_V23_SCHEMA_VERSION: str = (
    "coordpy.substrate_adapter_v23.v1")

W78_SUBSTRATE_TIER_SUBSTRATE_V23_FULL: str = "substrate_v23_full"

W78_SUBSTRATE_V23_NEW_AXES: tuple[str, ...] = (
    "long_horizon_reconstruction_trajectory_cid",
    "long_horizon_reconstruction_length_per_layer",
    "long_horizon_reconstruction_pressure_gate_per_layer",
)

W78_SUBSTRATE_V23_CAPABILITY_AXES: tuple[str, ...] = (
    *W77_SUBSTRATE_V22_CAPABILITY_AXES,
    *W78_SUBSTRATE_V23_NEW_AXES,
)


@dataclasses.dataclass(frozen=True)
class SubstrateCapabilityV23:
    backend_name: str
    backend_url: str
    capabilities: tuple[tuple[str, str], ...]
    tier: str
    probe_notes: tuple[str, ...]

    def cap(self, axis: str) -> str:
        for ax, val in self.capabilities:
            if ax == axis:
                return val
        return "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W78_SUBSTRATE_ADAPTER_V23_SCHEMA_VERSION,
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
            "kind": "substrate_capability_v23",
            "capability": self.to_dict()})


def _decide_tier_v23(caps: dict[str, str]) -> str:
    if caps.get("text") != "yes":
        return SUBSTRATE_TIER_UNREACHABLE
    has_v23 = all(
        caps.get(ax) == "yes"
        for ax in W78_SUBSTRATE_V23_NEW_AXES)
    has_v22 = all(
        caps.get(ax) == "yes"
        for ax in W77_SUBSTRATE_V22_NEW_AXES)
    if has_v23 and has_v22:
        return W78_SUBSTRATE_TIER_SUBSTRATE_V23_FULL
    if has_v22:
        return W77_SUBSTRATE_TIER_SUBSTRATE_V22_FULL
    if caps.get("logits") == "yes":
        return SUBSTRATE_TIER_LOGITS_ONLY
    if caps.get("embeddings") == "yes":
        return SUBSTRATE_TIER_EMBEDDINGS_ONLY
    return SUBSTRATE_TIER_TEXT_ONLY


def probe_tiny_substrate_v23_adapter(
        *, label: str = "tiny_substrate_v23",
) -> SubstrateCapabilityV23:
    caps = {ax: "yes" for ax in W78_SUBSTRATE_V23_CAPABILITY_AXES}
    return SubstrateCapabilityV23(
        backend_name=str(label),
        backend_url="in-process://coordpy.tiny_substrate_v23",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W78_SUBSTRATE_V23_CAPABILITY_AXES),
        tier=_decide_tier_v23(caps),
        probe_notes=(
            "tiny in-repo numpy transformer V23 + V22 axes + per-"
            "turn long-horizon-reconstruction trajectory CID + "
            "per-layer long-horizon-reconstruction length label + "
            "per-layer long-horizon-reconstruction-pressure gate + "
            "KV bridge V23 nineteen-target ridge + cache V21 "
            "eighteen-objective ridge + replay V19 twenty-six-"
            "regime ridge + long-horizon-reconstruction-aware "
            "routing head + deep substrate hybrid V23 twenty-"
            "three-way loop + multi-agent substrate coordinator "
            "V14 + team-consensus controller V13",
            "still NOT a frontier model; still does NOT prove "
            "third-party substrate access",
        ),
    )


def probe_v22_substrate_adapter_as_v23(
        cap: SubstrateCapabilityV22,
) -> SubstrateCapabilityV23:
    base = {ax: val for ax, val in cap.capabilities}
    for ax in W78_SUBSTRATE_V23_NEW_AXES:
        base.setdefault(ax, "no")
    tier = _decide_tier_v23(base)
    return SubstrateCapabilityV23(
        backend_name=str(cap.backend_name),
        backend_url=str(cap.backend_url),
        capabilities=tuple(
            (ax, base.get(ax, "no"))
            for ax in W78_SUBSTRATE_V23_CAPABILITY_AXES),
        tier=str(tier),
        probe_notes=tuple(cap.probe_notes) + (
            "wrapped from W77 substrate adapter V22",),
    )


def probe_synthetic_v23_adapter(
        *, label: str = "synthetic",
) -> SubstrateCapabilityV23:
    caps = {ax: "no" for ax in W78_SUBSTRATE_V23_CAPABILITY_AXES}
    caps["text"] = "yes"
    return SubstrateCapabilityV23(
        backend_name=str(label),
        backend_url="in-process://coordpy.synthetic_llm",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W78_SUBSTRATE_V23_CAPABILITY_AXES),
        tier=_decide_tier_v23(caps),
        probe_notes=(
            "synthetic deterministic backend; no substrate "
            "access; for hermetic capsule-layer testing only",),
    )


@dataclasses.dataclass(frozen=True)
class SubstrateAdapterV23Matrix:
    probed_at_ns: int
    capabilities: tuple[SubstrateCapabilityV23, ...]

    def by_name(self) -> dict[str, SubstrateCapabilityV23]:
        return {c.backend_name: c for c in self.capabilities}

    def has_v23_full(self) -> bool:
        return any(
            c.tier == W78_SUBSTRATE_TIER_SUBSTRATE_V23_FULL
            for c in self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W78_SUBSTRATE_ADAPTER_V23_SCHEMA_VERSION,
            "probed_at_ns": int(self.probed_at_ns),
            "capabilities": [
                c.to_dict() for c in self.capabilities],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "substrate_adapter_v23_matrix",
            "capabilities": [
                c.cid() for c in self.capabilities],
        })


def probe_all_v23_adapters(
        *, probe_ollama: bool = False,
        probe_openai: bool = False,
        ollama_url: str | None = None,
) -> SubstrateAdapterV23Matrix:
    caps: list[SubstrateCapabilityV23] = []
    caps.append(probe_tiny_substrate_v23_adapter())
    caps.append(probe_synthetic_v23_adapter())
    if probe_ollama:
        oll = probe_ollama_adapter(
            ollama_url=ollama_url
            or "http://localhost:11434")
        v21 = SubstrateCapabilityV21(
            backend_name=str(oll.backend_name),
            backend_url=str(oll.backend_url),
            capabilities=tuple(
                (ax, val) for ax, val in oll.capabilities),
            tier=str(oll.tier),
            probe_notes=tuple(oll.probe_notes))
        v22 = probe_v21_substrate_adapter_as_v22(v21)
        caps.append(probe_v22_substrate_adapter_as_v23(v22))
    if probe_openai:
        oai = probe_openai_compatible_adapter()
        v21 = SubstrateCapabilityV21(
            backend_name=str(oai.backend_name),
            backend_url=str(oai.backend_url),
            capabilities=tuple(
                (ax, val) for ax, val in oai.capabilities),
            tier=str(oai.tier),
            probe_notes=tuple(oai.probe_notes))
        v22 = probe_v21_substrate_adapter_as_v22(v21)
        caps.append(probe_v22_substrate_adapter_as_v23(v22))
    return SubstrateAdapterV23Matrix(
        probed_at_ns=int(time.time_ns()),
        capabilities=tuple(caps))


__all__ = [
    "W78_SUBSTRATE_ADAPTER_V23_SCHEMA_VERSION",
    "W78_SUBSTRATE_TIER_SUBSTRATE_V23_FULL",
    "W78_SUBSTRATE_V23_NEW_AXES",
    "W78_SUBSTRATE_V23_CAPABILITY_AXES",
    "SubstrateCapabilityV23",
    "probe_tiny_substrate_v23_adapter",
    "probe_v22_substrate_adapter_as_v23",
    "probe_synthetic_v23_adapter",
    "SubstrateAdapterV23Matrix",
    "probe_all_v23_adapters",
]
