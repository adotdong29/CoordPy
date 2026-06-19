"""W77 — Substrate Adapter V22.

Strictly extends W76's ``coordpy.substrate_adapter_v21``. V22 adds
three new capability axes that the W77 V22 substrate satisfies and
hosted backends do not:

  * ``replacement_after_restart_after_compound_chain_trajectory_cid``
  * ``replacement_after_restart_after_compound_chain_length_per_layer``
  * ``replacement_after_restart_after_compound_chain_pressure_gate_per_layer``

V22 adds a new top tier:

  * ``substrate_v22_full`` — only the W77 V22 in-repo runtime
    satisfies every axis.

``W77-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
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
from .substrate_adapter_v21 import (
    SubstrateCapabilityV21,
    W76_SUBSTRATE_TIER_SUBSTRATE_V21_FULL,
    W76_SUBSTRATE_V21_CAPABILITY_AXES,
    W76_SUBSTRATE_V21_NEW_AXES,
    _sha256_hex,
)


W77_SUBSTRATE_ADAPTER_V22_SCHEMA_VERSION: str = (
    "coordpy.substrate_adapter_v22.v1")

W77_SUBSTRATE_TIER_SUBSTRATE_V22_FULL: str = "substrate_v22_full"

W77_SUBSTRATE_V22_NEW_AXES: tuple[str, ...] = (
    "replacement_after_restart_after_compound_chain_trajectory_cid",
    "replacement_after_restart_after_compound_chain_length_per_layer",
    "replacement_after_restart_after_compound_chain_pressure_gate_per_layer",
)

W77_SUBSTRATE_V22_CAPABILITY_AXES: tuple[str, ...] = (
    *W76_SUBSTRATE_V21_CAPABILITY_AXES,
    *W77_SUBSTRATE_V22_NEW_AXES,
)


@dataclasses.dataclass(frozen=True)
class SubstrateCapabilityV22:
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
            "schema": W77_SUBSTRATE_ADAPTER_V22_SCHEMA_VERSION,
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
            "kind": "substrate_capability_v22",
            "capability": self.to_dict()})


def _decide_tier_v22(caps: dict[str, str]) -> str:
    if caps.get("text") != "yes":
        return SUBSTRATE_TIER_UNREACHABLE
    has_v22 = all(
        caps.get(ax) == "yes" for ax in W77_SUBSTRATE_V22_NEW_AXES)
    has_v21 = all(
        caps.get(ax) == "yes" for ax in W76_SUBSTRATE_V21_NEW_AXES)
    if has_v22 and has_v21:
        return W77_SUBSTRATE_TIER_SUBSTRATE_V22_FULL
    if has_v21:
        return W76_SUBSTRATE_TIER_SUBSTRATE_V21_FULL
    if caps.get("logits") == "yes":
        return SUBSTRATE_TIER_LOGITS_ONLY
    if caps.get("embeddings") == "yes":
        return SUBSTRATE_TIER_EMBEDDINGS_ONLY
    return SUBSTRATE_TIER_TEXT_ONLY


def probe_tiny_substrate_v22_adapter(
        *, label: str = "tiny_substrate_v22",
) -> SubstrateCapabilityV22:
    caps = {ax: "yes" for ax in W77_SUBSTRATE_V22_CAPABILITY_AXES}
    return SubstrateCapabilityV22(
        backend_name=str(label),
        backend_url="in-process://coordpy.tiny_substrate_v22",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W77_SUBSTRATE_V22_CAPABILITY_AXES),
        tier=_decide_tier_v22(caps),
        probe_notes=(
            "tiny in-repo numpy transformer V22 (24 layers, GQA "
            "8q/4kv, RMSNorm, SwiGLU, d_model=64, d_key=8, "
            "vocab=259) + V21 axes + per-turn post-restart-"
            "replacement-after-compound-chain trajectory CID + "
            "per-layer post-restart-replacement length label + "
            "per-layer post-restart-replacement-pressure gate + "
            "KV bridge V22 eighteen-target ridge + cache V20 "
            "seventeen-objective ridge + replay V18 twenty-five-"
            "regime ridge + post-restart-replacement-aware "
            "routing head + deep substrate hybrid V22 twenty-two-"
            "way loop + multi-agent substrate coordinator V13 + "
            "team-consensus controller V12",
            "still NOT a frontier model; still does NOT prove "
            "third-party substrate access",
        ),
    )


def probe_v21_substrate_adapter_as_v22(
        cap: SubstrateCapabilityV21,
) -> SubstrateCapabilityV22:
    base = {ax: val for ax, val in cap.capabilities}
    for ax in W77_SUBSTRATE_V22_NEW_AXES:
        base.setdefault(ax, "no")
    tier = _decide_tier_v22(base)
    return SubstrateCapabilityV22(
        backend_name=str(cap.backend_name),
        backend_url=str(cap.backend_url),
        capabilities=tuple(
            (ax, base.get(ax, "no"))
            for ax in W77_SUBSTRATE_V22_CAPABILITY_AXES),
        tier=str(tier),
        probe_notes=tuple(cap.probe_notes) + (
            "wrapped from W76 substrate adapter V21",),
    )


def probe_synthetic_v22_adapter(
        *, label: str = "synthetic",
) -> SubstrateCapabilityV22:
    caps = {ax: "no" for ax in W77_SUBSTRATE_V22_CAPABILITY_AXES}
    caps["text"] = "yes"
    return SubstrateCapabilityV22(
        backend_name=str(label),
        backend_url="in-process://coordpy.synthetic_llm",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W77_SUBSTRATE_V22_CAPABILITY_AXES),
        tier=_decide_tier_v22(caps),
        probe_notes=(
            "synthetic deterministic backend; no substrate "
            "access; for hermetic capsule-layer testing only",),
    )


@dataclasses.dataclass(frozen=True)
class SubstrateAdapterV22Matrix:
    probed_at_ns: int
    capabilities: tuple[SubstrateCapabilityV22, ...]

    def by_name(self) -> dict[str, SubstrateCapabilityV22]:
        return {c.backend_name: c for c in self.capabilities}

    def has_v22_full(self) -> bool:
        return any(
            c.tier == W77_SUBSTRATE_TIER_SUBSTRATE_V22_FULL
            for c in self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W77_SUBSTRATE_ADAPTER_V22_SCHEMA_VERSION,
            "probed_at_ns": int(self.probed_at_ns),
            "capabilities": [
                c.to_dict() for c in self.capabilities],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "substrate_adapter_v22_matrix",
            "capabilities": [
                c.cid() for c in self.capabilities],
        })


def probe_all_v22_adapters(
        *, probe_ollama: bool = False,
        probe_openai: bool = False,
        ollama_url: str | None = None,
) -> SubstrateAdapterV22Matrix:
    caps: list[SubstrateCapabilityV22] = []
    caps.append(probe_tiny_substrate_v22_adapter())
    caps.append(probe_synthetic_v22_adapter())
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
        caps.append(probe_v21_substrate_adapter_as_v22(v21))
    if probe_openai:
        oai = probe_openai_compatible_adapter()
        v21 = SubstrateCapabilityV21(
            backend_name=str(oai.backend_name),
            backend_url=str(oai.backend_url),
            capabilities=tuple(
                (ax, val) for ax, val in oai.capabilities),
            tier=str(oai.tier),
            probe_notes=tuple(oai.probe_notes))
        caps.append(probe_v21_substrate_adapter_as_v22(v21))
    return SubstrateAdapterV22Matrix(
        probed_at_ns=int(time.time_ns()),
        capabilities=tuple(caps))


__all__ = [
    "W77_SUBSTRATE_ADAPTER_V22_SCHEMA_VERSION",
    "W77_SUBSTRATE_TIER_SUBSTRATE_V22_FULL",
    "W77_SUBSTRATE_V22_NEW_AXES",
    "W77_SUBSTRATE_V22_CAPABILITY_AXES",
    "SubstrateCapabilityV22",
    "probe_tiny_substrate_v22_adapter",
    "probe_v21_substrate_adapter_as_v22",
    "probe_synthetic_v22_adapter",
    "SubstrateAdapterV22Matrix",
    "probe_all_v22_adapters",
]
