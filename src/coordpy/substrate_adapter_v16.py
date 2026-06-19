"""W71 — Substrate Adapter V16.

Strictly extends W70's ``coordpy.substrate_adapter_v15``. V16 adds
three new capability axes that the W71 V16 substrate satisfies and
hosted backends do not:

  * ``delayed_repair_trajectory_cid`` — V16 per-turn content-
    addressed SHA-256 over V15 repair primitives + restart events
    + delay windows
  * ``restart_dominance_per_layer`` — V16 per-layer argmax in
    [0..7] (V15's [0..6] + restart_dominance)
  * ``delayed_repair_gate_per_layer`` — V16 per-layer delay-
    conditioned throttle

V16 adds a new top tier:

  * ``substrate_v16_full`` — only the W71 V16 in-repo runtime
    satisfies every axis.

Hosted backends remain text-only at the HTTP surface;
``W71-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
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
    probe_ollama_adapter,
    probe_openai_compatible_adapter,
)
from .substrate_adapter_v8 import (
    W63_SUBSTRATE_TIER_SUBSTRATE_V8_FULL,
    W63_SUBSTRATE_V8_NEW_AXES,
)
from .substrate_adapter_v9 import (
    W64_SUBSTRATE_TIER_SUBSTRATE_V9_FULL,
    W64_SUBSTRATE_V9_NEW_AXES,
)
from .substrate_adapter_v10 import (
    W65_SUBSTRATE_TIER_SUBSTRATE_V10_FULL,
    W65_SUBSTRATE_V10_NEW_AXES,
)
from .substrate_adapter_v11 import (
    W66_SUBSTRATE_TIER_SUBSTRATE_V11_FULL,
    W66_SUBSTRATE_V11_NEW_AXES,
)
from .substrate_adapter_v12 import (
    W67_SUBSTRATE_TIER_SUBSTRATE_V12_FULL,
    W67_SUBSTRATE_V12_NEW_AXES,
)
from .substrate_adapter_v13 import (
    W68_SUBSTRATE_TIER_SUBSTRATE_V13_FULL,
    W68_SUBSTRATE_V13_NEW_AXES,
)
from .substrate_adapter_v14 import (
    W69_SUBSTRATE_TIER_SUBSTRATE_V14_FULL,
    W69_SUBSTRATE_V14_NEW_AXES,
)
from .substrate_adapter_v15 import (
    SubstrateCapabilityV15,
    W70_SUBSTRATE_TIER_SUBSTRATE_V15_FULL,
    W70_SUBSTRATE_V15_CAPABILITY_AXES,
    W70_SUBSTRATE_V15_NEW_AXES,
)


W71_SUBSTRATE_ADAPTER_V16_SCHEMA_VERSION: str = (
    "coordpy.substrate_adapter_v16.v1")

W71_SUBSTRATE_TIER_SUBSTRATE_V16_FULL: str = "substrate_v16_full"

W71_SUBSTRATE_V16_NEW_AXES: tuple[str, ...] = (
    "delayed_repair_trajectory_cid",
    "restart_dominance_per_layer",
    "delayed_repair_gate_per_layer",
)

W71_SUBSTRATE_V16_CAPABILITY_AXES: tuple[str, ...] = (
    *W70_SUBSTRATE_V15_CAPABILITY_AXES,
    *W71_SUBSTRATE_V16_NEW_AXES,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class SubstrateCapabilityV16:
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
            "schema": W71_SUBSTRATE_ADAPTER_V16_SCHEMA_VERSION,
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
            "kind": "substrate_capability_v16",
            "capability": self.to_dict()})


def _decide_tier_v16(caps: dict[str, str]) -> str:
    if caps.get("text") != "yes":
        return SUBSTRATE_TIER_UNREACHABLE
    has_v16 = all(
        caps.get(ax) == "yes" for ax in W71_SUBSTRATE_V16_NEW_AXES)
    has_v15 = all(
        caps.get(ax) == "yes" for ax in W70_SUBSTRATE_V15_NEW_AXES)
    has_v14 = all(
        caps.get(ax) == "yes" for ax in W69_SUBSTRATE_V14_NEW_AXES)
    has_v13 = all(
        caps.get(ax) == "yes" for ax in W68_SUBSTRATE_V13_NEW_AXES)
    has_v12 = all(
        caps.get(ax) == "yes" for ax in W67_SUBSTRATE_V12_NEW_AXES)
    has_v11 = all(
        caps.get(ax) == "yes" for ax in W66_SUBSTRATE_V11_NEW_AXES)
    has_v10 = all(
        caps.get(ax) == "yes" for ax in W65_SUBSTRATE_V10_NEW_AXES)
    has_v9 = all(
        caps.get(ax) == "yes" for ax in W64_SUBSTRATE_V9_NEW_AXES)
    has_v8 = all(
        caps.get(ax) == "yes" for ax in W63_SUBSTRATE_V8_NEW_AXES)
    if (has_v16 and has_v15 and has_v14 and has_v13 and has_v12
            and has_v11 and has_v10 and has_v9 and has_v8):
        return W71_SUBSTRATE_TIER_SUBSTRATE_V16_FULL
    if (has_v15 and has_v14 and has_v13 and has_v12 and has_v11
            and has_v10 and has_v9 and has_v8):
        return W70_SUBSTRATE_TIER_SUBSTRATE_V15_FULL
    if (has_v14 and has_v13 and has_v12 and has_v11
            and has_v10 and has_v9 and has_v8):
        return W69_SUBSTRATE_TIER_SUBSTRATE_V14_FULL
    if (has_v13 and has_v12 and has_v11
            and has_v10 and has_v9 and has_v8):
        return W68_SUBSTRATE_TIER_SUBSTRATE_V13_FULL
    if has_v12 and has_v11 and has_v10 and has_v9 and has_v8:
        return W67_SUBSTRATE_TIER_SUBSTRATE_V12_FULL
    if has_v11 and has_v10 and has_v9 and has_v8:
        return W66_SUBSTRATE_TIER_SUBSTRATE_V11_FULL
    if has_v10 and has_v9 and has_v8:
        return W65_SUBSTRATE_TIER_SUBSTRATE_V10_FULL
    if has_v9 and has_v8:
        return W64_SUBSTRATE_TIER_SUBSTRATE_V9_FULL
    if has_v8:
        return W63_SUBSTRATE_TIER_SUBSTRATE_V8_FULL
    if caps.get("logits") == "yes":
        return SUBSTRATE_TIER_LOGITS_ONLY
    if caps.get("embeddings") == "yes":
        return SUBSTRATE_TIER_EMBEDDINGS_ONLY
    return SUBSTRATE_TIER_TEXT_ONLY


def probe_tiny_substrate_v16_adapter(
        *, label: str = "tiny_substrate_v16",
) -> SubstrateCapabilityV16:
    caps = {ax: "yes" for ax in W71_SUBSTRATE_V16_CAPABILITY_AXES}
    return SubstrateCapabilityV16(
        backend_name=str(label),
        backend_url="in-process://coordpy.tiny_substrate_v16",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W71_SUBSTRATE_V16_CAPABILITY_AXES),
        tier=_decide_tier_v16(caps),
        probe_notes=(
            "tiny in-repo numpy transformer V16 (18 layers, GQA "
            "8q/4kv, RMSNorm, SwiGLU, d_model=64, d_key=8, "
            "vocab=259) + V15 axes + per-turn delayed-repair-"
            "trajectory CID + per-layer restart-dominance label + "
            "per-layer delayed-repair gate + KV bridge V16 twelve-"
            "target ridge + cache V14 eleven-objective ridge + "
            "replay V12 nineteen-regime ridge + restart-aware "
            "routing head + deep substrate hybrid V16 sixteen-way "
            "loop + multi-agent substrate coordinator V7 + team-"
            "consensus controller V6",
            "still NOT a frontier model; still does NOT prove "
            "third-party substrate access",
        ),
    )


def probe_v15_substrate_adapter_as_v16(
        cap: SubstrateCapabilityV15,
) -> SubstrateCapabilityV16:
    base = {ax: val for ax, val in cap.capabilities}
    for ax in W71_SUBSTRATE_V16_NEW_AXES:
        base.setdefault(ax, "no")
    tier = _decide_tier_v16(base)
    return SubstrateCapabilityV16(
        backend_name=str(cap.backend_name),
        backend_url=str(cap.backend_url),
        capabilities=tuple(
            (ax, base.get(ax, "no"))
            for ax in W71_SUBSTRATE_V16_CAPABILITY_AXES),
        tier=str(tier),
        probe_notes=tuple(cap.probe_notes) + (
            "wrapped from W70 substrate adapter V15",),
    )


def probe_synthetic_v16_adapter(
        *, label: str = "synthetic",
) -> SubstrateCapabilityV16:
    caps = {ax: "no" for ax in W71_SUBSTRATE_V16_CAPABILITY_AXES}
    caps["text"] = "yes"
    return SubstrateCapabilityV16(
        backend_name=str(label),
        backend_url="in-process://coordpy.synthetic_llm",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W71_SUBSTRATE_V16_CAPABILITY_AXES),
        tier=_decide_tier_v16(caps),
        probe_notes=(
            "synthetic deterministic backend; no substrate access; "
            "for hermetic capsule-layer testing only",),
    )


@dataclasses.dataclass(frozen=True)
class SubstrateAdapterV16Matrix:
    probed_at_ns: int
    capabilities: tuple[SubstrateCapabilityV16, ...]

    def by_name(self) -> dict[str, SubstrateCapabilityV16]:
        return {c.backend_name: c for c in self.capabilities}

    def has_v16_full(self) -> bool:
        return any(
            c.tier == W71_SUBSTRATE_TIER_SUBSTRATE_V16_FULL
            for c in self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W71_SUBSTRATE_ADAPTER_V16_SCHEMA_VERSION,
            "probed_at_ns": int(self.probed_at_ns),
            "capabilities": [
                c.to_dict() for c in self.capabilities],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "substrate_adapter_v16_matrix",
            "capabilities": [
                c.cid() for c in self.capabilities],
        })


def probe_all_v16_adapters(
        *, probe_ollama: bool = False,
        probe_openai: bool = False,
        ollama_url: str | None = None,
) -> SubstrateAdapterV16Matrix:
    caps: list[SubstrateCapabilityV16] = []
    caps.append(probe_tiny_substrate_v16_adapter())
    caps.append(probe_synthetic_v16_adapter())
    if probe_ollama:
        oll = probe_ollama_adapter(
            ollama_url=ollama_url
            or "http://localhost:11434")
        v16 = probe_v15_substrate_adapter_as_v16(
            SubstrateCapabilityV15(
                backend_name=str(oll.backend_name),
                backend_url=str(oll.backend_url),
                capabilities=tuple(
                    (ax, val) for ax, val in oll.capabilities),
                tier=str(oll.tier),
                probe_notes=tuple(oll.probe_notes)))
        caps.append(v16)
    if probe_openai:
        oai = probe_openai_compatible_adapter()
        v16 = probe_v15_substrate_adapter_as_v16(
            SubstrateCapabilityV15(
                backend_name=str(oai.backend_name),
                backend_url=str(oai.backend_url),
                capabilities=tuple(
                    (ax, val) for ax, val in oai.capabilities),
                tier=str(oai.tier),
                probe_notes=tuple(oai.probe_notes)))
        caps.append(v16)
    return SubstrateAdapterV16Matrix(
        probed_at_ns=int(time.time_ns()),
        capabilities=tuple(caps))


__all__ = [
    "W71_SUBSTRATE_ADAPTER_V16_SCHEMA_VERSION",
    "W71_SUBSTRATE_TIER_SUBSTRATE_V16_FULL",
    "W71_SUBSTRATE_V16_NEW_AXES",
    "W71_SUBSTRATE_V16_CAPABILITY_AXES",
    "SubstrateCapabilityV16",
    "probe_tiny_substrate_v16_adapter",
    "probe_v15_substrate_adapter_as_v16",
    "probe_synthetic_v16_adapter",
    "SubstrateAdapterV16Matrix",
    "probe_all_v16_adapters",
]
