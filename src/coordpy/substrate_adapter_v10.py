"""W65 — Substrate Adapter V10.

Strictly extends W64's ``coordpy.substrate_adapter_v9``. V10 adds
four new capability axes that the W65 V10 substrate satisfies and
hosted backends do not:

  * ``hidden_write_merit`` — V10 per-(layer, head, slot) merit
    scalar
  * ``role_kv_bank`` — V10 per-role KV cache offset matrix slot
  * ``substrate_checkpoint`` — V10 token-bounded snapshot /
    restore primitive
  * ``v10_gate_score`` — V10 per-layer composite gate score

V10 adds a new top tier:

  * ``substrate_v10_full`` — only the W65 V10 in-repo runtime
    satisfies every axis.

Hosted backends remain text-only at the HTTP surface;
``W65-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward
the W56..W64 cap unchanged.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from typing import Any, Sequence

from .substrate_adapter import (
    SUBSTRATE_TIER_EMBEDDINGS_ONLY,
    SUBSTRATE_TIER_LOGITS_ONLY,
    SUBSTRATE_TIER_SUBSTRATE_FULL,
    SUBSTRATE_TIER_TEXT_ONLY,
    SUBSTRATE_TIER_UNREACHABLE,
    SUBSTRATE_CAPABILITY_AXES,
    probe_ollama_adapter,
    probe_openai_compatible_adapter,
)
from .substrate_adapter_v2 import W57_SUBSTRATE_V2_NEW_AXES
from .substrate_adapter_v3 import (
    W58_SUBSTRATE_V3_NEW_AXES,
    W58_SUBSTRATE_TIER_SUBSTRATE_V3_FULL,
)
from .substrate_adapter_v4 import (
    W59_SUBSTRATE_V4_NEW_AXES,
    W59_SUBSTRATE_TIER_SUBSTRATE_V4_FULL,
)
from .substrate_adapter_v5 import (
    W60_SUBSTRATE_TIER_SUBSTRATE_V5_FULL,
    W60_SUBSTRATE_V5_NEW_AXES,
)
from .substrate_adapter_v6 import (
    W61_SUBSTRATE_TIER_SUBSTRATE_V6_FULL,
    W61_SUBSTRATE_V6_NEW_AXES,
)
from .substrate_adapter_v7 import (
    W62_SUBSTRATE_TIER_SUBSTRATE_V7_FULL,
    W62_SUBSTRATE_V7_NEW_AXES,
)
from .substrate_adapter_v8 import (
    W63_SUBSTRATE_TIER_SUBSTRATE_V8_FULL,
    W63_SUBSTRATE_V8_NEW_AXES,
)
from .substrate_adapter_v9 import (
    SubstrateCapabilityV9,
    W64_SUBSTRATE_ADAPTER_V9_SCHEMA_VERSION,
    W64_SUBSTRATE_TIER_SUBSTRATE_V9_FULL,
    W64_SUBSTRATE_V9_CAPABILITY_AXES,
    W64_SUBSTRATE_V9_NEW_AXES,
)


W65_SUBSTRATE_ADAPTER_V10_SCHEMA_VERSION: str = (
    "coordpy.substrate_adapter_v10.v1")

W65_SUBSTRATE_TIER_SUBSTRATE_V10_FULL: str = "substrate_v10_full"

W65_SUBSTRATE_V10_NEW_AXES: tuple[str, ...] = (
    "hidden_write_merit",
    "role_kv_bank",
    "substrate_checkpoint",
    "v10_gate_score",
)

W65_SUBSTRATE_V10_CAPABILITY_AXES: tuple[str, ...] = (
    *W64_SUBSTRATE_V9_CAPABILITY_AXES,
    *W65_SUBSTRATE_V10_NEW_AXES,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class SubstrateCapabilityV10:
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
            "schema": W65_SUBSTRATE_ADAPTER_V10_SCHEMA_VERSION,
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
            "kind": "substrate_capability_v10",
            "capability": self.to_dict()})


def _decide_tier_v10(caps: dict[str, str]) -> str:
    if caps.get("text") != "yes":
        return SUBSTRATE_TIER_UNREACHABLE
    has_v10 = all(
        caps.get(ax) == "yes" for ax in W65_SUBSTRATE_V10_NEW_AXES)
    has_v9 = all(
        caps.get(ax) == "yes" for ax in W64_SUBSTRATE_V9_NEW_AXES)
    has_v8 = all(
        caps.get(ax) == "yes" for ax in W63_SUBSTRATE_V8_NEW_AXES)
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


def probe_tiny_substrate_v10_adapter(
        *, label: str = "tiny_substrate_v10",
) -> SubstrateCapabilityV10:
    caps = {ax: "yes" for ax in W65_SUBSTRATE_V10_CAPABILITY_AXES}
    return SubstrateCapabilityV10(
        backend_name=str(label),
        backend_url="in-process://coordpy.tiny_substrate_v10",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W65_SUBSTRATE_V10_CAPABILITY_AXES),
        tier=_decide_tier_v10(caps),
        probe_notes=(
            "tiny in-repo numpy transformer V10 (12 layers, GQA "
            "8q/4kv, RMSNorm, SwiGLU, d_model=64, d_key=8, "
            "vocab=259) + V9 axes + per-(layer, head, slot) "
            "hidden-write-merit + per-role KV bank + substrate "
            "checkpoint/restore + per-layer V10 composite gate "
            "score + KV bridge V10 six-target ridge + HSB V9 "
            "six-target ridge + prefix V9 K=64 drift curve + "
            "attention V9 five-stage clamp + cache V8 five-"
            "objective ridge + per-role eviction head + replay "
            "V6 eight-regime ridge + per-role per-regime ridge + "
            "multi-agent abstain head + deep substrate hybrid "
            "V10 ten-way loop + multi-agent substrate "
            "coordinator",
            "still NOT a frontier model; still does NOT prove "
            "third-party substrate access",
        ),
    )


def probe_v9_substrate_adapter_as_v10(
        cap: SubstrateCapabilityV9,
) -> SubstrateCapabilityV10:
    base = {ax: val for ax, val in cap.capabilities}
    for ax in W65_SUBSTRATE_V10_NEW_AXES:
        base.setdefault(ax, "no")
    tier = _decide_tier_v10(base)
    return SubstrateCapabilityV10(
        backend_name=str(cap.backend_name),
        backend_url=str(cap.backend_url),
        capabilities=tuple(
            (ax, base.get(ax, "no"))
            for ax in W65_SUBSTRATE_V10_CAPABILITY_AXES),
        tier=str(tier),
        probe_notes=tuple(cap.probe_notes) + (
            "wrapped from W64 substrate adapter V9",),
    )


def probe_synthetic_v10_adapter(
        *, label: str = "synthetic",
) -> SubstrateCapabilityV10:
    caps = {ax: "no" for ax in W65_SUBSTRATE_V10_CAPABILITY_AXES}
    caps["text"] = "yes"
    return SubstrateCapabilityV10(
        backend_name=str(label),
        backend_url="in-process://coordpy.synthetic_llm",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W65_SUBSTRATE_V10_CAPABILITY_AXES),
        tier=_decide_tier_v10(caps),
        probe_notes=(
            "synthetic deterministic backend; no substrate access; "
            "for hermetic capsule-layer testing only",),
    )


@dataclasses.dataclass(frozen=True)
class SubstrateAdapterV10Matrix:
    probed_at_ns: int
    capabilities: tuple[SubstrateCapabilityV10, ...]

    def by_name(self) -> dict[str, SubstrateCapabilityV10]:
        return {c.backend_name: c for c in self.capabilities}

    def has_v10_full(self) -> bool:
        return any(
            c.tier == W65_SUBSTRATE_TIER_SUBSTRATE_V10_FULL
            for c in self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W65_SUBSTRATE_ADAPTER_V10_SCHEMA_VERSION,
            "probed_at_ns": int(self.probed_at_ns),
            "capabilities": [
                c.to_dict() for c in self.capabilities],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "substrate_adapter_v10_matrix",
            "capabilities": [
                c.cid() for c in self.capabilities],
        })


def probe_all_v10_adapters(
        *, probe_ollama: bool = False,
        probe_openai: bool = False,
        ollama_url: str | None = None,
) -> SubstrateAdapterV10Matrix:
    caps: list[SubstrateCapabilityV10] = []
    caps.append(probe_tiny_substrate_v10_adapter())
    caps.append(probe_synthetic_v10_adapter())
    if probe_ollama:
        oll = probe_ollama_adapter(
            ollama_url=ollama_url
            or "http://localhost:11434")
        v10 = probe_v9_substrate_adapter_as_v10(
            SubstrateCapabilityV9(
                backend_name=str(oll.backend_name),
                backend_url=str(oll.backend_url),
                capabilities=tuple(
                    (ax, val) for ax, val in oll.capabilities),
                tier=str(oll.tier),
                probe_notes=tuple(oll.probe_notes)))
        caps.append(v10)
    if probe_openai:
        oai = probe_openai_compatible_adapter()
        v10 = probe_v9_substrate_adapter_as_v10(
            SubstrateCapabilityV9(
                backend_name=str(oai.backend_name),
                backend_url=str(oai.backend_url),
                capabilities=tuple(
                    (ax, val) for ax, val in oai.capabilities),
                tier=str(oai.tier),
                probe_notes=tuple(oai.probe_notes)))
        caps.append(v10)
    return SubstrateAdapterV10Matrix(
        probed_at_ns=int(time.time_ns()),
        capabilities=tuple(caps))


__all__ = [
    "W65_SUBSTRATE_ADAPTER_V10_SCHEMA_VERSION",
    "W65_SUBSTRATE_TIER_SUBSTRATE_V10_FULL",
    "W65_SUBSTRATE_V10_NEW_AXES",
    "W65_SUBSTRATE_V10_CAPABILITY_AXES",
    "SubstrateCapabilityV10",
    "probe_tiny_substrate_v10_adapter",
    "probe_v9_substrate_adapter_as_v10",
    "probe_synthetic_v10_adapter",
    "SubstrateAdapterV10Matrix",
    "probe_all_v10_adapters",
]
