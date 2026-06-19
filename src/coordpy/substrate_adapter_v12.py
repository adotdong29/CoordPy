"""W67 — Substrate Adapter V12.

Strictly extends W66's ``coordpy.substrate_adapter_v11``. V12 adds
four new capability axes that the W67 V12 substrate satisfies and
hosted backends do not:

  * ``branch_merge_witness`` — V12 per-(layer, head, slot)
    branch-merge witness tensor
  * ``role_dropout_recovery_flag`` — V12 per-role-pair
    role-dropout-recovery boolean
  * ``substrate_snapshot_fork`` — V12 content-addressed
    snapshot-fork primitive
  * ``v12_gate_score`` — V12 per-layer composite gate score

V12 adds a new top tier:

  * ``substrate_v12_full`` — only the W67 V12 in-repo runtime
    satisfies every axis.

Hosted backends remain text-only at the HTTP surface;
``W67-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
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
    SubstrateCapabilityV11,
    W66_SUBSTRATE_ADAPTER_V11_SCHEMA_VERSION,
    W66_SUBSTRATE_TIER_SUBSTRATE_V11_FULL,
    W66_SUBSTRATE_V11_CAPABILITY_AXES,
    W66_SUBSTRATE_V11_NEW_AXES,
)


W67_SUBSTRATE_ADAPTER_V12_SCHEMA_VERSION: str = (
    "coordpy.substrate_adapter_v12.v1")

W67_SUBSTRATE_TIER_SUBSTRATE_V12_FULL: str = "substrate_v12_full"

W67_SUBSTRATE_V12_NEW_AXES: tuple[str, ...] = (
    "branch_merge_witness",
    "role_dropout_recovery_flag",
    "substrate_snapshot_fork",
    "v12_gate_score",
)

W67_SUBSTRATE_V12_CAPABILITY_AXES: tuple[str, ...] = (
    *W66_SUBSTRATE_V11_CAPABILITY_AXES,
    *W67_SUBSTRATE_V12_NEW_AXES,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class SubstrateCapabilityV12:
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
            "schema": W67_SUBSTRATE_ADAPTER_V12_SCHEMA_VERSION,
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
            "kind": "substrate_capability_v12",
            "capability": self.to_dict()})


def _decide_tier_v12(caps: dict[str, str]) -> str:
    if caps.get("text") != "yes":
        return SUBSTRATE_TIER_UNREACHABLE
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


def probe_tiny_substrate_v12_adapter(
        *, label: str = "tiny_substrate_v12",
) -> SubstrateCapabilityV12:
    caps = {ax: "yes" for ax in W67_SUBSTRATE_V12_CAPABILITY_AXES}
    return SubstrateCapabilityV12(
        backend_name=str(label),
        backend_url="in-process://coordpy.tiny_substrate_v12",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W67_SUBSTRATE_V12_CAPABILITY_AXES),
        tier=_decide_tier_v12(caps),
        probe_notes=(
            "tiny in-repo numpy transformer V12 (14 layers, GQA "
            "8q/4kv, RMSNorm, SwiGLU, d_model=64, d_key=8, "
            "vocab=259) + V11 axes + per-(layer, head, slot) "
            "branch-merge witness tensor + per-role-pair "
            "role-dropout-recovery flag + substrate snapshot-fork "
            "primitive + per-layer V12 composite gate score + KV "
            "bridge V12 eight-target ridge + HSB V11 eight-target "
            "ridge + prefix V11 K=128 drift curve + attention V11 "
            "seven-stage clamp + cache V10 seven-objective ridge + "
            "replay V8 twelve-regime ridge + branch-merge-routing "
            "head + deep substrate hybrid V12 twelve-way loop + "
            "multi-agent substrate coordinator V3 + team-consensus "
            "controller V2",
            "still NOT a frontier model; still does NOT prove "
            "third-party substrate access",
        ),
    )


def probe_v11_substrate_adapter_as_v12(
        cap: SubstrateCapabilityV11,
) -> SubstrateCapabilityV12:
    base = {ax: val for ax, val in cap.capabilities}
    for ax in W67_SUBSTRATE_V12_NEW_AXES:
        base.setdefault(ax, "no")
    tier = _decide_tier_v12(base)
    return SubstrateCapabilityV12(
        backend_name=str(cap.backend_name),
        backend_url=str(cap.backend_url),
        capabilities=tuple(
            (ax, base.get(ax, "no"))
            for ax in W67_SUBSTRATE_V12_CAPABILITY_AXES),
        tier=str(tier),
        probe_notes=tuple(cap.probe_notes) + (
            "wrapped from W66 substrate adapter V11",),
    )


def probe_synthetic_v12_adapter(
        *, label: str = "synthetic",
) -> SubstrateCapabilityV12:
    caps = {ax: "no" for ax in W67_SUBSTRATE_V12_CAPABILITY_AXES}
    caps["text"] = "yes"
    return SubstrateCapabilityV12(
        backend_name=str(label),
        backend_url="in-process://coordpy.synthetic_llm",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W67_SUBSTRATE_V12_CAPABILITY_AXES),
        tier=_decide_tier_v12(caps),
        probe_notes=(
            "synthetic deterministic backend; no substrate access; "
            "for hermetic capsule-layer testing only",),
    )


@dataclasses.dataclass(frozen=True)
class SubstrateAdapterV12Matrix:
    probed_at_ns: int
    capabilities: tuple[SubstrateCapabilityV12, ...]

    def by_name(self) -> dict[str, SubstrateCapabilityV12]:
        return {c.backend_name: c for c in self.capabilities}

    def has_v12_full(self) -> bool:
        return any(
            c.tier == W67_SUBSTRATE_TIER_SUBSTRATE_V12_FULL
            for c in self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W67_SUBSTRATE_ADAPTER_V12_SCHEMA_VERSION,
            "probed_at_ns": int(self.probed_at_ns),
            "capabilities": [
                c.to_dict() for c in self.capabilities],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "substrate_adapter_v12_matrix",
            "capabilities": [
                c.cid() for c in self.capabilities],
        })


def probe_all_v12_adapters(
        *, probe_ollama: bool = False,
        probe_openai: bool = False,
        ollama_url: str | None = None,
) -> SubstrateAdapterV12Matrix:
    caps: list[SubstrateCapabilityV12] = []
    caps.append(probe_tiny_substrate_v12_adapter())
    caps.append(probe_synthetic_v12_adapter())
    if probe_ollama:
        oll = probe_ollama_adapter(
            ollama_url=ollama_url
            or "http://localhost:11434")
        v12 = probe_v11_substrate_adapter_as_v12(
            SubstrateCapabilityV11(
                backend_name=str(oll.backend_name),
                backend_url=str(oll.backend_url),
                capabilities=tuple(
                    (ax, val) for ax, val in oll.capabilities),
                tier=str(oll.tier),
                probe_notes=tuple(oll.probe_notes)))
        caps.append(v12)
    if probe_openai:
        oai = probe_openai_compatible_adapter()
        v12 = probe_v11_substrate_adapter_as_v12(
            SubstrateCapabilityV11(
                backend_name=str(oai.backend_name),
                backend_url=str(oai.backend_url),
                capabilities=tuple(
                    (ax, val) for ax, val in oai.capabilities),
                tier=str(oai.tier),
                probe_notes=tuple(oai.probe_notes)))
        caps.append(v12)
    return SubstrateAdapterV12Matrix(
        probed_at_ns=int(time.time_ns()),
        capabilities=tuple(caps))


__all__ = [
    "W67_SUBSTRATE_ADAPTER_V12_SCHEMA_VERSION",
    "W67_SUBSTRATE_TIER_SUBSTRATE_V12_FULL",
    "W67_SUBSTRATE_V12_NEW_AXES",
    "W67_SUBSTRATE_V12_CAPABILITY_AXES",
    "SubstrateCapabilityV12",
    "probe_tiny_substrate_v12_adapter",
    "probe_v11_substrate_adapter_as_v12",
    "probe_synthetic_v12_adapter",
    "SubstrateAdapterV12Matrix",
    "probe_all_v12_adapters",
]
