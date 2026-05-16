"""W66 — Substrate Adapter V11.

Strictly extends W65's ``coordpy.substrate_adapter_v10``. V11 adds
four new capability axes that the W66 V11 substrate satisfies and
hosted backends do not:

  * ``replay_trust_ledger`` — V11 per-(layer, head, slot) replay-
    trust scalar
  * ``team_failure_recovery_flag`` — V11 per-role team-failure-
    recovery boolean
  * ``substrate_snapshot_diff`` — V11 content-addressed snapshot-
    diff primitive
  * ``v11_gate_score`` — V11 per-layer composite gate score

V11 adds a new top tier:

  * ``substrate_v11_full`` — only the W66 V11 in-repo runtime
    satisfies every axis.

Hosted backends remain text-only at the HTTP surface;
``W66-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
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
    SubstrateCapabilityV10,
    W65_SUBSTRATE_ADAPTER_V10_SCHEMA_VERSION,
    W65_SUBSTRATE_TIER_SUBSTRATE_V10_FULL,
    W65_SUBSTRATE_V10_CAPABILITY_AXES,
    W65_SUBSTRATE_V10_NEW_AXES,
)


W66_SUBSTRATE_ADAPTER_V11_SCHEMA_VERSION: str = (
    "coordpy.substrate_adapter_v11.v1")

W66_SUBSTRATE_TIER_SUBSTRATE_V11_FULL: str = "substrate_v11_full"

W66_SUBSTRATE_V11_NEW_AXES: tuple[str, ...] = (
    "replay_trust_ledger",
    "team_failure_recovery_flag",
    "substrate_snapshot_diff",
    "v11_gate_score",
)

W66_SUBSTRATE_V11_CAPABILITY_AXES: tuple[str, ...] = (
    *W65_SUBSTRATE_V10_CAPABILITY_AXES,
    *W66_SUBSTRATE_V11_NEW_AXES,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class SubstrateCapabilityV11:
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
            "schema": W66_SUBSTRATE_ADAPTER_V11_SCHEMA_VERSION,
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
            "kind": "substrate_capability_v11",
            "capability": self.to_dict()})


def _decide_tier_v11(caps: dict[str, str]) -> str:
    if caps.get("text") != "yes":
        return SUBSTRATE_TIER_UNREACHABLE
    has_v11 = all(
        caps.get(ax) == "yes" for ax in W66_SUBSTRATE_V11_NEW_AXES)
    has_v10 = all(
        caps.get(ax) == "yes" for ax in W65_SUBSTRATE_V10_NEW_AXES)
    has_v9 = all(
        caps.get(ax) == "yes" for ax in W64_SUBSTRATE_V9_NEW_AXES)
    has_v8 = all(
        caps.get(ax) == "yes" for ax in W63_SUBSTRATE_V8_NEW_AXES)
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


def probe_tiny_substrate_v11_adapter(
        *, label: str = "tiny_substrate_v11",
) -> SubstrateCapabilityV11:
    caps = {ax: "yes" for ax in W66_SUBSTRATE_V11_CAPABILITY_AXES}
    return SubstrateCapabilityV11(
        backend_name=str(label),
        backend_url="in-process://coordpy.tiny_substrate_v11",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W66_SUBSTRATE_V11_CAPABILITY_AXES),
        tier=_decide_tier_v11(caps),
        probe_notes=(
            "tiny in-repo numpy transformer V11 (13 layers, GQA "
            "8q/4kv, RMSNorm, SwiGLU, d_model=64, d_key=8, "
            "vocab=259) + V10 axes + per-(layer, head, slot) "
            "replay-trust ledger + per-role team-failure-recovery "
            "flag + substrate snapshot-diff primitive + per-layer "
            "V11 composite gate score + KV bridge V11 seven-target "
            "ridge + HSB V10 seven-target ridge + prefix V10 K=96 "
            "drift curve + attention V10 six-stage clamp + cache "
            "V9 six-objective ridge + replay V7 nine-regime ridge "
            "+ team-substrate-routing head + deep substrate "
            "hybrid V11 eleven-way loop + multi-agent substrate "
            "coordinator V2 + team-consensus controller",
            "still NOT a frontier model; still does NOT prove "
            "third-party substrate access",
        ),
    )


def probe_v10_substrate_adapter_as_v11(
        cap: SubstrateCapabilityV10,
) -> SubstrateCapabilityV11:
    base = {ax: val for ax, val in cap.capabilities}
    for ax in W66_SUBSTRATE_V11_NEW_AXES:
        base.setdefault(ax, "no")
    tier = _decide_tier_v11(base)
    return SubstrateCapabilityV11(
        backend_name=str(cap.backend_name),
        backend_url=str(cap.backend_url),
        capabilities=tuple(
            (ax, base.get(ax, "no"))
            for ax in W66_SUBSTRATE_V11_CAPABILITY_AXES),
        tier=str(tier),
        probe_notes=tuple(cap.probe_notes) + (
            "wrapped from W65 substrate adapter V10",),
    )


def probe_synthetic_v11_adapter(
        *, label: str = "synthetic",
) -> SubstrateCapabilityV11:
    caps = {ax: "no" for ax in W66_SUBSTRATE_V11_CAPABILITY_AXES}
    caps["text"] = "yes"
    return SubstrateCapabilityV11(
        backend_name=str(label),
        backend_url="in-process://coordpy.synthetic_llm",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W66_SUBSTRATE_V11_CAPABILITY_AXES),
        tier=_decide_tier_v11(caps),
        probe_notes=(
            "synthetic deterministic backend; no substrate access; "
            "for hermetic capsule-layer testing only",),
    )


@dataclasses.dataclass(frozen=True)
class SubstrateAdapterV11Matrix:
    probed_at_ns: int
    capabilities: tuple[SubstrateCapabilityV11, ...]

    def by_name(self) -> dict[str, SubstrateCapabilityV11]:
        return {c.backend_name: c for c in self.capabilities}

    def has_v11_full(self) -> bool:
        return any(
            c.tier == W66_SUBSTRATE_TIER_SUBSTRATE_V11_FULL
            for c in self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W66_SUBSTRATE_ADAPTER_V11_SCHEMA_VERSION,
            "probed_at_ns": int(self.probed_at_ns),
            "capabilities": [
                c.to_dict() for c in self.capabilities],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "substrate_adapter_v11_matrix",
            "capabilities": [
                c.cid() for c in self.capabilities],
        })


def probe_all_v11_adapters(
        *, probe_ollama: bool = False,
        probe_openai: bool = False,
        ollama_url: str | None = None,
) -> SubstrateAdapterV11Matrix:
    caps: list[SubstrateCapabilityV11] = []
    caps.append(probe_tiny_substrate_v11_adapter())
    caps.append(probe_synthetic_v11_adapter())
    if probe_ollama:
        oll = probe_ollama_adapter(
            ollama_url=ollama_url
            or "http://localhost:11434")
        v11 = probe_v10_substrate_adapter_as_v11(
            SubstrateCapabilityV10(
                backend_name=str(oll.backend_name),
                backend_url=str(oll.backend_url),
                capabilities=tuple(
                    (ax, val) for ax, val in oll.capabilities),
                tier=str(oll.tier),
                probe_notes=tuple(oll.probe_notes)))
        caps.append(v11)
    if probe_openai:
        oai = probe_openai_compatible_adapter()
        v11 = probe_v10_substrate_adapter_as_v11(
            SubstrateCapabilityV10(
                backend_name=str(oai.backend_name),
                backend_url=str(oai.backend_url),
                capabilities=tuple(
                    (ax, val) for ax, val in oai.capabilities),
                tier=str(oai.tier),
                probe_notes=tuple(oai.probe_notes)))
        caps.append(v11)
    return SubstrateAdapterV11Matrix(
        probed_at_ns=int(time.time_ns()),
        capabilities=tuple(caps))


__all__ = [
    "W66_SUBSTRATE_ADAPTER_V11_SCHEMA_VERSION",
    "W66_SUBSTRATE_TIER_SUBSTRATE_V11_FULL",
    "W66_SUBSTRATE_V11_NEW_AXES",
    "W66_SUBSTRATE_V11_CAPABILITY_AXES",
    "SubstrateCapabilityV11",
    "probe_tiny_substrate_v11_adapter",
    "probe_v10_substrate_adapter_as_v11",
    "probe_synthetic_v11_adapter",
    "SubstrateAdapterV11Matrix",
    "probe_all_v11_adapters",
]
