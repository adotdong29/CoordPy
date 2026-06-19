"""W58 M18 — Substrate Adapter V3.

Strictly extends W57's ``coordpy.substrate_adapter_v2``. V3 adds
five new capability axes that the W58 V3 substrate satisfies and
hosted backends do not:

  * ``kv_importance_track``  — per-token KV importance vector
  * ``flop_counter``         — real fp64 flop count per forward
  * ``partial_forward``      — suffix-of-layers forward
  * ``fitted_inject_scale``  — KV bridge V3's coordinate descent
  * ``cache_controller``     — learned/importance retention head

V3 adds a new top tier:

  * ``substrate_v3_full`` — only the W58 V3 in-repo runtime
    satisfies every axis 1..17.

Hosted backends remain text-only at the HTTP surface;
``W58-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
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
    SubstrateCapability,
    probe_ollama_adapter,
    probe_openai_compatible_adapter,
    probe_tiny_substrate_adapter,
)
from .substrate_adapter_v2 import (
    SubstrateAdapterV2Matrix,
    SubstrateCapabilityV2,
    W57_SUBSTRATE_TIER_SUBSTRATE_V2_FULL,
    W57_SUBSTRATE_V2_CAPABILITY_AXES,
    W57_SUBSTRATE_V2_NEW_AXES,
    probe_synthetic_v2_adapter,
    probe_tiny_substrate_v2_adapter,
    probe_w56_substrate_adapter_as_v2,
)


W58_SUBSTRATE_ADAPTER_V3_SCHEMA_VERSION: str = (
    "coordpy.substrate_adapter_v3.v1")

W58_SUBSTRATE_TIER_SUBSTRATE_V3_FULL: str = "substrate_v3_full"

W58_SUBSTRATE_V3_NEW_AXES: tuple[str, ...] = (
    "kv_importance_track",
    "flop_counter",
    "partial_forward",
    "fitted_inject_scale",
    "cache_controller",
)

W58_SUBSTRATE_V3_CAPABILITY_AXES: tuple[str, ...] = (
    *W57_SUBSTRATE_V2_CAPABILITY_AXES,
    *W58_SUBSTRATE_V3_NEW_AXES,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class SubstrateCapabilityV3:
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
            "schema": W58_SUBSTRATE_ADAPTER_V3_SCHEMA_VERSION,
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
            "kind": "substrate_capability_v3",
            "capability": self.to_dict()})


def _decide_tier_v3(caps: dict[str, str]) -> str:
    if caps.get("text") != "yes":
        return SUBSTRATE_TIER_UNREACHABLE
    has_w56 = all(
        caps.get(ax) == "yes"
        for ax in SUBSTRATE_CAPABILITY_AXES)
    has_w57 = all(
        caps.get(ax) == "yes"
        for ax in W57_SUBSTRATE_V2_NEW_AXES)
    has_w58 = all(
        caps.get(ax) == "yes"
        for ax in W58_SUBSTRATE_V3_NEW_AXES)
    if has_w56 and has_w57 and has_w58:
        return W58_SUBSTRATE_TIER_SUBSTRATE_V3_FULL
    if has_w56 and has_w57:
        return W57_SUBSTRATE_TIER_SUBSTRATE_V2_FULL
    if has_w56:
        return SUBSTRATE_TIER_SUBSTRATE_FULL
    if (caps.get("logits") == "yes"
            or caps.get("logprobs") == "yes"):
        return SUBSTRATE_TIER_LOGITS_ONLY
    if caps.get("embeddings") == "yes":
        return SUBSTRATE_TIER_EMBEDDINGS_ONLY
    return SUBSTRATE_TIER_TEXT_ONLY


def probe_tiny_substrate_v3_adapter(
        *, label: str = "tiny_substrate_v3",
) -> SubstrateCapabilityV3:
    caps = {ax: "yes" for ax in W58_SUBSTRATE_V3_CAPABILITY_AXES}
    return SubstrateCapabilityV3(
        backend_name=str(label),
        backend_url="in-process://coordpy.tiny_substrate_v3",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W58_SUBSTRATE_V3_CAPABILITY_AXES),
        tier=_decide_tier_v3(caps),
        probe_notes=(
            "tiny in-repo numpy transformer V3 with GQA + RMSNorm "
            "+ SwiGLU + KV importance tracking + real flop counter "
            "+ partial-forward + KV bridge V3 fitted inject scale "
            "+ cache controller; bounded scope (5 layers, 8 query "
            "heads, 4 kv heads, d_model=64, vocab=259)",
            "still NOT a frontier model; still does NOT prove "
            "third-party substrate access",
        ),
    )


def probe_v2_substrate_adapter_as_v3(
        cap: SubstrateCapabilityV2,
) -> SubstrateCapabilityV3:
    """Wrap a V2 capability into the V3 envelope. The five new
    axes are set to ``yes`` only if the wrapped backend is the
    in-repo tiny V2 substrate that has the W57 paths but lacks
    W58 paths — otherwise ``no``."""
    base = {ax: val for ax, val in cap.capabilities}
    for ax in W58_SUBSTRATE_V3_NEW_AXES:
        base.setdefault(ax, "no")
    tier = _decide_tier_v3(base)
    return SubstrateCapabilityV3(
        backend_name=str(cap.backend_name),
        backend_url=str(cap.backend_url),
        capabilities=tuple(
            (ax, base.get(ax, "no"))
            for ax in W58_SUBSTRATE_V3_CAPABILITY_AXES),
        tier=str(tier),
        probe_notes=tuple(cap.probe_notes) + (
            "wrapped from W57 substrate adapter V2",),
    )


def probe_synthetic_v3_adapter(
        *, label: str = "synthetic",
) -> SubstrateCapabilityV3:
    caps = {ax: "no" for ax in W58_SUBSTRATE_V3_CAPABILITY_AXES}
    caps["text"] = "yes"
    return SubstrateCapabilityV3(
        backend_name=str(label),
        backend_url="in-process://coordpy.synthetic_llm",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W58_SUBSTRATE_V3_CAPABILITY_AXES),
        tier=_decide_tier_v3(caps),
        probe_notes=(
            "synthetic deterministic backend; no substrate "
            "access; for hermetic capsule-layer testing only",
        ),
    )


@dataclasses.dataclass(frozen=True)
class SubstrateAdapterV3Matrix:
    probed_at_ns: int
    capabilities: tuple[SubstrateCapabilityV3, ...]

    def by_name(self) -> dict[str, SubstrateCapabilityV3]:
        return {c.backend_name: c for c in self.capabilities}

    def has_v3_full(self) -> bool:
        return any(
            c.tier == W58_SUBSTRATE_TIER_SUBSTRATE_V3_FULL
            for c in self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W58_SUBSTRATE_ADAPTER_V3_SCHEMA_VERSION,
            "probed_at_ns": int(self.probed_at_ns),
            "capabilities": [
                c.to_dict() for c in self.capabilities],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "substrate_adapter_v3_matrix",
            "capabilities": [
                c.cid() for c in self.capabilities],
        })


def probe_all_v3_adapters(
        *, probe_ollama: bool = True,
        probe_openai: bool = True,
        ollama_url: str | None = None,
        openai_url: str | None = None,
        ollama_timeout: float = 1.5,
) -> SubstrateAdapterV3Matrix:
    caps: list[SubstrateCapabilityV3] = [
        probe_tiny_substrate_v3_adapter(),
        probe_v2_substrate_adapter_as_v3(
            probe_tiny_substrate_v2_adapter()),
        probe_v2_substrate_adapter_as_v3(
            probe_w56_substrate_adapter_as_v2(
                probe_tiny_substrate_adapter())),
        probe_synthetic_v3_adapter(),
    ]
    if probe_ollama:
        caps.append(
            probe_v2_substrate_adapter_as_v3(
                probe_w56_substrate_adapter_as_v2(
                    probe_ollama_adapter(
                        base_url=ollama_url,
                        timeout=float(ollama_timeout)))))
    if probe_openai:
        caps.append(
            probe_v2_substrate_adapter_as_v3(
                probe_w56_substrate_adapter_as_v2(
                    probe_openai_compatible_adapter(
                        base_url=openai_url))))
    return SubstrateAdapterV3Matrix(
        probed_at_ns=time.monotonic_ns(),
        capabilities=tuple(caps),
    )


__all__ = [
    "W58_SUBSTRATE_ADAPTER_V3_SCHEMA_VERSION",
    "W58_SUBSTRATE_TIER_SUBSTRATE_V3_FULL",
    "W58_SUBSTRATE_V3_NEW_AXES",
    "W58_SUBSTRATE_V3_CAPABILITY_AXES",
    "SubstrateCapabilityV3",
    "SubstrateAdapterV3Matrix",
    "probe_tiny_substrate_v3_adapter",
    "probe_v2_substrate_adapter_as_v3",
    "probe_synthetic_v3_adapter",
    "probe_all_v3_adapters",
]
