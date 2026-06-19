"""W57 M-supporting — Substrate Adapter V2.

Extends W56's substrate adapter with **finer capability tiers**
and an honest declarative answer for the V2 substrate:

  1. ``text``                  — generate(prompt) -> str
  2. ``embeddings``            — encode(text) -> vector
  3. ``logprobs``              — per-token logprobs
  4. ``logits``                — full vocab logits per step
  5. ``hidden_states``         — per-layer residual stream
  6. ``kv_cache_read``         — read-side KV access
  7. ``kv_cache_write``        — KV injection
  8. ``attention_weights``     — per-head causal attention maps
  9. ``attention_bias_write``  — NEW: per-head attention bias hook
 10. ``prefix_state_reuse``    — NEW: save/load prefix states
 11. ``cache_eviction``        — NEW: evict_lru / evict_weighted
 12. ``logit_lens``            — NEW: per-layer unembed logits

The W56 capability set (axes 1..8) was satisfied by the W56 tiny
substrate. The V2 substrate (W57 M1) satisfies axes 1..12.
Third-party hosted backends remain unable to reach axes 4..12 on
their HTTP surface.

V2 records the same five tiers as W56 plus one new top tier:

  * ``substrate_v2_full`` — the V2 substrate reaches every axis
  * ``substrate_full``    — W56 tier, satisfies 1..8 only
  * ``logits_only``, ``embeddings_only``, ``text_only``,
    ``unreachable`` — same as W56.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from typing import Any, Sequence

from .substrate_adapter import (
    SUBSTRATE_ADAPTER_SCHEMA_VERSION,
    SUBSTRATE_CAPABILITY_AXES,
    SUBSTRATE_TIER_EMBEDDINGS_ONLY,
    SUBSTRATE_TIER_LOGITS_ONLY,
    SUBSTRATE_TIER_SUBSTRATE_FULL,
    SUBSTRATE_TIER_TEXT_ONLY,
    SUBSTRATE_TIER_UNREACHABLE,
    SubstrateAdapterMatrix,
    SubstrateCapability,
    probe_synthetic_adapter,
    probe_tiny_substrate_adapter,
    probe_ollama_adapter,
    probe_openai_compatible_adapter,
)


W57_SUBSTRATE_ADAPTER_V2_SCHEMA_VERSION: str = (
    "coordpy.substrate_adapter_v2.v1")

W57_SUBSTRATE_TIER_SUBSTRATE_V2_FULL: str = "substrate_v2_full"

W57_SUBSTRATE_V2_NEW_AXES: tuple[str, ...] = (
    "attention_bias_write",
    "prefix_state_reuse",
    "cache_eviction",
    "logit_lens",
)

W57_SUBSTRATE_V2_CAPABILITY_AXES: tuple[str, ...] = (
    *SUBSTRATE_CAPABILITY_AXES,
    *W57_SUBSTRATE_V2_NEW_AXES,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class SubstrateCapabilityV2:
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
            "schema": W57_SUBSTRATE_ADAPTER_V2_SCHEMA_VERSION,
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
            "kind": "substrate_capability_v2",
            "capability": self.to_dict()})


def _decide_tier_v2(caps: dict[str, str]) -> str:
    if caps.get("text") != "yes":
        return SUBSTRATE_TIER_UNREACHABLE
    has_w56 = all(
        caps.get(ax) == "yes"
        for ax in SUBSTRATE_CAPABILITY_AXES)
    has_new = all(
        caps.get(ax) == "yes"
        for ax in W57_SUBSTRATE_V2_NEW_AXES)
    if has_w56 and has_new:
        return W57_SUBSTRATE_TIER_SUBSTRATE_V2_FULL
    if has_w56:
        return SUBSTRATE_TIER_SUBSTRATE_FULL
    if (caps.get("logits") == "yes"
            or caps.get("logprobs") == "yes"):
        return SUBSTRATE_TIER_LOGITS_ONLY
    if caps.get("embeddings") == "yes":
        return SUBSTRATE_TIER_EMBEDDINGS_ONLY
    return SUBSTRATE_TIER_TEXT_ONLY


def probe_tiny_substrate_v2_adapter(
        *, label: str = "tiny_substrate_v2",
) -> SubstrateCapabilityV2:
    caps = {ax: "yes" for ax in W57_SUBSTRATE_V2_CAPABILITY_AXES}
    return SubstrateCapabilityV2(
        backend_name=str(label),
        backend_url="in-process://coordpy.tiny_substrate_v2",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W57_SUBSTRATE_V2_CAPABILITY_AXES),
        tier=_decide_tier_v2(caps),
        probe_notes=(
            "tiny in-repo numpy transformer V2; real KV / hidden "
            "states / attention / RoPE / logit-lens / prefix-state "
            "reuse / attention-bias hook / cache eviction; bounded "
            "scope (4 layers, 8 heads, d_model=64, vocab=259)",
            "this is NOT a frontier model and does NOT prove "
            "third-party substrate access",
        ),
    )


def probe_w56_substrate_adapter_as_v2(
        cap: SubstrateCapability,
) -> SubstrateCapabilityV2:
    """Wrap a W56 SubstrateCapability into the V2 envelope.

    Adds the four new axes with values:
      * ``attention_bias_write``: ``no`` (W56 did not expose)
      * ``prefix_state_reuse``: ``no``
      * ``cache_eviction``: ``no``
      * ``logit_lens``: ``no``
    """
    base = {
        ax: val for ax, val in cap.capabilities}
    for ax in W57_SUBSTRATE_V2_NEW_AXES:
        base.setdefault(ax, "no")
        if cap.tier == SUBSTRATE_TIER_UNREACHABLE:
            base[ax] = "no"
    tier = _decide_tier_v2(base)
    return SubstrateCapabilityV2(
        backend_name=str(cap.backend_name),
        backend_url=str(cap.backend_url),
        capabilities=tuple(
            (ax, base.get(ax, "no"))
            for ax in W57_SUBSTRATE_V2_CAPABILITY_AXES),
        tier=str(tier),
        probe_notes=tuple(cap.probe_notes) + (
            "wrapped from W56 substrate adapter",),
    )


def probe_synthetic_v2_adapter(
        *, label: str = "synthetic",
) -> SubstrateCapabilityV2:
    caps = {ax: "no" for ax in W57_SUBSTRATE_V2_CAPABILITY_AXES}
    caps["text"] = "yes"
    return SubstrateCapabilityV2(
        backend_name=str(label),
        backend_url="in-process://coordpy.synthetic_llm",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W57_SUBSTRATE_V2_CAPABILITY_AXES),
        tier=_decide_tier_v2(caps),
        probe_notes=(
            "synthetic deterministic backend; no substrate "
            "access; for hermetic capsule-layer testing only",
        ),
    )


@dataclasses.dataclass(frozen=True)
class SubstrateAdapterV2Matrix:
    probed_at_ns: int
    capabilities: tuple[SubstrateCapabilityV2, ...]

    def by_name(self) -> dict[str, SubstrateCapabilityV2]:
        return {c.backend_name: c for c in self.capabilities}

    def has_v2_full(self) -> bool:
        return any(
            c.tier == W57_SUBSTRATE_TIER_SUBSTRATE_V2_FULL
            for c in self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W57_SUBSTRATE_ADAPTER_V2_SCHEMA_VERSION,
            "probed_at_ns": int(self.probed_at_ns),
            "capabilities": [
                c.to_dict() for c in self.capabilities],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "substrate_adapter_v2_matrix",
            "capabilities": [
                c.cid() for c in self.capabilities],
        })


def probe_all_v2_adapters(
        *, probe_ollama: bool = True,
        probe_openai: bool = True,
        ollama_url: str | None = None,
        openai_url: str | None = None,
        ollama_timeout: float = 1.5,
) -> SubstrateAdapterV2Matrix:
    caps: list[SubstrateCapabilityV2] = [
        probe_tiny_substrate_v2_adapter(),
        probe_w56_substrate_adapter_as_v2(
            probe_tiny_substrate_adapter()),
        probe_synthetic_v2_adapter(),
    ]
    if probe_ollama:
        caps.append(probe_w56_substrate_adapter_as_v2(
            probe_ollama_adapter(
                base_url=ollama_url,
                timeout=float(ollama_timeout))))
    if probe_openai:
        caps.append(probe_w56_substrate_adapter_as_v2(
            probe_openai_compatible_adapter(
                base_url=openai_url)))
    return SubstrateAdapterV2Matrix(
        probed_at_ns=time.monotonic_ns(),
        capabilities=tuple(caps),
    )


__all__ = [
    "W57_SUBSTRATE_ADAPTER_V2_SCHEMA_VERSION",
    "W57_SUBSTRATE_TIER_SUBSTRATE_V2_FULL",
    "W57_SUBSTRATE_V2_NEW_AXES",
    "W57_SUBSTRATE_V2_CAPABILITY_AXES",
    "SubstrateCapabilityV2",
    "SubstrateAdapterV2Matrix",
    "probe_tiny_substrate_v2_adapter",
    "probe_w56_substrate_adapter_as_v2",
    "probe_synthetic_v2_adapter",
    "probe_all_v2_adapters",
]
