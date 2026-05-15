"""W60 — Substrate Adapter V5.

Strictly extends W59's ``coordpy.substrate_adapter_v4``. V5 adds
seven new capability axes that the W60 V5 substrate satisfies and
hosted backends do not:

  * ``multi_segment_prefix_reuse`` — V5 multi-segment partial
    reuse split into reuse + recompute + drop
  * ``per_head_attention_receive`` — V5 per-(layer, head, position)
    cumulative attention-receive matrix
  * ``per_layer_per_head_jacobian`` — V5 per-(layer, head)
    linearised logit Jacobian table
  * ``corruption_flag_channel`` — V5 per-(layer, position)
    corruption flag table
  * ``trained_eviction`` — cache controller V3 trained-eviction
    closed-form ridge head
  * ``replay_controller`` — W60 ReplayController choose-reuse /
    choose-recompute / choose-fallback / choose-abstain policy
  * ``hidden_vs_kv_compare`` — HSB V4 vs KV V4 head-to-head
    harness on a target logit direction

V5 adds a new top tier:

  * ``substrate_v5_full`` — only the W60 V5 in-repo runtime
    satisfies every axis 1..29.

Hosted backends remain text-only at the HTTP surface;
``W60-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
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
    probe_tiny_substrate_adapter,
)
from .substrate_adapter_v2 import (
    W57_SUBSTRATE_V2_NEW_AXES,
    probe_tiny_substrate_v2_adapter,
    probe_w56_substrate_adapter_as_v2,
)
from .substrate_adapter_v3 import (
    W58_SUBSTRATE_V3_NEW_AXES,
    W58_SUBSTRATE_TIER_SUBSTRATE_V3_FULL,
    probe_tiny_substrate_v3_adapter,
    probe_v2_substrate_adapter_as_v3,
)
from .substrate_adapter_v4 import (
    SubstrateCapabilityV4,
    W59_SUBSTRATE_ADAPTER_V4_SCHEMA_VERSION,
    W59_SUBSTRATE_TIER_SUBSTRATE_V4_FULL,
    W59_SUBSTRATE_V4_CAPABILITY_AXES,
    W59_SUBSTRATE_V4_NEW_AXES,
    probe_tiny_substrate_v4_adapter,
    probe_v3_substrate_adapter_as_v4,
)


W60_SUBSTRATE_ADAPTER_V5_SCHEMA_VERSION: str = (
    "coordpy.substrate_adapter_v5.v1")

W60_SUBSTRATE_TIER_SUBSTRATE_V5_FULL: str = "substrate_v5_full"

W60_SUBSTRATE_V5_NEW_AXES: tuple[str, ...] = (
    "multi_segment_prefix_reuse",
    "per_head_attention_receive",
    "per_layer_per_head_jacobian",
    "corruption_flag_channel",
    "trained_eviction",
    "replay_controller",
    "hidden_vs_kv_compare",
)

W60_SUBSTRATE_V5_CAPABILITY_AXES: tuple[str, ...] = (
    *W59_SUBSTRATE_V4_CAPABILITY_AXES,
    *W60_SUBSTRATE_V5_NEW_AXES,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class SubstrateCapabilityV5:
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
            "schema": W60_SUBSTRATE_ADAPTER_V5_SCHEMA_VERSION,
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
            "kind": "substrate_capability_v5",
            "capability": self.to_dict()})


def _decide_tier_v5(caps: dict[str, str]) -> str:
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
    has_w59 = all(
        caps.get(ax) == "yes"
        for ax in W59_SUBSTRATE_V4_NEW_AXES)
    has_w60 = all(
        caps.get(ax) == "yes"
        for ax in W60_SUBSTRATE_V5_NEW_AXES)
    if has_w56 and has_w57 and has_w58 and has_w59 and has_w60:
        return W60_SUBSTRATE_TIER_SUBSTRATE_V5_FULL
    if has_w56 and has_w57 and has_w58 and has_w59:
        return W59_SUBSTRATE_TIER_SUBSTRATE_V4_FULL
    if has_w56 and has_w57 and has_w58:
        return W58_SUBSTRATE_TIER_SUBSTRATE_V3_FULL
    if has_w56 and has_w57:
        from .substrate_adapter_v2 import (
            W57_SUBSTRATE_TIER_SUBSTRATE_V2_FULL,
        )
        return W57_SUBSTRATE_TIER_SUBSTRATE_V2_FULL
    if has_w56:
        return SUBSTRATE_TIER_SUBSTRATE_FULL
    if (caps.get("logits") == "yes"
            or caps.get("logprobs") == "yes"):
        return SUBSTRATE_TIER_LOGITS_ONLY
    if caps.get("embeddings") == "yes":
        return SUBSTRATE_TIER_EMBEDDINGS_ONLY
    return SUBSTRATE_TIER_TEXT_ONLY


def probe_tiny_substrate_v5_adapter(
        *, label: str = "tiny_substrate_v5",
) -> SubstrateCapabilityV5:
    caps = {ax: "yes" for ax in W60_SUBSTRATE_V5_CAPABILITY_AXES}
    return SubstrateCapabilityV5(
        backend_name=str(label),
        backend_url="in-process://coordpy.tiny_substrate_v5",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W60_SUBSTRATE_V5_CAPABILITY_AXES),
        tier=_decide_tier_v5(caps),
        probe_notes=(
            "tiny in-repo numpy transformer V5 with 7 layers + GQA "
            "+ RMSNorm + SwiGLU + per-(layer, head, position) "
            "cumulative attention-receive + per-(layer, head) "
            "linearised logit Jacobian table + per-(layer, "
            "position) corruption flag channel + multi-segment "
            "partial-prefix split / replay (reuse + recompute + "
            "drop) + KV bridge V5 multi-direction closed-form "
            "ridge fit + HSB V4 per-(layer, head) closed-form "
            "ridge fit + cache controller V3 composite_v3 + "
            "trained_eviction + W60 ReplayController; bounded "
            "scope (7 layers, 8 query heads, 4 kv heads, "
            "d_model=64, vocab=259)",
            "still NOT a frontier model; still does NOT prove "
            "third-party substrate access",
        ),
    )


def probe_v4_substrate_adapter_as_v5(
        cap: SubstrateCapabilityV4,
) -> SubstrateCapabilityV5:
    """Wrap a V4 capability into the V5 envelope. The seven new
    axes are set to ``yes`` only if the wrapped backend is the
    in-repo tiny V4 substrate (which has W59 paths but lacks W60
    paths). For all other V4 wrappers default to ``no``."""
    base = {ax: val for ax, val in cap.capabilities}
    for ax in W60_SUBSTRATE_V5_NEW_AXES:
        base.setdefault(ax, "no")
    tier = _decide_tier_v5(base)
    return SubstrateCapabilityV5(
        backend_name=str(cap.backend_name),
        backend_url=str(cap.backend_url),
        capabilities=tuple(
            (ax, base.get(ax, "no"))
            for ax in W60_SUBSTRATE_V5_CAPABILITY_AXES),
        tier=str(tier),
        probe_notes=tuple(cap.probe_notes) + (
            "wrapped from W59 substrate adapter V4",),
    )


def probe_synthetic_v5_adapter(
        *, label: str = "synthetic",
) -> SubstrateCapabilityV5:
    caps = {ax: "no" for ax in W60_SUBSTRATE_V5_CAPABILITY_AXES}
    caps["text"] = "yes"
    return SubstrateCapabilityV5(
        backend_name=str(label),
        backend_url="in-process://coordpy.synthetic_llm",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W60_SUBSTRATE_V5_CAPABILITY_AXES),
        tier=_decide_tier_v5(caps),
        probe_notes=(
            "synthetic deterministic backend; no substrate "
            "access; for hermetic capsule-layer testing only",
        ),
    )


@dataclasses.dataclass(frozen=True)
class SubstrateAdapterV5Matrix:
    probed_at_ns: int
    capabilities: tuple[SubstrateCapabilityV5, ...]

    def by_name(self) -> dict[str, SubstrateCapabilityV5]:
        return {c.backend_name: c for c in self.capabilities}

    def has_v5_full(self) -> bool:
        return any(
            c.tier == W60_SUBSTRATE_TIER_SUBSTRATE_V5_FULL
            for c in self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W60_SUBSTRATE_ADAPTER_V5_SCHEMA_VERSION,
            "probed_at_ns": int(self.probed_at_ns),
            "capabilities": [
                c.to_dict() for c in self.capabilities],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "substrate_adapter_v5_matrix",
            "capabilities": [
                c.cid() for c in self.capabilities],
        })


def probe_all_v5_adapters(
        *, probe_ollama: bool = True,
        probe_openai: bool = True,
        ollama_url: str | None = None,
        openai_url: str | None = None,
        ollama_timeout: float = 1.5,
) -> SubstrateAdapterV5Matrix:
    caps: list[SubstrateCapabilityV5] = [
        probe_tiny_substrate_v5_adapter(),
        probe_v4_substrate_adapter_as_v5(
            probe_tiny_substrate_v4_adapter()),
        probe_v4_substrate_adapter_as_v5(
            probe_v3_substrate_adapter_as_v4(
                probe_tiny_substrate_v3_adapter())),
        probe_v4_substrate_adapter_as_v5(
            probe_v3_substrate_adapter_as_v4(
                probe_v2_substrate_adapter_as_v3(
                    probe_tiny_substrate_v2_adapter()))),
        probe_v4_substrate_adapter_as_v5(
            probe_v3_substrate_adapter_as_v4(
                probe_v2_substrate_adapter_as_v3(
                    probe_w56_substrate_adapter_as_v2(
                        probe_tiny_substrate_adapter())))),
        probe_synthetic_v5_adapter(),
    ]
    if probe_ollama:
        caps.append(
            probe_v4_substrate_adapter_as_v5(
                probe_v3_substrate_adapter_as_v4(
                    probe_v2_substrate_adapter_as_v3(
                        probe_w56_substrate_adapter_as_v2(
                            probe_ollama_adapter(
                                base_url=ollama_url,
                                timeout=float(ollama_timeout)))))))
    if probe_openai:
        caps.append(
            probe_v4_substrate_adapter_as_v5(
                probe_v3_substrate_adapter_as_v4(
                    probe_v2_substrate_adapter_as_v3(
                        probe_w56_substrate_adapter_as_v2(
                            probe_openai_compatible_adapter(
                                base_url=openai_url))))))
    return SubstrateAdapterV5Matrix(
        probed_at_ns=time.monotonic_ns(),
        capabilities=tuple(caps),
    )


__all__ = [
    "W60_SUBSTRATE_ADAPTER_V5_SCHEMA_VERSION",
    "W60_SUBSTRATE_TIER_SUBSTRATE_V5_FULL",
    "W60_SUBSTRATE_V5_NEW_AXES",
    "W60_SUBSTRATE_V5_CAPABILITY_AXES",
    "SubstrateCapabilityV5",
    "SubstrateAdapterV5Matrix",
    "probe_tiny_substrate_v5_adapter",
    "probe_v4_substrate_adapter_as_v5",
    "probe_synthetic_v5_adapter",
    "probe_all_v5_adapters",
]
