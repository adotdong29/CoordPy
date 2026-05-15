"""W61 — Substrate Adapter V6.

Strictly extends W60's ``coordpy.substrate_adapter_v5``. V6 adds
seven new capability axes that the W61 V6 substrate satisfies and
hosted backends do not:

  * ``cache_key_axis`` — V6 per-(layer, position, d_key)
    content-addressable cache key tensor
  * ``hidden_write_trace`` — V6 per-(layer, head) cumulative L2 of
    hidden-state injections written by HSB V5
  * ``replay_age_channel`` — V6 per-(layer, position) integer
    forwards-since-write channel
  * ``cross_layer_coupling`` — V6 per-(layer_i, layer_j)
    cross-layer attention-coupling diagnostic
  * ``bilinear_retrieval_head`` — cache controller V4 bilinear M
    matrix over (query_feature ⊗ cache_key) closed-form ridge fit
  * ``trained_replay_thresholds`` — replay controller V2 closed-
    form ridge fit of (reuse / recompute / fallback / abstain)
    decision thresholds with per-decision confidence
  * ``attention_pattern_target`` — KV bridge V6 attention-pattern
    fit and HSB V5 multi-target stacked fit

V6 adds a new top tier:

  * ``substrate_v6_full`` — only the W61 V6 in-repo runtime
    satisfies every axis 1..36.

Hosted backends remain text-only at the HTTP surface;
``W61-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
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
    W59_SUBSTRATE_V4_NEW_AXES,
    W59_SUBSTRATE_TIER_SUBSTRATE_V4_FULL,
    probe_tiny_substrate_v4_adapter,
    probe_v3_substrate_adapter_as_v4,
)
from .substrate_adapter_v5 import (
    SubstrateCapabilityV5,
    W60_SUBSTRATE_ADAPTER_V5_SCHEMA_VERSION,
    W60_SUBSTRATE_TIER_SUBSTRATE_V5_FULL,
    W60_SUBSTRATE_V5_CAPABILITY_AXES,
    W60_SUBSTRATE_V5_NEW_AXES,
    probe_tiny_substrate_v5_adapter,
    probe_v4_substrate_adapter_as_v5,
)


W61_SUBSTRATE_ADAPTER_V6_SCHEMA_VERSION: str = (
    "coordpy.substrate_adapter_v6.v1")

W61_SUBSTRATE_TIER_SUBSTRATE_V6_FULL: str = "substrate_v6_full"

W61_SUBSTRATE_V6_NEW_AXES: tuple[str, ...] = (
    "cache_key_axis",
    "hidden_write_trace",
    "replay_age_channel",
    "cross_layer_coupling",
    "bilinear_retrieval_head",
    "trained_replay_thresholds",
    "attention_pattern_target",
)

W61_SUBSTRATE_V6_CAPABILITY_AXES: tuple[str, ...] = (
    *W60_SUBSTRATE_V5_CAPABILITY_AXES,
    *W61_SUBSTRATE_V6_NEW_AXES,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class SubstrateCapabilityV6:
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
            "schema": W61_SUBSTRATE_ADAPTER_V6_SCHEMA_VERSION,
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
            "kind": "substrate_capability_v6",
            "capability": self.to_dict()})


def _decide_tier_v6(caps: dict[str, str]) -> str:
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
    has_w61 = all(
        caps.get(ax) == "yes"
        for ax in W61_SUBSTRATE_V6_NEW_AXES)
    if (has_w56 and has_w57 and has_w58 and has_w59
            and has_w60 and has_w61):
        return W61_SUBSTRATE_TIER_SUBSTRATE_V6_FULL
    if (has_w56 and has_w57 and has_w58 and has_w59
            and has_w60):
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


def probe_tiny_substrate_v6_adapter(
        *, label: str = "tiny_substrate_v6",
) -> SubstrateCapabilityV6:
    caps = {ax: "yes" for ax in W61_SUBSTRATE_V6_CAPABILITY_AXES}
    return SubstrateCapabilityV6(
        backend_name=str(label),
        backend_url="in-process://coordpy.tiny_substrate_v6",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W61_SUBSTRATE_V6_CAPABILITY_AXES),
        tier=_decide_tier_v6(caps),
        probe_notes=(
            "tiny in-repo numpy transformer V6 with 8 layers + GQA "
            "+ RMSNorm + SwiGLU + per-(layer, head, position) "
            "cumulative attention-receive + per-(layer, head) "
            "linearised logit Jacobian table + per-(layer, "
            "position) corruption flag channel + per-(layer, "
            "position, d_key) content-addressable cache keys + "
            "per-(layer, head) cumulative hidden-write trace + "
            "per-(layer, position) replay-age channel + per-"
            "(layer_i, layer_j) cross-layer coupling matrix + "
            "multi-segment partial-prefix reuse + KV bridge V6 "
            "multi-target ridge + attention-pattern ridge + HSB "
            "V5 multi-target 3-D δ ridge + attention steering V5 "
            "per-(L, H, Q, K) 4-D budget + signed-coefficient "
            "falsifier + cache controller V4 bilinear retrieval "
            "+ trained corruption-aware floor + replay controller "
            "V2 trained thresholds + abstain confidence; bounded "
            "scope (8 layers, 8 query heads, 4 kv heads, "
            "d_model=64, d_key=8, vocab=259)",
            "still NOT a frontier model; still does NOT prove "
            "third-party substrate access",
        ),
    )


def probe_v5_substrate_adapter_as_v6(
        cap: SubstrateCapabilityV5,
) -> SubstrateCapabilityV6:
    """Wrap a V5 capability into the V6 envelope. The seven new
    axes are set to ``no`` for non-V6 backends."""
    base = {ax: val for ax, val in cap.capabilities}
    for ax in W61_SUBSTRATE_V6_NEW_AXES:
        base.setdefault(ax, "no")
    tier = _decide_tier_v6(base)
    return SubstrateCapabilityV6(
        backend_name=str(cap.backend_name),
        backend_url=str(cap.backend_url),
        capabilities=tuple(
            (ax, base.get(ax, "no"))
            for ax in W61_SUBSTRATE_V6_CAPABILITY_AXES),
        tier=str(tier),
        probe_notes=tuple(cap.probe_notes) + (
            "wrapped from W60 substrate adapter V5",),
    )


def probe_synthetic_v6_adapter(
        *, label: str = "synthetic",
) -> SubstrateCapabilityV6:
    caps = {ax: "no" for ax in W61_SUBSTRATE_V6_CAPABILITY_AXES}
    caps["text"] = "yes"
    return SubstrateCapabilityV6(
        backend_name=str(label),
        backend_url="in-process://coordpy.synthetic_llm",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W61_SUBSTRATE_V6_CAPABILITY_AXES),
        tier=_decide_tier_v6(caps),
        probe_notes=(
            "synthetic deterministic backend; no substrate "
            "access; for hermetic capsule-layer testing only",
        ),
    )


@dataclasses.dataclass(frozen=True)
class SubstrateAdapterV6Matrix:
    probed_at_ns: int
    capabilities: tuple[SubstrateCapabilityV6, ...]

    def by_name(self) -> dict[str, SubstrateCapabilityV6]:
        return {c.backend_name: c for c in self.capabilities}

    def has_v6_full(self) -> bool:
        return any(
            c.tier == W61_SUBSTRATE_TIER_SUBSTRATE_V6_FULL
            for c in self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W61_SUBSTRATE_ADAPTER_V6_SCHEMA_VERSION,
            "probed_at_ns": int(self.probed_at_ns),
            "capabilities": [
                c.to_dict() for c in self.capabilities],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "substrate_adapter_v6_matrix",
            "capabilities": [
                c.cid() for c in self.capabilities],
        })


def probe_all_v6_adapters(
        *, probe_ollama: bool = True,
        probe_openai: bool = True,
        ollama_url: str | None = None,
        openai_url: str | None = None,
        ollama_timeout: float = 1.5,
) -> SubstrateAdapterV6Matrix:
    caps: list[SubstrateCapabilityV6] = [
        probe_tiny_substrate_v6_adapter(),
        probe_v5_substrate_adapter_as_v6(
            probe_tiny_substrate_v5_adapter()),
        probe_v5_substrate_adapter_as_v6(
            probe_v4_substrate_adapter_as_v5(
                probe_tiny_substrate_v4_adapter())),
        probe_v5_substrate_adapter_as_v6(
            probe_v4_substrate_adapter_as_v5(
                probe_v3_substrate_adapter_as_v4(
                    probe_tiny_substrate_v3_adapter()))),
        probe_v5_substrate_adapter_as_v6(
            probe_v4_substrate_adapter_as_v5(
                probe_v3_substrate_adapter_as_v4(
                    probe_v2_substrate_adapter_as_v3(
                        probe_tiny_substrate_v2_adapter())))),
        probe_v5_substrate_adapter_as_v6(
            probe_v4_substrate_adapter_as_v5(
                probe_v3_substrate_adapter_as_v4(
                    probe_v2_substrate_adapter_as_v3(
                        probe_w56_substrate_adapter_as_v2(
                            probe_tiny_substrate_adapter()))))),
        probe_synthetic_v6_adapter(),
    ]
    if probe_ollama:
        caps.append(
            probe_v5_substrate_adapter_as_v6(
                probe_v4_substrate_adapter_as_v5(
                    probe_v3_substrate_adapter_as_v4(
                        probe_v2_substrate_adapter_as_v3(
                            probe_w56_substrate_adapter_as_v2(
                                probe_ollama_adapter(
                                    base_url=ollama_url,
                                    timeout=float(
                                        ollama_timeout))))))))
    if probe_openai:
        caps.append(
            probe_v5_substrate_adapter_as_v6(
                probe_v4_substrate_adapter_as_v5(
                    probe_v3_substrate_adapter_as_v4(
                        probe_v2_substrate_adapter_as_v3(
                            probe_w56_substrate_adapter_as_v2(
                                probe_openai_compatible_adapter(
                                    base_url=openai_url)))))))
    return SubstrateAdapterV6Matrix(
        probed_at_ns=time.monotonic_ns(),
        capabilities=tuple(caps),
    )


__all__ = [
    "W61_SUBSTRATE_ADAPTER_V6_SCHEMA_VERSION",
    "W61_SUBSTRATE_TIER_SUBSTRATE_V6_FULL",
    "W61_SUBSTRATE_V6_NEW_AXES",
    "W61_SUBSTRATE_V6_CAPABILITY_AXES",
    "SubstrateCapabilityV6",
    "SubstrateAdapterV6Matrix",
    "probe_tiny_substrate_v6_adapter",
    "probe_v5_substrate_adapter_as_v6",
    "probe_synthetic_v6_adapter",
    "probe_all_v6_adapters",
]
