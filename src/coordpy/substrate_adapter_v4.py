"""W59 M18 — Substrate Adapter V4.

Strictly extends W58's ``coordpy.substrate_adapter_v3``. V4 adds
five new capability axes that the W59 V4 substrate satisfies and
hosted backends do not:

  * ``partial_prefix_reuse``    — V4 partial-prefix split + replay
  * ``cache_retrieval``         — controller V2 retrieval head
  * ``closed_form_ridge``       — bridge V4 ridge fit
  * ``per_head_kl_fit``         — attention V3 per-head KL clip
  * ``hidden_target_fit``       — HSB V3 target-logit fit

V4 adds a new top tier:

  * ``substrate_v4_full`` — only the W59 V4 in-repo runtime
    satisfies every axis 1..22.

Hosted backends remain text-only at the HTTP surface;
``W59-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
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
    W57_SUBSTRATE_V2_CAPABILITY_AXES,
    W57_SUBSTRATE_V2_NEW_AXES,
    probe_tiny_substrate_v2_adapter,
    probe_w56_substrate_adapter_as_v2,
)
from .substrate_adapter_v3 import (
    SubstrateAdapterV3Matrix,
    SubstrateCapabilityV3,
    W58_SUBSTRATE_V3_CAPABILITY_AXES,
    W58_SUBSTRATE_V3_NEW_AXES,
    W58_SUBSTRATE_TIER_SUBSTRATE_V3_FULL,
    probe_synthetic_v3_adapter,
    probe_tiny_substrate_v3_adapter,
    probe_v2_substrate_adapter_as_v3,
)


W59_SUBSTRATE_ADAPTER_V4_SCHEMA_VERSION: str = (
    "coordpy.substrate_adapter_v4.v1")

W59_SUBSTRATE_TIER_SUBSTRATE_V4_FULL: str = "substrate_v4_full"

W59_SUBSTRATE_V4_NEW_AXES: tuple[str, ...] = (
    "partial_prefix_reuse",
    "cache_retrieval",
    "closed_form_ridge",
    "per_head_kl_fit",
    "hidden_target_fit",
)

W59_SUBSTRATE_V4_CAPABILITY_AXES: tuple[str, ...] = (
    *W58_SUBSTRATE_V3_CAPABILITY_AXES,
    *W59_SUBSTRATE_V4_NEW_AXES,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class SubstrateCapabilityV4:
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
            "schema": W59_SUBSTRATE_ADAPTER_V4_SCHEMA_VERSION,
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
            "kind": "substrate_capability_v4",
            "capability": self.to_dict()})


def _decide_tier_v4(caps: dict[str, str]) -> str:
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


def probe_tiny_substrate_v4_adapter(
        *, label: str = "tiny_substrate_v4",
) -> SubstrateCapabilityV4:
    caps = {ax: "yes" for ax in W59_SUBSTRATE_V4_CAPABILITY_AXES}
    return SubstrateCapabilityV4(
        backend_name=str(label),
        backend_url="in-process://coordpy.tiny_substrate_v4",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W59_SUBSTRATE_V4_CAPABILITY_AXES),
        tier=_decide_tier_v4(caps),
        probe_notes=(
            "tiny in-repo numpy transformer V4 with GQA + RMSNorm "
            "+ SwiGLU + cumulative-EMA KV importance + real flop "
            "counter + partial-prefix split/replay + KV bridge V4 "
            "closed-form ridge fit + per-head KL clip + HSB V3 "
            "target-logit fit + cache controller V2 retrieval; "
            "bounded scope (6 layers, 8 query heads, 4 kv heads, "
            "d_model=64, vocab=259)",
            "still NOT a frontier model; still does NOT prove "
            "third-party substrate access",
        ),
    )


def probe_v3_substrate_adapter_as_v4(
        cap: SubstrateCapabilityV3,
) -> SubstrateCapabilityV4:
    """Wrap a V3 capability into the V4 envelope. The five new
    axes are set to ``yes`` only if the wrapped backend is the
    in-repo tiny V3 substrate (which has the W58 paths but lacks
    W59 paths). For all other V3 wrappers default to ``no``."""
    base = {ax: val for ax, val in cap.capabilities}
    for ax in W59_SUBSTRATE_V4_NEW_AXES:
        base.setdefault(ax, "no")
    tier = _decide_tier_v4(base)
    return SubstrateCapabilityV4(
        backend_name=str(cap.backend_name),
        backend_url=str(cap.backend_url),
        capabilities=tuple(
            (ax, base.get(ax, "no"))
            for ax in W59_SUBSTRATE_V4_CAPABILITY_AXES),
        tier=str(tier),
        probe_notes=tuple(cap.probe_notes) + (
            "wrapped from W58 substrate adapter V3",),
    )


def probe_synthetic_v4_adapter(
        *, label: str = "synthetic",
) -> SubstrateCapabilityV4:
    caps = {ax: "no" for ax in W59_SUBSTRATE_V4_CAPABILITY_AXES}
    caps["text"] = "yes"
    return SubstrateCapabilityV4(
        backend_name=str(label),
        backend_url="in-process://coordpy.synthetic_llm",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W59_SUBSTRATE_V4_CAPABILITY_AXES),
        tier=_decide_tier_v4(caps),
        probe_notes=(
            "synthetic deterministic backend; no substrate "
            "access; for hermetic capsule-layer testing only",
        ),
    )


@dataclasses.dataclass(frozen=True)
class SubstrateAdapterV4Matrix:
    probed_at_ns: int
    capabilities: tuple[SubstrateCapabilityV4, ...]

    def by_name(self) -> dict[str, SubstrateCapabilityV4]:
        return {c.backend_name: c for c in self.capabilities}

    def has_v4_full(self) -> bool:
        return any(
            c.tier == W59_SUBSTRATE_TIER_SUBSTRATE_V4_FULL
            for c in self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W59_SUBSTRATE_ADAPTER_V4_SCHEMA_VERSION,
            "probed_at_ns": int(self.probed_at_ns),
            "capabilities": [
                c.to_dict() for c in self.capabilities],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "substrate_adapter_v4_matrix",
            "capabilities": [
                c.cid() for c in self.capabilities],
        })


def probe_all_v4_adapters(
        *, probe_ollama: bool = True,
        probe_openai: bool = True,
        ollama_url: str | None = None,
        openai_url: str | None = None,
        ollama_timeout: float = 1.5,
) -> SubstrateAdapterV4Matrix:
    caps: list[SubstrateCapabilityV4] = [
        probe_tiny_substrate_v4_adapter(),
        probe_v3_substrate_adapter_as_v4(
            probe_tiny_substrate_v3_adapter()),
        probe_v3_substrate_adapter_as_v4(
            probe_v2_substrate_adapter_as_v3(
                probe_tiny_substrate_v2_adapter())),
        probe_v3_substrate_adapter_as_v4(
            probe_v2_substrate_adapter_as_v3(
                probe_w56_substrate_adapter_as_v2(
                    probe_tiny_substrate_adapter()))),
        probe_synthetic_v4_adapter(),
    ]
    if probe_ollama:
        caps.append(
            probe_v3_substrate_adapter_as_v4(
                probe_v2_substrate_adapter_as_v3(
                    probe_w56_substrate_adapter_as_v2(
                        probe_ollama_adapter(
                            base_url=ollama_url,
                            timeout=float(ollama_timeout))))))
    if probe_openai:
        caps.append(
            probe_v3_substrate_adapter_as_v4(
                probe_v2_substrate_adapter_as_v3(
                    probe_w56_substrate_adapter_as_v2(
                        probe_openai_compatible_adapter(
                            base_url=openai_url)))))
    return SubstrateAdapterV4Matrix(
        probed_at_ns=time.monotonic_ns(),
        capabilities=tuple(caps),
    )


__all__ = [
    "W59_SUBSTRATE_ADAPTER_V4_SCHEMA_VERSION",
    "W59_SUBSTRATE_TIER_SUBSTRATE_V4_FULL",
    "W59_SUBSTRATE_V4_NEW_AXES",
    "W59_SUBSTRATE_V4_CAPABILITY_AXES",
    "SubstrateCapabilityV4",
    "SubstrateAdapterV4Matrix",
    "probe_tiny_substrate_v4_adapter",
    "probe_v3_substrate_adapter_as_v4",
    "probe_synthetic_v4_adapter",
    "probe_all_v4_adapters",
]
