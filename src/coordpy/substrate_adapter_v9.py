"""W64 — Substrate Adapter V9.

Strictly extends W63's ``coordpy.substrate_adapter_v8``. V9 adds
five new capability axes that the W64 V9 substrate satisfies and
hosted backends do not:

  * ``hidden_wins_primary`` — V9 per-(layer, head, slot) signed
    primary-decision flag
  * ``replay_dominance_witness`` — V9 per-(layer, head, slot)
    replay-dominance scalar
  * ``attention_entropy_probe`` — V9 per-layer attention-distribution
    Shannon entropy (calibrated to [0, 1])
  * ``cache_similarity_matrix`` — V9 per-(layer, head, slot, slot)
    cosine similarity over cache_keys
  * ``hidden_state_trust_ledger`` — V9 per-(layer, head) EMA over
    hidden-state-bridge V8 decisions

V9 adds a new top tier:

  * ``substrate_v9_full`` — only the W64 V9 in-repo runtime
    satisfies every axis.

Hosted backends remain text-only at the HTTP surface;
``W64-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward
the W56..W63 cap unchanged.
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
from .substrate_adapter_v2 import (
    W57_SUBSTRATE_V2_NEW_AXES,
)
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
    SubstrateCapabilityV8,
    W63_SUBSTRATE_ADAPTER_V8_SCHEMA_VERSION,
    W63_SUBSTRATE_TIER_SUBSTRATE_V8_FULL,
    W63_SUBSTRATE_V8_CAPABILITY_AXES,
    W63_SUBSTRATE_V8_NEW_AXES,
    probe_v7_substrate_adapter_as_v8,
)


W64_SUBSTRATE_ADAPTER_V9_SCHEMA_VERSION: str = (
    "coordpy.substrate_adapter_v9.v1")

W64_SUBSTRATE_TIER_SUBSTRATE_V9_FULL: str = "substrate_v9_full"

W64_SUBSTRATE_V9_NEW_AXES: tuple[str, ...] = (
    "hidden_wins_primary",
    "replay_dominance_witness",
    "attention_entropy_probe",
    "cache_similarity_matrix",
    "hidden_state_trust_ledger",
)

W64_SUBSTRATE_V9_CAPABILITY_AXES: tuple[str, ...] = (
    *W63_SUBSTRATE_V8_CAPABILITY_AXES,
    *W64_SUBSTRATE_V9_NEW_AXES,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class SubstrateCapabilityV9:
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
            "schema": W64_SUBSTRATE_ADAPTER_V9_SCHEMA_VERSION,
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
            "kind": "substrate_capability_v9",
            "capability": self.to_dict()})


def _decide_tier_v9(caps: dict[str, str]) -> str:
    if caps.get("text") != "yes":
        return SUBSTRATE_TIER_UNREACHABLE
    has_w56 = all(
        caps.get(ax) == "yes" for ax in SUBSTRATE_CAPABILITY_AXES)
    has_w57 = all(
        caps.get(ax) == "yes" for ax in W57_SUBSTRATE_V2_NEW_AXES)
    has_w58 = all(
        caps.get(ax) == "yes" for ax in W58_SUBSTRATE_V3_NEW_AXES)
    has_w59 = all(
        caps.get(ax) == "yes" for ax in W59_SUBSTRATE_V4_NEW_AXES)
    has_w60 = all(
        caps.get(ax) == "yes" for ax in W60_SUBSTRATE_V5_NEW_AXES)
    has_w61 = all(
        caps.get(ax) == "yes" for ax in W61_SUBSTRATE_V6_NEW_AXES)
    has_w62 = all(
        caps.get(ax) == "yes" for ax in W62_SUBSTRATE_V7_NEW_AXES)
    has_w63 = all(
        caps.get(ax) == "yes" for ax in W63_SUBSTRATE_V8_NEW_AXES)
    has_w64 = all(
        caps.get(ax) == "yes" for ax in W64_SUBSTRATE_V9_NEW_AXES)
    if (has_w56 and has_w57 and has_w58 and has_w59
            and has_w60 and has_w61 and has_w62 and has_w63
            and has_w64):
        return W64_SUBSTRATE_TIER_SUBSTRATE_V9_FULL
    if (has_w56 and has_w57 and has_w58 and has_w59
            and has_w60 and has_w61 and has_w62 and has_w63):
        return W63_SUBSTRATE_TIER_SUBSTRATE_V8_FULL
    if (has_w56 and has_w57 and has_w58 and has_w59
            and has_w60 and has_w61 and has_w62):
        return W62_SUBSTRATE_TIER_SUBSTRATE_V7_FULL
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


def probe_tiny_substrate_v9_adapter(
        *, label: str = "tiny_substrate_v9",
) -> SubstrateCapabilityV9:
    caps = {ax: "yes" for ax in W64_SUBSTRATE_V9_CAPABILITY_AXES}
    return SubstrateCapabilityV9(
        backend_name=str(label),
        backend_url="in-process://coordpy.tiny_substrate_v9",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W64_SUBSTRATE_V9_CAPABILITY_AXES),
        tier=_decide_tier_v9(caps),
        probe_notes=(
            "tiny in-repo numpy transformer V9 (11 layers, GQA "
            "8q/4kv, RMSNorm, SwiGLU, d_model=64, d_key=8, "
            "vocab=259) + V8 axes + per-(layer, head, slot) "
            "hidden-wins-primary + per-(layer, head, slot) "
            "replay-dominance-witness + per-layer "
            "attention-entropy probe + per-(layer, head, slot, "
            "slot) cache-similarity matrix + per-(layer, head) "
            "hidden-state-trust ledger + KV bridge V9 five-target "
            "ridge fit + HSB V8 five-target ridge fit + prefix V8 "
            "role-conditional drift-curve predictor + attention "
            "V8 four-stage clamp + cache controller V7 four-"
            "objective ridge + similarity-aware eviction head + "
            "composite_v7 mixture + replay V5 seven-regime head + "
            "four-way bridge classifier + replay-dominance-"
            "primary head + deep substrate hybrid V9 nine-way "
            "loop",
            "still NOT a frontier model; still does NOT prove "
            "third-party substrate access",
        ),
    )


def probe_v8_substrate_adapter_as_v9(
        cap: SubstrateCapabilityV8,
) -> SubstrateCapabilityV9:
    base = {ax: val for ax, val in cap.capabilities}
    for ax in W64_SUBSTRATE_V9_NEW_AXES:
        base.setdefault(ax, "no")
    tier = _decide_tier_v9(base)
    return SubstrateCapabilityV9(
        backend_name=str(cap.backend_name),
        backend_url=str(cap.backend_url),
        capabilities=tuple(
            (ax, base.get(ax, "no"))
            for ax in W64_SUBSTRATE_V9_CAPABILITY_AXES),
        tier=str(tier),
        probe_notes=tuple(cap.probe_notes) + (
            "wrapped from W63 substrate adapter V8",),
    )


def probe_synthetic_v9_adapter(
        *, label: str = "synthetic",
) -> SubstrateCapabilityV9:
    caps = {ax: "no" for ax in W64_SUBSTRATE_V9_CAPABILITY_AXES}
    caps["text"] = "yes"
    return SubstrateCapabilityV9(
        backend_name=str(label),
        backend_url="in-process://coordpy.synthetic_llm",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W64_SUBSTRATE_V9_CAPABILITY_AXES),
        tier=_decide_tier_v9(caps),
        probe_notes=(
            "synthetic deterministic backend; no substrate "
            "access; for hermetic capsule-layer testing only",),
    )


@dataclasses.dataclass(frozen=True)
class SubstrateAdapterV9Matrix:
    probed_at_ns: int
    capabilities: tuple[SubstrateCapabilityV9, ...]

    def by_name(self) -> dict[str, SubstrateCapabilityV9]:
        return {c.backend_name: c for c in self.capabilities}

    def has_v9_full(self) -> bool:
        return any(
            c.tier == W64_SUBSTRATE_TIER_SUBSTRATE_V9_FULL
            for c in self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W64_SUBSTRATE_ADAPTER_V9_SCHEMA_VERSION,
            "probed_at_ns": int(self.probed_at_ns),
            "capabilities": [
                c.to_dict() for c in self.capabilities],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "substrate_adapter_v9_matrix",
            "capabilities": [
                c.cid() for c in self.capabilities],
        })


def probe_all_v9_adapters(
        *, probe_ollama: bool = False,
        probe_openai: bool = False,
        ollama_url: str | None = None,
        openai_url: str | None = None,
        ollama_timeout: float = 1.5,
        openai_timeout: float = 1.5,
) -> SubstrateAdapterV9Matrix:
    caps: list[SubstrateCapabilityV9] = []
    caps.append(probe_tiny_substrate_v9_adapter())
    caps.append(probe_synthetic_v9_adapter())
    if probe_ollama:
        try:
            ollama_cap = probe_ollama_adapter(
                base_url=ollama_url, timeout=ollama_timeout)
            from .substrate_adapter_v2 import (
                probe_w56_substrate_adapter_as_v2,
            )
            from .substrate_adapter_v3 import (
                probe_v2_substrate_adapter_as_v3,
            )
            from .substrate_adapter_v4 import (
                probe_v3_substrate_adapter_as_v4,
            )
            from .substrate_adapter_v5 import (
                probe_v4_substrate_adapter_as_v5,
            )
            from .substrate_adapter_v6 import (
                probe_v5_substrate_adapter_as_v6,
            )
            from .substrate_adapter_v7 import (
                probe_v6_substrate_adapter_as_v7,
            )
            v2 = probe_w56_substrate_adapter_as_v2(ollama_cap)
            v3 = probe_v2_substrate_adapter_as_v3(v2)
            v4 = probe_v3_substrate_adapter_as_v4(v3)
            v5 = probe_v4_substrate_adapter_as_v5(v4)
            v6 = probe_v5_substrate_adapter_as_v6(v5)
            v7 = probe_v6_substrate_adapter_as_v7(v6)
            v8 = probe_v7_substrate_adapter_as_v8(v7)
            v9 = probe_v8_substrate_adapter_as_v9(v8)
            caps.append(v9)
        except Exception:
            pass
    if probe_openai:
        try:
            openai_cap = probe_openai_compatible_adapter(
                base_url=openai_url, timeout=openai_timeout)
            from .substrate_adapter_v2 import (
                probe_w56_substrate_adapter_as_v2,
            )
            from .substrate_adapter_v3 import (
                probe_v2_substrate_adapter_as_v3,
            )
            from .substrate_adapter_v4 import (
                probe_v3_substrate_adapter_as_v4,
            )
            from .substrate_adapter_v5 import (
                probe_v4_substrate_adapter_as_v5,
            )
            from .substrate_adapter_v6 import (
                probe_v5_substrate_adapter_as_v6,
            )
            from .substrate_adapter_v7 import (
                probe_v6_substrate_adapter_as_v7,
            )
            v2 = probe_w56_substrate_adapter_as_v2(openai_cap)
            v3 = probe_v2_substrate_adapter_as_v3(v2)
            v4 = probe_v3_substrate_adapter_as_v4(v3)
            v5 = probe_v4_substrate_adapter_as_v5(v4)
            v6 = probe_v5_substrate_adapter_as_v6(v5)
            v7 = probe_v6_substrate_adapter_as_v7(v6)
            v8 = probe_v7_substrate_adapter_as_v8(v7)
            v9 = probe_v8_substrate_adapter_as_v9(v8)
            caps.append(v9)
        except Exception:
            pass
    return SubstrateAdapterV9Matrix(
        probed_at_ns=int(time.time_ns()),
        capabilities=tuple(caps))


__all__ = [
    "W64_SUBSTRATE_ADAPTER_V9_SCHEMA_VERSION",
    "W64_SUBSTRATE_TIER_SUBSTRATE_V9_FULL",
    "W64_SUBSTRATE_V9_NEW_AXES",
    "W64_SUBSTRATE_V9_CAPABILITY_AXES",
    "SubstrateCapabilityV9",
    "SubstrateAdapterV9Matrix",
    "probe_tiny_substrate_v9_adapter",
    "probe_v8_substrate_adapter_as_v9",
    "probe_synthetic_v9_adapter",
    "probe_all_v9_adapters",
]
