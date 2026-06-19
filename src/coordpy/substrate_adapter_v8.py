"""W63 — Substrate Adapter V8.

Strictly extends W62's ``coordpy.substrate_adapter_v7``. V8 adds
five new capability axes that the W63 V8 substrate satisfies and
hosted backends do not:

  * ``hidden_vs_kv_contention`` — V8 per-(layer, head, slot)
    signed hidden vs KV contention
  * ``hidden_state_confidence_probe`` — V8 per-layer confidence
    in [0, 1] derived from logit-lens entropy
  * ``replay_determinism_channel`` — V8 per-(layer, head, slot)
    cache-write determinism flag
  * ``prefix_reuse_trust_ledger`` — V8 per-(layer, head) EMA of
    prefix-reuse decisions
  * ``cross_layer_head_coupling`` — V8 per-(layer, head, layer,
    head) cosine coupling matrix

V8 adds a new top tier:

  * ``substrate_v8_full`` — only the W63 V8 in-repo runtime
    satisfies every axis.

Hosted backends remain text-only at the HTTP surface;
``W63-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward
the W56..W62 cap unchanged.
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
    SubstrateCapabilityV7,
    W62_SUBSTRATE_ADAPTER_V7_SCHEMA_VERSION,
    W62_SUBSTRATE_TIER_SUBSTRATE_V7_FULL,
    W62_SUBSTRATE_V7_CAPABILITY_AXES,
    W62_SUBSTRATE_V7_NEW_AXES,
    probe_v6_substrate_adapter_as_v7,
)


W63_SUBSTRATE_ADAPTER_V8_SCHEMA_VERSION: str = (
    "coordpy.substrate_adapter_v8.v1")

W63_SUBSTRATE_TIER_SUBSTRATE_V8_FULL: str = "substrate_v8_full"

W63_SUBSTRATE_V8_NEW_AXES: tuple[str, ...] = (
    "hidden_vs_kv_contention",
    "hidden_state_confidence_probe",
    "replay_determinism_channel",
    "prefix_reuse_trust_ledger",
    "cross_layer_head_coupling",
)

W63_SUBSTRATE_V8_CAPABILITY_AXES: tuple[str, ...] = (
    *W62_SUBSTRATE_V7_CAPABILITY_AXES,
    *W63_SUBSTRATE_V8_NEW_AXES,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class SubstrateCapabilityV8:
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
            "schema": W63_SUBSTRATE_ADAPTER_V8_SCHEMA_VERSION,
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
            "kind": "substrate_capability_v8",
            "capability": self.to_dict()})


def _decide_tier_v8(caps: dict[str, str]) -> str:
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


def probe_tiny_substrate_v8_adapter(
        *, label: str = "tiny_substrate_v8",
) -> SubstrateCapabilityV8:
    caps = {ax: "yes" for ax in W63_SUBSTRATE_V8_CAPABILITY_AXES}
    return SubstrateCapabilityV8(
        backend_name=str(label),
        backend_url="in-process://coordpy.tiny_substrate_v8",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W63_SUBSTRATE_V8_CAPABILITY_AXES),
        tier=_decide_tier_v8(caps),
        probe_notes=(
            "tiny in-repo numpy transformer V8 (10 layers, GQA "
            "8q/4kv, RMSNorm, SwiGLU, d_model=64, d_key=8, "
            "vocab=259) + V7 axes + per-(layer, head, slot) "
            "hidden-vs-KV contention + per-layer hidden-state "
            "confidence probe + per-(layer, head, slot) replay "
            "determinism channel + per-(layer, head) prefix-"
            "reuse trust ledger + per-(L, H, L, H) cross-layer-"
            "head coupling matrix + KV bridge V8 four-target "
            "ridge fit + HSB V7 four-target ridge fit + prefix "
            "V7 token-content-conditional drift-curve predictor "
            "+ attention V7 three-stage clamp + cache controller "
            "V6 three-objective ridge + trained retrieval-repair "
            "head + composite_v6 mixture + replay V4 six-regime "
            "head + three-way bridge classifier + deep substrate "
            "hybrid V8 eight-way loop",
            "still NOT a frontier model; still does NOT prove "
            "third-party substrate access",
        ),
    )


def probe_v7_substrate_adapter_as_v8(
        cap: SubstrateCapabilityV7,
) -> SubstrateCapabilityV8:
    base = {ax: val for ax, val in cap.capabilities}
    for ax in W63_SUBSTRATE_V8_NEW_AXES:
        base.setdefault(ax, "no")
    tier = _decide_tier_v8(base)
    return SubstrateCapabilityV8(
        backend_name=str(cap.backend_name),
        backend_url=str(cap.backend_url),
        capabilities=tuple(
            (ax, base.get(ax, "no"))
            for ax in W63_SUBSTRATE_V8_CAPABILITY_AXES),
        tier=str(tier),
        probe_notes=tuple(cap.probe_notes) + (
            "wrapped from W62 substrate adapter V7",),
    )


def probe_synthetic_v8_adapter(
        *, label: str = "synthetic",
) -> SubstrateCapabilityV8:
    caps = {ax: "no" for ax in W63_SUBSTRATE_V8_CAPABILITY_AXES}
    caps["text"] = "yes"
    return SubstrateCapabilityV8(
        backend_name=str(label),
        backend_url="in-process://coordpy.synthetic_llm",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W63_SUBSTRATE_V8_CAPABILITY_AXES),
        tier=_decide_tier_v8(caps),
        probe_notes=(
            "synthetic deterministic backend; no substrate "
            "access; for hermetic capsule-layer testing only",),
    )


@dataclasses.dataclass(frozen=True)
class SubstrateAdapterV8Matrix:
    probed_at_ns: int
    capabilities: tuple[SubstrateCapabilityV8, ...]

    def by_name(self) -> dict[str, SubstrateCapabilityV8]:
        return {c.backend_name: c for c in self.capabilities}

    def has_v8_full(self) -> bool:
        return any(
            c.tier == W63_SUBSTRATE_TIER_SUBSTRATE_V8_FULL
            for c in self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W63_SUBSTRATE_ADAPTER_V8_SCHEMA_VERSION,
            "probed_at_ns": int(self.probed_at_ns),
            "capabilities": [
                c.to_dict() for c in self.capabilities],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "substrate_adapter_v8_matrix",
            "capabilities": [
                c.cid() for c in self.capabilities],
        })


def probe_all_v8_adapters(
        *, probe_ollama: bool = False,
        probe_openai: bool = False,
        ollama_url: str | None = None,
        openai_url: str | None = None,
        ollama_timeout: float = 1.5,
        openai_timeout: float = 1.5,
) -> SubstrateAdapterV8Matrix:
    caps: list[SubstrateCapabilityV8] = []
    caps.append(probe_tiny_substrate_v8_adapter())
    caps.append(probe_synthetic_v8_adapter())
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
            v2 = probe_w56_substrate_adapter_as_v2(ollama_cap)
            v3 = probe_v2_substrate_adapter_as_v3(v2)
            v4 = probe_v3_substrate_adapter_as_v4(v3)
            v5 = probe_v4_substrate_adapter_as_v5(v4)
            v6 = probe_v5_substrate_adapter_as_v6(v5)
            v7 = probe_v6_substrate_adapter_as_v7(v6)
            v8 = probe_v7_substrate_adapter_as_v8(v7)
            caps.append(v8)
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
            v2 = probe_w56_substrate_adapter_as_v2(openai_cap)
            v3 = probe_v2_substrate_adapter_as_v3(v2)
            v4 = probe_v3_substrate_adapter_as_v4(v3)
            v5 = probe_v4_substrate_adapter_as_v5(v4)
            v6 = probe_v5_substrate_adapter_as_v6(v5)
            v7 = probe_v6_substrate_adapter_as_v7(v6)
            v8 = probe_v7_substrate_adapter_as_v8(v7)
            caps.append(v8)
        except Exception:
            pass
    return SubstrateAdapterV8Matrix(
        probed_at_ns=int(time.time_ns()),
        capabilities=tuple(caps))


__all__ = [
    "W63_SUBSTRATE_ADAPTER_V8_SCHEMA_VERSION",
    "W63_SUBSTRATE_TIER_SUBSTRATE_V8_FULL",
    "W63_SUBSTRATE_V8_NEW_AXES",
    "W63_SUBSTRATE_V8_CAPABILITY_AXES",
    "SubstrateCapabilityV8",
    "SubstrateAdapterV8Matrix",
    "probe_tiny_substrate_v8_adapter",
    "probe_v7_substrate_adapter_as_v8",
    "probe_synthetic_v8_adapter",
    "probe_all_v8_adapters",
]
