"""W62 — Substrate Adapter V7.

Strictly extends W61's ``coordpy.substrate_adapter_v6``. V7 adds
four new capability axes that the W62 V7 substrate satisfies and
hosted backends do not:

  * ``cache_write_ledger`` — V7 per-(layer, head, slot) cumulative
    L2 of any bridge injection
  * ``logit_lens_probe`` — V7 per-layer logit-lens diagnostic
  * ``attention_receive_delta`` — V7 per-(layer, head, position)
    forward-to-forward attention-receive delta
  * ``replay_trust_ledger`` — V7 per-(layer, head) EMA of replay
    decisions

V7 adds a new top tier:

  * ``substrate_v7_full`` — only the W62 V7 in-repo runtime
    satisfies every axis 1..40.

Hosted backends remain text-only at the HTTP surface;
``W62-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward
the W56..W61 cap unchanged.
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
    SubstrateCapabilityV6,
    W61_SUBSTRATE_ADAPTER_V6_SCHEMA_VERSION,
    W61_SUBSTRATE_TIER_SUBSTRATE_V6_FULL,
    W61_SUBSTRATE_V6_CAPABILITY_AXES,
    W61_SUBSTRATE_V6_NEW_AXES,
    probe_v5_substrate_adapter_as_v6,
)


W62_SUBSTRATE_ADAPTER_V7_SCHEMA_VERSION: str = (
    "coordpy.substrate_adapter_v7.v1")

W62_SUBSTRATE_TIER_SUBSTRATE_V7_FULL: str = "substrate_v7_full"

W62_SUBSTRATE_V7_NEW_AXES: tuple[str, ...] = (
    "cache_write_ledger",
    "logit_lens_probe",
    "attention_receive_delta",
    "replay_trust_ledger",
)

W62_SUBSTRATE_V7_CAPABILITY_AXES: tuple[str, ...] = (
    *W61_SUBSTRATE_V6_CAPABILITY_AXES,
    *W62_SUBSTRATE_V7_NEW_AXES,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class SubstrateCapabilityV7:
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
            "schema": W62_SUBSTRATE_ADAPTER_V7_SCHEMA_VERSION,
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
            "kind": "substrate_capability_v7",
            "capability": self.to_dict()})


def _decide_tier_v7(caps: dict[str, str]) -> str:
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


def probe_tiny_substrate_v7_adapter(
        *, label: str = "tiny_substrate_v7",
) -> SubstrateCapabilityV7:
    caps = {ax: "yes" for ax in W62_SUBSTRATE_V7_CAPABILITY_AXES}
    return SubstrateCapabilityV7(
        backend_name=str(label),
        backend_url="in-process://coordpy.tiny_substrate_v7",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W62_SUBSTRATE_V7_CAPABILITY_AXES),
        tier=_decide_tier_v7(caps),
        probe_notes=(
            "tiny in-repo numpy transformer V7 (9 layers, GQA "
            "8q/4kv, RMSNorm, SwiGLU, d_model=64, d_key=8, "
            "vocab=259) + V6 axes + per-(layer, head, slot) "
            "cache-write ledger + per-layer logit-lens probe + "
            "per-(layer, head, position) attention-receive "
            "delta + per-(layer, head) replay-trust ledger + "
            "KV bridge V7 three-target ridge fit + HSB V6 "
            "three-target ridge fit + prefix V6 drift-curve "
            "predictor + attention V6 two-stage clamp + cache "
            "controller V5 two-objective ridge + trained "
            "repair head + composite_v5 mixture + replay V3 "
            "per-regime head + hidden-vs-KV regime classifier "
            "+ deep substrate hybrid V7 seven-way loop",
            "still NOT a frontier model; still does NOT prove "
            "third-party substrate access",
        ),
    )


def probe_v6_substrate_adapter_as_v7(
        cap: SubstrateCapabilityV6,
) -> SubstrateCapabilityV7:
    base = {ax: val for ax, val in cap.capabilities}
    for ax in W62_SUBSTRATE_V7_NEW_AXES:
        base.setdefault(ax, "no")
    tier = _decide_tier_v7(base)
    return SubstrateCapabilityV7(
        backend_name=str(cap.backend_name),
        backend_url=str(cap.backend_url),
        capabilities=tuple(
            (ax, base.get(ax, "no"))
            for ax in W62_SUBSTRATE_V7_CAPABILITY_AXES),
        tier=str(tier),
        probe_notes=tuple(cap.probe_notes) + (
            "wrapped from W61 substrate adapter V6",),
    )


def probe_synthetic_v7_adapter(
        *, label: str = "synthetic",
) -> SubstrateCapabilityV7:
    caps = {ax: "no" for ax in W62_SUBSTRATE_V7_CAPABILITY_AXES}
    caps["text"] = "yes"
    return SubstrateCapabilityV7(
        backend_name=str(label),
        backend_url="in-process://coordpy.synthetic_llm",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W62_SUBSTRATE_V7_CAPABILITY_AXES),
        tier=_decide_tier_v7(caps),
        probe_notes=(
            "synthetic deterministic backend; no substrate "
            "access; for hermetic capsule-layer testing only",),
    )


@dataclasses.dataclass(frozen=True)
class SubstrateAdapterV7Matrix:
    probed_at_ns: int
    capabilities: tuple[SubstrateCapabilityV7, ...]

    def by_name(self) -> dict[str, SubstrateCapabilityV7]:
        return {c.backend_name: c for c in self.capabilities}

    def has_v7_full(self) -> bool:
        return any(
            c.tier == W62_SUBSTRATE_TIER_SUBSTRATE_V7_FULL
            for c in self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W62_SUBSTRATE_ADAPTER_V7_SCHEMA_VERSION,
            "probed_at_ns": int(self.probed_at_ns),
            "capabilities": [
                c.to_dict() for c in self.capabilities],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "substrate_adapter_v7_matrix",
            "capabilities": [
                c.cid() for c in self.capabilities],
        })


def probe_all_v7_adapters(
        *, probe_ollama: bool = False,
        probe_openai: bool = False,
        ollama_url: str | None = None,
        openai_url: str | None = None,
        ollama_timeout: float = 1.5,
        openai_timeout: float = 1.5,
) -> SubstrateAdapterV7Matrix:
    caps: list[SubstrateCapabilityV7] = []
    caps.append(probe_tiny_substrate_v7_adapter())
    caps.append(probe_synthetic_v7_adapter())
    if probe_ollama:
        try:
            ollama_cap = probe_ollama_adapter(
                base_url=ollama_url, timeout=ollama_timeout)
            # Adapt v1 → v6 → v7.
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
            v2 = probe_w56_substrate_adapter_as_v2(ollama_cap)
            v3 = probe_v2_substrate_adapter_as_v3(v2)
            v4 = probe_v3_substrate_adapter_as_v4(v3)
            v5 = probe_v4_substrate_adapter_as_v5(v4)
            v6 = probe_v5_substrate_adapter_as_v6(v5)
            v7 = probe_v6_substrate_adapter_as_v7(v6)
            caps.append(v7)
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
            v2 = probe_w56_substrate_adapter_as_v2(openai_cap)
            v3 = probe_v2_substrate_adapter_as_v3(v2)
            v4 = probe_v3_substrate_adapter_as_v4(v3)
            v5 = probe_v4_substrate_adapter_as_v5(v4)
            v6 = probe_v5_substrate_adapter_as_v6(v5)
            v7 = probe_v6_substrate_adapter_as_v7(v6)
            caps.append(v7)
        except Exception:
            pass
    return SubstrateAdapterV7Matrix(
        probed_at_ns=int(time.time_ns()),
        capabilities=tuple(caps))


__all__ = [
    "W62_SUBSTRATE_ADAPTER_V7_SCHEMA_VERSION",
    "W62_SUBSTRATE_TIER_SUBSTRATE_V7_FULL",
    "W62_SUBSTRATE_V7_NEW_AXES",
    "W62_SUBSTRATE_V7_CAPABILITY_AXES",
    "SubstrateCapabilityV7",
    "SubstrateAdapterV7Matrix",
    "probe_tiny_substrate_v7_adapter",
    "probe_v6_substrate_adapter_as_v7",
    "probe_synthetic_v7_adapter",
    "probe_all_v7_adapters",
]
