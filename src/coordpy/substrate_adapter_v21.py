"""W76 — Substrate Adapter V21.

Strictly extends W75's ``coordpy.substrate_adapter_v20``. V21 adds
three new capability axes that the W76 V21 substrate satisfies and
hosted backends do not:

  * ``compound_chain_then_restart_trajectory_cid`` — V21 per-turn
    content-addressed SHA-256 over V20 compound-chain-repair
    trajectory CID + twelve primitive event chains + post-
    compound-chain-restart windows
  * ``compound_chain_then_restart_length_per_layer`` — V21 per-
    layer label in [0..12] (V20's [0..11] +
    restart_after_compound_chain_repair)
  * ``compound_chain_then_restart_pressure_gate_per_layer`` —
    V21 per-layer chain-then-restart-conditioned throttle

V21 adds a new top tier:

  * ``substrate_v21_full`` — only the W76 V21 in-repo runtime
    satisfies every axis.

Hosted backends remain text-only at the HTTP surface;
``W76-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
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
    W66_SUBSTRATE_TIER_SUBSTRATE_V11_FULL,
    W66_SUBSTRATE_V11_NEW_AXES,
)
from .substrate_adapter_v12 import (
    W67_SUBSTRATE_TIER_SUBSTRATE_V12_FULL,
    W67_SUBSTRATE_V12_NEW_AXES,
)
from .substrate_adapter_v13 import (
    W68_SUBSTRATE_TIER_SUBSTRATE_V13_FULL,
    W68_SUBSTRATE_V13_NEW_AXES,
)
from .substrate_adapter_v14 import (
    W69_SUBSTRATE_TIER_SUBSTRATE_V14_FULL,
    W69_SUBSTRATE_V14_NEW_AXES,
)
from .substrate_adapter_v15 import (
    W70_SUBSTRATE_TIER_SUBSTRATE_V15_FULL,
    W70_SUBSTRATE_V15_NEW_AXES,
)
from .substrate_adapter_v16 import (
    W71_SUBSTRATE_TIER_SUBSTRATE_V16_FULL,
    W71_SUBSTRATE_V16_NEW_AXES,
)
from .substrate_adapter_v17 import (
    W72_SUBSTRATE_TIER_SUBSTRATE_V17_FULL,
    W72_SUBSTRATE_V17_NEW_AXES,
)
from .substrate_adapter_v18 import (
    W73_SUBSTRATE_TIER_SUBSTRATE_V18_FULL,
    W73_SUBSTRATE_V18_NEW_AXES,
)
from .substrate_adapter_v19 import (
    W74_SUBSTRATE_TIER_SUBSTRATE_V19_FULL,
    W74_SUBSTRATE_V19_NEW_AXES,
)
from .substrate_adapter_v20 import (
    SubstrateCapabilityV20,
    W75_SUBSTRATE_TIER_SUBSTRATE_V20_FULL,
    W75_SUBSTRATE_V20_CAPABILITY_AXES,
    W75_SUBSTRATE_V20_NEW_AXES,
)


W76_SUBSTRATE_ADAPTER_V21_SCHEMA_VERSION: str = (
    "coordpy.substrate_adapter_v21.v1")

W76_SUBSTRATE_TIER_SUBSTRATE_V21_FULL: str = "substrate_v21_full"

W76_SUBSTRATE_V21_NEW_AXES: tuple[str, ...] = (
    "compound_chain_then_restart_trajectory_cid",
    "compound_chain_then_restart_length_per_layer",
    "compound_chain_then_restart_pressure_gate_per_layer",
)

W76_SUBSTRATE_V21_CAPABILITY_AXES: tuple[str, ...] = (
    *W75_SUBSTRATE_V20_CAPABILITY_AXES,
    *W76_SUBSTRATE_V21_NEW_AXES,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class SubstrateCapabilityV21:
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
            "schema": W76_SUBSTRATE_ADAPTER_V21_SCHEMA_VERSION,
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
            "kind": "substrate_capability_v21",
            "capability": self.to_dict()})


def _decide_tier_v21(caps: dict[str, str]) -> str:
    if caps.get("text") != "yes":
        return SUBSTRATE_TIER_UNREACHABLE
    has_v21 = all(
        caps.get(ax) == "yes" for ax in W76_SUBSTRATE_V21_NEW_AXES)
    has_v20 = all(
        caps.get(ax) == "yes" for ax in W75_SUBSTRATE_V20_NEW_AXES)
    has_v19 = all(
        caps.get(ax) == "yes" for ax in W74_SUBSTRATE_V19_NEW_AXES)
    has_v18 = all(
        caps.get(ax) == "yes" for ax in W73_SUBSTRATE_V18_NEW_AXES)
    has_v17 = all(
        caps.get(ax) == "yes" for ax in W72_SUBSTRATE_V17_NEW_AXES)
    has_v16 = all(
        caps.get(ax) == "yes" for ax in W71_SUBSTRATE_V16_NEW_AXES)
    has_v15 = all(
        caps.get(ax) == "yes" for ax in W70_SUBSTRATE_V15_NEW_AXES)
    has_v14 = all(
        caps.get(ax) == "yes" for ax in W69_SUBSTRATE_V14_NEW_AXES)
    has_v13 = all(
        caps.get(ax) == "yes" for ax in W68_SUBSTRATE_V13_NEW_AXES)
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
    if (has_v21 and has_v20 and has_v19 and has_v18 and has_v17
            and has_v16 and has_v15 and has_v14 and has_v13
            and has_v12 and has_v11 and has_v10 and has_v9
            and has_v8):
        return W76_SUBSTRATE_TIER_SUBSTRATE_V21_FULL
    if (has_v20 and has_v19 and has_v18 and has_v17 and has_v16
            and has_v15 and has_v14 and has_v13 and has_v12
            and has_v11 and has_v10 and has_v9 and has_v8):
        return W75_SUBSTRATE_TIER_SUBSTRATE_V20_FULL
    if (has_v19 and has_v18 and has_v17 and has_v16 and has_v15
            and has_v14 and has_v13 and has_v12 and has_v11
            and has_v10 and has_v9 and has_v8):
        return W74_SUBSTRATE_TIER_SUBSTRATE_V19_FULL
    if (has_v18 and has_v17 and has_v16 and has_v15 and has_v14
            and has_v13 and has_v12 and has_v11 and has_v10
            and has_v9 and has_v8):
        return W73_SUBSTRATE_TIER_SUBSTRATE_V18_FULL
    if (has_v17 and has_v16 and has_v15 and has_v14 and has_v13
            and has_v12 and has_v11 and has_v10
            and has_v9 and has_v8):
        return W72_SUBSTRATE_TIER_SUBSTRATE_V17_FULL
    if (has_v16 and has_v15 and has_v14 and has_v13 and has_v12
            and has_v11 and has_v10 and has_v9 and has_v8):
        return W71_SUBSTRATE_TIER_SUBSTRATE_V16_FULL
    if (has_v15 and has_v14 and has_v13 and has_v12 and has_v11
            and has_v10 and has_v9 and has_v8):
        return W70_SUBSTRATE_TIER_SUBSTRATE_V15_FULL
    if (has_v14 and has_v13 and has_v12 and has_v11
            and has_v10 and has_v9 and has_v8):
        return W69_SUBSTRATE_TIER_SUBSTRATE_V14_FULL
    if (has_v13 and has_v12 and has_v11
            and has_v10 and has_v9 and has_v8):
        return W68_SUBSTRATE_TIER_SUBSTRATE_V13_FULL
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


def probe_tiny_substrate_v21_adapter(
        *, label: str = "tiny_substrate_v21",
) -> SubstrateCapabilityV21:
    caps = {ax: "yes" for ax in W76_SUBSTRATE_V21_CAPABILITY_AXES}
    return SubstrateCapabilityV21(
        backend_name=str(label),
        backend_url="in-process://coordpy.tiny_substrate_v21",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W76_SUBSTRATE_V21_CAPABILITY_AXES),
        tier=_decide_tier_v21(caps),
        probe_notes=(
            "tiny in-repo numpy transformer V21 (23 layers, GQA "
            "8q/4kv, RMSNorm, SwiGLU, d_model=64, d_key=8, "
            "vocab=259) + V20 axes + per-turn compound-chain-"
            "then-restart trajectory CID + per-layer chain-then-"
            "restart-length label + per-layer chain-then-restart-"
            "pressure gate + KV bridge V21 seventeen-target ridge "
            "+ cache V19 sixteen-objective ridge + replay V17 "
            "24-regime ridge + chain-then-restart-aware routing "
            "head + deep substrate hybrid V21 twenty-one-way loop "
            "+ multi-agent substrate coordinator V12 + team-"
            "consensus controller V11",
            "still NOT a frontier model; still does NOT prove "
            "third-party substrate access",
        ),
    )


def probe_v20_substrate_adapter_as_v21(
        cap: SubstrateCapabilityV20,
) -> SubstrateCapabilityV21:
    base = {ax: val for ax, val in cap.capabilities}
    for ax in W76_SUBSTRATE_V21_NEW_AXES:
        base.setdefault(ax, "no")
    tier = _decide_tier_v21(base)
    return SubstrateCapabilityV21(
        backend_name=str(cap.backend_name),
        backend_url=str(cap.backend_url),
        capabilities=tuple(
            (ax, base.get(ax, "no"))
            for ax in W76_SUBSTRATE_V21_CAPABILITY_AXES),
        tier=str(tier),
        probe_notes=tuple(cap.probe_notes) + (
            "wrapped from W75 substrate adapter V20",),
    )


def probe_synthetic_v21_adapter(
        *, label: str = "synthetic",
) -> SubstrateCapabilityV21:
    caps = {ax: "no" for ax in W76_SUBSTRATE_V21_CAPABILITY_AXES}
    caps["text"] = "yes"
    return SubstrateCapabilityV21(
        backend_name=str(label),
        backend_url="in-process://coordpy.synthetic_llm",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in W76_SUBSTRATE_V21_CAPABILITY_AXES),
        tier=_decide_tier_v21(caps),
        probe_notes=(
            "synthetic deterministic backend; no substrate "
            "access; for hermetic capsule-layer testing only",),
    )


@dataclasses.dataclass(frozen=True)
class SubstrateAdapterV21Matrix:
    probed_at_ns: int
    capabilities: tuple[SubstrateCapabilityV21, ...]

    def by_name(self) -> dict[str, SubstrateCapabilityV21]:
        return {c.backend_name: c for c in self.capabilities}

    def has_v21_full(self) -> bool:
        return any(
            c.tier == W76_SUBSTRATE_TIER_SUBSTRATE_V21_FULL
            for c in self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W76_SUBSTRATE_ADAPTER_V21_SCHEMA_VERSION,
            "probed_at_ns": int(self.probed_at_ns),
            "capabilities": [
                c.to_dict() for c in self.capabilities],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "substrate_adapter_v21_matrix",
            "capabilities": [
                c.cid() for c in self.capabilities],
        })


def probe_all_v21_adapters(
        *, probe_ollama: bool = False,
        probe_openai: bool = False,
        ollama_url: str | None = None,
) -> SubstrateAdapterV21Matrix:
    caps: list[SubstrateCapabilityV21] = []
    caps.append(probe_tiny_substrate_v21_adapter())
    caps.append(probe_synthetic_v21_adapter())
    if probe_ollama:
        oll = probe_ollama_adapter(
            ollama_url=ollama_url
            or "http://localhost:11434")
        v21 = probe_v20_substrate_adapter_as_v21(
            SubstrateCapabilityV20(
                backend_name=str(oll.backend_name),
                backend_url=str(oll.backend_url),
                capabilities=tuple(
                    (ax, val) for ax, val in oll.capabilities),
                tier=str(oll.tier),
                probe_notes=tuple(oll.probe_notes)))
        caps.append(v21)
    if probe_openai:
        oai = probe_openai_compatible_adapter()
        v21 = probe_v20_substrate_adapter_as_v21(
            SubstrateCapabilityV20(
                backend_name=str(oai.backend_name),
                backend_url=str(oai.backend_url),
                capabilities=tuple(
                    (ax, val) for ax, val in oai.capabilities),
                tier=str(oai.tier),
                probe_notes=tuple(oai.probe_notes)))
        caps.append(v21)
    return SubstrateAdapterV21Matrix(
        probed_at_ns=int(time.time_ns()),
        capabilities=tuple(caps))


__all__ = [
    "W76_SUBSTRATE_ADAPTER_V21_SCHEMA_VERSION",
    "W76_SUBSTRATE_TIER_SUBSTRATE_V21_FULL",
    "W76_SUBSTRATE_V21_NEW_AXES",
    "W76_SUBSTRATE_V21_CAPABILITY_AXES",
    "SubstrateCapabilityV21",
    "probe_tiny_substrate_v21_adapter",
    "probe_v20_substrate_adapter_as_v21",
    "probe_synthetic_v21_adapter",
    "SubstrateAdapterV21Matrix",
    "probe_all_v21_adapters",
]
