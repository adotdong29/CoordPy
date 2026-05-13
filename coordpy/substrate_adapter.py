"""W56 M2 — Substrate adapter.

This module honestly classifies backends by what they can expose
to a coordinated runtime. The honest answer for most third-party
hosted models is *text_only*: an HTTP request goes in, a
generated string comes out, and nothing in between is visible.
Some local stacks expose **embeddings** (Ollama's
``/api/embeddings``) or **logprobs** (OpenAI's
``logprobs=true``). Only an in-repo runtime (``tiny_substrate``)
can expose **hidden states**, **KV cache**, and **attention
weights** for read/write.

This adapter does NOT promise it can breach third-party substrate.
It records what is and is not accessible, so the W56 loop can pick
the right path without overclaiming.

Capability axes (in increasing depth):

  1. ``text``                — generate(prompt, ...) -> str
  2. ``embeddings``          — encode(text) -> vector
  3. ``logprobs``            — generate exposes per-token logprobs
  4. ``logits``              — full vocabulary logits per step
  5. ``hidden_states``       — per-layer residual stream tensors
  6. ``kv_cache_read``       — read-side KV cache access
  7. ``kv_cache_write``      — write-side KV cache injection
  8. ``attention_weights``   — per-head causal attention maps

Honest scope: only the ``tiny_substrate`` reaches axes 4-8. The
adapter's job is to make that explicit and reproducible.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Sequence


SUBSTRATE_ADAPTER_SCHEMA_VERSION: str = (
    "coordpy.substrate_adapter.v1")


SUBSTRATE_CAPABILITY_AXES: tuple[str, ...] = (
    "text",
    "embeddings",
    "logprobs",
    "logits",
    "hidden_states",
    "kv_cache_read",
    "kv_cache_write",
    "attention_weights",
)


SUBSTRATE_TIER_TEXT_ONLY: str = "text_only"
SUBSTRATE_TIER_EMBEDDINGS_ONLY: str = "embeddings_only"
SUBSTRATE_TIER_LOGITS_ONLY: str = "logits_only"
SUBSTRATE_TIER_SUBSTRATE_FULL: str = "substrate_full"
SUBSTRATE_TIER_UNREACHABLE: str = "unreachable"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class SubstrateCapability:
    """Per-backend capability matrix.

    Each axis is one of: ``"yes"``, ``"no"``, ``"unknown"``. The
    ``tier`` is the highest tier the backend reaches.
    """

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
            "schema": SUBSTRATE_ADAPTER_SCHEMA_VERSION,
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
            "kind": "substrate_capability",
            "capability": self.to_dict()})


def _decide_tier(
        caps: dict[str, str],
) -> str:
    """Map a capability dict to a single tier name.

    The tier is the deepest capability with ``"yes"``. ``"unknown"``
    is treated as ``"no"`` (we do not overclaim).
    """
    if caps.get("text") != "yes":
        return SUBSTRATE_TIER_UNREACHABLE
    if (caps.get("hidden_states") == "yes"
            and caps.get("kv_cache_read") == "yes"
            and caps.get("kv_cache_write") == "yes"
            and caps.get("attention_weights") == "yes"
            and caps.get("logits") == "yes"):
        return SUBSTRATE_TIER_SUBSTRATE_FULL
    if caps.get("logits") == "yes" or caps.get("logprobs") == "yes":
        return SUBSTRATE_TIER_LOGITS_ONLY
    if caps.get("embeddings") == "yes":
        return SUBSTRATE_TIER_EMBEDDINGS_ONLY
    return SUBSTRATE_TIER_TEXT_ONLY


def probe_tiny_substrate_adapter(
        *, label: str = "tiny_substrate",
) -> SubstrateCapability:
    """The honest answer for the in-repo tiny substrate.

    Reaches axes 1..8. The ``probe_notes`` record that this is
    a tiny in-repo research runtime, not a frontier model.
    """
    caps = {ax: "yes" for ax in SUBSTRATE_CAPABILITY_AXES}
    return SubstrateCapability(
        backend_name=str(label),
        backend_url="in-process://coordpy.tiny_substrate",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in SUBSTRATE_CAPABILITY_AXES),
        tier=_decide_tier(caps),
        probe_notes=(
            "tiny in-repo numpy transformer; real KV / hidden "
            "states / attention; bounded scope (2 layers, 4 "
            "heads, d_model=32, vocab=259)",
            "this is NOT a frontier model and does NOT prove "
            "third-party substrate access",
        ),
    )


def probe_synthetic_adapter(
        *, label: str = "synthetic",
) -> SubstrateCapability:
    """Synthetic backend: text-only, deterministic, no substrate."""
    caps = {ax: "no" for ax in SUBSTRATE_CAPABILITY_AXES}
    caps["text"] = "yes"
    return SubstrateCapability(
        backend_name=str(label),
        backend_url="in-process://coordpy.synthetic_llm",
        capabilities=tuple(
            (ax, caps[ax])
            for ax in SUBSTRATE_CAPABILITY_AXES),
        tier=_decide_tier(caps),
        probe_notes=(
            "synthetic deterministic backend; no substrate "
            "access; for hermetic capsule-layer testing only",
        ),
    )


def probe_ollama_adapter(
        *, base_url: str | None = None,
        timeout: float = 1.5,
) -> SubstrateCapability:
    """Probe an Ollama backend's HTTP capability surface honestly.

    Ollama exposes:
      * ``/api/generate`` — text generation;
      * ``/api/embeddings`` — embedding vectors for the
        ``embedding`` family of models;
      * (in some versions) per-token logprobs on the generate
        response when ``options={"logits_all": True}``.

    It does NOT expose hidden states, KV cache, or attention
    weights. The probe records what was actually reachable; if
    the HTTP endpoint is unreachable, the capability is
    ``unreachable`` and the tier is set accordingly.

    The probe is bounded by a short ``timeout`` so the W56
    benchmark can run hermetically (no network → unreachable,
    not test failure).
    """
    url = (
        base_url
        or os.environ.get("COORDPY_OLLAMA_URL")
        or "http://localhost:11434")
    notes: list[str] = []
    caps: dict[str, str] = {ax: "no" for ax in SUBSTRATE_CAPABILITY_AXES}
    # Probe /api/tags as a liveness check.
    try:
        req = urllib.request.Request(
            url.rstrip("/") + "/api/tags",
            headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        notes.append(
            f"ollama reachable; n_models={len(payload.get('models', []))}")
        caps["text"] = "yes"
        # Embeddings endpoint exists on Ollama for embed-capable models.
        caps["embeddings"] = "unknown"
        notes.append(
            "ollama /api/embeddings reachable only on embedding models")
        # Hidden states / KV / attention are not exposed.
        notes.append(
            "ollama HTTP surface does not expose hidden states, "
            "KV cache, or attention weights")
    except (urllib.error.URLError, urllib.error.HTTPError,
            ConnectionError, TimeoutError, OSError) as exc:
        notes.append(f"ollama unreachable at {url}: {type(exc).__name__}")
        caps["text"] = "no"
    tier = _decide_tier(caps)
    return SubstrateCapability(
        backend_name="ollama",
        backend_url=str(url),
        capabilities=tuple(
            (ax, caps[ax])
            for ax in SUBSTRATE_CAPABILITY_AXES),
        tier=tier,
        probe_notes=tuple(notes),
    )


def probe_openai_compatible_adapter(
        *, base_url: str | None = None,
        timeout: float = 1.5,
) -> SubstrateCapability:
    """Probe an OpenAI-compatible endpoint's capability surface.

    OpenAI-compatible APIs expose:
      * ``/v1/chat/completions`` — text generation;
      * (some) per-token logprobs via ``logprobs=true``;
      * ``/v1/embeddings`` — embedding endpoint on embedding
        models.

    They do NOT expose hidden states, KV cache, or attention
    weights. The probe records the conservative answer.
    """
    url = (
        base_url
        or os.environ.get("COORDPY_API_BASE_URL")
        or "https://api.openai.com")
    caps: dict[str, str] = {ax: "no" for ax in SUBSTRATE_CAPABILITY_AXES}
    notes: list[str] = []
    # We do not actually call the API in a probe; this is a
    # declarative capability. Honesty matters more than depth here.
    caps["text"] = "yes"
    caps["embeddings"] = "unknown"
    caps["logprobs"] = "unknown"
    notes.append(
        "openai-compatible HTTP surface does not expose hidden "
        "states, KV cache, or attention weights")
    return SubstrateCapability(
        backend_name="openai_compatible",
        backend_url=str(url),
        capabilities=tuple(
            (ax, caps[ax])
            for ax in SUBSTRATE_CAPABILITY_AXES),
        tier=_decide_tier(caps),
        probe_notes=tuple(notes),
    )


@dataclasses.dataclass(frozen=True)
class SubstrateAdapterMatrix:
    """A snapshot of per-backend capabilities at a probe time.

    The matrix is content-addressed for the W56 envelope; the W56
    loop reads its tier per backend and chooses whether to use
    substrate-bridge mechanisms.
    """

    probed_at_ns: int
    capabilities: tuple[SubstrateCapability, ...]

    def by_name(self) -> dict[str, SubstrateCapability]:
        return {c.backend_name: c for c in self.capabilities}

    def has_substrate_full(self) -> bool:
        return any(
            c.tier == SUBSTRATE_TIER_SUBSTRATE_FULL
            for c in self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SUBSTRATE_ADAPTER_SCHEMA_VERSION,
            "probed_at_ns": int(self.probed_at_ns),
            "capabilities": [
                c.to_dict() for c in self.capabilities],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "substrate_adapter_matrix",
            "capabilities": [
                c.cid() for c in self.capabilities],
        })


def probe_all_adapters(
        *, tiny_label: str = "tiny_substrate",
        probe_ollama: bool = True,
        probe_openai: bool = True,
        ollama_url: str | None = None,
        openai_url: str | None = None,
        ollama_timeout: float = 1.5,
) -> SubstrateAdapterMatrix:
    """Probe every adapter and return the combined matrix."""
    caps: list[SubstrateCapability] = [
        probe_tiny_substrate_adapter(label=tiny_label),
        probe_synthetic_adapter(),
    ]
    if probe_ollama:
        caps.append(probe_ollama_adapter(
            base_url=ollama_url, timeout=float(ollama_timeout)))
    if probe_openai:
        caps.append(probe_openai_compatible_adapter(
            base_url=openai_url))
    return SubstrateAdapterMatrix(
        probed_at_ns=time.monotonic_ns(),
        capabilities=tuple(caps),
    )


__all__ = [
    "SUBSTRATE_ADAPTER_SCHEMA_VERSION",
    "SUBSTRATE_CAPABILITY_AXES",
    "SUBSTRATE_TIER_TEXT_ONLY",
    "SUBSTRATE_TIER_EMBEDDINGS_ONLY",
    "SUBSTRATE_TIER_LOGITS_ONLY",
    "SUBSTRATE_TIER_SUBSTRATE_FULL",
    "SUBSTRATE_TIER_UNREACHABLE",
    "SubstrateCapability",
    "SubstrateAdapterMatrix",
    "probe_tiny_substrate_adapter",
    "probe_synthetic_adapter",
    "probe_ollama_adapter",
    "probe_openai_compatible_adapter",
    "probe_all_adapters",
]
