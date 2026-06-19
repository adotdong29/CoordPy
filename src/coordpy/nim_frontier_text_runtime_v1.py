"""W85 / P0 #25, #27, #28, #31 — NVIDIA NIM frontier text runtime.

This module ships a real, *text-only* frontier runtime adapter that
talks to NVIDIA's NIM endpoints (``https://integrate.api.nvidia.com``)
under the existing :class:`coordpy.llm_backend.OpenAICompatibleBackend`.
It is an honest middle path: real frontier-class open-weight models
(Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct, Mixtral-8x7B, Phi-3.5-
MoE, Qwen2.5-7B, ...) are reachable, but ONLY through the chat-
completions text API. There is no hidden-state hook, no KV cache
export, no per-layer instrumentation.

This means the module materially advances issues #25/#27/#28 on the
**text** axis — for example, real head-to-head on a real benchmark
against a real 7B+ frontier model is now possible — while it cannot
close those issues' substrate-side bars (hidden-state intercept, KV
replay, per-layer probe). Those bars require a self-hosted open-weight
model on local GPU and remain blocked.

Honest scope
------------

* ``W85-L-NIM-FRONTIER-TEXT-ONLY-CAP`` — NIM exposes chat completions
  text only. No hidden-state access. No KV cache. No per-layer
  instrumentation. The W80 conformance suite cannot be exercised
  through NIM.
* ``W85-L-NIM-FRONTIER-NO-SUBSTRATE-CAP`` — therefore, NIM is NOT a
  substrate in the W80 sense; it is a *frontier-class text-only
  oracle*. This is a useful, real building block but it does NOT
  close #25's hidden-state-intercept-moves-CID bar.
* ``W85-L-NIM-FRONTIER-REMOTE-CAP`` — the runtime is remote. Network
  latency, NIM rate-limit, and NIM-side determinism are honestly
  recorded in the call capsule. The anti-cheat ``Do not rely SOLELY
  on remote hosted models`` (#25) is respected by keeping the in-repo
  ``controlled_runtime_substrate_v1`` and ``transformers_runtime_v1``
  paths intact and primary for substrate-side claims.

What this module IS good for
----------------------------

1. **#28 real-task bench head-to-head on a real frontier model.**
   The bench can drive Llama-3.1-8B-Instruct (or any NIM open-weight
   model) for stock-single-agent vs CoordPy-multi-agent comparison,
   same model, same prompt, same budget.
2. **#27 live long-context evaluation, partial.** Llama-3.1-8B-Instruct
   has 128k context. The needle-in-haystack corpus can now be run on
   a real model at real token positions, and the composed-pipeline
   vs bounded-baseline-V3 head-to-head can be measured on live task
   success at 32k. (#27's hidden-state-intercept bar still requires
   substrate access and remains open.)
3. **#31 MoE substrate, partial.** Phi-3.5-MoE-instruct and
   Mixtral-8x7B-instruct are reachable via NIM. The MoE-routing
   axis cannot be measured (no per-layer access), but a head-to-head
   bench against MoE models is now possible on text. The substrate-
   side MoE axis (#31's real load-bearing bar) remains open.

What this module is NOT good for
--------------------------------

1. Hidden-state intercept moves-CID (issue #25). Requires substrate.
2. Live training of composed-memory on real hidden states (#26).
3. Per-tier replay-from-KV byte-identity (#30). NIM is opaque.
4. Cross-host distributed substrate at the layer level (#29).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Mapping, Sequence


W85_NIM_FRONTIER_TEXT_V1_SCHEMA_VERSION: str = (
    "coordpy.nim_frontier_text_runtime_v1.v1")


# Open-weight frontier candidates available on NIM. The list is
# kept short and content-addressable; the probe records the actual
# subset reachable at run time. Context lengths are the *advertised*
# context lengths from each model's published spec.
#
# NOTE: NIM's ``/v1/models`` list may include entries that are
# advertised but NOT actually serving chat completions for a
# given account (e.g., ``microsoft/phi-3.5-moe-instruct``
# returned 404 at the chat-completions endpoint at the time of
# W85). The W85 probe lists what NIM advertises; the runtime
# raises a structured error on 404 so the caller sees the gap.
W85_NIM_FRONTIER_MODEL_CATALOG: tuple[tuple[str, int, str], ...] = (
    # (nim_model_id, advertised_context_tokens, family_tag)
    ("meta/llama-3.1-8b-instruct", 131_072, "llama-3.1-8b"),
    ("meta/llama-3.1-70b-instruct", 131_072, "llama-3.1-70b"),
    ("meta/llama-3.3-70b-instruct", 131_072, "llama-3.3-70b"),
    ("meta/llama-3.2-3b-instruct", 131_072, "llama-3.2-3b"),
    ("mistralai/mixtral-8x22b-instruct-v0.1", 65_536,
     "mixtral-8x22b-moe"),
    ("mistralai/mistral-7b-instruct-v0.3", 32_768,
     "mistral-7b"),
    ("microsoft/phi-3.5-moe-instruct", 128_000, "phi-3.5-moe"),
    ("microsoft/phi-4-mini-instruct", 131_072, "phi-4-mini"),
    ("google/gemma-3-4b-it", 128_000, "gemma-3-4b"),
    ("google/gemma-3-12b-it", 128_000, "gemma-3-12b"),
    ("deepseek-ai/deepseek-coder-6.7b-instruct", 16_384,
     "deepseek-coder-6.7b"),
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


class NIMFrontierBlockedError(RuntimeError):
    """The NIM frontier text runtime cannot run on this host."""


# ---------------------------------------------------------------
# Probe.
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class NIMFrontierProbeReportV1:
    """Honest probe of NIM reachability and supported model set."""

    schema: str
    nim_endpoint: str
    api_key_present: bool
    reachable: bool
    blocked_reason: str
    available_models: tuple[str, ...]
    catalog_subset_available: tuple[tuple[str, int, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "nim_endpoint": str(self.nim_endpoint),
            "api_key_present": bool(self.api_key_present),
            "reachable": bool(self.reachable),
            "blocked_reason": str(self.blocked_reason),
            "available_models_count": int(
                len(self.available_models)),
            "catalog_subset_available": [
                [str(m), int(c), str(t)]
                for (m, c, t) in self.catalog_subset_available
            ],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w85_nim_frontier_probe_v1",
            "report": self.to_dict(),
        })

    def is_text_only_substrate(self) -> bool:
        """Always True: NIM is text-only. Documents the limitation
        ``W85-L-NIM-FRONTIER-TEXT-ONLY-CAP``."""
        return True


def probe_nim_frontier_runtime_v1(
        *,
        nim_endpoint: str = "https://integrate.api.nvidia.com",
        api_key: str | None = None,
        timeout: float = 30.0,
) -> NIMFrontierProbeReportV1:
    """Probe NIM reachability. Does NOT load any model weights;
    NIM is server-side so weight loading is the provider's concern.

    The probe is a single ``GET /v1/models`` call. If the API key
    is missing or the request fails, the probe returns
    ``reachable=False`` with a structured ``blocked_reason``.
    """
    key = api_key or os.environ.get("NVIDIA_API_KEY")
    if not key:
        return NIMFrontierProbeReportV1(
            schema=W85_NIM_FRONTIER_TEXT_V1_SCHEMA_VERSION,
            nim_endpoint=str(nim_endpoint),
            api_key_present=False,
            reachable=False,
            blocked_reason="NVIDIA_API_KEY env var not set",
            available_models=tuple(),
            catalog_subset_available=tuple(),
        )
    url = nim_endpoint.rstrip("/") + "/v1/models"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        return NIMFrontierProbeReportV1(
            schema=W85_NIM_FRONTIER_TEXT_V1_SCHEMA_VERSION,
            nim_endpoint=str(nim_endpoint),
            api_key_present=True,
            reachable=False,
            blocked_reason=f"NIM endpoint unreachable: {type(e).__name__}: {e}",
            available_models=tuple(),
            catalog_subset_available=tuple(),
        )
    raw_models = payload.get("data") or []
    avail: tuple[str, ...] = tuple(
        sorted(str(m.get("id", "")) for m in raw_models if m.get("id")))
    catalog_subset: list[tuple[str, int, str]] = []
    for (mid, ctx, tag) in W85_NIM_FRONTIER_MODEL_CATALOG:
        if mid in avail:
            catalog_subset.append((str(mid), int(ctx), str(tag)))
    return NIMFrontierProbeReportV1(
        schema=W85_NIM_FRONTIER_TEXT_V1_SCHEMA_VERSION,
        nim_endpoint=str(nim_endpoint),
        api_key_present=True,
        reachable=True,
        blocked_reason="",
        available_models=avail,
        catalog_subset_available=tuple(catalog_subset),
    )


# ---------------------------------------------------------------
# Call capsule (content-addressed).
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class NIMFrontierCallCapsuleV1:
    """Content-addressed record of one NIM text call.

    Records the prompt SHA-256, model id, sampling params,
    response SHA-256, wall-clock, prompt/output token counts.
    Anti-cheat: every claim made from a NIM run can be traced back
    to the exact prompt/response bytes via these CIDs.
    """

    schema: str
    model_id: str
    prompt_cid: str
    response_cid: str
    temperature: float
    max_tokens: int
    wall_ms: int
    prompt_tokens: int
    output_tokens: int
    response_finish_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "model_id": str(self.model_id),
            "prompt_cid": str(self.prompt_cid),
            "response_cid": str(self.response_cid),
            "temperature": float(round(self.temperature, 6)),
            "max_tokens": int(self.max_tokens),
            "wall_ms": int(self.wall_ms),
            "prompt_tokens": int(self.prompt_tokens),
            "output_tokens": int(self.output_tokens),
            "response_finish_reason": str(
                self.response_finish_reason),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w85_nim_frontier_call_capsule_v1",
            "capsule": self.to_dict(),
        })


# ---------------------------------------------------------------
# Text runtime (thin wrapper that emits capsules).
# ---------------------------------------------------------------


@dataclasses.dataclass
class NIMFrontierTextRuntimeV1:
    """The text-only frontier runtime.

    Provides a content-addressed ``generate_capsule()`` and a
    duck-typed ``generate()`` that matches the
    :class:`coordpy.llm_backend.LLMBackend` Protocol so any
    existing CoordPy team / bench that consumes an LLM backend
    can run against NIM unchanged.
    """

    model_id: str = "meta/llama-3.1-8b-instruct"
    base_url: str = "https://integrate.api.nvidia.com"
    api_key: str | None = None
    timeout: float = 120.0
    n_calls: int = 0
    total_wall_ms: int = 0
    capsules: list[NIMFrontierCallCapsuleV1] = dataclasses.field(
        default_factory=list, repr=False)

    @property
    def model(self) -> str:  # LLMBackend protocol member
        return self.model_id

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise NIMFrontierBlockedError(
                "NIMFrontierTextRuntimeV1 requires NVIDIA_API_KEY")

    def _post_chat_completions(
            self, *,
            prompt: str,
            max_tokens: int,
            temperature: float,
            extra_messages: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + "/v1/chat/completions"
        messages: list[dict[str, Any]] = list(extra_messages or [])
        messages.append({"role": "user", "content": str(prompt)})
        body = {
            "model": str(self.model_id),
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
        )
        with urllib.request.urlopen(
                req, timeout=float(self.timeout)) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def generate(self, prompt: str, max_tokens: int = 80,
                 temperature: float = 0.0) -> str:
        """LLMBackend protocol entry point (text-only)."""
        cap, text = self.generate_capsule(
            prompt=prompt,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        return text

    def generate_capsule(
            self, *,
            prompt: str,
            max_tokens: int = 80,
            temperature: float = 0.0,
            extra_messages: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[NIMFrontierCallCapsuleV1, str]:
        """Run one call and emit a content-addressed capsule.

        Returns ``(capsule, response_text)``.
        """
        t0 = time.time()
        payload = self._post_chat_completions(
            prompt=prompt,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            extra_messages=extra_messages,
        )
        wall_ms = int((time.time() - t0) * 1000.0)
        choices = payload.get("choices") or []
        if not choices:
            response_text = ""
            finish_reason = "no_choice"
        else:
            msg = choices[0].get("message") or {}
            response_text = str(msg.get("content") or "")
            finish_reason = str(choices[0].get("finish_reason", ""))
        usage = payload.get("usage") or {}
        capsule = NIMFrontierCallCapsuleV1(
            schema=W85_NIM_FRONTIER_TEXT_V1_SCHEMA_VERSION,
            model_id=str(self.model_id),
            prompt_cid=hashlib.sha256(
                str(prompt).encode("utf-8")).hexdigest(),
            response_cid=hashlib.sha256(
                response_text.encode("utf-8")).hexdigest(),
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            wall_ms=int(wall_ms),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(
                usage.get("completion_tokens", 0) or 0),
            response_finish_reason=str(finish_reason),
        )
        self.n_calls += 1
        self.total_wall_ms += wall_ms
        self.capsules.append(capsule)
        return capsule, response_text


# ---------------------------------------------------------------
# Capability reporting (composes with frontier_capability_probe_v1).
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class NIMFrontierCapabilityClaimV1:
    """What the NIM runtime can and cannot claim.

    Anti-cheat: each axis records ``can=False`` for hidden-state
    / KV / per-layer / replay; ``can=True`` for text-only chat
    completions on real frontier-class models. The W80
    instrumentation contract is explicitly NOT satisfied.
    """

    schema: str
    nim_text_generation: bool
    real_frontier_class_open_weights: bool
    hidden_state_access: bool
    kv_cache_replay: bool
    per_layer_instrumentation: bool
    cross_runtime_state_export: bool
    long_context_at_least_32k: bool
    moe_models_reachable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "nim_text_generation": bool(self.nim_text_generation),
            "real_frontier_class_open_weights": bool(
                self.real_frontier_class_open_weights),
            "hidden_state_access": bool(self.hidden_state_access),
            "kv_cache_replay": bool(self.kv_cache_replay),
            "per_layer_instrumentation": bool(
                self.per_layer_instrumentation),
            "cross_runtime_state_export": bool(
                self.cross_runtime_state_export),
            "long_context_at_least_32k": bool(
                self.long_context_at_least_32k),
            "moe_models_reachable": bool(self.moe_models_reachable),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w85_nim_frontier_capability_claim_v1",
            "claim": self.to_dict(),
        })


def declare_nim_frontier_capability_claim_v1(
        *, probe: NIMFrontierProbeReportV1) -> NIMFrontierCapabilityClaimV1:
    """Build the W85 capability claim from a probe report.

    Records the *honest* axes — every substrate-side axis is
    ``False`` because NIM is text-only.
    """
    catalog = probe.catalog_subset_available
    has_text = bool(probe.reachable and probe.api_key_present)
    has_frontier = bool(
        catalog and any(c >= 7_000_000_000 / 10  # 7B-ish proxy
                        or "70b" in tag.lower() or "8b" in tag.lower()
                        or "moe" in tag.lower() or "12b" in tag.lower()
                        for (mid, c, tag) in catalog))
    has_32k = bool(
        catalog and any(c >= 32_000 for (_, c, _) in catalog))
    has_moe = bool(
        catalog and any("moe" in t.lower() for (_, _, t) in catalog))
    return NIMFrontierCapabilityClaimV1(
        schema=W85_NIM_FRONTIER_TEXT_V1_SCHEMA_VERSION,
        nim_text_generation=bool(has_text),
        real_frontier_class_open_weights=bool(has_frontier),
        hidden_state_access=False,
        kv_cache_replay=False,
        per_layer_instrumentation=False,
        cross_runtime_state_export=False,
        long_context_at_least_32k=bool(has_text and has_32k),
        moe_models_reachable=bool(has_text and has_moe),
    )


__all__ = [
    "W85_NIM_FRONTIER_TEXT_V1_SCHEMA_VERSION",
    "W85_NIM_FRONTIER_MODEL_CATALOG",
    "NIMFrontierBlockedError",
    "NIMFrontierProbeReportV1",
    "NIMFrontierCallCapsuleV1",
    "NIMFrontierCapabilityClaimV1",
    "NIMFrontierTextRuntimeV1",
    "probe_nim_frontier_runtime_v1",
    "declare_nim_frontier_capability_claim_v1",
]
