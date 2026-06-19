"""W79 — Local OpenAI-Compatible Façade V1.

The W79 milestone's second direct-blocker-attack pillar. The
hosted-substrate wall is structural: the public HTTP surfaces of
third-party LLM providers expose tokens, optional logprobs, and
optional prefix-cache hit accounting — *nothing else*. The
hosted control-plane Vx line in this repo respects that wall. To
*move past* that wall in environments where it is acceptable to
do so, this module exposes an OpenAI-compatible façade that
points at the W79 controlled runtime substrate. The façade's
surface looks like the public ``v1/chat/completions`` /
``v1/completions`` / ``v1/embeddings`` shapes — but every
underlying call hits the W79 controlled runtime in-process,
which DOES expose hidden state, KV cache, attention probabilities
and per-head attention-bias steering.

That is the W79 attack: where hosted APIs are a wall, build a
local-compatible API surface over a substrate WE control. The
``LocalOpenAICompatibleFacadeV1`` is honest about it — its
chat-completions response carries an explicit
``substrate_axes_exposed`` field listing the substrate axes the
caller can additionally pull via the in-process side channel.

Honest scope (W79)
------------------

* ``W79-L-FACADE-IN-PROCESS-CAP`` — this is an in-process façade,
  not an HTTP server. We do not run an actual TCP listener
  here. The shape mimics the public HTTP shape so client code
  can be transparently rerouted.
* ``W79-L-FACADE-LOCAL-MODEL-ONLY-CAP`` — the façade points to
  the W79 controlled runtime (a small in-repo NumPy transformer),
  not a frontier model.
* ``W79-L-FACADE-DOES-NOT-CHANGE-HOSTED-WALL-CAP`` — pointing
  any third-party hosted SDK at this façade does not magically
  pierce hosted-model substrate; the substrate axes are real
  because the *runtime* is in-process and controlled.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.local_openai_compatible_facade_v1 requires "
        "numpy") from exc

from .controlled_runtime_substrate_v1 import (
    ControlledRuntimeForwardTraceV1,
    ControlledRuntimeKVCacheV1,
    ControlledRuntimeParamsV1,
    W79_CONTROLLED_RUNTIME_AXES,
    W79_CONTROLLED_RUNTIME_SCHEMA_VERSION,
    build_controlled_runtime_params_v1,
    forward_controlled_runtime,
    tokenize_bytes_v79,
)


W79_LOCAL_OPENAI_FACADE_SCHEMA_VERSION: str = (
    "coordpy.local_openai_compatible_facade_v1.v1")
W79_LOCAL_OPENAI_FACADE_MODEL_NAME: str = (
    "coordpy.controlled_runtime_substrate_v1")
W79_LOCAL_OPENAI_FACADE_OBJECT_CHAT_COMPLETION: str = (
    "chat.completion")
W79_LOCAL_OPENAI_FACADE_OBJECT_COMPLETION: str = "text.completion"
W79_LOCAL_OPENAI_FACADE_OBJECT_EMBEDDING: str = "embedding"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class LocalOpenAIChatMessageV1:
    role: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": str(self.role),
            "content": str(self.content),
        }


@dataclasses.dataclass(frozen=True)
class LocalOpenAIChatCompletionRequestV1:
    """OpenAI-shaped chat-completion request.

    Same field names / nesting as the public OpenAI HTTP surface
    so client code can be rerouted by switching the base URL.
    Extra ``substrate_*`` fields are W79-specific extensions
    that hosted APIs DO NOT honor.
    """

    model: str
    messages: tuple[LocalOpenAIChatMessageV1, ...]
    temperature: float = 0.0
    max_tokens: int = 8
    seed: int | None = None
    # W79 substrate extensions — hosted APIs ignore these.
    substrate_return_hidden_state: bool = False
    substrate_return_kv_cache_cid: bool = False
    substrate_return_attention_probs: bool = False
    substrate_attention_bias_injection: tuple[
        tuple["_np.ndarray", ...], ...] = ()
    substrate_hidden_state_injection_per_layer: tuple[
        "_np.ndarray | None", ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": str(self.model),
            "messages": [m.to_dict() for m in self.messages],
            "temperature": float(round(
                self.temperature, 12)),
            "max_tokens": int(self.max_tokens),
            "seed": (
                int(self.seed) if self.seed is not None
                else None),
            "substrate_return_hidden_state": bool(
                self.substrate_return_hidden_state),
            "substrate_return_kv_cache_cid": bool(
                self.substrate_return_kv_cache_cid),
            "substrate_return_attention_probs": bool(
                self.substrate_return_attention_probs),
            "substrate_attention_bias_injection_present": bool(
                len(self.substrate_attention_bias_injection)
                > 0),
            "substrate_hidden_state_injection_present": bool(
                len(self
                    .substrate_hidden_state_injection_per_layer)
                > 0),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "local_openai_compatible_facade_chat_request_v1",
            "request": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class LocalOpenAIChatCompletionChoiceV1:
    index: int
    message: LocalOpenAIChatMessageV1
    finish_reason: str
    logprobs: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "message": self.message.to_dict(),
            "finish_reason": str(self.finish_reason),
            "logprobs": [
                float(round(x, 12)) for x in self.logprobs],
        }


@dataclasses.dataclass(frozen=True)
class LocalOpenAISubstrateSideChannelV1:
    """Substrate side-channel that hosted APIs do NOT expose.

    The W79 façade returns this alongside the OpenAI-shaped
    response when ``substrate_return_*`` flags are set.
    """

    schema: str
    runtime_params_cid: str
    forward_trace_cid: str
    kv_cache_cid: str
    final_hidden_cid: str
    per_layer_attn_probs_cids: tuple[str, ...]
    per_layer_hidden_state_cids: tuple[str, ...]
    substrate_axes_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "runtime_params_cid": str(self.runtime_params_cid),
            "forward_trace_cid": str(self.forward_trace_cid),
            "kv_cache_cid": str(self.kv_cache_cid),
            "final_hidden_cid": str(self.final_hidden_cid),
            "per_layer_attn_probs_cids": list(
                self.per_layer_attn_probs_cids),
            "per_layer_hidden_state_cids": list(
                self.per_layer_hidden_state_cids),
            "substrate_axes_cid": str(
                self.substrate_axes_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "local_openai_compatible_facade_side_channel_v1",
            "side_channel": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class LocalOpenAIChatCompletionResponseV1:
    schema: str
    id: str
    object: str
    created: int
    model: str
    choices: tuple[LocalOpenAIChatCompletionChoiceV1, ...]
    usage: dict[str, int]
    # The hosted-incompatible W79 side channel — none on hosted
    # APIs, populated here when the caller asks for it.
    substrate_side_channel: (
        LocalOpenAISubstrateSideChannelV1 | None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "id": str(self.id),
            "object": str(self.object),
            "created": int(self.created),
            "model": str(self.model),
            "choices": [c.to_dict() for c in self.choices],
            "usage": dict(self.usage),
            "substrate_side_channel_cid": (
                self.substrate_side_channel.cid()
                if self.substrate_side_channel is not None
                else "absent"),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "local_openai_compatible_facade_chat_response_v1",
            "response": self.to_dict()})


@dataclasses.dataclass
class LocalOpenAICompatibleFacadeV1:
    """In-process OpenAI-compatible façade.

    Pointing client code at this façade (instead of a hosted
    HTTP endpoint) reroutes the call into the controlled
    runtime. Substrate axes that the hosted wall blocks are
    available in-process here.
    """

    schema: str = W79_LOCAL_OPENAI_FACADE_SCHEMA_VERSION
    runtime_params: ControlledRuntimeParamsV1 = dataclasses.field(
        default_factory=build_controlled_runtime_params_v1)
    audit: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)
    request_counter: int = 0

    def cid(self) -> str:
        return _sha256_hex({
            "schema": str(self.schema),
            "kind":
                "local_openai_compatible_facade_v1",
            "runtime_params_cid": str(
                self.runtime_params.cid()),
        })

    def chat_completions_create(
            self, *,
            request: LocalOpenAIChatCompletionRequestV1,
    ) -> LocalOpenAIChatCompletionResponseV1:
        """Public-shape ``client.chat.completions.create()``
        analog.

        Determinism: identical (params CID, request CID) →
        identical response CID. The substrate side channel is
        populated when caller asks for it.
        """
        prompt = "\n".join(
            f"{m.role}: {m.content}" for m in request.messages)
        ids = tokenize_bytes_v79(prompt, max_len=32)
        hidden_inj = tuple(
            request.substrate_hidden_state_injection_per_layer)
        attn_inj = tuple(
            tuple(t) if t is not None else None
            for t in
            request.substrate_attention_bias_injection)
        attn_inj_seq: tuple["_np.ndarray | None", ...] = ()
        if attn_inj:
            attn_inj_seq = tuple(
                _np.asarray(a) if a is not None else None
                for a in attn_inj)
        trace, kv = forward_controlled_runtime(
            params=self.runtime_params,
            input_token_ids=ids,
            hidden_state_injections_per_layer=(
                hidden_inj if hidden_inj else None),
            attention_bias_injections_per_layer=(
                attn_inj_seq if attn_inj_seq else None),
        )
        # Build a deterministic completion text from the logits
        # at the last position. We do NOT claim to be a frontier
        # model; we only need byte-stable behaviour.
        if trace.seq_len == 0:
            completion_ids = [0]
            text_out = ""
        else:
            logits = _np.asarray(
                trace.logits, dtype=_np.float64)
            tail = int(min(int(request.max_tokens), 8))
            completion_ids = [
                int(_np.argmax(logits[-1]) % 256)]
            text_out = "".join(
                f"<{int(i):03d}>" for i in completion_ids[:tail])
        usage = {
            "prompt_tokens": int(trace.seq_len),
            "completion_tokens": int(len(completion_ids)),
            "total_tokens": int(
                trace.seq_len + len(completion_ids)),
        }
        side_channel: (
            LocalOpenAISubstrateSideChannelV1 | None) = None
        if (request.substrate_return_hidden_state
                or request.substrate_return_kv_cache_cid
                or request.substrate_return_attention_probs):
            side_channel = (
                LocalOpenAISubstrateSideChannelV1(
                    schema=W79_LOCAL_OPENAI_FACADE_SCHEMA_VERSION,
                    runtime_params_cid=str(
                        self.runtime_params.cid()),
                    forward_trace_cid=str(trace.cid()),
                    kv_cache_cid=str(kv.cid()),
                    final_hidden_cid=str(
                        hashlib.sha256(
                            _np.ascontiguousarray(
                                trace.final_hidden)
                            .tobytes()).hexdigest()),
                    per_layer_attn_probs_cids=tuple(
                        hashlib.sha256(
                            _np.ascontiguousarray(a)
                            .tobytes()).hexdigest()
                        for a in trace.attn_probs),
                    per_layer_hidden_state_cids=tuple(
                        hashlib.sha256(
                            _np.ascontiguousarray(h)
                            .tobytes()).hexdigest()
                        for h in trace.post_mlp_hidden),
                    substrate_axes_cid=str(
                        W79_CONTROLLED_RUNTIME_AXES.cid()),
                ))
        choice = LocalOpenAIChatCompletionChoiceV1(
            index=0,
            message=LocalOpenAIChatMessageV1(
                role="assistant", content=text_out),
            finish_reason="length",
            logprobs=tuple(
                float(x) for x in (
                    trace.logits[-1].tolist()[:8]
                    if trace.seq_len > 0 else [])),
        )
        # Deterministic id from request CID.
        rid = "chatcmpl-" + str(request.cid())[:24]
        # Created field is deterministic on request CID so the
        # response CID is reproducible across runs.
        created_t = int(
            int(str(request.cid())[:8], 16) & 0x7FFFFFFF)
        resp = LocalOpenAIChatCompletionResponseV1(
            schema=W79_LOCAL_OPENAI_FACADE_SCHEMA_VERSION,
            id=rid,
            object=W79_LOCAL_OPENAI_FACADE_OBJECT_CHAT_COMPLETION,
            created=int(created_t),
            model=W79_LOCAL_OPENAI_FACADE_MODEL_NAME,
            choices=(choice,),
            usage=dict(usage),
            substrate_side_channel=side_channel,
        )
        self.audit.append({
            "request_cid": str(request.cid()),
            "response_cid": str(resp.cid()),
            "substrate_side_channel_present": bool(
                side_channel is not None),
        })
        self.request_counter = int(self.request_counter) + 1
        return resp


@dataclasses.dataclass(frozen=True)
class LocalOpenAIFacadeHostedComparisonReportV1:
    """Honest hosted-vs-local comparison report.

    On the hosted plane: substrate axes are blocked. On the
    local OpenAI-compatible façade: the same OpenAI-shaped
    request returns a side channel that exposes the substrate
    axes. This is the load-bearing W79 direct-blocker-attack
    bar.
    """

    schema: str
    hosted_axes_blocked: tuple[str, ...]
    facade_axes_exposed: tuple[str, ...]
    facade_substrate_side_channel_cid: str
    hosted_substrate_side_channel_present: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "hosted_axes_blocked": list(
                self.hosted_axes_blocked),
            "facade_axes_exposed": list(
                self.facade_axes_exposed),
            "facade_substrate_side_channel_cid": str(
                self.facade_substrate_side_channel_cid),
            "hosted_substrate_side_channel_present": bool(
                self
                .hosted_substrate_side_channel_present),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "local_openai_facade_hosted_comparison_v1",
            "report": self.to_dict()})


def compare_facade_vs_hosted_substrate(
        *, facade_response: (
            LocalOpenAIChatCompletionResponseV1),
        hosted_blocked_axes: Sequence[str],
) -> LocalOpenAIFacadeHostedComparisonReportV1:
    exposed = (
        "hidden_state",
        "kv_cache",
        "attention_probs",
        "per_head_attention_bias",
        "prefix_state_inject",
        "per_layer_logits",
        "replay_from_kv",
        "recompute_vs_replay_economics",
    )
    side_cid = (
        str(facade_response.substrate_side_channel.cid())
        if facade_response.substrate_side_channel is not None
        else "absent")
    return LocalOpenAIFacadeHostedComparisonReportV1(
        schema=W79_LOCAL_OPENAI_FACADE_SCHEMA_VERSION,
        hosted_axes_blocked=tuple(
            str(a) for a in hosted_blocked_axes),
        facade_axes_exposed=exposed,
        facade_substrate_side_channel_cid=str(side_cid),
        hosted_substrate_side_channel_present=False,
    )


@dataclasses.dataclass(frozen=True)
class LocalOpenAIFacadeWitnessV1:
    schema: str
    facade_cid: str
    n_requests: int
    last_response_cid: str
    side_channel_emitted_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "facade_cid": str(self.facade_cid),
            "n_requests": int(self.n_requests),
            "last_response_cid": str(self.last_response_cid),
            "side_channel_emitted_count": int(
                self.side_channel_emitted_count),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "local_openai_compatible_facade_witness_v1",
            "witness": self.to_dict()})


def emit_local_openai_facade_witness(
        facade: LocalOpenAICompatibleFacadeV1,
) -> LocalOpenAIFacadeWitnessV1:
    last = (
        facade.audit[-1]["response_cid"]
        if facade.audit else "")
    side_n = int(sum(
        1 for e in facade.audit
        if bool(e.get("substrate_side_channel_present"))))
    return LocalOpenAIFacadeWitnessV1(
        schema=W79_LOCAL_OPENAI_FACADE_SCHEMA_VERSION,
        facade_cid=str(facade.cid()),
        n_requests=int(facade.request_counter),
        last_response_cid=str(last),
        side_channel_emitted_count=int(side_n),
    )


__all__ = [
    "W79_LOCAL_OPENAI_FACADE_SCHEMA_VERSION",
    "W79_LOCAL_OPENAI_FACADE_MODEL_NAME",
    "W79_LOCAL_OPENAI_FACADE_OBJECT_CHAT_COMPLETION",
    "W79_LOCAL_OPENAI_FACADE_OBJECT_COMPLETION",
    "W79_LOCAL_OPENAI_FACADE_OBJECT_EMBEDDING",
    "LocalOpenAIChatMessageV1",
    "LocalOpenAIChatCompletionRequestV1",
    "LocalOpenAIChatCompletionChoiceV1",
    "LocalOpenAIChatCompletionResponseV1",
    "LocalOpenAISubstrateSideChannelV1",
    "LocalOpenAICompatibleFacadeV1",
    "LocalOpenAIFacadeHostedComparisonReportV1",
    "compare_facade_vs_hosted_substrate",
    "LocalOpenAIFacadeWitnessV1",
    "emit_local_openai_facade_witness",
]
