"""W81 / P1 #7 — Deployable Substrate Gateway V1.

The W79 ``local_openai_compatible_facade_v1`` proved the shape
(rerouting OpenAI-shaped client code into a controlled runtime
that *does* expose substrate axes), but it was honestly limited:
in-process only, no transport, no auth, no side-channel
endpoints — a proof-of-shape, not a deployable gateway.

W81 P1 #7 closes that gap. ``deployable_substrate_gateway_v1``
is a real deployable gateway with:

* HTTP transport via ``http.server`` (Python stdlib — no
  third-party server dependency)
* OpenAI-compatible request/response shapes for
  ``/v1/chat/completions`` and ``/v1/completions``
* Explicit substrate side-channel endpoints
  (``/v1/substrate/forward``, ``/v1/substrate/replay``,
  ``/v1/substrate/conformance``, ``/v1/substrate/parity``,
  ``/v1/substrate/capabilities``) — substrate access is
  surfaced via dedicated routes, not buried in a generic
  envelope
* Content-addressing on every request and response (request
  CID + response CID); the audit log is content-addressed too
* Deterministic replay hooks — every explicit substrate trace
  response (and every chat-completion response with
  side-channel opt-in) carries a
  ``substrate_forward_trace_cid`` + ``kv_cache_cid`` pair that
  the ``/v1/substrate/replay`` endpoint will accept to
  reproduce the original forward trace
* A bearer-token auth shim (config-driven; ``None`` means
  open for local dev), so the gateway can be locked down in
  research deployments
* A programmatic dispatch path (``dispatch``) used by tests
  and by callers that don't want to bind a TCP port

The gateway preserves all four design constraints from #7:

1. *Honesty.* The gateway never claims hosted-substrate access.
   Every substrate axis it exposes is real because the runtime
   underneath is *controlled* (`controlled_runtime_substrate_v1`
   in V1; the HF transformers runtime can be slotted in via
   the same shape — see W80 parity matrix).
2. *Explicit side channel.* Substrate access is routed through
   dedicated endpoints, never silently piggybacking on the
   chat-completion shape unless the caller opts in.
3. *Side channel is not magical.* Each substrate endpoint
   returns content-addressed bytes — caller can verify the
   returned trace bytes themselves.
4. *Minimal client changes.* The ``/v1/chat/completions``
   shape matches the OpenAI public surface enough that an
   OpenAI client whose ``base_url`` is pointed at the gateway
   gets a valid response; substrate endpoints are additive.

Honest scope (W81)
------------------

* ``W81-L-GATEWAY-V1-LOCAL-MODEL-ONLY-CAP`` — V1 routes to the
  W79 controlled runtime substrate (`controlled_runtime_substrate_v1`).
  The W80 HF transformers runtime can be slotted in too, but
  V1 default is the in-repo NumPy runtime so CI doesn't pay
  the transformers cost.
* ``W81-L-GATEWAY-V1-AUTH-SIMPLE-CAP`` — the auth shim is a
  bearer-token *equality* check. Not a full OAuth / signed-JWT
  story. Sufficient for research deployment, not for the
  open internet.
* ``W81-L-GATEWAY-V1-DOES-NOT-CHANGE-HOSTED-WALL-CAP`` —
  identical to the V1 facade's caveat: rerouting a hosted SDK
  at this gateway does NOT pierce third-party hosted-model
  substrate. The substrate axes are real only because the
  underlying *runtime* is controlled in-process.
* ``W81-L-GATEWAY-V1-NO-STREAMING-CAP`` — V1 does NOT
  implement Server-Sent-Events streaming. ``stream=true``
  requests are answered with a single non-streaming JSON
  body (with an explicit ``streaming_supported: false`` field
  in the response). The shape stays OpenAI-compatible; the
  delivery model differs.
* ``W81-L-GATEWAY-V1-NO-TLS-CAP`` — V1 binds plain HTTP. TLS
  is out of V1 scope; deploy behind a reverse proxy if you
  need it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.deployable_substrate_gateway_v1 requires numpy"
        ) from exc

from .controlled_runtime_substrate_v1 import (
    ControlledRuntimeKVCacheV1,
    ControlledRuntimeParamsV1,
    W79_CONTROLLED_RUNTIME_AXES,
    build_controlled_runtime_params_v1,
    forward_controlled_runtime,
    replay_from_kv_cache,
    tokenize_bytes_v79,
)
from .local_openai_compatible_facade_v1 import (
    LocalOpenAIChatCompletionRequestV1,
    LocalOpenAIChatMessageV1,
    LocalOpenAICompatibleFacadeV1,
)
from .runtime_instrumentation_v1 import (
    ControlledRuntimeInstrumentationAdapterV1,
    run_instrumentation_conformance_v1,
)


W81_GATEWAY_V1_SCHEMA_VERSION: str = (
    "coordpy.deployable_substrate_gateway_v1.v1")

W81_GATEWAY_PATH_CHAT_COMPLETIONS: str = "/v1/chat/completions"
W81_GATEWAY_PATH_COMPLETIONS: str = "/v1/completions"
W81_GATEWAY_PATH_SUBSTRATE_FORWARD: str = "/v1/substrate/forward"
W81_GATEWAY_PATH_SUBSTRATE_REPLAY: str = "/v1/substrate/replay"
W81_GATEWAY_PATH_SUBSTRATE_CONFORMANCE: str = (
    "/v1/substrate/conformance")
W81_GATEWAY_PATH_SUBSTRATE_CAPABILITIES: str = (
    "/v1/substrate/capabilities")
W81_GATEWAY_PATH_HEALTHZ: str = "/healthz"

W81_GATEWAY_ROUTES: tuple[str, ...] = (
    W81_GATEWAY_PATH_CHAT_COMPLETIONS,
    W81_GATEWAY_PATH_COMPLETIONS,
    W81_GATEWAY_PATH_SUBSTRATE_FORWARD,
    W81_GATEWAY_PATH_SUBSTRATE_REPLAY,
    W81_GATEWAY_PATH_SUBSTRATE_CONFORMANCE,
    W81_GATEWAY_PATH_SUBSTRATE_CAPABILITIES,
    W81_GATEWAY_PATH_HEALTHZ,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _ndarray_cid(arr: Any) -> str:
    a = _np.ascontiguousarray(_np.asarray(arr, dtype=_np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class GatewayConfigV1:
    """Deployment config for the gateway.

    ``bearer_token`` is the auth shim. ``None`` means open
    (local-dev). When set, requests MUST present an
    ``Authorization: Bearer <token>`` header.

    ``allow_substrate_side_channel`` toggles whether the chat
    completion endpoint will return substrate axes when the
    caller asks for them. Set to ``False`` to lock the
    gateway down to a pure-text shape for clients that should
    not learn about the side channel.
    """

    schema: str = W81_GATEWAY_V1_SCHEMA_VERSION
    bearer_token: str | None = None
    allow_substrate_side_channel: bool = True
    model_name: str = "coordpy.controlled_runtime_substrate_v1"

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w81_gateway_config_v1",
            "schema": str(self.schema),
            "auth_required": bool(self.bearer_token is not None),
            "allow_substrate_side_channel": bool(
                self.allow_substrate_side_channel),
            "model_name": str(self.model_name),
        })


@dataclasses.dataclass(frozen=True)
class GatewayResponseV1:
    """Result of a single dispatch call.

    ``status`` is an HTTP-style int (200, 400, 401, 404, 500).
    ``body`` is a JSON-serialisable dict. ``request_cid`` and
    ``response_cid`` content-address the request and response
    bodies respectively — the audit log is keyed on this pair.
    """

    schema: str
    path: str
    status: int
    body: dict[str, Any]
    request_cid: str
    response_cid: str

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w81_gateway_response_v1",
            "schema": str(self.schema),
            "path": str(self.path),
            "status": int(self.status),
            "request_cid": str(self.request_cid),
            "response_cid": str(self.response_cid),
        })


@dataclasses.dataclass
class DeployableSubstrateGatewayV1:
    """Deployable HTTP-shaped gateway over a controlled runtime.

    The gateway can be exercised in two modes:

    1. *Programmatic* — call ``gateway.dispatch(path, body,
       auth)`` directly. Used by tests, benchmarks, and
       in-process clients.
    2. *HTTP* — call ``gateway.serve_forever(host, port)`` to
       bind a real TCP listener.

    Both modes go through the same dispatch core, so the
    HTTP path is *literally* the programmatic path plus
    request parsing.
    """

    config: GatewayConfigV1 = dataclasses.field(
        default_factory=lambda: GatewayConfigV1())
    runtime_params: ControlledRuntimeParamsV1 = dataclasses.field(
        default_factory=build_controlled_runtime_params_v1)
    facade: LocalOpenAICompatibleFacadeV1 = dataclasses.field(
        default_factory=LocalOpenAICompatibleFacadeV1)
    audit: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)
    replay_registry_by_trace_cid: dict[str, dict[str, Any]] = (
        dataclasses.field(default_factory=dict))
    replay_registry_by_kv_cid: dict[str, dict[str, Any]] = (
        dataclasses.field(default_factory=dict))
    request_counter: int = 0

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w81_deployable_substrate_gateway_v1",
            "schema": str(W81_GATEWAY_V1_SCHEMA_VERSION),
            "config_cid": str(self.config.cid()),
            "runtime_params_cid": str(
                self.runtime_params.cid()),
        })

    # ------------------------------------------------------------
    # Dispatch core
    # ------------------------------------------------------------

    def dispatch(
            self, *,
            path: str,
            body: dict[str, Any] | None,
            auth_header: str | None = None,
    ) -> GatewayResponseV1:
        """Single entry point. All routes flow through here."""
        body_dict = dict(body or {})
        request_cid = _sha256_hex({
            "path": str(path), "body": body_dict})
        # Healthz is unauthenticated by design.
        if path == W81_GATEWAY_PATH_HEALTHZ:
            return self._respond(
                path=path, status=200,
                body={
                    "ok": True,
                    "gateway_cid": str(self.cid()),
                    "schema": str(W81_GATEWAY_V1_SCHEMA_VERSION),
                    "model": str(self.config.model_name),
                    "routes": list(W81_GATEWAY_ROUTES),
                },
                request_cid=request_cid)
        # Everything else respects the auth shim.
        if self.config.bearer_token is not None:
            expected = f"Bearer {self.config.bearer_token}"
            if (auth_header or "") != expected:
                return self._respond(
                    path=path, status=401,
                    body={
                        "error": {
                            "type": "unauthorized",
                            "message": (
                                "missing or invalid bearer "
                                "token"),
                        }},
                    request_cid=request_cid)
        if path == W81_GATEWAY_PATH_CHAT_COMPLETIONS:
            return self._handle_chat_completions(
                body=body_dict, request_cid=request_cid)
        if path == W81_GATEWAY_PATH_COMPLETIONS:
            return self._handle_completions(
                body=body_dict, request_cid=request_cid)
        if path == W81_GATEWAY_PATH_SUBSTRATE_FORWARD:
            return self._handle_substrate_forward(
                body=body_dict, request_cid=request_cid)
        if path == W81_GATEWAY_PATH_SUBSTRATE_REPLAY:
            return self._handle_substrate_replay(
                body=body_dict, request_cid=request_cid)
        if path == W81_GATEWAY_PATH_SUBSTRATE_CONFORMANCE:
            return self._handle_substrate_conformance(
                body=body_dict, request_cid=request_cid)
        if path == W81_GATEWAY_PATH_SUBSTRATE_CAPABILITIES:
            return self._handle_substrate_capabilities(
                body=body_dict, request_cid=request_cid)
        return self._respond(
            path=path, status=404,
            body={
                "error": {
                    "type": "not_found",
                    "message": f"no such route: {path}",
                    "valid_routes": list(W81_GATEWAY_ROUTES),
                }},
            request_cid=request_cid)

    def _respond(
            self, *, path: str, status: int,
            body: dict[str, Any], request_cid: str,
    ) -> GatewayResponseV1:
        # Embed request_cid into the response so the caller
        # can chain the audit.
        body_full = dict(body)
        body_full.setdefault("request_cid", request_cid)
        response_cid = _sha256_hex({
            "path": str(path), "status": int(status),
            "body": body_full})
        resp = GatewayResponseV1(
            schema=W81_GATEWAY_V1_SCHEMA_VERSION,
            path=str(path), status=int(status),
            body=dict(body_full),
            request_cid=str(request_cid),
            response_cid=str(response_cid))
        self.audit.append({
            "path": str(path),
            "status": int(status),
            "request_cid": str(request_cid),
            "response_cid": str(response_cid),
        })
        self.request_counter = int(self.request_counter) + 1
        return resp

    def _normalize_split(
            self, *,
            token_ids: list[int],
            requested_split: int | None = None,
    ) -> int:
        split = int(requested_split) if requested_split is not None else 1
        if split < 1 or split >= max(2, len(token_ids)):
            split = max(1, len(token_ids) // 2)
        return int(split)

    def _register_replay_artifact(
            self, *,
            source_path: str,
            prompt: str,
            token_ids: list[int],
            split_at: int,
            forward_trace_cid: str,
            kv_cache_cid: str,
    ) -> None:
        artifact = {
            "source_path": str(source_path),
            "prompt": str(prompt),
            "token_ids": [int(t) for t in token_ids],
            "split_at": int(split_at),
            "forward_trace_cid": str(forward_trace_cid),
            "kv_cache_cid": str(kv_cache_cid),
        }
        self.replay_registry_by_trace_cid[str(
            forward_trace_cid)] = dict(artifact)
        self.replay_registry_by_kv_cid[str(
            kv_cache_cid)] = dict(artifact)

    def _resolve_replay_artifact(
            self, *,
            forward_trace_cid: str | None,
            kv_cache_cid: str | None,
    ) -> dict[str, Any] | None:
        by_trace = None
        by_kv = None
        if forward_trace_cid:
            by_trace = self.replay_registry_by_trace_cid.get(
                str(forward_trace_cid))
            if by_trace is None:
                raise KeyError(
                    "unknown forward_trace_cid for replay")
        if kv_cache_cid:
            by_kv = self.replay_registry_by_kv_cid.get(
                str(kv_cache_cid))
            if by_kv is None:
                raise KeyError("unknown kv_cache_cid for replay")
        if by_trace is not None and by_kv is not None:
            if (str(by_trace["forward_trace_cid"])
                    != str(by_kv["forward_trace_cid"])
                    or str(by_trace["kv_cache_cid"])
                    != str(by_kv["kv_cache_cid"])):
                raise ValueError(
                    "forward_trace_cid and kv_cache_cid refer "
                    "to different registered traces")
        return (
            dict(by_trace) if by_trace is not None
            else (dict(by_kv) if by_kv is not None else None))

    # ------------------------------------------------------------
    # /v1/chat/completions
    # ------------------------------------------------------------

    def _handle_chat_completions(
            self, *, body: dict[str, Any],
            request_cid: str,
    ) -> GatewayResponseV1:
        try:
            messages = body.get("messages") or []
            msgs: list[LocalOpenAIChatMessageV1] = []
            for m in messages:
                msgs.append(LocalOpenAIChatMessageV1(
                    role=str(m.get("role", "user")),
                    content=str(m.get("content", ""))))
            max_tokens = int(body.get("max_tokens", 4))
            substrate_opts = body.get(
                "substrate_options") or {}
            want_side_channel = (
                bool(self.config.allow_substrate_side_channel)
                and bool(
                    substrate_opts.get("return_side_channel",
                                       False)))
            req = LocalOpenAIChatCompletionRequestV1(
                model=str(self.config.model_name),
                messages=tuple(msgs),
                temperature=0.0,
                max_tokens=int(max_tokens),
                seed=None,
                substrate_return_hidden_state=bool(
                    want_side_channel),
                substrate_return_kv_cache_cid=bool(
                    want_side_channel),
                substrate_return_attention_probs=bool(
                    want_side_channel),
                substrate_hidden_state_injection_per_layer=(),
                substrate_attention_bias_injection=(),
            )
            resp = self.facade.chat_completions_create(
                request=req)
            prompt = "\n".join(
                f"{m.role}: {m.content}" for m in msgs)
            stream_requested = bool(body.get("stream", False))
            # OpenAI shape: system_fingerprint is a deterministic
            # build-identifier; we set it to a runtime-params-CID
            # prefix so the gateway response is fully content-
            # addressed and reproducible.
            system_fingerprint = (
                "fp_" + str(self.runtime_params.cid())[:12])
            body_out = {
                "id": str(resp.id),
                "object": str(resp.object),
                "created": int(resp.created),
                "model": str(resp.model),
                "system_fingerprint": system_fingerprint,
                "choices": [
                    {
                        "index": int(c.index),
                        "message": {
                            "role": str(c.message.role),
                            "content": str(c.message.content),
                        },
                        "finish_reason": str(c.finish_reason),
                        # OpenAI shape: logprobs is either null
                        # (when not requested) or a structured
                        # object. V1 returns null on the
                        # chat-completions path; raw logits are
                        # available via /v1/substrate/forward.
                        "logprobs": None,
                        # Substrate side: raw final-position
                        # logits as a research extension. Hosted
                        # APIs do NOT carry this field.
                        "x_coordpy_raw_logits_head": list(
                            c.logprobs),
                    }
                    for c in resp.choices
                ],
                "usage": dict(resp.usage),
                "streaming_supported": False,
                "stream_requested": stream_requested,
                "facade_response_cid": str(resp.cid()),
            }
            if resp.substrate_side_channel is not None:
                sc = resp.substrate_side_channel
                ids = tokenize_bytes_v79(prompt, max_len=32)
                split = self._normalize_split(
                    token_ids=ids, requested_split=None)
                self._register_replay_artifact(
                    source_path=W81_GATEWAY_PATH_CHAT_COMPLETIONS,
                    prompt=prompt,
                    token_ids=ids,
                    split_at=split,
                    forward_trace_cid=str(
                        sc.forward_trace_cid),
                    kv_cache_cid=str(sc.kv_cache_cid))
                body_out["substrate_side_channel"] = {
                    "schema": str(sc.schema),
                    "runtime_params_cid": str(
                        sc.runtime_params_cid),
                    "forward_trace_cid": str(
                        sc.forward_trace_cid),
                    "kv_cache_cid": str(sc.kv_cache_cid),
                    "final_hidden_cid": str(
                        sc.final_hidden_cid),
                    "per_layer_attn_probs_cids": list(
                        sc.per_layer_attn_probs_cids),
                    "per_layer_hidden_state_cids": list(
                        sc.per_layer_hidden_state_cids),
                    "substrate_axes_cid": str(
                        sc.substrate_axes_cid),
                }
            return self._respond(
                path=W81_GATEWAY_PATH_CHAT_COMPLETIONS,
                status=200, body=body_out,
                request_cid=request_cid)
        except (KeyError, ValueError, TypeError) as exc:
            return self._respond(
                path=W81_GATEWAY_PATH_CHAT_COMPLETIONS,
                status=400,
                body={
                    "error": {
                        "type": "bad_request",
                        "message": str(exc),
                    }},
                request_cid=request_cid)

    # ------------------------------------------------------------
    # /v1/completions
    # ------------------------------------------------------------

    def _handle_completions(
            self, *, body: dict[str, Any],
            request_cid: str,
    ) -> GatewayResponseV1:
        try:
            prompt = str(body.get("prompt", ""))
            ids = tokenize_bytes_v79(prompt, max_len=32)
            trace, kv = forward_controlled_runtime(
                params=self.runtime_params,
                input_token_ids=ids,
                hidden_state_injections_per_layer=None,
                attention_bias_injections_per_layer=None)
            split = self._normalize_split(
                token_ids=ids, requested_split=None)
            self._register_replay_artifact(
                source_path=W81_GATEWAY_PATH_COMPLETIONS,
                prompt=prompt,
                token_ids=ids,
                split_at=split,
                forward_trace_cid=str(trace.cid()),
                kv_cache_cid=str(kv.cid()))
            if trace.seq_len > 0:
                logits = _np.asarray(
                    trace.logits, dtype=_np.float64)
                tok = int(_np.argmax(logits[-1]) % 256)
                text_out = f"<{tok:03d}>"
            else:
                text_out = ""
            return self._respond(
                path=W81_GATEWAY_PATH_COMPLETIONS,
                status=200,
                body={
                    "object": "text.completion",
                    "model": str(self.config.model_name),
                    "choices": [{
                        "index": 0,
                        "text": text_out,
                        "finish_reason": "length",
                    }],
                    "usage": {
                        "prompt_tokens": int(trace.seq_len),
                        "completion_tokens": 1,
                        "total_tokens": int(
                            trace.seq_len + 1),
                    },
                    "forward_trace_cid": str(trace.cid()),
                    "kv_cache_cid": str(kv.cid()),
                },
                request_cid=request_cid)
        except (KeyError, ValueError, TypeError) as exc:
            return self._respond(
                path=W81_GATEWAY_PATH_COMPLETIONS,
                status=400,
                body={
                    "error": {
                        "type": "bad_request",
                        "message": str(exc),
                    }},
                request_cid=request_cid)

    # ------------------------------------------------------------
    # /v1/substrate/forward — explicit substrate read endpoint.
    # ------------------------------------------------------------

    def _handle_substrate_forward(
            self, *, body: dict[str, Any],
            request_cid: str,
    ) -> GatewayResponseV1:
        try:
            prompt = str(body.get("prompt", ""))
            ids = tokenize_bytes_v79(prompt, max_len=32)
            trace, kv = forward_controlled_runtime(
                params=self.runtime_params,
                input_token_ids=ids,
                hidden_state_injections_per_layer=None,
                attention_bias_injections_per_layer=None)
            split = self._normalize_split(
                token_ids=ids, requested_split=None)
            self._register_replay_artifact(
                source_path=W81_GATEWAY_PATH_SUBSTRATE_FORWARD,
                prompt=prompt,
                token_ids=ids,
                split_at=split,
                forward_trace_cid=str(trace.cid()),
                kv_cache_cid=str(kv.cid()))
            return self._respond(
                path=W81_GATEWAY_PATH_SUBSTRATE_FORWARD,
                status=200,
                body={
                    "schema": str(W81_GATEWAY_V1_SCHEMA_VERSION),
                    "runtime_params_cid": str(
                        self.runtime_params.cid()),
                    "forward_trace_cid": str(trace.cid()),
                    "kv_cache_cid": str(kv.cid()),
                    "seq_len": int(trace.seq_len),
                    "final_hidden_cid": _ndarray_cid(
                        trace.final_hidden),
                    "per_layer_attn_probs_cids": [
                        _ndarray_cid(a)
                        for a in trace.attn_probs],
                    "per_layer_hidden_state_cids": [
                        _ndarray_cid(h)
                        for h in trace.post_mlp_hidden],
                    "substrate_axes_cid": str(
                        W79_CONTROLLED_RUNTIME_AXES.cid()),
                    "substrate_axes": list(
                        W79_CONTROLLED_RUNTIME_AXES.to_dict().keys()),
                },
                request_cid=request_cid)
        except (KeyError, ValueError, TypeError) as exc:
            return self._respond(
                path=W81_GATEWAY_PATH_SUBSTRATE_FORWARD,
                status=400,
                body={
                    "error": {
                        "type": "bad_request",
                        "message": str(exc),
                    }},
                request_cid=request_cid)

    # ------------------------------------------------------------
    # /v1/substrate/replay — deterministic replay-from-KV hook.
    # ------------------------------------------------------------

    def _handle_substrate_replay(
            self, *, body: dict[str, Any],
            request_cid: str,
    ) -> GatewayResponseV1:
        try:
            requested_forward_trace_cid = body.get(
                "forward_trace_cid")
            requested_kv_cache_cid = body.get("kv_cache_cid")
            artifact = self._resolve_replay_artifact(
                forward_trace_cid=(
                    str(requested_forward_trace_cid)
                    if requested_forward_trace_cid is not None
                    else None),
                kv_cache_cid=(
                    str(requested_kv_cache_cid)
                    if requested_kv_cache_cid is not None
                    else None))
            replay_source = "prompt"
            registered_forward_trace_cid = None
            registered_kv_cache_cid = None
            if artifact is not None:
                prompt = str(artifact["prompt"])
                ids = [int(t) for t in artifact["token_ids"]]
                split = int(artifact["split_at"])
                replay_source = "registered_trace"
                registered_forward_trace_cid = str(
                    artifact["forward_trace_cid"])
                registered_kv_cache_cid = str(
                    artifact["kv_cache_cid"])
            else:
                prompt = str(body.get("prompt", ""))
                ids = tokenize_bytes_v79(prompt, max_len=32)
                split = self._normalize_split(
                    token_ids=ids,
                    requested_split=body.get("split_at", 1))
            prefix = ids[:split]
            suffix = ids[split:]
            trace_a, kv_a = forward_controlled_runtime(
                params=self.runtime_params,
                input_token_ids=prefix,
                hidden_state_injections_per_layer=None,
                attention_bias_injections_per_layer=None)
            trace_b, kv_b = replay_from_kv_cache(
                params=self.runtime_params,
                kv_cache=kv_a,
                new_token_ids=suffix)
            trace_full, kv_full = forward_controlled_runtime(
                params=self.runtime_params,
                input_token_ids=ids,
                hidden_state_injections_per_layer=None,
                attention_bias_injections_per_layer=None)
            self._register_replay_artifact(
                source_path=W81_GATEWAY_PATH_SUBSTRATE_REPLAY,
                prompt=prompt,
                token_ids=ids,
                split_at=split,
                forward_trace_cid=str(trace_full.cid()),
                kv_cache_cid=str(kv_full.cid()))
            replay_logits = _np.asarray(
                trace_b.logits, dtype=_np.float64)
            full_logits = _np.asarray(
                trace_full.logits, dtype=_np.float64)
            if (replay_logits.size > 0
                    and full_logits.size > 0):
                diff = float(_np.max(_np.abs(
                    replay_logits[-1] - full_logits[-1])))
            else:
                diff = 0.0
            return self._respond(
                path=W81_GATEWAY_PATH_SUBSTRATE_REPLAY,
                status=200,
                body={
                    "schema": str(W81_GATEWAY_V1_SCHEMA_VERSION),
                    "replay_source": str(replay_source),
                    "requested_forward_trace_cid": (
                        str(requested_forward_trace_cid)
                        if requested_forward_trace_cid
                        is not None else None),
                    "requested_kv_cache_cid": (
                        str(requested_kv_cache_cid)
                        if requested_kv_cache_cid
                        is not None else None),
                    "registered_forward_trace_cid": (
                        str(registered_forward_trace_cid)
                        if registered_forward_trace_cid
                        is not None else None),
                    "registered_kv_cache_cid": (
                        str(registered_kv_cache_cid)
                        if registered_kv_cache_cid
                        is not None else None),
                    "split_at": int(split),
                    "prefix_kv_cache_cid": str(kv_a.cid()),
                    "replay_trace_cid": str(trace_b.cid()),
                    "recompute_trace_cid": str(trace_full.cid()),
                    "recompute_kv_cache_cid": str(kv_full.cid()),
                    "registered_forward_trace_match": (
                        bool(
                            registered_forward_trace_cid
                            == str(trace_full.cid()))
                        if registered_forward_trace_cid
                        is not None else None),
                    "registered_kv_cache_match": (
                        bool(
                            registered_kv_cache_cid
                            == str(kv_full.cid()))
                        if registered_kv_cache_cid
                        is not None else None),
                    "final_logit_max_abs_diff": float(diff),
                    "byte_identical_within_1e_minus_8": bool(
                        diff < 1e-8),
                },
                request_cid=request_cid)
        except (KeyError, ValueError, TypeError) as exc:
            return self._respond(
                path=W81_GATEWAY_PATH_SUBSTRATE_REPLAY,
                status=400,
                body={
                    "error": {
                        "type": "bad_request",
                        "message": str(exc),
                    }},
                request_cid=request_cid)

    # ------------------------------------------------------------
    # /v1/substrate/conformance — W80 contract conformance.
    # ------------------------------------------------------------

    def _handle_substrate_conformance(
            self, *, body: dict[str, Any],
            request_cid: str,
    ) -> GatewayResponseV1:
        adapter = ControlledRuntimeInstrumentationAdapterV1(
            runtime_params=self.runtime_params)
        report = run_instrumentation_conformance_v1(adapter)
        return self._respond(
            path=W81_GATEWAY_PATH_SUBSTRATE_CONFORMANCE,
            status=200,
            body={
                "schema": str(W81_GATEWAY_V1_SCHEMA_VERSION),
                "backend_id": str(report.backend_id),
                "backend_runtime_id": str(
                    report.backend_runtime_id),
                "n_pass": int(report.n_pass),
                "n_fail": int(report.n_fail),
                "n_skip": int(report.n_skip),
                "report_cid": str(report.cid()),
                "per_axis": [
                    {"axis": str(a), "status": str(s)}
                    for a, s in report.per_axis
                ],
            },
            request_cid=request_cid)

    # ------------------------------------------------------------
    # /v1/substrate/capabilities — declared axes + schema info.
    # ------------------------------------------------------------

    def _handle_substrate_capabilities(
            self, *, body: dict[str, Any],
            request_cid: str,
    ) -> GatewayResponseV1:
        return self._respond(
            path=W81_GATEWAY_PATH_SUBSTRATE_CAPABILITIES,
            status=200,
            body={
                "schema": str(W81_GATEWAY_V1_SCHEMA_VERSION),
                "model": str(self.config.model_name),
                "substrate_axes": list(
                    W79_CONTROLLED_RUNTIME_AXES.to_dict().keys()),
                "substrate_axes_cid": str(
                    W79_CONTROLLED_RUNTIME_AXES.cid()),
                "side_channel_enabled": bool(
                    self.config.allow_substrate_side_channel),
                "auth_required": bool(
                    self.config.bearer_token is not None),
                "supported_endpoints": list(
                    W81_GATEWAY_ROUTES),
                "honest_caveats": [
                    "gateway does NOT pierce hosted-model "
                    "substrate; substrate access is real "
                    "because the runtime underneath is "
                    "controlled in-process",
                    "v1 routes to controlled_runtime_substrate_v1 "
                    "(not a frontier-scale model)",
                ],
            },
            request_cid=request_cid)

    # ------------------------------------------------------------
    # HTTP server binding (optional).
    # ------------------------------------------------------------

    def serve_forever(
            self, *, host: str = "127.0.0.1",
            port: int = 0,
    ) -> "GatewayHTTPServer":
        """Bind a real TCP listener that delegates to dispatch.

        ``port=0`` lets the OS pick a free port (useful for
        tests). Returns a ``GatewayHTTPServer`` whose
        ``actual_port`` attribute carries the bound port.
        """
        srv = GatewayHTTPServer(
            gateway=self, host=str(host), port=int(port))
        srv.start()
        return srv


class _GatewayHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that forwards to gateway.dispatch."""

    gateway: DeployableSubstrateGatewayV1

    def log_message(  # noqa: D401 — match BaseHTTPRequestHandler
            self, *_args, **_kwargs) -> None:  # type: ignore
        return  # silence default access log under tests

    def do_POST(self) -> None:  # noqa: N802 — stdlib name
        self._do(method="POST")

    def do_GET(self) -> None:  # noqa: N802 — stdlib name
        self._do(method="GET")

    def _do(self, *, method: str) -> None:
        gateway = type(self).gateway
        path = self.path or "/"
        length = int(self.headers.get("Content-Length") or 0)
        raw = (self.rfile.read(length) if length > 0
               else b"")
        body: dict[str, Any] | None
        if raw:
            try:
                body = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                body = None
        else:
            body = None
        auth = self.headers.get("Authorization")
        resp = gateway.dispatch(
            path=path, body=body, auth_header=auth)
        payload = json.dumps(
            resp.body, sort_keys=True,
            separators=(",", ":")).encode("utf-8")
        self.send_response(int(resp.status))
        self.send_header(
            "Content-Type", "application/json; charset=utf-8")
        self.send_header(
            "X-CoordPy-Gateway-Cid", str(resp.cid()))
        self.send_header(
            "X-CoordPy-Request-Cid", str(resp.request_cid))
        self.send_header(
            "X-CoordPy-Response-Cid", str(resp.response_cid))
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class GatewayHTTPServer:
    """Threaded HTTP server wrapper.

    Use ``stop()`` to shut down cleanly. ``actual_port`` is
    only set after ``start()`` has bound the socket.
    """

    def __init__(
            self, *,
            gateway: DeployableSubstrateGatewayV1,
            host: str = "127.0.0.1",
            port: int = 0) -> None:
        self.gateway = gateway
        self.host = str(host)
        self.port = int(port)
        self.actual_port: int = -1
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        # We bind a fresh handler class so type(self).gateway
        # resolves to this gateway instance, even if two
        # gateways are running side by side.
        gateway_ref = self.gateway

        class _BoundHandler(_GatewayHandler):
            gateway = gateway_ref

        self._server = HTTPServer(
            (self.host, self.port), _BoundHandler)
        self.actual_port = int(
            self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="coordpy-w81-gateway-v1")
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


@dataclasses.dataclass(frozen=True)
class GatewayEndToEndBenchReportV1:
    """End-to-end bench report: gateway routes a chat-completion.

    Carries the gateway CID, the chat-completion response CID,
    and the substrate-forward CID — so the bench is content-
    addressed end to end.
    """

    schema: str
    gateway_cid: str
    chat_response_cid: str
    substrate_forward_cid: str
    replay_response_cid: str
    conformance_n_pass: int
    conformance_n_fail: int
    routed_endpoints: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "gateway_cid": str(self.gateway_cid),
            "chat_response_cid": str(self.chat_response_cid),
            "substrate_forward_cid": str(
                self.substrate_forward_cid),
            "replay_response_cid": str(
                self.replay_response_cid),
            "conformance_n_pass": int(self.conformance_n_pass),
            "conformance_n_fail": int(self.conformance_n_fail),
            "routed_endpoints": list(self.routed_endpoints),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w81_gateway_end_to_end_bench_v1",
            "report": self.to_dict()})


def run_gateway_end_to_end_bench_v1(
        *,
        gateway: DeployableSubstrateGatewayV1 | None = None,
) -> GatewayEndToEndBenchReportV1:
    """End-to-end bench routing through the gateway.

    Drives chat completion + substrate forward + substrate
    replay + substrate conformance in one shot, producing a
    content-addressed report. This is the load-bearing
    "routes through the gateway end to end" evidence for the
    P1 #7 definition of done.
    """
    gw = gateway or DeployableSubstrateGatewayV1()
    chat = gw.dispatch(
        path=W81_GATEWAY_PATH_CHAT_COMPLETIONS,
        body={
            "model": gw.config.model_name,
            "messages": [
                {"role": "system",
                 "content": "you are a substrate-aware agent"},
                {"role": "user",
                 "content": "what is the substrate trace cid?"},
            ],
            "max_tokens": 4,
            "substrate_options": {
                "return_side_channel": True,
            },
        })
    fwd = gw.dispatch(
        path=W81_GATEWAY_PATH_SUBSTRATE_FORWARD,
        body={"prompt": "the gateway end-to-end bench prompt"})
    rep = gw.dispatch(
        path=W81_GATEWAY_PATH_SUBSTRATE_REPLAY,
        body={
            "prompt": "the gateway end-to-end bench prompt",
            "split_at": 3,
        })
    conf = gw.dispatch(
        path=W81_GATEWAY_PATH_SUBSTRATE_CONFORMANCE,
        body={})
    return GatewayEndToEndBenchReportV1(
        schema=W81_GATEWAY_V1_SCHEMA_VERSION,
        gateway_cid=str(gw.cid()),
        chat_response_cid=str(chat.response_cid),
        substrate_forward_cid=str(fwd.response_cid),
        replay_response_cid=str(rep.response_cid),
        conformance_n_pass=int(
            conf.body.get("n_pass", 0)),
        conformance_n_fail=int(
            conf.body.get("n_fail", 0)),
        routed_endpoints=(
            W81_GATEWAY_PATH_CHAT_COMPLETIONS,
            W81_GATEWAY_PATH_SUBSTRATE_FORWARD,
            W81_GATEWAY_PATH_SUBSTRATE_REPLAY,
            W81_GATEWAY_PATH_SUBSTRATE_CONFORMANCE,
        ),
    )


@dataclasses.dataclass(frozen=True)
class GatewayWitnessV1:
    """Detached witness for the gateway state."""

    schema: str
    gateway_cid: str
    config_cid: str
    runtime_params_cid: str
    request_counter: int

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w81_gateway_witness_v1",
            "schema": str(self.schema),
            "gateway_cid": str(self.gateway_cid),
            "config_cid": str(self.config_cid),
            "runtime_params_cid": str(self.runtime_params_cid),
            "request_counter": int(self.request_counter),
        })


def emit_gateway_witness_v1(
        *, gateway: DeployableSubstrateGatewayV1,
) -> GatewayWitnessV1:
    return GatewayWitnessV1(
        schema=W81_GATEWAY_V1_SCHEMA_VERSION,
        gateway_cid=str(gateway.cid()),
        config_cid=str(gateway.config.cid()),
        runtime_params_cid=str(gateway.runtime_params.cid()),
        request_counter=int(gateway.request_counter),
    )
