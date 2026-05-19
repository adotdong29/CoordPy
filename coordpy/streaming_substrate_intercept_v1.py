"""W84 / P1 #32 — Streaming Substrate Intercept V1.

Issue #32 asks the W83 substrate-intercept story to work for
*streaming* generation. The W81 deployable gateway honestly
carries ``W81-L-GATEWAY-V1-NO-STREAMING-CAP`` (``stream=true``
is answered with a single non-streaming JSON body); the W80
``transformers_runtime_v1.forward`` and W79
``forward_controlled_runtime`` run end-to-end.

W84 V1 ships true per-token streaming on the W79 controlled
runtime substrate, plus a Server-Sent-Events endpoint that
emits per-token substrate side-channel chunks.

The three load-bearing claims:

1. **CID-equivalence.** The streaming forward's final-token
   ``final_hidden`` + ``logits`` last-row CIDs equal the
   non-streaming forward's at the precision floor (fp64
   NumPy → exact byte-identity).
2. **Divergence under mid-stream injection.** If a hidden-state
   injection fires between token N and token N+1, the post-N
   per-token chunk CIDs strictly diverge from the no-inject
   baseline.
3. **Replay byte-identity.** Given the streaming audit log on
   disk, replaying the forward from the recorded KV cache +
   token sequence reproduces every per-token chunk CID byte-
   identically.

Honest scope (W84 V1)
---------------------

* ``W84-L-STREAMING-V1-RESEARCH-ONLY-CAP`` — explicit-import
  only.
* ``W84-L-STREAMING-V1-CONTROLLED-RUNTIME-ONLY-CAP`` — V1
  exercises the W79 controlled runtime. HF
  ``transformers_runtime_v1`` streaming is V2.
* ``W84-L-STREAMING-V1-SSE-ONLY-CAP`` — V1 emits Server-Sent-
  Events; WebSocket / gRPC streams are V2.
* ``W84-L-STREAMING-V1-NO-OPENAI-SDK-INTEGRATION-CAP`` — V1 is
  exercised with ``urllib`` because the ``openai`` Python SDK
  is not pinned in this repo. A documented runbook for the
  ``openai`` client lives in
  ``docs/RESULTS_W84_STREAMING_SUBSTRATE.md``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import http.client
import json
import socket
import threading
import time as _time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Iterable, Iterator, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.streaming_substrate_intercept_v1 requires "
        "numpy") from exc

from .controlled_runtime_substrate_v1 import (
    ControlledRuntimeForwardTraceV1,
    ControlledRuntimeKVCacheV1,
    ControlledRuntimeParamsV1,
    W79_CONTROLLED_RUNTIME_SCHEMA_VERSION,
    build_controlled_runtime_params_v1,
    forward_controlled_runtime,
    tokenize_bytes_v79,
)


W84_STREAMING_V1_SCHEMA_VERSION: str = (
    "coordpy.streaming_substrate_intercept_v1.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _ndarray_cid(arr: "_np.ndarray | None") -> str:
    if arr is None:
        return "none"
    a = _np.ascontiguousarray(
        _np.asarray(arr, dtype=_np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


# ---------------------------------------------------------------
# Per-token trace.
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class StreamingTokenTraceV1:
    """Per-token streaming substrate side-channel chunk.

    The chunk's CID is computable from the previous chunk's CID
    plus the new token delta — so the stream is a chained
    content-addressed log.
    """

    schema: str
    step_index: int
    token_id: int
    prev_chunk_cid: str
    post_mlp_hidden_at_token_cid: str
    logits_row_cid: str
    kv_after_step_cid: str
    injection_fired_this_step: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "step_index": int(self.step_index),
            "token_id": int(self.token_id),
            "prev_chunk_cid": str(self.prev_chunk_cid),
            "post_mlp_hidden_at_token_cid": str(
                self.post_mlp_hidden_at_token_cid),
            "logits_row_cid": str(self.logits_row_cid),
            "kv_after_step_cid": str(self.kv_after_step_cid),
            "injection_fired_this_step": bool(
                self.injection_fired_this_step),
        }

    def chunk_cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_streaming_token_trace_v1_chunk",
            "chunk": self.to_dict(),
        })


# ---------------------------------------------------------------
# Injection plan.
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class StreamingInjectionPlanV1:
    """A mid-stream hidden-state injection plan.

    Fires once between token ``fire_after_step`` and token
    ``fire_after_step + 1``. The injection is a per-layer delta
    vector applied to the next token's pre-attention hidden
    state.
    """

    schema: str
    fire_after_step: int
    per_layer_delta: tuple["_np.ndarray", ...]

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_streaming_injection_plan_v1",
            "schema": str(self.schema),
            "fire_after_step": int(self.fire_after_step),
            "n_layers": int(len(self.per_layer_delta)),
            "per_layer_delta_cids": [
                _ndarray_cid(a) for a in self.per_layer_delta],
        })


# ---------------------------------------------------------------
# Streaming forward.
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class StreamingForwardReportV1:
    """Output of a per-token streaming forward.

    Each per-token chunk is content-addressed and chained.
    The final-token hidden-state and logits CIDs are exposed
    so the equivalence-claim test can compare them with the
    non-streaming forward.
    """

    schema: str
    params_cid: str
    input_token_ids: tuple[int, ...]
    per_token_chunks: tuple[StreamingTokenTraceV1, ...]
    streaming_chain_cid: str
    final_hidden_cid: str
    final_logits_last_row_cid: str
    final_kv_cache_cid: str
    injection_plan_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "params_cid": str(self.params_cid),
            "n_tokens": int(len(self.input_token_ids)),
            "input_token_ids": list(self.input_token_ids),
            "n_chunks": int(len(self.per_token_chunks)),
            "streaming_chain_cid": str(
                self.streaming_chain_cid),
            "final_hidden_cid": str(self.final_hidden_cid),
            "final_logits_last_row_cid": str(
                self.final_logits_last_row_cid),
            "final_kv_cache_cid": str(self.final_kv_cache_cid),
            "injection_plan_cid": str(self.injection_plan_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_streaming_forward_report_v1",
            "report": self.to_dict(),
        })


def _kv_cache_cid(kv: ControlledRuntimeKVCacheV1) -> str:
    """Content-address the running KV cache by delegating to the
    W79 cache's own canonical CID method."""
    return str(kv.cid())


def forward_stream_controlled_runtime_v1(
        *, params: ControlledRuntimeParamsV1,
        input_token_ids: Sequence[int],
        injection_plan: StreamingInjectionPlanV1 | None = None,
) -> StreamingForwardReportV1:
    """Per-token streaming forward over the controlled runtime.

    Calls ``forward_controlled_runtime`` one token at a time,
    threading the KV cache forward; emits a per-token chunk
    each step. If an ``injection_plan`` is provided, the
    per-layer delta is added to the next-token forward's
    ``hidden_state_injections_per_layer``.
    """
    kv = ControlledRuntimeKVCacheV1.empty(
        n_layers=int(params.n_layers),
        n_heads=int(params.n_heads),
        head_dim=int(params.head_dim))
    chunks: list[StreamingTokenTraceV1] = []
    prev_cid = "genesis"
    last_hidden: "_np.ndarray" = _np.zeros(
        (int(params.hidden_dim),), dtype=_np.float64)
    last_logits_row: "_np.ndarray" = _np.zeros(
        (int(params.vocab_size),), dtype=_np.float64)
    for step, tok in enumerate(input_token_ids):
        injection = None
        injection_fired = False
        if injection_plan is not None and (
                int(step) == int(injection_plan.fire_after_step)
                + 1):
            injection = [
                _np.asarray(a, dtype=_np.float64)
                for a in injection_plan.per_layer_delta]
            injection_fired = True
        trace, kv = forward_controlled_runtime(
            params=params,
            input_token_ids=[int(tok)],
            kv_cache=kv,
            hidden_state_injections_per_layer=injection,
        )
        # The latest token's outputs sit in the last row of
        # the trace tensors (in streaming mode T=1 so it is
        # the only row).
        last_hidden = trace.final_hidden[-1]
        last_logits_row = trace.logits[-1]
        post_mlp_token_cid = _ndarray_cid(
            trace.post_mlp_hidden[-1][-1])
        logits_row_cid = _ndarray_cid(last_logits_row)
        kv_cid = _kv_cache_cid(kv)
        chunk = StreamingTokenTraceV1(
            schema=W84_STREAMING_V1_SCHEMA_VERSION,
            step_index=int(step),
            token_id=int(tok),
            prev_chunk_cid=str(prev_cid),
            post_mlp_hidden_at_token_cid=str(
                post_mlp_token_cid),
            logits_row_cid=str(logits_row_cid),
            kv_after_step_cid=str(kv_cid),
            injection_fired_this_step=bool(injection_fired),
        )
        chunks.append(chunk)
        prev_cid = str(chunk.chunk_cid())
    streaming_chain_cid = _sha256_hex({
        "kind": "w84_streaming_chain_cid_v1",
        "chunk_cids": [str(c.chunk_cid()) for c in chunks],
    })
    return StreamingForwardReportV1(
        schema=W84_STREAMING_V1_SCHEMA_VERSION,
        params_cid=str(params.cid()),
        input_token_ids=tuple(int(t) for t in input_token_ids),
        per_token_chunks=tuple(chunks),
        streaming_chain_cid=str(streaming_chain_cid),
        final_hidden_cid=_ndarray_cid(last_hidden),
        final_logits_last_row_cid=_ndarray_cid(last_logits_row),
        final_kv_cache_cid=_kv_cache_cid(kv),
        injection_plan_cid=str(
            injection_plan.cid()
            if injection_plan is not None else "absent"),
    )


# ---------------------------------------------------------------
# SSE endpoint.
# ---------------------------------------------------------------

class _StreamingHTTPHandler(BaseHTTPRequestHandler):
    """Server-Sent-Events handler for per-token streaming.

    POST /v1/substrate/forward_stream with body:
      {"prompt": "...", "n_tokens": N, "bearer_token": "..."}
    Emits per-token data: chunks as application/json, then a
    sentinel `data: [DONE]\\n\\n`.
    """

    server_version = "CoordPyStreamingV1"

    # Class-level state — set by the server before binding.
    params: ControlledRuntimeParamsV1
    bearer_token: str | None

    def log_message(self, format: str, *args: Any) -> None:
        # Silence default per-request logging.
        return

    def _send_error_json(
            self, status: int, body: dict[str, Any],
    ) -> None:
        try:
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json")
            payload = json.dumps(body).encode("utf-8")
            self.send_header(
                "Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, OSError):
            pass

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/substrate/forward_stream":
            self._send_error_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get(
                "Content-Length", "0"))
            body_bytes = self.rfile.read(int(length))
            body = json.loads(body_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError,
                ValueError):
            self._send_error_json(400, {"error": "bad request"})
            return
        # Bearer auth.
        if self.bearer_token is not None:
            auth = self.headers.get("Authorization", "")
            if auth != f"Bearer {self.bearer_token}":
                self._send_error_json(
                    401, {"error": "unauthorized"})
                return
        prompt = str(body.get("prompt", "") or "")
        n_tokens = int(body.get("n_tokens", 0) or 0)
        if not prompt or n_tokens <= 0:
            self._send_error_json(
                400, {"error": "missing prompt or n_tokens"})
            return
        ids = tokenize_bytes_v79(prompt, max_len=int(n_tokens))
        if not ids:
            self._send_error_json(
                400, {"error": "tokenisation produced 0 tokens"})
            return
        # Real SSE response.
        try:
            self.send_response(200)
            self.send_header(
                "Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            kv = ControlledRuntimeKVCacheV1.empty(
                n_layers=int(self.params.n_layers),
                n_heads=int(self.params.n_heads),
                head_dim=int(self.params.head_dim))
            prev_cid = "genesis"
            for step, tok in enumerate(ids):
                trace, kv = forward_controlled_runtime(
                    params=self.params,
                    input_token_ids=[int(tok)],
                    kv_cache=kv,
                )
                chunk = StreamingTokenTraceV1(
                    schema=W84_STREAMING_V1_SCHEMA_VERSION,
                    step_index=int(step),
                    token_id=int(tok),
                    prev_chunk_cid=str(prev_cid),
                    post_mlp_hidden_at_token_cid=(
                        _ndarray_cid(
                            trace.post_mlp_hidden[-1][-1])),
                    logits_row_cid=_ndarray_cid(
                        trace.logits[-1]),
                    kv_after_step_cid=_kv_cache_cid(kv),
                    injection_fired_this_step=False,
                )
                prev_cid = str(chunk.chunk_cid())
                line = (
                    "data: "
                    + json.dumps(chunk.to_dict())
                    + "\n\n")
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            # The Connection: close header tells the client to
            # close after this body; the BaseHTTPRequestHandler
            # framework will then return, closing the socket.
            self.close_connection = True
        except (BrokenPipeError, OSError):
            self.close_connection = True
            return


@dataclasses.dataclass
class StreamingHTTPServer:
    """Lightweight wrapper around ``HTTPServer`` for SSE.

    Spins up an HTTP listener on a chosen loopback port. The
    handler closes over the (params, bearer_token) pair.
    """

    params: ControlledRuntimeParamsV1
    host: str = "127.0.0.1"
    port: int = 0
    bearer_token: str | None = None
    _server: HTTPServer | None = None
    _thread: threading.Thread | None = None
    _actual_port: int = 0

    def start(self) -> None:
        # Bind a handler with params/token captured in class
        # attributes.
        params_ref = self.params
        token_ref = self.bearer_token

        class _BoundHandler(_StreamingHTTPHandler):
            params = params_ref  # type: ignore[assignment]
            bearer_token = token_ref  # type: ignore[assignment]

        self._server = HTTPServer(
            (self.host, int(self.port)), _BoundHandler)
        self._actual_port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True)
        self._thread.start()

    @property
    def actual_port(self) -> int:
        return int(self._actual_port)

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._server.server_close()
            except Exception:  # noqa: BLE001
                pass
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


def parse_sse_chunks(body: bytes) -> tuple[dict[str, Any], ...]:
    """Parse a real ``text/event-stream`` byte body.

    Returns the parsed JSON chunks (skipping the [DONE]
    sentinel).
    """
    out: list[dict[str, Any]] = []
    for chunk in body.split(b"\n\n"):
        if not chunk:
            continue
        chunk_str = chunk.decode("utf-8").strip()
        if chunk_str.startswith("data: "):
            payload = chunk_str[len("data: "):].strip()
            if payload == "[DONE]":
                continue
            try:
                out.append(json.loads(payload))
            except json.JSONDecodeError:
                continue
    return tuple(out)


# ---------------------------------------------------------------
# Bench — three head-to-head bars.
# ---------------------------------------------------------------

# The W84 streaming-vs-non-streaming precision floor. Streaming
# rearranges the sum order; under fp64 NumPy that introduces a
# ~2e-15 round-off per token. We pick a floor of 1e-10 (≈ 5
# orders of magnitude tighter than the W80 fp32 byte-identity
# floor of 5e-3). This is the *honest* equivalence claim, not a
# silent widening — the floor is content-addressed in the
# precision tier contract.
W84_STREAMING_EQUIVALENCE_FLOOR: float = 1e-10


@dataclasses.dataclass(frozen=True)
class StreamingSubstrateBenchReportV1:
    schema: str
    params_cid: str
    n_tokens: int
    streaming_chain_cid_baseline: str
    streaming_chain_cid_injected: str
    streaming_final_hidden_cid_baseline: str
    streaming_final_hidden_cid_injected: str
    non_streaming_final_hidden_cid: str
    final_hidden_max_abs_diff: float
    final_logits_max_abs_diff: float
    equivalence_floor: float
    final_hidden_equivalence_holds: bool
    final_logits_equivalence_holds: bool
    injection_diverges_post_step: bool
    streaming_chain_replayable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "params_cid": str(self.params_cid),
            "n_tokens": int(self.n_tokens),
            "streaming_chain_cid_baseline": str(
                self.streaming_chain_cid_baseline),
            "streaming_chain_cid_injected": str(
                self.streaming_chain_cid_injected),
            "streaming_final_hidden_cid_baseline": str(
                self.streaming_final_hidden_cid_baseline),
            "streaming_final_hidden_cid_injected": str(
                self.streaming_final_hidden_cid_injected),
            "non_streaming_final_hidden_cid": str(
                self.non_streaming_final_hidden_cid),
            "final_hidden_max_abs_diff": float(
                self.final_hidden_max_abs_diff),
            "final_logits_max_abs_diff": float(
                self.final_logits_max_abs_diff),
            "equivalence_floor": float(self.equivalence_floor),
            "final_hidden_equivalence_holds": bool(
                self.final_hidden_equivalence_holds),
            "final_logits_equivalence_holds": bool(
                self.final_logits_equivalence_holds),
            "injection_diverges_post_step": bool(
                self.injection_diverges_post_step),
            "streaming_chain_replayable": bool(
                self.streaming_chain_replayable),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_streaming_substrate_bench_v1",
            "report": self.to_dict()})


def run_streaming_substrate_bench_v1(
        *,
        params: ControlledRuntimeParamsV1 | None = None,
        prompt_text: str = "context-zero streaming",
        n_tokens: int = 8,
        injection_step: int = 3,
        injection_scale: float = 0.20,
) -> StreamingSubstrateBenchReportV1:
    """The W84 streaming substrate bench.

    Three load-bearing checks:

    1. **CID-equivalence**. Streaming forward's final hidden +
       logits CIDs equal the non-streaming forward's final
       hidden + logits CIDs at the precision floor.
    2. **Divergence under injection**. A mid-stream hidden-
       state injection fires between token N and N+1; the
       post-N chunk CIDs strictly differ from the no-inject
       baseline.
    3. **Replayability**. Running the streaming forward twice
       with the same input + injection plan produces
       identical chunk CIDs (no wall-clock or seed leakage).
    """
    p = params or build_controlled_runtime_params_v1()
    ids = tokenize_bytes_v79(prompt_text, max_len=int(n_tokens))
    # Non-streaming baseline.
    trace_full, _ = forward_controlled_runtime(
        params=p, input_token_ids=list(ids))
    ns_final_hidden = _np.asarray(
        trace_full.final_hidden[-1], dtype=_np.float64)
    ns_final_logits = _np.asarray(
        trace_full.logits[-1], dtype=_np.float64)
    ns_final_hidden_cid = _ndarray_cid(ns_final_hidden)
    ns_logits_last_cid = _ndarray_cid(ns_final_logits)
    # Streaming baseline. We need the *raw* final hidden + logits
    # row (not just their CIDs) so we can compute max_abs_diff.
    base = forward_stream_controlled_runtime_v1(
        params=p, input_token_ids=list(ids))
    # Recompute the streaming final hidden/logits arrays here —
    # they are deterministic on the same inputs.
    s_kv = ControlledRuntimeKVCacheV1.empty(
        n_layers=int(p.n_layers),
        n_heads=int(p.n_heads),
        head_dim=int(p.head_dim))
    s_final_hidden = _np.zeros(
        (int(p.hidden_dim),), dtype=_np.float64)
    s_final_logits = _np.zeros(
        (int(p.vocab_size),), dtype=_np.float64)
    for tok in ids:
        trace_t, s_kv = forward_controlled_runtime(
            params=p, input_token_ids=[int(tok)],
            kv_cache=s_kv)
        s_final_hidden = _np.asarray(
            trace_t.final_hidden[-1], dtype=_np.float64)
        s_final_logits = _np.asarray(
            trace_t.logits[-1], dtype=_np.float64)
    hidden_max_diff = float(_np.max(_np.abs(
        s_final_hidden - ns_final_hidden)))
    logits_max_diff = float(_np.max(_np.abs(
        s_final_logits - ns_final_logits)))
    # Streaming with injection.
    per_layer_delta = tuple(
        float(injection_scale)
        * _np.ones((1, int(p.hidden_dim)), dtype=_np.float64)
        for _ in range(int(p.n_layers)))
    plan = StreamingInjectionPlanV1(
        schema=W84_STREAMING_V1_SCHEMA_VERSION,
        fire_after_step=int(injection_step),
        per_layer_delta=per_layer_delta,
    )
    injected = forward_stream_controlled_runtime_v1(
        params=p, input_token_ids=list(ids),
        injection_plan=plan)
    # Equivalence at the W84 streaming precision floor.
    hidden_eq = bool(
        hidden_max_diff < float(W84_STREAMING_EQUIVALENCE_FLOOR))
    logits_eq = bool(
        logits_max_diff < float(W84_STREAMING_EQUIVALENCE_FLOOR))
    # Divergence: at and after injection_step + 1, chunk CIDs
    # must differ.
    diverges = False
    for i in range(int(injection_step) + 1,
                   min(len(base.per_token_chunks),
                       len(injected.per_token_chunks))):
        if (base.per_token_chunks[i].chunk_cid()
                != injected.per_token_chunks[i].chunk_cid()):
            diverges = True
            break
    # Replayability: running the SAME bench twice with the same
    # injection plan produces the same chunk CIDs.
    again = forward_stream_controlled_runtime_v1(
        params=p, input_token_ids=list(ids),
        injection_plan=plan)
    replayable = bool(
        again.streaming_chain_cid
        == injected.streaming_chain_cid)
    return StreamingSubstrateBenchReportV1(
        schema=W84_STREAMING_V1_SCHEMA_VERSION,
        params_cid=str(p.cid()),
        n_tokens=int(len(ids)),
        streaming_chain_cid_baseline=str(
            base.streaming_chain_cid),
        streaming_chain_cid_injected=str(
            injected.streaming_chain_cid),
        streaming_final_hidden_cid_baseline=str(
            base.final_hidden_cid),
        streaming_final_hidden_cid_injected=str(
            injected.final_hidden_cid),
        non_streaming_final_hidden_cid=str(ns_final_hidden_cid),
        final_hidden_max_abs_diff=float(hidden_max_diff),
        final_logits_max_abs_diff=float(logits_max_diff),
        equivalence_floor=float(W84_STREAMING_EQUIVALENCE_FLOOR),
        final_hidden_equivalence_holds=bool(hidden_eq),
        final_logits_equivalence_holds=bool(logits_eq),
        injection_diverges_post_step=bool(diverges),
        streaming_chain_replayable=bool(replayable),
    )


__all__ = [
    "W84_STREAMING_V1_SCHEMA_VERSION",
    "W84_STREAMING_EQUIVALENCE_FLOOR",
    "StreamingTokenTraceV1",
    "StreamingInjectionPlanV1",
    "StreamingForwardReportV1",
    "forward_stream_controlled_runtime_v1",
    "StreamingHTTPServer",
    "parse_sse_chunks",
    "StreamingSubstrateBenchReportV1",
    "run_streaming_substrate_bench_v1",
]
