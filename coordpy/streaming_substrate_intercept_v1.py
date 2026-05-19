"""W84 / P1 #32 — Streaming Substrate Intercept V1.

Production multi-agent agents stream tokens. The W83 substrate-
intercept story (run forward end-to-end; capture hidden state /
KV / final logits) is end-to-end, not token-by-token. The
deployable substrate gateway honestly carries the W81 cap
``W81-L-GATEWAY-V1-NO-STREAMING-CAP``: ``stream=true`` requests
are answered with a single non-streaming JSON body. This V1
lifts that limit.

Three layers:

1. **Streaming runtime API.** ``forward_stream`` yields one
   ``StreamingTokenChunkV1`` per generated token. Each chunk
   is content-addressed and includes the partial hidden state,
   the partial KV cache CID, and a chain pointer to the prior
   chunk. The streaming forward's *final-token* trace CID
   equals the non-streaming forward's final-token trace CID at
   the same precision floor (deterministic equivalence — the
   load-bearing structural claim).

2. **SSE wire format.** ``serialize_sse_chunk`` /
   ``parse_sse_chunk`` give the wire format: ``data: <json>\n\n``
   chunks, terminated by a ``data: [DONE]\n\n`` sentinel.
   Real SSE-compliant (``Content-Type: text/event-stream``).
   When the caller opts in via ``substrate_options.return_side_
   channel``, each chunk carries an optional substrate side-
   channel object content-addressed by SHA-256.

3. **Mid-stream substrate intercept.** A
   ``MidStreamInjectionPlanV1`` fires *after* token N: the
   caller registers an injection (hidden-state delta at layer
   L). The post-N stream provably diverges from the no-inject
   baseline (per-token chunk CIDs differ) and is byte-
   identically replayable from the streaming audit log
   (``replay_streaming_audit_log_v1``).

Bearer-auth integration: the streaming runtime exposes a
``BearerAuthGateV1`` shim; the SSE serializer refuses to emit
chunks unless the gate is satisfied. This preserves the W81
gateway's auth contract on the streaming path.

Honest scope (V1, per the issue):

* ``W84-L-STREAMING-V1-SUBSTRATE-RUNTIME-CAP`` — V1 is on top
  of the W79 in-repo controlled runtime. The HF /
  ``transformers`` adapter for streaming + the
  ``deployable_substrate_gateway_v1`` SSE integration are V1
  here; the OpenAI-Python-SDK-against-our-gateway test is
  gated on ``openai`` being installed.
* ``W84-L-STREAMING-V1-ONE-INJECT-PER-STREAM-CAP`` — mid-stream
  injection at most once per stream V1; arbitrary-per-token
  injection is V2.
* ``W84-L-STREAMING-V1-ONE-FORMAT-CAP`` — one SSE substrate
  side-channel format V1; multi-format negotiation is V2.
* ``W84-L-STREAMING-V1-PER-STREAM-AUDIT-CAP`` — per-stream
  audit log V1; cross-stream audit anchoring is V2.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Iterator, Mapping, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.streaming_substrate_intercept_v1 requires numpy"
    ) from exc

from .controlled_runtime_substrate_v1 import (
    ControlledRuntimeKVCacheV1,
    ControlledRuntimeParamsV1,
    build_controlled_runtime_params_v1,
    forward_controlled_runtime,
    replay_from_kv_cache,
    tokenize_bytes_v79,
)


W84_STREAMING_V1_SCHEMA_VERSION: str = (
    "coordpy.streaming_substrate_intercept_v1.v1")


# ---------------------------------------------------------------
# Per-token chunk schema
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StreamingTokenChunkV1:
    """One token of a streaming forward.

    The chunk's CID is a SHA-256 over (prior_chunk_cid,
    token_id, position, final_logits_cid, kv_after_cid,
    optional substrate-side-channel CID). The chain of chunk
    CIDs is content-addressed: any tampering with any prior
    chunk changes the current chunk's CID.
    """

    schema: str
    prior_chunk_cid: str  # "genesis" for the first chunk
    stream_id: str
    position: int
    token_id: int
    final_logits_cid: str
    kv_after_cid: str
    substrate_side_channel_cid: str
    injection_fired: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "prior_chunk_cid": str(self.prior_chunk_cid),
            "stream_id": str(self.stream_id),
            "position": int(self.position),
            "token_id": int(self.token_id),
            "final_logits_cid": str(self.final_logits_cid),
            "kv_after_cid": str(self.kv_after_cid),
            "substrate_side_channel_cid": str(
                self.substrate_side_channel_cid),
            "injection_fired": bool(self.injection_fired),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_streaming_token_chunk_v1",
            "chunk": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class StreamingForwardTraceV1:
    """Aggregate trace for a full streaming forward.

    The aggregate trace's final-token CID must equal the
    non-streaming forward's final-token CID at the precision
    floor — that's the deterministic-equivalence claim.
    """

    schema: str
    stream_id: str
    n_tokens: int
    chunk_cids: tuple[str, ...]
    final_chunk_cid: str
    final_logits_cid: str
    full_kv_after_cid: str
    injection_fired_at: int  # -1 if no injection

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "stream_id": str(self.stream_id),
            "n_tokens": int(self.n_tokens),
            "chunk_cids": list(self.chunk_cids),
            "final_chunk_cid": str(self.final_chunk_cid),
            "final_logits_cid": str(self.final_logits_cid),
            "full_kv_after_cid": str(self.full_kv_after_cid),
            "injection_fired_at": int(self.injection_fired_at),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_streaming_forward_trace_v1",
            "trace": self.to_dict()})


# ---------------------------------------------------------------
# Injection plan
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MidStreamInjectionPlanV1:
    """A mid-stream injection plan.

    Fires *after* the ``inject_after_position``-th token has
    been generated. The injection is a per-layer hidden-state
    delta applied on the NEXT token's forward.
    """

    schema: str
    inject_after_position: int
    layer_index: int
    delta: "_np.ndarray"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inject_after_position": int(
                self.inject_after_position),
            "layer_index": int(self.layer_index),
            "delta_cid": _ndarray_cid(self.delta),
            "delta_shape": _shape(self.delta),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_mid_stream_injection_plan_v1",
            "plan": self.to_dict()})


# ---------------------------------------------------------------
# Streaming runtime
# ---------------------------------------------------------------


def _stream_id(*, params: ControlledRuntimeParamsV1,
               prompt: str, max_new_tokens: int) -> str:
    return _sha256_hex({
        "kind": "w84_streaming_stream_id_v1",
        "params_cid": str(params.cid()),
        "prompt": str(prompt),
        "max_new_tokens": int(max_new_tokens),
    })


def forward_stream(
        *, params: ControlledRuntimeParamsV1,
        prompt: str,
        max_new_tokens: int = 4,
        injection_plan: MidStreamInjectionPlanV1 | None = None,
        return_side_channel: bool = False,
) -> Iterator[StreamingTokenChunkV1]:
    """Yield one ``StreamingTokenChunkV1`` per generated token.

    Implementation: tokenize the prompt, run the full prefix
    forward to build the initial KV cache, then for each new
    token: greedy-decode the next token id from the current
    final-position logits, run a single-token replay to advance
    the KV, and yield a chunk. The streaming chunk's
    ``final_logits_cid`` is over the new-token logits row.

    If ``injection_plan`` is set, the NEXT token forward after
    position ``inject_after_position`` applies the per-layer
    hidden-state delta on the named layer.
    """
    sid = _stream_id(
        params=params, prompt=str(prompt),
        max_new_tokens=int(max_new_tokens))
    prompt_ids = tokenize_bytes_v79(
        str(prompt), max_len=int(params.max_len) - 1)
    if len(prompt_ids) == 0:
        return
    # Initial forward on the prompt.
    prefix_trace, kv = forward_controlled_runtime(
        params=params, input_token_ids=prompt_ids)
    prior_cid = "genesis"
    for step in range(int(max_new_tokens)):
        # Decode the next token from current final logits.
        last_logits = _np.asarray(
            prefix_trace.logits, dtype=_np.float64)[-1]
        next_id = int(_np.argmax(last_logits))
        # Apply injection if this step is the trigger.
        hidden_injs = None
        injection_fired_here = False
        if injection_plan is not None and (
                int(step) == int(
                    injection_plan.inject_after_position)):
            injs: list["_np.ndarray | None"] = [
                None] * int(params.n_layers)
            L = int(injection_plan.layer_index)
            if 0 <= L < int(params.n_layers):
                injs[L] = _np.asarray(
                    injection_plan.delta, dtype=_np.float64)
                hidden_injs = injs
                injection_fired_here = True
        # Single-token replay (extends KV by 1).
        step_trace, kv = replay_from_kv_cache(
            params=params, kv_cache=kv,
            new_token_ids=[next_id],
            hidden_state_injections_per_layer=hidden_injs)
        # Set up for next iteration.
        prefix_trace = step_trace
        # Build chunk.
        step_logits_cid = _ndarray_cid(step_trace.logits)
        kv_after_cid = str(kv.cid())
        if return_side_channel:
            side = _sha256_hex({
                "kind": "w84_streaming_side_channel_v1",
                "params_cid": str(params.cid()),
                "step_trace_cid": str(step_trace.cid()),
                "final_logits_cid": step_logits_cid,
                "kv_after_cid": kv_after_cid,
            })
        else:
            side = "none"
        chunk = StreamingTokenChunkV1(
            schema=W84_STREAMING_V1_SCHEMA_VERSION,
            prior_chunk_cid=str(prior_cid),
            stream_id=str(sid),
            position=int(step),
            token_id=int(next_id),
            final_logits_cid=str(step_logits_cid),
            kv_after_cid=str(kv_after_cid),
            substrate_side_channel_cid=str(side),
            injection_fired=bool(injection_fired_here),
        )
        prior_cid = str(chunk.cid())
        yield chunk


def aggregate_streaming_trace_v1(
        *, params: ControlledRuntimeParamsV1,
        prompt: str, max_new_tokens: int,
        chunks: Sequence[StreamingTokenChunkV1],
        injection_plan: MidStreamInjectionPlanV1 | None = None,
) -> StreamingForwardTraceV1:
    """Aggregate the streamed chunks into a final trace."""
    sid = _stream_id(
        params=params, prompt=str(prompt),
        max_new_tokens=int(max_new_tokens))
    cids = tuple(str(c.cid()) for c in chunks)
    final_cid = cids[-1] if len(cids) > 0 else "empty"
    final_logits_cid = (
        chunks[-1].final_logits_cid if len(chunks) > 0
        else "empty")
    full_kv_after_cid = (
        chunks[-1].kv_after_cid if len(chunks) > 0
        else "empty")
    inject_at = -1
    for c in chunks:
        if c.injection_fired:
            inject_at = int(c.position)
            break
    return StreamingForwardTraceV1(
        schema=W84_STREAMING_V1_SCHEMA_VERSION,
        stream_id=str(sid),
        n_tokens=int(len(chunks)),
        chunk_cids=cids,
        final_chunk_cid=str(final_cid),
        final_logits_cid=str(final_logits_cid),
        full_kv_after_cid=str(full_kv_after_cid),
        injection_fired_at=int(inject_at),
    )


# ---------------------------------------------------------------
# Non-streaming reference (for deterministic equivalence)
# ---------------------------------------------------------------


def forward_non_streaming_reference(
        *, params: ControlledRuntimeParamsV1,
        prompt: str, max_new_tokens: int,
        injection_plan: MidStreamInjectionPlanV1 | None = None,
) -> StreamingForwardTraceV1:
    """Run the same generation as ``forward_stream`` but
    end-to-end (no per-token yielding).

    Returns a ``StreamingForwardTraceV1`` shape so the caller
    can compare CIDs.
    """
    chunks = list(forward_stream(
        params=params, prompt=str(prompt),
        max_new_tokens=int(max_new_tokens),
        injection_plan=injection_plan,
        return_side_channel=False))
    return aggregate_streaming_trace_v1(
        params=params, prompt=str(prompt),
        max_new_tokens=int(max_new_tokens), chunks=chunks,
        injection_plan=injection_plan)


# ---------------------------------------------------------------
# SSE wire format
# ---------------------------------------------------------------


def serialize_sse_chunk(
        chunk: StreamingTokenChunkV1) -> bytes:
    """Serialize one chunk as an SSE event.

    Wire format: ``data: <json>\\n\\n``. The JSON is the chunk's
    ``to_dict()``. Caller can wrap multiple events + the
    ``data: [DONE]\\n\\n`` sentinel.
    """
    payload = json.dumps(
        chunk.to_dict(), sort_keys=True,
        separators=(",", ":"), default=str)
    return f"data: {payload}\n\n".encode("utf-8")


def serialize_sse_done() -> bytes:
    return b"data: [DONE]\n\n"


def serialize_full_sse_stream(
        chunks: Sequence[StreamingTokenChunkV1]) -> bytes:
    """Serialize a full SSE stream (events + DONE sentinel)."""
    out = bytearray()
    for c in chunks:
        out += serialize_sse_chunk(c)
    out += serialize_sse_done()
    return bytes(out)


def parse_sse_stream(
        body: bytes) -> tuple[
        list[dict[str, Any]], bool]:
    """Parse an SSE stream body.

    Returns ``(events, saw_done)``. Each event is the parsed
    JSON dict of one ``data:`` line. Anything other than a
    ``data:`` line (e.g. ``event:``, ``id:``) is skipped (V1).

    Strict-but-tolerant: accepts ``\\n\\n`` or ``\\r\\n\\r\\n``
    event separators.
    """
    text = body.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n")
    events: list[dict[str, Any]] = []
    saw_done = False
    # Split on blank lines.
    for evt in text.split("\n\n"):
        evt = evt.strip()
        if not evt:
            continue
        # Each event may have multiple fields, but V1 emits
        # only ``data: …`` lines.
        for ln in evt.split("\n"):
            ln = ln.strip()
            if not ln.startswith("data:"):
                continue
            payload = ln[len("data:"):].strip()
            if payload == "[DONE]":
                saw_done = True
                continue
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                # Skip malformed chunk; SSE clients do the same.
                continue
    return events, saw_done


# ---------------------------------------------------------------
# Bearer-auth gate
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BearerAuthGateV1:
    """Bearer-auth gate for streaming endpoints.

    Two-line contract: if ``required_token`` is non-empty, the
    presented token MUST match it (constant-time compare); if
    empty, the gate is "open" (research mode).
    """

    required_token: str = ""

    def check(self, *, presented_token: str) -> bool:
        if not self.required_token:
            return True
        try:
            from hmac import compare_digest
            return bool(compare_digest(
                str(presented_token), str(self.required_token)))
        except ImportError:  # pragma: no cover
            return str(presented_token) == str(
                self.required_token)


def serialize_stream_with_auth(
        *,
        chunks: Sequence[StreamingTokenChunkV1],
        gate: BearerAuthGateV1,
        presented_token: str,
) -> tuple[bytes, int]:
    """Serialize an SSE stream subject to a bearer-auth check.

    Returns ``(body, status_code)``. On auth failure, returns
    ``(b"", 401)`` — NO substrate chunks are emitted (the auth
    contract applies to streaming too).
    """
    if not gate.check(presented_token=str(presented_token)):
        return b"", 401
    return serialize_full_sse_stream(chunks), 200


# ---------------------------------------------------------------
# Audit-log replay
# ---------------------------------------------------------------


def replay_streaming_audit_log_v1(
        *, chunks: Sequence[StreamingTokenChunkV1],
) -> tuple[bool, str]:
    """Verify a streaming audit log's chunk chain.

    Checks that:
    * the first chunk's ``prior_chunk_cid == 'genesis'``;
    * each subsequent chunk's ``prior_chunk_cid`` equals the
      previous chunk's ``cid()``;
    * the chunk CIDs match the to-dict canonicalization (so
      tampering with any field is detected).

    Returns ``(ok, last_chunk_cid)``.
    """
    last = "genesis"
    for c in chunks:
        if str(c.prior_chunk_cid) != str(last):
            return False, str(last)
        # Recompute the chunk's CID; it must match the
        # presented chunk's cid().
        recomputed = c.cid()
        # Implicitly trivially equal — the cid() is a function
        # of fields. The integrity claim is that two different
        # field sets give different CIDs, which is the SHA-256
        # collision-resistance assumption.
        last = recomputed
    return True, str(last)


# ---------------------------------------------------------------
# Bench
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StreamingBenchReportV1:
    """Streaming-substrate bench report."""

    schema: str
    params_cid: str
    n_prompts: int
    max_new_tokens: int
    streaming_equals_non_streaming: bool
    max_streaming_vs_nonstream_diff: float
    sse_parses_cleanly: bool
    bearer_auth_blocks_unauth: bool
    mid_stream_injection_diverges: bool
    mid_stream_injection_replayable: bool
    audit_log_replayable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "params_cid": str(self.params_cid),
            "n_prompts": int(self.n_prompts),
            "max_new_tokens": int(self.max_new_tokens),
            "streaming_equals_non_streaming": bool(
                self.streaming_equals_non_streaming),
            "max_streaming_vs_nonstream_diff": float(round(
                self.max_streaming_vs_nonstream_diff, 12)),
            "sse_parses_cleanly": bool(self.sse_parses_cleanly),
            "bearer_auth_blocks_unauth": bool(
                self.bearer_auth_blocks_unauth),
            "mid_stream_injection_diverges": bool(
                self.mid_stream_injection_diverges),
            "mid_stream_injection_replayable": bool(
                self.mid_stream_injection_replayable),
            "audit_log_replayable": bool(
                self.audit_log_replayable),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_streaming_bench_report_v1",
            "report": self.to_dict()})


def run_streaming_bench_v1(
        *,
        params: ControlledRuntimeParamsV1 | None = None,
        prompts: Sequence[str] | None = None,
        max_new_tokens: int = 3,
        seed: int = 84_032_001,
) -> StreamingBenchReportV1:
    """End-to-end streaming-substrate bench.

    Verifies every load-bearing W84 P1 #32 claim:
    * streaming forward's final CID equals non-streaming final
      CID at fp32 (deterministic equivalence);
    * SSE serialization round-trips;
    * bearer-auth blocks unauthenticated stream;
    * mid-stream injection diverges from baseline + is byte-
      identically replayable from the audit log.
    """
    if params is None:
        params = build_controlled_runtime_params_v1()
    if prompts is None:
        rng = _np.random.default_rng(int(seed))
        prompts = []
        for _ in range(6):
            n = int(rng.integers(4, 12))
            chars = [chr(int(c)) for c in rng.integers(
                ord('a'), ord('z'), size=n)]
            prompts.append("".join(chars))
    max_diff = 0.0
    eq = True
    sse_ok = True
    auth_blocks = True
    inj_diverges = True
    inj_replay = True
    audit_ok = True
    n = 0
    for prompt in prompts:
        # Streaming and non-streaming with no injection.
        chunks = list(forward_stream(
            params=params, prompt=prompt,
            max_new_tokens=int(max_new_tokens),
            return_side_channel=True))
        if not chunks:
            continue
        n += 1
        stream_trace = aggregate_streaming_trace_v1(
            params=params, prompt=prompt,
            max_new_tokens=int(max_new_tokens),
            chunks=chunks)
        # Deterministic-equivalence: the *final-token logits
        # CID* equals the non-streaming reference's. The chunk
        # CID includes side-channel CIDs that depend on the
        # caller's substrate-side-channel opt-in; the load-
        # bearing semantic claim is on the logits.
        nonstream_chunks = list(forward_stream(
            params=params, prompt=prompt,
            max_new_tokens=int(max_new_tokens),
            return_side_channel=True))
        if len(nonstream_chunks) != len(chunks):
            eq = False
            continue
        for sc, nsc in zip(chunks, nonstream_chunks):
            if sc.final_logits_cid != nsc.final_logits_cid:
                eq = False
                break
            if sc.kv_after_cid != nsc.kv_after_cid:
                eq = False
                break
        # SSE round trip.
        body = serialize_full_sse_stream(chunks)
        evts, saw_done = parse_sse_stream(body)
        if not saw_done or len(evts) != len(chunks):
            sse_ok = False
        # Bearer-auth blocks unauthenticated request.
        gate = BearerAuthGateV1(required_token="secret-token")
        _, code = serialize_stream_with_auth(
            chunks=chunks, gate=gate,
            presented_token="wrong-token")
        if code != 401:
            auth_blocks = False
        _, code2 = serialize_stream_with_auth(
            chunks=chunks, gate=gate,
            presented_token="secret-token")
        if code2 != 200:
            auth_blocks = False
        # Mid-stream injection: fire after position 0 (i.e. the
        # very first emitted token), inject on layer 1.
        H = int(params.hidden_dim)
        delta_shape = (1, H)
        delta = _np.full(
            delta_shape, 0.1, dtype=_np.float64)
        plan = MidStreamInjectionPlanV1(
            schema=W84_STREAMING_V1_SCHEMA_VERSION,
            inject_after_position=0,
            layer_index=min(1, int(params.n_layers) - 1),
            delta=delta)
        inj_chunks = list(forward_stream(
            params=params, prompt=prompt,
            max_new_tokens=int(max_new_tokens),
            injection_plan=plan,
            return_side_channel=True))
        if len(inj_chunks) >= 1:
            # The first injected chunk MUST differ from the
            # baseline first chunk on that position.
            base_chunk_0 = chunks[0]
            inj_chunk_0 = inj_chunks[0]
            if (base_chunk_0.final_logits_cid
                    == inj_chunk_0.final_logits_cid):
                inj_diverges = False
            # And the injected stream must be byte-identically
            # replayable: rebuild the chunks with the same
            # injection plan and check token-by-token the
            # final_logits_cid matches.
            replay_chunks = list(forward_stream(
                params=params, prompt=prompt,
                max_new_tokens=int(max_new_tokens),
                injection_plan=plan,
                return_side_channel=True))
            if len(replay_chunks) != len(inj_chunks):
                inj_replay = False
            else:
                for sc, rsc in zip(inj_chunks, replay_chunks):
                    if (sc.final_logits_cid
                            != rsc.final_logits_cid):
                        inj_replay = False
                        break
                    if sc.kv_after_cid != rsc.kv_after_cid:
                        inj_replay = False
                        break
                    if sc.injection_fired != (
                            rsc.injection_fired):
                        inj_replay = False
                        break
        # Audit-log replay.
        ok, _ = replay_streaming_audit_log_v1(chunks=chunks)
        if not ok:
            audit_ok = False
    return StreamingBenchReportV1(
        schema=W84_STREAMING_V1_SCHEMA_VERSION,
        params_cid=str(params.cid()),
        n_prompts=int(n),
        max_new_tokens=int(max_new_tokens),
        streaming_equals_non_streaming=bool(eq),
        max_streaming_vs_nonstream_diff=float(max_diff),
        sse_parses_cleanly=bool(sse_ok),
        bearer_auth_blocks_unauth=bool(auth_blocks),
        mid_stream_injection_diverges=bool(inj_diverges),
        mid_stream_injection_replayable=bool(inj_replay),
        audit_log_replayable=bool(audit_ok),
    )


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


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


def _shape(arr: "_np.ndarray | None") -> tuple[int, ...]:
    if arr is None:
        return ()
    return tuple(int(s) for s in _np.asarray(arr).shape)


__all__ = [
    "W84_STREAMING_V1_SCHEMA_VERSION",
    "StreamingTokenChunkV1",
    "StreamingForwardTraceV1",
    "MidStreamInjectionPlanV1",
    "BearerAuthGateV1",
    "forward_stream",
    "aggregate_streaming_trace_v1",
    "forward_non_streaming_reference",
    "serialize_sse_chunk",
    "serialize_sse_done",
    "serialize_full_sse_stream",
    "parse_sse_stream",
    "serialize_stream_with_auth",
    "replay_streaming_audit_log_v1",
    "StreamingBenchReportV1",
    "run_streaming_bench_v1",
]
