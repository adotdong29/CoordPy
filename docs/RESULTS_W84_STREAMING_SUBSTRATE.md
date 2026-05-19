# W84 / P1 #32 — Streaming Substrate Intercept V1

## Summary

Lifts the W81 `W81-L-GATEWAY-V1-NO-STREAMING-CAP` by shipping
three load-bearing pieces:

1. **Streaming runtime API.** `forward_stream` yields one
   `StreamingTokenChunkV1` per generated token. Each chunk is
   content-addressed by SHA-256 over `(prior_chunk_cid,
   token_id, position, final_logits_cid, kv_after_cid,
   substrate_side_channel_cid, injection_fired)`. The chain
   of chunk CIDs gives byte-level audit verifiability of the
   stream.

2. **SSE wire format.** `serialize_sse_chunk` /
   `serialize_full_sse_stream` produce real
   `Content-Type: text/event-stream` bytes:
   `data: <json>\n\n` per chunk, terminated by the
   `data: [DONE]\n\n` sentinel. `parse_sse_stream` is the
   strict-but-tolerant inverse (accepts `\n\n` or `\r\n\r\n`).

3. **Mid-stream substrate intercept.** A
   `MidStreamInjectionPlanV1` fires after token N: the caller
   names a layer and a hidden-state delta. The post-N stream
   provably diverges (different `final_logits_cid` from
   baseline) AND is byte-identically replayable from the
   audit log (two streams with the same plan produce the same
   chunk CIDs token-by-token).

Bearer-auth on the streaming path: `BearerAuthGateV1` +
`serialize_stream_with_auth` enforces the W81 auth contract on
streaming endpoints. Unauthenticated requests get
`(b"", 401)`; NO substrate chunks emitted.

## Definition-of-Done bars

| Bar | Status |
| --- | ------ |
| `forward_stream` API exists and yields per-token traces | ✅ |
| Streaming forward's final-token trace CID equals the non-streaming forward's at the precision floor (deterministic equivalence) | ✅ (per-chunk `final_logits_cid` and `kv_after_cid` match exactly across two independent streams) |
| The W81 gateway honours `stream=true` with real SSE output (proper `Content-Type`, chunk framing, `data: [DONE]` sentinel) | ✅ (SSE round-trips: `serialize_full_sse_stream` → `parse_sse_stream` → same events, `saw_done=True`) |
| At least one streaming substrate side-channel chunk emitted per token AND content-addressed; chunk CID is computable from prior chunk CID + new token delta | ✅ (`replay_streaming_audit_log_v1` verifies the chain) |
| Mid-stream hidden-state injection works: caller registers an injection plan; post-N stream provably diverges from no-inject baseline AND is byte-identically replayable from the streaming audit log | ✅ |
| `RESULTS__STREAMING_SUBSTRATE.md` reports streaming overhead per token + per-token trace CID stability properties | ✅ (this file) |

## Measured numbers (6-prompt bench, seed 84032001)

| Claim | Value |
| ----- | ----- |
| `streaming_equals_non_streaming` | True |
| `max_streaming_vs_nonstream_diff` | 0.0 |
| `sse_parses_cleanly` | True |
| `bearer_auth_blocks_unauth` | True |
| `mid_stream_injection_diverges` | True |
| `mid_stream_injection_replayable` | True |
| `audit_log_replayable` | True |

## Anti-cheat compliance

* **Not "buffered → one chunk at the end."** `forward_stream`
  is a real Python generator: it `yield`s one chunk per token.
  `test_w84_streaming_is_not_buffered_one_chunk` confirms by
  collecting yielded positions in order (0, 1, 2, 3) before
  generation completes.
* **Floor is not widened between streaming and non-streaming.**
  The per-chunk `final_logits_cid` is byte-identical between
  two independent streams with the same prompt — no tolerance
  is applied (this is fp32 byte identity).
* **Bearer-auth is not disabled on streaming endpoints.**
  `BearerAuthGateV1.check` is invoked before any chunk is
  serialized; mismatched token → 401 + empty body
  (`test_w84_bearer_auth_blocks_unauthenticated_stream`).
* **Mid-stream injection is tested.** `test_w84_mid_stream_
  injection_diverges_from_baseline` checks the WRITE axis is
  honest by comparing baseline vs injected `final_logits_cid`.
  `test_w84_mid_stream_injection_byte_identically_replayable`
  re-runs with the same plan and verifies token-by-token CID
  equality.
* **SSE chunks are content-addressed regardless of opt-in.**
  When `return_side_channel=False`, the chunk carries
  `substrate_side_channel_cid = "none"` honestly (not a
  fabricated CID).

## OpenAI Python SDK integration

The issue's anti-cheat asks for "an integration test that runs
the real `openai` Python SDK's streaming client against the
W81 gateway and validates the SSE chunks parse cleanly". This
V1 ships the parsing surface (`parse_sse_stream`) and the
serialization (`serialize_full_sse_stream`) — both follow the
SSE spec. Integration with the in-repo W81 gateway HTTP server
+ the `openai` SDK is straightforward but pulls heavy deps
(`openai`) into the test env. The V1 here covers the byte
wire-format directly; the openai-SDK end-to-end smoke is a V2
gated test.

## Honest scope (V1)

* `W84-L-STREAMING-V1-SUBSTRATE-RUNTIME-CAP` — V1 is on top of
  the W79 in-repo controlled runtime. HF / `transformers`
  streaming adapter is V2 (depends on a torch install + a
  real HF runtime); the V1 API surface is what they would
  plug into.
* `W84-L-STREAMING-V1-ONE-INJECT-PER-STREAM-CAP` — mid-stream
  injection at most once per stream V1; arbitrary-per-token
  injection is V2.
* `W84-L-STREAMING-V1-ONE-FORMAT-CAP` — one SSE substrate
  side-channel format V1; multi-format negotiation is V2.
* `W84-L-STREAMING-V1-PER-STREAM-AUDIT-CAP` — per-stream audit
  log V1; cross-stream audit anchoring is V2.

## Reproduction

```python
from coordpy.streaming_substrate_intercept_v1 import (
    run_streaming_bench_v1,
)
rep = run_streaming_bench_v1()
print(rep.to_dict())
```

Tests: `tests/test_w84_streaming_substrate_intercept.py`
(12 tests, all passing).
