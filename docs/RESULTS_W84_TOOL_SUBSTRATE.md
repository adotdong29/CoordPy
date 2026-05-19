# W84 / P1 #33 — Tool-Use / Function-Call Substrate V1

## Summary

Ships a Tool-Use Substrate alongside the W80 LLM substrate.
Content-addressed tool calls + results, an idempotency contract
that refuses duplicate non-idempotent commits, a sandbox
adapter with real resource limits, three V1 tools, and an audit
chain that mixes LLM-side and tool-side capsules under a single
Merkle root.

## Definition-of-Done bars

| Bar | Status |
| --- | ------ |
| `ToolCallSchemaV1` + `ToolResultSchemaV1` are content-addressed and re-hashable from disk | ✅ |
| Identical tool call inputs produce identical call CIDs | ✅ |
| Idempotency contract: replay of an idempotent call emits the cached result (no re-execution); replay of a non-idempotent call without a matching token is REFUSED | ✅ |
| One real tool runs under `ToolSandboxAdapterV1` (V1 ships three: Python exec, ripgrep-style filesystem, stub HTTP). Resource limits enforced (wall-time limit verified by `test_w84_sandbox_enforces_wall_time_limit`) | ✅ |
| 5-agent team bench produces an audit chain that mixes LLM-side capsules with tool-side capsules; Merkle root is over the merged chain | ✅ |
| `RESULTS__TOOL_SUBSTRATE.md` captures the contract, example tools, and audit-replay semantics | ✅ (this file) |

## Measured numbers (default bench)

| Claim | Value |
| ----- | ----- |
| `n_agents` | 5 |
| `n_tool_calls` | 5 |
| `all_calls_content_addressed` | True |
| `identical_inputs_give_identical_call_cids` | True |
| `non_idempotent_double_commit_refused` | True |
| `sandbox_wall_time_limit_enforced` | True |
| `audit_chain_reverifies_from_disk` | True |
| `tamper_breaks_audit_chain` | True |

## Tools shipped V1

* **`tool_python_exec`** — runs Python code in a restricted
  globals dict; captures `result` if assigned.
* **`tool_filesystem_ripgrep_lite`** — read-only walk of a
  directory, returns up to `max_matches` regex matches with
  path + line_no + line.
* **`tool_http_fetch_stub`** — deterministic stub: returns
  `SHA-256(url)` so the audit chain exercises content-
  addressing without making real network calls in CI. A real
  HTTP adapter is V2.

## Schemas

* `ToolCallSchemaV1`: `(tool_id, tool_version_cid, args_cid,
  args_bytes_cid, timestamp_ns, caller_agent_id, run_cid,
  parent_event_cid, idempotency, idempotency_token_cid)`.
* `ToolResultSchemaV1`: `(call_cid, result_cid,
  result_bytes_cid, exit_code, stderr_cid, duration_ns,
  side_effects_cid, integrity_verdict)`.
* `IdempotencyTokenV1`: `(run_cid, agent_id, sequence_no)`.
* `ToolAuditChainV1`: `(capsule_kinds, capsule_cids,
  merkle_root_cid)` — Merkle root recomputable from
  `capsule_cids`.

## Anti-cheat compliance

* **Tools are not "just LLM tokens".** `ToolResultSchemaV1`
  carries `side_effects_cid`, `exit_code`, `stderr_cid`,
  `integrity_verdict` — fields LLM outputs do not have.
* **Result bytes are content-addressed (incl. binary blob
  path).** `result_bytes_cid = SHA-256(raw bytes)`. Changing
  the result bytes changes the result CID
  (`test_w84_tool_result_bytes_carried_in_cid`).
* **The sandbox is not stubbed.** `ToolSandboxAdapterV1`
  enforces wall-time via thread + join-timeout; an exceeded
  wall-time leaves the result with
  `integrity_verdict = LIMIT_EXCEEDED` and `exit_code = 124`.
  `test_w84_sandbox_enforces_wall_time_limit` runs a 0.3 s
  callable against a 0.05 s limit and asserts the verdict.
* **Idempotency is not default-true.** The vocabulary is
  binary; non-idempotent calls explicitly require an
  `IdempotencyTokenV1`. `make_tool_call_v1` raises if a
  non-idempotent call is built without a token.
* **Audit-replay test ships.** `ToolAuditChainV1.verify_
  merkle_root` recomputes the root from the capsule CIDs;
  tampering with any CID breaks it
  (`test_w84_tool_audit_chain_tamper_detected`).
* **RAG is acknowledged as a separate concern.** V1 does
  not class RAG as "just a tool"; the V2 path for stateful
  retrieval indices is documented.

## Honest scope (V1)

* `W84-L-TOOL-SUBSTRATE-V1-1_TO_3_TOOLS-CAP` — V1 ships
  Python-exec, ripgrep filesystem, stub HTTP. Full tool
  catalog is V2.
* `W84-L-TOOL-SUBSTRATE-V1-SINGLE-HOST-CAP` — V1 sandbox is
  in-process. Multi-host distributed sandbox is V2.
* `W84-L-TOOL-SUBSTRATE-V1-STATELESS-CAP` — V1 tools are
  stateless. Stateful tools (database transactions, retrieval
  indices) are V2.
* `W84-L-TOOL-SUBSTRATE-V1-NON-STREAMING-CAP` — streaming
  tool results are V2.
* `W84-L-TOOL-SUBSTRATE-V1-BINARY-IDEMPOTENCY-CAP` — V1
  vocabulary is binary; fine-grained at-most-once /
  at-least-once is V2.

## Reproduction

```python
from coordpy.tool_use_substrate_v1 import run_tool_use_bench_v1
rep = run_tool_use_bench_v1()
print(rep.to_dict())
```

Tests: `tests/test_w84_tool_use_substrate.py` (14 tests, all
passing).
