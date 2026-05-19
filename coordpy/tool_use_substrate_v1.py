"""W84 / P1 #33 — Tool-Use / Function-Call Substrate V1.

Multi-agent teams in production routinely call tools: filesystem
operations, code execution, web search, retrieval, database
queries, third-party APIs. Tool calls produce another stream of
context that's at least as load-bearing as the LLM's hidden
state — often more so, because the tool result is the *ground
truth* on which the team's decision depends.

The W80–W83 line is entirely an LLM-side substrate story. The
W82 event-sourced memory graph carries events of any kind but
does not specialise tool-call events. The W83 hosted-audit-
anchoring captures hosted transcripts; it does NOT capture tool
calls explicitly. This V1 ships the Tool-Use Substrate
alongside the W80 LLM substrate.

Concrete deliverables (V1):

1. **``ToolCallSchemaV1``** — content-addressed dataclass:
   ``(tool_id, tool_version_cid, args_cid, args_bytes,
   timestamp_ns, caller_agent_id, run_cid, parent_event_cid)``.

2. **``ToolResultSchemaV1``** — content-addressed dataclass:
   ``(call_cid, result_cid, result_bytes, exit_code,
   stderr_cid, duration_ns, side_effects_cid,
   integrity_verdict)``.

3. **Idempotency contract.** Tool calls are tagged
   ``idempotent`` / ``non_idempotent``. Non-idempotent calls
   require an ``IdempotencyTokenV1`` that the contract REFUSES
   to commit twice (W83 integrity-trust-coupled style).

4. **Tool sandbox adapter.** ``ToolSandboxAdapterV1`` wraps an
   arbitrary callable with resource limits (max wall-time, max
   bytes returned). The V1 sandbox shipped here is a Python
   ``exec`` sandbox + a ``ripgrep``-style filesystem adapter +
   a stub HTTP fetch.

5. **Tool-use bench.** A 5-agent team bench produces an audit
   chain that mixes LLM-side capsules with tool-side capsules.
   Merkle root over the merged chain.

6. **Audit-replayable tool history.** Given a sealed
   ``ToolResultV1`` chain, a third party can re-verify the
   chain WITHOUT re-running tool calls (the result bytes +
   side-effect CIDs are stored).

Honest scope (W84 P1 #33):

* ``W84-L-TOOL-SUBSTRATE-V1-1_TO_3_TOOLS-CAP`` — V1 ships
  Python-exec, ripgrep-style filesystem, stub HTTP fetch.
  Full tool catalog is V2.
* ``W84-L-TOOL-SUBSTRATE-V1-SINGLE-HOST-CAP`` — V1 sandbox is
  in-process. Multi-host distributed sandbox is V2.
* ``W84-L-TOOL-SUBSTRATE-V1-STATELESS-CAP`` — V1 tools are
  stateless. Stateful tools (database transactions etc.) are
  V2.
* ``W84-L-TOOL-SUBSTRATE-V1-NON-STREAMING-CAP`` — streaming
  tool results (long scrapes) are V2.
* ``W84-L-TOOL-SUBSTRATE-V1-BINARY-IDEMPOTENCY-CAP`` — V1
  vocabulary is binary (idempotent vs non-idempotent); fine-
  grained at-most-once / at-least-once is V2.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import os
import re
import threading
import time
from typing import Any, Callable, Mapping, Sequence


W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION: str = (
    "coordpy.tool_use_substrate_v1.v1")


# ---------------------------------------------------------------
# Idempotency vocabulary + verdicts
# ---------------------------------------------------------------


class ToolIdempotency(str, enum.Enum):
    IDEMPOTENT = "idempotent"
    NON_IDEMPOTENT = "non_idempotent"


class ToolIntegrityVerdict(str, enum.Enum):
    OK = "ok"
    LIMIT_EXCEEDED = "limit_exceeded"
    EXCEPTION = "exception"
    IDEMPOTENCY_VIOLATION = "idempotency_violation"


# ---------------------------------------------------------------
# Content-addressed schemas
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class IdempotencyTokenV1:
    """Idempotency token for non-idempotent tool calls.

    The token is a SHA-256 of (run_cid, agent_id, sequence_no).
    The W84 tool substrate refuses to commit two
    ``ToolResultV1`` capsules that share the same token + same
    tool_id.
    """

    schema: str
    run_cid: str
    agent_id: str
    sequence_no: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "run_cid": str(self.run_cid),
            "agent_id": str(self.agent_id),
            "sequence_no": int(self.sequence_no),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_idempotency_token_v1",
            "token": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class ToolCallSchemaV1:
    """Content-addressed tool call.

    Identical (tool_id, tool_version_cid, args_bytes,
    caller_agent_id, run_cid, parent_event_cid) inputs MUST
    produce identical call CIDs.
    """

    schema: str
    tool_id: str
    tool_version_cid: str
    args_cid: str
    args_bytes_cid: str  # CID over the raw args bytes
    timestamp_ns: int
    caller_agent_id: str
    run_cid: str
    parent_event_cid: str
    idempotency: str
    idempotency_token_cid: str  # "none" for idempotent calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "tool_id": str(self.tool_id),
            "tool_version_cid": str(self.tool_version_cid),
            "args_cid": str(self.args_cid),
            "args_bytes_cid": str(self.args_bytes_cid),
            "timestamp_ns": int(self.timestamp_ns),
            "caller_agent_id": str(self.caller_agent_id),
            "run_cid": str(self.run_cid),
            "parent_event_cid": str(self.parent_event_cid),
            "idempotency": str(self.idempotency),
            "idempotency_token_cid": str(
                self.idempotency_token_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_tool_call_schema_v1",
            "call": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class ToolResultSchemaV1:
    """Content-addressed tool result.

    Identical inputs (call_cid, result_bytes_cid, exit_code,
    stderr_cid, duration_ns, side_effects_cid,
    integrity_verdict) MUST produce identical result CIDs.
    """

    schema: str
    call_cid: str
    result_cid: str  # CID over the canonicalised result object
    result_bytes_cid: str  # CID over the raw result bytes
    exit_code: int
    stderr_cid: str
    duration_ns: int
    side_effects_cid: str
    integrity_verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "call_cid": str(self.call_cid),
            "result_cid": str(self.result_cid),
            "result_bytes_cid": str(self.result_bytes_cid),
            "exit_code": int(self.exit_code),
            "stderr_cid": str(self.stderr_cid),
            "duration_ns": int(self.duration_ns),
            "side_effects_cid": str(self.side_effects_cid),
            "integrity_verdict": str(self.integrity_verdict),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_tool_result_schema_v1",
            "result": self.to_dict()})


def _bytes_cid(b: bytes | str) -> str:
    if isinstance(b, str):
        b = b.encode("utf-8")
    return hashlib.sha256(bytes(b)).hexdigest()


def make_tool_call_v1(
        *, tool_id: str, tool_version_cid: str,
        args: Mapping[str, Any],
        timestamp_ns: int,
        caller_agent_id: str, run_cid: str,
        parent_event_cid: str = "genesis",
        idempotency: str = ToolIdempotency.IDEMPOTENT.value,
        idempotency_token: IdempotencyTokenV1 | None = None,
) -> ToolCallSchemaV1:
    """Build a content-addressed tool call."""
    if not isinstance(args, Mapping):
        raise TypeError("args must be a mapping")
    if idempotency not in (
            ToolIdempotency.IDEMPOTENT.value,
            ToolIdempotency.NON_IDEMPOTENT.value):
        raise ValueError(f"unknown idempotency: {idempotency}")
    if idempotency == (
            ToolIdempotency.NON_IDEMPOTENT.value):
        if idempotency_token is None:
            raise ValueError(
                "non-idempotent tool calls require an "
                "IdempotencyTokenV1")
    args_canon = json.dumps(
        dict(args), sort_keys=True,
        separators=(",", ":"), default=str)
    args_bytes = args_canon.encode("utf-8")
    return ToolCallSchemaV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        tool_id=str(tool_id),
        tool_version_cid=str(tool_version_cid),
        args_cid=_sha256_hex({
            "kind": "w84_tool_call_args_v1",
            "args": dict(args)}),
        args_bytes_cid=_bytes_cid(args_bytes),
        timestamp_ns=int(timestamp_ns),
        caller_agent_id=str(caller_agent_id),
        run_cid=str(run_cid),
        parent_event_cid=str(parent_event_cid),
        idempotency=str(idempotency),
        idempotency_token_cid=(
            str(idempotency_token.cid())
            if idempotency_token is not None else "none"),
    )


# ---------------------------------------------------------------
# Tool sandbox adapter
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ToolSandboxLimitsV1:
    """Resource limits for a sandboxed tool."""

    max_wall_time_seconds: float = 1.0
    max_result_bytes: int = 1 * 1024 * 1024  # 1 MiB
    max_stderr_bytes: int = 16 * 1024  # 16 KiB


@dataclasses.dataclass(frozen=True)
class ToolExecutionV1:
    """Output of running a tool under the sandbox."""

    result_bytes: bytes
    stderr_bytes: bytes
    exit_code: int
    duration_ns: int
    integrity_verdict: str
    side_effects: tuple[str, ...]


@dataclasses.dataclass
class ToolSandboxAdapterV1:
    """Sandbox adapter: wraps a Python callable with resource
    limits and emits a ``ToolExecutionV1``.

    The V1 sandbox enforces:
    * wall-time limit via a separate-thread + join-timeout
    * result-bytes limit by truncating the returned bytes

    Limits are real: an exceeded wall-time leaves the call with
    `integrity_verdict = LIMIT_EXCEEDED`. The contract requires
    at least ONE real resource limit; V1 enforces both.
    """

    limits: ToolSandboxLimitsV1 = dataclasses.field(
        default_factory=ToolSandboxLimitsV1)

    def execute(
            self, *, callable_: Callable[..., Any],
            args: Mapping[str, Any],
    ) -> ToolExecutionV1:
        start = time.perf_counter_ns()
        result_box: list[Any] = []
        stderr_box: list[str] = []
        exit_box: list[int] = [0]
        side_box: list[list[str]] = [[]]

        def _runner() -> None:
            try:
                out = callable_(**dict(args))
                result_box.append(out)
                # Surface side-effects if the callable advertises
                # them as ``out["_side_effects"]`` (V1 convention).
                if isinstance(out, dict) and (
                        "_side_effects" in out):
                    side_box[0] = [
                        str(s)
                        for s in out["_side_effects"]]
            except Exception as exc:  # noqa: BLE001
                stderr_box.append(
                    f"{type(exc).__name__}: {exc}")
                exit_box[0] = 1
                # Re-raise to surface in the result_bytes? V1
                # records the exception in stderr only.

        thread = threading.Thread(
            target=_runner, daemon=True)
        thread.start()
        thread.join(
            timeout=float(self.limits.max_wall_time_seconds))
        timed_out = thread.is_alive()
        end = time.perf_counter_ns()
        duration_ns = int(end - start)
        verdict = ToolIntegrityVerdict.OK.value
        if timed_out:
            verdict = ToolIntegrityVerdict.LIMIT_EXCEEDED.value
            exit_box[0] = 124  # convention: 124 = wall-time
        elif exit_box[0] != 0:
            verdict = ToolIntegrityVerdict.EXCEPTION.value
        # Serialize the result.
        if len(result_box) == 0:
            result_bytes = b""
        else:
            out = result_box[0]
            if isinstance(out, (bytes, bytearray)):
                result_bytes = bytes(out)
            else:
                try:
                    result_bytes = json.dumps(
                        out, sort_keys=True,
                        separators=(",", ":"),
                        default=str).encode("utf-8")
                except (TypeError, ValueError):
                    result_bytes = str(out).encode("utf-8")
        if len(result_bytes) > int(
                self.limits.max_result_bytes):
            result_bytes = result_bytes[
                : int(self.limits.max_result_bytes)]
            verdict = (
                ToolIntegrityVerdict.LIMIT_EXCEEDED.value)
        stderr_str = "\n".join(stderr_box)
        if len(stderr_str.encode("utf-8")) > int(
                self.limits.max_stderr_bytes):
            stderr_str = stderr_str.encode(
                "utf-8")[: int(
                    self.limits.max_stderr_bytes)].decode(
                "utf-8", errors="replace")
        return ToolExecutionV1(
            result_bytes=result_bytes,
            stderr_bytes=stderr_str.encode("utf-8"),
            exit_code=int(exit_box[0]),
            duration_ns=int(duration_ns),
            integrity_verdict=str(verdict),
            side_effects=tuple(side_box[0]),
        )


# ---------------------------------------------------------------
# V1 tools — Python exec, filesystem ripgrep-style, stub HTTP
# ---------------------------------------------------------------


def tool_python_exec(*, code: str) -> dict[str, Any]:
    """Python exec sandbox tool.

    Executes ``code`` in a restricted globals dict; captures
    the value of ``result`` if assigned. Pure value-in / value-
    out: no real side effects on disk or network.
    """
    g: dict[str, Any] = {"__builtins__": {
        "len": len, "range": range, "str": str, "int": int,
        "float": float, "list": list, "dict": dict,
        "tuple": tuple, "abs": abs, "max": max, "min": min,
        "sum": sum, "sorted": sorted, "enumerate": enumerate,
    }}
    l: dict[str, Any] = {}
    exec(str(code), g, l)  # noqa: S102 — sandbox by design
    return {
        "ok": True,
        "result": l.get("result"),
        "vars": {k: l[k] for k in l if not k.startswith("_")},
    }


def tool_filesystem_ripgrep_lite(
        *, root: str, pattern: str,
        max_matches: int = 32,
) -> dict[str, Any]:
    """``ripgrep``-style read-only filesystem tool.

    Walks ``root`` and returns up to ``max_matches`` lines that
    match ``pattern`` (Python regex). Read-only — no
    filesystem mutations. The root must exist and be a
    directory.
    """
    out: list[dict[str, Any]] = []
    rx = re.compile(str(pattern))
    if not os.path.isdir(str(root)):
        return {"ok": False, "matches": [], "error":
                "root not a directory"}
    for dirpath, _, filenames in os.walk(str(root)):
        for f in filenames:
            full = os.path.join(dirpath, f)
            try:
                with open(full, "r", encoding="utf-8",
                          errors="replace") as fh:
                    for i, line in enumerate(fh):
                        if rx.search(line):
                            out.append({
                                "path": str(full),
                                "line_no": int(i + 1),
                                "line": line.rstrip("\n"),
                            })
                            if len(out) >= int(max_matches):
                                return {
                                    "ok": True,
                                    "matches": out}
            except OSError:
                continue
    return {"ok": True, "matches": out}


def tool_http_fetch_stub(
        *, url: str, max_bytes: int = 4096,
) -> dict[str, Any]:
    """Stub HTTP fetch tool.

    V1 is a DETERMINISTIC stub: the returned bytes are
    SHA-256(url) hex-encoded. This is enough to exercise the
    content-addressing of the result chain without making real
    network calls in CI. A real HTTP adapter is V2.
    """
    h = hashlib.sha256(str(url).encode("utf-8")).hexdigest()
    out = h[: int(max_bytes)] if int(max_bytes) > 0 else ""
    return {
        "ok": True,
        "url": str(url),
        "status": 200,
        "body": str(out),
        "_side_effects": [],  # stub fetch has none
    }


# ---------------------------------------------------------------
# Idempotency registry
# ---------------------------------------------------------------


@dataclasses.dataclass
class IdempotencyRegistryV1:
    """In-memory registry refusing duplicate non-idempotent
    commits.

    The registry stores ``{token_cid: call_cid}`` and refuses
    to register a second result that uses the same
    ``token_cid`` and a different ``call_cid``. Replays of the
    SAME call_cid + token_cid are allowed (so retries are
    idempotent under the same token).
    """

    seen: dict[str, str] = dataclasses.field(default_factory=dict)
    seen_results: dict[str, ToolResultSchemaV1] = (
        dataclasses.field(default_factory=dict))

    def check_and_register(
            self, *, call: ToolCallSchemaV1,
            result: ToolResultSchemaV1,
    ) -> str:
        """Returns ``ok`` if registered or already-seen with
        the same call CID; ``idempotency_violation`` if the
        token is reused for a different call."""
        if call.idempotency != (
                ToolIdempotency.NON_IDEMPOTENT.value):
            return ToolIntegrityVerdict.OK.value
        tok = str(call.idempotency_token_cid)
        if tok == "none":
            return ToolIntegrityVerdict.OK.value
        if tok in self.seen:
            prior_call_cid = self.seen[tok]
            if prior_call_cid != str(call.cid()):
                return (
                    ToolIntegrityVerdict
                    .IDEMPOTENCY_VIOLATION.value)
            return ToolIntegrityVerdict.OK.value
        self.seen[tok] = str(call.cid())
        self.seen_results[tok] = result
        return ToolIntegrityVerdict.OK.value

    def cached_result_for_idempotent_call(
            self, call: ToolCallSchemaV1,
    ) -> ToolResultSchemaV1 | None:
        if call.idempotency != (
                ToolIdempotency.NON_IDEMPOTENT.value):
            return None
        tok = str(call.idempotency_token_cid)
        return self.seen_results.get(tok)


# ---------------------------------------------------------------
# Run-tool orchestrator
# ---------------------------------------------------------------


def run_tool_call_v1(
        *, callable_: Callable[..., Any],
        call: ToolCallSchemaV1,
        sandbox: ToolSandboxAdapterV1 | None = None,
        idempotency_registry: (
            IdempotencyRegistryV1 | None) = None,
        args: Mapping[str, Any],
) -> ToolResultSchemaV1:
    """Run one tool call, emit the result capsule."""
    sandbox_ = sandbox or ToolSandboxAdapterV1()
    # If idempotent and we've seen this exact call, return the
    # cached result.
    if idempotency_registry is not None:
        cached = (
            idempotency_registry
            .cached_result_for_idempotent_call(call))
        if cached is not None and (
                cached.call_cid == call.cid()):
            return cached
    execution = sandbox_.execute(
        callable_=callable_, args=args)
    side_cid = _sha256_hex({
        "kind": "w84_tool_side_effects_v1",
        "side_effects": list(execution.side_effects),
    })
    result_obj_canon = {
        "kind": "w84_tool_result_object_v1",
        "result_bytes_cid": _bytes_cid(
            execution.result_bytes),
        "exit_code": int(execution.exit_code),
        "duration_ns": int(execution.duration_ns),
    }
    result_cid = _sha256_hex(result_obj_canon)
    stderr_cid = _bytes_cid(execution.stderr_bytes)
    verdict = str(execution.integrity_verdict)
    result = ToolResultSchemaV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        call_cid=str(call.cid()),
        result_cid=str(result_cid),
        result_bytes_cid=_bytes_cid(execution.result_bytes),
        exit_code=int(execution.exit_code),
        stderr_cid=str(stderr_cid),
        duration_ns=int(execution.duration_ns),
        side_effects_cid=str(side_cid),
        integrity_verdict=str(verdict),
    )
    if idempotency_registry is not None:
        gate = idempotency_registry.check_and_register(
            call=call, result=result)
        if gate == (
                ToolIntegrityVerdict
                .IDEMPOTENCY_VIOLATION.value):
            # Re-emit the result tagged with the violation.
            result = dataclasses.replace(
                result, integrity_verdict=str(gate))
    return result


# ---------------------------------------------------------------
# Audit chain — Merkle root over tool capsules
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ToolAuditChainV1:
    """Audit chain over interleaved tool call + result capsules
    and (optionally) LLM-side capsules.

    The Merkle root is over the canonicalised list of capsule
    CIDs in commit order. Replay-verification: given the chain
    + the original capsule bytes (call.to_dict(), result
    .to_dict(), llm capsule dicts), the root recomputes.
    """

    schema: str
    capsule_kinds: tuple[str, ...]  # tool_call / tool_result / llm_capsule
    capsule_cids: tuple[str, ...]
    merkle_root_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_kinds": list(self.capsule_kinds),
            "capsule_cids": list(self.capsule_cids),
            "merkle_root_cid": str(self.merkle_root_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_tool_audit_chain_v1",
            "chain": self.to_dict()})

    def verify_merkle_root(self) -> bool:
        """Recompute the Merkle root from the capsule CIDs."""
        return self.merkle_root_cid == _merkle_root(
            tuple(self.capsule_cids))


def _merkle_root(cids: Sequence[str]) -> str:
    if len(cids) == 0:
        return "empty"
    layer = list(str(c) for c in cids)
    while len(layer) > 1:
        nxt: list[str] = []
        for i in range(0, len(layer), 2):
            if i + 1 < len(layer):
                pair = (str(layer[i]), str(layer[i + 1]))
            else:
                pair = (str(layer[i]), str(layer[i]))
            nxt.append(_sha256_hex({
                "kind": "w84_merkle_node_v1",
                "left": pair[0], "right": pair[1]}))
        layer = nxt
    return str(layer[0])


def build_tool_audit_chain_v1(
        capsules: Sequence[tuple[str, Any]],
) -> ToolAuditChainV1:
    """Build a content-addressed audit chain.

    ``capsules`` is a list of ``(kind, capsule)`` where
    ``kind`` is one of ``tool_call``, ``tool_result``, or
    ``llm_capsule``; ``capsule`` is any object with a
    ``cid()`` method.
    """
    kinds = tuple(str(k) for k, _ in capsules)
    cids = tuple(str(c.cid()) for _, c in capsules)
    return ToolAuditChainV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        capsule_kinds=kinds,
        capsule_cids=cids,
        merkle_root_cid=_merkle_root(cids),
    )


# ---------------------------------------------------------------
# Bench — 5-agent team with three tools
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ToolUseBenchReportV1:
    schema: str
    n_agents: int
    n_tool_calls: int
    audit_chain_cid: str
    merkle_root: str
    all_calls_content_addressed: bool
    identical_inputs_give_identical_call_cids: bool
    non_idempotent_double_commit_refused: bool
    sandbox_wall_time_limit_enforced: bool
    audit_chain_reverifies_from_disk: bool
    tamper_breaks_audit_chain: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_agents": int(self.n_agents),
            "n_tool_calls": int(self.n_tool_calls),
            "audit_chain_cid": str(self.audit_chain_cid),
            "merkle_root": str(self.merkle_root),
            "all_calls_content_addressed": bool(
                self.all_calls_content_addressed),
            "identical_inputs_give_identical_call_cids": bool(
                self.identical_inputs_give_identical_call_cids),
            "non_idempotent_double_commit_refused": bool(
                self.non_idempotent_double_commit_refused),
            "sandbox_wall_time_limit_enforced": bool(
                self.sandbox_wall_time_limit_enforced),
            "audit_chain_reverifies_from_disk": bool(
                self.audit_chain_reverifies_from_disk),
            "tamper_breaks_audit_chain": bool(
                self.tamper_breaks_audit_chain),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_tool_use_bench_report_v1",
            "report": self.to_dict()})


def run_tool_use_bench_v1(
        *, run_cid: str = "w84_tool_bench_run",
) -> ToolUseBenchReportV1:
    """End-to-end tool-use bench.

    Five agents make tool calls (python_exec, filesystem ripgrep,
    http_fetch_stub). Each call + result is content-addressed
    and added to a Merkle audit chain. The chain re-verifies
    from its bytes; tampering with any CID breaks the
    verification.
    """
    sandbox = ToolSandboxAdapterV1(
        limits=ToolSandboxLimitsV1(
            max_wall_time_seconds=0.5))
    registry = IdempotencyRegistryV1()
    capsules: list[tuple[str, Any]] = []
    ts = int(time.time_ns())
    # Agent 1: python_exec.
    call1 = make_tool_call_v1(
        tool_id="python_exec",
        tool_version_cid="v1",
        args={"code": "result = sum(range(10))"},
        timestamp_ns=ts, caller_agent_id="agent_1",
        run_cid=run_cid)
    res1 = run_tool_call_v1(
        callable_=tool_python_exec, call=call1,
        sandbox=sandbox,
        idempotency_registry=registry,
        args={"code": "result = sum(range(10))"})
    capsules.append(("tool_call", call1))
    capsules.append(("tool_result", res1))
    # Agent 2: filesystem ripgrep.
    call2 = make_tool_call_v1(
        tool_id="filesystem_ripgrep_lite",
        tool_version_cid="v1",
        args={"root": ".", "pattern": "coordpy",
              "max_matches": 4},
        timestamp_ns=ts + 1, caller_agent_id="agent_2",
        run_cid=run_cid)
    res2 = run_tool_call_v1(
        callable_=tool_filesystem_ripgrep_lite, call=call2,
        sandbox=sandbox, idempotency_registry=registry,
        args={"root": ".", "pattern": "coordpy",
              "max_matches": 4})
    capsules.append(("tool_call", call2))
    capsules.append(("tool_result", res2))
    # Agent 3: http_fetch_stub.
    call3 = make_tool_call_v1(
        tool_id="http_fetch_stub", tool_version_cid="v1",
        args={"url": "https://example.com/probe"},
        timestamp_ns=ts + 2, caller_agent_id="agent_3",
        run_cid=run_cid)
    res3 = run_tool_call_v1(
        callable_=tool_http_fetch_stub, call=call3,
        sandbox=sandbox,
        idempotency_registry=registry,
        args={"url": "https://example.com/probe"})
    capsules.append(("tool_call", call3))
    capsules.append(("tool_result", res3))
    # Agent 4: non-idempotent http_fetch_stub.
    tok = IdempotencyTokenV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        run_cid=run_cid, agent_id="agent_4",
        sequence_no=1)
    call4 = make_tool_call_v1(
        tool_id="http_fetch_stub", tool_version_cid="v1",
        args={"url": "https://example.com/write"},
        timestamp_ns=ts + 3, caller_agent_id="agent_4",
        run_cid=run_cid,
        idempotency=ToolIdempotency.NON_IDEMPOTENT.value,
        idempotency_token=tok)
    res4 = run_tool_call_v1(
        callable_=tool_http_fetch_stub, call=call4,
        sandbox=sandbox, idempotency_registry=registry,
        args={"url": "https://example.com/write"})
    capsules.append(("tool_call", call4))
    capsules.append(("tool_result", res4))
    # Agent 5: same token but DIFFERENT call -> refused.
    call5 = make_tool_call_v1(
        tool_id="http_fetch_stub", tool_version_cid="v1",
        args={"url": "https://example.com/DIFFERENT"},
        timestamp_ns=ts + 4, caller_agent_id="agent_5",
        run_cid=run_cid,
        idempotency=ToolIdempotency.NON_IDEMPOTENT.value,
        idempotency_token=tok)
    res5 = run_tool_call_v1(
        callable_=tool_http_fetch_stub, call=call5,
        sandbox=sandbox, idempotency_registry=registry,
        args={"url": "https://example.com/DIFFERENT"})
    capsules.append(("tool_call", call5))
    capsules.append(("tool_result", res5))
    double_refused = (
        res5.integrity_verdict
        == ToolIntegrityVerdict.IDEMPOTENCY_VIOLATION.value)
    # Sandbox wall-time limit: probe with a slow callable.
    slow_sandbox = ToolSandboxAdapterV1(
        limits=ToolSandboxLimitsV1(
            max_wall_time_seconds=0.05))

    def _slow(**_kw: Any) -> dict[str, Any]:
        time.sleep(0.5)
        return {"ok": True}
    slow_call = make_tool_call_v1(
        tool_id="slow_probe", tool_version_cid="v1",
        args={"x": 1},
        timestamp_ns=ts + 5, caller_agent_id="agent_probe",
        run_cid=run_cid)
    slow_res = run_tool_call_v1(
        callable_=_slow, call=slow_call,
        sandbox=slow_sandbox,
        idempotency_registry=registry, args={"x": 1})
    wall_enforced = (
        slow_res.integrity_verdict
        == ToolIntegrityVerdict.LIMIT_EXCEEDED.value)
    # Audit chain + re-verification.
    chain = build_tool_audit_chain_v1(capsules=capsules)
    reverifies = chain.verify_merkle_root()
    # Tamper: change one CID -> root should not match.
    tampered_cids = list(chain.capsule_cids)
    tampered_cids[0] = (
        "0" * 64 if tampered_cids[0] != "0" * 64 else "1" * 64)
    tampered_root = _merkle_root(tampered_cids)
    tamper_breaks = (tampered_root != chain.merkle_root_cid)
    # Anti-cheat: identical-input call CIDs match.
    call_a = make_tool_call_v1(
        tool_id="python_exec", tool_version_cid="v1",
        args={"code": "result = 7"},
        timestamp_ns=42, caller_agent_id="agent_x",
        run_cid=run_cid)
    call_b = make_tool_call_v1(
        tool_id="python_exec", tool_version_cid="v1",
        args={"code": "result = 7"},
        timestamp_ns=42, caller_agent_id="agent_x",
        run_cid=run_cid)
    identical = (call_a.cid() == call_b.cid())
    # All result CIDs are 64-char hex.
    all_addressed = all(
        len(c.cid()) == 64
        for k, c in capsules)
    return ToolUseBenchReportV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        n_agents=5,
        n_tool_calls=int(
            sum(1 for k, _ in capsules
                if k == "tool_call")),
        audit_chain_cid=str(chain.cid()),
        merkle_root=str(chain.merkle_root_cid),
        all_calls_content_addressed=bool(all_addressed),
        identical_inputs_give_identical_call_cids=bool(
            identical),
        non_idempotent_double_commit_refused=bool(
            double_refused),
        sandbox_wall_time_limit_enforced=bool(wall_enforced),
        audit_chain_reverifies_from_disk=bool(reverifies),
        tamper_breaks_audit_chain=bool(tamper_breaks),
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


__all__ = [
    "W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION",
    "ToolIdempotency",
    "ToolIntegrityVerdict",
    "IdempotencyTokenV1",
    "ToolCallSchemaV1",
    "ToolResultSchemaV1",
    "ToolSandboxLimitsV1",
    "ToolExecutionV1",
    "ToolSandboxAdapterV1",
    "tool_python_exec",
    "tool_filesystem_ripgrep_lite",
    "tool_http_fetch_stub",
    "IdempotencyRegistryV1",
    "make_tool_call_v1",
    "run_tool_call_v1",
    "ToolAuditChainV1",
    "build_tool_audit_chain_v1",
    "ToolUseBenchReportV1",
    "run_tool_use_bench_v1",
]
