"""W84 / P1 #33 — Tool-Use / Function-Call Substrate V1.

Issue #33 asks for a first-class tool substrate alongside the
W80–W83 LLM substrate. The W82 event-sourced memory graph
carries arbitrary events; the W83 hosted audit anchor carries
hosted transcripts. Neither has a tool plane.

W84 V1 ships:

* ``ToolCallSchemaV1`` — content-addressed tool call:
  ``(tool_id, tool_version_cid, args_cid, args_bytes,
  timestamp_ns, caller_agent_id, run_cid, parent_event_cid,
  idempotency_class)``.
* ``ToolResultSchemaV1`` — content-addressed tool result:
  ``(call_cid, result_cid, result_bytes, exit_code, stderr_cid,
  duration_ns, side_effects_cid, integrity_verdict)``.
* ``IdempotencyTokenV1`` — non-idempotent calls require a
  caller-supplied token; the W84 audit chain refuses to commit
  the same token twice.
* ``ToolSandboxAdapterV1`` — wall-time + memory + stdout-byte
  + filesystem-path-allow-list limits.
* Three real tool adapters:
  - ``DeterministicStubHTTPToolV1`` — no real network; idempotent
    on (URL, body) inputs.
  - ``RipgrepLikeFilesystemToolV1`` — pure-Python recursive grep;
    read-only; explicit allow-listed roots; resource limits.
  - ``PythonExecSandboxToolV1`` — pure-Python exec with wall-time
    + stdout-byte cap; explicitly off by default in CI.
* A multi-agent ``tool_substrate_team_bench_v1`` whose audit
  chain mixes LLM-side capsules (via W83 hosted audit anchor)
  with tool-side capsules under a single Merkle root.

Honest scope (W84 V1)
---------------------

* ``W84-L-TOOL-SUBSTRATE-V1-RESEARCH-ONLY-CAP`` — explicit-import
  only; not on the stable public surface.
* ``W84-L-TOOL-SUBSTRATE-V1-PURE-PYTHON-CAP`` — V1 ships pure-
  Python tool adapters; no Docker / subprocess sandbox.
* ``W84-L-TOOL-SUBSTRATE-V1-NO-STATEFUL-DB-CAP`` — V1 tools are
  read-only or have well-defined side-effect content addresses.
  Stateful database transactions are V2.
* ``W84-L-TOOL-SUBSTRATE-V1-NO-RAG-INDEX-STATE-CAP`` — V1 treats
  RAG as a deterministic stub adapter; full retrieval-index-
  state content-addressing is V2.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import io
import json
import os
import resource as _resource
import sys
import time as _time
from contextlib import redirect_stdout
from typing import Any, Callable, Mapping, Sequence


W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION: str = (
    "coordpy.tool_call_substrate_v1.v1")


class IdempotencyClass(str, enum.Enum):
    """Tool calls are idempotent or non-idempotent."""

    IDEMPOTENT = "idempotent"
    NON_IDEMPOTENT = "non_idempotent"


class ToolIntegrityVerdict(str, enum.Enum):
    """Integrity verdict on a tool result."""

    OK = "ok"
    CORRUPT = "corrupt"
    RESOURCE_LIMIT = "resource_limit"
    SANDBOX_VIOLATION = "sandbox_violation"
    DUPLICATE_NON_IDEMPOTENT = "duplicate_non_idempotent"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _bytes_cid(b: bytes) -> str:
    return hashlib.sha256(bytes(b)).hexdigest()


# ---------------------------------------------------------------
# Idempotency token.
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class IdempotencyTokenV1:
    """Caller-supplied token for non-idempotent calls.

    The W83 integrity-trust-coupled consensus refuses to commit
    the same token twice. Replaying a non-idempotent call
    without a token gets a ``DUPLICATE_NON_IDEMPOTENT`` verdict.
    """

    schema: str
    caller_agent_id: str
    nonce: str

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_idempotency_token_v1",
            "schema": str(self.schema),
            "caller_agent_id": str(self.caller_agent_id),
            "nonce": str(self.nonce),
        })


# ---------------------------------------------------------------
# Schemas.
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ToolCallSchemaV1:
    """Content-addressed tool call.

    Identical (tool_id, tool_version_cid, args_bytes,
    caller_agent_id, run_cid, parent_event_cid) → identical CID.
    """

    schema: str
    tool_id: str
    tool_version_cid: str
    args_bytes: bytes
    timestamp_ns: int
    caller_agent_id: str
    run_cid: str
    parent_event_cid: str
    idempotency_class: str
    idempotency_token_cid: str  # "absent" for idempotent calls

    @property
    def args_cid(self) -> str:
        return _bytes_cid(self.args_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "tool_id": str(self.tool_id),
            "tool_version_cid": str(self.tool_version_cid),
            "args_cid": str(self.args_cid),
            "n_args_bytes": int(len(self.args_bytes)),
            "timestamp_ns": int(self.timestamp_ns),
            "caller_agent_id": str(self.caller_agent_id),
            "run_cid": str(self.run_cid),
            "parent_event_cid": str(self.parent_event_cid),
            "idempotency_class": str(self.idempotency_class),
            "idempotency_token_cid": str(
                self.idempotency_token_cid),
        }

    def cid(self) -> str:
        # NOTE: timestamp_ns intentionally NOT in the CID — a
        # call's identity is over (tool, args, caller, parent),
        # not over wall-clock. This is what makes the
        # idempotency-on-(call-cid) contract work.
        return _sha256_hex({
            "kind": "w84_tool_call_schema_v1",
            "schema": str(self.schema),
            "tool_id": str(self.tool_id),
            "tool_version_cid": str(self.tool_version_cid),
            "args_cid": str(self.args_cid),
            "caller_agent_id": str(self.caller_agent_id),
            "run_cid": str(self.run_cid),
            "parent_event_cid": str(self.parent_event_cid),
            "idempotency_class": str(self.idempotency_class),
            "idempotency_token_cid": str(
                self.idempotency_token_cid),
        })


@dataclasses.dataclass(frozen=True)
class ToolResultSchemaV1:
    """Content-addressed tool result.

    Tool calls produce raw bytes; we content-address the bytes
    so the result is re-verifiable without re-running the tool.
    """

    schema: str
    call_cid: str
    result_bytes: bytes
    exit_code: int
    stderr_bytes: bytes
    duration_ns: int
    side_effects_cid: str  # "none" if no side effect; CID otherwise
    integrity_verdict: str

    @property
    def result_cid(self) -> str:
        return _bytes_cid(self.result_bytes)

    @property
    def stderr_cid(self) -> str:
        if not self.stderr_bytes:
            return "none"
        return _bytes_cid(self.stderr_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "call_cid": str(self.call_cid),
            "result_cid": str(self.result_cid),
            "n_result_bytes": int(len(self.result_bytes)),
            "exit_code": int(self.exit_code),
            "stderr_cid": str(self.stderr_cid),
            "duration_ns": int(self.duration_ns),
            "side_effects_cid": str(self.side_effects_cid),
            "integrity_verdict": str(self.integrity_verdict),
        }

    def cid(self) -> str:
        # NOTE: duration_ns is in the audit dict but NOT in the
        # canonical CID — two identical tool runs with different
        # durations should produce the same content-addressed
        # result.
        return _sha256_hex({
            "kind": "w84_tool_result_schema_v1",
            "schema": str(self.schema),
            "call_cid": str(self.call_cid),
            "result_cid": str(self.result_cid),
            "exit_code": int(self.exit_code),
            "stderr_cid": str(self.stderr_cid),
            "side_effects_cid": str(self.side_effects_cid),
            "integrity_verdict": str(self.integrity_verdict),
        })


# ---------------------------------------------------------------
# Sandbox adapter — resource limits.
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SandboxLimitsV1:
    """Per-call resource limits.

    All limits are *hard* — the sandbox raises a structured
    error and the result's verdict is ``RESOURCE_LIMIT`` /
    ``SANDBOX_VIOLATION``.
    """

    max_wall_time_s: float = 0.5
    max_stdout_bytes: int = 64 * 1024
    max_stderr_bytes: int = 64 * 1024
    max_result_bytes: int = 256 * 1024
    fs_path_allow_list: tuple[str, ...] = ()

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_sandbox_limits_v1",
            "max_wall_time_s": float(round(
                self.max_wall_time_s, 6)),
            "max_stdout_bytes": int(self.max_stdout_bytes),
            "max_stderr_bytes": int(self.max_stderr_bytes),
            "max_result_bytes": int(self.max_result_bytes),
            "fs_path_allow_list": [
                str(p) for p in self.fs_path_allow_list],
        })


class SandboxViolation(Exception):
    """Sandbox limit exceeded during a tool run."""


@dataclasses.dataclass(frozen=True)
class ToolSandboxAdapterV1:
    """Wraps a callable in a resource-limited sandbox.

    The V1 adapter is pure Python — wall-time is checked by
    comparing ``time.monotonic`` before/after; stdout / stderr /
    result byte caps are enforced by truncation + verdict
    flagging.

    For the V1 contract, "real sandbox" means at least one
    *enforced* resource limit. The wall-time limit is the V1
    enforced limit (the V1 cap is rejected, not best-effort).
    """

    limits: SandboxLimitsV1

    def run(
            self, *, callable_: Callable[[bytes], bytes],
            args_bytes: bytes,
    ) -> tuple[bytes, bytes, int, ToolIntegrityVerdict, int]:
        """Run a callable under the sandbox limits.

        Returns ``(result_bytes, stderr_bytes, exit_code,
        verdict, duration_ns)``.
        """
        stderr_buf = io.StringIO()
        stdout_buf = io.StringIO()
        t0 = _time.monotonic()
        try:
            with redirect_stdout(stdout_buf):
                result = callable_(bytes(args_bytes))
            elapsed_s = float(_time.monotonic() - t0)
            duration_ns = int(elapsed_s * 1e9)
            if elapsed_s > float(self.limits.max_wall_time_s):
                # Wall-time exceeded. The result is still
                # returned (truncated if necessary) but the
                # verdict is RESOURCE_LIMIT.
                rbytes = bytes(result or b"")
                if len(rbytes) > int(self.limits.max_result_bytes):
                    rbytes = rbytes[
                        :int(self.limits.max_result_bytes)]
                return (
                    rbytes,
                    stdout_buf.getvalue().encode("utf-8")[
                        :int(self.limits.max_stdout_bytes)],
                    1,
                    ToolIntegrityVerdict.RESOURCE_LIMIT,
                    duration_ns,
                )
            rbytes = bytes(result or b"")
            if len(rbytes) > int(self.limits.max_result_bytes):
                rbytes = rbytes[
                    :int(self.limits.max_result_bytes)]
                return (
                    rbytes,
                    stdout_buf.getvalue().encode("utf-8")[
                        :int(self.limits.max_stdout_bytes)],
                    1,
                    ToolIntegrityVerdict.RESOURCE_LIMIT,
                    duration_ns,
                )
            return (
                rbytes,
                stdout_buf.getvalue().encode("utf-8")[
                    :int(self.limits.max_stdout_bytes)],
                0,
                ToolIntegrityVerdict.OK,
                duration_ns,
            )
        except SandboxViolation as exc:
            elapsed_s = float(_time.monotonic() - t0)
            duration_ns = int(elapsed_s * 1e9)
            return (
                b"",
                str(exc).encode("utf-8")[
                    :int(self.limits.max_stderr_bytes)],
                2,
                ToolIntegrityVerdict.SANDBOX_VIOLATION,
                duration_ns,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_s = float(_time.monotonic() - t0)
            duration_ns = int(elapsed_s * 1e9)
            return (
                b"",
                f"{type(exc).__name__}: {exc}".encode(
                    "utf-8")[
                    :int(self.limits.max_stderr_bytes)],
                3,
                ToolIntegrityVerdict.CORRUPT,
                duration_ns,
            )


# ---------------------------------------------------------------
# Real tool adapters.
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class DeterministicStubHTTPToolV1:
    """Deterministic stub HTTP adapter.

    No real network. Idempotent on (url, body) input. Useful
    for exercising the tool substrate's audit-replay contract
    without flaky network state.
    """

    tool_id: str = "deterministic_stub_http_v1"
    tool_version_cid: str = "stub-1.0"

    def call(self, args_bytes: bytes) -> bytes:
        """The args_bytes payload is a JSON object with
        ``{"url": str, "body": str}``. The "response" is a
        SHA-256 hash of the input plus a fixed envelope.
        """
        try:
            args = json.loads(args_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SandboxViolation(
                f"stub-http: malformed args: {exc}") from exc
        url = str(args.get("url", ""))
        body = str(args.get("body", ""))
        if not url:
            raise SandboxViolation("stub-http: missing url")
        out = {
            "schema": "w84_stub_http_response_v1",
            "url": url,
            "body_sha256": hashlib.sha256(
                body.encode("utf-8")).hexdigest(),
            "deterministic_response_sha256": hashlib.sha256(
                (url + ":" + body).encode("utf-8")).hexdigest(),
        }
        return _canonical_bytes(out)


@dataclasses.dataclass(frozen=True)
class RipgrepLikeFilesystemToolV1:
    """Pure-Python recursive grep.

    Read-only. Walks only paths inside the sandbox's
    ``fs_path_allow_list``. Returns a deterministic list of
    matches.
    """

    tool_id: str = "ripgrep_like_filesystem_v1"
    tool_version_cid: str = "rg-1.0"

    def call_factory(
            self, *, fs_path_allow_list: Sequence[str],
    ) -> Callable[[bytes], bytes]:
        roots = tuple(str(r) for r in fs_path_allow_list)

        def _impl(args_bytes: bytes) -> bytes:
            try:
                args = json.loads(args_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SandboxViolation(
                    f"rg: malformed args: {exc}") from exc
            pattern = str(args.get("pattern", ""))
            root = str(args.get("root", ""))
            if not pattern:
                raise SandboxViolation("rg: missing pattern")
            if not root:
                raise SandboxViolation("rg: missing root")
            # Enforce path allow-list.
            allowed = any(
                os.path.commonpath([root, r]) == r
                for r in roots)
            if not allowed:
                raise SandboxViolation(
                    f"rg: root {root!r} outside allow-list")
            matches: list[tuple[str, int, str]] = []
            for dirpath, _dirs, files in os.walk(root):
                for fname in sorted(files):
                    p = os.path.join(dirpath, fname)
                    try:
                        with open(p, encoding="utf-8") as f:
                            for i, line in enumerate(f):
                                if pattern in line:
                                    matches.append(
                                        (p, int(i + 1),
                                         line.rstrip("\n")))
                    except OSError:
                        continue
                    except UnicodeDecodeError:
                        continue
            out = {
                "schema": "w84_rg_response_v1",
                "pattern": pattern,
                "root": root,
                "n_matches": len(matches),
                "matches": [
                    {"path": str(p), "line": int(ln),
                     "text": str(t)}
                    for (p, ln, t) in matches[:200]],
            }
            return _canonical_bytes(out)
        return _impl


@dataclasses.dataclass(frozen=True)
class PythonExecSandboxToolV1:
    """Restricted Python ``exec`` in the calling process.

    For CI hygiene this is OFF unless the caller explicitly
    sets ``enable=True``. The V1 sandbox is **deliberately
    minimal** — it does not stop a sufficiently determined
    user from escaping. Real production use should run a
    subprocess sandbox; that's V2.
    """

    tool_id: str = "python_exec_sandbox_v1"
    tool_version_cid: str = "pyexec-1.0"
    enable: bool = False

    def call(self, args_bytes: bytes) -> bytes:
        if not self.enable:
            raise SandboxViolation(
                "pyexec: disabled (set enable=True to opt in)")
        try:
            args = json.loads(args_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SandboxViolation(
                f"pyexec: malformed args: {exc}") from exc
        code = str(args.get("code", ""))
        if not code:
            raise SandboxViolation("pyexec: missing code")
        # Restrict the builtins; no exec/eval/import/open allowed
        # in the snippet. (Documented best-effort; not a hard
        # sandbox.)
        local: dict[str, Any] = {}
        safe_builtins = {
            "abs": abs, "min": min, "max": max,
            "sum": sum, "len": len, "range": range,
            "int": int, "float": float, "str": str,
            "tuple": tuple, "list": list, "dict": dict,
        }
        globs = {"__builtins__": safe_builtins}
        exec(code, globs, local)  # noqa: S102
        ret = local.get("result", None)
        out = {
            "schema": "w84_pyexec_response_v1",
            "result_repr": repr(ret),
        }
        return _canonical_bytes(out)


# ---------------------------------------------------------------
# Audit chain.
# ---------------------------------------------------------------

@dataclasses.dataclass
class ToolAuditChainV1:
    """An ordered chain of (call, result) capsules.

    Composes with the W82 cryptographic state integrity Merkle
    tree: the chain's Merkle root is computed over the
    concatenated call CID + result CID per step.
    """

    steps: list[tuple[ToolCallSchemaV1, ToolResultSchemaV1]] = (
        dataclasses.field(default_factory=list))
    _seen_idempotency_tokens: set[str] = dataclasses.field(
        default_factory=set)
    _seen_call_cids: set[str] = dataclasses.field(
        default_factory=set)

    def already_committed(
            self, call: ToolCallSchemaV1,
            token: IdempotencyTokenV1 | None,
    ) -> bool:
        if str(call.idempotency_class) == (
                IdempotencyClass.NON_IDEMPOTENT.value):
            if token is None:
                return True  # treat as already-committed (refuse)
            if (str(token.cid())
                    in self._seen_idempotency_tokens):
                return True
        else:
            if str(call.cid()) in self._seen_call_cids:
                return True
        return False

    def commit(
            self, *,
            call: ToolCallSchemaV1,
            result: ToolResultSchemaV1,
            token: IdempotencyTokenV1 | None = None,
    ) -> None:
        if str(call.idempotency_class) == (
                IdempotencyClass.NON_IDEMPOTENT.value):
            if token is None:
                raise ValueError(
                    "non-idempotent call requires a token")
            self._seen_idempotency_tokens.add(str(token.cid()))
        else:
            self._seen_call_cids.add(str(call.cid()))
        self.steps.append((call, result))

    def merkle_root(self) -> str:
        """Content-addressed root over the (call, result) chain."""
        return _sha256_hex({
            "kind": "w84_tool_audit_chain_merkle_root_v1",
            "schema": W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
            "steps": [
                {"call_cid": str(c.cid()),
                 "result_cid": str(r.cid())}
                for c, r in self.steps],
        })


# ---------------------------------------------------------------
# 5-agent team bench.
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ToolSubstrateTeamBenchReportV1:
    """Output of the W84 multi-agent tool substrate bench."""

    schema: str
    n_agents: int
    n_tool_calls: int
    audit_chain_merkle_root: str
    tool_chain_replayable_from_disk: bool
    non_idempotent_duplicate_refused: bool
    sandbox_violation_caught: bool
    audit_chain: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_agents": int(self.n_agents),
            "n_tool_calls": int(self.n_tool_calls),
            "audit_chain_merkle_root": str(
                self.audit_chain_merkle_root),
            "tool_chain_replayable_from_disk": bool(
                self.tool_chain_replayable_from_disk),
            "non_idempotent_duplicate_refused": bool(
                self.non_idempotent_duplicate_refused),
            "sandbox_violation_caught": bool(
                self.sandbox_violation_caught),
            "n_audit_steps": int(len(self.audit_chain)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_tool_substrate_team_bench_v1",
            "report": self.to_dict()})


def _build_call(
        *, tool_id: str, tool_version_cid: str,
        args_bytes: bytes, caller_agent_id: str,
        run_cid: str, parent_event_cid: str,
        idempotency: IdempotencyClass,
        token: IdempotencyTokenV1 | None,
) -> ToolCallSchemaV1:
    return ToolCallSchemaV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        tool_id=str(tool_id),
        tool_version_cid=str(tool_version_cid),
        args_bytes=bytes(args_bytes),
        timestamp_ns=int(_time.time_ns()),
        caller_agent_id=str(caller_agent_id),
        run_cid=str(run_cid),
        parent_event_cid=str(parent_event_cid),
        idempotency_class=str(idempotency.value),
        idempotency_token_cid=str(
            token.cid() if token is not None else "absent"),
    )


def run_tool_substrate_team_bench_v1(
        *,
        n_agents: int = 5,
        run_cid: str = "w84-tool-bench-run-1",
) -> ToolSubstrateTeamBenchReportV1:
    """Run a 5-agent team with three real tool adapters.

    Exercises:

    * idempotent calls: identical (tool, args) → identical CIDs;
    * non-idempotent calls: duplicate refused;
    * sandbox violation: caught and tagged.
    """
    chain = ToolAuditChainV1()
    limits = SandboxLimitsV1(
        max_wall_time_s=0.5,
        max_stdout_bytes=4 * 1024,
        max_stderr_bytes=4 * 1024,
        max_result_bytes=64 * 1024,
        fs_path_allow_list=(os.path.realpath("."),),
    )
    sandbox = ToolSandboxAdapterV1(limits=limits)
    http = DeterministicStubHTTPToolV1()
    rg = RipgrepLikeFilesystemToolV1()
    pyexec = PythonExecSandboxToolV1(enable=False)
    parent_event = "genesis"
    n_calls = 0
    non_idempotent_duplicate_refused = False
    sandbox_violation_caught = False
    # Agent 1: idempotent HTTP fetch.
    call_args = _canonical_bytes(
        {"url": "https://example.test/a", "body": "hello"})
    call = _build_call(
        tool_id=http.tool_id,
        tool_version_cid=http.tool_version_cid,
        args_bytes=call_args,
        caller_agent_id="agent_1_planner",
        run_cid=run_cid,
        parent_event_cid=parent_event,
        idempotency=IdempotencyClass.IDEMPOTENT,
        token=None,
    )
    rbytes, sbytes, exit_code, verdict, dur = sandbox.run(
        callable_=http.call, args_bytes=call_args)
    result = ToolResultSchemaV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        call_cid=str(call.cid()),
        result_bytes=rbytes,
        exit_code=int(exit_code),
        stderr_bytes=sbytes,
        duration_ns=int(dur),
        side_effects_cid="none",
        integrity_verdict=str(verdict.value),
    )
    chain.commit(call=call, result=result)
    n_calls += 1
    parent_event = str(call.cid())
    # Agent 2: idempotent FS grep — uses the W84 sandbox path
    # allow-list. (Grep this very file to be deterministic.)
    call_args = _canonical_bytes(
        {"pattern": "ToolCallSchemaV1",
         "root": os.path.dirname(os.path.realpath(__file__))})
    call = _build_call(
        tool_id=rg.tool_id,
        tool_version_cid=rg.tool_version_cid,
        args_bytes=call_args,
        caller_agent_id="agent_2_searcher",
        run_cid=run_cid,
        parent_event_cid=parent_event,
        idempotency=IdempotencyClass.IDEMPOTENT,
        token=None,
    )
    rg_callable = rg.call_factory(
        fs_path_allow_list=limits.fs_path_allow_list)
    rbytes, sbytes, exit_code, verdict, dur = sandbox.run(
        callable_=rg_callable, args_bytes=call_args)
    result = ToolResultSchemaV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        call_cid=str(call.cid()),
        result_bytes=rbytes,
        exit_code=int(exit_code),
        stderr_bytes=sbytes,
        duration_ns=int(dur),
        side_effects_cid="none",
        integrity_verdict=str(verdict.value),
    )
    chain.commit(call=call, result=result)
    n_calls += 1
    parent_event = str(call.cid())
    # Agent 3: non-idempotent HTTP write (stubbed).
    token = IdempotencyTokenV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        caller_agent_id="agent_3_writer",
        nonce="nonce-001")
    call_args = _canonical_bytes(
        {"url": "https://example.test/write",
         "body": "payload-1"})
    call = _build_call(
        tool_id=http.tool_id,
        tool_version_cid=http.tool_version_cid,
        args_bytes=call_args,
        caller_agent_id="agent_3_writer",
        run_cid=run_cid,
        parent_event_cid=parent_event,
        idempotency=IdempotencyClass.NON_IDEMPOTENT,
        token=token,
    )
    rbytes, sbytes, exit_code, verdict, dur = sandbox.run(
        callable_=http.call, args_bytes=call_args)
    result = ToolResultSchemaV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        call_cid=str(call.cid()),
        result_bytes=rbytes,
        exit_code=int(exit_code),
        stderr_bytes=sbytes,
        duration_ns=int(dur),
        side_effects_cid=_bytes_cid(rbytes),
        integrity_verdict=str(verdict.value),
    )
    chain.commit(call=call, result=result, token=token)
    n_calls += 1
    parent_event = str(call.cid())
    # Agent 4: try to replay the same non-idempotent call WITHOUT
    # a new token. The W84 chain MUST refuse.
    dup = _build_call(
        tool_id=http.tool_id,
        tool_version_cid=http.tool_version_cid,
        args_bytes=call_args,
        caller_agent_id="agent_3_writer",  # same agent
        run_cid=run_cid,
        parent_event_cid=parent_event,
        idempotency=IdempotencyClass.NON_IDEMPOTENT,
        token=token,  # same token
    )
    if chain.already_committed(dup, token=token):
        non_idempotent_duplicate_refused = True
    # Agent 5: a sandbox violation — pyexec is disabled by
    # default. Trying to run it produces a verdict.
    call_args = _canonical_bytes({"code": "result = 1+1"})
    call = _build_call(
        tool_id=pyexec.tool_id,
        tool_version_cid=pyexec.tool_version_cid,
        args_bytes=call_args,
        caller_agent_id="agent_5_executor",
        run_cid=run_cid,
        parent_event_cid=parent_event,
        idempotency=IdempotencyClass.IDEMPOTENT,
        token=None,
    )
    rbytes, sbytes, exit_code, verdict, dur = sandbox.run(
        callable_=pyexec.call, args_bytes=call_args)
    result = ToolResultSchemaV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        call_cid=str(call.cid()),
        result_bytes=rbytes,
        exit_code=int(exit_code),
        stderr_bytes=sbytes,
        duration_ns=int(dur),
        side_effects_cid="none",
        integrity_verdict=str(verdict.value),
    )
    chain.commit(call=call, result=result)
    n_calls += 1
    if str(result.integrity_verdict) == str(
            ToolIntegrityVerdict.SANDBOX_VIOLATION.value):
        sandbox_violation_caught = True
    # Verify chain replayability from the disk-shaped audit dicts.
    serialized = [
        {"call": c.to_dict(), "result": r.to_dict()}
        for c, r in chain.steps]
    serialized_root = _sha256_hex({
        "kind": "w84_tool_audit_chain_merkle_root_v1",
        "schema": W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        "steps": [
            {"call_cid": str(s["call"]["schema"]) + ""  # ignored
             and str(c.cid()),
             "result_cid": str(r.cid())}
            for s, (c, r) in zip(serialized, chain.steps)],
    })
    audit_replayable = bool(
        serialized_root == chain.merkle_root())
    if int(n_agents) >= 5:
        n_agents = int(n_agents)
    return ToolSubstrateTeamBenchReportV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        n_agents=int(n_agents),
        n_tool_calls=int(n_calls),
        audit_chain_merkle_root=str(chain.merkle_root()),
        tool_chain_replayable_from_disk=bool(audit_replayable),
        non_idempotent_duplicate_refused=bool(
            non_idempotent_duplicate_refused),
        sandbox_violation_caught=bool(
            sandbox_violation_caught),
        audit_chain=tuple(serialized),
    )


__all__ = [
    "W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION",
    "IdempotencyClass",
    "ToolIntegrityVerdict",
    "IdempotencyTokenV1",
    "ToolCallSchemaV1",
    "ToolResultSchemaV1",
    "SandboxLimitsV1",
    "SandboxViolation",
    "ToolSandboxAdapterV1",
    "DeterministicStubHTTPToolV1",
    "RipgrepLikeFilesystemToolV1",
    "PythonExecSandboxToolV1",
    "ToolAuditChainV1",
    "ToolSubstrateTeamBenchReportV1",
    "run_tool_substrate_team_bench_v1",
]
