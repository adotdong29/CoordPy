"""W84 / P1 #33 — Tool substrate V1 tests."""

from __future__ import annotations

import os

import pytest

from coordpy.tool_call_substrate_v1 import (
    DeterministicStubHTTPToolV1,
    IdempotencyClass,
    IdempotencyTokenV1,
    PythonExecSandboxToolV1,
    RipgrepLikeFilesystemToolV1,
    SandboxLimitsV1,
    SandboxViolation,
    ToolAuditChainV1,
    ToolCallSchemaV1,
    ToolIntegrityVerdict,
    ToolResultSchemaV1,
    ToolSandboxAdapterV1,
    W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
    run_tool_substrate_team_bench_v1,
)


# ---------------------------------------------------------------
# Schema content-addressing.
# ---------------------------------------------------------------

def test_w84_identical_inputs_produce_identical_call_cids():
    c1 = ToolCallSchemaV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        tool_id="t", tool_version_cid="v1",
        args_bytes=b"args-1",
        timestamp_ns=1,  # NOT part of CID
        caller_agent_id="a", run_cid="r",
        parent_event_cid="p",
        idempotency_class=IdempotencyClass.IDEMPOTENT.value,
        idempotency_token_cid="absent",
    )
    c2 = ToolCallSchemaV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        tool_id="t", tool_version_cid="v1",
        args_bytes=b"args-1",
        timestamp_ns=2,  # different wall clock
        caller_agent_id="a", run_cid="r",
        parent_event_cid="p",
        idempotency_class=IdempotencyClass.IDEMPOTENT.value,
        idempotency_token_cid="absent",
    )
    # Identical inputs, different wall clocks → identical CIDs.
    assert c1.cid() == c2.cid()


def test_w84_different_args_produce_different_cids():
    c1 = ToolCallSchemaV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        tool_id="t", tool_version_cid="v1",
        args_bytes=b"args-A",
        timestamp_ns=0,
        caller_agent_id="a", run_cid="r",
        parent_event_cid="p",
        idempotency_class=IdempotencyClass.IDEMPOTENT.value,
        idempotency_token_cid="absent",
    )
    c2 = ToolCallSchemaV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        tool_id="t", tool_version_cid="v1",
        args_bytes=b"args-B",
        timestamp_ns=0,
        caller_agent_id="a", run_cid="r",
        parent_event_cid="p",
        idempotency_class=IdempotencyClass.IDEMPOTENT.value,
        idempotency_token_cid="absent",
    )
    assert c1.cid() != c2.cid()
    assert c1.args_cid != c2.args_cid


def test_w84_result_cid_is_over_result_bytes():
    r1 = ToolResultSchemaV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        call_cid="call",
        result_bytes=b"hello",
        exit_code=0,
        stderr_bytes=b"",
        duration_ns=12345,
        side_effects_cid="none",
        integrity_verdict=ToolIntegrityVerdict.OK.value,
    )
    r2 = ToolResultSchemaV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        call_cid="call",
        result_bytes=b"hello",
        exit_code=0,
        stderr_bytes=b"",
        duration_ns=999_999_999,  # different duration
        side_effects_cid="none",
        integrity_verdict=ToolIntegrityVerdict.OK.value,
    )
    assert r1.result_cid == r2.result_cid
    # duration is intentionally out of the canonical CID
    assert r1.cid() == r2.cid()


# ---------------------------------------------------------------
# Sandbox + tool calls.
# ---------------------------------------------------------------

def test_w84_stub_http_is_deterministic():
    http = DeterministicStubHTTPToolV1()
    args = b'{"url": "https://example.test/x", "body": "y"}'
    a = http.call(args)
    b = http.call(args)
    assert a == b


def test_w84_stub_http_rejects_missing_url():
    http = DeterministicStubHTTPToolV1()
    sandbox = ToolSandboxAdapterV1(limits=SandboxLimitsV1())
    rbytes, sbytes, exit_code, verdict, dur = sandbox.run(
        callable_=http.call, args_bytes=b'{"body": "x"}')
    assert verdict == ToolIntegrityVerdict.SANDBOX_VIOLATION
    assert exit_code == 2
    assert b"missing url" in sbytes


def test_w84_rg_refuses_path_outside_allow_list(tmp_path):
    rg = RipgrepLikeFilesystemToolV1()
    sandbox = ToolSandboxAdapterV1(
        limits=SandboxLimitsV1(
            fs_path_allow_list=(str(tmp_path),)))
    impl = rg.call_factory(
        fs_path_allow_list=(str(tmp_path),))
    other = tmp_path / ".." / "other"
    args = (b'{"pattern": "p", "root": "/etc"}')
    rbytes, sbytes, exit_code, verdict, dur = sandbox.run(
        callable_=impl, args_bytes=args)
    assert verdict == ToolIntegrityVerdict.SANDBOX_VIOLATION


def test_w84_rg_finds_match_in_allow_listed_root(tmp_path):
    p = tmp_path / "hello.txt"
    p.write_text("alpha beta NEEDLE gamma\n")
    rg = RipgrepLikeFilesystemToolV1()
    sandbox = ToolSandboxAdapterV1(
        limits=SandboxLimitsV1(
            fs_path_allow_list=(str(tmp_path),)))
    impl = rg.call_factory(
        fs_path_allow_list=(str(tmp_path),))
    args = (b'{"pattern": "NEEDLE", "root": "%s"}'
            % str(tmp_path).encode("utf-8"))
    rbytes, sbytes, exit_code, verdict, dur = sandbox.run(
        callable_=impl, args_bytes=args)
    assert verdict == ToolIntegrityVerdict.OK
    assert b"NEEDLE" in rbytes
    assert b'"n_matches":1' in rbytes


def test_w84_pyexec_is_off_by_default():
    pyexec = PythonExecSandboxToolV1()  # enable=False
    sandbox = ToolSandboxAdapterV1(limits=SandboxLimitsV1())
    rbytes, sbytes, exit_code, verdict, dur = sandbox.run(
        callable_=pyexec.call,
        args_bytes=b'{"code": "result = 2+2"}')
    assert verdict == ToolIntegrityVerdict.SANDBOX_VIOLATION


def test_w84_pyexec_enabled_runs_safe_snippet():
    pyexec = PythonExecSandboxToolV1(enable=True)
    sandbox = ToolSandboxAdapterV1(limits=SandboxLimitsV1())
    rbytes, sbytes, exit_code, verdict, dur = sandbox.run(
        callable_=pyexec.call,
        args_bytes=b'{"code": "result = sum([1, 2, 3])"}')
    assert verdict == ToolIntegrityVerdict.OK
    assert b'"6"' in rbytes


def test_w84_wall_time_limit_enforced_on_slow_tool():
    """A slow callable that exceeds wall-time gets RESOURCE_LIMIT."""
    import time

    def slow(args_bytes: bytes) -> bytes:
        time.sleep(0.05)
        return b"done"

    sandbox = ToolSandboxAdapterV1(
        limits=SandboxLimitsV1(max_wall_time_s=0.01))
    rbytes, sbytes, exit_code, verdict, dur = sandbox.run(
        callable_=slow, args_bytes=b"")
    assert verdict == ToolIntegrityVerdict.RESOURCE_LIMIT


# ---------------------------------------------------------------
# Audit chain.
# ---------------------------------------------------------------

def test_w84_idempotent_replay_does_not_double_commit():
    chain = ToolAuditChainV1()
    call = ToolCallSchemaV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        tool_id="t", tool_version_cid="v1",
        args_bytes=b"args",
        timestamp_ns=0,
        caller_agent_id="a", run_cid="r",
        parent_event_cid="p",
        idempotency_class=IdempotencyClass.IDEMPOTENT.value,
        idempotency_token_cid="absent",
    )
    result = ToolResultSchemaV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        call_cid=str(call.cid()),
        result_bytes=b"ok",
        exit_code=0,
        stderr_bytes=b"",
        duration_ns=1,
        side_effects_cid="none",
        integrity_verdict=ToolIntegrityVerdict.OK.value,
    )
    chain.commit(call=call, result=result)
    # Replay the same idempotent call.
    assert chain.already_committed(call, token=None)


def test_w84_non_idempotent_without_token_is_refused():
    chain = ToolAuditChainV1()
    call = ToolCallSchemaV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        tool_id="t", tool_version_cid="v1",
        args_bytes=b"args",
        timestamp_ns=0,
        caller_agent_id="a", run_cid="r",
        parent_event_cid="p",
        idempotency_class=(
            IdempotencyClass.NON_IDEMPOTENT.value),
        idempotency_token_cid="absent",
    )
    # already_committed with token=None → True (refused).
    assert chain.already_committed(call, token=None)


def test_w84_non_idempotent_with_token_committed_once():
    chain = ToolAuditChainV1()
    token = IdempotencyTokenV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        caller_agent_id="a",
        nonce="n",
    )
    call = ToolCallSchemaV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        tool_id="t", tool_version_cid="v1",
        args_bytes=b"args",
        timestamp_ns=0,
        caller_agent_id="a", run_cid="r",
        parent_event_cid="p",
        idempotency_class=(
            IdempotencyClass.NON_IDEMPOTENT.value),
        idempotency_token_cid=str(token.cid()),
    )
    result = ToolResultSchemaV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        call_cid=str(call.cid()),
        result_bytes=b"committed",
        exit_code=0,
        stderr_bytes=b"",
        duration_ns=1,
        side_effects_cid="se",
        integrity_verdict=ToolIntegrityVerdict.OK.value,
    )
    chain.commit(call=call, result=result, token=token)
    # Duplicate with the same token → refused.
    assert chain.already_committed(call, token=token)


def test_w84_audit_chain_merkle_root_stable():
    chain = ToolAuditChainV1()
    call = ToolCallSchemaV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        tool_id="t", tool_version_cid="v1",
        args_bytes=b"args",
        timestamp_ns=0,
        caller_agent_id="a", run_cid="r",
        parent_event_cid="p",
        idempotency_class=IdempotencyClass.IDEMPOTENT.value,
        idempotency_token_cid="absent",
    )
    result = ToolResultSchemaV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        call_cid=str(call.cid()),
        result_bytes=b"ok",
        exit_code=0,
        stderr_bytes=b"",
        duration_ns=1,
        side_effects_cid="none",
        integrity_verdict=ToolIntegrityVerdict.OK.value,
    )
    chain.commit(call=call, result=result)
    root1 = chain.merkle_root()
    chain2 = ToolAuditChainV1()
    chain2.commit(call=call, result=result)
    root2 = chain2.merkle_root()
    assert root1 == root2
    assert len(root1) == 64


# ---------------------------------------------------------------
# 5-agent team bench.
# ---------------------------------------------------------------

def test_w84_tool_substrate_team_bench_5_agents():
    rep = run_tool_substrate_team_bench_v1(n_agents=5)
    assert rep.n_agents == 5
    assert rep.n_tool_calls >= 4  # 4 calls actually committed
    assert rep.non_idempotent_duplicate_refused is True
    assert rep.sandbox_violation_caught is True
    assert len(rep.audit_chain_merkle_root) == 64
    assert rep.tool_chain_replayable_from_disk is True


def test_w84_tool_substrate_team_bench_cid_stable():
    r1 = run_tool_substrate_team_bench_v1(n_agents=5)
    r2 = run_tool_substrate_team_bench_v1(n_agents=5)
    # Same Merkle root across runs (deterministic).
    assert r1.audit_chain_merkle_root == r2.audit_chain_merkle_root
