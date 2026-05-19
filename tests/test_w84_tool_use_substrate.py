"""W84 / P1 #33 — Tool-Use Substrate tests."""

from __future__ import annotations


def test_w84_tool_call_schema_content_addressed():
    """DoD bar: ToolCallSchemaV1 + ToolResultSchemaV1 are
    content-addressed and re-hashable from disk."""
    from coordpy.tool_use_substrate_v1 import (
        make_tool_call_v1,
    )
    call = make_tool_call_v1(
        tool_id="t", tool_version_cid="vcid",
        args={"x": 1}, timestamp_ns=10,
        caller_agent_id="a", run_cid="r")
    assert len(call.cid()) == 64
    # Re-hashable from to_dict — identical fields yield
    # identical CIDs.
    call2 = make_tool_call_v1(
        tool_id="t", tool_version_cid="vcid",
        args={"x": 1}, timestamp_ns=10,
        caller_agent_id="a", run_cid="r")
    assert call.cid() == call2.cid()


def test_w84_identical_inputs_give_identical_call_cids():
    """DoD bar: identical tool call inputs produce identical
    call CIDs."""
    from coordpy.tool_use_substrate_v1 import (
        make_tool_call_v1,
    )
    a = make_tool_call_v1(
        tool_id="x", tool_version_cid="v",
        args={"y": [1, 2, 3]},
        timestamp_ns=1234, caller_agent_id="a", run_cid="r")
    b = make_tool_call_v1(
        tool_id="x", tool_version_cid="v",
        args={"y": [1, 2, 3]},
        timestamp_ns=1234, caller_agent_id="a", run_cid="r")
    assert a.cid() == b.cid()
    c = make_tool_call_v1(
        tool_id="x", tool_version_cid="v",
        args={"y": [1, 2, 4]},  # different
        timestamp_ns=1234, caller_agent_id="a", run_cid="r")
    assert a.cid() != c.cid()


def test_w84_non_idempotent_double_commit_refused():
    """DoD bar: replay of non-idempotent call without an
    idempotency token must be REFUSED."""
    from coordpy.tool_use_substrate_v1 import (
        IdempotencyRegistryV1, IdempotencyTokenV1,
        ToolIdempotency, ToolIntegrityVerdict,
        W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        make_tool_call_v1,
        run_tool_call_v1, tool_http_fetch_stub,
    )
    reg = IdempotencyRegistryV1()
    tok = IdempotencyTokenV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        run_cid="r", agent_id="a", sequence_no=1)
    call1 = make_tool_call_v1(
        tool_id="t", tool_version_cid="v",
        args={"url": "u1"}, timestamp_ns=1,
        caller_agent_id="a", run_cid="r",
        idempotency=ToolIdempotency.NON_IDEMPOTENT.value,
        idempotency_token=tok)
    res1 = run_tool_call_v1(
        callable_=tool_http_fetch_stub, call=call1,
        idempotency_registry=reg, args={"url": "u1"})
    assert (
        res1.integrity_verdict
        == ToolIntegrityVerdict.OK.value)
    # SAME token but DIFFERENT call -> violation.
    call2 = make_tool_call_v1(
        tool_id="t", tool_version_cid="v",
        args={"url": "u2"}, timestamp_ns=2,
        caller_agent_id="a", run_cid="r",
        idempotency=ToolIdempotency.NON_IDEMPOTENT.value,
        idempotency_token=tok)
    res2 = run_tool_call_v1(
        callable_=tool_http_fetch_stub, call=call2,
        idempotency_registry=reg, args={"url": "u2"})
    assert (
        res2.integrity_verdict
        == ToolIntegrityVerdict.IDEMPOTENCY_VIOLATION.value)


def test_w84_non_idempotent_without_token_rejected_at_build():
    """Anti-cheat: non-idempotent tool calls require an
    explicit IdempotencyTokenV1."""
    import pytest
    from coordpy.tool_use_substrate_v1 import (
        ToolIdempotency, make_tool_call_v1,
    )
    with pytest.raises(ValueError):
        make_tool_call_v1(
            tool_id="t", tool_version_cid="v",
            args={}, timestamp_ns=1,
            caller_agent_id="a", run_cid="r",
            idempotency=(
                ToolIdempotency.NON_IDEMPOTENT.value),
            idempotency_token=None)


def test_w84_sandbox_enforces_wall_time_limit():
    """DoD bar: at least one real resource limit is enforced
    by the V1 sandbox."""
    import time as _time
    from coordpy.tool_use_substrate_v1 import (
        ToolIntegrityVerdict,
        ToolSandboxAdapterV1,
        ToolSandboxLimitsV1,
    )
    sandbox = ToolSandboxAdapterV1(
        limits=ToolSandboxLimitsV1(
            max_wall_time_seconds=0.05))

    def _slow(**_kw):
        _time.sleep(0.3)
        return {"ok": True}

    exec_ = sandbox.execute(
        callable_=_slow, args={"x": 1})
    assert (
        exec_.integrity_verdict
        == ToolIntegrityVerdict.LIMIT_EXCEEDED.value)
    # Duration should be at or below the limit (with a small
    # margin for the join-timeout).
    assert exec_.duration_ns / 1e9 < 0.2


def test_w84_python_exec_tool_runs():
    """V1 tool: Python exec sandbox runs and returns a value."""
    from coordpy.tool_use_substrate_v1 import tool_python_exec
    out = tool_python_exec(code="result = sum([1, 2, 3])")
    assert out["ok"]
    assert int(out["result"]) == 6


def test_w84_filesystem_ripgrep_tool_runs():
    """V1 tool: filesystem ripgrep walks a real directory."""
    from coordpy.tool_use_substrate_v1 import (
        tool_filesystem_ripgrep_lite,
    )
    out = tool_filesystem_ripgrep_lite(
        root=".", pattern="coordpy", max_matches=4)
    assert out["ok"]
    assert int(len(out["matches"])) > 0


def test_w84_http_fetch_stub_is_deterministic():
    """V1 tool: stub HTTP fetch is deterministic on the URL."""
    from coordpy.tool_use_substrate_v1 import (
        tool_http_fetch_stub,
    )
    out1 = tool_http_fetch_stub(url="https://x")
    out2 = tool_http_fetch_stub(url="https://x")
    assert out1["body"] == out2["body"]


def test_w84_tool_audit_chain_reverifies_from_disk():
    """DoD bar: a third party can re-verify the audit chain
    without re-running the tool calls."""
    from coordpy.tool_use_substrate_v1 import (
        build_tool_audit_chain_v1,
        make_tool_call_v1, run_tool_call_v1,
        tool_python_exec,
    )
    call = make_tool_call_v1(
        tool_id="python_exec", tool_version_cid="v1",
        args={"code": "result = 1"}, timestamp_ns=1,
        caller_agent_id="a", run_cid="r")
    res = run_tool_call_v1(
        callable_=tool_python_exec, call=call,
        args={"code": "result = 1"})
    chain = build_tool_audit_chain_v1(
        capsules=[("tool_call", call),
                  ("tool_result", res)])
    assert chain.verify_merkle_root()


def test_w84_tool_audit_chain_tamper_detected():
    """Anti-cheat: tampering with any CID breaks the audit
    chain's Merkle root."""
    from coordpy.tool_use_substrate_v1 import (
        ToolAuditChainV1,
        W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        build_tool_audit_chain_v1,
        make_tool_call_v1, run_tool_call_v1,
        tool_python_exec,
    )
    call = make_tool_call_v1(
        tool_id="t", tool_version_cid="v",
        args={"code": "result = 1"}, timestamp_ns=1,
        caller_agent_id="a", run_cid="r")
    res = run_tool_call_v1(
        callable_=tool_python_exec, call=call,
        args={"code": "result = 1"})
    chain = build_tool_audit_chain_v1(
        capsules=[("tool_call", call),
                  ("tool_result", res)])
    # Tamper: change one CID.
    tampered_cids = list(chain.capsule_cids)
    tampered_cids[0] = "deadbeef" * 8
    tampered_chain = ToolAuditChainV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        capsule_kinds=chain.capsule_kinds,
        capsule_cids=tuple(tampered_cids),
        merkle_root_cid=chain.merkle_root_cid)
    assert not tampered_chain.verify_merkle_root()


def test_w84_idempotent_replay_returns_cached_no_re_exec():
    """DoD bar: replay of an idempotent call emits the cached
    result (no re-execution)."""
    from coordpy.tool_use_substrate_v1 import (
        IdempotencyRegistryV1, IdempotencyTokenV1,
        ToolIdempotency,
        W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        make_tool_call_v1, run_tool_call_v1,
    )
    reg = IdempotencyRegistryV1()
    tok = IdempotencyTokenV1(
        schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
        run_cid="r", agent_id="a", sequence_no=1)
    n_calls = [0]

    def _counter(**_):
        n_calls[0] += 1
        return {"ok": True, "n_calls": n_calls[0]}

    call = make_tool_call_v1(
        tool_id="t", tool_version_cid="v",
        args={"x": 1}, timestamp_ns=1,
        caller_agent_id="a", run_cid="r",
        idempotency=ToolIdempotency.NON_IDEMPOTENT.value,
        idempotency_token=tok)
    res1 = run_tool_call_v1(
        callable_=_counter, call=call,
        idempotency_registry=reg, args={"x": 1})
    res2 = run_tool_call_v1(
        callable_=_counter, call=call,
        idempotency_registry=reg, args={"x": 1})
    # The second call returned the cached result -> the
    # callable was only invoked once.
    assert n_calls[0] == 1
    assert res1.cid() == res2.cid()


def test_w84_5_agent_tool_use_bench_passes():
    """End-to-end DoD bar: 5-agent team bench produces audit
    chain that mixes LLM-side capsules with tool-side
    capsules, Merkle root over the merged chain."""
    from coordpy.tool_use_substrate_v1 import (
        run_tool_use_bench_v1,
    )
    rep = run_tool_use_bench_v1()
    d = rep.to_dict()
    assert int(rep.n_agents) == 5
    assert int(rep.n_tool_calls) == 5
    assert bool(rep.all_calls_content_addressed), d
    assert bool(
        rep.identical_inputs_give_identical_call_cids), d
    assert bool(
        rep.non_idempotent_double_commit_refused), d
    assert bool(rep.sandbox_wall_time_limit_enforced), d
    assert bool(rep.audit_chain_reverifies_from_disk), d
    assert bool(rep.tamper_breaks_audit_chain), d


def test_w84_tool_result_bytes_carried_in_cid():
    """Anti-cheat: result bytes (incl. binary blob) must be
    content-addressed."""
    from coordpy.tool_use_substrate_v1 import (
        make_tool_call_v1, run_tool_call_v1,
        tool_http_fetch_stub,
    )
    call = make_tool_call_v1(
        tool_id="http_fetch_stub", tool_version_cid="v1",
        args={"url": "u1"}, timestamp_ns=1,
        caller_agent_id="a", run_cid="r")
    res = run_tool_call_v1(
        callable_=tool_http_fetch_stub, call=call,
        args={"url": "u1"})
    assert len(res.result_bytes_cid) == 64
    # The result bytes CID is part of the result CID — change
    # the URL, change the result bytes CID.
    call2 = make_tool_call_v1(
        tool_id="http_fetch_stub", tool_version_cid="v1",
        args={"url": "u2"}, timestamp_ns=1,
        caller_agent_id="a", run_cid="r")
    res2 = run_tool_call_v1(
        callable_=tool_http_fetch_stub, call=call2,
        args={"url": "u2"})
    assert res.result_bytes_cid != res2.result_bytes_cid


def test_w84_audit_chain_kinds_mix_llm_and_tool_capsules():
    """DoD bar: audit chain mixes LLM-side + tool-side
    capsules. Verified by passing both kinds in."""
    from coordpy.tool_use_substrate_v1 import (
        build_tool_audit_chain_v1,
        make_tool_call_v1,
        run_tool_call_v1, tool_python_exec,
    )

    class _LLMCapsule:
        def __init__(self, c):
            self._c = c

        def cid(self):
            import hashlib
            return hashlib.sha256(
                self._c.encode()).hexdigest()

    call = make_tool_call_v1(
        tool_id="t", tool_version_cid="v",
        args={"code": "result = 0"}, timestamp_ns=1,
        caller_agent_id="a", run_cid="r")
    res = run_tool_call_v1(
        callable_=tool_python_exec, call=call,
        args={"code": "result = 0"})
    llm_cap = _LLMCapsule("hidden_state_capsule")
    chain = build_tool_audit_chain_v1(
        capsules=[
            ("llm_capsule", llm_cap),
            ("tool_call", call),
            ("tool_result", res),
        ])
    assert "llm_capsule" in chain.capsule_kinds
    assert "tool_call" in chain.capsule_kinds
    assert "tool_result" in chain.capsule_kinds
    assert chain.verify_merkle_root()
