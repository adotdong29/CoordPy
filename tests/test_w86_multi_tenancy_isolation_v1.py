"""Tests for ``coordpy.multi_tenancy_isolation_v1``."""

from __future__ import annotations

import pytest

cryptography = pytest.importorskip("cryptography")

from coordpy.multi_tenancy_isolation_v1 import (  # noqa: E402
    CrossTenantAccessDeniedEventV1,
    MultiTenancyBenchReportV1,
    MultiTenantGatewayV1,
    TenantBudgetV1,
    TenantIdentityV1,
    TenantKeyPairV1,
    TenantPolicyV1,
    TenantTokenV1,
    issue_tenant_token_v1,
    run_two_tenant_isolation_bench_v1,
)


def _register_test_tenant(
        gateway, tenant_id, seed,
        max_cost=10.0):
    key = TenantKeyPairV1.from_seed(tenant_id, seed)
    pol = TenantPolicyV1()
    bud = TenantBudgetV1(max_total_cost_usd=max_cost)
    ident = TenantIdentityV1(
        tenant_id=tenant_id,
        public_key_bytes=key.public_key_bytes,
        policy_cid=pol.cid())
    gateway.register_tenant(
        identity=ident, policy=pol, budget=bud,
        public_key_bytes=key.public_key_bytes)
    return key, ident


def test_tenant_identity_cid_stable():
    key = TenantKeyPairV1.from_seed("t1", 100)
    pol = TenantPolicyV1()
    id1 = TenantIdentityV1(
        "t1", key.public_key_bytes, pol.cid())
    id2 = TenantIdentityV1(
        "t1", key.public_key_bytes, pol.cid())
    assert id1.cid() == id2.cid()


def test_tenant_token_verifies_under_own_pubkey():
    key = TenantKeyPairV1.from_seed("t1", 100)
    pol = TenantPolicyV1()
    ident = TenantIdentityV1(
        "t1", key.public_key_bytes, pol.cid())
    tok = issue_tenant_token_v1(key, ident)
    assert tok.verify(key.public_key_bytes)


def test_tenant_token_fails_under_wrong_pubkey():
    key_a = TenantKeyPairV1.from_seed("a", 1)
    key_b = TenantKeyPairV1.from_seed("b", 2)
    pol = TenantPolicyV1()
    id_a = TenantIdentityV1(
        "a", key_a.public_key_bytes, pol.cid())
    tok_a = issue_tenant_token_v1(key_a, id_a)
    # Verifying with B's key fails.
    assert not tok_a.verify(key_b.public_key_bytes)


def test_register_two_tenants_isolated_state():
    g = MultiTenantGatewayV1()
    _register_test_tenant(g, "alpha", 1)
    _register_test_tenant(g, "beta", 2)
    assert "alpha" in g.tenants
    assert "beta" in g.tenants
    assert (
        g.tenants["alpha"].event_graph.root_event_id
        == g.tenants["beta"].event_graph.root_event_id)
    # But the graphs are distinct objects.
    assert (
        g.tenants["alpha"].event_graph is not
        g.tenants["beta"].event_graph)


def test_cannot_register_duplicate_tenant():
    g = MultiTenantGatewayV1()
    _register_test_tenant(g, "alpha", 1)
    with pytest.raises(ValueError):
        _register_test_tenant(g, "alpha", 2)


def test_cross_tenant_read_is_refused():
    g = MultiTenantGatewayV1()
    _, id_a = _register_test_tenant(g, "alpha", 1)
    key_a = TenantKeyPairV1.from_seed("alpha", 1)
    tok_a = issue_tenant_token_v1(key_a, id_a)
    _, id_b = _register_test_tenant(g, "beta", 2)
    key_b = TenantKeyPairV1.from_seed("beta", 2)
    tok_b = issue_tenant_token_v1(key_b, id_b)
    g.append_event_for_tenant(
        token=tok_b, event_id="b_secret",
        kind="secret_event",
        payload_bytes=b"super_secret_B",
        parent_event_ids=(
            g.tenants["beta"].event_graph.root_event_id,),
        timestamp_ns=1)
    # A attempts to read B's event with A's token.
    res = g.read_event_for_tenant(
        token=tok_a, event_id="b_secret",
        target_tenant_id="beta", now_ns=10)
    assert res is None
    # And a denial event was recorded in A's chain.
    assert len(g.tenants["alpha"].denied_events) == 1
    den = g.tenants["alpha"].denied_events[0]
    assert den.requesting_tenant_id == "alpha"
    assert den.target_tenant_id == "beta"


def test_budget_isolation_holds():
    g = MultiTenantGatewayV1()
    _, id_a = _register_test_tenant(g, "alpha", 1, max_cost=1.0)
    _, id_b = _register_test_tenant(g, "beta", 2, max_cost=1.0)
    key_a = TenantKeyPairV1.from_seed("alpha", 1)
    key_b = TenantKeyPairV1.from_seed("beta", 2)
    tok_a = issue_tenant_token_v1(key_a, id_a)
    # Drain A.
    for i in range(100):
        res = g.append_event_for_tenant(
            token=tok_a,
            event_id=f"drain_{i:03d}",
            kind="drain",
            payload_bytes=f"x_{i}".encode(),
            parent_event_ids=(
                g.tenants["alpha"].event_graph.root_event_id,),
            timestamp_ns=i, cost_usd=0.5)
        if not res.accepted:
            break
    # B's budget untouched.
    assert g.tenants["beta"].budget.spent_cost_usd == 0.0


def test_audit_anchors_distinct_across_tenants():
    g = MultiTenantGatewayV1()
    _, id_a = _register_test_tenant(g, "alpha", 1)
    _, id_b = _register_test_tenant(g, "beta", 2)
    key_a = TenantKeyPairV1.from_seed("alpha", 1)
    key_b = TenantKeyPairV1.from_seed("beta", 2)
    tok_a = issue_tenant_token_v1(key_a, id_a)
    tok_b = issue_tenant_token_v1(key_b, id_b)
    # Write the SAME-PAYLOAD events to both — anchors must
    # still be distinct.
    for i in range(3):
        for tok, name in ((tok_a, "alpha"), (tok_b, "beta")):
            g.append_event_for_tenant(
                token=tok,
                event_id=f"{name}_evt_{i:03d}",
                kind="same",
                payload_bytes=b"same_payload",
                parent_event_ids=(
                    g.tenants[name].event_graph.root_event_id,),
                timestamp_ns=i)
    a_anchor = g.get_audit_anchor(token=tok_a, now_ns=100)
    b_anchor = g.get_audit_anchor(token=tok_b, now_ns=100)
    assert a_anchor.merkle_root() != b_anchor.merkle_root()


def test_token_swap_refused():
    """A token signed by tenant A's key but claiming tenant B's
    identity MUST fail verification."""
    g = MultiTenantGatewayV1()
    _, id_a = _register_test_tenant(g, "alpha", 1)
    _, id_b = _register_test_tenant(g, "beta", 2)
    key_a = TenantKeyPairV1.from_seed("alpha", 1)
    # Build a token claiming tenant_id="beta" but signed with
    # A's key.
    import json as _json
    bad_token = TenantTokenV1(
        tenant_id="beta",
        tenant_cid=id_b.cid(),
        nonce="malicious",
        signature_bytes=key_a.sign(
            _json.dumps(
                {
                    "kind": "w86_tenant_token_v1",
                    "tenant_id": "beta",
                    "tenant_cid": id_b.cid(),
                    "nonce": "malicious",
                },
                sort_keys=True,
                separators=(",", ":")).encode("utf-8")))
    assert not g.verify_token(bad_token)


def test_bench_meets_all_dod_bars():
    rep = run_two_tenant_isolation_bench_v1()
    assert rep.cross_tenant_read_refused is True
    assert rep.cross_tenant_denial_event_emitted is True
    assert rep.audit_anchors_distinct is True
    assert rep.budget_isolation_holds is True
    assert rep.token_swap_refused is True
    assert rep.no_b_bytes_in_a_chain is True


def test_bench_report_cid_deterministic():
    r1 = run_two_tenant_isolation_bench_v1()
    r2 = run_two_tenant_isolation_bench_v1()
    assert r1.report_cid == r2.report_cid


def test_cross_tenant_denial_event_is_content_addressed():
    den = CrossTenantAccessDeniedEventV1(
        requesting_tenant_id="alpha",
        requesting_tenant_cid="alpha_cid",
        target_tenant_id="beta",
        target_tenant_cid="beta_cid",
        requested_event_id="evt_001",
        denied_at_ns=12345,
        denial_reason="not_allowed")
    assert len(den.cid()) == 64


def test_event_graph_objects_distinct_per_tenant():
    """Physical partitioning (not just logical tenant_id field)."""
    g = MultiTenantGatewayV1()
    _register_test_tenant(g, "alpha", 1)
    _register_test_tenant(g, "beta", 2)
    a_graph = g.tenants["alpha"].event_graph
    b_graph = g.tenants["beta"].event_graph
    # They start equal (both empty + genesis), but are distinct
    # Python objects.
    assert a_graph is not b_graph
    # Mutations to A don't show up in B.
    key_a = TenantKeyPairV1.from_seed("alpha", 1)
    pol = TenantPolicyV1()
    id_a = TenantIdentityV1(
        "alpha", key_a.public_key_bytes, pol.cid())
    tok_a = issue_tenant_token_v1(key_a, id_a)
    g.append_event_for_tenant(
        token=tok_a, event_id="a_only",
        kind="x", payload_bytes=b"a",
        parent_event_ids=(a_graph.root_event_id,),
        timestamp_ns=1)
    assert "a_only" in g.tenants["alpha"].event_graph.nodes
    assert "a_only" not in g.tenants["beta"].event_graph.nodes


def test_policy_cid_mismatch_rejected():
    g = MultiTenantGatewayV1()
    key = TenantKeyPairV1.from_seed("t1", 1)
    pol_a = TenantPolicyV1(max_total_events=100)
    pol_b = TenantPolicyV1(max_total_events=200)
    ident = TenantIdentityV1(
        "t1", key.public_key_bytes, pol_a.cid())
    bud = TenantBudgetV1()
    with pytest.raises(ValueError):
        g.register_tenant(
            identity=ident, policy=pol_b, budget=bud,
            public_key_bytes=key.public_key_bytes)
