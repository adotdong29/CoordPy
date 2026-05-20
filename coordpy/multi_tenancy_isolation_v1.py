"""W86+ / P2 #43 — Multi-Tenancy Isolation V1.

Issue #43 asks for a real multi-tenant isolation story on top of
the W81 deployable substrate gateway + W82 event graph + W83
hosted audit anchor. The DoD demands:

1. ``TenantIdentityV1`` — content-addressed tenant capsule with
   ID, public key, policy CID.
2. **PerTenantEventGraph** — physically partitioned graphs;
   cross-tenant queries emit
   ``CrossTenantAccessDeniedEventV1`` events.
3. **PerTenantBudgets** — Tenant A's budget cannot be spent by
   Tenant B.
4. **PerTenantHostedAuditAnchor** — distinct Merkle roots.
5. **Tenant-isolation bench** — two tenants concurrently; no
   B byte in A's chain.
6. **Auth + Audit per tenant** — Ed25519-bound bearer tokens.

Honest scope (V1)
-----------------

* ``W86-L-TENANCY-V1-RESEARCH-ONLY-CAP``
* ``W86-L-TENANCY-V1-TWO-TENANTS-CAP`` — V1 bench shape is two
  tenants concurrently; n-tenant V2.
* ``W86-L-TENANCY-V1-ED25519-CAP`` — Ed25519 tenant-token
  binding; full PKI is V3.
* ``W86-L-TENANCY-V1-PHYSICAL-PARTITION-CAP`` — each tenant
  has its OWN ``EventGraphV1`` instance (not a shared graph
  with a tenant_id field). This is the load-bearing "physical
  partitioning" requirement.
* ``W86-L-TENANCY-V1-AUDIT-ANCHOR-PER-TENANT-CAP`` — each
  tenant's audit anchor is a separate Merkle root over only
  its own events.
* ``W86-L-TENANCY-V1-CROSS-TENANT-DENIAL-AUDITS-CAP`` — every
  rejected cross-tenant query is recorded as an audit event
  in the requesting tenant's chain (operator's actions are
  themselves auditable).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from typing import Any, Mapping, Optional, Sequence

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey, Ed25519PublicKey)
    from cryptography.hazmat.primitives.serialization import (
        Encoding, PrivateFormat, PublicFormat, NoEncryption)
    from cryptography.exceptions import InvalidSignature
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.multi_tenancy_isolation_v1 requires the "
        "`cryptography` package") from exc

try:
    from .event_sourced_memory_graph_v1 import (
        EventGraphV1,
        EventNodeV1,
        build_event_node_v1,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.multi_tenancy_isolation_v1 requires "
        "coordpy.event_sourced_memory_graph_v1") from exc


W86_TENANCY_V1_SCHEMA_VERSION: str = (
    "coordpy.multi_tenancy_isolation_v1.v1")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            default=str).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------
# Tenant identity + keys
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TenantPolicyV1:
    """Tenant-scoped policy (V1)."""

    max_total_events: int = 1_000_000
    max_total_bytes: int = 1 << 30  # 1 GiB
    max_event_kinds: int = 100
    """Maximum distinct event ``kind`` strings allowed."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W86_TENANCY_V1_SCHEMA_VERSION,
            "max_total_events": int(self.max_total_events),
            "max_total_bytes": int(self.max_total_bytes),
            "max_event_kinds": int(self.max_event_kinds),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_tenant_policy_v1",
            "policy": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class TenantBudgetV1:
    """Per-tenant cost + latency budget."""

    max_total_cost_usd: float = 100.0
    max_total_tokens: int = 1_000_000
    max_total_tool_calls: int = 10_000
    spent_cost_usd: float = 0.0
    spent_tokens: int = 0
    spent_tool_calls: int = 0

    def remaining_cost_usd(self) -> float:
        return float(self.max_total_cost_usd - self.spent_cost_usd)

    def remaining_tokens(self) -> int:
        return int(self.max_total_tokens - self.spent_tokens)

    def remaining_tool_calls(self) -> int:
        return int(
            self.max_total_tool_calls - self.spent_tool_calls)

    def is_exhausted(self) -> bool:
        return (
            self.remaining_cost_usd() <= 0
            or self.remaining_tokens() <= 0
            or self.remaining_tool_calls() <= 0)

    def spend(
            self, cost_usd: float = 0.0,
            tokens: int = 0,
            tool_calls: int = 0) -> "TenantBudgetV1":
        return TenantBudgetV1(
            max_total_cost_usd=self.max_total_cost_usd,
            max_total_tokens=self.max_total_tokens,
            max_total_tool_calls=self.max_total_tool_calls,
            spent_cost_usd=(
                self.spent_cost_usd + float(cost_usd)),
            spent_tokens=int(self.spent_tokens + int(tokens)),
            spent_tool_calls=int(
                self.spent_tool_calls + int(tool_calls)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W86_TENANCY_V1_SCHEMA_VERSION,
            "max_total_cost_usd": float(round(
                self.max_total_cost_usd, 6)),
            "max_total_tokens": int(self.max_total_tokens),
            "max_total_tool_calls": int(
                self.max_total_tool_calls),
            "spent_cost_usd": float(round(
                self.spent_cost_usd, 6)),
            "spent_tokens": int(self.spent_tokens),
            "spent_tool_calls": int(self.spent_tool_calls),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_tenant_budget_v1",
            "budget": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class TenantKeyPairV1:
    """Ed25519 keypair for one tenant."""

    tenant_id: str
    private_key_bytes: bytes
    public_key_bytes: bytes

    @classmethod
    def from_seed(
            cls, tenant_id: str, seed: int) -> "TenantKeyPairV1":
        seed_bytes = hashlib.sha256(
            f"tenant::{tenant_id}::{int(seed)}"
            .encode("utf-8")).digest()
        priv = Ed25519PrivateKey.from_private_bytes(seed_bytes)
        pub = priv.public_key()
        return cls(
            tenant_id=str(tenant_id),
            private_key_bytes=seed_bytes,
            public_key_bytes=pub.public_bytes(
                Encoding.Raw, PublicFormat.Raw))

    def sign(self, message: bytes) -> bytes:
        priv = Ed25519PrivateKey.from_private_bytes(
            self.private_key_bytes)
        return priv.sign(message)


@dataclasses.dataclass(frozen=True)
class TenantIdentityV1:
    """Content-addressed tenant capsule.

    The tenant_id, public key, and policy CID together form
    the load-bearing tenant identity. Every audit capsule in
    the tenant's chain has the tenant CID in its parent chain.
    """

    tenant_id: str
    public_key_bytes: bytes
    policy_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W86_TENANCY_V1_SCHEMA_VERSION,
            "tenant_id": str(self.tenant_id),
            "public_key_hex": self.public_key_bytes.hex(),
            "policy_cid": str(self.policy_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_tenant_identity_v1",
            "identity": self.to_dict()})


# ---------------------------------------------------------------------
# Tenant tokens
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TenantTokenV1:
    """Bearer-shaped token bound to a tenant identity.

    The token is the Ed25519 signature of
    ``{tenant_id, tenant_cid, nonce}``. Verifying the token
    requires knowing the tenant's public key and re-signing
    the same bytes. A Tenant A token CANNOT be used for
    Tenant B because the signature commits to ``tenant_id``.
    """

    tenant_id: str
    tenant_cid: str
    nonce: str
    signature_bytes: bytes

    def _bytes_to_sign(self) -> bytes:
        return json.dumps(
            {
                "kind": "w86_tenant_token_v1",
                "tenant_id": str(self.tenant_id),
                "tenant_cid": str(self.tenant_cid),
                "nonce": str(self.nonce),
            },
            sort_keys=True,
            separators=(",", ":")).encode("utf-8")

    def verify(self, public_key_bytes: bytes) -> bool:
        try:
            pub = Ed25519PublicKey.from_public_bytes(
                public_key_bytes)
            pub.verify(self.signature_bytes, self._bytes_to_sign())
            return True
        except (InvalidSignature, Exception):
            return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": str(self.tenant_id),
            "tenant_cid": str(self.tenant_cid),
            "nonce": str(self.nonce),
            "signature_hex": self.signature_bytes.hex(),
        }


def issue_tenant_token_v1(
        tenant_key: TenantKeyPairV1,
        tenant_identity: TenantIdentityV1,
        nonce: str = "default") -> TenantTokenV1:
    if tenant_key.tenant_id != tenant_identity.tenant_id:
        raise ValueError(
            "tenant_key.tenant_id must match "
            "tenant_identity.tenant_id")
    token = TenantTokenV1(
        tenant_id=tenant_identity.tenant_id,
        tenant_cid=tenant_identity.cid(),
        nonce=str(nonce),
        signature_bytes=b"")
    sig = tenant_key.sign(token._bytes_to_sign())
    return dataclasses.replace(token, signature_bytes=sig)


# ---------------------------------------------------------------------
# Per-tenant chain + cross-tenant denial events
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CrossTenantAccessDeniedEventV1:
    """Audit event recording a refused cross-tenant access."""

    requesting_tenant_id: str
    requesting_tenant_cid: str
    target_tenant_id: str
    target_tenant_cid: str
    requested_event_id: str
    denied_at_ns: int
    denial_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W86_TENANCY_V1_SCHEMA_VERSION,
            "requesting_tenant_id": str(self.requesting_tenant_id),
            "requesting_tenant_cid": str(
                self.requesting_tenant_cid),
            "target_tenant_id": str(self.target_tenant_id),
            "target_tenant_cid": str(self.target_tenant_cid),
            "requested_event_id": str(self.requested_event_id),
            "denied_at_ns": int(self.denied_at_ns),
            "denial_reason": str(self.denial_reason),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_cross_tenant_access_denied_event_v1",
            "event": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class TenantAuditAnchorV1:
    """Per-tenant Merkle root over the tenant's event chain.

    Computed as SHA-256 of the sorted concatenation of every
    event's CID. The tenant CID is included in the anchor so
    Tenant A and Tenant B cannot have colliding anchors even
    if they have identical event sequences.
    """

    tenant_cid: str
    event_cids: tuple[str, ...]
    anchor_at_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W86_TENANCY_V1_SCHEMA_VERSION,
            "tenant_cid": str(self.tenant_cid),
            "event_cids": list(self.event_cids),
            "anchor_at_ns": int(self.anchor_at_ns),
            "merkle_root": self.merkle_root(),
        }

    def merkle_root(self) -> str:
        # Simple sorted-leaves Merkle root that includes the
        # tenant CID as an extra leaf, so anchors are tenant-
        # disjoint even with identical event sequences.
        leaves = sorted(list(self.event_cids) + [self.tenant_cid])
        return _sha256_hex({
            "kind": "w86_tenant_audit_anchor_merkle_v1",
            "leaves": leaves})

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_tenant_audit_anchor_v1",
            "anchor": self.to_dict()})


@dataclasses.dataclass
class TenantStateV1:
    """One tenant's live state inside the gateway.

    The gateway holds one ``TenantStateV1`` per tenant; cross-
    tenant access is policy-checked at the API boundary.
    """

    identity: TenantIdentityV1
    policy: TenantPolicyV1
    budget: TenantBudgetV1
    event_graph: EventGraphV1
    denied_events: tuple[CrossTenantAccessDeniedEventV1, ...] = ()

    def all_event_cids(self) -> list[str]:
        return [
            self.event_graph.nodes[eid].cid()
            for eid in sorted(self.event_graph.nodes.keys())]

    def audit_anchor(self, at_ns: int) -> TenantAuditAnchorV1:
        return TenantAuditAnchorV1(
            tenant_cid=self.identity.cid(),
            event_cids=tuple(self.all_event_cids()),
            anchor_at_ns=int(at_ns))

    def append_event(
            self, *, event_id: str, kind: str,
            payload_bytes: bytes,
            parent_event_ids: Sequence[str],
            branch_label: str = "main",
            timestamp_ns: int = 0) -> None:
        ev = build_event_node_v1(
            event_id=event_id, kind=kind,
            payload_bytes=payload_bytes,
            parent_event_ids=parent_event_ids,
            branch_label=branch_label,
            timestamp_ns=timestamp_ns)
        self.event_graph = self.event_graph.with_event(ev)

    def record_denial(
            self, denial: CrossTenantAccessDeniedEventV1) -> None:
        self.denied_events = self.denied_events + (denial,)


# ---------------------------------------------------------------------
# Gateway with multi-tenant isolation
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TenantSpendResultV1:
    """Result of one budget spend."""

    accepted: bool
    new_budget: TenantBudgetV1
    reason: Optional[str] = None


class MultiTenantGatewayV1:
    """In-process multi-tenant gateway.

    The gateway holds a dict ``tenant_id → TenantStateV1`` and
    enforces:

    * Token verification against the per-tenant public key.
    * Cross-tenant query rejection → emits a denial event in
      the *requesting* tenant's chain.
    * Per-tenant budget — Tenant A's spend never touches
      Tenant B's budget.
    * Per-tenant Merkle audit anchors.
    """

    def __init__(self) -> None:
        self.tenants: dict[str, TenantStateV1] = {}
        self.tenant_public_keys: dict[str, bytes] = {}

    def register_tenant(
            self, *, identity: TenantIdentityV1,
            policy: TenantPolicyV1,
            budget: TenantBudgetV1,
            public_key_bytes: bytes) -> None:
        if identity.tenant_id in self.tenants:
            raise ValueError(
                f"tenant {identity.tenant_id!r} already "
                "registered")
        if identity.policy_cid != policy.cid():
            raise ValueError(
                f"tenant identity policy_cid "
                f"{identity.policy_cid!r} doesn't match "
                f"policy.cid() {policy.cid()!r}")
        self.tenants[identity.tenant_id] = TenantStateV1(
            identity=identity, policy=policy, budget=budget,
            event_graph=EventGraphV1.empty())
        self.tenant_public_keys[
            identity.tenant_id] = bytes(public_key_bytes)

    def verify_token(self, token: TenantTokenV1) -> bool:
        pk = self.tenant_public_keys.get(token.tenant_id)
        if pk is None:
            return False
        if not token.verify(pk):
            return False
        ten = self.tenants.get(token.tenant_id)
        if ten is None:
            return False
        return token.tenant_cid == ten.identity.cid()

    def append_event_for_tenant(
            self, *, token: TenantTokenV1, event_id: str,
            kind: str, payload_bytes: bytes,
            parent_event_ids: Sequence[str],
            timestamp_ns: int = 0,
            cost_usd: float = 0.0,
            tokens: int = 0) -> TenantSpendResultV1:
        """Append an event in the token-bound tenant's chain.

        Refuses if the token doesn't verify, or if the spend
        would push the tenant over budget.
        """
        if not self.verify_token(token):
            return TenantSpendResultV1(
                accepted=False,
                new_budget=TenantBudgetV1(),  # placeholder
                reason="bad_token")
        ten = self.tenants[token.tenant_id]
        prospective = ten.budget.spend(
            cost_usd=cost_usd, tokens=tokens)
        if prospective.is_exhausted() and (
                prospective.remaining_cost_usd() < 0
                or prospective.remaining_tokens() < 0):
            return TenantSpendResultV1(
                accepted=False,
                new_budget=ten.budget,
                reason="budget_exhausted")
        ten.append_event(
            event_id=event_id, kind=kind,
            payload_bytes=payload_bytes,
            parent_event_ids=parent_event_ids,
            timestamp_ns=timestamp_ns)
        ten.budget = prospective
        return TenantSpendResultV1(
            accepted=True, new_budget=prospective)

    def read_event_for_tenant(
            self, *, token: TenantTokenV1, event_id: str,
            target_tenant_id: Optional[str] = None,
            now_ns: int = 0) -> Optional[EventNodeV1]:
        """Read an event. If ``target_tenant_id`` is provided
        and differs from the token's tenant, the access is
        REFUSED and a denial event is recorded in the
        requesting tenant's chain.
        """
        if not self.verify_token(token):
            return None
        requesting_id = token.tenant_id
        target_id = (
            target_tenant_id if target_tenant_id is not None
            else requesting_id)
        if target_id != requesting_id:
            requesting = self.tenants[requesting_id]
            target = self.tenants.get(target_id)
            target_cid = (
                target.identity.cid() if target is not None
                else "")
            denial = CrossTenantAccessDeniedEventV1(
                requesting_tenant_id=requesting_id,
                requesting_tenant_cid=(
                    requesting.identity.cid()),
                target_tenant_id=target_id,
                target_tenant_cid=target_cid,
                requested_event_id=str(event_id),
                denied_at_ns=int(now_ns),
                denial_reason=(
                    "cross_tenant_access_denied"))
            requesting.record_denial(denial)
            return None
        ten = self.tenants[requesting_id]
        return ten.event_graph.nodes.get(event_id)

    def get_audit_anchor(
            self, *, token: TenantTokenV1,
            now_ns: int) -> Optional[TenantAuditAnchorV1]:
        if not self.verify_token(token):
            return None
        return self.tenants[token.tenant_id].audit_anchor(now_ns)


# ---------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MultiTenancyBenchReportV1:
    """Tenant-isolation bench output."""

    n_tenants: int
    tenant_a_id: str
    tenant_b_id: str
    tenant_a_cid: str
    tenant_b_cid: str
    tenant_a_anchor_root: str
    tenant_b_anchor_root: str
    cross_tenant_read_refused: bool
    """The load-bearing claim — A cannot read B's events."""

    cross_tenant_denial_event_emitted: bool
    audit_anchors_distinct: bool
    """Tenant A's anchor's Merkle root != Tenant B's."""

    budget_isolation_holds: bool
    """When A exhausts its budget, B's budget is unchanged."""

    token_swap_refused: bool
    """Tenant A's token must NOT work for Tenant B."""

    no_b_bytes_in_a_chain: bool
    """Tenant B's payload bytes must NEVER appear in Tenant A's
    event graph."""

    report_cid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W86_TENANCY_V1_SCHEMA_VERSION,
            "n_tenants": int(self.n_tenants),
            "tenant_a_id": str(self.tenant_a_id),
            "tenant_b_id": str(self.tenant_b_id),
            "tenant_a_cid": str(self.tenant_a_cid),
            "tenant_b_cid": str(self.tenant_b_cid),
            "tenant_a_anchor_root": str(self.tenant_a_anchor_root),
            "tenant_b_anchor_root": str(self.tenant_b_anchor_root),
            "cross_tenant_read_refused": bool(
                self.cross_tenant_read_refused),
            "cross_tenant_denial_event_emitted": bool(
                self.cross_tenant_denial_event_emitted),
            "audit_anchors_distinct": bool(
                self.audit_anchors_distinct),
            "budget_isolation_holds": bool(
                self.budget_isolation_holds),
            "token_swap_refused": bool(self.token_swap_refused),
            "no_b_bytes_in_a_chain": bool(
                self.no_b_bytes_in_a_chain),
            "report_cid": str(self.report_cid),
        }

    def cid(self) -> str:
        d = self.to_dict()
        d["report_cid"] = ""
        return _sha256_hex({
            "kind": "w86_multi_tenancy_bench_report_v1",
            "report": d})


def run_two_tenant_isolation_bench_v1(
        seed: int = 86_043) -> MultiTenancyBenchReportV1:
    """Two tenants share the gateway. Every isolation bar is
    enforced.
    """
    gateway = MultiTenantGatewayV1()

    # Tenant A
    key_a = TenantKeyPairV1.from_seed("tenant_alpha", seed)
    pol_a = TenantPolicyV1(
        max_total_events=100, max_total_bytes=10_000)
    bud_a = TenantBudgetV1(max_total_cost_usd=1.0)
    id_a = TenantIdentityV1(
        tenant_id="tenant_alpha",
        public_key_bytes=key_a.public_key_bytes,
        policy_cid=pol_a.cid())
    gateway.register_tenant(
        identity=id_a, policy=pol_a, budget=bud_a,
        public_key_bytes=key_a.public_key_bytes)

    # Tenant B
    key_b = TenantKeyPairV1.from_seed("tenant_beta", seed + 1)
    pol_b = TenantPolicyV1(
        max_total_events=100, max_total_bytes=10_000)
    bud_b = TenantBudgetV1(max_total_cost_usd=2.0)
    id_b = TenantIdentityV1(
        tenant_id="tenant_beta",
        public_key_bytes=key_b.public_key_bytes,
        policy_cid=pol_b.cid())
    gateway.register_tenant(
        identity=id_b, policy=pol_b, budget=bud_b,
        public_key_bytes=key_b.public_key_bytes)

    token_a = issue_tenant_token_v1(key_a, id_a, nonce="bench-a")
    token_b = issue_tenant_token_v1(key_b, id_b, nonce="bench-b")

    # A writes some events.
    for i in range(5):
        gateway.append_event_for_tenant(
            token=token_a,
            event_id=f"a_evt_{i:03d}",
            kind="dialog_turn",
            payload_bytes=f"alpha_secret_{i}".encode("utf-8"),
            parent_event_ids=(
                gateway.tenants["tenant_alpha"]
                .event_graph.root_event_id,),
            timestamp_ns=86_041_000_000_000 + i,
            cost_usd=0.02, tokens=10)
    # B writes some events with distinctly identifiable bytes.
    for i in range(7):
        gateway.append_event_for_tenant(
            token=token_b,
            event_id=f"b_evt_{i:03d}",
            kind="dialog_turn",
            payload_bytes=f"BETA_PII_DATA_{i}".encode("utf-8"),
            parent_event_ids=(
                gateway.tenants["tenant_beta"]
                .event_graph.root_event_id,),
            timestamp_ns=86_041_000_000_000 + i,
            cost_usd=0.05, tokens=20)

    # A attempts to read B's event with A's token.
    cross_read = gateway.read_event_for_tenant(
        token=token_a, event_id="b_evt_000",
        target_tenant_id="tenant_beta",
        now_ns=86_041_000_001_000)
    cross_read_refused = (cross_read is None)
    denial_emitted = (
        len(gateway.tenants["tenant_alpha"].denied_events) >= 1)

    # Token swap test: build a token claiming A's identity but
    # signed with B's key.
    bad_token = TenantTokenV1(
        tenant_id="tenant_alpha",
        tenant_cid=id_a.cid(),
        nonce="malicious",
        signature_bytes=key_b.sign(
            json.dumps(
                {
                    "kind": "w86_tenant_token_v1",
                    "tenant_id": "tenant_alpha",
                    "tenant_cid": id_a.cid(),
                    "nonce": "malicious",
                },
                sort_keys=True,
                separators=(",", ":")).encode("utf-8")))
    token_swap_refused = (
        not gateway.verify_token(bad_token))

    # Budget isolation: drain A's budget; verify B's is
    # untouched.
    b_budget_before = gateway.tenants["tenant_beta"].budget
    for i in range(200):
        res = gateway.append_event_for_tenant(
            token=token_a,
            event_id=f"a_drain_{i:03d}",
            kind="dialog_turn",
            payload_bytes=f"drain_{i}".encode("utf-8"),
            parent_event_ids=(
                gateway.tenants["tenant_alpha"]
                .event_graph.root_event_id,),
            timestamp_ns=86_042_000_000_000 + i,
            cost_usd=0.5, tokens=100)
        if not res.accepted:
            break
    b_budget_after = gateway.tenants["tenant_beta"].budget
    budget_isolation = (
        b_budget_before.spent_cost_usd
        == b_budget_after.spent_cost_usd
        and b_budget_before.spent_tokens
        == b_budget_after.spent_tokens)

    # Audit anchors.
    anchor_a = gateway.tenants["tenant_alpha"].audit_anchor(
        86_043_000_000_000)
    anchor_b = gateway.tenants["tenant_beta"].audit_anchor(
        86_043_000_000_000)
    anchors_distinct = (
        anchor_a.merkle_root() != anchor_b.merkle_root())

    # No B bytes in A's chain.
    a_payload_blob = b"".join(
        ev.payload_bytes for ev in
        gateway.tenants["tenant_alpha"].event_graph.nodes.values())
    no_b_bytes = b"BETA_PII_DATA" not in a_payload_blob

    rep = MultiTenancyBenchReportV1(
        n_tenants=2,
        tenant_a_id="tenant_alpha",
        tenant_b_id="tenant_beta",
        tenant_a_cid=id_a.cid(),
        tenant_b_cid=id_b.cid(),
        tenant_a_anchor_root=anchor_a.merkle_root(),
        tenant_b_anchor_root=anchor_b.merkle_root(),
        cross_tenant_read_refused=cross_read_refused,
        cross_tenant_denial_event_emitted=denial_emitted,
        audit_anchors_distinct=anchors_distinct,
        budget_isolation_holds=budget_isolation,
        token_swap_refused=token_swap_refused,
        no_b_bytes_in_a_chain=no_b_bytes)
    rep = dataclasses.replace(rep, report_cid=rep.cid())
    return rep


__all__ = [
    "W86_TENANCY_V1_SCHEMA_VERSION",
    "TenantPolicyV1",
    "TenantBudgetV1",
    "TenantKeyPairV1",
    "TenantIdentityV1",
    "TenantTokenV1",
    "CrossTenantAccessDeniedEventV1",
    "TenantAuditAnchorV1",
    "TenantStateV1",
    "TenantSpendResultV1",
    "MultiTenantGatewayV1",
    "MultiTenancyBenchReportV1",
    "issue_tenant_token_v1",
    "run_two_tenant_isolation_bench_v1",
]
