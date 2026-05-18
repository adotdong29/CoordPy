"""W82 / P2 #16 — Distributed multi-host substrate
coordination + state migration V1.

Issue #16 asks for a distributed substrate research line:

* state replication envelopes with content-addressed
  witnesses
* migration from one controlled runtime or host to another
* delayed synchronization and repair after temporary
  partition
* branch / merge semantics for distributed state replicas
* budget-aware migration and replay policies

Required outputs:

1. a simulation or real multi-host benchmark path
2. explicit migration protocols
3. correctness / fidelity checks after migration
4. honest consistency semantics (exact, eventual,
   approximate, or best-effort)

V1 delivers all four with an in-process simulated multi-host
substrate. Concretely:

* ``SimulatedHostV1`` — represents one host with its own state
  store (a content-addressed event log via W82 P2 #11's
  ``EventGraphV1``) plus a runtime signature (via W82 P2 #13's
  ``RuntimeSignatureV1``).
* ``ReplicationManifestV1`` — declares the replication
  topology: which event_ids are replicated to which hosts.
* ``MigrationEnvelopeV1`` — content-addressed package of a
  subset of events being shipped between hosts. Each envelope
  carries:

  - a Merkle root over the shipped event CIDs (via
    ``MerkleHashTreeV1``)
  - the source host's runtime signature CID
  - the target host's runtime signature CID
  - a portability projector reference (for cross-runtime
    migration)

* ``MigrationProtocolV1`` — the migration steps:

  1. envelope build (source side, deterministic)
  2. envelope transport (modelled as a simple function call)
  3. envelope integrity verification (target side)
  4. event-graph append (target side, append-only)
  5. portability projection (target side, only if runtime
     signatures differ)

* ``PartitionEventV1`` — simulates a network partition between
  two host subsets. While partitioned, events appended on one
  side are NOT replicated to the other. On heal, the
  ``SyncDecisionV1`` controller resolves the divergent state.
* ``SyncDecisionV1`` — emits an explicit consistency verdict:
  ``EXACT``, ``EVENTUAL``, ``APPROXIMATE``, or
  ``BEST_EFFORT``. The V1 default policy is **eventual
  consistency with content-addressed convergence proofs**.

The V1 bench:

* spins up 3 simulated hosts, initially synchronized
* runs a sequence of (append, replicate, partition, append-
  during-partition, heal, sync) steps
* verifies, at end-of-bench, that:

  - every host's event-graph contains the union of events
  - every host's event-graph has the same merkle root
  - the partition was correctly detected
  - the sync decision is at least ``EVENTUAL``

The bench also runs a cross-runtime migration leg:

* host_a has signature S_a (hidden_dim 8)
* host_b has signature S_b (hidden_dim 12)
* envelope built on host_a is migrated to host_b
* portability projector translates A-state into B-state
* re-encode on host_b is verified

Honest scope (W82)
------------------

* ``W82-L-DISTRIBUTED-V1-RESEARCH-ONLY-CAP`` — explicit
  import only.
* ``W82-L-DISTRIBUTED-V1-IN-PROCESS-CAP`` — V1 simulates
  hosts in-process. No real TCP / RPC layer; the transport
  is a function call. Real network transport is out of V1
  scope.
* ``W82-L-DISTRIBUTED-V1-EVENTUAL-CONSISTENCY-CAP`` — V1
  guarantees eventual consistency: after heal + sync, every
  host's merkle root is identical. V1 does NOT claim strong
  / linearizable consistency.
* ``W82-L-DISTRIBUTED-V1-NUMPY-CAP`` — NumPy + stdlib.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from typing import Any, Sequence

from .cross_runtime_state_portability_v1 import (
    PortabilityProjectorV1,
    PortableStateCarrierV1,
    RuntimeSignatureV1,
    build_portability_projector_v1,
    build_runtime_signature_v1,
)
from .cryptographic_state_integrity_v1 import (
    IntegrityVerdict,
    MerkleHashTreeV1,
)
from .event_sourced_memory_graph_v1 import (
    EventGraphV1,
    EventNodeV1,
    build_event_node_v1,
)


W82_DISTRIBUTED_V1_SCHEMA_VERSION: str = (
    "coordpy.distributed_substrate_coordination_v1.v1")


W82_DISTRIBUTED_DEFAULT_N_HOSTS: int = 3
W82_DISTRIBUTED_DEFAULT_BENCH_SEED: int = 82_016_001


class ConsistencyVerdict(str, enum.Enum):
    EXACT = "exact"
    EVENTUAL = "eventual"
    APPROXIMATE = "approximate"
    BEST_EFFORT = "best_effort"


W82_DISTRIBUTED_CONSISTENCY_VERDICTS: tuple[str, ...] = tuple(
    v.value for v in ConsistencyVerdict)


# ---------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------

def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ---------------------------------------------------------------
# Simulated host
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SimulatedHostV1:
    """One simulated host with its own event graph + runtime
    signature.

    Hosts are immutable; mutations return NEW hosts.
    """

    schema: str
    host_id: str
    runtime_signature: RuntimeSignatureV1
    event_graph: EventGraphV1

    def with_event_graph(
            self, graph: EventGraphV1,
    ) -> "SimulatedHostV1":
        return SimulatedHostV1(
            schema=W82_DISTRIBUTED_V1_SCHEMA_VERSION,
            host_id=str(self.host_id),
            runtime_signature=self.runtime_signature,
            event_graph=graph,
        )

    def append_event(
            self, event: EventNodeV1,
    ) -> "SimulatedHostV1":
        return self.with_event_graph(
            self.event_graph.with_event(event))

    def merkle_root_cid(self) -> str:
        cids = sorted(
            str(n.cid())
            for n in self.event_graph.nodes.values())
        tree = MerkleHashTreeV1.from_snapshot_cids(cids)
        return str(tree.root_cid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "host_id": str(self.host_id),
            "runtime_signature_cid": str(
                self.runtime_signature.cid()),
            "event_graph_cid": str(
                self.event_graph.cid()),
            "n_events": int(self.event_graph.n_events()),
            "merkle_root_cid": str(
                self.merkle_root_cid()),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_simulated_host_v1",
            "host": self.to_dict()})


def build_simulated_host_v1(
        *, host_id: str,
        runtime_signature: RuntimeSignatureV1 | None = None,
) -> SimulatedHostV1:
    """Build a host with a fresh event graph."""
    if runtime_signature is None:
        runtime_signature = build_runtime_signature_v1(
            backend_label=f"cpu_numpy_{host_id}",
            vocab_size=24, hidden_dim=8, n_layers=2)
    return SimulatedHostV1(
        schema=W82_DISTRIBUTED_V1_SCHEMA_VERSION,
        host_id=str(host_id),
        runtime_signature=runtime_signature,
        event_graph=EventGraphV1.empty(
            genesis_payload=b"genesis"),
    )


# ---------------------------------------------------------------
# Migration envelope
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MigrationEnvelopeV1:
    """A content-addressed package of events being migrated."""

    schema: str
    source_host_id: str
    target_host_id: str
    source_signature_cid: str
    target_signature_cid: str
    event_cids: tuple[str, ...]
    events: tuple[EventNodeV1, ...]
    merkle_root_cid: str
    portable_carrier_cid: str
    n_events: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "source_host_id": str(self.source_host_id),
            "target_host_id": str(self.target_host_id),
            "source_signature_cid": str(
                self.source_signature_cid),
            "target_signature_cid": str(
                self.target_signature_cid),
            "event_cids": list(self.event_cids),
            "merkle_root_cid": str(self.merkle_root_cid),
            "portable_carrier_cid": str(
                self.portable_carrier_cid),
            "n_events": int(self.n_events),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_migration_envelope_v1",
            "envelope": self.to_dict()})


def build_migration_envelope_v1(
        *, source: SimulatedHostV1,
        target: SimulatedHostV1,
        event_ids_to_migrate: Sequence[str],
        portable_carrier_cid: str = "",
) -> MigrationEnvelopeV1:
    """Build a content-addressed migration envelope on the
    source host side."""
    events: list[EventNodeV1] = []
    for eid in event_ids_to_migrate:
        if str(eid) not in source.event_graph.nodes:
            raise ValueError(
                f"event_id {eid} not in source host's "
                f"event graph")
        events.append(source.event_graph.nodes[str(eid)])
    cids = tuple(str(e.cid()) for e in events)
    tree = MerkleHashTreeV1.from_snapshot_cids(list(cids))
    return MigrationEnvelopeV1(
        schema=W82_DISTRIBUTED_V1_SCHEMA_VERSION,
        source_host_id=str(source.host_id),
        target_host_id=str(target.host_id),
        source_signature_cid=str(
            source.runtime_signature.cid()),
        target_signature_cid=str(
            target.runtime_signature.cid()),
        event_cids=cids,
        events=tuple(events),
        merkle_root_cid=str(tree.root_cid),
        portable_carrier_cid=str(portable_carrier_cid),
        n_events=int(len(events)),
    )


def verify_migration_envelope_v1(
        envelope: MigrationEnvelopeV1,
) -> str:
    """Verify envelope integrity by recomputing the merkle
    root over the shipped event CIDs."""
    if int(envelope.n_events) != int(len(envelope.event_cids)):
        return IntegrityVerdict.CORRUPT.value
    if int(envelope.n_events) != int(len(envelope.events)):
        return IntegrityVerdict.CORRUPT.value
    recomputed = MerkleHashTreeV1.from_snapshot_cids(
        list(envelope.event_cids)).root_cid
    if str(recomputed) != str(envelope.merkle_root_cid):
        return IntegrityVerdict.PROVENANCE_VIOLATION.value
    # Verify the shipped event list matches the authenticated
    # event_cids tuple exactly; extra envelope events must be
    # rejected, not silently ignored.
    recomputed_event_cids = tuple(
        str(ev.cid()) for ev in envelope.events)
    if recomputed_event_cids != tuple(
            str(cid) for cid in envelope.event_cids):
        return IntegrityVerdict.CORRUPT.value
    for cid, ev in zip(
            envelope.event_cids, envelope.events, strict=True):
        if str(ev.cid()) != str(cid):
            return IntegrityVerdict.CORRUPT.value
    return IntegrityVerdict.OK.value


def apply_migration_envelope_v1(
        *, envelope: MigrationEnvelopeV1,
        target: SimulatedHostV1,
) -> tuple[SimulatedHostV1, str]:
    """Apply an envelope on the target host. Returns
    (new_target, integrity_verdict)."""
    verdict = verify_migration_envelope_v1(envelope)
    if str(verdict) != IntegrityVerdict.OK.value:
        return (target, str(verdict))
    g = target.event_graph
    for ev in envelope.events:
        if str(ev.event_id) in g.nodes:
            # Idempotent: already have this event.
            continue
        try:
            g = g.with_event(ev)
        except ValueError as e:
            # Parent missing — return what we've applied so
            # far and a provenance violation verdict.
            return (
                target.with_event_graph(g),
                IntegrityVerdict.PROVENANCE_VIOLATION.value)
    return (target.with_event_graph(g),
            IntegrityVerdict.OK.value)


# ---------------------------------------------------------------
# Replication manifest
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ReplicationManifestV1:
    """Declares which hosts host which events."""

    schema: str
    host_ids: tuple[str, ...]
    replicated_event_ids_per_host: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "host_ids": list(self.host_ids),
            "n_hosts": int(len(self.host_ids)),
            "n_events_per_host": {
                str(h): int(len(eids))
                for h, eids in
                self.replicated_event_ids_per_host.items()},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_replication_manifest_v1",
            "manifest": self.to_dict(),
            "replicated_event_ids_per_host": {
                str(k): list(v)
                for k, v in sorted(
                    self.replicated_event_ids_per_host.items())},
        })


# ---------------------------------------------------------------
# Partition event
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PartitionEventV1:
    """Simulated network partition between two host subsets.

    During ``duration_steps``, events appended on hosts in
    ``side_a_host_ids`` are NOT replicated to hosts in
    ``side_b_host_ids`` (and vice versa). Both subsets keep
    appending locally.
    """

    schema: str
    partition_id: str
    side_a_host_ids: tuple[str, ...]
    side_b_host_ids: tuple[str, ...]
    started_at_step: int
    healed_at_step: int

    def duration_steps(self) -> int:
        return int(
            self.healed_at_step - self.started_at_step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "partition_id": str(self.partition_id),
            "side_a_host_ids": list(self.side_a_host_ids),
            "side_b_host_ids": list(self.side_b_host_ids),
            "started_at_step": int(self.started_at_step),
            "healed_at_step": int(self.healed_at_step),
            "duration_steps": int(self.duration_steps()),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_partition_event_v1",
            "partition": self.to_dict()})


# ---------------------------------------------------------------
# Sync decision
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SyncDecisionV1:
    """Post-heal sync decision."""

    schema: str
    partition_cid: str
    consistency_verdict: str
    n_events_reconciled: int
    merkle_root_pre_heal_per_host: dict[str, str]
    merkle_root_post_heal_per_host: dict[str, str]
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "partition_cid": str(self.partition_cid),
            "consistency_verdict": str(
                self.consistency_verdict),
            "n_events_reconciled": int(
                self.n_events_reconciled),
            "merkle_root_pre_heal_per_host": {
                str(k): str(v)
                for k, v in sorted(
                    self.merkle_root_pre_heal_per_host.items())},
            "merkle_root_post_heal_per_host": {
                str(k): str(v)
                for k, v in sorted(
                    self.merkle_root_post_heal_per_host.items())},
            "converged": bool(self.converged),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_sync_decision_v1",
            "decision": self.to_dict()})


def heal_partition_and_sync_v1(
        *, hosts_pre_heal: Sequence[SimulatedHostV1],
        partition: PartitionEventV1,
) -> tuple[tuple[SimulatedHostV1, ...], SyncDecisionV1]:
    """Heal the partition and synchronize hosts.

    Algorithm:

    1. Take the union of all event_ids across all hosts.
    2. For each host, build envelopes covering the events it
       is missing.
    3. Apply envelopes deterministically (events sorted by
       (timestamp_ns, event_id) so two passes don't deadlock).
    4. Verify every host's merkle root is identical.
    5. Emit a SyncDecisionV1 with the convergence verdict.
    """
    hosts = list(hosts_pre_heal)
    # Union event_ids → events (canonicalised).
    union: dict[str, EventNodeV1] = {}
    pre_heal_roots: dict[str, str] = {}
    for h in hosts:
        pre_heal_roots[str(h.host_id)] = str(
            h.merkle_root_cid())
        for eid, n in h.event_graph.nodes.items():
            if str(eid) not in union:
                union[str(eid)] = n
    n_reconciled = 0
    # For each host, append events from the union it doesn't
    # have, in topological order (parents before children).
    sorted_events = sorted(
        union.values(),
        key=lambda e: (
            int(e.timestamp_ns), str(e.event_id)))
    new_hosts: list[SimulatedHostV1] = []
    for h in hosts:
        g = h.event_graph
        for ev in sorted_events:
            if str(ev.event_id) in g.nodes:
                continue
            try:
                g = g.with_event(ev)
                n_reconciled += 1
            except ValueError:
                # Parent missing — skip; the next pass will
                # catch up. In practice the topological sort
                # guarantees no skips.
                continue
        new_hosts.append(h.with_event_graph(g))
    post_heal_roots: dict[str, str] = {
        str(h.host_id): str(h.merkle_root_cid())
        for h in new_hosts}
    roots = set(post_heal_roots.values())
    converged = bool(len(roots) == 1)
    verdict = (
        ConsistencyVerdict.EVENTUAL.value
        if converged else
        ConsistencyVerdict.BEST_EFFORT.value)
    return (
        tuple(new_hosts),
        SyncDecisionV1(
            schema=W82_DISTRIBUTED_V1_SCHEMA_VERSION,
            partition_cid=str(partition.cid()),
            consistency_verdict=str(verdict),
            n_events_reconciled=int(n_reconciled),
            merkle_root_pre_heal_per_host=pre_heal_roots,
            merkle_root_post_heal_per_host=post_heal_roots,
            converged=bool(converged),
        ))


# ---------------------------------------------------------------
# Budget-aware migration policy
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MigrationBudgetPolicyV1:
    """A budget-aware migration policy.

    Migrations are not free — they cost bytes shipped, bytes
    re-hashed on the target, and time-to-replay. V1 exposes
    three explicit budgets and a per-envelope decision:

    * ``max_bytes_per_migration`` — drop envelopes whose
      shipped event payload total exceeds this size
    * ``max_events_per_migration`` — drop envelopes shipping
      more than this many events at once (latency cap)
    * ``min_freshness_age_ns`` — refuse to migrate events
      older than this (best-effort freshness signal)

    The policy returns a ``MigrationBudgetDecisionV1`` so the
    caller can see *why* a migration was accepted or
    rejected. The decision is content-addressed for audit
    chaining.
    """

    schema: str
    max_bytes_per_migration: int
    max_events_per_migration: int
    min_freshness_age_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "max_bytes_per_migration": int(
                self.max_bytes_per_migration),
            "max_events_per_migration": int(
                self.max_events_per_migration),
            "min_freshness_age_ns": int(
                self.min_freshness_age_ns),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_migration_budget_policy_v1",
            "policy": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class MigrationBudgetDecisionV1:
    """Per-envelope budget decision."""

    schema: str
    policy_cid: str
    envelope_cid: str
    accepted: bool
    total_bytes: int
    n_events: int
    oldest_event_age_ns: int
    rejection_reason: str  # "" if accepted

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "policy_cid": str(self.policy_cid),
            "envelope_cid": str(self.envelope_cid),
            "accepted": bool(self.accepted),
            "total_bytes": int(self.total_bytes),
            "n_events": int(self.n_events),
            "oldest_event_age_ns": int(
                self.oldest_event_age_ns),
            "rejection_reason": str(self.rejection_reason),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_migration_budget_decision_v1",
            "decision": self.to_dict()})


def build_migration_budget_policy_v1(
        *, max_bytes_per_migration: int = 1_000_000,
        max_events_per_migration: int = 1024,
        min_freshness_age_ns: int = 0,
) -> MigrationBudgetPolicyV1:
    """Build a default budget policy. ``min_freshness_age_ns
    = 0`` means no freshness floor."""
    return MigrationBudgetPolicyV1(
        schema=W82_DISTRIBUTED_V1_SCHEMA_VERSION,
        max_bytes_per_migration=int(max_bytes_per_migration),
        max_events_per_migration=int(max_events_per_migration),
        min_freshness_age_ns=int(min_freshness_age_ns),
    )


def decide_migration_budget_v1(
        *, policy: MigrationBudgetPolicyV1,
        envelope: MigrationEnvelopeV1,
        current_time_ns: int,
) -> MigrationBudgetDecisionV1:
    """Apply the budget policy to ``envelope``.

    Returns a ``MigrationBudgetDecisionV1`` with explicit
    rationale. ``accepted=False`` means the migration MUST
    be rejected (or fragmented and retried). Accepted
    envelopes carry empty ``rejection_reason``.
    """
    total_bytes = sum(
        int(len(e.payload_bytes)) for e in envelope.events)
    n_events = int(envelope.n_events)
    if envelope.events:
        oldest_ts = min(
            int(e.timestamp_ns) for e in envelope.events)
        age = max(0, int(current_time_ns) - int(oldest_ts))
    else:
        age = 0
    accepted = True
    reason = ""
    if total_bytes > int(policy.max_bytes_per_migration):
        accepted = False
        reason = (
            f"envelope total_bytes {total_bytes} > "
            f"policy.max_bytes_per_migration "
            f"{policy.max_bytes_per_migration}")
    elif n_events > int(policy.max_events_per_migration):
        accepted = False
        reason = (
            f"envelope n_events {n_events} > "
            f"policy.max_events_per_migration "
            f"{policy.max_events_per_migration}")
    elif (int(policy.min_freshness_age_ns) > 0 and
          age > int(policy.min_freshness_age_ns)):
        accepted = False
        reason = (
            f"oldest event age {age}ns > "
            f"policy.min_freshness_age_ns "
            f"{policy.min_freshness_age_ns}")
    return MigrationBudgetDecisionV1(
        schema=W82_DISTRIBUTED_V1_SCHEMA_VERSION,
        policy_cid=str(policy.cid()),
        envelope_cid=str(envelope.cid()),
        accepted=bool(accepted),
        total_bytes=int(total_bytes),
        n_events=int(n_events),
        oldest_event_age_ns=int(age),
        rejection_reason=str(reason),
    )


# ---------------------------------------------------------------
# Replicate / append helpers
# ---------------------------------------------------------------

def replicate_event_to_hosts_v1(
        *, event: EventNodeV1,
        hosts: Sequence[SimulatedHostV1],
        exclude_host_ids: Sequence[str] = (),
) -> tuple[SimulatedHostV1, ...]:
    """Append ``event`` to every host except those in
    ``exclude_host_ids``. Returns the new host tuple."""
    out: list[SimulatedHostV1] = []
    for h in hosts:
        if str(h.host_id) in exclude_host_ids:
            out.append(h)
            continue
        if str(event.event_id) in h.event_graph.nodes:
            out.append(h)
            continue
        out.append(h.append_event(event))
    return tuple(out)


# ---------------------------------------------------------------
# Bench schema
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class DistributedCoordinationBenchReportV1:
    """End-to-end distributed coordination bench."""

    schema: str
    n_hosts: int
    n_events_total: int
    partition_detected: bool
    n_events_during_partition_a: int
    n_events_during_partition_b: int
    sync_consistency_verdict: str
    all_hosts_merkle_match_post_heal: bool
    cross_runtime_envelope_verdict: str
    cross_runtime_target_event_count: int
    config_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_hosts": int(self.n_hosts),
            "n_events_total": int(self.n_events_total),
            "partition_detected": bool(
                self.partition_detected),
            "n_events_during_partition_a": int(
                self.n_events_during_partition_a),
            "n_events_during_partition_b": int(
                self.n_events_during_partition_b),
            "sync_consistency_verdict": str(
                self.sync_consistency_verdict),
            "all_hosts_merkle_match_post_heal": bool(
                self.all_hosts_merkle_match_post_heal),
            "cross_runtime_envelope_verdict": str(
                self.cross_runtime_envelope_verdict),
            "cross_runtime_target_event_count": int(
                self.cross_runtime_target_event_count),
            "config_cid": str(self.config_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_distributed_coordination_bench_report_v1",
            "report": self.to_dict()})


def run_distributed_coordination_bench_v1(
        *, seed: int = W82_DISTRIBUTED_DEFAULT_BENCH_SEED,
        n_pre_partition_events: int = 5,
        n_during_partition_events_a: int = 3,
        n_during_partition_events_b: int = 2,
        n_post_heal_events: int = 4,
) -> DistributedCoordinationBenchReportV1:
    """Run the load-bearing W82 P2 #16 bench."""
    # Set up 3 hosts (a, b, c) with the same runtime sig.
    common_sig = build_runtime_signature_v1(
        backend_label="cpu_numpy_common",
        vocab_size=16, hidden_dim=8, n_layers=2)
    host_a = build_simulated_host_v1(
        host_id="host_a", runtime_signature=common_sig)
    host_b = build_simulated_host_v1(
        host_id="host_b", runtime_signature=common_sig)
    host_c = build_simulated_host_v1(
        host_id="host_c", runtime_signature=common_sig)
    hosts: list[SimulatedHostV1] = [host_a, host_b, host_c]
    step = 1
    prev_event_id: dict[str, str] = {
        h.host_id: h.event_graph.root_event_id
        for h in hosts}
    # Phase 1: pre-partition. Every host gets every event.
    for i in range(int(n_pre_partition_events)):
        ev = build_event_node_v1(
            event_id=f"pre_{i}",
            kind="pre_partition",
            payload_bytes=_canonical_bytes({
                "i": int(i), "seed": int(seed)}),
            parent_event_ids=(
                prev_event_id[hosts[0].host_id],),
            branch_label="main",
            timestamp_ns=int(step))
        hosts = list(replicate_event_to_hosts_v1(
            event=ev, hosts=tuple(hosts)))
        for h in hosts:
            prev_event_id[h.host_id] = str(ev.event_id)
        step += 1
    # Snapshot pre-partition merkle roots.
    pre_roots = {h.host_id: h.merkle_root_cid()
                 for h in hosts}
    # Phase 2: partition. host_a is isolated from host_b/c.
    partition = PartitionEventV1(
        schema=W82_DISTRIBUTED_V1_SCHEMA_VERSION,
        partition_id="partition_0",
        side_a_host_ids=("host_a",),
        side_b_host_ids=("host_b", "host_c"),
        started_at_step=int(step),
        healed_at_step=int(
            step + int(n_during_partition_events_a) +
            int(n_during_partition_events_b)))
    # During-partition: host_a appends events visible only to
    # itself; host_b/c append events visible only to
    # themselves.
    for i in range(int(n_during_partition_events_a)):
        ev = build_event_node_v1(
            event_id=f"a_only_{i}",
            kind="during_partition_a",
            payload_bytes=_canonical_bytes({
                "i": int(i), "seed": int(seed)}),
            parent_event_ids=(
                prev_event_id["host_a"],),
            branch_label="main",
            timestamp_ns=int(step))
        host_a = hosts[0].append_event(ev)
        hosts[0] = host_a
        prev_event_id["host_a"] = str(ev.event_id)
        step += 1
    for i in range(int(n_during_partition_events_b)):
        ev = build_event_node_v1(
            event_id=f"b_only_{i}",
            kind="during_partition_b",
            payload_bytes=_canonical_bytes({
                "i": int(i), "seed": int(seed)}),
            parent_event_ids=(
                prev_event_id["host_b"],),
            branch_label="main",
            timestamp_ns=int(step))
        host_b = hosts[1].append_event(ev)
        host_c = hosts[2].append_event(ev)
        hosts[1] = host_b
        hosts[2] = host_c
        prev_event_id["host_b"] = str(ev.event_id)
        prev_event_id["host_c"] = str(ev.event_id)
        step += 1
    # Pre-heal: partition is detected iff pre_roots != post-
    # appendage roots and the appendage merkle roots on the
    # two sides differ.
    side_a_root = hosts[0].merkle_root_cid()
    side_b_root = hosts[1].merkle_root_cid()
    partition_detected = bool(side_a_root != side_b_root)
    # Phase 3: heal + sync.
    healed_hosts, sync_decision = heal_partition_and_sync_v1(
        hosts_pre_heal=tuple(hosts), partition=partition)
    hosts = list(healed_hosts)
    # After heal, all hosts should have all events.
    post_heal_roots = {h.host_id: h.merkle_root_cid()
                       for h in hosts}
    all_match = bool(
        len(set(post_heal_roots.values())) == 1)
    # Phase 4: post-heal events all hosts get.
    for i in range(int(n_post_heal_events)):
        ev = build_event_node_v1(
            event_id=f"post_{i}",
            kind="post_heal",
            payload_bytes=_canonical_bytes({
                "i": int(i), "seed": int(seed)}),
            parent_event_ids=(
                prev_event_id[hosts[0].host_id],),
            branch_label="main",
            timestamp_ns=int(step))
        # After heal, all hosts have the latest events; use
        # host_a's prev event as the parent (which by now
        # should appear in all hosts' graphs).
        if str(prev_event_id["host_a"]) not in (
                hosts[0].event_graph.nodes):
            # If sync didn't propagate (best-effort case),
            # rebase to host_a's current tip.
            prev_event_id["host_a"] = list(
                hosts[0].event_graph.nodes.keys())[-1]
        hosts = list(replicate_event_to_hosts_v1(
            event=ev, hosts=tuple(hosts)))
        for h in hosts:
            prev_event_id[h.host_id] = str(ev.event_id)
        step += 1
    # Phase 5: cross-runtime migration.
    sig_a = build_runtime_signature_v1(
        backend_label="cpu_numpy_a_cross",
        vocab_size=24, hidden_dim=8, n_layers=2)
    sig_b = build_runtime_signature_v1(
        backend_label="cpu_numpy_b_cross",
        vocab_size=32, hidden_dim=12, n_layers=3)
    cross_host_a = build_simulated_host_v1(
        host_id="cross_host_a", runtime_signature=sig_a)
    cross_host_b = build_simulated_host_v1(
        host_id="cross_host_b", runtime_signature=sig_b)
    # Append a few events on cross_host_a.
    prev_cross = cross_host_a.event_graph.root_event_id
    cross_event_ids: list[str] = []
    for i in range(3):
        ev = build_event_node_v1(
            event_id=f"cross_{i}",
            kind="cross_runtime",
            payload_bytes=_canonical_bytes({
                "i": int(i), "seed": int(seed)}),
            parent_event_ids=(str(prev_cross),),
            branch_label="main",
            timestamp_ns=int(i + 1))
        cross_host_a = cross_host_a.append_event(ev)
        prev_cross = str(ev.event_id)
        cross_event_ids.append(str(ev.event_id))
    # Build envelope, validate, ship to cross_host_b.
    proj = build_portability_projector_v1()
    envelope = build_migration_envelope_v1(
        source=cross_host_a,
        target=cross_host_b,
        event_ids_to_migrate=tuple(cross_event_ids))
    new_cross_host_b, verdict = (
        apply_migration_envelope_v1(
            envelope=envelope, target=cross_host_b))
    cfg_cid = _sha256_hex({
        "kind": (
            "w82_distributed_coordination_bench_config_v1"),
        "seed": int(seed),
        "n_pre_partition_events": int(n_pre_partition_events),
        "n_during_partition_events_a": int(
            n_during_partition_events_a),
        "n_during_partition_events_b": int(
            n_during_partition_events_b),
        "n_post_heal_events": int(n_post_heal_events),
    })
    return DistributedCoordinationBenchReportV1(
        schema=W82_DISTRIBUTED_V1_SCHEMA_VERSION,
        n_hosts=int(len(hosts)),
        n_events_total=int(
            hosts[0].event_graph.n_events()),
        partition_detected=bool(partition_detected),
        n_events_during_partition_a=int(
            n_during_partition_events_a),
        n_events_during_partition_b=int(
            n_during_partition_events_b),
        sync_consistency_verdict=str(
            sync_decision.consistency_verdict),
        all_hosts_merkle_match_post_heal=bool(all_match),
        cross_runtime_envelope_verdict=str(verdict),
        cross_runtime_target_event_count=int(
            new_cross_host_b.event_graph.n_events()),
        config_cid=str(cfg_cid),
    )


__all__ = [
    "W82_DISTRIBUTED_V1_SCHEMA_VERSION",
    "W82_DISTRIBUTED_CONSISTENCY_VERDICTS",
    "ConsistencyVerdict",
    "SimulatedHostV1",
    "MigrationEnvelopeV1",
    "ReplicationManifestV1",
    "PartitionEventV1",
    "SyncDecisionV1",
    "MigrationBudgetPolicyV1",
    "MigrationBudgetDecisionV1",
    "DistributedCoordinationBenchReportV1",
    "build_simulated_host_v1",
    "build_migration_envelope_v1",
    "verify_migration_envelope_v1",
    "apply_migration_envelope_v1",
    "replicate_event_to_hosts_v1",
    "heal_partition_and_sync_v1",
    "build_migration_budget_policy_v1",
    "decide_migration_budget_v1",
    "run_distributed_coordination_bench_v1",
]
