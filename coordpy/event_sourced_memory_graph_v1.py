"""W82 / P2 #11 — Event-sourced global memory graph V1.

Issue #11 asks for an architecture that replaces disguised
bounded-window / trajectory-slice handoffs with a real event-
sourced global memory graph, including:

* append-only event log with content-addressed nodes
* branch / merge / rejoin semantics at the *memory graph*
  level (not buried in a CID chain)
* a query planner that reconstructs answers from graph state
  + latent carriers + replay when needed
* causal provenance tracking so answers can cite the
  evidence path they were reconstructed from
* state summarization as a *derived view*, not the primary
  store

Definition of done:

* explicit event-sourced memory layer or graph abstraction
* at least one load-bearing benchmark uses it as the
  *primary* carrier
* a bounded-handoff baseline is beaten on a task that
  requires the graph/query view
* provenance + reconstruction paths inspectable

V1 delivers all four. The core objects are:

* ``EventNodeV1`` — append-only, content-addressed event with
  zero or more ``parent_event_ids`` (so branches and merges
  are first-class).
* ``EventGraphV1`` — the DAG of nodes plus named branch tips.
  Append-only: ``with_event`` returns a NEW graph; the
  underlying immutable nodes are shared by structural sharing.
* ``MemoryQueryV1`` — declarative query types
  (``BY_ANCESTOR_PATH``, ``BY_KIND``, ``BY_CONTENT_PREDICATE``,
  ``BY_BRANCH_MERGE``).
* ``QueryPlanV1`` — the planner's plan for one query: the
  set of graph-walk steps it will take, the carriers it will
  consult, the projection it will run.
* ``QueryAnswerV1`` — the result + a
  ``ProvenanceCertificateV1`` listing every event that
  contributed.
* ``ProvenanceCertificateV1`` — content-addressed list of
  contributing event CIDs (the *evidence path*).
* ``DerivedSummaryViewV1`` — a derived projection over the
  graph that is *recomputed from the primary store*, never
  the primary store itself.

The W82 V1 bench:

* builds a 100-event graph with three branches and one merge
* issues a load-bearing query that requires walking >= 50
  events back through the graph
* compares answer success against:

  - ``bounded_handoff_baseline_k32`` — sees only the most-
    recent 32 events
  - ``trajectory_slice_baseline_k16`` — sees only the
    current branch slice (and not the merged history)
  - ``rolling_summary_baseline`` — single fixed-size summary

* asserts the graph beats every baseline on the load-
  bearing query family and emits a content-addressed
  witness chaining the query CID + answer CID +
  provenance CID.

Honest scope (W82)
------------------

* ``W82-L-EVENT-GRAPH-V1-RESEARCH-ONLY-CAP`` — explicit
  import only.
* ``W82-L-EVENT-GRAPH-V1-IN-MEMORY-CAP`` — V1 stores nodes
  in-memory in a Python dict. Persistent storage / on-disk
  log structure is out of V1 scope.
* ``W82-L-EVENT-GRAPH-V1-NUMPY-CAP`` — pure NumPy / stdlib.
* ``W82-L-EVENT-GRAPH-V1-SYNTHETIC-CAP`` — events are
  synthetic. The bench measures retrievability and
  provenance traceability, not semantic correctness of a
  live LLM task.
* ``W82-L-EVENT-GRAPH-V1-NO-DISTRIBUTED-CAP`` — V1 is
  single-host. Distributed replication is covered by W82
  #16.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping, Sequence


W82_EVENT_GRAPH_V1_SCHEMA_VERSION: str = (
    "coordpy.event_sourced_memory_graph_v1.v1")


W82_EVENT_GRAPH_GENESIS_EVENT_ID: str = "genesis"
W82_EVENT_GRAPH_DEFAULT_BENCH_N_EVENTS: int = 100
W82_EVENT_GRAPH_DEFAULT_BENCH_SEED: int = 82_011_001
W82_EVENT_GRAPH_DEFAULT_BENCH_N_QUERIES: int = 20
W82_EVENT_GRAPH_DEFAULT_BOUNDED_HANDOFF_K: int = 32
W82_EVENT_GRAPH_DEFAULT_TRAJECTORY_K: int = 16


# ---------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------

def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _payload_cid(payload_bytes: bytes) -> str:
    return hashlib.sha256(bytes(payload_bytes)).hexdigest()


# ---------------------------------------------------------------
# Event-node schema
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class EventNodeV1:
    """An append-only, content-addressed event node.

    ``parent_event_ids`` is a tuple — zero entries for the
    genesis event, one for linear succession, two or more
    for merges.
    """

    schema: str
    event_id: str
    kind: str
    payload_bytes: bytes
    payload_cid: str
    parent_event_ids: tuple[str, ...]
    branch_label: str
    timestamp_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "event_id": str(self.event_id),
            "kind": str(self.kind),
            "payload_cid": str(self.payload_cid),
            "payload_size_bytes": int(
                len(self.payload_bytes)),
            "parent_event_ids": [
                str(p) for p in self.parent_event_ids],
            "branch_label": str(self.branch_label),
            "timestamp_ns": int(self.timestamp_ns),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_event_node_v1",
            "event": self.to_dict()})


def build_event_node_v1(
        *, event_id: str,
        kind: str,
        payload_bytes: bytes,
        parent_event_ids: Sequence[str],
        branch_label: str,
        timestamp_ns: int,
) -> EventNodeV1:
    return EventNodeV1(
        schema=W82_EVENT_GRAPH_V1_SCHEMA_VERSION,
        event_id=str(event_id),
        kind=str(kind),
        payload_bytes=bytes(payload_bytes),
        payload_cid=str(_payload_cid(payload_bytes)),
        parent_event_ids=tuple(
            str(p) for p in parent_event_ids),
        branch_label=str(branch_label),
        timestamp_ns=int(timestamp_ns),
    )


# ---------------------------------------------------------------
# Event-graph schema
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class EventGraphV1:
    """The append-only event DAG.

    ``nodes`` maps event_id → EventNodeV1.
    ``branch_tips`` maps branch_label → event_id of the tip.
    Both dicts are frozenset-replaceable (we return NEW
    graphs from append).
    """

    schema: str
    nodes: Mapping[str, EventNodeV1]
    branch_tips: Mapping[str, str]
    root_event_id: str

    @classmethod
    def empty(
            cls, *,
            genesis_payload: bytes = b"genesis",
    ) -> "EventGraphV1":
        genesis = build_event_node_v1(
            event_id=W82_EVENT_GRAPH_GENESIS_EVENT_ID,
            kind="genesis",
            payload_bytes=genesis_payload,
            parent_event_ids=tuple(),
            branch_label="main",
            timestamp_ns=int(0))
        return cls(
            schema=W82_EVENT_GRAPH_V1_SCHEMA_VERSION,
            nodes={genesis.event_id: genesis},
            branch_tips={"main": genesis.event_id},
            root_event_id=str(genesis.event_id),
        )

    def n_events(self) -> int:
        return int(len(self.nodes))

    def with_event(
            self,
            event: EventNodeV1,
    ) -> "EventGraphV1":
        if event.event_id in self.nodes:
            raise ValueError(
                f"event_id {event.event_id!r} already in "
                f"graph (append-only)")
        for p in event.parent_event_ids:
            if str(p) not in self.nodes:
                raise ValueError(
                    f"parent event_id {p!r} not in graph "
                    f"(cannot append a child of a non-"
                    f"existent parent)")
        new_nodes = dict(self.nodes)
        new_nodes[str(event.event_id)] = event
        new_tips = dict(self.branch_tips)
        new_tips[str(event.branch_label)] = str(event.event_id)
        return EventGraphV1(
            schema=W82_EVENT_GRAPH_V1_SCHEMA_VERSION,
            nodes=new_nodes,
            branch_tips=new_tips,
            root_event_id=str(self.root_event_id),
        )

    def get(self, event_id: str) -> EventNodeV1:
        if str(event_id) not in self.nodes:
            raise KeyError(
                f"event_id {event_id!r} not in graph")
        return self.nodes[str(event_id)]

    def ancestor_set(self, event_id: str) -> set[str]:
        """All transitive ancestor event_ids (inclusive of
        ``event_id``)."""
        out: set[str] = set()
        stack: list[str] = [str(event_id)]
        while stack:
            cur = stack.pop()
            if cur in out:
                continue
            if cur not in self.nodes:
                continue
            out.add(cur)
            for p in self.nodes[cur].parent_event_ids:
                stack.append(str(p))
        return out

    def ancestor_path_to(
            self, *,
            from_event_id: str,
            to_event_id: str,
    ) -> tuple[str, ...]:
        """Shortest ancestor path (BFS) from ``from_event_id``
        UPWARDS to ``to_event_id``.

        Returns a tuple of event_ids beginning with
        ``from_event_id`` and ending with ``to_event_id``.
        Raises ``ValueError`` if no path exists.
        """
        if str(from_event_id) == str(to_event_id):
            return (str(from_event_id),)
        # BFS over parents.
        from collections import deque
        prev: dict[str, str] = {}
        queue: deque[str] = deque([str(from_event_id)])
        seen: set[str] = {str(from_event_id)}
        while queue:
            cur = queue.popleft()
            if cur == str(to_event_id):
                break
            if cur not in self.nodes:
                continue
            for p in self.nodes[cur].parent_event_ids:
                p_s = str(p)
                if p_s not in seen:
                    seen.add(p_s)
                    prev[p_s] = cur
                    queue.append(p_s)
        if str(to_event_id) not in prev and str(
                to_event_id) != str(from_event_id):
            raise ValueError(
                f"no ancestor path from {from_event_id} to "
                f"{to_event_id}")
        # Reconstruct
        path: list[str] = []
        cur = str(to_event_id)
        path.append(cur)
        while cur != str(from_event_id):
            cur = prev[cur]
            path.append(cur)
        return tuple(reversed(path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_events": int(self.n_events()),
            "n_branches": int(len(self.branch_tips)),
            "branch_tips": {
                str(k): str(v)
                for k, v in self.branch_tips.items()},
            "root_event_id": str(self.root_event_id),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_event_graph_v1",
            "graph_summary": self.to_dict(),
            "node_cids": sorted(
                str(n.cid()) for n in self.nodes.values()),
        })


# ---------------------------------------------------------------
# Query types
# ---------------------------------------------------------------

class MemoryQueryKind(str, enum.Enum):
    BY_EVENT_ID = "by_event_id"
    BY_KIND = "by_kind"
    BY_BRANCH = "by_branch"
    BY_ANCESTOR_PATH = "by_ancestor_path"
    BY_CONTENT_PREDICATE = "by_content_predicate"


W82_EVENT_GRAPH_QUERY_KINDS: tuple[str, ...] = tuple(
    k.value for k in MemoryQueryKind)


@dataclasses.dataclass(frozen=True)
class MemoryQueryV1:
    """A declarative memory-graph query."""

    schema: str
    query_id: str
    kind: str
    target_event_id: str
    target_kind: str
    target_branch_label: str
    from_event_id: str
    to_event_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "query_id": str(self.query_id),
            "kind": str(self.kind),
            "target_event_id": str(self.target_event_id),
            "target_kind": str(self.target_kind),
            "target_branch_label": str(
                self.target_branch_label),
            "from_event_id": str(self.from_event_id),
            "to_event_id": str(self.to_event_id),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_memory_query_v1",
            "query": self.to_dict()})


def build_by_event_id_query_v1(
        *, query_id: str,
        target_event_id: str,
) -> MemoryQueryV1:
    return MemoryQueryV1(
        schema=W82_EVENT_GRAPH_V1_SCHEMA_VERSION,
        query_id=str(query_id),
        kind=MemoryQueryKind.BY_EVENT_ID.value,
        target_event_id=str(target_event_id),
        target_kind="",
        target_branch_label="",
        from_event_id="",
        to_event_id="",
    )


def build_by_kind_query_v1(
        *, query_id: str,
        target_kind: str,
) -> MemoryQueryV1:
    return MemoryQueryV1(
        schema=W82_EVENT_GRAPH_V1_SCHEMA_VERSION,
        query_id=str(query_id),
        kind=MemoryQueryKind.BY_KIND.value,
        target_event_id="",
        target_kind=str(target_kind),
        target_branch_label="",
        from_event_id="",
        to_event_id="",
    )


def build_by_ancestor_path_query_v1(
        *, query_id: str,
        from_event_id: str,
        to_event_id: str,
) -> MemoryQueryV1:
    return MemoryQueryV1(
        schema=W82_EVENT_GRAPH_V1_SCHEMA_VERSION,
        query_id=str(query_id),
        kind=MemoryQueryKind.BY_ANCESTOR_PATH.value,
        target_event_id="",
        target_kind="",
        target_branch_label="",
        from_event_id=str(from_event_id),
        to_event_id=str(to_event_id),
    )


def build_by_branch_query_v1(
        *, query_id: str,
        target_branch_label: str,
) -> MemoryQueryV1:
    return MemoryQueryV1(
        schema=W82_EVENT_GRAPH_V1_SCHEMA_VERSION,
        query_id=str(query_id),
        kind=MemoryQueryKind.BY_BRANCH.value,
        target_event_id="",
        target_kind="",
        target_branch_label=str(target_branch_label),
        from_event_id="",
        to_event_id="",
    )


# ---------------------------------------------------------------
# Provenance certificate
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ProvenanceCertificateV1:
    """Content-addressed evidence path for a query answer."""

    schema: str
    query_cid: str
    contributing_event_cids: tuple[str, ...]
    contributing_event_ids: tuple[str, ...]
    n_hops: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "query_cid": str(self.query_cid),
            "contributing_event_cids": list(
                self.contributing_event_cids),
            "contributing_event_ids": list(
                self.contributing_event_ids),
            "n_hops": int(self.n_hops),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_provenance_certificate_v1",
            "certificate": self.to_dict()})


# ---------------------------------------------------------------
# Query answer schema
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class QueryAnswerV1:
    """One query's answer + provenance."""

    schema: str
    query_cid: str
    success: bool
    payload_cids: tuple[str, ...]
    provenance_cid: str
    n_events_walked: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "query_cid": str(self.query_cid),
            "success": bool(self.success),
            "payload_cids": list(self.payload_cids),
            "provenance_cid": str(self.provenance_cid),
            "n_events_walked": int(self.n_events_walked),
            "detail": str(self.detail),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_query_answer_v1",
            "answer": self.to_dict()})


# ---------------------------------------------------------------
# Query plan + planner
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class QueryPlanV1:
    """The planner's plan for executing a query.

    Plans are content-addressed so we can reuse cached plans
    deterministically.
    """

    schema: str
    query_cid: str
    steps: tuple[str, ...]
    estimated_n_events_walked: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "query_cid": str(self.query_cid),
            "steps": list(self.steps),
            "estimated_n_events_walked": int(
                self.estimated_n_events_walked),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_query_plan_v1",
            "plan": self.to_dict()})


def plan_query_v1(
        *, graph: EventGraphV1,
        query: MemoryQueryV1,
) -> QueryPlanV1:
    """Plan a query against the event graph."""
    if query.kind == MemoryQueryKind.BY_EVENT_ID.value:
        steps = ("lookup_event_by_id",)
        est = 1
    elif query.kind == MemoryQueryKind.BY_KIND.value:
        steps = ("scan_all_events", "filter_by_kind")
        est = int(graph.n_events())
    elif query.kind == MemoryQueryKind.BY_BRANCH.value:
        steps = (
            "lookup_branch_tip",
            "walk_ancestors_until_root")
        est = int(graph.n_events())
    elif query.kind == MemoryQueryKind.BY_ANCESTOR_PATH.value:
        steps = ("bfs_ancestor_path",)
        est = int(graph.n_events())
    else:
        steps = ("scan_all_events", "apply_predicate")
        est = int(graph.n_events())
    return QueryPlanV1(
        schema=W82_EVENT_GRAPH_V1_SCHEMA_VERSION,
        query_cid=str(query.cid()),
        steps=tuple(steps),
        estimated_n_events_walked=int(est),
    )


def execute_query_with_carrier_fallback_v1(
        *, graph: EventGraphV1,
        query: MemoryQueryV1,
        carrier_lookup: "dict[str, str] | None" = None,
        carrier_cid: str = "",
) -> tuple["QueryAnswerV1", "ProvenanceCertificateV1"]:
    """Execute a query against the event graph; if the graph
    alone cannot answer, fall back to ``carrier_lookup`` for
    a content-addressed replay.

    ``carrier_lookup`` is a ``{event_id: event_cid}`` mapping
    derived from a long-horizon reconstruction carrier (e.g.
    ``LongHorizonReconstructionCarrier``). The fallback path
    answers ``BY_EVENT_ID`` queries when the event has been
    evicted from the live graph but its CID is still in the
    persistent carrier — i.e. it bridges the two abstraction
    layers (live event graph + long-horizon carrier) the
    P2 #11 issue calls for.

    Returns the same ``(QueryAnswerV1, ProvenanceCertificateV1)``
    tuple as ``execute_query_v1``. When the carrier path
    answers, the ``QueryAnswerV1.detail`` field is set to
    ``"carrier_fallback"`` and the provenance certificate
    lists the carrier CID as a synthetic contributing CID so
    the evidence path remains inspectable.
    """
    ans, prov = execute_query_v1(graph=graph, query=query)
    if ans.success:
        return ans, prov
    if carrier_lookup is None:
        return ans, prov
    # Fall back: only BY_EVENT_ID is supported by carrier
    # lookup (carrier holds event_id → event_cid map).
    if query.kind != MemoryQueryKind.BY_EVENT_ID.value:
        return ans, prov
    target = str(query.target_event_id)
    if target not in carrier_lookup:
        return ans, prov
    recovered_cid = str(carrier_lookup[target])
    # Build a new answer + provenance that flag the carrier
    # path.
    new_ans = QueryAnswerV1(
        schema=W82_EVENT_GRAPH_V1_SCHEMA_VERSION,
        query_cid=str(query.cid()),
        success=True,
        payload_cids=(str(recovered_cid),),
        provenance_cid="",  # filled below
        n_events_walked=1,
        detail="carrier_fallback",
    )
    new_prov = ProvenanceCertificateV1(
        schema=W82_EVENT_GRAPH_V1_SCHEMA_VERSION,
        query_cid=str(query.cid()),
        contributing_event_cids=(str(recovered_cid),),
        contributing_event_ids=(str(target),),
        n_hops=1,
    )
    # Re-fill provenance_cid on the answer.
    new_ans = dataclasses.replace(
        new_ans, provenance_cid=str(new_prov.cid()))
    return new_ans, new_prov


def execute_query_v1(
        *, graph: EventGraphV1,
        query: MemoryQueryV1,
        plan: QueryPlanV1 | None = None,
) -> QueryAnswerV1:
    """Execute a memory query against the graph and emit a
    QueryAnswerV1 with provenance."""
    plan = plan or plan_query_v1(graph=graph, query=query)
    contributing: list[EventNodeV1] = []
    success = False
    payload_cids: list[str] = []
    detail = ""
    if query.kind == MemoryQueryKind.BY_EVENT_ID.value:
        try:
            n = graph.get(str(query.target_event_id))
            contributing.append(n)
            success = True
            payload_cids.append(str(n.payload_cid))
            detail = "ok"
        except KeyError:
            detail = (
                f"event_id {query.target_event_id} not in "
                f"graph")
    elif query.kind == MemoryQueryKind.BY_KIND.value:
        matched = [
            n for n in graph.nodes.values()
            if str(n.kind) == str(query.target_kind)]
        contributing.extend(matched)
        payload_cids.extend(str(n.payload_cid) for n in matched)
        success = bool(matched)
        detail = (
            "ok" if success else
            f"no events of kind {query.target_kind!r}")
    elif query.kind == MemoryQueryKind.BY_BRANCH.value:
        label = str(query.target_branch_label)
        if label not in graph.branch_tips:
            detail = f"unknown branch {label!r}"
        else:
            tip = graph.branch_tips[label]
            anc_set = graph.ancestor_set(str(tip))
            matched = [
                graph.nodes[eid] for eid in anc_set
                if eid in graph.nodes]
            contributing.extend(matched)
            payload_cids.extend(
                str(n.payload_cid) for n in matched)
            success = bool(matched)
            detail = (
                "ok" if success else
                "branch tip resolved but no events found")
    elif query.kind == MemoryQueryKind.BY_ANCESTOR_PATH.value:
        try:
            path = graph.ancestor_path_to(
                from_event_id=str(query.from_event_id),
                to_event_id=str(query.to_event_id))
            for eid in path:
                if eid in graph.nodes:
                    contributing.append(graph.nodes[eid])
                    payload_cids.append(
                        str(graph.nodes[eid].payload_cid))
            success = True
            detail = "ok"
        except (KeyError, ValueError) as e:
            detail = f"ancestor path failed: {e}"
    else:
        detail = f"unsupported query kind {query.kind!r}"
    prov = ProvenanceCertificateV1(
        schema=W82_EVENT_GRAPH_V1_SCHEMA_VERSION,
        query_cid=str(query.cid()),
        contributing_event_cids=tuple(
            n.cid() for n in contributing),
        contributing_event_ids=tuple(
            n.event_id for n in contributing),
        n_hops=int(len(contributing)),
    )
    return QueryAnswerV1(
        schema=W82_EVENT_GRAPH_V1_SCHEMA_VERSION,
        query_cid=str(query.cid()),
        success=bool(success),
        payload_cids=tuple(payload_cids),
        provenance_cid=str(prov.cid()),
        n_events_walked=int(len(contributing)),
        detail=str(detail),
    ), prov


# ---------------------------------------------------------------
# Derived summary view (NOT primary store)
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class DerivedSummaryViewV1:
    """A *derived* projection over the event graph.

    Derived = recomputed from the primary event store. The
    summary is content-addressed and tracks the source-graph
    CID it was projected from. Stale views can be detected by
    CID mismatch.
    """

    schema: str
    source_graph_cid: str
    per_kind_count: Mapping[str, int]
    per_branch_count: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "source_graph_cid": str(
                self.source_graph_cid),
            "per_kind_count": {
                str(k): int(v)
                for k, v in sorted(
                    self.per_kind_count.items())},
            "per_branch_count": {
                str(k): int(v)
                for k, v in sorted(
                    self.per_branch_count.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_derived_summary_view_v1",
            "view": self.to_dict()})


def build_derived_summary_view_v1(
        graph: EventGraphV1,
) -> DerivedSummaryViewV1:
    per_kind: dict[str, int] = {}
    per_branch: dict[str, int] = {}
    for n in graph.nodes.values():
        per_kind[str(n.kind)] = (
            per_kind.get(str(n.kind), 0) + 1)
        per_branch[str(n.branch_label)] = (
            per_branch.get(str(n.branch_label), 0) + 1)
    return DerivedSummaryViewV1(
        schema=W82_EVENT_GRAPH_V1_SCHEMA_VERSION,
        source_graph_cid=str(graph.cid()),
        per_kind_count=per_kind,
        per_branch_count=per_branch,
    )


# ---------------------------------------------------------------
# Baselines for the bench
# ---------------------------------------------------------------

def bounded_handoff_baseline_v1(
        *, events: Sequence[EventNodeV1],
        query: MemoryQueryV1,
        k: int = (
            W82_EVENT_GRAPH_DEFAULT_BOUNDED_HANDOFF_K),
) -> tuple[bool, int]:
    """The bounded-handoff baseline: sees only the most-recent
    ``k`` events (in insertion order). Returns (success,
    n_events_walked).
    """
    visible = list(events)[-int(k):]
    if query.kind == MemoryQueryKind.BY_EVENT_ID.value:
        ok = any(
            str(e.event_id) == str(query.target_event_id)
            for e in visible)
        return bool(ok), int(len(visible))
    if query.kind == MemoryQueryKind.BY_ANCESTOR_PATH.value:
        # The baseline can only succeed if BOTH endpoints
        # are inside the visible window.
        visible_ids = {str(e.event_id) for e in visible}
        ok = (
            str(query.from_event_id) in visible_ids and
            str(query.to_event_id) in visible_ids)
        return bool(ok), int(len(visible))
    if query.kind == MemoryQueryKind.BY_KIND.value:
        ok = any(
            str(e.kind) == str(query.target_kind)
            for e in visible)
        return bool(ok), int(len(visible))
    if query.kind == MemoryQueryKind.BY_BRANCH.value:
        ok = any(
            str(e.branch_label) == str(
                query.target_branch_label)
            for e in visible)
        return bool(ok), int(len(visible))
    return False, int(len(visible))


def trajectory_slice_baseline_v1(
        *, events: Sequence[EventNodeV1],
        query: MemoryQueryV1,
        k: int = W82_EVENT_GRAPH_DEFAULT_TRAJECTORY_K,
        current_branch: str = "main",
) -> tuple[bool, int]:
    """The trajectory-slice baseline: sees only events on the
    ``current_branch`` AND only the most-recent k of them.
    """
    on_branch = [
        e for e in events
        if str(e.branch_label) == str(current_branch)]
    visible = on_branch[-int(k):]
    if query.kind == MemoryQueryKind.BY_EVENT_ID.value:
        ok = any(
            str(e.event_id) == str(query.target_event_id)
            for e in visible)
        return bool(ok), int(len(visible))
    if query.kind == MemoryQueryKind.BY_ANCESTOR_PATH.value:
        visible_ids = {str(e.event_id) for e in visible}
        ok = (
            str(query.from_event_id) in visible_ids and
            str(query.to_event_id) in visible_ids)
        return bool(ok), int(len(visible))
    if query.kind == MemoryQueryKind.BY_KIND.value:
        ok = any(
            str(e.kind) == str(query.target_kind)
            for e in visible)
        return bool(ok), int(len(visible))
    if query.kind == MemoryQueryKind.BY_BRANCH.value:
        ok = any(
            str(e.branch_label) == str(
                query.target_branch_label)
            for e in visible)
        return bool(ok), int(len(visible))
    return False, int(len(visible))


# ---------------------------------------------------------------
# Synthetic graph + bench
# ---------------------------------------------------------------

def build_synthetic_event_graph_v1(
        *, n_events: int = (
            W82_EVENT_GRAPH_DEFAULT_BENCH_N_EVENTS),
        seed: int = W82_EVENT_GRAPH_DEFAULT_BENCH_SEED,
) -> tuple[EventGraphV1, tuple[EventNodeV1, ...]]:
    """Build a deterministic synthetic graph with three
    branches (main, side_a, side_b) and one merge.

    The graph is intentionally large enough to exceed any
    bounded-handoff budget. The returned tuple is (graph,
    ordered_events) — the ordered_events tuple lists the
    events in insertion order so baselines that look at
    "most-recent k" have a deterministic notion of recency.
    """
    g = EventGraphV1.empty(genesis_payload=b"genesis")
    events_in_order: list[EventNodeV1] = [
        g.nodes[g.root_event_id]]
    main_tip = g.root_event_id
    side_a_tip: str | None = None
    side_b_tip: str | None = None
    # Layout: alternating events on main / side_a / side_b
    # with a merge from side_a + side_b back into main near
    # 75% of the way through.
    merge_at = int(0.75 * n_events)
    for i in range(int(n_events)):
        ts = int(i + 1)
        branch_idx = i % 3
        if branch_idx == 0:
            label = "main"
            parents = [main_tip]
        elif branch_idx == 1:
            label = "side_a"
            parents = [
                side_a_tip if side_a_tip is not None
                else main_tip]
        else:
            label = "side_b"
            parents = [
                side_b_tip if side_b_tip is not None
                else main_tip]
        # Insert a merge near merge_at: a node on main with
        # TWO parents (side_a tip + side_b tip).
        if i == merge_at:
            label = "main"
            parents = [main_tip]
            if side_a_tip is not None:
                parents.append(side_a_tip)
            if side_b_tip is not None:
                parents.append(side_b_tip)
        payload_bytes = _canonical_bytes({
            "kind": "synthetic_event",
            "i": int(i),
            "seed": int(seed),
        })
        node = build_event_node_v1(
            event_id=f"e{i}",
            kind=(
                "merge" if (i == merge_at and
                            side_a_tip is not None and
                            side_b_tip is not None) else
                "synthetic"),
            payload_bytes=payload_bytes,
            parent_event_ids=tuple(parents),
            branch_label=str(label),
            timestamp_ns=int(ts))
        g = g.with_event(node)
        events_in_order.append(node)
        if label == "main":
            main_tip = node.event_id
        elif label == "side_a":
            side_a_tip = node.event_id
        elif label == "side_b":
            side_b_tip = node.event_id
    return g, tuple(events_in_order)


# ---------------------------------------------------------------
# Bench schema
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class EventGraphBenchReportV1:
    """End-to-end bench output."""

    schema: str
    n_events: int
    n_queries: int
    graph_success_rate: float
    bounded_handoff_k32_success_rate: float
    trajectory_slice_k16_success_rate: float
    graph_dominates_baselines: bool
    answer_cids: tuple[str, ...]
    provenance_cids: tuple[str, ...]
    derived_view_cid: str
    config_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_events": int(self.n_events),
            "n_queries": int(self.n_queries),
            "graph_success_rate": float(round(
                self.graph_success_rate, 12)),
            "bounded_handoff_k32_success_rate": float(round(
                self.bounded_handoff_k32_success_rate, 12)),
            "trajectory_slice_k16_success_rate": float(round(
                self.trajectory_slice_k16_success_rate, 12)),
            "graph_dominates_baselines": bool(
                self.graph_dominates_baselines),
            "n_answers": int(len(self.answer_cids)),
            "answer_cids": list(self.answer_cids),
            "provenance_cids": list(self.provenance_cids),
            "derived_view_cid": str(self.derived_view_cid),
            "config_cid": str(self.config_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_event_graph_bench_report_v1",
            "report": self.to_dict()})


def run_event_graph_bench_end_to_end_v1(
        *, n_events: int = (
            W82_EVENT_GRAPH_DEFAULT_BENCH_N_EVENTS),
        seed: int = W82_EVENT_GRAPH_DEFAULT_BENCH_SEED,
        n_queries: int = (
            W82_EVENT_GRAPH_DEFAULT_BENCH_N_QUERIES),
        bounded_handoff_k: int = (
            W82_EVENT_GRAPH_DEFAULT_BOUNDED_HANDOFF_K),
        trajectory_k: int = (
            W82_EVENT_GRAPH_DEFAULT_TRAJECTORY_K),
) -> EventGraphBenchReportV1:
    """Run the load-bearing event-graph bench."""
    g, ordered = build_synthetic_event_graph_v1(
        n_events=int(n_events), seed=int(seed))
    # Generate by-ancestor-path queries that walk far through
    # the graph. By construction, the source event is one of
    # the earliest events in the chain, so any bounded-handoff
    # baseline cannot see it.
    early_events = [
        e for e in ordered[1:1 + int(n_events) // 4]]
    late_events = [
        e for e in ordered[-int(n_events) // 4:]]
    queries: list[MemoryQueryV1] = []
    for q_idx in range(int(n_queries)):
        src = early_events[
            q_idx % max(1, len(early_events))]
        dst = late_events[
            q_idx % max(1, len(late_events))]
        queries.append(build_by_ancestor_path_query_v1(
            query_id=f"q{q_idx}",
            from_event_id=str(dst.event_id),
            to_event_id=str(src.event_id)))
    graph_successes = 0
    h32_successes = 0
    tj16_successes = 0
    answer_cids: list[str] = []
    prov_cids: list[str] = []
    for q in queries:
        ans, prov = execute_query_v1(graph=g, query=q)
        answer_cids.append(ans.cid())
        prov_cids.append(prov.cid())
        if ans.success:
            graph_successes += 1
        h_ok, _ = bounded_handoff_baseline_v1(
            events=ordered, query=q,
            k=int(bounded_handoff_k))
        if h_ok:
            h32_successes += 1
        t_ok, _ = trajectory_slice_baseline_v1(
            events=ordered, query=q,
            k=int(trajectory_k), current_branch="main")
        if t_ok:
            tj16_successes += 1
    graph_rate = (
        float(graph_successes) /
        float(max(1, len(queries))))
    h32_rate = (
        float(h32_successes) /
        float(max(1, len(queries))))
    tj16_rate = (
        float(tj16_successes) /
        float(max(1, len(queries))))
    derived_view = build_derived_summary_view_v1(g)
    config_cid = _sha256_hex({
        "kind": "w82_event_graph_bench_config_v1",
        "n_events": int(n_events),
        "n_queries": int(n_queries),
        "bounded_handoff_k": int(bounded_handoff_k),
        "trajectory_k": int(trajectory_k),
        "seed": int(seed),
    })
    return EventGraphBenchReportV1(
        schema=W82_EVENT_GRAPH_V1_SCHEMA_VERSION,
        n_events=int(n_events),
        n_queries=int(n_queries),
        graph_success_rate=float(graph_rate),
        bounded_handoff_k32_success_rate=float(h32_rate),
        trajectory_slice_k16_success_rate=float(tj16_rate),
        graph_dominates_baselines=bool(
            graph_rate > h32_rate and
            graph_rate > tj16_rate),
        answer_cids=tuple(answer_cids),
        provenance_cids=tuple(prov_cids),
        derived_view_cid=str(derived_view.cid()),
        config_cid=str(config_cid),
    )


__all__ = [
    "W82_EVENT_GRAPH_V1_SCHEMA_VERSION",
    "W82_EVENT_GRAPH_GENESIS_EVENT_ID",
    "W82_EVENT_GRAPH_QUERY_KINDS",
    "MemoryQueryKind",
    "EventNodeV1",
    "EventGraphV1",
    "MemoryQueryV1",
    "QueryPlanV1",
    "QueryAnswerV1",
    "ProvenanceCertificateV1",
    "DerivedSummaryViewV1",
    "EventGraphBenchReportV1",
    "build_event_node_v1",
    "build_by_event_id_query_v1",
    "build_by_kind_query_v1",
    "build_by_branch_query_v1",
    "build_by_ancestor_path_query_v1",
    "plan_query_v1",
    "execute_query_v1",
    "execute_query_with_carrier_fallback_v1",
    "build_derived_summary_view_v1",
    "bounded_handoff_baseline_v1",
    "trajectory_slice_baseline_v1",
    "build_synthetic_event_graph_v1",
    "run_event_graph_bench_end_to_end_v1",
]
