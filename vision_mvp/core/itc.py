"""Interval Tree Clocks — causality that scales with branches, not N.

Almeida, Baquero & Fonte (2008). Classical vector clocks are O(N) per stamp,
a problem when N is dynamic. ITC represents time as a *stamp* (id, event),
where id is a binary tree encoding the agent's "split" history and event is
a labeled tree counting events per id region. Key operations:

  fork(s)   -> (s1, s2)   split the id into two disjoint parts
  join(s1, s2) -> s        recombine two stamps that were forked
  event(s) -> s'           record one event under this stamp's id
  leq(a, b) -> bool        a ≤ b in causal order

ITC stamp size is linear in the number of concurrent *forks* ever taken in
the stamp's history — not in N. For a team with bounded concurrency this is
O(log N) or better.

This is a *reference* implementation — readable, correct, but not optimized
for large trees. Sufficient for CASR's dynamic-membership experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


# An Id is either a leaf 0, leaf 1, or an internal node (Id, Id).
# An Event is either a leaf int, or (int, Event, Event).
Id = Union[int, tuple]
Event = Union[int, tuple]


# ----------------- Id operations -----------------

def _id_norm(i: Id) -> Id:
    """Collapse redundant internal nodes: (0, 0) -> 0, (1, 1) -> 1."""
    if isinstance(i, int):
        return i
    l, r = i
    l, r = _id_norm(l), _id_norm(r)
    if l == r == 0:
        return 0
    if l == r == 1:
        return 1
    return (l, r)


def _id_split(i: Id) -> tuple[Id, Id]:
    """Split an id into two disjoint halves."""
    if i == 0:
        return 0, 0
    if i == 1:
        return (1, 0), (0, 1)
    l, r = i
    if l == 0:
        rl, rr = _id_split(r)
        return (0, rl), (0, rr)
    if r == 0:
        ll, lr = _id_split(l)
        return (ll, 0), (lr, 0)
    return (l, 0), (0, r)


def _id_sum(a: Id, b: Id) -> Id:
    """Combine two disjoint ids into one."""
    if a == 0:
        return b
    if b == 0:
        return a
    if a == 1 or b == 1:
        raise ValueError("ids are not disjoint")
    al, ar = a
    bl, br = b
    return _id_norm((_id_sum(al, bl), _id_sum(ar, br)))


# ----------------- Event operations -----------------

def _ev_leq(e1: Event, e2: Event) -> bool:
    """Tree-leq on event trees (pointwise)."""
    if isinstance(e1, int) and isinstance(e2, int):
        return e1 <= e2
    if isinstance(e1, int):
        n2, l2, r2 = e2
        return e1 <= n2
    if isinstance(e2, int):
        n1, l1, r1 = e1
        return _ev_leq(_ev_lift(_ev_max(l1, r1), n1), e2)
    n1, l1, r1 = e1
    n2, l2, r2 = e2
    if n1 > n2:
        return False
    return _ev_leq(_ev_lift(l1, n1), _ev_lift(l2, n2)) and \
           _ev_leq(_ev_lift(r1, n1), _ev_lift(r2, n2))


def _ev_lift(e: Event, n: int) -> Event:
    if isinstance(e, int):
        return e + n
    m, l, r = e
    return (m + n, l, r)


def _ev_max(a: Event, b: Event) -> int:
    if isinstance(a, int) and isinstance(b, int):
        return max(a, b)
    if isinstance(a, int):
        n2, l2, r2 = b
        return max(a, n2 + max(_ev_max(l2, 0), _ev_max(r2, 0)))
    if isinstance(b, int):
        return _ev_max(b, a)
    # both tuples
    n1, l1, r1 = a
    n2, l2, r2 = b
    return max(n1 + max(_ev_max(l1, 0), _ev_max(r1, 0)),
               n2 + max(_ev_max(l2, 0), _ev_max(r2, 0)))


def _ev_norm(e: Event) -> Event:
    """Collapse (n, m, m) -> n+m when m is an int, etc."""
    if isinstance(e, int):
        return e
    n, l, r = e
    l = _ev_norm(l)
    r = _ev_norm(r)
    if isinstance(l, int) and isinstance(r, int) and l == r:
        return n + l
    return (n, l, r)


def _ev_join(a: Event, b: Event) -> Event:
    if isinstance(a, int) and isinstance(b, int):
        return max(a, b)
    if isinstance(a, int):
        return _ev_join((a, 0, 0), b)
    if isinstance(b, int):
        return _ev_join(a, (b, 0, 0))
    n1, l1, r1 = a
    n2, l2, r2 = b
    if n1 > n2:
        return _ev_join(b, a)
    d = n2 - n1
    return _ev_norm((n1, _ev_join(l1, _ev_lift(l2, d)), _ev_join(r1, _ev_lift(r2, d))))


def _ev_inc(i: Id, e: Event) -> Event:
    """Record one event at id `i` on event tree `e`."""
    if isinstance(i, int):
        if i == 0:
            return e
        # id is 1 → bump the whole region
        if isinstance(e, int):
            return e + 1
        n, l, r = e
        return _ev_norm((n + 1, l, r))
    il, ir = i
    if isinstance(e, int):
        # materialise to a node
        e_node = (e, 0, 0)
    else:
        e_node = e
    n, l, r = e_node
    new_l = _ev_inc(il, l)
    new_r = _ev_inc(ir, r)
    return _ev_norm((n, new_l, new_r))


# ----------------- public Stamp class -----------------

@dataclass
class Stamp:
    """ITC stamp = (id, event)."""
    id: Id = 1
    event: Event = 0

    # --- factory ---

    @classmethod
    def seed(cls) -> "Stamp":
        return cls(id=1, event=0)

    # --- operations ---

    def fork(self) -> tuple["Stamp", "Stamp"]:
        i1, i2 = _id_split(self.id)
        return Stamp(id=i1, event=self.event), Stamp(id=i2, event=self.event)

    def peek(self) -> "Stamp":
        return Stamp(id=0, event=self.event)

    def event_tick(self) -> "Stamp":
        return Stamp(id=self.id, event=_ev_inc(self.id, self.event))

    def join(self, other: "Stamp") -> "Stamp":
        return Stamp(
            id=_id_sum(self.id, other.id),
            event=_ev_join(self.event, other.event),
        )

    def leq(self, other: "Stamp") -> bool:
        return _ev_leq(self.event, other.event)

    def concurrent_with(self, other: "Stamp") -> bool:
        return not self.leq(other) and not other.leq(self)
