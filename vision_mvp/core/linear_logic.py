"""MLL proof-net fragment — compile-time protocol checker.

Linear logic (Girard, 1987) treats propositions as *resources* that cannot be
duplicated or discarded except under explicit modalities. Its multiplicative
fragment MLL has a beautiful graphical presentation — proof nets — whose
correctness is decidable via a purely graph-theoretic check:

  Danos–Regnier (1989): a proof structure is a proof net iff every
  switching graph is acyclic and connected.

A switching graph resolves each ⅋ link by picking exactly one of its two
premises; a proof structure with n ⅋ links has 2ⁿ switching graphs, all of
which must be trees.

We use this as a *static* session-type checker for inter-agent protocols: a
message pattern is "resource-correct" iff its proof structure passes the
Danos-Regnier test. Catches double-consumption and leak bugs at team-construct
time rather than at runtime.

This is the small MLL-without-units fragment: atoms, tensor ⊗, par ⅋, negation,
and axioms. No exponentials, no units — sufficient for the kinds of protocol
patterns CASR uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product as cartesian_product
from typing import Literal


LinkKind = Literal["axiom", "cut", "tensor", "par"]


@dataclass
class ProofStructure:
    """A proof structure = conclusions (nodes) + links between them."""

    n_nodes: int
    links: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)
    # Each link is (kind, indices_of_node_premises_or_conclusions)

    def axiom(self, a: int, b: int) -> None:
        """Axiom link between two conclusions a, b (complementary atoms)."""
        self.links.append(("axiom", (a, b)))

    def cut(self, a: int, b: int) -> None:
        """Cut between two conclusions a, b."""
        self.links.append(("cut", (a, b)))

    def tensor(self, a: int, b: int, result: int) -> None:
        """A ⊗ B connecting premises a, b to conclusion `result`."""
        self.links.append(("tensor", (a, b, result)))

    def par(self, a: int, b: int, result: int) -> None:
        """A ⅋ B connecting premises a, b to conclusion `result`."""
        self.links.append(("par", (a, b, result)))


def _switching_edges(
    structure: ProofStructure, par_choices: tuple[int, ...],
) -> list[tuple[int, int]]:
    """Construct the switching-graph edges given a par-choice tuple.

    `par_choices[i] ∈ {0, 1}` picks which premise of the i-th par link to
    *retain* as the connected edge; the other premise is disconnected.
    """
    edges = []
    par_idx = 0
    for kind, ids in structure.links:
        if kind == "axiom":
            edges.append((ids[0], ids[1]))
        elif kind == "cut":
            edges.append((ids[0], ids[1]))
        elif kind == "tensor":
            # connect both premises to result
            edges.append((ids[0], ids[2]))
            edges.append((ids[1], ids[2]))
        elif kind == "par":
            pick = par_choices[par_idx]
            edges.append((ids[pick], ids[2]))
            par_idx += 1
    return edges


def _is_tree(n: int, edges: list[tuple[int, int]]) -> bool:
    """True iff (nodes, edges) is a tree: connected and acyclic."""
    if not edges and n == 1:
        return True
    # Disjoint-set union with cycle-detection
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            return False      # cycle
        parent[ru] = rv
    # Connected iff |edges| = n − 1 and all sharing a single component
    if len(edges) != n - 1:
        return False
    roots = {find(i) for i in range(n)}
    return len(roots) == 1


def is_proof_net(structure: ProofStructure) -> bool:
    """Danos–Regnier check: all switching graphs are trees."""
    n_par = sum(1 for k, _ in structure.links if k == "par")
    for choice in cartesian_product(*([0, 1] for _ in range(n_par))):
        edges = _switching_edges(structure, choice)
        if not _is_tree(structure.n_nodes, edges):
            return False
    return True


@dataclass
class SessionTypeCheckResult:
    ok: bool
    reason: str


def check_session(structure: ProofStructure) -> SessionTypeCheckResult:
    """Friendly wrapper — yes/no plus a human-readable reason."""
    if is_proof_net(structure):
        return SessionTypeCheckResult(True, "valid MLL proof net")
    return SessionTypeCheckResult(
        False,
        "some switching graph fails the tree test — "
        "protocol duplicates or leaks a resource",
    )
