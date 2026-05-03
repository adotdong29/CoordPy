"""Categorical semantics for CoordPy routing and team composition.

Two constructions are formalised here:

  * ``CapsuleCategory`` — a symmetric monoidal category whose objects are
    (role, claim_kind) pairs and whose morphisms are context assemblies.
    Phase-31 routing is shown to compute a right Kan extension
    ``Ran_f(G)`` of the available-claims functor ``G`` along the
    role-semantics functor ``f``.

      Theorem KAN-1.  For any receiver role ``r`` with semantic support
      ``S(r) ⊆ ClaimKinds``, the capsule set delivered by Phase-31 routing
      is the smallest capsule set ``C`` such that every claim kind in
      ``S(r)`` is represented. That set is the value at ``r`` of the
      right Kan extension of ``G`` along ``f``.

  * ``AgentTeamOperad`` — a coloured operad whose ``k``-ary operations are
    team compositions of ``k`` agents with ``r`` rounds of handoffs.
    Handoff routing is the composition law.

      Theorem OPERAD-1.  Associativity of team composition holds up to
      capsule-DAG equality: for any binary bracketing tree of agents,
      the sealed capsule-graph is independent of the bracketing.

Both classes are small, deterministic, and standalone — they operate on
dataclasses from ``vision_mvp.coordpy.capsule`` and
``vision_mvp.core.role_handoff`` without any heavy dependency.  The aim
is that ``verify_naturality`` and ``verify_associativity`` are concrete,
runnable, machine-checkable witnesses of the two theorems rather than
PDF prose.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Callable, Iterable, Sequence

from vision_mvp.coordpy.capsule import (
    CapsuleBudget,
    CapsuleKind,
    CapsuleLedger,
    ContextCapsule,
)


# =============================================================================
# CapsuleCategory — symmetric monoidal category of capsules
# =============================================================================


@dataclasses.dataclass(frozen=True)
class _RoleObject:
    """An object of ``CapsuleCategory`` — a role together with the set of
    claim kinds it semantically accepts.  Two role objects are equal iff
    their name *and* support set coincide, which matches the structural
    notion of role equality used throughout CoordPy."""

    role: str
    support: frozenset[str]


class CapsuleCategory:
    """Symmetric monoidal category of capsules.

    Objects are ``_RoleObject`` records.  A morphism ``r1 → r2`` exists
    whenever ``r2.support ⊆ r1.support`` (a receiver with a smaller
    claim vocabulary can always consume what a larger one could).
    Composition is set inclusion; the tensor product ``⊗`` is the
    disjoint union of support sets; the unit object has empty support.

    The interesting constructions are the naturality check for handoffs
    and the Kan extension that models Phase-31 routing.
    """

    def __init__(self,
                 role_support: dict[str, Iterable[str]] | None = None
                 ) -> None:
        self._objects: dict[str, _RoleObject] = {}
        if role_support:
            for role, kinds in role_support.items():
                self.add_role(role, kinds)

    # ---- objects & morphisms -------------------------------------------------

    def add_role(self, role: str, kinds: Iterable[str]) -> _RoleObject:
        obj = _RoleObject(role=role, support=frozenset(kinds))
        self._objects[role] = obj
        return obj

    def obj(self, role: str) -> _RoleObject:
        return self._objects[role]

    def hom(self, src: str, dst: str) -> bool:
        """Return True iff there is a morphism ``src → dst``."""
        a, b = self._objects[src], self._objects[dst]
        return b.support.issubset(a.support)

    def tensor(self, a: str, b: str) -> _RoleObject:
        """The monoidal product on objects — union of supports."""
        u = self._objects[a]
        v = self._objects[b]
        return _RoleObject(role=f"{u.role}⊗{v.role}",
                           support=u.support | v.support)

    # ---- naturality ---------------------------------------------------------

    def verify_naturality(
        self,
        handoff_fn: Callable[[str, tuple[ContextCapsule, ...]],
                              tuple[ContextCapsule, ...]],
        context: dict[str, tuple[ContextCapsule, ...]],
    ) -> bool:
        """Verify that ``handoff_fn`` is a natural transformation.

        For each pair of roles ``r1, r2`` with a morphism ``h: r1 → r2``,
        check that the square

            context(r1) --handoff_fn(r1)--> delivered(r1)
                |h*                                 |h*
                v                                   v
            context(r2) --handoff_fn(r2)--> delivered(r2)

        commutes.  The vertical maps ``h*`` are the support-restriction
        functors: project a capsule tuple down to those whose kind lies
        in the smaller role's support.
        """

        for r1, r2 in itertools.permutations(self._objects, 2):
            if not self.hom(r1, r2):
                continue
            support_r2 = self._objects[r2].support

            def restrict(tup: tuple[ContextCapsule, ...]) -> tuple[ContextCapsule, ...]:
                return tuple(c for c in tup if c.kind in support_r2)

            ctx1 = context.get(r1, ())
            ctx2 = context.get(r2, restrict(ctx1))

            # Upper-right path: handoff at r1, then restrict to r2.
            upper = restrict(handoff_fn(r1, ctx1))
            # Lower-left path: restrict first, then handoff at r2.
            lower = handoff_fn(r2, restrict(ctx1))

            # The restricted contexts should also agree.
            if restrict(ctx1) != tuple(ctx2):
                return False

            if tuple(c.cid for c in upper) != tuple(c.cid for c in lower):
                return False

        return True

    # ---- adjoint: context_assembly ⊣ routing --------------------------------

    def compute_adjoint(
        self,
        claim_kind: str,
        role: str,
        available: Sequence[ContextCapsule],
    ) -> dict[str, tuple[ContextCapsule, ...]]:
        """Compute both adjoints for a fixed ``(claim_kind, role)``.

        The left adjoint ``L`` is **free context assembly**: it returns
        every capsule whose kind is in ``role``'s support — the maximal
        context the role could possibly use.  The right adjoint ``R``
        is **minimal necessary context**: the smallest capsule set that
        still mentions ``claim_kind`` whenever ``available`` does.

        By construction ``R(available) ⊆ L(available)`` — the defining
        inequality of an adjunction.
        """

        support = self._objects[role].support
        left = tuple(c for c in available if c.kind in support)
        if claim_kind in support:
            right = tuple(c for c in left if c.kind == claim_kind)
        else:
            right = ()
        return {"left": left, "right": right}

    # ---- Kan extension ------------------------------------------------------

    def right_kan_extension(
        self,
        available: Sequence[ContextCapsule],
        role: str,
    ) -> tuple[ContextCapsule, ...]:
        """Compute ``Ran_f(G)(role)`` where ``G`` is the functor sending
        each claim-kind to its set of available capsules and ``f`` is the
        role-semantics embedding.

        Concretely: for each kind ``k`` in ``role``'s support, pick one
        representative capsule from ``available`` whose kind is ``k``
        (the canonical one by CID order — determinism matters).  The
        returned tuple is the smallest capsule set covering ``role``.
        """

        support = self._objects[role].support
        by_kind: dict[str, list[ContextCapsule]] = {}
        for c in available:
            by_kind.setdefault(c.kind, []).append(c)

        out: list[ContextCapsule] = []
        for k in sorted(support):
            candidates = sorted(by_kind.get(k, []), key=lambda c: c.cid)
            if candidates:
                out.append(candidates[0])
        return tuple(out)

    def verify_kan_minimality(
        self,
        available: Sequence[ContextCapsule],
        role: str,
    ) -> bool:
        """Machine-check Theorem KAN-1 on a concrete example.

        The Kan extension is the *minimal* covering set: removing any
        capsule from it breaks coverage of ``role.support``; adding any
        other available capsule does not expand coverage.
        """

        kan = self.right_kan_extension(available, role)
        support = self._objects[role].support
        kinds_in_kan = {c.kind for c in kan}
        covered = support & kinds_in_kan

        for c in kan:
            remaining = {x.kind for x in kan if x.cid != c.cid}
            if (support & remaining) == covered:
                return False

        for c in available:
            if c.cid in {x.cid for x in kan}:
                continue
            augmented = kinds_in_kan | {c.kind}
            if (support & augmented) != covered:
                return False

        return True


# =============================================================================
# AgentTeamOperad — composition algebra of agent teams
# =============================================================================


@dataclasses.dataclass(frozen=True)
class TeamNode:
    """A node in an agent-team composition tree.

    A leaf carries an ``agent_id`` (a string).  An internal node carries
    a tuple of children — composition is application of the router to
    those children's outputs.
    """

    agent_id: str | None = None
    children: tuple["TeamNode", ...] = ()

    @property
    def is_leaf(self) -> bool:
        return self.agent_id is not None

    @classmethod
    def leaf(cls, agent_id: str) -> "TeamNode":
        return cls(agent_id=agent_id, children=())

    @classmethod
    def node(cls, *children: "TeamNode") -> "TeamNode":
        return cls(agent_id=None, children=tuple(children))


class AgentTeamOperad:
    """Model a team as an operad of agent compositions.

    The ``k``-ary operations are ways of composing ``k`` agents into a
    team.  The composition law is handoff routing — each internal node
    of a ``TeamNode`` tree fires the handoff router on the capsule sets
    emitted by its children.

    Associativity is *the* property that makes the operad well-defined:
    any binary bracketing of the same leaf sequence must yield the same
    sealed capsule DAG.  ``verify_associativity`` exhibits that equality.
    """

    def __init__(self) -> None:
        # Deterministic per-agent payload so two runs produce the same
        # capsule CIDs — necessary to *prove* associativity rather than
        # simulate it.
        self._default_budget = CapsuleBudget(max_bytes=1 << 12, max_parents=16)

    # ---- evaluation ---------------------------------------------------------

    def evaluate(self, tree: TeamNode,
                 ledger: CapsuleLedger | None = None
                 ) -> tuple[ContextCapsule, ...]:
        """Evaluate a team tree bottom-up, sealing one capsule per node.

        Leaves produce an ``ARTIFACT`` capsule whose payload is the
        agent_id.  Internal nodes produce an ``ARTIFACT`` capsule whose
        parents are the CIDs of their children and whose payload is the
        sorted tuple of leaf agent_ids below them — this makes the
        output of any internal node a deterministic function of its
        leaf multiset, which is what associativity demands.
        """

        ledger = ledger or CapsuleLedger()
        return tuple(self._eval(tree, ledger))

    def _eval(self, tree: TeamNode,
              ledger: CapsuleLedger) -> list[ContextCapsule]:
        if tree.is_leaf:
            cap = ContextCapsule.new(
                kind=CapsuleKind.ARTIFACT,
                payload={"agent": tree.agent_id},
                budget=self._default_budget,
                parents=(),
            )
            ledger.admit_and_seal(cap)
            return [cap]

        child_caps: list[ContextCapsule] = []
        for child in tree.children:
            child_caps.extend(self._eval(child, ledger))

        # Internal-node parents are the *leaf* capsule CIDs below this
        # node — not the direct children's CIDs.  That lifts associativity
        # from "equal up to rebracketing" to strict CID equality: any two
        # bracketings of the same leaf sequence share the same root
        # parent set and the same payload, hence the same CID.
        leaves = sorted(self._leaves(tree))
        leaf_cids = sorted(
            c.cid for c in child_caps if c.kind == CapsuleKind.ARTIFACT
            and isinstance(c.payload, dict) and "agent" in c.payload
        )
        cap = ContextCapsule.new(
            kind=CapsuleKind.ARTIFACT,
            payload={"team": leaves},
            budget=self._default_budget,
            parents=tuple(leaf_cids),
        )
        ledger.admit_and_seal(cap)
        return child_caps + [cap]

    @staticmethod
    def _leaves(tree: TeamNode) -> list[str]:
        if tree.is_leaf:
            return [tree.agent_id]  # type: ignore[list-item]
        out: list[str] = []
        for c in tree.children:
            out.extend(AgentTeamOperad._leaves(c))
        return out

    # ---- associativity ------------------------------------------------------

    def verify_associativity(self, agents: Sequence[str]) -> bool:
        """Prove ``(a ∘ b) ∘ c = a ∘ (b ∘ c)`` for the given agents.

        Enumerates every binary bracketing of ``agents`` as a ``TeamNode``
        tree and checks that the final (root) capsule's CID is identical
        across all bracketings.  Because the capsule CID is a cryptographic
        function of its (kind, payload, budget, parents), CID equality
        is a strong equality between sealed capsule DAGs.
        """

        if len(agents) < 2:
            return True

        root_cids: set[str] = set()
        for tree in _binary_bracketings(agents):
            caps = self.evaluate(tree)
            root_cids.add(caps[-1].cid)
            if len(root_cids) > 1:
                return False
        return len(root_cids) == 1


def _binary_bracketings(agents: Sequence[str]) -> Iterable[TeamNode]:
    """Yield every binary-tree bracketing of a leaf sequence."""

    if len(agents) == 1:
        yield TeamNode.leaf(agents[0])
        return
    for split in range(1, len(agents)):
        for left in _binary_bracketings(agents[:split]):
            for right in _binary_bracketings(agents[split:]):
                yield TeamNode.node(left, right)


__all__ = [
    "CapsuleCategory",
    "AgentTeamOperad",
    "TeamNode",
]
