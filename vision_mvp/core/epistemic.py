"""Dynamic Epistemic Logic (DEL) — Kripke models + common knowledge.

A (pointed) Kripke model is a tuple (W, {R_i}, V, w_0):

  W    : finite set of possible worlds
  R_i  : indistinguishability relation for agent i (equivalence relation)
  V    : valuation, maps each atomic proposition to the set of worlds where
         it holds
  w_0  : actual world

K_i φ ("agent i knows φ") holds at w iff φ holds at every w' with w R_i w'.
E_G φ ("every agent in G knows φ") is the finite conjunction K_i φ for i ∈ G.
C_G φ ("φ is common knowledge in G") is the greatest fixed point of the
monotone operator X ↦ E_G(φ ∧ X). For finite W this coincides with E_G^n φ
for n ≥ |W|.

Equivalent graph characterisation: C_G φ holds at w iff φ holds at every
world reachable from w under the transitive closure of (⋃_{i ∈ G} R_i).

This module implements the announcement fragment of DEL sufficient to encode
the Coordinated-Attack impossibility theorem and to measure depth-k mutual
knowledge inside a running CASR team.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KripkeModel:
    """Finite Kripke model for DEL."""

    worlds: list[str]
    # Agent -> list of equivalence classes (each class is a set of world names)
    relations: dict[str, list[set[str]]] = field(default_factory=dict)
    # Proposition -> set of worlds where it is true
    valuation: dict[str, set[str]] = field(default_factory=dict)
    actual: str = ""

    def __post_init__(self):
        if self.actual and self.actual not in self.worlds:
            raise ValueError(f"actual world {self.actual!r} not in worlds list")

    # --------- convenience constructors ----------

    def set_relation(self, agent: str, equivalence_classes: list[set[str]]) -> None:
        """Install agent's indistinguishability relation from a partition."""
        # sanity: partition must cover all worlds exactly once
        union = set().union(*equivalence_classes) if equivalence_classes else set()
        if union != set(self.worlds):
            raise ValueError(
                f"agent {agent}'s partition must cover all worlds exactly once"
            )
        self.relations[agent] = equivalence_classes

    def valuate(self, prop: str, worlds: set[str]) -> None:
        bad = worlds - set(self.worlds)
        if bad:
            raise ValueError(f"unknown worlds in valuation: {bad}")
        self.valuation[prop] = set(worlds)

    # --------- relation queries ----------

    def indistinguishable(self, agent: str, w: str) -> set[str]:
        """Worlds agent cannot distinguish from w."""
        if agent not in self.relations:
            raise ValueError(f"no relation for agent {agent}")
        for cls in self.relations[agent]:
            if w in cls:
                return set(cls)
        raise ValueError(f"world {w} not covered by {agent}'s partition")

    # --------- modalities ----------

    def knows(self, agent: str, prop: str, w: str) -> bool:
        """K_i p: agent i knows p at world w."""
        sat = self.valuation.get(prop, set())
        return self.indistinguishable(agent, w).issubset(sat)

    def everyone_knows(self, group: list[str], prop: str, w: str) -> bool:
        return all(self.knows(a, prop, w) for a in group)

    def mutual_knowledge_depth(self, group: list[str], prop: str, w: str) -> int:
        """Largest n for which E_G^n p holds at w, capped at |W|+1."""
        sat = set(self.valuation.get(prop, set()))
        if w not in sat:
            return 0
        current = sat
        depth = 0
        n_worlds = len(self.worlds)
        for _ in range(n_worlds + 1):
            # E_G(current) = { w' | for every agent i in group, R_i(w') ⊆ current }
            new = set()
            for wp in self.worlds:
                all_know = True
                for a in group:
                    if not self.indistinguishable(a, wp).issubset(current):
                        all_know = False
                        break
                if all_know:
                    new.add(wp)
            if not new or w not in new:
                return depth
            depth += 1
            if new == current:
                return depth
            current = new
        return depth

    def common_knowledge(self, group: list[str], prop: str, w: str) -> bool:
        """C_G p holds iff p holds at every world reachable from w under
        the transitive closure of ⋃_{i ∈ G} R_i.
        """
        sat = self.valuation.get(prop, set())
        # BFS from w under the union of agent partitions
        seen = {w}
        frontier = {w}
        while frontier:
            nxt: set[str] = set()
            for wp in frontier:
                for a in group:
                    nxt |= self.indistinguishable(a, wp)
            nxt -= seen
            seen |= nxt
            frontier = nxt
        return seen.issubset(sat)

    # --------- public-announcement update ----------

    def announce(self, prop: str) -> "KripkeModel":
        """Public-announcement update: restrict model to worlds where prop holds."""
        keep = self.valuation.get(prop, set())
        new_worlds = [w for w in self.worlds if w in keep]
        new_rel = {}
        for a, cls in self.relations.items():
            new_rel[a] = [c & keep for c in cls if (c & keep)]
        new_val = {p: (ws & keep) for p, ws in self.valuation.items()}
        actual = self.actual if self.actual in keep else ""
        return KripkeModel(
            worlds=new_worlds, relations=new_rel, valuation=new_val, actual=actual,
        )


def coordinated_attack_model() -> KripkeModel:
    """The classic 2-general coordinated-attack Kripke model (Halpern-Moses).

    Two worlds: "attack_ok" and "don't_attack". Agent A observes the plan
    (can distinguish the two); agent B cannot, as long as the messenger has
    not arrived. No public-announcement can force common knowledge of
    "attack_ok" without at least one side being able to distinguish the
    worlds — replicating the classical impossibility.
    """
    m = KripkeModel(worlds=["attack_ok", "dont_attack"], actual="attack_ok")
    m.set_relation("A", [{"attack_ok"}, {"dont_attack"}])
    # B cannot tell the two worlds apart
    m.set_relation("B", [{"attack_ok", "dont_attack"}])
    m.valuate("attack_ok", {"attack_ok"})
    return m
