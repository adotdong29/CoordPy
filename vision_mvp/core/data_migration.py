"""Functorial data migration — Δ, Σ, Π over finite category schemas.

Spivak (2012). Schemas = finite categories (objects = types, morphisms =
foreign-key-like functions). An instance of schema S is a functor S → Set;
a schema morphism F: S → T induces three adjoint functors on instances:

  Δ_F  (pullback)     : inst(T) → inst(S)   — "rename / project"
  Σ_F  (left adjoint) : inst(S) → inst(T)   — "union over fibers"
  Π_F  (right adjoint): inst(S) → inst(T)   — "product over fibers"

These compose the data-integration primitives that make heterogeneous-agent
interop sound. In CASR: agents with different schemas (A, B) are bridged
via a schema morphism; pulling (Δ) or pushing (Σ / Π) their instances over
the morphism gives the correct merged dataset without ad-hoc glue.

Implementation here is the simplest finite-set version: objects = strings,
morphisms = dict mappings, instances = dicts of object -> set of elements,
with morphism "actions" as dicts element -> element.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Schema:
    """Finite category: objects + morphisms + composition closure.

    `morphisms` maps (src, dst, name) → None (presence indicates existence);
    compositions are checked at runtime.
    """
    objects: set[str] = field(default_factory=set)
    morphisms: dict[tuple[str, str, str], None] = field(default_factory=dict)

    def add_object(self, obj: str) -> None:
        self.objects.add(obj)

    def add_morphism(self, src: str, dst: str, name: str) -> None:
        if src not in self.objects or dst not in self.objects:
            raise ValueError("src and dst must be objects in the schema")
        self.morphisms[(src, dst, name)] = None


@dataclass
class Instance:
    """Functor S → Set: each object gets a set; each morphism gets a function."""
    schema: Schema
    data: dict[str, set] = field(default_factory=dict)
    mor_action: dict[tuple[str, str, str], dict] = field(default_factory=dict)

    def set_object(self, obj: str, elements: set) -> None:
        if obj not in self.schema.objects:
            raise ValueError(f"unknown object {obj}")
        self.data[obj] = set(elements)

    def set_morphism(
        self, src: str, dst: str, name: str, action: dict,
    ) -> None:
        key = (src, dst, name)
        if key not in self.schema.morphisms:
            raise ValueError(f"no such morphism: {key}")
        # Action must be total on self.data[src] and land in self.data[dst]
        for x in self.data.get(src, set()):
            if x not in action:
                raise ValueError(f"morphism {key} is not total on {src}")
            if action[x] not in self.data.get(dst, set()):
                raise ValueError(
                    f"morphism {key} carries {x} outside {dst}={self.data.get(dst, set())}"
                )
        self.mor_action[key] = dict(action)


@dataclass
class SchemaMorphism:
    """F: S → T. obj_map[s] ∈ T.objects; mor_map[(src, dst, name)] is the
    corresponding T-morphism (src', dst', name').
    """
    source: Schema
    target: Schema
    obj_map: dict[str, str] = field(default_factory=dict)
    mor_map: dict[tuple[str, str, str], tuple[str, str, str]] = field(
        default_factory=dict
    )


# ----------------- Delta (pullback) -----------------

def delta(F: SchemaMorphism, instance_T: Instance) -> Instance:
    """Δ_F: an S-instance whose object-set is the T-instance's image.

    For every S-object s, Δ_F(I)(s) = I(F(s)).
    For every S-morphism m : s → s', Δ_F(I)(m) = I(F(m)).
    """
    I_out = Instance(schema=F.source)
    for s in F.source.objects:
        t = F.obj_map[s]
        I_out.data[s] = set(instance_T.data.get(t, set()))
    for (src, dst, name), t_mor in F.mor_map.items():
        if t_mor in instance_T.mor_action:
            I_out.mor_action[(src, dst, name)] = dict(instance_T.mor_action[t_mor])
    return I_out


# ----------------- Sigma (left adjoint) -----------------

def sigma(F: SchemaMorphism, instance_S: Instance) -> Instance:
    """Σ_F: T-instance whose elements are union-over-fibers of the S-instance.

    Σ_F(I)(t) = ⋃_{s | F(s) = t} I(s)  (as a disjoint union, represented here
    as a union with tags (s, x) to keep elements distinct).
    """
    I_out = Instance(schema=F.target)
    for t in F.target.objects:
        acc = set()
        for s in F.source.objects:
            if F.obj_map.get(s) == t:
                for x in instance_S.data.get(s, set()):
                    acc.add((s, x))
        I_out.data[t] = acc
    return I_out


# ----------------- Pi (right adjoint) -----------------

def pi(F: SchemaMorphism, instance_S: Instance) -> Instance:
    """Π_F: T-instance whose elements are products over fibers.

    Π_F(I)(t) = ∏_{s | F(s) = t} I(s)  (as tuples, one entry per fibre member).
    """
    I_out = Instance(schema=F.target)
    for t in F.target.objects:
        fiber = sorted([s for s in F.source.objects if F.obj_map.get(s) == t])
        if not fiber:
            I_out.data[t] = set()
            continue

        # Cartesian product of the fibre's instance sets
        from itertools import product as cart
        choice_sets = [list(instance_S.data.get(s, set())) for s in fiber]
        if any(not c for c in choice_sets):
            I_out.data[t] = set()
            continue
        products = set()
        for tup in cart(*choice_sets):
            products.add(tuple(zip(fiber, tup)))
        I_out.data[t] = products
    return I_out
