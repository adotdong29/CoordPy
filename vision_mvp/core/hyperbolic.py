"""Hyperbolic embeddings for agent address space (Lorentz model).

Why hyperbolic? A tree with branching factor b has b^d nodes at depth d —
matching hyperbolic space, which has exponential volume growth with radius.
Trees embed in hyperbolic space with arbitrarily low distortion, whereas
Euclidean embeddings suffer Ω(log n) distortion (Linial-London-Rabinovich,
Bourgain).

For a task-decomposition tree with hundreds of agents organized by
specialty, placing each agent at a point in H^n lets every subtree have
its own exponentially-sized "room" at constant radius — sibling subtrees
don't leak messages into each other.

Implementation: Lorentz model (also called hyperboloid model). Each point
x ∈ H^n satisfies ⟨x, x⟩_L = -x_0^2 + Σ x_i^2 = -1.

Distance:
  d_L(x, y) = arcosh( -⟨x, y⟩_L )

This module is a self-contained, pure-numpy implementation — no torch.
For the scale we care about (≤ 1000 agents, d ≤ 8), it's plenty fast.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field


def minkowski_inner(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Lorentz inner product: -x0*y0 + Σ xi*yi."""
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(axis=-1)


def lorentz_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Hyperbolic distance between x and y on H^n (Lorentz model).

    For valid points on H^n, inner ≤ -1, so -inner ≥ 1. Numerical noise
    can make -inner slightly below 1, which would make arccosh NaN.
    We clamp from below to 1.0 exactly so self-distance is exactly 0.
    """
    inner = minkowski_inner(x, y)
    arg = np.maximum(-inner, 1.0)     # clamp from below to 1.0
    return np.arccosh(arg)


def project_to_hyperboloid(x: np.ndarray) -> np.ndarray:
    """Given a point near the hyperboloid, project onto it by setting
    x_0 = sqrt(1 + ||x_{1:}||^2)."""
    x = x.copy()
    spatial = x[..., 1:]
    sq = (spatial * spatial).sum(axis=-1)
    x[..., 0] = np.sqrt(1.0 + sq)
    return x


def exp_map(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Exponential map at x in direction v (tangent vector).

    exp_x(v) = cosh(||v||) * x + sinh(||v||) * v / ||v||

    For a tangent vector v ∈ T_x H^n, the Minkowski inner product
    ⟨v, v⟩_L = -v_0² + Σv_i² is POSITIVE (v is spacelike). That's the
    squared Riemannian norm we want.
    """
    vv = minkowski_inner(v, v)              # ≥ 0 for tangent vectors
    vv = np.maximum(vv, 1e-12)
    norm = np.sqrt(vv)
    n = norm[..., None] if x.ndim > 1 else norm
    return np.cosh(n) * x + np.sinh(n) * v / (n + 1e-12)


def log_map(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Logarithmic map: given points x, y on H^n, return the tangent vector
    at x pointing toward y."""
    inner = minkowski_inner(x, y)
    arg = np.maximum(-inner, 1.0)
    d = np.arccosh(arg)
    w = y + inner[..., None] * x
    wnorm = np.sqrt(np.maximum(minkowski_inner(w, w), 1e-12))
    return d[..., None] * w / (wnorm[..., None] + 1e-12)


def origin(dim: int) -> np.ndarray:
    """Origin of H^dim in Lorentz coordinates: (1, 0, 0, …, 0)."""
    o = np.zeros(dim + 1)
    o[0] = 1.0
    return o


@dataclass
class HyperbolicAddressBook:
    """Assign each agent a point in H^n based on a task-decomposition tree.

    Tree node → hyperbolic point by placing the root at origin and each
    child at a fixed distance δ from its parent, separated from siblings
    by angle 2π/branching.

    Agents subscribe to a HOROBALL around their assigned point; a message
    published at point μ is delivered to agent i iff the Lorentzian inner
    product satisfies a horoball membership test.
    """
    dim: int = 2     # smallest hyperbolic dim that still captures structure
    branch_step: float = 1.0
    _coords: dict = field(default_factory=dict)  # agent_id -> (D+1,) array
    _horoball_params: dict = field(default_factory=dict)

    def embed_tree(self, tree: dict) -> None:
        """Embed a task-tree (dict: agent_id -> list of child agent_ids).

        Places root at origin(D), then recursively places children.
        """
        roots = set(tree.keys()) - {c for cs in tree.values() for c in cs}
        # Single synthetic root to anchor multiple roots if needed
        o = origin(self.dim)
        real_roots = sorted(roots)
        if len(real_roots) == 1:
            self._coords[real_roots[0]] = o
            self._place_children(real_roots[0], tree, depth=1)
        else:
            # Distribute roots around a circle on the first sphere
            n = len(real_roots)
            for i, r in enumerate(real_roots):
                angle = 2 * math.pi * i / max(n, 1)
                v = np.zeros(self.dim + 1)
                v[1] = self.branch_step * math.cos(angle)
                v[2 % (self.dim + 1)] = self.branch_step * math.sin(angle)
                self._coords[r] = exp_map(o, v)
                self._place_children(r, tree, depth=1)

    def _place_children(self, parent_id, tree, depth):
        kids = tree.get(parent_id, [])
        if not kids:
            return
        parent_pt = self._coords[parent_id]
        n = len(kids)
        # Tangent direction pointing "outward" — we use a simple approach:
        # step in orthogonal directions, preserving the hyperboloid.
        for i, kid in enumerate(kids):
            angle = 2 * math.pi * i / n
            v = np.zeros(self.dim + 1)
            # spatial dims: put component in dim 1 and dim 2 for 2D, etc.
            v[1 % (self.dim + 1)] += self.branch_step * math.cos(angle)
            v[2 % (self.dim + 1)] += self.branch_step * math.sin(angle)
            # Project v to tangent space at parent_pt
            v_tan = v + minkowski_inner(parent_pt, v) * parent_pt
            self._coords[kid] = exp_map(parent_pt, v_tan)
            self._place_children(kid, tree, depth + 1)

    def distance(self, agent_a: int, agent_b: int) -> float:
        return float(lorentz_distance(self._coords[agent_a], self._coords[agent_b]))

    def set_coord(self, agent_id: int, coord: np.ndarray) -> None:
        self._coords[agent_id] = project_to_hyperboloid(coord)

    def get_coord(self, agent_id: int) -> np.ndarray:
        return self._coords[agent_id].copy()

    def recipients_within(self, message_point: np.ndarray, radius: float) -> list[int]:
        """Return all agents within hyperbolic `radius` of message_point."""
        out = []
        for aid, coord in self._coords.items():
            if lorentz_distance(message_point, coord) <= radius:
                out.append(aid)
        return out
