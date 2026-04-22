"""BKT vortex diagnostic for 2-D angular team fields.

The Berezinskii–Kosterlitz–Thouless transition (1970s) is a topological phase
transition in 2-D XY-models where bound vortex pairs unbind into free
vortices. In a coordination context: if we lay agents on a 2-D grid and give
each a phase θ_i, a *vortex* is a plaquette (2×2 cell) around which the
cumulative phase difference is ±2π. Below the BKT temperature, vortices come
in tight ± pairs; above, they unbind into free vortices — diagnostic of a
team fragmenting into locally-coordinated but globally-disordered regions.

This module computes vortex density on a rectangular grid of phase
assignments. It's a pure topology diagnostic — no dynamics assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_angle(delta: np.ndarray) -> np.ndarray:
    """Wrap each entry of delta into (-π, π]."""
    return (delta + np.pi) % (2 * np.pi) - np.pi


@dataclass
class BKTReport:
    n_plus_vortices: int
    n_minus_vortices: int
    total_vortices: int
    bound_pair_fraction: float    # fraction of + vortices with a − nearby
    density: float                 # total vortices / # plaquettes

    def summary(self) -> str:
        return (
            f"vortices +/−: {self.n_plus_vortices}/{self.n_minus_vortices}  "
            f"density={self.density:.4f}  "
            f"bound={self.bound_pair_fraction:.2f}"
        )


def plaquette_charges(phases: np.ndarray) -> np.ndarray:
    """Signed vortex charge per plaquette on a (H, W) phase grid.

    Walks each plaquette (i, j) clockwise:
        (i,j) → (i,j+1) → (i+1,j+1) → (i+1,j) → (i,j)
    and accumulates wrapped phase differences. Returns (H-1, W-1) array of
    {-1, 0, +1} integers (the winding number).
    """
    phi = np.asarray(phases, dtype=float)
    if phi.ndim != 2:
        raise ValueError("phases must be a 2-D grid")
    H, W = phi.shape
    if H < 2 or W < 2:
        return np.zeros((max(H - 1, 0), max(W - 1, 0)), dtype=int)

    # Differences along each edge, wrapped.
    d_right = wrap_angle(phi[:, 1:] - phi[:, :-1])     # (H, W-1)
    d_down = wrap_angle(phi[1:, :] - phi[:-1, :])      # (H-1, W)

    # Clockwise walk: +right (top), +down (right col), -right (bottom), -down (left col).
    top = d_right[:-1, :]      # (H-1, W-1)
    right = d_down[:, 1:]      # (H-1, W-1)
    bottom = -d_right[1:, :]   # (H-1, W-1)
    left = -d_down[:, :-1]     # (H-1, W-1)

    total = top + right + bottom + left
    # Round to integer winding
    return np.round(total / (2 * np.pi)).astype(int)


def bound_pair_fraction(charges: np.ndarray) -> float:
    """Fraction of + vortices with a − vortex at Chebyshev distance ≤ 2."""
    pluses = np.argwhere(charges == 1)
    minuses = np.argwhere(charges == -1)
    if pluses.size == 0:
        return 1.0
    bound = 0
    for p in pluses:
        if minuses.size == 0:
            continue
        dists = np.max(np.abs(minuses - p), axis=1)
        if dists.min() <= 2:
            bound += 1
    return bound / len(pluses)


def bkt_report(phases: np.ndarray) -> BKTReport:
    charges = plaquette_charges(phases)
    n_plus = int(np.sum(charges == 1))
    n_minus = int(np.sum(charges == -1))
    total = n_plus + n_minus
    n_plaq = charges.size if charges.size > 0 else 1
    return BKTReport(
        n_plus_vortices=n_plus,
        n_minus_vortices=n_minus,
        total_vortices=total,
        bound_pair_fraction=bound_pair_fraction(charges),
        density=total / n_plaq,
    )
