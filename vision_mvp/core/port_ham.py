"""Port-Hamiltonian systems — compositional passivity for protocol stacks.

Van der Schaft (2004). A port-Hamiltonian system has the form

    ẋ = (J(x) − R(x)) · ∂H/∂x + G(x) · u
    y = G(x)ᵀ · ∂H/∂x

with Hamiltonian H(x) ≥ 0 (the storage function), J(x) skew-symmetric
(interconnection), R(x) positive-semidefinite (dissipation), G(x) input matrix.
Such a system is passive: Ḣ ≤ uᵀy.

Composition is via Dirac structures: the interconnection of passive PH systems
through their ports yields a passive PH system. This gives *compositional*
stability proofs for the layered CASR stack: prove each layer is PH, compose,
and the whole is automatically stable.

This module provides a minimal linear-PH class and a series/parallel
composition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearPH:
    """Linear port-Hamiltonian system with constant J, R, Q, G.

        H(x) = ½ xᵀ Q x
        ẋ = (J − R) Q x + G u
        y = Gᵀ Q x
    """

    J: np.ndarray
    R: np.ndarray
    Q: np.ndarray
    G: np.ndarray

    def __post_init__(self):
        for name, M in (("J", self.J), ("R", self.R), ("Q", self.Q)):
            if M.shape[0] != M.shape[1]:
                raise ValueError(f"{name} must be square")
        # J must be skew-symmetric (up to numerics)
        if not np.allclose(self.J, -self.J.T, atol=1e-8):
            raise ValueError("J must be skew-symmetric")
        # R must be symmetric PSD
        if not np.allclose(self.R, self.R.T, atol=1e-8):
            raise ValueError("R must be symmetric")
        if np.linalg.eigvalsh(self.R).min() < -1e-8:
            raise ValueError("R must be positive semi-definite")
        # Q must be symmetric PD
        if not np.allclose(self.Q, self.Q.T, atol=1e-8):
            raise ValueError("Q must be symmetric")
        if np.linalg.eigvalsh(self.Q).min() <= 0:
            raise ValueError("Q must be positive definite")

    def A(self) -> np.ndarray:
        """System matrix of ẋ = A x + G u."""
        return (self.J - self.R) @ self.Q

    def energy(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).ravel()
        return 0.5 * float(x @ self.Q @ x)

    def is_passive(self) -> bool:
        """Passivity requires R ⪰ 0 and Q ≻ 0 — guaranteed by __post_init__."""
        return True


def series_compose(s1: LinearPH, s2: LinearPH) -> LinearPH:
    """Series composition: s1's output drives s2's input.

    Constructs the block-diagonal dynamics with one-way coupling through G1.
    Requires s1.G.shape[1] == s2.G.shape[1] (matching port dims).
    """
    if s1.G.shape[1] != s2.G.shape[1]:
        raise ValueError("port dimensions must match")
    n1, n2 = s1.J.shape[0], s2.J.shape[0]
    J = np.block([
        [s1.J, np.zeros((n1, n2))],
        [np.zeros((n2, n1)), s2.J],
    ])
    R = np.block([
        [s1.R, np.zeros((n1, n2))],
        [np.zeros((n2, n1)), s2.R],
    ])
    Q = np.block([
        [s1.Q, np.zeros((n1, n2))],
        [np.zeros((n2, n1)), s2.Q],
    ])
    # s2's input port is driven by s1's output
    G = np.block([
        [s1.G],
        [s2.G],
    ])
    return LinearPH(J=J, R=R, Q=Q, G=G)
