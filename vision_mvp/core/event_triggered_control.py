"""Event-triggered control — Lyapunov-grounded broadcast suppression.

The existing `core/event_trigger.py` is a schema-level disagreement detector
for LLM refinement. This module is the *control-theoretic* event-trigger layer
for the numeric CASR stack: for a linear discrete system

    x_{k+1} = A x_k

each agent keeps a last-broadcast estimate x̂_k. Broadcast only when the
error e_k = x_k − x̂_k crosses a Lyapunov threshold:

    ‖e_k‖² > σ ‖x_k‖²

Heemels, Johansson, Tabuada (2012) show that for any 0 < σ < σ*, the closed-
loop system remains input-to-state stable, where

    σ* = λ_min(Q) / λ_max(AᵀPA)

with P the discrete Lyapunov solution of AᵀPA − P = −Q. This upgrades the
Stage-3 surprise filter from an empirical heuristic to a controller with
provable ISS guarantees — directly grounding the "suppress unsurprising
events" idea that CASR currently uses without a stability proof.

For scalability beyond scipy we provide a pure-NumPy Lyapunov solver via
iterative P_{k+1} = AᵀP_kA + Q (geometrically convergent when ρ(A) < 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def spectral_radius(A: np.ndarray) -> float:
    """Largest |eigenvalue| of A."""
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def solve_discrete_lyapunov(
    A: np.ndarray, Q: np.ndarray,
    tol: float = 1e-10, max_iter: int = 10_000,
) -> np.ndarray:
    """Solve AᵀPA − P = −Q (Stein equation) by fixed-point iteration.

    Converges iff ρ(A) < 1. Each iteration is one matmul.
    """
    A = np.asarray(A, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if Q.shape != A.shape:
        raise ValueError("Q must match A's shape")
    if spectral_radius(A) >= 1.0:
        raise ValueError("A must be Schur-stable (ρ(A) < 1) for convergence")

    P = np.zeros_like(A)
    for _ in range(max_iter):
        P_new = A.T @ P @ A + Q
        if np.max(np.abs(P_new - P)) < tol:
            return P_new
        P = P_new
    return P


@dataclass
class EventTriggerReport:
    sigma_star: float        # maximum stable threshold
    lam_min_Q: float         # smallest eigenvalue of Q (>0 required)
    lam_max_AtPA: float      # largest eigenvalue of AᵀPA (bound)
    spectral_radius: float   # ρ(A); must be < 1
    P: np.ndarray            # Lyapunov matrix

    def summary(self) -> str:
        return (
            f"ρ(A)={self.spectral_radius:.3f}  "
            f"σ*={self.sigma_star:.4f}  "
            f"(λ_min(Q)={self.lam_min_Q:.3f}, "
            f"λ_max(AᵀPA)={self.lam_max_AtPA:.3f})"
        )


def synthesize_threshold(
    A: np.ndarray, Q: np.ndarray | None = None,
) -> EventTriggerReport:
    """Compute the maximum σ such that ‖e‖² > σ‖x‖² triggering preserves ISS.

    If Q is None, defaults to Q = I (identity). Raises if A is not Schur-stable.
    """
    A = np.asarray(A, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if Q is None:
        Q = np.eye(A.shape[0])
    else:
        Q = np.asarray(Q, dtype=float)

    rho = spectral_radius(A)
    if rho >= 1.0:
        raise ValueError(f"ρ(A)={rho:.4f} ≥ 1 — system is not Schur-stable")

    P = solve_discrete_lyapunov(A, Q)
    lam_min_Q = float(np.linalg.eigvalsh(Q).min())
    lam_max_AtPA = float(np.linalg.eigvalsh(A.T @ P @ A).max())
    if lam_max_AtPA <= 0:
        raise ValueError("AᵀPA has no positive eigenvalues — degenerate system")

    sigma_star = lam_min_Q / lam_max_AtPA
    return EventTriggerReport(
        sigma_star=sigma_star,
        lam_min_Q=lam_min_Q,
        lam_max_AtPA=lam_max_AtPA,
        spectral_radius=rho,
        P=P,
    )


@dataclass
class EventTrigger:
    """Per-agent trigger with provable ISS when σ < σ*."""

    sigma: float

    def should_broadcast(
        self, state: np.ndarray, last_broadcast: np.ndarray,
    ) -> bool:
        """Threshold check: ‖e‖² > σ ‖x‖².

        With ‖x‖ = 0, the only-when-x-zero edge case: state is exactly at
        equilibrium; the error must be exactly 0 to skip. Any nonzero error
        triggers.
        """
        state = np.asarray(state, dtype=float).ravel()
        last = np.asarray(last_broadcast, dtype=float).ravel()
        if state.shape != last.shape:
            raise ValueError("state and last_broadcast must match shape")
        e_norm2 = float(np.dot(state - last, state - last))
        x_norm2 = float(np.dot(state, state))
        if x_norm2 < 1e-12:
            return e_norm2 > 1e-12
        return e_norm2 > self.sigma * x_norm2


def simulate(
    A: np.ndarray,
    x0: np.ndarray,
    sigma: float,
    n_steps: int = 100,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run the triggered-broadcast closed loop for `n_steps`.

    Each step:
      1. If ‖x − x̂‖² > σ‖x‖², broadcast: set x̂ ← x, count the event.
      2. x_{k+1} = A x_k.

    Returns (state_trajectory, broadcast_mask, n_broadcasts).
    """
    A = np.asarray(A, dtype=float)
    x = np.asarray(x0, dtype=float).ravel().copy()
    x_hat = x.copy()
    states = [x.copy()]
    bcasts = []
    trig = EventTrigger(sigma=sigma)
    count = 0
    for _ in range(n_steps):
        if trig.should_broadcast(x, x_hat):
            x_hat = x.copy()
            count += 1
            bcasts.append(True)
        else:
            bcasts.append(False)
        x = A @ x
        states.append(x.copy())
    return np.array(states), np.array(bcasts), count
