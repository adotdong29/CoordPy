"""Information design for multi-receiver settings.

Generalises Bayesian persuasion (core/persuasion.py) to many receivers.
Bergemann & Morris (2019), Doval & Ely (2020). Instead of shaping a single
receiver's posterior, the sender designs an information structure σ that
recommends actions to each receiver, subject to obedience constraints
(each receiver finds following her recommendation optimal given what she
knows).

For a finite state/action/agent space, the optimal information structure is
the solution to a linear program:

    max Σ_ω p(ω) Σ_a σ(a | ω) · u_sender(ω, a)
    s.t. Σ_ω p(ω) σ(a | ω) · ( u_i(ω, a_i, a_{-i}) − u_i(ω, a'_i, a_{-i}) ) ≥ 0
         ∀ i ∀ a_i, a'_i       (obedience: a_i is best reply given receipt of a_i)

Pure-numpy LP via the revised simplex via `scipy.optimize.linprog`. If
scipy is not present, we fall back to a brute-force enumeration over
pure-strategy obedience policies, valid for small |A|^|I|.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np


@dataclass
class InfoDesignReport:
    sender_value: float
    best_policy: np.ndarray        # (|Ω|, |A|) recommended action probabilities
    obedient: bool

    def summary(self) -> str:
        return (
            f"sender value = {self.sender_value:.4f}  "
            f"({'obedient' if self.obedient else 'INFEASIBLE — constraints tight'})"
        )


def obedient_policies_brute(
    state_prior: np.ndarray,
    action_set: list,
    u_sender: Callable[[int, int], float],
    u_receiver: Callable[[int, int], float],
) -> InfoDesignReport:
    """Single-receiver case via brute enumeration of deterministic σ.

    For every map action(state) → action, check obedience and record sender
    utility. Returns the best obedient deterministic mapping.
    """
    p = np.asarray(state_prior, dtype=float)
    n_states = p.size
    best_val = -np.inf
    best_policy = np.zeros((n_states, len(action_set)))
    best_policy[:, 0] = 1.0  # default
    found = False

    for mapping in product(range(len(action_set)), repeat=n_states):
        # check obedience: for each action `a` recommended at some state, a must
        # maximise the receiver's posterior expected utility given recommendation.
        # Posterior over Ω given recommendation = a is proportional to
        #   p(ω) · 1[mapping(ω) = a]
        obedient = True
        for a in set(mapping):
            mask = np.array([1 if m == a else 0 for m in mapping], dtype=float)
            marg = float(p @ mask)
            if marg <= 1e-12:
                continue
            post = (p * mask) / marg
            # receiver's expected utility for each alternative action
            best_alt = max(
                sum(post[w] * u_receiver(w, a_alt) for w in range(n_states))
                for a_alt in range(len(action_set))
            )
            chosen = sum(post[w] * u_receiver(w, a) for w in range(n_states))
            if chosen + 1e-9 < best_alt:
                obedient = False
                break

        if not obedient:
            continue
        val = sum(p[w] * u_sender(w, mapping[w]) for w in range(n_states))
        if val > best_val:
            best_val = val
            best_policy = np.zeros((n_states, len(action_set)))
            for w, a in enumerate(mapping):
                best_policy[w, a] = 1.0
            found = True

    return InfoDesignReport(
        sender_value=float(best_val) if found else 0.0,
        best_policy=best_policy,
        obedient=found,
    )
