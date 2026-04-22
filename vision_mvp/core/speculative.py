"""Speculative decoding between agents — unbiased via rejection sampling.

Leviathan, Kalman, Matias (ICML 2023). A small "draft" model proposes tokens,
and the target model verifies them in parallel; accepted prefixes save the
target model from re-sampling. The crucial property is *unbiasedness*: the
resulting sampling distribution is *exactly* the target's distribution under
a carefully-designed rejection step.

For token i with draft probability q(·) and target probability p(·):
  - If p(x_i)/q(x_i) ≥ 1, accept.
  - Else accept with probability p(x_i)/q(x_i); on reject, resample from
    (p − q)_+ / ‖(p − q)_+‖₁ instead.

This module implements the math without requiring real LLMs. Agents pass
distributions rather than tokens; the rejection-sampling decision is
analytic. Plug this into any two-agent proposal/verify workflow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpecDecodeReport:
    accepted: bool
    token: int
    p_draft: float
    p_target: float
    acceptance_prob: float


def rejection_sample(
    p_draft: np.ndarray,
    p_target: np.ndarray,
    rng: np.random.Generator,
) -> SpecDecodeReport:
    """Sample one token from p_target using a draft sample from p_draft.

    Returns the decision (accept/reject) and the final token. `accepted=True`
    means the draft's sampled token was reused; `False` means we resampled
    from the "residual" distribution (p − q)_+.
    """
    p_draft = np.asarray(p_draft, dtype=float)
    p_target = np.asarray(p_target, dtype=float)
    if p_draft.shape != p_target.shape:
        raise ValueError("p_draft and p_target must have same shape")
    if not np.isclose(p_draft.sum(), 1.0, atol=1e-6):
        raise ValueError("p_draft must sum to 1")
    if not np.isclose(p_target.sum(), 1.0, atol=1e-6):
        raise ValueError("p_target must sum to 1")

    # 1. Sample a token from the draft.
    x = int(rng.choice(p_draft.size, p=p_draft))
    q_x = float(p_draft[x])
    p_x = float(p_target[x])
    if q_x < 1e-12:
        # Draft gave zero probability — defensively reject
        accept_prob = 0.0
    else:
        accept_prob = min(1.0, p_x / q_x)
    if rng.random() < accept_prob:
        return SpecDecodeReport(
            accepted=True, token=x,
            p_draft=q_x, p_target=p_x,
            acceptance_prob=accept_prob,
        )

    # 2. Reject — resample from (p − q)_+ normalised.
    residual = np.maximum(p_target - p_draft, 0.0)
    s = residual.sum()
    if s < 1e-12:
        # Degenerate: fallback to p_target
        residual = p_target
        s = residual.sum()
    residual = residual / s
    x2 = int(rng.choice(p_target.size, p=residual))
    return SpecDecodeReport(
        accepted=False, token=x2,
        p_draft=float(p_draft[x2]), p_target=float(p_target[x2]),
        acceptance_prob=accept_prob,
    )


def expected_acceptance_rate(
    p_draft: np.ndarray, p_target: np.ndarray,
) -> float:
    """Theoretical 1-step acceptance probability for given draft/target pair.

    E_{x ~ q}[min(1, p(x)/q(x))] = Σ_x min(q(x), p(x)) = 1 − ½ TV(q, p).
    """
    q = np.asarray(p_draft, dtype=float)
    p = np.asarray(p_target, dtype=float)
    return float(np.minimum(p, q).sum())


def speculative_multi_token(
    draft_fn,
    target_fn,
    n_tokens: int,
    rng: np.random.Generator,
) -> list[SpecDecodeReport]:
    """Speculative decoding for a sequence of tokens.

    draft_fn(history)  -> probability vector for the next token
    target_fn(history) -> the target's probability vector for the next token
    """
    history: list[int] = []
    reports = []
    while len(history) < n_tokens:
        q = np.asarray(draft_fn(history), dtype=float)
        p = np.asarray(target_fn(history), dtype=float)
        r = rejection_sample(q, p, rng)
        history.append(r.token)
        reports.append(r)
    return reports
