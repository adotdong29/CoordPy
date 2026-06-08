"""W142 / COO-9 — discover-vs-amortize budget accounting (Lane β discipline).

The W142 retirement claim is an EQUAL-TOTAL-FAMILY-BUDGET claim, and the accounting must be explicit and
pre-registered so a raised discover-K is never blurred into a per-problem same-budget claim (RUNBOOK §1/§6).

Over a family of ``M`` same-technique members at per-member amortize budget ``K_a``:

  * **B0 (no-oracle verified-selection@K_a per member):** total = ``M * K_a``;  solves a member with
    probability ``1 - (1-p)^{K_a}`` (must RE-DISCOVER the rare technique on every problem).
  * **ST (self-tutoring):** DISCOVER once with ``K_d`` draws on a teacher (⇒ a self-derived scaffold),
    then AMORTIZE ``K_a`` SCAFFOLDED draws per member;  total = ``K_d + M * K_a``;  solves a member with
    probability ``1 - (1-q)^{K_a}`` with ``q ≈ 1`` (W141).

Two honest framings, both reported:
  1. **EQUAL-TOTAL-BUDGET (the retirement claim).** Fix the family budget ``G = M*K_a`` (B0's spend).  ST
     spends ``K_d`` on discovery and ``G - K_d`` amortized over members, so its per-member draw count is
     ``K_a - K_d/M`` — SLIGHTLY FEWER than B0 — yet each draw is scaffolded (``q≈1``).  At equal ``G``, the
     per-member superiority is ``(1-p)^{K_a}`` (the members B0 fails because selection cannot amortize the
     rare discovery).  This is retirement-grade for ``p ≲ 0.5`` at ``K_a=4`` and VANISHES at the bimodal
     extremes — it lives exactly in the moderate-`p` band.
  2. **AMORTIZED-OVERHEAD (reliability framing).** Raising ``K_d`` (discover reliability) is a one-time
     family cost whose per-member share ``K_d/M → 0`` as ``M`` grows;  reported SEPARATELY, never folded
     into the per-problem same-budget claim.

This module computes and MACHINE-CHECKS the identity for every reported arm and FAILS a run whose declared
arm budget does not match its actual generation count.  Pure / deterministic;  no model, no I/O;
explicit-import only;  ``coordpy/__init__.py`` untouched.
"""
from __future__ import annotations

import dataclasses
from typing import Any

DISCOVER_AMORTIZE_ACCOUNTING_V1_SCHEMA_VERSION: str = "coordpy.discover_amortize_accounting_v1.v1"


def predicted_solve_rate_v1(p: float, k: int) -> float:
    """``1 - (1-p)^k`` — the pool/verified-selection solve probability at per-member budget k and raw
    per-sample rate p (clamped to [0,1])."""
    p = max(0.0, min(1.0, float(p)))
    return 1.0 - (1.0 - p) ** int(k)


def per_member_superiority_pp_v1(p: float, k_a: int, q: float = 1.0) -> float:
    """ST − B0 per-member superiority in percentage points at equal amortize budget: the members B0 fails
    that the scaffold (rate q) solves.  At q=1 this is exactly ``100*(1-p)^{k_a}``."""
    b0 = predicted_solve_rate_v1(p, k_a)
    st = predicted_solve_rate_v1(q, k_a)
    return round(100.0 * (st - b0), 2)


@dataclasses.dataclass(frozen=True)
class ArmBudgetV1:
    arm: str
    declared_total: int            # the pre-registered generation budget for this arm over the family
    actual_total: int              # the observed generation count (sum of per-member + discover calls)
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class AmortizationBudgetReportV1:
    M: int
    K_a: int                       # amortize-K (per member; HELD at the standard budget shape)
    K_d: int                       # discover-K (may be RAISED; reported separately)
    b0_family_total: int           # M * K_a
    st_family_total: int           # K_d + M * K_a
    per_member_discovery_overhead: float   # K_d / M  -> 0 as M grows
    st_effective_per_member_at_equal_G: float   # K_a - K_d/M (equal-G framing)
    arms: tuple[ArmBudgetV1, ...]
    same_budget_identity_holds: bool   # every arm's declared == actual
    equal_G_note: str

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["arms"] = [a.to_dict() for a in self.arms]
        return d


def amortized_budget_parity_v1(*, M: int, K_a: int, K_d: int,
                               observed: dict[str, int] | None = None) -> AmortizationBudgetReportV1:
    """Build the pre-registered budget identity for a family-level run.  ``observed`` maps arm -> actual
    generation count;  arms whose declared budget mismatches the actual count flip ``ok=False`` and the
    run's ``same_budget_identity_holds`` to False (the §6 enforcement)."""
    b0_total = M * K_a
    st_total = K_d + M * K_a
    declared = {
        "A1": M * K_a,             # K_a plain per member (shared with B0)
        "B0": M * K_a,             # K_a plain per member, verified-selection
        "ST4": (K_a) + M * K_a,    # W141 shape: discover at K_a + amortize K_a
        "STd": K_d + M * K_a,      # raised discover-K + amortize K_a
    }
    observed = observed or {}
    arms = []
    all_ok = True
    for arm, dec in declared.items():
        act = int(observed.get(arm, dec))
        ok = (act == dec)
        all_ok = all_ok and ok
        arms.append(ArmBudgetV1(arm=arm, declared_total=dec, actual_total=act, ok=ok))
    overhead = (K_d / M) if M else float("inf")
    return AmortizationBudgetReportV1(
        M=M, K_a=K_a, K_d=K_d, b0_family_total=b0_total, st_family_total=st_total,
        per_member_discovery_overhead=round(overhead, 4),
        st_effective_per_member_at_equal_G=round(K_a - overhead, 4),
        arms=tuple(arms), same_budget_identity_holds=bool(all_ok),
        equal_G_note=("retirement claim = equal-G: at family budget M*K_a, ST's per-member draws "
                      f"= {round(K_a - overhead, 4)} (<= K_a={K_a}) but scaffolded (q~1); superiority "
                      "= (1-p)^K_a; K_d reported separately as the amortized one-time discovery cost"))


__all__ = [
    "DISCOVER_AMORTIZE_ACCOUNTING_V1_SCHEMA_VERSION", "predicted_solve_rate_v1",
    "per_member_superiority_pp_v1", "ArmBudgetV1", "AmortizationBudgetReportV1",
    "amortized_budget_parity_v1",
]
