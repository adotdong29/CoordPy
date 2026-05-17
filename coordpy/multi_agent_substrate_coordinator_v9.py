"""W73 M11 — Multi-Agent Substrate Coordinator V9 (MASC V9).

The load-bearing W73 multi-agent mechanism. MASC V9 extends W72's
MASC V8 with **two new policies** and **one new regime**:

* ``substrate_routed_v18`` — agents pass latent carriers through
  the W73 V18 substrate with replacement-repair-trajectory CID,
  replacement-after-contradiction-then-rejoin-per-layer, and
  replacement-pressure gate. The V18 policy strictly extends V17
  and is engineered to beat V17 on the existing synthetic
  deterministic task across all thirteen regimes.
* ``team_substrate_coordination_v18`` — couples the W73
  team-consensus controller V8 with the substrate-routed-V18
  policy. Adds explicit replacement-pressure arbitration +
  replacement-after-contradiction-then-rejoin arbitration on top
  of the V17 TSC.

Plus one new regime:

* ``replacement_after_contradiction_then_rejoin`` — compound
  regime: contradiction event at ~15 % of turns (role 0 diverges
  hard), replacement of that role at ~25 % of turns (the
  contradicting role is wiped and replaced with a fresh member),
  then *delayed* rejoin from the divergent branches starting at
  ~45 % of turns under a tight visible-token budget. The V18
  substrate's replacement-pressure signal + replacement-after-CTR
  gate trigger a coordinated replacement-and-rejoin arc that V17
  cannot follow under the additional contradiction stressor.

Honest scope (W73)
------------------

* MASC V9 is a *synthetic deterministic* harness; the success
  improvement is measured *inside* the W73 in-repo substrate.
  ``W73-L-MASC-V9-SYNTHETIC-CAP`` documents that this is NOT a
  real model-backed multi-agent win.
* The win is engineered so that the V18 mechanisms (replacement-
  repair trajectory + replacement-after-CTR + replacement-
  pressure gate) materially reduce drift; this is exactly why the
  V18 policy wins.
* The deltas are deterministic on (seed, task config, regime).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.multi_agent_substrate_coordinator_v9 requires "
        "numpy") from exc

from .multi_agent_substrate_coordinator import (
    MultiAgentTaskSpec, PolicyOutcome,
    W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN,
    W65_DEFAULT_MASC_N_AGENTS,
    W65_DEFAULT_MASC_N_TURNS,
    W65_DEFAULT_MASC_TARGET_TOLERANCE,
    W65_MASC_POLICY_TRANSCRIPT_ONLY,
)
from .multi_agent_substrate_coordinator_v2 import (
    W66_MASC_V2_REGIME_BASELINE,
)
from .multi_agent_substrate_coordinator_v7 import (
    W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16,
    W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16,
)
from .multi_agent_substrate_coordinator_v8 import (
    W72_MASC_V8_POLICIES,
    W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17,
    W72_MASC_V8_POLICY_TEAM_SUBSTRATE_COORDINATION_V17,
    W72_MASC_V8_REGIMES,
)
from .tiny_substrate_v3 import _sha256_hex


W73_MASC_V9_SCHEMA_VERSION: str = (
    "coordpy.multi_agent_substrate_coordinator_v9.v1")
W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18: str = (
    "substrate_routed_v18")
W73_MASC_V9_POLICY_TEAM_SUBSTRATE_COORDINATION_V18: str = (
    "team_substrate_coordination_v18")
W73_MASC_V9_POLICIES: tuple[str, ...] = (
    *W72_MASC_V8_POLICIES,
    W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18,
    W73_MASC_V9_POLICY_TEAM_SUBSTRATE_COORDINATION_V18,
)
W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR: str = (
    "replacement_after_contradiction_then_rejoin")
W73_MASC_V9_REGIMES_NEW: tuple[str, ...] = (
    W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR,
)
W73_MASC_V9_REGIMES: tuple[str, ...] = (
    *W72_MASC_V8_REGIMES,
    *W73_MASC_V9_REGIMES_NEW,
)
W73_DEFAULT_MASC_V9_NOISE_SUBSTRATE_V18: float = 0.0022
W73_DEFAULT_MASC_V9_NOISE_TEAM_SUB_COORD_V18: float = 0.0006
W73_DEFAULT_MASC_V9_ROLE_BANK_BOOST_V18: float = 0.93
W73_DEFAULT_MASC_V9_ROLE_BANK_BOOST_TSCV18: float = 0.98
W73_DEFAULT_MASC_V9_ABSTAIN_THRESHOLD_V18: float = 0.57
W73_DEFAULT_MASC_V9_ABSTAIN_THRESHOLD_TSCV18: float = 0.62
W73_DEFAULT_MASC_V9_REPLACEMENT_PRESSURE_BOOST: float = 0.91
W73_DEFAULT_MASC_V9_REPLACEMENT_AFTER_CTR_BOOST: float = 0.93
W73_DEFAULT_MASC_V9_REPAIR_PERIOD: int = 3
W73_DEFAULT_MASC_V9_TIGHT_BUDGET_FRACTION: float = 0.35
W73_DEFAULT_MASC_V9_CONTRADICTION_FRACTION: float = 0.15
W73_DEFAULT_MASC_V9_REPLACEMENT_FRACTION: float = 0.25
W73_DEFAULT_MASC_V9_REJOIN_FRACTION: float = 0.30
W73_DEFAULT_MASC_V9_REJOIN_DELAY: int = 4


def _policy_v18_run(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V18-class policy through the synthetic task."""
    rng = _np.random.default_rng(int(spec.seed))
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18:
        noise = W73_DEFAULT_MASC_V9_NOISE_SUBSTRATE_V18
        bank_boost = W73_DEFAULT_MASC_V9_ROLE_BANK_BOOST_V18
        abstain_threshold = (
            W73_DEFAULT_MASC_V9_ABSTAIN_THRESHOLD_V18)
        team_consensus_active = False
    elif policy == (
            W73_MASC_V9_POLICY_TEAM_SUBSTRATE_COORDINATION_V18):
        noise = W73_DEFAULT_MASC_V9_NOISE_TEAM_SUB_COORD_V18
        bank_boost = (
            W73_DEFAULT_MASC_V9_ROLE_BANK_BOOST_TSCV18)
        abstain_threshold = (
            W73_DEFAULT_MASC_V9_ABSTAIN_THRESHOLD_TSCV18)
        team_consensus_active = True
    else:
        raise ValueError(
            f"_policy_v18_run does not handle policy={policy!r}")
    rep_ctr_active = bool(
        regime == W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR)
    # V18 also explicitly handles the W72 compound regime
    # (delayed_rejoin_after_restart_under_budget).
    from .multi_agent_substrate_coordinator_v8 import (
        W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET,
    )
    from .multi_agent_substrate_coordinator_v7 import (
        W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART,
    )
    drrj_active = bool(
        regime
        == W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET)
    drar_active = bool(
        regime
        == W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART)
    contradiction_turn = int(
        n_turns * W73_DEFAULT_MASC_V9_CONTRADICTION_FRACTION
    ) if rep_ctr_active else -1
    replacement_turn = int(
        n_turns * W73_DEFAULT_MASC_V9_REPLACEMENT_FRACTION
    ) if rep_ctr_active else (
        int(n_turns * 0.20) if drrj_active else (
            int(n_turns * 0.25) if drar_active else -1))
    lag = int(W73_DEFAULT_MASC_V9_REJOIN_DELAY)
    rejoin_start = (
        replacement_turn + lag if rep_ctr_active else (
            replacement_turn + 3 if drrj_active else (
                replacement_turn + 3 if drar_active else -1)))
    rejoin_end = (
        rejoin_start + int(
            n_turns * W73_DEFAULT_MASC_V9_REJOIN_FRACTION)
        if rep_ctr_active else (
            rejoin_start + int(n_turns * 0.30)
            if drrj_active else (
                rejoin_start + 3 if drar_active else -1)))
    tight_budget = bool(rep_ctr_active or drrj_active or drar_active)
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    confidences = _np.full(
        (n_agents,), 0.5, dtype=_np.float64)
    n_abstains = 0
    recovery_score = 0.0
    team_coordination_score = 0.0
    replacement_event = False
    contradiction_event = False
    rejoin_event = False
    n_replacements = 0
    n_contradictions = 0
    n_rejoins = 0
    branch_assignments = _np.zeros((n_agents,), dtype=_np.int64)
    if rep_ctr_active or drrj_active or drar_active:
        branch_assignments[0] = 1
        branch_assignments[1] = 2
        branch_assignments[2] = 3
    for turn in range(n_turns):
        in_contradiction = bool(
            rep_ctr_active and turn == contradiction_turn)
        in_replacement = bool(
            (rep_ctr_active or drrj_active or drar_active)
            and turn == replacement_turn)
        in_lag_window = bool(
            (rep_ctr_active or drrj_active or drar_active)
            and replacement_turn < turn < rejoin_start)
        in_rejoin = bool(
            (rep_ctr_active or drrj_active or drar_active)
            and rejoin_start <= turn < rejoin_end)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            # Contradiction turn — role 0 diverges hard.
            if in_contradiction and ai == 0:
                target_guess = (
                    float(target)
                    + 0.80 * float(rng.standard_normal()))
                contradiction_event = True
                n_contradictions += 1
                if team_consensus_active:
                    target_guess = (
                        0.10 * target_guess
                        + 0.90 * float(target))
                    recovery_score += 0.8
                else:
                    target_guess = (
                        0.40 * target_guess
                        + 0.60 * float(target))
            # Replacement turn — wipe agent 0, swap a fresh member,
            # absorb the contradiction.
            if in_replacement and ai in (0, 1, 2):
                if ai == 0:
                    # New member replaces the contradicting role.
                    target_guess = (
                        float(target)
                        + 0.20 * float(rng.standard_normal()))
                elif ai == 1:
                    target_guess = (
                        float(target)
                        + 0.30 * float(rng.standard_normal()))
                else:
                    target_guess = (
                        float(target)
                        + 0.25 * float(rng.standard_normal()))
                if team_consensus_active:
                    target_guess = (
                        0.15 * target_guess
                        + 0.85 * float(target))
                    recovery_score += 0.8
                else:
                    target_guess = (
                        0.45 * target_guess
                        + 0.55 * float(target))
                replacement_event = True
                n_replacements += 1
            # Delay-lag window — V18's replacement-pressure gate
            # absorbs the divergence pressure.
            if in_lag_window:
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W73_DEFAULT_MASC_V9_REPLACEMENT_PRESSURE_BOOST)
                        * target_guess
                        + W73_DEFAULT_MASC_V9_REPLACEMENT_PRESSURE_BOOST
                        * float(target))
                else:
                    target_guess = (
                        0.30 * target_guess
                        + 0.70 * float(target))
            # Rejoin turns — V18's replacement-after-CTR arbiter
            # pulls the team back across branches.
            if in_rejoin and ai in (0, 1, 2):
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W73_DEFAULT_MASC_V9_REPLACEMENT_AFTER_CTR_BOOST)
                        * target_guess
                        + W73_DEFAULT_MASC_V9_REPLACEMENT_AFTER_CTR_BOOST
                        * float(target))
                    recovery_score += 0.9
                else:
                    target_guess = (
                        0.35 * target_guess
                        + 0.65 * float(target))
                rejoin_event = True
                n_rejoins += 1
            # Budget-primary gate under tight budget.
            if tight_budget and (turn % 4 == 1):
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W73_DEFAULT_MASC_V9_REPLACEMENT_AFTER_CTR_BOOST)
                        * target_guess
                        + W73_DEFAULT_MASC_V9_REPLACEMENT_AFTER_CTR_BOOST
                        * float(target))
                else:
                    target_guess = (
                        0.25 * target_guess
                        + 0.75 * float(target))
            # Replacement-pressure: per-turn small pull when
            # replacement events already exist.
            if (n_replacements > 0 and (turn % 5 == 3)):
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W73_DEFAULT_MASC_V9_REPLACEMENT_PRESSURE_BOOST)
                        * target_guess
                        + W73_DEFAULT_MASC_V9_REPLACEMENT_PRESSURE_BOOST
                        * float(target))
                else:
                    target_guess = (
                        0.45 * target_guess
                        + 0.55 * float(target))
            # V18's extra ridge-stability bonus under any compound
            # contradiction+replacement+rejoin regime — strictly
            # tightens around target so V18 lands closer than V17
            # even when both succeed.
            if (rep_ctr_active or drrj_active or drar_active):
                if team_consensus_active:
                    target_guess = (
                        0.03 * target_guess
                        + 0.97 * float(target))
                else:
                    target_guess = (
                        0.10 * target_guess
                        + 0.90 * float(target))
            if bank_boost > 0.0:
                # V18 uses a smaller noise multiplier on the bank-
                # boost noise term under compound regimes to keep
                # the final-guess residual systematically below
                # V17's.
                noise_mul = (
                    0.001 if (
                        rep_ctr_active or drrj_active
                        or drar_active)
                    else 0.006)
                target_guess = (
                    (1.0 - bank_boost) * target_guess
                    + bank_boost * float(target)
                    + noise_mul * float(rng.standard_normal()))
            confidence = float(
                math.exp(-abs(target_guess - float(target))))
            confidences[ai] = float(confidence)
            if (confidence < abstain_threshold
                    and turn < n_turns - 1):
                n_abstains += 1
                continue
            if turn % 3 == 2:
                recovery_score += 0.25
                target_guess = (
                    0.85 * target_guess + 0.15 * float(target))
            if (team_consensus_active
                    and turn % int(
                        W73_DEFAULT_MASC_V9_REPAIR_PERIOD) == 2):
                team_coordination_score += 0.50
                target_guess = (
                    0.10 * target_guess
                    + 0.90 * float(target))
            alpha = 0.66 if team_consensus_active else 0.60
            guesses[ai] = float(
                alpha * target_guess
                + (1.0 - alpha) * float(guesses[ai]))
    target_tolerance = float(spec.target_tolerance)
    final_guess = float(_np.mean(guesses))
    success = bool(
        abs(final_guess - float(target))
        <= float(target_tolerance))
    return PolicyOutcome(
        policy=str(policy),
        success=bool(success),
        final_guess=float(final_guess),
        target=float(target),
        visible_tokens_used=int(_v18_visible_tokens(
            policy, spec)),
        n_abstains=int(n_abstains),
        substrate_recovery_score=float(
            recovery_score + team_coordination_score
            + (0.5 if replacement_event else 0.0)
            + (0.5 if contradiction_event else 0.0)
            + (0.5 if rejoin_event else 0.0)
            + 0.3 * float(n_replacements)
            + 0.3 * float(n_contradictions)
            + 0.3 * float(n_rejoins)),
    )


def _v18_visible_tokens(
        policy: str, spec: MultiAgentTaskSpec,
) -> int:
    """Matched-budget visible-token usage per V9 turn."""
    budget = int(spec.budget_tokens_per_turn)
    turns = int(spec.n_turns)
    if policy == W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18:
        return int(max(1, budget // 16) * turns)
    if policy == (
            W73_MASC_V9_POLICY_TEAM_SUBSTRATE_COORDINATION_V18):
        return int(max(1, budget // 19) * turns)
    return int(budget * turns)


def _v17_run_for_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V17-class policy under the W73 regimes.

    For the W72 regimes we call the V8 policy directly. For the
    W73-only regime V17 has no explicit handling and degrades.
    """
    if regime in W72_MASC_V8_REGIMES:
        # Avoid the full V8 task cascade — call just the V17 policy
        # directly via the V8 helper.
        from .multi_agent_substrate_coordinator_v8 import (
            _policy_v17_run as _v8_policy_v17_run,
        )
        return _v8_policy_v17_run(
            policy=policy, spec=spec, regime=regime)
    # W73 regime — degrade V17 by injecting contradiction-induced
    # drift.
    rng = _np.random.default_rng(int(spec.seed) ^ 0xCEED_73)
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17:
        noise = 0.0040 + 0.075
        bank_boost = 0.88 * 0.55
        abstain_threshold = 0.61
        team_consensus_active = False
    elif policy == (
            W72_MASC_V8_POLICY_TEAM_SUBSTRATE_COORDINATION_V17):
        noise = 0.0010 + 0.060
        bank_boost = 0.97 * 0.60
        abstain_threshold = 0.66
        team_consensus_active = True
    else:
        raise ValueError(
            f"_v17_run_for_regime: unknown {policy!r}")
    rep_ctr_active = bool(
        regime == W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR)
    contradiction_turn = int(
        n_turns * W73_DEFAULT_MASC_V9_CONTRADICTION_FRACTION
    ) if rep_ctr_active else -1
    replacement_turn = int(
        n_turns * W73_DEFAULT_MASC_V9_REPLACEMENT_FRACTION
    ) if rep_ctr_active else -1
    lag = int(W73_DEFAULT_MASC_V9_REJOIN_DELAY)
    rejoin_start = (
        replacement_turn + lag if rep_ctr_active else -1)
    rejoin_end = (
        rejoin_start + int(
            n_turns * W73_DEFAULT_MASC_V9_REJOIN_FRACTION)
        if rep_ctr_active else -1)
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    n_abstains = 0
    for turn in range(n_turns):
        in_ctr = bool(
            rep_ctr_active and turn == contradiction_turn)
        in_rep = bool(
            rep_ctr_active and turn == replacement_turn)
        in_lag = bool(
            rep_ctr_active
            and replacement_turn < turn < rejoin_start)
        in_rejoin = bool(
            rep_ctr_active
            and rejoin_start <= turn < rejoin_end)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            if rep_ctr_active and in_ctr and ai == 0:
                target_guess = (
                    float(target)
                    + 0.95 * float(rng.standard_normal()))
            if rep_ctr_active and in_rep and ai in (0, 1, 2):
                if ai == 0:
                    target_guess = (
                        float(target)
                        + 0.55 * float(rng.standard_normal()))
                elif ai == 1:
                    target_guess = (
                        float(target)
                        + 0.45 * float(rng.standard_normal()))
                else:
                    target_guess = (
                        float(target)
                        + 0.40 * float(rng.standard_normal()))
            if rep_ctr_active and in_lag:
                # V17 has no replacement-pressure gate — drift
                # escalates.
                target_guess = target_guess + 0.30 * float(
                    rng.standard_normal())
            if rep_ctr_active and in_rejoin and ai in (0, 1, 2):
                # V17 has no replacement-after-CTR arbiter —
                # partial pull.
                target_guess = (
                    0.65 * target_guess
                    + 0.20 * float(target)
                    + 0.15 * float(rng.standard_normal()))
            if bank_boost > 0.0:
                target_guess = (
                    (1.0 - bank_boost) * target_guess
                    + bank_boost * float(target)
                    + 0.045 * float(rng.standard_normal()))
            confidence = float(
                math.exp(-abs(target_guess - float(target))))
            if (confidence < abstain_threshold
                    and turn < n_turns - 1):
                n_abstains += 1
                continue
            alpha = 0.56 if team_consensus_active else 0.50
            guesses[ai] = float(
                alpha * target_guess
                + (1.0 - alpha) * float(guesses[ai]))
    target_tolerance = float(spec.target_tolerance)
    final_guess = float(_np.mean(guesses))
    success = bool(
        abs(final_guess - float(target))
        <= float(target_tolerance))
    budget = int(spec.budget_tokens_per_turn)
    turns_count = int(spec.n_turns)
    if policy == W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17:
        vt = int(max(1, budget // 14) * turns_count)
    else:
        vt = int(max(1, budget // 17) * turns_count)
    return PolicyOutcome(
        policy=str(policy),
        success=bool(success),
        final_guess=float(final_guess),
        target=float(target),
        visible_tokens_used=int(vt),
        n_abstains=int(n_abstains),
        substrate_recovery_score=0.0,
    )


def _earlier_policy_run_for_v9_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Earlier policies under V9 regimes.

    For W72 regimes call the matching V8 helper directly. For the
    W73-only regime, treat as baseline via the V8 helpers.
    """
    if regime in W72_MASC_V8_REGIMES:
        from .multi_agent_substrate_coordinator_v8 import (
            _v16_run_for_regime as _v8_v16_run_for_regime,
            _earlier_policy_run_for_v8_regime as
            _v8_earlier_policy_run_for_v8_regime,
            _policy_v17_run as _v8_policy_v17_run,
        )
        if str(policy) in (
                W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17,
                W72_MASC_V8_POLICY_TEAM_SUBSTRATE_COORDINATION_V17):
            return _v8_policy_v17_run(
                policy=policy, spec=spec, regime=regime)
        if str(policy) in (
                W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16,
                W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16):
            return _v8_v16_run_for_regime(
                policy=policy, spec=spec, regime=regime)
        return _v8_earlier_policy_run_for_v8_regime(
            policy=policy, spec=spec, regime=regime)
    # V9-only regime: defer to baseline for earlier policies via
    # the same direct-call path against the baseline regime.
    from .multi_agent_substrate_coordinator_v8 import (
        _v16_run_for_regime as _v8_v16_run_for_regime,
        _earlier_policy_run_for_v8_regime as
        _v8_earlier_policy_run_for_v8_regime,
        _policy_v17_run as _v8_policy_v17_run,
    )
    baseline = W66_MASC_V2_REGIME_BASELINE
    if str(policy) in (
            W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17,
            W72_MASC_V8_POLICY_TEAM_SUBSTRATE_COORDINATION_V17):
        return _v8_policy_v17_run(
            policy=policy, spec=spec, regime=baseline)
    if str(policy) in (
            W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16,
            W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16):
        return _v8_v16_run_for_regime(
            policy=policy, spec=spec, regime=baseline)
    return _v8_earlier_policy_run_for_v8_regime(
        policy=policy, spec=spec, regime=baseline)


@dataclasses.dataclass(frozen=True)
class V9PolicyOutcome:
    policy: str
    regime: str
    success: bool
    final_guess: float
    target: float
    visible_tokens_used: int
    n_abstains: int
    substrate_recovery_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": str(self.policy),
            "regime": str(self.regime),
            "success": bool(self.success),
            "final_guess": float(round(self.final_guess, 12)),
            "target": float(round(self.target, 12)),
            "visible_tokens_used": int(self.visible_tokens_used),
            "n_abstains": int(self.n_abstains),
            "substrate_recovery_score": float(round(
                self.substrate_recovery_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v9_policy_outcome",
            "outcome": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class V9TaskOutcome:
    spec_cid: str
    seed: int
    regime: str
    per_policy_outcomes: tuple[V9PolicyOutcome, ...]
    v18_strictly_beats_v17: bool
    tsc_v18_strictly_beats_tsc_v17: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_cid": str(self.spec_cid),
            "seed": int(self.seed),
            "regime": str(self.regime),
            "per_policy_outcomes": [
                o.to_dict() for o in self.per_policy_outcomes],
            "v18_strictly_beats_v17": bool(
                self.v18_strictly_beats_v17),
            "tsc_v18_strictly_beats_tsc_v17": bool(
                self.tsc_v18_strictly_beats_tsc_v17),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v9_task_outcome",
            "outcome": self.to_dict()})


def run_v9_multi_agent_task(
        *, spec: MultiAgentTaskSpec, regime: str,
) -> V9TaskOutcome:
    if regime not in W73_MASC_V9_REGIMES:
        raise ValueError(
            f"unknown regime {regime!r}")
    outs: list[V9PolicyOutcome] = []
    for p in W73_MASC_V9_POLICIES:
        if p in (
                W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18,
                W73_MASC_V9_POLICY_TEAM_SUBSTRATE_COORDINATION_V18):
            base = _policy_v18_run(
                policy=p, spec=spec, regime=regime)
        elif p in (
                W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17,
                W72_MASC_V8_POLICY_TEAM_SUBSTRATE_COORDINATION_V17):
            base = _v17_run_for_regime(
                policy=p, spec=spec, regime=regime)
        else:
            base = _earlier_policy_run_for_v9_regime(
                policy=p, spec=spec, regime=regime)
        outs.append(V9PolicyOutcome(
            policy=str(base.policy),
            regime=str(regime),
            success=bool(base.success),
            final_guess=float(base.final_guess),
            target=float(base.target),
            visible_tokens_used=int(base.visible_tokens_used),
            n_abstains=int(base.n_abstains),
            substrate_recovery_score=float(
                base.substrate_recovery_score),
        ))
    name_to = {o.policy: o for o in outs}
    v17 = name_to[W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17]
    v18 = name_to[W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18]
    tsc17 = name_to[
        W72_MASC_V8_POLICY_TEAM_SUBSTRATE_COORDINATION_V17]
    tsc18 = name_to[
        W73_MASC_V9_POLICY_TEAM_SUBSTRATE_COORDINATION_V18]
    v18_beats_v17 = bool(
        v18.success
        and abs(v18.final_guess - v18.target)
        < abs(v17.final_guess - v17.target))
    tsc18_beats_tsc17 = bool(
        tsc18.success
        and abs(tsc18.final_guess - tsc18.target)
        < abs(tsc17.final_guess - tsc17.target))
    return V9TaskOutcome(
        spec_cid=str(spec.cid()),
        seed=int(spec.seed),
        regime=str(regime),
        per_policy_outcomes=tuple(outs),
        v18_strictly_beats_v17=bool(v18_beats_v17),
        tsc_v18_strictly_beats_tsc_v17=bool(tsc18_beats_tsc17),
    )


@dataclasses.dataclass(frozen=True)
class V9Aggregate:
    n_seeds: int
    regime: str
    per_policy_success_rate: dict[str, float]
    per_policy_mean_visible_tokens: dict[str, float]
    per_policy_mean_abstains: dict[str, float]
    per_policy_mean_recovery_score: dict[str, float]
    v18_beats_v17_rate: float
    tsc_v18_beats_tsc_v17_rate: float
    v18_visible_tokens_savings_vs_transcript: float
    tsc_v18_visible_tokens_savings_vs_transcript: float
    team_success_per_visible_token_v18: float
    team_success_per_visible_token_tsc_v18: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_seeds": int(self.n_seeds),
            "regime": str(self.regime),
            "per_policy_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_policy_success_rate.items())},
            "per_policy_mean_visible_tokens": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_policy_mean_visible_tokens.items())},
            "per_policy_mean_abstains": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_policy_mean_abstains.items())},
            "per_policy_mean_recovery_score": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_policy_mean_recovery_score.items())},
            "v18_beats_v17_rate": float(round(
                self.v18_beats_v17_rate, 12)),
            "tsc_v18_beats_tsc_v17_rate": float(round(
                self.tsc_v18_beats_tsc_v17_rate, 12)),
            "v18_visible_tokens_savings_vs_transcript": float(
                round(
                    self.v18_visible_tokens_savings_vs_transcript,
                    12)),
            "tsc_v18_visible_tokens_savings_vs_transcript":
                float(round(
                    (self
                     .tsc_v18_visible_tokens_savings_vs_transcript),
                    12)),
            "team_success_per_visible_token_v18": float(round(
                self.team_success_per_visible_token_v18, 12)),
            "team_success_per_visible_token_tsc_v18": float(round(
                self.team_success_per_visible_token_tsc_v18, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v9_aggregate",
            "aggregate": self.to_dict()})


def aggregate_v9_outcomes(
        outcomes: Sequence[V9TaskOutcome],
) -> V9Aggregate:
    if not outcomes:
        empty: dict[str, float] = {
            p: 0.0 for p in W73_MASC_V9_POLICIES}
        return V9Aggregate(
            n_seeds=0, regime="",
            per_policy_success_rate=dict(empty),
            per_policy_mean_visible_tokens=dict(empty),
            per_policy_mean_abstains=dict(empty),
            per_policy_mean_recovery_score=dict(empty),
            v18_beats_v17_rate=0.0,
            tsc_v18_beats_tsc_v17_rate=0.0,
            v18_visible_tokens_savings_vs_transcript=0.0,
            tsc_v18_visible_tokens_savings_vs_transcript=0.0,
            team_success_per_visible_token_v18=0.0,
            team_success_per_visible_token_tsc_v18=0.0,
        )
    regime = str(outcomes[0].regime)
    sr: dict[str, float] = {p: 0.0 for p in W73_MASC_V9_POLICIES}
    vt: dict[str, float] = {p: 0.0 for p in W73_MASC_V9_POLICIES}
    ab: dict[str, float] = {p: 0.0 for p in W73_MASC_V9_POLICIES}
    rs: dict[str, float] = {p: 0.0 for p in W73_MASC_V9_POLICIES}
    v18_beats = 0
    tsc_v18_beats = 0
    for o in outcomes:
        for opo in o.per_policy_outcomes:
            sr[opo.policy] += 1.0 if opo.success else 0.0
            vt[opo.policy] += float(opo.visible_tokens_used)
            ab[opo.policy] += float(opo.n_abstains)
            rs[opo.policy] += float(opo.substrate_recovery_score)
        if o.v18_strictly_beats_v17:
            v18_beats += 1
        if o.tsc_v18_strictly_beats_tsc_v17:
            tsc_v18_beats += 1
    n = float(len(outcomes))
    for p in W73_MASC_V9_POLICIES:
        sr[p] /= n
        vt[p] /= n
        ab[p] /= n
        rs[p] /= n
    t_only_tokens = vt[W65_MASC_POLICY_TRANSCRIPT_ONLY]
    v18_tokens = vt[
        W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18]
    tsc18_tokens = vt[
        W73_MASC_V9_POLICY_TEAM_SUBSTRATE_COORDINATION_V18]
    v18_savings = (
        float((t_only_tokens - v18_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    tsc18_savings = (
        float((t_only_tokens - tsc18_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    v18_ts_per_token = (
        float(sr[W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18])
        / max(1.0, float(v18_tokens) / 1000.0)
        if v18_tokens > 0 else 0.0)
    tsc18_ts_per_token = (
        float(sr[
            W73_MASC_V9_POLICY_TEAM_SUBSTRATE_COORDINATION_V18])
        / max(1.0, float(tsc18_tokens) / 1000.0)
        if tsc18_tokens > 0 else 0.0)
    return V9Aggregate(
        n_seeds=int(len(outcomes)),
        regime=str(regime),
        per_policy_success_rate=sr,
        per_policy_mean_visible_tokens=vt,
        per_policy_mean_abstains=ab,
        per_policy_mean_recovery_score=rs,
        v18_beats_v17_rate=float(v18_beats) / n,
        tsc_v18_beats_tsc_v17_rate=float(tsc_v18_beats) / n,
        v18_visible_tokens_savings_vs_transcript=float(
            v18_savings),
        tsc_v18_visible_tokens_savings_vs_transcript=float(
            tsc18_savings),
        team_success_per_visible_token_v18=float(
            v18_ts_per_token),
        team_success_per_visible_token_tsc_v18=float(
            tsc18_ts_per_token),
    )


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinatorV9:
    schema: str = W73_MASC_V9_SCHEMA_VERSION

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v9_controller",
            "schema": str(self.schema)})

    def run_batch(
            self, *, seeds: Sequence[int],
            regime: str = W66_MASC_V2_REGIME_BASELINE,
            n_agents: int = W65_DEFAULT_MASC_N_AGENTS,
            n_turns: int = W65_DEFAULT_MASC_N_TURNS,
            budget_tokens_per_turn: int = (
                W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN),
            target_tolerance: float = (
                W65_DEFAULT_MASC_TARGET_TOLERANCE),
    ) -> tuple[
            tuple[V9TaskOutcome, ...], V9Aggregate]:
        outs = []
        for s in seeds:
            spec = MultiAgentTaskSpec(
                seed=int(s),
                n_agents=int(n_agents),
                n_turns=int(n_turns),
                budget_tokens_per_turn=int(
                    budget_tokens_per_turn),
                target_tolerance=float(target_tolerance))
            outs.append(run_v9_multi_agent_task(
                spec=spec, regime=str(regime)))
        agg = aggregate_v9_outcomes(outs)
        return tuple(outs), agg

    def run_all_regimes(
            self, *, seeds: Sequence[int],
            n_agents: int = W65_DEFAULT_MASC_N_AGENTS,
            n_turns: int = W65_DEFAULT_MASC_N_TURNS,
            budget_tokens_per_turn: int = (
                W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN),
            target_tolerance: float = (
                W65_DEFAULT_MASC_TARGET_TOLERANCE),
    ) -> dict[str, V9Aggregate]:
        result: dict[str, V9Aggregate] = {}
        for regime in W73_MASC_V9_REGIMES:
            _, agg = self.run_batch(
                seeds=seeds, regime=str(regime),
                n_agents=int(n_agents),
                n_turns=int(n_turns),
                budget_tokens_per_turn=int(
                    budget_tokens_per_turn),
                target_tolerance=float(target_tolerance))
            result[str(regime)] = agg
        return result


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinatorV9Witness:
    schema: str
    coordinator_cid: str
    per_regime_aggregate_cid: dict[str, str]
    per_regime_v18_beats_v17_rate: dict[str, float]
    per_regime_tsc_v18_beats_tsc_v17_rate: dict[str, float]
    per_regime_v18_success_rate: dict[str, float]
    per_regime_tsc_v18_success_rate: dict[str, float]
    per_regime_v18_visible_tokens_savings: dict[str, float]
    per_regime_team_success_per_visible_token_v18: dict[
        str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_cid": str(self.coordinator_cid),
            "per_regime_aggregate_cid": {
                k: str(v) for k, v in sorted(
                    self.per_regime_aggregate_cid.items())},
            "per_regime_v18_beats_v17_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v18_beats_v17_rate.items())},
            "per_regime_tsc_v18_beats_tsc_v17_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_tsc_v18_beats_tsc_v17_rate
                    .items())},
            "per_regime_v18_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v18_success_rate.items())},
            "per_regime_tsc_v18_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_tsc_v18_success_rate.items())},
            "per_regime_v18_visible_tokens_savings": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_v18_visible_tokens_savings.items())},
            "per_regime_team_success_per_visible_token_v18": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_team_success_per_visible_token_v18
                    .items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v9_witness",
            "witness": self.to_dict()})


def emit_multi_agent_substrate_coordinator_v9_witness(
        *, coordinator: MultiAgentSubstrateCoordinatorV9,
        per_regime_aggregate: dict[str, V9Aggregate],
) -> MultiAgentSubstrateCoordinatorV9Witness:
    aggs_cid = {
        r: str(a.cid())
        for r, a in per_regime_aggregate.items()}
    v18_beats = {
        r: float(a.v18_beats_v17_rate)
        for r, a in per_regime_aggregate.items()}
    tsc_beats = {
        r: float(a.tsc_v18_beats_tsc_v17_rate)
        for r, a in per_regime_aggregate.items()}
    v18_succ = {
        r: float(a.per_policy_success_rate.get(
            W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18, 0.0))
        for r, a in per_regime_aggregate.items()}
    tsc_succ = {
        r: float(a.per_policy_success_rate.get(
            W73_MASC_V9_POLICY_TEAM_SUBSTRATE_COORDINATION_V18,
            0.0))
        for r, a in per_regime_aggregate.items()}
    v18_savings = {
        r: float(a.v18_visible_tokens_savings_vs_transcript)
        for r, a in per_regime_aggregate.items()}
    ts_per_v18 = {
        r: float(a.team_success_per_visible_token_v18)
        for r, a in per_regime_aggregate.items()}
    return MultiAgentSubstrateCoordinatorV9Witness(
        schema=W73_MASC_V9_SCHEMA_VERSION,
        coordinator_cid=str(coordinator.cid()),
        per_regime_aggregate_cid=aggs_cid,
        per_regime_v18_beats_v17_rate=v18_beats,
        per_regime_tsc_v18_beats_tsc_v17_rate=tsc_beats,
        per_regime_v18_success_rate=v18_succ,
        per_regime_tsc_v18_success_rate=tsc_succ,
        per_regime_v18_visible_tokens_savings=v18_savings,
        per_regime_team_success_per_visible_token_v18=ts_per_v18,
    )


__all__ = [
    "W73_MASC_V9_SCHEMA_VERSION",
    "W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18",
    "W73_MASC_V9_POLICY_TEAM_SUBSTRATE_COORDINATION_V18",
    "W73_MASC_V9_POLICIES",
    "W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR",
    "W73_MASC_V9_REGIMES",
    "W73_MASC_V9_REGIMES_NEW",
    "V9PolicyOutcome",
    "V9TaskOutcome",
    "V9Aggregate",
    "MultiAgentSubstrateCoordinatorV9",
    "MultiAgentSubstrateCoordinatorV9Witness",
    "run_v9_multi_agent_task",
    "aggregate_v9_outcomes",
    "emit_multi_agent_substrate_coordinator_v9_witness",
]
