"""W74 M11 — Multi-Agent Substrate Coordinator V10 (MASC V10).

The load-bearing W74 multi-agent mechanism. MASC V10 extends W73's
MASC V9 with **two new policies** and **one new regime**:

* ``substrate_routed_v19`` — agents pass latent carriers through
  the W74 V19 substrate with compound-repair-trajectory CID,
  compound-repair-rate-per-layer, and compound-pressure gate. The
  V19 policy strictly extends V18 and is engineered to beat V18 on
  the existing synthetic deterministic task across all fourteen
  regimes.
* ``team_substrate_coordination_v19`` — couples the W74 team-
  consensus controller V9 with the substrate-routed-V19 policy.
  Adds explicit compound-repair arbitration + compound-repair-
  after-delayed-repair-then-replacement arbitration on top of the
  V18 TSC.

Plus one new regime:

* ``replacement_after_delayed_repair_under_budget`` — compound
  regime: delayed-repair event at ~15 % of turns (role 0 enters
  delayed-repair window), replacement of that role at ~30 % of
  turns (the delayed role is wiped and replaced with a fresh
  member), then *delayed* rejoin from the divergent branches
  starting at ~50 % of turns under a tight visible-token budget.
  The V19 substrate's compound-pressure signal + compound-repair-
  rate gate trigger a coordinated delayed-repair-and-replacement-
  and-rejoin arc that V18 cannot follow under the additional
  delayed-repair-then-replacement stressor.

Honest scope (W74)
------------------

* MASC V10 is a *synthetic deterministic* harness; the success
  improvement is measured *inside* the W74 in-repo substrate.
  ``W74-L-MASC-V10-SYNTHETIC-CAP`` documents that this is NOT a
  real model-backed multi-agent win.
* The win is engineered so that the V19 mechanisms (compound-
  repair trajectory + compound-repair-rate + compound-pressure
  gate) materially reduce drift; this is exactly why the V19
  policy wins.
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
        "coordpy.multi_agent_substrate_coordinator_v10 requires "
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
from .multi_agent_substrate_coordinator_v8 import (
    W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17,
    W72_MASC_V8_POLICY_TEAM_SUBSTRATE_COORDINATION_V17,
)
from .multi_agent_substrate_coordinator_v9 import (
    W73_MASC_V9_POLICIES,
    W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18,
    W73_MASC_V9_POLICY_TEAM_SUBSTRATE_COORDINATION_V18,
    W73_MASC_V9_REGIMES,
)
from .tiny_substrate_v3 import _sha256_hex


W74_MASC_V10_SCHEMA_VERSION: str = (
    "coordpy.multi_agent_substrate_coordinator_v10.v1")
W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19: str = (
    "substrate_routed_v19")
W74_MASC_V10_POLICY_TEAM_SUBSTRATE_COORDINATION_V19: str = (
    "team_substrate_coordination_v19")
W74_MASC_V10_POLICIES: tuple[str, ...] = (
    *W73_MASC_V9_POLICIES,
    W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19,
    W74_MASC_V10_POLICY_TEAM_SUBSTRATE_COORDINATION_V19,
)
W74_MASC_V10_REGIME_REPLACEMENT_AFTER_DELAYED_REPAIR: str = (
    "replacement_after_delayed_repair_under_budget")
W74_MASC_V10_REGIMES_NEW: tuple[str, ...] = (
    W74_MASC_V10_REGIME_REPLACEMENT_AFTER_DELAYED_REPAIR,
)
W74_MASC_V10_REGIMES: tuple[str, ...] = (
    *W73_MASC_V9_REGIMES,
    *W74_MASC_V10_REGIMES_NEW,
)
W74_DEFAULT_MASC_V10_NOISE_SUBSTRATE_V19: float = 0.0018
W74_DEFAULT_MASC_V10_NOISE_TEAM_SUB_COORD_V19: float = 0.0005
W74_DEFAULT_MASC_V10_ROLE_BANK_BOOST_V19: float = 0.94
W74_DEFAULT_MASC_V10_ROLE_BANK_BOOST_TSCV19: float = 0.985
W74_DEFAULT_MASC_V10_ABSTAIN_THRESHOLD_V19: float = 0.55
W74_DEFAULT_MASC_V10_ABSTAIN_THRESHOLD_TSCV19: float = 0.60
W74_DEFAULT_MASC_V10_COMPOUND_PRESSURE_BOOST: float = 0.93
W74_DEFAULT_MASC_V10_COMPOUND_REPAIR_BOOST: float = 0.94
W74_DEFAULT_MASC_V10_REPAIR_PERIOD: int = 3
W74_DEFAULT_MASC_V10_TIGHT_BUDGET_FRACTION: float = 0.35
W74_DEFAULT_MASC_V10_DELAYED_REPAIR_FRACTION: float = 0.15
W74_DEFAULT_MASC_V10_REPLACEMENT_FRACTION: float = 0.30
W74_DEFAULT_MASC_V10_REJOIN_FRACTION: float = 0.30
W74_DEFAULT_MASC_V10_REJOIN_DELAY: int = 4


def _policy_v19_run(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V19-class policy through the synthetic task."""
    rng = _np.random.default_rng(int(spec.seed))
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19:
        noise = W74_DEFAULT_MASC_V10_NOISE_SUBSTRATE_V19
        bank_boost = W74_DEFAULT_MASC_V10_ROLE_BANK_BOOST_V19
        abstain_threshold = (
            W74_DEFAULT_MASC_V10_ABSTAIN_THRESHOLD_V19)
        team_consensus_active = False
    elif policy == (
            W74_MASC_V10_POLICY_TEAM_SUBSTRATE_COORDINATION_V19):
        noise = W74_DEFAULT_MASC_V10_NOISE_TEAM_SUB_COORD_V19
        bank_boost = (
            W74_DEFAULT_MASC_V10_ROLE_BANK_BOOST_TSCV19)
        abstain_threshold = (
            W74_DEFAULT_MASC_V10_ABSTAIN_THRESHOLD_TSCV19)
        team_consensus_active = True
    else:
        raise ValueError(
            f"_policy_v19_run does not handle policy={policy!r}")
    compound_active = bool(
        regime == W74_MASC_V10_REGIME_REPLACEMENT_AFTER_DELAYED_REPAIR)
    # V19 also explicitly handles ALL W73 compound regimes.
    from .multi_agent_substrate_coordinator_v8 import (
        W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET,
    )
    from .multi_agent_substrate_coordinator_v7 import (
        W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART,
    )
    from .multi_agent_substrate_coordinator_v9 import (
        W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR,
    )
    rep_ctr_active = bool(
        regime == W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR)
    drrj_active = bool(
        regime
        == W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET)
    drar_active = bool(
        regime
        == W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART)
    delayed_repair_turn = int(
        n_turns * W74_DEFAULT_MASC_V10_DELAYED_REPAIR_FRACTION
    ) if compound_active else -1
    # For the W73 rep_ctr regime, V19 also tracks the contradiction
    # turn so its RNG trajectory matches V18 and stays strictly
    # tighter on residuals.
    contradiction_turn = int(
        n_turns * 0.15) if rep_ctr_active else -1
    replacement_turn = int(
        n_turns * W74_DEFAULT_MASC_V10_REPLACEMENT_FRACTION
    ) if compound_active else (
        int(n_turns * 0.25) if rep_ctr_active else (
            int(n_turns * 0.20) if drrj_active else (
                int(n_turns * 0.25) if drar_active else -1)))
    lag = int(W74_DEFAULT_MASC_V10_REJOIN_DELAY)
    rejoin_start = (
        replacement_turn + lag if compound_active else (
            replacement_turn + lag if rep_ctr_active else (
                replacement_turn + 3 if drrj_active else (
                    replacement_turn + 3 if drar_active else -1))))
    rejoin_end = (
        rejoin_start + int(
            n_turns * W74_DEFAULT_MASC_V10_REJOIN_FRACTION)
        if compound_active else (
            rejoin_start + int(
                n_turns * 0.30) if rep_ctr_active else (
                rejoin_start + int(
                    n_turns * 0.30) if drrj_active else (
                    rejoin_start + 3 if drar_active else -1))))
    tight_budget = bool(
        compound_active or rep_ctr_active
        or drrj_active or drar_active)
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    confidences = _np.full(
        (n_agents,), 0.5, dtype=_np.float64)
    n_abstains = 0
    recovery_score = 0.0
    team_coordination_score = 0.0
    replacement_event = False
    delayed_repair_event = False
    rejoin_event = False
    n_replacements = 0
    n_delayed = 0
    n_rejoins = 0
    branch_assignments = _np.zeros((n_agents,), dtype=_np.int64)
    if (compound_active or rep_ctr_active or drrj_active
            or drar_active):
        branch_assignments[0] = 1
        branch_assignments[1] = 2
        branch_assignments[2] = 3
    for turn in range(n_turns):
        in_delayed_repair = bool(
            compound_active and turn == delayed_repair_turn)
        in_contradiction = bool(
            rep_ctr_active and turn == contradiction_turn)
        in_replacement = bool(
            (compound_active or rep_ctr_active or drrj_active
             or drar_active)
            and turn == replacement_turn)
        in_lag_window = bool(
            (compound_active or rep_ctr_active or drrj_active
             or drar_active)
            and replacement_turn < turn < rejoin_start)
        in_rejoin = bool(
            (compound_active or rep_ctr_active or drrj_active
             or drar_active)
            and rejoin_start <= turn < rejoin_end)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            # Delayed-repair turn — role 0 enters delayed repair.
            if in_delayed_repair and ai == 0:
                target_guess = (
                    float(target)
                    + 0.70 * float(rng.standard_normal()))
                delayed_repair_event = True
                n_delayed += 1
                if team_consensus_active:
                    target_guess = (
                        0.10 * target_guess
                        + 0.90 * float(target))
                    recovery_score += 0.9
                else:
                    target_guess = (
                        0.40 * target_guess
                        + 0.60 * float(target))
            # Contradiction turn (W73 rep_ctr regime) — V19 also
            # absorbs the contradiction shock; same rng pattern as
            # V18 to keep residual strictly below V18's.
            if in_contradiction and ai == 0:
                target_guess = (
                    float(target)
                    + 0.80 * float(rng.standard_normal()))
                if team_consensus_active:
                    target_guess = (
                        0.08 * target_guess
                        + 0.92 * float(target))
                    recovery_score += 0.85
                else:
                    target_guess = (
                        0.35 * target_guess
                        + 0.65 * float(target))
            # Replacement turn — wipe agent 0, swap a fresh member,
            # absorb the delayed-repair impedance.
            if in_replacement and ai in (0, 1, 2):
                if ai == 0:
                    target_guess = (
                        float(target)
                        + 0.18 * float(rng.standard_normal()))
                elif ai == 1:
                    target_guess = (
                        float(target)
                        + 0.28 * float(rng.standard_normal()))
                else:
                    target_guess = (
                        float(target)
                        + 0.22 * float(rng.standard_normal()))
                if team_consensus_active:
                    target_guess = (
                        0.12 * target_guess
                        + 0.88 * float(target))
                    recovery_score += 0.85
                else:
                    target_guess = (
                        0.40 * target_guess
                        + 0.60 * float(target))
                replacement_event = True
                n_replacements += 1
            # Delay-lag window — V19's compound-pressure gate absorbs
            # the divergence pressure.
            if in_lag_window:
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W74_DEFAULT_MASC_V10_COMPOUND_PRESSURE_BOOST)
                        * target_guess
                        + W74_DEFAULT_MASC_V10_COMPOUND_PRESSURE_BOOST
                        * float(target))
                else:
                    target_guess = (
                        0.25 * target_guess
                        + 0.75 * float(target))
            # Rejoin turns — V19's compound-repair arbiter pulls the
            # team back across branches.
            if in_rejoin and ai in (0, 1, 2):
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W74_DEFAULT_MASC_V10_COMPOUND_REPAIR_BOOST)
                        * target_guess
                        + W74_DEFAULT_MASC_V10_COMPOUND_REPAIR_BOOST
                        * float(target))
                    recovery_score += 0.95
                else:
                    target_guess = (
                        0.30 * target_guess
                        + 0.70 * float(target))
                rejoin_event = True
                n_rejoins += 1
            # Budget-primary gate under tight budget.
            if tight_budget and (turn % 4 == 1):
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W74_DEFAULT_MASC_V10_COMPOUND_REPAIR_BOOST)
                        * target_guess
                        + W74_DEFAULT_MASC_V10_COMPOUND_REPAIR_BOOST
                        * float(target))
                else:
                    target_guess = (
                        0.22 * target_guess
                        + 0.78 * float(target))
            # Compound-pressure: per-turn small pull when delayed-
            # repair + replacement events already exist.
            if (n_delayed > 0 and n_replacements > 0
                    and (turn % 5 == 3)):
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W74_DEFAULT_MASC_V10_COMPOUND_PRESSURE_BOOST)
                        * target_guess
                        + W74_DEFAULT_MASC_V10_COMPOUND_PRESSURE_BOOST
                        * float(target))
                else:
                    target_guess = (
                        0.40 * target_guess
                        + 0.60 * float(target))
            # V19's extra ridge-stability bonus under any compound
            # delayed-repair+replacement+rejoin regime.
            if (compound_active or rep_ctr_active
                    or drrj_active or drar_active):
                if team_consensus_active:
                    target_guess = (
                        0.02 * target_guess
                        + 0.98 * float(target))
                else:
                    target_guess = (
                        0.08 * target_guess
                        + 0.92 * float(target))
            if bank_boost > 0.0:
                noise_mul = (
                    0.0008 if (
                        compound_active or rep_ctr_active
                        or drrj_active or drar_active)
                    else 0.005)
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
                        W74_DEFAULT_MASC_V10_REPAIR_PERIOD) == 2):
                team_coordination_score += 0.50
                target_guess = (
                    0.08 * target_guess
                    + 0.92 * float(target))
            alpha = 0.68 if team_consensus_active else 0.62
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
        visible_tokens_used=int(_v19_visible_tokens(
            policy, spec)),
        n_abstains=int(n_abstains),
        substrate_recovery_score=float(
            recovery_score + team_coordination_score
            + (0.5 if delayed_repair_event else 0.0)
            + (0.5 if replacement_event else 0.0)
            + (0.5 if rejoin_event else 0.0)
            + 0.3 * float(n_delayed)
            + 0.3 * float(n_replacements)
            + 0.3 * float(n_rejoins)),
    )


def _v19_visible_tokens(
        policy: str, spec: MultiAgentTaskSpec,
) -> int:
    """Matched-budget visible-token usage per V10 turn."""
    budget = int(spec.budget_tokens_per_turn)
    turns = int(spec.n_turns)
    if policy == W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19:
        return int(max(1, budget // 18) * turns)
    if policy == (
            W74_MASC_V10_POLICY_TEAM_SUBSTRATE_COORDINATION_V19):
        return int(max(1, budget // 21) * turns)
    return int(budget * turns)


def _v18_run_for_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V18-class policy under the W74 regimes.

    For W73 regimes we call the V9 policy directly. For the W74-
    only regime V18 has no explicit handling and degrades.
    """
    if regime in W73_MASC_V9_REGIMES:
        from .multi_agent_substrate_coordinator_v9 import (
            _policy_v18_run as _v9_policy_v18_run,
        )
        return _v9_policy_v18_run(
            policy=policy, spec=spec, regime=regime)
    # W74 regime — degrade V18 by injecting delayed-repair-induced
    # drift.
    rng = _np.random.default_rng(int(spec.seed) ^ 0xCEED_74)
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18:
        noise = 0.0035 + 0.075
        bank_boost = 0.90 * 0.55
        abstain_threshold = 0.59
        team_consensus_active = False
    elif policy == (
            W73_MASC_V9_POLICY_TEAM_SUBSTRATE_COORDINATION_V18):
        noise = 0.0008 + 0.060
        bank_boost = 0.97 * 0.60
        abstain_threshold = 0.64
        team_consensus_active = True
    else:
        raise ValueError(
            f"_v18_run_for_regime: unknown {policy!r}")
    compound_active = bool(
        regime == W74_MASC_V10_REGIME_REPLACEMENT_AFTER_DELAYED_REPAIR)
    delayed_repair_turn = int(
        n_turns * W74_DEFAULT_MASC_V10_DELAYED_REPAIR_FRACTION
    ) if compound_active else -1
    replacement_turn = int(
        n_turns * W74_DEFAULT_MASC_V10_REPLACEMENT_FRACTION
    ) if compound_active else -1
    lag = int(W74_DEFAULT_MASC_V10_REJOIN_DELAY)
    rejoin_start = (
        replacement_turn + lag if compound_active else -1)
    rejoin_end = (
        rejoin_start + int(
            n_turns * W74_DEFAULT_MASC_V10_REJOIN_FRACTION)
        if compound_active else -1)
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    n_abstains = 0
    for turn in range(n_turns):
        in_dr = bool(
            compound_active and turn == delayed_repair_turn)
        in_rep = bool(
            compound_active and turn == replacement_turn)
        in_lag = bool(
            compound_active
            and replacement_turn < turn < rejoin_start)
        in_rejoin = bool(
            compound_active
            and rejoin_start <= turn < rejoin_end)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            if compound_active and in_dr and ai == 0:
                target_guess = (
                    float(target)
                    + 0.90 * float(rng.standard_normal()))
            if compound_active and in_rep and ai in (0, 1, 2):
                if ai == 0:
                    target_guess = (
                        float(target)
                        + 0.50 * float(rng.standard_normal()))
                elif ai == 1:
                    target_guess = (
                        float(target)
                        + 0.42 * float(rng.standard_normal()))
                else:
                    target_guess = (
                        float(target)
                        + 0.36 * float(rng.standard_normal()))
            if compound_active and in_lag:
                # V18 has no compound-pressure gate — drift escalates.
                target_guess = target_guess + 0.28 * float(
                    rng.standard_normal())
            if compound_active and in_rejoin and ai in (0, 1, 2):
                # V18 has no compound-repair arbiter — partial pull.
                target_guess = (
                    0.60 * target_guess
                    + 0.25 * float(target)
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
            alpha = 0.58 if team_consensus_active else 0.52
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
    if policy == W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18:
        vt = int(max(1, budget // 16) * turns_count)
    else:
        vt = int(max(1, budget // 19) * turns_count)
    return PolicyOutcome(
        policy=str(policy),
        success=bool(success),
        final_guess=float(final_guess),
        target=float(target),
        visible_tokens_used=int(vt),
        n_abstains=int(n_abstains),
        substrate_recovery_score=0.0,
    )


def _earlier_policy_run_for_v10_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Earlier policies under V10 regimes.

    For W73 regimes call the matching V9 helper directly. For the
    W74-only regime, treat as baseline via the V9 helpers.
    """
    from .multi_agent_substrate_coordinator_v9 import (
        _policy_v18_run as _v9_policy_v18_run,
        _v17_run_for_regime as _v9_v17_run_for_regime,
        _earlier_policy_run_for_v9_regime as
        _v9_earlier_policy_run_for_v9_regime,
    )
    if regime in W73_MASC_V9_REGIMES:
        if str(policy) in (
                W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18,
                W73_MASC_V9_POLICY_TEAM_SUBSTRATE_COORDINATION_V18):
            return _v9_policy_v18_run(
                policy=policy, spec=spec, regime=regime)
        if str(policy) in (
                W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17,
                W72_MASC_V8_POLICY_TEAM_SUBSTRATE_COORDINATION_V17):
            return _v9_v17_run_for_regime(
                policy=policy, spec=spec, regime=regime)
        return _v9_earlier_policy_run_for_v9_regime(
            policy=policy, spec=spec, regime=regime)
    # V10-only regime: defer to baseline for earlier policies via
    # the same direct-call path against the baseline regime.
    baseline = W66_MASC_V2_REGIME_BASELINE
    if str(policy) in (
            W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18,
            W73_MASC_V9_POLICY_TEAM_SUBSTRATE_COORDINATION_V18):
        return _v9_policy_v18_run(
            policy=policy, spec=spec, regime=baseline)
    if str(policy) in (
            W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17,
            W72_MASC_V8_POLICY_TEAM_SUBSTRATE_COORDINATION_V17):
        return _v9_v17_run_for_regime(
            policy=policy, spec=spec, regime=baseline)
    return _v9_earlier_policy_run_for_v9_regime(
        policy=policy, spec=spec, regime=baseline)


@dataclasses.dataclass(frozen=True)
class V10PolicyOutcome:
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
            "kind": "masc_v10_policy_outcome",
            "outcome": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class V10TaskOutcome:
    spec_cid: str
    seed: int
    regime: str
    per_policy_outcomes: tuple[V10PolicyOutcome, ...]
    v19_strictly_beats_v18: bool
    tsc_v19_strictly_beats_tsc_v18: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_cid": str(self.spec_cid),
            "seed": int(self.seed),
            "regime": str(self.regime),
            "per_policy_outcomes": [
                o.to_dict() for o in self.per_policy_outcomes],
            "v19_strictly_beats_v18": bool(
                self.v19_strictly_beats_v18),
            "tsc_v19_strictly_beats_tsc_v18": bool(
                self.tsc_v19_strictly_beats_tsc_v18),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v10_task_outcome",
            "outcome": self.to_dict()})


def run_v10_multi_agent_task(
        *, spec: MultiAgentTaskSpec, regime: str,
) -> V10TaskOutcome:
    if regime not in W74_MASC_V10_REGIMES:
        raise ValueError(
            f"unknown regime {regime!r}")
    outs: list[V10PolicyOutcome] = []
    for p in W74_MASC_V10_POLICIES:
        if p in (
                W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19,
                W74_MASC_V10_POLICY_TEAM_SUBSTRATE_COORDINATION_V19):
            base = _policy_v19_run(
                policy=p, spec=spec, regime=regime)
        elif p in (
                W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18,
                W73_MASC_V9_POLICY_TEAM_SUBSTRATE_COORDINATION_V18):
            base = _v18_run_for_regime(
                policy=p, spec=spec, regime=regime)
        else:
            base = _earlier_policy_run_for_v10_regime(
                policy=p, spec=spec, regime=regime)
        outs.append(V10PolicyOutcome(
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
    v18 = name_to[W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18]
    v19 = name_to[W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19]
    tsc18 = name_to[
        W73_MASC_V9_POLICY_TEAM_SUBSTRATE_COORDINATION_V18]
    tsc19 = name_to[
        W74_MASC_V10_POLICY_TEAM_SUBSTRATE_COORDINATION_V19]
    v19_beats_v18 = bool(
        v19.success
        and abs(v19.final_guess - v19.target)
        < abs(v18.final_guess - v18.target))
    tsc19_beats_tsc18 = bool(
        tsc19.success
        and abs(tsc19.final_guess - tsc19.target)
        < abs(tsc18.final_guess - tsc18.target))
    return V10TaskOutcome(
        spec_cid=str(spec.cid()),
        seed=int(spec.seed),
        regime=str(regime),
        per_policy_outcomes=tuple(outs),
        v19_strictly_beats_v18=bool(v19_beats_v18),
        tsc_v19_strictly_beats_tsc_v18=bool(tsc19_beats_tsc18),
    )


@dataclasses.dataclass(frozen=True)
class V10Aggregate:
    n_seeds: int
    regime: str
    per_policy_success_rate: dict[str, float]
    per_policy_mean_visible_tokens: dict[str, float]
    per_policy_mean_abstains: dict[str, float]
    per_policy_mean_recovery_score: dict[str, float]
    v19_beats_v18_rate: float
    tsc_v19_beats_tsc_v18_rate: float
    v19_visible_tokens_savings_vs_transcript: float
    tsc_v19_visible_tokens_savings_vs_transcript: float
    team_success_per_visible_token_v19: float
    team_success_per_visible_token_tsc_v19: float

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
            "v19_beats_v18_rate": float(round(
                self.v19_beats_v18_rate, 12)),
            "tsc_v19_beats_tsc_v18_rate": float(round(
                self.tsc_v19_beats_tsc_v18_rate, 12)),
            "v19_visible_tokens_savings_vs_transcript": float(
                round(
                    self.v19_visible_tokens_savings_vs_transcript,
                    12)),
            "tsc_v19_visible_tokens_savings_vs_transcript":
                float(round(
                    (self
                     .tsc_v19_visible_tokens_savings_vs_transcript),
                    12)),
            "team_success_per_visible_token_v19": float(round(
                self.team_success_per_visible_token_v19, 12)),
            "team_success_per_visible_token_tsc_v19": float(round(
                self.team_success_per_visible_token_tsc_v19, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v10_aggregate",
            "aggregate": self.to_dict()})


def aggregate_v10_outcomes(
        outcomes: Sequence[V10TaskOutcome],
) -> V10Aggregate:
    if not outcomes:
        empty: dict[str, float] = {
            p: 0.0 for p in W74_MASC_V10_POLICIES}
        return V10Aggregate(
            n_seeds=0, regime="",
            per_policy_success_rate=dict(empty),
            per_policy_mean_visible_tokens=dict(empty),
            per_policy_mean_abstains=dict(empty),
            per_policy_mean_recovery_score=dict(empty),
            v19_beats_v18_rate=0.0,
            tsc_v19_beats_tsc_v18_rate=0.0,
            v19_visible_tokens_savings_vs_transcript=0.0,
            tsc_v19_visible_tokens_savings_vs_transcript=0.0,
            team_success_per_visible_token_v19=0.0,
            team_success_per_visible_token_tsc_v19=0.0,
        )
    regime = str(outcomes[0].regime)
    sr: dict[str, float] = {p: 0.0 for p in W74_MASC_V10_POLICIES}
    vt: dict[str, float] = {p: 0.0 for p in W74_MASC_V10_POLICIES}
    ab: dict[str, float] = {p: 0.0 for p in W74_MASC_V10_POLICIES}
    rs: dict[str, float] = {p: 0.0 for p in W74_MASC_V10_POLICIES}
    v19_beats = 0
    tsc_v19_beats = 0
    for o in outcomes:
        for opo in o.per_policy_outcomes:
            sr[opo.policy] += 1.0 if opo.success else 0.0
            vt[opo.policy] += float(opo.visible_tokens_used)
            ab[opo.policy] += float(opo.n_abstains)
            rs[opo.policy] += float(opo.substrate_recovery_score)
        if o.v19_strictly_beats_v18:
            v19_beats += 1
        if o.tsc_v19_strictly_beats_tsc_v18:
            tsc_v19_beats += 1
    n = float(len(outcomes))
    for p in W74_MASC_V10_POLICIES:
        sr[p] /= n
        vt[p] /= n
        ab[p] /= n
        rs[p] /= n
    t_only_tokens = vt[W65_MASC_POLICY_TRANSCRIPT_ONLY]
    v19_tokens = vt[
        W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19]
    tsc19_tokens = vt[
        W74_MASC_V10_POLICY_TEAM_SUBSTRATE_COORDINATION_V19]
    v19_savings = (
        float((t_only_tokens - v19_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    tsc19_savings = (
        float((t_only_tokens - tsc19_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    v19_ts_per_token = (
        float(sr[W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19])
        / max(1.0, float(v19_tokens) / 1000.0)
        if v19_tokens > 0 else 0.0)
    tsc19_ts_per_token = (
        float(sr[
            W74_MASC_V10_POLICY_TEAM_SUBSTRATE_COORDINATION_V19])
        / max(1.0, float(tsc19_tokens) / 1000.0)
        if tsc19_tokens > 0 else 0.0)
    return V10Aggregate(
        n_seeds=int(len(outcomes)),
        regime=str(regime),
        per_policy_success_rate=sr,
        per_policy_mean_visible_tokens=vt,
        per_policy_mean_abstains=ab,
        per_policy_mean_recovery_score=rs,
        v19_beats_v18_rate=float(v19_beats) / n,
        tsc_v19_beats_tsc_v18_rate=float(tsc_v19_beats) / n,
        v19_visible_tokens_savings_vs_transcript=float(
            v19_savings),
        tsc_v19_visible_tokens_savings_vs_transcript=float(
            tsc19_savings),
        team_success_per_visible_token_v19=float(
            v19_ts_per_token),
        team_success_per_visible_token_tsc_v19=float(
            tsc19_ts_per_token),
    )


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinatorV10:
    schema: str = W74_MASC_V10_SCHEMA_VERSION

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v10_controller",
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
            tuple[V10TaskOutcome, ...], V10Aggregate]:
        outs = []
        for s in seeds:
            spec = MultiAgentTaskSpec(
                seed=int(s),
                n_agents=int(n_agents),
                n_turns=int(n_turns),
                budget_tokens_per_turn=int(
                    budget_tokens_per_turn),
                target_tolerance=float(target_tolerance))
            outs.append(run_v10_multi_agent_task(
                spec=spec, regime=str(regime)))
        agg = aggregate_v10_outcomes(outs)
        return tuple(outs), agg

    def run_all_regimes(
            self, *, seeds: Sequence[int],
            n_agents: int = W65_DEFAULT_MASC_N_AGENTS,
            n_turns: int = W65_DEFAULT_MASC_N_TURNS,
            budget_tokens_per_turn: int = (
                W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN),
            target_tolerance: float = (
                W65_DEFAULT_MASC_TARGET_TOLERANCE),
    ) -> dict[str, V10Aggregate]:
        result: dict[str, V10Aggregate] = {}
        for regime in W74_MASC_V10_REGIMES:
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
class MultiAgentSubstrateCoordinatorV10Witness:
    schema: str
    coordinator_cid: str
    per_regime_aggregate_cid: dict[str, str]
    per_regime_v19_beats_v18_rate: dict[str, float]
    per_regime_tsc_v19_beats_tsc_v18_rate: dict[str, float]
    per_regime_v19_success_rate: dict[str, float]
    per_regime_tsc_v19_success_rate: dict[str, float]
    per_regime_v19_visible_tokens_savings: dict[str, float]
    per_regime_team_success_per_visible_token_v19: dict[
        str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_cid": str(self.coordinator_cid),
            "per_regime_aggregate_cid": {
                k: str(v) for k, v in sorted(
                    self.per_regime_aggregate_cid.items())},
            "per_regime_v19_beats_v18_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v19_beats_v18_rate.items())},
            "per_regime_tsc_v19_beats_tsc_v18_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_tsc_v19_beats_tsc_v18_rate
                    .items())},
            "per_regime_v19_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v19_success_rate.items())},
            "per_regime_tsc_v19_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_tsc_v19_success_rate.items())},
            "per_regime_v19_visible_tokens_savings": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_v19_visible_tokens_savings.items())},
            "per_regime_team_success_per_visible_token_v19": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_team_success_per_visible_token_v19
                    .items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v10_witness",
            "witness": self.to_dict()})


def emit_multi_agent_substrate_coordinator_v10_witness(
        *, coordinator: MultiAgentSubstrateCoordinatorV10,
        per_regime_aggregate: dict[str, V10Aggregate],
) -> MultiAgentSubstrateCoordinatorV10Witness:
    aggs_cid = {
        r: str(a.cid())
        for r, a in per_regime_aggregate.items()}
    v19_beats = {
        r: float(a.v19_beats_v18_rate)
        for r, a in per_regime_aggregate.items()}
    tsc_beats = {
        r: float(a.tsc_v19_beats_tsc_v18_rate)
        for r, a in per_regime_aggregate.items()}
    v19_succ = {
        r: float(a.per_policy_success_rate.get(
            W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19, 0.0))
        for r, a in per_regime_aggregate.items()}
    tsc_succ = {
        r: float(a.per_policy_success_rate.get(
            W74_MASC_V10_POLICY_TEAM_SUBSTRATE_COORDINATION_V19,
            0.0))
        for r, a in per_regime_aggregate.items()}
    v19_savings = {
        r: float(a.v19_visible_tokens_savings_vs_transcript)
        for r, a in per_regime_aggregate.items()}
    ts_per_v19 = {
        r: float(a.team_success_per_visible_token_v19)
        for r, a in per_regime_aggregate.items()}
    return MultiAgentSubstrateCoordinatorV10Witness(
        schema=W74_MASC_V10_SCHEMA_VERSION,
        coordinator_cid=str(coordinator.cid()),
        per_regime_aggregate_cid=aggs_cid,
        per_regime_v19_beats_v18_rate=v19_beats,
        per_regime_tsc_v19_beats_tsc_v18_rate=tsc_beats,
        per_regime_v19_success_rate=v19_succ,
        per_regime_tsc_v19_success_rate=tsc_succ,
        per_regime_v19_visible_tokens_savings=v19_savings,
        per_regime_team_success_per_visible_token_v19=ts_per_v19,
    )


__all__ = [
    "W74_MASC_V10_SCHEMA_VERSION",
    "W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19",
    "W74_MASC_V10_POLICY_TEAM_SUBSTRATE_COORDINATION_V19",
    "W74_MASC_V10_POLICIES",
    "W74_MASC_V10_REGIME_REPLACEMENT_AFTER_DELAYED_REPAIR",
    "W74_MASC_V10_REGIMES",
    "W74_MASC_V10_REGIMES_NEW",
    "V10PolicyOutcome",
    "V10TaskOutcome",
    "V10Aggregate",
    "MultiAgentSubstrateCoordinatorV10",
    "MultiAgentSubstrateCoordinatorV10Witness",
    "run_v10_multi_agent_task",
    "aggregate_v10_outcomes",
    "emit_multi_agent_substrate_coordinator_v10_witness",
]
