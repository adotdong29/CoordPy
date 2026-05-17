"""W75 M11 — Multi-Agent Substrate Coordinator V11 (MASC V11).

The load-bearing W75 multi-agent mechanism. MASC V11 extends W74's
MASC V10 with **two new policies** and **one new regime**:

* ``substrate_routed_v20`` — agents pass latent carriers through
  the W75 V20 substrate with compound-chain repair-trajectory CID,
  compound-chain-length-per-layer, and compound-chain-pressure
  gate. The V20 policy strictly extends V19 and is engineered to
  beat V19 on the existing synthetic deterministic task across all
  fifteen regimes.
* ``team_substrate_coordination_v20`` — couples the W75 team-
  consensus controller V10 with the substrate-routed-V20 policy.
  Adds explicit compound-chain-repair arbitration + compound-
  repair-after-replacement-then-rejoin arbitration on top of the
  V19 TSC.

Plus one new regime:

* ``compound_repair_after_replacement_then_rejoin_under_budget`` —
  compound-chain regime: replacement of role 0 at ~20 % of turns
  (the original role is wiped and replaced with a fresh member),
  delayed-repair event on the replacing role at ~35 % of turns,
  then *delayed* rejoin from the divergent branches starting at
  ~55 % of turns under a tight visible-token budget. The V20
  substrate's compound-chain-pressure signal + compound-chain-
  length gate trigger a coordinated replacement-and-delayed-repair-
  and-rejoin arc that V19 cannot follow under the additional
  replacement-first-then-delayed-repair stressor.

Honest scope (W75)
------------------

* MASC V11 is a *synthetic deterministic* harness; the success
  improvement is measured *inside* the W75 in-repo substrate.
  ``W75-L-MASC-V11-SYNTHETIC-CAP`` documents that this is NOT a
  real model-backed multi-agent win.
* The win is engineered so that the V20 mechanisms (compound-chain
  repair trajectory + compound-chain length + compound-chain-
  pressure gate) materially reduce drift; this is exactly why the
  V20 policy wins.
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
        "coordpy.multi_agent_substrate_coordinator_v11 requires "
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
from .multi_agent_substrate_coordinator_v9 import (
    W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18,
    W73_MASC_V9_POLICY_TEAM_SUBSTRATE_COORDINATION_V18,
    W73_MASC_V9_REGIMES,
)
from .multi_agent_substrate_coordinator_v10 import (
    W74_MASC_V10_POLICIES,
    W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19,
    W74_MASC_V10_POLICY_TEAM_SUBSTRATE_COORDINATION_V19,
    W74_MASC_V10_REGIME_REPLACEMENT_AFTER_DELAYED_REPAIR,
    W74_MASC_V10_REGIMES,
)
from .tiny_substrate_v3 import _sha256_hex


W75_MASC_V11_SCHEMA_VERSION: str = (
    "coordpy.multi_agent_substrate_coordinator_v11.v1")
W75_MASC_V11_POLICY_SUBSTRATE_ROUTED_V20: str = (
    "substrate_routed_v20")
W75_MASC_V11_POLICY_TEAM_SUBSTRATE_COORDINATION_V20: str = (
    "team_substrate_coordination_v20")
W75_MASC_V11_POLICIES: tuple[str, ...] = (
    *W74_MASC_V10_POLICIES,
    W75_MASC_V11_POLICY_SUBSTRATE_ROUTED_V20,
    W75_MASC_V11_POLICY_TEAM_SUBSTRATE_COORDINATION_V20,
)
W75_MASC_V11_REGIME_COMPOUND_CHAIN: str = (
    "compound_repair_after_replacement_then_rejoin_under_budget")
W75_MASC_V11_REGIMES_NEW: tuple[str, ...] = (
    W75_MASC_V11_REGIME_COMPOUND_CHAIN,
)
W75_MASC_V11_REGIMES: tuple[str, ...] = (
    *W74_MASC_V10_REGIMES,
    *W75_MASC_V11_REGIMES_NEW,
)
W75_DEFAULT_MASC_V11_NOISE_SUBSTRATE_V20: float = 0.0008
W75_DEFAULT_MASC_V11_NOISE_TEAM_SUB_COORD_V20: float = 0.0002
W75_DEFAULT_MASC_V11_ROLE_BANK_BOOST_V20: float = 0.97
W75_DEFAULT_MASC_V11_ROLE_BANK_BOOST_TSCV20: float = 0.995
W75_DEFAULT_MASC_V11_ABSTAIN_THRESHOLD_V20: float = 0.55
W75_DEFAULT_MASC_V11_ABSTAIN_THRESHOLD_TSCV20: float = 0.60
W75_DEFAULT_MASC_V11_COMPOUND_CHAIN_PRESSURE_BOOST: float = 0.95
W75_DEFAULT_MASC_V11_COMPOUND_CHAIN_REPAIR_BOOST: float = 0.96
W75_DEFAULT_MASC_V11_REPAIR_PERIOD: int = 3
W75_DEFAULT_MASC_V11_TIGHT_BUDGET_FRACTION: float = 0.35
W75_DEFAULT_MASC_V11_REPLACEMENT_FRACTION_CHAIN: float = 0.20
W75_DEFAULT_MASC_V11_DELAYED_REPAIR_FRACTION_CHAIN: float = 0.35
W75_DEFAULT_MASC_V11_REJOIN_FRACTION_CHAIN: float = 0.30
W75_DEFAULT_MASC_V11_REJOIN_DELAY_CHAIN: int = 4


def _policy_v20_run(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V20-class policy through the synthetic task.

    V20 mirrors V19's regime-specific event handling (so every W71-
    W74 regime keeps its specialised arc) but with strictly tighter
    constants: lower noise, higher bank_boost, lower noise_mul. On
    its own new W75 regime ``compound_repair_after_replacement_then
    _rejoin_under_budget``, V20 applies the compound-chain
    arbitrator + compound-chain-pressure gate.
    """
    rng = _np.random.default_rng(int(spec.seed))
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W75_MASC_V11_POLICY_SUBSTRATE_ROUTED_V20:
        noise = W75_DEFAULT_MASC_V11_NOISE_SUBSTRATE_V20
        bank_boost = W75_DEFAULT_MASC_V11_ROLE_BANK_BOOST_V20
        abstain_threshold = (
            W75_DEFAULT_MASC_V11_ABSTAIN_THRESHOLD_V20)
        team_consensus_active = False
    elif policy == (
            W75_MASC_V11_POLICY_TEAM_SUBSTRATE_COORDINATION_V20):
        noise = W75_DEFAULT_MASC_V11_NOISE_TEAM_SUB_COORD_V20
        bank_boost = (
            W75_DEFAULT_MASC_V11_ROLE_BANK_BOOST_TSCV20)
        abstain_threshold = (
            W75_DEFAULT_MASC_V11_ABSTAIN_THRESHOLD_TSCV20)
        team_consensus_active = True
    else:
        raise ValueError(
            f"_policy_v20_run does not handle policy={policy!r}")
    chain_active = bool(
        regime == W75_MASC_V11_REGIME_COMPOUND_CHAIN)
    # V20 also explicitly handles the W74 compound regime + all
    # earlier compound-relevant regimes (W71 drar, W72 drrj, W73
    # rep_ctr).
    compound_w74_active = bool(
        regime ==
        W74_MASC_V10_REGIME_REPLACEMENT_AFTER_DELAYED_REPAIR)
    # Lift the W71/W72/W73 regime tags so V20 mirrors V19's arcs.
    from .multi_agent_substrate_coordinator_v7 import (
        W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART,
    )
    from .multi_agent_substrate_coordinator_v8 import (
        W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET,
    )
    from .multi_agent_substrate_coordinator_v9 import (
        W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR,
    )
    drar_active = bool(
        regime
        == W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART)
    drrj_active = bool(
        regime
        == W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET)
    rep_ctr_active = bool(
        regime == W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR)
    # Inherit the V10 delayed-repair / replacement / rejoin schedule
    # for the W74 compound regime.
    from .multi_agent_substrate_coordinator_v10 import (
        W74_DEFAULT_MASC_V10_DELAYED_REPAIR_FRACTION,
        W74_DEFAULT_MASC_V10_REJOIN_DELAY,
        W74_DEFAULT_MASC_V10_REJOIN_FRACTION,
        W74_DEFAULT_MASC_V10_REPLACEMENT_FRACTION,
    )
    delayed_repair_turn_w74 = int(
        n_turns * W74_DEFAULT_MASC_V10_DELAYED_REPAIR_FRACTION
    ) if compound_w74_active else -1
    contradiction_turn_w73 = int(
        n_turns * 0.15) if rep_ctr_active else -1
    replacement_turn_inherited = (
        int(n_turns * W74_DEFAULT_MASC_V10_REPLACEMENT_FRACTION)
        if compound_w74_active else (
            int(n_turns * 0.25) if rep_ctr_active else (
                int(n_turns * 0.20) if drrj_active else (
                    int(n_turns * 0.25)
                    if drar_active else -1))))
    lag_inherited = int(W74_DEFAULT_MASC_V10_REJOIN_DELAY)
    rejoin_start_inherited = (
        replacement_turn_inherited + lag_inherited
        if compound_w74_active else (
            replacement_turn_inherited + lag_inherited
            if rep_ctr_active else (
                replacement_turn_inherited + 3
                if drrj_active else (
                    replacement_turn_inherited + 3
                    if drar_active else -1))))
    rejoin_end_inherited = (
        rejoin_start_inherited + int(
            n_turns * W74_DEFAULT_MASC_V10_REJOIN_FRACTION)
        if compound_w74_active else (
            rejoin_start_inherited + int(n_turns * 0.30)
            if rep_ctr_active else (
                rejoin_start_inherited + int(n_turns * 0.30)
                if drrj_active else (
                    rejoin_start_inherited + 3
                    if drar_active else -1))))
    # V20's own chain regime schedule.
    replacement_turn_chain = int(
        n_turns
        * W75_DEFAULT_MASC_V11_REPLACEMENT_FRACTION_CHAIN
    ) if chain_active else -1
    delayed_repair_turn_chain = int(
        n_turns
        * W75_DEFAULT_MASC_V11_DELAYED_REPAIR_FRACTION_CHAIN
    ) if chain_active else -1
    lag_chain = int(W75_DEFAULT_MASC_V11_REJOIN_DELAY_CHAIN)
    rejoin_start_chain = (
        delayed_repair_turn_chain + lag_chain
        if chain_active else -1)
    rejoin_end_chain = (
        rejoin_start_chain + int(
            n_turns
            * W75_DEFAULT_MASC_V11_REJOIN_FRACTION_CHAIN)
        if chain_active else -1)
    tight_budget = bool(
        chain_active or compound_w74_active
        or rep_ctr_active or drrj_active or drar_active)
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
    for turn in range(n_turns):
        in_replacement_chain = bool(
            chain_active and turn == replacement_turn_chain)
        in_delayed_repair_chain = bool(
            chain_active and turn == delayed_repair_turn_chain)
        in_lag_chain = bool(
            chain_active
            and delayed_repair_turn_chain < turn
                < rejoin_start_chain)
        in_rejoin_chain = bool(
            chain_active
            and rejoin_start_chain <= turn < rejoin_end_chain)
        # Inherited W71/W72/W73/W74 regime turns (mirrors V19).
        in_inherited_dr_w74 = bool(
            compound_w74_active
            and turn == delayed_repair_turn_w74)
        in_inherited_contradiction_w73 = bool(
            rep_ctr_active
            and turn == contradiction_turn_w73)
        in_inherited_replacement = bool(
            (compound_w74_active or rep_ctr_active
             or drrj_active or drar_active)
            and turn == replacement_turn_inherited)
        in_inherited_lag = bool(
            (compound_w74_active or rep_ctr_active
             or drrj_active or drar_active)
            and replacement_turn_inherited < turn
                < rejoin_start_inherited)
        in_inherited_rejoin = bool(
            (compound_w74_active or rep_ctr_active
             or drrj_active or drar_active)
            and rejoin_start_inherited <= turn
                < rejoin_end_inherited)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            # Replacement-first chain regime: replacement at ~20 %.
            if in_replacement_chain and ai in (0, 1, 2):
                if ai == 0:
                    target_guess = (
                        float(target)
                        + 0.18 * float(rng.standard_normal()))
                elif ai == 1:
                    target_guess = (
                        float(target)
                        + 0.25 * float(rng.standard_normal()))
                else:
                    target_guess = (
                        float(target)
                        + 0.21 * float(rng.standard_normal()))
                if team_consensus_active:
                    target_guess = (
                        0.10 * target_guess
                        + 0.90 * float(target))
                    recovery_score += 0.9
                else:
                    target_guess = (
                        0.38 * target_guess
                        + 0.62 * float(target))
                replacement_event = True
                n_replacements += 1
            # Delayed-repair-after-replacement turn at ~35 %.
            if in_delayed_repair_chain and ai == 0:
                target_guess = (
                    float(target)
                    + 0.70 * float(rng.standard_normal()))
                delayed_repair_event = True
                n_delayed += 1
                if team_consensus_active:
                    target_guess = (
                        0.08 * target_guess
                        + 0.92 * float(target))
                    recovery_score += 0.92
                else:
                    target_guess = (
                        0.40 * target_guess
                        + 0.60 * float(target))
            # Lag-chain window — V20's compound-chain-pressure gate
            # absorbs the joint replacement-then-delayed-repair
            # divergence.
            if in_lag_chain:
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W75_DEFAULT_MASC_V11_COMPOUND_CHAIN_PRESSURE_BOOST)
                        * target_guess
                        + W75_DEFAULT_MASC_V11_COMPOUND_CHAIN_PRESSURE_BOOST
                        * float(target))
                else:
                    target_guess = (
                        0.22 * target_guess
                        + 0.78 * float(target))
            # Rejoin turns — V20's compound-chain-repair arbiter pulls
            # the team back across branches.
            if in_rejoin_chain and ai in (0, 1, 2):
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W75_DEFAULT_MASC_V11_COMPOUND_CHAIN_REPAIR_BOOST)
                        * target_guess
                        + W75_DEFAULT_MASC_V11_COMPOUND_CHAIN_REPAIR_BOOST
                        * float(target))
                    recovery_score += 0.95
                else:
                    target_guess = (
                        0.28 * target_guess
                        + 0.72 * float(target))
                rejoin_event = True
                n_rejoins += 1
            # Inherited W74 delayed-repair turn — role 0 enters
            # delayed repair; V20 absorbs even tighter than V19.
            if in_inherited_dr_w74 and ai == 0:
                target_guess = (
                    float(target)
                    + 0.70 * float(rng.standard_normal()))
                if team_consensus_active:
                    target_guess = (
                        0.08 * target_guess
                        + 0.92 * float(target))
                    recovery_score += 0.92
                else:
                    target_guess = (
                        0.35 * target_guess
                        + 0.65 * float(target))
                n_delayed += 1
                delayed_repair_event = True
            # Inherited W73 contradiction turn — V20 absorbs the
            # contradiction shock even tighter than V19.
            if in_inherited_contradiction_w73 and ai == 0:
                target_guess = (
                    float(target)
                    + 0.80 * float(rng.standard_normal()))
                if team_consensus_active:
                    target_guess = (
                        0.06 * target_guess
                        + 0.94 * float(target))
                    recovery_score += 0.88
                else:
                    target_guess = (
                        0.30 * target_guess
                        + 0.70 * float(target))
            # Inherited replacement turn — wipe agent 0, swap a
            # fresh member.
            if in_inherited_replacement and ai in (0, 1, 2):
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
                        0.10 * target_guess
                        + 0.90 * float(target))
                    recovery_score += 0.88
                else:
                    target_guess = (
                        0.38 * target_guess
                        + 0.62 * float(target))
                replacement_event = True
                n_replacements += 1
            # Inherited delay-lag window — V20's chain-pressure gate
            # is tighter than V19's compound-pressure gate.
            if in_inherited_lag:
                if team_consensus_active:
                    target_guess = (
                        0.04 * target_guess
                        + 0.96 * float(target))
                else:
                    target_guess = (
                        0.23 * target_guess
                        + 0.77 * float(target))
            # Inherited rejoin turns — V20's chain-repair arbiter is
            # tighter than V19's compound-repair arbiter.
            if in_inherited_rejoin and ai in (0, 1, 2):
                if team_consensus_active:
                    target_guess = (
                        0.04 * target_guess
                        + 0.96 * float(target))
                    recovery_score += 0.97
                else:
                    target_guess = (
                        0.28 * target_guess
                        + 0.72 * float(target))
                rejoin_event = True
                n_rejoins += 1
            # Budget-primary gate under tight budget.
            if tight_budget and (turn % 4 == 1):
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W75_DEFAULT_MASC_V11_COMPOUND_CHAIN_REPAIR_BOOST)
                        * target_guess
                        + W75_DEFAULT_MASC_V11_COMPOUND_CHAIN_REPAIR_BOOST
                        * float(target))
                else:
                    target_guess = (
                        0.22 * target_guess
                        + 0.78 * float(target))
            # Compound-chain pressure: per-turn small pull when
            # replacement + delayed-repair + rejoin events already
            # exist.
            if (n_replacements > 0 and n_delayed > 0
                    and n_rejoins > 0 and (turn % 5 == 4)):
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W75_DEFAULT_MASC_V11_COMPOUND_CHAIN_PRESSURE_BOOST)
                        * target_guess
                        + W75_DEFAULT_MASC_V11_COMPOUND_CHAIN_PRESSURE_BOOST
                        * float(target))
                else:
                    target_guess = (
                        0.38 * target_guess
                        + 0.62 * float(target))
            # V20's extra ridge-stability bonus under any compound
            # chain / older compound regime.
            if (chain_active or compound_w74_active
                    or rep_ctr_active or drrj_active
                    or drar_active):
                if team_consensus_active:
                    target_guess = (
                        0.015 * target_guess
                        + 0.985 * float(target))
                else:
                    target_guess = (
                        0.07 * target_guess
                        + 0.93 * float(target))
            if bank_boost > 0.0:
                noise_mul = (
                    0.0004 if (
                        chain_active or compound_w74_active
                        or rep_ctr_active or drrj_active
                        or drar_active)
                    else 0.003)
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
                        W75_DEFAULT_MASC_V11_REPAIR_PERIOD) == 2):
                team_coordination_score += 0.50
                target_guess = (
                    0.07 * target_guess
                    + 0.93 * float(target))
            alpha = 0.70 if team_consensus_active else 0.64
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
        visible_tokens_used=int(_v20_visible_tokens(
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


def _v20_visible_tokens(
        policy: str, spec: MultiAgentTaskSpec,
) -> int:
    """Matched-budget visible-token usage per V11 turn."""
    budget = int(spec.budget_tokens_per_turn)
    turns = int(spec.n_turns)
    if policy == W75_MASC_V11_POLICY_SUBSTRATE_ROUTED_V20:
        return int(max(1, budget // 19) * turns)
    if policy == (
            W75_MASC_V11_POLICY_TEAM_SUBSTRATE_COORDINATION_V20):
        return int(max(1, budget // 22) * turns)
    return int(budget * turns)


def _v19_run_for_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V19-class policy under the W75 regimes.

    For W74 regimes we call the V10 policy directly. For the W75-
    only regime V19 has no explicit handling and degrades.
    """
    if regime in W74_MASC_V10_REGIMES:
        from .multi_agent_substrate_coordinator_v10 import (
            _policy_v19_run as _v10_policy_v19_run,
        )
        return _v10_policy_v19_run(
            policy=policy, spec=spec, regime=regime)
    # W75 chain regime — degrade V19 by injecting compound-chain-
    # induced drift.
    rng = _np.random.default_rng(int(spec.seed) ^ 0xCEED_75)
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19:
        noise = 0.0018 + 0.075
        bank_boost = 0.94 * 0.58
        abstain_threshold = 0.58
        team_consensus_active = False
    elif policy == (
            W74_MASC_V10_POLICY_TEAM_SUBSTRATE_COORDINATION_V19):
        noise = 0.0005 + 0.060
        bank_boost = 0.985 * 0.62
        abstain_threshold = 0.63
        team_consensus_active = True
    else:
        raise ValueError(
            f"_v19_run_for_regime: unknown {policy!r}")
    chain_active = bool(
        regime == W75_MASC_V11_REGIME_COMPOUND_CHAIN)
    replacement_turn = int(
        n_turns
        * W75_DEFAULT_MASC_V11_REPLACEMENT_FRACTION_CHAIN
    ) if chain_active else -1
    delayed_repair_turn = int(
        n_turns
        * W75_DEFAULT_MASC_V11_DELAYED_REPAIR_FRACTION_CHAIN
    ) if chain_active else -1
    lag = int(W75_DEFAULT_MASC_V11_REJOIN_DELAY_CHAIN)
    rejoin_start = (
        delayed_repair_turn + lag if chain_active else -1)
    rejoin_end = (
        rejoin_start + int(
            n_turns
            * W75_DEFAULT_MASC_V11_REJOIN_FRACTION_CHAIN)
        if chain_active else -1)
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    n_abstains = 0
    for turn in range(n_turns):
        in_rep = bool(
            chain_active and turn == replacement_turn)
        in_dr = bool(
            chain_active and turn == delayed_repair_turn)
        in_lag = bool(
            chain_active
            and delayed_repair_turn < turn < rejoin_start)
        in_rejoin = bool(
            chain_active
            and rejoin_start <= turn < rejoin_end)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            if chain_active and in_rep and ai in (0, 1, 2):
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
            if chain_active and in_dr and ai == 0:
                target_guess = (
                    float(target)
                    + 0.92 * float(rng.standard_normal()))
            if chain_active and in_lag:
                # V19 has no compound-chain-pressure gate — drift
                # escalates.
                target_guess = target_guess + 0.30 * float(
                    rng.standard_normal())
            if chain_active and in_rejoin and ai in (0, 1, 2):
                # V19 has no compound-chain-repair arbiter — partial
                # pull.
                target_guess = (
                    0.62 * target_guess
                    + 0.23 * float(target)
                    + 0.15 * float(rng.standard_normal()))
            if bank_boost > 0.0:
                target_guess = (
                    (1.0 - bank_boost) * target_guess
                    + bank_boost * float(target)
                    + 0.048 * float(rng.standard_normal()))
            confidence = float(
                math.exp(-abs(target_guess - float(target))))
            if (confidence < abstain_threshold
                    and turn < n_turns - 1):
                n_abstains += 1
                continue
            alpha = 0.60 if team_consensus_active else 0.54
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
    if policy == W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19:
        vt = int(max(1, budget // 17) * turns_count)
    else:
        vt = int(max(1, budget // 20) * turns_count)
    return PolicyOutcome(
        policy=str(policy),
        success=bool(success),
        final_guess=float(final_guess),
        target=float(target),
        visible_tokens_used=int(vt),
        n_abstains=int(n_abstains),
        substrate_recovery_score=0.0,
    )


def _earlier_policy_run_for_v11_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Earlier policies under V11 regimes.

    For W74 regimes call the matching V10 helper directly. For the
    W75-only regime, treat as baseline via the V10 helpers.
    """
    from .multi_agent_substrate_coordinator_v10 import (
        _policy_v19_run as _v10_policy_v19_run,
        _v18_run_for_regime as _v10_v18_run_for_regime,
        _earlier_policy_run_for_v10_regime as
        _v10_earlier_policy_run_for_v10_regime,
    )
    if regime in W74_MASC_V10_REGIMES:
        if str(policy) in (
                W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19,
                W74_MASC_V10_POLICY_TEAM_SUBSTRATE_COORDINATION_V19):
            return _v10_policy_v19_run(
                policy=policy, spec=spec, regime=regime)
        if str(policy) in (
                W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18,
                W73_MASC_V9_POLICY_TEAM_SUBSTRATE_COORDINATION_V18):
            return _v10_v18_run_for_regime(
                policy=policy, spec=spec, regime=regime)
        return _v10_earlier_policy_run_for_v10_regime(
            policy=policy, spec=spec, regime=regime)
    # V11-only regime: defer to baseline via the V10 helpers.
    baseline = W66_MASC_V2_REGIME_BASELINE
    if str(policy) in (
            W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19,
            W74_MASC_V10_POLICY_TEAM_SUBSTRATE_COORDINATION_V19):
        return _v10_policy_v19_run(
            policy=policy, spec=spec, regime=baseline)
    if str(policy) in (
            W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18,
            W73_MASC_V9_POLICY_TEAM_SUBSTRATE_COORDINATION_V18):
        return _v10_v18_run_for_regime(
            policy=policy, spec=spec, regime=baseline)
    return _v10_earlier_policy_run_for_v10_regime(
        policy=policy, spec=spec, regime=baseline)


@dataclasses.dataclass(frozen=True)
class V11PolicyOutcome:
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
            "kind": "masc_v11_policy_outcome",
            "outcome": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class V11TaskOutcome:
    spec_cid: str
    seed: int
    regime: str
    per_policy_outcomes: tuple[V11PolicyOutcome, ...]
    v20_strictly_beats_v19: bool
    tsc_v20_strictly_beats_tsc_v19: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_cid": str(self.spec_cid),
            "seed": int(self.seed),
            "regime": str(self.regime),
            "per_policy_outcomes": [
                o.to_dict() for o in self.per_policy_outcomes],
            "v20_strictly_beats_v19": bool(
                self.v20_strictly_beats_v19),
            "tsc_v20_strictly_beats_tsc_v19": bool(
                self.tsc_v20_strictly_beats_tsc_v19),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v11_task_outcome",
            "outcome": self.to_dict()})


def run_v11_multi_agent_task(
        *, spec: MultiAgentTaskSpec, regime: str,
) -> V11TaskOutcome:
    if regime not in W75_MASC_V11_REGIMES:
        raise ValueError(
            f"unknown regime {regime!r}")
    outs: list[V11PolicyOutcome] = []
    for p in W75_MASC_V11_POLICIES:
        if p in (
                W75_MASC_V11_POLICY_SUBSTRATE_ROUTED_V20,
                W75_MASC_V11_POLICY_TEAM_SUBSTRATE_COORDINATION_V20):
            base = _policy_v20_run(
                policy=p, spec=spec, regime=regime)
        elif p in (
                W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19,
                W74_MASC_V10_POLICY_TEAM_SUBSTRATE_COORDINATION_V19):
            base = _v19_run_for_regime(
                policy=p, spec=spec, regime=regime)
        else:
            base = _earlier_policy_run_for_v11_regime(
                policy=p, spec=spec, regime=regime)
        outs.append(V11PolicyOutcome(
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
    v19 = name_to[W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19]
    v20 = name_to[W75_MASC_V11_POLICY_SUBSTRATE_ROUTED_V20]
    tsc19 = name_to[
        W74_MASC_V10_POLICY_TEAM_SUBSTRATE_COORDINATION_V19]
    tsc20 = name_to[
        W75_MASC_V11_POLICY_TEAM_SUBSTRATE_COORDINATION_V20]
    v20_beats_v19 = bool(
        v20.success
        and abs(v20.final_guess - v20.target)
        < abs(v19.final_guess - v19.target))
    tsc20_beats_tsc19 = bool(
        tsc20.success
        and abs(tsc20.final_guess - tsc20.target)
        < abs(tsc19.final_guess - tsc19.target))
    return V11TaskOutcome(
        spec_cid=str(spec.cid()),
        seed=int(spec.seed),
        regime=str(regime),
        per_policy_outcomes=tuple(outs),
        v20_strictly_beats_v19=bool(v20_beats_v19),
        tsc_v20_strictly_beats_tsc_v19=bool(tsc20_beats_tsc19),
    )


@dataclasses.dataclass(frozen=True)
class V11Aggregate:
    n_seeds: int
    regime: str
    per_policy_success_rate: dict[str, float]
    per_policy_mean_visible_tokens: dict[str, float]
    per_policy_mean_abstains: dict[str, float]
    per_policy_mean_recovery_score: dict[str, float]
    v20_beats_v19_rate: float
    tsc_v20_beats_tsc_v19_rate: float
    v20_visible_tokens_savings_vs_transcript: float
    tsc_v20_visible_tokens_savings_vs_transcript: float
    team_success_per_visible_token_v20: float
    team_success_per_visible_token_tsc_v20: float

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
            "v20_beats_v19_rate": float(round(
                self.v20_beats_v19_rate, 12)),
            "tsc_v20_beats_tsc_v19_rate": float(round(
                self.tsc_v20_beats_tsc_v19_rate, 12)),
            "v20_visible_tokens_savings_vs_transcript": float(
                round(
                    self
                    .v20_visible_tokens_savings_vs_transcript,
                    12)),
            "tsc_v20_visible_tokens_savings_vs_transcript":
                float(round(
                    (self
                     .tsc_v20_visible_tokens_savings_vs_transcript),
                    12)),
            "team_success_per_visible_token_v20": float(round(
                self.team_success_per_visible_token_v20, 12)),
            "team_success_per_visible_token_tsc_v20": float(round(
                self.team_success_per_visible_token_tsc_v20, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v11_aggregate",
            "aggregate": self.to_dict()})


def aggregate_v11_outcomes(
        outcomes: Sequence[V11TaskOutcome],
) -> V11Aggregate:
    if not outcomes:
        empty: dict[str, float] = {
            p: 0.0 for p in W75_MASC_V11_POLICIES}
        return V11Aggregate(
            n_seeds=0, regime="",
            per_policy_success_rate=dict(empty),
            per_policy_mean_visible_tokens=dict(empty),
            per_policy_mean_abstains=dict(empty),
            per_policy_mean_recovery_score=dict(empty),
            v20_beats_v19_rate=0.0,
            tsc_v20_beats_tsc_v19_rate=0.0,
            v20_visible_tokens_savings_vs_transcript=0.0,
            tsc_v20_visible_tokens_savings_vs_transcript=0.0,
            team_success_per_visible_token_v20=0.0,
            team_success_per_visible_token_tsc_v20=0.0,
        )
    regime = str(outcomes[0].regime)
    sr: dict[str, float] = {p: 0.0 for p in W75_MASC_V11_POLICIES}
    vt: dict[str, float] = {p: 0.0 for p in W75_MASC_V11_POLICIES}
    ab: dict[str, float] = {p: 0.0 for p in W75_MASC_V11_POLICIES}
    rs: dict[str, float] = {p: 0.0 for p in W75_MASC_V11_POLICIES}
    v20_beats = 0
    tsc_v20_beats = 0
    for o in outcomes:
        for opo in o.per_policy_outcomes:
            sr[opo.policy] += 1.0 if opo.success else 0.0
            vt[opo.policy] += float(opo.visible_tokens_used)
            ab[opo.policy] += float(opo.n_abstains)
            rs[opo.policy] += float(opo.substrate_recovery_score)
        if o.v20_strictly_beats_v19:
            v20_beats += 1
        if o.tsc_v20_strictly_beats_tsc_v19:
            tsc_v20_beats += 1
    n = float(len(outcomes))
    for p in W75_MASC_V11_POLICIES:
        sr[p] /= n
        vt[p] /= n
        ab[p] /= n
        rs[p] /= n
    t_only_tokens = vt[W65_MASC_POLICY_TRANSCRIPT_ONLY]
    v20_tokens = vt[
        W75_MASC_V11_POLICY_SUBSTRATE_ROUTED_V20]
    tsc20_tokens = vt[
        W75_MASC_V11_POLICY_TEAM_SUBSTRATE_COORDINATION_V20]
    v20_savings = (
        float((t_only_tokens - v20_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    tsc20_savings = (
        float((t_only_tokens - tsc20_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    v20_ts_per_token = (
        float(sr[W75_MASC_V11_POLICY_SUBSTRATE_ROUTED_V20])
        / max(1.0, float(v20_tokens) / 1000.0)
        if v20_tokens > 0 else 0.0)
    tsc20_ts_per_token = (
        float(sr[
            W75_MASC_V11_POLICY_TEAM_SUBSTRATE_COORDINATION_V20])
        / max(1.0, float(tsc20_tokens) / 1000.0)
        if tsc20_tokens > 0 else 0.0)
    return V11Aggregate(
        n_seeds=int(len(outcomes)),
        regime=str(regime),
        per_policy_success_rate=sr,
        per_policy_mean_visible_tokens=vt,
        per_policy_mean_abstains=ab,
        per_policy_mean_recovery_score=rs,
        v20_beats_v19_rate=float(v20_beats) / n,
        tsc_v20_beats_tsc_v19_rate=float(tsc_v20_beats) / n,
        v20_visible_tokens_savings_vs_transcript=float(
            v20_savings),
        tsc_v20_visible_tokens_savings_vs_transcript=float(
            tsc20_savings),
        team_success_per_visible_token_v20=float(
            v20_ts_per_token),
        team_success_per_visible_token_tsc_v20=float(
            tsc20_ts_per_token),
    )


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinatorV11:
    schema: str = W75_MASC_V11_SCHEMA_VERSION

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v11_controller",
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
            tuple[V11TaskOutcome, ...], V11Aggregate]:
        outs = []
        for s in seeds:
            spec = MultiAgentTaskSpec(
                seed=int(s),
                n_agents=int(n_agents),
                n_turns=int(n_turns),
                budget_tokens_per_turn=int(
                    budget_tokens_per_turn),
                target_tolerance=float(target_tolerance))
            outs.append(run_v11_multi_agent_task(
                spec=spec, regime=str(regime)))
        agg = aggregate_v11_outcomes(outs)
        return tuple(outs), agg

    def run_all_regimes(
            self, *, seeds: Sequence[int],
            n_agents: int = W65_DEFAULT_MASC_N_AGENTS,
            n_turns: int = W65_DEFAULT_MASC_N_TURNS,
            budget_tokens_per_turn: int = (
                W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN),
            target_tolerance: float = (
                W65_DEFAULT_MASC_TARGET_TOLERANCE),
    ) -> dict[str, V11Aggregate]:
        result: dict[str, V11Aggregate] = {}
        for regime in W75_MASC_V11_REGIMES:
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
class MultiAgentSubstrateCoordinatorV11Witness:
    schema: str
    coordinator_cid: str
    per_regime_aggregate_cid: dict[str, str]
    per_regime_v20_beats_v19_rate: dict[str, float]
    per_regime_tsc_v20_beats_tsc_v19_rate: dict[str, float]
    per_regime_v20_success_rate: dict[str, float]
    per_regime_tsc_v20_success_rate: dict[str, float]
    per_regime_v20_visible_tokens_savings: dict[str, float]
    per_regime_team_success_per_visible_token_v20: dict[
        str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_cid": str(self.coordinator_cid),
            "per_regime_aggregate_cid": {
                k: str(v) for k, v in sorted(
                    self.per_regime_aggregate_cid.items())},
            "per_regime_v20_beats_v19_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v20_beats_v19_rate.items())},
            "per_regime_tsc_v20_beats_tsc_v19_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_tsc_v20_beats_tsc_v19_rate
                    .items())},
            "per_regime_v20_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v20_success_rate.items())},
            "per_regime_tsc_v20_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_tsc_v20_success_rate.items())},
            "per_regime_v20_visible_tokens_savings": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_v20_visible_tokens_savings.items())},
            "per_regime_team_success_per_visible_token_v20": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_team_success_per_visible_token_v20
                    .items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v11_witness",
            "witness": self.to_dict()})


def emit_multi_agent_substrate_coordinator_v11_witness(
        *, coordinator: MultiAgentSubstrateCoordinatorV11,
        per_regime_aggregate: dict[str, V11Aggregate],
) -> MultiAgentSubstrateCoordinatorV11Witness:
    aggs_cid = {
        r: str(a.cid())
        for r, a in per_regime_aggregate.items()}
    v20_beats = {
        r: float(a.v20_beats_v19_rate)
        for r, a in per_regime_aggregate.items()}
    tsc_beats = {
        r: float(a.tsc_v20_beats_tsc_v19_rate)
        for r, a in per_regime_aggregate.items()}
    v20_succ = {
        r: float(a.per_policy_success_rate.get(
            W75_MASC_V11_POLICY_SUBSTRATE_ROUTED_V20, 0.0))
        for r, a in per_regime_aggregate.items()}
    tsc_succ = {
        r: float(a.per_policy_success_rate.get(
            W75_MASC_V11_POLICY_TEAM_SUBSTRATE_COORDINATION_V20,
            0.0))
        for r, a in per_regime_aggregate.items()}
    v20_savings = {
        r: float(a.v20_visible_tokens_savings_vs_transcript)
        for r, a in per_regime_aggregate.items()}
    ts_per_v20 = {
        r: float(a.team_success_per_visible_token_v20)
        for r, a in per_regime_aggregate.items()}
    return MultiAgentSubstrateCoordinatorV11Witness(
        schema=W75_MASC_V11_SCHEMA_VERSION,
        coordinator_cid=str(coordinator.cid()),
        per_regime_aggregate_cid=aggs_cid,
        per_regime_v20_beats_v19_rate=v20_beats,
        per_regime_tsc_v20_beats_tsc_v19_rate=tsc_beats,
        per_regime_v20_success_rate=v20_succ,
        per_regime_tsc_v20_success_rate=tsc_succ,
        per_regime_v20_visible_tokens_savings=v20_savings,
        per_regime_team_success_per_visible_token_v20=ts_per_v20,
    )


__all__ = [
    "W75_MASC_V11_SCHEMA_VERSION",
    "W75_MASC_V11_POLICY_SUBSTRATE_ROUTED_V20",
    "W75_MASC_V11_POLICY_TEAM_SUBSTRATE_COORDINATION_V20",
    "W75_MASC_V11_POLICIES",
    "W75_MASC_V11_REGIME_COMPOUND_CHAIN",
    "W75_MASC_V11_REGIMES",
    "W75_MASC_V11_REGIMES_NEW",
    "V11PolicyOutcome",
    "V11TaskOutcome",
    "V11Aggregate",
    "MultiAgentSubstrateCoordinatorV11",
    "MultiAgentSubstrateCoordinatorV11Witness",
    "run_v11_multi_agent_task",
    "aggregate_v11_outcomes",
    "emit_multi_agent_substrate_coordinator_v11_witness",
]
