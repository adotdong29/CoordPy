"""W82 / P2 #10 — Far-Horizon Blackout Reconstruction Benchmark V1.

Issue #10 asks for benchmark families that *intentionally exceed*
the comfort zone of today's short synthetic regimes — horizons of
1e3, 1e4, 1e5 or more, interleaved irrelevant traffic, multiple
restart/replacement cycles, branch/rejoin after deep divergence,
and partial loss during blackout. The required outputs are:

1. an explicit horizon ladder (not one convenient default)
2. failure *curves* (success-rate as a function of horizon),
   not single-point wins
3. honest comparison against the canonical bounded baselines
   (transcript-only, bounded-window-k, rolling-summary) AND the
   current substrate line (``long_horizon_reconstruction_
   substrate_v2``)
4. explicit memory-economics metrics for every (strategy,
   horizon, seed) cell: visible tokens, replay cost, recompute
   flops, reconstruction fidelity, task success under budget

V1 delivers all four.

The bench:

* generates ``FarHorizonBlackoutScenarioV1`` instances at each
  horizon on the ladder. A scenario embeds a *source event* at
  ``source_turn`` and asks the substrate to reconstruct that
  event's CID at ``current_turn = source_turn + horizon``. The
  intervening ``horizon`` turns include:

  - ``n_irrelevant_pre`` and ``n_irrelevant_post`` interleaved
    irrelevant events (so the source is not the most recent),
  - ``n_restart_cycles`` simulated restart-and-replacement
    cycles that throw away all visible state but leave the
    persistent carrier intact, and
  - ``branch_rejoin_offset`` turns at which a branch diverges
    and rejoins, so any window-shaped strategy that snapshots
    pre-branch will see corrupted post-rejoin state.

* runs the four canonical strategies at each scenario:

  - ``transcript_only`` — scans the most recent N turns within a
    fixed visible-token budget (default 800 tokens ≈ 32 turns).
  - ``bounded_window_k`` — the W79 bounded-window line, four
    explicit k's: 4, 32, 64, 128.
  - ``rolling_summary`` — the W79 cross-prompt-summary worst
    case, capped at ``summary_budget_turns`` turns with
    ``summary_fidelity`` < 0.5 outside the budget.
  - ``lhr_substrate_v2`` — the W79 long-horizon reconstruction
    substrate V2 reading from a content-addressed carrier.

* records, per (strategy, horizon, seed) cell:

  - ``task_success`` (bool — did the strategy return the correct
    source event CID?)
  - ``reconstruction_fidelity`` (float in [0, 1])
  - ``visible_tokens_used`` (int)
  - ``replay_flops`` (int)
  - ``recompute_flops`` (int)
  - ``carrier_walks`` (int)

* aggregates into ``FarHorizonFailureCurveV1`` per strategy —
  ``success_rate_at(h)`` is the fraction of seeds that
  succeeded at horizon ``h``.

* emits a content-addressed witness covering the full bench
  result. The witness CID is deterministic on the seed +
  config and is suitable for tamper-evident chaining.

The V1 bench is *not* tuned to make the substrate look easy.
The failure frontier is mapped at every horizon on the ladder.
The W82 V1 bar is:

* the LHR substrate V2 strictly dominates all bounded baselines
  on the load-bearing far-horizon ladder (success rate gap ≥ 0.5
  at horizon 1000+); and
* the substrate's *own* failure mode is mapped honestly — for
  source-turns outside the carrier (carrier-depth-bounded), the
  substrate fails by construction and the bench reports that
  failure rather than hiding it.

Honest scope (W82)
------------------

* ``W82-L-FAR-HORIZON-V1-RESEARCH-ONLY-CAP`` — explicit-import
  only. Not auto-imported by ``coordpy`` top-level.
* ``W82-L-FAR-HORIZON-V1-SYNTHETIC-CAP`` — events are
  deterministic synthetic CIDs (same family as the W78 default
  carrier). The bench measures reconstruction *fidelity*; it
  does not validate that the underlying events would be
  semantically meaningful in a live runtime.
* ``W82-L-FAR-HORIZON-V1-NO-LIVE-RUNTIME-CAP`` — the substrate
  side reads from a synthetic carrier, not a live transformer.
  The W80 transformers runtime line is orthogonal.
* ``W82-L-FAR-HORIZON-V1-NUMPY-CAP`` — pure NumPy / stdlib.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Sequence

from .bounded_window_baseline_v1 import (
    W78_ROLLING_SUMMARY_LABEL,
)
from .bounded_window_baseline_v2 import (
    W79_CROSS_PROMPT_SUMMARY_LABEL,
    W79_DEFAULT_CROSS_PROMPT_SUMMARY_BUDGET_TURNS,
    W79_DEFAULT_CROSS_PROMPT_SUMMARY_FIDELITY,
)
from .long_horizon_reconstruction_substrate_v1 import (
    LongHorizonCarrierEntry,
    LongHorizonReconstructionCarrier,
)


W82_FAR_HORIZON_BLACKOUT_BENCHMARK_V1_SCHEMA_VERSION: str = (
    "coordpy.far_horizon_blackout_benchmark_v1.v1")

# ---------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------

W82_DEFAULT_HORIZON_LADDER: tuple[int, ...] = (
    1000, 4000, 16_000, 64_000)
W82_DEFAULT_N_SEEDS_PER_HORIZON: int = 4
W82_DEFAULT_VISIBLE_TOKEN_BUDGET: int = 800
W82_DEFAULT_TOKENS_PER_TURN: int = 25
W82_DEFAULT_RECOMPUTE_FLOPS_PER_TOKEN: int = 1000
W82_DEFAULT_SUBSTRATE_WALK_FLOPS_PER_HOP: int = 80
W82_DEFAULT_MERKLE_FANOUT: int = 4
W82_DEFAULT_SUMMARY_BUDGET_TURNS: int = (
    W79_DEFAULT_CROSS_PROMPT_SUMMARY_BUDGET_TURNS)
W82_DEFAULT_SUMMARY_FIDELITY: float = (
    W79_DEFAULT_CROSS_PROMPT_SUMMARY_FIDELITY)
W82_DEFAULT_N_IRRELEVANT_PRE: int = 32
W82_DEFAULT_N_IRRELEVANT_POST: int = 32
W82_DEFAULT_N_RESTART_CYCLES: int = 2
W82_DEFAULT_BRANCH_REJOIN_OFFSET: int = 8
W82_DEFAULT_BOUNDED_WINDOW_KS: tuple[int, ...] = (4, 32, 64, 128)
W82_DEFAULT_SEED: int = 82_010_001

# Strategy labels
W82_STRATEGY_TRANSCRIPT_ONLY: str = "transcript_only"
W82_STRATEGY_BOUNDED_WINDOW_FMT: str = "bounded_window_k{k}"
W82_STRATEGY_ROLLING_SUMMARY: str = "rolling_summary"
W82_STRATEGY_LHR_SUBSTRATE_V2: str = "lhr_substrate_v2"


# ---------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------

def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _event_cid(*, scenario_seed: int, turn_index: int,
               kind: str) -> str:
    return _sha256_hex({
        "kind": "w82_far_horizon_event",
        "scenario_seed": int(scenario_seed),
        "turn_index": int(turn_index),
        "event_kind": str(kind),
    })


# ---------------------------------------------------------------
# Scenario schema
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class FarHorizonBlackoutScenarioV1:
    """One far-horizon scenario.

    A scenario asks: at ``current_turn``, reconstruct the event
    that occurred at ``source_turn``. The horizon (i.e. blackout
    width) is ``current_turn - source_turn``. The intervening
    turns include irrelevant traffic, simulated restart cycles,
    and a branch/rejoin window.
    """

    schema: str
    seed: int
    horizon_turns: int
    source_turn: int
    current_turn: int
    n_irrelevant_pre: int
    n_irrelevant_post: int
    n_restart_cycles: int
    branch_rejoin_offset: int
    source_event_cid: str
    visible_tokens_budget: int
    tokens_per_turn: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "seed": int(self.seed),
            "horizon_turns": int(self.horizon_turns),
            "source_turn": int(self.source_turn),
            "current_turn": int(self.current_turn),
            "n_irrelevant_pre": int(self.n_irrelevant_pre),
            "n_irrelevant_post": int(self.n_irrelevant_post),
            "n_restart_cycles": int(self.n_restart_cycles),
            "branch_rejoin_offset": int(self.branch_rejoin_offset),
            "source_event_cid": str(self.source_event_cid),
            "visible_tokens_budget": int(self.visible_tokens_budget),
            "tokens_per_turn": int(self.tokens_per_turn),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_far_horizon_scenario_v1",
            "scenario": self.to_dict()})


def build_far_horizon_blackout_scenario_v1(
        *, seed: int,
        horizon_turns: int,
        n_irrelevant_pre: int = W82_DEFAULT_N_IRRELEVANT_PRE,
        n_irrelevant_post: int = W82_DEFAULT_N_IRRELEVANT_POST,
        n_restart_cycles: int = W82_DEFAULT_N_RESTART_CYCLES,
        branch_rejoin_offset: int = (
            W82_DEFAULT_BRANCH_REJOIN_OFFSET),
        visible_tokens_budget: int = (
            W82_DEFAULT_VISIBLE_TOKEN_BUDGET),
        tokens_per_turn: int = W82_DEFAULT_TOKENS_PER_TURN,
) -> FarHorizonBlackoutScenarioV1:
    if int(horizon_turns) <= 0:
        raise ValueError(
            f"horizon_turns must be > 0, got {int(horizon_turns)}")
    source_turn = int(n_irrelevant_pre)
    current_turn = int(source_turn + int(horizon_turns))
    source_cid = _event_cid(
        scenario_seed=int(seed),
        turn_index=int(source_turn),
        kind="source")
    return FarHorizonBlackoutScenarioV1(
        schema=(
            W82_FAR_HORIZON_BLACKOUT_BENCHMARK_V1_SCHEMA_VERSION),
        seed=int(seed),
        horizon_turns=int(horizon_turns),
        source_turn=int(source_turn),
        current_turn=int(current_turn),
        n_irrelevant_pre=int(n_irrelevant_pre),
        n_irrelevant_post=int(n_irrelevant_post),
        n_restart_cycles=int(n_restart_cycles),
        branch_rejoin_offset=int(branch_rejoin_offset),
        source_event_cid=str(source_cid),
        visible_tokens_budget=int(visible_tokens_budget),
        tokens_per_turn=int(tokens_per_turn),
    )


def build_far_horizon_blackout_scenarios_v1(
        *, horizons: Sequence[int] = W82_DEFAULT_HORIZON_LADDER,
        n_per_horizon: int = (
            W82_DEFAULT_N_SEEDS_PER_HORIZON),
        base_seed: int = W82_DEFAULT_SEED,
        n_irrelevant_pre: int = W82_DEFAULT_N_IRRELEVANT_PRE,
        n_irrelevant_post: int = W82_DEFAULT_N_IRRELEVANT_POST,
        n_restart_cycles: int = W82_DEFAULT_N_RESTART_CYCLES,
        branch_rejoin_offset: int = (
            W82_DEFAULT_BRANCH_REJOIN_OFFSET),
        visible_tokens_budget: int = (
            W82_DEFAULT_VISIBLE_TOKEN_BUDGET),
        tokens_per_turn: int = W82_DEFAULT_TOKENS_PER_TURN,
) -> tuple[FarHorizonBlackoutScenarioV1, ...]:
    """Build a deterministic ladder of scenarios.

    Returns ``n_per_horizon * len(horizons)`` scenarios. Seeds
    are deterministic on ``base_seed`` and the (horizon, seed_
    idx) pair.
    """
    out: list[FarHorizonBlackoutScenarioV1] = []
    for h_idx, h in enumerate(horizons):
        for s_idx in range(int(n_per_horizon)):
            sub_seed = int(
                _sha256_hex({
                    "kind": "w82_scenario_seed",
                    "base_seed": int(base_seed),
                    "h_idx": int(h_idx),
                    "horizon": int(h),
                    "s_idx": int(s_idx),
                })[:12], 16) & 0x7FFFFFFF
            out.append(
                build_far_horizon_blackout_scenario_v1(
                    seed=int(sub_seed),
                    horizon_turns=int(h),
                    n_irrelevant_pre=int(n_irrelevant_pre),
                    n_irrelevant_post=int(n_irrelevant_post),
                    n_restart_cycles=int(n_restart_cycles),
                    branch_rejoin_offset=int(
                        branch_rejoin_offset),
                    visible_tokens_budget=int(
                        visible_tokens_budget),
                    tokens_per_turn=int(tokens_per_turn)))
    return tuple(out)


def _scenario_restart_turns_v1(
        scenario: FarHorizonBlackoutScenarioV1,
) -> tuple[int, ...]:
    src = int(scenario.source_turn)
    last_turn = int(scenario.current_turn)
    restart_step = max(
        1,
        int(scenario.horizon_turns) //
        max(1, int(scenario.n_restart_cycles) + 1))
    return tuple(
        int(src + (i + 1) * restart_step)
        for i in range(int(scenario.n_restart_cycles))
        if int(src + (i + 1) * restart_step) < last_turn)


def _visible_strategy_restart_erases_source_v1(
        scenario: FarHorizonBlackoutScenarioV1,
) -> bool:
    return bool(any(
        int(rt) > int(scenario.source_turn)
        for rt in _scenario_restart_turns_v1(scenario)))


def _visible_strategy_branch_conflict_v1(
        scenario: FarHorizonBlackoutScenarioV1,
) -> bool:
    return bool(
        0 < int(scenario.branch_rejoin_offset) <=
        int(scenario.horizon_turns))


# ---------------------------------------------------------------
# Carrier construction for a scenario
# ---------------------------------------------------------------

def build_carrier_for_scenario_v1(
        scenario: FarHorizonBlackoutScenarioV1,
        *, carrier_depth: int | None = None,
) -> LongHorizonReconstructionCarrier:
    """Build the persistent latent carrier that the substrate
    reads from for ``scenario``.

    The carrier is content-addressed and embeds the source
    event at ``scenario.source_turn`` plus surrounding
    irrelevant events plus restart/replacement entries plus
    a branch/rejoin marker.

    If ``carrier_depth`` is None, the carrier covers from
    turn 0 up to and including ``scenario.current_turn``.
    Set ``carrier_depth`` smaller than ``current_turn + 1`` to
    simulate a truncated carrier (then the substrate will fail
    for sources that fall outside the truncation).
    """
    last_turn = int(scenario.current_turn)
    if carrier_depth is None:
        n = int(last_turn + 1)
    else:
        n = int(carrier_depth)
        if n <= 0:
            raise ValueError("carrier_depth must be > 0")
    entries: list[LongHorizonCarrierEntry] = []
    src = int(scenario.source_turn)
    rejoin = src + int(scenario.branch_rejoin_offset)
    restart_turns = set(_scenario_restart_turns_v1(scenario))
    # We include entries from the most-recent ``n`` turns
    # ending at ``last_turn``. This means the carrier may not
    # cover early turns if ``carrier_depth < last_turn + 1``.
    start_turn = int(max(0, last_turn - n + 1))
    for ti in range(start_turn, last_turn + 1):
        if ti == src:
            kind = "source"
            cid_ = str(scenario.source_event_cid)
        elif ti == rejoin:
            kind = "branch_rejoin"
            cid_ = _event_cid(
                scenario_seed=int(scenario.seed),
                turn_index=int(ti),
                kind=kind)
        elif ti in restart_turns:
            kind = "restart"
            cid_ = _event_cid(
                scenario_seed=int(scenario.seed),
                turn_index=int(ti),
                kind=kind)
        else:
            kind = "irrelevant"
            cid_ = _event_cid(
                scenario_seed=int(scenario.seed),
                turn_index=int(ti),
                kind=kind)
        entries.append(LongHorizonCarrierEntry(
            turn_index=int(ti), event_cid=str(cid_)))
    return LongHorizonReconstructionCarrier.from_entries(
        entries)


# ---------------------------------------------------------------
# Outcome schema
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class FarHorizonBlackoutOutcomeV1:
    """One strategy's outcome on one scenario."""

    schema: str
    scenario_cid: str
    strategy_label: str
    horizon_turns: int
    task_success: bool
    reconstructed_event_cid: str
    reconstruction_fidelity: float
    visible_tokens_used: int
    replay_flops: int
    recompute_flops: int
    carrier_walks: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "scenario_cid": str(self.scenario_cid),
            "strategy_label": str(self.strategy_label),
            "horizon_turns": int(self.horizon_turns),
            "task_success": bool(self.task_success),
            "reconstructed_event_cid": str(
                self.reconstructed_event_cid),
            "reconstruction_fidelity": float(round(
                self.reconstruction_fidelity, 12)),
            "visible_tokens_used": int(self.visible_tokens_used),
            "replay_flops": int(self.replay_flops),
            "recompute_flops": int(self.recompute_flops),
            "carrier_walks": int(self.carrier_walks),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_far_horizon_outcome_v1",
            "outcome": self.to_dict()})


# ---------------------------------------------------------------
# Strategy callables
# ---------------------------------------------------------------

StrategyCallable = Callable[
    [FarHorizonBlackoutScenarioV1],
    FarHorizonBlackoutOutcomeV1,
]


def _transcript_only_strategy(
        scenario: FarHorizonBlackoutScenarioV1,
) -> FarHorizonBlackoutOutcomeV1:
    """Scan the most-recent N turns within the visible budget.

    Fails iff source is outside the visible window. The visible
    window holds ``visible_tokens_budget / tokens_per_turn``
    turns (the most recent ones, ending at ``current_turn``).
    """
    budget_turns = int(
        scenario.visible_tokens_budget //
        max(1, int(scenario.tokens_per_turn)))
    earliest_visible = int(
        scenario.current_turn - max(0, budget_turns - 1))
    can_see_source = bool(
        int(scenario.source_turn) >= earliest_visible)
    restart_erases_source = _visible_strategy_restart_erases_source_v1(
        scenario)
    branch_conflict = _visible_strategy_branch_conflict_v1(
        scenario)
    success = bool(
        can_see_source and
        not restart_erases_source and
        not branch_conflict)
    recon_cid = (
        str(scenario.source_event_cid) if success else "")
    fidelity = 1.0 if success else 0.0
    # Recompute cost: scan all visible turns.
    visible_tokens_used = int(
        min(budget_turns, scenario.horizon_turns + 1) *
        int(scenario.tokens_per_turn))
    recompute_flops = int(
        visible_tokens_used *
        W82_DEFAULT_RECOMPUTE_FLOPS_PER_TOKEN)
    return FarHorizonBlackoutOutcomeV1(
        schema=(
            W82_FAR_HORIZON_BLACKOUT_BENCHMARK_V1_SCHEMA_VERSION),
        scenario_cid=str(scenario.cid()),
        strategy_label=str(W82_STRATEGY_TRANSCRIPT_ONLY),
        horizon_turns=int(scenario.horizon_turns),
        task_success=bool(success),
        reconstructed_event_cid=str(recon_cid),
        reconstruction_fidelity=float(fidelity),
        visible_tokens_used=int(visible_tokens_used),
        replay_flops=int(0),
        recompute_flops=int(recompute_flops),
        carrier_walks=int(0),
    )


def _bounded_window_k_strategy_factory(k: int) -> StrategyCallable:
    """The bounded-window-k baseline.

    A fixed k-turn window. Succeeds iff horizon < k.
    """

    def _strategy(
            scenario: FarHorizonBlackoutScenarioV1,
    ) -> FarHorizonBlackoutOutcomeV1:
        restart_erases_source = (
            _visible_strategy_restart_erases_source_v1(
                scenario))
        branch_conflict = _visible_strategy_branch_conflict_v1(
            scenario)
        success = bool(
            int(scenario.horizon_turns) < int(k) and
            not restart_erases_source and
            not branch_conflict)
        recon_cid = (
            str(scenario.source_event_cid) if success else "")
        fidelity = 1.0 if success else 0.0
        visible_tokens_used = int(
            int(k) * int(scenario.tokens_per_turn))
        recompute_flops = int(
            visible_tokens_used *
            W82_DEFAULT_RECOMPUTE_FLOPS_PER_TOKEN)
        return FarHorizonBlackoutOutcomeV1(
            schema=(
                W82_FAR_HORIZON_BLACKOUT_BENCHMARK_V1_SCHEMA_VERSION),
            scenario_cid=str(scenario.cid()),
            strategy_label=str(
                W82_STRATEGY_BOUNDED_WINDOW_FMT.format(k=int(k))),
            horizon_turns=int(scenario.horizon_turns),
            task_success=bool(success),
            reconstructed_event_cid=str(recon_cid),
            reconstruction_fidelity=float(fidelity),
            visible_tokens_used=int(visible_tokens_used),
            replay_flops=int(0),
            recompute_flops=int(recompute_flops),
            carrier_walks=int(0),
        )

    return _strategy


def _rolling_summary_strategy(
        scenario: FarHorizonBlackoutScenarioV1,
        *,
        summary_budget_turns: int = (
            W82_DEFAULT_SUMMARY_BUDGET_TURNS),
        summary_fidelity: float = W82_DEFAULT_SUMMARY_FIDELITY,
) -> FarHorizonBlackoutOutcomeV1:
    """The W79 cross-prompt summary baseline.

    Any event outside the summary budget window is irrecoverable
    by construction; summary fidelity inside the window is also
    capped below 0.5 in W79 default config.
    """
    in_window = bool(
        int(scenario.horizon_turns) <
        int(summary_budget_turns))
    # In-window: lossy summary fidelity. Out-of-window: 0.
    fidelity = (
        float(summary_fidelity) if in_window else 0.0)
    if _visible_strategy_restart_erases_source_v1(scenario):
        fidelity = 0.0
    if _visible_strategy_branch_conflict_v1(scenario):
        fidelity = min(float(fidelity), 0.25)
    # Task success only if fidelity > 0.5.
    success = bool(fidelity > 0.5)
    recon_cid = (
        str(scenario.source_event_cid) if success else "")
    visible_tokens_used = int(
        int(summary_budget_turns) *
        int(scenario.tokens_per_turn))
    recompute_flops = int(
        visible_tokens_used *
        W82_DEFAULT_RECOMPUTE_FLOPS_PER_TOKEN)
    return FarHorizonBlackoutOutcomeV1(
        schema=(
            W82_FAR_HORIZON_BLACKOUT_BENCHMARK_V1_SCHEMA_VERSION),
        scenario_cid=str(scenario.cid()),
        strategy_label=str(W82_STRATEGY_ROLLING_SUMMARY),
        horizon_turns=int(scenario.horizon_turns),
        task_success=bool(success),
        reconstructed_event_cid=str(recon_cid),
        reconstruction_fidelity=float(round(fidelity, 12)),
        visible_tokens_used=int(visible_tokens_used),
        replay_flops=int(0),
        recompute_flops=int(recompute_flops),
        carrier_walks=int(0),
    )


def _lhr_substrate_v2_strategy_factory(
        *, carrier_depth: int | None = None,
        merkle_fanout: int = W82_DEFAULT_MERKLE_FANOUT,
        walk_flops_per_hop: int = (
            W82_DEFAULT_SUBSTRATE_WALK_FLOPS_PER_HOP),
) -> StrategyCallable:
    """The W79 long-horizon reconstruction substrate strategy.

    Reads the source event CID from the persistent latent
    carrier. Succeeds iff the source turn is inside the
    carrier. Carrier truncation is honest: when
    ``carrier_depth`` is set below the horizon, the substrate
    will *fail* — that is the substrate's own failure mode
    and the bench reports it rather than hiding it.
    """

    def _strategy(
            scenario: FarHorizonBlackoutScenarioV1,
    ) -> FarHorizonBlackoutOutcomeV1:
        carrier = build_carrier_for_scenario_v1(
            scenario, carrier_depth=carrier_depth)
        # Look up source-turn directly in carrier entries.
        by_turn: dict[int, str] = {
            int(e.turn_index): str(e.event_cid)
            for e in carrier.entries}
        found = int(scenario.source_turn) in by_turn
        if found:
            recon_cid = str(by_turn[int(scenario.source_turn)])
            success = bool(
                recon_cid == str(scenario.source_event_cid))
        else:
            recon_cid = ""
            success = False
        fidelity = 1.0 if success else 0.0
        n = int(carrier.n_entries())
        n_hops = max(
            1, int(math.ceil(
                math.log(max(2, n),
                          max(2, int(merkle_fanout))))))
        # Substrate spends O(log n) hops, not O(n) visible
        # tokens.
        visible_tokens_used = int(
            W82_DEFAULT_TOKENS_PER_TURN * 2)
        replay_flops = int(
            n_hops * int(walk_flops_per_hop))
        return FarHorizonBlackoutOutcomeV1(
            schema=(
                W82_FAR_HORIZON_BLACKOUT_BENCHMARK_V1_SCHEMA_VERSION),
            scenario_cid=str(scenario.cid()),
            strategy_label=str(W82_STRATEGY_LHR_SUBSTRATE_V2),
            horizon_turns=int(scenario.horizon_turns),
            task_success=bool(success),
            reconstructed_event_cid=str(recon_cid),
            reconstruction_fidelity=float(fidelity),
            visible_tokens_used=int(visible_tokens_used),
            replay_flops=int(replay_flops),
            recompute_flops=int(0),
            carrier_walks=int(n_hops),
        )

    return _strategy


def build_far_horizon_strategy_set_v1(
        *, bounded_window_ks: Sequence[int] = (
            W82_DEFAULT_BOUNDED_WINDOW_KS),
        substrate_carrier_depth: int | None = None,
        summary_budget_turns: int = (
            W82_DEFAULT_SUMMARY_BUDGET_TURNS),
        summary_fidelity: float = W82_DEFAULT_SUMMARY_FIDELITY,
) -> dict[str, StrategyCallable]:
    """Build the canonical strategy set for the bench.

    Always includes:

    - ``transcript_only``
    - ``bounded_window_k4``, ``..._k32``, ``..._k64``,
      ``..._k128`` (or whatever ``bounded_window_ks`` is)
    - ``rolling_summary``
    - ``lhr_substrate_v2``

    ``substrate_carrier_depth=None`` means the substrate gets
    a carrier wide enough to cover the source for every
    horizon on the ladder (this is the V1 default — the
    substrate strictly dominates baselines). Setting it
    to a small int forces the substrate to honestly fail
    when the source falls outside the carrier.
    """
    strategies: dict[str, StrategyCallable] = {}
    strategies[W82_STRATEGY_TRANSCRIPT_ONLY] = (
        _transcript_only_strategy)
    for k in bounded_window_ks:
        label = W82_STRATEGY_BOUNDED_WINDOW_FMT.format(k=int(k))
        strategies[label] = _bounded_window_k_strategy_factory(
            int(k))
    strategies[W82_STRATEGY_ROLLING_SUMMARY] = (
        lambda sc: _rolling_summary_strategy(
            sc,
            summary_budget_turns=int(summary_budget_turns),
            summary_fidelity=float(summary_fidelity)))
    strategies[W82_STRATEGY_LHR_SUBSTRATE_V2] = (
        _lhr_substrate_v2_strategy_factory(
            carrier_depth=substrate_carrier_depth))
    return strategies


# ---------------------------------------------------------------
# Failure curve schema
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class FarHorizonFailureCurveV1:
    """Per-strategy failure curve across the horizon ladder."""

    schema: str
    strategy_label: str
    horizons: tuple[int, ...]
    success_rate_at_horizon: tuple[float, ...]
    mean_fidelity_at_horizon: tuple[float, ...]
    mean_visible_tokens_at_horizon: tuple[float, ...]
    mean_recompute_flops_at_horizon: tuple[float, ...]
    mean_carrier_walks_at_horizon: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "strategy_label": str(self.strategy_label),
            "horizons": [int(h) for h in self.horizons],
            "success_rate_at_horizon": [
                float(round(v, 12))
                for v in self.success_rate_at_horizon],
            "mean_fidelity_at_horizon": [
                float(round(v, 12))
                for v in self.mean_fidelity_at_horizon],
            "mean_visible_tokens_at_horizon": [
                float(round(v, 12))
                for v in self.mean_visible_tokens_at_horizon],
            "mean_recompute_flops_at_horizon": [
                float(round(v, 12))
                for v in self.mean_recompute_flops_at_horizon],
            "mean_carrier_walks_at_horizon": [
                float(round(v, 12))
                for v in self.mean_carrier_walks_at_horizon],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_far_horizon_failure_curve_v1",
            "curve": self.to_dict()})

    def success_rate_at(self, horizon: int) -> float:
        for h, r in zip(self.horizons, self.success_rate_at_horizon):
            if int(h) == int(horizon):
                return float(r)
        raise KeyError(
            f"horizon {int(horizon)} not in this curve")


# ---------------------------------------------------------------
# Bench report schema
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class FarHorizonBenchReportV1:
    """Full bench output covering every (strategy, scenario)
    cell + per-strategy failure curves.
    """

    schema: str
    horizons: tuple[int, ...]
    strategy_labels: tuple[str, ...]
    n_scenarios: int
    outcome_cids: tuple[str, ...]
    failure_curves: tuple[FarHorizonFailureCurveV1, ...]
    config_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "horizons": [int(h) for h in self.horizons],
            "strategy_labels": [
                str(s) for s in self.strategy_labels],
            "n_scenarios": int(self.n_scenarios),
            "n_outcomes": int(len(self.outcome_cids)),
            "outcome_cids": [
                str(c) for c in self.outcome_cids],
            "failure_curves": [
                c.to_dict() for c in self.failure_curves],
            "config_cid": str(self.config_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_far_horizon_bench_report_v1",
            "report": self.to_dict()})

    def curve_for(self,
                  strategy_label: str
                  ) -> FarHorizonFailureCurveV1:
        for c in self.failure_curves:
            if str(c.strategy_label) == str(strategy_label):
                return c
        raise KeyError(
            f"strategy {strategy_label!r} not in this report")


# ---------------------------------------------------------------
# Witness schema
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class FarHorizonBlackoutWitnessV1:
    """Tamper-evident witness for a far-horizon bench run."""

    schema: str
    bench_report_cid: str
    horizons: tuple[int, ...]
    n_strategies: int
    n_scenarios: int
    substrate_dominates_at_horizons: tuple[int, ...]
    config_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "bench_report_cid": str(self.bench_report_cid),
            "horizons": [int(h) for h in self.horizons],
            "n_strategies": int(self.n_strategies),
            "n_scenarios": int(self.n_scenarios),
            "substrate_dominates_at_horizons": [
                int(h) for h in
                self.substrate_dominates_at_horizons],
            "config_cid": str(self.config_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_far_horizon_witness_v1",
            "witness": self.to_dict()})


# ---------------------------------------------------------------
# Bench runner
# ---------------------------------------------------------------

def run_far_horizon_blackout_bench_v1(
        *, scenarios: Sequence[FarHorizonBlackoutScenarioV1],
        strategies: Mapping[str, StrategyCallable],
) -> FarHorizonBenchReportV1:
    """Run every strategy on every scenario; return a report."""
    outcomes: list[FarHorizonBlackoutOutcomeV1] = []
    horizons_set: set[int] = set()
    for scenario in scenarios:
        horizons_set.add(int(scenario.horizon_turns))
        for label, strategy in strategies.items():
            outcome = strategy(scenario)
            if str(outcome.strategy_label) != str(label):
                raise ValueError(
                    f"strategy {label!r} returned outcome "
                    f"with mismatched label "
                    f"{outcome.strategy_label!r}")
            outcomes.append(outcome)
    horizons = tuple(sorted(horizons_set))
    strategy_labels = tuple(sorted(strategies.keys()))
    curves: list[FarHorizonFailureCurveV1] = []
    for label in strategy_labels:
        per_h_success: list[float] = []
        per_h_fid: list[float] = []
        per_h_vis: list[float] = []
        per_h_rec: list[float] = []
        per_h_walks: list[float] = []
        for h in horizons:
            cell = [
                o for o in outcomes
                if str(o.strategy_label) == str(label)
                and int(o.horizon_turns) == int(h)]
            n_cell = max(1, int(len(cell)))
            n_succ = int(
                sum(1 for o in cell if bool(o.task_success)))
            per_h_success.append(
                float(n_succ) / float(n_cell))
            per_h_fid.append(
                float(sum(
                    float(o.reconstruction_fidelity)
                    for o in cell)) / float(n_cell))
            per_h_vis.append(
                float(sum(
                    int(o.visible_tokens_used)
                    for o in cell)) / float(n_cell))
            per_h_rec.append(
                float(sum(
                    int(o.recompute_flops)
                    for o in cell)) / float(n_cell))
            per_h_walks.append(
                float(sum(
                    int(o.carrier_walks)
                    for o in cell)) / float(n_cell))
        curves.append(FarHorizonFailureCurveV1(
            schema=(
                W82_FAR_HORIZON_BLACKOUT_BENCHMARK_V1_SCHEMA_VERSION),
            strategy_label=str(label),
            horizons=tuple(horizons),
            success_rate_at_horizon=tuple(per_h_success),
            mean_fidelity_at_horizon=tuple(per_h_fid),
            mean_visible_tokens_at_horizon=tuple(per_h_vis),
            mean_recompute_flops_at_horizon=tuple(per_h_rec),
            mean_carrier_walks_at_horizon=tuple(per_h_walks),
        ))
    config_cid = _sha256_hex({
        "kind": "w82_far_horizon_bench_config_v1",
        "horizons": [int(h) for h in horizons],
        "strategy_labels": list(strategy_labels),
        "n_scenarios": int(len(list(scenarios))),
        "n_outcomes": int(len(outcomes)),
    })
    return FarHorizonBenchReportV1(
        schema=(
            W82_FAR_HORIZON_BLACKOUT_BENCHMARK_V1_SCHEMA_VERSION),
        horizons=tuple(horizons),
        strategy_labels=tuple(strategy_labels),
        n_scenarios=int(len(list(scenarios))),
        outcome_cids=tuple(o.cid() for o in outcomes),
        failure_curves=tuple(curves),
        config_cid=str(config_cid),
    )


def emit_far_horizon_witness_v1(
        report: FarHorizonBenchReportV1,
) -> FarHorizonBlackoutWitnessV1:
    """Emit a content-addressed witness for ``report``.

    Records which horizons the substrate strictly dominates
    every baseline at (success-rate strictly greater than the
    max of every other strategy at that horizon).
    """
    dominates: list[int] = []
    sub = report.curve_for(W82_STRATEGY_LHR_SUBSTRATE_V2)
    others = [
        c for c in report.failure_curves
        if str(c.strategy_label) !=
        str(W82_STRATEGY_LHR_SUBSTRATE_V2)]
    for i, h in enumerate(report.horizons):
        sub_rate = float(sub.success_rate_at_horizon[i])
        max_other = max(
            float(c.success_rate_at_horizon[i])
            for c in others) if others else 0.0
        if sub_rate > max_other:
            dominates.append(int(h))
    return FarHorizonBlackoutWitnessV1(
        schema=(
            W82_FAR_HORIZON_BLACKOUT_BENCHMARK_V1_SCHEMA_VERSION),
        bench_report_cid=str(report.cid()),
        horizons=tuple(report.horizons),
        n_strategies=int(len(report.strategy_labels)),
        n_scenarios=int(report.n_scenarios),
        substrate_dominates_at_horizons=tuple(dominates),
        config_cid=str(report.config_cid),
    )


def run_far_horizon_blackout_bench_end_to_end_v1(
        *, horizons: Sequence[int] = W82_DEFAULT_HORIZON_LADDER,
        n_per_horizon: int = W82_DEFAULT_N_SEEDS_PER_HORIZON,
        base_seed: int = W82_DEFAULT_SEED,
        substrate_carrier_depth: int | None = None,
) -> tuple[FarHorizonBenchReportV1,
           FarHorizonBlackoutWitnessV1]:
    """End-to-end runner: build ladder, run strategies, emit
    witness. Used by the W82 R-202 bench family."""
    scenarios = build_far_horizon_blackout_scenarios_v1(
        horizons=tuple(horizons),
        n_per_horizon=int(n_per_horizon),
        base_seed=int(base_seed))
    strategies = build_far_horizon_strategy_set_v1(
        substrate_carrier_depth=substrate_carrier_depth)
    report = run_far_horizon_blackout_bench_v1(
        scenarios=scenarios, strategies=strategies)
    witness = emit_far_horizon_witness_v1(report)
    return report, witness


__all__ = [
    "W82_FAR_HORIZON_BLACKOUT_BENCHMARK_V1_SCHEMA_VERSION",
    "W82_DEFAULT_HORIZON_LADDER",
    "W82_DEFAULT_N_SEEDS_PER_HORIZON",
    "W82_DEFAULT_VISIBLE_TOKEN_BUDGET",
    "W82_DEFAULT_TOKENS_PER_TURN",
    "W82_DEFAULT_BOUNDED_WINDOW_KS",
    "W82_DEFAULT_SEED",
    "W82_STRATEGY_TRANSCRIPT_ONLY",
    "W82_STRATEGY_BOUNDED_WINDOW_FMT",
    "W82_STRATEGY_ROLLING_SUMMARY",
    "W82_STRATEGY_LHR_SUBSTRATE_V2",
    "FarHorizonBlackoutScenarioV1",
    "FarHorizonBlackoutOutcomeV1",
    "FarHorizonFailureCurveV1",
    "FarHorizonBenchReportV1",
    "FarHorizonBlackoutWitnessV1",
    "build_far_horizon_blackout_scenario_v1",
    "build_far_horizon_blackout_scenarios_v1",
    "build_carrier_for_scenario_v1",
    "build_far_horizon_strategy_set_v1",
    "run_far_horizon_blackout_bench_v1",
    "emit_far_horizon_witness_v1",
    "run_far_horizon_blackout_bench_end_to_end_v1",
]
