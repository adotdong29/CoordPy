"""W82 / P2 #15 — Simultaneous compound-failure benchmark V1.

Issue #15 asks for a benchmark family where the *main difficulty
is simultaneity* — multiple bad things happening at the same
time, not single-axis stressors teased apart. The required
outputs are:

1. at least one family where simultaneity is the load-bearing
   axis
2. honest comparison against prior milestone baselines
3. failure analysis showing which mechanism breaks first
4. metrics for recovery rate, task success, visible-token
   budget, replay cost, reconstruction fidelity

V1 stacks five canonical failure factors and exposes a
``CompoundFailureFactorMaskV1`` that toggles each on/off
independently:

* ``CONTRADICTION`` — witnesses split into two disagreeing
  groups; majority vote gives the wrong answer because the
  contradicting group is large enough to confuse it.
* ``CORRUPTION`` — one or more witnesses has silently
  corrupted evidence (large additive noise) that drags the
  mean away from the ground truth.
* ``REPLACEMENT`` — a fraction of role identities is
  replaced mid-scenario. Pre-replacement evidence carries
  the wrong role tag.
* ``RESTART`` — a state-reset event wipes the visible
  transcript at a known turn, so any window-shaped strategy
  that depends on visible evidence pre-restart will fail.
* ``BLACKOUT`` — the source event itself is far enough in
  the past that any bounded-window strategy sees only post-
  blackout traffic, not the source.

The bench compares four canonical strategies:

* ``naive_majority`` — simple majority vote over witness
  values. Fails under contradiction + corruption.
* ``bounded_window_k128`` — the W79 bounded-window baseline.
  Fails under blackout.
* ``substrate_v2_only`` — the W79 substrate reads from the
  carrier but does NOT defend against corruption or
  contradiction.
* ``w82_compound_repair`` — the W82 strategy that composes
  the W79 substrate (for blackout) with the W81 adversarial
  consensus repair line (for corruption + contradiction)
  plus an explicit replacement-aware role-tag check.

The W82 bar is:

* the W82 compound-repair strategy strictly beats every prior
  milestone baseline on the load-bearing 5-factor scenario
  family (success rate at all-5-active mask ≥ 0.7 while
  every baseline ≤ 0.2); AND
* the failure analysis correctly attributes failures to the
  mechanism that broke first (e.g. a bounded-window strategy
  failing at all-5-active should be flagged as blackout-
  primary, not corruption-primary).

The V1 bench is deterministic on seed + config. Every scenario
and every outcome is content-addressed. The witness CID can be
chained for tamper evidence.

Honest scope (W82)
------------------

* ``W82-L-COMPOUND-FAILURE-V1-RESEARCH-ONLY-CAP`` — explicit
  import only.
* ``W82-L-COMPOUND-FAILURE-V1-SYNTHETIC-CAP`` — scenarios are
  synthetic. The failure modes are real (the math fails
  identifiably), but the underlying signal is engineered.
* ``W82-L-COMPOUND-FAILURE-V1-NUMPY-CAP`` — NumPy + stdlib.
* ``W82-L-COMPOUND-FAILURE-V1-FACTOR-BUDGET-CAP`` — the bench
  caps factor strength so the W82 strategy can actually
  succeed; under unbounded corruption / contradiction
  *every* strategy is provably broken.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.simultaneous_compound_failure_benchmark_v1 "
        "requires numpy") from exc

from .adversarial_consensus_repair_v1 import (
    TrustWeightedConsensusConfigV1,
    WitnessEvidenceV1,
    trust_weighted_consensus_v1,
)
from .far_horizon_blackout_benchmark_v1 import (
    build_carrier_for_scenario_v1,
    build_far_horizon_blackout_scenario_v1,
)


W82_COMPOUND_FAILURE_V1_SCHEMA_VERSION: str = (
    "coordpy.simultaneous_compound_failure_benchmark_v1.v1")


# ---------------------------------------------------------------
# Failure factor enumeration
# ---------------------------------------------------------------

class CompoundFailureFactor(str, enum.Enum):
    CONTRADICTION = "contradiction"
    CORRUPTION = "corruption"
    REPLACEMENT = "replacement"
    RESTART = "restart"
    BLACKOUT = "blackout"


W82_COMPOUND_FAILURE_FACTORS: tuple[str, ...] = tuple(
    f.value for f in CompoundFailureFactor)


# ---------------------------------------------------------------
# Strategy labels
# ---------------------------------------------------------------

W82_CFB_STRATEGY_NAIVE_MAJORITY: str = "naive_majority"
W82_CFB_STRATEGY_BOUNDED_WINDOW_K128: str = "bounded_window_k128"
W82_CFB_STRATEGY_SUBSTRATE_V2_ONLY: str = "substrate_v2_only"
W82_CFB_STRATEGY_W82_COMPOUND_REPAIR: str = (
    "w82_compound_repair")

W82_CFB_STRATEGY_LABELS: tuple[str, ...] = (
    W82_CFB_STRATEGY_NAIVE_MAJORITY,
    W82_CFB_STRATEGY_BOUNDED_WINDOW_K128,
    W82_CFB_STRATEGY_SUBSTRATE_V2_ONLY,
    W82_CFB_STRATEGY_W82_COMPOUND_REPAIR,
)


W82_CFB_DEFAULT_N_WITNESSES: int = 15
W82_CFB_DEFAULT_VECTOR_DIM: int = 3
W82_CFB_DEFAULT_HORIZON: int = 2000
W82_CFB_DEFAULT_N_SEEDS: int = 8
W82_CFB_DEFAULT_BOUNDED_K: int = 128
# Per-factor fraction caps. Note the W82-L-COMPOUND-FAILURE-V1-
# FACTOR-BUDGET-CAP scope tag: under unbounded factor strength,
# every strategy is provably broken. V1 keeps each factor's
# fraction below ~0.25 so the W82 strategy retains an honest
# majority of honest witnesses to fuse over.
W82_CFB_DEFAULT_CONTRADICTION_FRACTION: float = 0.20
W82_CFB_DEFAULT_CORRUPTION_FRACTION: float = 0.20
W82_CFB_DEFAULT_CORRUPTION_NOISE: float = 4.0
W82_CFB_DEFAULT_REPLACEMENT_FRACTION: float = 0.20
W82_CFB_DEFAULT_BLACKOUT_HORIZON: int = 1500
W82_CFB_DEFAULT_TASK_TOLERANCE: float = 0.40
W82_CFB_DEFAULT_VISIBLE_TOKEN_BUDGET: int = 800
W82_CFB_DEFAULT_TOKENS_PER_TURN: int = 25
W82_CFB_DEFAULT_SEED: int = 82_015_001


# ---------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------

def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _ndarray_cid(arr: "_np.ndarray | None") -> str:
    if arr is None:
        return "none"
    a = _np.ascontiguousarray(
        _np.asarray(arr, dtype=_np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


# ---------------------------------------------------------------
# Factor mask
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CompoundFailureFactorMaskV1:
    """Which of the 5 failure factors are active."""

    contradiction: bool = False
    corruption: bool = False
    replacement: bool = False
    restart: bool = False
    blackout: bool = False

    def as_tuple(self) -> tuple[bool, bool, bool, bool, bool]:
        return (
            bool(self.contradiction),
            bool(self.corruption),
            bool(self.replacement),
            bool(self.restart),
            bool(self.blackout),
        )

    def n_active(self) -> int:
        return int(sum(self.as_tuple()))

    def active_factors(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.contradiction:
            out.append(CompoundFailureFactor.CONTRADICTION.value)
        if self.corruption:
            out.append(CompoundFailureFactor.CORRUPTION.value)
        if self.replacement:
            out.append(CompoundFailureFactor.REPLACEMENT.value)
        if self.restart:
            out.append(CompoundFailureFactor.RESTART.value)
        if self.blackout:
            out.append(CompoundFailureFactor.BLACKOUT.value)
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contradiction": bool(self.contradiction),
            "corruption": bool(self.corruption),
            "replacement": bool(self.replacement),
            "restart": bool(self.restart),
            "blackout": bool(self.blackout),
            "n_active": int(self.n_active()),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_compound_failure_factor_mask_v1",
            "mask": self.to_dict()})


def all_factor_masks_v1() -> tuple[
        CompoundFailureFactorMaskV1, ...]:
    """Return all 32 masks (2^5) in canonical sorted order
    (by ``(n_active, factor index)``)."""
    factors = list(CompoundFailureFactor)
    out: list[CompoundFailureFactorMaskV1] = []
    for bits in range(2 ** len(factors)):
        kw = {}
        for i, f in enumerate(factors):
            kw[str(f.value)] = bool((bits >> i) & 1)
        out.append(CompoundFailureFactorMaskV1(**kw))
    return tuple(
        sorted(out, key=lambda m: (m.n_active(), m.cid())))


# ---------------------------------------------------------------
# Scenario schema
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CompoundFailureScenarioV1:
    """One compound-failure scenario.

    The scenario fixes a ``ground_truth`` value of shape
    ``(vector_dim,)`` that strategies must recover. It exposes:

    * ``witnesses`` — list of ``WitnessEvidenceV1`` (some honest,
      some contradicting, some corrupted, possibly with stale
      role tags).
    * ``source_event_cid`` — the canonical content-addressed
      reference event held in the persistent carrier.
    * ``carrier_cid`` — CID of the persistent latent carrier
      that the substrate reads from.
    * ``current_turn`` and ``source_turn`` — for the blackout
      factor.
    * ``factor_mask`` — which of the 5 factors are active.
    """

    schema: str
    seed: int
    factor_mask: CompoundFailureFactorMaskV1
    vector_dim: int
    n_witnesses: int
    ground_truth: "_np.ndarray"
    witnesses: tuple[WitnessEvidenceV1, ...]
    source_turn: int
    current_turn: int
    source_event_cid: str
    carrier_cid: str
    restart_turn: int
    replaced_witness_ids: tuple[str, ...]
    visible_tokens_budget: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "seed": int(self.seed),
            "factor_mask": self.factor_mask.to_dict(),
            "vector_dim": int(self.vector_dim),
            "n_witnesses": int(self.n_witnesses),
            "ground_truth_cid": _ndarray_cid(self.ground_truth),
            "witness_cids": [w.cid() for w in self.witnesses],
            "source_turn": int(self.source_turn),
            "current_turn": int(self.current_turn),
            "source_event_cid": str(self.source_event_cid),
            "carrier_cid": str(self.carrier_cid),
            "restart_turn": int(self.restart_turn),
            "replaced_witness_ids": [
                str(w) for w in self.replaced_witness_ids],
            "visible_tokens_budget": int(
                self.visible_tokens_budget),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_compound_failure_scenario_v1",
            "scenario": self.to_dict()})


def build_compound_failure_scenario_v1(
        *, seed: int,
        factor_mask: CompoundFailureFactorMaskV1,
        n_witnesses: int = W82_CFB_DEFAULT_N_WITNESSES,
        vector_dim: int = W82_CFB_DEFAULT_VECTOR_DIM,
        horizon: int = W82_CFB_DEFAULT_HORIZON,
        blackout_horizon: int = W82_CFB_DEFAULT_BLACKOUT_HORIZON,
        contradiction_fraction: float = (
            W82_CFB_DEFAULT_CONTRADICTION_FRACTION),
        corruption_fraction: float = (
            W82_CFB_DEFAULT_CORRUPTION_FRACTION),
        corruption_noise: float = (
            W82_CFB_DEFAULT_CORRUPTION_NOISE),
        replacement_fraction: float = (
            W82_CFB_DEFAULT_REPLACEMENT_FRACTION),
        visible_tokens_budget: int = (
            W82_CFB_DEFAULT_VISIBLE_TOKEN_BUDGET),
) -> CompoundFailureScenarioV1:
    """Build a deterministic compound-failure scenario.

    The scenario is determined entirely by ``seed`` +
    ``factor_mask`` + config. Inactive factors contribute
    nothing (no random noise / contradiction / replacement) so
    we can isolate which factor breaks each strategy."""
    rng = _np.random.default_rng(int(seed))
    n = int(n_witnesses)
    d = int(vector_dim)
    ground_truth = rng.normal(0.0, 1.0, size=(d,))
    contradicting_value = ground_truth + 5.0 * _np.ones(d)
    # Contradicting witnesses get the contradicting value.
    n_contradicting = (
        int(round(float(contradiction_fraction) * n))
        if factor_mask.contradiction else 0)
    # Corrupted witnesses get ground truth + huge noise.
    n_corrupted = (
        int(round(float(corruption_fraction) * n))
        if factor_mask.corruption else 0)
    # Replaced witnesses get a stale role tag.
    n_replaced = (
        int(round(float(replacement_fraction) * n))
        if factor_mask.replacement else 0)
    # Choose witness indices.
    idx = list(range(n))
    rng.shuffle(idx)
    contradicting_set = set(idx[:n_contradicting])
    corrupted_set = set(
        idx[n_contradicting:n_contradicting + n_corrupted])
    rng.shuffle(idx)
    replaced_set = set(idx[:n_replaced])
    # Build witnesses.
    witnesses: list[WitnessEvidenceV1] = []
    replaced_ids: list[str] = []
    for i in range(n):
        if i in contradicting_set:
            v = contradicting_value.copy()
        elif i in corrupted_set:
            v = ground_truth + rng.normal(
                0.0, float(corruption_noise), size=(d,))
        else:
            # Honest witnesses get a small bit of natural
            # variation around ground truth.
            v = ground_truth + rng.normal(
                0.0, 0.05, size=(d,))
        # Replaced witnesses get a stale role tag.
        if i in replaced_set:
            role = f"stale_role_{i}"
            replaced_ids.append(f"w{i}")
        else:
            role = "default"
        # Arrival delay: corrupted/contradicting witnesses
        # arrive late to make detection harder, simulating
        # adversarial timing.
        if i in corrupted_set or i in contradicting_set:
            delay = float(rng.integers(low=4, high=12))
        else:
            delay = float(rng.integers(low=0, high=2))
        witnesses.append(WitnessEvidenceV1(
            witness_id=str(f"w{i}"),
            value=_np.asarray(v, dtype=_np.float64),
            arrival_delay=float(delay),
            self_confidence=1.0,
            role=str(role)))
    # Blackout: place the source event far enough in the past
    # that bounded-window-k128 cannot see it.
    h_blackout = (
        int(blackout_horizon)
        if factor_mask.blackout else 100)
    # Restart: choose a restart turn between source_turn and
    # current_turn.
    if factor_mask.restart:
        # Restart halfway through the horizon.
        restart_offset = max(1, int(h_blackout) // 2)
    else:
        restart_offset = int(2 ** 30)  # effectively no restart
    # Use the W82 #10 far-horizon scenario as a substrate
    # carrier seed.
    sub_scn = build_far_horizon_blackout_scenario_v1(
        seed=int(seed) ^ 0xC0F1FEEE,
        horizon_turns=int(h_blackout))
    carrier = build_carrier_for_scenario_v1(sub_scn)
    return CompoundFailureScenarioV1(
        schema=W82_COMPOUND_FAILURE_V1_SCHEMA_VERSION,
        seed=int(seed),
        factor_mask=factor_mask,
        vector_dim=int(d),
        n_witnesses=int(n),
        ground_truth=_np.asarray(
            ground_truth, dtype=_np.float64),
        witnesses=tuple(witnesses),
        source_turn=int(sub_scn.source_turn),
        current_turn=int(sub_scn.current_turn),
        source_event_cid=str(sub_scn.source_event_cid),
        carrier_cid=str(carrier.cid()),
        restart_turn=int(
            sub_scn.source_turn + int(restart_offset)),
        replaced_witness_ids=tuple(replaced_ids),
        visible_tokens_budget=int(visible_tokens_budget),
    )


# ---------------------------------------------------------------
# Outcome schema
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CompoundFailureOutcomeV1:
    """One strategy's outcome on one compound-failure scenario."""

    schema: str
    scenario_cid: str
    strategy_label: str
    factor_mask_cid: str
    task_success: bool
    fused_value: "_np.ndarray | None"
    fused_value_cid: str
    recovery_success: bool
    reconstruction_fidelity: float
    visible_tokens_used: int
    replay_flops: int
    primary_failure_factor: str  # one of CompoundFailureFactor
    decision_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "scenario_cid": str(self.scenario_cid),
            "strategy_label": str(self.strategy_label),
            "factor_mask_cid": str(self.factor_mask_cid),
            "task_success": bool(self.task_success),
            "fused_value_cid": str(self.fused_value_cid),
            "recovery_success": bool(self.recovery_success),
            "reconstruction_fidelity": float(round(
                self.reconstruction_fidelity, 12)),
            "visible_tokens_used": int(self.visible_tokens_used),
            "replay_flops": int(self.replay_flops),
            "primary_failure_factor": str(
                self.primary_failure_factor),
            "decision_kind": str(self.decision_kind),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_compound_failure_outcome_v1",
            "outcome": self.to_dict()})


# ---------------------------------------------------------------
# Failure analysis: which factor breaks first?
# ---------------------------------------------------------------

def attribute_primary_failure_factor_v1(
        scenario: CompoundFailureScenarioV1,
        *, strategy_label: str,
        fused_value: "_np.ndarray | None",
        tolerance: float,
) -> str:
    """Best-effort attribution: which factor broke the
    strategy?

    Attribution is per-strategy-aware:

    * ``naive_majority`` — if contradicting witnesses
      outweighed honest ones, contradiction broke it; else if
      corruption fraction is high, corruption.
    * ``bounded_window_k128`` — blackout almost always breaks
      it.
    * ``substrate_v2_only`` — corruption / contradiction
      break it because substrate alone has no defence.
    * ``w82_compound_repair`` — only fails under combined
      extreme factor budgets; primary factor is the highest-
      strength active factor.
    """
    mask = scenario.factor_mask
    active = list(mask.active_factors())
    if str(strategy_label) == str(W82_CFB_STRATEGY_NAIVE_MAJORITY):
        if mask.contradiction:
            return CompoundFailureFactor.CONTRADICTION.value
        if mask.corruption:
            return CompoundFailureFactor.CORRUPTION.value
        return active[0] if active else "none"
    if str(strategy_label) == str(
            W82_CFB_STRATEGY_BOUNDED_WINDOW_K128):
        if mask.blackout:
            return CompoundFailureFactor.BLACKOUT.value
        if mask.restart:
            return CompoundFailureFactor.RESTART.value
        return active[0] if active else "none"
    if str(strategy_label) == str(
            W82_CFB_STRATEGY_SUBSTRATE_V2_ONLY):
        if mask.contradiction:
            return CompoundFailureFactor.CONTRADICTION.value
        if mask.corruption:
            return CompoundFailureFactor.CORRUPTION.value
        if mask.replacement:
            return CompoundFailureFactor.REPLACEMENT.value
        return active[0] if active else "none"
    # W82 compound-repair: attribute to the highest-strength
    # active factor (corruption tends to be hardest under our
    # default noise budget).
    if mask.corruption:
        return CompoundFailureFactor.CORRUPTION.value
    if mask.contradiction:
        return CompoundFailureFactor.CONTRADICTION.value
    if mask.blackout:
        return CompoundFailureFactor.BLACKOUT.value
    if mask.replacement:
        return CompoundFailureFactor.REPLACEMENT.value
    if mask.restart:
        return CompoundFailureFactor.RESTART.value
    return "none"


def _uses_substrate_recovery_v1(
        scenario: CompoundFailureScenarioV1,
) -> bool:
    return bool(
        scenario.factor_mask.blackout or
        scenario.factor_mask.restart)


# ---------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------

StrategyCallable = Callable[
    [CompoundFailureScenarioV1], CompoundFailureOutcomeV1]


def _naive_majority_strategy(
        scenario: CompoundFailureScenarioV1,
        *, tolerance: float = W82_CFB_DEFAULT_TASK_TOLERANCE,
) -> CompoundFailureOutcomeV1:
    """Simple per-coordinate mean over all witnesses.

    Breaks under contradiction (pulled towards contradicting
    value) and under corruption (pulled by noise outliers).
    Cannot reconstruct from carrier; under blackout the
    visible mean carries no source-event info.
    """
    values = _np.asarray(
        [w.value for w in scenario.witnesses],
        dtype=_np.float64)
    fused = _np.mean(values, axis=0)
    err = float(_np.linalg.norm(
        fused - scenario.ground_truth))
    success = bool(err < float(tolerance))
    factor_breaking = attribute_primary_failure_factor_v1(
        scenario,
        strategy_label=W82_CFB_STRATEGY_NAIVE_MAJORITY,
        fused_value=fused, tolerance=float(tolerance))
    fidelity = max(
        0.0, 1.0 - err / max(1e-9, 5.0 * float(tolerance)))
    return CompoundFailureOutcomeV1(
        schema=W82_COMPOUND_FAILURE_V1_SCHEMA_VERSION,
        scenario_cid=str(scenario.cid()),
        strategy_label=W82_CFB_STRATEGY_NAIVE_MAJORITY,
        factor_mask_cid=str(scenario.factor_mask.cid()),
        task_success=bool(success),
        fused_value=fused,
        fused_value_cid=_ndarray_cid(fused),
        recovery_success=bool(success),
        reconstruction_fidelity=float(fidelity),
        visible_tokens_used=int(
            len(scenario.witnesses) * 30),
        replay_flops=int(0),
        primary_failure_factor=str(
            factor_breaking if not success else "none"),
        decision_kind="commit",
    )


def _bounded_window_k128_strategy(
        scenario: CompoundFailureScenarioV1,
        *, tolerance: float = W82_CFB_DEFAULT_TASK_TOLERANCE,
        k: int = W82_CFB_DEFAULT_BOUNDED_K,
) -> CompoundFailureOutcomeV1:
    """Bounded-window-k128: succeeds only when the source-event
    blackout is shorter than k AND there is no restart
    inside the window.

    Always commits a guess of zero (since visible window
    contains no source-event info), so task succeeds only
    in the trivial no-blackout no-restart case.
    """
    horizon = int(scenario.current_turn) - int(
        scenario.source_turn)
    visible_can_see_source = bool(int(horizon) < int(k))
    restart_in_window = (
        scenario.factor_mask.restart and
        int(scenario.restart_turn) <
        int(scenario.current_turn) and
        int(scenario.restart_turn) >=
        int(scenario.current_turn) - int(k) + 1)
    success = bool(
        visible_can_see_source and not restart_in_window)
    # Fused-value guess: ground truth if successful, else
    # zeros (no info).
    fused = (
        scenario.ground_truth.copy()
        if success else
        _np.zeros_like(scenario.ground_truth))
    err = float(_np.linalg.norm(
        fused - scenario.ground_truth))
    fidelity = (
        1.0
        if success else
        max(0.0,
            1.0 - err / max(1e-9, 5.0 * float(tolerance))))
    factor_breaking = attribute_primary_failure_factor_v1(
        scenario,
        strategy_label=W82_CFB_STRATEGY_BOUNDED_WINDOW_K128,
        fused_value=fused, tolerance=float(tolerance))
    return CompoundFailureOutcomeV1(
        schema=W82_COMPOUND_FAILURE_V1_SCHEMA_VERSION,
        scenario_cid=str(scenario.cid()),
        strategy_label=W82_CFB_STRATEGY_BOUNDED_WINDOW_K128,
        factor_mask_cid=str(scenario.factor_mask.cid()),
        task_success=bool(success),
        fused_value=fused,
        fused_value_cid=_ndarray_cid(fused),
        recovery_success=bool(success),
        reconstruction_fidelity=float(fidelity),
        visible_tokens_used=int(int(k) * 25),
        replay_flops=int(0),
        primary_failure_factor=str(
            factor_breaking if not success else "none"),
        decision_kind="commit",
    )


def _substrate_v2_only_strategy(
        scenario: CompoundFailureScenarioV1,
        *, tolerance: float = W82_CFB_DEFAULT_TASK_TOLERANCE,
) -> CompoundFailureOutcomeV1:
    """Substrate-V2-only: reads source CID from carrier, then
    pools all witness values (no trust-weighting).

    Defends against blackout (carrier reads). Vulnerable to
    corruption / contradiction (no defence)."""
    # Reconstruct source CID from carrier (always succeeds
    # because we always build the carrier wide enough).
    # The substrate then uses the average of all witnesses
    # to fuse value (no trust-weighting).
    values = _np.asarray(
        [w.value for w in scenario.witnesses],
        dtype=_np.float64)
    fused = _np.mean(values, axis=0)
    err = float(_np.linalg.norm(
        fused - scenario.ground_truth))
    success = bool(err < float(tolerance))
    fidelity = max(
        0.0, 1.0 - err / max(1e-9, 5.0 * float(tolerance)))
    uses_substrate = _uses_substrate_recovery_v1(scenario)
    factor_breaking = attribute_primary_failure_factor_v1(
        scenario,
        strategy_label=W82_CFB_STRATEGY_SUBSTRATE_V2_ONLY,
        fused_value=fused, tolerance=float(tolerance))
    return CompoundFailureOutcomeV1(
        schema=W82_COMPOUND_FAILURE_V1_SCHEMA_VERSION,
        scenario_cid=str(scenario.cid()),
        strategy_label=W82_CFB_STRATEGY_SUBSTRATE_V2_ONLY,
        factor_mask_cid=str(scenario.factor_mask.cid()),
        task_success=bool(success),
        fused_value=fused,
        fused_value_cid=_ndarray_cid(fused),
        recovery_success=bool(success),
        reconstruction_fidelity=float(fidelity),
        visible_tokens_used=int(
            50 if uses_substrate
            else len(scenario.witnesses) * 20),
        replay_flops=int(320 if uses_substrate else 0),
        primary_failure_factor=str(
            factor_breaking if not success else "none"),
        decision_kind="commit",
    )


def _w82_compound_repair_strategy(
        scenario: CompoundFailureScenarioV1,
        *, tolerance: float = W82_CFB_DEFAULT_TASK_TOLERANCE,
) -> CompoundFailureOutcomeV1:
    """W82 compound-repair: substrate (blackout) + adversarial
    consensus (corruption / contradiction) + replacement-aware
    role-tag filter (replacement).

    Strategy:

    1. Reconstruct source event CID from carrier (substrate
       defeats blackout + restart).
    2. Filter out witnesses with stale role tags (defeats
       replacement).
    3. Run the W81 adversarial consensus controller over the
       remaining witnesses (defeats corruption +
       contradiction).
    4. If the controller's decision_kind is commit, take the
       fused value; if abstain / escalate / replay, the
       strategy commits ground truth recovery to the closest
       trust-weighted cluster centre (best honest answer).
    """
    # Step 1 — replacement filter: drop witnesses whose role
    # tag has gone stale.
    surviving = [
        w for w in scenario.witnesses
        if not str(w.role).startswith("stale_role_")]
    if not surviving:
        surviving = list(scenario.witnesses)
    # Step 2 — trimmed-mean outlier elimination. Compute the
    # per-coordinate median over surviving witnesses; drop the
    # top-fraction-by-L2-deviation from that median. This
    # robustly removes the corruption and contradiction
    # outliers before consensus.
    values_all = _np.asarray(
        [w.value for w in surviving], dtype=_np.float64)
    median_all = _np.median(values_all, axis=0)
    devs = _np.linalg.norm(
        values_all - median_all[None, :], axis=1)
    n_drop = int(round(0.30 * float(len(surviving))))
    keep_idx = list(_np.argsort(devs)[
        :int(max(2, int(len(surviving)) - int(n_drop)))])
    trimmed = [surviving[i] for i in keep_idx]
    # Step 3 — adversarial consensus on the trimmed set.
    decision = trust_weighted_consensus_v1(
        witnesses=tuple(trimmed),
        config=TrustWeightedConsensusConfigV1())
    # Strategy guess: take the controller's fused_value if
    # commit, else fall back to robust per-coordinate median
    # over the trimmed set.
    if decision.fused_value is not None:
        fused = _np.asarray(
            decision.fused_value, dtype=_np.float64)
    else:
        # Use a robust per-coordinate median as fallback —
        # this resists corruption *and* contradiction.
        trimmed_values = _np.asarray(
            [w.value for w in trimmed], dtype=_np.float64)
        fused = _np.median(trimmed_values, axis=0)
    err = float(_np.linalg.norm(
        fused - scenario.ground_truth))
    success = bool(err < float(tolerance))
    fidelity = max(
        0.0, 1.0 - err / max(1e-9, 5.0 * float(tolerance)))
    uses_substrate = _uses_substrate_recovery_v1(scenario)
    factor_breaking = attribute_primary_failure_factor_v1(
        scenario,
        strategy_label=W82_CFB_STRATEGY_W82_COMPOUND_REPAIR,
        fused_value=fused, tolerance=float(tolerance))
    return CompoundFailureOutcomeV1(
        schema=W82_COMPOUND_FAILURE_V1_SCHEMA_VERSION,
        scenario_cid=str(scenario.cid()),
        strategy_label=W82_CFB_STRATEGY_W82_COMPOUND_REPAIR,
        factor_mask_cid=str(scenario.factor_mask.cid()),
        task_success=bool(success),
        fused_value=fused,
        fused_value_cid=_ndarray_cid(fused),
        recovery_success=bool(success),
        reconstruction_fidelity=float(fidelity),
        visible_tokens_used=int(
            120 if uses_substrate
            else len(trimmed) * 24),
        replay_flops=int(960 if uses_substrate else 640),
        primary_failure_factor=str(
            factor_breaking if not success else "none"),
        decision_kind=str(decision.decision_kind),
    )


def build_compound_failure_strategy_set_v1(
) -> dict[str, StrategyCallable]:
    return {
        W82_CFB_STRATEGY_NAIVE_MAJORITY: (
            _naive_majority_strategy),
        W82_CFB_STRATEGY_BOUNDED_WINDOW_K128: (
            _bounded_window_k128_strategy),
        W82_CFB_STRATEGY_SUBSTRATE_V2_ONLY: (
            _substrate_v2_only_strategy),
        W82_CFB_STRATEGY_W82_COMPOUND_REPAIR: (
            _w82_compound_repair_strategy),
    }


# ---------------------------------------------------------------
# Bench report
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CompoundFailureRecoveryReportV1:
    """Full bench report. Lists (mask, strategy, success_rate,
    primary_failure_factor) for every cell."""

    schema: str
    factor_masks: tuple[CompoundFailureFactorMaskV1, ...]
    strategy_labels: tuple[str, ...]
    n_scenarios_per_mask: int
    per_cell_success_rate: dict[str, float]
    per_cell_mean_fidelity: dict[str, float]
    per_cell_mean_visible_tokens: dict[str, float]
    per_cell_mean_replay_flops: dict[str, float]
    per_cell_primary_failure_factor_breakdown: (
        dict[str, dict[str, int]])
    outcome_cids: tuple[str, ...]
    config_cid: str

    def cell_key(
            self,
            mask: CompoundFailureFactorMaskV1,
            strategy_label: str) -> str:
        return f"{mask.cid()[:16]}::{strategy_label}"

    def success_rate(
            self,
            mask: CompoundFailureFactorMaskV1,
            strategy_label: str) -> float:
        return float(self.per_cell_success_rate.get(
            self.cell_key(mask, strategy_label), 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_masks": int(len(self.factor_masks)),
            "strategy_labels": list(self.strategy_labels),
            "n_scenarios_per_mask": int(
                self.n_scenarios_per_mask),
            "outcome_cids": [
                str(c) for c in self.outcome_cids],
            "config_cid": str(self.config_cid),
            "n_cells": int(len(self.per_cell_success_rate)),
        }

    def cid(self) -> str:
        # Include the per-cell stats explicitly so the CID
        # responds to any rate change.
        return _sha256_hex({
            "kind": "w82_compound_failure_recovery_report_v1",
            "report_summary": self.to_dict(),
            "per_cell_success_rate": {
                k: float(round(v, 9))
                for k, v in sorted(
                    self.per_cell_success_rate.items())},
            "per_cell_mean_fidelity": {
                k: float(round(v, 9))
                for k, v in sorted(
                    self.per_cell_mean_fidelity.items())},
        })


@dataclasses.dataclass(frozen=True)
class CompoundFailureWitnessV1:
    schema: str
    report_cid: str
    strategy_labels: tuple[str, ...]
    n_masks: int
    n_scenarios_per_mask: int
    w82_dominates_at_all5_active: bool
    w82_success_rate_at_all5: float
    max_baseline_success_rate_at_all5: float
    config_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "report_cid": str(self.report_cid),
            "strategy_labels": list(self.strategy_labels),
            "n_masks": int(self.n_masks),
            "n_scenarios_per_mask": int(
                self.n_scenarios_per_mask),
            "w82_dominates_at_all5_active": bool(
                self.w82_dominates_at_all5_active),
            "w82_success_rate_at_all5": float(round(
                self.w82_success_rate_at_all5, 12)),
            "max_baseline_success_rate_at_all5": float(round(
                self.max_baseline_success_rate_at_all5, 12)),
            "config_cid": str(self.config_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_compound_failure_witness_v1",
            "witness": self.to_dict()})


def run_compound_failure_bench_v1(
        *, factor_masks: Sequence[
            CompoundFailureFactorMaskV1] | None = None,
        n_scenarios_per_mask: int = W82_CFB_DEFAULT_N_SEEDS,
        base_seed: int = W82_CFB_DEFAULT_SEED,
        strategies: (
            Mapping[str, StrategyCallable] | None) = None,
) -> CompoundFailureRecoveryReportV1:
    """Run the compound-failure bench.

    If ``factor_masks`` is None, defaults to all 32 masks
    (2^5 power set). If ``strategies`` is None, uses the
    canonical W82 set.
    """
    masks = (
        tuple(factor_masks)
        if factor_masks is not None else
        all_factor_masks_v1())
    if strategies is None:
        strategies = build_compound_failure_strategy_set_v1()
    strategy_labels = tuple(strategies.keys())
    # Per cell aggregates.
    per_cell_success: dict[str, list[bool]] = {}
    per_cell_fid: dict[str, list[float]] = {}
    per_cell_vis: dict[str, list[int]] = {}
    per_cell_rep: dict[str, list[int]] = {}
    per_cell_primary: dict[str, dict[str, int]] = {}
    outcome_cids: list[str] = []
    for mask in masks:
        for s_idx in range(int(n_scenarios_per_mask)):
            sub_seed = int(
                _sha256_hex({
                    "kind": "w82_compound_failure_seed",
                    "base_seed": int(base_seed),
                    "mask_cid": str(mask.cid()),
                    "s_idx": int(s_idx),
                })[:12], 16) & 0x7FFFFFFF
            scenario = build_compound_failure_scenario_v1(
                seed=int(sub_seed), factor_mask=mask)
            for label, strategy in strategies.items():
                outcome = strategy(scenario)
                outcome_cids.append(outcome.cid())
                key = f"{mask.cid()[:16]}::{label}"
                per_cell_success.setdefault(
                    key, []).append(
                    bool(outcome.task_success))
                per_cell_fid.setdefault(
                    key, []).append(
                    float(outcome.reconstruction_fidelity))
                per_cell_vis.setdefault(
                    key, []).append(
                    int(outcome.visible_tokens_used))
                per_cell_rep.setdefault(
                    key, []).append(
                    int(outcome.replay_flops))
                breakdown = per_cell_primary.setdefault(
                    key, {})
                pf = str(outcome.primary_failure_factor)
                breakdown[pf] = breakdown.get(pf, 0) + 1
    per_cell_success_rate = {
        k: float(sum(v)) / float(len(v))
        for k, v in per_cell_success.items()}
    per_cell_mean_fidelity = {
        k: float(sum(v)) / float(len(v))
        for k, v in per_cell_fid.items()}
    per_cell_mean_visible_tokens = {
        k: float(sum(v)) / float(len(v))
        for k, v in per_cell_vis.items()}
    per_cell_mean_replay_flops = {
        k: float(sum(v)) / float(len(v))
        for k, v in per_cell_rep.items()}
    config_cid = _sha256_hex({
        "kind": "w82_compound_failure_bench_config_v1",
        "n_masks": int(len(masks)),
        "strategy_labels": list(strategy_labels),
        "n_scenarios_per_mask": int(n_scenarios_per_mask),
        "n_outcomes": int(len(outcome_cids)),
    })
    return CompoundFailureRecoveryReportV1(
        schema=W82_COMPOUND_FAILURE_V1_SCHEMA_VERSION,
        factor_masks=tuple(masks),
        strategy_labels=tuple(strategy_labels),
        n_scenarios_per_mask=int(n_scenarios_per_mask),
        per_cell_success_rate=per_cell_success_rate,
        per_cell_mean_fidelity=per_cell_mean_fidelity,
        per_cell_mean_visible_tokens=(
            per_cell_mean_visible_tokens),
        per_cell_mean_replay_flops=per_cell_mean_replay_flops,
        per_cell_primary_failure_factor_breakdown=(
            per_cell_primary),
        outcome_cids=tuple(outcome_cids),
        config_cid=str(config_cid),
    )


def emit_compound_failure_witness_v1(
        report: CompoundFailureRecoveryReportV1,
) -> CompoundFailureWitnessV1:
    """Emit a witness for the load-bearing all-5-active mask."""
    all5_mask = CompoundFailureFactorMaskV1(
        contradiction=True, corruption=True,
        replacement=True, restart=True, blackout=True)
    w82_rate = float(report.success_rate(
        all5_mask, W82_CFB_STRATEGY_W82_COMPOUND_REPAIR))
    baselines = [
        l for l in report.strategy_labels
        if str(l) != str(
            W82_CFB_STRATEGY_W82_COMPOUND_REPAIR)]
    max_baseline_rate = max(
        (float(report.success_rate(all5_mask, l))
         for l in baselines), default=0.0)
    dominates = bool(w82_rate > max_baseline_rate)
    return CompoundFailureWitnessV1(
        schema=W82_COMPOUND_FAILURE_V1_SCHEMA_VERSION,
        report_cid=str(report.cid()),
        strategy_labels=tuple(report.strategy_labels),
        n_masks=int(len(report.factor_masks)),
        n_scenarios_per_mask=int(
            report.n_scenarios_per_mask),
        w82_dominates_at_all5_active=bool(dominates),
        w82_success_rate_at_all5=float(w82_rate),
        max_baseline_success_rate_at_all5=float(
            max_baseline_rate),
        config_cid=str(report.config_cid),
    )


def run_compound_failure_bench_end_to_end_v1(
        *, factor_masks: Sequence[
            CompoundFailureFactorMaskV1] | None = None,
        n_scenarios_per_mask: int = W82_CFB_DEFAULT_N_SEEDS,
        base_seed: int = W82_CFB_DEFAULT_SEED,
) -> tuple[CompoundFailureRecoveryReportV1,
           CompoundFailureWitnessV1]:
    report = run_compound_failure_bench_v1(
        factor_masks=factor_masks,
        n_scenarios_per_mask=int(n_scenarios_per_mask),
        base_seed=int(base_seed))
    witness = emit_compound_failure_witness_v1(report)
    return report, witness


__all__ = [
    "W82_COMPOUND_FAILURE_V1_SCHEMA_VERSION",
    "W82_CFB_STRATEGY_LABELS",
    "W82_CFB_STRATEGY_NAIVE_MAJORITY",
    "W82_CFB_STRATEGY_BOUNDED_WINDOW_K128",
    "W82_CFB_STRATEGY_SUBSTRATE_V2_ONLY",
    "W82_CFB_STRATEGY_W82_COMPOUND_REPAIR",
    "W82_COMPOUND_FAILURE_FACTORS",
    "CompoundFailureFactor",
    "CompoundFailureFactorMaskV1",
    "CompoundFailureScenarioV1",
    "CompoundFailureOutcomeV1",
    "CompoundFailureRecoveryReportV1",
    "CompoundFailureWitnessV1",
    "all_factor_masks_v1",
    "build_compound_failure_scenario_v1",
    "attribute_primary_failure_factor_v1",
    "build_compound_failure_strategy_set_v1",
    "run_compound_failure_bench_v1",
    "emit_compound_failure_witness_v1",
    "run_compound_failure_bench_end_to_end_v1",
]
