"""W122 / COO-9 — matched-family multi-seed CLOSURE on official ICPC + ICPC
different-mechanism (M3) signal AUDIT.

W120 ran the W89 mechanism on a >=30 official-ICPC RESISTANT battlefield (seed 120001)
=> B-A1 = +0.00 pp FAIL.  W121 ran the matched EXPOSED control on the SAME official ICPC
family (seed 120001) => B-A1 = +3.33 pp FAIL.  Both within the +-3.34 pp null band =>
W121 `CONFOUND_WEAKENS / bounded ceiling HARDENS`, with ONE remaining live caveat:
**single seed each side.**

W122 closes that caveat directly.  It runs ONE paired new seed on BOTH fields, evaluates
the 2-seed aggregate under a pre-committed SYMMETRIC closure rule, and earns a 3rd paired
seed ONLY if the 2-seed aggregate is ambiguous.  This module is the PURE / deterministic /
NIM-free closure-decision layer + the Lane-beta ICPC mechanism-signal audit.  It REUSES
the W121 exposed-control constants + three-branch interpreter and the W120/W121 bench
report shape with NO duplication; the NIM spend lives in
``scripts/run_w122_paired_seed_pilot.py``.

Two pre-committed decision surfaces (``docs/RUNBOOK_W122.md``):

* **Lane alpha closure (section 2/3):** ``interpret_paired_closure_v1`` maps the 2-seed
  (or 3-seed) MEAN B-A1 per field to one of four branches —
  B1 ``MATCHED_FAMILY_NULL_SURVIVES_MULTISEED_CAVEAT_CLOSED`` (both means in the +-3.34
  null band => single-seed caveat CLOSED, confound stays WEAKENED multi-seed);
  B2 ``EXPOSED_MARGIN_RESTRENGTHENS_CONTAMINATION`` (exposed >= +5 while resistant null);
  B3 ``RESISTANT_SUPERIORITY_MULTISEED_CANDIDATE`` (resistant >= +5 => candidate third
  retirement, with-or-without exposed);
  B4 ``AMBIGUOUS_THIRD_PAIRED_SEED_EARNED`` (anything else => earn exactly one more paired
  seed).  Precedence: B3 > B2 > B1 > B4.  Per-seed PASS_MECHANISM_DRIVEN gates qualify a
  >=5 pp mean as a CLEAN rise.

* **Lane beta mechanism audit (section 4):** ``audit_icpc_mechanism_signal_v1`` decides
  whether the strongest non-reflexion mechanism (M3, the executor-grounded structured-
  failure patcher) has a materially-useful signal on ICPC that reflexion does NOT already
  have.  M3's load-bearing edge is the explicit expected/actual contract extracted from
  the FAILING test.  On official ICPC the hidden oracle is SECRET token-diff (anti-cheat:
  it returns only "wrong answer on a hidden case", never the expected value), so the only
  expected-value signal is the PUBLIC SAMPLES — which the existing reflexion bench already
  feeds.  M3's EXCLUSIVE signal (an expected/actual contract reflexion lacks) is therefore
  structurally ~0 on ICPC => KILL the lane NIM-free.  The audit is machine-checkable and
  falsifiable (flip the grader regime to one that reveals the hidden expected, or feed a
  sidecar with >= the floor of hidden-only-with-expected turns, and the verdict flips to
  BUILD).
"""
from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Optional, Sequence

from .coordpy_icpc_exposed_control_v1 import (
    AMBIGUITY_BAND_PP,
    EXPOSED_MARGIN_PASS_PP,
    interpret_exposed_vs_resistant_v1,
)

W122_PAIRED_SEED_CLOSURE_V1_SCHEMA_VERSION: str = (
    "coordpy.coordpy_icpc_paired_seed_closure_v1.v1")

# Carried VERBATIM from W121 (RUNBOOK_W122 section 2).
NULL_BAND_PP: float = AMBIGUITY_BAND_PP           # 3.34
MARGIN_PASS_PP: float = EXPOSED_MARGIN_PASS_PP    # 5.00
M3_SIGNAL_FLOOR: float = 0.33                     # W111/W106 33% floor (Lane beta)

# Verdict labels emitted by the verbatim W108 gate evaluator.
VERDICT_PASS_MECHANISM: str = "PASS_MECHANISM_DRIVEN"
VERDICT_PASS_NON_MECHANISM: str = "PASS_NON_MECHANISM_DRIVEN"
VERDICT_FAIL: str = "FAIL"

# The two fields and their existing + paired seeds (RUNBOOK_W122 section 2/6).
W122_RESISTANT_FIELD: str = "resistant"           # W120 battlefield (CID 01bf9ef8...)
W122_EXPOSED_FIELD: str = "exposed"               # W121 control      (CID 32d15db5...)
W122_EXISTING_SEED: int = 120_001                 # already run (W120 + W121)
W122_PAIRED_SEED: int = 120_002                   # the new paired seed
W122_THIRD_SEED: int = 120_003                    # conditional (B4 only)


# ======================================================== per-field seed aggregation

@dataclasses.dataclass(frozen=True)
class FieldSeedResultV1:
    """One seed's outcome on one field (resistant or exposed)."""
    field: str
    seed: int
    b_minus_a1_pp: float
    verdict_label: str
    a0_pass_at_1_pct: float
    mlb2_rescue_rate: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def is_clean_pass(self) -> bool:
        return self.verdict_label == VERDICT_PASS_MECHANISM


@dataclasses.dataclass(frozen=True)
class FieldAggregateV1:
    """Multi-seed aggregate for one field."""
    field: str
    seeds: tuple[int, ...]
    per_seed_b_minus_a1_pp: tuple[float, ...]
    mean_b_minus_a1_pp: float
    n_seeds: int
    all_seeds_clean_pass: bool
    per_seed_a0_pct: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def aggregate_field_seeds_v1(
        results: Sequence[FieldSeedResultV1]) -> FieldAggregateV1:
    """Mean B-A1 across the seeds run on one field. Order-independent in the mean;
    seeds reported sorted for determinism."""
    if not results:
        raise ValueError("aggregate_field_seeds_v1 needs >=1 seed result")
    field = results[0].field
    if any(r.field != field for r in results):
        raise ValueError("all results must be for the same field")
    ordered = sorted(results, key=lambda r: int(r.seed))
    margins = tuple(float(r.b_minus_a1_pp) for r in ordered)
    mean = sum(margins) / float(len(margins))
    return FieldAggregateV1(
        field=str(field),
        seeds=tuple(int(r.seed) for r in ordered),
        per_seed_b_minus_a1_pp=margins,
        mean_b_minus_a1_pp=float(round(mean, 4)),
        n_seeds=int(len(ordered)),
        all_seeds_clean_pass=bool(all(r.is_clean_pass for r in ordered)),
        per_seed_a0_pct=tuple(float(r.a0_pass_at_1_pct) for r in ordered))


# ======================================================== the symmetric closure rule

# Branch labels (RUNBOOK_W122 section 2).
B1_CAVEAT_CLOSED: str = "MATCHED_FAMILY_NULL_SURVIVES_MULTISEED_CAVEAT_CLOSED"
B2_EXPOSED_RESTRENGTHENS: str = "EXPOSED_MARGIN_RESTRENGTHENS_CONTAMINATION"
B3_RESISTANT_CANDIDATE: str = "RESISTANT_SUPERIORITY_MULTISEED_CANDIDATE"
B4_AMBIGUOUS: str = "AMBIGUOUS_THIRD_PAIRED_SEED_EARNED"


@dataclasses.dataclass(frozen=True)
class PairedClosureVerdictV1:
    schema: str
    resistant: FieldAggregateV1
    exposed: FieldAggregateV1
    null_band_pp: float
    margin_pass_pp: float
    resistant_mean_pp: float
    exposed_mean_pp: float
    resistant_in_null_band: bool
    exposed_in_null_band: bool
    resistant_shows_margin: bool
    exposed_shows_margin: bool
    branch: str
    caveat_closed: bool
    third_seed_earned: bool
    w121_delta: str
    interpretation: str
    # the single-seed W121 contrast re-derived on the multi-seed means (reuse)
    multiseed_exposed_vs_resistant_outcome: str

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return d


def interpret_paired_closure_v1(
        *,
        resistant: FieldAggregateV1,
        exposed: FieldAggregateV1,
        null_band_pp: float = NULL_BAND_PP,
        margin_pass_pp: float = MARGIN_PASS_PP,
) -> PairedClosureVerdictV1:
    """Pre-committed SYMMETRIC closure rule (RUNBOOK_W122 section 2).

    Precedence B3 > B2 > B1 > B4 over the multi-seed MEAN B-A1 per field.  A >=5 pp mean
    only counts as a CLEAN rise (B3/B2) if EVERY seed on that field is
    PASS_MECHANISM_DRIVEN; otherwise the >=5 pp mean is MARGIN_WITHOUT_MECHANISM and falls
    through to B1 (if it is also a null on the other axis — impossible when one mean >=5)
    or to B4.  Because a non-clean >=5 mean cannot be B1, it lands in B4 (earn the 3rd
    seed to resolve)."""
    rm = float(resistant.mean_b_minus_a1_pp)
    em = float(exposed.mean_b_minus_a1_pp)

    r_null = bool(abs(rm) <= null_band_pp)
    e_null = bool(abs(em) <= null_band_pp)
    r_margin_raw = bool(rm >= margin_pass_pp)
    e_margin_raw = bool(em >= margin_pass_pp)
    # a margin only counts as a CLEAN rise if every seed on that field is a clean pass
    r_margin = bool(r_margin_raw and resistant.all_seeds_clean_pass)
    e_margin = bool(e_margin_raw and exposed.all_seeds_clean_pass)

    if r_margin:                                  # B3 (resistant superiority candidate)
        branch = B3_RESISTANT_CANDIDATE
        caveat_closed = False
        third = False
        delta = ("OVERTURNS the bounded resistant ceiling at multi-seed: the mechanism "
                 "beats A1 on contamination-RESISTANT official ICPC code across seeds "
                 "(every resistant seed PASS_MECHANISM_DRIVEN). Candidate THIRD retirement "
                 "on resistant code; W123 = full resistant retirement bench (3+ seeds x "
                 ">=100, the W89/W105 bar). Do NOT declare a retirement on n=30 seeds.")
        interp = (f"RESISTANT mean B-A1 = {rm:+.2f} pp >= {margin_pass_pp:+.2f} (clean per "
                  f"seed); EXPOSED mean {em:+.2f} pp. The matched-family resistant null is "
                  f"OVERTURNED multi-seed.")
    elif e_margin and r_null:                     # B2 (exposed margin restrengthens)
        branch = B2_EXPOSED_RESTRENGTHENS
        caveat_closed = True
        third = False
        delta = ("REVISES W121 back UP: the exposed margin was single-seed-noisy at W121; "
                 "de-noising reveals it (every exposed seed PASS_MECHANISM_DRIVEN) while "
                 "resistant stays null => contamination reading RE-STRENGTHENS; the "
                 "difficulty/family-ease loophole closes via an exposed margin.")
        interp = (f"EXPOSED mean B-A1 = {em:+.2f} pp >= {margin_pass_pp:+.2f} (clean per "
                  f"seed) while RESISTANT mean {rm:+.2f} pp stays in the +-{null_band_pp} "
                  f"null band. Exposure within ICPC reproduces the margin once de-noised.")
    elif r_null and e_null:                       # B1 (caveat closed; the null survives)
        branch = B1_CAVEAT_CLOSED
        caveat_closed = True
        third = False
        delta = ("CONFIRMS + HARDENS W121: the matched-family null SURVIVES multiple "
                 "seeds. Contamination-confound stays WEAKENED and is now MULTI-SEED; the "
                 "bounded ceiling HARDENS to multi-seed HumanEval-family-(ease/structure)-"
                 "specific @ 70B. The single-seed caveat is CLOSED.")
        interp = (f"BOTH means in the +-{null_band_pp} null band (RESISTANT {rm:+.2f}, "
                  f"EXPOSED {em:+.2f} pp). On the SAME official ICPC family at matched "
                  f"difficulty, flipping only exposure does NOT reopen the mechanism "
                  f"margin across seeds. Single-seed caveat removed.")
    else:                                         # B4 (ambiguous => earn the 3rd seed)
        branch = B4_AMBIGUOUS
        caveat_closed = False
        third = True
        delta = ("AMBIGUOUS at this seed count: a mean lands in the (%.2f, %.2f) gap, or "
                 "a >=5 mean lacks clean per-seed gates, or the fields disagree in "
                 "direction. Earn EXACTLY ONE more paired seed on BOTH fields; do NOT buy "
                 "a 4th." % (null_band_pp, margin_pass_pp))
        interp = (f"RESISTANT mean {rm:+.2f} pp, EXPOSED mean {em:+.2f} pp: not jointly "
                  f"decisive under the +-{null_band_pp}/{margin_pass_pp:+.2f} rule. Third "
                  f"paired seed earned.")

    # Reuse the W121 single-seed interpreter on the multi-seed means for continuity.
    ms = interpret_exposed_vs_resistant_v1(
        exposed_b_minus_a1=em, resistant_b_minus_a1=rm,
        exposed_margin_pass_pp=margin_pass_pp, ambiguity_band_pp=null_band_pp)

    return PairedClosureVerdictV1(
        schema=W122_PAIRED_SEED_CLOSURE_V1_SCHEMA_VERSION,
        resistant=resistant, exposed=exposed,
        null_band_pp=float(null_band_pp), margin_pass_pp=float(margin_pass_pp),
        resistant_mean_pp=float(rm), exposed_mean_pp=float(em),
        resistant_in_null_band=bool(r_null), exposed_in_null_band=bool(e_null),
        resistant_shows_margin=bool(r_margin), exposed_shows_margin=bool(e_margin),
        branch=str(branch), caveat_closed=bool(caveat_closed),
        third_seed_earned=bool(third), w121_delta=str(delta),
        interpretation=str(interp),
        multiseed_exposed_vs_resistant_outcome=str(ms.outcome))


# ======================================================== Lane beta — ICPC M3 signal audit

# Failing-reflexion-turn signal classes (RUNBOOK_W122 section 4).
TURN_PUBLIC_SAMPLE_WRONG: str = "PUBLIC_SAMPLE_WRONG"   # expected shown to reflexion ALREADY
TURN_HIDDEN_ONLY: str = "HIDDEN_ONLY"                   # samples pass, hidden fails; expected SECRET
TURN_RUNTIME_TRACEBACK: str = "RUNTIME_TRACEBACK"       # stderr traceback; reflexion shows it ALREADY
TURN_TIMEOUT: str = "TIMEOUT"
TURN_NO_SIGNAL: str = "NO_SIGNAL"

_ATTEMPT_SPLIT_RE = re.compile(r"--- Attempt \d+ ")
_ERR_TOKEN_RE = re.compile(r"(Error|Exception|Traceback)")
_SAMPLE_WRONG_RE = re.compile(r"WRONG \(expected")
_REFLEXION_MARKER = "reflective debugging loop"


def classify_reflexion_turn_v1(prompt: str) -> Optional[str]:
    """Classify the MOST-RECENT prior-attempt feedback embedded in a reflexion prompt.

    Returns one of the TURN_* classes, or None if ``prompt`` is not a reflexion prompt
    (attempt-0 initial prompts carry no prior feedback).  Pure, deterministic; reads ONLY
    the prompt text (the same public signal the model saw)."""
    p = str(prompt or "")
    if _REFLEXION_MARKER not in p:
        return None
    blocks = _ATTEMPT_SPLIT_RE.split(p)
    last = blocks[-1] if len(blocks) > 1 else p
    has_stderr = ("Executor stderr (tail):" in last
                  and "Executor stderr (tail):\n\n" not in last)
    if "TIMEOUT" in last:
        return TURN_TIMEOUT
    if has_stderr and _ERR_TOKEN_RE.search(last):
        return TURN_RUNTIME_TRACEBACK
    if _SAMPLE_WRONG_RE.search(last):
        return TURN_PUBLIC_SAMPLE_WRONG
    rejected = "REJECTED by the judge" in last
    samples_all_pass = (("sample 1: PASS" in last)
                        and ("WRONG" not in last)
                        and ("TIMEOUT" not in last)
                        and ("runtime error" not in last))
    if samples_all_pass and rejected:
        return TURN_HIDDEN_ONLY
    return TURN_NO_SIGNAL


@dataclasses.dataclass(frozen=True)
class IcpcMechanismSignalAuditV1:
    schema: str
    n_reflexion_turns: int
    counts: dict[str, int]
    public_sample_wrong_fraction: float
    hidden_only_fraction: float
    runtime_traceback_fraction: float
    grader_reveals_hidden_expected: bool
    m3_exclusive_signal_fraction: float
    m3_signal_floor: float
    verdict: str            # "BUILD_M3_PROBE" | "KILL_M3_LANE_NIM_FREE"
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


VERDICT_BUILD_M3: str = "BUILD_M3_PROBE"
VERDICT_KILL_M3: str = "KILL_M3_LANE_NIM_FREE"


def audit_icpc_mechanism_signal_v1(
        *,
        turn_classes: Sequence[str],
        grader_reveals_hidden_expected: bool,
        m3_signal_floor: float = M3_SIGNAL_FLOOR,
) -> IcpcMechanismSignalAuditV1:
    """Pre-committed Lane-beta earn rule (RUNBOOK_W122 section 4).

    ``grader_reveals_hidden_expected`` is a property of the grading REGIME, not the model:
    BigCodeBench's hidden ``unittest`` oracle prints ``actual != expected`` => True; the
    official-ICPC SECRET token-diff oracle returns only "wrong answer on a hidden case" =>
    False.  M3's EXCLUSIVE signal (an expected/actual contract reflexion does NOT already
    have from the public samples) is available only on HIDDEN_ONLY turns AND only when the
    regime reveals the hidden expected.  BUILD iff that exclusive fraction >= the floor;
    else KILL NIM-free."""
    turns = [t for t in turn_classes if t is not None]
    n = len(turns)
    counts = {k: 0 for k in (TURN_PUBLIC_SAMPLE_WRONG, TURN_HIDDEN_ONLY,
                             TURN_RUNTIME_TRACEBACK, TURN_TIMEOUT, TURN_NO_SIGNAL)}
    for t in turns:
        counts[t] = counts.get(t, 0) + 1
    denom = float(n) or 1.0
    psw = counts[TURN_PUBLIC_SAMPLE_WRONG] / denom
    hid = counts[TURN_HIDDEN_ONLY] / denom
    rtb = counts[TURN_RUNTIME_TRACEBACK] / denom
    exclusive = hid if grader_reveals_hidden_expected else 0.0
    build = bool(exclusive >= float(m3_signal_floor))
    if build:
        verdict = VERDICT_BUILD_M3
        rationale = (
            f"m3_exclusive_signal_fraction={exclusive:.3f} >= floor {m3_signal_floor:.2f}: "
            "the grading regime reveals the hidden expected on enough HIDDEN_ONLY turns "
            "that M3 holds an expected/actual contract reflexion lacks => a fair M3 probe "
            "is EARNED.")
    else:
        verdict = VERDICT_KILL_M3
        rationale = (
            f"m3_exclusive_signal_fraction={exclusive:.3f} < floor {m3_signal_floor:.2f}. "
            "On official ICPC the hidden oracle is SECRET token-diff (returns only 'wrong "
            "answer on a hidden case', never the expected value), so M3's load-bearing "
            "differentiator (an expected/actual contract from the FAILING hidden test) is "
            "structurally absent; the only expected-value signal is the public samples, "
            f"which the existing reflexion bench ALREADY feeds ({psw:.0%} of failing "
            "turns). M3 reduces to a reflexion prompt-variant on this family and was "
            "already sub-reflexion (12.5% < 25%) where it had its FULL edge (W111). "
            "=> KILL the lane NIM-free; the same-family ceiling is mechanism-robust, not "
            "merely reflexion-specific.")
    return IcpcMechanismSignalAuditV1(
        schema=W122_PAIRED_SEED_CLOSURE_V1_SCHEMA_VERSION,
        n_reflexion_turns=int(n), counts=dict(counts),
        public_sample_wrong_fraction=float(round(psw, 4)),
        hidden_only_fraction=float(round(hid, 4)),
        runtime_traceback_fraction=float(round(rtb, 4)),
        grader_reveals_hidden_expected=bool(grader_reveals_hidden_expected),
        m3_exclusive_signal_fraction=float(round(exclusive, 4)),
        m3_signal_floor=float(m3_signal_floor),
        verdict=str(verdict), rationale=str(rationale))


def classify_sidecar_turns_v1(sidecar_records: Sequence[dict]) -> list[str]:
    """Classify every reflexion turn in a list of sidecar call records (each a dict with a
    ``prompt`` field). Non-reflexion (attempt-0) records are skipped."""
    out: list[str] = []
    for rec in sidecar_records:
        cls = classify_reflexion_turn_v1(str(rec.get("prompt", "")))
        if cls is not None:
            out.append(cls)
    return out


# ======================================================== W123 fire condition

@dataclasses.dataclass(frozen=True)
class W123FireConditionV1:
    schema: str
    closure_branch: str
    fires_on: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def w123_fire_condition_v1(closure_branch: str) -> W123FireConditionV1:
    """Pre-committed W123 branch logic (RUNBOOK_W122 section 8)."""
    if closure_branch == B1_CAVEAT_CLOSED:
        fires = ("multi-seed null confirmed => W123 = accept the multi-seed-hardened "
                 "bounded ceiling + pursue a GENUINELY DIFFERENT axis (NOT another ICPC "
                 "seed/rerun), OR a stronger-than-Maverick primary-KNOWN model on BOTH "
                 "battlefields if one opens.")
    elif closure_branch == B2_EXPOSED_RESTRENGTHENS:
        fires = ("exposed margin restrengthens contamination => W123 = a second matched "
                 "exposure control on a DIFFERENT family to confirm the restrengthening "
                 "(NOT a third ICPC seed).")
    elif closure_branch == B3_RESISTANT_CANDIDATE:
        fires = ("resistant superiority candidate => W123 = the full resistant retirement "
                 "bench (3+ seeds x >=100 problems, the W89/W105 bar) to confirm a THIRD "
                 "retirement on resistant code.")
    else:  # B4
        fires = ("ambiguous after the 3rd seed => register the residual ambiguity; W123 = "
                 "accept the bounded claim or escalate to a larger n PER FIELD (not more "
                 "seeds at n=30).")
    return W123FireConditionV1(
        schema=W122_PAIRED_SEED_CLOSURE_V1_SCHEMA_VERSION,
        closure_branch=str(closure_branch), fires_on=str(fires))


__all__ = [
    "W122_PAIRED_SEED_CLOSURE_V1_SCHEMA_VERSION",
    "NULL_BAND_PP", "MARGIN_PASS_PP", "M3_SIGNAL_FLOOR",
    "W122_RESISTANT_FIELD", "W122_EXPOSED_FIELD",
    "W122_EXISTING_SEED", "W122_PAIRED_SEED", "W122_THIRD_SEED",
    "FieldSeedResultV1", "FieldAggregateV1", "aggregate_field_seeds_v1",
    "B1_CAVEAT_CLOSED", "B2_EXPOSED_RESTRENGTHENS", "B3_RESISTANT_CANDIDATE",
    "B4_AMBIGUOUS",
    "PairedClosureVerdictV1", "interpret_paired_closure_v1",
    "TURN_PUBLIC_SAMPLE_WRONG", "TURN_HIDDEN_ONLY", "TURN_RUNTIME_TRACEBACK",
    "TURN_TIMEOUT", "TURN_NO_SIGNAL",
    "classify_reflexion_turn_v1", "classify_sidecar_turns_v1",
    "IcpcMechanismSignalAuditV1", "audit_icpc_mechanism_signal_v1",
    "VERDICT_BUILD_M3", "VERDICT_KILL_M3",
    "W123FireConditionV1", "w123_fire_condition_v1",
]
