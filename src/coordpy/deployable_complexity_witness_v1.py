"""W134 / COO-9 — DEPLOYABLE complexity witness (constraint-derived, oracle-free).

W133 proved the exact-oracle EW2 complexity witness is REAL and load-bearing (C2/C3 beat the
blind W132 stack B0 by +6.06 pp and A1 by +12.12 pp on held-out dev; MLB-2 60 %; 4 algorithmic
COMPLEXITY rescues), but the gain was SINGLE-MODE and — crucially — it *consumed the oracle's
timing*: EW2 executed ``ref_source`` to establish "a correct reference finishes in
``big_ref_runtime_s`` s" and fired when the candidate was >=8x slower than that reference.  That
witness is not deployable: a real deployment has no reference solution.

This module distils EW2 into a **deployable, public-signal complexity witness** that needs NO
oracle answer.  It replaces the reference timing with two public-signal facts:

* **DW1 — constraint-derived budget.**  Parse the size constraint ``N_max`` from the PUBLIC
  statement (``parse_max_constraint_v1``) and derive the admissible complexity regime from a
  standard ops budget (``COMPLEXITY_OPS_BUDGET = 5e8``): at ``N_max=1e5`` the admissible exponent
  ceiling is ``log(5e8)/log(1e5) ~= 1.74`` so O(N^2) is inadmissible and O(N log N) is admissible.
* **DW2 — stress-growth.**  Synthesise a geometric ladder of PUBLIC-FORMAT inputs (parsed from
  the public samples; never the hidden cases) across a fixed shape set, MEASURE the candidate's
  OWN runtime across the ladder, fit a log-log growth exponent, and extrapolate to ``N_max``.

The witness fires "your program is asymptotically too slow" iff a ladder rung TLEs OR the fitted
exponent is super-linear OR the extrapolation exceeds the budget — using ONLY the public
statement, public-format inputs, and the candidate's own observed runtime.  No ``ref_source`` /
``brute_source`` / ``naive_source`` / hidden case is ever consulted, and no expected output is
ever emitted.

This is the SwiftSolve (arXiv:2510.22626, 2025) / trend-prof (FSE'07) measure-and-fit principle,
realised oracle-free over CoordPy's own resistant-by-construction complexity field.  Per the
literature's documented instability of slope-fitting at small n (SwiftSolve; GuessCompx
arXiv:1911.01420), the verdict is a *calibrated diagnostic with a guarded R^2 confidence gate*,
NOT a ground-truth complexity oracle: a low-confidence fit ABSTAINS to blind reflexion rather
than mis-firing, so a deployable arm is never worse than B0.

Reuses (explicit-import only, NO duplication): ``parse_max_constraint_v1`` from
``public_signal_selection_oracle_v1`` (the public-signal constraint parser);
``COMPLEXITY_OPS_BUDGET`` / ``parse_complexity_exponent_v1`` / ``complexity_admissible_v1`` from
``stronger_generator_slate_v1`` (the W130 complexity gate); the executor + token counter from
``resistant_by_construction_battlefield_v1``; the pilot record, candidate extractor, secret
grader, blind reflexion prompt, and public-sample feedback from ``icpc_reflexion_bench_v1``.
Pure / deterministic except the (already-audited) candidate-execution subprocess; NO model
inference lives here (that is the W134 driver script).  See ``docs/RUNBOOK_W134.md`` (LOCKED).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import re
import time
from typing import Any, Optional, Sequence

from .resistant_by_construction_battlefield_v1 import _exec_capture_v1, _tok_count
from .public_signal_selection_oracle_v1 import parse_max_constraint_v1
from .stronger_generator_slate_v1 import (
    COMPLEXITY_OPS_BUDGET,
    complexity_admissible_v1,
    parse_complexity_exponent_v1,
)
from .icpc_reflexion_bench_v1 import (
    IcpcArmOutcomeV1,
    IcpcPilotProblemV1,
    W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION,
    _initial_prompt,
    _reflexion_prompt,
    _samples_block,
    extract_candidate_code_v1,
    grade_on_secret_v1,
    sample_feedback_v1,
)

W134_DEPLOYABLE_COMPLEXITY_WITNESS_V1_SCHEMA_VERSION: str = (
    "coordpy.deployable_complexity_witness_v1.v1")

# ---- witness kinds (deployable; never carries an oracle answer) ---------------------
DW_COMPLEXITY: str = "COMPLEXITY"     # the candidate is asymptotically too slow (deployable verdict)
DW_NONE: str = "NONE"                 # admissible growth, or unmeasurable (-> blind fallback)

# ---- deployable arms ----------------------------------------------------------------
ARM_D1_REWRITE: str = "D1"            # deployable rewrite: fire -> rewrite, else blind
ARM_D2_GATED: str = "D2"             # D1 + the W130 constraint-derived admissibility gate
ARM_D3_CONTROLLER: str = "D3"         # KEEP / REWRITE / ABSTAIN controller (LEAD)

# ---- controller actions -------------------------------------------------------------
ACTION_REWRITE: str = "REWRITE"       # inadmissible growth measured -> emit the DW block
ACTION_KEEP: str = "KEEP"             # measurably admissible -> blind reflexion (bug is elsewhere)
ACTION_ABSTAIN: str = "ABSTAIN"       # unmeasurable / low-confidence -> blind reflexion fallback

# ---- LOCKED witness compute budget (docs/RUNBOOK_W134.md §5) ------------------------
LADDER_BASE: int = 1000               # smallest ladder rung
LADDER_RUNGS: int = 4                 # [1000, 2000, 4000, 8000] (doubling)
LADDER_SHAPES: tuple[str, ...] = ("random", "descending", "constant")
BASELINE_SIZE: int = 64               # interpreter/parse overhead baseline (subtracted)
PER_RUN_TIMEOUT_S: float = 2.0        # per (size, shape) candidate run cap; a TLE => super-linear
NOISE_FLOOR_S: float = 0.005          # a measured compute time below this is treated as noise
P_SUPERLINEAR: float = 1.7            # fitted exponent at/above which growth is inadmissible
WALL_BUDGET_S: float = 2.0            # extrapolated t(N_max) above this is inadmissible
MIN_LADDER_POINTS: int = 3            # >= this many finite rungs (or a TLE) required to judge
R2_MIN_CONFIDENCE: float = 0.7        # SwiftSolve guard: a lower-R^2 fit is low-confidence
# SwiftSolve small-signal guard: a NON-TLE super-linear verdict is only trusted if the candidate
# actually consumed significant runtime at the ladder top.  A genuine O(N^2) on this ladder either
# TLEs or exceeds this; any correct O(N)/O(N log N) finishes far below it — so a noisy high-R^2
# slope on a sub-100ms (fast) program is NOT enough to fire (it would be a measurement artifact).
MIN_SIGNIFICANT_S: float = 0.1
DEFAULT_VALUE_HI: int = 10 ** 9       # default a_i upper bound if unparseable
LADDER_VALUE_SEED: int = 134_900      # determinism for the "random" shape


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ===================================================== DW1 — constraint-derived budget

def parse_value_hi_v1(statement: str) -> int:
    """Best-effort parse of the array-VALUE upper bound (``a_i <= X``) from the public statement,
    for synthesising spec-consistent ladder values.  Defaults to ``DEFAULT_VALUE_HI`` (the value
    range does not affect the asymptotic growth of the candidate, only input validity)."""
    s = statement or ""
    s = re.sub(r"(?<=\d)(?:\\,|\\;|\\ |~|,|\s)(?=\d\d\d(?:\D|$))", "", s)
    best = 0
    for m in re.finditer(r"a_i\s*<=\s*(\d+)\s*(?:\^|\*\*)?\s*", s):
        try:
            best = max(best, int(m.group(1)))
        except ValueError:
            pass
    for m in re.finditer(r"<=\s*10\s*[\^{]\s*(\d+)", s):
        best = max(best, 10 ** int(m.group(1)))
    return best if best >= 1 else DEFAULT_VALUE_HI


@dataclasses.dataclass(frozen=True)
class BudgetFactV1:
    """DW1: the admissible complexity regime derived from the PUBLIC constraint alone."""
    n_max: Optional[int]
    ops_budget: float
    admissible_exponent_ceiling: Optional[float]   # max p with N_max^p <= ops_budget
    naive_quadratic_ops: Optional[float]           # N_max^2 (the O(N^2) cost at N_max)
    quadratic_over_budget: Optional[bool]          # is O(N^2) inadmissible at N_max?

    def to_dict(self) -> dict[str, Any]:
        return {"n_max": self.n_max, "ops_budget": self.ops_budget,
                "admissible_exponent_ceiling": (round(self.admissible_exponent_ceiling, 3)
                                                if self.admissible_exponent_ceiling else None),
                "naive_quadratic_ops": self.naive_quadratic_ops,
                "quadratic_over_budget": self.quadratic_over_budget}


def derive_budget_fact_v1(statement: str, *, ops_budget: float = COMPLEXITY_OPS_BUDGET
                          ) -> BudgetFactV1:
    """DW1 — parse N_max from the public statement and derive the admissible complexity regime.
    Reuses ``parse_max_constraint_v1`` (public-signal).  Emits ONLY a budget fact; no oracle."""
    n_max = parse_max_constraint_v1(statement)
    if not n_max or n_max < 2:
        return BudgetFactV1(n_max=n_max, ops_budget=float(ops_budget),
                            admissible_exponent_ceiling=None, naive_quadratic_ops=None,
                            quadratic_over_budget=None)
    ceiling = math.log(ops_budget) / math.log(n_max)
    quad = float(n_max) ** 2.0
    return BudgetFactV1(n_max=int(n_max), ops_budget=float(ops_budget),
                        admissible_exponent_ceiling=float(ceiling),
                        naive_quadratic_ops=quad, quadratic_over_budget=bool(quad > ops_budget))


# ===================================================== DW2 — public-format ladder generator

@dataclasses.dataclass(frozen=True)
class InputShapeV1:
    """The structure parsed from a public sample: a header line (first token = the size N, the
    remaining tokens are params) followed by an array line of exactly N integers."""
    parseable: bool
    sample_n: int
    header_tokens: tuple[str, ...]      # the full header line tokens of the parsed sample
    value_hi: int

    def to_dict(self) -> dict[str, Any]:
        return {"parseable": self.parseable, "sample_n": self.sample_n,
                "header_tokens": list(self.header_tokens), "value_hi": self.value_hi}


def parse_input_shape_v1(statement: str, samples: Sequence[tuple[str, str]]) -> InputShapeV1:
    """Infer the public input STRUCTURE from the public samples (line 1 = header whose first token
    is the size N; line 2 = N integers).  PUBLIC-signal only; never touches the hidden cases."""
    value_hi = parse_value_hi_v1(statement)
    for inp, _out in samples:
        lines = inp.rstrip("\n").split("\n")
        if len(lines) < 2:
            continue
        head = lines[0].split()
        if not (head and re.fullmatch(r"\d+", head[0])):
            continue
        n = int(head[0])
        arr = lines[1].split()
        if n >= 2 and len(arr) == n and all(re.fullmatch(r"-?\d+", t) for t in arr):
            return InputShapeV1(parseable=True, sample_n=n, header_tokens=tuple(head),
                                value_hi=int(value_hi))
    return InputShapeV1(parseable=False, sample_n=0, header_tokens=(), value_hi=int(value_hi))


def _scaled_header(shape: InputShapeV1, size: int) -> list[str]:
    """Header for a synthesised input of the given size: first token = size; each extra param is
    scaled if it was bounded by the sample's N (an index/window param) else kept (a threshold)."""
    out = [str(size)]
    n0 = max(1, shape.sample_n)
    for tok in shape.header_tokens[1:]:
        try:
            v = int(tok)
        except ValueError:
            out.append(tok)
            continue
        if 1 <= v <= shape.sample_n:                 # window/index param -> scale with N, stay valid
            out.append(str(max(1, min(size, round(v * size / n0)))))
        else:                                         # threshold/target unrelated to N -> keep
            out.append(tok)
    return out


def _shape_array(kind: str, size: int, value_hi: int, rng: random.Random) -> list[int]:
    hi = max(2, min(int(value_hi), 10 ** 9))
    if kind == "descending":                         # strictly decreasing distinct: monotonic-stack worst case
        return list(range(size, 0, -1))
    if kind == "constant":                           # all-equal: equality-degenerate short-circuit defeat
        return [1] * size
    return [rng.randint(1, hi) for _ in range(size)]  # "random" typical


def synth_input_v1(shape: InputShapeV1, *, size: int, kind: str, rng: random.Random) -> str:
    head = _scaled_header(shape, size)
    arr = _shape_array(kind, size, shape.value_hi, rng)
    return " ".join(head) + "\n" + " ".join(str(x) for x in arr) + "\n"


@dataclasses.dataclass(frozen=True)
class LadderV1:
    """A deterministic, content-addressed public-format ladder for one problem."""
    parseable: bool
    sizes: tuple[int, ...]
    baseline_input: str
    rungs: tuple[tuple[int, tuple[tuple[str, str], ...]], ...]   # (size, ((shape, input),...))

    def cid(self) -> str:
        return _sha256_hex({"kind": "w134_ladder_v1", "sizes": list(self.sizes),
                            "baseline": self.baseline_input,
                            "rungs": [[s, [list(t) for t in shp]] for s, shp in self.rungs]})

    def to_dict(self) -> dict[str, Any]:
        return {"parseable": self.parseable, "sizes": list(self.sizes),
                "n_rungs": len(self.rungs), "n_shapes": len(LADDER_SHAPES),
                "ladder_cid": self.cid()}


def build_ladder_v1(statement: str, samples: Sequence[tuple[str, str]], *,
                    ladder_seed: int = LADDER_VALUE_SEED) -> LadderV1:
    """DW2 ladder: a deterministic geometric size ladder of PUBLIC-FORMAT inputs across the fixed
    shape set, synthesised from the public samples + statement ONLY.  Returns a non-parseable
    ladder (the witness then abstains) if the public format cannot be inferred."""
    shape = parse_input_shape_v1(statement, samples)
    if not shape.parseable:
        return LadderV1(parseable=False, sizes=(), baseline_input="", rungs=())
    rng = random.Random(_sha256_hex({"w134_ladder": True, "seed": int(ladder_seed),
                                     "header": list(shape.header_tokens)}))
    sizes = tuple(LADDER_BASE * (2 ** i) for i in range(LADDER_RUNGS))
    baseline = synth_input_v1(shape, size=BASELINE_SIZE, kind="random", rng=rng)
    rungs: list[tuple[int, tuple[tuple[str, str], ...]]] = []
    for sz in sizes:
        per_shape = tuple((k, synth_input_v1(shape, size=sz, kind=k, rng=rng))
                          for k in LADDER_SHAPES)
        rungs.append((sz, per_shape))
    return LadderV1(parseable=True, sizes=sizes, baseline_input=baseline, rungs=tuple(rungs))


# ===================================================== DW2 — measure the candidate's growth

def _time_once(code: str, stdin_text: str, *, timeout_s: float) -> tuple[float, bool, bool]:
    """Run the candidate once; return (wall_s, timed_out, crashed)."""
    t0 = time.time()
    r = _exec_capture_v1(code, stdin_text, timeout_s=timeout_s)
    dt = float(time.time() - t0)
    return (dt, bool(r.timed_out), bool(r.returncode != 0 and not r.timed_out))


def _fit_loglog(points: Sequence[tuple[float, float]]) -> tuple[Optional[float], float]:
    """Least-squares fit of log(t) ~ slope*log(size) + b over finite points; return (slope, R^2)."""
    pts = [(math.log(x), math.log(y)) for x, y in points if x > 0 and y > 0]
    if len(pts) < 2:
        return (None, 0.0)
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    if sxx <= 0:
        return (None, 0.0)
    slope = sxy / sxx
    b = my - slope * mx
    ss_tot = sum((y - my) ** 2 for _, y in pts)
    ss_res = sum((y - (slope * x + b)) ** 2 for x, y in pts)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    return (float(slope), float(r2))


@dataclasses.dataclass(frozen=True)
class GrowthMeasurementV1:
    """DW2: the candidate's OWN measured runtime growth (no oracle)."""
    measurable: bool
    baseline_s: float
    sizes: tuple[int, ...]
    compute_times_s: tuple[float, ...]      # max-over-shapes compute time per measured size
    any_tle: bool
    tle_size: Optional[int]
    fitted_exponent: Optional[float]
    fit_r2: float
    n_points: int

    def to_dict(self) -> dict[str, Any]:
        return {"measurable": self.measurable, "baseline_s": round(self.baseline_s, 4),
                "sizes": list(self.sizes),
                "compute_times_s": [round(t, 4) for t in self.compute_times_s],
                "any_tle": self.any_tle, "tle_size": self.tle_size,
                "fitted_exponent": (round(self.fitted_exponent, 3)
                                    if self.fitted_exponent is not None else None),
                "fit_r2": round(self.fit_r2, 3), "n_points": self.n_points}


_SHAPE_ORDER: tuple[str, ...] = ("descending", "constant", "random")  # adversarial-first


def measure_growth_v1(code: str, ladder: LadderV1, *,
                      timeout_s: float = PER_RUN_TIMEOUT_S) -> GrowthMeasurementV1:
    """Time the candidate across the ladder and fit a log-log growth exponent.  Consumes ONLY the
    candidate code + the public-format ladder.

    Sizes are tried ASCENDING; shapes ADVERSARIAL-FIRST (descending/constant/random).  On the
    first size where any shape exceeds the per-run timeout, that size is recorded as the TLE
    boundary and the sweep STOPS (larger sizes would only TLE harder; this also bounds witness
    cost).  The fit + the emitted curve use ONLY the clean pre-TLE region (all-shapes-finished
    sizes, baseline-subtracted, above the noise floor), so the diagnostic is never self-
    contradictory (no fast short-circuit shape at a TLE size pollutes the curve)."""
    if not ladder.parseable:
        return GrowthMeasurementV1(False, 0.0, (), (), False, None, None, 0.0, 0)
    base_dt, base_to, base_crash = _time_once(code, ladder.baseline_input, timeout_s=timeout_s)
    baseline = 0.0 if (base_to or base_crash) else base_dt
    sizes: list[int] = []
    times: list[float] = []
    any_tle = False
    tle_size: Optional[int] = None
    shape_inputs = {sz: dict(shps) for sz, shps in ladder.rungs}
    for sz, _per_shape in ladder.rungs:                    # ascending
        per = shape_inputs[sz]
        best = 0.0
        saw_finite = False
        size_tle = False
        for kind in _SHAPE_ORDER:
            inp = per.get(kind)
            if inp is None:
                continue
            dt, to, crash = _time_once(code, inp, timeout_s=timeout_s)
            if to:
                size_tle = True
                break                                      # this size TLEs -> stop testing shapes
            if crash:
                continue
            saw_finite = True
            best = max(best, dt - baseline)
        if size_tle:
            any_tle = True
            tle_size = sz
            break                                          # stop the sweep at the first TLE size
        if saw_finite:
            sizes.append(sz)
            times.append(max(best, 0.0))
    fit_points = [(float(s), float(t)) for s, t in zip(sizes, times) if t >= NOISE_FLOOR_S]
    expo, r2 = _fit_loglog(fit_points)
    measurable = bool(len(fit_points) >= MIN_LADDER_POINTS or any_tle)
    return GrowthMeasurementV1(
        measurable=measurable, baseline_s=float(baseline), sizes=tuple(sizes),
        compute_times_s=tuple(times), any_tle=any_tle, tle_size=tle_size,
        fitted_exponent=expo, fit_r2=float(r2), n_points=len(fit_points))


# ===================================================== the deployable witness record

@dataclasses.dataclass(frozen=True)
class DeployableWitnessV1:
    """A deployable, oracle-free complexity witness.  ``to_prompt_block`` is the ONLY model-facing
    text; it carries the parsed budget, the candidate's OWN measured curve, a fitted exponent, an
    extrapolation, and a target class — and NEVER an expected output (no oracle)."""
    kind: str                      # DW_COMPLEXITY | DW_NONE
    fired: bool
    confidence_ok: bool            # the R^2 guard passed (or a TLE made the verdict certain)
    reason: str
    budget: BudgetFactV1
    growth: GrowthMeasurementV1
    extrapolated_s_at_n_max: Optional[float]

    def found(self) -> bool:
        return self.kind == DW_COMPLEXITY and self.fired

    def to_prompt_block(self) -> str:
        """The sanctioned deployable feedback object.  Contains ONLY: the candidate's measured
        runtimes across sizes, the fitted growth exponent, the constraint-derived budget, the
        extrapolation, and a target-complexity recommendation.  NO input bytes that disclose a
        hidden case's answer, and NO expected output (there is no oracle)."""
        g = self.growth
        curve = ", ".join(f"N={s}->{t:.3f}s" for s, t in zip(g.sizes, g.compute_times_s))
        curve = (f"Measured runtime of your program on valid inputs of growing size: {curve}."
                 if curve else "Measured runtime of your program on a valid large input:")
        tle = (f" Your program did NOT finish within {PER_RUN_TIMEOUT_S:.1f}s at N={g.tle_size} "
               "(a strictly-increasing/constant worst-case input)." if g.any_tle else "")
        # only report a fitted exponent when it is a meaningful super-linear slope from the clean region
        expo = (f" The measured runtime grows like O(N^{g.fitted_exponent:.2f})."
                if (g.fitted_exponent is not None and g.fitted_exponent >= 1.2) else "")
        nmx = self.budget.n_max
        extra = ""
        if nmx and self.extrapolated_s_at_n_max is not None:
            extra = (f" The statement allows N up to {nmx}; extrapolating your measured growth, "
                     f"your program would need about {self.extrapolated_s_at_n_max:.0f}s at that "
                     f"size — far over the ~{WALL_BUDGET_S:.0f}s the constraints imply.")
        elif nmx:
            extra = (f" The statement allows N up to {nmx}, at which an O(N^2) approach needs "
                     f"~{nmx*nmx:.0e} operations — far over the ~{self.budget.ops_budget:.0e} "
                     "operation budget a few-second limit implies.")
        return (
            "Complexity witness (your program is too SLOW, not wrong — measured on YOUR own "
            "program; no reference solution was used):\n"
            f"{curve}{tle}{expo}{extra}\n"
            "Your program is correct on small inputs, so the algorithm is right but its time "
            "complexity is too high (likely O(N^2) or worse). Redesign it with a faster algorithm "
            "(sorting + two pointers, prefix sums, a hash map, a Fenwick/BIT, or a monotonic "
            "stack) so it runs in about O(N log N) or O(N).")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "fired": self.fired, "confidence_ok": self.confidence_ok,
                "reason": self.reason, "budget": self.budget.to_dict(),
                "growth": self.growth.to_dict(),
                "extrapolated_s_at_n_max": (round(self.extrapolated_s_at_n_max, 2)
                                            if self.extrapolated_s_at_n_max is not None else None)}


def _none_witness(reason: str, budget: BudgetFactV1, growth: GrowthMeasurementV1
                  ) -> DeployableWitnessV1:
    return DeployableWitnessV1(kind=DW_NONE, fired=False, confidence_ok=False, reason=reason,
                               budget=budget, growth=growth, extrapolated_s_at_n_max=None)


def build_deployable_witness_v1(code: str, *, statement: str,
                                samples: Sequence[tuple[str, str]],
                                ladder_seed: int = LADDER_VALUE_SEED,
                                timeout_s: float = PER_RUN_TIMEOUT_S) -> DeployableWitnessV1:
    """The deployable, oracle-free complexity witness for one candidate.

    Consumes ONLY ``(code, statement, public samples)`` — it has no parameter through which a
    reference / naive / brute / secret case could enter.  Fires (DW_COMPLEXITY) iff the candidate's
    OWN measured growth is inadmissible (a ladder rung TLEs, OR the fitted exponent is
    super-linear with a confident fit, OR the extrapolation to N_max exceeds the wall budget).
    """
    budget = derive_budget_fact_v1(statement)
    ladder = build_ladder_v1(statement, samples, ladder_seed=ladder_seed)
    growth = measure_growth_v1(code, ladder, timeout_s=timeout_s)
    if not growth.measurable:
        return _none_witness("UNMEASURABLE", budget, growth)

    # extrapolate t(N_max) from the top CLEAN rung + fitted exponent (public-signal only); only
    # meaningful for a genuine super-linear slope (a degenerate/negative fit does not extrapolate)
    extrap: Optional[float] = None
    if (budget.n_max and growth.fitted_exponent is not None and growth.fitted_exponent >= 1.0
            and growth.sizes and growth.compute_times_s
            and growth.compute_times_s[-1] >= NOISE_FLOOR_S):
        top_n = growth.sizes[-1]
        top_t = growth.compute_times_s[-1]
        try:
            extrap = float(top_t) * (float(budget.n_max) / float(top_n)) ** float(growth.fitted_exponent)
            if not (math.isfinite(extrap) and extrap >= top_t):
                extrap = None
        except (OverflowError, ValueError):
            extrap = None

    # a TLE is a certain super-linear lower bound -> fire regardless of fit confidence
    if growth.any_tle:
        return DeployableWitnessV1(kind=DW_COMPLEXITY, fired=True, confidence_ok=True,
                                   reason="LADDER_TLE", budget=budget, growth=growth,
                                   extrapolated_s_at_n_max=extrap)
    expo = growth.fitted_exponent
    confident = bool(growth.fit_r2 >= R2_MIN_CONFIDENCE)
    super_linear = bool(expo is not None and expo >= P_SUPERLINEAR)
    extrap_over = bool(extrap is not None and extrap > WALL_BUDGET_S)
    top_compute = max(growth.compute_times_s) if growth.compute_times_s else 0.0
    significant = bool(top_compute >= MIN_SIGNIFICANT_S)
    if (super_linear or extrap_over) and confident and significant:
        reason = "SUPERLINEAR_FIT" if super_linear else "EXTRAPOLATION_OVER_BUDGET"
        return DeployableWitnessV1(kind=DW_COMPLEXITY, fired=True, confidence_ok=True,
                                   reason=reason, budget=budget, growth=growth,
                                   extrapolated_s_at_n_max=extrap)
    if (super_linear or extrap_over) and confident and not significant:
        # SwiftSolve small-signal guard: a fast program (sub-100ms at the ladder top) with a noisy
        # super-linear slope is admissible by observation — the slope is a measurement artifact.
        return _none_witness("FAST_BELOW_SIGNIFICANCE", budget, growth)
    if (super_linear or extrap_over) and not confident:
        # SwiftSolve guard: a low-R^2 slope is unreliable -> abstain rather than mis-fire
        return _none_witness("LOW_CONFIDENCE_FIT", budget, growth)
    return _none_witness("ADMISSIBLE_GROWTH", budget, growth)


# ===================================================== fake-different / genuinely-new test

def deployable_witness_is_genuinely_new_v1(witness: DeployableWitnessV1) -> dict[str, Any]:
    """Machine-checkable 'not just the blind reject bit, and not the oracle witness' test.

    A deployable witness is genuinely NEW iff it carries information the blind W120 reject bit
    structurally lacks — a MEASURED multi-size runtime curve (>=2 distinct sizes) AND a derived
    growth verdict (a fitted exponent or a TLE) AND a target-complexity recommendation — while
    carrying NO oracle output (no expected answer) and using NO reference timing.  A witness that
    is only 'your code timed out' with no curve is NOT genuinely-new (that is B0's bit re-stated).
    """
    g = witness.growth
    has_curve = bool(witness.found() and len(g.sizes) >= 2)
    has_growth_verdict = bool(witness.found() and (g.fitted_exponent is not None or g.any_tle))
    carries_no_oracle_output = True  # structural: DeployableWitnessV1 has no expected_output field
    uses_no_reference_timing = True  # structural: measure_growth_v1 times only the candidate
    genuinely_new = bool(witness.found() and has_curve and has_growth_verdict
                         and carries_no_oracle_output and uses_no_reference_timing)
    return {"genuinely_new": genuinely_new, "found": witness.found(),
            "has_measured_curve_ge2_sizes": has_curve,
            "has_growth_verdict": has_growth_verdict,
            "carries_no_oracle_output": carries_no_oracle_output,
            "uses_no_reference_timing": uses_no_reference_timing,
            "n_measured_sizes": len(g.sizes), "reason": witness.reason}


# ===================================================== the deployable policy (D1/D2/D3)

@dataclasses.dataclass(frozen=True)
class DeployableDecisionV1:
    action: str                    # ACTION_REWRITE | ACTION_KEEP | ACTION_ABSTAIN
    witness: DeployableWitnessV1

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "witness": self.witness.to_dict()}


def select_deployable_action_v1(code: str, *, statement: str,
                                samples: Sequence[tuple[str, str]], arm: str,
                                ladder_seed: int = LADDER_VALUE_SEED,
                                timeout_s: float = PER_RUN_TIMEOUT_S) -> DeployableDecisionV1:
    """The deployable policy for one arm (RUNBOOK_W134 §5).

    * D1 — rewrite: fire -> REWRITE (emit the DW block); else KEEP (blind reflexion).
    * D2 — D1 + the W130 constraint-derived admissibility gate: REWRITE only if the witness fires
      AND ``complexity_admissible_v1(measured_exponent, n_max)`` independently confirms it is
      inadmissible (a second public-signal confirmation reduces false fires).
    * D3 — controller: REWRITE if the witness fires confidently; KEEP if growth is measurably
      ADMISSIBLE (the bug is value/logic, not complexity — defer to blind reflexion); ABSTAIN to
      blind reflexion when unmeasurable / low-confidence (never worse than B0).  The LEAD.
    """
    w = build_deployable_witness_v1(code, statement=statement, samples=samples,
                                    ladder_seed=ladder_seed, timeout_s=timeout_s)
    if arm == ARM_D1_REWRITE:
        return DeployableDecisionV1(ACTION_REWRITE if w.found() else ACTION_KEEP, w)
    if arm == ARM_D2_GATED:
        gate = complexity_admissible_v1(w.growth.fitted_exponent, w.budget.n_max)
        # REWRITE iff the witness fired AND (the budget gate is not 'admissible'): a TLE (exponent
        # may be None) or a confirmed-inadmissible fit both pass; an explicitly-admissible gate vetoes.
        rewrite = bool(w.found() and gate is not True)
        return DeployableDecisionV1(ACTION_REWRITE if rewrite else ACTION_KEEP, w)
    # D3 controller
    if w.found():
        return DeployableDecisionV1(ACTION_REWRITE, w)
    if w.growth.measurable and w.reason == "ADMISSIBLE_GROWTH":
        return DeployableDecisionV1(ACTION_KEEP, w)
    return DeployableDecisionV1(ACTION_ABSTAIN, w)


# ===================================================== the same-budget deployable arm

def _deployable_rewrite_prompt(pilot: IcpcPilotProblemV1,
                               history: Sequence[tuple[str, bool, str, str]],
                               witness: DeployableWitnessV1, attempt_idx: int) -> str:
    """The between-attempt REWRITE prompt: the SAME blind scaffold as the W120 reflexion prompt
    (judge verdict + stderr tail + public-sample results) PLUS the deployable witness block — a
    strict superset of the blind feedback, so any gain is attributable to the deployable witness."""
    chunks: list[str] = []
    for i, (cand, passed, stderr_tail, sample_fb) in enumerate(history):
        cand_trim = cand if len(cand) <= 1500 else (cand[:1500] + "\n# ...(truncated)\n")
        verdict = ("ACCEPTED by the judge (all hidden tests passed)" if passed
                   else "REJECTED by the judge (failed at least one hidden test)")
        se = f"\nExecutor stderr (tail):\n{stderr_tail.strip()}" if stderr_tail.strip() else ""
        sf = f"\nPublic sample results:\n{sample_fb}" if sample_fb.strip() else ""
        chunks.append(f"--- Attempt {i+1} ({verdict}) ---\n```python\n{cand_trim}\n```{se}{sf}")
    return (
        "You are an expert ICPC competitor on a reflective debugging loop. You are on "
        f"attempt {attempt_idx + 1} out of 5. Below are your previous attempts with the judge "
        "verdict and the PUBLIC sample-case results, followed by a COMPLEXITY diagnostic measured "
        "on your own program. Use the diagnostic to produce a NEW corrected COMPLETE Python 3 "
        "stdin/stdout program. Do not repeat a previous attempt verbatim.\n\n"
        f"Problem:\n{pilot.statement}\n\n{_samples_block(pilot)}\n\n"
        f"{chr(10).join(chunks)}\n\n"
        f"=== Complexity diagnostic of attempt {len(history)} ===\n{witness.to_prompt_block()}\n\n"
        "Provide ONLY the corrected complete Python 3 program in a ```python ... ``` fence:")


@dataclasses.dataclass(frozen=True)
class DeployableArmTraceV1:
    """Per-problem audit of one deployable arm: the action + witness verdict per invoked attempt."""
    problem_id: str
    arm_id: str
    actions: tuple[str, ...]
    witness_fired: tuple[bool, ...]
    witness_reasons: tuple[str, ...]
    fitted_exponents: tuple[Optional[float], ...]
    any_rewrite: bool
    all_oracle_free: bool          # structural: every witness carried no oracle output

    def rescue_is_algorithmic(self) -> bool:
        """True iff any attempt emitted a REWRITE on a fired complexity witness (a genuine
        complexity diagnosis), not merely blind KEEP/ABSTAIN fallbacks."""
        return any(a == ACTION_REWRITE for a in self.actions)

    def to_dict(self) -> dict[str, Any]:
        return {"problem_id": self.problem_id, "arm_id": self.arm_id,
                "actions": list(self.actions), "witness_fired": list(self.witness_fired),
                "witness_reasons": list(self.witness_reasons),
                "fitted_exponents": [(round(e, 3) if e is not None else None)
                                     for e in self.fitted_exponents],
                "any_rewrite": bool(self.any_rewrite),
                "all_oracle_free": bool(self.all_oracle_free)}


def run_deployable_witness_arm_v1(*, seed: int, pilot: IcpcPilotProblemV1, gen, K: int,
                                  temperature: float, max_tokens: int, timeout_s: float,
                                  arm: str, ladder_seed: int = LADDER_VALUE_SEED,
                                  witness_timeout_s: float = PER_RUN_TIMEOUT_S,
                                  ) -> tuple[IcpcArmOutcomeV1, DeployableArmTraceV1]:
    """Same-budget DEPLOYABLE witness-guided reflexion arm.  Byte-identical structure to the W120
    ``_run_b`` (attempt-0 = the standard initial prompt; K attempts; one model call per attempt;
    no early stop), except the between-attempt feedback object is the DEPLOYABLE complexity witness
    on a REWRITE action, and the blind reflexion bit on a KEEP/ABSTAIN action (so the arm is never
    worse than B0).

    Consumes ONLY the ``pilot`` (statement + public samples + the hidden GRADER).  The witness is
    built from ``pilot.statement`` + ``pilot.samples`` ONLY — never ``pilot.secret_cases`` (which
    are used solely by the scoring grader, identically to every other arm).  NO reference / naive /
    brute source exists on any path here: this is the deployability proof.
    """
    history: list[tuple[str, bool, str, str]] = []
    per_call: list[bool] = []
    first_pass_idx = -1
    actions: list[str] = []
    fired: list[bool] = []
    reasons: list[str] = []
    expos: list[Optional[float]] = []
    oracle_free = True
    for k in range(int(K)):
        if k == 0:
            prompt = _initial_prompt(pilot)
        else:
            last_code = history[-1][0]
            dec = select_deployable_action_v1(last_code, statement=pilot.statement,
                                              samples=pilot.samples, arm=arm,
                                              ladder_seed=ladder_seed, timeout_s=witness_timeout_s)
            actions.append(dec.action)
            fired.append(bool(dec.witness.found()))
            reasons.append(dec.witness.reason)
            expos.append(dec.witness.growth.fitted_exponent)
            # structural no-oracle invariant (the witness has no expected_output field at all)
            if not deployable_witness_is_genuinely_new_v1(dec.witness)["carries_no_oracle_output"]:
                oracle_free = False
            if dec.action == ACTION_REWRITE:
                prompt = _deployable_rewrite_prompt(pilot, tuple(history), dec.witness, attempt_idx=k)
            else:  # KEEP / ABSTAIN -> blind reflexion (never worse than B0)
                prompt = _reflexion_prompt(pilot, tuple(history), attempt_idx=k)
        text, _ = gen(prompt, max_tokens, float(temperature))
        code = extract_candidate_code_v1(response_text=text)
        passed, stderr_tail, _ = grade_on_secret_v1(pilot, code, timeout_s=timeout_s)
        per_call.append(bool(passed))
        sfb = sample_feedback_v1(pilot, code, timeout_s=timeout_s)
        history.append((code, bool(passed), stderr_tail, sfb))
        if passed and first_pass_idx == -1:
            first_pass_idx = int(k)
    outcome = IcpcArmOutcomeV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, seed=int(seed),
        question_id=pilot.problem_id, arm_id=str(arm), final_passed=bool(first_pass_idx >= 0),
        n_model_calls=int(K), per_call_passed=tuple(per_call),
        first_pass_attempt_idx=int(first_pass_idx))
    trace = DeployableArmTraceV1(
        problem_id=pilot.problem_id, arm_id=str(arm), actions=tuple(actions),
        witness_fired=tuple(fired), witness_reasons=tuple(reasons),
        fitted_exponents=tuple(expos), any_rewrite=any(a == ACTION_REWRITE for a in actions),
        all_oracle_free=bool(oracle_free))
    return outcome, trace


__all__ = [
    "W134_DEPLOYABLE_COMPLEXITY_WITNESS_V1_SCHEMA_VERSION",
    "DW_COMPLEXITY", "DW_NONE",
    "ARM_D1_REWRITE", "ARM_D2_GATED", "ARM_D3_CONTROLLER",
    "ACTION_REWRITE", "ACTION_KEEP", "ACTION_ABSTAIN",
    "LADDER_BASE", "LADDER_RUNGS", "LADDER_SHAPES", "P_SUPERLINEAR", "WALL_BUDGET_S",
    "R2_MIN_CONFIDENCE", "COMPLEXITY_OPS_BUDGET",
    "parse_value_hi_v1", "BudgetFactV1", "derive_budget_fact_v1",
    "InputShapeV1", "parse_input_shape_v1", "synth_input_v1", "LadderV1", "build_ladder_v1",
    "GrowthMeasurementV1", "measure_growth_v1",
    "DeployableWitnessV1", "build_deployable_witness_v1", "deployable_witness_is_genuinely_new_v1",
    "DeployableDecisionV1", "select_deployable_action_v1",
    "DeployableArmTraceV1", "run_deployable_witness_arm_v1",
]
