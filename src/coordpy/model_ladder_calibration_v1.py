"""W137 / COO-9 — model-ladder hardness calibration (HC3 + HC4).

W136's decisive lesson: once the I/O confound is removed, the W132 textbook-DP traps are ONE-SHOT for
a 70B model — zero feedback-mechanism headroom.  A battlefield is only a valid hard algorithm test if
its items are *hard but not impossible* at the target scale and they DISCRIMINATE between model tiers.
Per Lane-γ primary-source research — tinyBenchmarks (arXiv:2402.14992) fits item difficulty +
discrimination; "Lost in Benchmarks" (arXiv:2505.15055) shows most items have near-zero discrimination
and must be culled — the selection criterion is **discrimination, not raw difficulty**.

This module runs single-shot A0 (and a small multi-sample A1) across a model ladder and admits the
**headroom band**:

* **HC3 — strong-anchor A0 headroom**: reject a template the strong anchor ONE-SHOTS (A0 pass-rate
  >= ``hc3_ceiling``) — a saturated item has no room for any mechanism to add value (the W136 failure).
* **HC4 — floor / not universally dead**: reject a template on which the strong anchor passes NOTHING
  even with K samples (A0 == 0 AND A1 == 0) — capability-dead at this scale, no mechanism-sensitive
  headroom (the W128–W131 generation ceiling).

A surviving template is in the mid-difficulty, mechanism-sensitive band.  The discrimination signal
(small-tier vs strong-tier pass-rate gap) is recorded for the IRT report but is not hard-required
(a template both tiers fail single-shot yet the strong tier solves with samples is genuine headroom).

The module is gen-AGNOSTIC: it takes ``gen_for_model(model_id) -> gen`` so the driver script wires
the NIM client (or any ``Callable[[str,int,float], tuple[str,int]]``).  Calibration instances are
minted from a seed range DISJOINT from the train/dev/eval/frontier splits, so calibration spends no
budget against the graded splits.  Explicit-import only; ``coordpy/__init__.py`` untouched.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional, Sequence

from .hard_battlefield_slate_v2 import ParserNeutralTemplateV2, build_hard_slate_v2
from .resistant_by_construction_battlefield_v1 import _sha256_hex, mint_problem_v1
from .icpc_reflexion_bench_v1 import (
    _initial_prompt, extract_candidate_code_v1, grade_on_secret_v1)

MODEL_LADDER_CALIBRATION_V1_SCHEMA_VERSION: str = "coordpy.model_ladder_calibration_v1.v1"

# calibration seeds: DISJOINT from the corpus split seed ranges (137_1xx..137_4xx in the corpus)
CALIBRATION_SEED_BASE: int = 137_900_000

# A ``gen`` matches the bench contract: gen(prompt, max_tokens, temperature) -> (text, wall_ms).
GenFn = Callable[[str, int, float], "tuple[str, int]"]


@dataclasses.dataclass(frozen=True)
class CalibrationModelV1:
    model_id: str
    tier: str            # "small" | "strong"
    is_anchor: bool      # the frontier anchor (the strong tier the earn rule references)


# default ladder: one smaller reachable dev model + the frontier anchor (the W105 retirement model)
LADDER_V1: tuple[CalibrationModelV1, ...] = (
    CalibrationModelV1(model_id="meta/llama-3.1-8b-instruct", tier="small", is_anchor=False),
    CalibrationModelV1(model_id="meta/llama-3.3-70b-instruct", tier="strong", is_anchor=True),
)


@dataclasses.dataclass(frozen=True)
class ModelTemplateStatV1:
    model_id: str
    tier: str
    a0_passed: tuple[bool, ...]      # one per calibration instance (single-shot, temp 0)
    a1_passed: tuple[bool, ...]      # one per A1 instance (any-of-K, temp 0.7); may be empty
    n_calls: int

    @property
    def a0_rate(self) -> float:
        return sum(self.a0_passed) / len(self.a0_passed) if self.a0_passed else 0.0

    @property
    def a1_rate(self) -> float:
        return sum(self.a1_passed) / len(self.a1_passed) if self.a1_passed else 0.0

    @property
    def best_rate(self) -> float:
        return max(self.a0_rate, self.a1_rate)

    def to_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "tier": self.tier,
                "a0_passed": list(self.a0_passed), "a1_passed": list(self.a1_passed),
                "a0_rate": round(self.a0_rate, 4), "a1_rate": round(self.a1_rate, 4),
                "n_calls": int(self.n_calls)}


@dataclasses.dataclass(frozen=True)
class TemplateCalibrationV1:
    template_name: str
    family: str
    mode: str
    per_model: tuple[ModelTemplateStatV1, ...]
    # HC verdicts
    hc3_has_headroom: bool           # strong anchor A0 not saturated
    hc4_not_dead: bool               # strong anchor passes something (A0 or A1 > 0)
    discriminates: bool              # strong best_rate > small best_rate (IRT signal)
    admitted: bool                   # HC3 ∧ HC4
    reason: str

    def anchor(self) -> Optional[ModelTemplateStatV1]:
        return next((m for m in self.per_model if m.tier == "strong"), None)

    def to_dict(self) -> dict[str, Any]:
        return {"template_name": self.template_name, "family": self.family, "mode": self.mode,
                "per_model": [m.to_dict() for m in self.per_model],
                "hc3_has_headroom": bool(self.hc3_has_headroom),
                "hc4_not_dead": bool(self.hc4_not_dead),
                "discriminates": bool(self.discriminates),
                "admitted": bool(self.admitted), "reason": self.reason}


def _run_a0(pilot, gen: GenFn, *, max_tokens: int, timeout_s: float) -> bool:
    text, _ = gen(_initial_prompt(pilot), int(max_tokens), 0.0)
    code = extract_candidate_code_v1(response_text=text)
    passed, _, _ = grade_on_secret_v1(pilot, code, timeout_s=float(timeout_s))
    return bool(passed)


def _run_a1(pilot, gen: GenFn, *, K: int, max_tokens: int, timeout_s: float) -> bool:
    for _ in range(int(K)):
        text, _ = gen(_initial_prompt(pilot), int(max_tokens), 0.7)
        code = extract_candidate_code_v1(response_text=text)
        passed, _, _ = grade_on_secret_v1(pilot, code, timeout_s=float(timeout_s))
        if passed:
            return True
    return False


def calibrate_template_v1(template: ParserNeutralTemplateV2, *,
                          gen_for_model: Callable[[str], GenFn],
                          ladder: Sequence[CalibrationModelV1] = LADDER_V1,
                          n_a0: int = 3, n_a1: int = 1, K_a1: int = 3,
                          hc3_ceiling: float = 0.80, max_tokens: int = 1536,
                          timeout_s: float = 8.0, mint_timeout_s: Optional[float] = None,
                          minted_date: str = "2026-06-04",
                          on_call: Optional[Callable[[str, str, int], None]] = None,
                          ) -> TemplateCalibrationV1:
    """Calibrate one template across the ladder.  Mints ``n_a0`` calibration instances (disjoint
    seeds), runs A0 at each model and A1(K) at the strong anchor only, then applies HC3 ∧ HC4.

    ``mint_timeout_s`` (default = ``timeout_s``) bounds the answer-key/gate subprocesses at MINT
    time; model code is always graded at the locked ``timeout_s``.  Minting is timeout-invariant
    (the naive TLEs and the ref finishes under both), so a shorter mint timeout only speeds the
    $0 build — it does not change the graded budget."""
    mt = float(mint_timeout_s) if mint_timeout_s is not None else float(timeout_s)
    # mint disjoint calibration instances
    pilots = []
    for r in range(int(n_a0)):
        seed = CALIBRATION_SEED_BASE + r
        mp = mint_problem_v1(template.minted, global_seed=seed, timeout_s=mt)
        pilots.append(mp.to_pilot_problem(minted_date=minted_date))

    per_model: list[ModelTemplateStatV1] = []
    for cm in ladder:
        gen = gen_for_model(cm.model_id)
        a0 = []
        for i, p in enumerate(pilots):
            if on_call:
                on_call(template.minted.name, cm.model_id, i)
            a0.append(_run_a0(p, gen, max_tokens=max_tokens, timeout_s=timeout_s))
        a1: list[bool] = []
        calls = len(pilots)
        if cm.is_anchor and n_a1 > 0:
            for p in pilots[:int(n_a1)]:
                a1.append(_run_a1(p, gen, K=K_a1, max_tokens=max_tokens, timeout_s=timeout_s))
                calls += int(K_a1)
        per_model.append(ModelTemplateStatV1(
            model_id=cm.model_id, tier=cm.tier, a0_passed=tuple(a0), a1_passed=tuple(a1),
            n_calls=calls))

    strong = next((m for m in per_model if m.tier == "strong"), per_model[-1])
    small = next((m for m in per_model if m.tier == "small"), per_model[0])
    hc3 = bool(strong.a0_rate < float(hc3_ceiling))            # not one-shot-saturated
    hc4 = bool(strong.best_rate > 0.0)                         # solvable at strong scale (not dead)
    disc = bool(strong.best_rate > small.best_rate)
    admitted = bool(hc3 and hc4)
    if admitted:
        reason = "ADMITTED_HEADROOM_BAND"
    elif not hc3:
        reason = f"HC3_SATURATED(strong_a0={strong.a0_rate:.2f}>={hc3_ceiling})"
    else:
        reason = f"HC4_DEAD(strong_best={strong.best_rate:.2f}==0)"
    return TemplateCalibrationV1(
        template_name=template.minted.name, family=template.minted.family,
        mode=template.minted.mode, per_model=tuple(per_model),
        hc3_has_headroom=hc3, hc4_not_dead=hc4, discriminates=disc, admitted=admitted,
        reason=reason)


@dataclasses.dataclass(frozen=True)
class CalibrationReportV1:
    schema: str
    ladder: tuple[str, ...]
    hc3_ceiling: float
    per_template: tuple[TemplateCalibrationV1, ...]

    def surviving_template_names(self) -> tuple[str, ...]:
        return tuple(t.template_name for t in self.per_template if t.admitted)

    def calibration_cid(self) -> str:
        return _sha256_hex({"k": "w137_calibration_v1", "ladder": list(self.ladder),
                            "hc3_ceiling": self.hc3_ceiling,
                            "verdicts": {t.template_name: t.admitted for t in self.per_template}})

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "ladder": list(self.ladder),
                "hc3_ceiling": self.hc3_ceiling,
                "n_admitted": len(self.surviving_template_names()),
                "surviving": list(self.surviving_template_names()),
                "calibration_cid": self.calibration_cid(),
                "per_template": [t.to_dict() for t in self.per_template]}


def run_calibration_v1(*, gen_for_model: Callable[[str], GenFn],
                       templates: Optional[Sequence[ParserNeutralTemplateV2]] = None,
                       ladder: Sequence[CalibrationModelV1] = LADDER_V1,
                       n_a0: int = 3, n_a1: int = 1, K_a1: int = 3, hc3_ceiling: float = 0.80,
                       max_tokens: int = 1536, timeout_s: float = 8.0,
                       mint_timeout_s: Optional[float] = None,
                       minted_date: str = "2026-06-04",
                       on_call: Optional[Callable[[str, str, int], None]] = None,
                       ) -> CalibrationReportV1:
    tpls = list(templates) if templates is not None else build_hard_slate_v2()
    recs = [calibrate_template_v1(
        t, gen_for_model=gen_for_model, ladder=ladder, n_a0=n_a0, n_a1=n_a1, K_a1=K_a1,
        hc3_ceiling=hc3_ceiling, max_tokens=max_tokens, timeout_s=timeout_s,
        mint_timeout_s=mint_timeout_s, minted_date=minted_date, on_call=on_call) for t in tpls]
    return CalibrationReportV1(
        schema=MODEL_LADDER_CALIBRATION_V1_SCHEMA_VERSION,
        ladder=tuple(cm.model_id for cm in ladder), hc3_ceiling=float(hc3_ceiling),
        per_template=tuple(recs))


__all__ = [
    "MODEL_LADDER_CALIBRATION_V1_SCHEMA_VERSION", "CALIBRATION_SEED_BASE", "GenFn",
    "CalibrationModelV1", "LADDER_V1", "ModelTemplateStatV1", "TemplateCalibrationV1",
    "calibrate_template_v1", "CalibrationReportV1", "run_calibration_v1",
]
