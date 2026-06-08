"""W142 / COO-9 — moderate-`p` family screen (Lane α core).

Decides, per candidate family, whether it is an ADMITTED moderate-`p` retirement-supply family, under the
locked RUNBOOK_W142 rule.  The screen runs the $0 gates FIRST (reject before any NIM), then measures the
FAIR raw efficient-rate `p` over a NEUTRAL-PROMPT BANK on the frontier anchor, then applies the band rule:

  G1 parser-neutral  ∧  G2 exact-oracle discriminating  ∧  G3 gated-accumulator extractable  ∧
  G4 novelty/near-dup  ∧  G5 fair `p ∈ [0.10,0.50]` with Wilson-95% excluding 0 and 1.

G1–G4 are $0 (no model).  G5 is the only NIM gate.  ``p`` is measured under the TRULY-NEUTRAL
self-consistency prompt bank (§2 / §13.3: names NO technique, NO efficiency/time-limit/largest-input cue,
NO data structure) — a hidden-bank pass ⇒ correct AND efficient (the naive TLEs/wrong-answers on hidden by
construction), so ``p̂ = passes / K`` is the model's intrinsic efficient-solution rate.  Per FormatSpread
(arXiv:2310.11324) `p` is reported as the MEDIAN over the phrasing bank + the spread; the band screen uses
the median.  The IRT peak-Fisher-information framing (metabench arXiv:2407.12844) justifies the band:
discrimination is maximal at `p≈0.5`, zero at the bimodal extremes.

The decisive $0 result: G3 (gated-accumulator extractability) REJECTS the prefix-hash + binary-search-on-
answer veins (the technique lives in dict-maintenance / the answer is not an accumulator), machine-checking
the W142 de-risk finding.  ``grade_on_secret_v1`` is SCORING-ONLY (never inside the mechanism).  Pure /
deterministic except the audited execution + the model gen; explicit-import only; ``__init__.py`` untouched.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional, Sequence

from .resistant_by_construction_battlefield_v1 import MintedProblemV1, mint_problem_v1, _exec_capture_v1
from .hard_battlefield_slate_v2 import ParserNeutralTemplateV2
from .icpc_reflexion_bench_v1 import extract_candidate_code_v1, grade_on_secret_v1
from .parser_neutral_io_v1 import parser_neutrality_gate_v1
from .headroom_band_calibration_v2 import wilson_interval_v1
from .self_tutoring_technique_extractor_v1 import compile_tutor_from_winner_v1
from .moderate_p_family_slate_v1 import (
    ScreenCandidateV1, MODERATE_P_LO, MODERATE_P_HI, build_screen_slate_v1)

MODERATE_P_FAMILY_SCREEN_V1_SCHEMA_VERSION: str = "coordpy.moderate_p_family_screen_v1.v1"

GenFn = Callable[[str, int, float], Any]
MINTED_DATE: str = "2026-06-08"


# ----------------------------------------------------------------- fair neutral-prompt bank (§2/§13.3)

def _fnb_canonical(statement: str) -> str:
    """The exact W141-v4 truly-neutral self-consistency prompt (byte-identical to the retirement
    baseline)."""
    return (f"{statement}\n\n"
            "Write a Python 3 program that solves this problem. Read all input from stdin and write the "
            "answer to stdout in the exact format shown. Return ONLY one ```python code block.")


def _fnb_variant(statement: str) -> str:
    """A second neutral phrasing (FormatSpread robustness).  Names NO technique, NO efficiency/time/size
    cue, NO data structure — only a benign rewording of the same task framing."""
    return (f"{statement}\n\n"
            "Provide a complete Python 3 solution for the task above. It should read the input from "
            "standard input and print the required output in the exact format described. Respond with "
            "exactly one ```python code block and nothing else.")


FNB_PROMPTS: tuple[Callable[[str], str], ...] = (_fnb_canonical, _fnb_variant)


def _gen_text(gen: GenFn, prompt: str, max_tokens: int, temperature: float) -> str:
    out = gen(prompt, max_tokens, temperature)
    if isinstance(out, tuple):
        return out[0] or ""
    return out or ""


def _passes_secret(problem: MintedProblemV1, code: str, timeout_s: float) -> bool:
    """SCORING ONLY — grade on the hidden bank.  NEVER inside the mechanism."""
    if not code.strip():
        return False
    pilot = problem.to_pilot_problem(minted_date=MINTED_DATE)
    passed, _stderr, _n = grade_on_secret_v1(pilot, code, timeout_s=timeout_s)
    return bool(passed)


# ----------------------------------------------------------------- $0 gates (no NIM)

@dataclasses.dataclass(frozen=True)
class Dollar0GatesV1:
    g1_parser_neutral: bool
    g2_discriminating: bool        # ref passes hidden AND naive FAILS hidden
    g3_extractable: bool           # compiled tutor AND n_pred_holes >= 1 (gated accumulator)
    g4_novel: bool
    n_pred_holes: int
    n_add_holes: int
    g3_reason: str
    all_pass: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def dollar0_gates_v1(cand: ScreenCandidateV1, *, timeout_s: float = 6.0,
                     known_algo_sigs: Sequence[str] = (), known_statements: Sequence[str] = (),
                     teacher_seed: int = 1) -> Dollar0GatesV1:
    """Run the four $0 admission gates on a candidate's CANONICAL reference (no model)."""
    template = cand.factory(cand.knob)
    minted = template.minted
    problem = mint_problem_v1(minted, global_seed=teacher_seed)
    pilot = problem.to_pilot_problem(minted_date=MINTED_DATE)

    # G1 parser-neutrality on all (public + secret) inputs
    inputs = [inp for inp, _ in problem.samples] + [inp for inp, _ in problem.secret_cases]
    try:
        rec = parser_neutrality_gate_v1(inputs, template.io_shape)
        g1 = bool(rec.is_parser_neutral)
    except Exception:  # noqa: BLE001
        g1 = False

    # G2 exact-oracle discriminating: ref passes the hidden bank, naive FAILS it
    ref_pass = _passes_secret(problem, minted.ref_source, timeout_s)
    naive_fail = not _passes_secret(problem, minted.naive_source, timeout_s)
    g2 = bool(ref_pass and naive_fail)

    # G3 gated-accumulator extractability: the ref must compile a clean leak-passing tutor whose holed
    # skeleton GATES the accumulator (n_pred_holes >= 1) — the prefix-hash/BSoA controls fail here.
    tutor, rep = compile_tutor_from_winner_v1(minted.ref_source, template, problem, timeout_s=timeout_s)
    n_pred = rep.extracted.n_pred_holes if rep.extracted else 0
    n_add = rep.extracted.n_add_holes if rep.extracted else 0
    g3 = bool(rep.compiled and tutor is not None and n_pred >= 1)
    g3_reason = rep.reason if not rep.compiled else (f"compiled_n_pred={n_pred}")

    # G4 novelty: distinct algo_sig + statement vs the existing battlefield slate and the other candidates
    g4 = bool(minted.algo_sig not in set(known_algo_sigs)
              and minted.statement not in set(known_statements))

    all_pass = bool(g1 and g2 and g3 and g4)
    return Dollar0GatesV1(g1, g2, g3, g4, n_pred, n_add, g3_reason, all_pass)


# ----------------------------------------------------------------- G5 fair-p measurement (the only NIM gate)

@dataclasses.dataclass(frozen=True)
class FairPResultV1:
    prompt_idx: int
    n: int
    passes: int
    pass_flags: tuple[bool, ...]

    @property
    def p_hat(self) -> float:
        return (self.passes / self.n) if self.n else 0.0


def measure_fair_p_v1(problem: MintedProblemV1, *, gen: GenFn, K: int, prompt_idx: int = 0,
                      temperature: float = 0.7, max_tokens: int = 1536, timeout_s: float = 4.0
                      ) -> FairPResultV1:
    """Generate K candidates under FNB phrasing ``prompt_idx`` and grade each on the hidden bank
    (SCORING ONLY).  ``p̂`` = fraction passing = the model's fair intrinsic efficient-solution rate."""
    pilot = problem.to_pilot_problem(minted_date=MINTED_DATE)
    builder = FNB_PROMPTS[prompt_idx % len(FNB_PROMPTS)]
    prompt = builder(pilot.statement)
    flags: list[bool] = []
    for _ in range(K):
        code = extract_candidate_code_v1(response_text=_gen_text(gen, prompt, max_tokens, temperature))
        flags.append(_passes_secret(problem, code, timeout_s))
    return FairPResultV1(prompt_idx=prompt_idx, n=len(flags), passes=sum(flags),
                         pass_flags=tuple(flags))


@dataclasses.dataclass(frozen=True)
class FamilyScreenResultV1:
    family: str
    vein: str
    mode: str
    knob: int
    gates: Dollar0GatesV1
    per_prompt: tuple[FairPResultV1, ...]
    p_median: float
    p_min: float
    p_max: float
    wilson_lo: float
    wilson_hi: float
    n_total: int
    passes_total: int
    in_band: bool                  # p_median in [LO,HI] AND Wilson excludes 0 and 1
    admitted: bool                 # all $0 gates AND in_band
    expect_extractable: bool

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["gates"] = self.gates.to_dict()
        d["per_prompt"] = [dataclasses.asdict(p) for p in self.per_prompt]
        return d


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def screen_family_v1(cand: ScreenCandidateV1, *, gen: Optional[GenFn], K_screen: int = 12,
                     prompt_indices: Sequence[int] = (0, 1), temperature: float = 0.7,
                     max_tokens: int = 1536, timeout_s: float = 4.0, teacher_seed: int = 1,
                     known_algo_sigs: Sequence[str] = (), known_statements: Sequence[str] = (),
                     band_lo: float = MODERATE_P_LO, band_hi: float = MODERATE_P_HI
                     ) -> FamilyScreenResultV1:
    """Full per-family screen: $0 gates, then (if gen and gates pass) fair-`p` over the prompt bank +
    Wilson + the band verdict.  If ``gen`` is None or the $0 gates fail, p-measurement is SKIPPED
    (admitted=False) — the $0 reject is free."""
    template = cand.factory(cand.knob)
    gates = dollar0_gates_v1(cand, timeout_s=max(timeout_s, 6.0), known_algo_sigs=known_algo_sigs,
                             known_statements=known_statements, teacher_seed=teacher_seed)
    per_prompt: list[FairPResultV1] = []
    if gen is not None and gates.all_pass:
        problem = mint_problem_v1(template.minted, global_seed=teacher_seed)
        for pi in prompt_indices:
            per_prompt.append(measure_fair_p_v1(problem, gen=gen, K=K_screen, prompt_idx=pi,
                                                temperature=temperature, max_tokens=max_tokens,
                                                timeout_s=timeout_s))
    ps = [r.p_hat for r in per_prompt]
    p_med = _median(ps) if ps else 0.0
    n_total = sum(r.n for r in per_prompt)
    passes_total = sum(r.passes for r in per_prompt)
    lo, hi = wilson_interval_v1(passes_total, n_total) if n_total else (0.0, 1.0)
    in_band = bool(per_prompt and band_lo <= p_med <= band_hi and lo > 0.0 and hi < 1.0)
    admitted = bool(gates.all_pass and in_band)
    return FamilyScreenResultV1(
        family=cand.family, vein=cand.vein, mode=cand.mode, knob=cand.knob, gates=gates,
        per_prompt=tuple(per_prompt), p_median=round(p_med, 4),
        p_min=round(min(ps), 4) if ps else 0.0, p_max=round(max(ps), 4) if ps else 0.0,
        wilson_lo=round(lo, 4), wilson_hi=round(hi, 4), n_total=n_total, passes_total=passes_total,
        in_band=in_band, admitted=admitted, expect_extractable=cand.expect_extractable)


@dataclasses.dataclass(frozen=True)
class ScreenVerdictV1:
    n_candidates: int
    n_dollar0_pass: int
    n_admitted: int
    admitted_families: tuple[str, ...]
    admitted_veins: tuple[str, ...]
    admitted_modes: tuple[str, ...]
    g3_predictions_correct: bool   # every expect_extractable matched the actual G3 verdict
    span_ok: bool                  # >=3 admitted families OR >=2 admitted modes
    lane_alpha_success: bool       # span_ok AND >=2 distinct technique veins (not 1 meta-technique x3)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def summarize_screen_v1(results: Sequence[FamilyScreenResultV1],
                        slate: Sequence[ScreenCandidateV1]) -> ScreenVerdictV1:
    by_fam = {r.family: r for r in results}
    g3_ok = all((by_fam[c.family].gates.g3_extractable == c.expect_extractable)
                for c in slate if c.family in by_fam)
    adm = [r for r in results if r.admitted]
    fams = tuple(r.family for r in adm)
    veins = tuple(sorted({r.vein for r in adm}))
    modes = tuple(sorted({r.mode for r in adm}))
    span_ok = bool(len(fams) >= 3 or len(modes) >= 2)
    return ScreenVerdictV1(
        n_candidates=len(results), n_dollar0_pass=sum(1 for r in results if r.gates.all_pass),
        n_admitted=len(adm), admitted_families=fams, admitted_veins=veins, admitted_modes=modes,
        g3_predictions_correct=bool(g3_ok), span_ok=span_ok,
        lane_alpha_success=bool(span_ok and len(veins) >= 2))


__all__ = [
    "MODERATE_P_FAMILY_SCREEN_V1_SCHEMA_VERSION", "FNB_PROMPTS", "MINTED_DATE",
    "Dollar0GatesV1", "dollar0_gates_v1", "FairPResultV1", "measure_fair_p_v1",
    "FamilyScreenResultV1", "screen_family_v1", "ScreenVerdictV1", "summarize_screen_v1",
]
