"""W140 / COO-9 — cross-tier TUTOR bench: arm slate + tutor-usability prior + tutor controller (Lane β).

The W140 empirical lane.  Mirrors the W139 capability-matched cross-tier bench VERBATIM in structure
(same K=5 same-budget accounting, same grader, same gate) and changes exactly ONE thing: the payload
injected at the reflexion step is a compiled FAMILY-LEVEL TUTOR (``family_tutor_compiler_v1``) instead
of the raw per-problem witness.  This makes the bench a clean ablation:

  * ``C0`` (W139 raw witness reflexion) — the per-problem diagnostic the 8B could not act on (0/5).
  * ``T1/T2/T3`` (tutor reflexion) — IDENTICAL loop, the payload is the family card / witness→rewrite /
    compressed tutor.  Does compiling the witness into a family-level teaching object lift the weak tier?
  * ``T4`` (the LEAD) — the W139 capability-matched controller, fed a TUTOR: route APPLY vs KEEP by a
    MEASURED per-tier ``tutor_usability_rate`` (the tutor analogue of W139's witness-usability).  KEEP on
    a tutor-ineligible tier ⇒ ``T4 ≡ A1`` (non-negativity by construction, the W139 fix preserved).
  * ``T6`` — a content-free negative control; the fake-different test MUST classify it FAKE_DIFFERENT.

Same-budget: every arm K attempts, attempt-0 the byte-identical standard prompt, one model call per
attempt, no early stop, graded on ``secret_cases``.  The tutor is STATIC prompt text compiled $0
OUTSIDE the K budget (the family card is reused for every instance of the family).  Pure / deterministic
except the audited execution subprocess + the injected ``gen`` (model inference lives in the driver).
Explicit-import only; ``coordpy/__init__.py`` untouched.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

from .resistant_by_construction_battlefield_v1 import (
    DISC_TIMEOUT, MintedProblemV1, _sha256_hex, mint_problem_v1)
from .hard_battlefield_slate_v2 import ParserNeutralTemplateV2
from .icpc_reflexion_bench_v1 import (
    IcpcArmOutcomeV1, IcpcPilotProblemV1, W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION,
    _initial_prompt, _samples_block, extract_candidate_code_v1, grade_on_secret_v1, sample_feedback_v1)
from .band_mechanism_bench_v1 import (  # re-export verbatim (same gate + fake-different discipline)
    GateVerdictV1, evaluate_gate_v1, fake_different_report_v1)
from .family_tutor_compiler_v1 import (
    OBS_TIMEOUT, OBS_WRONG_ANSWER, TC1_CARD, TC2_REWRITE, TC3_COMPRESSED, TC5_STAGED, T6_NEG,
    COMPILERS_BY_KIND, FamilyTutorV1, compile_witness_rewrite_tutor_v1, tutor_is_genuinely_new_v1)

CROSS_TIER_TUTOR_BENCH_V1_SCHEMA_VERSION: str = "coordpy.cross_tier_tutor_bench_v1.v1"

# the W140 tutor-controller arm ids
T4_ARM: str = "T4"                          # the capability-matched TUTOR controller (LEAD)
DEFAULT_TAU_TU: float = 0.34                # tutor-eligibility threshold (== W139 tau_wu, MLB floor)
TUTOR_USABILITY_SEED_BASE: int = 140_150_000  # disjoint from corpus/calibration seeds

# control-flow actions (audit + fake-different evidence), mirroring the W139 controller
ACT_PLAIN: str = "PLAIN"
ACT_KEEP_PLAIN: str = "KEEP_PLAIN"          # tutor-ineligible tier: plain self-consistency draw
ACT_TUTOR_APPLY: str = "TUTOR_APPLY"        # eligible tier: commit the tutored rewrite
ACT_KEEP_REVERT: str = "KEEP_REVERT"        # per-problem revert: tutor worsened public -> plain draw


def tutor_observed_kind_for_template(template: ParserNeutralTemplateV2) -> str:
    """The FAMILY-LEVEL failure mode that routes TC2 (no instance/secret access): a TIMEOUT-discriminated
    family fails by being too slow; an OUTPUT_MISMATCH family fails by a dropped-stage wrong answer."""
    return OBS_TIMEOUT if template.minted.discriminator == DISC_TIMEOUT else OBS_WRONG_ANSWER


# ===================================================== same-budget arm slate (RUNBOOK_W140 §8)

@dataclasses.dataclass(frozen=True)
class TutorArmSpecV1:
    arm_id: str
    label: str
    consumes: str
    hypothesis: str
    tc_kind: str                 # the family_tutor_compiler kind this arm injects ("" for A*/B0)
    is_lead: bool = False
    is_negative_control: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"arm_id": self.arm_id, "label": self.label, "consumes": self.consumes,
                "hypothesis": self.hypothesis, "tc_kind": self.tc_kind,
                "is_lead": bool(self.is_lead), "is_negative_control": bool(self.is_negative_control)}


TUTOR_ARM_SLATE_V1: tuple[TutorArmSpecV1, ...] = (
    TutorArmSpecV1("A0", "plain single-shot", "statement + public samples",
                   "baseline pass@1; no feedback", ""),
    TutorArmSpecV1("A1", "same-budget self-consistency", "statement + public samples (K i.i.d.)",
                   "headroom baseline; the +5pp bar every tutor arm must beat at anchor AND weak tier", ""),
    TutorArmSpecV1("B0", "blind reflexion", "judge-reject bit + stderr + public-sample results",
                   "non-negativity reference; every tutor arm must be >= B0 - 1.0pp per tier", ""),
    TutorArmSpecV1("C0", "W139 raw-witness reflexion", "per-problem owned-oracle diagnostic",
                   "the W139 baseline the 8B could not act on (witness-usability 0.00); the floor", ""),
    TutorArmSpecV1("T1", "family algorithm card", "TC1 family-level technique card",
                   "naming the technique (vs diagnosing slowness) is the signal a weak model can use",
                   TC1_CARD),
    TutorArmSpecV1("T2", "witness-to-rewrite tutor", "TC2 routed root-cause + holed skeleton",
                   "failure-localisation + worked scaffold is the biggest lever over the bare diagnostic",
                   TC2_REWRITE),
    TutorArmSpecV1("T3", "compressed stress tutor", "TC3 smallest-sufficient teaching object",
                   "the smallest dose lifts the weak tier most (long text breaks 8B instruction-following)",
                   TC3_COMPRESSED),
    TutorArmSpecV1("T4", "keep/apply/abstain TUTOR controller", "TC2 routed by per-tier tutor-usability",
                   "capability-matched tutor routing turns cross-tier SAFETY into cross-tier USEFULNESS",
                   TC2_REWRITE, is_lead=True),
    TutorArmSpecV1("T5", "weak-tier staged fallback", "TC5 progressive disclosure",
                   "dose-matched escalation lifts the weak tier without the long-text harm", TC5_STAGED),
    TutorArmSpecV1("T6", "negative-control tutor", "a content-free do-better instruction",
                   "must classify FAKE_DIFFERENT; proves the bench rewards real teaching not decoration",
                   T6_NEG, is_negative_control=True),
)


def compile_family_tutors_v1(template: ParserNeutralTemplateV2,
                             ) -> dict[str, FamilyTutorV1]:
    """Compile every TC-kind tutor a bench arm needs for one family (the family card is reused across
    all instances of the family — $0 at bench time)."""
    return {kind: COMPILERS_BY_KIND[kind](template)
            for kind in (TC1_CARD, TC2_REWRITE, TC3_COMPRESSED, TC5_STAGED, T6_NEG)}


# ===================================================== the tutor-reflexion prompt

def _tutor_reflexion_prompt(problem: IcpcPilotProblemV1, history, tutor: FamilyTutorV1,
                            *, observed_kind: str, attempt_idx: int) -> str:
    """The between-attempt prompt: the SAME scaffold as the W120 blind reflexion prompt (judge verdict
    bit + stderr tail + public-sample results) PLUS the compiled FAMILY-LEVEL TUTOR block — a strict
    superset of the blind feedback, so any gain over B0 is attributable to the tutor (exactly the C0
    accounting, with the witness block swapped for the tutor block)."""
    chunks: list[str] = []
    for i, (cand, passed, stderr_tail, sample_fb) in enumerate(history):
        cand_trim = cand if len(cand) <= 1500 else (cand[:1500] + "\n# ...(truncated)\n")
        verdict = ("ACCEPTED by the judge (all hidden tests passed)" if passed
                   else "REJECTED by the judge (failed at least one hidden test)")
        se = f"\nExecutor stderr (tail):\n{stderr_tail.strip()}" if stderr_tail.strip() else ""
        sf = f"\nPublic sample results:\n{sample_fb}" if sample_fb.strip() else ""
        chunks.append(f"--- Attempt {i+1} ({verdict}) ---\n```python\n{cand_trim}\n```{se}{sf}")
    tblock = tutor.to_prompt_block(observed_kind=observed_kind)
    return (
        "You are an expert ICPC competitor on a reflective debugging loop. You are on "
        f"attempt {attempt_idx + 1} out of 5. Below are your previous attempts with the judge "
        "verdict and the PUBLIC sample-case results, followed by a TECHNIQUE TUTOR for this problem "
        "family. Use the tutor to produce a NEW corrected COMPLETE Python 3 stdin/stdout program. "
        "Do not repeat a previous attempt verbatim.\n\n"
        f"Problem:\n{problem.statement}\n\n"
        f"{_samples_block(problem)}\n\n"
        f"{chr(10).join(chunks)}\n\n"
        f"{tblock}\n\n"
        "Provide ONLY the corrected complete Python 3 program in a ```python ... ``` fence:")


def _pub_pass_count(pilot: IcpcPilotProblemV1, code: str, *, timeout_s: float) -> int:
    return int(str(sample_feedback_v1(pilot, code, timeout_s=timeout_s)).count(": PASS"))


# ===================================================== tutor-usability capability prior (per tier)

@dataclasses.dataclass(frozen=True)
class TutorUsabilityV1:
    model_id: str
    tier: str
    family: str
    tc_kind: str
    n_probed: int
    n_eligible: int           # plain candidate failed secret
    n_flipped: int            # the single tutor-reflexion attempt flipped secret-FAIL -> secret-PASS
    tau: float

    @property
    def rate(self) -> float:
        return self.n_flipped / self.n_eligible if self.n_eligible else 0.0

    @property
    def tutor_eligible(self) -> bool:
        return bool(self.n_eligible > 0 and self.rate >= self.tau)

    def to_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "tier": self.tier, "family": self.family,
                "tc_kind": self.tc_kind, "n_probed": int(self.n_probed),
                "n_eligible": int(self.n_eligible), "n_flipped": int(self.n_flipped),
                "tutor_usability_rate": round(self.rate, 4), "tau": self.tau,
                "tutor_eligible": self.tutor_eligible}


def measure_tutor_usability_v1(template: ParserNeutralTemplateV2, *, gen: Callable,
                               model_id: str, tier: str, tutor: FamilyTutorV1, n_tu: int = 6,
                               seed_base: int = TUTOR_USABILITY_SEED_BASE, tau: float = DEFAULT_TAU_TU,
                               max_tokens: int = 1536, timeout_s: float = 8.0,
                               mint_timeout_s: float = 1.0, temperature: float = 0.7,
                               minted_date: str = "2026-06-07") -> TutorUsabilityV1:
    """Measure, per tier, how often ONE tutor-reflexion attempt converts a secret-FAILING plain
    candidate into a secret-PASSING one — the W140 capability prior ``T4`` routes on (the tutor
    analogue of W139's witness-usability).  Computed at CALIBRATION on FRESH instances (disjoint
    seeds, TRAIN base); $0 in the graded bench budget.  THE W140 THESIS: does this lift the 8B above
    tau where the raw-witness rate was 0.00?"""
    observed_kind = tutor_observed_kind_for_template(template)
    eligible = 0
    flipped = 0
    for r in range(int(n_tu)):
        mp = mint_problem_v1(template.minted, global_seed=int(seed_base) + r,
                             timeout_s=float(mint_timeout_s))
        pilot = mp.to_pilot_problem(minted_date=minted_date)
        text, _ = gen(_initial_prompt(pilot), int(max_tokens), 0.0)
        code = extract_candidate_code_v1(response_text=text)
        passed, stderr_tail, _ = grade_on_secret_v1(pilot, code, timeout_s=float(timeout_s))
        if passed:
            continue  # nothing to repair on this instance
        eligible += 1
        sfb = sample_feedback_v1(pilot, code, timeout_s=float(timeout_s))
        tprompt = _tutor_reflexion_prompt(
            pilot, ((code, False, stderr_tail, sfb),), tutor, observed_kind=observed_kind,
            attempt_idx=1)
        ttext, _ = gen(tprompt, int(max_tokens), float(temperature))
        tcode = extract_candidate_code_v1(response_text=ttext)
        tpassed, _, _ = grade_on_secret_v1(pilot, tcode, timeout_s=float(timeout_s))
        if tpassed:
            flipped += 1
    return TutorUsabilityV1(model_id=str(model_id), tier=str(tier), family=template.minted.family,
                            tc_kind=tutor.tc_kind, n_probed=int(n_tu), n_eligible=int(eligible),
                            n_flipped=int(flipped), tau=float(tau))


# ===================================================== a single tutor (diagnostic) arm: T1/T2/T3/T5

@dataclasses.dataclass(frozen=True)
class TutorArmTraceV1:
    problem_id: str
    arm_id: str
    tc_kind: str
    observed_kind: str
    n_tutor_attempts: int
    leakage_clean: bool

    def rescue_is_algorithmic(self) -> bool:
        """A tutor rescue is algorithmic by construction (the tutor carries a named technique/skeleton,
        never a parsing/formatting fix); parser-neutrality (HC1) already excludes I/O-format gains."""
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"problem_id": self.problem_id, "arm_id": self.arm_id, "tc_kind": self.tc_kind,
                "observed_kind": self.observed_kind, "n_tutor_attempts": int(self.n_tutor_attempts),
                "leakage_clean": bool(self.leakage_clean)}


def run_tutor_arm_v1(*, seed: int, template: ParserNeutralTemplateV2, problem: MintedProblemV1,
                     tutor: FamilyTutorV1, gen: Callable, K: int, temperature: float, max_tokens: int,
                     timeout_s: float, minted_date: str, arm_id: str,
                     ) -> tuple[IcpcArmOutcomeV1, TutorArmTraceV1]:
    """A single tutor-reflexion arm (T1/T2/T3/T5): attempt-0 plain, attempts 1..K-1 inject the family
    tutor.  Same-budget K, graded on secret; the tutor is reused $0 across the family."""
    pilot = problem.to_pilot_problem(minted_date=str(minted_date))
    observed_kind = tutor_observed_kind_for_template(template)
    history: list[tuple[str, bool, str, str]] = []
    per_call: list[bool] = []
    first_pass = -1
    n_tutor = 0
    for k in range(int(K)):
        if k == 0:
            prompt = _initial_prompt(pilot)
        else:
            prompt = _tutor_reflexion_prompt(pilot, tuple(history), tutor,
                                             observed_kind=observed_kind, attempt_idx=k)
            n_tutor += 1
        text, _ = gen(prompt, int(max_tokens), float(temperature))
        code = extract_candidate_code_v1(response_text=text)
        passed, stderr_tail, _ = grade_on_secret_v1(pilot, code, timeout_s=float(timeout_s))
        sfb = sample_feedback_v1(pilot, code, timeout_s=float(timeout_s))
        per_call.append(bool(passed))
        if passed and first_pass == -1:
            first_pass = int(k)
        history.append((code, bool(passed), stderr_tail, sfb))
    outcome = IcpcArmOutcomeV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, seed=int(seed),
        question_id=problem.problem_id, arm_id=str(arm_id),
        final_passed=bool(first_pass >= 0), n_model_calls=int(K),
        per_call_passed=tuple(per_call), first_pass_attempt_idx=int(first_pass))
    trace = TutorArmTraceV1(problem_id=problem.problem_id, arm_id=str(arm_id), tc_kind=tutor.tc_kind,
                            observed_kind=observed_kind, n_tutor_attempts=int(n_tutor),
                            leakage_clean=True)
    return outcome, trace


# ===================================================== the capability-matched TUTOR controller (T4)

@dataclasses.dataclass(frozen=True)
class TutorControllerTraceV1:
    problem_id: str
    tier: str
    tutor_eligible: bool
    actions: tuple[str, ...]
    reverted: bool

    def n_distinct_action_types(self) -> int:
        return len(set(self.actions))

    def rescue_is_algorithmic(self) -> bool:
        """KEEP is structural by construction (== A1); an APPLY rescue carries the named-technique
        tutor (never a parsing fix)."""
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"problem_id": self.problem_id, "tier": self.tier,
                "tutor_eligible": bool(self.tutor_eligible), "actions": list(self.actions),
                "reverted": bool(self.reverted),
                "n_distinct_action_types": self.n_distinct_action_types()}


def run_tutor_controller_arm_v1(*, seed: int, template: ParserNeutralTemplateV2,
                                problem: MintedProblemV1, tutor: FamilyTutorV1, gen: Callable, K: int,
                                temperature: float, max_tokens: int, timeout_s: float, minted_date: str,
                                tutor_eligible: bool,
                                ) -> tuple[IcpcArmOutcomeV1, TutorControllerTraceV1]:
    """T4 — the capability-matched TUTOR controller.  Same-budget K.  KEEP=plain when the tier is not
    tutor-eligible (so ``T4 ≡ A1``, never hurts — the W139 non-negativity guarantee preserved);
    APPLY-the-tutor-with-per-problem-revert when it is (revert if the tutored draft strictly worsens
    the public-sample pass count)."""
    pilot = problem.to_pilot_problem(minted_date=str(minted_date))
    observed_kind = tutor_observed_kind_for_template(template)
    history: list[tuple[str, bool, str, str]] = []
    per_call: list[bool] = []
    actions: list[str] = []
    first_pass = -1
    reverted = False
    for k in range(int(K)):
        if k == 0:
            prompt, action = _initial_prompt(pilot), ACT_PLAIN
        elif not tutor_eligible:
            prompt, action = _initial_prompt(pilot), ACT_KEEP_PLAIN
        elif reverted:
            prompt, action = _initial_prompt(pilot), ACT_KEEP_REVERT
        else:
            prompt = _tutor_reflexion_prompt(pilot, tuple(history), tutor,
                                             observed_kind=observed_kind, attempt_idx=k)
            action = ACT_TUTOR_APPLY
        text, _ = gen(prompt, int(max_tokens), float(temperature))
        code = extract_candidate_code_v1(response_text=text)
        passed, stderr_tail, _ = grade_on_secret_v1(pilot, code, timeout_s=float(timeout_s))
        sfb = sample_feedback_v1(pilot, code, timeout_s=float(timeout_s))
        per_call.append(bool(passed))
        actions.append(action)
        if passed and first_pass == -1:
            first_pass = int(k)
        if action == ACT_TUTOR_APPLY and len(history) >= 1:
            if _pub_pass_count(pilot, code, timeout_s=float(timeout_s)) < \
                    int(str(history[-1][3]).count(": PASS")):
                reverted = True
        history.append((code, bool(passed), stderr_tail, sfb))
    outcome = IcpcArmOutcomeV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, seed=int(seed),
        question_id=problem.problem_id, arm_id=T4_ARM, final_passed=bool(first_pass >= 0),
        n_model_calls=int(K), per_call_passed=tuple(per_call), first_pass_attempt_idx=int(first_pass))
    trace = TutorControllerTraceV1(problem_id=problem.problem_id, tier="",
                                   tutor_eligible=bool(tutor_eligible), actions=tuple(actions),
                                   reverted=bool(reverted))
    return outcome, trace


def run_tutor_split_arm_v1(*, seed: int, template: ParserNeutralTemplateV2, problem: MintedProblemV1,
                           tutor: FamilyTutorV1, gen: Callable, K: int, temperature: float,
                           max_tokens: int, timeout_s: float, minted_date: str, n_plain: int = 2,
                           ) -> tuple[IcpcArmOutcomeV1, TutorArmTraceV1]:
    """DIVERSITY-PRESERVING split-K (W140-iter2): the first ``n_plain`` of K attempts are PLAIN i.i.d.
    draws (preserving self-consistency's diversity / the mass it catches by luck), the remaining
    K-n_plain attempts inject the family tutor.  Same-budget K.  Rationale (arXiv:2503.08681): tutor
    and resampling stop competing for the same K slots — the arm cannot lose more than the retained
    plain draws would, while still adding the technique mode.  Graded on secret."""
    pilot = problem.to_pilot_problem(minted_date=str(minted_date))
    observed_kind = tutor_observed_kind_for_template(template)
    history: list[tuple[str, bool, str, str]] = []
    per_call: list[bool] = []
    first_pass = -1
    n_tutor = 0
    for k in range(int(K)):
        if k < int(n_plain) or not history:
            prompt = _initial_prompt(pilot)
        else:
            prompt = _tutor_reflexion_prompt(pilot, tuple(history), tutor,
                                             observed_kind=observed_kind, attempt_idx=k)
            n_tutor += 1
        text, _ = gen(prompt, int(max_tokens), float(temperature))
        code = extract_candidate_code_v1(response_text=text)
        passed, stderr_tail, _ = grade_on_secret_v1(pilot, code, timeout_s=float(timeout_s))
        sfb = sample_feedback_v1(pilot, code, timeout_s=float(timeout_s))
        per_call.append(bool(passed))
        if passed and first_pass == -1:
            first_pass = int(k)
        history.append((code, bool(passed), stderr_tail, sfb))
    outcome = IcpcArmOutcomeV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, seed=int(seed),
        question_id=problem.problem_id, arm_id="T4split", final_passed=bool(first_pass >= 0),
        n_model_calls=int(K), per_call_passed=tuple(per_call), first_pass_attempt_idx=int(first_pass))
    trace = TutorArmTraceV1(problem_id=problem.problem_id, arm_id="T4split", tc_kind=tutor.tc_kind,
                            observed_kind=observed_kind, n_tutor_attempts=int(n_tutor), leakage_clean=True)
    return outcome, trace


def tutor_controller_is_genuinely_new_v1(trace: TutorControllerTraceV1) -> dict[str, Any]:
    """Structural 'not a relabel' check: an eligible T4 run uses >=2 distinct action types (a tutor
    plane + a plain/keep plane); an ineligible run is exactly A1 (KEEP) — honest non-action that must
    NOT be counted as a mechanism rescue."""
    eligible = trace.tutor_eligible
    used_tutor = any(a == ACT_TUTOR_APPLY for a in trace.actions)
    routed = trace.n_distinct_action_types() >= 2
    return {"genuinely_new": bool(eligible and used_tutor and routed),
            "tutor_eligible": eligible, "used_tutor": used_tutor,
            "routed_multi_action": routed, "is_keep_noop": bool(not eligible)}


__all__ = [
    "CROSS_TIER_TUTOR_BENCH_V1_SCHEMA_VERSION", "T4_ARM", "DEFAULT_TAU_TU",
    "TUTOR_USABILITY_SEED_BASE", "ACT_PLAIN", "ACT_KEEP_PLAIN", "ACT_TUTOR_APPLY", "ACT_KEEP_REVERT",
    "tutor_observed_kind_for_template", "TutorArmSpecV1", "TUTOR_ARM_SLATE_V1",
    "compile_family_tutors_v1", "TutorUsabilityV1", "measure_tutor_usability_v1",
    "TutorArmTraceV1", "run_tutor_arm_v1", "TutorControllerTraceV1", "run_tutor_controller_arm_v1",
    "run_tutor_split_arm_v1", "tutor_controller_is_genuinely_new_v1",
    # re-exported verbatim from band_mechanism_bench_v1
    "GateVerdictV1", "evaluate_gate_v1", "fake_different_report_v1", "tutor_is_genuinely_new_v1",
]
