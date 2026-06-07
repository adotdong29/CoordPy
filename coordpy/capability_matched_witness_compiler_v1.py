"""W139 / COO-9 — capability-matched witness compiler + large-probe counterexample (Lane β).

W138 found the exact-oracle complexity witness earns at the 70B anchor (+40 vs A1 / +25 vs B0,
≥3 families) but FAILS cross-tier: it HURTS the 8B (−25 — the 8B writes broken efficient code below
its own self-consistency baseline) and the counterexample 2nd mode never fired (it searched only
≤400-token probes, so the large-input counterexample of the HIDDEN_EDGE families was invisible).

This module adds the two genuinely-new W139 instruments (reusing the W133 witness engine VERBATIM —
``select_witness_v1`` / ``_witness_reflexion_prompt`` / ``build_witness_probe_set_v1`` / the audited
grader — with ZERO edits to existing modules):

* **The capability-matched controller (arm ``Cm``, the W139 LEAD).** A same-budget K-attempt arm that
  routes APPLY-witness vs KEEP-self-consistency by *measured* generator capability, so the witness can
  never lower a weak generator's pass-rate (R5/R2; small models need a verifier matched to their
  capability — arXiv:2404.17140; intrinsic revision can degrade — arXiv:2310.01798):
    - if the tier is NOT witness-eligible (its calibrated ``witness_usability_rate < τ``) → ALL K
      attempts are plain self-consistency draws ⇒ ``Cm ≡ A1`` (KEEP; non-negativity guarantee);
    - if witness-eligible → attempt-0 standard, attempt-1 witness-reflexion (the per-problem probe),
      then for k≥2 APPLY the witness iff the probe did NOT strictly worsen the public-sample pass
      count, else REVERT this problem to plain self-consistency. Consumes only (i) the per-tier
      capability prior (paid at calibration, $0 in the graded budget) and (ii) public-sample pass
      counts of its own candidates ($0). Never reads the secret or any oracle program.

* **The large-probe counterexample witness (arm ``Nb``).** ``build_large_probe_set_v1`` raises the
  probe token cap to ``LARGE_PROBE_TOKEN_CAP`` and generates from a moderate-N probe template so a
  counterexample that only appears on larger inputs (the HIDDEN_EDGE / WRONG_ANSWER families) is found;
  it is then fed to ``run_witness_arm_v1(arm=C1, ...)`` unchanged. The leakage guard is identical (the
  probe input is checked byte-disjoint from the graded secret cases).

* **The witness-usability capability prior** (``measure_witness_usability_v1``): per tier, on
  repair-eligible calibration instances, how often a single witness-reflexion attempt flips a
  secret-FAILING plain candidate to secret-PASSING — the measured capability signal ``Cm`` routes on.

Pure / deterministic except the (already-audited) program-execution subprocess; NO model inference
lives here (that is the W139 driver script).  Explicit-import only; ``coordpy/__init__.py`` untouched.
"""
from __future__ import annotations

import dataclasses
import random
from typing import Any, Callable, Optional

from .resistant_by_construction_battlefield_v1 import (
    MintedProblemV1, MintedTemplateV1, _exec_capture_v1, _sha256_hex, _tok_count, mint_problem_v1)
from .icpc_reflexion_bench_v1 import (
    IcpcArmOutcomeV1, IcpcPilotProblemV1, W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION,
    _initial_prompt, extract_candidate_code_v1, grade_on_secret_v1, sample_feedback_v1)
from .exact_oracle_witness_v1 import (
    ARM_C1_COUNTEREXAMPLE, ARM_C2_COMPLEXITY, ARM_C3_CONTROLLER, N_SMALL_PROBE_ROUNDS,
    OBS_TIMEOUT, OBS_WRONG_ANSWER, WITNESS_COMPLEXITY, WITNESS_COUNTEREXAMPLE,
    WitnessProbeSetV1, _witness_reflexion_prompt, build_witness_probe_set_v1, select_witness_v1)

CAPABILITY_MATCHED_WITNESS_COMPILER_V1_SCHEMA_VERSION: str = (
    "coordpy.capability_matched_witness_compiler_v1.v1")

CM_ARM: str = "Cm"                  # the capability-matched controller (LEAD)
NB_ARM: str = "Nb"                  # the large-probe counterexample witness
DEFAULT_TAU_WU: float = 0.34       # witness-eligibility threshold (matched to the MLB rescue floor)
LARGE_PROBE_TOKEN_CAP: int = 6000  # raised from exact_oracle_witness_v1.SMALL_PROBE_TOKEN_CAP = 400
WITNESS_USABILITY_SEED_BASE: int = 139_950_000   # disjoint from band-calibration + corpus seeds

# control-flow actions the controller can take on an attempt (audit + fake-different evidence)
ACT_PLAIN: str = "PLAIN"                 # attempt-0 standard prompt (== A1 attempt 0)
ACT_KEEP_PLAIN: str = "KEEP_PLAIN"       # incapable tier: plain self-consistency draw
ACT_WITNESS_APPLY: str = "WITNESS_APPLY" # capable tier: commit the witnessed rewrite
ACT_KEEP_REVERT: str = "KEEP_REVERT"     # per-problem revert: witness worsened public -> plain draw
ACT_PLAIN_NO_WITNESS: str = "PLAIN_NO_WITNESS"  # eligible but no witness fired -> plain draw


# ===================================================== $0 public-sample signal

def _pub_pass_count_from_feedback(sample_fb: str) -> int:
    """Count PUBLIC sample passes from ``sample_feedback_v1`` output ($0, public-only)."""
    return int(str(sample_fb).count(": PASS"))


def _public_pass_count_v1(pilot: IcpcPilotProblemV1, code: str, *, timeout_s: float) -> int:
    return _pub_pass_count_from_feedback(sample_feedback_v1(pilot, code, timeout_s=timeout_s))


# ===================================================== witness-usability capability prior

@dataclasses.dataclass(frozen=True)
class WitnessUsabilityV1:
    model_id: str
    tier: str
    n_probed: int
    n_eligible: int          # plain candidate failed secret AND a witness fired
    n_flipped: int           # the single witness attempt flipped secret-FAIL -> secret-PASS
    tau: float

    @property
    def rate(self) -> float:
        return self.n_flipped / self.n_eligible if self.n_eligible else 0.0

    @property
    def witness_eligible(self) -> bool:
        return bool(self.n_eligible > 0 and self.rate >= self.tau)

    def to_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "tier": self.tier, "n_probed": int(self.n_probed),
                "n_eligible": int(self.n_eligible), "n_flipped": int(self.n_flipped),
                "witness_usability_rate": round(self.rate, 4), "tau": self.tau,
                "witness_eligible": self.witness_eligible}


def measure_witness_usability_v1(template, *, gen: Callable[[str, int, float], Any], model_id: str,
                                 tier: str, n_wu: int = 6, witness_seed: int,
                                 seed_base: int = WITNESS_USABILITY_SEED_BASE,
                                 witness_arm: str = ARM_C3_CONTROLLER, tau: float = DEFAULT_TAU_WU,
                                 max_tokens: int = 1536, timeout_s: float = 8.0,
                                 witness_timeout_s: float = 2.0, mint_timeout_s: float = 1.0,
                                 temperature: float = 0.7, minted_date: str = "2026-06-07",
                                 ) -> WitnessUsabilityV1:
    """Measure, for one tier, how often ONE owned-oracle witness-reflexion attempt converts a
    secret-FAILING plain candidate into a secret-PASSING one — the capability prior ``Cm`` routes on.
    Computed at CALIBRATION on FRESH instances (disjoint seeds); $0 in the graded bench budget."""
    eligible = 0
    flipped = 0
    for r in range(int(n_wu)):
        mp = mint_problem_v1(template.minted, global_seed=int(seed_base) + r,
                             timeout_s=float(mint_timeout_s))
        pilot = mp.to_pilot_problem(minted_date=minted_date)
        text, _ = gen(_initial_prompt(pilot), int(max_tokens), 0.0)
        code = extract_candidate_code_v1(response_text=text)
        passed, _, _ = grade_on_secret_v1(pilot, code, timeout_s=float(timeout_s))
        if passed:
            continue  # nothing to repair on this instance
        probe = build_witness_probe_set_v1(template.minted, mp, witness_seed=int(witness_seed),
                                           timeout_s=float(witness_timeout_s))
        witness = select_witness_v1(code, mp, probe, template.minted, arm=witness_arm,
                                    timeout_s=float(witness_timeout_s))
        if not witness.found():
            continue  # witness cannot fire here -> not a usability opportunity
        eligible += 1
        sfb = sample_feedback_v1(pilot, code, timeout_s=float(timeout_s))
        wprompt = _witness_reflexion_prompt(pilot, ((code, False, "", sfb),), witness, attempt_idx=1)
        wtext, _ = gen(wprompt, int(max_tokens), float(temperature))
        wcode = extract_candidate_code_v1(response_text=wtext)
        wpassed, _, _ = grade_on_secret_v1(pilot, wcode, timeout_s=float(timeout_s))
        if wpassed:
            flipped += 1
    return WitnessUsabilityV1(model_id=str(model_id), tier=str(tier), n_probed=int(n_wu),
                              n_eligible=int(eligible), n_flipped=int(flipped), tau=float(tau))


# ===================================================== large-probe counterexample set (Nb)

def build_large_probe_set_v1(probe_template: MintedTemplateV1, problem: MintedProblemV1, *,
                             witness_seed: int, large_cap: int = LARGE_PROBE_TOKEN_CAP,
                             n_rounds: int = N_SMALL_PROBE_ROUNDS, ref_timeout_s: float = 8.0,
                             ) -> WitnessProbeSetV1:
    """Build a counterexample probe set whose ``.small`` carries LARGE structured inputs (up to
    ``large_cap`` tokens) generated by a moderate-N ``probe_template`` of the same family, so a
    counterexample that only appears on larger inputs is found.  Byte-disjoint from the graded
    secret cases (no teaching-to-the-test).  This is the ONLY change vs W138's small-probe set —
    it directly tests whether W138's N0 +0 was a probe-size firing artifact, not a capability limit.
    """
    secret_inputs = {inp for inp, _ in problem.secret_cases}
    rng = random.Random(_sha256_hex(
        {"large_witness_probe": True, "name": probe_template.name, "seed": int(witness_seed)}))
    raw: list[str] = []
    for _ in range(int(n_rounds)):
        try:
            raw.extend(probe_template.gen_public(rng))
            raw.extend(probe_template.gen_hidden(rng))
        except Exception:
            break
    seen: set[str] = set()
    small: list[tuple[str, str]] = []
    for inp in raw:
        if inp in seen:
            continue
        seen.add(inp)
        if inp in secret_inputs:
            continue
        if _tok_count(inp) <= int(large_cap):
            r = _exec_capture_v1(probe_template.ref_source, inp, timeout_s=float(ref_timeout_s))
            if not r.timed_out and r.returncode == 0:
                small.append((inp, r.stdout))
    small.sort(key=lambda s: (_tok_count(s[0]), s[0]))
    return WitnessProbeSetV1(
        problem_id=problem.problem_id, witness_seed=int(witness_seed), small=tuple(small),
        big_input=None, big_ref_runtime_s=0.0, big_input_tokens=0,
        secret_input_set_size=len(secret_inputs))


def build_combined_probe_set_v1(graded_template: MintedTemplateV1, probe_template: MintedTemplateV1,
                                problem: MintedProblemV1, *, witness_seed: int,
                                large_cap: int = LARGE_PROBE_TOKEN_CAP, ref_timeout_s: float = 8.0,
                                ) -> WitnessProbeSetV1:
    """A probe set the C3 controller can route on for BOTH modes: LARGE counterexample probes (from
    the moderate ``probe_template``, reviving the 2nd mode) PLUS the complexity stress ``big_input``
    (from the graded large-N ``graded_template``).  This is what arm ``Cm`` consumes."""
    large = build_large_probe_set_v1(probe_template, problem, witness_seed=int(witness_seed),
                                     large_cap=int(large_cap), ref_timeout_s=float(ref_timeout_s))
    std = build_witness_probe_set_v1(graded_template, problem, witness_seed=int(witness_seed),
                                     timeout_s=float(ref_timeout_s))
    return WitnessProbeSetV1(
        problem_id=problem.problem_id, witness_seed=int(witness_seed), small=large.small,
        big_input=std.big_input, big_ref_runtime_s=std.big_ref_runtime_s,
        big_input_tokens=std.big_input_tokens, secret_input_set_size=std.secret_input_set_size)


# ===================================================== the capability-matched controller (Cm)

@dataclasses.dataclass(frozen=True)
class CapMatchedTraceV1:
    problem_id: str
    tier: str
    witness_eligible: bool
    actions: tuple[str, ...]            # the action taken on each of the K attempts
    witness_observed_kinds: tuple[str, ...]
    reverted: bool
    all_leakage_clean: bool

    def n_distinct_action_types(self) -> int:
        return len(set(self.actions))

    def rescue_is_algorithmic(self) -> bool:
        """True iff a committed witness reported a WRONG_ANSWER/TIMEOUT (algorithmic) failure, OR the
        tier KEPT plain (a KEEP outcome is structural by construction — it is exactly A1)."""
        if not self.witness_eligible:
            return True
        return any(ok in (OBS_WRONG_ANSWER, OBS_TIMEOUT) for ok in self.witness_observed_kinds)

    def to_dict(self) -> dict[str, Any]:
        return {"problem_id": self.problem_id, "tier": self.tier,
                "witness_eligible": bool(self.witness_eligible), "actions": list(self.actions),
                "witness_observed_kinds": list(self.witness_observed_kinds),
                "reverted": bool(self.reverted),
                "n_distinct_action_types": self.n_distinct_action_types(),
                "all_leakage_clean": bool(self.all_leakage_clean)}


def run_capability_matched_arm_v1(*, seed: int, template: MintedTemplateV1, problem: MintedProblemV1,
                                  probe: WitnessProbeSetV1, gen, K: int, temperature: float,
                                  max_tokens: int, timeout_s: float, minted_date: str,
                                  witness_eligible: bool, witness_arm: str = ARM_C3_CONTROLLER,
                                  witness_timeout_s: float = 2.0,
                                  ) -> tuple[IcpcArmOutcomeV1, CapMatchedTraceV1]:
    """The capability-matched controller arm.  Same-budget (K attempts, one model call each, no early
    stop); graded on ``problem.secret_cases``.  KEEP=plain when the tier is not witness-eligible (so
    ``Cm ≡ A1``, never hurts); APPLY-with-per-problem-revert when it is."""
    pilot = problem.to_pilot_problem(minted_date=str(minted_date))
    history: list[tuple[str, bool, str, str]] = []
    per_call: list[bool] = []
    actions: list[str] = []
    wobs: list[str] = []
    first_pass_idx = -1
    reverted = False
    leak_clean = True
    for k in range(int(K)):
        if k == 0:
            prompt, action = _initial_prompt(pilot), ACT_PLAIN
        elif not witness_eligible:
            prompt, action = _initial_prompt(pilot), ACT_KEEP_PLAIN
        else:
            last_code = history[-1][0]
            witness = select_witness_v1(last_code, problem, probe, template, arm=witness_arm,
                                        timeout_s=float(witness_timeout_s))
            if not witness.found():
                prompt, action = _initial_prompt(pilot), ACT_PLAIN_NO_WITNESS
            elif reverted:
                prompt, action = _initial_prompt(pilot), ACT_KEEP_REVERT
            else:
                if witness.observed_kind:
                    wobs.append(witness.observed_kind)
                if not witness.leakage_clean:
                    leak_clean = False
                prompt = _witness_reflexion_prompt(pilot, tuple(history), witness, attempt_idx=k)
                action = ACT_WITNESS_APPLY
        text, _ = gen(prompt, int(max_tokens), float(temperature))
        code = extract_candidate_code_v1(response_text=text)
        passed, stderr_tail, _ = grade_on_secret_v1(pilot, code, timeout_s=float(timeout_s))
        sfb = sample_feedback_v1(pilot, code, timeout_s=float(timeout_s))
        per_call.append(bool(passed))
        actions.append(action)
        if passed and first_pass_idx == -1:
            first_pass_idx = int(k)
        # per-problem online revert: if the FIRST applied witness strictly worsens the public-sample
        # pass count vs the pre-witness candidate, this model cannot implement the witnessed algorithm
        # here -> revert the rest of this problem to plain self-consistency (KEEP).
        if action == ACT_WITNESS_APPLY and len(history) >= 1:
            pre_pub = _pub_pass_count_from_feedback(history[-1][3])
            cur_pub = _pub_pass_count_from_feedback(sfb)
            if cur_pub < pre_pub:
                reverted = True
        history.append((code, bool(passed), stderr_tail, sfb))
    outcome = IcpcArmOutcomeV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, seed=int(seed),
        question_id=problem.problem_id, arm_id=CM_ARM,
        final_passed=bool(first_pass_idx >= 0), n_model_calls=int(K),
        per_call_passed=tuple(per_call), first_pass_attempt_idx=int(first_pass_idx))
    trace = CapMatchedTraceV1(
        problem_id=problem.problem_id, tier="", witness_eligible=bool(witness_eligible),
        actions=tuple(actions), witness_observed_kinds=tuple(wobs), reverted=bool(reverted),
        all_leakage_clean=bool(leak_clean))
    return outcome, trace


def capability_matched_is_genuinely_new_v1(trace: CapMatchedTraceV1) -> dict[str, Any]:
    """Structural 'not a relabel' check: an eligible Cm run uses >=2 distinct action types (a witness
    plane + a plain/keep plane) routed on the oracle/public digest; an ineligible run is exactly A1
    (KEEP), which is honest non-action (it must NOT be counted as a mechanism rescue)."""
    eligible = trace.witness_eligible
    used_witness = any(a in (ACT_WITNESS_APPLY,) for a in trace.actions)
    routed = trace.n_distinct_action_types() >= 2
    genuinely_new = bool(eligible and used_witness and routed)
    return {"genuinely_new": genuinely_new, "witness_eligible": eligible,
            "used_witness": used_witness, "routed_multi_action": routed,
            "is_keep_noop": bool(not eligible)}


__all__ = [
    "CAPABILITY_MATCHED_WITNESS_COMPILER_V1_SCHEMA_VERSION", "CM_ARM", "NB_ARM",
    "DEFAULT_TAU_WU", "LARGE_PROBE_TOKEN_CAP", "WITNESS_USABILITY_SEED_BASE",
    "ACT_PLAIN", "ACT_KEEP_PLAIN", "ACT_WITNESS_APPLY", "ACT_KEEP_REVERT", "ACT_PLAIN_NO_WITNESS",
    "_public_pass_count_v1", "WitnessUsabilityV1", "measure_witness_usability_v1",
    "build_large_probe_set_v1", "build_combined_probe_set_v1", "CapMatchedTraceV1",
    "run_capability_matched_arm_v1", "capability_matched_is_genuinely_new_v1",
]
