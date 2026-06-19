"""W133 / COO-9 — exact-oracle WITNESS instrument on the resistant-by-construction battlefield.

W132 minted a CoordPy-owned, exact-oracle, novelty-guarded resistant-by-construction
battlefield and ran the exact W105 retirement model on it: a clean FAIL (B - A1 = +3.33 pp;
the same-budget reflexion mechanism rescued exactly ONE complexity problem and left six
capability-bound traps unsolved by every arm).  That closed the "maybe the official
benchmarks were the wrong test" escape, but it left one honest lever un-pulled: **W132's
reflexion feedback is a BLIND hidden-test reject bit** — between attempts the model sees only
the judge verdict (accepted / rejected), the executor stderr tail, and the PUBLIC sample
results (which all pass, because the trap is constructed to look right on public samples).
The model is never told *which* input breaks it or *what the answer should be*.

This module turns the battlefield we now OWN into a TEACHER.  Because every minted problem
ships an exact ``ref_source`` answer-key oracle, we can compute a far richer, *sanctioned*
feedback object than a reject bit — an **exact-oracle witness**:

* **COUNTEREXAMPLE witness (EW1 / EW3 / EW4)** — a small, public-style input ``X`` on which
  the candidate disagrees with the reference, reported as ``(X, expected = ref(X),
  observed = candidate(X))`` with a deterministic minimality/shrink trace.  EW1 is the
  generic value-counterexample; EW3 biases the probe toward constructed corner cases
  (the HIDDEN_EDGE failure mode); EW4 uses a small-n exhaustive cross-check (the SEARCH_ENUM
  mode).  All three are the same object produced by different probe distributions.
* **COMPLEXITY witness (EW2)** — for a candidate that is value-correct but asymptotically too
  slow (the COMPLEXITY_BLIND mode), a size-growth witness: "your program did not finish
  within ``T_probe`` s on an input of size N≈``big_n`` while a correct reference finishes in
  ``ref_runtime`` s — your algorithm is too slow."  No hidden case is leaked; only the
  structured timing fact derived from the public scale.

Anti-cheat / no-leakage (LOCKED — see ``docs/RUNBOOK_W133.md`` §3):

* the witness is the ONLY sanctioned feedback object; ``ref_source`` / ``brute_source`` /
  ``naive_source`` are NEVER placed in any model-facing path — the model sees ONLY the
  witness triple ``(X, expected, observed)`` (an oracle *output*, not the oracle program);
* witness probe inputs are drawn from a FRESH ``witness_seed`` stream, structurally disjoint
  from the graded ``secret_cases`` (which come from the mint seed); ``leakage_clean`` asserts
  ``probe_input`` is not byte-equal to any graded secret-case input, so the model is graded
  on hidden cases it never saw (the witness tests GENERALISATION, not memorisation);
* every witness is reproducible from the content-addressed witness API (a fixed
  ``witness_seed`` + the deterministic probe builder), so the feedback is auditable.

The witness-guided arm (``run_witness_arm_v1``) is a strict **same-budget** swap of the W120
reflexion arm ``_run_b``: identical K, identical model, identical temperature, identical
attempt-0 prompt; the ONLY change is that the between-attempt feedback object is the witness
instead of the blind reject bit.  It is scored by the SAME audited grader
(``grade_on_secret_v1``) and the SAME W108 evaluator that scored W89 / W105 / W120 / W132, so
"C - A1" is computed byte-identically to "B - A1".

Reuses (explicit-import only, NO duplication): the minted-problem record + the exact-oracle
subprocess from ``resistant_by_construction_battlefield_v1``; the pilot-problem record, the
candidate extractor, the secret grader, and the prompt scaffolds from
``icpc_reflexion_bench_v1``.  Pure / deterministic except the (already-audited) program-execution
subprocess; NO model inference lives here (that is the W133 driver script).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import time
from typing import Any, Optional, Sequence

from .resistant_by_construction_battlefield_v1 import (
    DISC_TIMEOUT,
    ExecCaptureV1,
    MintedProblemV1,
    MintedTemplateV1,
    _exec_capture_v1,
    _tok_count,
)
from .icpc_reflexion_bench_v1 import (
    IcpcArmOutcomeV1,
    W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION,
    _initial_prompt,
    _samples_block,
    extract_candidate_code_v1,
    grade_on_secret_v1,
    sample_feedback_v1,
)
from .coordpy_icpc_battlefield_v1 import judge_icpc_output_v1

W133_EXACT_ORACLE_WITNESS_V1_SCHEMA_VERSION: str = (
    "coordpy.exact_oracle_witness_v1.v1")

# ---- witness kinds -----------------------------------------------------------------
WITNESS_COUNTEREXAMPLE: str = "COUNTEREXAMPLE"   # EW1/EW3/EW4: a value disagreement
WITNESS_COMPLEXITY: str = "COMPLEXITY"           # EW2: a too-slow (timing) witness
WITNESS_NONE: str = "NONE"                        # candidate value-correct + fast on the probe

# ---- witness arms (the same-budget feedback policies) ------------------------------
ARM_C1_COUNTEREXAMPLE: str = "C1"   # EW1 only (value counterexample)
ARM_C2_COMPLEXITY: str = "C2"       # EW2 only (complexity witness)
ARM_C3_CONTROLLER: str = "C3"       # auto-select: counterexample else complexity (LEAD)

# observed-failure kinds for the witness triple
OBS_WRONG_ANSWER: str = "WRONG_ANSWER"
OBS_TIMEOUT: str = "TIMEOUT"
OBS_RUNTIME_ERROR: str = "RUNTIME_ERROR"

# ---- probe sizing (LOCKED; the witness compute budget) -----------------------------
N_SMALL_PROBE_ROUNDS: int = 16     # gen_public/gen_hidden re-draws at the witness seed
SMALL_PROBE_TOKEN_CAP: int = 400   # an input is "small / public-style" iff <= this
COMPLEXITY_PROBE_TIMEOUT_S: float = 2.0   # the candidate must finish a big input within this
REF_FAST_CEILING_S: float = 1.0    # a complexity witness requires ref to finish well under this
MAX_OBSERVED_CHARS: int = 240      # truncate the candidate's observed output in the prompt


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ===================================================== the witness record (model-facing)

@dataclasses.dataclass(frozen=True)
class WitnessV1:
    """A single exact-oracle witness.  ``to_prompt_block`` is the ONLY model-facing text;
    it carries an oracle OUTPUT (``expected``), never the oracle PROGRAM."""

    kind: str                  # WITNESS_COUNTEREXAMPLE | WITNESS_COMPLEXITY | WITNESS_NONE
    ew_family: str             # EW1 | EW2 | EW3 | EW4 | NONE (which probe found it)
    probe_input: str           # the small failing input X (or the big stress input for EW2)
    probe_input_tokens: int
    expected_output: str       # ref(X) — for COUNTEREXAMPLE; "" for COMPLEXITY (not leaked huge)
    observed_output: str       # candidate(X), truncated; "" if TIMEOUT
    observed_kind: str         # OBS_WRONG_ANSWER | OBS_TIMEOUT | OBS_RUNTIME_ERROR
    big_n_tokens: int          # for COMPLEXITY: the stress input token count
    ref_runtime_s: float       # for COMPLEXITY: ref's wall-time on the stress input
    cand_runtime_s: float      # for COMPLEXITY: candidate wall-time (>= timeout if it TLEs)
    shrink_steps: int          # how many shrink reductions produced the reported minimal X
    leakage_clean: bool        # probe_input is NOT byte-equal to any graded secret-case input

    def found(self) -> bool:
        return self.kind in (WITNESS_COUNTEREXAMPLE, WITNESS_COMPLEXITY)

    def to_prompt_block(self) -> str:
        """The sanctioned feedback object shown to the model.  Contains ONLY an input, the
        candidate's observed behaviour, and (for a counterexample) the oracle's expected
        OUTPUT — never any solver source."""
        if self.kind == WITNESS_COUNTEREXAMPLE:
            obs = (f"your program TIMED OUT" if self.observed_kind == OBS_TIMEOUT
                   else f"your program crashed (runtime error)"
                   if self.observed_kind == OBS_RUNTIME_ERROR
                   else f"your program printed:\n{self.observed_output}")
            return (
                "Exact counterexample (a concrete input your program gets wrong; the hidden "
                "tests are NOT shown, but this small input exposes the same bug):\n"
                f"Input:\n{self.probe_input.rstrip()}\n\n"
                f"Correct output (from the reference):\n{self.expected_output.rstrip()}\n\n"
                f"On that input {obs}\n\n"
                "Your program passes the public samples but is WRONG on this input. Find the "
                "general bug (do NOT special-case this one input — the hidden tests use "
                "different inputs that trigger the same bug) and rewrite the program.")
        if self.kind == WITNESS_COMPLEXITY:
            return (
                "Complexity witness (your program is too SLOW, not wrong):\n"
                f"On a large valid input (about {self.big_n_tokens} numbers) your program did "
                f"NOT finish within {COMPLEXITY_PROBE_TIMEOUT_S:.1f} s, while a correct "
                f"reference solves the same input in {self.ref_runtime_s:.3f} s. Your output "
                "is correct on small inputs, so the algorithm is right but its time complexity "
                "is too high (likely O(N^2) or worse). Redesign it with a faster algorithm "
                "(e.g. sorting + two pointers, prefix sums, a hash map, a Fenwick/BIT, or a "
                "monotonic stack) so it runs in about O(N log N) or O(N).")
        return ("No small counterexample was found and the program finished quickly; "
                "re-examine the problem statement for an overlooked case.")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "ew_family": self.ew_family,
                "probe_input_tokens": int(self.probe_input_tokens),
                "observed_kind": self.observed_kind,
                "expected_output_len": len(self.expected_output),
                "observed_output_len": len(self.observed_output),
                "big_n_tokens": int(self.big_n_tokens),
                "ref_runtime_s": round(float(self.ref_runtime_s), 4),
                "cand_runtime_s": round(float(self.cand_runtime_s), 4),
                "shrink_steps": int(self.shrink_steps),
                "leakage_clean": bool(self.leakage_clean),
                "probe_input_sha256": _sha256_hex(self.probe_input)}


def _none_witness() -> WitnessV1:
    return WitnessV1(kind=WITNESS_NONE, ew_family="NONE", probe_input="",
                     probe_input_tokens=0, expected_output="", observed_output="",
                     observed_kind="", big_n_tokens=0, ref_runtime_s=0.0,
                     cand_runtime_s=0.0, shrink_steps=0, leakage_clean=True)


# ===================================================== the deterministic witness probe set

@dataclasses.dataclass(frozen=True)
class WitnessProbeSetV1:
    """A per-problem, content-addressed bundle of fresh witness inputs (disjoint from the
    graded secret cases) with their reference answer keys pre-computed once.

    ``small`` = ``(input, ref_output)`` pairs whose input is <= SMALL_PROBE_TOKEN_CAP tokens
    (the counterexample search space).  ``big`` = the largest fresh input + ref's wall-time on
    it (the complexity-witness stress case), or ``None`` if the family has no large input.
    """

    problem_id: str
    witness_seed: int
    small: tuple[tuple[str, str], ...]
    big_input: Optional[str]
    big_ref_runtime_s: float
    big_input_tokens: int
    secret_input_set_size: int

    def cid(self) -> str:
        return _sha256_hex({"kind": "w133_witness_probe_set_v1",
                            "problem_id": self.problem_id,
                            "witness_seed": int(self.witness_seed),
                            "small_inputs": [s[0] for s in self.small],
                            "big_input": self.big_input or ""})

    def to_dict(self) -> dict[str, Any]:
        return {"problem_id": self.problem_id, "witness_seed": int(self.witness_seed),
                "n_small": len(self.small),
                "has_big": self.big_input is not None,
                "big_input_tokens": int(self.big_input_tokens),
                "big_ref_runtime_s": round(float(self.big_ref_runtime_s), 4),
                "secret_input_set_size": int(self.secret_input_set_size),
                "probe_set_cid": self.cid()}


def build_witness_probe_set_v1(template: MintedTemplateV1, problem: MintedProblemV1, *,
                               witness_seed: int,
                               timeout_s: float = COMPLEXITY_PROBE_TIMEOUT_S,
                               ) -> WitnessProbeSetV1:
    """Deterministically build the fresh witness probe set for one problem.

    The probe inputs are produced by re-running the template's OWN ``gen_public`` /
    ``gen_hidden`` generators under a FRESH ``witness_seed`` stream (so they are the same
    *distribution* as the problem's cases but DIFFERENT instances), then split into small
    (counterexample search) and big (complexity stress).  ``ref_source`` is executed to label
    the small inputs (the answer key); the big input is timed once.  Any probe input that
    collides byte-for-byte with a graded secret-case input is dropped (structural disjointness).
    """
    secret_inputs = {inp for inp, _ in problem.secret_cases}
    rng = random.Random(_sha256_hex(
        {"witness_probe": True, "name": template.name, "seed": int(witness_seed)}))
    raw: list[str] = []
    for _ in range(N_SMALL_PROBE_ROUNDS):
        try:
            raw.extend(template.gen_public(rng))
            raw.extend(template.gen_hidden(rng))
        except Exception:
            break

    seen: set[str] = set()
    small: list[tuple[str, str]] = []
    big_input: Optional[str] = None
    big_tokens = 0
    for inp in raw:
        if inp in seen:
            continue
        seen.add(inp)
        if _tok_count(inp) <= SMALL_PROBE_TOKEN_CAP:
            # a COUNTEREXAMPLE reveals (input, expected_output), so the small probe inputs
            # MUST be disjoint from the graded secret cases (no teaching-to-the-test).
            if inp in secret_inputs:
                continue
            r = _exec_capture_v1(template.ref_source, inp, timeout_s=8.0)
            if not r.timed_out and r.returncode == 0:
                small.append((inp, r.stdout))
        else:
            # a COMPLEXITY witness reveals ONLY a timing fact + the input SIZE (never the
            # input bytes and never an expected output — see WitnessV1.to_prompt_block), so a
            # big stress input may coincide with a graded worst-case without leaking any
            # answer; keep the largest to drive the EW2 size-growth witness.
            t = _tok_count(inp)
            if t > big_tokens:
                big_input, big_tokens = inp, t

    big_ref_rt = 0.0
    if big_input is not None:
        t0 = time.time()
        rbig = _exec_capture_v1(template.ref_source, big_input, timeout_s=8.0)
        big_ref_rt = float(time.time() - t0)
        if rbig.timed_out or rbig.returncode != 0:
            big_input, big_tokens, big_ref_rt = None, 0, 0.0

    # smallest-first so the counterexample search reports the most minimal input
    small.sort(key=lambda s: (_tok_count(s[0]), s[0]))
    return WitnessProbeSetV1(
        problem_id=problem.problem_id, witness_seed=int(witness_seed),
        small=tuple(small), big_input=big_input, big_ref_runtime_s=big_ref_rt,
        big_input_tokens=int(big_tokens), secret_input_set_size=len(secret_inputs))


# ===================================================== witness search (the EW slate)

def _grade_one(code: str, inp: str, expected: str, kind: str, float_tol: float,
               timeout_s: float) -> tuple[str, str, float]:
    """Run ``code`` on ``inp``; return (observed_kind, observed_stdout, wall_s).
    observed_kind == "" means it PASSED (matched ``expected``)."""
    t0 = time.time()
    r = _exec_capture_v1(code, inp, timeout_s=timeout_s)
    dt = float(time.time() - t0)
    if r.timed_out:
        return (OBS_TIMEOUT, "", dt)
    if r.returncode != 0:
        return (OBS_RUNTIME_ERROR, "", dt)
    if judge_icpc_output_v1(got_stdout=r.stdout, expected=expected, kind=kind,
                            float_tol=float(float_tol)):
        return ("", r.stdout, dt)
    return (OBS_WRONG_ANSWER, r.stdout, dt)


def _shrink_counterexample(code: str, inp: str, expected_fn, kind: str, float_tol: float,
                           timeout_s: float, max_steps: int = 24) -> tuple[str, str, str, int]:
    """Deterministically shrink a failing input toward a minimal one by dropping trailing
    array elements (and fixing the leading count header if present), keeping the reduction
    iff the candidate still disagrees with the reference on it.  Returns
    (min_input, expected_on_min, observed_kind_on_min, n_steps)."""
    cur = inp
    cur_exp = expected_fn(cur)
    cur_obs_kind, _, _ = _grade_one(code, cur, cur_exp, kind, float_tol, timeout_s)
    steps = 0
    lines = cur.split("\n")
    # Only attempt the common "<header...>\n<space-separated array>" shape.
    if len(lines) >= 2 and lines[-1].split():
        for _ in range(max_steps):
            toks = lines[-1].split()
            if len(toks) <= 1:
                break
            cand_toks = toks[:-1]
            head = lines[:-1]
            # if the first header token equals the old array length, decrement it
            htoks = head[0].split() if head else []
            if htoks and htoks[0].isdigit() and int(htoks[0]) == len(toks):
                htoks[0] = str(len(cand_toks))
                head = [" ".join(htoks)] + head[1:]
            cand = "\n".join(head + [" ".join(cand_toks)])
            exp = expected_fn(cand)
            if exp is None:
                break
            ok_kind, _, _ = _grade_one(code, cand, exp, kind, float_tol, timeout_s)
            if ok_kind:                 # still failing -> accept the smaller input
                cur, cur_exp, cur_obs_kind = cand, exp, ok_kind
                lines = cand.split("\n")
                steps += 1
            else:
                break
    return (cur, cur_exp, cur_obs_kind, steps)


def find_counterexample_witness_v1(code: str, problem: MintedProblemV1,
                                   probe: WitnessProbeSetV1, template: MintedTemplateV1, *,
                                   timeout_s: float = COMPLEXITY_PROBE_TIMEOUT_S) -> WitnessV1:
    """EW1/EW3/EW4: find the SMALLEST fresh input on which the candidate disagrees with the
    reference, with a deterministic shrink trace.  Returns WITNESS_NONE if the candidate is
    value-correct on every small probe input."""
    secret_inputs = {inp for inp, _ in problem.secret_cases}

    def _ref_expected(inp: str) -> Optional[str]:
        r = _exec_capture_v1(template.ref_source, inp, timeout_s=8.0)
        if r.timed_out or r.returncode != 0:
            return None
        return r.stdout

    for inp, expected in probe.small:                 # already smallest-first
        obs_kind, obs_out, _ = _grade_one(
            code, inp, expected, problem.kind, problem.float_tol, timeout_s)
        if not obs_kind:
            continue
        min_inp, min_exp, min_kind, steps = _shrink_counterexample(
            code, inp, _ref_expected, problem.kind, problem.float_tol, timeout_s)
        obs_show = ""
        if min_kind == OBS_WRONG_ANSWER:
            _, obs_show, _ = _grade_one(code, min_inp, min_exp, problem.kind,
                                        problem.float_tol, timeout_s)
            obs_show = obs_show[:MAX_OBSERVED_CHARS]
        return WitnessV1(
            kind=WITNESS_COUNTEREXAMPLE, ew_family="EW1",
            probe_input=min_inp, probe_input_tokens=_tok_count(min_inp),
            expected_output=min_exp, observed_output=obs_show, observed_kind=min_kind,
            big_n_tokens=0, ref_runtime_s=0.0, cand_runtime_s=0.0, shrink_steps=int(steps),
            leakage_clean=bool(min_inp not in secret_inputs))
    return _none_witness()


def find_complexity_witness_v1(code: str, problem: MintedProblemV1,
                               probe: WitnessProbeSetV1, *,
                               timeout_s: float = COMPLEXITY_PROBE_TIMEOUT_S) -> WitnessV1:
    """EW2: a too-slow witness.  Requires a big fresh input on which the reference is fast
    (< REF_FAST_CEILING_S) and the candidate does NOT finish within ``timeout_s``."""
    # ref must finish within the candidate's probe window (a sanity bound on the reference,
    # robust to host load — NOT a tight absolute ceiling, which was load-fragile); the genuine
    # speed gap is enforced relatively by the 8x-ref `too_slow` test below.
    if probe.big_input is None or probe.big_ref_runtime_s >= float(timeout_s):
        return _none_witness()
    t0 = time.time()
    r = _exec_capture_v1(code, probe.big_input, timeout_s=timeout_s)
    dt = float(time.time() - t0)
    too_slow = bool(r.timed_out or dt >= max(timeout_s, 8.0 * probe.big_ref_runtime_s))
    if not too_slow:
        return _none_witness()
    # EW2 reveals ONLY a timing fact + the input SIZE (to_prompt_block shows neither the input
    # bytes nor any expected output), so it is structurally leakage-clean even when the stress
    # input coincides with a graded worst-case (no answer is ever shown).
    return WitnessV1(
        kind=WITNESS_COMPLEXITY, ew_family="EW2",
        probe_input=probe.big_input, probe_input_tokens=int(probe.big_input_tokens),
        expected_output="", observed_output="", observed_kind=OBS_TIMEOUT,
        big_n_tokens=int(probe.big_input_tokens), ref_runtime_s=float(probe.big_ref_runtime_s),
        cand_runtime_s=float(dt), shrink_steps=0, leakage_clean=True)


def select_witness_v1(code: str, problem: MintedProblemV1, probe: WitnessProbeSetV1,
                      template: MintedTemplateV1, *, arm: str,
                      timeout_s: float = COMPLEXITY_PROBE_TIMEOUT_S) -> WitnessV1:
    """The witness policy for a given arm.

    * C1 — counterexample only (EW1/EW3/EW4).
    * C2 — complexity witness only (EW2).
    * C3 (LEAD) — behaviour-routed controller: try the value counterexample first; if the
      candidate is value-correct on every small probe (no counterexample) AND there is a big
      input it is too slow on, emit the complexity witness.  The route is chosen from OBSERVED
      behaviour, never from a leaked mode label.
    """
    if arm == ARM_C1_COUNTEREXAMPLE:
        return find_counterexample_witness_v1(code, problem, probe, template, timeout_s=timeout_s)
    if arm == ARM_C2_COMPLEXITY:
        return find_complexity_witness_v1(code, problem, probe, timeout_s=timeout_s)
    # C3 controller
    w = find_counterexample_witness_v1(code, problem, probe, template, timeout_s=timeout_s)
    if w.found():
        return w
    return find_complexity_witness_v1(code, problem, probe, timeout_s=timeout_s)


# ===================================================== witness-reflexion prompt + arm

def _witness_reflexion_prompt(problem, history, witness: WitnessV1, attempt_idx: int) -> str:
    """The between-attempt prompt: the SAME scaffold as the W120 reflexion prompt (judge
    verdict bit + stderr tail + public-sample results) PLUS the exact-oracle witness block —
    a strict superset of the blind feedback, so any gain is attributable to the witness."""
    chunks: list[str] = []
    for i, (cand, passed, stderr_tail, sample_fb) in enumerate(history):
        cand_trim = cand if len(cand) <= 1500 else (cand[:1500] + "\n# ...(truncated)\n")
        verdict = ("ACCEPTED by the judge (all hidden tests passed)" if passed
                   else "REJECTED by the judge (failed at least one hidden test)")
        se = f"\nExecutor stderr (tail):\n{stderr_tail.strip()}" if stderr_tail.strip() else ""
        sf = f"\nPublic sample results:\n{sample_fb}" if sample_fb.strip() else ""
        chunks.append(f"--- Attempt {i+1} ({verdict}) ---\n"
                      f"```python\n{cand_trim}\n```{se}{sf}")
    wblock = witness.to_prompt_block()
    return (
        "You are an expert ICPC competitor on a reflective debugging loop. You are on "
        f"attempt {attempt_idx + 1} out of 5. Below are your previous attempts with the judge "
        "verdict and the PUBLIC sample-case results, followed by an EXACT diagnostic of the "
        "last attempt's failure. Use the diagnostic to produce a NEW corrected COMPLETE "
        "Python 3 stdin/stdout program. Do not repeat a previous attempt verbatim.\n\n"
        f"Problem:\n{problem.statement}\n\n"
        f"{_samples_block(problem)}\n\n"
        f"{chr(10).join(chunks)}\n\n"
        f"=== Exact diagnostic of attempt {len(history)} ===\n{wblock}\n\n"
        "Provide ONLY the corrected complete Python 3 program in a ```python ... ``` fence:")


@dataclasses.dataclass(frozen=True)
class WitnessArmTraceV1:
    """Per-problem audit of one witness arm: which witness fired on each invoked attempt."""
    problem_id: str
    arm_id: str
    witness_kinds: tuple[str, ...]          # witness kind used to seed each attempt >=1
    witness_ew_families: tuple[str, ...]
    witness_observed_kinds: tuple[str, ...]  # OBS_* the witness reported per attempt
    any_witness_found: bool
    all_leakage_clean: bool

    def rescue_is_algorithmic(self) -> bool:
        """True iff any witness reported a WRONG_ANSWER or TIMEOUT failure (an algorithmic /
        complexity bug) — i.e. the arm's effect is NOT only trivial parse/crash repair."""
        return any(ok in (OBS_WRONG_ANSWER, OBS_TIMEOUT)
                   for ok in self.witness_observed_kinds)

    def to_dict(self) -> dict[str, Any]:
        return {"problem_id": self.problem_id, "arm_id": self.arm_id,
                "witness_kinds": list(self.witness_kinds),
                "witness_ew_families": list(self.witness_ew_families),
                "witness_observed_kinds": list(self.witness_observed_kinds),
                "any_witness_found": bool(self.any_witness_found),
                "all_leakage_clean": bool(self.all_leakage_clean)}


def run_witness_arm_v1(*, seed: int, template: MintedTemplateV1, problem: MintedProblemV1,
                       probe: WitnessProbeSetV1, gen, K: int, temperature: float,
                       max_tokens: int, timeout_s: float, arm: str, minted_date: str,
                       witness_timeout_s: float = COMPLEXITY_PROBE_TIMEOUT_S,
                       ) -> tuple[IcpcArmOutcomeV1, WitnessArmTraceV1]:
    """Same-budget witness-guided reflexion arm.  Byte-identical structure to the W120 ``_run_b``
    (attempt-0 = the standard initial prompt; K attempts total; one model call per attempt; no
    early stop), except the between-attempt feedback object is the exact-oracle witness.

    The model under test is graded ONLY on ``problem.secret_cases`` (the audited grader), and
    the witness is computed from a FRESH disjoint probe set — never the graded cases.
    """
    pilot = problem.to_pilot_problem(minted_date=str(minted_date))  # statement+samples only
    history: list[tuple[str, bool, str, str]] = []
    per_call: list[bool] = []
    first_pass_idx = -1
    wkinds: list[str] = []
    wfams: list[str] = []
    wobs: list[str] = []
    leak_clean = True
    for k in range(int(K)):
        if k == 0:
            prompt = _initial_prompt(pilot)
        else:
            last_code = history[-1][0]
            witness = select_witness_v1(last_code, problem, probe, template, arm=arm,
                                        timeout_s=witness_timeout_s)
            wkinds.append(witness.kind)
            wfams.append(witness.ew_family)
            if witness.observed_kind:
                wobs.append(witness.observed_kind)
            if witness.found() and not witness.leakage_clean:
                leak_clean = False
            prompt = _witness_reflexion_prompt(pilot, tuple(history), witness, attempt_idx=k)
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
        question_id=problem.problem_id, arm_id=str(arm),
        final_passed=bool(first_pass_idx >= 0), n_model_calls=int(K),
        per_call_passed=tuple(per_call), first_pass_attempt_idx=int(first_pass_idx))
    trace = WitnessArmTraceV1(
        problem_id=problem.problem_id, arm_id=str(arm),
        witness_kinds=tuple(wkinds), witness_ew_families=tuple(wfams),
        witness_observed_kinds=tuple(wobs),
        any_witness_found=any(kk in (WITNESS_COUNTEREXAMPLE, WITNESS_COMPLEXITY)
                              for kk in wkinds),
        all_leakage_clean=bool(leak_clean))
    return outcome, trace


# ===================================================== fake-different / self-test surface

def witness_is_genuinely_new_v1(witness: WitnessV1, problem: MintedProblemV1) -> dict[str, Any]:
    """Machine-checkable 'not just judge-bit-plus-words' test.  A witness is genuinely NEW
    iff it carries information the blind W120 feedback structurally cannot: a failing input
    that is NOT one of the public samples, plus (for a counterexample) an oracle-computed
    expected output.  Returns the decision + evidence."""
    sample_inputs = {inp for inp, _ in problem.samples}
    is_new_input = bool(witness.found() and witness.probe_input not in sample_inputs)
    carries_oracle_output = bool(
        witness.kind == WITNESS_COMPLEXITY
        or (witness.kind == WITNESS_COUNTEREXAMPLE and witness.expected_output != ""))
    genuinely_new = bool(witness.found() and is_new_input and carries_oracle_output
                         and witness.leakage_clean)
    return {"genuinely_new": genuinely_new, "found": witness.found(),
            "input_not_a_public_sample": is_new_input,
            "carries_oracle_output": carries_oracle_output,
            "leakage_clean": bool(witness.leakage_clean), "kind": witness.kind}


__all__ = [
    "W133_EXACT_ORACLE_WITNESS_V1_SCHEMA_VERSION",
    "WITNESS_COUNTEREXAMPLE", "WITNESS_COMPLEXITY", "WITNESS_NONE",
    "ARM_C1_COUNTEREXAMPLE", "ARM_C2_COMPLEXITY", "ARM_C3_CONTROLLER",
    "OBS_WRONG_ANSWER", "OBS_TIMEOUT", "OBS_RUNTIME_ERROR",
    "WitnessV1", "WitnessProbeSetV1", "build_witness_probe_set_v1",
    "find_counterexample_witness_v1", "find_complexity_witness_v1", "select_witness_v1",
    "WitnessArmTraceV1", "run_witness_arm_v1", "witness_is_genuinely_new_v1",
]
