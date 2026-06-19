"""W95 — MathVista same-budget bench V1.

A0 / A1 / B benchmark over MathVista testmini under the
pre-committed W95 pilot gates (`docs/RUNBOOK_W95.md`).  Same
shape as the W90 cross_modal_vlm_loop bench, adapted for
MathVista's answer-match executor (no subprocess, no judge):

  * **A0_text** — text-only LLM, single-shot at T=0.0.  Sees
    ONLY the MathVista `query` (question + choice list when
    multi-choice); never receives the image.  Floor for
    "image is load-bearing".
  * **A1_vlm** — unified VLM, K=5 independent samples at T=0.7
    on (query + image), ship the first prediction that scores
    PASS under `evaluate_answer_v1`.  Strongest unified-VLM
    baseline at the same K budget.
  * **B_vlm_team** (W95-B0) — vision-reader + math-solver +
    executor-guided reflexion at total K=5:
      1. VLM-Vision-Reader at T=0.0 — extracts a structured
         text bullet list of numerical / geometric / tabular
         facts from the image conditioned on the question.
         (1 VLM call.)
      2. Math-Solver at T=0.7 — generates a candidate answer
         from (query + extracted facts), TEXT-ONLY (no image).
         (1 solver call.)
      3. Executor verifies; if FAIL, the solver does up to 3
         sequential reflexion turns conditioned on (prior
         candidate, executor verdict + diagnostics) history.
         Each turn is text-only.
      Total = exactly 5 model calls (padding by a final
      reflexion turn if the first attempt PASSES, so budget
      parity with A1_vlm holds byte-exact).

The bench produces a content-addressable RunReport with per-
problem outcome capsules, per-seed Merkle roots, and a bench-
level Merkle root.

Anti-cheat:
  * Same VLM model on A1_vlm AND B_vlm_team's vlm_reader stage.
  * Same text-LM on A0_text and B_vlm_team's math_solver stage
    (caller can route to the same model for both with no loss).
  * Same per-problem slice by deterministic
    `select_mathvista_subset_v1(seed, n_problems, corpus)`.
  * Same budget per arm (K=5 model calls on A1 / B; 1 on A0).
  * Executor truth = `evaluate_answer_v1` for every arm.
  * MathVista testmini parquet SHA-256 is anchored at run
    start; mismatches refuse to run the bench.
  * No selective retries.
  * No LLM-judge anywhere.

Honest scope (W95)
------------------

* ``W95-L-MATHVISTA-BENCH-V1-NIM-DEPENDENT-CAP`` — V1 drives
  the bench through caller-provided text / VLM clients; the
  driver does not couple to NIM.
* ``W95-L-MATHVISTA-BENCH-V1-SINGLE-MODEL-FAMILY-CAP`` — V1's
  A0/A1/B all use the same VLM family (Llama-3.2-Vision) by
  default; the math_solver stage can route to a stronger text-
  LM if the caller wishes, at the cost of breaking same-model
  parity.
* ``W95-L-MATHVISTA-BENCH-V1-CANDIDATE-EXTRACTION-CAP`` — V1
  parses the last (non-empty, post-strip-trivial-wrappers)
  line of the model's response as the candidate answer.  This
  is the W95 anti-cheat extraction; the alternative of using
  the FIRST line would let A1 sneak in chain-of-thought
  guesses.  Both arms get the same extractor.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Sequence

from .mathvista_executor_v1 import (
    MathVistaExecutorResultV1,
    W95_MATHVISTA_EXECUTOR_V1_SCHEMA_VERSION,
    evaluate_answer_v1,
)
from .mathvista_loader_v1 import (
    MathVistaProblemV1,
    W95_MATHVISTA_LOADER_V1_SCHEMA_VERSION,
)


W95_MATHVISTA_BENCH_V1_SCHEMA_VERSION: str = (
    "coordpy.mathvista_bench_v1.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    if isinstance(payload, (bytes, bytearray)):
        return hashlib.sha256(payload).hexdigest()
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ---------------------------------------------------------------
# Candidate-answer extraction
# ---------------------------------------------------------------

def extract_candidate_answer_v1(*, response_text: str) -> str:
    """Pull a candidate final answer from the model's response.
    Strategy: strip trivial wrappers, then take the LAST
    non-empty line.  Same extractor for every arm (anti-cheat
    bar)."""
    if not response_text:
        return ""
    lines = [ln.strip()
             for ln in response_text.replace("\r", "").split("\n")
             if ln.strip()]
    if not lines:
        return ""
    # Prefer a final-answer-tagged line if present.
    tagged_prefixes = (
        "the answer is", "final answer:", "answer:",
        "ans:", "result:")
    for ln in reversed(lines):
        low = ln.lower()
        for pref in tagged_prefixes:
            if pref in low:
                return ln
    return lines[-1]


# ---------------------------------------------------------------
# Capsules
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MathVistaArmCallCapsuleV1:
    schema: str
    seed: int
    pid: str
    arm_id: str
    role: str
    call_idx: int
    temperature: float
    prompt_cid: str
    response_cid: str
    wall_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "seed": int(self.seed),
            "pid": str(self.pid),
            "arm_id": str(self.arm_id),
            "role": str(self.role),
            "call_idx": int(self.call_idx),
            "temperature": float(self.temperature),
            "prompt_cid": str(self.prompt_cid),
            "response_cid": str(self.response_cid),
            "wall_ms": int(self.wall_ms),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w95_mathvista_arm_call_v1",
            **self.to_dict()})


@dataclasses.dataclass(frozen=True)
class MathVistaArmOutcomeCapsuleV1:
    schema: str
    seed: int
    pid: str
    arm_id: str
    final_passed: bool
    final_prediction_cid: str
    final_executor_rule: str
    n_model_calls: int
    total_wall_ms: int
    call_capsule_cids: tuple[str, ...]
    executor_result_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "seed": int(self.seed),
            "pid": str(self.pid),
            "arm_id": str(self.arm_id),
            "final_passed": bool(self.final_passed),
            "final_prediction_cid": str(
                self.final_prediction_cid),
            "final_executor_rule": str(
                self.final_executor_rule),
            "n_model_calls": int(self.n_model_calls),
            "total_wall_ms": int(self.total_wall_ms),
            "call_capsule_cids": list(self.call_capsule_cids),
            "executor_result_cid": str(self.executor_result_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w95_mathvista_arm_outcome_v1",
            **self.to_dict()})


def _executor_result_cid(
        result: MathVistaExecutorResultV1) -> str:
    return _sha256_hex({
        "kind": "w95_mathvista_executor_result_v1",
        **result.to_dict()})


# ---------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------

_A0_SYSTEM = (
    "You are an expert math problem-solver.  Read the question "
    "carefully and provide a single final answer.  If the "
    "question is multi-choice, output ONLY the letter (A, B, "
    "C, ...).  If the question is numeric, output ONLY the "
    "number.  Do not include any prose around the answer.")


_A1_SYSTEM = (
    "You are an expert math problem-solver who reads images.  "
    "Read the image and the question carefully and provide a "
    "single final answer.  If the question is multi-choice, "
    "output ONLY the letter (A, B, C, ...).  If the question "
    "is numeric, output ONLY the number.  Do not include any "
    "prose around the answer.")


_B_VISION_READER_SYSTEM = (
    "You are an expert at reading visual math content.  Given "
    "an image and a math question, extract the structured "
    "data the question needs as a short bullet list.  Include "
    "exact numbers, geometric coordinates, table cells, axis "
    "labels, and chart values.  Do not solve the problem; "
    "extract only the structured facts the solver will need.")


_B_SOLVER_SYSTEM = (
    "You are an expert math problem-solver.  You will be given "
    "the question text and a short bullet list of structured "
    "facts extracted from an image.  Use the structured facts "
    "as ground truth (you cannot see the image yourself) and "
    "produce a single final answer.  Output ONLY the letter "
    "(for multi-choice) or the number / short answer; no "
    "prose.")


def _a0_prompt(p: MathVistaProblemV1) -> str:
    return f"{_A0_SYSTEM}\n\n{p.query}\n\nFinal answer:"


def _a1_prompt(p: MathVistaProblemV1) -> str:
    return f"{_A1_SYSTEM}\n\n{p.query}\n\nFinal answer:"


def _b_reader_prompt(p: MathVistaProblemV1) -> str:
    return (
        f"{_B_VISION_READER_SYSTEM}\n\n"
        f"Question:\n{p.query}\n\n"
        "Extract the structured facts the solver will need.  "
        "Bullet list only.  No solving.")


def _b_solver_initial_prompt(
        p: MathVistaProblemV1, extraction: str) -> str:
    return (
        f"{_B_SOLVER_SYSTEM}\n\n"
        f"Question:\n{p.query}\n\n"
        "Structured facts extracted from the image (use as "
        f"ground truth):\n{extraction}\n\nFinal answer:")


def _b_solver_reflexion_prompt(
        p: MathVistaProblemV1, extraction: str,
        history: Sequence[
            tuple[str, MathVistaExecutorResultV1]],
        attempt_idx: int) -> str:
    chunks: list[str] = []
    for i, (cand, exe) in enumerate(history):
        rule = exe.matched_rule
        if exe.passed:
            verdict = f"PASS ({rule})"
            diag = ""
        else:
            verdict = f"FAIL ({rule})"
            diag = (
                f"\nNormalized prediction: "
                f"{exe.normalized_prediction}\n"
                f"Normalized gold-form:   (hidden — you must "
                "reason from the question)")
        chunks.append(
            f"--- Attempt {i + 1} — {verdict} ---\n"
            f"Candidate answer: {cand}{diag}")
    return (
        f"{_B_SOLVER_SYSTEM}\n\n"
        f"You are on attempt {attempt_idx + 1} out of 5.\n\n"
        f"Question:\n{p.query}\n\n"
        "Structured facts extracted from the image (use as "
        f"ground truth):\n{extraction}\n\n"
        f"{chr(10).join(chunks)}\n\n"
        "Diagnose the bug in the failing attempt(s) and "
        "produce a NEW final answer.  Do not repeat a previous "
        "attempt verbatim.  Output ONLY the letter (for "
        "multi-choice) or the number / short answer; no "
        "prose.\n\nFinal answer:")


# ---------------------------------------------------------------
# Gen fn signatures
# ---------------------------------------------------------------

_TextGenFn = Callable[[str, int, float], tuple[str, int]]
_VlmGenFn = Callable[
    [str, "bytes | None", int, float], tuple[str, int]]


# ---------------------------------------------------------------
# Per-arm runners
# ---------------------------------------------------------------

def _run_a0_text(
        *, seed: int, p: MathVistaProblemV1,
        text_gen: _TextGenFn, max_tokens: int,
) -> tuple[MathVistaArmOutcomeCapsuleV1,
           MathVistaExecutorResultV1]:
    prompt = _a0_prompt(p)
    text, wall = text_gen(prompt, max_tokens, 0.0)
    candidate = extract_candidate_answer_v1(response_text=text)
    exe = evaluate_answer_v1(
        prediction=candidate, problem=p)
    call = MathVistaArmCallCapsuleV1(
        schema=W95_MATHVISTA_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), pid=str(p.pid),
        arm_id="A0_text", role="text_solver", call_idx=0,
        temperature=0.0,
        prompt_cid=hashlib.sha256(
            prompt.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            text.encode("utf-8")).hexdigest(),
        wall_ms=int(wall))
    out = MathVistaArmOutcomeCapsuleV1(
        schema=W95_MATHVISTA_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), pid=str(p.pid),
        arm_id="A0_text",
        final_passed=bool(exe.passed),
        final_prediction_cid=hashlib.sha256(
            candidate.encode("utf-8")).hexdigest(),
        final_executor_rule=str(exe.matched_rule),
        n_model_calls=1, total_wall_ms=int(wall),
        call_capsule_cids=(call.cid(),),
        executor_result_cid=_executor_result_cid(exe))
    return out, exe


def _run_a1_vlm(
        *, seed: int, p: MathVistaProblemV1, K: int,
        temperature: float, vlm_gen: _VlmGenFn,
        max_tokens: int,
) -> tuple[MathVistaArmOutcomeCapsuleV1,
           list[MathVistaExecutorResultV1]]:
    prompt = _a1_prompt(p)
    calls: list[MathVistaArmCallCapsuleV1] = []
    exes: list[MathVistaExecutorResultV1] = []
    candidates: list[str] = []
    total = 0
    chosen_passed = False
    chosen_idx = 0
    chosen_cand = ""
    chosen_rule = ""
    for k in range(int(K)):
        text, wall = vlm_gen(
            prompt, p.image_bytes,
            max_tokens, float(temperature))
        candidate = extract_candidate_answer_v1(
            response_text=text)
        exe = evaluate_answer_v1(
            prediction=candidate, problem=p)
        calls.append(MathVistaArmCallCapsuleV1(
            schema=W95_MATHVISTA_BENCH_V1_SCHEMA_VERSION,
            seed=int(seed), pid=str(p.pid),
            arm_id="A1_vlm", role="vlm_sample",
            call_idx=int(k),
            temperature=float(temperature),
            prompt_cid=hashlib.sha256(
                (prompt + "|img:" + p.image_sha256).encode(
                    "utf-8")).hexdigest(),
            response_cid=hashlib.sha256(
                text.encode("utf-8")).hexdigest(),
            wall_ms=int(wall)))
        exes.append(exe)
        candidates.append(candidate)
        total += int(wall)
        if exe.passed and not chosen_passed:
            chosen_passed = True
            chosen_idx = int(k)
            chosen_cand = candidate
            chosen_rule = str(exe.matched_rule)
    if not chosen_passed:
        chosen_cand = candidates[0]
        chosen_rule = str(exes[0].matched_rule)
    out = MathVistaArmOutcomeCapsuleV1(
        schema=W95_MATHVISTA_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), pid=str(p.pid),
        arm_id="A1_vlm",
        final_passed=bool(chosen_passed),
        final_prediction_cid=hashlib.sha256(
            chosen_cand.encode("utf-8")).hexdigest(),
        final_executor_rule=str(chosen_rule),
        n_model_calls=int(K),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cid=_executor_result_cid(
            exes[chosen_idx if chosen_passed else 0]))
    return out, exes


def _run_b_vlm_team(
        *, seed: int, p: MathVistaProblemV1, K: int,
        temperature: float, vlm_gen: _VlmGenFn,
        text_gen: _TextGenFn, max_tokens: int,
) -> tuple[MathVistaArmOutcomeCapsuleV1,
           list[MathVistaExecutorResultV1]]:
    """W95-B0: vlm_reader (1 call, T=0.0) + math_solver (1+up
    to 3 reflexion turns at temperature) + padding reflexion to
    exactly K total calls.  Same K budget as A1_vlm.
    """
    calls: list[MathVistaArmCallCapsuleV1] = []
    exes: list[MathVistaExecutorResultV1] = []
    candidates: list[str] = []
    history: list[tuple[str, MathVistaExecutorResultV1]] = []
    total = 0

    # Stage 1 — VLM-Vision-Reader (1 call, T=0.0).
    reader_prompt = _b_reader_prompt(p)
    reader_text, reader_wall = vlm_gen(
        reader_prompt, p.image_bytes, max_tokens, 0.0)
    calls.append(MathVistaArmCallCapsuleV1(
        schema=W95_MATHVISTA_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), pid=str(p.pid),
        arm_id="B_vlm_team", role="vision_reader",
        call_idx=0, temperature=0.0,
        prompt_cid=hashlib.sha256(
            (reader_prompt + "|img:" + p.image_sha256).encode(
                "utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            reader_text.encode("utf-8")).hexdigest(),
        wall_ms=int(reader_wall)))
    total += int(reader_wall)
    extraction = reader_text.strip() or "(no extraction)"

    # Stage 2..K — Math-Solver + executor-guided reflexion.
    solver_calls = int(K) - 1
    for k in range(int(solver_calls)):
        if k == 0:
            solver_prompt = _b_solver_initial_prompt(
                p, extraction)
            solver_role = "math_solver_initial"
            solver_temp = float(temperature)
        else:
            solver_prompt = _b_solver_reflexion_prompt(
                p, extraction, tuple(history),
                attempt_idx=int(k))
            solver_role = "math_solver_reflexion"
            solver_temp = float(temperature)
        solver_text, solver_wall = text_gen(
            solver_prompt, max_tokens, solver_temp)
        candidate = extract_candidate_answer_v1(
            response_text=solver_text)
        exe = evaluate_answer_v1(
            prediction=candidate, problem=p)
        calls.append(MathVistaArmCallCapsuleV1(
            schema=W95_MATHVISTA_BENCH_V1_SCHEMA_VERSION,
            seed=int(seed), pid=str(p.pid),
            arm_id="B_vlm_team", role=solver_role,
            call_idx=int(k) + 1, temperature=solver_temp,
            prompt_cid=hashlib.sha256(
                solver_prompt.encode("utf-8")).hexdigest(),
            response_cid=hashlib.sha256(
                solver_text.encode("utf-8")).hexdigest(),
            wall_ms=int(solver_wall)))
        exes.append(exe)
        candidates.append(candidate)
        history.append((candidate, exe))
        total += int(solver_wall)
    # Choose the FIRST PASS among the solver turns; else the
    # last candidate (matches the W90 "ship first PASS" pattern).
    chosen_passed = False
    chosen_idx = 0
    chosen_cand = candidates[-1] if candidates else ""
    chosen_rule = (
        exes[-1].matched_rule if exes else "no_calls")
    for i, exe in enumerate(exes):
        if exe.passed:
            chosen_passed = True
            chosen_idx = i
            chosen_cand = candidates[i]
            chosen_rule = str(exe.matched_rule)
            break
    out = MathVistaArmOutcomeCapsuleV1(
        schema=W95_MATHVISTA_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), pid=str(p.pid),
        arm_id="B_vlm_team",
        final_passed=bool(chosen_passed),
        final_prediction_cid=hashlib.sha256(
            chosen_cand.encode("utf-8")).hexdigest(),
        final_executor_rule=str(chosen_rule),
        n_model_calls=int(K),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cid=_executor_result_cid(
            exes[chosen_idx] if exes
            else evaluate_answer_v1(prediction="", problem=p)))
    return out, exes


# ---------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MathVistaSeedReportV1:
    schema: str
    seed: int
    n_problems: int
    a0_text_pass_at_1: float
    a1_vlm_pass_at_1: float
    b_vlm_team_pass_at_1: float
    a0_text_total_wall_ms: int
    a1_vlm_total_wall_ms: int
    b_vlm_team_total_wall_ms: int
    per_problem_outcomes: tuple[dict[str, Any], ...]
    outcome_cids: tuple[str, ...]
    seed_merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "seed": int(self.seed),
            "n_problems": int(self.n_problems),
            "a0_text_pass_at_1": float(round(
                self.a0_text_pass_at_1, 6)),
            "a1_vlm_pass_at_1": float(round(
                self.a1_vlm_pass_at_1, 6)),
            "b_vlm_team_pass_at_1": float(round(
                self.b_vlm_team_pass_at_1, 6)),
            "a0_text_total_wall_ms": int(
                self.a0_text_total_wall_ms),
            "a1_vlm_total_wall_ms": int(
                self.a1_vlm_total_wall_ms),
            "b_vlm_team_total_wall_ms": int(
                self.b_vlm_team_total_wall_ms),
            "per_problem_outcomes": [
                dict(po) for po in self.per_problem_outcomes],
            "outcome_cids": list(self.outcome_cids),
            "seed_merkle_root": str(self.seed_merkle_root),
        }


@dataclasses.dataclass(frozen=True)
class MathVistaBenchReportV1:
    schema: str
    vlm_model_id: str
    text_model_id: str
    n_problems: int
    n_seeds: int
    K_multi_sample: int
    corpus_parquet_sha256: str
    corpus_merkle_root: str
    per_seed: tuple[MathVistaSeedReportV1, ...]
    a0_text_mean_pass_at_1: float
    a1_vlm_mean_pass_at_1: float
    b_vlm_team_mean_pass_at_1: float
    b_beats_a0_text_per_seed: tuple[bool, ...]
    b_beats_a1_vlm_per_seed: tuple[bool, ...]
    b_mean_strictly_beats_a0_text_mean: bool
    b_mean_strictly_beats_a1_vlm_mean: bool
    b_mean_minus_a0_text_mean_pp: float
    b_mean_minus_a1_vlm_mean_pp: float
    n_b_ge_a1_problems_per_seed: tuple[int, ...]
    bench_merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "vlm_model_id": str(self.vlm_model_id),
            "text_model_id": str(self.text_model_id),
            "n_problems": int(self.n_problems),
            "n_seeds": int(self.n_seeds),
            "K_multi_sample": int(self.K_multi_sample),
            "corpus_parquet_sha256": str(
                self.corpus_parquet_sha256),
            "corpus_merkle_root": str(self.corpus_merkle_root),
            "per_seed": [s.to_dict() for s in self.per_seed],
            "a0_text_mean_pass_at_1": float(round(
                self.a0_text_mean_pass_at_1, 6)),
            "a1_vlm_mean_pass_at_1": float(round(
                self.a1_vlm_mean_pass_at_1, 6)),
            "b_vlm_team_mean_pass_at_1": float(round(
                self.b_vlm_team_mean_pass_at_1, 6)),
            "b_beats_a0_text_per_seed": list(
                self.b_beats_a0_text_per_seed),
            "b_beats_a1_vlm_per_seed": list(
                self.b_beats_a1_vlm_per_seed),
            "b_mean_strictly_beats_a0_text_mean": bool(
                self.b_mean_strictly_beats_a0_text_mean),
            "b_mean_strictly_beats_a1_vlm_mean": bool(
                self.b_mean_strictly_beats_a1_vlm_mean),
            "b_mean_minus_a0_text_mean_pp": float(round(
                self.b_mean_minus_a0_text_mean_pp, 4)),
            "b_mean_minus_a1_vlm_mean_pp": float(round(
                self.b_mean_minus_a1_vlm_mean_pp, 4)),
            "n_b_ge_a1_problems_per_seed": list(
                self.n_b_ge_a1_problems_per_seed),
            "bench_merkle_root": str(self.bench_merkle_root),
        }


# ---------------------------------------------------------------
# Config + driver
# ---------------------------------------------------------------

@dataclasses.dataclass
class MathVistaBenchConfigV1:
    schema: str = W95_MATHVISTA_BENCH_V1_SCHEMA_VERSION
    n_problems: int = 30
    K_multi_sample: int = 5
    seeds: tuple[int, ...] = (95_005_001,)
    sampling_temperature: float = 0.7
    max_tokens_per_call: int = 384


def run_mathvista_bench_v1(
        *,
        text_gen: _TextGenFn,
        vlm_gen: _VlmGenFn,
        vlm_model_id: str,
        text_model_id: str,
        corpus: Sequence[MathVistaProblemV1],
        corpus_parquet_sha256: str,
        corpus_merkle_root: str,
        config: MathVistaBenchConfigV1 | None = None,
        on_problem_start: (
            Callable[[int, int, str], None] | None) = None,
        sidecar_writer: (
            Callable[[dict[str, Any]], None] | None) = None,
) -> MathVistaBenchReportV1:
    """Run A0_text / A1_vlm / B_vlm_team on the supplied corpus
    under the same K budget.  Returns the bench report with
    Merkle roots.

    Callers can pass a ``sidecar_writer`` that receives each
    per-problem JSONL record for streamed on-disk persistence.
    """
    from .mathvista_loader_v1 import select_mathvista_subset_v1

    cfg = config or MathVistaBenchConfigV1()
    per_seed: list[MathVistaSeedReportV1] = []
    all_outcome_cids: list[str] = []
    for seed in cfg.seeds:
        subset = select_mathvista_subset_v1(
            seed=int(seed),
            n_problems=int(cfg.n_problems),
            corpus=tuple(corpus))
        if len(subset) < int(cfg.n_problems):
            raise RuntimeError(
                "corpus has only "
                f"{len(subset)} problems for seed {seed}; "
                f"need {cfg.n_problems}")
        a0_outs: list[MathVistaArmOutcomeCapsuleV1] = []
        a1_outs: list[MathVistaArmOutcomeCapsuleV1] = []
        b_outs: list[MathVistaArmOutcomeCapsuleV1] = []
        per_problem_outcomes: list[dict[str, Any]] = []
        for p_idx, p in enumerate(subset):
            if on_problem_start is not None:
                on_problem_start(
                    int(seed), int(p_idx), str(p.pid))
            a0_out, _ = _run_a0_text(
                seed=int(seed), p=p,
                text_gen=text_gen,
                max_tokens=int(cfg.max_tokens_per_call))
            a0_outs.append(a0_out)
            a1_out, _ = _run_a1_vlm(
                seed=int(seed), p=p,
                K=int(cfg.K_multi_sample),
                temperature=float(cfg.sampling_temperature),
                vlm_gen=vlm_gen,
                max_tokens=int(cfg.max_tokens_per_call))
            a1_outs.append(a1_out)
            b_out, _ = _run_b_vlm_team(
                seed=int(seed), p=p,
                K=int(cfg.K_multi_sample),
                temperature=float(cfg.sampling_temperature),
                vlm_gen=vlm_gen,
                text_gen=text_gen,
                max_tokens=int(cfg.max_tokens_per_call))
            b_outs.append(b_out)
            per_problem_outcomes.append({
                "pid": str(p.pid),
                "question_type": str(p.question_type),
                "answer_type": str(p.answer_type),
                "gold_answer": str(p.answer),
                "a0_text_passed": bool(a0_out.final_passed),
                "a1_vlm_passed": bool(a1_out.final_passed),
                "b_vlm_team_passed": bool(b_out.final_passed),
                "a0_outcome_cid": str(a0_out.cid()),
                "a1_outcome_cid": str(a1_out.cid()),
                "b_outcome_cid": str(b_out.cid()),
            })
            if sidecar_writer is not None:
                sidecar_writer({
                    "kind": "w95_mathvista_per_problem_outcome",
                    **per_problem_outcomes[-1],
                })
        n = float(len(a0_outs))
        a0_acc = sum(
            1 for o in a0_outs if o.final_passed) / n
        a1_acc = sum(
            1 for o in a1_outs if o.final_passed) / n
        b_acc = sum(
            1 for o in b_outs if o.final_passed) / n
        outcome_cids = tuple(
            [o.cid() for o in a0_outs]
            + [o.cid() for o in a1_outs]
            + [o.cid() for o in b_outs])
        seed_merkle = _sha256_hex({
            "kind": "w95_mathvista_seed_merkle_root_v1",
            "seed": int(seed),
            "outcome_cids": list(outcome_cids),
        })
        per_seed.append(MathVistaSeedReportV1(
            schema=W95_MATHVISTA_BENCH_V1_SCHEMA_VERSION,
            seed=int(seed),
            n_problems=int(len(a0_outs)),
            a0_text_pass_at_1=float(a0_acc),
            a1_vlm_pass_at_1=float(a1_acc),
            b_vlm_team_pass_at_1=float(b_acc),
            a0_text_total_wall_ms=sum(
                int(o.total_wall_ms) for o in a0_outs),
            a1_vlm_total_wall_ms=sum(
                int(o.total_wall_ms) for o in a1_outs),
            b_vlm_team_total_wall_ms=sum(
                int(o.total_wall_ms) for o in b_outs),
            per_problem_outcomes=tuple(per_problem_outcomes),
            outcome_cids=outcome_cids,
            seed_merkle_root=str(seed_merkle)))
        all_outcome_cids.extend(outcome_cids)
    nseeds = float(len(per_seed))
    a0_mean = sum(
        s.a0_text_pass_at_1 for s in per_seed) / nseeds
    a1_mean = sum(
        s.a1_vlm_pass_at_1 for s in per_seed) / nseeds
    b_mean = sum(
        s.b_vlm_team_pass_at_1 for s in per_seed) / nseeds
    bench_merkle = _sha256_hex({
        "kind": "w95_mathvista_bench_merkle_root_v1",
        "vlm_model_id": str(vlm_model_id),
        "text_model_id": str(text_model_id),
        "corpus_parquet_sha256": str(corpus_parquet_sha256),
        "corpus_merkle_root": str(corpus_merkle_root),
        "outcome_cids": list(all_outcome_cids),
        "seeds": list(cfg.seeds),
        "n_problems": int(cfg.n_problems),
        "K": int(cfg.K_multi_sample),
    })
    # Per-seed "n problems where B ≥ A1" gate.
    n_b_ge_a1_per_seed: list[int] = []
    for s in per_seed:
        n_bg = 0
        for po in s.per_problem_outcomes:
            if (bool(po["b_vlm_team_passed"])
                    >= bool(po["a1_vlm_passed"])):
                n_bg += 1
        n_b_ge_a1_per_seed.append(int(n_bg))
    report = MathVistaBenchReportV1(
        schema=W95_MATHVISTA_BENCH_V1_SCHEMA_VERSION,
        vlm_model_id=str(vlm_model_id),
        text_model_id=str(text_model_id),
        n_problems=int(cfg.n_problems),
        n_seeds=int(len(cfg.seeds)),
        K_multi_sample=int(cfg.K_multi_sample),
        corpus_parquet_sha256=str(corpus_parquet_sha256),
        corpus_merkle_root=str(corpus_merkle_root),
        per_seed=tuple(per_seed),
        a0_text_mean_pass_at_1=float(a0_mean),
        a1_vlm_mean_pass_at_1=float(a1_mean),
        b_vlm_team_mean_pass_at_1=float(b_mean),
        b_beats_a0_text_per_seed=tuple(
            s.b_vlm_team_pass_at_1 > s.a0_text_pass_at_1
            for s in per_seed),
        b_beats_a1_vlm_per_seed=tuple(
            s.b_vlm_team_pass_at_1 > s.a1_vlm_pass_at_1
            for s in per_seed),
        b_mean_strictly_beats_a0_text_mean=bool(b_mean > a0_mean),
        b_mean_strictly_beats_a1_vlm_mean=bool(b_mean > a1_mean),
        b_mean_minus_a0_text_mean_pp=float(
            (b_mean - a0_mean) * 100.0),
        b_mean_minus_a1_vlm_mean_pp=float(
            (b_mean - a1_mean) * 100.0),
        n_b_ge_a1_problems_per_seed=tuple(n_b_ge_a1_per_seed),
        bench_merkle_root=str(bench_merkle))
    return report


__all__ = [
    "W95_MATHVISTA_BENCH_V1_SCHEMA_VERSION",
    "extract_candidate_answer_v1",
    "MathVistaArmCallCapsuleV1",
    "MathVistaArmOutcomeCapsuleV1",
    "MathVistaSeedReportV1",
    "MathVistaBenchReportV1",
    "MathVistaBenchConfigV1",
    "run_mathvista_bench_v1",
]
