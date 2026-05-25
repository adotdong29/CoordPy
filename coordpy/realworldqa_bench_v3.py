"""W98 — RealWorldQA bench V3 (D2-B2 candidate).

Direct-vision final-turn answerer.  W98 candidate B2.  Mirrors
``coordpy.realworldqa_bench_v1`` (W97 D2-B0) for everything that
is NOT the structural fix.

What changes vs W97 D2-B0
-------------------------
1. **Direct-vision final-turn answerer** — turn 4 (the 5th
   model call) is a VLM call with FULL image access; it
   receives the original image + the W95-B0-style free-text
   extraction + the question + the prior text-solver
   candidates and produces the canonical answer.  Mechanism:
   keep the image alive at the decision boundary on the
   failure cluster where unified-VLM K=5 wins (5 / 5 W97
   unique-A1-rescues).
2. **First-PASS short-circuit preserved on text-solver turns**
   — the final-turn VLM is invoked ONLY if all text-solver
   turns FAIL.  D2-B0's 22 / 30 both-pass and 3 / 30
   unique-B-rescues are mechanistically protected: the bench
   ships the first PASSing text-solver answer; the final-turn
   VLM only runs when the text-solver chain has failed.

Critical distinction from W96-C C1 verifier
-------------------------------------------
W96-C C1's final turn was a BINARY agree/disagree verifier
on a prior candidate.  Empirically the verifier rescue rate
was 0/11 at 11B and 1/7 at 90B (= not load-bearing).

W98 B2's final turn is a COMMITTED ANSWERER with full image
access; the decision surface is the original question, not a
meta-decision about a prior answer.  It is invoked only on the
failure cluster where text-solver turns have all FAILed —
exactly the cluster where A1 K=5 wins (5 / 5 in W97).  The
mechanism gives the failure cluster the same image access A1
has but with the additional structured-extraction context.

Everything else is byte-identical to W97 D2-B0 (same K=5
budget, same VLM model on every arm, same executor, same A0 /
A1 baselines, same content-addressable capsules, same audit-
chain Merkle).

Honest scope (W98 B2)
---------------------

* ``W98-L-REALWORLDQA-BENCH-V3-FINAL-TURN-VLM-CAP`` — V3's
  final turn is a VLM call with the same model family as the
  scene reader.  Same anti-cheat (same VLM on every arm).
* ``W98-L-REALWORLDQA-BENCH-V3-SHORT-CIRCUIT-CAP`` — the final
  VLM turn fires only when all text-solver turns FAIL.  Wins
  short-circuit on the first text-solver PASS.
* ``W98-L-REALWORLDQA-BENCH-V3-NIM-DEPENDENT-CAP`` — V3 drives
  the bench through caller-provided text / VLM clients; the
  driver does not couple to NIM.
* ``W98-L-REALWORLDQA-BENCH-V3-K5-EXACT-CAP`` — every problem
  uses exactly K=5 model calls on the B arm: 1 reader + 3
  text-solver + 1 final-VLM (when short-circuit not hit) OR 1
  reader + N text-solver + (3-N) padding text-solver retries
  on the same prompt (when text-solver PASSes and the final
  VLM call is skipped).  Padding ensures wall budget parity
  with A1 K=5; the chosen answer is the first PASS.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Sequence

from .realworldqa_executor_v1 import (
    RealWorldQAExecutorResultV1,
    W96_REALWORLDQA_EXECUTOR_V1_SCHEMA_VERSION,
    evaluate_realworldqa_answer_v1,
)
from .realworldqa_loader_v1 import (
    RealWorldQAProblemV1,
    W96_REALWORLDQA_LOADER_V1_SCHEMA_VERSION,
)
from .realworldqa_bench_v2 import (
    detect_question_type_v2,
    extract_candidate_answer_v1,
)


W98_REALWORLDQA_BENCH_V3_SCHEMA_VERSION: str = (
    "coordpy.realworldqa_bench_v3.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    if isinstance(payload, (bytes, bytearray)):
        return hashlib.sha256(payload).hexdigest()
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ---------------------------------------------------------------
# Capsules
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RealWorldQAV3ArmCallCapsule:
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
            "kind": "w98_realworldqa_v3_arm_call",
            **self.to_dict()})


@dataclasses.dataclass(frozen=True)
class RealWorldQAV3ArmOutcomeCapsule:
    schema: str
    seed: int
    pid: str
    arm_id: str
    question_type: str
    final_turn_vlm_invoked: bool
    final_turn_vlm_rescued: bool
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
            "question_type": str(self.question_type),
            "final_turn_vlm_invoked": bool(
                self.final_turn_vlm_invoked),
            "final_turn_vlm_rescued": bool(
                self.final_turn_vlm_rescued),
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
            "kind": "w98_realworldqa_v3_arm_outcome",
            **self.to_dict()})


def _executor_result_cid(
        result: RealWorldQAExecutorResultV1) -> str:
    return _sha256_hex({
        "kind": "w98_realworldqa_v3_executor_result",
        **result.to_dict()})


# ---------------------------------------------------------------
# Prompts (V3 = W95-B0 reader/solver + new final-VLM answerer)
# ---------------------------------------------------------------

_A0_SYSTEM = (
    "You are answering a real-world image question.  Read the "
    "question carefully and provide a single final answer.  If "
    "the question is multi-choice, output ONLY the letter (A, "
    "B, C, ...).  If the question expects a number, output ONLY "
    "the number.  If the question expects a short text answer "
    "(a color, a yes/no, a single word), output ONLY that short "
    "answer.  Do not include any prose around the answer.")


_A1_SYSTEM = (
    "You are answering a real-world image question.  Look at "
    "the image and read the question carefully, then provide a "
    "single final answer.  If the question is multi-choice, "
    "output ONLY the letter (A, B, C, ...).  If the question "
    "expects a number, output ONLY the number.  If the "
    "question expects a short text answer (a color, a yes/no, "
    "a single word), output ONLY that short answer.  Do not "
    "include any prose around the answer.")


_B_SCENE_READER_SYSTEM = (
    "You are an expert visual scene reader.  Given a real-world "
    "image and a question about it, extract the structured "
    "facts the solver will need as a short bullet list.  "
    "Include: visible objects (with approximate spatial region: "
    "left / center / right and top / middle / bottom), object "
    "counts, colors and other attributes, spatial relations "
    "between objects, any visible text in the scene (signs, "
    "labels), and any action / activity visible.  Do NOT answer "
    "the question; only extract the structured facts the solver "
    "will need.")


_B_SOLVER_SYSTEM = (
    "You are answering a real-world image question.  You will "
    "be given the question text and a short bullet list of "
    "structured facts extracted from the image.  Use the "
    "structured facts as ground truth (you cannot see the image "
    "yourself) and produce a single final answer.  Output ONLY "
    "the letter (for multi-choice) or the number / short answer "
    "(color, yes/no, single word); no prose.")


_B_FINAL_VLM_ANSWERER_SYSTEM = (
    "You are the FINAL answerer.  You see the original image, "
    "a structured extraction of the scene, the question, and "
    "the prior text-only candidate answers (all of which have "
    "been judged incorrect by an automated checker).  Look at "
    "the image carefully, reconcile against the extraction, "
    "and produce the canonical final answer to the question.  "
    "Output formatting rules:\n"
    "  - If the question is multi-choice (A, B, C, D options), "
    "output ONLY the single letter.\n"
    "  - If the question is yes/no, output ONLY `Yes` or "
    "`No`.  NEVER output a number for a yes/no question.\n"
    "  - If the question asks for a number, output ONLY the "
    "number.\n"
    "  - Otherwise output ONLY a short word / phrase.\n"
    "  Do not include prose around the answer.")


def _a0_prompt(p: RealWorldQAProblemV1) -> str:
    return f"{_A0_SYSTEM}\n\nQuestion: {p.question}\n\nFinal answer:"


def _a1_prompt(p: RealWorldQAProblemV1) -> str:
    return f"{_A1_SYSTEM}\n\nQuestion: {p.question}\n\nFinal answer:"


def _b_reader_prompt(p: RealWorldQAProblemV1) -> str:
    return (
        f"{_B_SCENE_READER_SYSTEM}\n\n"
        f"Question:\n{p.question}\n\n"
        "Extract the structured facts the solver will need.  "
        "Bullet list only.  No answer.")


def _b_solver_initial_prompt(
        p: RealWorldQAProblemV1, extraction: str) -> str:
    return (
        f"{_B_SOLVER_SYSTEM}\n\n"
        f"Question:\n{p.question}\n\n"
        "Structured facts extracted from the image (use as "
        f"ground truth):\n{extraction}\n\nFinal answer:")


def _b_solver_reflexion_prompt(
        p: RealWorldQAProblemV1, extraction: str,
        history: Sequence[
            tuple[str, RealWorldQAExecutorResultV1]],
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
        f"You are on attempt {attempt_idx + 1} out of 4.\n\n"
        f"Question:\n{p.question}\n\n"
        "Structured facts extracted from the image (use as "
        f"ground truth):\n{extraction}\n\n"
        f"{chr(10).join(chunks)}\n\n"
        "Diagnose the failing attempt(s) and produce a NEW "
        "final answer.  Do not repeat a previous attempt "
        "verbatim.  Output ONLY the letter (for multi-choice) "
        "or the number / short answer; no prose.\n\n"
        "Final answer:")


def _b_final_vlm_prompt(
        p: RealWorldQAProblemV1, extraction: str,
        history: Sequence[
            tuple[str, RealWorldQAExecutorResultV1]],
        question_type: str) -> str:
    chunks: list[str] = []
    for i, (cand, exe) in enumerate(history):
        chunks.append(
            f"  attempt {i + 1}: {cand}  ({exe.matched_rule})")
    return (
        f"{_B_FINAL_VLM_ANSWERER_SYSTEM}\n\n"
        f"Question (type={question_type}):\n{p.question}\n\n"
        "Structured extraction from a prior reader pass:\n"
        f"{extraction}\n\n"
        "Prior text-only candidate answers (all judged "
        "incorrect):\n"
        f"{chr(10).join(chunks)}\n\n"
        "Now look at the original image and produce the "
        "canonical final answer.\n\n"
        "Final answer:")


# ---------------------------------------------------------------
# Gen fn signatures
# ---------------------------------------------------------------

_TextGenFn = Callable[[str, int, float], tuple[str, int]]
_VlmGenFn = Callable[
    [str, "bytes | None", int, float], tuple[str, int]]


# ---------------------------------------------------------------
# Per-arm runners (A0, A1 = byte-identical to V1; B = NEW)
# ---------------------------------------------------------------

def _run_a0_text(
        *, seed: int, p: RealWorldQAProblemV1,
        text_gen: _TextGenFn, max_tokens: int,
) -> tuple[RealWorldQAV3ArmOutcomeCapsule,
           RealWorldQAExecutorResultV1, str]:
    qt = detect_question_type_v2(p.question)
    prompt = _a0_prompt(p)
    text, wall = text_gen(prompt, max_tokens, 0.0)
    candidate = extract_candidate_answer_v1(response_text=text)
    exe = evaluate_realworldqa_answer_v1(
        prediction=candidate, problem=p)
    call = RealWorldQAV3ArmCallCapsule(
        schema=W98_REALWORLDQA_BENCH_V3_SCHEMA_VERSION,
        seed=int(seed), pid=str(p.pid),
        arm_id="A0_text", role="text_solver", call_idx=0,
        temperature=0.0,
        prompt_cid=hashlib.sha256(
            prompt.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            text.encode("utf-8")).hexdigest(),
        wall_ms=int(wall))
    out = RealWorldQAV3ArmOutcomeCapsule(
        schema=W98_REALWORLDQA_BENCH_V3_SCHEMA_VERSION,
        seed=int(seed), pid=str(p.pid),
        arm_id="A0_text",
        question_type=qt,
        final_turn_vlm_invoked=False,
        final_turn_vlm_rescued=False,
        final_passed=bool(exe.passed),
        final_prediction_cid=hashlib.sha256(
            candidate.encode("utf-8")).hexdigest(),
        final_executor_rule=str(exe.matched_rule),
        n_model_calls=1, total_wall_ms=int(wall),
        call_capsule_cids=(call.cid(),),
        executor_result_cid=_executor_result_cid(exe))
    return out, exe, qt


def _run_a1_vlm(
        *, seed: int, p: RealWorldQAProblemV1, K: int,
        temperature: float, vlm_gen: _VlmGenFn,
        max_tokens: int,
) -> tuple[RealWorldQAV3ArmOutcomeCapsule,
           list[RealWorldQAExecutorResultV1], str]:
    qt = detect_question_type_v2(p.question)
    prompt = _a1_prompt(p)
    calls: list[RealWorldQAV3ArmCallCapsule] = []
    exes: list[RealWorldQAExecutorResultV1] = []
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
        exe = evaluate_realworldqa_answer_v1(
            prediction=candidate, problem=p)
        calls.append(RealWorldQAV3ArmCallCapsule(
            schema=W98_REALWORLDQA_BENCH_V3_SCHEMA_VERSION,
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
    out = RealWorldQAV3ArmOutcomeCapsule(
        schema=W98_REALWORLDQA_BENCH_V3_SCHEMA_VERSION,
        seed=int(seed), pid=str(p.pid),
        arm_id="A1_vlm",
        question_type=qt,
        final_turn_vlm_invoked=False,
        final_turn_vlm_rescued=False,
        final_passed=bool(chosen_passed),
        final_prediction_cid=hashlib.sha256(
            chosen_cand.encode("utf-8")).hexdigest(),
        final_executor_rule=str(chosen_rule),
        n_model_calls=int(K),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cid=_executor_result_cid(
            exes[chosen_idx if chosen_passed else 0]))
    return out, exes, qt


def _run_b_direct_vision_final(
        *, seed: int, p: RealWorldQAProblemV1, K: int,
        temperature: float, vlm_gen: _VlmGenFn,
        text_gen: _TextGenFn, max_tokens: int,
) -> tuple[RealWorldQAV3ArmOutcomeCapsule,
           list[RealWorldQAExecutorResultV1], str]:
    """W98 B2 (D2-B2): 1 scene reader (T=0.0, VLM) + 3 text
    solver/reflexion (T=temperature) + 1 FINAL VLM answerer
    (T=0.0; sees image) when text-solver chain FAILs; total
    K=5 byte-exact.  Short-circuit on first text-solver PASS:
    pads with additional text-solver retries on the same
    prompt to keep wall budget parity with A1 K=5.
    """
    if int(K) < 4:
        raise ValueError(
            "W98 B2 requires K >= 4 (1 reader + 2 solver + 1 "
            f"final = 4 minimum); got K={K}")
    qt = detect_question_type_v2(p.question)
    calls: list[RealWorldQAV3ArmCallCapsule] = []
    text_exes: list[RealWorldQAExecutorResultV1] = []
    candidates: list[str] = []
    history: list[tuple[str, RealWorldQAExecutorResultV1]] = []
    total = 0

    # Stage 1 — VLM scene reader (1 call, T=0.0).
    reader_prompt = _b_reader_prompt(p)
    reader_text, reader_wall = vlm_gen(
        reader_prompt, p.image_bytes, max_tokens, 0.0)
    calls.append(RealWorldQAV3ArmCallCapsule(
        schema=W98_REALWORLDQA_BENCH_V3_SCHEMA_VERSION,
        seed=int(seed), pid=str(p.pid),
        arm_id="B_direct_vision_final", role="scene_reader",
        call_idx=0, temperature=0.0,
        prompt_cid=hashlib.sha256(
            (reader_prompt + "|img:" + p.image_sha256).encode(
                "utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            reader_text.encode("utf-8")).hexdigest(),
        wall_ms=int(reader_wall)))
    total += int(reader_wall)
    extraction = reader_text.strip() or "(no extraction)"

    # Stages 2..(K-1) — text solver + reflexion (K-2 turns).
    # Reserve the LAST call for the final-VLM answerer.
    text_solver_budget = int(K) - 2  # = 3 when K=5
    for k in range(int(text_solver_budget)):
        if k == 0:
            solver_prompt = _b_solver_initial_prompt(
                p, extraction)
            solver_role = "text_solver_initial"
        else:
            solver_prompt = _b_solver_reflexion_prompt(
                p, extraction, tuple(history),
                attempt_idx=int(k))
            solver_role = "text_solver_reflexion"
        solver_text, solver_wall = text_gen(
            solver_prompt, max_tokens, float(temperature))
        candidate = extract_candidate_answer_v1(
            response_text=solver_text)
        exe = evaluate_realworldqa_answer_v1(
            prediction=candidate, problem=p)
        calls.append(RealWorldQAV3ArmCallCapsule(
            schema=W98_REALWORLDQA_BENCH_V3_SCHEMA_VERSION,
            seed=int(seed), pid=str(p.pid),
            arm_id="B_direct_vision_final", role=solver_role,
            call_idx=int(k) + 1,
            temperature=float(temperature),
            prompt_cid=hashlib.sha256(
                solver_prompt.encode("utf-8")).hexdigest(),
            response_cid=hashlib.sha256(
                solver_text.encode("utf-8")).hexdigest(),
            wall_ms=int(solver_wall)))
        text_exes.append(exe)
        candidates.append(candidate)
        history.append((candidate, exe))
        total += int(solver_wall)

    # Short-circuit detection: did any text-solver turn PASS?
    text_solver_pass_idx = -1
    for i, exe in enumerate(text_exes):
        if exe.passed:
            text_solver_pass_idx = int(i)
            break
    final_vlm_invoked = False
    final_vlm_rescued = False
    final_exe: RealWorldQAExecutorResultV1
    final_cand: str
    final_rule: str

    if text_solver_pass_idx >= 0:
        # SHORT-CIRCUIT: text-solver passed; skip final VLM
        # answerer.  Pad with text-solver retries on the same
        # reflexion prompt to keep wall + call budget parity.
        final_cand = candidates[text_solver_pass_idx]
        final_exe = text_exes[text_solver_pass_idx]
        final_rule = str(final_exe.matched_rule)
        # Pad to exactly K total calls.
        n_pad = int(K) - len(calls)
        for j in range(int(n_pad)):
            pad_prompt = _b_solver_reflexion_prompt(
                p, extraction, tuple(history),
                attempt_idx=int(text_solver_budget) + int(j))
            pad_text, pad_wall = text_gen(
                pad_prompt, max_tokens, float(temperature))
            calls.append(RealWorldQAV3ArmCallCapsule(
                schema=W98_REALWORLDQA_BENCH_V3_SCHEMA_VERSION,
                seed=int(seed), pid=str(p.pid),
                arm_id="B_direct_vision_final",
                role="text_solver_short_circuit_pad",
                call_idx=int(text_solver_budget) + 1
                + int(j),
                temperature=float(temperature),
                prompt_cid=hashlib.sha256(
                    pad_prompt.encode("utf-8")).hexdigest(),
                response_cid=hashlib.sha256(
                    pad_text.encode("utf-8")).hexdigest(),
                wall_ms=int(pad_wall)))
            total += int(pad_wall)
    else:
        # All text-solver turns FAILed; invoke final VLM
        # answerer with full image access.
        final_vlm_invoked = True
        final_vlm_prompt = _b_final_vlm_prompt(
            p, extraction, tuple(history), qt)
        final_text, final_wall = vlm_gen(
            final_vlm_prompt, p.image_bytes,
            max_tokens, 0.0)
        final_cand = extract_candidate_answer_v1(
            response_text=final_text)
        final_exe = evaluate_realworldqa_answer_v1(
            prediction=final_cand, problem=p)
        final_rule = str(final_exe.matched_rule)
        final_vlm_rescued = bool(final_exe.passed)
        calls.append(RealWorldQAV3ArmCallCapsule(
            schema=W98_REALWORLDQA_BENCH_V3_SCHEMA_VERSION,
            seed=int(seed), pid=str(p.pid),
            arm_id="B_direct_vision_final",
            role="final_vlm_answerer",
            call_idx=int(text_solver_budget) + 1,
            temperature=0.0,
            prompt_cid=hashlib.sha256(
                (final_vlm_prompt + "|img:"
                 + p.image_sha256).encode(
                     "utf-8")).hexdigest(),
            response_cid=hashlib.sha256(
                final_text.encode("utf-8")).hexdigest(),
            wall_ms=int(final_wall)))
        total += int(final_wall)

    out = RealWorldQAV3ArmOutcomeCapsule(
        schema=W98_REALWORLDQA_BENCH_V3_SCHEMA_VERSION,
        seed=int(seed), pid=str(p.pid),
        arm_id="B_direct_vision_final",
        question_type=qt,
        final_turn_vlm_invoked=bool(final_vlm_invoked),
        final_turn_vlm_rescued=bool(final_vlm_rescued),
        final_passed=bool(final_exe.passed),
        final_prediction_cid=hashlib.sha256(
            final_cand.encode("utf-8")).hexdigest(),
        final_executor_rule=str(final_rule),
        n_model_calls=int(K),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cid=_executor_result_cid(final_exe))
    return out, text_exes + [final_exe], qt


# ---------------------------------------------------------------
# Report dataclasses (mirror V2 layout)
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RealWorldQAV3SeedReport:
    schema: str
    seed: int
    n_problems: int
    a0_text_pass_at_1: float
    a1_vlm_pass_at_1: float
    b_direct_vision_final_pass_at_1: float
    a0_text_total_wall_ms: int
    a1_vlm_total_wall_ms: int
    b_direct_vision_final_total_wall_ms: int
    final_vlm_invocation_count: int
    final_vlm_rescue_count: int
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
            "b_direct_vision_final_pass_at_1": float(round(
                self.b_direct_vision_final_pass_at_1, 6)),
            "a0_text_total_wall_ms": int(
                self.a0_text_total_wall_ms),
            "a1_vlm_total_wall_ms": int(
                self.a1_vlm_total_wall_ms),
            "b_direct_vision_final_total_wall_ms": int(
                self.b_direct_vision_final_total_wall_ms),
            "final_vlm_invocation_count": int(
                self.final_vlm_invocation_count),
            "final_vlm_rescue_count": int(
                self.final_vlm_rescue_count),
            "per_problem_outcomes": [
                dict(po) for po in self.per_problem_outcomes],
            "outcome_cids": list(self.outcome_cids),
            "seed_merkle_root": str(self.seed_merkle_root),
        }


@dataclasses.dataclass(frozen=True)
class RealWorldQAV3BenchReport:
    schema: str
    vlm_model_id: str
    text_model_id: str
    n_problems: int
    n_seeds: int
    K_multi_sample: int
    corpus_parquet_shard_sha256: tuple[str, ...]
    corpus_merkle_root: str
    per_seed: tuple[RealWorldQAV3SeedReport, ...]
    a0_text_mean_pass_at_1: float
    a1_vlm_mean_pass_at_1: float
    b_direct_vision_final_mean_pass_at_1: float
    b_beats_a0_text_per_seed: tuple[bool, ...]
    b_beats_a1_vlm_per_seed: tuple[bool, ...]
    b_mean_strictly_beats_a0_text_mean: bool
    b_mean_strictly_beats_a1_vlm_mean: bool
    b_mean_minus_a0_text_mean_pp: float
    b_mean_minus_a1_vlm_mean_pp: float
    n_b_ge_a1_problems_per_seed: tuple[int, ...]
    final_vlm_invocation_count_total: int
    final_vlm_rescue_count_total: int
    question_type_distribution: dict[str, int]
    bench_merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "vlm_model_id": str(self.vlm_model_id),
            "text_model_id": str(self.text_model_id),
            "n_problems": int(self.n_problems),
            "n_seeds": int(self.n_seeds),
            "K_multi_sample": int(self.K_multi_sample),
            "corpus_parquet_shard_sha256": list(
                self.corpus_parquet_shard_sha256),
            "corpus_merkle_root": str(self.corpus_merkle_root),
            "per_seed": [s.to_dict() for s in self.per_seed],
            "a0_text_mean_pass_at_1": float(round(
                self.a0_text_mean_pass_at_1, 6)),
            "a1_vlm_mean_pass_at_1": float(round(
                self.a1_vlm_mean_pass_at_1, 6)),
            "b_direct_vision_final_mean_pass_at_1": float(
                round(self.b_direct_vision_final_mean_pass_at_1,
                      6)),
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
            "final_vlm_invocation_count_total": int(
                self.final_vlm_invocation_count_total),
            "final_vlm_rescue_count_total": int(
                self.final_vlm_rescue_count_total),
            "question_type_distribution": dict(
                self.question_type_distribution),
            "bench_merkle_root": str(self.bench_merkle_root),
        }


# ---------------------------------------------------------------
# Config + driver
# ---------------------------------------------------------------

@dataclasses.dataclass
class RealWorldQAV3BenchConfig:
    schema: str = W98_REALWORLDQA_BENCH_V3_SCHEMA_VERSION
    n_problems: int = 30
    K_multi_sample: int = 5
    seeds: tuple[int, ...] = (96_504_002,)
    sampling_temperature: float = 0.7
    max_tokens_per_call: int = 384


def run_realworldqa_bench_v3(
        *,
        text_gen: _TextGenFn,
        vlm_gen: _VlmGenFn,
        vlm_model_id: str,
        text_model_id: str,
        corpus: Sequence[RealWorldQAProblemV1],
        corpus_parquet_shard_sha256: tuple[str, ...],
        corpus_merkle_root: str,
        config: RealWorldQAV3BenchConfig | None = None,
        on_problem_start: (
            Callable[[int, int, str], None] | None) = None,
        sidecar_writer: (
            Callable[[dict[str, Any]], None] | None) = None,
) -> RealWorldQAV3BenchReport:
    from .realworldqa_loader_v1 import (
        select_realworldqa_subset_v1)

    cfg = config or RealWorldQAV3BenchConfig()
    per_seed: list[RealWorldQAV3SeedReport] = []
    all_outcome_cids: list[str] = []
    qt_counter: dict[str, int] = {}
    inv_total = 0
    res_total = 0
    for seed in cfg.seeds:
        subset = select_realworldqa_subset_v1(
            seed=int(seed),
            n_problems=int(cfg.n_problems),
            corpus=tuple(corpus))
        if len(subset) < int(cfg.n_problems):
            raise RuntimeError(
                "corpus has only "
                f"{len(subset)} problems for seed {seed}; "
                f"need {cfg.n_problems}")
        a0_outs: list[RealWorldQAV3ArmOutcomeCapsule] = []
        a1_outs: list[RealWorldQAV3ArmOutcomeCapsule] = []
        b_outs: list[RealWorldQAV3ArmOutcomeCapsule] = []
        per_problem_outcomes: list[dict[str, Any]] = []
        seed_inv = 0
        seed_res = 0
        for p_idx, p in enumerate(subset):
            if on_problem_start is not None:
                on_problem_start(
                    int(seed), int(p_idx), str(p.pid))
            a0_out, _, qt = _run_a0_text(
                seed=int(seed), p=p,
                text_gen=text_gen,
                max_tokens=int(cfg.max_tokens_per_call))
            a0_outs.append(a0_out)
            a1_out, _, _ = _run_a1_vlm(
                seed=int(seed), p=p,
                K=int(cfg.K_multi_sample),
                temperature=float(cfg.sampling_temperature),
                vlm_gen=vlm_gen,
                max_tokens=int(cfg.max_tokens_per_call))
            a1_outs.append(a1_out)
            b_out, _, _ = _run_b_direct_vision_final(
                seed=int(seed), p=p,
                K=int(cfg.K_multi_sample),
                temperature=float(cfg.sampling_temperature),
                vlm_gen=vlm_gen,
                text_gen=text_gen,
                max_tokens=int(cfg.max_tokens_per_call))
            b_outs.append(b_out)
            if b_out.final_turn_vlm_invoked:
                seed_inv += 1
            if b_out.final_turn_vlm_rescued:
                seed_res += 1
            qt_counter[qt] = qt_counter.get(qt, 0) + 1
            per_problem_outcomes.append({
                "pid": str(p.pid),
                "question": str(p.question),
                "gold_answer": str(p.answer),
                "question_type": str(qt),
                "a0_text_passed": bool(a0_out.final_passed),
                "a1_vlm_passed": bool(a1_out.final_passed),
                "b_direct_vision_final_passed": bool(
                    b_out.final_passed),
                "b_final_vlm_invoked": bool(
                    b_out.final_turn_vlm_invoked),
                "b_final_vlm_rescued": bool(
                    b_out.final_turn_vlm_rescued),
                "a0_outcome_cid": str(a0_out.cid()),
                "a1_outcome_cid": str(a1_out.cid()),
                "b_outcome_cid": str(b_out.cid()),
            })
            if sidecar_writer is not None:
                sidecar_writer({
                    "kind": (
                        "w98_realworldqa_v3_per_problem_outcome"),
                    **per_problem_outcomes[-1],
                })
        inv_total += seed_inv
        res_total += seed_res
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
            "kind": "w98_realworldqa_v3_seed_merkle_root",
            "seed": int(seed),
            "outcome_cids": list(outcome_cids),
        })
        per_seed.append(RealWorldQAV3SeedReport(
            schema=W98_REALWORLDQA_BENCH_V3_SCHEMA_VERSION,
            seed=int(seed),
            n_problems=int(len(a0_outs)),
            a0_text_pass_at_1=float(a0_acc),
            a1_vlm_pass_at_1=float(a1_acc),
            b_direct_vision_final_pass_at_1=float(b_acc),
            a0_text_total_wall_ms=sum(
                int(o.total_wall_ms) for o in a0_outs),
            a1_vlm_total_wall_ms=sum(
                int(o.total_wall_ms) for o in a1_outs),
            b_direct_vision_final_total_wall_ms=sum(
                int(o.total_wall_ms) for o in b_outs),
            final_vlm_invocation_count=int(seed_inv),
            final_vlm_rescue_count=int(seed_res),
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
        s.b_direct_vision_final_pass_at_1
        for s in per_seed) / nseeds
    bench_merkle = _sha256_hex({
        "kind": "w98_realworldqa_v3_bench_merkle_root",
        "vlm_model_id": str(vlm_model_id),
        "text_model_id": str(text_model_id),
        "corpus_parquet_shard_sha256": list(
            corpus_parquet_shard_sha256),
        "corpus_merkle_root": str(corpus_merkle_root),
        "outcome_cids": list(all_outcome_cids),
        "seeds": list(cfg.seeds),
        "n_problems": int(cfg.n_problems),
        "K": int(cfg.K_multi_sample),
    })
    n_b_ge_a1_per_seed: list[int] = []
    for s in per_seed:
        n_bg = 0
        for po in s.per_problem_outcomes:
            if (bool(po["b_direct_vision_final_passed"])
                    >= bool(po["a1_vlm_passed"])):
                n_bg += 1
        n_b_ge_a1_per_seed.append(int(n_bg))
    report = RealWorldQAV3BenchReport(
        schema=W98_REALWORLDQA_BENCH_V3_SCHEMA_VERSION,
        vlm_model_id=str(vlm_model_id),
        text_model_id=str(text_model_id),
        n_problems=int(cfg.n_problems),
        n_seeds=int(len(cfg.seeds)),
        K_multi_sample=int(cfg.K_multi_sample),
        corpus_parquet_shard_sha256=tuple(
            corpus_parquet_shard_sha256),
        corpus_merkle_root=str(corpus_merkle_root),
        per_seed=tuple(per_seed),
        a0_text_mean_pass_at_1=float(a0_mean),
        a1_vlm_mean_pass_at_1=float(a1_mean),
        b_direct_vision_final_mean_pass_at_1=float(b_mean),
        b_beats_a0_text_per_seed=tuple(
            s.b_direct_vision_final_pass_at_1
            > s.a0_text_pass_at_1 for s in per_seed),
        b_beats_a1_vlm_per_seed=tuple(
            s.b_direct_vision_final_pass_at_1
            > s.a1_vlm_pass_at_1 for s in per_seed),
        b_mean_strictly_beats_a0_text_mean=bool(b_mean > a0_mean),
        b_mean_strictly_beats_a1_vlm_mean=bool(b_mean > a1_mean),
        b_mean_minus_a0_text_mean_pp=float(
            (b_mean - a0_mean) * 100.0),
        b_mean_minus_a1_vlm_mean_pp=float(
            (b_mean - a1_mean) * 100.0),
        n_b_ge_a1_problems_per_seed=tuple(n_b_ge_a1_per_seed),
        final_vlm_invocation_count_total=int(inv_total),
        final_vlm_rescue_count_total=int(res_total),
        question_type_distribution=dict(qt_counter),
        bench_merkle_root=str(bench_merkle))
    return report


__all__ = [
    "W98_REALWORLDQA_BENCH_V3_SCHEMA_VERSION",
    "extract_candidate_answer_v1",
    "RealWorldQAV3ArmCallCapsule",
    "RealWorldQAV3ArmOutcomeCapsule",
    "RealWorldQAV3SeedReport",
    "RealWorldQAV3BenchReport",
    "RealWorldQAV3BenchConfig",
    "run_realworldqa_bench_v3",
]
