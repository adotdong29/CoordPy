"""W99 — RealWorldQA bench V4 (D2-B4 candidate).

Typed scene-graph extraction + question-typed solver, WITH the
``direct_answer_hint`` field REMOVED from the schema and the
typed solver no longer anchored to a reader-supplied hint.  W99
candidate B4.

What changes vs W98 B1 (``realworldqa_bench_v2``)
-------------------------------------------------
W98 B1 cheap pilot at 11B FAILed Phase 2 by **B − A1 = −6.67
pp**.  Per-problem failure-cluster mining diagnosed the proximate
cause: the typed solver became too confident in the reader's
``direct_answer_hint`` (often wrong on multi-choice) and stopped
the K=4 reflexion-cycling that drove D2-B0's multi-choice wins.
Net 4 yes/no rescues − 5 multi-choice regressions = −1 problem.

B4's structural fix is **minimal**: strip the
``direct_answer_hint`` field from the typed schema and from the
solver prompts.  Everything else is byte-identical to B1:

* the typed schema still has ``state``, ``orientation``,
  ``depth``, and ``text_in_object`` primitives (the 5/5 yes/no
  recovery surface);
* the question-typed solver prompts still partition the answer
  space by ``yes_no | multi_choice_letter | numeric |
  short_text``;
* first-PASS short-circuit on text-solver turns is preserved.

Hypothesis (locked BEFORE any NIM call)
---------------------------------------
With the hint removed, B4 keeps the W98 B1 yes/no recovery
(4/5 of W97 unique-A1-rescues) AND recovers the 5 W98 B1 multi-
choice regressions by restoring the solver's K=4 reflexion-
cycling discipline that D2-B0 had.  Subjective probability of
clearing the +5 pp Phase 2 bar on the 96_504_002 slice: ~ 35-50
% (higher than B1 because the regression mechanism is removed
without giving up the load-bearing schema fix).

What B4 is categorically NOT
----------------------------
* B4 is NOT bounded-window / compaction / prose-summary — same
  rule as B1.  The schema is a content-addressed structured
  representation, not a generic memory trick.
* B4 is NOT a verifier — W96-C C1 was empirically refuted.  B4
  has no final-VLM and no agree/disagree mechanism.
* B4 is NOT a router/switch — B5 (``realworldqa_bench_v5``) is
  the explicit switch baseline.  B4 uses one canonical solver
  prompt per question type, not a routing decision.

Anti-cheat (carry-forward from W88-W98)
---------------------------------------
* Same VLM model on every arm (A0 / A1 / B-reader / B-solver).
* K=5 byte-exact: 1 reader + 4 typed solver turns.
* Executor = ``evaluate_realworldqa_answer_v1`` everywhere.
* No LLM judge.  No selective retries.  Same content-addressed
  capsules; per-call sidecars; per-seed + bench Merkle roots.

Honest scope (W99 B4)
---------------------

* ``W99-L-REALWORLDQA-BENCH-V4-HINT-REMOVAL-CAP`` — V4 strips
  the ``direct_answer_hint`` field.  It does NOT alter the
  schema's other primitives, and does NOT add new ones.
* ``W99-L-REALWORLDQA-BENCH-V4-NIM-DEPENDENT-CAP`` — V4 drives
  through caller-provided text / VLM clients.
* ``W99-L-REALWORLDQA-BENCH-V4-QUESTION-TYPE-PARSER-CAP`` —
  V4 reuses ``detect_question_type_v2`` verbatim (deterministic
  rule-based; no oracle).
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
    QUESTION_TYPE_MULTI_CHOICE_LETTER,
    QUESTION_TYPE_NUMERIC,
    QUESTION_TYPE_SHORT_TEXT,
    QUESTION_TYPE_YES_NO,
    detect_question_type_v2,
    extract_candidate_answer_v1,
)


W99_REALWORLDQA_BENCH_V4_SCHEMA_VERSION: str = (
    "coordpy.realworldqa_bench_v4.v1")


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
class RealWorldQAV4ArmCallCapsule:
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
            "kind": "w99_realworldqa_v4_arm_call",
            **self.to_dict()})


@dataclasses.dataclass(frozen=True)
class RealWorldQAV4ArmOutcomeCapsule:
    schema: str
    seed: int
    pid: str
    arm_id: str
    question_type: str
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
            "kind": "w99_realworldqa_v4_arm_outcome",
            **self.to_dict()})


def _executor_result_cid(
        result: RealWorldQAExecutorResultV1) -> str:
    return _sha256_hex({
        "kind": "w99_realworldqa_v4_executor_result",
        **result.to_dict()})


# ---------------------------------------------------------------
# Prompts (typed; W99 B4 = B1 minus direct_answer_hint)
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


# Notice: the schema below is byte-identical to B1's EXCEPT:
#  * the `direct_answer_hint` field is REMOVED;
#  * the solver prompts (below) no longer reference any hint.
_B_TYPED_SCENE_READER_SYSTEM = (
    "You are an expert visual scene reader.  Given a real-world "
    "image and a question about it, extract a STRUCTURED scene "
    "graph that the downstream text solver will use to answer "
    "the question.  The solver cannot see the image, so the "
    "scene graph MUST contain every fact needed.\n\n"
    "Output a JSON object with EXACTLY these top-level keys "
    "(use the literal string \"unknown\" for fields you cannot "
    "extract from the image):\n\n"
    "  scene_summary: a single sentence describing the scene.\n"
    "  objects: a list of objects, each with fields:\n"
    "    label   — what the object is (car, traffic_light, "
    "stop_sign, person, ...).\n"
    "    color   — primary color word.\n"
    "    x_region — left | center | right | unknown.\n"
    "    y_region — top | middle | bottom | unknown.\n"
    "    depth   — near | mid | far | unknown.\n"
    "    orientation — facing_left | facing_right | "
    "facing_forward | facing_backward | facing_down | "
    "facing_up | unknown.\n"
    "    state   — on | off | green | red | yellow | moving | "
    "stationary | open | closed | unknown.\n"
    "    text_in_object — any text written on this object "
    "(e.g. STOP on a stop_sign); empty string if none.\n"
    "  counts_by_label: a JSON object mapping label -> integer "
    "(e.g. {\"car\": 2, \"stop_sign\": 1, "
    "\"traffic_light\": 1}).\n"
    "  spatial_relations: a list of relations, each with "
    "fields a, b, relation (left_of | right_of | above | "
    "below | in_front_of | behind | inside | unknown), "
    "near (true | false).\n"
    "  text_in_scene: a list of {location, text} for any "
    "readable text in the scene (signs, labels, etc.).\n"
    "  uncertain: a list of feature names you could not "
    "extract reliably.\n\n"
    "Do NOT include any direct answer hint, any "
    "\"direct_answer_hint\" field, or any guess at the "
    "question's answer.  Extract structured facts only.\n\n"
    "Output ONLY the JSON object, no prose, no markdown "
    "fences.  Do not answer the question outside the JSON.")


_B_TYPED_SOLVER_SYSTEM_TEMPLATE = (
    "You are answering a real-world image question.  You will "
    "be given the question and a STRUCTURED JSON scene graph "
    "extracted from the image.  You CANNOT see the image; you "
    "must reason from the scene graph alone.  Treat the scene "
    "graph as ground truth.\n\n"
    "QUESTION TYPE: {qt_label}\n\n"
    "OUTPUT FORMAT (STRICT): {qt_format}\n\n"
    "{qt_advice}")


_QUESTION_TYPE_PROMPT_BLOCKS: dict[str, tuple[str, str, str]] = {
    QUESTION_TYPE_YES_NO: (
        "yes/no question",
        "Output ONLY the single word `Yes` or `No`.  NEVER "
        "output a number.  NEVER output any other word.",
        "Inspect the scene graph for the relevant state, "
        "orientation, or count.  Derive the yes/no answer "
        "directly from the primitives.  Examples: \"Is the "
        "light green?\" with `traffic_light.state = red` -> "
        "`No`.  \"Are there any stop signs?\" with "
        "`counts_by_label.stop_sign >= 1` -> `Yes`.  Do not "
        "guess; reason from the schema."),
    QUESTION_TYPE_MULTI_CHOICE_LETTER: (
        "multi-choice (letter)",
        "Output ONLY the single uppercase letter (A, B, C, "
        "or D) of the correct option.  NEVER output the "
        "option text; NEVER output prose.",
        "Read each option carefully.  Match the option text "
        "against the scene graph's primitives.  Examples: "
        "\"Which direction is the vehicle traveling?\" with "
        "`objects[i].orientation = facing_left` and option B "
        "= Left -> `B`.  If the scene graph does not directly "
        "resolve the choice, use elimination across options."),
    QUESTION_TYPE_NUMERIC: (
        "numeric question",
        "Output ONLY the number (an integer or decimal).  "
        "NEVER output prose.",
        "Use `counts_by_label` if the question asks how many "
        "of a category.  Use object size / distance fields "
        "where applicable.  Reason from the schema; do not "
        "guess."),
    QUESTION_TYPE_SHORT_TEXT: (
        "short-text question",
        "Output ONLY the short answer (a single word or "
        "short phrase).  NEVER output prose around the "
        "answer.",
        "Match the question to the scene graph's color / "
        "state / text_in_object / text_in_scene fields as "
        "appropriate."),
}


def _b_typed_solver_system(qt: str) -> str:
    label, fmt, advice = _QUESTION_TYPE_PROMPT_BLOCKS[qt]
    return _B_TYPED_SOLVER_SYSTEM_TEMPLATE.format(
        qt_label=label, qt_format=fmt, qt_advice=advice)


def _a0_prompt(p: RealWorldQAProblemV1) -> str:
    return f"{_A0_SYSTEM}\n\nQuestion: {p.question}\n\nFinal answer:"


def _a1_prompt(p: RealWorldQAProblemV1) -> str:
    return f"{_A1_SYSTEM}\n\nQuestion: {p.question}\n\nFinal answer:"


def _b_typed_reader_prompt(p: RealWorldQAProblemV1) -> str:
    return (
        f"{_B_TYPED_SCENE_READER_SYSTEM}\n\n"
        f"Question:\n{p.question}\n\n"
        "Output the JSON object now.")


def _b_typed_solver_initial_prompt(
        p: RealWorldQAProblemV1, scene_graph_text: str,
        qt: str) -> str:
    return (
        f"{_b_typed_solver_system(qt)}\n\n"
        f"Question:\n{p.question}\n\n"
        f"Scene graph (JSON):\n{scene_graph_text}\n\n"
        "Final answer:")


def _b_typed_solver_reflexion_prompt(
        p: RealWorldQAProblemV1, scene_graph_text: str, qt: str,
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
                "reason from the question + scene graph)")
        chunks.append(
            f"--- Attempt {i + 1} — {verdict} ---\n"
            f"Candidate answer: {cand}{diag}")
    return (
        f"{_b_typed_solver_system(qt)}\n\n"
        f"You are on attempt {attempt_idx + 1} out of 5.\n\n"
        f"Question:\n{p.question}\n\n"
        f"Scene graph (JSON):\n{scene_graph_text}\n\n"
        f"{chr(10).join(chunks)}\n\n"
        "Diagnose the failing attempt(s) and produce a NEW "
        "final answer.  Do not repeat a previous attempt "
        "verbatim.  Follow the OUTPUT FORMAT rules above.\n\n"
        "Final answer:")


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
        *, seed: int, p: RealWorldQAProblemV1,
        text_gen: _TextGenFn, max_tokens: int,
) -> tuple[RealWorldQAV4ArmOutcomeCapsule,
           RealWorldQAExecutorResultV1, str]:
    qt = detect_question_type_v2(p.question)
    prompt = _a0_prompt(p)
    text, wall = text_gen(prompt, max_tokens, 0.0)
    candidate = extract_candidate_answer_v1(response_text=text)
    exe = evaluate_realworldqa_answer_v1(
        prediction=candidate, problem=p)
    call = RealWorldQAV4ArmCallCapsule(
        schema=W99_REALWORLDQA_BENCH_V4_SCHEMA_VERSION,
        seed=int(seed), pid=str(p.pid),
        arm_id="A0_text", role="text_solver", call_idx=0,
        temperature=0.0,
        prompt_cid=hashlib.sha256(
            prompt.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            text.encode("utf-8")).hexdigest(),
        wall_ms=int(wall))
    out = RealWorldQAV4ArmOutcomeCapsule(
        schema=W99_REALWORLDQA_BENCH_V4_SCHEMA_VERSION,
        seed=int(seed), pid=str(p.pid),
        arm_id="A0_text",
        question_type=qt,
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
) -> tuple[RealWorldQAV4ArmOutcomeCapsule,
           list[RealWorldQAExecutorResultV1], str]:
    qt = detect_question_type_v2(p.question)
    prompt = _a1_prompt(p)
    calls: list[RealWorldQAV4ArmCallCapsule] = []
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
        calls.append(RealWorldQAV4ArmCallCapsule(
            schema=W99_REALWORLDQA_BENCH_V4_SCHEMA_VERSION,
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
    out = RealWorldQAV4ArmOutcomeCapsule(
        schema=W99_REALWORLDQA_BENCH_V4_SCHEMA_VERSION,
        seed=int(seed), pid=str(p.pid),
        arm_id="A1_vlm",
        question_type=qt,
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


def _run_b_typed_no_hint_vlm_team(
        *, seed: int, p: RealWorldQAProblemV1, K: int,
        temperature: float, vlm_gen: _VlmGenFn,
        text_gen: _TextGenFn, max_tokens: int,
) -> tuple[RealWorldQAV4ArmOutcomeCapsule,
           list[RealWorldQAExecutorResultV1], str]:
    """W99 B4 (D2-B4): typed scene-graph reader (1 VLM call,
    T=0.0; schema sans direct_answer_hint) + question-typed text
    solver (1 + up to 3 reflexion turns) at total K=5 byte-exact.
    Same K budget as A1_vlm and W98 B1.
    """
    qt = detect_question_type_v2(p.question)
    calls: list[RealWorldQAV4ArmCallCapsule] = []
    exes: list[RealWorldQAExecutorResultV1] = []
    candidates: list[str] = []
    history: list[tuple[str, RealWorldQAExecutorResultV1]] = []
    total = 0

    # Stage 1 — Typed VLM scene-graph reader (1 call, T=0.0).
    reader_prompt = _b_typed_reader_prompt(p)
    reader_text, reader_wall = vlm_gen(
        reader_prompt, p.image_bytes, max_tokens, 0.0)
    calls.append(RealWorldQAV4ArmCallCapsule(
        schema=W99_REALWORLDQA_BENCH_V4_SCHEMA_VERSION,
        seed=int(seed), pid=str(p.pid),
        arm_id="B_vlm_team_v4", role="typed_scene_reader_no_hint",
        call_idx=0, temperature=0.0,
        prompt_cid=hashlib.sha256(
            (reader_prompt + "|img:" + p.image_sha256).encode(
                "utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            reader_text.encode("utf-8")).hexdigest(),
        wall_ms=int(reader_wall)))
    total += int(reader_wall)
    scene_graph_text = (
        reader_text.strip() or "(no scene graph)")

    # Stage 2..K — Typed text solver + executor-guided reflexion.
    solver_calls = int(K) - 1
    for k in range(int(solver_calls)):
        if k == 0:
            solver_prompt = _b_typed_solver_initial_prompt(
                p, scene_graph_text, qt)
            solver_role = "typed_text_solver_initial_no_hint"
        else:
            solver_prompt = _b_typed_solver_reflexion_prompt(
                p, scene_graph_text, qt, tuple(history),
                attempt_idx=int(k))
            solver_role = "typed_text_solver_reflexion_no_hint"
        solver_text, solver_wall = text_gen(
            solver_prompt, max_tokens, float(temperature))
        candidate = extract_candidate_answer_v1(
            response_text=solver_text)
        exe = evaluate_realworldqa_answer_v1(
            prediction=candidate, problem=p)
        calls.append(RealWorldQAV4ArmCallCapsule(
            schema=W99_REALWORLDQA_BENCH_V4_SCHEMA_VERSION,
            seed=int(seed), pid=str(p.pid),
            arm_id="B_vlm_team_v4", role=solver_role,
            call_idx=int(k) + 1,
            temperature=float(temperature),
            prompt_cid=hashlib.sha256(
                solver_prompt.encode("utf-8")).hexdigest(),
            response_cid=hashlib.sha256(
                solver_text.encode("utf-8")).hexdigest(),
            wall_ms=int(solver_wall)))
        exes.append(exe)
        candidates.append(candidate)
        history.append((candidate, exe))
        total += int(solver_wall)

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
    out = RealWorldQAV4ArmOutcomeCapsule(
        schema=W99_REALWORLDQA_BENCH_V4_SCHEMA_VERSION,
        seed=int(seed), pid=str(p.pid),
        arm_id="B_vlm_team_v4",
        question_type=qt,
        final_passed=bool(chosen_passed),
        final_prediction_cid=hashlib.sha256(
            chosen_cand.encode("utf-8")).hexdigest(),
        final_executor_rule=str(chosen_rule),
        n_model_calls=int(K),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cid=_executor_result_cid(
            exes[chosen_idx] if exes
            else evaluate_realworldqa_answer_v1(
                prediction="", problem=p)))
    return out, exes, qt


# ---------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RealWorldQAV4SeedReport:
    schema: str
    seed: int
    n_problems: int
    a0_text_pass_at_1: float
    a1_vlm_pass_at_1: float
    b_vlm_team_v4_pass_at_1: float
    a0_text_total_wall_ms: int
    a1_vlm_total_wall_ms: int
    b_vlm_team_v4_total_wall_ms: int
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
            "b_vlm_team_v4_pass_at_1": float(round(
                self.b_vlm_team_v4_pass_at_1, 6)),
            "a0_text_total_wall_ms": int(
                self.a0_text_total_wall_ms),
            "a1_vlm_total_wall_ms": int(
                self.a1_vlm_total_wall_ms),
            "b_vlm_team_v4_total_wall_ms": int(
                self.b_vlm_team_v4_total_wall_ms),
            "per_problem_outcomes": [
                dict(po) for po in self.per_problem_outcomes],
            "outcome_cids": list(self.outcome_cids),
            "seed_merkle_root": str(self.seed_merkle_root),
        }


@dataclasses.dataclass(frozen=True)
class RealWorldQAV4BenchReport:
    schema: str
    vlm_model_id: str
    text_model_id: str
    n_problems: int
    n_seeds: int
    K_multi_sample: int
    corpus_parquet_shard_sha256: tuple[str, ...]
    corpus_merkle_root: str
    per_seed: tuple[RealWorldQAV4SeedReport, ...]
    a0_text_mean_pass_at_1: float
    a1_vlm_mean_pass_at_1: float
    b_vlm_team_v4_mean_pass_at_1: float
    b_beats_a0_text_per_seed: tuple[bool, ...]
    b_beats_a1_vlm_per_seed: tuple[bool, ...]
    b_mean_strictly_beats_a0_text_mean: bool
    b_mean_strictly_beats_a1_vlm_mean: bool
    b_mean_minus_a0_text_mean_pp: float
    b_mean_minus_a1_vlm_mean_pp: float
    n_b_ge_a1_problems_per_seed: tuple[int, ...]
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
            "b_vlm_team_v4_mean_pass_at_1": float(round(
                self.b_vlm_team_v4_mean_pass_at_1, 6)),
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
            "question_type_distribution": dict(
                self.question_type_distribution),
            "bench_merkle_root": str(self.bench_merkle_root),
        }


# ---------------------------------------------------------------
# Config + driver
# ---------------------------------------------------------------

@dataclasses.dataclass
class RealWorldQAV4BenchConfig:
    schema: str = W99_REALWORLDQA_BENCH_V4_SCHEMA_VERSION
    n_problems: int = 30
    K_multi_sample: int = 5
    seeds: tuple[int, ...] = (96_504_002,)
    sampling_temperature: float = 0.7
    max_tokens_per_call: int = 512


def run_realworldqa_bench_v4(
        *,
        text_gen: _TextGenFn,
        vlm_gen: _VlmGenFn,
        vlm_model_id: str,
        text_model_id: str,
        corpus: Sequence[RealWorldQAProblemV1],
        corpus_parquet_shard_sha256: tuple[str, ...],
        corpus_merkle_root: str,
        config: RealWorldQAV4BenchConfig | None = None,
        on_problem_start: (
            Callable[[int, int, str], None] | None) = None,
        sidecar_writer: (
            Callable[[dict[str, Any]], None] | None) = None,
) -> RealWorldQAV4BenchReport:
    """Run A0_text / A1_vlm / B_vlm_team_v4 on the supplied
    corpus at the same K budget."""
    from .realworldqa_loader_v1 import (
        select_realworldqa_subset_v1)

    cfg = config or RealWorldQAV4BenchConfig()
    per_seed: list[RealWorldQAV4SeedReport] = []
    all_outcome_cids: list[str] = []
    qt_counter: dict[str, int] = {}
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
        a0_outs: list[RealWorldQAV4ArmOutcomeCapsule] = []
        a1_outs: list[RealWorldQAV4ArmOutcomeCapsule] = []
        b_outs: list[RealWorldQAV4ArmOutcomeCapsule] = []
        per_problem_outcomes: list[dict[str, Any]] = []
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
            b_out, _, _ = _run_b_typed_no_hint_vlm_team(
                seed=int(seed), p=p,
                K=int(cfg.K_multi_sample),
                temperature=float(cfg.sampling_temperature),
                vlm_gen=vlm_gen,
                text_gen=text_gen,
                max_tokens=int(cfg.max_tokens_per_call))
            b_outs.append(b_out)
            qt_counter[qt] = qt_counter.get(qt, 0) + 1
            per_problem_outcomes.append({
                "pid": str(p.pid),
                "question": str(p.question),
                "gold_answer": str(p.answer),
                "question_type": str(qt),
                "a0_text_passed": bool(a0_out.final_passed),
                "a1_vlm_passed": bool(a1_out.final_passed),
                "b_vlm_team_v4_passed": bool(b_out.final_passed),
                "a0_outcome_cid": str(a0_out.cid()),
                "a1_outcome_cid": str(a1_out.cid()),
                "b_outcome_cid": str(b_out.cid()),
            })
            if sidecar_writer is not None:
                sidecar_writer({
                    "kind": (
                        "w99_realworldqa_v4_per_problem_outcome"),
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
            "kind": "w99_realworldqa_v4_seed_merkle_root",
            "seed": int(seed),
            "outcome_cids": list(outcome_cids),
        })
        per_seed.append(RealWorldQAV4SeedReport(
            schema=W99_REALWORLDQA_BENCH_V4_SCHEMA_VERSION,
            seed=int(seed),
            n_problems=int(len(a0_outs)),
            a0_text_pass_at_1=float(a0_acc),
            a1_vlm_pass_at_1=float(a1_acc),
            b_vlm_team_v4_pass_at_1=float(b_acc),
            a0_text_total_wall_ms=sum(
                int(o.total_wall_ms) for o in a0_outs),
            a1_vlm_total_wall_ms=sum(
                int(o.total_wall_ms) for o in a1_outs),
            b_vlm_team_v4_total_wall_ms=sum(
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
        s.b_vlm_team_v4_pass_at_1 for s in per_seed) / nseeds
    bench_merkle = _sha256_hex({
        "kind": "w99_realworldqa_v4_bench_merkle_root",
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
            if (bool(po["b_vlm_team_v4_passed"])
                    >= bool(po["a1_vlm_passed"])):
                n_bg += 1
        n_b_ge_a1_per_seed.append(int(n_bg))
    report = RealWorldQAV4BenchReport(
        schema=W99_REALWORLDQA_BENCH_V4_SCHEMA_VERSION,
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
        b_vlm_team_v4_mean_pass_at_1=float(b_mean),
        b_beats_a0_text_per_seed=tuple(
            s.b_vlm_team_v4_pass_at_1 > s.a0_text_pass_at_1
            for s in per_seed),
        b_beats_a1_vlm_per_seed=tuple(
            s.b_vlm_team_v4_pass_at_1 > s.a1_vlm_pass_at_1
            for s in per_seed),
        b_mean_strictly_beats_a0_text_mean=bool(b_mean > a0_mean),
        b_mean_strictly_beats_a1_vlm_mean=bool(b_mean > a1_mean),
        b_mean_minus_a0_text_mean_pp=float(
            (b_mean - a0_mean) * 100.0),
        b_mean_minus_a1_vlm_mean_pp=float(
            (b_mean - a1_mean) * 100.0),
        n_b_ge_a1_problems_per_seed=tuple(n_b_ge_a1_per_seed),
        question_type_distribution=dict(qt_counter),
        bench_merkle_root=str(bench_merkle))
    return report


__all__ = [
    "W99_REALWORLDQA_BENCH_V4_SCHEMA_VERSION",
    "RealWorldQAV4ArmCallCapsule",
    "RealWorldQAV4ArmOutcomeCapsule",
    "RealWorldQAV4SeedReport",
    "RealWorldQAV4BenchReport",
    "RealWorldQAV4BenchConfig",
    "run_realworldqa_bench_v4",
]
