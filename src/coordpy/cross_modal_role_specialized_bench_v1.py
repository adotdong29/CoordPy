"""W92 / Post-W91 — Cross-modal role-specialized bench V1.

Architectural pivot from W90/W91's VLM-in-loop (single-model
multi-turn).  W92 ships a TRUE multi-agent cross-modal team
with 3 distinct roles, each using the strongest available
model for its sub-task:

  * VLM-Planner (90B-Vision): reads image + signature →
    structured plan.
  * Code-Implementer (70B Llama-3.3-Instruct, text-only): reads
    signature + plan → code.
  * VLM-Verifier (90B-Vision): reads image + signature +
    candidate code + executor stderr → critique.

Same K=5 model-call budget as A1_vlm (which is 5× 90B-Vision
unified VLM samples).  W92's mix: 2× VLM + 3× code-LM.  Note
B uses LESS total compute than A1_vlm (the code-LM is smaller
than the VLM), so if B wins, the win is doubly informative.

Pipeline (5 model calls):

  Turn 0: VLM-Planner             (image, prompt) → plan
  Turn 1: Code-Implementer-v1     (prompt, plan)  → code_v1
                                  + executor(code_v1)
  Turn 2: VLM-Verifier            (image, prompt, code_v1,
                                   stderr_v1)     → critique
  Turn 3: Code-Implementer-v2     (prompt, plan, code_v1,
                                   stderr_v1,
                                   critique)      → code_v2
                                  + executor(code_v2)
  Turn 4: Code-Implementer-v3     (prompt, plan, code_v1,
                                   code_v2, stderr history,
                                   critique)      → code_v3
                                  + executor(code_v3)

Ship first PASS among (code_v1, code_v2, code_v3); else
lex-smallest CID.

Compared to A1_vlm = K=5 i.i.d. samples from the unified VLM:
* B has 3 code-producing attempts (v1, v2, v3) and 2
  vision-grounded planning/critiquing turns.
* The image stays in context for vision-relevant turns
  (Planner, Verifier).  Code-LM turns get the image content
  ONLY through the Planner's plan + the Verifier's critique
  — i.e., the cross-modal information flow is structured.
* Each turn uses the strongest model for its role (VLM for
  vision; specialist code-LM for code).

Anti-cheat
----------

* Same VLM model on A1_vlm AND B's Planner + Verifier.
* Same code-LM model on A0_text + B's Implementer.
* Same K=5 model-call budget per arm.
* Same task subset per seed across arms (reuses
  `coordpy.cross_modal_code_bench_v1.select_cross_modal_subset_v1`).
* No selective retries.

Honest scope (W92)
------------------

* ``W92-L-CROSS-MODAL-ROLE-SPEC-V1-NIM-DEPENDENT-CAP`` —
  provider determinism beyond temperature=0 is not assumed.
* ``W92-L-CROSS-MODAL-ROLE-SPEC-V1-SAME-VLM-BOTH-ROLES-CAP`` —
  Planner and Verifier use the same VLM model with different
  prompts.  A model-specialized Planner/Verifier pair (e.g.,
  GPT-4o Planner + Claude-4-Vision Verifier) is V2.
* ``W92-L-CROSS-MODAL-ROLE-SPEC-V1-K5-FIXED-PIPELINE-CAP`` —
  V1 uses a fixed 2:3 VLM:code-LM call split.  Adaptive budget
  allocation (e.g., more code-LM retries if Planner is
  uncertain) is V2.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Sequence

from .cross_modal_code_bench_v1 import (
    CrossModalProblemV1,
    select_cross_modal_subset_v1,
    synthesize_cross_modal_corpus_v1,
)
from .humaneval_real_bench_v1 import (
    W86_HUMANEVAL_EXECUTOR_KILL_AFTER_S,
    W86_HUMANEVAL_EXECUTOR_TIMEOUT_S,
    HumanEvalArmCallCapsuleV1,
    HumanEvalArmOutcomeCapsuleV1,
    HumanEvalExecutorResultV1,
    HumanEvalProblemV1,
    extract_candidate_code_v1,
    run_humaneval_executor_v1,
)


W92_CROSS_MODAL_ROLE_SPECIALIZED_BENCH_V1_SCHEMA_VERSION: str = (
    "coordpy.cross_modal_role_specialized_bench_v1.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


_VLM_PLANNER_SYSTEM = (
    "You are a vision-grounded planner for code generation.  "
    "Read the image and the function signature carefully and "
    "output a STRUCTURED PLAN that a downstream code-only model "
    "can implement reliably.")


_VLM_VERIFIER_SYSTEM = (
    "You are a vision-grounded code reviewer.  Read the image, "
    "the function signature, the candidate code, and the Python "
    "executor's stderr.  Diagnose whether the bug is in the "
    "vision-understanding (the plan misread the image) or in "
    "the code-implementation (the code misunderstood the plan).  "
    "Output a STRUCTURED CRITIQUE the implementer can act on.")


_CODE_IMPLEMENTER_SYSTEM = (
    "You are a specialist Python code implementer.  You do NOT "
    "see the image directly.  Read the function signature and "
    "the plan from the vision planner, plus any prior attempts "
    "and executor stderr and verifier critique.  Output ONLY "
    "the complete corrected Python function in a "
    "```python ... ``` code fence.")


def _vlm_planner_prompt(p: CrossModalProblemV1) -> str:
    return (
        f"{_VLM_PLANNER_SYSTEM}\n\n"
        "The function's specification is shown in the attached "
        "image (input/output examples and edge cases).  The "
        "function's signature is:\n\n"
        f"```python\n{p.stripped_prompt}```\n\n"
        "Produce a PLAN with these sections:\n"
        "1. **Function behaviour**: 1-2 sentence summary of "
        "what the function computes.\n"
        "2. **Input/Output examples**: every example pair "
        "from the image, written as `input -> output`.\n"
        "3. **Edge cases**: any boundary conditions visible "
        "in the image.\n"
        "4. **Algorithm sketch**: 1-3 sentence sketch of an "
        "implementation strategy.\n\n"
        "Output ONLY the plan in plain text — do NOT write code.")


def _code_implementer_prompt(
        p: CrossModalProblemV1, plan: str,
        history: Sequence[tuple[str, HumanEvalExecutorResultV1]],
        critique: str | None,
        attempt_idx: int,
) -> str:
    history_chunks: list[str] = []
    for i, (cand, exe) in enumerate(history):
        cand_trim = cand
        if len(cand_trim) > 1500:
            cand_trim = cand_trim[:1500] + "\n# ...\n"
        if exe.passed:
            verdict = "PASSED visible tests"
            stderr_excerpt = ""
        else:
            verdict = (
                f"FAILED (returncode={exe.returncode}"
                + (", TIMED OUT" if exe.timed_out else "")
                + ")")
            stderr_text = exe.stderr_tail.strip()
            stderr_excerpt = (
                f"\nExecutor stderr (tail):\n{stderr_text}"
                if stderr_text else "")
        history_chunks.append(
            f"--- Attempt {i+1} ({verdict}) ---\n"
            f"```python\n{cand_trim}\n```{stderr_excerpt}")
    critique_block = ""
    if critique:
        critique_block = (
            "\n\nVision-Grounded Critique (from VLM-Verifier):\n"
            f"{critique.strip()}\n")
    history_block = (
        "\n\n" + "\n".join(history_chunks)
        if history_chunks else "")
    return (
        f"{_CODE_IMPLEMENTER_SYSTEM}\n\n"
        f"You are on attempt {attempt_idx + 1}.\n\n"
        f"Target function signature:\n```python\n"
        f"{p.stripped_prompt}```\n\n"
        f"Plan (from VLM-Planner):\n{plan.strip()}"
        f"{history_block}"
        f"{critique_block}"
        "\n\nProvide ONLY the corrected complete Python "
        "function in a ```python ... ``` fence:")


def _vlm_verifier_prompt(
        p: CrossModalProblemV1, code: str,
        exe: HumanEvalExecutorResultV1,
        plan: str,
) -> str:
    code_trim = code
    if len(code_trim) > 1500:
        code_trim = code_trim[:1500] + "\n# ...\n"
    if exe.passed:
        verdict = "PASSED visible tests"
        stderr_excerpt = ""
    else:
        verdict = (
            f"FAILED (returncode={exe.returncode}"
            + (", TIMED OUT" if exe.timed_out else "")
            + ")")
        stderr_text = exe.stderr_tail.strip()
        stderr_excerpt = (
            f"\nExecutor stderr (tail):\n{stderr_text}"
            if stderr_text else "")
    return (
        f"{_VLM_VERIFIER_SYSTEM}\n\n"
        f"Target function signature:\n```python\n"
        f"{p.stripped_prompt}```\n\n"
        f"Plan that the implementer was given:\n{plan.strip()}\n\n"
        f"Candidate code (returned by Code-Implementer):\n"
        f"```python\n{code_trim}\n```\n\n"
        f"Executor verdict: {verdict}{stderr_excerpt}\n\n"
        "Compare the candidate code to the image's specification.  "
        "Diagnose:\n"
        "1. **Was the plan correct?** (Did the planner read the "
        "image correctly?)\n"
        "2. **Did the code implement the plan?** (Did the "
        "implementer follow the plan?)\n"
        "3. **What is the bug class?** (off-by-one, wrong "
        "return type, missing edge case, plan-vision-error, "
        "implementer-misread-plan, etc.)\n"
        "4. **What specific fix does the implementer need to "
        "make?**\n\n"
        "Output a SHORT structured critique (3-6 sentences).")


def _a0_text_prompt(p: CrossModalProblemV1) -> str:
    return (
        "You are an expert Python programmer.  Complete the "
        "following Python function.  Provide the full function "
        "including the signature.\n\n"
        f"```python\n{p.stripped_prompt}```\n\n"
        "Your complete solution (in a ```python ... ``` fence):")


def _a1_vlm_prompt(p: CrossModalProblemV1) -> str:
    return (
        "You are an expert Python programmer.  Complete the "
        "following Python function.  The example input/output "
        "behaviour is shown in the attached image.  Provide the "
        "full function including the signature.\n\n"
        f"```python\n{p.stripped_prompt}```\n\n"
        "Your complete solution (in a ```python ... ``` fence):")


_TextGenFn = Callable[[str, int, float], tuple[str, int]]
_VlmGenFn = Callable[
    [str, "bytes | None", int, float], tuple[str, int]]


def _stripped_to_humaneval_problem(
        p: CrossModalProblemV1,
) -> HumanEvalProblemV1:
    return HumanEvalProblemV1(
        task_id=str(p.task_id),
        prompt=str(p.stripped_prompt),
        canonical_solution="",
        test=str(p.test),
        entry_point=str(p.entry_point))


def _run_a0_text(
        *, seed: int, p: CrossModalProblemV1,
        text_gen: _TextGenFn, max_tokens: int,
        executor_kwargs: dict[str, Any],
) -> tuple[HumanEvalArmOutcomeCapsuleV1,
           HumanEvalExecutorResultV1]:
    he_problem = _stripped_to_humaneval_problem(p)
    prompt = _a0_text_prompt(p)
    text, wall = text_gen(prompt, max_tokens, 0.0)
    code = extract_candidate_code_v1(
        response_text=text, prompt=p.stripped_prompt,
        entry_point=p.entry_point)
    exe = run_humaneval_executor_v1(
        problem=he_problem, candidate_code=code,
        **executor_kwargs)
    call = HumanEvalArmCallCapsuleV1(
        schema=W92_CROSS_MODAL_ROLE_SPECIALIZED_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), task_id=str(p.task_id),
        arm_id="A0_text", role="code_solver", call_idx=0,
        temperature=0.0,
        prompt_cid=hashlib.sha256(
            prompt.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            text.encode("utf-8")).hexdigest(),
        wall_ms=int(wall))
    out = HumanEvalArmOutcomeCapsuleV1(
        schema=W92_CROSS_MODAL_ROLE_SPECIALIZED_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), task_id=str(p.task_id),
        arm_id="A0_text",
        final_passed=bool(exe.passed),
        final_candidate_code_cid=str(exe.candidate_code_cid),
        n_model_calls=1, n_executor_calls=1,
        total_wall_ms=int(wall + exe.wall_ms),
        call_capsule_cids=(call.cid(),),
        executor_result_cids=(exe.cid(),))
    return out, exe


def _run_a1_vlm(
        *, seed: int, p: CrossModalProblemV1, K: int,
        temperature: float, vlm_gen: _VlmGenFn,
        max_tokens: int, executor_kwargs: dict[str, Any],
) -> tuple[HumanEvalArmOutcomeCapsuleV1,
           list[HumanEvalExecutorResultV1]]:
    he_problem = _stripped_to_humaneval_problem(p)
    prompt = _a1_vlm_prompt(p)
    calls: list[HumanEvalArmCallCapsuleV1] = []
    exes: list[HumanEvalExecutorResultV1] = []
    total = 0
    chosen_passed = False
    chosen_code_cid = ""
    for k in range(int(K)):
        text, wall = vlm_gen(
            prompt, p.image_bytes, max_tokens,
            float(temperature))
        code = extract_candidate_code_v1(
            response_text=text, prompt=p.stripped_prompt,
            entry_point=p.entry_point)
        exe = run_humaneval_executor_v1(
            problem=he_problem, candidate_code=code,
            **executor_kwargs)
        calls.append(HumanEvalArmCallCapsuleV1(
            schema=(
                W92_CROSS_MODAL_ROLE_SPECIALIZED_BENCH_V1_SCHEMA_VERSION),
            seed=int(seed), task_id=str(p.task_id),
            arm_id="A1_vlm", role="vlm_sample",
            call_idx=int(k),
            temperature=float(temperature),
            prompt_cid=hashlib.sha256(
                (prompt + "|img:" + p.image_cid).encode(
                    "utf-8")).hexdigest(),
            response_cid=hashlib.sha256(
                text.encode("utf-8")).hexdigest(),
            wall_ms=int(wall)))
        exes.append(exe)
        total += int(wall) + int(exe.wall_ms)
        if exe.passed and not chosen_passed:
            chosen_passed = True
            chosen_code_cid = str(exe.candidate_code_cid)
    if not chosen_passed:
        chosen_code_cid = str(exes[0].candidate_code_cid)
    return HumanEvalArmOutcomeCapsuleV1(
        schema=W92_CROSS_MODAL_ROLE_SPECIALIZED_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), task_id=str(p.task_id),
        arm_id="A1_vlm",
        final_passed=bool(chosen_passed),
        final_candidate_code_cid=str(chosen_code_cid),
        n_model_calls=int(K), n_executor_calls=int(K),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cids=tuple(e.cid() for e in exes)), exes


def _run_b_role_spec(
        *, seed: int, p: CrossModalProblemV1, K: int,
        temperature: float,
        vlm_gen: _VlmGenFn, text_gen: _TextGenFn,
        max_tokens: int, executor_kwargs: dict[str, Any],
) -> tuple[HumanEvalArmOutcomeCapsuleV1,
           list[HumanEvalExecutorResultV1]]:
    """B: role-specialized team — VLM-Planner, Code-Implementer
    (x3), VLM-Verifier.  Same K=5 budget as A1_vlm.

    5 model calls (in order):
      0: VLM-Planner            (T=0.0)
      1: Code-Implementer-v1    (T=temperature)
      2: VLM-Verifier           (T=0.0)
      3: Code-Implementer-v2    (T=temperature)
      4: Code-Implementer-v3    (T=temperature)

    Note: K=5 is a hard constraint inherited from W88/W89/W90/W91.
    """
    if int(K) != 5:
        raise ValueError(
            f"role-specialized bench requires K=5, got {K}")
    he_problem = _stripped_to_humaneval_problem(p)
    calls: list[HumanEvalArmCallCapsuleV1] = []
    exes: list[HumanEvalExecutorResultV1] = []
    total = 0

    # Turn 0: VLM-Planner
    planner_prompt = _vlm_planner_prompt(p)
    plan_text, plan_wall = vlm_gen(
        planner_prompt, p.image_bytes, max_tokens, 0.0)
    plan_text_clean = plan_text  # used in subsequent turns
    calls.append(HumanEvalArmCallCapsuleV1(
        schema=W92_CROSS_MODAL_ROLE_SPECIALIZED_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), task_id=str(p.task_id),
        arm_id="B_role_spec", role="vlm_planner",
        call_idx=0,
        temperature=0.0,
        prompt_cid=hashlib.sha256(
            (planner_prompt + "|img:" + p.image_cid).encode(
                "utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            plan_text.encode("utf-8")).hexdigest(),
        wall_ms=int(plan_wall)))
    total += int(plan_wall)

    # Turn 1: Code-Implementer-v1 (no prior history; no critique
    # yet)
    history: list[tuple[str, HumanEvalExecutorResultV1]] = []
    impl_v1_prompt = _code_implementer_prompt(
        p, plan_text_clean, history, critique=None,
        attempt_idx=0)
    impl_v1_text, impl_v1_wall = text_gen(
        impl_v1_prompt, max_tokens, float(temperature))
    code_v1 = extract_candidate_code_v1(
        response_text=impl_v1_text,
        prompt=p.stripped_prompt,
        entry_point=p.entry_point)
    exe_v1 = run_humaneval_executor_v1(
        problem=he_problem, candidate_code=code_v1,
        **executor_kwargs)
    calls.append(HumanEvalArmCallCapsuleV1(
        schema=W92_CROSS_MODAL_ROLE_SPECIALIZED_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), task_id=str(p.task_id),
        arm_id="B_role_spec", role="code_implementer_v1",
        call_idx=1,
        temperature=float(temperature),
        prompt_cid=hashlib.sha256(
            impl_v1_prompt.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            impl_v1_text.encode("utf-8")).hexdigest(),
        wall_ms=int(impl_v1_wall)))
    exes.append(exe_v1)
    history.append((code_v1, exe_v1))
    total += int(impl_v1_wall) + int(exe_v1.wall_ms)

    # Turn 2: VLM-Verifier (T=0)
    verifier_prompt = _vlm_verifier_prompt(
        p, code_v1, exe_v1, plan_text_clean)
    critique_text, verifier_wall = vlm_gen(
        verifier_prompt, p.image_bytes, max_tokens, 0.0)
    calls.append(HumanEvalArmCallCapsuleV1(
        schema=W92_CROSS_MODAL_ROLE_SPECIALIZED_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), task_id=str(p.task_id),
        arm_id="B_role_spec", role="vlm_verifier",
        call_idx=2,
        temperature=0.0,
        prompt_cid=hashlib.sha256(
            (verifier_prompt + "|img:" + p.image_cid).encode(
                "utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            critique_text.encode("utf-8")).hexdigest(),
        wall_ms=int(verifier_wall)))
    total += int(verifier_wall)

    # Turn 3: Code-Implementer-v2 (with plan, history, critique)
    impl_v2_prompt = _code_implementer_prompt(
        p, plan_text_clean, tuple(history), critique_text,
        attempt_idx=1)
    impl_v2_text, impl_v2_wall = text_gen(
        impl_v2_prompt, max_tokens, float(temperature))
    code_v2 = extract_candidate_code_v1(
        response_text=impl_v2_text,
        prompt=p.stripped_prompt,
        entry_point=p.entry_point)
    exe_v2 = run_humaneval_executor_v1(
        problem=he_problem, candidate_code=code_v2,
        **executor_kwargs)
    calls.append(HumanEvalArmCallCapsuleV1(
        schema=W92_CROSS_MODAL_ROLE_SPECIALIZED_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), task_id=str(p.task_id),
        arm_id="B_role_spec", role="code_implementer_v2",
        call_idx=3,
        temperature=float(temperature),
        prompt_cid=hashlib.sha256(
            impl_v2_prompt.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            impl_v2_text.encode("utf-8")).hexdigest(),
        wall_ms=int(impl_v2_wall)))
    exes.append(exe_v2)
    history.append((code_v2, exe_v2))
    total += int(impl_v2_wall) + int(exe_v2.wall_ms)

    # Turn 4: Code-Implementer-v3 (cumulative history + critique
    # re-used)
    impl_v3_prompt = _code_implementer_prompt(
        p, plan_text_clean, tuple(history), critique_text,
        attempt_idx=2)
    impl_v3_text, impl_v3_wall = text_gen(
        impl_v3_prompt, max_tokens, float(temperature))
    code_v3 = extract_candidate_code_v1(
        response_text=impl_v3_text,
        prompt=p.stripped_prompt,
        entry_point=p.entry_point)
    exe_v3 = run_humaneval_executor_v1(
        problem=he_problem, candidate_code=code_v3,
        **executor_kwargs)
    calls.append(HumanEvalArmCallCapsuleV1(
        schema=W92_CROSS_MODAL_ROLE_SPECIALIZED_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), task_id=str(p.task_id),
        arm_id="B_role_spec", role="code_implementer_v3",
        call_idx=4,
        temperature=float(temperature),
        prompt_cid=hashlib.sha256(
            impl_v3_prompt.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            impl_v3_text.encode("utf-8")).hexdigest(),
        wall_ms=int(impl_v3_wall)))
    exes.append(exe_v3)
    history.append((code_v3, exe_v3))
    total += int(impl_v3_wall) + int(exe_v3.wall_ms)

    # Selection: first PASS among (v1, v2, v3); else
    # lex-smallest CID.
    final_passed = False
    final_code_cid = ""
    for e in exes:
        if e.passed:
            final_passed = True
            final_code_cid = str(e.candidate_code_cid)
            break
    if not final_passed and exes:
        cids = sorted(str(e.candidate_code_cid) for e in exes)
        final_code_cid = cids[0]

    out = HumanEvalArmOutcomeCapsuleV1(
        schema=W92_CROSS_MODAL_ROLE_SPECIALIZED_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), task_id=str(p.task_id),
        arm_id="B_role_spec",
        final_passed=bool(final_passed),
        final_candidate_code_cid=str(final_code_cid),
        n_model_calls=5,
        n_executor_calls=int(len(exes)),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cids=tuple(e.cid() for e in exes))
    return out, exes


# ---------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CrossModalRoleSpecSeedReportV1:
    schema: str
    seed: int
    n_problems: int
    a0_text_pass_at_1: float
    a1_vlm_pass_at_1: float
    b_role_spec_pass_at_1: float
    a0_text_total_wall_ms: int
    a1_vlm_total_wall_ms: int
    b_role_spec_total_wall_ms: int
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
            "b_role_spec_pass_at_1": float(round(
                self.b_role_spec_pass_at_1, 6)),
            "a0_text_total_wall_ms": int(
                self.a0_text_total_wall_ms),
            "a1_vlm_total_wall_ms": int(
                self.a1_vlm_total_wall_ms),
            "b_role_spec_total_wall_ms": int(
                self.b_role_spec_total_wall_ms),
            "outcome_cids": list(self.outcome_cids),
            "seed_merkle_root": str(self.seed_merkle_root),
        }


@dataclasses.dataclass(frozen=True)
class CrossModalRoleSpecBenchReportV1:
    schema: str
    vlm_model_id: str
    text_model_id: str
    n_problems: int
    n_seeds: int
    K_multi_sample: int
    per_seed: tuple[CrossModalRoleSpecSeedReportV1, ...]
    a0_text_mean_pass_at_1: float
    a1_vlm_mean_pass_at_1: float
    b_role_spec_mean_pass_at_1: float
    b_role_spec_beats_a0_text_per_seed: tuple[bool, ...]
    b_role_spec_beats_a1_vlm_per_seed: tuple[bool, ...]
    b_role_spec_mean_strictly_beats_a0_text_mean: bool
    b_role_spec_mean_strictly_beats_a1_vlm_mean: bool
    b_role_spec_mean_minus_a0_text_mean_pp: float
    b_role_spec_mean_minus_a1_vlm_mean_pp: float
    bench_merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "vlm_model_id": str(self.vlm_model_id),
            "text_model_id": str(self.text_model_id),
            "n_problems": int(self.n_problems),
            "n_seeds": int(self.n_seeds),
            "K_multi_sample": int(self.K_multi_sample),
            "per_seed": [s.to_dict() for s in self.per_seed],
            "a0_text_mean_pass_at_1": float(round(
                self.a0_text_mean_pass_at_1, 6)),
            "a1_vlm_mean_pass_at_1": float(round(
                self.a1_vlm_mean_pass_at_1, 6)),
            "b_role_spec_mean_pass_at_1": float(round(
                self.b_role_spec_mean_pass_at_1, 6)),
            "b_role_spec_beats_a0_text_per_seed": list(
                self.b_role_spec_beats_a0_text_per_seed),
            "b_role_spec_beats_a1_vlm_per_seed": list(
                self.b_role_spec_beats_a1_vlm_per_seed),
            "b_role_spec_mean_strictly_beats_a0_text_mean": bool(
                self.b_role_spec_mean_strictly_beats_a0_text_mean),
            "b_role_spec_mean_strictly_beats_a1_vlm_mean": bool(
                self.b_role_spec_mean_strictly_beats_a1_vlm_mean),
            "b_role_spec_mean_minus_a0_text_mean_pp": float(round(
                self.b_role_spec_mean_minus_a0_text_mean_pp, 4)),
            "b_role_spec_mean_minus_a1_vlm_mean_pp": float(round(
                self.b_role_spec_mean_minus_a1_vlm_mean_pp, 4)),
            "bench_merkle_root": str(self.bench_merkle_root),
        }


# ---------------------------------------------------------------
# Config + driver
# ---------------------------------------------------------------

@dataclasses.dataclass
class CrossModalRoleSpecBenchConfigV1:
    schema: str = (
        W92_CROSS_MODAL_ROLE_SPECIALIZED_BENCH_V1_SCHEMA_VERSION)
    n_problems: int = 12
    K_multi_sample: int = 5
    seeds: tuple[int, ...] = (
        90_046_001, 90_046_002, 90_046_003,
        90_046_004, 90_046_005, 90_046_006, 90_046_007)
    sampling_temperature: float = 0.7
    max_tokens_per_call: int = 768
    executor_timeout_s: float = (
        W86_HUMANEVAL_EXECUTOR_TIMEOUT_S)
    executor_kill_after_s: float = (
        W86_HUMANEVAL_EXECUTOR_KILL_AFTER_S)
    min_doctest_lines: int = 2
    strip_mode: str = "all_docstring"


def run_cross_modal_role_specialized_bench_v1(
        *,
        text_gen: _TextGenFn,
        vlm_gen: _VlmGenFn,
        vlm_model_id: str,
        text_model_id: str,
        corpus: Sequence[HumanEvalProblemV1],
        config: (
            CrossModalRoleSpecBenchConfigV1 | None) = None,
        on_problem_start: (
            Callable[[int, int, str], None] | None) = None,
) -> tuple[CrossModalRoleSpecBenchReportV1,
           tuple[CrossModalProblemV1, ...]]:
    cfg = config or CrossModalRoleSpecBenchConfigV1()
    executor_kwargs = {
        "timeout_s": float(cfg.executor_timeout_s),
        "kill_after_s": float(cfg.executor_kill_after_s),
    }
    cross_corpus = synthesize_cross_modal_corpus_v1(
        corpus, min_doctest_lines=int(cfg.min_doctest_lines),
        strip_mode=str(cfg.strip_mode))
    if int(len(cross_corpus)) < int(cfg.n_problems):
        raise RuntimeError(
            f"cross-modal corpus has only {len(cross_corpus)} "
            f"problems with ≥ {cfg.min_doctest_lines} doctest "
            f"lines; need {cfg.n_problems}")
    per_seed: list[CrossModalRoleSpecSeedReportV1] = []
    all_outcome_cids: list[str] = []
    for seed in cfg.seeds:
        subset = select_cross_modal_subset_v1(
            corpus=cross_corpus,
            n_problems=int(cfg.n_problems), seed=int(seed))
        a0_outs: list[HumanEvalArmOutcomeCapsuleV1] = []
        a1_outs: list[HumanEvalArmOutcomeCapsuleV1] = []
        b_outs: list[HumanEvalArmOutcomeCapsuleV1] = []
        for p_idx, p in enumerate(subset):
            if on_problem_start is not None:
                on_problem_start(
                    int(seed), int(p_idx), str(p.task_id))
            a0_out, _ = _run_a0_text(
                seed=int(seed), p=p, text_gen=text_gen,
                max_tokens=int(cfg.max_tokens_per_call),
                executor_kwargs=executor_kwargs)
            a0_outs.append(a0_out)
            a1_out, _ = _run_a1_vlm(
                seed=int(seed), p=p,
                K=int(cfg.K_multi_sample),
                temperature=float(cfg.sampling_temperature),
                vlm_gen=vlm_gen,
                max_tokens=int(cfg.max_tokens_per_call),
                executor_kwargs=executor_kwargs)
            a1_outs.append(a1_out)
            b_out, _ = _run_b_role_spec(
                seed=int(seed), p=p,
                K=int(cfg.K_multi_sample),
                temperature=float(cfg.sampling_temperature),
                vlm_gen=vlm_gen, text_gen=text_gen,
                max_tokens=int(cfg.max_tokens_per_call),
                executor_kwargs=executor_kwargs)
            b_outs.append(b_out)
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
            "kind": "w92_cross_modal_role_spec_seed_merkle_root",
            "seed": int(seed),
            "outcome_cids": list(outcome_cids),
        })
        per_seed.append(CrossModalRoleSpecSeedReportV1(
            schema=(
                W92_CROSS_MODAL_ROLE_SPECIALIZED_BENCH_V1_SCHEMA_VERSION),
            seed=int(seed),
            n_problems=int(len(a0_outs)),
            a0_text_pass_at_1=float(a0_acc),
            a1_vlm_pass_at_1=float(a1_acc),
            b_role_spec_pass_at_1=float(b_acc),
            a0_text_total_wall_ms=sum(
                int(o.total_wall_ms) for o in a0_outs),
            a1_vlm_total_wall_ms=sum(
                int(o.total_wall_ms) for o in a1_outs),
            b_role_spec_total_wall_ms=sum(
                int(o.total_wall_ms) for o in b_outs),
            outcome_cids=outcome_cids,
            seed_merkle_root=str(seed_merkle)))
        all_outcome_cids.extend(outcome_cids)
    nseeds = float(len(per_seed))
    a0_mean = sum(
        s.a0_text_pass_at_1 for s in per_seed) / nseeds
    a1_mean = sum(
        s.a1_vlm_pass_at_1 for s in per_seed) / nseeds
    b_mean = sum(
        s.b_role_spec_pass_at_1 for s in per_seed) / nseeds
    bench_merkle = _sha256_hex({
        "kind": "w92_cross_modal_role_spec_bench_merkle_root",
        "vlm_model_id": str(vlm_model_id),
        "text_model_id": str(text_model_id),
        "outcome_cids": list(all_outcome_cids),
        "seeds": list(cfg.seeds),
    })
    report = CrossModalRoleSpecBenchReportV1(
        schema=W92_CROSS_MODAL_ROLE_SPECIALIZED_BENCH_V1_SCHEMA_VERSION,
        vlm_model_id=str(vlm_model_id),
        text_model_id=str(text_model_id),
        n_problems=int(cfg.n_problems),
        n_seeds=int(len(cfg.seeds)),
        K_multi_sample=int(cfg.K_multi_sample),
        per_seed=tuple(per_seed),
        a0_text_mean_pass_at_1=float(a0_mean),
        a1_vlm_mean_pass_at_1=float(a1_mean),
        b_role_spec_mean_pass_at_1=float(b_mean),
        b_role_spec_beats_a0_text_per_seed=tuple(
            s.b_role_spec_pass_at_1 > s.a0_text_pass_at_1
            for s in per_seed),
        b_role_spec_beats_a1_vlm_per_seed=tuple(
            s.b_role_spec_pass_at_1 > s.a1_vlm_pass_at_1
            for s in per_seed),
        b_role_spec_mean_strictly_beats_a0_text_mean=bool(
            b_mean > a0_mean),
        b_role_spec_mean_strictly_beats_a1_vlm_mean=bool(
            b_mean > a1_mean),
        b_role_spec_mean_minus_a0_text_mean_pp=float(
            (b_mean - a0_mean) * 100.0),
        b_role_spec_mean_minus_a1_vlm_mean_pp=float(
            (b_mean - a1_mean) * 100.0),
        bench_merkle_root=str(bench_merkle))
    return report, cross_corpus


__all__ = [
    "W92_CROSS_MODAL_ROLE_SPECIALIZED_BENCH_V1_SCHEMA_VERSION",
    "CrossModalRoleSpecSeedReportV1",
    "CrossModalRoleSpecBenchReportV1",
    "CrossModalRoleSpecBenchConfigV1",
    "run_cross_modal_role_specialized_bench_v1",
]
