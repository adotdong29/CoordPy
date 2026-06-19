"""W90 / Post-W89 — Cross-modal VLM-in-loop reflexion bench V1.

The W88 / W89 cross-modal experiments tested a VLM-extract +
code-LM-generate split.  That split was FALSIFIED at three
model scales (W88 V1, W89 P2, W89 P3): the unified VLM at K=5
beat the split by ~5.6 pp on doctest_only and ~27.8 pp on
all_docstring.

W90 P2 is a STRUCTURAL pivot: drop the split entirely.  Use the
SAME VLM in the loop across all K=5 turns, with the image
in context every turn, conditioning each reflexion turn on
(prior_candidate, executor_stderr) history.  No text-only
extraction handoff; no loss at the modality boundary.

Three arms (same fair-budget K=5 contract as W88 / W89):

* ``A0_text`` — text-only LLM (no image), single-shot at T=0.
* ``A1_vlm`` — single-agent VLM, K=5 INDEPENDENT samples at T=0.7,
  ship first PASS by executor.  Same as W88's A1_vlm arm.
* ``B_vlm_loop`` — SAME VLM, K=5 SEQUENTIAL turns at T=0.7, each
  conditioned on the cumulative (candidate, executor_stderr)
  history.  Image in context every turn.

For B_vlm_loop to retire the W88
`W88-L-CROSS-MODAL-CODE-V1-SPLIT-NOT-LOAD-BEARING-CAP`
carry-forward, ALL 6 retirement bars from `RUNBOOK_W88.md` must
be met:

  (1) B > A0_text mean
  (2) B > A1_vlm mean
  (3) B − A0_text margin ≥ +5.0 pp
  (4) B − A1_vlm margin ≥ +5.0 pp
  (5) B beats A0_text on > half seeds
  (6) B beats A1_vlm on > half seeds

The corpus is reused from `coordpy.cross_modal_code_bench_v1`
(synthesised HumanEval-Visual, two strip modes).  The default
is `doctest_only` (matches W88 V1 and W89 P3 head-to-head
comparison).

Anti-cheat:

* Same VLM model on A1_vlm AND B_vlm_loop's every turn.
* Same task subset per seed across arms (the W88
  `select_cross_modal_subset_v1(seed)` discipline preserved).
* Same prompt budget per arm (K=5 model calls).
* Same retry policy.
* No selective retries.
* Audit chain re-derives offline.

Honest scope
------------

* ``W90-L-CROSS-MODAL-VLM-LOOP-V1-NIM-DEPENDENT-CAP`` — V1
  drives the bench through any VLM client; provider determinism
  beyond temperature=0 is not assumed.
* ``W90-L-CROSS-MODAL-VLM-LOOP-V1-SINGLE-MODEL-CAP`` — V1 uses
  the same VLM model on every turn.  "Multi-agent" here means
  multiple ROLES across turns (initial solver, reflexion-driven
  refiner) — not multiple distinct models.  Retirement of the
  W88 carry-forward via this bench means the SPECIFIC W88
  split (VLM-extract + code-LM-generate) is replaced by the
  VLM-in-loop architecture; the W88 split's empirical failure
  stands.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Sequence

from .cross_modal_code_bench_v1 import (
    CrossModalProblemV1,
    W88_CROSS_MODAL_CODE_BENCH_V1_SCHEMA_VERSION,
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


W90_CROSS_MODAL_VLM_LOOP_BENCH_V1_SCHEMA_VERSION: str = (
    "coordpy.cross_modal_vlm_loop_bench_v1.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


_VLM_SOLVE_SYSTEM = (
    "You are an expert Python programmer.  When given a "
    "function signature and an image containing example inputs "
    "and outputs, output ONLY the complete function (inside a "
    "```python ... ``` code fence).  Do not include any prose "
    "before or after the code fence.")


def _vlm_initial_prompt(p: CrossModalProblemV1) -> str:
    return (
        f"{_VLM_SOLVE_SYSTEM}\n\n"
        "Complete the following Python function.  The example "
        "input/output behaviour is shown in the attached image.  "
        "Provide the full function including the signature.\n\n"
        f"```python\n{p.stripped_prompt}```\n\n"
        "Your complete solution:")


def _vlm_reflexion_prompt(
        p: CrossModalProblemV1,
        history: Sequence[tuple[str, HumanEvalExecutorResultV1]],
        attempt_idx: int,
) -> str:
    chunks: list[str] = []
    for i, (cand, exe) in enumerate(history):
        cand_trim = cand
        if len(cand_trim) > 1500:
            cand_trim = cand_trim[:1500] + "\n# ... (truncated)\n"
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
        chunks.append(
            f"--- Attempt {i+1} ({verdict}) ---\n"
            f"```python\n{cand_trim}\n```{stderr_excerpt}")
    return (
        f"{_VLM_SOLVE_SYSTEM}\n\n"
        "[Role: reflective code generator]\n"
        f"You are on attempt {attempt_idx + 1} out of 5.\n\n"
        "The function's example input/output behaviour is "
        "shown in the attached image (same image as on every "
        "prior attempt).\n\n"
        f"Target function:\n```python\n{p.stripped_prompt}```\n\n"
        f"{chr(10).join(chunks)}\n\n"
        "Diagnose the bug class in the failing attempt(s) and "
        "produce a NEW corrected complete Python function.  Do "
        "not repeat a previous attempt verbatim.  Provide ONLY "
        "the corrected function in a ```python ... ``` fence:")


def _vlm_a0_text_prompt(p: CrossModalProblemV1) -> str:
    return (
        f"{_VLM_SOLVE_SYSTEM}\n\n"
        "Complete the following Python function.  Provide the "
        "full function including the signature.\n\n"
        f"```python\n{p.stripped_prompt}```\n\n"
        "Your complete solution:")


_TextGenFn = Callable[[str, int, float], tuple[str, int]]
# (prompt_text, image_bytes_or_None, max_tokens, temperature)
#   -> (response_text, wall_ms)
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
    prompt = _vlm_a0_text_prompt(p)
    text, wall = text_gen(prompt, max_tokens, 0.0)
    code = extract_candidate_code_v1(
        response_text=text, prompt=p.stripped_prompt,
        entry_point=p.entry_point)
    exe = run_humaneval_executor_v1(
        problem=he_problem, candidate_code=code,
        **executor_kwargs)
    call = HumanEvalArmCallCapsuleV1(
        schema=W90_CROSS_MODAL_VLM_LOOP_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), task_id=str(p.task_id),
        arm_id="A0_text", role="code_solver", call_idx=0,
        temperature=0.0,
        prompt_cid=hashlib.sha256(
            prompt.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            text.encode("utf-8")).hexdigest(),
        wall_ms=int(wall))
    out = HumanEvalArmOutcomeCapsuleV1(
        schema=W90_CROSS_MODAL_VLM_LOOP_BENCH_V1_SCHEMA_VERSION,
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
    prompt = _vlm_initial_prompt(p)
    calls: list[HumanEvalArmCallCapsuleV1] = []
    exes: list[HumanEvalExecutorResultV1] = []
    total = 0
    chosen_passed = False
    chosen_code_cid = ""
    for k in range(int(K)):
        text, wall = vlm_gen(
            prompt, p.image_bytes, max_tokens, float(temperature))
        code = extract_candidate_code_v1(
            response_text=text, prompt=p.stripped_prompt,
            entry_point=p.entry_point)
        exe = run_humaneval_executor_v1(
            problem=he_problem, candidate_code=code,
            **executor_kwargs)
        calls.append(HumanEvalArmCallCapsuleV1(
            schema=(
                W90_CROSS_MODAL_VLM_LOOP_BENCH_V1_SCHEMA_VERSION),
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
    out = HumanEvalArmOutcomeCapsuleV1(
        schema=W90_CROSS_MODAL_VLM_LOOP_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), task_id=str(p.task_id),
        arm_id="A1_vlm",
        final_passed=bool(chosen_passed),
        final_candidate_code_cid=str(chosen_code_cid),
        n_model_calls=int(K), n_executor_calls=int(K),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cids=tuple(e.cid() for e in exes))
    return out, exes


def _run_b_vlm_loop(
        *, seed: int, p: CrossModalProblemV1, K: int,
        temperature: float, vlm_gen: _VlmGenFn,
        max_tokens: int, executor_kwargs: dict[str, Any],
) -> tuple[HumanEvalArmOutcomeCapsuleV1,
           list[HumanEvalExecutorResultV1]]:
    """B: same VLM, K=5 sequential turns, image in context every
    turn, conditioned on cumulative (candidate, executor_stderr)
    history.  Same budget as A1_vlm.
    """
    he_problem = _stripped_to_humaneval_problem(p)
    history: list[tuple[str, HumanEvalExecutorResultV1]] = []
    calls: list[HumanEvalArmCallCapsuleV1] = []
    exes: list[HumanEvalExecutorResultV1] = []
    total = 0
    for k in range(int(K)):
        if k == 0:
            prompt = _vlm_initial_prompt(p)
        else:
            prompt = _vlm_reflexion_prompt(
                p, tuple(history), attempt_idx=int(k))
        text, wall = vlm_gen(
            prompt, p.image_bytes, max_tokens, float(temperature))
        code = extract_candidate_code_v1(
            response_text=text, prompt=p.stripped_prompt,
            entry_point=p.entry_point)
        exe = run_humaneval_executor_v1(
            problem=he_problem, candidate_code=code,
            **executor_kwargs)
        calls.append(HumanEvalArmCallCapsuleV1(
            schema=(
                W90_CROSS_MODAL_VLM_LOOP_BENCH_V1_SCHEMA_VERSION),
            seed=int(seed), task_id=str(p.task_id),
            arm_id="B_vlm_loop",
            role="reflexion" if k > 0 else "initial",
            call_idx=int(k),
            temperature=float(temperature),
            prompt_cid=hashlib.sha256(
                (prompt + "|img:" + p.image_cid).encode(
                    "utf-8")).hexdigest(),
            response_cid=hashlib.sha256(
                text.encode("utf-8")).hexdigest(),
            wall_ms=int(wall)))
        exes.append(exe)
        history.append((code, exe))
        total += int(wall) + int(exe.wall_ms)
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
        schema=W90_CROSS_MODAL_VLM_LOOP_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), task_id=str(p.task_id),
        arm_id="B_vlm_loop",
        final_passed=bool(final_passed),
        final_candidate_code_cid=str(final_code_cid),
        n_model_calls=int(K),
        n_executor_calls=int(len(exes)),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cids=tuple(e.cid() for e in exes))
    return out, exes


# ---------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CrossModalVlmLoopSeedReportV1:
    schema: str
    seed: int
    n_problems: int
    a0_text_pass_at_1: float
    a1_vlm_pass_at_1: float
    b_vlm_loop_pass_at_1: float
    a0_text_total_wall_ms: int
    a1_vlm_total_wall_ms: int
    b_vlm_loop_total_wall_ms: int
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
            "b_vlm_loop_pass_at_1": float(round(
                self.b_vlm_loop_pass_at_1, 6)),
            "a0_text_total_wall_ms": int(
                self.a0_text_total_wall_ms),
            "a1_vlm_total_wall_ms": int(
                self.a1_vlm_total_wall_ms),
            "b_vlm_loop_total_wall_ms": int(
                self.b_vlm_loop_total_wall_ms),
            "outcome_cids": list(self.outcome_cids),
            "seed_merkle_root": str(self.seed_merkle_root),
        }


@dataclasses.dataclass(frozen=True)
class CrossModalVlmLoopBenchReportV1:
    schema: str
    vlm_model_id: str
    text_model_id: str
    n_problems: int
    n_seeds: int
    K_multi_sample: int
    per_seed: tuple[CrossModalVlmLoopSeedReportV1, ...]
    a0_text_mean_pass_at_1: float
    a1_vlm_mean_pass_at_1: float
    b_vlm_loop_mean_pass_at_1: float
    b_vlm_loop_beats_a0_text_per_seed: tuple[bool, ...]
    b_vlm_loop_beats_a1_vlm_per_seed: tuple[bool, ...]
    b_vlm_loop_mean_strictly_beats_a0_text_mean: bool
    b_vlm_loop_mean_strictly_beats_a1_vlm_mean: bool
    b_vlm_loop_mean_minus_a0_text_mean_pp: float
    b_vlm_loop_mean_minus_a1_vlm_mean_pp: float
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
            "b_vlm_loop_mean_pass_at_1": float(round(
                self.b_vlm_loop_mean_pass_at_1, 6)),
            "b_vlm_loop_beats_a0_text_per_seed": list(
                self.b_vlm_loop_beats_a0_text_per_seed),
            "b_vlm_loop_beats_a1_vlm_per_seed": list(
                self.b_vlm_loop_beats_a1_vlm_per_seed),
            "b_vlm_loop_mean_strictly_beats_a0_text_mean": bool(
                self.b_vlm_loop_mean_strictly_beats_a0_text_mean),
            "b_vlm_loop_mean_strictly_beats_a1_vlm_mean": bool(
                self.b_vlm_loop_mean_strictly_beats_a1_vlm_mean),
            "b_vlm_loop_mean_minus_a0_text_mean_pp": float(round(
                self.b_vlm_loop_mean_minus_a0_text_mean_pp, 4)),
            "b_vlm_loop_mean_minus_a1_vlm_mean_pp": float(round(
                self.b_vlm_loop_mean_minus_a1_vlm_mean_pp, 4)),
            "bench_merkle_root": str(self.bench_merkle_root),
        }


# ---------------------------------------------------------------
# Config + driver
# ---------------------------------------------------------------

@dataclasses.dataclass
class CrossModalVlmLoopBenchConfigV1:
    schema: str = (
        W90_CROSS_MODAL_VLM_LOOP_BENCH_V1_SCHEMA_VERSION)
    n_problems: int = 12
    K_multi_sample: int = 5
    seeds: tuple[int, ...] = (90_046_001, 90_046_002, 90_046_003)
    sampling_temperature: float = 0.7
    max_tokens_per_call: int = 768
    executor_timeout_s: float = (
        W86_HUMANEVAL_EXECUTOR_TIMEOUT_S)
    executor_kill_after_s: float = (
        W86_HUMANEVAL_EXECUTOR_KILL_AFTER_S)
    min_doctest_lines: int = 2
    strip_mode: str = "doctest_only"


def run_cross_modal_vlm_loop_bench_v1(
        *,
        text_gen: _TextGenFn,
        vlm_gen: _VlmGenFn,
        vlm_model_id: str,
        text_model_id: str,
        corpus: Sequence[HumanEvalProblemV1],
        config: (
            CrossModalVlmLoopBenchConfigV1 | None) = None,
        on_problem_start: (
            Callable[[int, int, str], None] | None) = None,
) -> tuple[CrossModalVlmLoopBenchReportV1,
           tuple[CrossModalProblemV1, ...]]:
    cfg = config or CrossModalVlmLoopBenchConfigV1()
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
    per_seed: list[CrossModalVlmLoopSeedReportV1] = []
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
            b_out, _ = _run_b_vlm_loop(
                seed=int(seed), p=p,
                K=int(cfg.K_multi_sample),
                temperature=float(cfg.sampling_temperature),
                vlm_gen=vlm_gen,
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
            "kind": "w90_cross_modal_vlm_loop_seed_merkle_root",
            "seed": int(seed),
            "outcome_cids": list(outcome_cids),
        })
        per_seed.append(CrossModalVlmLoopSeedReportV1(
            schema=(
                W90_CROSS_MODAL_VLM_LOOP_BENCH_V1_SCHEMA_VERSION),
            seed=int(seed),
            n_problems=int(len(a0_outs)),
            a0_text_pass_at_1=float(a0_acc),
            a1_vlm_pass_at_1=float(a1_acc),
            b_vlm_loop_pass_at_1=float(b_acc),
            a0_text_total_wall_ms=sum(
                int(o.total_wall_ms) for o in a0_outs),
            a1_vlm_total_wall_ms=sum(
                int(o.total_wall_ms) for o in a1_outs),
            b_vlm_loop_total_wall_ms=sum(
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
        s.b_vlm_loop_pass_at_1 for s in per_seed) / nseeds
    bench_merkle = _sha256_hex({
        "kind": "w90_cross_modal_vlm_loop_bench_merkle_root",
        "vlm_model_id": str(vlm_model_id),
        "text_model_id": str(text_model_id),
        "outcome_cids": list(all_outcome_cids),
        "seeds": list(cfg.seeds),
    })
    report = CrossModalVlmLoopBenchReportV1(
        schema=W90_CROSS_MODAL_VLM_LOOP_BENCH_V1_SCHEMA_VERSION,
        vlm_model_id=str(vlm_model_id),
        text_model_id=str(text_model_id),
        n_problems=int(cfg.n_problems),
        n_seeds=int(len(cfg.seeds)),
        K_multi_sample=int(cfg.K_multi_sample),
        per_seed=tuple(per_seed),
        a0_text_mean_pass_at_1=float(a0_mean),
        a1_vlm_mean_pass_at_1=float(a1_mean),
        b_vlm_loop_mean_pass_at_1=float(b_mean),
        b_vlm_loop_beats_a0_text_per_seed=tuple(
            s.b_vlm_loop_pass_at_1 > s.a0_text_pass_at_1
            for s in per_seed),
        b_vlm_loop_beats_a1_vlm_per_seed=tuple(
            s.b_vlm_loop_pass_at_1 > s.a1_vlm_pass_at_1
            for s in per_seed),
        b_vlm_loop_mean_strictly_beats_a0_text_mean=bool(
            b_mean > a0_mean),
        b_vlm_loop_mean_strictly_beats_a1_vlm_mean=bool(
            b_mean > a1_mean),
        b_vlm_loop_mean_minus_a0_text_mean_pp=float(
            (b_mean - a0_mean) * 100.0),
        b_vlm_loop_mean_minus_a1_vlm_mean_pp=float(
            (b_mean - a1_mean) * 100.0),
        bench_merkle_root=str(bench_merkle))
    return report, cross_corpus


__all__ = [
    "W90_CROSS_MODAL_VLM_LOOP_BENCH_V1_SCHEMA_VERSION",
    "CrossModalVlmLoopSeedReportV1",
    "CrossModalVlmLoopBenchReportV1",
    "CrossModalVlmLoopBenchConfigV1",
    "run_cross_modal_vlm_loop_bench_v1",
]
