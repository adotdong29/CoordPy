"""W96-C — MathVista same-budget bench V2 (C1 VLM-Verifier-Final-Turn).

V2 forks the W95-B0 V1 bench module to address the structural
mechanism that W96-A Phase 3 empirically identified as the cap
on the W95-B0 architecture: the ``math_solver`` and all reflexion
turns are **blind to the image**, so problems where the
``vlm_reader``'s text extraction is lossy are unrecoverable in
B at multi-seed retirement scale.  At 90B-Vision this hurts:
A1-only rescues jump from 33 (11B) to 45 (90B), B-only rescues
drop from 44 to 30, and B − A1 swings from +3.67 pp to −5.00 pp.

W96-C C1's architecture refinement (this V2 bench's B arm):

  * Reuse the V1 ``vlm_reader`` (1 VLM call, T=0.0) and
    ``math_solver`` (1 text call, T=temperature) and the first
    two ``math_solver`` reflexion turns (text-only, T=temperature)
    unchanged.  These three solver turns are the W95-B0 path
    that already produces B-only rescues at 11B and 90B.
  * **Replace the third reflexion turn** with a
    ``vlm_verifier_final`` call (1 VLM call, T=0.0) that SEES
    the image, the structured extraction, the three prior text-
    only candidate answers, and their executor verdicts.  The
    verifier produces a final answer USING THE IMAGE DIRECTLY,
    regaining the image-grounded reasoning that the W95-B0
    architecture lost at the math_solver handoff.

Budget shape (byte-exact K=5):

  W95-B0 V1:  vlm_reader + solver + reflexion + reflexion + reflexion = 5
  W96-C V2:   vlm_reader + solver + reflexion + reflexion + vlm_verifier_final = 5

Selection rule (anti-regression):

  1. If any of (solver_v1, reflexion_v1, reflexion_v2) PASS the
     executor → ship that text-only candidate (this preserves
     every W95-B0 B-only rescue that completes within the first
     3 solver turns; the only B-only rescues this loses are those
     that needed the 4th solver turn in V1).
  2. Else if vlm_verifier_final PASS the executor → ship the
     verifier's answer (this is the A1-only-rescue territory
     that V1 could not access).
  3. Else → ship the vlm_verifier_final answer (it has at least
     looked at the image; better last-guess than the third
     text-only reflexion).

A0 and A1 are byte-identical to V1's A0 and A1 (same prompt
text, same gen-function signature, same per-arm budget).

Anti-cheat (carry-forward from W95):

  * Same VLM model on A1_vlm AND B_vlm_team_v2's vlm_reader AND
    vlm_verifier_final.
  * Same text-LM on A0_text and the B_vlm_team_v2 solver / 2
    reflexion turns.
  * Same per-problem slice by deterministic
    ``select_mathvista_subset_v1(seed, n_problems, corpus)``.
  * Same budget per arm (K=5 calls on A1 and B; 1 on A0).
  * Same executor truth (``evaluate_answer_v1``).
  * Same content-addressed capsule shapes (re-use V1's
    ``MathVistaArmCallCapsuleV1`` / ``MathVistaArmOutcomeCapsuleV1``
    so audit-chain code is identical).
  * No selective retries.  No LLM judge.

Honest scope (W96-C V2)
-----------------------

* ``W96-L-MATHVISTA-BENCH-V2-VLM-VERIFIER-FINAL-K5-CAP`` — V2
  trades one text-only reflexion turn for one VLM-verifier turn
  at fixed K=5.  V2 does NOT add budget; it re-allocates within
  the W95 contract so that the comparison vs A1_vlm K=5 stays
  honest.
* ``W96-L-MATHVISTA-BENCH-V2-SAME-MODEL-FAMILY-CAP`` — V2's
  A0 / A1 / B all use the same VLM family (Llama-3.2-Vision)
  by default.  A model-specialized verifier (e.g., a stronger
  VLM only for the verifier turn) is V3.
* ``W96-L-MATHVISTA-BENCH-V2-CANDIDATE-EXTRACTION-CAP`` — V2
  parses the last (non-empty, post-strip-trivial-wrappers) line
  of the model's response as the candidate answer (re-uses V1's
  ``extract_candidate_answer_v1``).  Same extractor for every
  arm (anti-cheat).
* ``W96-L-MATHVISTA-BENCH-V2-VLM-VERIFIER-SHIPS-NEW-ANSWER-CAP``
  — the V2 verifier produces a FREE-FORM new answer rather than
  emitting an accept/reject token over the existing candidates.
  This is intentional: the verifier is the team's last chance
  to access the image directly, so it must be able to output an
  answer the text-only solver chain never proposed.  Accept-or-
  reject-only variants would be cheaper to parse but cannot
  rescue A1-only territory; that variant is W96-C-V2b (out of
  scope for V2).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Sequence

from .mathvista_bench_v1 import (
    MathVistaArmCallCapsuleV1,
    MathVistaArmOutcomeCapsuleV1,
    MathVistaSeedReportV1,
    MathVistaBenchReportV1,
    MathVistaBenchConfigV1,
    W95_MATHVISTA_BENCH_V1_SCHEMA_VERSION,
    _b_reader_prompt,
    _b_solver_initial_prompt,
    _b_solver_reflexion_prompt,
    _run_a0_text,
    _run_a1_vlm,
    extract_candidate_answer_v1,
)
from .mathvista_executor_v1 import (
    MathVistaExecutorResultV1,
    evaluate_answer_v1,
)
from .mathvista_loader_v1 import (
    MathVistaProblemV1,
)


W96_MATHVISTA_BENCH_V2_SCHEMA_VERSION: str = (
    "coordpy.mathvista_bench_v2.v2")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    if isinstance(payload, (bytes, bytearray)):
        return hashlib.sha256(payload).hexdigest()
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _executor_result_cid(
        result: MathVistaExecutorResultV1) -> str:
    return _sha256_hex({
        "kind": "w95_mathvista_executor_result_v1",
        **result.to_dict()})


# ---------------------------------------------------------------
# VLM-Verifier-Final prompt (the W96-C C1 mechanism)
# ---------------------------------------------------------------

_VLM_VERIFIER_FINAL_SYSTEM = (
    "You are an expert math problem-solver who reads images.  "
    "A text-only solver chain has been working on this problem; "
    "you can SEE the image directly while they could not.  Read "
    "the image carefully — pay attention to features the text "
    "extraction may have missed (small numbers, axis labels, "
    "geometric relations, color-coded data, tick marks).  Then "
    "look at the candidate answers the text-only solvers "
    "produced and the executor's verdict on each.  Produce a "
    "SINGLE FINAL ANSWER using your direct view of the image.  "
    "Do NOT explain your reasoning.  If the question is multi-"
    "choice, output ONLY the letter (A, B, C, ...).  If the "
    "question is numeric, output ONLY the number.")


def _b_vlm_verifier_final_prompt_v2(
        p: MathVistaProblemV1,
        extraction: str,
        candidates: Sequence[str],
        exes: Sequence[MathVistaExecutorResultV1],
) -> str:
    """Render the VLM-Verifier-Final prompt.  The verifier sees
    the image (via the VLM gen-function), the question, the
    structured extraction the vlm_reader produced, every text-
    only candidate answer the solver chain produced, and the
    executor's PASS / FAIL verdict on each.  Output: a single
    final answer line in the same shape the bench's
    extract_candidate_answer_v1 will parse."""
    history_chunks: list[str] = []
    for i, (cand, exe) in enumerate(zip(candidates, exes)):
        rule = exe.matched_rule
        if exe.passed:
            verdict = f"PASS ({rule})"
            diag = ""
        else:
            verdict = f"FAIL ({rule})"
            diag = (
                f"\n  Normalized prediction: "
                f"{exe.normalized_prediction}")
        history_chunks.append(
            f"  - Text-only attempt {i + 1}: "
            f"candidate=`{cand}` → {verdict}{diag}")
    history_block = (
        "\n".join(history_chunks) if history_chunks
        else "  (no prior attempts)")
    return (
        f"{_VLM_VERIFIER_FINAL_SYSTEM}\n\n"
        f"Question:\n{p.query}\n\n"
        "Structured facts extracted from the image by the "
        f"vision-reader (may be incomplete or wrong):\n"
        f"{extraction}\n\n"
        "Prior text-only solver attempts (the solver could not "
        f"see the image):\n{history_block}\n\n"
        "Re-examine the IMAGE directly.  Produce a single final "
        "answer based on what you see in the image.  Output "
        "ONLY the letter (multi-choice) or the number / short "
        "answer; no prose.\n\nFinal answer:")


# ---------------------------------------------------------------
# Gen fn signatures
# ---------------------------------------------------------------

_TextGenFn = Callable[[str, int, float], tuple[str, int]]
_VlmGenFn = Callable[
    [str, "bytes | None", int, float], tuple[str, int]]


# ---------------------------------------------------------------
# B arm runner (W96-C C1)
# ---------------------------------------------------------------

def _run_b_vlm_team_v2(
        *, seed: int, p: MathVistaProblemV1, K: int,
        temperature: float, vlm_gen: _VlmGenFn,
        text_gen: _TextGenFn, max_tokens: int,
) -> tuple[MathVistaArmOutcomeCapsuleV1,
           list[MathVistaExecutorResultV1]]:
    """W96-C C1 (V2 B arm).

    K=5 byte-exact:
      Turn 0: vlm_reader            (T=0.0, sees image)
      Turn 1: math_solver_initial   (T=temperature, text-only)
      Turn 2: math_solver_reflexion (T=temperature, text-only)
      Turn 3: math_solver_reflexion (T=temperature, text-only)
      Turn 4: vlm_verifier_final    (T=0.0, sees image + history)

    Selection (in order):
      1. First text-only PASS among solver/reflexion turns.
      2. Else if vlm_verifier_final PASS → verifier answer.
      3. Else → vlm_verifier_final answer (last guess; image-
         grounded so better than text-only last attempt).
    """
    if int(K) != 5:
        raise ValueError(
            f"B_vlm_team_v2 requires K=5, got {K}")

    calls: list[MathVistaArmCallCapsuleV1] = []
    exes: list[MathVistaExecutorResultV1] = []
    candidates: list[str] = []
    history: list[tuple[str, MathVistaExecutorResultV1]] = []
    total = 0

    # Turn 0 — VLM-Vision-Reader (1 call, T=0.0).
    reader_prompt = _b_reader_prompt(p)
    reader_text, reader_wall = vlm_gen(
        reader_prompt, p.image_bytes, max_tokens, 0.0)
    calls.append(MathVistaArmCallCapsuleV1(
        schema=W96_MATHVISTA_BENCH_V2_SCHEMA_VERSION,
        seed=int(seed), pid=str(p.pid),
        arm_id="B_vlm_team_v2", role="vision_reader",
        call_idx=0, temperature=0.0,
        prompt_cid=hashlib.sha256(
            (reader_prompt + "|img:" + p.image_sha256).encode(
                "utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            reader_text.encode("utf-8")).hexdigest(),
        wall_ms=int(reader_wall)))
    total += int(reader_wall)
    extraction = reader_text.strip() or "(no extraction)"

    # Turns 1..3 — Math-Solver-v1 / reflexion-v1 / reflexion-v2.
    # V2 has 3 text-only solver turns (V1 had 4); the freed
    # budget is spent on the VLM-Verifier-Final turn at slot 4.
    n_solver_turns = 3
    for k in range(n_solver_turns):
        if k == 0:
            solver_prompt = _b_solver_initial_prompt(
                p, extraction)
            solver_role = "math_solver_initial"
        else:
            solver_prompt = _b_solver_reflexion_prompt(
                p, extraction, tuple(history),
                attempt_idx=int(k))
            solver_role = "math_solver_reflexion"
        solver_text, solver_wall = text_gen(
            solver_prompt, max_tokens, float(temperature))
        candidate = extract_candidate_answer_v1(
            response_text=solver_text)
        exe = evaluate_answer_v1(
            prediction=candidate, problem=p)
        calls.append(MathVistaArmCallCapsuleV1(
            schema=W96_MATHVISTA_BENCH_V2_SCHEMA_VERSION,
            seed=int(seed), pid=str(p.pid),
            arm_id="B_vlm_team_v2", role=solver_role,
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

    # Check whether any text-only solver already PASSed; this
    # short-circuits the verifier so we ship the W95-B0 win.
    text_only_pass_idx = -1
    for i, exe in enumerate(exes):
        if exe.passed:
            text_only_pass_idx = i
            break

    # Turn 4 — VLM-Verifier-Final (1 call, T=0.0, sees image
    # + extraction + all 3 prior candidates + executor verdicts).
    # We ALWAYS run the verifier (no early termination by text-
    # only PASS) so the K=5 budget is byte-exact on EVERY problem
    # — same shape as A1_vlm's K=5 i.i.d. samples even on easy
    # problems.  The W95-B0 V1 bench also padded by running all
    # solver turns even after a PASS for the same reason.
    verifier_prompt = _b_vlm_verifier_final_prompt_v2(
        p, extraction, candidates, exes)
    verifier_text, verifier_wall = vlm_gen(
        verifier_prompt, p.image_bytes, max_tokens, 0.0)
    verifier_candidate = extract_candidate_answer_v1(
        response_text=verifier_text)
    verifier_exe = evaluate_answer_v1(
        prediction=verifier_candidate, problem=p)
    calls.append(MathVistaArmCallCapsuleV1(
        schema=W96_MATHVISTA_BENCH_V2_SCHEMA_VERSION,
        seed=int(seed), pid=str(p.pid),
        arm_id="B_vlm_team_v2", role="vlm_verifier_final",
        call_idx=int(n_solver_turns) + 1, temperature=0.0,
        prompt_cid=hashlib.sha256(
            (verifier_prompt + "|img:" + p.image_sha256).encode(
                "utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            verifier_text.encode("utf-8")).hexdigest(),
        wall_ms=int(verifier_wall)))
    total += int(verifier_wall)
    candidates.append(verifier_candidate)
    exes.append(verifier_exe)

    # Selection: text-only PASS short-circuits (preserves W95-B0
    # rescues), then verifier-PASS rescues A1-only territory,
    # then verifier-FAIL is shipped as the best-effort last guess.
    if text_only_pass_idx >= 0:
        chosen_idx = text_only_pass_idx
        chosen_passed = True
        chosen_cand = candidates[text_only_pass_idx]
        chosen_rule = str(exes[text_only_pass_idx].matched_rule)
    elif verifier_exe.passed:
        chosen_idx = len(exes) - 1  # verifier index
        chosen_passed = True
        chosen_cand = verifier_candidate
        chosen_rule = str(verifier_exe.matched_rule)
    else:
        chosen_idx = len(exes) - 1  # verifier index (last guess)
        chosen_passed = False
        chosen_cand = verifier_candidate
        chosen_rule = str(verifier_exe.matched_rule)

    out = MathVistaArmOutcomeCapsuleV1(
        schema=W96_MATHVISTA_BENCH_V2_SCHEMA_VERSION,
        seed=int(seed), pid=str(p.pid),
        arm_id="B_vlm_team_v2",
        final_passed=bool(chosen_passed),
        final_prediction_cid=hashlib.sha256(
            chosen_cand.encode("utf-8")).hexdigest(),
        final_executor_rule=str(chosen_rule),
        n_model_calls=int(K),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cid=_executor_result_cid(
            exes[chosen_idx]))
    return out, exes


# ---------------------------------------------------------------
# Report dataclasses (V2 report carries the v2 B arm fields)
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MathVistaBenchReportV2:
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
    b_vlm_team_v2_mean_pass_at_1: float
    b_beats_a0_text_per_seed: tuple[bool, ...]
    b_beats_a1_vlm_per_seed: tuple[bool, ...]
    b_mean_strictly_beats_a0_text_mean: bool
    b_mean_strictly_beats_a1_vlm_mean: bool
    b_mean_minus_a0_text_mean_pp: float
    b_mean_minus_a1_vlm_mean_pp: float
    n_b_ge_a1_problems_per_seed: tuple[int, ...]
    # V2-specific: how often the verifier was the load-bearing
    # call (text-only fails → verifier PASS → ship verifier).
    n_verifier_rescues_per_seed: tuple[int, ...]
    n_text_only_passes_per_seed: tuple[int, ...]
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
            "b_vlm_team_v2_mean_pass_at_1": float(round(
                self.b_vlm_team_v2_mean_pass_at_1, 6)),
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
            "n_verifier_rescues_per_seed": list(
                self.n_verifier_rescues_per_seed),
            "n_text_only_passes_per_seed": list(
                self.n_text_only_passes_per_seed),
            "bench_merkle_root": str(self.bench_merkle_root),
        }


@dataclasses.dataclass
class MathVistaBenchConfigV2:
    schema: str = W96_MATHVISTA_BENCH_V2_SCHEMA_VERSION
    n_problems: int = 30
    K_multi_sample: int = 5
    seeds: tuple[int, ...] = (95_005_001,)
    sampling_temperature: float = 0.7
    max_tokens_per_call: int = 384


def run_mathvista_bench_v2(
        *,
        text_gen: _TextGenFn,
        vlm_gen: _VlmGenFn,
        vlm_model_id: str,
        text_model_id: str,
        corpus: Sequence[MathVistaProblemV1],
        corpus_parquet_sha256: str,
        corpus_merkle_root: str,
        config: MathVistaBenchConfigV2 | None = None,
        on_problem_start: (
            Callable[[int, int, str], None] | None) = None,
        sidecar_writer: (
            Callable[[dict[str, Any]], None] | None) = None,
) -> MathVistaBenchReportV2:
    """Run A0_text / A1_vlm / B_vlm_team_v2 on the supplied
    corpus under the W96-C V2 (C1) contract.  A0 and A1 are
    byte-identical to V1; B is the W96-C C1 VLM-Verifier-Final-
    Turn architecture.  Returns the V2 bench report with Merkle
    roots and verifier-rescue accounting.
    """
    from .mathvista_loader_v1 import select_mathvista_subset_v1

    cfg = config or MathVistaBenchConfigV2()
    per_seed: list[MathVistaSeedReportV1] = []
    all_outcome_cids: list[str] = []
    n_verifier_rescues_per_seed: list[int] = []
    n_text_only_passes_per_seed: list[int] = []
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
        b_text_only_passes: list[bool] = []
        b_verifier_rescues: list[bool] = []
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
            b_out, b_exes = _run_b_vlm_team_v2(
                seed=int(seed), p=p,
                K=int(cfg.K_multi_sample),
                temperature=float(cfg.sampling_temperature),
                vlm_gen=vlm_gen,
                text_gen=text_gen,
                max_tokens=int(cfg.max_tokens_per_call))
            b_outs.append(b_out)
            # Verifier-rescue accounting: text-only PASS = any of
            # the first 3 solver attempts passed; verifier-rescue
            # = text-only all FAIL AND verifier PASS.
            text_only_pass = any(
                exe.passed for exe in b_exes[:3])
            verifier_rescue = (
                (not text_only_pass)
                and b_exes[3].passed
                if len(b_exes) >= 4 else False)
            b_text_only_passes.append(text_only_pass)
            b_verifier_rescues.append(verifier_rescue)
            per_problem_outcomes.append({
                "pid": str(p.pid),
                "question_type": str(p.question_type),
                "answer_type": str(p.answer_type),
                "gold_answer": str(p.answer),
                "a0_text_passed": bool(a0_out.final_passed),
                "a1_vlm_passed": bool(a1_out.final_passed),
                "b_vlm_team_v2_passed": bool(b_out.final_passed),
                "b_text_only_passed": bool(text_only_pass),
                "b_verifier_rescued": bool(verifier_rescue),
                "a0_outcome_cid": str(a0_out.cid()),
                "a1_outcome_cid": str(a1_out.cid()),
                "b_outcome_cid": str(b_out.cid()),
            })
            if sidecar_writer is not None:
                sidecar_writer({
                    "kind": "w96c_mathvista_v2_per_problem_outcome",
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
            "kind": "w96c_mathvista_v2_seed_merkle_root_v1",
            "seed": int(seed),
            "outcome_cids": list(outcome_cids),
        })
        per_seed.append(MathVistaSeedReportV1(
            schema=W96_MATHVISTA_BENCH_V2_SCHEMA_VERSION,
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
        n_text_only_passes_per_seed.append(
            int(sum(1 for v in b_text_only_passes if v)))
        n_verifier_rescues_per_seed.append(
            int(sum(1 for v in b_verifier_rescues if v)))
    nseeds = float(len(per_seed))
    a0_mean = sum(
        s.a0_text_pass_at_1 for s in per_seed) / nseeds
    a1_mean = sum(
        s.a1_vlm_pass_at_1 for s in per_seed) / nseeds
    b_mean = sum(
        s.b_vlm_team_pass_at_1 for s in per_seed) / nseeds
    bench_merkle = _sha256_hex({
        "kind": "w96c_mathvista_v2_bench_merkle_root_v1",
        "vlm_model_id": str(vlm_model_id),
        "text_model_id": str(text_model_id),
        "corpus_parquet_sha256": str(corpus_parquet_sha256),
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
            if (bool(po["b_vlm_team_v2_passed"])
                    >= bool(po["a1_vlm_passed"])):
                n_bg += 1
        n_b_ge_a1_per_seed.append(int(n_bg))
    report = MathVistaBenchReportV2(
        schema=W96_MATHVISTA_BENCH_V2_SCHEMA_VERSION,
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
        b_vlm_team_v2_mean_pass_at_1=float(b_mean),
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
        n_verifier_rescues_per_seed=tuple(
            n_verifier_rescues_per_seed),
        n_text_only_passes_per_seed=tuple(
            n_text_only_passes_per_seed),
        bench_merkle_root=str(bench_merkle))
    return report


__all__ = [
    "W96_MATHVISTA_BENCH_V2_SCHEMA_VERSION",
    "MathVistaBenchReportV2",
    "MathVistaBenchConfigV2",
    "run_mathvista_bench_v2",
]
