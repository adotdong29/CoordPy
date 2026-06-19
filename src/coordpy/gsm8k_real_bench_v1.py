"""W85 / P0 #28 — GSM8K real-task multi-agent bench.

Implements a real, content-addressed head-to-head bench on the
GSM8K test set from the canonical
``openai/grade-school-math`` GitHub repository. The bench drives
three arms under one model:

* ``A0`` — stock single-shot zero-shot CoT (the canonical
  GSM8K-paper baseline, ``Let's think step by step``). 1 call /
  problem at ``temperature=0``.
* ``A1`` — stock self-consistency at the same call-budget as B
  (K=5 independent samples at ``temperature=0.7`` + majority vote).
* ``B`` — CoordPy multi-agent pipeline at the same call-budget
  (K=5 calls = solver, alt-solver, critic, reviser, judge), then
  majority vote.

The arms run on the SAME problems / SAME seeds / SAME model.
The metric is the GSM8K published metric: exact-match accuracy
on the final numeric answer.

Anti-cheat
----------

* The GSM8K test corpus SHA-256 is verified against the canonical
  upstream commit; a corrupted or substituted corpus refuses to
  proceed.
* No problem in the eval subset is shown to the model with its
  answer — the answers are stripped before prompting.
* No arm is retried on failure. Each seed × arm × problem is
  exactly one set of calls. No selective rerunning.
* No model swap between arms.
* "Task success" is the GSM8K definition: the model's final
  numeric answer equals the gold answer (parsed from
  ``#### N``).
* Per-task / per-arm capsules are content-addressed; the bench
  chain has a Merkle root that is re-verifiable from disk by a
  third party without re-running the model.

Honest scope
------------

* ``W85-L-GSM8K-BENCH-V1-NIM-DEPENDENT-CAP`` — V1 drives the
  bench through any ``LLMBackend``-shaped client. Provider
  determinism beyond temperature=0 is not assumed.
* ``W85-L-GSM8K-BENCH-V1-NUMERIC-EXTRACTION-CAP`` — V1 extracts
  the final integer from the response using a conservative
  regex pattern. Edge cases (units like "$18", commas like
  "1,234") are normalised to integers in a documented way.
* ``W85-L-GSM8K-BENCH-V1-NETWORK-FETCH-CAP`` — V1 fetches the
  canonical corpus from GitHub raw on first use and caches by
  content-address; offline re-runs use the cache.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import re
import time
import urllib.request
from collections import Counter
from typing import Any, Callable, Mapping, Sequence


W85_GSM8K_REAL_BENCH_V1_SCHEMA_VERSION: str = (
    "coordpy.gsm8k_real_bench_v1.v1")


GSM8K_TEST_RAW_URL: str = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)


# SHA-256 of the canonical GSM8K test.jsonl file fetched
# 2026-05-19 from the upstream openai/grade-school-math repo at
# branch master. The bench refuses to proceed if a different
# corpus is loaded.
GSM8K_TEST_RAW_EXPECTED_SHA256: str = (
    "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
)


GSM8K_TEST_EXPECTED_PROBLEM_COUNT: int = 1319


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


class GSM8KCorpusError(RuntimeError):
    """Raised when the GSM8K corpus cannot be loaded or fails its
    SHA-256 check."""


# ---------------------------------------------------------------
# Corpus loader.
# ---------------------------------------------------------------


def _default_cache_path() -> str:
    cache = os.environ.get(
        "COORDPY_GSM8K_CACHE",
        os.path.expanduser("~/.cache/coordpy/gsm8k_test.jsonl"))
    return cache


def load_gsm8k_test_corpus_v1(
        *,
        cache_path: str | None = None,
        url: str = GSM8K_TEST_RAW_URL,
        expected_sha256: str = GSM8K_TEST_RAW_EXPECTED_SHA256,
        timeout: float = 60.0,
) -> tuple[tuple[str, str], ...]:
    """Load the canonical GSM8K test corpus, verifying SHA-256.

    Returns a tuple of ``(question, gold_answer_text)`` rows in
    the upstream file order. The gold answer is the raw answer
    string ending with ``#### N``. Use :func:`parse_gold_int_v1`
    to extract the integer.

    On first use, fetches the corpus from GitHub raw and caches.
    On later uses, reads the cache directly. Both paths verify
    SHA-256.
    """
    path = cache_path or _default_cache_path()
    if os.path.exists(path):
        with open(path, "rb") as f:
            raw = f.read()
    else:
        try:
            with urllib.request.urlopen(url, timeout=float(timeout)) as r:
                raw = r.read()
        except Exception as e:  # noqa: BLE001
            raise GSM8KCorpusError(
                f"GSM8K corpus fetch failed: {type(e).__name__}: {e} "
                f"(url={url})")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(raw)
    actual_sha = hashlib.sha256(raw).hexdigest()
    if str(actual_sha).lower() != str(expected_sha256).lower():
        raise GSM8KCorpusError(
            "GSM8K corpus SHA-256 mismatch: "
            f"actual={actual_sha} expected={expected_sha256}. "
            "Refusing to use a possibly-tampered corpus.")
    rows: list[tuple[str, str]] = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows.append((str(row["question"]), str(row["answer"])))
    if len(rows) != GSM8K_TEST_EXPECTED_PROBLEM_COUNT:
        raise GSM8KCorpusError(
            f"GSM8K corpus had {len(rows)} rows, "
            f"expected {GSM8K_TEST_EXPECTED_PROBLEM_COUNT}")
    return tuple(rows)


_GOLD_ANSWER_PATTERN = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def parse_gold_int_v1(answer_text: str) -> int | None:
    """Parse the final numeric answer from a GSM8K answer string.

    GSM8K formats the final answer as ``#### N`` (sometimes with
    commas). Returns ``None`` if no match. Floats are coerced to
    ``int`` only when they have zero fractional part — otherwise
    they are returned as their integer part (GSM8K answers are
    integers in practice).
    """
    m = _GOLD_ANSWER_PATTERN.search(str(answer_text))
    if m is None:
        return None
    token = m.group(1).replace(",", "")
    try:
        as_float = float(token)
        return int(round(as_float))
    except (ValueError, TypeError):
        return None


# Conservative numeric-extraction regex for model responses.
# Looks for the LAST integer/decimal in the text — GSM8K answers
# are always at the end of CoT in the literature. Allows commas
# and a leading minus sign.
_RESPONSE_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z])-?\d{1,3}(?:,\d{3})+|"
    r"(?<![A-Za-z\d.])-?\d+(?:\.\d+)?"
)


def parse_model_int_v1(response_text: str) -> int | None:
    """Extract the model's final integer answer from a response.

    The literature convention for GSM8K is "the last numeric
    expression in the model's response is the final answer". We
    use that convention here. Returns ``None`` if no numeric
    found.

    Edge cases:
    * ``$1,234`` → 1234.
    * ``18.0`` → 18.
    * ``18.5`` → 18 (rounded). GSM8K answers are integers, so a
      fractional answer is rounded for grading.
    * ``"the answer is 18"`` → 18.

    Anti-cheat: the bench does not look for "boxed{N}" or other
    benchmark-specific tokens because GSM8K's published metric is
    last-number extraction.
    """
    matches = _RESPONSE_NUMBER_PATTERN.findall(str(response_text))
    if not matches:
        return None
    token = str(matches[-1]).replace(",", "")
    try:
        as_float = float(token)
        return int(round(as_float))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------
# Per-call / per-problem capsules.
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GSM8KArmCallCapsuleV1:
    """One model call inside one arm on one problem."""

    schema: str
    seed: int
    problem_idx: int
    arm_id: str
    role: str
    call_idx: int
    temperature: float
    prompt_cid: str
    response_cid: str
    extracted_answer: int | None
    wall_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "seed": int(self.seed),
            "problem_idx": int(self.problem_idx),
            "arm_id": str(self.arm_id),
            "role": str(self.role),
            "call_idx": int(self.call_idx),
            "temperature": float(round(self.temperature, 6)),
            "prompt_cid": str(self.prompt_cid),
            "response_cid": str(self.response_cid),
            "extracted_answer": (
                None if self.extracted_answer is None
                else int(self.extracted_answer)),
            "wall_ms": int(self.wall_ms),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w85_gsm8k_arm_call_capsule_v1",
            "capsule": self.to_dict(),
        })


@dataclasses.dataclass(frozen=True)
class GSM8KArmOutcomeCapsuleV1:
    """The arm's final decision on one problem under one seed."""

    schema: str
    seed: int
    problem_idx: int
    arm_id: str
    gold_answer: int
    final_answer: int | None
    correct: bool
    n_calls: int
    total_wall_ms: int
    call_capsule_cids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "seed": int(self.seed),
            "problem_idx": int(self.problem_idx),
            "arm_id": str(self.arm_id),
            "gold_answer": int(self.gold_answer),
            "final_answer": (
                None if self.final_answer is None
                else int(self.final_answer)),
            "correct": bool(self.correct),
            "n_calls": int(self.n_calls),
            "total_wall_ms": int(self.total_wall_ms),
            "call_capsule_cids": list(self.call_capsule_cids),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w85_gsm8k_arm_outcome_capsule_v1",
            "outcome": self.to_dict(),
        })


@dataclasses.dataclass(frozen=True)
class GSM8KSeedReportV1:
    """All three arms' outcomes across a fixed problem set under
    one seed."""

    schema: str
    seed: int
    n_problems: int
    a0_accuracy: float
    a1_accuracy: float
    b_accuracy: float
    a0_total_wall_ms: int
    a1_total_wall_ms: int
    b_total_wall_ms: int
    outcome_cids: tuple[str, ...]
    seed_merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "seed": int(self.seed),
            "n_problems": int(self.n_problems),
            "a0_accuracy": float(round(self.a0_accuracy, 6)),
            "a1_accuracy": float(round(self.a1_accuracy, 6)),
            "b_accuracy": float(round(self.b_accuracy, 6)),
            "a0_total_wall_ms": int(self.a0_total_wall_ms),
            "a1_total_wall_ms": int(self.a1_total_wall_ms),
            "b_total_wall_ms": int(self.b_total_wall_ms),
            "outcome_cids_count": int(len(self.outcome_cids)),
            "seed_merkle_root": str(self.seed_merkle_root),
        }


@dataclasses.dataclass(frozen=True)
class GSM8KBenchReportV1:
    """The full bench report across seeds."""

    schema: str
    model_id: str
    n_problems: int
    n_seeds: int
    K_multi_sample: int
    per_seed: tuple[GSM8KSeedReportV1, ...]
    a0_mean_accuracy: float
    a1_mean_accuracy: float
    b_mean_accuracy: float
    b_beats_a0_per_seed: tuple[bool, ...]
    b_beats_a1_per_seed: tuple[bool, ...]
    b_strictly_beats_a0_on_all_seeds: bool
    b_strictly_beats_a1_on_all_seeds: bool
    b_mean_strictly_beats_a0_mean: bool
    b_mean_strictly_beats_a1_mean: bool
    bench_merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "model_id": str(self.model_id),
            "n_problems": int(self.n_problems),
            "n_seeds": int(self.n_seeds),
            "K_multi_sample": int(self.K_multi_sample),
            "per_seed": [s.to_dict() for s in self.per_seed],
            "a0_mean_accuracy": float(round(
                self.a0_mean_accuracy, 6)),
            "a1_mean_accuracy": float(round(
                self.a1_mean_accuracy, 6)),
            "b_mean_accuracy": float(round(
                self.b_mean_accuracy, 6)),
            "b_beats_a0_per_seed": list(self.b_beats_a0_per_seed),
            "b_beats_a1_per_seed": list(self.b_beats_a1_per_seed),
            "b_strictly_beats_a0_on_all_seeds": bool(
                self.b_strictly_beats_a0_on_all_seeds),
            "b_strictly_beats_a1_on_all_seeds": bool(
                self.b_strictly_beats_a1_on_all_seeds),
            "b_mean_strictly_beats_a0_mean": bool(
                self.b_mean_strictly_beats_a0_mean),
            "b_mean_strictly_beats_a1_mean": bool(
                self.b_mean_strictly_beats_a1_mean),
            "bench_merkle_root": str(self.bench_merkle_root),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w85_gsm8k_bench_report_v1",
            "report": self.to_dict(),
        })


# ---------------------------------------------------------------
# Prompt templates.
# ---------------------------------------------------------------


_COT_SYSTEM = (
    "You are a careful grade-school math assistant. Solve the "
    "problem step by step, then give the final integer answer "
    "on a new line as exactly: ANSWER: <integer>")


def _cot_prompt(question: str) -> str:
    return (
        f"{_COT_SYSTEM}\n\nQuestion: {question}\n\n"
        "Let's think step by step.")


def _multi_agent_solver_prompt(question: str, role_hint: str) -> str:
    return (
        f"{_COT_SYSTEM}\n\n[Persona: {role_hint}]\n"
        f"Question: {question}\n\nLet's think step by step.")


def _multi_agent_critic_prompt(question: str, solutions: Sequence[str]) -> str:
    bullet = "\n".join(
        f"--- Candidate {i+1} ---\n{s}"
        for i, s in enumerate(solutions))
    return (
        f"{_COT_SYSTEM}\n\n[Persona: rigorous critic]\n"
        f"Question: {question}\n\nCandidate solutions:\n{bullet}\n\n"
        "Identify the strongest reasoning chain, point out any "
        "arithmetic mistakes, then give your own final answer.")


def _multi_agent_reviser_prompt(
        question: str, solutions: Sequence[str], critic: str) -> str:
    bullet = "\n".join(
        f"--- Candidate {i+1} ---\n{s}"
        for i, s in enumerate(solutions))
    return (
        f"{_COT_SYSTEM}\n\n[Persona: meticulous reviser]\n"
        f"Question: {question}\n\nCandidate solutions:\n{bullet}\n\n"
        f"Critic feedback:\n{critic}\n\n"
        "Produce a corrected, clean step-by-step solution.")


def _multi_agent_judge_prompt(
        question: str, all_answers: Sequence[int | None],
        solutions: Sequence[str]) -> str:
    bullet = "\n".join(
        f"--- Candidate {i+1} (answer={a}) ---\n{s}"
        for i, (s, a) in enumerate(zip(solutions, all_answers)))
    return (
        f"{_COT_SYSTEM}\n\n[Persona: final judge]\n"
        f"Question: {question}\n\nProposed solutions and answers:\n{bullet}\n\n"
        "Pick the most consistent and arithmetically correct "
        "answer. State the answer on the final line as ANSWER: "
        "<integer>.")


def _multi_agent_personas() -> tuple[str, ...]:
    # Five distinct solver personas to drive diversity at K=5.
    return (
        "algebraic — translate to equations first, then solve",
        "concrete — work with named units (apples, cars, dollars)",
        "step-by-step elementary teacher",
        "rapid mental-arithmetic prodigy",
        "verifier — solve once, then re-check by substitution",
    )


# ---------------------------------------------------------------
# Per-arm runners.
# ---------------------------------------------------------------


_GenerateFn = Callable[[str, int, float], tuple[str, int]]
# (prompt, max_tokens, temperature) -> (response_text, wall_ms)


def _run_a0_single_shot(
        *,
        seed: int,
        problem_idx: int,
        question: str,
        gold: int,
        gen: _GenerateFn,
        max_tokens: int,
) -> GSM8KArmOutcomeCapsuleV1:
    prompt = _cot_prompt(question)
    text, wall = gen(prompt, max_tokens, 0.0)
    extracted = parse_model_int_v1(text)
    call = GSM8KArmCallCapsuleV1(
        schema=W85_GSM8K_REAL_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed),
        problem_idx=int(problem_idx),
        arm_id="A0",
        role="cot",
        call_idx=0,
        temperature=0.0,
        prompt_cid=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        extracted_answer=extracted,
        wall_ms=int(wall),
    )
    correct = (extracted is not None and int(extracted) == int(gold))
    return GSM8KArmOutcomeCapsuleV1(
        schema=W85_GSM8K_REAL_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed),
        problem_idx=int(problem_idx),
        arm_id="A0",
        gold_answer=int(gold),
        final_answer=extracted,
        correct=bool(correct),
        n_calls=1,
        total_wall_ms=int(wall),
        call_capsule_cids=(call.cid(),),
    )


def _majority_vote(answers: Sequence[int | None]) -> int | None:
    """Majority vote that ignores ``None``s. Ties broken by the
    smaller integer (deterministic)."""
    valid = [int(a) for a in answers if a is not None]
    if not valid:
        return None
    c = Counter(valid)
    top_count = max(c.values())
    candidates = sorted(a for a, n in c.items() if n == top_count)
    return int(candidates[0])


def _run_a1_self_consistency(
        *,
        seed: int,
        problem_idx: int,
        question: str,
        gold: int,
        K: int,
        temperature: float,
        gen: _GenerateFn,
        max_tokens: int,
) -> GSM8KArmOutcomeCapsuleV1:
    prompt = _cot_prompt(question)
    calls: list[GSM8KArmCallCapsuleV1] = []
    answers: list[int | None] = []
    total = 0
    for k in range(K):
        text, wall = gen(prompt, max_tokens, float(temperature))
        ext = parse_model_int_v1(text)
        calls.append(GSM8KArmCallCapsuleV1(
            schema=W85_GSM8K_REAL_BENCH_V1_SCHEMA_VERSION,
            seed=int(seed),
            problem_idx=int(problem_idx),
            arm_id="A1",
            role="sc_sample",
            call_idx=int(k),
            temperature=float(temperature),
            prompt_cid=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            response_cid=hashlib.sha256(
                text.encode("utf-8")).hexdigest(),
            extracted_answer=ext,
            wall_ms=int(wall),
        ))
        answers.append(ext)
        total += int(wall)
    voted = _majority_vote(answers)
    correct = (voted is not None and int(voted) == int(gold))
    return GSM8KArmOutcomeCapsuleV1(
        schema=W85_GSM8K_REAL_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed),
        problem_idx=int(problem_idx),
        arm_id="A1",
        gold_answer=int(gold),
        final_answer=voted,
        correct=bool(correct),
        n_calls=int(K),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
    )


def _run_b_coordpy_multi_agent(
        *,
        seed: int,
        problem_idx: int,
        question: str,
        gold: int,
        K: int,
        temperature: float,
        gen: _GenerateFn,
        max_tokens: int,
) -> GSM8KArmOutcomeCapsuleV1:
    """CoordPy 5-call multi-agent pipeline (K must be 5)."""
    if int(K) != 5:
        raise ValueError(
            f"arm B requires K=5, got {K}; the pipeline shape is "
            f"solver1, solver2, critic, reviser, judge")
    personas = _multi_agent_personas()
    calls: list[GSM8KArmCallCapsuleV1] = []
    answers: list[int | None] = []
    solutions: list[str] = []
    total = 0
    # Call 0: solver persona 1
    p0 = _multi_agent_solver_prompt(question, personas[0])
    t0_text, t0_wall = gen(p0, max_tokens, float(temperature))
    a0 = parse_model_int_v1(t0_text)
    calls.append(GSM8KArmCallCapsuleV1(
        schema=W85_GSM8K_REAL_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), problem_idx=int(problem_idx),
        arm_id="B", role="solver_1", call_idx=0,
        temperature=float(temperature),
        prompt_cid=hashlib.sha256(p0.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(t0_text.encode("utf-8")).hexdigest(),
        extracted_answer=a0, wall_ms=int(t0_wall)))
    answers.append(a0); solutions.append(t0_text); total += int(t0_wall)
    # Call 1: solver persona 2
    p1 = _multi_agent_solver_prompt(question, personas[1])
    t1_text, t1_wall = gen(p1, max_tokens, float(temperature))
    a1 = parse_model_int_v1(t1_text)
    calls.append(GSM8KArmCallCapsuleV1(
        schema=W85_GSM8K_REAL_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), problem_idx=int(problem_idx),
        arm_id="B", role="solver_2", call_idx=1,
        temperature=float(temperature),
        prompt_cid=hashlib.sha256(p1.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(t1_text.encode("utf-8")).hexdigest(),
        extracted_answer=a1, wall_ms=int(t1_wall)))
    answers.append(a1); solutions.append(t1_text); total += int(t1_wall)
    # Call 2: critic on solutions so far
    p2 = _multi_agent_critic_prompt(question, solutions)
    t2_text, t2_wall = gen(p2, max_tokens, float(temperature))
    a2 = parse_model_int_v1(t2_text)
    calls.append(GSM8KArmCallCapsuleV1(
        schema=W85_GSM8K_REAL_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), problem_idx=int(problem_idx),
        arm_id="B", role="critic", call_idx=2,
        temperature=float(temperature),
        prompt_cid=hashlib.sha256(p2.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(t2_text.encode("utf-8")).hexdigest(),
        extracted_answer=a2, wall_ms=int(t2_wall)))
    answers.append(a2); total += int(t2_wall)
    # Call 3: reviser using both solver outputs + critic
    p3 = _multi_agent_reviser_prompt(question, solutions, t2_text)
    t3_text, t3_wall = gen(p3, max_tokens, float(temperature))
    a3 = parse_model_int_v1(t3_text)
    calls.append(GSM8KArmCallCapsuleV1(
        schema=W85_GSM8K_REAL_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), problem_idx=int(problem_idx),
        arm_id="B", role="reviser", call_idx=3,
        temperature=float(temperature),
        prompt_cid=hashlib.sha256(p3.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(t3_text.encode("utf-8")).hexdigest(),
        extracted_answer=a3, wall_ms=int(t3_wall)))
    answers.append(a3); solutions.append(t3_text); total += int(t3_wall)
    # Call 4: judge gives final answer at t=0 (deterministic)
    p4 = _multi_agent_judge_prompt(
        question, answers + [a3], solutions + [t2_text])
    t4_text, t4_wall = gen(p4, max_tokens, 0.0)
    a4 = parse_model_int_v1(t4_text)
    calls.append(GSM8KArmCallCapsuleV1(
        schema=W85_GSM8K_REAL_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), problem_idx=int(problem_idx),
        arm_id="B", role="judge", call_idx=4,
        temperature=0.0,
        prompt_cid=hashlib.sha256(p4.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(t4_text.encode("utf-8")).hexdigest(),
        extracted_answer=a4, wall_ms=int(t4_wall)))
    total += int(t4_wall)
    # Final decision: prefer judge if confident, else majority vote.
    valid_solvers = [a for a in (a0, a1, a3) if a is not None]
    voted = _majority_vote(answers + [a4])
    # Use judge's answer if present; otherwise voted majority.
    if a4 is not None:
        # If judge agrees with at least one solver, take judge.
        # If judge disagrees with all solvers, fall back to vote
        # over (solver1, solver2, reviser, judge) — this is the
        # coordinated outcome.
        final_answer = a4 if a4 in valid_solvers else voted
    else:
        final_answer = voted
    correct = (final_answer is not None
               and int(final_answer) == int(gold))
    return GSM8KArmOutcomeCapsuleV1(
        schema=W85_GSM8K_REAL_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), problem_idx=int(problem_idx),
        arm_id="B",
        gold_answer=int(gold),
        final_answer=final_answer,
        correct=bool(correct),
        n_calls=5, total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
    )


# ---------------------------------------------------------------
# Problem-subset selection.
# ---------------------------------------------------------------


def select_problem_subset_v1(
        *,
        corpus: Sequence[tuple[str, str]],
        n_problems: int,
        seed: int,
) -> tuple[tuple[int, str, int], ...]:
    """Deterministically select ``n_problems`` from the corpus.

    Returns ``((problem_idx_in_corpus, question, gold_int), ...)``
    drawn without replacement. Problems whose gold answer fails to
    parse are skipped honestly (do not silently degrade size).
    """
    rng = random.Random(int(seed))
    indices = list(range(len(corpus)))
    rng.shuffle(indices)
    out: list[tuple[int, str, int]] = []
    for idx in indices:
        if len(out) >= int(n_problems):
            break
        q, gold_text = corpus[idx]
        g = parse_gold_int_v1(gold_text)
        if g is None:
            continue
        out.append((int(idx), str(q), int(g)))
    if len(out) < int(n_problems):
        raise GSM8KCorpusError(
            f"could not assemble {n_problems} parsable problems "
            f"from a corpus of {len(corpus)}; got {len(out)}")
    return tuple(out)


# ---------------------------------------------------------------
# Bench driver.
# ---------------------------------------------------------------


@dataclasses.dataclass
class GSM8KBenchConfigV1:
    schema: str = W85_GSM8K_REAL_BENCH_V1_SCHEMA_VERSION
    n_problems: int = 30
    K_multi_sample: int = 5
    seeds: tuple[int, ...] = (85_001, 85_002, 85_003)
    sampling_temperature: float = 0.7
    max_tokens_per_call: int = 512


def run_gsm8k_real_bench_v1(
        *,
        gen: _GenerateFn,
        model_id: str,
        corpus: Sequence[tuple[str, str]],
        config: GSM8KBenchConfigV1 | None = None,
        on_problem_start: Callable[[int, int, int], None] | None = None,
) -> GSM8KBenchReportV1:
    """Run the three arms on the GSM8K subset for each seed.

    ``gen(prompt, max_tokens, temperature) -> (text, wall_ms)`` is
    the call-level abstraction; pass any backend wrapper.

    Returns a content-addressed bench report.
    """
    cfg = config or GSM8KBenchConfigV1()
    per_seed: list[GSM8KSeedReportV1] = []
    all_outcome_cids: list[str] = []
    for seed in cfg.seeds:
        subset = select_problem_subset_v1(
            corpus=corpus, n_problems=int(cfg.n_problems),
            seed=int(seed))
        a0_outs: list[GSM8KArmOutcomeCapsuleV1] = []
        a1_outs: list[GSM8KArmOutcomeCapsuleV1] = []
        b_outs: list[GSM8KArmOutcomeCapsuleV1] = []
        for (p_idx, (orig_idx, q, gold)) in enumerate(subset):
            if on_problem_start is not None:
                on_problem_start(int(seed), int(p_idx), int(orig_idx))
            a0 = _run_a0_single_shot(
                seed=int(seed), problem_idx=int(orig_idx),
                question=q, gold=int(gold), gen=gen,
                max_tokens=int(cfg.max_tokens_per_call))
            a0_outs.append(a0)
            a1 = _run_a1_self_consistency(
                seed=int(seed), problem_idx=int(orig_idx),
                question=q, gold=int(gold),
                K=int(cfg.K_multi_sample),
                temperature=float(cfg.sampling_temperature),
                gen=gen, max_tokens=int(cfg.max_tokens_per_call))
            a1_outs.append(a1)
            b = _run_b_coordpy_multi_agent(
                seed=int(seed), problem_idx=int(orig_idx),
                question=q, gold=int(gold),
                K=int(cfg.K_multi_sample),
                temperature=float(cfg.sampling_temperature),
                gen=gen, max_tokens=int(cfg.max_tokens_per_call))
            b_outs.append(b)
        a0_acc = sum(1 for o in a0_outs if o.correct) / float(len(a0_outs))
        a1_acc = sum(1 for o in a1_outs if o.correct) / float(len(a1_outs))
        b_acc = sum(1 for o in b_outs if o.correct) / float(len(b_outs))
        a0_wall = sum(int(o.total_wall_ms) for o in a0_outs)
        a1_wall = sum(int(o.total_wall_ms) for o in a1_outs)
        b_wall = sum(int(o.total_wall_ms) for o in b_outs)
        outcome_cids = tuple(
            [o.cid() for o in a0_outs]
            + [o.cid() for o in a1_outs]
            + [o.cid() for o in b_outs])
        seed_merkle = _sha256_hex({
            "kind": "w85_gsm8k_seed_merkle_root",
            "seed": int(seed),
            "outcome_cids": list(outcome_cids),
        })
        per_seed.append(GSM8KSeedReportV1(
            schema=W85_GSM8K_REAL_BENCH_V1_SCHEMA_VERSION,
            seed=int(seed),
            n_problems=int(len(a0_outs)),
            a0_accuracy=float(a0_acc),
            a1_accuracy=float(a1_acc),
            b_accuracy=float(b_acc),
            a0_total_wall_ms=int(a0_wall),
            a1_total_wall_ms=int(a1_wall),
            b_total_wall_ms=int(b_wall),
            outcome_cids=outcome_cids,
            seed_merkle_root=str(seed_merkle),
        ))
        all_outcome_cids.extend(outcome_cids)
    a0_mean = sum(s.a0_accuracy for s in per_seed) / float(len(per_seed))
    a1_mean = sum(s.a1_accuracy for s in per_seed) / float(len(per_seed))
    b_mean = sum(s.b_accuracy for s in per_seed) / float(len(per_seed))
    b_beats_a0 = tuple(s.b_accuracy > s.a0_accuracy for s in per_seed)
    b_beats_a1 = tuple(s.b_accuracy > s.a1_accuracy for s in per_seed)
    bench_merkle = _sha256_hex({
        "kind": "w85_gsm8k_bench_merkle_root",
        "model_id": str(model_id),
        "outcome_cids": list(all_outcome_cids),
        "seeds": list(cfg.seeds),
    })
    return GSM8KBenchReportV1(
        schema=W85_GSM8K_REAL_BENCH_V1_SCHEMA_VERSION,
        model_id=str(model_id),
        n_problems=int(cfg.n_problems),
        n_seeds=int(len(cfg.seeds)),
        K_multi_sample=int(cfg.K_multi_sample),
        per_seed=tuple(per_seed),
        a0_mean_accuracy=float(a0_mean),
        a1_mean_accuracy=float(a1_mean),
        b_mean_accuracy=float(b_mean),
        b_beats_a0_per_seed=b_beats_a0,
        b_beats_a1_per_seed=b_beats_a1,
        b_strictly_beats_a0_on_all_seeds=bool(all(b_beats_a0)),
        b_strictly_beats_a1_on_all_seeds=bool(all(b_beats_a1)),
        b_mean_strictly_beats_a0_mean=bool(b_mean > a0_mean),
        b_mean_strictly_beats_a1_mean=bool(b_mean > a1_mean),
        bench_merkle_root=str(bench_merkle),
    )


__all__ = [
    "W85_GSM8K_REAL_BENCH_V1_SCHEMA_VERSION",
    "GSM8K_TEST_RAW_URL",
    "GSM8K_TEST_RAW_EXPECTED_SHA256",
    "GSM8K_TEST_EXPECTED_PROBLEM_COUNT",
    "GSM8KCorpusError",
    "GSM8KArmCallCapsuleV1",
    "GSM8KArmOutcomeCapsuleV1",
    "GSM8KSeedReportV1",
    "GSM8KBenchReportV1",
    "GSM8KBenchConfigV1",
    "load_gsm8k_test_corpus_v1",
    "parse_gold_int_v1",
    "parse_model_int_v1",
    "select_problem_subset_v1",
    "run_gsm8k_real_bench_v1",
]
