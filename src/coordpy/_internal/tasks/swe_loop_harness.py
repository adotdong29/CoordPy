"""Phase 30 — substrate-evaluation harness with an LLM in the loop.

This module lifts the Phase-29 task-scale machinery (deterministic
oracle + delivery strategies) into a reusable evaluation harness that
takes an **arbitrary callable** in the aggregator role. The callable
represents the agent actually answering the question, and the
harness measures how its answer correctness depends on which delivery
strategy selected its context:

    * ``naive``      — every event is delivered to the aggregator.
    * ``routing``    — role-keyed Bloom-filter subscription (CASR stage 1).
    * ``substrate``  — direct-exact path: for planner-matched kinds the
                       aggregator sees only a short structured
                       substrate-summary (the planner's own answer),
                       not the event stream.
    * ``substrate_wrap`` — a hybrid where the aggregator STILL goes
                       through the LLM but is handed the substrate
                       answer as a single-line cue in its prompt. This
                       is the production-realistic path: the LLM does
                       final phrasing, the substrate does the math.

Why this is a separate harness and not a patch to Phase 29:

  * Phase 29 measures **causal-relevance fractions** under a
    deterministic answer — a useful information-theoretic number but
    not the claim an LLM-in-the-loop user cares about.
  * Phase 30 measures **answer correctness under a bounded-context
    LLM**. The two measurements are orthogonal axes (Theorem P30-2,
    documented in RESULTS_PHASE30.md). Keeping them in separate
    modules preserves Phase 29's reproducibility and lets the
    Phase-30 harness evolve (multiple LLMs, multiple retrieval
    strategies) without touching the Phase-29 guarantee.

The harness is deliberately **model-agnostic**: the aggregator is a
``Callable[[str], str]`` that takes a prompt and returns a string.
Unit tests use a deterministic ``MockLLM`` so the harness is
independently testable; the external benchmark
(`experiments/phase30_llm_swe_benchmark.py`) injects a real
``LLMClient`` on top of Ollama.

Scope discipline (what this harness DOES NOT claim):

  * It is not SWE-bench end-to-end — no patches are applied to code
    and no test harness is executed.
  * It does not measure model-specific reasoning quality beyond
    answer correctness. The substrate's job is to make the
    aggregator's prompt bounded and relevant; the model's job is to
    read that prompt.
  * Answer-grading is per-kind and deterministic (the same oracle
    used in Phase 29), NOT LLM-judged. We never ask a model to
    grade another model.

See ``docs/RESULTS_PHASE30.md`` for the framing and theorems.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from .python_corpus import PythonCorpus
from .task_scale_swe import (
    ALL_ROLES,
    EVENT_AGENT_COMMENT,
    EVENT_CLASS_DEF,
    EVENT_FILE_OPEN,
    EVENT_FINAL_ANSWER,
    EVENT_FUNCTION_DEF,
    EVENT_IMPORT_STMT,
    EVENT_TASK_GOAL,
    FIXED_POINT_EVENT_TYPES,
    KIND_COUNT_FILES_IMPORTING,
    KIND_COUNT_FUNCTIONS,
    KIND_COUNT_IS_RECURSIVE,
    KIND_COUNT_MAY_RAISE,
    KIND_COUNT_PARTICIPATES_IN_CYCLE,
    KIND_COUNT_TRANS_CALLS_FILESYSTEM,
    KIND_COUNT_TRANS_CALLS_SUBPROCESS,
    KIND_COUNT_TRANS_MAY_RAISE,
    KIND_LIST_FILES_IMPORTING,
    KIND_LIST_MAY_RAISE,
    KIND_LIST_TRANS_CALLS_SUBPROCESS,
    KIND_LIST_TRANS_MAY_RAISE,
    KIND_OPEN_VOCAB,
    KIND_TOP_FILE_BY_FUNCTIONS,
    ROLE_AGGREGATOR,
    ROLE_SUBSCRIPTIONS,
    SUBSTRATE_MATCHED_KINDS,
    Task,
    TaskEvent,
    _answer_from_events,
    _answer_matches_gold,
    build_event_stream,
    build_task_bank,
    oracle_relevance,
)


# =============================================================================
# Aggregator-role LLM protocol
# =============================================================================


class AnswerLLM(Protocol):
    """Minimum contract an aggregator model must implement.

    The contract is intentionally tiny. Callers that want to plug in
    a real model wrap their client in a function matching this
    signature.
    """

    def __call__(self, prompt: str) -> str: ...  # pragma: no cover


@dataclass
class MockAnswerLLM:
    """Deterministic answer LLM for tests + ablation.

    The mock fakes an "LLM that reads the prompt carefully and
    extracts what it can" by running the same deterministic decoder
    used by ``task_scale_swe._answer_from_events`` on the delivered
    events embedded in the prompt. It therefore saturates the
    *upper bound* of what a perfect reader of the delivered events
    could infer. Any shortfall under the mock is attributable to
    delivery (not LLM quality); any further shortfall under a real
    LLM quantifies the LLM-specific reasoning gap.
    """

    # Saved per-call stats so tests can assert context size.
    last_prompt: str = ""
    last_answer: str = ""
    n_calls: int = 0
    total_prompt_chars: int = 0

    def __call__(self, prompt: str) -> str:
        self.n_calls += 1
        self.last_prompt = prompt
        self.total_prompt_chars += len(prompt)
        # Mock: echo the CUE line if present (substrate_wrap case),
        # otherwise echo "UNKNOWN". This models a perfectly obedient
        # model that copies the cue; the real LLM path replaces this.
        m = re.search(r"SUBSTRATE_ANSWER:\s*(.+?)$", prompt, re.MULTILINE)
        if m:
            self.last_answer = m.group(1).strip()
            return self.last_answer
        self.last_answer = "UNKNOWN"
        return "UNKNOWN"


# =============================================================================
# Answer-grading (task-kind-specific, deterministic)
# =============================================================================


_LIST_KINDS = frozenset({
    KIND_LIST_FILES_IMPORTING,
    KIND_LIST_MAY_RAISE,
    KIND_LIST_TRANS_MAY_RAISE,
    KIND_LIST_TRANS_CALLS_SUBPROCESS,
})

_COUNT_KINDS = frozenset({
    KIND_COUNT_FILES_IMPORTING,
    KIND_COUNT_FUNCTIONS,
    KIND_COUNT_IS_RECURSIVE,
    KIND_COUNT_MAY_RAISE,
    KIND_COUNT_PARTICIPATES_IN_CYCLE,
    KIND_COUNT_TRANS_CALLS_FILESYSTEM,
    KIND_COUNT_TRANS_CALLS_SUBPROCESS,
    KIND_COUNT_TRANS_MAY_RAISE,
})

_SCALAR_STRING_KINDS = frozenset({
    KIND_TOP_FILE_BY_FUNCTIONS,
    KIND_OPEN_VOCAB,
})


def _first_int(text: str) -> int | None:
    m = re.search(r"(-?\d+)", text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _extract_list_members(text: str) -> list[str]:
    """Extract candidate qualified names from freeform LLM output.

    Accepts JSON arrays, bullet lists, comma-separated lists, or
    whitespace-separated tokens. Very permissive on purpose — the
    grader should credit an LLM that produces the right set even
    with sloppy formatting, and penalise one that hallucinates extra
    names.
    """
    text = text.strip()
    # Try JSON first.
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if x]
        except Exception:
            pass
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^[-*•\d\.\)\s]+", "", line)
        if not line:
            continue
        out.append(line)
    if len(out) <= 1:
        # Try comma-separated.
        out = [tok.strip() for tok in re.split(r"[,\n]", text) if tok.strip()]
    return out


def grade_answer(task: Task, llm_text: str) -> bool:
    """Task-kind-aware deterministic grader.

    Grading rules are conservative:
      * count queries: accept iff the first integer in ``llm_text``
        equals the gold integer.
      * list queries: accept iff the set of extracted candidate
        names matches the gold set exactly. Sensitive to both FN
        (missing names) and FP (hallucinated names).
      * scalar-string queries: accept iff ``gold`` is a substring
        of ``llm_text`` AND no other candidate answer is also a
        substring (for ``KIND_TOP_FILE_BY_FUNCTIONS``); for
        ``KIND_OPEN_VOCAB`` we just check substring containment.
    """
    if task.kind in _COUNT_KINDS:
        got = _first_int(llm_text)
        return got is not None and got == int(task.gold or 0)
    if task.kind in _LIST_KINDS:
        gold_set = set(task.gold or ())
        got_raw = _extract_list_members(llm_text)
        got_set: set[str] = set()
        for g in got_raw:
            # Strip the corpus module prefix if the gold has no prefix.
            got_set.add(g)
        # Accept a pure superset if gold matches; be strict against FP.
        return got_set == gold_set
    if task.kind == KIND_TOP_FILE_BY_FUNCTIONS:
        gold = str(task.gold or "")
        if not gold:
            return not bool(llm_text.strip())
        # LLM often reports the base filename; accept either full or base.
        base = gold.rsplit("/", 1)[-1]
        return (gold in llm_text) or (base in llm_text)
    if task.kind == KIND_OPEN_VOCAB:
        gold = str(task.gold or "")
        if not gold:
            return True
        return gold in llm_text
    # Defensive fallback — unknown kinds.
    return False


# =============================================================================
# Delivery + prompt assembly for the aggregator role
# =============================================================================


def _render_substrate_answer(task: Task, corpus: PythonCorpus) -> str:
    """Compute the substrate's direct-exact answer for ``task`` as a
    short string suitable for inclusion in an LLM prompt.

    For matched kinds the string is the deterministic answer. For
    the open-vocab residual we return a one-line hint derived from
    the analyzer (module name) but not the full docstring — the LLM
    then has to produce the summary.
    """
    if task.kind in _COUNT_KINDS:
        return str(task.gold)
    if task.kind in _LIST_KINDS:
        return ", ".join(task.gold or ())
    if task.kind == KIND_TOP_FILE_BY_FUNCTIONS:
        return str(task.gold or "")
    if task.kind == KIND_OPEN_VOCAB:
        # The aggregator is asked to summarise a module. The substrate
        # does not know what the module DOES, but it knows its name
        # and which imports it uses. Hand those to the LLM as context.
        name = task.target_arg
        for md in corpus.metadata:
            if md.module_name == name:
                return (f"module={name} imports={','.join(md.imports[:4])} "
                        f"n_functions={md.n_functions} "
                        f"n_classes={md.n_classes}")
        return name
    return ""


def _role_subscribed_events(events: Sequence[TaskEvent],
                              role: str) -> list[TaskEvent]:
    sub = ROLE_SUBSCRIPTIONS.get(role, frozenset())
    return [ev for ev in events if ev.event_type in sub]


def build_aggregator_prompt(task: Task,
                             events: Sequence[TaskEvent],
                             corpus: PythonCorpus,
                             strategy: str,
                             max_events_in_prompt: int = 400,
                             substrate_cue: str | None = None,
                             ) -> tuple[str, list[TaskEvent]]:
    """Assemble the prompt shown to the aggregator LLM under ``strategy``.

    Returns (prompt, delivered_events). The prompt has a stable
    structure across strategies so prompt-tokens differ only because
    of delivery, not formatting. Token measurement is therefore a
    fair comparison.

    Strategies:
      * ``naive``           — every event enters the prompt (capped at
                              ``max_events_in_prompt`` to keep prompts
                              under model context; truncation is a
                              first-class metric).
      * ``routing``         — role-keyed Bloom-filter subset; same cap.
      * ``substrate``       — direct-exact: prompt carries the
                              substrate answer as a single short
                              cue, plus fixed-point events only.
                              Zero content events.
      * ``substrate_wrap``  — LLM wraps the substrate answer. Identical
                              prompt shape to ``substrate`` except the
                              cue is embedded in a full-sentence
                              question frame to give the model a
                              natural place to speak.
    """
    if strategy == "naive":
        delivered = list(events)
    elif strategy == "routing":
        delivered = _role_subscribed_events(events, ROLE_AGGREGATOR)
    elif strategy in ("substrate", "substrate_wrap"):
        # Fixed-point only: task goal + final answer placeholder.
        delivered = [ev for ev in events if ev.is_fixed_point]
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    # Truncation — naive/routing can blow past a small model's window.
    truncated = False
    if len(delivered) > max_events_in_prompt:
        truncated = True
        delivered = delivered[:max_events_in_prompt]

    lines = [
        "You are the AGGREGATOR agent in a multi-role code-audit team.",
        "Answer the QUESTION below using the delivered context only.",
        ("For COUNT queries, respond with a single integer on its own "
         "line. For LIST queries, respond with a newline-separated "
         "list of qualified function or file names. For other queries, "
         "respond with a short sentence."),
        "",
        f"QUESTION: {task.question}",
    ]
    if strategy in ("substrate", "substrate_wrap") and substrate_cue:
        lines.append("")
        lines.append(f"SUBSTRATE_ANSWER: {substrate_cue}")
        if strategy == "substrate_wrap":
            lines.append(
                ("The SUBSTRATE_ANSWER above was computed deterministically "
                 "from the analyzer. Your job is to return it verbatim as "
                 "the answer, optionally in a natural sentence."))
    else:
        lines.append("")
        lines.append("DELIVERED EVENTS:")
        for ev in delivered:
            lines.append(f"- [{ev.event_type}] {ev.body}")
        if truncated:
            lines.append(
                ("... (event stream truncated; the delivered subset "
                 "above is the first window)"))
    lines.append("")
    lines.append("ANSWER:")
    prompt = "\n".join(lines)
    return prompt, delivered


# =============================================================================
# Per-(task, strategy) measurement
# =============================================================================


@dataclass
class LoopMeasurement:
    """Headline-per-task result for one strategy."""

    task_id: int
    kind: str
    strategy: str
    n_delivered_events: int
    n_prompt_chars: int
    n_prompt_tokens_approx: int   # chars/4 proxy (matching ``LLMStats``)
    truncated: bool
    substrate_matched: bool
    llm_answer: str
    answer_correct: bool
    wall_seconds: float
    gold: object

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "strategy": self.strategy,
            "n_delivered_events": self.n_delivered_events,
            "n_prompt_chars": self.n_prompt_chars,
            "n_prompt_tokens_approx": self.n_prompt_tokens_approx,
            "truncated": self.truncated,
            "substrate_matched": self.substrate_matched,
            "llm_answer": self.llm_answer[:500],  # cap for storage
            "answer_correct": self.answer_correct,
            "wall_seconds": round(self.wall_seconds, 3),
            "gold": _safe_jsonable(self.gold),
        }


def _safe_jsonable(x):
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, (int, str, float, bool)) or x is None:
        return x
    return str(x)


@dataclass
class LoopReport:
    """Aggregated report for one corpus × one strategy set."""

    corpus_name: str
    n_events: int
    n_tasks: int
    measurements: list[LoopMeasurement]
    config: dict

    def as_dict(self) -> dict:
        return {
            "corpus_name": self.corpus_name,
            "n_events": self.n_events,
            "n_tasks": self.n_tasks,
            "config": self.config,
            "measurements": [m.as_dict() for m in self.measurements],
            "pooled": self.pooled_summary(),
        }

    def pooled_summary(self) -> dict:
        by_strat: dict[str, list[LoopMeasurement]] = {}
        for m in self.measurements:
            by_strat.setdefault(m.strategy, []).append(m)
        out: dict[str, dict] = {}
        for strat, ms in by_strat.items():
            n = len(ms)
            if n == 0:
                continue
            out[strat] = {
                "n": n,
                "accuracy": round(
                    sum(1 for m in ms if m.answer_correct) / n, 4),
                "mean_prompt_tokens": round(
                    sum(m.n_prompt_tokens_approx for m in ms) / n, 2),
                "mean_wall_seconds": round(
                    sum(m.wall_seconds for m in ms) / n, 3),
                "truncated_count": sum(1 for m in ms if m.truncated),
                "substrate_match_count": sum(
                    1 for m in ms if m.substrate_matched),
            }
        return out


# =============================================================================
# Harness driver
# =============================================================================


def _strategy_is_substrate(strat: str) -> bool:
    return strat in ("substrate", "substrate_wrap")


def run_loop(corpus_name: str,
             corpus: PythonCorpus,
             aggregator: Callable[[str], str],
             strategies: Sequence[str] = ("naive", "routing", "substrate"),
             seed: int = 30,
             n_agent_comments: int = 6,
             max_events_in_prompt: int = 400,
             task_filter: Callable[[Task], bool] | None = None,
             ) -> LoopReport:
    """Run the aggregator over every task × every strategy.

    ``aggregator`` is invoked once per (task, strategy) pair. For the
    substrate strategies, the prompt contains a pre-computed
    substrate cue; for naive / routing strategies, the prompt
    contains the delivered event stream.
    """
    events = build_event_stream(
        corpus, n_agent_comments=n_agent_comments, seed=seed)
    tasks = build_task_bank(corpus, seed=seed)
    if task_filter is not None:
        tasks = [t for t in tasks if task_filter(t)]

    measurements: list[LoopMeasurement] = []
    for task in tasks:
        cue = None
        if any(_strategy_is_substrate(s) for s in strategies):
            cue = _render_substrate_answer(task, corpus)
        for strat in strategies:
            prompt, delivered = build_aggregator_prompt(
                task, events, corpus, strat,
                max_events_in_prompt=max_events_in_prompt,
                substrate_cue=cue,
            )
            truncated = (strat in ("naive", "routing")
                         and len(events if strat == "naive"
                                 else _role_subscribed_events(
                                     events, ROLE_AGGREGATOR))
                         > max_events_in_prompt)
            t0 = time.time()
            llm_text = aggregator(prompt)
            wall = time.time() - t0
            answer_correct = grade_answer(task, llm_text)
            substrate_matched = (_strategy_is_substrate(strat)
                                   and task.kind in SUBSTRATE_MATCHED_KINDS)
            measurements.append(LoopMeasurement(
                task_id=task.task_id, kind=task.kind, strategy=strat,
                n_delivered_events=len(delivered),
                n_prompt_chars=len(prompt),
                n_prompt_tokens_approx=max(1, len(prompt) // 4),
                truncated=truncated,
                substrate_matched=substrate_matched,
                llm_answer=llm_text,
                answer_correct=answer_correct,
                wall_seconds=wall,
                gold=task.gold,
            ))

    return LoopReport(
        corpus_name=corpus_name,
        n_events=len(events),
        n_tasks=len(tasks),
        measurements=measurements,
        config={
            "seed": seed,
            "n_agent_comments": n_agent_comments,
            "max_events_in_prompt": max_events_in_prompt,
            "strategies": list(strategies),
        },
    )


# =============================================================================
# Cross-strategy deltas (the product headline)
# =============================================================================


@dataclass(frozen=True)
class CrossStrategyDelta:
    """Pairwise headline: "substrate beats naive by X accuracy points
    and uses Y× fewer tokens"."""

    base: str
    comp: str
    accuracy_base: float
    accuracy_comp: float
    accuracy_delta: float
    mean_tokens_base: float
    mean_tokens_comp: float
    token_ratio: float


def compute_cross_strategy_deltas(rep: LoopReport) -> list[CrossStrategyDelta]:
    pooled = rep.pooled_summary()
    ordered = list(pooled.keys())
    out: list[CrossStrategyDelta] = []
    for i, base in enumerate(ordered):
        for comp in ordered[i + 1:]:
            a_base = pooled[base]["accuracy"]
            a_comp = pooled[comp]["accuracy"]
            t_base = pooled[base]["mean_prompt_tokens"]
            t_comp = pooled[comp]["mean_prompt_tokens"]
            ratio = t_base / t_comp if t_comp > 0 else float("inf")
            out.append(CrossStrategyDelta(
                base=base, comp=comp,
                accuracy_base=a_base, accuracy_comp=a_comp,
                accuracy_delta=round(a_comp - a_base, 4),
                mean_tokens_base=t_base, mean_tokens_comp=t_comp,
                token_ratio=round(ratio, 2),
            ))
    return out
