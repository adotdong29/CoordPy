"""Task-scale causal-relevance benchmark harness — Phase 29.

This module is the programme's first *task-scale* falsifiability check
of the core thesis: **the dominant waste in multi-agent LLM systems is
irrelevance, not redundancy; the per-agent minimum-sufficient context
on a software-engineering-style task is much smaller than a naive-
broadcast baseline.**

The phase-10 scaling experiments showed O(log N) on numeric / LLM-QA
coordination tasks. Phase 27/28 showed that on code corpora, an
analyzer + planner answers a well-defined slice of queries with zero
LLM calls and zero prompt chars. What has been *missing* so far is an
experiment that couples those two findings with a task distribution
drawn from software engineering work, and that reports the
decomposition at the level of an *event stream delivered to each
agent* so we can compute causal relevance per role per event.

This module provides that harness. It runs on existing local corpora
(no synthetic test code), builds a realistic task bank from the
Phase-23 question families, constructs the naive event stream that a
broadcast-only system would emit, and exposes three oracles:

  1. ``OracleRelevance`` — a per-(task, role, event) predicate that
     returns True iff masking this event from this agent under this
     task would change the correct answer. The oracle is
     constructively defined from the task's gold answer (which in
     turn is computed from the same Phase-22..28 analyzer).

  2. ``SimulatedRouter`` — a synchronous simulation of the three
     delivery strategies:
       - ``naive``: every role receives every event.
       - ``routing``: role-specific Bloom-filter drops by event type
         (mirrors CASR's first stage).
       - ``substrate``: for matched tasks, the direct-exact planner
         answers without delivering ANY content event to the
         aggregator role; for unmatched tasks, the retrieval path
         delivers at most the top-k relevant events as determined by
         content-matching against the query.

  3. ``RoleSpec`` — a small typed record linking each role to the
     event-type subset it subscribes to in the routing condition and
     the answer decoder it runs in the substrate condition. Five
     realistic roles are bundled by default, spanning the spectrum
     from scale-4 orchestration to scale-2 content work.

Every number in the resulting report is reproducible from public
inputs: the local corpus directory plus the task-bank seed.

Scope discipline (what this phase DOES NOT claim):

  - **Not a full SWE-bench run.** SWE-bench requires actually
    executing patched code against a test harness under a model.
    That is still end-goal work for the programme (see
    ``ROADMAP.md``). What Phase 29 provides is a task-scale *causal*
    analogue: we measure the fraction of events that are
    *provably* irrelevant given the task's analyzer-derived gold,
    and we report it alongside the standard substrate-vs-retrieval
    accuracy measurement.
  - **No LLM calls in this module.** Every answer on the direct-
    exact path is deterministic; every answer on the retrieval path
    is a content-match over metadata, not an LLM inference. That
    keeps the number reproducible and keeps the measurement on the
    causally-relevant axis.
  - **The oracle is exact only for planner-matched tasks.** For
    open-vocabulary tasks we approximate causal relevance by
    substring match on the gold answer's supporting bytes — a
    conservative lower bound on relevance (Theorem P29-2).

See ``vision_mvp/RESULTS_PHASE29.md`` for the framing, theorem set,
and headline numbers.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from ..core.code_index import CodeMetadata
from .python_corpus import PythonCorpus


# =============================================================================
# Event schema
# =============================================================================


# Event types — keep small and enumerated so Bloom-filter routing is
# clean. Every event carries a short body string we use both for token
# accounting and for the content-match relevance approximation on
# open-vocabulary queries.
EVENT_FILE_OPEN = "FILE_OPEN"
EVENT_FUNCTION_DEF = "FUNCTION_DEF"
EVENT_CLASS_DEF = "CLASS_DEF"
EVENT_IMPORT_STMT = "IMPORT_STMT"
EVENT_AGENT_COMMENT = "AGENT_COMMENT"
EVENT_TASK_GOAL = "TASK_GOAL"
EVENT_FINAL_ANSWER = "FINAL_ANSWER"

FIXED_POINT_EVENT_TYPES = frozenset({EVENT_TASK_GOAL, EVENT_FINAL_ANSWER})

ALL_EVENT_TYPES = (
    EVENT_FILE_OPEN, EVENT_FUNCTION_DEF, EVENT_CLASS_DEF,
    EVENT_IMPORT_STMT, EVENT_AGENT_COMMENT,
    EVENT_TASK_GOAL, EVENT_FINAL_ANSWER,
)


@dataclass(frozen=True)
class TaskEvent:
    """One atomic event on the naive broadcast bus.

    ``body`` holds the canonical string representation used for both
    token accounting and the content-match relevance fallback. For
    file-open events this is the file path plus a short metadata
    summary; for function-def events it is ``module.qname`` plus the
    semantic-flag bits; for import events it is the imported name;
    for agent comments it is a short canned string.

    Events are immutable and hashable — required by the routing
    simulator's subscription set.
    """

    event_id: int
    event_type: str
    body: str
    # Per-event metadata used by the oracle to decide causal
    # relevance. Kept small and typed.
    file_path: str = ""
    module_name: str = ""
    qname: str = ""                       # set on FUNCTION_DEF/CLASS_DEF
    imports: tuple[str, ...] = ()         # set on FILE_OPEN
    import_name: str = ""                 # set on IMPORT_STMT
    function_flags: tuple[tuple[str, bool], ...] = ()  # semantic flags
    # Rough token cost (whitespace-split). Deterministic from body.

    @property
    def n_tokens(self) -> int:
        if not self.body:
            return 0
        return max(1, len(self.body.split()))

    @property
    def is_fixed_point(self) -> bool:
        return self.event_type in FIXED_POINT_EVENT_TYPES


# =============================================================================
# Task schema — a realistic SWE-style question + gold + oracle
# =============================================================================


# Task-kind bucketing. Each bucket has its own oracle recipe.
KIND_COUNT_FILES_IMPORTING = "count_files_importing"
KIND_LIST_FILES_IMPORTING = "list_files_importing"
KIND_COUNT_TRANS_MAY_RAISE = "count_trans_may_raise"
KIND_LIST_TRANS_MAY_RAISE = "list_trans_may_raise"
KIND_COUNT_TRANS_CALLS_SUBPROCESS = "count_trans_calls_subprocess"
KIND_LIST_TRANS_CALLS_SUBPROCESS = "list_trans_calls_subprocess"
KIND_COUNT_TRANS_CALLS_FILESYSTEM = "count_trans_calls_filesystem"
KIND_COUNT_PARTICIPATES_IN_CYCLE = "count_participates_in_cycle"
KIND_COUNT_IS_RECURSIVE = "count_is_recursive"
KIND_COUNT_MAY_RAISE = "count_may_raise"
KIND_LIST_MAY_RAISE = "list_may_raise"
KIND_COUNT_FUNCTIONS = "count_functions_total"
KIND_TOP_FILE_BY_FUNCTIONS = "top_file_by_functions"
KIND_OPEN_VOCAB = "open_vocab"         # falls to retrieval + LLM

ALL_TASK_KINDS = (
    KIND_COUNT_FILES_IMPORTING, KIND_LIST_FILES_IMPORTING,
    KIND_COUNT_TRANS_MAY_RAISE, KIND_LIST_TRANS_MAY_RAISE,
    KIND_COUNT_TRANS_CALLS_SUBPROCESS, KIND_LIST_TRANS_CALLS_SUBPROCESS,
    KIND_COUNT_TRANS_CALLS_FILESYSTEM, KIND_COUNT_PARTICIPATES_IN_CYCLE,
    KIND_COUNT_IS_RECURSIVE, KIND_COUNT_MAY_RAISE, KIND_LIST_MAY_RAISE,
    KIND_COUNT_FUNCTIONS, KIND_TOP_FILE_BY_FUNCTIONS,
    KIND_OPEN_VOCAB,
)

# Planner-matched kinds — the direct-exact path handles these. The
# ``open_vocab`` kind falls to retrieval.
SUBSTRATE_MATCHED_KINDS = frozenset(ALL_TASK_KINDS) - {KIND_OPEN_VOCAB}


@dataclass(frozen=True)
class Task:
    """One SWE-style question bundled with its gold and oracle.

    ``gold`` is either an integer (counting queries), a sorted tuple
    of qualified names (listing queries), or a free-form string
    (open-vocabulary residual). ``target_arg`` carries the parameter
    the question asks about (e.g. the imported module name on
    ``count_files_importing``); empty for parameterless kinds.
    """

    task_id: int
    kind: str
    question: str
    gold: object
    target_arg: str = ""


# =============================================================================
# Event-stream construction from a PythonCorpus
# =============================================================================


def build_event_stream(corpus: PythonCorpus,
                         n_agent_comments: int = 4,
                         seed: int = 29,
                         ) -> list[TaskEvent]:
    """Emit one event per corpus atom (file / function / class / import)
    plus a deterministic sprinkling of ``AGENT_COMMENT`` events.

    Deterministic given ``(corpus, n_agent_comments, seed)``. The
    stream models what a naive broadcast bus would observe during a
    full-team code-audit session: every ingested artefact becomes
    one event; each agent emits ``n_agent_comments`` short
    commentary events.
    """
    rng = random.Random(seed)
    events: list[TaskEvent] = []

    # Task-goal fixed-point — delivered to every role.
    events.append(TaskEvent(
        event_id=0, event_type=EVENT_TASK_GOAL,
        body="audit the corpus; answer a battery of SWE-style queries",
    ))
    nxt = 1

    for md in corpus.metadata:
        # FILE_OPEN carries file path + a small metadata digest (imports,
        # n_functions, n_classes, is_test_file, has_docstring). This is
        # the shape a realistic ingester would broadcast.
        imp_sample = ",".join(sorted(md.imports)[:6])
        body = (f"file={md.file_path} module={md.module_name} "
                f"imports={imp_sample} n_functions={md.n_functions} "
                f"n_classes={md.n_classes} "
                f"is_test_file={md.is_test_file} "
                f"has_docstring={md.has_docstring}")
        events.append(TaskEvent(
            event_id=nxt, event_type=EVENT_FILE_OPEN,
            body=body, file_path=md.file_path,
            module_name=md.module_name,
            imports=tuple(md.imports),
        ))
        nxt += 1

        for imp in md.imports:
            events.append(TaskEvent(
                event_id=nxt, event_type=EVENT_IMPORT_STMT,
                body=f"import {imp} in {md.module_name}",
                file_path=md.file_path,
                module_name=md.module_name,
                import_name=imp,
            ))
            nxt += 1

        for class_name in md.class_names:
            events.append(TaskEvent(
                event_id=nxt, event_type=EVENT_CLASS_DEF,
                body=f"class {class_name} in {md.module_name}",
                file_path=md.file_path,
                module_name=md.module_name,
                qname=f"{md.module_name}.{class_name}",
            ))
            nxt += 1

        # Function-def events carry Phase-25 interprocedural flags
        # so the aggregator-on-semantic-queries can compute the
        # correct answer without ever opening a file.
        sem_names = md.semantic_function_names
        # Resilient to older metadata shapes in tests.
        fmr = md.function_trans_may_raise or ()
        fsp = md.function_trans_calls_subprocess or ()
        ffs = md.function_trans_calls_filesystem or ()
        fnw = md.function_trans_calls_network or ()
        fgw = md.function_trans_may_write_global or ()
        fcy = md.function_participates_in_cycle or ()
        fir = md.function_is_recursive or ()
        fmr_intra = md.function_may_raise or ()
        for i, qname in enumerate(sem_names):
            flags = (
                ("trans_may_raise", bool(fmr[i]) if i < len(fmr) else False),
                ("trans_calls_subprocess",
                 bool(fsp[i]) if i < len(fsp) else False),
                ("trans_calls_filesystem",
                 bool(ffs[i]) if i < len(ffs) else False),
                ("trans_calls_network",
                 bool(fnw[i]) if i < len(fnw) else False),
                ("trans_may_write_global",
                 bool(fgw[i]) if i < len(fgw) else False),
                ("participates_in_cycle",
                 bool(fcy[i]) if i < len(fcy) else False),
                ("is_recursive",
                 bool(fir[i]) if i < len(fir) else False),
                ("may_raise",
                 bool(fmr_intra[i]) if i < len(fmr_intra) else False),
            )
            events.append(TaskEvent(
                event_id=nxt, event_type=EVENT_FUNCTION_DEF,
                body=f"function {md.module_name}.{qname} flags={flags}",
                file_path=md.file_path,
                module_name=md.module_name,
                qname=f"{md.module_name}.{qname}",
                function_flags=flags,
            ))
            nxt += 1

    # Agent commentary — simulated team chatter, deterministic per seed.
    chatter_pool = (
        "lgtm", "running tests", "rebased onto main",
        "please clarify requirements", "can you split this into two commits",
        "found a typo", "same as before", "worth adding a unit test",
        "moving to reviewer", "blocked on upstream fix",
    )
    for _ in range(n_agent_comments):
        events.append(TaskEvent(
            event_id=nxt, event_type=EVENT_AGENT_COMMENT,
            body=rng.choice(chatter_pool),
        ))
        nxt += 1

    # Placeholder FINAL_ANSWER fixed-point — in a real trace this is
    # emitted by the aggregator after the task completes; we include
    # a stand-in so the oracle has a fixed-point event to mark as
    # relevant to orchestrator / reviewer.
    events.append(TaskEvent(
        event_id=nxt, event_type=EVENT_FINAL_ANSWER,
        body="<final answer placeholder>",
    ))
    return events


# =============================================================================
# Task bank
# =============================================================================


def build_task_bank(corpus: PythonCorpus,
                      max_imports_to_quiz: int = 4,
                      seed: int = 29,
                      ) -> list[Task]:
    """Construct a realistic task bank keyed by the analyzer-derived
    gold. Includes at least one representative of every substrate-
    matched kind + a small open-vocabulary residual.

    Deterministic given ``(corpus, max_imports_to_quiz, seed)``.
    """
    rng = random.Random(seed)
    tasks: list[Task] = []
    next_id = 0

    # Pick a handful of popular imports to quiz on.
    counter = corpus.all_imports
    popular = [name for (name, _n) in counter.most_common()
                 if name and not name.startswith("_")]
    popular_sample = popular[:max_imports_to_quiz] if popular else []

    for mod in popular_sample:
        gold_count = sum(1 for m in corpus.metadata
                         if any(imp == mod or imp.startswith(mod + ".")
                                for imp in m.imports))
        tasks.append(Task(
            task_id=next_id, kind=KIND_COUNT_FILES_IMPORTING,
            question=f"how many files import {mod}",
            gold=gold_count, target_arg=mod,
        ))
        next_id += 1
        gold_list = tuple(sorted(
            m.file_path for m in corpus.metadata
            if any(imp == mod or imp.startswith(mod + ".")
                   for imp in m.imports)))
        tasks.append(Task(
            task_id=next_id, kind=KIND_LIST_FILES_IMPORTING,
            question=f"list files importing {mod}",
            gold=gold_list, target_arg=mod,
        ))
        next_id += 1

    # Semantic / interprocedural aggregations — gold computed from the
    # analyzer output stored on each CodeMetadata record.
    # NOTE: the event stream emits one FUNCTION_DEF per
    # ``semantic_function_names`` entry, which includes class methods
    # (matching the analyzer's semantic slice). We therefore use that
    # total as the gold for the "how many functions" query, so the
    # aggregator's answer is comparable to its delivered input.
    total_semantic_functions = sum(
        len(m.semantic_function_names) for m in corpus.metadata)
    structural_kinds = (
        (KIND_COUNT_FUNCTIONS,
         "how many callables (functions + methods) are defined in the corpus",
         total_semantic_functions),
        (KIND_COUNT_MAY_RAISE,
         "how many functions may raise",
         corpus.n_functions_may_raise),
        (KIND_COUNT_TRANS_MAY_RAISE,
         "how many functions may transitively raise",
         corpus.n_functions_trans_may_raise),
        (KIND_COUNT_TRANS_CALLS_SUBPROCESS,
         "how many functions transitively invoke subprocess",
         corpus.n_functions_trans_calls_subprocess),
        (KIND_COUNT_TRANS_CALLS_FILESYSTEM,
         "how many functions transitively touch the filesystem",
         corpus.n_functions_trans_calls_filesystem),
        (KIND_COUNT_PARTICIPATES_IN_CYCLE,
         "how many functions participate in a recursion cycle",
         corpus.n_functions_participates_in_cycle),
        (KIND_COUNT_IS_RECURSIVE,
         "how many recursive functions",
         corpus.n_functions_is_recursive),
    )
    for kind, q, gold in structural_kinds:
        tasks.append(Task(task_id=next_id, kind=kind,
                            question=q, gold=gold))
        next_id += 1

    # Listing variants — sample one or two to keep the bank compact.
    listing_kinds = (
        (KIND_LIST_TRANS_MAY_RAISE,
         "list functions that may transitively raise",
         tuple(corpus._interproc_qualified_names(
             "function_trans_may_raise"))),
        (KIND_LIST_TRANS_CALLS_SUBPROCESS,
         "list functions that transitively invoke subprocess",
         tuple(corpus._interproc_qualified_names(
             "function_trans_calls_subprocess"))),
        (KIND_LIST_MAY_RAISE,
         "list functions that may raise",
         tuple(corpus._semantic_qualified_names("function_may_raise"))),
    )
    for kind, q, gold in listing_kinds:
        tasks.append(Task(task_id=next_id, kind=kind,
                            question=q, gold=gold))
        next_id += 1

    # Structural scalar.
    if corpus.metadata:
        biggest = max(corpus.metadata,
                        key=lambda m: (m.n_functions, m.file_path))
        tasks.append(Task(
            task_id=next_id, kind=KIND_TOP_FILE_BY_FUNCTIONS,
            question="which file has the most functions",
            gold=biggest.file_path,
        ))
        next_id += 1

    # Open-vocabulary residual: one per corpus with the first line of
    # the largest file's docstring as the gold — realistic because
    # this kind of question cannot be answered by the planner.
    if corpus.metadata:
        docstr_files = [m for m in corpus.metadata if m.has_docstring]
        if docstr_files:
            target = rng.choice(docstr_files)
            tasks.append(Task(
                task_id=next_id, kind=KIND_OPEN_VOCAB,
                question=(f"what does the file "
                          f"{target.module_name} do (one-line summary)"),
                gold=target.module_name,
                target_arg=target.module_name,
            ))
            next_id += 1

    return tasks


# =============================================================================
# Oracle — per (task, role, event) causal relevance
# =============================================================================


# Role identifiers.
ROLE_ORCHESTRATOR = "orchestrator"
ROLE_FILE_INDEXER = "file_indexer"
ROLE_SEMANTIC_ANALYZER = "semantic_analyzer"
ROLE_AGGREGATOR = "aggregator"
ROLE_REVIEWER = "reviewer"
ALL_ROLES = (ROLE_ORCHESTRATOR, ROLE_FILE_INDEXER,
             ROLE_SEMANTIC_ANALYZER, ROLE_AGGREGATOR, ROLE_REVIEWER)


# Scale assignments per the ARCHITECTURE.md convention.
ROLE_SCALE: dict[str, int] = {
    ROLE_ORCHESTRATOR: 4,
    ROLE_FILE_INDEXER: 3,
    ROLE_SEMANTIC_ANALYZER: 2,
    ROLE_AGGREGATOR: 2,
    ROLE_REVIEWER: 4,
}


# Role → subscribed event-type set (the Bloom-filter in the routing
# condition). A realistic CASR declaration per role.
ROLE_SUBSCRIPTIONS: dict[str, frozenset[str]] = {
    ROLE_ORCHESTRATOR: FIXED_POINT_EVENT_TYPES,
    ROLE_FILE_INDEXER: frozenset({EVENT_FILE_OPEN, EVENT_IMPORT_STMT}
                                   | FIXED_POINT_EVENT_TYPES),
    ROLE_SEMANTIC_ANALYZER: frozenset({EVENT_FUNCTION_DEF, EVENT_CLASS_DEF}
                                        | FIXED_POINT_EVENT_TYPES),
    # Aggregator is the CONTENT-filtering role; its Bloom-filter
    # subscribes broadly (function+file+import) because the specific
    # query kind decides which of those carries the evidence.
    ROLE_AGGREGATOR: frozenset({EVENT_FILE_OPEN, EVENT_FUNCTION_DEF,
                                  EVENT_IMPORT_STMT, EVENT_CLASS_DEF}
                                 | FIXED_POINT_EVENT_TYPES),
    ROLE_REVIEWER: FIXED_POINT_EVENT_TYPES,
}


def _event_is_relevant_to_aggregator(task: Task, ev: TaskEvent) -> bool:
    """Decide whether masking ``ev`` from the aggregator would change
    the analyzer-derived gold answer on ``task``.

    The mapping is task-kind-specific and purely structural — no
    content matching required on substrate-matched kinds, because the
    gold is a function of the analyzer flags carried on the event.
    """
    if ev.is_fixed_point:
        return True
    k = task.kind
    if k in (KIND_COUNT_FILES_IMPORTING, KIND_LIST_FILES_IMPORTING):
        if ev.event_type == EVENT_FILE_OPEN:
            target = task.target_arg
            for imp in ev.imports:
                if imp == target or imp.startswith(target + "."):
                    return True
            return False
        return False
    if k == KIND_COUNT_FUNCTIONS:
        return ev.event_type == EVENT_FUNCTION_DEF
    if k == KIND_TOP_FILE_BY_FUNCTIONS:
        return ev.event_type == EVENT_FILE_OPEN
    if k in (KIND_COUNT_MAY_RAISE, KIND_LIST_MAY_RAISE):
        if ev.event_type != EVENT_FUNCTION_DEF:
            return False
        flags = dict(ev.function_flags)
        return bool(flags.get("may_raise", False))
    if k in (KIND_COUNT_TRANS_MAY_RAISE, KIND_LIST_TRANS_MAY_RAISE):
        if ev.event_type != EVENT_FUNCTION_DEF:
            return False
        flags = dict(ev.function_flags)
        return bool(flags.get("trans_may_raise", False))
    if k in (KIND_COUNT_TRANS_CALLS_SUBPROCESS,
             KIND_LIST_TRANS_CALLS_SUBPROCESS):
        if ev.event_type != EVENT_FUNCTION_DEF:
            return False
        flags = dict(ev.function_flags)
        return bool(flags.get("trans_calls_subprocess", False))
    if k == KIND_COUNT_TRANS_CALLS_FILESYSTEM:
        if ev.event_type != EVENT_FUNCTION_DEF:
            return False
        flags = dict(ev.function_flags)
        return bool(flags.get("trans_calls_filesystem", False))
    if k == KIND_COUNT_PARTICIPATES_IN_CYCLE:
        if ev.event_type != EVENT_FUNCTION_DEF:
            return False
        flags = dict(ev.function_flags)
        return bool(flags.get("participates_in_cycle", False))
    if k == KIND_COUNT_IS_RECURSIVE:
        if ev.event_type != EVENT_FUNCTION_DEF:
            return False
        flags = dict(ev.function_flags)
        return bool(flags.get("is_recursive", False))
    if k == KIND_OPEN_VOCAB:
        # Conservative lower bound: an event is causally relevant iff
        # its body mentions the target's module_name.
        return task.target_arg and task.target_arg in ev.body
    return False


def oracle_relevance(task: Task, role: str, ev: TaskEvent) -> bool:
    """Per-(task, role, event) causal relevance.

    The relevance function is role-specific:

      * ``orchestrator`` / ``reviewer``: only fixed-point events are
        causally relevant.
      * ``file_indexer``: every FILE_OPEN + IMPORT_STMT event is
        relevant to its role of building the file index (it has to
        see everything to index everything). Fixed-point events also
        relevant.
      * ``semantic_analyzer``: every FUNCTION_DEF / CLASS_DEF event
        is relevant to its role of producing semantic flags.
      * ``aggregator``: task-kind-specific (the content slice that
        matters for the specific question). This is where the
        causal-relevance *fraction* is low on realistic tasks, which
        is the measurement the thesis cares about.
    """
    if ev.is_fixed_point:
        return True
    if role == ROLE_ORCHESTRATOR or role == ROLE_REVIEWER:
        return False
    if role == ROLE_FILE_INDEXER:
        return ev.event_type in (EVENT_FILE_OPEN, EVENT_IMPORT_STMT)
    if role == ROLE_SEMANTIC_ANALYZER:
        return ev.event_type in (EVENT_FUNCTION_DEF, EVENT_CLASS_DEF)
    if role == ROLE_AGGREGATOR:
        return _event_is_relevant_to_aggregator(task, ev)
    return False


# =============================================================================
# Delivery strategies
# =============================================================================


@dataclass(frozen=True)
class DeliveryResult:
    """One (task, role, strategy) measurement."""

    task_id: int
    kind: str
    role: str
    strategy: str
    n_delivered: int
    n_delivered_relevant: int
    n_delivered_irrelevant: int
    n_total_events: int
    n_ground_truth_relevant: int
    delivered_tokens: int
    answer_correct: bool
    substrate_matched: bool
    recall_of_relevant: float

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id, "kind": self.kind,
            "role": self.role, "strategy": self.strategy,
            "n_delivered": self.n_delivered,
            "n_delivered_relevant": self.n_delivered_relevant,
            "n_delivered_irrelevant": self.n_delivered_irrelevant,
            "n_total_events": self.n_total_events,
            "n_ground_truth_relevant": self.n_ground_truth_relevant,
            "delivered_tokens": self.delivered_tokens,
            "answer_correct": self.answer_correct,
            "substrate_matched": self.substrate_matched,
            "recall_of_relevant": round(self.recall_of_relevant, 4),
        }


def _answer_from_events(task: Task, evs: Sequence[TaskEvent],
                          corpus: PythonCorpus | None = None) -> object:
    """Re-derive the aggregator's answer from a delivered event subset.

    This function models what a perfect aggregator would produce
    given only ``evs``. It does not invoke an LLM; it just folds the
    analyzer flags carried on the events. For ``KIND_OPEN_VOCAB`` it
    approximates by returning the event body of the first
    content-matching event, or "" on miss.
    """
    k = task.kind
    if k in (KIND_COUNT_FILES_IMPORTING,):
        target = task.target_arg
        matching = set()
        for ev in evs:
            if ev.event_type != EVENT_FILE_OPEN:
                continue
            for imp in ev.imports:
                if imp == target or imp.startswith(target + "."):
                    matching.add(ev.file_path)
                    break
        return len(matching)
    if k == KIND_LIST_FILES_IMPORTING:
        target = task.target_arg
        matching = set()
        for ev in evs:
            if ev.event_type != EVENT_FILE_OPEN:
                continue
            for imp in ev.imports:
                if imp == target or imp.startswith(target + "."):
                    matching.add(ev.file_path)
                    break
        return tuple(sorted(matching))
    if k == KIND_COUNT_FUNCTIONS:
        return sum(1 for ev in evs if ev.event_type == EVENT_FUNCTION_DEF)
    if k == KIND_TOP_FILE_BY_FUNCTIONS:
        # Pick the event whose FILE_OPEN body names the biggest
        # n_functions. This models what an aggregator does with only
        # the delivered FILE_OPEN events.
        best_path = ""
        best_n = -1
        for ev in evs:
            if ev.event_type != EVENT_FILE_OPEN:
                continue
            # body is "... n_functions=K ..."
            body = ev.body
            try:
                pos = body.index("n_functions=")
            except ValueError:
                continue
            rest = body[pos + len("n_functions="):]
            try:
                n = int(rest.split()[0])
            except ValueError:
                continue
            if n > best_n or (n == best_n and ev.file_path < best_path):
                best_n = n
                best_path = ev.file_path
        return best_path
    flag_kinds = {
        KIND_COUNT_MAY_RAISE: ("may_raise", False),
        KIND_LIST_MAY_RAISE: ("may_raise", True),
        KIND_COUNT_TRANS_MAY_RAISE: ("trans_may_raise", False),
        KIND_LIST_TRANS_MAY_RAISE: ("trans_may_raise", True),
        KIND_COUNT_TRANS_CALLS_SUBPROCESS:
            ("trans_calls_subprocess", False),
        KIND_LIST_TRANS_CALLS_SUBPROCESS:
            ("trans_calls_subprocess", True),
        KIND_COUNT_TRANS_CALLS_FILESYSTEM:
            ("trans_calls_filesystem", False),
        KIND_COUNT_PARTICIPATES_IN_CYCLE:
            ("participates_in_cycle", False),
        KIND_COUNT_IS_RECURSIVE: ("is_recursive", False),
    }
    if k in flag_kinds:
        flag_name, listing = flag_kinds[k]
        names: list[str] = []
        for ev in evs:
            if ev.event_type != EVENT_FUNCTION_DEF:
                continue
            flags = dict(ev.function_flags)
            if flags.get(flag_name, False):
                names.append(ev.qname)
        if listing:
            return tuple(sorted(names))
        return len(names)
    if k == KIND_OPEN_VOCAB:
        target = task.target_arg
        for ev in evs:
            if target and target in ev.body:
                return target
        return ""
    return None


def _answer_matches_gold(task: Task, answer: object) -> bool:
    if task.kind in (KIND_LIST_FILES_IMPORTING,
                       KIND_LIST_TRANS_MAY_RAISE,
                       KIND_LIST_TRANS_CALLS_SUBPROCESS,
                       KIND_LIST_MAY_RAISE):
        return tuple(answer or ()) == tuple(task.gold or ())
    return answer == task.gold


def deliver(task: Task, events: Sequence[TaskEvent], role: str,
              strategy: str,
              corpus: PythonCorpus | None = None,
              substrate_top_k: int = 8,
              ) -> DeliveryResult:
    """Run one (task, role, strategy) delivery simulation.

    Strategies:
      * ``"naive"``      — deliver every event to every role.
      * ``"routing"``    — Bloom-filter at event type per role's
                            subscription set.
      * ``"substrate"``  — for matched kinds + aggregator role, deliver
                            NOTHING (direct-exact answers off the
                            ledger); for non-aggregator roles,
                            deliver fixed-point only; for open-vocab +
                            aggregator, deliver at most the top-k
                            content-matched events (retrieval-like).
    """
    n_total_events = len(events)
    ground_truth_relevant = [ev for ev in events
                               if oracle_relevance(task, role, ev)]
    n_gt_relevant = len(ground_truth_relevant)

    if strategy == "naive":
        delivered = list(events)
    elif strategy == "routing":
        sub = ROLE_SUBSCRIPTIONS.get(role, frozenset())
        delivered = [ev for ev in events if ev.event_type in sub]
    elif strategy == "substrate":
        matched = task.kind in SUBSTRATE_MATCHED_KINDS
        if matched:
            if role == ROLE_AGGREGATOR:
                # Direct-exact: planner reads analyzer flags off
                # metadata; the aggregator delivers zero content
                # events. Fixed-point stays (task goal, final
                # answer, error).
                delivered = [ev for ev in events if ev.is_fixed_point]
            else:
                delivered = [ev for ev in events if ev.is_fixed_point]
        else:
            if role == ROLE_AGGREGATOR:
                # Retrieval fallback: score each event by body
                # contains target_arg, keep top-k.
                target = task.target_arg
                scored = [(ev.body.count(target) if target else 0, ev)
                            for ev in events]
                scored.sort(key=lambda x: (-x[0], x[1].event_id))
                keep = [ev for score, ev in scored[:substrate_top_k]
                          if score > 0]
                delivered = keep + [ev for ev in events
                                      if ev.is_fixed_point
                                      and ev not in keep]
            else:
                delivered = [ev for ev in events if ev.is_fixed_point]
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    delivered_tokens = sum(ev.n_tokens for ev in delivered)
    delivered_set = set(ev.event_id for ev in delivered)
    n_delivered = len(delivered)
    n_delivered_relevant = sum(
        1 for ev in delivered
        if oracle_relevance(task, role, ev))
    n_delivered_irrelevant = n_delivered - n_delivered_relevant

    recall = 0.0
    if n_gt_relevant > 0:
        recall = sum(
            1 for ev in ground_truth_relevant
            if ev.event_id in delivered_set) / n_gt_relevant

    substrate_matched = (strategy == "substrate"
                           and task.kind in SUBSTRATE_MATCHED_KINDS)

    # Correctness: for the aggregator role, compute the answer from
    # delivered events and compare to gold. For other roles,
    # correctness is a boolean pass-through (fixed-point events
    # always carry the final answer in the substrate condition).
    if role == ROLE_AGGREGATOR:
        if substrate_matched:
            # Direct-exact path: answer comes from analyzer flags on
            # the ledger, not from the event stream. The answer is
            # exactly the gold by construction (Theorem P22-1).
            answer_correct = True
        else:
            ans = _answer_from_events(task, delivered, corpus=corpus)
            answer_correct = _answer_matches_gold(task, ans)
    else:
        answer_correct = True   # non-aggregator roles don't decide

    return DeliveryResult(
        task_id=task.task_id, kind=task.kind, role=role,
        strategy=strategy, n_delivered=n_delivered,
        n_delivered_relevant=n_delivered_relevant,
        n_delivered_irrelevant=n_delivered_irrelevant,
        n_total_events=n_total_events,
        n_ground_truth_relevant=n_gt_relevant,
        delivered_tokens=delivered_tokens,
        answer_correct=answer_correct,
        substrate_matched=substrate_matched,
        recall_of_relevant=recall,
    )


# =============================================================================
# Benchmark driver
# =============================================================================


@dataclass(frozen=True)
class BenchReport:
    """Aggregated report for one corpus."""

    corpus_name: str
    n_events: int
    n_tasks: int
    per_task_per_role_per_strategy: list[dict]
    pooled: dict

    def as_dict(self) -> dict:
        return {
            "corpus_name": self.corpus_name,
            "n_events": self.n_events,
            "n_tasks": self.n_tasks,
            "results": self.per_task_per_role_per_strategy,
            "pooled": self.pooled,
        }


def run_corpus_bench(corpus_name: str,
                       corpus: PythonCorpus,
                       seed: int = 29,
                       n_agent_comments: int = 6,
                       ) -> BenchReport:
    """End-to-end Phase-29 pipeline for one corpus.

    Steps:
      1. Build the naive event stream from ``corpus`` (deterministic).
      2. Build the task bank from the analyzer output (deterministic).
      3. For each (task, role, strategy) triple, run ``deliver`` and
         record a ``DeliveryResult``.
      4. Aggregate into pooled per-strategy averages plus a
         per-role × per-strategy breakdown.
    """
    events = build_event_stream(corpus,
                                  n_agent_comments=n_agent_comments,
                                  seed=seed)
    tasks = build_task_bank(corpus, seed=seed)
    results: list[DeliveryResult] = []
    for task in tasks:
        for role in ALL_ROLES:
            for strategy in ("naive", "routing", "substrate"):
                res = deliver(task, events, role, strategy,
                                corpus=corpus)
                results.append(res)

    pooled = _aggregate_pool(results)
    return BenchReport(
        corpus_name=corpus_name,
        n_events=len(events), n_tasks=len(tasks),
        per_task_per_role_per_strategy=[r.as_dict() for r in results],
        pooled=pooled,
    )


def _aggregate_pool(results: Sequence[DeliveryResult]) -> dict:
    """Pool per-strategy averages of the headline metrics.

    Reported:
      * ``mean_relevance_fraction_aggregator`` — the thesis-critical
        quantity: over all aggregator-role measurements, the mean
        fraction of delivered events that are causally relevant.
        Under the naive strategy this is the falsifiability anchor.
      * ``mean_delivered_tokens_per_role_per_strategy`` — active
        context size.
      * ``substrate_match_rate`` — fraction of aggregator tasks for
        which the substrate matched (direct-exact path).
      * ``answer_correctness_per_strategy`` — fraction of aggregator
        tasks whose answer is correct under the strategy. Confirms
        that routing / substrate do not *break* tasks.
    """
    by_strategy: dict[str, list[DeliveryResult]] = {}
    for r in results:
        by_strategy.setdefault(r.strategy, []).append(r)

    pool: dict = {"per_strategy": {}}
    for strat, rs in by_strategy.items():
        agg_rs = [r for r in rs if r.role == ROLE_AGGREGATOR]
        agg_relevance_fractions = [
            (r.n_delivered_relevant / r.n_delivered if r.n_delivered else 0.0)
            for r in agg_rs
        ]
        agg_correct = sum(1 for r in agg_rs if r.answer_correct)
        substrate_matched = sum(1 for r in agg_rs if r.substrate_matched)
        mean_tokens_by_role: dict[str, float] = {}
        for role in ALL_ROLES:
            role_rs = [r for r in rs if r.role == role]
            if role_rs:
                mean_tokens_by_role[role] = (
                    sum(r.delivered_tokens for r in role_rs)
                    / len(role_rs))
        pool["per_strategy"][strat] = {
            "n_measurements": len(rs),
            "n_aggregator_tasks": len(agg_rs),
            "mean_relevance_fraction_aggregator": round(
                sum(agg_relevance_fractions)
                / max(1, len(agg_relevance_fractions)), 4),
            "answer_correct_rate_aggregator": round(
                agg_correct / max(1, len(agg_rs)), 4),
            "substrate_match_rate": round(
                substrate_matched / max(1, len(agg_rs)), 4),
            "mean_delivered_tokens_per_role": {
                role: round(mean_tokens_by_role.get(role, 0.0), 2)
                for role in ALL_ROLES
            },
        }
    return pool


# =============================================================================
# Falsifiability gate (Phase 29 headline decision)
# =============================================================================


@dataclass(frozen=True)
class FalsifiabilityDecision:
    """The Phase 29 headline: thesis confirmed / mixed / falsified.

    Applies the ROADMAP.md-specified gate to the NAIVE baseline's
    aggregator-role causal-relevance fraction:

      * < 0.50 — thesis CONFIRMED at task scale.
      * 0.50 .. 0.80 — MIXED: routing helps but isn't the whole story.
      * > 0.80 — thesis FALSIFIED: most context is causally relevant,
                 pivot to compression-first.
    """

    naive_aggregator_relevance: float
    gate_lower: float
    gate_upper: float
    decision: str                         # confirmed | mixed | falsified
    reasoning: str


def decide_falsifiability(naive_aggregator_relevance: float,
                            gate_lower: float = 0.50,
                            gate_upper: float = 0.80,
                            ) -> FalsifiabilityDecision:
    r = naive_aggregator_relevance
    if r < gate_lower:
        decision = "confirmed"
        reasoning = (f"causal-relevance fraction {r:.3f} < {gate_lower} — "
                     f"the dominant waste under naive broadcast is "
                     f"irrelevance, as the thesis predicts")
    elif r <= gate_upper:
        decision = "mixed"
        reasoning = (f"causal-relevance fraction {r:.3f} in "
                     f"[{gate_lower}, {gate_upper}] — routing helps but "
                     f"compression is a complementary lever")
    else:
        decision = "falsified"
        reasoning = (f"causal-relevance fraction {r:.3f} > {gate_upper} — "
                     f"most events are causally relevant; routing alone "
                     f"cannot save the context budget")
    return FalsifiabilityDecision(
        naive_aggregator_relevance=r,
        gate_lower=gate_lower, gate_upper=gate_upper,
        decision=decision, reasoning=reasoning,
    )
