"""Phase 19 — Lossless context substrate vs. map-reduce baseline.

Two modes (more later):

  1. **map_reduce**: the existing Phase-8/9 pattern. Each section is fed to
     an LLM with a "summarise the concrete facts" prompt; the per-section
     summaries are pooled into ONE prompt, and a synthesiser LLM call
     answers the question against the pooled summaries. The synthesiser
     never sees the source text.

  2. **lossless**: each section goes into a `ContextLedger` byte-for-byte;
     a `BoundedRetrievalWorker` searches the ledger, fetches exact spans
     of the top-k handles, and answers from those verbatim excerpts.
     No summarisation step exists in this path.

Per question we score:
  - `exact_correct`: did the answer literally contain the ground-truth
                     string (incident ID, vendor, ticket, SLA value, etc.)?
  - `prompt_chars`:  size of the prompt sent to the answering LLM call
  - `fetch_count`:   for lossless mode only — number of ledger fetches
  - `fact_in_input`: did the gold fact appear ANYWHERE in the answering
                     LLM's input? (For map-reduce: was the fact preserved
                     across the summary step? For lossless: was the fact
                     in any of the fetched excerpts?)

That last metric is the key Phase-19 diagnostic: it cleanly separates
**substrate loss** (fact never made it to the LLM) from **LLM error**
(fact was in input, model still got it wrong).

LLM modes:
  - `--mode mock`     uses a deterministic mock LLM that copies any
                      ground-truth fact it sees in the prompt into the
                      answer. Validates the substrate end-to-end without
                      Ollama. This is the smoke path.
  - `--mode ollama`   uses local Ollama via `core/llm_client.py`.

Reproduce smoke:
    python -m vision_mvp.experiments.phase19_lossless --mode mock --n 24

Reproduce real:
    python -m vision_mvp.experiments.phase19_lossless --mode ollama --n 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.bounded_worker import BoundedRetrievalWorker
from vision_mvp.core.context_ledger import ContextLedger, hash_embedding
from vision_mvp.tasks.needle_corpus import NeedleCorpus, NeedleQuestion


# ---------- LLM adapters -------------------------------------------------


class MockLLM:
    """State-based mock LLM with two distinct behaviours.

    State is held externally so the runner can switch modes between
    setup (summarisation) and answer (extraction). The two modes are
    independently realistic baselines:

      - mode='summarise': returns the FIRST line of the input. Real LLMs
        asked for a "2-3 sentence summary" tend to compress narrative and
        drop rare entities (incident IDs, ticket numbers). Returning
        title-only is a clean stand-in: anything in the section title
        survives, anything in the body does not.

      - mode='extract' with current_gold set: scans the prompt for the
        SPECIFIC gold of the active question. Returns it if present,
        else "not found in input". This isolates substrate loss from
        LLM ability — a perfect-extractor LLM tells us exactly whether
        the substrate delivered the fact to the prompt.

    Counts every call for reporting.
    """

    def __init__(self):
        self.mode: str = "extract"
        self.current_gold: str | None = None
        self.n_calls: int = 0

    def __call__(self, prompt: str) -> str:
        self.n_calls += 1
        if self.mode == "summarise":
            # Real LLMs asked for a 2-3 sentence summary keep the lead
            # (who/what/where) and drop the rare numerical entities
            # (ticket id, SLA value, MTTD/MTTR hours). We model this by
            # returning the first ~2 sentences of the chunk after the
            # prompt header. A real model would also paraphrase, losing
            # additional exact strings — so this is a generous baseline.
            chunk = self._extract_chunk(prompt)
            return f"SUMMARY: {self._first_n_sentences(chunk, n=2, max_chars=200)}"
        # extract mode
        if self.current_gold is None:
            return "(no active question)"
        if self.current_gold.lower() in prompt.lower():
            return f"Answer: {self.current_gold}"
        return "Answer: not found in input"

    @staticmethod
    def _extract_chunk(prompt: str) -> str:
        """The chunk lives between the two `---` markers."""
        first = prompt.find("---")
        if first == -1:
            return prompt
        second = prompt.find("---", first + 3)
        if second == -1:
            return prompt[first + 3:].strip()
        return prompt[first + 3:second].strip()

    @staticmethod
    def _first_n_sentences(text: str, n: int = 2, max_chars: int = 200) -> str:
        """Take the first n sentences (period-delimited) capped at max_chars."""
        out: list[str] = []
        for s in text.replace("\n", " ").split("."):
            s = s.strip()
            if not s:
                continue
            out.append(s + ".")
            if len(out) >= n:
                break
        joined = " ".join(out)
        return joined[:max_chars]


def _ollama_llm(model: str, max_tokens: int) -> Callable[[str], str]:
    """Returns a callable that hits local Ollama. Imports at use-time so the
    mock smoke path doesn't require an Ollama install."""
    from vision_mvp.core.llm_client import LLMClient
    client = LLMClient(model=model)
    def llm(prompt: str) -> str:
        return client.generate(prompt, max_tokens=max_tokens, temperature=0.0)
    llm.client = client    # expose for token accounting
    return llm


def _ollama_embed_factory(model: str, dim: int) -> Callable[[str], np.ndarray]:
    """Returns an embed_fn backed by Ollama. Uses dim-dimensional projection
    of Ollama's native embedding (truncated/padded to `dim`).

    For the smoke path we use `hash_embedding` instead — see callsites."""
    from vision_mvp.core.llm_client import LLMClient
    client = LLMClient(model=model)
    def embed(text: str) -> np.ndarray:
        v = client.embed(text)
        v = np.asarray(v, dtype=float)
        if v.size >= dim:
            return v[:dim]
        out = np.zeros(dim, dtype=float)
        out[:v.size] = v
        return out
    embed.client = client
    return embed


# ---------- The two modes ------------------------------------------------


@dataclass
class QuestionResult:
    question: str
    gold: str
    kind: str
    answer: str
    exact_correct: bool
    prompt_chars: int
    fact_in_input: bool
    fetch_count: int = 0
    fetched_bytes: int = 0
    cited_cids: list[str] = field(default_factory=list)
    source_section: tuple[int, ...] = field(default_factory=tuple)
    extra: dict = field(default_factory=dict)


@dataclass
class ModeReport:
    mode: str
    questions: list[QuestionResult] = field(default_factory=list)
    setup_seconds: float = 0.0
    answer_seconds: float = 0.0
    n_llm_setup_calls: int = 0
    n_llm_answer_calls: int = 0
    extras: dict = field(default_factory=dict)

    def aggregate(self) -> dict:
        n = len(self.questions)
        if n == 0:
            return {}
        n_exact = sum(1 for q in self.questions if q.exact_correct)
        n_in_input = sum(1 for q in self.questions if q.fact_in_input)
        n_in_input_correct = sum(
            1 for q in self.questions
            if q.fact_in_input and q.exact_correct)
        prompt_mean = sum(q.prompt_chars for q in self.questions) / n
        prompt_max = max(q.prompt_chars for q in self.questions)
        return {
            "mode": self.mode,
            "n_questions": n,
            "exact_correct": n_exact,
            "exact_correct_rate": round(n_exact / n, 4),
            "fact_in_input": n_in_input,
            "fact_in_input_rate": round(n_in_input / n, 4),
            "llm_accuracy_given_fact": (
                round(n_in_input_correct / max(n_in_input, 1), 4)),
            "mean_prompt_chars": round(prompt_mean, 1),
            "max_prompt_chars": prompt_max,
            "mean_fetch_count": round(
                sum(q.fetch_count for q in self.questions) / n, 2),
            "mean_fetched_bytes": round(
                sum(q.fetched_bytes for q in self.questions) / n, 1),
            "setup_seconds": round(self.setup_seconds, 2),
            "answer_seconds": round(self.answer_seconds, 2),
            "n_llm_setup_calls": self.n_llm_setup_calls,
            "n_llm_answer_calls": self.n_llm_answer_calls,
            "extras": self.extras,
        }


# ---------- Map-reduce baseline (lossy summarisation) --------------------


_MR_SUMMARY_PROMPT = (
    "You are reading ONE chunk of a long incident review. Produce a 2-3 "
    "sentence summary of the FACTS in this chunk relevant to systemic-risk "
    "analysis. Be concise.\n\n---\n{chunk}\n---\n\nSummary:"
)

_MR_ANSWER_PROMPT = (
    "You are answering a question from per-chunk team-member summaries. "
    "Each summary covers one chunk of a longer incident review. Use ONLY "
    "what the summaries say.\n\n"
    "Question: {question}\n\n"
    "Summaries:\n{summaries}\n\nAnswer:"
)


def run_map_reduce(
    corpus: NeedleCorpus,
    llm,
    summary_prompt_chars_max: int,
    answer_prompt_chars_max: int,
    progress: Callable[[str], None] = print,
) -> ModeReport:
    """Map: summarise each section. Reduce: per-question synthesis.

    Summaries are computed once and reused across questions — this is the
    fair comparison: a deployment that re-summarises per question would be
    substantially more expensive."""
    rep = ModeReport(mode="map_reduce")
    sections = corpus.sections

    # ---- map step: summarise each section
    t0 = time.time()
    summaries: list[str] = []
    if isinstance(llm, MockLLM):
        llm.mode = "summarise"
        llm.current_gold = None
    for i, sec in enumerate(sections):
        prompt = _MR_SUMMARY_PROMPT.format(chunk=sec[:summary_prompt_chars_max])
        s = llm(prompt)
        summaries.append(s)
        rep.n_llm_setup_calls += 1
        if (i + 1) % max(1, len(sections) // 5) == 0:
            progress(f"  [map] summarised {i+1}/{len(sections)} sections")
    rep.setup_seconds = time.time() - t0
    rep.extras["summary_total_chars"] = sum(len(s) for s in summaries)
    rep.extras["section_total_chars"] = sum(len(s) for s in sections)
    rep.extras["compression_ratio"] = round(
        rep.extras["summary_total_chars"] / max(1, rep.extras["section_total_chars"]),
        3)

    # ---- reduce step: one LLM call per question
    pooled = "\n\n".join(f"Member {i+1}: {s}" for i, s in enumerate(summaries))
    # If pooled exceeds the answer budget, naive truncation (a real-world
    # failure mode that summarisation pipelines silently hit).
    if len(pooled) > answer_prompt_chars_max:
        pooled = pooled[:answer_prompt_chars_max] + "\n[...truncated]"
        rep.extras["pooled_truncated"] = True
    else:
        rep.extras["pooled_truncated"] = False

    t0 = time.time()
    if isinstance(llm, MockLLM):
        llm.mode = "extract"
    for q in corpus.questions:
        prompt = _MR_ANSWER_PROMPT.format(question=q.question, summaries=pooled)
        if isinstance(llm, MockLLM):
            llm.current_gold = q.gold
        ans = llm(prompt)
        rep.n_llm_answer_calls += 1

        # Was the gold fact preserved across summarisation?
        fact_in_input = q.gold.lower() in pooled.lower()
        rep.questions.append(QuestionResult(
            question=q.question,
            gold=q.gold,
            kind=q.kind,
            answer=ans,
            exact_correct=NeedleCorpus.score_exact(ans, q),
            prompt_chars=len(prompt),
            fact_in_input=fact_in_input,
            source_section=q.source_section,
        ))
    rep.answer_seconds = time.time() - t0
    return rep


# ---------- Lossless retrieval (Phase 19) --------------------------------


def run_lossless(
    corpus: NeedleCorpus,
    llm: Callable[[str], str],
    embed_fn: Callable[[str], np.ndarray],
    embed_dim: int,
    prompt_budget_chars: int,
    top_k: int,
    fetch_chars_per_handle: int,
    progress: Callable[[str], None] = print,
) -> ModeReport:
    """Each section → ledger as exact bytes; per-question worker fetches."""
    rep = ModeReport(mode="lossless")

    # ---- setup: index sections into the ledger
    t0 = time.time()
    ledger = ContextLedger(embed_dim=embed_dim, embed_fn=embed_fn)
    n_index_calls = 0
    for i, sec in enumerate(corpus.sections):
        ledger.put(sec, metadata={
            "section_idx": i,
            "incident_id": corpus.section_meta[i]["incident_id"],
        })
        n_index_calls += 1     # one embed call per section
        if (i + 1) % max(1, len(corpus.sections) // 5) == 0:
            progress(f"  [index] put {i+1}/{len(corpus.sections)} sections")
    rep.setup_seconds = time.time() - t0
    rep.n_llm_setup_calls = 0          # embeddings are not generation calls
    rep.extras["n_embed_calls"] = n_index_calls
    rep.extras["ledger_bytes_stored"] = ledger.stats_dict()["bytes_stored"]

    # ---- worker
    worker = BoundedRetrievalWorker(
        ledger=ledger,
        llm_call=llm,
        prompt_budget_chars=prompt_budget_chars,
        top_k=top_k,
        fetch_chars_per_handle=fetch_chars_per_handle,
    )

    # ---- answer phase
    t0 = time.time()
    if isinstance(llm, MockLLM):
        llm.mode = "extract"
    for q in corpus.questions:
        if isinstance(llm, MockLLM):
            llm.current_gold = q.gold
        r = worker.answer(q.question)
        rep.n_llm_answer_calls += 1

        # Did the worker actually pull the right section?
        candidate_sections = []
        for handle, _sim in r.candidates:
            mi = handle.metadata_dict().get("section_idx")
            if mi is not None:
                candidate_sections.append(mi)
        retrieval_recall = any(
            ss in candidate_sections for ss in q.source_section)

        # Was the gold literally in any cited excerpt?
        fact_in_input = q.gold.lower() in r.prompt.lower()

        rep.questions.append(QuestionResult(
            question=q.question,
            gold=q.gold,
            kind=q.kind,
            answer=r.answer,
            exact_correct=NeedleCorpus.score_exact(r.answer, q),
            prompt_chars=r.prompt_chars,
            fact_in_input=fact_in_input,
            fetch_count=r.fetch_count,
            fetched_bytes=r.fetched_bytes,
            cited_cids=r.cited_cids,
            source_section=q.source_section,
            extra={"retrieval_recall_at_k": retrieval_recall,
                   "candidate_sections": candidate_sections},
        ))
    rep.answer_seconds = time.time() - t0
    return rep


# ---------- Oracle (full doc in prompt) ----------------------------------


def run_oracle(
    corpus: NeedleCorpus,
    llm,
    prompt_budget_chars: int,
) -> ModeReport:
    """Single agent, full document in the prompt. Skipped if doc > budget."""
    rep = ModeReport(mode="oracle")
    full_doc = corpus.document
    if len(full_doc) > prompt_budget_chars:
        rep.extras["skipped"] = "document_exceeds_budget"
        rep.extras["doc_chars"] = len(full_doc)
        rep.extras["budget_chars"] = prompt_budget_chars
        return rep

    t0 = time.time()
    if isinstance(llm, MockLLM):
        llm.mode = "extract"
    for q in corpus.questions:
        prompt = (
            "You are a senior risk analyst. Read the following document and "
            "answer the question with EXACT facts from the document.\n\n"
            f"---\n{full_doc}\n---\n\nQuestion: {q.question}\n\nAnswer:"
        )
        if isinstance(llm, MockLLM):
            llm.current_gold = q.gold
        ans = llm(prompt)
        rep.n_llm_answer_calls += 1
        rep.questions.append(QuestionResult(
            question=q.question,
            gold=q.gold,
            kind=q.kind,
            answer=ans,
            exact_correct=NeedleCorpus.score_exact(ans, q),
            prompt_chars=len(prompt),
            fact_in_input=q.gold.lower() in full_doc.lower(),
            source_section=q.source_section,
        ))
    rep.answer_seconds = time.time() - t0
    return rep


# ---------- CLI ----------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mock", "ollama"], default="mock",
                    help="LLM backend. mock = no Ollama needed (smoke path)")
    ap.add_argument("--n", type=int, default=24,
                    help="number of corpus sections")
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--model", default="qwen2.5:0.5b",
                    help="(ollama mode) model id")
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--prompt-budget", type=int, default=4000,
                    help="chars in the answering LLM's prompt")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--fetch-chars", type=int, default=600,
                    help="bytes per fetched handle in the lossless worker")
    ap.add_argument("--summary-prompt-cap", type=int, default=4000,
                    help="cap on a single section in the map-reduce summary "
                         "prompt (real Ollama 4k context limit)")
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--skip-mr", action="store_true",
                    help="skip the map-reduce baseline")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # ---- corpus
    corpus = NeedleCorpus(n_sections=args.n, seed=args.seed)
    corpus.build()
    print(f"Corpus: {corpus.word_count} words across {len(corpus.sections)} "
          f"sections; {len(corpus.questions)} needle questions.", flush=True)
    print(f"Mode: {args.mode}", flush=True)

    # ---- adapters
    if args.mode == "mock":
        llm = MockLLM()
        embed_fn = lambda t: hash_embedding(t, dim=args.embed_dim)
    else:
        llm = _ollama_llm(args.model, max_tokens=200)
        embed_fn = _ollama_embed_factory(args.model, dim=args.embed_dim)

    reports: dict[str, ModeReport] = {}

    # ---- map-reduce baseline
    if not args.skip_mr:
        print("\n" + "=" * 78)
        print("MODE: map_reduce baseline (summarise → pool → answer)")
        print("=" * 78, flush=True)
        rep_mr = run_map_reduce(
            corpus, llm,
            summary_prompt_chars_max=args.summary_prompt_cap,
            answer_prompt_chars_max=args.prompt_budget,
        )
        reports["map_reduce"] = rep_mr
        agg = rep_mr.aggregate()
        print(f"map_reduce → exact={agg['exact_correct']}/{agg['n_questions']} "
              f"({agg['exact_correct_rate']*100:.1f}%) "
              f"fact_in_input={agg['fact_in_input']}/{agg['n_questions']} "
              f"({agg['fact_in_input_rate']*100:.1f}%)", flush=True)

    # ---- lossless
    print("\n" + "=" * 78)
    print("MODE: lossless retrieval (Phase 19 — exact ledger + bounded worker)")
    print("=" * 78, flush=True)
    rep_l = run_lossless(
        corpus, llm, embed_fn=embed_fn, embed_dim=args.embed_dim,
        prompt_budget_chars=args.prompt_budget, top_k=args.top_k,
        fetch_chars_per_handle=args.fetch_chars,
    )
    reports["lossless"] = rep_l
    agg = rep_l.aggregate()
    print(f"lossless → exact={agg['exact_correct']}/{agg['n_questions']} "
          f"({agg['exact_correct_rate']*100:.1f}%) "
          f"fact_in_input={agg['fact_in_input']}/{agg['n_questions']} "
          f"({agg['fact_in_input_rate']*100:.1f}%)", flush=True)

    # ---- oracle
    if not args.skip_oracle:
        print("\n" + "=" * 78)
        print("MODE: oracle (full doc in prompt; skipped if too large)")
        print("=" * 78, flush=True)
        rep_o = run_oracle(corpus, llm, prompt_budget_chars=128_000)
        reports["oracle"] = rep_o
        agg_o = rep_o.aggregate()
        if "skipped" in rep_o.extras:
            print(f"oracle SKIPPED: {rep_o.extras['skipped']} "
                  f"(doc {rep_o.extras['doc_chars']} > budget "
                  f"{rep_o.extras['budget_chars']})", flush=True)
        else:
            print(f"oracle → exact={agg_o['exact_correct']}/"
                  f"{agg_o['n_questions']} "
                  f"({agg_o['exact_correct_rate']*100:.1f}%) "
                  f"fact_in_input={agg_o['fact_in_input']}/"
                  f"{agg_o['n_questions']}", flush=True)

    # ---- summary table
    print("\n" + "=" * 78)
    print("SCOREBOARD")
    print("=" * 78, flush=True)
    head = (f"{'mode':>14} | {'exact':>10} | {'fact_in':>10} | "
            f"{'mean_pmt':>9} | {'max_pmt':>8}")
    print(head)
    print("-" * len(head))
    for name, rep in reports.items():
        a = rep.aggregate()
        if not a:
            print(f"{name:>14} | {'(skipped)':>10} | "
                  f"{'-':>10} | {'-':>9} | {'-':>8}")
            continue
        ec = f"{a['exact_correct']}/{a['n_questions']}"
        fi = f"{a['fact_in_input']}/{a['n_questions']}"
        print(f"{name:>14} | {ec:>10} | {fi:>10} | "
              f"{a['mean_prompt_chars']:>9.0f} | {a['max_prompt_chars']:>8}")

    if args.out:
        out = {
            "config": vars(args),
            "corpus_word_count": corpus.word_count,
            "n_sections": len(corpus.sections),
            "n_questions": len(corpus.questions),
            "reports": {
                name: {
                    "aggregate": rep.aggregate(),
                    "questions": [asdict(q) for q in rep.questions],
                }
                for name, rep in reports.items()
            },
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nWrote {args.out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
