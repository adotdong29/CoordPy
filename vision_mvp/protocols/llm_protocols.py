"""LLM coordination protocols — naive broadcast vs vision-stack.

Both protocols share the same shape:
  Round 0: each agent generates an initial answer to the question using
           only its persona (no context).
  Rounds 1..K-1: each agent updates its answer given some "context".

The protocols differ only in what `context` each agent sees:

  naive_broadcast:
      context = concatenation of ALL other agents' current answers.
      Token cost per agent per round: O(N · avg_answer_len).

  vision_stack:
      Agents' answers are embedded → projected onto shared manifold
      (streaming PCA on embeddings) → shared summary read by everyone.
      A small orchestrator LLM call decodes the summary into one short
      sentence (the "consensus so far"), which is then broadcast.
      Token cost per agent per round: O(|consensus_sentence|) ≈ O(1).
      Plus one orchestrator LLM call per round (independent of N).

Both protocols use the same LLM model, same personas, same questions.
The only difference is the routing of context. That is what we are
measuring.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from ..core.llm_client import LLMClient, LLMStats
from ..core.llm_agent import LLMAgent
from ..core.learned_manifold import StreamingPCA
from ..tasks.llm_consensus import LLMQuestion, assign_personas


@dataclass
class LLMRunResult:
    protocol: str
    n_agents: int
    rounds: int
    question: str
    ground_truth: str
    final_answers: list[str]
    accuracy: float
    agreement: float
    llm_stats: LLMStats
    per_round_accuracy: list[float]


def _score(answers: list[str], q: LLMQuestion) -> tuple[float, float]:
    """Return (accuracy, agreement)."""
    norm = [q.normalize(a) for a in answers]
    acc = sum(1 for n in norm if n == q.ground_truth) / max(len(norm), 1)
    # Agreement: fraction of agents matching the plurality answer
    if not norm:
        return 0.0, 0.0
    counts: dict[str, int] = {}
    for n in norm:
        counts[n] = counts.get(n, 0) + 1
    agree = max(counts.values()) / len(norm)
    return acc, agree


def run_llm_naive(question: LLMQuestion, n_agents: int, rounds: int = 3,
                  model: str = "qwen2.5:0.5b", seed: int = 0) -> LLMRunResult:
    client = LLMClient(model=model)
    personas = assign_personas(n_agents)
    agents = [LLMAgent(agent_id=i, persona=p, client=client)
              for i, p in enumerate(personas)]
    per_round_acc: list[float] = []

    # Round 0: initial answers (no context)
    for a in agents:
        a.respond(question.question, context="")
    answers = [a.text_state for a in agents]
    acc, _ = _score(answers, question)
    per_round_acc.append(acc)

    # Subsequent rounds: each agent sees all others' answers
    for r in range(1, rounds):
        snapshot = list(answers)
        for i, a in enumerate(agents):
            others = [snapshot[j] for j in range(n_agents) if j != i]
            # Context is the list of other agents' answers
            ctx = "\n".join(f"- {s}" for s in others)
            a.respond(question.question, context=ctx)
        answers = [a.text_state for a in agents]
        acc, _ = _score(answers, question)
        per_round_acc.append(acc)

    acc, agree = _score(answers, question)
    return LLMRunResult(
        protocol="naive",
        n_agents=n_agents,
        rounds=rounds,
        question=question.question,
        ground_truth=question.ground_truth,
        final_answers=answers,
        accuracy=acc,
        agreement=agree,
        llm_stats=client.stats,
        per_round_accuracy=per_round_acc,
    )


def run_llm_vision(question: LLMQuestion, n_agents: int, rounds: int = 3,
                   model: str = "qwen2.5:0.5b", seed: int = 0,
                   manifold_dim: int | None = None) -> LLMRunResult:
    client = LLMClient(model=model)
    personas = assign_personas(n_agents)
    agents = [LLMAgent(agent_id=i, persona=p, client=client)
              for i, p in enumerate(personas)]
    per_round_acc: list[float] = []

    # Round 0
    for a in agents:
        a.respond(question.question, context="")
    answers = [a.text_state for a in agents]
    acc, _ = _score(answers, question)
    per_round_acc.append(acc)

    # Embed initial answers to initialize manifold
    embs = client.embed_batch(answers)
    if not embs:
        # degenerate — return round-0 results
        acc, agree = _score(answers, question)
        return LLMRunResult("vision", n_agents, rounds, question.question,
                            question.ground_truth, answers, acc, agree,
                            client.stats, per_round_acc)
    emb_matrix = np.array(embs, dtype=np.float64)
    d = emb_matrix.shape[1]
    m = manifold_dim or max(2, math.ceil(math.log2(max(n_agents, 2))))
    pca = StreamingPCA.build(d, m, lr=0.3, seed=seed)
    # Warm-start PCA with the first batch
    pca.update_batch(emb_matrix)

    consensus_sentence = "(none yet)"

    for r in range(1, rounds):
        # Embed current answers (batch call = O(1) for accounting)
        embs = client.embed_batch([a.text_state for a in agents])
        emb_matrix = np.array(embs, dtype=np.float64)

        # Update manifold with the mean embedding (√N-denoised signal)
        pca.update(emb_matrix.mean(axis=0))

        # Project and aggregate — the shared summary is the mean projection
        projections = emb_matrix @ pca.basis            # (N, m)
        summary_m = projections.mean(axis=0)             # (m,)

        # Reconstruct a d-dim "centroid embedding"
        centroid_emb = pca.basis @ summary_m             # (d,)

        # Find the agent whose current answer is closest to the centroid;
        # use its text as the "consensus sentence" for broadcast.
        # Cosine similarity is robust to norm differences.
        norms = np.linalg.norm(emb_matrix, axis=1) + 1e-8
        cen_norm = np.linalg.norm(centroid_emb) + 1e-8
        cos = (emb_matrix @ centroid_emb) / (norms * cen_norm)
        best = int(np.argmax(cos))
        consensus_sentence = agents[best].text_state

        # Each agent updates with just this single sentence as context.
        for a in agents:
            a.respond(question.question, context=consensus_sentence)
        answers = [a.text_state for a in agents]
        acc, _ = _score(answers, question)
        per_round_acc.append(acc)

    acc, agree = _score(answers, question)
    return LLMRunResult(
        protocol="vision",
        n_agents=n_agents,
        rounds=rounds,
        question=question.question,
        ground_truth=question.ground_truth,
        final_answers=answers,
        accuracy=acc,
        agreement=agree,
        llm_stats=client.stats,
        per_round_accuracy=per_round_acc,
    )
