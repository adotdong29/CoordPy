"""LLM-backed agent: persona + text state + embedding hook.

Each agent holds:
  - persona: a short string giving it a biased perspective
  - text_state: its current answer/opinion (natural language)
  - embedding: cached embedding of text_state (refreshed when it changes)

The agent's `respond(question, context)` produces a new text_state by
prompting the LLM with:
  <persona>
  <context>     ← optional: shared summary or peers' answers
  Question: <question>
  Answer briefly:

That answer becomes the new text_state. The embedding is what flows through
the shared latent manifold.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from .llm_client import LLMClient


@dataclass
class LLMAgent:
    agent_id: int
    persona: str
    client: LLMClient
    text_state: str = ""
    embedding: np.ndarray = None  # type: ignore

    def respond(self, question: str, context: str = "",
                max_tokens: int = 60) -> str:
        prompt_parts = [self.persona.strip()]
        if context:
            prompt_parts.append(f"Shared context so far: {context.strip()}")
        prompt_parts.append(f"Question: {question.strip()}")
        prompt_parts.append("Answer in one short sentence:")
        prompt = "\n\n".join(prompt_parts)
        self.text_state = self.client.generate(prompt, max_tokens=max_tokens)
        self.embedding = None   # invalidate — refresh on demand
        return self.text_state

    def refresh_embedding(self) -> np.ndarray:
        if self.embedding is None and self.text_state:
            emb = self.client.embed(self.text_state)
            self.embedding = np.array(emb, dtype=np.float64)
        return self.embedding
