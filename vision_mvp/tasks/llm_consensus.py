"""LLM coordination task — Collaborative Classification with Private Perspectives.

The team is asked a factual / classification question. Each agent gets a
different persona that biases their initial take. The team must converge
on the correct answer through a small number of communication rounds.

Example instances (question, correct_label, n_classes):
  Q: "Is a whale a fish or a mammal? Answer with one word: fish / mammal."
     → mammal
  Q: "Which number is larger: 11 or 9? Answer with just the number."
     → 11
  Q: "Is 'The movie was a masterpiece.' positive, negative, or neutral?"
     → positive

Personas (one per agent) are short biasing prompts like:
  "You tend to trust scientific consensus."
  "You are skeptical of received wisdom and prefer evidence."
  "You are a literary critic who values nuance."

Ground truth is fixed per question. Success metric:
  - Accuracy: fraction of agents' final answers that equal ground truth
  - Convergence: std of answer distribution at end; 0 = full agreement

The task generator knows ground truth for scoring; agents never see it.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class LLMQuestion:
    question: str
    ground_truth: str
    valid_answers: list[str]   # lowercased, the label alphabet

    def normalize(self, raw: str) -> str:
        raw = raw.lower().strip().strip(".").strip()
        # Find any valid answer in the output
        for v in self.valid_answers:
            if v in raw:
                return v
        return raw.split()[0] if raw else ""


DEFAULT_QUESTIONS = [
    LLMQuestion(
        question=("Is a whale a fish or a mammal? "
                  "Answer with one word: fish or mammal."),
        ground_truth="mammal",
        valid_answers=["mammal", "fish"],
    ),
    LLMQuestion(
        question=("Is 11 greater than 9? Answer yes or no."),
        ground_truth="yes",
        valid_answers=["yes", "no"],
    ),
    LLMQuestion(
        question=("Classify the sentiment of 'That movie was absolutely wonderful' "
                  "as positive, negative, or neutral. Answer with one word."),
        ground_truth="positive",
        valid_answers=["positive", "negative", "neutral"],
    ),
]


DEFAULT_PERSONAS = [
    "You trust established scientific consensus and take mainstream views seriously.",
    "You are a skeptical contrarian who questions common assumptions.",
    "You are a literal-minded engineer who wants exact, precise answers.",
    "You are a humanities scholar who appreciates nuance and context.",
    "You are cautious and like to double-check before committing.",
    "You like to be decisive and go with the first reasonable answer.",
    "You are a biology teacher who recalls what's taught in high school.",
    "You are a curious polymath interested in many fields.",
    "You are a practical person who wants the most useful answer.",
    "You tend to agree with the majority but weight expertise highly.",
    "You are detail-oriented and examine each word of a question.",
    "You value simplicity and prefer the most obvious interpretation.",
    "You are a careful reasoner who walks through edge cases.",
    "You trust your gut instincts on everyday questions.",
    "You are a linguist attentive to how questions are phrased.",
    "You are a data-driven analyst who prefers empirical answers.",
    "You are a philosopher who considers multiple perspectives.",
    "You are a journalist trained to verify claims.",
    "You are an educator who thinks about what is widely known.",
    "You are a scientist who weighs evidence carefully.",
]


def assign_personas(n_agents: int) -> list[str]:
    """Cycle through personas so even N > 20 works deterministically."""
    return [DEFAULT_PERSONAS[i % len(DEFAULT_PERSONAS)] for i in range(n_agents)]
