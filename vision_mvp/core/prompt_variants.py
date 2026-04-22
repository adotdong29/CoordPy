"""Phase 38 Part C — prompt-engineering calibration study.

Phase 37 Theorem P37-1 measured that real local LLMs
(qwen2.5:0.5b and qwen2.5-coder:7b) emit well-formed JSON on
100 % of Phase-35 contested calls but are semantically wrong
on 90 %. The dominant failure is ``sem_root_as_symptom``:
oracle = INDEPENDENT_ROOT, replier = DOWNSTREAM_SYMPTOM. This
is a *semantic* phenomenon — not a parse failure, not an
out-of-vocab emission — and Conjecture C37-1 named it as
*task- and prompt-specific*, i.e. the same two models under
a different prompt should yield a different calibration
distribution.

This module ships the machinery for a disciplined prompt-
variation study:

  * Five named prompt variants, each a small surgical edit to
    the Phase-36 ``build_thread_reply_prompt`` default.
  * A ``PromptVariantConfig`` record carrying the variant's
    label and the builder callable, so every measurement is
    reproducible by name.
  * A ``build_thread_reply_prompt_variant`` function that
    takes the variant name and returns the variant's prompt.

The five variants are designed to exercise the hypothesis
that the Phase-37 "emit DOWNSTREAM_SYMPTOM by default" bias
is a property of the PROMPT, not of the MODELS, by varying
the prompt along four axes:

  1. ``default``          — the Phase-36 prompt. Baseline.
  2. ``contrastive``      — adds a one-line explicit
     contrast: "If your payload names NO upstream cause, the
     answer is INDEPENDENT_ROOT; if your payload names or
     references an upstream claim, it is DOWNSTREAM_SYMPTOM;
     else UNCERTAIN." The bias-reducing hypothesis is this
     removes the "I'd better be modest and say symptom" LLM
     default.
  3. ``few_shot``         — prepends two explicit examples:
     one INDEPENDENT_ROOT, one DOWNSTREAM_SYMPTOM, with
     concrete payload+answer pairs. Classical prompt-
     engineering pattern; tests whether anchoring examples
     shift the bias.
  4. ``rubric``           — prepends a three-step decision
     rubric: (i) identify the topic of the payload; (ii) check
     if the payload names an upstream claim kind; (iii) pick
     the matching class. The hypothesis: structured
     deliberation reduces default-to-DS bias.
  5. ``forced_order``     — enforces a specific output
     ordering: the model first emits a one-token diagnosis
     tag (root / symptom / unclear), then the JSON line.
     Tests whether forced sequencing before the JSON emission
     influences the chosen class.

Every variant is still bounded by:

  * ``witness_token_cap`` (same as Phase-36).
  * ``allowed_reply_kinds`` (Phase-36 IR/DS/UNCERTAIN).
  * ``fallback_reply_kind = UNCERTAIN`` on parse failure.

So a variant that *shifts the calibration distribution* does
so without enlarging the substrate's typed-reply contract.
This is the point: prompt engineering is the *model-side*
defense, composable with Phase-37 ensembles on the
*substrate-side*.

Scope discipline
----------------

  * This module does NOT change ``LLMReplyConfig`` or
    ``LLMThreadReplier``. Variants are built on top of the
    same replier by swapping the prompt builder.
  * Variants are deliberately *surgical edits* — one-line or
    few-line additions. We do not ship a total prompt rewrite;
    the point is to measure whether small targeted changes
    move the calibration.
  * The variants are *not* fitted to any single scenario; each
    is written against the generic Phase-35 bank structure.
    Any per-scenario movement reported is a population
    measurement.
  * This module does NOT evaluate closed-source models. The
    Phase-38 study uses a deterministic mock that bakes in a
    per-variant bias so the experiment is sub-second and
    reproducible without an Ollama dependency. The real-LLM
    sweep is enumerated in the experiment driver as the
    follow-up: rerun under ``--models qwen2.5:0.5b ...``.

Theoretical anchor: RESULTS_PHASE38.md § B.3 (Theorem P38-4,
Conjecture C38-3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from vision_mvp.core.llm_thread_replier import (
    LLMReplyConfig,
)


PromptBuilder = Callable[
    [str, str, str, str,
     Sequence[tuple[str, str, str]],
     Sequence[object] | None,
     LLMReplyConfig],
    str]


# =============================================================================
# Variant names
# =============================================================================


PROMPT_VARIANT_DEFAULT = "default"
PROMPT_VARIANT_CONTRASTIVE = "contrastive"
PROMPT_VARIANT_FEW_SHOT = "few_shot"
PROMPT_VARIANT_RUBRIC = "rubric"
PROMPT_VARIANT_FORCED_ORDER = "forced_order"


ALL_PROMPT_VARIANTS = (
    PROMPT_VARIANT_DEFAULT,
    PROMPT_VARIANT_CONTRASTIVE,
    PROMPT_VARIANT_FEW_SHOT,
    PROMPT_VARIANT_RUBRIC,
    PROMPT_VARIANT_FORCED_ORDER,
)


# =============================================================================
# Prompt builders
# =============================================================================


def _header(role: str) -> list[str]:
    return [
        f"You are the {role.upper()} role in a multi-role "
        "incident-response team. A coordination thread has "
        "been opened to decide which of several candidate "
        "root-cause claims is an isolated cause vs a "
        "downstream symptom."
    ]


def _candidate_block(candidate_role: str,
                       candidate_kind: str,
                       candidate_payload: str,
                       other_candidates: Sequence[
                           tuple[str, str, str]]) -> list[str]:
    lines = [(f"YOUR CLAIM: [{candidate_role}/{candidate_kind}] "
              f"{candidate_payload}")]
    if other_candidates:
        lines.append("OTHER CANDIDATE CLAIMS IN THREAD:")
        for (r, k, p) in other_candidates:
            lines.append(f"- [{r}/{k}] {p}")
    return lines


def _output_contract(cfg: LLMReplyConfig) -> list[str]:
    kinds_str = ", ".join(cfg.allowed_reply_kinds)
    return [
        f"ALLOWED REPLY KINDS: [{kinds_str}]",
        "REPLY FORMAT (one JSON line, no surrounding text):",
        '  {"reply_kind": "<ALLOWED_REPLY_KIND>", '
        '"witness": "<short evidence string, '
        f"≤ {cfg.witness_token_cap} whitespace tokens>"
        '"}',
    ]


def build_default(role: str, candidate_role: str,
                    candidate_kind: str,
                    candidate_payload: str,
                    other_candidates: Sequence[
                        tuple[str, str, str]] = (),
                    role_events: Sequence[object] | None = None,
                    cfg: LLMReplyConfig = LLMReplyConfig(),
                    ) -> str:
    """Phase-36 default prompt (identical shape to
    ``llm_thread_replier.build_thread_reply_prompt``).
    """
    lines = _header(role)
    lines.extend(_candidate_block(
        candidate_role, candidate_kind, candidate_payload,
        other_candidates))
    lines.append(
        "Based on YOUR role-local evidence, classify YOUR "
        "CLAIM.")
    lines.extend(_output_contract(cfg))
    if cfg.include_role_events and role_events:
        lines.append("")
        lines.append("YOUR ROLE-LOCAL EVENTS (may be empty):")
        cap = cfg.max_events_in_prompt
        for ev in list(role_events)[:cap]:
            body = getattr(ev, "body", str(ev))
            lines.append(f"- {body}")
    lines.append("")
    lines.append("REPLY:")
    return "\n".join(lines)


def build_contrastive(role: str, candidate_role: str,
                        candidate_kind: str,
                        candidate_payload: str,
                        other_candidates: Sequence[
                            tuple[str, str, str]] = (),
                        role_events: Sequence[object] | None = None,
                        cfg: LLMReplyConfig = LLMReplyConfig(),
                        ) -> str:
    """Default + an explicit contrastive decision sentence."""
    lines = _header(role)
    lines.extend(_candidate_block(
        candidate_role, candidate_kind, candidate_payload,
        other_candidates))
    lines.extend([
        "DECISION CONTRAST:",
        "  If YOUR CLAIM's payload names no upstream cause, "
        "reply INDEPENDENT_ROOT.",
        "  If YOUR CLAIM's payload references or is caused by "
        "an upstream claim visible here, reply "
        "DOWNSTREAM_SYMPTOM.",
        "  Only if your evidence is truly ambiguous, reply "
        "UNCERTAIN.",
    ])
    lines.extend(_output_contract(cfg))
    lines.append("")
    lines.append("REPLY:")
    return "\n".join(lines)


def build_few_shot(role: str, candidate_role: str,
                     candidate_kind: str,
                     candidate_payload: str,
                     other_candidates: Sequence[
                         tuple[str, str, str]] = (),
                     role_events: Sequence[object] | None = None,
                     cfg: LLMReplyConfig = LLMReplyConfig(),
                     ) -> str:
    """Default + two concrete examples (one IR, one DS)."""
    lines = _header(role)
    lines.extend([
        "EXAMPLES (illustrative; not about your current scenario):",
        "  Example 1:",
        "    CLAIM: [network/TLS_EXPIRED] tls service=api "
        "reason=expired cert_path=/etc/ssl",
        "    CORRECT REPLY: "
        '{"reply_kind": "INDEPENDENT_ROOT", '
        '"witness": "reason=expired cert"}',
        "  Example 2:",
        "    CLAIM: [monitor/ERROR_RATE_SPIKE] error_rate=0.30 "
        "service=api",
        "    CORRECT REPLY: "
        '{"reply_kind": "DOWNSTREAM_SYMPTOM", '
        '"witness": "error_rate=0.30 downstream"}',
    ])
    lines.extend(_candidate_block(
        candidate_role, candidate_kind, candidate_payload,
        other_candidates))
    lines.append(
        "Based on YOUR role-local evidence, classify YOUR "
        "CLAIM.")
    lines.extend(_output_contract(cfg))
    lines.append("")
    lines.append("REPLY:")
    return "\n".join(lines)


def build_rubric(role: str, candidate_role: str,
                   candidate_kind: str,
                   candidate_payload: str,
                   other_candidates: Sequence[
                       tuple[str, str, str]] = (),
                   role_events: Sequence[object] | None = None,
                   cfg: LLMReplyConfig = LLMReplyConfig(),
                   ) -> str:
    """Default + a three-step decision rubric."""
    lines = _header(role)
    lines.extend(_candidate_block(
        candidate_role, candidate_kind, candidate_payload,
        other_candidates))
    lines.extend([
        "DECISION RUBRIC (apply in order):",
        "  Step 1: Identify the topic of YOUR CLAIM's payload. "
        "Is it a PRIMARY fault (cert expiry, deadlock, "
        "dns misroute, disk fill, oom kill, cron overrun) or a "
        "DOWNSTREAM metric (error_rate, latency spike, uptime, "
        "slow query, pool exhaustion)?",
        "  Step 2: If the payload names a PRIMARY fault AND "
        "YOUR evidence does not reference an upstream cause, "
        "reply INDEPENDENT_ROOT.",
        "  Step 3: If the payload is a DOWNSTREAM metric or "
        "you can name an upstream cause from another role's "
        "candidate claim, reply DOWNSTREAM_SYMPTOM. If truly "
        "ambiguous, reply UNCERTAIN.",
    ])
    lines.extend(_output_contract(cfg))
    lines.append("")
    lines.append("REPLY:")
    return "\n".join(lines)


def build_forced_order(role: str, candidate_role: str,
                         candidate_kind: str,
                         candidate_payload: str,
                         other_candidates: Sequence[
                             tuple[str, str, str]] = (),
                         role_events: Sequence[object] | None = None,
                         cfg: LLMReplyConfig = LLMReplyConfig(),
                         ) -> str:
    """Default + forced two-step emission (tag, then JSON)."""
    lines = _header(role)
    lines.extend(_candidate_block(
        candidate_role, candidate_kind, candidate_payload,
        other_candidates))
    lines.extend([
        "OUTPUT ORDER (emit these two lines exactly, in "
        "order):",
        "  LINE 1: TAG=<root|symptom|unclear>",
        "  LINE 2: one JSON line of the reply, matching the "
        "vocabulary and witness format below.",
    ])
    lines.extend(_output_contract(cfg))
    lines.append("")
    lines.append("REPLY:")
    return "\n".join(lines)


# =============================================================================
# Registry
# =============================================================================


VARIANT_BUILDERS: dict[str, PromptBuilder] = {
    PROMPT_VARIANT_DEFAULT: build_default,
    PROMPT_VARIANT_CONTRASTIVE: build_contrastive,
    PROMPT_VARIANT_FEW_SHOT: build_few_shot,
    PROMPT_VARIANT_RUBRIC: build_rubric,
    PROMPT_VARIANT_FORCED_ORDER: build_forced_order,
}


def build_thread_reply_prompt_variant(variant: str,
                                        role: str,
                                        candidate_role: str,
                                        candidate_kind: str,
                                        candidate_payload: str,
                                        other_candidates: Sequence[
                                            tuple[str, str, str]] = (),
                                        role_events: Sequence[object]
                                            | None = None,
                                        cfg: LLMReplyConfig = LLMReplyConfig(),
                                        ) -> str:
    """Dispatch to the named variant's builder."""
    try:
        builder = VARIANT_BUILDERS[variant]
    except KeyError:
        raise ValueError(
            f"unknown prompt variant {variant!r}; "
            f"expected one of {ALL_PROMPT_VARIANTS}")
    return builder(role, candidate_role, candidate_kind,
                    candidate_payload, other_candidates,
                    role_events, cfg)


# =============================================================================
# Stats container for a per-variant sweep
# =============================================================================


@dataclass
class PromptVariantReport:
    """Per-variant aggregate counters.

    Holds the per-call calibration bucket histogram and the
    per-strategy pooled accuracy.
    """

    variant: str
    calibration_rates: dict = field(default_factory=dict)
    pooled_by_strategy: dict = field(default_factory=dict)
    n_prompt_chars_total: int = 0
    n_calls: int = 0

    def as_dict(self) -> dict:
        return {
            "variant": self.variant,
            "calibration_rates": dict(self.calibration_rates),
            "pooled_by_strategy": dict(self.pooled_by_strategy),
            "n_prompt_chars_total": self.n_prompt_chars_total,
            "n_calls": self.n_calls,
        }


__all__ = [
    "PROMPT_VARIANT_DEFAULT", "PROMPT_VARIANT_CONTRASTIVE",
    "PROMPT_VARIANT_FEW_SHOT", "PROMPT_VARIANT_RUBRIC",
    "PROMPT_VARIANT_FORCED_ORDER", "ALL_PROMPT_VARIANTS",
    "VARIANT_BUILDERS", "build_thread_reply_prompt_variant",
    "build_default", "build_contrastive", "build_few_shot",
    "build_rubric", "build_forced_order",
    "PromptVariantReport",
]
