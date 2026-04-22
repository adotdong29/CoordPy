"""Phase 36 Part B — LLM-driven thread reply path.

Phase 35's dynamic-coordination primitive runs producer-local
causality extraction through ``infer_causality_hypothesis`` — a
deterministic, scenario-matched oracle with precision and recall
1.00 by construction. That was the right ceiling: it isolated
the substrate's expressivity (thread fires, resolution closes,
answer flips) from the reflection capability of a real LLM.

Phase 36 Part B shifts the producer's reply to an LLM while
keeping the thread primitive *strictly typed and strictly
bounded*:

  * member set stays fixed;
  * reply budget per member stays at ``max_replies_per_member``;
  * the reply_kind vocabulary stays in the Phase-35 enum
    (``INDEPENDENT_ROOT`` / ``DOWNSTREAM_SYMPTOM`` /
    ``UNCERTAIN`` / ``AGREE`` / ``DISAGREE`` / ``DEFER_TO``);
  * the witness is still bounded by ``witness_token_cap``.

The LLM is only the *decider* — it sees a narrowly-framed prompt
with the candidate claims, its own role-local evidence, and the
allowed reply kinds, and returns one JSON line. The parser
enforces a well-typed reply — out-of-vocabulary or malformed
replies are rejected (and recorded). This is the explicit
discipline that distinguishes the Phase-36 substrate from an
unbounded group chat.

What this module provides
-------------------------

* ``LLMReplyConfig``              — prompt / parsing knobs.
* ``LLMThreadReplier``            — callable that produces a
  ``(reply_kind, witness)`` tuple given (scenario, role, kind,
  payload). Drops in as a ``CausalityExtractor`` for the
  existing Phase-35 runner when adapted via
  ``causality_extractor_from_replier``.
* ``LLMReplierStats``             — per-run counters (n_calls,
  malformed, out_of_vocab, cache_hits).
* ``DeterministicMockReplier``    — a tiny deterministic "LLM"
  used by unit tests; mirrors the oracle to within a drop rate.
* ``parse_llm_reply_json`` / ``build_thread_reply_prompt``
  — helpers.

Scope discipline (what this module does NOT do)
-----------------------------------------------

  * It does NOT change ``EscalationThread``, the resolution
    rule, or the subscription table.
  * It does NOT attempt to teach the LLM the full causality
    ontology of the Phase-31 incident domain. The prompt only
    asks the model to emit one of the allowed reply kinds.
  * It does NOT free the producer from the typed-handoff
    discipline. The output is filtered on parse: any unknown
    reply_kind is refused; the fallback is UNCERTAIN.
  * It does NOT model an adversarial LLM — just a noisy /
    imperfect one. Adversarial-reply scenarios belong to
    ``core/reply_noise``.

Theoretical anchor: RESULTS_PHASE36.md § B.3 (Theorem P36-3,
Conjecture C36-6).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from vision_mvp.core.dynamic_comm import (
    REPLY_AGREE, REPLY_DISAGREE, REPLY_DEFER_TO,
    REPLY_DOWNSTREAM_SYMPTOM, REPLY_INDEPENDENT_ROOT,
    REPLY_UNCERTAIN, ALL_REPLY_KINDS,
)


LLMCall = Callable[[str], str]


# Default causality vocabulary — the Phase-36 benchmark only uses
# these three. An LLM that returns any other class is handled as
# malformed (defaults to UNCERTAIN).
DEFAULT_REPLY_KINDS = (REPLY_INDEPENDENT_ROOT,
                        REPLY_DOWNSTREAM_SYMPTOM,
                        REPLY_UNCERTAIN)


@dataclass(frozen=True)
class LLMReplyConfig:
    """Knobs for the LLM-driven reply path.

    * ``temperature``       — decoding temperature. 0.0 for
      deterministic replies.
    * ``max_tokens``        — cap on reply length. 60 is enough
      for one JSON line.
    * ``witness_token_cap`` — mirror of the thread's
      ``witness_token_cap``. The parser clamps witness tokens to
      this limit so a chatty LLM cannot overrun the bound.
    * ``allowed_reply_kinds`` — tuple of allowed reply kinds.
      Defaults to the causality trio.
    * ``include_role_events`` — if True, the prompt includes the
      producer's own role-local events (truncated to
      ``max_events_in_prompt``). If False, only the candidate
      claims are passed — stress test for Phase-36's boundary
      between "reflection with evidence" and "reflection from
      header alone".
    * ``max_events_in_prompt`` — per-role event cap.
    * ``fallback_reply_kind`` — reply to emit on parse failure or
      out-of-vocab response. Defaults to ``REPLY_UNCERTAIN``.
    """

    temperature: float = 0.0
    max_tokens: int = 60
    witness_token_cap: int = 12
    allowed_reply_kinds: tuple[str, ...] = DEFAULT_REPLY_KINDS
    include_role_events: bool = False
    max_events_in_prompt: int = 6
    fallback_reply_kind: str = REPLY_UNCERTAIN


def build_thread_reply_prompt(role: str,
                                candidate_role: str,
                                candidate_kind: str,
                                candidate_payload: str,
                                other_candidates: Sequence[tuple[str, str, str]],
                                role_events: Sequence[object] | None,
                                cfg: LLMReplyConfig,
                                ) -> str:
    """Construct a narrow prompt asking the LLM to classify one
    candidate claim.

    The prompt is deliberately terse: role blurb, candidate claims,
    allowed reply kinds, required output shape. No chain-of-thought
    block, no free-form reasoning — one JSON line out.
    """
    kinds_str = ", ".join(cfg.allowed_reply_kinds)
    lines: list[str] = []
    lines.append(
        f"You are the {role.upper()} role in a multi-role incident-"
        "response team. A coordination thread has been opened to "
        "decide which of several candidate root-cause claims is "
        "an isolated cause vs a downstream symptom.")
    lines.append(
        f"YOUR CLAIM: [{candidate_role}/{candidate_kind}] "
        f"{candidate_payload}")
    if other_candidates:
        lines.append("OTHER CANDIDATE CLAIMS IN THREAD:")
        for (r, k, p) in other_candidates:
            lines.append(f"- [{r}/{k}] {p}")
    lines.append(
        "Based on YOUR role-local evidence, classify YOUR CLAIM.")
    lines.append(
        f"ALLOWED REPLY KINDS: [{kinds_str}]")
    lines.append(
        "REPLY FORMAT (one JSON line, no surrounding text):")
    lines.append(
        '  {"reply_kind": "<ALLOWED_REPLY_KIND>", '
        '"witness": "<short evidence string, '
        f"≤ {cfg.witness_token_cap} whitespace tokens>"
        '"}')
    if cfg.include_role_events and role_events:
        lines.append("")
        lines.append("YOUR ROLE-LOCAL EVENTS (may be empty):")
        cap = cfg.max_events_in_prompt
        for ev in list(role_events)[:cap]:
            body = getattr(ev, "body", str(ev))
            eid = getattr(ev, "event_id", None)
            if eid is None:
                eid = getattr(ev, "doc_id", None)
            if eid is not None:
                lines.append(f"- id={eid}  {body}")
            else:
                lines.append(f"- {body}")
        if len(role_events) > cap:
            lines.append("- ... (further events elided)")
    lines.append("")
    lines.append("REPLY:")
    return "\n".join(lines)


_REPLY_JSON_RE = re.compile(
    r'\{\s*"reply_kind"\s*:\s*"([A-Z_][A-Z_0-9]*)"\s*,\s*'
    r'"witness"\s*:\s*"([^"\n]*)"\s*\}')


def parse_llm_reply_json(text: str,
                          cfg: LLMReplyConfig,
                          ) -> tuple[str, str, bool]:
    """Parse the LLM's one-line JSON reply.

    Returns a tuple ``(reply_kind, witness, well_formed)``.

    Validation rules:
      * If no JSON object matches → ``(fallback_reply_kind, "",
        False)``.
      * If the reply_kind is not in ``cfg.allowed_reply_kinds`` →
        ``(fallback_reply_kind, witness, False)``.
      * Otherwise, clamp witness to ``witness_token_cap`` tokens
        and return ``(reply_kind, witness, True)``.
    """
    m = _REPLY_JSON_RE.search(text)
    if not m:
        return cfg.fallback_reply_kind, "", False
    kind = m.group(1).strip()
    witness = m.group(2).strip()
    tokens = witness.split()
    if len(tokens) > cfg.witness_token_cap:
        witness = " ".join(tokens[:cfg.witness_token_cap])
    if kind not in cfg.allowed_reply_kinds:
        return cfg.fallback_reply_kind, witness, False
    return kind, witness, True


# =============================================================================
# Stats + cache
# =============================================================================


@dataclass
class LLMReplierStats:
    """Per-run counters for the LLM replier.

    * ``n_calls``            — number of LLM generate calls.
    * ``n_cache_hits``       — number of (scenario, role, kind)
      prompts satisfied from cache.
    * ``n_well_formed``      — parsed, in-vocab replies.
    * ``n_malformed``        — parse failures (JSON missing).
    * ``n_out_of_vocab``     — parsed but reply_kind not in
      allowed set.
    * ``total_prompt_chars`` — approx token count / 4 proxy.
    * ``total_reply_chars``  — same for replies.
    """

    n_calls: int = 0
    n_cache_hits: int = 0
    n_well_formed: int = 0
    n_malformed: int = 0
    n_out_of_vocab: int = 0
    total_prompt_chars: int = 0
    total_reply_chars: int = 0

    def as_dict(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "n_cache_hits": self.n_cache_hits,
            "n_well_formed": self.n_well_formed,
            "n_malformed": self.n_malformed,
            "n_out_of_vocab": self.n_out_of_vocab,
            "total_prompt_chars": self.total_prompt_chars,
            "total_reply_chars": self.total_reply_chars,
        }


def _reply_cache_key(scenario_id: str, role: str, kind: str,
                     payload: str) -> str:
    blob = f"{scenario_id}|{role}|{kind}|{payload}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# =============================================================================
# LLMThreadReplier
# =============================================================================


@dataclass
class LLMThreadReplier:
    """LLM-driven thread reply path.

    ``__call__(scenario, role, kind, payload, other_candidates,
    role_events=None)`` returns ``(reply_kind, witness,
    well_formed_bool)``.

    Parameters:
      * ``llm_call``  — ``Callable[[str], str]``. In production,
        a closure over ``LLMClient.generate``. In tests, a
        ``DeterministicMockReplier``.
      * ``config``    — ``LLMReplyConfig``.
      * ``stats``     — ``LLMReplierStats`` counters.
      * ``cache``     — optional ``dict[str, str]`` for
        deterministic replay. Key: hash of (scenario_id, role,
        kind, payload). Value: raw LLM reply.
    """

    llm_call: LLMCall
    config: LLMReplyConfig = field(default_factory=LLMReplyConfig)
    stats: LLMReplierStats = field(default_factory=LLMReplierStats)
    cache: dict[str, str] | None = None

    def __call__(self,
                 scenario: object,
                 role: str,
                 kind: str,
                 payload: str,
                 other_candidates: Sequence[tuple[str, str, str]] = (),
                 role_events: Sequence[object] | None = None,
                 ) -> tuple[str, str, bool]:
        sid = str(getattr(scenario, "scenario_id", "default"))
        key = _reply_cache_key(sid, role, kind, payload)
        reply_text: str | None = None
        if self.cache is not None:
            reply_text = self.cache.get(key)
            if reply_text is not None:
                self.stats.n_cache_hits += 1
        if reply_text is None:
            prompt = build_thread_reply_prompt(
                role=role,
                candidate_role=role, candidate_kind=kind,
                candidate_payload=payload,
                other_candidates=other_candidates,
                role_events=role_events, cfg=self.config)
            self.stats.n_calls += 1
            self.stats.total_prompt_chars += len(prompt)
            reply_text = self.llm_call(prompt)
            if self.cache is not None:
                self.cache[key] = reply_text
        self.stats.total_reply_chars += len(reply_text)
        reply_kind, witness, well_formed = parse_llm_reply_json(
            reply_text, self.config)
        if well_formed:
            self.stats.n_well_formed += 1
        else:
            # Distinguish "no JSON" from "JSON but out of vocab".
            if _REPLY_JSON_RE.search(reply_text):
                self.stats.n_out_of_vocab += 1
            else:
                self.stats.n_malformed += 1
        return reply_kind, witness, well_formed


# =============================================================================
# Bridging adaptor: CausalityExtractor ← LLMThreadReplier
# =============================================================================


def causality_extractor_from_replier(
        replier: LLMThreadReplier,
        scenario_cache: dict[str, dict] | None = None,
        ) -> Callable[[object, str, str, str], str]:
    """Wrap an ``LLMThreadReplier`` as a ``CausalityExtractor``
    (shape ``(scenario, role, kind, payload) -> str``).

    The returned extractor ignores the thread's per-call
    ``other_candidates`` context — it is a flat per-(role, kind,
    payload) classifier. The Phase-35 runner uses this shape.
    Downstream Phase-36 callers who want the richer 'other
    candidates' context should call ``LLMThreadReplier``
    directly.

    The output string shape matches ``infer_causality_hypothesis``:
      * INDEPENDENT_ROOT      → "INDEPENDENT_ROOT"
      * DOWNSTREAM_SYMPTOM    → "DOWNSTREAM_SYMPTOM_OF:" + kind
      * UNCERTAIN / AGREE /
        DISAGREE / DEFER_TO   → "UNCERTAIN" (conservative)
      * malformed             → "UNCERTAIN"
    """

    def _extract(scenario: object, role: str,
                 kind: str, payload: str) -> str:
        reply_kind, _witness, well_formed = replier(
            scenario=scenario, role=role, kind=kind,
            payload=payload, other_candidates=(),
            role_events=None)
        if not well_formed:
            return "UNCERTAIN"
        if reply_kind == REPLY_INDEPENDENT_ROOT:
            return "INDEPENDENT_ROOT"
        if reply_kind == REPLY_DOWNSTREAM_SYMPTOM:
            return "DOWNSTREAM_SYMPTOM_OF:" + kind
        return "UNCERTAIN"

    return _extract


# =============================================================================
# Deterministic mock replier for unit tests
# =============================================================================


@dataclass
class DeterministicMockReplier:
    """A tiny deterministic "LLM" for the Phase-36 test suite.

    It reads the candidate-claim block from the prompt and emits
    one JSON line whose reply_kind depends on a per-(role, kind)
    map in ``kind_replies``. If no map entry exists it defaults
    to ``REPLY_UNCERTAIN``.

    Parameters:
      * ``kind_replies``        — dict[(role, claim_kind)] →
        reply_kind. The oracle's answer for this test case.
      * ``malformed_prob``      — if > 0, on a hash-determined
        subset of prompts emit a malformed JSON line. Used to
        exercise the parser's fallback path.
      * ``witness_template``    — format string for the witness
        payload (e.g. ``"evidence for {kind}"``).
    """

    kind_replies: dict[tuple[str, str], str] = field(default_factory=dict)
    malformed_prob: float = 0.0
    witness_template: str = "evidence for {kind}"
    _calls: int = 0

    def __call__(self, prompt: str) -> str:
        self._calls += 1
        m = re.search(
            r"YOUR CLAIM:\s*\[([\w_]+)/([\w_]+)\]\s*(.+)", prompt)
        if not m:
            return '{"reply_kind": "UNCERTAIN", "witness": ""}'
        role = m.group(1)
        kind = m.group(2)
        reply_kind = self.kind_replies.get(
            (role, kind), REPLY_UNCERTAIN)
        # Deterministic malformed emission based on a hash of
        # (role, kind, self._calls).
        if self.malformed_prob > 0:
            h = hash((role, kind, self._calls)) & 0xFFFF
            if h / 0xFFFF < self.malformed_prob:
                return "sorry I'm not sure what the right answer is"
        witness = self.witness_template.format(kind=kind)
        # Cap witness at 6 tokens to be safe under typical cap=12.
        witness = " ".join(witness.split()[:6])
        return ('{"reply_kind": "' + reply_kind + '", '
                '"witness": "' + witness + '"}')


__all__ = [
    "DEFAULT_REPLY_KINDS", "LLMReplyConfig", "LLMReplierStats",
    "LLMThreadReplier", "DeterministicMockReplier",
    "parse_llm_reply_json", "build_thread_reply_prompt",
    "causality_extractor_from_replier",
]
