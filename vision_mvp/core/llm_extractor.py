"""Phase 33 — LLM-driven claim extractor.

Phase 31/32 evaluated the typed-handoff substrate with *regex-based*
extractors whose recall and precision on causal events were 1.0 by
construction. That was the right ceiling experiment — it isolated the
*substrate* story from the *extractor* story. Real agent-team products
do not have perfect extractors. Production roles call an LLM on their
role-local evidence, ask it to summarise / classify, and forward the
result as a typed claim. Phase 33 closes the gap between the Phase-32
synthetic noise model and how a real agent-team system behaves by
shipping an LLM-driven extractor path on top of the unchanged
``role_handoff`` substrate.

What this module provides
-------------------------

* ``LLMExtractor`` — a drop-in replacement for any Phase-31/32
  ``extract_claims_for_role`` callable. It calls an LLM on the raw
  role-local evidence and parses the reply into typed
  ``(claim_kind, payload, event_ids)`` tuples. The contract is
  identical to the regex extractor's contract, so the rest of the
  substrate (router / inbox / decoder / failure-attribution) is
  unchanged.

* ``LLMExtractorConfig`` — knobs for the prompt template (include /
  exclude role description, include / exclude known-kinds list, few-
  shot block), LLM decoder parameters (temperature, max_tokens),
  and the tag-set restriction (filter-on-output the extracted kinds
  against ``known_kinds_by_role`` so the LLM cannot invent a kind).

* ``DeterministicCache`` — a small keyed in-process LRU cache around
  the LLM call, keyed on the (model, role, scenario_id, event-body
  digest) tuple. Enables the calibration benchmark to re-run
  deterministically with zero extra LLM calls on the second pass.

* ``DeterministicMockExtractorLLM`` — a tiny deterministic "LLM"
  used by tests. Shares the same ``Callable[[str], str]`` shape as
  ``LLMClient.generate``. Given a prompt, it extracts claim-kind
  hints from the role description (no real model call).

Extractor contract (shared with regex extractors)
-------------------------------------------------

An extractor of the shape

    extractor(role, events, scenario) -> list[(claim_kind, payload, evids)]

MUST satisfy:

    * **Emits** — each tuple ``(kind, payload, evids)`` satisfies
      ``kind in known_kinds_by_role[role]`` and ``len(evids) >= 1``.
    * **Dropped claim** — a gold causal emission (a ``(role, kind,
      evids)`` triple from ``scenario.causal_chain``) that is *not*
      produced by the extractor. Equivalently, a *recall* failure on
      the gold-causal set.
    * **Mislabeled claim** — an emission whose evids overlap a gold
      causal emission's evids but whose ``kind`` does not match the
      gold kind. Equivalently, a *type-confusion* failure on an
      otherwise-correct witness.
    * **Spurious claim** — an emission whose evids do NOT overlap
      any gold causal emission's evids. Equivalently, a *precision*
      failure (the extractor produced a claim from a distractor
      event, or invented a claim entirely).

These three categories are the same taxonomy the Phase-32 noise
wrapper parameterises (``drop_prob``, ``mislabel_prob``,
``spurious_prob``), which is why the Phase-32 sweep's curves are
directly comparable to an LLM extractor's empirical noise profile
(Phase 33 Part B).

Scope discipline (what this module does NOT do)
----------------------------------------------

  * It does NOT modify ``role_handoff``. The substrate primitive is
    unchanged byte-for-byte from Phase 31/32; the LLM extractor
    produces the same extractor-tuple shape the regex extractors do.
  * It does NOT stream tokens or maintain role-local state across
    events. One LLM call per (role, scenario) boundary; stateless
    otherwise. Role-local memory is future work.
  * It does NOT try to harden the LLM's output against adversarial
    manipulation: the extractor trusts the role-local event stream,
    which for the Phase-31/32/33 benchmarks comes from a deterministic
    generator.
  * It does NOT include retrieval, re-ranking, or a retrieval-
    augmented decoder. The extractor reads the delivered role-local
    events and emits typed claims; anything richer is a different
    layer.

Theoretical anchor: RESULTS_PHASE33.md § B.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


# =============================================================================
# Types
# =============================================================================


ClaimTuple = tuple[str, str, tuple[int, ...]]
LLMCall = Callable[[str], str]
Extractor = Callable[[str, Sequence[Any], Any], list[ClaimTuple]]


# =============================================================================
# Prompt construction
# =============================================================================


_DEFAULT_ROLE_DESCRIPTIONS: dict[str, str] = {
    # Incident triage
    "monitor": ("You are the MONITOR role in an incident-response team. "
                "You read service-level metric samples and report "
                "error-rate and latency spikes."),
    "db_admin": ("You are the DB_ADMIN role. You read SQL statistics "
                 "and log lines and report slow queries, connection-"
                 "pool exhaustion, and deadlock events."),
    "sysadmin": ("You are the SYSADMIN role. You read OS events and "
                 "system logs and report disk-fill conditions, cron "
                 "overruns, and OOM kills."),
    "network": ("You are the NETWORK role. You read firewall hits, "
                "network flows, and DNS queries and report TLS "
                "expirations, DNS misroutes, and firewall block surges."),
    # Compliance review
    "legal": ("You are the LEGAL role. You read contract clauses and "
              "flag missing liability caps, unfavourable auto-renewal "
              "terms, and restrictive termination clauses."),
    "security": ("You are the SECURITY role. You read security-"
                 "questionnaire answers and flag missing at-rest "
                 "encryption, missing SSO, stale pentests, and "
                 "inadequate incident SLAs."),
    "privacy": ("You are the PRIVACY role. You read data-processing "
                "inventories and flag missing DPAs, unauthorised "
                "cross-border transfers, uncapped retention, and "
                "undisclosed PII categories."),
    "finance": ("You are the FINANCE role. You read line items and "
                "flag spend above the authority cap and aggressive "
                "payment terms."),
    # Security escalation (Phase 33)
    "soc_analyst": ("You are the SOC_ANALYST role. You read SIEM alerts "
                    "and signature hits and report authentication "
                    "spikes, malware detections, and suspicious "
                    "lateral movement."),
    "ir_engineer": ("You are the IR_ENGINEER role. You read endpoint "
                    "forensics — process trees, file hashes, persistence "
                    "artifacts — and report host compromises, "
                    "persistence, and data staging."),
    "threat_intel": ("You are the THREAT_INTEL role. You read "
                     "indicator-of-compromise matches and report known-"
                     "bad IP hits, malicious domains, and attributed "
                     "threat-actor TTPs."),
    "data_steward": ("You are the DATA_STEWARD role. You read data-"
                     "classification inventories and flag regulated "
                     "data exposure (PII, PCI, PHI), notification-"
                     "clock triggers, and cross-tenant leakage."),
}


def default_role_description(role: str) -> str:
    return _DEFAULT_ROLE_DESCRIPTIONS.get(
        role,
        f"You are the {role.upper()} role in a multi-role agent team. "
        "Extract typed claims from the events you observe.")


@dataclass(frozen=True)
class LLMExtractorConfig:
    """Knobs for the LLM extractor.

    * ``temperature``       — decoding temperature (default 0.0 for
      deterministic extraction).
    * ``max_tokens``        — cap on the LLM's reply length per
      (role, scenario) call. 256 is enough for 5–10 claims.
    * ``include_event_ids`` — if True, the prompt labels each event
      with its numeric id so the LLM can cite them. Set False to
      stress-test the prompt format.
    * ``include_few_shot``  — if True, the prompt prepends a short
      few-shot block showing the expected output shape (one JSON
      record per claim). Helps small models; frontier models are
      insensitive.
    * ``filter_unknown_kinds`` — if True, parsed claim kinds outside
      ``known_kinds_by_role[role]`` are dropped. The substrate would
      drop them anyway via ``RoleSubscriptionTable`` with no consumer,
      but dropping at the extractor boundary keeps the noise
      attribution clean.
    * ``role_description_overrides`` — per-role override of the role
      blurb used in the prompt. Callers wiring a bespoke domain
      override this.
    """

    temperature: float = 0.0
    max_tokens: int = 256
    include_event_ids: bool = True
    include_few_shot: bool = True
    filter_unknown_kinds: bool = True
    role_description_overrides: dict[str, str] = field(default_factory=dict)


def build_extractor_prompt(role: str,
                            events: Sequence[Any],
                            known_kinds: Sequence[str],
                            cfg: LLMExtractorConfig,
                            ) -> str:
    """Construct the per-role extractor prompt.

    Input events must expose two attributes: ``event_id`` (or
    ``doc_id``) and ``body``. The prompt is deliberately narrow:
    role blurb, known claim-kinds list, the role-local events, the
    required output format (one JSON line per claim). A few-shot
    block is prepended when ``cfg.include_few_shot`` is True.
    """
    desc = (cfg.role_description_overrides.get(role)
            or default_role_description(role))
    kinds_str = ", ".join(known_kinds)
    lines: list[str] = []
    lines.append(desc)
    lines.append(
        "Read the EVENTS below and emit one JSON object per typed "
        "claim. Do not emit any claim whose kind is not in the "
        "allowed list. Each claim must be one JSON object per line.")
    lines.append(f"ALLOWED_CLAIM_KINDS: [{kinds_str}]")
    lines.append(
        "OUTPUT FORMAT (one JSON object per line, no surrounding text):"
        "\n"
        '  {"kind": "<ALLOWED_CLAIM_KIND>", '
        '"payload": "<short witness string>", '
        '"event_ids": [<int list>]}')
    if cfg.include_few_shot:
        lines.append("")
        lines.append("EXAMPLE (for a different role, illustrative):")
        lines.append('  {"kind": "DISK_FILL_CRITICAL", '
                      '"payload": "/var/log used=99% fs=/", '
                      '"event_ids": [42]}')
    lines.append("")
    lines.append("EVENTS:")
    for ev in events:
        eid = getattr(ev, "event_id", None)
        if eid is None:
            eid = getattr(ev, "doc_id", None)
        body = getattr(ev, "body", "")
        if cfg.include_event_ids and eid is not None:
            lines.append(f"- id={eid}  {body}")
        else:
            lines.append(f"- {body}")
    lines.append("")
    lines.append("CLAIMS:")
    return "\n".join(lines)


# =============================================================================
# LLM output parsing
# =============================================================================


_JSON_CLAIM_RE = re.compile(
    r'\{\s*"kind"\s*:\s*"([A-Z_][A-Z_0-9]*)"\s*,\s*'
    r'"payload"\s*:\s*"([^"\n]*)"\s*,\s*'
    r'"event_ids"\s*:\s*\[([^\]]*)\]\s*\}')


def parse_llm_claims(text: str) -> list[ClaimTuple]:
    """Parse an LLM reply into ``(kind, payload, evids)`` tuples.

    Robust to leading / trailing prose, whitespace, and multiple
    objects per line. Uses a regex over the JSON-like shape rather
    than ``json.loads`` because small models often emit slightly
    malformed JSON — we recover as many well-formed claims as we can
    and silently drop the rest (those become ``dropped`` claims in
    the Phase-33 calibration taxonomy).

    The regex anchors on the three required keys (``kind``,
    ``payload``, ``event_ids``). Any extra keys are ignored.
    """
    out: list[ClaimTuple] = []
    for m in _JSON_CLAIM_RE.finditer(text):
        kind = m.group(1).strip()
        payload = m.group(2).strip()
        raw_ids = m.group(3)
        evids: list[int] = []
        for tok in re.findall(r"-?\d+", raw_ids):
            try:
                evids.append(int(tok))
            except ValueError:
                continue
        out.append((kind, payload, tuple(evids)))
    return out


# =============================================================================
# Deterministic cache — keyed on (role, scenario_id, event-digest)
# =============================================================================


def _events_digest(events: Sequence[Any]) -> str:
    payload = []
    for ev in events:
        eid = getattr(ev, "event_id", None)
        if eid is None:
            eid = getattr(ev, "doc_id", None)
        payload.append({"id": eid, "body": getattr(ev, "body", "")})
    blob = json.dumps(payload, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class DeterministicCache:
    """In-process cache keyed on (model, role, scenario_id, events-digest).

    Keeps the LLM extractor calibration cheap on second / third pass:
    Phase 33 Part B needs to compare the LLM extractor's output against
    the regex extractor's output on the same events, and needs the
    comparison to be repeatable. The cache stores the raw LLM reply,
    so the parser can be upgraded independently of the call.

    No eviction by default (the benchmark uses O(few hundred) unique
    keys). Pass ``max_entries`` to cap.
    """

    model: str = ""
    max_entries: int | None = None
    _store: dict[str, str] = field(default_factory=dict)

    def key(self, role: str, scenario_id: str,
            events: Sequence[Any]) -> str:
        return f"{self.model}|{role}|{scenario_id}|{_events_digest(events)}"

    def get(self, role: str, scenario_id: str,
            events: Sequence[Any]) -> str | None:
        return self._store.get(self.key(role, scenario_id, events))

    def put(self, role: str, scenario_id: str,
            events: Sequence[Any], reply: str) -> None:
        k = self.key(role, scenario_id, events)
        if self.max_entries is not None and \
                len(self._store) >= self.max_entries and \
                k not in self._store:
            self._store.pop(next(iter(self._store)))
        self._store[k] = reply

    def __len__(self) -> int:
        return len(self._store)


# =============================================================================
# LLMExtractor
# =============================================================================


@dataclass
class LLMExtractorStats:
    """Per-extractor call / token counters, exposed for the Phase-33
    calibration benchmark to report alongside accuracy numbers.
    """

    n_calls: int = 0
    n_cache_hits: int = 0
    n_parsed_claims: int = 0
    n_dropped_unknown_kind: int = 0
    total_prompt_chars: int = 0
    total_reply_chars: int = 0

    def as_dict(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "n_cache_hits": self.n_cache_hits,
            "n_parsed_claims": self.n_parsed_claims,
            "n_dropped_unknown_kind": self.n_dropped_unknown_kind,
            "total_prompt_chars": self.total_prompt_chars,
            "total_reply_chars": self.total_reply_chars,
        }


@dataclass
class LLMExtractor:
    """LLM-driven extractor for one team's role set.

    * ``llm_call``            — a ``Callable[[str], str]`` that takes a
      prompt and returns a reply. In tests, use
      ``DeterministicMockExtractorLLM``; in production, pass a closure
      around ``LLMClient.generate``.
    * ``known_kinds_by_role`` — per-role allowed claim-kind sets. The
      extractor refuses to emit any kind outside this set (so the
      substrate's type-safety invariants are preserved even when the
      LLM hallucinates).
    * ``config``              — ``LLMExtractorConfig`` knobs.
    * ``cache``               — optional ``DeterministicCache``. If
      provided, reused across calls.
    * ``stats``               — ``LLMExtractorStats`` counters.

    The extractor returns a list of ``(claim_kind, payload,
    event_ids)`` tuples — identical to what the regex extractors
    return, so it plugs into ``run_handoff_protocol(extractor=...)``
    without any substrate changes.
    """

    llm_call: LLMCall
    known_kinds_by_role: Mapping[str, Sequence[str]]
    config: LLMExtractorConfig = field(default_factory=LLMExtractorConfig)
    cache: DeterministicCache | None = None
    stats: LLMExtractorStats = field(default_factory=LLMExtractorStats)

    def __call__(self, role: str, events: Sequence[Any],
                 scenario: Any) -> list[ClaimTuple]:
        known = tuple(self.known_kinds_by_role.get(role, ()))
        if not events or not known:
            return []
        sid = str(getattr(scenario, "scenario_id", "default"))
        prompt = build_extractor_prompt(role, events, known, self.config)
        self.stats.n_calls += 1
        self.stats.total_prompt_chars += len(prompt)
        reply: str | None = None
        if self.cache is not None:
            reply = self.cache.get(role, sid, events)
            if reply is not None:
                self.stats.n_cache_hits += 1
        if reply is None:
            reply = self.llm_call(prompt)
            if self.cache is not None:
                self.cache.put(role, sid, events, reply)
        self.stats.total_reply_chars += len(reply)
        claims = parse_llm_claims(reply)
        allowed = set(known)
        kept: list[ClaimTuple] = []
        event_ids = set()
        for ev in events:
            eid = getattr(ev, "event_id", None)
            if eid is None:
                eid = getattr(ev, "doc_id", None)
            if eid is not None:
                event_ids.add(eid)
        for (kind, payload, evids) in claims:
            if self.config.filter_unknown_kinds and kind not in allowed:
                self.stats.n_dropped_unknown_kind += 1
                continue
            valid_evids = tuple(e for e in evids if e in event_ids)
            if not valid_evids:
                # If the LLM cited no valid event id, attach the first
                # event id so downstream attribution still has an anchor.
                if event_ids:
                    valid_evids = (next(iter(sorted(event_ids))),)
                else:
                    continue
            kept.append((kind, payload, valid_evids))
        self.stats.n_parsed_claims += len(kept)
        return kept


# =============================================================================
# DeterministicMockExtractorLLM — used by unit tests
# =============================================================================


@dataclass
class DeterministicMockExtractorLLM:
    """A tiny deterministic "LLM" for tests.

    It reads the event bodies from the prompt, emits one JSON claim
    per event whose body matches a role-specific keyword map (similar
    to the regex extractor), and misses or over-emits according to the
    noise parameters. This is not meant to be a realistic LLM — it is
    a stand-in that lets the Phase-33 test suite run without network
    IO while still exercising the ``LLMExtractor`` plumbing.

    For a realistic LLM, pass a closure around ``LLMClient.generate``
    to ``LLMExtractor`` directly.

    Parameters:
      * ``keyword_to_kind`` — map from keyword (lowercased substring)
        to claim kind. Used to pick a kind per event body.
      * ``drop_prob``       — per-call fraction of correct claims to
        drop (recall noise). Applied deterministically based on a
        hash of the prompt.
      * ``spurious_body``   — if set, emits one additional claim with
        this payload and a fresh (invented) kind. Used by tests.
    """

    keyword_to_kind: dict[str, str] = field(default_factory=dict)
    drop_prob: float = 0.0
    spurious_body: str | None = None
    spurious_kind: str | None = None
    _calls: int = 0

    def __call__(self, prompt: str) -> str:
        self._calls += 1
        body_re = re.compile(r"^- (?:id=(\-?\d+)\s+)?(.+)$", re.MULTILINE)
        claims: list[str] = []
        seen_events = False
        for m in body_re.finditer(prompt):
            ev_id_s = m.group(1)
            if ev_id_s is None:
                continue
            seen_events = True
            ev_id = int(ev_id_s)
            body = m.group(2).strip().lower()
            for kw, kind in self.keyword_to_kind.items():
                if kw in body:
                    # Deterministic "drop" based on hash.
                    if (hash((self._calls, ev_id, kind)) & 0xFFFF) \
                            / 0xFFFF < self.drop_prob:
                        continue
                    claims.append(
                        '{"kind": "' + kind + '", '
                        '"payload": "' + body.replace('"', "'")[:80] + '", '
                        '"event_ids": [' + str(ev_id) + ']}')
                    break
        if self.spurious_body and self.spurious_kind and seen_events:
            claims.append(
                '{"kind": "' + self.spurious_kind + '", '
                '"payload": "' + self.spurious_body + '", '
                '"event_ids": [-999]}')
        return "\n".join(claims)
